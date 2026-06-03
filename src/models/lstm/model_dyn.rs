// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic Recurrent Cell Mesh for LSTM Inference (Fallback).
//!
//! Enables loading models with arbitrary LSTM topologies
//! (e.g., more than 2 layers, custom hidden sizes) that lack
//! static SoA implementations via Const Generics.
//! To respect the RT Lock-Free path, all memory manipulation is
//! resolved with buffers passed directly to the iterative vector math pipeline of SIMMathConfig.

/// Dynamically managed LSMT layer at runtime
#[derive(Clone)]
pub struct LstmDynLayer {
    /// Weight `(Input + Hidden)` by `(4 * Hidden)` matrix, 1D linearized.
    pub input_hidden_weights: Vec<u16>,
    /// Linear array (H * 4) with the activation biases of each gate.
    pub bias: Vec<f32>,
    /// Local State tensor, merges the incoming sample and the previous frame's final moment. [I + H]
    pub state: Vec<f32>,
    /// Global state mirrored in BF16 for VNNI processing.
    pub state_bf16: Vec<u16>,
    /// Restricted block of mathematical recursion of the LSTM cell. [H]
    pub cell_state: Vec<f32>,
    /// Accumulation buffers for vector evaluation over the layer. [H * 4]
    pub gates: Vec<f32>,
    /// Auxiliary buffer of contiguous scope. [H]
    pub tanh_cs: Vec<f32>,
    /// Physical dimension of the initial vector `[1]` or previous contiguity in an Lstm2.
    pub input_size: usize,
    /// Latent mathematical dimension.
    pub hidden_size: usize,
}

impl LstmDynLayer {
    /// Unfolds the mesh over SimdMathConfig cycles
    ///
    /// # Safety
    /// Natively depends on matrices filled via strict allocator in the C++ Fallback parser (Loader CLI).
    pub unsafe fn process_sample<M: crate::math::common::SimdMath>(&mut self, input: &[f32]) {
        // ih = Input + Hidden: The total size of the vector entering the computation (X_t + H_t-1)
        let ih = self.input_size + self.hidden_size;
        // h = Hidden: The dimension of the internal state and the output of this layer
        let h = self.hidden_size;

        // 1. Update state with input sample (replace in the prefix)
        // The 'state' vector contains [Current_Input | Previous_Hidden].
        // Here, we inject the new audio at the beginning of the buffer for the next multiplication.
        self.state[..self.input_size].copy_from_slice(input);

        // OPTIMIZATION: "Prefetch" tells the CPU to bring data from RAM to L1 Cache
        // BEFORE the heavy computation loop starts. This reduces access latency (Memory Stall).
        unsafe {
            // T0 = Bring to all cache levels (L1, L2, L3)
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                self.state.as_ptr().cast::<i8>(),
            );
            // If the vector is large, prefetch the next "cache line" (64 bytes/16 floats)
            if ih > 16 {
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                    self.state.as_ptr().add(16).cast::<i8>(),
                );
            }
        }

        // 2. Linear Dot Products -> Fills GATES and adds BIAS via GEMV
        if M::IS_BF16 {
            // Sync BF16 state
            unsafe { M::f32_to_bf16(&self.state, &mut self.state_bf16) };

            unsafe {
                M::gemv_overwrite_bf16_4gate(
                    &self.state_bf16,
                    &self.input_hidden_weights,
                    &self.bias,
                    &mut self.gates,
                    h,
                    true,
                );
            }
        } else {
            unsafe {
                M::gemv_overwrite_4gate(
                    &self.state,
                    &self.input_hidden_weights,
                    &self.bias,
                    &mut self.gates,
                    h,
                    true,
                );
            }
        }

        // 3. Fused Kernel: Activations + State Update
        // Performs sigmoids (I, F, O) and tanh (G), updates cell_state and hidden_state (state[input_size..])
        // in a single fused step, reducing memory traffic from 3 passes to 1.
        unsafe {
            M::fused_lstm_gates_dyn(
                &mut self.gates,
                &mut self.cell_state,
                &mut self.state[self.input_size..],
                h,
            );
        }
    }

    /// Returns the visible part of the hidden state.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[self.input_size..]
    }

    /// Zeros the layer's internal states (hidden and cell).
    pub fn reset_states(&mut self) {
        self.state.fill(0.0);
        self.cell_state.fill(0.0);
        self.gates.fill(0.0);
        self.tanh_cs.fill(0.0);
    }
}

/// Final Dynamic LSTM Wrapper.
///
/// **Scientific Reference:** Hochreiter, S., & Schmidhuber, J. (1997). *"Long Short-Term Memory."*
pub struct LstmDynModel {
    /// Linearly stacked set of LSTM subroutines.
    /// In complex models, we can have 2, 3 or more layers where one feeds the other.
    pub layers: Vec<LstmDynLayer>,
    /// Linear prediction filter of the output (Head) — quantized.
    pub head_weights: Vec<u16>,
    /// Output head weights in full f32 precision (mixed-precision selective).
    pub head_weights_f32: Vec<f32>,
    /// Projected linear output accumulator.
    pub head_bias: f32,
    /// Whether to use f32 head weights instead of quantized.
    pub use_f32_head: bool,
}

impl LstmDynModel {
    /// Synchronous Audio Processing. Requires RT instantiation.
    /// This function processes an audio block, sample by sample.
    ///
    /// **For Scientists and Devs:** The `dispatch_simd!` macro inspects the CPU (e.g., has AVX-512? AVX2?)
    /// immediately (pre-resolved at startup) and dispatches to the `process_internal` function
    /// parameterized with the optimized SIMD math type (`M: SimdMath`).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Restricted function for neural processing where the compiler substitutes generic `M`
    /// with the appropriate intrinsic instructions (like AVX2 FMA) from the selected SIMDMath.
    unsafe fn process_internal<M: crate::math::common::SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) {
        let num_frames = input.len();

        for i in 0..num_frames {
            // Take the current audio sample.
            let current_sample = [input[i]];
            // 'hidden_out' starts as the input sample and gets updated
            // as we pass through each layer of the network.
            let mut hidden_out: &[f32] = &current_sample;

            // Layer loop: The output (hidden state) of layer N
            // serves as the input for layer N+1.
            for layer in self.layers.iter_mut() {
                unsafe { layer.process_sample::<M>(hidden_out) };
                // Get the hidden state that was just computed for the next layer
                hidden_out = layer.get_hidden_state();
            }

            // Output Layer (Head): Multiply the last layer's hidden state
            // by the 'head' weights and add the bias to get the final sample.
            let head_out = if self.use_f32_head {
                crate::math::common::scalar_ref::dot_product_f32_native(hidden_out, &self.head_weights_f32)
            } else {
                unsafe { M::dot_product(hidden_out, &self.head_weights) }
            } + self.head_bias;

            // Save the result to the output buffer
            output[i] = head_out;
        }
    }

    /// Synchronous Audio Processing (Fallback)
    /// Executes the LSTM internal state prewarm (Pre-warm).
    /// Dispatch ensures that the prewarm silence is calculated at the same speed
    /// as real processing.
    pub fn prewarm(&mut self, num_samples: usize) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal, num_samples);
        }
    }

    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    unsafe fn prewarm_internal<M: crate::math::common::SimdMath>(&mut self, num_samples: usize) {
        for layer in self.layers.iter_mut() {
            layer.state[..layer.input_size].fill(0.0);
        }

        const CHUNK: usize = 512;
        let mut zero_out = [0.0f32; CHUNK];
        let zero_in = [0.0f32; CHUNK];
        let mut rem = num_samples;

        while rem > 0 {
            let n = rem.min(CHUNK);
            unsafe { self.process_internal::<M>(&zero_in[..n], &mut zero_out[..n]) };
            rem -= n;
        }
    }

    /// Zeros the internal states of all LSTM layers.
    pub fn reset_states(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_states();
        }
    }
}
