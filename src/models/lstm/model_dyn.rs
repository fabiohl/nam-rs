// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic LSTM model with arbitrary number of layers and hidden size.
//!
//! This is the runtime fallback for topologies that do not match the 10 static
//! const-generic profiles (`LstmModel1` / `LstmModel2`).

use super::layer_dyn::LstmLayerDyn;
use crate::math::common::AlignedVec;
use crate::math::common::half::f16_bits_to_f32;

/// LSTM model with dynamically-allocated layers and head weights.
///
/// Composed of a `Vec<LstmLayerDyn>` chained sequentially:
///   - Layer 0 receives the mono audio input (input_size=1).
///   - Layers 1..N receive the previous layer's hidden state as input.
///   - Final output = `dot(hidden_last, head_weights) + head_bias`.
///
/// ## SIMD dispatch
/// Follows the same `dispatch_simd!( @self, ... )` pattern as the static
/// LSTM models. The outer dispatch selects the target-feature variant
/// (`process_avx2`, `process_avx512`, `process_avx512_vnni_bf16`), which
/// calls each layer's `#[target_feature]` kernel directly, avoiding
/// double dispatch per sample.
pub struct LstmModelDyn {
    /// Dynamically-sized LSTM layers.
    ///
    /// Layer 0 has `input_size=1` (mono audio), subsequent layers have
    /// `input_size=hidden_size` (stacked recurrent input).
    pub layers: Vec<LstmLayerDyn>,
    /// Output head (linear projection) weights — quantized.
    pub head_weights: AlignedVec<u16>,
    /// Output head weights in full f32 precision (mixed-precision selective).
    pub head_weights_f32: AlignedVec<f32>,
    /// Output head bias.
    pub head_bias: f32,
    /// Whether to use f32 head weights instead of quantized.
    pub use_f32_head: bool,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Expected sample rate (Hz) for prewarm calculation. Default: `48000.0`.
    pub expected_sample_rate: f64,
}

impl LstmModelDyn {
    /// Creates a new zero-initialized dynamic LSTM model.
    ///
    /// Allocates `num_layers` × `LstmLayerDyn` with the appropriate
    /// input_size (1 for layer 0, `hidden_size` for the rest) and
    /// pre-allocates `head_weights` / `head_weights_f32`.
    pub fn new(num_layers: usize, hidden_size: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let input_size = if i == 0 { 1 } else { hidden_size };
            layers.push(LstmLayerDyn::new(input_size, hidden_size));
        }

        Self {
            layers,
            head_weights: AlignedVec::new(hidden_size, 0u16),
            head_weights_f32: AlignedVec::new(hidden_size, 0.0f32),
            head_bias: 0.0,
            use_f32_head: false,
            prewarm_on_reset: true,
            expected_sample_rate: 48000.0,
        }
    }

    // =====================================================================
    // AVX2 specialization (x86-64-v3 baseline)
    // =====================================================================

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let n_layers = self.layers.len();
            debug_assert!(n_layers > 0, "LstmModelDyn requires at least one layer");
            let layers_ptr = self.layers.as_mut_ptr();

            if self.use_f32_head {
                for (s, &val) in input.iter().enumerate() {
                    (*layers_ptr).process_sample_avx2(&[val]);

                    for i in 1..n_layers {
                        let prev = &*layers_ptr.add(i - 1);
                        let hidden = &prev.state[prev.input_size..];
                        (*layers_ptr.add(i)).process_sample_avx2(hidden);
                    }

                    let last = &*layers_ptr.add(n_layers - 1);
                    let h = last.get_hidden_state();
                    let dot = crate::math::common::scalar_ref::dot_product_f32_native(
                        h,
                        &self.head_weights_f32,
                    );
                    output[s] = dot + self.head_bias;
                }
            } else {
                for (s, &val) in input.iter().enumerate() {
                    (*layers_ptr).process_sample_avx2(&[val]);

                    for i in 1..n_layers {
                        let prev = &*layers_ptr.add(i - 1);
                        let hidden = &prev.state[prev.input_size..];
                        (*layers_ptr.add(i)).process_sample_avx2(hidden);
                    }

                    let last = &*layers_ptr.add(n_layers - 1);
                    let h = last.get_hidden_state();
                    let dot = crate::math::gemm::dot_product_avx2(h, &self.head_weights);
                    output[s] = dot + self.head_bias;
                }
            }
        }
    }

    // =====================================================================
    // AVX-512 specialization (F+VL)
    // =====================================================================

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn process_avx512(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let n_layers = self.layers.len();
            debug_assert!(n_layers > 0, "LstmModelDyn requires at least one layer");
            let layers_ptr = self.layers.as_mut_ptr();

            if self.use_f32_head {
                for (s, &val) in input.iter().enumerate() {
                    (*layers_ptr).process_sample_avx512(&[val]);

                    for i in 1..n_layers {
                        let prev = &*layers_ptr.add(i - 1);
                        let hidden = &prev.state[prev.input_size..];
                        (*layers_ptr.add(i)).process_sample_avx512(hidden);
                    }

                    let last = &*layers_ptr.add(n_layers - 1);
                    let h = last.get_hidden_state();
                    let dot = crate::math::common::scalar_ref::dot_product_f32_native(
                        h,
                        &self.head_weights_f32,
                    );
                    output[s] = dot + self.head_bias;
                }
            } else {
                for (s, &val) in input.iter().enumerate() {
                    (*layers_ptr).process_sample_avx512(&[val]);

                    for i in 1..n_layers {
                        let prev = &*layers_ptr.add(i - 1);
                        let hidden = &prev.state[prev.input_size..];
                        (*layers_ptr.add(i)).process_sample_avx512(hidden);
                    }

                    let last = &*layers_ptr.add(n_layers - 1);
                    let h = last.get_hidden_state();
                    let dot = crate::math::gemm::dot_product_avx512(h, &self.head_weights);
                    output[s] = dot + self.head_bias;
                }
            }
        }
    }

    // =====================================================================
    // AVX-512 BF16 VNNI specialization
    // =====================================================================

    #[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
    unsafe fn process_avx512_vnni_bf16(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let n_layers = self.layers.len();
            debug_assert!(n_layers > 0, "LstmModelDyn requires at least one layer");
            let layers_ptr = self.layers.as_mut_ptr();

            if self.use_f32_head {
                for (s, &val) in input.iter().enumerate() {
                    (*layers_ptr).process_sample_avx512_vnni_bf16(&[val]);

                    for i in 1..n_layers {
                        let prev = &*layers_ptr.add(i - 1);
                        let hidden = &prev.state[prev.input_size..];
                        (*layers_ptr.add(i)).process_sample_avx512_vnni_bf16(hidden);
                    }

                    let last = &*layers_ptr.add(n_layers - 1);
                    let h_f32 = last.get_hidden_state();
                    let dot = crate::math::common::scalar_ref::dot_product_f32_native(
                        h_f32,
                        &self.head_weights_f32,
                    );
                    output[s] = dot + self.head_bias;
                }
            } else {
                for (s, &val) in input.iter().enumerate() {
                    (*layers_ptr).process_sample_avx512_vnni_bf16(&[val]);

                    for i in 1..n_layers {
                        let prev = &*layers_ptr.add(i - 1);
                        let hidden = &prev.state[prev.input_size..];
                        (*layers_ptr.add(i)).process_sample_avx512_vnni_bf16(hidden);
                    }

                    let last = &*layers_ptr.add(n_layers - 1);
                    let h = last.get_hidden_state_bf16();
                    let dot = crate::math::gemm::dot_product_bf16_avx512(h, &self.head_weights);
                    output[s] = dot + self.head_bias;
                }
            }
        }
    }

    // =====================================================================
    // Dispatch hub
    // =====================================================================

    /// Processes an audio block through the dynamic LSTM model.
    ///
    /// Dispatches to the highest available SIMD kernel based on the global
    /// configuration validated at startup.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::common::dispatch_simd!(
                @self,
                process_avx512_vnni_bf16,
                process_avx512,
                process_avx2,
                input,
                output
            );
        }
    }

    /// Scalar processing (fallback).
    ///
    /// Exclusively for parity tests. Extremely slow.
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
            == crate::math::common::InstructionSet::Avx512VnniBf16;
        let hidden_size = self.head_weights.len();
        let n_layers = self.layers.len();

        debug_assert!(n_layers > 0, "LstmModelDyn requires at least one layer");
        let layers_ptr = self.layers.as_mut_ptr();

        for s in 0..input.len() {
            unsafe {
                (*layers_ptr).process_sample_scalar(&[input[s]], is_bf16);

                for i in 1..n_layers {
                    let prev = &*layers_ptr.add(i - 1);
                    let hidden = &prev.state[prev.input_size..];
                    let hidden_copy: Vec<f32> = hidden.to_vec();
                    (*layers_ptr.add(i)).process_sample_scalar(&hidden_copy, is_bf16);
                }

                let last = &*layers_ptr.add(n_layers - 1);
                let hidden_last = last.get_hidden_state();
                let dot = if self.use_f32_head {
                    crate::math::common::scalar_ref::dot_product_f32_native(
                        hidden_last,
                        &self.head_weights_f32,
                    )
                } else {
                    let mut dot = 0.0;
                    for (j, &h_val) in hidden_last.iter().enumerate().take(hidden_size) {
                        let w = self.head_weights[j];
                        let w_f32 = if is_bf16 {
                            f32::from_bits((w as u32) << 16)
                        } else {
                            f16_bits_to_f32(w)
                        };
                        dot += h_val * w_f32;
                    }
                    dot
                };
                output[s] = dot + self.head_bias;
            }
        }
    }

    /// Resets all layers' internal states (hidden, cell, gates) to zero.
    pub fn reset_states(&mut self) {
        for layer in &mut self.layers {
            layer.reset_states();
        }
    }

    /// Resets only the input slots of all layers, preserving hidden and cell state.
    ///
    /// Used during prewarm to avoid discarding the initial `_xh` and `_c` states
    /// loaded from the NAM file.
    pub fn reset_input_slots(&mut self) {
        for layer in &mut self.layers {
            layer.reset_input_slot();
        }
    }
}
