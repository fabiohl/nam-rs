// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 2-layer LSTM model with pipelining and SIMD dispatch.

use super::layer::LstmLayer;

macro_rules! define_lstm2_process_pipelined {
    (
        $fn_name:ident,
        $target_meta:meta,
        $layer_proc:ident,
        $dot_prod:path,
        $get_h2:ident
    ) => {
        // NOTE: Injects #[inline(always)] for AVX2 or #[target_feature] for extensions.
        #[$target_meta]
        unsafe fn $fn_name(&mut self, input: &[f32], output: &mut [f32]) {
            unsafe {
                let len = input.len();
                if len >= 1 {
                    // --- Pipelining Technique (Assembly Line) ---
                    // For maximum speed, we process the two LSTM layers in parallel.
                    // While Layer 2 finishes the previous sound, Layer 1 already begins the next.

                    // 1. Prologue: Process the very first frame only in Layer 1.
                    self.layer1.$layer_proc(&[input[0]]);
                    let mut prev_h1 = [0.0; H];
                    prev_h1.copy_from_slice(self.layer1.get_hidden_state());

                    // 2. Main Loop: Where the 'teamwork' happens.
                    // Layer 1 and Layer 2 operate independently on different frames (i and i-1).
                    for i in 1..len {
                        let current_input = [input[i]];

                        // These two calls now run without depending on each other in this cycle!
                        self.layer1.$layer_proc(&current_input);
                        self.layer2.$layer_proc(&prev_h1);

                        // 3. Output Projection: Convert the Layer 2 neuron 'vote'
                        // into a real audio value using a Dot Product.
                        let h2 = self.layer2.$get_h2();
                        let dot = if self.use_f32_head {
                            let h2_f32 = self.layer2.get_hidden_state();
                            crate::math::common::scalar_ref::dot_product_f32_native(h2_f32, &self.head_weights_f32)
                        } else {
                            $dot_prod(h2, &self.head_weights)
                        };
                        output[i - 1] = dot + self.head_bias;

                        // Save Layer 1's result for Layer 2 to use on the next iteration.
                        prev_h1.copy_from_slice(self.layer1.get_hidden_state());
                    }

                    // 4. Epilogue: Process the last remaining frame in Layer 2.
                    self.layer2.$layer_proc(&prev_h1);
                    let h2 = self.layer2.$get_h2();
                    let dot = if self.use_f32_head {
                        let h2_f32 = self.layer2.get_hidden_state();
                        crate::math::common::scalar_ref::dot_product_f32_native(h2_f32, &self.head_weights_f32)
                    } else {
                        $dot_prod(h2, &self.head_weights)
                    };
                    output[len - 1] = dot + self.head_bias;
                }
            }
        }
    };
}

/// 2-layer LSTM model.
pub struct LstmModel2<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> {
    /// Model layer 1.
    pub layer1: LstmLayer<1, H, H1_IH, H_H4>,
    /// Model layer 2.
    pub layer2: LstmLayer<H, H, H2_IH, H_H4>,
    /// Output head weights (quantized).
    pub head_weights: [u16; H],
    /// Output head weights in full f32 precision (mixed-precision selective).
    pub head_weights_f32: [f32; H],
    /// Output head bias.
    pub head_bias: f32,
    /// Whether to use f32 head weights instead of quantized.
    pub use_f32_head: bool,
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize>
    LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Creates a new 2-layer LSTM model.
    pub fn new() -> Self {
        Self {
            layer1: LstmLayer::new(),
            layer2: LstmLayer::new(),
            head_weights: [0u16; H],
            head_weights_f32: [0.0f32; H],
            head_bias: 0.0,
            use_f32_head: false,
        }
    }
    define_lstm2_process_pipelined!(
        process_avx2,
        inline(always),
        process_sample_avx2,
        crate::math::gemm::dot_product_avx2,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        process_sample_avx512,
        crate::math::gemm::dot_product_avx512,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx2vnni,
        target_feature(enable = "avxvnni"),
        process_sample_avx2vnni,
        crate::math::gemm::dot_product_avx2,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx512vnni,
        target_feature(enable = "avx512f,avx512vl,avx512vnni"),
        process_sample_avx512vnni,
        crate::math::gemm::dot_product_avx512,
        get_hidden_state
    );

    define_lstm2_process_pipelined!(
        process_avx512_vnni_bf16,
        target_feature(enable = "avx512f,avx512vl,avx512bf16"),
        process_sample_avx512_vnni_bf16,
        crate::math::gemm::dot_product_bf16_avx512,
        get_hidden_state_bf16
    );
    /// Processes an audio block through the model (SIMD dispatch).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::common::dispatch_simd!(
                self,
                process_avx512_vnni_bf16,
                process_avx512vnni,
                process_avx512,
                process_avx2vnni,
                process_avx2,
                input,
                output
            );
        }
    }
    /// Scalar processing (fallback).
    ///
    /// # Note
    /// Exclusively for parity tests. Extremely slow.
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
            == crate::math::common::InstructionSet::Avx512VnniBf16;
        for i in 0..input.len() {
            self.layer1.process_sample_scalar(&[input[i]], is_bf16);
            self.layer2
                .process_sample_scalar(self.layer1.get_hidden_state(), is_bf16);
            let hidden2 = self.layer2.get_hidden_state();
            let dot = if self.use_f32_head {
                crate::math::common::scalar_ref::dot_product_f32_native(hidden2, &self.head_weights_f32)
            } else {
                let mut dot = 0.0;
                for (j, &h_val) in hidden2.iter().enumerate().take(H) {
                    let w = self.head_weights[j];
                    let w_f32 = if is_bf16 {
                        f32::from_bits((w as u32) << 16)
                    } else {
                        half::f16::from_bits(w).to_f32()
                    };
                    dot += h_val * w_f32;
                }
                dot
            };
            output[i] = dot + self.head_bias;
        }
    }
    /// Resets the internal states.
    pub fn reset_states(&mut self) {
        self.layer1.reset_states();
        self.layer2.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> Default
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn default() -> Self {
        Self::new()
    }
}
