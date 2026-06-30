// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 1-layer LSTM model with SIMD dispatch.

use super::layer::LstmLayer;
use crate::math::common::half::f16_bits_to_f32;

macro_rules! define_lstm1_process {
    (
        $fn_name:ident,
        $target_meta:meta,
        $layer_proc:ident,
        $dot_prod:path,
        $get_h:ident
    ) => {
        #[$target_meta]
        unsafe fn $fn_name(&mut self, input: &[f32], output: &mut [f32]) {
            unsafe {
                if self.use_f32_head {
                    for (i, &val) in input.iter().enumerate() {
                        self.layer.$layer_proc(&[val]);
                        let h_f32 = self.layer.get_hidden_state();
                        output[i] = $crate::math::common::scalar_ref::dot_product_f32_native_kahan(
                            h_f32,
                            &self.head_weights_f32,
                        ) + self.head_bias;
                    }
                } else {
                    for (i, &val) in input.iter().enumerate() {
                        self.layer.$layer_proc(&[val]);
                        output[i] =
                            $dot_prod(self.layer.$get_h(), &self.head_weights) + self.head_bias;
                    }
                }
            }
        }
    };
}

/// 1-layer LSTM model.
pub struct LstmModel1<const H: usize, const H1_IH: usize, const H_H4: usize> {
    /// The model's single layer.
    pub layer: LstmLayer<1, H, H1_IH, H_H4>,
    /// Output head weights (Linear Projection) — quantized.
    pub head_weights: [u16; H],
    /// Output head weights in full f32 precision (mixed-precision selective).
    pub head_weights_f32: [f32; H],
    /// Output head bias.
    pub head_bias: f32,
    /// Whether to use f32 head weights instead of quantized.
    pub use_f32_head: bool,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Expected sample rate (Hz) for prewarm calculation. Default: `48000.0`.
    pub expected_sample_rate: f64,
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmModel1<H, H1_IH, H_H4> {
    /// Creates a new 1-layer LSTM model.
    pub fn new() -> Self {
        Self {
            layer: LstmLayer::new(),
            head_weights: [0u16; H],
            head_weights_f32: [0.0f32; H],
            head_bias: 0.0,
            use_f32_head: false,
            prewarm_on_reset: true,
            expected_sample_rate: 48000.0,
        }
    }
    define_lstm1_process!(
        process_avx2,
        target_feature(enable = "avx2,fma,f16c"),
        process_sample_avx2,
        crate::math::gemm::dot_product_avx2,
        get_hidden_state
    );

    define_lstm1_process!(
        process_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        process_sample_avx512,
        crate::math::gemm::dot_product_avx512,
        get_hidden_state
    );

    define_lstm1_process!(
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
    /// # Note
    /// Exclusively for parity tests. Extremely slow.
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
            == crate::math::common::InstructionSet::Avx512VnniBf16;
        for i in 0..input.len() {
            self.layer.process_sample_scalar(&[input[i]], is_bf16);
            let hidden = self.layer.get_hidden_state();
            let dot = if self.use_f32_head {
                crate::math::common::scalar_ref::dot_product_f32_native_kahan(
                    hidden,
                    &self.head_weights_f32,
                )
            } else {
                let mut dot = 0.0;
                for (j, &h_val) in hidden.iter().enumerate().take(H) {
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
            output[i] = dot + self.head_bias;
        }
    }
    /// Resets the internal states.
    pub fn reset_states(&mut self) {
        self.layer.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> Default for LstmModel1<H, H1_IH, H_H4> {
    fn default() -> Self {
        Self::new()
    }
}
