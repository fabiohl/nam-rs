// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Modelo LSTM de 1 camada com despacho SIMD.

use super::layer::LstmLayer;

macro_rules! define_lstm1_process {
    (
        $fn_name:ident,
        $target_meta:meta,
        $layer_proc:ident,
        $dot_prod:path,
        $get_h:ident
    ) => {
        // NOTE: Injeta #[inline(always)] para AVX2 ou #[target_feature] para extensões.
        #[$target_meta]
        unsafe fn $fn_name(&mut self, input: &[f32], output: &mut [f32]) {
            unsafe {
                // Processamento Simples: Para modelos de 1 camada,
                // apenas passamos o áudio pela camada e projetamos o resultado final.
                for (i, &val) in input.iter().enumerate() {
                    self.layer.$layer_proc(&[val]);

                    // Transformamos a saída da rede neural no sinal de áudio final.
                    let h = self.layer.$get_h();
                    let dot = $dot_prod(h, &self.head_weights);
                    output[i] = dot + self.head_bias;
                }
            }
        }
    };
}

/// Modelo LSTM de 1 camada.
pub struct LstmModel1<const H: usize, const H1_IH: usize, const H_H4: usize> {
    /// Camada única do modelo.
    pub layer: LstmLayer<1, H, H1_IH, H_H4>,
    /// Pesos do cabeçalho de saída (Linear Projection).
    pub head_weights: [u16; H],
    /// Bias do cabeçalho de saída.
    pub head_bias: f32,
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmModel1<H, H1_IH, H_H4> {
    /// Cria um novo modelo LSTM de 1 camada.
    pub fn new() -> Self {
        Self {
            layer: LstmLayer::new(),
            head_weights: [0u16; H],
            head_bias: 0.0,
        }
    }
    define_lstm1_process!(
        process_avx2,
        inline(always),
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
        process_avx2vnni,
        target_feature(enable = "avxvnni"),
        process_sample_avx2vnni,
        crate::math::gemm::dot_product_avx2,
        get_hidden_state
    );

    define_lstm1_process!(
        process_avx512vnni,
        target_feature(enable = "avx512f,avx512vl,avx512vnni"),
        process_sample_avx512vnni,
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
    /// Processa um bloco de áudio através do modelo (SIMD dispatch).
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
    /// Processamento escalar (fallback).
    ///
    /// # Atenção
    /// Exclusivo para testes de paridade. Extremamente lento.
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
            == crate::math::common::InstructionSet::Avx512VnniBf16;
        for i in 0..input.len() {
            self.layer.process_sample_scalar(&[input[i]], is_bf16);
            let hidden = self.layer.get_hidden_state();
            let mut dot = 0.0;
            for (j, &h_val) in hidden.iter().enumerate().take(H) {
                let w = self.head_weights[j];
                let w_f32 = if is_bf16 {
                    f32::from_bits((w as u32) << 16)
                } else {
                    half::f16::from_bits(w).to_f32()
                };
                dot += h_val * w_f32;
            }
            output[i] = dot + self.head_bias;
        }
    }
    /// Reseta os estados internos.
    pub fn reset_states(&mut self) {
        self.layer.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> Default for LstmModel1<H, H1_IH, H_H4> {
    fn default() -> Self {
        Self::new()
    }
}
