// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Modelo LSTM de 2 camadas com pipeline e despacho SIMD.

use super::layer::LstmLayer;

macro_rules! define_lstm2_process_pipelined {
    (
        $fn_name:ident,
        $target_meta:meta,
        $layer_proc:ident,
        $dot_prod:path,
        $get_h2:ident
    ) => {
        // NOTE: Injeta #[inline(always)] para AVX2 ou #[target_feature] para extensões.
        #[$target_meta]
        unsafe fn $fn_name(&mut self, input: &[f32], output: &mut [f32]) {
            unsafe {
                let len = input.len();
                if len >= 1 {
                    // --- Técnica de Pipelining (Linha de Montagem) ---
                    // Para máxima velocidade, processamos as duas camadas do LSTM em paralelo.
                    // Enquanto a Camada 2 termina o som anterior, a Camada 1 já inicia o próximo.

                    // 1. Prólogo: Processamos o primeiríssimo frame apenas na Camada 1.
                    self.layer1.$layer_proc(&[input[0]]);
                    let mut prev_h1 = [0.0; H];
                    prev_h1.copy_from_slice(self.layer1.get_hidden_state());

                    // 2. Loop Principal: Onde o 'trabalho em equipe' acontece.
                    // Camada 1 e Camada 2 operam de forma independente sobre frames diferentes (i e i-1).
                    for i in 1..len {
                        let current_input = [input[i]];

                        // Estas duas chamadas rodam agora sem depender uma da outra neste ciclo!
                        self.layer1.$layer_proc(&current_input);
                        self.layer2.$layer_proc(&prev_h1);

                        // 3. Projeção de Saída: Convertemos a 'votação' dos neurônios da Camada 2
                        // em um valor real de áudio usando um produto escalar (Dot Product).
                        let h2 = self.layer2.$get_h2();
                        let dot = $dot_prod(h2, &self.head_weights);
                        output[i - 1] = dot + self.head_bias;

                        // Guardamos o resultado da Camada 1 para a Camada 2 usar na próxima volta.
                        prev_h1.copy_from_slice(self.layer1.get_hidden_state());
                    }

                    // 4. Epílogo: Processamos o último frame que sobrou na Camada 2.
                    self.layer2.$layer_proc(&prev_h1);
                    let h2 = self.layer2.$get_h2();
                    let dot = $dot_prod(h2, &self.head_weights);
                    output[len - 1] = dot + self.head_bias;
                }
            }
        }
    };
}

/// Modelo LSTM de 2 camadas.
pub struct LstmModel2<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> {
    /// Camada 1 do modelo.
    pub layer1: LstmLayer<1, H, H1_IH, H_H4>,
    /// Camada 2 do modelo.
    pub layer2: LstmLayer<H, H, H2_IH, H_H4>,
    /// Pesos do cabeçalho de saída.
    pub head_weights: [u16; H],
    /// Bias do cabeçalho de saída.
    pub head_bias: f32,
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize>
    LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Cria um novo modelo LSTM de 2 camadas.
    pub fn new() -> Self {
        Self {
            layer1: LstmLayer::new(),
            layer2: LstmLayer::new(),
            head_weights: [0u16; H],
            head_bias: 0.0,
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
            self.layer1.process_sample_scalar(&[input[i]], is_bf16);
            self.layer2
                .process_sample_scalar(self.layer1.get_hidden_state(), is_bf16);
            let hidden2 = self.layer2.get_hidden_state();
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
            output[i] = dot + self.head_bias;
        }
    }
    /// Reseta os estados internos.
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
