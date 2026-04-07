// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha de Células Recorrentes Otimizada (LSTM) para inferência NAM.
//!
//! Este módulo implementa a abstração LSTM otimizada com as instruções de `core::arch::x86_64` (FMA).
//! Utiliza arrays contíguos baseados em _Const Generics_ para evitar saltos condicionais na compilação.
//! O processamento adota a Estrutura de Arrays (SoA).

#[cfg(target_arch = "x86_64")]
use crate::math::fastmath::{simd_sigmoid, simd_tanh};
#[cfg(target_arch = "x86_64")]
use crate::math::simd::dot_product_avx2;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Uma camada individual do modelo LSTM.
///
/// Parâmetros Constantes:
/// * `I` = Input Size
/// * `H` = Hidden Size
/// * `IH` = Input + Hidden Size
/// * `H4` = 4 * Hidden Size
pub struct LstmLayer<const I: usize, const H: usize, const IH: usize, const H4: usize> {
    /// Matriz 1D agregada contendo os pesos (Input + Hidden) em Structure of Arrays.
    pub input_hidden_weights: [[f32; IH]; H4],
    /// Bias lineares extraídos do modelo (tamanho `4 * Hidden`).
    pub bias: [f32; H4],
    /// Estado global contendo [Input | Hidden].
    pub state: [f32; IH],
    /// Estado da Célula Interna LSTM.
    pub cell_state: [f32; H],
    /// Ativações lineares antes das portas C e Tanh.
    pub gates: [f32; H4],
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> LstmLayer<I, H, IH, H4> {
    /// Instancia uma nova camada LSTM, zero-iniciada via pré-alocação SoA contínua.
    pub fn new() -> Self {
        Self {
            input_hidden_weights: [[0.0; IH]; H4],
            bias: [0.0; H4],
            state: [0.0; IH],
            cell_state: [0.0; H],
            gates: [0.0; H4],
        }
    }

    /// Retorna o fatiamento da memória do estado atual que engloba a porção `Hidden`.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[I..]
    }

    /// Processa uma amostra de entrada com o estado interno do LSTM.
    ///
    /// # Safety
    /// Requer suporte garantido a instruções AVX2 e FMA no hardware de destino x86_64,
    /// sob pena de `SIGILL` (Illegal Instruction).
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn process_sample(&mut self, input: &[f32]) {
        unsafe {
            self.state[..I].copy_from_slice(&input[..I]);

            for i in 0..H4 {
                let dot = dot_product_avx2(&self.input_hidden_weights[i], &self.state);
                self.gates[i] = dot + self.bias[i];
            }

            let f_offset = H;
            let g_offset = 2 * H;
            let o_offset = 3 * H;
            let h_offset = I; // para estado oculto

            let mut i = 0;

            while i + 8 <= H {
                let g_f = _mm256_loadu_ps(self.gates.as_ptr().add(i + f_offset));
                let g_i = _mm256_loadu_ps(self.gates.as_ptr().add(i));
                let g_g = _mm256_loadu_ps(self.gates.as_ptr().add(i + g_offset));
                let c_s = _mm256_loadu_ps(self.cell_state.as_ptr().add(i));

                let sig_f = simd_sigmoid(g_f);
                let sig_i = simd_sigmoid(g_i);
                let tanh_g = simd_tanh(g_g);

                let mul1 = _mm256_mul_ps(sig_f, c_s);
                let mul2 = _mm256_mul_ps(sig_i, tanh_g);
                let new_c_s = _mm256_add_ps(mul1, mul2);
                _mm256_storeu_ps(self.cell_state.as_mut_ptr().add(i), new_c_s);

                let g_o = _mm256_loadu_ps(self.gates.as_ptr().add(i + o_offset));
                let sig_o = simd_sigmoid(g_o);
                let tanh_cs = simd_tanh(new_c_s);
                let h_val = _mm256_mul_ps(sig_o, tanh_cs);
                _mm256_storeu_ps(self.state.as_mut_ptr().add(i + h_offset), h_val);

                i += 8;
            }

            if i < H {
                let tail_len = H - i;
                let mut temp_gf = [0.0; 8];
                let mut temp_gi = [0.0; 8];
                let mut temp_gg = [0.0; 8];
                let mut temp_go = [0.0; 8];
                let mut temp_cs = [0.0; 8];

                for j in 0..tail_len {
                    temp_gf[j] = self.gates[i + j + f_offset];
                    temp_gi[j] = self.gates[i + j];
                    temp_gg[j] = self.gates[i + j + g_offset];
                    temp_go[j] = self.gates[i + j + o_offset];
                    temp_cs[j] = self.cell_state[i + j];
                }

                let g_f = _mm256_loadu_ps(temp_gf.as_ptr());
                let g_i = _mm256_loadu_ps(temp_gi.as_ptr());
                let g_g = _mm256_loadu_ps(temp_gg.as_ptr());
                let c_s = _mm256_loadu_ps(temp_cs.as_ptr());

                let sig_f = simd_sigmoid(g_f);
                let sig_i = simd_sigmoid(g_i);
                let tanh_g = simd_tanh(g_g);

                let mul1 = _mm256_mul_ps(sig_f, c_s);
                let mul2 = _mm256_mul_ps(sig_i, tanh_g);
                let new_c_s = _mm256_add_ps(mul1, mul2);

                let g_o = _mm256_loadu_ps(temp_go.as_ptr());
                let sig_o = simd_sigmoid(g_o);
                let tanh_cs = simd_tanh(new_c_s);
                let h_val = _mm256_mul_ps(sig_o, tanh_cs);

                let mut out_cs = [0.0; 8];
                let mut out_h = [0.0; 8];
                _mm256_storeu_ps(out_cs.as_mut_ptr(), new_c_s);
                _mm256_storeu_ps(out_h.as_mut_ptr(), h_val);

                for j in 0..tail_len {
                    self.cell_state[i + j] = out_cs[j];
                    self.state[i + j + h_offset] = out_h[j];
                }
            }
        }
    }
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> Default
    for LstmLayer<I, H, IH, H4>
{
    fn default() -> Self {
        Self::new()
    }
}

/// Modelo LSTM com 1 camada recorrente (ex: NAM profile 1x8, 1x16).
pub struct LstmModel1<const H: usize, const H1_IH: usize, const H_H4: usize> {
    /// Camada única da malha.
    pub layer: LstmLayer<1, H, H1_IH, H_H4>,
    /// Pesos de extração direcional (Cabeça).
    pub head_weights: [f32; H],
    /// # Safety
    pub head_bias: f32,
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmModel1<H, H1_IH, H_H4> {
    /// Inicializa a arquitetura em repouso.
    pub fn new() -> Self {
        Self {
            layer: LstmLayer::new(),
            head_weights: [0.0; H],
            head_bias: 0.0,
        }
    }

    /// Executa inferência sobre um arranjo contíguo de amostras de aúdio em blocos,
    /// avaliando a memória recorrente local.
    ///
    /// # Safety
    /// Exige compatibilidade com `avx2` e `fma` ativados no processador x86_64 hospedeiro.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            for i in 0..input.len() {
                let sample = [input[i]];
                self.layer.process_sample(&sample);
                let hidden = self.layer.get_hidden_state();
                let dot = dot_product_avx2(&self.head_weights, hidden);
                output[i] = dot + self.head_bias;
            }
        }
    }
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> Default for LstmModel1<H, H1_IH, H_H4> {
    fn default() -> Self {
        Self::new()
    }
}

/// Modelo LSTM empilhado contendo 2 malhas interligadas (ex: NAM profile 2x8, 2x16).
pub struct LstmModel2<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> {
    /// Primeira camada adjunta da entrada mono canalizada.
    pub layer1: LstmLayer<1, H, H1_IH, H_H4>,
    /// Segunda camada da cadeia em profundidade.
    pub layer2: LstmLayer<H, H, H2_IH, H_H4>,
    /// Pesos densos para colapso de saída auditiva final.
    pub head_weights: [f32; H],
    /// Constante bias para modulação de gain-staging global da malha empilhada.
    pub head_bias: f32,
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize>
    LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Consolida a arquitetura completa do modelo de 2 camadas contiguamente alocada.
    pub fn new() -> Self {
        Self {
            layer1: LstmLayer::new(),
            layer2: LstmLayer::new(),
            head_weights: [0.0; H],
            head_bias: 0.0,
        }
    }

    /// Rotina paralela e de estado contínuo (`sample-by-sample`) executada sequencialmente num bloco.
    ///
    /// # Safety
    /// Exige compatibilidade com `avx2` e `fma` em sistema de processamento com arquitetura x86_64.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            for i in 0..input.len() {
                let sample = [input[i]];
                self.layer1.process_sample(&sample);
                let hidden1 = self.layer1.get_hidden_state();

                self.layer2.process_sample(hidden1);
                let hidden2 = self.layer2.get_hidden_state();

                let dot = dot_product_avx2(&self.head_weights, hidden2);
                output[i] = dot + self.head_bias;
            }
        }
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> Default
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_model1_allocation() {
        let model: LstmModel1<8, 9, 32> = LstmModel1::new();
        assert_eq!(model.layer.gates.len(), 32);
        assert_eq!(model.layer.state.len(), 9);
    }

    #[test]
    fn test_lstm_model2_allocation() {
        let model: LstmModel2<16, 17, 32, 64> = LstmModel2::new();
        assert_eq!(model.layer1.input_hidden_weights.len(), 64);
        assert_eq!(model.layer2.input_hidden_weights[0].len(), 32);
    }
}
