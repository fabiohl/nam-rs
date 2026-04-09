// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha de Células Recorrentes Otimizada (LSTM) para inferência NAM.
//!
//! Este módulo implementa a abstração LSTM otimizada com as instruções de `core::arch::x86_64` (FMA).
//! Utiliza arrays contíguos baseados em _Const Generics_ para evitar saltos condicionais na compilação.
//! O processamento adota a Estrutura de Arrays (SoA).

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

macro_rules! define_lstm_process {
    (
        $fn_name:ident,
        $target_meta:meta,
        $dot_product:path,
        $step:expr,
        $load:ident,
        $store:ident,
        $add:ident,
        $mul:ident,
        $tanh:path,
        $sigmoid:path
    ) => {
        #[cfg(target_arch = "x86_64")]
        #[$target_meta]
        /// Processa uma amostra de entrada com o estado interno da camada.
        ///
        /// # Safety
        /// Requer suporte das features garantidas pelo caller.
        pub unsafe fn $fn_name(&mut self, input: &[f32]) {
            unsafe {
                self.state[..I].copy_from_slice(&input[..I]);

                for i in 0..H4 {
                    let dot = $dot_product(&self.input_hidden_weights[i], &self.state);
                    self.gates[i] = dot + self.bias[i];
                }

                let f_offset = H;
                let g_offset = 2 * H;
                let o_offset = 3 * H;
                let h_offset = I; // para estado oculto

                let mut i = 0;

                while i + $step <= H {
                    let g_f = $load(self.gates.as_ptr().add(i + f_offset));
                    let g_i = $load(self.gates.as_ptr().add(i));
                    let g_g = $load(self.gates.as_ptr().add(i + g_offset));
                    let c_s = $load(self.cell_state.as_ptr().add(i));

                    let sig_f = $sigmoid(g_f);
                    let sig_i = $sigmoid(g_i);
                    let tanh_g = $tanh(g_g);

                    let mul1 = $mul(sig_f, c_s);
                    let mul2 = $mul(sig_i, tanh_g);
                    let new_c_s = $add(mul1, mul2);
                    $store(self.cell_state.as_mut_ptr().add(i), new_c_s);

                    let g_o = $load(self.gates.as_ptr().add(i + o_offset));
                    let sig_o = $sigmoid(g_o);
                    let tanh_cs = $tanh(new_c_s);
                    let h_val = $mul(sig_o, tanh_cs);
                    $store(self.state.as_mut_ptr().add(i + h_offset), h_val);

                    i += $step;
                }

                if i < H {
                    let tail_len = H - i;
                    let mut temp_gf = [0.0; $step];
                    let mut temp_gi = [0.0; $step];
                    let mut temp_gg = [0.0; $step];
                    let mut temp_go = [0.0; $step];
                    let mut temp_cs = [0.0; $step];

                    for j in 0..tail_len {
                        temp_gf[j] = self.gates[i + j + f_offset];
                        temp_gi[j] = self.gates[i + j];
                        temp_gg[j] = self.gates[i + j + g_offset];
                        temp_go[j] = self.gates[i + j + o_offset];
                        temp_cs[j] = self.cell_state[i + j];
                    }

                    let g_f = $load(temp_gf.as_ptr());
                    let g_i = $load(temp_gi.as_ptr());
                    let g_g = $load(temp_gg.as_ptr());
                    let c_s = $load(temp_cs.as_ptr());

                    let sig_f = $sigmoid(g_f);
                    let sig_i = $sigmoid(g_i);
                    let tanh_g = $tanh(g_g);

                    let mul1 = $mul(sig_f, c_s);
                    let mul2 = $mul(sig_i, tanh_g);
                    let new_c_s = $add(mul1, mul2);

                    let g_o = $load(temp_go.as_ptr());
                    let sig_o = $sigmoid(g_o);
                    let tanh_cs = $tanh(new_c_s);
                    let h_val = $mul(sig_o, tanh_cs);

                    let mut out_cs = [0.0; $step];
                    let mut out_h = [0.0; $step];
                    $store(out_cs.as_mut_ptr(), new_c_s);
                    $store(out_h.as_mut_ptr(), h_val);

                    for j in 0..tail_len {
                        self.cell_state[i + j] = out_cs[j];
                        self.state[i + j + h_offset] = out_h[j];
                    }
                }
            }
        }
    };
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

    define_lstm_process!(
        process_sample_avx2,
        target_feature(enable = "avx2,fma"),
        crate::math::simd::dot_product_avx2,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        crate::math::fastmath::simd_tanh,
        crate::math::fastmath::simd_sigmoid
    );

    define_lstm_process!(
        process_sample_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        crate::math::simd::dot_product_avx512,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::fastmath::simd_tanh_avx512,
        crate::math::fastmath::simd_sigmoid_avx512
    );
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
    /// Bias escalar da projeção de saída (cabeça linear).
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
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            for i in 0..input.len() {
                let sample = [input[i]];
                self.layer.process_sample_avx2(&sample);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx2(&self.head_weights, hidden);
                output[i] = dot + self.head_bias;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn process_avx512(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            for i in 0..input.len() {
                let sample = [input[i]];
                self.layer.process_sample_avx512(&sample);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx512(&self.head_weights, hidden);
                output[i] = dot + self.head_bias;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    /// Processa o array em tempo de execução validando features CPU.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            unsafe { self.process_avx512(input, output) }
        } else {
            unsafe { self.process_avx2(input, output) }
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
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            for i in 0..input.len() {
                let sample = [input[i]];
                self.layer1.process_sample_avx2(&sample);
                let hidden1 = self.layer1.get_hidden_state();

                self.layer2.process_sample_avx2(hidden1);
                let hidden2 = self.layer2.get_hidden_state();

                let dot = crate::math::simd::dot_product_avx2(&self.head_weights, hidden2);
                output[i] = dot + self.head_bias;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn process_avx512(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            for i in 0..input.len() {
                let sample = [input[i]];
                self.layer1.process_sample_avx512(&sample);
                let hidden1 = self.layer1.get_hidden_state();
                self.layer2.process_sample_avx512(hidden1);
                let hidden2 = self.layer2.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx512(&self.head_weights, hidden2);
                output[i] = dot + self.head_bias;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    /// Processa o array em tempo de execução validando features CPU.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            unsafe { self.process_avx512(input, output) }
        } else {
            unsafe { self.process_avx2(input, output) }
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

    #[test]
    fn test_lstm_model1_process_zeros() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();
            let input = [1.0, -1.0, 0.5, -0.5];
            let mut output = [0.0f32; 4];

            model.process(&input, &mut output);

            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "LSTM1×8: saída [{}] é NaN/Inf: {}", i, v);
            }
        }
    }

    #[test]
    fn test_lstm_model2_process_deterministic() {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut model_a: LstmModel2<8, 9, 16, 32> = LstmModel2::new();
            let mut model_b: LstmModel2<8, 9, 16, 32> = LstmModel2::new();

            let input = [1.0, -1.0, 0.5, -0.5];
            let mut out_a = [0.0f32; 4];
            let mut out_b = [0.0f32; 4];

            model_a.process(&input, &mut out_a);
            model_b.process(&input, &mut out_b);

            for i in 0..4 {
                assert!(
                    (out_a[i] - out_b[i]).abs() < 1e-6,
                    "LSTM2×8 não-determinístico em [{}]: {} vs {}",
                    i,
                    out_a[i],
                    out_b[i]
                );
            }
        }
    }

    #[test]
    fn test_lstm_gate_order_consistency() {
        // Valida que o layout [i|f|g|o] no offset [0, H, 2H, 3H] é respeitado.
        let model: LstmModel1<8, 9, 32> = LstmModel1::new();
        assert_eq!(model.layer.gates.len(), 32); // 4 * H = 4 * 8 = 32
        assert_eq!(model.layer.input_hidden_weights.len(), 32); // H4 = 32
        assert_eq!(model.layer.input_hidden_weights[0].len(), 9); // IH = I + H = 1 + 8 = 9
    }
}
