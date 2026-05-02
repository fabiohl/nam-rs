// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha de Células Recorrentes Otimizada (LSTM) para inferência NAM.
//!
//! Este módulo implementa a abstração LSTM otimizada com as instruções de `core::arch::x86_64` (FMA).
//! Utiliza arrays contíguos baseados em _Const Generics_ para evitar saltos condicionais na compilação.
//! O processamento adota a Estrutura de Arrays (SoA).

use core::arch::x86_64::*;

/// Uma camada individual do modelo LSTM.
///
/// **Referência Científica:** Hochreiter, S., & Schmidhuber, J. (1997). *"Long Short-Term Memory."*
///
/// Parâmetros Constantes:
/// * `I` = Input Size
/// * `H` = Hidden Size
/// * `IH` = Input + Hidden Size
/// * `H4` = 4 * Hidden Size
pub struct LstmLayer<const I: usize, const H: usize, const IH: usize, const H4: usize> {
    /// Matriz 3D agregada contendo os pesos interfolhados horizontalmente [I, F, C, O] por neurônio e entrada.
    pub input_hidden_weights: [[[u16; 4]; IH]; H],
    /// Bias lineares extraídos do modelo (tamanho `4 * Hidden`).
    pub bias: [f32; H4],
    /// Estado global contendo [Input | Hidden] em F32.
    pub state: [f32; IH],
    /// Estado global espelhado em BF16 para processamento VNNI.
    pub state_bf16: [u16; IH],
    /// Estado da Célula Interna LSTM.
    pub cell_state: [f32; H],
    /// Ativações lineares antes das portas C e Tanh.
    pub gates: [f32; H4],
}

macro_rules! define_lstm_process {
    (
        $fn_name:ident,
        $target_meta:meta,
        $dot_product_interleaved_f32:path,
        $dot_product_interleaved_bf16:path,
        $step:expr,
        $load:ident,
        $store:ident,
        $add:ident,
        $mul:ident,
        $tanh:path,
        $sigmoid:path,
        $fused_gates:path,
        $is_bf16:expr
    ) => {
        /// Executa o processamento de uma amostra de áudio através da camada LSTM.
        ///
        /// # Safety
        /// O chamador deve garantir que `input` tenha tamanho suficiente (pelo menos `I`)
        /// e que as instruções SIMD solicitadas em `target_meta` estejam disponíveis.
        #[$target_meta]
        pub unsafe fn $fn_name(&mut self, input: &[f32]) {
            unsafe {
                // 1. Prepara o buffer de estado: [Input | Hidden State]
                self.state[..I].copy_from_slice(&input[..I]);

                // Mirror para BF16 se necessário
                if $is_bf16 {
                    for i in 0..IH {
                        self.state_bf16[i] = (self.state[i].to_bits() >> 16) as u16;
                    }
                }

                _mm_prefetch::<{ _MM_HINT_T0 }>(self.state.as_ptr().cast::<i8>());
                if IH > 16 {
                    _mm_prefetch::<{ _MM_HINT_T0 }>(self.state.as_ptr().add(16).cast::<i8>());
                }

                // 3. Cálculo das Portas (Gates)
                for i in 0..H {
                    let w_slice = &self.input_hidden_weights[i];
                    let dots = if $is_bf16 {
                        $dot_product_interleaved_bf16(w_slice, &self.state_bf16)
                    } else {
                        $dot_product_interleaved_f32(w_slice, &self.state)
                    };

                    self.gates[i] = dots[0] + self.bias[i];
                    self.gates[i + H] = dots[1] + self.bias[i + H];
                    self.gates[i + 2 * H] = dots[2] + self.bias[i + 2 * H];
                    self.gates[i + 3 * H] = dots[3] + self.bias[i + 3 * H];
                }

                let f_offset = H;
                let g_offset = 2 * H;
                let o_offset = 3 * H;
                let h_offset = I;

                let mut i = 0;
                while i + $step <= H {
                    let g_f = $load(self.gates.as_ptr().add(i + f_offset));
                    let g_i = $load(self.gates.as_ptr().add(i));
                    let g_g = $load(self.gates.as_ptr().add(i + g_offset));
                    let g_o = $load(self.gates.as_ptr().add(i + o_offset));
                    let c_s = $load(self.cell_state.as_ptr().add(i));

                    let (new_c_s, h_s) = $fused_gates(g_f, g_i, g_g, g_o, c_s);

                    $store(self.cell_state.as_mut_ptr().add(i), new_c_s);
                    $store(self.state.as_mut_ptr().add(h_offset + i), h_s);

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
                    let g_o = $load(temp_go.as_ptr());
                    let c_s = $load(temp_cs.as_ptr());

                    let (new_c_s, h_val) = $fused_gates(g_f, g_i, g_g, g_o, c_s);

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
    /// Usamos arrays fixos (Const Generics) para garantir que a memória seja contígua,
    /// o que é vital para performance e previsibilidade em tempo real.
    pub fn new() -> Self {
        Self {
            input_hidden_weights: [[[0u16; 4]; IH]; H],
            bias: [0.0; H4],
            state: [0.0; IH],
            state_bf16: [0u16; IH],
            cell_state: [0.0; H],
            gates: [0.0; H4],
        }
    }

    /// Retorna o fatiamento da memória do estado atual que engloba a porção `Hidden`.
    /// Em uma LSTM, o estado completo é frequentemente [Input | Hidden].
    /// Esta função extrai apenas o Hidden State para ser usado na próxima camada ou na saída.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[I..]
    }

    /// Retorna o fatiamento da memória do estado atual em BF16.
    #[inline(always)]
    pub fn get_hidden_state_bf16(&self) -> &[u16] {
        &self.state_bf16[I..]
    }

    // Geramos a implementação AVX2 da função de processamento.
    // O compilador usará instruções de 256 bits (processa 8 f32 por vez).
    define_lstm_process!(
        process_sample_avx2,
        inline(always),
        crate::math::simd::dot_product_4x_interleaved_avx2,
        crate::math::simd::dot_product_4x_interleaved_bf16_avx512,
        8,                                   // $step: Processa 8 elementos por instrução
        _mm256_loadu_ps,                     // Carregamento não alinhado (unaligned load)
        _mm256_storeu_ps,                    // Armazenamento não alinhado (unaligned store)
        _mm256_add_ps,                       // Soma vetorial
        _mm256_mul_ps,                       // Multiplicação vetorial
        crate::math::fastmath::simd_tanh,    // Tanh otimizada para AVX2
        crate::math::fastmath::simd_sigmoid, // Sigmoid otimizada para AVX2
        crate::math::fastmath::fused_lstm_gates_avx2,
        false
    );

    // Geramos a implementação AVX-512 da função de processamento.
    // Usado em processadores modernos (ex: Intel Tiger Lake+ / AMD Zen 4+).
    // Processa 16 f32 por vez (512 bits).
    define_lstm_process!(
        process_sample_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        crate::math::simd::dot_product_4x_interleaved_avx512,
        crate::math::simd::dot_product_4x_interleaved_bf16_avx512,
        16, // $step: Processa 16 elementos por instrução
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::fastmath::simd_tanh_avx512, // Tanh otimizada para AVX-512
        crate::math::fastmath::simd_sigmoid_avx512, // Sigmoid otimizada para AVX-512
        crate::math::fastmath::fused_lstm_gates_avx512,
        false
    );

    define_lstm_process!(
        process_sample_avx2vnni,
        target_feature(enable = "avxvnni"),
        crate::math::simd::dot_product_4x_interleaved_avx2,
        crate::math::simd::dot_product_4x_interleaved_bf16_avx512,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        crate::math::fastmath::simd_tanh,
        crate::math::fastmath::simd_sigmoid,
        crate::math::fastmath::fused_lstm_gates_avx2,
        false
    );

    define_lstm_process!(
        process_sample_avx512vnni,
        target_feature(enable = "avx512f,avx512vl,avx512vnni"),
        crate::math::simd::dot_product_4x_interleaved_avx512,
        crate::math::simd::dot_product_4x_interleaved_bf16_avx512,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::fastmath::simd_tanh_avx512,
        crate::math::fastmath::simd_sigmoid_avx512,
        crate::math::fastmath::fused_lstm_gates_avx512,
        false
    );

    define_lstm_process!(
        process_sample_avx512_vnni_bf16,
        target_feature(enable = "avx512f,avx512vl,avx512bf16"),
        crate::math::simd::dot_product_4x_interleaved_avx512,
        crate::math::simd::dot_product_4x_interleaved_bf16_avx512, // Native AVX-512 BF16 VNNI
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::fastmath::simd_tanh_avx512,
        crate::math::fastmath::simd_sigmoid_avx512,
        crate::math::fastmath::fused_lstm_gates_avx512,
        true
    );

    /// Zera os estados internos (hidden e cell) da camada.
    /// Essencial ao trocar de preset ou iniciar um playback para evitar "estalos"
    /// ou carregar resíduos de áudios processados anteriormente.
    pub fn reset_states(&mut self) {
        self.state.fill(0.0);
        self.cell_state.fill(0.0);
        self.gates.fill(0.0);
    }
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> Default
    for LstmLayer<I, H, IH, H4>
{
    /// Implementação padrão que apenas chama o construtor `new`.
    fn default() -> Self {
        Self::new()
    }
}

/// Modelo LSTM com 1 camada recorrente (ex: NAM profile 1x8, 1x16).
pub struct LstmModel1<const H: usize, const H1_IH: usize, const H_H4: usize> {
    /// Camada única da malha.
    pub layer: LstmLayer<1, H, H1_IH, H_H4>,
    /// Pesos de extração direcional (Cabeça).
    pub head_weights: [u16; H],
    /// Bias escalar da projeção de saída (cabeça linear).
    pub head_bias: f32,
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmModel1<H, H1_IH, H_H4> {
    /// Inicializa a arquitetura em repouso.
    pub fn new() -> Self {
        Self {
            layer: LstmLayer::new(),
            head_weights: [0u16; H],
            head_bias: 0.0,
        }
    }

    /// Executa inferência sobre um arranjo contíguo de amostras de aúdio em blocos,
    /// avaliando a memória recorrente local.
    ///
    /// # Safety
    /// Exige compatibilidade com `avx2` e `fma` ativados no processador x86_64 hospedeiro.
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer.process_sample_avx2(&[input[i]]);
                h0.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx2(&[input[i + 1]]);
                h1.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx2(&[input[i + 2]]);
                h2.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx2(&[input[i + 3]]);
                h3.copy_from_slice(self.layer.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx2(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer.process_sample_avx2(&sample);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx2(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn process_avx512(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer.process_sample_avx512(&[input[i]]);
                h0.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx512(&[input[i + 1]]);
                h1.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx512(&[input[i + 2]]);
                h2.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx512(&[input[i + 3]]);
                h3.copy_from_slice(self.layer.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx512(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer.process_sample_avx512(&sample);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx512(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avxvnni")]
    unsafe fn process_avx2vnni(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer.process_sample_avx2vnni(&[input[i]]);
                h0.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx2vnni(&[input[i + 1]]);
                h1.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx2vnni(&[input[i + 2]]);
                h2.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx2vnni(&[input[i + 3]]);
                h3.copy_from_slice(self.layer.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx2(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer.process_sample_avx2vnni(&sample);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx2(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn process_avx512vnni(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer.process_sample_avx512vnni(&[input[i]]);
                h0.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx512vnni(&[input[i + 1]]);
                h1.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx512vnni(&[input[i + 2]]);
                h2.copy_from_slice(self.layer.get_hidden_state());

                self.layer.process_sample_avx512vnni(&[input[i + 3]]);
                h3.copy_from_slice(self.layer.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx512(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer.process_sample_avx512vnni(&sample);
                let hidden = self.layer.get_hidden_state();
                let dot = crate::math::simd::dot_product_avx512(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }
    #[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
    unsafe fn process_avx512_vnni_bf16(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0u16; H];
                let mut h1 = [0u16; H];
                let mut h2 = [0u16; H];
                let mut h3 = [0u16; H];

                self.layer.process_sample_avx512_vnni_bf16(&[input[i]]);
                h0.copy_from_slice(self.layer.get_hidden_state_bf16());

                self.layer.process_sample_avx512_vnni_bf16(&[input[i + 1]]);
                h1.copy_from_slice(self.layer.get_hidden_state_bf16());

                self.layer.process_sample_avx512_vnni_bf16(&[input[i + 2]]);
                h2.copy_from_slice(self.layer.get_hidden_state_bf16());

                self.layer.process_sample_avx512_vnni_bf16(&[input[i + 3]]);
                h3.copy_from_slice(self.layer.get_hidden_state_bf16());

                let dots = crate::math::simd::dot_product_bf16_4x_native_avx512(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer.process_sample_avx512_vnni_bf16(&sample);
                let hidden = self.layer.get_hidden_state_bf16();
                let dot =
                    crate::math::simd::dot_product_bf16_native_avx512(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    /// Processa o array em tempo de execução via v-table estática LazyLock.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(
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

    /// Zera os estados internos da malha LSTM.
    pub fn reset_states(&mut self) {
        self.layer.reset_states();
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
    pub head_weights: [u16; H],
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
            head_weights: [0u16; H],
            head_bias: 0.0,
        }
    }

    /// Rotina paralela e de estado contínuo (`sample-by-sample`) executada sequencialmente num bloco.
    ///
    /// # Safety
    /// Exige compatibilidade com `avx2` e `fma` em sistema de processamento com arquitetura x86_64.
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer1.process_sample_avx2(&[input[i]]);
                self.layer2
                    .process_sample_avx2(self.layer1.get_hidden_state());
                h0.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx2(&[input[i + 1]]);
                self.layer2
                    .process_sample_avx2(self.layer1.get_hidden_state());
                h1.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx2(&[input[i + 2]]);
                self.layer2
                    .process_sample_avx2(self.layer1.get_hidden_state());
                h2.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx2(&[input[i + 3]]);
                self.layer2
                    .process_sample_avx2(self.layer1.get_hidden_state());
                h3.copy_from_slice(self.layer2.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx2(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer1.process_sample_avx2(&sample);
                let hidden1 = self.layer1.get_hidden_state();

                self.layer2.process_sample_avx2(hidden1);
                let hidden2 = self.layer2.get_hidden_state();

                let dot = crate::math::simd::dot_product_avx2(hidden2, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn process_avx512(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer1.process_sample_avx512(&[input[i]]);
                self.layer2
                    .process_sample_avx512(self.layer1.get_hidden_state());
                h0.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx512(&[input[i + 1]]);
                self.layer2
                    .process_sample_avx512(self.layer1.get_hidden_state());
                h1.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx512(&[input[i + 2]]);
                self.layer2
                    .process_sample_avx512(self.layer1.get_hidden_state());
                h2.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx512(&[input[i + 3]]);
                self.layer2
                    .process_sample_avx512(self.layer1.get_hidden_state());
                h3.copy_from_slice(self.layer2.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx512(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer1.process_sample_avx512(&sample);
                let hidden1 = self.layer1.get_hidden_state();

                self.layer2.process_sample_avx512(hidden1);
                let hidden2 = self.layer2.get_hidden_state();

                let dot = crate::math::simd::dot_product_avx512(hidden2, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avxvnni")]
    unsafe fn process_avx2vnni(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer1.process_sample_avx2vnni(&[input[i]]);
                self.layer2
                    .process_sample_avx2vnni(self.layer1.get_hidden_state());
                h0.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx2vnni(&[input[i + 1]]);
                self.layer2
                    .process_sample_avx2vnni(self.layer1.get_hidden_state());
                h1.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx2vnni(&[input[i + 2]]);
                self.layer2
                    .process_sample_avx2vnni(self.layer1.get_hidden_state());
                h2.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx2vnni(&[input[i + 3]]);
                self.layer2
                    .process_sample_avx2vnni(self.layer1.get_hidden_state());
                h3.copy_from_slice(self.layer2.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx2(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer1.process_sample_avx2vnni(&sample);
                let hidden1 = self.layer1.get_hidden_state();

                self.layer2.process_sample_avx2vnni(hidden1);
                let hidden2 = self.layer2.get_hidden_state();

                let dot = crate::math::simd::dot_product_avx2(hidden2, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512vnni")]
    unsafe fn process_avx512vnni(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0.0; H];
                let mut h1 = [0.0; H];
                let mut h2 = [0.0; H];
                let mut h3 = [0.0; H];

                self.layer1.process_sample_avx512vnni(&[input[i]]);
                self.layer2
                    .process_sample_avx512vnni(self.layer1.get_hidden_state());
                h0.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx512vnni(&[input[i + 1]]);
                self.layer2
                    .process_sample_avx512vnni(self.layer1.get_hidden_state());
                h1.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx512vnni(&[input[i + 2]]);
                self.layer2
                    .process_sample_avx512vnni(self.layer1.get_hidden_state());
                h2.copy_from_slice(self.layer2.get_hidden_state());

                self.layer1.process_sample_avx512vnni(&[input[i + 3]]);
                self.layer2
                    .process_sample_avx512vnni(self.layer1.get_hidden_state());
                h3.copy_from_slice(self.layer2.get_hidden_state());

                let dots = crate::math::simd::dot_product_batch_4x_avx512(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer1.process_sample_avx512vnni(&sample);
                let hidden1 = self.layer1.get_hidden_state();

                self.layer2.process_sample_avx512vnni(hidden1);
                let hidden2 = self.layer2.get_hidden_state();

                let dot = crate::math::simd::dot_product_avx512(hidden2, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
    unsafe fn process_avx512_vnni_bf16(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            let mut i = 0;
            let len = input.len();

            while i + 4 <= len {
                let mut h0 = [0u16; H];
                let mut h1 = [0u16; H];
                let mut h2 = [0u16; H];
                let mut h3 = [0u16; H];

                self.layer1.process_sample_avx512_vnni_bf16(&[input[i]]);
                self.layer2
                    .process_sample_avx512_vnni_bf16(self.layer1.get_hidden_state());
                h0.copy_from_slice(self.layer2.get_hidden_state_bf16());

                self.layer1.process_sample_avx512_vnni_bf16(&[input[i + 1]]);
                self.layer2
                    .process_sample_avx512_vnni_bf16(self.layer1.get_hidden_state());
                h1.copy_from_slice(self.layer2.get_hidden_state_bf16());

                self.layer1.process_sample_avx512_vnni_bf16(&[input[i + 2]]);
                self.layer2
                    .process_sample_avx512_vnni_bf16(self.layer1.get_hidden_state());
                h2.copy_from_slice(self.layer2.get_hidden_state_bf16());

                self.layer1.process_sample_avx512_vnni_bf16(&[input[i + 3]]);
                self.layer2
                    .process_sample_avx512_vnni_bf16(self.layer1.get_hidden_state());
                h3.copy_from_slice(self.layer2.get_hidden_state_bf16());

                let dots = crate::math::simd::dot_product_bf16_4x_native_avx512(
                    &h0,
                    &h1,
                    &h2,
                    &h3,
                    &self.head_weights,
                );

                output[i] = dots[0] + self.head_bias;
                output[i + 1] = dots[1] + self.head_bias;
                output[i + 2] = dots[2] + self.head_bias;
                output[i + 3] = dots[3] + self.head_bias;

                i += 4;
            }

            while i < len {
                let sample = [input[i]];
                self.layer1.process_sample_avx512_vnni_bf16(&sample);
                self.layer2
                    .process_sample_avx512_vnni_bf16(self.layer1.get_hidden_state());
                let hidden = self.layer2.get_hidden_state_bf16();
                let dot =
                    crate::math::simd::dot_product_bf16_native_avx512(hidden, &self.head_weights);
                output[i] = dot + self.head_bias;
                i += 1;
            }
        }
    }

    /// Processa o array em tempo de execução via v-table estática LazyLock.
    ///
    /// Utiliza `SimdMathConfig::get().instruction_set` avaliado no startup
    /// para despachar sem leitura atômica em hot-path.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(
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

    /// Zera os estados internos das camadas LSTM.
    /// Deve ser chamado sempre que o processamento de um novo sinal começar do zero.
    pub fn reset_states(&mut self) {
        self.layer1.reset_states();
        self.layer2.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> Default
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Cria uma instância padrão do modelo de 2 camadas.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "lstm_test.rs"]
mod lstm_test;
