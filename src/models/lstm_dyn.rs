// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
// Portado da implementação original do NeuralAudio por Mike Oliphant.

//! Malha de Células Recorrentes Dinâmica para inferência LSTM (Fallback).
//!
//! Permite carregamento de modelos com topologias LSTM arbitrárias
//! (e.g., mais de 2 camadas, tamanhos ocultos customizados) que não possuem
//! implementações SoA estáticas via Const Generics.
//! Para respeitar o path RT Lock-Free, toda manipulação de memória é
//! resolvida com buffers passados diretamente à pipeline iterativa vetorial math da SIMMathConfig.

use crate::math::simd::SimdMathConfig;

/// Camada Dinâmica LSMT gerida em tempo de execução
#[derive(Clone)]
pub struct LstmDynLayer {
    /// Peso `(Input + Hidden)` por `(4 * Hidden)` matriz dimensionada 1D e linearizada.
    pub input_hidden_weights: Vec<f32>,
    /// Array linear (H * 4) com os desvios de ativação de cada porta.
    pub bias: Vec<f32>,
    /// Tensor de Estado local, funde o sample entrante e o momento final do frame anterior. [I + H]
    pub state: Vec<f32>,
    /// Bloco restrito de recursividade matemática da célula LSTM. [H]
    pub cell_state: Vec<f32>,
    /// Buffers de acúmulo da avaliação vetorial sobre a camada. [H * 4]
    pub gates: Vec<f32>,
    /// Buffer auxiliar de escopo contíguo. [H]
    pub tanh_cs: Vec<f32>,
    /// Dimensão física do vetor inicial `[1]` ou contiguidade prévia num Lstm2.
    pub input_size: usize,
    /// Dimensão latente matemática.
    pub hidden_size: usize,
}

impl LstmDynLayer {
    /// Desdobra a malha sobre os ciclos do SimdMathConfig
    ///
    /// # Safety
    /// Depende nativamente das matrizes preenchidas via alocador estrito no C++ Fallback parser (Loader CLI).
    pub unsafe fn process_sample(&mut self, input: &[f32], math: &SimdMathConfig) {
        let ih = self.input_size + self.hidden_size;
        let h = self.hidden_size;

        // 1. Atualiza state com sample de input (substitui no prefixo)
        self.state[..self.input_size].copy_from_slice(input);

        unsafe {
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                self.state.as_ptr().cast::<i8>(),
            );
            if ih > 16 {
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                    self.state.as_ptr().add(16).cast::<i8>(),
                );
            }
        }

        // 2. Linear Dot Products -> Preenche GATES e adiciona BIAS
        for i in 0..h {
            let start_i = (i * 4) * ih;
            let w_i = &self.input_hidden_weights[start_i..start_i + ih];

            let start_f = (i * 4 + 1) * ih;
            let w_f = &self.input_hidden_weights[start_f..start_f + ih];

            let start_c = (i * 4 + 2) * ih;
            let w_c = &self.input_hidden_weights[start_c..start_c + ih];

            let start_o = (i * 4 + 3) * ih;
            let w_o = &self.input_hidden_weights[start_o..start_o + ih];

            let dots = unsafe { (math.dot_product_4x)(w_i, w_f, w_c, w_o, &self.state) };

            self.gates[i] = dots[0] + self.bias[i];
            self.gates[i + h] = dots[1] + self.bias[i + h];
            self.gates[i + 2 * h] = dots[2] + self.bias[i + 2 * h];
            self.gates[i + 3 * h] = dots[3] + self.bias[i + 3 * h];
        }

        // 3. Funções de Ativação (FastMath SIMD via Slices in-place)
        // No NAM (se IFGO), a ordem das portas concatenadas nos pesos é: Input, Forget, Cell, Output
        // gates[0..h] : Input (Sigmoid)
        // gates[h..2h] : Forget (Sigmoid)
        // gates[2h..3h] : Cell/Grau-G (Tanh)
        // gates[3h..4h] : Output (Sigmoid)
        unsafe {
            (math.sigmoid_slice)(&mut self.gates[0..2 * h]); // As portas Input e Forget são Sigmoid (contíguas)
            (math.tanh_slice)(&mut self.gates[2 * h..3 * h]); // A porta Cell (C/G) é Tanh
            (math.sigmoid_slice)(&mut self.gates[3 * h..4 * h]); // A porta Output é Sigmoid
        }

        // 4. Element-wise: state propagation para o cell_state e hidden_state
        for j in 0..h {
            let sig_i = self.gates[j];
            let sig_f = self.gates[j + h];
            let tanh_g = self.gates[j + 2 * h];

            let new_cs = sig_f * self.cell_state[j] + sig_i * tanh_g;
            self.cell_state[j] = new_cs;
            // Prepara para Tanh
            self.tanh_cs[j] = new_cs;
        }

        // 5. Calcula Tanh(New Cell State) usando o vetor alocador.
        unsafe {
            (math.tanh_slice)(&mut self.tanh_cs[0..h]);
        }

        // 6. Fecha Output state e avança momento
        for j in 0..h {
            let sig_o = self.gates[j + 3 * h];
            let h_val = sig_o * self.tanh_cs[j];
            self.state[self.input_size + j] = h_val;
        }
    }

    /// Retorna a fração visível oculta do estado transiente para passar entre layers.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[self.input_size..]
    }

    /// Zera os estados internos (hidden e cell) da camada.
    pub fn reset_states(&mut self) {
        self.state.fill(0.0);
        self.cell_state.fill(0.0);
        self.gates.fill(0.0);
        self.tanh_cs.fill(0.0);
    }
}

/// Invólucro Dinâmico LSTM final.
///
/// **Referência Científica:** Hochreiter, S., & Schmidhuber, J. (1997). *"Long Short-Term Memory."*
pub struct LstmDynModel {
    /// Conjunto linearizado empilhado das subrotinas LSTM.
    pub layers: Vec<LstmDynLayer>,
    /// Filtro de predição linear da saída.
    pub head_weights: Vec<f32>,
    /// Acumulador linear projetado de saída.
    pub head_bias: f32,
}

impl LstmDynModel {
    /// Processamento Síncrono de Áudio. Requer instanciamento RT.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let math = crate::math::simd::SimdMathConfig::get();
        let num_frames = input.len();

        for i in 0..num_frames {
            let current_sample = [input[i]];
            let mut hidden_out: &[f32] = &current_sample;

            for layer in self.layers.iter_mut() {
                unsafe {
                    layer.process_sample(hidden_out, math);
                }
                hidden_out = layer.get_hidden_state();
            }

            let head_out =
                unsafe { (math.dot_product)(&self.head_weights, hidden_out) } + self.head_bias;
            output[i] = head_out;
        }
    }

    /// Processamento Síncrono de Áudio (Fallback)
    /// Executa o aquecimento do loop inicial da máquina transiente.
    ///
    /// Processa `num_samples` amostras zero em chunks de 512 para estabilizar
    /// os estados oculto e celular. Consistente com `LstmModel1`/`LstmModel2`.
    /// Ref C++: `Prewarm(2048, 64)` — o default de 2048 amostras é propagado
    /// pela trait `NamModel::prewarm(num_samples)` no despacho.
    pub fn prewarm(&mut self, num_samples: usize) {
        self.reset_states();

        const CHUNK: usize = 512;
        let mut zero_out = [0.0f32; CHUNK];
        let zero_in = [0.0f32; CHUNK];
        let mut rem = num_samples;

        while rem > 0 {
            let n = rem.min(CHUNK);
            self.process(&zero_in[..n], &mut zero_out[..n]);
            rem -= n;
        }
    }

    /// Zera os estados internos das camadas LSTM.
    pub fn reset_states(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_states();
        }
    }
}
