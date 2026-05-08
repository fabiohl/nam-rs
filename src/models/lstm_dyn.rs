// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha de Células Recorrentes Dinâmica para inferência LSTM (Fallback).
//!
//! Permite carregamento de modelos com topologias LSTM arbitrárias
//! (e.g., mais de 2 camadas, tamanhos ocultos customizados) que não possuem
//! implementações SoA estáticas via Const Generics.
//! Para respeitar o path RT Lock-Free, toda manipulação de memória é
//! resolvida com buffers passados diretamente à pipeline iterativa vetorial math da SIMMathConfig.

/// Camada Dinâmica LSMT gerida em tempo de execução
#[derive(Clone)]
pub struct LstmDynLayer {
    /// Peso `(Input + Hidden)` por `(4 * Hidden)` matriz dimensionada 1D e linearizada.
    pub input_hidden_weights: Vec<u16>,
    /// Array linear (H * 4) com os desvios de ativação de cada porta.
    pub bias: Vec<f32>,
    /// Tensor de Estado local, funde o sample entrante e o momento final do frame anterior. [I + H]
    pub state: Vec<f32>,
    /// Estado global espelhado em BF16 para processamento VNNI.
    pub state_bf16: Vec<u16>,
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
    pub unsafe fn process_sample<M: crate::math::simd::SimdMath>(&mut self, input: &[f32]) {
        // ih = Input + Hidden: O tamanho total do vetor que entra no cálculo (X_t + H_t-1)
        let ih = self.input_size + self.hidden_size;
        // h = Hidden: A dimensão do estado interno e da saída desta camada
        let h = self.hidden_size;

        // 1. Atualiza state com sample de input (substitui no prefixo)
        // O vetor 'state' contém [Input_da_vez | Hidden_da_rodada_anterior].
        // Aqui, injetamos o novo áudio no começo do buffer para a próxima multiplicação.
        self.state[..self.input_size].copy_from_slice(input);

        // OTIMIZAÇÃO: "Prefetch" avisa a CPU para trazer os dados da memória RAM para o Cache L1
        // ANTES de começarmos o loop pesado de cálculos. Isso reduz a latência de acesso (Memory Stall).
        unsafe {
            // T0 = Traz para todos os níveis de cache (L1, L2, L3)
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                self.state.as_ptr().cast::<i8>(),
            );
            // Se o vetor for grande, fazemos o prefetch da próxima "cache line" (64 bytes/16 floats)
            if ih > 16 {
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                    self.state.as_ptr().add(16).cast::<i8>(),
                );
            }
        }

        // 2. Linear Dot Products -> Preenche GATES e adiciona BIAS via GEMV
        if M::IS_BF16 {
            // Sincroniza estado BF16
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

        // 3. Kernel fundido: Ativações + Atualização de Estado
        // Realiza sigmoids (I, F, O) e tanh (G), atualiza cell_state e hidden_state (state[input_size..])
        // em um único passo fundido, reduzindo o tráfego de memória de 3 passes para 1.
        unsafe {
            M::fused_lstm_gates_dyn(
                &mut self.gates,
                &mut self.cell_state,
                &mut self.state[self.input_size..],
                h,
            );
        }
    }

    /// Retorna a parte visível do estado oculto (hidden state).
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
    /// Em modelos complexos, podemos ter 2, 3 ou mais camadas onde uma alimenta a outra.
    pub layers: Vec<LstmDynLayer>,
    /// Filtro de predição linear da saída (Head).
    /// Após todas as camadas LSTM, aplicamos uma última camada linear para transformar
    /// o estado oculto final no valor do sample de áudio.
    pub head_weights: Vec<u16>,
    /// Acumulador linear projetado de saída.
    pub head_bias: f32,
}

impl LstmDynModel {
    /// Processamento Síncrono de Áudio. Requer instanciamento RT.
    /// Esta função processa um bloco de áudio, amostra por amostra.
    ///
    /// **Para Cientistas e Devs:** A macro `dispatch_simd!` inspeciona o CPU (ex: tem AVX-512? AVX2?)
    /// de forma imediata (pré-resolvida no startup) e despacha para a função `process_internal`
    /// parametrizada com o tipo de matemática SIMD otimizado (`M: SimdMath`).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Função restrita para processamento neural onde o compilador substitui genéricos `M`
    /// pelas instruções intrínsecas adequadas (como FMA AVX2) da SIMDMath selecionada.
    unsafe fn process_internal<M: crate::math::simd::SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) {
        let num_frames = input.len();

        for i in 0..num_frames {
            // Pegamos o sample atual de áudio.
            let current_sample = [input[i]];
            // O 'hidden_out' começa como o sample de entrada e vai sendo atualizado
            // conforme passamos por cada camada da rede.
            let mut hidden_out: &[f32] = &current_sample;

            // Loop de camadas: A saída (hidden state) da camada N
            // serve como entrada (input) para a camada N+1.
            for layer in self.layers.iter_mut() {
                unsafe { layer.process_sample::<M>(hidden_out) };
                // Pega o estado oculto que acabou de ser calculado para a próxima camada
                hidden_out = layer.get_hidden_state();
            }

            // Camada de Saída (Head): Multiplicamos o estado oculto da última camada
            // pelos pesos do 'head' e somamos o bias para obter o sample final.
            let head_out =
                unsafe { M::dot_product(hidden_out, &self.head_weights) } + self.head_bias;

            // Salva o resultado no buffer de saída
            output[i] = head_out;
        }
    }

    /// Processamento Síncrono de Áudio (Fallback)
    /// Executa o aquecimento (Pre-warm) do estado interno da LSTM.
    /// O dispatch garante que o silêncio de pre-warm seja calculado na mesma velocidade
    /// que o processamento real.
    pub fn prewarm(&mut self, num_samples: usize) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, prewarm_internal, num_samples);
        }
    }

    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    unsafe fn prewarm_internal<M: crate::math::simd::SimdMath>(&mut self, num_samples: usize) {
        self.reset_states();

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

    /// Zera os estados internos de todas as camadas LSTM.
    pub fn reset_states(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_states();
        }
    }
}
