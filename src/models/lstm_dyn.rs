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

        // 2. Linear Dot Products -> Preenche GATES e adiciona BIAS
        // Aqui processamos cada unidade oculta (h). Os pesos estão organizados de forma
        // "intercalada" (interleaved) para que possamos calcular as 4 portas da LSTM
        // (Input, Forget, Cell, Output) simultaneamente usando otimizações SIMD.
        for i in 0..h {
            // Calcula o deslocamento inicial no array linear de pesos para esta unidade 'i'
            let start = i * ih * 4;
            let w_interleaved = &self.input_hidden_weights[start..start + ih * 4];

            // PERFORMANCE: Convertemos o slice de f32 em um slice de blocos [f32; 4].
            // Isso permite que a função de dot product processe os pesos das 4 portas de uma vez,
            // tratando cada conexão (input/hidden) como um vetor de 4 elementos.
            let w_slice = unsafe {
                core::slice::from_raw_parts(w_interleaved.as_ptr() as *const [f32; 4], ih)
            };

            // Calcula o produto escalar quádruplo: soma(peso_gate[0..3] * state_val)
            // O resultado 'dots' contém 4 valores: [Soma_I, Soma_F, Soma_C, Soma_O]
            let dots = unsafe { M::dot_product_4x_interleaved(w_slice, &self.state) };

            // Distribui os resultados para o buffer de portas (gates) somando o bias correspondente.
            // O buffer 'gates' armazena os valores de forma contígua por porta:
            // [Input...|Forget...|Cell...|Output...] -> cada bloco tem tamanho 'h'.
            self.gates[i] = dots[0] + self.bias[i]; // Gate I (Input)
            self.gates[i + h] = dots[1] + self.bias[i + h]; // Gate F (Forget)
            self.gates[i + 2 * h] = dots[2] + self.bias[i + 2 * h]; // Gate C (Cell/G)
            self.gates[i + 3 * h] = dots[3] + self.bias[i + 3 * h]; // Gate O (Output)
        }

        // 3. Funções de Ativação (FastMath SIMD via Slices in-place)
        // No NAM (se IFGO), a ordem das portas concatenadas nos pesos é: Input, Forget, Cell, Output
        // Aplicamos Sigmoid nas portas de controle (0, 1 e 3) para que os valores fiquem entre 0 e 1,
        // funcionando como "válvulas" de fluxo de informação.
        // A porta 2 (Cell/G) usa Tanh para normalizar a nova informação entre -1 e 1.
        unsafe {
            // gates[0..h] : Input (Sigmoid) | gates[h..2h] : Forget (Sigmoid)
            M::sigmoid_slice(&mut self.gates[0..2 * h]);
            // gates[2h..3h] : Cell/Grau-G (Tanh)
            M::tanh_slice(&mut self.gates[2 * h..3 * h]);
            // gates[3h..4h] : Output (Sigmoid)
            M::sigmoid_slice(&mut self.gates[3 * h..4 * h]);
        }

        // 4. Element-wise: state propagation para o cell_state e hidden_state
        // Esta é a "alma" da LSTM: a célula de memória (Cell State).
        for j in 0..h {
            let sig_i = self.gates[j]; // Válvula de Entrada
            let sig_f = self.gates[j + h]; // Válvula de Esquecimento
            let tanh_g = self.gates[j + 2 * h]; // Nova informação candidata

            // Equação fundamental:
            // Novo_Estado_Célula = (O que lembrar do passado) + (O que aprender do presente)
            let new_cs = sig_f * self.cell_state[j] + sig_i * tanh_g;
            self.cell_state[j] = new_cs;

            // Preparamos uma cópia para o cálculo do Tanh na próxima etapa
            self.tanh_cs[j] = new_cs;
        }

        // 5. Calcula Tanh(New Cell State) usando o vetor alocador.
        // Normalizamos o estado da célula para ser filtrado pela porta de saída.
        unsafe {
            M::tanh_slice(&mut self.tanh_cs[0..h]);
        }

        // 6. Fecha Output state e avança momento
        // O Hidden State (Saída) é o estado da célula "filtrado" pela porta de saída (Sigmoid).
        for j in 0..h {
            let sig_o = self.gates[j + 3 * h]; // Válvula de Saída
            let h_val = sig_o * self.tanh_cs[j];

            // IMPORTANTE: Atualizamos a segunda parte do vetor 'state'.
            // Na próxima amostra de áudio (next sample), este h_val será o "Hidden anterior".
            self.state[self.input_size + j] = h_val;
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
    pub head_weights: Vec<f32>,
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
                unsafe { M::dot_product(&self.head_weights, hidden_out) } + self.head_bias;

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
