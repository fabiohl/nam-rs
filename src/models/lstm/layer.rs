// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Malha de Células Recorrentes Otimizada (LSTM) para inferência NAM.

use core::arch::x86_64::*;

/// Uma camada individual do modelo LSTM otimizada para SIMD.
pub struct LstmLayer<const I: usize, const H: usize, const IH: usize, const H4: usize> {
    /// Pesos da camada no layout Gate-Major.
    pub input_hidden_weights: [[[u16; H]; IH]; 4],
    /// Bias lineares da camada.
    pub bias: [f32; H4],
    /// Buffer de estado consolidado [Input | Hidden].
    pub state: [f32; IH],
    /// Espelho do estado em BF16 para aceleração VNNI.
    pub state_bf16: [u16; IH],
    /// Estado da célula (C) do LSTM.
    pub cell_state: [f32; H],
    /// Ativações intermediárias das portas.
    pub gates: [f32; H4],
}

macro_rules! define_lstm_process {
    (
        $fn_name:ident,
        $target_meta:meta,
        $simd_math:ty,
        $gemv_4gate:path,
        $gemv_4gate_bf16:path,
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
        /// Processa uma amostra através da camada LSTM.
        ///
        /// # Safety
        /// O chamador deve garantir suporte às instruções SIMD especificadas.
        // NOTE: $target_meta permite injetar #[inline(always)] para AVX2 (nosso baseline x86-64-v3)
        // ou #[target_feature] para extensões superiores, garantindo codegen correto.
        #[$target_meta]
        pub unsafe fn $fn_name(&mut self, input: &[f32]) {
            unsafe {
                // 1. Alimentamos a 'memória' do modelo com o novo fragmento de áudio.
                self.state[..I].copy_from_slice(&input[..I]);

                // 2. Se o hardware suportar BF16 (Brain Floating Point), convertemos os dados.
                // Isso mantém a escala do áudio mas usa metade do espaço, acelerando o cálculo.
                if $is_bf16 {
                    use $crate::math::common::SimdMath;
                    <$simd_math>::f32_to_bf16(&self.state[..I], &mut self.state_bf16[..I]);
                }

                // 3. 'Prefetch': Avisamos o processador para buscar os pesos na memória RAM
                // um pouco antes de precisarmos deles, evitando que o cálculo pare para esperar os dados.
                _mm_prefetch::<{ _MM_HINT_T0 }>(self.state.as_ptr().cast::<i8>());

                // 4. Multiplicação de Matriz-Vetor (GEMV):
                // Aqui multiplicamos a entrada e o estado anterior pelos pesos (o 'cérebro' treinado).
                // O resultado ativa as 4 'portas' do LSTM: Esquecimento, Entrada, Candidata e Saída.
                if $is_bf16 {
                    $gemv_4gate_bf16(
                        &self.state_bf16,
                        self.input_hidden_weights[0].as_flattened(),
                        self.input_hidden_weights[1].as_flattened(),
                        self.input_hidden_weights[2].as_flattened(),
                        self.input_hidden_weights[3].as_flattened(),
                        &self.bias,
                        &mut self.gates,
                        true,
                    );
                } else {
                    $gemv_4gate(
                        &self.state,
                        self.input_hidden_weights[0].as_flattened(),
                        self.input_hidden_weights[1].as_flattened(),
                        self.input_hidden_weights[2].as_flattened(),
                        self.input_hidden_weights[3].as_flattened(),
                        &self.bias,
                        &mut self.gates,
                        true,
                    );
                }

                // 5. Mapeamos onde cada uma das 4 portas começa no nosso buffer de cálculo.
                let f_offset = H; // Porta de Esquecimento (Forget)
                let g_offset = 2 * H; // Porta de Atualização (Cell Candidate)
                let o_offset = 3 * H; // Porta de Saída (Output)
                let h_offset = I; // Onde guardamos o resultado final para o próximo passo

                let mut i = 0;
                // 6. Loop Principal (SIMD): Processamos vários neurônios em paralelo (8 ou 16 por vez).
                while i + $step <= H {
                    // Carregamos os valores pré-calculados das 4 portas e da memória atual (cell state).
                    let g_f = $load(self.gates.as_ptr().add(i + f_offset));
                    let g_i = $load(self.gates.as_ptr().add(i));
                    let g_g = $load(self.gates.as_ptr().add(i + g_offset));
                    let g_o = $load(self.gates.as_ptr().add(i + o_offset));
                    let c_s = $load(self.cell_state.as_ptr().add(i));

                    // 'Fused Gates': A mágica do LSTM acontece aqui.
                    // Decidimos o que esquecer da memória antiga e o que aprender da nova entrada.
                    let (new_c_s, h_s) = $fused_gates(g_f, g_i, g_g, g_o, c_s);

                    // Salvamos a nova memória (longo prazo) e a nova saída (curto prazo).
                    $store(self.cell_state.as_mut_ptr().add(i), new_c_s);
                    $store(self.state.as_mut_ptr().add(h_offset + i), h_s);

                    if $is_bf16 {
                        use $crate::math::common::SimdMath;
                        <$simd_math>::store_bf16(
                            self.state_bf16.as_mut_ptr().add(h_offset + i),
                            h_s,
                        );
                    }
                    i += $step;
                }

                // 7. Tratamento de 'Cauda': Se o número de neurônios não for um múltiplo exato do
                // processamento paralelo, cuidamos dos últimos elementos individualmente.
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
                        if $is_bf16 {
                            self.state_bf16[i + j + h_offset] = (out_h[j].to_bits() >> 16) as u16;
                        }
                    }
                }
            }
        }
    };
}
impl<const I: usize, const H: usize, const IH: usize, const H4: usize> LstmLayer<I, H, IH, H4> {
    /// Cria uma nova camada LSTM zerada.
    pub fn new() -> Self {
        Self {
            input_hidden_weights: [[[0u16; H]; IH]; 4],
            bias: [0.0; H4],
            state: [0.0; IH],
            state_bf16: [0u16; IH],
            cell_state: [0.0; H],
            gates: [0.0; H4],
        }
    }
    /// Retorna uma referência ao estado oculto atual.
    #[inline(always)]
    pub fn get_hidden_state(&self) -> &[f32] {
        &self.state[I..]
    }
    /// Retorna uma referência ao estado oculto atual espelhado em BF16.
    #[inline(always)]
    pub fn get_hidden_state_bf16(&self) -> &[u16] {
        &self.state_bf16[I..]
    }

    // --- Especializações por Hardware (SIMD) ---
    // Criamos versões diferentes da mesma lógica para tirar o máximo proveito de cada CPU.

    // 1. Especialização AVX2 (Padrão para x86-64-v3):
    // Utiliza registradores de 256 bits (processando 8 floats em paralelo) com aproximadores
    // rápidos de Tanh e Sigmoid via SIMD AVX2.
    define_lstm_process!(
        process_sample_avx2,
        inline(always),
        crate::math::common::Avx2Math,
        crate::math::gemm::gemv_4gate_avx2,
        crate::math::common::gemv_4gate_bf16_fallback,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        crate::math::activations::simd_tanh_avx2,
        crate::math::activations::simd_sigmoid_avx2,
        crate::math::lstm::fused_lstm_gates_avx2,
        false
    );

    // 2. Especialização AVX-512 (F/VL):
    // Utiliza registradores de 512 bits (processando 16 floats de uma só vez). Ideal para
    // servidores ou CPUs Intel/AMD mais recentes que suportam instruções vetoriais estendidas.
    define_lstm_process!(
        process_sample_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        crate::math::common::Avx512Math,
        crate::math::gemm::gemv_4gate_avx512,
        crate::math::common::gemv_4gate_bf16_fallback, // Sem BF16 nativo aqui
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::activations::simd_tanh_avx512,
        crate::math::activations::simd_sigmoid_avx512,
        crate::math::lstm::fused_lstm_gates_avx512,
        false
    );

    // 3. Especialização AVX2 VNNI (Vector Neural Network Instructions):
    // Voltado para processamento de redes neurais em CPUs que suportam aceleração VNNI
    // sobre registradores de 256 bits, otimizando o throughput aritmético.
    define_lstm_process!(
        process_sample_avx2vnni,
        target_feature(enable = "avxvnni"),
        crate::math::common::Avx2Math,
        crate::math::gemm::gemv_4gate_avx2,
        crate::math::common::gemv_4gate_bf16_fallback,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        crate::math::activations::simd_tanh_avx2,
        crate::math::activations::simd_sigmoid_avx2,
        crate::math::lstm::fused_lstm_gates_avx2,
        false
    );

    // 4. Especialização AVX-512 VNNI:
    // Combina a largura de registrador de 512 bits (16 floats) com a aceleração de hardware VNNI
    // para ganho massivo de throughput em operações GEMV.
    define_lstm_process!(
        process_sample_avx512vnni,
        target_feature(enable = "avx512f,avx512vl,avx512vnni"),
        crate::math::common::Avx512VnniMath,
        crate::math::gemm::gemv_4gate_avx512,
        crate::math::common::gemv_4gate_bf16_fallback,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::activations::simd_tanh_avx512,
        crate::math::activations::simd_sigmoid_avx512,
        crate::math::lstm::fused_lstm_gates_avx512,
        false
    );

    // O ápice da otimização: Uso de BF16 nativo para velocidade extrema.
    define_lstm_process!(
        process_sample_avx512_vnni_bf16,
        target_feature(enable = "avx512f,avx512vl,avx512bf16"),
        crate::math::common::Avx512VnniBf16Math,
        crate::math::gemm::gemv_4gate_avx512,
        crate::math::gemm::gemv_4gate_bf16_avx512,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::activations::simd_tanh_avx512,
        crate::math::activations::simd_sigmoid_avx512,
        crate::math::lstm::fused_lstm_gates_avx512,
        true
    );

    /// Processamento escalar (fallback) para testes e benchmarks.
    ///
    /// Esta é a versão 'manual' e lenta, usada apenas como referência para garantir
    /// que as versões ultra-rápidas acima não tenham erros matemáticos.
    #[inline(always)]
    pub fn process_sample_scalar(&mut self, input: &[f32], is_bf16: bool) {
        let ih = I + H;
        let h = H;
        self.state[..I].copy_from_slice(&input[..I]);

        // Multiplicação manual dos pesos pelas entradas (4 portas).
        for k in 0..4 {
            let target_gate_offset = k * h;
            for i in 0..h {
                let mut sum = 0.0;
                for (j, &s) in self.state.iter().enumerate().take(ih) {
                    let w = self.input_hidden_weights[k][j][i];
                    let w_f32 = if is_bf16 {
                        f32::from_bits((w as u32) << 16)
                    } else {
                        half::f16::from_bits(w).to_f32()
                    };
                    sum += w_f32 * s;
                }
                self.gates[target_gate_offset + i] = sum + self.bias[target_gate_offset + i];
            }
        }

        // Ativação manual das portas do LSTM.
        for j in 0..h {
            let gf = self.gates[j + h];
            let gi = self.gates[j];
            let gg = self.gates[j + 2 * h];
            let go = self.gates[j + 3 * h];
            let cs = self.cell_state[j];

            // Funções de ativação (Sigmoid e Tanh) que moldam o sinal.
            let f = 0.5 * (1.0 + (gf * 0.5).tanh());
            let i = 0.5 * (1.0 + (gi * 0.5).tanh());
            let g = gg.tanh();
            let o = 0.5 * (1.0 + (go * 0.5).tanh());

            let new_cs = f * cs + i * g;
            let h_val = o * new_cs.tanh();

            self.cell_state[j] = new_cs;
            self.state[I + j] = h_val;
        }
    }

    /// Reseta apenas o slot de entrada, preservando o estado oculto e o estado da célula.
    ///
    /// Usado no prewarm para evitar descartar os estados iniciais `_xh` e `_c`
    /// carregados do arquivo NAM.
    pub fn reset_input_slot(&mut self) {
        self.state[..I].fill(0.0);
    }

    /// Reseta os estados internos para zero.
    ///
    /// Importante para 'limpar a memória' do modelo entre diferentes execuções.
    pub fn reset_states(&mut self) {
        self.state.fill(0.0);
        self.cell_state.fill(0.0);
        self.gates.fill(0.0);
    }
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> Default
    for LstmLayer<I, H, IH, H4>
{
    fn default() -> Self {
        Self::new()
    }
}


