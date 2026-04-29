// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha CNN Causal Estática para inferência WaveNet (Design Orientado a Dados, SoA).
//!
//! Todas as estruturas utilizam `Const Generics` nas dimensões matemáticas e vetores pré-alocados
//! garantindo uma política de instanciamento estrito (Zero-Allocation durante processamento).
//! As loops dinâmicos resolvem cálculos em sequências FMA determinísticas via AVX2.

#![allow(clippy::needless_range_loop)]

use crate::math::simd::SimdMath;

/// Máximo de frames a processar em um pulso do callback.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Padding temporal circular das memórias no framework de Ring Buffers.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;

/// Convolução Causal Dilatada (WaveNet Conv1D).
#[derive(Clone)]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Matriz achatada de pesos do tamanho OUT * K * IN.
    pub weights: Vec<f32>,
    /// Viés causal, atrelado se do_bias for verdadeiro. Total: OUT.
    pub bias: Vec<f32>,
    /// Determina se o array de bias deve ser somado.
    pub do_bias: bool,
    /// Fator de diluição no eixo temporal causacional (Ex: 1, 2, 4.. 512).
    pub dilation: usize,
}

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Executa convolução causal num array bidirecional flat (`layer_buffer`).
    ///
    /// ## Otimização: Software Prefetch Proativo
    ///
    /// Para dilatações grandes (256, 512), os acessos ao `layer_buffer` saltam
    /// milhares de floats entre taps consecutivos do kernel, provocando cache
    /// misses L1 previsíveis. O `_mm_prefetch` emitido para o **próximo tap**
    /// enquanto o tap atual é processado via FMA permite ao memory subsystem
    /// trazer a cache line proativamente — custo de 1 ciclo (mascarado pelo
    /// pipeline FMA), benefício de ~5–10% de latência em layers com dilatação alta.
    ///
    /// # Safety
    /// Depende dinamicamente da trait `SimdMath` fornecida.
    ///
    /// Processa um único frame aplicando convolução ao ring buffer (otimizado via FMA 4x).
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
    ) {
        // [PASSO 1: Inicialização do Acumulador]
        // Prepara o vetor de saída `out_frame`. Se a convolução incluir um "bias" (deslocamento),
        // nós copiamos esses valores diretamente. Se não, garantimos que comece em zero.
        // Como o output é acumulativo (+=) mais adiante, esta base inicial é crítica.
        if self.do_bias {
            out_frame.copy_from_slice(&self.bias[0..OUT]);
        } else {
            out_frame.fill(0.0);
        }

        // [PASSO 2: Iteração do Kernel (Receptive Field)]
        // Iteramos sobre cada "tap" do kernel. Para um WaveNet clássico, K = 3.
        for k in 0..K {
            // [OTIMIZAÇÃO L1: Software Prefetching]
            // A convolução dilatada significa que os frames passados não estão adjacentes na memória,
            // mas sim afastados pelo fator `dilation`. Para evitar que o processador pare (stall)
            // esperando o próximo dado vir da RAM, usamos uma instrução intrínseca `_mm_prefetch`.
            // Ela solicita antecipadamente o carregamento do PRÓXIMO tap (k + 1) para o Cache L1.
            if k + 1 < K {
                let next_offset = (self.dilation as isize) * ((k as isize) + 2 - (K as isize));
                let next_frame_idx = (frame_idx as isize) + next_offset;
                let next_addr =
                    unsafe { layer_buffer.as_ptr().add((next_frame_idx as usize) * IN) };
                unsafe {
                    core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                        next_addr.cast::<i8>(),
                    );
                }
            }

            // [PASSO 3: Cálculo do Índice Dilatado]
            // Encontra a posição temporal correta no `layer_buffer` do histórico (passado causal)
            // baseando-se no tap atual `k` e fator de dilatação.
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let current_frame_idx = (frame_idx as isize) + offset;
            let in_slice_start = (current_frame_idx as usize) * IN;

            // Fatiamos os dados de entrada de todos os canais para aquele instante histórico.
            let in_slice =
                unsafe { layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN) };

            // [PASSO 4: Multiplicação Acumulativa SIMD (Otimização FMA 4x)]
            // Para maximizar o throughput, calculamos as saídas em blocos de 4 canais.
            // Os dados da matriz de pesos (flat array) são acessados em pedaços de `IN` tamanho,
            // executando 4 dot-products (produto escalar) simultaneamente contra o mesmo `in_slice`.
            let mut out_c = 0;
            while out_c + 4 <= OUT {
                let w0_start = (out_c * K + k) * IN;
                let w1_start = ((out_c + 1) * K + k) * IN;
                let w2_start = ((out_c + 2) * K + k) * IN;
                let w3_start = ((out_c + 3) * K + k) * IN;

                let w0 = unsafe { self.weights.get_unchecked(w0_start..w0_start + IN) };
                let w1 = unsafe { self.weights.get_unchecked(w1_start..w1_start + IN) };
                let w2 = unsafe { self.weights.get_unchecked(w2_start..w2_start + IN) };
                let w3 = unsafe { self.weights.get_unchecked(w3_start..w3_start + IN) };

                // O dispatch SIMD resolve isso na microarquitetura alvo usando registradores vetoriais.
                let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_slice) };

                // Acumulamos o resultado do tap `k` no output final deste frame.
                unsafe {
                    *out_frame.get_unchecked_mut(out_c) += r0;
                    *out_frame.get_unchecked_mut(out_c + 1) += r1;
                    *out_frame.get_unchecked_mut(out_c + 2) += r2;
                    *out_frame.get_unchecked_mut(out_c + 3) += r3;
                }
                out_c += 4;
            }

            // [PASSO 5: Cauda da Camada (Tail Loop)]
            // Trata eventuais canais residuais se OUT não for múltiplo de 4.
            while out_c < OUT {
                let w_start = (out_c * K + k) * IN;
                let w = unsafe { self.weights.get_unchecked(w_start..w_start + IN) };
                let r = unsafe { M::dot_product(in_slice, w) };
                unsafe {
                    *out_frame.get_unchecked_mut(out_c) += r;
                }
                out_c += 1;
            }
        }
    }

    /// Processa bloco iterativo sequencial.
    /// Para eficiência no cache, em vez de processar toda a camada por múltiplos blocos,
    /// limitamos a chamadas consecutivas quadro a quadro (`process_single_frame`).
    ///
    /// # Safety
    /// Pointer must be valid e num_frames deve estar contido nos limites do layer_buffer.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            // [PASSO: Delegação por Frame]
            // Fatia o buffer de saída (output multi-canal do tamanho `OUT`) e despacha para cálculo.
            let out_frame = unsafe { block.get_unchecked_mut(i * OUT..i * OUT + OUT) };
            unsafe {
                self.process_single_frame::<M>(layer_buffer, out_frame, buffer_start + i);
            }
        }
    }
}

/// Camada Densa 1x1 baseada num Matmul vetorizado linear.
#[derive(Clone)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    /// Matriz de pesos lineares (OUT * IN).
    pub weights: Vec<f32>,
    /// Condição temporal para o deslocador de tensor bias.
    pub bias: Vec<f32>,
    /// Determina se o array de bias deve ser somado.
    pub do_bias: bool,
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    /// Processa um único frame do Dense Layer acumulando com o output atual (otimizado com FMA 4x).
    ///
    /// # Safety
    /// Depende dinamicamente da trait `SimdMath`.
    #[inline(always)]
    pub unsafe fn process_acc_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        // [PASSO 1: Otimização Loop Unrolling 4x]
        // Para uma camada densa 1x1, realizamos a projeção de IN canais para OUT canais.
        // O `while` agrupa as operações em 4 canais de saída simultâneos, reduzindo
        // substancialmente o gargalo de instruções (Instruction Bottleneck).
        let mut out_c = 0;
        while out_c + 4 <= OUT {
            // Extrai fatias contíguas (IN floats) correspondentes a 4 filtros de peso.
            let w0 = unsafe { self.weights.get_unchecked(out_c * IN..(out_c + 1) * IN) };
            let w1 = unsafe {
                self.weights
                    .get_unchecked((out_c + 1) * IN..(out_c + 2) * IN)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked((out_c + 2) * IN..(out_c + 3) * IN)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked((out_c + 3) * IN..(out_c + 4) * IN)
            };

            // [PASSO 2: Produto Escalar Multi-Vetorial]
            // M::dot_product_4x multiplica 4 vetores de pesos contra o ÚNICO vetor de input (in_frame).
            // A arquitetura x86 lida com o "broadcast" do vetor in_frame nas unidades FMA,
            // maximizando o uso dos registradores e economizando chamadas da RAM.
            let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };

            // Extração de bias de forma preditiva.
            let b0 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            let b1 = if self.do_bias {
                self.bias[out_c + 1]
            } else {
                0.0
            };
            let b2 = if self.do_bias {
                self.bias[out_c + 2]
            } else {
                0.0
            };
            let b3 = if self.do_bias {
                self.bias[out_c + 3]
            } else {
                0.0
            };

            // [PASSO 3: Acumulação In-Place]
            // Como esta função é `process_acc` (Accumulate), nós SOMAMOS (`+=`)
            // os resultados computados mais o bias ao valor já existente no buffer de saída.
            // Isso é fundamental na arquitetura WaveNet para a mistura entre blocos.
            unsafe {
                *out_frame.get_unchecked_mut(out_c) += r0 + b0;
                *out_frame.get_unchecked_mut(out_c + 1) += r1 + b1;
                *out_frame.get_unchecked_mut(out_c + 2) += r2 + b2;
                *out_frame.get_unchecked_mut(out_c + 3) += r3 + b3;
            }
            out_c += 4;
        }

        // [PASSO 4: Loop Residual (Tail)]
        // Processa os canais remanescentes se `OUT` não for divisível por 4.
        while out_c < OUT {
            let w = unsafe { self.weights.get_unchecked(out_c * IN..(out_c + 1) * IN) };
            let r = unsafe { M::dot_product(in_frame, w) };
            let b = if self.do_bias { self.bias[out_c] } else { 0.0 };
            unsafe {
                *out_frame.get_unchecked_mut(out_c) += r + b;
            }
            out_c += 1;
        }
    }

    /// Processa um único frame substituindo o buffer existente.
    ///
    /// # Safety
    /// Depende dinamicamente da trait `SimdMath`.
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        // [PASSO 1: Otimização Loop Unrolling 4x]
        // Diferente de `process_acc_single_frame`, esta função SUBSTITUI (`=`) os valores no `out_frame`
        // em vez de acumulá-los (`+=`). Executamos 4 projeções de canais em paralelo (SIMD 4x).
        let mut out_c = 0;
        while out_c + 4 <= OUT {
            // [PASSO 1.1: Fatiamento Estático de Pesos]
            // Pegamos 4 linhas da matriz de pesos (cada uma contendo IN elementos).
            // Ao usar `.get_unchecked`, evitamos os branches de bounds check em rust,
            // garantindo que o compilador vectorize e execute em O(1) de overhead.
            let w0 = unsafe { self.weights.get_unchecked(out_c * IN..(out_c + 1) * IN) };
            let w1 = unsafe {
                self.weights
                    .get_unchecked((out_c + 1) * IN..(out_c + 2) * IN)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked((out_c + 2) * IN..(out_c + 3) * IN)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked((out_c + 3) * IN..(out_c + 4) * IN)
            };

            // [PASSO 2: Produto Escalar SIMD Simultâneo]
            // Executa FMA (Fused Multiply-Add) de 4 canais contra a mesma entrada em um único dispatch.
            let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };

            // [PASSO 3: Resolução Preditiva de Bias]
            // O pipeline FMA é profundo. Enquanto ele resolve o dot_product, nós usamos a ALU
            // escalar para resolver a adição do Bias. O compilador emite branches CMOV (Conditional Move).
            let b0 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            let b1 = if self.do_bias {
                self.bias[out_c + 1]
            } else {
                0.0
            };
            let b2 = if self.do_bias {
                self.bias[out_c + 2]
            } else {
                0.0
            };
            let b3 = if self.do_bias {
                self.bias[out_c + 3]
            } else {
                0.0
            };

            // [PASSO 4: Substituição no Buffer de Saída]
            // Realizamos a soma final e atribuímos DIRETAMENTE `=` ao output (não é cumulativo `+=`).
            unsafe {
                *out_frame.get_unchecked_mut(out_c) = r0 + b0;
                *out_frame.get_unchecked_mut(out_c + 1) = r1 + b1;
                *out_frame.get_unchecked_mut(out_c + 2) = r2 + b2;
                *out_frame.get_unchecked_mut(out_c + 3) = r3 + b3;
            }
            out_c += 4;
        }

        // [PASSO 5: Cauda da Computação (Tail Loop)]
        // Quando `OUT` não for múltiplo de 4, os canais excedentes caem aqui e são
        // processados 1 a 1 de forma clássica.
        while out_c < OUT {
            let w = unsafe { self.weights.get_unchecked(out_c * IN..(out_c + 1) * IN) };
            let r = unsafe { M::dot_product(in_frame, w) };
            let b = if self.do_bias { self.bias[out_c] } else { 0.0 };
            unsafe {
                *out_frame.get_unchecked_mut(out_c) = r + b;
            }
            out_c += 1;
        }
    }

    /// Processa o Dense acumulando com o estado corrente de output.
    ///
    /// # Safety
    /// Despacho matemático via trait inlined.
    #[inline(always)]
    pub unsafe fn process_acc_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        // [DIFERENCIAL: Processamento Contíguo de Bloco]
        // Esta função investe os loops: em vez de iterar sobre quadros no anel externo e canais
        // no interno (como Conv1D), ela itera sobre blocos de CANAIS no anel externo e sobre os QUADROS (frames)
        // no anel interno. A matriz densa (`weights`) é lida apenas uma vez do L2 para o L1,
        // e todo o `input` buffer é projetado contra ela, maximizando o reuso de registradores (Register Renaming).
        let mut out_c = 0;
        while out_c + 4 <= OUT {
            // [PASSO 1: Lock dos Pesos L1]
            // Carrega e segura os 4 perfis de peso no L1.
            let w0 = unsafe { self.weights.get_unchecked(out_c * IN..(out_c + 1) * IN) };
            let w1 = unsafe {
                self.weights
                    .get_unchecked((out_c + 1) * IN..(out_c + 2) * IN)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked((out_c + 2) * IN..(out_c + 3) * IN)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked((out_c + 3) * IN..(out_c + 4) * IN)
            };

            let b0 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            let b1 = if self.do_bias {
                self.bias[out_c + 1]
            } else {
                0.0
            };
            let b2 = if self.do_bias {
                self.bias[out_c + 2]
            } else {
                0.0
            };
            let b3 = if self.do_bias {
                self.bias[out_c + 3]
            } else {
                0.0
            };

            // [PASSO 2: Iteração Temporal Massiva (Frames)]
            // Despacha os frames continuamente contra as matrizes de pesos trancadas (cacheadas) no topo.
            // Para buffers `num_frames` = 64, são 64 FMA dispatches sem recarregar o `w0..w3`
            // Isso aumenta drasticamente o TFLOPS alcançável em microarquiteturas modernas.
            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };

                // Acúmulo somativo (`+=`) que consolida o bypass/skip do Dense Layer.
                unsafe {
                    *output.get_unchecked_mut(i * OUT + out_c) += r0 + b0;
                    *output.get_unchecked_mut(i * OUT + out_c + 1) += r1 + b1;
                    *output.get_unchecked_mut(i * OUT + out_c + 2) += r2 + b2;
                    *output.get_unchecked_mut(i * OUT + out_c + 3) += r3 + b3;
                }
            }
            out_c += 4;
        }

        // [PASSO 3: Loop Residual]
        while out_c < OUT {
            let weight_slice = unsafe { self.weights.get_unchecked(out_c * IN..out_c * IN + IN) };
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let sum = unsafe { M::dot_product(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * OUT + out_c) += sum + bias;
                }
            }
            out_c += 1;
        }
    }

    #[inline(always)]
    /// Processa bloco iterativo substituindo (OVERWRITE) os valores passados em vez de acumular.
    /// É comumente invocado por instâncias de fechamento tensorial (ex: Rechannel Head) e portões de entrada.
    ///
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        let mut out_c = 0;
        // [PASSO 1: Distribuição FMA Paralela]
        // Resolve blocos 4x de canais para utilizar plenamente a pipeline matemática AVX2 (256-bit).
        while out_c + 4 <= OUT {
            let w0 = unsafe { self.weights.get_unchecked(out_c * IN..(out_c + 1) * IN) };
            let w1 = unsafe {
                self.weights
                    .get_unchecked((out_c + 1) * IN..(out_c + 2) * IN)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked((out_c + 2) * IN..(out_c + 3) * IN)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked((out_c + 3) * IN..(out_c + 4) * IN)
            };

            let b0 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            let b1 = if self.do_bias {
                self.bias[out_c + 1]
            } else {
                0.0
            };
            let b2 = if self.do_bias {
                self.bias[out_c + 2]
            } else {
                0.0
            };
            let b3 = if self.do_bias {
                self.bias[out_c + 3]
            } else {
                0.0
            };

            // [PASSO 2: Varredura de Otimização L1 (Frames Inner Loop)]
            // Mantendo os pesos (`w0`, `w1`, etc) trancados na memória Cache L1 da CPU, iteramos sobre
            // a temporalidade de quadros do áudio (frames) extraindo throughput superior.
            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };

                // Gravação direta (`=`) no buffer de saída da projeção desta camada Densa 1x1.
                unsafe {
                    *output.get_unchecked_mut(i * OUT + out_c) = r0 + b0;
                    *output.get_unchecked_mut(i * OUT + out_c + 1) = r1 + b1;
                    *output.get_unchecked_mut(i * OUT + out_c + 2) = r2 + b2;
                    *output.get_unchecked_mut(i * OUT + out_c + 3) = r3 + b3;
                }
            }
            out_c += 4;
        }

        while out_c < OUT {
            let weight_slice = unsafe { self.weights.get_unchecked(out_c * IN..out_c * IN + IN) };
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let sum = unsafe { M::dot_product(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * OUT + out_c) = sum + bias;
                }
            }
            out_c += 1;
        }
    }
}

/// Célula Convolucional Completa (WaveNet Layer).
#[derive(Clone)]
pub struct WaveNetLayer<const COND: usize, const CH: usize, const K: usize> {
    /// Malha de convolução Causal 1D paramétrica dilatada desta camada.
    pub conv1d: Conv1d<CH, CH, K>,
    /// Rede em injeção de mistura Condisional.
    pub input_mixin: DenseLayer<COND, CH>,
    /// Transformação afim linear de descompressão 1x1 da camada.
    pub one_by_one: DenseLayer<CH, CH>,
}

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processa uma camada integral do WaveNet, iterando `FastMath` em AVX2.
    ///
    /// # Safety
    /// Despacho matemático via ponteiro para funções intrínsecas inlined.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_block_internal<M: SimdMath>(
        &self,
        condition: &[f32],
        head_input: &mut [f32],
        output: &mut [f32],
        layer_buffer: &[f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        unsafe {
            for i in 0..num_frames {
                // Buffer transiente em stack. `CH` é pequeno e alinhado (ex: 8 ou 16).
                // Isso evita completamente alocação dinâmica.
                let mut temp = [0.0f32; CH];

                // [PASSO 1: Convolução Causal Dilatada]
                // Aplica o filtro da camada atual acessando o histórico no Ring Buffer.
                // O resultado vai para `temp`.
                self.conv1d
                    .process_single_frame::<M>(layer_buffer, &mut temp, buffer_start + i);

                // [PASSO 2: Condicionamento (Input Mixin)]
                // Acumula (+-) a projeção da matriz densa sobre o sinal de entrada
                // que condiciona globalmente o modelo. Isso soma direto em `temp`.
                let cond_frame = &condition[i * COND..i * COND + COND];
                self.input_mixin
                    .process_acc_single_frame::<M>(cond_frame, &mut temp);

                // [PASSO 3: Função de Ativação Tanh (FastMath Minimax)]
                // Em vez do tradicional Gated Activation (tanh * sigm) do WaveNet original,
                // arquiteturas eficientes de NAM usam frequentemente Tanh pura para velocidade,
                // substituindo std::math por polinomiais AVX2.
                M::tanh_slice(&mut temp);

                // [PASSO 4: Projeção de Saída da Célula (Skip Connection / Head)]
                // A saída ativada `temp` é acumulada no `head_input`, que agrega os resultados
                // de TODAS as camadas para produzir o somatório que forma a amostra final de áudio.
                let head_frame = &mut head_input[i * CH..i * CH + CH];
                for j in 0..CH {
                    *head_frame.get_unchecked_mut(j) += temp[j];
                }

                // [PASSO 5: Transformação 1x1 (Residual Connection)]
                // Multiplicamos `temp` por uma matriz 1x1 (one_by_one) para gerar o output deste frame.
                let out_frame = &mut output[i * CH..i * CH + CH];
                self.one_by_one.process_single_frame::<M>(&temp, out_frame);

                // [PASSO 6: Soma Residual]
                // Somamos a saída transformativa ao sinal de entrada puro da camada
                // (armazenado em `layer_buffer[buffer_start + i]`). Esse `out_frame`
                // será o input da PRÓXIMA camada na rede.
                let lb_start = (buffer_start + i) * CH;
                for j in 0..CH {
                    *out_frame.get_unchecked_mut(j) += *layer_buffer.get_unchecked(lb_start + j);
                }
            }
        }
    }
}

/// Gerencia a memória buffer de uma célula WaveNet.
///
/// Alinhamento de 64 bytes (uma cache line) garante que `buffer_start` e
/// `receptive_field_size` não compartilhem cache line com o estado da camada
/// adjacente ao iterar `states_ptr.add(i)` no hot-path do `process()`.
///
/// # Rewind amortizado
///
/// O `rewind_buffer` executa `copy_within` de `receptive_field_size × CH` floats
/// a cada ~24 × 64 = 1536 amostras processadas (~32 ms a 48 kHz). É um custo
/// amortizado aceitável (~6–10 µs uma vez a cada ~24 callbacks). Spikes
/// observados em benchmarks refletem esse evento caindo dentro do intervalo
/// medido; em produção o efeito é diluído no jitter total do sistema.
#[repr(align(64))]
#[derive(Clone)]
pub struct WaveNetLayerState {
    /// Vetor base plano linear do Ring Buffer (zero alocações em contexto DSP).
    pub layer_buffer: std::vec::Vec<f32>,
    /// Ponteiro numérico do frame atual (avança a cada frame processado).
    pub buffer_start: usize,
    /// Dimensão física do espaço vetorial receptivo (tamanho do histórico de dilatação).
    pub receptive_field_size: usize,
}

impl WaveNetLayerState {
    /// Construtor alocador estático do Estado (executar antes do Thread DSP).
    pub fn new(channels: usize, receptive_field_size: usize, alloc_num: usize) -> Self {
        // [PASSO 1: Cálculo do Tamanho do Buffer Temporal]
        // O `buffer_frames` não é apenas o tamanho do histórico necessário (receptive_field_size).
        // Adicionamos um "padding" circular (LAYER_ARRAY_BUFFER_PADDING + 1) vezes WAVENET_MAX_NUM_FRAMES.
        // Isso nos dá folga suficiente para avançar o ponteiro linearmente muitas vezes antes de precisarmos
        // fazer uma cópia em bloco (rewind) para resetar a posição e evitar vazamento de memória.
        let buffer_frames =
            receptive_field_size + (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES;
        let buffer = vec![0.0f32; buffer_frames * channels];

        // [PASSO 2: Offset Inicial (Jittering Alocado)]
        // Subtraímos um valor baseado em `alloc_num`. Cada camada recebe um buffer_start sutilmente
        // diferente. Isso garante que os eventos de "rewind_buffer" aconteçam em MOMENTOS DIFERENTES
        // para cada camada durante a execução em tempo-real. Evita-se um pico (spike) de CPU concentrado,
        // diluindo o custo computacional do rewind ao longo do tempo (Balanceamento de Carga temporal).
        let start = buffer_frames
            - (WAVENET_MAX_NUM_FRAMES * ((alloc_num % LAYER_ARRAY_BUFFER_PADDING) + 1));

        Self {
            layer_buffer: buffer,
            buffer_start: start,
            receptive_field_size,
        }
    }

    /// Executa um passo do ponteiro do Ring Buffer. Se chegar na margem, chama Re-Wind.
    pub fn advance_frames(&mut self, num_frames: usize, channels: usize) {
        // [PASSO 1: Avanço Linear]
        // Diferente de Ring Buffers tradicionais que usam operações de módulo (ex: `ptr % size`) no hot-path,
        // o que custa ciclos consideráveis de CPU nas interações do L1, nós simplesmente andamos com o ponteiro
        // de forma puramente linear.
        self.buffer_start += num_frames;
        let buffer_frames = self.layer_buffer.len() / channels;

        // [PASSO 2: Condição de Borda (Threshold)]
        // Se, ao avançar, percebermos que não sobrará espaço para processar o próximo bloco
        // completo (WAVENET_MAX_NUM_FRAMES), nós deflagramos o evento de rebobinagem.
        if self.buffer_start + WAVENET_MAX_NUM_FRAMES > buffer_frames {
            self.rewind_buffer(channels);
        }
    }

    /// Rebina a memória do Ring Buffer para evitar overflow circular conservando o hitspace L1.
    ///
    /// # Invariante
    ///
    /// `buffer_start >= receptive_field_size` (garantido por `advance_frames`).
    /// Violar este invariante causaria underflow na subtração `buffer_start - receptive_field_size`,
    /// resultando em acesso fora dos limites do buffer e comportamento indefinido.
    pub fn rewind_buffer(&mut self, channels: usize) {
        debug_assert!(
            self.buffer_start >= self.receptive_field_size,
            "rewind_buffer: buffer_start ({}) < receptive_field_size ({})",
            self.buffer_start,
            self.receptive_field_size
        );
        // [PASSO 1: Cálculo de Limites]
        // O `start` se torna a margem mínima estrita necessária (o tamanho do campo receptivo causal) para que,
        // no próximo quadro, a convolução dilatada consiga "olhar para trás" o suficiente
        // sem causar violação de segmento ou acesso sujo.
        let start = self.receptive_field_size;

        // [PASSO 2: Fatias Base]
        // Vamos capturar a porção EXATA do histórico válido (que tem o tamanho `receptive_field_size`)
        // a partir da posição *atual* (`self.buffer_start`) para levá-la de volta ao início.
        let from = (self.buffer_start - self.receptive_field_size) * channels;
        let to = (start - self.receptive_field_size) * channels;
        let len = self.receptive_field_size * channels;

        // [PASSO 3: Cópia Overlapada (Memmove nativo)]
        // `copy_within` em Rust é extremamente rápido e compila intrinsecamente para um `memmove` otimizado
        // em Assembly (como AVX). É aqui que a "rebobinagem" acontece efetivamente:
        // deslizamos toda a janela temporal atual intacta para o início do buffer plano.
        self.layer_buffer.copy_within(from..from + len, to);
        self.buffer_start = start;
    }

    /// Preenche o buffer histórico para estabilizar o modelo no estado de warm-up.
    pub fn copy_buffer(&mut self, channels: usize) {
        // [PASSO: Propagação de Carga Constante]
        // Na fase de "pre-warm" (quando ligamos o áudio), nós injetamos sinal zero (silêncio).
        // Este loop copia o estado limpo para trás iterativamente preenchendo todo o `receptive_field_size`.
        // É de vital importância para que o modelo numérico não exploda (produza NaNs ou "clicks" sonoros) ao
        // tentar realizar as suas primeiras convoluções causais usando memória não-inicializada.
        for offset in 1..=self.receptive_field_size {
            let src = self.buffer_start * channels;
            let dst = (self.buffer_start - offset) * channels;
            self.layer_buffer.copy_within(src..src + channels, dst);
        }
    }
}

/// Unidade de Múltiplos Layers agrupados do WaveNet.
pub struct WaveNetLayerArray<
    const IN: usize,
    const COND: usize,
    const CH: usize,
    const K: usize,
    const HEAD: usize,
> {
    /// Vec com a topologia estrutural (comprimento define blocos).
    pub layers: Vec<WaveNetLayer<COND, CH, K>>,
    /// Estados do RingBuffer, um para cada Layer do sistema.
    pub states: Vec<WaveNetLayerState>,
    /// Abertura tensorial inicial `Dense`.
    pub rechannel: DenseLayer<IN, CH>,
    /// Fechamento tensorial final gerando projeção Head.
    pub head_rechannel: DenseLayer<CH, HEAD>,

    /// Array temporário pre-alocado para saídas.
    pub array_outputs: std::vec::Vec<f32>,
    /// Acumulador intermediário CH-sized para contribuições das camadas antes da projeção Head.
    pub head_accum: std::vec::Vec<f32>,
    /// Memória alocada da projeção Linear global (HEAD-sized).
    pub head_outputs: std::vec::Vec<f32>,
    /// Tamanho do campo dimensional (receptive field global) para roteamentos.
    pub receptive_field_size: usize,
}

impl<const IN: usize, const COND: usize, const CH: usize, const K: usize, const HEAD: usize>
    WaveNetLayerArray<IN, COND, CH, K, HEAD>
{
    /// Processamento central da Array. Totalmente blindado contra alocações.
    ///
    /// # Safety
    /// Ponteiros de states iteram internamente sem bounds checks.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_block_internal<M: SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();

        // [PASSO 1: Zero-Acumulador]
        // Zera o acumulador das saídas "Skip Connections" (Head) para este bloco de frames.
        // É essencial pois cada camada do array somará sua contribuição aqui.
        self.head_accum[0..num_frames * CH].fill(0.0);

        unsafe {
            // [PASSO 2: Abertura Dimensional (Rechannel)]
            // A camada `rechannel` pega o sinal de áudio mono de entrada e o expande
            // para `CH` canais (ex: de 1 para 16). O resultado é armazenado no buffer
            // de histórico da PRIMEIRA camada (`state_0`).
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * CH],
                num_frames,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            // [PASSO 3: Cascata de Camadas (Feed-Forward)]
            // Iteramos em profundidade sobre a Array de WaveNet. O output (residual) de
            // uma camada vira a entrada da PRÓXIMA camada, armazenada no buffer desta próxima.
            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    // Para a última camada, não há "próximo estado" para salvar o residual.
                    // Nós jogamos o residual no `self.array_outputs`, para ser o final da Array.
                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..num_frames * CH],
                        &mut self.array_outputs[0..num_frames * CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        num_frames,
                    );
                } else {
                    // Preparamos o buffer da PRÓXIMA camada para receber o residual gerado aqui.
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..num_frames * CH],
                        &mut next_state.layer_buffer[next_start..next_start + num_frames * CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        num_frames,
                    );
                }

                // [PASSO 4: Avanço Temporal]
                // Anda o ponteiro circular de histórico. Se passar do limite predefinido,
                // ele rebobina automaticamente (`rewind_buffer`).
                current_state.advance_frames(num_frames, CH);
            }

            // [PASSO 5: Fechamento Dimensional (Head Rechannel)]
            // A matriz densa afunila o acumulador (soma das skip-connections de todas as camadas,
            // de tamanho `CH`) para uma menor dimensão `HEAD` (ex: 16 -> 8 ou 16 -> 1).
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * CH],
                &mut self.head_outputs[0..num_frames * HEAD],
                num_frames,
            );
        }
    }

    /// Processa dados no modo Pre-warm para inicializar e estabilizar a memória temporal.
    ///
    /// [EXPLICAÇÃO CIENTÍFICA]
    /// Redes neurais de convolução causal como WaveNet possuem um estado interno que depende
    /// ativamente de N passos no passado (Receptive Field). Ao carregar um modelo novo, a
    /// memória da rede (Ring Buffers) alocada possui "zeros" puritanos ou lixo computacional.
    /// O Pre-warm alimenta um sinal inerte (Silêncio Absoluto) contínuo para a rede de modo a
    /// preencher toda a janela do passado. Os transientes resultantes desse cold-start "escoam"
    /// silenciosamente para o limbo, garantindo que a primeira amostra de áudio ao ligar a placa
    /// soe orgânica e estável, sem estalos ou cliques (clicks/pops).
    #[inline(always)]
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    pub unsafe fn prewarm_internal<M: SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();

        // [PASSO 1: Zero-Acumulador]
        // Preparativos de warm-up. O acumulador de skip-connections é limpo para este 1 único
        // frame iterativo (num_frames = 1).
        self.head_accum[0..CH].fill(0.0);

        unsafe {
            // [PASSO 2: Abertura Dimensional Simulada]
            // Expande os `layer_inputs` estáticos (geralmente [0.0]) de mono para o barramento de `CH` canais
            // e escreve na primeira camada temporal do modelo (`state_0`).
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + CH],
                1,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                // [PASSO 3: Propagação do Estado Estático]
                // DIFERENCIAL IMPORTANTE: Em vez de avançar o ponteiro (como no áudio em tempo real),
                // nós chamamos `copy_buffer` que literalmente recopia o valor recém-processado
                // retroativamente, preenchendo TODO o Receptive Field desta camada específica de uma só vez.
                current_state.copy_buffer(CH);

                // [PASSO 4: Avaliação Numérica "Fantasma"]
                // Efetua um ciclo de avaliação completo do tensor da camada, propagando o sinal nulo
                // pela rede. Embora a entrada seja silêncio, camadas possuem matrizes de Bias que
                // agregam valor real, ou seja, o "silêncio" de saída da rede *não* é exatamente zero.
                if i == last_layer {
                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..CH],
                        &mut self.array_outputs[0..CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        1,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..CH],
                        &mut next_state.layer_buffer[next_start..next_start + CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        1,
                    );
                }
            }

            // [PASSO 5: Fechamento]
            // Resolve a passagem do frame inicial nulo pela camada densa de fechamento (Head Rechannel).
            // Ao final deste fluxo, a arquitetura está purgada, alinhada e perfeitamente equilibrada
            // no ponto numérico inerte do amplificador real. Pronta para processar sinal musical.
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..CH],
                &mut self.head_outputs[0..HEAD],
                1,
            );
        }
    }
}

/// Modelo Completo do WaveNet contendo Dois Blocos de Layer Arrays heterogêneos.
///
/// **Referência Científica:** van den Oord, A., et al. (2016). *"WaveNet: A Generative Model for Raw Audio."* DeepMind.
///
/// `CH` = canais da Array1 (layer 0 do JSON, ex: 16 para Standard)
/// `K`  = kernel size (sempre 3)
/// `HEAD` = head_size da Array1 = canais da Array2 (ex: 8 para Standard)
///
/// Array2 usa `HEAD` canais e projeta para 1 saída (`HEAD2=1`),
/// seguindo o padrão C++: `WaveNetLayerArrayT<CH, 1, 1, HEAD, K, Dilations, true>`.
pub struct WaveNetModel<const CH: usize, const K: usize, const HEAD: usize> {
    /// Array interno 01: IN=1, COND=1, CH canais, HEAD saídas, sem HeadBias.
    pub array1: WaveNetLayerArray<1, 1, CH, K, HEAD>,
    /// Array interno 02: IN=CH, COND=1, HEAD canais, 1 saída, com HeadBias.
    pub array2: WaveNetLayerArray<CH, 1, HEAD, K, 1>,
    /// Escala de compensação da voltagem final (Target Output Scale).
    pub head_scale: f32,
    /// Maior buffer circular requerido na raiz temporal do Kernel.
    pub receptive_field_size: usize,
}

impl<const CH: usize, const K: usize, const HEAD: usize> WaveNetModel<CH, K, HEAD> {
    /// Resolve o forward total e produz amostras de onda em zero alocação (DSP).
    ///
    /// Combina as saídas de ambas as arrays: `sum(head1) + sum(head2)` × `head_scale`.
    ///
    /// **Para Cientistas e Devs:** Aqui acontece o "pulo do gato" da performance (SIMD Dispatch).
    /// Em vez de usarmos `if/else` lentos a cada frame para checar a CPU (AVX2 vs AVX-512),
    /// a macro `dispatch_simd!` avalia o hardware uma única vez e "teletransporta" a execução
    /// para uma versão clonada (monomorfizada) desta função estritamente otimizada para o seu processador.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { crate::math::simd::dispatch_simd!(self, process_internal, input, output) };
    }

    #[inline(always)]
    /// Rotina genérica e veloz que implementa a rede neural (WaveNet).
    /// A restrição `<M: SimdMath>` obriga o compilador a gerar o assembly focado em
    /// registradores grandes (256-bit ou 512-bit) sem ramificações (branchless).
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        let mut pos = 0;
        // [PROCESSAMENTO EM CHUNKS (BLOCOS)]
        // Para manter invariantes de zero-allocation (sem alocar vetores RAM temporários)
        // e respeitar a hierarquia restrita de Cache L1/L2, limitamos o processamento
        // a `WAVENET_MAX_NUM_FRAMES` (tipicamente 64 amostras) por vez.
        // Esse loop iterará até consumir todo o buffer (ex: 256, 512, 1024 frames).
        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                // [PASSO 1: Array1 Forward]
                // Condicionamento e Input (1D: 1 canal) -> formatado como blocos de IN frames.
                // Na topologia NAM padrão, esta Array realiza convoluções usando dilatações enormes
                // (ex: de 1 a 512, 1 a 512 sucessivamente) para capturar sub-graves de amplificadores.
                // Seu output entra em `array1.array_outputs` e os skips em `array1.head_outputs`.
                self.array1
                    .process_block_internal::<M>(in_slice, in_slice, num_frames);

                // [PASSO 2: Array2 Forward]
                // A segunda array atua tipicamente como uma camada perceptron de fechamento
                // (dimensões menores, dilatações apenas de 1, processando o "mix" vindo da Array1).
                let array1_outputs = &self.array1.array_outputs[0..num_frames * CH];
                self.array2
                    .process_block_internal::<M>(array1_outputs, in_slice, num_frames);
            }

            // [PASSO 3: Soma das Skips + Escala Final SIMD]
            // Somatório SIMD das projeções Head de ambas as arrays e escala pela `head_scale`.
            //
            // Para HEAD=8 (Standard): um único `_mm256_loadu_ps` carrega 8 floats consecutivos, e a
            // soma horizontal gera um escalar, substituindo 8 loads + 8 adds sequenciais.
            // Para HEAD=4 (Nano): usamos `_mm_loadu_ps` (128-bit) + `_mm_hadd_ps` × 2.
            // Modelos customizados usarão o Fallback escalar automático.
            for i in 0..num_frames {
                let head_ptr = self.array1.head_outputs.as_ptr();
                let head1_sum = if HEAD == 8 {
                    unsafe {
                        let v = core::arch::x86_64::_mm256_loadu_ps(head_ptr.add(i * HEAD));
                        // Horizontal sum: [a b c d e f g h]
                        // hadd → [a+b c+d e+f g+h _ _ _ _] (128-bit lanes)
                        let h1 = core::arch::x86_64::_mm256_hadd_ps(v, v);
                        let h2 = core::arch::x86_64::_mm256_hadd_ps(h1, h1);
                        // Extrair lane 0 (a+b+c+d) e lane 4 (e+f+g+h)
                        let lo = core::arch::x86_64::_mm256_castps256_ps128(h2);
                        let hi = core::arch::x86_64::_mm256_extractf128_ps::<1>(h2);
                        let sum128 = core::arch::x86_64::_mm_add_ss(lo, hi);
                        core::arch::x86_64::_mm_cvtss_f32(sum128)
                    }
                } else if HEAD == 4 {
                    unsafe {
                        let v = core::arch::x86_64::_mm_loadu_ps(head_ptr.add(i * HEAD));
                        let h1 = core::arch::x86_64::_mm_hadd_ps(v, v);
                        let h2 = core::arch::x86_64::_mm_hadd_ps(h1, h1);
                        core::arch::x86_64::_mm_cvtss_f32(h2)
                    }
                } else {
                    let mut s = 0.0f32;
                    for j in 0..HEAD {
                        s += unsafe { *head_ptr.add(i * HEAD + j) };
                    }
                    s
                };

                // O head final do Array2 gera a amostra float. Somamos ao mix da Array1.
                let final_sum = head1_sum + self.array2.head_outputs[i]; // HEAD2=1

                // Escrevemos a tensão elétrica analógica reconstruída no array de áudio local.
                output[pos + i] = final_sum * self.head_scale;
            }
            pos += num_frames;
        }
    }

    /// Estabiliza o modelo processando silêncio (Zero Input) para aquecimento (Pre-warm).
    ///
    /// O dispatch AVX-512 vs AVX2 é feito via `SimdMathConfig::get().is_avx512` —
    /// leitura atômica Relaxed de um `LazyLock` inicializado no startup, sem chamar
    /// `is_x86_feature_detected!` a cada invocação (cold-path, mas consistente com
    /// o padrão de dispatch do restante do codebase).
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Prewarm estritamente otimizado para a arquitetura AVX-512.
    ///
    /// # Safety
    /// Exige processador suportado (AVX-512).
    #[target_feature(enable = "avx512f,avx512vl")]
    pub unsafe fn prewarm_avx512(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::simd::Avx512Math>() };
    }

    /// Prewarm estritamente otimizado para arquitetura AVX2.
    ///
    /// # Safety
    /// Exige processador x86-64-v3 (AVX2).
    pub unsafe fn prewarm_avx2(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::simd::Avx2Math>() };
    }

    #[inline(always)]
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        unsafe {
            self.array1
                .prewarm_internal::<M>(&layer_inputs_1, &condition);
        }
        let array1_outputs = &self.array1.array_outputs[0..CH];
        unsafe {
            self.array2
                .prewarm_internal::<M>(array1_outputs, &condition);
        }
    }
}

#[cfg(test)]
#[path = "wavenet_test.rs"]
mod wavenet_test;
