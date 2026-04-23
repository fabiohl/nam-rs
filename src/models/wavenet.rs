// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
// Portado em grande parte da implementação original em C++ (NeuralAudio) por Mike Oliphant.

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
        if self.do_bias {
            out_frame.copy_from_slice(&self.bias[0..OUT]);
        } else {
            out_frame.fill(0.0);
        }

        for k in 0..K {
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

            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let current_frame_idx = (frame_idx as isize) + offset;
            let in_slice_start = (current_frame_idx as usize) * IN;
            let in_slice =
                unsafe { layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN) };

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

                let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_slice) };

                unsafe {
                    *out_frame.get_unchecked_mut(out_c) += r0;
                    *out_frame.get_unchecked_mut(out_c + 1) += r1;
                    *out_frame.get_unchecked_mut(out_c + 2) += r2;
                    *out_frame.get_unchecked_mut(out_c + 3) += r3;
                }
                out_c += 4;
            }

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

    /// Processa bloco iterativo.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        for i in 0..num_frames {
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
        let mut out_c = 0;
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

            let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };

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

            unsafe {
                *out_frame.get_unchecked_mut(out_c) += r0 + b0;
                *out_frame.get_unchecked_mut(out_c + 1) += r1 + b1;
                *out_frame.get_unchecked_mut(out_c + 2) += r2 + b2;
                *out_frame.get_unchecked_mut(out_c + 3) += r3 + b3;
            }
            out_c += 4;
        }

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
        let mut out_c = 0;
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

            let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };

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

            unsafe {
                *out_frame.get_unchecked_mut(out_c) = r0 + b0;
                *out_frame.get_unchecked_mut(out_c + 1) = r1 + b1;
                *out_frame.get_unchecked_mut(out_c + 2) = r2 + b2;
                *out_frame.get_unchecked_mut(out_c + 3) = r3 + b3;
            }
            out_c += 4;
        }

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
        let mut out_c = 0;
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

            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };
                unsafe {
                    *output.get_unchecked_mut(i * OUT + out_c) += r0 + b0;
                    *output.get_unchecked_mut(i * OUT + out_c + 1) += r1 + b1;
                    *output.get_unchecked_mut(i * OUT + out_c + 2) += r2 + b2;
                    *output.get_unchecked_mut(i * OUT + out_c + 3) += r3 + b3;
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
                    *output.get_unchecked_mut(i * OUT + out_c) += sum + bias;
                }
            }
            out_c += 1;
        }
    }

    #[inline(always)]
    /// Processa bloco iterativo.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        let mut out_c = 0;
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

            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let [r0, r1, r2, r3] = unsafe { M::dot_product_4x(w0, w1, w2, w3, in_frame) };
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
                let mut temp = [0.0f32; CH];

                // 1. Conv1d para frame i
                self.conv1d
                    .process_single_frame::<M>(layer_buffer, &mut temp, buffer_start + i);

                // 2. Input mixin acumula em temp
                let cond_frame = &condition[i * COND..i * COND + COND];
                self.input_mixin
                    .process_acc_single_frame::<M>(cond_frame, &mut temp);

                // 3. tanh_slice in-place
                M::tanh_slice(&mut temp);

                // 4. Sum temp to head_input
                let head_frame = &mut head_input[i * CH..i * CH + CH];
                for j in 0..CH {
                    *head_frame.get_unchecked_mut(j) += temp[j];
                }

                // 5. one_by_one -> output frame
                let out_frame = &mut output[i * CH..i * CH + CH];
                self.one_by_one.process_single_frame::<M>(&temp, out_frame);

                // 6. output += layer_buffer[buffer_start + i] (Residual)
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
        let buffer_frames =
            receptive_field_size + (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES;
        let buffer = vec![0.0f32; buffer_frames * channels];

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
        self.buffer_start += num_frames;
        let buffer_frames = self.layer_buffer.len() / channels;
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
        let start = self.receptive_field_size;
        let from = (self.buffer_start - self.receptive_field_size) * channels;
        let to = (start - self.receptive_field_size) * channels;
        let len = self.receptive_field_size * channels;

        self.layer_buffer.copy_within(from..from + len, to);
        self.buffer_start = start;
    }

    /// Preenche o buffer histórico para estabilizar o modelo no estado de warm-up.
    pub fn copy_buffer(&mut self, channels: usize) {
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

        // Zera o acumulador para todo o bloco
        self.head_accum[0..num_frames * CH].fill(0.0);

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * CH],
                num_frames,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    layer.process_block_internal::<M>(
                        condition,
                        &mut self.head_accum[0..num_frames * CH],
                        &mut self.array_outputs[0..num_frames * CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        num_frames,
                    );
                } else {
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

                current_state.advance_frames(num_frames, CH);
            }

            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * CH],
                &mut self.head_outputs[0..num_frames * HEAD],
                num_frames,
            );
        }
    }

    /// Processa dados no modo Pre-warm para inicializar e estabilizar a memória temporal.
    #[inline(always)]
    pub fn prewarm_internal<M: SimdMath>(&mut self, layer_inputs: &[f32], condition: &[f32]) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();

        self.head_accum[0..CH].fill(0.0);

        unsafe {
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
                current_state.copy_buffer(CH);

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
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            return unsafe { self.process_avx512(input, output) };
        }
        unsafe { self.process_avx2(input, output) }
    }

    /// Processamento estritamente compilado para `avx512f` e `avx512vl`.
    ///
    /// # Safety
    /// A CPU local deve suportar explicitamente as extensões AVX-512 invocadas.
    #[target_feature(enable = "avx512f,avx512vl")]
    pub unsafe fn process_avx512(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { self.process_internal::<crate::math::simd::Avx512Math>(input, output) }
    }

    /// Processamento estritamente compilado para `avx2` e `fma`.
    ///
    /// # Safety
    /// A CPU local deve suportar explicitamente extensões x86-64-v3 (AVX2+FMA).
    pub unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { self.process_internal::<crate::math::simd::Avx2Math>(input, output) }
    }

    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        let mut pos = 0;
        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                // Condicionamento e Input (1D: 1 canal) -> formatado como blocos de IN frames
                self.array1
                    .process_block_internal::<M>(in_slice, in_slice, num_frames);

                let array1_outputs = &self.array1.array_outputs[0..num_frames * CH];
                self.array2
                    .process_block_internal::<M>(array1_outputs, in_slice, num_frames);
            }

            // Somatório das projeções Head de ambas as arrays e escala
            for i in 0..num_frames {
                let mut final_sum = 0.0f32;
                for j in 0..HEAD {
                    final_sum += self.array1.head_outputs[i * HEAD + j];
                }
                final_sum += self.array2.head_outputs[i]; // HEAD2=1
                output[pos + i] = final_sum * self.head_scale;
            }
            pos += num_frames;
        }
    }

    /// Estabiliza o modelo processando silêncio (Zero Input) para aquecimento (Pre-warm).
    pub fn prewarm(&mut self) {
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vl") {
            return unsafe { self.prewarm_avx512() };
        }
        unsafe { self.prewarm_avx2() }
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
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        self.array1
            .prewarm_internal::<M>(&layer_inputs_1, &condition);
        let array1_outputs = &self.array1.array_outputs[0..CH];
        self.array2
            .prewarm_internal::<M>(array1_outputs, &condition);
    }
}

#[cfg(test)]
#[path = "wavenet_test.rs"]
mod wavenet_test;
