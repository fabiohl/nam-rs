// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha CNN Causal Estática para inferência WaveNet (Design Orientado a Dados, SoA).
//!
//! Todas as estruturas utilizam `Const Generics` nas dimensões matemáticas e vetores pré-alocados
//! garantindo uma política de instanciamento estrito (Zero-Allocation durante processamento).
//! As loops dinâmicos resolvem cálculos em sequências FMA determinísticas via AVX2.

#![allow(clippy::needless_range_loop)]

use crate::math::simd::dot_product_avx2;

#[cfg(target_arch = "x86_64")]
use crate::math::fastmath::simd_tanh;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
    /// # Safety
    /// Depende nativamente do conjunto de instruções `AVX2` e `FMA`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn process_frame(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
    ) {
        for out_c in 0..OUT {
            let mut sum = if self.do_bias { self.bias[out_c] } else { 0.0 };

            for k in 0..K {
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
                let frame_idx = (buffer_start as isize) + offset;

                let in_slice_start = (frame_idx as usize) * IN;
                let in_slice = &layer_buffer[in_slice_start..in_slice_start + IN];

                let weight_slice_start = (out_c * K + k) * IN;
                let weight_slice = &self.weights[weight_slice_start..weight_slice_start + IN];

                unsafe {
                    sum += dot_product_avx2(in_slice, weight_slice);
                }
            }

            block[out_c] = sum;
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
    /// Processa o Dense acumulando com o estado corrente de output.
    ///
    /// # Safety
    /// Requer suporte dinâmico a AVX2 e FMA no Host.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn process_acc(&self, input: &[f32], output: &mut [f32]) {
        for out_c in 0..OUT {
            let weight_slice = &self.weights[out_c * IN..out_c * IN + IN];
            let sum = unsafe { dot_product_avx2(input, weight_slice) };

            if self.do_bias {
                output[out_c] += sum + self.bias[out_c];
            } else {
                output[out_c] += sum;
            }
        }
    }

    /// Processa o Dense sobrescrevendo o slice do output.
    ///
    /// # Safety
    /// Requer suporte dinâmico a AVX2 e FMA no Host.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn process(&self, input: &[f32], output: &mut [f32]) {
        for out_c in 0..OUT {
            let weight_slice = &self.weights[out_c * IN..out_c * IN + IN];
            let sum = unsafe { dot_product_avx2(input, weight_slice) };

            if self.do_bias {
                output[out_c] = sum + self.bias[out_c];
            } else {
                output[out_c] = sum;
            }
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
    /// Requer suporte dinâmico a AVX2 e FMA no Host.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn process(
        &self,
        condition: &[f32],
        head_input: &mut [f32],
        output: &mut [f32],
        layer_buffer: &[f32],
        buffer_start: usize,
    ) {
        let mut block = [0.0f32; CH];

        unsafe {
            self.conv1d
                .process_frame(layer_buffer, &mut block, buffer_start);
            self.input_mixin.process_acc(condition, &mut block);

            // Ativação Tanh usando Intrínsecos Vetorizados
            let mut i = 0;
            while i + 8 <= CH {
                let va = _mm256_loadu_ps(block.as_ptr().add(i));
                let vt = simd_tanh(va);
                _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);
                i += 8;
            }
            while i < CH {
                block[i] = block[i].tanh();
                i += 1;
            }

            // Sum block to head_input
            for j in 0..CH {
                head_input[j] += block[j];
            }

            self.one_by_one.process(&block, output);
        }

        // output += layer_buffer[buffer_start] (Residual connection)
        let lb_start = buffer_start * CH;
        for j in 0..CH {
            output[j] += layer_buffer[lb_start + j];
        }
    }
}

/// Gerencia a memória buffer de uma célula WaveNet.
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
    pub fn rewind_buffer(&mut self, channels: usize) {
        let start = self.receptive_field_size;
        let from = (self.buffer_start - self.receptive_field_size) * channels;
        let to = (start - self.receptive_field_size) * channels;
        let len = self.receptive_field_size * channels;

        self.layer_buffer.copy_within(from..from + len, to);
        self.buffer_start = start;
    }

    /// Trinca retroativa da Memória Receptiva no estado inicial de Warm-up de estado.
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

    /// Conexão transiente de pre-alocação zero-copy ao próximo vetor.
    pub array_outputs: std::vec::Vec<f32>,
    /// Memória alocada da projeção Linear global.
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
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn process(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        head_inputs: &mut [f32],
    ) {
        let states_ptr = self.states.as_mut_ptr();

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel
                .process(layer_inputs, &mut state_0.layer_buffer[start..start + CH]);

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    layer.process(
                        condition,
                        head_inputs,
                        &mut self.array_outputs[0..CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process(
                        condition,
                        head_inputs,
                        &mut next_state.layer_buffer[next_start..next_start + CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                    );
                }

                current_state.advance_frames(1, CH);
            }

            self.head_rechannel
                .process(head_inputs, &mut self.head_outputs[0..HEAD]);
        }
    }

    /// Invoca a transposição artificial do modelo em Pre-warm estabilizando memória temporal.
    pub fn prewarm(&mut self, layer_inputs: &[f32], condition: &[f32], head_inputs: &mut [f32]) {
        let states_ptr = self.states.as_mut_ptr();

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel
                .process(layer_inputs, &mut state_0.layer_buffer[start..start + CH]);

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);
                current_state.copy_buffer(CH);

                if i == last_layer {
                    layer.process(
                        condition,
                        head_inputs,
                        &mut self.array_outputs[0..CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process(
                        condition,
                        head_inputs,
                        &mut next_state.layer_buffer[next_start..next_start + CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                    );
                }
            }

            self.head_rechannel
                .process(head_inputs, &mut self.head_outputs[0..HEAD]);
        }
    }
}

/// Modelo Completo do WaveNet contendo Múltiplos Blocos Diatônicos (Arrays).
pub struct WaveNetModel<const CH: usize, const K: usize, const HEAD: usize> {
    /// Array interno 01: IN=1, COND=1, HasBias=False
    pub array1: WaveNetLayerArray<1, 1, CH, K, HEAD>,
    /// Array interno 02: IN=CH, COND=1, HasBias=True
    pub array2: WaveNetLayerArray<CH, 1, CH, K, HEAD>,
    /// Escala de compensação da voltagem final (Target Output Scale).
    pub head_scale: f32,
    /// Maior buffer circular requerido na raiz temporal do Kernel.
    pub receptive_field_size: usize,
}

impl<const CH: usize, const K: usize, const HEAD: usize> WaveNetModel<CH, K, HEAD> {
    /// Resolve o forward total e produz amostras de onda em zero alocação (DSP).
    #[cfg(target_arch = "x86_64")]
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let num_frames = input.len();

        for i in 0..num_frames {
            let sample = input[i];
            let condition = [sample];
            let layer_inputs_1 = [sample];
            let mut head_array = [0.0f32; HEAD];

            unsafe {
                self.array1
                    .process(&layer_inputs_1, &condition, &mut head_array);

                let array1_outputs = &self.array1.array_outputs[0..CH];
                self.array2
                    .process(array1_outputs, &condition, &mut head_array);
            }

            let mut final_sum = 0.0;
            for j in 0..HEAD {
                final_sum += self.array2.head_outputs[j];
            }
            output[i] = final_sum * self.head_scale;
        }
    }

    /// Estabiliza os transientes inicias causais por tempo de propagação (Zero Input).
    pub fn prewarm(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];
        let mut head_array = [0.0f32; HEAD];

        self.array1
            .prewarm(&layer_inputs_1, &condition, &mut head_array);
        let array1_outputs = &self.array1.array_outputs[0..CH];
        self.array2
            .prewarm(array1_outputs, &condition, &mut head_array);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
