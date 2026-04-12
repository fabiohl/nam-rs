// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha CNN Causal Dinâmica para inferência WaveNet (Fallback).
//!
//! Permite carregamento de modelos com topologias (canais, dilatações, etc.) não cobertas
//! pelas restrições dos Const Generics. A alocação ocorre apenas na thread hospedeira
//! (durante construtor) permitindo zero-allocation e RT-safety no caminho DSP, trocando
//! unroll do compilador (estático) por iterações dinâmicas de matriz em SIMD.

#![allow(clippy::needless_range_loop)]

use crate::math::simd::SimdMathConfig;
use crate::models::wavenet::WaveNetLayerState;

/// Estrutura para convolução causal 1D com dimensões limitadas em runtime.
#[derive(Clone)]
pub struct Conv1dDyn {
    /// Pesos da convolução arranjados contiguamente.
    pub weights: Vec<f32>,
    /// Vetor de bias somado aos canais de saída.
    pub bias: Vec<f32>,
    /// Flag indicando se o bias deve ser aplicado.
    pub do_bias: bool,
    /// Fator de dilatação da camada para acesso casual na fita temporal.
    pub dilation: usize,
    /// Quantidade de canais de entrada.
    pub in_ch: usize,
    /// Quantidade de canais de saída.
    pub out_ch: usize,
    /// Tamanho físico do kernel (fator de retardo causais no buffer).
    pub kernel: usize,
}

impl Conv1dDyn {
    /// Processa um frame temporal aplicando a convolução sobre o histórico em buffer livre de alocação.
    ///
    /// # Safety
    /// Depende da instância estrita de `SimdMathConfig` referenciar uma SIMD suportada.
    pub unsafe fn process_frame(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        math: &SimdMathConfig,
    ) {
        for out_c in 0..self.out_ch {
            let mut sum = if self.do_bias { self.bias[out_c] } else { 0.0 };

            for k in 0..self.kernel {
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let frame_idx = (buffer_start as isize) + offset;

                let in_slice_start = (frame_idx as usize) * self.in_ch;
                let in_slice = &layer_buffer[in_slice_start..in_slice_start + self.in_ch];

                let weight_slice_start = (out_c * self.kernel + k) * self.in_ch;
                let weight_slice =
                    &self.weights[weight_slice_start..weight_slice_start + self.in_ch];

                unsafe {
                    sum += (math.dot_product)(in_slice, weight_slice);
                }
            }

            block[out_c] = sum;
        }
    }
}

/// Camada Dense (fully-connected / projeção linear 1×1) avaliada dinamicamente.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Matriz densa de pesos `[Output][Input]`.
    pub weights: Vec<f32>,
    /// Bias unificado somado a saída.
    pub bias: Vec<f32>,
    /// Flag ativando bias.
    pub do_bias: bool,
    /// Dimensão da entrada (Input Size).
    pub in_size: usize,
    /// Dimensão projetada final (Output Size).
    pub out_size: usize,
}

impl DenseLayerDyn {
    /// Acumula as predições no vetor `output`. Utilizado pelo pipeline interno de blend `input_mixin`.
    ///
    /// # Safety
    /// Depende do `SimdMathConfig` ser válido nativamente.
    pub unsafe fn process_acc(&self, input: &[f32], output: &mut [f32], math: &SimdMathConfig) {
        for out_c in 0..self.out_size {
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let sum = unsafe { (math.dot_product)(input, weight_slice) };

            if self.do_bias {
                output[out_c] += sum + self.bias[out_c];
            } else {
                output[out_c] += sum;
            }
        }
    }

    /// Processa o dot product substituindo a memória em `output`.
    ///
    /// # Safety
    /// Requer `SimdMathConfig` validamente instanciado.
    pub unsafe fn process(&self, input: &[f32], output: &mut [f32], math: &SimdMathConfig) {
        for out_c in 0..self.out_size {
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let sum = unsafe { (math.dot_product)(input, weight_slice) };

            if self.do_bias {
                output[out_c] = sum + self.bias[out_c];
            } else {
                output[out_c] = sum;
            }
        }
    }
}

/// O elemento atomizado de conexão na malha, que interliga as equações diferenciais convolutivas.
#[derive(Clone)]
pub struct WaveNetLayerDyn {
    /// Núcleo convolutivo casual com histórico.
    pub conv1d: Conv1dDyn,
    /// Misturador acoplante de input local (residuum).
    pub input_mixin: DenseLayerDyn,
    /// Transformador 1x1 associado ao output/head da inferência final.
    pub one_by_one: DenseLayerDyn,
    /// Tamanho fixo do número de canais desta arquitetura.
    pub ch: usize,
}

impl WaveNetLayerDyn {
    /// Engata as equações de passagem, transferindo estados de Buffer Circular de camada em camada.
    ///
    /// # Safety
    /// Requer instâncias estritas do buffer interno.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process(
        &self,
        condition: &[f32],
        head_input: &mut [f32],
        output: &mut [f32],
        layer_buffer: &[f32],
        buffer_start: usize,
        block: &mut [f32],
        math: &SimdMathConfig,
    ) {
        for v in block.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            self.conv1d
                .process_frame(layer_buffer, block, buffer_start, math);
            self.input_mixin.process_acc(condition, block, math);

            (math.tanh_slice)(block);

            for j in 0..self.ch {
                head_input[j] += block[j];
            }

            self.one_by_one.process(block, output, math);
        }

        let lb_start = buffer_start * self.ch;
        for j in 0..self.ch {
            output[j] += layer_buffer[lb_start + j];
        }
    }
}

/// Representa a topologia vertical inteira de um galho WaveNet dinâmico, suportando múltiplas dilatações seq.
pub struct WaveNetLayerArrayDyn {
    /// Lista empilhada de camadas com suas respectivas dilatações fixadas na RAM em loading.
    pub layers: Vec<WaveNetLayerDyn>,
    /// Registro espelho local de fita de retardo. Mantido para as passagens lock-free circulares.
    pub states: Vec<WaveNetLayerState>,
    /// Redimensionador denso do canal de entrada.
    pub rechannel: DenseLayerDyn,
    /// Redimensionador denso para a malha da cabeça de soma paramétrica.
    pub head_rechannel: DenseLayerDyn,
    /// Acumulador temporário das cadeias sequenciais.
    pub array_outputs: std::vec::Vec<f32>,
    /// Acumulador das ativações projetadas pela malha Head.
    pub head_accum: std::vec::Vec<f32>,
    /// Projeções finais da cabeça em andamento (somatório de projeções de múltiplas camadas).
    pub head_outputs: std::vec::Vec<f32>,
    /// Buffer de estado auxiliar reutilizado para inibir heap allocation nas threads de RT.
    pub block_buffer: std::vec::Vec<f32>,
    /// Tamanho analítico global de latência causal desta cascata. (Usado no prewarm).
    pub receptive_field_size: usize,
    /// Eixo transversal de Canais base (`C`).
    pub ch: usize,
    /// Redução projetada somatória.
    pub head: usize,
}

impl WaveNetLayerArrayDyn {
    /// Realiza inferência síncrona não recursiva entre todas as camadas do Array. Mutação sem alocação em heap.
    ///
    /// # Safety
    /// Depende nativamente das matrizes preenchidas via alocador estrito no C++ Fallback parser (Loader CLI).
    pub unsafe fn process(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        math: &SimdMathConfig,
    ) {
        let ch = self.ch;
        let head = self.head;

        let states_ptr = self.states.as_mut_ptr();

        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + ch],
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    layer.process(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut self.array_outputs[0..ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..ch],
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut next_state.layer_buffer[next_start..next_start + ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..ch],
                        math,
                    );
                }

                current_state.advance_frames(1, ch);
            }

            self.head_rechannel.process(
                &self.head_accum[0..ch],
                &mut self.head_outputs[0..head],
                math,
            );
        }
    }

    /// Executa um aquecimento transiente espelhando buffer em todo o pre-gap RT causal.
    pub fn prewarm(&mut self, layer_inputs: &[f32], condition: &[f32], math: &SimdMathConfig) {
        let ch = self.ch;
        let head = self.head;
        let states_ptr = self.states.as_mut_ptr();

        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + ch],
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);
                current_state.copy_buffer(ch);

                if i == last_layer {
                    layer.process(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut self.array_outputs[0..ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..ch],
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut next_state.layer_buffer[next_start..next_start + ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..ch],
                        math,
                    );
                }
            }

            self.head_rechannel.process(
                &self.head_accum[0..ch],
                &mut self.head_outputs[0..head],
                math,
            );
        }
    }
}

/// Invólucro Dinâmico final. Comporta Arrays interconectados conforme a abstração de Inferência `NamModel`.
pub struct WaveNetDynModel {
    /// O galho Primário com maior parte do campo causal da WaveNet.
    pub array1: WaveNetLayerArrayDyn,
    /// Galho Secundário, redutor mono causal.
    pub array2: WaveNetLayerArrayDyn,
    /// Ajuste de volume master computado pré-linearização.
    pub head_scale: f32,
    /// Carga total de frames que este modelo assimila antes da saída confiável.
    pub receptive_field_size: usize,
    /// Dimensões da convergência interna final do head.
    pub head: usize,
}

impl WaveNetDynModel {
    /// Loop matriz causal para preenchimento bloco de áudio contíguo (via Inversão SIMD).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let math = &crate::math::simd::SimdMathConfig::current();
        let num_frames = input.len();

        for i in 0..num_frames {
            let sample = input[i];
            let condition = [sample];
            let layer_inputs_1 = [sample];

            unsafe {
                self.array1.process(&layer_inputs_1, &condition, math);

                let array1_outputs = &self.array1.array_outputs[..];
                self.array2.process(array1_outputs, &condition, math);
            }

            let mut final_sum = 0.0f32;
            for j in 0..self.head {
                final_sum += self.array1.head_outputs[j];
            }
            final_sum += self.array2.head_outputs[0];
            output[i] = final_sum * self.head_scale;
        }
    }

    /// Realiza `Prewarm` lock-free inicial. Evita instabilidade analítica dos buffers transientes.
    pub fn prewarm(&mut self) {
        let math = &crate::math::simd::SimdMathConfig::current();
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        self.array1.prewarm(&layer_inputs_1, &condition, math);
        let array1_outputs = &self.array1.array_outputs[..];
        self.array2.prewarm(array1_outputs, &condition, math);
    }
}
