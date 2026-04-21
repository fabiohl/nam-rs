// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
// Portado em grande parte da implementação original em C++ (NeuralAudio) por Mike Oliphant.

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
    /// ## Otimização: Software Prefetch Proativo
    ///
    /// Idêntico ao `Conv1d` estático: para dilatações grandes, emite `_mm_prefetch`
    /// para o próximo tap do kernel enquanto o tap atual é processado via FMA,
    /// eliminando L1 cache misses previsíveis (~5–10% ganho em layers dilatadas).
    ///
    /// # Safety
    /// Depende da instância estrita de `SimdMathConfig` referenciar uma SIMD suportada.
    /// Processa bloco iterativo.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        for out_c in 0..self.out_ch {
            for k in 0..self.kernel {
                if k + 1 < self.kernel {
                    let next_offset =
                        (self.dilation as isize) * ((k as isize) + 2 - (self.kernel as isize));
                    let next_frame_idx = (buffer_start as isize) + next_offset;
                    let next_addr = unsafe {
                        layer_buffer
                            .as_ptr()
                            .add((next_frame_idx as usize) * self.in_ch)
                    };
                    unsafe {
                        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
                            next_addr.cast::<i8>(),
                        );
                    }
                }

                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let base_frame_idx = (buffer_start as isize) + offset;
                let weight_slice_start = (out_c * self.kernel + k) * self.in_ch;
                let weight_slice = unsafe {
                    self.weights
                        .get_unchecked(weight_slice_start..weight_slice_start + self.in_ch)
                };

                if k == 0 {
                    let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;
                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c) =
                                bias + (math.dot_product)(in_slice, weight_slice);
                        }
                    }
                } else {
                    for i in 0..num_frames {
                        let frame_idx = base_frame_idx + (i as isize);
                        let in_slice_start = (frame_idx as usize) * self.in_ch;
                        let in_slice = unsafe {
                            layer_buffer.get_unchecked(in_slice_start..in_slice_start + self.in_ch)
                        };
                        unsafe {
                            *block.get_unchecked_mut(i * self.out_ch + out_c) +=
                                (math.dot_product)(in_slice, weight_slice);
                        }
                    }
                }
            }
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
    pub unsafe fn process_acc_block(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        for out_c in 0..self.out_size {
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame = unsafe {
                    input.get_unchecked(i * self.in_size..i * self.in_size + self.in_size)
                };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * self.out_size + out_c) += sum + bias;
                }
            }
        }
    }

    /// Processa o dot product substituindo a memória em `output`.
    ///
    /// # Safety
    /// Requer `SimdMathConfig` validamente instanciado.
    pub unsafe fn process_acc_block_strided(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
        math: &SimdMathConfig,
    ) {
        for out_c in 0..self.out_size {
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame =
                    unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * out_stride + out_c) += sum + bias;
                }
            }
        }
    }

    /// Processa bloco strided.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block_strided(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
        math: &SimdMathConfig,
    ) {
        for out_c in 0..self.out_size {
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame =
                    unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * out_stride + out_c) = sum + bias;
                }
            }
        }
    }

    /// Processa bloco iterativo.
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn process_block(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        for out_c in 0..self.out_size {
            let weight_slice =
                &self.weights[out_c * self.in_size..out_c * self.in_size + self.in_size];
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame = unsafe {
                    input.get_unchecked(i * self.in_size..i * self.in_size + self.in_size)
                };
                let sum = unsafe { (math.dot_product)(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * self.out_size + out_c) = sum + bias;
                }
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
    /// Ativa o mecanismo de Gated Activation: `tanh(z[0..ch]) ⊙ sigmoid(z[ch..2*ch])`.
    ///
    /// Quando `true`, o `conv1d` deve ter `out_ch = 2 * ch`.
    pub gated: bool,
}

impl WaveNetLayerDyn {
    /// Engata as equações de passagem, transferindo estados de Buffer Circular de camada em camada.
    ///
    /// Quando `gated == false` (padrão): aplica `tanh(x)` sobre `block[0..ch]`.
    /// Quando `gated == true`: aplica ativação gated `tanh(x[0..ch]) ⊙ sigmoid(x[ch..2*ch])`.
    ///   - O `conv1d` deve ter `out_ch = 2 * ch`; `input_mixin` acumula apenas nos primeiros `ch`.
    ///   - Resultado gated em `block[0..ch]` é somado ao head e passado ao one_by_one.
    ///
    /// # Safety
    /// Requer instâncias estritas do buffer interno e `block` com tamanho
    /// `ch` (não-gated) ou `2*ch` (gated).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_block_internal(
        &self,
        condition: &[f32],
        head_input: &mut [f32],
        output: &mut [f32],
        layer_buffer: &[f32],
        buffer_start: usize,
        block: &mut [f32],
        num_frames: usize,
        math: &SimdMathConfig,
    ) {
        let ch = self.ch;

        for v in block.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            self.conv1d
                .process_block(layer_buffer, block, buffer_start, num_frames, math);

            if self.gated {
                self.input_mixin.process_acc_block_strided(
                    condition,
                    block,
                    num_frames,
                    1,
                    2 * self.ch,
                    math,
                );

                for i in 0..num_frames {
                    let block_start = i * self.conv1d.out_ch;
                    let block_frame = &mut block[block_start..block_start + self.conv1d.out_ch];

                    let (z1, z2) = block_frame.split_at_mut(self.ch);
                    (math.tanh_slice)(z1);
                    (math.sigmoid_slice)(z2);

                    for j in 0..self.ch {
                        *z1.get_unchecked_mut(j) *= *z2.get_unchecked(j);
                    }
                }
            } else {
                self.input_mixin
                    .process_acc_block_strided(condition, block, num_frames, 1, self.ch, math);
                (math.tanh_slice)(&mut block[0..num_frames * self.ch]);
            }

            for i in 0..num_frames {
                let head_frame = &mut head_input[i * ch..i * ch + ch];
                let block_frame = &block[i * (if self.gated { 2 * ch } else { ch })
                    ..i * (if self.gated { 2 * ch } else { ch }) + ch];
                for j in 0..ch {
                    *head_frame.get_unchecked_mut(j) += *block_frame.get_unchecked(j);
                }
            }

            // in_stride depende do layout do block: 2*ch quando gated (slots tanh+sigmoid),
            // ch quando não-gated (apenas slot tanh). Usar stride errado faz o one_by_one
            // ler de offsets incorretos, produzindo saída silenciosamente corrompida.
            let in_stride = if self.gated { 2 * self.ch } else { self.ch };
            self.one_by_one
                .process_block_strided(block, output, num_frames, in_stride, self.ch, math);
        }

        unsafe {
            for i in 0..num_frames {
                let out_frame = output.get_unchecked_mut(i * ch..i * ch + ch);
                let lb_start = (buffer_start + i) * ch;
                for j in 0..ch {
                    *out_frame.get_unchecked_mut(j) += *layer_buffer.get_unchecked(lb_start + j);
                }
            }
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
    ///
    /// Tamanho: `ch` quando `gated == false`, ou `2 * ch` quando `gated == true`,
    /// para acomodar os dois slots (tanh + sigmoid) do mecanismo gated.
    pub block_buffer: std::vec::Vec<f32>,
    /// Tamanho efetivo do `block_buffer`. Igual a `ch` ou `2*ch` conforme gated.
    pub block_size: usize,
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
        debug_assert_eq!(
            self.layers.len(),
            self.states.len(),
            "WaveNetLayerArrayDyn: layers ({}) ≠ states ({})",
            self.layers.len(),
            self.states.len()
        );
        let ch = self.ch;
        let head = self.head;

        let states_ptr = self.states.as_mut_ptr();

        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process_block(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + ch],
                1, // prewarm is 1 frame
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut self.array_outputs[0..ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut next_state.layer_buffer[next_start..next_start + ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                        math,
                    );
                }

                current_state.advance_frames(1, ch);
            }

            self.head_rechannel.process_block(
                &self.head_accum[0..ch],
                &mut self.head_outputs[0..head],
                1,
                math,
            );
        }
    }

    /// Executa um aquecimento transiente espelhando buffer em todo o pre-gap RT causal.
    pub fn prewarm(&mut self, layer_inputs: &[f32], condition: &[f32], math: &SimdMathConfig) {
        debug_assert_eq!(
            self.layers.len(),
            self.states.len(),
            "WaveNetLayerArrayDyn: layers ({}) ≠ states ({})",
            self.layers.len(),
            self.states.len()
        );
        let ch = self.ch;
        let head = self.head;
        let states_ptr = self.states.as_mut_ptr();

        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process_block(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + ch],
                1, // prewarm is 1 frame
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);
                current_state.copy_buffer(ch);

                if i == last_layer {
                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut self.array_outputs[0..ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process_block_internal(
                        condition,
                        &mut self.head_accum[0..ch],
                        &mut next_state.layer_buffer[next_start..next_start + ch],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        &mut self.block_buffer[0..block_size],
                        1,
                        math,
                    );
                }
            }

            self.head_rechannel.process_block(
                &self.head_accum[0..ch],
                &mut self.head_outputs[0..head],
                1,
                math,
            );
        }
    }
}

/// Invólucro Dinâmico final. Comporta Arrays interconectados conforme a abstração de Inferência `NamModel`.
///
/// **Referência Científica:** van den Oord, A., et al. (2016). *"WaveNet: A Generative Model for Raw Audio."* DeepMind.
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
        let math = crate::math::simd::SimdMathConfig::get();
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
        let math = crate::math::simd::SimdMathConfig::get();
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        self.array1.prewarm(&layer_inputs_1, &condition, math);
        let array1_outputs = &self.array1.array_outputs[..];
        self.array2.prewarm(array1_outputs, &condition, math);
    }
}

// =============================================================================
// Testes Unitários
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::simd::SimdMathConfig;
    use crate::models::wavenet::WaveNetLayerState;

    /// Constrói um `Conv1dDyn` mínimo com `kernel=1`, `dilation=1`.
    ///
    /// - `in_ch`: canais de entrada
    /// - `out_ch`: canais de saída (2×ch quando gated)
    /// - `weight`: valor fixo para todos os pesos (facilita cálculo analítico)
    fn make_conv1d(in_ch: usize, out_ch: usize, weight: f32) -> Conv1dDyn {
        Conv1dDyn {
            weights: vec![weight; out_ch * in_ch], // kernel=1
            bias: vec![0.0; out_ch],
            do_bias: false,
            dilation: 1,
            in_ch,
            out_ch,
            kernel: 1,
        }
    }

    /// Constrói um `DenseLayerDyn` identidade (peso=0, bias=0, sem efeito).
    fn make_dense_zero(in_size: usize, out_size: usize) -> DenseLayerDyn {
        DenseLayerDyn {
            weights: vec![0.0; out_size * in_size],
            bias: vec![0.0; out_size],
            do_bias: false,
            in_size,
            out_size,
        }
    }

    /// Verifica que `WaveNetLayerDyn` com `gated=true` produz `tanh(conv) ⊙ sigmoid(conv)`.
    ///
    /// Configuração sintética (CH=1, kernel=1, dilation=1):
    /// - `conv1d` IN=1, OUT=2, peso=1.0 → out[0]=x, out[1]=x (ambos os slots recebem x)
    /// - `input_mixin` e `one_by_one` com pesos zero (sem contribuição externa)
    /// - `layer_buffer[buffer_start] = x = 0.7` → residual adicionado ao output
    ///
    /// Saída esperada em `head_input[0]`: `tanh(x) * sigmoid(x)`.
    #[test]
    fn test_gated_layer_dyn_process() {
        let ch = 1usize;
        let x = 0.7f32;

        // Montar layer_buffer: [x] na posição buffer_start (buffer_start=1, buffer_frames=2 → size=2)
        let buffer_start = 1usize;
        let layer_buffer = vec![0.0f32, x]; // índice 1 = x

        // Conv1d: IN=1, OUT=2 (gated), kernel=1, weight=1.0 → out[0]=x, out[1]=x
        let conv1d = make_conv1d(ch, 2 * ch, 1.0);

        let layer = WaveNetLayerDyn {
            conv1d,
            input_mixin: make_dense_zero(1, ch), // condition=&[0.0] → zero contrib
            one_by_one: make_dense_zero(ch, ch), // zero → output permanece 0 antes do residual
            ch,
            gated: true,
        };

        let condition = [0.0f32];
        let mut head_input = vec![0.0f32; ch];
        let mut output = vec![0.0f32; ch];
        let mut block = vec![0.0f32; 2 * ch];

        let math = SimdMathConfig::current();

        unsafe {
            layer.process_block_internal(
                &condition,
                &mut head_input,
                &mut output,
                &layer_buffer,
                buffer_start,
                &mut block,
                1,
                &math,
            );
        }

        // Esperado: tanh(x) * sigmoid(x) para cada canal
        let expected_activation = x.tanh() * (0.5 * (1.0 + (0.5 * x).tanh())); // sigmoid(x)
        // head_input deve acumular block[0..ch] após gated
        let eps = 1e-5f32;
        assert!(
            (head_input[0] - expected_activation).abs() < eps,
            "head_input[0] deveria ser tanh(x)*sigmoid(x)={}, obteve {}",
            expected_activation,
            head_input[0]
        );

        // output[0] = one_by_one(block[0..ch]=0) + layer_buffer[buffer_start*ch + 0] = 0 + x = x
        assert!(
            (output[0] - x).abs() < eps,
            "output[0] deveria ser residual x={}, obteve {}",
            x,
            output[0]
        );
    }

    /// Verifica que `gated=false` mantém o comportamento original: `tanh(conv + mixin)`.
    #[test]
    fn test_non_gated_layer_dyn_process() {
        let ch = 1usize;
        let x = 0.7f32;

        let buffer_start = 1usize;
        let layer_buffer = vec![0.0f32, x];

        // Conv1d: IN=1, OUT=1 (não-gated), weight=1.0 → out[0]=x
        let conv1d = make_conv1d(ch, ch, 1.0);

        let layer = WaveNetLayerDyn {
            conv1d,
            input_mixin: make_dense_zero(1, ch),
            one_by_one: make_dense_zero(ch, ch),
            ch,
            gated: false,
        };

        let condition = [0.0f32];
        let mut head_input = vec![0.0f32; ch];
        let mut output = vec![0.0f32; ch];
        let mut block = vec![0.0f32; ch];

        let math = SimdMathConfig::current();

        unsafe {
            layer.process_block_internal(
                &condition,
                &mut head_input,
                &mut output,
                &layer_buffer,
                buffer_start,
                &mut block,
                1,
                &math,
            );
        }

        let expected = x.tanh();
        let eps = 1e-5f32;
        assert!(
            (head_input[0] - expected).abs() < eps,
            "head_input[0] deveria ser tanh(x)={}, obteve {}",
            expected,
            head_input[0]
        );
    }

    /// Verifica que `WaveNetLayerState` e pool de buffers são corretamente mantidos
    /// ao construir um `WaveNetLayerArrayDyn` com `block_size = 2*ch` quando gated.
    #[test]
    fn test_wavenet_layer_array_dyn_block_size_gated() {
        let ch = 4usize;
        let block_size = 2 * ch;

        // Construir WaveNetLayerArrayDyn manualmente com block_size=2*ch
        let state = WaveNetLayerState::new(ch, 0, 0); // RF=0 apenas para alocação
        let conv1d = Conv1dDyn {
            weights: vec![0.0; 2 * ch * ch],
            bias: vec![0.0; 2 * ch],
            do_bias: false,
            dilation: 1,
            in_ch: ch,
            out_ch: 2 * ch,
            kernel: 1,
        };
        let layer = WaveNetLayerDyn {
            conv1d,
            input_mixin: make_dense_zero(1, ch),
            one_by_one: make_dense_zero(ch, ch),
            ch,
            gated: true,
        };

        let array = WaveNetLayerArrayDyn {
            layers: vec![layer],
            states: vec![state],
            rechannel: make_dense_zero(1, ch),
            head_rechannel: make_dense_zero(ch, 1),
            array_outputs: vec![0.0; ch],
            head_accum: vec![0.0; ch],
            head_outputs: vec![0.0; 1],
            block_buffer: vec![0.0; block_size],
            block_size,
            receptive_field_size: 0,
            ch,
            head: 1,
        };

        assert_eq!(
            array.block_buffer.len(),
            2 * ch,
            "block_buffer deve ter tamanho 2*ch para gated"
        );
        assert_eq!(array.block_size, 2 * ch);
    }
}
