// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha CNN Causal Dinâmica para inferência WaveNet (Fallback).
//!
//! Permite carregamento de modelos com topologias (canais, dilatações, etc.) não cobertas
//! pelas restrições dos Const Generics. A alocação ocorre apenas na thread hospedeira
//! (durante construtor) permitindo zero-allocation e RT-safety no caminho DSP, trocando
//! unroll do compilador (estático) por iterações dinâmicas de matriz em SIMD.

use crate::math::simd::AlignedVec;
use crate::models::wavenet_common::{
    DenseLayerDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerDyn, WaveNetLayerState,
    WavenetProcessContext,
};
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

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
    pub array_outputs: AlignedVec<f32>,
    /// Acumulador das ativações projetadas pela malha Head.
    pub head_accum: AlignedVec<f32>,
    /// Projeções finais da cabeça em andamento (somatório de projeções de múltiplas camadas).
    pub head_outputs: AlignedVec<f32>,
    /// Buffer de estado auxiliar reutilizado para inibir heap allocation nas threads de RT.
    pub block_buffer: AlignedVec<f32>,
    /// Tamanho efetivo do `block_buffer`. Igual a `ch` ou `2*ch` conforme gated.
    pub block_size: usize,
    /// Tamanho analítico global de latência causal desta cascata.
    pub receptive_field_size: usize,
    /// Eixo transversal de Canais base (`C`).
    pub ch: usize,
    /// Redução projetada somatória.
    pub head: usize,
    /// Cache do último condicionamento f32.
    pub last_condition: AlignedVec<f32>,
    /// Cache do último condicionamento BF16.
    pub last_condition_bf16: AlignedVec<u16>,
    /// Flag de inicialização do cache.
    pub condition_init: bool,
}

impl WaveNetLayerArrayDyn {
    /// ORQUESTRADOR DE INFERÊNCIA (Cascade Process):
    /// Realiza a inferência síncrona de todas as camadas do Array em cascata.
    ///
    /// # Safety
    /// Depende da integridade das matrizes carregadas e dos estados de buffer circular.
    pub unsafe fn process<M: crate::math::simd::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
    ) {
        unsafe {
            self.process_internal_generic::<M>(layer_inputs, condition, num_frames, false);
        }
    }

    /// AQUECIMENTO DE ESTADO (Pre-warm).
    pub fn prewarm<M: crate::math::simd::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
    ) {
        unsafe {
            self.process_internal_generic::<M>(layer_inputs, condition, 1, true);
        }
    }

    /// Implementação genérica que unifica o processamento normal e o pre-warm.
    /// [TA5.5] Redução de duplicidade lógica em ~70%.
    #[inline(always)]
    unsafe fn process_internal_generic<M: crate::math::simd::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
        prewarm_mode: bool,
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let ch = self.ch;
        let head = self.head;
        let states_ptr = self.states.as_mut_ptr();

        // 1) RESET DO ACUMULADOR DE CABEÇA
        self.head_accum[..num_frames * ch].fill(0.0);

        // 2) Lazy BF16 Conversion
        if M::IS_BF16 {
            let changed =
                prewarm_mode || !self.condition_init || condition != &self.last_condition[..];
            if changed {
                unsafe {
                    M::f32_to_bf16(condition, &mut self.last_condition_bf16);
                }
                self.last_condition.copy_from_slice(condition);
                self.condition_init = true;
            }
        }

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;

            // 3) RECHANNEL (Entrada -> Residual)
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * ch],
                num_frames,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;
            let block_size = self.block_size;

            // 4) CASCATEAMENTO DE CAMADAS
            for i in 0..num_layers {
                let layer = &self.layers[i];
                let current_state = &mut *states_ptr.add(i);

                // [T2.2] Software Prefetch do próximo estado na cascata (L1).
                if i + 1 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 1) as *const i8);
                }
                if i + 2 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 2) as *const i8);
                }

                // [PASSO 4.1: Pre-fill Ring Buffer (Backwards)]
                // Se estivermos em modo pre-warm, replicamos a entrada atual para todo o passado.
                if prewarm_mode {
                    let start_idx = current_state.buffer_start * ch;
                    for offset in 1..=current_state.receptive_field_size {
                        let dst_idx = (current_state.buffer_start - offset) * ch;
                        for j in 0..ch {
                            current_state.layer_buffer[dst_idx + j] =
                                current_state.layer_buffer[start_idx + j];
                            current_state.layer_buffer_bf16[dst_idx + j] =
                                current_state.layer_buffer_bf16[start_idx + j];
                        }
                    }
                }

                let ctx = WavenetProcessContext {
                    condition,
                    condition_bf16: &self.last_condition_bf16,
                    head_input: &mut self.head_accum[0..num_frames * ch],
                    output: if i == last_layer {
                        &mut self.array_outputs[0..num_frames * ch]
                    } else {
                        let next_state = &mut *states_ptr.add(i + 1);
                        let next_start = next_state.buffer_start * ch;
                        &mut next_state.layer_buffer[next_start..next_start + num_frames * ch]
                    },
                    output_bf16: None,
                    layer_buffer: &current_state.layer_buffer,
                    layer_buffer_bf16: &current_state.layer_buffer_bf16,
                    buffer_start: current_state.buffer_start,
                    block: &mut self.block_buffer[0..num_frames * block_size],
                    num_frames,
                };

                layer.process_block_internal::<M>(ctx);

                // No modo pre-warm não avançamos o ponteiro circular (estabilização estática).
                if !prewarm_mode {
                    current_state.advance_frames(num_frames, ch);
                }
            }

            // 5) HEAD RECHANNEL (Skip Sum -> Output)
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * ch],
                &mut self.head_outputs[0..num_frames * head],
                num_frames,
            );
        }
    }
}

/// Invólucro Dinâmico final. Comporta Arrays interconectados.
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
    /// Processa o bloco de áudio na matriz causal.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Variante clonada e puramente otimizada para SIMD `M` do processamento da rede inteira.
    ///
    /// # Safety
    /// Deve ser invocado apenas via macro `dispatch_simd!`.
    unsafe fn process_internal<M: crate::math::simd::SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) {
        let total_frames = input.len();
        let mut pos = 0;
        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                self.array1.process::<M>(in_slice, in_slice, num_frames);
                let array1_outputs = &self.array1.array_outputs[0..num_frames * self.array1.ch];
                self.array2
                    .process::<M>(array1_outputs, in_slice, num_frames);
            }

            unsafe {
                M::batch_wavenet_head_sum_dyn(
                    &self.array1.head_outputs[0..num_frames * self.head],
                    &self.array2.head_outputs[0..num_frames],
                    &mut output[pos..pos + num_frames],
                    self.head,
                    self.head_scale,
                );
            }
            pos += num_frames;
        }
    }

    /// Realiza o `Prewarm` inicial para estabilizar os buffers.
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// # Safety
    /// Deve ser invocado apenas via macro `dispatch_simd!`.
    unsafe fn prewarm_internal<M: crate::math::simd::SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];
        self.array1.prewarm::<M>(&layer_inputs_1, &condition);
        let array1_outputs = &self.array1.array_outputs[0..self.array1.ch];
        self.array2.prewarm::<M>(array1_outputs, &condition);
    }
}

#[cfg(test)]
#[path = "wavenet_dyn_test.rs"]
mod wavenet_dyn_test;
