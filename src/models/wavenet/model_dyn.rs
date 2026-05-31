// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Modelo WaveNet dinâmico (fallback para topologias não-cobertas por Const Generics).

use super::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext};
use super::conv1d_dyn::Conv1dDyn;
use crate::math::common::{AlignedVec, SimdMath};
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

/// Camada Dense 1x1 com dimensões dinâmicas.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Pesos da matriz [OUT][IN].
    pub weights: AlignedVec<u16>,
    /// Bias [OUT].
    pub bias: AlignedVec<f32>,
    /// Flag de aplicação de bias.
    pub do_bias: bool,
    /// Dimensão da entrada.
    pub in_size: usize,
    /// Dimensão da saída.
    pub out_size: usize,
}

impl DenseLayerDyn {
    /// Processa a camada fundindo com o acumulador de saída.
    ///
    /// # Safety
    ///
    /// Depende da validade dos ponteiros de entrada/saída e do alinhamento SIMD.
    #[inline(always)]
    /// # Safety
    /// `output` deve ter tamanho pelo menos `num_frames * self.out_size`.
    pub unsafe fn process_acc_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::fused_add_gemm_batch(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Processa a camada fundindo com a soma do residual.
    ///
    /// # Safety
    ///
    /// O chamador deve garantir que os buffers residual e output tenham tamanhos compatíveis.
    #[inline(always)]
    pub unsafe fn process_residual_batch<M: SimdMath>(
        &self,
        input: &[f32],
        residual: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::fused_gemm_residual_batch(
                input,
                &self.weights,
                &self.bias,
                residual,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Processa a camada substituindo o output.
    ///
    /// # Safety
    /// `output` deve ter tamanho pelo menos `num_frames * self.out_size`.
    /// Depende da validade dos buffers de entrada e saída para num_frames.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::gemv_overwrite_batch(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Processa a camada usando BF16.
    ///
    /// # Safety
    /// `output` deve ter tamanho pelo menos `num_frames * self.out_size`.
    /// Requer que `M::IS_BF16` seja true e que os buffers de entrada/saída sejam válidos.
    #[inline(always)]
    pub unsafe fn process_block_bf16<M: SimdMath>(
        &self,
        input: &[u16],
        output: &mut [f32],
        num_frames: usize,
    ) {
        unsafe {
            M::gemv_overwrite_batch_bf16(
                input,
                &self.weights,
                &self.bias,
                output,
                num_frames,
                self.do_bias,
            );
        }
    }

    /// Projeção fundida para um único frame.
    ///
    /// # Safety
    ///
    /// Depende da validade dos buffers de frame único.
    #[inline(always)]
    pub unsafe fn process_fused<M: SimdMath>(&self, in_frame: &[f32], out_frame: &mut [f32]) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }
}

/// Camada WaveNet com dimensões dinâmicas.
#[derive(Clone)]
pub struct WaveNetLayerDyn {
    /// Núcleo convolutivo casual.
    pub conv1d: Conv1dDyn,
    /// Misturador de entrada local (residuum).
    pub input_mixin: DenseLayerDyn,
    /// Transformador 1x1 associado ao output final.
    pub one_by_one: DenseLayerDyn,
    /// Quantidade de canais base.
    pub ch: usize,
    /// Ativa o mecanismo de Gated Activation.
    pub gated: bool,
}

impl WaveNetLayerDyn {
    /// Executa o processamento interno de uma camada WaveNet com Tiling Dual-Frame.
    /// # Safety
    /// `ctx.block` deve ser grande o suficiente para conter `num_frames * out_ch` amostras.
    /// Orquestrador Interno da Camada WaveNet.
    /// Esta função é o 'maestro' que coordena todos os passos matemáticos
    /// necessários para processar uma única camada da rede neural.
    pub unsafe fn process_block_internal<M: SimdMath>(&self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            condition_bf16,
            head_input,
            output,
            layer_buffer,
            layer_buffer_bf16,
            buffer_start,
            block,
            num_frames,
            mut output_bf16,
        } = ctx;
        let ch = self.ch;
        let out_ch = self.conv1d.out_ch;

        // --- Buffer Temporário na Stack ---
        // Usamos um buffer alinhado diretamente na memória de execução (pilha).
        // Isso evita alocações lentas e garante que o processamento seja
        // determinístico e ultra-rápido para áudio em tempo real.
        #[repr(align(64))]
        struct AlignedMixinBuffer([f32; 4096]);
        let mut mixin_out = AlignedMixinBuffer([0.0f32; 4096]);

        let mixin_len = num_frames * ch;
        debug_assert!(
            mixin_len <= 4096,
            "mixin_len overflow: {} (max 4096)",
            mixin_len
        );
        let mixin_out_slice = &mut mixin_out.0[..mixin_len];

        unsafe {
            // Decidimos entre o caminho BF16 (mais rápido) ou F32 (padrão)
            if M::IS_BF16 {
                // 1. Mixin (Preparação):
                // Processamos as condições externas (como gain/tone) em lote.
                self.input_mixin.process_block_bf16::<M>(
                    condition_bf16,
                    mixin_out_slice,
                    num_frames,
                );

                // 2. Conv1D (O Núcleo):
                // Aplicamos a convolução dilatada que 'ouve' o passado.
                let mut i = 0;
                let active_block = &mut block[..num_frames * out_ch];
                let mut chunks = active_block.chunks_exact_mut(2 * out_ch);
                for chunk in chunks.by_ref() {
                    let (out_f0, out_f1) = chunk.split_at_mut(out_ch);
                    let mix_idx = i * ch;
                    let m_f0 = &mixin_out_slice[mix_idx..mix_idx + ch];
                    let m_f1 = &mixin_out_slice[mix_idx + ch..mix_idx + 2 * ch];

                    self.conv1d.process_dual_frame_bf16::<M>(
                        layer_buffer_bf16,
                        out_f0,
                        out_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        Some(m_f0),
                        Some(m_f1),
                    );
                    i += 2;
                }
                let rem = chunks.into_remainder();
                if !rem.is_empty() {
                    let mix_idx = i * ch;
                    let m = &mixin_out_slice[mix_idx..mix_idx + ch];
                    self.conv1d.process_single_frame_bf16::<M>(
                        layer_buffer_bf16,
                        rem,
                        buffer_start + i,
                        Some(m),
                    );
                }
            } else {
                // Caminho padrão F32 (Idêntico ao acima, mas com precisão total).
                // 1. Mixin
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);

                // 2. Conv1D
                let mut i = 0;
                let active_block = &mut block[..num_frames * out_ch];
                let mut chunks = active_block.chunks_exact_mut(2 * out_ch);
                for chunk in chunks.by_ref() {
                    let (out_f0, out_f1) = chunk.split_at_mut(out_ch);
                    let mix_idx = i * ch;
                    let m_f0 = &mixin_out_slice[mix_idx..mix_idx + ch];
                    let m_f1 = &mixin_out_slice[mix_idx + ch..mix_idx + 2 * ch];

                    self.conv1d.process_dual_frame::<M>(
                        layer_buffer,
                        out_f0,
                        out_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        Some(m_f0),
                        Some(m_f1),
                    );
                    i += 2;
                }
                let rem = chunks.into_remainder();
                if !rem.is_empty() {
                    let mix_idx = i * ch;
                    let m = &mixin_out_slice[mix_idx..mix_idx + ch];
                    self.conv1d.process_single_frame::<M>(
                        layer_buffer,
                        rem,
                        buffer_start + i,
                        Some(m),
                    );
                }
            }

            // 3. Ativação (Não-Linearidade):
            // Aplicamos funções como Tanh ou Gated para dar o 'caráter' do som.
            if self.gated {
                // Gated Activation: Funciona como uma porta que abre e fecha seletivamente.
                M::gated_activation_and_accumulate_block(
                    head_input,
                    &mut block[..num_frames * 2 * ch],
                    ch,
                );

                // [T1.2] Otimização: Re-alinhamos os dados para que o próximo passo (GEMM)
                // seja processado em um único bloco contínuo de memória.
                for i in 1..num_frames {
                    block.copy_within(i * 2 * ch..i * 2 * ch + ch, i * ch);
                }
            } else {
                // Tanh: Ativação clássica que 'achata' o sinal para manter a estabilidade.
                M::tanh_and_accumulate_block(head_input, &mut block[..num_frames * ch]);
            }

            // 4. Residual + 1x1 (A Mistura Final):
            // Somamos o som original (residual) ao que acabamos de processar.
            // Isso permite que a rede aprenda transformações complexas sem perder a base.
            let lb_offset = buffer_start * ch;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * ch);

            self.one_by_one.process_residual_batch::<M>(
                &block[..num_frames * ch],
                residual_slice,
                output,
                num_frames,
            );

            // 5. Conversão BF16 Final:
            // Se estivermos usando o modo rápido, limpamos os dados para a próxima camada.
            if let (true, Some(bf16_out)) = (M::IS_BF16, output_bf16.as_mut()) {
                M::f32_to_bf16(output, bf16_out);
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
    pub unsafe fn process<M: crate::math::common::SimdMath>(
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
    pub fn prewarm<M: crate::math::common::SimdMath>(
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
    unsafe fn process_internal_generic<M: crate::math::common::SimdMath>(
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
                        debug_assert!(current_state.buffer_start >= offset, "backfill underflow: bs={}, off={}", current_state.buffer_start, offset);
                        let Some(dst_start) = current_state.buffer_start.checked_sub(offset) else {
                            log::error!("backfill underflow: bs={}, off={}", current_state.buffer_start, offset);
                            continue;
                        };
                        let dst_idx = dst_start * ch;
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
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Variante clonada e puramente otimizada para SIMD `M` do processamento da rede inteira.
    ///
    /// # Safety
    /// Deve ser invocado apenas via macro `dispatch_simd!`.
    unsafe fn process_internal<M: crate::math::common::SimdMath>(
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
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// # Safety
    /// Deve ser invocado apenas via macro `dispatch_simd!`.
    unsafe fn prewarm_internal<M: crate::math::common::SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];
        self.array1.prewarm::<M>(&layer_inputs_1, &condition);
        let array1_outputs = &self.array1.array_outputs[0..self.array1.ch];
        self.array2.prewarm::<M>(array1_outputs, &condition);
    }
}
