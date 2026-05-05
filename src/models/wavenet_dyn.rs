// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha CNN Causal Dinâmica para inferência WaveNet (Fallback).
//!
//! Permite carregamento de modelos com topologias (canais, dilatações, etc.) não cobertas
//! pelas restrições dos Const Generics. A alocação ocorre apenas na thread hospedeira
//! (durante construtor) permitindo zero-allocation e RT-safety no caminho DSP, trocando
//! unroll do compilador (estático) por iterações dinâmicas de matriz em SIMD.

//! Malha CNN Causal Dinâmica para inferência WaveNet (Fallback).

use crate::models::wavenet::WaveNetLayerState;

/// Estrutura para convolução causal 1D com dimensões limitadas em runtime.
#[derive(Clone)]
pub struct Conv1dDyn {
    /// Pesos da convolução arranjados contiguamente.
    pub weights: Vec<u16>,
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
    /// Estratégia de prefetch pré-calculada (Eliminação de Branch).
    pub prefetch_fn: crate::math::simd::PrefetchFn,
}

impl Conv1dDyn {
    /// Processa um bloco temporal aplicando a convolução sobre o histórico em buffer livre de alocação.
    ///
    /// [TA2] Fused Conv1D+Mixin: Inicializa acumuladores com bias + mixin opcional.
    /// [TE1] Channel-First Tiling: Inverte loops para manter acumuladores SIMD quentes.
    /// [TE2] Prefetch 2-stage: Hint agressivo para dilatações >= 128.
    ///
    /// # Safety
    /// Depende da instância estrita de `SimdMathConfig` referenciar uma SIMD suportada.
    pub unsafe fn process_block<M: crate::math::simd::SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        unsafe {
            self.process_block_internal::<M>(layer_buffer, block, buffer_start, num_frames, None);
        }
    }

    /// Processa um bloco de amostras com mixin.
    ///
    /// # Safety
    /// O chamador deve garantir que `layer_buffer`, `block` e `mixin` possuam o tamanho
    /// correto para o número de frames e canais.
    pub unsafe fn process_block_with_mixin<M: crate::math::simd::SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: &[f32],
    ) {
        unsafe {
            self.process_block_internal::<M>(
                layer_buffer,
                block,
                buffer_start,
                num_frames,
                Some(mixin),
            );
        }
    }

    /// Processa o bloco completo da camada, incluindo mixin, ativação e acumulação de head.
    ///
    /// # Safety
    /// O chamador deve garantir que todos os buffers de estado e o `block` de entrada/saída
    /// estejam corretamente alinhados e possuam o tamanho exigido pela topologia.
    #[inline(always)]
    unsafe fn process_block_internal<M: crate::math::simd::SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        // [TE1] Inversão de Loop: Channel-First Tiling.
        // Itera sobre frames, então blocos de canais, então taps.
        // Isso mantém os acumuladores nos registros SIMD para todos os taps de um frame.
        let num_blocks = self.out_ch / 4;

        for i in 0..num_frames {
            let out_frame_start = i * self.out_ch;
            let current_frame_idx = buffer_start + i;

            // [TE1] Tap Pointers: Pre-calculamos os ponteiros para cada "tap" da convolução.
            // Evita o re-calculo de dilatação/offsets dentro do loop de blocos de canais.
            let mut tap_ptrs = [core::ptr::null::<f32>(); 8];
            let k_limit = self.kernel.min(8);
            for (k, tap_ptr) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let in_slice_start = ((current_frame_idx as isize) + offset) as usize * self.in_ch;
                unsafe {
                    *tap_ptr = layer_buffer.as_ptr().add(in_slice_start);

                    // Prefetch via estratégia pré-calculada (Branchless)
                    (self.prefetch_fn)(
                        *tap_ptr,
                        self.dilation * self.in_ch,
                        k,
                        self.kernel,
                        self.dilation,
                    );
                }
            }

            // Processamento dos blocos de canais
            for b in 0..num_blocks {
                let out_c = b * 4;
                let mut r0;
                let mut r1;
                let mut r2;
                let mut r3;

                unsafe {
                    // [TA2] Inicialização com Bias + Mixin
                    if let Some(m) = mixin {
                        let mix_idx = i * self.out_ch + out_c;
                        if self.do_bias {
                            r0 = *self.bias.get_unchecked(out_c) + *m.get_unchecked(mix_idx);
                            r1 =
                                *self.bias.get_unchecked(out_c + 1) + *m.get_unchecked(mix_idx + 1);
                            r2 =
                                *self.bias.get_unchecked(out_c + 2) + *m.get_unchecked(mix_idx + 2);
                            r3 =
                                *self.bias.get_unchecked(out_c + 3) + *m.get_unchecked(mix_idx + 3);
                        } else {
                            r0 = *m.get_unchecked(mix_idx);
                            r1 = *m.get_unchecked(mix_idx + 1);
                            r2 = *m.get_unchecked(mix_idx + 2);
                            r3 = *m.get_unchecked(mix_idx + 3);
                        }
                    } else if self.do_bias {
                        r0 = *self.bias.get_unchecked(out_c);
                        r1 = *self.bias.get_unchecked(out_c + 1);
                        r2 = *self.bias.get_unchecked(out_c + 2);
                        r3 = *self.bias.get_unchecked(out_c + 3);
                    } else {
                        r0 = 0.0;
                        r1 = 0.0;
                        r2 = 0.0;
                        r3 = 0.0;
                    }

                    for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                        let w_start = (b * self.kernel + k) * self.in_ch * 4;
                        let w_slice: &[[u16; 4]] = {
                            let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                            core::slice::from_raw_parts(ptr, self.in_ch)
                        };

                        let [t0, t1, t2, t3] = if !tap_ptr.is_null() {
                            let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                            M::dot_product_4x_interleaved(w_slice, in_slice)
                        } else {
                            let offset = (self.dilation as isize)
                                * ((k as isize) + 1 - (self.kernel as isize));
                            let in_slice_start =
                                ((current_frame_idx as isize) + offset) as usize * self.in_ch;
                            let in_slice = layer_buffer
                                .get_unchecked(in_slice_start..in_slice_start + self.in_ch);
                            M::dot_product_4x_interleaved(w_slice, in_slice)
                        };
                        r0 += t0;
                        r1 += t1;
                        r2 += t2;
                        r3 += t3;
                    }

                    *block.get_unchecked_mut(out_frame_start + out_c) = r0;
                    *block.get_unchecked_mut(out_frame_start + out_c + 1) = r1;
                    *block.get_unchecked_mut(out_frame_start + out_c + 2) = r2;
                    *block.get_unchecked_mut(out_frame_start + out_c + 3) = r3;
                }
            }

            // Cauda de canais (Remainder)
            let mut out_c = num_blocks * 4;
            while out_c < self.out_ch {
                let mut r;
                unsafe {
                    if let Some(m) = mixin {
                        let mix_idx = i * self.out_ch + out_c;
                        r = if self.do_bias { self.bias[out_c] } else { 0.0 } + m[mix_idx];
                    } else {
                        r = if self.do_bias { self.bias[out_c] } else { 0.0 };
                    }

                    for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                        let r_tap = if !tap_ptr.is_null() {
                            let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                            let w_start = (out_c * self.kernel + k) * self.in_ch;
                            let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                            M::dot_product(in_slice, w)
                        } else {
                            let offset = (self.dilation as isize)
                                * ((k as isize) + 1 - (self.kernel as isize));
                            let in_slice_start =
                                ((current_frame_idx as isize) + offset) as usize * self.in_ch;
                            let in_slice = layer_buffer
                                .get_unchecked(in_slice_start..in_slice_start + self.in_ch);
                            let w_start = (out_c * self.kernel + k) * self.in_ch;
                            let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                            M::dot_product(in_slice, w)
                        };
                        r += r_tap;
                    }
                    *block.get_unchecked_mut(out_frame_start + out_c) = r;
                }
                out_c += 1;
            }
        }
    }
}

/// Camada Dense (fully-connected / projeção linear 1×1) avaliada dinamicamente.
#[derive(Clone)]
pub struct DenseLayerDyn {
    /// Matriz densa de pesos `[Output][Input]`.
    pub weights: Vec<u16>,
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
    /// ACUMULAÇÃO DENSA (Projeção Linear 1x1):
    /// Processa a multiplicação de matriz (entrada x pesos) e ACUMULA o resultado no vetor `output`.
    /// [TA3] Otimização: Chama o kernel Batch GEMM para maximizar reuso de pesos.
    ///
    /// # Safety
    /// Depende do `SimdMathConfig` estar validamente instanciado para a arquitetura alvo.
    pub unsafe fn process_acc_block<M: crate::math::simd::SimdMath>(
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

    /// Executa a projeção 1x1 fundida com a soma do residual: Y = X_res + Bias + W * Z.
    /// [TF3] Otimização: Elimina a necessidade de cópia prévia do residual para o output.
    ///
    /// # Safety
    /// O chamador deve garantir tamanhos compatíveis e validade dos buffers.
    #[inline(always)]
    pub unsafe fn process_residual_batch<M: crate::math::simd::SimdMath>(
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

    /// Executa a projeção fundida (W*in + bias) somando ao buffer de saída (Residual Fusion).
    ///
    /// # Safety
    /// O chamador deve garantir que `in_frame` e `out_frame` tenham tamanhos compatíveis com `in_size` e `out_size`.
    #[inline(always)]
    pub unsafe fn process_fused<M: crate::math::simd::SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// ACUMULAÇÃO STRIDED (Salto de Memória):
    /// Similar ao `process_acc_block`, mas suporta `strides` (saltos).
    /// Útil quando os dados de entrada ou saída não estão perfeitamente contíguos
    /// (ex: processando sub-blocos ou canais intercalados).
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e ponteiros de buffer com espaço suficiente para os strides.
    pub unsafe fn process_acc_block_strided<M: crate::math::simd::SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
    ) {
        debug_assert!(
            self.out_size <= 1024,
            "DenseLayerDyn::process_acc_block_strided: out_size ({}) excede o buffer stack (1024)",
            self.out_size,
        );
        let mut tmp = [0.0f32; 1024];
        let tmp_slice = &mut tmp[..self.out_size];

        for i in 0..num_frames {
            let in_slice =
                unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
            let out_base = i * out_stride;

            // Limpa o buffer temporário
            tmp_slice.fill(0.0);

            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, tmp_slice, self.do_bias);
            }

            // Scatter (Acúmulo)
            for (j, &val) in tmp_slice.iter().enumerate() {
                unsafe {
                    *output.get_unchecked_mut(out_base + j) += val;
                }
            }
        }
    }

    /// PROCESSO DENSO STRIDED (Atribuição Direta):
    /// Realiza a transformação linear com saltos (strides), mas SOBRESCREVE o buffer de saída.
    /// Diferente de `process_acc_block_strided` (que usa `+=`), aqui usamos `=`.
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e ponteiros com acesso seguro conforme os strides.
    pub unsafe fn process_block_strided<M: crate::math::simd::SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
    ) {
        debug_assert!(
            self.out_size <= 1024,
            "DenseLayerDyn::process_block_strided: out_size ({}) excede o buffer stack (1024)",
            self.out_size,
        );
        let mut tmp = [0.0f32; 1024];
        let tmp_slice = &mut tmp[..self.out_size];

        for i in 0..num_frames {
            let in_slice =
                unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
            let out_base = i * out_stride;

            tmp_slice.fill(0.0);

            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, tmp_slice, self.do_bias);
            }

            // Scatter (Atribuição Direta)
            for (j, &val) in tmp_slice.iter().enumerate() {
                unsafe {
                    *output.get_unchecked_mut(out_base + j) = val;
                }
            }
        }
    }

    /// PROCESSO DENSO (Contíguo):
    /// Esta é a versão mais simples e rápida da projeção linear, usada quando os buffers
    /// de entrada e saída estão perfeitamente contíguos em memória.
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e buffers com tamanho compatível com `in_size` e `out_size`.
    pub unsafe fn process_block<M: crate::math::simd::SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            let in_slice = unsafe { input.get_unchecked(i * self.in_size..(i + 1) * self.in_size) };
            let out_slice =
                unsafe { output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size) };

            out_slice.fill(0.0);
            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
            }
        }
    }

    /// ACUMULAÇÃO STRIDED BF16:
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e buffers com tamanho compatível.
    pub unsafe fn process_acc_block_strided_bf16<M: crate::math::simd::SimdMath>(
        &self,
        input: &[u16],
        output: &mut [f32],
        num_frames: usize,
        in_stride: usize,
        out_stride: usize,
    ) {
        debug_assert!(
            self.out_size <= 1024,
            "DenseLayerDyn::process_acc_block_strided_bf16: out_size ({}) excede o buffer stack (1024)",
            self.out_size,
        );
        let mut tmp = [0.0f32; 1024];
        let tmp_slice = &mut tmp[..self.out_size];

        for i in 0..num_frames {
            let in_slice =
                unsafe { input.get_unchecked(i * in_stride..i * in_stride + self.in_size) };
            let out_base = i * out_stride;

            tmp_slice.fill(0.0);

            unsafe {
                M::gemv_overwrite_bf16(
                    in_slice,
                    &self.weights,
                    &self.bias,
                    tmp_slice,
                    self.do_bias,
                );
            }

            // Scatter (Acúmulo)
            for (j, &val) in tmp_slice.iter().enumerate() {
                unsafe {
                    *output.get_unchecked_mut(out_base + j) += val;
                }
            }
        }
    }

    /// PROCESSO DENSO BF16:
    ///
    /// # Safety
    /// Requer `SimdMathConfig` válido e buffers com tamanho compatível.
    pub unsafe fn process_block_bf16<M: crate::math::simd::SimdMath>(
        &self,
        input: &[u16],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            let in_slice = unsafe { input.get_unchecked(i * self.in_size..(i + 1) * self.in_size) };
            let out_slice =
                unsafe { output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size) };

            out_slice.fill(0.0);
            unsafe {
                M::gemv_overwrite_bf16(
                    in_slice,
                    &self.weights,
                    &self.bias,
                    out_slice,
                    self.do_bias,
                );
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

/// Contexto de processamento para WaveNet dinâmico.
pub struct WavenetDynProcessContext<'a> {
    /// Buffer de condicionamento.
    pub condition: &'a [f32],
    /// Buffer de condicionamento pré-convertido para BF16.
    pub condition_bf16: &'a [u16],
    /// Acumulador para a cabeça de saída.
    pub head_input: &'a mut [f32],
    /// Buffer de saída da camada.
    pub output: &'a mut [f32],
    /// Buffer circular de histórico.
    pub layer_buffer: &'a [f32],
    /// Índice inicial no buffer circular.
    pub buffer_start: usize,
    /// Buffer temporário de bloco (reutilizado).
    pub block: &'a mut [f32],
    /// Número de frames a processar.
    pub num_frames: usize,
}

impl WaveNetLayerDyn {
    /// MOTOR DE PASSAGEM DA CAMADA (Fluxo WaveNet):
    /// Aqui as equações de uma camada WaveNet são executadas sequencialmente.
    /// O fluxo segue o padrão: Convolução -> Condicionamento -> Ativação (Gated) -> Skip -> Residual.
    ///
    /// # Safety
    /// Requer instâncias estritas do buffer interno e `block` com tamanho
    /// `ch` (não-gated) ou `2*ch` (gated).
    pub unsafe fn process_block_internal<M: crate::math::simd::SimdMath>(
        &self,
        ctx: WavenetDynProcessContext<'_>,
    ) {
        let WavenetDynProcessContext {
            condition,
            condition_bf16,
            head_input,
            output,
            layer_buffer,
            buffer_start,
            block,
            num_frames,
        } = ctx;
        let ch = self.ch;

        unsafe {
            // [PASSO 1: Condicionamento (Input Mixin)]
            // Buffer temporário na stack para o Mixin.
            // 4096 f32 = 16KB. Cobre até 64 canais com 64 frames.
            let mut mixin_out = [0.0f32; 4096];
            let mixin_len = num_frames * self.conv1d.out_ch;

            // Verificação de segurança para o buffer de stack.
            // Se exceder (raro em NAM), processamos em modo degradado ou panic em debug.
            debug_assert!(
                mixin_len <= 4096,
                "Mixin buffer overflow: {} > 4096",
                mixin_len
            );
            let mixin_out_slice = &mut mixin_out[..mixin_len.min(4096)];

            // Zeramos apenas o necessário.
            for v in mixin_out_slice.iter_mut() {
                *v = 0.0;
            }
            if M::IS_BF16 {
                self.input_mixin.process_block_bf16::<M>(
                    condition_bf16,
                    mixin_out_slice,
                    num_frames,
                );
            } else {
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);
            }

            // [FASE 1: Linear - Conv1D + Mixin]
            // [TA2] Convolução fundida com o Mixin.
            self.conv1d.process_block_with_mixin::<M>(
                layer_buffer,
                block,
                buffer_start,
                num_frames,
                mixin_out_slice,
            );

            // [FASE 2 & 3: Ativação e Head Update]
            if self.gated {
                // [TF2] Fused Gated Activation SIMD.
                M::gated_activation_and_accumulate_block(
                    head_input,
                    &mut block[..num_frames * 2 * self.ch],
                    self.ch,
                );
            } else {
                // [TE3] Fusão Tanh + Head Accumulate em passagem única.
                M::tanh_and_accumulate_block(head_input, &mut block[..num_frames * ch]);
            }

            // [FASE 3: Saída - 1x1 Residual]
            // [TF3] Otimização: Projeção 1x1 fundida com a soma do residual em lote.
            let in_stride = if self.gated { 2 * self.ch } else { self.ch };
            let lb_offset = buffer_start * ch;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * ch);

            if self.gated || in_stride != ch {
                // Se o stride for diferente (gated), processamos per-frame.
                for i in 0..num_frames {
                    let out_frame = output.get_unchecked_mut(i * ch..i * ch + ch);
                    let res_frame = &residual_slice[i * ch..i * ch + ch];
                    out_frame.copy_from_slice(res_frame);
                    let block_frame = &block[i * in_stride..i * in_stride + ch];
                    self.one_by_one.process_fused::<M>(block_frame, out_frame);
                }
            } else {
                // Caso comum: Elimina cópia via kernel fundido.
                self.one_by_one.process_residual_batch::<M>(
                    block,
                    residual_slice,
                    output,
                    num_frames,
                );
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
    /// Cache do último condicionamento f32.
    pub last_condition: Vec<f32>,
    /// Cache do último condicionamento BF16.
    pub last_condition_bf16: Vec<u16>,
    /// Flag de inicialização do cache.
    pub condition_init: bool,
}

impl WaveNetLayerArrayDyn {
    /// ORQUESTRADOR DE INFERÊNCIA (Cascade Process):
    /// Realiza a inferência síncrona de todas as camadas do Array em cascata.
    /// Este é o ponto de entrada principal para processar um bloco de áudio.
    ///
    /// # Safety
    /// Depende da integridade das matrizes carregadas e dos estados de buffer circular.
    pub unsafe fn process<M: crate::math::simd::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
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

        // 1) RESET DO ACUMULADOR DE CABEÇA (Skip Connections):
        // Antes de começar, zeramos o buffer onde todas as camadas vão somar seus outputs parciais.
        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        // [PASSO 2: Lazy BF16 Conversion]
        if M::IS_BF16 {
            let mut changed = !self.condition_init;
            if !changed && condition != self.last_condition.as_slice() {
                changed = true;
            }

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

            // 2) RECHANNEL (Entrada -> Residual):
            // Projeta o áudio de entrada (1 canal) para a dimensão residual (`ch`).
            // O resultado é escrito DIRETAMENTE na fita de retardo (buffer circular) da 1ª camada.
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * ch],
                num_frames,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            // 3) CASCATEAMENTO DE CAMADAS:
            // Itera sobre todas as camadas da rede.
            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    // ÚLTIMA CAMADA: O output residual final vai para `array_outputs`.
                    layer.process_block_internal::<M>(WavenetDynProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..num_frames * ch],
                        output: &mut self.array_outputs[0..num_frames * ch],
                        layer_buffer: &current_state.layer_buffer,
                        buffer_start: current_state.buffer_start,
                        block: &mut self.block_buffer[0..num_frames * block_size],
                        num_frames,
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    // CONEXÃO ENTRE CAMADAS:
                    // A saída residual da camada 'i' é injetada DIRETAMENTE no buffer da camada 'i+1'.
                    // Isso economiza cópias de memória e mantém os dados quentes no cache.
                    layer.process_block_internal::<M>(WavenetDynProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..num_frames * ch],
                        output: &mut next_state.layer_buffer
                            [next_start..next_start + num_frames * ch],
                        layer_buffer: &current_state.layer_buffer,
                        buffer_start: current_state.buffer_start,
                        block: &mut self.block_buffer[0..num_frames * block_size],
                        num_frames,
                    });
                }

                // 4) ATUALIZAÇÃO DOS PONTEIROS CIRCULARES:
                // Move o índice 'buffer_start' para a frente, "envelhecendo" as amostras no buffer causal.
                current_state.advance_frames(num_frames, ch);
            }

            // 5) HEAD RECHANNEL (Skip Sum -> Output):
            // Pega o somatório de todas as skip connections e realiza a projeção final 1x1.
            // O resultado (`head_outputs`) é o que será usado para prever o próximo valor do áudio.
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * ch],
                &mut self.head_outputs[0..num_frames * head],
                num_frames,
            );
        }
    }

    /// AQUECIMENTO DE ESTADO (Pre-warm):
    /// Preenche os buffers circulares com valores iniciais para evitar artefatos de áudio (cliques/pops)
    /// no início da reprodução. Essencial para redes com histórico (convoluções causais).
    pub fn prewarm<M: crate::math::simd::SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
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

        // Inicializa acumuladores.
        for v in self.head_accum.iter_mut() {
            *v = 0.0;
        }

        // [PASSO 2: Lazy BF16 Conversion]
        if M::IS_BF16 {
            let mut changed = !self.condition_init;
            if !changed && condition != self.last_condition.as_slice() {
                changed = true;
            }

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

            // Projeção inicial do frame de aquecimento.
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + ch],
                1, // O aquecimento processa frame a frame (1 frame).
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            let block_size = self.block_size;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                // POPULANDO O HISTÓRICO:
                // Preenche todo o histórico (Receptive Field) com o valor recém-processado.
                // Com VirtualRingBuffer, como buffer_start >= N, podemos recuar linearmente com segurança.
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

                if i == last_layer {
                    layer.process_block_internal::<M>(WavenetDynProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..ch],
                        output: &mut self.array_outputs[0..ch],
                        layer_buffer: &current_state.layer_buffer,
                        buffer_start: current_state.buffer_start,
                        block: &mut self.block_buffer[0..block_size],
                        num_frames: 1,
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * ch;

                    layer.process_block_internal::<M>(WavenetDynProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..ch],
                        output: &mut next_state.layer_buffer[next_start..next_start + ch],
                        layer_buffer: &current_state.layer_buffer,
                        buffer_start: current_state.buffer_start,
                        block: &mut self.block_buffer[0..block_size],
                        num_frames: 1,
                    });
                }
            }

            // Projeção final do frame de aquecimento.
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..ch],
                &mut self.head_outputs[0..head],
                1,
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
    /// Processa o bloco de áudio na matriz causal.
    ///
    /// **Para Cientistas e Devs:** Todo o processamento passa obrigatoriamente pela macro `dispatch_simd!`.
    /// Essa tática de multiversioning anula custos de ramificação em tempo de execução (branches condicionais)
    /// para selecionar qual conjunto de intruções de CPU usar (AVX2, AVX-512). O compilador gera clones desta
    /// classe de processamento especializados para cada chipset.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Variante clonada e puramente otimizada para SIMD `M` do processamento da rede inteira.
    unsafe fn process_internal<M: crate::math::simd::SimdMath>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) {
        let total_frames = input.len();

        let mut pos = 0;
        // Processa em janelas para limitar cache misses (chunking).
        // Os buffers internos circulares dependem destas iterações menores e densas.
        while pos < total_frames {
            let num_frames =
                (total_frames - pos).min(crate::models::wavenet::WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                // array1 engloba a porção majoritária do campo receptivo da WaveNet.
                self.array1.process::<M>(in_slice, in_slice, num_frames);

                let array1_outputs = &self.array1.array_outputs[0..num_frames * self.array1.ch];
                // array2 recebe o processado do array1 e o sinal original no canal de condição.
                // É utilizado para condensar o resultado ou aplicar finalizações paramétricas.
                self.array2
                    .process::<M>(array1_outputs, in_slice, num_frames);
            }

            // Mixagem Final (Master Blend):
            // Combina a predição condensada de head do Array1 com o Head do Array2,
            // e os redimensiona/escala para o intervalo de áudio -1.0 a +1.0.
            #[allow(clippy::needless_range_loop)]
            for i in 0..num_frames {
                let mut final_sum = 0.0f32;
                for j in 0..self.head {
                    final_sum += self.array1.head_outputs[i * self.head + j];
                }
                final_sum += self.array2.head_outputs[i];
                output[pos + i] = final_sum * self.head_scale;
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
    /// Call this via `dispatch_simd!` macro only.
    unsafe fn prewarm_internal<M: crate::math::simd::SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        // Roda o estado com uma amostra preenchida de 0.0 para preencher os buffers internos
        // e impedir ruídos espúrios ao trocar modelos (os pipelines de convolução precisam
        // de amostras causais passadas).
        self.array1.prewarm::<M>(&layer_inputs_1, &condition);
        let array1_outputs = &self.array1.array_outputs[..];
        self.array2.prewarm::<M>(array1_outputs, &condition);
    }
}

// =============================================================================
// Testes Unitários
// =============================================================================

#[cfg(test)]
#[path = "wavenet_dyn_test.rs"]
mod wavenet_dyn_test;
