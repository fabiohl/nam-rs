// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Componentes comuns e dinâmicos para arquiteturas WaveNet.
//!
//! Contém as estruturas fundamentais (Conv1D, Dense, Layer) que operam com
//! dimensões definidas em runtime, servindo como base para o modelo dinâmico
//! e futuros estágios da arquitetura A2.

use crate::dsp::vring::VirtualRingBuffer;
use crate::math::simd::{AlignedVec, SimdMath};

/// Máximo de frames a processar em um pulso do callback.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Padding temporal circular das memórias no framework de Ring Buffers.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;

/// Estrutura para convolução causal 1D com dimensões dinâmicas.
#[derive(Clone)]
pub struct Conv1dDyn {
    /// Pesos da convolução [OUT][KERNEL][IN] (quantizados u16).
    pub weights: AlignedVec<u16>,
    /// Vetor de bias [OUT].
    pub bias: AlignedVec<f32>,
    /// Flag indicando se o bias deve ser aplicado.
    pub do_bias: bool,
    /// Fator de dilatação temporal.
    pub dilation: usize,
    /// Quantidade de canais de entrada.
    pub in_ch: usize,
    /// Quantidade de canais de saída.
    pub out_ch: usize,
    /// Tamanho físico do kernel.
    pub kernel: usize,
    /// Estratégia de prefetch pré-calculada.
    pub prefetch_fn: crate::math::simd::PrefetchFn,
}

impl Conv1dDyn {
    /// Processa um bloco de amostras com mixin opcional.
    ///
    /// # Safety
    ///
    /// O chamador deve garantir que os buffers possuam o tamanho correto e
    /// que `M` seja compatível com a CPU atual.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch / 4;

        for i in 0..num_frames {
            let out_frame_start = i * self.out_ch;
            let current_frame_idx = buffer_start + i;

            // [TE1] Tap Pointers: Pre-calculamos os ponteiros para cada "tap" da convolução.
            let mut tap_ptrs = [core::ptr::null::<f32>(); 8];
            let k_limit = self.kernel.min(8);
            for (k, tap_ptr) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let in_slice_start = ((current_frame_idx as isize) + offset) as usize * self.in_ch;
                unsafe {
                    *tap_ptr = layer_buffer.as_ptr().add(in_slice_start);

                    (self.prefetch_fn)(
                        *tap_ptr,
                        self.dilation * self.in_ch,
                        k,
                        self.kernel,
                        self.dilation,
                    );
                }
            }

            for b in 0..num_blocks {
                let out_c = b * 4;
                let mut r0;
                let mut r1;
                let mut r2;
                let mut r3;

                unsafe {
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

                        let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                        let [t0, t1, t2, t3] = M::dot_product_4x_interleaved(w_slice, in_slice);
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

            // Remainder canais
            let mut out_c = num_blocks * 4;
            while out_c < self.out_ch {
                let mut r = if self.do_bias { self.bias[out_c] } else { 0.0 };
                if let Some(m) = mixin {
                    r += m[i * self.out_ch + out_c];
                }

                for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                    unsafe {
                        let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                        let w_start = (out_c * self.kernel + k) * self.in_ch;
                        let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                        r += M::dot_product(in_slice, w);
                    }
                }
                block[out_frame_start + out_c] = r;
                out_c += 1;
            }
        }
    }

    /// Processa um bloco de amostras usando BF16 na memória circular (layer_buffer).
    ///
    /// # Safety
    ///
    /// Buffers devem ser válidos e `M::IS_BF16` deve ser true.
    #[inline(always)]
    pub unsafe fn process_block_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch / 4;

        for i in 0..num_frames {
            let out_frame_start = i * self.out_ch;
            let current_frame_idx = buffer_start + i;

            // [TE1] Tap Pointers: Pre-calculamos os ponteiros para cada "tap" da convolução em BF16.
            let mut tap_ptrs = [core::ptr::null::<u16>(); 8];
            let k_limit = self.kernel.min(8);
            for (k, tap_ptr) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
                let in_slice_start = ((current_frame_idx as isize) + offset) as usize * self.in_ch;
                unsafe {
                    *tap_ptr = layer_buffer.as_ptr().add(in_slice_start);

                    // Prefetch continua operando em f32-ptr por simplicidade, cast seguro aqui.
                    (self.prefetch_fn)(
                        *tap_ptr as *const f32,
                        self.dilation * self.in_ch,
                        k,
                        self.kernel,
                        self.dilation,
                    );
                }
            }

            for b in 0..num_blocks {
                let out_c = b * 4;
                let mut r0;
                let mut r1;
                let mut r2;
                let mut r3;

                unsafe {
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

                        let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                        let [t0, t1, t2, t3] =
                            M::dot_product_4x_interleaved_bf16(w_slice, in_slice);
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

            // Remainder canais
            let mut out_c = num_blocks * 4;
            while out_c < self.out_ch {
                let mut r = if self.do_bias { self.bias[out_c] } else { 0.0 };
                if let Some(m) = mixin {
                    r += m[i * self.out_ch + out_c];
                }

                for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                    unsafe {
                        let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                        let w_start = (out_c * self.kernel + k) * self.in_ch;
                        let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                        r += M::dot_product_bf16(in_slice, w);
                    }
                }
                block[out_frame_start + out_c] = r;
                out_c += 1;
            }
        }
    }
}

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
    ///
    /// Depende da validade dos buffers de entrada e saída para num_frames.
    #[inline(always)]
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            unsafe {
                let in_slice = input.get_unchecked(i * self.in_size..(i + 1) * self.in_size);
                let out_slice =
                    output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size);
                out_slice.fill(0.0);
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
            }
        }
    }

    /// Processa a camada usando BF16.
    ///
    /// # Safety
    ///
    /// Requer que `M::IS_BF16` seja true e que os buffers de entrada/saída sejam válidos.
    #[inline(always)]
    pub unsafe fn process_block_bf16<M: SimdMath>(
        &self,
        input: &[u16],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            unsafe {
                let in_slice = input.get_unchecked(i * self.in_size..(i + 1) * self.in_size);
                let out_slice =
                    output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size);
                out_slice.fill(0.0);
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

/// Contexto de processamento para otimizar a passagem de parâmetros no hot-path da WaveNet.
/// Unifica as necessidades dos modelos estáticos (const generic) e dinâmicos.
pub struct WavenetProcessContext<'a> {
    /// Buffer de condicionamento (sidechain).
    pub condition: &'a [f32],
    /// Buffer de condicionamento pré-convertido em BF16.
    pub condition_bf16: &'a [u16],
    /// Acumulador Head (Skip-Connection).
    pub head_input: &'a mut [f32],
    /// Buffer de saída da camada (para a próxima camada ou output final).
    pub output: &'a mut [f32],
    /// Buffer de saída opcional em BF16 (para a próxima camada em CPUs BF16).
    pub output_bf16: Option<&'a mut [u16]>,
    /// Buffer circular da camada corrente (delay line).
    pub layer_buffer: &'a [f32],
    /// Buffer circular (fita de retardo) em BF16.
    pub layer_buffer_bf16: &'a [u16],
    /// Índice inicial no buffer circular.
    pub buffer_start: usize,
    /// Número de frames a processar.
    pub num_frames: usize,
    /// Buffer temporário na stack para cálculos intermediários.
    pub block: &'a mut [f32],
}

/// Gerencia a memória buffer de uma célula WaveNet.
///
/// Alinhamento de 64 bytes (uma cache line) garante que `buffer_start` e
/// `receptive_field_size` não compartilhem cache line com o estado da camada
/// adjacente ao iterar `states_ptr.add(i)` no hot-path do `process()`.
#[repr(align(64))]
#[derive(Clone)]
pub struct WaveNetLayerState {
    /// Buffer Circular Virtual (zero alocações em contexto DSP, eliminação de rewind).
    pub layer_buffer: VirtualRingBuffer<f32>,
    /// Buffer Circular Virtual em BF16 para processamento VNNI.
    pub layer_buffer_bf16: VirtualRingBuffer<u16>,
    /// Ponteiro numérico do frame atual (avança a cada frame processado).
    pub buffer_start: usize,
    /// Dimensão física do espaço vetorial receptivo (tamanho do histórico de dilatação).
    pub receptive_field_size: usize,
}

impl WaveNetLayerState {
    /// Construtor alocador estático do Estado (executar antes do Thread DSP).
    pub fn new(channels: usize, receptive_field_size: usize, alloc_num: usize) -> Self {
        // [PASSO 1: Cálculo do Tamanho do Buffer Temporal]
        // O buffer precisa acomodar o campo receptivo e o padding de blocos.
        // Arredondamento para página é feito internamente pelo VirtualRingBuffer.
        let min_buffer_frames =
            receptive_field_size + (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES;

        let buffer = VirtualRingBuffer::<f32>::new(min_buffer_frames * channels);
        let buffer_bf16 = VirtualRingBuffer::<u16>::new(min_buffer_frames * channels);

        let actual_buffer_frames = buffer.size() / channels;

        // [PASSO 2: Offset Inicial (Jittering Alocado)]
        // Posicionamos o ponteiro inicial na segunda metade do mapeamento virtual (offset N).
        // Isso permite olhar para trás (receptive field) sem cruzar o início do buffer virtual.
        let jitter = (alloc_num % LAYER_ARRAY_BUFFER_PADDING) + 1;
        let start = actual_buffer_frames * 2 - (WAVENET_MAX_NUM_FRAMES * jitter);

        Self {
            layer_buffer: buffer,
            layer_buffer_bf16: buffer_bf16,
            buffer_start: start,
            receptive_field_size,
        }
    }

    /// Executa um passo do ponteiro do Ring Buffer. Se chegar na margem, volta para o início.
    pub fn advance_frames(&mut self, num_frames: usize, channels: usize) {
        self.buffer_start += num_frames;
        let buffer_frames = self.layer_buffer.size() / channels;

        // [VIRTUAL RING BUFFER]
        // Se o próximo bloco de tamanho máximo (64) puder ultrapassar o limite do mapeamento 2N,
        // retrocedemos o ponteiro para a primeira metade (mantendo a paridade de endereço virtual).
        // Isso garante que [buffer_start .. buffer_start + 64] seja sempre um acesso seguro.
        if self.buffer_start + WAVENET_MAX_NUM_FRAMES > buffer_frames * 2 {
            self.buffer_start -= buffer_frames;
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
    /// Executa o processamento interno de uma camada WaveNet.
    ///
    /// # Safety
    ///
    /// O contexto deve conter referências válidas aos buffers do RT-thread.
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

        // Buffer temporário na stack para o Mixin (16KB) com alinhamento de 64 bytes.
        #[repr(align(64))]
        struct AlignedMixinBuffer([f32; 4096]);
        let mut mixin_out = AlignedMixinBuffer([0.0f32; 4096]);

        let mixin_len = num_frames * self.conv1d.out_ch;
        assert!(
            mixin_len <= 4096,
            "mixin_len overflow: {} (max 4096)",
            mixin_len
        );
        let mixin_out_slice = &mut mixin_out.0[..mixin_len];

        unsafe {
            if M::IS_BF16 {
                // [T25] Path BF16 Otimizado: Uso de memórias circulares BF16 e kernels interleaved BF16.
                self.input_mixin.process_block_bf16::<M>(
                    condition_bf16,
                    mixin_out_slice,
                    num_frames,
                );

                self.conv1d.process_block_bf16::<M>(
                    layer_buffer_bf16,
                    block,
                    buffer_start,
                    num_frames,
                    Some(mixin_out_slice),
                );
            } else {
                // Path f32 padrão.
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);

                self.conv1d.process_block::<M>(
                    layer_buffer,
                    block,
                    buffer_start,
                    num_frames,
                    Some(mixin_out_slice),
                );
            }

            if self.gated {
                M::gated_activation_and_accumulate_block(
                    head_input,
                    &mut block[..num_frames * 2 * ch],
                    ch,
                );
            } else {
                M::tanh_and_accumulate_block(head_input, &mut block[..num_frames * ch]);
            }

            let in_stride = if self.gated { 2 * ch } else { ch };
            let lb_offset = buffer_start * ch;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * ch);

            if self.gated || in_stride != ch {
                for i in 0..num_frames {
                    let out_f = output.get_unchecked_mut(i * ch..i * ch + ch);
                    out_f.copy_from_slice(&residual_slice[i * ch..i * ch + ch]);
                    self.one_by_one
                        .process_fused::<M>(&block[i * in_stride..i * in_stride + ch], out_f);
                }
            } else {
                self.one_by_one.process_residual_batch::<M>(
                    block,
                    residual_slice,
                    output,
                    num_frames,
                );
            }

            // 4. [T25] Fusão BF16: Conversão em lote se necessário para a próxima camada.
            if let (true, Some(bf16_out)) = (M::IS_BF16, output_bf16.as_mut()) {
                M::f32_to_bf16(output, bf16_out);
            }
        }
    }
}
