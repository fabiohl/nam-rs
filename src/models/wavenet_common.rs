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
    /// Processa dois frames simultaneamente para reduzir overhead de carregamento de pesos.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// `out_f0` e `out_f1` devem ter tamanho compatível com `self.out_ch`.
    /// `mixin_f0` e `mixin_f1` devem ter tamanho pelo menos `self.out_ch` se fornecidos.
    pub unsafe fn process_dual_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch / 4;

        // Tap Pointers para f0 e f1
        let mut tap_ptrs_f0 = [core::ptr::null::<f32>(); 8];
        let mut tap_ptrs_f1 = [core::ptr::null::<f32>(); 8];
        let k_limit = self.kernel.min(8);

        for k in 0..k_limit {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            let in_start_f0 = ((idx_f0 as isize) + offset) as usize * self.in_ch;
            let in_start_f1 = ((idx_f1 as isize) + offset) as usize * self.in_ch;
            unsafe {
                tap_ptrs_f0[k] = layer_buffer.as_ptr().add(in_start_f0);
                tap_ptrs_f1[k] = layer_buffer.as_ptr().add(in_start_f1);

                (self.prefetch_fn)(
                    tap_ptrs_f0[k],
                    self.dilation * self.in_ch,
                    k,
                    self.kernel,
                    self.dilation,
                );
            }
        }

        let in_ch = self.in_ch;
        let kernel = self.kernel;
        let do_bias = self.do_bias;

        for b in 0..num_blocks {
            let out_c = b * 4;
            let (mut r0_f0, mut r1_f0, mut r2_f0, mut r3_f0);
            let (mut r0_f1, mut r1_f1, mut r2_f1, mut r3_f1);

            unsafe {
                // Inicialização com bias/mixin para frame 0
                if let Some(m) = mixin_f0 {
                    if do_bias {
                        r0_f0 = *self.bias.get_unchecked(out_c) + *m.get_unchecked(out_c);
                        r1_f0 = *self.bias.get_unchecked(out_c + 1) + *m.get_unchecked(out_c + 1);
                        r2_f0 = *self.bias.get_unchecked(out_c + 2) + *m.get_unchecked(out_c + 2);
                        r3_f0 = *self.bias.get_unchecked(out_c + 3) + *m.get_unchecked(out_c + 3);
                    } else {
                        r0_f0 = *m.get_unchecked(out_c);
                        r1_f0 = *m.get_unchecked(out_c + 1);
                        r2_f0 = *m.get_unchecked(out_c + 2);
                        r3_f0 = *m.get_unchecked(out_c + 3);
                    }
                } else if do_bias {
                    r0_f0 = *self.bias.get_unchecked(out_c);
                    r1_f0 = *self.bias.get_unchecked(out_c + 1);
                    r2_f0 = *self.bias.get_unchecked(out_c + 2);
                    r3_f0 = *self.bias.get_unchecked(out_c + 3);
                } else {
                    r0_f0 = 0.0;
                    r1_f0 = 0.0;
                    r2_f0 = 0.0;
                    r3_f0 = 0.0;
                }

                // Inicialização com bias/mixin para frame 1
                if let Some(m) = mixin_f1 {
                    if do_bias {
                        r0_f1 = *self.bias.get_unchecked(out_c) + *m.get_unchecked(out_c);
                        r1_f1 = *self.bias.get_unchecked(out_c + 1) + *m.get_unchecked(out_c + 1);
                        r2_f1 = *self.bias.get_unchecked(out_c + 2) + *m.get_unchecked(out_c + 2);
                        r3_f1 = *self.bias.get_unchecked(out_c + 3) + *m.get_unchecked(out_c + 3);
                    } else {
                        r0_f1 = *m.get_unchecked(out_c);
                        r1_f1 = *m.get_unchecked(out_c + 1);
                        r2_f1 = *m.get_unchecked(out_c + 2);
                        r3_f1 = *m.get_unchecked(out_c + 3);
                    }
                } else if do_bias {
                    r0_f1 = *self.bias.get_unchecked(out_c);
                    r1_f1 = *self.bias.get_unchecked(out_c + 1);
                    r2_f1 = *self.bias.get_unchecked(out_c + 2);
                    r3_f1 = *self.bias.get_unchecked(out_c + 3);
                } else {
                    r0_f1 = 0.0;
                    r1_f1 = 0.0;
                    r2_f1 = 0.0;
                    r3_f1 = 0.0;
                }

                for k in 0..kernel {
                    let tap_f0 = *tap_ptrs_f0.get_unchecked(k);
                    let tap_f1 = *tap_ptrs_f1.get_unchecked(k);

                    let w_start = (b * kernel + k) * in_ch * 4;
                    let w_slice: &[[u16; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                        core::slice::from_raw_parts(ptr, in_ch)
                    };

                    let in_f0 = core::slice::from_raw_parts(tap_f0, in_ch);
                    let in_f1 = core::slice::from_raw_parts(tap_f1, in_ch);

                    let (t_f0, t_f1) =
                        M::dot_product_4x_interleaved_dual_frame(w_slice, in_f0, in_f1);
                    r0_f0 += t_f0[0];
                    r1_f0 += t_f0[1];
                    r2_f0 += t_f0[2];
                    r3_f0 += t_f0[3];
                    r0_f1 += t_f1[0];
                    r1_f1 += t_f1[1];
                    r2_f1 += t_f1[2];
                    r3_f1 += t_f1[3];
                }

                *out_f0.get_unchecked_mut(out_c) = r0_f0;
                *out_f0.get_unchecked_mut(out_c + 1) = r1_f0;
                *out_f0.get_unchecked_mut(out_c + 2) = r2_f0;
                *out_f0.get_unchecked_mut(out_c + 3) = r3_f0;

                *out_f1.get_unchecked_mut(out_c) = r0_f1;
                *out_f1.get_unchecked_mut(out_c + 1) = r1_f1;
                *out_f1.get_unchecked_mut(out_c + 2) = r2_f1;
                *out_f1.get_unchecked_mut(out_c + 3) = r3_f1;
            }
        }

        // Remainder canais
        let mut out_c = num_blocks * 4;
        while out_c < self.out_ch {
            let mut r_f0 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            let mut r_f1 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            unsafe {
                if let Some(m) = mixin_f0 {
                    r_f0 += m[out_c];
                }
                if let Some(m) = mixin_f1 {
                    r_f1 += m[out_c];
                }

                for k in 0..self.kernel {
                    let in_f0 = core::slice::from_raw_parts(tap_ptrs_f0[k], self.in_ch);
                    let in_f1 = core::slice::from_raw_parts(tap_ptrs_f1[k], self.in_ch);
                    let w_start = (out_c * self.kernel + k) * self.in_ch;
                    let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                    r_f0 += M::dot_product(in_f0, w);
                    r_f1 += M::dot_product(in_f1, w);
                }
            }
            out_f0[out_c] = r_f0;
            out_f1[out_c] = r_f1;
            out_c += 1;
        }
    }

    /// Processa um bloco de amostras com mixin opcional.
    #[inline(always)]
    /// # Safety
    /// `output` deve ter tamanho pelo menos `num_frames * self.out_size`.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;
        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);
            let (m_f0, m_f1) = if let Some(m) = mixin {
                (
                    Some(&m[i * self.out_ch..(i + 1) * self.out_ch]),
                    Some(&m[(i + 1) * self.out_ch..(i + 2) * self.out_ch]),
                )
            } else {
                (None, None)
            };

            unsafe {
                self.process_dual_frame::<M>(
                    layer_buffer,
                    out_f0,
                    out_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    m_f0,
                    m_f1,
                );
            }
            i += 2;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let m = mixin.map(|m| &m[i * self.out_ch..(i + 1) * self.out_ch]);
            unsafe {
                self.process_single_frame::<M>(layer_buffer, rem, buffer_start + i, m);
            }
        }
    }

    /// Processa um único frame (Helper para o loop de bloco).
    #[inline(always)]
    /// # Safety
    /// `out_frame` deve ter tamanho compatível com `self.out_ch`.
    /// `mixin` deve ter tamanho pelo menos `self.out_ch` se fornecido.
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch / 4;
        let mut tap_ptrs = [core::ptr::null::<f32>(); 8];
        let k_limit = self.kernel.min(8);
        for (k, tap_ptr) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * self.in_ch;
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
                    if self.do_bias {
                        r0 = *self.bias.get_unchecked(out_c) + m.get(out_c).copied().unwrap_or(0.0);
                        r1 = *self.bias.get_unchecked(out_c + 1)
                            + m.get(out_c + 1).copied().unwrap_or(0.0);
                        r2 = *self.bias.get_unchecked(out_c + 2)
                            + m.get(out_c + 2).copied().unwrap_or(0.0);
                        r3 = *self.bias.get_unchecked(out_c + 3)
                            + m.get(out_c + 3).copied().unwrap_or(0.0);
                    } else {
                        r0 = m.get(out_c).copied().unwrap_or(0.0);
                        r1 = m.get(out_c + 1).copied().unwrap_or(0.0);
                        r2 = m.get(out_c + 2).copied().unwrap_or(0.0);
                        r3 = m.get(out_c + 3).copied().unwrap_or(0.0);
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

                *out_frame.get_unchecked_mut(out_c) = r0;
                *out_frame.get_unchecked_mut(out_c + 1) = r1;
                *out_frame.get_unchecked_mut(out_c + 2) = r2;
                *out_frame.get_unchecked_mut(out_c + 3) = r3;
            }
        }

        let mut out_c = num_blocks * 4;
        while out_c < self.out_ch {
            let mut r = if self.do_bias { self.bias[out_c] } else { 0.0 };
            unsafe {
                if let Some(m) = mixin {
                    r += m.get(out_c).copied().unwrap_or(0.0);
                }
                for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                    let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                    let w_start = (out_c * self.kernel + k) * self.in_ch;
                    let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                    r += M::dot_product(in_slice, w);
                }
            }
            out_frame[out_c] = r;
            out_c += 1;
        }
    }

    /// Processa dois frames simultaneamente usando BF16 na memória circular.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// `out_f0` e `out_f1` devem ter tamanho compatível com `self.out_ch`.
    /// `mixin_f0` e `mixin_f1` devem ter tamanho pelo menos `self.out_ch` se fornecidos.
    pub unsafe fn process_dual_frame_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch / 4;

        // Tap Pointers para f0 e f1 em BF16
        let mut tap_ptrs_f0 = [core::ptr::null::<u16>(); 8];
        let mut tap_ptrs_f1 = [core::ptr::null::<u16>(); 8];
        let k_limit = self.kernel.min(8);

        for k in 0..k_limit {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            let in_start_f0 = ((idx_f0 as isize) + offset) as usize * self.in_ch;
            let in_start_f1 = ((idx_f1 as isize) + offset) as usize * self.in_ch;
            unsafe {
                tap_ptrs_f0[k] = layer_buffer.as_ptr().add(in_start_f0);
                tap_ptrs_f1[k] = layer_buffer.as_ptr().add(in_start_f1);

                (self.prefetch_fn)(
                    tap_ptrs_f0[k] as *const f32,
                    self.dilation * self.in_ch,
                    k,
                    self.kernel,
                    self.dilation,
                );
            }
        }

        let in_ch = self.in_ch;
        let kernel = self.kernel;
        let do_bias = self.do_bias;

        for b in 0..num_blocks {
            let out_c = b * 4;
            let (mut r0_f0, mut r1_f0, mut r2_f0, mut r3_f0);
            let (mut r0_f1, mut r1_f1, mut r2_f1, mut r3_f1);

            unsafe {
                // Inicialização com bias/mixin para frame 0
                if let Some(m) = mixin_f0 {
                    let m_val0 = m.get(out_c).copied().unwrap_or(0.0);
                    let m_val1 = m.get(out_c + 1).copied().unwrap_or(0.0);
                    let m_val2 = m.get(out_c + 2).copied().unwrap_or(0.0);
                    let m_val3 = m.get(out_c + 3).copied().unwrap_or(0.0);
                    if do_bias {
                        r0_f0 = *self.bias.get_unchecked(out_c) + m_val0;
                        r1_f0 = *self.bias.get_unchecked(out_c + 1) + m_val1;
                        r2_f0 = *self.bias.get_unchecked(out_c + 2) + m_val2;
                        r3_f0 = *self.bias.get_unchecked(out_c + 3) + m_val3;
                    } else {
                        r0_f0 = m_val0;
                        r1_f0 = m_val1;
                        r2_f0 = m_val2;
                        r3_f0 = m_val3;
                    }
                } else if do_bias {
                    r0_f0 = *self.bias.get_unchecked(out_c);
                    r1_f0 = *self.bias.get_unchecked(out_c + 1);
                    r2_f0 = *self.bias.get_unchecked(out_c + 2);
                    r3_f0 = *self.bias.get_unchecked(out_c + 3);
                } else {
                    r0_f0 = 0.0;
                    r1_f0 = 0.0;
                    r2_f0 = 0.0;
                    r3_f0 = 0.0;
                }

                // Inicialização com bias/mixin para frame 1
                if let Some(m) = mixin_f1 {
                    let m_val0 = m.get(out_c).copied().unwrap_or(0.0);
                    let m_val1 = m.get(out_c + 1).copied().unwrap_or(0.0);
                    let m_val2 = m.get(out_c + 2).copied().unwrap_or(0.0);
                    let m_val3 = m.get(out_c + 3).copied().unwrap_or(0.0);
                    if do_bias {
                        r0_f1 = *self.bias.get_unchecked(out_c) + m_val0;
                        r1_f1 = *self.bias.get_unchecked(out_c + 1) + m_val1;
                        r2_f1 = *self.bias.get_unchecked(out_c + 2) + m_val2;
                        r3_f1 = *self.bias.get_unchecked(out_c + 3) + m_val3;
                    } else {
                        r0_f1 = m_val0;
                        r1_f1 = m_val1;
                        r2_f1 = m_val2;
                        r3_f1 = m_val3;
                    }
                } else if do_bias {
                    r0_f1 = *self.bias.get_unchecked(out_c);
                    r1_f1 = *self.bias.get_unchecked(out_c + 1);
                    r2_f1 = *self.bias.get_unchecked(out_c + 2);
                    r3_f1 = *self.bias.get_unchecked(out_c + 3);
                } else {
                    r0_f1 = 0.0;
                    r1_f1 = 0.0;
                    r2_f1 = 0.0;
                    r3_f1 = 0.0;
                }

                for k in 0..kernel {
                    let tap_f0 = *tap_ptrs_f0.get_unchecked(k);
                    let tap_f1 = *tap_ptrs_f1.get_unchecked(k);

                    let w_start = (b * kernel + k) * in_ch * 4;
                    let w_slice: &[[u16; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                        core::slice::from_raw_parts(ptr, in_ch)
                    };

                    let in_f0 = core::slice::from_raw_parts(tap_f0, in_ch);
                    let in_f1 = core::slice::from_raw_parts(tap_f1, in_ch);

                    let (t_f0, t_f1) =
                        M::dot_product_4x_interleaved_dual_frame_bf16(w_slice, in_f0, in_f1);
                    r0_f0 += t_f0[0];
                    r1_f0 += t_f0[1];
                    r2_f0 += t_f0[2];
                    r3_f0 += t_f0[3];
                    r0_f1 += t_f1[0];
                    r1_f1 += t_f1[1];
                    r2_f1 += t_f1[2];
                    r3_f1 += t_f1[3];
                }

                *out_f0.get_unchecked_mut(out_c) = r0_f0;
                *out_f0.get_unchecked_mut(out_c + 1) = r1_f0;
                *out_f0.get_unchecked_mut(out_c + 2) = r2_f0;
                *out_f0.get_unchecked_mut(out_c + 3) = r3_f0;

                *out_f1.get_unchecked_mut(out_c) = r0_f1;
                *out_f1.get_unchecked_mut(out_c + 1) = r1_f1;
                *out_f1.get_unchecked_mut(out_c + 2) = r2_f1;
                *out_f1.get_unchecked_mut(out_c + 3) = r3_f1;
            }
        }

        // Remainder canais
        let mut out_c = num_blocks * 4;
        while out_c < self.out_ch {
            let mut r_f0 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            let mut r_f1 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            unsafe {
                if let Some(m) = mixin_f0 {
                    r_f0 += m[out_c];
                }
                if let Some(m) = mixin_f1 {
                    r_f1 += m[out_c];
                }

                for k in 0..self.kernel {
                    let in_f0 = core::slice::from_raw_parts(tap_ptrs_f0[k], self.in_ch);
                    let in_f1 = core::slice::from_raw_parts(tap_ptrs_f1[k], self.in_ch);
                    let w_start = (out_c * self.kernel + k) * self.in_ch;
                    let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                    r_f0 += M::dot_product_bf16(in_f0, w);
                    r_f1 += M::dot_product_bf16(in_f1, w);
                }
            }
            out_f0[out_c] = r_f0;
            out_f1[out_c] = r_f1;
            out_c += 1;
        }
    }

    /// Processa um bloco de amostras usando BF16 na memória circular (layer_buffer).
    /// # Safety
    /// `output` deve ter tamanho pelo menos `num_frames * self.out_size`.
    #[inline(always)]
    pub unsafe fn process_block_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;
        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);
            let (m_f0, m_f1) = if let Some(m) = mixin {
                (
                    Some(&m[i * self.out_ch..(i + 1) * self.out_ch]),
                    Some(&m[(i + 1) * self.out_ch..(i + 2) * self.out_ch]),
                )
            } else {
                (None, None)
            };

            unsafe {
                self.process_dual_frame_bf16::<M>(
                    layer_buffer,
                    out_f0,
                    out_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    m_f0,
                    m_f1,
                );
            }
            i += 2;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let m = mixin.map(|m| &m[i * self.out_ch..(i + 1) * self.out_ch]);
            unsafe {
                self.process_single_frame_bf16::<M>(layer_buffer, rem, buffer_start + i, m);
            }
        }
    }

    /// Processa um único frame BF16 (Helper para o loop de bloco).
    #[inline(always)]
    /// # Safety
    /// `out_frame` deve ter tamanho compatível com `self.out_ch`.
    /// `mixin` deve ter tamanho pelo menos `self.out_ch` se fornecido.
    pub unsafe fn process_single_frame_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch / 4;
        let mut tap_ptrs = [core::ptr::null::<u16>(); 8];
        let k_limit = self.kernel.min(8);
        for (k, tap_ptr) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * self.in_ch;
            unsafe {
                *tap_ptr = layer_buffer.as_ptr().add(in_slice_start);
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
                    if self.do_bias {
                        r0 = *self.bias.get_unchecked(out_c) + m.get(out_c).copied().unwrap_or(0.0);
                        r1 = *self.bias.get_unchecked(out_c + 1)
                            + m.get(out_c + 1).copied().unwrap_or(0.0);
                        r2 = *self.bias.get_unchecked(out_c + 2)
                            + m.get(out_c + 2).copied().unwrap_or(0.0);
                        r3 = *self.bias.get_unchecked(out_c + 3)
                            + m.get(out_c + 3).copied().unwrap_or(0.0);
                    } else {
                        r0 = m.get(out_c).copied().unwrap_or(0.0);
                        r1 = m.get(out_c + 1).copied().unwrap_or(0.0);
                        r2 = m.get(out_c + 2).copied().unwrap_or(0.0);
                        r3 = m.get(out_c + 3).copied().unwrap_or(0.0);
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
                    let [t0, t1, t2, t3] = M::dot_product_4x_interleaved_bf16(w_slice, in_slice);
                    r0 += t0;
                    r1 += t1;
                    r2 += t2;
                    r3 += t3;
                }

                *out_frame.get_unchecked_mut(out_c) = r0;
                *out_frame.get_unchecked_mut(out_c + 1) = r1;
                *out_frame.get_unchecked_mut(out_c + 2) = r2;
                *out_frame.get_unchecked_mut(out_c + 3) = r3;
            }
        }

        let mut out_c = num_blocks * 4;
        while out_c < self.out_ch {
            let mut r = if self.do_bias { self.bias[out_c] } else { 0.0 };
            unsafe {
                if let Some(m) = mixin {
                    r += m.get(out_c).copied().unwrap_or(0.0);
                }
                for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                    let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);
                    let w_start = (out_c * self.kernel + k) * self.in_ch;
                    let w = self.weights.get_unchecked(w_start..w_start + self.in_ch);
                    r += M::dot_product_bf16(in_slice, w);
                }
            }
            out_frame[out_c] = r;
            out_c += 1;
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
    ///
    /// Depende da validade dos buffers de entrada e saída para num_frames.
    #[inline(always)]
    /// # Safety
    /// `output` deve ter tamanho pelo menos `num_frames * self.out_size`.
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
                M::gemv_overwrite(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
            }
        }
    }

    /// Processa a camada usando BF16.
    ///
    /// # Safety
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
        for i in 0..num_frames {
            unsafe {
                let in_slice = input.get_unchecked(i * self.in_size..(i + 1) * self.in_size);
                let out_slice =
                    output.get_unchecked_mut(i * self.out_size..(i + 1) * self.out_size);
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
    /// Executa o processamento interno de uma camada WaveNet com Tiling Dual-Frame.
    /// # Safety
    /// `ctx.block` deve ser grande o suficiente para conter `num_frames * out_ch` amostras.
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

        // Buffer temporário na stack para o Mixin (16KB) com alinhamento de 64 bytes.
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
            if M::IS_BF16 {
                // 1. Mixin (Batch)
                self.input_mixin.process_block_bf16::<M>(
                    condition_bf16,
                    mixin_out_slice,
                    num_frames,
                );

                // 2. Conv1D (Tiled Loop)
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
                // 1. Mixin (Batch)
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);

                // 2. Conv1D (Tiled Loop)
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

            // 3. Activation (Batch)
            if self.gated {
                M::gated_activation_and_accumulate_block(
                    head_input,
                    &mut block[..num_frames * 2 * ch],
                    ch,
                );
                // [T1.2] Otimização: Compactação para processamento Residual em lote.
                // Re-alinhamos os outputs ativados [f0_act, f0_sig, f1_act, f1_sig]
                // para [f0_act, f1_act, ...] permitindo GEMM de 1x1 em lote.
                for i in 1..num_frames {
                    block.copy_within(i * 2 * ch..i * 2 * ch + ch, i * ch);
                }
            } else {
                M::tanh_and_accumulate_block(head_input, &mut block[..num_frames * ch]);
            }

            // 4. Residual + 1x1 (Batch)
            let lb_offset = buffer_start * ch;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * ch);

            self.one_by_one.process_residual_batch::<M>(
                &block[..num_frames * ch],
                residual_slice,
                output,
                num_frames,
            );

            // 5. BF16 Conversion
            if let (true, Some(bf16_out)) = (M::IS_BF16, output_bf16.as_mut()) {
                M::f32_to_bf16(output, bf16_out);
            }
        }
    }
}
