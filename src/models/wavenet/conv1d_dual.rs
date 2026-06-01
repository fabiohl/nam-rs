// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Processamento Dual-Frame da Convolução Causal WaveNet.
//!
//! Extensão do `Conv1d` com métodos que processam dois frames simultaneamente
//! (Temporal Tiling), maximizando a reutilização de pesos nos registradores.

use super::conv1d::{Conv1d, ConvInput};
use crate::math::common::SimdMath;

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Variante fundida que processa dois frames simultaneamente, adicionando vetores Mixin (condicionamento) diretamente nos acumuladores.
    /// Esta abordagem maximiza a utilização dos pesos carregados nos registradores (Temporal Tiling).
    ///
    /// # Safety
    /// `layer_buffer` e `mixin` devem possuir os tamanhos adequados.
    /// Processamento Dual Frame com Mixin:
    /// Esta função calcula dois momentos do áudio de uma vez só, já integrando
    /// as configurações externas (mixin) para economizar tempo de processamento.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_dual_frame_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: &[f32],
        mixin_f1: &[f32],
    ) {
        unsafe {
            self.process_dual_frame_internal::<M>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                Some(mixin_f0),
                Some(mixin_f1),
            );
        }
    }

    /// Organizador Interno:
    /// Prepara os dados para o 'Motor Universal' (Generic), decidindo como
    /// as informações serão enviadas para o cálculo.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn process_dual_frame_internal<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        unsafe {
            self.process_dual_frame_generic::<M, f32>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                mixin_f0,
                mixin_f1,
            );
        }
    }

    /// Motor Universal (Generic Engine):
    /// Esta é a inteligência central que faz a matemática pesada. Graças ao uso
    /// de tipos genéricos (T: ConvInput), este mesmo código funciona tanto no
    /// modo de precisão total quanto no modo ultra-rápido (BF16).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn process_dual_frame_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        // --- 1. Preparação (Bias e Mixin) ---
        if let (Some(m0), Some(m1)) = (mixin_f0, mixin_f1) {
            if self.do_bias {
                out_frame_f0.copy_from_slice(&self.bias[0..OUT]);
                out_frame_f1.copy_from_slice(&self.bias[0..OUT]);
                unsafe {
                    M::accumulate_head(out_frame_f0, m0);
                    M::accumulate_head(out_frame_f1, m1);
                }
            } else {
                out_frame_f0.copy_from_slice(m0);
                out_frame_f1.copy_from_slice(m1);
            }
        } else if self.do_bias {
            out_frame_f0.copy_from_slice(&self.bias[0..OUT]);
            out_frame_f1.copy_from_slice(&self.bias[0..OUT]);
        } else {
            out_frame_f0.fill(0.0);
            out_frame_f1.fill(0.0);
        }

        // --- 2. Busca no Passado (Dilatação) ---
        let mut in_taps_f0 = [[T::default(); IN]; K];
        let mut in_taps_f1 = [[T::default(); IN]; K];
        for k in 0..K {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_slice_start_f0 = ((frame_idx_f0 as isize) + offset) as usize * IN;
            let in_slice_start_f1 = ((frame_idx_f1 as isize) + offset) as usize * IN;
            unsafe {
                in_taps_f0.get_unchecked_mut(k).copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start_f0..in_slice_start_f0 + IN),
                );
                in_taps_f1.get_unchecked_mut(k).copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start_f1..in_slice_start_f1 + IN),
                );
                (self.prefetch_fn)(
                    T::cast_ptr(layer_buffer.as_ptr().add(in_slice_start_f0)),
                    self.dilation * IN,
                    k,
                    K,
                    self.dilation,
                );
            }
        }

        let num_blocks = OUT.div_ceil(4);
        let mut out_c = 0;

        // --- 3. Loop de Cálculo Central (Blocks de 4) ---
        for b in 0..num_blocks {
            let mut r0_f0;
            let mut r1_f0;
            let mut r2_f0;
            let mut r3_f0;
            let mut r0_f1;
            let mut r1_f1;
            let mut r2_f1;
            let mut r3_f1;

            unsafe {
                r0_f0 = *out_frame_f0.get_unchecked(out_c);
                r0_f1 = *out_frame_f1.get_unchecked(out_c);
                if OUT.is_multiple_of(4) || out_c + 3 < OUT {
                    r1_f0 = *out_frame_f0.get_unchecked(out_c + 1);
                    r2_f0 = *out_frame_f0.get_unchecked(out_c + 2);
                    r3_f0 = *out_frame_f0.get_unchecked(out_c + 3);

                    r1_f1 = *out_frame_f1.get_unchecked(out_c + 1);
                    r2_f1 = *out_frame_f1.get_unchecked(out_c + 2);
                    r3_f1 = *out_frame_f1.get_unchecked(out_c + 3);
                } else {
                    r1_f0 = if out_c + 1 < OUT {
                        *out_frame_f0.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    };
                    r2_f0 = if out_c + 2 < OUT {
                        *out_frame_f0.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    };
                    r3_f0 = if out_c + 3 < OUT {
                        *out_frame_f0.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    };

                    r1_f1 = if out_c + 1 < OUT {
                        *out_frame_f1.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    };
                    r2_f1 = if out_c + 2 < OUT {
                        *out_frame_f1.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    };
                    r3_f1 = if out_c + 3 < OUT {
                        *out_frame_f1.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    };
                }
            }

            for k in 0..K {
                let w_start = (b * K + k) * IN * 4;
                let w_slice: &[[u16; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let in_slice_f0 = &in_taps_f0[k];
                let in_slice_f1 = &in_taps_f1[k];

                let (t_f0, t_f1) = unsafe {
                    T::dot_product_4x_interleaved_dual_frame::<M>(w_slice, in_slice_f0, in_slice_f1)
                };

                r0_f0 += t_f0[0];
                r1_f0 += t_f0[1];
                r2_f0 += t_f0[2];
                r3_f0 += t_f0[3];
                r0_f1 += t_f1[0];
                r1_f1 += t_f1[1];
                r2_f1 += t_f1[2];
                r3_f1 += t_f1[3];
            }

            unsafe {
                *out_frame_f0.get_unchecked_mut(out_c) = r0_f0;
                *out_frame_f1.get_unchecked_mut(out_c) = r0_f1;
                if OUT.is_multiple_of(4) || out_c + 3 < OUT {
                    *out_frame_f0.get_unchecked_mut(out_c + 1) = r1_f0;
                    *out_frame_f0.get_unchecked_mut(out_c + 2) = r2_f0;
                    *out_frame_f0.get_unchecked_mut(out_c + 3) = r3_f0;

                    *out_frame_f1.get_unchecked_mut(out_c + 1) = r1_f1;
                    *out_frame_f1.get_unchecked_mut(out_c + 2) = r2_f1;
                    *out_frame_f1.get_unchecked_mut(out_c + 3) = r3_f1;
                } else {
                    if out_c + 1 < OUT {
                        *out_frame_f0.get_unchecked_mut(out_c + 1) = r1_f0;
                    }
                    if out_c + 2 < OUT {
                        *out_frame_f0.get_unchecked_mut(out_c + 2) = r2_f0;
                    }
                    if out_c + 3 < OUT {
                        *out_frame_f0.get_unchecked_mut(out_c + 3) = r3_f0;
                    }

                    if out_c + 1 < OUT {
                        *out_frame_f1.get_unchecked_mut(out_c + 1) = r1_f1;
                    }
                    if out_c + 2 < OUT {
                        *out_frame_f1.get_unchecked_mut(out_c + 2) = r2_f1;
                    }
                    if out_c + 3 < OUT {
                        *out_frame_f1.get_unchecked_mut(out_c + 3) = r3_f1;
                    }
                }
            }
            out_c += 4;
        }
    }

    /// Variante fundida BF16 que processa dois frames simultaneamente, adicionando vetores Mixin diretamente nos acumuladores.
    /// Esta abordagem maximiza a utilização dos pesos (VNNI) carregados nos registradores (Temporal Tiling).
    ///
    /// # Safety
    /// O chamador deve garantir que `layer_buffer` e `mixin` possuam os tamanhos adequados.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn process_dual_frame_bf16_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: &[f32],
        mixin_f1: &[f32],
    ) {
        unsafe {
            self.process_dual_frame_bf16_internal::<M>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                Some(mixin_f0),
                Some(mixin_f1),
            );
        }
    }

    /// Organizador Interno BF16:
    /// Encaminha os dados de 16 bits para o Motor Universal de cálculo.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    unsafe fn process_dual_frame_bf16_internal<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        unsafe {
            self.process_dual_frame_generic::<M, u16>(
                layer_buffer,
                out_frame_f0,
                out_frame_f1,
                frame_idx_f0,
                frame_idx_f1,
                mixin_f0,
                mixin_f1,
            );
        }
    }
}
