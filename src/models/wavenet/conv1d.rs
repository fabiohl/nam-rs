// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Malha CNN Causal Estática para inferência WaveNet (Design Orientado a Dados, SoA).
//!
//! Todas as estruturas utilizam `Const Generics` nas dimensões matemáticas e vetores pré-alocados
//! garantindo uma política de instanciamento estrito (Zero-Allocation durante processamento).
//! As loops dinâmicos resolvem cálculos em sequências FMA determinísticas via AVX2.

//! Módulo de Inferência WaveNet (Arquitetura Causal Dilatada).

use crate::math::common::{AlignedVec, PrefetchFn, SimdMath};

/// Convolução Causal Dilatada (WaveNet Conv1D).
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Matriz achatada de pesos do tamanho OUT * K * IN.
    pub weights: AlignedVec<u16>,
    /// Viés causal, atrelado se do_bias for verdadeiro. Total: OUT.
    pub bias: AlignedVec<f32>,
    /// Determina se o array de bias deve ser somado.
    pub do_bias: bool,
    /// Fator de diluição no eixo temporal causacional (Ex: 1, 2, 4.. 512).
    pub dilation: usize,
    /// Estratégia de prefetch pré-calculada (Eliminação de Branch).
    pub prefetch_fn: PrefetchFn,
}

/// Ponte de Dados (ConvInput):
/// Esta trait é uma ponte que permite ao NAM-rs usar exatamente o mesmo código
/// para dois tipos de números: decimais comuns (f32) e números compactos (u16/BF16).
/// Isso evita duplicar lógica complexa e facilita a manutenção.
trait ConvInput: Copy + Default {
    /// Versão 4x: Calcula 4 canais ao mesmo tempo.
    unsafe fn dot_product_4x_interleaved<M: SimdMath>(
        weights: &[[u16; 4]],
        state: &[Self],
    ) -> [f32; 4];

    /// Versão Dual Frame: Calcula 4 canais de DOIS frames simultaneamente.
    unsafe fn dot_product_4x_interleaved_dual_frame<M: SimdMath>(
        weights: &[[u16; 4]],
        state_f0: &[Self],
        state_f1: &[Self],
    ) -> ([f32; 4], [f32; 4]);

    /// Ajuste de Ponteiro: Garante que o endereço de memória esteja no formato correto.
    fn cast_ptr(ptr: *const Self) -> *const f32;
}

// 1. Modo de Precisão Total (f32):
// Usado em computadores que priorizam a fidelidade absoluta do som.
impl ConvInput for f32 {
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved<M: SimdMath>(
        weights: &[[u16; 4]],
        state: &[Self],
    ) -> [f32; 4] {
        unsafe { M::dot_product_4x_interleaved(weights, state) }
    }
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame<M: SimdMath>(
        weights: &[[u16; 4]],
        state_f0: &[Self],
        state_f1: &[Self],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe { M::dot_product_4x_interleaved_dual_frame(weights, state_f0, state_f1) }
    }
    #[inline(always)]
    fn cast_ptr(ptr: *const Self) -> *const f32 {
        ptr
    }
}

// 2. Modo 'Turbo' (u16/BF16):
// Usado para ganhar velocidade. O formato BF16 corta o tamanho dos dados pela metade,
// permitindo que o processador calcule muito mais rápido com uma perda de qualidade
// que é imperceptível ao ouvido humano.
impl ConvInput for u16 {
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved<M: SimdMath>(
        weights: &[[u16; 4]],
        state: &[Self],
    ) -> [f32; 4] {
        unsafe { M::dot_product_4x_interleaved_bf16(weights, state) }
    }
    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame<M: SimdMath>(
        weights: &[[u16; 4]],
        state_f0: &[Self],
        state_f1: &[Self],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe { M::dot_product_4x_interleaved_dual_frame_bf16(weights, state_f0, state_f1) }
    }
    #[inline(always)]
    fn cast_ptr(ptr: *const Self) -> *const f32 {
        ptr as *const f32
    }
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
        unsafe {
            self.process_single_frame_internal::<M>(layer_buffer, out_frame, frame_idx, None);
        }
    }

    /// Variante fundida que adiciona um vetor Mixin (condicionamento) diretamente no acumulador.
    #[inline(always)]
    /// Soma o mixin e processa a Conv1D para um único frame.
    ///
    /// # Safety
    /// O chamador deve garantir que `frame_idx` e `mixin` sejam válidos.
    pub unsafe fn process_single_frame_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: &[f32],
    ) {
        unsafe {
            self.process_single_frame_internal::<M>(
                layer_buffer,
                out_frame,
                frame_idx,
                Some(mixin),
            );
        }
    }

    #[inline(always)]
    unsafe fn process_single_frame_internal<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        unsafe {
            self.process_single_frame_generic::<M, f32>(layer_buffer, out_frame, frame_idx, mixin);
        }
    }

    #[inline(always)]
    unsafe fn process_single_frame_generic<M: SimdMath, T: ConvInput>(
        &self,
        layer_buffer: &[T],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        // [PASSO 1: Inicialização do Acumulador]
        if let Some(m) = mixin {
            if self.do_bias {
                out_frame.copy_from_slice(&self.bias[0..OUT]);
                unsafe {
                    M::accumulate_head(out_frame, m);
                }
            } else {
                out_frame.copy_from_slice(m);
            }
        } else if self.do_bias {
            out_frame.copy_from_slice(&self.bias[0..OUT]);
        } else {
            out_frame.fill(0.0);
        }

        // [PASSO 2: Iteração do Kernel (Receptive Field)]
        // Inversão de Loop: Channel-First Tiling.
        // Processamos todos os taps (K) para um bloco de canais de saída antes de mover para o próximo.
        // Isso mantém os acumuladores nos registros SIMD, reduzindo tráfego de cache L1.

        // Pre-carregamento dos taps (Input data) para o bloco atual.
        // Como K e IN são pequenos (ex: 3 e 16), o custo de cópia para a stack é compensado
        // pela eliminação de re-cálculos de endereços e maior localidade no loop b-first.
        let mut in_taps = [[T::default(); IN]; K];
        for (k, in_tap) in in_taps.iter_mut().enumerate() {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * IN;
            unsafe {
                in_tap.copy_from_slice(
                    layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN),
                );
            }

            // Prefetch via estratégia pré-calculada (Branchless)
            unsafe {
                (self.prefetch_fn)(
                    T::cast_ptr(layer_buffer.as_ptr().add(in_slice_start)),
                    self.dilation * IN,
                    k,
                    K,
                    self.dilation,
                );
            }
        }

        // Processamento de Convolução 1D por Blocos Intercalados (Interleaved):
        // Para otimizar o throughput de cálculo e uso de cache, processamos os canais de saída
        // agrupados em blocos de 4 elementos. Isso permite computar 4 saídas em paralelo usando
        // instruções SIMD que lêem os pesos e as entradas de forma altamente combinada.
        let num_blocks = OUT.div_ceil(4);
        let mut out_c = 0;

        for b in 0..num_blocks {
            let mut r0;
            let mut r1;
            let mut r2;
            let mut r3;

            // Carrega os 4 acumuladores temporários a partir do frame de saída atual.
            unsafe {
                r0 = *out_frame.get_unchecked(out_c);
                if OUT.is_multiple_of(4) || out_c + 3 < OUT {
                    r1 = *out_frame.get_unchecked(out_c + 1);
                    r2 = *out_frame.get_unchecked(out_c + 2);
                    r3 = *out_frame.get_unchecked(out_c + 3);
                } else {
                    r1 = if out_c + 1 < OUT {
                        *out_frame.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    };
                    r2 = if out_c + 2 < OUT {
                        *out_frame.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    };
                    r3 = if out_c + 3 < OUT {
                        *out_frame.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    };
                }
            }

            // Para cada tap (atraso/deslocamento no buffer de áudio circular) da convolução
            for (k, in_slice) in in_taps.iter().enumerate() {
                let w_start = (b * K + k) * IN * 4;
                let w_slice: &[[u16; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                // Realiza o produto escalar intercalado de 4 canais de uma só vez.
                let [t0, t1, t2, t3] =
                    unsafe { T::dot_product_4x_interleaved::<M>(w_slice, in_slice) };
                r0 += t0;
                r1 += t1;
                r2 += t2;
                r3 += t3;
            }

            // Grava de volta os 4 acumuladores processados no buffer de saída in-place.
            unsafe {
                *out_frame.get_unchecked_mut(out_c) = r0;
                if OUT.is_multiple_of(4) || out_c + 3 < OUT {
                    *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    *out_frame.get_unchecked_mut(out_c + 3) = r3;
                } else {
                    if out_c + 1 < OUT {
                        *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    }
                    if out_c + 2 < OUT {
                        *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    }
                    if out_c + 3 < OUT {
                        *out_frame.get_unchecked_mut(out_c + 3) = r3;
                    }
                }
            }
            out_c += 4;
        }
    }

    /// Processa um único frame usando buffers BF16 (VNNI).
    ///
    /// # Safety
    /// O chamador deve garantir que `layer_buffer` e `out_frame` tenham tamanhos
    /// compatíveis com as dimensões `IN` e `OUT` da camada, e que as instruções
    /// SIMD solicitadas pelo despachante `M` estejam disponíveis.
    #[inline(always)]
    pub unsafe fn process_single_frame_bf16<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame: &mut [f32],
        frame_idx: usize,
    ) {
        unsafe {
            self.process_single_frame_bf16_internal::<M>(layer_buffer, out_frame, frame_idx, None);
        }
    }

    /// Variante fundida BF16 que adiciona um vetor Mixin diretamente no acumulador.
    #[inline(always)]
    /// Soma o mixin e processa a Conv1D (BF16) para um único frame.
    ///
    /// # Safety
    /// O chamador deve garantir que `frame_idx` e `mixin` sejam válidos.
    pub unsafe fn process_single_frame_bf16_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: &[f32],
    ) {
        unsafe {
            self.process_single_frame_bf16_internal::<M>(
                layer_buffer,
                out_frame,
                frame_idx,
                Some(mixin),
            );
        }
    }

    #[inline(always)]
    unsafe fn process_single_frame_bf16_internal<M: SimdMath>(
        &self,
        layer_buffer: &[u16],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        unsafe {
            self.process_single_frame_generic::<M, u16>(layer_buffer, out_frame, frame_idx, mixin);
        }
    }

    /// Processa bloco iterativo sequencial.
    /// Para eficiência no cache, em vez de processar toda a camada por múltiplos blocos,
    /// limitamos a chamadas consecutivas quadro a quadro (`process_single_frame`).
    ///
    /// # Safety
    /// Pointer must be valid e num_frames deve estar contido nos limites do layer_buffer.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            // [PASSO: Delegação por Frame]
            // Fatia o buffer de saída (output multi-canal do tamanho `OUT`) e despacha para cálculo.
            let out_frame = unsafe { block.get_unchecked_mut(i * OUT..i * OUT + OUT) };
            unsafe {
                self.process_single_frame::<M>(layer_buffer, out_frame, buffer_start + i);
            }
        }
    }

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
        // Começamos cada cálculo preenchendo os frames com o valor base (bias)
        // e somando o mixin da camada anterior, se existirem.
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
        // Localizamos no buffer onde estão os sons antigos ('taps') que
        // influenciarão o som atual, respeitando a dilatação da camada.
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
                // Software Prefetch: Avisamos o processador para já ir buscando os próximos dados.
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
        // Processamos a rede neural em blocos de 4 canais de saída.
        // Esta é a parte que consome mais CPU.
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
                // Carregamos o que já calculamos até agora (bias + mixin).
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

                // A Mágica do Dual Frame:
                // Multiplicamos o mesmo peso por DOIS frames de áudio ao mesmo tempo.
                // Note que funciona tanto para F32 quanto para BF16 graças à trait T.
                let (t_f0, t_f1) = unsafe {
                    T::dot_product_4x_interleaved_dual_frame::<M>(w_slice, in_slice_f0, in_slice_f1)
                };

                // Somamos a influência deste tap nos acumuladores.
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
                // Devolvemos o resultado final para o buffer de saída.
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
    /// Modo Dual Frame BF16 (Turbo):
    /// O caminho mais rápido para processar dois frames simultaneamente
    /// usando a eficiência de memória do formato de 16 bits (BF16).
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
