// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Componentes comuns e dinâmicos para arquiteturas WaveNet.
//!
//! Contém as estruturas fundamentais (Conv1D, Dense, Layer) que operam com
//! dimensões definidas em runtime, servindo como base para o modelo dinâmico
//! e futuros estágios da arquitetura A2.
//!
//! IMPORTANTE: O suporte à arquitetura A2 está em estágio de "placeholder"
//! aguardando estabilização da implementação de referência.

use crate::math::common::{AlignedVec, PrefetchFn, SimdMath};

/// Máximo de frames a processar em um pulso do callback.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Padding temporal circular das memórias no framework de Ring Buffers.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;
/// Limite máximo suportado para o tamanho do kernel.
pub const MAX_KERNEL: usize = 16;

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
    pub prefetch_fn: PrefetchFn,
}

impl Conv1dDyn {
    /// Processa dois frames simultaneamente para reduzir overhead de carregamento de pesos.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    /// `out_f0` e `out_f1` devem ter tamanho compatível com `self.out_ch`.
    /// `mixin_f0` e `mixin_f1` devem ter tamanho pelo menos `self.out_ch` se fornecidos.
    /// Processa dois frames de áudio simultaneamente (Dual Frame) para máxima eficiência.
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
        // --- Processamento em Par (Dual Frame) ---
        // Para economizar energia da CPU, calculamos dois momentos do áudio (f0 e f1) ao mesmo tempo.
        // Isso aproveita melhor os dados que já estão 'quentes' no cache do processador.
        let num_blocks = self.out_ch.div_ceil(4);
        debug_assert!(self.weights.len() >= num_blocks * 4 * self.in_ch * self.kernel);

        debug_assert!(
            self.kernel <= MAX_KERNEL,
            "kernel {} excede MAX_KERNEL",
            self.kernel
        );
        // 'Tap Pointers': São como mãos que buscam amostras de áudio no passado.
        let mut tap_ptrs_f0 = [core::ptr::null::<f32>(); MAX_KERNEL];
        let mut tap_ptrs_f1 = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = self.kernel.min(MAX_KERNEL);

        // 1. Localização no Tempo (Dilatação):
        // O WaveNet usa 'Dilatação' para olhar para trás no tempo.
        // Em vai de olhar apenas para o vizinho imediato, ele pula amostras para
        // conseguir 'ouvir' padrões de longa duração (como o ritmo de uma guitarra).
        for k in 0..k_limit {
            // Calculamos a distância exata no passado baseada na dilatação e no tamanho do kernel.
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            let in_start_f0 = ((idx_f0 as isize) + offset) as usize * self.in_ch;
            let in_start_f1 = ((idx_f1 as isize) + offset) as usize * self.in_ch;

            unsafe {
                // Guardamos o endereço de onde buscar esses sons antigos.
                tap_ptrs_f0[k] = layer_buffer.as_ptr().add(in_start_f0);
                tap_ptrs_f1[k] = layer_buffer.as_ptr().add(in_start_f1);

                // Avisamos a CPU para buscar esses dados na RAM antecipadamente (Prefetch).
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

        // 2. Loop de Cálculo por Blocos:
        // Processamos a rede neural em blocos de 4 para otimizar a matemática do processador.
        for b in 0..num_blocks {
            let out_c = b * 4;
            // Variáveis temporárias (registradores) para guardar os resultados parciais dos dois frames.
            let (mut r0_f0, mut r1_f0, mut r2_f0, mut r3_f0);
            let (mut r0_f1, mut r1_f1, mut r2_f1, mut r3_f1);

            unsafe {
                // 3. Inicialização de cada Bloco (Bias e Mixin):
                // Começamos o cálculo de cada 'neurônio' somando o valor base (bias) e
                // o resultado da camada anterior (mixin), se existirem.
                let (mv0_f0, mv1_f0, mv2_f0, mv3_f0) = if let Some(m) = mixin_f0 {
                    if out_c + 3 < m.len() {
                        (
                            *m.get_unchecked(out_c),
                            *m.get_unchecked(out_c + 1),
                            *m.get_unchecked(out_c + 2),
                            *m.get_unchecked(out_c + 3),
                        )
                    } else {
                        {
                            let mut v = [0.0f32; 4];
                            for (i, val) in v.iter_mut().enumerate() {
                                if out_c + i < m.len() {
                                    *val = *m.get_unchecked(out_c + i);
                                }
                            }
                            (v[0], v[1], v[2], v[3])
                        }
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                if do_bias {
                    r0_f0 = *self.bias.get_unchecked(out_c) + mv0_f0;
                    r1_f0 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1_f0;
                    r2_f0 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2_f0;
                    r3_f0 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3_f0;
                } else {
                    r0_f0 = mv0_f0;
                    r1_f0 = mv1_f0;
                    r2_f0 = mv2_f0;
                    r3_f0 = mv3_f0;
                }

                // Repetimos o processo para o segundo frame (f1) do par.
                let (mv0_f1, mv1_f1, mv2_f1, mv3_f1) = if let Some(m) = mixin_f1 {
                    if out_c + 3 < m.len() {
                        (
                            *m.get_unchecked(out_c),
                            *m.get_unchecked(out_c + 1),
                            *m.get_unchecked(out_c + 2),
                            *m.get_unchecked(out_c + 3),
                        )
                    } else {
                        {
                            let mut v = [0.0f32; 4];
                            for (i, val) in v.iter_mut().enumerate() {
                                if out_c + i < m.len() {
                                    *val = *m.get_unchecked(out_c + i);
                                }
                            }
                            (v[0], v[1], v[2], v[3])
                        }
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                if do_bias {
                    r0_f1 = *self.bias.get_unchecked(out_c) + mv0_f1;
                    r1_f1 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1_f1;
                    r2_f1 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2_f1;
                    r3_f1 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3_f1;
                } else {
                    r0_f1 = mv0_f1;
                    r1_f1 = mv1_f1;
                    r2_f1 = mv2_f1;
                    r3_f1 = mv3_f1;
                }

                // 4. Loop de Convolução (O Coração do WaveNet):
                // Aqui cruzamos os dados do passado (kernel) com os pesos aprendidos.
                for k in 0..kernel {
                    let tap_f0 = *tap_ptrs_f0.get_unchecked(k);
                    let tap_f1 = *tap_ptrs_f1.get_unchecked(k);

                    // Localizamos onde estão os 'pesos' (conhecimento) para este neurônio específico.
                    let w_start = (b * kernel + k) * in_ch * 4;
                    let w_slice: &[[u16; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                        core::slice::from_raw_parts(ptr, in_ch)
                    };

                    let in_f0 = core::slice::from_raw_parts(tap_f0, in_ch);
                    let in_f1 = core::slice::from_raw_parts(tap_f1, in_ch);

                    // 'Dot Product Intercalado': Otimização de elite.
                    // Multiplicamos o MESMO peso por dois fragmentos de áudio diferentes ao mesmo tempo.
                    let (t_f0, t_f1) =
                        M::dot_product_4x_interleaved_dual_frame(w_slice, in_f0, in_f1);

                    // Acumulamos os resultados para ambos os frames.
                    r0_f0 += t_f0[0];
                    r1_f0 += t_f0[1];
                    r2_f0 += t_f0[2];
                    r3_f0 += t_f0[3];
                    r0_f1 += t_f1[0];
                    r1_f1 += t_f1[1];
                    r2_f1 += t_f1[2];
                    r3_f1 += t_f1[3];
                }

                // 5. Armazenamento Final:
                // Guardamos o som processado nos buffers de saída para a próxima camada da rede.
                if out_c + 3 < self.out_ch {
                    *out_f0.get_unchecked_mut(out_c) = r0_f0;
                    *out_f0.get_unchecked_mut(out_c + 1) = r1_f0;
                    *out_f0.get_unchecked_mut(out_c + 2) = r2_f0;
                    *out_f0.get_unchecked_mut(out_c + 3) = r3_f0;

                    *out_f1.get_unchecked_mut(out_c) = r0_f1;
                    *out_f1.get_unchecked_mut(out_c + 1) = r1_f1;
                    *out_f1.get_unchecked_mut(out_c + 2) = r2_f1;
                    *out_f1.get_unchecked_mut(out_c + 3) = r3_f1;
                } else {
                    let r_f0 = [r0_f0, r1_f0, r2_f0, r3_f0];
                    let r_f1 = [r0_f1, r1_f1, r2_f1, r3_f1];
                    for lane in 0..4 {
                        if out_c + lane < self.out_ch {
                            *out_f0.get_unchecked_mut(out_c + lane) = r_f0[lane];
                            *out_f1.get_unchecked_mut(out_c + lane) = r_f1[lane];
                        }
                    }
                }
            }
        }
    }

    /// Processa um bloco de amostras com mixin opcional.
    #[inline(always)]
    /// # Safety
    /// `output` deve ter tamanho pelo menos `num_frames * self.out_size`.
    /// Processa um bloco inteiro de amostras de áudio.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        // --- Processamento de Bloco (Batch Processing) ---
        // Para ganhar eficiência, não processamos uma amostra de cada vez.
        // Agrupamos o áudio em 'blocos' para que o processador possa trabalhar
        // em fluxo contínuo, sem interrupções.
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;

        // 1. Divisão em Pares:
        // Tentamos sempre processar as amostras de duas em duas (chunks de 2).
        // Isso permite usar a função 'Dual Frame' que vimos antes, economizando muita CPU.
        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            // Dividimos o pedaço de memória em dois: um para o frame atual e outro para o próximo.
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);

            // Preparamos os dados das camadas anteriores (mixin) para ambos os frames.
            let (m_f0, m_f1) = if let Some(m) = mixin {
                let start0 = i * self.out_ch;
                let end0 = (start0 + self.out_ch).min(m.len());
                let start1 = (i + 1) * self.out_ch;
                let end1 = (start1 + self.out_ch).min(m.len());
                (
                    if start0 < m.len() {
                        Some(&m[start0..end0])
                    } else {
                        None
                    },
                    if start1 < m.len() {
                        Some(&m[start1..end1])
                    } else {
                        None
                    },
                )
            } else {
                (None, None)
            };

            unsafe {
                // Chamamos a função ultra-otimizada que processa o par.
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

        // 2. Tratamento de Sobras (Remainder):
        // Se o tamanho do bloco for ímpar, sobrará uma última amostra sozinha.
        // Processamos ela individualmente aqui.
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
        // --- Modo de Frame Único ---
        // Esta função é o 'estepe'. Ela entra em ação quando não podemos
        // processar em pares (como na última amostra de um bloco com tamanho ímpar).
        let num_blocks = self.out_ch.div_ceil(4);
        debug_assert!(
            self.kernel <= MAX_KERNEL,
            "kernel {} excede MAX_KERNEL",
            self.kernel
        );
        debug_assert!(self.weights.len() >= num_blocks * 4 * self.in_ch * self.kernel);
        let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = self.kernel.min(MAX_KERNEL);

        // 1. Localização no Passado:
        // Assim como no modo Dual Frame, buscamos onde estão as amostras antigas (dilatadas).
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

        // 2. Loop de Cálculo Principal:
        // Mesmo no modo simples, ainda tentamos processar 4 canais de uma vez
        // para não perder velocidade.
        for b in 0..num_blocks {
            let out_c = b * 4;
            let mut r0;
            let mut r1;
            let mut r2;
            let mut r3;

            unsafe {
                // Inicialização com o valor base e mixin da camada anterior.
                let (mv0, mv1, mv2, mv3) = if let Some(m) = mixin {
                    if out_c + 3 < m.len() {
                        (
                            *m.get_unchecked(out_c),
                            *m.get_unchecked(out_c + 1),
                            *m.get_unchecked(out_c + 2),
                            *m.get_unchecked(out_c + 3),
                        )
                    } else {
                        {
                            let mut v = [0.0f32; 4];
                            for (i, val) in v.iter_mut().enumerate() {
                                if out_c + i < m.len() {
                                    *val = *m.get_unchecked(out_c + i);
                                }
                            }
                            (v[0], v[1], v[2], v[3])
                        }
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                if self.do_bias {
                    r0 = *self.bias.get_unchecked(out_c) + mv0;
                    r1 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1;
                    r2 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2;
                    r3 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3;
                } else {
                    r0 = mv0;
                    r1 = mv1;
                    r2 = mv2;
                    r3 = mv3;
                }

                // Aplicamos os pesos sobre o frame de áudio.
                for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                    let w_start = (b * self.kernel + k) * self.in_ch * 4;
                    let w_slice: &[[u16; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                        core::slice::from_raw_parts(ptr, self.in_ch)
                    };

                    let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);

                    // Produto Escalar Intercalado para 4 canais simultâneos.
                    let [t0, t1, t2, t3] = M::dot_product_4x_interleaved(w_slice, in_slice);
                    r0 += t0;
                    r1 += t1;
                    r2 += t2;
                    r3 += t3;
                }

                // Guardamos os resultados parciais.
                if out_c + 3 < self.out_ch {
                    *out_frame.get_unchecked_mut(out_c) = r0;
                    *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    *out_frame.get_unchecked_mut(out_c + 3) = r3;
                } else {
                    let r = [r0, r1, r2, r3];
                    for (lane, &val) in r.iter().enumerate() {
                        if out_c + lane < self.out_ch {
                            *out_frame.get_unchecked_mut(out_c + lane) = val;
                        }
                    }
                }
            }
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
        // --- Versão BF16 (Brain Floating Point) ---
        // Esta é a versão 'turbinada' do processamento. Usamos números de 16 bits (BF16)
        // em vez de 32 bits. Isso corta o uso de memória pela metade e permite que a CPU
        // processe o dobro de dados no mesmo tempo em hardwares compatíveis.
        let num_blocks = self.out_ch.div_ceil(4);
        debug_assert!(
            self.kernel <= MAX_KERNEL,
            "kernel {} excede MAX_KERNEL",
            self.kernel
        );
        debug_assert!(self.weights.len() >= num_blocks * 4 * self.in_ch * self.kernel);

        // Tap Pointers para f0 e f1 em BF16
        let mut tap_ptrs_f0 = [core::ptr::null::<u16>(); MAX_KERNEL];
        let mut tap_ptrs_f1 = [core::ptr::null::<u16>(); MAX_KERNEL];
        let k_limit = self.kernel.min(MAX_KERNEL);

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
                // 1. Inicialização do Acumulador para o Frame 0:
                // Se houver um mixin (conexão residual paralela), carregamos os 4 canais correspondentes.
                // Se a largura dos canais for divisível por 4, fazemos a leitura rápida sem verificação de limites;
                // caso contrário, usamos um array temporário com verificação segura de limites.
                let (mv0_f0, mv1_f0, mv2_f0, mv3_f0) = if let Some(m) = mixin_f0 {
                    if out_c + 3 < m.len() {
                        (
                            *m.get_unchecked(out_c),
                            *m.get_unchecked(out_c + 1),
                            *m.get_unchecked(out_c + 2),
                            *m.get_unchecked(out_c + 3),
                        )
                    } else {
                        {
                            let mut v = [0.0f32; 4];
                            for (i, val) in v.iter_mut().enumerate() {
                                if out_c + i < m.len() {
                                    *val = *m.get_unchecked(out_c + i);
                                }
                            }
                            (v[0], v[1], v[2], v[3])
                        }
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                // Adiciona o viés (bias) do neurônio ao acumulador do Frame 0, se ativo.
                if do_bias {
                    r0_f0 = *self.bias.get_unchecked(out_c) + mv0_f0;
                    r1_f0 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1_f0;
                    r2_f0 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2_f0;
                    r3_f0 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3_f0;
                } else {
                    r0_f0 = mv0_f0;
                    r1_f0 = mv1_f0;
                    r2_f0 = mv2_f0;
                    r3_f0 = mv3_f0;
                }

                // 2. Inicialização do Acumulador para o Frame 1:
                // Repete o processo de carregamento do mixin para o segundo frame da dupla.
                let (mv0_f1, mv1_f1, mv2_f1, mv3_f1) = if let Some(m) = mixin_f1 {
                    if out_c + 3 < m.len() {
                        (
                            *m.get_unchecked(out_c),
                            *m.get_unchecked(out_c + 1),
                            *m.get_unchecked(out_c + 2),
                            *m.get_unchecked(out_c + 3),
                        )
                    } else {
                        {
                            let mut v = [0.0f32; 4];
                            for (i, val) in v.iter_mut().enumerate() {
                                if out_c + i < m.len() {
                                    *val = *m.get_unchecked(out_c + i);
                                }
                            }
                            (v[0], v[1], v[2], v[3])
                        }
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                // Adiciona o viés (bias) ao acumulador do Frame 1, se ativo.
                if do_bias {
                    r0_f1 = *self.bias.get_unchecked(out_c) + mv0_f1;
                    r1_f1 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1_f1;
                    r2_f1 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2_f1;
                    r3_f1 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3_f1;
                } else {
                    r0_f1 = mv0_f1;
                    r1_f1 = mv1_f1;
                    r2_f1 = mv2_f1;
                    r3_f1 = mv3_f1;
                }

                // 3. Loop sobre os Taps da Convolução Dilatada:
                // Para cada atraso do kernel, buscamos os buffers correspondentes a ambos os frames
                // e realizamos um produto escalar intercalado simultâneo (dual-frame).
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

                    // Produto Escalar Intercalado BF16 em Frame Duplo (Dual-Frame):
                    // Executa a inferência de 4 saídas ao mesmo tempo para ambos os frames, aproveitando o
                    // alinhamento espacial de dados em cache e instruções de multiplicação/acumulação SIMD.
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

                if out_c + 3 < self.out_ch {
                    *out_f0.get_unchecked_mut(out_c) = r0_f0;
                    *out_f0.get_unchecked_mut(out_c + 1) = r1_f0;
                    *out_f0.get_unchecked_mut(out_c + 2) = r2_f0;
                    *out_f0.get_unchecked_mut(out_c + 3) = r3_f0;

                    *out_f1.get_unchecked_mut(out_c) = r0_f1;
                    *out_f1.get_unchecked_mut(out_c + 1) = r1_f1;
                    *out_f1.get_unchecked_mut(out_c + 2) = r2_f1;
                    *out_f1.get_unchecked_mut(out_c + 3) = r3_f1;
                } else {
                    let r_f0 = [r0_f0, r1_f0, r2_f0, r3_f0];
                    let r_f1 = [r0_f1, r1_f1, r2_f1, r3_f1];
                    for lane in 0..4 {
                        if out_c + lane < self.out_ch {
                            *out_f0.get_unchecked_mut(out_c + lane) = r_f0[lane];
                            *out_f1.get_unchecked_mut(out_c + lane) = r_f1[lane];
                        }
                    }
                }
            }
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
        // --- Processamento de Bloco BF16 (O Ápice da Performance) ---
        // Este é o 'Caminho de Ouro' para modelos pesados de WaveNet.
        // Ele combina o processamento em blocos (batches) com o formato de memória
        // compacto BF16, permitindo que a CPU processe o áudio com o mínimo de esforço.
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;

        // 1. Divisão em Pares (Chunking):
        // Continuamos usando a estratégia de processar de dois em dois para
        // ativar as funções 'Dual Frame' otimizadas para BF16.
        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);
            let (m_f0, m_f1) = if let Some(m) = mixin {
                let start0 = i * self.out_ch;
                let end0 = (start0 + self.out_ch).min(m.len());
                let start1 = (i + 1) * self.out_ch;
                let end1 = (start1 + self.out_ch).min(m.len());
                (
                    if start0 < m.len() {
                        Some(&m[start0..end0])
                    } else {
                        None
                    },
                    if start1 < m.len() {
                        Some(&m[start1..end1])
                    } else {
                        None
                    },
                )
            } else {
                (None, None)
            };

            unsafe {
                // Chamada da versão 'turbinada' BF16.
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

        // 2. Tratamento de Sobras (Remainder):
        // Caso o bloco tenha tamanho ímpar, cuidamos da última amostra usando BF16.
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
        // --- Modo BF16 de Frame Único ---
        // Este é o 'plano B' do caminho de alta performance. Ele entra em ação
        // para processar amostras que sobraram de blocos ímpares, usando
        // a economia de memória do formato BF16.
        let num_blocks = self.out_ch.div_ceil(4);
        debug_assert!(
            self.kernel <= MAX_KERNEL,
            "kernel {} excede MAX_KERNEL",
            self.kernel
        );
        debug_assert!(self.weights.len() >= num_blocks * 4 * self.in_ch * self.kernel);
        let mut tap_ptrs = [core::ptr::null::<u16>(); MAX_KERNEL];
        let k_limit = self.kernel.min(MAX_KERNEL);

        // 1. Localização com Dilatação (BF16):
        // Buscamos o passado no buffer de 16 bits.
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

        // 2. Loop de Cálculo 4x (BF16):
        // Mantemos a eficiência processando 4 canais de uma vez,
        // mas agora focados em uma única amostra de tempo.
        for b in 0..num_blocks {
            let out_c = b * 4;
            let mut r0;
            let mut r1;
            let mut r2;
            let mut r3;

            unsafe {
                let (mv0, mv1, mv2, mv3) = if let Some(m) = mixin {
                    if out_c + 3 < m.len() {
                        (
                            *m.get_unchecked(out_c),
                            *m.get_unchecked(out_c + 1),
                            *m.get_unchecked(out_c + 2),
                            *m.get_unchecked(out_c + 3),
                        )
                    } else {
                        {
                            let mut v = [0.0f32; 4];
                            for (i, val) in v.iter_mut().enumerate() {
                                if out_c + i < m.len() {
                                    *val = *m.get_unchecked(out_c + i);
                                }
                            }
                            (v[0], v[1], v[2], v[3])
                        }
                    }
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                };

                if self.do_bias {
                    r0 = *self.bias.get_unchecked(out_c) + mv0;
                    r1 = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mv1;
                    r2 = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mv2;
                    r3 = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mv3;
                } else {
                    r0 = mv0;
                    r1 = mv1;
                    r2 = mv2;
                    r3 = mv3;
                }

                for (k, &tap_ptr) in tap_ptrs.iter().enumerate().take(self.kernel) {
                    let w_start = (b * self.kernel + k) * self.in_ch * 4;
                    let w_slice: &[[u16; 4]] = {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                        core::slice::from_raw_parts(ptr, self.in_ch)
                    };

                    let in_slice = core::slice::from_raw_parts(tap_ptr, self.in_ch);

                    // Produto Escalar Intercalado BF16 para 4 canais.
                    let [t0, t1, t2, t3] = M::dot_product_4x_interleaved_bf16(w_slice, in_slice);
                    r0 += t0;
                    r1 += t1;
                    r2 += t2;
                    r3 += t3;
                }

                if out_c + 3 < self.out_ch {
                    *out_frame.get_unchecked_mut(out_c) = r0;
                    *out_frame.get_unchecked_mut(out_c + 1) = r1;
                    *out_frame.get_unchecked_mut(out_c + 2) = r2;
                    *out_frame.get_unchecked_mut(out_c + 3) = r3;
                } else {
                    let r = [r0, r1, r2, r3];
                    for (lane, &val) in r.iter().enumerate() {
                        if out_c + lane < self.out_ch {
                            *out_frame.get_unchecked_mut(out_c + lane) = val;
                        }
                    }
                }
            }
        }
    }
}
