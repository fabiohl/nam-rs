// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Malha CNN Causal Estática para inferência WaveNet (Design Orientado a Dados, SoA).
//!
//! Todas as estruturas utilizam `Const Generics` nas dimensões matemáticas e vetores pré-alocados
//! garantindo uma política de instanciamento estrito (Zero-Allocation durante processamento).
//! As loops dinâmicos resolvem cálculos em sequências FMA determinísticas via AVX2.

//! Módulo de Inferência WaveNet (Arquitetura Causal Dilatada).

use crate::dsp::vring::VirtualRingBuffer;
use crate::math::simd::SimdMath;

/// Máximo de frames a processar em um pulso do callback.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Padding temporal circular das memórias no framework de Ring Buffers.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;

/// Convolução Causal Dilatada (WaveNet Conv1D).
#[derive(Clone)]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Matriz achatada de pesos do tamanho OUT * K * IN.
    pub weights: Vec<u16>,
    /// Viés causal, atrelado se do_bias for verdadeiro. Total: OUT.
    pub bias: Vec<f32>,
    /// Determina se o array de bias deve ser somado.
    pub do_bias: bool,
    /// Fator de diluição no eixo temporal causacional (Ex: 1, 2, 4.. 512).
    pub dilation: usize,
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
        // [PASSO 1: Inicialização do Acumulador]
        if self.do_bias {
            out_frame.copy_from_slice(&self.bias[0..OUT]);
        } else {
            out_frame.fill(0.0);
        }

        // [PASSO 2: Iteração do Kernel (Receptive Field)]
        for k in 0..K {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let current_frame_idx = (frame_idx as isize) + offset;
            let in_slice_start = (current_frame_idx as usize) * IN;

            // Prefetch adaptativo com lookahead para cobrir latência de memória.
            // Avança 16 floats (2 vetores AVX2) à frente do ponteiro de leitura atual.
            let lookahead_offset = 16;
            let prefetch_ptr =
                unsafe { layer_buffer.as_ptr().add(in_slice_start + lookahead_offset) };
            unsafe {
                crate::math::simd::adaptive_prefetch_f32(prefetch_ptr, self.dilation);
            }

            let in_slice =
                unsafe { layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN) };

            let mut out_c = 0;
            let num_blocks = OUT / 4;

            // [T19] Loop Interleaved: Processa 4 canais de saída por iteração
            // com um único stream de leitura de pesos (contiguidade máxima).
            for b in 0..num_blocks {
                let w_start = b * K * IN * 4 + k * IN * 4;
                let w_slice: &[[u16; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let [r0, r1, r2, r3] = unsafe { M::dot_product_4x_interleaved(w_slice, in_slice) };

                unsafe {
                    *out_frame.get_unchecked_mut(out_c) += r0;
                    *out_frame.get_unchecked_mut(out_c + 1) += r1;
                    *out_frame.get_unchecked_mut(out_c + 2) += r2;
                    *out_frame.get_unchecked_mut(out_c + 3) += r3;
                }
                out_c += 4;
            }

            // Loop de Cauda (Remainder): Processa canais restantes se OUT % 4 != 0
            while out_c < OUT {
                let w_start = out_c * K * IN + k * IN;
                let w = unsafe { self.weights.get_unchecked(w_start..w_start + IN) };
                let r = unsafe { M::dot_product(in_slice, w) };
                unsafe {
                    *out_frame.get_unchecked_mut(out_c) += r;
                }
                out_c += 1;
            }
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
        if self.do_bias {
            out_frame.copy_from_slice(&self.bias[0..OUT]);
        } else {
            out_frame.fill(0.0);
        }

        for k in 0..K {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let current_frame_idx = (frame_idx as isize) + offset;
            let in_slice_start = (current_frame_idx as usize) * IN;

            // Prefetch adaptativo para o próximo "tap" do kernel temporal (dilation-aware).
            if k + 1 < K {
                let prefetch_ptr =
                    unsafe { layer_buffer.as_ptr().add(in_slice_start + self.dilation) };
                unsafe {
                    crate::math::simd::adaptive_prefetch_f32(prefetch_ptr.cast(), self.dilation);
                }
            }

            let in_slice =
                unsafe { layer_buffer.get_unchecked(in_slice_start..in_slice_start + IN) };

            let mut out_c = 0;
            let num_blocks = OUT / 4;

            // [T19] Loop Interleaved BF16: Eficiência máxima para VNNI
            for b in 0..num_blocks {
                let w_start = b * K * IN * 4 + k * IN * 4;
                let w_slice: &[[u16; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [u16; 4];
                    core::slice::from_raw_parts(ptr, IN)
                };

                let [r0, r1, r2, r3] =
                    unsafe { M::dot_product_4x_interleaved_bf16(w_slice, in_slice) };

                unsafe {
                    *out_frame.get_unchecked_mut(out_c) += r0;
                    *out_frame.get_unchecked_mut(out_c + 1) += r1;
                    *out_frame.get_unchecked_mut(out_c + 2) += r2;
                    *out_frame.get_unchecked_mut(out_c + 3) += r3;
                }
                out_c += 4;
            }

            // Loop de Cauda BF16
            while out_c < OUT {
                let w_start = out_c * K * IN + k * IN;
                let w = unsafe { self.weights.get_unchecked(w_start..w_start + IN) };
                let r = unsafe { M::dot_product_bf16(in_slice, w) };
                unsafe {
                    *out_frame.get_unchecked_mut(out_c) += r;
                }
                out_c += 1;
            }
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
}

/// Camada Densa 1x1 baseada num Matmul vetorizado linear.
#[derive(Clone)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    /// Matriz de pesos lineares (OUT * IN).
    pub weights: Vec<u16>,
    /// Condição temporal para o deslocador de tensor bias.
    pub bias: Vec<f32>,
    /// Determina se o array de bias deve ser somado.
    pub do_bias: bool,
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    /// Processa um único frame do Dense Layer acumulando com o output atual (otimizado com FMA 4x).
    ///
    /// # Safety
    /// Depende dinamicamente da trait `SimdMath`.
    #[inline(always)]
    pub unsafe fn process_acc_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// Executa a projeção fundida (W*in + bias) somando ao buffer de saída (Residual Fusion).
    ///
    /// # Safety
    /// O chamador deve garantir que `in_frame` e `out_frame` tenham tamanhos compatíveis com `IN` e `OUT`.
    #[inline(always)]
    pub unsafe fn process_fused<M: SimdMath>(&self, in_frame: &[f32], out_frame: &mut [f32]) {
        unsafe {
            M::fused_add_gemv(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// Processa um único frame substituindo o buffer existente.
    ///
    /// # Safety
    /// O chamador deve garantir que `in_frame` e `out_frame` tenham tamanhos compatíveis com `IN` e `OUT`.
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        in_frame: &[f32],
        out_frame: &mut [f32],
    ) {
        unsafe {
            M::gemv_overwrite(in_frame, &self.weights, &self.bias, out_frame, self.do_bias);
        }
    }

    /// Processa o Dense acumulando com o estado corrente de output.
    ///
    /// # Safety
    /// O chamador deve garantir que `input` e `output` tenham tamanhos compatíveis com `IN` e `OUT` e `num_frames`.
    #[inline(always)]
    pub unsafe fn process_acc_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            let in_slice = unsafe { input.get_unchecked(i * IN..(i + 1) * IN) };
            let out_slice = unsafe { output.get_unchecked_mut(i * OUT..(i + 1) * OUT) };

            unsafe {
                M::fused_add_gemv(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
            }
        }
    }

    #[inline(always)]
    /// Processa bloco iterativo substituindo (OVERWRITE) os valores passados em vez de acumular.
    ///
    /// # Safety
    /// O chamador deve garantir que `input` e `output` tenham tamanhos compatíveis com `IN` e `OUT` e `num_frames`.
    pub unsafe fn process_block<M: SimdMath>(
        &self,
        input: &[f32],
        output: &mut [f32],
        num_frames: usize,
    ) {
        for i in 0..num_frames {
            let in_slice = unsafe { input.get_unchecked(i * IN..(i + 1) * IN) };
            let out_slice = unsafe { output.get_unchecked_mut(i * OUT..(i + 1) * OUT) };

            unsafe {
                M::gemv_overwrite(in_slice, &self.weights, &self.bias, out_slice, self.do_bias);
            }
        }
    }

    /// Processa a camada densa usando BF16.
    ///
    /// # Safety
    /// O chamador deve garantir que `input` e `output` tenham tamanhos
    /// compatíveis com as dimensões `IN` e `OUT` da camada, e que as instruções
    /// SIMD solicitadas pelo despachante `M` estejam disponíveis.
    pub unsafe fn process_bf16<M: SimdMath>(&self, input: &[u16], output: &mut [f32]) {
        let num_frames = output.len() / OUT;
        let mut out_c = 0;

        while out_c + 4 <= OUT {
            let w0 = unsafe { self.weights.get_unchecked(out_c * IN..(out_c + 1) * IN) };
            let w1 = unsafe {
                self.weights
                    .get_unchecked((out_c + 1) * IN..(out_c + 2) * IN)
            };
            let w2 = unsafe {
                self.weights
                    .get_unchecked((out_c + 2) * IN..(out_c + 3) * IN)
            };
            let w3 = unsafe {
                self.weights
                    .get_unchecked((out_c + 3) * IN..(out_c + 4) * IN)
            };

            let b0 = if self.do_bias { self.bias[out_c] } else { 0.0 };
            let b1 = if self.do_bias {
                self.bias[out_c + 1]
            } else {
                0.0
            };
            let b2 = if self.do_bias {
                self.bias[out_c + 2]
            } else {
                0.0
            };
            let b3 = if self.do_bias {
                self.bias[out_c + 3]
            } else {
                0.0
            };

            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let [r0, r1, r2, r3] = unsafe { M::dot_product_bf16_4x(w0, w1, w2, w3, in_frame) };

                unsafe {
                    *output.get_unchecked_mut(i * OUT + out_c) = r0 + b0;
                    *output.get_unchecked_mut(i * OUT + out_c + 1) = r1 + b1;
                    *output.get_unchecked_mut(i * OUT + out_c + 2) = r2 + b2;
                    *output.get_unchecked_mut(i * OUT + out_c + 3) = r3 + b3;
                }
            }
            out_c += 4;
        }

        while out_c < OUT {
            let weight_slice = unsafe { self.weights.get_unchecked(out_c * IN..out_c * IN + IN) };
            let bias = if self.do_bias { self.bias[out_c] } else { 0.0 };
            for i in 0..num_frames {
                let in_frame = unsafe { input.get_unchecked(i * IN..i * IN + IN) };
                let sum = unsafe { M::dot_product_bf16(in_frame, weight_slice) };
                unsafe {
                    *output.get_unchecked_mut(i * OUT + out_c) = sum + bias;
                }
            }
            out_c += 1;
        }
    }
}

/// Célula Convolucional Completa (WaveNet Layer).
#[derive(Clone)]
pub struct WaveNetLayer<const COND: usize, const CH: usize, const K: usize> {
    /// Malha de convolução Causal 1D paramétrica dilatada desta camada.
    pub conv1d: Conv1d<CH, CH, K>,
    /// Rede em injeção de mistura Condisional.
    pub input_mixin: DenseLayer<COND, CH>,
    /// Transformação afim linear de descompressão 1x1 da camada.
    pub one_by_one: DenseLayer<CH, CH>,
}

/// Contexto de processamento para otimizar a passagem de parâmetros no hot-path da WaveNet.
pub struct WavenetProcessContext<'a> {
    /// Buffer de condicionamento (sidechain).
    pub condition: &'a [f32],
    /// Acumulador para a cabeça de saída.
    pub head_input: &'a mut [f32],
    /// Buffer de saída da camada.
    pub output: &'a mut [f32],
    /// Buffer circular (fita de retardo) em F32.
    pub layer_buffer: &'a [f32],
    /// Buffer circular (fita de retardo) em BF16.
    pub layer_buffer_bf16: &'a [u16],
    /// Índice inicial no buffer circular.
    pub buffer_start: usize,
    /// Número de frames a processar.
    pub num_frames: usize,
}

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processa uma camada integral do WaveNet, iterando `FastMath` em AVX2.
    ///
    /// # Safety
    /// Despacho matemático via ponteiro para funções intrínsecas inlined.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(&self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            head_input,
            output,
            layer_buffer,
            layer_buffer_bf16,
            buffer_start,
            num_frames,
        } = ctx;

        unsafe {
            // Buffer transiente em stack para condicionamento em BF16 se necessário.
            let mut cond_bf16 = [0u16; COND];
            if M::IS_BF16 {
                M::f32_to_bf16(condition, &mut cond_bf16);
            }

            // [PASSO 2: Condicionamento (Input Mixin)]
            // Buffer stack de 1024 f32 = WAVENET_MAX_NUM_FRAMES(64) × max_CH(16).
            // O Rust estável não permite `CH` em expressões const, então usamos
            // o limite superior. O debug_assert abaixo captura qualquer topologia
            // futura que exceda este tamanho (ex: CH=32, num_frames=64 → 2048 > 1024).
            debug_assert!(
                num_frames * CH <= 1024,
                "process_block_internal: num_frames*CH ({}) excede o buffer stack (1024)",
                num_frames * CH,
            );
            let mut mixin_out = [0.0f32; 1024];
            let mixin_out_slice = &mut mixin_out[..num_frames * CH];
            if M::IS_BF16 {
                self.input_mixin
                    .process_bf16::<M>(&cond_bf16, mixin_out_slice);
            } else {
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);
            }

            // [T20] "Ahead-of-Time Conditioning": Otimização de Fusão Temporal
            // Buffer temporário na stack para armazenar os resultados intermediários (Conv1D + Mixin)
            // antes da ativação. CH=16 * MAX_FRAMES=64 = 1024 elementos (4KB).
            let mut conv_plus_mixin = [0.0f32; 1024];
            let conv_slice = &mut conv_plus_mixin[..num_frames * CH];

            // [FASE 1: Linear - Conv1D + Mixin]
            // Iteramos sobre todos os frames apenas para as operações lineares.
            // Isso maximiza o reuso dos pesos da Conv1D (que são grandes) na Cache L1.
            for i in 0..num_frames {
                let out_ptr = conv_slice.as_mut_ptr().add(i * CH);
                let out_frame = core::slice::from_raw_parts_mut(out_ptr, CH);

                if M::IS_BF16 {
                    self.conv1d.process_single_frame_bf16::<M>(
                        layer_buffer_bf16,
                        out_frame,
                        buffer_start + i,
                    );
                } else {
                    self.conv1d.process_single_frame::<M>(
                        layer_buffer,
                        out_frame,
                        buffer_start + i,
                    );
                }

                // Soma o Mixin (Condicionamento pré-calculado)
                let mix_idx = i * CH;
                let mixin_slice = mixin_out.get_unchecked(mix_idx..mix_idx + CH);
                for (o, m) in out_frame.iter_mut().zip(mixin_slice) {
                    *o += *m;
                }
            }

            // [FASE 2: Ativação em Lote]
            // Aplicamos a Tanh em todo o bloco de uma vez só.
            // O throughput SIMD é muito maior processando 1024 elementos contíguos
            // do que 16 elementos por chamada.
            M::activation_tanh_block(conv_slice);

            // [FASE 3: Não-Linear & Saída - Head Update + 1x1 Residual]
            // Agora processamos os canais de saída. Os pesos da 1x1 entram na cache aqui.
            for i in 0..num_frames {
                let lb_start = (buffer_start + i) * CH;
                let conv_idx = i * CH;
                let temp = conv_slice.get_unchecked(conv_idx..conv_idx + CH);

                // Head Update (Skip-Connection)
                let h_idx = i * CH;
                M::accumulate_head(head_input.get_unchecked_mut(h_idx..h_idx + CH), temp);

                // Projeção 1x1 + Soma Residual
                let out_ptr = output.as_mut_ptr().add(i * CH);
                let out_slice = core::slice::from_raw_parts_mut(out_ptr, CH);
                out_slice.copy_from_slice(layer_buffer.get_unchecked(lb_start..lb_start + CH));

                self.one_by_one.process_fused::<M>(temp, out_slice);
            }
        }
    }
}

/// Gerencia a memória buffer de uma célula WaveNet.
///
/// Alinhamento de 64 bytes (uma cache line) garante que `buffer_start` e
/// `receptive_field_size` não compartilhem cache line com o estado da camada
/// adjacente ao iterar `states_ptr.add(i)` no hot-path do `process()`.
///
/// # Rewind amortizado
///
/// O `rewind_buffer` executa `copy_within` de `receptive_field_size × CH` floats
/// a cada ~24 × 64 = 1536 amostras processadas (~32 ms a 48 kHz). É um custo
/// amortizado aceitável (~6–10 µs uma vez a cada ~24 callbacks). Spikes
/// observados em benchmarks refletem esse evento caindo dentro do intervalo
/// medido; em produção o efeito é diluído no jitter total do sistema.
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
        // Se o ponteiro avançar demais, nós simplesmente subtraímos o tamanho base (N).
        // Como o buffer é mapeado em 2*N, e buffer_start está sempre entre N e 2*N-1,
        // o acesso a [buffer_start - receptive_field] é sempre seguro (>= 0).
        if self.buffer_start >= buffer_frames * 2 {
            self.buffer_start -= buffer_frames;
        }
    }
}

/// Unidade de Múltiplos Layers agrupados do WaveNet.
pub struct WaveNetLayerArray<
    const IN: usize,
    const COND: usize,
    const CH: usize,
    const K: usize,
    const HEAD: usize,
> {
    /// Vec com a topologia estrutural (comprimento define blocos).
    pub layers: Vec<WaveNetLayer<COND, CH, K>>,
    /// Estados do RingBuffer, um para cada Layer do sistema.
    pub states: Vec<WaveNetLayerState>,
    /// Abertura tensorial inicial `Dense`.
    pub rechannel: DenseLayer<IN, CH>,
    /// Fechamento tensorial final gerando projeção Head.
    pub head_rechannel: DenseLayer<CH, HEAD>,

    /// Array temporário pre-alocado para saídas.
    pub array_outputs: std::vec::Vec<f32>,
    /// Acumulador intermediário CH-sized para contribuições das camadas antes da projeção Head.
    pub head_accum: std::vec::Vec<f32>,
    /// Memória alocada da projeção Linear global (HEAD-sized).
    pub head_outputs: std::vec::Vec<f32>,
    /// Tamanho do campo dimensional (receptive field global) para roteamentos.
    pub receptive_field_size: usize,
}

impl<const IN: usize, const COND: usize, const CH: usize, const K: usize, const HEAD: usize>
    WaveNetLayerArray<IN, COND, CH, K, HEAD>
{
    /// Processamento central da Array. Totalmente blindado contra alocações.
    ///
    /// # Safety
    /// Ponteiros de states iteram internamente sem bounds checks.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();

        // [PASSO 1: Zero-Acumulador]
        // Zera o acumulador das saídas "Skip Connections" (Head) para este bloco de frames.
        // É essencial pois cada camada do array somará sua contribuição aqui.
        self.head_accum[0..num_frames * CH].fill(0.0);

        unsafe {
            // [PASSO 2: Abertura Dimensional (Rechannel)]
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * CH],
                num_frames,
            );

            if M::IS_BF16 {
                M::f32_to_bf16(
                    &state_0.layer_buffer[start..start + num_frames * CH],
                    &mut state_0.layer_buffer_bf16[start..start + num_frames * CH],
                );
            }

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            // [PASSO 3: Cascata de Inferência das Camadas]
            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..num_frames * CH],
                        output: &mut self.array_outputs[0..num_frames * CH],
                        layer_buffer: &current_state.layer_buffer,
                        layer_buffer_bf16: &current_state.layer_buffer_bf16,
                        buffer_start: current_state.buffer_start,
                        num_frames,
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..num_frames * CH],
                        output: &mut next_state.layer_buffer
                            [next_start..next_start + num_frames * CH],
                        layer_buffer: &current_state.layer_buffer,
                        layer_buffer_bf16: &current_state.layer_buffer_bf16,
                        buffer_start: current_state.buffer_start,
                        num_frames,
                    });

                    if M::IS_BF16 {
                        M::f32_to_bf16(
                            &next_state.layer_buffer[next_start..next_start + num_frames * CH],
                            &mut next_state.layer_buffer_bf16
                                [next_start..next_start + num_frames * CH],
                        );
                    }
                }

                current_state.advance_frames(num_frames, CH);
            }

            // [PASSO 5: Fechamento Dimensional (Head Rechannel)]
            // A matriz densa afunila o acumulador (soma das skip-connections de todas as camadas,
            // de tamanho `CH`) para uma menor dimensão `HEAD` (ex: 16 -> 8 ou 16 -> 1).
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * CH],
                &mut self.head_outputs[0..num_frames * HEAD],
                num_frames,
            );
        }
    }

    /// Processa dados no modo Pre-warm para inicializar e estabilizar a memória temporal.
    ///
    /// [EXPLICAÇÃO CIENTÍFICA]
    /// Redes neurais de convolução causal como WaveNet possuem um estado interno que depende
    /// ativamente de N passos no passado (Receptive Field). Ao carregar um modelo novo, a
    /// memória da rede (Ring Buffers) alocada possui "zeros" puritanos ou lixo computacional.
    /// O Pre-warm alimenta um sinal inerte (Silêncio Absoluto) contínuo para a rede de modo a
    /// preencher toda a janela do passado. Os transientes resultantes desse cold-start "escoam"
    /// silenciosamente para o limbo, garantindo que a primeira amostra de áudio ao ligar a placa
    /// soe orgânica e estável, sem estalos ou cliques (clicks/pops).
    #[inline(always)]
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    pub unsafe fn prewarm_internal<M: SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();

        // [PASSO 1: Zero-Acumulador]
        // Preparativos de warm-up. O acumulador de skip-connections é limpo para este 1 único
        // frame iterativo (num_frames = 1).
        self.head_accum[0..CH].fill(0.0);

        unsafe {
            // [PASSO 2: Abertura Dimensional Simulada]
            // Expande os `layer_inputs` estáticos (geralmente [0.0]) de mono para o barramento de `CH` canais
            // e escreve na primeira camada temporal do modelo (`state_0`).
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + CH],
                1,
            );

            if M::IS_BF16 {
                M::f32_to_bf16(
                    &state_0.layer_buffer[start..start + CH],
                    &mut state_0.layer_buffer_bf16[start..start + CH],
                );
            }

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                // [PASSO 3: Propagação do Estado Estático]
                // DIFERENCIAL IMPORTANTE: Em vez de avançar o ponteiro (como no áudio em tempo real),
                // nós preenchemos todo o histórico (Receptive Field) com o valor recém-processado.
                // Com VirtualRingBuffer, como buffer_start >= N, podemos recuar linearmente com segurança.
                let start_idx = current_state.buffer_start * CH;
                for offset in 1..=current_state.receptive_field_size {
                    let dst_idx = (current_state.buffer_start - offset) * CH;
                    for j in 0..CH {
                        current_state.layer_buffer[dst_idx + j] =
                            current_state.layer_buffer[start_idx + j];
                        current_state.layer_buffer_bf16[dst_idx + j] =
                            current_state.layer_buffer_bf16[start_idx + j];
                    }
                }

                // [PASSO 4: Avaliação Numérica "Fantasma"]
                // Efetua um ciclo de avaliação completo do tensor da camada, propagando o sinal nulo
                // pela rede. Embora a entrada seja silêncio, camadas possuem matrizes de Bias que
                // agregam valor real, ou seja, o "silêncio" de saída da rede *não* é exatamente zero.
                if i == last_layer {
                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..CH],
                        output: &mut self.array_outputs[0..CH],
                        layer_buffer: &current_state.layer_buffer,
                        layer_buffer_bf16: &current_state.layer_buffer_bf16,
                        buffer_start: current_state.buffer_start,
                        num_frames: 1,
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..CH],
                        output: &mut next_state.layer_buffer[next_start..next_start + CH],
                        layer_buffer: &current_state.layer_buffer,
                        layer_buffer_bf16: &current_state.layer_buffer_bf16,
                        buffer_start: current_state.buffer_start,
                        num_frames: 1,
                    });

                    if M::IS_BF16 {
                        M::f32_to_bf16(
                            &next_state.layer_buffer[next_start..next_start + CH],
                            &mut next_state.layer_buffer_bf16[next_start..next_start + CH],
                        );
                    }
                }
            }

            // [PASSO 5: Fechamento]
            // Resolve a passagem do frame inicial nulo pela camada densa de fechamento (Head Rechannel).
            // Ao final deste fluxo, a arquitetura está purgada, alinhada e perfeitamente equilibrada
            // no ponto numérico inerte do amplificador real. Pronta para processar sinal musical.
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..CH],
                &mut self.head_outputs[0..HEAD],
                1,
            );
        }
    }
}

/// Modelo Completo do WaveNet contendo Dois Blocos de Layer Arrays heterogêneos.
///
/// **Referência Científica:** van den Oord, A., et al. (2016). *"WaveNet: A Generative Model for Raw Audio."* DeepMind.
///
/// `CH` = canais da Array1 (layer 0 do JSON, ex: 16 para Standard)
/// `K`  = kernel size (sempre 3)
/// `HEAD` = head_size da Array1 = canais da Array2 (ex: 8 para Standard)
///
/// Array2 usa `HEAD` canais e projeta para 1 saída (`HEAD2=1`),
/// seguindo o padrão C++: `WaveNetLayerArrayT<CH, 1, 1, HEAD, K, Dilations, true>`.
pub struct WaveNetModel<const CH: usize, const K: usize, const HEAD: usize> {
    /// Array interno 01: IN=1, COND=1, CH canais, HEAD saídas, sem HeadBias.
    pub array1: WaveNetLayerArray<1, 1, CH, K, HEAD>,
    /// Array interno 02: IN=CH, COND=1, HEAD canais, 1 saída, com HeadBias.
    pub array2: WaveNetLayerArray<CH, 1, HEAD, K, 1>,
    /// Escala de compensação da voltagem final (Target Output Scale).
    pub head_scale: f32,
    /// Maior buffer circular requerido na raiz temporal do Kernel.
    pub receptive_field_size: usize,
}

impl<const CH: usize, const K: usize, const HEAD: usize> WaveNetModel<CH, K, HEAD> {
    /// Resolve o forward total e produz amostras de onda em zero alocação (DSP).
    ///
    /// Combina as saídas de ambas as arrays: `sum(head1) + sum(head2)` × `head_scale`.
    ///
    /// **Para Cientistas e Devs:** Aqui acontece o "pulo do gato" da performance (SIMD Dispatch).
    /// Em vez de usarmos `if/else` lentos a cada frame para checar a CPU (AVX2 vs AVX-512),
    /// a macro `dispatch_simd!` avalia o hardware uma única vez e "teletransporta" a execução
    /// para uma versão clonada (monomorfizada) desta função estritamente otimizada para o seu processador.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        unsafe { crate::math::simd::dispatch_simd!(self, process_internal, input, output) };
    }

    #[inline(always)]
    /// Rotina genérica e veloz que implementa a rede neural (WaveNet).
    /// A restrição `<M: SimdMath>` obriga o compilador a gerar o assembly focado em
    /// registradores grandes (256-bit ou 512-bit) sem ramificações (branchless).
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let total_frames = input.len();
        if total_frames == 0 {
            return;
        }

        let mut pos = 0;
        // [PROCESSAMENTO EM CHUNKS (BLOCOS)]
        // Para manter invariantes de zero-allocation (sem alocar vetores RAM temporários)
        // e respeitar a hierarquia restrita de Cache L1/L2, limitamos o processamento
        // a `WAVENET_MAX_NUM_FRAMES` (tipicamente 64 amostras) por vez.
        // Esse loop iterará até consumir todo o buffer (ex: 256, 512, 1024 frames).
        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            unsafe {
                // [PASSO 1: Array1 Forward]
                // Condicionamento e Input (1D: 1 canal) -> formatado como blocos de IN frames.
                // Na topologia NAM padrão, esta Array realiza convoluções usando dilatações enormes
                // (ex: de 1 a 512, 1 a 512 sucessivamente) para capturar sub-graves de amplificadores.
                // Seu output entra em `array1.array_outputs` e os skips em `array1.head_outputs`.
                self.array1
                    .process_block_internal::<M>(in_slice, in_slice, num_frames);

                // [PASSO 2: Array2 Forward]
                // A segunda array atua tipicamente como uma camada perceptron de fechamento
                // (dimensões menores, dilatações apenas de 1, processando o "mix" vindo da Array1).
                let array1_outputs = &self.array1.array_outputs[0..num_frames * CH];
                self.array2
                    .process_block_internal::<M>(array1_outputs, in_slice, num_frames);
            }

            // [PASSO 3: Soma das Skips + Escala Final SIMD]
            // Somatório SIMD das projeções Head de ambas as arrays e escala pela `head_scale`.
            for i in 0..num_frames {
                let head_ptr = self.array1.head_outputs.as_ptr();
                let head1_sum = unsafe { M::horizontal_sum::<HEAD>(head_ptr.add(i * HEAD)) };

                // O head final do Array2 gera a amostra float. Somamos ao mix da Array1.
                let final_sum = head1_sum + self.array2.head_outputs[i]; // HEAD2=1

                // Escrevemos a tensão elétrica analógica reconstruída no array de áudio local.
                output[pos + i] = final_sum * self.head_scale;
            }
            pos += num_frames;
        }
    }

    /// Estabiliza o modelo processando silêncio (Zero Input) para aquecimento (Pre-warm).
    ///
    /// O dispatch AVX-512 vs AVX2 é feito via `SimdMathConfig::get().is_avx512` —
    /// leitura atômica Relaxed de um `LazyLock` inicializado no startup, sem chamar
    /// `is_x86_feature_detected!` a cada invocação (cold-path, mas consistente com
    /// o padrão de dispatch do restante do codebase).
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::simd::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Prewarm estritamente otimizado para a arquitetura AVX-512.
    ///
    /// # Safety
    /// Exige processador suportado (AVX-512).
    #[target_feature(enable = "avx512f,avx512vl")]
    pub unsafe fn prewarm_avx512(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::simd::Avx512Math>() };
    }

    /// Prewarm estritamente otimizado para arquitetura AVX2.
    ///
    /// # Safety
    /// Exige processador x86-64-v3 (AVX2).
    pub unsafe fn prewarm_avx2(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::simd::Avx2Math>() };
    }

    #[inline(always)]
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    unsafe fn prewarm_internal<M: SimdMath>(&mut self) {
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        unsafe {
            self.array1
                .prewarm_internal::<M>(&layer_inputs_1, &condition);
        }
        let array1_outputs = &self.array1.array_outputs[0..CH];
        unsafe {
            self.array2
                .prewarm_internal::<M>(array1_outputs, &condition);
        }
    }
}

#[cfg(test)]
#[path = "wavenet_test.rs"]
mod wavenet_test;
