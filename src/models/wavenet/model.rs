use super::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext};
use super::conv1d::Conv1d;
use super::dense::DenseLayer;
use crate::math::common::{AlignedVec, SimdMath};
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

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

// (Contexto de processamento movido para wavenet_common.rs)

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processa uma camada integral do WaveNet, iterando `FastMath` em AVX2.
    ///
    /// # Safety
    /// Despacho matemático via ponteiro para funções intrínsecas inlined.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath>(&self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            condition_bf16,
            head_input,
            output,
            mut output_bf16,
            layer_buffer,
            layer_buffer_bf16,
            buffer_start,
            num_frames,
            ..
        } = ctx;

        unsafe {
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
                    .process_bf16::<M>(condition_bf16, mixin_out_slice);
            } else {
                self.input_mixin
                    .process_block::<M>(condition, mixin_out_slice, num_frames);
            }

            // "Ahead-of-Time Conditioning": Otimização de Fusão Temporal Buffer
            // temporário na stack para armazenar os resultados intermediários (Conv1D + Mixin)
            // antes da ativação. CH=16 * MAX_FRAMES=64 = 1024 elementos (4KB).
            let mut conv_plus_mixin = [0.0f32; 1024];
            let conv_slice = &mut conv_plus_mixin[..num_frames * CH];

            // [FASE 1: Linear - Conv1D + Mixin]
            // Dual-Frame Tiling: Processamos 2 frames por iteração para amortizar
            // o carregamento de pesos da Conv1D nos registradores.
            let mut i = 0;
            let mut chunks = conv_slice.chunks_exact_mut(2 * CH);
            for chunk in chunks.by_ref() {
                let (out_frame_f0, out_frame_f1) = chunk.split_at_mut(CH);

                let mix_idx_f0 = i * CH;
                let mix_idx_f1 = (i + 1) * CH;
                let mixin_f0 = mixin_out.get_unchecked(mix_idx_f0..mix_idx_f0 + CH);
                let mixin_f1 = mixin_out.get_unchecked(mix_idx_f1..mix_idx_f1 + CH);

                if M::IS_BF16 {
                    self.conv1d.process_dual_frame_bf16_with_mixin::<M>(
                        layer_buffer_bf16,
                        out_frame_f0,
                        out_frame_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        mixin_f0,
                        mixin_f1,
                    );
                } else {
                    self.conv1d.process_dual_frame_with_mixin::<M>(
                        layer_buffer,
                        out_frame_f0,
                        out_frame_f1,
                        buffer_start + i,
                        buffer_start + i + 1,
                        mixin_f0,
                        mixin_f1,
                    );
                }
                i += 2;
            }

            // Trata o frame residual (ímpar)
            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                let mix_idx = i * CH;
                let mixin_slice = mixin_out.get_unchecked(mix_idx..mix_idx + CH);

                if M::IS_BF16 {
                    self.conv1d.process_single_frame_bf16_with_mixin::<M>(
                        layer_buffer_bf16,
                        rem,
                        buffer_start + i,
                        mixin_slice,
                    );
                } else {
                    self.conv1d.process_single_frame_with_mixin::<M>(
                        layer_buffer,
                        rem,
                        buffer_start + i,
                        mixin_slice,
                    );
                }
            }

            // [FASE 2 & 3: Ativação e Head Update Fundidos]
            // Aplicamos a Tanh e acumulamos no Head em uma única passagem de memória.
            // Isso reduz a pressão sobre a largura de banda (evita 1 leitura e 1 escrita extra).
            M::tanh_and_accumulate_block(head_input, conv_slice);

            // [FASE 3: Saída - 1x1 Residual]
            // [TF3] Otimização: Projeção 1x1 fundida com a soma do residual em lote.
            // Elimina a cópia prévia do estado original (layer_buffer) para o output.
            let lb_offset = buffer_start * CH;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * CH);

            self.one_by_one.process_residual_batch::<M>(
                conv_slice,
                residual_slice,
                output,
                num_frames,
            );

            // 4. [T25] Fusão BF16: Conversão em lote se necessário
            if let (true, Some(bf16_out)) = (M::IS_BF16, output_bf16.as_mut()) {
                M::f32_to_bf16(output, bf16_out);
            }
        }
    }
}

// (Estado da camada movido para wavenet_common.rs)

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
    /// Acumulador temporário de saída do Array.
    pub array_outputs: AlignedVec<f32>,
    /// Acumulador intermediário CH-sized para contribuições das camadas antes da projeção Head.
    pub head_accum: AlignedVec<f32>,
    /// Memória alocada da projeção Linear global (HEAD-sized).
    pub head_outputs: AlignedVec<f32>,
    /// Tamanho do campo dimensional (receptive field global) para roteamentos.
    pub receptive_field_size: usize,
    /// Tamanho do buffer compartilhado de ativação.
    pub block_size: usize,
    /// Acumulador temporário para blocos (pre-alocado).
    pub block_buffer: AlignedVec<f32>,

    /// Buffer de condicionamento cacheado em BF16.
    pub last_condition_bf16: [u16; COND],
    /// Cópia do último condicionamento f32 para comparação.
    pub last_condition: [f32; COND],
    /// Flag para primeira inicialização do cache.
    pub condition_init: bool,
}

impl<const IN: usize, const COND: usize, const CH: usize, const K: usize, const HEAD: usize>
    WaveNetLayerArray<IN, COND, CH, K, HEAD>
{
    /// Processamento central da Array. Totalmente blindado contra alocações.
    ///
    /// # Safety
    /// Ponteiros de states iteram internamente sem bounds checks.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath, const PREWARM: bool>(
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

        // [PASSO 2: Lazy BF16 Conversion]
        if M::IS_BF16 {
            let changed = PREWARM || !self.condition_init || condition != &self.last_condition[..];

            if changed {
                unsafe {
                    M::f32_to_bf16(condition, &mut self.last_condition_bf16);
                }
                self.last_condition.copy_from_slice(condition);
                self.condition_init = true;
            }
        }

        unsafe {
            // [PASSO 3: Abertura Dimensional (Rechannel)]
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

            // [PASSO 4: Cascata de Inferência das Camadas]
            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if PREWARM {
                    // [PASSO 4.1: Propagação do Estado Estático (Backfill)]
                    // Em modo prewarm, replicamos a amostra atual para todo o histórico (Receptive Field).
                    // Isso garante que a rede estabilize instantaneamente para o valor estacionário.
                    let start_idx = current_state.buffer_start * CH;
                    let src_range = start_idx..start_idx + CH;

                    for offset in 1..=current_state.receptive_field_size {
                        debug_assert!(current_state.buffer_start >= offset, "backfill underflow: bs={}, off={}", current_state.buffer_start, offset);
                        let Some(dst_start) = current_state.buffer_start.checked_sub(offset) else {
                            log::error!("backfill underflow: bs={}, off={}", current_state.buffer_start, offset);
                            continue;
                        };
                        let dst_idx = dst_start * CH;
                        current_state
                            .layer_buffer
                            .copy_within(src_range.clone(), dst_idx);

                        if M::IS_BF16 {
                            current_state
                                .layer_buffer_bf16
                                .copy_within(src_range.clone(), dst_idx);
                        }
                    }
                }

                // Software Prefetch do próximo estado na cascata.
                // Trazemos a linha de cache do estado i+1 (e i+2 se possível) para o L1
                // enquanto o processador resolve o pipeline aritmético da camada atual.
                if i + 1 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 1) as *const i8);
                }
                if i + 2 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 2) as *const i8);
                }

                if i == last_layer {
                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..num_frames * CH],
                        output: &mut self.array_outputs[0..num_frames * CH],
                        output_bf16: None,
                        layer_buffer: &current_state.layer_buffer[..],
                        layer_buffer_bf16: &current_state.layer_buffer_bf16[..],
                        buffer_start: current_state.buffer_start,
                        num_frames,
                        block: &mut self.block_buffer[0..num_frames * self.block_size],
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let n_start = next_state.buffer_start * CH;
                    let next_layer_buffer =
                        &mut next_state.layer_buffer[n_start..n_start + num_frames * CH];
                    let next_layer_buffer_bf16 = if M::IS_BF16 {
                        Some(&mut next_state.layer_buffer_bf16[n_start..n_start + num_frames * CH])
                    } else {
                        None
                    };

                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        condition_bf16: &self.last_condition_bf16,
                        head_input: &mut self.head_accum[0..num_frames * CH],
                        output: next_layer_buffer,
                        output_bf16: next_layer_buffer_bf16,
                        layer_buffer: &current_state.layer_buffer[..],
                        layer_buffer_bf16: &current_state.layer_buffer_bf16[..],
                        buffer_start: current_state.buffer_start,
                        num_frames,
                        block: &mut self.block_buffer[0..num_frames * self.block_size],
                    });
                }

                if !PREWARM {
                    current_state.advance_frames(num_frames, CH);
                }
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
        unsafe {
            // Unificação via código compartilhado: processamos 1 frame com flag de prewarm ativa.
            // O [PASSO 4.1] dentro de `process_block_internal` cuidará do backfill.
            self.process_block_internal::<M, true>(layer_inputs, condition, 1);
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
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
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
                    .process_block_internal::<M, false>(in_slice, in_slice, num_frames);

                // [PASSO 2: Array2 Forward]
                // A segunda array atua tipicamente como uma camada perceptron de fechamento
                // (dimensões menores, dilatações apenas de 1, processando o "mix" vindo da Array1).
                let array1_outputs = &self.array1.array_outputs[0..num_frames * CH];
                self.array2.process_block_internal::<M, false>(
                    array1_outputs,
                    in_slice,
                    num_frames,
                );
            }

            // [PASSO 3: Soma das Skips + Escala Final SIMD]
            // Somatório SIMD das projeções Head de ambas as arrays e escala pela `head_scale`.
            unsafe {
                M::batch_wavenet_head_sum::<HEAD>(
                    &self.array1.head_outputs[0..num_frames * HEAD],
                    &self.array2.head_outputs[0..num_frames],
                    &mut output[pos..pos + num_frames],
                    self.head_scale,
                );
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
    #[cold]
    pub fn prewarm(&mut self) {
        unsafe {
            crate::math::common::dispatch_simd!(self, prewarm_internal);
        }
    }

    /// Prewarm estritamente otimizado para a arquitetura AVX-512.
    ///
    /// # Safety
    /// Exige processador suportado (AVX-512).
    #[target_feature(enable = "avx512f,avx512vl")]
    #[cold]
    pub unsafe fn prewarm_avx512(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::common::Avx512Math>() };
    }

    /// Prewarm estritamente otimizado para arquitetura AVX2.
    ///
    /// # Safety
    /// Exige processador x86-64-v3 (AVX2).
    #[cold]
    pub unsafe fn prewarm_avx2(&mut self) {
        unsafe { self.prewarm_internal::<crate::math::common::Avx2Math>() };
    }

    #[inline(always)]
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    #[cold]
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
