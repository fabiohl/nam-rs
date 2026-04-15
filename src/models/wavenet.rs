// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
// Portado em grande parte da implementação original em C++ (NeuralAudio) por Mike Oliphant.

//! Malha CNN Causal Estática para inferência WaveNet (Design Orientado a Dados, SoA).
//!
//! Todas as estruturas utilizam `Const Generics` nas dimensões matemáticas e vetores pré-alocados
//! garantindo uma política de instanciamento estrito (Zero-Allocation durante processamento).
//! As loops dinâmicos resolvem cálculos em sequências FMA determinísticas via AVX2.

#![allow(clippy::needless_range_loop)]

use crate::math::simd::SimdMathConfig;

/// Máximo de frames a processar em um pulso do callback.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Padding temporal circular das memórias no framework de Ring Buffers.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;

/// Convolução Causal Dilatada (WaveNet Conv1D).
#[derive(Clone)]
pub struct Conv1d<const IN: usize, const OUT: usize, const K: usize> {
    /// Matriz achatada de pesos do tamanho OUT * K * IN.
    pub weights: Vec<f32>,
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
    /// # Safety
    /// Depende dinamicamente da V-Table `SimdMathConfig` fornecida.
    pub unsafe fn process_frame(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        math: &SimdMathConfig,
    ) {
        for out_c in 0..OUT {
            let mut sum = if self.do_bias { self.bias[out_c] } else { 0.0 };

            for k in 0..K {
                let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
                let frame_idx = (buffer_start as isize) + offset;

                let in_slice_start = (frame_idx as usize) * IN;
                let in_slice = &layer_buffer[in_slice_start..in_slice_start + IN];

                let weight_slice_start = (out_c * K + k) * IN;
                let weight_slice = &self.weights[weight_slice_start..weight_slice_start + IN];

                unsafe {
                    sum += (math.dot_product)(in_slice, weight_slice);
                }
            }

            block[out_c] = sum;
        }
    }
}

/// Camada Densa 1x1 baseada num Matmul vetorizado linear.
#[derive(Clone)]
pub struct DenseLayer<const IN: usize, const OUT: usize> {
    /// Matriz de pesos lineares (OUT * IN).
    pub weights: Vec<f32>,
    /// Condição temporal para o deslocador de tensor bias.
    pub bias: Vec<f32>,
    /// Determina se o array de bias deve ser somado.
    pub do_bias: bool,
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    /// Processa o Dense acumulando com o estado corrente de output.
    ///
    /// # Safety
    /// Despacho matemático via ponteiro para funções intrínsecas inlined.
    pub unsafe fn process_acc(&self, input: &[f32], output: &mut [f32], math: &SimdMathConfig) {
        for out_c in 0..OUT {
            let weight_slice = &self.weights[out_c * IN..out_c * IN + IN];
            let sum = unsafe { (math.dot_product)(input, weight_slice) };

            if self.do_bias {
                output[out_c] += sum + self.bias[out_c];
            } else {
                output[out_c] += sum;
            }
        }
    }

    /// Processa o Dense sobrescrevendo o slice do output.
    ///
    /// # Safety
    /// Despacho matemático via ponteiro para funções intrínsecas inlined.
    pub unsafe fn process(&self, input: &[f32], output: &mut [f32], math: &SimdMathConfig) {
        for out_c in 0..OUT {
            let weight_slice = &self.weights[out_c * IN..out_c * IN + IN];
            let sum = unsafe { (math.dot_product)(input, weight_slice) };

            if self.do_bias {
                output[out_c] = sum + self.bias[out_c];
            } else {
                output[out_c] = sum;
            }
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

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processa uma camada integral do WaveNet, iterando `FastMath` em AVX2.
    ///
    /// # Safety
    /// Despacho matemático via ponteiro para funções intrínsecas inlined.
    pub unsafe fn process(
        &self,
        condition: &[f32],
        head_input: &mut [f32],
        output: &mut [f32],
        layer_buffer: &[f32],
        buffer_start: usize,
        math: &SimdMathConfig,
    ) {
        let mut block = [0.0f32; CH];

        unsafe {
            self.conv1d
                .process_frame(layer_buffer, &mut block, buffer_start, math);
            self.input_mixin.process_acc(condition, &mut block, math);

            // Ativação Tanh usando V-Table do SIMD HW Config
            (math.tanh_slice)(&mut block);

            // Sum block to head_input
            for j in 0..CH {
                head_input[j] += block[j];
            }

            self.one_by_one.process(&block, output, math);
        }

        // output += layer_buffer[buffer_start] (Residual connection)
        let lb_start = buffer_start * CH;
        for j in 0..CH {
            output[j] += layer_buffer[lb_start + j];
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
    /// Vetor base plano linear do Ring Buffer (zero alocações em contexto DSP).
    pub layer_buffer: std::vec::Vec<f32>,
    /// Ponteiro numérico do frame atual (avança a cada frame processado).
    pub buffer_start: usize,
    /// Dimensão física do espaço vetorial receptivo (tamanho do histórico de dilatação).
    pub receptive_field_size: usize,
}

impl WaveNetLayerState {
    /// Construtor alocador estático do Estado (executar antes do Thread DSP).
    pub fn new(channels: usize, receptive_field_size: usize, alloc_num: usize) -> Self {
        let buffer_frames =
            receptive_field_size + (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES;
        let buffer = vec![0.0f32; buffer_frames * channels];

        let start = buffer_frames
            - (WAVENET_MAX_NUM_FRAMES * ((alloc_num % LAYER_ARRAY_BUFFER_PADDING) + 1));

        Self {
            layer_buffer: buffer,
            buffer_start: start,
            receptive_field_size,
        }
    }

    /// Executa um passo do ponteiro do Ring Buffer. Se chegar na margem, chama Re-Wind.
    pub fn advance_frames(&mut self, num_frames: usize, channels: usize) {
        self.buffer_start += num_frames;
        let buffer_frames = self.layer_buffer.len() / channels;
        if self.buffer_start + WAVENET_MAX_NUM_FRAMES > buffer_frames {
            self.rewind_buffer(channels);
        }
    }

    /// Rebina a memória do Ring Buffer para evitar overflow circular conservando o hitspace L1.
    pub fn rewind_buffer(&mut self, channels: usize) {
        let start = self.receptive_field_size;
        let from = (self.buffer_start - self.receptive_field_size) * channels;
        let to = (start - self.receptive_field_size) * channels;
        let len = self.receptive_field_size * channels;

        self.layer_buffer.copy_within(from..from + len, to);
        self.buffer_start = start;
    }

    /// Trinca retroativa da Memória Receptiva no estado inicial de Warm-up de estado.
    pub fn copy_buffer(&mut self, channels: usize) {
        for offset in 1..=self.receptive_field_size {
            let src = self.buffer_start * channels;
            let dst = (self.buffer_start - offset) * channels;
            self.layer_buffer.copy_within(src..src + channels, dst);
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

    /// Conexão transiente de pre-alocação zero-copy ao próximo vetor.
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
    pub unsafe fn process(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        math: &SimdMathConfig,
    ) {
        debug_assert_eq!(
            self.layers.len(),
            self.states.len(),
            "WaveNetLayerArray: layers ({}) ≠ states ({})",
            self.layers.len(),
            self.states.len()
        );
        let states_ptr = self.states.as_mut_ptr();

        // Zera o acumulador CH-sized para contribuições das camadas ao head.
        // `fill` emite vmovups vetorizado em vez de loop escalar.
        self.head_accum.fill(0.0);

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + CH],
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);

                if i == last_layer {
                    layer.process(
                        condition,
                        &mut self.head_accum[0..CH],
                        &mut self.array_outputs[0..CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process(
                        condition,
                        &mut self.head_accum[0..CH],
                        &mut next_state.layer_buffer[next_start..next_start + CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        math,
                    );
                }

                current_state.advance_frames(1, CH);
            }

            self.head_rechannel.process(
                &self.head_accum[0..CH],
                &mut self.head_outputs[0..HEAD],
                math,
            );
        }
    }

    /// Invoca a transposição artificial do modelo em Pre-warm estabilizando memória temporal.
    pub fn prewarm(&mut self, layer_inputs: &[f32], condition: &[f32], math: &SimdMathConfig) {
        debug_assert_eq!(
            self.layers.len(),
            self.states.len(),
            "WaveNetLayerArray: layers ({}) ≠ states ({})",
            self.layers.len(),
            self.states.len()
        );
        let states_ptr = self.states.as_mut_ptr();

        // Zera o acumulador CH-sized para contribuições das camadas ao head.
        // `fill` emite vmovups vetorizado em vez de loop escalar.
        self.head_accum.fill(0.0);

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + CH],
                math,
            );

            let num_layers = self.layers.len();
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter().enumerate() {
                let current_state = &mut *states_ptr.add(i);
                current_state.copy_buffer(CH);

                if i == last_layer {
                    layer.process(
                        condition,
                        &mut self.head_accum[0..CH],
                        &mut self.array_outputs[0..CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        math,
                    );
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let next_start = next_state.buffer_start * CH;

                    layer.process(
                        condition,
                        &mut self.head_accum[0..CH],
                        &mut next_state.layer_buffer[next_start..next_start + CH],
                        &current_state.layer_buffer,
                        current_state.buffer_start,
                        math,
                    );
                }
            }

            self.head_rechannel.process(
                &self.head_accum[0..CH],
                &mut self.head_outputs[0..HEAD],
                math,
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
    /// `SimdMathConfig::current()` é hoistado fora do loop de frames para evitar
    /// redespacho via CPUID a cada amostra (reduziria throughput em ~20 µs/bloco).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Hoist: despacha a v-table matemática UMA vez por bloco DSP, não por sample.
        // `is_x86_feature_detected!` serializa o pipeline via CPUID — chamá-lo 64×
        // por callback desperdiça ~0.3 µs cada = ~20 µs/bloco evitáveis.
        let math = &crate::math::simd::SimdMathConfig::current();
        let num_frames = input.len();

        for i in 0..num_frames {
            let sample = input[i];
            let condition = [sample];
            let layer_inputs_1 = [sample];

            unsafe {
                self.array1.process(&layer_inputs_1, &condition, math);

                // Array2 recebe a saída da Array1 como input (CH canais)
                let array1_outputs = &self.array1.array_outputs[0..CH];
                self.array2.process(array1_outputs, &condition, math);
            }

            // Somatório das projeções Head de ambas as arrays
            let mut final_sum = 0.0f32;
            for j in 0..HEAD {
                final_sum += self.array1.head_outputs[j];
            }
            // Array2 projeta para HEAD2=1
            final_sum += self.array2.head_outputs[0];
            output[i] = final_sum * self.head_scale;
        }
    }

    /// Estabiliza os transientes inicias causais por tempo de propagação (Zero Input).
    pub fn prewarm(&mut self) {
        let math = &crate::math::simd::SimdMathConfig::current();
        let condition = [0.0f32];
        let layer_inputs_1 = [0.0f32];

        self.array1.prewarm(&layer_inputs_1, &condition, math);
        let array1_outputs = &self.array1.array_outputs[0..CH];
        self.array2.prewarm(array1_outputs, &condition, math);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constrói um WaveNetModel<4, 3, 2> mínimo para testes com dados zerados.
    /// Array1: IN=1, COND=1, CH=4, K=3, HEAD=2
    /// Array2: IN=4, COND=1, CH=2 (=HEAD), K=3, HEAD2=1
    fn build_tiny_wavenet() -> WaveNetModel<4, 3, 2> {
        let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 3> {
            WaveNetLayer {
                conv1d: Conv1d {
                    weights: vec![0.01; 4 * 3 * 4],
                    bias: vec![0.0; 4],
                    do_bias: false,
                    dilation,
                },
                input_mixin: DenseLayer {
                    weights: vec![0.01; 4],
                    bias: vec![0.0; 4],
                    do_bias: false,
                },
                one_by_one: DenseLayer {
                    weights: vec![0.01; 4 * 4],
                    bias: vec![0.0; 4],
                    do_bias: false,
                },
            }
        };

        // Array2: CH=2 (=HEAD), layers com COND=1, CH=2
        let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 3> {
            WaveNetLayer {
                conv1d: Conv1d {
                    weights: vec![0.01; 2 * 3 * 2],
                    bias: vec![0.0; 2],
                    do_bias: false,
                    dilation,
                },
                input_mixin: DenseLayer {
                    weights: vec![0.01; 2],
                    bias: vec![0.0; 2],
                    do_bias: false,
                },
                one_by_one: DenseLayer {
                    weights: vec![0.01; 2 * 2],
                    bias: vec![0.0; 2],
                    do_bias: false,
                },
            }
        };

        let dilations_1 = [1, 2, 4];
        let dilations_2 = [1, 2, 4];

        let rf1 = *dilations_1.iter().max().unwrap_or(&1) * (3 - 1);
        let rf2 = *dilations_2.iter().max().unwrap_or(&1) * (3 - 1);

        // Construção manual das arrays com const generics explícitos.
        let layers_1: Vec<WaveNetLayer<1, 4, 3>> =
            dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
        let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
            .map(|i| WaveNetLayerState::new(4, rf1, i))
            .collect();

        let array1 = WaveNetLayerArray::<1, 1, 4, 3, 2> {
            layers: layers_1,
            states: states_1,
            rechannel: DenseLayer {
                weights: vec![0.01; 4],
                bias: vec![0.0; 4],
                do_bias: false,
            },
            head_rechannel: DenseLayer {
                weights: vec![0.01; 2 * 4],
                bias: vec![0.0; 2],
                do_bias: false,
            },
            array_outputs: vec![0.0; 4],
            head_accum: vec![0.0; 4],
            head_outputs: vec![0.0; 2],
            receptive_field_size: rf1,
        };

        // Array2: IN=4(=CH), COND=1, CH=2(=HEAD), K=3, HEAD2=1
        let layers_2: Vec<WaveNetLayer<1, 2, 3>> =
            dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
        let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
            .map(|i| WaveNetLayerState::new(2, rf2, i))
            .collect();

        let array2 = WaveNetLayerArray::<4, 1, 2, 3, 1> {
            layers: layers_2,
            states: states_2,
            rechannel: DenseLayer {
                weights: vec![0.01; 4 * 2],
                bias: vec![0.0; 2],
                do_bias: false,
            },
            head_rechannel: DenseLayer {
                weights: vec![0.01; 2],
                bias: vec![0.0; 1],
                do_bias: true, // array2 HasHeadBias=true
            },
            array_outputs: vec![0.0; 2],
            head_accum: vec![0.0; 2],
            head_outputs: vec![0.0; 1],
            receptive_field_size: rf2,
        };

        WaveNetModel {
            array1,
            array2,
            head_scale: 0.02,
            receptive_field_size: rf1.max(rf2),
        }
    }

    #[test]
    fn test_wavenet_model_allocation() {
        let model = build_tiny_wavenet();
        assert_eq!(model.array1.layers.len(), 3);
        assert_eq!(model.array2.layers.len(), 3);
        assert_eq!(model.array1.head_outputs.len(), 2); // HEAD1=2
        assert_eq!(model.array2.head_outputs.len(), 1); // HEAD2=1 (sempre fixo)
        assert!((model.head_scale - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_wavenet_prewarm_no_nan() {
        let mut model = build_tiny_wavenet();
        model.prewarm();

        // Verificar que os buffers internos não contêm NaN/Inf após prewarm
        for state in &model.array1.states {
            for &v in &state.layer_buffer {
                assert!(v.is_finite(), "NaN/Inf detectado no array1 após prewarm");
            }
        }
        for state in &model.array2.states {
            for &v in &state.layer_buffer {
                assert!(v.is_finite(), "NaN/Inf detectado no array2 após prewarm");
            }
        }
    }

    #[test]
    fn test_wavenet_process_zeros() {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut model = build_tiny_wavenet();
            model.prewarm();

            let input = [0.0f32; 16];
            let mut output = [0.0f32; 16];

            model.process(&input, &mut output);

            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "Amostra de saída [{}] é NaN/Inf: {}", i, v);
            }
        }
    }

    #[test]
    fn test_wavenet_process_deterministic() {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            let mut model_a = build_tiny_wavenet();
            let mut model_b = build_tiny_wavenet();

            model_a.prewarm();
            model_b.prewarm();

            let input = [0.1f32; 8];
            let mut out_a = [0.0f32; 8];
            let mut out_b = [0.0f32; 8];

            model_a.process(&input, &mut out_a);
            model_b.process(&input, &mut out_b);

            for i in 0..8 {
                assert!(
                    (out_a[i] - out_b[i]).abs() < 1e-6,
                    "Resultado não-determinístico na amostra [{}]: {} vs {}",
                    i,
                    out_a[i],
                    out_b[i]
                );
            }
        }
    }
}
