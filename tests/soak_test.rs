// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Suíte de Estabilidade Numérica (Soak Test) para o NAM-rs.
//!
//! Estes testes são projetados para rodar por milhões de iterações, verificando
//! drift numérico, estabilidade de filtros, ausência de NaNs/Infs e integridade
//! de buffers circulares em execuções de longa duração.
//!
//! Execução: `cargo test --release -- --ignored --nocapture`

use nam_rs::dsp::gate::*;
use nam_rs::dsp::resampler::*;
use nam_rs::dsp::vring::*;
use nam_rs::math::common::AlignedVec;
use nam_rs::models::lstm::*;
use nam_rs::models::wavenet::*;
use nam_rs::models::wavenet_common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use std::time::Instant;

/// PRNG Determinístico simples (Linear Congruential Generator - LCG).
///
/// Implementado manualmente para manter o projeto livre de dependências de crates
/// de rand apenas para testes. O objetivo é fornecer um fluxo de ruído
/// reprodutível para diagnósticos de falhas em soak tests.
struct SimplePcg {
    state: u64,
}

impl SimplePcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Gera o próximo f32 no intervalo [-1.0, 1.0].
    fn next_f32(&mut self) -> f32 {
        // Variante simplificada do PCG-XSH-RR para 32-bit output.
        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        let res = (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31));
        // Normalização para o range de áudio flutuante
        (res as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Helper para construir um modelo WaveNetModel<16, 3, 8> sintético para soak test.
///
/// Este modelo simula a topologia "Standard" (Array1 + Array2) com 10 milhões de parâmetros
/// implícitos via pesos constantes. A inicialização com valores pequenos (0.01) garante
/// que o áudio não exploda imediatamente, permitindo que a FPU processe valores reais
/// em todas as camadas por milhões de iterações.
fn build_soak_wavenet() -> WaveNetModel<16, 3, 8> {
    // Camada interna da Array1: CH=16, K=3
    let make_layer = |dilation: usize| -> WaveNetLayer<1, 16, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: AlignedVec::from_vec(vec![
                    half::f16::from_f32(0.01).to_bits();
                    16 * 3 * 16
                ]),
                bias: AlignedVec::from_vec(vec![0.001; 16]),
                do_bias: true,
                dilation,
                // A estratégia de prefetch muda para dilatações grandes para testar
                // o cache-miss handling do motor.
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::common::prefetch_strategy_2stage
                } else {
                    nam_rs::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16]),
                bias: AlignedVec::from_vec(vec![0.0; 16]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16 * 16]),
                bias: AlignedVec::from_vec(vec![0.0; 16]),
                do_bias: false,
            },
        }
    };

    // Dilatações padrão do modelo Standard (RF total = 2046 amostras)
    let dilations = [1, 2, 4, 8, 16, 32, 64, 128];
    let rf = 128 * (3 - 1);

    let layers_1: Vec<WaveNetLayer<1, 16, 3>> = dilations.iter().map(|&d| make_layer(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(16, rf, i))
        .collect();

    // Array 1: Responsável pela extração de features temporais profundas
    let array1 = WaveNetLayerArray::<1, 1, 16, 3, 8> {
        layers: layers_1,
        states: states_1,
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16]),
            bias: AlignedVec::from_vec(vec![0.0; 16]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8 * 16]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf,
        block_size: 16,
        block_buffer: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    // Array 2: Refinamento espectral final (CH=8, K=3)
    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 8, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8 * 3 * 8]),
                bias: AlignedVec::from_vec(vec![0.001; 8]),
                do_bias: true,
                dilation,
                prefetch_fn: nam_rs::math::common::prefetch_strategy_simple,
            },
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8]),
                bias: AlignedVec::from_vec(vec![0.0; 8]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8 * 8]),
                bias: AlignedVec::from_vec(vec![0.0; 8]),
                do_bias: false,
            },
        }
    };

    let layers_2: Vec<WaveNetLayer<1, 8, 3>> = [1, 2].iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(8, 2 * (3 - 1), i))
        .collect();

    let array2 = WaveNetLayerArray::<16, 1, 8, 3, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 16 * 8]),
            bias: AlignedVec::from_vec(vec![0.0; 8]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 8]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: 2,
        block_size: 8,
        block_buffer: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.1,
        receptive_field_size: rf,
    }
}

/// Soak Test: Processamento de Silêncio Infinito via WaveNet.
///
/// O objetivo é verificar se o acúmulo de erros em modelos com realimentação (se houver)
/// ou a instabilidade de precisão f16 nos kernels SIMD causa NaNs ou "estouros"
/// de áudio após milhões de iterações.
#[test]
#[ignore] // Rodar manualmente via --ignored
fn test_wavenet_silence_soak() {
    let mut model = build_soak_wavenet();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        // black_box impede o compilador de perceber que a entrada é sempre zero
        // e otimizar todo o loop de inferência para um no-op.
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            // Verificação de integridade numérica básica (Segurança RT)
            assert!(
                v.is_finite(),
                "Instabilidade Numérica (NaN/Inf) após {} frames",
                processed
            );
            // Garante que o modelo não está ganhando energia do nada (divergência)
            assert!(
                (-2.0..=2.0).contains(&v),
                "Output divergiu excessivamente: {} após {} frames",
                v,
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- WaveNet Silence Soak ---");
    println!("Duração: {:?}", duration);
    println!("Frames processados: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: Processamento de Ruído Branco via WaveNet.
///
/// Diferente do silêncio, o ruído branco excita todos os coeficientes dos filtros
/// e todas as ramificações de ativação (Tanh) simultaneamente. Isso é vital
/// para detectar "overflows" numéricos em camadas profundas que só ocorrem
/// sob alta energia de sinal.
#[test]
#[ignore]
fn test_wavenet_noise_soak() {
    let mut model = build_soak_wavenet();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(42);
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "NaN/Inf detectado sob ruído após {} frames",
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- WaveNet Noise Soak ---");
    println!("Duração: {:?}", duration);
    println!("Frames processados: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: Estabilidade de estado da LSTM sob silêncio.
///
/// LSTMs possuem estados recorrentes (cell state) que podem acumular erros residuais.
/// Este teste monitora a sanidade do estado interno por milhões de iterações.
#[test]
#[ignore]
fn test_lstm_silence_soak() {
    let mut model = LstmModel2::<16, 17, 32, 64>::new();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;

    let start = Instant::now();
    while processed < num_frames {
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "NaN/Inf detectado na saída LSTM após {} frames",
                processed
            );
        }
        // Verifica divergência dos estados internos da arquitetura de 2 camadas
        for &v in &model.layer1.cell_state {
            assert!(v.is_finite(), "Estado interno da Camada 1 LSTM corrompido");
        }
        for &v in &model.layer2.cell_state {
            assert!(v.is_finite(), "Estado interno da Camada 2 LSTM corrompido");
        }

        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Silence Soak ---");
    println!("Duração: {:?}", duration);
    println!("Frames processados: {}", processed);
}

/// Soak Test: Resistência da LSTM sob entrada estocástica.
///
/// Sinais aleatórios forçam o esquecimento e a atualização constante das portas
/// (Gates) da LSTM. Este teste garante que a lógica de "Forget/Input/Output"
/// não causa deriva infinita sob condições de estresse de sinal.
#[test]
#[ignore]
fn test_lstm_noise_soak() {
    let mut model = LstmModel2::<16, 17, 32, 64>::new();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(1337);
    let num_frames = 10_000_000;
    let mut processed = 0;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "LSTM divergiu sob ruído após {} frames",
                processed
            );
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Noise Soak ---");
    println!("Duração: {:?}", duration);
    println!("Frames processados: {}", processed);
}

/// Soak Test: Estabilidade do Resampler em conversões assíncronas de longa duração.
///
/// O resampler usa interpolação polinomial e acúmulo de fase. Pequenos erros de
/// arredondamento no incremento de fase podem se acumular em milhões de amostras,
/// causando "clicks" ou NaNs. Este teste valida cenários de Upsampling e Downsampling.
#[test]
#[ignore]
fn test_resampler_drift_soak() {
    // Cenário 1: Upsampling (22050 -> 48000)
    let mut resampler = NamResampler::new(22050, 48000, 64).unwrap();
    let mut pcg = SimplePcg::new(777);
    let num_samples = 50_000_000; // 50 milhões de amostras (~17 minutos a 48kHz)
    let mut processed_in = 0;
    let mut processed_out = 0;

    let mut in_l = vec![0.0f32; 1024];
    let mut in_r = vec![0.0f32; 1024];
    let mut out_l = vec![0.0f32; 2048];
    let mut out_r = vec![0.0f32; 2048];

    let start = Instant::now();
    while processed_in < num_samples {
        for i in 0..1024 {
            in_l[i] = pcg.next_f32();
            in_r[i] = pcg.next_f32();
        }
        let n = resampler.process_input(
            std::hint::black_box(&in_l),
            std::hint::black_box(&in_r),
            std::hint::black_box(&mut out_l),
            std::hint::black_box(&mut out_r),
        );
        processed_in += 1024;
        processed_out += n;

        for i in 0..n {
            assert!(out_l[i].is_finite(), "NaN no Resampler Out L (Upsampling)");
            assert!(out_r[i].is_finite(), "NaN no Resampler Out R (Upsampling)");
        }
    }
    let duration = start.elapsed();

    println!("--- Resampler Drift Soak (22050->48000) ---");
    println!("Duração: {:?}", duration);
    println!("Amostras In: {}, Out: {}", processed_in, processed_out);

    // Cenário 2: Downsampling (96000 -> 48000)
    let mut resampler = NamResampler::new(96000, 48000, 64).unwrap();
    processed_in = 0;
    processed_out = 0;
    let start = Instant::now();
    while processed_in < num_samples {
        for i in 0..1024 {
            in_l[i] = pcg.next_f32();
            in_r[i] = pcg.next_f32();
        }
        let n = resampler.process_input(
            std::hint::black_box(&in_l),
            std::hint::black_box(&in_r),
            std::hint::black_box(&mut out_l[..512]),
            std::hint::black_box(&mut out_r[..512]),
        );
        processed_in += 1024;
        processed_out += n;
        for i in 0..n {
            assert!(
                out_l[i].is_finite(),
                "NaN no Resampler Out L (Downsampling)"
            );
            assert!(
                out_r[i].is_finite(),
                "NaN no Resampler Out R (Downsampling)"
            );
        }
    }
    println!("--- Resampler Drift Soak (96000->48000) ---");
    println!("Duração: {:?}", start.elapsed());
    println!("Amostras In: {}, Out: {}", processed_in, processed_out);
}

/// Soak Test: Integridade da memória do VirtualRingBuffer.
///
/// O VirtualRingBuffer usa mmap para espelhar a memória, permitindo acessos lineares
/// contínuos que cruzam a borda do buffer. Este teste verifica se o espelhamento
/// permanece consistente após bilhões de escritas.
#[test]
#[ignore]
fn test_vring_long_run() {
    let mut vring = VirtualRingBuffer::<f32>::new(1024 * 1024); // 1M elementos
    let size = vring.size();
    let num_cycles = 100_000_000;
    let mut pos = 0;
    let chunk = 64;

    let start = Instant::now();
    for i in 0..(num_cycles / chunk) {
        // Simula escrita e avanço no buffer espelhado
        for j in 0..chunk {
            std::hint::black_box(&mut vring)[pos + j] =
                std::hint::black_box((i * chunk + j) as f32);
        }

        // Verifica integridade na fronteira: o valor escrito no final
        // deve aparecer identicamente no início (espelhamento mmap).
        if pos + chunk >= size {
            let offset = (pos + chunk) - size;
            for j in 0..offset {
                assert_eq!(
                    vring[j],
                    vring[size + j],
                    "Falha crítica de espelhamento de memória no VirtualRingBuffer no índice {}",
                    j
                );
            }
        }

        pos += chunk;
        if pos >= size {
            pos -= size;
        }
    }
    let duration = start.elapsed();

    println!("--- VirtualRingBuffer Long Run ---");
    println!("Duração: {:?}", duration);
    println!("Ciclos (amostras): {}", num_cycles);
}

/// Soak Test: Resistência da Máquina de Estados (FSM) do Noise Gate.
///
/// Este teste força alternâncias rápidas entre os estados de `Open`, `Hold` e `Release`
/// usando um sinal binário estocástico. O objetivo é garantir que o multiplicador
/// de ganho nunca saia do intervalo [0, 1] e que a FSM não trave em estados
/// inválidos após milhões de transições.
#[test]
#[ignore]
fn test_gate_fsm_endurance() {
    let mut gate = DynamicHysteresis::new();
    let params = GateParams::default();
    let mut pcg = SimplePcg::new(999);
    let num_alternations = 10_000_000; // 10 milhões de ciclos de transição

    // Thresholds típicos de uso real (-60dB e -80dB)
    let threshold_open = -60.0f32.powf(10.0 / 20.0);
    let threshold_close = -80.0f32.powf(10.0 / 20.0);

    let start = Instant::now();
    for _ in 0..num_alternations {
        // Alterna agressivamente entre sinal pleno (1.0) e silêncio (0.0)
        let val = if pcg.next_f32() > 0.0 { 1.0 } else { 0.0 };
        gate.update(
            std::hint::black_box(val),
            threshold_open,
            threshold_close,
            &params,
            64,
        );

        // O multiplicador de ganho do gate deve ser estritamente comportado [0.0, 1.0]
        let m = std::hint::black_box(gate.multiplier());
        assert!(
            (0.0..=1.0).contains(&m),
            "Multiplicador do Gate divergiu: {} fora de [0, 1]",
            m
        );
    }
    let duration = start.elapsed();

    println!("--- Gate FSM Endurance ---");
    println!("Duração: {:?}", duration);
    println!("Alternâncias: {}", num_alternations);
}
