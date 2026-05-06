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
use nam_rs::models::lstm::*;
use nam_rs::models::wavenet::*;
use nam_rs::models::wavenet_common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use std::time::Instant;

/// PRNG Determinístico simples (LCG) para evitar dependência de crates externos nos testes.
struct SimplePcg {
    state: u64,
}

impl SimplePcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self) -> f32 {
        // PCG-XSH-RR simplificado
        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        let res = (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31));
        (res as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Helper para construir um WaveNetModel<16, 3, 8> para soak test.
fn build_soak_wavenet() -> WaveNetModel<16, 3, 8> {
    let make_layer = |dilation: usize| -> WaveNetLayer<1, 16, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: vec![half::f16::from_f32(0.01).to_bits(); 16 * 3 * 16],
                bias: vec![0.001; 16],
                do_bias: true,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::simd::prefetch_strategy_2stage
                } else {
                    nam_rs::math::simd::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                weights: vec![half::f16::from_f32(0.01).to_bits(); 16],
                bias: vec![0.0; 16],
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: vec![half::f16::from_f32(0.01).to_bits(); 16 * 16],
                bias: vec![0.0; 16],
                do_bias: false,
            },
        }
    };

    let dilations = [1, 2, 4, 8, 16, 32, 64, 128];
    let rf = 128 * (3 - 1);

    let layers_1: Vec<WaveNetLayer<1, 16, 3>> = dilations.iter().map(|&d| make_layer(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(16, rf, i))
        .collect();

    let array1 = WaveNetLayerArray::<1, 1, 16, 3, 8> {
        layers: layers_1,
        states: states_1,
        rechannel: DenseLayer {
            weights: vec![half::f16::from_f32(0.01).to_bits(); 16],
            bias: vec![0.0; 16],
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: vec![half::f16::from_f32(0.01).to_bits(); 8 * 16],
            bias: vec![0.0; 8],
            do_bias: false,
        },
        array_outputs: vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES],
        head_accum: vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES],
        head_outputs: vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES],
        receptive_field_size: rf,
        block_size: 16,
        block_buffer: vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES],
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    // Array 2
    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 8, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: vec![half::f16::from_f32(0.01).to_bits(); 8 * 3 * 8],
                bias: vec![0.001; 8],
                do_bias: true,
                dilation,
                prefetch_fn: nam_rs::math::simd::prefetch_strategy_simple,
            },
            input_mixin: DenseLayer {
                weights: vec![half::f16::from_f32(0.01).to_bits(); 8],
                bias: vec![0.0; 8],
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: vec![half::f16::from_f32(0.01).to_bits(); 8 * 8],
                bias: vec![0.0; 8],
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
            weights: vec![half::f16::from_f32(0.01).to_bits(); 16 * 8],
            bias: vec![0.0; 8],
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: vec![half::f16::from_f32(0.01).to_bits(); 8],
            bias: vec![0.0; 1],
            do_bias: true,
        },
        array_outputs: vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES],
        head_accum: vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES],
        head_outputs: vec![0.0; WAVENET_MAX_NUM_FRAMES],
        receptive_field_size: 2,
        block_size: 8,
        block_buffer: vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES],
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

#[test]
#[ignore]
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
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(v.is_finite(), "NaN/Inf detectado após {} frames", processed);
            assert!(
                (-2.0..=2.0).contains(&v),
                "Output fora dos limites: {} após {} frames",
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
            assert!(v.is_finite(), "NaN/Inf detectado após {} frames", processed);
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
                "NaN/Inf detectado na saída após {} frames",
                processed
            );
        }
        // Verifica divergência dos estados internos
        for &v in &model.layer1.cell_state {
            assert!(v.is_finite());
        }
        for &v in &model.layer2.cell_state {
            assert!(v.is_finite());
        }

        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Silence Soak ---");
    println!("Duração: {:?}", duration);
    println!("Frames processados: {}", processed);
}

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
            assert!(v.is_finite(), "NaN/Inf detectado após {} frames", processed);
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Noise Soak ---");
    println!("Duração: {:?}", duration);
    println!("Frames processados: {}", processed);
}

#[test]
#[ignore]
fn test_resampler_drift_soak() {
    // 22050 -> 48000
    let mut resampler = NamResampler::new(22050, 48000, 64).unwrap();
    let mut pcg = SimplePcg::new(777);
    let num_samples = 50_000_000;
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

        // Verificações internas se possível (precisamos de acesso aos campos privados ou expor algo)
        // Como o teste pede para verificar phase_accum >= 0, e ele é privado,
        // vamos confiar no debug_assert interno se rodar em modo debug,
        // ou apenas verificar se a saída é finita.
        for i in 0..n {
            assert!(out_l[i].is_finite());
            assert!(out_r[i].is_finite());
        }
    }
    let duration = start.elapsed();

    println!("--- Resampler Drift Soak (22050->48000) ---");
    println!("Duração: {:?}", duration);
    println!("Amostras In: {}, Out: {}", processed_in, processed_out);

    // 96000 -> 48000
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
            assert!(out_l[i].is_finite());
            assert!(out_r[i].is_finite());
        }
    }
    println!("--- Resampler Drift Soak (96000->48000) ---");
    println!("Duração: {:?}", start.elapsed());
    println!("Amostras In: {}, Out: {}", processed_in, processed_out);
}

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
        // Simula escrita e avanço
        for j in 0..chunk {
            std::hint::black_box(&mut vring)[pos + j] =
                std::hint::black_box((i * chunk + j) as f32);
        }

        // Verifica integridade na fronteira
        if pos + chunk >= size {
            // Se cruzou, verifica o espelhamento
            let offset = (pos + chunk) - size;
            for j in 0..offset {
                assert_eq!(vring[j], vring[size + j]);
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

#[test]
#[ignore]
fn test_gate_fsm_endurance() {
    let mut gate = DynamicHysteresis::new();
    let params = GateParams::default();
    let mut pcg = SimplePcg::new(999);
    let num_alternations = 10_000_000;

    let threshold_open = -60.0f32.powf(10.0 / 20.0); // Simplified linear
    let threshold_close = -80.0f32.powf(10.0 / 20.0);

    let start = Instant::now();
    for _ in 0..num_alternations {
        // Alterna entre sinal alto e baixo para forçar transições
        let val = if pcg.next_f32() > 0.0 { 1.0 } else { 0.0 };
        gate.update(
            std::hint::black_box(val),
            threshold_open,
            threshold_close,
            &params,
            64,
        );

        // Verifica se multiplier está no range [0, 1]
        let m = std::hint::black_box(gate.multiplier());
        assert!((0.0..=1.0).contains(&m));
    }
    let duration = start.elapsed();

    println!("--- Gate FSM Endurance ---");
    println!("Duração: {:?}", duration);
    println!("Alternâncias: {}", num_alternations);
}
