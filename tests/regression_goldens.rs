// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Testes de Regressão por Vetor de Referência (Golden Vectors)
//!
//! Este arquivo contém a suíte de testes de regressão do NAM-rs. O objetivo desses testes é
//! garantir a estabilidade e a corretude matemática das nossas implementações das redes neurais
//! (WaveNet e LSTM) comparando suas saídas contra vetores de referência previamente salvos
//! (arquivos `.bin` gerados a partir de saídas validadas).
//!
//! Se qualquer alteração nos modelos ou nos kernels matemáticos otimizados (como AVX2, AVX-512)
//! alterar a saída numérica de áudio além de um limite mínimo tolerado (MSE < 1e-6), o teste falhará.
//! Isso previne regressões de precisão durante otimizações de performance.

use nam_rs::math::common::AlignedVec;
use nam_rs::models::lstm::{LstmModel1, LstmModel2};
use nam_rs::models::wavenet::*;
use nam_rs::models::wavenet::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};

/// Carrega um vetor de referência binário ("golden vector") a partir do diretório `tests/golden/`.
///
/// O arquivo de referência armazena amostras de áudio no formato de ponto flutuante de 32 bits (f32)
/// gravadas em formato Little-Endian. É o som original gerado pelo modelo de referência
/// para validação matemática.
fn load_golden(name: &str) -> Vec<f32> {
    let path = format!("tests/golden/{}.bin", name);
    let mut file = File::open(path).expect("Falha ao abrir arquivo golden");
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .expect("Falha ao ler dados golden");

    // Transforma o array de bytes em f32 lendo-o de 4 em 4 bytes (tamanho de f32)
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Gera um vetor de áudio contendo uma onda senoidal pura de teste.
///
/// É utilizada como um sinal de entrada previsível para excitar a rede neural e medir a saída.
///
/// * `freq`: Frequência da senoide em Hertz (ex: 440.0 para afinação Lá).
/// * `sr`: Taxa de amostragem em Hertz (ex: 48000.0).
/// * `len`: Quantidade de amostras de áudio a serem geradas.
fn generate_sine(freq: f32, sr: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
        .collect()
}

/// Calcula o Erro Quadrático Médio (Mean Squared Error - MSE) entre dois vetores de áudio.
///
/// Mede o desvio médio ao quadrado entre a saída real do modelo e a saída esperada (golden).
/// Quanto menor o MSE, mais idênticos são os vetores.
fn calculate_mse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "Os vetores de comparação precisam ter o mesmo tamanho"
    );
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    sum / a.len() as f32
}

/// Auxiliar para construir um conjunto de camadas WaveNet preenchido com pesos constantes para teste.
///
/// Isso inicializa a topologia interna de convolução 1D, mixer e conexões residuais de cada
/// camada usando um valor constante de peso, permitindo testar a inferência com pesos controlados.
///
/// * `dilations`: Vetor com os fatores de dilatação de cada camada do stack.
/// * `has_head_bias`: Se o head de rechannel final deve ter bias ativado.
/// * `val`: Valor flutuante constante usado para preencher todos os pesos (convertido para f16/bf16).
/// * `alloc_offset`: Posição inicial para o fatiamento de buffers circulares de estado na thread RT.
fn build_wavenet_layer_array<
    const IN: usize,
    const COND: usize,
    const CH: usize,
    const K: usize,
    const HEAD: usize,
>(
    dilations: &[usize],
    has_head_bias: bool,
    val: f32,
    alloc_offset: &mut usize,
) -> WaveNetLayerArray<IN, COND, CH, K, HEAD> {
    // Converte o valor f32 base para representação em f16 (16-bit float) usada no armazenamento de pesos
    let bits = half::f16::from_f32(val).to_bits();

    // Função auxiliar para inicializar uma única camada
    let make_layer = |dilation: usize| -> WaveNetLayer<COND, CH, K> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: AlignedVec::from_vec(vec![bits; CH * K * CH]),
                bias: AlignedVec::from_vec(vec![val; CH]),
                do_bias: true,
                dilation,
                // Define a estratégia de prefetch no cache da CPU baseado na dilatação
                prefetch_fn: if dilation >= 128 {
                    nam_rs::math::common::prefetch_strategy_2stage
                } else {
                    nam_rs::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![bits; CH * COND]),
                bias: AlignedVec::from_vec(vec![val; CH]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![bits; CH * CH]),
                bias: AlignedVec::from_vec(vec![val; CH]),
                do_bias: true,
            },
        }
    };

    let layers: Vec<WaveNetLayer<COND, CH, K>> = dilations.iter().map(|&d| make_layer(d)).collect();
    let rf: usize = dilations.iter().map(|&d| (K - 1) * d).sum();

    // Aloca as fatias de memória (estados) de cada camada no buffer circular
    let states: Vec<WaveNetLayerState> = (0..layers.len())
        .map(|i| {
            WaveNetLayerState::new(CH, rf, *alloc_offset + i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();
    *alloc_offset += layers.len();

    WaveNetLayerArray {
        layers,
        states,
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![bits; CH * IN]),
            bias: AlignedVec::from_vec(vec![val; CH]),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![bits; HEAD * CH]),
            bias: AlignedVec::from_vec(vec![val; HEAD]),
            do_bias: has_head_bias,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf,
        block_size: CH,
        block_buffer: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; COND],
        last_condition_bf16: [0; COND],
        condition_init: false,
    }
}

/// Auxiliar para instanciar um modelo WaveNet de duas arrays de camadas completo para testes.
///
/// Une duas arrays de camadas (Array 1 e Array 2) configuradas com dilatações e parâmetros específicos,
/// simulando a topologia padrão de inferência de rede profunda do Neural Amp Modeler.
fn build_wavenet_custom<const CH: usize, const K: usize, const HEAD: usize>(
    dils0: &[usize],
    dils1: &[usize],
    val: f32,
) -> WaveNetModel<CH, K, HEAD> {
    let mut alloc_num = 0;
    let array1 = build_wavenet_layer_array::<1, 1, CH, K, HEAD>(dils0, false, val, &mut alloc_num);
    let array2 = build_wavenet_layer_array::<CH, 1, HEAD, K, 1>(dils1, true, val, &mut alloc_num);
    let rf = array1.receptive_field_size.max(array2.receptive_field_size);
    WaveNetModel {
        array1,
        array2,
        head_scale: val,
        receptive_field_size: rf,
    }
}

/// Preenche os pesos e bias de um modelo LSTM de camada única (`LstmModel1`) com valores constantes.
///
/// Facilita os testes numéricos deterministicos para o modelo LSTM de 1 camada.
fn fill_lstm1_weights<const H: usize, const IH: usize, const H4: usize>(
    model: &mut LstmModel1<H, IH, H4>,
    val: f32,
) {
    let bits = half::f16::from_f32(val).to_bits();
    for k in 0..4 {
        for j in 0..IH {
            model.layer.input_hidden_weights[k][j].fill(bits);
        }
    }
    model.layer.bias.fill(val);
    model.head_weights.fill(bits);
    model.head_bias = val;
}

/// Preenche os pesos e bias de um modelo LSTM de dupla camada (`LstmModel2`) com valores constantes.
///
/// Facilita os testes numéricos deterministicos para o modelo LSTM de 2 camadas.
fn fill_lstm2_weights<const H: usize, const H1_IH: usize, const H2_IH: usize, const H4: usize>(
    model: &mut LstmModel2<H, H1_IH, H2_IH, H4>,
    val: f32,
) {
    let bits = half::f16::from_f32(val).to_bits();
    for k in 0..4 {
        for j in 0..H1_IH {
            model.layer1.input_hidden_weights[k][j].fill(bits);
        }
    }
    model.layer1.bias.fill(val);
    for k in 0..4 {
        for j in 0..H2_IH {
            model.layer2.input_hidden_weights[k][j].fill(bits);
        }
    }
    model.layer2.bias.fill(val);
    model.head_weights.fill(bits);
    model.head_bias = val;
}

#[test]
fn test_golden_wavenet_standard() {
    // Testa a topologia WaveNet Standard (16 canais, kernel 3, head 8)
    // Garantindo que a inferência do áudio reproduz o resultado binário salvo sem drifts
    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];
    let expected = load_golden("wavenet_standard");
    let dils = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let mut model = build_wavenet_custom::<16, 3, 8>(&dils, &dils, 0.01);
    model.prewarm();
    model.process(&input, &mut output);
    let mse = calculate_mse(&output, &expected);
    assert!(mse < 1e-6, "WaveNet Standard MSE regressão falhou: {}", mse);
}

#[test]
fn test_golden_wavenet_lite() {
    // Testa a topologia WaveNet Lite (12 canais, kernel 3, head 6)
    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];
    let expected = load_golden("wavenet_lite");
    let dils0 = [1, 2, 4, 8, 16, 32, 64];
    let dils1 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let mut model = build_wavenet_custom::<12, 3, 6>(&dils0, &dils1, 0.01);
    model.prewarm();
    model.process(&input, &mut output);
    let mse = calculate_mse(&output, &expected);
    assert!(mse < 1e-6, "WaveNet Lite MSE regressão falhou: {}", mse);
}

#[test]
fn test_golden_wavenet_feather() {
    // Testa a topologia WaveNet Feather (8 canais, kernel 3, head 4)
    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];
    let expected = load_golden("wavenet_feather");
    let dils0 = [1, 2, 4, 8, 16, 32, 64];
    let dils1 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let mut model = build_wavenet_custom::<8, 3, 4>(&dils0, &dils1, 0.01);
    model.prewarm();
    model.process(&input, &mut output);
    let mse = calculate_mse(&output, &expected);
    assert!(mse < 1e-6, "WaveNet Feather MSE regressão falhou: {}", mse);
}

#[test]
fn test_golden_wavenet_nano() {
    // Testa a topologia WaveNet Nano (4 canais, kernel 3, head 2)
    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];
    let expected = load_golden("wavenet_nano");
    let dils0 = [1, 2, 4, 8, 16, 32, 64];
    let dils1 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let mut model = build_wavenet_custom::<4, 3, 2>(&dils0, &dils1, 0.01);
    model.prewarm();
    model.process(&input, &mut output);
    let mse = calculate_mse(&output, &expected);
    assert!(mse < 1e-6, "WaveNet Nano MSE regressão falhou: {}", mse);
}

#[test]
fn test_golden_lstm_1x8() {
    // Testa a topologia LSTM 1x8 (1 camada, 8 neurônios de estado oculto)
    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];
    let expected = load_golden("lstm_1x8");
    let mut model = LstmModel1::<8, 9, 32>::new();
    fill_lstm1_weights(&mut model, 0.01);
    model.reset_states();
    model.process(&input, &mut output);
    let mse = calculate_mse(&output, &expected);
    assert!(mse < 1e-6, "LSTM 1x8 MSE regressão falhou: {}", mse);
}

#[test]
fn test_golden_lstm_1x16() {
    // Testa a topologia LSTM 1x16 (1 camada, 16 neurônios de estado oculto)
    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];
    let expected = load_golden("lstm_1x16");
    let mut model = LstmModel1::<16, 17, 64>::new();
    fill_lstm1_weights(&mut model, 0.01);
    model.reset_states();
    model.process(&input, &mut output);
    let mse = calculate_mse(&output, &expected);
    assert!(mse < 1e-6, "LSTM 1x16 MSE regressão falhou: {}", mse);
}

#[test]
fn test_golden_lstm_2x16() {
    // Testa a topologia LSTM 2x16 (2 camadas com 16 neurônios cada)
    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];
    let expected = load_golden("lstm_2x16");
    let mut model = LstmModel2::<16, 17, 32, 64>::new();
    fill_lstm2_weights(&mut model, 0.01);
    model.reset_states();
    model.process(&input, &mut output);
    let mse = calculate_mse(&output, &expected);
    assert!(mse < 1e-6, "LSTM 2x16 MSE regressão falhou: {}", mse);
}

// =============================================================================
// Regeneração de Golden Vectors
// =============================================================================

fn save_golden(name: &str, data: &[f32]) {
    let path = format!("tests/golden/{}.bin", name);
    let mut file = File::create(path).expect("Falha ao criar arquivo golden");
    let bytes: Vec<u8> = data
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();
    file.write_all(&bytes)
        .expect("Falha ao escrever dados golden");
}

/// Regenera todos os golden vectors de referência.
///
/// Este teste é marcado como `#[ignore]` pois realiza I/O (escrita de arquivos .bin)
/// e só deve ser executado manualmente quando uma otimização validada alterar a saída
/// numérica dos modelos.
///
/// ## Uso
///
/// ```sh
/// cargo test regenerate_goldens -- --ignored
/// ```
#[test]
#[ignore]
fn regenerate_goldens() {
    create_dir_all("tests/golden").expect("Falha ao criar diretório tests/golden");

    let input = generate_sine(440.0, 48000.0, 1024);
    let mut output = vec![0.0f32; 1024];

    let std_dils = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let lite_dils_0 = [1, 2, 4, 8, 16, 32, 64];
    let lite_dils_1 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    // WaveNet Standard
    {
        let mut model = build_wavenet_custom::<16, 3, 8>(&std_dils, &std_dils, 0.01);
        model.prewarm();
        model.process(&input, &mut output);
        save_golden("wavenet_standard", &output);
    }

    // WaveNet Lite
    {
        let mut model = build_wavenet_custom::<12, 3, 6>(&lite_dils_0, &lite_dils_1, 0.01);
        model.prewarm();
        model.process(&input, &mut output);
        save_golden("wavenet_lite", &output);
    }

    // WaveNet Feather
    {
        let mut model = build_wavenet_custom::<8, 3, 4>(&lite_dils_0, &lite_dils_1, 0.01);
        model.prewarm();
        model.process(&input, &mut output);
        save_golden("wavenet_feather", &output);
    }

    // WaveNet Nano
    {
        let mut model = build_wavenet_custom::<4, 3, 2>(&lite_dils_0, &lite_dils_1, 0.01);
        model.prewarm();
        model.process(&input, &mut output);
        save_golden("wavenet_nano", &output);
    }

    // LSTM 1x8
    {
        let mut model = LstmModel1::<8, 9, 32>::new();
        fill_lstm1_weights(&mut model, 0.01);
        model.reset_states();
        model.process(&input, &mut output);
        save_golden("lstm_1x8", &output);
    }

    // LSTM 1x16
    {
        let mut model = LstmModel1::<16, 17, 64>::new();
        fill_lstm1_weights(&mut model, 0.01);
        model.reset_states();
        model.process(&input, &mut output);
        save_golden("lstm_1x16", &output);
    }

    // LSTM 2x16
    {
        let mut model = LstmModel2::<16, 17, 32, 64>::new();
        fill_lstm2_weights(&mut model, 0.01);
        model.reset_states();
        model.process(&input, &mut output);
        save_golden("lstm_2x16", &output);
    }
}
