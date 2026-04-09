// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Model Dispatcher — converte `NamModelData` (parsers JSON/NAMB) em `Box<DynamicModel>`.
//!
//! Toda a lógica de construção e alocação ocorre exclusivamente na thread CLI,
//! garantindo que o `Box<DynamicModel>` resultante esteja pronto para injeção na
//! thread DSP via SPSC sem nenhuma alocação no caminho RT.
//!
//! Os pesos são consumidos sequencialmente por um `WeightCursor` cursor-forward,
//! com verificação de exaustão ao final para detectar modelos inconsistentes.

use anyhow::{Context, bail};

use crate::loader::nam_json::{
    NamModelData, NamWavenetTopology, get_lstm_topology, get_wavenet_topology,
};
use crate::models::DynamicModel;
use crate::models::lstm::{LstmLayer, LstmModel1, LstmModel2};
use crate::models::wavenet::{
    Conv1d, DenseLayer, WaveNetLayer, WaveNetLayerArray, WaveNetLayerState, WaveNetModel,
};

// =============================================================================
// WeightCursor — Leitura sequencial determinística dos pesos planificados
// =============================================================================

/// Cursor de leitura forward-only sobre o vetor de pesos planificados.
///
/// Garante:
/// - Nenhum peso é lido fora dos limites (`read_slice` / `read_f32`)
/// - Todos os pesos foram consumidos no final (`verify_exhausted`)
struct WeightCursor<'a> {
    /// Referência ao slice completo de pesos do modelo.
    data: &'a [f32],
    /// Posição corrente do cursor (avança a cada leitura).
    pos: usize,
}

impl<'a> WeightCursor<'a> {
    /// Cria um novo cursor sobre a fatia de pesos.
    fn new(data: &'a [f32]) -> Self {
        Self { data, pos: 0 }
    }

    /// Lê uma fatia contígua de `len` pesos, avançando o cursor.
    fn read_slice(&mut self, len: usize) -> anyhow::Result<&'a [f32]> {
        if self.pos + len > self.data.len() {
            bail!(
                "Pesos insuficientes: necessários {} a partir da posição {}, disponíveis {}",
                len,
                self.pos,
                self.data.len()
            );
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    /// Lê um único escalar `f32`, avançando o cursor.
    fn read_f32(&mut self) -> anyhow::Result<f32> {
        let s = self.read_slice(1)?;
        Ok(s[0])
    }

    /// Verifica que todos os pesos foram consumidos. Falha se restarem pesos.
    fn verify_exhausted(&self) -> anyhow::Result<()> {
        if self.pos != self.data.len() {
            bail!(
                "Modelo com pesos inconsistentes: consumidos {}, total {}",
                self.pos,
                self.data.len()
            );
        }
        Ok(())
    }
}

// =============================================================================
// Ponto de Entrada Público
// =============================================================================

/// Constrói um `Box<DynamicModel>` a partir dos dados brutos parseados.
///
/// Bifurca por arquitetura (`"WaveNet"` / `"LSTM"`) e delega para os
/// construtores especializados com const generics.
pub fn build_model(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    match data.architecture.as_str() {
        "WaveNet" => build_wavenet(data),
        "LSTM" => build_lstm(data),
        other => bail!("Arquitetura não suportada: '{}'", other),
    }
}

// =============================================================================
// WaveNet — Construção por Topologia
// =============================================================================

/// Detecta a topologia do WaveNet e bifurca para o construtor const-generic correto.
fn build_wavenet(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    let topo = get_wavenet_topology(data)
        .context("Topologia WaveNet desconhecida (verifique canais, dilatações e flags)")?;

    match topo {
        NamWavenetTopology::Standard => build_wavenet_typed::<16, 3, 8>(data, topo),
        NamWavenetTopology::Lite => build_wavenet_typed::<12, 3, 6>(data, topo), // C++: InternalWaveNetDefinitionT<12, 6>
        NamWavenetTopology::Feather => build_wavenet_typed::<8, 3, 4>(data, topo),
        NamWavenetTopology::Nano => build_wavenet_typed::<4, 3, 2>(data, topo),
    }
}

/// Constrói um `WaveNetModel<CH, K, HEAD>` com pesos lidos sequencialmente.
///
/// Layout dos pesos (C++ WaveNet.h `SetWeights`):
/// ```text
/// [array1.rechannel] [array1.layers...] [array1.head_rechannel]
/// [array2.rechannel] [array2.layers...] [array2.head_rechannel]
/// [head_scale]
/// ```
fn build_wavenet_typed<const CH: usize, const K: usize, const HEAD: usize>(
    data: &NamModelData,
    topo: NamWavenetTopology,
) -> anyhow::Result<Box<DynamicModel>> {
    let mut cursor = WeightCursor::new(&data.weights);

    // Extrair dilatações de cada array a partir da configuração JSON
    let l0 = &data.config.layers[0];
    let l1 = &data.config.layers[1];
    let dils_0 = l0
        .dilations
        .as_deref()
        .context("Dilatações ausentes na layer 0")?;
    let dils_1 = l1
        .dilations
        .as_deref()
        .context("Dilatações ausentes na layer 1")?;

    // Rastrear allocNum global conforme C++ WaveNetModelT::constructor
    let mut alloc_num = 0usize;

    // Array 1: IN=1, COND=1, HasHeadBias=false
    let array1 = build_wavenet_array::<1, 1, CH, K, HEAD>(
        &mut cursor,
        dils_0,
        false, // HasHeadBias da array1 (C++: false)
        &mut alloc_num,
    )?;

    // Array 2: IN=CH, COND=1, CH2=HEAD, HEAD2=1, HasHeadBias=true
    // C++: WaveNetLayerArrayT<CH, 1, 1, HEAD, K, Dilations, true>
    let array2 = build_wavenet_array::<CH, 1, HEAD, K, 1>(
        &mut cursor,
        dils_1,
        true, // HasHeadBias da array2 (C++: true)
        &mut alloc_num,
    )?;

    // Último peso do modelo: head_scale (C++ WaveNet.h L372)
    let head_scale = cursor.read_f32()?;

    // Garante exaustão completa dos pesos
    cursor.verify_exhausted()?;

    let rf = array1.receptive_field_size.max(array2.receptive_field_size);

    let model = WaveNetModel::<CH, K, HEAD> {
        array1,
        array2,
        head_scale,
        receptive_field_size: rf,
    };

    println!(
        "[Dispatcher] WaveNet {:?} construído — CH={}, K={}, HEAD={}, head_scale={:.6}, pesos={}",
        topo,
        CH,
        K,
        HEAD,
        head_scale,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel(Box::new(model))))
}

/// Constrói uma `WaveNetLayerArray` lendo pesos cursor-forward.
///
/// Layout por array (C++ `WaveNetLayerArrayT::SetWeights`):
/// ```text
/// rechannel.weights[IN*CH]
/// for layer in layers:
///     conv1d.weights[CH*K*CH] + conv1d.bias[CH]         (DoBias=true)
///     input_mixin.weights[COND*CH]                       (DoBias=false)
///     one_by_one.weights[CH*CH] + one_by_one.bias[CH]    (DoBias=true)
/// head_rechannel.weights[CH*HEAD] + head_rechannel.bias[HEAD]? (HasHeadBias)
/// ```
fn build_wavenet_array<
    const IN: usize,
    const COND: usize,
    const CH: usize,
    const K: usize,
    const HEAD: usize,
>(
    cursor: &mut WeightCursor<'_>,
    dilations: &[usize],
    has_head_bias: bool,
    alloc_num: &mut usize,
) -> anyhow::Result<WaveNetLayerArray<IN, COND, CH, K, HEAD>> {
    // 1. Rechannel: Dense<IN, CH> sem bias (C++: DenseLayerT<InputSize, Channels, false>)
    let rechannel = read_dense_layer::<IN, CH>(cursor, false)?;

    // 2. Layers: uma para cada dilatação
    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        // conv1d: Conv1d<CH, CH, K> com bias (C++: Conv1DT<Ch, Ch, K, true, D>)
        let conv1d = read_conv1d_weights::<CH, CH, K>(cursor, dilation, true)?;
        // input_mixin: Dense<COND, CH> sem bias (C++: DenseLayerT<Cond, Ch, false>)
        let input_mixin = read_dense_layer::<COND, CH>(cursor, false)?;
        // one_by_one: Dense<CH, CH> com bias (C++: DenseLayerT<Ch, Ch, true>)
        let one_by_one = read_dense_layer::<CH, CH>(cursor, true)?;

        layers.push(WaveNetLayer {
            conv1d,
            input_mixin,
            one_by_one,
        });

        // Estado do Ring Buffer com RF per-layer: (K-1) * dilation
        let rf = (K - 1) * dilation;
        states.push(WaveNetLayerState::new(CH, rf, *alloc_num));
        *alloc_num += 1;
    }

    // 3. Head Rechannel: Dense<CH, HEAD> bias condicional (HasHeadBias)
    let head_rechannel = read_dense_layer::<CH, HEAD>(cursor, has_head_bias)?;

    // Campo receptivo total do array = soma dos RF individuais
    let receptive_field_size: usize = dilations.iter().map(|&d| (K - 1) * d).sum();

    Ok(WaveNetLayerArray {
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: vec![0.0; CH],
        head_accum: vec![0.0; CH],
        head_outputs: vec![0.0; HEAD],
        receptive_field_size,
    })
}

// =============================================================================
// LSTM — Construção por Geometria
// =============================================================================

/// Detecta a geometria LSTM (num_layers × hidden_size) e despacha ao construtor correto.
fn build_lstm(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    let (num_layers, hidden_size) = get_lstm_topology(data)
        .context("Geometria LSTM não detectável (verifique num_layers e hidden_size)")?;

    match (num_layers, hidden_size) {
        (1, 8) => build_lstm_1layer::<8, 9, 32>(data, hidden_size),
        (1, 12) => build_lstm_1layer::<12, 13, 48>(data, hidden_size),
        (1, 16) => build_lstm_1layer::<16, 17, 64>(data, hidden_size),
        (1, 24) => build_lstm_1layer::<24, 25, 96>(data, hidden_size),
        (2, 8) => build_lstm_2layer::<8, 9, 16, 32>(data, num_layers, hidden_size),
        (2, 12) => build_lstm_2layer::<12, 13, 24, 48>(data, num_layers, hidden_size),
        (2, 16) => build_lstm_2layer::<16, 17, 32, 64>(data, num_layers, hidden_size),
        _ => bail!(
            "Geometria LSTM não suportada: {}×{}",
            num_layers,
            hidden_size
        ),
    }
}

/// Constrói um `LstmModel1<H, H1_IH, H_H4>` com pesos lidos sequencialmente.
///
/// Layout LSTM NAM (C++ `LSTMLayerT::SetNAMWeights`):
/// ```text
/// layer.input_hidden_weights[H4 * IH]  (row-major)
/// layer.bias[H4]
/// layer.initial_hidden_state[H]
/// layer.initial_cell_state[H]
/// head_weights[H]
/// head_bias
/// ```
fn build_lstm_1layer<const H: usize, const H1_IH: usize, const H_H4: usize>(
    data: &NamModelData,
    hidden_size: usize,
) -> anyhow::Result<Box<DynamicModel>> {
    let mut cursor = WeightCursor::new(&data.weights);

    // Layer 1: input_size=1
    let layer = read_lstm_layer::<1, H, H1_IH, H_H4>(&mut cursor)?;

    // Head: pesos da projeção linear de saída
    let head_weights_data = cursor.read_slice(H)?;
    let mut head_weights = [0.0f32; H];
    head_weights.copy_from_slice(head_weights_data);
    let head_bias = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let model = LstmModel1::<H, H1_IH, H_H4> {
        layer,
        head_weights,
        head_bias,
    };

    println!(
        "[Dispatcher] LSTM 1×{} construído — pesos={}",
        hidden_size,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel(Box::new(model))))
}

/// Constrói um `LstmModel2<H, H1_IH, H2_IH, H_H4>` com pesos lidos sequencialmente.
fn build_lstm_2layer<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize>(
    data: &NamModelData,
    num_layers: usize,
    hidden_size: usize,
) -> anyhow::Result<Box<DynamicModel>> {
    let mut cursor = WeightCursor::new(&data.weights);

    // Layer 1: input_size=1
    let layer1 = read_lstm_layer::<1, H, H1_IH, H_H4>(&mut cursor)?;

    // Layer 2: input_size=H (estado oculto da camada anterior)
    let layer2 = read_lstm_layer::<H, H, H2_IH, H_H4>(&mut cursor)?;

    // Head: pesos da projeção final
    let head_weights_data = cursor.read_slice(H)?;
    let mut head_weights = [0.0f32; H];
    head_weights.copy_from_slice(head_weights_data);
    let head_bias = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let model = LstmModel2::<H, H1_IH, H2_IH, H_H4> {
        layer1,
        layer2,
        head_weights,
        head_bias,
    };

    println!(
        "[Dispatcher] LSTM {}×{} construído — pesos={}",
        num_layers,
        hidden_size,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel(Box::new(model))))
}

// =============================================================================
// Helpers — Leitura de Componentes Individuais
// =============================================================================

/// Lê os pesos de um `Conv1d<IN, OUT, K>` aplicando transposição de layout.
///
/// O C++ lê `[out][in][k]` (Conv1DT::SetWeights) mas o Rust armazena
/// como `[out * K * IN + k * IN + in_c]` — requer scatter do eixo `k ↔ in`.
fn read_conv1d_weights<const IN: usize, const OUT: usize, const K: usize>(
    cursor: &mut WeightCursor<'_>,
    dilation: usize,
    do_bias: bool,
) -> anyhow::Result<Conv1d<IN, OUT, K>> {
    let total = OUT * IN * K;
    let raw = cursor.read_slice(total)?;

    // Transposição: C++ lê (out, in, k) → Rust layout (out, k, in)
    let mut weights = vec![0.0f32; total];
    let mut idx = 0;
    for out_c in 0..OUT {
        for in_c in 0..IN {
            for k in 0..K {
                weights[out_c * K * IN + k * IN + in_c] = raw[idx];
                idx += 1;
            }
        }
    }

    let bias = if do_bias {
        cursor.read_slice(OUT)?.to_vec()
    } else {
        vec![0.0; OUT]
    };

    Ok(Conv1d {
        weights,
        bias,
        do_bias,
        dilation,
    })
}

/// Lê os pesos de um `DenseLayer<IN, OUT>` (layout compatível sem transposição).
///
/// Tanto C++ (`DenseLayerT::SetWeights`) quanto Rust usam layout `[out][in]` row-major.
fn read_dense_layer<const IN: usize, const OUT: usize>(
    cursor: &mut WeightCursor<'_>,
    do_bias: bool,
) -> anyhow::Result<DenseLayer<IN, OUT>> {
    let weights = cursor.read_slice(OUT * IN)?.to_vec();

    let bias = if do_bias {
        cursor.read_slice(OUT)?.to_vec()
    } else {
        vec![0.0; OUT]
    };

    Ok(DenseLayer {
        weights,
        bias,
        do_bias,
    })
}

/// Lê os pesos de uma `LstmLayer<I, H, IH, H4>`.
///
/// Layout NAM JSON (C++ `LSTMLayerT::SetNAMWeights`):
/// ```text
/// input_hidden_weights: [H4 rows × IH cols] — row-major, mapeamento direto
/// bias:                 [H4]
/// initial_hidden:       [H]  → state[I..I+H]
/// initial_cell_state:   [H]  → cell_state[0..H]
/// ```
fn read_lstm_layer<const I: usize, const H: usize, const IH: usize, const H4: usize>(
    cursor: &mut WeightCursor<'_>,
) -> anyhow::Result<LstmLayer<I, H, IH, H4>> {
    let mut layer = LstmLayer::<I, H, IH, H4>::new();

    // 1. input_hidden_weights: [H4][IH] row-major (equivalente ao C++ SetNAMWeights)
    for row in 0..H4 {
        let row_data = cursor.read_slice(IH)?;
        layer.input_hidden_weights[row].copy_from_slice(row_data);
    }

    // 2. bias: [H4] valores
    let bias_data = cursor.read_slice(H4)?;
    layer.bias.copy_from_slice(bias_data);

    // 3. initial hidden state: [H] → armazenado em state[I..I+H]
    let hidden_init = cursor.read_slice(H)?;
    layer.state[I..I + H].copy_from_slice(hidden_init);

    // 4. initial cell state: [H]
    let cell_init = cursor.read_slice(H)?;
    layer.cell_state.copy_from_slice(cell_init);

    Ok(layer)
}

// =============================================================================
// Testes Unitários
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::nam_json::{NamConfig, NamLayerConfig, NamModelData};

    /// Gera um `NamModelData` WaveNet sintético com pesos zerados (0.01).
    fn make_wavenet_data(
        channels: usize,
        head_size: usize,
        dils_0: &[usize],
        dils_1: &[usize],
        total_weights: usize,
    ) -> NamModelData {
        NamModelData {
            version: Some("0.5.4".to_string()),
            architecture: "WaveNet".to_string(),
            config: NamConfig {
                layers: vec![
                    NamLayerConfig {
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: Some(head_size),
                        channels: Some(channels),
                        kernel_size: Some(3),
                        dilations: Some(dils_0.to_vec()),
                        activation: Some("Tanh".to_string()),
                        gated: Some(false),
                        head_bias: Some(false),
                    },
                    NamLayerConfig {
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: Some(head_size),
                        channels: Some(channels),
                        kernel_size: Some(3),
                        dilations: Some(dils_1.to_vec()),
                        activation: Some("Tanh".to_string()),
                        gated: Some(false),
                        head_bias: Some(true),
                    },
                ],
                head: None,
                head_scale: Some(0.02),
                num_layers: None,
                hidden_size: None,
            },
            weights: vec![0.01; total_weights],
            sample_rate: Some(48000.0),
            metadata: None,
        }
    }

    /// Gera um `NamModelData` LSTM sintético com pesos zerados (0.01).
    fn make_lstm_data(num_layers: usize, hidden_size: usize, total_weights: usize) -> NamModelData {
        NamModelData {
            version: Some("0.5.4".to_string()),
            architecture: "LSTM".to_string(),
            config: NamConfig {
                layers: vec![],
                head: None,
                head_scale: None,
                num_layers: Some(num_layers),
                hidden_size: Some(hidden_size),
            },
            weights: vec![0.01; total_weights],
            sample_rate: Some(48000.0),
            metadata: None,
        }
    }

    // ---- WaveNet Standard: CH=16, K=3, HEAD=8, 10+10 layers -----------------
    // Array1: rechannel(16) + 10×(conv(768+16)+mixin(16)+o2o(256+16)) + head(16×8=128)    = 10864
    // Array2: rechannel(16×8=128) + 10×(conv(192+8)+mixin(8)+o2o(64+8)) + head(8×1+1=9) = 2937
    // head_scale: 1 → Total: 13802

    #[test]
    fn test_build_wavenet_standard() {
        let std_d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let data = make_wavenet_data(16, 8, &std_d, &std_d, 13802);
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "Falha ao construir WaveNet Standard: {:?}",
            result.err()
        );
    }

    // ---- WaveNet Feather: CH=8, K=3, HEAD=4, 7+13 layers --------------------
    // Array1: rechannel(8) + 7×(conv(192+8)+mixin(8)+o2o(64+8)) + head(8×4=32)         = 2000
    // Array2: rechannel(8×4=32) + 13×(conv(48+4)+mixin(4)+o2o(16+4)) + head(4×1+1=5) = 1025
    // head_scale: 1 → Total: 3026

    #[test]
    fn test_build_wavenet_feather() {
        let lite_d = [1, 2, 4, 8, 16, 32, 64];
        let lite_d2 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let data = make_wavenet_data(8, 4, &lite_d, &lite_d2, 3026);
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "Falha ao construir WaveNet Feather: {:?}",
            result.err()
        );
    }

    // ---- LSTM 1×8 -----------------------------------------------------------
    // Layer: 32*9 + 32 + 8 + 8 = 336
    // Head: 8 + 1 = 9 → Total: 345

    #[test]
    fn test_build_lstm1x8() {
        let data = make_lstm_data(1, 8, 345);
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "Falha ao construir LSTM 1×8: {:?}",
            result.err()
        );
    }

    // ---- LSTM 2×16 ----------------------------------------------------------
    // Layer1: 64*17+64+16+16 = 1184
    // Layer2: 64*32+64+16+16 = 2144
    // Head: 16+1 = 17 → Total: 3345

    #[test]
    fn test_build_lstm2x16() {
        let data = make_lstm_data(2, 16, 3345);
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "Falha ao construir LSTM 2×16: {:?}",
            result.err()
        );
    }

    // ---- Rejeição: arquitetura desconhecida ----------------------------------

    #[test]
    fn test_reject_unknown_architecture() {
        let data = NamModelData {
            version: Some("0.5.4".to_string()),
            architecture: "ResNet".to_string(),
            config: NamConfig {
                layers: vec![],
                head: None,
                head_scale: None,
                num_layers: None,
                hidden_size: None,
            },
            weights: vec![0.01; 100],
            sample_rate: Some(48000.0),
            metadata: None,
        };
        let result = build_model(&data);
        assert!(result.is_err(), "Deveria rejeitar arquitetura 'ResNet'");
    }

    // ---- Rejeição: pesos insuficientes (underflow) --------------------------

    #[test]
    fn test_reject_weight_underflow() {
        let std_d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        // Standard exige 13802 pesos; fornecer apenas 100 deve falhar
        let data = make_wavenet_data(16, 8, &std_d, &std_d, 100);
        let result = build_model(&data);
        assert!(result.is_err(), "Deveria falhar com pesos insuficientes");
    }

    // =========================================================================
    // Sprint 8.3/T-4 — Rejeição de topologias não-suportadas
    // =========================================================================

    /// T-4: WaveNet com channels=32 não é suportado — deve retornar Err.
    #[test]
    fn test_reject_wavenet_unsupported_channels() {
        let std_d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        // channels=32 não é Standard(16), Lite(12), Feather(8) ou Nano(4)
        let data = make_wavenet_data(32, 16, &std_d, &std_d, 100_000);
        let result = build_model(&data);
        assert!(
            result.is_err(),
            "Deveria rejeitar WaveNet com channels=32 (topologia não suportada)"
        );
    }

    /// T-4: LSTM com 3 camadas não é suportado — deve retornar Err.
    #[test]
    fn test_reject_lstm_unsupported_geometry() {
        let data = make_lstm_data(3, 8, 10_000);
        let result = build_model(&data);
        assert!(
            result.is_err(),
            "Deveria rejeitar LSTM 3×8 (geometria não suportada)"
        );
    }

    // =========================================================================
    // Sprint 8.3/T-6 — Exaustão do WeightCursor para todas as topologias
    // =========================================================================

    /// T-6: Verifica que `build_model()` consome 100% dos pesos para cada perfil WaveNet.
    ///
    /// Contagem de pesos por topologia (calculada manualmente a partir do layout C++):
    /// - Standard (CH=16, K=3, HEAD=8, 10+10 layers):   13802
    /// - Lite     (CH=12, K=3, HEAD=6, 7+13 layers):     6554
    /// - Feather  (CH=8,  K=3, HEAD=4, 7+13 layers):     3026
    /// - Nano     (CH=4,  K=3, HEAD=2, 7+13 layers):      842
    #[test]
    fn test_weight_exhaustion_all_wavenet_topologies() {
        let std_d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        let lite_d0 = [1, 2, 4, 8, 16, 32, 64];
        let lite_d1 = [128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

        // Standard: CH=16, HEAD=8, 13802 pesos
        let data = make_wavenet_data(16, 8, &std_d, &std_d, 13802);
        assert!(
            build_model(&data).is_ok(),
            "WaveNet Standard com 13802 pesos deveria passar"
        );

        // Lite: CH=12, HEAD=6, 6554 pesos
        let data = make_wavenet_data(12, 6, &lite_d0, &lite_d1, 6554);
        assert!(
            build_model(&data).is_ok(),
            "WaveNet Lite com 6554 pesos deveria passar"
        );

        // Feather: CH=8, HEAD=4, 3026 pesos
        let data = make_wavenet_data(8, 4, &lite_d0, &lite_d1, 3026);
        assert!(
            build_model(&data).is_ok(),
            "WaveNet Feather com 3026 pesos deveria passar"
        );

        // Nano: CH=4, HEAD=2, 842 pesos
        let data = make_wavenet_data(4, 2, &lite_d0, &lite_d1, 842);
        assert!(
            build_model(&data).is_ok(),
            "WaveNet Nano com 842 pesos deveria passar"
        );
    }

    /// T-6: Verifica exaustão de pesos para todos os perfis LSTM suportados.
    ///
    /// Layout: layer(H4*IH + H4 + H + H) + head(H + 1).
    /// - 1×8:  345,  1×12: 709,  1×16: 1201, 1×24: 2569
    /// - 2×8:  905,  2×12: 1933, 2×16: 3345
    #[test]
    fn test_weight_exhaustion_all_lstm_topologies() {
        let cases: &[(&str, usize, usize, usize)] = &[
            ("1×8", 1, 8, 345),
            ("1×12", 1, 12, 709),
            ("1×16", 1, 16, 1201),
            ("1×24", 1, 24, 2569),
            ("2×8", 2, 8, 905),
            ("2×12", 2, 12, 1933),
            ("2×16", 2, 16, 3345),
        ];

        for &(name, nl, hs, w) in cases {
            let data = make_lstm_data(nl, hs, w);
            let result = build_model(&data);
            assert!(
                result.is_ok(),
                "LSTM {name} com {w} pesos deveria passar, mas falhou: {:?}",
                result.err()
            );
        }
    }

    /// T-6: Verifica que 1 peso extra causa falha (overflow de cursor).
    #[test]
    fn test_weight_overflow_extra_peso() {
        let std_d = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        // Standard exige 13802 pesos; fornecer 13803 (1 extra) deve falhar na exaustão
        let data = make_wavenet_data(16, 8, &std_d, &std_d, 13803);
        let result = build_model(&data);
        assert!(
            result.is_err(),
            "Deveria falhar com 1 peso extra (overflow do cursor)"
        );
    }
}
