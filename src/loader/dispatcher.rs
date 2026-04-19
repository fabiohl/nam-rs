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
use log::info;

use crate::loader::nam_json::{
    NamModelData, NamWavenetTopology, get_lstm_topology, get_wavenet_topology,
};
use crate::models::DynamicModel;
use crate::models::lstm::{LstmLayer, LstmModel1, LstmModel2};
use crate::models::lstm_dyn::{LstmDynLayer, LstmDynModel};
use crate::models::wavenet::{
    Conv1d, DenseLayer, WaveNetLayer, WaveNetLayerArray, WaveNetLayerState, WaveNetModel,
};
use crate::models::wavenet_dyn::{
    Conv1dDyn, DenseLayerDyn, WaveNetDynModel, WaveNetLayerArrayDyn, WaveNetLayerDyn,
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
// WaveNet — Validação de Activation
// =============================================================================

/// Valida o campo `activation` em todas as layers de um modelo WaveNet.
///
/// Percorre `data.config.layers` e rejeita com erro descritivo qualquer valor
/// que não seja `"Tanh"` (único tipo de ativação suportado). `None` é aceito
/// e tratado como `"Tanh"` por compatibilidade com modelos legados.
///
/// # Errors
/// Retorna `Err` se alguma layer declarar `activation != "Tanh"`.
fn validate_layer_activations(data: &NamModelData) -> anyhow::Result<()> {
    for (idx, layer) in data.config.layers.iter().enumerate() {
        let act = layer.activation.as_deref().unwrap_or("Tanh");
        if act != "Tanh" {
            bail!(
                "Ativação '{}' na layer {} não é suportada. Apenas 'Tanh' é implementado.",
                act,
                idx
            );
        }
    }
    Ok(())
}

// =============================================================================
// WaveNet — Construção por Topologia
// =============================================================================

/// Detecta a topologia do WaveNet e bifurca para o construtor const-generic correto.
fn build_wavenet(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    let topo_opt = get_wavenet_topology(data);

    match topo_opt {
        Some(NamWavenetTopology::Standard) => {
            build_wavenet_typed::<16, 3, 8>(data, NamWavenetTopology::Standard)
        }
        Some(NamWavenetTopology::Lite) => {
            build_wavenet_typed::<12, 3, 6>(data, NamWavenetTopology::Lite)
        }
        Some(NamWavenetTopology::Feather) => {
            build_wavenet_typed::<8, 3, 4>(data, NamWavenetTopology::Feather)
        }
        Some(NamWavenetTopology::Nano) => {
            build_wavenet_typed::<4, 3, 2>(data, NamWavenetTopology::Nano)
        }
        None => build_wavenet_dynamic(data),
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
    // Valida ativações antes de qualquer leitura de pesos
    validate_layer_activations(data)?;

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

    info!(
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
// WaveNet — Construtor Dinâmico (Fallback)
// =============================================================================

/// Constrói um `WaveNetDynModel` com pesos lidos sequencialmente (fallback dinâmico).
///
/// Visível publicamente para testes de paridade numérica dinâmico ↔ estático.
pub fn build_wavenet_dynamic(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    if data.config.layers.len() != 2 {
        bail!("WaveNet dinâmico exige 2 arrays");
    }

    // Valida ativações antes de qualquer leitura de pesos
    validate_layer_activations(data)?;

    let mut cursor = WeightCursor::new(&data.weights);

    let l0 = &data.config.layers[0];
    let l1 = &data.config.layers[1];

    let ch1 = l0.channels.context("Layer 0: sem channels")?;
    let k1 = l0.kernel_size.unwrap_or(3);
    let head1 = l0.head_size.context("Layer 0: sem head_size")?;
    let dils_0 = l0.dilations.as_deref().context("Layer 0: sem dilations")?;
    let b1 = l0.head_bias.unwrap_or(false);

    let dils_1 = l1.dilations.as_deref().context("Layer 1: sem dilations")?;
    let b2 = l1.head_bias.unwrap_or(true);

    let mut alloc_num = 0usize;

    let array1 = build_wavenet_array_dyn(
        &mut cursor,
        1,
        1,
        ch1,
        k1,
        head1,
        dils_0,
        b1,
        l0.gated.unwrap_or(false),
        &mut alloc_num,
    )?;

    let array2 = build_wavenet_array_dyn(
        &mut cursor,
        ch1,
        1,
        head1,
        k1,
        1, // HEAD2 sempre 1 para mono out
        dils_1,
        b2,
        l1.gated.unwrap_or(false),
        &mut alloc_num,
    )?;

    let head_scale = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let rf = array1.receptive_field_size.max(array2.receptive_field_size);

    let model = WaveNetDynModel {
        array1,
        array2,
        head_scale,
        receptive_field_size: rf,
        head: head1,
    };

    info!(
        "[Dispatcher] WaveNet Dinâmico construído — CH={}, K={}, HEAD={}, PESOS={}",
        ch1,
        k1,
        head1,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel(Box::new(model))))
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
        _ => build_lstm_dynamic(data, num_layers, hidden_size),
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

    info!(
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

    info!(
        "[Dispatcher] LSTM {}×{} construído — pesos={}",
        num_layers,
        hidden_size,
        data.weights.len()
    );

    Ok(Box::new(DynamicModel(Box::new(model))))
}

// =============================================================================
// LSTM — Construtor Dinâmico (Fallback)
// =============================================================================

/// Constrói um `LstmDynModel` com pesos lidos sequencialmente (fallback dinâmico).
///
/// Visível publicamente para testes de paridade numérica dinâmico ↔ estático.
pub fn build_lstm_dynamic(
    data: &NamModelData,
    num_layers: usize,
    hidden_size: usize,
) -> anyhow::Result<Box<DynamicModel>> {
    let mut cursor = WeightCursor::new(&data.weights);
    let mut layers = Vec::with_capacity(num_layers);

    let mut current_input_size = 1;

    for _ in 0..num_layers {
        let raw_weights =
            cursor.read_slice(hidden_size * 4 * (current_input_size + hidden_size))?;

        let ih = current_input_size + hidden_size;
        let mut input_hidden_weights = vec![0.0; raw_weights.len()];

        for i in 0..hidden_size {
            input_hidden_weights[(i * 4) * ih..(i * 4 + 1) * ih]
                .copy_from_slice(&raw_weights[i * ih..(i + 1) * ih]);
            input_hidden_weights[(i * 4 + 1) * ih..(i * 4 + 2) * ih]
                .copy_from_slice(&raw_weights[(i + hidden_size) * ih..(i + hidden_size + 1) * ih]);
            input_hidden_weights[(i * 4 + 2) * ih..(i * 4 + 3) * ih].copy_from_slice(
                &raw_weights[(i + 2 * hidden_size) * ih..(i + 2 * hidden_size + 1) * ih],
            );
            input_hidden_weights[(i * 4 + 3) * ih..(i * 4 + 4) * ih].copy_from_slice(
                &raw_weights[(i + 3 * hidden_size) * ih..(i + 3 * hidden_size + 1) * ih],
            );
        }

        let bias = cursor.read_slice(hidden_size * 4)?.to_vec();

        // initial_hidden_state [H] -> loaded into state[current_input_size..current_input_size+hidden_size]
        let hidden_init = cursor.read_slice(hidden_size)?;
        let mut state = vec![0.0; current_input_size + hidden_size];
        state[current_input_size..current_input_size + hidden_size].copy_from_slice(hidden_init);

        // initial_cell_state [H]
        let cell_init = cursor.read_slice(hidden_size)?;
        let mut cell_state = vec![0.0; hidden_size];
        cell_state.copy_from_slice(cell_init);

        layers.push(LstmDynLayer {
            input_hidden_weights,
            bias,
            state,
            cell_state,
            gates: vec![0.0; hidden_size * 4],
            tanh_cs: vec![0.0; hidden_size],
            input_size: current_input_size,
            hidden_size,
        });

        current_input_size = hidden_size;
    }

    let head_weights = cursor.read_slice(hidden_size)?.to_vec();
    let head_bias = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let model = LstmDynModel {
        layers,
        head_weights,
        head_bias,
    };

    info!(
        "[Dispatcher] LSTM Dinâmico {}×{} construído — pesos={}",
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

    // 1. input_hidden_weights: [H4][IH] intercalado (I, F, C, O por neurônio)
    let raw_weights = cursor.read_slice(H4 * IH)?;
    for i in 0..H {
        layer.input_hidden_weights[i * 4].copy_from_slice(&raw_weights[i * IH..(i + 1) * IH]);
        layer.input_hidden_weights[i * 4 + 1]
            .copy_from_slice(&raw_weights[(i + H) * IH..(i + H + 1) * IH]);
        layer.input_hidden_weights[i * 4 + 2]
            .copy_from_slice(&raw_weights[(i + 2 * H) * IH..(i + 2 * H + 1) * IH]);
        layer.input_hidden_weights[i * 4 + 3]
            .copy_from_slice(&raw_weights[(i + 3 * H) * IH..(i + 3 * H + 1) * IH]);
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

fn read_conv1d_weights_dyn(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    k: usize,
    dilation: usize,
    do_bias: bool,
) -> anyhow::Result<Conv1dDyn> {
    let total = out_size * in_size * k;
    let raw = cursor.read_slice(total)?;

    let mut weights = vec![0.0f32; total];
    let mut idx = 0;
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            for step in 0..k {
                weights[out_c * k * in_size + step * in_size + in_c] = raw[idx];
                idx += 1;
            }
        }
    }

    let bias = if do_bias {
        cursor.read_slice(out_size)?.to_vec()
    } else {
        vec![0.0; out_size]
    };

    Ok(Conv1dDyn {
        weights,
        bias,
        do_bias,
        dilation,
        in_ch: in_size,
        out_ch: out_size,
        kernel: k,
    })
}

fn read_dense_layer_dyn(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    do_bias: bool,
) -> anyhow::Result<DenseLayerDyn> {
    let weights = cursor.read_slice(out_size * in_size)?.to_vec();

    let bias = if do_bias {
        cursor.read_slice(out_size)?.to_vec()
    } else {
        vec![0.0; out_size]
    };

    Ok(DenseLayerDyn {
        weights,
        bias,
        do_bias,
        in_size,
        out_size,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_wavenet_array_dyn(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    cond_size: usize,
    ch: usize,
    k: usize,
    head: usize,
    dilations: &[usize],
    has_head_bias: bool,
    gated: bool,
    alloc_num: &mut usize,
) -> anyhow::Result<WaveNetLayerArrayDyn> {
    // out_ch do conv1d: dobrado quando gated para produzir slots tanh + sigmoid.
    let conv_out_ch = if gated { 2 * ch } else { ch };

    let rechannel = read_dense_layer_dyn(cursor, in_size, ch, false)?;

    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        let conv1d = read_conv1d_weights_dyn(cursor, ch, conv_out_ch, k, dilation, true)?;
        let input_mixin = read_dense_layer_dyn(cursor, cond_size, ch, false)?;
        let one_by_one = read_dense_layer_dyn(cursor, ch, ch, true)?;

        layers.push(WaveNetLayerDyn {
            conv1d,
            input_mixin,
            one_by_one,
            ch,
            gated,
        });

        let rf = (k - 1) * dilation;
        states.push(WaveNetLayerState::new(ch, rf, *alloc_num));
        *alloc_num += 1;
    }

    let head_rechannel = read_dense_layer_dyn(cursor, ch, head, has_head_bias)?;

    let receptive_field_size: usize = dilations.iter().map(|&d| (k - 1) * d).sum();

    // block_size: 2*ch quando gated (slots tanh + sigmoid), ch caso contrário.
    let block_size = if gated { 2 * ch } else { ch };

    Ok(WaveNetLayerArrayDyn {
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: vec![0.0; ch],
        head_accum: vec![0.0; ch],
        head_outputs: vec![0.0; head],
        block_buffer: vec![0.0; block_size],
        block_size,
        receptive_field_size,
        ch,
        head,
    })
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
    // Rejeição de topologias não-suportadas e Fallback dinâmico
    // =========================================================================

    /// Transição para suporte Dinâmico: WaveNet com channels arbitrário funciona via fallback
    #[test]
    fn test_build_wavenet_dynamic_arbitrary_channels() {
        let std_d = [1, 2, 4];
        let data = make_wavenet_data(24, 12, &std_d, &std_d, 9578);
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "Deveria carregar WaveNet com channels=24 dinamicamente: {:?}",
            result.err()
        );
    }

    /// Transição para suporte Dinâmico: LSTM com múltiplas camadas e tamanhos ocultos customizados.
    #[test]
    fn test_build_lstm_dynamic_arbitrary() {
        let data = make_lstm_data(3, 8, 1465); // 336 + 560 + 560 + 9 = 1465
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "Deveria carregar LSTM 3×8 dinamicamente: {:?}",
            result.err()
        );
    }

    // =========================================================================
    // Exaustão do WeightCursor para todas as topologias
    // =========================================================================

    /// Verifica que `build_model()` consome 100% dos pesos para cada perfil WaveNet.
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

    /// Verifica exaustão de pesos para todos os perfis LSTM suportados.
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

    /// Verifica que 1 peso extra causa falha (overflow de cursor).
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

    // =========================================================================
    // T1 · C2 — Validação do campo `activation`
    // =========================================================================

    /// Helper: gera `NamModelData` com `activation` customizado em ambas as layers.
    fn make_wavenet_data_with_activation(
        activation_0: Option<&str>,
        activation_1: Option<&str>,
    ) -> NamModelData {
        let std_d = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
        NamModelData {
            version: Some("0.5.4".to_string()),
            architecture: "WaveNet".to_string(),
            config: NamConfig {
                layers: vec![
                    NamLayerConfig {
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: Some(8),
                        channels: Some(16),
                        kernel_size: Some(3),
                        dilations: Some(std_d.clone()),
                        activation: activation_0.map(str::to_string),
                        gated: Some(false),
                        head_bias: Some(false),
                    },
                    NamLayerConfig {
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: Some(8),
                        channels: Some(16),
                        kernel_size: Some(3),
                        dilations: Some(std_d),
                        activation: activation_1.map(str::to_string),
                        gated: Some(false),
                        head_bias: Some(true),
                    },
                ],
                head: None,
                head_scale: Some(0.02),
                num_layers: None,
                hidden_size: None,
            },
            weights: vec![0.01; 13802],
            sample_rate: Some(48000.0),
            metadata: None,
        }
    }

    /// Modelos com `activation: "ReLU"` devem ser rejeitados com mensagem descritiva.
    #[test]
    fn test_reject_unsupported_activation() {
        // ReLU na layer 0 — deve falhar
        let data = make_wavenet_data_with_activation(Some("ReLU"), Some("Tanh"));
        let result = build_model(&data);
        assert!(
            result.is_err(),
            "Deveria rejeitar activation='ReLU', mas retornou Ok"
        );
        let msg = format!("{:?}", result.err().unwrap());
        assert!(
            msg.contains("ReLU"),
            "Mensagem de erro deveria conter 'ReLU', obteve: {msg}"
        );

        // ReLU na layer 1 — deve falhar igualmente
        let data = make_wavenet_data_with_activation(Some("Tanh"), Some("ReLU"));
        let result = build_model(&data);
        assert!(
            result.is_err(),
            "Deveria rejeitar activation='ReLU' na layer 1, mas retornou Ok"
        );
        let msg = format!("{:?}", result.err().unwrap());
        assert!(
            msg.contains("ReLU"),
            "Mensagem de erro da layer 1 deveria conter 'ReLU', obteve: {msg}"
        );
    }

    /// Modelos com `activation: "Tanh"` devem ser aceitos sem erro.
    #[test]
    fn test_accept_tanh_activation() {
        let data = make_wavenet_data_with_activation(Some("Tanh"), Some("Tanh"));
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "activation='Tanh' deveria ser aceito, mas falhou: {:?}",
            result.err()
        );
    }

    /// Modelos sem campo `activation` (None) devem ser aceitos (default = Tanh).
    #[test]
    fn test_accept_missing_activation() {
        let data = make_wavenet_data_with_activation(None, None);
        let result = build_model(&data);
        assert!(
            result.is_ok(),
            "activation=None (default Tanh) deveria ser aceito, mas falhou: {:?}",
            result.err()
        );
    }

    // =========================================================================
    // T2 · C1 — Testes de Gated Activations no Path Dinâmico
    // =========================================================================

    /// Constrói um `NamModelData` WaveNet dinâmico com `gated` configurável por array.
    ///
    /// Layout de pesos para CH=4, HEAD=2, K=3, dils=[1,2] (2 layers por array):
    ///
    /// **Array1** (IN=1, COND=1, CH=4, gated=true, HEAD=2, no head_bias):
    /// - rechannel: 1×4 = 4
    /// - layer×2: conv1d(4→8, K=3) = 8×3×4 + 8 = 104; input_mixin(1×4) = 4; o2o(4×4 + 4) = 20 → 128×2 = 256
    /// - head_rechannel: 4×2 = 8
    /// - subtotal: 268
    ///
    /// **Array2** (IN=4, COND=1, CH=2, gated=false, HEAD=1, with head_bias):
    /// - rechannel: 4×2 = 8
    /// - layer×2: conv1d(2→2, K=3) = 2×3×2 + 2 = 14; input_mixin(1×2) = 2; o2o(2×2 + 2) = 6 → 22×2 = 44
    /// - head_rechannel: 2×1 + 1 = 3
    /// - subtotal: 55
    ///
    /// head_scale: 1
    /// **Total: 268 + 55 + 1 = 324**
    fn make_wavenet_gated_data(gated_0: bool, gated_1: bool, total_weights: usize) -> NamModelData {
        let dils = vec![1usize, 2];
        NamModelData {
            version: Some("0.5.4".to_string()),
            architecture: "WaveNet".to_string(),
            config: NamConfig {
                layers: vec![
                    NamLayerConfig {
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: Some(2),
                        channels: Some(4),
                        kernel_size: Some(3),
                        dilations: Some(dils.clone()),
                        activation: Some("Tanh".to_string()),
                        gated: Some(gated_0),
                        head_bias: Some(false),
                    },
                    NamLayerConfig {
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: Some(2),
                        channels: Some(4),
                        kernel_size: Some(3),
                        dilations: Some(dils),
                        activation: Some("Tanh".to_string()),
                        gated: Some(gated_1),
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

    /// Verifica que o dispatcher constrói corretamente um WaveNet dinâmico com `gated=true` no array1.
    ///
    /// Topologia: CH=4, K=3, HEAD=2, dils=[1,2] (channels≠Standard/Lite/Feather/Nano → path dinâmico).
    ///
    /// **Contagem de pesos (array1 gated=true, array2 gated=false):**
    /// - Array1: rechannel(4) + 2×[conv(8×3×4+8=104) + mixin(4) + o2o(20)] + head(8) = 4 + 256 + 8 = 268
    /// - Array2: rechannel(8) + 2×[conv(2×3×2+2=14) + mixin(2) + o2o(6)] + head(3) = 8 + 44 + 3 = 55
    /// - head_scale: 1 → **Total: 324**
    #[test]
    fn test_build_wavenet_dynamic_gated() {
        // gated=true no array1, gated=false no array2 → fallback dinâmico (CH=4 não é Standard)
        let data = make_wavenet_gated_data(true, false, 324);
        let result = build_wavenet_dynamic(&data);
        assert!(
            result.is_ok(),
            "WaveNet dinâmico gated=true deveria construir com sucesso: {:?}",
            result.err()
        );
    }

    /// Verifica que gated=false no array1 e array2 (path dinâmico) produz exatamente
    /// o mesmo número de pesos que a contagem não-gated.
    ///
    /// **Contagem de pesos (ambos gated=false):**
    /// - Array1: rechannel(4) + 2×[conv(4×3×4+4=52) + mixin(4) + o2o(20)] + head(8) = 4 + 152 + 8 = 164
    /// - Array2: rechannel(8) + 2×[conv(2×3×2+2=14) + mixin(2) + o2o(6)] + head(3) = 8 + 44 + 3 = 55
    /// - head_scale: 1 → **Total: 220**
    #[test]
    fn test_build_wavenet_dynamic_non_gated() {
        let data = make_wavenet_gated_data(false, false, 220);
        let result = build_wavenet_dynamic(&data);
        assert!(
            result.is_ok(),
            "WaveNet dinâmico gated=false deveria construir com sucesso: {:?}",
            result.err()
        );
    }
}
