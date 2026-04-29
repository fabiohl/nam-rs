// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::WeightCursor;
use crate::loader::nam_json::{NamModelData, NamWavenetTopology, get_wavenet_topology};
use crate::models::DynamicModel;
use crate::models::wavenet::{
    Conv1d, DenseLayer, WaveNetLayer, WaveNetLayerArray, WaveNetLayerState, WaveNetModel,
};
use crate::models::wavenet_dyn::{
    Conv1dDyn, DenseLayerDyn, WaveNetDynModel, WaveNetLayerArrayDyn, WaveNetLayerDyn,
};
use anyhow::{Context, bail};
use log::info;

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
pub(crate) fn build_wavenet(data: &NamModelData) -> anyhow::Result<Box<DynamicModel>> {
    let topo_opt = get_wavenet_topology(data);

    match topo_opt {
        Some(NamWavenetTopology::Standard) => {
            let model = build_wavenet_typed::<16, 3, 8>(data, NamWavenetTopology::Standard)?;
            Ok(Box::new(DynamicModel::WavenetStandard(Box::new(model))))
        }
        Some(NamWavenetTopology::Lite) => {
            let model = build_wavenet_typed::<12, 3, 6>(data, NamWavenetTopology::Lite)?;
            Ok(Box::new(DynamicModel::WavenetLite(Box::new(model))))
        }
        Some(NamWavenetTopology::Feather) => {
            let model = build_wavenet_typed::<8, 3, 4>(data, NamWavenetTopology::Feather)?;
            Ok(Box::new(DynamicModel::WavenetFeather(Box::new(model))))
        }
        Some(NamWavenetTopology::Nano) => {
            let model = build_wavenet_typed::<4, 3, 2>(data, NamWavenetTopology::Nano)?;
            Ok(Box::new(DynamicModel::WavenetNano(Box::new(model))))
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
pub(crate) fn build_wavenet_typed<const CH: usize, const K: usize, const HEAD: usize>(
    data: &NamModelData,
    topo: NamWavenetTopology,
) -> anyhow::Result<WaveNetModel<CH, K, HEAD>> {
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

    // O WaveNet é organizado em dois grandes blocos chamados "Arrays".
    // Imagine cada Array como um rack de efeitos complexo.

    // Array 1: O primeiro estágio de processamento.
    let array1 = build_wavenet_array::<1, 1, CH, K, HEAD>(
        &mut cursor,
        dils_0,
        false, // HasHeadBias da array1 (C++: false)
        &mut alloc_num,
    )?;

    // Array 2: O segundo estágio, que recebe o que o primeiro processou.
    let array2 = build_wavenet_array::<CH, 1, HEAD, K, 1>(
        &mut cursor,
        dils_1,
        true, // HasHeadBias da array2 (C++: true)
        &mut alloc_num,
    )?;

    // O 'head_scale' é o botão de volume final do modelo inteiro.
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

    Ok(model)
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
pub(crate) fn build_wavenet_array<
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
    // 1. Rechannel: Aqui transformamos o sinal de entrada (1 fio) em vários
    // canais internos (ex: 16 fios) para que a rede tenha mais "espaço" para pensar.
    let rechannel = read_dense_layer::<IN, CH>(cursor, false)?;

    // 2. Camadas (Layers): Criamos uma camada para cada "dilatação".
    // Uma dilatação é como um eco: a rede olha para o que aconteceu há 1, 2, 4, 8... amostras atrás.
    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        // conv1d: É o filtro matemático principal que processa o áudio e seus "ecos".
        let conv1d = read_conv1d_weights::<CH, CH, K>(cursor, dilation, true)?;

        // input_mixin: Adiciona informações externas ao processamento (como ajustes do usuário).
        let input_mixin = read_dense_layer::<COND, CH>(cursor, false)?;

        // one_by_one: Um ajuste final de "mistura" de canais para cada momento do som.
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

    // 3. Head Rechannel: Transforma os vários canais internos de volta no
    // formato de saída (geralmente reduzindo para preparar para o próximo estágio).
    let head_rechannel = read_dense_layer::<CH, HEAD>(cursor, has_head_bias)?;

    // Campo receptivo: É o tempo total de "memória" que este bloco tem (quantas amostras do passado ele olha).
    let receptive_field_size: usize = dilations.iter().map(|&d| (K - 1) * d).sum();

    Ok(WaveNetLayerArray {
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: vec![0.0; CH * crate::models::wavenet::WAVENET_MAX_NUM_FRAMES],
        head_accum: vec![0.0; CH * crate::models::wavenet::WAVENET_MAX_NUM_FRAMES],
        head_outputs: vec![0.0; HEAD * crate::models::wavenet::WAVENET_MAX_NUM_FRAMES],
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

    // No WaveNet dinâmico, criamos os Arrays sem saber os tamanhos fixos de antemão.
    // É como montar um rack de efeitos onde os módulos podem ter qualquer tamanho.

    // Primeiro bloco de processamento.
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

    // Segundo bloco, que recebe o sinal já "expandido" pelo primeiro.
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

    // Volume final do modelo.
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

    Ok(Box::new(DynamicModel::WavenetDyn(Box::new(model))))
}
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

    // Aqui acontece uma reorganização importante (Transposição):
    // Os pesos do modelo original (treinado em Python/C++) estão em uma ordem
    // que não é a melhor para o Rust processar áudio em tempo real.
    // Nós "embaralhamos" os dados aqui para que o processador possa ler os
    // números em sequência perfeita durante o processamento, o que é muito mais rápido.
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
pub(crate) fn build_wavenet_array_dyn(
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
    // Se a rede for "gated" (com comportas), ela precisa de duas vezes mais espaço
    // de saída para calcular as funções Tanh e Sigmoid simultaneamente.
    let conv_out_ch = if gated { 2 * ch } else { ch };

    let rechannel = read_dense_layer_dyn(cursor, in_size, ch, false)?;

    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        // Criamos cada camada com sua respectiva "distância de memória" (dilatação).
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

        // O 'rf' (Receptive Field) é quanto tempo de áudio esta camada consegue "lembrar".
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
        array_outputs: vec![0.0; ch * crate::models::wavenet::WAVENET_MAX_NUM_FRAMES],
        head_accum: vec![0.0; ch * crate::models::wavenet::WAVENET_MAX_NUM_FRAMES],
        head_outputs: vec![0.0; head * crate::models::wavenet::WAVENET_MAX_NUM_FRAMES],
        block_buffer: vec![0.0; block_size * crate::models::wavenet::WAVENET_MAX_NUM_FRAMES],
        block_size,
        receptive_field_size,
        ch,
        head,
    })
}
