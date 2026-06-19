// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::layout;
use crate::loader::nam_json::{FreeWavenetGeometry, NamModelData};
use crate::math::common::AlignedVec;
use crate::models::wavenet::{
    DenseLayerDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn, WaveNetLayerDyn,
    WaveNetLayerState, WaveNetModelDyn,
};
use log::info;

// =============================================================================
// Dynamic WaveNet array constructor
// =============================================================================

/// Builds a `WaveNetLayerArrayDyn` with runtime dimensions.
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
#[allow(clippy::too_many_arguments)]
fn build_wavenet_array_dyn(
    cursor: &mut WeightCursor<'_>,
    in_ch: usize,
    cond: usize,
    ch: usize,
    k: usize,
    head: usize,
    dilations: &[usize],
    has_head_bias: bool,
    alloc_num: &mut usize,
) -> anyhow::Result<WaveNetLayerArrayDyn> {
    let rechannel = layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, in_ch, ch, false)?;

    let mut layers = Vec::with_capacity(dilations.len());
    let mut states = Vec::with_capacity(dilations.len());

    for &dilation in dilations {
        let conv1d = layout::read_conv1d_weights_typed::<crate::models::wavenet::Conv1dDyn>(
            cursor, ch, ch, k, dilation, true,
        )?;

        let input_mixin =
            layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, cond, ch, false)?;

        let one_by_one = layout::read_dense_weights_typed::<DenseLayerDyn>(cursor, ch, ch, true)?;

        layers.push(WaveNetLayerDyn::new(ch, conv1d, input_mixin, one_by_one));

        let rf = (k - 1) * dilation;
        states.push(WaveNetLayerState::new(ch, rf, *alloc_num)?);
        *alloc_num += 1;
    }

    let head_rechannel =
        layout::read_dense_head_weights_typed::<DenseLayerDyn>(cursor, ch, head, has_head_bias)?;

    let receptive_field_size: usize = dilations.iter().map(|&d| (k - 1) * d).sum();

    let block_size = ch;
    let block_buffer = AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0);
    let num_layers = layers.len();

    Ok(WaveNetLayerArrayDyn {
        in_ch,
        cond,
        ch,
        k,
        head,
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0),
        head_accum: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0),
        head_outputs: AlignedVec::new(head * WAVENET_MAX_NUM_FRAMES, 0.0),
        receptive_field_size,
        block_size,
        block_buffer,
        effective_layers: num_layers,
    })
}

// =============================================================================
// Dynamic WaveNet model entry point
// =============================================================================

/// Builds a `WaveNetModelDyn` from parsed model data using runtime dimensions.
///
/// Consumes the geometry description from [`FreeWavenetGeometry`] and reads
/// weights sequentially in the same layout as the const-generic path.
///
/// Weight layout (C++ WaveNet.h `SetWeights`):
/// ```text
/// [array1.rechannel] [array1.layers...] [array1.head_rechannel]
/// [array2.rechannel] [array2.layers...] [array2.head_rechannel]
/// [head_scale]
/// ```
pub(crate) fn build_wavenet_dynamic(
    data: &NamModelData,
    geom: &FreeWavenetGeometry,
) -> anyhow::Result<WaveNetModelDyn> {
    super::validate_layer_activations(data)?;

    if geom.num_arrays != 2 {
        anyhow::bail!(
            "WaveNet A1 dynamic engine requires exactly 2 layer arrays (found {}). \
             Models with {} arrays are not yet supported in NAM-rs.",
            geom.num_arrays,
            geom.num_arrays
        );
    }

    let ch = geom.channels;
    let k = geom.kernel_size;
    let head = geom.head_size;
    let cond = geom.condition_size;

    let mut cursor = WeightCursor::new(&data.weights, data.weights_layout);

    debug_assert!(geom.num_arrays >= 2);
    debug_assert!(geom.dilations.len() >= 2);

    let dils_0 = &geom.dilations[0];
    let dils_1 = &geom.dilations[1];

    let mut alloc_num = 0usize;

    // Array1: IN=1, COND=condition_size, CH channels, HEAD outputs, no head bias
    let array1 = build_wavenet_array_dyn(
        &mut cursor,
        1,    // in_ch
        cond, // cond
        ch,   // ch
        k,    // k
        head, // head
        dils_0,
        false, // has_head_bias
        &mut alloc_num,
    )?;

    // Array2: IN=ch, COND=condition_size, CH=head, HEAD=1, with head bias
    let array2 = build_wavenet_array_dyn(
        &mut cursor,
        ch,   // in_ch
        cond, // cond
        head, // ch
        k,    // k
        1,    // head
        dils_1,
        true, // has_head_bias
        &mut alloc_num,
    )?;

    let head_scale = cursor.read_f32()?;

    cursor.verify_exhausted()?;

    let rf = array1.receptive_field_size.max(array2.receptive_field_size);

    let model = WaveNetModelDyn {
        ch,
        k,
        head,
        array1,
        array2,
        head_scale,
        receptive_field_size: rf,
    };

    info!(
        "[Dispatcher] WaveNet Dynamic built — CH={}, K={}, HEAD={}, arrays={}, \
         dilations0={:?}, dilations1={:?}, head_scale={:.6}, weights={}",
        ch,
        k,
        head,
        geom.num_arrays,
        dils_0,
        dils_1,
        head_scale,
        data.weights.len()
    );

    Ok(model)
}
