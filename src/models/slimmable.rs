// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `SlimmableModel` trait — models that can dynamically scale quality/complexity
//! at runtime without reallocation.
//!
//! This is the official NAM architecture for runtime quality scaling.
//!
//! ## Channel slicing infrastructure
//!
//! The extraction/slicing functions below allow the instantiation engine to
//! produce a new `WaveNetModelDyn` by slicing the first N channels of every
//! weight matrix. This enables the async thread to create a lightweight copy
//! of the full model that can be atomically swapped via the SPSC GC pipeline
//! (`gc_cascade` → `drain_gc_channels`), keeping the DSP hot-path
//! zero-alloc and lock-free.
//!
//! Weights are stored in SIMD-friendly interleaved layouts:
//! - `Conv1dDyn`: `[block][kernel][in_ch][4]` — 4-wide lane-interleaved
//! - `DenseLayerDyn`: column-major `weights[in_c * out_ch + out_c]`

use crate::math::common::AlignedVec;
use crate::models::wavenet::{
    Conv1dDyn, DenseLayerDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn, WaveNetLayerDyn,
    WaveNetLayerState, WaveNetModelDyn,
};

/// Trait for models that can dynamically scale quality/complexity at runtime
/// without reallocation.
///
/// The value `0.0` represents the minimum quality/cheapest model,
/// and `1.0` represents the maximum quality/full model.
///
/// Implementors:
/// - `ContainerModel`: selects between pre-built submodels by threshold.
/// - `SlimmableWavenet` (planned): channel-slices a single network.
pub trait SlimmableModel {
    /// Sets the slimmable quality/size level.
    ///
    /// `val` is in `[0.0, 1.0]` where `0.0` = minimum quality and `1.0` = full quality.
    fn set_slimmable_size(&mut self, val: f32);
}

// =============================================================================
// Weight extraction / slicing infrastructure
// =============================================================================

/// Slices a `Conv1dDyn` to a reduced channel configuration.
///
/// The convolution weight layout is `[block][kernel][in_ch][4]`
/// (4-wide lane-interleaved). This function extracts only the first
/// `new_in_ch` input channels and first `new_out_ch` output channels,
/// keeping the same `kernel` size and `dilation`.
///
/// # Panics
/// Panics if `new_in_ch > conv.in_ch` or `new_out_ch > conv.out_ch`.
pub fn slice_conv1d(conv: &Conv1dDyn, new_in_ch: usize, new_out_ch: usize) -> Conv1dDyn {
    assert!(
        new_in_ch <= conv.in_ch,
        "slice_conv1d: new_in_ch ({}) > in_ch ({})",
        new_in_ch,
        conv.in_ch
    );
    assert!(
        new_out_ch <= conv.out_ch,
        "slice_conv1d: new_out_ch ({}) > out_ch ({})",
        new_out_ch,
        conv.out_ch
    );

    let new_num_blocks = new_out_ch.div_ceil(4);
    let kernel = conv.kernel;
    let new_weights_len = new_num_blocks * 4 * new_in_ch * kernel;
    let mut new_weights = AlignedVec::new(new_weights_len, 0.0f32);

    for b in 0..new_num_blocks {
        for k in 0..kernel {
            for in_c in 0..new_in_ch {
                let src_idx = ((b * kernel + k) * conv.in_ch + in_c) * 4;
                let dst_idx = ((b * kernel + k) * new_in_ch + in_c) * 4;
                let src = &conv.weights[src_idx..src_idx + 4];
                let dst = &mut new_weights[dst_idx..dst_idx + 4];
                dst.copy_from_slice(src);
            }
        }
    }

    let mut new_bias = AlignedVec::new(new_out_ch, 0.0f32);
    new_bias.copy_from_slice(&conv.bias[..new_out_ch]);

    Conv1dDyn {
        weights: new_weights,
        bias: new_bias,
        do_bias: conv.do_bias,
        dilation: conv.dilation,
        in_ch: new_in_ch,
        out_ch: new_out_ch,
        num_blocks: new_num_blocks,
        kernel,
        prefetch_fn: conv.prefetch_fn,
    }
}

/// Slices a `DenseLayerDyn` to a reduced channel configuration.
///
/// Dense weights are stored in column-major layout:
/// `weights[in_c * out_ch + out_c]`.
/// This extracts the first `new_in_ch` input channels and first `new_out_ch`
/// output channels.
///
/// # Panics
/// Panics if `new_in_ch > dense.in_ch` or `new_out_ch > dense.out_ch`.
pub fn slice_dense(dense: &DenseLayerDyn, new_in_ch: usize, new_out_ch: usize) -> DenseLayerDyn {
    assert!(
        new_in_ch <= dense.in_ch,
        "slice_dense: new_in_ch ({}) > in_ch ({})",
        new_in_ch,
        dense.in_ch
    );
    assert!(
        new_out_ch <= dense.out_ch,
        "slice_dense: new_out_ch ({}) > out_ch ({})",
        new_out_ch,
        dense.out_ch
    );

    let mut new_weights = AlignedVec::new(new_in_ch * new_out_ch, 0.0f32);

    for in_c in 0..new_in_ch {
        let src_start = in_c * dense.out_ch;
        let dst_start = in_c * new_out_ch;
        new_weights[dst_start..dst_start + new_out_ch]
            .copy_from_slice(&dense.weights[src_start..src_start + new_out_ch]);
    }

    let mut new_bias = AlignedVec::new(new_out_ch, 0.0f32);
    new_bias.copy_from_slice(&dense.bias[..new_out_ch]);

    DenseLayerDyn {
        in_ch: new_in_ch,
        out_ch: new_out_ch,
        weights: new_weights,
        bias: new_bias,
        do_bias: dense.do_bias,
    }
}

/// Creates a new `WaveNetLayerDyn` with reduced internal channel count.
///
/// Slices all three internal tensors:
/// - `conv1d`: `(ch, ch)` → `(new_ch, new_ch)`
/// - `input_mixin`: `(cond, ch)` → `(cond, new_ch)`
/// - `one_by_one`: `(ch, ch)` → `(new_ch, new_ch)`
pub fn slice_wavenet_layer(layer: &WaveNetLayerDyn, new_ch: usize) -> WaveNetLayerDyn {
    let conv1d = slice_conv1d(&layer.conv1d, new_ch, new_ch);
    let input_mixin = slice_dense(&layer.input_mixin, layer.input_mixin.in_ch, new_ch);
    let one_by_one = slice_dense(&layer.one_by_one, new_ch, new_ch);
    WaveNetLayerDyn::new(new_ch, conv1d, input_mixin, one_by_one)
}

/// Creates a new `WaveNetLayerArrayDyn` with reduced internal channel count.
///
/// Rebuilds all sub-components (rechannel, layers, states, head_rechannel)
/// with the new channel dimensions. States are freshly allocated via
/// `WaveNetLayerState::new` — prewarm will stabilize them later.
///
/// `new_in_ch`: input channels for this array (1 for the first array,
///              `new_ch` for subsequent arrays).
/// `alloc_num`: allocation counter for state jitter (pass a `&mut usize`
///              that increments across all arrays in the model).
pub fn slice_wavenet_array(
    array: &WaveNetLayerArrayDyn,
    new_in_ch: usize,
    new_ch: usize,
    alloc_num: &mut usize,
) -> std::io::Result<WaveNetLayerArrayDyn> {
    let rechannel = slice_dense(&array.rechannel, new_in_ch, new_ch);

    let mut layers = Vec::with_capacity(array.layers.len());
    let mut states = Vec::with_capacity(array.layers.len());

    for layer in &array.layers {
        layers.push(slice_wavenet_layer(layer, new_ch));
        let rf = (layer.conv1d.kernel - 1) * layer.conv1d.dilation;
        states.push(WaveNetLayerState::new(new_ch, rf, *alloc_num)?);
        *alloc_num += 1;
    }

    let head_rechannel = slice_dense(&array.head_rechannel, new_ch, array.head);

    let receptive_field_size: usize = array
        .layers
        .iter()
        .map(|l| (l.conv1d.kernel - 1) * l.conv1d.dilation)
        .sum();

    let block_size = new_ch;
    let num_layers = layers.len();

    Ok(WaveNetLayerArrayDyn {
        in_ch: new_in_ch,
        cond: array.cond,
        ch: new_ch,
        k: array.k,
        head: array.head,
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0),
        head_accum: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0),
        head_outputs: AlignedVec::new(array.head * WAVENET_MAX_NUM_FRAMES, 0.0),
        receptive_field_size,
        block_size,
        block_buffer: AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0),
        effective_layers: num_layers,
    })
}

/// Creates a new `WaveNetModelDyn` with all internal channels reduced to
/// `new_ch`. This is the primary entry point for the SPSC GC swap pipeline.
///
/// Each layer array's internal `ch` is reduced to `new_ch`. The head
/// projection and condition dimensions remain unchanged.
///
/// ## Limitations
///
/// - **`condition_dsp`**: Set to `None` in the sliced model. The `condition_dsp`
///   sub-model (`Box<StaticModel>`) cannot be cloned generically. Future work
///   (Task 2.1.2) may address this by rebuilding the condition DSP from the
///   original JSON or cloning it.
/// - **`post_stack_head`**: Cloned as-is (not affected by channel slicing).
///
/// # Panics
/// Panics if `new_ch` is zero or exceeds the original channel count,
/// or if arrays have non-uniform channel counts.
pub fn slice_wavenet_model(
    model: &WaveNetModelDyn,
    new_ch: usize,
) -> std::io::Result<WaveNetModelDyn> {
    assert!(new_ch > 0, "slice_wavenet_model: new_ch must be > 0");
    assert!(
        new_ch <= model.ch,
        "slice_wavenet_model: new_ch ({}) > model.ch ({})",
        new_ch,
        model.ch
    );

    let min_array_ch = model.arrays.iter().map(|a| a.ch).min().unwrap_or(model.ch);
    assert!(
        new_ch <= min_array_ch,
        "slice_wavenet_model: new_ch ({}) exceeds minimum array channel count ({})",
        new_ch,
        min_array_ch
    );

    let mut alloc_num = 0usize;
    let mut arrays = Vec::with_capacity(model.arrays.len());

    for (i, array) in model.arrays.iter().enumerate() {
        let in_ch = if i == 0 { 1 } else { new_ch };
        arrays.push(slice_wavenet_array(array, in_ch, new_ch, &mut alloc_num)?);
    }

    let cond = model.arrays[0].cond;
    let cond_dsp_output_size = cond * WAVENET_MAX_NUM_FRAMES;

    let head_out_ch = model
        .post_stack_head
        .as_ref()
        .map(|h| h.out_channels())
        .unwrap_or(1);
    let head_output_scratch = AlignedVec::new(head_out_ch * WAVENET_MAX_NUM_FRAMES, 0.0);

    let mut rf = arrays
        .iter()
        .map(|a| a.receptive_field_size)
        .max()
        .unwrap_or(0);
    if let Some(ref head_proc) = model.post_stack_head {
        rf += head_proc.receptive_field() - 1;
    }

    Ok(WaveNetModelDyn {
        ch: new_ch,
        k: model.k,
        head: model.head,
        arrays,
        head_scale: model.head_scale,
        receptive_field_size: rf,
        condition_dsp: None,
        condition_dsp_output: AlignedVec::new(cond_dsp_output_size, 0.0),
        post_stack_head: model.post_stack_head.clone(),
        head_output_scratch,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::common::AlignedVec;
    use crate::models::wavenet::{
        Conv1dDyn, DenseLayerDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn, WaveNetLayerDyn,
        WaveNetLayerState, WaveNetModelDyn,
    };

    const TEST_KERNEL: usize = 3;
    const TEST_DILATION: usize = 2;
    const CH_FULL: usize = 8;
    const CH_SLIM: usize = 4;

    fn make_conv1d(in_ch: usize, out_ch: usize) -> Conv1dDyn {
        let kernel = TEST_KERNEL;
        let num_blocks = out_ch.div_ceil(4);
        let weights_len = num_blocks * 4 * in_ch * kernel;
        let mut weights = AlignedVec::new(weights_len, 0.0f32);
        for i in 0..weights_len {
            weights[i] = (i + 1) as f32;
        }
        let mut bias = AlignedVec::new(out_ch, 0.0f32);
        for i in 0..out_ch {
            bias[i] = (i + 100) as f32;
        }
        Conv1dDyn {
            weights,
            bias,
            do_bias: true,
            dilation: TEST_DILATION,
            in_ch,
            out_ch,
            num_blocks,
            kernel,
            prefetch_fn: crate::math::common::prefetch_strategy_simple
                as unsafe fn(*const f32, usize, usize, usize, usize),
        }
    }

    fn make_dense(in_ch: usize, out_ch: usize) -> DenseLayerDyn {
        let mut weights = AlignedVec::new(in_ch * out_ch, 0.0f32);
        for in_c in 0..in_ch {
            for out_c in 0..out_ch {
                weights[in_c * out_ch + out_c] = ((in_c * out_ch + out_c) as f32) + 1.0;
            }
        }
        let mut bias = AlignedVec::new(out_ch, 0.0f32);
        for i in 0..out_ch {
            bias[i] = (i + 200) as f32;
        }
        DenseLayerDyn {
            in_ch,
            out_ch,
            weights,
            bias,
            do_bias: true,
        }
    }

    fn make_wavenet_layer(ch: usize) -> WaveNetLayerDyn {
        let conv1d = make_conv1d(ch, ch);
        let input_mixin = make_dense(1, ch);
        let one_by_one = make_dense(ch, ch);
        WaveNetLayerDyn::new(ch, conv1d, input_mixin, one_by_one)
    }

    fn make_wavenet_array(
        in_ch: usize,
        ch: usize,
        head: usize,
        dilations: &[usize],
    ) -> WaveNetLayerArrayDyn {
        let rechannel = make_dense(in_ch, ch);
        let num_layers = dilations.len();
        let mut layers = Vec::with_capacity(num_layers);
        let mut states = Vec::with_capacity(num_layers);
        for (alloc_num, &d) in dilations.iter().enumerate() {
            let mut layer = make_wavenet_layer(ch);
            layer.conv1d.dilation = d;
            let rf = (TEST_KERNEL - 1) * d;
            states.push(WaveNetLayerState::new(ch, rf, alloc_num).unwrap());
            layers.push(layer);
        }
        let head_rechannel = make_dense(ch, head);
        let receptive_field_size: usize = dilations.iter().map(|&d| (TEST_KERNEL - 1) * d).sum();
        let block_size = ch;
        WaveNetLayerArrayDyn {
            in_ch,
            cond: 1,
            ch,
            k: TEST_KERNEL,
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
            block_buffer: AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0),
            effective_layers: num_layers,
        }
    }

    fn make_full_model(ch: usize, head: usize) -> WaveNetModelDyn {
        let dilations = [1, 2, 4];
        let array1 = make_wavenet_array(1, ch, head, &dilations);
        let array2 = make_wavenet_array(ch, head, 1, &dilations);
        let rf = array1.receptive_field_size.max(array2.receptive_field_size);
        WaveNetModelDyn {
            ch,
            k: TEST_KERNEL,
            head,
            arrays: vec![array1, array2],
            head_scale: 0.02,
            receptive_field_size: rf,
            condition_dsp: None,
            condition_dsp_output: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
            post_stack_head: None,
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0),
        }
    }

    // =====================================================================
    // slice_conv1d tests
    // =====================================================================

    #[test]
    fn test_slice_conv1d_dims() {
        let conv = make_conv1d(CH_FULL, CH_FULL);
        let sliced = slice_conv1d(&conv, CH_SLIM, CH_SLIM);
        assert_eq!(sliced.in_ch, CH_SLIM);
        assert_eq!(sliced.out_ch, CH_SLIM);
        assert_eq!(sliced.kernel, TEST_KERNEL);
        assert_eq!(sliced.dilation, TEST_DILATION);
        assert!(sliced.do_bias);
        assert_eq!(sliced.num_blocks, CH_SLIM.div_ceil(4));
        assert_eq!(
            sliced.weights.len(),
            sliced.num_blocks * 4 * CH_SLIM * TEST_KERNEL
        );
        assert_eq!(sliced.bias.len(), CH_SLIM);
    }

    #[test]
    fn test_slice_conv1d_weights_match() {
        let conv = make_conv1d(CH_FULL, CH_FULL);
        let sliced = slice_conv1d(&conv, CH_SLIM, CH_SLIM);

        for b in 0..sliced.num_blocks {
            for k in 0..TEST_KERNEL {
                for in_c in 0..CH_SLIM {
                    let src_idx = ((b * TEST_KERNEL + k) * CH_FULL + in_c) * 4;
                    let dst_idx = ((b * TEST_KERNEL + k) * CH_SLIM + in_c) * 4;
                    assert_eq!(
                        &sliced.weights[dst_idx..dst_idx + 4],
                        &conv.weights[src_idx..src_idx + 4],
                        "mismatch at block={} k={} in_c={}",
                        b,
                        k,
                        in_c
                    );
                }
            }
        }
    }

    #[test]
    fn test_slice_conv1d_bias_match() {
        let conv = make_conv1d(CH_FULL, CH_FULL);
        let sliced = slice_conv1d(&conv, CH_SLIM, CH_SLIM);
        assert_eq!(&sliced.bias[..CH_SLIM], &conv.bias[..CH_SLIM]);
    }

    #[test]
    #[should_panic(expected = "slice_conv1d: new_in_ch")]
    fn test_slice_conv1d_bigger_in_ch_panics() {
        let conv = make_conv1d(CH_FULL, CH_FULL);
        slice_conv1d(&conv, CH_FULL + 1, CH_FULL);
    }

    #[test]
    #[should_panic(expected = "slice_conv1d: new_out_ch")]
    fn test_slice_conv1d_bigger_out_ch_panics() {
        let conv = make_conv1d(CH_FULL, CH_FULL);
        slice_conv1d(&conv, CH_FULL, CH_FULL + 1);
    }

    // =====================================================================
    // slice_dense tests
    // =====================================================================

    #[test]
    fn test_slice_dense_dims() {
        let dense = make_dense(CH_FULL, CH_FULL);
        let sliced = slice_dense(&dense, CH_SLIM, CH_SLIM);
        assert_eq!(sliced.in_ch, CH_SLIM);
        assert_eq!(sliced.out_ch, CH_SLIM);
        assert_eq!(sliced.do_bias, dense.do_bias);
        assert_eq!(sliced.weights.len(), CH_SLIM * CH_SLIM);
        assert_eq!(sliced.bias.len(), CH_SLIM);
    }

    #[test]
    fn test_slice_dense_weights_match() {
        let dense = make_dense(CH_FULL, CH_FULL);
        let sliced = slice_dense(&dense, CH_SLIM, CH_SLIM);

        for in_c in 0..CH_SLIM {
            for out_c in 0..CH_SLIM {
                let src_idx = in_c * CH_FULL + out_c;
                let dst_idx = in_c * CH_SLIM + out_c;
                assert_eq!(
                    sliced.weights[dst_idx], dense.weights[src_idx],
                    "mismatch at in_c={} out_c={}",
                    in_c, out_c
                );
            }
        }
    }

    #[test]
    fn test_slice_dense_bias_match() {
        let dense = make_dense(CH_FULL, CH_FULL);
        let sliced = slice_dense(&dense, CH_SLIM, CH_SLIM);
        assert_eq!(&sliced.bias[..CH_SLIM], &dense.bias[..CH_SLIM]);
    }

    #[test]
    fn test_slice_dense_asymmetric() {
        let dense = make_dense(8, 12);
        let sliced = slice_dense(&dense, 4, 6);
        assert_eq!(sliced.in_ch, 4);
        assert_eq!(sliced.out_ch, 6);
        assert_eq!(sliced.weights.len(), 24);
        assert_eq!(sliced.bias.len(), 6);
        for in_c in 0..4usize {
            for out_c in 0..6usize {
                assert_eq!(
                    sliced.weights[in_c * 6 + out_c],
                    dense.weights[in_c * 12 + out_c]
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "slice_dense: new_in_ch")]
    fn test_slice_dense_bigger_in_ch_panics() {
        let dense = make_dense(CH_FULL, CH_FULL);
        slice_dense(&dense, CH_FULL + 1, CH_FULL);
    }

    // =====================================================================
    // slice_wavenet_layer tests
    // =====================================================================

    #[test]
    fn test_slice_wavenet_layer_dims() {
        let layer = make_wavenet_layer(CH_FULL);
        let sliced = slice_wavenet_layer(&layer, CH_SLIM);

        assert_eq!(sliced.conv1d.in_ch, CH_SLIM);
        assert_eq!(sliced.conv1d.out_ch, CH_SLIM);
        assert_eq!(sliced.input_mixin.in_ch, 1);
        assert_eq!(sliced.input_mixin.out_ch, CH_SLIM);
        assert_eq!(sliced.one_by_one.in_ch, CH_SLIM);
        assert_eq!(sliced.one_by_one.out_ch, CH_SLIM);
        assert_eq!(sliced.scratch_mixin.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
        assert_eq!(sliced.scratch_conv.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
    }

    #[test]
    fn test_slice_wavenet_layer_weights_preserved() {
        let layer = make_wavenet_layer(CH_FULL);
        let sliced = slice_wavenet_layer(&layer, CH_SLIM);

        let conv_sliced = slice_conv1d(&layer.conv1d, CH_SLIM, CH_SLIM);
        let mixin_sliced = slice_dense(&layer.input_mixin, 1, CH_SLIM);
        let obo_sliced = slice_dense(&layer.one_by_one, CH_SLIM, CH_SLIM);

        assert_eq!(&*sliced.conv1d.weights, &*conv_sliced.weights);
        assert_eq!(&*sliced.input_mixin.weights, &*mixin_sliced.weights);
        assert_eq!(&*sliced.one_by_one.weights, &*obo_sliced.weights);
    }

    // =====================================================================
    // slice_wavenet_array tests
    // =====================================================================

    #[test]
    fn test_slice_wavenet_array_dims() {
        let dilations = [1, 2, 4];
        let array = make_wavenet_array(1, CH_FULL, 4, &dilations);
        let mut alloc_num = 0;
        let sliced = slice_wavenet_array(&array, 1, CH_SLIM, &mut alloc_num).unwrap();

        assert_eq!(sliced.in_ch, 1);
        assert_eq!(sliced.ch, CH_SLIM);
        assert_eq!(sliced.head, 4);
        assert_eq!(sliced.cond, 1);
        assert_eq!(sliced.layers.len(), 3);
        assert_eq!(sliced.states.len(), 3);
        assert_eq!(sliced.rechannel.in_ch, 1);
        assert_eq!(sliced.rechannel.out_ch, CH_SLIM);
        assert_eq!(sliced.head_rechannel.in_ch, CH_SLIM);
        assert_eq!(sliced.head_rechannel.out_ch, 4);
        assert_eq!(sliced.array_outputs.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
        assert_eq!(sliced.head_accum.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
        assert_eq!(sliced.block_size, CH_SLIM);
        assert_eq!(sliced.effective_layers, 3);
    }

    #[test]
    fn test_slice_wavenet_array_preserves_weights() {
        let dilations = [1, 2, 4];
        let array = make_wavenet_array(1, CH_FULL, 4, &dilations);
        let mut alloc_num = 0;
        let sliced = slice_wavenet_array(&array, 1, CH_SLIM, &mut alloc_num).unwrap();

        let rec_expected = slice_dense(&array.rechannel, 1, CH_SLIM);
        assert_eq!(&*sliced.rechannel.weights, &*rec_expected.weights);

        for (i, (orig, slic)) in array.layers.iter().zip(sliced.layers.iter()).enumerate() {
            let conv_expected = slice_conv1d(&orig.conv1d, CH_SLIM, CH_SLIM);
            assert_eq!(
                &*slic.conv1d.weights, &*conv_expected.weights,
                "conv1d mismatch at layer {}",
                i
            );
        }

        let head_expected = slice_dense(&array.head_rechannel, CH_SLIM, 4);
        assert_eq!(&*sliced.head_rechannel.weights, &*head_expected.weights);
    }

    // =====================================================================
    // slice_wavenet_model / slice_channels tests
    // =====================================================================

    #[test]
    fn test_slice_wavenet_model_dims() {
        let model = make_full_model(CH_FULL, CH_SLIM);
        let sliced = slice_wavenet_model(&model, CH_SLIM).unwrap();

        assert_eq!(sliced.ch, CH_SLIM);
        assert_eq!(sliced.head, CH_SLIM);
        assert_eq!(sliced.k, TEST_KERNEL);
        assert_eq!(sliced.head_scale, 0.02);
        assert_eq!(sliced.arrays.len(), 2);

        assert_eq!(sliced.arrays[0].ch, CH_SLIM);
        assert_eq!(sliced.arrays[0].in_ch, 1);
        assert_eq!(sliced.arrays[1].ch, CH_SLIM);
        assert_eq!(sliced.arrays[1].in_ch, CH_SLIM);
        assert!(sliced.condition_dsp.is_none());
        assert_eq!(sliced.arrays[0].effective_layers, 3);
        assert_eq!(sliced.arrays[1].effective_layers, 3);
    }

    #[test]
    fn test_slice_wavenet_model_through_method() {
        let model = make_full_model(CH_FULL, CH_SLIM);
        let sliced = model.slice_channels(CH_SLIM).unwrap();
        assert_eq!(sliced.ch, CH_SLIM);
        assert_eq!(sliced.arrays.len(), 2);
        assert_eq!(sliced.arrays[0].ch, CH_SLIM);
        assert_eq!(sliced.arrays[1].ch, CH_SLIM);
    }

    #[test]
    fn test_slice_wavenet_model_preserves_inference_shape() {
        let mut model = make_full_model(CH_FULL, CH_SLIM);
        let sliced = slice_wavenet_model(&model, CH_SLIM).unwrap();

        model.prewarm();

        let input = vec![0.5f32; 64];
        let mut output_full = vec![0.0f32; 64];
        let mut output_slim = vec![0.0f32; 64];

        model.process(&input, &mut output_full);

        let mut sliced_mut = sliced;
        sliced_mut.prewarm();
        sliced_mut.process(&input, &mut output_slim);

        assert_eq!(output_full.len(), output_slim.len());
    }

    #[test]
    #[should_panic(expected = "slice_wavenet_model: new_ch must be > 0")]
    fn test_slice_wavenet_model_zero_ch_panics() {
        let model = make_full_model(CH_FULL, CH_SLIM);
        slice_wavenet_model(&model, 0).unwrap();
    }

    #[test]
    #[should_panic(expected = "slice_wavenet_model: new_ch")]
    fn test_slice_wavenet_model_too_large_ch_panics() {
        let model = make_full_model(CH_FULL, CH_SLIM);
        slice_wavenet_model(&model, CH_FULL + 1).unwrap();
    }

    #[test]
    fn test_slice_wavenet_model_arrays_different_ch() {
        let model = make_full_model(8, 4);
        let sliced = slice_wavenet_model(&model, 4).unwrap();
        assert_eq!(sliced.ch, 4);
        assert_eq!(sliced.arrays[0].ch, 4);
        assert_eq!(sliced.arrays[0].in_ch, 1);
        assert_eq!(sliced.arrays[1].ch, 4);
        assert_eq!(sliced.arrays[1].in_ch, 4);
    }

    #[test]
    #[should_panic(expected = "exceeds minimum array channel count")]
    fn test_slice_wavenet_model_exceeds_min_array_ch_panics() {
        let model = make_full_model(8, 4);
        slice_wavenet_model(&model, 5).unwrap();
    }
}
