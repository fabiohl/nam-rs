// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `SlimmableModel` trait — models that can dynamically scale quality/complexity
//! at runtime without reallocation.
//!
//! This is the official NAM architecture for runtime quality scaling.
//!
//! ## Architectural divergence from C++ NAM — `SlimmableWavenet` staging
//!
//! The C++ reference implementation (`NeuralAmpModelerCore`) uses
//! `std::atomic<shared_ptr<WaveNet>>` for model staging: the DSP thread
//! atomically swaps the `shared_ptr` and the old model's reference count is
//! decremented inline. If the count reaches zero, the destructor runs on the
//! audio thread — potentially triggering `WaveNet`'s heap deallocation
//! (hundreds of KB to MB) inside the real-time callback.
//!
//! NAM-rs **intentionally diverges** from this design to guarantee strict
//! zero-heap-deallocation on the DSP hot-path. The old model is never dropped
//! on the audio thread. Instead, it is packed into a `GcItem::Model(…)` and
//! routed through the **SPSC GC pipeline**:
//!
//! - **RT thread (producer)**: `gc_cascade(item, &mut producer, &mut parking_lot)`
//!   attempts to push the `GcItem` into a lock-free `rtrb` SPSC ring buffer.
//!   If full, the item cascades into a small fixed-size `parking_lot` overflow
//!   array (16 slots, also lock-free). If the parking lot is exhausted, the
//!   item is safely dropped on the RT thread as a last-resort fallback (this
//!   should never occur under normal operation).
//!
//! - **Main thread (consumer)**: Pumps the GC channel periodically via
//!   `drain_gc_channels(&mut consumer, &overflow, &rt_status)`, pulling all
//!   pending `GcItem`s and dropping them outside the real-time callback —
//!   typically during Pipewire housekeeping cycles or CLAP plugin idle
//!   callbacks.
//!
//! This architecture removes all `std::sync::atomic`, `std::shared_ptr`,
//! and reference-counting contention from the audio path. The only
//! synchronization between the two threads is the SPSC ring buffer's atomic
//! head/tail indices — a single `fetch_add` on the producer side, no CAS loops,
//! no lock contention, and zero heap deallocation on the RT thread.
//!
//! ### Lifecycle of a slimmable swap
//!
//! 1. Async/main thread calls `model.slice_channels(target_ch)` to build a
//!    new `WaveNetModelDyn` with reduced channel count (allocating weights and
//!    states). Prewarm stabilizes the convolutional state.
//!
//! 2. The new model replaces the old via `Option<Box<StaticModel>>::replace()`
//!    — a simple pointer swap, no atomics needed on the stack-local `Option`.
//!
//! 3. The old `Box<StaticModel>` is handed to `gc_cascade` for non-RT disposal.
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
//! - `Conv1dDyn`: `[block][kernel][in_ch][W]` — W-wide lane-interleaved (W = 4, 8, or 16)
//! - `DenseLayerDyn`: column-major `weights[in_c * out_ch + out_c]`

use crate::common::diagnostics::NamErrorCode;
use crate::common::spsc::GcItem;
use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
use crate::math::common::AlignedVec;
use crate::models::wavenet::{
    Conv1dDyn, DenseLayerDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn, WaveNetLayerDyn,
    WaveNetLayerState, WaveNetModelDyn,
};
use crate::models::{NamModel, StaticModel};

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

    /// Returns the breakpoints at which slimmable quality transitions occur.
    ///
    /// Each breakpoint represents a normalized value in `[0.0, 1.0]` where the
    /// model switches to a different submodel or internal quality tier. Hosts
    /// and plugins can use these to map and snap discrete quality parameters.
    ///
    /// Defaults to an empty vector for models without discrete breakpoints.
    fn slimmable_breakpoints(&self) -> Vec<f64> {
        vec![]
    }
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
pub fn slice_conv1d(
    conv: &Conv1dDyn,
    new_in_ch: usize,
    new_out_ch: usize,
) -> Result<Conv1dDyn, NamErrorCode> {
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

    let dst_width = select_interleave_width(new_out_ch);
    let src_width = conv.interleave_width;
    let new_num_blocks = new_out_ch.div_ceil(dst_width);
    let kernel = conv.kernel;
    let new_weights_len = new_num_blocks * dst_width * new_in_ch * kernel;
    let mut new_weights = AlignedVec::new(new_weights_len, 0.0f32)?;

    for src_b in 0..new_out_ch.div_ceil(src_width) {
        for k in 0..kernel {
            for in_c in 0..new_in_ch {
                for lane in 0..src_width {
                    let out_c = src_b * src_width + lane;
                    if out_c >= new_out_ch {
                        break;
                    }
                    let dst_b = out_c / dst_width;
                    let dst_lane = out_c % dst_width;
                    let src_idx =
                        (src_b * kernel + k) * conv.in_ch * src_width + in_c * src_width + lane;
                    let dst_idx =
                        (dst_b * kernel + k) * new_in_ch * dst_width + in_c * dst_width + dst_lane;
                    new_weights[dst_idx] = conv.weights[src_idx];
                }
            }
        }
    }

    let mut new_bias = AlignedVec::new(new_out_ch, 0.0f32)?;
    new_bias.copy_from_slice(&conv.bias[..new_out_ch]);

    Ok(Conv1dDyn {
        weights: new_weights,
        bias: new_bias,
        do_bias: conv.do_bias,
        dilation: conv.dilation,
        in_ch: new_in_ch,
        out_ch: new_out_ch,
        num_blocks: new_num_blocks,
        interleave_width: dst_width,
        kernel,
        prefetch_fn: conv.prefetch_fn,
    })
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
pub fn slice_dense(
    dense: &DenseLayerDyn,
    new_in_ch: usize,
    new_out_ch: usize,
) -> Result<DenseLayerDyn, NamErrorCode> {
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

    let mut new_weights = AlignedVec::new(new_in_ch * new_out_ch, 0.0f32)?;

    for in_c in 0..new_in_ch {
        let src_start = in_c * dense.out_ch;
        let dst_start = in_c * new_out_ch;
        new_weights[dst_start..dst_start + new_out_ch]
            .copy_from_slice(&dense.weights[src_start..src_start + new_out_ch]);
    }

    let mut new_bias = AlignedVec::new(new_out_ch, 0.0f32)?;
    new_bias.copy_from_slice(&dense.bias[..new_out_ch]);

    Ok(DenseLayerDyn {
        in_ch: new_in_ch,
        out_ch: new_out_ch,
        weights: new_weights,
        bias: new_bias,
        do_bias: dense.do_bias,
    })
}

/// Creates a new `WaveNetLayerDyn` with reduced internal channel count.
///
/// Slices all three internal tensors:
/// - `conv1d`: `(ch, ch)` → `(new_ch, new_ch)`
/// - `input_mixin`: `(cond, ch)` → `(cond, new_ch)`
/// - `one_by_one`: `(ch, ch)` → `(new_ch, new_ch)`
pub fn slice_wavenet_layer(
    layer: &WaveNetLayerDyn,
    new_ch: usize,
) -> Result<WaveNetLayerDyn, NamErrorCode> {
    let conv1d = slice_conv1d(&layer.conv1d, new_ch, new_ch)?;
    let input_mixin = slice_dense(&layer.input_mixin, layer.input_mixin.in_ch, new_ch)?;
    let one_by_one = slice_dense(&layer.one_by_one, new_ch, new_ch)?;
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
    let rechannel = slice_dense(&array.rechannel, new_in_ch, new_ch)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;

    let mut layers = Vec::with_capacity(array.layers.len());
    let mut states = Vec::with_capacity(array.layers.len());

    for layer in &array.layers {
        layers.push(
            slice_wavenet_layer(layer, new_ch).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}"))
            })?,
        );
        let rf = (layer.conv1d.kernel - 1) * layer.conv1d.dilation;
        states.push(WaveNetLayerState::new(new_ch, rf, *alloc_num)?);
        *alloc_num += 1;
    }

    let head_rechannel = slice_dense(&array.head_rechannel, new_ch, array.head)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;

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
        array_outputs: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?,
        head_accum: AlignedVec::new(new_ch * WAVENET_MAX_NUM_FRAMES, 0.0)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?,
        head_outputs: AlignedVec::new(array.head * WAVENET_MAX_NUM_FRAMES, 0.0)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?,
        receptive_field_size,
        block_size,
        block_buffer: AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?,
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
/// - **`condition_dsp`**: Cloned from the original model. Only `WavenetDyn`
///   sub-models are supported for deep cloning; other variants fall back to
///   `None` (matching pre-Task-2.1.2 behavior). The `post_stack_head` and
///   `condition_dsp_output` buffers are allocated fresh.
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
    let head_output_scratch = AlignedVec::new(head_out_ch * WAVENET_MAX_NUM_FRAMES, 0.0)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?;

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
        condition_dsp: crate::models::clone_condition_dsp(&model.condition_dsp),
        condition_dsp_output: AlignedVec::new(cond_dsp_output_size, 0.0)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, format!("{e}")))?,
        post_stack_head: model.post_stack_head.clone(),
        head_output_scratch,
        prewarm_on_reset: model.prewarm_on_reset,
    })
}

/// Usage:
/// Creates a full clone of a WaveNet model for storage, enabling main-thread
/// slimmable rebuilds. The clone uses `slice_wavenet_model` with the original
/// channel count — the immutable weights are duplicated, while states and scratch
/// buffers are allocated fresh (to be reused for inference during rebuild).
pub fn clone_wavenet_for_slimmable_storage(
    model: &WaveNetModelDyn,
) -> std::io::Result<Box<StaticModel>> {
    let full_copy = slice_wavenet_model(model, model.ch)?;
    Ok(Box::new(StaticModel::WavenetDyn(Box::new(full_copy))))
}

/// Centralized helper for slimmable WaveNet channel rebuild.
///
/// Checks if a model slot needs a WaveNet channel count change
/// and performs the allocation-intensive `slice_channels` + GC swap.
///
/// Must be called **before** DSP to keep the hot-path zero-alloc.
/// Callers should obtain `target_ch` via `AdaptiveCompute::take_slimmable_rebuild()`
/// before invoking this function.
///
/// `max_buffer_size`: if `Some(n)`, calls `set_max_buffer_size(n)` after prewarm
/// (needed by CLAP). Pass `None` for standalone mode.
/// `on_gc`: callback to dispose the old model (e.g., `gc_cascade` or `push_to_gc`).
/// `on_slice_error`: callback invoked when `slice_channels` fails.
#[inline(always)]
pub fn try_slimmable_rebuild_single(
    model: &mut Option<Box<StaticModel>>,
    target_ch: usize,
    max_buffer_size: Option<usize>,
    on_gc: &mut impl FnMut(GcItem),
    on_slice_error: &mut impl FnMut(),
) {
    if let Some(model_inner) = model.as_ref()
        && let StaticModel::WavenetDyn(w) = model_inner.as_ref()
        && w.ch != target_ch
    {
        match w.slice_channels(target_ch) {
            Ok(mut new_model) => {
                new_model.prewarm();
                if let Some(max) = max_buffer_size
                    && new_model.set_max_buffer_size(max).is_err()
                {
                    return;
                }
                let old = model.replace(Box::new(StaticModel::WavenetDyn(Box::new(new_model))));
                if let Some(old) = old {
                    on_gc(GcItem::Model(old));
                }
            }
            Err(_) => {
                on_slice_error();
            }
        }
    }
}

#[cfg(test)]
#[path = "slimmable_test.rs"]
mod tests;
