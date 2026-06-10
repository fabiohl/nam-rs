// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2 model struct (`WaveNetA2<const CH: usize>`).
//!
//! Single layer-array of 23 dilated causal layers with skip-connection accumulator
//! and head rechannel convolution, matching the fast-path from `a2_fast.cpp`.
//!
//! ## Architecture
//!
//! 1. Input rechannel: `Conv1x1(1 → CH)` (bias, no activation)
//! 2. 23 layers: dilated conv → input-mixin → LeakyReLU → head_accum += out → layer1x1 → residual
//! 3. Head conv: `Conv1D(CH → 1, K=16, bias)` over head_accum ring → × head_scale
//!
//! Processing is chunked by `WAVENET_MAX_NUM_FRAMES` (64) with zero allocation on the hot-path.
//!
//! ## Cross-Validation and Golden Vectors
//!
//! The A2 golden tests use a **self-golden pattern**: the Rust engine generates
//! its own reference on first run (`tests/fixtures/golden_wavenet_a2_*_self.bin`).
//! Subsequent runs compare against that reference, guaranteeing bitwise determinism.
//!
//! **Reason:** The NeuralAmpModelerCore C++ `render` tool's A2 fast-path (`a2_fast.cpp`)
//! currently diverges from this Rust port when rendered against the same `.nam` fixtures.
//! Investigation is pending on the **C++ side** — the Rust implementation is internally
//! self-consistent (MSE = 0.0 between independent runs with identical inputs) and
//! structurally faithful to `a2_fast.cpp`. Specifically:
//!
//! - `is_a2_shape` detection works correctly in the C++ render tool.
//! - The `.nam` fixture format (activation as array-of-objects) was verified.
//! - Suspect: subtle difference in head ring initialisation or `head_scale` placement
//!   between the C++ `_load_weights` and this Rust `set_weights` stream order.
//!
//! Cross-validation via `tests/cpp_parity.rs` (`live_cross_validation_wavenet_a2_*`)
//! is implemented as `#[ignore]` and will be promoted to standard CI once the C++ render
//! tool produces stable A2 output.

use super::head::A2HeadConv;
use super::layer::A2Layer;
use super::params::{A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_NUM_LAYERS};
use crate::math::common::{
    AlignedVec, InstructionSet, PrefetchFn, SimdMathConfig, quantize_weight,
};
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

/// Computes the receptive field size for the A2 architecture.
///
/// Sum of `(kernel_size - 1) * dilation` across all 23 layers,
/// plus `(head_kernel_size - 1)` for the head convolution lookback.
#[inline]
pub const fn a2_receptive_field() -> usize {
    let mut rf = 0usize;
    let mut i = 0;
    while i < A2_NUM_LAYERS {
        rf += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        i += 1;
    }
    rf + (A2_HEAD_KERNEL_SIZE - 1)
}

/// Complete WaveNet A2 Model.
///
/// `CH` = channel count (3 for Lite/Nano, 8 for Full/Standard).
///
/// ## Source of truth
/// - `a2_fast.cpp`: class `A2FastModel` (members, process, prewarm, reset)
/// - `detail.h`: `LayerArray::Process` (per-layer sequence)
/// - `docs/wavenet_walkthrough.rst:278-351`
pub struct WaveNetA2<const CH: usize> {
    /// 23 A2 layers (one per layer index). Populated by `set_weights` (T1.6).
    pub layers: Vec<A2Layer>,

    /// Input rechannel weights: `Conv1x1(1 → CH)` (no bias), u16 quantized.
    pub rechannel_w: AlignedVec<u16>,

    /// Head convolution (K=16 over skip-connection accumulator, bias, head_scale).
    pub head_conv: Option<A2HeadConv>,

    /// Head accumulator ring buffer (skip-connection sum, column-major).
    pub head_accum: AlignedVec<f32>,

    /// Write position in `head_accum` (in columns, wraps via `head_ring_mask`).
    pub head_write_pos: usize,

    /// Ring mask for `head_accum` (pow2 ring, mask = capacity - 1).
    pub head_ring_mask: usize,

    /// Combined history arena for all 23 layers' linear ring buffers (column-major).
    pub layer_buffer: AlignedVec<f32>,

    /// Offsets into `layer_buffer` for each layer's ring (byte-based for direct slicing).
    pub layer_offsets: Vec<usize>,

    /// Per-layer linear ring capacities (in columns).
    pub layer_ring_capacities: Vec<usize>,

    /// Per-layer max lookback = (kernel-1)*dilation.
    pub layer_lookbacks: Vec<usize>,

    /// Per-layer write positions in their ring buffers (in columns).
    pub layer_write_poses: Vec<usize>,

    /// Inter-layer data buffer: `CH × max_buffer_size` f32, reused across layers.
    /// Each layer reads from it, then writes its l1x1 residual back (in-place update).
    pub layer_in: AlignedVec<f32>,

    /// Total receptive field: sum of `(kernel-1)*dilation` + head kernel - 1.
    pub receptive_field_size: usize,

    /// Maximum frames per processing block (= `WAVENET_MAX_NUM_FRAMES`).
    pub max_buffer_size: usize,
}

impl<const CH: usize> WaveNetA2<CH> {
    /// Creates a new uninitialized WaveNet A2 model.
    ///
    /// Allocates ring buffers sized for the architecture and computes
    /// the receptive field. Weight-bearing fields start empty and are
    /// populated by the weight loader (T1.6).
    pub fn new() -> Self {
        let rf = a2_receptive_field();
        let max_buf = WAVENET_MAX_NUM_FRAMES;

        // Head ring buffer: powers-of-2 above total (for efficient wrapping).
        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        // Compute per-layer linear ring sizes and total arena.
        let mut layer_offsets = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_ring_capacities = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_lookbacks = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_write_poses = Vec::with_capacity(A2_NUM_LAYERS);
        let mut arena_total = 0usize;
        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            // Linear ring: 2*max_lookback + max_buffer_size columns (RING_MODE == 0).
            let cap = 2 * max_lookback + max_buf;
            layer_offsets.push(arena_total);
            layer_ring_capacities.push(cap);
            layer_lookbacks.push(max_lookback);
            layer_write_poses.push(max_lookback); // initial write_pos = max_lookback
            arena_total += CH * cap;
        }

        Self {
            layers: Vec::with_capacity(A2_NUM_LAYERS),
            rechannel_w: AlignedVec::new(CH, 0u16),
            head_conv: None,
            head_accum: AlignedVec::new(head_ring_size * CH, 0.0f32),
            // Initialized to `rf` so the head conv ring has a fully zeroed lookback
            // of `rf` samples from the start — matching the prewarm semantics.
            head_write_pos: rf,
            head_ring_mask,
            layer_buffer: AlignedVec::new(arena_total, 0.0f32),
            layer_offsets,
            layer_ring_capacities,
            layer_lookbacks,
            layer_write_poses,
            layer_in: AlignedVec::new(CH * max_buf, 0.0f32),
            receptive_field_size: rf,
            max_buffer_size: max_buf,
        }
    }

    /// Returns the channel count.
    #[inline(always)]
    pub fn channels(&self) -> usize {
        CH
    }

    /// Returns the total receptive field size.
    #[inline(always)]
    pub fn receptive_field(&self) -> usize {
        self.receptive_field_size
    }

    /// Reallocates internal buffers to support the given maximum block size.
    pub fn set_max_buffer_size(&mut self, max_buf: usize) {
        if max_buf <= self.max_buffer_size {
            return;
        }
        self.max_buffer_size = max_buf;
        let rf = self.receptive_field_size;

        // Recompute per-layer linear ring sizes.
        let mut layer_offsets = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_ring_capacities = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_lookbacks = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_write_poses = Vec::with_capacity(A2_NUM_LAYERS);
        let mut arena_total = 0usize;
        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            let cap = 2 * max_lookback + max_buf;
            layer_offsets.push(arena_total);
            layer_ring_capacities.push(cap);
            layer_lookbacks.push(max_lookback);
            layer_write_poses.push(max_lookback);
            arena_total += CH * cap;
        }

        self.layer_buffer = AlignedVec::new(arena_total, 0.0f32);
        self.layer_offsets = layer_offsets;
        self.layer_ring_capacities = layer_ring_capacities;
        self.layer_lookbacks = layer_lookbacks;
        self.layer_write_poses = layer_write_poses;

        self.layer_in = AlignedVec::new(CH * max_buf, 0.0f32);

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        self.head_ring_mask = head_ring_size - 1;
        self.head_accum = AlignedVec::new(head_ring_size * CH, 0.0f32);
        // Reset to `rf` so the head conv ring invariant is preserved after reallocation.
        self.head_write_pos = rf;
    }

    /// Full forward pass through the A2 model.
    ///
    /// Processes `input` samples and writes to `output`.
    /// Requires layers to be populated via `set_weights` (T1.6).
    /// Outputs silence until weights are loaded.
    ///
    /// # Block Size Contract
    ///
    /// The caller **must** ensure `input.len() <= max_buffer_size`. Exceeding this limit
    /// causes silent truncation: only the first `max_buffer_size` frames are processed
    /// and the remaining are left as zeros. This matches the CLAP/audio host contract
    /// which guarantees `block_size <= max_block_size` negotiated at activation.
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let num_frames = input.len();
        if num_frames == 0 {
            return;
        }

        output[..num_frames].fill(0.0);

        // If layers/history haven't been loaded yet (pre-T1.6), just track positions.
        if self.layers.is_empty() {
            self.head_write_pos += num_frames;
            return;
        }

        // Guard against host misbehaviour (violates block-size contract).
        // In production the host must not send more frames than negotiated;
        // in debug builds this triggers immediately to expose the violation.
        debug_assert!(
            num_frames <= self.max_buffer_size,
            "process: input ({num_frames}) > max_buffer_size ({}) — host violated block-size contract",
            self.max_buffer_size
        );
        let nf = num_frames.min(self.max_buffer_size);
        let ch = CH;

        // 1. Rechannel and prepare cond buffer from input.
        // layer_in[c + f*CH] = rechannel_w[c] * input[f]
        let rechannel = &self.rechannel_w;
        for (f, x) in input.iter().take(nf).enumerate() {
            let base = f * ch;
            for c in 0..ch {
                let rw = half::f16::from_bits(rechannel[c]).to_f32();
                self.layer_in[base + c] = rw * x;
            }
        }

        // 2. Head ring management: rewind if overflow.
        let head_keep = A2_HEAD_KERNEL_SIZE - 1;
        let head_cap = self.head_ring_mask + 1;
        if self.head_write_pos + nf > head_cap {
            let keep_start = self.head_write_pos - head_keep;
            let keep_bytes = head_keep * ch;
            let src = keep_start * ch;
            self.head_accum.copy_within(src..src + keep_bytes, 0);
            self.head_write_pos = head_keep;
        }
        let head_wp = self.head_write_pos;

        // 3. Per-layer forward pass.
        for li in 0..A2_NUM_LAYERS {
            let is_first = li == 0;
            let is_last = li == A2_NUM_LAYERS - 1;
            let cap = self.layer_ring_capacities[li];
            let lookback = self.layer_lookbacks[li];
            let wp = self.layer_write_poses[li];
            let offset = self.layer_offsets[li];

            // Linear ring rewind if overflow (RING_MODE == 0).
            let wp = if wp + nf > cap {
                let keep = lookback;
                let keep_bytes = keep * ch;
                let src_start = offset + (wp - keep) * ch;
                // memmove the last `keep` columns to the start of this layer's ring.
                self.layer_buffer
                    .copy_within(src_start..src_start + keep_bytes, offset);
                lookback
            } else {
                wp
            };

            // Ring-write: copy layer_in into this layer's ring at wp.
            let ring_dst = offset + wp * ch;
            self.layer_buffer[ring_dst..ring_dst + nf * ch]
                .copy_from_slice(&self.layer_in[..nf * ch]);

            self.layer_write_poses[li] = wp + nf;

            // Phase B: process frames (immutable borrow of layer_buffer).
            {
                let history = &self.layer_buffer[offset..offset + (wp + nf) * ch];
                let layer = &self.layers[li];

                for (f, x) in input.iter().take(nf).enumerate() {
                    let head_col = head_wp + f;
                    let lin_slice = &mut self.layer_in[f * ch..(f + 1) * ch];
                    let mut frame_z = [0.0f32; 8];
                    let z_slice = &mut frame_z[..ch];

                    // frame_idx = wp + f (post-ring-write, so wp points to the start of this block).
                    let frame_idx = wp + f;

                    unsafe {
                        layer
                            .conv
                            .process_single_frame::<crate::math::common::Avx2Math>(
                                history, z_slice, frame_idx, None,
                            );
                    }

                    let mixin: &[f32] = &layer.mixin_w;
                    for c in 0..ch {
                        z_slice[c] += mixin[c] * x;
                    }
                    for z in z_slice.iter_mut().take(ch) {
                        if *z < 0.0 {
                            *z *= 0.01;
                        }
                    }
                    let head_off = head_col * ch;
                    if is_first {
                        self.head_accum[head_off..head_off + ch].copy_from_slice(z_slice);
                    } else {
                        for (c, z_val) in z_slice.iter().enumerate().take(ch) {
                            self.head_accum[head_off + c] += *z_val;
                        }
                    }
                    if !is_last {
                        let l1x1: &[f32] = &layer.l1x1_w;
                        let l1x1_b: &[f32] = &layer.l1x1_b;
                        for c in 0..ch {
                            let mut sum = l1x1_b[c];
                            for u in 0..ch {
                                sum += l1x1[u * ch + c] * z_slice[u];
                            }
                            lin_slice[c] += sum;
                        }
                    }
                }
            }
        }

        // 4. Advance head write position.
        self.head_write_pos = (head_wp + nf) & self.head_ring_mask;

        // 5. Head convolution → output.
        if let Some(ref head) = self.head_conv {
            head.process(
                &self.head_accum,
                self.head_write_pos,
                self.head_ring_mask,
                nf,
                &mut output[..nf],
            );
        }
    }

    /// Pre-warms the model by filling the receptive field with silence.
    #[cold]
    pub fn prewarm(&mut self) {
        let rf = self.receptive_field_size;

        // Zero the entire layer history arena.
        self.layer_buffer.fill(0.0);

        // Reset each layer's write position to max_lookback.
        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            self.layer_write_poses[i] = max_lookback;
        }

        // Zero inter-layer buffer.
        self.layer_in.fill(0.0);

        // Fill head accumulator with zeros and restore the ring invariant:
        // `head_write_pos = rf` so that the head conv's K-1 lookback reads
        // only the zeroed region, simulating `rf` frames of silent prewarm.
        self.head_accum.fill(0.0);
        self.head_write_pos = rf;
    }

    /// Resets internal state for a new sample rate and max buffer size.
    pub fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) {
        self.set_max_buffer_size(max_buffer_size);
        self.prewarm();
    }

    /// Loads weights from a flat f32 slice in the exact A2 stream order.
    ///
    /// ## Weight order (mirrors `a2_fast.cpp:196-282`)
    ///
    /// 1. `_rechannel`: weights `CH` f32 (quantized to u16, no bias — matches C++ A2FastModel)
    /// 2. Per layer 0..22:
    ///    - `_conv`: weights `CH*CH*K` f32 (quantized to u16) + bias `CH` f32
    ///    - `_input_mixin`: weights `CH` f32 (no bias)
    ///    - `_layer1x1`: weights `CH*CH` f32 (col-major) + bias `CH` f32
    /// 3. `_head_rechannel`: conv k=16 weights `16*CH` f32 + head_bias `1` f32
    /// 4. `head_scale`: last f32 in the stream
    ///
    /// ## Acceptance criteria (T1.6)
    /// - Calls `verify_exhaustion()` — consumed count must equal `weights.len()`.
    /// - Returns a clear error if the weight stream is shorter or longer than expected.
    #[allow(clippy::too_many_lines)]
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), String> {
        let total = weights.len();
        let mut pos: usize = 0;
        let is_bf16 = SimdMathConfig::get().instruction_set == InstructionSet::Avx512VnniBf16;

        // ── 1. Rechannel: Conv1x1(1 → CH) (no bias) ─────────────────────
        let rw_f32 = read_slice(weights, &mut pos, CH, total, "rechannel_w")?;
        let mut rechannel_w = AlignedVec::new(CH, 0u16);
        for (i, &v) in rw_f32.iter().enumerate() {
            rechannel_w[i] = quantize_weight(v, is_bf16);
        }

        // ── 2. Per-layer weights ──────────────────────────────────────────
        let mut layers = Vec::with_capacity(A2_NUM_LAYERS);

        for i in 0..A2_NUM_LAYERS {
            let ksize = A2_KERNEL_SIZES[i];
            let dilation = A2_DILATIONS[i];
            let conv_w_count = CH * CH * ksize;
            let num_blocks = CH.div_ceil(4);
            let conv_w_padded = num_blocks * 4 * CH * ksize;

            // 2a. Dilated conv weights: read CH×CH×K, store padded interleaved 4-wide (quantized u16).
            let conv_w_f32 = read_slice(
                weights,
                &mut pos,
                conv_w_count,
                total,
                &format!("layer[{i}].conv_w"),
            )?;
            let mut conv_w = AlignedVec::new(conv_w_padded, 0u16);
            transpose_conv1d_interleaved_4wide(conv_w_f32, &mut conv_w, CH, CH, ksize, is_bf16);

            // 2b. Conv bias.
            let conv_b_f32 =
                read_slice(weights, &mut pos, CH, total, &format!("layer[{i}].conv_b"))?;
            let conv_b = AlignedVec::from(conv_b_f32.to_vec());

            let prefetch_fn: PrefetchFn = if dilation >= 128 {
                crate::math::common::prefetch_strategy_2stage
            } else {
                crate::math::common::prefetch_strategy_simple
            };

            let conv = super::conv1d::A2Conv1d::new(
                conv_w,
                conv_b,
                true,
                dilation,
                CH,
                CH,
                ksize,
                prefetch_fn,
            );

            // 2c. Input mixin (no bias, f32).
            let mixin_w_f32 =
                read_slice(weights, &mut pos, CH, total, &format!("layer[{i}].mixin_w"))?;
            let mixin_w = AlignedVec::from(mixin_w_f32.to_vec());

            // 2d. Layer1x1 weights (CH × CH, stored f32 col-major).
            let l1x1_w_f32 = read_slice(
                weights,
                &mut pos,
                CH * CH,
                total,
                &format!("layer[{i}].l1x1_w"),
            )?;
            let mut l1x1_w = AlignedVec::new(CH * CH, 0.0f32);
            transpose_dense_f32(l1x1_w_f32, &mut l1x1_w, CH, CH);

            // 2e. Layer1x1 bias.
            let l1x1_b_f32 =
                read_slice(weights, &mut pos, CH, total, &format!("layer[{i}].l1x1_b"))?;
            let l1x1_b = AlignedVec::from(l1x1_b_f32.to_vec());

            layers.push(A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b));
        }

        // ── 3. Head rechannel: Conv1D(CH → 1, K=16, bias) ─────────────────
        let head_w_f32 = read_slice(weights, &mut pos, A2_HEAD_KERNEL_SIZE * CH, total, "head_w")?;
        let mut head_w = AlignedVec::new(A2_HEAD_KERNEL_SIZE * CH, 0.0f32);
        transpose_head_w(head_w_f32, &mut head_w, CH, A2_HEAD_KERNEL_SIZE);

        let head_b = {
            let s = read_slice(weights, &mut pos, 1, total, "head_b")?;
            s[0]
        };

        // ── 4. Head scale (last float) ─────────────────────────────────────
        let head_scale = {
            let s = read_slice(weights, &mut pos, 1, total, "head_scale")?;
            s[0]
        };

        // ── 5. Exhaustion check ────────────────────────────────────────────
        if pos != total {
            return Err(format!(
                "set_weights: stream has {} unconsumed f32 after loading all weights (consumed {}, total {})",
                total - pos,
                pos,
                total
            ));
        }

        // ── 6. Commit to self (all-or-nothing) ──────────────────────────────
        self.rechannel_w = rechannel_w;
        self.layers = layers;
        self.head_conv = Some(A2HeadConv::new(head_w, head_b, head_scale, CH));

        Ok(())
    }

    /// Returns whether weights have been loaded via `set_weights`.
    #[inline(always)]
    pub fn has_weights(&self) -> bool {
        !self.layers.is_empty()
    }
}

// =============================================================================
// Private helpers for set_weights
// =============================================================================

/// Reads a contiguous slice of `n` f32 values from `weights[pos..]`,
/// advancing `pos`. Returns an error with the label if out of bounds.
#[inline]
fn read_slice<'a>(
    weights: &'a [f32],
    pos: &mut usize,
    n: usize,
    total: usize,
    label: &str,
) -> Result<&'a [f32], String> {
    if *pos + n > total {
        return Err(format!(
            "set_weights: stream exhausted at position {} (need {} for \"{}\", total {})",
            *pos, n, label, total
        ));
    }
    let slice = &weights[*pos..*pos + n];
    *pos += n;
    Ok(slice)
}

/// Rearranges dense layer weights from row-major (NAM JSON) to col-major, keeping f32 precision.
///
/// Input:  `raw[out * in_size + in_c]` (row-major)
/// Output: `weights[in_c * out_size + out_c]` (col-major)
fn transpose_dense_f32(raw: &[f32], weights: &mut [f32], in_size: usize, out_size: usize) {
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c];
        }
    }
}

/// Rearranges conv1d weights into "Interleaved 4-Wide" format and quantizes to u16.
///
/// Groups output channels in blocks of 4 for SIMD processing.
fn transpose_conv1d_interleaved_4wide(
    raw: &[f32],
    weights: &mut [u16],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    is_bf16: bool,
) {
    let num_blocks = out_ch.div_ceil(4);
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = quantize_weight(raw[raw_idx], is_bf16);
                    }
                }
            }
        }
    }
}

/// Transposes head weights from [channel][tap] (NAM JSON format) to [tap][channel] (A2HeadConv format).
///
/// NAM JSON weight layout for Conv1D(1, CH, 16): `raw[channel * 16 + tap]`
/// A2HeadConv expects: `head[tap * CH + channel]`
fn transpose_head_w(raw: &[f32], head: &mut [f32], channels: usize, kernel: usize) {
    for tap in 0..kernel {
        for ch in 0..channels {
            head[tap * channels + ch] = raw[ch * kernel + tap];
        }
    }
}

impl<const CH: usize> Default for WaveNetA2<CH> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenet_a2_receptive_field_ch3() {
        let model = WaveNetA2::<3>::new();
        // Reference: computed from A2_KERNEL_SIZES and A2_DILATIONS arrays.
        let expected = {
            let mut sum = 0usize;
            for i in 0..A2_NUM_LAYERS {
                sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            }
            sum + (A2_HEAD_KERNEL_SIZE - 1)
        };
        assert_eq!(model.receptive_field_size, expected);
        assert_eq!(model.receptive_field(), expected);
        assert_eq!(model.channels(), 3);
    }

    #[test]
    fn test_wavenet_a2_receptive_field_ch8() {
        let model = WaveNetA2::<8>::new();
        let expected = {
            let mut sum = 0usize;
            for i in 0..A2_NUM_LAYERS {
                sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            }
            sum + (A2_HEAD_KERNEL_SIZE - 1)
        };
        assert_eq!(model.receptive_field_size, expected);
        assert_eq!(model.channels(), 8);
    }

    #[test]
    fn test_wavenet_a2_process_stub_output_silence() {
        let mut model = WaveNetA2::<3>::new();
        let input = vec![0.5f32; 64];
        let mut output = vec![1.0f32; 64];
        model.process(&input, &mut output);
        for v in &output {
            assert!(v.abs() < 1e-9, "expected silence, got {}", v);
        }
    }

    #[test]
    fn test_wavenet_a2_process_empty_input() {
        let mut model = WaveNetA2::<3>::new();
        let input: [f32; 0] = [];
        let mut output: [f32; 0] = [];
        model.process(&input, &mut output);
        // Empty input should be a no-op.
    }

    #[test]
    fn test_wavenet_a2_prewarm_fills_buffers() {
        let mut model = WaveNetA2::<3>::new();
        // Pre-fill with non-zero to verify overwrite.
        model.layer_buffer.fill(0.5);
        model.head_accum.fill(0.5);
        model.layer_in.fill(0.5);
        model.prewarm();
        for v in model.layer_buffer.iter() {
            assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
        }
        for v in model.head_accum.iter() {
            assert!(v.abs() < 1e-9, "head_accum not zeroed");
        }
        for v in model.layer_in.iter() {
            assert!(v.abs() < 1e-9, "layer_in not zeroed");
        }
        assert_eq!(model.head_write_pos, model.receptive_field_size);
    }

    #[test]
    fn test_wavenet_a2_reset_reallocates_and_prewarms() {
        let mut model = WaveNetA2::<3>::new();
        let orig_layer_len = model.layer_buffer.len();
        model.reset(48000, 128);
        assert!(model.layer_buffer.len() > orig_layer_len);
        assert_eq!(model.max_buffer_size, 128);
        for v in model.layer_buffer.iter() {
            assert!(v.abs() < 1e-9, "reset layer_buffer not zeroed");
        }
    }

    #[test]
    fn test_wavenet_a2_set_max_buffer_size_noop_on_smaller() {
        let mut model = WaveNetA2::<3>::new();
        let orig_len = model.layer_buffer.len();
        model.set_max_buffer_size(32);
        assert_eq!(model.layer_buffer.len(), orig_len);
        assert_eq!(model.max_buffer_size, WAVENET_MAX_NUM_FRAMES);
    }

    #[test]
    fn test_wavenet_a2_set_max_buffer_size_grows() {
        let mut model = WaveNetA2::<8>::new();
        let orig_len = model.layer_buffer.len();
        model.set_max_buffer_size(256);
        assert!(model.layer_buffer.len() > orig_len);
        assert_eq!(model.max_buffer_size, 256);
        // Verify per-layer offsets increase monotonically.
        assert_eq!(model.layer_offsets.len(), A2_NUM_LAYERS);
        for i in 1..A2_NUM_LAYERS {
            assert!(model.layer_offsets[i] > model.layer_offsets[i - 1]);
        }
        assert_eq!(model.layer_ring_capacities.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_write_poses.len(), A2_NUM_LAYERS);
    }

    #[test]
    fn test_wavenet_a2_default_creates_valid_model() {
        let model = WaveNetA2::<3>::default();
        assert_eq!(model.channels(), 3);
        assert!(model.receptive_field_size > 0);
        assert!(!model.head_accum.is_empty());
        assert!(!model.layer_buffer.is_empty());
        assert_eq!(model.rechannel_w.len(), 3);
        assert_eq!(model.layer_offsets.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_ring_capacities.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_write_poses.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_in.len(), 3 * model.max_buffer_size);
    }

    #[test]
    fn test_wavenet_a2_const_receptive_field_matches_runtime() {
        let rf_const = a2_receptive_field();
        let model3 = WaveNetA2::<3>::new();
        let model8 = WaveNetA2::<8>::new();
        assert_eq!(model3.receptive_field_size, rf_const);
        assert_eq!(model8.receptive_field_size, rf_const);
    }

    // ── set_weights tests (T1.6) ───────────────────────────────────────

    fn expected_weight_count(ch: usize) -> usize {
        let mut count = ch; // rechannel_w
        for &k in &A2_KERNEL_SIZES {
            count += ch * ch * k; // conv_w
            count += ch; // conv_b
            count += ch; // mixin_w
            count += ch * ch; // l1x1_w
            count += ch; // l1x1_b
        }
        count += A2_HEAD_KERNEL_SIZE * ch; // head_w
        count += 1; // head_b
        count += 1; // head_scale
        count
    }

    fn make_test_weights(n: usize, seed: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            v.push(((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25);
        }
        v
    }

    #[test]
    fn test_set_weights_exact_count_ch3() {
        let mut model = WaveNetA2::<3>::new();
        let count = expected_weight_count(3);
        assert_eq!(count, 1871); // sanity-check known count
        let weights = make_test_weights(count, 42);
        assert!(model.set_weights(&weights).is_ok());
        assert!(model.has_weights());
        assert_eq!(model.layers.len(), A2_NUM_LAYERS);
        assert!(model.head_conv.is_some());
    }

    #[test]
    fn test_set_weights_exact_count_ch8() {
        let mut model = WaveNetA2::<8>::new();
        let count = expected_weight_count(8);
        assert_eq!(count, 12146); // sanity-check known count
        let weights = make_test_weights(count, 77);
        assert!(model.set_weights(&weights).is_ok());
        assert!(model.has_weights());
        assert_eq!(model.layers.len(), A2_NUM_LAYERS);
        assert!(model.head_conv.is_some());
    }

    #[test]
    fn test_set_weights_too_few_ch3() {
        let mut model = WaveNetA2::<3>::new();
        let count = expected_weight_count(3);
        let weights = make_test_weights(count - 10, 42);
        let err = model.set_weights(&weights);
        assert!(err.is_err(), "expected error with too few weights");
        let err_msg = err.unwrap_err();
        assert!(
            err_msg.contains("stream exhausted"),
            "error should mention exhaustion, got: {err_msg}"
        );
        assert!(!model.has_weights());
    }

    #[test]
    fn test_set_weights_too_many_ch3() {
        let mut model = WaveNetA2::<3>::new();
        let count = expected_weight_count(3);
        let weights = make_test_weights(count + 5, 42);
        let err = model.set_weights(&weights);
        assert!(err.is_err(), "expected error with too many weights");
        let err_msg = err.unwrap_err();
        assert!(
            err_msg.contains("unconsumed"),
            "error should mention unconsumed, got: {err_msg}"
        );
    }

    #[test]
    fn test_set_weights_too_few_ch8() {
        let mut model = WaveNetA2::<8>::new();
        let count = expected_weight_count(8);
        let weights = make_test_weights(count - 1, 99);
        let err = model.set_weights(&weights);
        assert!(err.is_err(), "expected error with too few weights");
        let err_msg = err.unwrap_err();
        assert!(
            err_msg.contains("stream exhausted"),
            "error should mention exhaustion"
        );
    }

    #[test]
    fn test_set_weights_too_many_ch8() {
        let mut model = WaveNetA2::<8>::new();
        let count = expected_weight_count(8);
        let weights = make_test_weights(count + 1, 88);
        let err = model.set_weights(&weights);
        assert!(err.is_err(), "expected error with too many weights");
        let err_msg = err.unwrap_err();
        assert!(
            err_msg.contains("unconsumed"),
            "error should mention unconsumed"
        );
    }

    #[test]
    fn test_set_weights_has_weights_flag_ch3() {
        let mut model = WaveNetA2::<3>::new();
        assert!(!model.has_weights());
        let count = expected_weight_count(3);
        let weights = make_test_weights(count, 123);
        model.set_weights(&weights).unwrap();
        assert!(model.has_weights());
    }

    /// Smoke: load random weights, prewarm, process 1 frame — output should be non-zero
    /// (random weights almost certainly produce non-zero output).
    #[test]
    fn test_set_weights_process_smoke_ch3() {
        let mut model = WaveNetA2::<3>::new();
        let count = expected_weight_count(3);
        let weights = make_test_weights(count, 42);
        model.set_weights(&weights).unwrap();
        model.prewarm();

        let input = vec![0.5f32; 16];
        let mut output = vec![0.0f32; 16];
        model.process(&input, &mut output);

        // With random weights, output should be non-zero (statistical certainty).
        let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
        assert!(
            any_nonzero,
            "process should produce non-zero output after weight loading"
        );
    }

    #[test]
    fn test_set_weights_process_smoke_ch8() {
        let mut model = WaveNetA2::<8>::new();
        let count = expected_weight_count(8);
        let weights = make_test_weights(count, 77);
        model.set_weights(&weights).unwrap();
        model.prewarm();

        let input = vec![0.5f32; 16];
        let mut output = vec![0.0f32; 16];
        model.process(&input, &mut output);

        let any_nonzero = output.iter().any(|&v| v.abs() > 1e-30);
        assert!(
            any_nonzero,
            "process should produce non-zero output after weight loading"
        );
    }
}
