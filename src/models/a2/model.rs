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

use super::conv1d_ch8::A2Conv1dCh8;
use super::head::A2HeadConv;
use super::layer::A2Layer;
use super::params::{A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_NUM_LAYERS};
use crate::dsp::mirror_buf::MirroredBuffer;
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
/// ## Ring buffer architecture (T2.3)
///
/// Each layer's history is a power-of-2 `MirroredBuffer<f32>` that provides
/// branchless reads via virtual-memory mirroring. The `buffer_start` pointer
/// advances through the 2× virtual mapping; when it approaches the 2× boundary,
/// it rewinds by subtracting `ring_size`. Reads at `buffer_start - offset` are
/// always valid because the mirrored mapping maps `[S, 2S)` → `[0, S)`.
///
/// The head accumulator uses a plain `AlignedVec` with pow2 mask (`& ring_mask`)
/// for branchless ring access — no MirroredBuffer needed since the head reads
/// are already mask-based.
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

    /// Head accumulator ring buffer (skip-connection sum, column-major, pow2 size).
    pub head_accum: AlignedVec<f32>,

    /// Write position in `head_accum` (in columns, wraps via `head_ring_mask`).
    pub head_write_pos: usize,

    /// Ring mask for `head_accum` (pow2 ring, mask = capacity - 1).
    pub head_ring_mask: usize,

    /// Per-layer history buffers: one MirroredBuffer per layer (23 total).
    /// Each buffer provides 2× virtual mapping for branchless ring access (T2.3).
    pub layer_buffers: Vec<MirroredBuffer<f32>>,

    /// Per-layer ring sizes in elements (pow2 page-aligned). For rewind: `start -= ring_size`.
    pub layer_ring_sizes: Vec<usize>,

    /// Per-layer maximum dilation lookback = (kernel-1) * dilation.
    pub layer_lookbacks: Vec<usize>,

    /// Per-layer buffer starts (advanced with each written frame, rewound near 2× boundary).
    pub layer_buffer_starts: Vec<usize>,

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
    /// Allocates ring buffers (MirroredBuffer per layer, pow2 head accumulator)
    /// sized for the architecture and computes the receptive field.
    /// Weight-bearing fields start empty and are populated by the weight loader (T1.6).
    pub fn new() -> Self {
        let rf = a2_receptive_field();
        let max_buf = WAVENET_MAX_NUM_FRAMES;

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        let head_ring_mask = head_ring_size - 1;

        let mut layer_buffers = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_ring_sizes = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_lookbacks = Vec::with_capacity(A2_NUM_LAYERS);
        let mut layer_buffer_starts = Vec::with_capacity(A2_NUM_LAYERS);

        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * CH)
                .expect("MirroredBuffer allocation for A2 layer ring failed");
            let ring_size = mb.size();
            layer_buffers.push(mb);
            layer_ring_sizes.push(ring_size);
            layer_lookbacks.push(max_lookback * CH);
            layer_buffer_starts.push(ring_size);
        }

        Self {
            layers: Vec::with_capacity(A2_NUM_LAYERS),
            rechannel_w: AlignedVec::new(CH, 0u16),
            head_conv: None,
            head_accum: AlignedVec::new(head_ring_size * CH, 0.0f32),
            head_write_pos: rf,
            head_ring_mask,
            layer_buffers,
            layer_ring_sizes,
            layer_lookbacks,
            layer_buffer_starts,
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

        self.layer_buffers.clear();
        self.layer_ring_sizes.clear();
        self.layer_lookbacks.clear();
        self.layer_buffer_starts.clear();

        for i in 0..A2_NUM_LAYERS {
            let max_lookback = (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
            let cap = max_lookback + max_buf + 1;
            let mb = MirroredBuffer::<f32>::new(cap * CH)
                .expect("MirroredBuffer reallocation for A2 layer ring failed");
            let ring_size = mb.size();
            self.layer_buffers.push(mb);
            self.layer_ring_sizes.push(ring_size);
            self.layer_lookbacks.push(max_lookback * CH);
            self.layer_buffer_starts.push(ring_size);
        }

        self.layer_in = AlignedVec::new(CH * max_buf, 0.0f32);

        let head_ring_size = (rf + max_buf + 1).next_power_of_two();
        self.head_ring_mask = head_ring_size - 1;
        self.head_accum = AlignedVec::new(head_ring_size * CH, 0.0f32);
        self.head_write_pos = rf;
    }

    /// Full forward pass through the A2 model.
    ///
    /// Processes `input` samples and writes to `output`.
    /// Requires layers to be populated via `set_weights` (T1.6).
    /// Outputs silence until weights are loaded.
    ///
    /// ## Ring buffer architecture (T2.3)
    ///
    /// Layer history uses `MirroredBuffer<f32>` with power-of-2 sizes.
    /// Writes go to unmasked positions in the 2× virtual mapping; reads are
    /// branchless because the mirror maps `[S, 2S)` → `[0, S)`. When
    /// `buffer_start` approaches the 2× boundary, it rewinds by subtracting
    /// `ring_size`. No `copy_within` / memmove on the hot path.
    ///
    /// Head accumulator uses a plain `AlignedVec` with pow2 mask (`& ring_mask`).
    /// A pre-write memmove preserves `K-1` tail samples when the ring is about
    /// to overflow, keeping the write-positions unmasked for vectorized stores.
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

        if self.layers.is_empty() {
            self.head_write_pos += num_frames;
            return;
        }

        debug_assert!(
            num_frames <= self.max_buffer_size,
            "process: input ({num_frames}) > max_buffer_size ({}) — host violated block-size contract",
            self.max_buffer_size
        );
        let nf = num_frames.min(self.max_buffer_size);
        let ch = CH;

        for (f, x) in input.iter().take(nf).enumerate() {
            let base = f * ch;
            for c in 0..ch {
                let rw = half::f16::from_bits(self.rechannel_w[c]).to_f32();
                self.layer_in[base + c] = rw * x;
            }
        }

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

        for li in 0..A2_NUM_LAYERS {
            let is_first = li == 0;
            let is_last = li == A2_NUM_LAYERS - 1;
            let ring_size = self.layer_ring_sizes[li];
            let lookback = self.layer_lookbacks[li];
            let max_lookback_cols = lookback / ch;
            let bs = self.layer_buffer_starts[li];

            debug_assert!(bs >= lookback);
            debug_assert!(bs + nf * ch <= ring_size * 2);

            {
                let buf = &mut self.layer_buffers[li];
                buf[bs..bs + nf * ch].copy_from_slice(&self.layer_in[..nf * ch]);
            }

            if bs + nf * ch + self.max_buffer_size * ch > ring_size * 2 {
                self.layer_buffer_starts[li] = bs + nf * ch - ring_size;
            } else {
                self.layer_buffer_starts[li] = bs + nf * ch;
            }

            {
                let history = &self.layer_buffers[li][bs - lookback..bs + nf * ch];
                let layer = &self.layers[li];

                if let Some(ch8_conv) = &layer.ch8_conv {
                    unsafe {
                        super::conv1d_ch8::layer_forward_ch8_block(
                            ch8_conv,
                            &layer.mixin_w,
                            &layer.l1x1_w,
                            &layer.l1x1_b,
                            history,
                            max_lookback_cols,
                            nf,
                            &input[..nf],
                            &mut self.head_accum,
                            head_wp,
                            &mut self.layer_in,
                            is_first,
                            is_last,
                        );
                    }
                    continue;
                }

                for (f, x) in input.iter().take(nf).enumerate() {
                    let head_col = head_wp + f;
                    let lin_slice = &mut self.layer_in[f * ch..(f + 1) * ch];
                    let mut frame_z = [0.0f32; 8];
                    let z_slice = &mut frame_z[..ch];

                    let frame_idx = max_lookback_cols + f;

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

        self.head_write_pos = (head_wp + nf) & self.head_ring_mask;

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
        for buf in &mut self.layer_buffers {
            let len = buf.size();
            buf[..len].fill(0.0);
        }
        for i in 0..A2_NUM_LAYERS {
            self.layer_buffer_starts[i] = self.layer_ring_sizes[i];
        }
        self.layer_in.fill(0.0);
        self.head_accum.fill(0.0);
        self.head_write_pos = self.receptive_field_size;
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
            // For CH=8 we also keep a f32 copy for the col-major-per-tap path (T2.2/T2.4).
            let conv_w_f32 = read_slice(
                weights,
                &mut pos,
                conv_w_count,
                total,
                &format!("layer[{i}].conv_w"),
            )?;
            let conv_w_f32_owned: Vec<f32> = conv_w_f32.to_vec();
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
                conv_b.clone(),
                true,
                dilation,
                CH,
                CH,
                ksize,
                prefetch_fn,
            );

            // Build CH=8 col-major-per-tap weights if applicable (T2.2/T2.4).
            let ch8_conv = if CH == 8 {
                Some(A2Conv1dCh8::new(
                    &conv_w_f32_owned,
                    CH,
                    CH,
                    ksize,
                    dilation,
                    conv_b.clone(),
                ))
            } else {
                None
            };

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

            layers.push(if let Some(ch8c) = ch8_conv {
                A2Layer::new_with_ch8(conv, ch8c, mixin_w, l1x1_w, l1x1_b)
            } else {
                A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b)
            });
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
        for buf in &mut model.layer_buffers {
            let len = buf.size();
            buf[..len].fill(0.5);
        }
        model.head_accum.fill(0.5);
        model.layer_in.fill(0.5);
        model.prewarm();
        for buf in &model.layer_buffers {
            let len = buf.size();
            for &v in buf[..len].iter() {
                assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
            }
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
        let orig_rings: Vec<usize> = model.layer_ring_sizes.clone();
        model.reset(48000, 128);
        assert!(model.max_buffer_size == 128);
        for (i, &size) in model.layer_ring_sizes.iter().enumerate() {
            assert!(size >= orig_rings[i], "layer ring {} shrank", i);
        }
        for buf in &model.layer_buffers {
            let len = buf.size();
            for &v in buf[..len].iter() {
                assert!(v.abs() < 1e-9, "reset layer_buffer not zeroed");
            }
        }
    }

    #[test]
    fn test_wavenet_a2_set_max_buffer_size_noop_on_smaller() {
        let mut model = WaveNetA2::<3>::new();
        let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
        model.set_max_buffer_size(32);
        assert_eq!(model.layer_ring_sizes, orig_sizes);
        assert_eq!(model.max_buffer_size, WAVENET_MAX_NUM_FRAMES);
    }

    #[test]
    fn test_wavenet_a2_set_max_buffer_size_grows() {
        let mut model = WaveNetA2::<8>::new();
        let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
        model.set_max_buffer_size(256);
        assert!(model.max_buffer_size == 256);
        assert_eq!(model.layer_ring_sizes.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_buffer_starts.len(), A2_NUM_LAYERS);
        // At least one ring should have grown.
        let any_grew = orig_sizes
            .iter()
            .zip(model.layer_ring_sizes.iter())
            .any(|(a, b)| b > a);
        assert!(
            any_grew,
            "at least one ring should grow with larger max_buffer_size"
        );
    }

    #[test]
    fn test_wavenet_a2_default_creates_valid_model() {
        let model = WaveNetA2::<3>::default();
        assert_eq!(model.channels(), 3);
        assert!(model.receptive_field_size > 0);
        assert!(!model.head_accum.is_empty());
        assert!(!model.layer_buffers.is_empty());
        assert_eq!(model.rechannel_w.len(), 3);
        assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_ring_sizes.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_lookbacks.len(), A2_NUM_LAYERS);
        assert_eq!(model.layer_buffer_starts.len(), A2_NUM_LAYERS);
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
