// SPDX-License-Identifier: Apache-2.0

//! Weight loading for WaveNet A2 models.
//!
//! Parses a flat f32 weight stream in NAM JSON order and populates the
//! model layers with quantized weights and col-major-per-tap layouts.

use super::WaveNetA2;
use crate::math::common::{
    AlignedVec, InstructionSet, PrefetchFn, SimdMathConfig, quantize_weight,
};
use crate::models::a2::conv1d_ch3::A2Conv1dCh3;
use crate::models::a2::conv1d_ch8::A2Conv1dCh8;
use crate::models::a2::head::A2HeadConv;
use crate::models::a2::layer::A2Layer;
use crate::models::a2::params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_NUM_LAYERS,
};

impl<const CH: usize> WaveNetA2<CH> {
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

            let conv = crate::models::a2::conv1d::A2Conv1d::new(
                conv_w,
                conv_b.clone(),
                true,
                dilation,
                CH,
                CH,
                ksize,
                prefetch_fn,
            );

            // Build CH=3 col-major-per-tap f32 weights if applicable (ÉPICO 2 fix).
            let ch3_conv = if CH == 3 {
                Some(A2Conv1dCh3::new(
                    conv_w_f32, CH, CH, ksize, dilation, conv_b_f32,
                ))
            } else {
                None
            };

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

            layers.push(match (ch3_conv, ch8_conv) {
                (Some(ch3c), _) => A2Layer::new_with_ch3(conv, ch3c, mixin_w, l1x1_w, l1x1_b),
                (_, Some(ch8c)) => A2Layer::new_with_ch8(conv, ch8c, mixin_w, l1x1_w, l1x1_b),
                _ => A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b),
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
