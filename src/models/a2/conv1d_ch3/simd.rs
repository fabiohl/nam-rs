// SPDX-License-Identifier: Apache-2.0

//! CH=3 A2 fast-path SIMD kernels (AVX2+FMA).
//!
//! These functions use f32 native weights and AVX2 batching for the post-conv stage.

use super::A2Conv1dCh3;
use crate::models::a2::params::A2_LEAKY_SLOPE;
use core::arch::x86_64::*;

// AVX2 kernels — unrolled CH=3, f32 native weights, no f16 decode
// =============================================================================

/// Fully unrolled K=6 GEMV for CH=3 with f32 native weights.
///
/// 6 taps × 3 input channels = 18 `_mm_fmadd_ps` instructions.
/// Each FMA loads 4 f32 weights (3 valid + 1 zero padding) via `_mm_loadu_ps`.
///
/// # Safety
/// - `weights` must have at least `kernel * 16` f32 elements.
/// - `layer_buffer` positions for all 6 taps must be valid.
/// - `out_frame` must have at least 4 elements (3 valid + 1 scratch).
#[target_feature(enable = "fma")]
unsafe fn conv1d_ch3_k6_f32(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    const CH: usize = 3;
    let d = dilation as isize;
    let buf = layer_buffer.as_ptr();
    let w_ptr = weights.as_ptr();

    // Pre-compute tap base offsets (in elements, CH=3 stride).
    let fi = frame_idx as isize;
    let t0 = (fi + d * (1 - 6)) as usize * CH;
    let t1 = (fi + d * (2 - 6)) as usize * CH;
    let t2 = (fi + d * (3 - 6)) as usize * CH;
    let t3 = (fi + d * (4 - 6)) as usize * CH;
    let t4 = (fi + d * (5 - 6)) as usize * CH;
    let t5 = fi as usize * CH; // tap 5 = current frame (offset 0)

    // Load bias [b0, b1, b2, 0.0]
    let mut acc = _mm_loadu_ps(bias.as_ptr());

    // Unrolled: 6 taps × 3 input channels = 18 FMAs
    // For each tap: load the 4-wide weight column for each input channel
    macro_rules! fma3 {
        ($tap_base:expr, $k:expr) => {
            // in=0
            {
                let wp = w_ptr.add($k * 16 + 0 * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 0));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }
            // in=1
            {
                let wp = w_ptr.add($k * 16 + 1 * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 1));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }
            // in=2
            {
                let wp = w_ptr.add($k * 16 + 2 * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 2));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }
        };
    }

    fma3!(t0, 0);
    fma3!(t1, 1);
    fma3!(t2, 2);
    fma3!(t3, 3);
    fma3!(t4, 4);
    fma3!(t5, 5);

    _mm_storeu_ps(out_frame.as_mut_ptr(), acc);
}

/// Fully unrolled K=15 GEMV for CH=3 with f32 native weights.
///
/// 15 taps × 3 input channels = 45 `_mm_fmadd_ps` instructions.
#[target_feature(enable = "fma")]
unsafe fn conv1d_ch3_k15_f32(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    const CH: usize = 3;
    let d = dilation as isize;
    let k_limit: isize = 15;
    let buf = layer_buffer.as_ptr();
    let w_ptr = weights.as_ptr();

    let fi = frame_idx as isize;

    macro_rules! tap {
        ($idx:expr) => {
            (fi + d * (($idx as isize) + 1 - k_limit)) as usize * CH
        };
    }

    let t0 = tap!(0);
    let t1 = tap!(1);
    let t2 = tap!(2);
    let t3 = tap!(3);
    let t4 = tap!(4);
    let t5 = tap!(5);
    let t6 = tap!(6);
    let t7 = tap!(7);
    let t8 = tap!(8);
    let t9 = tap!(9);
    let t10 = tap!(10);
    let t11 = tap!(11);
    let t12 = tap!(12);
    let t13 = tap!(13);
    let t14 = tap!(14);

    let mut acc = _mm_loadu_ps(bias.as_ptr());

    macro_rules! fma3 {
        ($tap_base:expr, $k:expr) => {{
            let wp = w_ptr.add($k * 16 + 0 * 4);
            let wv = _mm_loadu_ps(wp);
            let sv = _mm_set1_ps(*buf.add($tap_base + 0));
            acc = _mm_fmadd_ps(wv, sv, acc);
        }
        {
            let wp = w_ptr.add($k * 16 + 1 * 4);
            let wv = _mm_loadu_ps(wp);
            let sv = _mm_set1_ps(*buf.add($tap_base + 1));
            acc = _mm_fmadd_ps(wv, sv, acc);
        }
        {
            let wp = w_ptr.add($k * 16 + 2 * 4);
            let wv = _mm_loadu_ps(wp);
            let sv = _mm_set1_ps(*buf.add($tap_base + 2));
            acc = _mm_fmadd_ps(wv, sv, acc);
        }};
    }

    fma3!(t0, 0);
    fma3!(t1, 1);
    fma3!(t2, 2);
    fma3!(t3, 3);
    fma3!(t4, 4);
    fma3!(t5, 5);
    fma3!(t6, 6);
    fma3!(t7, 7);
    fma3!(t8, 8);
    fma3!(t9, 9);
    fma3!(t10, 10);
    fma3!(t11, 11);
    fma3!(t12, 12);
    fma3!(t13, 13);
    fma3!(t14, 14);

    _mm_storeu_ps(out_frame.as_mut_ptr(), acc);
}

/// Dispatches to the unrolled K=6 or K=15 f32-native kernel.
#[inline(always)]
pub unsafe fn conv1d_ch3_f32_dispatch(
    conv: &A2Conv1dCh3,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    match conv.kernel {
        6 => conv1d_ch3_k6_f32(
            &conv.weights,
            &conv.bias,
            conv.dilation,
            layer_buffer,
            frame_idx,
            out_frame,
        ),
        15 => conv1d_ch3_k15_f32(
            &conv.weights,
            &conv.bias,
            conv.dilation,
            layer_buffer,
            frame_idx,
            out_frame,
        ),
        _ => unreachable!(),
    }
}

// =============================================================================
// Full layer forward pass — conv + post-conv with AVX2 batching
// =============================================================================

/// Full layer forward pass for CH=3 using f32-native conv + AVX2 post-conv batching.
///
/// Processes `num_frames` through:
/// 1. Dilated conv (frame-by-frame, unrolled SSE+FMA, f32 native weights)
/// 2. Post-conv in pairs of frames via `__m256` (AVX2 x86-64-v3 baseline):
///    mixin → LeakyReLU → head accumulate → l1x1 residual
///
/// ## AVX2 batching (T=2 per YMM)
///
/// Each frame produces 4 floats (3 channels + 1 zero padding), fitting one XMM.
/// Two consecutive frames are packed into one YMM (`__m256`, 8 floats = 2×4).
/// This allows mixin, LeakyReLU, head-acc and l1x1 to run as 256-bit operations,
/// halving the number of SIMD instructions in the post-conv stage.
///
/// The scalar tail (0 or 1 frame) is handled without AVX2.
///
/// # Safety
/// Buffers must be sized as indicated by the `debug_assert!` guards.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn layer_forward_ch3_block(
    conv: &A2Conv1dCh3,
    mixin_w: &[f32], // [3] f32 mixin weights
    l1x1_w: &[f32],  // [9] f32 col-major l1x1 weights (padded to [12]? no, use 3×3)
    l1x1_b: &[f32],  // [3] f32 l1x1 bias
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    input_cond: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    const CH: usize = 3;
    const CH_PAD: usize = 4; // SIMD stride (CH=3 padded to 4)

    debug_assert!(mixin_w.len() >= CH);
    debug_assert!(l1x1_w.len() >= CH * CH);
    debug_assert!(l1x1_b.len() >= CH);
    debug_assert!(num_frames <= 64);
    debug_assert!(layer_in.len() >= num_frames * CH);
    debug_assert!(input_cond.len() >= num_frames);

    // ── 1. Conv: frame-by-frame unrolled f32-native kernel ─────────────────
    // z_buf stores conv output: [num_frames][4] (CH=3 + 1 pad lane)
    let mut z_buf = [0.0f32; 64 * CH_PAD]; // 256 f32, stack-allocated

    for f in 0..num_frames {
        let frame_idx = frame_start + f;
        let z_slice = &mut z_buf[f * CH_PAD..(f + 1) * CH_PAD];
        conv1d_ch3_f32_dispatch(conv, layer_buffer, frame_idx, z_slice);
    }

    // ── 2. Post-conv: AVX2 T=2 pairs ───────────────────────────────────────
    // Layout in z_buf: frame f is at z_buf[f * 4 .. f * 4 + 4]
    // Pair (f, f+1) occupies 8 contiguous floats → one __m256 load/store.

    // Broadcast mixin and l1x1 constants as __m128 (4-wide, CH=3+pad).
    let mixin_v4 = _mm_setr_ps(mixin_w[0], mixin_w[1], mixin_w[2], 0.0);
    // Broadcast mixin as __m256 (two frames at once).
    let mixin_v8 = _mm256_set_m128(mixin_v4, mixin_v4);

    // l1x1 row-major for GEMV: l1x1_w[u * CH + c] = weight from input u to output c.
    // For each u, load the output row [w[u*3+0], w[u*3+1], w[u*3+2], 0.0] as __m128.
    // In the AVX2 loop, we compute: acc += z[u] * l1x1_row_u (for u=0,1,2), then add bias.
    let l1x1_row0 = _mm_setr_ps(l1x1_w[0], l1x1_w[1], l1x1_w[2], 0.0); // u=0 → [w[0,0], w[0,1], w[0,2], 0]
    let l1x1_row1 = _mm_setr_ps(l1x1_w[CH], l1x1_w[CH + 1], l1x1_w[CH + 2], 0.0); // u=1
    let l1x1_row2 = _mm_setr_ps(l1x1_w[2 * CH], l1x1_w[2 * CH + 1], l1x1_w[2 * CH + 2], 0.0); // u=2
    let l1x1_b_v4 = _mm_setr_ps(l1x1_b[0], l1x1_b[1], l1x1_b[2], 0.0);

    // Broadcast rows as __m256 for processing 2 frames at once.
    let l1x1_b_v8 = _mm256_set_m128(l1x1_b_v4, l1x1_b_v4);
    let l1x1_row0_v8 = _mm256_set_m128(l1x1_row0, l1x1_row0);
    let l1x1_row1_v8 = _mm256_set_m128(l1x1_row1, l1x1_row1);
    let l1x1_row2_v8 = _mm256_set_m128(l1x1_row2, l1x1_row2);

    let slope_v8 = _mm256_set1_ps(A2_LEAKY_SLOPE);
    let zero_v8 = _mm256_setzero_ps();

    let n_paired = (num_frames / 2) * 2;

    for f in (0..n_paired).step_by(2) {
        let z_off = f * CH_PAD;
        let cond0 = input_cond[f];
        let cond1 = input_cond[f + 1];

        // Load z pair [z_f0[0..4], z_f1[0..4]] as one __m256.
        let mut zv = _mm256_loadu_ps(z_buf.as_ptr().add(z_off));

        // 2a. Mixin: z += mixin_w * cond (per-frame scalar broadcast).
        let cond_v8 = _mm256_setr_ps(cond0, cond0, cond0, cond0, cond1, cond1, cond1, cond1);
        zv = _mm256_fmadd_ps(mixin_v8, cond_v8, zv);

        // 2b. LeakyReLU(0.01) branchless.
        let mask = _mm256_cmp_ps(zv, zero_v8, _CMP_LT_OS);
        let zv_leaky = _mm256_mul_ps(zv, slope_v8);
        zv = _mm256_blendv_ps(zv, zv_leaky, mask);

        // Store back for l1x1 pass.
        _mm256_storeu_ps(z_buf.as_mut_ptr().add(z_off), zv);

        // 2c. Head accumulate (both frames).
        // NOTE: head_accum stride is CH=3. Cannot use _mm_storeu_ps (4 floats would corrupt next slot).
        let head_off0 = (head_col + f) * CH;
        let head_off1 = (head_col + f + 1) * CH;
        let zv_lo = _mm256_castps256_ps128(zv);
        let zv_hi = _mm256_extractf128_ps(zv, 1);
        let z0_lo = _mm_cvtss_f32(zv_lo);
        let z1_lo = _mm_cvtss_f32(_mm_shuffle_ps(zv_lo, zv_lo, 0x55));
        let z2_lo = _mm_cvtss_f32(_mm_shuffle_ps(zv_lo, zv_lo, 0xAA));
        let z0_hi = _mm_cvtss_f32(zv_hi);
        let z1_hi = _mm_cvtss_f32(_mm_shuffle_ps(zv_hi, zv_hi, 0x55));
        let z2_hi = _mm_cvtss_f32(_mm_shuffle_ps(zv_hi, zv_hi, 0xAA));
        if is_first {
            *head_accum.get_unchecked_mut(head_off0) = z0_lo;
            *head_accum.get_unchecked_mut(head_off0 + 1) = z1_lo;
            *head_accum.get_unchecked_mut(head_off0 + 2) = z2_lo;
            *head_accum.get_unchecked_mut(head_off1) = z0_hi;
            *head_accum.get_unchecked_mut(head_off1 + 1) = z1_hi;
            *head_accum.get_unchecked_mut(head_off1 + 2) = z2_hi;
        } else {
            *head_accum.get_unchecked_mut(head_off0) += z0_lo;
            *head_accum.get_unchecked_mut(head_off0 + 1) += z1_lo;
            *head_accum.get_unchecked_mut(head_off0 + 2) += z2_lo;
            *head_accum.get_unchecked_mut(head_off1) += z0_hi;
            *head_accum.get_unchecked_mut(head_off1 + 1) += z1_hi;
            *head_accum.get_unchecked_mut(head_off1 + 2) += z2_hi;
        }

        // 2d. L1x1 residual (skipped on last layer) — pair of frames via AVX2.
        if !is_last {
            // Extract z components for l1x1: broadcast each input channel across both frames.
            // z_f0 is in low 128 bits, z_f1 in high 128 bits.
            // For each output c: acc[f] += l1x1_w[0*CH+c]*z[0] + l1x1_w[1*CH+c]*z[1] + l1x1_w[2*CH+c]*z[2]
            // Use col-major: accumulate column-by-column (u=0,1,2).
            let z0_f0 = _mm_cvtss_f32(zv_lo); // z[f0][0]
            let z1_f0 = _mm_cvtss_f32(_mm_shuffle_ps(zv_lo, zv_lo, 0x55)); // z[f0][1]
            let z2_f0 = _mm_cvtss_f32(_mm_shuffle_ps(zv_lo, zv_lo, 0xAA)); // z[f0][2]
            let z0_f1 = _mm_cvtss_f32(zv_hi);
            let z1_f1 = _mm_cvtss_f32(_mm_shuffle_ps(zv_hi, zv_hi, 0x55));
            let z2_f1 = _mm_cvtss_f32(_mm_shuffle_ps(zv_hi, zv_hi, 0xAA));

            // Broadcast z components as __m256 pairs [f0_z_u, f0_z_u, f0_z_u, f0_z_u, f1_z_u, f1_z_u, f1_z_u, f1_z_u]
            let zu0_v8 = _mm256_setr_ps(z0_f0, z0_f0, z0_f0, z0_f0, z0_f1, z0_f1, z0_f1, z0_f1);
            let zu1_v8 = _mm256_setr_ps(z1_f0, z1_f0, z1_f0, z1_f0, z1_f1, z1_f1, z1_f1, z1_f1);
            let zu2_v8 = _mm256_setr_ps(z2_f0, z2_f0, z2_f0, z2_f0, z2_f1, z2_f1, z2_f1, z2_f1);

            let mut acc8 = l1x1_b_v8;
            acc8 = _mm256_fmadd_ps(zu0_v8, l1x1_row0_v8, acc8);
            acc8 = _mm256_fmadd_ps(zu1_v8, l1x1_row1_v8, acc8);
            acc8 = _mm256_fmadd_ps(zu2_v8, l1x1_row2_v8, acc8);

            // Accumulate into layer_in (add to existing residual).
            // NOTE: layer_in stride is CH=3, not 4. Cannot use _mm_storeu_ps (would corrupt next frame).
            // Extract 3 valid lanes from acc_lo/acc_hi and accumulate scalar.
            let acc_lo = _mm256_castps256_ps128(acc8);
            let acc_hi = _mm256_extractf128_ps(acc8, 1);
            let lin_off0 = f * CH;
            let lin_off1 = (f + 1) * CH;
            // Frame f0: acc_lo lanes [0,1,2]
            *layer_in.get_unchecked_mut(lin_off0) += _mm_cvtss_f32(acc_lo);
            *layer_in.get_unchecked_mut(lin_off0 + 1) +=
                _mm_cvtss_f32(_mm_shuffle_ps(acc_lo, acc_lo, 0x55));
            *layer_in.get_unchecked_mut(lin_off0 + 2) +=
                _mm_cvtss_f32(_mm_shuffle_ps(acc_lo, acc_lo, 0xAA));
            // Frame f1: acc_hi lanes [0,1,2]
            *layer_in.get_unchecked_mut(lin_off1) += _mm_cvtss_f32(acc_hi);
            *layer_in.get_unchecked_mut(lin_off1 + 1) +=
                _mm_cvtss_f32(_mm_shuffle_ps(acc_hi, acc_hi, 0x55));
            *layer_in.get_unchecked_mut(lin_off1 + 2) +=
                _mm_cvtss_f32(_mm_shuffle_ps(acc_hi, acc_hi, 0xAA));
        }
    }

    // ── 3. Scalar tail (0 or 1 remaining frame) ─────────────────────────────
    #[allow(clippy::needless_range_loop)]
    for f in n_paired..num_frames {
        let z_off = f * CH_PAD;
        let cond = input_cond[f];

        // 3a. Mixin (scalar).
        for c in 0..CH {
            z_buf[z_off + c] += mixin_w[c] * cond;
        }

        // 3b. LeakyReLU.
        for c in 0..CH {
            if z_buf[z_off + c] < 0.0 {
                z_buf[z_off + c] *= A2_LEAKY_SLOPE;
            }
        }

        // 3c. Head accumulate.
        let head_off = (head_col + f) * CH;
        if is_first {
            head_accum[head_off..head_off + CH].copy_from_slice(&z_buf[z_off..z_off + CH]);
        } else {
            for c in 0..CH {
                head_accum[head_off + c] += z_buf[z_off + c];
            }
        }

        // 3d. L1x1 residual.
        if !is_last {
            let lin_off = f * CH;
            for c in 0..CH {
                let mut sum = l1x1_b[c];
                for u in 0..CH {
                    sum += l1x1_w[u * CH + c] * z_buf[z_off + u];
                }
                layer_in[lin_off + c] += sum;
            }
        }
    }
}
