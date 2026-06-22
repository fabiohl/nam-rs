// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::arch::x86_64::*;

// ── Batched f32 GEMV ──────────────────────────────────────────────────────────

/// Batch GEMV overwrite with bias using native f32 weights via AVX2.
///
/// Strategy is shape-dependent:
/// - OUT ≤ 4: batch across `num_frames` — 8 frames per YMM accumulator,
///   broadcast one weight per `in_c` iteration.
/// - OUT ≥ 8: vectorize across output channels (same as `gemv_overwrite_avx2`
///   but with f32 weights loaded directly).
/// - Otherwise: scalar fallback.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_with_bias_f32_avx2(
    in_frames: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    if out_len == 1 {
        for n in 0..num_frames {
            let mut acc = _mm256_setzero_ps();
            let mut in_c = 0;
            while in_c + 8 <= in_len {
                let v_in = _mm256_loadu_ps(in_frames.as_ptr().add(n * in_len + in_c));
                let v_w = _mm256_loadu_ps(weights.as_ptr().add(in_c));
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
                in_c += 8;
            }
            if in_c < in_len {
                let mut buf_in = [0.0f32; 8];
                let mut buf_w = [0.0f32; 8];
                let rem = in_len - in_c;
                for i in 0..rem {
                    buf_in[i] = *in_frames.get_unchecked(n * in_len + in_c + i);
                    buf_w[i] = *weights.get_unchecked(in_c + i);
                }
                let v_in = _mm256_loadu_ps(buf_in.as_ptr());
                let v_w = _mm256_loadu_ps(buf_w.as_ptr());
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
            }
            // Horizontal sum: store to stack and scalar-reduce
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let sum: f32 = tmp.iter().sum();
            *out_frames.get_unchecked_mut(n) = sum + *bias.get_unchecked(0);
        }
        return;
    }

    if out_len <= 4 {
        crate::math::common::scalar_ref::gemv_with_bias_f32_fallback(
            in_frames, weights, bias, out_frames, num_frames,
        );
        return;
    }

    if out_len >= 8 {
        let mut out_c = 0;
        while out_c + 8 <= out_len {
            let mut f = 0;
            while f < num_frames {
                let mut acc0 = _mm256_loadu_ps(bias.as_ptr().add(out_c));
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut acc4 = _mm256_setzero_ps();
                let mut acc5 = _mm256_setzero_ps();
                let mut acc6 = _mm256_setzero_ps();
                let mut acc7 = _mm256_setzero_ps();
                let mut in_c = 0;
                while in_c + 8 <= in_len {
                    let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                    let vs1 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 1));
                    let vs2 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 2));
                    let vs3 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 3));
                    let vs4 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 4));
                    let vs5 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 5));
                    let vs6 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 6));
                    let vs7 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 7));

                    let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                    let w0 = _mm256_loadu_ps(w_ptr);
                    acc0 = _mm256_fmadd_ps(vs0, w0, acc0);
                    let w1 = _mm256_loadu_ps(w_ptr.add(out_len));
                    acc1 = _mm256_fmadd_ps(vs1, w1, acc1);
                    let w2 = _mm256_loadu_ps(w_ptr.add(2 * out_len));
                    acc2 = _mm256_fmadd_ps(vs2, w2, acc2);
                    let w3 = _mm256_loadu_ps(w_ptr.add(3 * out_len));
                    acc3 = _mm256_fmadd_ps(vs3, w3, acc3);
                    let w4 = _mm256_loadu_ps(w_ptr.add(4 * out_len));
                    acc4 = _mm256_fmadd_ps(vs4, w4, acc4);
                    let w5 = _mm256_loadu_ps(w_ptr.add(5 * out_len));
                    acc5 = _mm256_fmadd_ps(vs5, w5, acc5);
                    let w6 = _mm256_loadu_ps(w_ptr.add(6 * out_len));
                    acc6 = _mm256_fmadd_ps(vs6, w6, acc6);
                    let w7 = _mm256_loadu_ps(w_ptr.add(7 * out_len));
                    acc7 = _mm256_fmadd_ps(vs7, w7, acc7);

                    in_c += 8;
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                acc2 = _mm256_add_ps(acc2, acc3);
                acc4 = _mm256_add_ps(acc4, acc5);
                acc6 = _mm256_add_ps(acc6, acc7);
                acc0 = _mm256_add_ps(acc0, acc2);
                acc4 = _mm256_add_ps(acc4, acc6);
                acc0 = _mm256_add_ps(acc0, acc4);
                let mut tail = [0.0f32; 8];
                while in_c < in_len {
                    let inp = *in_frames.get_unchecked(f * in_len + in_c);
                    let base_idx = in_c * out_len + out_c;
                    for (j, t) in tail.iter_mut().enumerate() {
                        *t = f32::mul_add(inp, *weights.get_unchecked(base_idx + j), *t);
                    }
                    in_c += 1;
                }
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(tail.as_ptr()));
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                f += 1;
            }
            out_c += 8;
        }
        // Scalar tail for remaining out_c
        for n in 0..num_frames {
            for oc in out_c..out_len {
                let mut sum = *bias.get_unchecked(oc);
                for in_c in 0..in_len {
                    sum += *in_frames.get_unchecked(n * in_len + in_c)
                        * *weights.get_unchecked(in_c * out_len + oc);
                }
                *out_frames.get_unchecked_mut(n * out_len + oc) = sum;
            }
        }
        return;
    }

    // Generic fallback: call the scalar implementation
    crate::math::common::scalar_ref::gemv_with_bias_f32_fallback(
        in_frames, weights, bias, out_frames, num_frames,
    );
}

/// Batch GEMV overwrite without bias using native f32 weights via AVX2.
///
/// Strategy is shape-dependent:
/// - OUT ≤ 4: batch across `num_frames` — 8 frames per YMM accumulator,
///   broadcast one weight per `in_c` iteration.
/// - OUT ≥ 8: vectorize across output channels (same as `gemv_overwrite_avx2`
///   but with f32 weights loaded directly).
/// - Otherwise: scalar fallback.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_no_bias_f32_avx2(
    in_frames: &[f32],
    weights: &[f32],
    out_frames: &mut [f32],
    num_frames: usize,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    if out_len == 1 {
        for n in 0..num_frames {
            let mut acc = _mm256_setzero_ps();
            let mut in_c = 0;
            while in_c + 8 <= in_len {
                let v_in = _mm256_loadu_ps(in_frames.as_ptr().add(n * in_len + in_c));
                let v_w = _mm256_loadu_ps(weights.as_ptr().add(in_c));
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
                in_c += 8;
            }
            if in_c < in_len {
                let mut buf_in = [0.0f32; 8];
                let mut buf_w = [0.0f32; 8];
                let rem = in_len - in_c;
                for i in 0..rem {
                    buf_in[i] = *in_frames.get_unchecked(n * in_len + in_c + i);
                    buf_w[i] = *weights.get_unchecked(in_c + i);
                }
                let v_in = _mm256_loadu_ps(buf_in.as_ptr());
                let v_w = _mm256_loadu_ps(buf_w.as_ptr());
                acc = _mm256_fmadd_ps(v_in, v_w, acc);
            }
            // Horizontal sum: store to stack and scalar-reduce
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let sum: f32 = tmp.iter().sum();
            *out_frames.get_unchecked_mut(n) = sum;
        }
        return;
    }

    if out_len <= 4 {
        crate::math::common::scalar_ref::gemv_no_bias_f32_fallback(
            in_frames, weights, out_frames, num_frames,
        );
        return;
    }

    if out_len >= 8 {
        let mut out_c = 0;
        while out_c + 8 <= out_len {
            let mut f = 0;
            while f < num_frames {
                let mut acc0 = _mm256_setzero_ps();
                let mut acc1 = _mm256_setzero_ps();
                let mut acc2 = _mm256_setzero_ps();
                let mut acc3 = _mm256_setzero_ps();
                let mut acc4 = _mm256_setzero_ps();
                let mut acc5 = _mm256_setzero_ps();
                let mut acc6 = _mm256_setzero_ps();
                let mut acc7 = _mm256_setzero_ps();
                let mut in_c = 0;
                while in_c + 8 <= in_len {
                    let vs0 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c));
                    let vs1 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 1));
                    let vs2 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 2));
                    let vs3 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 3));
                    let vs4 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 4));
                    let vs5 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 5));
                    let vs6 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 6));
                    let vs7 = _mm256_set1_ps(*in_frames.get_unchecked(f * in_len + in_c + 7));

                    let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
                    let w0 = _mm256_loadu_ps(w_ptr);
                    acc0 = _mm256_fmadd_ps(vs0, w0, acc0);
                    let w1 = _mm256_loadu_ps(w_ptr.add(out_len));
                    acc1 = _mm256_fmadd_ps(vs1, w1, acc1);
                    let w2 = _mm256_loadu_ps(w_ptr.add(2 * out_len));
                    acc2 = _mm256_fmadd_ps(vs2, w2, acc2);
                    let w3 = _mm256_loadu_ps(w_ptr.add(3 * out_len));
                    acc3 = _mm256_fmadd_ps(vs3, w3, acc3);
                    let w4 = _mm256_loadu_ps(w_ptr.add(4 * out_len));
                    acc4 = _mm256_fmadd_ps(vs4, w4, acc4);
                    let w5 = _mm256_loadu_ps(w_ptr.add(5 * out_len));
                    acc5 = _mm256_fmadd_ps(vs5, w5, acc5);
                    let w6 = _mm256_loadu_ps(w_ptr.add(6 * out_len));
                    acc6 = _mm256_fmadd_ps(vs6, w6, acc6);
                    let w7 = _mm256_loadu_ps(w_ptr.add(7 * out_len));
                    acc7 = _mm256_fmadd_ps(vs7, w7, acc7);

                    in_c += 8;
                }
                acc0 = _mm256_add_ps(acc0, acc1);
                acc2 = _mm256_add_ps(acc2, acc3);
                acc4 = _mm256_add_ps(acc4, acc5);
                acc6 = _mm256_add_ps(acc6, acc7);
                acc0 = _mm256_add_ps(acc0, acc2);
                acc4 = _mm256_add_ps(acc4, acc6);
                acc0 = _mm256_add_ps(acc0, acc4);
                let mut tail = [0.0f32; 8];
                while in_c < in_len {
                    let inp = *in_frames.get_unchecked(f * in_len + in_c);
                    let base_idx = in_c * out_len + out_c;
                    for (j, t) in tail.iter_mut().enumerate() {
                        *t = f32::mul_add(inp, *weights.get_unchecked(base_idx + j), *t);
                    }
                    in_c += 1;
                }
                acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(tail.as_ptr()));
                _mm256_storeu_ps(out_frames.as_mut_ptr().add(f * out_len + out_c), acc0);
                f += 1;
            }
            out_c += 8;
        }
        // Scalar tail for remaining out_c
        for n in 0..num_frames {
            for oc in out_c..out_len {
                let mut sum = 0.0;
                for in_c in 0..in_len {
                    sum += *in_frames.get_unchecked(n * in_len + in_c)
                        * *weights.get_unchecked(in_c * out_len + oc);
                }
                *out_frames.get_unchecked_mut(n * out_len + oc) = sum;
            }
        }
        return;
    }

    // Generic fallback: call the scalar implementation
    crate::math::common::scalar_ref::gemv_no_bias_f32_fallback(
        in_frames, weights, out_frames, num_frames,
    );
}
