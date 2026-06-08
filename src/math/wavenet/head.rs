// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! WaveNet Head Sum (Batch) kernels — AVX2, AVX-512 and dynamic dispatch.
//!
//! Depends on `common::utility::horizontal_sum_avx2` and `common::utility::horizontal_sum_avx512`
//! for the horizontal sum operations.

/// Batch sum of WaveNet Head projections using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn batch_wavenet_head_sum_avx2<const HEAD: usize>(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let num_frames = output.len();
    for i in 0..num_frames {
        let ptr = head1.as_ptr().add(i * HEAD);
        // Sum the internal channels of each frame.
        let sum = crate::math::common::utility::horizontal_sum_avx2(ptr, HEAD);
        // Add the residual input and apply the volume (scale).
        *output.get_unchecked_mut(i) = (sum + *head2.get_unchecked(i)) * scale;
    }
}

/// Dynamic dispatch for `batch_wavenet_head_sum` via AVX2.
/// Dispatches to the appropriate const generic (1 or 16) or uses AVX2 fallback.
#[target_feature(enable = "avx2")]
pub unsafe fn batch_wavenet_head_sum_dyn_avx2(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    head: usize,
    scale: f32,
) {
    match head {
        1 => batch_wavenet_head_sum_avx2::<1>(head1, head2, output, scale),
        16 => batch_wavenet_head_sum_avx2::<16>(head1, head2, output, scale),
        _ => {
            let num_frames = output.len();
            for i in 0..num_frames {
                let h1 = crate::math::common::utility::horizontal_sum_avx2(
                    head1.as_ptr().add(i * head),
                    head,
                );
                output[i] = (h1 + head2[i]) * scale;
            }
        }
    }
}

/// Specialized kernel for WaveNet Head sum using AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_wavenet_head_sum_avx512<const HEAD: usize>(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    let num_frames = output.len();
    for i in 0..num_frames {
        let h1 =
            crate::math::common::utility::horizontal_sum_avx512(head1.as_ptr().add(i * HEAD), HEAD);
        *output.get_unchecked_mut(i) = (h1 + *head2.get_unchecked(i)) * scale;
    }
}

/// Dynamic dispatch for `batch_wavenet_head_sum` via AVX-512.
/// Dispatches to the appropriate const generic (1 or 16) or uses AVX-512 fallback.
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_wavenet_head_sum_dyn_avx512(
    head1: &[f32],
    head2: &[f32],
    output: &mut [f32],
    head: usize,
    scale: f32,
) {
    match head {
        1 => batch_wavenet_head_sum_avx512::<1>(head1, head2, output, scale),
        16 => batch_wavenet_head_sum_avx512::<16>(head1, head2, output, scale),
        _ => {
            let num_frames = output.len();
            for i in 0..num_frames {
                let h1 = crate::math::common::utility::horizontal_sum_avx512(
                    head1.as_ptr().add(i * head),
                    head,
                );
                output[i] = (h1 + head2[i]) * scale;
            }
        }
    }
}
