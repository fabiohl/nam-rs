// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Accumulation and activation kernels for WaveNet — AVX2, AVX-512 and scalar fallback.
//!
//! Extracted from `simd/avx2.rs`, `simd/avx512.rs` and `common/scalar_ref.rs`
//! during Task 3.4.

use core::arch::x86_64::*;

/// Accumulates src into dest using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn accumulate_head_avx2(dest: &mut [f32], src: &[f32]) {
    let len = dest.len();
    let mut i = 0;
    while i + 8 <= len {
        let vs = _mm256_loadu_ps(src.as_ptr().add(i));
        let vd = _mm256_loadu_ps(dest.as_ptr().add(i));
        _mm256_storeu_ps(dest.as_mut_ptr().add(i), _mm256_add_ps(vd, vs));
        i += 8;
    }
    while i < len {
        dest[i] += src[i];
        i += 1;
    }
}

/// Applies tanh in-place on block and accumulates into head_input using AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_accumulate_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);

        let vh = _mm256_loadu_ps(head_input.as_ptr().add(i));
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), _mm256_add_ps(vh, vt));
        i += 8;
    }
    while i < len {
        let val = block[i].tanh();
        block[i] = val;
        head_input[i] += val;
        i += 1;
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and accumulates into head_input using AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gated_activation_and_accumulate_block_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 8 <= ch {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_avx2(z1, z2);
            let activated = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);

            let vh = _mm256_loadu_ps(head_input.as_ptr().add(head_offset + c));
            _mm256_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm256_add_ps(vh, activated),
            );
            c += 8;
        }
        while c < ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] += activated;
            c += 1;
        }
    }
}

/// Applies the "gate" activation (tanh * sigmoid) to audio blocks.
/// Imagine each sound goes through two filters: one that shapes the timbre (tanh)
/// and another that controls the intensity (sigmoid). The result is added to "head_input".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gated_activation_and_accumulate_block_avx512(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        // Process 16 channels at a time.
        while c + 16 <= ch {
            let z1 = _mm512_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm512_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            // Apply the complex mathematical functions in a blazing-fast manner.
            let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_avx512(z1, z2);
            let activated = _mm512_mul_ps(tanh_z1, sig_z2);

            _mm512_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);

            let vh = _mm512_loadu_ps(head_input.as_ptr().add(head_offset + c));
            _mm512_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                _mm512_add_ps(vh, activated),
            );
            c += 16;
        }
        // Any leftover channels? Handle them one by one.
        while c < ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] += activated;
            c += 1;
        }
    }
}

/// Accumulation of the network "Head".
/// In WaveNet-like architectures, the various layers (blocks) of the network contribute
/// to a common result called "head". This function simply sums those contributions.
pub unsafe fn accumulate_head_fallback(dest: &mut [f32], src: &[f32]) {
    let len = core::cmp::min(dest.len(), src.len());
    for i in 0..len {
        unsafe {
            // Sum the contents of 'src' into 'dest'.
            *dest.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
    }
}

/// Applies the 'Tanh' activation and accumulates into the main output.
/// Tanh (Hyperbolic Tangent) is a function that "squashes" any number to
/// be between -1.0 and 1.0. It is very common in guitar amplifier modeling.
pub unsafe fn tanh_and_accumulate_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh(); // Apply the "squashing".
        block[i] = activated; // Save the squashed value in the block.
        head_input[i] += activated; // Add the same value to the "head" accumulator.
    }
}

/// Gated Activation + Accumulation.
/// This technique uses two signals (z1 and z2):
/// 1. z1 goes through a Tanh (holds the "information").
/// 2. z2 goes through a Sigmoid (acts as a "volume" or "gate" for z1).
///
/// At the end, we multiply the two. It's as if z2 decides how much of z1 will pass through.
pub unsafe fn gated_activation_and_accumulate_block_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize, // Number of channels.
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        for c in 0..ch {
            // The block contains z1 and z2 side by side.
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];

            // Complex activation: tanh(z1) * sigmoid(z2).
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));

            block[block_offset + c] = activated;
            head_input[head_offset + c] += activated;
        }
    }
}

/// Applies tanh in-place on block and overwrites head_input using AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_and_overwrite_block_avx2(head_input: &mut [f32], block: &mut [f32]) {
    let len = block.len();
    let mut i = 0;
    while i + 8 <= len {
        let vb = _mm256_loadu_ps(block.as_ptr().add(i));
        let vt = crate::math::activations::simd_tanh_avx2(vb);
        _mm256_storeu_ps(block.as_mut_ptr().add(i), vt);
        _mm256_storeu_ps(head_input.as_mut_ptr().add(i), vt);
        i += 8;
    }
    while i < len {
        let val = block[i].tanh();
        block[i] = val;
        head_input[i] = val;
        i += 1;
    }
}

/// Applies gated activation (tanh * sigmoid) in-place on block and overwrites head_input using AVX2.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gated_activation_and_overwrite_block_avx2(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 8 <= ch {
            let z1 = _mm256_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm256_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_avx2(z1, z2);
            let activated = _mm256_mul_ps(tanh_z1, sig_z2);

            _mm256_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);
            _mm256_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                activated,
            );
            c += 8;
        }
        while c < ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] = activated;
            c += 1;
        }
    }
}

/// Applies the "gate" activation (tanh * sigmoid) to audio blocks.
/// Overwrites to "head_input".
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn gated_activation_and_overwrite_block_avx512(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        let mut c = 0;
        while c + 16 <= ch {
            let z1 = _mm512_loadu_ps(block.as_ptr().add(block_offset + c));
            let z2 = _mm512_loadu_ps(block.as_ptr().add(block_offset + ch + c));

            let (tanh_z1, sig_z2) = crate::math::activations::simd_tanh_sigmoid_dual_avx512(z1, z2);
            let activated = _mm512_mul_ps(tanh_z1, sig_z2);

            _mm512_storeu_ps(block.as_mut_ptr().add(block_offset + c), activated);
            _mm512_storeu_ps(
                head_input.as_mut_ptr().add(head_offset + c),
                activated,
            );
            c += 16;
        }
        while c < ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] = activated;
            c += 1;
        }
    }
}

/// Applies the 'Tanh' activation and overwrites the main output.
pub unsafe fn tanh_and_overwrite_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh();
        block[i] = activated;
        head_input[i] = activated;
    }
}

/// Gated Activation + Overwrite.
pub unsafe fn gated_activation_and_overwrite_block_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize,
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        for c in 0..ch {
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));
            block[block_offset + c] = activated;
            head_input[head_offset + c] = activated;
        }
    }
}

