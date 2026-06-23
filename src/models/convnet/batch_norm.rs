// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 1-Dimensional Batch Normalization layer for Inference-Only DSP.
//!
//! During inference, batch normalization is an affine transformation
//! `y = x * scale + offset` where the fused parameters are:
//!
//! ```text
//! scale  = gamma / sqrt(running_var + eps)
//! offset = beta - gamma * running_mean / sqrt(running_var + eps)
//! ```
//!
//! These are precomputed at construction time so the hot-path is a single
//! FMA per element (lowered to `vfmadd231ps` on x86-64-v3).

use crate::math::common::AlignedVec;

/// 1-Dimensional Batch Normalization layer (inference-only).
///
/// Stores per-channel pre-fused affine parameters for the hot-path:
/// `output[ch + f * n_ch] = input[ch + f * n_ch] * scale[ch] + offset[ch]`.
///
/// Input/output layout is frame-interleaved:
/// `[f0_c0, f0_c1, ..., f0_c{n-1}, f1_c0, ...]`.
#[derive(Clone)]
#[repr(align(64))]
pub struct BatchNorm1D {
    /// Number of channels.
    pub num_channels: usize,
    /// Fused multiplicative factor per channel.
    pub scale: AlignedVec<f32>,
    /// Fused additive factor per channel.
    pub offset: AlignedVec<f32>,
}

impl BatchNorm1D {
    /// Constructs a `BatchNorm1D` from the raw training parameters,
    /// fusing them into a single affine transform for inference.
    ///
    /// # Parameters
    /// - `gamma`: learnable scale (length `num_channels`).
    /// - `beta`: learnable bias (length `num_channels`).
    /// - `running_mean`: EMA of activation means (length `num_channels`).
    /// - `running_var`: EMA of activation variances (length `num_channels`).
    /// - `eps`: small constant for numerical stability (e.g. 1e-5).
    ///
    /// # Panics
    /// Panics if any variance is negative.
    pub fn from_params(
        num_channels: usize,
        gamma: &[f32],
        beta: &[f32],
        running_mean: &[f32],
        running_var: &[f32],
        eps: f32,
    ) -> Self {
        assert_eq!(gamma.len(), num_channels);
        assert_eq!(beta.len(), num_channels);
        assert_eq!(running_mean.len(), num_channels);
        assert_eq!(running_var.len(), num_channels);

        let mut scale = AlignedVec::new(num_channels, 0.0f32);
        let mut offset = AlignedVec::new(num_channels, 0.0f32);

        for c in 0..num_channels {
            let var = running_var[c];
            assert!(var >= 0.0, "running_var[{c}] = {var} is negative");
            let inv_std = 1.0 / (var + eps).sqrt();
            scale[c] = gamma[c] * inv_std;
            offset[c] = beta[c] - running_mean[c] * scale[c];
        }

        Self {
            num_channels,
            scale,
            offset,
        }
    }

    /// Constructs a `BatchNorm1D` from already-fused parameters.
    ///
    /// Useful when the fused scale/offset are provided directly by the
    /// model loader (e.g. from a `.namb` blob prepared by the training script).
    pub fn from_fused(num_channels: usize, scale: &[f32], offset: &[f32]) -> Self {
        assert_eq!(scale.len(), num_channels);
        assert_eq!(offset.len(), num_channels);

        let mut s = AlignedVec::new(num_channels, 0.0f32);
        let mut o = AlignedVec::new(num_channels, 0.0f32);
        s.copy_from_slice(scale);
        o.copy_from_slice(offset);

        Self {
            num_channels,
            scale: s,
            offset: o,
        }
    }

    /// Applies the batch normalization affine transform in-place.
    ///
    /// `data` layout: `[f0_c0, f0_c1, ..., f0_c{n_ch-1}, f1_c0, ...]`.
    /// `data.len()` must be `num_frames * num_channels`.
    ///
    /// # Safety
    /// The caller must ensure `data.len() == num_frames * num_channels`.
    #[inline(always)]
    pub unsafe fn process(&self, data: &mut [f32], num_frames: usize) {
        debug_assert_eq!(data.len(), num_frames * self.num_channels);
        unsafe {
            process_avx2(
                data,
                &self.scale,
                &self.offset,
                self.num_channels,
                num_frames,
            );
        }
    }

    /// Scalar reference implementation — always available, used for testing parity.
    ///
    /// # Safety
    /// `data.len()` must equal `num_frames * self.num_channels`.
    #[inline(always)]
    pub unsafe fn process_scalar(&self, data: &mut [f32], num_frames: usize) {
        unsafe {
            process_scalar_ref(
                data,
                &self.scale,
                &self.offset,
                self.num_channels,
                num_frames,
            );
        }
    }
}

/// AVX2+FMA kernel: 8 frames per channel, strided by `n_ch`.
///
/// For each channel, broadcasts `scale[c]` and `offset[c]` into SIMD
/// registers and processes frames in chunks of 8 using `vfmadd231ps`.
///
/// # Safety
/// `data.len()` must equal `num_frames * n_ch`. `scale` and `offset` must
/// each have at least `n_ch` elements.
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn process_avx2(
    data: &mut [f32],
    scale: &[f32],
    offset: &[f32],
    n_ch: usize,
    num_frames: usize,
) {
    use core::arch::x86_64::*;

    let mut gather_buf = [0.0f32; 8];

    for ch in 0..n_ch {
        let s = unsafe { _mm256_set1_ps(*scale.get_unchecked(ch)) };
        let o = unsafe { _mm256_set1_ps(*offset.get_unchecked(ch)) };

        let mut f = 0usize;
        while f + 8 <= num_frames {
            let base = ch + f * n_ch;
            unsafe {
                for lane in 0..8 {
                    *gather_buf.get_unchecked_mut(lane) = *data.get_unchecked(base + lane * n_ch);
                }
                let xv = _mm256_loadu_ps(gather_buf.as_ptr());
                let yv = _mm256_fmadd_ps(xv, s, o);
                _mm256_storeu_ps(gather_buf.as_mut_ptr(), yv);
                for lane in 0..8 {
                    *data.get_unchecked_mut(base + lane * n_ch) = *gather_buf.get_unchecked(lane);
                }
            }
            f += 8;
        }

        for f in f..num_frames {
            let idx = ch + f * n_ch;
            unsafe {
                *data.get_unchecked_mut(idx) = (*data.get_unchecked(idx))
                    .mul_add(*scale.get_unchecked(ch), *offset.get_unchecked(ch));
            }
        }
    }
}

/// Scalar reference: plain `mul_add` loop, channel-major across frames.
///
/// # Safety
/// `data.len()` must equal `num_frames * n_ch`.
#[inline(always)]
unsafe fn process_scalar_ref(
    data: &mut [f32],
    scale: &[f32],
    offset: &[f32],
    n_ch: usize,
    num_frames: usize,
) {
    for ch in 0..n_ch {
        let s = unsafe { *scale.get_unchecked(ch) };
        let o = unsafe { *offset.get_unchecked(ch) };
        for f in 0..num_frames {
            let idx = ch + f * n_ch;
            unsafe {
                *data.get_unchecked_mut(idx) = (*data.get_unchecked(idx)).mul_add(s, o);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bn_simple() -> BatchNorm1D {
        // gamma = [1.0, 2.0], beta = [0.0, 0.0]
        // running_mean = [0.0, 0.0], running_var = [4.0, 4.0], eps = 0.0
        // scale = [1/2=0.5, 2/2=1.0], offset = [0, 0]
        BatchNorm1D::from_params(2, &[1.0, 2.0], &[0.0, 0.0], &[0.0, 0.0], &[4.0, 4.0], 0.0)
    }

    #[test]
    fn test_fused_params_simple() {
        let bn = make_bn_simple();
        assert!((bn.scale[0] - 0.5).abs() < 1e-7);
        assert!((bn.scale[1] - 1.0).abs() < 1e-7);
        assert!((bn.offset[0] - 0.0).abs() < 1e-7);
        assert!((bn.offset[1] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_fused_params_with_beta_mean() {
        // gamma=[1,1], beta=[2,3], mean=[1,2], var=[1,1], eps=0
        // scale = [1/1=1, 1/1=1]
        // offset[0] = 2 - 1*1 = 1
        // offset[1] = 3 - 2*1 = 1
        let bn =
            BatchNorm1D::from_params(2, &[1.0, 1.0], &[2.0, 3.0], &[1.0, 2.0], &[1.0, 1.0], 0.0);
        assert!((bn.scale[0] - 1.0).abs() < 1e-7);
        assert!((bn.scale[1] - 1.0).abs() < 1e-7);
        assert!((bn.offset[0] - 1.0).abs() < 1e-7);
        assert!((bn.offset[1] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_epsilon_effect() {
        // gamma=[1,1], beta=[0,0], mean=[0,0], var=[0,0], eps=1e-2
        // inv_std = 1/sqrt(0.01) = 1/0.1 = 10
        // scale = 10, offset = 0
        let bn = BatchNorm1D::from_params(1, &[1.0], &[0.0], &[0.0], &[0.0], 1e-2);
        assert!((bn.scale[0] - 10.0).abs() < 1e-7);
        assert!((bn.offset[0] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_process_scalar_identity_bn() {
        // scale=1, offset=0 -> output == input
        let bn = BatchNorm1D::from_fused(3, &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0]);

        let mut data = vec![
            1.0, 2.0, 3.0, // frame 0
            4.0, 5.0, 6.0, // frame 1
            7.0, 8.0, 9.0, // frame 2
        ];

        unsafe {
            bn.process_scalar(&mut data, 3);
        }

        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
        }
    }

    #[test]
    fn test_process_scalar_simple() {
        let bn = make_bn_simple(); // scale=[0.5, 1.0], offset=[0,0]

        let mut data = vec![
            2.0, 3.0, // frame 0
            4.0, 5.0, // frame 1
            6.0, 7.0, // frame 2
        ];

        unsafe {
            bn.process_scalar(&mut data, 3);
        }

        // ch0 * 0.5: 2*0.5=1, 4*0.5=2, 6*0.5=3
        // ch1 * 1.0: 3*1.0=3, 5*1.0=5, 7*1.0=7
        let expected = [1.0, 3.0, 2.0, 5.0, 3.0, 7.0];
        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
        }
    }

    #[test]
    fn test_process_scalar_with_offset() {
        // scale=[2, 0.5], offset=[1, -2]
        let bn = BatchNorm1D::from_fused(2, &[2.0, 0.5], &[1.0, -2.0]);

        let mut data = vec![
            0.0, 4.0, // frame 0
            2.0, 8.0, // frame 1
        ];

        unsafe {
            bn.process_scalar(&mut data, 2);
        }

        // ch0: 0*2+1=1, 2*2+1=5
        // ch1: 4*0.5-2=0, 8*0.5-2=2
        let expected = [1.0, 0.0, 5.0, 2.0];
        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
        }
    }

    #[test]
    fn test_single_channel() {
        let bn = BatchNorm1D::from_fused(1, &[3.0], &[-1.0]);

        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        unsafe {
            bn.process_scalar(&mut data, 5);
        }

        // 1*3-1=2, 2*3-1=5, 3*3-1=8, 4*3-1=11, 5*3-1=14
        let expected = [2.0, 5.0, 8.0, 11.0, 14.0];
        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-7, "mismatch at index {i}: {a} != {b}");
        }
    }

    #[test]
    fn test_single_frame_many_channels() {
        let bn = BatchNorm1D::from_fused(10, &[1.0; 10], &[0.0; 10]);

        let mut data: Vec<f32> = (0..10).map(|i| i as f32).collect();

        unsafe {
            bn.process_scalar(&mut data, 1);
        }

        for (i, &val) in data.iter().enumerate() {
            assert!((val - i as f32).abs() < 1e-7);
        }
    }

    #[test]
    fn test_from_fused_roundtrip() {
        let bn1 = BatchNorm1D::from_params(
            3,
            &[2.0, 1.0, 0.5],
            &[0.0, 1.0, -1.0],
            &[0.1, 0.2, 0.3],
            &[1.0, 4.0, 0.25],
            1e-5,
        );

        let bn2 = BatchNorm1D::from_fused(3, &bn1.scale, &bn1.offset);

        for c in 0..3 {
            assert!((bn1.scale[c] - bn2.scale[c]).abs() < 1e-7);
            assert!((bn1.offset[c] - bn2.offset[c]).abs() < 1e-7);
        }
    }

    /// Validates that `from_fused` constructor preserves exact values.
    #[test]
    fn test_from_fused_exact() {
        let bn = BatchNorm1D::from_fused(2, &[1.0f32, 2.0], &[3.0f32, 4.0]);
        assert_eq!(bn.num_channels, 2);
        assert!((bn.scale[0] - 1.0).abs() < f32::EPSILON);
        assert!((bn.scale[1] - 2.0).abs() < f32::EPSILON);
        assert!((bn.offset[0] - 3.0).abs() < f32::EPSILON);
        assert!((bn.offset[1] - 4.0).abs() < f32::EPSILON);
    }

    /// Validates that `process_scalar` and `process` produce identical output.
    #[test]
    fn test_scalar_simd_parity() {
        let bn = make_bn_simple();

        let original = vec![
            1.0, -2.0, // frame 0
            3.0, 4.0, // frame 1
            -5.0, 6.0, // frame 2
            7.0, -8.0, // frame 3
            0.5, -0.3, // frame 4
            2.5, 1.5, // frame 5
            -1.0, 3.0, // frame 6
            4.5, -2.5, // frame 7
            9.0, 0.0, // frame 8
            -3.0, 5.0, // frame 9
        ];

        let mut data_simd = original.clone();
        unsafe {
            bn.process(&mut data_simd, 10);
        }

        let mut data_scalar = original;
        unsafe {
            bn.process_scalar(&mut data_scalar, 10);
        }

        for (i, (&a, &b)) in data_simd.iter().zip(data_scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "SIMD vs scalar mismatch at index {i}: {a} != {b}"
            );
        }
    }

    /// Validates across a range of frame counts (including > 8 for AVX2 tail).
    #[test]
    fn test_scalar_simd_parity_various_frame_counts() {
        let bn = BatchNorm1D::from_params(
            4,
            &[0.8, 1.2, 0.5, 2.0],
            &[0.1, -0.1, 0.0, 0.3],
            &[0.0, 0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0, 1.0],
            1e-5,
        );

        for n_frames in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 64, 100] {
            let mut original = Vec::with_capacity(n_frames * 4);
            for f in 0..n_frames {
                original.push((f as f32) * 0.1);
                original.push(-(f as f32) * 0.1);
                original.push(1.0);
                original.push(-1.0);
            }

            let mut data_simd = original.clone();
            unsafe {
                bn.process(&mut data_simd, n_frames);
            }

            let mut data_scalar = original;
            unsafe {
                bn.process_scalar(&mut data_scalar, n_frames);
            }

            for (i, (&a, &b)) in data_simd.iter().zip(data_scalar.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "SIMD vs scalar mismatch at n_frames={n_frames}, index {i}: {a} != {b}"
                );
            }
        }
    }

    /// Validates all public invariants when model parameters are zeroed.
    #[test]
    fn test_zero_gamma_noop() {
        // gamma=0 -> scale=0, offset=beta
        let bn =
            BatchNorm1D::from_params(2, &[0.0, 0.0], &[0.5, -0.5], &[1.0, 2.0], &[1.0, 1.0], 0.0);
        assert!((bn.scale[0] - 0.0).abs() < 1e-7);
        assert!((bn.scale[1] - 0.0).abs() < 1e-7);
        assert!((bn.offset[0] - 0.5).abs() < 1e-7);
        assert!((bn.offset[1] - (-0.5)).abs() < 1e-7);

        let mut data = vec![100.0, 200.0, 300.0, 400.0];
        unsafe {
            bn.process_scalar(&mut data, 2);
        }
        // output should just be offset
        let expected = [0.5, -0.5, 0.5, -0.5];
        for (i, (&a, &b)) in data.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "mismatch at index {i}");
        }
    }

    #[test]
    #[should_panic(expected = "negative")]
    fn test_negative_variance_panics() {
        BatchNorm1D::from_params(1, &[1.0], &[0.0], &[0.0], &[-1.0], 0.0);
    }

    #[test]
    fn test_clone_preserves_behavior() {
        let bn1 = BatchNorm1D::from_params(
            3,
            &[0.5, 1.0, 2.0],
            &[0.1, 0.2, 0.3],
            &[0.0, 1.0, 0.0],
            &[4.0, 1.0, 1.0],
            0.0,
        );
        let bn2 = bn1.clone();

        let mut data1 = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let mut data2 = data1.clone();

        unsafe {
            bn1.process_scalar(&mut data1, 2);
            bn2.process_scalar(&mut data2, 2);
        }

        assert_eq!(data1, data2);
    }

    /// Verifies that `BatchNorm1D` uses 64-byte alignment.
    #[test]
    fn test_struct_alignment() {
        assert_eq!(std::mem::align_of::<BatchNorm1D>(), 64);
    }
}
