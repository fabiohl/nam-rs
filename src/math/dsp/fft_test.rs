// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Tolerance for floating-point comparisons.
const EPS_F32: f32 = 1e-5;
const EPS_F64: f64 = 1e-9;

// -----------------------------------------------------------------
// helpers
// -----------------------------------------------------------------

fn assert_slice_approx_eq_f32(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < eps,
            "mismatch at index {i}: {x} vs {y} (diff {})",
            (x - y).abs()
        );
    }
}

fn assert_slice_approx_eq_f64(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < eps,
            "mismatch at index {i}: {x} vs {y} (diff {})",
            (x - y).abs()
        );
    }
}

// -----------------------------------------------------------------
// construction
// -----------------------------------------------------------------

#[test]
#[should_panic(expected = "power of two")]
fn rejects_non_power_of_two() {
    FftPlanner::<f32>::new(3);
}

#[test]
#[should_panic(expected = "power of two")]
fn rejects_non_power_of_two_f64() {
    FftPlanner::<f64>::new(6);
}

#[test]
#[should_panic(expected = "positive")]
fn rejects_zero() {
    FftPlanner::<f32>::new(0);
}

#[test]
fn accepts_powers_of_two() {
    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let fft = FftPlanner::<f64>::new(n);
        assert_eq!(fft.len(), n);
        assert_eq!(fft.bit_reverse.len(), n);
        assert_eq!(fft.twiddle_re.len(), n / 2);
        assert_eq!(fft.twiddle_im.len(), n / 2);
    }
}

// -----------------------------------------------------------------
// impulse → DC
// -----------------------------------------------------------------

#[test]
fn impulse_dc_f32() {
    for n in [1, 2, 4, 8, 16, 32, 64] {
        let fft = FftPlanner::<f32>::new(n);
        let mut re = vec![0.0f32; n];
        let mut im = vec![0.0f32; n];
        re[0] = 1.0;
        fft.process(&mut re, &mut im);
        let expected = vec![1.0f32; n];
        assert_slice_approx_eq_f32(&re, &expected, EPS_F32);
        let expected_im = vec![0.0f32; n];
        assert_slice_approx_eq_f32(&im, &expected_im, EPS_F32);
    }
}

#[test]
fn impulse_dc_f64() {
    let fft = FftPlanner::<f64>::new(8);
    let mut re = vec![0.0f64; 8];
    let mut im = vec![0.0f64; 8];
    re[0] = 1.0;
    fft.process(&mut re, &mut im);
    for (i, v) in re.iter().enumerate() {
        assert!((v - 1.0).abs() < EPS_F64, "re[{i}] = {v}");
    }
    for (i, v) in im.iter().enumerate() {
        assert!(v.abs() < EPS_F64, "im[{i}] = {v}");
    }
}

// -----------------------------------------------------------------
// round-trip: forward → inverse ≡ identity
// -----------------------------------------------------------------

#[test]
fn roundtrip_f32() {
    for n in [1, 2, 4, 8, 16, 32, 64, 128] {
        let fft = FftPlanner::<f32>::new(n);
        let original_re: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let original_im: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();

        let mut re = original_re.clone();
        let mut im = original_im.clone();

        fft.process(&mut re, &mut im);
        fft.process_inverse(&mut re, &mut im);

        assert_slice_approx_eq_f32(&re, &original_re, 1e-3f32 * n as f32);
        assert_slice_approx_eq_f32(&im, &original_im, 1e-3f32 * n as f32);
    }
}

#[test]
fn roundtrip_f64() {
    for n in [1, 2, 4, 8, 16, 32, 64, 128] {
        let fft = FftPlanner::<f64>::new(n);
        let original_re: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let original_im: Vec<f64> = (0..n).map(|i| -(i as f64)).collect();

        let mut re = original_re.clone();
        let mut im = original_im.clone();

        fft.process(&mut re, &mut im);
        fft.process_inverse(&mut re, &mut im);

        assert_slice_approx_eq_f64(&re, &original_re, 1e-6f64 * n as f64);
        assert_slice_approx_eq_f64(&im, &original_im, 1e-6f64 * n as f64);
    }
}

// -----------------------------------------------------------------
// linearity: FFT(a + b) = FFT(a) + FFT(b)
// -----------------------------------------------------------------

#[test]
fn linearity_f64() {
    let n = 16;
    let fft = FftPlanner::<f64>::new(n);

    let a_re: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let a_im: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
    let b_re: Vec<f64> = (0..n).map(|i| (2.0 * i as f64).cos()).collect();
    let b_im: Vec<f64> = (0..n).map(|i| (3.0 * i as f64).sin()).collect();

    // FFT(a)
    let mut fa_re = a_re.clone();
    let mut fa_im = a_im.clone();
    fft.process(&mut fa_re, &mut fa_im);

    // FFT(b)
    let mut fb_re = b_re.clone();
    let mut fb_im = b_im.clone();
    fft.process(&mut fb_re, &mut fb_im);

    // FFT(a+b)
    let mut sum_re: Vec<f64> = a_re.iter().zip(&b_re).map(|(x, y)| x + y).collect();
    let mut sum_im: Vec<f64> = a_im.iter().zip(&b_im).map(|(x, y)| x + y).collect();
    fft.process(&mut sum_re, &mut sum_im);

    // Sum of individual FFTs
    for i in 0..n {
        let expected_re = fa_re[i] + fb_re[i];
        let expected_im = fa_im[i] + fb_im[i];
        assert!(
            (sum_re[i] - expected_re).abs() < EPS_F64,
            "linearity re[{i}]: {} vs {}",
            sum_re[i],
            expected_re
        );
        assert!(
            (sum_im[i] - expected_im).abs() < EPS_F64,
            "linearity im[{i}]: {} vs {}",
            sum_im[i],
            expected_im
        );
    }
}

// -----------------------------------------------------------------
// known values: N=4
// -----------------------------------------------------------------

#[test]
fn known_n4_f64() {
    let fft = FftPlanner::<f64>::new(4);
    let mut re = vec![1.0, 2.0, 3.0, 4.0];
    let mut im = vec![0.0; 4];
    fft.process(&mut re, &mut im);

    // FFT of [1,2,3,4]:
    // DC   = 1+2+3+4 = 10
    // bin1 = (1+2w+3w^2+4w^3) where w=e^{-2πi/4}=-i
    //       = 1 - 2i - 3 + 4i = -2 + 2i
    // bin2 = 1 - 2 + 3 - 4 = -2
    // bin3 = conjugate of bin1 = -2 - 2i
    assert!((re[0] - 10.0).abs() < EPS_F64);
    assert!((im[0] - 0.0).abs() < EPS_F64);
    assert!((re[1] - (-2.0)).abs() < EPS_F64);
    assert!((im[1] - 2.0).abs() < EPS_F64);
    assert!((re[2] - (-2.0)).abs() < EPS_F64);
    assert!((im[2] - 0.0).abs() < EPS_F64);
    assert!((re[3] - (-2.0)).abs() < EPS_F64);
    assert!((im[3] - (-2.0)).abs() < EPS_F64);
}

// -----------------------------------------------------------------
// known values: N=8 — single cosine
// -----------------------------------------------------------------

#[test]
fn cosine_n8_f64() {
    let n = 8;
    let fft = FftPlanner::<f64>::new(n);
    let freq = 2.0; // two cycles in N=8 → bin 2 and bin 6
    let mut re: Vec<f64> = (0..n)
        .map(|i| (TAU * freq * i as f64 / n as f64).cos())
        .collect();
    let mut im = vec![0.0f64; n];
    fft.process(&mut re, &mut im);

    // Cosine at bin 2: peaks at indices 2 and 6 with amplitude n/2 = 4
    assert!((re[2] - 4.0).abs() < EPS_F64, "re[2] = {}", re[2]);
    assert!((re[6] - 4.0).abs() < EPS_F64, "re[6] = {}", re[6]);
    // Other bins should be ~0
    for &i in &[0, 1, 3, 4, 5, 7] {
        assert!(re[i].abs() < 1e-9, "re[{i}] = {}", re[i]);
        assert!(im[i].abs() < 1e-9, "im[{i}] = {}", im[i]);
    }
}

// -----------------------------------------------------------------
// process panics on wrong buffer length
// -----------------------------------------------------------------

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_wrong_len() {
    let fft = FftPlanner::<f32>::new(8);
    let mut re = vec![0.0f32; 7];
    let mut im = vec![0.0f32; 8];
    fft.process(&mut re, &mut im);
}

// -----------------------------------------------------------------
// N=1 edge case
// -----------------------------------------------------------------

#[test]
fn n1_f64() {
    let fft = FftPlanner::<f64>::new(1);
    let mut re = vec![42.0];
    let mut im = vec![7.0];
    fft.process(&mut re, &mut im);
    assert!((re[0] - 42.0).abs() < EPS_F64);
    assert!((im[0] - 7.0).abs() < EPS_F64);
    fft.process_inverse(&mut re, &mut im);
    assert!((re[0] - 42.0).abs() < EPS_F64);
    assert!((im[0] - 7.0).abs() < EPS_F64);
}

// =================================================================
// RFFT tests
// =================================================================

// -----------------------------------------------------------------
// construction
// -----------------------------------------------------------------

#[test]
fn rfft_accepts_n2() {
    let _rfft = RfftPlanner::<f32>::new(2);
}

#[test]
#[should_panic(expected = "power of two")]
fn rfft_rejects_non_power_of_two_6() {
    RfftPlanner::<f64>::new(6);
}

#[test]
#[should_panic(expected = "positive")]
fn rfft_rejects_zero() {
    RfftPlanner::<f32>::new(0);
}

#[test]
fn rfft_accepts_powers_of_two() {
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let rfft = RfftPlanner::<f64>::new(n);
        assert_eq!(rfft.len(), n);
        assert_eq!(rfft.fft_n2.len(), n / 2);
        assert_eq!(rfft.post_twiddle_re.len(), n / 2 + 1);
        assert_eq!(rfft.post_twiddle_im.len(), n / 2 + 1);
        assert_eq!(rfft.scratch_re.len(), n / 2);
        assert_eq!(rfft.scratch_im.len(), n / 2);
    }
}

// -----------------------------------------------------------------
// impulse → DC (all bins = 1.0 in re, 0.0 in im)
// -----------------------------------------------------------------

#[test]
fn rfft_impulse_dc_f64() {
    for n in [2, 4, 8, 16, 32, 64] {
        let mut rfft = RfftPlanner::<f64>::new(n);
        let input = {
            let mut v = vec![0.0f64; n];
            v[0] = 1.0;
            v
        };
        let n_out = n / 2 + 1;
        let mut out_re = vec![0.0f64; n_out];
        let mut out_im = vec![0.0f64; n_out];
        rfft.process_forward(&input, &mut out_re, &mut out_im);

        for (i, v) in out_re.iter().enumerate() {
            assert!((v - 1.0).abs() < EPS_F64, "n={n} re[{i}] = {v}");
        }
        for (i, v) in out_im.iter().enumerate() {
            assert!(v.abs() < EPS_F64, "n={n} im[{i}] = {v}");
        }
    }
}

// -----------------------------------------------------------------
// known values: N=4 matches full complex FFT
// -----------------------------------------------------------------

#[test]
fn rfft_known_n4_f64() {
    let mut rfft = RfftPlanner::<f64>::new(4);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut out_re = vec![0.0f64; 3];
    let mut out_im = vec![0.0f64; 3];
    rfft.process_forward(&input, &mut out_re, &mut out_im);

    // From full complex FFT of [1,2,3,4]:
    // bin0 = 10, bin1 = -2 + 2i, bin2 = -2
    assert!((out_re[0] - 10.0).abs() < EPS_F64);
    assert!((out_im[0] - 0.0).abs() < EPS_F64);
    assert!((out_re[1] - (-2.0)).abs() < EPS_F64);
    assert!((out_im[1] - 2.0).abs() < EPS_F64);
    assert!((out_re[2] - (-2.0)).abs() < EPS_F64);
    assert!((out_im[2] - 0.0).abs() < EPS_F64);
}

// -----------------------------------------------------------------
// known values: N=8 — single cosine matches full complex FFT
// -----------------------------------------------------------------

#[test]
fn rfft_cosine_n8_f64() {
    let n = 8;
    let mut rfft = RfftPlanner::<f64>::new(n);
    let freq = 2.0;
    let input: Vec<f64> = (0..n)
        .map(|i| (TAU * freq * i as f64 / n as f64).cos())
        .collect();

    // Full complex reference
    let fft = FftPlanner::<f64>::new(n);
    let mut ref_re = input.clone();
    let mut ref_im = vec![0.0f64; n];
    fft.process(&mut ref_re, &mut ref_im);

    let n_out = n / 2 + 1;
    let mut out_re = vec![0.0f64; n_out];
    let mut out_im = vec![0.0f64; n_out];
    rfft.process_forward(&input, &mut out_re, &mut out_im);

    for i in 0..n_out {
        assert!(
            (out_re[i] - ref_re[i]).abs() < 1e-9,
            "n={n} re[{i}]: rfft={} vs ref={}",
            out_re[i],
            ref_re[i]
        );
        assert!(
            (out_im[i] - ref_im[i]).abs() < 1e-9,
            "n={n} im[{i}]: rfft={} vs ref={}",
            out_im[i],
            ref_im[i]
        );
    }
}

// -----------------------------------------------------------------
// RFFT vs full complex FFT for various N (roundtrip parity)
// -----------------------------------------------------------------

#[test]
fn rfft_parity_vs_complex_f64() {
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let mut rfft = RfftPlanner::<f64>::new(n);

        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();

        // Full complex FFT reference
        let fft = FftPlanner::<f64>::new(n);
        let mut ref_re = input.clone();
        let mut ref_im = vec![0.0f64; n];
        fft.process(&mut ref_re, &mut ref_im);

        let n_out = n / 2 + 1;
        let mut out_re = vec![0.0f64; n_out];
        let mut out_im = vec![0.0f64; n_out];
        rfft.process_forward(&input, &mut out_re, &mut out_im);

        for i in 0..n_out {
            assert!(
                (out_re[i] - ref_re[i]).abs() < EPS_F64,
                "n={n} re[{i}]: rfft={} vs ref={}",
                out_re[i],
                ref_re[i]
            );
            assert!(
                (out_im[i] - ref_im[i]).abs() < EPS_F64,
                "n={n} im[{i}]: rfft={} vs ref={}",
                out_im[i],
                ref_im[i]
            );
        }
    }
}

// -----------------------------------------------------------------
// RFFT f32 parity vs full complex FFT
// -----------------------------------------------------------------

#[test]
fn rfft_parity_vs_complex_f32() {
    for n in [2, 4, 8, 16, 32, 64, 128] {
        let mut rfft = RfftPlanner::<f32>::new(n);

        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();

        let fft = FftPlanner::<f32>::new(n);
        let mut ref_re = input.clone();
        let mut ref_im = vec![0.0f32; n];
        fft.process(&mut ref_re, &mut ref_im);

        let n_out = n / 2 + 1;
        let mut out_re = vec![0.0f32; n_out];
        let mut out_im = vec![0.0f32; n_out];
        rfft.process_forward(&input, &mut out_re, &mut out_im);

        for i in 0..n_out {
            assert!(
                (out_re[i] - ref_re[i]).abs() < 1e-4,
                "n={n} re[{i}]: rfft={} vs ref={}",
                out_re[i],
                ref_re[i]
            );
            assert!(
                (out_im[i] - ref_im[i]).abs() < 1e-4,
                "n={n} im[{i}]: rfft={} vs ref={}",
                out_im[i],
                ref_im[i]
            );
        }
    }
}

// -----------------------------------------------------------------
// N=2 edge case
// -----------------------------------------------------------------

#[test]
fn rfft_n2_f64() {
    let mut rfft = RfftPlanner::<f64>::new(2);
    let input = vec![3.0, 7.0];
    let mut out_re = vec![0.0f64; 2];
    let mut out_im = vec![0.0f64; 2];
    rfft.process_forward(&input, &mut out_re, &mut out_im);

    // DC = 3 + 7 = 10, Nyquist = 3 - 7 = -4
    assert!((out_re[0] - 10.0).abs() < EPS_F64);
    assert!((out_im[0] - 0.0).abs() < EPS_F64);
    assert!((out_re[1] - (-4.0)).abs() < EPS_F64);
    assert!((out_im[1] - 0.0).abs() < EPS_F64);
}

// -----------------------------------------------------------------
// panics on wrong buffer sizes
// -----------------------------------------------------------------

#[test]
#[should_panic(expected = "input length mismatch")]
fn rfft_wrong_input_len() {
    let mut rfft = RfftPlanner::<f32>::new(8);
    let input = vec![0.0f32; 7];
    let mut out_re = vec![0.0f32; 5];
    let mut out_im = vec![0.0f32; 5];
    rfft.process_forward(&input, &mut out_re, &mut out_im);
}

#[test]
#[should_panic(expected = "out_re length mismatch")]
fn rfft_wrong_out_len() {
    let mut rfft = RfftPlanner::<f32>::new(8);
    let input = vec![0.0f32; 8];
    let mut out_re = vec![0.0f32; 4];
    let mut out_im = vec![0.0f32; 5];
    rfft.process_forward(&input, &mut out_re, &mut out_im);
}

// -----------------------------------------------------------------
// linearity: RFFT(a + b) = RFFT(a) + RFFT(b)
// -----------------------------------------------------------------

#[test]
fn rfft_linearity_f64() {
    let n = 16;
    let mut rfft = RfftPlanner::<f64>::new(n);

    let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
    let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();

    let n_out = n / 2 + 1;
    let mut ra_re = vec![0.0f64; n_out];
    let mut ra_im = vec![0.0f64; n_out];
    let mut rb_re = vec![0.0f64; n_out];
    let mut rb_im = vec![0.0f64; n_out];
    rfft.process_forward(&a, &mut ra_re, &mut ra_im);
    rfft.process_forward(&b, &mut rb_re, &mut rb_im);

    let sum: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let mut sum_re = vec![0.0f64; n_out];
    let mut sum_im = vec![0.0f64; n_out];
    rfft.process_forward(&sum, &mut sum_re, &mut sum_im);

    for i in 0..n_out {
        let expected_re = ra_re[i] + rb_re[i];
        let expected_im = ra_im[i] + rb_im[i];
        assert!(
            (sum_re[i] - expected_re).abs() < EPS_F64,
            "linearity re[{i}]: {} vs {}",
            sum_re[i],
            expected_re
        );
        assert!(
            (sum_im[i] - expected_im).abs() < EPS_F64,
            "linearity im[{i}]: {} vs {}",
            sum_im[i],
            expected_im
        );
    }
}

// =================================================================
// IRFFT tests
// =================================================================

#[test]
fn irfft_roundtrip_f64() {
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let mut rfft = RfftPlanner::<f64>::new(n);

        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let n_out = n / 2 + 1;
        let mut fwd_re = vec![0.0f64; n_out];
        let mut fwd_im = vec![0.0f64; n_out];
        rfft.process_forward(&input, &mut fwd_re, &mut fwd_im);

        let mut out = vec![0.0f64; n];
        rfft.process_inverse(&mut fwd_re, &mut fwd_im, &mut out);

        for i in 0..n {
            assert!(
                (out[i] - input[i]).abs() < EPS_F64,
                "n={n} roundtrip[{i}]: {} vs {} (diff {})",
                out[i],
                input[i],
                (out[i] - input[i]).abs()
            );
        }
    }
}

#[test]
fn irfft_roundtrip_f32() {
    for n in [2, 4, 8, 16, 32, 64, 128] {
        let mut rfft = RfftPlanner::<f32>::new(n);

        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();
        let n_out = n / 2 + 1;
        let mut fwd_re = vec![0.0f32; n_out];
        let mut fwd_im = vec![0.0f32; n_out];
        rfft.process_forward(&input, &mut fwd_re, &mut fwd_im);

        let mut out = vec![0.0f32; n];
        rfft.process_inverse(&mut fwd_re, &mut fwd_im, &mut out);

        assert_slice_approx_eq_f32(&out, &input, 1e-3f32 * n as f32);
    }
}

#[test]
fn irfft_impulse_recovery_f64() {
    for n in [2, 4, 8, 16, 32, 64] {
        let mut rfft = RfftPlanner::<f64>::new(n);

        let mut input = vec![0.0f64; n];
        input[0] = 1.0;

        let n_out = n / 2 + 1;
        let mut fwd_re = vec![0.0f64; n_out];
        let mut fwd_im = vec![0.0f64; n_out];
        rfft.process_forward(&input, &mut fwd_re, &mut fwd_im);

        let mut out = vec![0.0f64; n];
        rfft.process_inverse(&mut fwd_re, &mut fwd_im, &mut out);

        for i in 0..n {
            assert!(
                (out[i] - input[i]).abs() < EPS_F64,
                "n={n} impulse[{i}]: {} vs {}",
                out[i],
                input[i]
            );
        }
    }
}

#[test]
fn irfft_n2_f64() {
    let mut rfft = RfftPlanner::<f64>::new(2);
    let input = vec![3.0, 7.0];
    let n_out = 2;
    let mut fwd_re = vec![0.0f64; n_out];
    let mut fwd_im = vec![0.0f64; n_out];
    rfft.process_forward(&input, &mut fwd_re, &mut fwd_im);

    let mut out = vec![0.0f64; 2];
    rfft.process_inverse(&mut fwd_re, &mut fwd_im, &mut out);

    assert_slice_approx_eq_f64(&out, &input, EPS_F64);
}

#[test]
fn irfft_known_n4_f64() {
    let mut rfft = RfftPlanner::<f64>::new(4);

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut fwd_re = vec![0.0f64; 3];
    let mut fwd_im = vec![0.0f64; 3];
    rfft.process_forward(&input, &mut fwd_re, &mut fwd_im);

    assert!((fwd_re[0] - 10.0).abs() < EPS_F64);
    assert!((fwd_im[0]).abs() < EPS_F64);
    assert!((fwd_re[1] - (-2.0)).abs() < EPS_F64);
    assert!((fwd_im[1] - 2.0).abs() < EPS_F64);
    assert!((fwd_re[2] - (-2.0)).abs() < EPS_F64);
    assert!((fwd_im[2]).abs() < EPS_F64);

    let mut out = vec![0.0f64; 4];
    rfft.process_inverse(&mut fwd_re, &mut fwd_im, &mut out);

    assert_slice_approx_eq_f64(&out, &input, EPS_F64);
}

#[test]
fn irfft_cosine_roundtrip_f64() {
    let n = 8;
    let mut rfft = RfftPlanner::<f64>::new(n);
    let freq = 2.0;
    let input: Vec<f64> = (0..n)
        .map(|i| (TAU * freq * i as f64 / n as f64).cos())
        .collect();

    let n_out = n / 2 + 1;
    let mut fwd_re = vec![0.0f64; n_out];
    let mut fwd_im = vec![0.0f64; n_out];
    rfft.process_forward(&input, &mut fwd_re, &mut fwd_im);

    let mut out = vec![0.0f64; n];
    rfft.process_inverse(&mut fwd_re, &mut fwd_im, &mut out);

    assert_slice_approx_eq_f64(&out, &input, EPS_F64);
}

#[test]
#[should_panic(expected = "in_re length mismatch")]
fn irfft_wrong_in_re_len() {
    let rfft = RfftPlanner::<f32>::new(8);
    let mut in_re = vec![0.0f32; 4];
    let mut in_im = vec![0.0f32; 5];
    let mut out = vec![0.0f32; 8];
    rfft.process_inverse(&mut in_re, &mut in_im, &mut out);
}

#[test]
#[should_panic(expected = "in_im length mismatch")]
fn irfft_wrong_in_im_len() {
    let rfft = RfftPlanner::<f32>::new(8);
    let mut in_re = vec![0.0f32; 5];
    let mut in_im = vec![0.0f32; 4];
    let mut out = vec![0.0f32; 8];
    rfft.process_inverse(&mut in_re, &mut in_im, &mut out);
}

#[test]
#[should_panic(expected = "output length mismatch")]
fn irfft_wrong_out_len() {
    let rfft = RfftPlanner::<f32>::new(8);
    let mut in_re = vec![0.0f32; 5];
    let mut in_im = vec![0.0f32; 5];
    let mut out = vec![0.0f32; 7];
    rfft.process_inverse(&mut in_re, &mut in_im, &mut out);
}

#[test]
fn irfft_parity_vs_complex_f64() {
    for n in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let rfft = RfftPlanner::<f64>::new(n);

        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();

        let fft = FftPlanner::<f64>::new(n);
        let mut ref_re = input.clone();
        let mut ref_im = vec![0.0f64; n];
        fft.process(&mut ref_re, &mut ref_im);

        let n_out = n / 2 + 1;
        let mut in_re = vec![0.0f64; n_out];
        let mut in_im = vec![0.0f64; n_out];
        in_re.copy_from_slice(&ref_re[..n_out]);
        in_im.copy_from_slice(&ref_im[..n_out]);

        let mut out = vec![0.0f64; n];
        rfft.process_inverse(&mut in_re, &mut in_im, &mut out);

        for i in 0..n {
            assert!(
                (out[i] - input[i]).abs() < EPS_F64,
                "n={n} parity[{i}]: {} vs {} (diff {})",
                out[i],
                input[i],
                (out[i] - input[i]).abs()
            );
        }
    }
}

#[test]
fn irfft_parity_vs_complex_f32() {
    for n in [2, 4, 8, 16, 32, 64, 128] {
        let rfft = RfftPlanner::<f32>::new(n);

        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).cos()).collect();

        let fft = FftPlanner::<f32>::new(n);
        let mut ref_re = input.clone();
        let mut ref_im = vec![0.0f32; n];
        fft.process(&mut ref_re, &mut ref_im);

        let n_out = n / 2 + 1;
        let mut in_re = vec![0.0f32; n_out];
        let mut in_im = vec![0.0f32; n_out];
        in_re.copy_from_slice(&ref_re[..n_out]);
        in_im.copy_from_slice(&ref_im[..n_out]);

        let mut out = vec![0.0f32; n];
        rfft.process_inverse(&mut in_re, &mut in_im, &mut out);

        assert_slice_approx_eq_f32(&out, &input, 1e-4);
    }
}
