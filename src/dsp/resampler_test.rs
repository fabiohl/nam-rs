// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_bypass_48k() {
    // When input rate equals output rate (e.g., 48kHz to 48kHz),
    // the system enters "Bypass" mode (shortcut), just copying sound without processing.
    let mut rs = NamResampler::new(48_000, 48_000, 256).expect("new failed");
    assert!(rs.is_bypass(), "48k should be bypass");

    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let input_r = [5.0f32, 4.0, 3.0, 2.0, 1.0];
    let mut output = [0.0f32; 5];
    let mut output_r = [0.0f32; 5];

    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);
    assert_eq!(n, 5);
    assert_eq!(output, input, "bypass must copy exactly L");
    assert_eq!(output_r, input_r, "bypass must copy exactly R");

    let n2 = rs.process_output(&input, &input_r, &mut output, &mut output_r);
    assert_eq!(n2, 5);
    assert_eq!(output, input);
    assert_eq!(output_r, input_r);
}

#[test]
fn test_downsample_96k_to_48k() {
    // Downsampling Test: Going from high resolution (96kHz) to standard (48kHz).
    // We expect the number of output samples to be approximately half the number of input samples.
    let chunk = 512usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");
    assert!(!rs.is_bypass());

    let input = vec![0.5f32; chunk];
    let input_r = vec![0.5f32; chunk];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);

    let expected_approx = chunk / 2;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "96k→48k: expected ~{expected_approx} samples, got {n}"
    );
}

#[test]
fn test_upsample_44k_to_48k() {
    // Upsampling Test: Going from CD quality (44.1kHz) to studio standard (48kHz).
    // Here the number of samples increases slightly (approximately 10% more).
    let chunk = 441usize;
    let mut rs = NamResampler::new(44_100, 48_000, chunk).expect("new failed");
    assert!(!rs.is_bypass());

    let input = vec![0.3f32; chunk];
    let input_r = vec![0.3f32; chunk];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);

    let expected_approx = (chunk as f64 * 48_000.0 / 44_100.0) as usize;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "44.1k→48k: expected ~{expected_approx} samples, got {n}"
    );
}

#[test]
fn test_output_upsample_48k_to_96k() {
    // Output Upsampling Test: When the internal model sound (48kHz)
    // needs to be sent to a sound card configured at 96kHz.
    let chunk = 256usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let inner_out_size = chunk / 2;
    let input = vec![0.4f32; inner_out_size];
    let input_r = vec![0.4f32; inner_out_size];
    let mut output = vec![0.0f32; chunk * 2];
    let mut output_r = vec![0.0f32; chunk * 2];
    let n = rs.process_output(&input, &input_r, &mut output, &mut output_r);

    let expected_approx = inner_out_size * 2;
    assert!(
        n >= expected_approx.saturating_sub(64) && n <= expected_approx + 64,
        "48k→96k (output): expected ~{expected_approx} samples, got {n}"
    );
}

#[test]
fn test_roundtrip_96k() {
    // "Roundtrip" Test: The most rigorous test.
    // We convert from 96kHz to 48kHz (one way) and then back to 96kHz (return).
    // At the end of this "phone game", the sound should be preserved and have energy (cannot be silent).
    let chunk = 1024usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let n_total = chunk * 4;
    let input: Vec<f32> = (0..n_total)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 96_000.0).sin())
        .collect();
    let input_r = input.clone();

    let mut total_mid = 0usize;
    let mut total_out = 0usize;
    let mut mid_energy_sum = 0.0f32;
    let mut out_energy_sum = 0.0f32;

    for start in (0..n_total).step_by(chunk) {
        let end = (start + chunk).min(n_total);
        let blk_l = &input[start..end];
        let blk_r = &input_r[start..end];

        let mut mid = vec![0.0f32; chunk];
        let mut mid_r = vec![0.0f32; chunk];
        let n_mid = rs.process_input(blk_l, blk_r, &mut mid, &mut mid_r);
        total_mid += n_mid;
        mid_energy_sum += mid[..n_mid].iter().map(|x| x * x).sum::<f32>();

        if n_mid > 0 {
            let mut out = vec![0.0f32; chunk * 2];
            let mut out_r = vec![0.0f32; chunk * 2];
            let n_out = rs.process_output(&mid[..n_mid], &mid_r[..n_mid], &mut out, &mut out_r);
            total_out += n_out;
            out_energy_sum += out[..n_out].iter().map(|x| x * x).sum::<f32>();
        }
    }

    assert!(
        total_mid > 0,
        "process_input produced no samples across {n_total} frames"
    );
    assert!(
        total_out > 0,
        "process_output produced no samples (mid_total={total_mid})"
    );
    assert!(
        mid_energy_sum > 0.0,
        "Intermediate energy (96→48) is zero (mid_total={total_mid})"
    );

    let energy_in = input.iter().map(|x| x * x).sum::<f32>() / n_total as f32;
    let energy_out = out_energy_sum / total_out.max(1) as f32;

    assert!(
        energy_out > energy_in * 0.05,
        "Roundtrip energy collapsed: in={energy_in:.4}, out={energy_out:.4}, \
         mid_samples={total_mid}, out_samples={total_out}, mid_energy={mid_energy_sum:.4}"
    );
}

#[test]
fn test_impulse_response_input() {
    // "Impulse" Test: We send a single digital click (a value of 1.0)
    // followed by silence to see how the filter reacts. It's like striking a bell
    // and measuring the resulting vibration: the sound should appear and fade naturally.
    let chunk = 512usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let mut input = vec![0.0f32; chunk];
    input[0] = 1.0;
    let mut input_r = vec![0.0f32; chunk];
    input_r[0] = 1.0;

    let mut output = vec![0.0f32; chunk];
    let mut output_r = vec![0.0f32; chunk];
    let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);
    assert!(n > 0);

    let energy: f32 = output[..n].iter().map(|x| x * x).sum();
    assert!(
        energy > 0.0 && energy.is_finite(),
        "Invalid impulse response: energy={energy}"
    );
    let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak <= 1.5, "Excessive peak: {peak:.4}");
}

#[test]
fn test_impulse_response_output() {
    // Same logic as the previous test (click/bell test), but applied to the output.
    // We verify that the output signal maintains energy and causes no weird distortions.
    let chunk = 256usize;
    let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new failed");

    let inner_out_approx = chunk / 2;
    let mut input = vec![0.0f32; inner_out_approx];
    input[0] = 1.0;
    let mut input_r = vec![0.0f32; inner_out_approx];
    input_r[0] = 1.0;

    let mut output = vec![0.0f32; chunk];
    let mut output_r = vec![0.0f32; chunk];
    let n = rs.process_output(&input, &input_r, &mut output, &mut output_r);
    assert!(n > 0);

    let energy: f32 = output[..n].iter().map(|x| x * x).sum();
    assert!(
        energy > 0.0 && energy.is_finite(),
        "Invalid impulse response (output): energy={energy}"
    );
    let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(peak <= 4.0, "Excessive peak (output): {peak:.4}");
}

#[test]
fn test_phase_accum_underflow_guard() {
    // We use ResamplerCore directly to test the internal logic
    let mut core = ResamplerCore::new(44100, 48000);

    // Simulates starting at 0
    core.phase_accum = 0;

    let in_l = [0.0f32; 64];
    let in_r = [0.0f32; 64];
    let mut out_l = [0.0f32; 64];
    let mut out_r = [0.0f32; 64];

    // Processing should occur without panic
    let n = dispatch_simd!(core, process_internal, &in_l, &in_r, &mut out_l, &mut out_r);
    assert!(n > 0);
}

#[test]
fn test_resampler_micro_soak() {
    // "Micro-stability" test for CI.
    // Processes 1M samples across various rates and checks invariants.
    let rate_pairs = [
        (44100, 48000),
        (48000, 44100),
        (96000, 48000),
        (22050, 48000),
        (88200, 48000),
    ];

    let chunk_size = 512;
    let n_iterations = 2000; // Total ~1M samples per pair

    let in_l = vec![0.1f32; chunk_size];
    let in_r = vec![0.1f32; chunk_size];
    let mut out_l = vec![0.0f32; chunk_size * 4];
    let mut out_r = vec![0.0f32; chunk_size * 4];

    for (from, to) in rate_pairs {
        let mut rs = NamResampler::new(from, to, chunk_size).unwrap();

        for _ in 0..n_iterations {
            let n = rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);

            // Basic invariant: output must be finite
            for i in 0..n {
                assert!(out_l[i].is_finite());
                assert!(out_r[i].is_finite());
            }

            // Access internal core to check accumulator (via ResamplerCore)
            if let Some(ref core) = rs.inner {
                let num_phases_fp = (NUM_PHASES as u64) << 40;
                // The accumulator may be >= NUM_PHASES if the input block finished
                // before consuming what's needed for the next output sample.
                assert!(
                    core.phase_accum < num_phases_fp + core.phase_step * 2,
                    "Overflow detected in {}->{}",
                    from,
                    to
                );
            }
        }
    }
}

#[test]
fn test_latency_calculation() {
    // 1. Bypass: latency should be 0
    let rs_48 = NamResampler::new(48_000, 48_000, 256).unwrap();
    assert_eq!(rs_48.latency_samples(48_000), 0);

    // 2. 44.1k -> 48k (Upsampling)
    // latency_in  = 16 * (44100 / 48000) = 14.7
    // latency_out = 16
    // total = 30.7 -> round -> 31 samples
    let rs_44 = NamResampler::new(44_100, 48_000, 256).unwrap();
    assert_eq!(rs_44.latency_samples(44_100), 31);

    // 3. 96k -> 48k (Downsampling)
    // latency_in  = 16 * (96000 / 48000) = 32
    // latency_out = 16
    // total = 48 samples
    let rs_96 = NamResampler::new(96_000, 48_000, 256).unwrap();
    assert_eq!(rs_96.latency_samples(96_000), 48);

    // 4. Zero rate: should return 0 (guardrail)
    assert_eq!(rs_44.latency_samples(0), 0);
}

#[test]
fn test_resampler_mono_equivalence() {
    // 1. Equivalence Test in Bypass mode (48kHz -> 48kHz)
    {
        let mut rs = NamResampler::new(48_000, 48_000, 256).unwrap();
        assert!(rs.is_bypass());

        let in_l = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out_l_stereo = [0.0f32; 5];
        let mut out_r_stereo = [0.0f32; 5];
        let mut out_l_mono = [0.0f32; 5];
        let mut out_r_mono = [0.0f32; 5];

        let n_stereo = rs.process_input(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo);
        let n_mono = rs.process_input_mono(&in_l, &mut out_l_mono, &mut out_r_mono);

        assert_eq!(n_stereo, n_mono);
        assert_eq!(out_l_stereo, out_l_mono);
        assert_eq!(out_r_stereo, out_r_mono);
        assert_eq!(out_l_mono, out_r_mono);

        // Same thing for process_output
        let n_out_stereo = rs.process_output(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo);
        let n_out_mono = rs.process_output_mono(&in_l, &mut out_l_mono, &mut out_r_mono);

        assert_eq!(n_out_stereo, n_out_mono);
        assert_eq!(out_l_stereo, out_l_mono);
        assert_eq!(out_r_stereo, out_r_mono);
    }

    // 2. Equivalence Test in active mode (44.1kHz -> 48kHz)
    {
        let chunk = 256;
        let mut rs_stereo = NamResampler::new(44_100, 48_000, chunk).unwrap();
        let mut rs_mono = NamResampler::new(44_100, 48_000, chunk).unwrap();

        let in_l: Vec<f32> = (0..chunk).map(|i| (i as f32 * 0.05).sin()).collect();
        let mut out_l_stereo = vec![0.0f32; chunk * 2];
        let mut out_r_stereo = vec![0.0f32; chunk * 2];
        let mut out_l_mono = vec![0.0f32; chunk * 2];
        let mut out_r_mono = vec![0.0f32; chunk * 2];

        let n_stereo = rs_stereo.process_input(&in_l, &in_l, &mut out_l_stereo, &mut out_r_stereo);
        let n_mono = rs_mono.process_input_mono(&in_l, &mut out_l_mono, &mut out_r_mono);

        assert_eq!(n_stereo, n_mono);
        for i in 0..n_stereo {
            assert!(
                (out_l_stereo[i] - out_l_mono[i]).abs() < 1e-4,
                "Mismatch L at index {}: stereo={}, mono={}",
                i,
                out_l_stereo[i],
                out_l_mono[i]
            );
            assert!(
                (out_r_stereo[i] - out_r_mono[i]).abs() < 1e-4,
                "Mismatch R at index {}: stereo={}, mono={}",
                i,
                out_r_stereo[i],
                out_r_mono[i]
            );
            assert_eq!(out_l_mono[i], out_r_mono[i]);
        }

        // Output stage (48kHz -> 44.1kHz)
        let mut rs_out_stereo = NamResampler::new(48_000, 44_100, chunk).unwrap();
        let mut rs_out_mono = NamResampler::new(48_000, 44_100, chunk).unwrap();

        let in_mid = &out_l_stereo[..n_stereo];
        let mut out_final_l_stereo = vec![0.0f32; chunk * 2];
        let mut out_final_r_stereo = vec![0.0f32; chunk * 2];
        let mut out_final_l_mono = vec![0.0f32; chunk * 2];
        let mut out_final_r_mono = vec![0.0f32; chunk * 2];

        let n_final_stereo = rs_out_stereo.process_output(
            in_mid,
            in_mid,
            &mut out_final_l_stereo,
            &mut out_final_r_stereo,
        );
        let n_final_mono =
            rs_out_mono.process_output_mono(in_mid, &mut out_final_l_mono, &mut out_final_r_mono);

        assert_eq!(n_final_stereo, n_final_mono);
        for i in 0..n_final_stereo {
            assert!(
                (out_final_l_stereo[i] - out_final_l_mono[i]).abs() < 1e-4,
                "Mismatch final L at index {}: stereo={}, mono={}",
                i,
                out_final_l_stereo[i],
                out_final_l_mono[i]
            );
            assert!(
                (out_final_r_stereo[i] - out_final_r_mono[i]).abs() < 1e-4,
                "Mismatch final R at index {}: stereo={}, mono={}",
                i,
                out_final_r_stereo[i],
                out_final_r_mono[i]
            );
            assert_eq!(out_final_l_mono[i], out_final_r_mono[i]);
        }
    }
}

#[test]
fn test_fixed_point_drift_random_ratios() {
    let ratios = [
        (44100.0, 48000.0),
        (48000.0, 44100.0),
        (96000.0, 48000.0),
        (88200.0, 48000.0),
    ];

    for &(from, to) in &ratios {
        let phase_step_f64 = (from / to) * NUM_PHASES as f64;
        let phase_step_u64 = (phase_step_f64 * ((1u64 << 40) as f64)).round() as u64;

        let mut accum_f64 = NUM_PHASES as f64;
        let mut accum_u64 = (NUM_PHASES as u64) << 40;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        for _ in 0..100_000 {
            while accum_f64 >= NUM_PHASES as f64 {
                accum_f64 -= NUM_PHASES as f64;
            }
            while accum_u64 >= num_phases_fp {
                accum_u64 -= num_phases_fp;
            }

            let phase_idx_f64 = accum_f64 as usize;
            let frac_f64 = accum_f64 - phase_idx_f64 as f64;

            let frac_bits = accum_u64 & ((1u64 << 40) - 1);
            let frac_u64 = frac_bits as f64 * (1.0 / (1u64 << 40) as f64);

            let diff = (frac_f64 - frac_u64).abs();
            assert!(
                diff < 1e-7,
                "Drift exceeds 1e-7: from={}, to={}, diff={}",
                from,
                to,
                diff
            );

            accum_f64 += phase_step_f64;
            accum_u64 += phase_step_u64;
        }
    }
}
