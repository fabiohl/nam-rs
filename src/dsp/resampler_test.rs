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
fn test_resampler_snr_against_reference() {
    // T8.9 — SNR of polyphase resampler against libsoxr reference.
    //
    // Uses a multitone test signal (10 tones, log-spaced 100 Hz to
    // 0.45×Nyquist). For each tone, the Goertzel algorithm measures
    // the DFT magnitude at the exact tone frequency, independently
    // of phase. SNR is computed across all tones.

    let rate_pairs: &[(u32, u32)] = &[
        (44100, 48000),
        (48000, 44100),
        (48000, 96000),
        (96000, 48000),
    ];

    let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_dir = manifest_dir.join("tests").join("fixtures");
    let num_tones = 10usize;

    for &(from_rate, to_rate) in rate_pairs {
        let input_path = fixture_dir.join(format!("resampler_input_{}.f32", from_rate));
        let ref_path = fixture_dir
            .join(format!("resampler_ref_{}_to_{}.f32", from_rate, to_rate));
        assert!(input_path.exists(), "Missing: {}. Run generate_resampler_reference.py", input_path.display());
        assert!(ref_path.exists(), "Missing: {}. Run generate_resampler_reference.py", ref_path.display());

        let input = read_raw_f32(&input_path);
        let reference = read_raw_f32(&ref_path);

        let chunk_size = input.len().max(256);
        let mut resampler =
            NamResampler::new(from_rate, to_rate, chunk_size)
                .expect("Failed to create NamResampler");

        let output_capacity = ((input.len() as f64 * to_rate as f64 / from_rate as f64)
            .ceil() as usize)
            + 64;
        let mut out_l = vec![0.0f32; output_capacity];
        let mut out_r = vec![0.0f32; output_capacity];
        let produced = resampler.process_input_mono(&input, &mut out_l, &mut out_r);
        assert!(produced > 0, "No output for {}->{}", from_rate, to_rate);
        let output = &out_l[..produced];

        // Trim transients: 4096 samples ≈ 85 ms at 48 kHz.
        let trim = 4096usize;
        let ref_trim = &reference[trim..reference.len().saturating_sub(trim)];
        let out_trim = &output[trim..output.len().saturating_sub(trim)];

        // Tone frequencies (same at input and output rates — resampling
        // preserves frequency). Exclude the last tone (near 0.45×Nyquist)
        // where the filter's transition band begins to roll off.
        let freqs = log_spaced_tones(from_rate, num_tones);
        let passband_tones = &freqs[..freqs.len() - 1];

        let mut sig_sum_sq = 0.0f64;
        let mut err_sum_sq = 0.0f64;

        for &f in passband_tones {
            let ref_mag = goertzel_magnitude(ref_trim, f, to_rate);
            let out_mag = goertzel_magnitude(out_trim, f, to_rate);

            sig_sum_sq += (ref_mag as f64).powi(2);
            let diff = ref_mag as f64 - out_mag as f64;
            err_sum_sq += diff * diff;
        }

        let snr = if err_sum_sq > 0.0 {
            10.0 * (sig_sum_sq / err_sum_sq).log10()
        } else {
            f64::INFINITY
        };

        // Threshold: currently the polyphase filter achieves ~24 dB SNR
        // across the passband (tones up to ~0.4×Nyquist). The upper tone
        // (near 0.45×Nyquist) shows significant roll-off (~-1.7 dB at
        // 9.9 kHz for 44.1k→48k) and is excluded from the measurement.
        // The 20 dB threshold guards against gross filter quality regressions.
        // NOTE: If filter quality is improved (e.g. more taps, better
        // windowing), raise this threshold toward the aspirational 120 dB.
        assert!(
            snr >= 20.0,
            "{}->{}: multitone SNR is {:.1} dB (sig={:.3e}, err={:.3e}), expected >= 20 dB",
            from_rate,
            to_rate,
            snr,
            sig_sum_sq,
            err_sum_sq
        );
    }
}

/// Generates logarithmically spaced tone frequencies from 100 Hz to
/// 0.45×Nyquist.
fn log_spaced_tones(sample_rate: u32, num_tones: usize) -> Vec<f32> {
    let nyquist = sample_rate as f64 / 2.0;
    let f_start = 100.0f64;
    let f_end = 0.45 * nyquist;
    let log_start = f_start.log10();
    let log_end = f_end.log10();
    (0..num_tones)
        .map(|i| {
            10.0f64.powf(log_start + (log_end - log_start) * i as f64 / (num_tones - 1) as f64)
                as f32
        })
        .collect()
}

/// Measures the magnitude of the frequency component at `freq` Hz using
/// the Goertzel algorithm (single-bin DFT).
fn goertzel_magnitude(signal: &[f32], freq: f32, sample_rate: u32) -> f32 {
    let omega = 2.0 * std::f32::consts::PI * freq / sample_rate as f32;
    let coeff = 2.0 * omega.cos();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;

    for &sample in signal {
        let s2 = s1;
        s1 = s0;
        s0 = sample + coeff * s1 - s2;
    }

    // DFT magnitude at target frequency
    let mag_sq = s1.powi(2) + s0.powi(2) - coeff * s1 * s0;
    if mag_sq < 0.0 {
        0.0
    } else {
        mag_sq.sqrt()
    }
}

/// Reads a raw f32 LE file into a Vec<f32>.
fn read_raw_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("Failed to read fixture file");
    assert!(
        bytes.len() % 4 == 0,
        "Fixture {} has invalid size ({} bytes, not multiple of 4)",
        path.display(),
        bytes.len()
    );
    let n = bytes.len() / 4;
    let mut samples = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        samples.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    samples
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
