<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Perceptual Validation & Measurement Framework

This document describes the complete measurement and perceptual validation infrastructure
for cross-validating NAM-rs inference quality against precision references (C++
NeuralAmpModelerCore, f64 oracle) and for standalone audio fidelity assessment.

## Measurement Philosophy

NAM-rs validates inference quality through two independent references:

1. **Parity Reference** — C++ NAMCore (f32): Measures implementation agreement against
   the upstream reference. Since both share the same numerical approximations (f32 +
   Padé tanh + minimax sigmoid + FMA accumulation), ESR targets are orders of magnitude
   lower (1e-5 to 3e-7) than modeling error baselines. See `tests/cpp_parity.rs` and
   `tests/golden_vectors.rs`.

2. **Absolute Correction** — f64 Oracle: Measures the absolute error floor of the
   production f32 path against an ideal double-precision computation with exact
   activation functions (`f64::tanh`, `f64::exp`) and compensated accumulation.
   Quantifies intrinsic quality loss from each approximation layer
   (quantization, activation, accumulation). See `src/testing/reference_oracle.rs`.

All measurement routines run off the real-time audio thread. They are pure analytics
with zero heap allocation in hot loops (aside from FFT planner construction).

---

## ESR — Error-to-Signal Ratio (Primary Scale-Robust Metric)

**File:** `src/testing/perceptual.rs:51` | `tests/common/metrics.rs:40`
**f64 variant:** `src/testing/reference_oracle.rs:201`

```text
ESR = Σ(rᵢ − tᵢ)² / Σ rᵢ²
```

Linear scale. Converted to dB via `10 · log₁₀(ESR)`.

### Why ESR?

Absolute MSE is not robust to scale mismatch — a signal 270× larger in amplitude
produces MSE ~10² even when SNR is 51–57 dB. ESR normalizes error by reference
energy, making it invariant to linear scaling. This is the primary threshold for
all parity gates.

**Limitation.** ESR is a global time-domain error metric — it is insensitive to
aliasing artifacts (Sato & Smith, DAFx 2025) and does not correlate linearly with
human auditory perception (Wright & Välimäki, ICASSP 2020). This is why nam-rs
supplements ESR with spectral metrics (MR-STFT, ASR) that capture frequency-domain
and aliasing-specific degradation modes that ESR alone cannot detect.

### Interpretation

| ESR range      | Meaning                                         |
| -------------- | ----------------------------------------------- |
| 0              | Perfect match                                   |
| 1              | Test signal has zero correlation with reference |
| < 1e-9         | Bit-identical (linear models, same-ISA)         |
| < 1e-7         | Bit-identical precision (typical same-ISA)      |
| ~1e-5          | Typical implementation parity (numerical noise) |
| ~6e-3 (median) | A1-Standard modeling error vs analog hardware   |
| > 1.0          | Complete divergence                             |

## 3-Tier Gate Hierarchy

NAM-rs validation uses a three-tier gate system that governs how thresholds evolve from tight per-model
values through sample-rate and stress-signal relaxation, ultimately bounded by absolute sentinels.
See `tests/common/validation.rs:435` (`get_calibrated_threshold`) and `tests/cpp_parity.rs:334-385`.

### Tier 1 — Per-Model Calibrated Thresholds

Defined in `tests/common/validation.rs:435` (`get_calibrated_threshold`). Each model entry stores
empirically measured `(mse_limit, min_snr_db, max_esr, mrstft_max)` at 48 kHz with 2048-sample v1
stress signal. Source measurements are documented in code comments.

| Model                                | SNR dB | ESR max | MR-STFT max | Notes                                                  |
| ------------------------------------ | ------ | ------- | ----------- | ------------------------------------------------------ |
| WaveNet Standard (CH=16)             | 105    | 3.0e-11 | 0.05        | Golden only                                            |
| WaveNet A1 Standard / Official CH=16 | 85     | 3.0e-9  | 0.05        | Live parity                                            |
| WaveNet Feather (CH=8)               | 100    | 1.0e-10 | 0.05        | Golden only                                            |
| WaveNet Nano (CH=4)                  | 95     | 3.0e-10 | 0.05        | Golden only                                            |
| WaveNet Lite (CH=12)                 | 105    | 3.5e-11 | 0.05        | Golden only                                            |
| WaveNet Official (CH=3)              | 14     | 3.5e-2  | 0.05        | Live parity, extreme compression                       |
| WaveNet Cond-DSP                     | 100    | 1.0e-10 | 0.35        | cond=3, dynamic path (see §MR-STFT Sensitivity Caveat) |
| WaveNet Dyn Free-Shape               | 90     | 1.0e-11 | 0.05        | CH=7→4, head_scale=0.02                                |
| Nondist Models (3×)                  | 100    | 1.0e-10 | 0.05        | APP-EVH, Boss BD-2, Slammin Marshall                   |
| A2-Full (CH=8)                       | 70     | 8.0e-8  | 0.05        | Gating+LeakyReLU                                       |
| A2-Lite (CH=3)                       | 80     | 6.0e-9  | 0.05        | Gating+tanh                                            |
| A2-FiLM-Lite (CH=3)                  | 12     | 2.0e-2  | 0.60        | FiLM active, RF1                                       |
| A2-FiLM-Full (CH=8)                  | 30     | 5.0e-4  | 0.55        | FiLM active, RF1                                       |
| A2 Dyn Gated CH=8                    | 85     | 1.0e-9  | 0.05        | Gating+LeakyReLU                                       |
| A2 Dyn Blended CH=3                  | 110    | 1.0e-12 | 0.05        | Blend+Tanh gate                                        |
| A2 Example (Slimmable)               | 70     | 8.0e-9  | 0.08        | SlimmableContainer                                     |
| ConvNet Test                         | 140    | 1.0e-10 | 0.05        | Self-golden consistency                                |
| LSTM 1×16                            | 12     | 6.5e-2  | 0.15        | Recurrent drift (see §LSTM Recurrent Drift)            |
| LSTM 2×8                             | 18     | 2.0e-2  | 0.12        | Recurrent drift                                        |
| LSTM Official (H=3)                  | 22     | 6.0e-3  | 0.22        | Recurrent drift                                        |
| LSTM-Dyn 1×7                         | 80     | 3.5e-9  | 0.08        | Non-catalog geometry, 48 kHz only                      |
| Linear                               | 140    | 1.0e-10 | —           | Bit-exact proxy                                        |

Fallback formulas (when a model has no calibrated entry):

- **Golden vectors** (`topology_thresholds`, `validation.rs:642`): LSTM `snr = (30 - complexity×0.65)`
- **Live parity** (`live_parity_thresholds`, `validation.rs:684`): LSTM `snr = (85 - complexity×1.0)`
  Both are stricter than calibrated values and gated by Tier 3.

### Tier 2 — Stress Signal × Sample-Rate Relaxation

Applied only in v2 multi-SR tests (`cpp_parity.rs:334-363`). Compensates for numerical accumulation
over the 100× longer stress signal (5s vs 42.7ms) and for higher sample rates:

```text
sr_ratio = sample_rate / 48000

LSTM:     snr_relaxation = (3.5 × sr_ratio).min(10.0)   // capped at 10 dB
WaveNet:  snr_relaxation = (1.5 × sr_ratio).min(4.0)    // capped at 4 dB
Resample: snr -= 1.5; mse ×= 1.5; esr ×= 1.5            // only when actual_sr ≠ model_sr
```

- LSTM gets a steeper relaxation because recurrent state drift is proportional to step count
  (see §LSTM Recurrent Drift below). Full formula: `min_snr -= snr_relaxation; mse ×= 10^(snr_relax/10); esr ×= 10^(snr_relax/10)`
- At 96 kHz: LSTM relaxes 7.0 dB; at 192 kHz: 10.0 dB (capped).
- The relaxation is **deliberate** — it exists to distinguish "expected format limitation"
  from "unexpected engine regression." Tier 3 is the backstop.

### Tier 3 — ABSOLUTE_ESR_CAP / ABSOLUTE_SNR_FLOOR Sentinel

After all Tier 2 relaxation, absolute sentinels (`cpp_parity.rs:374-384`) clamp the result:

```text
ABSOLUTE_ESR_CAP  = A2ESR_A1_STANDARD_MEDIAN = 6.23e-3   // baseline "good" from t3k-mushra
ABSOLUTE_SNR_FLOOR = 5.0 dB                              // absolute minimum SNR meaning
```

If `max_esr > ABSOLUTE_ESR_CAP` after relaxation, it is scaled back to 6.23e-3 and `mse_limit`
is proportionally tightened. `min_snr_db` is clamped to at least 5.0 dB.

**Purpose:** The cap acts as a sentinel, not a pass/fail criterion expected to always succeed.
When a model exceeds it (as all LSTM models do in v2), the test **intentionally fails** and
routes the case to recurrent drift triage (RCA concluded: f16c quantization was the root cause, now removed).
This ensures that "passing" always means "at least as precise as WaveNet A1-Std f32 native" —
preventing the relaxation chain from silently absorbing real regressions.

---

## MR-STFT — Multi-Resolution STFT Loss (Hard + Soft Spectral Gate)

**File:** `src/testing/perceptual.rs:126`

```text
MR-STFT = Σ_w weight[w] · mean_frame( L1_sc + L2_sc )
```

Where:

- `w` ∈ [256, 1024, 4096] with Hann window, hop = w/4
- L1_sc = (1/F) Σ_f |ln|X_ref[f]| − ln|X_test[f]||  (spectral convergence)
- L2_sc = √( (1/F) Σ_f (ln|X_ref[f]| − ln|X_test[f]|)² )  (log-magnitude loss)
- F = w/2+1 frequency bins (unique non-redundant FFT bins)
- Weights = [0.1, 0.3, 0.5] from t3k-mushra golden calibration

### Why MR-STFT?

Single-window STFT loss is biased toward the chosen time-frequency resolution
trade-off. MR-STFT combines three window sizes to capture narrow-band and
transient errors simultaneously. As a regression-detection gate, this metric
captures frequency-domain degradation modes that ESR alone misses — making it
a stronger indicator of spectral fidelity changes than time-domain error alone.
Note: MR-STFT is used here as a spectral regression gate, not as a direct proxy
for subjective perceptual quality (which requires human listening tests).

### Dual Gate System

MR-STFT uses a per-model calibrated threshold with a dual enforcement strategy
(`tests/common/validation.rs:252-277`, `342-349`):

**Hard gate — 44.1/48 kHz (native rates):** MR-STFT < `mrstft_max` from the
calibrated threshold table (§3-Tier Gate Hierarchy, Tier 1). Failures at these
rates **assert-panic** the test — there is no excuse for spectral degradation
at the model's training sample rate. Per-model `mrstft_max` values range from
0.05 (WaveNet SKU, near bit-exact) to 0.22 (LSTM Official, recurrent drift).

**Soft gate — 88.2/96/192 kHz (elevated rates):** `MRSTFT_SOFT_THRESHOLD = 0.15`
(`tests/common/validation.rs:264`). Informational only — not a hard assertion.
Higher sample rates accumulate more recurrent artifacts in LSTM architectures
(see §LSTM Recurrent Drift), making a hard gate inappropriate until S5 characterizes
the relationship precisely.

FFT via `crate::math::dsp::fft::FftPlanner` (native, SoA, zero-alloc). Purely
scalar (non-RT), suitable for test validation.

Golden cross-check: `tests/fixtures/mrstft_golden.bin` generated by
`tests/fixtures/scripts/gen_mrstft_golden.py` (Python reference). When the
golden file is present, `test_mr_stft_parity_with_python` in
`tests/common/perceptual.rs` validates bit-parity within 1e-6 absolute tolerance.

### MR-STFT Sensitivity Caveat — Spectrally Sparse Signals

Models with spectrally sparse output — many frequency bins near zero across
consecutive frames — can yield artificially elevated MR-STFT values even when
time-domain fidelity is virtually perfect (ESR ≈ 1e-14, SNR > 140 dB). This is
not signal degradation — it is a known limitation of the log-magnitude
computation in bins with energy near the noise floor.

**Mechanism.** MR-STFT computes `|ln|X_ref[f]| − ln|X_test[f]||` (spectral
convergence) and the L2 analogue (log-magnitude loss) per bin. When a frequency
bin is near zero in both reference and test signals, small absolute differences
of ~1e-7 produce large log-ratios because `ln(ε_ref) − ln(ε_test)` diverges as
`ε → 0`:

```text
ln(1e-15) − ln(2e-15) = ln(0.5) ≈ −0.693   (large relative difference)
|X_ref − X_test| = 1e-15                    (negligible absolute difference)
```

The per-bin log-ratio contribution inflates the frame-level mean even when the
absolute sample error is below machine f32 epsilon. This effect is negligible
in spectrally dense signals (most bins well above the noise floor) but dominates
the MR-STFT score in models with extended near-silent regions.

**Concrete example: `wavenet_condition_dsp`.** The `wavenet_condition_dsp`
model (CH=3, cond=3, dynamic sub-path) exemplifies this pattern in v2
multi-sample-rate testing at 48 kHz:

| Metric  | v1 (2048 samples, 42.7 ms) | v2 (240k samples, 5.0 s) |
| ------- | -------------------------- | ------------------------ |
| ESR     | 1.13e-14 (−139.5 dB)       | 8.93e-15 (−140.5 dB)     |
| SNR     | 139.5 dB                   | 140.5 dB                 |
| MR-STFT | 0.021                      | **0.336** (16× increase) |

The v2 signal is 117× longer but ESR **improves** (lower noise-to-signal ratio
over more samples). The MR-STFT rises 16× purely because the longer signal
accumulates more near-zero-bin frames — the condition path drives much of the
spectrum to near-silence for extended periods, amplifying the log-ratio artifact.
This was confirmed by analysis ruling out internal state drift: the
time-domain signal is virtually bit-exact.

**Calibrated thresholds for `condition_dsp`.** The MR-STFT hard gate at native
rates (44.1/48 kHz) uses per-model calibrated `mrstft_max` from Tier 1. The v2
stress signal (5s) triggers Tier 2 relaxation on `mrstft_max` (same formula as
ESR/SNR: `10^(snr_relaxation/5.0)` for WaveNet). Neither value reflects audio
degradation — both accommodate the log-magnitude sensitivity to spectral sparsity:

| Metric       | v1 (at 48 kHz) | v2 (at 48 kHz, relaxed) | Relaxation           |
| ------------ |:--------------:|:-----------------------:| -------------------- |
| `mrstft_max` | 0.35           | 0.698                   | `10^(snr_relax/5.0)` |
| `max_esr`    | 1.0e-10        | relaxed¹                | Tier 2 relaxation    |
| `min_snr_db` | 100.0          | relaxed¹                | Tier 2 relaxation    |

¹ Relaxation formula: `min_snr −= 1.5 × sr_ratio` (capped at 4 dB), applied
symmetrically to `mse_limit` and `max_esr`. At 48 kHz where `sr_ratio = 1.0`,
the relaxation is 1.5 dB.

**Practical guidance.** For any model where the conditioning or gating path
drives significant spectral regions to near-zero:

1. **ESR is the decisive gate.** If ESR/SNR are within calibrated bounds, high
   MR-STFT is a metric artifact, not a fidelity defect.
2. **Calibrate `mrstft_max` per model empirically.** The 0.05 default inherited
   from dense WaveNet models is inappropriate for sparse-output architectures.
   Use v1 (42.7 ms) measurements as the floor and let Tier 2 relaxation handle
   the v2 5-second signal.
3. **Do not apply unconditional spectral thresholds.** A single `mrstft_max`
   across all models would either be too strict for sparse models (false
   positives) or too lenient for dense models (masking real degradation).

---

## ASR — Aliasing-to-Signal Ratio (DAFx 2025)

**File:** `src/testing/aliasing.rs:133`

```text
ASR = Σ E_aliased / Σ E_harmonic
ASR_dB = 10 · log₁₀(ASR_linear)
```

Measures the energy ratio of aliased (non-harmonic) components to harmonic
components when a pure sine wave is processed through a nonlinear system.

### Algorithm

1. Apply 4-term Blackman-Harris window (a₀=0.35875, a₁=0.48829, a₂=0.14128, a₃=0.01168)
2. Real FFT via `RfftPlanner<f64>`
3. Noise floor = median of all bin magnitudes
4. Peak threshold = max(noise_floor × 6.0, max_mag × 1e-4)
5. Detect local maxima in magnitude spectrum (skip DC)
6. Classify peaks as harmonic (k·f₀ within 1.5 bins) or aliased
7. ASR = sum of aliased energies / sum of harmonic energies

### Why ASR?

Neural amp models contain nonlinearities (tanh, sigmoid) that generate harmonics
beyond Nyquist — these fold back as aliasing. ASR quantifies anti-aliasing quality
inherent to the model architecture and any built-in oversampling. It fingerprints
aliasing behavior and gates regressions.

### Test Pitch Table

`MUSICAL_PITCHES` (`src/testing/aliasing.rs:24`): E2 (82.41 Hz) through E5 (659.25 Hz)
plus a stress tone at 2017 Hz (incommensurate with 48 kHz Nyquist to avoid coincidental
harmonic fold-back). High gain (+12 dB, `HIGH_GAIN = 4.0`) drives nonlinearities harder
for stress testing.

### Aggregate Functions

- `asr_aggregate()` (`src/testing/aliasing.rs:247`): Arithmetic mean of linear ASR values → dB
- `asr_worst_case()` (`src/testing/aliasing.rs:261`): Maximum ASR across all pitches

### Interpretation ASR

- ASR < −60 dB or linear < 1e-6: effectively alias-free (linear system)
- ASR > −30 dB: significant aliasing (hard-clip, severe nonlinearity)
- No hard CI threshold — informational/diagnostic gate used to fingerprint model behavior

Tests: `src/testing/aliasing_test.rs` (unit), `tests/spectral_fidelity.rs:34-370`
(integration + model fingerprints).

---

## Farina Exponential Sine Sweep — FR + THD

**File:** `src/testing/spectral.rs:224`

Simultaneous measurement of impulse response (frequency response magnitude/phase)
and THD per harmonic order via deconvolution, following Farina (AES Convention 108, 2000).

### Sweep Generation

```text
x(t) = sin[φ(t)],  φ(t) = ω₁·T / ln(ω₂/ω₁) · (exp(t·ln(ω₂/ω₁)/T) − 1)
```

Instantaneous phase computed per sample via `omega1 * duration_s / ln_ratio * ((t_norm * ln_ratio).exp_m1())`.

### Inverse Filter

```text
F[k] = conj(S[k]) / (|S[k]|² + ε)   (frequency-domain matched filter)
```

Normalized to peak amplitude 0.95. Compensates the −3 dB/octave spectral envelope
of the exponential sweep.

### Deconvolution & Analysis

1. Generate exponential sweep [f₁, f₂] over duration_s at sample_rate
2. Process through `process_fn` closure
3. Circular convolution (FFT multiplication) of system output × inverse filter
4. Extract linear IR from start to half-time (T·ln(1.5)/ln(f₂/f₁))
5. Compute FR magnitude (dB) and phase from FFT of linear IR
6. Extract harmonic distortion kernels per order at time lag: Δt_k = T · ln(k) / ln(f₂/f₁)
7. THD per order: `thd_k = 100 · √(energy_k / fund_energy)` for k ≥ 2
8. THD total: `thd_total = 100 · √( Σ (thd_k/100)² for k ≥ 2)`

### Why Farina?

Unlike stepped-sine or MLS, the exponential sweep separates linear and nonlinear
components in the time domain after deconvolution. Each harmonic order produces
its own impulse response at a predictable time lag, enabling simultaneous FR,
phase, and per-order THD measurement in a single pass.

### Result Struct

`FarinaResult` (`src/testing/spectral.rs:72`): `sample_rate`, `f1`, `f2`,
`duration_s`, `ir_linear`, `fr_magnitude_db`, `fr_phase_rad`, `freq_axis`,
`thd_by_order: Vec<(u32, f64)>`, `thd_total_percent`.

Tests: `src/testing/spectral_test.rs:108-204` (unit), `tests/spectral_fidelity.rs:398-436`
(model measurements, `#[ignore]`).

---

## THD+N — AES17

**File:** `src/testing/spectral.rs:456`

Total Harmonic Distortion + Noise per AES17 standard, using a 997 Hz pure tone.

### Algorithm THD+N

1. Generate pure tone at f₀ Hz (default 997 Hz per AES17)
2. Process through `process_fn` closure
3. Biquad notch-filter the fundamental (Q ≈ 5, second-order design)
4. Discard 2000 samples for biquad settling
5. THD+N = 100% · RMS(notched) / RMS(total)
6. THD+N_dB = 20 · log₁₀(thdn_percent / 100)

### Why THD+N?

Industry-standard single-number distortion metric. Measures everything that is
not the fundamental — harmonic distortion, intermodulation, noise, and aliasing
combined. The notch-filter approach is computationally efficient and numerically
stable.

### Interpretation THD+N

- Linear system (unity gain): THD+N < 2% (test tolerance)
- Hard-clip: 5–40% depending on clipping severity
- Guitar amp model: 5–30% typical (distortion is the intended effect)

`ThdnResult` (`src/testing/spectral.rs:366`): `f0`, `sample_rate`, `thdn_percent`,
`thdn_db`, `rms_notched`, `rms_total`.

Tests: `src/testing/spectral_test.rs:248-280` (unit), `tests/spectral_fidelity.rs:438-455`
(model measurements, `#[ignore]`).

---

## IMD — SMPTE/DIN

**File:** `src/testing/spectral.rs:579`

Intermodulation Distortion per SMPTE standard: 60 Hz + 7 kHz two-tone, 4:1 amplitude ratio.

### Algorithm IMD

1. Generate two-tone: 60 Hz + 7 kHz, 4:1 amplitude ratio (SMPTE standard)
2. Process through `process_fn` closure
3. FFT with 4-term Blackman-Harris window
4. Identify carrier bin (f_high = 7 kHz) and sidebands at f_high ± n·f_low
5. Sideband search range: ±2 bins around expected frequency
6. IMD(%) = 100 · √( Σ sideband_mag² ) / carrier_mag
7. IMD_dB = 20 · log₁₀(imd_percent / 100)

### Why SMPTE IMD?

THD+N at a single frequency can miss intermodulation products generated when
multiple frequencies interact through nonlinearities. Guitar signals are
polyphonic — IMD captures the cross-modulation between low (power chord) and
high (harmonic) frequencies that is perceptually critical for amp modeling.

### Interpretation IMD

- Linear system: IMD < 5% (test tolerance)
- Hard-clip: IMD > 1%
- Higher IMD correlates with "muddy" distorted tones where low-frequency content
  modulates high-frequency clarity

`SmpteImdResult` (`src/testing/spectral.rs:525`): `f_low`, `f_high`, `ratio`,
`sample_rate`, `imd_percent`, `imd_db`, `sideband_percents`.

Tests: `src/testing/spectral_test.rs:299-325` (unit), `tests/spectral_fidelity.rs:458-501`
(model measurements, `#[ignore]`).

---

## LUFS — ITU-R BS.1770-4 Integrated Loudness (Full 2-Pass Gating)

**File:** `src/testing/perceptual.rs:328` (`compute_integrated_lufs`)

Full implementation of ITU-R BS.1770-4 integrated loudness with absolute and
relative gating. Single-channel mono computation.

### Algorithm LUFS

1. Apply K-weighting (pre-filter HP ~38 Hz + RLB high-shelf +4 dB > 1–2 kHz)
   - Pre-filter:  H(z) = (1 − 2z⁻¹ + z⁻²) / (1 − 1.99004745z⁻¹ + 0.99007225z⁻²)
   - Shelf:       H(z) = (1.53512486 − 2.69169619z⁻¹ + 1.19839281z⁻²)
                       / (1.0 − 1.69065929z⁻¹ + 0.73248077z⁻²)
2. Divide into 400 ms blocks with 75% overlap (`LUFS_BLOCK_MS=400`)
3. Compute mean-square power per block
4. Pass 1 — absolute gate: discard blocks ≤ −70 LUFS (`LUFS_ABS_GATE = −70.0`)
5. Compute ungated integrated loudness from surviving blocks
6. Pass 2 — relative gate: discard blocks below (ungated − 10 LU) (`LUFS_REL_GATE = −10.0`)
7. Integrated LUFS = −0.691 + 10 · log₁₀(mean of surviving block powers)

### Why LUFS?

Standardized loudness measurement. Provides a perceptually-weighted energy
assessment that correlates with human loudness perception. Used as a plausibility
gate on golden reference output — implausible LUFS values indicate upstream bugs
(e.g., scaling errors).

### Plausibility Gate

`LUFS_PLAUSIBLE_MIN = −50.0`, `LUFS_PLAUSIBLE_MAX = +10.0`
(`tests/common/validation.rs:22-23`).

The gate enforces that: a golden vector with LUFS −67 (near-silence)
passed all time-domain checks undetected — the signal was structurally valid
but perceptually meaningless. Now any reference signal outside [−50, +10] LUFS
triggers a "GOLDEN DEFECT" warning.

**Short-signal tolerance:** Signals shorter than 400ms (below one BS.1770-4
integration block) bypass the LUFS gate — the measurement produces non-finite
values. This is automatic in `report_dsp_fidelity` (`validation.rs:183-187`).

**Opt-out:** `report_dsp_fidelity_no_lufs` (`validation.rs:85`) skips the gate
for IR convolution goldens, where input is an impulse and high amplitude is
legitimate. The gate is also informational (`ⓘ`, not `✗`) when explicitly
`check_lufs_gate=false` (`validation.rs:361-365`).

---

## LRA — EBU Tech 3342 Loudness Range

**File:** `src/testing/perceptual.rs:397` (`compute_lra`)

Quantifies the macro-dynamic range of a program — the statistical distribution
of loudness over time, not peak-to-average ratio.

### Algorithm LRA

1. Compute short-term loudness (3-second blocks, non-overlapping, `LRA_BLOCK_MS=3000`)
2. Pass 1 — absolute gate: discard blocks ≤ −70 LUFS
3. Compute mean of surviving blocks → L_ASG (absolute-gated loudness)
4. Pass 2 — relative gate at −20 LU (`LRA_REL_GATE = −20.0`): discard blocks < (L_ASG − 20)
5. Sort remaining blocks by loudness
6. LRA = P95 − P10 (linear interpolation, C=1 method per EBU Tech 3342 Annex)

### Why LRA?

Steady-state test tones have LRA ≈ 0. Dynamic signals (music, sweeps) have
LRA 12–18 LU. LRA serves as a signal-type sanity check — extremely high LRA on
a golden test vector suggests a defect.

Tests: `src/testing/perceptual_test.rs:218-304`.

---

## True-Peak — ITU-R BS.1770-4 Annex 2 (dBTP)

**File:** `src/testing/perceptual.rs:652` (`compute_true_peak_db`)

Measures the inter-sample peak — the true analog peak after D/A reconstruction —
which can exceed 0 dBFS even when all digital samples are ≤ 0 dBFS (Gibbs phenomenon).

### Algorithm True-Peak

1. 4× oversampling via BS.1770-4 Annex 2 48-tap polyphase FIR
   - 4 phases × 12 taps each (`BS1770_PHASES`, `src/testing/perceptual.rs:587-612`)
   - `y[4n+p] = Σ x[n−k] · phase_p[k]` for k=0..11
2. Peak absolute value of upsampled signal
3. dBTP = 20 · log₁₀(peak_abs)
4. Returns −∞ for empty or all-zero input

### Why True-Peak?

Sample-peak detectors miss inter-sample overshoots that cause clipping in D/A
converters. dBTP catches true analog-domain peaks, preventing downstream hardware
clipping. Critical for output-stage validation.

### RT-Safety Note

True-peak with 48-tap FIR × 4× oversampling (~48 MAC/sample) is **not used in
the RT hot-path**. The DSP output stage uses sample-peak only for clipping
detection. True-peak is off-RT QA/telemetry only.

### Additional Functions

- `find_true_peak_overs()` (`src/testing/perceptual.rs:681`): Returns Vec of all overs
- `oversample_4x()` (`src/testing/perceptual.rs:713`): Returns full upsampled f64 signal

Tests: `src/testing/perceptual_test.rs:462-658`.

---

## Combined Loudness Measurement

**File:** `src/testing/perceptual.rs:522` (`measure_loudness`)

```rust
pub struct LoudnessResult {
    pub integrated_lufs: f64,
    pub lra: f64,
    pub true_peak_db: f64,
    pub short_term: Vec<f64>,  // per-block LKFS values
}
```

Computes LUFS + LRA + dBTP in a single pass, sharing the K-weighting filter
between LUFS and LRA.

Tests: `src/testing/perceptual_test.rs:346-408`.

---

## f64 Reference Oracle — Absolute Error Floor

**File:** `src/testing/reference_oracle.rs:232` (`oracle_forward`)

Computes the ideal forward pass of WaveNet, LSTM, and A2 topologies using f64
arithmetic, exact activation functions (`f64::tanh`, `f64::exp`), and Kahan/Neumaier
compensated accumulation.

### Why an Oracle?

The production path (f32 + Padé tanh + minimax sigmoid + FMA accumulation) shares
the same limitations as C++ NAMCore. The oracle provides an **independent**
high-precision reference that:

1. Measures the **absolute error floor** of the f32 production path
2. Permits **source decomposition** — isolating the contribution of each
   approximation (weight quantization, activation, accumulation) to total error

### Decomposition Pipeline

`run_decomposition()` (`src/testing/reference_oracle.rs:775`) runs the oracle
under 5 configurations and returns a `DecompositionResult` (`src/testing/reference_oracle.rs:765`):

| Field              | What it isolates                                      |
| ------------------ | ----------------------------------------------------- |
| `esr_f32_vs_f64`   | Full production f32 vs ideal f64 oracle (total error) |
| `esr_quant_f16c`   | f16c weight quantization error only                   |
| `esr_quant_bf16`   | bf16 weight quantization error only                   |
| `esr_activation`   | Padé tanh + minimax sigmoid vs exact f64 activations  |
| `esr_accumulation` | f32 accumulation vs Kahan/Neumaier f64 accumulation   |

### Two References — Parity vs Absolute

| Reference         | Type     | What it measures                      | Typical target         |
| ----------------- | -------- | ------------------------------------- | ---------------------- |
| C++ NAMCore (f32) | Parity   | Implementation agreement (shared f32) | < mse_limit (Tier 1–3) |
| f64 Oracle        | Absolute | Intrinsic quality loss from f32 path  | Varies by architecture |

The parity reference answers "Is our f32 code compatible with upstream?" The
absolute reference answers "How much quality did we lose by using f32?"

**Interop-vs-Correction in practice (post-T8.2 corrected oracle):** For WaveNet,
ESR(vs NAMCore) ≈ 1e-13 and ESR(vs f64 oracle, prewarm-paired) = 6.13e-14 (−132 dB)
— virtually indistinguishable from the numerical floor. For LSTM, ESR(vs NAMCore)
≈ 2.61e-2 (v2, 240k steps, 5s) while ESR(vs f64 oracle, prewarm-paired) = 3.57e-3
(−24.5 dB) — the interop drift is ~7× larger than the absolute precision floor.
Both engines share the f16c+f32 recurrent drift, producing a 2.61e-2 interop gap,
but each individually diverges from the ideal by only 3.57e-3 when measured with
matched prewarm. The pre-T8.2 oracle reported ~1.0 due to unmatched-state
architectural divergence — a ~300× inflation now corrected. The per-model
calibrated thresholds reflect this reality: LSTM entries are looser than WaveNet
because the recurrent f16c+f32 drift is shared with NAMCore. The dominant sources
are Padé activation (~7.6e-4 ΔESR) and f16c quantization (~5.1e-5), with f32
accumulation negligible (~7.2e-13).

Tests: `tests/reference_oracle_f64.rs:67-268`.

---

## LSTM Recurrent State Quantization Drift

LSTM models (`BossLSTM-1×16`, `BossLSTM-2×8`, `LSTM Official H=3`) exhibit
ESR significantly above the WaveNet A1-Std baseline (6.23e-3), even at native
sample rates where no resampling is involved. This is **not a NAM-rs regression**
— the limitation is inherent to the `.nam` format with f16c-quantized weights in
recurrent architectures.

### Mechanism

The LSTM cell state update accumulates f16c quantization error across time steps:

```text
cₜ = fₜ · cₜ₋₁ + iₜ · gₜ
hₜ = oₜ · tanh(cₜ)
```

All four gates are computed via 4-way GEMV using f16c-quantized weights
(`src/models/lstm/layer.rs:11` — `[[[u16; H]; IH]; 4]`). Each time step injects
~2.3e-3 per-gate quantization error into `cₜ`. The forget gate `fₜ` (typically
0.9–0.99 in trained networks) partially decays old errors, limiting the
accumulation to a steady-state: `⟨ESR⟩ ∝ σ²ε / (1 − ⟨f⟩²)`.

### Empirical evidence

| Sequence           | LSTM Steps | ESR (vs NAMCore) | MR-STFT | ESR (vs f64 oracle) | Notes                       |
| ------------------ | ---------- | ---------------- | ------- | ------------------- | --------------------------- |
| 2,048 (42.7ms, v1) | 2,048      | 1.04e-2          | 0.098   | —                   | Passes all gates            |
| 240,000 (5s, v2)   | 240,000    | 2.61e-2          | 0.87    | 3.57e-3 (−24.5 dB)  | Prewarm-paired oracle match |
| 480,000            | 480,000    | 6.09e-2          | 1.38    | —                   | Fails all gates             |
| 960,000            | 960,000    | 1.42e-1          | 1.93    | —                   | Worst case                  |

The ESR grows sub-quadratically (not ∝N²), confirming the forget gate limits
accumulation to a rate-dependent steady-state. The oracle vs production ESR
(3.57e-3 at 48 kHz v2 with prewarm-paired state) is ~7× smaller than the
nam-rs vs NAMCore interop gap (2.61e-2) — both engines share the f16c+f32
recurrent drift, but their mutual gap exceeds the absolute precision floor.

### Hypotheses ruled out

Systematic testing ruled out: band-edge filtering (no DC-block in pipeline),
denormal dither (±1e-11 symmetric, −220 dBFS), aliasing from non-linearity
(ASR = −68.8 dB at 48 kHz), and harness resample path (48 kHz bypasses resampler).
None of these contribute significantly to the observed ESR.

### Classification

The divergence has two distinct layers:

1. **Interop (nam-rs vs NAMCore):** ESR = 2.61e-2 (v2, 240k steps @ 48 kHz).
   The two engines share the same f16c weights and f32 accumulation model but
   execute with different code paths — the observed 2.61e-2 is the real,
   measurable recurrent drift between the two implementations.

2. **Absolute correction (production f32 vs f64 ideal):** ESR = 3.57e-3
   (−24.5 dB) with prewarm-paired state matching (T8.2/T8.3, 2026-06-28).
   This is the true precision floor of the f16c+f32 production path vs
   double-precision arithmetic — ~300× smaller than the pre-T8.2 oracle
   reading of ~1.0, which was inflated by unmatched-initial-state architectural
   divergence between oracle and production. The decomposition (T8.3) isolates
   the dominant sources: Padé tanh activation (~7.6e-4 ΔESR) and f16c weight
   quantization (~5.1e-5), with f32 accumulation negligible (~7.2e-13).

The pre-T8.2 conclusion _"both diverge from the ideal by ~1.0"_ was contaminated
by the oracle's unmatched state; the corrected picture is: the interop gap
(2.61e-2) persists and is real, but the absolute precision floor (3.57e-3) is
much lower, confirming the recurrent drift mechanism while correcting its
magnitude vs the ideal.

### Impact on gates

- **Tier 1 thresholds** for LSTM models (6.5e-2, 2.0e-2, 6.0e-3) are calibrated
  from empirical interop measurements and reflect the recurrent drift shared with
  NAMCore.
- **Tier 3 cap** (6.23e-3) is lower than all LSTM interop measurements in v2 —
  these models **systematically fail** the absolute sentinel, intentionally.
  The cap acts as a visible gating mechanism; `ABSOLUTE_ESR_CAP_LSTM = 0.08`
  (T8.4, 2026-06-28) provides architecture-specific sentinel headroom
  (2.61e-2 × 3 margin).
- **MR-STFT hard gate** at 44.1/48 kHz also fails for LSTM in v2 due to broadband
  spectral error from quantization noise in the recurrent state.
- **Oracle fidelity gates** (`LSTM_ESR_LIMIT = 7.0e-3` in
  `tests/reference_oracle_f64.rs`, calibrated from prewarm-paired 3.57e-3 × 2)
  are now below the project placebo line (ESR < 1.0), restoring the gate's
  ability to detect regressions vs absolute precision.

### Qualification of T3.3 conclusion ("not fixable without changing format")

**T3.3 (2026-06-27)** concluded: _"ESR ≈ 1.0 vs ideal = inherent floor of f16c
quantization… not fixable without altering the model format."_

**Post-T8.2/T8.3 (2026-06-28) correction:**

- The "~1.0" was **architectural divergence** in the oracle (unmatched prewarm
  state), not f16c precision loss. The real absolute floor is 3.57e-3 — ~300×
  smaller.
- The **mechanism** of recurrent state quantization drift is correct and remains
  valid. The forget-gate leak, step-proportional ESR growth, and broadband
  spectral error distribution are all confirmed.
- The interop gap (2.61e-2 vs NAMCore at v2/240k) **persists after T8.2** — it
  is real recurrent drift shared by both f32 engines, not an oracle artifact.
- Whether this is "not fixable without changing the format" depends on scope:
  - **vs NAMCore (interop):** The 2.61e-2 gap between implementations **may** be
    reducible by aligning the recurrence execution (e.g., matching bf16 state
    precision, FMA ordering in the cell update). The E8 root-cause investigation
    (AC-7) will determine how much of this gap is addressable.
  - **vs f64 ideal (absolute):** The 3.57e-3 floor is dominated by Padé
    activation (~7.6e-4) and f16c quantization (~5.1e-5) — both are intrinsic to
    the current format. Kahan accumulation in the LSTM head (planned E4/S5)
    targets the recurrent head accumulation but not the body weight quantization.
  - **Practical verdict:** The interop gap is the user-visible metric (sound
    difference vs NAMCore), and it may be partially addressable. The absolute
    gap vs ideal is smaller than originally assumed and bounded by the f16c
    format, confirming the format is indeed the bottleneck — but ~300× less
    severely than T3.3's original statement implied.

RCA concluded: f16c quantization was the root cause of the drift, now eliminated in v2.

---

## Fidelity Report — Multi-Metric Pass

**File:** `tests/common/validation.rs:55` (`report_dsp_fidelity`)

Single-pass multi-metric report for golden vector and parity validation.
Computes in one traversal over reference/test signals:

| Metric           | Computation                                     | Gate / Target                                        |
| ---------------- | ----------------------------------------------- | ---------------------------------------------------- |
| MSE              | noise_power / n                                 | < mse_limit (Tier 1–3 relaxed)                       |
| MAE              | max absolute difference                         | informational                                        |
| SNR              | 10 · log₁₀(signal_power / noise_power)          | > min_snr_db (Tier 1–3 relaxed)                      |
| PSNR             | 10 · log₁₀(peak_ref² / mse)                     | informational                                        |
| Equivalent Bits  | −0.5 · log₂(mse / signal_avg_power)             | informational                                        |
| ESR              | noise_power / signal_power                      | < max_esr (primary gate, Tier 1–3)                   |
| MR-STFT          | multi-resolution spectral loss                  | < mrstft_max (hard @ 44.1/48k, soft @ higher rates)  |
| LUFS (reference) | integrated loudness (BS.1770-4 2-pass)          | [−50, +10] plausibility (skipped for <400ms signals) |
| dBTP (reference) | true-peak (BS.1770-4 Annex 2, 4× polyphase)     | informational                                        |
| Anchor SNR       | SNR of test against 3.5 kHz 1-pole LP reference | degradation baseline                                 |
| Fidelity Margin  | SNR − anchor_SNR                                | > 8.0 dB target                                      |

`Fidelity Margin` quantifies how much better the test signal matches the reference
than a degraded anchor (3.5 kHz low-pass). Target > 8.0 dB ensures meaningful
fidelity above a low-quality baseline.

Variant `report_dsp_fidelity_no_lufs` (`tests/common/validation.rs:85`) skips
LUFS gate for cases where high signal amplitude is legitimate (e.g., IR convolution
goldens). Short signals (<400ms, shorter than one BS.1770-4 integration block)
also bypass the LUFS gate automatically.

---

## ISA Parity & Performance Gates

**File:** `tests/isa_parity.rs`

End-to-end cross-ISA determinism validation. Runs golden vectors through each
supported SIMD ISA path and asserts output parity.

### ISA Override Infrastructure

`TEST_ISA_OVERRIDE: AtomicU8` (`src/math/common/dispatch/detect.rs:31`) allows
forcing a specific ISA path:

- `encode_isa_override(isa)` → byte value written to `TEST_ISA_OVERRIDE`
- `effective_instruction_set()` → reads override; falls back to native if unset
- `dispatch_simd!` macro → resolves to concrete ISA method via `SIMD_MATH.instruction_set`
- `IsaGuard` → RAII guard that restores override on drop; `ISA_LOCK: Mutex` serializes between threads

### Test Matrix

| Mode             | ISA Pair         | Models                                                             | Gate                     |
| ---------------- | ---------------- | ------------------------------------------------------------------ | ------------------------ |
| Self-consistency | AVX2 → AVX2      | WN-Std, WN-Feather, WN-Nano, LSTM-1×16, LSTM-2×8, A2-Full, A2-Lite | MSE = 0 (bit-exact)      |
| Cross-ISA        | AVX2 → AVX-512   | Same 7 models (requires AVX-512 HW, `#[ignore]`)                   | ESR < budget (see below) |
| Cross-ISA        | AVX2 → VNNI-BF16 | WN-Std, WN-Nano (requires VNNI+BF16 HW, `#[ignore]`)               | ESR < budget × 10        |

### Per-Architecture ESR Budgets

`tests/isa_parity.rs:243-250`:

| Budget            | Value | Target                    | Rationale                                   |
| ----------------- | ----- | ------------------------- | ------------------------------------------- |
| `WN_ESR_BUDGET`   | 1e-3  | WaveNet cross-ISA ESR     | Conservative; f32 accumulation dominates    |
| `LSTM_ESR_BUDGET` | 1e-2  | LSTM cross-ISA ESR        | Recurrent accumulation amplifies ISA diff   |
| `A2_ESR_BUDGET`   | 1e-3  | A2 cross-ISA ESR          | Gate-fused path; identical across ISA       |
| VNNI-BF16 ×10     | —     | BF16 quantization penalty | Extra quant layer beyond AVX2 f32 precision |

### Running

```sh
# CI (AVX2 self-consistency only, always runs)
cargo test --release --test isa_parity

# Full matrix (requires AVX-512 + VNNI-BF16 hardware)
cargo test --release --test isa_parity -- --ignored --test-threads=1 --nocapture
```

Self-consistency tests (8 tests, non-ignored) assert `MSE = 0.0` bit-exact output
between two independent AVX2 executions. Cross-ISA tests assert ESR within
calibrated budgets.

Kernel-level scalar-vs-SIMD parity is covered by unit tests (`gemv_test.rs`,
`dot_4x/8x/16x_test.rs`, `proptest_math.rs`). The ISA parity suite adds
end-to-end model-level cross-ISA coverage.

---

## RT Telemetry & Diagnostic Metrics

**File:** `src/dsp/telemetry.rs:41` (`LatencyHistogram`)

32-bin exponential histogram (2⁵ ns to 2³⁶ ns), lock-free atomic bins.
Used for profiling inference latency in the production hot path.

- `record(duration_ns)` — RT-safe via `fetch_max` + `fetch_add` with Relaxed ordering
- `get_percentile(p)` — approximate percentile via cumulative count scan
- `get_exact_max()` / `take_exact_max()` — per-cycle max latency
- Polled by `src/standalone/rt_setup/telemetry.rs:181` every 100 cycles:
  P50/P99/exact-max reporting

Diagnostic flags (`src/standalone/rt_setup/telemetry.rs`): clipping detection
(`RT_STATUS_HAS_CLIPPED`, sample-peak only), DSP overloads, clock drift via
`drain_dropped_frames()`, silence/fading transitions, GC overflow, rate change
notification, and `NamDiagnostic::DeadlineExceeded` when inference budget is exceeded.

---

## Stress Signal Generators

**File:** `src/testing/stress.rs`

| Generator      | Duration | Components                                                                                                            |
| -------------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `v1` (line 40) | 42.7 ms  | Low-E harmonics + chirp 220→3520 Hz + impulse at 25%                                                                  |
| `v2` (line 92) | 5.0 s    | 6 sections: single note w/bend, power chord, palm mute, pinch harmonic + saw sweep, bass amp low-A, chord decay C-E-G |

Default sample rates: [44100, 48000, 88200, 96000, 192000].

---

## Published Baselines (t3k-mushra / A2Esr.tsx:19-38)

Empirical ESR measurements from the Tone3000 dataset (NAM models trained on real gear):

| Model           | Q1 ESR  | **Median ESR** | Q3 ESR  | Median dB    | Interpretation       |
| --------------- | ------- | -------------- | ------- | ------------ | -------------------- |
| NAM A1-Standard | 0.00218 | **0.00623**    | 0.01571 | **−22.1 dB** | Baseline 2024 "good" |
| NAM A2-Full     | 0.00114 | **0.00334**    | 0.00913 | **−24.8 dB** | State-of-art 2026    |

**Context:** These values compare trained NAM models vs analog hardware — they include
modeling error. Nam-rs comparing vs C++ reference should achieve ESR orders of magnitude
lower (1e-5 to 1e-7, ≈ −50 to −70 dB), since differences are purely implementation
numerical noise, not training error.

---

## Gate Calibration Policy

This policy governs how every threshold and gate in the project is derived, maintained,
and reviewed. It formalizes the methodology from [AC-5](TODO-findings.md)
("Methodology 'calibrate until it passes' inverts the purpose of the test") and
[AC-9](TODO-findings.md) ("The 'calibrate until it passes +
declare done' pattern recurred at the oracle level") of the project's correctness
audit. All gates in `tests/threshold_calibration.rs`, `tests/cpp_parity.rs`,
`tests/reference_oracle_f64.rs`, and `tests/common/validation.rs` must comply.

### Rule 1 — Derivation from Validated Reference

**Every threshold must derive from a metric whose validity has been independently
established.** A metric is "validated" when it has been corroborated by an external,
independent reference (e.g., NumPy f64 anchor for the f64 oracle; NAMCore C++ for
interop parity). No gate may be derived from a metric whose only evidence is
"the current output passes."

- **Correct:** `LSTM_ESR_LIMIT = 7.0e-3` — derived from `ESR(f32 prod vs f64 oracle, prewarm-paired) = 3.57e-3` with the f64 oracle independently anchored against NumPy f64 (ESR < 1e-12). Measured and validated.
- **Incorrect:** Setting a threshold to `current_esr × 1.5` without an external anchor proving the metric measures what it claims.

### Rule 2 — Never Exceed the Placebo Line

**Fidelity gates must never exceed the project's placebo boundary: ESR < 1.0,
MR-STFT < 0.5.** A gate with ESR ≥ 1.0 or MR-STFT ≥ 0.5 is a placebo — it
cannot catch any regression because it sits above the signal's own energy
or above the point of total spectral collapse.

- **Enforced by:** `test_all_thresholds_anti_placebo` in `tests/threshold_calibration.rs:264` (per-model gates) and `test_oracle_gates_below_placebo_threshold` in `tests/threshold_calibration.rs:341` (oracle gates).
- **No per-model carve-outs permitted.** The anti-placebo rules apply uniformly to all models without `starts_with(...)` exemptions. Architectural differences (e.g., LSTM recurrent drift) are accommodated through parametrized, measured, and documented bounds — never through blanket exemptions.
- **Historical violations (canonical examples of what NOT to do):** Placebo oracle gates `WAVENET_ESR_LIMIT = 3.5` / `LSTM_ESR_LIMIT = 1.8` (post-E7, pre-T8.2), and the LSTM carve-out in the MR-STFT anti-placebo rule that permitted `mrstft_max = 0.85` — both corrected in T8.2–T8.6.

### Rule 3 — Mandatory Measurement Provenance Comment

**Every calibrated threshold entry must carry a `// Measured:` comment** documenting:
the measured value, the conditions (sample rate, signal duration, prewarm),
and the margin applied to derive the limit. The format is:

```text
// Measured: <value> @ <conditions>  →  limit = <value × margin>
```

This requirement applies to Tier 1 calibrated thresholds in `tests/common/validation.rs`,
oracle ESR limits in `tests/common/constants.rs`, per-architecture caps in
`tests/cpp_parity.rs`, and any future calibrated gate. Entries without a provenance
comment are caught by `test_all_calibrated_entries_have_measurement_comments` in
`tests/threshold_calibration.rs:222`.

- **Example (correct):** `// Measured: 3.57e-3 ESR @ 48 kHz, 24k prewarm + 256 sweep → limit = 3.57e-3 × 2 = 7.0e-3`
- **Example (incorrect):** A threshold with no comment, or a comment like `// Relaxed from 0.1` without stating the measurement that justified it.

### Rule 4 — Relaxation Requires Link to Independent Measurement

**Any loosening of a gate requires linking to an independent measurement that
justifies it.** "The current output passes" is not a valid justification.
The justification must cite:

1. **What was measured** (metric, model, sample rate, signal).
2. **How the reference was validated** (external anchor, independent tool).
3. **Why the new limit still provides meaningful regression protection**
   (i.e., the limit − measured value margin is small enough to detect a real
   degradation).

- **Correct:** Relaxing `ABSOLUTE_ESR_CAP_LSTM` from 6.23e-3 to 0.08 based on `ESR(v2, 240k steps, 48 kHz, prewarm-paired vs oracle) = 3.57e-3`, with a 3× margin justified by the oracle's confirmed f64 correctness (external NumPy anchor). The limit 0.08 is still < 1.0 (non-placebo) and 3.57e-3 is documented as the steady-state floor.
- **Incorrect:** Iterating thresholds upward until CI turns green, as observed in T3.5 (MR-STFT 0.15→0.85 across 5 iterations).

### Rule 5 — Mandatory Sanity-Check on Metric Meaning

**Before declaring any gate "completed," the sum of modeled error sources
must be consistent with the total measured error** within a reasonable bound
(≤ 10× divergence). This ensures the metric being gated actually measures what
is claimed — not an architectural artifact masquerading as precision loss.

```text
Σ ΔESR(modeled sources: f16c + activation + accumulation) ≈ ESR(total measured)
```

If the ratio `ESR(total) / ΣΔESR(sources)` exceeds 10, the absolute number is
suspect and cannot be used as a gate until the root cause of the divergence is
understood and closed.

- **Historical example.** The pre-T8.2 oracle reported `ESR(WaveNet prod vs oracle) = 2.47` while the decomposition showed `ΣΔESR(all sources) = 9.60e-7` — a ratio of ~2.6 million ×. The ESR was dominated by **architectural divergence** (unmatched prewarm state between oracle and production), not precision loss. Deriving `WAVENET_ESR_LIMIT = 3.5` from the 2.47 was prevented by this rule; post-T8.2 the limit was corrected to 1e-12.
- **After T8.2/T8.3.** The prewarm-paired ESR (3.57e-3 for LSTM, 6.13e-14 for WaveNet, 4.28e-10 for A2) is now consistent with the decomposition (≤ 10× gap), confirming the corrected gates measure actual precision loss.

### Rule 6 — Independence Must Not Be Circular

**A reference oracle is only "independent" if it is validated against a ground
truth that is a separate codebase from the one it judges — and that
independence must be re-proven whenever either side changes.** A second
implementation written to _mirror_ the implementation it is supposed to check
provides no protection: a shared conceptual bug passes silently in both.

- **The trap (canonical example).** The S5 external anchor `validate_oracle_f64.py`
  was written to "match the Rust oracle layout" (shared flat buffer, transposed
  weight indexing). When the Rust oracle was buggy, the Python reproduced the
  _same_ bug, so `ESR(oracle vs anchor) < 1e-12` looked like proof of correctness
  while both were wrong. T8.2 fixed the Rust oracle and the hidden divergence
  surfaced (LSTM ESR jumped to 21.3).
- **The fix (T8.13).** The anchor must agree with a **third, independent code
  path** — the f32 **production engine** — not just with the oracle. Ground truth
  for "which layout is correct" is determined **empirically** (run production,
  see which reference it matches), never by reasoning or by copying the other
  implementation. After T8.13 the three paths agree mutually:
  production f32 ↔ Rust f64 oracle ↔ NumPy f64 reference.
- **Requirement.** Any change to the oracle (`src/testing/reference_oracle.rs`)
  **invalidates the anchors** and must be paired, in the same change set, with
  (a) regenerating the anchors via the independent script and (b) re-confirming
  `ESR(reference vs production) ≈ f32/f16c floor` — not merely `ESR(reference vs
  oracle) < 1e-12`. Anchor tests must never be left `#[ignore]`d without a
  tracked task to restore them.

### Rule 7 — Fix the Code, Not the Test's Scope

**When a tightened gate fails for a subset of inputs, the response is to fix the
code, raise the bound with a documented measurement (Rule 4), or formally record
an accepted limitation — never to silently remove the failing inputs from the
test.** Narrowing a test's scope until it passes is the same anti-pattern as
"calibrate until it passes" (AC-5/AC-9), in disguise.

- **The trap (canonical example).** T8.4 tightened `ABSOLUTE_ESR_CAP_LSTM` to
  0.08, which the LSTM v2 parity test failed at non-native sample rates
  (88.2/96/192 kHz). Those rates were quietly dropped from the running test and
  the code comment claimed they were "tested separately under `#[ignore]`" —
  but no such test existed. Coverage vanished behind a false comment.
- **The rule.** If a subset must be excluded, the exclusion must be: (1) **true**
  (the comment describes reality), (2) **justified** by a stated technical reason
  (e.g., "the comparison is architecturally unequal because nam-rs resamples to
  the model's native rate while the C++ reference runs at the raw rate"), and
  (3) **tracked** as a documented limitation in `TODO-findings.md`, with the
  measured numbers recorded so the gap stays visible.

### Policy Compliance Checklist

When introducing or modifying a calibrated gate:

- [ ] Rule 1: The metric is measured against an **independently validated reference** (external anchor, not self-referential).
- [ ] Rule 2: The gate is **below the placebo line** (ESR < 1.0, MR-STFT < 0.5). The anti-placebo meta-tests will enforce this.
- [ ] Rule 3: The threshold carries a **`// Measured:` comment** with the measured value, conditions, and margin.
- [ ] Rule 4: Any relaxation has a **link to the independent measurement** in the commit message or PR description.
- [ ] Rule 5: The **sanity-check** was performed (`Σ sources ≈ total` within 10×); if the divergence is larger, the gate is marked as provisional and not declared "done."
- [ ] Rule 6: If an oracle/reference changed, its **independence was re-proven against production** (a separate code path), and no anchor test is left `#[ignore]`d without a tracked restoration task.
- [ ] Rule 7: No failing input was **removed from a test to make it pass**; any exclusion is true, justified, and tracked as a documented limitation.

### Cross-References in Code

This policy is referenced in:

- `tests/threshold_calibration.rs` — meta-tests that enforce Rules 2, 3, and 5 at CI time.
- `tests/reference_oracle_f64.rs` — oracle ESR gates that comply with Rules 1, 2, 3, and 5 after T8.2/T8.3.
- `tests/common/validation.rs` — `get_calibrated_threshold()` and Tier 1 entries, which carry `// Measured:` comments (Rule 3).

---

## Quick Reference — File/Line Map

| Metric            | Implementation                        | Tests                                                                |
| ----------------- | ------------------------------------- | -------------------------------------------------------------------- |
| ESR (f32)         | `src/testing/perceptual.rs:51`        | `tests/common/metrics.rs:40`                                         |
| ESR (f64)         | `src/testing/reference_oracle.rs:201` | `tests/reference_oracle_f64.rs:67`                                   |
| MR-STFT           | `src/testing/perceptual.rs:126`       | `tests/common/validation.rs:264`                                     |
| ASR               | `src/testing/aliasing.rs:133`         | `src/testing/aliasing_test.rs:*`, `tests/spectral_fidelity.rs:34`    |
| Farina FR+THD     | `src/testing/spectral.rs:224`         | `src/testing/spectral_test.rs:108`, `tests/spectral_fidelity.rs:401` |
| THD+N AES17       | `src/testing/spectral.rs:456`         | `src/testing/spectral_test.rs:248`, `tests/spectral_fidelity.rs:441` |
| IMD SMPTE         | `src/testing/spectral.rs:579`         | `src/testing/spectral_test.rs:299`, `tests/spectral_fidelity.rs:461` |
| LUFS              | `src/testing/perceptual.rs:328`       | `src/testing/perceptual_test.rs:30`                                  |
| LRA               | `src/testing/perceptual.rs:397`       | `src/testing/perceptual_test.rs:218`                                 |
| True-Peak dBTP    | `src/testing/perceptual.rs:652`       | `src/testing/perceptual_test.rs:462`                                 |
| Combined Loudness | `src/testing/perceptual.rs:522`       | `src/testing/perceptual_test.rs:346`                                 |
| f64 Oracle        | `src/testing/reference_oracle.rs:232` | `tests/reference_oracle_f64.rs:67`                                   |
| Fidelity Report   | `tests/common/validation.rs:55`       | `tests/cpp_parity.rs`, `tests/golden_vectors.rs`                     |
| ISA Parity        | `tests/isa_parity.rs:144`             | `tests/isa_parity.rs:257`                                            |
| RT Telemetry      | `src/dsp/telemetry.rs:41`             | `src/dsp/telemetry.rs:114`                                           |
| Stress Signals    | `src/testing/stress.rs:40,92`         | `src/testing/stress_test.rs`                                         |

---

## References

- t3k-mushra: <https://github.com/tone-3000/t3k-mushra> (MIT license)
- A2Esr.tsx baselines: `src/testing/perceptual.rs` constants
- NAM A2 Technical Report (Atkinson 2023)
- ITU-R BS.1770-4: Algorithms to measure audio programme loudness
- EBU Tech 3342: Loudness Range
- AES Convention 108 (2000): Farina — Simultaneous measurement of impulse response and distortion with a swept-sine technique
- AES17: Measurement of digital audio equipment
- SMPTE RP 120: Intermodulation distortion measurements
- DAFx 2025: Sato & Smith — Aliasing-to-Signal Ratio (ASR)
- ICASSP 2020: Yamamoto, Song & Kim — Multi-Resolution STFT Loss (Parallel WaveGAN)

---

## Attribution

- `A2ESR_A1_STANDARD_MEDIAN`, `A2ESR_A2_FULL_MEDIAN`, and related constants are derived
  from published data in t3k-mushra/A2Esr.tsx (MIT-licensed).
- LUFS/LRA computation implements ITU-R BS.1770-4 and EBU Tech 3342.
- ESR formula follows the standard definition used in NAM literature (Wright et al., Applied Sciences 2020).
- MR-STFT loss follows Yamamoto, Song & Kim (ICASSP 2020) — Parallel WaveGAN.
- ASR follows Sato & Smith, DAFx 2025.
- Farina method follows AES Convention 108 (2000).
