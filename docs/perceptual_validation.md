<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

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

### Per-Model Calibrated Thresholds

Defined in `tests/common/validation.rs:409` (`get_calibrated_threshold`).
Fine-tuned thresholds per architecture, channel count, and model family.

| Model            | Ch  | ESR max | SNR min | Gate Type   |
| ---------------- | --- | ------- | ------- | ----------- |
| WaveNet Standard | 16  | 3.0e-11 | 105 dB  | Golden only |
| WaveNet Feather  | 8   | 1.0e-10 | 100 dB  | Golden only |
| WaveNet Nano     | 4   | 3.0e-10 | 95 dB   | Golden only |
| A2-Full          | 8   | 8.0e-8  | 70 dB   | Golden      |
| A2-Lite          | 3   | 6.0e-9  | 80 dB   | Golden      |
| WaveNet Official | 3   | 3.5e-2  | 14 dB   | Live parity |
| LSTM 1×16        | 1   | 6.5e-2  | 12 dB   | Live parity |
| LSTM 2×8         | 2   | 2.0e-2  | 18 dB   | Live parity |
| LSTM Official    | 1   | 6.0e-3  | 22 dB   | Live parity |
| WaveNet Lite     | 12  | 3.5e-11 | 105 dB  | Golden only |
| Linear           | —   | 1e-10   | 140 dB  | Both        |

### Conservative Parity Gate

```text
NAM_RS_CPP_PARITY_ESR_MAX = 1e-3 (≈ −30 dB)
```

~6× (~8 dB) below A1-Standard median (6.23e-3). Deliberately loose to tolerate
variation in less-covered sample rates (44.1k, 192k) while catching major
implementation regressions.

---

## MR-STFT — Multi-Resolution STFT Loss (Soft Spectral Gate)

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
transient errors simultaneously. This metric correlates more strongly with
perceived audio quality than ESR alone.

### Soft Gate

`MRSTFT_SOFT_THRESHOLD = 0.15` (`tests/common/validation.rs:247`). Informational
only — not a hard assertion. Values above 0.15 suggest spectral artifacts warrant
investigation.

FFT via `crate::math::dsp::fft::FftPlanner` (native, SoA, zero-alloc). Purely
scalar (non-RT), suitable for test validation.

Golden cross-check: `tests/fixtures/mrstft_golden.bin` generated by
`tests/fixtures/scripts/gen_mrstft_golden.py` (Python reference). When the
golden file is present, `test_mr_stft_parity_with_python` in
`tests/common/perceptual.rs` validates bit-parity within 1e-6 absolute tolerance.

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

Tests: `src/testing/perceptual_test.rs:30-170`.

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
| C++ NAMCore (f32) | Parity   | Implementation agreement (shared f32) | ESR < 1e-3 (loose)     |
| f64 Oracle        | Absolute | Intrinsic quality loss from f32 path  | Varies by architecture |

The parity reference answers "Is our f32 code compatible with upstream?" The
absolute reference answers "How much quality did we lose by using f32?"

Tests: `tests/reference_oracle_f64.rs:67-268`.

---

## Fidelity Report — Multi-Metric Pass

**File:** `tests/common/validation.rs:58` (`report_dsp_fidelity`)

Single-pass multi-metric report for golden vector and parity validation.
Computes in one traversal over reference/test signals:

| Metric           | Computation                                     | Gate / Target            |
| ---------------- | ----------------------------------------------- | ------------------------ |
| MSE              | noise_power / n                                 | < mse_limit              |
| MAE              | max absolute difference                         | informational            |
| SNR              | 10 · log₁₀(signal_power / noise_power)          | > min_snr_db             |
| PSNR             | 10 · log₁₀(peak_ref² / mse)                     | informational            |
| Equivalent Bits  | −0.5 · log₂(mse / signal_avg_power)             | informational            |
| ESR              | noise_power / signal_power                      | < max_esr (primary gate) |
| MR-STFT          | multi-resolution spectral loss                  | < 0.15 (soft)            |
| LUFS (reference) | integrated loudness gate                        | [−50, +10] plausibility  |
| dBTP (reference) | true-peak measurement                           | informational            |
| Anchor SNR       | SNR of test against 3.5 kHz 1-pole LP reference | degradation baseline     |
| Fidelity Margin  | SNR − anchor_SNR                                | > 8.0 dB target          |

`Fidelity Margin` quantifies how much better the test signal matches the reference
than a degraded anchor (3.5 kHz low-pass). Target > 8.0 dB ensures meaningful
fidelity above a low-quality baseline.

Variant `report_dsp_fidelity_no_lufs` (`tests/common/validation.rs:85`) skips
LUFS gate for cases where high signal amplitude is legitimate (e.g., IR convolution).

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

## Quick Reference — File/Line Map

| Metric            | Implementation                        | Tests                                                                |
| ----------------- | ------------------------------------- | -------------------------------------------------------------------- |
| ESR (f32)         | `src/testing/perceptual.rs:51`        | `tests/common/metrics.rs:40`                                         |
| ESR (f64)         | `src/testing/reference_oracle.rs:201` | `tests/reference_oracle_f64.rs:67`                                   |
| MR-STFT           | `src/testing/perceptual.rs:126`       | `tests/common/validation.rs:245`                                     |
| ASR               | `src/testing/aliasing.rs:133`         | `src/testing/aliasing_test.rs:*`, `tests/spectral_fidelity.rs:34`    |
| Farina FR+THD     | `src/testing/spectral.rs:224`         | `src/testing/spectral_test.rs:108`, `tests/spectral_fidelity.rs:401` |
| THD+N AES17       | `src/testing/spectral.rs:456`         | `src/testing/spectral_test.rs:248`, `tests/spectral_fidelity.rs:441` |
| IMD SMPTE         | `src/testing/spectral.rs:579`         | `src/testing/spectral_test.rs:299`, `tests/spectral_fidelity.rs:461` |
| LUFS              | `src/testing/perceptual.rs:328`       | `src/testing/perceptual_test.rs:30`                                  |
| LRA               | `src/testing/perceptual.rs:397`       | `src/testing/perceptual_test.rs:218`                                 |
| True-Peak dBTP    | `src/testing/perceptual.rs:652`       | `src/testing/perceptual_test.rs:462`                                 |
| Combined Loudness | `src/testing/perceptual.rs:522`       | `src/testing/perceptual_test.rs:346`                                 |
| f64 Oracle        | `src/testing/reference_oracle.rs:232` | `tests/reference_oracle_f64.rs:67`                                   |
| Fidelity Report   | `tests/common/validation.rs:58`       | `tests/cpp_parity.rs`, `tests/golden_vectors.rs`                     |
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

---

## Attribution

- `A2ESR_A1_STANDARD_MEDIAN`, `A2ESR_A2_FULL_MEDIAN`, and related constants are derived
  from published data in t3k-mushra/A2Esr.tsx (MIT-licensed).
- LUFS/LRA computation implements ITU-R BS.1770-4 and EBU Tech 3342.
- ESR formula follows the standard definition used in NAM literature (Yamamoto et al. 2020).
- ASR follows Sato & Smith, DAFx 2025.
- Farina method follows AES Convention 108 (2000).
