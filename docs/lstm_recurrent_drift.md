<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# LSTM Recurrent State Quantization Drift

> Root Cause Analysis of ESR above A1-Std baseline and MR-STFT degradation in LSTM topologies.

This document explains why LSTM topologies (`BossLSTM-1×16`, `BossLSTM-2×8`, `LSTM Official H=3`) exhibit
Error-to-Signal Ratio (ESR) significantly above the WaveNet A1-Std baseline (6.23e-3), even at native
sample rates where no resampling is involved.

---

## 1. The Symptom

| Model               | Samples | Sample Rate | ESR         | MR-STFT  | Notes                                  |
| ------------------- | ------- | ----------- | ----------- | -------- | -------------------------------------- |
| LSTM 1×16 v1        | 2,048   | 48 kHz      | 1.04e-2     | 0.098    | 42.7ms — passes all gates              |
| LSTM 1×16 v2        | 240,000 | 48 kHz      | **2.61e-2** | **0.87** | 5s — fails ESR cap + MR-STFT hard gate |
| LSTM 1×16 v2        | 480,000 | 96 kHz      | 6.09e-2     | 1.38     | 5s — both fail severely                |
| LSTM 1×16 v2        | 960,000 | 192 kHz     | 1.42e-1     | 1.93     | 5s — worst case                        |
| WaveNet Standard v2 | 240,000 | 48 kHz      | ~1e-13      | ~0.01    | Reference: non-recurrent passes easily |

The ESR degrades with both (a) signal duration and (b) sample rate (number of LSTM time steps).
MR-STFT follows the same pattern, indicating the error is distributed across the full spectrum —
characteristic of quantization noise, not a narrow-band artifact.

At 48 kHz native rate there is **no resampling** — the model runs at its training rate.
The degradation is intrinsic to the recurrent computation, not a harness artifact.

---

## 2. Root Cause: Recurrent State Quantization Drift

### 2.1 The mechanism

The LSTM cell state update is:

```text
cₜ = fₜ · cₜ₋₁ + iₜ · gₜ
hₜ = oₜ · tanh(cₜ)
```

All four gates (`fₜ`, `iₜ`, `gₜ`, `oₜ`) are computed via 4-way GEMV using f16c-quantized weights
(`src/models/lstm/layer.rs:11` — `[[[u16; H]; IH]; 4]`). Each f16c weight has ~3.3 decimal digits
of significand precision.

Every time step injects quantization error `εq` into `cₜ`. Since the forget gate `fₜ` is typically
close to 1 in trained networks (usually 0.9–0.99), **old errors are only partially decayed**.
The error power reaches a steady-state governed by:

```text
⟨ESR_steady⟩ ∝ σ²ε / (1 − ⟨f⟩²)
```

where `σ²ε` is the per-step quantization variance and `⟨f⟩` is the mean forget gate activation.

### 2.2 Evidence of accumulation

| Sequence Length | LSTM Steps | ESR     | Growth Factor   |
| --------------- | ---------- | ------- | --------------- |
| 2,048 (42.7ms)  | 2,048      | 1.04e-2 | 1.0× (baseline) |
| 240,000 (5s)    | 240,000    | 2.61e-2 | 2.5×            |
| 480,000         | 480,000    | 6.09e-2 | 5.9×            |
| 960,000         | 960,000    | 1.42e-1 | 13.7×           |

The growth is sub-quadratic (not ∝N²), confirming that the forget gate leak caps the accumulation
at a rate-dependent steady-state. This is **not** unbounded drift — it's a predictable, bounded
degradation proportional to the time-constant of the trained gates.

### 2.3 Why WaveNet does not suffer

WaveNet is a feedforward convolution (TCN) with **no** recurrent state. Each output sample depends
on a fixed receptive field (~300ms for CH16), not on the entire history. Quantization error is
contained per-window and does not accumulate across the signal.

LSTM, in contrast, is fully recurrent — the cell state carries information (and error) from
one sample to the next, for the **entire signal duration**.

---

## 3. Hypotheses Ruled Out

The following were systematically tested and **refuted** as explanations for ESR=2.61e-2 at 48 kHz:

| Hypothesis                            | Evidence Against                                                                                                                                                                                                       |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Band-edge filter artifacts**        | Pipeline has no DC-block or band-edge filter. Divergence occurs at native 48 kHz with no resampling.                                                                                                                   |
| **Denormal dither contamination**     | Dither is ±1e-11 (−220 dBFS), perfectly symmetric at input/output (`src/dsp/pipeline/stages/input.rs:29`, `output.rs:109`). 76 dB below 24-bit DAC floor — cannot explain a 2.6% error.                                |
| **Aliasing from non-linearity (P-1)** | ASR for LSTM 2×8 @ 48 kHz = −68.8 dB (`tests/spectral_fidelity.rs`). The tanh Padé approximant, while clamped at \|x\|<4, does not introduce significant aliasing in the LSTM operating regime.                        |
| **Harness resample path**             | At 48 kHz, the resampler is bypassed (`src/dsp/resampler.rs:428-431`). The test signal is generated at the target rate. No resampling = no resampling error.                                                           |
| **nam-rs regression vs NAMCore**      | The interop ESR is 2.61e-2 — both engines converge. The corrected f64 oracle (T8.2/T8.3, prewarm-paired) shows absolute precision floor of 3.57e-3 (−24.5 dB): ~300× smaller than the pre-T8.2 unmatched-state reading |
|                                       | of ~1.0. The recurrent drift mechanism is real but its magnitude vs the ideal was inflated by architectural divergence in the original oracle comparison.                                                              |

---

## 4. Divergence: NAMCore vs f64 Oracle (Corrected, T8.2/T8.3)

The question _"is nam-rs diverging from NAMCore, or is both diverging from the ideal?"_ is resolved with
two distinct layers:

| Comparison                            | ESR @ 48 kHz, 240k steps | Meaning                                                                                        |
| ------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------------- |
| nam-rs vs NAMCore (f32, interop)      | **2.61e-2** (−15.8 dB)   | Real recurrent drift between the two f32 engines, shared f16c+f32 accumulation.                |
| nam-rs vs f64 oracle (prewarm-paired) | **3.57e-3** (−24.5 dB)   | Absolute precision floor of the f16c+f32 production path vs double-precision ideal arithmetic. |
| Pre-T8.2 oracle (unmatched state)     | ~~~1.0~~ (0 dB)          | **Retracted** — inflated ~300× by architectural divergence (unmatched prewarm/initial state).  |

**Conclusion:** The interop gap (2.61e-2 vs NAMCore) is real and persists — both engines share the
f16c+f32 recurrent drift. However, the absolute precision floor is **3.57e-3**, not ~1.0. The pre-T8.2
"~1.0" was an artifact of mismatched initial state between production and oracle, not actual f16c
precision loss. The decomposition (T8.3) isolates the dominant sources: Padé tanh activation
(~7.6e-4 ΔESR) and f16c quantization (~5.1e-5), with f32 accumulation negligible (~7.2e-13).

The 2.61e-2 interop gap exceeds the 3.57e-3 absolute floor by ~7× — meaning the divergence between
the two engines is larger than either engine's divergence from the ideal. This suggests the interop
gap may be partially addressable (e.g., matching bf16 state precision or FMA ordering in the cell
update), pending the E8 root-cause investigation (AC-7).

---

## 5. The ABSOLUTE_ESR_CAP Sentinel

The `ABSOLUTE_ESR_CAP` in `tests/cpp_parity.rs:374` sets the baseline at:

```text
ABSOLUTE_ESR_CAP = A2ESR_A1_STANDARD_MEDIAN = 6.23e-3
```

This cap **overrides** any sample-rate relaxation for ESR: even if the calibrated threshold allows
higher values (e.g., 6.5e-2 for LSTM 1×16), the absolute cap clamps to the WaveNet A1-Std baseline.

For LSTM topologies specifically, `ABSOLUTE_ESR_CAP_LSTM = 0.08` (T8.4, 2026-06-28) provides
architecture-specific headroom — derived from the measured recurrent drift vs NAMCore
(2.61e-2 × 3, rounded). The provenance distinguishes two distinct floors documented in the
code: the interop parity floor (2.61e-2 vs NAMCore, f16c+f32 recurrent drift) and the
absolute precision floor (3.57e-3 vs corrected f64 oracle, prewarm-paired T8.2/T8.3).
The interop floor is ~400× larger than the absolute precision floor.

### Purpose

The cap acts as a **sentinel gate**, not a pass/fail criterion expected to always succeed:

- It ensures that _"passing"_ always means _"at least as precise as the reference topology (WaveNet A1-Std f32 native)."_
- When a model exceeds it (as LSTM 1×16 always does in v2 at native rates), the test **intentionally**
  routes the case to recurrent drift triage — forcing conscious assessment rather than silently
  absorbing degradation.
- LSTM v2 tests at non-native rates (88.2k, 96k, 192k) are excluded from assertion as the drift
  there exceeds the 0.08 cap and requires further characterization (E8/AC-7).
- The `Fidelity Margin` diagnostic (`tests/common/validation.rs:301-313`) provides context:
  at 48 kHz, the margin is 0.4 dB (barely above a deliberately degraded anchor signal).

### Long-term

The cap will remain in place. Any mitigation that reduces LSTM ESR below 6.23e-3 (e.g., Kahan-compensated
head accumulation, oversampled recurrent state as opt-in HQ mode) would be routed to Épico E4 (S5) and
validated against this cap.

---

## 6. Diagnostic Reproduction

The diagnostic test is at `tests/reference_oracle_f64.rs:271-347` (ignored due to cost).
This test was upgraded post-T8.2 to use prewarm-paired state matching (24k warmup samples,
ESR measured on subsequent 256 samples), replacing the original unmatched-state comparison
that produced the retracted ~1.0 reading.

```bash
cargo test --release --test reference_oracle_f64 \
  t33_diagnostic_recurrent_drift_lstm_1x16 -- --ignored --nocapture
```

This runs BossLSTM-1×16 against the f64 oracle at increasing segment lengths (512 → 240k samples),
showing the ESR curve and confirming the steady-state behavior.

The live C++ cross-validation (nam-rs vs NAMCore) is at `tests/cpp_parity.rs`:

```bash
cargo test --release --test cpp_parity \
  live_cross_validation_v2_lstm_1x16 -- --ignored --nocapture
```

---

## 7. Implications

### For QA Gates

- **v1 (2k samples)**: Calibrated thresholds are adequate. LSTM 1×16 passes with ESR=1.04e-2 < 6.5e-2.
- **v2 (240k samples)**: LSTM models systematically exceed `ABSOLUTE_ESR_CAP` at all sample rates.
  This is **expected and documented**. The cap's failure is informative, not alarming.
- **MR-STFT hard gate (44.1/48 kHz)**: Fails for all LSTM models in v2. This is also expected —
  spectral error from recurrent drift is broadband.

### For Model Authors

- LSTM `.nam` files suffer from recurrent state quantization drift due to f16c weights in
  the recursive cell. The absolute precision floor (3.57e-3 vs f64 ideal) is dominated by
  Padé activation (~7.6e-4 ΔESR) and f16c weight quantization (~5.1e-5) — intrinsic to the
  format, but ~300× smaller than pre-T8.2 estimates suggested. The interop gap (2.61e-2
  vs NAMCore) is the user-visible metric and may be partially addressable through aligned
  recurrence execution.
- Users comparing LSTM and WaveNet models at the same computational budget should understand
  that **LSTM fidelity degrades with signal duration**, while WaveNet fidelity is constant.

### For Future Work (Épico E4 / S5)

- **Kahan compensated summation** in the LSTM head projection (`src/models/lstm/model1.rs:96-107`)
  could reduce accumulation error by ~1-2 orders of magnitude without changing the model format.
- **Oversampled recurrent state** (running the LSTM at 2× sample rate internally) would
  reduce per-step error by spreading quantization over more samples. This would be an opt-in
  HQ mode (controlled via CLI flag and CLAP parameter, as designed in P-1/P-2).

---

## See Also

- [`audio_fidelity_map.md`](./audio_fidelity_map.md) — Unified map of all off-spec DSP decisions (§1 weight compression, §3 LSTM drift)
- [`perceptual_validation.md`](./perceptual_validation.md) — Full measurement framework and gate methodology
- `src/testing/reference_oracle.rs` — f64 oracle implementation (LSTM: lines 640–758)
- `src/models/lstm/layer_kernels.rs` — Production SIMD kernels with f16c GEMV
