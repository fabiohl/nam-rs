<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# FastMath: Transcendental Function Approximations

Architectural decisions, experiment results, and normative guidelines for approximating `tanh`, `sigmoid`, and related functions in the NAM-rs DSP hot-path.

> [!IMPORTANT]
> This document records **definitive decisions** validated by benchmarks. Do not alter production choices without running `cargo bench` and confirming there is no statistically significant regression (p < 0.05).

---

## 1. Production Decision: Tanh — Padé [5,4] with Hardware Division

### Production Function

```text
tanh(x) ≈ x · (x² + 105) · (x² + 945) / ((15x² + 420) · x² + 945)
```

Implemented in `src/math/activations/tanh.rs`:

- `simd_tanh_avx2(x: __m256)` — 8 floats, AVX2 + FMA
- `simd_tanh_dual_avx2(x1, x2: __m256)` — 16 floats, coefficients broadcast once
- `simd_tanh_avx512(x: __m512)` — 16 floats, AVX-512

### Solution Characteristics

| Property                                      | Value                                    |
|:--------------------------------------------- |:---------------------------------------- |
| Maximum absolute error in [-4, 4]             | ~2.32e-3                                 |
| Equivalence in mantissa bits                  | ~8.7 bits                                |
| SIMD operations (AVX2, 8 elem)                | ~9 ops                                   |
| Throughput `tanh_slice` (256 elem, AVX2)      | **~54 ns**                               |
| Throughput `tanh_slice` (256 elem, piecewise) | ~~163 ns~~                               |
| Gain vs. 7-segment piecewise (Epic 8)         | **−66.6%**                               |
| Coefficients                                  | `PADE_TANH_*` in `src/math/constants.rs` |

### Why `_mm256_div_ps` and not Newton-Raphson Iteration (NR2)?

Empirical experiment (E8.T04, 10M samples in [-4, 4]):

| Variant                    | Max Abs Err | RMS Error   | Throughput (256 elem) |
|:-------------------------- |:----------- |:----------- |:--------------------- |
| 7-seg Piecewise (E8.T02)   | 4.90e-3     | —           | ~163 ns               |
| Padé NR2 (rcp + 2× Newton) | 2.32e-3     | ≈ Div       | ~104 ns               |
| **Padé Div (hw div)**      | **2.32e-3** | **minimum** | **~63 ns**            |

The error ratio between NR2 and Division is **1.000×** — the double Newton-Raphson iteration **fully saturates** the f32 mantissa (24 bits). The reciprocal contributes no measurable drift. Therefore, `_mm256_div_ps` is the correct choice: simpler, faster, and technically equivalent in precision to NR2.

> [!NOTE]
> The intuition that `div_ps` is "slow" comes from older architectures. In modern microarchitectures (Intel Ice Lake, AMD Zen 3+), `_mm256_div_ps` has a latency of 10-14 cycles and a throughput of 1 per 5 cycles — lower than a cascade of 6 `blendv_ps` required by the piecewise alternative.

---

## 2. Failed Experiment: Piecewise 7-Segment for Tanh (E8.T02)

### What was tried

Replacing the Padé [5,4] with 7 polynomials of degree 5 with branchless blending via `_mm256_blendv_ps`, covering the [-4, 4] domain with variable-width segments.

### Original Motivation

Hypothesis: short segments allow coefficients with lower Chebyshev error per segment, improving global precision.

### Measured Results (Epic 8)

| Metric                       | Padé [5,4] (baseline) | 7-seg Piecewise (E8.T02)     |
|:---------------------------- |:--------------------- |:---------------------------- |
| SIMD operations              | ~9                    | **~28** (7 polys + 6 blends) |
| Max error [-4, 4]            | 2.32e-3               | **4.90e-3** (worse!)         |
| Throughput (256 elem)        | 63 ns                 | **163 ns** (+159%)           |
| `Prewarm_LSTM_2x16_2048samp` | baseline              | **+16%** regression          |

### Why it failed

1. **All 7 polynomials are evaluated unconditionally** (branchless). The cost does not depend on the input value — it is always 7× the cost of a single polynomial.
2. **Cascade of 6 `blendv_ps`** serializes Port 5 (shuffle unit), creating a sequential dependency bottleneck.
3. **Tanh has maximum curvature in [0, 1]** — smaller segments in this region improve local error, but coefficients obtained without `fpminimax` (Sollya) are not optimal.
4. **Conclusion:** Piecewise only outperforms Padé if it has ≤3 segments AND coefficients are recomputed via `fpminimax`. For f32, the cost of 7 segments never pays off.

> [!CAUTION]
> **Never replace the production `simd_tanh_avx2` path with a piecewise variant without benchmarking the LSTM prewarm (2048 samples).** The LSTM evaluates tanh 4x per cell per timestep — throughput errors scale linearly with depth and block size.

### Current Status

The piecewise implementation is preserved as `simd_tanh_piecewise_avx2` (`#[allow(dead_code)]`) for future research. If resumed, coefficients should be recomputed via:

```text
sollya> fpminimax(tanh(x), [|1,3,5|], [|SG...|], [a, b], floating, absolute);
```

for each segment, where `[a, b]` is the segment interval.

---

## 3. Production Decision: Sigmoid — Direct Minimax (Degree 17)

### Foundation

The previous implementation used the identity `σ(x) = 0.5 + 0.5 · tanh(x/2)`, propagating the tanh error and adding scaling operations.

### Adopted Solution (E8.T01)

Odd polynomial of degree 17 (9 terms) for the [-8, 8] domain, coefficients obtained via **Lawson's algorithm** (weighted minimax).

| Property                 | Tanh identity (baseline) | Direct Minimax (E8.T01)     |
|:------------------------ |:------------------------ |:--------------------------- |
| Max absolute error       | ~6.8e-4                  | **~4.09e-4** (1.67× better) |
| SIMD operations          | 16                       | **15**                      |
| Scalar throughput (LSTM) | baseline                 | **−20.25%**                 |

### Lesson Learned

Smooth symmetric functions on a compact domain (sigmoid in [-8, 8]) are better approximated by a single polynomial of suitable degree than by segments. Segmentation only pays off when there are sharp curvature changes or discontinuities.

---

## 4. Discovery: Newton-Raphson Reciprocal Adds No Drift in f32

Experiment E8.T04 empirically proved that **the double Newton-Raphson iteration (NR2) in the Padé division fully saturates the 24-bit f32 mantissa**. The maximum absolute error of NR2 vs. hardware division is a 1.000× ratio — indistinguishable within representable precision.

**Normative implication:** Wherever `rcp_ps + 2× Newton-Raphson` is used in the codebase to approximate a rational Padé division, it can be replaced with `div_ps` with no precision penalty and a potential throughput gain (fewer dependencies, lower register pressure).

---

## 5. On WaveNet vs. C++ and BF16 Drift

Parity comparison with NeuralAmpModelerCore revealed that WaveNet Standard SNR remains at **~9.5 dB** regardless of precision improvements in activations. This is because:

1. **The dominant drift is BF16 weight quantization** — u16 weights are converted to bf16 upon loading, introducing a rounding error of ~3.9e-3 per weight.
2. Improvements in tanh/sigmoid reduce **activation** drift (background), but **weight** drift is structurally larger.

### Hierarchy of Drift Sources (largest to smallest)

```text
1. BF16/F16 weight quantization                      (~3.9e-3 per element)
2. Tanh/Sigmoid activation approximation error      (~2.3e-3 Padé, ~4.9e-3 piecewise)
3. Floating-point accumulation (deep conv)           (O(N·ε), mitigated by Kahan)
4. Reciprocal in Padé division                       (≈0, saturated in 24 bits)
```

### Measured: Drift Source Decomposition (S1.T1.4)

Using a self-contained scalar reference engine (f32 weights + exact `f32::tanh`) for WaveNet Standard (CH=16, K=3, HEAD=8, 10+2 layers) with synthetic weights 0.01 and conv1d biases 0.001:

| Source                              | ESR (linear) | ESR (dB)  | Dominance |
|:------------------------------------|:-------------|:----------|:----------|
| (a) F16 weight quantization         | 3.24e-7      | −64.9     | **100%**  |
| (b) tanh Padé [5,4] approximation   | 8.49e-15     | −140.7    | ~0%       |
| (c) f32 accumulation (residual)     | ~0           | −∞        | ~0%       |
| **Total (a+b+c)**                   | **3.24e-7**  | **−64.9** | —         |

**Key findings:**

1. **Weight quantization dominates entirely** — F16 rounding accounts for essentially 100% of the ESR against the full-precision reference.
2. **Tanh Padé contribution is negligible** for this topology. The small synthetic weights (0.01) keep internal activations in the linear region of tanh where `tanh(x) ≈ x` with negligible error. This does NOT imply Padé is harmless for real NAM models — real weights are larger and produce activations with `|x| > 1` where Padé error (~2.32e-3 max) becomes significant.
3. **f32 accumulation error is below measurement noise** — the residual after subtracting quantization and tanh components is effectively zero, confirming that the existing Kahan-compensated primitives are sufficient.

**Recommendation (P2):** Sprint S4's exact mode should prioritize higher-precision weight storage (f32 or compensated F16) over improving the tanh approximation, since the measured weight-quantization ESR dominates by >8 orders of magnitude for this topology. However, a follow-up measurement with real model weights (which produce larger activation ranges) is needed to quantify the tanh contribution under realistic conditions.

### Path to Improving SNR

E8.T03 implements bias-tuning for BF16 (compensation at model load). The expected gain is ≥1.5 dB SNR on BF16-capable hardware (Intel Sapphire Rapids, AMD Zen 5+). To validate, a CI runner with a compatible CPU is required.

---

## 6. Anti-Subnormal Prevention with DC Dither (E8.T05)

### Problem

During fade-out/silence, near-zero values in LSTM/WaveNet activations enter the subnormal territory (< 1.175e-38 for f32). Subnormals have a high processing cost (hardware soft emulation) and can introduce "digital click" artifacts during fades.

### Adopted Solution

The constant `DENORMAL_DITHER_OFFSET = 1.0e-11` (-220 dBFS) is injected in `apply_input_stage` and removed in `apply_output_stage`.

- **76 dB below** the noise floor of a 24-bit DAC — completely inaudible.
- **Zero performance overhead** (2 trivial loops per frame, eclipsed by GEMV).
- Guarantee: no subnormal reaches activations during decay.

> [!TIP]
> If a future model exhibits "clicks" or "pops" at the output during prolonged silence, first verify that `DENORMAL_DITHER_OFFSET` is being applied correctly to the affected channel.

---

## 8. WaveNet Non-Zero Silence Policy (S1.T1.3)

### Phenomenon

With silence input, WaveNet produces a residual output of ~3.58e-5 (−89 dBFS).
The A2 architecture produces absolute zero under the same conditions.

### Root Cause (confirmed by decomposition test)

The residue is **not a bug**. It originates from the **conv1d bias** terms
(bias = 0.001 for each layer, 12 layers across both arrays):

1. `tanh(bias) ≈ tanh(0.001) ≈ 0.001` — bias passes through gated activation
2. These non-zero activations accumulate across layers via `one_by_one` dense projections
3. The final `head_scale` (0.1) scales the accumulated value to the output

Decomposition test (`tests/soak_test.rs:test_wavenet_silence_decomposition`):

- **Total residue:** 3.58e-5 (−89 dBFS)
- **Conv1D bias contribution:** 3.58e-5 (**100%** of total)
- **F16 quantization drift:** 0.0 (zero — 0 × weight = 0 regardless of rounding)

The A2 architecture only zeroes because it uses LeakyReLU(0)=0 with synthetic
weights — not the case for real WaveNet A1 models.

### Parity with C++ NAMCore

This is **faithful behavior** to NeuralAmpModelerCore v0.5.3:

> "Important: don't expect the model to be outputting zeroes after this. Neural
> networks don't know that there's anything special about 'zero', and forcing
> this gets rid of some possibilities (e.g. models that 'are noisy')."
> — `NAM/dsp.h:67` (tests/fixtures/NeuralAmpModelerCore/)

### Policy Decision

**Do NOT force the output to zero.** Forcing zero would:

1. Diverge from the C++ "bible" (NAMCore) — breaking parity
2. Eliminate legitimate noisy/saturated model behaviors

The interaction with noise-gate and true-bypass is the responsibility of the
gate layer (`src/dsp/gate.rs`), not the model inference path.

### DAZ/FTZ Coverage

Denormals-Are-Zero / Flush-To-Zero is active at all entry points to the
hot-path:

| Location                                    | Mechanism                    |
|:------------------------------------------- |:---------------------------- |
| `src/math/common/ops.rs:163`                | `set_daz_ftz()` helper       |
| `src/clap/processor/mod.rs:268`             | Reasserted every 1024 blocks |
| `src/standalone/rt_setup/thread.rs:72`      | Set at RT thread init        |

No denormal penalty is observed in the WaveNet hot-path. The 3.58e-5 residue
is a normal normalized f32 value.

### References

- `tests/soak_test.rs:53` — `test_wavenet_silence_soak` (`#[ignore]`, 10M frames)
- `tests/soak_test.rs:117` — `test_wavenet_silence_decomposition` (source isolation)
- `TODO-sprints.md:136` — T1.3 task definition
- `TODO-problemas.md:155` — P4 problem report

---

## 7. Summary of Normative Rules (Checklist)

For any future modification in `src/math/activations/`:

- [ ] **Benchmark `Prewarm_LSTM_2x16_2048samp`** — sensitive to tanh throughput. A regression > 5% is unacceptable.
- [ ] **Benchmark `FastMath_tanh_AVX2_256elem`** — validates the slice path micro-benchmark.
- [ ] **Do not use piecewise > 3 segments** without recomputing coefficients via Sollya `fpminimax`.
- [ ] **Do not replace `div_ps` with NR2** — both have the same precision in f32; NR2 is slower.
- [ ] **Maintain the `single` / `dual` separation** — dual must use shared coefficient broadcasts.
- [ ] **Verify symmetry** — tanh is an odd function. Any implementation must satisfy `f(-x) == -f(x)`.
- [ ] **Maximum error tolerance:** ≤ 5e-3 (tanh/sigmoid in LSTM inference), ≤ 1e-4 (sigmoid in initialization).
- [ ] **Run `cargo test --lib`** — all tests must pass without failure after any modification.

---

## Reference

- Kahan, W. "Further remarks on reducing truncation errors." *CACM*, 1965. (Kahan summation)
- Muller, J.-M. *Elementary Functions: Algorithms and Implementation*. 3rd ed. Birkhäuser, 2016. (Padé approximants)
- Intel® Intrinsics Guide — `_mm256_div_ps` latency/throughput per microarchitecture.
- [Sollya](https://www.sollya.org/) — tool for computing optimal `fpminimax` coefficients.
- `TODO-sprints.md` §Epic 8 — complete history of decisions and benchmark data.
