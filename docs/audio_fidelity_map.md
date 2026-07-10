<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Audio Fidelity Map — Off-Spec DSP Design Decisions

Every engineering decision in nam-rs that lies **outside the strict NAM model specification** and
influences audio fidelity, RT safety, or user experience is catalogued here. For each factor:
what it is, whether it is mandatory or optional, the sonic and performance impact, and where to
find the implementation.

The NAM specification defines only: topology (WaveNet, LSTM, A2, ConvNet), stored weight values,
and the mathematical forward pass. Everything below is a nam-rs implementation choice — not part
of the `.nam` / `.namb` file format contract.

---

## Quick Reference

| #   | Factor                                                      | Spec? | Mandatory?                         | User-Controllable?       | Quality Impact                                         | Status                |
|:---:|:----------------------------------------------------------- |:-----:|:----------------------------------:|:------------------------:|:------------------------------------------------------ |:---------------------:|
| 1   | **Weight compression (F16C/BF16)**                          | ❌    | Was under review — removed         | ❌ No                    | Removed (SQ5, 2026-07-06): LSTMs became faster; see §8 | Removed (SQ5)         |
| 2   | **Activation precision (Fast Padé / Standard exact-grade)** | ❌    | ✅ Default (Standard); Fast opt-in | ✅ CLI + CLAP            | −80…−97 dBFS (Fast) → ÷10,000 (Standard); ↓ aliasing   | ✅ Active             |
| 3   | **LSTM recurrent drift**                                    | ❌    | Partial (model-dependent)          | ✅ HF gates + Kahan head | ESR 2.6e-2 @48k → 1.4e-1 @192k (1×16); zeroed for 2×8  | ✅ Partially resolved |
| 4   | **Host sample rate resampler**                              | ❌    | ✅ When host ≠ 48 kHz              | ❌ No†                   | Passband ripple < 0.05 dB; stopband ≥ 25 dB            | ✅ Active             |
| 5   | **Neural stage oversampling**                               | ❌    | ❌ Off by default                  | ✅ CLI + CLAP            | Reduces aliasing; adds latency + CPU                   | ✅ Active             |
| 6   | **Denormal dither + FTZ/DAZ**                               | ❌    | ✅ Yes                             | ❌ No                    | No audible impact (−220 dBFS)                          | ✅ Active             |
| 7   | **Adaptive Compute (quality fallback)**                     | ❌    | ✅ Default                         | 🔶 `--slim` flag         | Silent quality drop under CPU load                     | ✅ Active             |

† Resampler quality (Standard/HQ) was rejected after benchmarking — see §4. Activation precision is exposed via CLI (`--activation fast|standard`) and CLAP (`PARAM_ACTIVATION=8`). Standard (exact-grade) is the universal default across all model families.

---

## 1. Weight Compression — F16C / BF16 *(REMOVED, 2026-07-06)*

This optimization was removed after the SQ5 PoC confirmed that f16c decompression overhead
outweighed any L1 cache benefit for LSTM models, and that the f16c quantization was the sole
cause of interop drift for several model topologies (see §8 — Histórico). The removal is the
production default. All models now use native f32 weights.

**Implementation.** All weight quantization/compression code paths in model loading (`src/models/*/set_weights.rs` or `src/models/*/model.rs`) have been removed. Weight tensors in all active models are loaded and processed natively as `f32` vectors. The conversion helper module (`src/math/common/half.rs`) and corresponding unused SIMD kernels are retained only as test references or benchmarks.

### 1.1 NAMCore parity — corrected

NAMCore does **not** quantize anything. All its weights are `Eigen::MatrixXf`/`VectorXf`
(native f32). The previous claim ("NAMCore also quantizes to f16c... both engines share the same
quantization floor") was **factually wrong** and has been retracted. With f16c removed, nam-rs
now matches NAMCore's native f32 representation for all weight matrices.

---

## 2. Activation Precision — Fast (Padé) vs Standard (exact-grade)

**What it is.** Neural models use non-linear activations. nam-rs replaces exact `f32::tanh` and
`f32::exp` with one of two selectable polynomial approximation modes, held in the
`ActivationPrecision` enum (`src/math/activations/mod.rs`) as a global atomic flag:

| Mode                   | Activation                                 | Max error               | Approx error (dBFS) | Cost            |
|:---------------------- |:------------------------------------------ |:-----------------------:|:-------------------:|:---------------:|
| **Fast**               | Padé [5,4] tanh, clamped \|x\| ≤ 4         | 2.32e-3                 | ≈ −53 dB            | Baseline        |
| **Fast**               | Minimax degree-17 sigmoid                  | 4.09e-4                 | ≈ −68 dB            | Baseline        |
| **Standard** (default) | Taylor-based exp kernels, degree-6 minimax | ~2.4e-7 (10,000× lower) | ≈ −133 dB           | +10–15% compute |

The Padé clamp at `|x| > 4` introduces a derivative discontinuity — a weak source of spectral
aliasing at high gain settings. The minimax sigmoid has no clamping discontinuity, and the
Standard kernels eliminate the clamp discontinuity entirely.

### 2.1 Fast mode — opt-in for CPU-constrained scenarios

`f32::tanh` on x86-64 costs ~150 ns per 256-element AVX2 vector via the scalar fallback path. The
Padé implementation runs in ~10 ns. For a large WaveNet model processing hundreds of thousands of
activations per 1.3 ms RT window, this can be the difference between real-time and not. Fast
(Padé/minimax) mode is an explicit opt-in (`--activation fast`) for CPU-constrained setups;
`Standard` (exact-grade polynomial exp-based) is the universal production default.

### 2.2 Standard mode — universal default

`ActivationPrecision::Standard` uses Taylor-based polynomial exp kernels for tanh/sigmoid
(max error ≈ 2.4e-7, ~10,000× lower than Fast). It is the universal default, active from
startup, and user-switchable at runtime — applied via CLI (`--activation standard|fast`) or
switched live (CLAP `PARAM_ACTIVATION=8`, at the block boundary, with GUI sync, persistence,
and an offline-render override forcing `Standard`) — via a single relaxed atomic store, with
no rebuild or allocation. **All model families — WaveNet (A1/A2), ConvNet, Linear, LSTM —
dispatch to both kernels**, including all LSTM gate paths (scalar, AVX2, AVX-512), completed
in Épico β. Full dispatch/topology matrix in
[`docs/fastmath-approximations.md`](fastmath-approximations.md) §10.

### 2.3 Interaction with oversampling

Standard (exact-grade) is most effective combined with 4× oversampling (§5): without OS, aliased harmonics
fold back into the baseband regardless of activation precision; with 4× OS, the half-band filter
removes those harmonics and exact-grade mode suppresses the residual high-order folding.

### 2.4 Precision context (validated post-S8)

With the f64 oracle confirmed correct (§3), the combined precision model (Padé activations + f32
accumulation) reproduces production to within ESR ≈ 1.04e-3 (LSTM, measured on `lstm.nam` H=3 —
see the provenance note in §3 below; validated on BossLSTM-1×16 at production duration with ESR = 2.61e-2),
8.62e-14 (WaveNet), 1.13e-13 (A2) — for the small official LSTM, the Padé (Fast) approximation
together with f32 precision fully explains the gap from ideal f64, with no unexplained "architectural" residue.
Fast (Padé) mode is the well-understood opt-in for CPU-constrained scenarios; Standard (exact-grade)
buys ~10,000× lower activation error as the universal default, audibly most relevant when paired with
oversampling (§5).

**Mandatory?** No. Standard (exact-grade) mode is the universal default. Fast (Padé) is opt-in via
`--activation fast` (CLI) or `PARAM_ACTIVATION` (CLAP).

**Implementation.** `src/math/activations/mod.rs`, `src/math/activations/tanh/production.rs`
(Fast: Padé/minimax), `src/math/activations/tanh/high_fidelity.rs` (Standard: exact-grade polynomial),
`src/math/activations/sigmoid/production.rs` (Fast), `src/math/activations/sigmoid/high_fidelity.rs`
(Standard). Full analysis in
[`docs/fastmath-approximations.md`](fastmath-approximations.md).

---

## 3. LSTM Recurrent State Drift

**What changed.** The f16c weight quantization was removed. The impact on LSTM interop fidelity varied by model:

| Model         | ESR vs NAMCore (pre-SQ5, f16c) | ESR vs NAMCore (post-SQ5, f32) | Conclusion                                 |
|:------------- |:------------------------------:|:------------------------------:|:------------------------------------------ |
| BossLSTM-1×16 | 1.04e-2                        | **1.04e-2**                    | Drift persists — f16c was NOT the cause    |
| BossLSTM-2×8  | 2.69e-3                        | **0.00e0**                     | Drift eliminated — f16c was the sole cause |

**Contrary to the previous assumption, the LSTM recurrent drift is not — or not only —
caused by f16c weight quantization.** BossLSTM-2×8 converged to bit-exact parity with NAMCore
once f16c was removed, proving that its drift at 48 kHz was entirely a quantization artifact. But
BossLSTM-1×16 showed the same 1.04e-2 ESR gap with NAMCore with or without f16c. The root cause
of this residual divergence was identified as the Padé activation approximation class losing calibration
for $|x| > 4$ (tanh) and $|x| > 8$ (sigmoid), which is exceeded by the high weight norms and pre-activation
magnitudes in larger hidden topologies like H=16.

**Standard (exact-grade) Universal Default.** As of Sprint 2 (Tarefa 1.2), `ActivationPrecision::Standard`
(polynomial exp-based exact math, max error ≈ 2.4e-7) is the **universal default** for all models —
LSTM, WaveNet (A1/A2), ConvNet, and Linear. The former per-model override that applied `HighFidelity`
only to the `BossLSTM` family was removed — activation precision is no longer model-specific.
With matched activations, the interop gap vs NAMCore collapsed to near-zero:

- **BossLSTM-1×16**: ESR vs NAMCore ≈ 1.42e-11 (-108.5 dB)
- **BossLSTM-2×8**: ESR vs NAMCore ≈ 1.67e-11 (-107.8 dB)

The historical per-sample-rate LSTM interop drifts (measured pre-SQ5 with f16c active) are preserved below for baseline reference only:

| Model     | 44.1 kHz | 48 kHz      | 88.2 kHz | 96 kHz  | 192 kHz     |
|:--------- |:--------:|:-----------:|:--------:|:-------:|:-----------:|
| LSTM 1×16 | 2.39e-2  | **2.61e-2** | 5.39e-2  | 6.09e-2 | **1.42e-1** |
| LSTM 2×8  | 3.41e-3  | **3.88e-3** | 1.18e-2  | 1.45e-2 | **4.20e-2** |

> **Note:** the 2×8 row applies to the *pre-removal* era; post-SQ5, BossLSTM-2×8 is bit-exact
> with NAMCore (ESR = 0.00e0) at 48 kHz in Standard mode, confirming exact-grade activation precision.

**F64 oracle floor — provenance correction.** The "3.57e-3" figure previously attributed to
BossLSTM-1×16 is actually measured on `lstm.nam` (H=3, the tiny official example, 256-sample
window) — see `test_oracle_lstm()` at `tests/reference_oracle_f64.rs:328`. Post-SQ5, this family
value was recalibrated to 3.41e-3. The model-specific f64-oracle floors (prewarm-paired, 24k prewarm

- 4096 acoustic samples) have been measured under Standard mode as:

- **BossLSTM-1×16**: ESR vs f64 oracle = **5.06e-2 (-13.0 dB)** (from `test_decomposition_boss_lstm_1x16`)
- **BossLSTM-2×8**: ESR vs f64 oracle = **1.73e-3 (-27.6 dB)** (from `test_decomposition_boss_lstm_2x8`)

**Mitigations carried forward.** The three mitigations shipped in Épico β remain active and
relevant for the residual drift in **BossLSTM-2×8**:

- **I6 — Standard (exact-grade) activations in LSTM gates** (primary): exp-based polynomial kernels
  (~2.4e-7 error vs ~2.32e-3 Padé/Fast) across scalar/AVX2/AVX-512 LSTM gate paths.
  *Cost:* +10–15% compute.
- **I4 — Kahan-compensated head accumulation**: compensated summation in the LSTM head
  projection (H→1). ~2 dB SNR gain in deep heads. *Cost:* negligible.
- **I5 — Oversampling ruled out for LSTM**: external oversampling reduces aliasing (8–11 dB) but
  **changes timbre drastically** (ESR > 1.0 vs Off) because the recurrent feedback delay is fixed
  in absolute samples — 2×/4× halves/quarters that window in seconds. **Not recommended** as a
  user control for LSTM, unlike WaveNet where oversampling is transparent.

Stateful ADAA for the recurrent cell (Holters 2019; Mikkonen & Werner 2025) was evaluated and
**not adopted** — conflicts with the polymorphic `dispatch_simd!` macro. Retained as a research
direction (`docs/research-references.md` R6, R7b).

**User impact.** For BossLSTM-1×16, the interop drift is gone — this model is now f32 native and
converges with NAMCore. For BossLSTM-2×8, degradation with signal duration and host sample rate
persists at pre-SQ5 levels. WaveNet models remain unaffected by recurrent drift.

**Implementation.** `src/models/lstm/layer_kernels.rs` (GEMV with f32 weights)

---

## 4. Host Sample Rate Adaptation (Polyphase Sinc Resampler)

**What it is.** NAM models are trained at 48 kHz. When a DAW host runs at a different rate
(44.1 kHz is the most common), nam-rs converts using a native minimum-phase polyphase FIR sinc
resampler (`NamResampler`, `src/dsp/resampler/mod.rs`).

**Configuration (post-S6).** 256 phases × 64 taps, Kaiser β=12 windowed sinc, minimum-phase
by default. A linear-phase variant is available as an internal option for offline use.

**When active.** Only when `host_rate ≠ 48 kHz`. When host == 48 kHz, a zero-cost bypass path
copies samples directly (no convolution). This is the common case for most dedicated audio
interfaces.

**Quality impact.**

| Metric                   | Value                                                                                                    |
|:------------------------ |:-------------------------------------------------------------------------------------------------------- |
| Passband ripple          | < 0.05 dB (to 0.45 × Nyquist)                                                                            |
| Stopband attenuation     | Filter design ≥ 105 dB; end-to-end multitone SNR ~31 dB (min-phase, interpolation-limited; gate ≥ 25 dB) |
| Group delay (min-phase)  | Asymmetric; slight high-frequency transient smearing                                                     |
| 44.1 kHz top-end rolloff | Post-S6 < 0.05 dB at 20 kHz (vs. −1.7 dB pre-S6)                                                         |

**Mandatory?** Yes, when the host rate differs from 48 kHz. Cannot be disabled without
breaking audio at non-48 kHz rates.

**User-controllable?** No. A "Resampler Quality: Standard/HQ" parameter was designed but discarded after quantitative benchmarks. The 64-tap HQ configuration is the permanent production default. Benchmarks showed that a 32-tap configuration saves only ~40 ns per 64-sample block (< 0.1% of the total pipeline) while severely degrading passband SNR from $\ge 100\text{ dB}$ to $\sim 24\text{ dB}$. Thus, Standard (32-tap) mode has been rejected to prioritize fidelity and avoid code complexity.

**Implementation.** `src/dsp/resampler/mod.rs`, `src/dsp/sinc_kernel.rs`.

---

## 5. Neural Stage Oversampling (HQ Mode)

**What it is.** Optional 2×/4× oversampling around the neural model to suppress aliasing from
non-linear activations (tanh, sigmoid, ReLU/LeakyReLU corner discontinuities). Based on
Kahles, Esqueda & Välimäki (JAES 2019).

**Architecture.** Each 2× stage uses a 25-tap Kaiser half-band FIR (β=12, >100 dB stopband).
The half-band property halves the non-zero taps, keeping the MAC budget manageable.
Pipeline: `upsample → model inference (at 2× or 4× rate) → downsample`.

| Mode          | Stages | Latency added                                | CPU overhead   |
|:------------- |:------:|:--------------------------------------------:|:--------------:|
| Off (default) | 0      | 0                                            | 0              |
| 2×            | 1      | 12 samples @ native rate (~0.25 ms @ 48 kHz) | ~2× model cost |
| 4×            | 2      | 24 samples @ native rate (~0.5 ms @ 48 kHz)  | ~4× model cost |

The latency above is tracked and reported to the host via `OversampleEngine::latency_samples()`
(post-Sprint B1). The CLAP plugin includes oversampling latency in both its initial latency
announcement and dynamic latency updates (`src/clap/processor/events.rs`, `mod.rs`).

**Mandatory?** No. Default is Off. Designed for offline/mixdown rendering and critical
listening where CPU headroom exists.

**User-controllable?** Yes, via:

- CLI: `--oversample off|2x|4x` (alias `--os`)
- CLAP: parameter ID=7, stepped enum, persisted in plugin state
- Changes trigger an off-RT rebuild (same SPSC path as model hot-swap)

**Quality impact.** Reduces ASR (Aliasing-to-Signal Ratio, Sato & Smith DAFx 2025). Measured
reduction is architecture-dependent; WaveNet benefits most from 4× (ReLU/tanh corner). Most
effective when combined with the Standard (exact-grade) activation mode (§2.2).

**Why ADAA was rejected.** Antiderivative Anti-Aliasing (Parker DAFx-16; Bilbao IEEE SPL 2017)
requires per-activation analytical antiderivatives, which conflicts with the polymorphic
`dispatch_simd!` macro and multi-architecture dispatch. The half-band approach is
activation-agnostic and transparent to the model dispatcher.

**Implementation.** `src/dsp/oversample.rs` (`OversampleEngine`, `OversampleFactor`),
`src/dsp/pipeline/stages/inference.rs` (integration point).

---

## 6. Denormal Prevention — Dither + FTZ/DAZ

**What it is.** Two complementary mechanisms prevent subnormal (denormal) floating-point numbers
from entering the neural model's internal state. Denormals trigger microcode assist on x86-64,
causing a 10–100× slowdown per affected operation — a real-time guarantee killer.

### 6.1 Deterministic Dither Offset

A fixed offset `DENORMAL_DITHER_OFFSET = 1e-11` (−220 dBFS) is **added** to input samples before
inference and **subtracted** from output samples afterward (`src/dsp/pipeline/stages/input.rs:29`,
`output.rs:109`). This keeps all values above the denormal threshold without any net signal change.

The cancellation is exact (same constant, same sign, opposite application): no rounding asymmetry
is introduced.

### 6.2 FTZ/DAZ (MXCSR register)

`set_daz_ftz()` (`src/math/common/ops.rs:188`) sets the SSE2 MXCSR bits at the start of the first
`process()` block and re-applies every 1024 blocks (hosts may reset MXCSR between callbacks).

- **FTZ (Flush-To-Zero):** output denormals are flushed to zero.
- **DAZ (Denormals-Are-Zero):** input denormals are treated as zero.

**Mandatory?** Yes. Both mechanisms are active unconditionally in the audio thread. There is no
user control and none is needed — these have zero audible effect.

**Implementation.** `src/dsp/pipeline/stages/input.rs`, `src/dsp/pipeline/stages/output.rs`,
`src/math/common/ops.rs`.

---

## 7. Adaptive Compute (Quality Fallback)

**What it is.** When the audio thread's p99 latency approaches the RT deadline (1.33 ms at 48 kHz
/ 64 samples), the Adaptive Compute FSM silently downgrades the active model from Full → Reduced →
Minimal quality tiers. Each tier uses a lighter sub-model (e.g., WaveNet Standard → WaveNet Nano).

During quality transitions WaveNet A1 models use a double-pass inference to crossfade between the
current and target quality tiers without audible glitches. WaveNet A2 models (A2-Full, A2-Lite,
A2-Dyn) do **not** support layer-skip (the mechanism that double-pass relies on); forcing them
through the double-pass path would corrupt their recurrent internal state (buffer history). As a
safety measure, A2 architectures are excluded from double-pass entirely — they transition between
quality tiers via a direct single-pass path, preserving audio integrity at the cost of a slightly
less smooth crossfade during adaptive downgrades.

**Mandatory?** The FSM is active by default. Override via:

- CLI: `--slim auto|full|lite`
- CLAP: dedicated parameter

Setting `--slim full` disables the fallback and always uses the full model.

**Quality impact.** Under CPU overload, users may hear a subtle change in character when the FSM
transitions — this is a deliberate trade-off (seamless playback > consistent quality). Without
Adaptive Compute, a CPU spike would cause audible dropouts (xruns).

**Implementation.** `src/dsp/adaptive_compute.rs`, `src/clap/processor/params.rs`,
`src/models/static_model.rs` (`supports_layer_skip()`), `src/dsp/pipeline/stages/inference.rs`.

---

## 8. Historical Context (Histórico)

Decisions on fidelity, trade-offs, and optimization are catalogued below:

- **SQ5 PoC (2026-07-06):** Removed the F16C weight compression/decompression feature because the runtime decoding overhead on x86-64-v3 outweighed the L1/L2 cache compression benefits for LSTM topologies, and f16c quantization was shown to introduce unnecessary interoperability drift. Native `f32` weights became the default.
- **Sprint S6 Resampler Optimization:** Discarded the dual resampler quality settings (Standard/HQ). HQ (64-tap minimum-phase Kaiser sinc) was made permanent since the lighter 32-tap resampler only saved ~40 ns per block while severely degrading the signal-to-noise ratio from ≥100 dB to ~24 dB.
- **Épico β / Sprint B1 Activation & Gate Updates:** Standardized the exact-grade activation mode runtime dispatch across all topologies (WaveNet, LSTM, ConvNet) using Taylor-based exp kernels and degree-6 minimax tanh — now the universal `Standard` default.
- **Sprint S10 FiLM Parity Correction (2026-07-10):** Resolved the long-standing A2 FiLM interop gap. The Python synthetic weight generator was corrected to apply the standard `+1.0` bias to the scale channels of the FiLM layer (identity-biased initialization). This collapsed the interop gap from SNR 18–36 dB to standard float32 precision limits (SNR 138+ dB, ESR ~1e-14) against the C++ Eigen path. A zero-biased chaos stress model (`wavenet_a2_film_chaos_stress`) was retained to verify engine consistency under numerical stress.

---

## See Also

- [`docs/fastmath-approximations.md`](fastmath-approximations.md) — Detailed Padé/minimax analysis, ULP bounds, bench numbers
- [`docs/perceptual_validation.md`](perceptual_validation.md) — Measurement framework, gate methodology, ABSOLUTE_ESR_CAP
- [`docs/architecture.md`](architecture.md) — §2.2 (weight compression — ver §8 acima), §5 (resampler + pipeline flow), §5.0O (oversampling)
- [`docs/research-references.md`](research-references.md) — Scientific references (Kahles 2019, Sato & Smith 2025, etc.)
- `src/dsp/oversample.rs` — Oversampling engine
- `src/dsp/resampler/mod.rs` — Polyphase sinc resampler
- `src/math/activations/` — All activation implementations
