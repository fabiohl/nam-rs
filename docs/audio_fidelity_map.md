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

| #   | Factor                                                  | Spec? | Mandatory?                   | User-Controllable?       | Quality Impact                                 | Status               |
|:---:|:------------------------------------------------------- |:-----:|:----------------------------:|:------------------------:|:---------------------------------------------- |:--------------------:|
| 1   | **Weight compression (F16C/BF16)**                      | ❌    | ✅ Yes                       | ❌ No                    | −80…−100 dBFS error; LSTM drift                | ✅ Active            |
| 2   | **Activation precision (Padé Standard / HighFidelity)** | ❌    | ✅ Default (Padé); HF opt-in | ✅ CLI + CLAP            | −80…−97 dBFS (Padé) → ÷10,000 (HF); ↓ aliasing | ✅ Active            |
| 3   | **LSTM recurrent drift**                                | ❌    | N/A (consequence of #1)      | ✅ HF gates + Kahan head | ESR 2.6e-2 @48k → 1.4e-1 @192k (5 s)           | ✅ Mitigated (β1–β3) |
| 4   | **Host sample rate resampler**                          | ❌    | ✅ When host ≠ 48 kHz        | ❌ No†                   | Passband ripple < 0.05 dB; stopband ≥ 25 dB    | ✅ Active            |
| 5   | **Neural stage oversampling**                           | ❌    | ❌ Off by default            | ✅ CLI + CLAP            | Reduces aliasing; adds latency + CPU           | ✅ Active            |
| 6   | **Denormal dither + FTZ/DAZ**                           | ❌    | ✅ Yes                       | ❌ No                    | No audible impact (−220 dBFS)                  | ✅ Active            |
| 7   | **Adaptive Compute (quality fallback)**                 | ❌    | ✅ Default                   | 🔶 `--slim` flag         | Silent quality drop under CPU load             | ✅ Active            |

† Resampler quality (Standard/HQ) was rejected after benchmarking — see §4. HF activation mode is exposed via CLI (`--activation`) and CLAP (`PARAM_ACTIVATION=8`). LSTM models fully support HighFidelity activation gates since Épico β (Sprint β1).

---

## 1. Weight Compression — F16C / BF16

**What it is.** NAM-rs converts all model weight matrices (Conv1d, DenseLayer, LSTM projections,
A2 rechannel/conv) from `f32` to 16-bit half-precision during model loading:

- **F16C** (default, AVX2 systems): `u16` storage with `_mm256_cvtph_ps` decompression. Mantissa
  10 bits → error per weight ≈ **3.9e-3** (~−48 dB).
- **BF16** (AVX-512 VNNI systems, Sapphire Rapids+): 7-bit mantissa, ~8× larger error, but enables
  native `_mm512_dpbf16_ps` dot-product without a separate conversion step — higher throughput.

**Why.** WaveNet Standard stores ~80 KB of weights as `f32`. Modern CPU L1 caches are ~32 KB per
core (AMD Zen / Intel Core). Halving weight size to ~40 KB (f16) eliminates most L1 misses,
reducing the per-block stall budget by ~14 cycles/miss. This is the single largest performance
driver for live playback at 32–64 sample blocks.

**Mandatory?** Yes — removing f16c compression would require doubling the weight memory, causing
L1 cache pressure that makes 32-sample blocks impractical on mid-range hardware.

**NAMCore parity.** NAMCore also quantizes to f16c during inference. Both engines share the same
quantization floor — neither is "more correct"; they are equivalent in this regard.

**Quality impact.**

| Model class                      | Quantization effect                                                                                         |
|:-------------------------------- |:----------------------------------------------------------------------------------------------------------- |
| WaveNet/ConvNet/A2 (feedforward) | Error ≈ −80 to −100 dBFS. Non-accumulating. Well below 16-bit PCM floor (−96 dBFS). Perceptually inaudible. |
| LSTM (recurrent)                 | Error accumulates across the cell state. See §3.                                                            |

**Implementation.** `src/models/*/set_weights.rs` (A2: `set_weights.rs:55`), `src/models/lstm/
layer_kernels.rs`, `src/math/common/half.rs`, `src/math/common/dispatch.rs`.

---

## 2. Activation Precision — Standard (Padé) vs HighFidelity

**What it is.** Neural models use non-linear activations. nam-rs replaces exact `f32::tanh` and
`f32::exp` with one of two selectable polynomial approximation modes, held in the
`ActivationPrecision` enum (`src/math/activations/mod.rs`) as a global atomic flag:

| Mode                   | Activation                                 | Max error               | Approx error (dBFS) | Cost            |
|:---------------------- |:------------------------------------------ |:-----------------------:|:-------------------:|:---------------:|
| **Standard** (default) | Padé [5,4] tanh, clamped \|x\| ≤ 4         | 2.32e-3                 | ≈ −53 dB            | Baseline        |
| **Standard** (default) | Minimax degree-17 sigmoid                  | 4.09e-4                 | ≈ −68 dB            | Baseline        |
| **HighFidelity**       | Taylor-based exp kernels, degree-6 minimax | ~2.4e-7 (10,000× lower) | ≈ −133 dB           | +10–15% compute |

The Padé clamp at `|x| > 4` introduces a derivative discontinuity — a weak source of spectral
aliasing at high gain settings. The minimax sigmoid has no clamping discontinuity, and the
HighFidelity kernels eliminate the clamp discontinuity entirely.

### 2.1 Standard mode — why approximate at all

`f32::tanh` on x86-64 costs ~150 ns per 256-element AVX2 vector via the scalar fallback path. The
Padé implementation runs in ~10 ns. For a WaveNet Standard processing hundreds of thousands of
activations per 1.3 ms RT window, this is the difference between real-time and not. Standard
(Padé/minimax) mode is therefore the production default.

### 2.2 HighFidelity mode — runtime switch

`ActivationPrecision::HighFidelity` replaces Padé/minimax with the Taylor-based polynomial exp
kernels above. It is opt-in and user-selectable at runtime — applied at startup (CLI
`--activation standard|hf`) or switched live (CLAP `PARAM_ACTIVATION=8`, at the block boundary,
with GUI sync, persistence, and an offline-render override forcing HighFidelity) — via a single
relaxed atomic store, with no rebuild or allocation. **All model families — WaveNet (A1/A2),
ConvNet, Linear, LSTM — dispatch to both kernels**, including all LSTM gate paths (scalar, AVX2,
AVX-512), completed in Épico β. Full dispatch/topology matrix in
[`docs/fastmath-approximations.md`](fastmath-approximations.md) §10.

### 2.3 Interaction with oversampling

HighFidelity is most effective combined with 4× oversampling (§5): without OS, aliased harmonics
fold back into the baseband regardless of activation precision; with 4× OS, the half-band filter
removes those harmonics and HF mode suppresses the residual high-order folding.

### 2.4 Precision context (validated post-S8)

With the f64 oracle confirmed correct (§3), the combined precision model (f16c + Padé + f32
accumulation) reproduces production to within ESR ≈ 6.9e-5 (LSTM), 6.7e-8 (WaveNet), 1.8e-7 (A2)
— the Padé approximation and f16c together fully explain the gap from ideal f64, with no
unexplained "architectural" residue. Standard (Padé) mode is therefore the well-understood
production default; HighFidelity buys ~10,000× lower activation error, audibly relevant only when
paired with oversampling (§5).

**Mandatory?** No. Standard (Padé) mode is the production default. HighFidelity is opt-in via
`--activation hf` (CLI) or `PARAM_ACTIVATION` (CLAP).

**Implementation.** `src/math/activations/mod.rs`, `src/math/activations/tanh/production.rs`
(Standard), `src/math/activations/tanh/high_fidelity.rs` (HighFidelity),
`src/math/activations/sigmoid.rs`. Full analysis in
[`docs/fastmath-approximations.md`](fastmath-approximations.md).

---

## 3. LSTM Recurrent State Quantization Drift _(consequence of §1)_

**What it is.** An emergent quality degradation specific to LSTM topologies caused by the
interaction between f16c weight quantization and recurrent state accumulation. Every LSTM time
step injects quantization error `εq` into the cell state `cₜ = fₜ·cₜ₋₁ + iₜ·gₜ`. The forget gate
`fₜ` (typically 0.9–0.99 in trained networks) only partially decays old error, so the error
reaches a steady state: `ESR_steady ∝ σ²ε / (1 − ⟨f⟩²)`. WaveNet/A2 are feedforward (no recurrent
state) and do not exhibit this — ESR stays ≈ 1e-13/1e-10 at any duration or rate.

**Two validated floors (production f32, prewarm-paired against the confirmed f64 oracle — see
[`perceptual_validation.md`](perceptual_validation.md)):**

| Reference                               | ESR (LSTM 1×16)               | Meaning                                                         |
|:--------------------------------------- |:-----------------------------:|:--------------------------------------------------------------- |
| vs f64 ideal (absolute precision floor) | 3.57e-3 (−24.5 dB)            | Intrinsic cost of f16c+f32 math, matched initial state          |
| vs NAMCore (interop, both f32 engines)  | 2.61e-2 (−15.8 dB) @ 5s/48kHz | Real drift between the two implementations, ~7× the floor above |

Both grow with signal duration and host sample rate (more recurrent steps); worst case measured
is 1.42e-1 at 192 kHz / 5 s. Full growth tables, rate breakdown (§2.4), and the ruled-out
hypotheses (§3) are in [`docs/lstm_recurrent_drift.md`](lstm_recurrent_drift.md) §1–§3. (A historical claim that
"ESR ≈ 1.0 is the f16c floor" was a bug in the pre-T8.2 oracle's unmatched initial state, not a
real limit — retracted; see that document §4.)

The parity test caps this drift with a **measured, rate-aware** bound — `≤ 96 kHz: 0.08`,
`> 96 kHz: 0.20` — never excluding any rate to make the gate pass (see
[`cpp_parity_map.md`](cpp_parity_map.md) §2.7 and `tests/cpp_parity.rs`).

**Mandatory?** Yes — intrinsic to f16c weight quantization in recurrent architectures. Three
mitigations shipped in Épico β (β1–β3):

- **I6 — HighFidelity activations in LSTM gates** (primary mitigation): exp-based polynomial
  kernels (~2.4e-7 error vs ~2.32e-3 Padé) across scalar/AVX2/AVX-512 LSTM gate paths.
  _Cost:_ +10–15% compute.
- **I4 — Kahan-compensated head accumulation**: compensated summation in the LSTM head
  projection (H→1). ~2 dB SNR gain in deep heads. _Cost:_ negligible.
- **I5 — Oversampling ruled out for LSTM**: external oversampling reduces aliasing (8–11 dB) but
  **changes timbre drastically** (ESR > 1.0 vs Off) because the recurrent feedback delay is fixed
  in absolute samples — 2×/4× halves/quarters that window in seconds. **Not recommended** as a
  user control for LSTM, unlike WaveNet where oversampling is transparent. Full ASR/ESR tables in
  `docs/lstm_recurrent_drift.md` §4.1.

Stateful ADAA for the recurrent cell (Holters 2019; Mikkonen & Werner 2025) was evaluated and
**not adopted** — conflicts with the polymorphic `dispatch_simd!` macro, same reason as §5.
Retained as a research direction (`docs/research-references.md` R6, R7b).

**User impact.** LSTM models degrade with signal duration and with host sample rate. WaveNet
models do not. For long sessions or high host rates, a WaveNet of equal budget is more faithful.

**Implementation.** `src/models/lstm/layer_kernels.rs` (GEMV with f16c weights). Full analysis in
[`docs/lstm_recurrent_drift.md`](lstm_recurrent_drift.md).

---

## 4. Host Sample Rate Adaptation (Polyphase Sinc Resampler)

**What it is.** NAM models are trained at 48 kHz. When a DAW host runs at a different rate
(44.1 kHz is the most common), nam-rs converts using a native minimum-phase polyphase FIR sinc
resampler (`NamResampler`, `src/dsp/resampler.rs`).

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

**Implementation.** `src/dsp/resampler.rs`, `src/dsp/sinc_kernel.rs`.

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
effective when combined with the HighFidelity activation mode (§2.2).

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

## See Also

- [`docs/fastmath-approximations.md`](fastmath-approximations.md) — Detailed Padé/minimax analysis, ULP bounds, bench numbers
- [`docs/lstm_recurrent_drift.md`](lstm_recurrent_drift.md) — Full RCA of LSTM ESR accumulation (§3 above)
- [`docs/perceptual_validation.md`](perceptual_validation.md) — Measurement framework, gate methodology, ABSOLUTE_ESR_CAP
- [`docs/architecture.md`](architecture.md) — §2 (weight compression), §3 (resampler), §4 (oversampling), §5 (pipeline flow)
- [`docs/research-references.md`](research-references.md) — Scientific references (Kahles 2019, Sato & Smith 2025, etc.)
- `src/dsp/oversample.rs` — Oversampling engine
- `src/dsp/resampler.rs` — Polyphase sinc resampler
- `src/math/activations/` — All activation implementations
