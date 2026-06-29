<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

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

| #   | Factor                                  | Spec? | Mandatory?              | User-Controllable?   | Quality Impact                              | Status              |
|:---:|:--------------------------------------- |:-----:|:-----------------------:|:--------------------:|:------------------------------------------- |:-------------------:|
| 1   | **Weight compression (F16C/BF16)**      | ❌    | ✅ Yes                  | ❌ No                | −80…−100 dBFS error; LSTM drift             | ✅ Active           |
| 2   | **Activation approximations (Padé)**    | ❌    | ✅ Default              | 🔶 Partial           | −80…−97 dBFS; ↑ aliasing                    | ✅ Active           |
| 3   | **LSTM recurrent drift**                | ❌    | N/A (consequence of #1) | ❌ No mitigation yet | ESR 2.6e-2 @48k → 1.4e-1 @192k (5 s)        | ✅ Documented       |
| 4   | **Host sample rate resampler**          | ❌    | ✅ When host ≠ 48 kHz   | ❌ No†               | Passband ripple < 0.05 dB; stopband ≥ 25 dB | ✅ Active           |
| 5   | **Neural stage oversampling**           | ❌    | ❌ Off by default       | ✅ CLI + CLAP        | Reduces aliasing; adds latency + CPU        | ✅ Active           |
| 6   | **Activation precision (HF mode)**      | ❌    | ❌ Off by default       | 🔶 Internal only†    | Error ÷ 10,000; ↓ aliasing                  | ⚠️ Not user-exposed |
| 7   | **Denormal dither + FTZ/DAZ**           | ❌    | ✅ Yes                  | ❌ No                | No audible impact (−220 dBFS)               | ✅ Active           |
| 8   | **Adaptive Compute (quality fallback)** | ❌    | ✅ Default              | 🔶 `--slim` flag     | Silent quality drop under CPU load          | ✅ Active           |

† Resampler quality (Standard/HQ) was deferred — see §4. HF activation mode has no runtime knob — see §6.

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

## 2. Activation Function Approximations (Padé / Standard mode)

**What it is.** Neural models use non-linear activations. nam-rs replaces exact `f32::tanh` and
`f32::exp` with polynomial approximations for throughput:

| Activation | Approximation                 | Max error | Approx error (dBFS) |
|:---------- |:----------------------------- |:---------:|:-------------------:|
| `tanh`     | Padé [5,4], clamped \|x\| ≤ 4 | 2.32e-3   | ≈ −53 dB            |
| `sigmoid`  | Minimax degree-17             | 4.09e-4   | ≈ −68 dB            |

The clamp at `|x| > 4` in the Padé introduces a derivative discontinuity — a weak source of
spectral aliasing at high gain settings. The minimax sigmoid has no clamping discontinuity.

**Why.** `f32::tanh` on x86-64 costs ~150 ns per 256-element AVX2 vector via the scalar fallback
path. The Padé implementation runs in ~10 ns. For a WaveNet Standard processing hundreds of
thousands of activations per 1.3 ms RT window, this is the difference between real-time and not.

**Optional HF mode.** `ActivationPrecision::HighFidelity` replaces Padé/minimax with Taylor-based
polynomial exp kernels (~degree-6). Error is ~10,000× lower (≈ −133 dB), and it eliminates the
`|x|>4` clamp discontinuity. This is most useful combined with oversampling (§5). See §6 for
current exposure status.

**Mandatory?** The Padé mode is the mandatory production default. The HF mode exists in code but
is not yet runtime-selectable by the user (see §6).

**Implementation.** `src/math/activations/tanh/production.rs` (Padé), `src/math/activations/
tanh/high_fidelity.rs` (HF), `src/math/activations/sigmoid.rs`. Full analysis in
[`docs/fastmath-approximations.md`](fastmath-approximations.md).

---

## 3. LSTM Recurrent State Quantization Drift _(consequence of §1)_

**What it is.** An emergent quality degradation specific to LSTM topologies caused by the
interaction between f16c weight quantization and recurrent state accumulation.

Each LSTM time step injects quantization error `εq` into the cell state `cₜ = fₜ·cₜ₋₁ + iₜ·gₜ`.
The forget gate `fₜ` (typically 0.9–0.99 in trained networks) only partially decays old error.
After enough steps the error reaches a steady state:

```text
ESR_steady ∝ σ²ε / (1 − ⟨f⟩²)
```

**The true precision floor (validated).** The f64 reference oracle — now confirmed correct by
three-way cross-validation (production f32 ↔ Rust f64 oracle ↔ independent NumPy f64; see
[`perceptual_validation.md`](perceptual_validation.md) and `tests/reference_oracle_f64.rs`) —
measures the LSTM's distance from the _ideal_ math:

```text
ESR(production f32 vs ideal f64, prewarm-paired @ 48 kHz) = 3.57e-3   ← real f16c floor (LSTM 1×16)
                                  WaveNet = 6.13e-14   A2 = 4.28e-10   ← feedforward, no drift
```

This is the genuine cost of f16c+f32 precision. (A historical claim that "ESR ≈ 1.0 was the f16c
floor" was a **bug in the old oracle**, not a real floor — corrected in Sprint S8 / T8.1–T8.2.)

**Two ways drift grows.** Both are measured against NAMCore (the C++ reference — itself also f16c,
so this is _interop_ drift shared by both engines, larger than the f64 floor above):

_(a) with signal duration_ — the recurrent state keeps accumulating until steady state:

| Signal (LSTM 1×16, 48 kHz) | ESR vs NAMCore | MR-STFT  |
|:-------------------------- |:--------------:|:--------:|
| v1 — 2,048 samp (43 ms)    | 1.04e-2        | 0.098    |
| v2 — 240,000 samp (5 s)    | **2.61e-2**    | **0.87** |

_(b) with sample rate_ — higher rates mean more recurrent steps per second of audio:

| Host rate | LSTM 1×16 ESR vs NAMCore | Notes                       |
|:--------- |:------------------------:|:--------------------------- |
| 44.1 kHz  | 2.39e-2                  | native                      |
| 48 kHz    | 2.61e-2                  | native (model's train rate) |
| 88.2 kHz  | 5.39e-2                  |                             |
| 96 kHz    | 6.09e-2                  |                             |
| 192 kHz   | **1.42e-1**              | 4× native — worst case      |

(Other LSTMs drift much less: 2×8 @ 192 kHz = 4.20e-2; the tiny Official H=3 ≈ 1.23e-3 flat.
WaveNet is feedforward — no recurrent state, ESR ≈ 1e-13 at every rate.)

The parity test caps this drift with a **measured, rate-aware** bound — `≤ 96 kHz: 0.08`,
`> 96 kHz: 0.20` — never excluding any rate to make the gate pass (see
[`cpp_parity_map.md`](cpp_parity_map.md) §4.5 and `tests/cpp_parity.rs`).

**Mandatory?** Yes — it is intrinsic to f16c weight quantization in recurrent architectures.
Removing it would require f32 weights (breaking format interoperability with NAMCore) or
compensated accumulation in the cell state (proposed mitigation — see §9).

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

| Metric                   | Value                                                      |
|:------------------------ |:---------------------------------------------------------- |
| Passband ripple          | < 0.05 dB (to 0.45 × Nyquist)                              |
| Stopband attenuation     | ≥ 25 dB (SNR gate; spec target: ≥ 100 dB, pending T8 work) |
| Group delay (min-phase)  | Asymmetric; slight high-frequency transient smearing       |
| 44.1 kHz top-end rolloff | Post-S6 < 0.05 dB at 20 kHz (vs. −1.7 dB pre-S6)           |

**Mandatory?** Yes, when the host rate differs from 48 kHz. Cannot be disabled without
breaking audio at non-48 kHz rates.

**User-controllable?** A "Resampler Quality: Standard/HQ" parameter was designed (Tarefa 5.7 /
`docs/architecture.md §3.2`) but deferred pending a Δμs benchmark. The 64-tap HQ config is
the production default; if cost proves non-negligible, a user-selectable Standard (32-tap) mode
may be exposed via CLI and CLAP.

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

**Mandatory?** No. Default is Off. Designed for offline/mixdown rendering and critical
listening where CPU headroom exists.

**User-controllable?** Yes, via:

- CLI: `--oversample off|2x|4x` (alias `--os`)
- CLAP: parameter ID=7, stepped enum, persisted in plugin state
- Changes trigger an off-RT rebuild (same SPSC path as model hot-swap)

**Quality impact.** Reduces ASR (Aliasing-to-Signal Ratio, Sato & Smith DAFx 2025). Measured
reduction is architecture-dependent; WaveNet benefits most from 4× (ReLU/tanh corner). Most
effective when combined with the HighFidelity activation mode (§6).

**Why ADAA was rejected.** Antiderivative Anti-Aliasing (Parker DAFx-16; Bilbao IEEE SPL 2017)
requires per-activation analytical antiderivatives, which conflicts with the polymorphic
`dispatch_simd!` macro and multi-architecture dispatch. The half-band approach is
activation-agnostic and transparent to the model dispatcher.

**Implementation.** `src/dsp/oversample.rs` (`OversampleEngine`, `OversampleFactor`),
`src/dsp/pipeline/stages/inference.rs` (integration point).

---

## 6. Activation Precision Mode (HighFidelity / Standard)

**What it is.** A runtime switch between two activation implementations:

| Mode                   | Activation                                 | Error                   | Aliasing                     | Cost            |
|:---------------------- |:------------------------------------------ |:-----------------------:|:----------------------------:|:---------------:|
| **Standard** (default) | Padé [5,4] tanh + minimax sigmoid          | ~2.3e-3                 | Higher (clamp discontinuity) | Baseline        |
| **HighFidelity**       | Taylor-based exp kernels, degree-6 minimax | ~2.4e-7 (10,000× lower) | Lower (no clamp)             | +10–15% compute |

The mode is selected via the `ActivationPrecision` enum in `src/math/activations/mod.rs` and
applied at model-load time (or hot-swap rebuild) via an atomic flag. The branch predictor
specialises to the active path during steady-state inference.

**Interaction with oversampling.** HighFidelity is most effective combined with 4× oversampling:
without OS, aliased harmonics fold back into the baseband regardless of activation precision;
with 4× OS, the half-band filter removes those harmonics and HF mode suppresses the residual
high-order folding.

**Mandatory?** No. Standard mode is the production default.

**⚠️ User-exposed?** **Not yet.** The `HighFidelity` path exists and is tested but has **no
runtime knob** (no CLI flag, no CLAP parameter). It is currently only selectable by code or
build-time configuration. A user-accessible control (e.g., `--activation hf|standard`) was
discussed in Tarefa 5.3 but not implemented as a runtime parameter.

**Precision context (validated post-S8).** With the f64 oracle now confirmed correct (§3), the
combined precision model (f16c + Padé + f32 accumulation) reproduces production to within
ESR ≈ 6.9e-5 (LSTM), 6.7e-8 (WaveNet), 1.8e-7 (A2) — i.e. the Padé approximation and f16c
together fully explain the gap from ideal f64; there is no unexplained "architectural" residue.
Standard (Padé) mode is therefore the well-understood production default; HighFidelity mode buys
~10,000× lower activation error, audibly relevant only when paired with oversampling (§5).

**Implementation.** `src/math/activations/mod.rs`, `src/math/activations/tanh/production.rs`,
`src/math/activations/tanh/high_fidelity.rs`. Detailed precision analysis in
[`docs/fastmath-approximations.md`](fastmath-approximations.md).

---

## 7. Denormal Prevention — Dither + FTZ/DAZ

**What it is.** Two complementary mechanisms prevent subnormal (denormal) floating-point numbers
from entering the neural model's internal state. Denormals trigger microcode assist on x86-64,
causing a 10–100× slowdown per affected operation — a real-time guarantee killer.

### 7.1 Deterministic Dither Offset

A fixed offset `DENORMAL_DITHER_OFFSET = 1e-11` (−220 dBFS) is **added** to input samples before
inference and **subtracted** from output samples afterward (`src/dsp/pipeline/stages/input.rs:29`,
`output.rs:109`). This keeps all values above the denormal threshold without any net signal change.

The cancellation is exact (same constant, same sign, opposite application): no rounding asymmetry
is introduced.

### 7.2 FTZ/DAZ (MXCSR register)

`set_daz_ftz()` (`src/math/common/ops.rs:188`) sets the SSE2 MXCSR bits at the start of the first
`process()` block and re-applies every 1024 blocks (hosts may reset MXCSR between callbacks).

- **FTZ (Flush-To-Zero):** output denormals are flushed to zero.
- **DAZ (Denormals-Are-Zero):** input denormals are treated as zero.

**Mandatory?** Yes. Both mechanisms are active unconditionally in the audio thread. There is no
user control and none is needed — these have zero audible effect.

**Implementation.** `src/dsp/pipeline/stages/input.rs`, `src/dsp/pipeline/stages/output.rs`,
`src/math/common/ops.rs`.

---

## 8. Adaptive Compute (Quality Fallback)

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

## 9. Pending / Open Work

| Item                                                                          | Status                        | Reference                                                                               |
|:----------------------------------------------------------------------------- |:-----------------------------:|:--------------------------------------------------------------------------------------- |
| HighFidelity activation mode user control (CLI/CLAP knob)                     | 🟡 Designed, not exposed (§6) | future                                                                                  |
| Runtime oversample switching (currently init-time only)                       | 🟡 Designed, off-RT TODO      | §5; `rt_callback/commands.rs` `TODO(oversample-rt)`; **⚠️ bloqueado por F2**            |
| Resampler quality selector (Standard 32T / HQ 64T)                            | 🟡 Designed, HQ is default    | §4; **⚠️ requer correção de F3 antes da implementação**                                 |
| Kahan-compensated LSTM head accumulation                                      | 🟡 Proposed mitigation for §3 | Épico E4                                                                                |
| Oversampled recurrent state (LSTM HQ mode)                                    | 🟡 Proposed mitigation for §3 | Épico E4                                                                                |
| **🔴 Oversampling: latência não reportada ao host (PDC)**                     | 🔴 Bug ativo                  | §5 (latência documentada: 12/24 amostras); [`TODO-findings.md`](../TODO-findings.md) F2 |
| **🔴 Resampler: fórmula de atraso de grupo errada para banco de fase mínima** | 🔴 Bug ativo                  | §4 (min-phase default confirmado); [`TODO-findings.md`](../TODO-findings.md) F3         |
| **Adaptive Compute + A2: double-pass corrompia estado recorrente**            | ✅ Resolvido (Sprint A1)      | §8 (Adaptive Compute); [`TODO-findings.md`](../TODO-findings.md) F1                     |
| Adaptive Compute: `prev_state` dessinc em transições encadeadas (< 32 ms)     | ✅ Resolvido (Sprint A1)      | §8; [`TODO-findings.md`](../TODO-findings.md) F4                                        |

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
