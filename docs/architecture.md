<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# NAM-rs Architecture: Standalone Neural Inference Client

The architecture of NAM-rs is designed for low-latency DSP processing and neural inference focused on audio equipment simulation (Neural Amp Modeler). Operating as a standalone PipeWire client (Stable) or as a CLAP plugin (Release) on Linux, it utilizes idiomatic Rust with a focus on RT (Real-Time) safety.

## 1. PipeWire Topology (Standalone Mode): Dual-Stream (Capture + Playback)

- **Virtual Sink (Audio/Sink):** NAM-rs declares itself as the default sound output via `pw_stream`. Apps connect automatically via WirePlumber.
- **Playback Stream (Stream/Output):** A second stream reads the processed audio and delivers it to the physical hardware, bypassing the limitations of monitor ports on virtual sinks.
- **DspBridge (Lock-Free Double-Buffer):** An aligned structure (128B) that isolates the streams. Capture writes to the inactive buffer (Release); playback reads from the active one (Acquire), synchronized by `AtomicU64` (generation).
- **True Stereo and Bypass:** Symmetric L/R inference in Standalone/Pipewire mode. Since the NAM standard is mono by definition, stereo operation is a convenience feature implemented in standalone. If R is silent or identical to L, the system skips R inference (saving ~50% CPU).

> **Note:** The Dual-Stream topology is preferred over `pw_filter` because it guarantees automatic "plug-and-play" routing by WirePlumber and due to the maturity of the safe wrappers in the `pipewire` crate.

## 2. Inference & Microarchitecture (SIMD x86-64-v3/v4)

- **Multiversioning via `dispatch_simd!` Macro:** Dynamic dispatch at model load time that selects the best SIMD kernel v-table (`Avx2Math`, `Avx512Math`, etc.). The use of macros for monomorphization ensures that the compiler emits native intrinsics without v-table overhead in the inference hot-path.
- **FastMath Activations & Gain LUT:** `simd_tanh` uses a **Padé [5,4]** rational approximant with hardware `_mm256_div_ps`; `simd_sigmoid` uses a direct **Minimax degree-17** polynomial. Maximum error: tanh ~2.32e-3 on [-4, 4], sigmoid ~4.09e-4 on [-8, 8] (see [fastmath-approximations.md](fastmath-approximations.md)). Includes an interpolated **Gain LUT (Look-Up Table)** for ultra-fast dB → Linear conversion in RT, avoiding expensive calls to `powf`.
- **Gated Activation Fusion (WaveNet A2):** Unification of `tanh` and `sigmoid` into a single native SIMD kernel, reducing register pressure and avoiding multiple passes over the activation vector.
- **Dot Product ILP:** Implementation with multiple independent accumulators (`sum0..sum3` in AVX2, `acc0..acc7` in AVX-512) to saturate FMA port throughput, breaking dependency chains.
- **Weight Compression (F16C/BF16):** Weights are stored in `f16` (Half-Precision) or `bf16` (Bfloat16) to reduce L1/L2 memory traffic. Precision selection and the corresponding on-the-fly conversion/decompression (via `_mm256_cvtph_ps`/`_mm512_cvtph_ps` for F16, or corresponding bit-unpacking for BF16) occur at runtime via dynamic dispatch managed by `SimdMathConfig` (initialized by the dispatcher based on the CPU's supported instruction set, such as AVX2, AVX-512 F16/BF16).
- **Gate-Major Layout & Fused 4-Gate GEMV (LSTM):** Transposition of weights to a `[Gate][Input][Hidden]` layout. The inference fuses the calculation of the 4 gates in a single pass over the state vector.
- **Layer Overlap Pipelining (LSTM 2-Layer):** Fine-grained parallelism where Layer 2 processes frame `N-1` simultaneously with frame `N` of Layer 1, increasing throughput in multi-layer models.
- **Native BF16 (AVX-512 BF16):** Native kernel support via `_mm512_dpbf16_ps` (VNNI-BF16) for Sapphire Rapids and newer CPUs. Includes the **Fused 4-Gate GEMV BF16** kernel for LSTM, eliminating scalar dispatch cost and doubling dot-product throughput relative to AVX2.
- **Fused Conv1d+Mixin (WaveNet):** The sum of the mixin vector is fused directly into the Conv1D accumulator.
- **Fused Tanh + Head Accumulate (WaveNet):** Native unification of the activation and skip-connection (head) phases into a single SIMD kernel (`tanh_and_accumulate_block`).
- **Fused Residual GEMV with Frame Tiling (WaveNet):** The residual calculation is fused into the GEMV of the next layer, utilizing **4-frame tiling (AVX2)** or **8-frame tiling (AVX-512)** to maximize weight reuse in registers.
- **Conv1D Tiling:** Block processing of multiple channels to maximize data reuse in SIMD registers and reduce cache latency in deep dilation models.
- **Linear Model (FIR Filter):** A fast non-neural FIR filter architecture implementing convolved input history with weights and a bias.

### Technical Decision: FastMath Precision vs. Performance

> **Decision:** `tanh` uses a Padé [5,4] rational approximant (`_mm256_div_ps`); `sigmoid` uses a direct Minimax degree-17 polynomial — both replacing IEEE-754 `libm` in the hot-path.
>
> **Trade-off:** ~2–3 decimal places of precision for ~10–20× throughput vs. scalar `libm`.
> Maximum error: **tanh ~2.32e-3** on [-4, 4], **sigmoid ~4.09e-4** on [-8, 8].
> The divergence vs. C++ is perceptually inaudible (below the 16-bit PCM quantization floor).
>
> **Validation:** Deterministic sweep, proptest (10k inputs), golden vectors cross-validation against NeuralAmpModelerCore (7 models).
>
> **References:** [src/math/activations/tanh/](../src/math/activations/tanh/), [src/math/activations/sigmoid.rs](../src/math/activations/sigmoid.rs), [docs/fastmath-approximations.md](fastmath-approximations.md), [tests/nam_infer_test.rs](../tests/nam_infer_test.rs).

### Technical Decision: Portability and Virtual Allocation of `MirroredBuffer`

> **Decision:** The `MirroredBuffer` structure performs virtual memory mirroring by mapping the same physical block twice consecutively to avoid logical wrap-around in the DSP hot-path. Primary support is strictly targeted at Linux using `memfd_create`. For non-Linux platforms, a fallback (stub) is provided that returns an incompatibility error (`Unsupported`).
>
> **Trade-off:** Using `memfd_create` on Linux offers an ideal way to allocate mirrored buffers without creating files on physical disk and without requiring complex cleanup on the filesystem. Since the production ecosystem of NAM-rs is exclusively focused on Linux (Standalone PipeWire and CLAP plugin), the implementation of stubs for other platforms is sufficient for static compilation portability of the crate, avoiding additional concurrency or I/O complexity in the cold loading path.

### NAMB Binary Format (Native Audio Model Binary)

The `.namb` format is an optimized evolution of the original JSON for real-time use.

- **NAMB v1:** Encapsulates the metadata JSON and the weights in `f32` (Little-Endian) in a single binary block with CRC32.
- **NAMB v2 (Pre-Transposed):** Stores weights directly in the final kernel layout (Gate-Major for LSTM or Interleaved-4 for WaveNet).
  - Eliminates the need for memory transposition during loading.
  - Reduces model swap latency from ~50ms to <1ms (cold loading path).
  - Identified by the header flag `NambHeader::layout_type`.

### WaveNet Data Flow (Inference Pipeline)

The diagram below illustrates the data flow in a WaveNet inference block, highlighting the fused operations that minimize memory traffic and maximize SIMD throughput:

```mermaid
graph TD
    In[/"Input Block (f32)"/] --> RC["Rechannel (Dense 1x1)"]
    RC --> MB["Mirrored Buffer (Delay Line)"]

    subgraph LayerCascade ["Layer Cascade (WaveNet Layers)"]
        direction TB
        L1["Layer 1"] --> L2["Layer 2"]
        L2 -.-> LN["Layer N"]
    end

    MB --> LayerCascade

    subgraph Internal ["Layer Micro-Architecture (Hot-Path)"]
        direction TB
        S1["Conv1D Tap Fetch (SIMD Prefetch)"] --> S2["Fused: Conv1D + Input Mixin"]
        S2 --> S3["Fused: Gated Activation (Tanh/Sigmoid)"]
        S3 --> S4["Fused: Head Accumulate (Skip Connection)"]
        S3 --> S5["Fused: 1x1 GEMV + Residual Addition"]
    end

    LayerCascade -.-> Internal

    LN --> HR["Head Rechannel (Final Dense)"]
    S4 -.-> HA["Head Accumulator (Skip Sum)"]
    HA --> HR
    HR --> SC["Output Scale + Clipping"]
    SC --> Out[/"Output Block (f32)"/]

    classDef fused fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    class S2,S3,S4,S5 fused;
```

### 6.1 Mixed-Precision Selective

To optimize the trade-off between computational latency and tonal accuracy, NAM-rs uses selective mixed precision (E8.T08). While the WaveNet backbone (including Conv1D convolutions, input_mixin, and one_by_one) is computed with weights compressed in F16 or BF16 to save cache bandwidth, the final output projection layer (`head_rechannel`) and the final projection in LSTMs use full floating-point precision (`f32`). Head inference executes a native f32 scalar GEMV (`process_block_f32_native`), ensuring 24-bit fidelity in the analytically most sensitive stage of the output.

### 6.2 Numerical Stability (Dither + Scalar-Fallback Kahan)

To prevent the accumulation of numerical drift and mathematical instabilities in long-duration runs:

- **Kahan Summation (E8.T06):** Employed in the interleaved 4x scalar fallback dot products (`scalar_ref/dot.rs`) to reduce relative accumulation error from $O(N \cdot \epsilon)$ to $O(\epsilon)$. The static conv1d paths (`conv1d.rs`, `conv1d_dual.rs`) use plain `+=` accumulation — error for K≤3 taps is below −129 dBFS per layer, inaudible (T13.2/T18.4).
- **Deterministic Dither (E8.T05):** Injection of an inaudible deterministic DC offset of $-220\text{ dBFS}$ ($1.0 \times 10^{-11}$) at the input stage ([apply_input_stage](../src/dsp/pipeline/stages/input.rs#L47) after gain) with corresponding compensatory subtraction at the output ([apply_output_stage](../src/dsp/pipeline/stages/output.rs#L21)). Keeps neural activations (tanh/sigmoid) out of subnormal (denormal) ranges during fade-outs or absolute silence, preventing pops and CPU spikes.

## 3. Time Management and Isolation (Strict RT)

- **DSP Thread (SCHED_FIFO):** Pinned via Core Affinity (`select_optimal_cpu`) preferring cores with lower IRQ load.
- **Zero-Allocation:** Strict prohibition of heap allocations, `Vec`, `Box`, or panics in the DSP thread.
- **Jitter Isolation:**
  - `mlockall` to prevent page faults.
  - PM QoS Lock (`/dev/cpu_dma_latency`) to disable deep C-States **globally across all system cores**, ensuring zero wakeup latency.
  - Disabling THP (Transparent Huge Pages) via `prctl` to prevent kernel compaction spikes.
- **High-Precision Telemetry (RDTSC):** Replacement of `Instant::now()` (vDSO syscall) with direct reading of the calibrated TSC in the RT callback. Guarantees ~1ns precision with ~1 CPU cycle overhead, eliminating kernel-induced jitter in DSP load measurement.
- **SPSC Channels (rtrb):** Lock-free communication between CLI (async) and DSP (RT). Payload aligned (128B) to prevent False Sharing.

## 4. Module Structure (Tripartite Organization)

From v1.4 onwards, NAM-rs adopts a clear modular structure to support multiple hosts (Standalone/PipeWire and Plugin/CLAP) without dependency pollution:

| Layer                              | Sub-modules                                                    | Responsibility                                                                                                          |
|:---------------------------------- |:-------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------- |
| **Common** (`src/common/`)         | `diagnostics`, `spsc`, `params`, `audio_host`                  | Shared infrastructure, inter-thread communication (SPSC), and host-agnostic abstractions.                               |
| **Standalone** (`src/standalone/`) | `pw_host`, `rt_setup`, `cli`, `colors`                         | Native Linux backend. Manages the PipeWire server, hardware setup (FIFO/Affinity), and the command-line interface.      |
| **CLAP** (`src/clap/`)             | `plugin`, `processor`, `param_smoother`, `extensions/`, `gui/` | Full CLAP plugin with DSP pipeline, parameters, persistence, egui/baseview visual interface, and anti-zipper smoothing. |
| **Math** (`src/math/`)             | `common/`, `activations/`, `gemm/`, `dsp/`...                  | Mathematical infrastructure modularized by domain, isolating low-level SIMD kernels from dispatch logic.                |
| **Core DSP** (`src/`)              | `dsp/`, `models/`, `loader/`                                   | The "brain" of NAM-rs. Neural inference algorithms and model parsing.                                                   |

### Architecture Layers Diagram

```mermaid
graph TD
    subgraph Host_Specific ["Host-Specific Layers (Frontends)"]
        Standalone["src/standalone/"]
        CLAP["src/clap/"]
    end

    subgraph Host_Agnostic ["Host-Agnostic Layers (Core)"]
        Common["src/common/"]
        DSP["src/dsp/"]
        Math["src/math/"]
        Models["src/models/"]
        Loader["src/loader/"]
    end

    Standalone --> Common
    Standalone --> DSP
    CLAP --> Common
    CLAP --> DSP

    DSP --> Math
    DSP --> Models
    Loader --> Models

    subgraph External_Deps ["External Dependencies"]
        PW["libpipewire-0.3"]
        Clack["clack-plugin"]
    end

    Standalone -.-> PW
    CLAP -.-> Clack
```

This separation ensures that building for CLAP does not drag in PipeWire dependencies and facilitates future portability to other operating systems.

## 4.1 Conditional Compilation Strategy (Feature Flags)

NAM-rs uses *feature flags* to isolate backends and reduce the final binary footprint:

| Build Profile            | Compilation Command                                              | Generated Asset        | Main Dependencies                                      |
|:------------------------ |:---------------------------------------------------------------- |:---------------------- |:------------------------------------------------------ |
| **Standalone** (default) | `cargo build`                                                    | Executable Binary      | `pipewire`, `rtrb`, `clap` (CLI)                       |
| **CLAP Plugin**          | `cargo build --no-default-features --features clap-plugin --lib` | `.so` Library (cdylib) | `clack-plugin`, `clack-extensions`, `egui`, `baseview` |
| **DSP Lib (Pure)**       | `cargo build --no-default-features --lib`                        | Rust Library (`.rlib`) | Core DSP only (no-std ready)                           |

## 5. DSP & Native Resampling

NAM-rs uses a native **Minimum-Phase Polyphase Sinc Resampler**, replacing external dependencies.

- **Advantages:** Eliminates pre-ringing (energy concentrated at the start), reduces algorithmic latency from ~1.5ms (linear phase) to ~0.7ms, and offers ~9x superior performance via dedicated AVX2/AVX-512 convolution.
- **Gate FSM:** Implements temporal and amplitude hysteresis (Schmitt Trigger) to prevent chattering at noise floor levels. Includes linear SIMD ramping for smooth transitions (fade-in/out), fused into a single stereo pass to optimize cache locality.

### Bidirectional DSP Flow

```text
PipeWire Input (Nk Hz)
    ▼ Gate FSM: Silence/Mono Detection + Gain Ramp
    │
    ▼ Input Gain (SIMD)
    │
    ▼ NamResampler::process_input (Nk → 48 kHz)
    │
    ▼ NamModel::process (Neural Inference @ 48 kHz)
    │
    ▼ NamResampler::process_output (48 kHz → Nk)
    │
    ▼ Output Gain (SIMD) + Clipping
    │
    ▼ IR Cabsim (UPOLS Convolution, Optional / Zero-Cost Bypass)
    │
    ▼ DspBridge (Lock-Free Write) → Playback Stream (Read) → Hardware
```

## 5.1 Adaptive Compute: Graceful CPU Fallback

To guarantee xrun-free operation in real-time audio threads under high CPU utilization, NAM-rs includes a dynamic **Adaptive Compute** sub-system.

- **Objective:** Gracefully lower the computational footprint of neural model inference when the audio thread approaches its deadline budget, preventing audible dropouts (xruns) at the cost of a transient, imperceptible decrease in model complexity.
- **Hysteresis FSM:** Prevents rapid toggling ("chattering") between states by using asymmetric thresholds and consecutive confirmation blocks:
  - **Full → Reduced:** Triggered after 3 consecutive blocks exceeding `0.70 * budget` (Conservative) or `0.55 * budget` (Aggressive). In WaveNet, skips 25% of layers; in LSTM, reduces to 1 layer.
  - **Reduced → Minimal:** Triggered after 3 consecutive blocks exceeding `0.85 * budget` (Conservative) or `0.70 * budget` (Aggressive). In WaveNet, skips 50% of layers; in LSTM, transitions to direct passthrough.
  - **Recovery:** Upgrades to the previous state only after 5 consecutive blocks remain safely below recovery thresholds (`0.35 * budget` for Conservative, `0.275 * budget` for Aggressive).
- **Linear Crossfade:** Integrates a 32 ms linear parameter crossfade between active layers to guarantee smooth, click-free structural transitions.
- **Deterministic Offline Bounce:** During offline rendering/export (`RenderMode::Offline` in CLAP), the host DAW does not operate under real-time constraints. To guarantee deterministic, maximum-quality output, the render mode transition forces `AdaptiveCompute` to `Off` (which resets the FSM state to `Full`), clears all active degradation status flags (`RT_STATUS_DEGRADE_REDUCED`, `RT_STATUS_DEGRADE_MINIMAL`), and ignores all block deadline measurements.
- **A2 slimmable degradation:** For A2 models delivered as a `SlimmableContainer`, this same FSM drives the runtime **A2-Full → A2-Lite** switch (selecting the lighter submodel under CPU pressure) instead of layer-skipping, reusing the crossfade machinery. See §7.

## 5.2 IR Cabsim — Impulse Response Convolution

The cabsim stage performs real-time convolution of the neural model output with a speaker cabinet impulse response (IR), simulating the physical cabinet/speaker coloration that follows amplifier modeling.

### Algorithm: Uniform-Partitioned Overlap-Save (UPOLS)

The convolution engine (`src/dsp/cabsim/conv.rs`) implements UPOLS in the frequency domain, following Gardner's efficient convolution design:

- **Partition size** equals the audio block size (typically 64–256 samples). The engine is reconstructed on buffer-size changes (`activate()` in CLAP, buffer swap in standalone).
- **FFT size** is `2 × partition_size` (rounded up to next power of two).
- **Kernel pre-FFT:** All IR partitions are transformed to the frequency domain once at construction time (outside the audio thread), so the hot-path only performs forward FFT of the input block and IFFT of the accumulated spectrum.
- **FDL (Frequency Delay Line):** A pre-allocated circular buffer of complex spectra stores the history of input FFTs. Each block shifts the FDL and computes `Σ(H_k × X_{i-k})` across all partitions before inverse FFT.
- **Latency** is exactly `partition_size` samples (one full audio block).

### Zero-Allocation Hot-Path

The `ConvEngine::process()` method performs zero heap allocations — all working buffers (input overlap, FFT scratch, FDL, accumulation buffer) are allocated once at construction. The bypass path (no IR loaded) is a single branch check with no measurable overhead.

### Pipeline Integration

The cabsim runs as an optional post-inference stage (Stage 3) in the DSP pipeline, positioned between inference and output processing:

```mermaid
graph TD
    Input[/"Input (f32)"/] --> Gate["Gate FSM + Input Gain"]
    Gate --> ResampUp["Resampler (Up: SR → 48kHz)"]
    ResampUp --> Infer["Neural Inference (NamModel::process)"]
    Infer --> ResampDown["Resampler (Down: 48kHz → SR)"]
    ResampDown --> OutGain["Output Gain + Clipping"]
    OutGain --> Ck{"Cabsim IR loaded?"}
    Ck -->|"Yes"| CabSim["UPOLS Convolution\n(ConvEngine::process)"]
    Ck -->|"No (bypass)"| Bridge["DspBridge Write"]
    CabSim --> Bridge
    Bridge --> Playback[/"Playback Stream → Hardware"/]

    classDef bypass fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray:5 5;
    class Ck bypass;
```

### IR Loading and Transfer

IR `.wav` files (mono, PCM16/24/float32) are loaded and resampled to the active sample rate via `CabSimIr::load()` (`src/dsp/cabsim/loader.rs`). The prepared IR and pre-built `ConvEngine` are transferred to the audio thread via lock-free SPSC — the same pattern used for model loading (GC cascade for old engine disposal).

### CLAP Integration

The CLAP plugin exposes IR loading via:

- **GUI file browser** (Zone 1, filtered to `.wav`)
- **State save/load** (`ir_path` serialized in the preset blob)
- **SPSC `ClapParamPayload::LoadCabIr`** for RT-safe engine swap
- **`activate()` reconstruction** on `max_frames_count` changes, storing raw IR samples for fast rebuild without re-reading the file

In standalone mode, the `--cab <path>` CLI flag triggers IR loading; the `cabsim_producer` SPSC channel handles runtime buffer-size changes.

For full architectural decisions on validation and cross-reference, see [TODO-sprints.md](../TODO-sprints.md) (Épico 4).

## 6. Testing Strategy & Quality

The testing philosophy of NAM-rs prioritizes **quality over quantity**: we maintain only the layers that provide high-confidence signals, without circular redundancies.

### Test Organization

The project follows a strict hierarchy to ensure internal logic and the public API are validated efficiently:

1. **Unit Tests:** Focused on the internal logic of each module. Small files (< 300 lines) keep tests `inline` via `mod tests`. Larger files (e.g., `resampler.rs`, `lstm/mod.rs`) use the suffix `_test.rs` in the same directory to maintain readability.
2. **Integration Tests:** Located in `tests/`, they exercise the complete pipeline, model loading, and real-time stability.
3. **Benchmarks:** Located in `benches/`, they use `criterion` to monitor performance regressions in critical kernels.

### Active Layers

| Layer                         | Location                                         | Strength as Ground Truth          | What it captures                                                                                                 |
|:----------------------------- |:------------------------------------------------ |:--------------------------------- |:---------------------------------------------------------------------------------------------------------------- |
| **Golden Vectors**            | `tests/nam_infer_test.rs`, `tests/cpp_parity.rs` | ✅✅ External anchoring to C++    | Errors in kernel composition, end-to-end regressions, and parity vs. canonical reference (NeuralAmpModelerCore). |
| **PropTests (random)**        | `tests/proptest_math.rs`                         | ✅ native `f64` and `f32::tanh()` | SIMD numerical errors (RMSE) and SIMD vs. Scalar parity over a wide input space.                                 |
| **Bit Unit Tests**            | `src/math/common/tests.rs`                       | ✅ Direct bit operation           | Correctness of f32↔bf16/f16 conversion, FMA, and hardware setup (DAZ/FTZ).                                       |
| **A1/A2 Compatibility**       | `tests/a2_loader.rs`                             | ✅ Format Specification           | Ensures new loaders accept old models (Regression) and perform correct fallback to A2.                           |
| **NAMB v2 Validation**        | `tests/namb_v2_validation.rs`                    | ✅ Layout Specification           | Validates correctness of the pre-transposed layout (Gate-Major/Interleaved) vs. classic loading.                 |
| **PipeWire Integration**      | `tests/pw_integration_test.rs`                   | —                                 | PipeWire host initialization, buffer processing, and safe teardown.                                              |
| **Zero-Allocation Guard**     | `tests/nam_infer_test.rs`                        | —                                 | Ensures the hot-path does not allocate heap via `CountingAllocator` (RT-Safety).                                 |
| **Fuzz Testing (`proptest`)** | `tests/proptest_parsers.rs`                      | —                                 | ~45,000 adversarial inputs against JSON/.namb parsers to prevent vulnerabilities and panics.                     |
| **Soak Test (Endurance)**     | `tests/soak_test.rs`                             | —                                 | Long-duration numerical stability (10M+ frames). `#[ignore]` in CI; run via `bash utils/tests-long.sh`           |

### Validation Pyramid: Complementary Roles of Scalar Reference and Golden Vectors

Two validation layers capture **different classes of bug** and are deliberately not redundant:

- **Scalar reference (`src/math/common/scalar_ref.rs`) — tight-band oracle.** SIMD↔scalar parity must hold to `~1e-6` (only floating-point reassociation differs). Driven by **PropTests** over a wide random input space (10k+ cases with independent `f64`/`f32::tanh()` references), it localizes kernel bugs, sweeps edge cases the golden never exercises (remainder lengths `n % 8`, denormals, alignment boundaries), runs in the **fast hermetic lane** (`cargo test`, no C++ toolchain), and is the shared **cross-ISA invariant** when new SIMD paths (e.g., AVX-512) are added.
  - **Not a production fallback.** There is no scalar runtime path for non-AVX2 CPUs: detection (`src/math/common/dispatch/detect.rs`) **fail-fasts** because x86-64-v3 (AVX2+FMA) is the mandatory baseline. In production, the `_fallback` functions act only as the **tail/remainder and small-N handlers inside the SIMD kernels** (and as the native-`f32` path for select LSTM head ops).
- **Golden Vectors vs. NeuralAmpModelerCore — loose-band external truth.** By design (see [docs/perceptual_validation.md](perceptual_validation.md)), goldens are **not bit-exact** with the C++ reference: divergence is dominated by the FastMath approximations (see [docs/fastmath-approximations.md](fastmath-approximations.md)), so they are validated against an adaptive **tolerance band** (SNR/ESR/MR-STFT). They anchor the *algorithm/spec* against canonical truth, end-to-end, in the slow lane (`#[ignore]`, `utils/tests-long.sh`).

> **Why both?** A kernel bug small enough to stay inside the loose golden band but large enough to break the tight scalar parity is caught only by the scalar oracle. Conversely, a spec error shared by both scalar and SIMD implementations is caught only by the external golden. Removing either layer leaves a corresponding blind spot.
>
> **History:** Earlier *fixed-input* parity tests against a `ScalarRefMath` struct were removed (circular — validating against themselves) in favor of the PropTest approach above; the `ScalarRefMath` struct was eliminated while the scalar delegates remained. The self-referential goldens (NeuralAudio, `tests/regression_goldens.rs`, `tests/golden/`) were replaced with external anchoring to [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore). Reference models cover WaveNet (Standard/Lite/Feather/Nano), LSTM (1×8/1×12/1×16/1×24/1×40/2×8/2×12/2×16/2×24), and Linear (FIR-based models), with 5 accuracy metrics (MSE, MAE, SNR, PSNR, equiv. bits) computed in a single-pass fusion. See [tests/fixtures/golden_gen_build.sh](../tests/fixtures/golden_gen_build.sh) and [docs/dependencies.md §6](dependencies.md#6-dependencies-for-c-cross-validation-optional).

### Principle: "Todo Golden Deve Poder Falhar" (Every Golden Must Be Able to Fail)

A golden test is a **gate** — it must be capable of catching regressions. Three pitfalls
turn goldens into placebos that grant false confidence:

1. **Self-golden:** Validating output against itself. Passes by definition; catches nothing.
2. **Neutralized threshold:** `SNR ≤ 0 dB`, `ESR ≥ 1.0`, or `MSE ≥ 1e29` without rigid
   SNR+ESR compensation. Metrics that can never fire.
3. **Silent heuristic fallback:** Models without a calibrated entry falling back to
   `topology_thresholds()` with undocumented, untraceable thresholds.

**Meta-test enforcement** (`tests/threshold_calibration.rs`, part of T3.3/T3.4):

- `test_all_golden_models_have_calibrated_thresholds` — Every committed golden `.bin` MUST
  have an explicit calibrated entry. No silent fallback.
- `test_all_calibrated_entries_have_measurement_comments` — Every entry MUST document its
  provenance with `// Measured: SNR=…, ESR=…`.
- `test_all_thresholds_anti_placebo` — Each threshold dimension is independently checked:
  SNR ≤ 0, ESR ≥ 1, and MSE ≥ 1e29 without rigid SNR+ESR all fail individually.

> **A2 Exception:** A2 Full/Lite intentionally use `mse_limit = 1e30` because their ESR
> gates are ultra-strict (≤ 8e-8) and SNR ≥ 70 dB. The anti-placebo test accepts
> `mse_limit ≥ 1e29` only when SNR ≥ 40 dB and ESR < 0.1. Total neutralization still fails.

### Benchmarks and Performance

- **Criterion Benches:** `benches/inference_bench.rs` measures inference latency per model and SIMD architecture.
- **Long Run Benchmarks:** `Long_Run_*` group in `benches/inference_bench.rs` activated via `long_bench` feature, with 4096-sample blocks and `measurement_time(30s)` to measure real throughput in continuous operation. Activated via `bash utils/tests-long.sh`.

### IR Cabsim Testing Strategy

The IR Cabsim convolution stage (`src/dsp/cabsim/`) follows the **component-level testing** approach of the project's validation pyramid. Each aspect is validated independently, avoiding circular redundancies:

| Layer                      | Location                      | Count | What it validates                                                                               |
|:-------------------------- |:----------------------------- |:-----:|:----------------------------------------------------------------------------------------------- |
| **Convolution unit tests** | `src/dsp/cabsim/conv_test.rs` | 11    | UPOLS engine: partition logic, FDL, tail handling, DC, noise                                    |
| **Golden parity**          | `tests/cabsim_golden.rs`      | 6     | UPOLS vs. direct convolution O(N²) reference — ESR < 1e-5 in short/medium/long/stress scenarios |
| **Heap-audit**             | `tests/cabsim_heap_audit.rs`  | 4     | Zero heap allocation on the hot-path (RT-Safety)                                                |
| **Bitwise determinism**    | `tests/cabsim_golden.rs`      | 1     | Same inputs → bit-identical outputs across two engines                                          |
| **Passthrough**            | `tests/cabsim_golden.rs`      | 1     | Empty IR → unity gain (output ≈ input)                                                          |

#### Decision: End-to-End Pipeline Test with Cabsim Considered Unnecessary

> **Decision:** No dedicated end-to-end integration test is implemented for the cabsim stage interacting with the full DSP pipeline (input → inference → cabsim → output).
>
> **Justification:**
>
> 1. **Each component is individually validated:** The cabsim stage has 11 unit tests covering its internal convolution logic, 6 golden parity tests anchoring UPOLS against the mathematically-rigorous direct convolution reference, and 4 heap-audit tests ensuring RT-safety. These layers collectively cover correctness, numerical precision, and the zero-allocation contract.
> 2. **Pipeline integration is structurally decoupled:** The cabsim stage (`capture.rs:60-72`) is a stateless pass-through block — it receives pre-allocated input/output buffers, applies convolution, and returns. It does not share mutable state with the inference, gate, or output stages, nor does it introduce cross-stage coupling that warrants integration-level testing.
> 3. **Integration verified by code review:** The stage's insertion point in `capture.rs` and its interaction with buffer sizing (`partition_size` vs. host `max_frames_count`) have been verified during architectural review and are documented in the pipeline flow diagram (§5.2).
> 4. **Golden parity tests exercise realistic data paths:** The golden tests use synthetic IRs and mixed-sine signals at realistic lengths (up to 65536 samples), covering the same data paths the pipeline would exercise.

## 7. A2 Architecture: Current State (Beta)

The A2 architecture is NAM's next-generation format (NeuralAmpModelerCore v0.5.2+). NAM-rs provides a complete, high-performance, real-time safe implementation of the fixed A2 fast-path (**A2-Full** with 8 channels and **A2-Lite** with 3 channels), matching the behavior of `NAM/wavenet/a2_fast.cpp`.

### Microarchitectural Optimizations

To run the deep 23-layer A2 network within real-time budgets under AVX2, the engine employs specialized kernels:

- **Fully Unrolled GEMV (A2-Lite, CH=3):** Transposes and fully unrolls the matrix-vector multiplication for 3 channels. Convolutions for both $K=6$ (18 FMAs) and $K=15$ (45 FMAs) are hardcoded without loop overhead (`src/models/a2/conv1d_ch3.rs`), achieving peak throughput on low-channel counts.
- **Tap-Major Frame-Tiled Convolution (A2-Full, CH=8):** Processes blocks using a $T=4$ frame-tiled broadcast-FMA strategy (`src/models/a2/conv1d_ch8.rs`). Weights are permutated once on load into a `col-major-per-tap` layout (`w[k * 64 + in * 8 + out]`), enabling contiguous 256-bit SIMD loads of 8 outputs and register reuse via `_mm256_broadcast_ss` of input frames.
- **Branchless Pow2 Rings (`MirroredBuffer`):** Buffers of history for dilations use a virtual double-mapped ring topology. Read lookbacks are mapped branchless via a power-of-two bitwise mask (`& ring_mask`). The write cursor advances circularly, eliminating expensive `copy_within` (memmove) shifts inside the layer hot-path.
- **Bypass of General A2 Overhead:** Features unused by production capturing (FiLM, heterogenous activations, dynamic gating/gated/blended modes, `condition_dsp`, `bottleneck ≠ channels`) are kept out of the hot-path, parsing them into stub surfaces to maintain backward compatibility while avoiding runtime overhead.

### Slimmable Container and FSM Integration

NAM-rs supports the official A2 distribution format, where models are bundled together inside a `SlimmableContainer` (defining a `"SlimmableContainer"` config with submodels):

- **Pre-Allocated Submodels:** Both A2-Full (CH=8) and A2-Lite (CH=3) submodels are loaded, prewarmed, and held in memory. Swapping between them involves zero heap allocations.
- **FSM-Driven Degradation:** The `AdaptiveCompute` FSM monitors block deadlines. Under high CPU load, it triggers a downgrade transition (**A2-Full → A2-Lite**), reducing the active channels from 8 to 3.
- **Linear Crossfade:** To prevent audible switching transients (clicks), transitions perform a 32 ms linear crossfade blend between the outputs of the active and pending models, utilizing pre-allocated scratch buffers.

## 8. DAW Integration (CLAP Integration)

The architecture of NAM-rs supports the decoupling necessary for execution as a CLAP (Clever Audio Plug-in) plugin, enabling use in DAWs (Digital Audio Workstations).

- **`AudioHost` Trait:** Defines the agnostic communication interface between the DSP engine and the host. Located in `src/common/audio_host.rs`.
- **Feature Flags:** Build is controlled by flags (`standalone` vs `clap-plugin`), ensuring system dependencies (such as `pipewire`) are removed in the plugin binary for maximum portability.
- **Agnostic Parameters:** `NamPluginParams` (in `src/common/params.rs`) centralizes plugin state (`input_gain_db`, `output_gain_db`, `gate_threshold_db`, `model_path`), facilitating mapping for DAW automation and state persistence (save/load).

## 8.1 CLAP Architecture: Threads and Lifecycle

CLAP mode is architecturally distinct from Standalone: the host controls the lifecycle and provides buffers directly in `process()`, eliminating the need for `DspBridge`.

### Thread Model

```mermaid
graph LR
    subgraph Main_Thread ["Main Thread (Host)"]
        MT_Plugin["NamClapPlugin"]
        MT_Main["NamClapMainThread"]
        MT_State["State Save/Load"]
        MT_Params["Params Flush"]
        MT_GC["GC Drain"]
        MT_Log["HostLog (RT Events)"]
    end

    subgraph Audio_Thread ["Audio Thread (Host)"]
        AT_Proc["NamClapProcessor"]
        AT_Pipeline["DSP Pipeline"]
        AT_Smoother["ParamSmoother (IIR 1-pole)"]
        AT_GC_Push["GC Push (obsolete models)"]
    end

    subgraph Shared ["NamClapShared (align 128B)"]
        SPSC_Params["SPSC: ClapParamPayload"]
        SPSC_GC["SPSC: GcItem"]
        GC_Overflow["GcOverflowBuffer"]
        RT_Flags["RtStatusFlags (AtomicU64)"]
    end

    MT_Main -- "push(Params|LoadModel)" --> SPSC_Params
    SPSC_Params -- "pop()" --> AT_Proc
    AT_GC_Push -- "push(old_model)" --> SPSC_GC
    SPSC_GC -- "pop() + drop()" --> MT_GC
    AT_Proc -- "set_flag()" --> RT_Flags
    RT_Flags -- "check_and_clear()" --> MT_Log
```

### Decision: No `DspBridge` in CLAP

The `DspBridge` (lock-free double-buffer) exists in Standalone mode to synchronize two independent PipeWire streams (capture and playback). In CLAP mode, the host calls a single `process()` — input and output are buffers of the same cycle. Therefore, CLAP **does not use `DspBridge`**. The `DspPipelineContext` is constructed on-stack in `process()` with pre-allocated buffers and executed directly.

### GC (Garbage Collection) Strategy

Model switching in the audio thread is RT-safe:

1. The main thread loads the model (cold-path with allocations) and sends it via SPSC `param_tx`.
2. The audio thread replaces the active model and sends the old one via `gc_tx` to be dropped outside the RT thread.
3. `on_main_thread()` drains `gc_rx` and executes `drop()` on obsolete models.
4. If the GC channel is full, it uses `GcOverflowBuffer` (overwrite ring buffer with `AtomicPtr`).

### Decision: Reject `SwapStrategy<T>` for Standalone/CLAP Deduplication

**Context:** Task T9.7 investigated unifying the object-swap + GC-cascade logic shared between
`src/standalone/pw_host/rt_callback/{resampler_swap,commands,cabsim_swap}.rs` and
`src/clap/processor/events.rs` into a common `SwapStrategy<T>` in `src/dsp/`.

**Analysis:**

| Duplication site                        | Inline SLOC | Fix                           | SLOC saved |
| --------------------------------------- | ----------: | ----------------------------- | ---------: |
| `resampler_swap.rs` inline GC cascade   |          20 | Replace with `gc_cascade()`   |         19 |
| `commands.rs` model GC cascade          |          18 | Replace with `gc_cascade()`   |         17 |
| `clap/processor/gc.rs` `push_to_gc()`   |          15 | Delegate to `gc_cascade()`    |         10 |
| **Total** with `gc_cascade()` reuse     |             |                               |     **46** |

A `SwapStrategy<T>` would encapsulate only the `std::mem::replace(active, new)` + GC cascade pair.
Each drain site has unique logic interleaved with the swap (rate tracking in `drain_resamplers`,
`inject_rt_status` in model swaps, `set_max_buffer_size` in CLAP), so the abstraction would not
eliminate additional lines beyond what `gc_cascade()` already does.

**Decision:** Reject `SwapStrategy<T>` in `src/dsp/`. Justification:

1. **Net savings < 100 lines:** Even with `gc_cascade()` reuse, the total savings are ~46 SLOC —
   well below the 100-line threshold. `SwapStrategy<T>` does not bridge the gap because the
   interleaved mutation logic is type-specific and cannot be generically abstracted.
2. **Layering violation:** Placing `SwapStrategy<T>` in `src/dsp/` would create a dependency from
   the DSP layer on `common::spsc::gc` (GC infrastructure), breaking the current layering where
   GC is a `common` concern, not a DSP concern.
3. **No RT benefit:** The swap itself is `std::mem::replace` — zero-cost and proven RT-safe.
   Adding an abstraction layer risks compiler optimization barriers without measurable gain.
4. **Existing `gc_cascade()` suffices:** The free function in `src/common/spsc/gc.rs` already
   abstracts the 3-tier cascade (SPSC → parking-lot → overflow). The residual inline copies
   in `resampler_swap.rs` and `commands.rs` are fixed by adopting it, not by adding a new type.

**Consequences:**

- The standalone inline GC cascades in `resampler_swap.rs` and `commands.rs` are replaced with
  direct `gc_cascade()` calls.
- `push_to_gc()` in `clap/processor/gc.rs` delegates to `gc_cascade()` instead of duplicating it.
- `drain_parking_lot()` remains a distinct concern (re-draining parked items on each audio block),
  which `gc_cascade()` does not handle.

**References:** [src/common/spsc/gc.rs](../src/common/spsc/gc.rs),
[src/standalone/pw_host/rt_callback/resampler_swap.rs](../src/standalone/pw_host/rt_callback/resampler_swap.rs),
[src/clap/processor/gc.rs](../src/clap/processor/gc.rs).

### CLAP Extensions and Graphical Interface

The plugin implements 11 CLAP extensions: `audio_ports`, `params`, `state`, `latency`, `track_info`, `remote_controls`, `param_indication`, `preset_load`, `render`, `state_context`, and `gui`. The plugin operates strictly in mono to accommodate standard DAW workflows (mono-in/mono-out), while the GUI uses `egui` + `baseview` over a pure X11 backend (600×275px), with complete isolation between the UI thread and audio thread via atomic fields and SPSC.

For details on each extension, graphical stack, and windowing strategy, see [docs/clap_integration.md](clap_integration.md).

## 8.2 CLAP DSP Pipeline and Parameter Flow

The CLAP plugin's audio processing engine is designed for zero-jitter, low-latency, real-time audio operations. The host DAW invokes the `process()` callback inside [PluginAudioProcessor::process](../src/clap/processor/mod.rs#L178), executing a sequential pipeline of event handling, DSP, and telemetry compilation.

### CLAP DSP Pipeline Diagram

The following Mermaid diagram traces the detailed layout of parameter updates, event queues, DSP processing, and real-time safe cleanup steps inside the audio processing thread:

```mermaid
graph TD
    %% Parameter Flow
    subgraph ParameterFlow ["Parameter Flow (Main/GUI -> RT)"]
        GUI["GUI Knobs (egui)"] -- "atomic writes +\nbump generation (Release)" --> Atomics["Shared ui_to_rt atomics"]
        HostState["Host Preset / State Load"] -- "Load Model / Params" --> SPSC_Tx["SPSC param_tx (Main Thread)"]
        DAW_Auto["DAW Automation / MIDI"] -- "Sample-Accurate Queue" --> HostEvents["Events Input Queue"]
    end

    %% Audio Thread Callback
    subgraph RTThread ["RT Audio Thread (process)"]
        Entry["PluginAudioProcessor::process()"] --> Prio["Priority & DAZ/FTZ Setup\n(first block)"]
        Prio --> PE["process_events()"]

        %% Event Draining in process_events
        subgraph EvDraining ["process_events() details"]
            SPSC_Rx["Drain param_rx"] --> |LoadModel| SwapModel["Swap Model & Resampler\nPush old to gc_tx"]
            HostEventsD["Drain Host Events\n(ParamValue/Mod)"] --> UpdateTgt["Update local parameter targets"]
            GenGuard{"generation != last_seen?"} -->|Yes| SyncAtomics["Sync from Shared Atomics"]
            SyncAtomics --> UpdateTgt
        end

        PE --> EvDraining
        EvDraining --> DSPProc["process_dsp_audio()"]

        %% DSP Processing in process_dsp_audio
        subgraph DSPPipeline ["process_dsp_audio() Pipeline"]
            AudioPorts["Iterate Audio Ports"] --> BypassCh{"Bypass active?"}
            BypassCh -->|Yes| RunBypass["process_bypass() (copy / zero)"]
            BypassCh -->|No| ChanExt["extract_channels()"]
            ChanExt --> InputGain["Apply Input Gain\n(SIMD + Smoother)"]
            InputGain --> InputStage["apply_input_stage()\n(Dither & Gate)"]
            InputStage --> GateCh{"Gate open?"}
            GateCh -->|No| ZeroOut["Fill output with 0.0"]
            GateCh -->|Yes| Infer["run_inference()\n- NamResampler (Up)\n- NamModel::process()\n- NamResampler (Down)"]
            Infer --> OutputStage["apply_output_stage()\n(Dither comp & Gate fade\n& Adaptive Compute check)"]
            ZeroOut --> OutputStage
            OutputStage --> OutputGain["Apply Output Gain\n(SIMD + Smoother)"]
            OutputGain --> OutputPeaks["compute_output_peaks()\nStore peaks in Shared"]
        end

        DSPProc --> DSPPipeline
        DSPPipeline --> Telemetry["process_telemetry()\n- Read TSC cycles\n- Heap-audit assert"]
    end

    %% GC Drop
    SwapModel -.-> |gc_tx| SPSC_Gc["SPSC gc_rx"]
    SPSC_Gc -.-> |drop()| GcDrop["Main Thread GC Drain"]
```

### 8.2.1 Pipeline Execution Flow

The processing execution pathway inside [process_dsp_audio](../src/clap/processor/dsp/orchestrator.rs#L16) consists of the following consecutive stages:

1. **Bypass Evaluation:** Evaluates whether active bypass is requested via [process_bypass](../src/clap/processor/dsp/bypass.rs#L11). If bypass is active, the pipeline copies input samples directly to the output ports, writes zero/clipping telemetry, and short-circuits the downstream DSP/inference modules.
2. **Channel Extraction:** Calls [extract_channels](../src/clap/processor/dsp/channels.rs#L10) to map host audio ports (which might be mono or stereo depending on the DAW configuration) to thread-local aligned input buffers.
3. **Input Gain Stage:** Multiplies the input samples by the configured input gain using SIMD operations, driven by a sample-accurate [ParamSmoother](../src/dsp/smoother.rs#L12) to prevent zipper noise.
4. **Gate Parameter Refresh:** If gate parameters have been marked dirty, the pipeline dynamically computes the linear squared thresholds for opening and closing the gate.
5. **Input Pipeline Stage (Dither & Gate):** Calls [apply_input_stage](../src/dsp/pipeline/stages/input.rs#L47). This function injects a deterministic $-220\text{ dBFS}$ dither offset to avoid denormal numbers (preventing CPU performance degradation) and evaluates the Noise Gate state machine ([GateState](../src/dsp/gate.rs#L60)). If the gate is closed, the processing skips inference and proceeds to immediately fill the output buffers with silence.
6. **Model Inference:** If the gate is open, calls [run_inference](../src/dsp/pipeline/stages/inference.rs#L113) to run the neural net. If the host sample rate differs from the model's native rate, the [NamResampler](../src/dsp/resampler.rs#L283) up-samples the buffer. Next, the active neural model runs inference (`NamModel::process`), and the resampler down-samples the result back to the host rate.
7. **Output Pipeline Stage (Dither compensation & Gate Fade):** Calls [apply_output_stage](../src/dsp/pipeline/stages/output.rs#L21). This stage subtracts the compensatory dither offset, applies linear fade-in/out transitions when the gate opens or closes, and measures block execution time to run the **Adaptive Compute** FSM.
8. **Output Gain Stage:** Multiplies the output by the output gain, smoothed via [ParamSmoother](../src/dsp/smoother.rs#L12).
9. **VU Peaks Telemetry:** Computes the output peaks via [compute_output_peaks](../src/clap/processor/dsp/peaks.rs#L10) and stores them in shared memory for the GUI using [store_peaks](../src/clap/processor/dsp/peaks.rs#L95).
10. **High-Precision Telemetry:** Reads CPU cycle metrics at the end of the block via `minstant::Instant` in [process_telemetry](../src/clap/processor/dsp/telemetry.rs#L10) to compute the actual DSP load without system call overhead.

### 8.2.2 Parameter Flow and Synchronization

Parameters (e.g., gain, gate threshold, bypass state, and neural model files) are synchronized across threads via a lock-free protocol in [process_events](../src/clap/processor/events.rs#L21). Synchronization handles three incoming paths:

- **SPSC Queue (`param_rx`):** The Main Thread processes expensive operations (like loading models from disk or allocating memory) and transfers the results via [ClapParamPayload](../src/clap/plugin/shared.rs#L17) to the RT thread. If loading a model, [cold_load_model](../src/clap/processor/events.rs#L136) replaces the active pointers on the RT thread and pushes the old instances to `gc_tx` so the Main Thread can safely drop them outside the RT context.
- **Host DAW Events Queue:** The host DAW feeds sample-accurate automation and MIDI events into the processing block's input queue. The RT thread reads these events sequentially to update local parameter targets.
- **GUI Atomics Sync:** GUI controls (e.g., egui knobs) write parameter updates to atomic variables in `NamClapShared::ui_to_rt` and increment `gui_param_generation` using `Release` ordering. The RT thread performs an `Acquire` check of the generation count; if they differ, it pulls the updated atomic values and aligns local targets.

---

## 8.3 Architectural Decisions

Detailed decisions regarding the framework (`clack-plugin`), GUI (`egui` + `baseview`), and target DAWs are documented in [docs/clap_integration.md](clap_integration.md). A comprehensive guide to the graphical user interface architecture, rendering lifecycle, and thread synchronization is available in [docs/gui-architecture.md](gui-architecture.md).

### 8.3.1 Graphical Interface and GUI Sub-modules (CLAP GUI)

The graphical interface is decomposed from its original monolithic state into a structure of readable and reusable modules located in [src/clap/gui/ui/](../src/clap/gui/ui/) (see the detailed [docs/gui-architecture.md](gui-architecture.md) for full architectural mapping):

- **`mod.rs`:** Main drawing orchestrator. The `draw_ui` function delegates to 5 zone functions: `draw_zone1_identity`, `draw_zone2_controls`, `draw_zone3_meters`, `draw_zone4_bypass`, and `draw_zone5_status_bar`.
- **`zones/`:** One file per GUI zone — `identity.rs` (Zone 1 logo + model loader), `controls.rs` (Zone 2 knobs), `meters.rs` (Zone 3 adaptive VU meters), `bypass_zone.rs` (Zone 4 bypass toggle).
- **`status_bar/`:** Zone 5 footer — `orchestrator.rs` (layout + toast + A2 warning), `telemetry.rs` (DSP load, sample rate, latency strings), `metadata.rs` (model metadata line).
- **`meter/`:** GPU-accelerated VU meter rendering — `glow.rs` (OpenGL GLSL shaders), `orchestrator.rs` (draw entry point), `cpu.rs` (software fallback), `readout.rs` (peak readout labels).
- **`bypass.rs`:** Interactive design and behavior of the bypass toggle switch.
- **`colors.rs`:** HSL definitions for the plugin color palette and accent color resolution.
- **`focus.rs`:** Keyboard focus cycle management (Tab/Shift+Tab navigation).
- **`knob.rs`:** Custom high-precision rotary control widgets with drag gesture, fine-tune, and reset support.
- **`simd.rs`:** Visual component displaying the active SIMD instruction set badge.
- **`state.rs`:** Local persistent GUI state, VU peak-hold, telemetry cache, toast and error expiration.
- **`vsep.rs`:** Styled vertical separator lines between zones.
- **`test.rs`:** Automated egui interface tests and mocks.

#### Technical Decision: Adaptive VU Meter Layout (Mono vs. Stereo)

> **Decision:** The Zone 3 VU meter uses a dynamic layout to display either a single centered mono meter or dual L/R stereo meters based on the host routing context [active_channel_count](../src/clap/plugin/shared.rs#L75), rather than being locked to a mono meter to match the DSP engine's mono processing.
>
> **Motivation:** Inform the user of the signal level present on the processed channel and detect routing imbalances or mismatch configurations in the host DAW.
>
> **Implementation:**
>
> 1. The audio processing thread determines the active host output channel count in [extract_channels](../src/clap/processor/dsp/channels.rs#L10).
> 2. It updates the atomic variable [active_channel_count](../src/clap/plugin/shared.rs#L75) in [shared.rs](../src/clap/plugin/shared.rs) via relaxed memory ordering (`Ordering::Relaxed`).
> 3. The GUI thread dynamically loads the count and checks if the count is $\ge 2$ in [meters.rs](../src/clap/gui/ui/zones/meters.rs#L40). If so, it renders dual visual meters (L and R); otherwise, it falls back to a single centered mono meter.
>
> **Trade-off:** Displaying a stereo VU meter with a mono DSP engine introduces no real-time performance overhead. Only the left channel (L) is processed by the neural model, and the right channel (R) is mapped to the peak value of either a bypass buffer (if stereo bypass mode is supported) or a copy of the same buffer, without running redundant neural inference.
>
> **References:** [channels.rs](../src/clap/processor/dsp/channels.rs), [meters.rs](../src/clap/gui/ui/zones/meters.rs), [shared.rs](../src/clap/plugin/shared.rs).

### 8.3.2 Math & SIMD — Modular Reorganization

- **Decision:** Fragmentation of the monolithic mathematical infrastructure into domain-specific modules (`activations/`, `gemm/`, `dsp/`, `lstm/`, `wavenet/`).
- **Justification:** Reduces cognitive noise in files with 2000+ lines, allows isolated unit testing per kernel, and facilitates compiler inlining audits.
- **Elimination of Redundancy (VNNI):** The `Avx2VnniMath` struct was eliminated and replaced with a type alias for `Avx2Math` in `common/avx2_impl.rs`. The `VPDPBUSD` (VNNI-Int8) instruction offers no gains for the floating-point kernels of NAM-rs. The cleanup of the `Avx2Vnni`/`Avx512Vnni` `InstructionSet` variants has been completed; only `Avx512VnniBf16` remains as an actively used VNNI variant (BF16 dot-product for weight-compressed WaveNet models).
- **Design Debt (Dual Dispatch):** The system uses a "Dual Dispatch" structure where the `loader` dispatches to the `model`, which in turn uses the `SimdMath` trait. The dispatch abstraction in `math/common/dispatch.rs` is a recognized design debt point.

### 8.3.3 GUI Conditional Rendering (Idle Reduce)

> **Decision:** Implement a conditional frame-skipping strategy inside the GUI event loop (`WindowHandler::on_frame` in `window/handler.rs`) to avoid redundant paint operations and minimize CPU usage when the interface is idle.
>
> **Motivation:** Immediate-mode GUIs (like `egui`) execute the layout and drawing code on every frame. Since baseview runs a continuous rendering loop (typically targeting ~30–45 FPS via a 30 ms interval), multiple active instances of the plugin in a host DAW would consume excessive CPU cycles even when displaying static information, which is a common source of user complaints in DAW environments.
>
> **Implementation:**
> Inside the baseview paint handler (`on_frame`), we snapshot the VU meter peak-hold states (`peak_l_hold`/`peak_r_hold`) and run egui's layout phase. We then evaluate a skip condition before executing the expensive tessellation and OpenGL paint commands:
>
> ```rust
> let should_skip = !self.dirty
>     && !has_short_repaint
>     && !hold_changed
>     && time_since_paint < std::time::Duration::from_millis(22);
> ```
>
> - `!self.dirty`: Evaluates to true if no user input events (mouse move, click, keystroke, window resize) have been registered since the last paint cycle.
> - `!has_short_repaint`: Bypasses the skip check if egui requests a repaint delay of less than 50 ms. This ensures active animations (e.g., toast alerts, warning timeouts, or loading indicators) remain fluid.
> - `!hold_changed`: Bypasses the skip check if the peak-hold levels are decaying. Once hold levels decay and stabilize at zero, they stop forcing repaints.
> - `time_since_paint < 22ms`: Throttles the UI rendering rate to a maximum of ~45 FPS when the UI is actively being painted, ensuring fluid visual feedback while capping maximum CPU load.
>
> **Consequences:**
>
> - **Toast/Loading Animations:** Visual elements that animate or fade out cannot rely solely on the host scheduling frame ticks or on passive repaint flags. Instead, they must actively call `request_repaint()` on the `egui::Context` during their active duration.
> - **Reduced Idle CPU:** CPU utilization drops to virtually 0% when the UI is open but static (no audio playing and no user interaction).
>
- **References:** [handler.rs](../src/clap/gui/window/handler.rs), [gui-architecture.md](gui-architecture.md#L154-L172).

### 8.3.4 CLAP Mono Design and Stereophonic Discard

> **Decision:** The CLAP plugin is strictly configured as mono-in/dup-out for its core DSP pipeline. When loading a model pair, `model_r` (the right-channel model) is immediately discarded to the real-time SPSC garbage collection channel at the time of preset swap.
>
> **Motivation:** Minimizes DSP overhead and matches the expected routing semantics of guitar processors in modern DAWs, which primarily process a single input channel. Running a dual-channel model when only the left channel is processed would double CPU utilization without providing any auditory benefit.
>
> **Implementation:**
> Inside `cold_load_model` ([events.rs](../src/clap/processor/events.rs#L160)), the swap logic replaces the left-channel model with `model_l`. If `model_r` is present in the payload, it is sent straight to `push_to_gc()` to be dropped by the main thread.
>
> **References:** [events.rs](../src/clap/processor/events.rs), [channels.rs](../src/clap/processor/dsp/channels.rs).

### 8.3.5 Gain Multiplier Fusion: Standalone vs. CLAP

> **Decision:** Apply user gain controls and model calibration adjustments separately in the CLAP plugin pipeline, whereas the standalone client pre-fuses them into single input/output multipliers.
>
> **Motivation:** The CLAP plugin must support sample-accurate DAW parameter automation. Combining the static model adjustments (`input_level_dbu` and `loudness` calibration) with automated user parameters inside the real-time process loop would prevent efficient, isolated smoothing, causing computational overhead or audible artifacts (clicks/pops).
>
> **Implementation:**
> Standalone computes a combined multiplier using `compute_gain_multipliers()` in [rt_setup/mod.rs](../src/standalone/rt_setup/mod.rs#L25). CLAP uses smoothers (`smoother_in`/`smoother_out`) on the user gains during `apply_input_gain`/`apply_output_gain`, applying model calibration multipliers separately via the `DspPipelineContext` [input_gain_mult](../src/clap/processor/dsp/orchestrator.rs#L70).
>
> **References:** [orchestrator.rs](../src/clap/processor/dsp/orchestrator.rs), [rt_setup/mod.rs](../src/standalone/rt_setup/mod.rs).

### 8.3.6 Multi-Tiered Real-Time Garbage Collection Cascade

> **Decision:** Implement a three-tiered fallback queue structure (SPSC Queue -> Parking Lot Array -> Ring Buffer Overflow) to manage the safe disposal of heap-allocated resources from the audio thread without blocking.
>
> **Motivation:** Dropping heavy structures (e.g. neural models, resamplers) is not real-time safe because it triggers system deallocations which can cause CPU spikes and audio dropouts. The audio thread must delegate deallocation to the main thread in a lock-free, zero-allocation manner.
>
> **Implementation:**
>
> 1. **Primary Queue:** A lock-free SPSC queue (`gc_tx`/`gc_rx` via `rtrb`) of capacity 32 handles normal swaps.
> 2. **Parking Lot Array:** A 16-slot static array ([parking_lot](../src/clap/processor/state.rs#L79)) buffers items if the primary queue is temporarily full (e.g., during rapid parameter swaps). Drained at the start of every block.
> 3. **Overflow Buffer:** A 64-capacity overwriting ring buffer (`GcOverflowBuffer`) handles overflow as a last resort, setting the `RT_STATUS_GC_OVERFLOW` flag if items are dropped.
>
> **References:** [gc.rs](../src/clap/processor/gc.rs), [gc.rs](../src/common/spsc/gc.rs).

### 8.3.7 Neural Amp Model Testing and Fixture Policies

> **Decision:** Establish a strict, multi-tiered hierarchy for fixture usage in testing (synthetic boundary tests, C++ validation goldens, and self-goldens) and mandate that mock fixtures must be complete if used in load-success verification.
>
> **Motivation:** Standardizes test verification to guarantee DSP correctness, prevent regression of mathematical parity with C++, and handle known upstream limitations (e.g., A2 rendering instabilities in the C++ library) without silently compromising test integrity.
>
> **Implementation:**
>
> - **Synthetic Fixtures:** Microscopic configurations (e.g., shape mismatches, custom activations) to test parser safety.
> - **C++ Goldens:** Step-by-step parity fixtures compared in `cpp_parity.rs`.
> - **Self-Goldens:** Used as a fallback regression shield when C++ is unstable, pinning the upstream reference commit.
> - **Mock Rule:** Incomplete mock files must never be passed to tests assuming load success; they must assert failure, ensuring that tightened validation rules do not cause obscure failures elsewhere.
>
> **References:** [TODO-sprints.md](../TODO-sprints.md#L967), [tests-long.sh](../utils/tests-long.sh).

### 8.3.8 WaveNet Heterogeneous Layer Array Skip-Connection Head Cascade

> **Decision:** Seed the head accumulator of the second heterogeneous layer array (`array2`) in WaveNet models with the projected skip-connection head outputs from the first array (`array1`), aligning exactly with the reference NeuralAmpModeler C++ behavior.
>
> **Motivation:** Ensures mathematically identical inference parity with reference models. Previously, the Rust implementation processed the array head outputs independently and summed them at the end, leading to significant tonal/numerical divergence.
>
> **Implementation:**
> The head output of `array1` is passed to the start of `array2` processing. Rather than initializing `head_accum` to zero at the beginning of `array2` layers, it is seeded with `array1`'s projected head outputs. The final output is then obtained exclusively by scaling the output of `array2`'s head projection by `head_scale`: `out = head_scale * array2.head_outputs`.
>
> **References:** [model.rs](../src/models/wavenet/model.rs#L90), [layer_array.rs](../src/models/wavenet/layer_array.rs).

## 9. Error Catalog (NamErrorCode)

Typed error codes for structured diagnostics. Defined in `src/common/diagnostics/error_codes.rs`.

| Range   | Category                   | Examples                                                |
|:------- |:-------------------------- |:------------------------------------------------------- |
| `E1xxx` | Model loading (I/O, parse) | `E1100` FILE_NOT_FOUND, `E1201` NAMB_CRC32_MISMATCH     |
| `E2xxx` | PipeWire / Audio           | `E2100` PIPEWIRE_INIT_FAILED, `E2300` SCHED_FIFO_DENIED |
| `E3xxx` | SPSC / Communication       | `E3100` PARAM_CHANNEL_FULL, `E3101` GC_OVERFLOW         |
| `E4xxx` | Runtime / CLI              | `E4100` INVALID_GAIN_VALUE, `E4101` UNKNOWN_COMMAND     |
| `E5xxx` | System / Hardware          | *(reserved for future CPU/memory diagnostics)*          |

Each emitted diagnostic includes version, architecture, and timestamp to enable automated triage via the `diagnostico` skill (see [SKILL.md](../.agents/skills/diagnostico/SKILL.md)).

## 10. References

The following repositories and specifications are the primary references for NAM-rs:

- [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) - Reference implementation of NAM.
- [CLAP (CLever Audio Plug-in)](https://cleveraudio.org/) - Specification of the CLAP plugin format.
- [Clack Framework](https://github.com/prokopyl/clack) - Rust infrastructure for implementing CLAP plugins.
- [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio) - Historical reference; original golden vectors migrated to anchor on NeuralAmpModelerCore (see §6).
