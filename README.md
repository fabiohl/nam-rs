<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# 🎸 NAM-rs 3.1

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-green.svg) ![CLAP](https://img.shields.io/badge/CLAP-gray.svg) [![crates.io](https://img.shields.io/crates/v/NeuralAmpModeler-rs.svg)](https://crates.io/crates/NeuralAmpModeler-rs) [![docs.rs](https://img.shields.io/docsrs/NeuralAmpModeler-rs.svg)](https://docs.rs/NeuralAmpModeler-rs)

**NAM-rs** is a cutting-edge, high-fidelity, high-performance, real-time [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) core and client written in pure Rust. Designed for guitar amplifiers, pedals, and studio gear simulation on Linux, it operates both as a **native PipeWire standalone application** and as a **CLAP plugin with an integrated GUI** for digital audio workstations (DAWs), as well as an embeddable Rust DSP library.

> **❤️‍🔥 NAM-rs is in "public beta" stage.** Feedbacks, bug reports, suggestions, and testing are very welcome (and needed 😉)!

---

## ⚡ Why NAM-rs? Key Advantages

* **Pure Rust & Zero-Allocation RT Safety:** Built from the ground up to guarantee zero heap allocations, zero locks, and zero blocking I/O on the audio thread. Thread promotion (`SCHED_FIFO`), lock-free SPSC queues, and CPU core affinity ensure rock-solid determinism under heavy real-time loads.
* **Extremely Fast SIMD Inference:** Hand-crafted AVX2 (`x86-64-v3`) baseline vectorization and optional AVX-512 kernels deliver up to 45× speedups over scalar baselines. Even the most consuming .nam models keeps CPU consumption down to ~3% of a 1.33 ms real-time deadline on a cheap consumer CPU.
* **Uncompromising Audio Parity:** Validated against three independent test oracles (canonical C++ NAMCore f32, double-precision f64 reference oracle, and Cross-ISA parity). WaveNet models achieve mathematical parity down to imperceptible levels (`~1e-14`), and LSTM models hit exact bit-level matches (`0.00e0`).
* **Flexible Operating Modes:** Runs as a native PipeWire standalone client or as a CLAP plugin featuring an egui-powered GUI on popular DAWs such as Bitwig and Reaper. Supports both Live mode (zero added latency) and HQ/Offline render mode (24-stage polyphase FIR oversampling with >100 dB stopband rejection).
* **Const-Generic Optimization:** Static WaveNet and LSTM variants leverage Rust const generics so kernel sizes and channel counts are known at compile time, enabling aggressive LLVM compiler optimization such as SIMD, instruction reordering and loop unrolling.

---

## 🧠 Supported Models

NAM-rs natively supports Neural Amp Modeler (A1 and A2 architectures), ConvNet topologies, and Impulse Response (`.wav`) convolution files.

* **Const-Generic Profiles (Static Mode — Maximum Performance):** Compile-time sized structures for peak execution speed.
  * **WaveNet A1:** Standard (16 channels), Lite (12 channels), Feather (8 channels), and Nano (4 channels)
  * **WaveNet A2:** Full (8 channels) and Lite (3 channels)
  * **LSTM:** 1 and 2 Layers (Hidden Size 3 to 40: `1×3`, `1×8`, `1×12`, `1×16`, `1×24`, `1×40`, `2×8`, `2×12`, `2×16`, `2×24`)
  * **Linear:** FIR-based model (dot product of input history with weights + bias or Partitioned FFT convolution)
* **Dynamic Fallback Profiles (Free-Shape Mode):**
  * **WaveNet Dyn**, **LSTM Dyn**, **WaveNet A2 Dyn / Cascade**, and **ConvNet**: Runtime-dimensioned topologies executing with guaranteed real-time safety.
  * **Slimmable Container:** Bundles multiple submodels for dynamic quality transitions.

---

## ✨ Architecture

NAM-rs adopts an opinionated design focused on four core pillars:
> **Note:** While not ideologically opposed to other platforms, opinionated choices were maintained across the entire stack to preserve focus and, above all, extract the absolute maximum from these technical decisions.

1. **Native Linux & Modern Plugin Format:** Standalone mode integrates directly with the PipeWire graph server. Plugin mode targets the modern CLAP standard exclusively, eliminating legacy LV2 or VST wrapper overhead.
2. **Accelerated SIMD Execution:** Requires `x86-64-v3` (AVX2 + FMA). FastMath activation functions (tanh, sigmoid) use Padé approximations and Newton-Raphson reciprocal square roots in 256-bit registers. AVX-512 multiversioning processes 16 floats or 32 bf16 values per instruction on ZMM hardware (Intel Xeon, AMD Zen 4+).
3. **Deterministic Audio Thread:** Audio processing runs on a dedicated thread promoted to `SCHED_FIFO` with explicit CPU core affinity to avoid cache misses. Control communication (CLI/GUI ↔ Audio) uses 128-byte aligned lock-free SPSC ring buffers.
4. **Compile-Time Safety & Code Quality:** Pure Rust design eliminates undefined behavior while giving LLVM complete visibility to optimize static WaveNet and LSTM topologies via const generics.

---

## 🏆 Quality & Performance Benchmarks

Our automated Quality Dashboard ensures top-tier performance without sacrificing audio fidelity:

* **Zero-Compromise Fidelity:** Bit-exact parity with canonical NAMCore C++. WaveNet models match within numerical precision (`~1e-14`), and BossLSTM models achieve an exact `0.00e0` error score.
* **Extreme CPU Headroom:** A full WaveNet Standard CH16 model processes a 64-sample block in just **42 µs**—consuming only **3.1%** of a strict 1.33 ms real-time deadline in consumer cheap CPUs. Light models like LSTM 1x16 consume under **0.6%**.
* **Rock-Solid Stability:** Stress tests confirm the Lock-Free SPSC architecture withstands over 1,000 concurrent model hot-swaps under heavy load with zero heap allocations on the audio thread and no clicks or dropouts.

---

## 🚀 Quick Start

### Prerequisites

* **OS & Kernel:** Modern Linux distribution with PipeWire audio server (nam-rs was developed and tested on Ubuntu 25.10 / 26.04).
* **CPU:** `x86-64-v3` processor with AVX2 and FMA support (Intel Haswell 2013+, AMD Excavator 2015+; 2019+ CPUs recommended).
* **Toolchain:** Rust 1.85+ (`rustup`/`cargo`).
* **Development Packages:**
  `sudo apt install build-essential cmake g++ python3 pkg-config pipewire pipewire-bin pipewire-utils libpipewire-0.3-dev clang libclang-dev qpwgraph vlc ffmpeg libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev git curl linux-tools-common linux-tools-generic linux-tools-$(uname -r) llvm-22-tools jq ripgrep fd-find`

* **Cargo Utilities:**
  * `cargo install cargo-edit`
  * `cargo install --git https://github.com/free-audio/clap-validator.git`
  * `cargo install cargo-pgo`
  * `cargo install cargo-show-asm`
  * `rustup component add llvm-tools-preview`

* **System Real-Time Configuration:**
  1. Add your user to the `audio` group: `sudo usermod -aG audio $USER`
  2. Configure real-time limits in `/etc/security/limits.d/audio.conf`:

     ```text
     @audio   -  rtprio     95
     @audio   -  memlock    unlimited
     ```

  3. Create a udev rule in `/etc/udev/rules.d/99-audio-dma-latency.rules` to lock CPU wake latency:

     ```text
     KERNEL=="cpu_dma_latency", GROUP="audio", MODE="0664"
     ```

  4. Reload rules or reboot: `sudo udevadm control --reload-rules && sudo udevadm trigger`

### Installation & Build

#### From Source (Recommended)

After cloning the repository, build and install release binaries (Standalone CLI to `~/.local/bin/nam-rs` and CLAP plugin to `~/.clap/nam-rs.clap`):

```bash
./utils/build-release.sh
```

---

## 💻 Usage

### Standalone Mode (CLI)

```bash
nam-rs -m /path/to/model.nam               # Default: live mode (zero added latency)
nam-rs --oversample 2x -m model.nam         # 2× oversampling (12-sample latency)
nam-rs --oversample 4x -m model.nam         # 4× oversampling (24-sample latency, maximum anti-aliasing)
nam-rs --oversample off --cab cab.wav -m model.nam  # Live mode + IR cabinet loading
```

The `--oversample` flag (`--os`) supports `off`, `2x`, or `4x`. `off` is optimized for live playing (minimum latency). `4x` is ideal for mixdown and offline rendering.

### CLAP Plugin

The CLAP plugin GUI presents an **Oversampling** control (**Off** | **2×** | **4×**) and is fully automatable via DAW parameter automation (`PARAM_OVERSAMPLE`). Model and oversampling changes execute lock-free without interrupting audio playback or causing clicks.

### Operational Modes Matrix

| Aspect               | Live (default)       | HQ / Offline               |
|:-------------------- |:-------------------- |:-------------------------- |
| Oversampling         | `Off`                | `4×`                       |
| Adaptive compute     | Active (FSM-driven)  | Disabled (deterministic)   |
| Latency              | 0 samples            | 24 samples (~0.54 ms)      |
| Recommended Use      | Monitoring, live gig | Export, mixdown, mastering |

### Diagnostic & Support Telemetry

Run diagnostic self-checks or output support telemetry:

* **Standalone:** `nam-rs --diagnose`
* **CLAP Plugin:** Click **"Copy Diagnostic"** (`ℹ`) in the GUI status bar to copy telemetry to clipboard and `~/.cache/nam-rs/diagnostic-<timestamp>.txt`.
* **Issue Reporting:** Paste the generated telemetry block into a GitHub Issue or submit it to the diagnostic assistant ([.agents/skills/diagnostico/SKILL.md](.agents/skills/diagnostico/SKILL.md)).

---

## 📦 **For Rust Developers [crates.io](https://crates.io/crates/NeuralAmpModeler-rs):**

📖 Full API documentation & integration guide: **[docs.rs/NeuralAmpModeler-rs](https://docs.rs/NeuralAmpModeler-rs)**

```toml
[dependencies]
NeuralAmpModeler-rs = { version = "x.y.z", default-features = false }
```

---

## 📚 Documentation & Technical References

* [docs/architecture.md](docs/architecture.md) — Comprehensive architecture specification, thread model, and module taxonomy
* [docs/audio_fidelity_map.md](docs/audio_fidelity_map.md) — Quality trade-off map for DSP decisions (activations, oversampling, adaptive compute)
* [docs/fastmath-approximations.md](docs/fastmath-approximations.md) — Tanh/sigmoid approximation benchmarks and math details
* [docs/clap_integration.md](docs/clap_integration.md) — CLAP thread model, lock-free GC, and GUI implementation
* [docs/namb-spec.md](docs/namb-spec.md) — Binary `.namb` container spec and CRC32 layout
* [docs/testing.md](docs/testing.md) — Test suite layout, execution phases, and verification rules
* [docs/perceptual_validation.md](docs/perceptual_validation.md) — Perceptual measurement framework (ESR, MR-STFT, ASR, LUFS)
* [docs/cpp_parity_map.md](docs/cpp_parity_map.md) — Line-by-line parity audit against NeuralAmpModelerCore
* [docs/benchmarks.md](docs/benchmarks.md) — Criterion benchmark methodology and performance regression gates
* [docs/research-references.md](docs/research-references.md) — Scientific literature and DSP reference bibliography
* [docs/functional-tests.md](docs/functional-tests.md) — Manual QA checklist for CLAP plugin verification
* [docs/postmortem-libm-symbol-interposition.md](docs/postmortem-libm-symbol-interposition.md) — Analysis of initialization symbol interposition resolution
* [tests/fixtures/README.md](tests/fixtures/README.md) — Golden vector formats and stress signal generation instructions

---

## 🧪 Tests & Quality Assurance

NAM-rs runs over **1,600+ automated checks** backed by three independent verification oracles:

| Oracle                   | Purpose                                                        | Target Baseline                                            | Test Suite                      |
|:------------------------ |:-------------------------------------------------------------- |:---------------------------------------------------------- |:------------------------------- |
| **C++ NAMCore** (f32)    | Parity with canonical C++ implementation                       | External golden vectors (ES-R, SNR, MR-STFT)               | `tests/cpp_parity.rs`           |
| **f64 Reference Oracle** | Absolute mathematical accuracy                                 | Double-precision forward pass                              | `tests/reference_oracle_f64.rs` |
| **ISA Parity**           | Vectorization consistency across platforms                     | Bit-equivalence between AVX2, AVX-512, and scalar          | `tests/isa_parity.rs`           |

### Automated Test Scripts

```bash
# 1. Formatting, SPDX checks, and Clippy lints
utils/lints.sh

# 2. Fast test suite (unit, integration, CLAP plugin audit)
utils/tests-quick.sh

# 3. Quality Dashboard (benchmarks & audio fidelity quality contract check)
utils/quality-dashboard.sh --check docs/quality-contract.txt
```

---

## 🤝 Contributing & Feedback

Contributions, feedback, and real-world testing are warmly welcomed!

* **Test Models:** Try your favorite `.nam` models and IR files and let us know your experience.
* **Report Issues:** Submit detailed bug reports or feature suggestions on GitHub.
* **Code & Docs:** Pull requests for optimizations, bug fixes, or documentation enhancements are welcome.

---

## 🙏 Credits & Acknowledgments

* **Steven Atkinson** — Creator of [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler) for pioneering deep learning amplifier modeling and sharing the ecosystem with the community.
* **Mike Oliphant** — Author of [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio), this codebase provided invaluable insight into WaveNet inference in the nam-rs birth.

---

## ⚖️ License & AI Transparency

### AI Transparency Note

The architecture, core engineering decisions, mathematical verification framework, and project orchestration are the intellectual work of the maintainer. The implementation was accelerated through pair programming with Artificial Intelligence (*Vibe Coding*) using models like Gemini, Claude, and DeepSeek within Google Antigravity IDE and Kilo Code.

### License

This project is licensed under the **Apache License, Version 2.0**. See [LICENSE.txt](LICENSE.txt) for details.
