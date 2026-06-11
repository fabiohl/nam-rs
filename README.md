<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# 🎸 NAM-rs 2.0.0

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-green.svg) ![CLAP](https://img.shields.io/badge/CLAP-gray.svg)

**NAM-rs** is a cutting-edge, high-fidelity real-time [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) client designed for simulating guitar amplifiers, pedals, and studio gear with absolute precision. Powered by custom minimum-phase polyphase FIR resamplers and hand-crafted SIMD vectorized kernels (AVX2/AVX-512 delivering up to 45x speedups over scalar baselines), it guarantees **absolute real-time safety** with zero heap allocations on the audio thread. Sonically, it maintains uncompromising fidelity: rigorous cross-validation (Cross-Validation v2) using multi-component stress signals and advanced perceptual metrics—such as Error-to-Signal Ratio (ESR) and Multi-Resolution STFT—confirms that the generated audio achieves mathematical and auditory parity with the canonical C++ reference implementation under any sample rate.

Two operation modes are available: **Standalone** (native PipeWire executable) and **CLAP Plugin** (`.so` library with full GUI for DAWs).

The standard NAM model is mono by definition. To respect traditional DAW workflows, the **CLAP Plugin** operates strictly as a mono-in/mono-out effect, letting the host DAW manage channel routing (such as routing a mono signal on a stereo track). In contrast, the **Standalone** mode supports stereo processing as a built-in convenience feature for native PipeWire audio environments, running with ultra-low latency and minimal CPU usage — a real payoff given that most native hardware setups operate in stereo.

---

## 🛠️ Operation Modes

NAM-rs can be compiled in two main modes via *feature flags*:

1. **Standalone (default):** Native Linux binary for PipeWire. Immediate musical use with low latency and direct integration via `qpwgraph`.
   
   ```bash
   # Default build (standalone)
   cargo build --release --features standalone
   ```

2. **CLAP Plugin:** `.so` library for use in DAWs (such as Bitwig Studio, Fender Studio Pro, etc.). Full GUI with knobs, VU meters, model loading (drag-and-drop is pending upstream `baseview` updates on Linux), telemetry, and 8 CLAP extensions.
   
   ```bash
   # CLAP Plugin build
   cargo build --release --no-default-features --features clap-plugin --lib
   ```

---

## ✨ Architecture

NAM-rs adopts an opinionated architecture focused on four pillars:

1. **Native Linux & Modern Architecture:** Standalone mode integrates directly with the PipeWire server as a native client, managing its audio ports directly in PipeWire's *Graph Engine*. Plugin mode supports only the CLAP format, which is highly efficient and modern. We chose not to support legacy (LV2) or overly complex (VST) formats.
2. **Ultra-Fast SIMD Inference:** The baseline target is `x86-64-v3` (AVX2 + FMA are mandatory). Activation functions (tanh, sigmoid) use FastMath approximations (Padé + Newton-Raphson rsqrt) in 256-bit registers. AVX-512 multiversioning is implemented via `Avx512Math` for ZMM hardware (Intel Xeon, AMD Zen 4+), processing 16 floats per instruction. WaveNet operates in **Batch GEMM** (blocks of up to 64 frames per invocation).
3. **Real-Time Determinism:** The DSP thread is promoted to `SCHED_FIFO` with strict CPU affinity (*Core Affinity*), preventing core migrations and cache misses. CLI ↔ DSP communication uses a 128-byte aligned SPSC ring buffer. **Zero heap allocations** are made during audio processing.
4. **Pure Rust:** The choice of Rust is not just about "hype". Besides high performance, being a compiled language structurally similar to C/C++, it offers a modern, expressive syntax with compile-time safety and performance guarantees. For example, static WaveNet variants in `src/models/wavenet/` use const generics so kernel size and channel count are known at compile time, enabling aggressive loop unrolling by LLVM.

---

## 🚀 Quick Start

### Prerequisites

* A relatively recent Linux Kernel and PipeWire audio server. Development and testing are performed on Ubuntu 25.10 and 26.04.

* An `x86-64-v3` processor with AVX2 and FMA support (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015). CPUs from 2019 onwards are highly recommended for NAM neural networks.

* A recent Rust toolchain (`rustup`/`cargo`). Version 1.94 was used during most of the development.

* Development packages:
  `sudo apt install build-essential cmake g++ python3 pkg-config pipewire pipewire-bin pipewire-utils libpipewire-0.3-dev clang libclang-dev qpwgraph libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev git curl linux-tools-common linux-tools-generic linux-tools-$(uname -r) bolt-22 jq ripgrep fd-find`

* Cargo utilities & Rustup components (required for QA & optimizations):
  
  * `cargo install cargo-edit`
  * `cargo install --git https://github.com/free-audio/clap-validator.git`
  * `cargo install cargo-pgo`
  * `cargo install cargo-show-asm`
  * `rustup component add llvm-tools-preview`

* To ensure the engine runs flawlessly under realistic NAM models (especially "Lite" and "Standard"), it is crucial to grant advanced SCHED policies to the binary. Add your user to the system's `audio` group and edit your limits:
  
  1. `sudo usermod -aG audio $USER`
  
  2. Create or edit the limits file (e.g., `sudo nano /etc/security/limits.d/audio.conf`):
     
     ```text
     @audio   -  rtprio     95
     @audio   -  memlock    unlimited
     ```

* Create a *udev* rule to allow the `audio` group to lock CPU wake latency (C-states):
  
  1. `sudo nano /etc/udev/rules.d/99-audio-dma-latency.rules`
  
  2. Reload rules or reboot: `sudo udevadm control --reload-rules && sudo udevadm trigger`
     
     ```text
     KERNEL=="cpu_dma_latency", GROUP="audio", MODE="0664"
     ```

* Setting your CPU scaling governor (`intel_pstate` or `amd_pstate`) to **Performance** is also highly recommended:
  
  * Modern desktops (such as GNOME on Ubuntu/Fedora or KDE Plasma) manage this natively via `power-profiles-daemon`.
  
  * If you prefer `tlp`, you can edit `/etc/tlp.conf`:
    
    ```text
    CPU_SCALING_GOVERNOR_ON_AC=performance
    CPU_SCALING_GOVERNOR_ON_BAT=powersave
    ```

### Build & Run (Standalone Mode)

```bash
git clone https://github.com/fabiohl/nam-rs.git
cd nam-rs
cargo build --release --features standalone
```

*Note: `.cargo/config.toml` allows configuring a build optimized specifically for your current CPU ("march=native").*

To start audio processing:

```bash
target/release/nam-rs --model tests/nam_files/NEVE1073-Standard.nam
target/release/nam-rs --model tests/fixtures/models/BossWN-standard.nam --input-gain -3.0 --output-gain 0.0
# On lower-end machines, increase the buffer size to reduce CPU load:
target/release/nam-rs --model HeavyModel.nam --buffer-size 512
```

You can use `qpwgraph &` as a visual PipeWire connection editor. Once started, the node appears in the PipeWire patchbay.

### Telemetry & Monitoring

Every 10 seconds, NAM-rs prints a performance report in the terminal to monitor processing health:

`📊 DSP Telemetry (10s): 262µs (Median) | 524µs (P99) | 1048µs (Max) [938 blocks]`

* **Median**: Typical processing cost per block. Values close to 0µs indicate that the *Silence Bypass* is active, saving CPU.
* **P99 (Stability)**: The most critical indicator. Shows that 99% of the blocks were processed below this time. If the P99 approaches your buffer time budget (e.g., 5333µs for a 256-sample buffer at 48kHz), the risk of audio dropouts (XRUNs) increases.
* **Max**: The worst-case latency recorded in the interval, useful for detecting spikes caused by OS interrupts.
* **Blocks**: The total count of processed blocks in the telemetry interval.

---

## 📚 Documentation

* [docs/architecture.md](docs/architecture.md) — Topology, modules, and design decisions
* [docs/dependencies.md](docs/dependencies.md) — System dependencies and Rust crates
* [tests/fixtures/README.md](tests/fixtures/README.md) — Golden vector format, stress signal design, and regeneration instructions
* [docs/benchmarks.md](docs/benchmarks.md) — How to interpret Criterion performance metrics
* [docs/clap_integration.md](docs/clap_integration.md) — CLAP (Clever Audio Plug-in) integration strategy
* [docs/troubleshooting.md](docs/troubleshooting.md) — Technical troubleshooting and diagnostics support

---

## 🧠 Supported Models

NAM-rs natively supports Neural Amp Modeler (.nam or .namb) files. Impulse Response (.wav) convolution (IR cabsim) is supported as an optional post-inference stage — load cabinet IRs via `--cab <path>` in standalone mode or via the GUI file browser in the CLAP plugin.
Currently, the "A1 Architecture" of NAM is fully supported. The "A2 Architecture" is supported (currently in Beta stage) with the fixed fast-path (**A2-Full** 8 ch / **A2-Lite** 3 ch), a `SlimmableContainer` for runtime Full↔Lite switching, and integrated into the Adaptive Compute FSM.

Two levels of parsing operations are provided:

* **Static Mode (Ultra Performance):** *Const Generics* structures sized at compile time.
  * **WaveNet:** Standard (16×8), Lite (12×6), Feather (8×4), and Nano (4×2)
  * **WaveNet A2:** Full (8 channels) and Lite (3 channels)
  * **LSTM:** 1 and 2 Layers (Hidden Size 8 to 40: `1×8`, `1×12`, `1×16`, `1×24`, `1×40`, `2×8`, `2×12`, `2×16`, `2×24`)
  * Non-catalogued geometries fail to load with a clear diagnostic error.

---

## 🧪 Tests & Validation

NAM-rs maintains a suite of approximately **350+ automated checks** anchored against the canonical [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) implementation. To simplify development and QA flows, use the scripts located under `utils/`:

```bash
# 1. Lint & Quality (Formatting + Clippy + Feature Matrix)
utils/lints.sh

# 2. Standard QA Suite (Unit/Integration Tests + CLAP Plugin Build/Heap-Audit + CLAP Validator)
utils/tests-cargo.sh

# 3. Long-Duration Stress & Audit Suite (Soak + Proptests + Heap-Audit/Parity + CLAP Release Validation + Benchmarks)
utils/tests-long.sh
```

For manual or specific execution, you can run cargo directly:

* **Inline Unit Tests:** `cargo test --lib`
* **Specific Integration Tests:** `cargo test --test nam_infer_test`
* **Fuzz Testing via proptest:** `cargo test --test proptest_parsers`

### Stability Testing (Soak Test)

To ensure the engine remains stable during hours of continuous usage, NAM-rs includes a **Soak Test** suite processing millions of frames (e.g., 10M+ of silence/noise, 100M+ ring buffer cycles). These tests are designed to detect:

* **Numerical Drift:** Rounding error accumulations in filters and resamplers.
* **FSM Stability:** Integrity of Gate counters and fade transitions.
* **Memory Resilience:** Ring buffer boundary stress in `MirroredBuffer`.

Run the full battery: `bash utils/tests-long.sh`

---

## 🤝 Contributing

Contributions are welcome! The project is in active development.

* Tests + tests + tests + tests...
* Although AVX-512 is supported, testing on a capable CPU is highly appreciated.
* Before submitting PRs, run the test suites mentioned above. Use the agentic workflows in `.agents/`.

---

## 🙏 Credits & Acknowledgments

This project builds upon the logic, science, and inspiration of notable works in the audio and AI communities:

* **Steven Atkinson** — For pioneering the [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler), his research on amplifier modeling with deep learning, and sharing the ecosystem.
* **Mike Oliphant** — For the exceptional [NeuralAudio (C++)](https://github.com/mikeoliphant/NeuralAudio) library, which served as a historical reference for the original A1 inference logic.

---

## ⚖️ License & Transparency (Vibe Coding)

**AI Transparency Note:** The architecture, rigorous engineering decisions, documentation, agent orchestrations, and curation of this project are the intellectual work of the maintainer. However, the source code itself was generated and iterated with the assistance of Artificial Intelligence (*Vibe Coding*), specifically using the Google Antigravity IDE.

This project is licensed under the **Apache License, Version 2.0**. See the `LICENSE` file for details.
