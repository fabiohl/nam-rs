<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->
# 🎸 NAM-rs 1.5.0

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-green.svg) ![CLAP](https://img.shields.io/badge/CLAP-gray.svg)

> ⚠️ **Standalone PipeWire:** STABLE | **CLAP Plugin:** BETA (GUI completa, 8 extensões)

**NAM-rs** is a real-time [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) client for simulating guitar amplifiers, pedals, and studio gear.

It aims to maintain parity with the standard NAM implementation while introducing several performance improvements and optimizations.

It is fully optimized for maximum performance and ultra-low latency. This is achieved through a clean and organized codebase, intensive use of modern SIMD instructions (AVX2/FMA/x86-64-v3), and modern PipeWire/Linux features.

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

2. **CLAP Plugin:** `.so` library for use in DAWs (such as Bitwig Studio, Fender Studio Pro, etc.). Full GUI with knobs, VU meters, drag-and-drop model loading, telemetry, and 8 CLAP extensions.

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
4. **Pure Rust:** The choice of Rust is not just about "hype". Besides high performance, being a compiled language structurally similar to C/C++, it offers a modern, expressive syntax with compile-time safety and performance guarantees. For example, static versions ([wavenet.rs](file:///home/fabio/nam-rs/src/dsp/wavenet.rs)) where kernel size and channels are known at compile time allow aggressive loop unrolling by LLVM.

---

## 🚀 Quick Start

### Prerequisites

* A relatively recent Linux Kernel and PipeWire audio server. Development and testing are performed on Ubuntu 25.10 and 26.04.
* An `x86-64-v3` processor with AVX2 and FMA support (Intel ≥ Haswell 2013, AMD ≥ Excavator 2015). CPUs from 2019 onwards are highly recommended for NAM neural networks.
* A recent Rust toolchain (`rustup`/`cargo`). Version 1.94 was used during most of the development.
* Development packages:
  `sudo apt install build-essential cmake pkg-config pipewire libpipewire-0.3-dev clang libclang-dev qpwgraph libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev git curl linux-tools-generic`
* Cargo utilities (required for QA):
  * `cargo install cargo-edit`
  * `cargo install --git https://github.com/free-audio/clap-validator.git`
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
* [docs/benchmarks.md](docs/benchmarks.md) — How to interpret Criterion performance metrics
* [docs/clap_integration.md](docs/clap_integration.md) — CLAP (Clever Audio Plug-in) integration strategy

---

## 🧠 Supported Models

NAM-rs natively supports Neural Amp Modeler (.nam or .namb) files. Impulse Response (.wav) files are not supported.
Currently, the "A1 Architecture" of NAM is fully supported. "A2 Architecture" support is in **staging** (scaffolding and loader are ready).

Two levels of parsing operations are provided:

* **Static Mode (Ultra Performance):** *Const Generics* structures sized at compile time.
  * **WaveNet:** Standard (16×8), Lite (12×6), Feather (8×4), and Nano (4×2)
  * **LSTM:** 1 and 2 Layers (Hidden Size 8 to 24: `1×8`, `1×12`, `1×16`, `1×24`, `2×8`, `2×12`, `2×16`)
* **Dynamic Mode (Absolute Flexibility):** Fallback activated automatically when loading `.nam` arrangements with uncatalogued geometries (arbitrary `num_layers` and `channels`), operating without *loop unrolling*.

---

## 🧪 Tests & Validation

NAM-rs maintains a suite of approximately **220 automated checks**. To simplify development and QA flows, use the scripts located under `utils/`:

```bash
# 1. Lint & Quality (Formatting + Clippy + Feature Matrix)
utils/lints.sh

# 2. Standard Suite (Unit + Integration + Fast Benchmarks)
utils/tests-cargo.sh

# 3. Soak & Stress Tests (Long-duration verification)
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
* **Memory Resilience:** Ring buffer boundary stress in `VirtualRingBuffer`.

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
* **Mike Oliphant** — For the exceptional [NeuralAudio (C++)](https://github.com/mikeoliphant/NeuralAudio) library, which served as a direct reference for porting inference logic to this engine.

---

## ⚖️ License & Transparency (Vibe Coding)

**AI Transparency Note:** The architecture, rigorous engineering decisions, documentation, agent orchestrations, and curation of this project are the intellectual work of the maintainer. However, the source code itself was generated and iterated with the assistance of Artificial Intelligence (*Vibe Coding*), specifically using the Google Antigravity IDE.

This project is licensed under the **Apache License, Version 2.0**. See the `LICENSE` file for details.

> [!NOTE]
> **Developer Documentation & Code Comments:** Please note that developer comments (`//`, `///`, `//!`) in the source code and reference documentation under the `docs/` folder will remain in Portuguese for now.
