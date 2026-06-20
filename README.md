<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# 🎸 NAM-rs 2.0

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/Linux%20x86__64-lightgrey.svg) ![PipeWire](https://img.shields.io/badge/PipeWire-green.svg) ![CLAP](https://img.shields.io/badge/CLAP-gray.svg)

**NAM-rs** is a cutting-edge, high-fidelity real-time [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) client designed for simulating guitar amplifiers, pedals, and studio gear with absolute precision. Powered by custom minimum-phase polyphase FIR resamplers and hand-crafted SIMD vectorized kernels (AVX2/AVX-512 delivering up to 45x speedups over scalar baselines), it guarantees **absolute real-time safety** with zero heap allocations on the audio thread. Sonically, it maintains uncompromising fidelity: rigorous cross-validation (Cross-Validation v2) using multi-component stress signals and advanced perceptual metrics—such as Error-to-Signal Ratio (ESR) and Multi-Resolution STFT—confirms that the generated audio achieves mathematical and auditory parity with the canonical C++ reference implementation under any sample rate.

Two operation modes are available: **Standalone** (native PipeWire executable, with stereo support) and **CLAP Plugin** (`.clap` library with full GUI for compatible DAWs).

---

## 🧠 Supported Models

NAM-rs natively supports Neural Amp Modeler (A1 and A2 architectures) and Impulse Response (.wav) convolution files.

> NOTE: The "A2 Architecture" is currently in Beta stage.

* **Static Mode (Ultra Performance):** *Const Generics* structures sized at compile time.
  * **WaveNet:** Standard (16×8), Lite (12×6), Feather (8×4), and Nano (4×2)
  * **WaveNet A2:** Full (8 channels) and Lite (3 channels)
  * **LSTM:** 1 and 2 Layers (Hidden Size 8 to 40: `1×8`, `1×12`, `1×16`, `1×24`, `1×40`, `2×8`, `2×12`, `2×16`, `2×24`)
  * **Linear:** FIR-based model (dot product of input history with weights + bias)
  * Non-catalogued geometries fail to load with a clear diagnostic error.

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
  `sudo apt install build-essential cmake g++ python3 pkg-config pipewire pipewire-bin pipewire-utils libpipewire-0.3-dev clang libclang-dev qpwgraph vlc ffmpeg libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev git curl linux-tools-common linux-tools-generic linux-tools-$(uname -r) bolt-22 jq ripgrep fd-find`

  > [!NOTE]
  > * `ffmpeg` is required to generate the resampler reference vectors via Python scripts.
  > * `vlc` is utilized as an audio playback generator in the manual standalone run script.

* Cargo utilities & Rustup components (required for QA & optimizations):

  * `cargo install cargo-edit`
  * `cargo install --git https://github.com/free-audio/clap-validator.git`
  * `cargo install cargo-pgo`
  * `cargo install cargo-show-asm`
  * `rustup component add llvm-tools-preview`

* **Third-Party Upstream Fixtures (C++ Parity)**:
  To run the intensive audit suite (`utils/tests-long.sh`), you must clone the upstream C++ projects (`NeuralAmpModelerCore` and `NeuralAmpModelerPlugin`) and build the reference tools. You can synchronize them easily by running:

  ```bash
  utils/mod-update.sh
  ```

* **Non-Distributable Local Models (`tests/fixtures/models-nondist/`)**:
  To test the engine locally against high-quality copyrighted models (such as `t3k` licensed files) without committing them to git, create a symbolic link named `tests/fixtures/models-nondist` pointing to your local directory containing these files. The integration test suite (`tests/nondist_validation.rs`) will automatically detect, parse, load, and perform self-consistency, block invariance, and stability checks on these models. If the directory is missing, these tests are skipped gracefully.

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

### Fresh Clone Setup

After cloning, a single command prepares all external dependencies:

```sh
./utils/mod-update.sh
```

This clones [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) (v0.5.3) and [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin) with submodules.

#### Running the Full Test Suite

```sh
# Standard tests (< 2 min):
./utils/tests-cargo.sh

# Long-duration audit (± 38 min, requires cmake + g++/clang++):
./utils/tests-long.sh
```

> [!NOTE]
> `tests-long.sh` **automatically regenerates** golden vectors if they are missing (requires C++ toolchain). Set `NAM_SKIP_GOLDEN_BUILD=1` to disable auto-regeneration.

### Build, install and run

*Note: `.cargo/config.toml` allows configuring a build optimized specifically for your current CPU ("march=native").*

> [!IMPORTANT]
> **Sudo Privilege Requirement:**
> Running `utils/build-release.sh` requires `sudo` privileges to temporarily adjust the kernel's `perf_event_paranoid` to `1` or lower. This is mandatory for CPU cycle profiling using `perf` to generate optimized LLVM BOLT layouts. If you want to bypass this interactivity, configure it persistently beforehand as detailed in [System Dependencies](docs/dependencies.md).

```bash
utils/build-release.sh
```

1. **CLAP Plugin:** The plugin will be saved as `~/.clap/nam-rs.clap` and must be automatically loaded by any DAW that supports CLAP format.

2. **Standalone Mode:** The cli will be saved as `~/.local/bin/nam-rs` and can be executed from any terminal.

---

## 📚 Documentation

* [docs/architecture.md](docs/architecture.md) — Topology, modules, and design decisions
* [docs/dependencies.md](docs/dependencies.md) — System dependencies and Rust crates
* [tests/fixtures/README.md](tests/fixtures/README.md) — Golden vector format, stress signal design, and regeneration instructions
* [docs/benchmarks.md](docs/benchmarks.md) — How to interpret Criterion performance metrics
* [docs/clap_integration.md](docs/clap_integration.md) — CLAP (Clever Audio Plug-in) integration strategy
* [docs/troubleshooting.md](docs/troubleshooting.md) — Technical troubleshooting and diagnostics support

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

---

## 🤝 Contributing

Contributions are welcome! The project is in active development.

The main and most needed now is TESTING and REAL WORLD USAGE.

---

## 🙏 Credits & Acknowledgments

This project builds upon the logic, science, and inspiration of notable works in the audio and AI communities:

* **Steven Atkinson** — For pioneering the [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler), his research on amplifier modeling with deep learning, and sharing the ecosystem.
* **Mike Oliphant** — For the exceptional [NeuralAudio (C++)](https://github.com/mikeoliphant/NeuralAudio) library, which served as a historical reference for the original A1 inference logic.

---

## ⚖️ License & Transparency (Vibe Coding)

**AI Transparency Note:** The architecture, rigorous engineering decisions, agent orchestrations, and curation of this project are intellectual work of the maintainer. However, the source code itself was generated and iterated with the assistance of Artificial Intelligence (*Vibe Coding*), specifically using models like Gemini, Claud and DeepSeek whithin Google Antigravity IDE and Kilo Code.

Despite all the rants and fights involving AI, I'm satisfied with the useful way AI had accelerated and made viable this project.

The human (me) in charge guided all the work with attention and care. I am very-very proud of the results and lesson learned of the possibilities of AI used with wisedom.

This project is licensed under the **Apache License, Version 2.0**. See the `LICENSE` file for details.
