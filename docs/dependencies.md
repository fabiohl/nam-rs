<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# NAM-rs Project Dependencies

This documentation lists and explains system and software dependencies configured in `Cargo.toml`. The primary goal is to justify these abstractions against strict architecture and performance rules (avoiding heavy or bloated libraries).

## 1. System Dependencies (Linux)

The following packages must be installed on the system to build and run NAM-rs. The consolidated command for Debian/Ubuntu systems is:

```bash
sudo apt install build-essential cmake pkg-config pipewire libpipewire-0.3-dev clang libclang-dev qpwgraph libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev git curl linux-tools-generic bolt-22
```

### Detailing by Role

* **Base Tools and Build**:

  * `build-essential` and `cmake`: Essential compilers and build utilities for the C/C++ ecosystem, from which some Rust dependencies derive.
  * `pkg-config`: Required for `cargo` to locate header paths and shared library (`.so`) paths on the system.
  * `clang` and `libclang-dev`: Requisite for `rust-bindgen` to translate PipeWire C headers to Rust bindings at compile time.
  * `libssl-dev`, `git`, and `curl`: Required for utility tools, version control, and installing Rust ecosystem components.
  * `linux-tools-generic`: Provides `perf`, essential for low-level profiling, optimizing the DSP Hot Path, and gathering profile data for LLVM BOLT.

* **Compiler-Grade Optimization (PGO + BOLT) (Optional)**:

  * `bolt-22`: LLVM BOLT post-link optimizer. The version must match the LLVM backend version of the installed Rust compiler (LLVM 22 for `rustc 1.96`).

* **Audio Backend and Tests (PipeWire)**:

  * `pipewire` and `libpipewire-0.3-dev`: Core processing headers. Only required for the `standalone` feature.
  * `qpwgraph`: Recommended utility for visual routing of the audio graph (optional but highly suggested for users).

* **Graphical Interface and Windowing**:

  * `libgtk-3-dev`, `libxcb-*`, `libxkbcommon-dev`: System libraries for native window support, X11/Wayland rendering, and keyboard management. Required by `egui` and `baseview` crates for the CLAP plugin interface.

## 2. Software Abstractions (Crates - Cargo.toml)

| Crate                            | Locked Version | Primary Role and Architectural Justification                                                                                                                                                                                                                                                  | Alternatives Considered                                                                       |
|:-------------------------------- |:-------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------- |
| **`anyhow`**                     | `^1.0`         | Lean error signaling in preparation and CLI threads, limiting the use of unwrap()/panic() which can lead to fatal failures.                                                                                                                                                                   | Rejected standard *Result* with custom Enums in favor of quick syntax.                        |
| **`libc`**                       | `^0.2`         | Binding to the C Standard Library for POSIX calls: `pthread_setschedparam` (SCHED_FIFO), `pthread_setaffinity_np` (core affinity), `mlockall`, `prctl` (THP disable), `sigaction` (SIGINT handler). Direct access without intermediate wrappers.                                              | Wrapper crates add dependency overhead with no benefit for well-documented POSIX calls.       |
| **`pipewire`**                   | `0.10.x`       | Rust bindings for `libpipewire 0.3`. Provides the native audio backend for modern Linux (low-latency). **Conditional on the `standalone` feature** (default). In builds for plugins (`--features clap-plugin`), this dependency is fully removed from the final binary, ensuring portability. | *jack*: bypassed to prioritize the modern native Linux audio ecosystem.                       |
| **`rtrb`**                       | `0.3.x`        | Lock-free SPSC ring buffer for CLI→DSP communication (parameters, models, resamplers). Used for all threads-transitioning payloads to avoid blocking the RT thread.                                                                                                                           | *crossbeam*: unnecessary overhead for pure SPSC; *ringbuf*: less ergonomic API.               |
| **`serde`** and **`serde_json`** | `^1.0`         | Deserialization of the `.nam` (JSON) format at startup. Parses nested fields (`config`, `weights`, `metadata`) robustly.                                                                                                                                                                      | Manual parser: rejected due to maintenance complexity with optional fields and weight arrays. |
| **`lexopt`**                     | `0.3.x`        | Minimalist and zero-alloc CLI parser. Extracts `--model`, `--input-gain`, `--output-gain`, and `--buffer-size` without macros or heavy dependencies.                                                                                                                                          | *clap*: significantly increases binary size for a simple CLI.                                 |
| **`log`**                        | `0.4.x`        | Standard logging facade for the Rust ecosystem. Enables `log::info!`, `log::warn!`, and `log::error!` with zero overhead when disabled.                                                                                                                                                       | Manual logging via `eprintln!`: does not offer filtering by level (`RUST_LOG`).               |
| **`env_logger`**                 | `0.11.x`       | Logging backend configurable via the `RUST_LOG` environment variable. Initialized once in `main()` with `info` as default. `default-features = false` to minimize dependencies (no regex, no native color formatting).                                                                        | *tracing*: unnecessary overhead for a CLI application without distributed instrumentation.    |
| **`half`**                       | `^2.7`         | Support for `f16` single-precision floats. Essential for weight compression (Weight Compression F16C) to target the L1 Cache.                                                                                                                                                                 | *f32*: consumes double the memory and causes L1 Cache bottlenecks in WaveNet Standard.        |
| **`minstant`**                   | `^0.1`         | High-precision telemetry based on RDTSC. Used to measure latency per block in the RT thread with negligible overhead.                                                                                                                                                                         | `std::time::Instant`: inconsistent in SCHED_FIFO and may incur unwanted syscalls.             |
| **`rustfft`**                    | `^6.4`         | High-performance FFT algorithm. Used exclusively offline (outside the RT thread) in `NamResampler::new()` for minimal phase transformation via Real Cepstrum.                                                                                                                                 | *realfft*: Wrapper over rustfft; we prefer direct rustfft with `default-features = false`.    |

## 3. Build and Automated Testing Dependencies (Dev-Dependencies)

Accessed secondarily via `cargo bench` and `cargo test`, these do not affect the final release binary footprint:

| Crate           | Locked Version | Primary Role and Architectural Justification                                                                                                                                                                                                                        |
|:--------------- |:-------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`criterion`** | `^0.8`         | Rigorous statistical metric performance evaluation. The `html_reports` flag is inactive to mitigate compile time in iterative stages where *FMA Latency per Vector* is evaluated (under a microsecond).                                                             |
| **`proptest`**  | `^1.11`        | Property-based testing for exhaustive validation of algorithmic limits in FastMath functions (`simd_tanh`, `simd_sigmoid`). Generates 10,000+ random vectors per run to sweep arithmetic holes caused by reciprocal square root Newton-Raphson (`_mm256_rsqrt_ps`). |

## 4. Additional Cargo Utilities & Rustup Components (QA, Dev & Optimization)

These components must be installed via `cargo install` or `rustup component add` to enable advanced maintenance, quality assurance, and compiler-level optimization routines:

| Utility / Component      | Installation Command                      | Role in the Project                                                                                                                                                    |
|:------------------------ |:----------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`cargo-edit`**         | `cargo install cargo-edit`                | Dependency management. Used in the `utils/mod-update.sh` script for the `cargo upgrade` command, ensuring libraries remain secure and up-to-date.                      |
| **`clippy`**             | `rustup component add clippy`             | Static linter for Rust. Used in the `utils/lints.sh` script to ensure adherence to best practices and avoid anti-patterns that could compromise performance or safety. |
| **`rustfmt`**            | `rustup component add rustfmt`            | Code formatter. Ensures visual consistency across the repository, essential for code review and maintenance by multiple contributors.                                  |
| **`clap-validator`**     | `cargo install clap-validator`            | Official command-line tool from the `free-audio` organization to validate compliance with the CLAP specification and identify potential issues or resource leaks.      |
| **`llvm-tools-preview`** | `rustup component add llvm-tools-preview` | Provides LLVM tools (like `llvm-profdata`) matching the exact rustc LLVM version, which is required for processing profile data in PGO builds.                         |
| **`cargo-pgo`**          | `cargo install cargo-pgo`                 | Cargo subcommand for Profile-Guided Optimization (PGO) and BOLT instrumentation and optimization workflow automation.                                                  |

## 5. Dependencies for Plugin and GUI (CLAP)

The following dependencies are implemented to enable CLAP plugin support and the embedded graphical interface:

| Crate              | Version  | Feature Flag  | Status                 | Justification                                                                                                                                                     |
|:------------------ |:-------- |:------------- |:---------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clack-plugin`     | `0.1`    | `clap-plugin` | Introduced in Sprint 1 | Rust API for implementing CLAP plugins. Typed abstraction over `clap-sys` with no runtime overhead. Chosen over `nih-plug` because it does not force VST3 or GUI. |
| `clack-extensions` | `0.1`    | `clap-plugin` | Introduced in Sprint 1 | CLAP spec extensions (params, state, gui, latency, track-info, remote-controls, param-indication). Separate crate from `clack-plugin` for modularity.             |
| `egui`             | `0.34.2` | `clap-plugin` | Introduced in Sprint 4 | Immediate mode GUI framework, pure Rust. GPU rendering via OpenGL (`egui_glow` and `glow`).                                                                       |
| `baseview`         | `0.1.1`  | `clap-plugin` | Introduced in Sprint 4 | Multiplatform native window for `egui` in a plugin context. Published and consumed from crates.io.                                                                |
| `rfd`              | `0.17.2` | `clap-plugin` | Introduced in Sprint 4 | Native and asynchronous File Dialog for loading models (.nam/.namb) via GUI.                                                                                      |

## 6. Dependencies for C++ Cross-Validation (Optional)

To regenerate golden vectors or perform live cross-validation against NeuralAmpModelerCore, the following packages are required:

```bash
sudo apt install cmake g++
```

* **cmake** (≥ 3.10): Build system for NeuralAmpModelerCore.
* **g++** (or `clang++`, C++20 compatible compiler): C++ compiler for the `render` tool.
* **cargo** (Rust): Test WAV generation (stress signal) and WAV→golden conversion — native binaries `gen_stress` and `wav_to_golden` replace the previous Python script block.

> [!NOTE]
> Python **is no longer required**. The Rust binaries `gen_stress` and `wav_to_golden` replace Python functions for signal generation and WAV parsing.
>
> These dependencies are **optional**. Golden vectors are pre-committed in the repository and validation tests run without C++ in normal `cargo test`. C++ is only required to:
>
> * Regenerate goldens: `./tests/fixtures/golden_gen_build.sh`
> * Perform live cross-validation: `./utils/tests-long.sh`
