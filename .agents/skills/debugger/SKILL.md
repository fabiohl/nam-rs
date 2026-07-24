---
name: debugger
description: Troubleshooting, diagnostic triage, and debugging skill for nam-rs. Guides AI agents through end-user support triage and developer real-time debugging.
---
<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Skill: Troubleshooting, Diagnostics & Debugging (nam-rs)

> **Source of Truth**: Official documentation in [`docs/`](../../docs/) (e.g. [`docs/architecture.md`](../../docs/architecture.md), [`docs/testing.md`](../../docs/testing.md)) and source code in [`src/`](../../src/) are the single sources of authority.

## When to use this skill

Use when analyzing end-user support requests, triaging diagnostic reports, or debugging real-time audio issues, DSP anomalies, PipeWire xruns, or CLAP plugin host integration failures.

---

## 1. End-User Support & Diagnostic Triage

### 1.1 Support Block Extraction

When an end-user posts a support request or error report, extract the structured diagnostic block (`nam-rs --diagnose`, CLAP GUI status bar `ℹ` button, or `~/.cache/nam-rs/crash-*.txt`):

```text
──── NAM-rs Diagnostic ────────────────────────────────────────────────
nam-rs v3.0.0 | E1001 | NAMB_CRC32_MISMATCH
model=NEVE1073-Standard.nam
sample_rate=48000
arch=x86_64
os=linux kernel=6.8.0-40-generic pipewire=1.2.0
features=none (baseline x86-64-v3 only)
timestamp=2026-07-23T20:40:00Z
────────────────────────────────────────────────
```

Key fields to extract:

- **Error Code & Mnemonic**: `Exxxx` code (e.g., `E1001`) and identifier (e.g., `NAMB_CRC32_MISMATCH`).
- **Runtime Parameters**: Model name/path, active sample rate, buffer size, RT priority, overloads.
- **System Environment**: Architecture (`x86_64`), OS, kernel version, PipeWire version, active CPU features.
- **Log Trace**: Check for recent `log::*` records appended to the diagnostic dump.

### 1.2 Error Code Location Table

Locate `NamErrorCode` definitions in [`src/common/diagnostics/error_codes.rs`](../../src/common/diagnostics/error_codes.rs) and follow the module investigation map:

| Error Code Range | Subsystem / Location | Key Investigation Focus |
| :--- | :--- | :--- |
| **`E1xxx`** | [`src/loader/`](../../src/loader/) | Model file existence, format (.nam JSON vs .namb binary), CRC32, weight shapes, topology support (WaveNet/LSTM/ConvNet/Linear), cab IR loading. |
| **`E2xxx`** | [`src/standalone/pw_host/`](../../src/standalone/pw_host/), [`src/dsp/`](../../src/dsp/) | PipeWire graph connection, sample rate negotiation, resampler init, SCHED_FIFO priority, buffer allocation. |
| **`E3xxx`** | [`src/common/spsc/`](../../src/common/spsc/), [`src/main.rs`](../../src/main.rs) | SPSC ring buffer capacity, producer/consumer lock-free state, GC overflow cascade. |
| **`E4xxx`** | [`src/standalone/cli.rs`](../../src/standalone/cli.rs), [`src/main.rs`](../../src/main.rs) | CLI argument parsing (`lexopt`), gain value validation, diagnose flag routing. |
| **`E5xxx`** | [`src/common/system_info.rs`](../../src/common/system_info.rs), [`src/main.rs`](../../src/main.rs) | CPU ISA baseline (`x86-64-v3`), AVX2/FMA execution, memory availability, panic hook initialization. |

### 1.3 Resolution Workflow

1. **Explain the Root Cause**: Summarize in accessible language what caused the issue.
2. **Provide Actionable Steps**: Give clear recovery instructions (e.g., re-downloading model file, checking PipeWire permissions, adjusting buffer size).
3. **Classify**: Determine if it is a user environment configuration issue (conclude guidance) or a codebase bug/deficiency (proceed to developer debugging).

---

## 2. Developer Debugging & RT Constraints

### 2.1 Evidence-Based Diagnosis

- **Never diagnose without logs**: Always inspect standard output logs (`log::*`), `DiagnosticBundle` output, or crash logs in `~/.cache/nam-rs/`.
- **Trace failure cause**: Follow exact function calls and variable states back to the point of origin in `src/`.

### 2.2 Hard Real-Time (RT) Safety Invariants

When debugging or fixing audio thread code (`src/standalone/pw_host/`, audio processing in `src/dsp/` and `src/models/`):

- **ZERO Heap Allocations / Drops**: No `Vec`, `Box`, `String`, or `Arc` drops inside `process()`. Resource swaps must use SPSC GC cascade (`gc_producer.push(...)`).
- **ZERO Blocking I/O**: No `println!`, `eprintln!`, `format!`, file access, or `std::sync::Mutex` locks on the RT path.
- **Atomic Telemetry**: RT thread signals errors or status changes exclusively via atomic bitmasks (`RtStatusFlags`). Main thread polls flags and emits `log::*` records off-RT.
- **Clean Up Debug Prints**: Remove all temporary `eprintln!`, `dbg!`, or test log assertions from RT paths before finalizing work.

### 2.3 Diagnostic Logging System

- **Unified `NamLogger` & Multi-Instance CLAP**: All off-RT logging uses `log::*`. `NamLogger` acts as the global backend (`OnceLock`), broadcasting records to terminal (CLI Standalone) and active CLAP hosts (`HostLog` weak sink list) across N plugin instances without logger collision.
- **Log Trace in Dumps & Crash Reports**: `LogBuffer` retains recent log messages in memory. Both `DiagnosticBundle::render()` and `panic_hook.rs` MUST include the `──── Recent Log Trace ────` section in support dumps and `~/.cache/nam-rs/crash-*.txt` files.
- **Panic Hook Stack Safety & File Rotation**: Crash reporting in `panic_hook.rs` uses an expanded 16 KiB buffer (`[u8; 16384]`) for zero-alloc stack safety during panics and automatically rotates crash files (retaining a maximum of 10 `crash-*.txt` files).
- **Domain Logging Mandate (Off-RT Only)**:
  - *Loaders & Parsers (`src/loader/`)*: Log file size, format (`.nam`/`.namb`), topology, weights, receptive field, and CabSim IR specs.
  - *DSP Infrastructure (`src/dsp/`)*: Log resampler init, oversampling factor changes (`Off`, `2×`, `4×`) with added latency in samples/ms, noise gate thresholds, and adaptive compute state transitions.
  - *CLAP & Host (`src/clap/`, `src/standalone/`)*: Log DAW host info, CLAP API version, render mode (`Realtime` vs `Offline`), preset paths, and PipeWire quantum renegotiations.
  - *Hot-Path Restriction*: Zero `log::*` in `process()` — RT state transitions are signaled strictly via atomic `RtStatusFlags` and polled off-RT.
