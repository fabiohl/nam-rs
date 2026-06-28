---
trigger: always_on
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Testing Conventions

To maintain project organization and performance, we follow these test organization rules:

## 1. Unit Tests

Unit tests should test the internal logic of each module.

- **Small Files (< 300 lines of source code, excluding `#[cfg(test)]` blocks):** Should keep tests
  **inline** at the end of the file, inside a block:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      // ... tests ...
  }
  ```

- **Large Files (>= 300 lines of source code, excluding `#[cfg(test)]` blocks):** Should move tests
  to a separate file with the suffix `_test.rs` in the same directory. The main file should include
  the test at the end:

  ```rust
  #[cfg(test)]
  #[path = "module_name_test.rs"]
  mod module_name_test;
  ```

> **Note:** The 300-line threshold measures **source code only** (lines before `#[cfg(test)]`),
> not the total file length including test blocks. When a file has < 300 source lines but many
> inline tests, inline tests remain valid — the threshold defines where tests are placed, not a
> hard cap on total file size.

## 2. Integration Tests

Tests that exercise the crate's public API or multiple integrated modules should be placed in the root `tests/` directory.

- **Standard/Fast Integration Tests:** Should be run as part of the daily developer verification flow via `utils/tests-quick.sh` (which runs all unit, integration, CLAP/heap-audit tests, `clap-validator`, and medium validation — C++ parity + proptests — in under 2.5 minutes).
- **Long-Duration Stress/Soak Tests:** Slow or heavy tests (such as soak/endurance checks, property-based parsing/math sweeps, C++ parity verifications) that aren't covered by the unified suite should be marked with `#[ignore]` and run exclusively via the decoupled long-duration audit suite `utils/tests-long.sh` (± 30 minutes).

## 3. Benchmarks

Performance benchmarks using the `criterion` framework should be placed in the root `benches/` directory.

- **Fast Benchmarks:** Run as part of standard iterations.
- **Long-Running / Throughput Benchmarks:** Benchmarks requiring long measurement times (e.g. 30s+ with the `long_bench` feature enabled) should be deferred to `utils/tests-long.sh` to keep normal workflows fast.

## 4. Test Verification Scripts

Developers should use the following scripts under `utils/` before pushing commits:

- `utils/lints.sh`: cargo fmt, check and clippy. Very useful for quick corrections.
- `utils/tests-quick.sh`: Runs the unified QA suite — unit/integration tests, medium validation (C++ parity + proptests), builds the debug CLAP library with heap audits, and performs strict validation. It's allowed to run once in IA tasks - as a final validation.
- `utils/tests-long.sh`: Runs the decoupled 5-phase long-duration audit (Soak, Proptests, Parity/Heap-Audits, Release CLAP Validation, and Long Benchmarks), logging execution output to `target/logs/`. NEVER run it as a pass in IA tasks.

## 6. Three Independent Test Oracles

The QA strategy rests on three independent, non-redundant oracles — each answers a fundamentally different question, and removing any one leaves a corresponding blind spot:

| Oracle                   | Question it answers                                            | Validation layer                | Placement       |
|:------------------------ |:-------------------------------------------------------------- |:------------------------------- |:--------------- |
| **C++ NAMCore** (f32)    | "Does NAM-rs match the canonical C++ reference?"               | `tests/cpp_parity.rs`           | `tests-long.sh` |
| **f64 Reference Oracle** | "Is NAM-rs mathematically correct (absolute numerical truth)?" | `tests/reference_oracle_f64.rs` | `tests-long.sh` |
| **ISA Parity**           | "Do AVX2, AVX-512, and scalar produce identical results?"      | `tests/isa_parity.rs`           | `tests-long.sh` |

- **C++ NAMCore (external truth):** Anchors NAM-rs against the canonical C++ reference via golden vectors with ES-R, SNR, MR-STFT perceptual gates. Divergence is expected and bounded by FastMath approximations — this is a **loose-band** gate.
- **f64 Oracle (absolute truth):** Double-precision forward pass for WaveNet/LSTM/A2. Decomposes total error by source (activation precision, weight compression, accumulation). This is a **tight-band** gate — it answers "how close is NAM-rs to ideal math?" independently of any external reference.
- **ISA Parity (consistency):** Verifies that AVX2, AVX-512, and scalar paths produce bit-identical results per model. Catches SIMD kernel regressions invisible to the other two oracles.

> **Why all three?** A kernel bug small enough to pass the C++ golden band can still break the f64 oracle. A spec error shared by scalar and SIMD is invisible to ISA parity but caught by the external golden. No oracle is redundant.

## 7. Test Value Hierarchy (Tiers)

Not all tests provide equal confidence. The 3-tier framework prioritizes **what to test** and **where to place it**:

| Tier | Category                                          | Examples                                                           | Guarantee              | CI Placement     |
|:---- |:------------------------------------------------- |:------------------------------------------------------------------ |:---------------------- |:---------------- |
| 1 🔴 | **Absolute correctness** (vs. mathematical truth) | f64 Oracle ES-R, golden vector SNR/ESR vs. C++ NAMCore             | Spec-level correctness | `tests-long.sh`  |
| 2 🟠 | **Relative consistency** (approx-vs-approx)       | Activation precision sweep (Standard vs. HighFidelity), ISA parity | No silent regressions  | `tests-long.sh`  |
| 3 🟡 | **Infrastructure integrity**                      | Unit tests, proptests, heap-audits, soak                           | Crash-free, RT-safe    | `tests-quick.sh` |

- **Tiers 1 & 2** exercise tests where there is an external reference. They run in the long suite (`tests-long.sh`).
- **Tier 3** covers hermetic checks (no external toolchain). Runs in the fast suite (`tests-quick.sh`, < 3 min).
- New SIMD kernels must pass Tiers 1 → 2 → 3 in order before merging.

## 8. Gate Types: Hard vs. Soft

Validation gates are classified by their enforcement strength:

- **Hard Gate:** Test **fails** if threshold is violated. Used when a metric directly measures correctness (e.g., SNR < threshold, ES-R > threshold, ASR > −70 dB). Block CI merge.
- **Soft Gate:** Test **warns but passes** if threshold is violated. Used for informational/diagnostic metrics that characterize quality without proving correctness (e.g., MR-STFT spectral distance on known-divergent models).

**MR-STFT gate specifics:** MR-STFT acts as a **hard gate** at 44.1/48 kHz (threshold `mrstft_max` calibrated per model) and a **soft gate** at 88.2/96/192 kHz. This reflects the resampler-inherent spectral error increase at higher sample rates.

## 9. Measurement Framework Conventions

The measurement library (`src/testing/`) is strictly **off-RT** — all functions allocate on the heap. Follow these conventions when adding or modifying measurement code:

- **Function placement:** Measurement functions belong in `src/testing/`, never in `src/dsp/` or hot-path code.
- **Gate calibration:** Every metric threshold must be explicitly documented with a measurement comment (`// Measured: SNR=..., ESR=...`). Silent fallback to `topology_thresholds()` is prohibited.
- **Baseline versioning:** Metric baselines (ASR, THD+N, FR) are versioned in `tests/fixtures/spectral_fidelity_baseline.json` and regenerated via `--accept` flag.
- **f64 oracle is ground truth:** When a metric disagrees between C++ NAMCore and the f64 oracle, the f64 oracle wins — it represents absolute mathematical truth, not an external implementation.
- **True-peak is off-RT only:** BS.1770-4 Annex 2 true-peak (4× polyphase FIR, 48 taps) is too expensive for the audio thread. Use sample-peak detection on RT path; true-peak only in integration tests.
- **ERROR CODES:** All error codes follow the `Exxxx` catalog and must be synchronized between `src/common/diagnostics/error_codes.rs` and `docs/architecture.md` §9.
