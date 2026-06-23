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

## 5. Code Requirements

- All new test files must include the Copyright and License header.
- Tests must not perform heap allocations if testing hot-path DSP code (use `CountingAllocator` when necessary).
