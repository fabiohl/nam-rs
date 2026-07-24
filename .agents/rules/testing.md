---
trigger: always_on
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Testing Conventions — AI Operational Rules

Non-negotiable rules only. For rationale, phase-by-phase breakdown, oracles, tiers,
and gate types, consult [docs/testing.md](../../docs/testing.md) and
[docs/perceptual_validation.md](../../docs/perceptual_validation.md) — do not
duplicate that content here.

## 1. Test Placement

- Unit tests: inline `#[cfg(test)] mod tests` if the file has **< 300 source lines**
  (excluding test code); otherwise move to a sibling `<module>_test.rs` included via
  `#[cfg(test)] #[path = "..."] mod ...;`.
- Integration tests: root `tests/` directory only.
- Benchmarks (`criterion`): root `benches/` directory only.
- Slow/heavy tests (soak, full proptest/fuzz counts, full C++ parity, cross-ISA,
  RT-safety, heap-audit) MUST be `#[ignore]`d — they run exclusively in
  `utils/tests-long.sh`, never in the default/quick loop.
- Measurement/off-RT test helpers belong in `src/testing/` — never in `src/dsp/` or
  any hot-path module.

## 2. Verification Scripts — Run Order & AI Restrictions

Each script strictly extends the previous one's scope — never repeat a check:

1. `utils/lints.sh` — static analysis only (fmt, SPDX, check, clippy).
2. `utils/tests-quick.sh` — agile first line. **Allowed to run once per AI task**, as
   a final validation.
3. `utils/quality-dashboard.sh --check docs/quality-contract.txt` — baseline-gated bench and audio quality checks.
4. `utils/tests-long.sh` — nightly/pre-release audit (± 50 min, unattended).
   **NEVER run it in an AI task, under any circumstance.** If validation of its
   scope is needed, ask the human operator to run it and report results.

## 3. Hard Requirements When Adding/Modifying Tests or Metrics

- Every metric threshold must ship with a measurement comment
  (`// Measured: SNR=..., ESR=...`). Silent fallback to default thresholds is
  prohibited.
- When C++ NAMCore parity and the f64 Oracle disagree, the **f64 Oracle wins**.
- Error codes must stay synchronized between
  `src/common/diagnostics/error_codes.rs` and `docs/architecture.md` §9.
- Never add a test to `utils/tests-long.sh` without first running it standalone to
  confirm it terminates and passes/skips cleanly — a hang there is worse than a
  failure (silently costs the full nightly window).

## 4. Logging & Diagnostics Verification in Tests

- **Log Buffer & Diagnostic Assertions:** When adding or updating off-RT loaders, builders, or diagnostic reporters (`bundle.rs`, `logger.rs`), include unit tests verifying that diagnostic log messages are properly captured in `LogBuffer` and included in `DiagnosticBundle::render()`.
- **Panic Hook & Heap Audit Parity:** Any change to `panic_hook.rs` or `DiagnosticBundle` formatting must validate the `#[cfg(feature = "heap-audit")]` test path (`format_panic_report_for_audit_test`) to ensure stack safety (`[u8; 16384]` limit) and zero unwanted heap allocations during crash report rendering.
- **Log Silence in Hot-Path Benchmarks:** Benchmarks (`benches/`) and real-time audio loop tests must explicitly verify that no `log::*` calls are executed within the audio callback or DSP hot-paths.
