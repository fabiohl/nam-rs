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
3. `utils/tests-performance-regression.sh` — baseline-gated bench check.
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
