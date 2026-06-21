---
trigger: glob
description: Mandatory quality-assurance (Linting) guidelines for closing AI submissions.
globs: **/*.rs, **/*.toml
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Quality and Linting at Activity Completion

1. **Incremental compilation**: Run `cargo check`, `cargo clippy` and `cargo build` at appropriate moments during work.
2. **Documentation**: If there were relevant architectural changes, trigger the `documentador` skill.
3. **Tests (if `.rs` modified)**: `cargo test` — no functionality breakage.
4. **Benchmarks (if `.rs` modified with performance goals)**: `cargo bench` — verify gain or at least no regression.
5. **Exhaustive fix**: Analyze each phase and only proceed when it passes without errors, warnings, or suspicious messages. Cycle: identify source → fix → re-run.
6. **Repo hygiene**: Remove temporary files, logs, or debug artifacts not listed in `.gitignore`.
