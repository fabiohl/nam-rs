---
name: refatora-rust
description: Refactor the structure (but not the logic and algorithms) of source code (rust)
---

# Skill: Refatora Rust

## When to use this skill

When the user requests structural refactoring of Rust source code without changing logic or algorithms.

## Instructions

* Ensure files are relatively small, atomic, modular, reusable, and easy to understand and maintain.
* Folder and file tree must be organized in an extremely logical layout for the project's purposes.
* Strictly follow **Real-Time Safety** and performance rules in `.agents/rules/rust.md` (avoid hidden heap allocations, locks, or blocking calls in the DSP hot paths).
* Follow testing conventions in `.agents/rules/testing.md` (for example, move unit tests to separate `_test.rs` files if the refactored file reaches or exceeds 300 lines of code).
* Remove dead/unused code, or useless text that no longer makes sense or serves any useful purpose.
* Be extremely careful not to modify the logic itself. That is not the goal here. Regressions are strictly forbidden.
* Finally, validate with `utils/lints.sh` and `utils/tests-quick.sh` — the work is only complete when they no longer display any warnings or errors.
* Trigger the `planejador-arquiteto` skill to transform the raised ideas into granular, very well-written and detailed sprints and technical tasks.
