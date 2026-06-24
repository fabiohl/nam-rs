---
name: refatora-rust
description: Refactor the structure (but not the logic and algorithms) of source code (rust)
---
<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Skill: Refatora Rust

## When to use this skill

When the user requests structural refactoring of Rust source code without changing logic or algorithms.

## Instructions

* It's PROHIBITED to make ligic or algorithmic changes.
* Ensure files are relatively small, atomic, modular, reusable, and easy to understand and maintain.
* Folder and file placement tree must be organized in an extremely logical layout for the project's purposes.
* Follow testing conventions in `.agents/rules/testing.md` (for example, move unit tests to separate `_test.rs` files if the refactored file reaches or exceeds 300 lines of code).
* Remove dead/unused code, or useless text that no longer makes sense or serves any useful purpose.
* Be extremely careful not to modify the logic itself. That is not the goal here. Regressions are strictly forbidden.
