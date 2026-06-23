---
name: refatora-doc
description: Documentation (markdown) and source code comments (rust) reviewer
---
<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Skill: Refatora Documentação

## When to use this skill

When the user requests a thorough review of documentation and source code comments.

## Instructions

* Your job here is to perform a meticulous inspection: file-by-file, line-by-line.
* Do not leave long blocks of code (50~100 lines) without any inline comments.
* All must be well-written, functional, and easy to understand for all stakeholders: junior and senior developers, architects, scientists, UX specialists, etc.
* Ensure system documentation (README.md and docs/*.md), AI agents (.agents/), comments, and test/bench suites (benches/, tests/, and src/) are updated to reflect the REAL current state of the code.
* Remove irrelevant references that do not contribute strictly to understanding the code, such as "sprint X", "review done on DDMMYYYY", "requested by PO", etc.
* Ensure internal links in Markdown documents work perfectly (using the relative references to the git repository).
* In Rust files (`.rs`), ensure correct usage of `///` to document public items and `//` for internal structural comments.
* Orient yourself by the highest standards of the `documentador` skill.
