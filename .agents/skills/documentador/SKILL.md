---
name: documentador
description: Technical and architectural documentation specialist. Ensures project knowledge (architecture and requirements) is always synchronized with the implementation.
---
<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Skill: Documentador

## When to use this skill

* Must be activated upon receiving explicit requests to document the system.
* Use this skill at the end of each development cycle or when there is a need to keep the project's "source of truth" up to date.

## Instructions

* Documentation must be coherent with the current reality of the source code
* It is easy to understand, lean, concise, and straight to the point.
* Remember that the best documentation is well-readable source code.
* Documentation should not allow code to "go off the rails".

## Document Hierarchy

1. **`docs/architecture.md`** — Architecture bible and primary source of truth.
2. **`README.md`** — Overview, installation, and usage.
3. **`.agents/`** — AI definitions. Should be updated if there are changes in implementation patterns.

## Documentation Principles

* **Guide, not substitute**: Documentation justifies the *why* of decisions and guides pattern usage. Implementation details belong in the code.
* **Traceability**: Always point to the file or function in the source code (e.g., "See `src/diagnostics.rs`").
* **DRY (Don't Repeat Your Code)**: Never duplicate code verbatim in documentation. Explain the concept and reference the file.
* **Synchronization**: Keep the `Exxxx` error catalog synchronized between `docs/architecture.md` and the `NamErrorCode` enum.

## Best Practices

* Justify critical decisions (e.g., why `SCHED_FIFO`? why `#[repr(align(128))]`?) to prevent regressions due to lack of historical context.
* Follow the rules in `.agents/rules/`.
