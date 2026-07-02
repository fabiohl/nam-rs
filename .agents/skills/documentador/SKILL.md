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
2. **`docs/*.md`** — Various important technical references.
3. **`README.md`** — Overview, installation, and usage.
4. **`.agents/`** — AI definitions. Should be updated if there are changes in implementation patterns.

## Documentation Principles

* **Guide, not substitute**: Documentation justifies the *why* of decisions and guides pattern usage. Implementation details belong in the code.
* **Traceability**: Always point to the file or function in the source code (e.g., "See `src/diagnostics.rs`").
* **DRY (Don't Repeat Your Code)**: Never duplicate code verbatim in documentation. Explain the concept and reference the file.
* **Synchronization**: Keep the `Exxxx` error catalog synchronized between `docs/architecture.md` and the `NamErrorCode` enum.
* **Technical Documentation Standardization:** Establish a uniform voice and structure across all code, comments, and reference materials to ensure flawless readability and seamless developer onboarding.
  * Adopt a "docs-as-code" approach.
  * Use an unified "voice and tone" style and "mood".
  * Avoid style deviations, confusing terminology, or inconsistent code comments.
  * Ensures that all technical assets read like a cohesive, single-authored masterpiece.

## Best Practices

* Justify critical decisions (e.g., why `SCHED_FIFO`? why `#[repr(align(128))]`?) to prevent regressions due to lack of historical context.
* Never make irrelevant statements that do not contribute strictly to understanding the code, such as "sprint X", "review done on DDMMYYYY", "requested by PO", etc.
* Follow the rules in `.agents/rules/`.
* Toda alteração em arquivos de catálogo de goldens, thresholds de validação, scripts de testes, ou dependências **deve** vir acompanhada da atualização correspondente nos documentos de referência **no mesmo commit**:
| Se alterar…                                        | Atualizar…                                                        |
|----------------------------------------------------|-------------------------------------------------------------------|
| `tests/fixtures/golden_gen_build.sh` (CATALOG)     | `tests/fixtures/README.md` (tabelas de modelo/golden)             |
| `tests/common/validation.rs` (thresholds)          | `tests/fixtures/README.md` (§Parity Thresholds)                   |
| `utils/tests-quick.sh` ou `utils/tests-long.sh`    | `docs/testing.md` (§Fases, §Test Coverage Matrix)                 |
| `.golden_manifest.sha256` (regeneração)            | Commit do manifesto junto com a alteração de modelo               |
| `Cargo.toml` (features)                            | `docs/testing.md` (§Feature Taxonomy, §Test Coverage Matrix)      |
Regra de enforcement: o revisor e a IA devem verificar sincronização doc↔código como
critério de aprovação de qualquer PR que toque nos arquivos acima.
