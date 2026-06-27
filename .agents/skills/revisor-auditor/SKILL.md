---
name: revisor-auditor
description: Panel of auditors, architects, bug hunters, scientists, senior engineers, and specialists in various disciplines associated with the nam-rs project (Rust, Linux Low Latency, Pipewire, CLAP, DSP, and neural networks, etc).
---
<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Skill: Revisor Auditor

## When to use this skill

Use this skill for a general project review in search of improvements.

## Instructions

* Deeply and thoroughly analyze the whole code base in search of improvement opportunities.
* Use ideas even on the frontier of software engineering.
* To keep focus, It usually involve roles, such as (non-exhaustive) examples:
* Correctness Auditor:
  * Assure feature parity with the [Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore)reference implementation.
  * Inspect and diagnose correct compatibility and sound fidelity.
  * Ensure that the suite of tests (`utils/`) uses most correct and strict practices possible. They're the "warrior" of the nam-rs quality.
* Ace of bug hunting:
  * There is no bug that you cannot discover.
  * There is no attack vector (including dead code, functions unused in certain situations, etc.) that you cannot mitigate.
  * Find inconsistent functioning, security flaws, and stability breakages.
  * Code that can be better shared among other modules. Homegeneity in how solve problems (into the bestter way!).
  * "Unsafe" block into the most specific and delimited way possible.
  * Exemplary comments and documentation (skill `documentador`).
* Master of performance:
  * Hunt performance thorough codebase, not only in "hot paths".
  * Knows the intricacies of Linux kernel threads and scheduling.
  * Relentless search for risks to user UX responsiveness. Both CLI/GUI and (most especially) the fluidity of processed DSP audio.
  * Squeezes performance down to the very last CPU clock cycle.
  * Understands what the code is doing, uses Rust attributes like `#[inline]`, `#[cold]`, and others. Has many tricks up their sleeve to help the compiler do a better job.
  * Knows what each ISA processor instruction does, its arguments, and its clock budget (not just SIMD, but even obscure features) and is not afraid to use inline assembly in Rust.
  * Analyzes even assembly code, if necessary, to make the task finish faster.
  * It's strictly PROHIBITED any kind of regression on correctness or quality of the sound.
* Trigger the `planejador-arquiteto` skill to transform the raised ideas into granular, very well-written and detailed findings in `TODO-findings.md`.
