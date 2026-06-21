---
name: revisor-auditor
description: Panel of auditors, architects, bug hunters, scientists, senior engineers, and specialists in various disciplines associated with the nam-rs project (Rust, Linux Low Latency, Pipewire, CLAP, DSP, and neural networks, etc).
---

# Skill: Revisor Auditor

## When to use this skill

Use this skill for a general project review in search of improvements.

## Instructions

* Deeply analyze all code in search of improvement opportunities. Examples (non-exhaustive):
  * Bugs in architectural adherence, functionality, security, performance, low latency, etc.
  * Inspect and diagnose strict architectural adherence and correct compatibility with the [Neural Amp Modeler Core](https://github.com/sdatkinson/NeuralAmpModelerCore) reference implementation.
  * Code that can be made inline or moved out of the hot-path;
  * Code that can be better shared among other modules.
  * Files and functions with size, placement and logical organization;
  * Careful review of the "code cycle budget" looking for more optimizations for modern CPU instructions, more results, for fewer clock cycles, etc.
  * Comprehensive coverage of good source code comments;
  * Exemplary documentation (skill `documentador`).
* Trigger the `planejador-arquiteto` skill to transform the raised ideas into granular, very well-written and detailed findings in `TODO-findings.md`.
