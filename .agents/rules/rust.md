---
trigger: glob
description: Diretrizes Rust para Áudio RT, Inferência Neural e Plugins.
globs: **/*.rs, **/*.toml
---

# Diretrizes Técnicas (NAM-rs)

* **Matemática & Perf:** Alvo x86-64-v3. Use `core::arch::x86_64::*` (AVX2/AVX-512 intrinsics) e FastMath Minimax via polinômios Padé/Minimax. Priorize algoritmos que favoreçam otimizações do compilador e evitem branch prediction failures.
* **Feature Flags & Portabilidade:**
  * `standalone` (default): Backend PipeWire nativo.
  * `clap-plugin`: Build agnóstico para integração CLAP (sem PipeWire).
  * Desacoplamento via trait `AudioHost` e parâmetros `NamPluginParams`.
* **Segurança RT (Thread DSP):**
  * **ZERO alocações (Heap), ZERO I/O e ZERO Locks.**
  * `SCHED_FIFO` (prio ~90), Core Affinity e `#[repr(align(128))]` para SPSC/Lock-free.
  * Tensores via *SoA* e *const generics* para performance determinística.
* **Arquitetura A2 (Staging):**
  * Utilize enums `ActivationType`, `GatingMode` e módulos stub (`film.rs`).
  * Modelos A2 (v0.6+) usam `WavenetA2Placeholder` para garantir paridade e fallback.
* **Higiene & Erros:**
  * Mínimo de dependências (`cargo add`). Unwraps restritos (use fallbacks no RT).
  * Erros via `NamDiagnostic`. Parsing robusto contra dados malformados (fuzzing).
* **Porte C++ & Legado:** Referências em `github.com/`. Mantenha layouts de memória e paridade matemática rigorosa vs C++.
* **Documentação:** Comentários `///` justificando decisões críticas de DSP/SIMD e SPSC.
