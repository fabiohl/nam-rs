# AGENTS.md — AI Coding Policies (resumo)

**Todas as políticas, regras e skills estão em [`.agents/`](.agents/). Consulte o diretório para documentação completa.**

## Estrutura

| Diretório         | Função                                                                                  |
| ----------------- | --------------------------------------------------------------------------------------- |
| `.agents/rules/`  | Regras obrigatórias (copyright, RT-safety Rust, testes, linting)                        |
| `.agents/skills/` | Personas de IA especializadas (debugger, planejador, implementador, documentador, etc.) |

## Regras essenciais

- **RT-Safety (DSP)**: zero heap drop, zero bloqueio, zero `unwrap()` na thread de áudio. Desalocação via SPSC GC.
- **Copyright**: cabeçalho SPDX `Apache-2.0` obrigatório em todo arquivo fonte.
- **Testes**: unitários inline (<300 linhas) ou `_test.rs` (≥300); integração em `tests/`; benches em `benches/`; soak tests com `#[ignore]`.
- **Linting**: ao finalizar, rodar `cargo check`, `cargo test`, `cargo bench` (se perf). Corrigir todos os warnings.

## Cadeia principal de skills

`diagnostico` → `debugger` → `planejador-arquiteto` (planejamento em `TODO-sprints.md`) → `implementador`
