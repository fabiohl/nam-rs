---
name: implementador
description: Equipe de engenheiros de vários graus de senioridade especializada na implementação técnica solicitada, atuando predominantemente em Rust.
---

# Skill: Implementador

## When to use this skill

Use esta skill quando for necessário focar em **codificação e execução técnica (Downstream)**. Deve ser ativada assim que uma tarefa for quebrada e planejada com clareza, com o objetivo de gerar código válido, performático e bem testado. "Missão dada é missão cumprida".

## Instructions

### 1. Contexto e Padrão Arquitetural Inegociáveis

Antes de implementar qualquer funcionalidade, consulte a referência mestra em `docs/architecture.md`, bem como o guia `docs/NAM-rs-referência.md` e obedeça os ciclos do desenvolvimento em `docs/NAM-rs-sprints.md`. Todas as restrições listadas em `.agents/rules/rust.md` devem ser estritamente observadas.

### 2. Stack de Crates Consolidada (Sprint 8 — não substituir sem discussão)

| Crate                  | Versão | Papel                                              |
| ---------------------- | ------ | -------------------------------------------------- |
| `pipewire`             | 0.9.x  | Host áudio PipeWire (standalone)                   |
| `rtrb`                 | 0.3.x  | SPSC Ring Buffers lock-free (param, GC, resampler) |
| `rubato`               | 0.16.x | Resampling SincFixedIn (Kaiser Window, RT-safe)    |
| `lexopt`               | 0.3.x  | Parser CLI leve e sem alocação dinâmica            |
| `serde` / `serde_json` | 1.x    | Desserialização de modelos `.nam` (JSON)           |
| `crc32fast`            | 1.x    | Verificação CRC32 de modelos `.namb`               |
| `libc`                 | 0.2.x  | `pthread_setaffinity_np`, `SCHED_FIFO`             |
| `anyhow`               | 1.x    | Propagação de erros fora do caminho RT             |

> **Nunca inserir crates diretamente no `Cargo.toml`** — use `cargo add` com as features específicas necessárias.

### 3. Padrões Consolidados — Não Reinventar

- **SPSC de parâmetros**: `rtrb::Producer<ParamPayload>` (enum `#[repr(align(128))]`) CLI→DSP.
- **SPSC de resampler**: `rtrb::Producer<NamResampler>` Main→callback (construção fora do RT, zero-alloc no `process()`).
- **GC por Drop-Delegation**: modelos obsoletos enviados via `rtrb::Producer<Box<DynamicModel>>` para thread GC fazer `drop()` fora do RT.
- **Comunicação RT→Main**: somente via `RtStatusFlags` (campos `AtomicU32`/`AtomicBool` em `Arc`). **Nunca** `println!` ou `eprintln!` dentro do `process()`.

### 4. Tratativas de Código Nativo e Performance Isolada

- O loop atômico impede **ZERO** alocação dinâmica. Tensores são implementados como SoA pré-alocados via `const generics` (aplicado nas matrizes LSTM e nas TCNN/WaveNet).
- Toda passagem de parâmetros externos para o DSP ocorre exclusivamente via `rtrb` SPSC com alinhamento a 128 bytes (mitiga False Sharing entre caches L1/L2).
- Substitua `std::f32::*` por polinômios FastMath Minimax onde a precisão de 1 ULP não é necessária (hiperbólicas, exponenciais, etc.).

### 5. AVX2/AVX-512 e Clean Builds

- Use `std::simd` com `const generics` SoA para dot-products LSTM/WaveNet. AVX2 é obrigatório (baseline x86-64-v3); AVX-512 é opcional via multiversioning (`#[target_feature(enable = "avx512f")]`) quando oferecer boas oportunidades de otimização de performance.
- O código deve passar sem warnings em `utils/lints.sh` (inclui `cargo fmt`, `cargo clippy`, e validação de copyright).

### 6. Projetos de Referência

Muito do trabalho envolverá analisar a implementação em C++ dos projetos abaixo e portar para Rust, fazendo as devidas adaptações.

| Repositório GitHub                                     | Pasta local                                                     |
| ------------------------------------------------------ | --------------------------------------------------------------- |
| <https://github.com/mikeoliphant/NeuralAudio>          | `github.com/mikeoliphant/NeuralAudio`                           |
| <https://github.com/p-ranav/argparse>                  | `github.com/mikeoliphant/NeuralAudio/Utils/deps/argparse`       |
| <https://github.com/Chowdhury-DSP/math_approx>         | `github.com/mikeoliphant/NeuralAudio/deps/math_approx`          |
| <https://github.com/mikeoliphant/NeuralAmpModelerCore> | `github.com/mikeoliphant/NeuralAudio/deps/NeuralAmpModelerCore` |
| <https://github.com/mikeoliphant/RTNeural>             | `github.com/mikeoliphant/NeuralAudio/deps/RTNeural`             |
