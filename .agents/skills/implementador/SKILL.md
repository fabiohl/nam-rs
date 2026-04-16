---
name: implementador
description: Equipe de engenheiros de vários graus de senioridade especializada na implementação técnica solicitada.
---

# Skill: Implementador

## When to use this skill

Use esta skill quando for necessário focar em **codificação e execução técnica (Downstream)**. Deve ser ativada assim que uma tarefa for quebrada e planejada com clareza, com o objetivo válido, performático e bem testado. "Missão dada é missão cumprida".

## Instructions

### 1. Contexto e Padrão Arquitetural Inegociáveis

Antes de implementar qualquer funcionalidade, consulte a referência mestra em `docs/architecture.md`. Todas as restrições listadas em `.agents/rules/rust.md` devem ser estritamente observadas.

### 2. Stack de Crates Consolidada

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

### 4. Tratamento de Erros — Sistema de Diagnósticos Estruturados (OBRIGATÓRIO)

O NAM-rs possui um sistema de diagnósticos em `src/diagnostics.rs`. **Todo erro ou aviso visível ao usuário** deve usar este sistema — nunca `eprintln!("Erro: ...")` ad-hoc.

#### Regras de uso

1. **Sempre importe**: `use crate::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};`
2. **Use o builder fluente**:

   ```rust
   NamDiagnostic::new(NamErrorCode::FileNotFound, &sys)
       .message("Arquivo de modelo não encontrado: \"modelo.nam\"")
       .hint("Verifique se o caminho está correto.")
       .param("file", &path_str)
       .emit();  // ou .emit_warning() para não-fatais
   ```

3. **Mensagens logging (não-erros):** continuam com `println!("[CLI] ...")` ou `println!("[NAM-rs] ...")`. O sistema de diagnósticos é exclusivo para **erros e avisos**.
4. **Novos cenários de erro**: Se o código introduz um novo ponto de falha que o usuário pode encontrar, verifique se existe um `NamErrorCode` adequado no catálogo. Se não existir, proponha uma nova variante seguindo a convenção de faixas (E1xxx modelo, E2xxx áudio, E3xxx SPSC, E4xxx CLI, E5xxx sistema).
5. **Thread RT**: O sistema de diagnósticos **nunca** é usado dentro do callback `process()`. Falhas no RT continuam sendo reportadas via flags atômicas (`RtStatusFlags`).
6. **SystemSnapshot**: É capturado uma vez em `main()` e propagado para todas as funções que emitem diagnósticos. Nunca crie um novo snapshot fora do startup.

#### Referência do catálogo

Consulte o enum `NamErrorCode` em `src/diagnostics.rs` para a lista completa de códigos. A documentação em `docs/architecture.md` (seção "4. Módulos Fonte Atuais") descreve o módulo.

### 5. Tratativas de Código Nativo e Performance Isolada

- O loop atômico impede **ZERO** alocação dinâmica. Tensores são implementados como SoA pré-alocados via `const generics` (aplicado nas matrizes LSTM e nas TCNN/WaveNet).
- Toda passagem de parâmetros externos para o DSP ocorre exclusivamente via `rtrb` SPSC com alinhamento a 128 bytes (mitiga False Sharing entre caches L1/L2).
- Substitua `std::f32::*` por polinômios FastMath Minimax onde a precisão de 1 ULP não é necessária (hiperbólicas, exponenciais, etc.).

### 6. AVX2/AVX-512 e Clean Builds

- Use `std::simd` com `const generics` SoA para dot-products LSTM/WaveNet. AVX2 é obrigatório (baseline x86-64-v3); AVX-512 é opcional via multiversioning (`#[target_feature(enable = "avx512f")]`) quando oferecer boas oportunidades de otimização de performance.
- O código deve passar sem warnings em `utils/lints.sh` (inclui `cargo fmt`, `cargo clippy`, e validação de copyright).
