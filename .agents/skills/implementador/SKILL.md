---
name: implementador
description: Equipe de engenheiros de vários graus de senioridade especializada na implementação técnica solicitada.
---

# Skill: Implementador

## When to use this skill

Use esta skill quando for necessário focar em **codificação e execução técnica (Downstream)**. Deve ser ativada assim que uma tarefa for quebrada e planejada com clareza, com o objetivo válido, performático e bem testado. "Missão dada é missão cumprida".

## Instructions

Vide `.agents/rules/rust.md` para diretrizes técnicas mandatórias (RT-Safety, SIMD, SPSC).

### 1. Contexto e Padrão Arquitetural Inegociáveis

Antes de implementar qualquer funcionalidade, consulte a referência mestra em `docs/architecture.md`. Todas as restrições listadas em `.agents/rules/rust.md` devem ser estritamente observadas.

### 2. Padrões Consolidados — Não Reinventar

- **SPSC de parâmetros**: `rtrb::Producer<ParamPayload>` (enum `#[repr(align(128))]`) CLI→DSP.
- **SPSC de resampler**: `rtrb::Producer<NamResampler>` Main→callback (construção fora do RT, zero-alloc no `process()`).
- **GC por Drop-Delegation**: modelos obsoletos enviados via `rtrb::Producer<Box<DynamicModel>>` para thread GC fazer `drop()` fora do RT.
- **Comunicação RT→Main**: somente via `RtStatusFlags` (campos `AtomicU32`/`AtomicBool` em `Arc`). **Nunca** `println!` ou `eprintln!` dentro do `process()`.

### 3. Tratamento de Erros — Sistema de Diagnósticos Estruturados (OBRIGATÓRIO)

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

### 4. Bons comentários de código-fonte

Conforme for implementando a codificação, SEMPRE vá inserindo bom número de comentários de código-fonte. Um bom comentário ajuda o dev júnior a entender o que está acontecendo e o arquiteto sênior a auditar a implementação.

### 5. Qualidade Final

O código deve passar sem warnings pelas skills `.agents/rules/copyright.md` e `.agents/rules/linting.md`.
