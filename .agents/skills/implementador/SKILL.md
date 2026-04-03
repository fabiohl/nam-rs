---
name: implementador
description: Equipe de engenheiros de vários graus de senioridade especializada na implementação técnica solicitada, atuando predominantemente em Rust.
---

# Skill: Implementador

## When to use this skill

Use esta skill quando for necessário focar em **codificação e execução técnica (Downstream)**. Deve ser ativada assim que uma tarefa for quebrada e planejada com clareza, com o objetivo de gerar código válido, performático e bem testado. "Missão dada é missão cumprida".

## Instructions

### 1. Contexto e Arquitetura

Antes de implementar qualquer coisa, leia:

- `docs/architecture.md` — Decisões arquiteturais fundamentais do projeto.
- `.agent/rules/rust.md` — Regras inegociáveis do domínio.

### 2. Rust e Áudio

- Siga os princípios definidos: **ZERO** alocações na DSP thread, uso estrito de **Ring Buffers SPSC**, alinhamento de cache (128 bytes) para evitar false sharing, e gravação baseada em **`io_uring`**.
- **Tratamento de erros**: Na thread DSP (Tempo Real) não emita pânico sob nenhuma hipótese; o silêncio lock-free comunicador é preferível a um panic. Na thread de controle e I/O, utilize `anyhow::Result` e contextualize (`.context()`).
- **Divisão de Responsabilidades**: A separação entre o callback de áudio em tempo real isolado e a thread não em tempo real que descarrega no disco deve ser sagrada. Nenhuma fronteira entre eles pode ser transpassada sem passar estritamente pelo Ring Buffer lock-free.

### 3. Bleeding Edge e Construção

- Tire o máximo proveito dos recursos de linguagem recentes (Rust 1.94, `std::simd` se benéfico, async avançado se usado para background, etc).
- Garanta que todo novo componente se comunique nativamente sem sobrecarga.
- Garanta que `cargo build` e os devidos `lints.sh` rodarão sempre, corrigindo exhaustivamente falhas e _warnings_.
