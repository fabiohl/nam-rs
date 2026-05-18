---
name: documentador
description: Especialista em documentação técnica e arquitetural. Garante que o conhecimento do projeto (arquitetura e requisitos) esteja sempre sincronizado com a implementação.
---

# Skill: Documentador

## When to use this skill

* Deve ser ativada ao receber solicitações expressas para documentar o sistema.
* Use esta skill ao final de cada ciclo de desenvolvimento ou quando houver necessidade de manter a "fonte da verdade" do projeto atualizada.

## Instructions

* A documentação deve ser coerente com a realidade atual do código-fonte
* É fácil de entender, enxuta, concisa e direta ao assunto.
* Lembre-se que a melhor documentação é um código-fonte bem legível.
* Já a documentação não permite o código "viajar na maionese".

## Hierarquia de Documentos

1. **`docs/architecture.md`** — Bíblia de arquitetura e fonte primária de verdade.
2. **`README.md`** — Visão geral, instalação e uso.
3. **`.agents/`** — Definições de IA. Devem ser atualizadas se houver mudança nos padrões de implementação.

## Princípios de Documentação

* **Guia, não substituto**: A documentação justifica o *porquê* das decisões e orienta o uso de patterns. Detalhes de implementação pertencem ao código.
* **Rastreabilidade**: Sempre aponte para o arquivo ou função no código-fonte (ex: "Veja `src/diagnostics.rs`").
* **DRY (Don't Repeat Your Code)**: Nunca duplique código verbatim na documentação. Explique o conceito e referencie o arquivo.
* **Sincronia**: Mantenha o catálogo de erros `Exxxx` sincronizado entre `docs/architecture.md` e o enum `NamErrorCode`.

## Boas Práticas

* Justifique decisões críticas (ex: por que `SCHED_FIFO`? por que `#[repr(align(128))]`?) para evitar regressões por desconhecimento do histórico.
* Guie-se pelas rules em `.agents/rules/`.
