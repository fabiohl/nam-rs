---
name: documentador
description: Especialista em documentação técnica e arquitetural. Garante que o conhecimento do projeto (arquitetura e requisitos) esteja sempre sincronizado com a implementação.
---

# Skill: Documentador

## When to use this skill

Use esta skill ao final de cada ciclo de desenvolvimento ou quando houver necessidade de manter a "fonte da verdade" do projeto atualizada. Deve ser ativada ao receber solicitações expressas para documentar o sistema, quando houver mudanças arquiteturais relevantes de DSP/Áudio, e para evitar o apodrecimento da documentação técnica.

## Instructions

### 1. Análise de Impacto

Antes de consolidar a documentação final ou propor mudanças, identifique quais documentos em `docs/` e regras em `.agent/rules/` são afetados pela nova funcionalidade/estrutura.

### 2. Sincronização de Arquitetura

- Toda decisão envolvendo novos pacotes `C` (`bindgen`), algoritmos em módulo `io_uring` e alterações de DSP **DEVE** estar rigorosamente atualizada no arquivo `docs/architecture.md`.
- Assegure que as skills em .agents/skills/ para que elas estejam preparadas e coerentes com a documentação.

### 3. Boas Práticas

- Preserve e melhore o que já existe (nunca apague conteúdo útil sem justificativa).
- Seja conciso e preciso — documentação em sistemas críticos de kernel/tempo-real engessa garantias que devem ser provadas matematicamente ou lógicas por trás de mitigações de memory cache.
