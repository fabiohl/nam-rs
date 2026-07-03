<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Plano Sprints para BUG-3

Este plano organiza as tarefas necessárias para diagnosticar, corrigir e validar o BUG-3 de maneira ágil, segura e altamente controlada.

---

## Sprint 1: Diagnóstico Seguro e Instrumentação (Foco em Segurança)

* **Objetivo:** Isolar o hang em ambiente controlado para coletar logs sem derrubar a sessão gráfica do usuário.

### Tarefas S1

* [ ] **T1.1 — Configuração de Sandbox e Restrições de Recursos**
  * Configurar um container Docker local ou sandbox cgroups limitando memória para 1 GB e CPU a 1 core.
  * Criar script wrapper usando `timeout -s KILL 10` para impedir travamentos prolongados na reprodução.

* [x] **T1.2 — Execução Diagnóstica com Instrumentação** ✅ 2026-07-03
  * Executar o teste alvo em modo release compilado com AddressSanitizer (ASan) ativo (`RUSTFLAGS="-Zsanitizer=address"`).
  * Capturar chamadas do sistema com `strace` e profiling com `perf` a partir de um terminal externo independente.
  * **Resultados:** Hang confirmado (CPU spin puro). ASan não detectou violações. Perf inconclusivo (binário stripped). Detalhes em `TODO-findings.md` §4.

* [ ] **T1.3 — Mapeamento de Pontos de Hang e Execução de Oráculo**
  * Inserir medições seguras off-RT temporárias para verificar se o travamento ocorre durante a desalocação do buffer ou durante a iteração de processamento do sinal.

---

## Sprint 2: Correção da Causa Raiz e Refatoração (Foco em Integridade)

* **Objetivo:** Eliminar o comportamento indefinido (UB) em `AlignedVec` e garantir o alinhamento seguro de buffers de áudio.

### Tarefas S2

* [ ] **T2.1 — Refatoração de Gerenciamento de Memória em `AlignedVec`**
  * Alterar o struct `AlignedVec` em `aligned.rs` para incluir o campo `capacity: usize` ou garantir que seu descarte (`Drop::drop`) use exatamente o layout original de alocação de capacidade.
  * Assegurar conformidade estrita com as regras de RT-Safety (sem alocações adicionais na thread de processamento).

* [ ] **T2.2 — Blindagem de Indexações de Ring Buffer (`X2Stage`)**
  * Substituir o acesso direto `get_unchecked` por indexações seguras verificadas provisoriamente.
  * Tratar possíveis indexações inconsistentes caso o tamanho do buffer de entrada divirja nos estágios de decimação e interpolação.

* [ ] **T2.3 — Correção de Potenciais Underflows Aritméticos**
  * Revisar a operação de modulo e wrapping em `abs_idx.wrapping_sub(tap_delay) % n` para atestar compatibilidade multiplataforma (32/64 bits) e segurança matemática.

---

## Sprint 3: Validação Completa e Testes de Regressão

* **Objetivo:** Trazer o teste de rejeição de aliasing de volta ao pipeline de testes contínuos sem regressions.

### Tarefas S3

* [ ] **T3.1 — Reativação do Teste na Suíte**
  * Remover a diretiva `#[ignore]` de `test_x2_aliasing_rejection` em `oversample_test.rs` de forma definitiva caso a estabilidade seja comprovada.

* [ ] **T3.2 — Execução Completa da Suíte de Testes Rápida**
  * Rodar `./utils/tests-quick.sh` para atestar que os outros testes do módulo DSP continuam passando perfeitamente.

* [ ] **T3.3 — Benchmark de Impacto de Performance**
  * Avaliar regressões de tempo de processamento com benchmarks Criterion off-line.
  * Verificar se a refatoração do heap allocator ou a substituição de indexações não impacta o orçamento de tempo em threads de áudio PipeWire.
