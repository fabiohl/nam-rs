<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Sprint Backlog — EP1 — Quick suite enxuto e honesto

Este documento detalha o planejamento ágil para execução do épico **EP1**, focado em tornar a suíte de testes rápida (`quick`) mais enxuta, veloz e correta em termos de reporte de status.

---

## Sprint 1: Otimização da Suíte de Testes Rápida (Quick)

### Tarefas Técnicas

#### [ ] Tarefa 1.1: Remover `rt_constraints` do Quick

* **Achado Associado:** [F-S1](file:///home/fabio/nam-rs/TODO-findings.md#L234) — `rt_constraints` compila e executa 0 testes no quick.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O binário de teste `rt_constraints` contém testes que dependem de features como `heap-audit` ou que são skipados intencionalmente no quick. Compilar esse binário em debug a cada run consome de 10 a 15 segundos desnecessariamente. Devemos remover `rt_constraints` da variável `_struct_targets` na suíte rápida. A cobertura real desse binário pertence exclusivamente à suíte longa (`tests-long.sh`).
* **Arquivos Afetados:**
  * [utils/tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh#L326) — remover a adição de `rt_constraints` em `_struct_targets`.
* **Critério de Aceitação:**
  * Executar [utils/tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) não deve mais compilar nem executar o alvo de teste `rt_constraints`.
  * Ganho imediato de tempo na execução da fase de testes estruturais em ambiente local.

---

#### [ ] Tarefa 1.2: Deduplicar Parity no Quick e Limpar Caps Mortos

* **Achado Associado:** [F-T2](file:///home/fabio/nam-rs/TODO-findings.md#L160) — `run_v1` é alias 1:1 de `run_v1_hf` (testes duplicados).
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Desde que a ativação `Standard` (exact-grade) virou o padrão universal do projeto, a função de comparação `run_v1` delega diretamente para `run_v1_hf`, resultando em execuções redundantes que medem exatamente o mesmo comportamento.
  * Consolidar os helpers: remover o sufixo `_hf` unificando `run_v1_hf` e `run_v1`, e unificando `run_v2_multi_sr_hf` e `run_v2_multi_sr` para usarem ativação `Standard` por padrão.
  * Remover os testes redundantes `quick_parity_hf_lstm_1x16` e `quick_parity_hf_wavenet_ch16` da suíte de testes rápidos.
  * Limpar constantes de cap de Fast-mode antigos (`ABSOLUTE_ESR_CAP_WAVENET`, etc.) que viraram dead code (ou documentar caso ainda sejam úteis na suíte de testes longos com parametrização explícita).
  * Manter a assinatura parametrizável (`use_hf`) em `run_render_comparison` para caso seja necessário reintroduzir testes específicos de Fast-mode no futuro.
* **Arquivos Afetados:**
  * [tests/parity/cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs#L579) — unificação dos helpers e remoção de testes duplicados (`quick_parity_hf_*` e `live_cross_validation_hf_*`).
* **Critério de Aceitação:**
  * Executar a suíte de paridade rápida (ex.: `cargo test --test parity quick_parity`) deve passar com sucesso executando apenas uma version de teste por modelo.
  * Apenas os testes `quick_parity_lstm_1x16`, `quick_parity_wavenet_ch16`, `quick_parity_a2_full` e `quick_parity_convnet` devem rodar como gates rápidos de paridade com C++.
  * Nenhuma regressão de cobertura efetiva.

---

#### [ ] Tarefa 1.3: Corrigir Status SKIPPED e Avisos Falso-Verdes no Tests-Long

* **Achado Associado:** [F-S5](file:///home/fabio/nam-rs/TODO-findings.md#L279) — tests-long: aviso "falso-verde" disparado para fase legitimamente SKIPPED.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Quando o daemon do PipeWire não está em execução, a fase correspondente do `tests-long.sh` retorna 77 (`SKIPPED`) em menos de 1 segundo. No entanto, o script emite um alerta de "falso-verde" (completado em < 1s) e reporta a fase como "PASSED" no resumo final.
  * Garantir a propagação correta do status 77 da função de execução até o relatório de auditoria final (`AUDIT SUMMARY`).
  * Suprimir o aviso de execução rápida (< 1s) quando a fase retornar status 77 (SKIPPED).
* **Arquivos Afetados:**
  * [utils/tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh#L428) — ajustar tratamento de status e avisos de duração na função `run_phase`.
* **Critério de Aceitação:**
  * Simular a ausência de PipeWire (ex.: rodando com `PIPEWIRE_REMOTE=does_not_exist`) e verificar se a fase é exibida como `SKIPPED` em amarelo, sem emitir o aviso de falso-verde de < 1s.
  * O relatório de resumo final (`AUDIT SUMMARY`) deve listar "SKIPPED" para a fase correspondente.
