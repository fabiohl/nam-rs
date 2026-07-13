<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Sprint Backlog — EP1 — Quick suite enxuto e honesto

Este documento detalha o planejamento ágil para execução do épico **EP1**, focado em tornar a suíte de testes rápida (`quick`) mais enxuta, veloz e correta em termos de reporte de status.

---

## Sprint 1: Otimização da Suíte de Testes Rápida (Quick)

### Tarefas Técnicas

#### [x] Tarefa 1.1: Remover `rt_constraints` do Quick

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

#### [x] Tarefa 1.2: Deduplicar Parity no Quick e Limpar Caps Mortos

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

#### [x] Tarefa 1.3: Corrigir Status SKIPPED e Avisos Falso-Verdes no Tests-Long

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

---

## Sprint 2: Dashboard Correto e Determinístico (EP2)

### [ ] Tarefa 2.1: Hotfix de Locale (LC_ALL=C) Global no Dashboard

* **Achado Associado:** [F-S2](file:///home/fabio/nam-rs/TODO-findings.md#L245) — Dashboard: aritmética awk sem LC_ALL=C.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  Garantir que todas as operações aritméticas e de formatação numérica no awk dentro do dashboard sejam imunes a locales regionais (como pt_BR, que usa vírgula decimal). Faremos isso definindo `export LC_ALL=C` no início do script de forma global, garantindo homogeneidade.
* **Arquivos Afetados:**
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Critério de Aceitação:**
  * Executar o dashboard com locales que usam vírgula (ex: `LANG=pt_BR.UTF-8 LC_NUMERIC=pt_BR.UTF-8`) deve computar e exibir as durações totais e parciais perfeitamente (com decimais), sem zerar os contadores.

---

### [ ] Tarefa 2.2: Adicionar A2-FiLM-InputMixinPre na Tabela de Sumário do Oráculo f64

* **Achado Associado:** [F-S4](file:///home/fabio/nam-rs/TODO-findings.md#L270) — Dashboard: "vs f64: N/A" para A2-FiLM-InputMixinPre.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O modelo `wavenet_a2_film_input_mixin_pre.nam` possui cobertura de teste individual no oráculo, mas não foi incluído no vetor de modelos que compõe o sumário. Isso faz com que a busca do dashboard resulte em `N/A`. Devemos incluí-lo explicitamente no vetor `models` dentro de `test_summary_table` em `reference_oracle_f64.rs`.
* **Arquivos Afetados:**
  * [tests/parity/reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs#L1187)
* **Critério de Aceitação:**
  * O teste `test_summary_table` deve imprimir a linha de sumário para `wavenet_a2_film_input_mixin_pre.nam` com família `A2-FiLM-InputMixinPre`.

---

### [ ] Tarefa 2.3: Emissão de Métricas em JSON Lines (JSONL)

* **Achado Associado:** [F-I1](file:///home/fabio/nam-rs/TODO-findings.md#L311) — Métricas machine-readable (JSON Lines) — eliminar scraping awk.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  Substituir o frágil scraping de texto humano por uma exportação determinística estruturada.
  * Criar thread-locals em `tests/common/validation.rs` (`METRIC_MODEL` e `METRIC_MODE`) para armazenar o nome do modelo e o modo de teste ativo.
  * Atualizar o helper `report_dsp_fidelity_impl` para capturar esses metadados e escrever uma linha JSON no arquivo especificado pela variável de ambiente `NAM_METRICS_JSONL` (se definida).
  * Atualizar `golden_vectors.rs` e `cpp_parity.rs` para definir essas variáveis thread-local antes de chamar os reportadores de fidelidade.
* **Arquivos Afetados:**
  * [tests/common/validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
  * [tests/models/golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs)
  * [tests/parity/cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
* **Critério de Aceitação:**
  * A execução dos testes de fidelidade (`golden_vectors` e `cpp_parity`) sob a variável `NAM_METRICS_JSONL=metrics.jsonl` deve gravar uma linha JSON válida por modelo contendo as chaves `label`, `esr`, `esr_db`, `snr_db`, `mrstft` e `mse`.

---

### [ ] Tarefa 2.4: Migração do Parser do Dashboard para JSON Lines com Fallback

* **Achado Associado:** [F-I1](file:///home/fabio/nam-rs/TODO-findings.md#L311) — Ingestão estruturada de métricas.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  Adaptar o dashboard para ler o arquivo JSON Lines gerado.
  * Implementar uma função `parse_jsonl_fidelity` no dashboard que lê o JSONL (usando `jq` se disponível ou `awk` regex simplificado) e popula as tabelas internas.
  * Integrar no parser `parse_golden_vectors`, mantendo o parser antigo de texto humano como um fallback resiliente caso o arquivo JSONL não esteja presente.
* **Arquivos Afetados:**
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh#L347)
* **Critério de Aceitação:**
  * O dashboard deve renderizar todas as informações de fidelidade a partir do arquivo JSONL sem erros de parsing ou drifts.

---

### [ ] Tarefa 2.5: Ingestão de Paridade com C++ (Fim do N/A no ConvNet)

* **Achado Associado:** [F-S3](file:///home/fabio/nam-rs/TODO-findings.md#L255) — "ConvNet vs NAMcore: N/A" no dashboard.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O dashboard mostra `N/A` para ConvNet porque o teste em `golden_vectors.rs` é um teste de consistência interna (self-golden), e não de paridade C++. A verdadeira paridade C++ é medida em `quick_parity_convnet` (em `cpp_parity.rs`).
  * Adicionar a fase de execução de `quick_parity` (utilizando `cargo test --test parity quick_parity`) no dashboard sob o escopo de fidelidade, escrevendo no JSONL de métricas.
  * Mapear o label `"Quick ConvNet"` para `"ConvNet Test"` no parser para preencher a linha do ConvNet no dashboard com o valor real medido (ex: `2.54e-5`, ~45.9 dB).
* **Arquivos Afetados:**
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Critério de Aceitação:**
  * O painel exibe a paridade real do ConvNet vs NAMcore (não mais `N/A`), com o valor exato extraído do teste `quick_parity_convnet`.

---

### [ ] Tarefa 2.6: Coluna f64 por Modelo (Aposentar `~fam.`)

* **Achado Associado:** [F-I2](file:///home/fabio/nam-rs/TODO-findings.md#L323) — ESR vs f64 por modelo — aposentar a aproximação por família.
* **Complexidade/Risco:** Médio.
* **Descrição:**
  Atualmente, a coluna "vs f64" exibe o oráculo de um único modelo representativo para toda a família (marcado com `~fam.`). Vamos substituir isso pela medição real individual de cada modelo.
  * Estender `test_summary_table` in `reference_oracle_f64.rs` para abranger todos os modelos golden ativos na suíte.
  * No dashboard, substituir `ESR_F64_FAMILY_MAP` por um mapa exato direto `ESR_F64_MODEL_MAP` que aponta cada label para o seu respectivo arquivo `.nam`.
  * Remover os caracteres de aproximação `~`, a legenda do rodapé e a respectiva variável `ESR_F64_EXACT_MATCH`. Modelos que não possuem teste de oráculo exibirão `N/A` de forma transparente.
* **Arquivos Afetados:**
  * [tests/parity/reference_oracle_f64.rs](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs)
  * [utils/quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Critério de Aceitação:**
  * A tabela exibe valores de oráculo f64 exatos e individuais por modelo, sem marcas de aproximação `~` ou legendas no rodapé sobre aproximação familiar.

---

### [ ] Tarefa 2.7: Documentar Tolerâncias no Contrato de Qualidade

* **Achado Associado:** [F-S6](file:///home/fabio/nam-rs/TODO-findings.md#L289) — Tolerâncias do contrato não documentadas.
* **Complexidade/Risco:** Baixo.
* **Descrição:**
  O arquivo `quality-contract.txt` serve como linha de base absoluta, mas não explicita as tolerâncias que o script de dashboard aplica ao validar (ex: +10% latência, 10x margem de ESR/MR-STFT). Vamos adicionar uma seção explicativa no topo do contrato documentando essas margens e as políticas de verificação/recalibração.
* **Arquivos Afetados:**
  * [docs/quality-contract.txt](file:///home/fabio/nam-rs/docs/quality-contract.txt)
* **Critério de Aceitação:**
  * Inserção da documentação no cabeçalho do contrato de modo que seja visível para humanos e ignorada pelo parser de validação do script.
