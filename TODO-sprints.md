<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO Sprints — Planejamento de Execução Ágil (nam-rs)

Este documento descreve os sprints, épicos e tarefas técnicas de engenharia planejados de forma ágil para sanar os apontamentos mapeados em `TODO-findings.md`. O objetivo é garantir segurança matemática, robustez do software e paridade total com a especificação C++.
Obs: Regularmente atualizar o "docs/cpp_parity_map.md" com o progresso obtido

---

## Matriz de Riscos e Prioridades

| Severidade     | Descrição                                                                    | Impacto no Sistema                                                                       |
|:-------------- |:---------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------- |
| 🔴 **Crítica** | Bug de Runtime com UB (Comportamento Indefinido) ou quebra grave de paridade | Pode causar falhas de segmentação, vazamentos ou áudio severamente corrompido.           |
| 🟠 **Alta**    | Falha de validação defensiva (fail-closed)                                   | Permite o carregamento de dados inconsistentes sem a devida rejeição preventiva.         |
| 🟡 **Média**   | Drift matemático secundário ou latência reportada incorretamente             | Impacta compatibilidade com hosts CLAP ou gera pequenos artefatos em cantos específicos. |
| ⚪ **Baixa**   | Documentação obsoleta ou ausência de testes em cenários incomuns             | Dificulta a manutenção do projeto, mas não gera falhas ativas de software.               |

---

## 🏃 Sprint 1: Correção de Bugs Críticos de Execução (Épico E-1)

**Foco:** Mitigar riscos de Comportamento Indefinido (UB) e corrigir anomalias de preaquecimento de modelos.

### 🔴 Tarefa T1.1: Ajustar Prewarm do `condition_dsp` LSTM no WaveNet [DONE]

* **Descrição:** Substituir o valor fixo `cond_dsp.prewarm(0)` pela contagem de amostras calculada dinamicamente pelo próprio sub-modelo.
* **Ações:**
  1. Alterar `src/models/wavenet/model_dyn.rs` (linha 349) para chamar `cond_dsp.prewarm(cond_dsp.prewarm_samples())`.
  2. Verificar se o prewarm de LSTMs filhas executa a quantidade adequada de silêncio para estabilizar o estado recorrente.
* **Verificação:** Rodar suite de testes WaveNet e garantir integridade.
* **Referência:** Finding 7.2.1.

### 🔴 Tarefa T1.2: Prevenir UB no `LstmModelDyn` para `num_layers == 0` [DONE]

* **Descrição:** Rejeitar modelos de LSTM que especifiquem contagem de camadas igual a zero na raiz e adicionar proteção defensiva no hot-path.
* **Ações:**
  1. Alterar `src/loader/nam_json/topology/lstm.rs` para rejeitar o carregamento se `num_layers == 0` retornando `None`.
  2. Inserir proteção nos métodos SIMD (`process_avx2`, `process_avx512`, `process_avx512_vnni_bf16`) em `src/models/lstm/model_dyn.rs`: `if self.layers.is_empty() { return; }` para evitar dereferenciamento de ponteiros nulos/dangling.
* **Verificação:** Executar testes unitários do LSTM (`cargo test --lib models::lstm`).
* **Referência:** Finding 7.2.3.

### 🟠 Tarefa T1.3: Adicionar Validação de Canais Mono no LSTM Loader [DONE]

* **Descrição:** Garantir que o loader do LSTM rejeite modelos estéreo ou multi-canal, uma vez que o nam-rs suporta apenas mono.
* **Ações:**
  1. Adicionar o campo `out_channels` no struct `NamConfig` em `src/loader/nam_json/model.rs`.
  2. Implementar checagem em `src/loader/nam_json/topology/lstm.rs` para rejeitar (retornar `None`) caso `in_channels` ou `out_channels` sejam diferentes de `1` (quando presentes).
* **Verificação:** Adicionar teste unitário no `nam_json_test.rs` validando rejeição de LSTM multi-canal.
* **Referência:** Finding 7.2.5.

---

## 🏃 Sprint 2: Validação Defensiva de Topologias WaveNet (Épico E-1)

**Foco:** Reforçar barreiras fail-closed no parser do WaveNet A1 para evitar processamento silencioso de dados inválidos.

### 🟠 Tarefa T2.1: Implementar Guardrails Fail-Closed no Path Dinâmico A1 [DONE]

* **Descrição:** Rejeitar modelos WaveNet A1 livres que utilizem recursos de gating ou FiLM não portados para a arquitetura A1.
* **Ações:**
  1. Inserir validação em `get_wavenet_topology` (`src/loader/nam_json/topology/wavenet.rs`): se for topologia livre/dinâmica A1 (`WavenetTopologyResult::Free`), percorrer as camadas JSON e rejeitar caso `gating_mode`, `head1x1`, `layer1x1`, ou objetos `FiLM` estejam ativos ou definidos.
* **Verificação:** `cargo test --lib loader` para assegurar integridade dos parsers.
* **Referência:** Finding 7.2.2.

### 🟠 Tarefa T2.2: Impedir Associação Incorreta de Catálogo WaveNet A1 com `condition_dsp` [DONE]

* **Descrição:** Garantir que modelos com sub-modelos de condicionamento não sejam mapeados para os SKUs estáticos rápidos que não processam condicionamento.
* **Ações:**
  1. Atualizar o bloco `catalog_compatible` em `src/loader/nam_json/topology/wavenet.rs` (linhas 337-339) para incluir a asserção `&& data.config.condition_dsp.is_none()`.
* **Verificação:** Adicionar teste negativo garantindo desvio correto para o dynamic engine caso `condition_dsp` esteja no JSON.
* **Referência:** Finding 7.2.4.

---

## 🏃 Sprint 3: Refatoração Cosmética e Ajustes de Latência (Épico E-2)

**Foco:** Sincronizar o cálculo de preaquecimento de modelos e comportamentos avançados com a referência C++.

### 🟡 Tarefa T3.1: Corrigir Cálculo de Amostras de Prewarm no WaveNet [DONE]

* **Descrição:** Garantir que o método `prewarm_samples()` retorne a soma correta de todos os arrays e inclua a latência de preaquecimento do sub-modelo `condition_dsp`.
* **Ações:**
  1. Corrigir `WaveNetModel::prewarm_samples()` para somar os campos receptivos de `array1` e `array2`.
  2. Corrigir `WaveNetModelDyn::prewarm_samples()` em `src/models/wavenet/model_dyn.rs` para somar os tamanhos de receptores de todas as camadas na lista, além de somar o valor de prewarm do `condition_dsp` (em vez de usar `.max()`).
* **Verificação:** Testes de prewarm e integridade.
* **Referência:** Finding 7.4.1.
* **Conclusão (2026-07-02):**
  * `WaveNetModel::prewarm_samples()` (mod.rs:85-86): alterado de `self.array1.receptive_field_size` para `self.array1.receptive_field_size + self.array2.receptive_field_size`.
  * `WaveNetModelDyn::prewarm_samples()` (mod.rs:111-120): alterado para somar `receptive_field_size` de todos os arrays via `.iter().map(|a| a.receptive_field_size).sum()` e usar `+=` para `cond_dsp.prewarm_samples()` em vez de `.max()`.
  * 1077 testes passam (incluindo todos os 96 testes wavenet e 11 testes de prewarm).

### 🟡 Tarefa T3.2: Ajustar Propagação de Saída de Cabeça no Cascade A2 [DONE]

* **Descrição:** Garantir que o cascade A2 dinâmico propague as saídas de áudio pós-recanalização, e não o buffer bruto acumulado da cabeça.
* **Ações:**
  1. Modificar a lógica de encadeamento em `src/models/a2/cascade.rs` (ou correspondente) para extrair o sinal recanalizado da cabeça do estágio anterior.
* **Verificação:** `cargo test` geral.
* **Referência:** Finding 7.4.2.
* **Conclusão (2026-07-02):**
  * `cascade.rs`: Adicionado buffer `intermediate_head_output` e campo `max_head_size` ao `WaveNetA2Cascade`.
  * `cascade.rs`: `process_internal` reestruturado para que arrays intermediários computem a saída de cabeça pós-recanalização (`cascade_head_finalize`) e a propaguem para o array seguinte através do novo método `cascade_seed_head_from_output`.
  * `process.rs`: Substituído o método `cascade_seed_head` (que copiava `head_accum` bruto com stride incorreto usando `channels` em vez de `head_accum_size`) por `cascade_seed_head_from_output` que recebe a saída de cabeça já processada.
  * 1077 testes passam (incluindo todos os 235 testes A2, 96 testes wavenet e o golden vector `test_golden_vectors_wavenet_condition_dsp` do modelo cascade multi-array).

### 🟡 Tarefa T3.3: Implementar Convolução de Cabeça Completa no A2 para `head_size > 1`

* **Descrição:** Habilitar suporte completo a Conv1D com bias e escala na finalização de modelos A2 que possuam tamanho de cabeça estendido.
* **Ações:**
  1. Modificar o processo em `src/models/a2/model/static/process.rs` (e correspondente dinâmico) para aplicar convolução 1D completa ao invés de projeção linear pura para `head_size > 1`.
* **Verificação:** Testes de fixtures sintéticos estendidos.
* **Referência:** Finding 7.4.3.

---

## 🏃 Sprint 4: Sincronização de Documentação e Baselines (Épico E-3 & E-4)

**Foco:** Limpeza de claims falsos na documentação e expansão de testes cruzados de regressão.

### ⚪ Tarefa T4.1: Atualizar `docs/testing.md` e `tests/fixtures/README.md`

* **Descrição:** Atualizar referências desatualizadas a testes do WaveNet Lite e classificar corretamente o comportamento real de `wavenet_a2_max.nam`.
* **Ações:**
  1. Editar `docs/testing.md` removendo as claims de desativação do WaveNet Lite.
  2. Atualizar a tabela de modelos em `tests/fixtures/README.md` indicando o carregamento correto de `wavenet_a2_max` e seu bloqueio por guardrail.
* **Verificação:** Visualização dos arquivos de documentação markdown.
* **Referência:** Finding 7.5.1, 7.5.2.

### ⚪ Tarefa T4.2: Atualizar Placeholders de Threshold do Flagship A2 Max

* **Descrição:** Ajustar comentários e baselines experimentais desatualizados sobre o ESR e SNR reais obtidos pelo flagship `wavenet_a2_max`.
* **Ações:**
  1. Atualizar em `tests/common/validation.rs` o comentário de threshold de `wavenet_a2_max` com o ESR real discrepante medido (`3.61e1`).
  2. Ajustar os comentários do teste `test_golden_vectors_wavenet_a2_max` em `tests/golden_vectors.rs` com os números de baseline reais.
* **Verificação:** `cargo test --test golden_vectors` (ignorado, mas compilável e coerente).
* **Referência:** Finding 7.5.3, 7.5.4.

### 🟡 Tarefa T4.3: Implementar Live Cross-Validation para Modelos Dinâmicos A2

* **Descrição:** Estender a suite `tests/cpp_parity.rs` para realizar renderização comparativa ao vivo contra o C++ para modelos de FiLM, Blended, Gated e o Flagship Max (uma vez que for destravado).
* **Ações:**
  1. Adicionar asserções sob macros de live render em `tests/cpp_parity.rs` visando testar o parser dinâmico diretamente contra as saídas geradas pelo executável `nam_render` do C++.
* **Verificação:** Execução de `tests/cpp_parity.rs` (ou `utils/tests-quick.sh`).
* **Referência:** Finding 7.6.2.
