<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO Sprints — Épico B (Fechamento de Cobertura)

Este documento contém o planejamento de **Sprints e Tarefas Técnicas** focado na execução do **ÉPICO B — Fechamento de cobertura de validação dos novos paths (🔴 CRÍTICO)**, de acordo com o plano documentado em `TODO-audit.md`.

## Visão Geral e Criticidade

Os caminhos dinâmicos (`WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn`) e a arquitetura `ConvNet` foram expostos e habilitados, mas carecem de instrumentação rigorosa na cadeia de testes. O risco reside no fato de que regressões de inferência ou problemas numéricos (subnormais, instabilidade) não seriam detectados.

### Análise Crítica: O que não pode passar batido?

1. **FiLM em A2 (F5):** O motor NAM C++ rejeita modelos FiLM no seu *fast-path*, jogando-os para o motor dinâmico. O NAM-rs hoje implementa isso no *fast-path* por um superset intencional, mas **nunca comprovado sonoramente**. Esta discrepância deve ser sanada e testada empiricamente (Tarefa B.1.1).
2. **Suporte C++ ao ConvNet:** A geração de goldens depende da capacidade do utilitário `render` (C++) do projeto NAMCore processar modelos ConvNet corretamente.
3. **Thresholds Calibrados:** Para os novos caminhos (ConvNet, Dinâmicos), não podemos nos dar ao luxo de criar "testes placebos" com bounds frouxos. As comparações devem garantir `SNR ≥ 70dB` contra o golden.
4. **PGO e Hot-paths (F4/F8):** É inútil fazer *benchmarks* se eles não forem pinçados pelo PGO (`utils/build-release.sh`). As novas funções do `inference_bench.rs` precisam impreterivelmente carregar os sufixos corretos (ex: `_64samp`) mapeados no script.

---

## SPRINT B.1: Roteamento e Setup de Fixtures

**Objetivo:** Obter os modelos `.nam` faltantes (representativos dos caminhos expostos) e firmar o roteamento paramétrico definitivo (F3, F4, F5).

### [X] Tarefa B.1.1: Resolução Definitiva da Política A2+FiLM (F5)

* **Resultado:** **Caso B** — fast-path `WaveNetA2<CH>` diverge do motor dinâmico C++.
  * CH=3: SNR 18.1 dB, CH=8: SNR 36.0 dB (ambos abaixo do limiar de 70 dB).
  * C++ `a2_fast.cpp` rejeita FiLM → fallback p/ `WaveNet` genérico. Rust agora equipara: FiLM ativo força `WaveNetA2Dyn`.
* **Ação Investigativa:** Gerados modelos `wavenet_a2_film_lite.nam` (CH=3) e `wavenet_a2_film_full.nam` (CH=8) com 4 chaves FiLM ativas (`conv_post_film`, `input_mixin_post_film`, `activation_post_film`, `layer1x1_post_film`), `condition_size=1`. Goldens C++ renderizados via `NeuralAmpModelerCore v0.5.3` (roteamento fallback p/ motor genérico).
* **Implementação da Política:**
  * Alterado `src/loader/nam_json/topology.rs` (`check_film_all_inactive`): quando FiLM ativo → retorna `A2TopologyResult::Dynamic`.
  * Adicionado `condition_size` ao `WaveNetA2Dyn` e suporte a carregamento de pesos FiLM em `dynamic.rs::set_weights()` (após `l1x1_b`).
  * Funções FiLM de `set_weights.rs` tornadas `pub(crate)` p/ reuso pelo motor dinâmico.
* **Testes:** `test_a2_film_routes_to_dynamic` (unitário em topology.rs). Smoke tests em `golden_vectors.rs` verificam roteamento p/ `WavenetA2Dyn` e finitude da saída.
* **Limpeza:** Eliminados comentários ambíguos e `if` vazio no bloco FiLM de `topology.rs`.
* **Nota p/ tarefas futuras:** Fixtures FiLM estão nos repositórios. Goldens C++ (`golden_wavenet_a2_film_*.bin`) foram gerados mas não são usados nos testes atuais pois o motor genérico C++ interpreta a stream A2 em ordem diferente. Dados brutos preservados para eventual calibração cruzada futura.

* **Ação Investigativa:** Gerar/modificar um modelo A2 (CH=3 ou CH=8) com `condition_size = 1` e matriz FiLM ativa. Renderizar um *golden* no C++ (que fará roteamento fallback pro dinâmico) e compará-lo via Rust (que fará roteamento via `WaveNetA2<CH>`).
* **Implementação da Política:**
  * **Caso A (Bate perfeitamente):** Aceitar formalmente o "superset". Manter o modelo como golden oficial de A2+FiLM.
  * **Caso B (Diverge/Falha):** Alterar `src/loader/nam_json/topology.rs` (função `check_film_all_inactive`) para retornar `Rejected` quando houver FiLM, forçando a queda para o `WaveNetA2Dyn` e equiparando com o roteamento C++.
* **Limpeza:** Eliminar comentários ambíguos e o `if` vazio associado no módulo `topology.rs`.

### [X] Tarefa B.1.2: Aquisição de Fixtures (ConvNet e Dinâmicos) (F3, F4)

* **Resultado:** Três modelos sintéticos gerados deterministicamente via `tests/fixtures/generate_b1_2_fixtures.py`.
* **ConvNet:** `convnet_test.nam` — 2 blocos (CH=8→4, K=3, Dil=[1,2,4], Tanh), sem post-stack head, `head_scale=0.02`. 157 pesos.
* **WaveNetDyn:** `wavenet_dyn_free.nam` — geometria livre com 2 arrays (CH=7→4, Dil=[1,2,4]+[8,16], K=3, Tanh), não casa com nenhum SKU do catálogo, roteia para `WaveNetModelDyn`. 872 pesos.
* **LstmDyn:** `lstm_dyn_test.nam` — 1 camada × 7 hidden units (não catalogado: 3,8,12,16,24,40), roteia para `LstmModelDyn`. 274 pesos.
* **Verificação:** Smoke tests em `tests/fixture_b1_2_smoke.rs` confirmam roteamento correto para `StaticModel::ConvNet`, `StaticModel::WavenetDyn` e `StaticModel::LstmDyn`.
* **Nota p/ tarefas futuras:** Goldens C++ e testes de paridade (Sprint B.2) dependem destes fixtures e do script `golden_gen_build.sh` ser atualizado para incluí-los.

---

## SPRINT B.2: Goldens e Paridade Estrita (F3, F4)

**Objetivo:** Expandir a suite de renderização C++ e assegurar validação numérica com thresholds calibrados (sem fallback placebo).

### [ ] Tarefa B.2.1: Ampliação do Gerador C++ (`golden_gen_build.sh`)

* **Ação:** Incluir os modelos obtidos na Sprint B.1 dentro das listas de renderização automática (`MODELS` e arrays V2) do script `tests/fixtures/golden_gen_build.sh`.
* **Validação:** Rodar o processo localmente via `NAM_AUTO_BUILD_GOLDENS=1 utils/tests-long.sh` e assegurar que o compilador e runtime C++ consigam despejar os `.bin` do ConvNet e dinâmicos de forma intacta.

### [ ] Tarefa B.2.2: Expansão da Suíte de Paridade

* **Ação (Golden Vectors):** Adicionar testes em `tests/golden_vectors.rs` para ConvNet e dinâmicos (`test_golden_vectors_convnet`, etc).
* **Ação (Threshold Calibration):** Modificar `tests/common/validation.rs` para incluir a tolerância calibrada de SNR exigindo nível realístico (`SNR ≥ 70 dB`) e anotar o ganho como `// Measured: ...`.
* **Ação (Live Cross-Validation):** Inserir os equivalentes em `tests/cpp_parity.rs` para a suite longa (`live_cross_validation_convnet`, etc).

---

## SPRINT B.3: RT-Safety (Soaks) e Performance (PGO)

**Objetivo:** Assegurar perenidade na thread de áudio e extração de máxima performance nativa na compilação *Release*.

### [ ] Tarefa B.3.1: Cobertura Soak (Endurance)

* **Ação:** Atualizar `tests/soak_test.rs` implementando instâncias de execução extrema (10M de frames) em blocos.
* **Escopo Exigido:**
  * ConvNet
  * WaveNetModelDyn
  * LstmModelDyn
  * WaveNetA2Dyn (Gated/Blended)
* **Aserções:** Ausência total de subnormais, `NaN` ou `Inf`; checagem restrita de *zero-allocs*.

### [ ] Tarefa B.3.2: Benches PGO-aware

* **Ação:** Adicionar funções em `benches/inference_bench.rs` focado em um bloco de áudio regime RT (64 samples).
* **Escopo Exigido:** 1 Modelo Dinâmico representativo e 1 ConvNet.
* **Atenção PGO:** Assegurar a parametrização do nome do teste/grupo de bench para incluir o sufixo necessário para *match* (ex: `_64samp`) conforme definido em `utils/build-release.sh`, caso contrário eles serão ignorados pelo perfilador BOLT/PGO.
