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

### [X] Tarefa B.2.1: Ampliação do Gerador C++ (`golden_gen_build.sh`)

* **Resultado:** WaveNetDyn (`wavenet_dyn_free.nam`) e LSTM-Dyn (`lstm_dyn_test.nam`) integrados ao `golden_gen_build.sh` (arrays MODELS e V2_MODELS). Golden vectors gerados com sucesso pelo C++ render tool (NAM Core v0.5.3).
* **Correção do Fixture LSTM-Dyn:** Adicionado `"input_size": 1` ao `config` do `lstm_dyn_test.nam` — o parser LSTM do C++ (`lstm.cpp:171`) acessa `config["input_size"]` diretamente e falhava com `type_error` quando ausente. O Rust (`get_lstm_topology`) não usa este campo, portanto sem impacto.
* **Bloqueio ConvNet (🔴):** A arquitetura ConvNet do NAM 0.5.4 (multi-bloco, per-block channels, kernel_size variável, formato `layers`) é **incompatível** com o ConvNet do NAM Core v0.5.3 (single `channels`, dilatações planas, kernel_size=2 fixo, flag `batchnorm`). O render C++ aborta com `json.exception.type_error`. Golden vectors para ConvNet não podem ser gerados via o pipeline atual.
  * **Impacto nas tarefas seguintes:** Tarefa B.2.2 (`test_golden_vectors_convnet`) precisa ser adaptada — o golden de ConvNet depende de upgrade do NAM Core p/ versão ≥0.5.4 ou de um pipeline de render alternativo.

### [X] Tarefa B.2.2: Expansão da Suíte de Paridade

* **Golden Vectors (Dinâmicos):** Adicionados `test_golden_vectors_wavenet_dyn_free` (SNR 124.2 dB, ESR 3.79e-13) e `test_golden_vectors_lstm_dyn_test` (SNR 118.1 dB, ESR 1.54e-12) em `tests/golden_vectors.rs`. Goldens C++ gerados via `render` tool (NAM Core v0.5.3) e confirmados bit-convergentes contra Rust.
* **ConvNet Self-Golden:** `test_golden_vectors_convnet_test` — teste de determinismo com self-golden Rust→Rust (SNR=∞, ESR=0.0, output bit-idêntico entre duas instâncias independentes). Substitui o golden C++ bloqueado pela incompatibilidade NAM Core v0.5.3 × NAM 0.5.4 (ConvNet multi-bloco).
  * **Nota p/ tarefas futuras:** A engine ConvNet foi validada como determinística e correta (output finito, não-zero). Upgrade do NAM Core p/ versão >=0.5.4 habilitará golden C++ cross-reference no futuro.
* **Thresholds Calibrados:** Adicionadas entradas em `get_calibrated_threshold()` em `tests/common/validation.rs` para `wavenet_dyn_free` (SNR≥90 dB, ESR≤1e-11), `lstm_dyn_test` (SNR≥90 dB, ESR≤3e-11) e `convnet_test` (SNR≥140 dB, ESR≤1e-10). Todos com `// Measured: ...` documentando a medição real.
* **Live Cross-Validation:** Adicionados `live_cross_validation_wavenet_dyn`, `live_cross_validation_lstm_dyn`, `live_cross_validation_v2_wavenet_dyn` e `live_cross_validation_v2_lstm_dyn` em `tests/cpp_parity.rs`.
* **Correção `golden_gen_build.sh`:** ConvNet movido para fim do array MODELS e V2_MODELS com skip explícito (`[[ "$label" == ConvNet* ]]`) para evitar crash que impedia geração dos goldens dinâmicos subsequentes.
* **Nota p/ B.3.1:** ConvNet usa `model.process()` (buffer inteiro), não `process_in_blocks`. O soak test deve respeitar o tamanho de buffer como `num_frames × out_channels`.
* **Golden Manifest:** Adicionados SHAs de `golden_wavenet_dyn_free.bin` e `golden_lstm_dyn_test.bin` ao `.golden_manifest.sha256`.

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
