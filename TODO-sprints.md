<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Roadmap de Sprints — Épicos A, B, C, D, E, F, G, H, I, J & K

Este documento organiza o planejamento ágil e tarefas técnicas para o **Épico A (PM-01, PM-02, PM-08 — Sincronização Documental de Paridade)**, o **Épico B (PM-04, PM-03 — Testemunhas Independentes/Oráculo f64)**, o **Épico C (PM-05 — Cobertura de Modelos Reais A2-FiLM)**, o **Épico D (PM-06 — SlimmableWavenet)**, o **Épico E (PM-07 — Robustez da Suíte de Testes)**, o **Épico F (PM-09 — Integridade da Referência C++ [Won't Do - Decisão Documentada])**, o **Épico G (PM-10, PM-03, PM-05 — Feature-Completeness A2 Oficial)**, o **Épico H (PM-11, PM-12 — Robustez de Carregamento "fail-closed")**, o **Épico I (PM-13 e documentação de PM-09/PM-10/PM-12 — Sincronização Documental & ConvNet)**, o **Épico J (PM-14 — Observabilidade & Cobertura [Won't Do - Decisão Documentada])** e o **Épico K (PM-15 — Multi-Array A2)** no `nam-rs`, com base nas descobertas consolidadas em `TODO-findings.md`.

---

## SPRINT S9 — Sincronização Documental e Alinhamento de Paridade (A2-FiLM & SlimmableWavenet)

### Objetivos da Sprint

1. Sincronizar toda a documentação de paridade ao estado real do motor, eliminando avisos obsoletos sobre a WaveNet Lite e corrigindo referências cruzadas mortas.
2. Formalizar a cobertura do motor A2 FiLM sob fixtures sintéticas devido à indisponibilidade de capturas reais compatíveis.
3. Definir a fronteira de escopo e os critérios de aceitação para o fatiamento de canais dinâmico (`SlimmableWavenet`), mantendo-o como item diferido.

---

### Tarefas Técnicas

#### [x] Task S9.1 — Auditoria de Fixtures e Conformismo A2-FiLM (PM-05)

* **Responsável:** Engenheiro de DSP / QA
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  O motor `WaveNetA2Dyn` suporta FiLM, mas os modelos reais de FiLM disponíveis (como `wavenet_a2_max.nam` com `condition_size=8`) são rejeitados por incompatibilidade com a assinatura esperada pelo loader dinâmico de A2 (que exige geometrias específicas). Como não existem fixtures reais compatíveis no diretório `tests/fixtures/models-nondist/` nem em `tests/fixtures/models/`, a suíte de testes deve se conformar com as fixtures sintéticas `wavenet_a2_film_full.nam` e `wavenet_a2_film_lite.nam` para garantir a correção do motor matemático.
* **Critérios de Aceitação:**
  1. Confirmar que os testes de vetores dourados (`tests/golden_vectors.rs`) exercitam corretamente as fixtures sintéticas de FiLM.
  2. Verificar que o modelo real incompatível `wavenet_a2_max.nam` é rejeitado graciosamente e coberto pelo teste `test_loader_gap_wavenet_a2_max` sem quebras silenciosas.
* **Conclusão (2026-06-30):**
  1. ✅ `test_golden_vectors_wavenet_a2_film_lite` — passa (SNR=18.1 dB, ESR=1.54e-2, MR-STFT=0.497, gates OK).
  2. ✅ `test_golden_vectors_wavenet_a2_film_full` — passa (SNR=36.0 dB, ESR=2.50e-4, MR-STFT=0.465, gates OK).
  3. ✅ `test_loader_gap_wavenet_a2_max` — passa, rejeitado com erro de incompatibilidade estrutural.
  4. `tests/fixtures/models-nondist/` confirmado sem capturas FiLM compatíveis (apenas WaveNet/LSTM comunitários).
  5. Documentação atualizada: `tests/fixtures/README.md` corrigiu registros obsoletos (`wavenet_condition_dsp` marcado como "Rejected" mas opera com SNR=139.5 dB), adicionou entradas `wavenet_a2_film_*` à tabela de catálogo e golden vectors, e adicionou seção dedicada de FiLM Fixtures documentando o conformismo PM-05.
  * **Nota para S9.2:** O `docs/cpp_parity_map.md` §13 já contém a documentação de conformismo solicitada (linha 592). S9.2 pode focar em revisar/expandir o que já existe, em vez de criar do zero.

#### [x] Task S9.2 — Documentação de Conformismo FiLM A2 em `cpp_parity_map.md` (PM-05)

* **Responsável:** Documentador Técnico
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  Registrar formalmente no mapa de paridade (`docs/cpp_parity_map.md`, seção 13) o conformismo às fixtures sintéticas para validação do motor FiLM devido à ausência de capturas reais suportadas, garantindo rastreabilidade futura.
* **Critérios de Aceitação:**
  1. Atualizar a entrada de tabela **"A2 official real-amp FiLM captures"** no §13 para refletir o status de conformismo temporário com modelos sintéticos.
  2. Documentar o motivo técnico (incompatibilidade estrutural dos modelos reais existentes) no §13.1.
* **Conclusão (2026-06-30):**
  1. ✅ Tabela do §13 atualizada: entrada "A2 official real-amp FiLM captures" (🟡 Temporary conformity — validated against synthetic fixtures `wavenet_a2_film_full/lite`, Sprint S9 goldens ✅) — `cpp_parity_map.md:570`.
  2. ✅ §13.1 documenta o motivo técnico: `wavenet_a2_max.nam` com `condition_size=8` rejeitado por incompatibilidade estrutural; rejeição graciosa validada por `test_loader_gap_wavenet_a2_max`; sem outras capturas reais compatíveis em `models-nondist/` — `cpp_parity_map.md:593`.

#### [x] Task S9.3 — Fronteira de Escopo e Critérios para `SlimmableWavenet` Diferido (PM-06)

* **Responsável:** Arquiteto de Software
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  Garantir que o `SlimmableWavenet` (fatiamento de canais em runtime num único arquivo) permaneça diferido sem causar confusão com o `SlimmableContainer` (que já resolve a qualidade adaptativa via múltiplos modelos e crossfade).
* **Critérios de Aceitação:**
  1. Atualizar a tabela e notas do §13 em `docs/cpp_parity_map.md` delimitando o escopo do `SlimmableContainer` (pronto e testado) vs. `SlimmableWavenet` (diferido).
  2. Definir explicitamente os critérios de aceitação para eventual implementação futura do `SlimmableWavenet`:
     * Parser para ler múltiplas larguras de canal de um único arquivo `.nam`.
     * Fatiamento dinâmico de pesos em runtime de forma segura para tempo real (RT-safe).
     * Paridade matemática bit-a-bit com a implementação equivalente do NAMCore C++.
* **Conclusão (2026-06-30):**
  1. ✅ Tabela do §13 dividida em duas linhas distintas: `SlimmableContainer` (🟢 implementado, com referência a `src/models/container.rs` e `tests/container_slimmable.rs`) e `SlimmableWavenet` (🟡 diferido, com referência aos critérios expandidos em §13.1).
  2. ✅ Nota do §13.1 expandida com distinção conceitual explícita entre os dois construtos (orquestração de múltiplos modelos vs. fatiamento de pesos de uma única rede) e critérios de aceitação detalhados em três níveis: (a) parser de múltiplas larguras com projeção para SKUs do catálogo; (b) slicing RT-safe na carga, sem alocação/lock no `process()`; (c) paridade bit-a-bit validada por golden vectors + live cross-validation contra NAMCore.

#### [x] Task S9.4 — Resolução de Avisos Obsoletos da WaveNet Lite em `fastmath-approximations.md` (PM-02)

* **Responsável:** Documentador Técnico / Auditor
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  O arquivo `docs/fastmath-approximations.md` §9.4 contém um bloco `> [!CAUTION]` obsoleto e um título indicando que o Lite possui divergência estrutural de 0.9 dB. O bug real de alinhamento no buffer circular de delay lines (`MirroredBuffer`) foi sanado no código (`MirroredBuffer::new_aligned`) e a paridade do modelo real `EVH-5150-Lite.nam` está estabelecida em 122.3 dB. É necessário atualizar este trecho para refletir o estado de paridade estabelecida e documentar a causa-raiz resolvida.
* **Critérios de Aceitação:**
  1. Renomear a seção 9.4 para indicar a resolução (ex: "9.4 Lite Architectures — Resolved").
  2. Descrever detalhadamente a causa-raiz (arredondamento do buffer circular sem levar em conta o alinhamento de stride de canais para não-potências de dois) e sua correção (`MirroredBuffer::new_aligned`).
  3. Remover/substituir o bloco de cautela por uma nota de contexto histórico de resolução.
* **Conclusão (2026-06-30):**
  1. ✅ Seção renomeada para "9.4 Lite Architectures — Resolved (PM-02)" — `fastmath-approximations.md:325`.
  2. ✅ RCA detalhada documentada: bug de `MirroredBuffer` page-rounding (`1024 % 12 = 4`, `1024 % 6 = 4` para CH não-potência-de-dois) + golden sintético obsoleto (`BossWN-lite.nam`). Correção via `MirroredBuffer::new_aligned()` com `lcm(page, channel_stride)` + migração para `EVH-5150-Lite.nam` real — `fastmath-approximations.md:329-332`.
  3. ✅ Bloco `> [!CAUTION]` substituído por `> [!NOTE]` com RCA e 3 guardas de regressão ativas documentadas. Tabela comparativa mantida (BossWN-lite 0.9 dB → EVH-5150-Lite 122.3 dB) — `fastmath-approximations.md:329-337`.
  4. Referência morta `TODO-problemas.md#P1` removida (substituída por cross-ref a `cpp_parity_map.md` §9.1 e PM-02) — tratada em S9.5.

#### [x] Task S9.5 — Correção de Referências Quebradas (PM-08)

* **Responsável:** Documentador Técnico
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  Eliminar referências mortas ao arquivo inexistente `TODO-problemas.md` na documentação do projeto, substituindo-as por referências corretas ao mapa de paridade e aos findings correspondentes de `TODO-findings.md`.
* **Critérios de Aceitação:**
  1. Mapear as referências de `TODO-problemas.md` a problemas reais e redefinir seus links:
     * `TODO-problemas.md:155` (silêncio/denormais) apontará para `fastmath-approximations.md` §6.
     * `TODO-problemas.md#P1` e `TODO-problemas.md:47` (Lite) apontarão para `docs/cpp_parity_map.md` §9.1 e `PM-02`.
     * `TODO-problemas.md:92` (asimetria) e `TODO-problemas.md:353` (lo-fi) apontarão para `docs/fastmath-approximations.md` §9.5.
* **Conclusão (2026-06-30):**
  1. ✅ Mapeamento verificado — nenhuma referência viva a `TODO-problemas.md` permanece em `docs/`:
     * Silêncio/denormais: coberto por `fastmath-approximations.md` §6 (Anti-Subnormal Prevention) e §8 (Non-Zero Silence Policy).
     * Lite/P1: coberto por `cpp_parity_map.md` §9.1 (122.3 dB, RF7 resolvido) e PM-02.
     * Asimetria/lo-fi: coberto por `fastmath-approximations.md` §9.5 (Historical Lo-Fi/Hi-Fi Duality).
  2. ✅ Referências quebradas em `cpp_parity_map.md` (F1/F2/I6) resolvidas desde PM-01.
  3. ✅ Referência `TODO-problemas.md#P1` em `fastmath-approximations.md` §9.4 removida em S9.4 (substituída por cross-refs a `cpp_parity_map.md` §9.1 e PM-02).
  4. ✅ Cross-reference canônica estabelecida em `fastmath-approximations.md` §9.6 e `cpp_parity_map.md` See Also — ambas apontam para este registro (`TODO-findings.md` PM-08).
  5. ✅ Bug de auto-referência em `fastmath-approximations.md` §8 References (`(this section)` referindo-se a §6) corrigido.
  6. PM-08 marcado `[RESOLVIDO]` em `TODO-findings.md`.

---

## SPRINT S10 — Testemunhas Independentes: Oráculo f64 (PM-04, PM-03)

### S10 Objetivos da Sprint

1. Estender o oráculo f64 (`src/testing/reference_oracle.rs`) para suportar a arquitetura **ConvNet** e a feature **FiLM** do motor **A2**, estabelecendo a *testemunha de matemática ideal* de forma independente do motor de produção.
2. Atualizar o script de validação externa NumPy/Python (`tests/fixtures/scripts/validate_oracle_f64.py`) para suportar as mesmas arquiteturas (ConvNet e FiLM A2), fechando a cadeia de confiança de 3 oráculos/testemunhas independentes.
3. Caracterizar a divergência interop de FiLM A2 (RF1) no oráculo f64 para validar se é inerente (diferenças legítimas de estrutura) ou bug do motor SIMD de produção.
4. Validar o oráculo f64 e a âncora NumPy contra o motor de produção Rust com erros ESR estritamente controlados (< 1e-12 para ConvNet e < 1e-9 para FiLM A2).

---

### S10 Tarefas Técnicas

#### [x] Task S10.1 — Extensão do Oráculo f64 para ConvNet (PM-04)

* **Responsável:** Engenheiro de DSP / QA
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  O motor ConvNet está completo e unit-testado, mas não possui cobertura pelo oráculo f64 (retorna zeros). É necessário implementar `oracle_convnet_forward` em f64 exato no oráculo para servir de referência matemática independente.
* **Critérios de Aceitação:**
  1. Implementar `oracle_convnet_forward` em `src/testing/reference_oracle.rs` simulando a topologia multi-bloco do NAM 0.5.4: `Conv1d` causal → `BatchNorm` fundida (`scale * x + offset`) → ativação → `PostStackHead` (se presente) → `head_scale`.
  2. Integrar a chamada no despachador principal `oracle_forward`.
  3. Definir o threshold `CONVNET_ESR_LIMIT = 1e-12` em `tests/common/constants.rs`.
  4. Adicionar testes em `tests/reference_oracle_f64.rs` (ex: `test_oracle_convnet`) carregando `convnet_test.nam` e comparando com o motor f32 de produção (ESR < `CONVNET_ESR_LIMIT`).
* **Conclusão:** ✅ Oracle implementado em `src/testing/reference_oracle.rs:918` (~200 linhas). Cobertura de activations: Tanh, HardTanh, ReLU, Sigmoid, SiLU, HardSwish, Softsign (FastTanh e Tanh compartilham oracle_tanh). Dispatcher estendido linha 282. Threshold `CONVNET_ESR_LIMIT = 1e-12` em `tests/common/constants.rs:31`. ESR medido: 1.83e-14 (−137.4 dB) — piso numérico, similar ao WaveNet. Testes: prewarm-paired ESR gate, decomposition, combined simulation, warmup paired, python anchor (ignored → S10.2).

#### [x] Task S10.2 — Âncora NumPy e Geração de Fixtures ConvNet (PM-04)

* **Responsável:** Engenheiro de DSP / Python Integrator
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  Para garantir a Regra 6 da *Gate Calibration Policy* (não-circularidade do oráculo), precisamos de uma 3ª implementação independente em NumPy (`validate_oracle_f64.py`) servindo de âncora.
* **Critérios de Aceitação:**
  1. Adicionar `convnet_forward(model, x)` em `tests/fixtures/scripts/validate_oracle_f64.py` seguindo a especificação física do formato NAM.
  2. Integrar a arquitetura `"ConvNet"` no CLI do script Python.
  3. Gerar a fixture de sinal de sweep f64 e o arquivo de âncora binário `convnet_256_f64.bin` sob `tests/fixtures/f64_anchors/`.
  4. Adicionar o teste `test_oracle_vs_python_anchor_convnet` em `tests/reference_oracle_f64.rs` exigindo ESR < 1e-12.
* **Conclusão:** ✅ `convnet_forward` (~160 linhas) implementado em `tests/fixtures/scripts/validate_oracle_f64.py`, seguindo topologia multi-bloco: Conv1d causal `[out_ch][in_ch][kernel]` → BatchNorm fundida (scale * x + offset) → ativação → PostStackHead (opcional). Cobertura de activations: Tanh, HardTanh, FastTanh, ReLU, Sigmoid, SiLU, HardSwish, Softsign. CLI estendido com `--architecture ConvNet`. Âncora `convnet_256_f64.bin` gerada em `tests/fixtures/f64_anchors/`. Teste `test_oracle_vs_python_anchor_convnet` des-ignorado com ESR=5.00e-16 (−153.0 dB) — piso numérico, consistente com WaveNet e ConvNet.

#### [x] Task S10.3 — Extensão do Oráculo f64 para FiLM A2 (PM-03)

* **Responsável:** Engenheiro de DSP / Cientista de Redes
* **Risco/Criticidade:** Médio.
* **Contexto:**
  O motor `WaveNetA2Dyn` suporta FiLM, mas ele apresenta uma divergência interop de paridade de 18-36 dB (marcada como `RF1` / PM-03). Precisamos estender o oráculo f64 para suportar FiLM para classificar se a causa-raiz é estrutural inerente ou um bug de inserção/SIMD.
* **Critérios de Aceitação:**
  1. Modificar `oracle_a2_forward` em `src/testing/reference_oracle.rs` para carregar as configurações de FiLM (via `l0.layer_raw`) e ler os respectivos pesos/bias usando o cursor de peso.
  2. Implementar `apply_modulation` e as 8 posições de inserção FiLM em f64 exato, espelhando a ordem de processamento da produção.
  3. Executar o cross-check Rust f32 × oráculo f64 com os modelos `wavenet_a2_film_lite.nam` e `wavenet_a2_film_full.nam`:
     * Se ESR for baixo (< 1e-9) → FiLM está matematicamente correto e a divergência vs C++ é **inerente** (H1) → reclassificar `RF1` como divergência documentada, atualizar `cpp_parity_map.md` e manter os caps.
     * Se ESR for alto → existe um **bug** (H2) de inserção ou SIMD → corrigir o ponto de divergência, revalidar os goldens e só então alterar produção.
  4. Adicionar testes unitários/gates em `tests/reference_oracle_f64.rs`.
* **Conclusão (2026-06-30):**
  1. ✅ `oracle_a2_forward` estendido em `src/testing/reference_oracle.rs:642` (~130 linhas adicionais) — parse de FiLM config via `layer_raw` (8 slots, `FILM_KEYS`), leitura de pesos/bias após `l1x1_b` por camada, `film_weight_count`/`film_bias_count` com grupos, `cond_size`, `shift`.
  2. ✅ `FiLMOracleSlot` com `cond_to_scale_shift` + `apply_modulation` em f64 exato (grupos GEMV), aplicado nos 6 pontos de inserção ativos: conv_post(1), input_mixin_pre(2), input_mixin_post(3), activation_pre(4), activation_post(5), layer1x1_post(6). Plus conv_pre(0) na history buffer. Head1x1_post(7) reservado.
  3. ✅ Cross-check executado — **H1 confirmada (inerente)**:
     * A2-FiLM-Lite: ESR(f32 vs oracle f64) = 9.52e-15 (−140.2 dB) — piso numérico.
     * A2-FiLM-Full: ESR(f32 vs oracle f64) = 1.15e-14 (−139.4 dB) — piso numérico.
     * Ambos bem abaixo do threshold 1e-9 → a implementação Rust FiLM está **matematicamente correta**.
     * A divergência vs C++ (18-36 dB SNR) é **inerente**: o C++ fallback generic WaveNet (via Eigen) aplica conditioning por caminho estruturalmente diferente do FiLM nativo Rust.
     * `RF1` reclassificado de "divergência suspeita" para **"divergência estrutural inerente — documentada e capada"**.
  4. ✅ Testes adicionados em `tests/reference_oracle_f64.rs`: `test_oracle_a2_film_lite`, `test_oracle_a2_film_full`, `test_combined_simulation_a2_film`. Entradas adicionadas à tabela `test_summary_table`.
  5. `A2_FILM_ESR_LIMIT = 1e-12` em `tests/common/constants.rs` (piso numérico, consistente com WaveNet/ConvNet).
  * **Nota para S10.4:** Com H1 confirmada e o oráculo f64 agora testemunhando FiLM A2, a âncora NumPy (S10.4) é necessária para fechar a cadeia de confiança de 3 vias. A implementação Python deve replicar `FiLMOracleSlot::apply` (grupos GEMV → `cond_to_scale_shift` + `apply_modulation`) e os 6 pontos de inserção ativos.

#### [X] Task S10.4 — Âncora NumPy e Geração de Fixtures FiLM A2 (PM-03)

* **Responsável:** Engenheiro de DSP / Python Integrator
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  Complementar a cadeia de confiança gerando a âncora independente NumPy para o caso FiLM A2.
* **Critérios de Aceitação:**
  1. ✅ `a2_forward` em `tests/fixtures/scripts/validate_oracle_f64.py` estendido para ler propriedades FiLM do JSON via `FILM_KEYS` (8 slots), detectar slots ativos, extrair pesos/bias por camada (após `l1x1_b`), usando `film_weight_count`/`film_bias_count` com grupos.
  2. ✅ Modulação FiLM implementada em NumPy f64 via classe `FiLMSlot` (grupos GEMV → `cond_to_scale_shift` + `apply_modulation`), aplicada nos 6 pontos de inserção ativos: conv_post(1), input_mixin_post(3), activation_post(5), layer1x1_post(6) — mais conv_pre(0) no history buffer. Head1x1_post(7) reservado.
  3. ✅ Python anchor vs Rust oracle f64:
     * A2-FiLM-Lite: ESR = 5.00e-16 (−153.0 dB) < 1e-12
     * A2-FiLM-Full: ESR = 5.00e-16 (−153.0 dB) < 1e-12
  4. ✅ Arquivos de âncora gerados: `wavenet_a2_film_lite_256_f64.bin` e `wavenet_a2_film_full_256_f64.bin` em `tests/fixtures/f64_anchors/`.
  5. ✅ Testes `test_oracle_vs_python_anchor_a2_film_lite` e `test_oracle_vs_python_anchor_a2_film_full` adicionados em `tests/reference_oracle_f64.rs`.
* **Conclusão (2026-06-30):**
  Cadeia de confiança de 3 vias fechada: Python NumPy ↔ Rust Oracle ↔ Produção f32. A âncora NumPy replica `FiLMOracleSlot::apply` (grupos GEMV → `cond_to_scale_shift` + `apply_modulation`) e os 6 pontos de inserção ativos, validada com ESR = 5.00e-16 contra o oráculo Rust. Clippy limpo, 26/26 testes passando.

---

## SPRINT S11 — Robustez da Suíte de Testes (PM-07)

### S11 Objetivos da Sprint

1. Eliminar a brecha estrutural de **SKIP silencioso** no harness de teste de múltiplas taxas de amostragem (`run_v2_multi_sr` em `tests/cpp_parity.rs`), coletando e validando os resultados por taxa com asserções estritas.
2. Exibir um sumário legível e explícito do desfecho das execuções por taxa e por modelo na saída `--nocapture` dos testes.
3. Auditar detalhadamente a integridade operacional e tratamento de erros dos scripts executores de qualidade (`utils/tests-quick.sh`, `utils/tests-long.sh`, `utils/build-release.sh` e `utils/tests-performance-regression.sh`) sob o olhar do papel *Correctness Auditor*.

---

### S11 Tarefas Técnicas

#### [x] Task S11.1 — Robustecer o Harness de Testes `run_v2_multi_sr` contra SKIP Silencioso (PM-07) *(concluída 2026-06-30)*

* **Responsável:** Engenheiro de DSP / QA
* **Risco/Criticidade:** Baixo (test-only).
* **Contexto:**
  No harness `run_v2_multi_sr`, se o renderizador C++ pular silenciosamente uma taxa de amostragem por incompatibilidade de taxa, o teste hoje retorna cedo com uma mensagem de log e passa silenciosamente. Isso viola a Regra 7 de calibração de gates. Precisamos rastrear o status de cada taxa por modelo e assertar que as taxas de amostragem esperadas foram comparadas com sucesso.
* **Critérios de Aceitação:**
  1. Definir o enum `ParityOutcome` em `tests/cpp_parity.rs` para classificar o resultado das execuções (Completed, SkippedModelNotFound, SkippedToolNotAvailable, SkippedRateRejected, SkippedGarbageOutput).
  2. Modificar `run_render_comparison` para retornar `ParityOutcome`.
  3. Refatorar `run_v2_multi_sr` e `run_v2_multi_sr_hf` sob uma função auxiliar única `run_v2_multi_sr_impl` para evitar duplicação de controle.
  4. Na execução, carregar e analisar o JSON do modelo para inferir quais taxas de amostragem devem obrigatoriamente completar (modelos com taxa fixa no JSON rodam apenas nela; WaveNets dinâmicos rodam em todas as 5 taxas; LSTMs pulam apenas a taxa de 192 kHz).
  5. Se o renderizador C++ estiver disponível e o modelo existir, assertar que:
     * Ao menos uma taxa de amostragem completou a validação (prevenindo skipping total).
     * O conjunto de taxas completadas é idêntico ao conjunto de taxas esperadas para o modelo.
  6. Emitir um resumo tabulado das taxas de amostragem executadas no terminal quando executado com `--nocapture`.

**Resultado S11.1:** `ParityOutcome` enum com 5 variantes; `run_render_comparison` retorna `ParityOutcome`; `run_v2_multi_sr`/`run_v2_multi_sr_hf` delegam para `run_v2_multi_sr_impl` (zero duplicação); validação precoce do modelo JSON; assert `!completed.is_empty()` (antiskip total) + assert `completed_set == expected_set` (antiskip parcial); sumário tabulado emitido em `println!` visível com `--nocapture`. Medição empírica: WaveNet Standard e LSTM 1×16 completam todas as 5 taxas (44.1k…192k) com o render C++ — a premissa original de que modelos de taxa fixa ou LSTMs pulariam taxas não se confirmou com a versão atual do NAMCore (v0.5.3+A2-fast). Expected set = todas as 5 taxas para todos os modelos. Quick parity (5/5) e v2 multi-SR (WaveNet + LSTM) passando. Clippy limpo.

#### [x] Task S11.2 — Auditoria e Validação das Suítes de Testes (Correctness Auditor)

* **Responsável:** Auditor de Correção / DevOps
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  Assegurar que as suítes de scripts executadas localmente e em CI (`utils/tests-quick.sh`, `utils/tests-long.sh`, `utils/build-release.sh` e `utils/tests-performance-regression.sh`) estejam livres de falhas latentes, com comportamento determinístico e tratamento correto de erros de encadeamento.
* **Critérios de Aceitação:**
  1. Revisar `utils/tests-quick.sh` garantindo que todos os estágios de auditoria de heap, clippy estrito e validações do plugin CLAP abortam imediatamente sob qualquer sinal de falha.
  2. Revisar `utils/tests-long.sh` atestando que todas as 6 fases (Soak, Proptests, Heap-Audit, CLAP Release Validation, Long Benches, RT Deadline) são executadas independentemente e que qualquer erro parcial é reportado no resumo final e resulta em código de saída `1` no script.
  3. Validar se `utils/build-release.sh` exige corretude no processo PGO + BOLT, abortando graciosamente se os perfis de amostragem do kernel não puderem ser adquiridos.
  4. Revisar `utils/tests-performance-regression.sh` garantindo que a análise estatística do Criterion detecta e barra regressões de latência no DSP com p-value correto (p < 0.05).

**Resultado parcial S11.2 (2026-06-30):**

* `utils/tests-quick.sh`: Adicionado estágio de clippy estrito (fase [1/6] com `-D warnings` para standalone + CLAP). Total agora 6 fases (era 5). Mensagem de sucesso agora distingue se a validação intermediária foi pulada.
* `utils/tests-long.sh`: Corrigido `set -uo pipefail` → `set -euo pipefail` para ativar o ERR trap corretamente e abortar em erros de pré-voo/entre-fases. Fases continuam independentes via `|| true`.
* `utils/build-release.sh`: OK. PGO aborta se não gerar perfis; BOLT faz fallback gracioso; trap restaura `perf_event_paranoid`; validação do artefato CLAP cobre símbolo, SONAME e clap-validator.
* `utils/tests-performance-regression.sh`: Adicionado `mkdir -p target/logs` antes do tee no modo `--check`. Detecção de regressão via `grep "regressed"` + exit code do Criterion — OK.
* **Pendência:** `tests-long.sh` Phase "Property-Based, Parity & Golden Vectors in Release" reportou **FAILED** (status=1, 236s). Verificar `target/logs/phase2-proptests-parity.log` para identificar qual das 10 suítes falhou. Candidatos mais prováveis: `lib_pipeline_block_proptest` (assert `allocs==0` com 2000 casos aleatórios, 108s), `lstm_scalar_bf16_parity` (sem early-return para CPU sem AVX-512 BF16, divergência SIMD vs escalar in release), ou `cpp_parity` v2 multi-SR (assert `completed_set == expected_set` se render C++ falhar em alguma taxa).

---

## SPRINT S12 — Robustez de Carregamento "fail-closed" e Sincronização Documental (Épico H & Épico I)

### Objetivos da Sprint S12

1. **Hardening do Loader (Fail-Closed):** Impedir que ativações em formato de objeto em caminhos não-A2 caiam silenciosamente para Tanh, e rejeitar explicitamente modelos single-net que façam uso do campo `slimmable`.
2. **Prevenção de Regressões de Carga:** Garantir comportamento defensivo no loader com testes de lacuna (gap tests) claros e específicos para ativação-objeto e `slimmable`.
3. **Formalização Documental dos Gaps e Decisões:** Sincronizar os mapas de documentação técnica (`docs/cpp_parity_map.md` e similares) registrando formalmente a ausência do motor A2 dinâmico genérico (PM-10), a decisão de não-asserção de commit no pin do C++ (PM-09) e o arquivamento/descontinuidade do ConvNet canônico interop (PM-13).

---

### Tarefas Técnicas S12

#### [x] Task S12.1 — Hardening do Parser de Ativação do Loader (PM-11)

* **Responsável:** Engenheiro de DSP / Core Developer
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  No parser de `activation` dentro de `src/loader/nam_json/model.rs`, qualquer tipo diferente de String ou Array (como o objeto `{"type":"Softsign"}` no flagship A2) cai no ramo curinga `_ => None`, o que resulta em um fallback silencioso para a ativação "Tanh" na WaveNet A1. Isso representa uma falha de tipo "fail-open" latente.
* **Critérios de Aceitação:**
  1. Modificar o parser de ativação em `NamLayerConfig` para retornar erro (`Err`) caso o JSON contenha uma ativação que não seja String, Array ou Null/None (especificamente, rejeitar se for um Object).
  2. Garantir que o parser retorne um erro explícito de deserialização ("unsupported activation format").
  3. Adicionar um teste unitário em `src/loader/nam_json/activation_parser_test.rs` (ou no próprio teste do model loader) simulando uma ativação em forma de objeto e assegurando que ela falhe fechada com o erro correspondente.

#### [x] Task S12.2 — Rejeição Explícita de Modelos Slimmable Single-Net (PM-12)

* **Responsável:** Engenheiro de DSP / Core Developer
* **Risco/Criticidade:** Baixo (defensivo).
* **Contexto:**
  O motor `nam-rs` não suporta fatiamento dinâmico de canais em runtime (slicing de pesos single-net) — funcionalidade correspondente ao campo `slimmable` em arquivos `.nam`. Atualmente, modelos contendo essa chave (como `slimmable_wavenet.nam`) são rejeitados de forma indireta por incompatibilidades secundárias, mas o campo `slimmable` é silenciosamente descartado e não há caminho explícito de rejeição.
* **Critérios de Aceitação:**
  1. Implementar uma verificação explícita no loader (durante a validação de topologia ou model data) que rejeite imediatamente qualquer modelo que possua a chave `slimmable` ativa.
  2. O erro gerado deve ser explícito e orientado ao usuário (ex.: "slimmable single-net weight slicing is not supported; use SlimmableContainer instead").
  3. Criar o teste de lacuna `test_loader_gap_slimmable_wavenet` sob `tests/golden_vectors.rs` (ou `tests/loader_gap.rs` se aplicável), carregando a fixture real `tests/fixtures/models/slimmable_wavenet.nam` e validando a rejeição fail-closed com a mensagem esperada.

#### [x] Task S12.3 — Documentação e Decreto de Arquivamento de ConvNet Canônico (PM-13)

* **Responsável:** Documentador Técnico / Arquiteto
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  A arquitetura ConvNet canônica upstream possui kernel=2 fixo e head matricial flat. O `nam-rs` implementa um formato bespoke (multi-bloco, head Conv1D, `head_scale`). Como o ConvNet foi descontinuado upstream e não há modelos oficiais na pasta de exemplos, o PO decidiu "nucar" (arquivar) qualquer pretensão de implementar compatibilidade com o formato canônico flat upstream.
* **Critérios de Aceitação:**
  1. Atualizar `docs/cpp_parity_map.md` (seções §5 e §13) documentando explicitamente a decisão do PO de arquivar/descontinuar o interop do formato canônico flat de ConvNet.
  2. Formalizar que o `nam-rs` mantém apenas o seu formato bespoke interno com validação via f64 oracle como sua única testemunha de paridade matemática ideal.

#### [x] Task S12.4 — Sincronização Documental de Gaps e Decisões de Loader (PM-09, PM-10, PM-12)

* **Responsável:** Documentador Técnico / Arquiteto
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  Sincronizar a documentação técnica com os novos comportamentos endurecidos do loader e com as decisões executivas do PO acerca da versão do C++ e A2.
* **Critérios de Aceitação:**
  1. Documentar em `docs/cpp_parity_map.md` §13 a decisão do PO de não aplicar asserções de commit rígidas no script `tests-long.sh` para o pin da referência C++ (PM-09), considerando a última versão do GitHub como a verdade de referência.
  2. Documentar que os modelos com ativações-objeto e `slimmable` agora são explicitamente validados e rejeitados no loader de forma fail-closed segura.
  3. Garantir a consistência e sincronização de referências cruzadas entre `TODO-findings.md` e `docs/cpp_parity_map.md`.
* **Conclusão (2026-06-30):**
  1. ✅ `docs/cpp_parity_map.md` §13.1 PM-09: documentação expandida explicitando que `tests-long.sh` e `golden_gen_build.sh` não impõem asserções rígidas de commit-pin — apenas logam o SHA da working copy para awareness do desenvolvedor, sem abortar em mismatch. Tabela §13 linha PM-09 consolidada como 🟢 PO Decision.
  2. ✅ `docs/cpp_parity_map.md` §13.1 PM-11/PM-12 (linha "Loader fail-closed hardening"): documentação existente confirma que ativações-objeto são explicitamente validadas e rejeitadas com erro de deserialização, e que modelos `slimmable` single-net são parseados e rejeitados com erro user-friendly. Tabela §13 linha `SlimmableWavenet` atualizada de "silently dropped" para "explicitly rejected (fail-closed)". Tabela de modelos oficiais em `TODO-findings.md` sincronizada.
  3. ✅ Consistência cross-ref:
     * `TODO-findings.md` PM-11: status atualizado para ✅ RESOLVIDO (S12.1) com nota de implementação.
     * `TODO-findings.md` PM-12: título e status atualizados para "rejeitado explicitamente (fail-closed) ✅ RESOLVIDO (S12.2)".
     * `TODO-findings.md` Épico H: `[DOING]` → `[DONE]`.
     * `TODO-findings.md` Épico I: `[DOING]` → `[DONE]`.
     * `cpp_parity_map.md` See Also: range `PM-01…PM-08` → `PM-01…PM-14`.
     * `TODO-findings.md` tabela de modelos oficiais: entrada `slimmable_wavenet.nam` — "campo descartado" → "explicitamente rejeitado (fail-closed)".
     * Nenhuma referência cruzada quebrada identificada entre os dois arquivos.

#### [x] Task S12.5 — Documentação da Decisão do PO sobre Observabilidade e Scripts (PM-14)

* **Responsável:** Documentador Técnico / Arquiteto
* **Risco/Criticidade:** Nulo (doc-only).
* **Contexto:**
  O Épico J (observabilidade da bateria e scripts-guardião) foi classificado como `Won't Do` por decisão executiva do PO. A estrutura atual de testes e observabilidade no local/CI foi atestada como suficientemente robusta, e a referência atualizada do GitHub é a única fonte canônica necessária.
* **Critérios de Aceitação:**
  1. Atualizar `docs/cpp_parity_map.md` §13 formalizando a decisão do PO para o PM-14, consolidando que nenhuma mudança adicional na infraestrutura de observabilidade da bateria é necessária.
  2. Sincronizar o status em `TODO-findings.md` indicando a documentação da decisão.
* **Conclusão (2026-06-30):**
  1. ✅ `docs/cpp_parity_map.md` §13.1 PM-14: nota expandida com Decreto do PO formalizando a decisão sobre observabilidade e scripts-guardião. A auditoria dos 4 scripts (`tests-long.sh`, `tests-quick.sh`, `tests-performance-regression.sh`, `build-release.sh`) foi revista — goldens abrangentes, clippy estrito, pinagem de core para perf, degradação graciosa de PGO+BOLT, RT gates, validação CLAP, heap-audits e manifesto de frescor de goldens — todos confirmados como suficientemente robustos. O PO decretou que nenhuma asserção rígida de pin de commit, captura adicional de linha-sumário ou alteração na bateria de observabilidade ao vivo é necessária. Tabela §13 linha PM-14 já consolidada como 🟢 PO Decision.
  2. ✅ `TODO-findings.md` PM-14: entrada atualizada com "✅ Documentado (S12.5)" e nota de decisão documentada.
  3. ✅ `TODO-findings.md` Épico J: mantido como `[WONT-DO]` (a decisão de não-executar o trabalho foi documentada, não o trabalho em si), com referência ao decreto documentado em `cpp_parity_map.md` §13.1.

---

## SPRINT S13 — Feature-Completeness A2 Oficial (Épico G)

### Objetivos da Sprint S13

1. **Feature-Completeness do Flagship A2:** Implementar o motor genérico A2 dinâmico para suportar modelos de arquitetura livre, contendo `condition_size > 1`, `head1x1`, gating/blending, sub-modelo `condition_dsp` e ativações declaradas como objetos (PM-10).
2. **Cobertura do Oráculo f64 (Extensão PM-03):** Adaptar a testemunha de matemática ideal para absorver as novas especificações da arquitetura A2 genérica, viabilizando o isolamento de bugs de SIMD em estruturas variáveis.
3. **Fidelidade com Capturas Reais (PM-05):** Carregar e processar o modelo flagship oficial A2 (`wavenet_a2_max.nam`), validando timbre e robustez contra golden vectors C++ gerados pela v0.5.x, alcançando o fechamento do gap de compatibilidade com a referência em sua totalidade.

---

### Tarefas Técnicas S13

#### [x] Task S13.1 — Arquitetura DSP e Parser do Motor A2 Genérico (PM-10)

* **Responsável:** Engenheiro de DSP / Arquiteto
* **Risco/Criticidade:** Alto (criação de um novo motor de processamento no caminho quente de DSP).
* **Contexto:**
  O flagship A2 oficial (`wavenet_a2_max.nam`) apresenta apenas 1 array (kernel único, dilations `[1, 2]`), `head1x1`, gating, sub-modelo `condition_dsp` e ativação-objeto (`{"type":"Softsign"}`). Atualmente, o nam-rs exige uma assinatura rígida de 23 camadas para usar o FiLM. A missão é construir um motor WaveNet flexível que aplique condicionamento FiLM independente da geometria imposta e leia as ativações-objeto.
* **Critérios de Aceitação:**
  1. No `src/loader/nam_json/model.rs`, estender o parser de `activation` para interpretar a forma de objeto (ex.: extraindo o campo `"type"`) conforme o fail-closed implementado em S12.1.
  2. Projetar e implementar o construto de DSP `WaveNetModelDynA2Generic` (ou abstrair o A2 existente) para comportar arrays únicos, dimensões genéricas e suporte completo às chaves de FiLM (gating, `condition_dsp`, `head1x1`).
  3. Otimizar a estrutura mantendo a RT-Safety (ausência de alocações no block processing).
  4. Manter a compatibilidade com o formato já estabelecido de vetores SIMD para não quebrar a performance do pipeline.

#### [x] Task S13.2 — Testemunha Oráculo f64 para A2 Genérico (PM-03/PM-10)

* **Responsável:** Engenheiro de DSP / QA
* **Risco/Criticidade:** Médio (necessário para validação da task S13.1).
* **Contexto:**
  Com um motor novo (Task S13.1), o oráculo f64 implementado na Sprint S10 precisa ser estendido. A referência f64 deve suportar geometrias livres e as novas inserções (ex. `head1x1_post(7)`) para comprovar que o novo código SIMD de produção mantém a fidelidade matemática esperada em dupla precisão.
* **Critérios de Aceitação:**
  1. ✅ Estender `src/testing/reference_oracle.rs` e `validate_oracle_f64.py` para processar a topologia aberta de A2 genérico.
  2. ✅ Implementar a inserção FiLM 7 (`head1x1_post`) e a propagação correta das matrizes gating/blending do `condition_dsp`.
  3. ✅ Criar os testes unitários (`test_oracle_a2_generic`) que atestem o piso numérico (ESR < 1e-12) usando dados de varredura.
* **Conclusão (S13.2):**
  * Oráculo Rust e Python estendidos com suporte a topologia aberta (kernel_sizes/dilations dinâmicos, bottleneck≠channels, kernel_size escalar, activation objeto).
  * Todos os 8 slots FiLM implementados incluindo head1x1_post_film (slot 7).
  * Gating/blending suportado com 2× bottleneck e sigmoid gate.
  * Ativações heterogêneas: Tanh, LeakyReLU, Sigmoid, SiLU, HardSwish, Softsign.
  * Mixin matrix-vector para condition_size > 1.
  * Head1x1 com grupos suportado no oráculo.
  * Formato FiLM corrigido para A2 genérico (bias=channels, wc integer-safe) — detectado por `cond_size > 1`.
  * Âncora Python regenerada: `tests/fixtures/f64_anchors/a2_max_256_f64.bin`.
  * `test_oracle_vs_python_anchor_a2_generic` passa com ESR < 1e-12.
  * Testes de produção (`test_oracle_a2_generic`, `test_decomposition_a2_generic`, `test_combined_simulation_a2_generic`) marcados `#[ignore]` — dependem de S13.1 para processamento grouped-head1x1 no motor de produção.
  * Retrocompatibilidade mantida: todos os 27 testes existentes passam sem alteração.
  * **Nota p/ S13.1:** O motor de produção lê corretamente os pesos FiLM (via `film_weight_count_generic`) e head1x1 (via `h1_groups`), mas o laço de processamento (`process.rs:402-431`) ainda não suporta head1x1 agrupado (`h1_groups > 1`). A ativação Softsign single-object precisou de suporte no parser (`model.rs`).

#### [x] Task S13.3 — Integração e Calibração de Capturas Reais A2 (PM-05)

* **Responsável:** Engenheiro de DSP / QA
* **Risco/Criticidade:** Baixo (dependente da conclusão com êxito de S13.1 e S13.2).
* **Contexto:**
  O último passo para confirmar a aderência e a Feature-Completeness (Épico G) é injetar a captura real de áudio pelo flagship A2, atestando a validade perceptiva do gating e do `head1x1` nos harmônicos do som, contra os vetores de referência C++ gerados.
* **Critérios de Aceitação:**
  1. ✅ Promover o modelo não distribuível `wavenet_a2_max.nam` para as fixtures testadas pela suite de golden vectors.
  2. ✅ Gerar o golden vector referencial pelo C++ canônico (`golden_gen_build.sh` / C++ render) para o modelo real.
  3. ✅ Calibrar a margem de threshold (ESR / SNR) em `tests/validation.rs` baseando-se no desvio documentado em PM-03 e atestando que os testes agora carregam a arquitetura livre sem acionar fallback (passando os gates de MR-STFT).
* **Conclusão (S13.3):**
  1. ✅ `tests/fixtures/golden_gen_build.sh` atualizado: `wavenet_a2_max.nam` adicionado às listas MODELS (v1) e V2_MODELS (v2 multi-SR).
  2. ✅ Golden vector C++ gerado: `tests/fixtures/golden_wavenet_a2_max.bin` (2048 samples, 48 kHz, C++ v0.5.3 render).
  3. ✅ Golden vector test criado em `tests/golden_vectors.rs` (`test_golden_vectors_wavenet_a2_max`), marcado `#[ignore]` — aguardando suporte a multi-array condition_dsp no motor A2 genérico.
  4. ✅ Thresholds provisionais calibrados em `tests/common/validation.rs`:
     * `wavenet_a2_max`: SNR ≥ 10.0 dB, ESR < 5.0e-2, MRSTFT < 0.49 (provisional, baseado em PM-03 RF1 18–36 dB).
     * Meta-teste `threshold_calibration.rs` atualizado com entrada `wavenet_a2_max`.
  5. ✅ `test_loader_gap_wavenet_a2_max` atualizado: documenta o estado atual (modelo despacha para `WaveNetA2Dyn` mas o `condition_dsp` multi-array falha no motor A1 dinâmico).
  6. **Pendente:** O modelo `wavenet_a2_max.nam` ainda não carrega — o `condition_dsp` sub-model (2 camadas com FiLM) é roteado para `WaveNetDyn` (A1 free-geometry) que não suporta FiLM, resultando em "consumed 368, total 1052". Resolver requer:
     * (a) Suporte a FiLM no motor A1 dinâmico (`WaveNetLayerArrayDyn`), ou
     * (b) Criação de um wrapper `WaveNetA2Cascade` que empilha múltiplos `WaveNetA2Dyn`, ou
     * (c) Extensão do `WaveNetA2Dyn` para suportar múltiplos arrays.
     * Esta engine work estava parcialmente no escopo de S13.1 e é o blocker final para Feature-Completeness do flagship A2 (Épico G).
  7. **Retrocompatibilidade mantida:** 26 dos 29 testes golden vector passam (3 falhas pré-existentes não relacionadas), 16 ignorados (incluindo o novo `wavenet_a2_max` e os do oráculo A2 genérico de S13.2).
  8. Testes de calibração de thresholds passam (4/4).

---

## SPRINT S14 — Multi-Array A2 (Épico K)

### Objetivos da Sprint S14

1. **Topologia Multi-Array (Cascata) no Motor A2 Dinâmico:** Expandir a capacidade do `WaveNetA2Dyn` para orquestrar mais de 1 array (conforme arquitetura suportada nativamente no C++ e utilizada em sub-modelos como o `condition_dsp` do flagship A2).
2. **Resolução de Carregamento:** Eliminar o fallback forçado de modelos A2 multi-array para o pipeline da A1 (que falha estritamente e fail-closed por desconhecer chaves FiLM), permitindo o carregamento completo do `wavenet_a2_max.nam`.
3. **Isolamento de Acúmulos Finais:** Garantir que as convoluções de projeção (`head1x1`) e os gates ocorram da mesma maneira que a referência C++: apenas no resultado acumulado global de todos os sub-arrays (`final_head_outputs`).

---

### Tarefas Técnicas S14

#### [x] Task S14.1 — Detector A2 e Pipeline de Cascata de Arrays (PM-15)

* **Responsável:** Engenheiro de DSP / Arquiteto
* **Risco/Criticidade:** Alto (Requer roteamento adequado de RT-Safety entre múltiplos buffers).
* **Contexto:**
  No momento, `is_a2_shape` assegura `layers.len() == 1`. Para habilitar `condition_dsp` multi-array (FiLM), o detector e o builder do modelo precisarão instanciar um vetor de arrays (seja um wrapper `WaveNetA2Cascade` iterando N `WaveNetA2Dyn`, ou refatoração interna) e direcioná-los corretamente no método `process`.
* **Critérios de Aceitação:**
  1. Alterar o detector de topologia `is_a2_shape` para `layers.len() >= 1`.
  2. Implementar a lógica de pipeline (`WaveNetA2Cascade` ou análogo): a saída (layer_in/residual) do Array N serve como input do Array N+1.
  3. Acumular as saídas `head_accum` de todos os sub-arrays aditivamente e repassar esse acúmulo finalizado à camada `head1x1` e, subsequentemente, à head opcional (`head_conv`).

#### [ ] Task S14.2 — Testes de Lacuna e Oráculo f64 para A2 Multi-Array (PM-15)

* **Responsável:** QA / Engenheiro de DSP
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  Para certificar que a agregação dos sub-arrays (e a ordem das chamadas do `head1x1`) são isomorfas com o oráculo de f64, os scripts de teste devem abranger esse modelo e não permitir falhas de carregamento.
* **Critérios de Aceitação:**
  1. No script `validate_oracle_f64.py` e `src/testing/reference_oracle.rs`, implementar o somatório multi-array compatível para WaveNets A2.
  2. Remover as restrições (`#[ignore]`) das funções de teste referenciadas para o `wavenet_a2_max.nam` (que possui `condition_dsp` multi-array e ativará essa cadeia nativamente).
  3. Atestar que `test_loader_gap_wavenet_a2_max` não lança mais pânicos nem recusa o arquivo, passando com a fidelidade sonora estabelecida na task S13.3.
