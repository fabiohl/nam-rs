<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Roadmap de Sprints — Épicos A, B, C & D

Este documento organiza o planejamento ágil e tarefas técnicas para o **Épico A (PM-01, PM-02, PM-08 — Sincronização Documental de Paridade)**, o **Épico B (PM-04, PM-03 — Testemunhas Independentes/Oráculo f64)**, o **Épico C (PM-05 — Cobertura de Modelos Reais A2-FiLM)** e o **Épico D (PM-06 — SlimmableWavenet)** no `nam-rs`, com base nas descobertas consolidadas em `TODO-findings.md`.

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

#### [ ] Task S10.3 — Extensão do Oráculo f64 para FiLM A2 (PM-03)

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

#### [ ] Task S10.4 — Âncora NumPy e Geração de Fixtures FiLM A2 (PM-03)

* **Responsável:** Engenheiro de DSP / Python Integrator
* **Risco/Criticidade:** Baixo.
* **Contexto:**
  Complementar a cadeia de confiança gerando a âncora independente NumPy para o caso FiLM A2.
* **Critérios de Aceitação:**
  1. Estender `a2_forward` em `tests/fixtures/scripts/validate_oracle_f64.py` para ler as propriedades FiLM do JSON e extrair seus pesos.
  2. Implementar a modulação FiLM em NumPy f64 nas posições correspondentes da rede.
  3. Validar a âncora Python contra o oráculo Rust f64 com ESR < 1e-12.
  4. Gerar os arquivos de âncora correspondentes (ex: `wavenet_a2_film_lite_256_f64.bin`) em `tests/fixtures/f64_anchors/`.
  5. Adicionar `test_oracle_vs_python_anchor_a2_film` em `tests/reference_oracle_f64.rs` assegurando o alinhamento de 3 vias.
