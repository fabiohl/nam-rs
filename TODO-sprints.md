<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints

Este documento gerencia o backlog de sprints baseado nos achados mapeados em `TODO-findings.md`. Ele define as tarefas técnicas, as responsabilidades e os critérios de aceitação para a implementação segura e incremental das melhorias do projeto.

---

## Eixo e Épico Correspondente

* **Épico**: [Épico E1 — "Nenhum teste verde sem evidência" (rigor da malha)](file:///home/fabio/nam-rs/TODO-findings.md#L644)
* **Objetivo Geral**: Eliminar falsos-verdes da malha de paridade C++ e endurecer as asserções de contrato de incompatibilidade, determinismo e instrumentação de status nos scripts de QA.
* **Complexidade/Risco**: Baixo (focado estritamente na infraestrutura de testes e scripts de validação).

---

## Sprint 1 — Rigor da Malha de Testes (Épico E1)

### Metas da Sprint (Sprint Goals)

1. **Assegurar a integridade da paridade v1/v1_hf**: Garantir que skips silenciosos do toolchain C++ não disfarcem falhas de validação, estabelecendo asserções reais ou marcas claras de skip-coverage.
2. **Prevenir retrocessos de descarte**: Blindar a malha de testes com um meta-teste estrutural que impeça a introdução de novos descartes de `ParityOutcome`.
3. **Formalizar contratos de incompatibilidade**: Converter testes de modelos sabidamente incompatíveis (ConvNet antes de F-A1) em verificações estritas de erro de formato em vez de placebos.
4. **Visibilidade real no Nightly/Long Loop**: Atualizar o gerador de relatórios do `tests-long.sh` para refletir status `SKIPPED` reais e alertar sobre execuções suspeitas.
5. **Elevar o determinismo**: Garantir determinismo exato de bit (MSE == 0.0) para ConvNet e documentar os limites conhecidos do oráculo f64.

### Papéis Necessários (Roles Needed)

* **Engenheiro de Software DSP / Rust**: Edição dos arquivos `cpp_parity.rs`, `threshold_calibration.rs`, e `golden_vectors.rs`.
* **Engenheiro de Infraestrutura de QA / DevOps**: Edição dos scripts Bash `tests-long.sh` e `_lib.sh`.

---

### Detalhamento das Tarefas Técnicas (Technical Tasks)

#### Tarefa T1.1 — Correção de run_v1/run_v1_hf e Meta-teste anti-let_ (F-B1)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
  * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs)
* **Descrição**:
  1. Alterar as funções auxiliares `run_v1` e `run_v1_hf` para que elas **retornem** o `ParityOutcome` em vez de descartá-lo via `let _ = ...`.
  2. Atualizar as asserções nos testes rápidos rápidos que rodam no `tests-quick.sh`:
     * `quick_parity_lstm_1x16`
     * `quick_parity_wavenet_ch16`
     * `quick_parity_a2_full`
     Se o renderizador C++ estiver disponível (compilado/compilável), **exigir** `ParityOutcome::Completed` via `assert_eq!(outcome, ParityOutcome::Completed)`. Caso contrário, permitir skip mas imprimir a mensagem padronizada no stderr: `SKIP-COVERAGE: <nome_do_teste>`.
  3. Atualizar os testes `live_cross_validation_*` marcados com `#[ignore]` (que rodam no `tests-long.sh` onde o toolchain C++ é pré-requisito obrigatório verificado na Fase 0) para exigirem `ParityOutcome::Completed` incondicionalmente.
  4. Adicionar um teste meta-estrutural em `threshold_calibration.rs` que analise o código-fonte de `cpp_parity.rs` e falhe se qualquer wrapper usar descarte silencioso (`let _ = run_render_comparison`).
* **Critério de Aceitação (DoD)**:
  * Compilação limpa sem warnings.
  * O meta-teste estrutural passa com sucesso.
  * A execução dos testes de paridade reflete a cobertura real sem falsos-verdes.

---

#### Tarefa T1.2 — Contrato de Incompatibilidade para quick_parity_convnet (F-B2)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
* **Descrição**:
  1. O modelo `convnet_test.nam` é arquiteturalmente incompatível com o renderizador C++ atual (conforme documentado em `generate_b1_2_fixtures.py`). Hoje o teste simplesmente faz skip silencioso e passa.
  2. Substituir `quick_parity_convnet` por `quick_parity_convnet_expected_incompatible`.
  3. O teste deve asseverar que o retorno de `run_render_comparison` é exatamente `ParityOutcome::SkippedGarbageOutput` ou `SkippedRateRejected` (ou que gera falha com código de saída diferente de 0) e que o log do erro contém a mensagem esperada da biblioteca `nlohmann/json` indicando a incompatibilidade.
  4. Se futuramente F-A1 for implementada (Opção A), esse teste voltará a ser de paridade real.
* **Critério de Aceitação (DoD)**:
  * O teste deixa de ser um skip placebo e passa a verificar ativamente o contrato de falha de incompatibilidade arquitetural de formato de dados.

---

#### Tarefa T1.3 — Status SKIPPED e Alertas no tests-long.sh (F-C6)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh)
* **Descrição**:
  1. Alterar a função `run_pipewire_phase` para retornar o código de status `77` (convenção POSIX para skip) quando o daemon do PipeWire estiver indisponível.
  2. Atualizar a função central `run_phase` para rastrear esse status especial: se o comando retornar `77`, registrar no array de status a string `SKIPPED` em vez de `PASSED` ou `FAILED`.
  3. No relatório final do summary, formatar `SKIPPED` em cor amarela (usando `YELLOW` do `_lib.sh`).
  4. Implementar um guard-rail em `run_phase`: se o status gravado for `PASSED` mas a duração total da fase for `< 1` segundo, emitir um aviso em destaque no terminal sinalizando uma fase possivelmente vazia/falsa-verde.
* **Critério de Aceitação (DoD)**:
  * Rodar o `tests-long.sh` [Feito pelo DEV humano] localmente (com ou sem PipeWire ativo) e comprovar que a fase correspondente registra `SKIPPED` corretamente na tabela final ou executa normalmente e passa.
  * Verificação de avisos de duração funcional.

---

#### Tarefa T1.4 — Endurecimento de Determinismo do ConvNet (F-B7)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs)
* **Descrição**:
  1. Em `tests/models/golden_vectors.rs`, o teste de determinismo do ConvNet compara duas execuções idênticas com tolerância de erro absoluto inferior a `1e-6`.
  2. Modificar o laço de asserção para exigir exatidão bitwise absoluta (`MSE == 0.0` ou diferença absoluta igual a `0.0`), alinhando o ConvNet com as exigências dos demais modelos.
  3. Investigar imediatamente qualquer divergência caso ocorra falha (eliminando potenciais fontes de acumuladores indeterministas na implementação da rede).
* **Critério de Aceitação (DoD)**:
  * Execução de `cargo test --test golden_vectors` passa com sucesso em modo release com tolerância zero.

  **Conclusão (2026-07-12)**: Assertiva alterada de `abs < 1e-6` para `abs == 0.0`. Teste `test_golden_vectors_convnet_test` passa em modo release sem divergências bitwise. O ConvNet já é deterministicamente exato; a tolerância anterior era excessivamente conservadora.

---

#### Tarefa T1.5 — Documentação do Piso f64 das Âncoras (F-B8)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md)
* **Descrição**:
  1. Documentar na seção do Oráculo de f64 em `docs/perceptual_validation.md` que o valor residual medido nas âncoras NumPy de aproximadamente `5.00e-16` para WaveNet, ConvNet e A2 é o limite teórico natural para a precisão f64 acumulada na janela de validação (≈ 2 × epsilon de precisão dupla).
  2. Esclarecer que o valor menor observado em LSTMs (≈ `3.49e-30`) decorre de caminhos matemáticos quase idênticos de bit e maior energia.
  3. Isso evitará suspeitas de clamps artificiais por futuros desenvolvedores ou auditores.
* **Critério de Aceitação (DoD)**:
  * Documentação atualizada e clara na seção de validação perceptual.

  **Conclusão (2026-07-12)**: Subseção "NumPy Anchor f64 Residual Floor" adicionada na seção do f64 Reference Oracle em `docs/perceptual_validation.md`, documentando o piso f64 teórico de ~5e-16 para feed-forward e ~3.49e-30 para LSTM, esclarecendo que `compute_esr_f64` não possui clamp/floor artificial.

---

## Critérios de Aceitação Gerais da Sprint (Definition of Done)

Para fechar o Épico E1 em definitivo:

1. `utils/lints.sh` executa sem nenhum erro de formatação, clippy, ou cabeçalho SPDX.
2. `utils/tests-quick.sh` roda com sucesso na Fase 1 (estrutural/debug) e Fase 2 (rápida de paridade/release).
3. Todas as modificações em arquivos de código Rust contêm a licença de Copyright no cabeçalho.
4. Nenhuma alteração adicionou código não real-time safe nas rotas DSP críticas.
