<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil e Tarefas Técnicas

> **Origem:** Gerado pela skill `planejador-arquiteto` com base nas constatações e proposta de solução de [`TODO-convnet_parity.md`](TODO-convnet_parity.md).
> **Foco:** Paridade estrita de inicialização de estado ConvNet ↔ NAMcore (resolução da lacuna ESR 2.54e-5 de transiente de reset).

---

## Visão Geral & Contexto Estratégico

A auditoria de paridade de 2026-07-27 (registrada em `TODO-convnet_parity.md`) diagnosticou que a divergência de 2.54e-5 (SNR 45.9 dB) entre a arquitetura ConvNet de produção do `nam-rs` e o golden do `NAMcore` é **exclusivamente um transiente de inicialização de estado (prewarm)** confinado às primeiras 62 amostras. Em regime permanente, os dois motores já apresentam concordância de 4.2e-9 (piso f32 do WAV golden).

Este plano organiza a resolução técnica em 3 Sprints ágeis contendo Tarefas Técnicas atômicas, prontas para execução futura pelos especialistas (`implementador`, `revisor-auditor`, `documentador`).

---

## Épico 1: Paridade Estrita ConvNet ↔ NAMcore (Prewarm State Sync)

### Resumo Executivo das Sprints

| Sprint       | Foco Principal                                                                                                       | Risco                                                                  | Esforço Estimado              |
|:------------ |:-------------------------------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------- |:----------------------------- |
| **Sprint 1** | Correção do Motor DSP (`model.rs`), remoção de código morto (`block.rs`) e teste de invariante                       | **BAIXO** (escrito e isolado em caminho frio)                          | ~15 linhas de código          |
| **Sprint 2** | Execução de testes de paridade (`quick_parity_convnet`), recalibração de gates e validação ágil (`tests-quick.sh`)   | **MÉDIO** (recalibração de threshold exige validação numérica precisa) | Re-execução e ajuste de gates |
| **Sprint 3** | Sincronização da documentação (`cpp_parity_map.md`), atualização do contrato (`quality-contract.txt`) e encerramento | **MÍNIMO**                                                             | Atualização de docs e scripts |

---

### Sprint 1: Correção do Motor DSP & Limpeza de Código Morto

#### `TASK-CONVNET-01`: Atualizar `ConvNetModel::prewarm` para propagação de silêncio [DONE]

- **Arquivos:** [`src/models/convnet/model.rs`](src/models/convnet/model.rs) e [`src/models/convnet/mod.rs`](src/models/convnet/mod.rs)
- **Risco:** **BAIXO** (caminho frio executado apenas em `reset()` / inicialização).
- **Referência:** `TODO-convnet_parity.md` § "Correção proposta", linhas 137–152.
- **Descrição:**
  1. Substituir a implementação de `prewarm()` / `prewarm_internal()` em `ConvNetModel` (que populava zeros literais por bloco isolado) pela réplica exata da semântica `nam::DSP::prewarm()` do NAMcore (`dsp.cpp:67-96`).
  2. A função deve criar um buffer de $N = \text{receptive\_field\_size} + 1 = 64$ amostras de silêncio (`0.0f32`) e processá-lo integralmente via `self.process(&zeros, &mut sink)`, descartando a saída e retendo o estado final nos ring buffers.
  3. Atualizar a implementação do trait `prewarm_samples(&self) -> usize` em `src/models/convnet/mod.rs` para retornar `self.receptive_field_size + 1` (64 amostras).
- **Critério de Aceite:** `model.prewarm()` compila sem warnings e popula o estado estacionário de entrada zero em todos os blocos `ConvNetBlock`.

#### `TASK-CONVNET-02`: Eliminar `ConvNetBlock::prewarm` descontinuado (Código Morto) [DONE]

- **Arquivos:** [`src/models/convnet/block.rs`](src/models/convnet/block.rs)
- **Risco:** **MÍNIMO**.
- **Referência:** `TODO-convnet_parity.md` § "Correção proposta", linha 157.
- **Descrição:**
  1. Remover as funções `pub fn prewarm(&mut self)` e `pub unsafe fn prewarm_internal<M: SimdMath>(&mut self)` de `ConvNetBlock`, uma vez que o prewarm agora é operado em nível de modelo via `self.process()`.
  2. Ajustar `convnet_block_test.rs` se houver chamadas diretas a esse método descontinuado.
- **Critério de Aceite:** Ausência de código morto no módulo de bloco; compilação limpa sem warnings.

#### `TASK-CONVNET-03`: Adicionar Teste Unitário de Invariante de Ponto Fixo Estacionário [DONE]

- **Arquivos:** [`src/models/convnet/convnet_model_test.rs`](src/models/convnet/convnet_model_test.rs)
- **Risco:** **BAIXO**.
- **Referência:** `TODO-convnet_parity.md` § "Plano de execução" (item 4).
- **Descrição:**
  1. Criar o teste `test_convnet_prewarm_fixed_point_invariant()`.
  2. O teste deve instanciar um `ConvNetModel`, chamar `prewarm()`, e em seguida processar 64 amostras de zeros, verificando que a saída é um sinal constante de DC estacionário idêntico à saída produzida por convergência explícita (sem saltos ou transientes adicionais pós-prewarm).
- **Critério de Aceite:** `cargo test --lib models::convnet::convnet_model_test` executado com 100% de aprovação.

---

### Sprint 2: Validação Paritária & Recalibração de Gates de Qualidade

#### `TASK-CONVNET-04`: Medição Empírica das Métricas Paritárias ConvNet ↔ Golden C++ [DONE]

- **Arquivos:** Suíte de testes `tests/parity/cpp_parity.rs` e `tests/parity/golden_vectors.rs`
- **Risco:** **CRÍTICO / MÉDIO** (ponto de validação da hipótese analítica).
- **Referência:** `TODO-convnet_parity.md` § "Plano de execução" (item 2).
- **Descrição:**
  1. Executar os testes de paridade:
     - `cargo test --test cpp_parity quick_parity_convnet -- --nocapture`
     - `cargo test --test cpp_parity live_cross_validation_convnet -- --nocapture`
     - `cargo test --test golden_vectors test_golden_vectors_convnet_test -- --nocapture`
     - `cargo test --test reference_oracle_f64 -- --nocapture`
  2. Coletar os valores exatos medidos de **SNR (dB)**, **ESR** e **MR-STFT**.
  3. Confirmar a queda do ESR da faixa de $2.54 \times 10^{-5}$ para o piso real da família ConvNet ($\approx 10^{-14} \dots 10^{-15}$, SNR $\ge 130$ dB).
- **Critério de Aceite:** Registro documentado das métricas reais em log e confirmação de concordância com o oráculo NumPy.

#### `TASK-CONVNET-05`: Recalibração dos Gates de Qualidade em `validation.rs` e `cpp_parity.rs` [DONE]

- **Arquivos:** [`tests/common/validation.rs`](tests/common/validation.rs) e [`tests/parity/cpp_parity.rs`](tests/parity/cpp_parity.rs)
- **Risco:** **MÉDIO** (alteração dos critérios de CI/CD).
- **Referência:** `TODO-convnet_parity.md` § "Plano de execução" (item 3).
- **Descrição:**
  1. Em `tests/common/validation.rs` (entrada `"convnet_test"`): recalibrar a tolerância de SNR 35 dB / ESR 1e-4 / MR-STFT 0.03 para os patamares paritários reais (**SNR ≥ 120 dB, ESR ≤ 1e-12, MR-STFT ≤ 1e-4**).
  2. Em `tests/parity/cpp_parity.rs`: atualizar `ABSOLUTE_ESR_CAP_CONVNET_HF` de `1e-3` para **`1e-10`**, alinhando ao cap da família WaveNet.
- **Critério de Aceite:** Todos os testes de paridade passam com os novos gates estritos ativados.

#### `TASK-CONVNET-06`: Validação Integrada da Suíte Ágil com `utils/tests-quick.sh` [DONE]

- **Arquivos:** `utils/tests-quick.sh`
- **Risco:** **MÉDIO**.
- **Referência:** Regras de IA `.agents/rules/testing.md` (item 2).
- **Descrição:**
  1. Executar `utils/tests-quick.sh` uma única vez como validação final do ciclo de desenvolvimento.
- **Critério de Aceite:** Suíte rápida de integração e linting concluída sem falhas ou regressões.

---

### Sprint 3: Documentação & Atualização de Contratos de Qualidade

#### `TASK-CONVNET-07`: Sincronização da Tabela de Paridade em `docs/cpp_parity_map.md` [DONE]

- **Arquivos:** [`docs/cpp_parity_map.md`](docs/cpp_parity_map.md)
- **Risco:** **MÍNIMO**.
- **Referência:** `TODO-convnet_parity.md` § "Plano de execução" (item 5).
- **Descrição:**
  1. Atualizar o status da arquitetura ConvNet na documentação de paridade C++, movendo-a da categoria "Divergência Calibrada" para **"IDÊNTICO (Paridade Total de Inicialização e Aritmética)"**.
  2. Registrar as métricas definitivas alcançadas pós-Sprint 2.
- **Critério de Aceite:** `docs/cpp_parity_map.md` reflete o estado paritário exato do modelo.

#### `TASK-CONVNET-08`: Atualização do Contrato de Qualidade via `quality-dashboard.sh`

- **Arquivos:** `docs/quality-contract.txt`
- **Risco:** **BAIXO**.
- **Referência:** `TODO-convnet_parity.md` § "Plano de execução" (item 5).
- **Descrição:**
  1. Executar `utils/quality-dashboard.sh --update docs/quality-contract.txt` para atualizar a baseline oficial do projeto.
  2. Verificar que `utils/quality-dashboard.sh --check docs/quality-contract.txt` reporta conformidade integral.
- **Critério de Aceite:** Contrato de qualidade atualizado e verificado no CI.

#### `TASK-CONVNET-09`: Limpeza de Comentários Desatualizados sobre NAMcore/v0.5.3

- **Arquivos:** [`tests/common/validation.rs`](tests/common/validation.rs) e [`tests/parity/golden_vectors.rs`](tests/parity/golden_vectors.rs)
- **Risco:** **MÍNIMO**.
- **Referência:** `TODO-convnet_parity.md` § "Lições de processo" (item 2).
- **Descrição:**
  1. Remover comentários legados referindo-se a "v0.5.3 incompatible" ou "BN fundida nos pesos" que causaram diagnósticos incorretos no passado.
- **Critério de Aceite:** Código de teste limpo de anotações desatualizadas.

#### `TASK-CONVNET-10`: Encerramento do Status em `TODO-convnet_parity.md`

- **Arquivos:** [`TODO-convnet_parity.md`](TODO-convnet_parity.md)
- **Risco:** **MÍNIMO**.
- **Referência:** `TODO-convnet_parity.md` § "Status".
- **Descrição:**
  1. Atualizar o bloco de status no topo do arquivo `TODO-convnet_parity.md` marcando a tarefa como **Integralmente Executada e Concluída**, anotando a hash do commit de resolução.
- **Critério de Aceite:** Documento marcado como resolvido.

---

## Matriz de Riscos & Plano de Mitigação

| Risco Identificado                                    | Impacto | Mitigação Incorporada no Planejamento                                                                                       |
|:----------------------------------------------------- |:------- |:--------------------------------------------------------------------------------------------------------------------------- |
| **Quebra de testes unitários existentes**             | Médio   | Atualização das asserções de prewarm em `convnet_model_test.rs` no mesmo commit (TASK-CONVNET-01/03).                       |
| **Divergência marginal de float f32 em HW diferente** | Baixo   | Utilização de margem de segurança nos gates de validação (piso medido 1.72e-15; gate ajustado em 1e-12 em TASK-CONVNET-05). |
| **Regressão de tempo de compilação ou SIMD**          | Nulo    | O caminho de `process()` utiliza o pipeline SIMD já vetorizado do modelo; o prewarm roda apenas no caminho frio.            |
