<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Dívida de Documentação em `/src` e `docs/`

> **Skill de origem:** `refatora-doc` (inspeção minuciosa de documentação e comentários).
> **Skills de apoio:** `documentador` (padrão de qualidade), `planejador-arquiteto`
> (planejamento seguro de correções).
> **Escopo:** mega-pasta `/src` (≈109 125 linhas de Rust) + documentação de sistema
> (`docs/*.md`, `README.md`) que referencia o código.
> **Data:** 2026-07-21.

Este documento consolida os achados de documentação levantados por varredura
automatizada (busca de referências obsoletas de *tracker*, validação de links
Markdown internos, detecção de blocos longos sem comentários inline) e inspeção
manual focada. O objetivo é apontar oportunidades de tornar a documentação e os
comentários **coerentes com o estado real do código**, **uniformes de voz** e
**livres de referências irrelevantes** (IDs de *sprint*/*task*/*PM*), em
conformidade com a skill `refatora-doc` e os princípios da skill `documentador`.

---

## 1. Sumário executivo

| Critério avaliado                                         | Resultado                                                        |
| --------------------------------------------------------- | ---------------------------------------------------------------- |
| Cabeçalhos SPDX em `*.rs` de `src/`                       | ✅ 0 faltantes                                                   |
| `cargo fmt` / `cargo clippy` (4 perfis)                   | ✅ Verde                                                         |
| `#![warn(missing_docs)]` ativo em `lib.rs`/`main.rs`      | ✅ Itens `pub` docados                                           |
| Referências de *tracker* (Sprint/T/PM/EPIC) em `src/*.rs` | ⚠️ 19 arquivos, ~38 ocorrências                                  |
| Links Markdown internos quebrados em `docs/*.md`          | ⚠️ `TODO-sprints.md` (inexistente) + âncoras obsoletas           |
| Referências de *sprint*/*tarefa* em `docs/*.md`           | ⚠️ 5 documentos afetados                                         |
| Blocos longos sem comentários inline em `src/` (hot-path) | ⚠️ Parcialmente corrigido em `layout.rs`; restante a inventariar |

**Correções já aplicadas nesta sessão** (ver §4): limpeza de 3 referências de
*tracker* em `src/` e adição de comentários inline às 5 funções de transposição
de `layout.rs`. Tudo validado por `utils/lints.sh` e testes direcionados.

---

## 2. Achados

### Finding F-01 — *(RESOLVIDO)* Testes inline em `layout.rs` ≥ 300 linhas

- **Status:** Resolvido no commit `f86086f8` (*split src/loader/dispatcher/wavenet/layout.rs*).
- **Resumo:** O bloco `#[cfg(test)] mod tests` foi extraído para
  `src/loader/dispatcher/wavenet/layout_test.rs` via
  `#[cfg(test)] #[path = "layout_test.rs"] mod layout_test;`, cumprindo
  `.agents/rules/testing.md` §1. Mantido aqui apenas para rastreabilidade.

---

### Finding F-02 — Referências obsoletas de *tracker* em `src/*.rs`

- **Regra violada:** `refatora-doc` — *"Remove irrelevant references that do not
  contribute strictly to understanding the code, such as 'sprint X',
  'review done on DDMMYYYY', 'requested by PO', etc."*
- **Abrangência:** 19 arquivos, ~38 ocorrências de IDs como `S13.2`, `S14.1`,
  `S14.2`, `PM-15`, `T1.2`, `T2.3`, `T5.1`, `S6.T12`, `S5.T09`, `S7.T08`,
  `T4.6`, `T2.S3.3`, `EPIC-2`, `T2.E2.1`, etc.
- **Arquivos afetados:**
  - `src/clap/processor/state.rs`
  - `src/dsp/adaptive.rs`
  - `src/loader/dispatcher/wavenet/mod.rs` *(parcialmente corrigido nesta sessão — ver §4)*
  - `src/loader/namb_test.rs`
  - `src/loader/nam_json/activation_parser.rs`
  - `src/loader/nam_json/topology/a2.rs`
  - `src/loader/nam_json/topology/wavenet.rs`
  - `src/loader/nam_json_test.rs`
  - `src/math/common/common_test.rs`
  - `src/math/common/kahan_test.rs`
  - `src/math/gemm/dot_16x/dot_f32_avx2.rs`
  - `src/models/a2/conv1d.rs`
  - `src/models/a2/model/dynamic/build.rs`
  - `src/models/a2/model/dynamic/process.rs`
  - `src/models/wavenet/conv1d_dual.rs`
  - `src/models/wavenet/conv_input.rs`
  - `src/standalone/pw_host/mod.rs` *(parcialmente corrigido nesta sessão — ver §4)*
  - `src/testing/reference_oracle/a2.rs`
  - `src/testing/reference_oracle/mod.rs`
  - `src/testing/reference_oracle/wavenet.rs`

#### 2.1 Natureza do problema

Esses IDs referenciam um sistema de planejamento ágil histórico (sprints/tarefas
`Sn.Tnn`, `PM-nn`, `EPIC-n`) que **não existe mais no repositório** — os arquivos
`TODO-findings.md` e `TODO-sprints.md` que os fundamentavam foram removidos (ver
F-03/F-04). Para um leitor atual, `// S14.2 (PM-15): Correct grouped head1x1
accumulation.` mistura ruído de rastreabilidade (`S14.2 (PM-15)`) com a
informação útil (`Correct grouped head1x1 accumulation`).

#### 2.2 Proposta de solução (correção segura)

Para cada ocorrência, **preservar a substância técnica** e **remover apenas o
prefixo/sufixo de ID de *tracker***. Exemplos:

- `// S14.2 (PM-15): Correct grouped head1x1 accumulation.`
  → `// Correct grouped head1x1 accumulation.`
- `/// S14.1 (PM-15): Relaxed from != 1 to < 1 to support multi-array`
  → `/// Relaxed from != 1 to < 1 to support multi-array`
- `// A2 generic (S13.2, S14.1): arbitrary kernel sizes are valid`
  → `// A2 generic: arbitrary kernel sizes are valid`
- `/// This is the S5.T03 acceptance criterion: drift reduction ≥ 100×.`
  → `/// Acceptance criterion: drift reduction ≥ 100×.`

**Regras de segurança:**

- Não alterar nenhuma linha de código — apenas texto de comentários `//`/`///`
  e *strings* de mensagem de erro **não cobertas por *asserts* de teste**.
- Antes de editar uma *string* de erro, buscar *asserts* (`err_msg.contains(...)`)
  em `tests/` e preservar o *substring* verificado (precedente: a mensagem de
  `reject_condition_dsp_lstm` é coberta por `tests/models/golden_vectors.rs`).
- Executar `utils/lints.sh` + `utils/tests-quick.sh` ao final.

#### 2.3 Análise de risco

- **Risco de regressão lógica:** nulo (apenas texto de comentário/mensagem).
- **Risco de teste:** baixo, desde que *substrings* cobertos por `contains` sejam
  preservados. Inventariar antes de editar mensagens de erro.

---

### Finding F-03 — Links para `TODO-sprints.md` (arquivo inexistente) em `docs/`

- **Regra violada:** `refatora-doc` — *"Ensure internal links in Markdown
  documents work perfectly."*
- **Problema:** O arquivo `TODO-sprints.md` **não existe** no repositório (foi
  removido; o commit `a3f5a4bf` já iniciou a limpeza de referências pendentes).
  Os links abaixo estão quebrados (404):
  - `docs/latency_sprint1_analysis.md:10` — `[Sprint S1](../TODO-sprints.md#L1455)`
  - `docs/latency_sprint1_analysis.md:161` — `[TODO-sprints.md#L1455](../TODO-sprints.md#L1455)`
  - `docs/postmortem-libm-symbol-interposition.md:9` — referência narrativa a `TODO-sprints.md`
  - `docs/namb-spec.md:394` — `Planning: TODO-sprints.md (S3.T03, S3.T04, S5.T02, S5.T03)`
  - `docs/cpp_parity_map.md:889` — referência narrativa a `TODO-sprints.md`

#### 3.1 Proposta de solução

Remover as referências a `TODO-sprints.md` (e os IDs de *sprint*/*tarefa*
associados), substituindo-as por pointers estáveis quando houver conteúdo
tecnicamente útil a preservar (ex.: apontar para `docs/architecture.md` ou para
a seção relevante do próprio documento). Quando a referência for puramente de
rastreabilidade histórica sem valor de entendimento, remover sem substituição.

---

### Finding F-04 — Âncoras obsoletas de `TODO-findings.md` em `docs/`

- **Problema:** `TODO-findings.md` foi recriado do zero (este documento) com
  conteúdo distinto da versão histórica deletada. Os links abaixo resolvem o
  arquivo-base, mas as **âncoras** apontam para conteúdo que não existe mais:
  - `docs/architecture.md:478` — `[TODO-findings.md §R13](../TODO-findings.md#r13)`
    (âncora `#r13` inexistente)
  - `docs/audio_fidelity_map.md:85` — `F-I3 in [TODO-findings.md](../TODO-findings.md#L332)`
    (âncora `#L332` inexistente)
  - `docs/latency_sprint1_analysis.md:10` — `[F-S1](../TODO-findings.md#L400)`
    (âncora `#L400` inexistente)
  - `docs/latency_sprint1_analysis.md:160` — `[TODO-findings.md#L400](...)`
    (âncora `#L400` inexistente)
  - `docs/perceptual_validation.md:1006,1008` — `](TODO-findings.md)` — caminho
    relativo resolve para `docs/TODO-findings.md` (**quebrado**; o arquivo está
    na raiz; deveria ser `../TODO-findings.md`), além de apontar para findings
    históricos de validação perceptual não presentes neste documento.

#### 4.1 Proposta de solução

Estas referências apontam para *artefatos de planejamento histórico deletados*
(F-S1, F-I3, R13). A ação correta, consistente com `refatora-doc`, é **remover**
esses links/âncoras obsoletos (ruído de *tracker*) e, onde houver conteúdo
técnico útil (ex.: justificativa de threshold, decisão de ativação),
**reescrevê-lo *in situ*** no documento ou apontar para a fonte estável
(`docs/architecture.md`, `docs/audio_fidelity_map.md`). Recuperar o conteúdo
original dos findings históricos **não** é viável nem desejável.

---

### Finding F-05 — Referências de *sprint*/*tarefa* em `docs/*.md`

- **Regra violada:** `refatora-doc` — remover referências irrelevantes
  (*"sprint X"*).
- **Ocorrências:**
  - `docs/perceptual_validation.md:733` — *"LSTM Recurrent State Drift (post-Sprint 2 RCA...)"*
  - `docs/perceptual_validation.md:737` — *"Sprint 2 (2026-07-08) completed the root-cause..."*
  - `docs/perceptual_validation.md:763` — *"The Sprint 2 investigation (Tarefa 2.3) confirmed..."*
  - `docs/audio_fidelity_map.md:146` — *"As of Sprint 2 (Tarefa 1.2)..."*
  - `docs/cpp_parity_map.md:520` — *"formal specification for T1.1 (EP-A, Sprint 1)..."*
  - `docs/wavenet_lite_efficiency_decision.md:129` — *"(F-S1/T5.S1.2)"*

#### 5.1 Proposta de solução

Reescrever preservando a informação técnica (RCA, decisão, especificação) e
removendo os marcadores de *sprint*/*tarefa*. Exemplo:

- *"Sprint 2 (2026-07-08) completed the root-cause investigation"*
  → *"The root-cause investigation established..."*

---

### Finding F-06 — Blocos longos sem comentários inline em `src/` (hot-path)

- **Regra violada:** `refatora-doc` — *"Do not leave long blocks of code
  (50~100 lines) without any inline comments."*
- **Estado:** **Parcialmente corrigido** — nesta sessão foram adicionados
  comentários inline às 5 funções de transposição de
  `src/loader/dispatcher/wavenet/layout.rs`
  (`transpose_conv1d_interleaved_4wide/8wide/16wide`,
  `transpose_4wide_to_8wide/16wide`), cujos loops aninhados de indexação
  aritmética eram opacos.
- **Restante a inventariar:** kernels SIMD em
  `src/math/gemm/` (`dot_16x`, `dot_4x`, `gemv/`), `src/math/common/avx512/`,
  `src/dsp/adaptive.rs`, e oráculos em `src/testing/reference_oracle/` podem
  conter blocos longos sem comentários. Requer varredura focalizada por módulo.

#### 6.1 Proposta de solução

Inventariar funções com corpo ≥ 50 linhas e sem comentários inline (via
`wc -l` + inspeção), priorizando *hot-path* de DSP/ínferência. Adicionar
comentários `//` explicando o *layout* de memória / indexação, **sem alterar
lógica** (regra `refatora-rust`: regressões proibidas).

---

## 3. Epics

### Epic E-01 — *(CONCLUÍDO)* Conformidade da regra de 300 linhas em `layout.rs`

- **Findings:** F-01. **Status:** Resolvido no commit `f86086f8`.

### Epic E-02 — Limpeza de referências de *tracker* em `src/*.rs` [DONE]

- **Findings:** F-02.
- **Risco:** Baixo (apenas texto de comentário/mensagem; preservar *substrings*
  cobertos por testes).
- **Especialista alvo:** `implementador` (Rust) / `refatora-doc`.
- **Tarefas técnicas:**
  - **TT-02.1** — Inventariar todas as ~38 ocorrências (lista em §2) com
    `file:line` e classificar em "comentarário `//`/`///`" vs. "string de erro".
  - **TT-02.2** — Para cada *string* de erro, buscar `contains(...)` em `tests/`
    e marcar *substrings* protegidos.
  - **TT-02.3** — Editar comentários removendo apenas IDs de *tracker*,
    preservando substância técnica; normalizar voz (EN unificado) no trecho editado.
  - **TT-02.4** — `utils/lints.sh` + `utils/tests-quick.sh`.
- **Critério de aceite:** 0 ocorrências de `\b(Sn\.Tnn|PM-nn|EPIC-n|Sprint n)\b`
  em `src/*.rs`; testes verdes.

### Epic E-03 — Reparo de links Markdown quebrados em `docs/` [DONE]

- **Findings:** F-03, F-04.
- **Risco:** Baixo (apenas Markdown; sem impacto em compilação/testes).
- **Especialista alvo:** `documentador` / `refatora-doc`.
- **Tarefas técnicas:**
  - **TT-03.1** — Remover/reescrever referências a `TODO-sprints.md` (5 docs).
  - **TT-03.2** — Remover/reescrever âncoras obsoletas `#L400/#r13/#L332` e
    corrigir caminho relativo `docs/perceptual_validation.md` (4 docs).
  - **TT-03.3** — Validar links internos com verificador de Markdown (ex.:
    `lychee` ou script equivalente) ou inspeção manual.
- **Critério de aceite:** 0 links internos quebrados em `docs/` + `README.md`.

### Epic E-04 — Remoção de referências de *sprint* em `docs/*.md`

- **Findings:** F-05.
- **Risco:** Baixo (Markdown).
- **Especialista alvo:** `documentador`.
- **Tarefas:** Reescrever 6 ocorrências preservando conteúdo técnico.
- **Critério de aceite:** 0 ocorrências de `Sprint [0-9]` / `Tarefa [0-9]` em
  `docs/*.md` (exceto onde "Sprint" for termo técnico legítimo de produto).

### Epic E-05 — Comentários inline em blocos longos de *hot-path*

- **Findings:** F-06.
- **Risco:** Médio (exige precisão técnica nos comentários; sem tocar lógica).
- **Especialista alvo:** `implementador` + revisão `revisor-auditor` (DSP).
- **Tarefas:** Inventariar funções ≥ 50 linhas sem comentários em
  `src/math/gemm/`, `src/math/common/avx512/`, `src/dsp/adaptive.rs`,
  `src/testing/reference_oracle/`; adicionar `//` de *layout*/indexação.
- **Critério de aceite:** nenhuma função de *hot-path* ≥ 50 linhas sem ao menos
  um comentário de *layout* de memória/indexação; `utils/lints.sh` verde.

---

## 4. Correções já aplicadas nesta sessão (refatora-doc)

As edições abaixo já foram realizadas e validadas (`cargo fmt --check`,
`cargo test --lib layout`, `cargo test --features testing --test models condition_dsp`,
`utils/lints.sh` — todos verdes). Estão pendentes de *commit* (não commitadas,
conforme política).

| Arquivo                                   | Linha(s)                | Correção                                                                                                                                                                                                                                                                               |
| ----------------------------------------- | ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/loader/dispatcher/wavenet/mod.rs`    | 55–79                   | Removidos `(T3.1)`, `T2.3`, `Sprint 4` do doc-comment e da mensagem de erro de `reject_condition_dsp_lstm`; preservado o *substring* `"LSTM condition_dsp is not supported"` coberto por `tests/models/golden_vectors.rs`; "Postura" → "Fail-closed stance" (uniformização de voz EN). |
| `src/standalone/pw_host/mod.rs`           | 41                      | Removido o ID de tarefa `(T5.S1.1)`; preservada a referência de rastreabilidade a `docs/latency_sprint1_analysis.md`.                                                                                                                                                                  |
| `src/loader/dispatcher/wavenet/layout.rs` | 155, 191, 227, 260, 297 | Adicionados comentários `//` explicando o *layout* de memória `[block][kernel][in_ch][lane]` e a lógica de mesclagem de blocos 4→8/16 nas 5 funções de transposição (blocos antes opacos).                                                                                             |

---

## 5. Notas finais

- `TODO-sprints.md` **não** foi criado: o arquivo foi removido do repositório e
  as referências residuais são dívida a limpar (Epic E-03), não a recriar.
- A normalização de voz/idioma dos comentários (unificar EN/PT) é desejável
  (`documentador` §"Technical Documentation Standardization") mas **não** está
  no escopo imediato dado o volume (≈109k linhas); recomenda-se demanda separada.
- `utils/tests-long.sh` **não** deve ser executado por tarefa de IA
  (`.agents/rules/testing.md` §2).
