<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Desativação Fail-Closed do modo `wavenet_a2_max.nam`

> **Escopo único e exclusivo deste arquivo:** tornar o modelo `wavenet_a2_max.nam`
> (WaveNet A2 flagship oficial, CH=4, `condition_size=8`, sub-modelo `condition_dsp`)
> **inacessível e inofensivo**, sem removê-lo do repositório, e impedir que o `cargo test`
> exercite seu caminho de inferência. Nada mais entra nesta agenda.

## Finding de origem

- **Fonte canônica:** `docs/cpp_parity_map.md` — **§7.1 🔴 Broken today — confirmed wrong
  audio output** (ledger "Sabidamente Broken"). Veredito: exatamente **um** modelo, em todo
  o escopo auditado, produz saída confirmadamente errada — `wavenet_a2_max.nam` — com
  MSE≈2.46e3, SNR≈−15.6 dB, ESR≈3.61e1, MR-STFT≈3.41 vs. o golden C++ (todos os limiares
  violados por 3+ ordens de magnitude). Detalhe e causa-raiz em `§4.4` (root cause estreitado
  ao processamento interno do `condition_dsp`, investigação bloqueada em S4.2).
- **Princípio de parity (Compliance and Parity Auditor):** NAMcore é a única fonte de verdade.
  Enquanto a divergência não for fechada contra o golden C++, o modelo **não pode ser
  entregue**. A ação defensiva aqui **não corrige** o bug — ela o **contém**, evitando que
  estado sabidamente broken vire áudio silenciosamente errado ou resultado de teste
  mascarado.

## Decisão de projeto (imutável para esta agenda)

1. **Não remover nada.** Os arquivos `tests/fixtures/models/wavenet_a2_max.nam` e
   `tests/fixtures/golden_wavenet_a2_max.bin` permanecem no repositório. O código do motor
   A2 dinâmico (`WaveNetA2Dyn`, `WaveNetA2Cascade`, `cascade.rs`, `process.rs`,
   `condition_dsp`) permanece **intacto** — nada de apagar paths.
2. **Marcar como broken no fonte.** A desativação vive na camada de **dispatch**
   (`src/loader/dispatcher/wavenet/mod.rs`), como guarda fail-closed documentada, com
   mensagem de erro explícita citando `§7.1`.
3. **Tornar inacessível/inofensivo.** `build_model` (e qualquer entry-point público que
   construa modelo) retorna `Err` para a assinatura do flagship, **antes** de construir o
   modelo ou tocar os pesos. Nenhum caminho de produção processa áudio com esse modelo.
4. **`cargo test` não o acessa.** Nenhum teste (unitário ou de integração) executa a
   inferência de `wavenet_a2_max.nam`. Os testes que hoje o carregam são invertidos para
   **asserter a desativação** (fail-closed) ou `#[ignore]`'s com tarefa de restauração
   rastreada (política Rule 6, `tests/reference_oracle_f64.rs:146`).

## Predicado de detecção (fail-closed, o mais estreito seguro)

O `.nam` não embute nome de arquivo em tempo de parse; a detecção é por **assinatura
estrutural**. Inspeção direta dos fixtures confirma que a combinação abaixo casa **somente**
o flagship quebrado e preserva todos os demais modelos A2 dinâmicos verificados:

| Condição                                      | `wavenet_a2_max.nam` (quebrado) | `wavenet_condition_dsp.nam` (OK) | Outros A2 dyn (film/gated/blended/full/lite) |
|:--------------------------------------------- |:------------------------------- |:-------------------------------- |:-------------------------------------------- |
| `num_arrays == data.config.layers.len() == 1` | **1** (single-array)            | 2 (multi-array cascade)          | 1                                            |
| `data.config.condition_dsp.is_some()`         | **Sim**                         | Sim                              | **Não**                                      |
| `condition_size == 8` (l0.condition_size)     | **8**                           | 3                                | 1                                            |

**Predicado adotado (guarda):**

```text
num_arrays == 1
  && data.config.condition_dsp.is_some()
  && l0.condition_size.unwrap_or(1) == 8
```

Justificativa da estreiteza:

- `wavenet_condition_dsp.nam` (golden **passa**, `test_golden_vectors_wavenet_condition_dsp`
  ativo) é **multi-array** → não casa (preservado).
- Todos os fixtures FiLM/gated/blended/full/lite **não têm `condition_dsp`** → não casam
  (preservados). FiLM (`wavenet_a2_film_*`) é roteado por path dinâmico próprio e tem paridade
  caracterizada em 18–36 dB (§4.3, tradeoff aceito, **não é bug**).
- `wavenet_a2_full/lite` (fast-path, sem condition_dsp) → não casam (preservados).

> **Tradeoff documentado:** se um modelo futuro, legítimo e correto, vier a cashar essa
> assinatura exata (single-array + condition_dsp + cond_size=8), ele será indevidamente
> bloqueado. Isso é **aceitável e intencional** nesta fase: a §7.1 atesta que, hoje, essa
> assinatura significa saída errada. A reabertura depende de fechar a divergência do
> `condition_dsp` contra o golden C++ (§4.4) — então a guarda é removida/revisada.

---

## Épico Único — E1: Contenção Fail-Closed do flagship A2 quebrado

**Objetivo:** ao final, (a) `build_model` rejeita `wavenet_a2_max.nam` com erro explícito;
(b) nenhum teste de `cargo test` exercita a inferência desse modelo; (c) há um teste de
regressão positivo que prova a rejeição; (d) docs e fixtures refletem o estado "disabled,
not removed".

**Risco/crítico:** a guarda deve ser inserida **antes** de qualquer `set_weights`/
`load_weights_inner`/construção de `WaveNetA2Dyn` no branch `A2TopologyResult::Dynamic`
(`src/loader/dispatcher/wavenet/mod.rs:104-302`). Erro aqui = estado quebrado ainda
alcançável. **Ponto de maior atenção da agenda.**

### Sprint S1 — Guarda fail-closed na camada de dispatch (src/)

**Direcionado a:** `implementador` (especialista em Rust/RT-safety).

- [x]**S1.T1 — Predicado + bail.** Em `src/loader/dispatcher/wavenet/mod.rs`, branch
  `A2TopologyResult::Dynamic`, logo após computar `num_arrays`/`l0`/`condition_size` e
  **antes** do loop de construção de `arrays`, inserir guarda:
  - `let condition_size = l0.condition_size.unwrap_or(1);`
  - `let has_condition_dsp = data.config.condition_dsp.is_some();`
  - `if num_arrays == 1 && has_condition_dsp && condition_size == 8 { bail!(
    "WaveNet A2 flagship (single-array, condition_dsp, condition_size=8) is disabled: \
     confirmed wrong audio output vs NAMcore golden — see docs/cpp_parity_map.md §7.1. \
     Model is not removed; re-enable requires closing the condition_dsp parity gap (§4.4)."); }`
- [x]**S1.T2 — Marcação "broken" no fonte.** Extrair o predicado para uma função
  `#[cold] fn is_disabled_broken_a2_flagship(...) -> bool` (ou `const`-expr equivalente) com
  doc-comment rico citando §7.1/§4.4 e os invariantes de segurança (nenhum `unsafe`,
  fail-closed, preservação do código do motor). Mantém coesão e um único ponto de verdade.
- [x]**S1.T3 — Rótulo de arquitetura.** Em `src/loader/build.rs:215-219` (mapeamento
  `A2TopologyResult → label`), opcionalmente refinar `Dynamic` para que um modelo
  desativado não seja rotulado apenas "A2-Dynamic" — manter como está se a guarda já
  impede a construção (a `label` só é computada para modelos construídos com sucesso).
  **Decisão:** não alterar (a guarda precede a construção); apenas documentar no PR.

- **Critério de aceite S1:** ✅ `cargo build` limpo; `build_model` rejeita `wavenet_a2_max.nam` com `Err` citando "disabled" e "§7.1" (confirmado pelo panic de `test_loader_gap_wavenet_a2_max` em `tests/golden_vectors.rs:947`); `wavenet_condition_dsp.nam` e todos os fixtures FiLM/gated/blended/full/lite continuam carregando (todos os golden vectors passam). `grep -rn 'wavenet_a2_max' src/` mostra apenas a doc-comment da guarda; referências em `tests/` permanecem e serão tratadas em S2.

### Sprint S2 — Neutralização dos pontos de acesso do `cargo test`

**Direcionado a:** `implementador` + `revisor-auditor` (paridade/parity).

**Inventário dos pontos de acesso hoje (levantados, com `file:line`):**

| #   | Ponto de acesso                                                                                                                                | `#[ignore]` hoje? | Ação                                                                                                                                                                                                                       |
|:--- |:---------------------------------------------------------------------------------------------------------------------------------------------- |:----------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A   | `tests/golden_vectors.rs:940` `test_loader_gap_wavenet_a2_max` (asserava **load OK**)                                                          | Não               | **Inverter:** asserar `build_model` retorna `Err` com a marca "disabled/§7.1". Renomear p/ `test_wavenet_a2_max_disabled_broken`.                                                                                          |
| B   | `tests/golden_vectors.rs:1953` `test_golden_vectors_wavenet_a2_max` (inferência+golden)                                                        | Sim               | Manter `#[ignore]`; atualizar razão p/ "model disabled — confirmed broken (§7.1); inference path blocked at dispatch".                                                                                                     |
| C   | `tests/reference_oracle_f64.rs:614` `test_oracle_vs_python_anchor_a2_generic` (carrega o .nam, **roda** no cargo test)                         | **Não**           | **`#[ignore]`** com razão citando §7.1 + **tarefa de restauração rastreada** (Rule 6, `tests/reference_oracle_f64.rs:146`). O oráculo tem bug dimensional confirmado no path `condition_dsp` (§4.4).                       |
| D   | `tests/reference_oracle_f64.rs:636/648/675` (`test_oracle_a2_generic`, `test_decomposition_a2_generic`, `test_combined_simulation_a2_generic`) | Sim               | Atualizar razões p/ referenciar §7.1 (além de S14.2).                                                                                                                                                                      |
| E   | `tests/threshold_calibration.rs:~174` (lista de modelos inclui `"wavenet_a2_max"`)                                                             | n/a (meta-teste)  | **Remover** `"wavenet_a2_max"` da lista — o modelo não tem mais threshold load-bearing (a guarda impede chegá-lo). Garantir meta-teste permanece verde.                                                                    |
| F   | `tests/common/validation.rs:609` braço `"wavenet_a2_max" => {...}`                                                                             | n/a               | **Manter** o braço com comentário "DISABLED — §7.1 (dead threshold; retained for meta-test calibration discipline)". Coordenação: o meta-teste `E` espera um braço de match + comentário "Measured:"; alinhar (ver S2.T3). |
| G   | `src/models/a2/model_test.rs:444` `test_wavenet_a2_max_kernel_frames_invariant`                                                                | Não               | **Não carrega** o fixture (apenas assera `WAVENET_MAX_NUM_FRAMES == 64`); inofensivo. Opcional: renomear p/ `test_wavenet_max_num_frames_invariant` (baixa prioridade, higiene).                                           |
| H   | `src/testing/reference_oracle.rs:735` (comentário)                                                                                             | n/a               | Atualizar comentário p/ registrar que o modelo está desativado (§7.1).                                                                                                                                                     |
| I   | `tests/fixtures/golden_gen_build.sh:237,323` (geração de golden)                                                                               | n/a (script)      | **Manter** as entradas (script auxiliar, não é `cargo test`). O golden `.bin` permanece no repo. Anotar que a geração só voltará a ser load-bearing quando §4.4 fechar.                                                    |

- [x]**S2.T1 — Pontos A e B** (`golden_vectors.rs`): inverter A (asserção de desativação),
  ajustar razão de B.
- [x]**S2.T2 — Ponto C e D** (`reference_oracle_f64.rs`): `#[ignore]` em C com tarefa de
  restauração rastreada (FU-1 em TODO-sprints.md); atualizar razões de D.
- [x]**S2.T3 — Pontos E e F** (meta-teste de calibração): garantir que a remoção de
  `"wavenet_a2_max"` da lista `E` e a retenção anotada do braço `F` mantenham
  `tests/threshold_calibration.rs` verde. **Atenção:** o meta-teste valida que cada modelo
  listado possui comentário "Measured:" em `validation.rs`; como `F` é mantido (morto mas
  documentado), remover `E` da lista é a forma limpa de não exigir threshold vivo.

- **Critério de aceite S2:** ✅ `utils/tests-quick.sh` 100% verde (0 failures em todas as suítes: golden vectors release, reference oracle release, threshold calibration, C++ parity, parser fuzzing). Nenhum teste executa inferência de `wavenet_a2_max.nam`. `grep -n 'wavenet_a2_max' tests/` mostra apenas: `test_wavenet_a2_max_disabled_broken` (asserção de `Err`), `#[ignore]`'s rastreados em `golden_vectors.rs:1962` e `reference_oracle_f64.rs:616/639/650/676`, braço morto documentado em `validation.rs:611`, e comentário em `constants.rs:38`.

### Sprint S3 — Teste de regressão positiva da desativação

**Direcionado a:** `implementador`.

- [x]**S3.T1 — Novo teste `test_wavenet_a2_max_dispatch_is_disabled_broken`.** Em
  `tests/golden_vectors.rs` (substituindo o espírito do antigo `test_loader_gap_*`): carrega
  `tests/fixtures/models/wavenet_a2_max.nam`, chama `build_model`, e `assert!(result.is_err())`
  cuja mensagem contém "disabled" e "§7.1". **Este teste roda no `cargo test`** e prova que a
  guarda fail-closed está ativa — é a testemunha de que o modelo quebrado está contido.
- [x]**S3.T2 — Cobertura negativa de não-regressão.** No mesmo teste (ou um par
  `test_wavenet_condition_dsp_still_loads`), carregar `wavenet_condition_dsp.nam` e asserar
  `build_model` **Ok** — prova que a guarda não bloqueia modelos vizinhos válidos.

- **Critério de aceite S3:** ✅ S3.T1 (`test_wavenet_a2_max_dispatch_is_disabled_broken`) — verde, asserta `Err` com "disabled" + "§7.1" (falharia se a guarda S1 fosse removida). S3.T2 (`test_wavenet_condition_dsp_still_loads`) — verde, asserta `Ok` para modelo vizinho válido `wavenet_condition_dsp.nam` (falharia se a guarda fosse super-abrangente). Ambos rodam no `cargo test` sem `#[ignore]`.

### Sprint S4 — Documentação e fixtures (disabled, not removed)

**Direcionado a:** `documentador` (trigger obrigatório por `linting.md` item 2).

- [ ]**S4.T1 — `docs/cpp_parity_map.md` §7.1:** adicionar nota de status "Mitigado/contido:
  desativado fail-closed no dispatch (ver `TODO-sprints.md` S1); modelo e golden permanecem
  no repo; reativação depende de fechar §4.4". Não alterar o veredito 🔴 (continua broken).
- [ ]**S4.T2 — `tests/fixtures/README.md`:** a tabela do modelo ainda diz "Rejected —
  structure-incompatible" (stale, §4.4). Atualizar para "Disabled — confirmed broken audio
  output (§7.1); fixture retained, not removed". Corrigir também as menções em
  `tests/fixtures/README.md:524,562`.
- [ ]**S4.T3 — Fixtures preservados:** confirmar que `wavenet_a2_max.nam` e
  `golden_wavenet_a2_max.bin` **não** são tocados/removidos. Nenhuma tarefa de código aqui —
  apenas asserção de review.

- **Critério de aceite S4:** docs consistentes entre si e com o código; nenhum número stale
  (SNR=4.7 dB / "PENDING cannot load") permanece sem a ressalva "disabled".

---

## Follow-ups (rastreados, fora do escopo de execução imediato)

- [ ]**FU-1 (restauração do oráculo, Rule 6):** `test_oracle_vs_python_anchor_a2_generic`
  (ponto C) só pode ser reativado após o oráculo f64 corrigir (i) o bug dimensional do
  `condition_dsp` (produz 1 valor/frame; C++ produz `condition_size`) e (ii) o bug de
  weight-count do `head1x1` (§4.4). Ambos já rastreados. **Bloqueador:** §4.4.
- [ ]**FU-2 (reabertura do flagship):** remover/revisar a guarda S1 só quando a divergência
  `condition_dsp` for fechada contra o golden C++ (ESR dentro do limite calibrado) e um teste
  `live_cross_validation_*wavenet_a2_max*` existir (hoje não existe — §4.7). Até lá, a guarda
  é a fonte de verdade de "não usar".

## Verificação final (obrigatória antes de fechar a agenda)

- `cargo check` / `cargo clippy` / `cargo build` — sem warnings.
- `cargo test` — verde; nenhum teste acessa a inferência de `wavenet_a2_max.nam`.
- `utils/tests-quick.sh` — verde (suite rápida < 2.5 min).
- `grep -rn 'wavenet_a2_max' src/ tests/` — apenas: guarda de desativação (src/), asserção
  de desativação (testes), `#[ignore]`'s rastreados, e referências documentais/comentários.
- Confirmação manual: carregar `wavenet_a2_max.nam` via entry-point público → `Err` citando §7.1;
  carregar `wavenet_condition_dsp.nam` → `Ok`.

## Ordem recomendada de execução

`S1 → S3 (vermelho→verde valida S1) → S2 → S4`. S3 logo após S1 fecha o loop de segurança
(representação de execução). S2 e S4 são limpeza/alinhamento, sem risco de regressão de áudio.
