<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings — Refatora Rust (Refatoração Estrutural do `nam-rs`)

> **Escopo:** Refatoração **exclusivamente estrutural** do código Rust em `src/`.
> **Regra de ouro:** Nenhuma alteração de lógica, algoritmo, fluxo de controle ou
> API pública é permitida. Apenas movimentação de código existente para arquivos
> irmãos, extração de testes inline, remoção de código morto verificado e
> reorganização de diretórios. **Regressões são estritamente proibidas.**
>
> **Skill de origem:** `refatora-rust`. **Planejamento:** `planejador-arquiteto`.
> **Regras aplicáveis:** `.agents/rules/testing.md` (colocação de testes),
> `.agents/rules/` gerais (RT-safety, SPDX, lint).

---

## Metodologia de verificação

Os achados abaixo foram coletados por agentes exploradores paralelos e
**verificados** antes da consolidação:

- **Contagem de linhas:** `find src -name '*.rs' | xargs wc -l`, ordenado
  decrescentemente. Linhas "src" = total menos o bloco `#[cfg(test)] mod tests`.
- **Dead code:** para cada item marcado `#[allow(dead_code)]` / `#[allow(unused_*)]`,
  foi feita busca exaustiva (`grep`) do nome do item em **todo** o repositório
  (`src/`, `tests/`, `benches/`), considerando também uso sob `#[cfg(feature)]`.
  Os candidatos TRULY_DEAD abaixo têm **zero chamadores** confirmados.
- **Splitting:** leitura integral de cada arquivo, enumeração de itens de topo e
  avaliação de risco de codegen (especialmente em kernels SIMD com
  `#[inline(always)]` / `#[target_feature]`).
- **Organização:** leitura dos `mod.rs` relevantes e cruzamento com as regras de
  colocação de testes.

Scripts de verificação final (a executar após cada Epic, na ordem):
`utils/lints.sh` → `utils/tests-quick.sh` (uma vez por tarefa de IA, na validação
final). `utils/tests-long.sh` **nunca** em tarefa de IA.

---

## A — Violações de colocação de testes

Regra (`testing.md` §1): arquivos com **>= 300 linhas de código-fonte** (excluído
o código de teste) **não** podem manter `#[cfg(test)] mod tests { ... }` inline;
devem mover o bloco para um irmão `<module>_test.rs` incluído via
`#[cfg(test)] #[path = "..."] mod tests;`.

### A1 — `src/models/linear.rs`

- **Total:** 949 · **Src:** 333 · **Teste inline:** 616 (linhas 334–949).
- **Bloco:** `#[cfg(test)] mod tests {` inicia na **linha 334**.
- **Ação:** mover linhas 334–949 para `src/models/linear_test.rs`; substituir por
  `#[cfg(test)] #[path = "linear_test.rs"] mod tests;`.
- **Risco:** ZERO — movimentação mecânica; o bloco já usa `use super::*`.

### A2 — `src/models/linear_fft.rs`

- **Total:** 739 · **Src:** 397 · **Teste inline:** 342 (linhas 398–739).
- **Bloco:** `mod tests {` inicia na **linha 398**.
- **Ação:** mover para `src/models/linear_fft_test.rs` e incluir via `#[path]`.
- **Risco:** ZERO.

### A3 — `src/math/dsp/fft_radix4.rs`

- **Total:** 532 · **Src:** 305 · **Teste inline:** 227 (linhas 306–532).
- **Bloco:** `mod tests {` inicia na **linha 306**.
- **Ação:** mover para `src/math/dsp/fft_radix4_test.rs` e incluir via `#[path]`.
- **Risco:** ZERO. (Arquivo é artefato de pesquisa, mas a regra aplica-se.)

### A4 — `src/math/common/aligned.rs`

- **Total:** 488 · **Src:** 368 · **Teste inline:** 120 (linhas 369–488).
- **Bloco:** `mod tests {` inicia na **linha 369**.
- **Ação:** mover para `src/math/common/aligned_test.rs` e incluir via `#[path]`.
- **Risco:** ZERO.

> **Observação (não-achado):** `src/standalone/cli.rs` (268 src) e
> `src/loader/nam_json/model.rs` (194 src) **estão abaixo** de 300 linhas-src;
> seus testes inline estão em conformidade. Nenhuma ação.

---

## B — Código morto / não utilizado

### B1 — Funções `load_*_accums` em `src/models/wavenet/conv_input.rs`  **[TRULY_DEAD, verificado]**

- **Linhas:** `load_4_accums` (12–39), `load_8_accums` (71–97),
  `load_16_accums` (123–167).
- **Evidência:** `grep` por `load_4_accums|load_8_accums|load_16_accums` em todo
  `src/` retorna **apenas as três definições** — **zero chamadores**.
  As funções gêmeas `store_4/8/16_accums` **não** têm `#[allow(dead_code)]` e são
  usadas (sinal adicional de que as `load_*` são resíduo de extração nunca
  concluída — os comentários dizem "Extracted verbatim from single-frame
  convolution kernel").
- **Ação:** remover as três funções + seus `#[allow(dead_code)]` e
  `#[allow(clippy::needless_range_loop)]` associados. Manter `store_*_accums`.
- **Risco:** ZERO (zero chamadores).

### B2 — `weight_f16c_to_f64` em `src/testing/reference_oracle.rs:171`  **[TRULY_DEAD]**

- **Evidência:** 0 chamadores no repositório; utilitário nunca conectado à lógica
  do oráculo.
- **Ação:** remover a função e o `#[allow(dead_code)]`.
- **Risco:** ZERO.

### B3 — Campo `phase_type` + getter em `src/dsp/resampler.rs`  **[TRULY_DEAD]**

- **Linhas:** campo `phase_type: PhaseType` (120) escrito em `ResamplerCore::new`
  mas **nunca lido** em código vivo; getter `fn phase_type(&self)` (151) com 0
  chamadores. O único leitor é o próprio getter morto.
- **Ação:** remover o getter (151) e o campo (120) **em conjunto**. A escrita em
  `new()` deve ser removida junto.
- **Risco:** ZERO (cuidado: revisar o construtor `new()` para remover a atribuição
  órfã). Verificar `PhaseType` — se ficar sem uso, considerar mantê-la (pode ser
  parte da API/estado); **não** remover o tipo, apenas o campo/getter.

### B4 — `log_spaced_tones_dense` em `src/dsp/resampler_test.rs:687`  **[TRULY_DEAD]**

- **Evidência:** 0 chamadores; apenas `log_spaced_tones` (variante esparsa) é
  usado nos testes.
- **Ação:** remover a função e o `#[allow(dead_code)]`.
- **Risco:** ZERO.

### B5 — Constantes mortas em `src/testing/perceptual.rs`  **[TRULY_DEAD]**

- **Linhas:** `A2ESR_A1_STANDARD_Q1` (22), `A2ESR_A1_STANDARD_Q3` (24),
  `A2ESR_A2_FULL_Q1` (28), `A2ESR_A2_FULL_Q3` (30), `NAM_RS_CPP_PARITY_ESR_MAX`
  (40).
- **Evidência:** 0 referências externas (apenas comentário próprio). As constantes
  MEDIAN e todas as funções públicas (`compute_esr`, `compute_lufs`, etc.) **são**
  usadas.
- **Ação:** remover as 5 constantes. **Após** remover, avaliar remover o
  `#![allow(dead_code)]` da linha 11 (módulo) — pode haver outros helpers internos
  mortos; remover só se `cargo check` não emitir novos warnings de dead_code que
  não existiam antes. Se surgirem, manter o `#![allow]` no módulo.
- **Risco:** ZERO para a remoção das constantes; MÉDIO para remover o
  `#![allow]` do módulo (necessita reauditoria).

### B6 — Anotações *stale* (item **é** usado; remover só a anotação)  **[baixo risco]**

| Arquivo:Linha                               | Item                                         | Evidência de uso                                           |
| ------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------- |
| `src/clap/processor/state.rs:122`           | `#[allow(dead_code)]` em `max_frames_count`  | 23 hits em `processor/mod.rs`, benches, standalone, testes |
| `src/loader/dispatcher/checked_arith.rs:37` | `#[allow(dead_code)]` em `fn checked_add`    | chamada em `lstm/weights.rs:104`                           |
| `src/loader/nam_json/topology/mod.rs:16`    | `#[allow(unused_imports)]` em `parse_semver` | cadeia de re-export viva (`nam_json/mod.rs:34`)            |

- **Ação:** remover apenas o atributo `#[allow(...)]` (manter o item).
- **Cautela — `src/dsp/pipeline/context.rs:87-97`:** os quatro campos `os_*`
  **são** usados em `capture.rs` e `stages/inference.rs`, mas o `#[allow(unused)]`
  pode ser necessário em combinações de feature específicas (standalone/clap-plugin
  sem test). **Manter** e marcar para revisão humana.

### B7 — Manter (anotação é *load-bearing*)  **[nenhuma ação]**

Para registro/documentação — **não remover**:

- `src/dsp/pipeline/output_pw.rs:24` `AppState` — struct RAII (drop mantém streams
  PipeWire vivos); campos nunca lidos por design.
- `src/dsp/pipeline/mod.rs:34,60` re-exports `DENORMAL_DITHER_OFFSET` e
  `test_util` sob `cfg(test)/feature=testing` — necessários em combos de feature.
- `src/models/a2/model/dynamic/process.rs:508,178,476` — supressões de clippy /
  atribuição genuinamente morta por fluxo de branch.
- `src/clap/plugin/mod.rs:198` `#[cfg_attr(test, allow(unused_mut))]` —
  mutabilidade condicional ao modo de teste.
- `src/testing/wav.rs:10` `#![allow(dead_code)]` — API pública usada (14+/5+ sites);
  helpers internos precisariam auditoria individual. Manter.
- **TESTING_ORACLE (anotação necessária em builds sem test):**
  `src/models/a2/model/set_weights.rs:276,286` (`film_weight_count_cfg`,
  `film_bias_count_cfg`) e `src/testing/spectral.rs:56` (`fn median`) — usados
  apenas pelos submódulos de teste `#[path]`. Manter o `#[allow]`.

---

## C — Splitting de arquivos grandes (não-teste)

Classificação por valor de refatoração × risco de codegen (RT-safety crítico).

### C1 — `src/math/common/scalar_ref/dot.rs`  **[HIGH valor, ZERO risco codegen]**

- **Src:** 603. Código escalar de referência, usado apenas em testes/benchmarks.
  Sem intrínsecos SIMD, sem `#[inline(always)]` cross-module em risco.
- **Proposta:** split por dimensionalidade.
  - `dot.rs` (mantém): `dot_product_fallback`, `dot_product_bf16_fallback`,
    `dot_product_f32_native`, `dot_product_f32_native_kahan`,
    `dot_product_bf16_4x_fallback`, `mod kahan_dot_tests`, + `pub use` dos irmãos.
  - `dot_4x.rs`: `dot_product_4x_*` (scalar, dual, interleaved, accumulate e seus
    dual_accumulate).
  - `dot_8x16x.rs`: `dot_product_8x_*` e `dot_product_16x_*` (scalar, dual,
    accumulate).
- **Risco:** ZERO. Maior alavanca de legibilidade sem qualquer preocupação de
  codegen.

### C2 — `src/models/a2/grouped_conv1d.rs`  **[HIGH]**

- **Src:** 692. Mistura struct + impl de alto nível, funções de referência e
  kernels SIMD (`#[target_feature(enable="avx2,fma")]`).
- **Proposta:**
  - `grouped_conv1d.rs` (mantém): struct `A2GroupedConv1d`, impl `new` /
    `process_single_frame` + o include `#[path]` de teste existente.
  - `simd.rs`: `process_single_frame_depthwise_avx2`, `load_mixin_4`,
    `grouped_conv1d_single_frame_simd`.
  - `reference.rs`: `grouped_conv1d_single_frame_ref`, `grouped_conv1d_block_ref`.
- **Risco:** BAIXO. Kernels são `pub(crate) unsafe` já isolados; mover entre
  irmãos no mesmo crate preserva codegen. Replicar `#![allow(...)]` no `simd.rs`.

### C3 — `src/models/a2/model/dynamic/process.rs`  **[HIGH]**

- **Src:** 645. Mistura métodos de processamento "normais" com uma família de
  métodos `cascade_*` (~160 linhas, linhas 308–467).
- **Proposta:**
  - `process.rs` (mantém): `process`, `process_internal`, `rechannel_prescale`,
    `advance_head_ring`, `layer_forward_dispatch`, `head_finalize`, fn livre
    `process_frame_dyn`.
  - `cascade.rs`: `cascade_layer_loop`, `cascade_head_finalize`,
    `cascade_write_mono_input`, `cascade_write_residual_input`,
    `cascade_set_condition`, `cascade_seed_head_from_output`.
- **Risco:** ZERO. Split de `impl WaveNetA2Dyn` em dois arquivos é idiomático;
  métodos `cascade_*` são `#[inline(always)]` e `pub(crate)` — inlining
  intra-crate garantido. **Atenção:** evitar colisão de nome com possível
  `cascade.rs` existente em `models/a2/model/` (existe `cascade.rs` no mesmo dir —
  usar outro nome, ex. `process_cascade.rs` ou `cascade_dispatch.rs`). Confirmar
  antes de executar.

### C4 — `src/dsp/resampler.rs`  **[HIGH]**

- **Src:** 592. Mistura `DelayLine`, `ResamplerCore` e `NamResampler` + impls.
- **Proposta:**
  - `resampler.rs` (mantém): `NamResampler` struct + impl + `#[path]` teste.
  - `delay_line.rs`: `DelayLine` struct + impl (~48 linhas).
  - `core.rs`: `ResamplerCore` struct + impl (~237 linhas).
- **Risco:** ZERO. `DelayLine::push` é `#[inline(always)]` — seguro entre irmãos.
  O dispatch de ISA já é feito por `match` em `process_static_*`, não afetado.
  **Ação combinada com B3** (remoção do `phase_type`) — executar B3 antes ou junto.

### C5 — `src/math/gemm/gemv/f16_avx2_specialized.rs`  **[MEDIUM]**

- **Src:** 612. Costura clara: 6 kernels `fused_add_gemv_avx2_*` (290) + 6
  `gemv_overwrite_avx2_*` (270), separados por banner de comentário, com 2 helpers
  privados compartilhados (`load_partial_ymm`, `store_partial_ymm`).
- **Proposta:** `f16_avx2_fused.rs` + `f16_avx2_overwrite.rs`; o arquivo atual
  retém os helpers e re-exporta.
- **Risco:** BAIXO. Dispatch por ponteiro-de-função (não trait monomorfizado);
  helpers já `#[inline(always)]`; cada fn tem `#[target_feature]` próprio.

### C6 — `src/math/activations/tanh/high_fidelity.rs`  **[MEDIUM]**

- **Src:** 573. Costura por ISA: AVX2 + scalar (linhas 1–358) vs AVX-512 (360–572).
- **Proposta:** `high_fidelity_avx512.rs` com todos os kernels AVX-512; o arquivo
  atual retém AVX2+scalar+teste e `pub use high_fidelity_avx512::*;`.
- **Risco:** BAIXO. Cada `#[target_feature]` é compilado isoladamente; constantes
  polinomiais são `const` (sempre inlined).

### C7 — `src/math/dsp/fft.rs`  **[MEDIUM]**

- **Src:** 572. `FftFloat` trait + impls + `FftPlanner` (~234) vs `RfftPlanner`
  (~203).
- **Proposta:** `rfft.rs` com `RfftPlanner<T>` + impl; `fft.rs` mantém trait +
  `FftPlanner` + teste. `rfft.rs` importa `use super::fft::{FftFloat, FftPlanner};`.
- **Risco:** BAIXO. `RfftPlanner::process_*` é Rust escalar puro (sem intrínsecos
  diretos); SIMD está em `FftPlanner::run_butterflies`, que permanece em `fft.rs`.

### C8 — `src/models/linear.rs`  **[MEDIUM] (+ extração de teste A1)**

- **Src:** 333. Construção/seleção de modo vs processamento/lifecycle.
- **Proposta:**
  - `linear.rs` (mantém): `LinearMode`, `LinearModel`, `largest_power_of_two_le`,
    `select_partition_size`, impl `new`/`resolve_mode`, impl `Sealed`, impl
    `NamModel`.
  - `process.rs`: impl `process_sample`/`process`/`prewarm`/`reset`.
- **Risco:** MÉDIO. `process_sample` é `#[inline(always)]` + `unsafe` (RT
  hot-path). Inlining intra-crate é garantido, mas recomenda-se **verificação de
  benchmark** (`benches/inference_bench`) antes/depois para confirmar ausência de
  regressão de codegen. Combinar com **A1**.

### C9 — `src/models/linear_fft.rs`  **[MEDIUM] (+ extração de teste A2)**

- **Src:** 397. Construção vs processamento.
- **Proposta:** `linear_fft.rs` (mantém `LinearFftState`, `Debug`, `new`,
  `h_fdl_re_partition`); `process.rs` (`reset`, `process_tail_block`).
- **Risco:** BAIXO. `process_tail_block` é `pub fn` regular (não `#[inline]`),
  chamado de `LinearModel::process_sample`; call site inalterado. Combinar com
  **A2**.

### C10 — `src/models/static_model.rs`  **[MEDIUM]**

- **Src:** 577. Query/classifier methods vs `impl NamModel`.
- **Proposta:** `nam_model.rs` com `impl NamModel for StaticModel`; arquivo atual
  mantém query methods + `clone_condition_dsp`.
- **Risco:** ZERO. Camada de delegação pura; ambos importam as variantes do enum.

### C11 — `src/clap/gui/ui/zones/identity.rs`  **[MEDIUM]**

- **Src:** 506. `draw_zone1_identity` (~383) vs helpers `spawn_file_dialog`/
  `spawn_ir_file_dialog` (~105).
- **Proposta:** `file_dialogs.rs` com os dois `spawn_*`.
- **Risco:** ZERO. Código de GUI, fora da thread RT.

### C12 — `src/dsp/oversample.rs`  **[MEDIUM]**

- **Src:** 472. `X2Stage` (~140) isolável.
- **Proposta:** `stage.rs` com `X2Stage` struct + impl; arquivo atual mantém o
  resto + `#[path]` teste.
- **Risco:** MÉDIO. `X2Stage::upsample`/`downsample` são hot-path RT **sem**
  `#[inline]`. **Pré-condição:** adicionar `#[inline]` (ou `#[inline(always)]`) a
  esses métodos ao mover, para preservar opportunity de inlining no call site.
  Verificar com benchmark.

### C13 — `src/models/slimmable.rs`  **[MEDIUM]**

- **Src:** 447. Trait vs infraestrutura de slicing.
- **Proposta:** `slicing.rs` com as 7 funções `slice_*` / `clone_wavenet_*` /
  `try_slimmable_rebuild_single`; arquivo atual mantém trait `SlimmableModel` +
  `#[path]` teste.
- **Risco:** ZERO. Funções de construction-time (thread principal, pipeline SPSC
  GC); nunca na thread RT.

### C14 — Não tocar (LOW/NONE)  **[nenhuma ação]**

| Arquivo                                | Class | Justificativa                                                                                                                                             |
| -------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/math/common/avx2_impl.rs`         | LOW   | `impl SimdMath for Avx2Math` monolítico, ~50 delegadores `#[inline(always)]`; split fragmentaria o trait e arriscaria codegen da camada de dispatch SIMD. |
| `src/math/common/traits.rs`            | NONE  | Definição única de `trait SimdMath` — não pode ser dividida entre módulos.                                                                                |
| `src/math/gemm/gemv/f32_avx512.rs`     | LOW   | Dois kernels SIMD paralelos (bias/no-bias); conceitualmente pares, mantidos juntos para comparação.                                                       |
| `src/math/gemm/gemv/f32_avx2.rs`       | LOW   | Idem, padrão paralelo a `f32_avx512`.                                                                                                                     |
| `src/math/common/aligned.rs`           | LOW   | 368 src; split de `Zeroable`/`Aligned64`/`AlignedVec` em 2-3 arquivos é over-fragmentação. **(Apenas extração de teste A4.)**                             |
| `src/math/gemm/gemm_batch/avx512.rs`   | LOW   | 3 kernels; irmãos AVX2 já existem; padrão monolítico consistente.                                                                                         |
| `src/math/dsp/fft_radix4.rs`           | NONE  | 305 src; struct+impl único; artefato de pesquisa. **(Apenas extração de teste A3.)**                                                                      |
| `src/dsp/pipeline/stages/inference.rs` | LOW   | ~408 linhas num único `run_inference`; helpers têm 1 call site cada — extrair criaria módulo artificial.                                                  |
| `src/dsp/adaptive.rs`                  | NONE  | O doc-comment do próprio arquivo (9–24) justifica a coesão: FSM de histerese + crossfade + threshold + redução de camada = unidade algorítmica.           |
| `src/models/a2/conv1d_ch3/simd.rs`     | LOW   | Kernel único consumido por `layer_forward_ch3_block`; extrair geraria arquivo de 1 consumidor.                                                            |
| `src/models/a2/conv1d_ch8/simd.rs`     | LOW   | 3 funções em níveis de abstração acoplados.                                                                                                               |
| `src/models/a2/film.rs`                | LOW   | 325 src; seções pequenas (40–130); split geraria fragmentos <130.                                                                                         |
| `src/loader/nam_json/validation.rs`    | LOW   | 480 src; visitantes agrupados mas pequenos; `validation_test.rs` já é irmão.                                                                              |

---

## D — Reorganização de diretórios

### D1 — Diretórios `test_files/` violam a convenção de colocação  **[REFACTOR_CANDIDATE]**

Regra (`testing.md` §1): testes unitários em arquivo irmão `<module>_test.rs`, e
testes de integração em `tests/` raiz. O padrão `test_files/` é uma indireção
não-padronizada e o nome é enganoso (sugere "arquivos de teste" genéricos, mas
são submódulos de teste unitário incluídos via `#[path]`).

**D1a — `src/dsp/pipeline/test_files/`** (3 arquivos):

| Arquivo          | Incluído de                                                    |
| ---------------- | -------------------------------------------------------------- |
| `bypass_test.rs` | `pipeline_test.rs` via `#[path = "test_files/bypass_test.rs"]` |
| `gate_test.rs`   | `pipeline_test.rs`                                             |
| `dither_test.rs` | `pipeline_test.rs`                                             |

- **Ação:** achatar para irmãos de `pipeline_test.rs`, renomeando para
  desambiguar (ex. `pipeline_bypass_test.rs`, `pipeline_gate_test.rs`,
  `pipeline_dither_test.rs`); atualizar os `#[path]` em `pipeline_test.rs`;
  remover o diretório vazio.
- **Risco:** ZERO.

**D1b — `src/models/wavenet/test_files/`** (7 arquivos):
6 são submódulos de teste (`conv1d_dyn_test.rs`, `conv1d_test.rs`, `dense_test.rs`,
`dynamic_parity_test.rs`, `post_stack_head_integration_test.rs`,
`wavenet_sub_test.rs`); **`ch12_diagnostic.rs` NÃO é teste** — é utilitário de
diagnóstico em runtime, mal colocado num diretório `test_files/`.

- **Ação:**
  - Mover os 6 arquivos de teste para irmãos de `wavenet_test.rs` (prefixo
    `wavenet_*` se necessário para evitar colisão) e atualizar `#[path]`.
  - Mover `ch12_diagnostic.rs` para `src/models/wavenet/ch12_diagnostic.rs` (fora
    de `test_files/`); atualizar o `#[path]` correspondente (ou tornar `mod`
    normal se deixar de ser cfg(test)).
  - Remover o diretório vazio.
- **Risco:** BAIXO. Verificar se `ch12_diagnostic.rs` é incluído sob `#[cfg(test)]`
  ou como módulo normal — se for runtime, a inclusão muda de natureza. Auditar o
  `#[path]` atual antes de mover.

### D2 — `math/gemm/dot.rs` é arquivo flat irmão de `dot_4x/`, `dot_8x/`, `dot_16x/`  **[INCONSISTENCY]**

- `dot.rs` contém um único `dot_product_avx2` (148 linhas) e não segue o padrão de
  diretório dos `dot_Nx/`. Nome ambíguo ("dot" vs "dot_Nx").
- **Ação:** renomear `dot.rs` → `dot_basic.rs` e atualizar `pub mod dot;` em
  `math/gemm/mod.rs`. Alternativa: migrar para `dot_4x/scalar.rs` (já tem variants
  escalares) — mas isso altera escopo público; **preferir renomear**.
- **Risco:** ZERO (renomeio mecânico).

### D3 — `avx2_impl.rs` flat vs `avx512/` diretório  **[INCONSISTENCY — manter]**

- `avx2_impl.rs` (1022) é arquivo flat; `avx512/` é diretório com 5 submódulos +
  2 flats. Mesmo `SimdMath` trait, organização divergente.
- **Decisão:** **NÃO refatorar** aqui. Split de `avx2_impl.rs` foi classificado
  LOW em C14 (risco de codegen no dispatch SIMD). A inconsistência estrutural é
  documentada mas aceita como trade-off de performance. Registrar como dívida
  técnica de baixa prioridade.

### D4 — `models/a2/model/cascade.rs` flat vs `dynamic/`, `static/` dirs  **[INCONSISTENCY — opcional]**

- `cascade.rs` (241) é flat; `dynamic/` e `static/` são diretórios paralelos.
- **Ação (opcional):** promover `cascade.rs` → `cascade/mod.rs` (sem sub-split).
  **Atenção:** C3 propõe criar `cascade.rs` (ou `process_cascade.rs`) em
  `dynamic/` — nomes diferentes, sem colisão, mas merece atenção ao executar.
- **Risco:** ZERO. Valor organizacional baixo (241 linhas é gerenciável flat);
  deixar como dívida opcional.

### D5 — `math/activations/tanh/` dir vs `sigmoid.rs` flat  **[INCONSISTENCY — opcional]**

- `tanh/` é a única ativação com subdiretório; `sigmoid.rs` (252) também tem
  variante high_fidelity mas é flat.
- **Ação (opcional):** promover `sigmoid.rs` → `sigmoid/{mod.rs, production.rs,
  high_fidelity.rs}` espelhando `tanh/`, **ou** achatar `tanh/` em `tanh.rs`.
  Preferência: **promover sigmoid** para consistência (maior clareza futura).
- **Risco:** ZERO. Deixar como dívida opcional.

### D6 — Testes `conv1d_ch3_test.rs`/`conv1d_ch8_test.rs` no diretório pai  **[INCONSISTENCY — menor]**

- Incluídos via `#[path = "../conv1d_ch3_test.rs"]` a partir de
  `conv1d_ch3/mod.rs`/`conv1d_ch8/mod.rs`. Funcional, mas diverge do padrão
  (testes dentro do dir do módulo).
- **Ação (opcional):** mover para dentro de `conv1d_ch3/`/`conv1d_ch8/` e ajustar
  `#[path]`. Valor baixo.
- **Risco:** ZERO.

---

## E — Resumo consolidado por Epic

### Epic 1 — Extração de testes inline (conformidade `testing.md`)

**Risco: ZERO.** Movimentação mecânica de bloco `#[cfg(test)] mod tests` → irmão
`_test.rs`. Validar com `utils/tests-quick.sh`.

- **T1.1** A1 — `linear.rs` → `linear_test.rs`
- **T1.2** A2 — `linear_fft.rs` → `linear_fft_test.rs`
- **T1.3** A3 — `fft_radix4.rs` → `fft_radix4_test.rs`
- **T1.4** A4 — `aligned.rs` → `aligned_test.rs`

### Epic 2 — Remoção de código morto verificado

**Risco: ZERO (itens verificados).** Validar com `cargo check` (sem novos warnings
indesejados) + `utils/tests-quick.sh`.

- **T2.1** B1 — remover `load_4/8/16_accums` (`conv_input.rs`)
- **T2.2** B2 — remover `weight_f16c_to_f64` (`reference_oracle.rs`)
- **T2.3** B3 — remover campo `phase_type` + getter (`resampler.rs`); revisar
  `new()`.
- **T2.4** B4 — remover `log_spaced_tones_dense` (`resampler_test.rs`)
- **T2.5** B5 — remover 5 constantes mortas (`perceptual.rs`); depois avaliar
  remoção do `#![allow(dead_code)]` do módulo.
- **T2.6** B6 — remover anotações *stale* (`state.rs:122`,
  `checked_arith.rs:37`, `topology/mod.rs:16`); **manter** `context.rs:87-97`.

### Epic 3 — Achatamento dos diretórios `test_files/`

**Risco: ZERO/BAIXO.** Movimentação de arquivos + atualização de `#[path]`.

- **T3.1** D1a — achatar `dsp/pipeline/test_files/` (3 arquivos)
- **T3.2** D1b — achatar `models/wavenet/test_files/` (6 testes) + mover
  `ch12_diagnostic.rs` (auditar natureza cfg/test antes)

### Epic 4 — Splitting de baixo risco (escalar/GUI/construction-time)

**Risco: ZERO.** Sem impacto em codegen de hot-path.

- **T4.1** C1 — `scalar_ref/dot.rs` → `dot_4x.rs` + `dot_8x16x.rs` (maior alavanca)
- **T4.2** C10 — `static_model.rs` → `nam_model.rs`
- **T4.3** C13 — `slimmable.rs` → `slicing.rs`
- **T4.4** C11 — `identity.rs` → `file_dialogs.rs`
- **T4.5** D2 — renomear `gemm/dot.rs` → `dot_basic.rs`

### Epic 5 — Splitting de risco MÉDIO (RT/SIMD/FFT) — com verificação de benchmark

**Risco: MÉDIO.** **Pré-requisito:** rodar `benches/inference_bench` (e
`long_inference_bench` sob feature `long_bench` se aplicável) antes/depois de
cada tarefa e confirmar ausência de regressão de codegen dentro do ruído.

- **T5.1** C2 — `grouped_conv1d.rs` → `simd.rs` + `reference.rs`
- **T5.2** C3 — `dynamic/process.rs` → `process_cascade.rs` (confirmar nome p/ não
  colidir com `model/cascade.rs` — ver D4)
- **T5.3** C4 — `resampler.rs` → `delay_line.rs` + `core.rs` (após T2.3)
- **T5.4** C5 — `f16_avx2_specialized.rs` → `f16_avx2_fused.rs` + `f16_avx2_overwrite.rs`
- **T5.5** C6 — `high_fidelity.rs` → `high_fidelity_avx512.rs`
- **T5.6** C7 — `fft.rs` → `rfft.rs`
- **T5.7** C12 — `oversample.rs` → `stage.rs` (**adicionar `#[inline]` em
  `X2Stage::upsample/downsample` ao mover**) + benchmark
- **T5.8** C8 — `linear.rs` → `process.rs` (após T1.1) + benchmark
- **T5.9** C9 — `linear_fft.rs` → `process.rs` (após T1.2) + benchmark

### Epic 6 — Reorganização estrutural opcional (baixo valor)

**Risco: ZERO, mas baixo valor organizacional.** Executar só se houver tempo/apetite.

- **T6.1** D4 — `cascade.rs` → `cascade/mod.rs` (opcional)
- **T6.2** D5 — promover `sigmoid.rs` → `sigmoid/` (opcional)
- **T6.3** D6 — mover `conv1d_ch3/ch8_test.rs` para dentro dos dirs (opcional)

---

## Sequência recomendada de execução

1. **Epic 1** (T1.1–T1.4) — estabelece conformidade de testes; base segura.
2. **Epic 2** (T2.1–T2.6) — limpa código morto; reduz ruído para próximos epics.
3. **Epic 3** (T3.1–T3.2) — resolve violações de convenção de diretório.
4. **Epic 4** (T4.1–T4.5) — splits de maior valor e zero risco de codegen.
5. **Epic 5** (T5.1–T5.9) — splits de risco médio, **cada tarefa isolada** com
   benchmark antes/depois e `utils/lints.sh` + `utils/tests-quick.sh`.
6. **Epic 6** (opcional) — só se demandado explicitamente.

**Gate de qualidade após cada Epic:** `utils/lints.sh` (fmt+SPDX+check+clippy) e,
na validação final de cada Epic, `utils/tests-quick.sh` (uma execução por tarefa
de IA). `utils/tests-long.sh` é **proibido** em tarefa de IA — solicitar ao
operador humano quando necessário.

## Pontos de atenção cruzados (riscos de colisão/execução)

- **C3 vs D4:** `dynamic/process.rs` split pode criar `cascade.rs` em
  `dynamic/`, enquanto `model/cascade.rs` já existe um nível acima. Usar
  `process_cascade.rs` para o split de C3 e deixar D4 separado.
- **C8/C9 vs Epic 1:** executar extração de teste (T1.1/T1.2) **antes** do split
  de processamento (T5.8/T5.9), senão o bloco de teste inline acaba no arquivo
  errado.
- **C4 vs B3:** remover `phase_type` (T2.3) **antes** do split de `resampler.rs`
  (T5.3) para que o split já comece limpo.
- **C12 (oversample):** o `#[inline]` adicionado em `X2Stage::upsample/downsample`
  é a única mudança "não-puramente-mecânica" de todo o plano — é a adição de um
  atributo para **preservar** o comportamento de codegen, não alterá-lo. Validar
  com benchmark.
- **B5/B7:** não remover `#![allow(dead_code)]` de `perceptual.rs`/`wav.rs` à
  cega — há helpers internos que podem ser mortos; reauditar antes.

---

*Arquivo gerado pela skill `planejador-arquiteto` a partir dos achados da skill
`refatora-rust`. O `TODO-sprints.md` só deve ser criado mediante solicitação
explícita.*
