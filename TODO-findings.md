<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Refatoração estrutural da pasta `tests/`

Documento de findings gerado pela skill `refatora-rust` (foco em `/tests`), com plano de correções
seguras delegado à skill `planejador-arquiteto`. O escopo é **exclusivamente estrutural**: nenhuma
lógica ou algoritmo deve ser alterado. Regressões são estritamente proibidas.

Agente de IA: estes findings são de baixíssimo risco e prontos para execução. Cada um traz a
evidência de coleta, a análise de segurança e a proposta de correção. Siga a seção "Epics" para
sequência recomendada.

---

## Metodologia de inspeção

Foram inspecionados os **73 arquivos `.rs`** da pasta `tests/` (excluindo `tests/fixtures/`, que
contém dados binários e fontes C++ de referência, não código Rust a refatorar). A inspeção
cruzou:

- Contagem de linhas por arquivo (`wc -l`).
- Árvore de módulos declarada nos agregadores de topo (`clap.rs`, `models.rs`, `parity.rs`,
  `perf_soak.rs`, `rt_constraints.rs`) via `#[path = "..."] mod ...;`.
- Detecção de arquivos órfãos (não referenciados por nenhum agregador).
- Busca por código morto/Inutilizado (`#[deprecated]`, chamadores zero, `#![allow(dead_code)]`
  mascarando itens sem uso, código comentado).
- Consistência de cabeçalhos SPDX.
- Marcadores `TODO/FIXME/DEPRECATED` (para distinguir texto inútil de referências legítimas a
  docs de rastreamento).

---

## Saúde geral do módulo

**Boa.** A organização é lógica e modular:

```text
tests/
├── clap.rs            (+ clap/        — 4 testes, feature-gated clap-plugin)
├── common/mod.rs      (+ 18 helpers   — módulo compartilhado por 5 test binaries)
├── loom_tests.rs      (standalone, #![cfg(loom)])
├── models.rs          (+ models/       — 29 testes de modelo/DSP)
├── parity.rs          (+ parity/       — 7 testes de paridade C++/oráculo/ISA)
├── perf_soak.rs       (+ perf_soak/    — 5 testes de soak/concorrência)
└── rt_constraints.rs  (+ rt_constraints/ — 5 testes de RT-safety/heap-audit)
```

Cada `.rs` de topo é um test binary independente; `common/` é compartilhado via `mod common;`
(resolvido para `common/mod.rs`). **Não há arquivos órfãos** — todos os 73 arquivos são
alcançáveis. Cabeçalhos SPDX **100% consistentes** (73/73).

Os findings abaixo são correções pontuais de baixo risco, não reestruturação ampla.

---

## F1 — `mod common;` morto em `tests/clap.rs` [DONE]

**Arquivo:** `tests/clap.rs:4`

**Fato:** O agregador `clap.rs` declara `mod common;`, compilando todo o módulo compartilhado
de helpers (18 submódulos, ~2.900 linhas no total) dentro do test binary `clap`. Contudo,
**nenhum dos 4 arquivos em `tests/clap/`** referencia `common`:

```shell
$ grep -rnE '\bcommon\b' tests/clap/
(nenhuma ref a 'common' em clap/)
```

Os testes `clap/*` usam diretamente `clack_host::prelude::*` e `std::*`; não consomem nenhum
helper de `common`.

**Análise de segurança (zero risco):**

- `mod common;` é uma declaração de módulo local. Se nada dentro do binary referencia itens de
  `common`, remover a declaração **não altera nenhum comportamento de teste** — apenas deixa de
  compilar código morto no binary `clap`.
- Os demais 4 agregadores (`models.rs`, `parity.rs`, `perf_soak.rs`, `rt_constraints.rs`) **sim**
  usam `common::` e **devem manter** a declaração. Confirmado por inspeção cruzada.
- `clap.rs` não declara `#[global_allocator]` (apenas `models.rs`/`perf_soak.rs`/`rt_constraints.rs`
  o fazem, e dependem de `common::alloc_audit::CountingAllocator`), então não há dependência
  indireta via allocator.

**Proposta de correção:**

1. Remover a linha `mod common;` de `tests/clap.rs:4`.
2. Verificar: `cargo test --test clap --features clap-plugin --no-run` (deve compilar sem
   warning de módulo não usado e sem erro).
3. Opcional: `cargo clippy --tests --features clap-plugin` para confirmar ausência de
   regressões de lint.

**Risco:** Nenhum. Mudança puramente estrutural; reduz o grafo de compilação do binary `clap`.

---

## F2 — `generate_stress_signal()` deprecated sem chamadores em `tests/common/signals.rs` [DONE]

**Arquivo:** `tests/common/signals.rs:4-11`

**Fato:** O arquivo define um wrapper deprecated:

```rust
#![allow(dead_code)]

use nam_rs::testing::stress::generate_stress_signal_v1;

#[deprecated(since = "1.5.0", note = "Use `generate_stress_signal_v1()` directly")]
pub fn generate_stress_signal() -> Vec<f32> {
    generate_stress_signal_v1()
}

pub use nam_rs::testing::aliasing::generate_sine_440hz;
```

Busca exaustiva no repositório inteiro confirma **zero chamadores**:

```shell
$ rg -nE 'generate_stress_signal\(\)' --glob '*.rs' .
./tests/common/signals.rs:9:pub fn generate_stress_signal() -> Vec<f32> {
```

O único hit é a própria definição. Enquanto isso, `common/mod.rs` reexporta
`generate_stress_signal_v1` e `generate_stress_signal_v2_default` **diretamente da library crate**
(linhas 33-34), não via `signals`. O outro item do arquivo — o re-export
`generate_sine_440hz` — tem **72 usos** confirmados e deve ser preservado.

**Análise de segurança (baixo risco):**

- A versão atual do crate é **3.0.0** (`Cargo.toml`). A deprecation foi marcada em `1.5.0`.
  Decorridos dois majors (1.5.0 → 2.x → 3.0.0), a janela de deprecation está amplamente
  encerrada — o wrapper é candidata legítima a remoção.
- `#![allow(dead_code)]` na linha 4 existe exclusivamente para mascarar este item sem uso; após
  removê-lo, o `allow` também pode ser retirado.
- O `use nam_rs::testing::stress::generate_stress_signal_v1;` (linha 6) só alimenta o wrapper
  deprecated; sem ele, torna-se import morto e deve ser removido junto.

**Proposta de correção:**

1. Remover de `tests/common/signals.rs`:
   - a diretiva `#![allow(dead_code)]` (linha 4);
   - o `use ... generate_stress_signal_v1;` (linha 6);
   - o bloco `#[deprecated]` + `pub fn generate_stress_signal()` (linhas 8-11).
2. Preservar: o cabeçalho SPDX e o `pub use nam_rs::testing::aliasing::generate_sine_440hz;`
   (linha 13).
3. Verificar: `cargo test --test models --no-run` (binary que mais consome `common::signals::*`
   via `pub use signals::*` em `common/mod.rs:36`); `cargo clippy --tests`.

**Risco:** Baixo. Wrapper deprecated sem chamadores; a versão canônica (`_v1`) permanece
disponível via `common/mod.rs` e diretamente da library crate. Nenhuma lógica alterada.

**Caveat / decisão do PO:** Se houver política de manter wrappers deprecated por mais um ciclo
mesmo sem chamadores internos, este finding pode ser diferido. Recomenda-se remoção dada a
distância de versão (3.0.0).

---

## F3 — Comentários com paths stale em `utils/tests-long.sh` [DONE]

**Arquivo:** `utils/tests-long.sh:526` e `:541`

**Fato:** Dois comentários referenciam paths de layout "plano" (`tests/<name>.rs`) que não
correspondem mais à realidade do layout modular atual:

```shell
utils/tests-long.sh:526:    # in tests/isa_parity.rs) — safe to run unconditionally on any machine.
utils/tests-long.sh:541:    # of tests/gate_fsm_proptest.rs, covers the DynamicHysteresis reversal
```

Os paths reais são:

- `tests/isa_parity.rs` → **`tests/parity/isa_parity.rs`** (submódulo do binary `parity`).
- `tests/gate_fsm_proptest.rs` → **`tests/models/gate_fsm_proptest.rs`** (submódulo do binary
  `models`).

**Análise de segurança (zero risco):** São apenas comentários; o `_test_flag` do script resolve
os nomes via `--test <entry> <entry>::<test>` e o `--test loom_tests` explícito, então a
execução não é afetada. É puramente dívida de documentação.

**Nota de escopo:** Este arquivo está em `utils/`, não em `tests/`. Foi detectado durante a
inspeção porque os comments citam paths sob `tests/`. Incluído aqui por completude; a edição
fica a critério do escopo definido pelo PO.

**Proposta de correção:**

1. Em `utils/tests-long.sh:526`, substituir `tests/isa_parity.rs` por
   `tests/parity/isa_parity.rs`.
2. Em `utils/tests-long.sh:541`, substituir `tests/gate_fsm_proptest.rs` por
   `tests/models/gate_fsm_proptest.rs`.

**Risco:** Nenhum (apenas comentários).

---

## F4 — Referências de rastreador (sprint/tarefa) stale em `tests/*.rs` (118 ocorrências) [DONE]

**Escopo:** 12 arquivos `.rs` sob `tests/` (excluindo `fixtures/`).

**Fato:** Há **118 ocorrências** de identificadores de rastreador de sprint/tarefa embutidos em
comentários de código. Os padrões encontrados:

- `Tarefa T?<n>` (ex.: `Tarefa 1.2`, `Tarefa 3.1`, `Tarefa 8.6`)
- `T<n>.<n>` (ex.: `T8.2`, `T4.3`, `T3.3`, `T4.7`)
- `SQ<n>.<n>` (ex.: `SQ5.5`)
- `S<n>.T0<n>` (ex.: `S2.T03`, `S10.3`)
- `α<n>.<n>` (ex.: `α2.2`, `α2.3`)
- `F-<n>` / `F<n>` / `F-X<n>` (ex.: `F-1`, `F5`, `F6`, `F-X2`)

**Distribuição por arquivo (top 12):**

| Arquivo                                 | Ocorrências |
| --------------------------------------- | ----------- |
| `tests/parity/reference_oracle_f64.rs`  | 25          |
| `tests/models/threshold_calibration.rs` | 22          |
| `tests/common/validation.rs`            | 22          |
| `tests/models/golden_vectors.rs`        | 17          |
| `tests/common/constants.rs`             | 9           |
| `tests/parity/cpp_parity.rs`            | 8           |
| `tests/models/proptest_parsers.rs`      | 7           |
| `tests/models/container_slimmable.rs`   | 7           |
| `tests/models/activation_precision.rs`  | 6           |
| `tests/loom_tests.rs`                   | 3           |
| `tests/models/zero_alloc_infer.rs`      | 2           |
| `tests/models/meta_coherence.rs`        | 2           |

**Análise de segurança (baixo risco):** São comentários e doc-comments — remover o identificador
do rastreador não altera lógica. A correção é **cirúrgica**: remover apenas o token de ID (ex.:
`Tarefa 3.1`, `SQ5.5:`, `(T4.1)`) preservando o texto descritivo adjacente que explica o *porquê*
(ex.: "post-weight-dequantization — near-bit-exact", "MR-STFT hard gate at 44.1/48 kHz").

Exemplo de correção em `tests/common/constants.rs:16`:

- Antes: `// ── T8.3 + SQ5.5: Re-derived fidelity gates (post-weight-dequantization) ──`
- Depois: `// ── Re-derived fidelity gates (post-weight-dequantization) ──`

**Reconciliação com `refatora-rust` (turno anterior):** O turno anterior classificou estes IDs como
"referências legítimas a docs de rastreamento" (não-issue §6). A skill `refatora-doc` é mais
específica: identifica "sprint X" / IDs de tarefa como referências irrelevantes que não contribuem
para o entendimento do código. **Prevalece o mandato da skill ativa (`refatora-doc`)** — os IDs
devem ser removidos, mas o conteúdo descritivo e as medições (`// Measured: SNR=...`) são
preservados integralmente.

**Risco:** Baixo. Comentário-only; sem mudança de lógica. Risco residual: inconsistência se a
limpeza for parcial — deve ser um sweep completo e atômico por arquivo.

---

## F5 — Referências dangling a `TODO-findings.md` (auditoria não persistida) no README [DONE — Opção B]

**Arquivo:** `tests/fixtures/README.md:141` e `:813-814`

**Fato:** O README referencia conteúdo de auditoria que **não existe** no `TODO-findings.md`:

- Linha 141: "See `TODO-findings.md` **F-X2** and **Sprint 3** completion notes for the audit
  that closed the gap."
- Linhas 813-814: "See **`TODO-findings.md`** at the repository root for the full findings and
  the remediation **Épicos (A–F)**."

O `TODO-findings.md` atual contém findings estruturais (F1/F2/F3, Epics E1/E2) — não a auditoria
do pipeline de geração de goldens (F-X2, Épicos A–F) que o README espera. A auditoria
(`revisor-auditor`, "Compliance and Parity Auditor") descrita no README §"Layer 0" aparentemente
nunca foi persistida em arquivo.

**Análise de segurança:** Sem risco de regressão (documentação). O problema é que a referência
aponta para conteúdo inexistente, induzindo o leitor a buscar findings que não estão lá.

**Proposta de correção (decisão de PO necessária):**

- **Opção A (preferida):** Solicitar à skill `revisor-auditor` que persista os findings da
  auditoria do `golden_gen_build.sh` (catalog coverage, `pipefail`/`errexit`, supply-chain
  `NeuralAmpModelerPlugin`) como uma seção dedicada em `TODO-findings.md`, usando os
  identificadores F-X2 / Épicos A–F já citados pelo README. Assim a referência torna-se válida.
- **Opção B:** Se a auditoria não deve ser persistida, remover as referências dangling
  (F-X2, Sprint 3, Épicos A–F) do README, preservando a descrição substantiva do que a
  auditoria encontrou (linhas 805-812).

**Risco:** Nenhum (documentação). Exige decisão sobre qual opção aplicar.

---

## F6 — Doc-comments `///` ausentes em itens públicos de `tests/common/`

**Arquivo:** `tests/common/` (módulo compartilhado por 5 test binaries)

**Fato:** Alguns itens públicos do módulo `common` carecem de doc-comment `///`, contrariando a
convenção da skill (`///` para itens públicos, `//` para comentários estruturais internos):

- `tests/common/manifest.rs:6` — `pub struct ManifestEntry` sem `///`.
- `tests/common/precision.rs:12` — `pub struct PrecisionGuard` e `impl` sem `///` (apenas o
  campo `_lock` é não-público; `new()` é público).
- `tests/common/metrics.rs:6,23` — `pub fn compute_mse`, `compute_max_abs_error` sem `///`
  (apenas `compute_esr` tem).
- `tests/common/constants.rs:6-47` — constantes `pub const` documentadas via `//` (comentário
  estrutural) em vez de `///` (doc de item público).

**Análise de segurança:** Zero risco (apenas adição de doc-comments). Não altera compilação
nem lógica.

**Proposta de correção:** Converter os `//` explicativos acima de itens `pub` em `///` onde
aplicável, e adicionar `///` breve aos structs/fns públicos listados. Manter os blocos de
metodologia/medição como `//` quando forem notas internas (não doc de API).

**Risco:** Nenhum (apenas comentários).

---

## Não-issues verificados (NÃO refatorar)

Estes pontos foram inspecionados e **deliberadamente não são findings**. Documentados aqui para
evitar esforço futuro de refatoração desperdiçado:

1. **Arquivos de teste de integração grandes (>1000 linhas):** `golden_vectors.rs` (2331),
   `proptest_parsers.rs` (1767), `reference_oracle_f64.rs` (1655), `cpp_parity.rs` (1477),
   `nam_infer_test.rs` (1119), `a2_loader.rs` (1069), `validation.rs` (1036).
   A regra de 300 linhas em `.agents/rules/testing.md` §1 aplica-se a **testes unitários inline
   em arquivos de `src/`**, não a testes de integração em `tests/`. Estes são test suites
   coesos por domínio; particioná-los reorganizaria a topologia de test binaries e adicionaria
   risco de regressão sem benefício estrutural. Manter como estão.

2. **`#![allow(dead_code)]` / `#![allow(unused_imports)]` em `common/*`:** Legítimo. O módulo
   `common/` é compilado em **5 test binaries distintos** com features diferentes (`clap-plugin`,
   `heap-audit`, `standalone`, etc.). Um item usado por apenas um binary parece "morto" para os
   demais. Os `allow` são a forma correta de suprimir esses falsos-positivos. Não remover.

3. **Código comentado em `tests/parity/cpp_parity.rs:1210-1222` e `:1292-1302`:** Testes
   intencionalmente desabilitados (`wavenet_a2_max`, §7.1), com comentário explicativo apontando
   para o guard fail-closed `is_disabled_broken_a2_flagship` e a lacuna de paridade §4.4. São
   placeholders úteis para reativação futura, não "texto inútil". Manter.

4. **`tests/proptest-regressions/`:** Persistência de seeds de falha do `proptest` (best practice
   do framework, via `FileFailurePersistence::SourceParallel`). Rastreabilidade de regressões
   em CI. Manter.

5. **`tests/loom_tests.rs` standalone:** Binário `#![cfg(loom)]` isolado, bem estruturado em 3
   seções (handshake, GC overflow, double-buffering DspBridge). Referenciado por
   `utils/tests-long.sh` via `--test loom_tests`. Estrutura sólida — nada a refatorar
   estruturalmente. (Os IDs `T8.2`/`T8.3`/`T8.4` nos cabeçalhos de seção são alvo do F4 —
   limpeza de doc, não de estrutura.)

6. **Referências a arquivos `TODO-*.md`:** As menções a `TODO-sprints.md`,
   `TODO-wavenet_a2_max.md`, `TODO-convnet_parity.md` são **links legítimos** a docs de
   rastreamento versionados — manter. **Distinção importante:** os *IDs de tarefa/sprint
   inline* embutidos em comentários de código (`T8.2`, `Tarefa 3.1`, `SQ5.5`, etc.) são alvo
   do F4 (limpeza de doc), não desta isenção. O `refatora-rust` (turno anterior) tratou-os como
   não-issues; o `refatora-doc` (turno atual) é mais específico e determina sua remoção.

7. **Cabeçalhos SPDX:** 100% consistentes em todos os 73 arquivos `.rs` (licença + copyright).
   Nada a ajustar.

---

## Epics

Agrupamento para execução otimizada — sequência por risco crescente, do mais seguro para o que
exige decisão de PO.

### Epic E1 — Limpeza de código morto confirmado (risco zero)

Engloba remoções com evidência conclusiva de ausência de uso, sem qualquer decisão de produto
pendente.

- **E1.T01 (← F1):** Remover `mod common;` morto de `tests/clap.rs:4`. Validar com
  `cargo test --test clap --features clap-plugin --no-run`.
- **E1.T02 (← F3):** Atualizar os 2 comentários stale em `utils/tests-long.sh:526,541` para os
  paths modulares reais.

**Verificação de aceitação do Epic:** `cargo clippy --tests --all-features` sem novos warnings;
`utils/lints.sh` verde.

### Epic E2 — Remoção de wrapper deprecated (risco baixo, decisão de PO)

Depende de confirmação de que a janela de deprecation (marcada em 1.5.0, hoje em 3.0.0) pode ser
encerrada.

- **E2.T01 (← F2):** Remover `#![allow(dead_code)]`, o `use generate_stress_signal_v1` e o
  `#[deprecated] pub fn generate_stress_signal()` de `tests/common/signals.rs:4-11`, preservando
  o re-export `generate_sine_440hz`. Validar com `cargo test --test models --no-run` e
  `cargo clippy --tests`.

**Verificação de aceitação do Epic:** compilação limpa; nenhum teste deixa de passar/skipar;
busca `rg 'generate_stress_signal\(\)'` retorna vazio.

### Epic E3 — Limpeza de documentação de `/tests` (refatora-doc)

Engloba correções de documentação identificadas pela skill `refatora-doc`. As correções de
baixo risco e alta confiança já aplicadas neste turno; o sweep de tracker refs (F4) e os itens
que exigem decisão (F5) permanecem como tarefas.

- **E3.T01 (aplicado ← README):** Corrigidos 11 paths stale flat→modular e 3 comandos
  `--test` incorretos em `tests/fixtures/README.md`. Removidos 3 tracker refs óbvios
  ("F5 resolved in D.1.1", "Sprint 2: …", "(T4.1)").
- **E3.T02 (aplicado ← proptest_parsers):** Adicionado comentário documentando as 8 arms do
  `match pattern % 8` em `tests/models/proptest_parsers.rs:1438` (bloco de 143 linhas sem
  comentário inline).
- **E3.T03 (← F4):** Sweep cirúrgico dos 118 IDs de rastreador (Tarefa/T8.x/SQ5.5/αx/F-x) nos
  12 arquivos `.rs` de `tests/`, removendo apenas o token de ID e preservando texto descritivo
  e medições. Deve ser atômico por arquivo para evitar inconsistência. Validar com
  `cargo clippy --tests --all-features` e `utils/lints.sh`.
- **E3.T04 (← F5):** Resolver as referências dangling a `TODO-findings.md` (F-X2, Épicos A–F)
  no `tests/fixtures/README.md:141,813-814` — exige decisão de PO (Opção A: persistir auditoria;
  Opção B: remover referências dangling).
- **E3.T05 (← F6):** Converter `//` explicativos em `///` acima de itens `pub` em
  `tests/common/{manifest,precision,metrics,constants}.rs` e adicionar doc-comments aos
  structs/fns públicos listados.

**Verificação de aceitação do Epic:** `utils/lints.sh` verde; `rg` pelos padrões de tracker ID
retorna vazio (exceto links `TODO-*.md` legítimos); README sem paths flat nem comandos
`--test` incorretos.

---

## Correções aplicadas neste turno (refatora-doc)

Estas correções já foram editadas diretamente nos arquivos durante a inspeção:

| Arquivo                                 | Correção                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------- |
| `tests/fixtures/README.md`              | 11 paths flat→modular (`tests/X.rs` → `tests/{models,parity}/X.rs`)             |
| `tests/fixtures/README.md:251-2`        | `--test golden_vectors` → `--test models` (2 comandos)                          |
| `tests/fixtures/README.md:901`          | `--test cpp_parity` → `--test parity`                                           |
| `tests/fixtures/README.md:114`          | Removido "F5 resolved in D.1.1."                                                |
| `tests/fixtures/README.md:715`          | Removido "(Sprint 2: Preservação do Teste de Estresse Numérico e Documentação)" |
| `tests/fixtures/README.md:743`          | Removido "(T4.1)"                                                               |
| `tests/models/proptest_parsers.rs:1438` | Adicionado comentário mapeando as 8 arms do `match` a seus casos adversariais   |

---

## Rastreabilidade

| Finding | Arquivo:linha                                                         | Epic.Tarefa | Risco  | Status      |
| ------- | --------------------------------------------------------------------- | ----------- | ------ | ----------- |
| F1      | `tests/clap.rs:4`                                                     | E1.T01      | Nenhum | ✅ Aplicado |
| F2      | `tests/common/signals.rs` (deletado; re-export movido p/ `mod.rs:31`) | E2.T01      | Baixo  | ✅ Aplicado |
| F3      | `utils/tests-long.sh:526,541`                                         | E1.T02      | Nenhum | ✅ Aplicado |
| F4      | 12 arquivos `tests/**/*.rs` (118×)                                    | E3.T03      | Baixo  | ✅ Aplicado |
| F5      | `tests/fixtures/README.md:141,813`                                    | E3.T04      | Nenhum | ✅ Aplicado |
| F6      | `tests/common/{manifest,precision,metrics,constants}.rs`              | E3.T05      | Nenhum | Planejado   |

> **Nota de estado (verificado neste turno):** F1 e F2 foram aplicados entre turnos por edição
> externa (cabeçalhos marcados `[DONE]`). Verificação direta confirma: `mod common;` removido de
> `tests/clap.rs` (F1 ✓); `tests/common/signals.rs` deletado e o re-export `generate_sine_440hz`
> migrado para `tests/common/mod.rs:31` (F2 ✓ — implementação mais limpa que a proposta original).
> F3 foi marcado `[DONE]` mas **não foi aplicado** — `utils/tests-long.sh:526,541` ainda contêm
> os paths stale; tag `[DONE]` removida por inexatidão.
>
> **Atenção:** o binary de teste `models` atualmente não compila (33 erros pré-existentes — API
> `StaticModel::process_scalar` renomeada para `process` em refactor em andamento, alheio a este
> trabalho). As edições deste turno (README + comentário em `proptest_parsers.rs`) não afetam
> compilação (markdown e comentários `//` apenas).
