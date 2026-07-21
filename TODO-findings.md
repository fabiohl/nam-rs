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

## F3 — Comentários com paths stale em `utils/tests-long.sh`

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
   seções (T8.2 handshake, T8.3 GC overflow, T8.4 double-buffering DspBridge). Referenciado por
   `utils/tests-long.sh` via `--test loom_tests`. Nada a refatorar.

6. **Marcadores `TODO` em arquivos de teste:** São referências legítimas a docs de rastreamento
   (`TODO-sprints.md`, `TODO-wavenet_a2_max.md`, `TODO-convnet_parity.md`) e a epics/tasks
   específicos. Não são dívida técnica a remover.

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

---

## Rastreabilidade

| Finding | Arquivo:linha                  | Epic.Tarefa | Risco    |
| ------- | ------------------------------ | ----------- | -------- |
| F1      | `tests/clap.rs:4`              | E1.T01      | Nenhum   |
| F2      | `tests/common/signals.rs:4-11` | E2.T01      | Baixo    |
| F3      | `utils/tests-long.sh:526,541`  | E1.T02      | Nenhum   |
