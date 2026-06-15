<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Auditoria da Infraestrutura de Testes (`utils/tests-cargo.sh`)

> Documento gerado pela skill `revisor-auditor` + `planejador-arquiteto`.
> Foco: tornar o motor de testes **seguro, estável, ágil e informativo**, mantendo o
> nam-rs em _guard rails_ que permitam "voar alto" com agilidade.
> Base de evidências: execução "a frio" registrada em `tests-cargo.log` (2405 linhas),
> análise dos scripts `utils/tests-cargo.sh`, `utils/lints.sh`, `utils/tests-long.sh`
> e do código-fonte de instrumentação de auditoria de heap.

---

## 0. Sumário Executivo

A suíte padrão (`utils/tests-cargo.sh`) é **funcionalmente correta** (413 + 488 testes
passando, 19/21 do `clap-validator` aprovados, 2 _skipped_ legítimos), porém sofre de
**problemas estruturais de eficiência e arquitetura de testes** que a tornam lenta e
desencorajam a execução frequente — exatamente o anti-padrão que o projeto quer evitar.

### Achados priorizados

| ID     | Achado                                                                                                                            | Severidade | Impacto principal                                                     |
| ------ | --------------------------------------------------------------------------------------------------------------------------------- |:----------:| --------------------------------------------------------------------- |
| **F1** | Estado **global mutável** na auditoria de alocação (`TRACKING_THREAD`/`ALLOC_COUNT`) força `--test-threads=1` em **toda** a suíte | 🔴 Crítico | Agilidade (raiz da lentidão) + risco de **falso-negativo** silencioso |
| **F2** | Execução **redundante da suíte inteira** nas fases 1 (default) e 3 (clap+heap-audit)                                              | 🟠 Alto    | Agilidade (≈ dobra o tempo de teste)                                  |
| **F3** | Funções `#[test]` dentro de módulos compartilhados `tests/common/` → executadas 1×/binário (76 execuções redundantes medidas)     | 🟠 Alto    | Agilidade + ruído de log                                              |
| **F4** | `--target-dir target/clap-test` separado → **recompilação total** da árvore GUI (egui/baseview/wayland/vello) ≈ 1m23s a frio      | 🟡 Médio   | Agilidade (cold run)                                                  |
| **F5** | Mensagem de duração imprecisa ("± 2 minutos") e **ausência de timing por fase / sumário**                                         | 🟡 Médio   | Informatividade                                                       |
| **F6** | Tratamento de erro inconsistente entre fases; sem detecção de regressão de _tempo_ de teste                                       | 🟢 Baixo   | Estabilidade/Observabilidade                                          |
| **F7** | Ruído de log `[CLAP_PLUGIN_ERROR]` em condição esperada (state-invalid) e validação só em _debug_                                 | 🟢 Baixo   | Informatividade / funcional                                           |

### Tempo medido (cold run, `tests-cargo.log`)

| Fase                                      | Compilação   | Execução de testes                         | Observação                     |
| ----------------------------------------- |:------------:|:------------------------------------------:| ------------------------------ |
| [1/4] `cargo test` (default)              | 44,22 s      | ~55 s (414 lib + integração, **1 thread**) | `target/debug`                 |
| [2/4] build CLAP `.so` (debug+heap-audit) | 30,68 s      | —                                          | `target/clap-test`, só `--lib` |
| [3/4] `cargo test` (clap+heap-audit)      | **1m 23s**   | ~66 s (490 lib + integração, **1 thread**) | recompila árvore GUI inteira   |
| [4/4] `clap-validator` (debug)            | —            | ~3 s                                       | 19 PASS / 2 SKIP               |
| **Total cold**                            | **≈ 2m 38s** | **≈ 2m 04s**                               | **≈ 4m 40s**                   |

> A etiqueta "± 2 minutos" só vale para _execução de testes em build morno_; a frio o
> custo real é ≈ 4m40s. A máquina de referência tem **16 núcleos** ociosos enquanto a
> suíte roda **single-thread**.

---

## ÉPICO 1 — Motor de Testes Padrão: agilidade, segurança e informatividade

> **Objetivo**: reduzir o tempo de _wall-clock_ da suíte padrão (alvo: **< 60 s morno**,
> **< 2 min frio**) **sem perder cobertura de regressão**, eliminar falsos-negativos
> latentes na auditoria de alocação e tornar a saída informativa e diagnosticável.
> Este épico trata **exclusivamente da infraestrutura de testes**, não da lógica do nam-rs.
> ⚠️ **Épico mais crítico do documento.** A Sprint 1.1 (F1) toca código de produção
> (`src/common/alloc_audit.rs`) que é a base das garantias de RT-Safety. Qualquer erro
> aqui pode mascarar regressões de alocação na thread de áudio. Exige revisão rigorosa.

---

### Sprint 1.1 — Eliminar o estado global da auditoria de alocação (raiz de F1)

**Contexto técnico (evidência).**
`src/dsp/pipeline/mod.rs:64-66` registra `CountingAllocator` como `#[global_allocator]`
sob `#[cfg(test)]` para **todo o binário de teste da lib** (não só sob `heap-audit`).
A contabilidade vive em **estáticos de processo** (`src/common/alloc_audit.rs:13-19`):

```rust
pub static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0); // thread observada
pub static ALLOC_COUNT:     AtomicUsize = AtomicUsize::new(0);
```

`TrackingGuard::new()` (`alloc_audit.rs:69-74`) grava o `tid` atual em `TRACKING_THREAD`
e zera `ALLOC_COUNT`. Se **dois** testes que usam `TrackingGuard` rodarem em paralelo:

1. Teste A (thread T1) define `TRACKING_THREAD = T1`.
2. Teste B (thread T2) **sobrescreve** `TRACKING_THREAD = T2`.
3. As alocações de A deixam de ser contadas → a asserção "zero alocações" de A
   **passa por engano** (falso-negativo). Uma regressão de alocação no _hot-path_
   passaria despercebida.

Esta é a razão real do `--test-threads=1` global em `tests-cargo.sh:30` e `:48`. O flag
é um **paliativo** que sacrifica a agilidade da suíte inteira para proteger um punhado
de testes. A cópia de integração (`tests/common/alloc_audit.rs`) replica o mesmo padrão.

- [x] **T1.1.1 — Tornar a contabilidade de alocação _thread-local_.**
  Substituir os estáticos globais por contadores _thread-local_ (`thread_local!` com
  `Cell<usize>` / `Cell<bool>`), de modo que cada thread de teste rastreie suas próprias
  alocações de forma isolada. O `CountingAllocator::alloc` incrementa o contador da
  thread corrente **apenas se** o rastreamento estiver ativo nessa thread. Eliminar a
  comparação por `tid` global.

  - _Critério de aceite_: dois testes com `TrackingGuard` ativos em threads distintas
    **não** interferem entre si; um teste de regressão prova o isolamento (ver T1.1.3).
  - _Arquivos_: `src/common/alloc_audit.rs`, `tests/common/alloc_audit.rs` (manter paridade).
  - _Atenção_: o alocador global é invocado **antes** da inicialização de alguns
    `thread_local!`; usar `try_with` e tratar o caso `AccessError` como "não rastreando"
    para evitar recursão/_panic_ durante a destruição de TLS.

- [x] **T1.1.2 — Adequar todos os consumidores de `TrackingGuard`/`ALLOC_COUNT`.**
  Revisar usos em `src/clap/processor_test.rs:717-790`, `tests/zero_alloc_infer.rs`,
  `tests/*_heap_audit.rs`, `src/dsp/pipeline/*_test.rs` e demais, ajustando a leitura do
  contador para a API _thread-local_.

  - _Critério de aceite_: `cargo test` (todas as features) compila e passa.

- [x] **T1.1.3 — Teste de regressão de isolamento paralelo.**
  Adicionar teste que dispara N threads, cada uma com `TrackingGuard`, executando
  padrões de alocação distintos simultaneamente, e valida que cada contador reflete
  **apenas** sua própria thread (detecta corrupção cruzada).

  - _Critério de aceite_: o teste falha com a implementação global antiga e passa com a
    _thread-local_.

- [x] **T1.1.4 — Remover `--test-threads=1` da auditoria de heap.**
  Após T1.1.1–T1.1.3, confirmar que os testes de heap-audit toleram paralelismo.

  - _Critério de aceite_: `cargo test --features clap-plugin,heap-audit` passa **sem**
    `--test-threads=1`, de forma determinística em ≥ 5 execuções consecutivas.

- [x] **T1.1.5 — Serializar apenas os testes que mutam estado de processo.**
  `tests/diagnostic_bundle.rs:297-302` faz `std::env::set_var("HOME"/"XDG_RUNTIME_DIR")`
  (global, _racy_). Em vez do flag global, isolar esses testes (ex.: `serial_test`, ou
  um _mutex_ de teste, ou execução em processo próprio — ver Sprint 1.5/nextest).

  - _Critério de aceite_: nenhum teste depende mais de `--test-threads=1` global.

> **Alternativa de baixo risco (avaliar em conjunto):** adotar **`cargo-nextest`**
> (Sprint 1.5). Como o nextest executa **cada teste em seu próprio processo**, os
> estáticos globais ficam naturalmente isolados, eliminando F1 **sem alterar** o código
> de produção. A Sprint 1.1 (thread-local) continua recomendada como _defesa em
> profundidade_ e para manter `cargo test` puro também seguro.

---

### Sprint 1.2 — Eliminar a execução redundante da suíte (F2)

**Contexto (evidência).** A fase 1 roda 414 testes da lib (default features) e a fase 3
roda 490 (clap+heap-audit). O superconjunto da fase 3 **re-executa praticamente todos**
os testes de `dsp::`, `math::`, `models::`, `loader::` — idênticos entre as duas
configurações — apenas para exercitar o delta real: módulos `clap::*`, os testes
`*_heap_audit` e os de ciclo de vida CLAP (`clap_lifecycle_test`, `clap_state_migration`,
`clap_multi_instance`). Na fase 1 esses binários CLAP aparecem como "running 0 tests"
(linhas 643-657, gated pela feature). Conclusão: a fase 3 paga o custo de rodar ~470
testes que **já passaram** na fase 1.

- [x] **T1.2.1 — Restringir a fase 3 ao delta CLAP + heap-audit.**
  Trocar o `cargo test --features clap-plugin,heap-audit` amplo por execução dirigida:
  testes da lib `clap::` + binários de integração relevantes
  (`a2_heap_audit`, `cabsim_heap_audit`, `resampler_heap_audit`, `diagnostic_bundle`
  com variante heap, `clap_lifecycle_test`, `clap_state_migration`, `clap_multi_instance`).

  - _Critério de aceite_: cobertura efetiva (testes que **só** existem sob clap/heap-audit)
    preservada; testes puramente numéricos rodam **uma única vez** (fase 1).
  - _Risco_: garantir que nenhum teste exclusivo da config clap/heap-audit seja perdido.
    Mapear explicitamente (ver T1.2.2).

- [x] **T1.2.2 — Inventário de cobertura por configuração de feature.**
  Documentar (tabela) quais testes só passam/existem em cada combinação de features,
  para justificar o que roda em qual fase e evitar lacunas.

  - _Critério de aceite_: tabela versionada (ex.: em `docs/testing.md`).

- [x] **T1.2.3 — Confirmar que o `.so` validado é o correto.**
  A fase 2 compila o `.so` (cdylib) e a fase 3 reusa `target/clap-test` para `cargo test`.
  Garantir que o `cargo test` da fase 3 não sobrescreva/invalide o `.so` consumido pelo
  `clap-validator` na fase 4 (hoje funciona, mas é frágil/implícito).

  - _Critério de aceite_: o binário validado é comprovadamente o artefato da fase 2.

---

### Sprint 1.3 — Sanear os módulos `tests/common/` (F3)

**Contexto (evidência).** `tests/common/mushra_primitives.rs:25` e
`tests/common/perceptual.rs:31` declaram `#[test]` **dentro de um módulo compartilhado**
(`mod common`, incluído por quase todos os binários de integração). Resultado: cada um é
**compilado e executado em cada binário** que faz `mod common`. Medição no log:
`test_mulberry32_parity_with_ts` e `test_mr_stft_parity_with_python` rodaram **38× cada**
(76 execuções no total) — em uma máquina de 16 núcleos, em série.

- [x] **T1.3.1 — Mover os `#[test]` de paridade para um único binário dedicado.**
  Criar `tests/parity_primitives.rs` (ou similar) contendo **apenas** esses dois testes,
  e remover os `#[test]` de `tests/common/mushra_primitives.rs` e
  `tests/common/perceptual.rs` (mantendo as funções utilitárias reutilizáveis).

  - _Critério de aceite_: cada teste de paridade roda **1× por suíte**; saída do log sem
    repetição; helpers continuam disponíveis para os demais testes.

- [x] **T1.3.2 — Lint/guarda contra regressão do anti-padrão.**
  Adicionar verificação (script ou teste meta) que falha se `#[test]` reaparecer em
  qualquer arquivo sob `tests/common/`.

  - _Critério de aceite_: a guarda detecta a reintrodução do padrão.

---

### Sprint 1.4 — Otimizar compilação e _target dirs_ (F4)

**Contexto (evidência).** A fase 3 usa `--target-dir target/clap-test` separado do
`target/` da fase 1. Como a feature `clap-plugin` arrasta a árvore GUI inteira
(egui, egui_glow, glow, baseview, wayland-_, vello__, skrifa, x11rb…), o primeiro
`cargo test` nesse diretório recompila **tudo do zero** (1m23s no log). O diretório
separado existe deliberadamente para **não** thrashar o `target/` principal entre
builds de features diferentes (`lints.sh` usa default features) — é um _trade-off_.

- [x] **T1.4.1 — Medir e decidir a estratégia de _target dir_.**
  Comparar três cenários e cronometrar (cold/warm): (a) status quo (target separado);
  (b) target único compartilhado entre fases; (c) target separado **persistente** porém
  com cache (sccache/`CARGO_INCREMENTAL`).

  - _Critério de aceite_: decisão documentada com números; escolher a que minimiza o
    tempo total **sem** reintroduzir thrash com `lints.sh`.

- [CANCELADO] **T1.4.2 — Avaliar `sccache` para a stack de testes.**
  A árvore GUI é estável (raramente muda); um cache de compilação cortaria o cold run
  da fase 3 drasticamente.
  > Nota do PO: Não necessariamente. Ele vai ser mexido em breve. Não vamos complicar!
  - _Critério de aceite_: PoC com ganho ≥ 50% no cold compile da fase 3, ou justificativa
    documentada para não adotar.

- [x] **T1.4.3 — Reaproveitar artefatos entre fase 2 e fase 3.**
  Investigar se a fase 2 (build `--lib` cdylib) pode compartilhar dependências com a
  fase 3 (test harness) no mesmo target, evitando recompilar a árvore comum duas vezes.

  - _Critério de aceite_: redução mensurável do tempo somado das fases 2+3.

---

### Sprint 1.5 — Adotar `cargo-nextest` e tornar a saída informativa (F1 alt + F5 + F6)

**Contexto.** O `cargo test` padrão: (a) roda em série quando forçado por `--test-threads=1`;
(b) imprime ~900 linhas de "... ok" por fase (ruído); (c) não dá _timing por teste_ nem
detecta regressão de tempo; (d) não isola estado global entre testes. O `cargo-nextest`
resolve (a)–(d): processo por teste (isolamento → cobre F1 sem mexer no código),
paralelismo agressivo, sumário conciso, _timeouts_ por teste, _retries_ e detecção de
testes lentos/_flaky_.

> **Atualização T1.5.1:** o PoC mostrou que nextest é ~2× mais lento que `cargo test`
> para esta suíte (testes muito rápidos, custo de processo domina). A Sprint 1.1 já
> removeu a serialização, anulando o ganho principal esperado. O foco do sprint agora
> é: (a) adotar nextest para **perfis pontuais** (`serial`/`heavy` — T1.5.2) e
> (b) portar a melhoria de saída informativa para `cargo test` (T1.5.3—T1.5.5).

- [x] **T1.5.1 — PoC `cargo-nextest` na fase 1 e 3.**
  Substituir `cargo test` por `cargo nextest run` (mantendo `cargo test --doc` à parte —
  o projeto tem 0 doctests hoje, então sem impacto). Medir wall-clock vs baseline.

  - **Resultado do PoC (2026-06-14):** nextest NÃO atinge o critério de ≥50% de redução.
    Em execução morna (pré-compilada) em 16 núcleos:

    | Fase   | `cargo test` | `cargo nextest` | Delta    |
    |--------|--------------|-----------------|----------|
    | 1      | ~32,1s       | ~62,8s          | **+96%** |
    | 3      | ~3,5s        | ~6,7s           | **+91%** |

    > A maioria dos testes é muito rápida (<0,05s) → o custo de criação de processo por
    > teste domina o runtime. O `cargo test` já roda em paralelo (Sprint 1.1 eliminou
    > `--test-threads=1`), anulando o principal ganho esperado com nextest.
  - **Achados positivos:**
    1. nextest expôs _falso-positivo_ no `test_rdtsc_nanos_significant`: passa no
       `cargo test` (processo aquecido por outros testes) mas falha em isolamento
       (nextest) — o teste depende de tempo de execução acumulado do processo.
    2. Isolamento por processo cobre F1 sem alterações de código (defesa em profundidade).
    3. Sumário conciso e _timing por teste_ → valor informativo (coberto por T1.5.3).
    4. `.config/nextest.toml` criado com configuração base. Script PoC em
       `utils/tests-cargo-nextest-poc.sh`.
  - **Recomendação:** não adotar `cargo-nextest` como runner padrão (perda de ~2× em
    velocidade). Manter instalado para diagnóstico pontual de testes _flaky_ e para o
    perfil `serial`/`heavy` (T1.5.2).
  - **Nota do PO:** Artefatos deletados. Não vamos continuar avaliando o nextest.
    Ele deveria ter trazido benefícios mais óbvios, que não melhorarão adiante

- [CANCELADO] **T1.5.2 — Perfis nextest com `serial`/`heavy`.**
  Configurar `.config/nextest.toml` com grupos: testes que mutam estado de processo
  (`diagnostic_bundle` env vars) em grupo serial; demais paralelos; _slow-timeout_ para
  flagrar testes que estouram orçamento de tempo.

  - _Critério de aceite_: testes serializáveis identificados e isolados; sumário lista
    os N testes mais lentos.

- [CANCELADO] **T1.5.3 — Timing por fase + tabela de sumário (estilo `tests-long.sh`).**
  Portar para `tests-cargo.sh` o padrão de `run_phase`/sumário de `utils/tests-long.sh:51-83,177-206`:
  duração por fase, status colorido e tabela final.

  - _Critério de aceite_: ao fim da suíte, tabela com `Fase | Duração | Status`.

- [CANCELADO] **T1.5.4 — Mensagem de duração honesta + modo cold/warm.**
  Corrigir o cabeçalho "± 2 minutos" para refletir cold (~4-5 min) vs warm (~1-2 min),
  ou medir e exibir dinamicamente.

  - _Critério de aceite_: cabeçalho não engana o desenvolvedor.

- [CANCELADO] **T1.5.5 — Padronizar tratamento de erro entre fases.**
  Hoje a fase 3 usa `set +e`/banner manual (`tests-cargo.sh:45-57`) e as demais confiam
  em `set -e`. Unificar (idealmente via função `run_phase` comum, igual ao long).

  - _Critério de aceite_: qualquer falha de fase produz banner consistente e _exit code_
    correto; `set -euo pipefail` preservado.

---

### Sprint 1.6 — Revisão de escopo: o que roda com frequência vs. o que deveria ser `#[ignore]`

> Princípio do projeto: a suíte padrão captura **regressões antes que se propaguem**;
> caça profunda e cenários extremos vivem em `utils/tests-long.sh`. Script lento é
> incentivo a pular — então a suíte padrão deve conter **somente** o que precisa rodar a
> cada commit.

- [x] **T1.6.1 — Auditar os binários de teste mais lentos da suíte padrão.**
  Da medição: `a2_heap_audit` (9,20 s, fase 3), `nam_infer_test` (5,05 s),
  `self_consistency` (3,87 s), `container_slimmable` (3,61 s — só 2 testes "reais",
  investigar _sleeps_/crossfade), `golden_vectors` (3,08 s). Decidir caso a caso:
  manter (regressão essencial) ou mover variante pesada para `#[ignore]`/long.

  - _Critério de aceite_: cada teste > 1 s justificado como "essencial para detecção
    rápida de regressão" ou rebaixado para a suíte longa.

- [x] **T1.6.2 — Confirmar que nada essencial está erroneamente `#[ignore]`.**
  Revisar os _ignored_ do log (proptests, soak, cpp_parity live, golden v2_*,
  `wavenet_lite` divergente) para garantir que a suíte longa de fato os cobre e que
  nenhum _gate_ rápido foi perdido.

  - _Critério de aceite_: matriz "teste → suíte (padrão/longa) → frequência" documentada.

- [x] **T1.6.3 — Avaliar fail-fast vs. visão completa.**
  Decidir se a suíte padrão deve parar no 1º erro (feedback rápido) ou usar
  `--no-fail-fast` (visão completa). Recomendado: fail-fast no padrão, completo no long.

  - _Critério de aceite_: comportamento explícito e documentado.

---

## ÉPICO 2 — Correções funcionais e de ruído reveladas pela suíte (pós-infra)

> Itens funcionais do nam-rs detectados durante a auditoria. **Não bloqueiam** o Épico 1;
> devem ser tratados após estabilizar a infraestrutura.

### Sprint 2.1 — Ruído e fidelidade do `clap-validator`

- [ ] **T2.1.1 — Rebaixar o nível de log do _state-invalid_.**
  `tests-cargo.log:2306`:
  `[CLAP_PLUGIN_ERROR] Failed to deserialize state (v0 legacy): EOF ... line 1 column 0`.
  Ocorre no teste `state-invalid` (carregar estado **vazio**), condição **esperada** em
  que o plugin corretamente retorna `false` (teste PASSED). Logar como `ERROR` polui a
  saída e pode alarmar. Rebaixar para `debug`/`trace` quando a entrada for vazia.

  - _Arquivos_: `src/clap/extensions/state.rs` (caminho de `Deserialize`, ~`:238`).
  - _Critério de aceite_: estado vazio não emite `[CLAP_PLUGIN_ERROR]`; comportamento
    de retorno inalterado; teste do validator continua PASSED.

- [ ] **T2.1.2 — Decidir sobre `clap.note-ports` (2 testes SKIPPED).**
  `process-note-*` são pulados por o plugin não implementar `note-ports` (linhas 2373/2377).
  Confirmar se é intencional (plugin de efeito sem MIDI) e documentar para não parecer
  lacuna.

  - _Critério de aceite_: decisão registrada em `docs/` (intencional) ou tarefa criada.

- [ ] **T2.1.3 — (Opcional) validação estrita em debug na suíte padrão.**
  A suíte padrão roda `clap-validator validate` (exit code) no `.so` **debug**; a longa
  usa `--json` + `jq` para falhar em _warnings_ (`tests-long.sh:128-137`). Avaliar trazer
  a checagem de _warnings_ via `--json` para a suíte padrão (barata, ~3 s).

  - _Critério de aceite_: _warnings_ do validator falham a suíte padrão, ou justificativa.

### Sprint 2.2 — Divergência numérica conhecida (registrada, não regressão nova)

- [ ] **T2.2.1 — Investigar WaveNet Lite (CH=12) vs C++.**
  `golden_vectors`: `test_golden_vectors_wavenet_lite` está `ignored` como
  "known-divergent ... SNR = 0,9 dB vs C++" (linhas 764/2046). SNR de 0,9 dB é
  perceptualmente relevante. Investigar a fonte da deriva (acúmulo, ordem de redução,
  bf16) e definir plano de correção ou aceitação formal.
  - _Critério de aceite_: causa-raiz documentada; teste reabilitado **ou** divergência
    formalmente aceita com justificativa numérica (skill `pesquisador-inovador`).

---

## Apêndice A — Mapa de evidências (arquivo:linha)

| Evidência                           | Localização                                                             |
| ----------------------------------- | ----------------------------------------------------------------------- |
| `--test-threads=1` global           | `utils/tests-cargo.sh:30`, `:48`                                        |
| Alocador global sob `#[cfg(test)]`  | `src/dsp/pipeline/mod.rs:64-66`                                         |
| Estáticos globais de auditoria      | `src/common/alloc_audit.rs:13-19`, `:69-74`                             |
| `#[test]` em módulo compartilhado   | `tests/common/mushra_primitives.rs:25`, `tests/common/perceptual.rs:31` |
| 38× execuções redundantes (cada)    | medido em `tests-cargo.log`                                             |
| Target dir separado / recompilação  | `utils/tests-cargo.sh:35,48`; `tests-cargo.log:1317` (1m 23s)           |
| `env::set_var` racy em teste        | `tests/diagnostic_bundle.rs:297-302`                                    |
| Comentário "sem `--test-threads=1`" | `tests/concurrency_stress.rs:9`                                         |
| Padrão `run_phase`/sumário a portar | `utils/tests-long.sh:51-83`, `:177-206`                                 |
| Ruído `[CLAP_PLUGIN_ERROR]`         | `tests-cargo.log:2306`; `src/clap/extensions/state.rs:238`              |
| WaveNet Lite divergente             | `tests-cargo.log:764`, `:2046`                                          |

## Apêndice B — Metas de aceite do Épico 1

- **Agilidade**: suíte padrão **< 60 s** (build morno) e **< 2 min** (frio) em 16 núcleos.
- **Segurança**: auditoria de alocação **sem falso-negativo** sob paralelismo (T1.1.3).
- **Estabilidade**: 0 _flakes_ em ≥ 5 execuções consecutivas pós-paralelização.
- **Informatividade**: sumário com timing por fase + lista dos testes mais lentos.
- **Cobertura**: nenhuma perda de teste exclusivo de configuração (T1.2.2 / T1.6.2).
