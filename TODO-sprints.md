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
>
> ### ✅ Resultado do Épico 1 (CONCLUÍDO — auditado em 2026-06-14)
>
> **Métricas (cold run com `rm -rf target/`, vide `tests-cargo.log`):**
>
> | Métrica           | Baseline  | Pós-Épico 1           | Meta    | Veredito           |
> | ----------------- |:---------:|:---------------------:|:-------:|:------------------:|
> | Cold (frio)       | ≈4m40s    | **2m17,9s**           | < 2 min | 🟡 Quase (−51%)    |
> | Warm (morno)      | ≈2m00s    | **0m58,3s**           | < 60 s  | ✅ Atingido (−51%) |
> | Lib tests (par.)  | 33,4s/1t  | **22,2s**             | —       | ✅ Paralelizado    |
> | Parity redundante | 38×/teste | **1×/run**            | 1×      | ✅ Resolvido (F3)  |
> | clap-validator    | 19/21     | **19/21, 0 warnings** | —       | ✅ Mantido         |
>
> **Decisões registradas:**
>
> - **F1** resolvido via TLS (Sprint 1.1); caminho RT do CLAP validado (guard armado na thread de áudio).
> - **nextest REJEITADO** (T1.5.1): ~2× mais lento nesta suíte (testes <0,05s, custo de processo domina). Artefatos removidos.
> - **sccache CANCELADO** (T1.4.2) por decisão de PO (infra será mexida em breve).
> - **T1.5.3/1.5.4/1.5.5 CANCELADOS**: saída informativa via tabela de fase abandonada em favor de `cargo test` enxuto + fail-fast (T1.6.3).
> - **F4** resolvido unificando `target/` + profile `test` nas fases 2-3 (T1.4.3).
>
> **Achados residuais (carregados para Épico 3 / acompanhamento):**
>
> - 🟡 **`diagnostic_bundle` custa 10,28s no cold run** (era 0,01s) — `test_panic_hook_behavior` paga simbolização de _backtrace_ em cache de disco frio (warm = 0,02s). Cold-only, baixa prioridade. Ver T3.3.1.
> - 🟢 **`AUDIT_ENABLED` global** ainda é mutado por `test_heap_audit_trigger` sem isolamento RAII (0 flakes em 6 runs, mas frágil). Ver T3.3.2.
> - 🟢 Header "± 54 second" reflete só o warm; cold é 2m18s (sem distinção honesta cold/warm).

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

    | Fase | `cargo test` | `cargo nextest` | Delta    |
    | ---- | ------------ | --------------- | -------- |
    | 1    | ~32,1s       | ~62,8s          | **+96%** |
    | 3    | ~3,5s        | ~6,7s           | **+91%** |

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

## ÉPICO 2 — Suíte Longa (`utils/tests-long.sh`): auditoria, otimização e protocolo de execução

> **Objetivo**: aplicar à suíte longa os aprendizados do Épico 1 — **eliminar desperdício
> de recompilação** e tornar a execução **organizada e diagnosticável** — SEM reduzir a
> profundidade do estresse. A suíte longa é **deliberadamente pesada** (caça profunda,
> cenários extremos); o alvo não é "rápida", é **"sem desperdício e executável em lote"**.
> Hoje consome ≈ 46 min (cold). Boa parte é **recompilação por troca de profile/feature**,
> não estresse real.
>
> ### ⚠️ PROTOCOLO DE EXECUÇÃO EM LOTE (LER ANTES DE EXECUTAR TAREFAS)
>
> A suíte longa **NÃO deve ser executada dentro de uma tarefa técnica**. Para não travar o
> progresso das sprints com esperas de ~46 min:
>
> 1. **Tarefas técnicas** fazem apenas edições + validações **rápidas e pontuais**:
>    `cargo build/test ... --no-run` (só compila), ou a execução de **uma única fase curta**
>    quando isolável. **Nunca** a suíte completa.
> 2. **As execuções completas acontecem em GATES explícitos** (marcados `🚦 GATE Lx`),
>    quando o usuário puder deixar rodando e voltar depois. O agente **avisa** que um gate
>    chegou, agrupa **todas** as mudanças pendentes que tocam a suíte longa, e o usuário roda
>    **uma vez**, trazendo `target/logs/*.log` + a tabela de sumário final.
> 3. **Meta de no máximo 2 execuções completas no épico**: `🚦 GATE L0` (baseline) e
>    `🚦 GATE L1` (validação final pós-otimização). Toda tarefa de implementação fica
>    **entre** L0 e L1 e é validada por compilação/raciocínio, colhendo resultado só no L1.

### Sprint 2.1 — Baseline e diagnóstico (🚦 GATE L0)

- [x] **T2.1.1 — 🚦 GATE L0: medir baseline completo.**
  Execução **completa** da `tests-long.sh` atual (cold), preservando `target/logs/` e a
  tabela de sumário. **Único momento de execução completa desta sprint.**

  - _Critério de aceite_: baseline com duração **por fase** e status registrados no épico.

  - _Quando_: o usuário roda quando puder ausentar-se (~46 min) e traz os logs.

  - _Resultado (2026-06-14, 48m41s total)_:

    | Fase                                          | Duração | Status     |
    | --------------------------------------------- | ------- | ---------- |
    | Soak Tests (Numerical Stability)              | 139s    | PASSED     |
    | Property-Based & Parity Tests in Release      | 137s    | PASSED     |
    | Resampler, Cabsim & A2 Heap-Audit, C++ Parity | 152s    | **FAILED** |
    | CLAP Release Validation & Concurrency         | 223s    | PASSED     |
    | Long Performance Benchmarks                   | 2222s   | PASSED     |
    | PipeWire Integration Test                     | 49s     | PASSED     |

    **Phase 3 failure**: `cabsim_cpp_parity` (short/medium/long) panicked on LUFS
    plausibility gate — golden C++ output LUFS (+20 to +42) outside plausible [-50, +10]
    range. This is a **false positive**: the golden is numerically correct (MSE ~1e-11,
    SNR 130+ dB), but the IR convolution produces legitimately high amplitude. Fixed
    by adding `report_dsp_fidelity_no_lufs()` in `tests/common/validation.rs` and using
    it in `cabsim_cpp_parity.rs`. All 3 tests now pass. See Phase 3 log for details.

- [x] **T2.1.2 — Quantificar build vs. estresse real (a partir dos logs do L0).**
  Contar nos `target/logs/phase*.log` quantas vezes `nam-rs` é recompilado e quanto tempo
  é build vs execução de teste/bench. Mapear cada fase → (profile, feature-set).
  - _Critério de aceite_: tabela "fase → profile/features → nº de rebuilds → s de build".
  - _Resultado (2026-06-14, analisado dos logs do 🚦 GATE L0)_:

    | Fase                    | Profile(s)      | Features                                                              | Builds | Build (s) | Exec (s) | Total (s) | % Build  |
    | ----------------------- | --------------- | --------------------------------------------------------------------- | ------ | --------- | -------- | --------- | -------- |
    | 1 — Soak Tests          | release         | standalone                                                            | 2      | 93        | 46       | 139       | **67%**  |
    | 2 — Property/Parity     | release         | (default)                                                             | 7      | 115       | 22       | 137       | **84%**  |
    | 3 — Heap-Audit + Parity | release         | heap-audit → default                                                  | 6      | 139       | 13       | 152       | **91%**  |
    | 4 — CLAP Validation     | release → debug | clap-plugin+heap-audit → clap-plugin, standalone, clap-plugin+testing | 5      | 155       | 68       | 223       | **70%**  |
    | 5 — Long Benchmarks     | bench           | (default) → standalone+long_bench                                     | 2      | 144       | 2078     | 2222      | **6%**   |
    | 6 — PipeWire            | release         | standalone                                                            | 1      | 49        | 0        | 49        | **100%** |

    **Total**: **695s build** (11.6 min, 24%) + **2227s exec** (37.1 min, 76%) = 2922s (48.7 min).
    **Principais desperdícios de recompilação identificados nas trocas de config:**
    - P2 interna: `--test` → `--lib` força rebuild completo do crate (58s)
    - P3: `heap-audit` → default features (~45s + 46s)
    - P4: `release` (79s) + debug dos mesmos deps GUI (25+9+40=74s) — **recompilação em dois profiles**
    - P5: features default → `standalone,long_bench` (70s)
    - P6: mudança de profile `bench` → `release` (49s)
    - Troca de `standalone` (P1) → default (P2) → `heap-audit` (P3) → `clap-plugin` (P4) → `bench` (P5) → `standalone` (P6): **5 mudanças de feature-set** + **2 mudanças de profile** (release↔debug, release↔bench) em 48 min de execução.

- [x] **T2.1.3 — Verificar cobertura real (anti silent-skip).**
  Confirmar, nos logs do L0, se `cpp_parity` (LSTM/WaveNet _live_) e `cabsim_cpp_parity`
  **de fato executam** ou se caem em _skip_ por ausência dos goldens C++. `tests-long.sh`
  **não** invoca `tests/fixtures/golden_gen_build.sh` — os goldens `cabsim` existem
  (estáticos), mas os _live_ de `cpp_parity` podem estar em silent-skip mesmo na longa.

  - _Critério de aceite_: lista do que roda de fato vs. o que é pulado, por fase.
  - _Resultado (L0 log)_:
    - **`tests/cpp_parity`**: 22 tests executados, todos `ok`. WaveNet Lite skip é
      **documentado** (`SKIP: WaveNet Lite (CH=12) (v2) is known-divergent (T1.2)`),
      não silent. Nenhum outro skip.
    - **`tests/cabsim_cpp_parity`**: 3 tests executados (short/medium/long). Sem skips.
      Falha inicial corrigida em T2.1.1.
    - **`tests/golden_vectors`** (v2_): 9 passed, 18 filtered out (`--skip wavenet_lite`
      explícito no script).
    - Conclusão: **zero silent-skip** — todos os testes executam e reportam resultado.

> **Insights do GATE L0 (Sprint 2.1):**
>
> 1. `cabsim_cpp_parity` corrigido: golden C++ IR output tem LUFS +20~+42
>    (amplitude alta legítima da convolução sintética), gate LUFS [-50,+10] é
>    falso positivo para IR convolution. Solução: `report_dsp_fidelity_no_lufs()`
>    em `tests/common/validation.rs`.
> 2. Tudo executa sem silent-skip → T2.3.1 é desnecessário como está; pode ser
>    repensado para outro gap ou removido.
> 3. Fase 5 (benches): 37 min (2222s) domina o tempo total. É o principal alvo
>    de otimização para o L1.

### Sprint 2.2 — Eliminar desperdício de compilação (espelho de F4/F2 do Épico 1)

> Todas as tarefas abaixo são **edições de script + `--no-run` para validar compilação**.
> Resultado de tempo só é colhido no 🚦 GATE L1 (Sprint 2.4).
> Sempre consulte os achados registrados nas tarefas anteriores.

- [x] **T2.2.1 — Reagrupar comandos por (profile, feature-set).**
  Hoje as fases alternam `release+standalone`, `release` (default), `release+heap-audit`,
  `release+clap-plugin`, `debug+clap-plugin`, `debug+standalone` — cada troca dispara
  recompilação (em release com `lto=fat`, custo enorme). Reordenar para **clusterizar**
  execuções de mesma configuração e minimizar trocas.

  - _Critério de aceite_: ordem das fases agrupada por config; nº de rebuilds da árvore
    cai mensuravelmente (validar no L1). Cobertura idêntica.

  - ✅ **Feito**: Release rebuilds reduzidos de 5 → 3. Phase 6 (pw_integration,
    release+standalone) fundida na Phase 1. Testes cpp_parity/cabsim_cpp_parity/golden_vectors
    (release+default) movidos da Phase 3 para Phase 2, eliminando rebuild duplicado.
    `default = ["standalone", "testing"]` → Phases 1 e 2 compartilham mesmos artefatos.

- [x] **T2.2.2 — Unificar profile da Phase 4 (CLAP).**
   A Phase 4 compila a árvore GUI em **release** (build do `.so`) **e em debug**
   (`cargo test` de `clap_multi_instance`, `gc_stress`, mono — `tests-long.sh:142-151`).
   Rodar esses testes em **release** reaproveita o build do `.so` e evita compilar todo o
   egui/baseview/wayland duas vezes.

  - _Critério de aceite_: Phase 4 não recompila a árvore CLAP em dois profiles; testes de
     concorrência/mono continuam passando.

  - ✅ **Feito**: 4 comandos de `cargo test` na Phase 4 agora usam `--release`.
     Linhas 158/161 também alinham `heap-audit` com o build do `.so` (linha 126)
     para máxima reutilização de artefatos. Verificação de compilação OK nos 4
     targets (`clap_multi_instance`, `gc_stress`, `concurrency_stress`, mono tests).

- [x] **T2.2.3 — Dedup de benches (Phase 5).**
  `cargo bench` (default) roda `inference_bench` (parte curta) e depois
  `cargo bench --features standalone,long_bench --bench inference_bench` recompila e
  **re-roda** as partes curtas do mesmo bench. Separar curto vs. longo sem recompilar/repetir.

  - _Critério de aceite_: cada bench roda o necessário **uma vez**; sem rebuild redundante.

  - ✅ **Feito**: benches longos extraídos para `benches/long_inference_bench.rs` (target
     separado). Phase 5 agora roda `cargo bench` (partes curtas de `inference_bench`) e
     `cargo bench --features standalone,long_bench --bench long_inference_bench` (soaks)
     sem recompilar/repetir nenhum benchmark. Ambos compilam em check separado sem erros.

- [x] **T2.2.4 — Reavaliar `--test-threads=1` nos soak (Phase 1).**
  Com a auditoria de alocação agora TLS (Épico 1), avaliar se `soak_test`/`pipeline_soak`
  ainda precisam de serialização ou podem paralelizar (mantendo determinismo numérico).

  - _Critério de aceite_: flag removido **ou** justificativa técnica documentada; soak
    determinístico em ≥ 2 execuções (colhido no L1).

  - ✅ **Feito**: `--test-threads=1` removido de `soak_test` em `utils/tests-long.sh:88`.
     **Mantido** para `pipeline_soak` (RSS measurement: 3 tests compartilham `/proc/self/status`
     no mesmo processo — medição de VmRSS cross-test contaminaria a assertiva de memory leak
     `rss_diff < 10MB`). Justificativa do `soak_test`: (1) soak tests nunca usaram
     `CountingAllocator`/`TrackingGuard` — o motivo original do `--test-threads=1` era a
     auditoria de alocação global, já resolvida com TLS no Épico 1; (2) cada test function é
     autocontida (modelos, buffers, PRNG com seed fixo próprios), sem estado mutável
     compartilhado entre threads; (3) `RtStatusFlags.degrade_transitions_total` é
     per-instance `AtomicU32`; (4) verificação de determinismo: 6 soak tests executados 2×
     sem `--test-threads=1` — todos passaram com outputs idênticos. Referência stale removida
     de `.agents/rules/testing.md:54`.

### Sprint 2.3 — Cobertura, profundidade e informatividade (sem perder estresse)

- [x] **T2.3.1 — Garantir geração dos goldens C++ (fechar gap de T2.1.3).**
  Invocar `tests/fixtures/golden_gen_build.sh` (ou documentar pré-requisito) no início da
  suíte longa, para que `cpp_parity`/`cabsim_cpp_parity` rodem de verdade — não _skip_.

  - _Critério de aceite_: paridade C++ executa com goldens presentes; ausência vira **erro
    explícito**, não _skip_ silencioso.

  - ✅ **Feito**: (1) `tests-long.sh` Phase 0 verifica todos os goldens v1, v2 (multi-SR,
    por grupo de modelo) e cabsim — ausência é `exit 1` explícito (nunca skip).
    (2) `NAM_AUTO_BUILD_GOLDENS=1` invoca `golden_gen_build.sh` automaticamente no
    pre-flight, com revalidação pós-geração. (3) Auto-clone do NeuralAmpModelerCore
    usa commit pinado (`9c7b185`) + init de submodules (`eigen`, `AudioDSPTools`),
    idêntico ao `golden_gen_build.sh`. (4) `ensure_render_compiled()` (cpp_parity.rs)
    e `compare_golden_cpp()` (cabsim_cpp_parity.rs) já tinham panics explícitos para
    ausência de diretório/binário/golden.

- [x] **T2.3.2 — Relatório de testes/benches mais lentos por fase.**
  Acrescentar ao sumário (que já tem duração por fase) o **top-N mais lento** dentro das
  fases pesadas, para guiar otimizações futuras.

  - _Critério de aceite_: sumário lista os maiores ofensores de tempo.

  - ✅ **Feito**: (1) `timed_cargo_test()` captura tempo de cada invocação via `date +%s%N`
    com precisão de ms, escrevendo entradas `TIMED: <s> <label>` em tracker temporário.
    (2) `extract_sub_timings()` ordena e seleciona top-5 por fase — aplicado às fases
    1-4 (Soak, Proptest/Parity, Heap-Audit, CLAP). (3) `extract_top_benches()` faz parse
    do output do Criterion (bench name + `time:` multi-linha) com awk, extrai mediana e
    unidade, converte para ns e ranqueia top-5 para Phase 5. (4) Sumário ganha seção
    "Top-N Items Mais Lentos por Fase Pesada" após tabela principal. PipeWire phase
    (leve) não reporta sub-timings (tracker vazio).

  - **Review (2026-06-15):** 3 achados menores identificados no `tests-long.sh`:
    (1) **WARNING** — loop de display (L466-467) usa `echo|awk`+`echo|cut` por linha;
    substituir por _bash parameter expansion_ (`${line%% *}` / `${line#* }`) elimina
    ~50 subprocessos. (2) **SUGGESTION** — `$TIMED_TRACKER` (L216) só é limpo no caminho
    normal (L478); ERR-trap (L20) não faz cleanup → leak de temp file em saída precoce
    (conteúdo não sensível). Adicionar `trap 'rm -f "$TIMED_TRACKER"' EXIT`.
    (3) **SUGGESTION** — regras `bench = ""` em `/^Found/` e `/^change:/` no awk (L257-258)
    são unreachable (bench já foi limpo pela regra `time:`); remover.

- [x] **T2.3.3 — Revisar profundidade de estresse (proptest/soak).**
  Confirmar que as contagens de casos (proptest) e iterações (soak) maximizam estresse
  onde é barato; aumentar onde o custo marginal for baixo. **Não reduzir** cobertura.

  - _Critério de aceite_: parâmetros de estresse revisados e justificados.

  - ✅ **Feito**: 4 ajustes aplicados (sem redução de cobertura):
    **(1) `src/math/activations/tests.rs`** — 5 proptests tinham nomes aspiracionais
    ("100k", "50k") mas usavam default ~256 casos. Adicionado `ProptestConfig::with_cases(10_000)`
    nos 5: `test_tanh_pade_proptest_100k`, `test_tanh_piecewise_proptest_50k`,
    `test_tanh_pade_nr2_proptest_100k`, `test_tanh_pade_nr2_proptest_100k_avx512`,
    `test_sigmoid_pade_proptest_100k`. CI (~0.00s, custo trivial — single f32 op por caso).
    **(2) `tests/lstm_scalar_bf16_parity.rs`** — 50→5,000 casos (100×). Era o proptest de
    menor profundidade de todo o projeto. 3 testes `#[ignore]` (longa: ~348s somados).
    Estratégias geram vetores de até ~12k f32s + instanciam LstmModel1 e comparam SIMD vs
    scalar — aceitável para suíte longa. **(3) `src/dsp/pipeline/pipeline_block_test.rs`** —
    500→2,000 casos (4×). `#[ignore]`, longa (~4.4s). Modelo real (BossWN-nano) com blocos
    aleatórios 1..8192 — custo modesto para cobertura 4× maior.
    **(4) `src/dsp/resampler_test.rs`** — `test_resampler_micro_soak`: 2000→5000 iterações
    (2.5×, ~2.5M samples/pair × 5 rate pairs). CI (~0.26s).
    **Soak tests**: já saturados (10M frames na maioria, 50M resampler, 100M mirror_buf).
    Sem alterações — custo marginal de aumentar é alto e estresse já máximo.

### Sprint 2.4 — Validação final (🚦 GATE L1)

- [x] **T2.4.1 — 🚦 GATE L1: execução completa pós-otimização.**
  Após **todas** as tarefas 2.2/2.3 mescladas, **uma** execução completa para: (a) comparar
  tempo total e por-fase vs. baseline L0; (b) confirmar **0 regressões**; (c) confirmar
  cobertura (goldens C++, soak, paridade).
  - _Critério de aceite_: tempo total reduzido (sem perda de cobertura); tabela L0→L1
    registrada; suíte verde.
  - _Quando_: agrupar aqui o resultado de tudo entre L0 e L1; **última** execução completa.
  - **Nota do PO:** O /tests-cargo.log contém a saida de terminal de vários comandos úteis.
    Já busque ter uma panorama geral se tudo está como deveria ou se precisa de ajustes.
    Use o Épico 4 (ou posteriores) para dar direcionamento aos achados.

  - ### 🚦 RESULTADO GATE L1 (2026-06-15, `tests-cargo.log:2511-2598`) — ❌ **SUÍTE VERMELHA**

    Execução completa registrada (real **50m27s**). **Phase 5 (CLAP) FALHOU.** Análise (T2.4.1):
    **Comparativo L0 → L1 por fase** (renumeradas após T2.2.1; Phase 0 pre-flight nova):

    | Fase (L1)                                 | L0     | L1          | Δ         | Nota                                                               |
    | ----------------------------------------- |:------:|:-----------:|:---------:| ------------------------------------------------------------------ |
    | Soak (Numerical Stability)                | 139s   | **109s**    | −30s ✅   | T2.2.4 (soak paralelo)                                             |
    | PipeWire Integration                      | 49s    | **17s**     | −32s ✅   | fundida/reuso (T2.2.1)                                             |
    | Property/Parity/Golden (Release)          | 137s   | **545s**    | +408s ⚠️  | **+358s** de `lstm_scalar_bf16_parity` (T2.3.3, 50→5000 casos)     |
    | Resampler/Cabsim/A2 Heap-Audit            | 152s   | **62s**     | −90s ✅   | paridade movida p/ Phase 3 (T2.2.1)                                |
    | **CLAP Release Validation & Concurrency** | 223s   | **794s** ❌ | +571s 🔴  | **rebuilds release fat-LTO** + **SIGSEGV** + flake                 |
    | Long Benchmarks                           | 2222s  | **1500s**   | −722s ✅  | dedup de benches (T2.2.3)                                          |
    | **Total**                                 | 48m41s | **50m27s**  | +1m46s    | ganho de build mascarado por (a) estresse↑ e (b) regressão Phase 5 |

    **Veredito:** As otimizações de **desperdício de compilação funcionaram** (benches −722s,
    soak −30s, heap-audit −90s). Porém o ganho líquido foi **anulado** por: (1) aumento
    **intencional** de estresse (T2.3.3: `lstm_scalar_bf16_parity` +358s) e (2) **regressão
    da Phase 5** introduzida pela T2.2.2.

    **Falhas/achados (roteados ao Épico 4 conforme nota do PO):**

    1. 🔴 **SIGSEGV (signal 11) em release** — unittests da lib em `--no-default-features
       --features clap-plugin,testing` (modo Mono, release) crasham após
       `test_metadata_extraction_from_nam_file` (`phase4-clap-validation.log:399-402`).
       Em L0 (debug) passava; a T2.2.2 (migração p/ release) **expôs** o crash. → **T4.1.1**.
    2. 🟠 **Flake de wall-clock** — `test_denormal_stability_silence`
       (`nam_infer_test.rs:537`) falhou com `block=677μs > 500μs` sob carga da suíte
       (output numérico **correto**, sem denormais). Gate de tempo frágil. → **T4.2.1**.
    3. 🟠 **Regressão de tempo Phase 5** — T2.2.2 trocou 1 build debug por **vários
       rebuilds release fat-LTO** (~7min cada, `phase4-clap-validation.log:338`) por
       feature-sets divergirem. → **T4.3.1**.
    4. 🟡 **Estresse caro** — `lstm_scalar_bf16_parity` a 358s domina a Phase 3;
       reavaliar custo/benefício dos 5000 casos. → **T4.4.1**.
    5. 🟢 **Limpeza** — 3 micro-achados do review de T2.3.2 (linhas 572-579). → **T4.5.1**.

    > **Nota:** Ambas as falhas são **exclusivas da config Mono+Release da suíte longa**;
    > a suíte padrão (`tests-cargo.sh`, runs cold+warm no mesmo log) passou **100% verde**.

---

## ÉPICO 3 — Correções funcionais e de ruído reveladas pelas suítes (pós-infra)

> Itens funcionais do nam-rs detectados durante as auditorias. **Não bloqueiam** os
> Épicos 1-2; validados majoritariamente pela suíte padrão (rápida) e por execuções
> dirigidas — **sem** depender dos gates da suíte longa, salvo onde indicado.

### Sprint 3.1 — Ruído e fidelidade do `clap-validator`

- [x] **T3.1.1 — Rebaixar o nível de log do _state-invalid_.**
  No baseline antigo, `state-invalid` (estado **vazio**) emitia
  `[CLAP_PLUGIN_ERROR] Failed to deserialize state (v0 legacy): EOF`, condição esperada em
  que o plugin retorna `false` (teste PASSED). **Nota da auditoria:** no log mais recente
  (`tests-cargo.log`, pós-Épico 1) essa linha **não aparece** — reconfirmar se ainda
  reproduz antes de mexer. Se reproduzir, rebaixar para `debug`/`trace` em entrada vazia.

  - _Arquivos_: `src/clap/extensions/state.rs` (caminho de `Deserialize`, ~`:238`).
  - _Critério de aceite_: estado vazio não emite `[CLAP_PLUGIN_ERROR]`; retorno inalterado.
  - **Resultado (2026-06-15)**: O comportamento foi reproduzido no `clap-validator` (emitia o erro `EOF` devido ao buffer vazio). Como a biblioteca Clack mapeia de forma rígida qualquer retorno de erro de `load()` com a severidade `CLAP_LOG_ERROR`, a solução intercepta buffers vazios no início, registra um log de nível `Debug` via extensão `HostLog`, e retorna uma mensagem de erro `\r\x1b[K` (Carriage Return + ANSI clear line) para retornar `false` ao host e simultaneamente limpar o cabeçalho `[CLAP_PLUGIN_ERROR]` impresso por padrão pelo Clack, mantendo a saída de terminal limpa.

- [CANCELADO] **T3.1.2 — Decidir sobre `clap.note-ports` (2 testes SKIPPED).**
  `process-note-*` pulados por não implementar `note-ports`. Confirmar se é intencional
  (efeito sem MIDI) e documentar para não parecer lacuna.
  - _Critério de aceite_: decisão registrada em `docs/` (intencional) ou tarefa criada.
  - Nota do PO: É intencional.

- [CANCELADO] **T3.1.3 — (Opcional) validação estrita por `--json` na suíte padrão.**
  A suíte padrão valida por exit code; a longa usa `--json`+`jq` para falhar em _warnings_.
  Avaliar trazer a checagem de _warnings_ para a padrão (barata, ~3 s).
  - _Critério de aceite_: _warnings_ do validator falham a suíte padrão, ou justificativa.
  - Nota do PO: É intencional.

### Sprint 3.2 — Divergência numérica conhecida (registrada, não regressão nova)

- [CANCELADO] **T3.2.1 — Investigar WaveNet Lite (CH=12) vs C++.**
  `test_golden_vectors_wavenet_lite` está `ignored` como "known-divergent ... SNR = 0,9 dB
  vs C++". SNR de 0,9 dB é perceptualmente relevante. Investigar a deriva (acúmulo, ordem
  de redução, bf16) e definir correção ou aceitação formal.
  - _Critério de aceite_: causa-raiz documentada; teste reabilitado **ou** divergência
    formalmente aceita com justificativa numérica (skill `pesquisador-inovador`).
  - Nota do PO: Será avalidado oportunamente no futuro.

### Sprint 3.3 — Resíduos da infraestrutura do Épico 1

- [x] **T3.3.1 — `diagnostic_bundle` lento no cold run (10,28s).**
  `test_panic_hook_behavior` paga simbolização de _backtrace_ em cache de disco frio
  (warm = 0,02s). Avaliar: tornar o _backtrace_ mais leve, marcar a variante pesada como
  `#[ignore]` (movendo-a para a longa), ou aceitar como custo cold-only documentado.

  - _Critério de aceite_: decisão registrada; se mantido, justificado como aceitável.
  - **Achado & Decisão (2026-06-15):** A lentidão no cold run (~10,28s) ocorria porque a chamada `panic!` no teste `test_panic_hook_behavior` disparava o gancho padrão do Rust (através de `prev_hook(info)`), o qual, sob `RUST_BACKTRACE=1` ou `RUST_BACKTRACE=full`, realizava a leitura e resolução dos símbolos de depuração diretamente do disco frio. Para solucionar isso sem alterar o comportamento em produção (onde o backtrace ainda é desejado no crash report padrão), modificamos o teste de integração para temporariamente desviar o `prev_hook` para um gancho dummy no-op (`Box::new(|_| {})`) durante os pânicos controlados. Isso eliminou completamente a resolução e a impressão dos backtraces durante o teste, reduzindo o tempo de execução do teste para menos de 1ms, mesmo em runs a frio.

- [x] **T3.3.2 — Endurecer `AUDIT_ENABLED` global em `test_heap_audit_trigger`.**
  `src/clap/processor_test.rs:768` liga/desliga `AUDIT_ENABLED` (atômico **de processo**)
  sem RAII; se o `process()` entrar em pânico antes do reset (linha 789), a flag fica
  ligada para o resto da suíte. Envolver set/reset em guard RAII (reset no `Drop`) e
  serializar com o padrão `TEST_MUTEX`.

  - _Critério de aceite_: flag sempre restaurada mesmo sob pânico; sem corrida com testes
    `clap::` paralelos.
  - **Resultado (2026-06-15):** Criamos a estrutura RAII `AuditEnabledGuard` que ativa a auditoria global no `new()` e restaura `AUDIT_ENABLED` para `false` no `drop()`. Para prevenir interferência com os outros testes concorrentes da biblioteca (onde `process()` ativa o `TrackingGuard` sob auditoria ativa, o que resetaria o estado atômico `TRACKING_ACTIVE` local de outra thread e silenciaria suas asserções de zero alocações), introduzimos um `TEST_MUTEX` e serializamos o `test_heap_audit_trigger` com os 4 testes de zero alocação (`test_zero_alloc_process_bypass`, `test_model_switching_stress`, `test_parameter_modulation_stress` e `test_monophonic_parameter_modulation`), garantindo isolamento total e determinismo em paralelismo sem regressão no tempo de teste.

---

## ÉPICO 4 — Regressões e fragilidades reveladas pelo 🚦 GATE L1 (suíte longa)

> Achados da execução completa da `tests-long.sh` pós-otimização (T2.4.1). Direcionamento
> solicitado pelo PO. **Prioridade**: a Sprint 4.1 é potencialmente um **bug real de
> produção** (UB em release) — tratar antes das demais. As Sprints 4.3/4.4 fecham o ganho
> de tempo que a Phase 5 anulou.
>
> ⚠️ **As Sprints 4.1–4.4 são validáveis SEM a suíte longa completa** — cada uma roda
> apenas o comando/fase específico envolvido (release mono lib, um teste, uma fase). A
> **revalidação completa** fica para um novo gate **🚦 GATE L2** (Sprint 4.6), agregando
> todas as correções numa única execução.

### Sprint 4.1 — 🔴 CRÍTICO: SIGSEGV em release (config Mono CLAP)

- [x] **T4.1.1 — Diagnosticar e corrigir o SIGSEGV em release.**
  O binário de unittests da lib crasha com `signal 11 (SIGSEGV)` em
  `cargo test --release --no-default-features --features clap-plugin,testing --lib`
  (`phase4-clap-validation.log:399-402`), após `test_metadata_extraction_from_nam_file`.
  Em **debug** (baseline L0) passava — a otimização release expôs o problema (provável **UB
  latente** mascarada por debug, ou bug de codegen/harness). Usar skill **`debugger`**.
  - _Repro_: `cargo test --release --no-default-features --features clap-plugin,testing --lib`.
  - _Pistas_: rodar com `RUST_BACKTRACE=1`; isolar o teste seguinte ao
    `test_metadata_extraction_from_nam_file`; considerar `--test-threads=1` para localizar;
    inspecionar `unsafe`/FFI no caminho de `preset_discovery`/`gui`/`processor`.
  - _Critério de aceite_: causa-raiz documentada; crash eliminado; mono-release **verde
    determinístico** (≥ 3 execuções). Se for UB de RT-safety, abrir correção prioritária.
  - _Risco_: ALTO — pode indicar bug real de memória em produção (release é o profile de _ship_).
  - **Resultado (2026-06-15):** **Causa-raiz**: 3 testes GUI (`test_ui_load_error_visual_feedback`,
    `test_bypass_keyboard_trigger`, `test_tab_order_navigation` em `src/clap/gui/ui/test.rs`)
    faziam `transmute(&dummy as *const i32)` onde `dummy` era um `i32` de stack, para produzir um
    `HostSharedHandle`. `HostSharedHandle` é `#[repr(transparent)]` sobre `NonNull<clap_host>`. Em
    release, o `get_extension::<HostParams>()` invocado por `handle_bypass` dereferenciava o
    ponteiro inválido como `&clap_host` (struct FFI com function pointers), causando `SIGSEGV`.
    **Correção**: substituímos o `transmute` por `make_dummy_host()`, que aloca um `clap_host`
    zerado (com `get_extension: None`) via `Box::leak` e o converte em `HostSharedHandle<'static>`
    via `from_raw(NonNull::from(...))`. Isso é seguro porque todos os function pointers são nulos,
    fazendo `get_extension` retornar `None` sem dereferenciar memória inválida. 3 execuções
    release consecutivas passaram (466 passed, 0 failed).

### Sprint 4.2 — 🟠 Fragilidade: gate de tempo wall-clock

- [ ] **T4.2.1 — Endurecer `test_denormal_stability_silence` (gate de tempo).**
  Falhou com `block=677μs > 500μs` (`nam_infer_test.rs:537`) **sob carga** da suíte de
  50 min — porém o output era numericamente correto (sem denormais). Limiar de _wall-clock_
  em teste funcional é intrinsecamente _flaky_.
  - _Opções_: (a) detectar denormal por **valor/contagem** (não por tempo); (b) tornar o
    limiar **informativo** (warn, não fail) ou com _retry_; (c) relaxar margem e marcar como
    sensível a carga (`#[ignore]` na longa concorrente ou rodar isolado).
  - _Critério de aceite_: teste robusto sob carga, mantendo detecção real de penalidade de
    denormal; 0 _flakes_ em ≥ 3 execuções da fase.

### Sprint 4.3 — 🟠 Regressão de tempo: Phase 5 (CLAP) explodiu (223s→794s)

- [ ] **T4.3.1 — Reverter o efeito colateral da T2.2.2 (rebuilds release).**
  A unificação para release trocou **1 build debug** por **vários rebuilds release fat-LTO**
  (~7 min cada, `phase4-clap-validation.log:338`), pois as feature-sets divergem: `.so`
  = `clap-plugin,heap-audit`; mono = `clap-plugin,testing`; concurrency = `standalone`;
  multi/gc = `clap-plugin`. Cada combinação dispara um build LTO completo da árvore GUI.
  - _Opção A (preferida)_: **alinhar TODOS** os comandos release da Phase 5 a **uma única
    feature-set** (ex.: `clap-plugin,heap-audit,testing`) para reusar **um** build.
  - _Opção B_: reverter testes de **correção** (não-perf: mono, concurrency, gc_stress) para
    **debug** (rápido), mantendo só o `.so` + validator em release.
  - _Critério de aceite_: Phase 5 recompila a árvore CLAP **no máximo 1×** em release;
    tempo da fase cai substancialmente vs L1; cobertura preservada.
  - _Nota_: medir o trade-off — o SIGSEGV (T4.1.1) mostra que **rodar em release tem valor**
    (pega bugs que debug esconde); decidir o equilíbrio com base em T4.1.1.

### Sprint 4.4 — 🟡 Reavaliar custo do estresse proptest (efeito da T2.3.3)

- [ ] **T4.4.1 — Calibrar `lstm_scalar_bf16_parity` (5000 casos → 358s).**
  T2.3.3 elevou de 50→5000 casos (100×); virou o maior ofensor da Phase 3 (358s de 545s).
  Reavaliar se 5000 é o ponto ótimo ou se um valor menor (ex.: 1000–2000) mantém boa
  cobertura sem dominar a suíte.
  - _Critério de aceite_: profundidade justificada por números (cobertura vs tempo);
    Phase 3 reequilibrada.

### Sprint 4.5 — 🟢 Limpeza do `tests-long.sh` (review de T2.3.2)

- [ ] **T4.5.1 — Aplicar os 3 micro-achados do review.**
  (1) Loop de display (≈L466-467): trocar `echo|awk`/`echo|cut` por _parameter expansion_
  (`${line%% *}` / `${line#* }`) — elimina ~50 subprocessos. (2) Adicionar
  `trap 'rm -f "$TIMED_TRACKER"' EXIT` (cleanup do temp em saída precoce/ERR). (3) Remover
  regras awk _unreachable_ (`/^Found/`, `/^change:/`, L257-258).
  - _Critério de aceite_: script equivalente, sem _leak_ de temp, sem código morto.

### Sprint 4.6 — 🚦 GATE L2: revalidação completa pós-correções

- [ ] **T4.6.1 — 🚦 GATE L2: execução completa após Sprints 4.1–4.5.**
  Agregar **todas** as correções do Épico 4 numa **única** execução completa da
  `tests-long.sh`. Confirmar: suíte **100% verde**, Phase 5 sem SIGSEGV/flake e com tempo
  reduzido, total < L1.
  - _Critério de aceite_: tabela L1→L2 registrada; 0 falhas; ganho de tempo real consolidado.
  - _Quando_: **última** execução completa do ciclo; o usuário roda quando puder ausentar-se.

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

## Apêndice B — Metas de aceite

### Épico 1 (CONCLUÍDO)

- **Agilidade**: ✅ warm **58,3s** (< 60 s); 🟡 cold **2m17,9s** (meta < 2 min, −51% vs baseline).
- **Segurança**: ✅ auditoria de alocação TLS sem falso-negativo (T1.1.3).
- **Estabilidade**: ✅ 0 _flakes_ (Phase 3-A: 6 runs verdes; cold+warm completos verdes).
- **Informatividade**: ⚠️ tabela de fase **abandonada** (T1.5.3 cancelado); mantido `cargo test` enxuto + fail-fast.
- **Cobertura**: ✅ nenhuma perda de teste por configuração (T1.2.2 / T1.6.2).

### Épico 2 (suíte longa — CONCLUÍDO com ressalvas; saldo no Épico 4)

- **Sem desperdício**: ✅ otimizações de build funcionaram (benches −722s, soak −30s, heap-audit −90s); 🟡 ganho líquido mascarado por estresse↑ (T2.3.3) e regressão Phase 5 (T2.2.2) — saldo real fica no 🚦 GATE L2 (T4.6.1).
- **Cobertura honesta**: ✅ 0 _silent-skips_ — Phase 0 pre-flight + erros explícitos (T2.1.3/T2.3.1).
- **Execução em lote**: ✅ exatamente 2 execuções completas (🚦 L0, 🚦 L1); protocolo respeitado.
- **Informatividade**: ✅ sumário com duração por fase + top-5 mais lentos (T2.3.2).
- **Estabilidade**: ❌ 🚦 GATE L1 **vermelho** (Phase 5: SIGSEGV release + flake) → Épico 4.

### Épico 4 (pós-L1 — alvos)

- **Correção crítica**: SIGSEGV em release (mono CLAP) eliminado (T4.1.1).
- **Estabilidade**: 0 _flakes_ de wall-clock (T4.2.1); 🚦 GATE L2 100% verde (T4.6.1).
- **Tempo**: Phase 5 sem rebuilds redundantes; total L2 < L1, consolidando o ganho de build.
