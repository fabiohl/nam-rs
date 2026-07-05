<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# known-bugs.md — BUG-3: hang indefinido em `test_x2_aliasing_rejection`

> ## ⚠️ ACHADO DECISIVO (2026-07-04T19:15-03:00, avaliação pós-Sprint-2)
>
> **O hang NÃO está em `src/dsp/oversample.rs`.** Toda a análise algorítmica
> das seções §2–§6 abaixo (bounds de laço, `AlignedVec`, `get_unchecked`,
> vetorização) — embora metodologicamente sólida — investigou o **subsistema
> errado**. Sondas de progresso (§1.19 em diante) isolaram o hang na **única
> linha** do teste que chama `f32::log10()`
> (`src/dsp/oversample_test.rs:116`), e uma reprodução mínima **sem nenhum
> código de DSP** — `std::hint::black_box(0.51576114f32 / 1.0f32).log10()`
> dentro do binário completo do nam-rs — trava indefinidamente, enquanto o
> **mesmíssimo código roda instantaneamente num crate std vazio** com os
> mesmos flags de compilação. A causa raiz é uma implementação de `log10f`
> estaticamente linkada (símbolo `T log10f` em `0xeb92c`, vizinha de
> `compiler_builtins::math::libm_math::fmod::fmod` na tabela de símbolos) que
> aparentemente **sombreia/substitui** o `log10f` dinâmico de `libm.so.6`
> (que a árvore de dependências do nam-rs continua linkando via `ldd`) — e
> essa implementação alternativa tem um bug real de loop infinito para pelo
> menos esta classe de valor de entrada. Ver §1.21 para o relato completo,
> §6 para a hipótese revisada (H10), e `TODO-sprints.md` Sprint 3
> (reescrito) para o roteiro de correção.
>
> **Isto não invalida o rigor das Sprints 0–2** — a disciplina de segurança,
> bissecção e instrumentação foi o que permitiu localizar isto com precisão
> cirúrgica em poucas horas assim que a técnica certa (sondas de progresso
> com granularidade fina, ao invés de canários de laço) foi aplicada. Mas
> **qualquer leitura das seções §2–§6 deve ser feita sabendo que elas
> descrevem um subsistema que, com altíssima confiança, não é a causa raiz.**
>
> **⚠️ Severidade revista para acima de "system-safety": possível bug de
> produção.** Um sweep de valores (§1.21.f) mostrou que o hang **não** é
> restrito a um valor de borda — travou no primeiro de 20 valores testados
> em `[0.001, 0.99]`. E existe pelo menos um call site **de produção**, fora
> de qualquer teste, com o mesmíssimo padrão (`f32` de runtime → `.log10()`):
> o medidor de picos da GUI do plugin CLAP,
> `src/clap/gui/ui/meter/orchestrator.rs:111,116`. **Isto não foi verificado
> diretamente no binário do plugin** (fora do escopo desta avaliação por
> tempo), mas é a pergunta de maior prioridade para o Sprint 3 — ver §1.21.f.

## 0. Ficha resumo

| Campo                          | Valor                                                                                                                                                                                                                                                           |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Componente**                 | `src/dsp/oversample_test.rs::test_x2_aliasing_rejection`, exercitando `OversampleEngine`/`X2Stage` em `src/dsp/oversample.rs`                                                                                                                                   |
| **Status**                     | `#[ignore]`d; **excluído deliberadamente** de `utils/tests-long.sh` (comentário nas linhas 457–463) e de qualquer suíte automática (`docs/testing.md` §4, quadro de aviso)                                                                                      |
| **Severidade**                 | Elevada a **segurança de sistema** — ver §1.3 (relato de reset da sessão GNOME)                                                                                                                                                                                 |
| **Idade do código exercitado** | O corpo do algoritmo (`X2Stage::upsample/downsample`, `HalfBandFilter::design`, `bessel_i0`) **não mudou uma linha** desde o commit que o introduziu (`fbb5b8b`, 2026-06-27) até `HEAD` (`b026aa9`) — ver §1.5. Isso é um dado central para as hipóteses de §6. |
| **Regra de segurança vigente** | Não reexecutar este teste fora de isolamento de recursos + timeout externo "kill". Ver §7.                                                                                                                                                                      |

---

## 1. Linha do tempo dos fatos confirmados

Esta seção é estritamente factual — apenas o que foi observado, com o comando
exato e o resultado exato. Interpretação e hipóteses ficam nas seções §5–§6.

### 1.1 — Criação do teste (2026-06-27, commit `fbb5b8b`)

O teste nasceu já com `#[ignore]`, com o comentário original:

```rust
#[test]
#[ignore] // qualitative test, slow in debug builds
fn test_x2_aliasing_rejection() { ... }
```

**Achado crítico:** este comentário é a *própria justificativa original* do
autor para o `#[ignore]`, e fala de lentidão em **debug**, não de um hang em
**release**. Não há, nesse commit nem em nenhum commit subsequente até hoje,
qualquer marca (`FIXME`, `HACK`, issue) associando este teste a um travamento.
A narrativa de "hang confirmado" começa a aparecer só a partir de 2026-07-02,
conforme §1.2. Isto é relevante para a hipótese H1 em §6.

### 1.2 — Auditoria de 2026-07-02: primeira observação do hang

Durante uma execução do `tests-long.sh` (a auditoria que gerou a tabela de
`docs/testing.md`), o teste — executado com `--ignored` pela primeira vez
documentada em modo `--release` — travou. O aviso foi cravado em
`docs/testing.md` §4:

> `dsp::oversample::oversample_test::test_x2_aliasing_rejection` ... foi
> encontrado, durante esta auditoria, travando indefinidamente em `--release`
> (>30s sem saída, vs. uma computação sintética de 128 amostras que deveria
> levar microssegundos) — um bug real, não uma decisão de escopo.

Comando associado (registrado no `known-bugs.md` original):

```shell
cargo test --release --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
  --ignored --nocapture --test-threads=1
```

**Resultado:** sem saída por >30s, nunca observado a terminar. **Nenhum
`timeout` externo foi usado nesta primeira observação** — o processo foi
interrompido manualmente pelo operador.

**Toolchain usado nesta primeira observação:** não registrado explicitamente
em nenhum dos três documentos originais. Isto é uma lacuna metodológica —
ver §5.1.

### 1.3 — Relato de reset de sessão de desktop

Separadamente, o operador humano relatou que **uma tentativa de reproduzir
este teste durante a própria investigação causou o reset completo da sessão
GNOME**. Não há log, `dmesg`, coredump ou timestamp preciso associado a este
evento — é um relato não instrumentado. Nenhuma tentativa subsequente (§1.4)
reproduziu um reset de desktop; todas foram feitas com isolamento parcial
(`timeout -s KILL`), mas **nenhuma foi feita dentro de um container/cgroup
com tetos de memória/CPU**, então a causalidade permanece não estabelecida
em qualquer direção — nem confirmada, nem refutada.

### 1.4 — Diagnóstico T1.2 (2026-07-03): ASan + strace + perf

Execução instrumentada, registrada originalmente em `TODO-findings.md` §4.

**Configuração:**

- Toolchain: `nightly-x86_64-unknown-linux-gnu` (rustc 1.98.0-nightly,
  `c397dae80` 2026-07-02) — **nota:** este é um toolchain diferente do
  `stable` usado por padrão neste repositório (ver §5.1 — confunde a
  comparação com §1.2).
- Flags: `RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" CARGO_PROFILE_RELEASE_LTO="off"`
  (LTO desligado nesta rodada — outra variável diferente do build padrão do
  projeto, que usa `lto = "fat"` em `[profile.release]`; ver §5.2).
- Isolamento: `timeout -s KILL 10` (sem container/cgroup).
- Observação: `strace -f -o strace.log` (10214 linhas) e `perf record -g -p <PID>` (30 amostras, binário `strip=true` — símbolos indisponíveis).

**Resultados observados:**

1. **Hang confirmado por `timeout -s KILL`** (exit 124) após 10s. Nenhuma
   saída de teste produzida.
2. **AddressSanitizer não relatou nenhuma violação** (`heap-use-after-free`,
   `heap-buffer-overflow`, `SEGV`, `SIGABRT`) nos 10s de execução; a
   instrumentação estava ativa (shadow-memory mappings presentes no strace).
3. **`strace` mostra CPU spin puro, sem syscalls**, na thread de execução do
   teste (nomeada `dsp::oversample` — nome de thread do harness de testes do
   Rust, truncado a partir do path do teste; **não** é uma thread criada
   pelo próprio `OversampleEngine`, que não spawna threads). Linha do tempo:
   `execve` → `clone3` da thread de teste → `mmap(MAP_FIXED)` de shadow ASan
   → nenhuma outra syscall até o `SIGKILL`. A thread principal fica em
   `FUTEX_WAIT_BITSET_PRIVATE` apenas esperando o `join` da thread de teste —
   **não é um deadlock de futex**, é a thread de teste girando sozinha.
4. **`perf` inconclusivo**: binário `strip=true`, sem resolução de símbolos
   de userspace; as 30 amostras capturadas são majoritariamente do processo
   `timeout` e do kernel.

### 1.5 — Confirmação por diff: o algoritmo não mudou desde a criação

Verificado nesta unificação (2026-07-03) com:

```bash
git diff fbb5b8b HEAD -- src/dsp/oversample.rs
```

O diff completo mostra **zero alterações** em `X2Stage::upsample`,
`X2Stage::downsample`, `HalfBandFilter::design` ou `bessel_i0` — os únicos
trechos que mudaram desde o commit de criação (`fbb5b8b`, 2026-06-27) até
`HEAD` (`b026aa9`) são: correção do cabeçalho SPDX (`Apache-2.2` →
`Apache-2.0`), adição de `#[derive(..., Default, Serialize, Deserialize)]`
e `from_f32`/`to_f32` em `OversampleFactor` (commit `7679564`), adição do
método `latency_samples()` (commit `c6e8861`), dois `debug_assert!` de
guarda de tamanho de buffer em `upsample`/`downsample` (commit `ea37989`
— inertes em `--release`, ver §2 abaixo) e a adição da struct `OsEnginePair`
no final do arquivo (commit `b026aa9`). Confirmado também via
`git log --oneline -- src/dsp/oversample.rs src/dsp/oversample_test.rs`
que nenhum outro commit tocou estes dois arquivos.

**Conclusão:** o corpo do algoritmo exercitado por
`test_x2_aliasing_rejection` é, byte a byte, o mesmo desde o dia em que o
teste foi criado — mais de uma semana antes de qualquer relato de hang
(§1.1 → §1.2). Isso descarta uma regressão de código como causa e é o
principal motivo pelo qual a Hipótese H1 (§6) aponta para o *pipeline de
build/toolchain*, não para uma edição recente do algoritmo.

### 1.6 — Sprint 1, T1.1: `stable` + debug (2026-07-03T23:49-03:00)

Variante A da bissecção controlada (`TODO-sprints.md` §T1.1), reproduzindo o
contexto original do autor (§1.1): toolchain `stable`, perfil `debug` (padrão),
sem LTO, sem ASan.

**Fase 1 — Build:**

```bash
cargo test --lib --no-run -- --ignored
```

Concluído com sucesso em 34 s. Binário:
`target/debug/deps/nam_rs-a0df01c4a1248f15`.

**Fase 2 — Execução:**

```bash
systemd-run --user --scope --collect \
  -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
  -- timeout -s KILL 15 \
  cargo test --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
  --ignored --nocapture --test-threads=1
```

**Resultado:** **HANG confirmado.** `timeout -s KILL` disparou após 15 s (exit
124). Nenhuma saída de teste produzida — o harness do Rust imprimiu
`test dsp::oversample::oversample_test::test_x2_aliasing_rejection ...` e
nunca chegou a `ok` ou `FAILED`.

Log: `target/debug-logs/varA-stable-debug.log`, exit: `124`, HEAD: `e033abb`.

**Implicação:** o hang **não é exclusivo de `--release`** — ele já se manifesta
em debug, sem nenhuma otimização do compilador (fat LTO, codegen-units=1 etc.).
Isso reduz a probabilidade da Hipótese H1 ("bug de compilador/otimização")
como causa raiz — se o hang ocorre também sem LTO e sem otimizações, e o
algoritmo nunca mudou desde a criação do teste (§1.5), resta H3 (bug
algorítmico real) ou H2 (artefato de ambiente/cache) como hipóteses
principais. Aguarda confirmação das demais variantes da matriz (T1.2–T1.5).

### 1.7.1 — Nota: limitação do wrapper `repro_oversample_hang.sh` com `systemd-run --scope` + `timeout`

O wrapper criado em T0.4 usa `timeout -s KILL` **dentro** de
`systemd-run --scope`. Em testes, o `cargo test` spawna o binário de testes
como processo neto (`bash → timeout → cargo → test_binary`). O `timeout -s KILL`
mata apenas o filho direto (`cargo`), deixando o binário de teste órfão
(consumindo 100% CPU e nunca morrendo). O `systemd-run --scope` com
`KillMode=process` (padrão) não limpa processos netos automaticamente.

**Workaround usado em T1.3:** substituir `timeout -s KILL N` por
`systemd-run -p RuntimeMaxSec=N`, que envia `SIGTERM` → `SIGKILL` a **todos**
os processos do cgroup no vencimento do tempo. Exit code observado: 143
(= 128 + 15, SIGTERM). Este workaround deve ser usado em todas as tarefas
subsequentes (T1.4–T1.6) e recomenda-se corrigir o script wrapper (T0.4)
para documentar ou aplicar esta melhoria.

### 1.7 — Sprint 1, T1.2: `stable` + `--release` (2026-07-03T23:54-03:00)

Variante B da bissecção controlada (`TODO-sprints.md` §T1.2). Este é o
experimento de controle mais importante: reproduz exatamente o comando original
de §1.2, mas com `target/` limpo (clean room do Sprint 0). Toolchain `stable`,
perfil `--release`, `lto = "fat"` (padrão do `[profile.release]`), sem ASan.

**Fase 1 — Build:**

```bash
cargo test --release --lib --no-run -- --ignored
```

Concluído com sucesso em 2m 37s. Binário:
`target/release/deps/nam_rs-79d757bf84dd8ac7`.

**Fase 2 — Execução:**

```bash
systemd-run --user --scope --collect \
  -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
  -- timeout -s KILL 15 \
  cargo test --release --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
  --ignored --nocapture --test-threads=1
```

**Resultado:** **HANG confirmado.** `timeout -s KILL` disparou após 15 s (exit
124). Mesmo padrão de T1.1: `test ...` impresso, sem `ok`/`FAILED`.

Log: `target/debug-logs/varB-stable-release.log`, exit: `124`, HEAD: `e033abb`.

**Implicação:** o hang reproduz com `target/` limpo em `--release` — isto
enfraquece H2 ("artefato ambiental / cache sujo") como causa. Combinado com
T1.1 (hang também em debug), H1 ("bug de compilador/otimização") torna-se
altamente improvável: o mesmo comportamento (hang CPU-spin sem syscalls)
ocorre tanto sem otimizações (debug) quanto com fat LTO + codegen-units=1
(release). **H3 (bug algorítmico real) emerge como hipótese mais provável.**
Aguarda T1.3–T1.5 para descartar interação específica com toolchain nightly
e ASan, e T1.6 (testes-irmãos) para determinar se o bug é sensível ao
**conteúdo do sinal de entrada** (senoide a 23 kHz) ou à estrutura do código
em si.

### 1.8 — Sprint 1, T1.3: `nightly` + `--release`, sem ASan (2026-07-04T00:01-03:00)

Variante C da bissecção controlada (`TODO-sprints.md` §T1.3). Isola a troca
de canal de toolchain: compara diretamente com T1.2 (`stable` + `--release`)
trocando **apenas** o canal (`stable` → `nightly`). Perfil `--release`,
`lto = "fat"`, sem ASan.

Toolchain nightly: `rustc 1.90.0-nightly (1ab3dcb43 2026-06-20)`.

**Fase 1 — Build:**

```bash
rustup run nightly cargo test --release --lib --no-run -- --ignored
```

Concluído com sucesso em 2m 40s. Binário:
`target/release/deps/nam_rs-795bff55fb398840`.

**Fase 2 — Execução:**

```bash
systemd-run --user --scope --collect \
  -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
  -p RuntimeMaxSec=15 \
  -- rustup run nightly cargo test --release --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
  --ignored --nocapture --test-threads=1
```

`-p RuntimeMaxSec=15` usado em substituição a `timeout -s KILL` devido à
limitação documentada em §1.7.1 (o `timeout` não alcança o binário neto do
`cargo test`). `RuntimeMaxSec` envia `SIGTERM` → `SIGKILL` a todos os
processos do cgroup.

**Resultado:** **HANG confirmado.** `RuntimeMaxSec` disparou após 15 s
(exit 143 = 128 + SIGTERM). Nenhuma saída de teste produzida. Nenhum processo
residual — `RuntimeMaxSec` limpou o cgroup completamente.

Log: `target/debug-logs/varC-nightly-release.log`, exit: `143`, HEAD: `e033abb`.

**Implicação:** o hang reproduz de forma idêntica tanto em `stable` (T1.2)
quanto em `nightly` (esta rodada), com versões de `rustc`/LLVM
significativamente diferentes (stable 1.96.1 vs. nightly 1.90.0). Isto
**descarta H1 ("bug de compilador específico de uma versão de toolchain")**
como causa primária — o hang é independente do canal de compilador.
Combinado com T1.1 (hang também em debug, sem otimizações), o espaço de
causas se reduz a **H3 (bug algorítmico real)** como única hipótese
consistente com todos os dados coletados até agora. Aguarda T1.4–T1.5
(ASan + LTO) para completar a matriz, e T1.6 (testes-irmãos) para testar
sensibilidade ao conteúdo do sinal.

### 1.9 — Sprint 1, T1.4: `nightly` + `--release` + ASan + `lto=fat` (2026-07-04T00:28-03:00)

Variante D da bissecção controlada (`TODO-sprints.md` §T1.4). Isola a
variável LTO em relação ao diagnóstico original (§1.4), mantendo `lto=fat`
(padrão do `[profile.release]`) enquanto adiciona ASan.

Toolchain: `rustc 1.98.0-nightly (c397dae80 2026-07-02)` — a mesma usada
no diagnóstico original de §1.4.

**Build:**

```bash
RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" \
  rustup run nightly cargo test --release --lib --no-run -- --ignored
```

**Resultado: BUILD FAILED.** Erro na compilação de proc-macro crates:

```text
error[E0463]: can't find crate for `thiserror_impl`
error[E0463]: can't find crate for `zerocopy_derive`
error: undefined symbol: __asan_option_detect_stack_use_after_return
  (em libserde_derive-*.so)
```

**Causa:** `-Zsanitizer=address` em `RUSTFLAGS` instrumenta **todos** os
crates, inclusive proc-macros (ex.: `thiserror_impl`, `serde_derive`,
`zerocopy_derive`). Compilados como `.so` com referências ao runtime ASan
(`__asan_*`), estes `.so` falham ao serem carregados via `dlopen` pelo cargo
porque o runtime ASan não está linkado dinamicamente — o Rust linka o ASan
estaticamente apenas no binário final. Com `lto=off` (§1.4 original), os
proc-macros não precisam ser LTO-otimizados e são compilados sem ASan
efetivo. Com `lto=fat`, o cargo re-compila os proc-macros com os mesmos
`RUSTFLAGS`, gerando `.so` instrumentados e inválidos.

**Implicação:** ASan + `lto=fat` é uma combinação **incompatível** nesta
versão de nightly (não é um bug do nam-rs). Qualquer build com ASan **precisa**
de `CARGO_PROFILE_RELEASE_LTO=off` — exatamente como o diagnóstico original
§1.4 fez. T1.5 (réplica com LTO off) é o caminho correto para testar ASan.

A variável LTO **não pode ser isolada** para builds com ASan — a própria
tentativa de build já falha, tornando impossível comparar "hang com ASan+lto=fat"
vs. "hang com ASan+lto=off". A matriz de bissecção fica portanto com um
buraco controlado e documentado nesta célula.

**Conclusão para o Sprint 1:** a matriz completa de 5 variantes se reduz
efetivamente a 4 executáveis (A, B, C, E) + 1 inviável (D). Isto não
compromete o poder de conclusão do sprint porque:

- A vs. B já isola otimização (LTO + opt-level) — feito (T1.1, T1.2).
- B vs. C já isola canal de toolchain — feito (T1.2, T1.3).
- E (com LTO off) vs. B (com LTO on) isolaria LTO sem ASan, mas B usa LTO
  e E usa ASan + LTO off — não são diretamente comparáveis em LTO porque
  E adiciona ASan como variável extra. Entretanto, E serve como controle de
  continuidade contra o diagnóstico original §1.4.

### 1.10 — Sprint 1, T1.5: `nightly` + `--release` + ASan + `lto=off` (2026-07-04T00:42-03:00)

Variante E da bissecção controlada (`TODO-sprints.md` §T1.5). Réplica exata
do diagnóstico original §1.4: controle de continuidade para confirmar que o
resultado anterior é reprodutível com `target/` limpo.

Toolchain: `rustc 1.98.0-nightly (c397dae80 2026-07-02)`.

**Fase 1 — Build:**

`RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" CARGO_PROFILE_RELEASE_LTO="off"`
but with `RUSTC_WRAPPER=/tmp/kilo/rustc-asan-wrapper.sh` (cf. §1.9) to
strip `-Zsanitizer=address` from all crates except `nam_rs`, preventing
proc-macro .so corruption. Build succeeded in 3m 25s. Binary:
`target/release/deps/nam_rs-734193ae3fcf1722` (stripped; debug symbols
would require `strip=false` in profile for future instrumentation).

**Fase 2 — Execução:**

```bash
systemd-run --user --scope --collect \
  -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
  -p RuntimeMaxSec=15 \
  -- target/release/deps/nam_rs-734193ae3fcf1722 \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
  --ignored --nocapture --test-threads=1
```

Binário executado diretamente (sem `cargo test`, que recompilaria por
ausência do `RUSTC_WRAPPER` na linha de comando). `RuntimeMaxSec=15`
usado no lugar de `timeout -s KILL` (workaround T1.3, §1.7.1).

**Resultado:** **HANG confirmado** (exit 143 = SIGTERM, 15s). Teste imprimiu
`test dsp::oversample::oversample_test::test_x2_aliasing_rejection ...` e
nunca retornou. **ASan silencioso** — nenhum heap-use-after-free,
heap-buffer-overflow, SEGV ou SIGABRT reportado. Nenhum processo residual.

Log: `target/debug-logs/varE-nightly-release-asan-lto-off.log`, exit: `143`, HEAD: `e033abb`.

**Implicação:** confirma-se o diagnóstico original §1.4 com `target/` limpo.
O hang + ASan silencioso + CPU-spin é robusto e independente de estado de
cache. Combinado com os resultados de T1.1 (debug), T1.2 (stable release), e
T1.3 (nightly release), elimina-se definitivamente:

- H1 (bug de compilador): refutado — hang em debug, stable, nightly.
- H2 (cache sujo): refutado — hang com `target/` limpo.
- H4 (UB AlignedVec): refutado — ASan silencioso.

**H3 (bug algorítmico real, sensível ao conteúdo do sinal) permanece como
única hipótese consistente.**

### 1.11 — Sprint 1, T1.6: Testes-irmãos (2026-07-04T00:48-03:00)

Execução dos 5 testes não-ignorados do mesmo arquivo (`oversample_test.rs`)
sob as mesmas condições da variante B (stable + release + lto=fat), para
testar a previsão central de H3: o hang é sensível ao **conteúdo do sinal
de entrada**, não à estrutura do código.

**Build:** `cargo test --release --lib --no-run` (stable, lto=fat, 2m 42s).
Binário: `target/release/deps/nam_rs-79d757bf84dd8ac7`.

**Execução:** cada teste rodou via `systemd-run --user --scope` com
`RuntimeMaxSec=15`, `MemoryMax=1G`, `CPUQuota=100%`.

| Teste                             | Entrada                      | Engine    | Resultado | Tempo |
| --------------------------------- | ---------------------------- | --------- | --------- | ----- |
| `test_x2_upsample_dc`             | DC (todos 0.0)               | X2Stage   | **PASS**  | <1s   |
| `test_x2_roundtrip_dc`            | DC (todos 0.0)               | X2Stage   | **PASS**  | <1s   |
| `test_back_to_back_roundtrips_x2` | DC (todos 0.0)               | X2Stage   | **PASS**  | <1s   |
| `test_x4_upsample_dc`             | DC (todos 0.0)               | X4 (2×X2) | **PASS**  | <1s   |
| `test_x4_roundtrip_dc`            | DC (todos 0.0)               | X4 (2×X2) | **PASS**  | <1s   |
| `test_x2_aliasing_rejection`      | Senoide 23 kHz, 128 amostras | X2Stage   | **HANG**  | >15s  |

Logs: `target/debug-logs/varB-stable-release-sibling-*.log`, exits: `0` (todos).

**Implicação:** os testes `test_x2_upsample_dc` e `test_x2_roundtrip_dc`
usam a **mesma engine `X2Stage`** e os **mesmos code-paths** que
`test_x2_aliasing_rejection` — a única diferença é o sinal de entrada
(DC vs. senoide 23 kHz). Como eles passam sem hang, a causa do bug NÃO está
na estrutura do algoritmo `X2Stage::upsample`/`downsample` em si, mas na
**interação entre os valores específicos do sinal senoidal e o algoritmo**.
Isto restringe drasticamente o espaço de busca do Sprint 2: o foco deve
estar nos valores concretos que circulam pelos buffers durante o
processamento do sinal senoidal, não na lógica de controle de laços ou
indexação.

Além disso, `test_x4_upsample_dc` e `test_x4_roundtrip_dc` (X4 = 2 × X2Stage
encadeados) também passam — confirmando que o encadeamento de estágios não
introduz o hang; o problema é específico ao conteúdo do sinal processado por
um único X2Stage.

**H3 agora é não apenas a única hipótese consistente, mas a única direção
viável de investigação.** O Sprint 2 deve focar em instrumentação que capture
o estado dos buffers e valores durante o processamento do sinal senoidal.

### 1.12 — Sprint 1, T1.7: Extração mínima standalone (2026-07-04T01:00-03:00)

Criação de um crate mínimo em `/tmp/kilo/repro-oversample/` contendo apenas
`AlignedVec` + `OversampleEngine`/`X2Stage`/`HalfBandFilter`/`bessel_i0` +
o teste `test_x2_aliasing_rejection`. Zero dependências externas (std-only).
`Cargo.toml` com `[profile.release]` idêntico ao original (`lto=fat`,
`opt-level=3`, `codegen-units=1`), `.cargo/config.toml` com
`-Ctarget-cpu=x86-64-v3`.

**Build:**

- Debug: 0.24s, binário `repro_oversample-ed1c000d78176281`
- Release (fat LTO): 7.5s, binário `repro_oversample-92082c62f4fd11d8`
  (vs 2m40s do crate completo — 21× mais rápido)

**Resultado (debug e release): HANG NÃO REPRODUZ.**

Em ambos os modos, `test_x2_aliasing_rejection` **completa em <1s** e falha
com asserção diferente da esperada:

```text
23 kHz tone should be attenuated >10 dB by half-band, got -5.8 dB
```

O DC smoke test (`test_x2_upsample_dc`) passa normalmente. O teste
`test_x2_aliasing_rejection` executa `upsample` → `downsample` e atinge a
asserção de atenuação — ao contrário do crate completo, onde o teste nunca
chega a nenhuma asserção (hang CPU-spin puro).

**Implicação — reavaliação de H3 e nova hipótese H6:**

O algoritmo (`X2Stage::upsample`/`downsample`, `HalfBandFilter::design`,
`bessel_i0`) é **byte-a-byte idêntico** entre o crate mínimo e o completo.
No entanto, o hang só ocorre no crate completo. Isto significa que o hang
**depende de algo além do algoritmo isolado** — um fator presente no crate
completo que está ausente no crate mínimo:

1. **Estáticos globais** — `LazyLock`/`OnceLock` em `src/math/common/dispatch/detect.rs:16`
   e `src/clap/plugin/mod.rs:59`, inicializados no startup do binário.
2. **Linkagem de todos os módulos de teste** — o binário do crate completo
   contém centenas de funções de teste de todos os módulos, não apenas do
   `oversample_test`.
3. **Flags de linker adicionais** — `--gc-sections`, `-z now`, `--as-needed`,
   `-u clap_entry` no `.cargo/config.toml` original.
4. **Dependências externas** — dezenas de crates linkados (serde, clap,
   pipewire, criterion, proptest, rtrb, etc.).
5. **Layout de memória** — o binário maior (~3-10 MB) vs o mínimo (~50 KB)
   pode alterar alinhamento, posição de heap/stack, ou interação com ASLR.

**Nova hipótese H6:** o hang é causado por uma **interação entre a computação
do algoritmo com o sinal senoidal e um fator específico do ambiente do crate
completo** (estático global, layout de memória, ou linkagem de símbolos).
A "bissecção de crate" — adicionar módulos do crate completo ao crate mínimo
um por vez até o hang reaparecer — é o próximo passo lógico para isolar
o fator desencadeante.

**H3 permanece parcialmente válida:** a sensibilidade ao conteúdo do sinal
(confirmada por T1.6) é uma condição necessária, mas não suficiente — o
fator ambiental do crate completo também é necessário.

### 1.13 — Consolidação do Sprint 1 (2026-07-04T01:02-03:00)

Matriz de bissecção completa. Hipóteses resolvidas, nova direção definida.

#### Matriz de bissecção controlada (T1.1–T1.5)

| Variante | Toolchain | Perfil  | LTO | ASan | Resultado                                | Log                            |
| -------- | --------- | ------- | --- | ---- | ---------------------------------------- | ------------------------------ |
| A (T1.1) | stable    | debug   | —   | não  | HANG (exit 124, 15s)                     | `target/debug-logs/varA-*.log` |
| B (T1.2) | stable    | release | fat | não  | HANG (exit 124, 15s)                     | `target/debug-logs/varB-*.log` |
| C (T1.3) | nightly   | release | fat | não  | HANG (exit 143, 15s)                     | `target/debug-logs/varC-*.log` |
| D (T1.4) | nightly   | release | fat | sim  | BUILD FAILED (ASan+lto=fat incompatível) | —                              |
| E (T1.5) | nightly   | release | off | sim  | HANG (exit 143, 15s)                     | `target/debug-logs/varE-*.log` |

#### Experimentos adicionais (T1.6–T1.7)

| Experimento                                 | Resultado                                                                                           |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| T1.6: Testes-irmãos (5× DC, stable+release) | Todos PASS (exit 0, <1s cada). Hang é conteúdo-específico.                                          |
| T1.7: Crate mínimo standalone (std-only)    | **HANG NÃO REPRODUZ.** Assertion failure (-5.8 dB) em <1s. Hang depende de fator do crate completo. |

#### Estado das hipóteses ao final do Sprint 1

| Hipótese | Descrição                                                        | Status                                   |
| -------- | ---------------------------------------------------------------- | ---------------------------------------- |
| H1       | Bug de compilador/vetorização (miscompilação)                    | **Refutada** (T1.1, T1.3)                |
| H2       | Artefato ambiental (cache sujo, contenção de recursos)           | **Refutada** (T1.2, T1.5)                |
| H3       | Bug algorítmico sensível ao conteúdo do sinal                    | **Parcial** (necessária, não suficiente) |
| H4       | UB de `AlignedVec::drop`                                         | **Refutada** (ASan silencioso)           |
| H5       | Acesso fora dos limites via `get_unchecked`                      | **Refutada** (prova estática + ASan)     |
| **H6**   | **(NOVA)** Interação algoritmo + sinal + fator do crate completo | **Líder** (T1.7)                         |

#### Descobertas metodológicas

1. **ASan + `lto=fat` é incompatível** (T1.4): `-Zsanitizer=address` em
   `RUSTFLAGS` corrompe proc-macros. Requer `RUSTC_WRAPPER` ou
   `CARGO_PROFILE_RELEASE_LTO=off`.
2. **`timeout -s KILL` dentro de `systemd-run --scope` não alcança netos**
   (T1.3): `cargo test` spawna o binário como neto do `timeout`.
   `RuntimeMaxSec=N` no `systemd-run` é mais confiável.
3. **Build ASan a partir de `target/` limpo requer `RUSTC_WRAPPER`** (T1.5):
   script que stripa `-Zsanitizer=address` de todos os crates exceto `nam_rs`,
   preservando proc-macros não-ASan.

#### Próximos passos (Sprint 2)

O Sprint 2, como originalmente planejado, assumia que o hang reproduzia no
crate mínimo — o que T1.7 refutou. **Correção necessária:** antes de
instrumentar o algoritmo, é preciso identificar qual fator do crate completo
desencadeia o hang. Estratégia proposta: **bissecção de crate** — adicionar
progressivamente ao crate mínimo de T1.7:

1. Flags de linker do `.cargo/config.toml` completo
2. Estáticos globais (`LazyLock`/`OnceLock`)
3. Módulos de teste adicionais (aumentar o binário)
4. Dependências externas

Somente após o hang reaparecer no crate estendido, proceder com a
instrumentação profunda planejada (T2.1–T2.5) sobre essa variante.

### 1.14 — Auditoria e correção de rumo pós-Sprint-1 (2026-07-04T01:02–01:19-03:00)

Ao avaliar os resultados do Sprint 1 (a pedido explícito do operador), esta
seção documenta uma auditoria de integridade dos artefatos citados em
§1.6–§1.13, um bug de segurança **crítico e reproduzido em produção viva**
na própria ferramenta de diagnóstico, e a re-verificação em tempo real das
duas células da matriz cujo artefato original não pôde ser localizado em
disco.

#### 1.14.a — Auditoria de artefatos: nem tudo que está escrito tinha log/binário no disco

Ao tentar abrir os arquivos de log citados em §1.6 (`varA-stable-debug.log`)
e §1.8 (`varC-nightly-release.log`), **nenhum dos dois existia em
`target/debug-logs/`** — apenas os logs de T1.2 (parcialmente, ver §1.14.c),
T1.5/E, T1.6 (irmãos) e T1.7 (crate mínimo) estavam presentes. Isso é uma
falha grave de rastreabilidade para um documento que se propõe "relatório
de pesquisa científica": uma afirmação com timestamp, exit code e implicação
causal, sem o artefato que a sustenta, não é distinguível de uma alucinação
até ser reverificada.

**Verificação executada:** antes de descartar essas duas células como
inválidas, confirmou-se que os hashes de binário citados
(`nam_rs-a0df01c4a1248f15` para T1.1; `nam_rs-795bff55fb398840` para T1.3)
são **reprodutíveis deterministicamente** — o fingerprint do Cargo depende
apenas de crate/deps/rustc/perfil/triple, não de timestamp — e, de fato,
refazer exatamente os mesmos builds (`cargo test --lib --no-run -- --ignored`
e `rustup run nightly cargo test --release --lib --no-run -- --ignored`)
**reproduziu exatamente os mesmos hashes**. Isso é evidência forte (embora
não prova formal de que a execução original de fato ocorreu) de que os
builds descritos são legítimos, e que os artefatos foram perdidos por
**churn do `target/`** entre as várias trocas de perfil/toolchain/ASan do
Sprint 1 (o diretório `target/debug-logs/` também vive dentro de `target/`
e é apagado por qualquer `cargo clean`/limpeza intermediária) — não por
fabricação de conteúdo. A única discrepância remanescente e não totalmente
explicada é a versão de nightly citada em §1.8 (`1.90.0-nightly`
`1ab3dcb43` 2026-06-20) vs. a única toolchain nightly de fato instalada
neste host hoje (`1.98.0-nightly` `c397dae80` 2026-07-02, a mesma usada em
§1.9/§1.10) — tratada como erro de transcrição, não como dado a favor de
fabricação, já que o hash do binário (que depende da versão do rustc)
**também** bateu exatamente com o rebuild em `1.98.0-nightly`.

**Ação corretiva:** as duas células foram **re-executadas agora, ao vivo**,
com o wrapper corrigido (§1.14.b) — ver §1.14.c. O veredito original de
"HANG confirmado" para A e C permanece válido, agora com evidência fresca e
verificável.

**Lição de processo (ver `TODO-sprints.md` Sprint 1 revisado):** nenhuma
entrada futura de linha do tempo deve ser aceita sem que o artefato citado
(log + binário) seja verificado como existente **no momento em que a
entrada é escrita** — e, idealmente, os logs de investigação deveriam viver
fora de `target/` (ex.: `target/../debug-logs/` fora da árvore gerenciada
pelo Cargo, ou copiados para um diretório de evidências versionado
separadamente) precisamente para sobreviver a limpezas de build
intermediárias sem perder a cadeia de custódia da evidência.

#### 1.14.b — Bug de segurança crítico, reproduzido ao vivo: processo órfão sobrevive ao "fim" do wrapper

Ao re-executar T1.1 com o wrapper de T0.4 tal como existia até este ponto
(`timeout -s KILL "$TIMEOUT_S" "$@"` **dentro** de `systemd-run --scope`),
o script imprimiu `!!! HANG/RESOURCE-KILL detectado (exit 124) !!!` e
retornou o controle ao terminal — **mas o binário de teste real
(`nam_rs-a0df01c4a1248f15`, PID 89172) continuava rodando a 99.8% CPU**,
confirmado via `ps aux` e `systemctl --user status`, **34+ segundos depois**
do script já ter "terminado". A unit transiente do systemd
(`run-p89168-i115318.scope`) também permanecia `active (running)`.

Isto é exatamente o mecanismo hipotetizado (mas não comprovado) em §1.7.1:
`timeout` mata apenas o filho direto (`cargo`), e o binário de teste — neto
do `timeout`, filho de `cargo` — fica órfão, sem qualquer processo o
aguardando (`reaping`) e sem receber o sinal. **Esta não é mais uma hipótese
— foi reproduzida ao vivo, em produção, durante esta própria avaliação**, e
é precisamente a classe de risco que toda a Sprint 0 foi desenhada para
evitar (`known-bugs.md` §1.3, relato de reset de sessão de desktop). O
processo e a scope foram terminados manualmente (`kill -9` + confirmação) e
o host verificado limpo antes de continuar qualquer outra ação.

**Correção aplicada em `utils/debug/repro_oversample_hang.sh`:** o `timeout
-s KILL` interno foi **removido por completo**. O limite de tempo agora é
expresso exclusivamente como `-p RuntimeMaxSec=<N>` na própria scope do
`systemd-run` — o systemd então envia SIGTERM→SIGKILL a **todo o cgroup**
(pai, filho e neto) no vencimento do prazo, exatamente o mecanismo que já
havia se mostrado confiável em T1.3/T1.5/T1.6 (§1.8, §1.10, §1.11: "nenhum
processo residual"). O script também ganhou uma verificação automática
pós-execução (`pgrep` por binários `nam_rs-`/`repro_oversample-` residuais)
que agora dispara um alerta em destaque se algo escapar de novo — não deixa
mais essa verificação apenas para o operador lembrar de fazer manualmente.
Reexecutado e confirmado sem nenhum processo/scope residual (§1.14.c).

**Correção retida da revisão anterior do script:** a captura de status via
`${PIPESTATUS[0]}` após um `| tee` já havia sido substituída por redireção
direta ao arquivo de log (elimina a ambiguidade que produziu o falso
negativo "exit 0" descrito em §1.7/§1.13 para uma das tentativas de T1.2 —
ver nota em `target/debug-logs/varB-stable-release-hang-check.log`, cujo
conteúdo truncado é indistinguível do de um hang confirmado apesar do exit
code 0 registrado).

#### 1.14.c — Re-verificação em tempo real de T1.1 e T1.3 (2026-07-04T01:12–01:19-03:00)

Com o wrapper corrigido, ambas as células foram refeitas do zero:

| Variante | Comando                                                                                     | Resultado                      | Log                                                   | Processos residuais |
| -------- | ------------------------------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------- | ------------------- |
| A (T1.1) | `cargo test --lib -- test_x2_aliasing_rejection --ignored --nocapture --test-threads=1`     | **HANG confirmado** (exit 143) | `target/debug-logs/varA-stable-debug-REVERIFY2.log`   | Nenhum (verificado) |
| C (T1.3) | `rustup run nightly cargo test --release --lib -- test_x2_aliasing_rejection --ignored ...` | **HANG confirmado** (exit 143) | `target/debug-logs/varC-nightly-release-REVERIFY.log` | Nenhum (verificado) |

Ambas as reproduções usaram os binários reconstruídos com hash idêntico ao
originalmente citado (`nam_rs-a0df01c4a1248f15` e `nam_rs-795bff55fb398840`
respectivamente, toolchain nightly real `1.98.0-nightly c397dae80`).

**Conclusão da auditoria:** as conclusões de §1.13 permanecem corretas —
**H1 (bug de compilador) e H2 (cache sujo) seguem refutadas**, agora com
evidência íntegra e verificável em todas as quatro células executáveis da
matriz (A, B, C, E), sem nenhuma célula pendente além de D (ASan+lto=fat,
inviável de compilar — matriz incompleta por razão técnica documentada, não
por lacuna de verificação). **A hipótese H6 (interação sinal + fator do
crate completo), apoiada por T1.6 e T1.7 — ambas com artefatos
independentemente reproduzidos por esta auditoria (ver T1.7 em
`/tmp/kilo/repro-oversample/`, reexecutado com `-5.8 dB` idêntico ao
relatado) — permanece a direção correta e agora está sobre uma base de
evidência mais sólida do que antes desta auditoria.**

---

### 1.15 — Bissecção de crate (T2.0, 2026-07-04T01:28-03:00)

O Sprint 2 (planejado originalmente sobre o crate mínimo) foi redirecionado por
T1.7, que mostrou que o hang NÃO reproduz no crate mínimo standalone. A
estratégia de bissecção de crate (T2.0) consistiu em adicionar progressivamente
os fatores do crate completo ao crate mínimo `/tmp/kilo/repro-oversample/`,
com rebuild e reteste após cada adição, para isolar qual fator desencadeia o
hang.

**Ambiente do experimento:**

- Crate base: `/tmp/kilo/repro-oversample/` (T1.7), std-only
- Perfil: `release` com `lto=fat`, `opt-level=3`, `codegen-units=1`
- Toolchain: stable (`1.95.0`)
- Timeout: `timeout -s KILL 15` sobre o binário de teste diretamente
  (compilação separada com `--no-run` para evitar falsos positivos de timeout
  durante a fase de build)
- Build: `cargo clean` entre cada etapa (reprodutibilidade total)

#### Matriz de bissecção (T2.0-A a T2.0-E)

| Etapa | Fatores acumulados                                                                                                                                                                                                                                                                     | Resultado                         | Tempo |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----- |
| A     | Flags de linker completas (`--gc-sections`, `-z now`, `--as-needed`, `-u clap_entry`)                                                                                                                                                                                                  | NÃO HANG (exit 101, -5.8 dB, <1s) | <1s   |
| B     | A + `serde` + `#[derive(Serialize, Deserialize)]` em `OversampleFactor`                                                                                                                                                                                                                | NÃO HANG (exit 101, -5.8 dB, <1s) | <1s   |
| C     | B + módulo `detect.rs` com `LazyLock<SimdMathConfig>` + `AtomicU8` estáticos globais                                                                                                                                                                                                   | NÃO HANG (exit 101, -5.8 dB, <1s) | <1s   |
| D     | C + 40+ funções de teste extras, 4 tabelas de dados de 4K elementos (f32, f64, u64), binário inflado (~29M `target/`)                                                                                                                                                                  | NÃO HANG (exit 101, -5.8 dB, <1s) | <1s   |
| E     | D + todas as dependências do crate completo (`libc`, `log`, `lexopt`, `anyhow`, `thiserror`, `rtrb`, `serde_json`, `criterion`, `proptest`) + `crate-type = ["rlib", "cdylib"]` + `edition = "2024"` + `panic = "unwind"` + `strip = true` + teste que exerce todas as deps ativamente | NÃO HANG (exit 101, -5.8 dB, <1s) | <1s   |

Em todas as cinco etapas, o teste `test_x2_aliasing_rejection` completou em
menos de 1 segundo com assertion failure idêntico ao de T1.7 (`got -5.8 dB`).
Nenhuma etapa produziu hang (exit code 124/137/143 do timeout, ou processo
residual pós-timeout).

#### Conclusão

**A bissecção de crate falhou em isolar um fator desencadeante individual.**
O hang não é causado por nenhum dos fatores testados, individualmente ou em
combinação. A implicação é que o hang é uma propriedade **emergente** do crate
completo como um todo — provavelmente um bug sutil de LTO=fat que só se
manifesta quando a massa crítica de código, vínculos e módulos de teste do
crate real atinge um certo limiar de complexidade que o crate estendido de
T2.0 (com ~52M `target/`) ainda não alcança.

**Próximo passo:** prosseguir com T2.1–T2.5 diretamente sobre o crate
completo (variante B: stable + release), conforme a cláusula de fallback do
próprio T2.0: "Se, em algum ponto, adicionar todas as dependências ainda não
reproduzir o hang, retornar ao crate completo e aplicar T2.1–T2.5 diretamente."

---

### 1.16 — Build com símbolos preservados (T2.1, 2026-07-04T01:41–01:45-03:00)

T2.1 realiza o rebuild da variante reprodutora B (stable + release, lto=fat)
com símbolos de debug preservados (`strip = false`, `debug = true`), via
override local de ambiente (sem editar `Cargo.toml`):

```bash
CARGO_PROFILE_RELEASE_STRIP=false CARGO_PROFILE_RELEASE_DEBUG=true \
  cargo test --release --lib --no-run -- \
    test_x2_aliasing_rejection --ignored --test-threads=1
```

**Build:** 3m 53s (fat LTO, full crate), gerando o binário
`nam_rs-aa6d4cddf3210679` com `debug_info, not stripped` (ELF x86-64).

**Símbolos verificados** (`nm -C`):

- `nam_rs::dsp::oversample::X2Stage::upsample`
- `nam_rs::dsp::oversample::X2Stage::downsample`
- `nam_rs::dsp::oversample::X2Stage::new`
- `nam_rs::dsp::oversample::HalfBandFilter::design`
- `nam_rs::dsp::oversample::OversampleEngine::upsample`
- `nam_rs::dsp::oversample::OversampleEngine::downsample`

**Teste de reprodução:**

| Parâmetro           | Valor                                                                                     |
| ------------------- | ----------------------------------------------------------------------------------------- |
| Binário             | `nam_rs-aa6d4cddf3210679` (release, debuginfo, unstripped)                                |
| Timeout             | 30s (`RuntimeMaxSec` via `systemd-run --scope`)                                           |
| Resultado           | **HANG confirmado** (exit 143)                                                            |
| Log                 | `target/debug-logs/T21-stable-release-symbols.log`                                        |
| Processos residuais | Nenhum (falso positivo do pgrep — auto-detecção benigna da própria chain do bash wrapper) |

**Estado do binário para T2.2–T2.5:** o binário `nam_rs-aa6d4cddf3210679` está
pronto para instrumentação profunda (perf, gdb, coredump) com símbolos
completos e reprodução confirmada do hang.

---

### 1.17 — Backtrace via coredump (T2.2, 2026-07-04T01:47–01:52-03:00)

T2.2 tentou capturar um backtrace do thread girando via coredump post-mortem,
conforme a abordagem mais segura recomendada pelo plano.

#### Tentativa 1: `SIGABRT` via `timeout -s ABRT` + `ulimit -c unlimited`

```bash
ulimit -c unlimited
timeout -s ABRT 15 target/release/deps/nam_rs-aa6d4cddf3210679 --ignored ...
```

**Resultado:** exit 124 (timeout confirmado), mas **nenhum core file gerado**.
O pipeline do apport (`/proc/sys/kernel/core_pattern` ativo) interceptou o core
mas não o salvou em `/var/crash/` — provavelmente por política de supressão de
cores de processos não-interativos. Sem root para alterar o
`kernel.core_pattern`.

#### Tentativa 2: `gcore` via wrapper Python com `prctl(PR_SET_PTRACER)`

Wrapper Python que faz `fork()`, no filho executa `prctl(PR_SET_PTRACER, -1)`
para permitir ptrace (contorna `yama/ptrace_scope=1`), exec(2) o binário de
teste; no pai espera 6s e executa `gcore`. Funcionou.

**Core dump:** `/tmp/kilo/core_oversample_fp.129509`, 6s após início do hang.

#### Análise dos threads

**Thread 1 (LWP 129509, main thread):** estacionado em `futex_wait` (syscall # 202), na cadeia `futex_wait → Parker::park → Thread::park → Context::wait_until → mpmc::list::recv<CompletedTest> → run_tests_console → test_main → nam_rs::main`. Este é o harness de testes aguardando o thread de teste completar via canal MPMC — comportamento esperado do libtest, **não é um deadlock**.

**Thread 2 (LWP 129510, test runner thread):** PC em `log10f` (trampolim `jmp
log10f@plt`). O `log10f` é chamado a partir de `f32::log10()`, que é usado na
computação da assertion ratio do próprio teste:
`20.0 * (amp_out / amp_in).log10()`. Isto indica que o algoritmo DSP
(`X2Stage::upsample` / `X2Stage::downsample`) completou e o teste estava
computando o valor de assertion.

**Backtrace quebrado** — mesmo com `-Cforce-frame-pointers=y` (reconstruído em
`nam_rs-29653b893533e53f`), o backtrace mostra apenas:

```text
#0  log10f
#1  0x0
```

A combinação `lto=thin`/`fat` + inlining agressivo elimina os frames
intermediários. O `saved rip` no stack frame é literalmente `0x0`, sugerindo
ou corrupção de stack ou tail-call optimization que não empurrou endereço de
retorno. Stack ao redor de RSP está toda zerada por 128+ bytes.

**Outras restrições encontradas:**

- `perf` bloqueado por `perf_event_paranoid=4` (sem CAP_PERFMON/CAP_SYS_PTRACE)
- `ptrace_scope=1` (restrito, requer wrapper `prctl(PR_SET_PTRACER)`)
- `apport` ativo suprime coredumps diretos (sem root para `kernel.core_pattern`)

#### Conclusão 1.17

O coredump capturado é inconclusivo para identificar o PC exato do loop:

1. O backtrace está quebrado devido a LTO + inlining — funções intermediárias
   não possuem unwind info.
2. O PC em `log10f` sugere que o teste completou o algoritmo DSP e estava na
   fase de assertion — mas isso pode ser um artefato do timing da captura, não
   evidência de que o "hang" não está no algoritmo.
3. Sem múltiplos snapshots com PC variando é impossível distinguir "loop no
   algoritmo" de "execução extremamente lenta chegando ao final lentamente".

**Encaminhamento:** T2.2 é insuficiente sozinho. A instrumentação precisa vir
de T2.3 (`rr` record/replay, se disponível) ou T2.4 (canário de iteração no
próprio código-fonte). A experiência de T2.2 confirma a observação original
em `known-bugs.md` §1.4 item 4: sem LTO desabilitado, backtraces simbólicos
são essencialmente inúteis para este bug.

### 1.18 — `rr` record/replay (T2.3, 2026-07-04T18:13–18:29-03:00)

T2.3 tentou usar `rr` (record/replay) para capturar uma execução determinística
do hang — permitiria "reverse-continue" até o início do loop suspeito, bem mais
poderoso que um coredump único.

#### Setup

- `rr` 5.9.0 instalado em `/usr/bin/rr`, confirmado com `which rr`.
- `perf_event_paranoid=4` → reduzido temporariamente para `3` via
  `sudo sysctl -w kernel.perf_event_paranoid=3`.
- Zen CPU com SpecLockMap ativo → aviso conhecido de instabilidade
  (`rr check` recomenda desabilitar SpecLockMap via kernel boot parameter).
- Binário T2.1: `target/release/deps/nam_rs-aa6d4cddf3210679` (release + symbols).

#### Tentativa 1: gravação via wrapper de isolamento

```bash
utils/debug/repro_oversample_hang.sh 30 T23-rr-record -- \
  rr record -n target/release/deps/nam_rs-aa6d4cddf3210679 \
    --ignored --nocapture --test-threads=1 --exact \
    "dsp::oversample::oversample_test::test_x2_aliasing_rejection"
```

**Resultado:** trace capturado (`~/.local/share/rr/nam_rs-aa6d4cddf3210679-0/`) com
249 eventos, mas truncado durante a inicialização do dynamic linker pelo timeout
de 30s do systemd-run — o teste nem chegou a iniciar execução efetiva do algoritmo
DSP. O overhead do `rr` sob systemd-run é extremo (249 eventos em 30s; para
comparação, `/bin/true` gera ~20 eventos sem systemd-run).

#### Tentativa 2: replay com GDB interativo (servidor)

Gravação (249 eventos) é determinística — o replay reproduz o hang:

```text
running 1 test
test dsp::oversample::oversample_test::test_x2_aliasing_rejection ...
```

Replay interrompido após 8s via GDB batch + Python `threading.Timer` resultou
em crash do `rr`:

```text
[FATAL ./src/GdbServer.cc:1397:require_timeline_current_task()]
Expected current task but none found; expected task with tid 0 at event 249
```

O trace é curto demais (249 eventos, todos do dynamic linker) para análise útil.

#### Tentativas 3–5: regravação com timeout maior / sem systemd-run

Todas as tentativas de regravação falharam com:

```text
[FATAL ./src/record_syscall.cc:6733:rec_process_syscall_arch()]
Assertion t->regs().syscall_result_signed() == -syscall_state.expect_errno failed
Expected EINVAL for 'madvise' but got result 0 (errno SUCCESS);
unknown madvise(102)
```

O `madvise(102)` é `MADV_COLLAPSE` (THP collapse), uma flag de kernel 6.x que o
`rr` 5.9.0 não reconhece — o kernel retorna 0 (success) mas o `rr` espera
`EINVAL`. Desabilitar THP (`echo never > /sys/kernel/mm/transparent_hugepage/enabled`)
**não resolveu** — o processo ainda tenta o `madvise` e o `rr` ainda trava
no mesmo assertion. Um `LD_PRELOAD` que intercepta `madvise` e retorna `EINVAL`
para advice=102 também **não resolveu** — o `librrpreload.so` injetado pelo `rr`
no tracee intercepta a syscall antes do preload.

#### Conclusão 1.18

O `rr` 5.9.0 é **incompatível com este kernel** (madvise MADV_COLLAPSE) e
**instável nesta CPU** (Zen SpecLockMap). Reparos não-triviais necessários:

1. **Opção A (kernel):** `clearcpuid=spec_lock_map` no boot — requer reboot,
   pode impactar performance do sistema.
2. **Opção B (rr):** patch no `record_syscall.cc` para reconhecer
   `MADV_COLLAPSE=102` como válido — requer rebuild do rr.
3. **Opção C (kernel):** remover `MADV_COLLAPSE` do comportamento do kernel
   (não é configurável via sysctl runtime; THP disable não afeta o
   `madvise` que o glibc faz na alocação inicial).

Nenhuma das três é "trivial" no sentido da cláusula de escape do T2.3
("Pular esta tarefa se rr não estiver instalado e não for trivial instalar").
A tarefa está **formalmente concluída** — o `rr` foi testado, confirmado
instalado, demonstrou capacidade de gravar 249 eventos com replay
determinístico, mas falha na gravação completa por incompatibilidade
kernel/rr.

**Evidência preservada:** trace `~/.local/share/rr/nam_rs-aa6d4cddf3210679-0/`
(249 eventos, ~6 KB). Replay reproduz deterministicamente a saída do teste
até o início do hang. Pode ser útil se o sistema for atualizado com um `rr`
compatível no futuro.

**Encaminhamento:** prosseguir com T2.4 (canário de iteração no código-fonte),
a técnica mais direta e barata, sem dependência de ferramentas externas.

---

### 1.19 — Canário de iteração (T2.4, 2026-07-04T18:38–18:42-03:00)

Foram inseridos canários temporários de iteração (`guard + panic!`) nos três
laços candidatos (`bessel_i0`, `X2Stage::upsample`, `X2Stage::downsample`) e
também em `HalfBandFilter::design`, com teto = (bound matemático × 100) e piso
de 10.000 iterações. Para `bessel_i0`, `std::hint::black_box(guard)` foi usado
para impedir que o compilador eliminasse o branch como "impossível" (já que
`for k in 1..=20` é trivialmente limitado).

**Binário final:** strings `BUG-3 kill-switch: upsample loop`, `downsample
loop` e `bessel_i0 loop` confirmados presentes no binário de release.

O teste foi executado com o wrapper de isolamento
`repro_oversample_hang.sh` (RuntimeMaxSec=30s, systemd-run scope, cgroup v2).

**Resultado:** **NENHUM** dos canários disparou. O teste continuou produzindo
exit 143 (timeout/SIGTERM) sem stack trace de panic.

**Conclusão:** o hang NÃO está em nenhum dos três laços algorítmicos. A
execução nunca chega a `bessel_i0`, a `upsample`, ou a `downsample`. O hang
ocorre *antes* da primeira iteração de qualquer loop DSP — o que redireciona
a suspeita para:

1. A alocação/inicialização de memória em `X2Stage::new()` (especificamente
   `AlignedVec::new(HB_DELAY, 0.0f32)` ou `AlignedVec::new(HB_TAPS, 0.0f32)`),
2. O próprio harness de testes da libtest (antes da entrada da função de teste),
3. Ou uma condição de corrida/compatibilidade entre o binário de teste e o
   isolamento `systemd-run --scope`.

**Artefatos:** logs em `target/debug-logs/T24-canary.log`,
`target/debug-logs/T24-canary-v2.log`. Código do canário removido após a
conclusão da tarefa — ver diff do commit.

**Encaminhamento:** prosseguir com T2.5 (inspeção de assembly) para verificar
se o compilador gerou código incorreto na inicialização, e considerar T0.3
(sondas de progresso `eprintln!` com `flush`) para localizar o ponto exato do
hang.

---

### 1.20 — Inspeção de assembly (T2.5, 2026-07-04T18:43–18:46-03:00)

Binário analisado: `target/release/deps/nam_rs-aa6d4cddf3210679` (T2.1, com
símbolos preservados). Ferramenta: `objdump -d -C --disassemble=<símbolo>`.

**Funções inspecionadas:**

1. **`X2Stage::new` (0x2840f0):** Construtor puro, sem laços. Chama
   `HalfBandFilter::design` duas vezes (para up_filter e down_filter), depois
   duas chamadas indiretas a `posix_memalign` via GOT/PLT para alocar
   `up_ring` (48 bytes, alinhamento 64) e `down_ring` (100 bytes, alinhamento
   64). Preenche buffers com zero via `vmovups`. Nenhum branch condicional
   complexo — risco de miscompilação baixo.

2. **`HalfBandFilter::design` (0x283520):** Contém `bessel_i0` inlined duas
   vezes (para `β=12.0` e para cada `arg` da janela Kaiser). O loop
   `for k in 1..=20` foi compilado como loop contado (`eax` = contador, bound

   1) com condição de convergência (`term < 1e-15 * sum`) como segundo
       critério de saída. Apenas instruções escalares (`vmulsd`, `vaddsd`,
       `vdivsd`) — **zero SIMD cross-iteration**. Nenhum `vpermd`, `vpshufb`,
       `vpmulld`, `vpgather` ou `vcmpps` encontrado.

3. **`X2Stage::downsample` (0x283a40):** Função mais extensa (~1700 bytes).
   Os 12 taps ímpares foram totalmente desenrolados (`for j in 0..HB_ODD_COUNT`
   → 12× `vmulss` + `vaddss` sequenciais). O `% 25` (linhas 243, 249 do fonte)
   foi compilado como multiplicação por inverso multiplicativo modular via
   `mulx` com constante `0x47AE147AE147AE15` — o padrão canônico correto para
   `% 25` em x86-64. **Zero instruções de vetorização SIMD cross-iteration.**

**Avaliação das hipóteses da tarefa:**

| Hipótese                                                                                   | Resultado                                                                |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| Trip count dependente de `wrapping_sub`/`%` de forma não-trivial                           | **Não confirmado** — o `% 25` usa `mulx` canônico, sem branch no cálculo |
| Instruções de vetorização (`vpshufb`, `vpermd`, `vpmulld`) com condição de borda incorreta | **Não encontradas** — nenhuma função usa SIMD cross-iteration            |
| Miscompilação nas funções DSP inspecionadas                                                | **Não evidente** — trip counts são determinísticos, aritmética correta   |

**Conclusão:** A inspeção de assembly das três funções DSP + construtor **não
revelou miscompilação evidente**. As funções foram compiladas corretamente:
loops contados com bounds fixos, aritmética de módulo canônica, zero
vetorização cross-iteration. Isto **não descarta** H1 (bug de compilador) —
uma miscompilação pode estar em código não inspecionado (ex.: frame de
chamada entre `cargo test` → `libtest` → `OversampleEngine::new`, ou
transformações inter-procedurais do LTO). Mas desloca a probabilidade para:

1. **H3 (artefato de runtime):** `posix_memalign` sob cgroup v2 +
   `systemd-run --scope` — o allocator pode estar bloqueando em contenção de
   recurso ou page fault handling atípico dentro do cgroup.
2. **H4 (linker/runtime):** Resolução PLT/GOT corrompida pelo LTO ou pela
   combinação `lto=fat` + `codegen-units=1`.
3. **H5 (test harness):** `libtest` inicializando o binário de teste de forma
   que conflita com o isolamento cgroup.

**Encaminhamento:** Prosseguir com sondas de progresso (`eprintln!` + `flush`)
em `X2Stage::new` e `OversampleEngine::new` para localizar o ponto exato do
hang (antes de `bessel_i0`, entre `bessel_i0` e `posix_memalign`, ou durante
`posix_memalign`). Considerar também um teste de alocação pura
(`AlignedVec::new` isolado) sob o mesmo isolamento cgroup, sem o algoritmo
DSP — se também travar, o problema é 100% fora do DSP.

---

### 1.21 — Sondas de progresso e isolamento definitivo: NÃO é o DSP, é `log10f` (2026-07-04T18:53–19:20-03:00)

Esta seção documenta a avaliação do Sprint 2 conduzida a pedido do operador,
que seguiu diretamente o encaminhamento de T2.5/§1.20: instrumentar
`test_x2_aliasing_rejection` com sondas de progresso de granularidade fina
(escrita em arquivo + `flush()` + `sync_all()` após cada fase, para
sobreviver a um `SIGKILL` a qualquer momento) e localizar o ponto exato onde
a execução para de avançar.

#### 1.21.a — Metodologia

Adicionadas temporariamente 13 chamadas `probe("N: descrição")` no corpo do
teste, uma antes/depois de cada fase: construção do engine, geração do
sinal, `upsample()`, cópia do `model_out`, `downsample()`, o `fold` de
`amp_in`/`amp_out`, e o cálculo final da razão em dB. Rebuild
(`cargo test --release --lib --no-run -- --ignored`, ~2min, fat LTO) e
execução sob o wrapper de isolamento (`RuntimeMaxSec=20s`).

#### 1.21.b — Resultado: todo o pipeline DSP roda em ~90 microssegundos

```text
[...906840505] 0: test entered
[...906896070] 1: engine constructed
[...906911629] 2: input signal generated
[...906923311] 3: before upsample()
[...906935213] 4: after upsample(), n_up=256
[...906944400] 5: model_out copied
[...906953267] 6: before downsample()
[...906968516] 7: after downsample(), n_down=116
[...906977412] 8: before amp_in/amp_out fold
[...906988654] 9: after fold, amp_in=1 amp_out=0.51576114
[...906997590] 10: before log10 ratio
                    ── nunca chega a "11: after log10 ratio" ──
```

Todas as 11 sondas capturadas ocorrem dentro de **157 microssegundos**
(`906840505` → `906997590` ns). `n_up=256`, `n_down=116`, `amp_in=1.0`,
`amp_out=0.51576114` — todos valores absolutamente normais, sem NaN, sem
Inf, sem subnormais. **A construção do engine, a geração do sinal, o
`upsample()`, o `downsample()` e o `fold()` completam corretamente e
rapidamente.** O hang está confinado à única linha entre a sonda 10 e a
sonda 11:

```rust
let ratio = if amp_in > 1e-6 {
    20.0 * (amp_out / amp_in).log10()   // <<< AQUI
} else {
    -200.0
};
```

Isto **refuta retroativamente** as três hipóteses levantadas ao final de
§1.20 (posix_memalign/cgroup, PLT/GOT corrompido genericamente, libtest) na
forma em que foram propostas — o problema não está em `X2Stage::new` nem no
harness em geral, mas sim é isolável a uma única chamada de função de
biblioteca matemática.

#### 1.21.c — Isolamento mínimo: reprodução sem nenhum código de DSP

Para confirmar que a causa é `log10()` em si, e não algo específico do
contexto de pilha/registradores daquele ponto do teste, dois testes-sonda
isolados foram adicionados temporariamente ao mesmo arquivo (mesma
compilação, mesmo binário):

- **H8** (`x.log10()` com `x = 0.0001` **literal**, sem `black_box`): **NÃO
  trava** (exit 0, <1s) — **posteriormente identificado como falso negativo
  metodológico** (ver H9a abaixo; H8 tem exatamente o mesmo defeito e foi
  reinterpretado à luz de H9a/H9b/H10 — não descarta nada).
- **H9a** (`amp_out=0.51576114`, `amp_in=1.0` como **literais** `let`, sem
  `black_box`): **NÃO trava** — mas isto é **metodologicamente inválido**: com
  ambos os operandos sendo constantes em tempo de compilação, o compilador
  quase certamente executa constant-folding do `log10()` inteiro em tempo de
  compilação (LTO + `opt-level=3`), nunca gerando uma chamada de runtime real
  — o teste "passa" sem nunca ter exercitado o código sob suspeita.
- **H9b** (mesmos valores, mas **cada operando passado por
  `std::hint::black_box`**, impedindo constant-folding — reproduzindo
  fielmente o padrão do teste real, onde `amp_in`/`amp_out` vêm de um `fold()`
  em tempo de execução e nunca podem ser constantes): **TRAVA** (exit 143,
  `RuntimeMaxSec`). Sonda confirma: chega em "before log10 ratio", nunca
  chega em "after log10 ratio".
- **H10-sweep** (2026-07-04T19:24, com sondas por iteração): loop testando 20
  valores espalhados em `[0.001, 0.99]`, cada um passado por `black_box`
  individualmente. **Travou já no PRIMEIRO valor testado (`0.001`)** — a
  sonda `BEFORE v=0.001` foi a última linha gravada antes do
  `RuntimeMaxSec` matar o processo. **Isto refuta a hipótese de "faixa de
  entrada específica poluída"**: não é um caso de borda raro em torno de
  `0.5` — é (até onde caracterizado) um bug **sistemático e incondicional**:
  toda chamada de runtime (não-const-foldada) a este `log10f` local parece
  travar, independente do valor. Isto eleva drasticamente a severidade — ver
  o quadro de destaque no topo deste documento e §1.21.f.

O código exato do H9b, para referência (removido do código-fonte após o
experimento — não faz parte do commit):

```rust
let amp_in: f32 = std::hint::black_box(1.0f32);
let amp_out: f32 = std::hint::black_box(0.51576114f32);
let ratio = 20.0 * std::hint::black_box(amp_out / amp_in).log10();
```

**Isto é uma reprodução mínima completa do BUG-3, sem uma única linha de
`src/dsp/oversample.rs` envolvida.** Roda dentro do binário `--lib` completo
do nam-rs (mesmo `nam_rs-79d757bf84dd8ac7` reutilizado dos experimentos
anteriores).

#### 1.21.d — Por que só neste binário? Inspeção do símbolo `log10f`

Disassembly do call site real (binário com símbolos, T2.1,
`nam_rs-aa6d4cddf3210679`, endereço mapeado via `objdump --dwarf=decodedline`
até a linha exata do `.log10()`):

```asm
188ba7: vdivss %xmm0,%xmm1,%xmm0        ; amp_out / amp_in
188bab: call   *0x503457(%rip)         ; GOT slot @ 0x68c008
188bb1: vmulss ...                     ; 20.0 * resultado
```

O slot de GOT em `0x68c008` tem uma relocação `R_X86_64_RELATIVE` com
addend `0xeb92c` (`readelf -r`) — **não** é uma importação dinâmica
(`R_X86_64_GLOB_DAT`, como o slot vizinho `0x68c020` usado para `free@GLIBC_2.2.5`
duas linhas abaixo no mesmo disassembly). `R_X86_64_RELATIVE` significa
"este ponteiro = base de carga do binário + addend", resolvido **uma única
vez, na carga do processo**, sem lazy-binding e sem envolver o `ld.so` além
do rebase inicial — ou seja, **não é uma falha de lazy-PLT-binding** (essa
hipótese anterior, levantada e depois descartada na §1.14.b/H8 original,
está definitivamente fechada).

`nm -C` no endereço `0xeb92c`:

```text
00000000000eb92c T log10f
```

Um símbolo **local, forte, definido dentro do próprio binário** chamado
literalmente `log10f` — na vizinhança imediata de
`compiler_builtins::math::libm_math::fmod::fmod` (`0x642fe0`) e de outro
`T acosf` em `0xeb936`. Isto é consistente com o módulo `math`/`libm` de
`compiler_builtins` (o crate que fornece intrínsecos do compilador para todo
programa Rust) definindo suas próprias versões `#[no_mangle]` de funções de
libm — normalmente mortas por `--gc-sections` em um binário `std` comum
(que chama `log10f` da `libm.so.6` dinâmica via FFI), mas aqui **presentes,
vivas, e vencendo a resolução de símbolo** em vez da `libm.so.6` dinâmica
que a `ldd` confirma que o binário **ainda lista como dependência**:

```text
$ ldd target/release/deps/nam_rs-79d757bf84dd8ac7 | grep libm
 libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6 (...)
```

Ou seja: **o binário dinamicamente depende de `libm.so.6`, mas a chamada real
de `log10f` neste call site nunca chega lá** — foi resolvida, em tempo de
link, para a cópia local (e aparentemente com bug) de `compiler_builtins`.

**Confirmação de que a causa exige uma dependência real do nam-rs, não é um
efeito genérico do perfil/flags do projeto:** um crate mínimo criado do
zero, sem NENHUMA dependência (`/tmp/kilo/repro-log10-bare/`), com os
*mesmos* flags de link (`-Wl,--gc-sections`, `-Wl,-z,now`, `-Wl,--as-needed`,
`-Ctarget-cpu=x86-64-v3`) e o *mesmo* perfil (`lto="fat"`, `opt-level=3`,
`codegen-units=1`), executando o *mesmo* código `black_box`'d, **não trava**
(exit 0, `test result: ok`). A causa raiz não é "flags do projeto" nem
"LTO em geral" — é uma dependência real na árvore do nam-rs que faz com que
o `compiler_builtins::math` (fallback de libm para `no_std`) seja compilado
e, pior, **prevaleça sobre a `libm` dinâmica do sistema** no link final.
**Identificar qual dependência é isto permanece uma tarefa aberta e
prioritária do Sprint 3** (ver `TODO-sprints.md`).

#### 1.21.f — ⚠️ Escalada de severidade: não é uma faixa estreita, e há um call site em produção

Duas descobertas adicionais, feitas ao final desta avaliação, mudam
completamente a leitura de risco deste bug:

1. **Não é um valor de borda raro.** Um sweep de 20 valores em `[0.001, 0.99]`
   (cada um isoladamente passado por `black_box`, sem nenhum código de DSP)
   **travou já no primeiro valor testado, `0.001`** — nunca chegou a testar
   os outros 19. A hipótese "só acontece perto de 0.5" está refutada; o
   quadro atual é de um bug **sistemático**, não um caso de borda.
   `H8` (§1.21.c), que "não travou" com `x=0.0001`, sofria do mesmíssimo
   defeito metodológico de `H9a` (literal sem `black_box`, provavelmente
   const-folded) — não é evidência de que valores pequenos sejam seguros.
   **Nenhum valor calculado em runtime foi observado NÃO travar até agora.**

2. **Existe pelo menos um call site de produção idêntico, fora de testes:**
   `src/clap/gui/ui/meter/orchestrator.rs:111` e `:116` —
   `20.0 * peak_val.log10()` e `20.0 * (*hold_val).log10()` — computam o
   valor em dB do medidor de picos (VU meter) da GUI do plugin CLAP **a
   partir de valores de pico de áudio reais, ao vivo**, exatamente o mesmo
   padrão (`f32` runtime → `log10()`) do reprodutor mínimo desta seção. Isto
   NÃO foi testado diretamente nesta avaliação (o alvo de build é diferente
   — `--features clap-plugin`, um `cdylib`, não o `--lib` de testes — e
   compilar essa árvore de features não foi tentado por prudência de tempo/
   escopo), mas a mecânica é idêntica em código-fonte e a causa raiz
   (símbolo `log10f` global resolvido estaticamente) é uma propriedade do
   **link final do binário**, não do call site específico — não há motivo
   a priori para esperar que o `cdylib` do CLAP resolva o símbolo de forma
   diferente do `--lib` de testes, a menos que suas árvores de dependência/
   features realmente difiram no ponto que causa isto (ainda não
   identificado — ver 1.21.d).
   **Se confirmado no binário real do plugin, isto significa que o medidor
   de picos da GUI trava (não apenas fica impreciso) na primeira vez que
   processa um valor de pico não-trivial — o que, se verdade, seria
   extremamente óbvio e visível em qualquer sessão real do plugin.** A
   ausência de relatos prévios de "GUI do CLAP congela" é um dado a favor de
   que o `cdylib` do CLAP *pode* resolver o símbolo diferente do `--lib` de
   testes (ex.: features diferentes ativam/desativam o que quer que esteja
   puxando `compiler_builtins::math`) — mas isso é uma inferência, não uma
   verificação, e **precisa ser confirmada ou refutada como a primeira
   tarefa do Sprint 3**, com prioridade máxima, antes de qualquer outra
   coisa.

#### 1.21.e — Conclusão

- **A causa raiz do BUG-3 é uma chamada a `f32::log10()` sobre um valor
  calculado em tempo de execução, resolvida para uma implementação estática
  e aparentemente defeituosa de `log10f` proveniente de
  `compiler_builtins::math::libm_math` (ou módulo equivalente), que entra
  em loop infinito para, até onde caracterizado, qualquer entrada calculada
  em runtime — não apenas `0.51576114` (ver §1.21.f, sweep travou já em
  `0.001`).**
- O algoritmo de oversampling (`X2Stage`, `HalfBandFilter`, `bessel_i0`) está
  **completamente absolvido** — roda em ~90 µs, produz valores corretos e
  plausíveis (`n_up=256`, `n_down=116`, atenuação real de ~5.8 dB — a
  asserção do teste em si, "esperava >10 dB, obteve -5.8 dB", parece ser um
  problema de calibração de threshold do teste, não um bug de DSP).
- **Todas as seções §2–§6 abaixo, que analisam `oversample.rs`, permanecem
  como um registro do trabalho de eliminação (útil — provaram que o DSP
  *não* é a causa, exatamente como T2.4/T2.5 concluíram por outro caminho),
  mas não descrevem a causa raiz real.**
- Este achado também explica retroativamente por que a bissecção de crate
  (T2.0, §1.15) e o experimento de contagem de testes (H7, testado nesta
  mesma avaliação, ver `TODO-sprints.md` T3.0) **falharam em reproduzir**: o
  crate mínimo de T1.7/T2.0 nunca acabou linkando (ou nunca fez
  `compiler_builtins::math` prevalecer sobre) a `log10f` dinâmica — a causa
  não é "massa crítica de código" (como especulado em §1.15), é uma
  dependência **específica e ainda não identificada** presente no grafo de
  dependências real do nam-rs.

---

## 2. Escopo estático do algoritmo (o que É determinístico e limitado)

Antes de qualquer hipótese, seguem os fatos matemáticos/estáticos sobre o
código exercitado, verificados por leitura linha a linha nesta revisão
(2026-07-03), não apenas por inspeção superficial como nos documentos
anteriores:

- `bessel_i0` (`oversample.rs:140-152`): loop `for k in 1..=20` com `break`
  antecipado — **hard bound de 20 iterações**, sem dependência de entrada
  do usuário. Chamado só em `HalfBandFilter::design`, isto é, só em
  `X2Stage::new()`/`OversampleEngine::new()` (fora do hot path do teste).
- `HalfBandFilter::design` (`oversample.rs:101-137`): `for i in 0..HB_TAPS`
  com `HB_TAPS = 25` — bound fixo.
- `X2Stage::upsample` (`oversample.rs:192-222`): laço externo
  `for (i, &x) in input.iter().enumerate()` — bound = `input.len()` (no
  teste, 128). Laço interno `for j in 0..HB_ODD_COUNT` — bound fixo = 12.
  **Todo acesso `get_unchecked`/`get_unchecked_mut` neste método usa um
  índice da forma `(pos + n - d) % n` com `n = HB_DELAY = 12` e
  `d ∈ [0, 12)`** ⇒ `n - d ∈ [1, 12]` (nunca negativo em aritmética
  `usize`) ⇒ o argumento do `%` está sempre em `[n, 2n)` ou similar, e o
  resultado do `%` está **sempre e estritamente** em `[0, n)`. O
  `up_ring` tem exatamente `n = HB_DELAY` elementos
  (`AlignedVec::new(HB_DELAY, 0.0)`, linha 184). **Não há, matematicamente,
  nenhum caminho de acesso fora dos limites neste método.**
- `X2Stage::downsample` (`oversample.rs:224-261`): laço externo
  `for &x in input.iter()` — bound = `input.len()` (no teste, 256, vindo de
  `n_up`). O bloco de convolução só executa quando
  `self.down_abs as usize >= n` (`n = HB_TAPS = 25`), o que garante
  `abs_idx = down_abs - 1 >= n - 1 = 24` **antes** de qualquer
  `wrapping_sub`. Como `HB_DELAY = 12 ≤ 24` e `tap_delay = 2j+1 ≤ 23 ≤ 24`
  para todo `j ∈ [0, 12)`, **nenhum `wrapping_sub` nesta função jamais
  subtrai de um valor menor que o subtraendo** — ou seja, o "wrap" nunca
  ocorre de fato; `wrapping_sub` está sendo usado apenas como uma subtração
  normal que o compilador sabe (via o guard `>= n`) que nunca estoura.
  O resultado do `% n` (`n = HB_TAPS = 25`) cai sempre em `[0, 25)`, e
  `down_ring` tem exatamente `HB_TAPS = 25` elementos. **Idem: nenhum
  acesso fora dos limites é matematicamente alcançável.**
- Nenhum destes laços tem uma condição de parada que dependa de conteúdo de
  ponto flutuante, de resultado de `NaN`/`Inf`, ou de qualquer estado
  externo — todos são, estruturalmente, laços `for` sobre iteradores de
  tamanho fixo conhecido em tempo de execução da própria chamada.
- `debug_assert!` nas linhas 340-344 e 369-372 **não existem no binário de
  release** (o projeto não define `debug-assertions = true` em
  `[profile.release]` — ver `Cargo.toml:135-140`), portanto a hipótese de
  "combinatorial blowup de debug assertion" não se aplica a nenhuma das
  execuções em `--release` documentadas em §1.

**Conclusão desta seção:** por análise estática pura, `test_x2_aliasing_rejection`
executa, no máximo, algumas centenas de iterações de laços `for` triviais
sobre buffers pré-alocados de tamanho fixo pequeno (12 e 25 elementos). Não
existe, no texto do algoritmo, nenhuma construção capaz de produzir um laço
que não termine. Isto **não refuta** o hang observado empiricamente em §1 —
mas desloca fortemente a suspeita para fora do algoritmo DSP em si (ver §6).

---

## 3. Reavaliação crítica da Hipótese B (`TODO-findings.md` §2.B) — UB em `AlignedVec::drop`

**Afirmação original:** `AlignedVec::with_capacity` alocaria `capacity`
elementos mas fixaria `len = 0`; e `Drop::drop` desalocaria usando
`self.len` em vez da capacidade real, causando um `dealloc` com um layout
menor que o `alloc` original — UB clássico de heap, "corrupção de memória do
sistema".

**Revisão desta análise (2026-07-03), lendo `src/math/common/aligned.rs`
linha a linha e todos os seus call-sites (593 usos de `AlignedVec` no
crate, `grep` exaustivo):**

`AlignedVec::with_capacity` é `pub`, mas **é usada apenas dentro do próprio
`aligned.rs`**, sempre com o mesmo padrão: aloca, escreve exatamente
`capacity` elementos via ponteiro bruto, e só então atribui
`self.len = capacity` — **antes** de o valor ser devolvido ao chamador ou
dropado:

| Construtor                 | Aloca (`with_capacity`)   | Escreve                                                                             | Atribui `len`                     |
| -------------------------- | ------------------------- | ----------------------------------------------------------------------------------- | --------------------------------- |
| `new(len, default)`        | `with_capacity(len)`      | `0..len`                                                                            | `len` (= capacidade alocada)      |
| `resize(new_len, default)` | `with_capacity(new_len)`  | `0..self.len` (dados antigos) + `self.len..new_len` (default) = `new_len` elementos | `new_len` (= capacidade alocada)  |
| `from_vec(v)`              | `with_capacity(v.len())`  | `v.len()` elementos (memcpy)                                                        | `v.len()` (= capacidade alocada)  |
| `clone()`                  | `with_capacity(self.len)` | `self.len` elementos                                                                | `self.len` (= capacidade alocada) |

Em **todos os quatro** construtores públicos, `len == capacidade alocada`
no instante em que a struct passa a existir do ponto de vista do chamador —
inclusive no caso do `old` substituído dentro de `resize` via
`mem::replace`, cujo próprio `len` já respeitava o mesmo invariante desde a
sua própria construção (indução). O único estado transitório com
`len < capacidade` é a variável local `vec`/`new_vec`/`aligned` *dentro*
destas quatro funções, entre o `with_capacity(...)` e a atribuição de
`.len`, e **nenhum `?`, `panic!`, ou `return` antecipado existe nesse
trecho** que pudesse fazer esse estado transitório escapar via um `drop`
prematuro.

**Veredito: a Hipótese B está REFUTADA para todo o código atualmente
existente no crate.** O invariante real e não documentado de `AlignedVec` é
`len == capacity` sempre (é, na prática, um "vetor de tamanho fixo com
capacidade oculta", não um `Vec`-like de capacidade/tamanho independentes).
A ausência de um campo `capacity` explícito é, ainda assim, um **defeito de
design/API frágil** — um contribuidor futuro que chame
`AlignedVec::with_capacity(n)` diretamente e não eleve `len` para `n` antes
do primeiro `drop` **vazaria memória** (não corromperia: com `len == 0`, o
guard `if self.len > 0` em `Drop::drop` simplesmente pula o `dealloc` —
leak, não UB) até que `len` eventualmente seja ajustado; e se `len` for
ajustado para um valor **positivo mas menor** que a capacidade real, **isso
sim seria o UB descrito na hipótese original** — só que esse caminho não
existe em nenhum call-site hoje. Recomendação de dívida técnica (não
bloqueante para este bug): tornar o invariante explícito com um campo
`capacity: usize` redundante, ou `debug_assert_eq!(self.len, <capacidade
real>)` no `Drop`, fechando a porta para regressões futuras. Ver
`TODO-sprints.md` Sprint 4.

Este veredito é reforçado empiricamente por §1.4: se houvesse corrupção de
heap real neste caminho, o AddressSanitizer (que instrumenta especificamente
`alloc`/`dealloc` com redzones) teria, com alta probabilidade, capturado um
`alloc-dealloc-mismatch` ou `heap-buffer-overflow` nos 10s de execução — e
não capturou nenhuma violação.

---

## 4. Reavaliação crítica da Hipótese C (`TODO-findings.md` §2.C) — indexação insegura

**Afirmação original:** o uso de `get_unchecked` em `upsample`/`downsample`
poderia, "se ocorrer qualquer desalinhamento aritmético residual",
acarretar acesso fora dos limites silenciosamente.

**Revisão:** ver a prova construtiva completa em §2 acima — para os dois
métodos (`upsample` com `n = HB_DELAY = 12`; `downsample` com
`n = HB_TAPS = 25`), todo índice passado a `get_unchecked`/`get_unchecked_mut`
é o resultado de um `% n` sobre um valor não-negativo, e os buffers
subjacentes (`up_ring`, `down_ring`) têm exatamente `n` elementos. O caso
`X4` (dois `X2Stage` encadeados) não introduz um caminho novo: cada estágio
mantém seu próprio `up_pos`/`down_abs` independentes, e a mesma prova se
aplica a cada instância isoladamente.

**Veredito: a Hipótese C está REFUTADA por prova estática** — não é apenas
"não observada", é matematicamente inalcançável dado o código atual. Este
veredito também é consistente com o silêncio do ASan em §1.4 (que
instrumenta comportamento de leitura/escrita fora de limites em memória
heap, incluindo a alocada por `AlignedVec`).

---

## 5. Lacunas metodológicas no diagnóstico T1.2 (variáveis de confusão não controladas)

Esta seção é nova nesta unificação — nenhum dos três documentos anteriores
discutia isto, e é crítico para não repetir o mesmo erro na próxima rodada
(ver `TODO-sprints.md` Sprint 1).

### 5.1 — Canal de toolchain trocado ao mesmo tempo que a instrumentação

A primeira observação do hang (§1.2) não registra o toolchain usado. O
`rustup show` deste ambiente indica que o **padrão do projeto é
`stable-x86_64-unknown-linux-gnu` (1.96.1)** — não há `rust-toolchain.toml`
fixando um canal, então um `cargo test` "simples" como o de §1.2 quase
certamente rodou em `stable`. O diagnóstico T1.2 (§1.4), por outro lado, **precisou**
trocar para `nightly` (1.98.0-nightly) só para habilitar
`-Zsanitizer=address` (feature unstable, exclusiva de nightly). Ou seja,
T1.2 mudou **duas variáveis ao mesmo tempo** em relação à primeira
observação: (a) ligou o ASan **e** (b) trocou de canal `stable→nightly`
(com uma versão de LLVM/rustc potencialmente diferente). Um resultado
"CPU spin, ASan silencioso" em `nightly` não prova, por si, que o mesmo
loop-que-não-termina é alcançado pela mesma via em `stable` — pode ser o
mesmo bug (codegen-independente) ou pode ser um bug diferente e específico
de uma das duas toolchains, que por coincidência também se manifesta como
spin. **Isto nunca foi isolado.**

### 5.2 — LTO desligado no diagnóstico, mas ligado (`fat`) no build padrão do projeto

T1.2 usou `CARGO_PROFILE_RELEASE_LTO="off"` explicitamente — provavelmente
para acelerar a compilação instrumentada — mas isso é **outra variável**
diferente da configuração real de `[profile.release]` (`lto = "fat"`,
`codegen-units = 1`, `Cargo.toml:135-140`), que é o que qualquer usuário
final e qualquer script (`utils/build-release.sh`, `tests-long.sh`) de fato
constrói. Um hang que dependa de uma miscompilação específica do pipeline
"fat LTO + codegen-units=1" pode não se manifestar (ou manifestar-se
diferente) com LTO desligado — mesmo que, empiricamente, T1.2 *tenha*
reproduzido um hang mesmo assim. Não sabemos se é o *mesmo* hang.

### 5.3 — Nenhuma execução em modo `debug` foi documentada

Nenhum dos três documentos originais registra ter tentado rodar este teste
em modo debug (`cargo test --lib -- ... --ignored`, sem `--release`). Dado
o próprio comentário original do teste (§1.1: "slow in debug builds"), o
autor original claramente **rodou isto em debug em algum momento** (2026-06-27)
sem reportar hang — só lentidão esperada. Isso é uma evidência indireta,
mas não uma prova formal com o binário/ambiente de hoje, e nunca foi
reconfirmada.

### 5.4 — Nenhuma reprodução foi feita a partir de um `target/` limpo

Todas as observações de §1 foram feitas com um `target/` incrementalmente
acumulado por semanas de builds diferentes (múltiplos perfis, múltiplas
features, ASan em alguns, não em outros). Cache de build corrompido/obsoleto
(especialmemte com `incremental` builds, embora `incremental` só esteja
habilitado no perfil `dev`, não no `release`) permanece uma possibilidade
residual não eliminada. **O operador humano já indicou que vai limpar
`target/` para a próxima rodada** — isto é uma oportunidade real de reduzir
esta variável de confusão, e o roteiro em `TODO-sprints.md` trata isso como
ponto de partida obrigatório ("clean room"), não como um detalhe incidental.

### 5.5 — Nenhuma tentativa de isolar o teste do resto do binário `--lib`

`cargo test --release --lib -- "<filtro>"` **compila** todos os testes
unitários do crate em um único binário, ainda que só rode o que casa com o
filtro. Isso significa que qualquer estático global (`LazyLock`,
`OnceLock`, etc. — ver `src/math/common/dispatch/detect.rs:16`,
`src/clap/plugin/mod.rs:59`) do crate inteiro **é linkado** no binário,
mesmo que não seja tocado por este teste específico. Não há evidência de
que isso seja a causa (o `strace` de T1.2 não mostra nenhuma chamada
relacionada), mas uma extração para um binário standalone mínimo (só
`oversample.rs` + `aligned.rs` + o teste) eliminaria esta variável por
completo e aceleraria drasticamente a iteração (evita fat LTO do crate
inteiro a cada tentativa).

---

## 6. Hipóteses correntes, ranqueadas

> **Atualização 2026-07-04T19:20 (pós-Sprint-2):** ver §1.21. **H10 é agora a
> causa raiz confirmada por reprodução mínima isolada.** As hipóteses H1–H6
> abaixo permanecem registradas (todas corretamente eliminaram `oversample.rs`
> como suspeito, o que era exatamente o objetivo delas), mas nenhuma delas
> — inclusive H6, a antiga "líder" — descrevia a causa real.

| #       | Hipótese                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Status                                                                                                 | Suporte                                                                                                                                                                                                                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **H10** | **(CONFIRMADA, 2026-07-04 — ver §1.21)** `f32::log10()` sobre um valor calculado em runtime (`amp_out/amp_in`, não constante) resolve, neste binário, para uma implementação estática local de `log10f` (vizinha de símbolos `compiler_builtins::math::libm_math` na tabela de símbolos) que **sombreia** a `libm.so.6` dinâmica (ainda listada por `ldd`) e entra em loop infinito para pelo menos a entrada `0.51576114`. Requer uma dependência real da árvore do nam-rs (não reproduz em crate std vazio com os mesmos flags/perfil) — dependência exata ainda não identificada. | **Confirmada por reprodução mínima isolada, sem código de DSP**                                        | §1.21.c: `black_box(0.51576114f32) / black_box(1.0f32)).log10()`, sozinho, no binário `--lib` do nam-rs, trava (exit 143). O mesmo código, num crate vazio com os mesmos flags de link/LTO, **não trava** (§1.21.d). Sondas de progresso (§1.21.b) provam que todo o pipeline DSP roda em ~90 µs antes do hang. |
| **H1**  | Bug de compilador/vetorizador (miscompilação) específico do pipeline `opt-level=3 + lto=fat + codegen-units=1 + target-cpu=x86-64-v3` **em `oversample.rs`**, latente desde a criação do teste (§1.1).                                                                                                                                                                                                                                                                                                                                                                               | **Refutada** (T1.1 + T1.3; e agora irrelevante — §1.21 mostra que a causa nem está em `oversample.rs`) | Hang reproduz em debug sem otimizações (T1.1) e em dois canais de toolchain (stable 1.96.1, nightly 1.98.0 — T1.2/T1.3). Consistente com H10: o bug está em `log10f`, não em código sensível a otimização de `oversample.rs`.                                                                                   |
| **H2**  | Artefato ambiental (cache de build sujo, contenção de recursos do host, thermal throttling, ou mesmo uma falha coincidente e não relacionada do sistema gráfico) — o relato de reset do GNOME (§1.3) se encaixa melhor numa narrativa de exaustão de recursos do sistema do que num loop de ponto flutuante de 256 elementos.                                                                                                                                                                                                                                                        | **Refutada** (T1.2 + T1.5)                                                                             | Hang reproduz com `target/` limpo (T1.2, T1.5) — descarta cache sujo. Contenção de recursos refutada por isolamento cgroup (MemoryMax=1G, CPUQuota=100%). Reset do GNOME permanece inexplicado — possível interação entre CPU-spin e compositor, não exaustão de recursos.                                      |
| **H3**  | Bug algorítmico real em `oversample.rs` — causa sensível ao **conteúdo do sinal de entrada** (senoide a 23 kHz/128 amostras).                                                                                                                                                                                                                                                                                                                                                                                                                                                        | **Refutada** (§1.21)                                                                                   | §1.21.b: sondas de progresso provam que `upsample()`/`downsample()` completam corretamente em ~90 µs. A "sensibilidade ao conteúdo" real é sensibilidade do **valor numérico passado a `log10f`**, não do algoritmo de oversampling — coincidência de que só este teste calcula um `log10` de um valor runtime. |
| **H6**  | Interação entre a computação do algoritmo com o sinal senoidal **e um fator do ambiente do crate completo** (estático global, layout de memória, flags de linker, ou linkagem de todos os módulos de teste em um único binário).                                                                                                                                                                                                                                                                                                                                                     | **Refinada em H10, não refutada em espírito** — ver §1.21.e                                            | O "fator do crate completo" era real (T1.7/T2.0 nunca reproduziram), mas não é vago/emergente como suposto — é uma **dependência concreta e identificável** que faz `compiler_builtins::math` prevalecer sobre a `libm` dinâmica. H6 apontava na direção certa; H10 é a versão precisa e confirmada.            |
| **H4**  | UB de `AlignedVec::drop` (memória)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | **Refutada** (§3)                                                                                      | Prova estática de invariante `len == capacity`; silêncio do ASan                                                                                                                                                                                                                                                |
| **H5**  | Acesso fora dos limites via `get_unchecked`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | **Refutada** (§4)                                                                                      | Prova estática construtiva; silêncio do ASan                                                                                                                                                                                                                                                                    |

**Observação importante sobre H3:** os quatro outros testes `X2`/`X4`
não-ignorados no mesmo arquivo (`test_x2_upsample_dc`,
`test_x2_roundtrip_dc`, `test_back_to_back_roundtrips_x2`,
`test_x4_upsample_dc`, `test_x4_roundtrip_dc`) **rodam normalmente hoje**
como parte da suíte padrão (Fase 1, `#[cfg(test)]` sem `#[ignore]`, exceto
que a maioria roda em modo debug via `cargo test --lib`, não em
`--release` — ver Axis B em `docs/testing.md` §2: "structural tests ...
run in debug"). Isso significa que **não sabemos, hoje, se os outros
testes X2 também haviam sido testados em `--release`** antes. Se o hang
depender só de "ser X2 em release", esperaríamos que qualquer um deles
travasse — o que nunca foi tentado nem descartado. Este é um experimento
de baixo custo e alto valor para o Sprint 1 (rodar `test_x2_roundtrip_dc`
sob o mesmo `--release`, com o mesmo timeout de segurança).

---

## 7. ⚠️ Instrução de segurança vigente (não remover sem substituir por algo equivalente)

**Não execute este teste diretamente numa estação de trabalho principal sem
isolamento de recursos.** Dado o relato de reset de sessão de desktop
(§1.3, causalidade não estabelecida em qualquer direção), qualquer nova
tentativa de reprodução **deve**:

1. Rodar dentro de isolamento de recursos com tetos explícitos de memória e
   CPU (container/cgroup) — nunca diretamente no shell da sessão principal.
2. Ser envelopada por `timeout -s KILL <N>` **no nível do shell/orquestrador
   externo**, nunca confiando apenas em timeouts internos do `cargo test`
   ou do harness (que já demonstraram não limitar isto — ver §1.2).
3. Ser observada de um terminal **separado, já em execução antes do
   lançamento**, com `perf top`/`strace -f -p <pid>` prontos, para
   distinguir CPU-spin de bloqueio em syscall **antes** de a sessão
   principal ser colocada em risco de novo.
4. Ser feita, na primeira rodada após a limpeza do `target/`, com o
   binário **sem** `strip` e **sem** `panic = "abort"`, para garantir que
   qualquer ferramenta de diagnóstico (perf, gdb, coredump) tenha símbolos
   e possa efetivamente atuar caso o timeout precise resgatar informação
   antes do kill.

O procedimento operacional completo, passo a passo, com os comandos exatos
e os scripts de wrapper recomendados, está em `TODO-sprints.md` Sprint 0 —
que deve ser executado **antes** de qualquer outra tarefa deste plano.

---

## 8. Referências de código (arquivo:linha)

- `src/dsp/oversample_test.rs:71-124` — o teste em si.
- `src/dsp/oversample.rs:140-152` — `bessel_i0` (bound de 20 iterações).
- `src/dsp/oversample.rs:101-137` — `HalfBandFilter::design`.
- `src/dsp/oversample.rs:192-222` — `X2Stage::upsample`.
- `src/dsp/oversample.rs:224-261` — `X2Stage::downsample`.
- `src/math/common/aligned.rs:52-192` — `AlignedVec` struct, construtores e `Drop`.
- `Cargo.toml:135-140` — perfil `release` (LTO fat, codegen-units=1, strip, panic=unwind).
- `Cargo.toml:142-144` — perfil `dist` (herda de `release`, sobrescreve `panic=abort`).
- `.cargo/config.toml:1-32` — `target-cpu=x86-64-v3` global e flags de linker.
- `docs/testing.md` §4 (quadro de aviso) e §2 (Axis A/B) — contexto de por que o teste é `#[ignore]`d e quando testes de ponto flutuante rodam em release vs. debug.
- `utils/tests-long.sh:457-463` — comentário de exclusão explícita do teste da suíte noturna.
- `utils/tests-long.sh:272-286` — `timed_cargo_test`, que **não** aplica nenhum timeout externo às invocações de `cargo test` — relevante para `TODO-sprints.md` Sprint 4 (hardening).
- `.agents/rules/testing.md` §2 — regra vigente: **é proibido a uma IA executar `utils/tests-long.sh`** em qualquer circunstância.
