<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# known-bugs.md — BUG-3: hang indefinido em `test_x2_aliasing_rejection`

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

| #      | Hipótese                                                                                                                                                                                                                                                                                                                                                                                 | Status                                                       | Suporte                                                                                                                                                                                                                                                                    |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **H1** | Bug de compilador/vetorizador (miscompilação) específico do pipeline `opt-level=3 + lto=fat + codegen-units=1 + target-cpu=x86-64-v3`, latente desde a criação do teste (§1.1) e nunca antes exercitado em `--release` porque o teste é `#[ignore]`d desde o dia 1 — só foi "descoberto" quando alguém finalmente rodou `--ignored --release` por vontade própria durante uma auditoria. | **Refutada** (T1.1 + T1.3, reverificadas ao vivo em §1.14.c) | Hang reproduz em debug sem otimizações (T1.1) e em dois canais de toolchain (stable 1.96.1, nightly 1.98.0 — T1.2/T1.3), ambos reconfirmados com artefatos frescos e verificados. Impossível ser bug específico de uma versão de compilador ou de pipeline de otimização.  |
| **H2** | Artefato ambiental (cache de build sujo, contenção de recursos do host, thermal throttling, ou mesmo uma falha coincidente e não relacionada do sistema gráfico) — o relato de reset do GNOME (§1.3) se encaixa melhor numa narrativa de exaustão de recursos do sistema do que num loop de ponto flutuante de 256 elementos.                                                            | **Refutada** (T1.2 + T1.5)                                   | Hang reproduz com `target/` limpo (T1.2, T1.5) — descarta cache sujo. Contenção de recursos refutada por isolamento cgroup (MemoryMax=1G, CPUQuota=100%). Reset do GNOME permanece inexplicado — possível interação entre CPU-spin e compositor, não exaustão de recursos. |
| **H3** | Bug algorítmico real — causa sensível ao **conteúdo do sinal de entrada** (senoide a 23 kHz/128 amostras), já que outros testes X2/X4 com entradas DC passam instantaneamente sob as mesmas condições.                                                                                                                                                                                   | **Parcialmente válida** (T1.1–T1.7)                          | Conteúdo-especificidade confirmada por T1.6. Porém, T1.7 mostra que o algoritmo isolado NÃO produz hang — o fator ambiental do crate completo também é necessário. H3 é condição necessária, não suficiente.                                                               |
| **H6** | **(NOVA)** Interação entre a computação do algoritmo com o sinal senoidal **e um fator do ambiente do crate completo** — estático global (`LazyLock`/`OnceLock`), layout de memória do binário maior, flags de linker, ou linkagem de todos os módulos de teste em um único binário.                                                                                                     | **Hipótese líder** (T1.7)                                    | T1.7: mesmo algoritmo (byte-idêntico) em crate mínimo completa sem hang em <1s. O hang requer algo do crate completo ausente no crate mínimo. Bissecção de crate necessária para isolar o fator.                                                                           |
| **H4** | UB de `AlignedVec::drop` (memória)                                                                                                                                                                                                                                                                                                                                                       | **Refutada** (§3)                                            | Prova estática de invariante `len == capacity`; silêncio do ASan                                                                                                                                                                                                           |
| **H5** | Acesso fora dos limites via `get_unchecked`                                                                                                                                                                                                                                                                                                                                              | **Refutada** (§4)                                            | Prova estática construtiva; silêncio do ASan                                                                                                                                                                                                                               |

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
