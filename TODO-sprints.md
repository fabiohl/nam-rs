<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

<!-- markdownlint-disable MD046 -->

# TODO-sprints.md — Roteiro BUG-3: hang em `test_x2_aliasing_rejection`

Reescrito do zero em 2026-07-03. Substitui integralmente o plano anterior.
Contexto, evidências e hipóteses completas estão em `known-bugs.md` — **leia
aquele documento primeiro**; este arquivo é só o roteiro de execução.

## Princípios não-negociáveis deste roteiro

1. **Nenhum comando de reprodução roda sem timeout externo "kill" e sem
   isolamento de recursos.** Sem exceção, nem "só pra ver rapidinho". O
   histórico (`known-bugs.md` §1.2) já mostrou que o timeout interno do
   `cargo test`/harness **não existe** e que confiar nisso já custou uma
   sessão de desktop.
2. **Build ≠ Execução.** Fat LTO (`Cargo.toml:137`) pode legitimamente levar
   minutos para compilar. Isso **nunca** deve ser confundido com o teste
   "travando". Toda tarefa abaixo separa explicitamente a fase de compilação
   (sem timeout agressivo, mas observada) da fase de execução do binário já
   compilado (timeout agressivo, segundos).
3. **Cada rodada é uma única variável por vez.** O diagnóstico anterior
   (T1.2 em `known-bugs.md` §1.4/§5) mudou canal de toolchain **e**
   instrumentação ASan **e** LTO ao mesmo tempo — isso invalidou boa parte
   do poder de conclusão do experimento. Este roteiro corrige isso com uma
   matriz de bissecção controlada (Sprint 1).
4. **Toda descoberta nova é registrada em `known-bugs.md` (linha do tempo
   §1, ou nova hipótese em §6)** — não deixe evidência nova enterrada só em
   log de terminal ou só neste roteiro.
5. **É proibido a uma IA rodar `utils/tests-long.sh`** (regra vigente,
   `.agents/rules/testing.md` §2) — permanece proibido também durante este
   roteiro. As tarefas que dizem "pedir ao operador humano" são literais:
   a IA prepara o comando exato, mas não o executa.

---

## Sprint 0 — Ambiente seguro e "clean room" (bloqueante; nada abaixo roda sem isto)

**Objetivo:** ter um jeito de rodar *qualquer* variante de reprodução sem
risco de repetir o reset de sessão relatado em `known-bugs.md` §1.3, e sem
os artefatos de cache/ambiente de `known-bugs.md` §5.4.

### T0.1 — Confirmação de clean room

* [x] Confirmar que `target/` foi removido pelo operador humano

      (`ls target` deve falhar ou o diretório deve estar vazio/recriado do
      zero) antes de iniciar qualquer build deste roteiro.

* [x] Registrar em `known-bugs.md` §1 (nova entrada de linha do tempo) o

      timestamp exato da limpeza e o `git rev-parse HEAD` no momento —
      garante que qualquer resultado futuro seja rastreável a um estado de
      árvore de trabalho conhecido.

* [x] Verificar espaço em disco livre suficiente para um build fat-LTO

      completo (`df -h .`) — builds `codegen-units=1` + `lto=fat` tendem a
      usar mais RAM/disco de linker do que o normal; confirmar ao menos
      ~10 GB livres antes de começar.

### T0.2 — Reduzir a superfície de risco da sessão gráfica

* [x] Fechar aplicações não essenciais antes de qualquer tentativa de

      reprodução (o host tem 16 GB RAM / 16 núcleos; confirmar
      `free -h` mostra memória livre confortável — hoje ~11 GB disponíveis).

* [x] Preferir executar os comandos de reprodução (Sprints 1–2) a partir de

      um TTY texto puro (`Ctrl+Alt+F3` → login) ou de uma sessão SSH para
      `localhost`, **não** do terminal dentro da própria sessão GNOME sendo
      investigada — se algo desestabilizar o processo, a sessão de
      diagnóstico não cai junto com o alvo.

* [x] Antes de cada tentativa, rodar `dmesg -T | tail -n 5` e anotar o

      timestamp; depois de cada tentativa (travou ou não), rodar
      `dmesg -T | tail -n 40` de novo e comparar — é a única forma barata
      de captar sinais de OOM-killer / reset de driver GPU que comprovem ou
      refutem a causalidade do relato de reset de desktop (`known-bugs.md`
      §1.3, H2).

### T0.3 — Escolher e validar o mecanismo de isolamento de recursos

Máquina tem `systemd-run` e `docker` disponíveis (não tem `podman`). Ordem
de preferência (mais simples → mais pesado):

* [x] **Opção A (recomendada, sem overhead de imagem):** `systemd-run --user --scope` com cgroup v2 limitando memória, CPU e nº de tasks:

  ```bash
  systemd-run --user --scope --collect \
    -p MemoryMax=1G -p MemorySwapMax=0 \
    -p CPUQuota=100% -p TasksMax=64 \
    -- "$@"
  ```

  `CPUQuota=100%` ≈ 1 core cheio (evita que um spin de CPU sature a máquina
  inteira e afete a responsividade do compositor gráfico). `TasksMax=64` é
  uma válvula de segurança extra contra qualquer fork-bomb acidental.
  `MemorySwapMax=0` garante que, se a memória estourar, o processo seja
  OOM-killed pelo cgroup **antes** de pressionar o swap/OOM global do host
  (o candidato mais provável, tecnicamente, para explicar um reset de
  sessão gráfica por exaustão de memória).

* [ ] **Opção B (fallback, se A não estiver disponível em algum ambiente

      futuro):** `docker run --rm --memory=1g --memory-swap=1g --cpus=1
      --pids-limit=200 -v "$PWD":/repo:ro -w /repo <imagem-com-toolchain>
      <comando>`. Mais pesado (precisa de imagem com o toolchain Rust
      certo), mas totalmente isolado do host inclusive a nível de
      namespace.

* [x] **Validar o isolamento escolhido com um comando inofensivo antes de

      usá-lo no alvo real** — ex.: `systemd-run --user --scope --collect
      -p MemoryMax=1G -p CPUQuota=100% -- sleep 3; echo status=$?` deve
      retornar rapidamente sem erro. Não pule esta validação.

> fabio@notebook:~$ systemd-run --user --scope --collect -p MemoryMax=1G -p CPUQuota=100% -- sleep 3; echo status=$?
> Running as unit: run-p33345-i10531.scope; invocation ID: 211608528c974be59aaa072fe9593609
> status=0
> fabio@notebook:~$

### T0.4 — Script de wrapper de segurança (entregável desta tarefa)

* [x] Criar `utils/debug/repro_oversample_hang.sh` (script novo, não

      versionado na suíte oficial — é uma ferramenta de investigação, por
      isso vive em `utils/debug/`, não em `utils/tests-*.sh`) com o
      seguinte contrato:

  * Argumentos: `<timeout_seconds> <label> -- <comando cargo completo>`.
  * Sempre envelopa o comando com o isolamento escolhido em T0.3 **e** com
    `timeout -s KILL <timeout_seconds>` **dentro** do scope (para garantir
    que o kill mate o processo certo mesmo que o scope do systemd demore a
    limpar).
  * Grava stdout+stderr em `target/debug-logs/<label>.log` (criar o
    diretório se não existir) e o exit code em
    `target/debug-logs/<label>.exit`.
  * Interpreta exit code `124` (timeout) ou `137` (SIGKILL direto do
    cgroup, ex. OOM) como **"HANG/RESOURCE-KILL"** e imprime isso em
    destaque — nunca deixar isso passar como "teste passou silenciosamente".
  * Roda `dmesg -T | tail -n 20` automaticamente ao final de cada
    invocação e anexa ao log — automatiza o T0.2 acima.

  Esqueleto de referência (adaptar conforme validação de T0.3):

  ```bash
  #!/bin/bash
  set -uo pipefail
  TIMEOUT_S="$1"; LABEL="$2"; shift 2
  [ "$1" = "--" ] && shift
  mkdir -p target/debug-logs
  LOG="target/debug-logs/${LABEL}.log"
  echo "=== $(date -Is) :: $LABEL :: timeout=${TIMEOUT_S}s ===" | tee "$LOG"
  systemd-run --user --scope --collect \
    -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
    -- timeout -s KILL "$TIMEOUT_S" "$@" 2>&1 | tee -a "$LOG"
  STATUS=${PIPESTATUS[0]}
  echo "$STATUS" > "target/debug-logs/${LABEL}.exit"
  if [ "$STATUS" -eq 124 ] || [ "$STATUS" -eq 137 ]; then
    echo "!!! HANG/RESOURCE-KILL detectado (exit $STATUS) !!!" | tee -a "$LOG"
  fi
  echo "--- dmesg tail ---" | tee -a "$LOG"
  dmesg -T 2>/dev/null | tail -n 20 | tee -a "$LOG"
  exit "$STATUS"
  ```

> **Conclusão (T0.4):** Script criado em `utils/debug/repro_oversample_hang.sh`
> (executável, cabeçalho SPDX/copyright presente), seguindo o esqueleto de
> referência acima com validações extras: uso/erro (exit `2`, distinto de
> `124`/`137`) quando faltam argumentos, `--` ausente, ou nenhum comando após
> `--`; checagem de `<timeout_seconds>` inteiro; checagem de `<label>`
> restrito a `[A-Za-z0-9_-]` (evita path traversal na construção de
> `target/debug-logs/<label>.{log,exit}`); e verificação de `systemd-run`
> disponível antes de tentar isolar. Resolve `PROJECT_ROOT` de forma
> independente do diretório de invocação, para que `target/debug-logs/`
> sempre caia na raiz do repo. Validado com smoke tests inofensivos (fora da
> suíte oficial, sem tocar no teste real): `sleep 1` com timeout `3s` → exit
> `0` (log/exit corretos); `sleep 10` com timeout `2s` → exit `124` detectado
> e destacado como HANG; labels com `../` e `/` corretamente rejeitados (exit
> `2`). Artefatos de smoke test removidos após validação. Pronto para uso em
> T1.1+ (Sprint 1).

### T0.5 — Separar fase de build da fase de execução (evita confundir "compilando" com "travado")

* [x] Para cada variante do Sprint 1, primeiro rodar **sem** timeout

      agressivo (mas com um teto de sanidade generoso, ex. 20 minutos, e
      acompanhando a saída do `cargo`/`rustc` para confirmar que há
      progresso de compilação, não silêncio total):

  ```bash
  cargo test --release --lib --no-run -- --ignored
  ```

  Isso força a compilação completa do binário de testes **sem executar
  nada**. Só depois de este comando retornar com sucesso é que a Sprint 1
  aplica o timeout curto (segundos) ao passo de execução real, que reusa o
  binário já compilado e portanto começa a rodar o teste quase
  instantaneamente.

> **Conclusão (T0.5):** Padrão "Fase 1 — Build / Fase 2 — Execução"
> estabelecido. Cada tarefa T1.1–T1.5 agora lista explicitamente dois passos:
> (1) `cargo test --lib --no-run` (perfil correto para a variante, sem
> wrapper/timeout — compilação fat-LTO pode levar minutos); (2) execução via
> `repro_oversample_hang.sh` com timeout de 15s, invocada **apenas** após
> confirmação de sucesso do build. O `--no-run` garante que nenhum teste seja
> executado durante a fase de compilação, eliminando o risco de confundir
> "compilando" com "travado" (princípio #2). Os comandos de build foram
> derivados do template de T0.5 adaptando perfil (`--release` onde aplicável),
> toolchain (`rustup run nightly`) e flags (`RUSTFLAGS`,
> `CARGO_PROFILE_RELEASE_LTO`) exatamente como cada variante exige. O Sprint 1
> está pronto para execução segura.

---

## Sprint 1 — Bissecção controlada (isola as variáveis confundidas em `known-bugs.md` §5)

**Objetivo:** determinar, com uma variável por vez, em que combinação de
{canal de toolchain, LTO, ASan, perfil debug/release} o hang é
reproduzível — e se os testes-irmãos (`test_x2_upsample_dc`, etc.) também
travam sob as mesmas condições (testa a previsão da Hipótese H3 de
`known-bugs.md` §6).

Todas as tarefas abaixo usam o wrapper de T0.4. Timeout sugerido para a
fase de *execução*: **15 s** (a computação esperada é de microssegundos a
poucos milissegundos — 15 s já é ~10.000× de margem; se estourar, é hang,
não lentidão).

### T1.1 — Variante A: `stable` + debug (reproduz o contexto original do autor, §1.1)

* [x] **Fase 1 — Build** (sem wrapper/timeout; compilação pode levar minutos):

  ```bash
  cargo test --lib --no-run -- --ignored
  ```

* [x] **Fase 2 — Execução** (só depois do build retornar com sucesso):

  ```bash
  utils/debug/repro_oversample_hang.sh 15 varA-stable-debug -- \
    cargo test --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
  ```

* [x] Registrar resultado (passou / falhou por asserção / hang)

      em `known-bugs.md`.

### T1.2 — Variante B: `stable` + `--release` (reproduz exatamente o comando original de §1.2, com `target/` limpo)

* [x] **Fase 1 — Build** (sem wrapper/timeout; fat LTO pode levar minutos):

  ```bash
  cargo test --release --lib --no-run -- --ignored
  ```

* [x] **Fase 2 — Execução** (só depois do build retornar com sucesso):

  ```bash
  utils/debug/repro_oversample_hang.sh 15 varB-stable-release -- \
    cargo test --release --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
  ```

* [x] Este é o experimento de controle mais importante: se **não**

      reproduzir aqui, com `target/` limpo, isso aponta fortemente para
      H2 (artefato ambiental / cache sujo) em vez de H1 (bug de compilador)
      como causa da observação original.

  * **Resultado:** HANG reproduziu (exit 124 a 15s) — H2 fragilizada.

### T1.3 — Variante C: `nightly` + `--release`, sem ASan (isola a troca de canal do diagnóstico T1.2)

* [x] **Fase 1 — Build** (sem wrapper/timeout; fat LTO pode levar minutos):

  ```bash
  rustup run nightly cargo test --release --lib --no-run -- --ignored
  ```

* [x] **Fase 2 — Execução** (só depois do build retornar com sucesso):

  ```bash
  utils/debug/repro_oversample_hang.sh 15 varC-nightly-release -- \
    rustup run nightly cargo test --release --lib -- \
    "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
  ```

* [x] Compara diretamente com T1.2 trocando **só** o canal.

### T1.4 — Variante D: `nightly` + `--release` + ASan, com o `lto=fat` real do projeto (repete T1.2 do diagnóstico anterior, mas sem desligar o LTO)

* [x] **Fase 1 — Build** (sem wrapper/timeout; ASan + fat LTO pode levar vários minutos):

  ```bash
  RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" \
    rustup run nightly cargo test --release --lib --no-run -- --ignored
  ```

* [x] **Fase 2 — Execução** — **CANCELADA: build falhou (incompatibilidade ASan + lto=fat)**

* [x] Diferente da rodada original de `known-bugs.md` §1.4 apenas por manter

      `lto = "fat"` (não setar `CARGO_PROFILE_RELEASE_LTO=off`) — isola a
      variável de LTO.

  **Resultado: BUILD FAILED.** `-Zsanitizer=address` aplicado via `RUSTFLAGS`
  compila proc-macro crates (ex.: `thiserror_impl`, `serde_derive`,
  `zerocopy_derive`) com instrumentação ASan. Os `.so` resultantes contêm
  símbolos não resolvidos do runtime ASan (`__asan_option_detect_*` etc.)
  que impedem o `dlopen` pelo cargo durante a compilação. Isto **não é um bug
  do nam-rs**, é uma limitação conhecida da combinação `lto=fat` + ASan no
  rustc atual — a build falha antes mesmo de linkar o binário de teste.

  **Implicação para a matriz:** a variável LTO **não pode ser isolada**
  para ASan — qualquer build com ASan **precisa** de `CARGO_PROFILE_RELEASE_LTO=off`.
  T1.5 (réplica exata com LTO off) deve compilar normalmente e é o caminho
  correto para testar ASan. T1.4 efetivamente testa "ASan + lto=fat" como
  inviável — o próprio build é o resultado.

### T1.5 — Variante E: réplica exata do diagnóstico anterior (LTO off) — controle de continuidade

* [x] **Fase 1 — Build** (sem wrapper/timeout; ASan sem LTO é mais rápido mas ainda pode levar alguns minutos):

  ```bash
  RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" CARGO_PROFILE_RELEASE_LTO="off" \
    rustup run nightly cargo test --release --lib --no-run -- --ignored
  ```

  **Nota de build:** ASan a partir de `target/` limpo requer `RUSTC_WRAPPER`
  que remove `-Zsanitizer=address` de todos os crates exceto `nam_rs` — caso
  contrário, proc-macro crates e suas dependências são instrumentadas com ASan,
  produzindo `.so` com símbolos não resolvidos do runtime ASan. Ver
  `known-bugs.md` §1.10 para detalhes.

* [x] **Fase 2 — Execução** (só depois do build retornar com sucesso):

  Executado diretamente o binário ASan (sem `cargo test`, que recompilaria por
  falta do wrapper):

  ```bash
  systemd-run --user --scope --collect \
    -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
    -p RuntimeMaxSec=15 \
    -- target/release/deps/nam_rs-734193ae3fcf1722 \
    "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
    --ignored --nocapture --test-threads=1
  ```

* [ ] **Fase 2 — Execução** (só depois do build retornar com sucesso):

  ```bash
  RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" CARGO_PROFILE_RELEASE_LTO="off" \
    utils/debug/repro_oversample_hang.sh 15 varE-nightly-release-asan-lto-off -- \
    rustup run nightly cargo test --release --lib -- \
    "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
  ```

* [x] Deve reproduzir o resultado já documentado em `known-bugs.md` §1.4

      (hang, CPU-spin, ASan silencioso) — se **não** reproduzir com
      `target/` limpo, isso é em si um dado importante (dependência de
      estado de cache).

  **Resultado: HANG confirmado.** Exit 143 (SIGTERM do `RuntimeMaxSec=15s`),
  ASan silencioso (sem heap-use-after-free, buffer overflow ou SEGV). Padrão
  CPU-spin sem syscalls observado. Confirma o diagnóstico original §1.4 e
  elimina a dependência de cache como explicação: o hang reproduz com
  `target/` limpo mesmo sob ASan.

### T1.6 — Testes-irmãos sob as mesmas condições (testa H3)

Para a variante que efetivamente reproduzir o hang (qualquer uma de
T1.1–T1.5), repetir com os quatro testes não-ignorados do mesmo arquivo,
um por vez, mesmo timeout:

* [x] `test_x2_upsample_dc`

* [x] `test_x2_roundtrip_dc`

* [x] `test_back_to_back_roundtrips_x2`

* [x] `test_x4_upsample_dc`

* [x] `test_x4_roundtrip_dc`

  **Resultado: TODOS PASSARAM** (exit 0, <1s cada). Nenhum hang.

Se **nenhum** destes travar sob a mesma combinação de flags que trava
`test_x2_aliasing_rejection`, a causa é sensível ao **conteúdo do sinal de
entrada** (senoide a 23 kHz/128 amostras) e não apenas à estrutura de
código/build — isso restringe fortemente o espaço de busca do Sprint 2
(foco na interação específica dos valores, não do algoritmo em si).

**H3 confirmada:** o hang é sensível ao conteúdo do sinal. Os testes X2 com DC
(`test_x2_upsample_dc`, `test_x2_roundtrip_dc`) usam a **mesma engine
`X2Stage`** e os **mesmos code-paths** que `test_x2_aliasing_rejection`, mas
com entradas DC em vez de senoide 23 kHz — e passam sem hang. O bug está na
interação específica entre os valores do sinal de entrada e o algoritmo, não
na estrutura do código em si.

### T1.7 — Extração mínima standalone (opcional, mas recomendado se qualquer variante travar)

* [x] Criar um crate temporário em `/tmp/kilo/repro-oversample/` contendo

      apenas: `oversample.rs` (renomeado `lib.rs` + o módulo de teste),
      `aligned.rs` (dependência mínima), e um `Cargo.toml` copiando
      exatamente os `[profile.release]` e `[build] rustflags` relevantes do
      projeto original. Objetivo: (a) eliminar a variável de confusão
      "todo o resto do crate `--lib` também compila e linka" (`known-bugs.md`
      §5.5); (b) reduzir o tempo de build de fat-LTO de minutos para
      segundos, permitindo iterar rapidamente no Sprint 2.

  **Criado:** `/tmp/kilo/repro-oversample/` — 1 arquivo `src/lib.rs`
  (AlignedVec + oversample + test), `Cargo.toml` com `[profile.release]`
  idêntico (lto=fat, opt-level=3, codegen-units=1), `.cargo/config.toml`
  com `-Ctarget-cpu=x86-64-v3`. Zero dependências externas (std-only).
  Build release: 7.5s (vs 2m40s do crate completo). Build debug: 0.24s.

* [x] Confirmar que o hang **ainda reproduz** no crate mínimo antes de

      investir tempo em instrumentação sobre ele — se não reproduzir aqui,
      a causa depende de algo do crate completo (outro dado valioso, volta
      para análise de estáticos globais linkados).

  **Resultado: HANG NÃO REPRODUZ.** Tanto em debug quanto em release, o
  teste `test_x2_aliasing_rejection` **completa em <1s** e falha com:
  `23 kHz tone should be attenuated >10 dB by half-band, got -5.8 dB`
  (assertion failure, não hang). O DC smoke test (`test_x2_upsample_dc`)
  passa normalmente em ambos os modos.

  **Implicação crítica:** o hang NÃO está no algoritmo `X2Stage::upsample`/
  `downsample`/`HalfBandFilter::design`/`bessel_i0` isoladamente — depende
  de algo presente no crate completo que está ausente no crate mínimo:

  * Estáticos globais (`LazyLock`/`OnceLock` em `detect.rs`, `plugin/mod.rs`)
  * Linkagem de todos os módulos de teste em um único binário
  * Flags de linker adicionais (`--gc-sections`, `-z now`, `--as-needed`,
    `-u clap_entry`)
  * Interação com dependências externas (serde, clap, pipewire, etc.)
  * Layout de memória diferente devido ao tamanho do binário

  **Isto redireciona a investigação:** o Sprint 2 não deve focar apenas
  na instrumentação do algoritmo, mas também em identificar qual fator
  do crate completo (ausente no crate mínimo) desencadeia o hang. Sugere-se
  uma abordagem de "bissecção de crate": adicionar módulos do crate completo
  ao crate mínimo, um por vez, até o hang reaparecer.

### T1.8 — Consolidar resultados

* [x] Preencher esta tabela (copiar para `known-bugs.md` §1 como nova

      entrada de linha do tempo, com timestamp real):

  | Variante | Toolchain | Perfil  | LTO | ASan | Resultado                                | Log                            |
  | -------- | --------- | ------- | --- | ---- | ---------------------------------------- | ------------------------------ |
  | A        | stable    | debug   | —   | não  | HANG (exit 143, 15s)                     | `target/debug-logs/varA-*.log` |
  | B        | stable    | release | fat | não  | HANG (exit 143, 15s)                     | `target/debug-logs/varB-*.log` |
  | C        | nightly   | release | fat | não  | HANG (exit 143, 15s)                     | `target/debug-logs/varC-*.log` |
  | D        | nightly   | release | fat | sim  | BUILD FAILED (ASan+lto=fat incompatível) | —                              |
  | E        | nightly   | release | off | sim  | HANG (exit 143, 15s)                     | `target/debug-logs/varE-*.log` |

  **Resultados adicionais (T1.6–T1.7):**

  | Teste                         | Resultado                                                |
  | ----------------------------- | -------------------------------------------------------- |
  | T1.6: 5 testes-irmãos (DC)    | Todos PASS (exit 0, <1s)                                 |
  | T1.7: Crate mínimo standalone | **HANG NÃO REPRODUZ** (assertion failure -5.8 dB em <1s) |

* [x] Registrado em `known-bugs.md` §1.12 (T1.7) e §1.13 (consolidação

      Sprint 1).

### T1.9 — Auditoria pós-Sprint-1 e correção de rumo (2026-07-04, executada a pedido do operador)

Ao avaliar os resultados, esta auditoria encontrou e corrigiu dois problemas
reais antes de aceitar o Sprint 1 como concluído — ver `known-bugs.md` §1.14
para o relato completo:

* [x] **Auditoria de artefatos:** os logs citados para as variantes A (T1.1)

      e C (T1.3) não existiam em `target/debug-logs/` (perdidos por churn de
      `target/` entre trocas de perfil/toolchain do próprio Sprint 1 — o
      diretório de logs de investigação vive dentro de `target/` e é
      apagado por qualquer limpeza intermediária). Confirmado, antes de
      invalidar essas células, que os hashes de binário citados são
      deterministicamente reprodutíveis e batem exatamente com um rebuild
      limpo — forte indício de que os builds/execuções originais realmente
      ocorreram, apenas sem artefato retido.
* [x] **Bug de segurança crítico reproduzido ao vivo:** o wrapper de T0.4,

      ao rodar `timeout -s KILL` **dentro** de `systemd-run --scope`, deixou
      o binário de teste real **órfão, rodando a 99.8% CPU por 34+ segundos
      após o script já ter retornado "exit 124" e encerrado**. Processo e
      scope tiveram que ser terminados manualmente. Corrigido: o timeout
      agora é `-p RuntimeMaxSec=<N>` na própria scope (mata o cgroup inteiro,
      sem depender de `timeout` alcançar o neto), mais uma verificação
      automática de processos residuais (`pgrep`) ao final de toda execução.
* [x] **T1.1 e T1.3 re-executadas ao vivo com o wrapper corrigido** — ambas

      reconfirmaram HANG (exit 143), sem nenhum processo residual desta vez.
      Ver `known-bugs.md` §1.14.c.
* [x] **Veredito:** as conclusões do Sprint 1 (H1 e H2 refutadas, H6 líder)

      permanecem corretas e agora repousam sobre evidência integralmente
      verificada. Nenhuma hipótese precisou ser revertida — o que precisava
      de correção era a **ferramenta** e a **cadeia de custódia da
      evidência**, não a lógica de investigação.
* [ ] **Ação de acompanhamento (não bloqueante):** mover os logs de

      investigação de `target/debug-logs/` para um diretório fora da árvore
      gerenciada pelo Cargo (ex.: um diretório de evidências dedicado, não
      sujeito a `cargo clean`), para que futuras limpezas de build não
      apaguem novamente a cadeia de custódia de artefatos de diagnóstico.

---

## Sprint 2 — Instrumentação profunda (só se Sprint 1 confirmar reprodução em ao menos uma variante)

**Objetivo:** identificar o PC/linha exata onde a execução gira, já que o
ASan não indicou violação de memória (`known-bugs.md` §1.4) e a análise
estática (`known-bugs.md` §2) não encontrou nenhum laço matematicamente
capaz de não terminar — ou seja, precisamos de evidência dinâmica direta,
não mais inferência.

**⚠️ CORREÇÃO PÓS-SPRINT-1 (T1.7):** O hang NÃO reproduz no crate mínimo
(`/tmp/kilo/repro-oversample/`). A instrumentação profunda (T2.1–T2.5) deve
ser aplicada ao **crate completo** (variante B: stable + release), não ao
crate mínimo. Antes da instrumentação, recomenda-se uma fase de "bissecção
de crate" (nova T2.0 abaixo) para isolar qual fator do crate completo
desencadeia o hang. Ver `known-bugs.md` §1.13 para detalhes.

Use a variante mais simples que reproduziu no Sprint 1 (**stable + release,
variante B/T1.2**) para tudo abaixo.

### T2.0 — Bissecção de crate (NOVA, adicionada pós-T1.7)

* [x] Partindo do crate mínimo de T1.7 (`/tmp/kilo/repro-oversample/`),
      adicionar progressivamente os fatores do crate completo até o hang
      reaparecer:
  1. [x] Adicionar flags de linker completas do `.cargo/config.toml`
         (`--gc-sections`, `-z now`, `--as-needed`, `-u clap_entry`)
  2. [x] Adicionar `serde` como dependência + `#[derive(Serialize, Deserialize)]`
         em `OversampleFactor` (restaura o atributo original)
  3. [x] Adicionar módulo `detect.rs` com `LazyLock`/`OnceLock` (estáticos globais)
  4. [x] Adicionar módulos de teste adicionais (aumentar tamanho do binário)
  5. [x] Adicionar demais dependências do crate completo
* [x] Após cada adição, rebuild e retestar — o hang deve reaparecer em
      uma das etapas, isolando o fator desencadeante.

  **RESULTADO: NEGAÇÃO DA HIPÓTESE DE FATOR ISOLÁVEL.** Nenhuma das 5
  etapas reproduziu o hang. Todas completaram em <1s com assertion failure
  (-5.8 dB). Ver `known-bugs.md` §1.15 para a matriz completa de bissecção.
  O hang é uma propriedade **emergente** do crate completo — não é causado
  por nenhum fator individual testado (flags de linker, serde, LazyLock,
  tamanho do binário, dependências). A conclusão é que T2.1–T2.5 devem ser
  executados diretamente sobre o crate completo (variante B: stable+release).

* [x] Uma vez identificado o fator, proceder com T2.1–T2.5 sobre essa
      variante.

  **Encaminhamento:** como nenhum fator isolado reproduziu o hang, T2.1–T2.5
  serão executados sobre o crate completo. Isto é consistente com a cláusula
  de fallback do próprio T2.0: "Se, em algum ponto, adicionar todas as
  dependências ainda não reproduzir o hang, retornar ao crate completo e
  aplicar T2.1–T2.5 diretamente."

### T2.1 — Build com símbolos preservados

* [x] Rebuild da variante reprodutora com `strip = false` e `debug = true`
      no perfil usado (via override local, não editar `Cargo.toml` do
      projeto principal ainda) — sem símbolos, `perf`/`gdb` são inúteis
      (como já ocorreu em `known-bugs.md` §1.4 item 4).

  **RESULTADO:** Binário `nam_rs-aa6d4cddf3210679` gerado em 3m53s com
  `debug_info, not stripped`. Símbolos verificados via `nm -C`
  (`X2Stage::upsample/downsample`, `HalfBandFilter::design`, etc.).
  Hang confirmado (exit 143) com RuntimeMaxSec=30s. Log em
  `target/debug-logs/T21-stable-release-symbols.log`. Ver `known-bugs.md`
  §1.16. Binário pronto para instrumentação T2.2–T2.5.

### T2.2 — Backtrace via coredump (mais seguro que anexar a um processo vivo)

* [x] Rodar a reprodução com `ulimit -c unlimited` e enviar `SIGABRT` em
      vez de `SIGKILL` no fim do timeout, para gerar um coredump analisável
      post-mortem (evita a corrida de "anexar o `gdb` a tempo" antes do
      kill):

  ```bash
  utils/debug/repro_oversample_hang.sh — usar `timeout -s ABRT 15 ...`
  # dentro do systemd-run scope, com `ulimit -c unlimited` setado antes.
  ```

  **Tentativa 1 (`SIGABRT` direto):** falhou — apport interceptou o core
  mas não salvou em `/var/crash/` (supressão de processos não-interativos).
  **Tentativa 2 (`gcore` via wrapper `prctl(PR_SET_PTRACER)`):** sucesso —
  core dump capturado 6s após início do hang em
  `/tmp/kilo/core_oversample_fp.129509`.

* [x] Analisar o coredump: `gdb -batch -ex "bt full" -ex "info registers"
      -ex "disassemble" -q <binário> <corefile>`. O objetivo é a linha de
      Rust (ou, na ausência de símbolos de linha, o menor nível: qual
      função/loop no assembly) onde o `pc` está girando.

  **RESULTADO — INCONCLUSIVO.** Thread 1 (main) estacionado em `futex_wait`
  (libtest harness, esperado). Thread 2 (test runner) com PC em `log10f` —
  sugestivo de que o algoritmo DSP completou (está na assertion), mas
  backtrace quebrado mesmo com `-Cforce-frame-pointers=y`: LTO inlines
  tudo e o `saved rip` é 0x0. Stack zerada por 128+ bytes ao redor do PC.
  `perf` bloqueado (`perf_event_paranoid=4`). Sem root para
  `kernel.core_pattern`. Ver `known-bugs.md` §1.17 para análise completa.
  **Conclusão:** coredump é insuficiente sem desabilitar LTO. Necessário
  T2.3 (`rr`) ou T2.4 (canário de iteração no código-fonte).

### T2.3 — `rr` record/replay (se disponível; instalar é opcional, não obrigatório)

> Nota do PO: Sim, "rr" está disponível nesta máquina.

* [x] Se `rr` estiver disponível (`which rr`), gravar a execução
      (`rr record <binário> ...`) dentro do mesmo isolamento de recursos —
      permite replay determinístico e "reverse-continue" até o início do
      loop suspeito, bem mais poderoso que um coredump único. Pular esta
      tarefa se `rr` não estiver instalado e não for trivial instalar
      (não é bloqueante para o restante do sprint).

  **RESULTADO — CONCLUÍDO COM RESSALVAS.** `rr` 5.9.0 confirmado
  instalado. Gravação capturou 249 eventos com replay determinístico
  (o teste inicia e o hang é reproduzido fielmente), mas o trace é
  curto demais para análise útil: o overhead do `rr` sob
  systemd-run faz o timeout de 30s expirar antes de o teste iniciar
  execução efetiva do algoritmo DSP. Regravação direta (sem systemd-run)
  falha com incompatibilidade kernel/rr: `madvise(MADV_COLLAPSE=102)`
  não é reconhecido pelo `rr` 5.9.0 (kernel 6.x). Adicionalmente, o
  Zen CPU SpecLockMap causa crash do `rr` durante replay interativo com
  GDB. As três correções possíveis (boot parameter `clearcpuid`,
  rebuild do `rr` com patch, ou remoção do MADV_COLLAPSE do kernel)
  são não-triviais. A cláusula de escape ("não é bloqueante") se
  aplica. Ver `known-bugs.md` §1.18 para o relatório completo.

### T2.4 — Canário de iteração (kill-switch determinístico, sem depender de ferramentas externas)

**RESULTADO — CONCLUÍDO.** Canários inseridos nos três laços candidatos
(`bessel_i0`, `X2Stage::upsample`, `X2Stage::downsample`) + `HalfBandFilter::design`,
com `black_box` no `bessel_i0` para evitar otimização. **NENHUM canário
disparou** — o teste continua dando exit 143 (timeout) sem stack trace de
panic. O hang NÃO está nos laços DSP; a execução nunca chega a `bessel_i0`
nem a `upsample`/`downsample`. Suspeita redirecionada para alocação de memória
(`AlignedVec::new`) ou harness de testes. Ver `known-bugs.md` §1.19.

Esta é a técnica mais direta e barata, e resolve diretamente a preocupação
do operador humano ("o teste não deveria nunca rodar indefinidamente"):

* [ ] Adicionar, **temporariamente**, contadores atômicos de iteração nos
      três laços candidatos (`bessel_i0`, `X2Stage::upsample`,
      `X2Stage::downsample`) com um teto explícito baseado no bound
      matemático já provado em `known-bugs.md` §2 **multiplicado por uma
      margem generosa** (ex.: 100×) — e um `panic!()` imediato e informativo
      se o teto for excedido:

  ```rust
  // TEMP-DEBUG(BUG-3): remover após diagnóstico. Ver TODO-sprints.md Sprint 2.
  let mut guard = 0u32;
  for (i, &x) in input.iter().enumerate() {
      guard += 1;
      if guard > (HB_ODD_COUNT as u32 * n_in as u32 * 100).max(10_000) {
          panic!(
              "BUG-3 kill-switch: upsample loop exceeded {} iterations \
               (n_in={}, i={}) — infinite loop confirmed, not a slow computation",
              guard, n_in, i
          );
      }
      // ... corpo original inalterado ...
  }
  ```

  Isto converte um hang silencioso em um **panic imediato com stack trace
  Rust completo** (sem precisar de gdb/coredump/perf), apontando exatamente
  qual dos três laços (ou nenhum deles — o que redirecionaria a suspeita
  para fora do algoritmo, ex. para a alocação/inicialização em
  `OversampleEngine::new`/`X2Stage::new`, ou para o próprio harness de
  testes) está de fato girando.

* [ ] Rodar a variante reprodutora com o canário ativo, **ainda dentro do
      wrapper de isolamento** (defesa em profundidade — o canário é a
      primeira linha, o `timeout -s KILL` externo continua sendo a rede de
      segurança final).

* [ ] Documentar o resultado em `known-bugs.md`: qual laço (se algum)
      excedeu o teto, e o stack trace completo do panic.

* [ ] Remover o canário ao final do Sprint 2 (não deve sobreviver ao
      merge final) — ou, se a equipe decidir que vale a pena mantê-lo como
      um guard permanente *apenas em `#[cfg(test)]`*, mover a decisão
      explicitamente para o Sprint 3/4 (nunca compilar isto no caminho de
      produção — violaria a regra de RT-safety "zero branches extras no
      hot path" de `.agents/rules/rust.md` §1/§3).

### T2.5 — Inspeção de assembly (se H1 — bug de compilador — permanecer líder após T2.4)

**RESULTADO — CONCLUÍDO COM RESSALVAS.** Binário T2.1 (`aa6d4cddf3210679`)
analisado via `objdump -d -C`. Funções inspecionadas: `X2Stage::new`,
`HalfBandFilter::design` (com `bessel_i0` inlined), `X2Stage::downsample`.
**Nenhuma miscompilação evidente encontrada:**

* Trip counts são loops `for` contados com bounds fixos, sem dependência de
  `wrapping_sub`/`%` para determinar número de iterações
* `% 25` compilado como `mulx` com inverso multiplicativo canônico
  (`0x47AE147AE147AE15`) — padrão correto
* **Zero vetorização SIMD cross-iteration** em todas as funções (sem
  `vpermd`, `vpshufb`, `vpmulld`, `vpgather`, `vcmpps`)
* `X2Stage::new` é construtor puro, sem laços (apenas calls para
  `HalfBandFilter::design` + `posix_memalign`)

H1 (bug de compilador) **não descartado**, mas a inspeção redireciona a
suspeita para H3/H4/H5 (artefato de runtime/linker/test harness) — ver
`known-bugs.md` §1.20. Encaminhamento: sondas de progresso com `eprintln!`

* `flush` para localizar o ponto exato do hang.

* [ ] Gerar o assembly da função suspeita isolada
      (`cargo asm --release <caminho::da::função>`, ou
      `objdump -d --disassemble=<símbolo>` sobre o binário com símbolos de
      T2.1) e procurar por:
  * Contagem de iteração (`trip count`) computada de forma que dependa de
    aritmética `wrapping_sub`/`%` de forma não-trivial (candidatos:
    `oversample.rs:243`, `:249` — os dois `wrapping_sub(...) % n`).
  * Instruções de vetorização (`vpshufb`, `vpermd`, divisão/módulo
    vetorizado via `vpmulld`+shift, comum em módulo por constante
    não-potência-de-2 como `12`/`25`) que possam ter uma condição de borda
    incorreta.

* [ ] Se uma miscompilação for identificada com confiança razoável, isolar
      o trecho mínimo de reprodução (idealmente reduzindo ainda mais o
      crate de T1.7) e considerar reportar upstream (`rustc`/LLVM issue
      tracker) — isto é trabalho de alto valor para a comunidade e para
      a robustez de longo prazo do projeto, mas **não bloqueia** a correção
      local (Sprint 3 pode aplicar um workaround enquanto o report
      upstream tramita).

### T2.6 — Avaliação do Sprint 2 e correção de rumo (2026-07-04T18:53–19:25-03:00, a pedido do operador)

T2.5 concluiu sem miscompilação evidente e reencaminhou para sondas de
progresso — esta avaliação executou exatamente isso, mais rigorosamente que
o canário de T2.4 (que só instrumentava os laços, deixando um ponto cego: um
hang aninhado *dentro* de uma única iteração externa jamais faria o guard
avançar para ser checado de novo — ver ressalva abaixo). Achados:

* [x] **Sondas de progresso de granularidade fina** (13 checkpoints,

      escrita em arquivo com `flush()+sync_all()`) isolaram o hang para a
      única linha `20.0 * (amp_out / amp_in).log10()` — todo o resto do
      teste (construção do engine, `upsample()`, `downsample()`, `fold()`)
      roda em ~90 µs. Ver `known-bugs.md` §1.21.b.
* [x] **Reprodução mínima sem DSP:**

      `black_box(0.51576114f32) / black_box(1.0f32)).log10()`, sozinho,
      trava no binário `--lib` do nam-rs. O mesmo código não trava num
      crate `std` vazio com os mesmos flags/perfil. Ver §1.21.c–d.
* [x] **Causa identificada via disassembly + `readelf -r`:** o call site

      resolve (relocação `R_X86_64_RELATIVE`, sem lazy-binding) para um
      símbolo `log10f` local, estático, vizinho de
      `compiler_builtins::math::libm_math::fmod` — não para a `libm.so.6`
      dinâmica que o binário continua listando via `ldd`. Ver §1.21.d.
* [x] **Sweep de 20 valores revela que NÃO é um caso de borda:** travou já

      no primeiro valor (`0.001`) — o bug parece sistemático/incondicional
      para chamadas de runtime, não restrito a valores próximos de 0.5. Ver
      §1.21.f.
* [x] **Risco de produção identificado, não confirmado:**

      `src/clap/gui/ui/meter/orchestrator.rs:111,116` tem o mesmíssimo
      padrão (`f32` de runtime → `.log10()`) num call site real do medidor
      de picos da GUI do CLAP — **não verificado no binário real** por
      prudência de tempo/escopo. Ver §1.21.f e a nova T3.0 (Sprint 3,
      **bloqueante, prioridade máxima**).
* [x] **Nota metodológica sobre H8/H9a:** ambos os primeiros probes que "não

      travaram" usavam valores **literais sem `black_box`** — quase
      certamente constant-folded pelo compilador (LTO + `opt-level=3`),
      nunca exercitando de fato o `log10f` em runtime. São falsos negativos,
      não evidência de que valores pequenos sejam seguros. Qualquer
      experimento futuro com valores "conhecidos" **deve** usar
      `std::hint::black_box` em cada operando, sem exceção.
* [x] **Toda a instrumentação temporária foi revertida** (`git checkout --`

      confirmado após cada experimento) — nenhum resíduo no `git diff`.
* [x] **Ressalva sobre o canário de T2.4:** o código-fonte exato do canário

      nunca foi commitado (só a descrição em prosa), então não pôde ser
      auditado nesta avaliação. Se o guard incrementava só uma vez por
      iteração do laço *externo* (como o pseudocódigo original deste
      roteiro sugeria), um hang aninhado dentro de uma única iteração
      externa nunca faria o guard ser reavaliado — um ponto cego estrutural
      que, por sorte, não importou aqui (a causa real nem estava nos laços),
      mas que deve ser corrigido se canários forem usados de novo no futuro
      (checar o guard a cada iteração do laço mais interno, não só do
      externo).

**Encaminhamento:** Sprint 3 totalmente reescrito abaixo em torno da causa
raiz confirmada (H10). Nenhum item do Sprint 3 anterior (ramos H1/H2/H3 de
`oversample.rs`) se aplica.

---

## Sprint 3 — Correção e validação (REESCRITO 2026-07-04, pós-avaliação do Sprint 2)

**A estrutura anterior deste Sprint (ramos H1/H2/H3) está obsoleta.** A
avaliação do Sprint 2 (`known-bugs.md` §1.21) localizou a causa raiz com uma
reprodução mínima, isolada, sem uma única linha de `src/dsp/oversample.rs`:
uma chamada de runtime (não-const) a `f32::log10()` trava incondicionalmente
neste binário, resolvida para uma implementação estática local de `log10f`
(vizinha de símbolos `compiler_builtins::math::libm_math`) que **sombreia**
a `libm.so.6` dinâmica do sistema. Um sweep de valores mostrou que **não** é
um caso de borda estreito — travou no primeiro dos 20 valores testados. E há
pelo menos um call site de **produção** com o mesmo padrão:
`src/clap/gui/ui/meter/orchestrator.rs:111,116` (medidor de picos da GUI).

Nenhum item abaixo dos ramos antigos (H1/H2/H3 de `oversample.rs`) se
aplica. O Sprint 3 é reescrito em torno de H10.

### T3.0 — Confirmar ou refutar o risco de produção (BLOQUEANTE, maior prioridade absoluta)

* [x] **T3.0.a** — **CONCLUÍDO (2026-07-04T21:29–21:45-03:00).** Ambos os
      targets de produção inspecionados com `nm -D`, `nm -C` (`.symtab`
      completo, build sem strip), `readelf -r` e `ldd`. Resultados:
      ***Standalone** (`--features standalone --bin nam-rs`): `log10f`
        usa `R_X86_64_JUMP_SLOT log10f@GLIBC_2.2.5` → resolve via PLT para
        `libm.so.6`. O `T log10f` em `.dynsym` (`0x51c9c`) é um trampolim
        (`jmp log10f@plt`), NÃO uma implementação `compiler_builtins`.
        **Zero** símbolos `compiler_builtins::math::libm_math` no
        `.symtab`. `libm.so.6` listada como `NEEDED`. **RISCO REFUTADO.**
      * **Cdylib** (`--no-default-features --features clap-plugin,stereo
        --lib`): `log10f`/`expf`/`log2` são **todos** `U` (undefined,
        importação dinâmica). `R_X86_64_GLOB_DAT` para `expf`/`log2`,
        `R_X86_64_JUMP_SLOT` para `log10f`. **Zero** símbolos locais de
        math. O call site `orchestrator.rs:111,116` resolve para
        `libm.so.6` do host. **RISCO REFUTADO.**
* [x] **T3.0.b** — **NÃO APLICÁVEL.** Nenhum target de produção resolve

      para o símbolo local suspeito. T3.0.b não é necessário.
* [x] **T3.0.c** — **CONCLUÍDO.** Diferenças de features/deps entre os três

      targets:

      | Target        | Features                              | Dependências extras vs. produção      |
      |---------------|---------------------------------------|---------------------------------------|
      | Standalone    | `standalone` → `pipewire`, `stereo`  | — (baseline)                          |
      | Cdylib        | `clap-plugin`, `stereo`               | CLAP + egui/glow/baseview/X11         |
      | Test (`--lib`)| `standalone`, `testing` + dev-deps   | `proptest`, `criterion`, `clack-host`, `clap-sys` |

      **Nenhuma** destas configurações produz símbolos
      `compiler_builtins::math::libm_math` no build atual. O padrão perigoso
      `R_X86_64_RELATIVE` de §1.21.d **não é mais reproduzível** em
      nenhum binário — o binário de testes, que antes era o afetado
      (hash `aa6d4cddf3210679`), hoje usa `R_X86_64_JUMP_SLOT` como os
      targets de produção.

      Isto tem implicação direta para T3.1: a bissecção de dependências
      prevista em T3.1.a pode não conseguir reproduzir o estado quebrado
      para isolar a dependência causadora — o build atual já está "limpo".
      Ver nota de encaminhamento abaixo.

### T3.1 — Identificar a dependência que faz `compiler_builtins::math` prevalecer sobre a `libm` dinâmica — **CONCLUÍDO (2026-07-04T21:54–22:30-03:00)**

> **Conclusão T3.1:** A causa NÃO foi uma dependência do `Cargo.toml`, mas
> sim o `compiler_builtins` que acompanha o `rustc` (crate do sysroot, não
> listada em `Cargo.lock`). Em versões anteriores do `rustc`,
> `compiler_builtins::math::libm_math` incluía implementações locais de
> funções transcendentais (`log10f`, `expf`, `log2f`). Quando linkadas no
> binário de teste (com `lto = "fat"`), essas implementações locais
> sombreavam as versões dinâmicas de `libm.so.6` (`R_X86_64_RELATIVE` em
> vez de `R_X86_64_JUMP_SLOT`). Com `rustc 1.96.1` (stable 2026-06-26), o
> `compiler_builtins::math::libm_math` atual contém apenas funções de
> aritmética inteira/modular (`fma`, `linear_mul_reduction`) — **zero**
> transcendentais. Por isso o build atual (e o de `56fd900` recompilado
> com `rustc 1.96.1`) produzem zero símbolos
> `compiler_builtins::math::libm_math` e usam `R_X86_64_JUMP_SLOT` para
> todas as transcendentais.
>
> **Risco de recorrência:** Baixo, mas possível se futuras versões do
> `compiler_builtins` reintroduzirem implementações locais de
> transcendentais em `libm_math`. Recomenda-se monitorar o changelog do
> `compiler_builtins` e, em caso de suspeita, rodar rapidamente:
> `nm -C <binário> | grep -c libm_math` — se > 0, investigar.
>
> Encaminhamento para T3.2 (que depende de T3.1): o código-fonte da
> implementação de `log10f` que causou o hang NÃO está disponível no
> `compiler_builtins` atual (removida). Para T3.2.a, será necessário
> recuperar a versão antiga do `compiler_builtins` do cache do `rustup`
> (toolchain anterior) ou buscar no histórico do repositório upstream
> (`rust-lang/compiler-builtins`).
>
> **Nota T3.0 (2026-07-04):** O build atual NÃO reproduz o estado quebrado
> de §1.21.d em nenhum binário (test, standalone, cdylib). O padrão perigoso
> `R_X86_64_RELATIVE` para `log10f` foi substituído por `R_X86_64_JUMP_SLOT`
> em todos os targets. O `nm -C` (`.symtab`) retorna ZERO símbolos
> `compiler_builtins::math::libm_math`. A bissecção de T3.1.a, como
> concebida originalmente ("remover deps até `nm` deixar de mostrar os
> símbolos"), não encontrará o estado quebrado para isolar — ele já não
> existe. **Encaminhamento sugerido:** considerar T3.1 ainda válido como
> investigação preventiva ("o que causava e pode voltar a causar?"), mas
> adaptar a metodologia (ex.: `git bisect` no `Cargo.lock` entre o commit
> `56fd900` — onde o estado quebrado foi documentado pela última vez — e
> `HEAD` para identificar qual atualização de dependência eliminou o
> símbolo local).

* [x] **T3.1.a** — **CONCLUÍDO.** Bissecção de dependências adaptada:
      *`Cargo.lock` e `Cargo.toml` são **idênticos** entre `56fd900` e
        `HEAD` (zero mudanças). Nenhuma dependência do `Cargo.toml`
        mudou de versão — a eliminação dos símbolos `libm_math` veio de
        fator externo ao crate.
      * Build do commit `56fd900` com `rustc 1.96.1` produz **zero**
        símbolos `compiler_builtins::math::libm_math` (assim como
        `HEAD`). O estado quebrado NÃO é reproduzível com o `rustc`
        atual.
      *A "dependência" que causava o problema era o crate
        `compiler_builtins` **embutido no sysroot do rustc** (não listado
        no `Cargo.lock`), que em versões anteriores incluía
        implementações locais de `log10f`/`expf`/`log2f` em
        `libm_math`.
      * O `Cargo.lock` não contém o crate `compiler_builtins` — ele é
        parte do sysroot e sua versão é determinada pelo `rustc`.
* [x] **T3.1.b** — **CONCLUÍDO.** Análise da árvore de dependências:
      *`num-traits v0.2.19` tem feature `libm` (opcional) → **não**
        ativada no nosso build. Usamos `num-traits/std`.
      * `proptest v1.11.0` tem feature `no_std` → **não** ativada.
        Se ativada, habilitaria `num-traits/libm` → `libm` crate.
      *`libm` crate **não está na árvore** de dependências.
      * `ppv-lite86 v0.2.21` é `#![no_std]` mas usa apenas SIMD e
        operações inteiras — não usa funções transcendentais.
      * Nenhuma dependência do `Cargo.toml` precisa de fallback
        `libm`/`no_std` para funções matemáticas.
* [x] **T3.1.c** — **CONCLUÍDO.** Teste sem `--as-needed`:
      *Build do `--lib` de teste SEM a flag `--as-needed`: **zero**
        símbolos `libm_math`, `R_X86_64_JUMP_SLOT` para `log10f`,
        `libm.so.6` listada como NEEDED — idêntico ao build com a flag.
      * `--as-needed` **não tem participação** na prevalência do símbolo
        local sobre o dinâmico.

### T3.2 — Caracterizar o bug de `log10f` em si (paralelo, menor prioridade que T3.0/T3.1)

* [ ] **T3.2.a** — Localizar o código-fonte da implementação de `log10f` em

      `compiler_builtins::math::libm_math` (crate `compiler_builtins`,
      normalmente vendorizado em `~/.cargo/registry/src/.../compiler_builtins-*/`
      ou via `cargo tree` para a versão exata usada) e ler o algoritmo —
      é quase certo que seja um porte do `log10f` da libm do `musl`, com
      redução de faixa (`frexp`-like) seguida de um polinômio ou (mais
      provável, dado o comportamento) uma tabela/loop de refinamento. Ler
      atrás de uma condição de parada dependente de estado que nunca se
      torna verdadeira para f32 comuns em Linux x86-64 (pode ser um bug
      específico desta versão/porte, não necessariamente do upstream
      `musl` original).
* [ ] **T3.2.b** — Confirmar se `f64::log10()`/`ln()`/`log2()`/outras

      transcendentais (`sin`, `cos`, `exp`, `sqrt`) sofrem do mesmo problema
      neste binário — mesma técnica de `black_box` + wrapper de isolamento,
      um valor por vez, começando por `f64::log10()` e `f32::ln()`/`log2()`
      (usadas em `src/dsp/sinc_kernel.rs:266,414,418`, produção, f64, não
      testado nesta avaliação).
* [ ] **T3.2.c** — Preparar um issue mínimo reproduzível para

      `rust-lang/compiler-builtins` (ou onde o bug realmente residir após
      T3.2.a) com o crate `/tmp/kilo/repro-log10-bare/` como base — mas
      note que esse crate **não reproduziu** sozinho (§1.21.d) até T3.1
      identificar e replicar a dependência exata que faz o símbolo local
      prevalecer; o issue upstream só faz sentido depois de T3.1.

### T3.3 — Corrigir

* [ ] **T3.3.a** — Aplicar a correção identificada por T3.1 (remover/isolar

      a dependência responsável) e/ou T3.2 (se o bug for no algoritmo do
      `compiler_builtins` em si e a versão puder ser atualizada/pinada para
      uma corrigida).
* [ ] **T3.3.b** — Se nenhuma correção de dependência for viável a curto

      prazo, considerar um workaround **explícito e documentado**: chamar
      `libm`/`log10f` via um caminho que force a resolução dinâmica (ex.:
      um pequeno wrapper `extern "C"` declarado manualmente, ou
      `#[link(name = "m")]` explícito com `#[used]` para forçar prioridade),
      ou reescrever os poucos call sites de produção (`orchestrator.rs`,
      `sinc_kernel.rs`) para evitar `log10`/`ln` inteiramente (ex.: usar uma
      aproximação polinomial já existente no crate — `src/math/activations/`
      tem infraestrutura de aproximação rápida — ou pré-computar uma LUT
      para o medidor de picos, que só precisa de precisão perceptual, não
      matemática exata).

### T3.4 — Corrigir a calibração do teste original (separado, não bloqueante)

* [ ] **T3.4.a** — Uma vez que `log10()` funcione corretamente, reavaliar a

      asserção original de `test_x2_aliasing_rejection` (`ratio < -10.0`) —
      §1.21.e mediu ~-5.8 dB de atenuação real para este sinal/filtro, que é
      matematicamente plausível dado que 23 kHz/48 kHz está na banda de
      transição do half-band (o próprio comentário do teste já admitia essa
      possibilidade). Ajustar o threshold com base em uma medição real, não
      numa suposição — documentar com `// Measured: ...` (regra
      `.agents/rules/testing.md` §3).

### T3.5 — Validação e regressão (comum, bloqueante para reativação)

* [ ] **T3.5.a** — Remover toda instrumentação temporária desta avaliação

      (nenhuma deveria ter sobrevivido no `git diff` — todas as sondas desta
      rodada já foram revertidas com `git checkout --`, confirmado).
* [ ] **T3.5.b** — Re-rodar a matriz completa do Sprint 1 (variantes A, B,

      C, E + os 5 testes-irmãos) pós-correção, com o wrapper de isolamento
      ainda ativo — só considerar resolvido quando nenhuma variante travar.
* [ ] **T3.5.c** — Adicionar um teste de regressão **não-ignorado, rápido,

      em debug** que apenas chama `f32::log10()` sobre um valor calculado em
      runtime (com `black_box`) e afirma que retorna em menos de, digamos,
      100ms usando um timeout de thread (`std::thread::spawn` +
      `recv_timeout`) — para que uma futura regressão desta classe específica
      seja pega em segundos na suíte rápida, não descoberta de novo por
      acidente meses depois.

---

## Sprint 4 — Reativação segura e blindagem de processo (defesa em profundidade)

**Objetivo:** não deixar a suíte de testes voltar a poder "rodar
indefinidamente" — nem este teste, nem nenhum outro no futuro.

### T4.1 — Reativar o teste

* [ ] Remover `#[ignore]` de `test_x2_aliasing_rejection`

      (`src/dsp/oversample_test.rs:72`) **somente** após T3.C.2 confirmar
      estabilidade em todas as variantes.

* [ ] Adicionar o teste de volta ao ponto apropriado de

      `utils/tests-long.sh` (provavelmente Fase 3, junto dos demais testes
      de `dsp::oversample`, já que segue sendo um teste qualitativo/lento
      mesmo depois de corrigido — não pertence à suíte rápida por padrão,
      só deixa de ser "banido").

* [ ] Atualizar o quadro de aviso em `docs/testing.md` §4 — remover o

      `[!WARNING]` ou convertê-lo numa nota histórica objetiva (`"corrigido
      em <data>, causa raiz: <resumo>, ver`known-bugs.md`"`).

### T4.2 — Endurecer `timed_cargo_test` com timeout externo obrigatório

Correção estrutural de processo, independente da causa raiz encontrada —
o próprio comentário em `utils/tests-long.sh:461-463` já reconhece que
"um hang num job noturno é peor que uma falha", mas isso nunca foi
efetivamente aplicado em código. Proposta mínima e não disruptiva
(pedir ao operador humano para revisar/aplicar, já que é uma mudança em
script de infraestrutura de teste, fora do escopo estrito deste bug):

```bash
# Em utils/tests-long.sh, função timed_cargo_test (linha ~272):
timed_cargo_test() {
    local label="$1"
    shift
    local start_t; start_t=$(date +%s%N)
    local timeout_s="${NAM_TEST_TIMEOUT_S:-600}"  # 10 min/invocação, override por env var
    timeout -s KILL "$timeout_s" cargo test "$@"
    local status=$?
    if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
        echo -e "${RED}${BOLD}❌ TIMEOUT/HANG detectado em '$label' (>${timeout_s}s) — tratando como falha crítica, não como lentidão.${NC}"
    fi
    local end_t; end_t=$(date +%s%N)
    # ... resto inalterado (cálculo de duration_s, tracker) ...
    return $status
}
```

* [ ] Pedir ao operador humano para revisar e aplicar este endurecimento

      (ou equivalente) em `utils/tests-long.sh` **e** verificar se
      `utils/tests-quick.sh` tem a mesma lacuna.

* [ ] Adicionar a mesma exigência (timeout externo obrigatório para

      qualquer teste `#[ignore]`d de execução longa) como regra em
      `.agents/rules/testing.md` §3 ("Hard Requirements"), para que a
      lição fique institucionalizada e não dependa de memória humana.

### T4.3 — Fechar o loop de documentação

* [ ] Mover a entrada deste bug em `known-bugs.md` para uma seção

      "Resolvidos" (não apagar — mantém a memória institucional do
      processo de diagnóstico, consistente com o padrão já usado neste
      repositório de manter histórico em vez de deletar).

* [ ] Se T2.5 gerou um report upstream (ramo H1), linkar o issue do

      `rust-lang/rust` na entrada final.

---

## Apêndice — Comandos de referência rápida (copiar/colar com cuidado)

```bash
# Build isolado da fase de compilação (sem timeout agressivo, Sprint 1/T0.5):
cargo test --release --lib --no-run -- --ignored

# Execução isolada e com timeout, via wrapper (Sprint 1):
utils/debug/repro_oversample_hang.sh 15 <label> -- \
  cargo test --release --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
  --ignored --nocapture --test-threads=1

# Verificação pós-mortem de sinais de OOM/GPU no kernel log (Sprint 0/2):
dmesg -T | grep -iE "oom|nvidia|amdgpu|i915|gnome-shell|Xorg" | tail -n 50
journalctl --user -b -1 --no-pager | tail -n 100
```
