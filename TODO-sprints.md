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

* [ ] Confirmar que `target/` foi removido pelo operador humano
      (`ls target` deve falhar ou o diretório deve estar vazio/recriado do
      zero) antes de iniciar qualquer build deste roteiro.
* [ ] Registrar em `known-bugs.md` §1 (nova entrada de linha do tempo) o
      timestamp exato da limpeza e o `git rev-parse HEAD` no momento —
      garante que qualquer resultado futuro seja rastreável a um estado de
      árvore de trabalho conhecido.
* [ ] Verificar espaço em disco livre suficiente para um build fat-LTO
      completo (`df -h .`) — builds `codegen-units=1` + `lto=fat` tendem a
      usar mais RAM/disco de linker do que o normal; confirmar ao menos
      ~10 GB livres antes de começar.

### T0.2 — Reduzir a superfície de risco da sessão gráfica

* [ ] Fechar aplicações não essenciais antes de qualquer tentativa de
      reprodução (o host tem 16 GB RAM / 16 núcleos; confirmar
      `free -h` mostra memória livre confortável — hoje ~11 GB disponíveis).
* [ ] Preferir executar os comandos de reprodução (Sprints 1–2) a partir de
      um TTY texto puro (`Ctrl+Alt+F3` → login) ou de uma sessão SSH para
      `localhost`, **não** do terminal dentro da própria sessão GNOME sendo
      investigada — se algo desestabilizar o processo, a sessão de
      diagnóstico não cai junto com o alvo.
* [ ] Antes de cada tentativa, rodar `dmesg -T | tail -n 5` e anotar o
      timestamp; depois de cada tentativa (travou ou não), rodar
      `dmesg -T | tail -n 40` de novo e comparar — é a única forma barata
      de captar sinais de OOM-killer / reset de driver GPU que comprovem ou
      refutem a causalidade do relato de reset de desktop (`known-bugs.md`
      §1.3, H2).

### T0.3 — Escolher e validar o mecanismo de isolamento de recursos

Máquina tem `systemd-run` e `docker` disponíveis (não tem `podman`). Ordem
de preferência (mais simples → mais pesado):

* [ ] **Opção A (recomendada, sem overhead de imagem):** `systemd-run --user --scope` com cgroup v2 limitando memória, CPU e nº de tasks:

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

* [ ] **Validar o isolamento escolhido com um comando inofensivo antes de
      usá-lo no alvo real** — ex.: `systemd-run --user --scope --collect
      -p MemoryMax=1G -p CPUQuota=100% -- sleep 3; echo status=$?` deve
      retornar rapidamente sem erro. Não pule esta validação.

### T0.4 — Script de wrapper de segurança (entregável desta tarefa)

* [ ] Criar `utils/debug/repro_oversample_hang.sh` (script novo, não
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

### T0.5 — Separar fase de build da fase de execução (evita confundir "compilando" com "travado")

* [ ] Para cada variante do Sprint 1, primeiro rodar **sem** timeout
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

```bash
utils/debug/repro_oversample_hang.sh 15 varA-stable-debug -- \
  cargo test --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
```

* [ ] Executar. Registrar resultado (passou / falhou por asserção / hang)
      em `known-bugs.md`.

### T1.2 — Variante B: `stable` + `--release` (reproduz exatamente o comando original de §1.2, com `target/` limpo)

```bash
utils/debug/repro_oversample_hang.sh 15 varB-stable-release -- \
  cargo test --release --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
```

* [ ] Executar. Este é o experimento de controle mais importante: se **não**
      reproduzir aqui, com `target/` limpo, isso aponta fortemente para
      H2 (artefato ambiental / cache sujo) em vez de H1 (bug de compilador)
      como causa da observação original.

### T1.3 — Variante C: `nightly` + `--release`, sem ASan (isola a troca de canal do diagnóstico T1.2)

```bash
rustup run nightly cargo test --release --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
# (envelopar com o wrapper T0.4 exatamente como acima, label varC-nightly-release)
```

* [ ] Executar. Compara diretamente com T1.2 trocando **só** o canal.

### T1.4 — Variante D: `nightly` + `--release` + ASan, com o `lto=fat` real do projeto (repete T1.2 do diagnóstico anterior, mas sem desligar o LTO)

```bash
RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" \
  rustup run nightly cargo test --release --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
```

* [ ] Executar (label `varD-nightly-release-asan-lto-fat`). Diferente da
      rodada original de `known-bugs.md` §1.4 apenas por manter
      `lto = "fat"` (não setar `CARGO_PROFILE_RELEASE_LTO=off`) — isola a
      variável de LTO.

### T1.5 — Variante E: réplica exata do diagnóstico anterior (LTO off) — controle de continuidade

```bash
RUSTFLAGS="-Zsanitizer=address -Ctarget-cpu=x86-64-v3" CARGO_PROFILE_RELEASE_LTO="off" \
  rustup run nightly cargo test --release --lib -- \
  "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1
```

* [ ] Executar (label `varE-nightly-release-asan-lto-off`). Deve reproduzir
      o resultado já documentado em `known-bugs.md` §1.4 (hang, CPU-spin,
      ASan silencioso) — se **não** reproduzir com `target/` limpo, isso é
      em si um dado importante (dependência de estado de cache).

### T1.6 — Testes-irmãos sob as mesmas condições (testa H3)

Para a variante que efetivamente reproduzir o hang (qualquer uma de
T1.1–T1.5), repetir com os quatro testes não-ignorados do mesmo arquivo,
um por vez, mesmo timeout:

* [ ] `test_x2_upsample_dc`
* [ ] `test_x2_roundtrip_dc`
* [ ] `test_back_to_back_roundtrips_x2`
* [ ] `test_x4_upsample_dc`
* [ ] `test_x4_roundtrip_dc`

Se **nenhum** destes travar sob a mesma combinação de flags que trava
`test_x2_aliasing_rejection`, a causa é sensível ao **conteúdo do sinal de
entrada** (senoide a 23 kHz/128 amostras) e não apenas à estrutura de
código/build — isso restringe fortemente o espaço de busca do Sprint 2
(foco na interação específica dos valores, não do algoritmo em si).

### T1.7 — Extração mínima standalone (opcional, mas recomendado se qualquer variante travar)

* [ ] Criar um crate temporário em `/tmp/kilo/repro-oversample/` contendo
      apenas: `oversample.rs` (renomeado `lib.rs` + o módulo de teste),
      `aligned.rs` (dependência mínima), e um `Cargo.toml` copiando
      exatamente os `[profile.release]` e `[build] rustflags` relevantes do
      projeto original. Objetivo: (a) eliminar a variável de confusão
      "todo o resto do crate `--lib` também compila e linka" (`known-bugs.md`
      §5.5); (b) reduzir o tempo de build de fat-LTO de minutos para
      segundos, permitindo iterar rapidamente no Sprint 2.
* [ ] Confirmar que o hang **ainda reproduz** no crate mínimo antes de
      investir tempo em instrumentação sobre ele — se não reproduzir aqui,
      a causa depende de algo do crate completo (outro dado valioso, volta
      para análise de estáticos globais linkados).

### T1.8 — Consolidar resultados

* [ ] Preencher esta tabela (copiar para `known-bugs.md` §1 como nova
      entrada de linha do tempo, com timestamp real):

  | Variante | Toolchain | Perfil  | LTO | ASan | Resultado | Log                            |
  | -------- | --------- | ------- | --- | ---- | --------- | ------------------------------ |
  | A        | stable    | debug   | —   | não  |           | `target/debug-logs/varA-*.log` |
  | B        | stable    | release | fat | não  |           |                                |
  | C        | nightly   | release | fat | não  |           |                                |
  | D        | nightly   | release | fat | sim  |           |                                |
  | E        | nightly   | release | off | sim  |           |                                |

---

## Sprint 2 — Instrumentação profunda (só se Sprint 1 confirmar reprodução em ao menos uma variante)

**Objetivo:** identificar o PC/linha exata onde a execução gira, já que o
ASan não indicou violação de memória (`known-bugs.md` §1.4) e a análise
estática (`known-bugs.md` §2) não encontrou nenhum laço matematicamente
capaz de não terminar — ou seja, precisamos de evidência dinâmica direta,
não mais inferência.

Use a variante mais simples que reproduziu no Sprint 1 (idealmente T1.7,
o crate mínimo, pela velocidade de iteração) para tudo abaixo.

### T2.1 — Build com símbolos preservados

* [ ] Rebuild da variante reprodutora com `strip = false` e `debug = true`
      no perfil usado (via override local, não editar `Cargo.toml` do
      projeto principal ainda) — sem símbolos, `perf`/`gdb` são inúteis
      (como já ocorreu em `known-bugs.md` §1.4 item 4).

### T2.2 — Backtrace via coredump (mais seguro que anexar a um processo vivo)

* [ ] Rodar a reprodução com `ulimit -c unlimited` e enviar `SIGABRT` em
      vez de `SIGKILL` no fim do timeout, para gerar um coredump analisável
      post-mortem (evita a corrida de "anexar o `gdb` a tempo" antes do
      kill):

  ```bash
  utils/debug/repro_oversample_hang.sh — usar `timeout -s ABRT 15 ...`
  # dentro do systemd-run scope, com `ulimit -c unlimited` setado antes.
  ```

* [ ] Analisar o coredump: `gdb -batch -ex "bt full" -ex "info registers"
      -ex "disassemble" -q <binário> <corefile>`. O objetivo é a linha de
      Rust (ou, na ausência de símbolos de linha, o menor nível: qual
      função/loop no assembly) onde o `pc` está girando.

### T2.3 — `rr` record/replay (se disponível; instalar é opcional, não obrigatório)

* [ ] Se `rr` estiver disponível (`which rr`), gravar a execução
      (`rr record <binário> ...`) dentro do mesmo isolamento de recursos —
      permite replay determinístico e "reverse-continue" até o início do
      loop suspeito, bem mais poderoso que um coredump único. Pular esta
      tarefa se `rr` não estiver instalado e não for trivial instalar
      (não é bloqueante para o restante do sprint).

### T2.4 — Canário de iteração (kill-switch determinístico, sem depender de ferramentas externas)

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

---

## Sprint 3 — Correção e validação (conteúdo depende do achado do Sprint 2)

O caminho exato aqui se ramifica pela causa raiz confirmada. Marcar apenas
o ramo que se aplicar; os outros ficam como referência para o futuro.

### Ramo H1 confirmado (bug de compilador/vetorização)

* [ ] **T3.H1.a** — Aplicar mitigação mínima e localizada: `#[inline(never)]`
      na função afetada (evita que o fat-LTO a funda de forma patológica
      com o call-site do teste) **e/ou** reescrever o módulo problemático
      para usar uma potência de 2 como tamanho de ring buffer (ex.: `16`
      em vez de `12`, `32` em vez de `25`, com os índices não usados
      deixados a zero) — troca `%` genérico por `& (n - 1)` (bitmask),
      eliminando por completo a operação de módulo por constante
      não-potência-de-2 que é o candidato mais concreto para o bug de
      vetorização. **Atenção:** isto muda o layout de `up_ring`/`down_ring`
      — reexecutar `test_halfband_filter_coefficients` e os testes de DC
      gain para confirmar que a mudança de tamanho de buffer não altera o
      resultado matemático (deve ser puramente uma otimização de
      indexação, os coeficientes e a lógica de convolução continuam
      idênticos).
* [ ] **T3.H1.b** — Se a mitigação de (a) não resolver, considerar fixar
      `rust-toolchain.toml` para uma versão de `stable` anterior/posterior
      conhecida como não afetada (requer primeiro identificar, via Sprint 1,
      em qual(is) canal(is)/versões o bug se manifesta).
* [ ] **T3.H1.c** — Preparar e (se aplicável) submeter um issue mínimo
      reproduzível ao repositório do `rust-lang/rust` — anexar o crate de
      T1.7 reduzido.

### Ramo H2 confirmado (artefato ambiental / cache)

* [ ] **T3.H2.a** — Nenhuma mudança de código necessária. Documentar em
      `known-bugs.md` que o hang é dependente de estado de build sujo e
      **não** reproduz em clean room — rebaixar a severidade de
      "system-safety" para "ambiente conhecido, mitigado por clean build".
* [ ] **T3.H2.b** — Adicionar uma nota preventiva em `docs/testing.md` ou
      `AGENTS.md` recomendando `cargo clean` periódico especificamente para
      builds com `lto=fat`+`codegen-units=1`, se a investigação apontar
      para corrupção específica de cache incremental/LTO (ainda que
      `incremental` só esteja ligado no perfil `dev`, vale confirmar se o
      cache de LTO do linker (`.rlib`/`liblto` intermediários) tem alguma
      participação).

### Ramo H3 confirmado (bug algorítmico real, localizado pelo canário/gdb do Sprint 2)

* [ ] **T3.H3.a** — Corrigir o bug exatamente no ponto identificado pelo
      Sprint 2 (não aplicar uma correção genérica/especulativa — o Sprint 2
      deve ter apontado a linha exata).
* [ ] **T3.H3.b** — Adicionar um teste de regressão específico e
      **não-ignorado** cobrindo o caso de borda exato que causava o loop
      (ex.: se for sensível ao conteúdo do sinal, um teste dedicado com
      esse sinal específico, rodando em debug para ser rápido e sempre
      ativo).

### Comum a todos os ramos

* [ ] **T3.C.1** — Remover completamente qualquer instrumentação temporária
      do Sprint 2 (canário de iteração, `eprintln!`, etc.) do código de
      produção — `git diff` deve mostrar apenas a correção real antes do
      commit.
* [ ] **T3.C.2** — Re-rodar a matriz completa do Sprint 1 (todas as 5
      variantes + os 5 testes-irmãos) pós-correção, ainda com o wrapper de
      isolamento/timeout ativo — a correção só é considerada validada
      quando **todas** as variantes que antes travavam agora terminam
      dentro do orçamento esperado (a asserção da linha 120-123 do teste
      pode passar ou falhar por motivos de precisão numérica — isso é
      aceitável e tratável separadamente; o que não é aceitável é qualquer
      novo hang).

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
