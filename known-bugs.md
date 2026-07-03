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

| #      | Hipótese                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Status                                           | Suporte                                                                                                                                                                                                                                                                                                                                             |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **H1** | Bug de compilador/vetorizador (miscompilação) específico do pipeline `opt-level=3 + lto=fat + codegen-units=1 + target-cpu=x86-64-v3`, latente desde a criação do teste (§1.1) e nunca antes exercitado em `--release` porque o teste é `#[ignore]`d desde o dia 1 — só foi "descoberto" quando alguém finalmente rodou `--ignored --release` por vontade própria durante uma auditoria.                                                                                                                                                          | **Hipótese líder**, não confirmada dinamicamente | Código do hot path idêntico desde a criação (§1.5); CPU-spin sem syscalls é a assinatura clássica de um laço vetorizado com contagem de iterações corrompida pelo compilador; aritmética modular com `n` não potência de 2 (`12`, `25`) dentro de laços é um padrão historicamente associado a bugs de vetorização em LLVM sob otimização agressiva |
| **H2** | Artefato ambiental (cache de build sujo, contenção de recursos do host, thermal throttling, ou mesmo uma falha coincidente e não relacionada do sistema gráfico) — o relato de reset do GNOME (§1.3) se encaixa melhor numa narrativa de exaustão de recursos do sistema do que num loop de ponto flutuante de 256 elementos.                                                                                                                                                                                                                     | Aberta, plausível                                | Nenhuma reprodução foi feita a partir de `target/` limpo (§5.4); nenhum container/cgroup foi usado em nenhuma tentativa (§1.3, §1.4); um reset de GNOME é mais consistente com OOM/GPU driver crash do que com um hang de CPU single-thread contido                                                                                                 |
| **H3** | Bug algorítmico real ainda não localizado por análise estática (ex.: um caminho não considerado nas provas de §2/§4, ou uma interação entre `X2Stage` #1 e #2 no caminho `X4` — mas o teste em questão só usa `X2`, então esta hipótese teria que explicar por que só se manifesta em `test_x2_aliasing_rejection` e não nos outros testes `X2` do mesmo arquivo, como `test_x2_upsample_dc`, `test_x2_roundtrip_dc`, `test_back_to_back_roundtrips_x2`, que usam a mesma engine com entradas de tamanho semelhante e **nunca reportaram hang**). | Aberta, mas com ônus de prova elevado            | Nenhuma linha de código exclusiva deste teste versus os demais testes `X2` do mesmo arquivo, exceto os *valores* de entrada (128 amostras @ 23 kHz/48 kHz senoidal vs. DC/degraus/ruído nos outros) — se H3 for verdadeira, a causa tem que ser sensível ao *conteúdo* do sinal, não à estrutura do código                                          |
| **H4** | UB de `AlignedVec::drop` (memória)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | **Refutada** (§3)                                | Prova estática de invariante `len == capacity`; silêncio do ASan                                                                                                                                                                                                                                                                                    |
| **H5** | Acesso fora dos limites via `get_unchecked`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | **Refutada** (§4)                                | Prova estática construtiva; silêncio do ASan                                                                                                                                                                                                                                                                                                        |

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
