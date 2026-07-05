<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# report-upstream.md — Plano de reporte upstream: interposição de símbolo entre `compiler-builtins` e `libm`

## 1. Resumo executivo

O bug corrigido em `docs/postmortem-libm-symbol-interposition.md` — uma
chamada de runtime a `f32::log10()` entrando em loop infinito por
interposição de símbolo ELF entre uma cópia local de `log10f`/`atan2f`/
`acosf` (originada em `compiler-builtins`/`libm`) e a `libm.so.6`
dinâmica — **não é um bug isolado do nam-rs**. É a manifestação de uma
classe de problema **já conhecida e ativamente discutida** no repositório
`rust-lang/rust`, com pelo menos uma issue aberta, um PR que reintroduziu o
comportamento, e participação recente de mantenedores seniores
(`@tgross35`, `@Amanieu`). O que a nossa investigação tem de genuinamente
**novo e valioso** para contribuir upstream:

1. **Um sintoma mais grave, ainda não relatado**: as issues existentes
   descrevem "a função errada é chamada, com resultado sutilmente
   diferente" (uma mudança de comportamento). A nossa reprodução mostra
   que, sob uma combinação específica de flags de link (`-Wl,-z,now`),
   isso pode se tornar um **loop infinito de dois `jmp`** — sem
   computação, sem syscalls, CPU a 100% para sempre. Nenhuma issue
   encontrada até agora menciona um hang.
2. **Símbolos fora do conjunto documentado como intencional**: as issues
   e o PR encontrados falam de um conjunto curado e explicitamente restrito
   a funções de "resultado exato" (`ceil`, `floor`, `sqrt`, `fmod`, etc.).
   O nosso caso envolve `log10f`, `atan2f`, `acosf` — funções
   transcendentais, inexatas por natureza, que (até onde conseguimos
   confirmar lendo o PR original) não deveriam estar nesse conjunto.
3. **Confirmado em um executável PIE, não só staticlib/dylib**: os
   relatos existentes são sobre linkagem de C com uma staticlib Rust. A
   mitigação em desenvolvimento (`-Zdefault-visibility=hidden`) é
   documentada como afetando **apenas shared objects**, não executáveis —
   ou seja, mesmo que essa flag avance, pode não cobrir o nosso caso (um
   `bin`/executável de teste ou standalone).
4. **Evidência de runtime, não só estática**: confirmamos via GDB, lendo o
   valor real escrito no slot de GOT em um processo vivo, exatamente para
   onde a chamada resolve — uma técnica que nenhum dos relatos existentes
   parece ter usado (eles inferem via `nm -uD`, que mostra o *pedido* de
   importação, não o valor efetivamente resolvido).

O plano abaixo prioriza **não duplicar** as issues existentes, reunir a
evidência que falta para uma contribuição definitiva, e então contribuir a
issue já aberta (ou abrir uma nova, cross-referenciada, se os mantenedores
concordarem que o sintoma difere o suficiente) com um relatório da
qualidade que o processo do `rust-lang/rust` espera.

---

## 2. Descoberta central: isto já é um bug rastreado upstream

Pesquisa (2026-07-04) encontrou a seguinte cadeia de trabalho relacionado,
em ordem cronológica:

### 2.1 — A causa histórica: `compiler-builtins` reintroduzindo símbolos "fracos"

- **[`rust-lang/compiler-builtins` commit `018616e`](https://github.com/rust-lang/compiler-builtins/commit/018616e78be0b6e213018c16b430d14ec1083cb)**
  ("Always have math functions but with `weak` linking attribute if we
  can") — tornou **todos** os símbolos de math disponíveis com linkagem
  `weak` em todas as plataformas que suportam. Causou regressões
  inesperadas: as rotinas do `compiler-builtins` (menos precisas/mais
  lentas) passaram a ser escolhidas no lugar da `libm` do sistema.
- **Commit `0fab77e`** ("Don't include `math` for `unix` e `wasi`
  targets") — revertiu isso para a maioria das plataformas Unix,
  adicionando o `#[cfg(not(any(..., unix, ...)))]` que encontramos ao ler
  o código-fonte de `compiler-builtins` durante esta investigação.
- **[`rust-lang/compiler-builtins#763`](https://github.com/rust-lang/compiler-builtins/pull/763)**
  ("Make a subset of `libm` symbols weakly available on all platforms",
  autor `@tgross35`, aprovado por `@Amanieu`, mesclado em 25/02/2025) —
  reintroduziu a disponibilidade fraca **apenas** para um conjunto curado
  e explicitamente restrito de funções que "produzem resultados exatos"
  (verificado por teste exaustivo): `cbrt`, `ceil`, `copysign`, `fabs`,
  `fdim`, `floor`, `fma`, `fmax`, `fmaximum`, `fmin`, `fminimum`, `fmod`,
  `rint`, `round`, `roundeven`, `sqrt`, `trunc` (e variantes f32/f16/f128).
  **`log10f`, `atan2f` e `acosf` não estão nesta lista** — são funções
  transcendentais, não "de resultado exato", e o PR é explícito sobre essa
  distinção ("Once more routines meet these criteria, we can move them...").

### 2.2 — As issues abertas por causa disso

- **[`rust-lang/rust#139487`](https://github.com/rust-lang/rust/issues/139487)**
  ("Different NaN behavior with various float functions on beta-vs-stable",
  `@alexcrichton`, abril/2025, **fechada como discussão/comportamento
  esperado**) — `floor()` de um NaN específico mudou de resultado entre
  stable e beta, bisectado até o PR #763. Fechada porque a mudança de
  comportamento foi considerada intencional/aceitável.

- **[`rust-lang/rust#142119`](https://github.com/rust-lang/rust/issues/142119)**
  ("Linking a Rust staticlib unexpectedly changes C math functions from
  libm to bundled ones from compiler-builtins", `@durin42`, 06/06/2025,
  **ainda ABERTA**) — uma staticlib Rust, linkada com um programa C que
  também usa `-lm`, faz `ceilf` (uma função C, não Rust!) ser satisfeita
  pela implementação do `compiler-builtins` em vez da `libm.so.6` real.
  Regressão confirmada entre `rustc 1.86.0` (funciona) e `1.87.0` (quebra).
  Labels atuais: `A-compiler-builtins`, `A-linkage`, `C-bug`, `E-hard`,
  `E-help-wanted`, `P-medium`, `T-compiler`, `T-libs`,
  `regression-from-stable-to-stable`. **Sem assignee.** Comentários
  relevantes:

  - O relator já havia notado o mesmo problema afetando o **Chromium**
    (<https://issues.chromium.org/issues/419258012#comment5>), com mudança
    de comportamento observável em produção.

  - Um reprodutor **em Rust puro** (sem staticlib/C) foi postado nos
    comentários:

    ```rust
    fn main() {
        let x: f32 = std::hint::black_box(1.1);
        dbg!(x.ceil());
    }
    ```

    confirmado por `@Noratrieb`/outros: em `rustc +1.84.1` mostra
    `U ceilf@GLIBC_2.2.5` (importação dinâmica correta); num nightly mais
    recente, não mostra — a chamada foi capturada pela cópia local.

  - `@Amanieu` (mantenedor sênior, `compiler-builtins` e `rustc`) comentou
    reconsiderando sua própria expectativa sobre a regra de precedência
    entre símbolo fraco estático e símbolo forte dinâmico: "So I
    previously thought that statically linked weak symbols always
    override strong dynamic symbols, but this may not actually be the
    case."

  - `@Noratrieb` respondeu com um [gist demonstrando exatamente o
    oposto](https://gist.github.com/Noratrieb/a58000d43527225ed5af0032890e00a5)
    — o símbolo fraco estático **vence** sobre o símbolo forte dinâmico na
    prática, confirmando o mecanismo. Isto é consistente, de forma
    independente, com o que encontramos ao ler o slot de GOT em runtime.

  - `@tgross35` perguntou diretamente: "is it intended that Rust programs
    now always call the compiler-builtins version?" — sem resposta
    conclusiva registrada até a data desta pesquisa.

### 2.3 — A mitigação em desenvolvimento (parcial, não estabilizada)

- **[`rust-lang/compiler-team#782`](https://github.com/rust-lang/compiler-team/issues/782)**
  e **[tracking issue `rust-lang/rust#131090`](https://github.com/rust-lang/rust/issues/131090)**
  — a flag `-Zdefault-visibility=hidden|protected|interposable`
  (unstable), que permitiria evitar exatamente esta classe de
  interposição. **A documentação oficial afirma explicitamente que esta
  flag "only affects building of shared objects and should have no effect
  on executables"** — ou seja, mesmo estabilizada, pode não cobrir o caso
  de um `bin`/executável (como o nosso).
- **[`rust-lang/rust#137736`](https://github.com/rust-lang/rust/pull/137736)**
  ("Don't attempt to export compiler-builtins symbols from rust dylibs",
  mesclado) — trabalho relacionado, mas específico de `dylib`.
- **[`rust-lang/rust#123427`](https://github.com/rust-lang/rust/issues/123427)**
  — mostra que a mecânica de visibilidade hidden/default para símbolos de
  intrínsecos/math ainda tem arestas não resolvidas mesmo internamente ao
  compilador.

### 2.4 — Conclusão desta seção

**Não devemos abrir uma issue nova do zero sem primeiro cross-referenciar
e, idealmente, comentar em `rust-lang/rust#142119`.** O mecanismo é o
mesmo; os mantenedores certos já estão engajados ali. A decisão de abrir
uma issue nova separada (por causa do sintoma mais grave — hang em vez de
resposta errada) deve ser deles, oferecida como sugestão no nosso
comentário, não decidida unilateralmente por nós.

---

## 3. O que é genuinamente novo na nossa investigação

Resumo do que temos e as issues existentes não têm — este é o valor real
que podemos entregar:

| Achado                                                                                                      | Já documentado upstream?                                                                                                                  |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `compiler-builtins` pode substituir chamadas de `libm` do sistema por sua própria implementação             | ✅ Sim (`#142119`, `#139487`)                                                                                                             |
| Confirmado com `ceilf`/`floor` (conjunto "resultado exato" do PR #763)                                      | ✅ Sim                                                                                                                                    |
| Confirmado com `log10f`/`atan2f`/`acosf` (funções transcendentais, fora do conjunto documentado do PR #763) | ❌ Não encontrado                                                                                                                         |
| Sintoma: resposta numericamente diferente                                                                   | ✅ Sim                                                                                                                                    |
| Sintoma: **loop infinito / hang** (sob `-Wl,-z,now`)                                                        | ❌ Não encontrado                                                                                                                         |
| Reprodução em programa Rust puro, sem staticlib/FFI C                                                       | ✅ Sim (comentário em `#142119`)                                                                                                          |
| Reprodução confirmada em **executável PIE de produção** (não apenas um `fn main()` de teste)                | ⚠️ Parcial — o reprodutor mínimo de `#142119` é um `fn main()`, mas ninguém documentou a leitura de GOT em runtime num binário real maior |
| Confirmação via leitura do valor real do slot de GOT em processo vivo (GDB)                                 | ❌ Não encontrado — os relatos existentes usam apenas `nm -uD` (estático)                                                                 |
| Mecanismo `trampolim → PLT → GOT → trampolim` documentado endereço por endereço                             | ❌ Não encontrado                                                                                                                         |

---

## 4. Partes interessadas (stakeholders)

| Parte                             | Papel                                                                                                                                                                                               | Por quê                                                                                                                  |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **`rust-lang/rust`**              | Repositório onde a issue-mãe (`#142119`) já vive; convenção observada é rastrear regressões de linkagem/`compiler-builtins` aqui, mesmo quando a causa raiz mora em outro repo                      | Ponto de contato primário                                                                                                |
| **`rust-lang/compiler-builtins`** | Biblioteca fonte do PR #763 que reintroduziu a disponibilidade fraca; dono da lógica de `full_availability`/`partial_availability`/`intrinsics!`                                                    | Ponto de contato técnico secundário, para quem entende a mecânica exata da macro `intrinsics!` e da atribuição de `weak` |
| **`rust-lang/libm`**              | Fonte vendorizada dos algoritmos matemáticos usados por `compiler-builtins::math` (não é o problema em si — o algoritmo está correto — mas é onde a curadoria "resultado exato" de #763 se origina) | Contato terciário, caso a investigação revele que o conjunto de funções disponibilizadas mudou de forma não documentada  |
| **`@tgross35`**                   | Autor do PR #763, mantenedor ativo de `compiler-builtins`, já participando de `#142119` e perguntando diretamente "é intencional?"                                                                  | Melhor pessoa para uma menção direta (`@`) num comentário — já está engajado no tópico                                   |
| **`@Amanieu`**                    | Mantenedor sênior, revisor do PR #763, comentando ativamente em `#142119` sobre a semântica de símbolo fraco/forte                                                                                  | Segunda melhor menção — pode esclarecer definitivamente a regra de precedência                                           |
| **`@durin42`**                    | Relator original de `#142119`                                                                                                                                                                       | Não precisa ser mencionado diretamente, mas o crédito/contexto do relato original deve ser preservado ao comentar        |
| **Projeto Chromium**              | Terceiro consumidor já afetado, citado em `#142119`                                                                                                                                                 | Não é alvo de contato nosso, mas é evidência de que o impacto é real e cross-projeto, útil citar                         |
| **nam-rs (nós)**                  | Terceiro relator independente, com uma reprodução em executável PIE de produção + evidência de hang + leitura de GOT em runtime                                                                     | Nosso papel: agregar uma dimensão de severidade e um ambiente de reprodução que os relatos existentes não cobrem         |

---

## 5. O que NÃO temos certeza suficiente para afirmar (importante ser honesto)

- **Não identificamos qual dependência exata do nam-rs (se alguma)
  "aciona" isto.** Dado o reprodutor de `#142119` (`black_box(1.1).ceil()`,
  um `fn main()` completamente vazio, sem dependências), é bem provável
  que **nenhuma dependência específica seja necessária** — isto pode
  acontecer com qualquer binário Rust simples nesta faixa de versões do
  `rustc`. Isso precisa ser confirmado (ver §6, passo 1) antes de afirmar
  qualquer coisa sobre "nossa árvore de dependências" no relatório final.
- **Não confirmamos se o símbolo local é `WEAK` ou `GLOBAL/STRONG`** com o
  rigor que este tópico exige. Nossa investigação original leu `nm -D`/
  `nm -C` e viu `T` (símbolo de texto definido), mas não registramos
  explicitamente a coluna de *binding* (`readelf --syms` distingue
  `GLOBAL` de `WEAK`; `nm` sozinho pode não deixar isso óbvio dependendo
  da flag usada). Isto é uma lacuna real da investigação anterior e a
  primeira coisa a fechar antes de escrever qualquer relatório (ver §6,
  passo 3) — é exatamente o ponto de incerteza que o próprio `@Amanieu`
  levantou em `#142119`.
- **Não confirmamos se `log10f`/`atan2f`/`acosf` já fazem parte,
  atualmente, de alguma lista "weak" mais recente e mais ampla do que a
  do PR #763** (fevereiro/2025) — dado que nosso `rustc` (1.96.1) é muito
  mais novo, é plausível que a lista tenha crescido desde então e que
  estas três funções já estejam formalmente incluídas (não seria mais um
  "bug" nesse caso, e sim uma característica cujo *efeito colateral*, o
  hang sob `-z,now`, ainda seria novo e reportável). Precisa ser verificado
  lendo o `compiler-builtins` na versão exata vendorizada pelo nosso
  toolchain (ver §6, passo 2).
- **Não confirmamos se `-Wl,-z,now` é condição necessária para o hang**,
  apenas que estava presente em todas as nossas reproduções (é um flag
  permanente do projeto, nunca testamos sem ele). Isto é importante: se o
  hang só ocorre sob `BIND_NOW`, isso é uma informação valiosa e específica
  a oferecer aos mantenedores (ver §6, passo 4).

---

## 6. Plano de investigação a executar antes de publicar qualquer coisa

Ordem recomendada — cada passo é rápido e decisivo; execute nesta ordem
porque os primeiros passos podem tornar os últimos desnecessários.

### Passo 1 — O reprodutor mínimo do `#142119` reproduz o HANG, sozinho, sem nam-rs?

```bash
mkdir -p /tmp/upstream-repro && cd /tmp/upstream-repro
cat > main.rs << 'EOF'
fn main() {
    let x: f32 = std::hint::black_box(0.51576114);
    let y: f32 = std::hint::black_box(1.0);
    println!("{}", (x / y).log10());
}
EOF
rustc -O -Clink-arg=-Wl,-z,now main.rs -o repro
timeout -s KILL 10 ./repro; echo "exit=$?"
```

Se isto travar **sozinho, sem nenhuma dependência do nam-rs**, é a prova
mais forte e mais simples possível — decide imediatamente que isto é
"qualquer binário Rust nesta versão de toolchain", não algo ligado à nossa
árvore de dependências. **Isto muda a prioridade/urgência do relatório
upstream (afeta potencialmente qualquer usuário de Rust, não um caso de
borda).** Rode isolado, com o wrapper de segurança já existente no
repositório (`utils/debug/repro_oversample_hang.sh`) se preferir isolamento
de recursos, embora para um binário deste tamanho um `timeout` simples já
seja suficientemente seguro.

### Passo 2 — A versão vendorizada de `compiler-builtins`/`libm` já inclui estas funções na lista "weak" documentada?

```bash
find ~/.rustup/toolchains/*/lib/rustlib/src/rust/library/compiler-builtins \
  -name mod.rs -path '*compiler-builtins/src/math*' \
  -exec grep -n "log10f\|atan2f\|acosf" {} +
```

Ler o resultado à luz do `#[cfg(...)]` que envolve o bloco onde essas
funções aparecem (`full_availability` = sempre disponível, sem gate;
`partial_availability` = gated `not(unix)`, ou seja, não deveria compilar
aqui). Documentar a **versão exata** de `compiler-builtins` (via
`cargo tree -p compiler_builtins` não funciona diretamente pois é uma dep
do sysroot, não do `Cargo.lock` — usar
`rustc --print sysroot` + procurar o `Cargo.lock` interno do próprio
toolchain, ou `rustc -Zunpretty=...` não é necessário; o caminho do
`rust-src` já expõe a versão no próprio checkout).

### Passo 3 — `WEAK` ou `GLOBAL`? Resolver a incerteza que o próprio Amanieu levantou

```bash
readelf --syms --wide <binário> | grep -E "\blog10f\b|\batan2f\b|\bacosf\b"
```

Olhar a coluna `Bind` (`GLOBAL`, `WEAK`, `LOCAL`). Se `WEAK`: reforça a
tese de que o mecanismo é "peso fraco estático vence peso forte dinâmico
sob resolução eager" (`-z,now`), exatamente o que o gist do Noratrieb
demonstrou — nosso caso seria uma confirmação direta e independente disso,
só que com uma consequência mais grave (hang). Se `GLOBAL`/strong:
seria um dado **novo e mais alarmante** — significaria que nem a
atribuição `weak` documentada está sendo respeitada no nosso pipeline de
build (fat LTO + `codegen-units=1` + `lld`), o que seria, por si só, uma
observação extra a oferecer aos mantenedores.

### Passo 4 — `-Wl,-z,now` é necessário para o hang?

Reproduzir o Passo 1 **sem** `-Clink-arg=-Wl,-z,now` (binding lazy padrão)
e comparar. Se o hang desaparece (ou se transforma em "resposta errada mas
sem travar", como nas issues existentes), isso confirma que **BIND_NOW é
o ingrediente que transforma uma "resposta errada" documentada em um
"loop infinito" novo** — a contribuição central do nosso relatório.

### Passo 5 — Bisecção de versão do `rustc` (padrão que os próprios mantenedores usam)

Instalar `cargo-bisect-rustc` (ferramenta oficial do projeto Rust para
exatamente este propósito) e localizar precisamente em qual nightly o
comportamento (ou ao menos a presença dos três símbolos) apareceu, para
oferecer um intervalo "funciona em X / quebra em Y" no mesmo formato que
`durin42` usou em `#142119` (`1.86.0` funciona / `1.87.0` quebra):

```bash
cargo install cargo-bisect-rustc
cargo bisect-rustc --start=2025-02-20 --end=2025-03-05 \
  --script=/tmp/upstream-repro/check.sh
```

(`check.sh` deve rodar o binário do Passo 1 sob um `timeout` e retornar
código de saída distinto para "travou" vs. "não travou" vs. "resultado
numericamente diferente de libm".) Dado que o PR #763 foi mesclado em
25/02/2025, a janela acima é o ponto de partida natural — mas, como nosso
caso envolve símbolos que talvez tenham sido adicionados **depois** de # 763,
pode ser necessário expandir a janela para meses mais recentes; o
Passo 2 deve informar isso primeiro.

### Passo 6 — Isolar se algum flag específico do nam-rs (fora `-z,now`) contribui

Repetir o Passo 1 adicionando, um por vez, os demais flags de
`.cargo/config.toml` (`--gc-sections`, `--as-needed`, `-u,clap_entry`,
`-Ctarget-cpu=x86-64-v3`) para confirmar que nenhum deles é
coincidentemente necessário além de `-z,now`. Espera-se que nenhum deles
importe — mas confirmar é barato e fortalece o relatório.

---

## 7. Rascunho do relatório (comentário em `#142119`, adaptar após executar §6)

> Preparado em inglês, no registro técnico-objetivo que o `rust-lang/rust`
> usa. **Não publicar sem primeiro rodar o plano de investigação da §6** —
> os campos marcados `[PREENCHER]` dependem diretamente desses resultados.

```markdown
Adding a data point to this issue: I hit the same underlying mechanism
(compiler-builtins-local math symbols winning ELF symbol interposition
over the dynamic libm), but with two differences that might be worth
tracking separately, and a stronger reproduction than a static `nm -uD`
check.

**Different symbols than the ones discussed here.** This issue and #139487
both trace back to compiler-builtins#763's curated "exact result" set
(`ceil`, `floor`, `sqrt`, etc). My case is `log10f`/`atan2f`/`acosf` —
transcendental functions that, as far as I can tell reading #763, were
explicitly *not* part of that curated set. [PREENCHER: confirmar/refutar
com o resultado do Passo 2 — se JÁ fazem parte de uma lista mais recente,
ajustar este parágrafo].

**A qualitatively worse symptom under `-Wl,-z,now`: an infinite hang, not
just a different numeric result.** With eager binding (`-C
link-arg=-Wl,-z,now`), I can reproduce a genuine infinite loop: the call
resolves to a local trampoline for `log10f`, which jumps to the PLT stub,
whose GOT slot — instead of pointing into `libm.so.6` — was resolved back
to the same local trampoline. Two `jmp` instructions, zero computation,
zero syscalls, 100% CPU forever. I confirmed this by attaching gdb to the
live (hung) process and reading the actual runtime value written into the
GOT slot (not just the relocation type from `readelf -r`, which only
shows what was *requested*, not what `ld.so` actually resolved):

```shell
(gdb) print/x *(void**)<runtime-GOT-slot-address>
$1 = <address inside the binary's own .text, not inside libm.so.6's mapped range>
(gdb) x/3i *(void**)<runtime-GOT-slot-address>
   <log10f>: jmp <log10f@plt>
```

Minimal reproducer (no staticlib/FFI, plain Rust, matching the one already
posted above but with an eager-bound transcendental function):

```rust
fn main() {
    let x: f32 = std::hint::black_box([PREENCHER: valor exato do Passo 1]);
    let y: f32 = std::hint::black_box(1.0);
    println!("{}", (x / y).log10());
}
```

```shell
rustc -O -Clink-arg=-Wl,-z,now main.rs -o repro
timeout -s KILL 10 ./repro; echo "exit=$?"   # hangs, exit 124/137
```

### Environment

`rustc --version --verbose`:

```text
rustc 1.96.1 (31fca3adb 2026-06-26)
binary: rustc
commit-hash: 31fca3adb283cc9dfd56b49cdee9a96eb9c96ffd
commit-date: 2026-06-26
host: x86_64-unknown-linux-gnu
release: 1.96.1
LLVM version: 22.1.2
```

- [PREENCHER: resultado do Passo 5 — versão em que passou a reproduzir]
- [PREENCHER: resultado do Passo 3 — símbolo é `WEAK` ou `GLOBAL`?]
- [PREENCHER: resultado do Passo 4 — reproduz sem `-z,now`?]

Happy to provide the full binary/gdb session, or help narrow this down
further — this looked severe enough (a real, reproducible hang, not just
a numeric difference) that I wanted to flag it here rather than open a
possibly-duplicate issue, but I'm glad to split it into its own issue
cross-referencing this one if that's more useful for tracking. cc
@tgross35 @Amanieu since you were both already discussing the
weak-vs-strong-symbol precedence question above.

---

## 8. Como e quando publicar (processo, considerando como o `rust-lang/rust` trabalha)

1. **Executar toda a §6 primeiro.** Publicar um relatório com lacunas
   (`[PREENCHER]` sem resposta) é exatamente o tipo de ruído que consome o
   tempo de mantenedores voluntários (`E-help-wanted`/`E-hard` já sinaliza
   que este tópico já é escasso em atenção de mantenedor) — melhor
   investir uma sessão de trabalho fechando as lacunas do que publicar
   rápido e incompleto.
2. **Publicar como comentário em [`#142119`](https://github.com/rust-lang/rust/issues/142119)
   primeiro, não como issue nova.** É a issue mais próxima, ainda aberta,
   com os mantenedores certos já engajados. Pedir explicitamente a opinião
   deles sobre separar em uma issue própria (o rascunho da §7 já inclui
   essa pergunta) — deixar a decisão de fragmentar ou não para quem
   triagem regularmente aquele repositório.
3. **Se pedirem para abrir uma issue nova**: usar o mesmo formato observado
   nas issues existentes (`Code` / `Version it worked on` / `Version with
   regression` / `Additional information`) — é o template de regressão
   padrão do projeto, reconhecível e mais rápido de triar. Aplicar os
   labels observados como convenção: `A-compiler-builtins`, `A-linkage`,
   `C-bug`, `T-compiler`, `T-libs` (não aplicar `regression-from-stable-to-stable`
   sozinho sem antes confirmar via bisecção — Passo 5 — que é
   genuinamente uma regressão entre duas stables, não apenas uma
   característica de uma versão específica de nightly).
4. **Ser paciente e específico, não insistente.** O label `E-hard` já
   avisa que a correção completa (visibilidade de símbolo corretamente
   aplicada a executáveis, não só shared objects) é trabalho não-trivial
   de compilador/linker. Nossa contribuição de maior valor é a
   **evidência**, não pressionar por uma correção rápida.
5. **Se o mecanismo permitir, oferecer o binário/ambiente para reprodução
   assistida** (ex.: um link para um contêiner/imagem reprodutível, ou os
   binários com símbolos preservados) — reduz a fricção para quem for
   investigar, e é uma prática bem vista nesse tipo de repositório de alto
   volume.
6. **Não mencionar o nam-rs como produto/projeto** além do necessário para
   contexto de reprodução — o interesse dos mantenedores é no
   comportamento do compilador, não no nosso projeto especificamente.
   Manter o foco estritamente na reprodução mínima da §7.

---

## 9. Referências

- [`rust-lang/rust#142119`](https://github.com/rust-lang/rust/issues/142119) — issue principal, aberta, alvo do nosso comentário
- [`rust-lang/rust#139487`](https://github.com/rust-lang/rust/issues/139487) — issue relacionada, fechada, mesma causa histórica (PR #763)
- [`rust-lang/rust#137578`](https://github.com/rust-lang/rust/issues/137578) — tracking issue mencionada em #142119
- [`rust-lang/compiler-builtins#763`](https://github.com/rust-lang/compiler-builtins/pull/763) — PR que reintroduziu disponibilidade fraca do conjunto "resultado exato"
- [`rust-lang/compiler-builtins#345`](https://github.com/rust-lang/compiler-builtins/issues/345) — discussão histórica sobre colisão de símbolos compiler-builtins vs. libc em FFI
- [`rust-lang/rust#131090`](https://github.com/rust-lang/rust/issues/131090) — tracking issue de `-Zdefault-visibility`
- [`rust-lang/compiler-team#782`](https://github.com/rust-lang/compiler-team/issues/782) — MCP da flag `-Zdefault-visibility`
- [Unstable Book: `default-visibility`](https://doc.rust-lang.org/unstable-book/compiler-flags/default-visibility.html) — nota explícita de que só afeta shared objects, não executáveis
- [Gist de `@Noratrieb`](https://gist.github.com/Noratrieb/a58000d43527225ed5af0032890e00a5) — demonstração de que símbolo fraco estático vence símbolo forte dinâmico
- [Chromium issue 419258012](https://issues.chromium.org/issues/419258012) — terceiro consumidor afetado, citado em #142119
- `docs/postmortem-libm-symbol-interposition.md` (este repositório) — o mecanismo completo, lições aprendidas, e a correção aplicada localmente no nam-rs
