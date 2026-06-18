<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Mapa de Otimização do nam-rs — Internalização de dependências "hot-path" ⚙️

> **Propósito**: este documento mapeia **dependências externas que entregam funcionalidade de
> alta performance / RT** e avalia, para cada uma, a **viabilidade de internalizá-la** dentro do
> `nam-rs` em busca de mais integração, controle e otimização — exatamente o que já foi feito com
> o **resampling** (antes genérico, hoje um polyphase FIR nativo com dispatch SIMD e ganhos
> palpáveis — ver `src/dsp/resampler.rs` + `src/dsp/sinc_kernel.rs`).
>
> Para **problemas/estranhezas de produto** (fidelidade, latência, denormais) consulte
> **`TODO-problemas.md`** (achados "P"); para **lacunas de funcionalidade** vs NeuralAmpModelerCore
> consulte **`TODO-features.md`** (achados "F"). Este documento trata do **terceiro eixo**:
> **performance estrutural via internalização** (achados "O").
>
> **Origem**: auditoria das dependências do `Cargo.toml` cruzada com o mapeamento de uso real no
> código (caminho RT × caminho de carga), conduzida pela skill `pesquisador-inovador`. Cada
> achado traz o uso real medido, o custo da dependência, a viabilidade de internalização e uma
> avaliação de custo × benefício.
>
> **Natureza**: nenhum "O" é um bug. São **oportunidades de performance e enxugamento** que
> seguem a filosofia do projeto: o `nam-rs` é **opinado a explorar ao máximo Linux + x86-64 + SIMD
> à mão**, e prefere **código próprio, integrado e tunado** a abstrações genéricas no hot-path —
> desde que a substituição seja **correta** (validada contra a dependência original como golden) e
> **RT-safe** (zero-alloc/zero-lock/zero-panic no hot-path, alocação só no load).
>
> **Filosofia (reafirmada)**: internalizar **não** é "reinventar a roda por esporte". É justificado
> quando (a) a dependência é **genérica demais** para o uso específico (como era o resampler), (b)
> arrasta **cadeia de build** desproporcional ao que se usa, ou (c) o uso real **deixa performance
> na mesa** que só o controle total destrava. Quando nenhum desses se aplica, **manter a
> dependência** é a decisão correta (vide `rtrb`, deferido).

---

## Sumário de Otimizações

| ID     | Achado (dependência → ação)                                                        | Ganho esperado                             | Esforço   | Hot-path? | Decisão           |
| ------ | ---------------------------------------------------------------------------------- | ------------------------------------------ |:---------:|:---------:| ----------------- |
| **O1** | **`half`** → internalizar f16↔f32 + usar F16C escalar nas caudas SIMD              | Build mais enxuto + ganho em caudas        | Baixo     | Parcial   | ✅ **DONE**       |
| **O2** | **`minstant`** → unificar com o clock TSC interno (`RtClock`)                      | −1 proc-macro + −`syn` v1; 1 clock único   | Baixo-Méd | Morno     | 🟢 Fazer          |
| **O3** | **`rustfft`** (cabsim UPOLS) → **RFFT real interna** + **MAC complexo SIMD**       | **~2× FFT + metade da memória** do cabsim  | Alto      | **Sim**   | ★ PoC faseada     |
| **O4** | **`rtrb`** → fila SPSC interna com padding de cache-line                           | Marginal (crate é minúsculo/zero-dep)      | Médio     | Controle  | ⚪ Deferir        |
| **O5** | **Cobertura SIMD x86-64-v3** → vetorizar lacunas escalares no hot-spot (auditoria) | Latência ↓ no hot-path (kernels escalares) | Baixo     | **Sim**   | 🟢 Fazer (índice) |

> **Sobre O5**: é uma classe **diferente** das demais (O1–O4 são internalização de dependências;
> O5 é **auditoria de cobertura SIMD do código próprio**). Pelo "toque esperto" do PO, os achados
> de O5 que tocam código que será expandido por **features** foram **inseridos como orientações de
> otimização nos achados F correspondentes** (`TODO-features.md`): A2 head conv + rechannel → **F3**;
> WaveNet `head_scale` → **F1**. O §O5 abaixo serve de **índice + guard-rail**; só o que **não** cabe
> num F permanece aqui como item próprio.
> **Não-achado relevante**: o `nam-rs` **não usa nenhum crate SIMD externo** (`wide`, `pulp`,
> `packed_simd`). Todo SIMD é `core::arch::x86_64` escrito à mão e dispatchado via CPUID
> (`src/math/common/dispatch/`). Não há nada a internalizar nesse eixo — **já está internalizado**,
> e essa é a razão de o resampler e os kernels GEMM/LSTM serem tão tunados.

### Matriz de priorização (esforço × impacto)

```text
IMPACTO
  ▲
A │                               O3  rustfft → RFFT interna  ★ flagship
L │
T │            O1  half
O │      O2  minstant
  │
B │        O5  cobertura SIMD (itens em F1/F3)         O4  rtrb  ⚪ (deferir)
A │
I │
X └──────────────────────────────────────────────────────────────────►
       BAIXO              MÉDIO                 ALTO         ESFORÇO

🟢 = fazer já    ★ = PoC profunda faseada    ⚪ = opcional/deferido
```

**Ordem recomendada**: O1 → O2 → O5 (lacunas escalares, baixo risco) → O3a (MAC SIMD) →
O3b (RFFT interna) → (O4, opcional). _Nota: o grosso de O5 é executado via F1/F3 quando essas
features forem implementadas; o resíduo próprio de O5 (limpeza) é trivial._

---

## Como o uso real foi medido (caminho RT × caminho de carga)

A audio-thread só é "hot" no que roda **por bloco/por amostra**. Construtores (`*::new()`) e o
loader rodam **fora** da thread RT e podem alocar/planejar à vontade. A classificação abaixo
separa rigorosamente os dois — é o que define se internalizar tem impacto de **runtime** ou
apenas de **build/manutenção**.

| Dependência | Uso no hot-path (RT)                                          | Uso fora do hot-path (cold)                                 |
| ----------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| `half`      | caudas escalares de kernels SIMD; fallback escalar puro       | quantização BF16 no load (`ops.rs:20`)                      |
| `rustfft`   | **`cabsim/conv.rs:218 process()`** (FFT fwd + IFFT por bloco) | `sinc_kernel.rs` (geração offline), `perceptual.rs` (teste) |
| `minstant`  | 1× por bloco no path CLAP (telemetria de ciclo)               | `Anchor::new()` no startup                                  |
| `rtrb`      | drenagem de comandos/GC por bloco (canais de **controle**)    | criação dos ring buffers no setup                           |

---

## O1 — ✅ `half`: internalizar f16↔f32 e usar F16C escalar nas caudas — [DONE] (verificado jun/2026)

> **Status final (jun/2026, auditoria revisor-auditor):** **CONCLUÍDO.** A verificação encontrou
> O1 inteiramente implementado:
>
> - **`src/math/common/half.rs`** (313 linhas) existe e contém: `f16_bits_to_f32` (software),
>   `f32_to_f16_bits` (software RNE, com o fix de carry do **P11** em `:117`), e as variantes
>   F16C de hardware `f16_bits_to_f32_f16c` (`_mm_cvtph_ps`, `:140`) e `f32_to_f16_bits_f16c`
>   (`_mm_cvtps_ph`, `:163`).
> - **`half` removido do `Cargo.toml`** — zero refs `half::` externas; 19 usos migrados para
>   `crate::math::common::half::`. (Resíduo no `Cargo.lock` é **transitivo** via `ciborium` →
>   `clack-*`, alheio ao f16 do nam-rs.)
> - **Todas as 16 caudas escalares dos kernels SIMD usam `f16_bits_to_f32_f16c`** (F16C
>   hardware) — nenhuma cauda usa software. O software só permanece (por design) em
>   `scalar_ref/`, paths `*_scalar` de teste do LSTM e `quantize_weight` (cold).
> - **Testes exaustivos** presentes: decode 65.536 padrões (software vs F16C bit-exato,
>   `half.rs:251`), encode 65.536 (`:268`) e regressão do carry P11 (`:291`).
>
> **Única pendência relacionada (roteada para O5/S2):** o **A2 rechannel**
> (`src/models/a2/model/mod.rs:268`) ainda usa `f16_bits_to_f32` **software** redundante por
> frame em vez de pré-decodar com F16C — micro-otimização de hot-path, ver **§O5 (S2)** abaixo.

**O que era.** O crate `half` fornecia o tipo IEEE-754 binary16. O `nam-rs` usava **apenas duas
operações** dele: `half::f16::from_bits(u16).to_f32()` e `half::f16::from_f32(f32).to_bits()`.

**Uso real (medido).** O loop SIMD quente **já não usa `half`** — usa o intrínseco de hardware
**F16C** (`_mm_cvtph_ps` / `_mm256_cvtph_ps`, ex. `src/models/a2/conv1d_ch3/mod.rs:201`). O crate
só aparece em:

- **fallback puramente escalar** (CPUs sem AVX2) — `src/math/common/scalar_ref/dot.rs:24`,
  `scalar_ref/gemm.rs:65,97,136`;
- **caudas escalares dos kernels SIMD** (quando o comprimento não é múltiplo de 8/16) —
  `src/math/gemm/dot_4x/avx2.rs:108`, `dot_4x/avx2_dual.rs:228`, `gemm/gemv_4gate/avx512.rs:103`;
- **quantização BF16 no load** (cold) — `src/math/common/ops.rs:20`;
- caminhos escalares de LSTM usados como teste/fallback — `models/lstm/layer_kernels.rs:238`,
  `model1.rs:126`, `model2.rs:167`.

**Custo da dependência.** `half` 2.7.1 arrasta **`zerocopy` 0.8 + `zerocopy-derive`** (proc-macro
→ `syn` 2, `quote`, `proc-macro2`). É uma **cadeia proc-macro pesada** para o que são duas funções
de bit-twiddling. `zerocopy` só entra no grafo de produção por causa do `half`.

**Viabilidade de internalização.** **Trivial.** ~40–60 linhas de manipulação de bits IEEE-754
binary16 (round-to-nearest-even, subnormais, NaN/Inf) num módulo próprio (ex.
`src/math/common/half.rs`). **Bônus de hot-path**: as caudas escalares já rodam sob
`#[target_feature(enable = "f16c")]` — trocar a conversão por software do `half` (com branches/
tabela) pelo intrínseco escalar **`_mm_cvtph_ps`** (1 instrução) é **remoção de dependência _e_
ganho** na cauda. O fallback puramente escalar (sem F16C) mantém a versão em software.

**Custo × benefício.** Custo **BAIXO** (~1 dia). Benefício **MÉDIO**: remove a cadeia
proc-macro/`zerocopy` (build mais rápido e grafo menor) + pequeno ganho de runtime nas caudas SIMD.
**Risco quase nulo**: existem só **65.536** padrões de bits f16 → teste **exaustivo** f16→f32
contra o crate `half` para **todos** os inputs antes de remover a dependência; round-trip f32→f16
sobre conjunto grande + edge cases (±0, subnormais, máximo normal, overflow→Inf, NaN). Após esse
teste, a substituição é **bit-exata por construção**.

**Diretrizes.**

1. Criar `half.rs` interno com `f16_bits_to_f32` (software) e `f32_to_f16_bits` (RNE).
2. Adicionar `f16_bits_to_f32_f16c` (intrínseco `_mm_cvtph_ps`) para as caudas sob `f16c`.
3. Substituir todos os `half::f16::*` (lista acima) pelas funções internas.
4. Teste exaustivo (65.536) + round-trip; só então **remover `half` do `Cargo.toml`**.
5. `cargo bench` em `dot_4x` / `conv1d` para confirmar ≥ paridade (esperado: leve ganho na cauda).

---

## O2 — 🟢 `minstant`: unificar com o clock TSC interno

**O que é.** `minstant` fornece um `Instant`/`Anchor` baseado em TSC com calibração. É usado no
**path CLAP** para medir o tempo de ciclo do `process()` (`src/clap/processor/mod.rs:32`,
`processor/dsp/orchestrator.rs:12`, `processor/dsp/telemetry.rs:5`) e no `Anchor::new()` do startup.

**Uso real (medido) — a duplicidade.** O path **standalone já tem sua própria calibração RDTSC**
(`src/standalone/rt_setup/tsc.rs`, via `core::arch::x86_64::_rdtsc()` + `TSC_FREQ_GHZ_X1000`). Ou
seja, **convivem duas implementações de clock**: a interna (standalone) e a do `minstant` (CLAP).

**Custo da dependência.** `minstant` arrasta **`ctor`** (proc-macro), **`web-time`** e — pior —
**`syn` v1**, forçando a compilação de **duas major versions** do `syn` no grafo.

**Viabilidade de internalização.** **Boa.** Promover o TSC clock interno a um `RtClock`/`TscInstant`
**único**, reusado por CLAP e standalone. Remove `minstant` + `ctor` + `web-time` + `syn` v1 e
unifica a telemetria de latência num só caminho auditável.

**Cuidado (não regredir robustez).** O `minstant` faz **detecção de _invariant TSC_ via CPUID** e
**fallback para `clock_gettime(CLOCK_MONOTONIC)`** em CPUs sem TSC invariante. A internalização
**deve preservar** isso: o `tsc.rs` atual precisa ganhar o probe CPUID (`0x80000007` bit 8) e o
fallback monotônico antes de virar o clock canônico.

**Custo × benefício.** Custo **BAIXO-MÉDIO** (~1–2 dias, foco em calibração + CPUID + fallback).
Benefício **MÉDIO**: −1 proc-macro, −`syn` v1, **um único clock** para todo o projeto (consistência
de telemetria — relevante para `TODO-problemas.md §P5/§P6`, picos de latência e quantização do
histograma). **Risco baixo** com teste de calibração (±0.1% vs `clock_gettime`) e teste do path de
fallback.

**Diretrizes.**

1. Estender `src/standalone/rt_setup/tsc.rs` (ou movê-lo para `common/`) com probe de invariant-TSC
   (CPUID) + fallback `CLOCK_MONOTONIC`.
2. Expor `RtClock { now() -> TscInstant, elapsed_ns() }` e migrar o path CLAP para ele.
3. Teste de calibração e de fallback; **remover `minstant`** do `Cargo.toml`.

---

## O3 — ★ `rustfft` (cabsim UPOLS): RFFT real interna + MAC complexo SIMD

Este é o achado de **maior potencial de ganho** — análogo, em espírito, ao que ocorreu com o
resampling. **PoC faseada** (do menor para o maior risco).

**O que é.** `rustfft` fornece FFT complexa mixed-radix com kernels SSE/AVX internos. No `nam-rs` é
usado no motor de convolução **UPOLS** do cabinet simulator (`src/dsp/cabsim/conv.rs`), que por
**bloco** faz: forward-FFT do segmento de entrada + IFFT do acumulador (passos 2 e 5 de
`process()`, linhas 238–296). É o **único uso de `rustfft` no hot-path** (os usos em
`sinc_kernel.rs` e `perceptual.rs` são offline/teste).

**Oportunidades de inovação (não-óbvias, medidas no código).**

1. **RFFT/IRFFT em vez de FFT complexa cheia.** A entrada é **real**
   (`Complex::new(sample, 0.0)`, `conv.rs:240`) e da IFFT **só se usa a parte real** (`conv.rs:303`).
   Hoje faz-se uma FFT **complexa cheia sobre dados reais** → ~2× de trabalho desperdiçado.
   Empacotando N reais numa FFT complexa de **N/2** + pós-processamento O(N) (e o dual na IFFT),
   **corta-se o custo de FFT pela metade** e o espectro vai de `N` para **N/2+1 bins**.
2. **Metade da pegada de memória.** Com RFFT, `h_fdl` e `fdl` (`conv.rs:48,51`) passam de `N` para
   `N/2+1` bins complexos cada. Como o **MAC particionado é memory-bound**, isso melhora cache
   diretamente — ganho que **se soma** ao da FFT.
3. **O MAC complexo está escalar!** Os passos 4 (`conv.rs:259–292`) são um **loop escalar** de
   `re/im`. **Vetorizar esse MAC é ganho imediato e independente da FFT** — com layout _split_
   (arrays `re[]` e `im[]` separados, em vez do interleaved atual) some-se shuffles e usa-se
   FMA AVX2/AVX-512 direto (reaproveitando os padrões de `src/math/dsp/stereo/`).
4. **FFT sempre power-of-two** (`(2*partition_size).next_power_of_two()`, `conv.rs:94`) → permite um
   radix-2/4/split-radix **especializado** e tunado com AVX-512, sem o overhead do dispatcher
   mixed-radix genérico do `rustfft`.

**Custo da dependência.** `rustfft` arrasta `num-complex`, `num-integer`, `primal-check`,
`strength_reduce`, `transpose` (sem proc-macros; build moderado). Internalizar a FFT **liberaria
também** `sinc_kernel.rs` e `perceptual.rs` a reusar o motor próprio (removendo `rustfft` inteiro).

**Viabilidade de internalização.** **Alta complexidade, mas escopo controlado** (só power-of-two,
validado contra `rustfft` como golden). Uma FFT correta e rápida exige twiddle tables, bit-reversal/
self-sorting e kernels AVX — não é trivial, mas é factível e de alto retorno **se o cabsim for
default-on**.

**Custo × benefício.** Custo **ALTO** (multi-sprint). Benefício **ALTO**: ~2× FFT + metade da
memória, por bloco, num caminho RT. **Risco** numérico mitigado por golden vs `rustfft` (erro
< 1e-5) e pelos testes de ESR do cabsim já existentes.

**Estratégia faseada.**

- **O3a — MAC complexo SIMD (ganho rápido, independente da FFT).** Vetorizar `conv.rs:259–292` +
  migrar `h_fdl`/`fdl` para layout _split_ `re[]`/`im[]`. **Não toca a FFT.** Mede o ganho do MAC
  isoladamente. _Esforço: Médio._
- **O3b — RFFT/IRFFT interna power-of-two (PoC).** Motor próprio validado contra `rustfft`; espectro
  em N/2+1 bins; `h_fdl`/`fdl` reduzidos à metade. _Esforço: Alto._
- **O3c — kernels AVX-512 dedicados + fusão FFT↔MAC.** Tuning final. _Esforço: Alto._

**Diretrizes.**

1. Antes de tudo: **bench de baseline** do `cabsim` (forward-FFT, MAC, IFFT separadamente) para
   atribuir ganho a cada fase.
2. O3a primeiro (baixo risco, não mexe na FFT) — colher o ganho do MAC.
3. O3b com `rustfft` ainda presente como **golden de referência** nos testes; só remover `rustfft`
   quando `sinc_kernel.rs`/`perceptual.rs` também migrarem (ou usarem um caminho de teste à parte).
4. Validar **ESR do cabsim** inalterado e erro < 1e-5 vs `rustfft` em IRs reais.

---

## O4 — ⚪ `rtrb`: internalizar é viável, mas de baixo retorno (deferir)

**O que é.** `rtrb` é uma fila **SPSC lock-free** bounded (estilo Lamport com índices cacheados),
**zero-dependências**. O `nam-rs` já a **encapsula** na sua própria infra
(`src/common/spsc/{mod,gc}.rs`) e a usa **só para canais de controle** — parâmetros, GC cascade e
swap de modelo (resampler/cabsim IR). **Os dados de áudio não passam por `rtrb`** (usam o
`mirror_buf` de huge-pages).

**Viabilidade de internalização.** Factível: uma fila SPSC com **padding de cache-line** (evitar
_false sharing_ entre os índices de produtor/consumidor) é ~150–250 linhas. Daria controle de
layout e integração mais fina com o GC cascade.

**Custo × benefício.** Custo **MÉDIO** (e exige stress/`loom` pesado para garantir o _memory
ordering_ lock-free). Benefício **BAIXO**: `rtrb` é minúsculo, **zero-dep** e bem testado — a
remoção de dependência quase não muda o build, e o risco de errar ordering sutil é real. **Decisão:
deferir.** Só reconsiderar se um perfil apontar _false sharing_ nos índices ou se a integração com
o GC cascade exigir semântica de _drop_ sob medida.

---

## O5 — 🟢 Auditoria de cobertura SIMD x86-64-v3 no hot-spot (índice + guard-rail)

**Contexto.** `.cargo/config.toml` fixa `-Ctarget-cpu=x86-64-v3` → **AVX2, FMA e F16C são garantidos
em tempo de compilação**. No hot-spot de inferência é, portanto, **proibido** deixar aritmética
escalar onde o v3 se aplica. Duas verdades guiaram a auditoria:

1. **O dispatcher é limpo** (`src/math/common/dispatch/`): não há variante escalar; faz `panic` se
   AVX2 ausente (`detect.rs:91`). A escolha em runtime é binária — **AVX2 (garantido) vs AVX-512
   (detectado)**. Logo o risco **não** é "fallback escalar em runtime".
2. **LLVM não auto-vetoriza reduções FP** em Rust safe (adição float não-associativa; sem
   `-ffast-math`). Portanto **todo laço de redução escalar no hot-path é genuinamente escalar** no
   binário — não há autovec salvando.

**Resultado da auditoria (achados confirmados — sempre-ativos e alcançáveis):**

| Achado              | Local                                | Natureza                                                                                                                  | Onde foi roteado             |
| ------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| **S1**              | `src/models/a2/head.rs:96-118`       | Head conv da A2 **100% escalar** (produção, todo bloco; CH=8 → 128 FMA escalares/frame)                                   | → **`TODO-features.md §F3`** |
| **S2**              | `src/models/a2/model/mod.rs:264-270` | Rechannel escalar **+ decode f16 redundante por frame**                                                                   | → **`TODO-features.md §F3`** |
| **S3** ✅ [DONE]    | `src/models/wavenet/model.rs:96-98`  | `head_scale` escalar dentro de fn `::<M: SimdMath>` (trivial: `M::apply_gain`) — vetorizado em S1.T1.1 (commit `adb4413`) | → **`TODO-features.md §F1`** |
| **S4**              | `src/dsp/cabsim/conv.rs:259-292`     | MAC complexo escalar                                                                                                      | → já é **§O3a** (acima)      |

> **Por que S1–S3 vivem em F1/F3 e não aqui (decisão do PO):** são otimizações de código que será
> **expandido/generalizado** por essas features — anexá-las como _orientação de otimização_ no F
> correspondente garante que o motor genérico **nasça SIMD** e não regrida ao padrão escalar atual.
> O detalhe acionável (file:line + estratégia de vetorização) está nos respectivos F.

**Resíduo próprio de O5 (não cabe em nenhum F — fica aqui):**

- 🟢 **[NOVO — auditoria jun/2026] S2 micro-otimização: A2 rechannel decodifica f16 em software,
  redundante por frame.** `src/models/a2/model/mod.rs:264-270` faz, dentro do laço de frames,
  `let rw = f16_bits_to_f32(self.rechannel_w[c])` — a versão **software** (`half.rs:18`), não a
  `f16_bits_to_f32_f16c` (`_mm_cvtph_ps`). Pior: as `CH` constantes `rechannel_w` **nunca mudam**,
  mas são re-decodadas a cada frame (CH=8 × 64 frames = 512 decodes/bloco; só 8 únicas). **Fix
  trivial e seguro:** pré-decodar `rechannel_w` para f32 **uma vez no load** (cold path) e o laço
  por frame vira um `mul` SIMD puro (ou `M::apply_gain`-like). Elimina ~504 decodes/bloco + tira
  a conversão do hot-path. Zero impacto numérico (bit-exato: F16C ≡ software, provado em O1).
  Conecta-se a **`TODO-features.md §F3`** (motor A2 geral), mas é autônomo o suficiente para uma
  micro-tarefa de O5. Detalhamento em `TODO-sprints.md`.

- ✅ **[DONE]** (S1.T1.2, commit `f7bed2b`: fallbacks escalares substituídos por `unreachable!()`).
  **Limpeza de cabeamento BF16 morto no AVX2.** `Avx2Math` cabeia `dot_product_bf16*`/`gemv_*_bf16`
  para fallbacks **escalares** (`src/math/common/avx2_impl.rs:42-99`), mas `is_bf16` só é `true` em
  `Avx512VnniBf16` (confirmado em `loader/dispatcher/lstm/static_builder.rs:28`,
  `wavenet/layout.rs:26`, `a2/model/set_weights.rs:41`) — no AVX2 os pesos são **F16** (com F16C
  SIMD). Ou seja, esses fallbacks escalares são **inalcançáveis**. **Não é lacuna de performance**;
  é apenas **código morto** a remover/documentar (esforço trivial, sem impacto de runtime).

**Não-achados (verificados — escopo da varredura):** caminhos `process_*_scalar` / `*_scalar_ref`
(LSTM, A2 layer, `conv1d_fallback`, `head` oracle, dot fallback) são **exclusivos de teste/paridade**
(ex.: `lstm/model1.rs:107` "Exclusively for parity tests"); o fallback genérico A2
(`model/mod.rs:364-418`) é **inalcançável** para o catálogo CH∈{3,8}; `ParamSmoother::tick` é IIR
**recursivo** (vetorização não-trivial, médio prazo); `gate.rs` é FSM de controle com aplicação de
ganho **já SIMD**; `wavenet/accumulate` já é AVX2/AVX-512.

**Guard-rail (vale para todo trabalho no hot-spot):** ✅ **[ATIVO]** (S2.T2.2: motor dinâmico nasceu
SIMD — dual-frame tiling + dot 4-wide interleaved; auditado e verde). Com x86-64-v3 garantido,
**nenhum laço de aritmética/redução f32/f16 por-amostra/por-bloco** deve permanecer escalar. Ao
implementar qualquer F que toque inferência, usar os kernels `SimdMath` (ou intrínsecos `core::arch`)
desde o início.

**📋 Parecer revisor-auditor (jun/2026) — planejado em `TODO-sprints.md` (Épico E-WN).** Roteamento
confirmado e refinado: **S3** (`head_scale` escalar, `model.rs:96-98`) é a fração de O5 que cabe ao
WaveNet — vira quick-win bit-exato `M::apply_gain` em **`S1.T1.1`** (kernel já existe,
`src/math/common/traits.rs:447`). A **limpeza BF16 morta** (`avx2_impl.rs:42-99`) é micro-tarefa
autônoma em **`S1.T1.2`** (código inalcançável, zero impacto numérico). O **guard-rail** ("nada escalar
no hot-spot") foi promovido a **regra de revisão** das sprints de motor dinâmico (**S2/S3**): o motor
genérico **nasce SIMD** (`S2.T2.2`). **Fora do escopo desta rodada**: S1 (head conv A2, `head.rs`) e S2
(rechannel A2, `model/mod.rs`) pertencem ao **motor A2/F3** — permanecem roteados a `TODO-features.md §F3`,
não às sprints WaveNet. Sprints WaveNet: **S1.T1.1** (head_scale), **S1.T1.2** (limpeza BF16),
**S2.T2.2** (born-SIMD do caminho dinâmico).

---

## Plano de PoC (resumo executável)

| PoC              | Entrega                                                                    | Critério de sucesso                                                                  |
| ---------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **O1**           | `math/common/half.rs` interno + F16C nas caudas                            | teste exaustivo 65.536 bate 100% com `half`; `half` removido; bench ≥ paridade       |
| **O2**           | `RtClock` TSC unificado (CLAP + standalone)                                | calibração ±0.1% vs `clock_gettime`; CPUID invariant-TSC + fallback; `minstant` fora |
| **O3a**          | MAC complexo do cabsim vetorizado + layout split `re[]/im[]`               | ESR do cabsim inalterado; bench do MAC ↓                                             |
| **O3b**          | RFFT/IRFFT interna power-of-two                                            | erro < 1e-5 vs `rustfft`; ~2× FFT; `h_fdl`/`fdl` em N/2+1                            |
| **O3c**          | kernels AVX-512 + fusão FFT↔MAC                                            | bench `cabsim` total ↓; `rustfft` removível                                          |
| **O5** (via F3)  | A2 head conv (`head.rs`) + rechannel (`model/mod.rs`) vetorizados AVX2/FMA | ESR/golden A2 inalterado; bench A2 ↓ no head/rechannel                               |
| **O5** (via F1)  | WaveNet `head_scale` → `M::apply_gain`                                     | bit-exato/ESR inalterado; remove laço escalar                                        |
| **O5** (limpeza) | remover cabeamento BF16 escalar morto no AVX2 (`avx2_impl.rs`)             | `cargo build`/`test` verdes; sem mudança numérica                                    |

Cada PoC fecha com `cargo check` + `cargo test` + `cargo bench` (perf) e correção de **todos** os
warnings, conforme `.agents/rules`.

---

## Recomendações de validação (padrão do projeto)

- **Golden contra a dependência original.** A correção de cada internalização é provada **contra a
  própria dependência** como referência: `half` (exaustivo 65.536), `rustfft` (erro < 1e-5 em IRs
  reais), `minstant`/`clock_gettime` (±0.1%). Internalizar **não** pode regredir números.
- **RT-safety preservada.** Toda alocação/planejamento permanece no `new()`/load; o hot-path segue
  zero-alloc, zero-lock, zero-panic (auditável com a feature `heap-audit`).
- **Bench antes e depois.** Sempre registrar baseline (`criterion`) por sub-operação para **atribuir**
  o ganho à fase certa — especialmente em O3 (FFT vs MAC vs IFFT).
- **Remoção só após verde total.** A dependência só sai do `Cargo.toml` quando todos os usos
  migraram e os goldens/benches passaram.

---

## Nota de método (para o `planejador-arquiteto`, quando acionado)

- Transformar estes "O" em Sprints/Tarefas em `TODO-sprints.md` respeitando a **ordem de
  dependência**: O1 e O2 são independentes e de baixo risco (boas primeiras sprints); O3 é faseado
  (O3a → O3b → O3c) e cada fase é uma sprint com critério de aceite numérico próprio; O4 fica como
  **sprint opcional deferida**.
- **O5 é transversal:** seu grosso (S1/S2/S3) é executado **dentro** das sprints de `F3` e `F1`
  (orientações de otimização já anexadas em `TODO-features.md`) — ao planejar essas features,
  incluir as tarefas de vetorização como critério de aceite. Só a **limpeza BF16 morta** é uma
  micro-tarefa autônoma de O5. O **guard-rail** de O5 ("nada escalar no hot-spot") deve ser regra
  de revisão para qualquer sprint que toque inferência.
- **Regra de ouro da internalização**: cada tarefa deve manter a dependência original como **golden
  de teste** até a substituição estar provada — e só então removê-la do `Cargo.toml`.
- Cruzar com `TODO-problemas.md §P5/§P6` (latência/telemetria) ao planejar O2, e com a precedência
  do **resampler** (`src/dsp/resampler.rs`) como padrão de internalização RT-safe.
