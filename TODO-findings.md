<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Pesquisa & Inovação: "Pending / Open Work" (`audio_fidelity_map.md §9`)

> **Skill geradora:** `pesquisador-inovador` → consolidação por `planejador-arquiteto`.
> **Chat Kilo Code:** TODO-findings para audio_fidelity_map
> **Foco:** entender **profundamente** o que é _real e necessário_ em cada um dos 5 itens
> pendentes de `docs/audio_fidelity_map.md §9`, separar o que é **fato** do que é **mito**,
> e propor mitigações **simples, seguras e factíveis**.
>
> **Restrições inegociáveis (premissas de todo o documento):**
>
> 1. **Não quebrar o padrão.** O contrato do formato `.nam` / `.namb` (topologia, valores de
>    peso armazenados, forward pass matemático) é intocável. Nenhuma mitigação pode alterar
>    como um modelo é _lido_ ou _interpretado_.
> 2. **Não quebrar os ideais do NAM-rs.** RT-safety estrita (zero alloc/lock/IO no hot-path,
>    zero heap-drop no RT — `.agents/rules/rust.md`), baixa latência ao vivo, e fidelidade
>    auditável (oráculo f64 + ASR + ESR + MR-STFT).
> 3. **Opt-in e default seguro.** Qualquer ganho de fidelidade que custe CPU/latência entra
>    como _opt-in_; o default de produção permanece o caminho ao vivo de menor latência.
>
> **Como ler:** cada finding `I1`…`I6` traz _Contexto_, _Estado atual verificado no código_
> (com `arquivo:linha`), _Análise crítica_ (o que é realmente necessário), _Proposta de
> solução_ (simples e segura), _Impacto na arquitetura/documentação_ e _Validação sugerida_.
> Os **Épicos** ao final agrupam para execução otimizada.

---

## Sumário executivo

| ID     | Item §9                                                 | Estado **real** verificado no código                                                                                  | Recomendação central                                                                                                  |
|:------ |:------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------- |
| **I1** | HighFidelity activation: controle do usuário (CLI/CLAP) | Infra **pronta e funcional**, mas `set_activation_precision()` **só é chamada por testes**. Sem CLI/CLAP/GUI.         | **FAZER.** É o item mais barato: alternar o modo é **apenas um store atômico** (sem rebuild/alloc). Expor CLI+CLAP.   |
| **I2** | Runtime oversample switching                            | **F2/PDC já corrigido.** Troca em runtime **já funciona 100% no CLAP**. Quebrado **só no standalone** (2 lacunas).    | **FAZER (escopo reduzido).** Espelhar o padrão _slimmable-rebuild_ no standalone + honrar `--oversample` no init.     |
| **I3** | Resampler quality selector (Standard 32T / HQ 64T)      | **F3/latência já corrigido**. Benchmark Δµs provou que 64T vs 32T economiza apenas 40 ns (<0.1% do pipeline).         | **RESOLVIDO (NÃO FAZER).** HQ-only é mantido permanentemente. Decisão documentada.                                    |
| **I4** | Kahan-compensated LSTM head accumulation                | Head f32-native usa `+=` simples (**sem Kahan**). Mas `decompose_error`: acumulação f32 é **negligível (~7,2e-13)**.  | **HIGIENE, não fix de §3.** Implementar Kahan no head é barato e correto, mas **não** move o drift de §3. Re-rotular. |
| **I5** | Oversampled recurrent state (LSTM HQ)                   | Sem OS interno no LSTM; o `OversampleEngine` externo **já roda o LSTM a 2×/4×**. §3(b): rate maior = **mais** drift.  | **REJEITAR caminho dedicado.** Mecanismo trocado (alias ≠ drift) + risco de mudar timbre. Caracterizar e documentar.  |
| **I6** | **(NOVO)** HF activations nos kernels fundidos do LSTM  | Kernels 4-gate fundidos **bypassam** o dispatch de `ActivationPrecision` (sempre Padé). Padé domina o piso (~7,6e-4). | **A ALAVANCA REAL de §3.** Maior ganho de fidelidade LSTM preservando formato. Surge desta pesquisa.                  |

> **Insight central desta pesquisa (detalhado em I6):** dos três componentes do piso de
> precisão absoluto do LSTM medidos pelo `decompose_error` — **Padé ≈ 7,6e-4**, f16c ≈ 5,1e-5,
> acumulação ≈ 7,2e-13 — o **termo dominante e controlável é a aproximação Padé da ativação**.
> Os dois mitigadores hoje listados em §9 para §3 (I4 Kahan e I5 oversampled state) atacam,
> respectivamente, o termo **mais negligível** (acumulação) e um **mecanismo diferente**
> (aliasing, não drift de quantização). A mitigação realmente eficaz para o piso do §3 é
> **levar o modo HighFidelity (I1) até os kernels do LSTM (I6)** — algo que hoje é uma
> limitação conhecida e documentada.

---

## Princípios de avaliação (o filtro "real vs. necessário")

Para cada item perguntamos, nesta ordem:

1. **O "bloqueador" citado ainda existe?** (Vários já foram resolvidos — confirmado no código.)
2. **O mecanismo proposto ataca a causa-raiz correta?** (Métrica certa, fonte de erro certa.)
3. **É a forma mais simples e segura?** (Reaproveitar padrões existentes > inventar novos.)
4. **Preserva formato e ideais?** (Opt-in, RT-safe, sem tocar no contrato `.nam`.)
5. **É mensurável com o arsenal atual?** (Oráculo f64, ASR, ESR, MR-STFT — `src/testing/`.)

---

## I1 — [FAZER · BAIXO RISCO] HighFidelity activation: expor controle ao usuário (CLI + CLAP)

### I1 — Contexto

`audio_fidelity_map.md §6` e `§9` registram que o modo `ActivationPrecision::HighFidelity`
existe, é testado, mas **não tem nenhum controle de runtime** (sem flag CLI, sem parâmetro
CLAP, sem widget de GUI). É "designed, not exposed".

### I1 — Estado atual (verificado)

- **Infra completa e funcional:**
  - `enum ActivationPrecision { Standard = 0, HighFidelity = 1 }` — `src/math/activations/mod.rs:57-64`.
  - Flag global atômica `ACTIVATION_MODE: AtomicUsize` — `src/math/activations/mod.rs:72-73`.
  - `set_activation_precision()` — `src/math/activations/mod.rs:80-82`; `activation_precision()` — `:86-91`.
  - Dispatch no caminho de áudio: `tanh_slice()` — `:94-101`; `sigmoid_slice()` — `:103-111` (ramo `if HighFidelity { …_hf } else { … }`).
- **Lacuna real:** `set_activation_precision()` **só é chamada por testes**
  (`tests/activation_precision.rs:301,304,329`). **Não há nenhum chamador de produção.**
  Grep confirma: zero `PARAM_ACTIVATION`, zero `--activation`, zero widget de GUI.
- **Espaço de parâmetro CLAP livre:** os IDs 0–7 estão ocupados (o último é
  `PARAM_OVERSAMPLE = 7`, `src/clap/extensions/params/mod.rs:9-24`); o próximo livre é **8**.
- **Limitação conhecida (importante):** os kernels fundidos 4-gate do LSTM **bypassam** o
  dispatch — chamam `simd_tanh_*`/`simd_sigmoid_*` diretamente
  (`src/math/lstm/gates.rs:29-45` AVX2, `:53-69` AVX-512; fallback escalar em
  `src/models/lstm/layer_kernels.rs:248-266`). Logo, hoje o HF afeta **WaveNet (A1+A2),
  ConvNet e Linear**, mas **não** o LSTM (tratado em **I6**).

### I1 — Análise crítica (o que é realmente necessário)

Este é, de longe, o item **mais simples e seguro** dos cinco — e o de maior relação
benefício/custo. Diferentemente do oversampling (I2), **alternar a precisão de ativação não
realoca nada**: é um único `store` atômico (`Relaxed`). O hot-path apenas lê a flag e segue
por outro ramo já compilado. As implicações:

- **RT-safety trivial.** A troca é um único `store` atômico `Relaxed` — **RT-safe por si só**
  (não aloca, não trava, não faz IO). **Não precisa** do mecanismo off-RT/SPSC do model swap.
  Para honrar o contrato já documentado no código (`mod.rs:69-77`: "set once… never changed
  mid-`process`"), aplicá-la **na borda do bloco** (no _param-flush_, como os demais
  parâmetros CLAP fazem no início de `process()`) — não no meio do laço de amostras. O único
  efeito colateral é uma reaprendizagem transitória do branch predictor — sem glitch, sem
  clique, sem realloc.
- **Sem risco de formato.** HF não toca pesos nem topologia; muda só a implementação da
  ativação. Compatível por construção.
- **Honestidade de UX.** Como o LSTM hoje ignora HF (até I6), o controle deve **documentar
  claramente** que, para modelos LSTM, o efeito é nulo enquanto I6 não for entregue.

> **Nota de paridade (relevante p/ docs):** a flag é **global de processo**, não por
> instância. Em standalone (uma engine) é perfeito. No CLAP, múltiplas instâncias no mesmo
> processo compartilhariam o modo. Como é opt-in e idempotente, o risco é baixo, mas a
> documentação deve registrar a semântica "última escrita vence" entre instâncias.

### I1 — Proposta de solução (simples e segura)

1. **CLI (standalone):** adicionar `--activation standard|hf` (alias `--act`) em
   `CliArgs` (`src/standalone/cli.rs:59-80`, parser em `:113-146`), espelhando exatamente o
   parsing de `--slim`/`--oversample`. No bootstrap, chamar `set_activation_precision(...)`
   **antes** de iniciar o PipeWire (junto do bloco `src/main.rs:191-192`).
2. **CLAP:** criar `PARAM_ACTIVATION = 8` (stepped 0/1, automatável), espelhando o
   `PARAM_OVERSAMPLE`:
   - Declaração/`ParamInfo` em `src/clap/extensions/params/main.rs` (value→text: `0→"Standard", 1→"HighFidelity"`).
   - Atômico `param_activation: AtomicU32` em `UiToRt` (`src/clap/plugin/shared.rs`).
   - Handler `set_activation(val)` no thread de áudio (`src/clap/processor/events.rs`) que
     chama `set_activation_precision(...)` **diretamente** (sem `NEEDS_*_REBUILD`).
   - Persistência em `state.rs` (save/load) e widget na GUI (`zones/controls.rs`), iguais ao oversample.
3. **Não** introduzir caminho off-RT: a ausência de rebuild é a vantagem deste item.

> Nota do PO: Assegurar existência de switch de ativação na CLI e na GUI. Escolher um default são.

### I1 — Impacto na arquitetura/documentação

- `audio_fidelity_map.md`: §6 deixa de ser "⚠️ Not user-exposed" → "✅ User-exposed (CLI+CLAP)";
  remover o item de §9; manter a ressalva LSTM (apontando para I6).
- `architecture.md §2 (Activation Precision Modes)`: documentar o novo controle e a semântica
  global-de-processo; manter a "Known limitation" do LSTM até I6.
- `docs/clap_integration.md`: novo parâmetro ID=8 na tabela.

### I1 — Validação sugerida

- `tests/activation_precision.rs`: estender para exercitar o caminho CLI/CLAP (já cobre o ESR
  via oráculo p/ WaveNet/ConvNet/Linear).
- Teste de "no realloc": garantir que alternar o modo no RT não dispara o `CountingAllocator`
  (reusar o guard de zero-alloc de `tests/nam_infer_test.rs`).

### I1 — Referências

- R1 (Sato & Smith 2025), R7 (Wright & Välimäki 2020) — relevância perceptual da precisão de ativação.
- `docs/fastmath-approximations.md` — análise de erro Padé vs HF.

---

## I2 — [FAZER · ESCOPO REDUZIDO] Runtime oversample switching: completar a paridade no standalone

### I2 — Contexto

§9 lista "Runtime oversample switching (currently init-time only)" com a nota
"`rt_callback/commands.rs` `TODO(oversample-rt)` (F2 PDC blocker)". A leitura ingênua sugere
que a troca em runtime está globalmente quebrada e bloqueada pela PDC. **A pesquisa mostra que
isso não é mais verdade.**

### I2 — Estado atual (verificado)

- **O bloqueador F2/PDC JÁ FOI RESOLVIDO.** A latência do oversampling **é reportada** ao host:
  - `OversampleEngine::latency_samples()` existe — `src/dsp/oversample.rs:322-333`
    (`Off→0`, `X2→12`, `X4→24`, com `HB_DELAY = 12` em `:31-34`).
  - Incluída na latência efetiva em `src/clap/processor/events.rs:117`
    (`effective_latency += self.os_l.latency_samples()`) **e** na latência inicial em
    `src/clap/processor/mod.rs:153`.
  - Notificação dinâmica ao host via `latency_ext.changed()` em
    `src/clap/plugin/main_thread/housekeeping.rs:239-248`.
- **A troca em runtime JÁ FUNCIONA 100 % no CLAP** (cadeia completa, RT-safe):
  `set_oversample` (`processor/params.rs:76-86`) → `apply_oversample` (sinaliza
  `RT_STATUS_NEEDS_OS_REBUILD` + `requested_os_factor`, `:115-125`) → rebuild **off-RT** em
  `housekeeping.rs:117-142` (`OversampleEngine::new` no main thread) → entrega via SPSC
  `SetOversample { os_l, os_r }` → `cold_load_os` faz o hot-swap e manda as engines antigas p/
  o GC (`events.rs:49-51,213-222`).
- **O que está realmente quebrado é só o STANDALONE — em dois pontos:**
  1. **CLI `--oversample` é um beco sem saída.** É parseado (`cli.rs:128-146`) e transportado
     em `PipewireHostConfig`, mas o valor é **descartado** em `run.rs:60` (destructuring `_os`)
     e o setup **fixa `Off`**: `capture/setup.rs:66`
     (`CaptureState::init(sys, OversampleFactor::Off)`). Ou seja, **a flag não faz nada hoje**.
  2. **Troca em runtime é no-op.** `commands.rs:108-135` ignora `ParamPayload::SetOversample`
     com um `TODO(oversample-rt)` explicativo, e o laço principal do standalone (`run.rs`)
     **não** trata `RT_STATUS_NEEDS_OS_REBUILD` (grep: zero ocorrências em `src/standalone/`).

### I2 — Análise crítica (o que é realmente necessário)

O "trabalho pendente" aqui é **muito menor** do que §9 sugere e **não tem bloqueador
técnico** — a PDC já está correta e o CLAP já é a prova de que o padrão funciona. Restam duas
coisas no standalone:

- **(a) Bug de usabilidade** (`--oversample` não tem efeito): é um defeito real, não uma
  feature pendente. Custa ~2 linhas (passar `args.oversample` ao `CaptureState::init` em vez
  de `Off`).
- **(b) Troca em runtime no standalone:** o próprio `TODO(oversample-rt)` já dá a receita:
  **espelhar o padrão _slimmable-rebuild_** que já existe no standalone (`try_slimmable_rebuild`
  / `drain_slimmable_models` em `run.rs`): RT seta flag → main thread reconstrói off-RT →
  entrega por SPSC → RT faz swap e descarta via GC. É **reaproveitar infraestrutura existente**,
  não criar nova — exatamente o "simples e seguro" pedido.

### I2 — Proposta de solução (simples e segura)

1. **Corrigir o init (quick fix, isolado):** em `capture/setup.rs:66` usar o fator vindo do
   CLI (propagar `oversample` por `run.rs`/`PipewireHostConfig` em vez de `_os`).
2. **Implementar a troca em runtime espelhando _slimmable_:**
   - No RT (`commands.rs`): ao receber `SetOversample(factor)`, gravar `requested_os_factor`
     em `rt_status` e setar `RT_STATUS_NEEDS_OS_REBUILD` (mesma flag já usada no CLAP,
     `src/common/spsc/status.rs:58,153`). **Não** realocar no RT.
   - No main loop (`run.rs`): adicionar o ramo que observa a flag, constrói
     `OversampleEngine::new(factor, …)` (L+R) off-RT e empurra por SPSC.
   - No RT: drenar o SPSC, fazer o swap e mandar as engines antigas ao GC (padrão `drain_*`).
3. **PDC:** nada a fazer — o standalone (PipeWire) não tem PDC de host; a latência é informada
   no relatório de diagnóstico. Garantir que o valor reportado reflita o novo fator.

### I2 — Impacto na arquitetura/documentação

- `audio_fidelity_map.md §5/§9`: atualizar para "runtime switching ✅ no CLAP; standalone
  alcançando paridade"; remover a menção a "F2 PDC blocker" (resolvido).
- `architecture.md §5.0O`: documentar a paridade standalone↔CLAP do rebuild de oversampling
  (mesma cascata GC/SPSC do model hot-swap).

### I2 — Validação sugerida

- Teste de impulso (já sugerido em F2 histórico): atraso de grupo medido do par up/down deve
  bater com `latency_samples()` (≤ 1 amostra) para 2× e 4×.
- Teste de integração standalone: alternar fator em runtime sem xrun e sem alloc no RT
  (guard de zero-alloc).

### I2 — Referências

- R3 (Kahles, Esqueda & Välimäki 2019) — design dos filtros half-band do oversampling.

---

## I3 — [MEDIR ANTES · PROVÁVEL NÃO-FAZER] Resampler quality selector (Standard 32T / HQ 64T)

### I3 — Contexto

§9 lista o seletor "Standard 32T / HQ 64T" como "Designed, HQ is default" com a nota
"(F3 latency formula blocker resolved)". A ideia original (Tarefa 5.7) foi **adiada à espera
de um benchmark Δµs**: se o custo do banco de 64 taps for não-desprezível, exporia-se um modo
Standard de 32 taps mais barato.

### I3 — Estado atual (verificado)

- **F3 (fórmula de latência) JÁ FOI RESOLVIDO.** `latency_samples()`
  (`src/dsp/resampler.rs:547-561`) usa `core.group_delay()`, que para bancos de **fase mínima**
  é o **centroide empírico** da resposta ao impulso (`calculate_centroid()`,
  `src/dsp/sinc_kernel.rs:214-223`; atribuído em `:134-147`). Bancos de fase linear continuam
  com `taps/2`. O `phase_type` é rastreado (`resampler.rs:128`). **Não há mais a
  superestimação de fase mínima.**
- **`TAPS_PER_PHASE = 64` é `const` de compilação** (`sinc_kernel.rs:66`); `NUM_PHASES = 256`
  (`:52`); `PROTO_LEN = NUM_PHASES * TAPS_PER_PHASE` (`:69`); `DELAY_LINE_LEN = TAPS_PER_PHASE*2`
  (`resampler.rs:51`). A `partition_polyphase()` usa `TAPS_PER_PHASE` (`:298-343`). **Porém** o
  kernel SIMD aceita `taps: usize` em runtime e `PolyphaseBank.taps_per_phase` é campo de runtime.
- `new_linear()` existe mas **nunca é chamado em produção** (só em testes).
- **Não existe** parâmetro/flag/caminho de 32 taps em produção.
- **Custo documentado:** `architecture.md:270` afirma que 64 taps vs 32 taps é **< 1 % do
  pipeline** quando o modelo neural está ativo, e o caso comum (host a 48 kHz) é **bypass de
  custo zero**.

### I3 — Análise crítica (o que é realmente necessário)

Aqui a inovação é **resistir a adicionar complexidade que provavelmente não se paga**:

- O **bloqueador citado (F3) já caiu**; o que resta é uma **decisão de custo**, não um
  problema técnico em aberto.
- O seletor 32T existe para **economizar CPU em hosts ≠ 48 kHz**. Mas: (i) o caminho mais
  comum é bypass (custo zero); (ii) quando há resample, o custo do resampler é **< 1 %** com o
  modelo ativo; (iii) reduzir para 32 taps **piora** a fidelidade (o doc registra ~24 dB de SNR
  no passband a 32T vs ≥ 100 dB a 64T) — ou seja, é uma troca **fidelidade↓ por CPU↓ marginal**,
  contrária ao ideal do projeto.
- Acrescentar o seletor implica: 2º banco pré-construído (parametrizar o gerador para
  `taps=32`), novo parâmetro CLAP (ID 8/9), wiring CLI, persistência, GUI, rebuild SPSC e
  cuidado com o `DELAY_LINE_LEN`. **Superfície de risco real** para um ganho de CPU duvidoso.

**Decisão:** decisão guiada por dados concluída. O benchmark Δµs provou que o custo do banco de 64 taps (HQ) é desprezível e que a economia do modo de 32 taps é insignificante, não justificando a perda de fidelidade e complexidade adicional de um seletor. Mantido **HQ-only** como padrão definitivo.

### I3 — Resultados do Benchmark (Passo 0 executado)

- **Upsampling (44.1 -> 48 kHz) por bloco de 256 amostras**:
  - HQ (64 Taps): **3.87 µs**
  - Standard (32 Taps): **3.71 µs**
  - Economia absoluta ($\Delta\mu\text{s}$): **0.16 µs** por 256 amostras (equivalente a **40 ns** por bloco de 64 amostras).
- **Inferencia WaveNet Standard (bloco de 64 amostras @ 48 kHz)**: **48.80 µs**.
- **Overhead relativo e economia**:
  - O overhead total do resampler de 64T (upsampling) em relação ao modelo neural é de apenas **~1.99%** do pipeline.
  - A economia de trocar 64T por 32T é de **~0.08%** do pipeline total de inferência (40 ns em 48.80 µs).
- **Conclusão**: Uma economia de <0.1% de CPU não justifica degradar a fidelidade do sinal (onde o SNR no passband cai de $\ge 100\text{ dB}$ com 64T para $\sim 24\text{ dB}$ com 32T) e adicionar complexidade na base de código. O item é **encerrado como HQ-only**.

### I3 — Impacto na arquitetura/documentação

- `audio_fidelity_map.md §4/§9` e `architecture.md §5`: atualizados para registrar que **F3 foi resolvido**, os resultados do benchmark, e que o seletor foi descartado mantendo o design **HQ-only** permanente.

### I3 — Validação sugerida

- Benchmark `criterion` dedicado (entregável do Passo 0).
- Caso implementado: teste de latência reportada vs atraso de grupo medido para os **dois**
  bancos (linear e mínimo, 32T e 64T).

### I3 — Referências

- R3 (Kahles, Esqueda & Välimäki 2019) — design do banco polifásico HQ.

---

## I4 — [HIGIENE, NÃO FIX DE §3] Kahan no head do LSTM: barato e correto, porém impacto ~nulo no drift

### I4 — Contexto

`lstm_recurrent_drift.md §7` e §9 propõem "Kahan-compensated head accumulation" como
mitigação para o drift recorrente do LSTM (§3), citando "reduzir o erro de acumulação em
1–2 ordens de magnitude".

### I4 — Estado atual (verificado)

- O head f32-native (`use_f32_head == true`) chama
  `dot_product_f32_native()` — `src/math/common/scalar_ref/dot.rs:66-75` — que é **soma `+=`
  pura, sem Kahan**. (A doc §6.1 afirma "f32 native p/ fidelidade de 24 bits"; isso vale para
  os **pesos** f32, mas a **acumulação** é f32 ingênua.)
- Utilitários Kahan **já existem**: `src/math/common/kahan.rs` (`kahan_add` `:120-124`,
  `KahanF32`, `Kahan4F32`, `NeumaierF64`) e já são usados no WaveNet
  (`scalar_ref/dot.rs:83-114`) e nas caudas escalares dos GEMV (`gemm/dot.rs:90,119,150`).
- **Dado decisivo (`decompose_error`, oráculo f64):** o piso de precisão absoluto do LSTM
  (3,57e-3 vs ideal) decompõe-se em **Padé ≈ 7,6e-4**, **f16c ≈ 5,1e-5** e **acumulação f32
  ≈ 7,2e-13** (`lstm_recurrent_drift.md:113`; `audio_fidelity_map.md:268`).

### I4 — Análise crítica (o que é realmente necessário — e o mito a desfazer)

**A acumulação f32 já é negligível (~7,2e-13), ~9 ordens de magnitude abaixo do termo
dominante (Padé).** Kahan compensa **exatamente** o erro de acumulação. Logo:

- **Como "fix de §3", Kahan no head é essencialmente um placebo.** Não há 1–2 ordens de
  magnitude a recuperar onde o erro já é ~1e-13. O drift do §3 vem dos **pesos f16c na
  recorrência** (intrínseco ao formato) e da **ativação Padé** (controlável — ver I6), não da
  soma do head.
- **Kahan não conserta quantização de pesos.** Mesmo aplicado na GEMV recorrente
  (cauda escalar `gemm/gemv_4gate/avx2.rs:155-158`, hoje `+=` puro), Kahan reduz erro de
  _soma_, não o erro de _representação_ do peso f16c. O `decompose_error` confirma que esse
  caminho não é o gargalo.

**Porém, há valor legítimo (re-rotulado):** como **higiene numérica e à prova de futuro**,
para sessões muito longas e heads largos (`1×40`, `2×24`, onde `H` chega a 40 → muitas somas
sequenciais, acima do limiar de 32 que a própria `kahan.rs:11` recomenda). É barato (o head é
escalar, fora do hot-path SIMD, roda 1×/amostra sobre `H ≤ 40` elementos) e **correto**.

### I4 — Proposta de solução (simples e segura)

1. Adicionar `dot_product_f32_native_kahan()` (ou um parâmetro) em
   `src/math/common/scalar_ref/dot.rs`, reusando `kahan_add` de `kahan.rs`. Usá-la no head
   f32-native (`models/lstm/model1.rs:21-27` e `model_dyn.rs`).
2. **Re-rotular honestamente** na documentação: trata-se de **robustez numérica**, **não** de
   mitigação mensurável do §3. Definir expectativa: `ΔESR ≈ 0` nas escalas atuais; benefício só
   em `N` patológico.
3. **Não** investir em Kahan na GEMV recorrente como "fix de §3" — `decompose_error` mostra que
   não move a agulha.

### I4 — Impacto na arquitetura/documentação

- `lstm_recurrent_drift.md §7` e `audio_fidelity_map.md §9`: **corrigir a narrativa** — Kahan
  no head é higiene; a alavanca de §3 é I6 (Padé→HF). `architecture.md §6.2` ganha nota de que
  o head passou a usar acumulação compensada.

### I4 — Validação sugerida

- Oráculo f64 (`tests/reference_oracle_f64.rs`): medir ESR do head com/sem Kahan em `1×16`,
  `1×40`, `2×24` a 5 s/48 kHz. Documentar que o Δ é ~ruído (confirma a tese).
- Soak test longo (`tests/soak_test.rs`) para `1×40`: confirmar estabilidade sem regressão.

### I4 — Referências

- Internas: `decompose_error()` (`src/testing/reference_oracle.rs`); `kahan.rs`. (Norma
  computacional; sem referência externa específica — alinhado ao P-4 do
  `research-references.md`.)

---

## I5 — [REJEITAR CAMINHO DEDICADO] Oversampled recurrent state: mecanismo trocado e risco de timbre

### I5 — Contexto

`lstm_recurrent_drift.md §7` e §9 propõem rodar o LSTM internamente a 2× ("oversampled
recurrent state") como modo HQ opt-in, alegando "reduzir o erro por passo espalhando a
quantização por mais amostras".

### I5 — Estado atual (verificado)

- **Não existe oversampling interno no LSTM** (grep `oversample` em `src/models/lstm/` = 0).
  O LSTM processa **amostra a amostra** (`models/lstm/model1.rs:21-35`).
- O **`OversampleEngine` externo já envolve o modelo inteiro** (`OversampleFactor::X2/X4`,
  `src/dsp/pipeline/stages/inference.rs:169-188`, `model_process_stereo_with_os`). Ou seja,
  **ligar `--oversample 2x` num modelo LSTM já roda o LSTM a 2× hoje** (no CLAP).
- **Dado de §3(b):** taxa maior ⇒ **mais** drift vs NAMCore (48 k = 2,61e-2; 96 k = 6,09e-2;
  192 k = 1,42e-1) — `lstm_recurrent_drift.md:21-22`.

### I5 — Análise crítica (a parte mais importante desta pesquisa)

Há **dois erros conceituais** na proposta original, confirmados por dados internos e pela
literatura recente:

1. **Mecanismo trocado: aliasing ≠ drift de quantização.** O §3 é sobre **drift de
   quantização f16c na recorrência**. Oversampling combate **aliasing de não-linearidade** —
   um artefato espectral diferente. Rodar mais passos **não reduz** o erro de representação dos
   pesos f16c; pelo contrário, **§3(b) mostra que mais passos = mais drift acumulado**. Como
   "fix de §3", o oversampled state é **contraproducente**.

2. **Risco de mudar o timbre (quebra de fidelidade ao modelo).** O LSTM é **dependente da taxa**
   de treino (48 kHz). O `OversampleEngine` externo **não ajusta o atraso de realimentação** da
   célula — então rodar a recorrência a 2× **altera a constante de tempo efetiva** dos gates →
   **muda o timbre**. A literatura de ponta confirma: Carson, Wright & Bilbao
   (DAFx/ICASSP/TASLP 2024–2025) mostram que oversampling **correto** de RNN exige **ajustar o
   comprimento do delay de realimentação para M amostras** + filtros dedicados de
   interpolação/decimação (Lagrange/minimax) para **preservar** a distorção harmônica do
   baseline. Sem isso, a tonalidade é "moderadamente afetada".

**Conclusão:** um caminho **dedicado** "oversampled recurrent state" como mitigação de §3 é
**injustificado** (mecanismo errado), **arriscado** (timbre) e **caro** (2–4× CPU). Não cria;
não complica.

**O que é real e útil aqui:**

- Oversampling **é** legítimo para **aliasing** (problema diferente, §2/§5/§6) — e o NAM-rs
  **já o oferece** via `--oversample` (I2). Para LSTM, deve ser **caracterizado e documentado**
  com o arsenal existente, **sem** novo código de modelo.
- Os caminhos **cientificamente corretos** (se algum dia perseguidos) são: **(a) RNN-ADAA**
  (Mikkonen & Werner, DAFx 2025) — reduz aliasing em todas as taxas com impacto **moderado** de
  timbre e **sem** fatores altos de OS; **(b) multirate RNN com delay = M** (Carson et al.). Os
  dois são grandes e esbarram na mesma objeção de dispatch que levou o projeto a **rejeitar
  ADAA** (`architecture.md §5.0O`). Ficam **catalogados como pesquisa futura**, não como item de
  fidelidade pendente.

### I5 — Proposta de solução (simples e segura)

1. **Não implementar** caminho dedicado de oversampling interno do LSTM.
2. **Caracterização empírica barata (testável hoje):** usar o `OversampleEngine` existente +
   `src/testing/` (oráculo f64, ASR, ESR, MR-STFT) para medir, num LSTM, `Off` vs `2×` vs `4×`:
   (a) ASR (esperado: melhora — aliasing cai); (b) ESR/MR-STFT **vs baseline 48 k**
   (esperado: piora/timbre muda — confirma a dependência de taxa). Publicar a tabela no doc.
3. **Documentar a orientação ao usuário:** "oversampling em LSTM serve a **anti-aliasing**, não
   ao drift de §3, e **pode alterar o timbre**; para fidelidade ao modelo, prefira não
   oversamplear LSTM, ou use WaveNet de orçamento equivalente."
4. **Catalogar pesquisa futura:** RNN-ADAA e multirate-RNN como referências (novas entradas em
   `research-references.md`).

### I5 — Impacto na arquitetura/documentação

- `lstm_recurrent_drift.md §7`: **remover** "oversampled recurrent state" da lista de
  mitigações de §3; substituir pela seção "Oversampling em LSTM: anti-aliasing, não anti-drift"
  com a tabela empírica e a ressalva de timbre.
- `audio_fidelity_map.md §9`: remover o item; `§5/§6`: nota sobre LSTM + oversampling.
- `research-references.md`: adicionar Mikkonen & Werner (2025) e Carson et al. (2024/2025).

### I5 — Validação sugerida

- Os próprios experimentos do passo 2 são a validação (ASR↓ e ESR/MR-STFT vs 48 k↑).

### I5 — Referências (novas — incluir em `research-references.md`)

- **Mikkonen, O.; Werner, K. J.** "Antiderivative Antialiasing for Recurrent Neural Networks."
  DAFx-2025, Ancona. (ADAA para GRU/LSTM; reduz aliasing em todas as taxas sem OS alto.)
- **Carson, A.; Wright, A.; Chowdhury, J.; Välimäki, V.; Bilbao, S.** "Sample Rate Independent
  Recurrent Neural Networks for Audio Effects Processing." DAFx-2024.
- **Carson, A.; Wright, A.; Bilbao, S.** "Interpolation filter design for sample rate
  independent audio effect RNNs." ICASSP 2025. + "Resampling filter design for multirate neural
  audio effect processing." IEEE TASLP, vol. 33, 2025.
- R2 (Carson, Wright & Bilbao 2025, fine-tuning) e R6 (Holters 2019, ADAA stateful) — já no acervo.

---

## I6 — [NOVO · A ALAVANCA REAL DE §3] HighFidelity activations nos kernels fundidos do LSTM

> **Este finding é a principal contribuição inovadora desta pesquisa.** Ele reposiciona todo o
> roadmap de fidelidade do §3.

### I6 — Contexto

Os itens I4 e I5 estão listados em §9 como **as** mitigações de §3, mas a análise mostra que
ambos erram o alvo. A pergunta correta é: **qual é o maior termo de erro controlável,
preservando o formato?**

### I6 — Estado atual (verificado)

- `decompose_error` (oráculo f64) do piso absoluto do LSTM (3,57e-3 vs ideal):
  **Padé ≈ 7,6e-4** ≫ f16c ≈ 5,1e-5 ≫ acumulação ≈ 7,2e-13 (`lstm_recurrent_drift.md:113`).
- O modo HighFidelity reduz o erro de ativação **~10.000×** (`audio_fidelity_map.md §2/§6`).
- **Mas os kernels fundidos 4-gate do LSTM bypassam o HF** (sempre Padé):
  `src/math/lstm/gates.rs:29-45` (AVX2), `:53-69` (AVX-512); fallback escalar
  `src/models/lstm/layer_kernels.rs:248-266` (usa `scalar_minimax_sigmoid` + `gg.tanh()`
  diretamente). Limitação reconhecida em `architecture.md §2` ("Known limitation") e
  `tests/activation_precision.rs:23-30`.

### I6 — Análise crítica (o raciocínio)

- O termo **dominante e controlável** do piso de precisão do LSTM é a **ativação Padé**
  (7,6e-4) — uma ordem acima do f16c (que é **intrínseco ao formato**, intocável) e ~9 ordens
  acima da acumulação (alvo de I4).
- Existe **uma ferramenta pronta** que ataca exatamente esse termo: o modo HighFidelity. Ela
  só **não alcança** o LSTM por uma limitação de implementação dos kernels fundidos.
- **Portanto, a mitigação mais eficaz, segura e fiel-ao-formato do §3 é levar o HF até os gates
  do LSTM (I6).** Isso preserva 100 % o formato (só muda a implementação da ativação), é opt-in
  (herda o controle de I1) e é RT-compatível.

> **Ressalva honesta (documentar):** HF aproxima o NAM-rs do **ideal matemático**, mas pode
> **aumentar a divergência interop vs NAMCore** (que também usa aproximações rápidas). Como o
> ideal declarado do NAM-rs é fidelidade à **matemática verdadeira do modelo** (não a
> bit-exatidão com o C++), HF é a opção "mais correta". O gate de paridade
> (`ABSOLUTE_ESR_CAP_LSTM`, `cpp_parity_map.md §4.5`) deve ser reavaliado para o caminho HF.

### I6 — Proposta de solução (faseada, segura)

1. **Fase 1 (correção, baixo risco):** implementar um caminho **escalar** HF para os gates do
   LSTM (tanh/sigmoid HF por elemento) selecionado pela mesma flag `activation_precision()`.
   Lento, porém **correto** e suficiente para validar o ganho de ESR via oráculo. Mantém o Padé
   como default ao vivo.
2. **Fase 2 (performance, se justificado):** kernels SIMD fundidos HF para os 4 gates
   (`gemm/gemv_4gate/*` + `math/lstm/gates.rs`), reusando os kernels exp-poly HF já existentes.
   Só avançar se a Fase 1 confirmar ganho perceptual/numérico relevante.
3. **Integra com I1:** nenhum novo controle — usa o `--activation hf` / `PARAM_ACTIVATION` de I1.
4. **Mais eficaz com oversampling (I5/§5):** HF + OS removem, juntos, o folding residual das
   ativações — documentar a sinergia para o modo "render offline".

### I6 — Impacto na arquitetura/documentação

- `architecture.md §2`: remover (ou atualizar) a "Known limitation" do LSTM quando entregue.
- `lstm_recurrent_drift.md §4/§7` e `audio_fidelity_map.md §3/§6/§9`: tornar I6 a **mitigação
  primária** de §3; rebaixar I4 (higiene) e I5 (anti-aliasing, não anti-drift).
- `cpp_parity_map.md §4.5`: nota sobre o caminho HF vs gate interop.

### I6 — Validação sugerida

- Oráculo f64: ESR do LSTM `1×16`/`2×8` em Padé vs HF (esperado: redução clara, aproximando-se
  do piso f16c ≈ 5e-5). ASR (`tests/spectral_fidelity.rs`) deve cair.
- `tests/isa_parity.rs`: paridade escalar↔SIMD do novo caminho HF do LSTM.
- `tests/cpp_parity.rs`: caracterizar (e, se preciso, recalibrar) o gate interop sob HF.

### I6 — Referências

- R1 (Sato & Smith 2025) — ASR e suavidade de ativação; R7 (Wright & Välimäki 2020) — peso
  perceptual; `docs/fastmath-approximations.md` — kernels HF existentes.

---

## Épicos (agrupamento para resolução otimizada)

> Agrupados por afinidade técnica, risco e dependência. A quebra em sprints/tarefas
> (`TODO-sprints.md`) só deve ser feita **quando solicitada** (skill `planejador-arquiteto`).

### Épico α — Controles de usuário de baixo risco (quick wins) [DONE]

- **Findings:** I1 (HF knob p/ WaveNet/ConvNet/Linear) + I2 (paridade de oversampling no standalone).
- **Objetivo:** expor o que **já existe e funciona**, com mínima superfície de risco. I1 é um
  store atômico (sem rebuild); I2 reaproveita o padrão _slimmable-rebuild_ e corrige um bug real
  (`--oversample` sem efeito).
- **Risco:** BAIXO. Não toca matemática de inferência. Validar com guards de zero-alloc.
- **Sequência sugerida:** I2 (corrigir init é ~2 linhas; depois runtime) → I1.

### Épico β — Fidelidade do §3 (LSTM), **reordenado pela evidência** [DONE]

- **Findings:** **I6 (primário — Padé→HF no LSTM)**, I4 (higiene, secundário), I5 (rejeitar
  caminho dedicado; caracterizar + documentar).
- **Objetivo:** atacar o **termo dominante e controlável** (ativação Padé) preservando o
  formato; corrigir a narrativa dos docs (I4/I5 não são as alavancas).
- **Risco:** MÉDIO-ALTO (toca kernels do LSTM). Fasear: escalar→SIMD; sempre validar contra
  oráculo f64 antes/depois; reavaliar o gate interop (`ABSOLUTE_ESR_CAP_LSTM`).
- **Dependência:** I6 herda o controle de I1 (fazer I1 antes).

### Épico γ — Resampler quality (decisão guiada por dados) [DONE]

- **Findings:** I3.
- **Objetivo:** **medir** (Δµs bench) e, muito provavelmente, **encerrar** como "HQ-only é o
  default permanente"; documentar que F3 já está resolvido. Implementar o seletor **só** se o
  bench provar custo material. **[Medição concluída: economia de 32T vs 64T é de apenas 40 ns por bloco (<0.1% do total), decidindo pelo descarte do seletor e manutenção de HQ-only permanente.]**
- **Risco:** BAIXO.

### Épico δ — Sincronização da documentação de fidelidade (a "arquitetura final") [DONE]

- **Findings:** transversal a I1–I6.
- **Objetivo:** deixar `audio_fidelity_map.md`, `lstm_recurrent_drift.md`, `architecture.md`,
  `clap_integration.md`, `cpp_parity_map.md` e `research-references.md` **consistentes** com as
  decisões acima (ver seção seguinte). Disparar a skill `documentador`.
- **Risco:** MÍNIMO (docs), mas **essencial** para não deixar mitos (I4/I5) cristalizados.

---

## Como ficará a arquitetura final (plano de documentação)

> Mapa do estado-alvo de cada decisão off-spec após os findings. Esta seção responde
> diretamente ao pedido de "planejar a devida documentação de como ficará a arquitetura final".

| Fator (audio_fidelity_map)         | Hoje                                  | Estado-alvo (pós-findings)                                                                                    | Finding  |
|:---------------------------------- |:------------------------------------- |:------------------------------------------------------------------------------------------------------------- |:-------- |
| §2/§6 Activation precision (HF)    | Infra pronta, **sem controle**        | **Exposto** via `--activation`/`PARAM_ACTIVATION=8` (WaveNet/ConvNet/Linear); LSTM coberto após I6.           | I1, I6   |
| §5 Oversampling (runtime switch)   | CLAP ✅ / standalone ✗ + CLI morto    | **Paridade CLAP↔standalone**; `--oversample` honrado no init e em runtime (PDC/F2 já ok).                     | I2       |
| §4 Resampler quality (32T/64T)     | HQ-only; F3 resolvido; seletor adiado | **Decisão documentada por bench**: provável HQ-only permanente; F3 marcado resolvido.                         | I3       |
| §3 LSTM drift — mitigação primária | Kahan head + OS state (propostos)     | **I6 (Padé→HF) é a alavanca**; Kahan = higiene; OS state = anti-aliasing (não anti-drift), c/ ressalva timbre | I4,I5,I6 |
| Referências                        | R1–R11                                | **+ Mikkonen & Werner 2025; + Carson et al. 2024/2025** (multirate RNN, RNN-ADAA).                            | I5       |

**Edições documentais previstas:**

- `docs/audio_fidelity_map.md`: §2/§6 (HF exposto), §4 (F3 resolvido + decisão de bench),
  §3 (reordenar mitigações), §5 (paridade de OS), e **§9 reescrita** (itens encerrados/movidos).
- `docs/lstm_recurrent_drift.md`: §4/§7 reescritos — I6 primário; I4 higiene; I5 vira
  "OS em LSTM = anti-aliasing, não anti-drift" com tabela empírica e ressalva de timbre.
- `docs/architecture.md`: §2 (controle de ativação + limitação LSTM até I6), §5.0O (paridade
  standalone de OS), §5 (F3/latência de fase mínima já empírica).
- `docs/clap_integration.md`: novo `PARAM_ACTIVATION=8` (e, se I3 prosseguir, quality em 9).
- `docs/cpp_parity_map.md`: ressalva do caminho HF vs `ABSOLUTE_ESR_CAP_LSTM`.
- `docs/research-references.md`: novas entradas (multirate RNN, RNN-ADAA) com rastreabilidade
  a I5/I6.

---

## Falsos pressupostos investigados e refutados (para evitar retrabalho)

1. **"F2/PDC bloqueia o runtime oversample switching."** _Refutado._ A latência do oversampling
   **já é reportada** (`events.rs:117`, `mod.rs:153`) e o CLAP **já troca em runtime**. O que
   falta é só paridade no standalone (I2).
2. **"F3 ainda superestima a latência de fase mínima."** _Refutado._ `latency_samples()` usa o
   **centroide empírico** do impulso de fase mínima (`sinc_kernel.rs:134-147,214-223`).
3. **"Kahan no head reduz o drift de §3 em 1–2 ordens."** _Refutado pelo `decompose_error`:_ a
   acumulação f32 já é ~7,2e-13; Kahan é higiene, não fix (I4).
4. **"Oversampled recurrent state reduz o drift de §3."** _Refutado:_ §3 é drift de
   quantização (f16c), não aliasing; e §3(b) mostra que **mais passos = mais drift**. Além
   disso, OS externo **muda o timbre** do LSTM (dependência de taxa) — I5.
5. **"HF já cobre todos os modelos."** _Refutado:_ os kernels fundidos do LSTM **bypassam** o
   HF (`math/lstm/gates.rs`); por isso I6.

---

## Rastreabilidade (Finding × §9 × código × docs)

| Finding | Item §9                       | Âncoras de código (principais)                                                               | Docs a atualizar                                    |
|:------- |:----------------------------- |:-------------------------------------------------------------------------------------------- |:--------------------------------------------------- |
| **I1**  | HF activation user control    | `math/activations/mod.rs:57-111`; `clap/extensions/params/*`; `standalone/cli.rs`            | audio_fidelity_map §6/§9; architecture §2; clap     |
| **I2**  | Runtime oversample switching  | `standalone/pw_host/rt_callback/commands.rs:108-135`; `capture/setup.rs:66`; `run.rs`        | audio_fidelity_map §5/§9; architecture §5.0O        |
| **I3**  | Resampler quality selector    | `dsp/sinc_kernel.rs:52,66,69`; `dsp/resampler.rs:51,128,547-561`; `benches/`                 | audio_fidelity_map §4/§9; architecture §5           |
| **I4**  | Kahan LSTM head               | `math/common/scalar_ref/dot.rs:66-75`; `math/common/kahan.rs`; `models/lstm/model1.rs:21-27` | lstm_recurrent_drift §7; architecture §6.2          |
| **I5**  | Oversampled recurrent state   | `dsp/oversample.rs`; `dsp/pipeline/stages/inference.rs:169-188`; `models/lstm/`              | lstm_recurrent_drift §7; audio_fidelity_map §5/§9   |
| **I6**  | (novo) HF nos kernels do LSTM | `math/lstm/gates.rs:29-69`; `models/lstm/layer_kernels.rs:248-266`; `gemm/gemv_4gate/*`      | architecture §2; lstm_recurrent_drift §4/§7; parity |

---

> **Próximo passo (sob solicitação):** transformar os Épicos α/β/γ/δ em `TODO-sprints.md`
> (tarefas atômicas, sequência, donos e gates de validação), conforme a skill
> `planejador-arquiteto`. Este documento **não** cria sprints por iniciativa própria.
