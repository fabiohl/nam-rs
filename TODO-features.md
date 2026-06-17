<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Mapa de Features do nam-rs — Aderência ao NeuralAmpModelerCore 🧩

> **Propósito**: este documento é o **mapeamento de recursos (features)** do `nam-rs` frente à
> implementação de referência **NeuralAmpModelerCore** — o que já existe, o que falta,
> importância, público real e diretrizes de implementação. Para **problemas/estranhezas de
> produto** (fidelidade, latência, denormais, gates frouxos), consulte **`TODO-problemas.md`**
> (achados "P"); para **otimização estrutural via internalização de dependências**
> ("hot-path": `half`, `rustfft`, `minstant`, `rtrb`), consulte **`TODO-optimize.md`**
> (achados "O").
>
> **Origem**: auditoria de aderência arquitetural do `nam-rs` contra a referência
> **NeuralAmpModelerCore v0.5.3** (commit pinado `9c7b185`, em
> `tests/fixtures/NeuralAmpModelerCore/`), cruzada com o panorama de modelos públicos do
> **TONE3000** (a maior biblioteca NAM do mundo, ~300 mil capturas, A2 como novo padrão desde
> mar/2026) e com o probe de carga verificado (abaixo).
>
> **Natureza**: os achados **F** são **lacunas de funcionalidade** — recursos **oficiais do
> NAMCore** (e/ou demandados pelo ecossistema real) ainda **não implementados** (ou apenas
> parciais) no `nam-rs`. Cada um é descrito com: _o que é_, _importância_, _público real atual_
> e _diretrizes de implementação_.
>
> **Filosofia (reafirmada)**: o `nam-rs` **não quer ser um porte** — quer **superar** o
> original. Mas a superação precisa ser **correta**: nenhum recurso oficial pode ser
> silenciosamente abandonado, e toda extensão deve preservar RT-safety (zero-alloc no
> hot-path, alocação só no load) e paridade auditável (ESR/SNR vs C++ v0.5.3).

---

## Sumário de Features

| ID      | Feature ausente / parcial                                                               | Impacto p/ Produto | Público real          | Esforço |
| ------- | --------------------------------------------------------------------------------------- |:------------------:| --------------------- |:-------:|
| **F1**  | **WaveNet genérico** (dispatcher dinâmico; qualquer CH/camadas/dilatações)              | 🔴 Crítico         | Todo o ecossistema A1 | Alto    |
| **F2**  | **Multi-condição / FiLM** (`condition_size > 1`, `condition_dsp`)                       | 🔴 Crítico (A2)    | Usuários A2 oficiais  | Alto    |
| **F3**  | **Motor A2 geral** (gating Gated/Blended, ativações heterogêneas, `head1x1`, `bn≠ch`)   | 🟠 Alto            | A2 avançado/futuro    | Alto    |
| **F4**  | **ConvNet** (arquitetura oficial `convnet.{cpp,h}`)                                     | 🟢 Baixo           | Nicho/legado          | Médio   |
| **F5**  | **SlimmableWavenet** (slicing dinâmico de canais por qualidade)                         | 🟠 Médio-Alto      | Pedais/embarcados     | Alto    |
| **F6**  | **Post-stack Head** (sub-objeto `head` multi-camada do WaveNet)                         | 🟡 Médio           | Modelos custom        | Médio   |
| **F7**  | **LSTM arbitrário** (`hidden_size`/`num_layers` fora do catálogo de 10 perfis)          | 🟠 Médio-Alto      | Capturas LSTM custom  | Médio   |
| **F8**  | **Biblioteca completa de ativações** (PReLU, SiLU, Hardswish, Softsign, LUT, fast-tanh) | 🟡 Médio           | A2 geral + custom     | Médio   |
| **F9**  | **Convoluções agrupadas/depthwise** (`groups > 1`)                                      | 🟡 Médio           | A2 geral + custom     | Médio   |
| **F10** | **Modelos multi-canal** (`in_channels`/`out_channels > 1`)                              | 🟢 Baixo           | Estéreo/experimental  | Médio   |
| **F11** | **Container aninhado** + cobertura `SlimmableContainer` real (modelo oficial)           | 🟢 Baixo           | Quality-scaling       | Baixo   |
| **F12** | **Tooling de pesquisa TONE3000 + expansão de `tests/fixtures/models/`** (panorama real) | 🔴 Habilitador     | Auditoria/golden      | Médio   |

> **Dependências**: F2 ⊃ F3 ⊃ {F8, F9} (FiLM destrava o motor A2 geral, que por sua vez
> exige gating + ativações + grouped conv). F1 é ortogonal e habilita o grosso do catálogo
> A1 custom. F12 é **pré-requisito de evidência** para todos: sem modelos reais variados em
> `tests/fixtures/models/`, não há como medir aderência nem promover goldens sintéticos→oficiais.

---

## Diagnóstico verificado (probe de carga em 2026-06-14)

Evidência empírica compartilhada por F1/F2/F3/F4/F6/F11. O dispatcher do `nam-rs` aceita hoje
**apenas um catálogo fixo** de topologias e **rejeita** o resto. Testando os modelos
**oficiais** do `NeuralAmpModelerCore_v0.5.3/example_models/`:

| Modelo oficial              | Geometria                               | Resultado no nam-rs                                             | Feature que destrava |
| --------------------------- | --------------------------------------- | --------------------------------------------------------------- | -------------------- |
| `wavenet_a1_standard.nam`   | WaveNet ch=16, cond=1 (real, 407 KB)    | ✅ **Carrega** (já é golden oficial)                            | —                    |
| `my_model.nam`              | == `wavenet_a1_standard` (md5 idêntico) | ✅ Carrega (redundante)                                         | —                    |
| `lstm.nam`                  | LSTM H=3, L=1                           | ✅ **Carrega** (já é golden oficial)                            | —                    |
| `wavenet.nam`               | WaveNet ch=3, cond=1, `[(3,2),(2,1)]`   | ❌ "topology not in catalog / dynamic fallback no longer avail."| **F1**               |
| `slimmable_wavenet.nam`     | WaveNet ch=3, cond=1, geometria livre   | ❌ "A2 shape not recognized"                                    | **F1 / F5**          |
| `wavenet_a2_max.nam`        | WaveNet ch=4, **cond=8** (FiLM)         | ❌ "only condition_size=1 is supported"                         | **F2 / F3**          |
| `wavenet_condition_dsp.nam` | WaveNet ch=3, **cond=3** (FiLM)         | ❌ "only condition_size=1 is supported"                         | **F2 / F3**          |
| `slimmable_container.nam`   | SlimmableContainer (3 submodelos)       | ❌ "submodel build failed" (depende dos acima)                  | **F5 / F11**         |

> **Leitura**: o engine de `SlimmableContainer` em si **já existe** (F11); `slimmable_container.nam`
> falha porque seus submodelos são `slimmable_wavenet` de geometria livre (F1/F5). Ou seja, as
> rejeições se concentram em **duas raízes**: geometria WaveNet fora do catálogo (F1) e
> condicionamento multi-canal/FiLM (F2). Resolver essas duas destrava a maioria dos ❌.

---

## F1 — 🔴 WaveNet genérico (dispatcher dinâmico) — 🟠 [EM ANDAMENTO]

> **Status (jun/2026):** **motor dinâmico pronto e auditado** (S2: `WaveNetModelDyn` +
> `layer_array_dyn`/`layer_dyn`/`dense_dyn`, scratch em heap suportando CH>16, born-SIMD;
> paridade bit-exata vs const-generic nos 4 SKUs — commits `7416b49`/`9ee0145`). Topologia
> generalizada para 3 vias `Known/Free/Rejected` (S2.T2.3, `e0e0685`). **Pendente o dispatch
> híbrido no loader (T3.1/S3)**: hoje geometria `Free` é detectada e validada, mas o build ainda
> a rejeita com mensagem clara ("pending T3.1"). `[DONE]` quando S3 fechar.

**O que é.** O NAMCore aceita **qualquer** geometria WaveNet (nº de camadas, canais,
dilatações, kernel, head) via `wavenet::create_config` (`wavenet/model.cpp:1239`,
`params.h`). O `nam-rs` substituiu isso por um **catálogo rígido** de 6 topologias estáticas
(Standard/Lite/Feather/Nano + A2-Full/A2-Lite) com const-generics; o _fallback dinâmico foi
removido_ e qualquer geometria fora do catálogo é **rejeitada** no load
(`src/loader/dispatcher/wavenet/mod.rs:133` — _"topology not in catalog and dynamic fallback
is no longer available"_). Confirmado pelo probe: `wavenet.nam` (ch=3, geometria livre) falha.

**Importância.** É o **caso de uso central** do ecossistema: a esmagadora maioria das
capturas A1 do TONE3000 (filtro `architecture=1` / `custom`) é treinada com geometria
arbitrária — não com os 6 SKUs canônicos. Sem F1, o `nam-rs` roda apenas modelos de
demonstração e capturas que por acaso casem com o catálogo.

**Público real.** Todos os usuários de modelos A1 custom (a base histórica do NAM, anterior
ao A2). Hoje, milhares de tones A1 no TONE3000.

**Diretrizes.**

- Reintroduzir um caminho **dinâmico** (geometria resolvida no load), **mantendo** o
  fast-path const-generic quando a geometria casar com um SKU conhecido (dispatch híbrido).
- **Toda alocação no load** (fora da audio-thread); hot-path permanece zero-alloc.
- Validar Rust↔C++ por **ESR/SNR** (não MSE absoluto) com `render` v0.5.3.
- Goldens conforme §Recomendações para gerar bons goldens (abaixo).

**🔧 Oportunidade de otimização (x86-64-v3) — ver `TODO-optimize.md §O5`.** Ao reintroduzir o
caminho dinâmico, o motor genérico deve usar os kernels `SimdMath` em **todo** passo
por-amostra/por-bloco e **não regredir para escalar** em operações element-wise (baseline é
x86-64-v3 → AVX2/FMA/F16C garantidos). O caminho WaveNet atual já tem uma lacuna a corrigir
junto: o ganho final `head_scale` é aplicado em laço escalar
(`src/models/wavenet/model.rs:96-98`) **dentro** de uma função `process_internal::<M: SimdMath>`
— deveria ser `M::apply_gain(out_slice, head_scale)` (kernel já existente). É evidente e
praticamente grátis.

**📋 Parecer revisor-auditor (jun/2026) — planejado em `TODO-sprints.md` (Épico E-WN).** Causa-raiz
confirmada: catálogo fechado (`mod.rs:133`, braço `None` em `mod.rs:108-142`), detecção rígida
(`topology.rs:84-119`) e remoção do caminho dinâmico no commit `d683b6e` (~1497 linhas). **`Conv1dDyn`
foi retido** (`src/models/wavenet/conv1d_dyn.rs`) e é a fundação reutilizável. O bloqueio estrutural é
`CH/K/HEAD` const-generic + scratch stack `[f32;1024]` que limita **CH≤16** (`layer.rs:45-56`); o motor
dinâmico precisa de **scratch em `AlignedVec` dimensionado no load**. Plano: **dispatch híbrido**
(fast-path const-generic para os SKUs + dinâmico para o resto), **nascendo SIMD** (guard-rail O5), com
paridade ESR/SNR vs C++ v0.5.3 (`model.cpp:828` aceita geometria livre/N arrays). **Escopo desta
rodada**: A1 geometria livre, **COND=1** (multi-cond é F2), **sem head pós-stack** (F6). A correção
O5/S3 do `head_scale` entra como quick-win. Sprints: **S2** (fundação dinâmica, 🔴 crítica), **S3**
(dispatch híbrido + goldens), **S1.T1.1** (`head_scale`→`M::apply_gain`).

---

## F2 — 🔴 Multi-condição / FiLM (`condition_size > 1`)

**O que é.** Conditioning arbitrário + **FiLM** (Feature-wise Linear Modulation,
`NAM/wavenet/film.h`) e `condition_dsp`. O `nam-rs` fixa `COND=1` como const-generic e
**rejeita** explicitamente multi-condição (`src/loader/nam_json/topology.rs:451` — _"only
condition_size=1 is supported. Multi-condition WaveNet is an official NAMCore feature not yet
implemented"_). As estruturas FiLM existem em `src/models/a2/film.rs` mas estão _"reservadas
p/ motor A2 geral (futuro)"_. Confirmado pelo probe: `wavenet_a2_max.nam` (cond=8) e
`wavenet_condition_dsp.nam` (cond=3) falham.

**Importância.** FiLM é o que destrava os modelos **A2 oficiais** que usam condicionamento
(`wavenet_a2_max.nam` cond=8, `wavenet_condition_dsp.nam` cond=3) e permitiria **elevar os
goldens A2 de sintético→oficial** (hoje os goldens A2 usam pesos sintéticos — ver
`tests/fixtures/README.md §wavenet_a2_full/lite`).

**Público real — ATENÇÃO (incerteza a resolver via F12).** É preciso **medir**, não supor:
a _fast-path_ A2 do NAMCore (`a2_fast.cpp`, `is_a2_shape`) é **sem FiLM** (CH=3/8,
LeakyReLU). Há forte indício de que os downloads A2 de produção do TONE3000 (que rodam em
chip de US$3) casam com essa fast-path — e portanto **já carregariam** no `nam-rs`. Os
modelos FiLM/cond seriam variantes de pesquisa (`*_max`, `*_condition_dsp`). **Conclusão só
com modelos A2 reais baixados (F12).** Se confirmado que produção = fast-path, F2 cai de
"crítico imediato" para "completude/futuro"; se houver FiLM em produção, F2 é bloqueante.

**Diretrizes.**

- Implementar `FiLM`, `GatingActivation`/`BlendingActivation` e `condition_dsp` espelhando
  `film.h`/`gating_activations.h`, com paridade escala-invariante (ESR/SNR).
- Converter os testes `test_loader_gap_wavenet_a2_max`/`_condition_dsp` de "afirmo que
  rejeita" → "afirmo que casa com o C++" (sem buraco de cobertura). Ver §Recomendações
  para gerar bons goldens (abaixo).

---

## F3 — 🟠 Motor A2 geral (além da fast-path)

**O que é.** Hoje o `nam-rs` só roda A2 na **fast-path** rígida (23 camadas, K=6/15,
LeakyReLU, `channels==bottleneck ∈ {3,8}`, sem gating/FiLM/`head1x1`). O NAMCore suporta
A2/WaveNet com **gating** (`GatingMode::{NONE,GATED,BLENDED}`), **`head1x1`/`layer1x1`**
configuráveis, **`bottleneck ≠ channels`**, e ativações heterogêneas por camada. No `nam-rs`
tudo isso está em `src/models/a2/{gating,params,activations}.rs` marcado _"reservado p/ motor
A2 geral (futuro)"_.

**Importância.** É o superconjunto que generaliza F2/F8/F9. Necessário para A2 não-canônico
e para "superar o original" com um motor A2 unificado (fast-path como caso especial otimizado).

**Público real.** Treinadores/experimentadores A2 e variantes futuras do TONE3000 que fujam
da fast-path canônica. Médio prazo.

**Diretrizes.** Construir um motor A2 _dinâmico_ (alocação no load) que detecte a fast-path e
faça downcast para o caminho const-generic SIMD; caso contrário, execute o caminho geral.
Reaproveitar F2 (FiLM/gating) e F8/F9 (ativações/grouped conv).

**🔧 Oportunidade de otimização (x86-64-v3) — ver `TODO-optimize.md §O5`.** Ao generalizar o motor
A2, vetorizar dois pontos hoje **escalares** na própria fast-path (baseline é x86-64-v3 → AVX2/FMA/
F16C garantidos; reduções FP **não** autovetorizam em Rust safe, logo continuam escalares no
binário):

- **Head conv** (`src/models/a2/head.rs:96-118`): laço aninhado `frames × 16 taps × CH`
  **totalmente escalar**, no caminho de **produção** de todo modelo A2, a cada bloco (para CH=8
  são 128 FMAs escalares por frame; `head.rs` não tem nenhum `_mm`/`target_feature`). Vetorizar
  AVX2/FMA: dot sobre os CH canais acumulado nos 16 taps, ou vetorização across-frames.
- **Rechannel Phase 0** (`src/models/a2/model/mod.rs:264-270`): broadcast-multiply escalar **e**
  redecodifica os mesmos `CH` pesos f16 **a cada frame** (são constantes). Decodificar uma única
  vez no load e usar broadcast-multiply AVX2 (sinergia com `TODO-optimize.md §O1` — internalização
  de `half`/F16C, que remove a chamada opaca que hoje **bloqueia** a autovetorização deste laço).

O motor A2 geral deve **nascer SIMD** nesses pontos, em vez de herdar o padrão escalar da fast-path.

---

## F4 — 🟢 ConvNet

**O que é.** Arquitetura oficial `ConvNet` (`NAM/convnet.{cpp,h}`, registrada como
`"ConvNet"` em `convnet.cpp:360`): blocos Conv1D (kernel=2) + BatchNorm opcional + ativação +
head denso. **Não implementada** no `nam-rs` (zero matches em `src/`); o dispatcher rejeita
com _"Unsupported architecture"_.

**Importância.** Baixa. É a arquitetura NAM mais antiga, hoje pouco usada; **não há modelo
oficial** entre os `example_models`, e o TONE3000 a classifica sob `custom` (volume marginal).

**Público real.** Nicho/legado — alguns modelos antigos da comunidade. Decisão de produto:
provavelmente **fora de escopo**, salvo demanda.

**Diretrizes (se entrar).** Implementar `ConvNetBlock` + `BatchNorm` (modo produção:
`y = x·scale + loc`) espelhando `convnet.cpp`; reusar Conv1D existente. Golden via `render`.

---

## F5 — 🟠 SlimmableWavenet (slicing dinâmico de canais)

**O que é.** WaveNet que reduz **dinamicamente** o nº de canais por array (`val ∈ [0,1]`)
extraindo subconjuntos de pesos — `NAM/wavenet/slimmable.{h,cpp}`. O `nam-rs` tem o **trait**
`SlimmableModel` e o `ContainerModel` (bundle de submodelos), mas o `SlimmableWavenet`
(slicing real de um único modelo) está **apenas planejado** (`src/models/slimmable.rs:17`).

**Importância.** Média-alta para o posicionamento "Linux + Pipewire + baixa latência" e para
**adaptive compute**: permite degradar qualidade↔CPU sem trocar de arquivo. Complementa o
`adaptive.rs` (que hoje só reduz camadas no WaveNet / faz passthrough no LSTM).

**Público real.** Cenários embarcados/ao vivo com orçamento de CPU variável (o mesmo público
que o A2-Lite atende). Diferencial competitivo do `nam-rs`.

**Diretrizes.** Implementar slicing por extração de pesos no **load/stage assíncrono**
(nunca na audio-thread); swap atômico de modelo (já há padrão SPSC GC). Exige `groups==1` e
head kernel=1 (como o C++). Integrar ao `SlimOverride`/`adaptive.rs`.

---

## F6 — 🟡 Post-stack Head (sub-objeto `head` do WaveNet)

**O que é.** Head multi-camada pós-stack (`detail::Head`, `HeadParams`), aplicado após os
layer-arrays com `head_scale`. O `nam-rs` **rejeita** (`topology.rs:464` — _"WaveNet 'head'
(post-stack sub-object) is not supported"_).

**Importância.** Média — alguns modelos custom (e potencialmente variantes A2) usam head
pós-stack. Relativamente contido.

**Público real.** Modelos custom que declaram `head`. Volume incerto (medir via F12).

**Diretrizes.** Implementar `Head` (Conv1D em cadeia + ativação) e somar ao `receptive_field`
de prewarm; já há gancho no fluxo de processamento.

---

## F7 — 🟠 LSTM arbitrário (fora do catálogo de 10 perfis)

**O que é.** O NAMCore aceita LSTM com **qualquer** `hidden_size`/`num_layers`
(`lstm.cpp`). O `nam-rs` só tem **10 perfis estáticos** (H ∈ {3,8,12,16,24,40}, L ∈ {1,2});
combinações fora disso falham no `build_lstm()`.

**Importância.** Média-alta — LSTM é uma das duas famílias mais usadas (e a de **maior
fidelidade** no `nam-rs`, ESR ~1e-7..1e-9; ver `TODO-problemas.md §P2`). Capturas LSTM com
H/L arbitrários são comuns.

**Público real.** Usuários de capturas LSTM custom (ex.: H=20, L=3). Hoje **rejeitados**.

**Diretrizes.** Caminho LSTM **dinâmico** (dimensões no load) preservando o caminho
const-generic SIMD como fast-path. Mesma filosofia de F1.

---

## F8 — 🟡 Biblioteca completa de ativações

**O que é.** O NAMCore oferece **12 ativações** (`activations.{cpp,h}`): Tanh, Hardtanh,
Fasttanh, ReLU, LeakyReLU, PReLU (per-channel), Sigmoid, SiLU/Swish, Hardswish,
LeakyHardtanh, Softsign — mais **toggle global fast-tanh** e **aceleração por LUT**. O
`nam-rs` exercita só Tanh (A1) e LeakyReLU (A2 fast-path); o resto está _"reservado p/ motor
A2 geral (futuro)"_ (`src/models/a2/activations.rs`).

**Importância.** Média; pré-requisito do motor A2 geral (F3) e de modelos custom que escolham
ativações alternativas.

**Público real.** A2 geral + modelos custom com ativações não-Tanh/LeakyReLU.

**Diretrizes.** Portar `ActivationConfig` (string **ou** objeto com params, ex.
`{"type":"LeakyReLU","negative_slope":0.2}`); aproveitar `src/math/activations/`
(já há `silu/relu/prelu/softsign/sigmoid/tanh` slice). Adicionar LUT opcional e o
toggle global de fast-tanh (interage com a política de fidelidade — `TODO-problemas.md §P2`).

---

## F9 — 🟡 Convoluções agrupadas / depthwise (`groups > 1`)

**O que é.** Conv1D/Conv1x1 com `groups>1` (inclui depthwise quando `groups==in==out`),
suportadas no NAMCore (`conv1d.cpp`, `dsp.cpp`). O `nam-rs` assume `groups==1` (rejeita o
resto na detecção de topologia).

**Importância.** Média; usado por geometrias eficientes (input mixin agrupado) e parte do
motor A2 geral.

**Público real.** Modelos custom/A2-geral com grouped conv. Médio prazo.

**Diretrizes.** Generalizar os kernels Conv1D/Conv1x1 para grupos; manter o fast-path
`groups==1` (caso dominante) intacto e SIMD-otimizado.

---

## F10 — 🟢 Modelos multi-canal (`in_channels`/`out_channels > 1`)

**O que é.** WaveNet/LSTM/ConvNet/Linear do NAMCore suportam múltiplos canais de entrada/
saída (default 1). O `nam-rs` é mono (1→1) em todos os modelos.

**Importância.** Baixa hoje — o ecossistema NAM é dominantemente mono (captura de gear).
Pode crescer com processamento estéreo/experimental.

**Público real.** Experimental/estéreo. Marginal atualmente.

**Diretrizes.** Avaliar caso a caso; o pipeline DSP já tem caminho estéreo (gate, gain), mas
os modelos assumem mono. Provável **fora de escopo** salvo demanda.

---

## F11 — 🟢 Container aninhado + cobertura `SlimmableContainer` real

**O que é.** O `nam-rs` implementa `SlimmableContainer` (`src/models/container.rs`,
crossfade 32 ms, até 8 submodelos) mas **rejeita aninhamento** (container-em-container,
`validation.rs:364`). Além disso, os goldens do container ainda dependem de submodelos
sintéticos (o probe mostra `slimmable_container.nam` oficial falhando por causa dos submodelos
`slimmable_wavenet` — depende de F1/F5).

**Importância.** Baixa — o aninhamento é raro; a feature em si já existe.

**Público real.** Quality-scaling avançado. Marginal.

**Diretrizes.** Decidir se aninhamento entra; priorizar **golden de `SlimmableContainer`
oficial** (via F12 + F1/F5) para validar o caminho com modelo real.

---

## F12 — 🔴 (Habilitador) TONE3000 e expansão de `tests/fixtures/models/` [DONE]

**O que:** **Pesquisar o TONE3000** e carregar o diretório
`/home/fabio/Cloud/Guias/Softwares/nam_t3k/` com modelos válidos faltantes.

**Por que é habilitador.** É **pré-requisito de evidência** para F1–F11: hoje a auditoria de
aderência depende de modelos sintéticos (A2-Full/Lite, Lite CH=12) e de poucos modelos reais.
Sem um corpus real e variado, não há como (a) **medir** a aderência ao NAMCore, (b) resolver
a **incerteza de F2** (produção A2 usa FiLM?), nem (c) **promover goldens sintéticos→oficiais**.

**Arquiteturas mais urgentes:**

> Use utils/check-model.py para testar arquivos de downloads
> Devido à falta de suporte, alguns estão apaarecendo como "Unknown (SlimmableContainer)", "Unsupported architecture". Isto deve se resolver progressivamente.

| Perfil do Modelo              | Feature/Problema Alvo               | Critérios de Busca (Campos JSON do `.nam`)                                                                                   | Finalidade nos Testes                                                                                                          |
| ----------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Real WaveNet Lite (CH=12)** | **P1** (Divergência Lite) / **F12** | `architecture: "WaveNet"`, `channels: 12`, `sample_rate` presente, pesos reais treinados.                                    | Substituir o modelo sintético `BossWN-lite.nam` e validar a arquitetura Lite com um sinal de áudio real sem o drift de 0.9 dB. |
| **A1 Custom Geometry**        | **F1** (WaveNet Genérico)           | `architecture: "WaveNet"`, com canais diferentes de {4, 8, 12, 16} ou com arrays de dilatação customizados.                  | Testar o dispatcher dinâmico de geometrias WaveNet livres quando F1 for implementada (fixture negativa que vira positiva).     |
| **A2 FiLM / Multi-Condição**  | **F2** (FiLM) / **F3** (A2 Geral)   | `architecture: "WaveNet"`, A2-shape, com `condition_size > 1` (ex: 3 ou 8) e/ou `gating_mode` diferente de `"none"`.         | Validar suporte a FiLM e gating de multi-condição do motor A2. Promover `wavenet_a2_max.nam` de fixture de erro para positiva. |
| **LSTM Custom Shape**         | **F7** (LSTM arbitrário)            | `architecture: "LSTM"`, com dimensões diferentes do catálogo estático de 10 perfis (ex: `hidden_size: 20`, `num_layers: 3`). | Validar o loader dinâmico de LSTM genérico com dimensões customizadas.                                                         |
| **Real Slimmable Container**  | **F5** (Slimmable) / **F11**        | `architecture: "SlimmableContainer"`, contendo submodelos reais funcionais.                                                  | Validar a transição CPU-degradada A2-Full -> A2-Lite em produção real com pesos reais de amplificador.                         |

**Nota de encerramento pelo PO**: Estamos bem cobertos de modelos A2 (que declaram se Full/Lite em um mesmo arquivo, que imagino atender o **Real Slimmable Container**) de qualidade em `tests/fixtures/models-nondist`, que é o que a comunidadevem disponibilizando no Tone3000.
Além dos já tradicionais e confiáveis catalogados em `tests/fixtures/README.md` (seção "Model Files and Trust Levels Registry").
Por enquanto está inviável encontrar topologias mais customizadas, já que a busca do tone3000 inviabiliza isto.
Já encontramos uma A1 Lite e mais alguns customizados - o que pode ser considerado uma vitória.
Vamos ter de ir nos virando com o que temos.

---

## Recomendações para gerar bons goldens destas features (quando forem implementadas)

Para manter o padrão exemplar já estabelecido, os goldens das novas features (sobretudo
F1/F2/F3 e o corpus de F12) devem seguir as mesmas regras:

- **Modelo oficial real, não sintético.** Usar diretamente os `.nam` oficiais que hoje falham —
  `wavenet.nam` (WaveNet genérico) e `wavenet_a2_max.nam` (FiLM) — como fonte. Isso **promove o
  A2 de sintético→oficial** e elimina a única ressalva de proveniência que resta nos goldens.
- **Cross-reference C++ pinado, gate scale-invariant.** Gerar a referência com o `render` do
  **v0.5.3** (mesmo pipeline do `golden_gen_build.sh`); validar Rust↔C++ por **ESR/SNR** (não
  MSE absoluto), como no A2 atual.
- **Conversão de `test_loader_gap_*` → golden positivo.** Cada modelo que hoje tem um teste de
  "rejeição" (`test_loader_gap_wavenet_a2_max`, `_condition_dsp`, `_slimmable_wavenet`,
  `_slimmable_container`) deve, ao ganhar suporte, **migrar** de "afirmo que rejeita" para
  "afirmo que casa com o C++" — sem deixar buraco de cobertura.
- **Calibração por medição documentada.** Registrar a entrada em `get_calibrated_threshold` com
  `// Measured: SNR=…, ESR=…` e margem (6–10 dB), como exige o meta-teste anti-placebo (T4.4).
- **Regime de amplitude realista.** Como os modelos oficiais são treinados, a saída já fica em
  faixa de áudio sã (pico ~0.1–0.5, LUFS ~−20) — o gate de plausibilidade LUFS (T4.3) passa
  naturalmente, **sem** o reescalonamento artificial que o A2 sintético exigiu (T2.5).
- **Cobertura multi-SR e determinismo.** Acrescentar os novos modelos aos gates v2 multi-SR
  (T4.2) e ao gate de determinismo bitwise por arquitetura (T4.5), fechando a malha de
  cobertura universal.

---

## Modelo de encomenda ao Claude

/revisor-auditor Vamos agora atacar os problemas identificados nos nossos arquivos "TODO". Começando por este aqui:
TODO-features.md:82-82

```cite
F1 — 🔴 WaveNet genérico (dispatcher dinâmico)
```

Transversalmente vamos pegar conhecimentos também destes achados aqui:
TODO-problemas.md:75-75

```cite
P2 — 🟠 Fidelidade da família WaveNet vs C++ é muito inferior à de LSTM/Linear
```

TODO-problemas.md:101-101

```cite
P3 — 🟠 Gates de golden frouxos em alguns cenários (guardião fraco onde mais importa)
```

TODO-problemas.md:127-127

```cite
P4 — 🟡 WaveNet não é "silencioso no silêncio"
```

TODO-optimize.md:264-264

```cite
O5 — 🟢 Auditoria de cobertura SIMD x86-64-v3 no hot-spot (índice + guard-rail)
```

Pesquise a fundo estes achados e entenda a situação. Use o NAMcore como a "bíblia de correção", mas já vá buscando otimizações e SIMD/ISAs modernos seguros on-the-fly.
Ao final acione a skill "planejador-arquiteto" para organizar isto em tarefas seguras e rápidas para entregar valor. Todas super detalhadas e precisas para uma execução segura.

---

## Nota de método (para o `planejador-arquiteto`, quando acionado)

- **Não** transformar estes F em sprints ainda (instrução do PO). Este documento é o **insumo**.
- Ao planejar: respeitar as **dependências** (F2⊃F3⊃{F8,F9}; F12 como habilitador transversal),
  a **RT-safety** (alocação só no load) e a regra de golden _"todo golden deve poder falhar"_.
- F12 deve vir cedo: sem corpus real, F1/F2/F3/F6/F7 não têm como ser **medidos** nem promovidos.
