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
| **F13** | **Migração de formato legado** (Keras `.json`, faixa de versão `.nam` 0.5.0–0.7.0)      | 🟢 Baixo           | Modelos antigos       | Médio   |

> **Dependências**: F2 ⊃ F3 ⊃ {F8, F9} (FiLM destrava o motor A2 geral, que por sua vez
> exige gating + ativações + grouped conv). F1 é ortogonal e habilita o grosso do catálogo
> A1 custom. F12 é **pré-requisito de evidência** para todos: sem modelos reais variados em
> `tests/fixtures/models/`, não há como medir aderência nem promover goldens sintéticos→oficiais.

---

## Diagnóstico verificado (probe de carga em 2026-06-14)

Evidência empírica compartilhada por F1/F2/F3/F4/F6/F11. O dispatcher do `nam-rs` aceita hoje
**apenas um catálogo fixo** de topologias e **rejeita** o resto. Testando os modelos
**oficiais** do `NeuralAmpModelerCore_v0.5.3/example_models/`:

| Modelo oficial              | Geometria                               | Resultado no nam-rs                                              | Feature que destrava |
| --------------------------- | --------------------------------------- | ---------------------------------------------------------------- | -------------------- |
| `wavenet_a1_standard.nam`   | WaveNet ch=16, cond=1 (real, 407 KB)    | ✅ **Carrega** (já é golden oficial)                             | —                    |
| `my_model.nam`              | == `wavenet_a1_standard` (md5 idêntico) | ✅ Carrega (redundante)                                          | —                    |
| `lstm.nam`                  | LSTM H=3, L=1                           | ✅ **Carrega** (já é golden oficial)                             | —                    |
| `wavenet.nam`               | WaveNet ch=3, cond=1, `[(3,2),(2,1)]`   | ❌ "topology not in catalog / dynamic fallback no longer avail." | **F1**               |
| `slimmable_wavenet.nam`     | WaveNet ch=3, cond=1, geometria livre   | ❌ "A2 shape not recognized"                                     | **F1 / F5**          |
| `wavenet_a2_max.nam`        | WaveNet ch=4, **cond=8** (FiLM)         | ❌ "only condition_size=1 is supported"                          | **F2 / F3**          |
| `wavenet_condition_dsp.nam` | WaveNet ch=3, **cond=3** (FiLM)         | ❌ "only condition_size=1 is supported"                          | **F2 / F3**          |
| `slimmable_container.nam`   | SlimmableContainer (3 submodelos)       | ❌ "submodel build failed" (depende dos acima)                   | **F5 / F11**         |

> **Leitura**: o engine de `SlimmableContainer` em si **já existe** (F11); `slimmable_container.nam`
> falha porque seus submodelos são `slimmable_wavenet` de geometria livre (F1/F5). Ou seja, as
> rejeições se concentram em **duas raízes**: geometria WaveNet fora do catálogo (F1) e
> condicionamento multi-canal/FiLM (F2). Resolver essas duas destrava a maioria dos ❌.

---

## F1 — 🔴 WaveNet genérico (dispatcher dinâmico)

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

## F12 — 🔴 (Habilitador) Tooling de pesquisa TONE3000 + expansão de `tests/fixtures/models/`

**O que é.** Mecanismo para **pesquisar o TONE3000 via API oficial** e compor um diretório
`tests/fixtures/models/` **completo, variado e documentado** (origem + finalidade), com
ênfase em **A2**. Entregue nesta auditoria: **`tests/fixtures/tone3000_research.py`** — CLI
(`survey`/`acquire`) que consulta a API v1, ranqueia candidatos por arquitetura/tamanho e
baixa fixtures redistribuíveis com manifesto de proveniência.

**Por que é habilitador.** É **pré-requisito de evidência** para F1–F11: hoje a auditoria de
aderência depende de modelos sintéticos (A2-Full/Lite, Lite CH=12) e de poucos modelos reais.
Sem um corpus real e variado, não há como (a) **medir** a aderência ao NAMCore, (b) resolver
a **incerteza de F2** (produção A2 usa FiLM?), nem (c) **promover goldens sintéticos→oficiais**.

**Realidade da API TONE3000 (apurada).**

- REST v1, **OAuth 2.0 + PKCE**, **autenticado** (chave `t3k_pub_…` + token de usuário). **Não
  há crawl anônimo nem dump público.** Rate limit 100 req/min; `/tones/search` é _fortemente_
  limitado (o próprio TONE3000 recomenda o fluxo `Select` para navegação).
- Endpoints úteis: `GET /tones/search?platform=nam&architecture={1|2|custom}&sizes=…&sort=downloads-all-time`,
  `GET /tones/{id}`, `GET /models?tone_id=…&architecture=…` (cada modelo traz `model_url`
  pré-assinado, `size` e `architecture_version`).
- **Não existe campo de "avaliação" (estrelas).** Métricas expostas: `downloads_count` e
  `favorites_count`. Usamos **favorites como proxy de aprovação** e ranqueamos por
  `score = downloads + 25·favorites` (transparente, auditável, **não** é métrica oficial).
  ESR por modelo aparece na página web do tone (ex.: A2-Full ESR 0.0005) — útil como sinal
  de qualidade de captura.
- **Mapeamento arquitetura.** `1`=A1 (WaveNet clássico + LSTM/Linear; sizes
  standard/lite/feather/nano ↔ CH=16/12/8/4), `2`=A2 (um único `.nam` roda como A2-Full **ou**
  A2-Lite), `custom`=geometrias arbitrárias (fixtures negativas para F1/F3/F7).

**⚠️ Restrição de LICENÇA (bloqueante para commit).** Cada tone tem licença
(`t3k`, `cc-by`, `cc-by-sa`, `cc0`, …). A **licença padrão `T3K` proíbe redistribuir** o
arquivo sem permissão do autor. **Só podem ser versionados** em `tests/fixtures/models/`
modelos **CC0/CC-BY(-SA)** ou com permissão explícita. O script aplica
`--redistributable-only` para nunca propor um fixture que não possamos vendorizar legalmente.
Alternativa: documentar o `tone_id` + comando de download (modelo **não** commitado), como já
se faz com mirrors gitignored.

**Sugestão de 1 modelo por arquitetura (metodologia + candidatos a verificar).**
A seleção final exige rodar o `survey` autenticado (números ao vivo + licença). Como guia,
candidatos de **alta tração** observados publicamente no TONE3000 (verificar downloads,
favorites e **licença** antes de vendorizar):

| Arquitetura (filtro)               | Critério                                 | Candidato observado (verificar licença/números) |
| ---------------------------------- | ---------------------------------------- | ----------------------------------------------- |
| **A2** (`architecture=2`)          | Maior tração + ESR baixo + clean de refª | _"Fender Deluxe Reverb (A2)"_ (~10k downloads)  |
| **A2** high-gain                   | Cobrir regime saturado/transientes       | _"Bogner Uberschall MKII Ultra High Gain A2"_   |
| **A1 Standard** (`size=standard`)  | Capturas A1 clássicas de alto download   | _"1980 Marshall JMP 2204 (EL34)" pack_          |
| **A1 Lite/Feather/Nano**           | Validar SKUs menores com modelo **real** | (resolver via `survey --architecture 1`)        |
| **custom** (`architecture=custom`) | Geometria fora do catálogo (F1/F3/F7)    | (resolver via `survey --architecture custom`)   |

> Objetivo do corpus: ≥1 modelo **real** por SKU A1 (eliminar o sintético Lite CH=12 — ver
> `TODO-problemas.md §P1`), ≥1 A2 clean + ≥1 A2 high-gain reais, e ≥1 `custom` por classe de
> geometria não suportada (fixture negativa que vira positiva quando F1/F3/F7 entrarem). Tudo
> com proveniência (tone_id, autor, licença, downloads/favorites, finalidade) no `README.md`
> dos fixtures.

**Diretrizes.**

1. Gerar token OAuth (PKCE) com uma conta TONE3000; rodar `survey` por arquitetura.
2. Filtrar por licença redistribuível; baixar via `acquire` + manifesto de proveniência.
3. **Probe-load** cada `.nam` no `nam-rs` para classificar: _carrega_ (fast-path/catálogo) vs
   _rejeitado_ (qual feature destrava: F1/F2/F3/F6/F7/F9). Isso **quantifica** o gap real.
4. Para os que carregam, gerar golden via `render` v0.5.3 e adicionar aos gates multi-SR.
5. Documentar tudo em `tests/fixtures/README.md` (origem + finalidade + licença).

---

## F13 — 🟢 Migração de formato legado / faixa de versão `.nam`

**O que é.** (a) Formato **Keras legado** (`.json` com `in_shape`/`layers`, ex.
`tests/fixtures/unsupported/tw40_blues_deluxe_deerinkstudios.json`) — não digerido pelo
`nam-rs`. (b) O NAMCore declara suporte à **faixa de versão de arquivo** 0.5.0–0.7.0
(`get_dsp.h`); convém confirmar que a cobertura de versão do `nam-rs` acompanha (incl.
campos de metadata: `loudness`, `input_level_dbu`, `output_level_dbu`).

**Importância.** Baixa — formato legado é raro; a maioria já está em `.nam` moderno.

**Público real.** Donos de exports antigos (pré-`.nam`). Marginal.

**Diretrizes.** Decidir se um conversor de formato entra no escopo (provável **fora**, salvo
demanda). Quanto à faixa de versão: adicionar testes de carga cobrindo 0.5.0…0.7.0 e os
campos de metadata, garantindo paridade de calibração (input/output level, loudness).

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

## Nota de método (para o `planejador-arquiteto`, quando acionado)

- **Não** transformar estes F em sprints ainda (instrução do PO). Este documento é o **insumo**.
- Ao planejar: respeitar as **dependências** (F2⊃F3⊃{F8,F9}; F12 como habilitador transversal),
  a **RT-safety** (alocação só no load) e a regra de golden _"todo golden deve poder falhar"_.
- F12 deve vir cedo: sem corpus real, F1/F2/F3/F6/F7 não têm como ser **medidos** nem promovidos.
