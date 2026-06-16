<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Plano de Execução 🗺️

> **Origem**: auditoria `revisor-auditor` (jun/2026) sobre os achados **F1** (`TODO-features.md:82`),
> **P2/P3/P4** (`TODO-problemas.md:75/101/127`) e **O5** (`TODO-optimize.md:264`), cruzada com a
> "bíblia de correção" **NeuralAmpModelerCore v0.5.3** (`tests/fixtures/NeuralAmpModelerCore/`).
> Transformação em sprints/tarefas pela skill `planejador-arquiteto`.
>
> **Tema central**: destravar o **WaveNet genérico** (qualquer geometria) — o caso de uso central
> do ecossistema A1 — sem regredir RT-safety, sem regredir fidelidade e **nascendo SIMD**
> (guard-rail x86-64-v3). Os achados de qualidade (P2/P3/P4) e otimização (O5) são tecidos como
> **camada de auditoria** que o motor genérico precisa satisfazer.
>
> **Regras inegociáveis** (`.agents/rules/`): RT-safety (alocação só no load; zero heap-drop, zero
> bloqueio, zero `unwrap()` no hot-path), copyright SPDX em todo arquivo tocado, testes
> inline (<300 linhas) ou `_test.rs` (≥300), goldens scale-invariant (ESR/SNR) e a regra de ouro
> _"todo golden deve poder falhar"_. Fechar cada tarefa com `cargo check`/`cargo test`/`cargo bench`
> (quando houver meta de perf) e **zero warnings**.

---

## Diagnóstico consolidado (insumo das sprints)

### Por que o WaveNet genérico está bloqueado (F1)

O dispatcher foi reduzido a um **catálogo fechado** de 6 topologias monomorfizadas via const-generics.
Qualquer geometria fora do catálogo é **rejeitada** no load:

- **Rejeição**: `src/loader/dispatcher/wavenet/mod.rs:133` (_"dynamic fallback is no longer available"_),
  no braço `None` do match (`mod.rs:108-142`).
- **Detecção rígida**: `get_wavenet_topology` exige **exatamente 2 layer-arrays** e casa
  `channels`+`dilations` contra 4 padrões fixos (`src/loader/nam_json/topology.rs:84-119`).
- **Caminho dinâmico removido**: commit `d683b6e` (10/jun/2026) deletou ~1497 linhas
  (`dispatcher/wavenet/dynamic.rs`, `models/wavenet/model_dyn.rs`, `layer_dyn.rs`, `dense_dyn.rs`).
  **`Conv1dDyn` foi retido** (`src/models/wavenet/conv1d_dyn.rs`, +`_dual`/`_kernels`) — é a fundação
  reutilizável.
- **Restrição estrutural**: `CH`, `K`, `HEAD` são **const-generics** (compile-time); o scratch de
  ativação é **stack `[f32;1024]`** com `const assert!(CH*64<=1024)` → **CH≤16**
  (`src/models/wavenet/layer.rs:45-56`). Um motor genérico precisa de **scratch em heap alinhado
  (`AlignedVec`) dimensionado no load** e dimensões em runtime.
- **Referência C++**: aceita N layer-arrays, K/CH/HEAD/dilatações arbitrários via `parse_config_json`
  (`wavenet/model.cpp:828`) iterando `std::vector` (`detail::LayerArray`, `model.cpp:380-511`). É o
  alvo de paridade.

> **Escopo de F1 nesta rodada**: geometria A1 livre com **COND=1** (multi-condição/FiLM é **F2**,
> fora de escopo) e **sem head pós-stack** (é **F6**, continua rejeitado). Hybrid-dispatch: o
> fast-path const-generic continua atendendo os SKUs conhecidos; o dinâmico cobre o resto.

### Achados de qualidade tecidos em F1

- **O5/S3 (trivial, grátis)**: `head_scale` aplicado em laço escalar dentro de função
  `process_internal::<M: SimdMath>` (`src/models/wavenet/model.rs:96-98`). O kernel já existe:
  `M::apply_gain` (`src/math/common/traits.rs:447`). Substituição é bit-exata.
- **O5 (limpeza)**: cabeamento BF16 escalar **inalcançável** no AVX2
  (`src/math/common/avx2_impl.rs:42-99`) — `is_bf16` só é `true` em `Avx512VnniBf16`. Código morto.
- **O5 (guard-rail)**: com x86-64-v3 garantido, **nenhum laço element-wise/redução f32/f16
  por-amostra/por-bloco** pode nascer escalar no caminho dinâmico. Reduções FP **não** autovetorizam
  em Rust safe — então o motor genérico deve usar `SimdMath` desde a primeira linha.
- **P2 (assimetria de fidelidade)**: WaveNet ESR ~6e-3 (~0,3–1%) vs LSTM/Linear ~1e-7..1e-9. Causa
  **por design**: aproximações FastMath (tanh Padé [5,4], err ~2,3e-3 — `src/math/activations/tanh/production.rs`)
  **somadas** à quantização de pesos BF16/F16 (err ~3,9e-3 por elemento — fonte **dominante**, ver
  `docs/fastmath-approximations.md`). Não há modo "exato". O motor genérico **não pode piorar** isso.
- **P3 (gates frouxos)**: thresholds calibrados em 12–60 dB SNR (`tests/common/validation.rs:339`),
  com meta-teste anti-placebo (`tests/threshold_calibration.rs:199`). Regressão genuína pode passar
  onde a fidelidade já é menor. F1 adiciona **novas** geometrias → exige gate honesto e que **possa
  falhar**.
- **P4 (silêncio não-zero)**: WaveNet produz ~3,6e-5 (−89 dBFS) no silêncio; A2 produz 0
  (`tests/soak_test.rs:60`). **Achado-chave do auditor**: isto é **fiel ao C++**, não bug — os
  biases (conv/mixin/1x1/rechannel) tornam `tanh(bias)≠0`; o próprio NAMCore documenta _"don't expect
  the model to be outputting zeroes"_ (`dsp.h:67`). **Zerar à força divergiria da bíblia.** A2 só
  zera por usar LeakyReLU(0)=0 + pesos sintéticos. DAZ/FTZ já ativo (`src/math/common/ops.rs:163`,
  reasserção em `src/clap/processor/mod.rs:268`).

---

## Mapa de Épicos e Sprints

| Épico    | Sprint | Foco                                                                   | Risco    | Entrega de valor   |
| -------- | ------ | ---------------------------------------------------------------------- |:--------:|:------------------:|
| **E-WN** | **S1** | Quick-wins de hot-path + auditoria de fidelidade (sem tocar topologia) | 🟢 Baixo | Imediato           |
| **E-WN** | **S2** | Fundação do **motor WaveNet dinâmico** (RT-safe, born-SIMD)            | 🔴 Alto  | Médio              |
| **E-WN** | **S3** | Dispatch híbrido + paridade C++ + goldens (P3 endurecido)              | 🟠 Médio | Alto (destrava F1) |
| **E-WN** | **S4** | (Opcional/deferida) Política de fidelidade WaveNet — modo "exato" (P2) | 🟡 Médio | Diferencial        |

> **Ordem de dependência**: S1 é independente e entrega valor já. S2 é o coração de F1 (risco alto).
> S3 depende de S2 (precisa do motor pronto para validar). S4 é opcional e pode vir depois, em
> paralelo a outras features. P4 é resolvido em S1 (diagnóstico+documentação). O5/S3 e O5-limpeza
> são micro-tarefas em S1. O guard-rail O5 é **regra de revisão** de S2/S3.

---

## ÉPICO E-WN — WaveNet Genérico, Fiel e Vetorizado

**Objetivo macro**: reintroduzir o suporte a **qualquer geometria WaveNet A1** (canais/kernel/head/
dilatações/nº de camadas livres), preservando o fast-path const-generic como caso especial otimizado,
**nascendo SIMD** e sem regredir fidelidade, RT-safety ou determinismo. Espelhar o C++ v0.5.3 como
referência e validar por ESR/SNR.

---

### 🟢 Sprint S1 — Quick-wins de hot-path e auditoria de fidelidade

> **Por quê primeiro**: entrega valor **imediato** e de **baixo risco**, sem tocar na topologia, e
> prepara o terreno (instrumentação de fidelidade, política de silêncio) para julgar o motor dinâmico
> de S2/S3. Todas as tarefas são bit-exatas ou puramente diagnósticas/documentais.

#### T1.1 — [O5/S3] Vetorizar `head_scale` (laço escalar → `M::apply_gain`) 🟢

- **Onde**: `src/models/wavenet/model.rs:96-98` (laço escalar `out_slice[i] = array2_head[i] * head_scale`),
  dentro de `process_internal::<M: SimdMath>`.
- **O quê**: substituir o laço por `M::apply_gain(out_slice, self.head_scale)` — o kernel já existe
  (`src/math/common/traits.rs:447`; impls AVX2 `src/math/common/avx2_impl.rs:525`, AVX-512 em
  `src/math/common/avx512/dsp/base.rs:91`). O `array2_head` é o próprio `head_outputs[0..num_frames]`;
  copiar para `out_slice` e aplicar ganho in-place (ou usar variante que escreve no destino).
- **Critério de aceite**: bit-exato vs baseline (ou ESR inalterado); `cargo test` verde; micro-bench
  WaveNet sem regressão (idealmente leve ganho no head). Remove o último laço escalar do `model.rs`.
- **Risco**: 🟢 trivial. **Atenção**: garantir semântica idêntica quando `num_frames` não é múltiplo
  do width SIMD (tail handling já é responsabilidade do kernel).

#### T1.2 — [O5 limpeza] Remover cabeamento BF16 escalar morto no AVX2 🟢

- **Onde**: `src/math/common/avx2_impl.rs:42-99` (`dot_product_bf16*`/`gemv_*_bf16` cabeados a
  fallbacks escalares).
- **O quê**: `is_bf16` só é `true` em `Avx512VnniBf16` (confirmado em
  `loader/dispatcher/lstm/static_builder.rs:28`, `wavenet/layout.rs:26`, `a2/model/set_weights.rs:41`);
  no AVX2 os pesos são F16 (com F16C SIMD). Logo esses fallbacks são **inalcançáveis** → remover ou
  documentar como `unreachable!`/`debug_assert!`. **Sem mudança numérica de runtime.**
- **Critério de aceite**: `cargo build`/`cargo test` verdes; zero warnings; nenhuma mudança numérica
  em nenhum golden. Diff puramente de remoção/documentação.
- **Risco**: 🟢 baixo. Confirmar que nenhum teste exercita propositalmente esse caminho antes de remover.

#### T1.3 — [P4] Diagnóstico e política do "silêncio não-zero" do WaveNet 🟢

- **Onde**: `tests/soak_test.rs:60` (`test_wavenet_silence_soak`); biases em
  `src/models/wavenet/{dense.rs,conv1d.rs,layer.rs}`, `src/loader/dispatcher/wavenet/bias_tune.rs`.
- **O quê** (investigação → decisão documentada):
  1. **Confirmar a fonte**: decompor o resíduo (~3,6e-5) — é `tanh(bias)` propagado (esperado) e/ou
     erro de quantização BF16/F16 e/ou denormal residual? Instrumentar um teste de decomposição.
  2. **Confirmar paridade com C++**: rodar `render` v0.5.3 no mesmo modelo com entrada de silêncio e
     verificar que o C++ **também** não zera (NAMCore documenta isso em `dsp.h:67`). **Se confirmado,
     NÃO zerar à força** — zerar divergiria da bíblia e quebraria paridade.
  3. **Decisão**: documentar formalmente (em `docs/` e/ou nota no teste) que o resíduo −89 dBFS é
     **comportamento fiel** ao NAMCore; registrar a interação com noise-gate/true-bypass (cabe à
     camada de gate, não ao modelo). Confirmar DAZ/FTZ cobrindo o hot-path WaveNet (liga-se a P5).
- **Critério de aceite**: teste de decomposição + comentário `// Measured:` com a origem do resíduo;
  evidência de paridade C++ (mesmo sinal de silêncio); doc atualizada. **Nenhuma** alteração que zere
  o silêncio à força (a menos que se prove que o C++ também zera — não é o caso esperado).
- **Risco**: 🟢 baixo (diagnóstico+doc). **Armadilha a evitar**: "consertar" zerando a saída.

#### T1.4 — [P2] Instrumentar e quantificar as fontes de drift da família WaveNet 🟢

- **Onde**: `docs/fastmath-approximations.md`; `src/math/activations/tanh/production.rs` (Padé [5,4]);
  caminho de quantização (`src/math/common/ops.rs:16` `quantize_weight`).
- **O quê**: medir, de forma isolada e reprodutível, a contribuição de cada fonte ao ESR WaveNet:
  (a) quantização BF16/F16 dos pesos (~3,9e-3, **dominante**), (b) tanh Padé (~2,3e-3), (c) acumulação
  f32. Gerar uma tabela `// Measured:` por arquitetura. **Sem alterar o engine** — é base de decisão
  para S4.
- **Critério de aceite**: relatório/teste `#[ignore]` que imprime a decomposição do ESR; doc
  `docs/fastmath-approximations.md` atualizada com a tabela e a recomendação (P2). Não altera números
  de produção.
- **Risco**: 🟢 baixo (medição). Conecta com S4 (modo exato opcional).

---

### 🔴 Sprint S2 — Fundação do motor WaveNet dinâmico (RT-safe, born-SIMD)

> **CRÍTICA / ALTO RISCO**: é o coração de F1 e reintroduz código removido em `d683b6e`. Toda
> alocação **no load**; hot-path zero-alloc/zero-lock/zero-panic (auditável com `heap-audit`). O motor
> genérico **nasce SIMD** (guard-rail O5) — proibido laço element-wise escalar.

#### T2.1 — Tipos dinâmicos do WaveNet (alloc no load, scratch em heap) 🔴

- **Onde**: `src/models/wavenet/` (reintroduzir, modernizados: `dense_dyn.rs`, `layer_dyn.rs`,
  `layer_array_dyn.rs`, `model_dyn.rs`); reaproveitar `conv1d_dyn.rs` (já existe, +`_dual`/`_kernels`).
- **O quê**: estruturas com dimensões em **runtime** (`ch`, `k`, `head`, `dilations: Vec`, N
  layer-arrays). **Diferença essencial vs o código antigo removido**: substituir o scratch stack
  `[f32;1024]` (limite CH≤16 — `layer.rs:45-56`) por **`AlignedVec<f32>` pré-alocado no load**,
  dimensionado por `ch*WAVENET_MAX_NUM_FRAMES`, permitindo **CH>16**. Conditioning ainda **COND=1**.
- **Critério de aceite**: testes unitários de paridade **numérica** dinâmico↔const-generic para as 4
  geometrias do catálogo (Standard/Lite/Feather/Nano) — devem bater (mesmo ESR). `heap-audit` confirma
  zero alocação no hot-path. Headers SPDX em todos os arquivos novos.
- **Risco**: 🔴 alto. **Pontos de atenção**: ordem de leitura de pesos idêntica ao C++
  (`model.cpp:623-644` — `head_scale` é o **último** f32); preservar o caminho **f32-native** do
  `head_rechannel` (fidelidade — `layer_array.rs:220`); ring-buffer/receptive-field por camada
  (`WaveNetLayerState`).

#### T2.2 — Vetorização born-SIMD do caminho dinâmico (guard-rail O5) 🔴

- **Onde**: kernels de `conv1d_dyn*`, `dense_dyn`, e o laço de ativação do `layer_dyn`.
- **O quê**: usar `SimdMath` (`M::tanh_and_overwrite_block`/`tanh_and_accumulate_block`, GEMV/dot
  vetorizados, `apply_gain`) em **todos** os passos por-amostra/por-bloco. Scratch alinhado
  (`AlignedVec`, 64B). **Nenhuma** redução/element-wise escalar no hot-path (reduções FP não
  autovetorizam — devem ser SIMD explícito).
- **Critério de aceite**: inspeção do hot-path dinâmico sem laço escalar (checklist O5);
  micro-bench do motor dinâmico em paridade-ish com o const-generic para os SKUs conhecidos
  (aceita-se overhead pequeno do dinâmico, mas **não** regressão escalar grosseira). `cargo bench`
  registrado.
- **Risco**: 🔴 alto (correção SIMD + RT). **Atenção**: `K` em runtime impede o array de taps
  compile-time — usar layout interleaved-4-wide do `Conv1dDyn` já existente; cuidar tail handling.

#### T2.3 — Generalizar a detecção/validação de topologia (sem regredir o catálogo) 🟠

- **Onde**: `src/loader/nam_json/topology.rs` (`get_wavenet_topology:84`,
  `validate_wavenet_features:444`).
- **O quê**: relaxar `get_wavenet_topology` para **não** rejeitar geometrias fora dos 4 padrões —
  retornar uma descrição de geometria (ch/k/head/dilations/n_arrays) quando não casar um SKU, em vez
  de `None`-rejeição. **Manter** as restrições de escopo desta rodada: `COND=1` (multi-cond é F2) e
  head pós-stack **rejeitado** (F6) — manter mensagens claras. O fast-path SKU continua detectado.
- **Critério de aceite**: a função passa a distinguir `{SKU conhecido | geometria livre válida |
  rejeição por feature não-suportada (F2/F6)}`. Testes cobrindo os 3 casos. Mensagens de erro de F2/F6
  permanecem (com referência à feature).
- **Risco**: 🟠 médio. **Atenção**: não afrouxar validações de segurança (tamanhos de pesos coerentes;
  evitar geometria que estoure orçamento de buffers).

---

### 🟠 Sprint S3 — Dispatch híbrido + paridade C++ + goldens endurecidos (P3)

> Depende de S2. Conecta o motor dinâmico ao loader e **prova** a paridade contra o C++, convertendo
> testes de "rejeição" em goldens positivos e endurecendo os gates frouxos (P3). É a sprint que
> **destrava F1 de fato** e entrega o valor ao usuário.

#### T3.1 — Dispatch híbrido no loader (fast-path SKU + fallback dinâmico) 🟠

- **Onde**: `src/loader/dispatcher/wavenet/mod.rs:108-142` (braço `None`); novo
  `dispatcher/wavenet/dynamic.rs`; `src/models/StaticModel` (variante para o modelo dinâmico).
- **O quê**: no braço hoje de rejeição, construir o **modelo dinâmico** (`build_wavenet_dynamic`) com
  alocação no load, **mantendo** o dispatch const-generic quando a geometria casar um SKU
  (`get_wavenet_topology` → fast-path). Atualizar a mensagem de erro para refletir que só F2/F6
  permanecem fora de escopo (não "fallback indisponível").
- **Critério de aceite**: probe de carga: `wavenet.nam` (ch=3, geometria livre — `TODO-features.md:69`)
  passa a **carregar**; os SKUs canônicos continuam no fast-path const-generic (verificar via log do
  dispatcher). `cargo test` verde.
- **Risco**: 🟠 médio. **Atenção**: enum `StaticModel` cresce — garantir que `process`/`prewarm`/
  `set_effective_layers` despachem corretamente para o caminho dinâmico.

#### T3.2 — Goldens oficiais: converter `test_loader_gap_*` → paridade C++ 🟠

- **Onde**: testes de "rejeição" (`test_loader_gap_wavenet*`); geração de golden via `render` v0.5.3
  (mesmo pipeline de `golden_gen_build.sh`); `tests/golden_vectors.rs`, `tests/common/validation.rs`.
- **O quê**: usar o `.nam` oficial `wavenet.nam` (geometria livre) como fonte; gerar referência C++ e
  validar Rust↔C++ por **ESR/SNR** (scale-invariant, não MSE absoluto). Migrar o teste de "afirmo que
  rejeita" → "afirmo que casa com o C++", **sem buraco de cobertura**.
- **Critério de aceite**: golden novo commitado; entrada em `get_calibrated_threshold`
  (`tests/common/validation.rs:339`) com comentário `// Measured: SNR=…, ESR=…` e margem 6–10 dB;
  o golden **pode falhar** (regra de ouro). Determinismo bitwise (`tests/self_consistency.rs`) e
  multi-SR (T4.2 em `tests/golden_vectors.rs`) cobrindo a nova geometria.
- **Risco**: 🟠 médio. **Atenção**: respeitar a calibração anti-placebo
  (`tests/threshold_calibration.rs:199`); regime de amplitude realista (sem reescalonamento artificial).

#### T3.3 — [P3] Endurecer gates frouxos onde a vigilância importa 🟠

- **Onde**: `tests/common/validation.rs:339` (tabela de thresholds);
  `tests/threshold_calibration.rs` (anti-placebo).
- **O quê**: triagem dos cenários de threshold baixo (12–18 dB): (a) se a divergência é **inerente**
  ao estímulo/SR (reamostragem), documentar e manter; (b) se há regressão evitável no engine
  (dinâmico inclusive), recalibrar com margem honesta; (c) avaliar gate **perceptual** (MR-STFT/LUFS)
  como complemento ao SNR cru nos casos difíceis. **Não** apertar onde causaria flakiness inerente.
- **Critério de aceite**: cada threshold revisado tem justificativa documentada (`// Measured:` +
  motivo); meta-teste anti-placebo continua verde; nenhum gate apertado a ponto de ficar flaky.
- **Risco**: 🟠 médio. **Atenção**: apertar gate sem entender a origem da divergência gera ruído de CI.

---

### 🟡 Sprint S4 — (Opcional / deferida) Política de fidelidade WaveNet — modo "exato" (P2)

> **Opcional e de médio prazo.** Só faz sentido após S1.T1.4 (medição). Entrega um **diferencial**
> (modo alta-fidelidade) sem afetar o caminho de produção padrão. Pode rodar em paralelo a outras
> features. Decisão de produto: implementar ou apenas documentar a política.

#### T4.1 — Modo "alta-fidelidade" opcional (sem FastMath / sem quantização) 🟡

- **Onde**: caminho de ativação (`src/math/activations/tanh/`), quantização de pesos
  (`src/math/common/ops.rs:16`), seleção de caminho no load.
- **O quê**: oferecer um modo **opcional** (feature flag ou config; **off por padrão**) que: (a)
  mantém pesos em **f32** (sem BF16/F16, fonte de drift dominante) e (b) usa tanh **exato** no
  hot-path. Trade-off: maior latência/uso de memória. Não altera o caminho de produção atual.
- **Critério de aceite**: com o modo ligado, ESR WaveNet cai para a faixa LSTM/Linear
  (objetivo ~1e-5 ou melhor) **medido**; com o modo desligado, números de produção **inalterados**
  (bit-exato vs baseline). Bench documentando o custo de latência do modo exato.
- **Risco**: 🟡 médio. **Atenção**: não vazar o caminho exato para o hot-path padrão; manter RT-safety
  no modo exato (alocação só no load).

#### T4.2 — Documentar a política de fidelidade do produto 🟢

- **Onde**: `docs/fastmath-approximations.md` e/ou `docs/` (arquitetura).
- **O quê**: registrar formalmente a barra de fidelidade (WaveNet ~0,3–1% por design vs LSTM/Linear
  exato), a recomendação de uso do modo exato, e a conexão com P1 (Lite é o caso extremo). Acionar a
  skill `documentador`.
- **Critério de aceite**: doc clara, sincronizada com a implementação; referência cruzada P1/P2/T1.4.
- **Risco**: 🟢 baixo.

---

## Nota de método e regras de revisão (transversais)

- **Guard-rail O5 (regra de revisão de S2/S3)**: qualquer PR que toque inferência WaveNet é **rejeitado**
  se introduzir laço de aritmética/redução f32/f16 por-amostra/por-bloco **escalar** no hot-path.
  Usar `SimdMath`/intrínsecos `core::arch` desde o início.
- **RT-safety**: alocação/planejamento só no `new()`/load; hot-path zero-alloc/zero-lock/zero-panic,
  auditável com a feature `heap-audit`.
- **Regra de ouro do golden**: todo golden deve **poder falhar**; calibrar por medição documentada
  (`// Measured: SNR=…, ESR=…`, margem 6–10 dB) com meta-teste anti-placebo.
- **Paridade é por ESR/SNR** (scale-invariant), nunca MSE absoluto; referência é o `render` do
  **NAMCore v0.5.3** pinado.
- **Escopo desta rodada**: F1 cobre **A1 geometria livre, COND=1, sem head pós-stack**. Multi-condição/
  FiLM (**F2/F3**), head pós-stack (**F6**), grouped conv (**F9**), ativações extras (**F8**),
  SlimmableWavenet (**F5**) e LSTM arbitrário (**F7**) permanecem fora — com mensagens de rejeição
  claras e referência à feature.
- **Notas de continuidade**: ao concluir S2/S3, se surgirem geometrias reais (corpus F12) que exijam
  N>2 layer-arrays ou CH muito alto, registrar aqui novas tarefas. P5 (pico de latência em silêncio)
  liga-se a T1.3 (DAZ/FTZ) — se o diagnóstico de silêncio revelar penalidade de denormal, abrir tarefa
  dedicada de gate de latência percentil.

---

## Impacto nos achados originais ao concluir este plano

> Mapa de **rastreabilidade**: o que cada sprint, ao ser concluída, marca nos documentos de origem.
> Marcar `[DONE]`/`[PARCIAL]` no achado correspondente **somente após** a sprint fechar com
> `cargo check`/`test`/`bench` verdes e zero warnings. Achados parciais devem registrar **o que ainda
> falta** e a feature/rodada que destrava o restante.

| Achado original                            | Sprints/Tarefas que o atendem        | Estado ao concluir o plano   | O que fica de fora (e onde resolve)                                                             |
| ------------------------------------------ | ------------------------------------ |:----------------------------:| ----------------------------------------------------------------------------------------------- |
| **F1** (`TODO-features.md:82`)             | S2 (T2.1–T2.3) + S3 (T3.1–T3.2)      | **[DONE]** (escopo A1)       | Multi-cond/FiLM → **F2/F3**; head pós-stack → **F6**; grouped conv → **F9**; Slimmable → **F5** |
| **P2** (`TODO-problemas.md:75`)            | S1 (T1.4) + S4 (T4.1–T4.2, opcional) | **[PARCIAL]**                | `[DONE]` só se S4 (modo exato) for implementada; senão fica medido+documentado (T1.4/T4.2)      |
| **P3** (`TODO-problemas.md:101`)           | S3 (T3.2, T3.3)                      | **[DONE]** (no escopo F1)    | Gates de cenários alheios a WaveNet/F1 seguem sua própria triagem                               |
| **P4** (`TODO-problemas.md:127`)           | S1 (T1.3)                            | **[DONE]** (diagnóstico+doc) | Resíduo **mantido** (fiel ao C++); só documentado. Liga-se a **P5** (latência denormal)         |
| **O5/S3** (`TODO-optimize.md:264`)         | S1 (T1.1 head_scale)                 | **[DONE]**                   | —                                                                                               |
| **O5 limpeza** (`TODO-optimize.md:291`)    | S1 (T1.2 BF16 morto)                 | **[DONE]**                   | —                                                                                               |
| **O5 guard-rail** (`TODO-optimize.md:307`) | S2 (T2.2) + regra de revisão S2/S3   | **[DONE]** (regra ativa)     | O5/S1 (head conv A2) e O5/S2 (rechannel A2) seguem em **`TODO-features.md §F3`**                |

**Ação de fechamento (obrigatória ao concluir cada sprint)** — espelhar o estado nos originais:

- **Ao fechar S1**: marcar `[DONE]` em **O5/S3** (`TODO-optimize.md` linha do achado S3), **O5 limpeza**
  e **P4**; marcar `[PARCIAL]` em **P2** com nota _"medição concluída (T1.4); modo exato pendente (S4)"_.
- **Ao fechar S2**: registrar nota de progresso em **F1** (_"motor dinâmico pronto; pendente dispatch
  híbrido S3"_) e marcar **O5 guard-rail** como regra ativa.
- **Ao fechar S3**: marcar `[DONE]` em **F1** (escopo A1) e **P3**; converter na tabela do §"Diagnóstico
  verificado" de `TODO-features.md` a linha de `wavenet.nam` de ❌ → ✅; remover a ressalva de "fallback
  indisponível" da mensagem do dispatcher.
- **Ao fechar S4** (se executada): marcar `[DONE]` em **P2**; caso S4 seja deferida, **P2** permanece
  `[PARCIAL]` com o modo exato como item futuro.

> **Regra de honestidade**: nenhum achado é marcado `[DONE]` por intenção — só após a verificação
> (testes/goldens/bench) exigida pela tarefa correspondente. Achados fora do escopo desta rodada
> (F2/F3/F5/F6/F8/F9/F7) **não** são tocados aqui.
