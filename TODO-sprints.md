<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# 🗺️ TODO-sprints — Porte da Arquitetura "A2" para o NAM-rs

> Plano de execução ágil para trazer a nova **Arquitetura A2** (consagrada no `NeuralAmpModelerCore`) ao `nam-rs`, com porte fiel do C++, testes automatizados e
> *golden vectors* como seguro anti-degradação. Alvo de performance: **x86-64-v3 (AVX2 + FMA)**. AVX-512 e ISAs avançadas ficam para um momento posterior.

**Fonte de verdade (C++):** `tests/fixtures/NeuralAmpModelerCore/` (espelho oficial do projeto). Toda tarefa de porte cita o(s) arquivo(s) e faixa(s) de linha de referência.

---

## 0. Contexto e Decisões de Escopo (fechadas)

A A2 foi **oficialmente lançada** (Core v0.5.2 / plugin v0.7.14). Na prática, o modelo A2 de produção é um **bundle de dois WaveNets de forma fixa** que compartilham exatamente o mesmo esqueleto, variando apenas a contagem de canais:
23 camadas, 1 *layer-array*, kernels `6/15`, dilations fixas, `LeakyReLU(0.01)`, *head conv*

| Modelo          | Canais | Esqueleto                                                                                    | Referência C++                   |
| --------------- | ------ | -------------------------------------------------------------------------------------------- | -------------------------------- |
| **A2-Full**     | 8      | 23 camadas, 1 *layer-array*, kernels `6/15`, dilations fixas, `LeakyReLU(0.01)`, *head conv* | `NAM/wavenet/a2_fast.cpp`        |
|                 |        | `k=16` com bias, `layer1x1` ativo, **sem** FiLM/gating/`condition_dsp`/`bottleneck≠channels` |  (`A2FastModel<8>`)              |
| **A2-Lite**     | 3      | Idêntico ao Full, só muda canais                                                             | `A2FastModel<3>`                 |
| **A1-Standard** | 16     | 2 *layer-arrays*, kernel 3 — **já implementado e validado**                                  | `src/models/wavenet/` (estático) |
| **A1-Nano**     | 4      | 2 *layer-arrays*, kernel 3 — **já implementado e validado**                                  | `src/models/wavenet/` (estático) |

**Decisões tomadas com o solicitante (ponto de decisão pré-plano):**

1. **Motor A2 = somente o *fast-path* fixo A2-Full/Lite.** Porte fiel de `a2_fast.cpp`. FiLM, gating `GATED`/`BLENDED`, `condition_dsp`, `bottleneck≠channels`, *grouped conv*, `head1x1` e ativações por-camada ficam **fora** (só são exercidos pelo modelo de teste `wavenet_a2_max.nam`).
2. **Slimmable = `SlimmableContainer` agora; `SlimmableWavenet` depois (sequenciamento, não descarte).** Ambos são arquiteturas **oficiais** do NAMCore (registradas, file version 0.7.0, com exemplos). Priorizamos o `SlimmableContainer` porque o A2 que a Tone3000 distribui **hoje** é, na prática, um *bundle de um nano + um standard* (confirmado por Mike Oliphant, autor do NeuralAudio/LV2: "the official A2 architecture looks to be a bundle of a nano and a standard model"), com o "quality scaling parameter" selecionando entre os dois. O `SlimmableWavenet` (fatiamento de canais de **rede única**, sem re-treino — tema do paper arXiv 2511.07470 e direção futura "uma captura que se escala sozinha") fica para o **Épico 6 (futuro)**, abaixo — planejado, não ignorado.
3. **Troca Full↔Lite integrada à FSM adaptativa de CPU** (`src/dsp/adaptive.rs`): auto-degradação Full→Lite sob pressão, com *crossfade*, mais override manual.
4. **IR Cabsim (.wav)** incluído como **épico separado pós-NAM** (convolução particionada FFT, ortogonal ao motor neural).
5. **Redução de burden** aprovada: (a) remover aliases VNNI mortos (`Avx2VnniMath`, variantes `Avx2Vnni`/`Avx512Vnni`); (b) remover `src/math/activations/experimental/piecewise_tanh.rs` (só `test+research`); (c) **aposentar** `WavenetA2Placeholder` ao final do Épico 1; (d) **remover os caminhos *dynamic*** (`WaveNetDynModel`/`LstmDynModel`) e seus *fallbacks* de loader — ver Sprint 1.5. Os 4 modelos-foco + container A2 não dependem deles.

### Fora de escopo (explícito)

- FiLM, `GatingMode::Gated`/`Blended`, `condition_dsp`, `head1x1`, *grouped convolution*, ativações por-camada heterogêneas, `bottleneck≠channels`.
- `SlimmableWavenet` (channel slicing de rede única) e `allowed_channels` — **adiado para o Épico 6 (futuro)**, não descartado. É formato oficial; só não é o que a distribuição A2 mainstream usa hoje.
- Golden do `wavenet_a2_max.nam` e `wavenet_condition_dsp.nam` (modelos de teste).
- AVX-512 / VNNI / BF16 dedicados à A2 (tratados em iteração futura de ISA).
- Caminhos *dynamic* (`WaveNetDynModel`/`LstmDynModel`): **removidos** — Modelos `.nam` de geometria fora do catálogo estático passam a falhar no load com erro claro (sem rodar). Isso também retira os *goldens* de cross-validation dos micro-modelos NAMCore (`golden_namcore_*`), que exercitavam geometrias não-padrão pela via dynamic.

---

## 1. Convenções Mandatórias (aplicáveis a todas as tarefas)

- **RT-Safety** (`.agents/rules/rust.md`): zero alloc/drop de heap na *audio thread*; sem `println!`/`format!`/locks; sem `unwrap()`/`expect()`; FTZ+DAZ; sinalização via `RtStatusFlags` (atômicos). Transferência de heap via SPSC GC.
- **SIMD x86-64-v3**: `AlignedVec<T>` (64 B), `chunks_exact`, *branchless*; reaproveitar o `SimdMath` trait + macro `dispatch_simd!` (preferir dispatch estático ao v-table).
- **Testes** (`.agents/rules/testing.md`): unidade inline se arquivo `< 300` linhas; senão `*_test.rs` irmão. Integração em `tests/`. Testes lentos/parity marcados `#[ignore]` (rodam em `utils/tests-long.sh`).
- **Copyright** (`.agents/rules/copyright.md`): cabeçalho SPDX em todo arquivo novo/editado.
- **Golden = seguro anti-degradação**: otimização *on-the-fly* só é aceita se todos os *golden vectors* permanecerem verdes dentro dos thresholds (ESR / MR-STFT / SNR adaptativo — ver `src/testing/perceptual.rs` e `tests/common/validation.rs`).
- **Pirâmide de validação (papéis complementares, não redundantes):** (1) **Referência escalar** (`src/math/common/scalar_ref/`) = oráculo de paridade **apertada** (`~1e-6`, via `proptest`), localização/bisseção de bugs de kernel, edge cases (`n % 8`, denormais, alinhamento) e invariante cross-ISA — roda na lane rápida `cargo test`, sem C++. **Não é fallback de produção** (sem AVX2 o `detect.rs` faz *fail-fast*); na prática também atua como tratamento de cauda/remainder dentro dos kernels SIMD. (2) **Golden vs NAMCore** = verdade externa de **banda larga** (tolerância por FastMath, ADR-002), end-to-end, lane lenta (`#[ignore]`) — pega erro de *algoritmo/spec*. Um bug que cabe na banda larga do golden mas quebra a paridade apertada do escalar só é pego pela camada (1); um erro de spec compartilhado por escalar+SIMD só é pego pela camada (2).
- **Lint final** (`.agents/rules/linting.md`): `utils/lints.sh` + `cargo check` por feature; acionar `documentador` em mudanças arquiteturais relevantes.

### Mapa de organização de arquivos (onde colocar)

| Domínio                  | Localização proposta                                                    | Justificativa                                      |
| ------------------------ | ----------------------------------------------------------------------- | -------------------------------------------------- |
| Estruturas/inferência A2 | `src/models/a2/` (já existe; expandir)                                  | Mantém A2 coeso junto ao scaffolding atual         |
| Kernels SIMD A2          | `src/math/wavenet/` (A2-specific) e `src/math/activations/` (LeakyReLU) | Toda matemática vetorizada vive em `src/math/`     |
| Container slimmable      | `src/models/container.rs` + trait em `src/models/slimmable.rs`          | Wrapper de modelos é responsabilidade de `models/` |
| Parser do container      | `src/loader/dispatcher/container/`                                      | Dispatch por arquitetura já vive no loader         |
| IR Cabsim                | `src/dsp/cabsim/` (estágio) + kernel FFT em `src/dsp/cabsim/conv.rs`    | É estágio de DSP pós-inferência                    |
| Fixtures/golden A2       | `tests/fixtures/` + `tests/fixtures/models/`                            | Segue convenção atual de golden                    |

---

## ÉPICO 0 — Fundação e Higiene 🧹

> Objetivo: reduzir burden e preparar o terreno antes de adicionar A2. Entrega rápida, baixo risco, mantém a suíte 100% verde.

### Sprint 0.1 — Limpeza e alinhamento

- **[T0.1] Remover aliases VNNI mortos.** [DONE]
  - Remover `Avx2VnniMath` (alias em `src/math/common/avx2_impl.rs:670`) e as variantes `Avx2Vnni`/`Avx512Vnni` de `src/math/common/dispatch/instruction_set.rs` e do v-table (`src/math/common/dispatch/config.rs`, `detect.rs`).
  - **Critério de aceite:** `cargo build` em todas as features; matriz de detecção de ISA reduzida a `Avx2`, `Avx512`, `Avx512VnniBf16` (este último mantido para BF16 nativo). Nenhuma regressão de golden/bench.
  - **Riscos:** garantir que nenhum *call-site* referencie os símbolos removidos.

- **[T0.2] Remover `experimental/piecewise_tanh`.** [DONE]
  - Remover `src/math/activations/experimental/` (gated por `test+research`) e a feature `research` do `Cargo.toml:63` se não houver outro consumidor.
  - **Critério de aceite:** `cargo check --all-features` limpo; sem referências órfãs; `utils/lints.sh` verde.

- **[T0.3] Consolidar o scaffolding A2 existente.**
  - Auditar `src/models/a2/{params,activations,film,gating}.rs`. `params.rs` já espelha `a2_fast.h` (constantes `A2_NUM_LAYERS=23`, `A2_KERNEL_SIZES`, `A2_DILATIONS`, `A2_LEAKY_SLOPE`). Marcar como *fora de escopo agora* (sem remover) os structs de FiLM/gating/`head1x1`/`bottleneck` que não serão usados pelo fast-path, documentando com `//! NOTE: reservado p/ motor A2 geral (futuro)`.
  - **Fonte de verdade:** `NAM/wavenet/a2_fast.h:30-43`.
  - **Critério de aceite:** documentação inline coerente; nada removido que a suíte A2 (T1.x) vá precisar.

---

## ÉPICO 1 — Núcleo de Inferência A2 (A2-Full/Lite) 🧠

> Objetivo: porte direto e **correto** do `a2_fast.cpp` (baseline, sem micro-opt agressiva), ancorado por *golden vectors*. Este épico entrega A2-Full e A2-Lite funcionais e validados.

### Sprint 1.1 — Primitivas compartilhadas

- **[T1.1] Kernel `LeakyReLU(0.01)` SIMD (AVX2/FMA).**
  - Implementar `leaky_relu_slice` em `src/math/activations/` (in-place, `chunks_exact(8)`, *branchless* via máscara/blend). Adicionar a **referência escalar (oráculo de teste + tratamento de cauda/remainder)** em `src/math/common/scalar_ref/` — **não** é fallback de produção para CPU sem AVX2 (o `detect.rs` faz *fail-fast*); serve como oráculo de paridade apertada (`~1e-6`, via `proptest`), bisseção de bugs, cobertura de edge cases (`n % 8`, denormais) e invariante cross-ISA para o futuro AVX-512.
  - **Fonte de verdade:** `NAM/activations.h` (`LeakyReLU`) e nota de uso em `NAM/wavenet/a2_fast.cpp:49-51` (`LeakyReLU(0.01) em todas as camadas`).
  - **Critério de aceite:** teste de paridade `< 1e-6` vs referência escalar; `#[cfg(test)]` inline ou `_test.rs`. Heap-audit zero.

- **[T1.2] Conv1D dilatada para A2 (kernels 6 e 15) sobre `mirror_buf`.**
  - Avaliar reuso de `src/models/wavenet/conv1d_dyn.rs` + `src/dsp/mirror_buf.rs`. A2 usa apenas `kernel_size ∈ {6,15}` e dilations fixas (`A2_DILATIONS`), com 1 canal de entrada na 1ª camada e `CH` canais nas demais. Garantir histórico via ring/mirror sem alloc.
  - **Fonte de verdade:** `a2_fast.cpp:417-690` (`_layer_forward_k`), `NAM/wavenet/detail.h` (`Layer`/`LayerArray`), `docs/wavenet_walkthrough.rst:47-214`.
  - **Critério de aceite:** convolução isolada bate com referência escalar em micro-teste; RT-safe.

- **[T1.3] *Head conv* A2 (`k=16`, bias, `head_scale`).**
  - Implementar a convolução de cabeça: `Conv1D(bottleneck→1, K=16, bias)` lida de ring com *tail-mirror*, seguida de multiplicação por `head_scale`.
  - **Fonte de verdade:** `a2_fast.cpp:119-124` (`_head_w`/`_head_b`/`_head_scale`), `a2_fast.cpp:722-743` (`_head_forward`).
  - **Critério de aceite:** saída do head bate com referência em micro-teste.

### Sprint 1.2 — Modelo A2 (baseline correto)

- **[T1.4] Struct do modelo `WaveNetA2<const CH: usize>` (CH=3 e CH=8).**
  - Criar `src/models/a2/model.rs`: 1 *layer-array* de 23 camadas + *rechannel* de entrada (`Conv1x1 input_size→CH`) + acumulador de *head* + *head conv* + `head_scale`. Processamento em blocos (consistente com `WAVENET_MAX_NUM_FRAMES`).
  - Implementar `NamModel` + `sealed::Sealed`; expor `process`, `prewarm`, `reset`, `set_max_buffer_size`, `receptive_field`, `channels`.
  - **Fonte de verdade:** `a2_fast.cpp` (classe `A2FastModel`), `detail.h` (`LayerArray::Process`), `docs/wavenet_walkthrough.rst:278-351`.
  - **Critério de aceite:** compila; `receptive_field` confere com soma de `(kernel-1)*dilation` + head; sem alloc no hot-path.

- **[T1.5] Camada A2 (`A2Layer`).**
  - Sequência: dilated conv (T1.2) → soma com `input_mixin` (`Conv1x1 condition→CH`, sem bias) → `LeakyReLU` (T1.1) → acumula no *head* → `layer1x1` (`Conv1x1 CH→CH`, bias) → conexão residual `out = input + layer1x1_out`.
  - **Fonte de verdade:** `a2_fast.cpp:514` (sequência conv→mixin→LeakyReLU→head →layer1x1 residual), `detail.h` (`Layer`), `docs/wavenet_walkthrough.rst:103-214`.
  - **Critério de aceite:** paridade camada-a-camada vs referência em micro-teste sintético.

- **[T1.6] Carga de pesos A2 (ordem do stream).**
  - Implementar `set_weights` respeitando a ordem exata: `_rechannel` → (por camada: `_conv` → `_input_mixin` → `_layer1x1`) → `_head_rechannel` (conv k=16, bias) → **`head_scale` (último float do stream)**.
  - **Fonte de verdade:** `a2_fast.cpp:196-282` (documenta a ordem), `a2_fast.cpp:264-275` (head + `head_scale` trailing), `generate_weights_a2.py:18-90` (contagem de pesos por bloco).
  - **Critério de aceite:** contagem de pesos consumidos == `weights.len()` (asserção); erro claro (sem panic em runtime RT) se divergir.

### Sprint 1.3 — Loader, dispatch e aposentadoria do placeholder

- **[T1.7] Dispatch A2 no loader.**
  - Em `src/loader/dispatcher/wavenet/mod.rs`, tornar a A2 um **branch de primeira classe** (não mais *fallback*-após-falha): detectar a forma via `is_a2_shape()` (`src/loader/nam_json/topology.rs:131`) **antes** do match de topologias A1 e construir `WaveNetA2<3>`/`WaveNetA2<8>`. Registrar novas variantes no enum `DynamicModel` (`src/models/mod.rs`) e no dispatch (`src/models/dynamic_model.rs`).
  - **Importante (pré-requisito da Sprint 1.5):** como o *fallback* dynamic será removido, o dispatch precisa decidir entre {A1 estático, A2 estático} de forma explícita; geometrias não reconhecidas retornam **erro de load claro**.
  - **Fonte de verdade:** `a2_fast.cpp:849-990` (`is_a2_shape`/`create_a2_fast_config`), `topology.rs:131-163`.
  - **Critério de aceite:** carregar um `.nam` A2-Full/Lite produz inferência real (não silêncio); `mock_a2.nam` ainda reconhecido.

- **[T1.8] Metadados e par de modelos.**
  - Atualizar `src/loader/loaded_model_pair.rs` (topologia/`weights_layout`) e `src/loader/build.rs` (calibração de ganhos via `input_level_dbu`/`loudness`, prewarm ≥ 2048 amostras) para A2.
  - **Critério de aceite:** `--model A2.nam` no standalone roda com telemetria; ganhos calibrados.

- **[T1.9] Aposentar `WavenetA2Placeholder`.**
  - Remover `src/models/a2/placeholder.rs`, a variante `WavenetA2` placeholder e o flag `RT_STATUS_A2_PLACEHOLDER`. Atualizar `tests/loader_a2_compat.rs` e `tests/a2_placeholder_interface.rs` para validar **inferência real**.
  - **Critério de aceite:** suíte verde sem o placeholder; nenhum caminho emite silêncio para A2 válido.

### Sprint 1.4 — Golden Tests A2 (seguro anti-degradação) 🧪

- **[T1.10] Gerador de fixtures A2-Full/Lite.**
  - Adaptar/derivar de `generate_weights_a2.py` um gerador determinístico (seed fixa) que emita `wavenet_a2_full.nam` (CH=8) e `wavenet_a2_lite.nam` (CH=3) com o esqueleto fixo (23 camadas, kernels/dilations canônicos, `LeakyReLU`, `head_scale`). Salvar em `tests/fixtures/models/`.
  - **Fonte de verdade:** `generate_weights_a2.py`, `a2_fast.h:30-43`.
  - **Critério de aceite:** arquivos carregam tanto no C++ `render` quanto no loader Rust (T1.7).

- **[T1.11] Estender `golden_gen_build.sh` para A2.**
  - Gerar `tests/fixtures/golden_wavenet_a2_full.bin` e `..._a2_lite.bin` (v1 e variantes v2 multi-SR) renderizando com o `render` do C++ (caminho genérico `WaveNet` = verdade; o `a2_fast` produz saída idêntica).
  - **Fonte de verdade:** `tests/fixtures/golden_gen_build.sh`, `src/bin/wav_to_golden.rs`, `src/bin/gen_stress.rs`.
  - **Critério de aceite:** novos `.golden.bin` no formato `[u32 N][f32×N in][f32×N out]`; documentado em `tests/fixtures/README.md`.

- **[T1.12] Testes de inferência golden + cross-validation viva.**
  - Adicionar casos em `tests/nam_infer_test.rs` (rápidos, pré-commit) e em `tests/cpp_parity.rs` (`#[ignore]`, vivos) para A2-Full/Lite. Definir thresholds adaptativos de SNR/ESR/MR-STFT para A2 em `tests/common/validation.rs` e baselines em `src/testing/perceptual.rs`.
  - **Critério de aceite:** golden verdes; ESR dentro do baseline; cross-val viva passa em `utils/tests-long.sh`.

- **[T1.13] RT-Safety e edge tests A2.**
  - Estender `tests/wavenet_prewarm_edge.rs`, heap-audit (`tests/resampler_heap_audit.rs` análogo p/ A2) e soak (`tests/pipeline_soak.rs`/`soak_test.rs`) cobrindo A2.
  - **Critério de aceite:** zero alloc no hot-path (CountingAllocator); estável em milhões de frames.

### Sprint 1.5 — Remoção dos caminhos *dynamic* (corte de burden) ✂️

> Executar **após** A2 e os 4 modelos-foco estarem validados (Sprints 1.1-1.4), garantindo que nenhum caminho de produção dependa do *fallback* dynamic.

- **[T1.14] Remover WaveNet dynamic.**
  - Remover `src/models/wavenet/{model_dyn,layer_dyn,conv1d_dyn,conv1d_dyn_dual,dense_dyn}.rs` (e correlatos), a variante `DynamicModel::WavenetDyn` e `src/loader/dispatcher/wavenet/dynamic.rs` (`build_wavenet_dynamic`).
  - Ajustar `dispatcher/wavenet/mod.rs:68` para retornar **erro de load** em geometria não-catalogada (sem panic, mensagem diagnóstica via `NamDiagnostic`).
  - **Critério de aceite:** `cargo build` limpo; modelos A1-Standard/Lite/Feather/Nano e A2-Full/Lite seguem carregando; `.nam` fora do catálogo falha com erro claro.

- **[T1.15] Remover LSTM dynamic.**
  - Remover `src/models/lstm/{model_dyn,layer_dyn}.rs` (e correlatos), a variante `DynamicModel::LstmDyn` e `src/loader/dispatcher/lstm/dynamic_builder.rs` (`build_lstm_dynamic`). Ajustar `lstm/dispatch.rs:52` para erro de load em `(num_layers, hidden)` não-catalogado.
  - **Critério de aceite:** aliases LSTM estáticos (1×8..2×24) seguem funcionando; geometria não-catalogada falha com erro claro.

- **[T1.16] Limpar fixtures/testes de cross-val dependentes do dynamic.**
  - Remover os modelos/goldens NAMCore micro (`tests/fixtures/models/{lstm,wavenet}.nam`, `golden_namcore_lstm_1x3.bin`, `golden_namcore_wn_micro.bin`) e os testes que os exercitam (`tests/dynamic_parity.rs` e casos correspondentes em `cpp_parity.rs`/`nam_infer_test.rs`). Atualizar `tests/fixtures/README.md` e o script `golden_gen_build.sh`.
  - **Critério de aceite:** suíte verde sem referências órfãs; `utils/tests-cargo.sh` e `utils/tests-long.sh` passam.

- **[T1.17] Simplificar enum/dispatch e documentar.**
  - Reduzir `DynamicModel` (`src/models/mod.rs`) e `dynamic_model.rs` às variantes estáticas remanescentes (A1 estáticos + A2-Full/Lite + LSTM estáticos).
  - Considerar renomear o enum (ex.: `StaticModel`) se "Dynamic" deixar de fazer sentido.
  - Atualizar `docs/architecture.md` e o `README.md` (seção de modelos suportados — remover a menção a "Dynamic Mode (Absolute Flexibility)").
  - **Critério de aceite:** API coerente; documentação alinhada; sem *dead code*.

---

## ÉPICO 2 — Otimização SIMD A2 (x86-64-v3) ⚡

> Objetivo: aplicar otimizações *on-the-fly* fiéis ao `a2_fast.cpp`, **protegidas pelos golden vectors** do Épico 1 (nenhuma quebra de correção).

### Sprint 2.1 — Kernels otimizados

- **[T2.1] Caminho CH=3 (A2-Lite): GEMV totalmente desenrolado.**
  - Portar a estratégia escalar/SIMD desenrolada para 3 canais.
  - **Fonte de verdade:** `a2_fast.cpp` (estratégia `Channels=3`, GEMV unrolled).
  - **Critério de aceite:** golden A2-Lite verde; ganho mensurável vs baseline T1.

- **[T2.2] Caminho CH=8 (A2-Full): *tap-major* frame-tiled (T=4) com broadcast-FMA.**
  - Portar a estratégia de *tiling* de 4 frames com *broadcast*-FMA e layout *col-major-per-tap*.
  - **Fonte de verdade:** `a2_fast.cpp` (estratégia `Channels=8`, T=4 tap-major).
  - **Critério de aceite:** golden A2-Full verde; ganho mensurável.

- **[T2.3] Ring `pow2` + *tail-mirror* para dilations e head.**
  - Consolidar buffers de histórico com máscara `pow2` e espelhamento de cauda (leitura *branchless*), reusando/estendendo `src/dsp/mirror_buf.rs`.
  - **Fonte de verdade:** `a2_fast.cpp:335-344,771-798` (ring pow2 + memmove rewind).
  - **Critério de aceite:** sem ramos no caminho de leitura; golden verde.

- **[T2.4] Permutação de pesos para layout SIMD.**
  - No `set_weights` (T1.6), permutar Conv1D de *row-major-per-tap* para *col-major-per-tap* (acesso amigável a SIMD), feito **uma vez** na carga.
  - **Fonte de verdade:** `a2_fast.cpp:196-282` (loader permutando layout).
  - **Critério de aceite:** golden verde; carga sem custo no hot-path.

### Sprint 2.2 — Validação de performance

- **[T2.5] Benchmarks Criterion A2-Full/Lite.**
  - Adicionar casos em `benches/inference_bench.rs` (e `dot_4x_bench.rs` se aplicável).
  - Medir µs/bloco a 48 kHz, buffers 64/128/256.
  - **Critério de aceite:** relatório de ganho documentado em `docs/benchmarks.md`; **zero regressão** em modelos A1; golden 100% verde.

---

## ÉPICO 3 — SlimmableContainer + Integração FSM Adaptativa 🔀

> Objetivo: carregar o **bundle oficial A2** (nano+standard) e trocar Full↔Lite em runtime, integrado à FSM de pressão de CPU já existente.

### Sprint 3.1 — Container

- **[T3.1] Trait `SlimmableModel` + parser `SlimmableContainer`.**
  - Criar `src/models/slimmable.rs` (`trait SlimmableModel { fn set_slimmable_size(&mut self, val: f32); }`) e parser `src/loader/dispatcher/container/` para a arquitetura `"SlimmableContainer"` (`config.submodels[] = {max_value, model}`), construindo cada submodelo via o dispatcher recursivo.
  - **Fonte de verdade:** `NAM/slimmable.h`, `NAM/container.h:18-64`, `NAM/container.cpp:149` (registro/parser).
  - **Critério de aceite:** `slimmable_container.nam` (exemplo) carrega; ordena submodelos por `max_value` ascendente; último cobre `>= 1.0`.

- **[T3.2] `ContainerModel` (despacho por threshold).**
  - Criar `src/models/container.rs`: guarda N submodelos pré-construídos; `set_slimmable_size(val)` seleciona índice por threshold e chama `reset()` no submodelo ativo; `process()` despacha ao ativo. **Todos** os submodelos pré-alocados/prewarmed na carga (zero alloc no switch).
  - **Fonte de verdade:** `NAM/container.cpp` (dispatch + seleção).
  - **Critério de aceite:** RT-safe na troca; golden por submodelo (A2-Full e A2-Lite) reaproveitando fixtures do Épico 1.

### Sprint 3.2 — Integração com a FSM adaptativa

- **[T3.3] Ligar `set_slimmable_size` à FSM de pressão de CPU.**
  - Mapear os estados de `src/dsp/adaptive.rs` (Full→Reduced→Minimal) para a seleção de submodelo (A2-Full ↔ A2-Lite), usando os limiares de P99/budget já monitorados pela telemetria (`src/dsp/telemetry.rs`).
  - **Critério de aceite:** sob carga simulada, o engine migra Full→Lite e retorna por histerese, sem realocar.

- **[T3.4] *Crossfade* sem cliques na troca.**
  - Reusar `src/dsp/smoother.rs`/lógica de *crossfade* da `adaptive.rs` para transição suave entre submodelos.
  - **Critério de aceite:** ausência de descontinuidade audível (teste de energia/continuidade no ponto de troca).

- **[T3.5] Override manual (CLI + CLAP).**
  - Flag de CLI (`src/standalone/cli.rs`) e parâmetro CLAP para fixar/forçar nível (Auto/Full/Lite). Manual sobrepõe a FSM.
  - **Critério de aceite:** `--slim auto|full|lite` funciona; param CLAP exposto; documentado.

- **[T3.6] Telemetria e testes de transição.**
  - Sinalizar nível ativo via `RtStatusFlags` (atômico) → log na main thread.
  - Testes de FSM (estilo `tests/gate_fsm_proptest.rs`) e soak de transições.
  - **Critério de aceite:** transições determinísticas e estáveis sob soak.

---

## ÉPICO 4 — IR Cabsim (.wav convolution) 🔊

> Objetivo: feature útil e **ortogonal à A2** — carregar um `.wav` de impulse response e convoluir (estágio pós-NAM). Convolução particionada FFT, RT-safe, reusando `rustfft` (já no `Cargo.toml`).

### Sprint 4.1 — Loader de IR

- **[T4.1] Loader de `.wav` IR.**
  - Loader robusto em `src/dsp/cabsim/loader.rs` para WAV mono (PCM16/24/float32), com resample para a SR ativa (reusar `src/dsp/resampler.rs`) e normalização opcional. Carga/preparo **fora** da audio thread; transferência via SPSC (estilo *resampler swap* em `src/common/spsc/`).
  - **Critério de aceite:** carrega os WAVs de exemplo em `tests/` (ex.: `amostra-guitarra-*_FAT_CAB.wav`); erros tratados sem panic.

### Sprint 4.2 — Convolução particionada

- **[T4.2] Uniform-Partitioned Overlap-Save (FFT).**
  - Implementar convolução particionada em `src/dsp/cabsim/conv.rs` (partições de tamanho = buffer, overlap-save), pré-FFT do kernel na carga; FDL (*frequency delay line*) pré-alocada; **zero alloc** no hot-path.
  - **Fonte de verdade conceitual:** literatura de UPOLS/partitioned convolution (não há referência no C++; é feature nova do nam-rs).
  - **Critério de aceite:** paridade vs convolução direta (referência ingênua) `ESR < 1e-5` em IR curto; latência == tamanho da partição documentada.

- **[T4.3] Estágio opcional no pipeline.**
  - Integrar como estágio pós-inferência em `src/dsp/pipeline/` com *bypass* de custo zero quando nenhum IR está carregado. Flag de CLI/param CLAP.
  - **Critério de aceite:** ativável/desativável em runtime sem clique; bypass não mede custo.

- **[T4.4] Testes + bench IR.**
  - Golden de convolução (IR sintético determinístico), heap-audit e bench Criterion. `#[ignore]` para os pesados.
  - **Critério de aceite:** golden verde; zero alloc; bench documentado.

---

## ÉPICO 5 — Documentação e Fechamento 📚

- **[T5.1] Atualizar documentação arquitetural** (acionar skill `documentador`):
  - `docs/architecture.md`:  motor A2, container, cabsim.
  - `README.md`: seção "Supported Models" — A2-Full/Lite agora suportados, IR cabsim disponível.
  - `docs/cpp_parity_map.md`: novo mapeamento A2 → C++.
  - `tests/fixtures/README.md`: com os novos golden (A2 e IR) e instruções de regeneração.
  - Skill `refatora-doc.md`

- **[T5.2] Rodadas de melhorias**
  - `revisor-auditor`
  - `pesquisador-inovador.md`
  - `refatora-rust.md`
  - `refatora-doc.md`

---

## ÉPICO 6 (FUTURO) — `SlimmableWavenet` (channel slicing de rede única) 🔮

> **Adiado por sequenciamento, não descartado.**
> É arquitetura **oficial** do NAMCore (registrada, file version 0.7.0) e direção declarada da NAM ("uma captura que se escala sozinha, sem versão *lite* separada").
> Será priorizado quando modelos `.nam` com o campo `"slimmable"` (rede única) tornarem-se comuns na distribuição mainstream — hoje o A2 mainstream usa o `SlimmableContainer`.

- **[T6.1] Parser `slimmable` por *layer-array*.**
  - Ler o campo `"slimmable": {"method": "slice_channels_uniform", "kwargs": {"allowed_channels": [...]}}`.
  - **Fonte de verdade:** `NAM/wavenet/slimmable.cpp` (`SlimmableWavenetConfig`),
    `example_models/slimmable_wavenet.nam`.

- **[T6.2] Extração de subconjunto de pesos por contagem de canais.**
  - Portar `extract_conv1d`/`extract_conv1x1`/`compute_slim_bottleneck` (mapeamento ratio→canais, fatiamento das primeiras `slim_out×slim_in` linhas/colunas).
  - **Fonte de verdade:** `NAM/wavenet/slimmable.cpp:21-90` (helpers de extração).

- **[T6.3] *Staging* RT-safe de troca de modelo.**
  - Reconstrução do WaveNet no *off-thread* e publicação via slot atômico (`Acquire`/`Release`), instalado antes do DSP — análogo ao `std::atomic<shared_ptr>` do C++, usando o padrão SPSC/GC do nam-rs (sem alloc/drop na audio thread).
  - **Fonte de verdade:** `NAM/wavenet/slimmable.h:64-92` (staging atômico).

- **[T6.4] Golden + parity.**
  - Gerar fixtures slimmable (`allowed_channels`) e validar cada ponto de operação contra o C++ (`cpp_parity`, `#[ignore]`).
  - **Critério de aceite:** golden verde em todos os níveis de canal; troca RT-safe (heap-audit zero) sob soak.

---

## 📌 Notas de Rastreabilidade C++ → Rust

| Componente A2           | C++ (fonte de verdade)                        | Rust (destino)                          |
| ----------------------- | --------------------------------------------- | --------------------------------------- |
| Forma fixa / constantes | `NAM/wavenet/a2_fast.h:30-43`                 | `src/models/a2/params.rs` (já alinhado) |
| Modelo fast-path        | `NAM/wavenet/a2_fast.cpp`                     | `src/models/a2/model.rs`                |
| Camada + sequência      | `a2_fast.cpp:417-690`, `NAM/wavenet/detail.h` | `src/models/a2/layer.rs`                |
| Head conv + scale       | `a2_fast.cpp:722-743`                         | `src/models/a2/head.rs`                 |
| Ordem de pesos          | `a2_fast.cpp:196-282`                         | `set_weights` (T1.6/T2.4)               |
| Detecção de forma       | `a2_fast.cpp:849-990`                         | `src/loader/nam_json/topology.rs`       |
| LeakyReLU               | `NAM/activations.h`                           | `src/math/activations/`                 |
| Container               | `NAM/container.{h,cpp}`, `NAM/slimmable.h`    | `src/models/{container,slimmable}.rs`   |
| Walkthrough (didático)  | `docs/wavenet_walkthrough.rst`                | —                                       |
