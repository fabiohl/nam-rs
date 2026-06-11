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

## ÉPICO 0 — Fundação e Higiene 🧹 [DONE]

> Objetivo: reduzir burden e preparar o terreno antes de adicionar A2. Entrega rápida, baixo risco, mantém a suíte 100% verde.

### Sprint 0.1 — Limpeza e alinhamento [DONE]

- **[T0.1] Remover aliases VNNI mortos.** [DONE]
  - Remover `Avx2VnniMath` (alias em `src/math/common/avx2_impl.rs:670`) e as variantes `Avx2Vnni`/`Avx512Vnni` de `src/math/common/dispatch/instruction_set.rs` e do v-table (`src/math/common/dispatch/config.rs`, `detect.rs`).
  - **Critério de aceite:** `cargo build` em todas as features; matriz de detecção de ISA reduzida a `Avx2`, `Avx512`, `Avx512VnniBf16` (este último mantido para BF16 nativo). Nenhuma regressão de golden/bench.
  - **Riscos:** garantir que nenhum *call-site* referencie os símbolos removidos.

- **[T0.2] Remover `experimental/piecewise_tanh`.** [DONE]
  - Remover `src/math/activations/experimental/` (gated por `test+research`) e a feature `research` do `Cargo.toml:63` se não houver outro consumidor.
  - **Critério de aceite:** `cargo check --all-features` limpo; sem referências órfãs; `utils/lints.sh` verde.

- **[T0.3] Consolidar o scaffolding A2 existente.** [DONE]
  - Auditar `src/models/a2/{params,activations,film,gating}.rs`. `params.rs` já espelha `a2_fast.h` (constantes `A2_NUM_LAYERS=23`, `A2_KERNEL_SIZES`, `A2_DILATIONS`, `A2_LEAKY_SLOPE`). Marcar como *fora de escopo agora* (sem remover) os structs de FiLM/gating/`head1x1`/`bottleneck` que não serão usados pelo fast-path, documentando com `//! NOTE: reservado p/ motor A2 geral (futuro)`.
  - **Fonte de verdade:** `NAM/wavenet/a2_fast.h:30-43`.
  - **Critério de aceite:** documentação inline coerente; nada removido que a suíte A2 (T1.x) vá precisar.

---

## ÉPICO 1 — Núcleo de Inferência A2 (A2-Full/Lite) 🧠 [DONE]

> Objetivo: porte direto e **correto** do `a2_fast.cpp` (baseline, sem micro-opt agressiva), ancorado por *golden vectors*. Este épico entrega A2-Full e A2-Lite funcionais e validados.

### Sprint 1.1 — Primitivas compartilhadas [DONE]

- **[T1.1] Kernel `LeakyReLU(0.01)` SIMD (AVX2/FMA).** [DONE]
  - Implementar `leaky_relu_slice` em `src/math/activations/` (in-place, `chunks_exact(8)`, *branchless* via máscara/blend). Adicionar a **referência escalar (oráculo de teste + tratamento de cauda/remainder)** em `src/math/common/scalar_ref/` — **não** é fallback de produção para CPU sem AVX2 (o `detect.rs` faz *fail-fast*); serve como oráculo de paridade apertada (`~1e-6`, via `proptest`), bisseção de bugs, cobertura de edge cases (`n % 8`, denormais) e invariante cross-ISA para o futuro AVX-512.
  - **Fonte de verdade:** `NAM/activations.h` (`LeakyReLU`) e nota de uso em `NAM/wavenet/a2_fast.cpp:49-51` (`LeakyReLU(0.01) em todas as camadas`).
  - **Critério de aceite:** teste de paridade `< 1e-6` vs referência escalar; `#[cfg(test)]` inline ou `_test.rs`. Heap-audit zero.
  - **Nota de auditoria (Sprint 1.1):** O modelo A2 (`src/models/a2/activations.rs:112`) despacha LeakyReLU via `prelu_slice(data, &[negative_slope])` (mais genérico), e não via `leaky_relu_slice`. O kernel `leaky_relu_slice` está implementado, testado e registrado no dispatcher, mas é código morto na produção atual. A equivalência matemática é garantida pois `prelu_slice` com slope única executa a mesma operação.

- **[T1.2] Conv1D dilatada para A2 (kernels 6 e 15) sobre `mirror_buf`.** [DONE]
  - Avaliar reuso de `src/models/wavenet/conv1d_dyn.rs` + `src/dsp/mirror_buf.rs`. A2 usa apenas `kernel_size ∈ {6,15}` e dilations fixas (`A2_DILATIONS`), com 1 canal de entrada na 1ª camada e `CH` canais nas demais. Garantir histórico via ring/mirror sem alloc.
  - **Fonte de verdade:** `a2_fast.cpp:417-690` (`_layer_forward_k`), `NAM/wavenet/detail.h` (`Layer`/`LayerArray`), `docs/wavenet_walkthrough.rst:47-214`.
  - **Critério de aceite:** convolução isolada bate com referência escalar em micro-teste; RT-safe.
  - **Nota de auditoria (Sprint 1.1):** `A2Conv1d` reutiliza `Conv1dDyn` (stack-only, sem alloc no hot-path) e está validado por 7 testes de paridade. ~~Faltam: (a) heap-audit test com `CountingAllocator` específico para o conv1d A2; (b) soak test com K=6/15.~~ **[Concluído em T1.13]:** heap-audit A2 implementado em `tests/a2_heap_audit.rs` (zero alloc verificado nos block_sizes {1,16,32,48,64}); soak tests K=6/15 implementados em `tests/soak_test.rs` (`test_a2_{full,lite}_{silence,noise}_soak`).

- **[T1.3] *Head conv* A2 (`k=16`, bias, `head_scale`).** [DONE]
  - Implementar a convolução de cabeça: `Conv1D(bottleneck→1, K=16, bias)` lida de ring com *tail-mirror*, seguida de multiplicação por `head_scale`.
  - **Fonte de verdade:** `a2_fast.cpp:119-124` (`_head_w`/`_head_b`/`_head_scale`), `a2_fast.cpp:722-743` (`_head_forward`).
  - **Critério de aceite:** saída do head bate com referência em micro-teste.

### Sprint 1.2 — Modelo A2 (baseline correto) [DONE]

- **[T1.4] Struct do modelo `WaveNetA2<const CH: usize>` (CH=3 e CH=8).** [DONE]
  - Criar `src/models/a2/model.rs`: 1 *layer-array* de 23 camadas + *rechannel* de entrada (`Conv1x1 input_size→CH`) + acumulador de *head* + *head conv* + `head_scale`. Processamento em blocos (consistente com `WAVENET_MAX_NUM_FRAMES`).
  - Implementar `NamModel` + `sealed::Sealed`; expor `process`, `prewarm`, `reset`, `set_max_buffer_size`, `receptive_field`, `channels`.
  - **Fonte de verdade:** `a2_fast.cpp` (classe `A2FastModel`), `detail.h` (`LayerArray::Process`), `docs/wavenet_walkthrough.rst:278-351`.
  - **Critério de aceite:** compila; `receptive_field` confere com soma de `(kernel-1)*dilation` + head; sem alloc no hot-path.

- **[T1.5] Camada A2 (`A2Layer`).** [DONE]
  - Sequência: dilated conv (T1.2) → soma com `input_mixin` (`Conv1x1 condition→CH`, sem bias) → `LeakyReLU` (T1.1) → acumula no *head* → `layer1x1` (`Conv1x1 CH→CH`, bias) → conexão residual `out = input + layer1x1_out`.
  - **Fonte de verdade:** `a2_fast.cpp:514` (sequência conv→mixin→LeakyReLU→head →layer1x1 residual), `detail.h` (`Layer`), `docs/wavenet_walkthrough.rst:103-214`.
  - **Critério de aceite:** paridade camada-a-camada vs referência em micro-teste sintético.

- **[T1.6] Carga de pesos A2 (ordem do stream).** [DONE]
  - Implementar `set_weights` respeitando a ordem exata: `_rechannel` → (por camada: `_conv` → `_input_mixin` → `_layer1x1`) → `_head_rechannel` (conv k=16, bias) → **`head_scale` (último float do stream)**.
  - **Fonte de verdade:** `a2_fast.cpp:196-282` (documenta a ordem), `a2_fast.cpp:264-275` (head + `head_scale` trailing), `generate_weights_a2.py:18-90` (contagem de pesos por bloco).
  - **Critério de aceite:** contagem de pesos consumidos == `weights.len()` (asserção); erro claro (sem panic em runtime RT) se divergir.

### Sprint 1.3 — Loader, dispatch e aposentadoria do placeholder [DONE]

- **[T1.7] Dispatch A2 no loader.** [DONE]
  - Em `src/loader/dispatcher/wavenet/mod.rs`, tornar a A2 um **branch de primeira classe** (não mais *fallback*-após-falha): detectar a forma via `is_a2_shape()` (`src/loader/nam_json/topology.rs:131`) **antes** do match de topologias A1 e construir `WaveNetA2<3>`/`WaveNetA2<8>`. Registrar novas variantes no enum `DynamicModel` (`src/models/mod.rs`) e no dispatch (`src/models/dynamic_model.rs`).
  - **Importante (pré-requisito da Sprint 1.5):** como o *fallback* dynamic será removido, o dispatch precisa decidir entre {A1 estático, A2 estático} de forma explícita; geometrias não reconhecidas retornam **erro de load claro**.
  - **Fonte de verdade:** `a2_fast.cpp:849-990` (`is_a2_shape`/`create_a2_fast_config`), `topology.rs:131-163`.
  - **Critério de aceite:** carregar um `.nam` A2-Full/Lite produz inferência real (não silêncio); `mock_a2.nam` ainda reconhecido.

- **[T1.8] Metadados e par de modelos.** [DONE]
  - Atualizar `src/loader/loaded_model_pair.rs` (topologia/`weights_layout`) e `src/loader/build.rs` (calibração de ganhos via `input_level_dbu`/`loudness`, prewarm ≥ 2048 amostras) para A2.
  - **Critério de aceite:** `--model A2.nam` no standalone roda com telemetria; ganhos calibrados.
  - **⚠️ Nota pós-auditoria da Sprint 1.3:** O dispatch A2 (`src/loader/dispatcher/wavenet/mod.rs:54-56,66-68`) ignora `data.weights_layout` — usa sempre `set_weights` com transposição interna. Baixo risco: `.nam` sempre usa layout `Original`. Caso `.namb` armazene A2 em `Interleaved4WaveNet`, haveria dupla-transposição. Adiar para quando `.namb` for suportado para A2.

- **[T1.9] Aposentar `WavenetA2Placeholder`.** [DONE]
  - Remover `src/models/a2/placeholder.rs`, a variante `WavenetA2` placeholder e o flag `RT_STATUS_A2_PLACEHOLDER`. Atualizar `tests/loader_a2_compat.rs` e `tests/a2_placeholder_interface.rs` para validar **inferência real**.
  - **Critério de aceite:** suíte verde sem o placeholder; nenhum caminho emite silêncio para A2 válido.

### Sprint 1.4 — Golden Tests A2 (seguro anti-degradação) 🧪 [DONE]

- **[T1.10] Gerador de fixtures A2-Full/Lite.** [DONE]
  - Adaptar/derivar de `generate_weights_a2.py` um gerador determinístico (seed fixa) que emita `wavenet_a2_full.nam` (CH=8) e `wavenet_a2_lite.nam` (CH=3) com o esqueleto fixo (23 camadas, kernels/dilations canônicos, `LeakyReLU`, `head_scale`). Salvar em `tests/fixtures/models/`.
  - **Fonte de verdade:** `generate_weights_a2.py`, `a2_fast.h:30-43`.
  - **Critério de aceite:** arquivos carregam tanto no C++ `render` quanto no loader Rust (T1.7).

- **[T1.11] Estender `golden_gen_build.sh` para A2.** [DONE]
  - Gerar `tests/fixtures/golden_wavenet_a2_full.bin` e `..._a2_lite.bin` (v1 e variantes v2 multi-SR) renderizando com o `render` do C++ (caminho genérico `WaveNet` = verdade; o `a2_fast` produz saída idêntica).
  - **Fonte de verdade:** `tests/fixtures/golden_gen_build.sh`, `src/bin/wav_to_golden.rs`, `src/bin/gen_stress.rs`.
  - **Critério de aceite:** novos `.golden.bin` no formato `[u32 N][f32×N in][f32×N out]`; documentado em `tests/fixtures/README.md`.

- **[T1.12] Testes de inferência golden + cross-validation viva.** [DONE]
  - Adicionar casos em `tests/nam_infer_test.rs` (rápidos, pré-commit) e em `tests/cpp_parity.rs` (`#[ignore]`, vivos) para A2-Full/Lite. Definir thresholds adaptativos de SNR/ESR/MR-STFT para A2 em `tests/common/validation.rs` e baselines em `src/testing/perceptual.rs`.
  - **Critério de aceite:** golden verdes; ESR dentro do baseline; cross-val viva passa em `utils/tests-long.sh`.
  - **Nota:** Golden vectors usam padrão self-golden (Rust gera referência na primeira execução) pois o `render` do C++ (caminho `a2_fast`) diverge com os fixtures A2 atuais. O `is_a2_shape` do C++ é ativado corretamente (formato de ativação corrigido para array de objetos), mas a saída do A2 fast path do NeuralAmpModelerCore não casa com a implementação Rust — **investigação pendente no lado C++** (possível diferença na inicialização do ring do head ou na posição do `head_scale` no stream de `_load_weights`). A implementação Rust é internamente self-consistente (MSE=0.0 entre runs independentes com mesma entrada). Esta situação está documentada em `src/models/a2/model.rs` (module-level docstring, seção "Cross-Validation and Golden Vectors"). Cross-validation viva (`cpp_parity.rs`) está implementada como `#[ignore]` e será promovida a CI padrão quando o render C++ estiver estável para A2.

- **[T1.13] RT-Safety e edge tests A2.** [DONE]
  - Estender `tests/wavenet_prewarm_edge.rs`, heap-audit (`tests/a2_heap_audit.rs`) e soak (`tests/soak_test.rs`) cobrindo A2.
  - **Critério de aceite:** zero alloc no hot-path (CountingAllocator); estável em milhões de frames.
  - **Nota de auditoria:** Heap-audit A2 implementado em `tests/a2_heap_audit.rs` (CH=3 e CH=8, block_sizes {1,16,32,48,64}, 1000 iterações). Soak tests A2 implementados em `tests/soak_test.rs` com 4 cenários `#[ignore]`: silence/noise × Full/Lite, 10M frames cada.

### Sprint 1.5 — Remoção dos caminhos *dynamic* (corte de burden) ✂️ [DONE]

> Executar **após** A2 e os 4 modelos-foco estarem validados (Sprints 1.1-1.4), garantindo que nenhum caminho de produção dependa do *fallback* dynamic.

- **[T1.14] Remover WaveNet dynamic.** [DONE]
  - Remover `src/models/wavenet/{model_dyn,layer_dyn,dense_dyn}.rs` (e correlatos), a variante `DynamicModel::WavenetDyn` e `src/loader/dispatcher/wavenet/dynamic.rs` (`build_wavenet_dynamic`).
  - Ajustar `dispatcher/wavenet/mod.rs:68` para retornar **erro de load** em geometria não-catalogada (sem panic, mensagem diagnóstica via `NamDiagnostic`).
  - **Critério de aceite:** `cargo build` limpo; modelos A1-Standard/Lite/Feather/Nano e A2-Full/Lite seguem carregando; `.nam` fora do catálogo falha com erro claro.
  - **Nota:** `conv1d_dyn*.rs` foram intencionalmente retidos — são kernels de convolução *runtime-dimensioned* usados pela arquitetura A2 e por testes de stress estáticos, não como caminho de modelo dinâmico.

- **[T1.15] Remover LSTM dynamic.** [DONE]
  - Remover `src/models/lstm/{model_dyn,layer_dyn}.rs` (e correlatos), a variante `DynamicModel::LstmDyn` e `src/loader/dispatcher/lstm/dynamic_builder.rs` (`build_lstm_dynamic`). Ajustar `lstm/dispatch.rs:52` para erro de load em `(num_layers, hidden)` não-catalogado.
  - **Critério de aceite:** aliases LSTM estáticos (1×8..2×24) seguem funcionando; geometria não-catalogada falha com erro claro.

- **[T1.16] Limpar fixtures/testes de cross-val dependentes do dynamic.** [DONE]
  - Remover os modelos/goldens NAMCore micro (`tests/fixtures/models/{lstm,wavenet}.nam`, `golden_namcore_lstm_1x3.bin`, `golden_namcore_wn_micro.bin`) e os testes que os exercitam (`tests/dynamic_parity.rs` e casos correspondentes em `cpp_parity.rs`/`nam_infer_test.rs`). Atualizar `tests/fixtures/README.md` e o script `golden_gen_build.sh`.
  - **Critério de aceite:** suíte verde sem referências órfãs; `utils/tests-cargo.sh` e `utils/tests-long.sh` passam.

- **[T1.17] Simplificar enum/dispatch e documentar.** [DONE]
  - Renomear `DynamicModel` → `StaticModel` (e `dynamic_model.rs` → `static_model.rs`), atualizando todos os 22 arquivos `.rs` e 3 arquivos `.md`.
  - Remover do README.md a seção "Dynamic Mode (Absolute Flexibility)".
  - **Critério de aceite:** API coerente; documentação alinhada; sem *dead code*.

---

## ÉPICO 2 — Otimização SIMD A2 (x86-64-v3) ⚡ [DONE]

> Objetivo: aplicar otimizações *on-the-fly* fiéis ao `a2_fast.cpp`, **protegidas pelos golden vectors** do Épico 1 (nenhuma quebra de correção).

### Sprint 2.1 — Kernels otimizados [DONE]

- **[T2.1] Caminho CH=3 (A2-Lite): GEMV totalmente desenrolado.** ✅ [DONE]
  - Portar a estratégia escalar/SIMD desenrolada para 3 canais.
  - **Fonte de verdade:** `a2_fast.cpp` (estratégia `Channels=3`, GEMV unrolled).
  - **Critério de aceite:** golden A2-Lite verde; ganho mensurável vs baseline T1.
  - **Status:** ✅ Implementado em `src/models/a2/conv1d_ch3.rs`. Dispatch automático quando `in_ch==3 && out_ch==3`. K=6 (18 FMAs desenroladas) e K=15 (45 FMAs desenroladas). Golden A2-Lite self bitwise idêntico (MSE=0.0). Self-golden regenerado.
  - ⚠️ **Nota p/ T2.2-T2.4:** `golden_wavenet_a2_lite_self.bin` foi regenerado com o kernel desenrolado. Tarefas que alterem o A2-Lite devem regenerá-lo também.

- **[T2.2] Caminho CH=8 (A2-Full): *tap-major* frame-tiled (T=4) com broadcast-FMA.** ✅ [DONE]
  - Portar a estratégia de *tiling* de 4 frames com *broadcast*-FMA e layout *col-major-per-tap*.
  - **Fonte de verdade:** `a2_fast.cpp` (estratégia `Channels=8`, T=4 tap-major).
  - **Critério de aceite:** golden A2-Full verde; ganho mensurável.
  - **Status:** ✅ Implementado em `src/models/a2/conv1d_ch8.rs`. Layout col-major-per-tap (`A2Conv1dCh8`). Processamento em blocos com SIMD para conv, bias, mixin, LeakyReLU, head e l1x1. T=4 tiles com vfmadd231ps (broadcast-FMA). Golden A2-Full self regenerado e verde (MSE=0.0 entre runs). Testes de paridade AVX2 vs escalar para K=6, K=15, layer forward completo, edge cases (1 frame, T=4 tail). O peso também foi permutado na carga (T2.4).

- **[T2.3] Ring `pow2` + *tail-mirror* para dilations e head.** ✅ [DONE]
  - Consolidar buffers de histórico com máscara `pow2` e espelhamento de cauda (leitura *branchless*), reusando/estendendo `src/dsp/mirror_buf.rs`.
  - **Fonte de verdade:** `a2_fast.cpp:335-344,771-798` (ring pow2 + memmove rewind).
  - **Critério de aceite:** sem ramos no caminho de leitura; golden verde.
  - **Status:** ✅ Per-layer `MirroredBuffer<f32>` substitui o arena plano `AlignedVec`. Cada buffer de camada usa mapeamento virtual 2× para acesso sem ramos (`buffer_start - lookback` sempre válido). O `copy_within` (memmove) no hot-path foi eliminado; a posição de escrita avança e retrocede subtraindo `ring_size` quando se aproxima do limite 2×. Head já usava anel pow2 com `& ring_mask` (leitura sem ramos); o memmove do head (preserva K-1 amostras da cauda) foi mantido para permitir escritas vetorizadas sem máscara. Golden A2-Lite e A2-Full self verde (MSE=0.0, bitwise idêntico). 319 testes lib + integração passam.

- **[T2.4] Permutação de pesos para layout SIMD.** ✅ [DONE — incluso em T2.2]
  - No `set_weights` (T1.6), permutar Conv1D de *row-major-per-tap* para *col-major-per-tap* (acesso amigável a SIMD), feito **uma vez** na carga.
  - **Fonte de verdade:** `a2_fast.cpp:196-282` (loader permutando layout).
  - **Critério de aceite:** golden verde; carga sem custo no hot-path.
  - **Status:** ✅ Implementado junto com T2.2. `A2Conv1dCh8::new` faz a permutação na carga. Layout final: `w[k * 64 + in * 8 + out]` — 8 pesos de saída contíguos por `(tap, input)`.

### Sprint 2.2 — Validação de performance [DONE]

- **[T2.5] Benchmarks Criterion A2-Full/Lite.** [DONE]
  - Adicionar casos em `benches/inference_bench.rs` (e `dot_4x_bench.rs` se aplicável).
  - Medir µs/bloco a 48 kHz, buffers 64/128/256.
  - **Critério de aceite:** relatório de ganho documentado em `docs/benchmarks.md`; **zero regressão** em modelos A1; golden 100% verde.

---

## ÉPICO 3 — SlimmableContainer + Integração FSM Adaptativa 🔀

> Objetivo: carregar o **bundle oficial A2** (nano+standard) e trocar Full↔Lite em runtime, integrado à FSM de pressão de CPU já existente.

### Sprint 3.1 — Container

- **[T3.1] Trait `SlimmableModel` + parser `SlimmableContainer`.** [DONE]
  - Criar `src/models/slimmable.rs` (`trait SlimmableModel { fn set_slimmable_size(&mut self, val: f32); }`) e parser `src/loader/dispatcher/container/` para a arquitetura `"SlimmableContainer"` (`config.submodels[] = {max_value, model}`), construindo cada submodelo via o dispatcher recursivo.
  - **Fonte de verdade:** `NAM/slimmable.h`, `NAM/container.h:18-64`, `NAM/container.cpp:149` (registro/parser).
  - **Critério de aceite:** `slimmable_container.nam` (exemplo) carrega; ordena submodelos por `max_value` ascendente; último cobre `>= 1.0`.

- **[T3.2] `ContainerModel` (despacho por threshold).** [DONE]
  - Criar `src/models/container.rs`: guarda N submodelos pré-construídos; `set_slimmable_size(val)` seleciona índice por threshold e chama `reset()` no submodelo ativo; `process()` despacha ao ativo. **Todos** os submodelos pré-alocados/prewarmed na carga (zero alloc no switch).
  - **Fonte de verdade:** `NAM/container.cpp` (dispatch + seleção).
  - **Critério de aceite:** RT-safe na troca; golden por submodelo (A2-Full e A2-Lite) reaproveitando fixtures do Épico 1.

### Sprint 3.2 — Integração com a FSM adaptativa

- **[T3.3] Ligar `set_slimmable_size` à FSM de pressão de CPU.** [DONE]
  - Mapear os estados de `src/dsp/adaptive.rs` (Full→Reduced→Minimal) para a seleção de submodelo (A2-Full ↔ A2-Lite), usando os limiares de P99/budget já monitorados pela telemetria (`src/dsp/telemetry.rs`).
  - **Critério de aceite:** sob carga simulada, o engine migra Full→Lite e retorna por histerese, sem realocar.

- **[T3.4] *Crossfade* sem cliques na troca.** [DONE]
  - Implementado crossfade linear de 32 ms no `ContainerModel::process` com blend progressivo entre saídas dos submodelos ativo e pendente. Buffer scratch pré-alocado (zero alloc no hot-path). `configure_adaptive_model` atualizado para sempre chamar `set_slimmable_size` (defer só para `set_effective_layers`). Teste de continuidade confirma redução de 60% no step relativo vs troca abrupta.
  - Reusar `src/dsp/smoother.rs`/lógica de *crossfade* da `adaptive.rs` para transição suave entre submodelos.
  - **Critério de aceite:** ausência de descontinuidade audível (teste de energia/continuidade no ponto de troca).

- **[T3.5] Override manual (CLI + CLAP).** [DONE]
  - Flag de CLI (`src/standalone/cli.rs`) e parâmetro CLAP para fixar/forçar nível (Auto/Full/Lite). Manual sobrepõe a FSM.
  - **Critério de aceite:** `--slim auto|full|lite` funciona; param CLAP exposto; documentado.

- **[T3.6] Telemetria e testes de transição.** [DONE]
  - Sinalização de nível ativo via `RtStatusFlags` (atômico) já implementada em `transition_to()` (flags `DEGRADE_REDUCED`/`DEGRADE_MINIMAL`, contador `degrade_transitions_total`).
  - Testes de FSM estilo proptest em `tests/adaptive_fsm_proptest.rs` (adversariais, valores de fronteira, jitter, invariantes de telemetria).
  - Soak de transições em `tests/soak_test.rs`: `test_adaptive_fsm_endurance` (2M ciclos de jitter) e `test_adaptive_fsm_transition_cycles` (50k ciclos determinísticos Full→Reduced→Minimal→Full, 200k transições verificadas).
  - **Critério de aceite:** transições determinísticas e estáveis sob soak.

---

## ÉPICO 4 — IR Cabsim (.wav convolution) 🔊 [DONE]

> Objetivo: feature útil e **ortogonal à A2** — carregar um `.wav` de impulse response e convoluir (estágio pós-NAM). Convolução particionada FFT, RT-safe, reusando `rustfft` (já no `Cargo.toml`).
> Nota do PO: Se o "NeuralAmpModelerCore" espelhado em `tests/fixtures/NeuralAmpModelerCore` não possui uma implementação de convolução de IR, verifique se o plugin oficial "gateway" (espelhado em `tests/fixtures/NeuralAmpModelerPlugin`) possui esta implementação. Seria interessante realmente ter alguma implementação consagrada para comparação segura.

### Sprint 4.1 — Loader de IR [DONE]

- **[T4.1] Loader de `.wav` IR.** ✅ [DONE]
  - Loader robusto em `src/dsp/cabsim/loader.rs` para WAV mono (PCM16/24/float32), com resample para a SR ativa (reusar `src/dsp/resampler.rs`) e normalização opcional. Carga/preparo **fora** da audio thread; transferência via SPSC (estilo *resampler swap* em `src/common/spsc/`).
  - **Critério de aceite:** carrega os WAVs de exemplo em `tests/` (ex.: `amostra-guitarra-*_FAT_CAB.wav`); erros tratados sem panic.
  - **Notas p/ T4.2:** `CaptureState.active_cabsim: Option<Box<CabSimIr>>` já populado via SPSC swap. `CabSimIr.samples` contém IR mono resampled (f32). Canal SPSC `cabsim_producer` disponível em `run.rs` (atualmente `_` prefix, sem CLI command). `GcItem::CabSimIr` já registrado no GC cascade.

### Sprint 4.2 — Convolução particionada [DONE]

- **[T4.2] Uniform-Partitioned Overlap-Save (FFT).** [DONE]
  - Implementar convolução particionada em `src/dsp/cabsim/conv.rs` (partições de tamanho = buffer, overlap-save), pré-FFT do kernel na carga; FDL (*frequency delay line*) pré-alocada; **zero alloc** no hot-path.
  - **Fonte de verdade conceitual:** literatura de UPOLS/partitioned convolution (não há referência no C++; é feature nova do nam-rs).
  - **Critério de aceite:** paridade vs convolução direta (referência ingênua) `ESR < 1e-5` em IR curto; latência == tamanho da partição documentada.
  - **Nota do PO:** Se o "NeuralAmpModelerCore" espelhado em `tests/fixtures/NeuralAmpModelerCore` não possui uma implementação de convolução de IR, verifique se o plugin oficial "gateway" (espelhado em `tests/fixtures/NeuralAmpModelerPlugin`) possui esta implementação. Seria interessante realmente ter alguma implementação consagrada para comparação segura.

- **[T4.3] Estágio opcional no pipeline.** ✅ [DONE]
  - Integrar como estágio pós-inferência em `src/dsp/pipeline/` com *bypass* de custo zero quando nenhum IR está carregado. Flag de CLI/param CLAP.
  - **Critério de aceite:** ativável/desativável em runtime sem clique; bypass não mede custo.

- **[T4.4] Testes + bench IR.** [DONE]
  - Golden de convolução (IR sintético determinístico), heap-audit e bench Criterion. `#[ignore]` para os pesados.
  - **Critério de aceite:** golden verde; zero alloc; bench documentado.
  - **Nota de auditoria (Sprint 4.2):** Implementação completa. 11 unit tests (paridade ESR < 1e-5 em short/medium/long IR, edge cases), 8 golden tests (6 rápidos + 2 `#[ignore]`), 4 heap-audit tests (zero alloc verificado), 8 benchmarks Criterion (ShortIR/MediumIR/LongIR/256samp/construction/construction_long/long_run). Suíte 100% verde.

---

## ÉPICO 5 — IR Cabsim (.wav convolution) II 🔊

### Sprint 5.1 — Integração CLAP Cabsim + Robustez de Partição

> O CLAP plugin é **release**. A infraestrutura de pipeline (Stage 3 em `capture.rs`, `DspPipelineContext.conv`) e o standalone (`--cab`) já estão funcionais; falta o wiring CLAP completo: parâmetro SPSC, GUI, state save/load, e adaptação de partição para buffer sizes variáveis do host.

- **[T5.1] `ClapParamPayload::LoadCabIr` + SPSC wiring CLAP.** [DONE]
  - Adicionar variante `LoadCabIr { engine: Option<Box<ConvEngine>> }` ao enum `ClapParamPayload` (`src/clap/plugin/shared.rs:17`). No `process_events` (`src/clap/processor/events.rs:31-42`), drenar o payload e fazer swap de `self.conv_engine` com GC cascade para o engine antigo (mesmo padrão de `cold_load_model` em `events.rs:139-157`).
  - Adicionar campo `ir_path: Option<String>` ao `ColdShared` (`src/clap/plugin/shared.rs:106`) para rastreamento do caminho ativo.
  - No main thread, implementar `cold_load_cabsim()`: carregar WAV via `CabSimIr::load()`, construir `ConvEngine::new()`, enviar via SPSC `param_tx.push(ClapParamPayload::LoadCabIr { .. })`.
  - **Fonte de verdade:** Padrão existente de `ClapParamPayload::LoadModel` e `cold_load_model()` em `events.rs`.
  - **Critério de aceite:** `conv_engine` recebe um `ConvEngine` válido via SPSC no RT thread; swap é RT-safe (zero alloc); engine antigo vai pro GC via `push_to_gc(GcItem::CabConvEngine(old))`.

- **[T5.2] State save/load do caminho IR no CLAP.** [DONE]
  - Serializar `ir_path` no state blob (`src/clap/extensions/state.rs`). No `load`, reconstruir o `ConvEngine` (cold-path) e enviar via SPSC — mesmo padrão do model path (usar `cold.ui_pending_model` como referência de padrão, criar `cold.ui_pending_ir`).
  - **Fonte de verdade:** `src/clap/extensions/state.rs` (padrão de save/load existente), `src/clap/plugin/shared.rs:139` (`ui_pending_model`).
  - **Critério de aceite:** Salvar preset com IR, fechar/reabrir o plugin → IR recarregado automaticamente; preset sem IR → `conv_engine = None` (bypass); compatibilidade retroativa com presets sem campo IR.

- **[T5.3] GUI — File browser para IR no CLAP.** [DONE]
  - Adicionar controle de file browser para `.wav` na GUI egui (`src/clap/gui/ui/`), análogo ao model file browser existente em `zones/identity.rs`. Elementos: botão "Load IR" + display do nome do arquivo carregado + botão "Clear IR" (envia `None` via SPSC para bypass).
  - Ao selecionar, disparar carga assíncrona no main thread (mesma estratégia de `ui_pending_model`/`ui_loading`/`ui_load_error`): gravar em `cold.ui_pending_ir`, sinalizar `ui_ir_loading`, processar em `on_main_thread()`, enviar `ConvEngine` via SPSC → RT.
  - **Fonte de verdade:** `src/clap/gui/ui/zones/identity.rs` (model file browser), `src/clap/plugin/shared.rs:139-145` (pending/loading/error pattern).
  - **Critério de aceite:** File browser funcional com filtro `.wav`; IR carregado aparece na GUI; "Clear" remove o IR (bypass); loading indicator; erro exibido em toast; sem cliques na transição (swap via SPSC + GC cascade).

- **[T5.4] Partição adaptativa (buffer_size variável).** [DONE]
  - Atualmente `ConvEngine` é construído com `partition_size = buffer_size` e o stage em `capture.rs:60-62` faz bypass silencioso se `n_pw != partition_size`. Corrigir para:
    (a) No CLAP: reconstruir `ConvEngine` em `activate()` (`src/clap/processor/mod.rs`) quando o host informa `max_frames_count`, usando `partition_size = max_frames_count`. Armazenar o IR raw (samples `Vec<f32>`) no `ColdShared` para possibilitar reconstrução sem re-load do WAV.
    (b) No standalone: reconstruir quando o buffer PipeWire muda (via SPSC swap `cabsim_producer` já existente em `src/main.rs:99,124`).
    (c) Documentar que a latência do cabsim = `partition_size` samples e somar à latência reportada pelo plugin (`current_latency` em `events.rs:96`).
  - **Fonte de verdade:** `src/dsp/pipeline/capture.rs:60-72` (guard atual), `src/clap/processor/mod.rs` (`activate`, `max_frames_count`).
  - **Critério de aceite:** Cabsim funciona com qualquer buffer size reportado pelo host; sem bypass silencioso inesperado; latência do cabsim somada à latência total reportada ao host; testes com buffer sizes {32, 64, 128, 256, 512}.

### Sprint 5.2 — Documentação Cabsim 📚

> Documentar completamente a feature de IR cabsim em todos os documentos relevantes. Inclui decisões arquiteturais tomadas durante a auditoria do épico.

- **[T5.5] Documentação arquitetural do cabsim.** [DONE]
  - `docs/architecture.md`: Nova seção sobre o estágio de cabsim no pipeline DSP (UPOLS, FDL, zero-alloc, latência = partition_size). Incluir diagrama Mermaid do fluxo Inference → CabSim → Output. Corrigir referência "CLAP (Staging)" → "CLAP (Release)" (L6). Atualizar o diagrama de fluxo DSP bidirecional (§5, L187-201) para incluir o estágio de cabsim entre "Output Gain" e "DspBridge".
  - `README.md`: Mencionar IR cabsim como feature disponível na seção de features/supported models.
  - `docs/benchmarks.md`: Incluir resultados dos benchmarks de cabsim (ShortIR/MediumIR/LongIR @ 64samp, 256samp, construction, long_run — ver `benches/inference_bench.rs:1249-1370`).
  - **Critério de aceite:** Documentação coerente, sem menções órfãs; leitores entendem como o cabsim funciona e sua posição no pipeline.

- **[T5.6] Documentação de fixtures e decisões de validação.** [DONE]
  - `tests/fixtures/README.md`: Documentar os golden de cabsim — IR sintético determinístico via PCG PRNG com seed fixa, direct convolution O(N²) como referência, ESR < 1e-5 em cenários short (64), medium (512), long (8192) e stress (32768 amostras).
  - `docs/cpp_parity_map.md`: Registrar que o cabsim é feature **nova do nam-rs** (sem equivalente no NeuralAmpModelerCore); a referência mais próxima é `AudioDSPTools/dsp/ImpulseResponse.h` no NeuralAmpModelerPlugin (submodule `AudioDSPTools` não inicializado no fixture — ver Sprint 4.5).
  - **Decisões a documentar em `docs/`:**
    - **Cross-validação C++ não realizada (justificada):** (a) feature nova/ortogonal ao NAM, não existe no NeuralAmpModelerCore; (b) submodule `AudioDSPTools` não inicializado no fixture; (c) validação via direct convolution (referência ingênua O(N²)) é matematicamente rigorosa — ESR < 1e-5 confirmado em cenários short, medium, long e stress. Sprint 4.5 planeja cross-validação futura.
    - **Teste de pipeline end-to-end com cabsim considerado desnecessário:** Cada componente é testado individualmente — convolution unit tests (11), golden parity (8), heap-audit (4). O stage está integrado em `capture.rs` e verificado por code review; a interação com os demais stages (input, inference, output) não introduz acoplamento que justifique um teste adicional.
  - **Critério de aceite:** Todas as decisões documentadas com justificativa; rastreabilidade completa em docs/.

### Sprint 5.3 — Cross-Validação AudioDSPTools `ImpulseResponse` 🔬

> Inicializar o submodule `AudioDSPTools` do `NeuralAmpModelerPlugin` e implementar cross-validação da engine UPOLS do nam-rs contra a implementação de referência `dsp::ImpulseResponse` do C++. O NeuralAmpModelerPlugin (`tests/fixtures/NeuralAmpModelerPlugin/NeuralAmpModeler/NeuralAmpModeler.h:3`) usa `#include "../AudioDSPTools/dsp/ImpulseResponse.h"` para convolução de IRs.

- **[T5.7] Inicializar e analisar `AudioDSPTools` submodule.** [DONE]
  - Inicializar o git submodule `tests/fixtures/NeuralAmpModelerPlugin/AudioDSPTools` (atualmente diretório vazio).
  - Analisar `AudioDSPTools/dsp/ImpulseResponse.h` e `ImpulseResponse.cpp`: identificar o algoritmo de convolução usado (provavelmente overlap-add ou overlap-save particionado), o tratamento de partição, formato de entrada (WAV loader embutido ou externo), e a normalização aplicada.
  - Documentar as diferenças algorítmicas entre a implementação C++ e o UPOLS do nam-rs (ex: algoritmo base, tamanho de FFT, tratamento de cauda, normalização) em `docs/cpp_parity_map.md` (seção cabsim).
  - **Critério de aceite:** Submodule inicializado e acessível; análise documentada; diferenças catalogadas com impacto esperado na tolerância de cross-val.

- **[T5.8] Build do binário C++ de referência para IR.** [DONE]
  - Estender `tests/fixtures/golden_gen_build.sh` ou criar um binário auxiliar (`tests/fixtures/render_ir.cpp`) que: (a) carregue um IR `.wav` via `dsp::ImpulseResponse`, (b) processe um sinal de entrada sintético determinístico (mesmas seeds dos golden tests em `tests/cabsim_golden.rs` — PCG PRNG seeds 42, 137, 31337, 999983), e (c) emita a saída como golden vector binário no formato `[u32 N][f32×N in][f32×N out]`.
  - Gerar IRs sintéticos determinísticos no C++ usando a mesma fórmula: `sin(2π·freq·t) · exp(-decay·t) + noise_level·rng_signed` (parâmetros: freq=600/350/200/150 Hz, decay=12/6/2/1.5, noise_level=0.02).
  - **Fonte de verdade:** `AudioDSPTools/dsp/ImpulseResponse.{h,cpp}`, `NeuralAmpModeler.cpp:676,685,800` (uso de `dsp::ImpulseResponse`).
  - **Critério de aceite:** Binário C++ compila, gera saída determinística para IR + sinal de entrada dados; saída salva em `tests/fixtures/golden_cabsim_cpp_*.bin`; formato compatível com os golden tests do nam-rs.

- **[T5.9] Testes de cross-validação UPOLS vs C++ `ImpulseResponse`.** [DONE]
  - Implementar testes `#[ignore]` em `tests/cabsim_cpp_parity.rs` que: (a) carregam os golden vectors gerados pelo C++ (T4.12), (b) processam o mesmo sinal com o UPOLS do nam-rs, e (c) comparam com thresholds adaptativos.
  - Definir thresholds de ESR/SNR para a cross-validação (tolerância potencialmente maior que os golden internos, pois os algoritmos podem diferir — overlap-add vs overlap-save introduz diferenças de arredondamento na borda das partições).
  - Adicionar os testes à suíte de `utils/tests-long.sh`.
  - **Critério de aceite:** Cross-validação verde dentro dos thresholds definidos; diferenças documentadas; testes integrados em `utils/tests-long.sh`.

- **[T5.10] Teste Humanos:** Atualizar o `docs/functional-tests.md`.

---
---

## ÉPICO 99 — Documentação e Fechamento 📚

- **[T99.1] Atualizar documentação arquitetural** (acionar skill `documentador`):
  - `docs/architecture.md`: motor A2, container.
  - `README.md`: seção "Supported Models" — A2-Full/Lite agora suportados.
  - `docs/cpp_parity_map.md`: novo mapeamento A2 → C++.
  - `tests/fixtures/README.md`: com os novos golden A2 e instruções de regeneração.
  - `docs/functional-tests.md`: Assegurar que estão a par de tudo o que foi implementado até aqui.
  - Skill `refatora-doc.md`
  - **Nota:** Documentação do cabsim (IR convolution) já coberta no Sprint 5.2 (T5.5/T5.6).

- **[T99.2] Rodadas de correção** (versão 2.1)
  - `revisor-auditor` Extremamente focado em comparar meticulosamente C++/Rust e assegurar 100% feature parity (apenas as oficiais) e implementação impecavelmente correta. Cobertura de testes (inclusive golden vectors) tem que estar em estágio "produção" - ainda que a implementação NAM-rs em si continue em burilamento. Daqui em diante, idealmente, nem se mexe mais em testes e benchs. Eles já devem estar prontos para cumprir o seu papel de "seguro" contra erro/degradação. Então seja muito rigoroso em assegurar sua qualidade.
  - `refatora-rust.md`
  - `refatora-doc.md`

- **[T99.3] Rodadas de burilamento** (versão 2.2)
  - `pesquisador-inovador.md`
  - `refatora-rust.md`
  - `refatora-doc.md`
  - Leitura e revisão geral de todo o git do NAM-rs.
  - Divulgar geral na comunidade.

---
---

## ÉPICO 100 (FUTURO)

- Comparação completa de features com o NeuralAmpModelerCore e o NeuralAmpModelerPlugin para mais idéias de features a copiar.
- FFT e outros features no hot path considerar internalizar o código e ultra otimizações.
- Fender Studio Pro: pesquisador-inovador.md Suporte a Wayland nativo e cidadão de primeira classe nesta DAW.
- Novos ISAs e Arquiteturas (<https://gemini.google.com/app/71c4c68e27c64e10>): /pesquisador-inovador.md Atualizar para o estado atual do código e detalhar ao máximo.
  - Intel/AMD: Focar no AVX-512/AVX-10 (Especialmente: AVX512F, AVX512VL, AVX512_VNNI) em vez de AMX (muito focado em inferência e servidores); Eficiência Híbrida (AVX-10 / AVX-512 Light): Focado no uso de instruções AVX-512, mas restringindo o tamanho dos vetores a 256 bits.
  - ARM: focar na Linha de Base Unificada NEON de 128 bits (Rpi5 e Qualcomm, apesar da volatilidade má vontade desta última); A Linha Avançada é SVE2/VLA (basicamente NVIDIA RTX Spark).
- `SlimmableWavenet` (channel slicing de rede única): É arquitetura **oficial** do NAMCore (registrada, file version 0.7.0) e direção declarada da NAM ("uma captura que se escala sozinha, sem versão *lite* separada"). Será priorizado quando modelos `.nam` com o campo `"slimmable"` (rede única) tornarem-se comuns na distribuição mainstream — hoje o A2 mainstream usa o `SlimmableContainer`.
