<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Mapa de Features do nam-rs — Aderência ao NeuralAmpModelerCore 🧩

> **Propósito**: mapeia **lacunas de funcionalidade** do `nam-rs` frente ao NAMCore v0.5.3.
>
> **Última auditoria**: 19/jun/2026 (revisor-auditor + planejador-arquiteto).
>
> **Filosofia**: o `nam-rs` quer **superar** o original, mas a superação deve ser **correta**:
> nenhum recurso oficial silenciosamente abandonado; RT-safety preservada; paridade
> auditável (ESR/SNR vs C++ v0.5.3).
>
> **Otimizações**: não é obrigatório ser excessivamente pedante nesta fase.
> Otimizações profundas serão tratadas adiante. Porém, otimizações **evidentes e
> imediatas** (como vetorização de hot-paths escalares) devem ser feitas já.
>
> **Baseline ISA**: x86-64-v3 (AVX2+FMA). Todo código deve tirar o máximo proveito
> destas instruções. Multiversioning para x86-64-v4 ou superior será tratado adiante.

---

## Sumário Ativo — Features Pendentes

| ID      | Feature ausente / parcial                                               | Impacto p/ Produto | Esforço |
| ------- | ----------------------------------------------------------------------- |:------------------:|:-------:|
| **F2**  | **Multi-condição / FiLM** (`condition_size > 1`, `condition_dsp`)       | 🟢 Concluído       | Alto    |
| **F3**  | **Motor A2 geral** (gating, ativações heterogêneas, `head1x1`, `bn≠ch`) | 🟢 Concluído       | Alto    |
| **F5**  | **SlimmableWavenet** (slicing dinâmico de canais)                       | 🟠 Médio-Alto      | Alto    |
| **F6**  | **Post-stack Head** (sub-objeto `head` multi-camada do WaveNet)         | 🟡 Médio           | Médio   |
| **F7**  | **LSTM arbitrário** (`hidden_size`/`num_layers` fora dos 10 perfis)     | 🟢 Concluído       | Médio   |
| **F8**  | **Biblioteca completa de ativações** (PReLU, SiLU, etc.)                | 🟢 Concluído       | Médio   |
| **F9**  | **Convoluções agrupadas/depthwise** (`groups > 1`)                      | 🟢 Concluído       | Médio   |
| **F4**  | **ConvNet** (arquitetura legada)                                        | 🟢 Baixo           | Médio   |
| **F10** | **Modelos multi-canal** (`in/out_channels > 1`)                         | 🟢 Baixo           | Médio   |
| **F11** | **Container aninhado** + cobertura `SlimmableContainer` real            | 🟢 Baixo           | Baixo   |

> **Dependências**: F2 ⊃ F3 ⊃ {F8, F9}. F7 é ortogonal. F5 depende de F2/F3 (FiLM/gating + SlimmableWaveNet). F6 é ortogonal mas é usado pelo WaveNet geral.

---

## Diagnóstico Verificado (probe de carga, jun/2026)

| Modelo oficial                | Resultado no nam-rs                                          | Feature que destrava       |
| ----------------------------- | ------------------------------------------------------------ | -------------------------- |
| `wavenet_a1_standard.nam`     | ✅ Carrega (golden oficial)                                  | —                          |
| `lstm.nam`                    | ✅ Carrega (golden oficial)                                  | —                          |
| `wavenet.nam` (CH=3, livre)   | ✅ Carrega (motor dinâmico)                                  | —                          |
| `slimmable_wavenet.nam`       | ❌ "A2 shape not recognized"                                 | **F5**                     |
| `wavenet_a2_max.nam` (cond=8) | ❌ "condition_size=8 not supported" (Aguardando F1)          | **F1** (Cond=8)            |
| `wavenet_condition_dsp.nam`   | ✅ Carrega (golden oficial c/ sub-modelo)                    | —                          |
| `slimmable_container.nam`     | ✅ Carrega (LSTM 1x3 + WaveNetDyn [3,2] + Nano [4,2])        | **F5 / F11** (Sprint 2.2)  |
| Modelos nondist 4-array       | ✅ Carrega (motor dinâmico de N arrays)                      | —                          |

---

## Achados da Re-Auditoria (jun/2026)

### Achado 1 — Motor Dinâmico WaveNet A1: Limite de 2 Layer Arrays

**Severidade**: 🟠 Funcional — modelos reais com >2 arrays rejeitados.

**Evidência**: `src/loader/dispatcher/wavenet/dynamic.rs:113-118` rejeita `num_arrays != 2`.
O teste `live_cross_validation_nondist_models` (`tests/cpp_parity.rs:346`) panickou
tentando carregar um modelo nondist com 4 layer arrays. Este é o único teste que
falha na suite `tests-long.sh` (Phase 2: FAILED, 22 passed / 1 failed).

**O que falta**:

- O NAMCore WaveNet model (`wavenet/model.cpp`) aceita N layer arrays arbitrários.
- O motor dinâmico do nam-rs precisa ser generalizado para N arrays.
- Isso não requer a generalização completa do A2, apenas loop sobre N layer arrays.

**Impacto**: modelos custom com >2 arrays (embora raros) são silenciosamente rejeitados.

---

### Achado 2 — A2 Head Conv: Hot-Path 100% Escalar

**Severidade**: 🟡 Desempenho — oportunidade de otimização evidente.

**Evidência**: `src/models/a2/head.rs:96-118` — laço aninhado `frames × 16 taps × CH`
usa `get_unchecked` escalar puro, sem nenhum intrínseco SIMD.

Para CH=8 são 128 FMAs escalares por frame. Para CH=3, 48 FMAs por frame.
Como o baseline é x86-64-v3 (AVX2+FMA), esta é uma vitória imediata.

**O que fazer**:

- CH=8: usar `_mm256_loadu_ps` + `_mm256_fmadd_ps` (8 FMAs em uma instrução).
  Padrão já utilizado com sucesso em `conv1d_ch8.rs:183-236`.
- CH=3: usar 128-bit SSE4.1 para 3 canais (mascaramento para os 4 lanes).
- Manter a referência escalar (`a2_head_block_scalar_ref`) como oracle de parity.

---

### Achado 3 — Ativações: Algumas Implementações Não Vetorizadas

**Severidade**: 🟡 Desempenho — oportunidade imediata em ativações comuns.

**Evidência**: `src/models/a2/activations.rs:91-161`:

- `HardTanh` (L92-95): loop `x.clamp(-1.0, 1.0)` — vectorizável com `vminps`/`vmaxps`.
- `FastTanh` (L99-101): loop escalar — o corpo racional é intrinsecamente SIMD-friendly.
- `HardSwish` (L135-139): loop escalar com clamp — vectorizável com `vminps`/`vmaxps`.
- `LeakyHardTanh` (L149-156): loop escalar com branches — vectorizável com masks branchless.

As funções `tanh_slice`, `relu_slice`, `prelu_slice`, `sigmoid_slice`, `silu_slice`,
`softsign_slice` delegam para `src/math/activations/` — verificar se já estão otimizadas.

**O que fazer**:

- Verificar e garantir que `src/math/activations/` usa padrões auto-vetorizáveis
  (chunks_exact, sem branches) para x86-64-v3.
- As ativações inline em `activations.rs` (HardTanh, FastTanh, HardSwish, LeakyHardTanh)
  devem ser refatoradas para seguir o mesmo padrão.

---

### Achado 4 — FiLM e Gating: Apenas Scaffolding (Stubs)

**Severidade**: ℹ️ Informativo — confirmado conforme esperado.

**Evidência**:

- `src/models/a2/film.rs`: structs `FiLMConfig`, trait `FiLMLayer` — sem implementação real.
- `src/models/a2/gating.rs`: enum `GatingMode`, structs `GatingActivationConfig`/
  `BlendingActivationConfig` — sem implementação real.

O NAMCore tem implementações completas em:

- `NAM/film.h:19-210`: FiLM com Conv1x1, scale+shift, batch processing.
- `NAM/gating_activations.h:25-246`: GatingActivation e BlendingActivation com
  buffers pré-alocados e paths otimizados.

**Impacto**: necessário para F2 (multi-condição) e F3 (motor A2 geral).

---

### Achado 5 — `condition_dsp`: Não Implementado

**Severidade**: 🟡 Funcional — impacta F2.

**Evidência**: O NAMCore suporta `condition_dsp` — um sub-DSP que transforma o
sinal de entrada em um vetor de conditioning antes de aplicar FiLM
(`NAM/wavenet/model.cpp:556-592`). O nam-rs menciona `condition_dsp` apenas em
`src/loader/nam_json/model.rs:102` como comentário.

**O que fazer**: implementar como parte de F2.

---

### Achado 6 — Conv1D/Conv1x1 com `groups > 1`: Não Implementado

**Severidade**: 🟡 Funcional — F9.

**Evidência**: O NAMCore `conv1d.h:32-130` suporta `groups` como parâmetro (depthwise,
grouped conv). O nam-rs assume `groups==1` em todas as convoluções.

**Impacto**: necessário para F3 (motor A2 geral) e F9. Também usado por FiLM
(`film.h:27`: `_cond_to_scale_shift(condition_dim, ..., groups)`).

---

### Achado 7 — ConvNet: Zero Implementação

**Severidade**: 🟢 Baixo impacto — arquitetura legada.

**Evidência**: O NAMCore tem `convnet.cpp`, `convnet.h` com `ConvNet` + `BatchNorm` +
`ConvNetBlock` + `_Head`. O nam-rs não tem nenhuma implementação de ConvNet
(zero matches em `src/`).

**Decisão**: fora de escopo até demanda real do mercado.

---

### Achado 8 — Cobertura de Testes: Gaps Identificados

**Severidade**: 🟠 Qualidade.

**Gaps identificados**:

1. **`tests-long.sh` Phase 2 FAILED**: o teste `live_cross_validation_nondist_models`
   falha porque o motor dinâmico rejeita modelos com >2 layer arrays. Este é um gap
   funcional real (Achado 1), não um bug de teste.

2. **Ativações A2 (F8)**: as ativações Tanh, HardTanh, FastTanh, ReLU, PReLU, Sigmoid,
   SiLU, HardSwish, LeakyHardTanh, Softsign existem em `activations.rs` mas apenas
   LeakyReLU é exercitada pelo fast-path A2. Os testes em `activations_test.rs`
   cobrem as implementações, mas não há golden C++ para validar paridade numérica
   das demais ativações no contexto A2.

3. **Modelos de borda**: não há teste golden para modelos com kernel_size não-padrão
   ou head_size não-padrão no motor dinâmico (apenas CH=3 com geometria livre).

4. **Container/Slimmable**: `tests/container_slimmable.rs` testa o ContainerModel
   mas depende de modelos reais para validar SlimmableWaveNet — que ainda não existe.

**O que está bem coberto** ✅:

- Golden vectors v1 (48kHz): 10 modelos WaveNet+LSTM+A2 com C++ parity.
- Golden vectors v2 (multi-SR): 44.1k, 48k, 88.2k, 96k, 192kHz para modelos-chave.
- Soak tests (51s), pipeline soak (23s): estabilidade numérica com >1M frames.
- Heap-audit: resampler, cabsim, A2, diagnostic bundle — zero allocations no RT.
- CLAP lifecycle, state migration, multi-instance, concurrency stress.
- Proptests: parsers, math, pipeline block, gate FSM, adaptive FSM.
- BF16 parity: LSTM gate e scalar.
- clap-validator: 19/19 passed (2 skipped — notas MIDI, que não se aplicam).

---

### Achado 9 — WaveNet A1 Dinâmico: Kernel Size e Head Size Hardcoded

**Severidade**: ℹ️ Informativo.

**Evidência**: O motor dinâmico (`src/models/wavenet/model_dyn.rs`) aceita `kernel_size`
e `head_size` arbitrários, mas o `FreeWavenetGeometry` (`topology.rs:30-41`) já propaga
esses valores. O problema do Achado 1 é especificamente o limite de 2 arrays.

---

### Achado 10 — `head_scale` Ausente no WaveNet A1

**Severidade**: ℹ️ Informativo — confirmado que WaveNet A1 não usa `head_scale`.

O `head_scale` é exclusivo da arquitetura A2 (`a2_fast.cpp:116`). O WaveNet A1
original (`model.cpp`) não tem `head_scale`. O nam-rs está correto.

---

## Diretrizes para Golden de Novas Features

- **Modelo oficial real, não sintético.** Usar os `.nam` oficiais que hoje falham como fonte.
- **Cross-reference C++ pinado, gate scale-invariant.** Validar por ESR/SNR (não MSE absoluto).
- **Conversão de `test_loader_gap_*` → golden positivo.** Migrar de "rejeita" para
  "casa com o C++" ao ganhar suporte.
- **Calibração por medição documentada.** Registrar em `get_calibrated_threshold`.
- **Cobertura multi-SR e determinismo.** Acrescentar aos gates v2 e determinismo bitwise.

---

## Plano de Implementação — Mega-Tópicos (Achados Organizados)

### MT1 — ✅ Infraestrutura de Conditioning e FiLM (F2 + F8 parcial + F9 parcial) Concluído (2026-06-19)

**Pré-requisitos**: nenhum (base para F3).
**Desafio**: 🔴 Alto — arquitetural, muitos módulos novos.
**Benefício**: 🔴 Crítico — destrava modelos A2 oficiais com `condition_size > 1`.

**Achados relacionados**: 4, 5, 6.

**Escopo detalhado**:

1. **Implementar FiLM real** (`src/models/a2/film.rs`):

   - Conv1x1 conditioning → scale[+shift], batch processing.
   - Paridade com `NAM/film.h` (211 linhas C++).
   - 8 pontos de inserção FiLM na layer A2 (`conv_pre_film`, `conv_post_film`,
     `input_mixin_pre_film`, `input_mixin_post_film`, `activation_pre_film`,
     `activation_post_film`, `layer1x1_post_film`, `head1x1_post_film`).
   - Cada ponto: `FiLMConfig { active, shift, groups }` → instância `FiLM` ou skip.
   - Implementar SIMD nativamente (x86-64-v3: `vfmadd231ps` para scale+shift).

2. **Implementar `condition_dsp`**:

   - Sub-DSP que transforma o sinal de entrada mono em vetor de conditioning.
   - Referência: `NAM/wavenet/model.cpp:556-592`.
   - O nam-rs precisa instanciar um DSP (WaveNet/LSTM/Linear) como `condition_dsp`.
   - RT-safety: o `condition_dsp` precisa ter seus buffers pré-alocados no load.

3. **Generalizar `condition_size` no loader**:

   - Remover a guarda `condition_size != Some(1)` em `topology.rs:338,593`.
   - Propagar `condition_size` como parâmetro dinâmico (não const-generic).
   - Manter fast-path `COND=1` como caso especial otimizado.

4. **Conv1x1 com `groups > 1`** (parcial de F9):

   - Necessário para FiLM (`_cond_to_scale_shift` pode ter `groups > 1`).
   - Implementar kernel grouped Conv1x1 com fast-path `groups==1`.

5. **Testes**:

   - Converter `test_loader_gap_*` (rejeição) → golden positivo.
   - Golden C++ para modelos `wavenet_a2_max.nam` e `wavenet_condition_dsp.nam`.
   - Paridade ESR/SNR vs C++ v0.5.3.

**Resumo da Implementação e Resultados**:

1. **Generalização de `condition_size` e Topologia**: Removida restrição de `COND=1`. Adicionado parsing dinâmico de `condition_size` no loader.
2. **Motor FiLM Completo**: Implementado e integrado via `FilmBlock` aos 8 pontos ativos da arquitetura A2. Zero branches (utilizando fallback genérico `empty()`) e nativamente vetorizado com `vfmadd231ps`.
3. **Parsing e DSP Condicional (`condition_dsp`)**: Sub-modelo aninhado adicionado a `WaveNetModelDyn`. DSP roda como pipeline condicional (mono -> sub-modelo multi-canal -> FiLM da camada de onda principal).
4. **Paridade C++ Absoluta**: Golden V1 e V2 gerados e validados (`wavenet_condition_dsp.nam`). SNR de ~139.5 dB alcançado. Performance sob 2ms verificada. Todas as validações `clap-validator` 19/19 OK.
5. **Bloqueios Identificados**: `wavenet_a2_max.nam` desbloqueou o gate de condição, mas caiu com precisão em modelo A2 sem dispatch A2 dinâmico. Motor dinâmico de A2 torna-se o próximo alvo principal (F3).

---

### MT2 — ✅ Motor A2 Geral (F3 + F8 + F9) Concluído (2026-06-19)

**Pré-requisitos**: MT1 (FiLM/conditioning).
**Desafio**: 🟠 Alto — generalização de um motor const-generic para dinâmico.
**Benefício**: 🟠 Alto — suporta qualquer modelo A2 (futuro-proof).

**Achados relacionados**: 2, 3, 4, 6.

**Resumo da Implementação e Resultados**:

1. **Motor A2 Dinâmico (`WaveNetA2Dyn`)**:
   - Implementado com buffers circulares pre-alocados para processamento `zero-alloc` e lock-free no hot-path.
   - Suporte completo a ativações heterogêneas (`gating` e `blending`), `head1x1` e `layer1x1` customizados.
   - Suporte a arquiteturas onde `bottleneck ≠ channels`.

2. **Gating e Blending (F8)**:
   - Implementados os modos de `Gating` e `Blending` com alocação estrita de `scratch buffers` (`z_scratch`, `gating_scratch`).
   - Parsing completo do `ActivationConfig` via `serde`.

3. **Convoluções Agrupadas (F9)**:
   - Suporte a `groups > 1` integrado à pipeline de convolução dinâmica (`A2Conv1d` enum).
   - Caminho otimizado (depthwise) instanciado dinamicamente.

4. **Golden Tests e Paridade Numérica**:
   - Modelos sintéticos C++ parity gerados (`a2_dynamic_gated_ch8.nam`, `a2_dynamic_blended_ch3.nam`).
   - SNR de ~103 dB (gating) e ~133 dB (blending) na validação cruzada.

> **Achado da Auditoria de Código:**
> Embora o motor funcione perfeitamente, o `revisor-auditor` identificou em `dynamic.rs` e `layer.rs` que o código das matrizes 1x1 (`layer1x1` e `head1x1`) está realizando laços internos pulando acessos de memória pelo tamanho do `channels` (`l1x1_w[u * ch + c]`), o que quebra a coerência de cache e a auto-vetorização. Isso gera um pequeno gargalo de desempenho desnecessário. Foi criada uma nova sprint de Otimização Imediata no TODO-sprints para tratar essa inversão de laço (DAXPY).

---

### MT3 — ✅ WaveNet A1 Dinâmico: Generalização para N Arrays (F1-ext) Concluído (2026-06-19)

**Pré-requisitos**: nenhum (ortogonal a MT1/MT2).
**Desafio**: 🟡 Médio — mudança localizada no motor dinâmico.
**Benefício**: 🟠 Alto — destrava modelos reais com >2 arrays e **corrige o único teste
que falha** na suite `tests-long.sh`.

**Achados relacionados**: 1.

**Escopo detalhado**:

1. **Generalizar o motor dinâmico para N layer arrays**:

   - Remover a guarda `num_arrays != 2` em `dynamic.rs:113-118`.
   - O modelo `WaveNetModelDyn` já tem `layer_arrays: Vec<...>` — o loop precisa
     ser generalizado de 2-array hardcoded para N-array.
   - O `head_bias` e `head_size` precisam ser extraídos do último array (como o
     C++ faz: `layers[N-1].head`).

2. **Testes**:

   - O teste `live_cross_validation_nondist_models` que hoje falha deve passar.
   - Adicionar testes unitários com 3 e 4 arrays (geometrias sintéticas).
   - Golden C++ com modelo nondist de 4 arrays.

**Resumo da Implementação e Resultados**:

1. **Topologia Genérica de Arrays**: O `WaveNetModelDyn` foi atualizado para conter um vetor `arrays: Vec<WaveNetLayerArrayDyn>` em vez de instâncias fixas `array1` e `array2`. O dispatcher itera de forma agnóstica repassando saídas e entradas pela corrente.
2. **Correção de Rejeição e Suporte ao C++**: A restrição no carregamento de JSON que obrigava a presença de apenas 2 sub-arrays foi removida e generalizada, o que possibilitou o teste de cross validation longo em modelos _nondist_ carregar com sucesso os pesos C++ parity.

---

### MT4 — ✅ LSTM Arbitrário (F7) Concluído (2026-06-19)

**Pré-requisitos**: nenhum (ortogonal).
**Desafio**: 🟡 Médio — mesma filosofia do dispatch híbrido de F1.
**Benefício**: 🟠 Médio-Alto — aceita qualquer modelo LSTM custom.

**Escopo detalhado**:

1. **Caminho LSTM dinâmico**:

   - Implementar kernel LSTM com dimensões no load (não const-generic).
   - Dispatch: se `(num_layers, hidden_size)` casa com os 10 perfis estáticos, usar
     fast-path const-generic; senão, cair no motor dinâmico.
   - Manter `StaticModel::Lstm*` enum para as 10 variantes otimizadas.

2. **Testes**:

   - Testar com `(1,32)`, `(2,16)`, `(3,8)` e outras geometrias incomuns.
   - Golden C++ com LSTM de geometria não-padrão.

**Resumo da Implementação e Resultados**:

1. **Estruturas Dinâmicas LSTM**: `LstmLayerDyn` e `LstmModelDyn` implementados em `src/models/lstm/` operando via vetores (`AlignedVec`), com integração segura (RT-safe heap allocation apenas em load time) das funções super dimensionais de `crate::math::gemm` já vetorizadas (AVX2, AVX-512 e fallback escalar).
2. **Dispatcher Híbrido**: O motor de despacho em `src/loader/dispatcher/lstm/dispatch.rs` foi redirecionado; ao invés de abortar no `match` de _const generics_, ele preenche via fallback os pesos alinhados pelo JSON na via `LstmDyn` de `StaticModel`. A compatibilidade cobre a totalidade do spec atual de modelos LSTM, validados pelo validador.

---

### MT5 — 🟡 Post-stack Head (F6) + SlimmableWavenet (F5) + Container (F11)

**Pré-requisitos**: MT1 (FiLM) e MT3 (N arrays) para F5.
**Desafio**: 🟠 Alto (F5), 🟡 Médio (F6), 🟢 Baixo (F11).
**Benefício**: 🟡 Médio — atende modelos custom e slimmable.

**Escopo detalhado**:

1. **Post-stack Head (F6)**:

   - Implementar `Head` (Conv1D em cadeia + ativação) conforme `convnet.h:108-118`.
   - Somar ao `receptive_field` de prewarm.
   - Remover guarda `"WaveNet 'head' (post-stack sub-object) is not supported (F6)"`
     em `topology.rs:601-607`.

2. **SlimmableWavenet (F5)**:

   - Implementar slicing por extração de pesos no load/stage assíncrono.
   - Swap atômico via SPSC GC (padrão existente).
   - Referência: `NAM/wavenet/slimmable.h`.
   - Integrar ao `adaptive.rs`.

3. **Container aninhado (F11)**:

   - `src/models/container.rs` existe mas rejeita aninhamento.
   - Depende de F5 para golden com modelo slimmable real.

---

### MT6 — 🟢 ConvNet + Multi-canal (F4 + F10)

**Pré-requisitos**: F8 (ativações), F9 (groups).
**Desafio**: 🟡 Médio.
**Benefício**: 🟢 Baixo — nicho/legado.
**Decisão**: fora de escopo até demanda real. Documentado para completude.

---

### MT7 — 🟡 Otimizações Imediatas (fazer agora) ✅ Concluído (2026-06-19)

**Pré-requisitos**: nenhum.
**Desafio**: 🟢 Baixo-Médio.
**Benefício**: 🟡 Médio — melhoria de desempenho mensurável sem risco funcional.

**Achados relacionados**: 2, 3.

**Resumo da Implementação e Resultados**:

1. **Vetorização do A2 Head Conv (Achado 2)**:
   - **CH=8**: Implementado kernel `head_process_ch8_avx2` usando `_mm256_loadu_ps` + `_mm256_fmadd_ps` em loop unrolled com frame-tiling $T=4$ e redução horizontal via `hsum_avx2`.
   - **CH=3**: Implementado kernel `head_process_ch3_sse` usando empacotamento com zero-padding explícito via `_mm_setr_ps` (Opção A) e acumulador 128 bits.
   - **Performance**:
     - CH=8 AVX2 obteve **~14.15×** de aceleração (~101 ns vs ~1430 ns da referência escalar).
     - CH=3 SSE obteve **~2.19×** de aceleração (~251 ns vs ~550 ns da referência escalar).

2. **Vetorização de Ativações Inline (Achado 3)**:
   - Implementados kernels SIMD otimizados para:
     - `HardTanh`: `_mm256_min_ps` e `_mm256_max_ps`.
     - `HardSwish`: branchless clamp + multiplication.
     - `LeakyHardTanh`: branchless blend.
     - `FastTanh`: vetorização da aproximação de Padé via `_mm256_div_ps` para máxima precisão f32.
   - As 4 ativações foram extraídas para módulos individuais em `src/math/activations/` (`hard_tanh.rs`, `hard_swish.rs`, `leaky_hard_tanh.rs`, `fast_tanh.rs`) e expostas globalmente através da tabela de dispatch `SIMD_MATH` (`detect_best_simd`), unificando o design com as demais ativações.

3. **Auditoria e Eliminação de Transcendentes no Hot-path**:
   - Auditados os fallbacks escalares de `silu.rs`, `sigmoid.rs`, `tanh/production.rs`, `tanh/high_fidelity.rs` e `fused.rs`.
   - Substituídas todas as chamadas a `f32::exp()` e `f32::tanh()` no código de produção por aproximações minimax racionais real-time safe (ex: `scalar_minimax_sigmoid` e `scalar_pade_tanh`).

4. **Verificação de Integridade**:
   - 505 testes unitários e de integração passando sem falhas.
   - Validador oficial `clap-validator` com 100% de conformidade.
   - Checksums SHA256 do binário de build idênticos nas fases 2 e 4.
   - Heap-audit executado confirmando zero alocações no hot-path de processamento do WaveNet A2.

---

## Nota de Método

- Respeitar dependências: MT1 → MT2 → MT5(F5). MT3 e MT4 são ortogonais.
- MT7 pode ser executado a qualquer momento (sem dependências).
- RT-safety: alocação só no load, hot-path zero-alloc/lock/panic.
- Regra de golden: "todo golden deve poder falhar".
- Ao implementar MT2: vetorizar head conv (MT7.1) como critério de aceite.
- Ao implementar MT4: mesma filosofia de dispatch híbrido de F1.
- **x86-64-v3 obrigatório**: todo novo código SIMD deve usar AVX2+FMA como baseline.
