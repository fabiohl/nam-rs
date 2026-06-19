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
| **F2**  | **Multi-condição / FiLM** (`condition_size > 1`, `condition_dsp`)       | 🔴 Crítico         | Alto    |
| **F3**  | **Motor A2 geral** (gating, ativações heterogêneas, `head1x1`, `bn≠ch`) | 🟠 Alto            | Alto    |
| **F5**  | **SlimmableWavenet** (slicing dinâmico de canais)                       | 🟠 Médio-Alto      | Alto    |
| **F6**  | **Post-stack Head** (sub-objeto `head` multi-camada do WaveNet)         | 🟡 Médio           | Médio   |
| **F7**  | **LSTM arbitrário** (`hidden_size`/`num_layers` fora dos 10 perfis)     | 🟠 Médio-Alto      | Médio   |
| **F8**  | **Biblioteca completa de ativações** (PReLU, SiLU, etc.)                | 🟡 Médio           | Médio   |
| **F9**  | **Convoluções agrupadas/depthwise** (`groups > 1`)                      | 🟡 Médio           | Médio   |
| **F4**  | **ConvNet** (arquitetura legada)                                        | 🟢 Baixo           | Médio   |
| **F10** | **Modelos multi-canal** (`in/out_channels > 1`)                         | 🟢 Baixo           | Médio   |
| **F11** | **Container aninhado** + cobertura `SlimmableContainer` real            | 🟢 Baixo           | Baixo   |

> **Dependências**: F2 ⊃ F3 ⊃ {F8, F9}. F7 é ortogonal. F5 depende de F2/F3 (FiLM/gating + SlimmableWaveNet). F6 é ortogonal mas é usado pelo WaveNet geral.

---

## Diagnóstico Verificado (probe de carga, jun/2026)

| Modelo oficial                | Resultado no nam-rs                     | Feature que destrava |
| ----------------------------- | --------------------------------------- | -------------------- |
| `wavenet_a1_standard.nam`     | ✅ Carrega (golden oficial)             | —                    |
| `lstm.nam`                    | ✅ Carrega (golden oficial)             | —                    |
| `wavenet.nam` (CH=3, livre)   | ✅ Carrega (motor dinâmico)             | —                    |
| `slimmable_wavenet.nam`       | ❌ "A2 shape not recognized"            | **F5**               |
| `wavenet_a2_max.nam` (cond=8) | ❌ "only condition_size=1 is supported" | **F2 / F3**          |
| `wavenet_condition_dsp.nam`   | ❌ "only condition_size=1 is supported" | **F2 / F3**          |
| `slimmable_container.nam`     | ❌ "submodel build failed"              | **F5 / F11**         |
| Modelos nondist 4-array       | ❌ "requires exactly 2 layer arrays"    | **F1-ext**           |

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

### MT1 — 🔴 Infraestrutura de Conditioning e FiLM (F2 + F8 parcial + F9 parcial)

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

---

### MT2 — 🟠 Motor A2 Geral (F3 + F8 + F9)

**Pré-requisitos**: MT1 (FiLM/conditioning).
**Desafio**: 🟠 Alto — generalização de um motor const-generic para dinâmico.
**Benefício**: 🟠 Alto — suporta qualquer modelo A2 (futuro-proof).

**Achados relacionados**: 2, 3, 4, 6.

**Escopo detalhado**:

1. **Motor A2 dinâmico**:

   - Alocação de buffers no load (como o WaveNet dinâmico existente).
   - Downcast para fast-path const-generic quando a geometria casa com CH=3 ou CH=8
     e sem gating/FiLM (preservar desempenho atual).
   - Suporte a `bottleneck ≠ channels`.
   - Suporte a `head1x1` e `layer1x1` configuráveis.

2. **GatingActivation e BlendingActivation**:

   - Implementar `src/models/a2/gating.rs` com lógica real.
   - Paridade com `NAM/gating_activations.h` (251 linhas C++).
   - Buffers pré-alocados, zero-alloc no hot-path.
   - Vetorizar com AVX2+FMA nativamente.

3. **Biblioteca completa de ativações (F8)**:

   - Portar `ActivationConfig` com parsing de string/objeto JSON.
   - Ativações heterogêneas por camada (array de `ActivationType`).
   - Toggle global de fast-tanh (já existe como conceito).
   - Todas as ativações existem em `src/models/a2/activations.rs` mas precisam
     de parsing JSON e wiring com o motor dinâmico.

4. **Conv1D com `groups > 1` completa (F9)**:

   - Generalizar `src/models/a2/conv1d.rs` para grupos.
   - Manter fast-path `groups==1` intacto e SIMD-otimizado.
   - Kernel depthwise (groups == channels) como caso especial otimizado.

5. **Vetorizar A2 Head Conv (Achado 2)**:

   - CH=8: `_mm256_loadu_ps` + `_mm256_fmadd_ps` — padrão já usado em `conv1d_ch8.rs`.
   - CH=3: SSE4.1 ou auto-vetorização com padding (tratar como CH=4 com mask).
   - Manter `a2_head_block_scalar_ref` como oracle de parity.
   - **Fazer isso já na etapa atual** (otimização evidente e imediata).

6. **Testes**:

   - Golden C++ para modelos com gating, blending, ativações heterogêneas.
   - Parity ESR/SNR para cada combinação de ativação vs C++.
   - Heap-audit para motor A2 dinâmico.

---

### MT3 — 🟠 WaveNet A1 Dinâmico: Generalização para N Arrays (F1-ext)

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

---

### MT4 — 🟠 LSTM Arbitrário (F7)

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

### MT7 — 🟡 Otimizações Imediatas (fazer agora)

**Pré-requisitos**: nenhum.
**Desafio**: 🟢 Baixo-Médio.
**Benefício**: 🟡 Médio — melhoria de desempenho mensurável sem risco funcional.

**Achados relacionados**: 2, 3.

**Escopo detalhado**:

1. **Vetorizar A2 Head Conv (Achado 2)**:

   - CH=8: `_mm256_loadu_ps` + `_mm256_fmadd_ps` (8-wide SIMD).
   - CH=3: SSE ou auto-vec com padding.
   - Critério de aceite: parity bitwise com `a2_head_block_scalar_ref`.
   - Benchmarkar com `criterion` antes/depois.

2. **Vetorizar ativações escalares (Achado 3)**:

   - `HardTanh`: `vminps(vmaxps(x, -1), 1)` — 2 instruções SIMD.
   - `FastTanh`: SIMD rational polynomial (8-wide).
   - `HardSwish`: `x * vminps(vmaxps(x+3, 0), 6) * (1/6)` — branchless SIMD.
   - `LeakyHardTanh`: branchless SIMD com masks (`vcmpps` + `vblendvps`).
   - Usar padrão `chunks_exact(8)` + zip para auto-vetorização.

3. **Auditar `src/math/activations/`**:

   - Confirmar que `tanh_slice`, `relu_slice`, `prelu_slice`, etc. usam padrões
     auto-vetorizáveis (chunks_exact, sem branches).
   - Se não usarem, refatorar.

---

## Nota de Método

- Respeitar dependências: MT1 → MT2 → MT5(F5). MT3 e MT4 são ortogonais.
- MT7 pode ser executado a qualquer momento (sem dependências).
- RT-safety: alocação só no load, hot-path zero-alloc/lock/panic.
- Regra de golden: "todo golden deve poder falhar".
- Ao implementar MT2: vetorizar head conv (MT7.1) como critério de aceite.
- Ao implementar MT4: mesma filosofia de dispatch híbrido de F1.
- **x86-64-v3 obrigatório**: todo novo código SIMD deve usar AVX2+FMA como baseline.
