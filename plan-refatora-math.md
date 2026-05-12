# Plano: Reorganização de `src/math/`

## Diagnóstico da Situação Atual

### Estrutura atual

```text
src/math/
├── mod.rs               # 10 linhas: pub mod fastmath; pub mod simd;
├── fastmath.rs           # 1228 linhas — MONÓLITO
│                         #   GainLUT + tanh + sigmoid + relu + softsign + silu + prelu
│                         #   (cada uma em AVX2 + AVX-512, + dual + slice + fused)
├── fastmath_test.rs      # 719 linhas de testes de sweep
└── simd/
    ├── mod.rs            # dispatch_simd! macro, re-exports
    ├── dispatch.rs       # SimdMathConfig, detect_best_simd, v-table
    ├── traits.rs         # SimdMath trait (33 métodos)
    ├── avx2.rs           # ~2092 linhas: Avx2Math impl + funções standalone
    ├── avx512.rs         # ~2266 linhas: Avx512Math, Avx512VnniMath, Avx512VnniBf16Math
    ├── scalar_ref.rs     # 583 linhas: implementações escalares de referência
    ├── ops.rs            # Prefetch, DAZ/FTZ, f32_to_bf16, compute_energy, max_diff
    ├── aligned.rs        # AlignedVec<T>
    ├── utility.rs        # hsum_avx2, hsum_avx512
    ├── simd_test.rs      # 178 linhas
    ├── avx2_test.rs      # 49 linhas
    └── avx512_test.rs    # 26 linhas
```

**Total**: ~7.661 linhas de código matemático + 374 linhas em `models/activations.rs`.

### Problemas identificados

1. **`fastmath.rs` é um monólito de 1228 linhas** misturando GainLUT, 6 funções de ativação (AVX2+AVX-512), operações fundidas e processamento de slices.

2. **Separação `simd/` perdeu o sentido**: Com x86-64-v3 mandatório, **tudo é SIMD**. A subpasta é apenas aninhamento sem significado.

3. **Duplicação de dispatch**: `models/activations.rs` reimplementa dispatch manual (`match SIMD_MATH.instruction_set`) que duplica o `dispatch_simd!`.

4. **Referências cruzadas confusas**: `avx2.rs`/`avx512.rs` chamam funções em `fastmath.rs` e vice-versa.

5. **`Avx2VnniMath` é 100% delegação**: Confirmado — todos os 33 métodos delegam para `Avx2Math::method(...)`. AVX2-VNNI não traz benefício para operações FP. Deve ser eliminado.

6. **Dual dispatch**: Coexistem trait genérica (`SimdMath`) e v-table (`SimdMathConfig`) — dois mecanismos para o mesmo propósito.

7. **`ops.rs` é heterogêneo**: Mistura utilitários de CPU (DAZ/FTZ, prefetch) com operações DSP (energia, max_diff).

8. **Aliases desnecessários**: `simd_tanh()` e `simd_sigmoid()` em `fastmath.rs` L1214-1224 são apenas wrappers para as versões AVX2.

### O que cada modelo de inferência usa

| Categoria            | LSTM                                          | WaveNet                                                      | DSP/Resampler                               |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------- |
| **Dot Products**     | dot_product, 4x_interleaved, bf16, dual_frame | 4x_interleaved, dual_frame                                   | -                                           |
| **GEMV/GEMM**        | gemv_overwrite, gemv_4gate, fused_add_gemv    | gemv_overwrite, fused_add_gemv, gemm_batch, gemm_residual    | -                                           |
| **Ativações**        | tanh, sigmoid                                 | tanh, sigmoid, relu, softsign, silu, prelu                   | -                                           |
| **Fused LSTM Gates** | fused_lstm_gates                              | -                                                            | -                                           |
| **WaveNet Head**     | -                                             | head_sum, accumulate_head, tanh_accumulate, gated_accumulate | -                                           |
| **Conv/Stereo**      | -                                             | convolve_stereo                                              | convolve_stereo                             |
| **Gain/Energy**      | -                                             | apply_gain, compute_energy_stereo                            | apply_gain, compute_energy_stereo, max_diff |
| **Common**           | AlignedVec, hsum, f32_to_bf16, DAZ/FTZ        | AlignedVec, hsum, f32_to_bf16                                | AlignedVec, DAZ/FTZ                         |

---

## Decisões de Design Consolidadas

| Decisão                            | Resolução                                                                    |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| Trait `SimdMath`                   | Manter monolítica (33 métodos). Decomposição em sub-traits é trabalho futuro |
| `scalar_ref.rs`                    | Fica em `common/` — oráculo centralizado                                     |
| `Avx2VnniMath`                     | **Eliminar** — substituir por `type Avx2VnniMath = Avx2Math`                 |
| `Avx512VnniMath`                   | **Mantém** — tem `dot_product_bf16_avx512` nativo (real)                     |
| Dual dispatch                      | Documentar como design debt; unificar a longo prazo                          |
| `simd_tanh`/`simd_sigmoid` aliases | Internalizar em `activations/`                                               |
| `gemv_4gate`                       | Vai para `gemm/`, não `lstm/` (evita dep. circular `common → lstm`)          |
| `compute_energy_*`, `max_diff_*`   | Saem de `ops.rs` → vão para `dsp/stereo.rs`                                  |
| `InstructionSet::Avx2Vnni`         | Manter no enum por ora (remoção seria breaking change separada)              |

---

## Regra de Ouro: Preservação de Comentários

> **Todo código movido DEVE levar consigo seus comentários, docstrings e anotações `///`.**
> A base de código possui documentação didática extensa e cuidadosamente elaborada.
> Qualquer perda de comentário durante a migração é considerada um **defeito**.

### Diretrizes

1. **Mover, não reescrever**: Copiar bloco inteiro (docstring + corpo + testes) sem editar conteúdo técnico
2. **Adaptar caminhos nas docstrings**: Referências a `fastmath.rs` → novo caminho
3. **Docstrings de módulo**: Cada `mod.rs` novo deve ter `//!` explicando propósito e consumidores
4. **Verificação diff**: Ao final de cada Sprint, confirmar que o total de comentários não diminuiu

---

## Estrutura-Alvo Final

```text
src/math/
├── mod.rs                          # Re-exports estáveis (macro, traits, types)
├── constants.rs                    # Coeficientes Minimax, clamp limits, LUT params
│
├── common/                         # ═══ FUNDAÇÃO ═══
│   ├── mod.rs
│   ├── traits.rs                   # SimdMath trait (33 métodos)
│   ├── dispatch.rs                 # InstructionSet, SimdMathConfig, SIMD_MATH, dispatch_simd!
│   ├── avx2_impl.rs               # Avx2Math + (type Avx2VnniMath = Avx2Math)
│   ├── avx512_impl.rs             # Avx512Math, Avx512VnniMath, Avx512VnniBf16Math
│   ├── aligned.rs                  # AlignedVec<T>
│   ├── utility.rs                  # hsum_avx2, hsum_avx512, horizontal_sum_avx512
│   ├── scalar_ref.rs               # Oráculo escalar centralizado
│   ├── ops.rs                      # f32_to_bf16, set_daz_ftz, PrefetchFn, prefetch_*
│   └── tests.rs                    # (SEM compute_energy/max_diff — vão para dsp/)
│
├── activations/                    # ═══ ATIVAÇÕES NÃO-LINEARES ═══
│   ├── mod.rs                      # Re-exports + dispatch unificado (tanh_slice, etc.)
│   ├── tanh.rs                     # AVX2 + AVX-512 lado a lado
│   ├── sigmoid.rs
│   ├── relu.rs
│   ├── prelu.rs
│   ├── softsign.rs
│   ├── silu.rs
│   ├── fused.rs                    # tanh_sigmoid_dual (ILP interleaved)
│   └── tests.rs
│
├── gemm/                           # ═══ ÁLGEBRA LINEAR ═══
│   ├── mod.rs
│   ├── dot.rs
│   ├── dot_4x.rs
│   ├── gemv.rs
│   ├── gemm_batch.rs
│   ├── gemv_bf16.rs
│   ├── gemv_4gate.rs              # 4-gate LSTM projection (aqui, não em lstm/)
│   └── tests.rs
│
├── lstm/                           # ═══ LSTM-EXCLUSIVO ═══
│   ├── mod.rs
│   ├── gates.rs                    # fused_lstm_gates (ativações fundidas)
│   └── tests.rs
│
├── wavenet/                        # ═══ WAVENET-EXCLUSIVO ═══
│   ├── mod.rs
│   ├── head.rs
│   ├── accumulate.rs
│   └── tests.rs
│
└── dsp/                            # ═══ OPERAÇÕES DSP ═══
    ├── mod.rs
    ├── gain_lut.rs
    ├── gain.rs
    ├── stereo.rs                   # + compute_energy_*, compute_max_diff_*
    └── tests.rs
```

### Alinhamento com a Trait `SimdMath`

Os `impl SimdMath` delegam para as funções nos módulos corretos:

```rust
// src/math/common/avx2_impl.rs
impl SimdMath for Avx2Math {
    fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::activations::tanh::tanh_slice_avx2(slice) }
    }
    fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { crate::math::gemm::dot::dot_product_avx2(a, b) }
    }
    // ...
}

/// AVX2-VNNI não traz benefício para operações FP (apenas inteiras).
pub type Avx2VnniMath = Avx2Math;
```

### Dispatch Unificado de Ativações

`activations/mod.rs` expõe funções com dispatch interno, eliminando a duplicação em `models/activations.rs`:

```rust
// src/math/activations/mod.rs
pub fn tanh_slice(data: &mut [f32]) {
    match SIMD_MATH.instruction_set {
        Avx512 | Avx512Vnni | Avx512VnniBf16 => unsafe { tanh::tanh_slice_avx512(data) },
        _ => unsafe { tanh::tanh_slice_avx2(data) },
    }
}
```

```rust
// src/models/activations.rs — DEPOIS (simplificado)
Self::Tanh => crate::math::activations::tanh_slice(data),
```

### O que NÃO muda

- A trait `SimdMath` permanece com 33 métodos (mesma interface)
- O macro `dispatch_simd!` permanece funcional
- `detect_best_simd()` permanece idêntico
- `AlignedVec<T>` permanece intacto
- Algoritmos e constantes polinomiais são preservados bit-identical

### Impacto nos Arquivos Consumidores

| Arquivo                        | Mudança principal                                          |
| ------------------------------ | ---------------------------------------------------------- |
| `models/lstm.rs`               | `crate::math::activations::*`, `crate::math::gemm::*`      |
| `models/wavenet.rs`            | `crate::math::common::*`, `gemm::*`, `wavenet::*`          |
| `models/activations.rs`        | Substituir dispatch manual por `activations::tanh_slice()` |
| `dsp/pipeline.rs`              | `crate::math::dsp::stereo::*`                              |
| `dsp/gate.rs`                  | `crate::math::common::SimdMath`                            |
| `dsp/resampler.rs`             | `crate::math::common::{AlignedVec, dispatch_simd}`         |
| `standalone/rt_setup.rs`       | `crate::math::common::set_daz_ftz`                         |
| `standalone/pw_host.rs`        | `crate::math::dsp::get_gain_lut`                           |
| `standalone/cli.rs`            | `crate::math::constants::*`                                |
| `loader/dispatcher/lstm.rs`    | `crate::math::common::f32_to_bf16`                         |
| `loader/dispatcher/wavenet.rs` | `crate::math::common::{AlignedVec, f32_to_bf16}`           |

---

## Riscos e Mitigações

| Risco                                                                | Severidade | Mitigação                                                             |
| -------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------- |
| Macro `dispatch_simd!` com caminhos hardcoded `$crate::math::simd::` | Alta       | Re-exports em `math/mod.rs` raiz como ponte transitória               |
| Quebra de imports em 48+ callsites                                   | Alta       | Migração gradual com re-exports; atualizar consumidores só no Épico 4 |
| Regressão de performance                                             | Alta       | Baseline capturado no Sprint 1.1; `cargo bench` a cada Sprint         |
| Perda de comentários/docstrings                                      | Alta       | Regra de Ouro; verificação diff a cada Sprint                         |
| `#![allow(unsafe_op_in_unsafe_fn)]` é file-level em `fastmath.rs`    | Média      | Ao desmembrar, aplicar `#[allow(...)]` seletivamente                  |
| `Avx2VnniMath` eliminado mas `InstructionSet::Avx2Vnni` mantido      | Baixa      | Enum preservado; dispatch aponta para `Avx2Math`                      |
| A2 ainda é placeholder                                               | Info       | Deixar espaço para `src/math/a2/` quando necessário                   |

---

## Plano de Execução — Épicos e Sprints

### Épico 1: Baseline e Fundação

**Objetivo**: Capturar métricas de referência e criar `common/` sem alterar comportamento.

#### Sprint 1.1 — Baseline de Performance e Testes

Capturar snapshot completo antes de qualquer alteração:

- `cargo bench -- --save-baseline pre-refactor`
- `cargo test --release`
- Documentar contagem de linhas (código + comentários) por arquivo

**Gate de saída**: Baseline salvo; zero falhas em testes.

#### Sprint 1.2 — `constants.rs`

Extrair constantes compartilhadas de `fastmath.rs` para `src/math/constants.rs`:

- Clamp limits (TANH/SIGMOID)
- Coeficientes Minimax/Padé
- Parâmetros de LUT (GAIN_LUT_SIZE, GAIN_MIN_DB, GAIN_MAX_DB)

**Gate de saída**: `cargo check` + `cargo test` passam. Constantes usadas por ≥2 arquivos centralizadas.

#### Sprint 1.3 — `common/` Foundation

Mover módulos de infraestrutura de `simd/` para `common/`:

- `traits.rs`, `dispatch.rs`, `aligned.rs`, `utility.rs` (+ `hsum_avx512`), `scalar_ref.rs`
- `ops.rs` (sem `compute_energy_*` e `compute_max_diff_*` — esses vão para `dsp/` depois)
- Criar `common/mod.rs` com re-exports

**Regra crítica**: `src/math/mod.rs` mantém re-exports transitórios para preservar caminhos antigos (`crate::math::simd::*`). O macro `dispatch_simd!` é atualizado internamente para `$crate::math::common::`.

**Gate de saída**: `cargo check` + `cargo test` passam. Imports antigos ainda funcionam via re-exports.

---

### Épico 2: Estrutura de Trait e Implementações

**Objetivo**: Estabilizar structs de implementação em `common/` antes de mover kernels.

#### Sprint 2.1 — `avx2_impl.rs` e `avx512_impl.rs`

Extrair structs e `impl SimdMath for ...` de `avx2.rs`/`avx512.rs` para `common/`. As implementações continuam chamando kernels no local antigo via caminhos absolutos.

**Simplificação AVX2-VNNI**: Substituir ~300 linhas de delegação por:

```rust
pub type Avx2VnniMath = Avx2Math;
```

**Gate de saída**: `cargo check` + `cargo test` passam. `avx2.rs`/`avx512.rs` sem blocos `impl SimdMath`.

#### Sprint 2.2 — Documentar Design Debt do Dual Dispatch

Adicionar documentação em `common/dispatch.rs` explicando:

- Por que coexistem trait genérica e v-table `SimdMathConfig`
- Quais consumidores usam qual mecanismo
- Plano futuro de unificação

**Gate de saída**: `cargo clippy` limpo; comentários de design debt presentes.

---

### Épico 3: Migração de Kernels por Domínio

**Objetivo**: Mover funções standalone para subpastas de domínio, uma categoria por Sprint.

> **Em cada Sprint deste Épico**: preservar integralmente todos os comentários, docstrings e anotações `///` de cada função movida. Adaptar apenas referências de caminhos nas docstrings. Executar `cargo check` após cada arquivo extraído.

#### Sprint 3.1 — `activations/`

Desmembrar `fastmath.rs` (1229L) em arquivos por ativação:

| Destino       | Funções de `fastmath.rs`                                                                      |
| ------------- | --------------------------------------------------------------------------------------------- |
| `tanh.rs`     | `simd_tanh_avx2`, `simd_tanh_dual_avx2`, `simd_tanh_avx512`, `tanh_slice_*`, `tanh()` escalar |
| `sigmoid.rs`  | `simd_sigmoid_*`, `sigmoid_slice_*`, `sigmoid()` escalar                                      |
| `relu.rs`     | `simd_relu_*`, `relu_slice_*`                                                                 |
| `prelu.rs`    | `simd_prelu_*`, `prelu_slice_*`                                                               |
| `softsign.rs` | `simd_softsign_*`, `softsign_slice_*`                                                         |
| `silu.rs`     | `simd_silu_*`, `silu_slice_*`                                                                 |
| `fused.rs`    | `simd_tanh_sigmoid_dual_*`, `simd_sigmoid_dual_*`                                             |

**Internalização**: Aliases `simd_tanh()` e `simd_sigmoid()` absorvidos — chamadores usam path direto.

**Dispatch unificado**: `activations/mod.rs` expõe `tanh_slice(data)` com dispatch interno, eliminando duplicação manual em `models/activations.rs`.

**Testes**: `fastmath_test.rs` (719L) → `activations/tests.rs`.

**Gate de saída**: `fastmath.rs` deixa de existir. `cargo test` passa.

#### Sprint 3.2 — `gemm/`

Extrair kernels de álgebra linear de `avx2.rs`/`avx512.rs`:

| Destino         | Funções                                                 |
| --------------- | ------------------------------------------------------- |
| `dot.rs`        | `dot_product_avx2/avx512`                               |
| `dot_4x.rs`     | `dot_product_4x_*`, `dual_frame_*`, `batch_4x_*`        |
| `gemv.rs`       | `gemv_overwrite_*`, `fused_add_gemv_*` (incl. `_small`) |
| `gemm_batch.rs` | `fused_add_gemm_batch_*`, `fused_gemm_residual_batch_*` |
| `gemv_bf16.rs`  | `gemv_overwrite_bf16_*`                                 |
| `gemv_4gate.rs` | `gemv_4gate_avx2/avx512`, `gemv_4gate_bf16_*`           |

**Gate de saída**: `cargo bench` sem regressão.

#### Sprint 3.3 — `lstm/`

| Destino    | Funções                                                |
| ---------- | ------------------------------------------------------ |
| `gates.rs` | `fused_lstm_gates_avx2/avx512`, `fused_lstm_gates_dyn` |

**Gate de saída**: `cargo test` passa.

#### Sprint 3.4 — `wavenet/`

| Destino         | Funções                                                                                       |
| --------------- | --------------------------------------------------------------------------------------------- |
| `head.rs`       | `batch_wavenet_head_sum_avx2/avx512` (const generic + dyn)                                    |
| `accumulate.rs` | `accumulate_head_*`, `tanh_and_accumulate_block_*`, `gated_activation_and_accumulate_block_*` |

**Gate de saída**: `cargo test` passa.

#### Sprint 3.5 — `dsp/`

| Destino       | Funções                                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------------------- |
| `gain_lut.rs` | `GainLUT`, `GAIN_LUT`, `get_gain_lut()`                                                                    |
| `gain.rs`     | `apply_gain_*`, `apply_gain_and_detect_clipping_stereo_*`, `apply_ramp_stereo_*`                           |
| `stereo.rs`   | `compute_energy_avx2` (de `ops.rs`), `compute_energy_stereo`, `compute_max_diff_avx2`, `convolve_stereo_*` |

**Gate de saída**: `cargo test` + `cargo bench` passam.

---

### Épico 4: Limpeza e Unificação de Imports

**Objetivo**: Eliminar código morto, atualizar consumidores, remover re-exports transitórios.

#### Sprint 4.1 — Atualizar Consumidores

Atualizar todos os `use crate::math::` nos arquivos consumidores (ver tabela de impacto acima). Substituir dispatch manual em `models/activations.rs` por chamadas unificadas.

**Gate de saída**: `cargo check` sem warnings de imports.

#### Sprint 4.2 — Remoção de Código Morto

- Remover `src/math/simd/` (já vazio)
- Remover `src/math/fastmath.rs` (já desmembrado)
- Remover re-exports transitórios de `math/mod.rs`
- Confirmar eliminação do `impl SimdMath for Avx2VnniMath` (~300L)

**Gate de saída**: `cargo check` + `cargo clippy` limpos. Zero dead code.

---

### Épico 5: Validação Final e Documentação

**Objetivo**: Garantir paridade total com baseline e atualizar docs.

#### Sprint 5.1 — Validação de Paridade

- `cargo test` — todos os testes passam (incluindo sweeps de erro máximo)
- `cargo bench` — comparar com baseline (threshold: <2% regressão)
- `utils/lints.sh` — limpo
- Verificar contagem de comentários vs baseline (sem perda)

**Gate de saída**: Paridade total confirmada.

#### Sprint 5.2 — Documentação Arquitetural

- Atualizar `docs/architecture.md` seção de matemática
- Docstrings `//!` em cada `mod.rs` novo
- Registrar decisões (eliminação `Avx2VnniMath`, design debt dual dispatch)

**Gate de saída**: Documentação sincronizada com implementação.

---

## Resumo

| Épico                      | Sprints | Risco | Complexidade                       |
| -------------------------- | ------- | ----- | ---------------------------------- |
| **1. Baseline e Fundação** | 3       | Baixo | Moderada (macro paths)             |
| **2. Traits e Impls**      | 2       | Médio | Moderada (Avx2VnniMath)            |
| **3. Migração de Kernels** | 5       | Alto  | Alta (7.661L, preservação de docs) |
| **4. Limpeza de Imports**  | 2       | Médio | Moderada (48+ callsites)           |
| **5. Validação e Docs**    | 2       | Baixo | Baixa                              |
| **Total**                  | **14**  |       | ~7.661 linhas reorganizadas        |

> **Sprint de maior risco**: 3.1 (activations/) — desmembra `fastmath.rs` que é o ponto de convergência entre `avx2.rs`, `avx512.rs` e `models/activations.rs`. Recomendação: execução atômica com `cargo check` após cada arquivo extraído.
> **Invariante de qualidade**: Nenhum Sprint fecha sem `cargo check` + `cargo test` passando. Nenhum Sprint do Épico 3 fecha sem verificação explícita de preservação de comentários.
