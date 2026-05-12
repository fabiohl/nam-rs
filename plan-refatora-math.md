# Plano: Reorganização de `src/math/`

> **Prompt:**
Gostaria de uma revisão da forma como estão distribuidos as funções matemáticas na pasta "src/math/".
Inicialmente a idéia era que o geral fique em "src/math/" e código otimizados para SIMD (AVX2 e AVX-512) sejam acionados sob demanda em "src/math/simd/".
Mas hoje parece-me que ficou tudo bem misturado.
Ainda mais considerando que é mandatório o uso de, no mínimo, x86-64-v3 (AVX2 e FMA) o absolutamente máximo possível. Se possível, multiversioning para (AVX-512) para ainda mais otimizações onde faz muito sentido para processadores que o suporte.
Então SIMD está em todo lugar, na real!
Acredito que pode ser mais lógico organizar funções (quebradas em arquviso com suas respectivas versões x86-64-v3 e x86-64-v4) - porém em subpastas logicamente organizadas por quem as utilizada, ficando a raiz "src/math/" com o absoluto comum a todos.
Continuaremos a ter uma pasta "math", mas orgnizações lógicas internas. Até pra facilitar o que faz parte ou não dos diversos modelos de inferência: LSTM, WaveNet, Arquiteturas A1 e A2 e o que mais tiver no código.
Estude a situação e me proponha uma nova organização.

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
    ├── avx2.rs           # ~2000+ linhas: Avx2Math impl + funções standalone
    ├── avx512.rs         # ~1700+ linhas: Avx512Math, Avx512VnniMath, Avx512VnniBf16Math
    ├── scalar_ref.rs     # 583 linhas: implementações escalares de referência
    ├── ops.rs            # Prefetch, DAZ/FTZ, f32_to_bf16, compute_energy, max_diff
    ├── aligned.rs        # AlignedVec<T>
    ├── utility.rs        # hsum_avx2, hsum_avx512
    ├── simd_test.rs      # 178 linhas
    ├── avx2_test.rs      # 49 linhas
    └── avx512_test.rs    # 26 linhas
```

### Problemas identificados

1. **`fastmath.rs` é um monólito de 1228 linhas** misturando:

   - GainLUT (utilitário DSP de ganho dB→linear)
   - Funções de ativação: tanh, sigmoid, relu, softsign, silu, prelu
   - Operações fundidas: tanh+sigmoid dual, fused_lstm_gates
   - Processamento de slices (in-place)
   - Todas com variantes AVX2 e AVX-512 no mesmo arquivo

2. **Separação `simd/` perdeu o sentido**: A idéia original era "geral em root, SIMD em simd/" — mas com x86-64-v3 (AVX2+FMA) mandatório, **tudo é SIMD**. A subpasta `simd/` é apenas um nível extra de aninhamento sem significado.

3. **Duplicação de dispatch**: `src/models/activations.rs` (370 linhas) implementa um dispatch manual via `match SIMD_MATH.instruction_set` que duplica a lógica que já existe no `dispatch_simd!` macro. As ativações deveriam ser despachadas pelo mesmo mecanismo que o resto.

4. **Referências cruzadas confusas**: `simd/avx2.rs` e `simd/avx512.rs` chamam `crate::math::fastmath::tanh_slice_avx2` — as funções de ativação vivem em `fastmath.rs`, mas são chamadas de dentro da implementação da trait `SimdMath`. Não há razão para essa separação.

5. **Testes fragmentados**: 4 arquivos de teste (`fastmath_test.rs`, `simd_test.rs`, `avx2_test.rs`, `avx512_test.rs`) com organização inconsistente.

### O que cada modelo de inferência usa

| Categoria            | LSTM                                                   | WaveNet                                                                                                   | DSP/Resampler                               |
| -------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| **Dot Products**     | dot_product, 4x_interleaved, bf16, dual_frame          | 4x_interleaved, dual_frame                                                                                | -                                           |
| **GEMV/GEMM**        | gemv_overwrite, gemv_4gate, fused_add_gemv, gemm_batch | gemv_overwrite, fused_add_gemv, gemm_batch, fused_gemm_residual_batch                                     | -                                           |
| **Activações**       | tanh, sigmoid                                          | tanh, sigmoid, relu, softsign, silu, prelu                                                                | -                                           |
| **Fused LSTM Gates** | fused_lstm_gates                                       | -                                                                                                         | -                                           |
| **WaveNet Head**     | -                                                      | batch_wavenet_head_sum, accumulate_head, tanh_and_accumulate_block, gated_activation_and_accumulate_block | -                                           |
| **Conv/Stereo**      | -                                                      | convolve_stereo                                                                                           | convolve_stereo                             |
| **Gain/Energy**      | -                                                      | apply_gain, compute_energy_stereo                                                                         | apply_gain, compute_energy_stereo, max_diff |
| **Prefetch**         | -                                                      | prefetch_strategies                                                                                       | -                                           |
| **GainLUT**          | -                                                      | sim (via DSP pipeline)                                                                                    | sim                                         |
| **Common**           | AlignedVec, hsum, f32_to_bf16, DAZ/FTZ                 | AlignedVec, hsum, f32_to_bf16                                                                             | AlignedVec, DAZ/FTZ                         |

### Baseline e Multiversioning

- **Baseline mandatório**: `-Ctarget-cpu=x86-64-v3` (AVX2 + FMA + BMI2 + F16C)
- **Multiversioning**: Em CPUs com suporte, as variantes AVX-512 (F, VL, VNNI, BF16) são selecionadas em runtime via `detect_best_simd()` → `dispatch_simd!` macro → trait `SimdMath`
- Constantes do polinômio (coeficientes Minimax/Padé) são compartilhadas entre AVX2 e AVX-512

---

## Proposta de Nova Organização

### Princípios

1. **Raiz `src/math/`**: apenas o que é absolutamente comum a todos (mod.rs, traits, dispatch, aligned, utility)
2. **Subpastas por categoria funcional**: organizadas pelo domínio do problema, não por ISA (já que SIMD é ubíquo)
3. **Cada arquivo = uma função matemática + suas variantes**: AVX2 (`__m256`) e AVX-512 (`__m512`) lado a lado, com constantes comuns no topo
4. **Subpastas específicas por modelo apenas onde não há sobreposição**: LSTM fused gates, WaveNet head/accumulate
5. **Constantes comuns extraídas**: coeficientes dos polinômios Minimax vivem em `constants.rs`
6. **Documentação de consumo**: cada módulo documenta quais modelos o utilizam

### Estrutura Proposta

```text
src/math/
│
├── mod.rs                          # pub mod common, activations, gemm, lstm, wavenet, dsp;
│                                   # Re-exports públicos: SimdMathConfig, SIMD_MATH, dispatch_simd!,
│                                   #   InstructionSet, AlignedVec
│
├── constants.rs                    # Constantes matemáticas compartilhadas
│                                   # TANH_CLAMP_LIMIT, SIGMOID_CLAMP_LIMIT,
│                                   # Coeficientes Minimax tanh (c0,c1,c2) e sigmoid (c2-c6),
│                                   # GAIN_LUT_SIZE, GAIN_MIN_DB, GAIN_MAX_DB
│                                   # Usado por: activations/*, dsp/gain_lut
│
├── common/                         # ═══ FUNDAÇÃO ABSOLUTAMENTE COMUM ═══
│   │                               # Usado por: TODOS os modelos
│   ├── mod.rs                      # Re-exports: SimdMath, SimdMathConfig, InstructionSet,
│   │                               #   dispatch_simd!, AlignedVec, SIMD_MATH, set_daz_ftz,
│   │                               #   f32_to_bf16, PrefetchFn, hsum_avx2, hsum_avx512
│   ├── traits.rs                   # SimdMath trait (move from simd/traits.rs)
│   │                               # + ScalarMath (para testes)
│   ├── dispatch.rs                 # InstructionSet enum, SimdMathConfig struct,
│   │                               #   detect_best_simd(), SIMD_MATH static
│   │                               #   dispatch_simd! macro
│   ├── aligned.rs                  # AlignedVec<T> (64-byte aligned buffer)
│   ├── utility.rs                  # hsum_avx2, hsum_avx512
│   ├── scalar_ref.rs               # Implementações escalares de referência
│   │                               #   dot_product_fallback, gemv_overwrite_fallback,
│   │                               #   fused_add_gemv_fallback, accumulate_head_fallback,
│   │                               #   tanh_and_accumulate_block_fallback, etc.
│   ├── ops.rs                      # f32_to_bf16, set_daz_ftz, PrefetchFn,
│   │                               #   prefetch_strategy_simple, prefetch_strategy_2stage,
│   │                               #   adaptive_prefetch_f32, adaptive_prefetch_2stage_f32
│   │                               #   compute_energy_avx2, compute_max_diff_avx2
│   └── tests.rs                    # Testes da foundation (hsum, aligned, ops, DAZ/FTZ, scalar_ref parity)
│
├── activations/                    # ═══ FUNÇÕES DE ATIVAÇÃO NÃO-LINEAR ═══
│   │                               # Usado por: LSTM, WaveNet, A1, A2
│   ├── mod.rs                      # Re-exports: todas as funções de ativação
│   │                               # + funções de slice (tanh_slice, sigmoid_slice, etc.)
│   │                               # + dispatch helpers
│   ├── tanh.rs                     # tanh(x) escalar, simd_tanh_avx2, simd_tanh_dual_avx2,
│   │                               #   simd_tanh_avx512, tanh_slice_avx2, tanh_slice_avx512
│   ├── sigmoid.rs                  # sigmoid(x) escalar, simd_sigmoid_avx2,
│   │                               #   simd_sigmoid_dual_avx2, simd_sigmoid_avx512
│   │                               #   sigmoid_slice_avx2, sigmoid_slice_avx512
│   ├── relu.rs                     # simd_relu_avx2, simd_relu_dual_avx2, relu_slice_avx2,
│   │                               #   simd_relu_avx512, relu_slice_avx512
│   ├── prelu.rs                    # simd_prelu_avx2, prelu_slice_avx2,
│   │                               #   simd_prelu_avx512, prelu_slice_avx512
│   ├── softsign.rs                 # simd_softsign_avx2, simd_softsign_dual_avx2,
│   │                               #   softsign_slice_avx2, simd_softsign_avx512,
│   │                               #   softsign_slice_avx512
│   ├── silu.rs                     # simd_silu_avx2, simd_silu_dual_avx2,
│   │                               #   silu_slice_avx2, simd_silu_avx512, silu_slice_avx512
│   ├── fused.rs                    # Operações fundidas:
│   │                               #   simd_tanh_sigmoid_dual_avx2,
│   │                               #   simd_tanh_sigmoid_dual_avx512
│   └── tests.rs                    # Sweeps + unit tests (move from fastmath_test.rs)
│                                   # + sweep_avx512 condicionais
│
├── gemm/                           # ═══ ÁLGEBRA LINEAR (Dot Products, GEMV, GEMM) ═══
│   │                               # Usado por: LSTM, WaveNet
│   ├── mod.rs                      # Re-exports
│   ├── dot.rs                      # dot_product_avx2, dot_product_avx512,
│   │                               #   dot_product_bf16_fallback (AVX2 usa f16, AVX-512 pode usar VNNI)
│   ├── dot_4x.rs                   # dot_product_4x_avx2, dot_product_4x_interleaved_avx2,
│   │                               #   dot_product_4x_interleaved_dual_frame_avx2,
│   │                               #   dot_product_batch_4x_avx2 (+ variantes AVX-512)
│   ├── gemv.rs                     # gemv_overwrite_avx2, fused_add_gemv_avx2,
│   │                               #   gemv_overwrite_avx512, fused_add_gemv_avx512,
│   │                               #   gemv_overwrite_avx512_small, fused_add_gemv_avx512_small
│   ├── gemm_batch.rs               # fused_add_gemm_batch_avx2/avx512,
│   │                               #   fused_gemm_residual_batch_avx2/avx512
│   ├── gemv_bf16.rs                # gemv_overwrite_bf16_fallback, gemv_4gate_bf16_fallback,
│   │                               #   gemv_4gate_bf16_avx512 (VNNI)
│   └── tests.rs                    # Testes de paridade AVX2 vs AVX-512 vs scalar_ref
│
├── lstm/                           # ═══ KERNELS ESPECÍFICOS LSTM ═══
│   │                               # Usado por: LSTM (não usado por WaveNet)
│   ├── mod.rs
│   ├── gates.rs                    # fused_lstm_gates_avx2, fused_lstm_gates_avx512
│   ├── gemv_4gate.rs               # gemv_4gate_avx2, gemv_4gate_avx512
│   │                               #   (move de avx2.rs/avx512.rs — funções standalone)
│   └── tests.rs
│
├── wavenet/                        # ═══ KERNELS ESPECÍFICOS WAVENET ═══
│   │                               # Usado por: WaveNet, WavenetDyn (não usado por LSTM)
│   ├── mod.rs
│   ├── head.rs                     # batch_wavenet_head_sum_avx2/avx512 (const generic + dyn)
│   ├── accumulate.rs               # accumulate_head_avx2/avx512,
│   │                               #   tanh_and_accumulate_block_avx2/avx512,
│   │                               #   gated_activation_and_accumulate_block_avx2/avx512
│   └── tests.rs
│
└── dsp/                            # ═══ OPERAÇÕES DSP (Ganho, Energia, Stereo) ═══
    │                               # Usado por: DSP pipeline, WaveNet, standalone host
    ├── mod.rs
    ├── gain_lut.rs                 # GainLUT struct + get_gain_lut() + GAIN_LUT static
    │                               #   (extraído de fastmath.rs)
    ├── gain.rs                     # apply_gain_avx2, apply_gain_avx512,
    │                               #   apply_gain_and_detect_clipping_stereo_avx2/avx512,
    │                               #   apply_gain_stereo_avx2/avx512,
    │                               #   apply_ramp_stereo_avx2/avx512
    ├── stereo.rs                   # compute_energy_stereo_avx2/avx512,
    │                               #   convolve_stereo_avx2/avx512,
    │                               #   compute_energy_avx2 (standalone, move from ops.rs)
    └── tests.rs                    # Testes de gain, stereo, gain_lut
```

### Alinhamento com a Trait `SimdMath`

Após a reorganização, as implementações da trait `SimdMath` (`Avx2Math`, `Avx512Math`, `Avx512VnniMath`, `Avx512VnniBf16Math`) ficam assim:

- **Avx2Math**: Continua em `src/math/common/avx2_impl.rs` (novo arquivo)
- **Avx512Math + Avx512VnniMath + Avx512VnniBf16Math**: Continua em `src/math/common/avx512_impl.rs` (novo arquivo)

Os métodos delegam para as funções nos módulos corretos:

```rust
impl SimdMath for Avx2Math {
    fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::activations::tanh::tanh_slice_avx2(slice) }
    }
    fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { crate::math::gemm::dot::dot_product_avx2(a, b) }
    }
    // ...
}
```

### Impacto nos Arquivos Consumidores

| Arquivo                            | Mudança necessária                                                                          |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `src/models/lstm.rs`               | Ajustar imports: `crate::math::fastmath::simd_tanh` → `crate::math::activations::simd_tanh` |
| `src/models/wavenet.rs`            | Ajustar imports para `common`, `gemm`, `wavenet`, `dsp`                                     |
| `src/models/activations.rs`        | Substituir dispatch manual por chamadas via módulo `activations` reorganizado               |
| `src/dsp/pipeline.rs`              | `compute_energy_stereo` → `crate::math::dsp::stereo::compute_energy_stereo`                 |
| `src/dsp/gate.rs`                  | `SimdMath` → `crate::math::common::SimdMath`                                                |
| `src/dsp/resampler.rs`             | `AlignedVec`, `dispatch_simd` → `crate::math::common`                                       |
| `src/standalone/rt_setup.rs`       | `set_daz_ftz()` → `crate::math::common::set_daz_ftz`                                        |
| `src/standalone/pw_host.rs`        | `get_gain_lut()` → `crate::math::dsp::get_gain_lut`                                         |
| `src/standalone/cli.rs`            | `GAIN_MAX_DB/MIN_DB` → `crate::math::constants`                                             |
| `src/loader/dispatcher/lstm.rs`    | `f32_to_bf16` → `crate::math::common::f32_to_bf16`                                          |
| `src/loader/dispatcher/wavenet.rs` | `AlignedVec`, `f32_to_bf16` → `crate::math::common`                                         |

### O que NÃO muda

- A trait `SimdMath` permanece a mesma interface (33 métodos)
- O macro `dispatch_simd!` permanece idêntico
- O sistema de detecção `detect_best_simd()` permanece idêntico
- `AlignedVec<T>` permanece funcionalmente idêntico
- Os algoritmos e constantes polinomiais são preservados bit-identical

### Riscos e Mitigações

| Risco                                | Mitigação                                                                                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Quebra de imports em muitos arquivos | Fazer a migração em passos: primeiro mover arquivos (mantendo re-exports no local antigo), depois atualizar consumidores, depois remover shims |
| Regressão de performance             | Rodar `cargo bench` e `cargo test` a cada passo                                                                                                |
| Complexidade excessiva da estrutura  | A estrutura proposta tem 6 subpastas vs 1 atual — aceitável dado o volume de código (4500+ linhas)                                             |
| A2 ainda é placeholder               | Deixar espaço para `src/math/a2/` quando necessário, sem criar pasta vazia agora                                                               |
| Perda da base de conhecimento        | Cuidadosamente levar junto com o código os respectivos comentários atualmente escritos - com adaptações às novas circunstâncias                |

---

## Plano de Execução (9 passos)

### Passo 1: Criar `constants.rs`

Extrair constantes de `fastmath.rs` (clamp limits, coeficientes Minimax, parâmetros de LUT) para `src/math/constants.rs`.

### Passo 2: Criar `src/math/common/` com foundation

Mover `traits.rs`, `dispatch.rs`, `aligned.rs`, `utility.rs`, `scalar_ref.rs`, `ops.rs` do antigo `simd/` para `common/`. Sem alterações de conteúdo. Criar `mod.rs` com re-exports.

### Passo 3: Criar `src/math/activations/`

Quebrar `fastmath.rs` extraindo cada função de ativação para seu próprio arquivo. Cada arquivo contém lado a lado as variantes AVX2 (`__m256`) e AVX-512 (`__m512`). `tests.rs` consolida os sweeps do `fastmath_test.rs`.

### Passo 4: Criar `src/math/gemm/`

Mover funções de dot product, GEMV, GEMM de `avx2.rs`/`avx512.rs` para `gemm/dot.rs`, `gemm/dot_4x.rs`, `gemm/gemv.rs`, `gemm/gemm_batch.rs`, `gemm/gemv_bf16.rs`. Cada arquivo com AVX2 + AVX-512 lado a lado.

### Passo 5: Criar `src/math/lstm/`

Mover `fused_lstm_gates_*` de `fastmath.rs` + `gemv_4gate_*` de `avx2.rs`/`avx512.rs` para `lstm/`.

### Passo 6: Criar `src/math/wavenet/`

Mover `batch_wavenet_head_sum_*`, `accumulate_head_*`, `tanh_and_accumulate_block_*`, `gated_activation_and_accumulate_block_*` de `avx2.rs`/`avx512.rs` para `wavenet/`.

### Passo 7: Criar `src/math/dsp/`

Mover `GainLUT` de `fastmath.rs` → `dsp/gain_lut.rs`. Mover `apply_gain_*`, `compute_energy_stereo_*`, `convolve_stereo_*` de `avx2.rs`/`avx512.rs` → `dsp/gain.rs`, `dsp/stereo.rs`.

### Passo 8: Refatorar `simd/avx2.rs` e `simd/avx512.rs` → `common/avx2_impl.rs` e `common/avx512_impl.rs`

Após mover todos os kernels para suas respectivas pastas, o que resta em `avx2.rs`/`avx512.rs` são apenas as implementações da trait `SimdMath` (`Avx2Math`, etc.) que delegam para os kernels. Mover essas structs + impl blocks para `common/avx2_impl.rs` e `common/avx512_impl.rs`.

### Passo 9: Atualizar imports em todos os consumidores + `src/math/mod.rs`

Atualizar `mod.rs` para nova estrutura. Atualizar todos os `use crate::math::` nos arquivos consumidores. Remover a antiga pasta `simd/`. Executar `cargo test` e `cargo bench` para validação final.

### Verificação

- `cargo check` — compilação sem erros
- `cargo test` — todos os testes passam (incluindo sweeps de erro máximo)
- `cargo clippy` — sem warnings
- `cargo bench` — sem regressão de performance (threshold: <2%)
