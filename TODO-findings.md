<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria Geral (revisor-auditor + refatora-rust)

> **Data:** 2026-06-22
> **Skills acionadas:** `revisor-auditor`, `refatora-rust`, `planejador-arquiteto`
> **Escopo:** Revisão profunda de todo o código-fonte (`src/`, `benches/`, `tests/`),
> dos logs de teste (`testes.log`) e da aderência arquitetural ao
> [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) e às
> regras em `.agents/rules/`.

---

## 0. Sumário Executivo

A base de código está em **excelente estado geral**. A auditoria confirmou pontos
fortes objetivos:

- **RT-Safety:** Nenhuma violação encontrada nos *hot paths* (CLAP `process()` e
  callback PipeWire). Zero alocação de heap, zero lock, zero `unwrap()`/`panic!`,
  zero `drop()` de heap na thread de áudio. O mecanismo SPSC GC
  (`src/common/spsc/gc.rs`) foi verificado e garante desalocação fora da thread RT
  via cascata SPSC → parking-lot → overflow buffer (`Box::into_raw` sem `drop`).
- **Qualidade de build:** `testes.log` mostra **0 warnings**, **770 testes unitários
  passando** (1 ignorado), suíte longa de ~48 min **100% PASSED** (soak, parity C++,
  golden vectors, heap-audit, clap-validator, concorrência).
- **Fidelidade ao NAM Core:** WaveNet (arrays duplos, convoluções dilatadas causais),
  LSTM (gate-major, pipelining 2 camadas), ConvNet e A2 (23 camadas, LeakyReLU 0.01,
  conv tap-major T=8 para CH=8) são portes fiéis, com otimizações Rust adicionais
  (dual-frame tiling, `MirroredBuffer` branchless, monomorfização por ISA).

As oportunidades de melhoria abaixo são, em sua maioria, **refinamentos de
performance e consistência arquitetural**, e não correções de bugs. A exceção
relevante é o **Finding F-01**, que tem impacto direto no caminho mais quente do
modelo mais caro (WaveNet CH16 = 4,45 ms/4096 samp, o benchmark mais lento).

### Baseline de performance (de `testes.log`, fase 6)

| Benchmark                                 | Tempo         | Observação                   |
| ----------------------------------------- | ------------- | ---------------------------- |
| `Long_Run_WaveNet/Standard_CH16_4096samp` | **4,4477 ms** | mais caro; alvo de F-01/F-10 |
| `Prewarm_A2Full_CH8_2048samp`             | 2,9029 ms     | A2 fast-path CH8             |
| `Prewarm_A2Lite_CH3_2048samp`             | 2,4514 ms     | A2 fast-path CH3             |
| `Long_Run_LSTM/2x16_4096samp`             | 818,04 µs     | —                            |
| `LSTM_2x24_Comparison/Scalar_Baseline`    | 581,51 µs     | baseline de referência       |

### Índice de Findings

| ID   | Severidade | Categoria                   | Título resumido                                                                                |
| ---- | ---------- | --------------------------- | ---------------------------------------------------------------------------------------------- |
| F-01 | **ALTA**   | Performance / Hot-path      | Dispatch em runtime por-chamada em `dot_product_4x` quebra a monomorfização da conv WaveNet/A2 |
| F-02 | MÉDIA      | Arquitetura                 | Débito de dispatch duplo (v-table `SimdMathConfig` vs trait `SimdMath`)                        |
| F-03 | MÉDIA      | Conformidade x86-64-v3      | Fallbacks `#[cfg(not(target_arch="x86_64"))]` mortos + projeto de-facto x86_64-only            |
| F-04 | BAIXA      | Conformidade x86-64-v3      | `is_x86_feature_detected!("avx2"/"fma")` em teste viola `rust.md §3`                           |
| F-05 | MÉDIA      | Organização (refatora-rust) | Arquivos/funções de hot-path grandes demais                                                    |
| F-06 | BAIXA      | Organização (refatora-rust) | Duplicação estrutural estático vs dinâmico                                                     |
| F-07 | BAIXA      | Limpeza                     | Código morto / test-only mal posicionado                                                       |
| F-08 | BAIXA      | Documentação                | Comentários enganosos sobre dispatch e hot-path                                                |
| F-09 | INFO       | Performance / Avant-garde   | Kernel de convolução 8/16-wide por canais de saída + reuso de acumuladores                     |
| F-10 | INFO       | DX / Tooling                | Cobertura de comentários `SAFETY` repetitivos e genéricos                                      |

---

## F-01 — [ALTA] Dispatch em runtime por-chamada em `dot_product_4x` quebra a monomorfização do hot-path WaveNet/A2

### F-01 — Localização

- `src/models/wavenet/conv_input.rs:116-122` (`dot_product_4x`)
- `src/models/wavenet/conv_input.rs:139-153` (`dot_product_4x_dual`)
- Chamadores no hot-path:
  - `src/models/wavenet/conv1d.rs:114` (`process_single_frame_with_mixin`, **produção**)
  - `src/models/wavenet/conv1d_dyn.rs:107` (`process_single_frame`, A2 + WaveNet dyn)
  - `src/models/wavenet/conv1d_dual.rs` (`process_dual_frame_with_mixin`, via `dot_product_4x_dual`)

### F-01 — Evidência

A função `process_internal::<M: SimdMath>` do WaveNet é corretamente monomorfizada
pela macro `dispatch_simd!` (`src/models/wavenet/model.rs:46`), e o comentário em
`model.rs:41-44` descreve o "truque de mágica": avaliar o hardware **uma única vez** e
"teletransportar" para a versão monomorfizada. A cadeia permanece genérica em `M` até
`WaveNetLayer::process_block_internal<M>` (`src/models/wavenet/layer.rs:28`).

Porém, no **folha da árvore de chamadas** — exatamente o laço interno mais quente — a
monomorfização é **descartada**:

```rust
// src/models/wavenet/conv_input.rs:116
#[inline(always)]
pub(crate) fn dot_product_4x(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    if is_x86_feature_detected!("avx512f") {                       // ← detecção POR CHAMADA
        unsafe { dot_product_4x_f32_avx512(weights, state) }
    } else {
        unsafe { dot_product_4x_f32_avx2(weights, state) }
    }
}
```

Esse `dot_product_4x` **não é genérico em `M`** e **não tem `#[target_feature]`**.
É invocado por bloco-de-4-canais × por tap × por camada × por array. Para o WaveNet
CH16 (2 arrays, múltiplas camadas, `OUT/4 = 4` blocos × `K = 3` taps × 64 frames/chunk),
isso representa **milhares de execuções por buffer** de:

1. Um *branch* sobre `is_x86_feature_detected!("avx512f")` (load atômico cacheado +
   teste + desvio) a cada chamada — apesar de o resultado ser invariante após o boot.
2. **Impossibilidade de inline** do kernel AVX-512: `dot_product_4x_f32_avx512` é
   `#[target_feature(enable = "avx512f,avx512vl")]` (`src/math/gemm/dot_4x/dot_f32_avx512.rs:52`)
   e **não pode** ser inlinado no chamador (que não tem AVX-512 no conjunto de features).
   O branch AVX-512 é, portanto, sempre uma chamada de função real com prólogo/epílogo.
3. **Perda de alocação de registradores entre taps:** como cada `dot_product_4x`
   retorna `[f32; 4]` por valor, os acumuladores fazem *round-trip* pela pilha/registradores
   de retorno a cada tap, em vez de permanecerem vivos em ZMM/YMM ao longo do laço de taps.

Isso contradiz tanto o comentário em `model.rs:41-44` quanto a regra
`.agents/rules/rust.md §3`: *"Dynamic dispatch should only exist for higher
extensions like AVX-512"* — a intenção é dispatch **uma vez no topo**, não a cada folha.

### F-01 — Impacto

- **Performance:** caminho mais quente do modelo mais caro (WaveNet, 4,45 ms). A
  eliminação do branch + inline do kernel correto + reuso de acumuladores em registradores
  deve render ganho mensurável (estimativa: single-digit a low-double-digit % no WaveNet
  e na via fallback do A2 dinâmico).
- **Consistência arquitetural:** restaura o invariante "dispatch único + monomorfização"
  já adotado em todo o resto do WaveNet/LSTM/DSP.

### F-01 — Proposta de solução

1. **Adicionar dois métodos f32-nativos ao trait `SimdMath`** (`src/math/common/traits.rs`,
   grupo "(A) Dot Products", junto de `dot_product_4x_interleaved`):

   ```rust
   unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4];
   unsafe fn dot_product_4x_f32_dual(
       weights: &[[f32; 4]], state_f0: &[f32], state_f1: &[f32],
   ) -> ([f32; 4], [f32; 4]);
   ```

   - `Avx2Math` (`avx2_impl.rs`) delega para `dot_product_4x_f32_avx2`.
   - `Avx512Math`/`Avx512VnniBf16Math` (`avx512/`) delegam para `dot_product_4x_f32_avx512`.

2. **Tornar genéricas as funções de convolução** que hoje chamam o wrapper:
   `Conv1d::process_single_frame_with_mixin`, `Conv1d::process_dual_frame_with_mixin`,
   `Conv1dDyn::process_single_frame`, `Conv1dDyn::process_block`/`process_dual_frame` →
   `<M: SimdMath>`, trocando `dot_product_4x(..)` por `M::dot_product_4x_f32(..)`.
   O parâmetro `M` **já está em escopo** no chamador (`layer.rs:28`, `layer_array.rs`,
   `a2/.../dynamic.rs`), então a propagação é mecânica.

3. **Remover** o wrapper `dot_product_4x`/`dot_product_4x_dual` de `conv_input.rs` (ou
   mantê-lo apenas sob `#[cfg(test)]` se algum teste o consumir diretamente).

4. **Validar** com `tests/cpp_parity.rs`, `tests/golden_vectors.rs` (paridade numérica
   < 2 ULP é preservada — mesmos kernels FMA) e re-rodar `benches/inference_bench.rs`
   (`Long_Run_WaveNet`) comparando antes/depois.

> **Risco:** Baixo-médio. Mudança puramente de roteamento/assinatura, sem alterar a
> matemática. Coberto por testes de paridade C++ e golden vectors existentes.

---

## F-02 — [MÉDIA] Débito de dispatch duplo (v-table `SimdMathConfig` vs trait `SimdMath`)

### F-02 — Localização

- `src/math/common/dispatch/config.rs:7-44` (nota "DESIGN DEBT" já documentada)
- `src/math/common/mod.rs:46-90` (macro `dispatch_simd!`, Modos 1/2/3)
- Consumidores do Modo 3 (v-table) em caminho de áudio: `src/dsp/pipeline/stages/input.rs`,
  `output.rs`, `src/math/activations/mod.rs` (todas as `*_slice` via `SIMD_MATH.func`).

### F-02 — Evidência

Coexistem dois mecanismos de dispatch:

1. **Mecanismo 1 (trait, monomorfização):** WaveNet, LSTM, gate, resampler.
2. **Mecanismo 2 (v-table, ponteiros de função):** pipeline DSP (gain, energia, dither,
   clipping) e ativações em fatia (`tanh_slice`, etc.), via chamadas indiretas
   `(SIMD_MATH.func)(args)`.

A própria nota em `config.rs:42` afirma: *"Priority: Medium (does not affect performance
on hot paths, which already use Mechanism 1 with monomorphization)"*. **Essa afirmação é
parcialmente imprecisa** à luz do F-01 (a conv WaveNet **não** estava 100% monomorfizada)
e porque os estágios `input.rs`/`output.rs` do pipeline (executados todo buffer) usam o
Modo 3 com chamada indireta (impede inline, ~1-2 ciclos + barreira de otimização por
chamada).

### F-02 — Impacto

- Manutenção: ~35 ponteiros de função em `SimdMathConfig` + boilerplate em
  `detect_best_simd()` duplicado por ISA (3×).
- Performance marginal: chamadas indiretas no pipeline de gain/energia/ativação.

### F-02 — Proposta de solução

Executar o "Unification plan" já descrito em `config.rs:35-40`, em fases:

1. Migrar os estágios de pipeline (`input.rs`, `output.rs`) e as ativações em fatia para
   `dispatch_simd!` Modo 1 (trait monomorfizado), recebendo `M: SimdMath` a partir de um
   único `dispatch_simd!` no topo do `capture_dsp_pipeline` (`src/dsp/pipeline/capture.rs`).
2. Reduzir `SimdMathConfig` a apenas `InstructionSet` + `name` + `is_avx512` (consultas de
   capacidade), removendo os ponteiros de função.
3. Eliminar `config_table!` boilerplate e as ~50 linhas redundantes em `detect.rs`.

> **Risco:** Médio. Toca o orquestrador do pipeline. Fazer após F-01 (que estabelece o
> padrão de propagação de `M`). Cobertura: `tests/spsc_pipeline.rs`, `tests/pipeline_soak.rs`,
> heap-audit.

---

## F-03 — [MÉDIA] Fallbacks `#[cfg(not(target_arch="x86_64"))]` mortos: projeto é de-facto x86_64-only

### F-03 — Localização

- `src/math/common/dispatch/detect.rs:24-96` — `detect_best_simd()` possui **apenas**
  o braço `#[cfg(target_arch = "x86_64")]`; **não há** retorno para não-x86 → **não
  compila** fora de x86_64.
- `src/models/wavenet/layer_array.rs:8` — `use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};`
  **sem** guarda `#[cfg]` → idem.
- `src/standalone/rt_setup/mod.rs:3` — `#![cfg(target_arch = "x86_64")]` (módulo inteiro x86).
- Branches mortos `#[cfg(not(target_arch = "x86_64"))]` espalhados em produção:
  `src/dsp/cabsim/conv.rs:338`, `src/models/a2/model/mod.rs:317`,
  `src/models/convnet/batch_norm.rs:123` e `:210`, `src/clap/processor/mod.rs:252`,
  `src/clap/processor/dsp/telemetry.rs:16`, `src/math/common/half.rs:148` e `:173`.

### F-03 — Evidência

Como `detect.rs` e `layer_array.rs` não compilam fora de x86_64, **todos** os ramos
`#[cfg(not(target_arch = "x86_64"))]` são código inalcançável em qualquer configuração
compilável. Isso contraria diretamente `.agents/rules/rust.md §3`:
*"AVX2 and FMA instructions must be used natively and unconditionally throughout the
entire codebase, including outside the hot path."* — ou seja, o projeto assume x86-64-v3
incondicionalmente; manter "fallbacks portáveis" é **complexidade morta** que polui a
leitura e dá falsa impressão de portabilidade.

### F-03 — Impacto

- Manutenção/legibilidade: ~7 arquivos com caminhos duplicados (SIMD + escalar) que nunca
  são compilados.
- Risco de divergência silenciosa (o ramo escalar pode "apodrecer" sem cobertura de teste).

### F-03 — Proposta de solução

Decisão de produto necessária (recomendação: **assumir x86_64-only**, coerente com a regra):

- **Opção A (recomendada):** remover todos os ramos `#[cfg(not(target_arch = "x86_64"))]`
  de produção, deixando o código x86-64-v3 nativo e incondicional. Adicionar, se desejado,
  um `compile_error!` claro em um ponto central (`lib.rs`) para arquiteturas não suportadas:

  ```rust
  #[cfg(not(target_arch = "x86_64"))]
  compile_error!("nam-rs requer x86-64-v3 (AVX2/FMA/BMI2). Veja .cargo/config.toml.");
  ```

- **Opção B:** se portabilidade ARM/aarch64 for meta real e futura, então `detect.rs`,
  `layer_array.rs` e `rt_setup` precisam de braços não-x86 **funcionais** e cobertos por CI —
  caso contrário os fallbacks existentes são teatro.

> **Risco:** Baixo (Opção A é remoção de código morto). Validar com `utils/lints.sh`.

---

## F-04 — [BAIXA] `is_x86_feature_detected!("avx2"/"fma")` em teste viola `rust.md §3`

### F-04 — Localização

- `src/math/activations/activations_test.rs:208`

### F-04 — Evidência

```rust
if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
    return Ok(());
}
```

AVX2 e FMA são **garantidos** pelo baseline x86-64-v3 (`.cargo/config.toml:11`). Esse
guard é *dead code* (nunca pula em máquina conforme) e é exatamente o padrão proibido por
`.agents/rules/rust.md §3`: *"Never ... use `is_x86_feature_detected!("avx2")` ..."*. A
regra não isenta testes. (Observação: as demais ~60 ocorrências de `is_x86_feature_detected!`
no código gateiam **AVX-512** — `avx512f/vl/dq/bf16/vnni/bw` —, que é legítimo por ser ISA
acima do baseline; ver `detect.rs:29,52` e `common/diagnostics/system_info.rs:95-110`.)

### F-04 — Impacto

Mínimo (apenas conformidade/limpeza), mas é a **única** transgressão literal do padrão
proibido encontrada na base.

### F-04 — Proposta de solução

Remover o guard inteiro (o teste sempre roda em x86-64-v3). Se houver intenção de proteger
builds não-x86, substituir por `#[cfg(target_arch = "x86_64")]` no atributo do teste — mas,
dado F-03, o ideal é simplesmente remover.

> **Risco:** Nenhum. Alteração de teste.

---

## F-05 — [MÉDIA] Arquivos e funções de hot-path grandes demais (refatora-rust)

### F-05 — Localização (arquivos de produção > 400 linhas, excluindo `_test.rs`)

| Arquivo                                 | Linhas | Observação                                       |
| --------------------------------------- | ------ | ------------------------------------------------ |
| `src/models/a2/model/dynamic.rs`        | 835    | `process()` ~240 linhas + `set_weights()` grande |
| `src/models/a2/grouped_conv1d.rs`       | 725    | múltiplos kernels (single/dual/depthwise)        |
| `src/models/a2/model/mod.rs`            | 656    | **`process()` ~320 linhas** (276-596)            |
| `src/models/convnet/batch_norm.rs`      | 543    | SIMD inline + ramos cfg                          |
| `src/models/wavenet/post_stack_head.rs` | 544    | inflado por testes-diagnóstico inline            |
| `src/math/gemm/gemm_batch/avx2.rs`      | 542    | vários kernels num só arquivo                    |
| `src/models/a2/conv1d_ch3/simd.rs`      | 529    | cauda escalar verbosa (466-528)                  |
| `src/clap/gui/ui/zones/identity.rs`     | 506    | GUI                                              |
| `src/math/dsp/fft.rs`                   | 506    | —                                                |

### F-05 — Evidência / Impacto

A regra `.agents/rules/testing.md` (resumida em `AGENTS.md`) pede testes inline somente
para arquivos < 300 linhas; vários arquivos acima misturam produção + testes inline (ex.:
`a2/activations.rs` tem 822 linhas, mas ~700 são testes). E `a2/model/mod.rs::process()`
com ~320 linhas numa única função dificulta navegação, revisão e profiling.

### F-05 — Proposta de solução (sem alterar lógica/algoritmos — refatora-rust)

1. **Extrair testes inline** para arquivos `_test.rs` irmãos onde o arquivo de produção
   ≥ 300 linhas (ex.: `post_stack_head.rs`, `convnet/model.rs`, `convnet/block.rs`).
2. **Decompor `WaveNetA2::process()`** (`a2/model/mod.rs:276`) em sub-funções `#[inline(always)]`
   coesas: `rechannel_prescale()`, `advance_head_ring()`, `layer_forward_dispatch()`,
   `head_finalize()`. Mesma decomposição para `WaveNetA2Dyn::process()` e
   `set_weights()` em `dynamic.rs`.
3. **Quebrar `gemm_batch/avx2.rs`** por kernel (`fused_add_gemm_batch.rs`,
   `fused_residual_batch.rs`) sob um `gemm_batch/` já existente.

> **Risco:** Baixo (movimentação estrutural). Obrigatório validar com `utils/lints.sh` +
> `utils/tests-quick.sh` sem warnings (exigência da skill `refatora-rust`).

---

## F-06 — [BAIXA] Duplicação estrutural estático vs dinâmico

### F-06 — Localização

- `src/models/wavenet/conv1d.rs` ↔ `conv1d_dyn.rs` (kernel single-frame quase idêntico).
- `src/models/wavenet/layer.rs` ↔ camada dinâmica (mesmo fluxo dual-frame + buffers 1024).
- `src/models/lstm/model1.rs` (`define_lstm1_process!`) ↔ `model2.rs`
  (`define_lstm2_process_pipelined!`): lógica de projeção do head (branch f32_head,
  `dot_product`, bias) duplicada.
- `WaveNetA2::prewarm()` (`a2/model/mod.rs:606`) ↔ `WaveNetA2Dyn::prewarm()`
  (`a2/model/dynamic.rs:776`): idênticos, incluindo `vec![0.0; block]`.
- `src/loader/dispatcher/wavenet/standard.rs::build_wavenet_typed()` ↔
  `dynamic.rs::build_wavenet_array_dyn()` (~150 linhas de leitura de pesos quase iguais).
- `src/loader/build.rs:32-58` ↔ `:89-116` (leitura/validação de tamanho de arquivo `.namb`
  vs `.nam`).

### F-06 — Proposta de solução

- Unificar o head-projection do LSTM num macro/fn compartilhado consumido por `model1`/`model2`.
- Extrair `prewarm` comum do A2 para `fn a2_prewarm_common(...)` (análogo a
  `src/models/lstm/prewarm.rs::lstm_prewarm_common`).
- Em `build.rs`, extrair `fn read_and_validate_model_bytes(path, sys) -> Result<Vec<u8>>`.
- Para conv1d/layer estático vs dinâmico: avaliar (após F-01) se a versão estática pode
  ser expressa como caso `const`-generic da dinâmica sem perder elisão de bounds-check.

> **Risco:** Baixo-médio. Caminho de loader é cold (sem RT). Conv/layer requer cuidado
> com bounds-check; cobrir com golden vectors.

---

## F-07 — [BAIXA] Código morto / test-only mal posicionado

### F-07 — Localização

- `src/models/wavenet/conv_input.rs:14-37` — `init_accum_with_bias_mixin` marcado
  `#[cfg_attr(not(test), allow(dead_code))]`; só é usado pelo `process_single_frame`
  test-only de `conv1d.rs`. Deveria viver no módulo de teste ou ser `#[cfg(test)]`.
- Após F-01: o wrapper `dot_product_4x`/`dot_product_4x_dual` se torna removível.
- `src/models/wavenet/conv1d.rs:130-224` — `process_single_frame`/`process_block`
  inteiramente `#[cfg(test)]`; considerar mover para `conv1d_test.rs`.

### F-07 — Proposta de solução

Mover helpers exclusivos de teste para os respectivos `_test.rs` (ou `#[cfg(test)] mod`),
eliminando `allow(dead_code)`. Coerente com `refatora-rust` ("remover código morto").

> **Risco:** Nenhum.

---

## F-08 — [BAIXA] Comentários enganosos sobre dispatch e hot-path

### F-08 — Localização

- `src/models/wavenet/model.rs:41-44` — descreve "avalia o hardware uma vez e teletransporta
  para versão monomorfizada"; **verdadeiro só após F-01** (hoje a folha re-detecta por chamada).
- `src/math/common/dispatch/config.rs:42` — "does not affect performance on hot paths" —
  impreciso (ver F-01/F-02).
- `src/models/wavenet/conv_input.rs:106-114,124-137` — docstrings descrevem a detecção em
  runtime como design intencional; revisar após F-01.

### F-08 — Proposta de solução

Atualizar os comentários junto com a implementação de F-01/F-02 (skill `refatora-doc`/
`documentador`), mantendo `docs/architecture.md` sincronizado (seção de dispatch SIMD).

> **Risco:** Nenhum.

---

## F-09 — [INFO] Avant-garde: kernel de convolução 8/16-wide por canais de saída + reuso de acumuladores

### F-09 — Contexto

Hoje a conv f32 processa **4 canais de saída por chamada** (`dot_product_4x`), com laço
externo iterando `OUT/4` blocos. Para WaveNet CH16 (OUT=16 → 4 blocos) e A2 CH8, há
oportunidade de **largura maior por iteração** e melhor reuso de pesos/estado.

### F-09 — Proposta de exploração (após F-01)

- Avaliar kernel `dot_product_8x_f32_avx2` (8 canais de saída por `__m256`) e
  `dot_product_16x_f32_avx512` (16 canais por `__m512`), reduzindo o número de chamadas e
  o overhead de redução horizontal, e mantendo acumuladores em registradores ao longo do
  laço de taps (habilitado pela monomorfização de F-01).
- Medir com `benches/dot_4x_bench.rs` (já existe) estendido para 8x/16x e
  `benches/inference_bench.rs` (`Long_Run_WaveNet`).
- Acionar skill `pesquisador-inovador` para prototipagem e validação de ULP/paridade.

> **Risco:** Médio (novo kernel SIMD). Gate por golden vectors + cpp_parity. Estritamente
> opt-in até comprovar ganho.

---

## F-10 — [INFO] Comentários `SAFETY` repetitivos e genéricos

### F-10 — Localização

- `src/math/common/dispatch/config.rs:56-150` — ~30 linhas idênticas
  `// SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.`
- Padrão semelhante em vários `avx512/*.rs`.

### F-10 — Evidência / Impacto

O lint `#![warn(clippy::undocumented_unsafe_blocks)]` (`src/math/common/mod.rs:4`) é
satisfeito, mas comentários genéricos repetidos não agregam informação de segurança real
(quais invariantes? alinhamento? bounds?). Reduz o valor da auditoria de `unsafe`.

### F-10 — Proposta de solução

Substituir, nos pontos de maior risco (kernels SIMD com `get_unchecked`/`loadu`), por
comentários `SAFETY` específicos (precondições de tamanho/alinhamento concretas). Para a
v-table, a documentação de segurança pode migrar para a doc do trait/método. Coerente com
skill `refatora-doc`.

> **Risco:** Nenhum (documentação).

---

## Epics (agrupamento para execução)

> Agrupados para maximizar segurança (testes de paridade como rede) e minimizar retrabalho.
> A ordem sugerida respeita dependências (E1 estabelece o padrão de propagação de `M`
> reutilizado por E2).

### EPIC A — Unificação de Dispatch SIMD e Hot-Path WaveNet/A2 [DONE]

**Findings:** F-01 (núcleo), F-02, F-08.
**Objetivo:** Eliminar dispatch em runtime nas folhas, completar a monomorfização do
pipeline e corrigir a documentação correlata.
**Sequência:** F-01 → F-02 → F-08.
**Critério de aceite:** `cpp_parity` + `golden_vectors` verdes; benchmark `Long_Run_WaveNet`
sem regressão (idealmente com ganho); zero warnings.
**Risco:** Médio-alto (toca o caminho mais quente) — **maior atenção da auditoria**.

### EPIC B — Conformidade x86-64-v3 e Limpeza de Portabilidade [DONE]

**Findings:** F-03, F-04, F-07.
**Objetivo:** Remover código morto não-x86, eliminar a transgressão `is_x86_feature_detected!("avx2")`
e helpers test-only mal posicionados; adicionar `compile_error!` guard.
**Sequência:** F-04 → F-03 → F-07.
**Critério de aceite:** `utils/lints.sh` limpo; build x86_64 inalterado.
**Risco:** Baixo (predominantemente remoção).

### EPIC C — Refatoração Estrutural (refatora-rust) [DONE]

**Findings:** F-05, F-06, F-10.
**Objetivo:** Reduzir tamanho de arquivos/funções de hot-path, extrair testes inline,
desduplicar estático vs dinâmico e melhorar comentários `SAFETY`.
**Sequência:** F-05 (extrair testes + decompor `process()`) → F-06 (desduplicação) → F-10.
**Critério de aceite:** Nenhuma mudança de lógica (golden vectors idênticos);
`utils/lints.sh` + `utils/tests-quick.sh` sem warnings/erros.
**Risco:** Baixo-médio.

### EPIC D — Exploração de Performance Avant-garde [DONE]

**Findings:** F-09.
**Objetivo:** Prototipar kernels de conv 8/16-wide e medir ganho real.
**Dependência:** Requer EPIC A concluído (monomorfização).
**Critério de aceite:** Ganho comprovado em benchmark com paridade ULP mantida; caso
contrário, descartar.
**Risco:** Médio — estritamente opt-in / experimental.

---

> **Próximo passo sugerido:** Para transformar estes Epics em sprints e tarefas técnicas
> atômicas (`TODO-sprints.md`), acionar novamente a skill `planejador-arquiteto`
> explicitamente. Este artefato (`TODO-findings.md`) cobre apenas os achados e propostas.

---
---

## RODADA 2 — Auditoria de Execução + Novos Achados (2026-06-23)

> **Skills acionadas:** `revisor-auditor`, `refatora-rust`, `planejador-arquiteto`
> **Gatilho:** Após conclusão dos EPICs A–D (F-01…F-10) e dos Sprints 1–3, foi
> solicitada (a) auditoria de execução do que foi implementado e (b) nova rodada
> de caça obsessiva a transgressões `is_x86_feature_detected!`, análise dos logs
> (`utils/tests-long.sh` — 6 fases PASSED) e varredura `refatora-rust`.
> **Máquina de auditoria:** x86-64-v3 **AVX2-only** (sem AVX-512 — confirmado em
> `/proc/cpuinfo` e por todos os testes/benches `avx512` marcados `ignored`).
> Esse detalhe é **central** para os achados NF-01 e NF-04.

## R2.0 — Veredito da Auditoria de Execução (EPICs A–D)

Verificação direta no código-fonte do estado pós-implementação:

| Finding                                                | Status                     | Evidência objetiva                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------ | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F-01** (dispatch por-chamada no conv)                | ✅ **Resolvido (WaveNet)** | `conv1d.rs` e `conv1d_dyn.rs` agora `process_single_frame<M: SimdMath>`, usando `M::dot_product_{4,8,16}x_f32`. O wrapper `dot_product_4x` com `is_x86_feature_detected!` foi **removido** de `conv_input.rs`. **Resíduo:** A2 dinâmico (ver NF-02).                                                            |
| **F-02** (dispatch duplo / v-table)                    | ✅ **Resolvido**           | `dispatch/config.rs` reduzido de ~150 → **33 linhas**; `SimdMathConfig` é só `{instruction_set, name, is_avx512}` — **zero ponteiros de função**. Tudo via `dispatch_simd!` + trait.                                                                                                                            |
| **F-03** (fallbacks não-x86 mortos)                    | ✅ **Resolvido**           | Resta **1** `cfg(not(target_arch="x86_64"))`: o guard central `compile_error!("NAM-rs requires x86_64 architecture")` em `lib.rs:23-24`. Demais ramos mortos eliminados (Opção A).                                                                                                                              |
| **F-04** (`is_x86_feature_detected!("avx2")` em teste) | ✅ **Resolvido**           | Zero ocorrências de `avx2`/`fma`/`bmi2` em todo o `src/`. As ~70 detecções restantes são **exclusivamente** `avx512*` (legítimas).                                                                                                                                                                              |
| **F-05** (arquivos grandes)                            | ⚠️ **Parcial / Regressão** | Testes extraídos ✓ (`post_stack_head_test.rs`, `convnet_model_test.rs`, `batch_norm_test.rs`); `gemm_batch/avx2.rs` dividido ✓ (`fused_add_gemm_batch.rs`, `fused_residual_batch.rs`). **Porém** `a2/model/dynamic.rs` **cresceu para 959 linhas** (maior do projeto) e `a2/model/mod.rs` para 726 (ver NF-03). |
| **F-06** (duplicação estático/dinâmico)                | ✅ **Resolvido**           | `a2_prewarm_common`, `read_and_validate_model_bytes`, head-projection LSTM unificados; const-generic avaliado e descartado com justificativa (Sprint 2).                                                                                                                                                        |
| **F-07** (código test-only mal posicionado)            | ✅ **Resolvido**           | `init_accum_with_bias_mixin` removido de `conv_input.rs` (substituído por `load/store_{4,8,16}_accums`).                                                                                                                                                                                                        |
| **F-08** (comentários enganosos)                       | ✅ **Resolvido**           | Comentários de dispatch atualizados (ex.: `model_dyn.rs:229-232` descreve corretamente a leitura `LazyLock` em vez de `is_x86_feature_detected!` por chamada).                                                                                                                                                  |
| **F-09** (kernels 8x/16x)                              | ✅ **Implementado**        | `gemm/dot_8x/` e `gemm/dot_16x/` criados, integrados em `conv1d.rs`/`conv1d_dyn.rs`, com benches e testes. **Porém** a integração introduziu NF-01, NF-04 e NF-05.                                                                                                                                              |
| **F-10** (comentários `SAFETY` genéricos)              | ⚠️ **Parcial / Mooted**    | Em `config.rs` ficou irrelevante (v-table eliminada). Comentários genéricos *"Inner safety guarantees are upheld by caller invariants…"* **ainda presentes** em `traits.rs:212,230,445,459,474,484` (ver NF-06).                                                                                                |

**Conclusão da auditoria:** O plano foi executado com alta fidelidade — **8 de 10 findings totalmente resolvidos**, suíte longa 100% PASSED (6 fases), 779 testes lib + 26 golden vectors sem regressão numérica. Contudo, a integração dos kernels largos (F-09/EPIC D) **introduziu uma regressão de princípio** no alvo baseline (AVX2) e **deixou otimizações na mesa**, detalhadas abaixo. A análise dos benchmarks (`target/logs/phase5-benchmarks.log`) foi decisiva para revelar o NF-01.

---

## NF-01 — [ALTA] Caminho de convolução 16-wide usa referência **escalar** no baseline AVX2 (x86-64-v3) — não extrai o máximo da ISA por padrão

### NF-01 — Localização

- `src/math/common/avx2_impl.rs:96-100` — `Avx2Math::dot_product_16x_f32` delega para
  `scalar_ref::dot_product_16x_f32_scalar`.
- `src/math/common/scalar_ref/dot.rs` — `dot_product_16x_f32_scalar`: laço escalar puro
  (16 × `mul_add` por elemento de `state`, **sem** `#[target_feature]`, sem intrínsecos).
- `src/loader/dispatcher/wavenet/layout.rs:362-370` — `select_interleave_width(out_ch)`
  retorna **16** para todo `out_ch` múltiplo de 16 (ex.: WaveNet CH16), **independente da ISA**.
- Consumidores: `conv1d.rs:103,210` e `conv1d_dyn.rs:257` (`M::dot_product_16x_f32`).

### NF-01 — Evidência (medições do log de benchmarks, máquina AVX2-only)

A combinação `select_interleave_width(16) = 16` + `Avx2Math::dot_product_16x_f32 = escalar`
significa que, **na ISA baseline do projeto (AVX2/x86-64-v3)**, o laço interno mais quente do
WaveNet CH16 (o modelo mais caro, 4,5 ms) é compilado a partir de uma função **literalmente
chamada `_scalar`**. Não há kernel AVX2 16-wide explícito (o diretório `gemm/dot_16x/` contém
apenas `dot_f32_avx512.rs` e `scalar.rs`).

Comparação direta de throughput (`target/logs/phase5-benchmarks.log`, `state = 64`):

| Kernel                             | Tempo (state=64) | Observação                                               |
| ---------------------------------- | ---------------- | -------------------------------------------------------- |
| `dot_8x_f32/avx2_64` (explícito)   | **15,46 ns**     | 8 canais, AVX2/FMA, 4 acumuladores                       |
| `dot_8x_f32/scalar_64` (auto-vec)  | 39,79 ns         | **2,57× mais lento** que o explícito                     |
| `dot_16x_f32/scalar_64` (auto-vec) | 55,79 ns         | 16 canais; ~**1,8×** mais lento que 2×`avx2_8x` (~31 ns) |

**Dois fatos críticos emergem:**

1. A auto-vetorização do LLVM **não** iguala o kernel explícito: o `dot_8x` AVX2 escrito à
   mão é **2,57× mais rápido** que a versão escalar auto-vetorizada equivalente. Logo, confiar
   na auto-vetorização do `_scalar` 16-wide **deixa performance na mesa**.
2. O caso comum WaveNet CH16 tem `in_ch = 16` (`state = 16`), onde `dot_16x_f32/scalar_16`
   (9,0 ns) ≈ 2×`dot_8x_f32/avx2_16` (2×4,77 = 9,54 ns) — por isso o benchmark de inferência
   **melhorou** ~2,7–4,3% mesmo nesta máquina. **Mas** isso é (a) **frágil** (depende do humor
   do auto-vetorizador entre versões de compilador) e (b) **degrada** para `in_ch` maiores
   (em `state=64`, perde ~45%).

Isso colide frontalmente com a diretriz do projeto: *"Otimize todo e qualquer código (mesmo
fora do hot path) para tirar o máximo de x86-64-v3 **POR PADRÃO**"* e com `.agents/rules/rust.md §3`
(AVX2/FMA nativos e incondicionais). Um kernel `_scalar` servindo o caminho mais quente da
ISA baseline é exatamente o anti-padrão que a diretriz proíbe — ainda que mascarado pela
auto-vetorização no caso `in_ch=16`.

### NF-01 — Impacto

- **Performance:** perda potencial de até ~1,8× no passo de convolução 16-wide para `in_ch`
  grande; perda garantida vs. kernel explícito em qualquer `in_ch` (auto-vec < explícito).
- **Risco/robustez:** regressão silenciosa se uma futura versão do `rustc`/LLVM deixar de
  auto-vetorizar o `_scalar` — o WaveNet CH16 cairia para escalar puro (4–8×) sem aviso.
- **Conformidade:** viola o princípio "max x86-64-v3 por padrão" no alvo baseline.

### NF-01 — Proposta de solução

1. **Implementar kernel AVX2 16-wide explícito** OU, de forma trivial e segura, fazer
   `Avx2Math::dot_product_16x_f32` **decompor em 2× `dot_product_8x_f32_avx2`** (a própria
   `Task 3.1` já antecipou essa decomposição como "otimização futura"):

   ```rust
   #[inline(always)]
   unsafe fn dot_product_16x_f32(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
       // reinterpreta cada linha [f32;16] como dois blocos [f32;8] contíguos
       let (lo, hi) = split_16_into_two_8(weights); // via cast de ponteiro, sem cópia
       let a = dot_product_8x_f32_avx2(lo, state);
       let b = dot_product_8x_f32_avx2(hi, state);
       concat_8_8(a, b)
   }
   ```

   Como `[[f32;16]]` é contíguo, os dois blocos de 8 são fatias reinterpretadas por cast de
   ponteiro (zero cópia). Mantém 2 YMM vivos por linha — uso pleno de AVX2.

2. **Validar paridade** com `dot_16x` tests (já existem `_vs_scalar`/`_vs_4x_decompose`) e
   re-rodar `dot_4x_bench`/`inference_bench` confirmando ganho em `in_ch` ≥ 32.

3. **Opcional (defesa em profundidade):** renomear `dot_product_16x_f32_scalar` para deixar
   claro que é **oráculo de teste**, não kernel de produção (ver NF-05).

> **Risco:** Baixo. A matemática é idêntica (mesmos FMAs); coberto por testes de decomposição
> e golden vectors. **Prioridade: ALTA** — é o único ponto onde o baseline AVX2 não está
> maximizado por construção explícita.

---

## NF-02 — [MÉDIA] Engine A2 dinâmico (`WaveNetA2Dyn`) faz dispatch de ISA **por-frame** — resíduo do F-01 não monomorfizado no topo

### NF-02 — Localização

- `src/models/a2/conv1d_dispatch.rs:30,71` — `A2Conv1d::process_single_frame` /
  `process_block` fazem `if is_x86_feature_detected!("avx512f") { …::<Avx512Math> } else { …::<Avx2Math> }`.
- Chamador de produção: `src/models/a2/model/dynamic.rs:847` (`WaveNetA2Dyn::layer_forward_dispatch`,
  laço `for f in 0..nf`) e `src/models/a2/layer.rs:265`.
- Roteamento de produção: `src/loader/dispatcher/wavenet/mod.rs:145-166` instancia
  `WaveNetA2Dyn` para `A2TopologyResult::Dynamic` (topologias A2 com `bottleneck != channels`,
  head1x1, etc.) — **não é teste nem `dynamic-engine`-only**.

### NF-02 — Evidência

`WaveNetA2Dyn::process()` (`dynamic.rs:580`) **não** usa `dispatch_simd!` — ao contrário do
WaveNet dinâmico (`model_dyn.rs:236`, que monomorfiza corretamente). Em vez disso, a seleção
AVX2/AVX-512 desce até a folha e é decidida **por frame** dentro de `conv1d_dispatch.rs`. Isso
recria exatamente o anti-padrão do F-01 (branch + impossibilidade de inline do kernel
`#[target_feature]` AVX-512), agora no engine A2 dinâmico.

Vale notar: `is_x86_feature_detected!("avx512f")` em si é a forma **permitida** (detecção de
ISA acima do baseline). O problema **não** é a transgressão `avx2`, e sim a **frequência**
(por-frame) e a quebra da arquitetura "dispatch único no topo + monomorfização" que o F-01
estabeleceu para o resto da base. A nota da `Task 2.4` já reconhecia explicitamente este
resíduo como adiado.

> **Nota de escopo:** o caminho A2 *fast-path* (CH3/CH8 const-generic, `model/mod.rs`) **não**
> sofre disso — o CH8 despacha **uma vez por camada** lendo `SIMD_MATH.instruction_set`
> (`mod.rs:474`), o que é aceitável. O fallback per-frame em `mod.rs:528` é
> `#[cfg(any(test, feature="dynamic-engine"))]`. Portanto, em produção, o impacto restringe-se
> às topologias A2 roteadas para `WaveNetA2Dyn`.

### NF-02 — Proposta de solução

Monomorfizar `WaveNetA2Dyn` no topo, espelhando `model_dyn.rs`:

1. Adicionar `process_internal<M: SimdMath>` e despachar via `crate::math::common::dispatch_simd!(self, process_internal)`
   no início de `WaveNetA2Dyn::process()`.
2. Propagar `M` por `layer_forward_dispatch::<M>` → tornar `A2Conv1d::process_single_frame<M>`
   genérico (remover o `if is_x86_feature_detected!` de `conv1d_dispatch.rs`), delegando para
   `Conv1dDyn::process_single_frame::<M>` (que **já** é genérico) e para o kernel grouped.
3. Idem para `A2Layer::process_single_frame` (`layer.rs:246`).

> **Risco:** Médio. Toca o engine dinâmico (produção para topologias exóticas). Coberto por
> `golden_vectors` (A2 Dynamic), `a2_heap_audit` e `model_test`. Fazer **após** NF-01 (reusa o
> padrão de propagação de `M`).

---

## NF-03 — [MÉDIA] Regressão de tamanho de arquivo no A2 (refatora-rust): `dynamic.rs` agora é o maior do projeto (959 linhas)

### NF-03 — Localização / Evidência

Levantamento atual (produção, excluindo `_test.rs`):

| Arquivo                            | Linhas  | Δ vs. F-05 original                  |
| ---------------------------------- | ------- | ------------------------------------ |
| `src/models/a2/model/dynamic.rs`   | **959** | 835 → 959 (**+124**)                 |
| `src/models/a2/model/mod.rs`       | **726** | 656 → 726 (**+70**)                  |
| `src/models/a2/grouped_conv1d.rs`  | 725     | ~igual                               |
| `src/math/common/avx2_impl.rs`     | 701     | cresceu com 8x/16x                   |
| `src/dsp/resampler.rs`             | 565     | —                                    |
| `src/math/common/traits.rs`        | 545     | +8x/16x (aceitável: é a *interface*) |
| `src/models/a2/conv1d_ch3/simd.rs` | 529     | —                                    |

A decomposição funcional das `Task 1.2/1.3` (sub-funções `#[inline(always)]`) foi feita, mas
**não reduziu o tamanho físico** dos arquivos — apenas reorganizou internamente. A integração
F-09 (`Task 3.3`) somou código novo de dispatch de largura, **inflando** `dynamic.rs` e `mod.rs`
acima do alvo modular do `refatora-rust` (arquivos pequenos, atômicos, coesos).

### NF-03 — Proposta de solução (sem alterar lógica)

Dividir `a2/model/dynamic.rs` em submódulos sob `a2/model/dynamic/`:

- `dynamic/mod.rs` — struct `WaveNetA2Dyn` + `new`/acessores.
- `dynamic/build.rs` — `set_weights()` e validações de pesos (rotina grande).
- `dynamic/process.rs` — `process()`, `rechannel_prescale`, `advance_head_ring`,
  `layer_forward_dispatch`, `head_finalize`.
- `dynamic/prewarm.rs` — `prewarm` + `a2_prewarm_common`.

Análogo para `a2/model/mod.rs` (separar `process`/`set_weights` do `WaveNetA2` estático).
Também avaliar `grouped_conv1d.rs` (725) e `conv1d_ch3/simd.rs` (529).

> **Risco:** Baixo (movimentação estrutural). Validar com `utils/lints.sh` + `utils/tests-quick.sh`
> (exigência da skill). Combina naturalmente com a refatoração de monomorfização do NF-02.

---

## NF-04 — [MÉDIA] Tiling dual-frame **desabilitado** para interleaving 8/16-wide (CH8/CH16) — otimização perdida

### NF-04 — Localização

- `src/models/wavenet/conv1d_dual.rs:40-58` — `process_dual_frame_with_mixin<M>`: se
  `select_interleave_width(OUT) != 4`, faz **fallback para duas chamadas single-frame**.
- `src/models/wavenet/conv1d_dyn_dual.rs` — mesmo padrão (per `Task 3.3`).

### NF-04 — Evidência

O *Temporal Tiling* (processar 2 frames simultâneos reutilizando os pesos carregados em
registradores) é descrito no próprio cabeçalho de `conv1d_dual.rs:4-7` como a razão de ser do
módulo. Após a integração F-09, esse caminho só permanece ativo para `interleave_width == 4`.
Para WaveNet **CH16** (16-wide) e A2 **CH8** (8-wide) — justamente os modelos mais pesados — o
tiling é **contornado**, recaindo em single-frame. Não existe kernel dual-frame 8x/16x
(`dot_product_8x_f32_dual` / `_16x_f32_dual` não estão no trait `SimdMath`).

O benchmark mostra que 16-wide single-frame ainda supera 4-wide dual-frame (CH16 melhorou),
mas isso indica que o **ótimo combinado** (16-wide **com** tiling dual-frame) está inexplorado.

### NF-04 — Proposta de solução

1. Adicionar ao trait `SimdMath` os kernels `dot_product_8x_f32_dual` e `dot_product_16x_f32_dual`
   (2 frames, reutilizando os loads de `weights[i]` para `state_f0[i]` e `state_f1[i]`).
2. Reativar `process_dual_frame_with_mixin` para 8/16-wide usando esses kernels.
3. No baseline AVX2, `dot_product_16x_f32_dual` decompõe em 2× `dot_product_8x_f32_dual_avx2`
   (coerente com NF-01).
4. Medir vs. single-frame 16-wide; **só incorporar se houver ganho** (mesma disciplina da `Task 3.4`).

> **Risco:** Médio (novo kernel SIMD + caminho dual). Gate por `golden_vectors` + `cpp_parity`.
> Estritamente opt-in até comprovação. **Depende de NF-01** (padrão de decomposição AVX2).

---

## NF-05 — [BAIXA] `refatora-rust`: kernels `_scalar` usados como produção e ausência de kernel AVX2 16x explícito

### NF-05 — Localização / Evidência

- `src/math/common/scalar_ref/dot.rs::dot_product_16x_f32_scalar` é simultaneamente (a) oráculo
  de paridade em testes e (b) **kernel de produção** do `Avx2Math::dot_product_16x_f32` (NF-01).
  O nome `_scalar` num caminho quente de produção é enganoso e dificulta auditoria.
- `gemm/dot_16x/` não possui arquivo `dot_f32_avx2.rs` (só `avx512` e `scalar`), quebrando a
  simetria de `dot_4x/` e `dot_8x/` (que têm kernels AVX2 explícitos).

### NF-05 — Proposta de solução

- Criar `gemm/dot_16x/dot_f32_avx2.rs` com o kernel explícito (ou a decomposição 2×8x do NF-01),
  restaurando a simetria `dot_4x` ⟷ `dot_8x` ⟷ `dot_16x`.
- Manter `_scalar` estritamente como referência/oráculo (`#[cfg(test)]` ou doc explícita).

> **Risco:** Nenhum (organização). Subsume-se ao NF-01.

---

## NF-06 — [INFO] Ruído em microbenchmarks `dot_8x` + comentários `SAFETY` genéricos remanescentes (F-10 parcial)

### NF-06 — Evidência

- **Microbenchmarks:** `dot_8x_f32/avx2_16` acusou *"regressed +21%"* e `avx2_256` *"+11,6%"*
  (`phase5-benchmarks.log:1011,1047`). São medições na casa de **4–76 ns**, dominadas por ruído
  térmico/variância entre execuções e pelo overhead fixo de redução horizontal em `state`
  pequeno. **Não** há regressão de inferência correspondente (WaveNet/A2/LSTM melhoraram). Ação:
  apenas **re-baselinar** o Criterion (`cargo bench -- --save-baseline`) e, opcionalmente,
  documentar que `state < 32` é dominado por overhead (esperado).
- **F-10 residual:** comentários `// SAFETY: Inner safety guarantees are upheld by caller
  invariants or the execution environment.` ainda aparecem genéricos em `traits.rs:212,230,445,459,474,484`.
  Substituir por precondições concretas (tamanhos/alinhamento) nos métodos f32-native (incluindo
  os novos `dot_product_{8,16}x_f32`, cuja doc só diz *"Buffers must be valid"*).

> **Risco:** Nenhum. Higiene de bench + documentação.

---

## Epics — Rodada 2

> Sequenciados para que o padrão estabelecido por NF-01 (decomposição AVX2 explícita) e por
> NF-02 (monomorfização no topo) seja reaproveitado pelos demais. Rede de segurança:
> `golden_vectors` + `cpp_parity` + `a2_heap_audit` + `utils/tests-long.sh`.

### EPIC E — Maximização AVX2 (x86-64-v3) no caminho de convolução largo [CRÍTICO] [DONE]

**Findings:** NF-01 (núcleo), NF-04, NF-05.
**Objetivo:** Garantir que **todo** o caminho de convolução extraia o máximo do baseline AVX2
por construção explícita (não por auto-vetorização frágil), e recuperar o tiling dual-frame
para 8/16-wide.
**Sequência:** NF-01 (kernel AVX2 16x = 2×8x) → NF-05 (simetria `dot_16x/dot_f32_avx2.rs`) →
NF-04 (kernels dual-frame 8x/16x, opt-in por benchmark).
**Critério de aceite:** `dot_16x`/`golden_vectors`/`cpp_parity` verdes; `inference_bench`
sem regressão e com ganho em `in_ch ≥ 32`; zero kernel `_scalar` no caminho de produção AVX2.
**Risco:** Médio (kernels SIMD) — **maior atenção**, pois é o coração da diretriz x86-64-v3.

### EPIC F — Monomorfização do A2 Dinâmico e Higiene Estrutural

**Findings:** NF-02 (núcleo), NF-03, NF-06.
**Objetivo:** Eliminar o dispatch ISA por-frame do `WaveNetA2Dyn`, dividir os arquivos A2
inflados e fechar pendências de bench/documentação.
**Sequência:** NF-02 (monomorfizar `WaveNetA2Dyn` via `dispatch_simd!`) → NF-03 (split
`a2/model/dynamic/` e `mod.rs`) → NF-06 (re-baseline bench + `SAFETY` específicos).
**Critério de aceite:** zero `is_x86_feature_detected!` em `conv1d_dispatch.rs`;
arquivos de produção A2 < ~500 linhas; `utils/lints.sh` + `utils/tests-quick.sh` limpos.
**Risco:** Médio (engine dinâmico de produção) — depende do padrão de `M` do EPIC E.

---

> **Observação metodológica:** Nenhuma das transgressões `is_x86_feature_detected!("avx2"/"fma"/"bmi2")`
> sobrevive na base (F-04 confirmado limpo). As 70 ocorrências remanescentes de
> `is_x86_feature_detected!` referenciam **somente** ISAs acima do baseline (`avx512*`) — uso
> legítimo de multiversioning. O achado de conformidade x86-64-v3 desta rodada **não** é uma
> transgressão literal, e sim a **omissão de um kernel AVX2 explícito** (NF-01), que viola o
> espírito da diretriz "tirar o máximo de x86-64-v3 por padrão".
