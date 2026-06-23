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
