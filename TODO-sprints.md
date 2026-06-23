<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Implementação

> **Data:** 2026-06-22
> **Origem dos achados:** [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md)
> **Foco:** EPIC A — Unificação de Dispatch SIMD e Hot-Path WaveNet/A2
> **Skills Associadas:** `planejador-arquiteto`, `implementador`, `pesquisador-inovador`, `revisor-auditor`

---

## 0. Visão Geral do Épico e Gestão de Riscos

O **EPIC A** visa otimizar e unificar o roteamento SIMD na base de código, eliminando branches atômicos de detecção de hardware por chamada no hot-path da convolução da WaveNet/A2 e eliminando o mecanismo de dispatch dinâmico secundário (v-table) no pipeline DSP.

### Matriz de Riscos

| Identificador | Risco                                                   | Severidade | Ação de Mitigação                                                         |
| ------------- | ------------------------------------------------------- | ---------- | ------------------------------------------------------------------------- |
| **R-1**       | Regressão de performance na monomorfização              | Média      | Executar benchmarks (`Long_Run_WaveNet` e `Prewarm_A2`) após cada sprint. |
| **R-2**       | Quebra de paridade matemática (C++ e Golden Vectors)    | Alta       | Executar `tests/cpp_parity.rs` e `tests/golden_vectors.rs` localmente.    |
| **R-3**       | Violação de RT-Safety (ex: alocações ocultas ou panics) | Alta       | Executar validação de heap e soak tests da suíte unificada.               |

---

## Sprint 1 — Monomorfização da Convolução f32 e Eliminação de Dispatch por Chamada (F-01) [DONE]

**Objetivo:** Eliminar branches e funções de detecção de CPU (`is_x86_feature_detected!`) do hot-path interno, propagando o parâmetro de tipo genérico `M: SimdMath` até as folhas da convolução.

### Tarefas Técnicas da Sprint 1

- [x] **T1.1: Adicionar os novos métodos f32 ao trait `SimdMath`**

  - **Descrição:** Declarar os métodos de multiplicação de acumuladores f32 diretamente no trait comum de SIMD.

  - **Assinaturas sugeridas:**

    ```rust
    unsafe fn dot_product_4x_f32(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4];
    unsafe fn dot_product_4x_f32_dual(
        weights: &[[f32; 4]], state_f0: &[f32], state_f1: &[f32]
    ) -> ([f32; 4], [f32; 4]);
    ```

  - **Arquivos envolvidos:**

    - [traits.rs](file:///home/fabio/nam-rs/src/math/common/traits.rs) (Definição dos métodos)
    - [avx2_impl.rs](file:///home/fabio/nam-rs/src/math/common/avx2_impl.rs) (Implementação delegando para kernels AVX2 correspondentes)
    - [base.rs](file:///home/fabio/nam-rs/src/math/common/avx512/gemv/base.rs) (Implementação delegando para kernels AVX-512 na macro `impl_avx512_gemv`)
    - [vnni_bf16.rs](file:///home/fabio/nam-rs/src/math/common/avx512/gemv/vnni_bf16.rs) (Redirecionamento estático na macro `impl_avx512vnni_bf16_gemv`)

  - **Responsável Sugerido:** `implementador`

  - **Risco:** Baixo

- [x] **T1.2: Propagar o parâmetro `<M: SimdMath>` nas estruturas de Convolução**

  - **Descrição:** Tornar as implementações de convolução que consomem `dot_product_4x` e `dot_product_4x_dual` genéricas em `M` e chamar as implementações do trait `M::dot_product_4x_f32` diretamente.
  - **Arquivos envolvidos:**
    - [conv1d.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d.rs) (Generic em `M`, alteração na chamada de processamento de frame)
    - [conv1d_dyn.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d_dyn.rs) (Generic em `M`)
    - [conv1d_dual.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d_dual.rs) (Generic em `M`)
    - [layer.rs](file:///home/fabio/nam-rs/src/models/wavenet/layer.rs) (Propagação de `M` nas chamadas)
  - **Responsável Sugerido:** `implementador` / `pesquisador-inovador`
  - **Risco:** Médio (garantir consistência com chamadores de alto nível)
  - **Nota Pós-Execução (T1.2):** O `Conv1dDyn` é compartilhado entre WaveNet e A2/ConvNet. O A2 (`a2/conv1d_dispatch.rs`) ainda usa detecção runtime de AVX-512 no nível `A2Conv1d` para selecionar `Avx512Math` vs `Avx2Math` — mas o loop interno de convolução (`Conv1dDyn::process_single_frame`) já está monomorfizado. A monomorfização completa do pipeline A2 exigirá envolver `A2Model::process` com `dispatch_simd!` (futuro sprint).

- [x] **T1.3: Limpeza de wrappers legados em `conv_input.rs`**

  - **Descrição:** Remover as funções livres `dot_product_4x` e `dot_product_4x_dual` que usavam `is_x86_feature_detected!`. Se necessário para testes legados, encapsular sob `#[cfg(test)]`.
  - **Arquivos envolvidos:**
    - [conv_input.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv_input.rs)
  - **Responsável Sugerido:** `implementador`
  - **Risco:** Baixo

### Critério de Aceite da Sprint 1

- Compilação bem-sucedida sem warnings.

- Suíte unificada rápida executada via `utils/tests-quick.sh` retornando sucesso (Golden Vectors e paridade C++ preservados).

- Benchmarks `Long_Run_WaveNet` (`benches/inference_bench.rs`) executados antes e depois evidenciando estabilidade ou melhoria no tempo de execução.

---

## Sprint 2 — Eliminação do Dispatch Dinâmico Duplo (V-Table `SimdMathConfig`) (F-02) [DONE]

**Objetivo:** Eliminar o overhead de chamadas indiretas via ponteiro de função (`Mode 3` da macro `dispatch_simd!`) unificando os consumidores do pipeline DSP e ativações no mecanismo estático monomorfizado da `SimdMath`.

### Tarefas Técnicas da Sprint 2

- [x] **T2.1: Migrar estágios do pipeline DSP para monomorfização estática**

  - **Descrição:** Envolver a execução do pipeline de captura em um bloco `dispatch_simd!` de Modo 1 (estático) no nível mais alto, passando `M` genérico para as funções de estágio (`apply_input_stage`, `apply_output_stage`, etc.).
  - **Arquivos envolvidos:**
    - [capture.rs](file:///home/fabio/nam-rs/src/dsp/pipeline/capture.rs) (Adicionar macro dispatch e propagar generic `M`)
    - [input.rs](file:///home/fabio/nam-rs/src/dsp/pipeline/stages/input.rs) (Adequar chamadas para usar `M::apply_gain`, `M::compute_energy`, etc.)
    - [output.rs](file:///home/fabio/nam-rs/src/dsp/pipeline/stages/output.rs) (Monomorfizar chamadas de ganho e proteção)
  - **Responsável Sugerido:** `planejador-arquiteto` / `implementador`
  - **Risco:** Alto (estágio crucial de tempo real)
  - **Nota Pós-Execução (T2.1):** `capture_dsp_pipeline` agora despacha estaticamente via `match SIMD_MATH.instruction_set` e chama `capture_dsp_pipeline_inner::<M>` que propaga `M` para `apply_input_stage_inner::<M>` e `apply_output_stage_inner::<M>`. As funções públicas `apply_input_stage`/`apply_output_stage` foram mantidas como wrappers (gated por `cfg(test, clap-plugin)`) para compatibilidade com o CLAP e testes — elas também despacham estaticamente, mas com um dispatch por estágio (vs único no `capture_dsp_pipeline`). O Modo 3 (`dispatch_simd!` v-table) foi eliminado do hot-path do pipeline; a remoção completa da v-table (`SimdMathConfig`) foi concluída em T2.2.

- [x] **T2.2: Reduzir a struct `SimdMathConfig` e remover ponteiros de função**

  - **Descrição:** Remover os ponteiros de função associados à v-table em `SimdMathConfig`, deixando apenas dados estáticos descritivos (conjunto de instruções, nome, etc.). Ajustar a tabela global `SIMD_MATH` e as rotinas de inicialização/detecção de hardware.
  - **Arquivos envolvidos:**
    - [config.rs](file:///home/fabio/nam-rs/src/math/common/dispatch/config.rs) (Alteração da struct e remoção de campos da v-table)
    - [detect.rs](file:///home/fabio/nam-rs/src/math/common/dispatch/detect.rs) (Eliminação de atribuições de ponteiros em `detect_best_simd()`)
    - [mod.rs](file:///home/fabio/nam-rs/src/math/common/mod.rs) (Remoção do Modo 3 do `dispatch_simd!`)
  - **Responsável Sugerido:** `implementador`
  - **Risco:** Médio
  - **Nota Pós-Execução (T2.2):** `SimdMathConfig` agora contém apenas `instruction_set`, `name` e `is_avx512`. Os 30+ ponteiros de função foram removidos. O `config_table!` foi simplificado para 3 parâmetros (isa, name, avx512). O Modo 3 do `dispatch_simd!` foi convertido de v-table (indireto) para dispatch estático monomorfizado via `match SIMD_MATH.instruction_set` chamando `<Avx2Math>::method()`, `<Avx512Math>::method()` ou `<Avx512VnniBf16Math>::method()`. As 8 funções de ativação que não estavam no trait `SimdMath` (`relu_slice`, `prelu_slice`, `softsign_slice`, `silu_slice`, `hard_tanh_slice`, `hard_swish_slice`, `fast_tanh_slice`, `leaky_hard_tanh_slice`) foram adicionadas ao trait e implementadas em `Avx2Math`, `Avx512Math` e `Avx512VnniBf16Math`. `activations/mod.rs` agora usa `dispatch_simd!` em vez de chamadas diretas à v-table. Zero v-table residual.

### Critério de Aceite da Sprint 2

- Suíte unificada de testes (`utils/tests-quick.sh`) passa completamente sem erros.

- Validação rigorosa de Heap-Audit certificando que a unificação não introduziu alocações em tempo de execução no caminho RT.

- Codebase livre de chamadas dinâmicas redundantes para dispatch matemático.

---

## Sprint 3 — Ajuste de Documentação e Comentários (F-08) [DONE]

**Objetivo:** Sincronizar todos os comentários técnicos da base de código e documentação arquitetural com a nova infraestrutura de dispatch unificada.

### Tarefas Técnicas da Sprint 3

- [x] **T3.1: Atualizar comentários do código fonte**

  - **Descrição:** Corrigir comentários e docstrings que descreviam o comportamento antigo (como o dispatch dinâmico por chamada ou a viabilidade de performance na v-table).
  - **Arquivos envolvidos:**
    - [model.rs](file:///home/fabio/nam-rs/src/models/wavenet/model.rs) (Atualizar explicação da monomorfização)
    - [config.rs](file:///home/fabio/nam-rs/src/math/common/dispatch/config.rs) (Remover comentários sobre o débito técnico resolvido)
    - [conv_input.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv_input.rs) (Revisar docstrings sobre kernels SIMD)
  - **Responsável Sugerido:** `documentador` / `refatora-doc`
  - **Risco:** Mínimo

- [x] **T3.2: Atualizar documentação arquitetural global**

  - **Descrição:** Refletir o design final unificado e o fim do double-dispatch na documentação de design do projeto.
  - **Arquivos envolvidos:**
    - Documentos relevantes em `docs/` (ex: `docs/architecture.md`, caso aplicável).
  - **Responsável Sugerido:** `documentador`
  - **Risco:** Mínimo

### Critério de Aceite da Sprint 3

- Revisão de texto limpa.
- Documentação condizente com a implementação monomorfizada final.
