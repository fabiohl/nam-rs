<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil (EPIC E)

Este documento descreve o detalhamento ágil de sprints e tarefas técnicas para a execução do **EPIC E — Maximização AVX2 (x86-64-v3) no caminho de convolução largo [CRÍTICO]**, com base nas descobertas (Findings) de [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

---

## Estrutura de Execução: EPIC E

**Objetivo:** Garantir que todo o caminho de convolução largo extraia a máxima performance do baseline `x86-64-v3` (AVX2/FMA) por construção explícita de hardware (eliminando dependências frágeis de auto-vetorização escalar) e recuperando o *temporal tiling* (dual-frame) para 8-wide e 16-wide.
**Caminho Crítico:** Sprint E.1 (Kernel AVX2 16x explícito) → Sprint E.2 (Kernels Dual-Frame) → Sprint E.3 (Ativação e QA).

---

### SPRINT E.1 — Redesign e Implementação do Kernel AVX2 16-wide Explícito [DONE]

**Foco:** Resolver o finding **NF-01** (Convolução 16-wide no baseline AVX2 rodando via kernel escalar lento de fallback).

#### [x] Tarefa E.1.1 — Implementar Kernel AVX2 16-wide Explícito (NF-01 / NF-05)

- **Descrição:** Criar o arquivo [dot_f32_avx2.rs](file:///home/fabio/nam-rs/src/math/gemm/dot_16x/dot_f32_avx2.rs).
- **Detalhes Técnicos:** Implementar `dot_product_16x_f32_avx2(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16]`.
  - Usar duas leituras `__m256` (`w_lo` e `w_hi`) para cada linha de pesos de tamanho 16.
  - Broadcast de `state[i]` uma única vez por iteração do loop para alimentar as FMAs de ambas as metades de pesos.
  - Desdobrar o loop sobre `state` por 4 (usando acumuladores independentes `acc_lo0..3` e `acc_hi0..3`) para quebrar a latência de FMA.
  - Finalizar com redução horizontal e armazenamento em `[f32; 16]`.
- **Risco:** Baixo-Médio. Requer atenção à corretude dos índices de leitura/acumuladores e alinhamento de memória.
- **Referência:** [TODO-findings.md: NF-01, NF-05](file:///home/fabio/nam-rs/TODO-findings.md#NF-01).

#### [x] Tarefa E.1.2 — Refatorar o Dispatch de Traits em `avx2_impl.rs` (NF-01) ✅ Concluído: Dispatch de `dot_product_16x_f32` em `avx2_impl.rs:96-100` agora chama `dot_16x::dot_product_16x_f32_avx2` (kernel AVX2 explícito), substituindo o fallback escalar

- **Descrição:** Alterar `Avx2Math::dot_product_16x_f32` em [avx2_impl.rs](file:///home/fabio/nam-rs/src/math/common/avx2_impl.rs#L96).
- **Detalhes Técnicos:** Substituir a chamada para a função de referência scalar pelo novo kernel explícito `crate::math::gemm::dot_16x::dot_product_16x_f32_avx2`.
- **Referência:** [TODO-findings.md: NF-01](file:///home/fabio/nam-rs/TODO-findings.md#NF-01).

#### [x] Tarefa E.1.3 — Modificar a Estrutura do Módulo `dot_16x` (NF-05) ✅ Concluído: `mod.rs` já expõe `pub mod dot_f32_avx2;` e re-exporta com `pub use dot_f32_avx2::*;`. Módulo integrado ao dispatch em `avx2_impl.rs:98`

- **Descrição:** Atualizar o arquivo [mod.rs](file:///home/fabio/nam-rs/src/math/gemm/dot_16x/mod.rs) da pasta `dot_16x`.
- **Detalhes Técnicos:** Expor o novo módulo `pub mod dot_f32_avx2;` e re-exportar suas funções públicas.

#### [x] Tarefa E.1.4 — Higiene do Kernel `_scalar` de Referência (NF-05) ✅ Concluído: `dot_16x/scalar.rs` e `scalar_ref/dot.rs` (`dot_product_16x_f32_scalar`) verificados como usados exclusivamente em testes/benchmarks (zero uso em dispatch de produção). Documentação atualizada nos dois módulos e no re-export de `gemm/mod.rs`. Adicionados testes de paridade AVX2 vs scalar (`test_dot_16x_f32_avx2_vs_scalar`, `test_dot_16x_f32_avx2_stress`)

- **Descrição:** Garantir que o oráculo scalar [scalar.rs](file:///home/fabio/nam-rs/src/math/gemm/dot_16x/scalar.rs) e [dot.rs](file:///home/fabio/nam-rs/src/math/common/scalar_ref/dot.rs) sejam usados exclusivamente em suites de teste/validação de paridade, documentando apropriadamente.

---

### SPRINT E.2 — Desenvolvimento de Kernels Dual-Frame (Tiling) 8-wide e 16-wide

**Foco:** Resolver o finding **NF-04** (Tiling temporal inativo para os modelos mais pesados WaveNet CH16 e A2 CH8).

#### [x] Tarefa E.2.1 — Adicionar Assinaturas de Trait em `traits.rs` (NF-04) ✅ Concluído: `dot_product_8x_f32_dual` e `dot_product_16x_f32_dual` adicionados ao trait `SimdMath` com default `unimplemented!()` bodies

- **Descrição:** Modificar o trait `SimdMath` em [traits.rs](file:///home/fabio/nam-rs/src/math/common/traits.rs).
- **Detalhes Técnicos:** Adicionar os métodos:

  ```rust
  unsafe fn dot_product_8x_f32_dual(
      weights: &[[f32; 8]],
      state_f0: &[f32],
      state_f1: &[f32],
  ) -> ([f32; 8], [f32; 8]);

  unsafe fn dot_product_16x_f32_dual(
      weights: &[[f32; 16]],
      state_f0: &[f32],
      state_f1: &[f32],
  ) -> ([f32; 16], [f32; 16]);
  ```

#### [x] Tarefa E.2.2 — Adicionar Referências de Cálculo Escalar em `dot.rs` (NF-04) ✅ Concluído: `dot_product_8x_f32_dual_scalar` e `dot_product_16x_f32_dual_scalar` implementados em `dot.rs:269-370` usando `mul_add` (FMA3). Seguem o mesmo padrão de `dot_product_4x_f32_dual_scalar` existente

- **Descrição:** Implementar `dot_product_8x_f32_dual_scalar` e `dot_product_16x_f32_dual_scalar` em [dot.rs](file:///home/fabio/nam-rs/src/math/common/scalar_ref/dot.rs) usando `mul_add` (FMA).

#### [x] Tarefa E.2.3 — Implementar Kernels Dual-Frame AVX2 (NF-04) ✅ Concluído: `dot_product_8x_f32_dual_avx2` em `dot_8x/dot_f32_avx2.rs` (8 acumuladores independentes, 4-way unrolling), `dot_product_16x_f32_dual_avx2` em `dot_16x/dot_f32_avx2.rs` (16 acumuladores independentes, 4-way unrolling), dispatch em `avx2_impl.rs`. Compila e todos os testes existentes passam.

- **Descrição:** Desenvolver as rotinas SIMD de temporal tiling sob a ISA AVX2.
- **Detalhes Técnicos:**
  - Em `dot_8x/dot_f32_avx2.rs`: criar `dot_product_8x_f32_dual_avx2` carregando `weights[i]` uma vez por linha para registrador `__m256`, multiplicando em duas FMAs separadas pelos respectivos broadcasts de `state_f0[i]` e `state_f1[i]` (acumuladores independentes para cada frame).
  - Em `dot_16x/dot_f32_avx2.rs`: criar `dot_product_16x_f32_dual_avx2` usando o mesmo padrão duplicado de leitura (lo/hi).
  - Atualizar [avx2_impl.rs](file:///home/fabio/nam-rs/src/math/common/avx2_impl.rs) para delegar a esses novos kernels.

#### [ ] Tarefa E.2.4 — Implementar Kernels Dual-Frame AVX-512 (NF-04)

- **Descrição:** Adaptar os traits e macros de execução de ponta (Sapphire Rapids/AVX-512) para manter compatibilidade.
- **Detalhes Técnicos:**
  - Em `dot_16x/dot_f32_avx512.rs`: implementar `dot_product_16x_f32_dual_avx512` usando registradores `__m512`.
  - Atualizar as macros em [base.rs](file:///home/fabio/nam-rs/src/math/common/avx512/gemv/base.rs) e [vnni_bf16.rs](file:///home/fabio/nam-rs/src/math/common/avx512/gemv/vnni_bf16.rs) para delegar os novos métodos.

---

### SPRINT E.3 — Integração nos Caminhos de Convolução e Validação Estrita

**Foco:** Integrar os novos kernels na WaveNet e fechar a rodada com 100% de paridade numérica.

#### [ ] Tarefa E.3.1 — Ativar Tiling Dual-Frame na Convolução WaveNet Estática (NF-04)

- **Descrição:** Modificar a função `process_dual_frame_with_mixin` em [conv1d_dual.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d_dual.rs).
- **Detalhes Técnicos:**
  - Atualizar a guarda inicial `if interleave_width != 4` para permitir a descida.
  - Implementar os branches `match interleave_width` para 8 e 16 usando `M::dot_product_8x_f32_dual` e `M::dot_product_16x_f32_dual` respectivamente, carregando e guardando os acumuladores correspondentes.

#### [ ] Tarefa E.3.2 — Ativar Tiling Dual-Frame na Convolução WaveNet Dinâmica (NF-04)

- **Descrição:** Fazer alteração correspondente em `process_dual` no arquivo [conv1d_dyn_dual.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d_dyn_dual.rs).

#### [ ] Tarefa E.3.3 — Criar Suítes de Testes Unitários de Convolução e Paridade

- **Descrição:** Escrever testes de validação em [dot_8x_test.rs](file:///home/fabio/nam-rs/src/math/gemm/dot_8x/dot_8x_test.rs) e [dot_16x_test.rs](file:///home/fabio/nam-rs/src/math/gemm/dot_16x/dot_16x_test.rs).
- **Detalhes Técnicos:** Certificar que os novos kernels batem bit a bit (ou com tolerância < 2 ULP) contra as implementações escalares correspondentes para múltiplos tamanhos e fracionamentos.

#### [ ] Tarefa E.3.4 — Execução Completa de QA e Profiling / Benchmarking

- **Descrição:** Executar o script unificado [utils/tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh) e rodar benchmarks (`cargo bench --bench inference_bench`) para comprovar o ganho de throughput do temporal tiling nas larguras 8x e 16x.
