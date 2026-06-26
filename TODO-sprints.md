<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints (NAM-rs)

Este documento contém o planejamento de sprints e tarefas técnicas estruturadas para o desenvolvimento do NAM-rs, garantindo paridade de desempenho e investigação rigorosa dirigida por benchmarks em relação ao NeuralAmpModelerCore v0.5.4 para o Épico G.

---

## Sprint 6: Épico G — "Benchmark Kernels GEMV Especializados" (F4)

**Escopo:** Realizar investigação dirigida por benchmarks para avaliar se a introdução de micro-kernels GEMV monomorfizados/especializados por dimensão (especificamente `1×4`, `4×4`, `4×6`, `8×4`, `8×6`, `8×8`) resulta em um ganho de desempenho superior a 5% em Rust (visando a arquitetura `x86-64-v3` nativa do projeto) em comparação com o kernel genérico atual de AVX2/FMA. Caso o ganho seja verificado, implementar e integrar tais kernels.
**Objetivo de Paridade:** Investigar a paridade de otimização em relação às especializações introduzidas no core do C++ do NeuralAmpModelerCore v0.5.4 (onde kernels unrolled e macros como `NAM_RESTRICT` são empregados).
**Estimativa:** 1 sprint.
**Risco Geral:** 🟢 Baixo — Abordagem estritamente baseada em medição (`benchmark-driven`). Não há risco para as funcionalidades existentes, pois a substituição de kernels ou a especialização só será ativada se comprovada estatisticamente pelos benchmarks, com testes unitários rigorosos de paridade.

---

### Tarefa 1. [BENCH] Micro-benchmarks de Kernels GEMV (F4)

- **Status:** `[ ]` **Não iniciada**
- **Arquivos Alvo:**
  - [`benches/gemv_bench.rs`](file:///home/fabio/nam-rs/benches/gemv_bench.rs) (Novo arquivo)
- **Descrição:**
  - Criar um suite de micro-benchmarks utilizando `criterion` para isolar a medição de desempenho dos kernels GEMV de f16c (e/ou f32 correspondente).
  - Configurar cenários de medição para as dimensões específicas do Épico G:
    - Out: 1, In: 4 (1×4)
    - Out: 4, In: 4 (4×4)
    - Out: 4, In: 6 (4×6)
    - Out: 8, In: 4 (8×4)
    - Out: 8, In: 6 (8×6)
    - Out: 8, In: 8 (8×8)
  - Medir cada uma destas dimensões com:
    1. O kernel genérico atual do NAM-rs (`fused_add_gemv_avx2`).
    2. Protótipos de kernels especializados escritos via unrolling estático/const generics.
- **Risco:** Baixo. Requer apenas isolar a execução matemática de multiplicação matriz-vetor sem alterar caminhos de áudio de produção.

---

### Tarefa 2. [DECISION] Análise de Desempenho e Tomada de Decisão (F4)

- **Status:** `[ ]` **Não iniciada**
- **Arquivos Alvo:**
  - N/A (Relatório textual a ser gerado/arquivado ou incluído em `TODO-findings.md`)
- **Descrição:**
  - Executar a suíte de benchmarks e analisar as médias de execução comparando a implementação genérica vs. especializada.
  - Aplicar o critério de aceitação de ganho de desempenho:
    - **Se ganho > 5%** em alguma dimensão crítica: Prosseguir para a implementação definitiva e integração (Tarefa 3).
    - **Se ganho <= 5%**: Abortar a alteração do código de produção, documentar a decisão e fechar a finding `F4` com a conclusão de que a otimização de loops do LLVM já atinge o pico de desempenho para o baseline `x86-64-v3` do NAM-rs.
- **Risco:** Baixo. Apenas análise de dados.

---

### Tarefa 3. [MODEL/MATH] Implementação de Kernels GEMV Especializados (F4)

- **Status:** `[ ]` **Não iniciada**
- **Arquivos Alvo:**
  - [`src/math/gemm/gemv/f16_avx2.rs`](file:///home/fabio/nam-rs/src/math/gemm/gemv/f16_avx2.rs)
  - [`src/math/gemm/gemv/mod.rs`](file:///home/fabio/nam-rs/src/math/gemm/gemv/mod.rs)
- **Descrição:**
  - (Condicional à aprovação da Tarefa 2) Implementar kernels especializados em Assembly inline ou intrinsics SIMD otimizadas (ex.: unrolling completo, bias fusionado no loop principal de processamento) para as dimensões vitoriosas.
  - Integrar os novos kernels especializados no dispatch estático da trait `SimdMath` ou diretamente na lógica de convolução/GEMV do NAM-rs, mantendo a retrocompatibilidade com tamanhos arbitrários.
- **Risco:** Médio. Alterações em código do hot-path matemático requerem cuidado extremo com alinhamento e corretude.

---

### Tarefa 4. [TEST] Testes Unitários de Consistência e Paridade Matemática (F4)

- **Status:** `[ ]` **Não iniciada**
- **Arquivos Alvo:**
  - [`src/math/gemm/gemv/gemv_test.rs`](file:///home/fabio/nam-rs/src/math/gemm/gemv/gemv_test.rs)
- **Descrição:**
  - Implementar testes unitários exaustivos comparando a saída de cada novo kernel especializado contra a referência de precisão (ou contra o kernel genérico existente).
  - Cobrir casos com e sem bias (`do_bias: true` / `false`), além de testar condições de contorno de floats (ex.: valores pequenos, denormais, etc.).
- **Risco:** Baixo. Essencial para garantir a ausência de regressões matemáticas no áudio.

---

### Tarefa 5. [QA] Validação de Lints e Integração de Qualidade (F4)

- **Status:** `[ ]` **Não iniciada**
- **Arquivos Alvo:**
  - Repositório Geral / Scripts de QA
- **Descrição:**
  - Executar a suíte de qualidade do NAM-rs (`utils/lints.sh` e `utils/tests-quick.sh`) para garantir que nenhuma nova otimização de baixo nível quebre diretrizes de compilação ou cause warnings de clippy no projeto.
- **Risco:** Baixo.
