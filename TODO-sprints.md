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

### Tarefa 1. [BENCH] Micro-benchmarks de Kernels GEMV (F4) [DONE]

- **Status:** `[x]` **Concluída**
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
- **Conclusão:** Suite criada em `benches/gemv_bench.rs`. Seis kernels especializados unrolled (1×4, 4×4, 4×6, 8×4, 8×6, 8×8) comparados contra `fused_add_gemv_avx2` genérico e `fused_add_gemv_fallback` escalar de referência. Sem alterações no código de produção.

---

### Tarefa 2. [DECISION] Análise de Desempenho e Tomada de Decisão (F4) [DONE]

- **Status:** `[x]` **Concluída**
- **Arquivos Alvo:**
  - [`benches/gemv_bench.rs`](file:///home/fabio/nam-rs/benches/gemv_bench.rs) (buffer overflow corrigido nos kernels 1×4, 4×4, 8×4)
- **Descrição:**
  - Executar a suíte de benchmarks e analisar as médias de execução comparando a implementação genérica vs. especializada.
  - Aplicar o critério de aceitação de ganho de desempenho:
    - **Se ganho > 5%** em alguma dimensão crítica: Prosseguir para a implementação definitiva e integração (Tarefa 3).
    - **Se ganho <= 5%**: Abortar a alteração do código de produção, documentar a decisão e fechar a finding `F4` com a conclusão de que a otimização de loops do LLVM já atinge o pico de desempenho para o baseline `x86-64-v3` do NAM-rs.
- **Risco:** Baixo. Apenas análise de dados.
- **Conclusão:** **APROVADO — Prosseguir para Tarefa 3.** Ganho >5% em **todas** as 6 dimensões testadas:
  - 1×4: +40.0% (15.0ns → 9.0ns)
  - 4×4: +50.4% (23.2ns → 11.5ns)
  - 4×6: +49.5% (28.1ns → 14.2ns)
  - 8×4: +72.9% (46.8ns → 12.7ns)
  - 8×6: +65.5% (62.6ns → 21.6ns)
  - 8×8: +21.2% (15.6ns → 12.3ns)
  - O kernel genérico atinge seu pico na dimensão 8×8 (bloco interno ótimo), mas ainda perde por 21% devido ao overhead de loop/branching. O ganho é massivo nas dimensões não-alinhadas ao bloco de 8 saídas.
  - **Nota:** Identificados e corrigidos buffer overflows nos kernels 1×4, 4×4 e 8×4 onde `_mm256_storeu_ps` escrevia 32 bytes em buffers de 16 bytes (4×f32). A correção usa array temporário `[f32; 8]` com cópia parcial dos primeiros `out_len` elementos. Os mesmos kernels também tem leituras sobredimensionadas (`_mm256_loadu_ps` lendo 32 bytes de buffers de bias/out_frame de 16 bytes) — funcionalmente inócuas pois os lanes 4-7 são descartados, mas tecnicamente UB. Recomenda-se corrigir na Tarefa 3.

---

### Tarefa 3. [MODEL/MATH] Implementação de Kernels GEMV Especializados (F4) [DONE]

- **Status:** `[x]` **Concluída**
- **Arquivos Alvo:**
  - [`src/math/gemm/gemv/f16_avx2.rs`](file:///home/fabio/nam-rs/src/math/gemm/gemv/f16_avx2.rs)
  - [`src/math/gemm/gemv/mod.rs`](file:///home/fabio/nam-rs/src/math/gemm/gemv/mod.rs)
- **Descrição:**
  - (APROVADO pela Tarefa 2) Implementar kernels especializados em Assembly inline ou intrinsics SIMD otimizadas (ex.: unrolling completo, bias fusionado no loop principal de processamento) para **todas** as 6 dimensões (1×4, 4×4, 4×6, 8×4, 8×6, 8×8), dado que todas tiveram ganho >21%.
  - Integrar os novos kernels especializados no dispatch estático da trait `SimdMath` ou diretamente na lógica de convolução/GEMV do NAM-rs, mantendo a retrocompatibilidade com tamanhos arbitrários.
  - **Correções de UB obrigatórias na implementação de produção:**
    - **Store YMM→buffer parcial**: Para dimensões com `out_len < 8` (1×4, 4×4, 4×6, 8×4, 8×6), usar array temporário `[f32; 8]` para store YMM e copiar apenas `out_len` elementos — `_mm256_storeu_ps` escreve 32 bytes e causará heap corruption se o buffer destino tiver menos de 8 f32.
    - **Load YMM de slice parcial**: `_mm256_loadu_ps` em slices `bias`/`out_frame` com menos de 8 elementos é UB (leitura sobredimensionada de 32 bytes sobre alocações de 16–24 bytes). Usar array temporário `[f32; 8]` zero-inicializado, copiar os `out_len` elementos do slice para o temp, e carregar YMM do temp. Ou usar `_mm_loadu_ps` (128-bit) + `_mm256_insertf128_ps` para compor o YMM sem overread.
- **Risco:** Médio. Alterações em código do hot-path matemático requerem cuidado extremo com alinhamento e corretude.
- **Conclusão:** Arquivo `src/math/gemm/gemv/f16_avx2_specialized.rs` criado com 12 kernels (6 dimensões × 2 modos: `fused_add_gemv` e `gemv_overwrite`). Dispatch integrado diretamente em `fused_add_gemv_avx2` e `gemv_overwrite_avx2` via `match (in_len, out_len)` no topo de cada função — sem alterar a trait `SimdMath`, mantendo retrocompatibilidade para tamanhos arbitrários via fallback ao kernel genérico. UB corrigidos: helpers `load_partial_ymm` e `store_partial_ymm` para slices de bias/out_frame < 8 elementos; `load_partial_f16_ymm` para pesos com linhas < 8 f16. 941 testes passam, 0 falhas.

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
