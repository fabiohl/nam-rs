<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints.md — Backlog Técnico NAM-rs

## Épico H — Micro-Otimizações e Refinamento de Código

---

### `TH1` — Cabeçalho SPDX nos Arquivos `math/` Remanescentes

**Arquivos:** `src/math/fastmath.rs`, `src/math/simd/fallback.rs`
**Prioridade:** Baixa (Higiene)

Estes 2 arquivos (além dos 2 já tratados em TF1/TF2) são os únicos do projeto
inteiro que não possuem o cabeçalho `SPDX-License-Identifier`. Total de 4 arquivos
a resolver (2 nesta tarefa, 2 já cobertos por TF1 e TF2).

**Ação:**

1. Adicionar ao topo de cada arquivo:

   ```rust
   // SPDX-License-Identifier: MIT OR Apache-2.0
   // Copyright (c) 2026 Fábio Henrique de Lima Silva.
   ```

2. Não alterar nenhuma outra linha.

**Critério de Aceite:** `grep -rL 'SPDX-License-Identifier' src/ --include='*.rs'`
retorna vazio.

---

### `TH2` — Atualizar Referência do Proptest de Sigmoid para Ground Truth Nativo

**Arquivos:** `tests/proptest_math.rs` (linha ~85), `src/math/fastmath_test.rs`
**Prioridade:** Baixa (Qualidade de Testes)

A referência de ground truth no proptest de sigmoid usa a identidade
`0.5 * (1.0 + (val * 0.5).tanh())` — equivalente matemático mas que introduz
uma camada extra de aproximação. Com a implementação direta via `exp` (TA6),
o ground truth mais preciso é `1.0f32 / (1.0 + (-val).exp())`.

**Ação:**

1. Atualizar `std_sigmoid` em `proptest_math.rs` para
   `|val: f32| 1.0f32 / (1.0 + (-val).exp())`.
2. Ajustar threshold para `2e-5` (reflete a qualidade real da implementação).
3. Idem para `fastmath_test.rs` (`test_simd_fastmath_sigmoid_mse`).

**Critério de Aceite:** Testes passam com threshold `2e-5` e referência `exp`-nativa.

---

### `TH3` — Inline `accumulate_head` e `tanh_and_accumulate_block` no AVX2

**Arquivos:** `src/math/simd/avx2.rs` (linhas ~778-785 — delegates para fallback)
**Prioridade:** Média (Performance)

Atualmente, `Avx2Math::accumulate_head` e `Avx2Math::tanh_and_accumulate_block`
delegam para implementações fallback escalares. Como essas funções operam sobre
vetores de 16 floats (CH=16), são candidatas naturais a vetorização AVX2 com
ganho direto no hot-path mais quente do WaveNet (`process_block_internal`).

**Ação:**

1. Implementar `accumulate_head_avx2`: loop sobre `_mm256_loadu_ps` + `_mm256_add_ps` +
   `_mm256_storeu_ps` para acumular `src` em `dest` de 8 em 8 floats.
2. Implementar `tanh_and_accumulate_block_avx2`: para cada chunk de CH floats, aplicar
   `simd_tanh` in-place no `block` e somar ao `head_input` via `_mm256_add_ps`.
3. Substituir as delegações em `Avx2Math`:
   - `accumulate_head` → `accumulate_head_avx2`
   - `tanh_and_accumulate_block` → `tanh_and_accumulate_block_avx2`
4. Manter o fallback escalar inalterado para arquiteturas não-AVX2.

**Critério de Aceite:**

- Benchmark `WaveNet_Standard_CH16_64samp_48kHz` mostra melhoria mensurável
  (estimativa: −2% a −5% do budget de "Act & Head" que é ~15%).
- Paridade numérica com golden vectors (MSE < 1e-10).
- Todos os proptest passam.

---

### `TH4` — Gated Activation AVX2 Nativa (Substituir Fallback Escalar)

**Arquivos:** `src/math/simd/avx2.rs` (linha ~788-794 — delegate para fallback)
**Prioridade:** Média (Performance — A2 Readiness)

A `gated_activation_and_accumulate_block` é a implementação futura para modelos A2
que usam `sigmoid(x) * tanh(x)` em vez de apenas `tanh(x)`. Atualmente delega para
um fallback escalar. Quando modelos A2 forem ativados, esta será uma das funções
mais chamadas no hot-path.

**Ação:**

1. Implementar `gated_activation_and_accumulate_block_avx2` usando:
   - `simd_tanh` para a metade inferior do bloco.
   - `simd_sigmoid_avx2` para a metade superior.
   - Multiplicação vetorial + acumulação no `head_input`.
2. Substituir a delegação em `Avx2Math`.
3. Idem para `Avx512Math` usando `simd_tanh_avx512` e `simd_sigmoid_avx512`.

**Critério de Aceite:** Benchmark isolado mostra >3x speedup vs fallback escalar.
Testes de paridade passam.

---

### `TH5` — `fused_gemm_residual_batch_avx2`: Frame Batching (4-Frame Tiling)

**Arquivos:** `src/math/simd/avx2.rs` (linhas ~624-674)
**Prioridade:** Baixa (Performance)

O kernel `fused_gemm_residual_batch_avx2` processa 1 frame por vez no loop interno
(ao contrário de `fused_add_gemm_batch_avx2` que usa tiling de 4 frames para maximizar
reuso de pesos nos registradores). Uniformizar a estratégia de batching pode reduzir
o budget do estágio "1x1 & Residual" (~25% do total).

**Ação:**

1. Adaptar o loop interno de `fused_gemm_residual_batch_avx2` para processar
   4 frames simultâneos (carregar 4 acumuladores de residual, 4 broadcasts de input,
   1 load de pesos compartilhado).
2. Manter o loop escalar de tail para frames restantes (< 4).
3. Replicar para AVX-512.

**Critério de Aceite:** Benchmark WaveNet mostra melhoria no estágio "1x1 & Residual".
Paridade com golden vectors.

### `TH6` — Sincronizar `docs/architecture.md` com Mudanças Estruturais

**Arquivos:** `docs/architecture.md`
**Prioridade:** Média (pós-conclusão dos épicos F e H)

Se TF1/TF2 mudarem a estrutura de `src/math/simd/` e TH1 mudar `src/models/`,
a tabela de módulos (Seção 4) e a descrição do SIMD (Seção 2) devem ser atualizadas.

**Ação:**

1. Atualizar a tabela "Estrutura de Módulos" para refletir submódulos `avx2/` e `avx512/`.
2. Adicionar `wavenet_ops` na descrição do módulo `models/` se TH1 for executado.
3. Ajustar a seção de SIMD se nomes de arquivos ou APIs mudarem.

**Critério de Aceite:** Documentação reflete com precisão a estrutura real do código.

---

## Épico Z — Teste de Estabilidade de Longa Duração (Soak Test)

**Impacto:** Confiança na estabilidade numérica e operacional para operação contínua
(8-10 horas) — detecção de drift, leaks de estado, overflow de contadores.

> **Nota:** É executado manualmente sob demanda, fora do CI/CD. O objetivo é estressar os **algoritmos**, não a máquina.

### `TZ1` — Soak Test: Suíte de Estabilidade Numérica

**Arquivos:** `tests/soak_test.rs` (novo)
**Prioridade:** Alta (executado manualmente, não no CI)

**Ação:**

1. Criar `tests/soak_test.rs` com os seguintes cenários:
   - **WaveNet Silence Soak:** Processa 10M+ frames de silêncio (0.0) por um
     `WaveNetModel<16, 3, 8>`. Verifica que a saída permanece boundada `[-2.0, 2.0]`.
   - **WaveNet Noise Soak:** Processa 10M+ frames de ruído branco (seeded PRNG
     determinístico) por `WaveNetModel`. Verifica ausência de NaN, ±Inf, divergência.
   - **LSTM Silence Soak:** Processa 10M+ frames de silêncio por `LstmModel2<16>`.
     Verifica que `cell_state` e `hidden_state` não divergem.
   - **LSTM Noise Soak:** Processa 10M+ frames de ruído por `LstmModel2<16>`.
     Verifica estabilidade.
   - **Resampler Drift Soak:** Processa 50M+ amostras (equivalente a ~17 minutos
     a 48kHz) com taxas não-padrão (22050→48000, 96000→48000). Verifica que
     `phase_accum` permanece ≥ 0 e que não há acumulação de erro aritmético.
   - **VirtualRingBuffer Long Run:** Executa 100M+ advance/write cycles no
     `VirtualRingBuffer` e verifica integridade da fronteira mapeada.
   - **Gate FSM Endurance:** Processa 10M+ alternâncias rápidas Open↔Closed↔FadingIn
     para verificar que o `ramp_counter` e `hold_counter` não apresentam overflow.
2. Marcar todos os testes com `#[ignore]` para que não rodem no CI rápido.
3. Adicionar ao final de cada teste um `println!` com as métricas coletadas
   (tempo total, frames processados, min/max saída) para análise manual.

**Critério de Aceite:** 10M+ frames sem panic, NaN, ±Inf ou divergência.
Testes ignorados por padrão. Rodam sob `cargo test -- --ignored --nocapture`.

---

### `TZ2` — Linha de Comando para Disparo Manual do Soak Test

**Arquivos:** `utils/soak-test.sh` (novo)
**Prioridade:** Alta

**Ação:**

1. Criar script shell `utils/soak-test.sh` que:
   - Imprime um aviso de que a bateria pode durar horas.
   - Executa `cargo test --release -- --ignored --nocapture --test-threads=1 2>&1 | tee soak-test.log`.
   - `--release` é mandatório para que os kernels SIMD sejam compilados com otimizações.
   - `--test-threads=1` garante que os testes não competem por recursos de CPU.
   - O log é salvo em `soak-test.log` na raiz do projeto para análise posterior.
   - Ao final, imprime um resumo com exit code.
2. Adicionar `soak-test.log` ao `.gitignore`.
3. Adicionar o cabeçalho de copyright no script.

**Critério de Aceite:** `bash utils/soak-test.sh` dispara a suíte corretamente.
O log é gravado. O script é idempotente.

---

### `TZ3` — Benchmarks Longos para Avaliação de Desempenho Fora de Aleatoriedades

**Arquivos:** `benches/inference_bench.rs` (modificação)
**Prioridade:** Média

Os benchmarks atuais do Criterion usam blocos curtos (64 amostras) com medição
estatística rápida (~50 iterações). Para modelos com variância alta (jitter de cache,
TLB misses em VRB, rewind amortizado), uma medição prolongada com blocos maiores
pode revelar o desempenho *real* em operação contínua.

**Ação:**

1. Adicionar grupo de benchmarks `Long_Run_*` no `inference_bench.rs`:
   - `Long_WaveNet_Standard_CH16_4096samp`: Processa 4096 amostras por iteração
     (equivalente a ~85ms de áudio). `measurement_time(30)`, `sample_size(100)`.
   - `Long_LSTM_2x16_4096samp`: Idem para LSTM 2-camadas.
   - `Long_Resampler_44100_to_48000_4096samp`: Idem para resampler com bloco grande.
2. Anotar estes benchmarks com `#[cfg(feature = "long_bench")]` para que não rodem
   na bateria rápida padrão (ativados via `cargo bench --features long_bench`).
3. Adicionar `long_bench = []` no `[features]` do `Cargo.toml`.

**Critério de Aceite:** Benchmarks longos rodam sob `cargo bench --features long_bench`.
Resultados gravados no Criterion. Sem regressão nos benchmarks curtos existentes.

**Nota do Demandante:** Eles não deve ser disparada via `cargo bench` simples.
Aciona-lo de forma similar ao `utils/soak-test.sh`.
