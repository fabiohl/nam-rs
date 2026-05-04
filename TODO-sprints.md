# 🚀 Backlog do Produto e Planejamento de Sprints Técnicas

## Épico A — Otimizações de Throughput SIMD

> Origem: Auditoria `pesquisador-inovador` (2026-05-03).
> Objetivo: Extrair ganhos incrementais de throughput nos kernels de inferência
> LSTM e WaveNet através de fusão de operações e melhor reuso de cache.

### TA1 · Fused 4-Gate GEMV para LSTM [DONE]

- **Arquivo(s):** `src/math/simd.rs`, `src/models/lstm.rs`
- **O quê:** Substituir as 4 chamadas separadas a `gemv_overwrite_*()` (uma por gate I, F, G, O) por uma única passagem sobre `state[]` que compute as 4 gates simultaneamente. O state é lido 4× hoje; com fusão, seria lido 1×.
- **Impacto estimado:** ~20-30% de redução no tempo da `LstmLayer` para H ≤ 24 (state cabe em L1).
- **Macro `define_lstm_process!`:** Deverá receber a nova função `gemv_4gate_*` em vez de chamar `$gemv_overwrite` 4 vezes.
- **Validação:** Regressão numérica (golden vectors) + benchmark Criterion comparativo (antes/depois).
- [x] Implementar `gemv_4gate_avx2` e `gemv_4gate_avx512` em `simd.rs`
- [x] Atualizar macro `define_lstm_process!` para consumir o novo kernel
- [x] Manter fallback escalar para testes
- [x] Benchmark comparativo

### TA2 · Fused Conv1d + Mixin no WaveNet [DONE]

- **Arquivo(s):** `src/models/wavenet.rs` (`WaveNetLayer::process_block_internal`)
- **O quê:** Eliminar o loop escalar `for (o, m) in out_frame.iter_mut().zip(mixin_slice)` na fase linear (PASSO 2). Fundir a soma do Mixin diretamente no acumulador SIMD do `Conv1d::process_single_frame`, evitando load/store intermediário.
- **Impacto estimado:** ~10% de redução no tempo por layer WaveNet.
- **Cuidado:** O Mixin já está pré-calculado em bloco (`process_block`). A fusão deve receber um ponteiro para o slice correto do Mixin no frame `i`.
- [x] Criar variante `process_single_frame_with_mixin<M>` em `Conv1d`
- [x] Integrar no `process_block_internal`
- [x] Manter path BF16 equivalente
- [x] Benchmark comparativo

### TA3 · Batch 1×1 Projection no WaveNet [DONE]

- **Arquivo(s):** `src/models/wavenet.rs` (`WaveNetLayer::process_block_internal` PASSO 3)
- **O quê:** Substituir N chamadas per-frame a `one_by_one.process_fused()` por uma única chamada em batch (GEMM sobre buffer contíguo de N×CH). Os pesos da 1×1 Dense são lidos N vezes hoje; em batch, seriam lidos 1×.
- **Impacto estimado:** ~8% de redução no tempo por layer WaveNet.
- **Pré-requisito:** Buffer temporário contíguo na stack (já existe `conv_plus_mixin` com 1024 slots).
- [x] Implementar `DenseLayer::process_fused_block<M>` (batch GEMV com soma residual)
- [x] Adaptar PASSO 3 para batch head_update + batch 1×1
- [x] Benchmark comparativo

### TA4 · LSTM Temporal Interleaving (AVX-512) [DONE]

- **Arquivo(s):** `src/models/lstm.rs`, `src/math/simd.rs`
- **O quê:** Implementar *Temporal Interleaving* entre camadas para esconder a latência de cálculo do estado recorrente. Processamento paralelo da Camada 1 (amostra $T$) e Camada 2 (amostra $T-1$) utilizando núcleos GEMV fundidos e AVX-512.
- **Impacto estimado:** ~15-20% de redução no tempo para modelos de 2 camadas ($H=8, H=16$).
- **Complexidade:** Alta. Requer sincronização rigorosa de buffers e pipeline "Prime/Drain".
- [x] Implementar pipeline interleaving no `LstmModel2`
- [x] Otimizar kernels `gemv_4gate_avx512` para $H=8$ e $H=16$ (BF16 native)
- [x] Validar paridade numérica via Golden Vectors
- [x] Benchmark comparativo final

---

## Épico C — Qualidade e Testes (Parcial)

> Apenas a tarefa TC2 foi aprovada para execução imediata.

### TC2 · Fuzz Testing dos Parsers NAMB/JSON [DONE]

- **Arquivo(s):** `tests/` (novo), `src/loader/namb.rs`, `src/loader/nam_json.rs`
- **O quê:** Integrar `cargo-fuzz` para encontrar panics e crashes em edge cases dos parsers de formato `.nam` (JSON) e `.namb` (binário). Cobrir: headers malformados, tamanhos inválidos, truncamento, valores NaN/Inf nos pesos, versões futuras desconhecidas.
- **Impacto:** Segurança e robustez contra arquivos corrompidos ou adversariais.
- [x] Criar `fuzz/` directory com targets `fuzz_namb` e `fuzz_nam_json`
- [x] Seed corpus a partir de modelos válidos existentes
- [x] Executar e corrigir quaisquer panics encontrados (Zero crashes em 7M+ execuções)
- [ ] Integrar no CI (opcional: limitar a 60s por target)

---

## Épico E — Otimizações WaveNet (Throughput Microarquitetural)

> Origem: Auditoria `pesquisador-inovador` (2026-05-03, 3ª rodada).
> Objetivo: Extrair ganhos incrementais no pipeline de inferência WaveNet
> através de fusão de operações de memória e otimização de cache hierarchy.
> Foco exclusivo no modelo mais popular (WaveNet Standard CH=16, K=3).

### TE1 · Inversão de Loop Conv1D: Channel-First Tiling [DONE]

- **Arquivo(s):** `src/models/wavenet.rs` (`Conv1d::process_single_frame_internal`)
- **O quê:** Inverter os loops da Conv1D de `for k { for b { FMA } }` para `for b { for k { FMA } }`. Isso mantém os registradores de saída "quentes" nos YMM/ZMM ao processar todos os K taps de um bloco antes de mover para o próximo, reduzindo stores intermediários de `K × OUT/4` para `OUT/4`.
- **Impacto estimado:** ~5-8% nos layers com dilatação baixa (1, 2, 4) onde os dados de entrada estão em L1. Impacto menor em dilatações altas (memory-bound).
- **Cuidado:** Validar que a mudança de ordem dos taps não afeta a acumulação numérica (comutatividade da soma FP — pode haver micro-diferenças em ULP). Manter fallback escalar inalterado para comparação.
- **Validação:** Regressão numérica (golden vectors WaveNet) + benchmark Criterion comparativo.
- [x] Refatorar loop interno em `process_single_frame_internal` (f32 path)
- [x] Refatorar loop interno em `process_single_frame_bf16_internal` (BF16 path)
- [x] Validar paridade numérica
- [x] Benchmark comparativo (Ganho: **~8.8%**)

### TE2 · Prefetch Bidirecional para Dilatações Extremas (256/512) [DONE]

- **Arquivo(s):** `src/models/wavenet.rs` (`Conv1d::process_single_frame_internal`), `src/math/simd.rs`
- **O quê:** Para layers com `dilation ≥ 128`, emitir 2 prefetches escalonados por tap: `T1` (L2) para tap T+2 e `T0` (L1) para tap T+1. Isso cria um pipeline de 2 estágios no cache subsystem que resolve o gap de latência DRAM→L1 (~180 ciclos) ao preparar os dados com 2 taps de antecedência.
- **Impacto estimado:** ~8-12% nos layers de dilatação máxima (512). Efeito global: ~2-3%.
- **Cuidado:** Prefetches são hints sem impacto na corretude. Validar em CPUs com prefetcher agressivo (Zen4) vs conservador.
- [x] Implementar `adaptive_prefetch_2stage_f32` em `simd.rs`
- [x] Integrar no loop da Conv1D para `k + 2 < K`
- [x] Benchmark comparativo em diferentes dilatações

### TE3 · Fusão Tanh + Head Accumulate em Passagem Única [DONE]

- **Arquivo(s):** `src/math/simd.rs` (novo kernel), `src/models/wavenet.rs` (`WaveNetLayer::process_block_internal`)
- **O quê:** Fundir as FASES 2 e 3 do `process_block_internal` em uma única passagem de memória: `activation_tanh_block` seguido de `accumulate_head` lê os mesmos 1024 floats 2 vezes. Um kernel fundido `tanh_and_accumulate_block(conv, head)` aplica tanh, armazena de volta e acumula no head — tudo com dados nos registros YMM/ZMM, sem viagem extra ao L1.
- **Impacto estimado:** ~5-7% no tempo total de `process_block_internal`. Elimina 1 passagem de leitura (4KB por bloco).
- **Macro:** Criar variantes AVX2 (`tanh_and_accumulate_avx2`) e AVX-512 (`tanh_and_accumulate_avx512`), adicionando ao trait `SimdMath`.
- **Validação:** Paridade numérica bit-a-bit (a fusão é determinística).
- [x] Implementar `tanh_and_accumulate_block_avx2` em `simd.rs`
- [x] Implementar variante AVX-512
- [x] Adicionar método ao trait `SimdMath`
- [x] Substituir FASES 2+3 em `process_block_internal`
- [x] Benchmark comparativo

---

## Épico F — Paridade WaveNet Dyn + Fusões Avançadas

> Origem: Auditoria `pesquisador-inovador` (2026-05-03, 4ª rodada).
> Objetivo: Fechar o gap de **240% overhead** entre o path estático (`wavenet.rs`)
> e o dinâmico (`wavenet_dyn.rs`), e aplicar fusões avançadas de kernels SIMD
> que beneficiam TODAS as topologias WaveNet (Standard, Lite, Feather, Nano + Dyn).
> Prioridade: ganhos de 2 dígitos percentuais.

### TF1 · Backport de Otimizações A+E para `wavenet_dyn.rs` ⭐ MAIOR IMPACTO DO PROJETO [DONE]

- **Arquivo(s):** `src/models/wavenet_dyn.rs` (refatoração principal), `src/math/simd.rs` (reuso)
- **O quê:** O path dinâmico estava **3 Épicos atrás** do estático. Backport metódico de 5 otimizações já validadas:
  1. **TA2** — Fused Conv1D+Mixin: Criar `Conv1dDyn::process_block_with_mixin<M>`
  2. **TE1** — Channel-First Tiling: Inverter loop Conv1D Dyn com pre-carregamento de taps
  3. **TE2** — Prefetch 2-stage: Integrar `adaptive_prefetch_2stage_f32` para `dilation ≥ 128`
  4. **TE3** — Tanh+Head Fusion: Substituir `M::tanh_slice` + head accumulate por `M::tanh_and_accumulate_block`
  5. **TA3** — Batch 1×1 Projection: Criar `DenseLayerDyn::process_fused_block<M>`
- **Impacto estimado:** **~40-60%** de redução no tempo do path dinâmico (120 µs → ~55-70 µs)
- [ ] Benchmark comparativo (antes/depois)

### TF2 · Fusão da Ativação Gated em SIMD [DONE]

- **Arquivo(s):** `src/math/simd.rs` (novo kernel), `src/math/fastmath.rs`, `src/models/wavenet_dyn.rs`
- **O quê:** Modelos WaveNet com `gated=true` executam 3 passagens separadas sobre os mesmos dados (`tanh_slice`, `sigmoid_slice`, multiplicação escalar). Fundir em um único kernel `fused_gated_activation_block` que aplica `tanh(z1) × sigmoid(z2)` em uma passagem SIMD, eliminando 2 loads e 1 store intermediário.
- **Impacto estimado:** **~15-25%** no tempo da ativação gated. Ganho global em modelos gated: **~5-8%**.
- **Variantes:** AVX2 (`fused_gated_activation_avx2`) e AVX-512 (`fused_gated_activation_avx512`).
- **Validação:** Paridade numérica bit-a-bit (fusão determinística).
- [ ] Implementar `fused_gated_activation_block_avx2` em `simd.rs`
- [ ] Implementar variante AVX-512
- [ ] Adicionar ao trait `SimdMath` e V-table
- [ ] Integrar em `WaveNetLayerDyn::process_block_internal` (branch gated)
- [ ] Benchmark comparativo com modelo gated

### TF3 · Eliminação do Residual Copy + GEMV Fundido (Todas as Topologias) [DONE]

- **Arquivo(s):** `src/math/simd.rs` (novo kernel), `src/models/wavenet.rs`, `src/models/wavenet_dyn.rs`
- **O quê:** Na FASE 3 do `process_block_internal`, o `copy_from_slice` do layer_buffer (residual) seguido de `process_fused_block` causa 2 passagens de memória. Criar `fused_gemv_residual_batch` que faz `out[i] = residual[i] + Σ(W[j] × in[j]) + bias` em uma única passagem, carregando o residual diretamente no acumulador SIMD.
- **Impacto estimado:** **~10-15%** no tempo da FASE 3. Ganho global end-to-end: **~5-8%**.
- **Risco:** Médio. Exige novo kernel SIMD (variação do `fused_add_gemm_batch`).
- **Abrangência:** Todas as topologias (Standard, Lite, Feather, Nano) + Dyn.
- [ ] Implementar `fused_gemv_residual_batch_avx2` em `simd.rs`
- [ ] Implementar variante AVX-512
- [ ] Adicionar ao trait `SimdMath`
- [ ] Substituir `copy_from_slice` + `process_fused_block` em `wavenet.rs`
- [ ] Substituir equivalente em `wavenet_dyn.rs`
- [ ] Benchmark comparativo (antes/depois)

---

## Observações — Itens Diferidos

> Itens identificados na auditoria `pesquisador-inovador` mas **não priorizados**
> para o momento atual. Mantidos aqui para referência futura.

### Épico B — Modernização e Portabilidade *(diferido)*

- **TB1** · Preparação CLAP Plugin — Separar `pw_host.rs` em trait `AudioHost`.
- **TB2** · Internacionalização (i18n) — Migrar strings PT-BR para constantes EN.
- **TB3** · ARM64/NEON Port — Abstrair `SimdMath` para suportar NEON.
- **Razão:** Não é o momento. Foco atual é otimização de throughput SIMD x86-64.

### Épico D — Arquitetura A2 *(diferido)*

- **TD1** · Scaffold A2 Architecture — Módulo `models/a2.rs` + variante `DynamicModel::A2`.
- **TD2** · A2 Weight Loader — Estender dispatcher para formato A2.
- **Razão:** Não é o momento. Aguardando definição oficial do formato A2 upstream.

### Tarefas C não priorizadas *(diferidas)*

- **TC1** · Benchmark VirtualRingBuffer vs copy_within — Quantificar ganho do mmap espelhado.
- **TC3** · Stress Test de GC Channel Overflow — Validar leak path sob carga extrema.
- **Razão:** Ganho marginal de validação. Podem ser retomadas após o Épico A.

### Tarefas E não priorizadas *(diferidas)*

- **TE4** · Winograd F(2,3) para Conv1D K=3 — Redução teórica de 33% em mults, mas aplicável apenas a layers com dilation=1 (2 de 10). Ganho global ~3-5%. Complexidade alta: exige processamento de 2 frames simultâneos, conflito com semântica causal dilatada. **Custo/benefício desfavorável.**

### Propostas descartadas (pesquisador-inovador, 4ª rodada)

- **Padding CH=12→16 (Lite):** Ganho ~10-15% apenas Conv1D Lite, mas requer padding em loading + aumento de footprint. Complexidade alta para benefício localizado.
- **SSE path para Nano CH=4:** Nano já é ultra-rápido (~5 µs). Ganho absoluto de ~1-2 µs não justifica path SSE separado.
- **Multi-frame pipelining entre layers:** Quebraria cadeia de dependência causal. Incompatível com WaveNet.
