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

### TA2 · Fused Conv1d + Mixin no WaveNet

- **Arquivo(s):** `src/models/wavenet.rs` (`WaveNetLayer::process_block_internal`)
- **O quê:** Eliminar o loop escalar `for (o, m) in out_frame.iter_mut().zip(mixin_slice)` na fase linear (PASSO 2). Fundir a soma do Mixin diretamente no acumulador SIMD do `Conv1d::process_single_frame`, evitando load/store intermediário.
- **Impacto estimado:** ~10% de redução no tempo por layer WaveNet.
- **Cuidado:** O Mixin já está pré-calculado em bloco (`process_block`). A fusão deve receber um ponteiro para o slice correto do Mixin no frame `i`.
- [ ] Criar variante `process_single_frame_with_mixin<M>` em `Conv1d`
- [ ] Integrar no `process_block_internal`
- [ ] Manter path BF16 equivalente
- [ ] Benchmark comparativo

### TA3 · Batch 1×1 Projection no WaveNet

- **Arquivo(s):** `src/models/wavenet.rs` (`WaveNetLayer::process_block_internal` PASSO 3)
- **O quê:** Substituir N chamadas per-frame a `one_by_one.process_fused()` por uma única chamada em batch (GEMM sobre buffer contíguo de N×CH). Os pesos da 1×1 Dense são lidos N vezes hoje; em batch, seriam lidos 1×.
- **Impacto estimado:** ~8% de redução no tempo por layer WaveNet.
- **Pré-requisito:** Buffer temporário contíguo na stack (já existe `conv_plus_mixin` com 1024 slots).
- [ ] Implementar `DenseLayer::process_fused_block<M>` (batch GEMV com soma residual)
- [ ] Adaptar PASSO 3 para batch head_update + batch 1×1
- [ ] Benchmark comparativo

### TA4 · LSTM Temporal Interleaving (AVX-512)

- **Arquivo(s):** `src/models/lstm.rs`
- **O quê:** Para modelos com H ≤ 16, processar 2 timesteps simultaneamente em registradores AVX-512 (512 bits = 16 lanes). Cada metade de 256 bits carrega o state de um timestep diferente. As ativações (sigmoid/tanh) operam nas 16 lanes de uma vez, amortizando a latência FMA.
- **Impacto estimado:** ~15% de redução no tempo para LSTM 1x8 e 1x16.
- **Complexidade:** Alta. Requer interleaving cuidadoso do cell_state e hidden_state entre as metades do registrador. Aplicável apenas a AVX-512 (step=16 ≥ 2×H).
- **Pré-requisito:** TA1 (Fused 4-Gate) deve estar implementado primeiro para evitar refatoração dupla.
- [ ] Prototipar `process_sample_avx512_2x` para H=8
- [ ] Generalizar para H=16 via const generics
- [ ] Benchmark comparativo (latência P99 e throughput)

---

## Épico C — Qualidade e Testes (Parcial)

> Apenas a tarefa TC2 foi aprovada para execução imediata.

### TC2 · Fuzz Testing dos Parsers NAMB/JSON

- **Arquivo(s):** `tests/` (novo), `src/loader/namb.rs`, `src/loader/nam_json.rs`
- **O quê:** Integrar `cargo-fuzz` para encontrar panics e crashes em edge cases dos parsers de formato `.nam` (JSON) e `.namb` (binário). Cobrir: headers malformados, tamanhos inválidos, truncamento, valores NaN/Inf nos pesos, versões futuras desconhecidas.
- **Impacto:** Segurança e robustez contra arquivos corrompidos ou adversariais.
- [ ] Criar `fuzz/` directory com targets `fuzz_namb` e `fuzz_nam_json`
- [ ] Seed corpus a partir de modelos válidos existentes
- [ ] Executar e corrigir quaisquer panics encontrados
- [ ] Integrar no CI (opcional: limitar a 60s por target)

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
