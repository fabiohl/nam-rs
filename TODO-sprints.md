<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints — NAM-rs Backlog de Melhorias

> Gerado pelo painel **Pesquisador-Inovador** após análise profunda do codebase.
> Escopo: Performance, Usabilidade, Limpeza de Código e Bugs.
> Fora de escopo: Arquitetura A2 e suporte CLAP.

---

## Sprint 1 — Performance: WaveNet Dual-Frame no Hot-Path [CONCLUIDO]

Impacto estimado: ~15-25% de redução de ciclos no forward WaveNet
> **Auditoria:** Sprint concluída e validada com sucesso. A implementação do Dual-Frame Tiling tanto na variante Standard quanto Dinâmica mitigou redundâncias de carregamento SIMD. As correções de estabilidade no `Conv1dDyn` (tratamento *gated* via `unwrap_or(0.0)`) e ajustes dos loops em lote garantiram robustez de RT (Zero-Alloc) com throughput aprimorado de ~7-10% confirmado por benchmarks em `inference_bench`.

### T1.1 — Dual-Frame Tiling na Cascata `process_block_internal` do `WaveNetLayerArray` [CONCLUIDO]

- **Arquivo:** `src/models/wavenet.rs` (linhas 958–1064)
- **Problema:** O loop da cascata de camadas em `process_block_internal` invoca `WaveNetLayer::process_block_internal` que, internamente, faz a Conv1D frame-a-frame. Os pesos da Conv1D são carregados da L1 para registradores **a cada frame**, mesmo sendo idênticos entre frames consecutivos.
- **Proposta:** Processar 2 frames por iteração no loop da **Fase 1** (linhas 863-886) usando `process_dual_frame_with_mixin` (que já existe, linhas 353-374). Isso amortiza o custo de carregamento de pesos pela metade, dobrando a eficiência do cache L1 → registradores FMA.
- **Detalhes:**
  - No loop `for i in 0..num_frames` da Fase 1 Linear (linha 863), processar `i, i+1` em pares usando `conv1d.process_dual_frame_with_mixin::<M>()`.
  - Tratar o frame remainder (ímpar) com `process_single_frame_with_mixin` existente.
  - A Fase 2 (`tanh_and_accumulate_block`) e Fase 3 (`process_residual_batch`) já operam em bloco, não precisam de mudança.
- **Risco:** Baixo. O kernel `process_dual_frame_generic` já está implementado e testado. Apenas a orquestração do loop muda.
- **Validação:** `cargo bench` (inference_bench) comparando throughput antes/depois.

### T1.2 — Dual-Frame Tiling no `WaveNetLayerDyn` (Modelo Dinâmico) [CONCLUIDO]

- **Arquivo:** `src/models/wavenet_common.rs` (`WaveNetLayerDyn::process_block_internal`, linhas 520–613)
- **Problema:** Análogo ao T1.1, mas para a variante dinâmica. O `Conv1dDyn::process_block` processa frame-a-frame sem reuso de pesos entre frames adjacentes.
- **Proposta:** Adicionar `Conv1dDyn::process_dual_frame` e usá-lo no loop do `WaveNetLayerDyn`.
- **Validação:** `cargo bench` + `cargo test`.

---

## Sprint 2 — Performance

### T2.1 — Eliminar Branch `is_avx512` do Inner Loop do Resampler [CONCLUIDO]

Impacto estimado: ~5-10% menos overhead por ciclo para 44.1→48 kHz

- **Arquivo:** `src/dsp/resampler.rs` (linhas 165–176)
- **Problema:** O dispatch SIMD (`is_avx512`) é testado **dentro do loop por amostra** em `ResamplerCore::process()`. Cada amostra de saída executa um branch preditivo para selecionar AVX2 vs AVX-512. Em blocos de 256 amostras, são 256 branches desnecessários — o hardware não muda durante a execução.
- **Proposta:** Elevar o dispatch para fora do loop usando o padrão de function pointer ou o mesmo pattern de `dispatch_simd!` macro já usado na inferência. Selecionar a função de convolução (`convolve_stereo_avx2` vs `convolve_stereo_avx512`) uma vez antes do loop e usar um ponteiro de função para toda a iteração.
- **Alternativa (mais elegante):** Refatorar `ResamplerCore::process` em `ResamplerCore::process_internal<ConvolveFn>` com o ponteiro de convolução como parâmetro genérico, e despachar no `NamResampler`.
- **Validação:** `cargo bench` (resampler bench ou inference_bench com modelos 44.1k).

### T2.2 — Fundir `apply_gain_simd` + `detect_clipping_stereo_simd` em Passagem Única [CONCLUIDO]

Impacto estimado: ~3-5% de redução no estágio de saída do pipeline

- **Arquivo:** `src/dsp/gain.rs` + `src/dsp/pipeline.rs` (linhas 310-327)
- **Problema:** No `apply_output_stage`, o pipeline faz 3 passagens separadas sobre os dados:
  1. `apply_gain_simd` L (leitura + escrita de todo o buffer L)
  2. `apply_gain_simd` R (leitura + escrita de todo o buffer R)
  3. `detect_clipping_stereo_simd` (leitura dos buffers L e R)
  - São 4 passagens de memória sobre os mesmos dados (256 amostras × 4 bytes × 2 canais × 4 passagens = 8 KB de tráfego L1 evitável).
- **Proposta:** Criar `apply_gain_and_detect_clipping_stereo_simd(left, right, gain, threshold) -> bool` que aplica ganho e detecta clipping em uma única passagem AVX2. Dentro do loop de 8 amostras: `mul_ps(vals, gain)` + `andnot_ps(sign, result)` + `cmp_ps(abs, threshold)` + `or_ps(accumulator)`.
- **Benefício extra:** Além de reduzir tráfego de cache, o early-exit do clipping agora acontece *durante* a aplicação de ganho, evitando a passagem final inteira quando clipping é detectado.
- **Validação:** `cargo bench` + testes de pipeline existentes.

#### 🔎 Auditoria de Revisão (Sprint 2)

- **T2.1:** Implementado elegantemente via `dispatch_simd!` em `process_input` / `process_output` (em `src/dsp/resampler.rs`), eliminando branch preditivo do inner-loop de `process_internal`. Resampler benchmarked com sucesso.
- **T2.2:** Fundido via kernel `apply_gain_and_detect_clipping_stereo` e integrado em `pipeline.rs`. O design da v-table global `SimdMathConfig` foi expandido.
- **Impacto no Futuro (T3.2):** O stage `apply_output_stage` agora sofre 2 passagens de memória (uma para a fusão Gain+Clip, e outra separada `apply_gain_rt` L+R). A Tarefa **T3.2** é a continuação natural e deverá fundir as chamadas `apply_gain_rt` do hysteresis para eliminar de vez tráfego espúrio.

---

## Sprint 3 — Bug/Limpeza

### T3.1 — Branch `GateState::Closed` Duplicado em `capture_dsp_pipeline` [CONCLUIDO]

Impacto: Redução de ~20 linhas mortas + 1 branch eliminado no hot-path

- **Arquivo:** `src/dsp/pipeline.rs` (linhas 383-403)
- **Bug:** Em `capture_dsp_pipeline`, quando `apply_input_stage` retorna `GateState::Closed`, o código faz `handle_silence_bypass` e retorna (linhas 385-388). Logo em seguida, o `match gate_state` (linhas 390-403) inclui um arm para `GateState::Closed` (linhas 391-394) que **nunca é alcançado**, pois o early return já aconteceu. É dead code.
- **Correção:** Remover o arm `GateState::Closed` do match (linhas 391-394). O match já é exaustivo sem ele pois o Closed já causou return.
- **Benefício:** Elimina dead code e um branch desnecessário na predição do match.
- **Validação:** `cargo test` (pipeline_test).

### T3.2 — `apply_gain_rt` Chamado Duas Vezes no Output Stage (Potencial de Fusão) [CONCLUIDO]

- **Arquivo:** `src/dsp/pipeline.rs` (linhas 321-322)
- **Observação:** `silence_hysteresis.apply_gain_rt` é chamado separadamente para L e R. Se a histerese aplica o mesmo multiplicador de fade para ambos os canais (o que é o caso — é um gate global), esses dois loops poderiam ser fundidos em uma passagem stereo única análoga à proposta T3.1.
- **Ação:** Investigar e, se confirmado, criar `apply_gain_rt_stereo` que processa ambos os canais em uma passagem.

### T3.3 — Saturação do Contador `dropped_frames` no DspBridge [CONCLUIDO]

- **Arquivo:** `src/dsp/pipeline.rs` (linhas 163, 365)
- **Problema:** O `dropped_frames` em `DspBridge` usa `AtomicU32` com `fetch_add(1, Relaxed)`. Após 4.29 bilhões de drops (improvável mas possível em sessões de semanas), o contador faz wrapping silencioso para 0, ocultando problemas. O `LatencyHistogram` já foi corrigido para usar saturação.
- **Proposta:** Usar `fetch_update` com `.min(u32::MAX - 1)` ou simplesmente aceitar o risco dado que 4 bilhões de drops é absurdamente improvável em uso real.
- **Decisão:** Aplicar apenas se for muito barato computacionalmente.

---

## Sprint 4 — Performance

### T4.1 — Habilitar o Path BF16 Real no GEMV 4-Gate LSTM [CONCLUIDO]

Impacto estimado: ~10-20% para modelos LSTM 2-camadas em CPUs com AVX-512 BF16

- **Arquivo:** `src/models/lstm.rs` (linhas 56-78)
- **Problema:** A macro `define_lstm_process!` para as variantes `avx2vnni` e `avx512vnni` passa `$is_bf16 = false`, mesmo tendo VNNI disponível. A variante `avx512_vnni_bf16` passa `true`, mas usa `gemv_4gate_bf16_fallback` como kernel BF16 — que é um fallback escalar.
- **Proposta:** Implementar um kernel `gemv_4gate_bf16_avx512` verdadeiro que use `VDPBF16PS` (AVX-512 BF16 dot product) para o GEMV 4-gate. Isso eliminaria as conversões BF16→F32 por elemento, processando diretamente em BF16 com acumulação F32.
- **Risco:** Médio. Requer nova função SIMD e validação de paridade numérica.
- **Pré-requisito:** Apenas CPUs com feature `avx512bf16` (Sapphire Rapids+).
- **Decisão:** Aplicar apenas se não envolver muita refatoração.
- **Validação:** `cargo bench` + testes de paridade LSTM.

### T4.2 — Substituir Loop Escalar de Backfill por `copy_within` no Prewarm [CONCLUIDO]

Impacto estimado: ~50-80% de redução no tempo de prewarm

- **Arquivo:** `src/models/wavenet.rs` (linhas 1121-1130)
- **Problema:** O prewarm do WaveNetLayerArray preenche o receptive field com um loop escalar `for offset in 1..=receptive_field_size` que copia elemento-a-elemento em ambos os buffers (`layer_buffer` e `layer_buffer_bf16`). Para um receptive field de 1024 frames × 16 canais = 16384 operações, isso é extremamente lento. Note-se que no benchmarking final da Sprint 3, observou-se uma leve regressão de performance adicional nesta função, tornando sua otimização ainda mais prioritária.
- **Proposta:** Substituir por `copy_within` (que o compilador otimiza para `memmove`/`memcpy`) ou `ptr::copy_nonoverlapping`:

  ```rust
  // Ao invés de:
  for offset in 1..=current_state.receptive_field_size {
      let dst_idx = (current_state.buffer_start - offset) * CH;
      for j in 0..CH {
          current_state.layer_buffer[dst_idx + j] = current_state.layer_buffer[start_idx + j];
      }
  }
  // Usar fill_from_slice ou copy repetida:
  let src = &current_state.layer_buffer[start_idx..start_idx + CH];
  for offset in 1..=current_state.receptive_field_size {
      let dst = (current_state.buffer_start - offset) * CH;
      current_state.layer_buffer[dst..dst + CH].copy_from_slice(src);
  }
  ```

- **Alternativa agressiva:** Usar `slice::fill()` se o valor é uniforme, ou `repeat_with` para preencher de uma vez.
- **Validação:** `cargo test` + verificar que os goldens de inferência continuam idênticos.

### T4.3 — Lazy Telemetria com Decimação

Impacto: Eliminação de 2 chamadas RDTSC + 1 multiplicação no callback

- **Arquivo:** `src/pw_host.rs` (callback de captura)
- **Problema:** O callback DSP faz `minstant::Instant::now()` no início e no fim de **cada** ciclo para alimentar o `LatencyHistogram`. `minstant` usa RDTSC, que custa ~20-30 ciclos. Para buffer de 64 samples a 48kHz (1.33ms budget), isso consome ~60 ciclos/callback desnecessariamente.
- **Proposta:** Fazer a medição de telemetria a cada N callbacks (ex: N=16). Um simples contador modular `frame_count & 0xF == 0` (branchless com AND bitmask) decide se mede ou não. O histograma perde resolução temporal (de 750 updates/s para ~47/s) mas isso é mais que suficiente para análise estatística.
- **Risco:** Nenhum. A telemetria é observacional, não afeta o processamento.
- **Decisão:** Aplique se for possível inserir facilmente sem mudar muita lógica, ao final de algum loop que já tenha. Possivelmente a cada periodicidade em que é efetivamente exibido para o usuário.
- **Validação:** `cargo bench` medindo overhead do callback completo.

---

## Sprint 5 — Performance

### T5.1 — Eliminar `fill(0.0)` Redundante em `process_block` Dinâmico

- **Arquivo:** `src/models/wavenet_common.rs` (linhas 354-369)
- **Problema:** `DenseLayerDyn::process_block` faz `out_slice.fill(0.0)` seguido de `M::fused_add_gemv()`. O `fused_add_gemv` *soma* ao output existente (acumulação). Porém, como o output é sempre zerado antes, o efeito líquido é um GEMV overwrite. Usar `M::gemv_overwrite()` diretamente eliminaria o `fill(0.0)`.
- **Impacto:** Reduz tráfego de escrita por um fator de 2 (elimina a passagem de zeragem).
- **Analogia:** É exatamente o que `DenseLayer::process_block` (estático, linha 713-727) já faz — chama `gemv_overwrite` diretamente.
- **Validação:** `cargo test` + `cargo bench`.

### T5.2 — `compute_energy_stereo_avx2` Fundido

- **Arquivo:** `src/dsp/pipeline.rs` (linhas 183-189) + `src/math/simd/`
- **Problema:** `apply_input_stage` calcula `compute_energy_avx2` separadamente para L e R, depois faz `max(energy_l, energy_r)`. Isso percorre os dois buffers em passagens separadas (2× leitura + 2× redução horizontal).
- **Proposta:** Criar `compute_energy_stereo_avx2(l, r) -> f32` que percorre ambos em uma única passagem, acumulando o máximo das energias dos dois canais com `_mm256_max_ps` no loop interno.
- **Impacto:** Corta pela metade o número de passagens de leitura no gate check.
- **Validação:** `cargo bench` + testes de gate.

---
