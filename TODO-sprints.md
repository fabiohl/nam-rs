<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->
# TODO-sprints — Plano de Sprints

> **Convenções de referência cruzada:**
>
> - `Sx.Ty` sem anotação refere-se a uma tarefa **neste documento**.
> - `Sx.Ty (anterior)` refere-se a trabalho já concluído em sprints passadas, cujo código já está no repositório.

## Épico 1 — Quick Wins: Determinismo RT e Performance Mono/Stereo [DONE]

> **Contexto:** Auditoria completa em 2026-06-04 revelou violações RT-safety no PipeWire callback, ausência de DAZ/FTZ no CLAP, overhead stereo redundante no plugin mono-only, e regressões de benchmark. Todas as tarefas deste épico são cirúrgicas — sem mudança arquitetural — e devem ser executadas **antes** dos épicos seguintes para assegurar uma base sólida.

### Sprint S1 — Correções RT-Safety Imediatas

#### Tarefa S1.T01 — Eliminar heap allocation no callback RT do PipeWire 🔥 [DONE]

- **Onde:** `src/standalone/pw_host/rt_callback.rs` — funções `receive_commands()` (linha ~126) e `drain_resamplers()` (linha ~34).
- **Problema 1:** `receive_commands()` cria `Vec::with_capacity(2)` para armazenar modelos antigos durante o swap. Isto é uma **violação direta** da regra de zero heap allocation no thread RT. Embora o path de model swap seja frio (executa apenas quando o modelo muda), a alocação pode causar jitter mensurável se o alocador global estiver sob contenção.
- **Problema 2:** `drain_resamplers()` executa `Box::new(new_rs)` ao receber um `NamResampler` pelo canal SPSC. A alocação heap deveria ocorrer no main thread antes do envio.
- **Solução técnica:**
  1. **`receive_commands()`:** Substituir `let mut old_models = Vec::with_capacity(2);` por `let mut old_models: [Option<Box<DynamicModel>>; 2] = [None, None];` (stack-allocated). Adaptar o loop de push-to-gc para iterar sobre o array filtrando `Some(...)`.
  2. **`drain_resamplers()`:** Alterar o tipo do canal SPSC de `Consumer<NamResampler>` para `Consumer<Box<NamResampler>>`. O main thread (em `src/standalone/pw_host/mod.rs`, local que envia o resampler) passa a fazer `Box::new(...)` antes de `producer.push(...)`. No callback RT, substituir `let new_rs = Box::new(new_rs);` por `let new_rs = new_rs;` (já é Box).
- **Critérios de aceitação:**
  - `cargo test --features heap-audit` não dispara `RT_STATUS_HEAP_ALLOC`.
  - `grep -rn "Vec::with_capacity\|Vec::new" src/standalone/pw_host/rt_callback.rs` retorna zero resultados.
  - `grep -rn "Box::new" src/standalone/pw_host/rt_callback.rs` retorna zero resultados (exceto em doc-comments).
  - Benchmarks `cargo bench` sem regressão (±1% tolerance).
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

#### Tarefa S1.T02 — Configurar DAZ/FTZ no primeiro bloco do CLAP audio thread 🔥 [DONE]

- **Onde:** `src/clap/processor/mod.rs` — dentro do bloco `if !self.prio_checked` (linhas ~289-305).
- **Problema:** O standalone chama `crate::math::common::set_daz_ftz()` via `rt_setup::apply_rt_thread_config()` na thread RT. O CLAP plugin **não faz isso**, confiando que o host DAW configura FTZ+DAZ. Hosts como Ardour, Zrythm, Qtractor e hosts menores **não garantem** FTZ/DAZ. Sem essas flags, denormals gerados internamente pelas activations do LSTM/WaveNet (sigmoid/tanh) podem causar penalidade de 50-100x por operação afetada, manifestando-se como xruns esporádicos sob carga.
- **Nota:** O plugin já injeta `DENORMAL_DITHER_OFFSET` (1e-11) na entrada para prevenir denormals no input, mas isto **não protege** contra denormals gerados internamente pelo modelo neural nas activations.
- **Solução técnica:**
  1. Adicionar `unsafe { crate::math::common::set_daz_ftz(); }` imediatamente após o bloco `if libc::pthread_getschedparam(...)` (mas dentro do `if !self.prio_checked`), garantindo execução one-time no primeiro bloco processado.
  2. O `set_daz_ftz()` já é idempotente e seguro para chamar múltiplas vezes (seta bits no MXCSR).
- **Critérios de aceitação:**
  - Teste unitário existente `test_set_daz_ftz` (`src/math/common/tests.rs`) continua verde.
  - Benchmark `WaveNet_Standard_CH16_64samp_48kHz` sem regressão.
  - Teste manual: plugin CLAP em host que não configura FTZ (Ardour) processa sinal near-silence (1e-20) sem aumento de latência p99.
- **Especialista:** `implementador`.
- **Esforço:** 0.25 dia.

#### Tarefa S1.T03 — Corrigir compensação assimétrica do denormal dither no output stage 💡 [DONE]

- **Onde:** `src/dsp/pipeline/stages.rs` — `apply_input_stage()` (linhas ~98-107) e `apply_output_stage()` (linhas ~250-255).
- **Problema:** O input stage adiciona `DENORMAL_DITHER_OFFSET` (1e-11) a `samples_l` sempre, e a `samples_r` **apenas quando `!process_mono`** (correto). Porém, o output stage subtrai o offset de **ambos** `resamp_out_l` e `resamp_out_r` incondicionalmente (loop nas linhas 251-254). Quando `process_mono=true`, o canal R não recebeu o offset na entrada, mas o output subtrai dele, introduzindo um DC residual de -1e-11 no canal R. Embora inaudível (-220 dBFS), viola o princípio de simetria matemática e pode acumular drift em chains longas.
- **Solução técnica:**
  1. Adicionar parâmetro `process_mono: bool` à assinatura de `apply_output_stage()`.
  2. Condicionar a subtração do offset no canal R: `if !process_mono { /* subtrair de R */ }`.
  3. Atualizar todos os call-sites: `src/clap/processor/dsp.rs` (linha ~315), `src/dsp/pipeline/stages.rs` (chamadas internas), e `src/standalone/pw_host/` (se aplicável via `capture_dsp_pipeline`).
  4. Atualizar testes em `src/dsp/pipeline/pipeline_test.rs` para verificar DC offset zero em R quando mono.
- **Critérios de aceitação:**
  - Novo teste unitário: output mono tem DC offset ≤ 1e-15 em ambos canais (float epsilon).
  - Todos os testes existentes passam sem regressão.
  - `cargo test --release --test soak_test` verde.
- **Especialista:** `implementador`.
- **Esforço:** 0.25 dia.

---

### Sprint S2 — Otimização Mono no CLAP Plugin

> **Contexto:** O CLAP plugin opera em modo mono-only (linhas 126-127 de `dsp.rs`: `active_channel_count=1, process_mono=true`). No entanto, o code path processa buffers R completos (memcpy, gain, peak detection) redundantemente. O código stereo deve ser preservado sob `#[cfg(feature = "stereo")]` para futuro suporte, mas o path default deve ser mono-optimizado.

#### Tarefa S2.T01 — Eliminar overhead stereo redundante no input processing do CLAP ⚠️ [DONE]

- **Onde:** `src/clap/processor/dsp.rs` — linhas ~156, ~170-213.
- **Problema:**
  1. **Linha 156:** `buf_host_r[..n_samples].copy_from_slice(&buf_host_l[..n_samples])` — copia L→R incondicionalmente. Em modo mono-only, a cópia é desnecessária porque o pipeline (`stages.rs:apply_input_stage`) e a inferência (`stages.rs:run_inference`) já lidam com mono via `process_mono` flag.
  2. **Linhas 186-188:** `apply_gain_and_detect_clipping_stereo()` processa ambos L e R, mesmo quando mono.
  3. **Linhas 195-208:** `apply_ramp_stereo()` e `compute_peak_abs_stereo()` idem.
- **Solução técnica:**
  1. Envolver a cópia L→R (linha 156) e todas as operações sobre `buf_host_r` no input gain em `#[cfg(feature = "stereo")]`, ou condicioná-las a `!self.process_mono`.
  2. Para o path mono: usar `apply_gain_and_detect_clipping` single-buffer (criar variante mono se não existir, ou reusar `apply_gain_simd` + inline clipping check).
  3. No path mono, `peak_r = peak_l` (ambos canais são idênticos).
  4. Preservar o código stereo completo sob `#[cfg(feature = "stereo")]` para futuro uso.
- **Critérios de aceitação:**
  - Benchmark `WaveNet_Standard_CH16_64samp_48kHz` melhora ≥ 2% (eliminação de memcpy 64*4=256 bytes + gain R + peak R).
  - Output L/R do plugin mono permanece bit-exact com a versão anterior.
  - `cargo test` e `cargo test --features heap-audit` verdes.
  - Cross-validation `cpp_parity` mantém MSE/SNR dentro das tolerâncias.
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 0.5 dia.

#### Tarefa S2.T02 — Eliminar overhead stereo redundante no output processing do CLAP ⚠️ [DONE]

- **Onde:** `src/clap/processor/dsp.rs` — linhas ~325-397.
- **Problema:**
  1. **Linhas 325-353:** `apply_gain_stereo` e `apply_ramp_stereo` processam `buf_out_r` desnecessariamente em modo mono.
  2. **Linhas 371-397:** `compute_peak_abs_stereo` itera sobre ambos buffers de output.
- **Solução técnica:**
  1. Path mono: aplicar gain/ramp apenas em `buf_out_l[..n_out]`.
  2. Copiar resultado L→R **apenas** no momento do write de output (linhas 365-369: `o_r[..n].copy_from_slice(&self.buf_out_l[..n])` em vez de `&self.buf_out_r[..n]`).
  3. Peak detection mono: computar apenas `peak_l`, setar `peak_r = peak_l`.
  4. Envolver path stereo em `#[cfg(feature = "stereo")]`.
- **Critérios de aceitação:**
  - Mesmos critérios de S2.T01 (benchmark, bit-exact, testes).
  - Ganho combinado S2.T01+S2.T02 ≥ 3% no benchmark principal.
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 0.25 dia (execução conjunta com S2.T01).

---

### Sprint S3 — Diagnóstico e Correção de Regressões de Benchmark

#### Tarefa S3.T01 — Investigar e corrigir regressão +28.5% no LSTM_1x8/SIMD_Fused_T3 ⚠️ [DONE]

- **Onde:** `benches/inference_bench.rs` (benchmark), `src/math/lstm/` (kernels LSTM), `src/math/gemm/` (GEMM fused).
- **Problema:** O benchmark `LSTM_1x8_Comparison/SIMD_Fused_T3` regrediu **+28.5%** (de ~2.29 µs para 2.98 µs). Enquanto isso, o LSTM 2x16 está estável (+1.1%). A regressão pode ser:
  - (a) Issue de code alignment/layout (LLVM inlining threshold afetando o 1x8 path diferentemente do 2x16);
  - (b) Regressão real introduzida em refactor recente do GEMM fused;
  - (c) Cache contention devido a mudança em struct layout.
- **Solução técnica:**
  1. Executar `git bisect` com o benchmark como critério de aceitação (threshold ≤ 2.35 µs) para identificar o commit que introduziu a regressão.
  2. Se code alignment: ajustar `#[repr(align)]` do struct do LSTM 1x8 ou adicionar `#[cold]` em paths não-hot para influenciar layout.
  3. Se code generation: verificar com `cargo asm` (crate `cargo-show-asm`) se o loop unrolling do 1x8 path está sendo afetado por inlining threshold. Considerar `#[inline(always)]` ou `#[inline(never)]` cirúrgicos.
  4. Se cache: verificar com `perf stat -e L1-dcache-load-misses` antes e depois do commit identificado.
- **Critérios de aceitação:**
  - Benchmark `LSTM_1x8_Comparison/SIMD_Fused_T3` retorna ao baseline (≤ 2.35 µs).
  - Nenhuma outra métrica regride (tolerância ±2%).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 1 dia.

#### Tarefa S3.T02 — Investigar regressão +5.6% no DotProduct_AVX2_256elem 💡 [DONE]

- **Onde:** `src/math/gemm/dot.rs` (kernel dot product), `benches/inference_bench.rs`.
- **Problema:** O benchmark `DotProduct_AVX2_256elem` regrediu **+5.6%** (de ~11.26 ns para 11.90 ns). O `DotProduct_AVX2_64elem` permanece estável. Possível issue de TLB/cache alignment específico para buffers de 256 elementos (1 KB — próximo da fronteira de cacheline L1).
- **Solução técnica:**
  1. `git bisect` para identificar commit.
  2. Verificar se padding/alignment dos buffers de teste mudou (devem ser `AlignedVec<f32>` com 64-byte alignment).
  3. Verificar se houve mudança no layout de `dot.rs` que afete o hot loop (e.g., adição de branches antes do loop principal).
  4. Se necessário, adicionar `#[repr(align(64))]` explícito nos buffers de benchmark.
- **Critérios de aceitação:**
  - Benchmark retorna a ≤ 11.4 ns.
  - Nenhum outro benchmark regride.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 0.5 dia.

---

### Sprint S4 — Cross-Validation WaveNet Standard v2 Multi-SR

#### Tarefa S4.T01 — Corrigir render CLI standalone para aceitar SR ≠ 48000 Hz com WaveNet Standard 💡 [DONE]

- **Onde:** `src/main.rs` ou `src/bin/` (CLI render), `src/dsp/resampler.rs`.
- **Problema:** O cross-validation test (`tests/cpp_parity.rs`) pula WaveNet Standard v2 quando o sample rate é ≠ 48000 Hz porque o render CLI standalone retorna exit code 1 com a mensagem "Input WAV sample rate (44100 Hz) does not match model expected rate (48000 Hz)". Os modelos LSTM passam nesses cenários porque o resampler é ativado. O WaveNet Standard deve suportar o mesmo path de resampling. Evidência nos logs: `target/logs/cpp-parity.log` linhas 82-85, 291-292, 300-301, 314-315.
- **Solução técnica:**
  1. Identificar no CLI render onde a validação de sample rate rejeita SR ≠ model_rate para WaveNet.
  2. Reusar o `NamResampler` existente para converter input WAV → model_rate → output WAV (mesmo path que LSTM já usa).
  3. Verificar se a restrição era intencional (possível bug no render, não no modelo — o plugin CLAP e standalone PipeWire já fazem resampling com WaveNet Standard).
  4. Atualizar testes de cross-validation para remover SKIPs e validar WaveNet Standard em 44100, 88200, 96000 e 192000 Hz.
- **Critérios de aceitação:**
  - `cargo test --release --test cpp_parity -- --ignored --nocapture` produz resultados (não SKIP) para WaveNet Standard v2 em todos os sample rates testados (44100, 88200, 96000, 192000 Hz).
  - MSE e SNR dentro das tolerâncias definidas pelo teste para cada SR.
  - Zero regressão nos resultados existentes de LSTM e WaveNet Nano/Feather.
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

---

## Épico 2 — Otimizações Gerais

### Sprint S5 — Quantização e Compressão de Modelos

#### Tarefa S5.T01 — INT8 weight quantization SmoothQuant para Conv1D heads ✨⚠️ [DESCARTADO]

> Nota do PO: Decidimos não valer a pena. Descartado.

- **Onde:** `src/loader/dispatcher/wavenet/` (heads de Conv1D — módulos `standard.rs` e `dynamic.rs`); novo `src/math/common/int8_quant.rs`; novo `weights_layout = SmoothQuantInt8`.
- **Problema/Oportunidade:** Pesos do `head_weights` (Conv1D 1×1 do output) **dominam memória** em WaveNet Standard (40 KB de pesos vs 8 KB de activations). INT8 weights + FP32 activations (per-channel scale) reduzem 4× memory bandwidth (cache-friendly em L1/L2). SmoothQuant migra outliers de activations para weights via per-channel scaling — proven 99.5% accuracy retention em LLM.cpp e NAM-class workloads.
- **Solução técnica:**
  1. **Treinamento-livre quantization** (post-training): para cada Conv1D head, computar per-channel scale `s_c = max(|W_c|) / 127`, armazenar `Q_W[c,i] = round(W[c,i] / s_c)` como `i8` + scale vector `s_c` como `f32`.
  2. **Kernel `dot_product_int8_avx512`** usando `_mm512_dpbusd_epi32` (AVX-512 VNNI) — 4× speedup vs F32 FMA em throughput INT8.
  3. **AMX path:** `_tile_dpbssd` para LSTM matmul INT8.
  4. **Encoder NAMB v3:** novo `weights_layout = SmoothQuantInt8` que serializa `[Q_W: i8, scales: f32]`. v3 bump justificado.
  5. **Auto-calibração:** durante o `loader/mod.rs`, opcional sweep de input típico (impulse response) para ajustar scales adversariamente.
  6. **Fallback:** se SmoothQuant falha calibração (golden delta > tolerância), reverter para BF16/FP32 com warning.
- **Pré-requisitos (obrigatórios — herdam invariantes já concluídas em sprints anteriores):**
  - Disciplina de layout sequencial e padding implícito (concluída em sprints anteriores — código já no repositório). SmoothQuant deve usar a mesma estratégia (bloco contíguo `[Q_W: i8 ..., scales: f32 ...]` por camada, padding para múltiplo do bloco SIMD).
  - Flag `FLAG_HAS_CRC32` explícito e spec NAMB (concluída em sprints anteriores). A seção `SmoothQuantInt8` deve ser adicionada à spec **antes** da implementação; bump explícito para NAMB v3 com `FLAG_HAS_QUANT_INT8`.
  - Round-trip encode/decode (concluído em sprints anteriores). Cobertura obrigatória do novo layout antes do merge.
- **Critérios de aceitação:**
  - Modelo WaveNet Standard quantizado: tamanho do arquivo 60% menor, MSE vs FP32 < 1e-3 em 60s de signal de teste.
  - Benchmark mostra ≥ 30% redução em latência média para WaveNet Standard.
  - Round-trip encode/decode preserva pesos com erro < 1/127, validado via harness estendido.
- **Especialista:** `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

#### Tarefa S5.T02 — INT4 weight packing experimental (AWQ-style) ✨💡 [DESCARTADO]

> Nota do PO: Decidimos não valer a pena. Descartado.

- **Onde:** extensão de S5.T01 para `weights_layout = AwqInt4`.
- **Problema/Oportunidade:** INT4 (4 bits) entrega 8× memory reduction. AWQ (Activation-aware Weight Quantization, Lin et al. 2023) preserva pesos "salientes" em FP16 e quantiza o resto em INT4. Apropriado para WaveNet com layers de magnitude variada (~1% dos pesos contribuem >50% do output).
- **Solução técnica:**
  1. Identificar 1% top-magnitude weights via análise off-line (script `utils/awq-calibrate.py` opcional, ou heuristic Rust).
  2. Layout: `[Q_W: u4 packed nibbles, salient_mask: bitmap, salient_values: f16, scales: f32]`.
  3. Decoder kernel: unpack INT4 → INT8 com LUT, depois INT8 dot product (reusa S5.T01 path).
  4. **Apenas catálogo dinâmico** (não Conv1D estático) — INT4 é override expressivo.
- **Pré-requisitos (obrigatórios):** S5.T01 (path INT8 + scales infra), spec NAMB v3 com `FLAG_HAS_QUANT_INT4` (adicionar à spec antes da implementação), round-trip estendido (cobertura do novo layout).
- **Critérios de aceitação:**
  - MSE < 5e-3 para WaveNet Standard quantizado AWQ vs FP32 (tolerância dobrada vs INT8).
  - Tamanho de arquivo 80% menor que FP32.
  - Round-trip encode/decode validado no harness estendido.
  - Feature `awq-int4` em Cargo (default off).
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 3 dias.

#### Tarefa S5.T03 — Kahan summation em acumuladores críticos ✨💡 [DONE]

- **Onde:** `src/math/gemm/dot.rs`, `dot_4x.rs` (acumuladores `horizontal_sum`).
- **Problema/Oportunidade:** Em LSTM de muitas amostras, drift de soma FP32 acumula erro de magnitude `~N · eps`. Kahan summation (compensated summation) reduz para `O(1)` em troca de 2 FMAs extras — tolerável fora do tightest inner loop.
- **Solução técnica:**
  1. Apenas em `horizontal_sum` (1× por bloco GEMM), não no inner FMA.
  2. Manter `compensation: f32` acumulador secundário.
- **Critérios de aceitação:** Drift vs scalar reference em LSTM de 1M amostras reduz ≥ 100×.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1 dia.

---

### Sprint S6 — Responsividade

#### Tarefa S6.T01 — Async model loading via io_uring ✨⚠️ [DESCARTADO]

> Nota do PO: Decidimos não valer a pena. Descartado.

- **Onde:** `src/loader/mod.rs:70-94`; novo `src/loader/async_io.rs`.
- **Problema/Oportunidade:** Hoje `std::fs::read` é síncrono — usuário arrastando modelo grande de 30 MiB em DAW vê **UI freeze** por ~100ms (SSD) ou ~2s (NFS). io_uring permite zero-syscall I/O completion + worker thread separado, mantendo UI responsiva.
- **Solução técnica:**
  1. Crate `io-uring` (sem deps pesadas além de libc).
  2. Worker thread dedicado lê arquivo via SQE/CQE assíncronos; main thread continua draw loop.
  3. Progress reporting via `Arc<AtomicU64>` (bytes lidos).
  4. Em main thread, "Loading..." status com progress bar.
  5. Fallback `std::fs::read` para kernels < 5.1.
- **Critérios de aceitação:**
  - UI permanece responsiva (>30 FPS) durante load de modelo 30 MiB.
  - Tempo de load ≤ `std::fs::read` baseline (não regredir).
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S6.T02 — Huge Pages (THP / MAP_HUGETLB) para weights e mirror buffer ✨⚠️ [DONE]

- **Onde:** `src/loader/mod.rs` (alocação de `AlignedVec<u16>` para pesos dinâmicos); `src/dsp/mirror_buf.rs` + `src/dsp/mirror_buf/linux.rs`.
- **Problema/Oportunidade:** Modelos WaveNet Standard alocam ~80 KB de pesos contíguos. Em páginas de 4 KB, esses pesos consomem ~20 entradas TLB; em **2 MiB huge pages**, **1 entrada TLB**. TLB miss em hotpath custa ~100 ciclos. Para audio de 32 spl @ 96k = 333 µs, eliminar TLB misses pode reduzir p99 em 5–15%.
- **Solução técnica:**
  1. **Allocator helper** `src/math/common/huge_alloc.rs`: tenta `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` primeiro; fallback `mmap` anonymous + `madvise(MADV_HUGEPAGE)` para THP transparent; fallback `Vec` standard.
  2. Substituir alocações de pesos > 1 MiB e mirror buffer (`mirror_buf.rs`) por esse allocator.
  3. **Métrica:** expor count via `RT_STATUS_HUGEPAGE_OK` flag (telemetria).
  4. **Cautela:** THP background scanning pode pausar threads — preferir explicit `MAP_HUGETLB`. Documentar setup: `echo 32 > /proc/sys/vm/nr_hugepages` ou cgroup hugetlb.2MB.max.
- **Critérios de aceitação:**
  - `perf stat -e dTLB-load-misses` mostra redução ≥ 50% no DSP thread.
  - Benchmark p99 latency reduz ≥ 5% em modelos grandes.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S6.T03 — Soft-degrade automático sob CPU pressure (graceful fallback) ✨🔥 [DONE]

- **Onde:** novo `src/dsp/adaptive.rs`; integração em `src/clap/processor/dsp.rs` e `src/standalone/pw_host/rt_callback.rs`.
- **Problema/Oportunidade:** Hoje, quando o host empurra a CPU acima do budget (live performance com 4 plugins + outros tracks, ou laptop em bateria com `cpufreq schedutil` agressivo), o nam-rs **falha hard** com `xrun` audível. **Soft-degrade** = detectar pressão precocemente (p95 do bloco anterior > 70% do budget) e **reduzir graciosamente** a carga (truncate receptive field, skip Conv1D layers profundos, ou desativar oversampling) com **crossfade** transparente. Trade controlado: pequena perda de fidelidade vs glitch audível.
- **Solução técnica:**

  1. **Hysteresis com 3 estados:** `Full / Reduced / Minimal`.
     - `Full → Reduced`: três blocos consecutivos com `latency_us > 0.70 * budget_us`.
     - `Reduced → Minimal`: três blocos consecutivos com `latency_us > 0.85 * budget_us`.
     - Caminho reverso: cinco blocos consecutivos abaixo do threshold inferior (histerese assimétrica — descida lenta evita oscilação).

  2. **Estratégias de redução** (por arquitetura):
     - **WaveNet:** `Reduced` desativa últimas N_dilation_layers (configurable, default 25% dos layers); `Minimal` desativa 50%.
     - **LSTM:** `Reduced` mantém apenas primeira camada (2×16 → 1×16); `Minimal` skip total + passa input com gain compensado.

  3. **Crossfade** entre estados (32 ms linear ramp, similar ao hot swap já implementado, mas **intra-modelo**).

  4. **Telemetria:** `RT_STATUS_DEGRADE_REDUCED` e `RT_STATUS_DEGRADE_MINIMAL` flags; counter `degrade_transitions_total`.

  5. **Param** `PARAM_ADAPTIVE_COMPUTE: enum { Off, Conservative, Aggressive }` (default Conservative no CLAP plugin; Off no standalone — usuário standalone tipicamente já tem sistema tunado).

  6. **UX feedback:** GUI ícone discreto em status bar quando `Reduced/Minimal` ativo, com tooltip explicativo.
- **Pré-requisitos:**
  - HDR Histograms para estatística p95 (já implementado em `dsp/telemetry.rs`).
  - Telemetria lock-free `fetch_add` (já implementada em `common/spsc.rs`).
  - Reset de estado RNN ao mudar de variante (já implementado via trait).
  - Crossfade de hot swap (já implementado — reusar máquina de crossfade).
- **Critérios de aceitação:**
  - Stress test `stress-ng --cpu 16 --cpu-load 90` durante 60 s com nam-rs ativo: zero xruns audíveis; transição para `Reduced` detectada em telemetria; retorno a `Full` quando stress termina.
  - Soak test 1h em laptop alimentado por bateria: nam-rs degrada graciosamente quando CPU thermal throttle ativa.
  - `cargo test --features heap-audit` zero alloc na transição.
- **Especialista:** `pesquisador-inovador` + `implementador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

---

### Sprint S7 — Compiler-Grade Optimization (PGO + BOLT)

> **Requisitos do Ambiente de Desenvolvimento:**
> Para executar esta sprint no Ubuntu Linux (versão mais recente), são necessárias as seguintes dependências no sistema (`apt`), no ecossistema Rust/Cargo (`rustup`/`cargo`) e configurações de privilégios de execução:
>
> 1. **Dependências do Sistema (`apt`):**
>    - `bolt-22`: Otimizador post-link LLVM BOLT. Deve ser instalada a versão 22 para compatibilidade com o LLVM 22.1.2 do `rustc 1.96.0`.
>      - Comando: `sudo apt install bolt-22` (ou `llvm-bolt` caso aponte para versão compatível).
>    - `linux-tools-generic` e `linux-tools-$(uname -r)`: Necessários para a ferramenta `perf`, utilizada para capturar os profiles de execução representativos do BOLT.
>      - Comando: `sudo apt install linux-tools-generic linux-tools-$(uname -r)`
>
> 2. **Componentes Rust e utilitários Cargo (`rustup` e `cargo`):**
>    - `llvm-tools-preview` (via `rustup`): Fornece o `llvm-profdata` da mesma versão exata do LLVM do compilador Rust instalado. Essencial para converter e mesclar os arquivos `.profraw` do PGO.
>      - Comando: `rustup component add llvm-tools-preview`
>    - `cargo-pgo` (via `#### Tarefa S7.T01 — Profile-Guided Optimization (PGO) build pipeline ✨⚠️ [DONE]

- **Onde:** `Cargo.toml`; novo `utils/build-release.sh` (consolidando scripts anteriores).
- **Problema/Oportunidade:** Rustc/LLVM PGO instrumenta build → roda workload representativo → coleta profile → rebuilda com `-Cprofile-use`. Tipicamente entrega 5–15% throughput em hotpath. Já standard em Firefox, Chromium.
- **Solução técnica:**
  1. Script de release unificado `./utils/build-release.sh` realiza o pipeline PGO de forma transparente e robusta.
  2. Preserva as flags de arquitetura e tempo real de `.cargo/config.toml` (extraídas via python de forma inteligente).
  3. Compila tanto o executável standalone (`nam-rs`) quanto o plugin CLAP (`nam-rs.clap`).
- **Critérios de aceitação:** Benchmark inference reduz ≥ 5% latência média em PGO build vs vanilla release.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

> **Notas da Entrega (2026-06-05):**
>
> - **Feature `pgo`** adicionada ao `Cargo.toml`.
> - **Script unificado `utils/build-release.sh`** consolida todo o pipeline em fases automatizadas e robustas.
> - **Workload representativo:** `cargo bench` cobre WaveNet Standard/Feather/Nano, LSTM 1x8/1x16/2x16/1x40/2x24, FastMath, dot products, e resampler.
> - **Preservação de Otimizações:** As flags de `.cargo/config.toml` (como `-Ctarget-cpu=x86-64-v3` e `-Clink-arg=-Wl,-z,now`) são extraídas e preservadas ativamente.

#### Tarefa S7.T02 — BOLT post-link layout optimization ✨💡 [DONE]

- **Onde:** `utils/build-release.sh` (integrando o fluxo pós-link anterior).
- **Problema/Oportunidade:** LLVM BOLT é a "última gota": reordena basic blocks no binário linkado para que hot paths fiquem em sequência (melhor L1i utilização). Combinado com PGO, mais 3–8%.
- **Solução técnica:**
  1. Integrado em `./utils/build-release.sh` de forma automatizada.
  2. Coleta dados de CPU cycles via amostragem de áudio real do PipeWire (se ativo) ou fallback automático para os benchmarks.
  3. Gera o executável final otimizado com BOLT em `~/.local/bin/nam-rs` e valida o SONAME/símbolos do CLAP `~/.clap/nam-rs.clap`.
- **Critérios de aceitação:** L1i miss rate (`perf stat`) reduz ≥ 20%; latency média -3-8%.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1 dia.

> **Notas da Implementação (2026-06-05):**
>
> - **Script unificado `utils/build-release.sh`** implementa todo o fluxo BOLT em fases sequenciais com fallbacks elegantes, garantindo entrega automática para `~/.local/bin/nam-rs` e `~/.clap/nam-rs.clap`.
> - **Amostragem PipeWire & Benchmark:** Realiza amostragem de áudio PipeWire se ativo, com fallback gracioso para os benchmarks se necessário.
> - **Auditoria de Validabilidade:** Executa checks automáticos de ELF 64-bit, SONAME e símbolo `clap_entry` antes da entrega final dos binários compilados em `--release`.
> - **Pré-requisitos:** `sudo apt install llvm-22-tools` (provê `llvm-bolt` + `perf2bolt`), `linux-tools-generic` (provê `perf`), `sysctl kernel.perf_event_paranoid=1`.

---

## Épico 3 — DSP e Suporte Técnico

### Sprint S8 — DSP Suplementar

#### Tarefa S8.T01 — IR cabsim convolution (uniformly-partitioned FFT) ✨🔥 [ADIADO]

- **Onde:** novo `src/dsp/ir_cab.rs`.
- **Problema/Oportunidade:** Workflow NAM é "amp + cabinet". Hoje, usuário precisa de plugin separado (Topaz, NadIR). Integrar cabsim com convolução IR (impulse response, .wav de 4096–8192 spl) **elimina um plugin do chain** e habilita workflow "amp+cab presets" únicos.
- **Solução técnica:**
  1. **Uniformly-Partitioned Convolution (UPC):** dividir IR em blocos de N=64 amostras; convolve cada bloco via FFT 128-point (já existe `rustfft`); somar com latência total = N.
  2. **Frequency-domain delay line** evita realocação por bloco.
  3. SIMD complex multiply em FFT bins.
  4. **CLAP IO format:** parâmetros `PARAM_IR_PATH` (file picker drag-drop), `PARAM_IR_GAIN`, `PARAM_IR_ENABLED`.
  5. Carregamento async via io_uring (S6.T01).
- **Critérios de aceitação:**
  - Convolução de IR 4096-tap em < 50% do block budget @ 48k/64 spl.
  - Match bit-perfect vs reference convolution (numpy.convolve) com FFT round-trip.
  - GUI: drag-drop file picker para IR (.wav).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 4 dias.

---

### Sprint S9 — Suporte ao Usuário & Diagnóstico de Campo (Observability sem regressão) ✨⚠️

> **Contexto e justificativa:** A skill `diagnostico` (vide `.agents/workflows/diagnostico.md`) espera receber um "bloco de suporte" colado pelo usuário contendo código de erro, mnemônico, parâmetros contextuais e info de sistema. Hoje o `Diagnostic::support_block()` (`src/common/diagnostics/diagnostic.rs`) só é gerado em **paths de erro** (`emit`/`emit_warning`). Cenários frequentes ficam descobertos:
>
> - Usuário relata "som baixo" / "dropouts" / "GUI travada" — sem erro tipado, nada para colar.
> - Usuário em hospedeiro CLAP (Bitwig/Reaper/FL Studio) não tem stderr acessível.
> - Crashes em hosts C++ (Bitwig) podem perder o `log::error!` antes do abort.
>
> **Princípios invariantes** (todas as tarefas):
>
> - **Zero hotpath cost:** coleta exclusivamente via `load(Relaxed)` em atomics já existentes. Nenhum novo flag/counter no `process()`. Bundle gerado on-demand no main thread.
> - **Zero alloc em RT:** toda I/O e formatação no main thread. Panic hook só roda fora do hotpath (após unwind iniciado).
> - **Segurança:** redação default de paths absolutos (`$HOME` → `~`); nunca embarcar conteúdo de pesos/áudio; opt-in `--diagnose-full` para incluir paths completos.
> - **Forward-compat:** o formato textual do bundle preserva contrato consumido pela skill `diagnostico` (Fase 1.1 do workflow). Novos campos são **anexados** em linhas próprias; parsers antigos da IA ignoram silenciosamente.

#### Tarefa S9.T01 — Refatorar `support_block()` para `DiagnosticBundle` desacoplado de erro 💡 [DONE]

- **Onde:** `src/common/diagnostics/diagnostic.rs` (atual `support_block` é método privado de `Diagnostic`).
- **Problema:** `support_block()` é privado e exige um `NamErrorCode` para ser construído. Não há API pública para "gerar bundle em estado nominal".
- **Solução técnica:**
  1. Extrair `pub struct DiagnosticBundle { system: SystemInfo, runtime: RuntimeSnapshot, error: Option<ErrorContext> }` em `src/common/diagnostics/diagnostic.rs`.
  2. `impl DiagnosticBundle { pub fn capture() -> Self; pub fn capture_with_error(code, params) -> Self; pub fn render(&self) -> String; }`.
  3. `RuntimeSnapshot` (vazio nesta tarefa — preenchido em S9.T04) — placeholder com `Default`.
  4. Refatorar `Diagnostic::support_block` para delegar ao novo `DiagnosticBundle::capture_with_error(...).render()`.
  5. Preservar o cabeçalho textual exato (`──── NAM-rs Diagnostic ...`) para retro-compat com skill `diagnostico`.
- **Critérios de aceitação:** `Diagnostic::emit` produz string byte-idêntica à anterior em paths de erro existentes. Novo `DiagnosticBundle::capture().render()` retorna bloco sem campo de erro.
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

#### Tarefa S9.T02 — Comando CLI `--diagnose` no standalone ⚠️ [DONE]

- **Onde:** `src/main.rs::parse_args`, `src/main.rs::cli_loop` (comando interativo `:diag`).
- **Problema:** Usuário do standalone PipeWire não tem como gerar bundle em estado nominal.
- **Solução técnica:**
  1. Flag CLI `nam-rs --diagnose` — imprime bundle imediatamente no stdout e sai com código 0, sem inicializar áudio.
  2. Comando interativo `:diag` (e alias `:support`) — chama `DiagnosticBundle::capture().render()` enquanto sessão está rodando, incluindo `RuntimeSnapshot` real (modelo carregado, SR efetivo, contadores).
  3. `:diag --full` (interativo) ou `--diagnose-full` (flag CLI) — desabilita redação de paths (`$HOME` permanece).
  4. Adicionar entrada em `--help` documentando ambos.
- **Critérios de aceitação:** `nam-rs --diagnose` imprime bundle em <100ms; `:diag` em sessão ativa inclui SR e modelo atual.
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

#### Tarefa S9.T03 — Botão "Copy Diagnostic" na GUI do CLAP ⚠️ [DONE]

- **Onde:** `src/clap/gui/ui/mod.rs` (status bar / nova zona "About"). O status bar reside em `ui/mod.rs` (função `draw_ui`), resultado do refactor de UI já concluído em sprint anterior.
- **Problema:** Usuário do plugin em DAW não tem acesso ao stderr do host. Sem botão na GUI, impossível obter bundle em hosts C++.
- **Solução técnica:**
  1. Botão pequeno na status bar (ou ícone "ℹ" abrindo modal "About / Diagnostic").
  2. Click → no **main thread** chamar `DiagnosticBundle::capture().render()` e:
     - (a) Copiar para clipboard via `arboard` (já é dependência leve — verificar; senão usar `egui::Context::output_mut(|o| o.copied_text = ...)`).
     - (b) Persistir cópia em `~/.cache/nam-rs/diagnostic-<unix_ts>.txt` (criar dir se necessário, `0o700` permissão).
  3. Toast/feedback visual ("Diagnostic copiado · arquivo em ~/.cache/nam-rs/").
  4. Botão "Open folder" abre o diretório via `xdg-open` (Linux).
  5. **Sem nova alocação em RT:** toda lógica em GUI thread (já fora do path RT).
- **Critérios de aceitação:**
  - Em Bitwig/Reaper: click no botão coloca bloco no clipboard pronto para colar em chat com dev.
  - Arquivo `~/.cache/nam-rs/diagnostic-*.txt` criado com permissão 0o600.
  - Teste smoke: heap-audit não dispara (alocação ocorre em GUI thread).
- **Especialista:** `implementador`.
- **Esforço:** 1 dia.

#### Tarefa S9.T04 — `RuntimeSnapshot` lock-free com estado RT-safe ⚠️ [DONE]

- **Onde:** `src/common/diagnostics/diagnostic.rs` (novo `RuntimeSnapshot`); consumidores em `src/clap/processor/dsp.rs` + `src/clap/processor/events.rs`, `src/standalone/pw_host/rt_callback.rs`, `src/dsp/telemetry.rs`.
- **Problema:** Bundle atual só tem versão + arch + features estáticos. Falta o **estado dinâmico** crítico para diagnóstico: modelo carregado (arquitetura/CH/RF/path basename), SR efetivo, buffer size, contadores de xrun/drain, RT prio aplicada, scheduler ativo (FIFO/DEADLINE), percentis de latência (HDR histograms em `telemetry.rs`), histórico recente de RT_STATUS flags.
- **Solução técnica:**
  1. Definir struct `RuntimeSnapshot` com campos:
     - `model: Option<ModelInfo { arch_label, channels, receptive_field, weights_layout, path_basename }>`
     - `audio: AudioInfo { sample_rate, buffer_size, channel_count, host_name (CLAP) }`
     - `rt: RtInfo { thread_priority, scheduler ("FIFO"/"DEADLINE"/"OTHER"), cpu_pinned, huge_pages_active }`
     - `telemetry: TelemetrySnapshot { p50_us, p99_us, p999_us, max_us, total_blocks, xruns, drains }`
     - `flags_seen: u64` (OR acumulado de RT_STATUS_* já vistos — main thread o mantém em `on_main_thread`)
  2. Coleta via `load(Relaxed)` em atomics já existentes (`AtomicU32`/`AtomicU64` em `telemetry.rs`, `spsc.rs`). Nenhum novo atomic no hotpath.
  3. `RuntimeSnapshot::capture(processor_or_host: &impl HasRuntimeSnapshot)` — trait com 1 método para CLAP processor e standalone host.
  4. `flags_seen` atualizado **no drain existente** (`on_main_thread` em CLAP, decimação 1-em-16 em standalone); zero custo extra.
  5. Renderização preserva contrato textual: cada campo em linha `chave=valor` (parser-friendly).
- **Critérios de aceitação:**
  - `cargo test --features heap-audit` confirma zero alloc em RT durante captura (toda alloc ocorre no main).
  - Bundle gerado em sessão ativa contém pelo menos: `model.arch`, `audio.sr`, `rt.prio`, `telemetry.p99`, `flags_seen` (hex).
  - Captura completa < 1ms.
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 1.5 dia.

#### Tarefa S9.T05 — Panic hook persiste `DiagnosticBundle` antes do abort 🔥

- **Onde:** `src/main.rs::main` (standalone); `src/clap/plugin/mod.rs` (init do plugin, `DefaultPluginFactory`); novo `src/common/panic_hook.rs`.
- **Problema:** Em hosts C++ (Bitwig, FL Studio), um panic Rust pode terminar o processo sem flush de `log::error!` — bundle perdido. Adicionalmente, panics fora de callbacks FFI ainda perdem info.
- **Solução técnica:**
  1. `pub fn install_panic_hook(component: &'static str)` em `src/common/panic_hook.rs`:
     - Captura `std::panic::PanicHookInfo` (location, message).
     - Tenta `DiagnosticBundle::capture()` (best-effort — pode falhar se runtime estado corrompido; tratar com `catch_unwind`).
     - Persiste em `~/.cache/nam-rs/crash-<unix_ts>-<component>.txt` (atomicamente: write+rename).
     - Encadeia o hook anterior (não substitui — usa `take_hook` + chain).
  2. Standalone chama `install_panic_hook("standalone")` no início do `main`.
  3. CLAP chama em `DefaultPluginFactory::new` (uma vez por processo; idempotente via `OnceLock`).
  4. **Não chamar dentro de threads RT** — o hook executa onde o panic ocorreu, e a coleta inclui I/O. Para tasks RT, o panic já é convertido em `set_flag(RT_STATUS_*)` — hook só é útil para panics fora de `process()`.
- **Critérios de aceitação:**
  - Standalone: `kill -SEGV` durante sessão NÃO dispara o hook (SIGSEGV não passa pelo Rust panic). Panic intencional (`panic!()` em CLI test) cria arquivo `~/.cache/nam-rs/crash-*.txt`.
  - CLAP: panic em GUI thread (não-RT) cria arquivo idem.
  - **Não** ativar para panic durante destruição do host (race com cleanup) — gated por `OnceLock<bool>` indicando "shutdown em progresso".
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 1 dia.

#### Tarefa S9.T06 — Sanitização e política de redação 💡

- **Onde:** `src/common/diagnostics/diagnostic.rs` (renderização do bundle).
- **Problema:** Bundle atual já redige pouco. Paths absolutos podem expor `/home/<user>/...` em logs públicos.
- **Solução técnica:**
  1. Helper `fn redact_path(p: &Path) -> String` substitui prefixo `$HOME` por `~` e `$XDG_RUNTIME_DIR` por `$XDG_RUNTIME_DIR`. Em `--diagnose-full`, retorna path bruto.
  2. `ModelInfo.path_basename` (não path completo) é o default; full path apenas em `--diagnose-full`.
  3. Nunca incluir: conteúdo de pesos, magnitudes de áudio, nomes de usuário/host (já não inclui).
  4. Documentar política em comentário do struct + em `docs/troubleshooting.md` (S9.T07).
- **Critérios de aceitação:**
  - Cobertura de redação consolidada em `tests/diagnostic_bundle.rs` (S9.T08, caso 3): bundle default não contém substring do `$HOME` real.
  - `--diagnose-full` inclui paths absolutos quando explicitamente solicitado.
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

#### Tarefa S9.T07 — Documentação `docs/troubleshooting.md` 💡 [DONE]

- **Onde:** novo `docs/troubleshooting.md`; link em `README.md`.
- **Problema:** Usuário não sabe como gerar/onde encontrar o bundle.
- **Solução técnica:**
  1. Seção "Como obter informações de suporte":
     - Standalone: `nam-rs --diagnose` ou `:diag` no shell interativo.
     - CLAP: botão "Copy Diagnostic" / ícone ℹ na GUI.
     - Crash: arquivos em `~/.cache/nam-rs/crash-*.txt`.
  2. Seção "O que está incluído (e o que NÃO está)" — política de redação (S9.T06).
  3. Seção "Como reportar":
     - Cole o bloco em issue do GitHub.
     - **Para suporte automatizado:** cole no chat acionando a skill `diagnostico` (referência ao workflow `.agents/workflows/diagnostico.md`).
  4. Screenshots/exemplos de bundle redigido.
  5. Atualizar `README.md` com link "Reportando problemas" → `docs/troubleshooting.md`.
- **Critérios de aceitação:** Doc revisto pela skill `documentador`; cobre os 3 cenários (standalone, CLAP, crash).
- **Especialista:** `documentador`.
- **Esforço:** 0.5 dia.

#### Tarefa S9.T08 — Testes de integração do pipeline de diagnóstico ⚠️

- **Onde:** novo `tests/diagnostic_bundle.rs`.
- **Problema:** Sem testes, regressões silenciosas no formato podem quebrar o consumo pela skill `diagnostico`.
- **Solução técnica:**
  1. Teste 1: `DiagnosticBundle::capture()` em ambiente mínimo (sem áudio ativo) — bundle válido e parseável.
  2. Teste 2: Bundle contém todos os campos obrigatórios do contrato com a skill `diagnostico` (Fase 1.1): código de erro (quando aplicável), mnemônico, arch, os, kernel, features, timestamp.
  3. Teste 3: `--diagnose-full` muda a redação; default redige `$HOME`.
  4. Teste 4: Round-trip — parse de regex simples do bundle gerado retorna campos esperados (smoke test de retro-compat).
  5. Teste 5 (feature-gated `heap-audit`): captura em estado ativo simulado não aloca em RT thread.
- **Critérios de aceitação:** 5 testes verdes; `cargo test diagnostic_bundle` < 1s.
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.
