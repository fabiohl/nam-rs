<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Plano de Sprints (Auditoria 2026-05-29 + Pesquisa Avant-Garde 2026-05-29)

> Plano de execução em **duas partes complementares**:
>
> **Parte I — Remediação (Épicos 1–8)** — decorrente da auditoria multi-disciplinar (DSP, SIMD/microarquitetura, modelos NN, plugin CLAP, host PipeWire/RT, loader/segurança) realizada em 29/05/2026 sobre todo o crate `nam-rs`. Foco em correção, soundness, paridade e dívida arquitetural.
>
> **Parte II — Inovações Avant-Garde (Épicos 9–13)** — decorrente do painel `pesquisador-inovador` em 29/05/2026, cobrindo fronteiras de 2026 em microarquitetura (Intel AMX, AVX10.2, ARM SVE2), compressão de modelos (INT8/INT4), kernel real-time (SCHED_DEADLINE, huge pages, eBPF), UX (hot swap com crossfade, IR cabsim, tone matching), portabilidade (Linux ARM64) e observabilidade empírica (differential fuzzing C++↔Rust, PGO/BOLT, HDR histograms).
> Estas inovações são **diferenciadores competitivos** — não corrigem bugs, mas constroem capacidades inéditas no ecossistema NAM em 2026.
> Cada tarefa é atômica, com referências `arquivo:linha` quando aplicável, critérios de aceitação e especialista alvo.
>
> Nota do PO 1: Arquitetura A2 está fora do escopo, ao menos por enquanto. É permitido apenas placeholders e outras medidas para evitar algo que possa se chocar com o A2 mais adiante.
> Nota do PO 2: Sempre assegure ótima cobertura de docsys e comentários rust inline.
> Nota do PO 3: O repositório oficial do NeuralAmpModelerCore está espelhado integralmente em `github.com/NeuralAmpModelerCore/`.
>
> **Legenda de severidade**
>
> - 🔥 Crítico (UB, paridade quebrada, DoS, panics em RT, soundness) — máximo de prioridade.
> - ⚠️ Alto (performance, manutenibilidade, hotpath subótimo, dívida arquitetural).
> - 💡 Médio (otimização, organização, ergonomia, documentação).
> - ✨ Inovação (capacidade nova, diferencial competitivo, UX disruptiva).
>
> **Especialistas alvo** (correspondem a skills disponíveis no projeto)
>
> - `implementador` (engenharia de aplicação Rust idiomático).
> - `revisor-auditor` (este painel, para validação final).
> - `documentador` (atualização de `docs/` e doccomments).
> - `pesquisador-inovador` (fronteira: AMX/AVX10/SVE2, NN compression, RT-OS).

## Notas Operacionais

### Parte I — Remediação

- **Ordem de execução recomendada:** Épico 12 (S21 HDR + diff fuzz primeiro — instrumenta o resto) → 9 (Quantização) → 10 (RT-OS) → 11 (UX) → 13 (Portabilidade & Hardware Especializado).
- **CI/QA gate por Sprint:**

  1. `bash utils/lints.sh` — formatação, clippy strict, feature matrix.

  2. `bash utils/tests-cargo.sh` — unit + integration

  3. `cargo bench inference_bench` — comparar contra baseline; sem regressão > 5%.
- **Convenções:**
  - PR/branch por Tarefa (`feat/S1-T01-bridgeref-soundness`).
  - Commit message inclui referência `[S1.T01]`.
  - Documentação atualizada (skill `documentador`) sempre que arquitetura muda.

### Parte II — Inovações Avant-Garde

- **Ordem de execução recomendada:** Épico 12 (S21 HDR + diff fuzz primeiro — instrumenta o resto) → 9 (Quantização) → 10 (RT-OS) → 11 (UX) → 13 (Portabilidade & Hardware Especializado).
- **Pré-requisitos:**
  - Épicos 1+2 (Parte I) **devem** estar concluídos antes da Parte II — base sólida de soundness e paridade é pré-condição para qualquer otimização agressiva.
  - Hardware de validação: pelo menos uma máquina com **AMX-capable CPU** (Sapphire Rapids, EC2 c7i, ou Granite Rapids). Para ARM: Graviton 4 EC2 ou hardware ARM64 compatível.
  - Kernel PREEMPT_RT 6.x disponível para Épico 10 (S16.T01/T03).
- **Conventions adicionais:**
  - Tasks com tag ✨ requerem documentação em `docs/innovation/<area>.md` com benchmarks empíricos antes do merge.
  - Cada inovação compete com baseline atual em `cargo bench`; merge bloqueado se causar regressão em features default.

---

## Parte II — Inovações Avant-Garde (Pesquisa & Próxima Geração)

Objetivo: capturar 4–20× speedup em hardware Intel Sapphire Rapids+ (AMX, AVX10.2) e expandir alvo para ARM64 (Apple Silicon, Ampere, Graviton), elevando o NAM-rs do estado "AVX-512 VNNI BF16" para o estado-da-arte de 2026.

---

## Épico 9 — Quantização e Compressão de Modelos

Objetivo: reduzir 2–4× a memória de pesos e 2–8× a banda do hotpath via INT8/INT4 quantization moderna (SmoothQuant/AWQ).

### Sprint S15 — INT8/INT4 Weight Quantization

#### Tarefa S15.T01 — INT8 weight quantization SmoothQuant para Conv1D heads ✨⚠️

- **Onde:** `src/loader/dispatcher/wavenet/` (heads de Conv1D — ver módulo `standard.rs` e `dynamic.rs`); novo `src/math/common/int8_quant.rs`; novo `weights_layout = SmoothQuantInt8`.
- **Problema/Oportunidade:** Pesos do `head_weights` (Conv1D 1×1 do output) **dominam memória** em WaveNet Standard (40 KB de pesos vs 8 KB de activations). INT8 weights + FP32 activations (per-channel scale) reduzem 4× memory bandwidth (cache-friendly em L1/L2). SmoothQuant migra outliers de activations para weights via per-channel scaling — proven 99.5% accuracy retention em LLM.cpp e NAM-class workloads.
- **Solução técnica:**

  1. **Treinamento-livre quantization** (post-training): para cada Conv1D head, computar per-channel scale `s_c = max(|W_c|) / 127`, armazenar `Q_W[c,i] = round(W[c,i] / s_c)` como `i8` + scale vector `s_c` como `f32`.

  2. **Kernel `dot_product_int8_avx512`** usando `_mm512_dpbusd_epi32` (AVX-512 VNNI) — 4× speedup vs F32 FMA em throughput INT8.

  3. **AMX path:** `_tile_dpbssd` (S23.T02-style) para LSTM matmul INT8.

  4. **Encoder NAMB v3:** novo `weights_layout = SmoothQuantInt8` que serializa `[Q_W: i8, scales: f32]`. v3 bump justificado.

  5. **Auto-calibração:** durante o `loader/mod.rs`, opcional sweep de input típico (impulse response) para ajustar scales adversariamente.

  6. **Fallback:** se SmoothQuant falha calibração (golden delta > tolerância), reverter para BF16/FP32 com warning.
- **Pré-requisitos (obrigatórios — herdam invariantes da Parte I):**
  - **S3.T03/S3.T04** — disciplina de layout sequencial e padding implícito; SmoothQuant deve usar a mesma estratégia (bloco contíguo `[Q_W: i8 ..., scales: f32 ...]` por camada, padding para múltiplo do bloco SIMD).
  - **S5.T03** (flag `FLAG_HAS_CRC32`) e **S5.T07** (spec NAMB) — a seção `SmoothQuantInt8` deve ser adicionada à spec **antes** da implementação; bump explícito para NAMB v3 com `FLAG_HAS_QUANT_INT8`.
  - **S13.T02** (round-trip) — cobertura obrigatória do novo layout antes do merge.
- **Critérios de aceitação:**
  - Modelo WaveNet Standard quantizado: tamanho do arquivo 60% menor, MSE vs FP32 < 1e-3 em 60s de signal de teste.
  - Benchmark mostra ≥ 30% redução em latência média para WaveNet Standard.
  - Round-trip encode/decode preserva pesos com erro < 1/127, validado via harness estendido de S13.T02.
- **Especialista:** `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

#### Tarefa S15.T02 — INT4 weight packing experimental (AWQ-style) ✨💡

- **Onde:** estensão de S15.T01 para `weights_layout = AwqInt4`.
- **Problema/Oportunidade:** INT4 (4 bits) entrega 8× memory reduction. AWQ (Activation-aware Weight Quantization, Lin et al. 2023) preserva pesos "salientes" em FP16 e quantiza o resto em INT4. Apropriado para WaveNet com layers de magnitude variada (~1% dos pesos contribuem >50% do output).
- **Solução técnica:**

  1. Identificar 1% top-magnitude weights via análise off-line (script `utils/awq-calibrate.py` opcional, ou heuristic Rust).

  2. Layout: `[Q_W: u4 packed nibbles, salient_mask: bitmap, salient_values: f16, scales: f32]`.

  3. Decoder kernel: unpack INT4 → INT8 com LUT, depois INT8 dot product (reusa S15.T01 path).

  4. **Apenas catálogo dinâmico** (não Conv1D estático) — INT4 é override expressivo.
- **Pré-requisitos (obrigatórios):** S15.T01 (path INT8 + scales infra), S5.T07 (spec NAMB v3 com `FLAG_HAS_QUANT_INT4`), S13.T02 (round-trip estendido).
- **Critérios de aceitação:**
  - MSE < 5e-3 para WaveNet Standard quantizado AWQ vs FP32 (tolerância dobrada vs INT8).
  - Tamanho de arquivo 80% menor que FP32.
  - Round-trip encode/decode validado no harness estendido de S13.T02.
  - Feature `awq-int4` em Cargo (default off).
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 3 dias.

---

## Épico 10 — Sistema Operacional & Real-Time Avant-Garde

Objetivo: capturar todo o potencial do kernel Linux 6.x (PREEMPT_RT mainline), reduzir TLB pressure com huge pages, observar a stack via eBPF e PMU, e migrar I/O de modelo para io_uring (async, sem bloqueio de main thread).

### Sprint S16 — Scheduler & Memória

#### Tarefa S16.T01 — Suporte opcional a SCHED_DEADLINE (CBS) ✨⚠️

- **Onde:** `src/standalone/rt_setup/thread.rs` (configuração de scheduler RT); novo CLI flag `--scheduler {fifo,deadline}`.
- **Problema/Oportunidade:** Paper Raspberry Pi 5 + PREEMPT_RT (arXiv 2604.19275, Abr/2026) demonstra que SCHED_DEADLINE bound max-latency em ≤197 µs sob carga heavy, **vs 224 µs de SCHED_FIFO p99**. CBS (Constant Bandwidth Server) garante admission control — impossível starvation. Apropriado para áudio onde "buffer fits in deadline" é uma garantia formal.
- **Solução técnica:**

  1. Param/flag `nam-rs --scheduler deadline` (default mantém FIFO para compat).

  2. `sched_setattr` com `sched_runtime = 80%·block_period`, `sched_deadline = block_period`, `sched_period = block_period`. Calcular dinamicamente após `node.latency` conhecido.

  3. Fallback automático para FIFO se `EBUSY` (admission control rejeitou) ou kernel < 3.14.

  4. Telemetria: log de deadline missed (via `SCHED_FLAG_RECLAIM` + `dl_runtime` query).

  5. Documentar setup em `docs/realtime-tuning.md` (criar): habilitar `sched_rt_runtime_us = -1`, ajustar cgroup cpu.max.
- **Critérios de aceitação:**
  - Em kernel PREEMPT_RT 6.x, flag `--scheduler deadline` ativa; cyclictest-equivalente interno mostra max-latency < FIFO sob `stress-ng --cpu 16`.
  - Smoke test em CI Linux PREEMPT_RT (GitHub Actions runner com kernel custom, ou local).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 2.5 dias.

#### Tarefa S16.T02 — Huge Pages (THP / MAP_HUGETLB) para weights e mirror buffer ✨⚠️

- **Onde:** `src/loader/mod.rs` (alocação de `AlignedVec<u16>` para pesos dinâmicos); `src/dsp/mirror_buf.rs` + `src/dsp/mirror_buf/linux.rs` (mirror buffer, renomeado de `vring.rs` no Épico 1).
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

#### Tarefa S16.T03 — Detecção e tuning específico para PREEMPT_RT kernel ✨💡

- **Onde:** `src/standalone/rt_setup/` (módulo `mod.rs` ou novo `preempt_rt.rs`); novo `is_preempt_rt()` check.
- **Problema/Oportunidade:** PREEMPT_RT mainline (kernel 6.x) tem semântica diferente: spinlocks viram sleeping locks, IRQs threadeadas. Comportamento "ideal" (`SCHED_FIFO prio 99`) pode mudar de "deve" para "pode preempar threaded-IRQ críticos".
- **Solução técnica:**

  1. Checar `/sys/kernel/realtime` ou `uname -v | grep PREEMPT_RT`.

  2. Se PREEMPT_RT: usar prio 80 (deixa headroom para `ksoftirqd` em prio 90+).

  3. Se vanilla: manter prio 90 (sem threaded IRQs).

  4. Habilitar `SCHED_DEADLINE` mais agressivo (S16.T01).

  5. Log informativo `📈 PREEMPT_RT kernel detected; tuning RT_PRIORITY=80`.
- **Critérios de aceitação:** Detecção correta em kernel 6.6-rt; tuning aplicado.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 0.5 dia.

#### Tarefa S16.T04 — perf_event_open: PMU counters em telemetria ✨💡

- **Onde:** novo `src/dsp/perf_counters.rs`.
- **Problema/Oportunidade:** Hoje a telemetria mede latência wall-clock. PMU (Performance Monitoring Unit) entrega **IPC, cache misses (L1/L2/L3), branch mispredictions, page-faults** — diagnóstico cirúrgico de regressões.
- **Solução técnica:**

  1. `perf_event_open` em modo `PERF_TYPE_HARDWARE`, grupo de 4 contadores (CYCLES, INSTRUCTIONS, CACHE_MISSES, BRANCH_MISSES).

  2. `mmap`-based ring buffer para read lock-free do main thread (kernel-side, sample-free reading).

  3. Expor via `dsp_pipeline_test`'s `RtStatus` para CLI/GUI debug overlay.

  4. Feature `pmu-counters` (default off — requer `CAP_SYS_ADMIN` ou `perf_event_paranoid <= 0`).
- **Critérios de aceitação:** `RUST_LOG=info nam-rs --pmu` mostra IPC histogram em sessão de 60s.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

### Sprint S17 — io_uring & async loading

#### Tarefa S17.T01 — Async model loading via io_uring ✨⚠️

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

#### Tarefa S17.T02 — eBPF tracing target para profiling production-grade ✨💡

- **Onde:** novo `utils/ebpf/dsp_latency.bt` (bpftrace script); doc em `docs/observability.md`.

- **Problema/Oportunidade:** Em produção, debugging de glitches específicos exige profiling não-intrusivo. eBPF traces no audio thread sem overhead mensurável; bpftrace one-liners permitem "quem causou esse xrun?".

- **Solução técnica:**

  1. Marcar funções RT-críticas com `#[no_mangle] #[link_section = ".text.rt"]` para uprobe attachment.

  2. `utils/ebpf/dsp_latency.bt`:

     ```bpftrace
     uprobe:nam-rs:nam_clap_processor_process { @start[tid] = nsecs; }
     uretprobe:nam-rs:nam_clap_processor_process /@start[tid]/ {
       @lat = hist((nsecs - @start[tid]) / 1000);
       delete(@start[tid]);
     }
     ```

  3. Doc em `docs/observability.md` com receitas comuns.

- **Critérios de aceitação:** Script roda; histograma sai consistente com telemetria interna; overhead < 0.1%.

- **Especialista:** `pesquisador-inovador`.

- **Esforço:** 1 dia.

---

## Épico 11 — UX Avant-Garde

Objetivo: ir muito além de "carrega modelo + ajusta gain". Construir o engine NAM mais sofisticado em UX, com hot model swap sem dropouts, A/B comparator, IR cabsim integrado, tone matching e controle remoto MIDI/OSC.

### Sprint S18 — Hot Swap & A/B

#### Tarefa S18.T01 — Hot model swap com crossfade ✨🔥

- **Onde:** `src/clap/processor/` (módulo `dsp.rs` — hotpath de processamento); `src/standalone/pw_host/rt_callback.rs`; `src/loader/mod.rs`.
- **Problema/Oportunidade:** Hoje, trocar de modelo causa um silêncio de ~50ms (load + prewarm). Crossfade sample-accurate elimina audible dropout, permitindo **A/B blind comparison** e workflow rápido de tone-hunting.
- **Solução técnica:**

  1. **Reader/Writer pattern de S1.T01 estendido:** introduzir `ModelReader { ptr: NonNull<dyn NamModel>, generation: u64 }` exposto à RT thread; o main thread mantém um `arc-swap::ArcSwap<Box<dyn NamModel>>` (ou equivalente custom lock-free) — **nunca** `&'static mut`. RT acessa via `&*ptr` com lifetime curto, dentro de uma única call de `process()`.

  2. **Double-slot lock-free:** dois slots `[ModelSlot; 2]` indexados por `active_idx: AtomicUsize` (Relaxed). Main thread escreve em `[1 - active_idx]`, RT lê de `[active_idx]`. Swap via single `store(Release)`. Implementar em `src/clap/processor/dsp.rs` (herdando o `NamClapProcessor` do `mod.rs`).

  3. Main thread carrega novo modelo em background (io_uring de S17.T01), prewarm em separate thread.

  4. RT thread detecta `pending_slot.is_loaded()` no início do bloco; inicia crossfade linear de 64 ms.

  5. Durante crossfade: processa input por **ambos** modelos (`old` lido do slot atual + `new` lido do slot pendente), mixa output via `α · new + (1-α) · old`, com `α` rampando de 0→1 ao longo de 64 ms.

  6. **Quiescência antes do drop:** ao fim do crossfade, RT marca `old_model_can_drop = true` (atomic). Main thread espera **pelo menos 2 blocos** depois do swap (period de quiescence) antes de chamar `drop` no Box antigo — garante que nenhuma RT thread retém ainda uma referência ao Box prestes a ser liberado.

  7. **Heap-audit gate:** o `drop` ocorre exclusivamente no main thread (zero allocação/liberação na RT thread, conforme `cargo test --features heap-audit` de S2.T01).

  8. Latência adicional durante crossfade: 64ms (aceitável; opcional).

  9. Param `PARAM_CROSSFADE_MS` (range 0–500, default 64).
- **Pré-requisitos (obrigatórios — herdam invariantes da Parte I):**
  - **S1.T01** (`DspBridgeReader`/`Writer` split, eliminação de `&'static mut`): mesma disciplina aplicada ao slot de modelo. Sem reintroduzir aliasing XOR-mut violado.
  - **S2.T01** (`heap-audit` sem panic): valida que o swap não aloca/libera na RT thread.
  - **S2.T03** (`alive_fence` / `safe_shared`): se a GUI ou um file picker thread interage com o slot, o mesmo padrão de fence deve ser aplicado para evitar UAF durante destruição do plugin.
  - **S4.T04** (`reset(sr, max_buf)` trait): o novo modelo deve ser inicializado via `reset` antes do swap (não confiar em `prewarm` legado).
  - **S17.T01** (io_uring async load): pré-condição para que o load não bloqueie main thread durante > 5ms.
- **Critérios de aceitação:**
  - Trocar de modelo durante reprodução musical: zero dropout audível (validar com soak test 1h).
  - `cargo +nightly miri test crossfade_model_swap` passa sem warning de aliasing.
  - `cargo test --features heap-audit` confirma zero allocs/drops na RT thread durante swap.
  - Stress-test fechando o plugin enquanto crossfade ativo (host destrói entre blocos): sem UAF (validado em CI fuzz `test_gui_drag_drop_fuzz` extendido).
  - Telemetria mostra crossfade duration consistente.
- **Especialista:** `implementador` + `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 3 dias.

#### Tarefa S18.T02 — A/B model comparator (snapshot bank) ✨⚠️

- **Onde:** `src/clap/extensions/state.rs` (schema de persistência, já versionado v1 em S11.T01); novo módulo `src/clap/ab_bank.rs`.
- **Problema/Oportunidade:** Workflow profissional exige A/B blind comparison. Hoje é um swap manual de arquivo — destrutivo. Snapshot bank com 8 slots persistentes permite comparação instantânea (atalho de teclado A/B/1-8).
- **Solução técnica:**

  1. `NamPluginParams` estendido com `Vec<SnapshotSlot>` (8 slots).

  2. Cada slot armazena `model_path: PathBuf`, `gate_db: f32`, `output_db: f32`.

  3. CLAP param `PARAM_ACTIVE_SLOT` (range 0–7, modulação OK = crossfade S18.T01 disparado).

  4. GUI: 8 botões + atalho keyboard.

  5. State versionado (v2 schema sobre v1 de S11.T01).
- **Critérios de aceitação:** Crossfade A/B funciona; state v2 persiste 8 slots.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

### Sprint S19 — DSP Suplementar

#### Tarefa S19.T01 — IR cabsim convolution (uniformly-partitioned FFT) ✨🔥

- **Onde:** novo `src/dsp/ir_cab.rs`.
- **Problema/Oportunedade:** Workflow NAM é "amp + cabinet". Hoje, usuário precisa de plugin separado (Topaz, NadIR). Integrar cabsim com convolução IR (impulse response, .wav de 4096–8192 spl) **eliminado um plugin do chain** e habilitando workflow "amp+cab presets" únicos.
- **Solução técnica:**

  1. **Uniformly-Partitioned Convolution (UPC):** dividir IR em blocos de N=64 amostras; convolve cada bloco via FFT 128-point (já existe `rustfft`); somar com latência total = N.

  2. **Frequency-domain delay line** evita realocação por bloco.

  3. SIMD complex multiply em FFT bins.

  4. **CLAP IO format:** parâmetros `PARAM_IR_PATH` (file picker drag-drop), `PARAM_IR_GAIN`, `PARAM_IR_ENABLED`.

  5. Carregamento async via io_uring (S17.T01).
- **Critérios de aceitação:**
  - Convolução de IR 4096-tap em < 50% do block budget @ 48k/64 spl.
  - Match bit-perfect vs reference convolution (numpy.convolve) com FFT round-trip.
  - GUI: drag-drop file picker para IR (.wav).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 4 dias.

#### Tarefa S19.T02 — Auto LUFS normalization ao trocar modelo ✨💡

- **Onde:** `src/dsp/lufs.rs` (novo); integração em `src/dsp/pipeline/stages.rs` (estágio de output do pipeline).
- **Problema/Oportunidade:** Modelos NAM variam ~20 dB de output entre si — trocar modelo causa **shock de volume**. Auto-LUFS normaliza para −18 LUFS-S (target broadcast) com ramp suave de 200ms.
- **Solução técnica:**

  1. Implementar BS.1770-4 LUFS meter (K-weighting pre-filter + RMS + gate −70 LUFS).

  2. Em swap de modelo, calcular `target_gain = -18 - measured_lufs` over 1s; ramp via `apply_ramp_stereo`.

  3. Param `PARAM_AUTO_LUFS: bool` (default on; usuário pode desligar para A/B blind).
- **Critérios de aceitação:** Trocar entre 4 modelos de catálogo: output LUFS-S converge para -18 ±1 LUFS após 1s.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S19.T03 — Spectrum analyzer pré/pós (visual feedback) ✨💡

- **Onde:** novo `src/clap/gui/spectrum.rs`; integração em `src/clap/gui/ui/mod.rs` (nova zona de spectrum).
- **Problema/Oportunedade:** Visual feedback de "o que o modelo está fazendo no espectro" é altamente educativo e diferencial UX. STFT 2048-point @ 30 Hz refresh; overlay pre/post.
- **Solução técnica:**

  1. Capture ring buffer (~256 ms) de input e output em SPSC.

  2. Main thread (GUI): STFT via `rustfft` 2048-point, Hann window, overlap 75%.

  3. Renderizar via egui_glow como linha + fill, log-frequency axis (20 Hz – 20 kHz).

  4. Toggle visibility via param `PARAM_SPECTRUM_ENABLED`.
- **Critérios de aceitação:** Spectrum render em 30 FPS sem hiccup; identifica trivialmente um cab high-cut a 5 kHz.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S19.T04 — Tone-matching mode (EQ correction em tempo real) ✨🔥

- **Onde:** novo `src/dsp/tone_match.rs`.
- **Problema/Oportunidade:** **Feature killer** — usuário fornece "target tone" (snippet de áudio referência), engine aprende correção EQ em ~5s e aplica em pós-modelo. Combina FFT de target e FFT atual, calcula response, projeta em IIR biquads ou FIR mínimo phase.
- **Solução técnica:**

  1. **Captura target:** 5-10s de audio de referência via drag-drop ou record button.

  2. **Average magnitude spectrum** (Welch's method) de target e current output.

  3. **EQ correction:** `H_corr(f) = |Target(f)| / |Current(f)|`, smoothed em log scale.

  4. **Projeção em filterbank:** 31-band graphic EQ (ISO 1/3 oct) ou 10 IIR biquads peaking.

  5. Aplicar como pós-modelo IIR (RT-safe, < 100 instruções por sample).
- **Critérios de aceitação:** Após tone match, MSE espectral entre output e target < -30 dB em região 100–8000 Hz.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 4 dias.

### Sprint S20 — Controle Remoto & Integração

#### Tarefa S20.T01 — MIDI Learn nativo (CC mapping) ✨⚠️

- **Onde:** novo `src/clap/extensions/note_ports.rs` (já parcialmente existe via `clack-extensions`); UI binding em `src/clap/gui/ui/mod.rs`.
- **Problema/Oportunidade:** Controlar gain/gate/model com pedal MIDI é workflow live essencial. CLAP suporta `note_ports` + `params` via MIDI mapping nativo.
- **Solução técnica:**

  1. Right-click em knob → "MIDI Learn" → próximo CC recebido bind ao param.

  2. Persistir mapping em state (v3 schema).

  3. CLAP event handling no `processor.rs`.
- **Critérios de aceitação:** Pedal MIDI controla gate threshold em DAW.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

#### Tarefa S20.T02 — OSC remote control (standalone) ✨💡

- **Onde:** novo `src/standalone/osc.rs`; flag CLI `--osc-port 9000`.
- **Problema/Oportunidade:** Standalone PipeWire usado em live performances; controlar via TouchOSC / iPad via UDP OSC sobre WiFi local é elegante e mainstream em pedalboard ecosystems.
- **Solução técnica:**

  1. Crate `rosc` (lightweight); thread separado escutando UDP 9000.

  2. Mapear `/nam/gain`, `/nam/gate`, `/nam/model_size` para params.

  3. Bidirecional: enviar telemetria (RT_STATUS, gain reduction meter) de volta para TouchOSC.
- **Critérios de aceitação:** TouchOSC controla nam-rs standalone via WiFi; latência < 20ms.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

---

## Épico 12 — Validação Empírica & Observabilidade Avant-Garde

Objetivo: capturar empiricamente a qualidade do engine via differential fuzzing C++↔Rust, métricas perceptuais (PESQ/STOI/MR-STFT), HDR histograms de latência e otimização compiler-grade (PGO + BOLT).

### Sprint S21 — Differential Validation & Métricas Perceptuais

#### Tarefa S21.T01 — Differential fuzzing C++↔Rust ~~com cargo-fuzz~~ ❌ [CANCELADA]

> **Cancelada por política do projeto:** `cargo-fuzz` requer toolchain `nightly`, que não é utilizado no projeto. Técnicas avançadas de fuzzing estão fora do escopo no momento. Reabrir junto com S5.T08 se a política for revisada.

#### Tarefa S21.T02 — Harness CI de regressão perceptual (ESR/MR-STFT delta tracking) ✨⚠️

- **Pré-condição (auditoria 2026-06-03):** S29.T01 (`Épico 15`, Parte I em `TODO-sprints.md`) **já entrega** as implementações de `compute_esr`, `compute_mr_stft`, e baselines `A2ESR_*`, `NAM_RS_CPP_PARITY_ESR_MAX` em `tests/common/perceptual.rs`. **Esta tarefa não re-implementa essas funções** — apenas consome o módulo público.
- **Onde:** `tests/perceptual_regression.rs` (novo; consumidor da API estabelecida por S29.T01 em `tests/common/perceptual.rs`). Eventualmente estender `tests/cpp_parity.rs` para participar do regression tracking.
- **Problema/Oportunidade:** S29.T01 entrega a fundação métrica (gate estático `ESR < 1e-3`), mas não rastreia **deltas históricos** (regressão entre runs). MSE+SNR em `cpp_parity` é gate absoluto; falta capturar drifts perceptuais sub-threshold que poderiam degradar audibilidade ao longo de várias sprints.
- **Solução técnica:**

  1. `use nam_rs_tests::common::perceptual::{compute_esr, compute_mr_stft, A2ESR_*};` (importar do módulo S29.T01).

  2. CI harness carrega goldens (`tests/fixtures/golden_*.bin`), executa modelo Rust atual, computa `(esr_now, mr_stft_now)`.

  3. Carrega `tests/fixtures/perceptual_baseline.json` (último known-good registrado por sprint anterior — committed quando ESR/MR-STFT melhora).

  4. Fail PR se `esr_now_dB - esr_baseline_dB > 1.0` (regressão ≥ 1 dB) **ou** `mr_stft_now > mr_stft_baseline * 1.10` (regressão > 10%).

  5. Comando manual `cargo run --bin update_perceptual_baseline` re-grava o JSON quando autor confirma que o delta é intencional (ex.: novo kernel SIMD com trade-off aceito).
- **Critérios de aceitação:** Harness roda em `utils/tests-long.sh`; gate de regressão dispara quando intencionalmente induzido (teste com modelo regredido manualmente); baseline JSON é versionado.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.0 dia (reduzido de 1.5 → S29.T01 já entregou as implementações base).

#### Tarefa S21.T03 — HDR Histograms para latência (lock-free, percentile-accurate) ✨⚠️

- **Onde:** `src/dsp/telemetry.rs` (substitui histograma bucket-linear atual).
- **Problema/Oportunidade:** Telemetria atual usa buckets lineares (S6.T01) — boa para count mas péssima para p99/p99.9. HDR Histogram (Gil Tene) usa bucket log-linear: 5 σ acurácia com 10× menos memória; já é o padrão em sistemas low-latency (Aeron, ZGC).
- **Solução técnica:**

  1. Crate `hdrhistogram` ou implementação inline (simple version, ~200 LoC).

  2. RT-safe: record via `fetch_add` em buckets pre-allocados; read em main thread.

  3. Export como Prometheus/OpenMetrics text (S21.T04 opcional).
- **Critérios de aceitação:** p99 e p99.9 reportados com erro < 1% vs cyclictest baseline.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S21.T04 — Continuous benchmark regression bot (criterion + JSON archive) ✨💡

- **Onde:** `.github/workflows/bench-regression.yml`; `benches/`.
- **Problema/Oportunidade:** Sem CI regression bot, slow performance creep passa despercebido. Criterion já produz JSON; arquivar em branch `benchmarks-archive` e comparar via gh-actions.
- **Solução técnica:**

  1. Cada PR roda subset de benches; compara vs `main` baseline; comenta diff no PR.

  2. Falha PR se regressão > 5% em hotpath (inference_bench).
- **Critérios de aceitação:** Bot ativo em PRs; histórico de benches em branch dedicado.
- **Especialista:** `implementador`.
- **Esforço:** 1 dia.

### Sprint S21.D — Suporte ao Usuário & Diagnóstico de Campo (Observability sem regressão) ✨⚠️

> **Contexto e justificativa:** A skill `diagnostico` (vide `.agents/workflows/diagnostico.md`) espera receber um "bloco de suporte" colado pelo usuário contendo código de erro, mnemônico, parâmetros contextuais e info de sistema. Hoje o `Diagnostic::support_block()` (`src/common/diagnostics/diagnostic.rs`, migrado do antigo `diagnostics.rs`) só é gerado em **paths de erro** (`emit`/`emit_warning`). Cenários frequentes ficam descobertos:
>
> - Usuário relata "som baixo" / "dropouts" / "GUI travada" — sem erro tipado, nada para colar.
> - Usuário em hospedeiro CLAP (Bitwig/Reaper/FL Studio) não tem stderr acessível.
> - Crashes em hosts C++ (Bitwig) podem perder o `log::error!` antes do abort.
>
> Esta sprint é **inserida** entre S21 (validação) e S22 (compiler-grade opt) por dependência lógica: HDR Histograms (S21.T03) alimentam o bundle com percentis, e nenhuma tarefa de S22+ depende desta sprint.
>
> **Princípios invariantes** (todas as tarefas):
>
> - **Zero hotpath cost:** coleta exclusivamente via `load(Relaxed)` em atomics já existentes. Nenhum novo flag/counter no `process()`. Bundle gerado on-demand no main thread.
> - **Zero alloc em RT:** toda I/O e formatação no main thread. Panic hook só roda fora do hotpath (após unwind iniciado).
> - **Segurança:** redação default de paths absolutos (`$HOME` → `~`); nunca embarcar conteúdo de pesos/áudio; opt-in `--diagnose-full` para incluir paths completos.
> - **Forward-compat:** o formato textual do bundle preserva contrato consumido pela skill `diagnostico` (Fase 1.1 do workflow). Novos campos são **anexados** em linhas próprias; parsers antigos da IA ignoram silenciosamente.

#### Tarefa S21.D.T01 — Refatorar `support_block()` para `DiagnosticBundle` desacoplado de erro 💡

- **Onde:** `src/common/diagnostics/diagnostic.rs` (atual `support_block` é método privado de `Diagnostic`, migrado de `diagnostics.rs`).
- **Problema:** `support_block()` é privado e exige um `NamErrorCode` para ser construído. Não há API pública para "gerar bundle em estado nominal".
- **Solução técnica:**

  1. Extrair `pub struct DiagnosticBundle { system: SystemInfo, runtime: RuntimeSnapshot, error: Option<ErrorContext> }` em `src/common/diagnostics/diagnostic.rs`.

  2. `impl DiagnosticBundle { pub fn capture() -> Self; pub fn capture_with_error(code, params) -> Self; pub fn render(&self) -> String; }`.

  3. `RuntimeSnapshot` (vazio nesta tarefa — preenchido em S21.D.T04) — placeholder com `Default`.

  4. Refatorar `Diagnostic::support_block` para delegar ao novo `DiagnosticBundle::capture_with_error(...).render()`.

  5. Preservar o cabeçalho textual exato (`──── NAM-rs Diagnostic ...`) para retro-compat com skill `diagnostico`.
- **Critérios de aceitação:** `Diagnostic::emit` produz string byte-idêntica à anterior em paths de erro existentes. Novo `DiagnosticBundle::capture().render()` retorna bloco sem campo de erro.
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

#### Tarefa S21.D.T02 — Comando CLI `--diagnose` no standalone ⚠️

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

#### Tarefa S21.D.T03 — Botão "Copy Diagnostic" na GUI do CLAP ⚠️

- **Onde:** `src/clap/gui/ui/mod.rs` (status bar / nova zona "About"); após refactor de S8.T01 o alvo é `src/clap/gui/ui/mod.rs` ou módulo dedicado.
- **Problema:** Usuário do plugin em DAW não tem acesso ao stderr do host. Sem botão na GUI, impossível obter bundle em hosts C++.
- **Dependência:** S8.T01 já concluído — `ui.rs` foi dividido em `src/clap/gui/ui/` (mod, state, knob, meter, bypass, colors, vsep, simd). O status bar reside em `ui/mod.rs` (função `draw_ui`).
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

#### Tarefa S21.D.T04 — `RuntimeSnapshot` lock-free com estado RT-safe ⚠️

- **Onde:** `src/common/diagnostics/diagnostic.rs` (novo `RuntimeSnapshot`); consumidores em `src/clap/processor/dsp.rs` + `src/clap/processor/events.rs`, `src/standalone/pw_host/rt_callback.rs`, `src/dsp/telemetry.rs`.
- **Problema:** Bundle atual só tem versão + arch + features estáticos. Falta o **estado dinâmico** crítico para diagnóstico: modelo carregado (arquitetura/CH/RF/path basename), SR efetivo, buffer size, contadores de xrun/drain, RT prio aplicada, scheduler ativo (FIFO/DEADLINE — S16.T01), percentis de latência (HDR — S21.T03), histórico recente de RT_STATUS flags.
- **Solução técnica:**

  1. Definir struct `RuntimeSnapshot` com campos:
     - `model: Option<ModelInfo { arch_label, channels, receptive_field, weights_layout, path_basename }>`
     - `audio: AudioInfo { sample_rate, buffer_size, channel_count, host_name (CLAP) }`
     - `rt: RtInfo { thread_priority, scheduler ("FIFO"/"DEADLINE"/"OTHER"), cpu_pinned, huge_pages_active }`
     - `telemetry: TelemetrySnapshot { p50_us, p99_us, p999_us, max_us, total_blocks, xruns, drains }`
     - `flags_seen: u64` (OR acumulado de RT_STATUS_* já vistos — main thread o mantém em `on_main_thread`)

  2. Coleta via `load(Relaxed)` em atomics já existentes (`AtomicU32`/`AtomicU64` em `telemetry.rs`, `spsc.rs`). Nenhum novo atomic no hotpath.

  3. `RuntimeSnapshot::capture(processor_or_host: &impl HasRuntimeSnapshot)` — trait com 1 método para CLAP processor e standalone host.

  4. `flags_seen` atualizado **no drain existente** (`on_main_thread` em CLAP, decimação 1-em-16 em standalone — vide S6.T05); zero custo extra.

  5. Renderização preserva contrato textual: cada campo em linha `chave=valor` (parser-friendly).
- **Critérios de aceitação:**
  - `cargo test --features heap-audit` confirma zero alloc em RT durante captura (toda alloc ocorre no main).
  - Bundle gerado em sessão ativa contém pelo menos: `model.arch`, `audio.sr`, `rt.prio`, `telemetry.p99`, `flags_seen` (hex).
  - Captura completa < 1ms.
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 1.5 dia.

#### Tarefa S21.D.T05 — Panic hook persiste `DiagnosticBundle` antes do abort 🔥

- **Onde:** `src/main.rs::main` (standalone); `src/clap/plugin/mod.rs` (init do plugin, `DefaultPluginFactory`); novo `src/common/panic_hook.rs`.
- **Problema:** Em hosts C++ (Bitwig, FL Studio), um panic Rust pode terminar o processo sem flush de `log::error!` — bundle perdido. Adicionalmente, a auditoria do Épico 1 (`window.rs`) eliminou panics em callbacks FFI, mas **qualquer panic residual fora do callback** ainda perde info.
- **Solução técnica:**

  1. `pub fn install_panic_hook(component: &'static str)` em `src/common/panic_hook.rs`:
     - Captura `std::panic::PanicHookInfo` (location, message).
     - Tenta `DiagnosticBundle::capture()` (best-effort — pode falhar se runtime estado corrompido; tratar com `catch_unwind`).
     - Persiste em `~/.cache/nam-rs/crash-<unix_ts>-<component>.txt` (atomicamente: write+rename).
     - Encadeia o hook anterior (não substitui — usa `take_hook` + chain).

  2. Standalone chama `install_panic_hook("standalone")` no início do `main`.

  3. CLAP chama em `DefaultPluginFactory::new` (uma vez por processo; idempotente via `OnceLock`).

  4. **Não chamar dentro de threads RT** — o hook executa onde o panic ocorreu, e a coleta inclui I/O. Para tasks RT, o panic já é convertido em `set_flag(RT_STATUS_*)` em S2.T01 — hook só é útil para panics fora de `process()`.
- **Critérios de aceitação:**
  - Standalone: `kill -SEGV` durante sessão NÃO dispara o hook (SIGSEGV não passa pelo Rust panic). Panic intencional (`panic!()` em CLI test) cria arquivo `~/.cache/nam-rs/crash-*.txt`.
  - CLAP: panic em GUI thread (não-RT) cria arquivo idem.
  - **Não** ativar para panic durante destruição do host (race com cleanup) — gated por `OnceLock<bool>` indicando "shutdown em progresso".
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 1 dia.

#### Tarefa S21.D.T06 — Sanitização e política de redação 💡

- **Onde:** `src/common/diagnostics/diagnostic.rs` (renderização do bundle).
- **Problema:** Bundle atual já redige pouco. Paths absolutos podem expor `/home/<user>/...` em logs públicos.
- **Solução técnica:**

  1. Helper `fn redact_path(p: &Path) -> String` substitui prefixo `$HOME` por `~` e `$XDG_RUNTIME_DIR` por `$XDG_RUNTIME_DIR`. Em `--diagnose-full`, retorna path bruto.

  2. `ModelInfo.path_basename` (não path completo) é o default; full path apenas em `--diagnose-full`.

  3. Nunca incluir: conteúdo de pesos, magnitudes de áudio, nomes de usuário/host (já não inclui).

  4. Documentar política em comentário do struct + em `docs/troubleshooting.md` (S21.D.T07).
- **Critérios de aceitação:**
  - Cobertura de redação consolidada em `tests/diagnostic_bundle.rs` (S21.D.T08, caso 3): bundle default não contém substring do `$HOME` real.
  - `--diagnose-full` inclui paths absolutos quando explicitamente solicitado.
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

#### Tarefa S21.D.T07 — Documentação `docs/troubleshooting.md` 💡

- **Onde:** novo `docs/troubleshooting.md`; link em `README.md`.
- **Problema:** Usuário não sabe como gerar/onde encontrar o bundle.
- **Solução técnica:**

  1. Seção "Como obter informações de suporte":
     - Standalone: `nam-rs --diagnose` ou `:diag` no shell interativo.
     - CLAP: botão "Copy Diagnostic" / ícone ℹ na GUI.
     - Crash: arquivos em `~/.cache/nam-rs/crash-*.txt`.

  2. Seção "O que está incluído (e o que NÃO está)" — política de redação (S21.D.T06).

  3. Seção "Como reportar":
     - Cole o bloco em issue do GitHub.
     - **Para suporte automatizado:** cole no chat acionando a skill `diagnostico` (referência ao workflow `.agents/workflows/diagnostico.md`).

  4. Screenshots/exemplos de bundle redigido.

  5. Atualizar `README.md` com link "Reportando problemas" → `docs/troubleshooting.md`.
- **Critérios de aceitação:** Doc revisto pela skill `documentador`; cobre os 3 cenários (standalone, CLAP, crash).
- **Especialista:** `documentador`.
- **Esforço:** 0.5 dia.

#### Tarefa S21.D.T08 — Testes de integração do pipeline de diagnóstico ⚠️

- **Onde:** `tests/diagnostic_bundle.rs` (novo).
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

> **Ordem recomendada de execução desta sprint:**
>
> 1. S21.D.T01 (refactor base) → 2. S21.D.T04 (`RuntimeSnapshot` — destrava conteúdo rico) → 3. S21.D.T06 (redação — antes de qualquer UI) → 4. S21.D.T02 (CLI) ‖ S21.D.T03 (GUI CLAP) em paralelo → 5. S21.D.T05 (panic hook) → 6. S21.D.T08 (testes) → 7. S21.D.T07 (docs).
>
> **Esforço total estimado:** ~6 dias (1 dev solo) ou ~3 dias com paralelização CLI/GUI.
>
> **Validação RT-zero-cost:** após implementação, rodar `cargo bench inference_bench` com e sem `--diagnose-full`; delta deve ser < 0.5% (dentro do ruído). Confirmar via `cargo test --features heap-audit` que captura **fora** de erro não aloca em RT.

### Sprint S22 — Compiler-Grade Optimization (PGO + BOLT)

#### Tarefa S22.T01 — Profile-Guided Optimization (PGO) build pipeline ✨⚠️

- **Onde:** `Cargo.toml`; `utils/build-pgo.sh`.
- **Problema/Oportunidade:** Rustc/LLVM PGO instrumenta build → roda workload representativo → coleta profile → rebuilda com `-Cprofile-use`. Tipicamente entrega 5–15% throughput em hotpath. Já standard em Firefox, Chromium.
- **Solução técnica:**

  1. Script multi-passo: build instrumented, roda `inference_bench` + `bench` real de modelos canônicos, coleta `.profraw`, merge, rebuilda release.

  2. Release shipped com PGO opcional via `cargo build --release --features pgo`.
- **Critérios de aceitação:** Benchmark inference reduz ≥ 5% latência média em PGO build vs vanilla release.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S22.T02 — BOLT post-link layout optimization ✨💡

- **Onde:** `utils/build-bolt.sh`.
- **Problema/Oportunedade:** LLVM BOLT é a "última gota": reordena basic blocks no binário linkado para que hot paths fiquem em sequência (melhor L1i utilização). Combinado com PGO, mais 3–8%.
- **Solução técnica:**

  1. Após PGO build, coletar `perf record` em workload representativo.

  2. `llvm-bolt nam-rs -o nam-rs.bolt -data=perf.data --reorder-blocks=cache+ --reorder-functions=hfsort`.

  3. Distribuir binário `.bolt` para release.
- **Critérios de aceitação:** L1i miss rate (`perf stat`) reduz ≥ 20%; latency média -3-8%.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1 dia.

#### Tarefa S22.T03 — Kahan summation em acumuladores críticos ✨💡

- **Onde:** `src/math/gemm/dot.rs`, `dot_4x.rs` (acumuladores horizontal_sum).
- **Problema/Oportunedade:** Em LSTM de muitas amostras, drift de soma FP32 acumula erro de magnitude `~N · eps`. Kahan summation (compensated summation) reduz para `O(1)` em troca de 2 FMAs extras — tolerável fora do tightest inner loop.
- **Solução técnica:**

  1. Apenas em horizontal_sum (1× por bloco GEMM), não no inner FMA.

  2. Manter `compensation: f32` acumulador secundário.
- **Critérios de aceitação:** Drift vs scalar reference em LSTM de 1M amostras reduz ≥ 100×.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1 dia.

---

## Épico 13 — Portabilidade & Arquiteturas de Hardware Especializadas

Objetivo: expandir nam-rs e aproveitar microarquitetura de hardware específica (AMX, AVX10, SVE2, NEON) e plataformas embarcadas ARM64, exigindo setups especiais de build e execução de testes em cloud ou hardware dedicado.

### Sprint S23 — Intel AMX & AVX10.2

#### Tarefa S23.T01 — Abertura do pipeline de build e CI para Intel AMX & AVX10.2 (via Intel SDE / Self-hosted VM) 💡

- **Onde:** `.github/workflows/` (pipelines de build/test/release).
- **Problema:** Atualmente, não há validação automatizada de compilação ou testes funcionais para as novas instruções Intel AMX e AVX10.2 em CI, aumentando o risco de regressões e quebras de build.
- **Solução técnica:**

  1. Configurar etapa de download e cache do **Intel Software Development Emulator (Intel SDE)** no pipeline de CI do GitHub Actions (usando ações como `petarpetrovt/setup-sde` ou script customizado).

  2. Executar a suite de testes unitários e de integração de AMX/AVX10.2 envelopando o binário de teste com `sde64 -spr -- cargo test --features amx-nightly`.

  3. Integrar flags de compilação no pipeline.
- **Critérios de aceitação:**
  - Pipeline de CI compila e passa nos testes unitários com emulação de CPU Sapphire Rapids (AMX) e Diamond Rapids (AVX10.2) com sucesso.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S23.T02 — Backend Intel AMX para LSTM 2-layer e WaveNet Standard (BF16) ✨🔥

- **Onde:** novo módulo `src/math/common/amx_impl.rs`; integração em `src/math/common/dispatch.rs` (novo nível `InstructionSet::Amx_Bf16` acima de `Avx512VnniBf16`, região ~linha 140-163 pós-refatoração).
- **Problema/Oportunidade:** Sapphire Rapids+ executa **`_tile_dpbf16ps`** (DST = A·Bᵀ + DST) em **um único ciclo de 1024 FMAs BF16** (16×64 BF16 × 64×16 BF16 → 16×16 FP32, ~2 TFLOPS BF16 por core a 2 GHz). Para LSTM `2×16` (matmul 32×80 por amostra) e WaveNet Standard (matmul de 16×16 com kernel-3), AMX entrega potencial **10–20×** speedup sobre AVX-512 VNNI BF16. A referência C++ ainda não usa AMX; nam-rs pode ser o **primeiro engine NAM com AMX nativo**.
- **Solução técnica:**

  1. **Layout AMX-friendly do encoder:** novo `weights_layout = AmxTile16x64Bf16` que organiza pesos em tiles de 16 linhas × 64 colunas (= 64 BF16 = 1 KB por tile), padding zero quando necessário. Decoder em `src/loader/dispatcher/lstm.rs` e `src/loader/dispatcher/wavenet/` (módulos `standard.rs`, `dynamic.rs`, `layout.rs`) carrega blocos em `AlignedVec<u16>` 64-aligned.

  2. **Trait `AmxBf16Math: SimdMath`** implementando `fused_add_gemv`, `fused_add_gemm_batch`, etc. Cada kernel:
     - Configurar palette 1 via `_tile_loadconfig()` (uma vez por activate).
     - `_tile_loadd::<TILE_A, STRIDE>(weights_ptr)` para tile A (16×32 BF16).
     - `_tile_loadd::<TILE_B, STRIDE>(input_ptr)` para tile B (32×16 BF16).
     - `_tile_dpbf16ps::<TILE_C, TILE_A, TILE_B>()` — acumula em FP32 no tile C.
     - `_tile_stored::<TILE_C, STRIDE>(output_ptr)` final.

  3. **AMX state preservation:** primeira chamada em activate emite `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` para habilitar XTILEDATA no kernel (Linux ≥5.16). Falha graciosa para fallback Avx512VnniBf16.

  4. **`#![feature(x86_amx_intrinsics)]`** requer nightly até estabilização. Gate via `#[cfg(feature = "amx-nightly")]` em Cargo, default off.

  5. Adicionar `RT_STATUS_AMX_ACTIVE` flag.
- **Pré-requisitos (obrigatórios — herdam invariantes da Parte I):**
  - **S3.T03** (padrão intercalado `[W_l, bias_l, hidden_init_l, cell_init_l]` por camada) — o layout AMX tile-block deve seguir a mesma disciplina sequencial para evitar o tipo de bug encoder↔decoder corrigido em LSTM.
  - **S3.T04** (padding implícito do encoder até múltiplos de bloco SIMD) — tiles AMX exigem 16-row blocks; replicar a estratégia do Interleaved-4 (zero-pad até `ceil(N/16)·16`) ao invés de tail-loops independentes.
  - **S5.T03** (flag `FLAG_HAS_CRC32` explícito) — o novo layout deve setar o flag CRC e nunca confiar em sentinel.
  - **S5.T07** (spec NAMB) — `docs/namb-spec.md` precisa ganhar seção "AmxTile16x64Bf16" antes da implementação, incluindo exemplos hex.
  - **S13.T02 (round-trip)** — cobertura obrigatória do novo layout no harness `tests/namb_v2_roundtrip.rs` (extendido) antes do merge.
  - Decisão: o novo layout dispara **bump explícito de NAMB para v3** (com `FLAG_HAS_AMX_TILE_LAYOUT` no header v3). Documentar em `docs/namb-spec.md` v3.
- **Critérios de aceitação:**
  - Em CPU Sapphire Rapids, dispatcher seleciona AMX; benchmark `inference_bench` mostra ≥8× speedup vs AVX-512 BF16 para LSTM 2×16 e ≥4× para WaveNet Standard.
  - Diferença numérica vs `ScalarRefMath` < 5e-3 (AMX usa BF16 mantissa de 7 bits, tolerância maior justifica-se).
  - `cargo test --features amx-nightly` passa cobertura golden em 4 modelos canônicos.
  - **Round-trip encode→decode do layout `AmxTile16x64Bf16` passa bit-perfect** (estende `tests/namb_v2_roundtrip.rs` de S13.T02).
  - Documentado em `docs/amx-backend.md` (setup XSAVE/permission, palette config, latência/throughput por kernel) e seção dedicada em `docs/namb-spec.md` v3.
- **Especialista:** `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 4–5 dias.

#### Tarefa S23.T03 — Dispatcher AVX10.2 (Diamond Rapids 2026) ✨⚠️

- **Onde:** `src/math/common/dispatch.rs`; novo `avx10_impl.rs` (opcional, pode reusar `avx512_impl` se ISA-equivalente).
- **Problema/Oportunidade:** Intel Diamond Rapids 2026 introduz **AVX10.2** unificando AVX-512 com novos data types: **FP16 nativo (FMA hpfp)**, **FP4** para inferência, e melhor scheduling. oneDNN 2026 já entrega `ONEDNN_MAX_CPU_ISA=AVX10_2_512_AMX_2`. Compilers ainda emergentes; estar pronto cedo é vantagem competitiva.
- **Solução técnica:**

  1. Adicionar `InstructionSet::Avx10_2` ao enum (entre AMX e AVX-512).

  2. Detecção via `is_x86_feature_detected!("avx10.2")` (estabilizar quando intrinsic landar; até lá usar `cpuid` direto).

  3. Substituir `simd_tanh_avx512`/`simd_sigmoid_avx512` por variantes **`_ph` (packed half)** — FMA FP16 nativo elimina 2× conversão F16↔F32 que dominam o hotpath de activations.

  4. Adicionar `dot_4x_fp16_avx10` kernel para Conv1D em WaveNet Feather/Nano (modelos pequenos onde overhead de conversão é proporcionalmente maior).
- **Critérios de aceitação:**
  - Dispatcher detecta AVX10.2 (gated em CPU emulado via SDE até hardware estar disponível).
  - Benchmark FP16 nativo ≥ 2× speedup em activations (sigmoid/tanh) vs AVX-512 BF16 com conversão.
  - Paridade vs `ScalarRefMath` < 1e-3.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2–3 dias.

### Sprint S24 — Portabilidade Linux ARM64 (NEON/SVE2 & Standalone RPi5/Asahi)

#### Tarefa S24.T01 — Abertura do pipeline de build e CI para ARM64 Linux 💡

- **Onde:** `.github/workflows/` (pipelines de build/test/release).
- **Problema:** Não há automação para compilar e testar nativamente ou via cross-compilation o target Linux ARM64, impedindo o deploy confiável em sistemas como Raspberry Pi 5 e servidores baseados em ARM64.
- **Solução técnica:**

  1. Adicionar o target `aarch64-unknown-linux-gnu` à matriz de build e testes do GitHub Actions.

  2. Configurar o ambiente com cross-compilers necessários (`gcc-aarch64-linux-gnu`) ou agentes aarch64 nativos.

  3. Executar a suite de testes unitários e de integração via QEMU user mode runner ou agentes nativos no CI.
- **Critérios de aceitação:**
  - Pipeline de CI compila e passa nos testes com sucesso para o target `aarch64-unknown-linux-gnu`.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S24.T02 — Backend NEON/SVE2 para processadores ARM64 Linux (Ampere, Graviton, Cortex) ✨🔥

- **Onde:** novo módulo `src/math/common/neon_impl.rs` (e `sve2_impl.rs` opcional); integração em `dispatch.rs`.
- **Problema/Oportunidade:** Ampere Altra/Graviton 4 (servidor ARM Neoverse-V2 com SVE2 256-bit) e processadores ARM64 como Cortex-A76/A78 (Raspberry Pi 5) representam alvos fundamentais para Linux. Hoje, nam-rs em ARM rodaria escalar — **inviável** para produção.
- **Solução técnica:**

  1. **NEON baseline:** trait `NeonMath` com kernels:
     - `gemv` usando `vfmaq_f32` (4-lane FMA) com 4 acumuladores.
     - `dot_product_4x` com layout interleaved-4 (já compatível com encoder atual).
     - `tanh/sigmoid` via Padé (S7.T09) — NEON ports diretos. **Nota (Auditoria Épico 4):** Constantes Padé [5,4] já estão centralizadas em `src/math/constants.rs` e são portáveis. Usar `vrecpeq_f32` + Newton-Raphson refinement para o recíproco (análogo ao `_mm256_rcp_ps` + `fnmadd` do AVX2).
     - Conversão F16↔F32 via `vcvt_f16_f32` (ARMv8.2-A FP16).

  2. **SVE2 advanced:** trait `Sve2Math` para Neoverse-V1+/V2 (Ampere, Graviton 4):
     - Vectores de comprimento variável (128–2048 bits, runtime via `svcntw`).
     - `svfmla_f32_z` predicado, eliminando tail loops.
     - `svbfdot_f32` para BF16 dot (ARMv8.6-A) — análogo a `_mm512_dpbf16_ps`.

  3. **Dispatcher:** `#[cfg(target_arch = "aarch64")]` com `std::arch::is_aarch64_feature_detected!("neon")` e `("sve2")`.

  4. **`mirror_buf.rs` portabilidade:** já parcialmente coberto por S1.T04. Em Linux ARM64, `memfd_create` funciona normalmente (fallback `mmap` anônimo para não-Linux já existe em `mirror_buf/fallback.rs`).

  5. **Build matrix CI:** `aarch64-unknown-linux-gnu` em GitHub Actions.
- **Critérios de aceitação:**
  - `cargo test --target aarch64-unknown-linux-gnu` passa com emulação QEMU ou nativa.
  - Paridade numérica `|err| < 5e-4` vs ScalarRefMath.
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 3 dias.

#### Tarefa S24.T03 — Linux ARM64 standalone (Raspberry Pi 5 / Asahi Linux) ✨⚠️

- **Onde:** build matrix CI; `utils/install.sh` para apt+pipewire em Raspbian/Asahi.
- **Problema/Oportunidade:** Raspberry Pi 5 (Cortex-A76, NEON, FP16) ou Asahi Linux em Apple Silicon entregam plataforma "stomp-box" de baixo custo. Combinado com S24.T02 (NEON backend) e S16.T01 (SCHED_DEADLINE), entrega standalone hardware NAM rivalizando DIMEHEAD/Anagram.
- **Solução técnica:**

  1. Cross-compile `aarch64-unknown-linux-gnu`.

  2. PipeWire 0.10 disponível em Debian 12/Ubuntu 22.04 ARM.

  3. Smoke test em Raspberry Pi 5 OS (kernel 6.6 PREEMPT_RT custom build).

  4. Documentar tuning em `docs/raspberry-pi-5.md`: GPU bypass, CPU isolcpus, cpufreq performance.
- **Critérios de aceitação:** RPi5 com guitar interface USB roda nam-rs standalone com latência < 10ms; LSTM 1×16 sem xruns.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 3 dias.

---

## Parte II — Continuação Avant-Garde 2026-06-03 (Épicos 16–19)

> Painel `pesquisador-inovador` em 03/06/2026, após consolidação dos Épicos 9–13 da Parte II e dos Épicos 14–15 (Hotpath Recovery & Cross-Validation v2 — Parte I, residentes em `TODO-sprints.md`).
>
> **Numeração:** Épicos 14–15 já alocados à Parte I; esta continuação da Parte II inicia em **Épico 16** para preservar identificadores únicos e rastreabilidade histórica.
>
> **Restrições reafirmadas:**
>
> - **A2 fora de escopo** — qualquer hook arquitetural deve permanecer como placeholder forward-compat (consistente com S4.T03 e S26.T01).
> - **Linux-only** — proposições x-platform (Windows/macOS) explicitamente recusadas; soluções aproveitam syscalls Linux (`memfd_create`, `landlock`, `mbind`, `clock_nanosleep`, `io_uring`) sem fallback portável.
> - **Formato CLAP único** — LV2/VST3/AU permanecem rejeitados (vide `README.md`, "Native Linux & Modern Architecture").
> - **Zero alocação em hotpath RT** — toda inovação deve respeitar o gate `cargo test --features heap-audit`.
>
> **Foco temático desta rodada:**
>
> 1. **Inferência inteligente sob pressão (Épico 16)** — sparsity estruturada 2:4, JIT kernel specialization via Cranelift, adaptive compute com hysterese.
> 2. **Determinismo RT extremo (Épico 17)** — NUMA-aware allocation, wizard `nam-rs --tune`, TSC-deadline timer alignment, black-box recorder forense.
> 3. **Catálogo inteligente & UX disruptiva (Épico 18)** — perceptual hash + similarity search, pedalboard chain (4 slots), oversampling 2×/4× linear-phase, controle remoto via WebSocket (phone browser), true-peak limiter BS.1770.
> 4. **Multi-instância & superfície de plataforma (Épico 19)** — pesos compartilhados entre instâncias CLAP via `memfd_create`+`memfd_seal`, PipeWire DSP node fast-lane, sandbox seccomp/landlock do loader, distribuição Flatpak/AppImage.
>
> **Pré-requisitos transversais (gate de início desta continuação):**
>
> - Épicos 1–8 (Parte I), 14–15 (Parte I continuação) e 9–13 (Parte II original) **concluídos**.
> - Baseline `cargo bench inference_bench` salvo em `target/criterion/baseline_post_e15/` (para regression bot do S21.T04).
> - HDR Histograms (S21.T03) operacional — alimenta telemetria do black-box recorder e adaptive compute.
> - `RuntimeSnapshot` (S21.D.T04) operacional — base para o forensic dump.
>
> **Convenções (idem Parte II original, com adições):**
>
> - Tasks ✨ requerem benchmark documentado em `docs/innovation/<area>.md` antes do merge.
> - Tasks que afetam multi-instância (Épico 19) exigem teste de 4 instâncias CLAP simultâneas no host `clack-host` (estende S13.T03).
> - Tasks que dependem de capability kernel (PREEMPT_RT, AMX, NUMA, io_uring) detectam fallback gracioso em runtime e logam o motivo via `log::info!` na inicialização.

---

## Épico 16 — Inferência Adaptativa & Compressão Estrutural

Objetivo: complementar a quantização do Épico 9 com **três eixos ortogonais** — sparsity estruturada (reduz operações), JIT kernel specialization (especializa código por modelo carregado) e adaptive compute (degrada graciosamente sob pressão de CPU). Combinados com INT8/INT4 do Épico 9 e AMX/AVX10.2 do Épico 13, formam o stack de inferência mais agressivo do ecossistema NAM em 2026.

### Sprint S25P — Structured Sparsity 2:4 (post-training pruning)

#### Tarefa S25P.T01 — Pruning offline 2:4 estruturado para Conv1D heads e LSTM matmul ✨⚠️

- **Onde:** novo binário utilitário `utils/sparsify-2of4/` (Rust, separado do crate principal); novo `weights_layout = StructuredSparse2of4Bf16` no encoder NAMB v3; integração em `src/math/common/dispatch.rs` (novo `InstructionSet` ortogonal ou flag por kernel).
- **Problema/Oportunidade:** Pesos de Conv1D heads e gates LSTM têm distribuição **fortemente leptocúrtica** (~30–40% dos pesos têm magnitude < 5% do máximo absoluto da camada — comprovado empiricamente em modelos do catálogo padrão NAM). Sparsity estruturada **2:4** (cada grupo de 4 pesos contém exatamente 2 zeros) é o sweet-spot da indústria: NVIDIA Ampere/Hopper, Intel Sapphire Rapids+ via AMX-sparse (Granite Rapids 2026), Apple M-series ML pipelines. **Crucialmente, 2:4 mantém regularidade de acesso à memória** — diferente de unstructured sparsity, que destrói throughput SIMD.
- **Solução técnica:**

  1. **Ferramenta offline** `utils/sparsify-2of4/main.rs`: carrega NAMB v2/v3, aplica magnitude-pruning 2:4 por camada (preserva os 2 maiores absolutos a cada 4 pesos contíguos), re-quantiza para BF16, salva como `*.2of4.nam`.

  2. **Layout:** `[mask_bitmap: u8 packed, vals_nonzero: bf16 packed]` — bitmap reduz 8× vs bool array; vals_nonzero ocupa apenas 50% do storage original. Total: ~52% do original (vs 100% denso, 25% INT8).

  3. **Kernel `dot_2of4_avx512`** (`src/math/common/avx512_impl/sparse_2of4.rs`):
     - Carrega 32 bf16 weights via `_mm512_loadu_si512` → 32 ativações via `_mm512_loadu_si512`.
     - Mask de 16 bits do bitmap → `_mm512_maskz_cvtpbh_ps` extrai apenas non-zeros.
     - `_mm512_fmadd_ps` com 2 acumuladores → 1.7× speedup prático em throughput vs denso BF16 (limitado por banda de memória residual e mask shuffle overhead).

  4. **AMX-Sparse path** (preliminar; gated em `amx-sparse-nightly`): se hardware Granite Rapids detectado e instrução `_tile_dpbf16ps_sparse` disponível, usar tile com mask. Fallback para `dot_2of4_avx512` em Sapphire Rapids/Zen.

  5. **Calibração de tolerância:** ferramenta offline computa ESR pre/post (usando `compute_esr` de S29.T01) e rejeita pruning se `esr_post / esr_pre > 1.5` (regressão > 50% em error-to-signal). Apenas Conv1D heads com gradiente leptocúrtico passam; LSTM gates frequentemente não.
- **Pré-requisitos (obrigatórios):**
  - **S15.T01** (kernel INT8 com per-channel scales) — infraestrutura de quantization-aware layout encoder/decoder reutilizada.
  - **S29.T01** (ESR/MR-STFT) — métrica de validação da degradação.
  - **S5.T07/S15.T01** — spec NAMB v3 com flag `FLAG_HAS_SPARSE_2OF4`.
  - **S13.T02** (round-trip estendido) — cobertura obrigatória do novo layout.
- **Critérios de aceitação:**
  - WaveNet Standard 2:4 pruned: tamanho 48% menor, ESR vs FP32 < 5e-3 (acima do gate de 1e-3 do `cpp_parity` — modelo 2:4 é **opt-in expressivo**, não default).
  - Benchmark mostra ≥ 1.5× throughput no kernel `dot_2of4_avx512` vs `dot_4x` denso (medido isoladamente em `dot_4x_bench` estendido).
  - Round-trip encode/decode bit-perfect (mask + vals).
  - Documentado em `docs/innovation/structured-sparsity.md` com gráfico ESR-vs-savings por arquitetura do catálogo.
- **Especialista:** `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

#### Tarefa S25P.T02 — Auto-pruning seletivo por camada via análise de magnitude ✨💡

- **Onde:** extensão de `utils/sparsify-2of4/` com modo `--auto`.
- **Problema/Oportunidade:** Pruning uniforme 2:4 em todas as camadas pode degradar excessivamente camadas com distribuição uniforme (rare). Análise de **kurtosis por camada** identifica quais aceitam pruning sem perda audível.
- **Solução técnica:**

  1. Calcular kurtosis Fisher (`E[(W-μ)⁴]/σ⁴ − 3`) por camada.

  2. Camadas com kurtosis > 3 (leptocúrticas) → 2:4 pruning.

  3. Camadas com kurtosis ≤ 3 → mantém denso BF16.

  4. Layout misto: header v3 contém `layer_sparsity_mask: u32` (bit por camada).
- **Pré-requisitos:** S25P.T01.
- **Critérios de aceitação:** Modelos com camadas heterogêneas (LSTM 2×16) preservam paridade ESR melhor que pruning uniforme.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

### Sprint S26P — JIT Kernel Specialization (Cranelift)

#### Tarefa S26P.T01 — Avaliação técnica e prototipagem de JIT via Cranelift ✨💡

- **Onde:** novo crate isolado `utils/jit-proto/` (não integra no nam-rs principal nesta tarefa).
- **Problema/Oportunidade:** O dispatcher atual emite um caminho genérico para LSTM `N×CH` com `CH` em [8, 16, 24, 40] e WaveNet com kernels em [3, 5, 8]. Loop unrolling depende de monomorphização via `const generics`, gerando **explosão de combinações pré-compiladas** ou queda para path genérico. **Cranelift** (backend usado pelo Wasmtime, mature, no_std-capable, Rust-native) permite emitir um kernel **especializado por modelo carregado** em ~5–20 ms na fase de `activate()`, eliminando branches e loop-tails residuais. Win prático esperado: 10–25% vs path genérico, **sem inflar binário base**.
- **Solução técnica:**

  1. Em `utils/jit-proto/`, implementar JIT mínimo que: aceita `(arch, channels, kernel_size, dilation)` → emite IR Cranelift para `dot_4x` especializado → executa via `cranelift-jit`.

  2. Comparar latência vs kernel genérico do crate principal em LSTM 2×16, WaveNet Standard CH16.

  3. Medir tempo de compile-time (deve ser < 50 ms para não impactar UX de load).

  4. **Decisão Go/No-Go documentada em `docs/innovation/jit-feasibility.md`:** integrar no crate principal apenas se speedup ≥ 10% **e** compile-time < 50 ms **e** dependency cost ≤ 2 MB (cranelift é ~1.5 MB stripped).
- **Critérios de aceitação:** Protótipo standalone executa LSTM 2×16 JIT com paridade ESR vs genérico < 1e-4; relatório `jit-feasibility.md` revisado por `revisor-auditor`.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 3 dias.

#### Tarefa S26P.T02 — Integração JIT no dispatcher (gated por feature) ✨⚠️

- **Onde:** `src/math/common/dispatch.rs`; novo `src/math/common/jit/`.
- **Problema/Oportunidade:** Após S26P.T01 confirmar viabilidade, integrar JIT como caminho **opcional** acima de AMX/AVX10/AVX-512 quando o modelo é carregado.
- **Solução técnica:**

  1. Feature Cargo `jit-cranelift` (default off).

  2. No `activate(sample_rate, max_buf)` do `NamModel`, se feature ativa e arch reconhecida, emitir kernel JIT em **thread separada** (não bloqueia main thread; usa `tokio` no? Não — usar `std::thread::spawn` simples; resultado armazenado em `OnceLock<Box<dyn JitKernel>>`).

  3. RT thread checa `jit_ready: AtomicBool`; até pronto, usa fallback genérico (zero penalidade durante warm-up).

  4. JIT kernel é `Box<dyn Fn(&mut [f32], &[f32], &Weights) + Send + Sync>` — invocação via vtable já cabe em budget RT.

  5. Telemetria: `RT_STATUS_JIT_ACTIVE` flag.
- **Pré-requisitos:** S26P.T01 (Go), S4.T04 (reset trait), S2.T01 (heap-audit em RT).
- **Critérios de aceitação:**
  - `cargo build --features jit-cranelift,clap-plugin` produz `.so` < 5 MB stripped.
  - Benchmark mostra ≥ 10% redução em latência média WaveNet Standard quando JIT ativo.
  - `cargo test --features heap-audit,jit-cranelift` confirma zero alloc na RT thread após warm-up.
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 3 dias (somente se S26P.T01 = Go).

### Sprint S27P — Adaptive Computation sob pressão

#### Tarefa S27P.T01 — Soft-degrade automático sob CPU pressure (graceful fallback) ✨🔥

- **Onde:** novo `src/dsp/adaptive.rs`; integração em `src/clap/processor/dsp.rs` e `src/standalone/pw_host/rt_callback.rs`.
- **Problema/Oportunidade:** Hoje, quando o host empurra a CPU acima do budget (live performance com 4 plugins + outros tracks, ou laptop em bateria com `cpufreq schedutil` agressivo), o nam-rs **falha hard** com `xrun` audível. **Soft-degrade** = detectar pressão precocemente (p95 do bloco anterior > 70% do budget) e **reduzir graciosamente** a carga (truncate receptive field, skip Conv1D layers profundos, ou desativar oversampling se S18P.T03 ativo) com **crossfade** transparente. Trade controlado: pequena perda de fidelidade vs glitch audível.
- **Solução técnica:**

  1. **Hysteresis com 3 estados:** `Full / Reduced / Minimal`.
     - `Full → Reduced`: três blocos consecutivos com `latency_us > 0.70 * budget_us`.
     - `Reduced → Minimal`: três blocos consecutivos com `latency_us > 0.85 * budget_us`.
     - Caminho reverso: cinco blocos consecutivos abaixo do threshold inferior (histerese assimétrica — descida lenta evita oscilação).

  2. **Estratégias de redução** (por arquitetura):
     - **WaveNet:** `Reduced` desativa últimas N_dilation_layers (configurable, default 25% dos layers); `Minimal` desativa 50%.
     - **LSTM:** `Reduced` mantém apenas primeira camada (2×16 → 1×16); `Minimal` skip total + passa input com gain compensado.

  3. **Crossfade** entre estados (32 ms linear ramp, similar ao S18.T01 hot swap, mas **intra-modelo**).

  4. **Telemetria:** `RT_STATUS_DEGRADE_REDUCED` e `RT_STATUS_DEGRADE_MINIMAL` flags; counter `degrade_transitions_total`.

  5. **Param** `PARAM_ADAPTIVE_COMPUTE: enum { Off, Conservative, Aggressive }` (default Conservative no CLAP plugin; Off no standalone — usuário standalone tipicamente já tem sistema tunado).

  6. **UX feedback:** GUI ícone discreto em status bar quando `Reduced/Minimal` ativo, com tooltip explicativo.
- **Pré-requisitos:**
  - **S21.T03** (HDR Histograms) — fonte da estatística p95.
  - **S6.T01** (telemetria lock-free `fetch_add`).
  - **S4.T04** (reset trait) — necessário para reset interno do estado RNN ao mudar de variante.
  - **S18.T01** (hot swap crossfade) — reusa máquina de crossfade.
- **Critérios de aceitação:**
  - Stress test `stress-ng --cpu 16 --cpu-load 90` durante 60 s com nam-rs ativo: zero xruns audíveis; transição para `Reduced` detectada em telemetria; retorno a `Full` quando stress termina.
  - Soak test 1h em laptop alimentado por bateria: nam-rs degrada graciosamente quando CPU thermal throttle ativa.
  - `cargo test --features heap-audit` zero alloc na transição.
- **Especialista:** `pesquisador-inovador` + `implementador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

---

## Épico 17 — Determinismo RT Extremo & Forensics

Objetivo: empurrar o teto de determinismo do Linux RT para sub-100 µs jitter consistente, automatizar setup de sistema (eliminando a barreira de entrada "tunar Linux"), e ganhar capacidade forense pós-glitch para diagnóstico definitivo.

### Sprint S28P — NUMA awareness & system tuning

#### Tarefa S28P.T01 — NUMA-aware weight allocation & thread pinning ✨⚠️

- **Onde:** `src/loader/mod.rs` (alocação de pesos); `src/standalone/rt_setup/affinity.rs` (pin de thread); novo `src/standalone/rt_setup/numa.rs`.
- **Problema/Oportunidade:** Em workstations dual-socket Xeon, Threadripper Pro e EPYC, alocações default (`Vec`/`malloc`) podem cair em nó NUMA **remoto** ao thread RT, causando ~2–3× latência de memória (remote node access cost). Para modelos com >100 KB de pesos hot em hotpath, isso é p99 killer.
- **Solução técnica:**

  1. Crate `hwloc-sys` ou wrapper direto sobre `numa.h` (libnuma).

  2. **Detecção:** ler `/sys/devices/system/node/node*/cpulist` para mapear CPU → NUMA node.

  3. **Strategy:** após `affinity.rs` pinar thread RT em CPU `C`, descobrir `node_of(C)` e:
     - Alocar pesos via `mmap` + `mbind(MPOL_BIND, node_of_C)`.
     - Alocar mirror buffer (`src/dsp/mirror_buf.rs`) idem.
     - Pré-faulting via `madvise(MADV_WILLNEED)` para evitar page fault em primeiro toque.

  4. **Fallback** silencioso para alocação default em sistemas single-node (sem ABI break).

  5. **Telemetria:** `RT_STATUS_NUMA_LOCAL` flag (set se confirmado same-node).

  6. Documentar em `docs/numa-tuning.md` (criar) com receitas para EPYC/Threadripper/dual Xeon.
- **Pré-requisitos:**
  - **S16.T02** (huge pages allocator) — compõe: huge pages + NUMA-bind via `mmap(MAP_HUGETLB) + mbind`.
  - **S6.T01** (telemetria flag-based).
- **Critérios de aceitação:**
  - Em sistema dual-socket Xeon: `perf stat -e node-loads,node-load-misses` em DSP thread mostra ≥ 95% local-node accesses (vs ~50% pré-tuning).
  - p99 latency reduz ≥ 10% em modelos grandes em hardware multi-socket.
  - Fallback silencioso em laptops single-socket.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2.5 dias.

#### Tarefa S28P.T02 — Wizard `nam-rs --tune`: auto-diagnóstico e sugestão de tuning ✨⚠️

- **Onde:** novo `src/standalone/tune_wizard.rs`; flag CLI `nam-rs --tune` (interativo) e `nam-rs --tune-report` (não-interativo, imprime JSON).
- **Problema/Oportunidade:** Hoje, extrair o máximo de latência de Linux exige conhecimento expert: `isolcpus`, `nohz_full`, `rcu_nocbs`, `cpufreq governor`, `irqbalance`, `tuned-adm profile latency-performance`, `sched_rt_runtime_us`, swappiness, cgroup audio. **Barreira de entrada é a #1 razão de "RT no Linux é difícil"**. Wizard interativo audita estado atual, identifica gaps e sugere comandos com explicação.
- **Solução técnica:**

  1. **Auditor (read-only)**: checa 12 dimensões — `cpufreq.governor`, `sched_rt_runtime_us`, `irq_affinity`, `vm.swappiness`, `kernel.timer_migration`, `kernel.numa_balancing`, `/proc/cmdline isolcpus/nohz_full`, `/sys/kernel/realtime`, PipeWire `quantum` / `rate`, presença de `rtkit`, group `audio` membership, `ulimit -r`.

  2. **Reporter:** print colorido (reusa `src/standalone/colors.rs`) com ✅/⚠️/❌ por checagem + explicação curta.

  3. **Suggester:** emite comandos prontos (com `sudo` quando necessário) e **dry-run** (não executa nada sem confirmação explícita do usuário).

  4. **Profile generator (opcional, com confirmação):** gera `/etc/tuned/profiles/nam-rs/tuned.conf` que é um profile `tuned` declarativo. Usuário executa `sudo tuned-adm profile nam-rs` para aplicar de forma reversível.

  5. **Modo `--tune-report` (JSON):** emite estado atual + recomendações em JSON estruturado, parseável (alimenta CI/scripts de telemetria de campo, e bundle de S21.D quando aplicável).
- **Pré-requisitos:**
  - **S21.D.T04** (`RuntimeSnapshot`) — reusa estrutura de coleta de runtime info; `tune-report` complementa o bundle.
- **Critérios de aceitação:**
  - `nam-rs --tune` em sistema "vanilla Ubuntu 25.10" identifica corretamente 12/12 dimensões; sugestões são tecnicamente válidas (revisadas por `revisor-auditor`).
  - Profile `tuned` gerado é reversível via `sudo tuned-adm profile balanced`.
  - `--tune-report` JSON valida contra schema em `docs/tune-report-schema.json`.
  - Doc `docs/realtime-tuning.md` (do S16.T01) referencia o wizard.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 3 dias.

### Sprint S29P — TSC-deadline & Black-box Recorder

#### Tarefa S29P.T01 — TSC-deadline timer alignment para wake-ups sub-µs ✨💡

- **Onde:** `src/standalone/rt_setup/tsc.rs` (já existente — extensão); integração em `src/standalone/pw_host/rt_callback.rs` (loop de polling).
- **Problema/Oportunidade:** Wake-ups de `clock_nanosleep` no kernel padrão têm jitter de ~10–30 µs (HPET ou LAPIC timer drift). **TSC-deadline timer** (Intel desde Westmere, AMD desde Zen 2) entrega wake-ups com jitter < 1 µs — usado por DPDK e HFT. Já há esqueleto em `tsc.rs`; falta integrar com loop de polling/wake do RT path.
- **Solução técnica:**

  1. Estender `tsc.rs`: helper `tsc_deadline_sleep_until(target_tsc: u64)` que usa `__rdtscp` + busy-spin curto (~500 ns) quando `target_tsc - rdtsc() < SPIN_THRESHOLD`, senão `clock_nanosleep(CLOCK_MONOTONIC, ABSTIME, ...)` convertendo TSC→ns.

  2. **TSC frequency calibration** (1× em activate, custo ~10 ms): `__rdtsc()` antes/depois de `clock_gettime(CLOCK_MONOTONIC, 1ms)` para estabelecer ratio.

  3. **Invariant TSC check:** `cpuid leaf 0x80000007 EDX bit 8` — se invariant não suportado, fallback silencioso para `clock_nanosleep` puro.

  4. **Aplicação:** loops de wake/poll no rt_callback path (apenas onde já há `clock_nanosleep` ou sleep curto); não cria novos threads de polling.

  5. Telemetria: `RT_STATUS_TSC_DEADLINE_ACTIVE` flag.
- **Pré-requisitos:**
  - Hardware com invariant TSC (todos x86_64 desde 2010).
- **Critérios de aceitação:**
  - cyclictest-style interno: histograma de jitter mostra p99 < 5 µs com TSC-deadline ativo (vs p99 ~25 µs sem).
  - Soak 30 min: zero drift acumulado (TSC reconciliado periodicamente vs CLOCK_MONOTONIC).
  - Fallback silencioso testado em CPU sem invariant TSC (forçado via cpuid mask em VM de teste).
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S29P.T02 — Black-box recorder: ring buffer audio+telemetria + dump on glitch ✨🔥

- **Onde:** novo `src/dsp/blackbox.rs`; integração em `src/clap/processor/dsp.rs`, `src/standalone/pw_host/rt_callback.rs`, `src/dsp/telemetry.rs`.
- **Problema/Oportunidade:** **Game changer para debugging de campo.** Usuário relata "som rachou aos 23 min do show de ontem" — hoje, nada pode ser feito. **Black-box recorder** mantém em RAM um **ring buffer rolante de últimos 5 s** de input + output + telemetria (timestamps, RT_STATUS flags, latência por bloco). Ao detectar glitch (xrun, drain, `RT_STATUS_*` de erro, ou comando explícito do usuário), **persiste forensicamente** em `~/.cache/nam-rs/blackbox-<unix_ts>.{wav,json}` para análise pós-mortem. Trickle-up natural do work feito em S21.D (DiagnosticBundle).
- **Solução técnica:**

  1. **Ring buffer** lock-free em shared memory: 5 s × 96 kHz × 2 ch × 4 B = ~3.8 MB (input) + 3.8 MB (output) + 64 KB telemetria. Alocado via `MAP_HUGETLB` (S16.T02) se disponível, fallback page-aligned.

  2. **Recording em RT thread:** **somente** writes lock-free de input/output blocks no ring (já existem buffers — apenas duplicação SPSC copy `memcpy`-friendly). Zero alocação. Custo: < 0.5% CPU overhead (medido empiricamente, validar contra Critério de Aceitação 3).

  3. **Trigger de dump (main thread):** dispara quando:
     - `RT_STATUS_DRAIN` ou `RT_STATUS_XRUN` ou `RT_STATUS_DEGRADE_MINIMAL` (S27P.T01) recém-observado.
     - Comando explícito `:blackbox` no CLI standalone, ou botão "Save blackbox" na GUI CLAP.
     - Panic hook de S21.D.T05 (forensics no crash).

  4. **Dump async via io_uring (S17.T01):** main thread serializa ring → WAV stereo (input track + output track interleaved) + JSON sidecar com timestamps, RT_STATUS history, p50/p99/p999 HDR histogram (S21.T03), modelo ativo, audio config. Arquivo escrito em ~50 ms (não-bloqueante).

  5. **Privacy gate:** **opt-in via `--blackbox` flag (standalone)** ou param `PARAM_BLACKBOX_ENABLED: bool` (CLAP, default off). Áudio do usuário é dado sensível.

  6. **Retention policy:** auto-delete arquivos > 7 dias em `~/.cache/nam-rs/blackbox-*.wav` na inicialização (com log informativo). Configurável via `--blackbox-retention-days`.

  7. **Sanitização:** WAV não contém metadados de usuário; JSON respeita política de redação de S21.D.T06.
- **Pré-requisitos:**
  - **S21.T03** (HDR Histograms) — fonte de estatísticas para o JSON sidecar.
  - **S21.D.T04** (`RuntimeSnapshot`) — embed no JSON.
  - **S21.D.T05** (panic hook) — extensão: panic hook chama `blackbox::dump_forensic()` antes de persistir bundle.
  - **S17.T01** (io_uring async) — para dump sem bloquear main.
  - **S16.T02** (huge pages) — alocação preferencial do ring.
  - **S6.T05** (telemetry decimation) — taxa de scrape do RT_STATUS para o ring.
- **Critérios de aceitação:**
  - Soak test 1h: ao injetar artificialmente `RT_STATUS_DRAIN` via comando debug, ring é persistido em < 100 ms; WAV abre em Audacity com os 5 s anteriores ao glitch.
  - JSON sidecar contém `p99_us`, `flags_history`, `model.arch`, `audio.sr`, timestamps monotônicos.
  - `cargo bench inference_bench` com `--blackbox` ativo: overhead < 1% vs baseline (gate de regressão).
  - `cargo test --features heap-audit` zero alloc em RT durante recording (ring pre-allocado).
  - Privacy: por default, blackbox **não** captura áudio; toggle explícito do usuário documentado em `docs/troubleshooting.md` (S21.D.T07 estendido).
- **Especialista:** `pesquisador-inovador` + `implementador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

---

## Épico 18 — Catálogo Inteligente, Workflow Pro & Qualidade Sonora

Objetivo: transformar nam-rs de "player de modelos" em **assistente criativo**. Busca por similaridade perceptual, encadeamento pedalboard multi-NAM+IR, oversampling profissional, controle remoto via smartphone e true-peak limiter — features que **definem produtos profissionais** (Neural DSP, Kemper) e que estão ausentes do ecossistema open NAM hoje.

### Sprint S30P — Catálogo Inteligente

#### Tarefa S30P.T01 — Perceptual hash & similarity search no catálogo de modelos ✨🔥

- **Onde:** novo `src/loader/perceptual_hash.rs`; sidecar `.namhash` por arquivo `.nam`/`.namb`; novo módulo `src/clap/gui/catalog_browser.rs` (e standalone `src/standalone/catalog.rs`).
- **Problema/Oportunidade:** Usuários típicos têm **dezenas a centenas de modelos NAM** baixados do ToneHunt. Encontrar "aquele clean que eu gostei semana passada" é frustrante — nomes são inconsistentes ("MyAmp_v3_final2.nam"). **Solução:** computar um **fingerprint espectral** por modelo (one-time, offline, ~200 ms por modelo), permitir buscas:
  - **"More like this"** — cosine similarity em catálogo local.
  - **2D tone map** — UMAP/t-SNE 2D dos fingerprints, navegável na GUI.
  - **Drag-drop audio → "find similar amp"** — fingerprint do áudio fornecido, retorna top-K modelos.
- **Solução técnica:**

  1. **Fingerprint:** alimentar modelo com **stress signal v2** (S28.T01 — chirp + noise + harmonics, 1 s @ 48 kHz); capturar output; computar:
     - Spectral envelope: FFT 8192, log-magnitude, smoothed em 1/3 oct (31 bandas ISO).
     - THD ratio em fundamental 220 Hz, 440 Hz, 880 Hz.
     - Compressor characteristic: gain-reduction-vs-input em 6 níveis de gain.
     - **Total: ~64 floats por modelo** (256 B) — armazenado em sidecar `<model>.namhash` (cached).

  2. **Cache:** ao indexar catálogo, computar fingerprints faltantes em background thread; persistir em `~/.cache/nam-rs/catalog-index.bin` (binário versionado, mmap-friendly).

  3. **Similarity search:** cosine similarity em `f32[64]` é < 1 µs/modelo — busca em 1000 modelos < 1 ms. Cabe em main thread.

  4. **UMAP 2D (offline, opcional):** binário `utils/umap-catalog/` projeta fingerprints em 2D; salva em `~/.cache/nam-rs/catalog-map.json`. GUI renderiza scatter plot navegável (clique no ponto → carrega modelo).

  5. **GUI integration:** new tab "Catalog" em `src/clap/gui/ui/mod.rs` com:
     - Search box ("amp like: `<model>`").
     - 2D tone map (renderer via `egui_glow`).
     - Sort by tags inferidos do fingerprint (Clean/Crunch/Lead/Bass — clustering simples).

  6. **Drag-drop audio:** estende `rfd` (já dependência) — usuário arrasta `.wav`/`.flac` de referência; pipeline computa fingerprint do áudio (estimando estágio amp a partir de signal); retorna top-K modelos.
- **Pré-requisitos:**
  - **S28.T01** (Stress Signal v2) — fonte canônica do impulse de fingerprinting.
  - **S17.T01** (io_uring async load) — indexação background sem bloquear UI.
  - **S29.T01** (`compute_esr`, `compute_mr_stft`) — bibliotecas reusadas para spectral analysis.
- **Critérios de aceitação:**
  - Catálogo de 100 modelos: indexação completa em < 30 s background; subsequente busca "more like this" retorna top-10 em < 5 ms.
  - Manual evaluation: para 10 modelos seed (variados — clean Fender, crunch Marshall, hi-gain Mesa), top-3 similares são manualmente classificados como "tonelly relacionados" por revisor humano (revisão pelo `pesquisador-inovador`).
  - UMAP 2D agrupa modelos clean/crunch/hi-gain em clusters visualmente separáveis.
  - Sidecar `.namhash` é portável (formato binário versionado documentado em `docs/perceptual-hash-format.md`).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 5 dias.

### Sprint S31P — Pedalboard Chain & Qualidade

#### Tarefa S31P.T01 — Pedalboard chain (4 slots NAM/IR/null serial) ✨⚠️

- **Onde:** novo `src/dsp/pedalboard.rs`; integração em `src/dsp/pipeline/mod.rs` e `src/clap/processor/dsp.rs`; UI em `src/clap/gui/ui/pedalboard.rs` (novo).
- **Problema/Oportunidade:** Workflow guitar real é **pedal → preamp → power amp → cab** = 3–5 estágios em série. Hoje, encadear isso exige hospedeiro DAW com 4 instâncias CLAP de nam-rs em série — pesado e desencorajado em standalone PipeWire. **4 slots reorderable internos** com `NAM | IR | Null` em cada, gerenciamento centralizado de estado, GUI única.
- **Solução técnica:**

  1. `pub struct PedalboardSlot { kind: SlotKind, enabled: bool, gain_db: f32, ... }` onde `SlotKind = { Nam(Box<dyn NamModel>), Ir(IrConvolver — S19.T01), Empty }`.

  2. `pub struct Pedalboard { slots: [PedalboardSlot; 4], reorder_perm: [u8; 4] }` — `reorder_perm` é vetor de índices para reordenar sem mover dados.

  3. **Processing:** `process(buf)` itera `reorder_perm`; cada slot ativo processa in-place.

  4. **Hot swap por slot:** reusa S18.T01 (crossfade 64 ms intra-slot).

  5. **State versionado v4** (sobre v3 de S20.T01).

  6. **UI:** zone nova no GUI mostra 4 slots como "pedals" arrastáveis (drag-to-reorder); cada slot clicável abre file picker para modelo/IR.

  7. **MIDI bindings:** estende S20.T01 com mapeamento de `PARAM_SLOT_N_ENABLED` (4 toggles).

  8. **Latency reporting:** soma latências dos IRs ativos (S19.T01) + 0 para NAMs (eles são block-aligned); reporta agregado via `clap_latency` ext.
- **Pré-requisitos:**
  - **S18.T01** (hot swap) — reuso direto da máquina crossfade.
  - **S19.T01** (IR cabsim) — slot IR.
  - **S20.T01** (MIDI learn) — bindings.
  - **S26P.T01** (JIT, opcional) — múltiplos NAMs em série amplificam ganho do JIT.
- **Critérios de aceitação:**
  - 3 NAMs + 1 IR encadeados em LSTM 1×16: latência total ≤ 5 ms @ 48k/64 spl em hardware AVX-512.
  - Reordenação via drag-drop preserva sound (zero clicks) graças ao crossfade.
  - State v4 persiste configuração; round-trip save/load em DAW.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 4 dias.

#### Tarefa S31P.T02 — Oversampling 2×/4× linear-phase polyphase FIR ✨⚠️

- **Onde:** novo `src/dsp/oversampler.rs`; integração em `src/dsp/pipeline/stages.rs` (estágios `Upsampler` antes do modelo + `Downsampler` depois).
- **Problema/Oportunidade:** Modelos NAM (especialmente high-gain) introduzem harmônicos que **aliasam** quando processados a SR nativo (44.1k/48k). Resultado: artefatos audíveis em frequências altas (efeito "tweet"). **Oversampling 2×/4×** processa o modelo em SR maior, aliasing fica fora da banda audível, downsample filtra. Linear-phase preserva transient — preferido para mastering; minimum-phase preserva CPU — preferido para live.
- **Solução técnica:**

  1. **Polyphase FIR** baseado em `src/dsp/sinc_kernel.rs` (já existe — reuso direto). Designs:
     - 2× LP: 64 taps, transition 0.45–0.5 fs, -100 dB stopband.
     - 4× LP: 128 taps, transition 0.45–0.5 fs/4, -100 dB.

  2. **Linear-phase vs minimum-phase toggle:** param `PARAM_OS_PHASE: enum { LinearPhase, MinPhase }`.
     - Linear-phase introduz latência (FIR length / 2) = ~32 spl @ 48k para 2×; reportado via `clap_latency`.
     - Min-phase tem latência zero mas distorce phase em transients.

  3. **SIMD:** polyphase decimation/interpolation já está estabelecido em `resampler.rs` — usar mesmo kernel SIMD `convolve_stereo_dual` (S7.T03) e `convolve_mono_dual` (S25.T05).

  4. **Param** `PARAM_OVERSAMPLE: enum { Off, 2x, 4x }` (default Off — opt-in, CPU cost 2–4×).

  5. **Combinação com pedalboard (S31P.T01):** oversampling envolve o pedalboard inteiro (não cada slot — economiza filtros redundantes).

  6. **Telemetria:** `RT_STATUS_OS_2X` / `RT_STATUS_OS_4X` flags.
- **Pré-requisitos:**
  - **S7.T03** (convolve_stereo_dual), **S25.T05** (convolve_mono_dual) — kernels SIMD.
  - **S31P.T01** (pedalboard, opcional) — escopo de envelopamento.
- **Critérios de aceitação:**
  - Stress signal v2 (S28.T01) processado a 2× OS: aliasing energy em região 18–24 kHz (Nyquist sub-harmonic) reduz ≥ 30 dB vs no-OS.
  - p99 latency @ 4× OS ≤ 4× p99 @ no-OS (escala linear esperada).
  - Linear-phase mostra symmetric impulse response em test fixture.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 3 dias.

#### Tarefa S31P.T03 — True-peak limiter BS.1770 (inter-sample peak detection) ✨💡

- **Onde:** novo `src/dsp/true_peak.rs`; integração após `Output gain` em `src/dsp/pipeline/stages.rs`.
- **Problema/Oportunidade:** Output digital pode ter peaks **inter-sample** que excedem 0 dBFS quando convertidos a DAC analógico (clipping audível em equipamento downstream). BS.1770-4 define true-peak: 4× oversampled peak detection. Atual nam-rs não protege contra isso.
- **Solução técnica:**

  1. 4× polyphase upsampler (reusa S31P.T02 kernels).

  2. Detector de peak em domain oversampled.

  3. Soft-knee limiter (3-stage release: 1 ms, 50 ms, 500 ms) atua apenas quando true-peak > -1 dBTP (default; configurável).

  4. Param `PARAM_TRUE_PEAK_DBTP: f32` (default -1.0, range -3.0 to 0.0).

  5. Compatível com pedalboard — sempre último estágio do chain.
- **Pré-requisitos:** S31P.T02 (kernels OS).
- **Critérios de aceitação:** Stress signal v2 processado: true-peak medido por ferramenta externa (`ffmpeg -af ebur128=peak=true`) confirma ≤ -1.0 dBTP.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

### Sprint S32P — Controle Remoto via Smartphone

#### Tarefa S32P.T01 — WebSocket remote control surface (smartphone/tablet via WiFi local) ✨⚠️

- **Onde:** novo `src/standalone/web_remote.rs`; flag CLI `--web-remote-port 9090`; assets HTML/JS embutidos via `include_str!`.
- **Problema/Oportunidade:** OSC (S20.T02) exige app dedicado (TouchOSC). **WebSocket + página HTML servida pelo standalone** = qualquer smartphone com browser controla nam-rs via WiFi local, sem instalar nada. Use case crítico: ajustar gain/preset durante ensaio sem ir até o laptop.
- **Solução técnica:**

  1. Crate `tungstenite` (sync WebSocket, ~50 KB, zero deps pesadas) + `tiny_http` (HTTP server mínimo, ~20 KB) — ou implementação inline (~300 LoC) para zero deps adicionais.

  2. Thread dedicada (não-RT) serve em `0.0.0.0:9090`:
     - `GET /` → HTML estático (single-page, ~30 KB).
     - `GET /control.js` → JS estático.
     - `WS /ws` → bidirecional: client → param updates; server → telemetria (gain reduction meter, p99 latency).

  3. **Auth:** token único por sessão (4-char base32, mostrado no terminal ao subir). URL completo: `http://laptop.local:9090/?token=ABCD`.

  4. **Restrição** padrão: bind em interface LAN only (não 0.0.0.0; usuário pode override com `--web-remote-bind 0.0.0.0`). Doc warning em README.

  5. **UI mobile-first:** layout responsivo, knobs grandes (touch-friendly), VU meters, lista de presets/snapshots (S18.T02).

  6. **Latência:** WS round-trip < 30 ms em WiFi local 5 GHz; param updates aplicados via SPSC bridge no main thread (não na RT thread).
- **Pré-requisitos:**
  - **S20.T02** (OSC remote) — mesmo registry de param↔channel reusado.
- **Critérios de aceitação:**
  - Smartphone Android/iOS via Chrome/Safari acessa `http://laptop.local:9090/`, controla gain e seleciona modelos.
  - Bind default em interface LAN; tentativa de bind em `0.0.0.0` sem flag explícita logged como warning.
  - Demo screencast em `docs/web-remote.md`.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 4 dias.

---

## Épico 19 — Multi-Instância & Plataforma Linux

Objetivo: explorar o que **só Linux permite** — `memfd` para compartilhar pesos entre instâncias (DAW com 4 amps = 1 cópia de weights), fast-lane na PipeWire graph, sandbox de loader para defesa em profundidade, distribuição via Flatpak/AppImage para zero-friction em qualquer distro.

### Sprint S33P — Shared Weights (DAW multi-instance optimization)

#### Tarefa S33P.T01 — Pesos compartilhados entre instâncias CLAP via `memfd_create` + refcount ✨🔥

- **Onde:** `src/loader/mod.rs` (carregamento); novo `src/loader/shared_weights.rs`; `src/math/common/dispatch.rs` (consumo de pesos compartilhados).
- **Problema/Oportunidade:** Workflow real em DAW para metal/rock = **4 instâncias** de ampsim em série (clean rhythm, dirty rhythm, lead, parallel cab). Hoje, cada instância carrega seu modelo ⇒ **4× memória + 4× page-fault cost** em load. Se 2+ instâncias usam o **mesmo modelo** (caso comum: dual rhythm de mesma cabine), pode-se compartilhar via `memfd_create + memfd_seal` (Linux-only) — **share read-only weight pages entre processos / cargo crates** sem cópia.
- **Solução técnica:**

  1. **Identificação de duplicação:** ao carregar modelo, computar `(canonical_path, crc32_payload)` chave. Manter `OnceLock<Mutex<HashMap<Key, Weak<SharedWeights>>>>` em static global (per-process — múltiplas instâncias do mesmo `.so` no mesmo DAW process compartilham).

  2. **Primeiro load:** cria `memfd_create("nam-rs/weights", MFD_CLOEXEC | MFD_ALLOW_SEALING)`; escreve weights; aplica `F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW` para imutabilidade; mmap; armazena `Arc<SharedWeights { fd, mapping, refcount }>` no map.

  3. **Loads subsequentes:** `Weak::upgrade()` retorna `Arc<SharedWeights>` existente; nova instância faz `mmap(fd, ..., MAP_PRIVATE)` para sua própria view (COW, mas sem write → nunca copia).

  4. **Refcount drop:** quando última instância drop, `Weak` → None; `Arc` drop fecha fd; weights liberados.

  5. **Métricas:** `RT_STATUS_SHARED_WEIGHTS_HIT` flag (set se segunda instância encontrou cache hit).

  6. **Edge case:** modelo no disco modificado entre loads (CRC32 mismatch) → cria nova entrada, antiga permanece para instâncias ativas (não invalida).
- **Pré-requisitos:**
  - **S5.T03** (CRC32 mandatory em NAMB v2) — chave de cache.
  - **S2.T01** (heap-audit) — validar que share não introduz alloc em RT.
  - **S13.T03** (stress test multi-instância) — estende teste para validar share hits.
- **Critérios de aceitação:**
  - Host de teste (`clack-host`) carregando 4 instâncias do mesmo modelo: memória total do processo ≤ 1.2× memória de 1 instância (vs 4× sem share).
  - Carregando 4 modelos distintos: zero share hits, comportamento idêntico ao baseline.
  - Soak 1h com hot-swap em uma das 4 instâncias: outras 3 continuam usando memória compartilhada original; instância swapada cria nova entrada; antiga liberada após `Weak::upgrade()` fail.
  - `cargo test --features heap-audit -- test_multi_instance_shared_weights` PASS.
- **Especialista:** `pesquisador-inovador` + `implementador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

### Sprint S34P — PipeWire Fast-Lane & Sandbox

#### Tarefa S34P.T01 — PipeWire DSP node "fast-lane" (inline no graph thread, zero thread-hop) ✨💡

- **Onde:** `src/standalone/pw_host/bridge.rs` (refator), `src/standalone/pw_host/rt_callback.rs`; investigação preliminar em `docs/innovation/pw-fast-lane.md`.
- **Problema/Oportunidade:** Standalone atual roda DSP em thread própria SCHED_FIFO; PipeWire entrega buffers via SPSC ring (S1 da Parte I). Há **1 thread hop** por bloco (PW graph thread → nam-rs RT thread). PipeWire 0.11+ permite `PW_PROP_NODE_PASSIVE = "true"` + node callback inline na graph thread, eliminando o hop. Win prático: ~5–15 µs por bloco de latência floor; menos useful em buffers grandes mas crítico em 64 spl @ 96 kHz.
- **Solução técnica:**

  1. **Estudo de viabilidade primeiro (1 dia):** documento `docs/innovation/pw-fast-lane.md` com análise da PipeWire 0.10/0.11 API e benchmark de "thread hop cost" via perf in current code.

  2. **Se viável** (~2 dias adicionais): refatorar `bridge.rs` para colocar DSP processing inline em callback registrado via `pw_filter_add_listener::process`, validando que `block_period` permanece deterministic.

  3. **Compatibility:** modo legacy preserved como `--pw-mode dedicated-thread`; novo `--pw-mode inline` (default a partir de PW ≥ 0.11).

  4. **Tradeoff:** inline limita complexidade do `process()` (sem alloc, sem locks) — já é nosso invariante. Mas limita uso de `log` macros (precisam ser RT-safe).

  5. **Telemetria:** `RT_STATUS_PW_INLINE` flag.
- **Pré-requisitos:**
  - **PipeWire 0.11+** disponível (Ubuntu 25.10 já tem).
  - **S6.T01/S6.T05** (telemetria) — RT-safe logging discipline já presente.
- **Critérios de aceitação:**
  - Documento de viabilidade revisado por `revisor-auditor` — Go/No-Go documentado.
  - Se Go: benchmark mostra p50 latency floor reduz ≥ 5 µs vs dedicated-thread em 64 spl @ 96 kHz.
  - Fallback automático para dedicated-thread em PipeWire < 0.11.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1 dia (estudo) + 2 dias condicionais (implementação).

#### Tarefa S34P.T02 — Sandbox seccomp/landlock para loader NAMB ✨⚠️

- **Onde:** novo `src/loader/sandbox.rs`; ativação em `src/loader/mod.rs::load_model()`.
- **Problema/Oportunidade:** Loader NAMB processa arquivo binário **untrusted** (pode vir de internet via ToneHunt). Bugs no parser (mesmo após hardening do Épico 3) podem ser explorados — vetor: arquivo malicioso embeddando shell escape. **Defesa em profundidade:** durante `load_model()`, aplicar **seccomp filter** que **bloqueia tudo exceto `read/mmap/munmap/exit/sigreturn`** + **landlock** restringindo filesystem ao FD aberto. Se loader for explorado para RCE, syscalls de escape (`execve`, `socket`, `open`, `connect`) retornam EPERM.
- **Solução técnica:**

  1. Crate `seccompiler` (~50 KB, mature, audited) ou inline BPF filter (~200 LoC).

  2. **Filter strict:** allowlist `read, mmap, munmap, mprotect (RO), close, exit, sigreturn, rt_sigreturn, futex, brk (limited)`.

  3. **Landlock** (Linux 5.13+): `LANDLOCK_ACCESS_FS_READ_FILE` apenas no FD do arquivo NAMB; nenhum dir; nenhum write.

  4. **Aplicação em fork model:** loader executa em **child process** isolado via `clone(CLONE_VM | ... )` — comunica weights resultantes via shared memfd (S33P.T01 reuse).

  5. **Alternativa simpler (avaliar):** aplicar seccomp **na thread do loader** sem fork — mais simples, menos isolation, mas requer cuidado para não afetar main/RT threads.

  6. **Fallback gracioso:** kernel < 5.13 ou seccomp unavailable → log warning, prosseguir sem sandbox.
- **Pré-requisitos:**
  - **S5.T01–S5.T09** (loader hardening Épico 3) — sandbox é **defesa em profundidade**, não substituto.
  - **S33P.T01** (memfd shared weights) — ABI para passar weights do sandbox out.
- **Critérios de aceitação:**
  - Smoke test: tentativa de loader carregar arquivo "malicioso" sintético que tenta `execve("/bin/sh")` é abortada com SIGSYS.
  - Loader carrega modelo legítimo sem regressão funcional (golden set passa).
  - Overhead de sandbox setup < 5 ms por load (medido em `nam-rs --diagnose` time `load_model`).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 3 dias.

### Sprint S35P — Distribuição & Onboarding

#### Tarefa S35P.T01 — Flatpak manifest para standalone ✨💡

- **Onde:** novo `dist/flatpak/com.github.fabiohl.NamRs.yml`; `dist/flatpak/README.md`.
- **Problema/Oportunidade:** Hoje, distribuir nam-rs em distros diferentes (Ubuntu 25.10, Fedora 41, Arch, Debian 13) exige builds separados ou cargo install manual. **Flatpak** entrega binário universal Linux com sandboxing nativo, dependências bundled (PipeWire shim, libs), e PipeWire portal integration funciona out-of-box.
- **Solução técnica:**

  1. Manifest Flatpak: runtime `org.freedesktop.Platform//24.08`, finish-args `--filesystem=xdg-music`, `--socket=pipewire`, `--device=dri`.

  2. Build via `flatpak-builder` em CI; publish em Flathub (PR separado).

  3. CLAP plugin **não** flatpak — não cabe modelo (CLAP é `.so` para host nativo).
- **Pré-requisitos:** Épicos 1–13 (estabilidade).
- **Critérios de aceitação:** `flatpak install nam-rs` em VM Ubuntu/Fedora limpa funciona; standalone abre interface USB e processa áudio.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

#### Tarefa S35P.T02 — AppImage para CLAP plugin (DAW-portable) ✨💡

- **Onde:** novo `dist/appimage/nam-rs-clap.AppDir/`; script `utils/build-appimage.sh`.
- **Problema/Oportunidade:** Usuários CLAP em Bitwig/Reaper precisam copiar `.so` para `~/.clap/`. **AppImage com installer GUI** (uma vez click → copia `.so` + presets default → checa CLAP_PATH) simplifica onboarding para usuário não-técnico.
- **Solução técnica:**

  1. `linuxdeploy --plugin clap-deploy` (custom plugin se não existe — fork de `linuxdeploy-plugin-gtk`).

  2. AppRun script: detecta `~/.clap/` existência, oferece instalar ou listar paths CLAP do sistema (`CLAP_PATH` env, `~/.clap`, `/usr/lib/clap`).
- **Pré-requisitos:** Épicos 1–13.
- **Critérios de aceitação:** Single AppImage run em Bitwig fresh install: plugin aparece em browser.
- **Especialista:** `implementador`.
- **Esforço:** 1.5 dia.

#### Tarefa S35P.T03 — NixOS module (declarative config) ✨💡

- **Onde:** novo `dist/nix/flake.nix`, `dist/nix/module.nix`.
- **Problema/Oportunidade:** Comunidade NixOS é **alvo prioritário** de audio pro em Linux (declarative tuning, reproducibility). Módulo Nix expõe nam-rs standalone como systemd-user-service + tuning de kernel/cpufreq declarativo.
- **Solução técnica:**

  1. Flake com derivation `nam-rs` (cargo build) + module exposing `services.nam-rs.{enable, model, autostart}`.

  2. Integra com `boot.kernelParams = ["isolcpus=..." "nohz_full=..." "rcu_nocbs=..."]` (S28P.T02 wizard suggestions, declarativos).
- **Pré-requisitos:** S28P.T02 (wizard — base de tuning).
- **Critérios de aceitação:** `nix run github:fabiohl/nam-rs#nam-rs -- --help` funciona; módulo importável em NixOS config; service starts automatically.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

---

## Resumo Executivo Parte II Continuação (Épicos 16–19)

| Sprint                                 | Tarefas    | Esforço (dias)               | Foco                                | Severidade alta (🔥) |
| -------------------------------------- | ---------- | ---------------------------- | ----------------------------------- | -------------------- |
| **S25P** (Sparsity 2:4)                | 2          | 5.5                          | Compressão estrutural               | —                    |
| **S26P** (JIT Cranelift)               | 2 (1 cond) | 3 + 3 cond.                  | Especialização runtime              | —                    |
| **S27P** (Adaptive Compute)            | 1          | 4                            | Resiliência sob pressão             | 🔥 T01               |
| **S28P** (NUMA + Tune wizard)          | 2          | 5.5                          | RT extremo + UX setup               | —                    |
| **S29P** (TSC + Black-box)             | 2          | 6                            | Determinismo + Forensics            | 🔥 T02               |
| **S30P** (Catálogo inteligente)        | 1          | 5                            | Workflow UX disruptiva              | 🔥 T01               |
| **S31P** (Pedalboard + OS + true-peak) | 3          | 8.5                          | Workflow pro + qualidade sonora     | —                    |
| **S32P** (Web remote)                  | 1          | 4                            | Controle remoto smartphone          | —                    |
| **S33P** (Shared weights)              | 1          | 4                            | Multi-instância (memfd)             | 🔥 T01               |
| **S34P** (PW fast-lane + sandbox)      | 2          | 4 + 2 cond.                  | RT floor + segurança                | —                    |
| **S35P** (Distribuição)                | 3          | 5.5                          | Onboarding (Flatpak/AppImage/NixOS) | —                    |
| **TOTAL Épicos 16–19**                 | 20 tarefas | **~55 + 5 cond. = ~60 dias** | —                                   | 4 críticas           |

### Ordem de execução agile sugerida (6 sprints de ~2 semanas)

- **Sprint #1 (semana 1–2) — Fundações de instrumentação:**
  - S29P.T02 (Black-box recorder — instrumenta o resto).
  - S28P.T01 (NUMA — base para benchmarks de RT-extremo).
  - S28P.T02 (Tune wizard — desbloqueia setup reprodutível em outras tasks).
- **Sprint #2 (semana 3–4) — Adaptive & Determinismo:**
  - S27P.T01 (Adaptive compute — 🔥, alimenta blackbox triggers).
  - S29P.T01 (TSC-deadline).
  - S33P.T01 (Shared weights — 🔥, validar com S13.T03 estendido).
- **Sprint #3 (semana 5–6) — Compressão & JIT:**
  - S25P.T01 (Sparsity 2:4 — depende S29.T01 já entregue).
  - S25P.T02 (Auto-pruning por camada).
  - S26P.T01 (JIT feasibility — paralelo, decisão ao final do sprint).
- **Sprint #4 (semana 7–8) — UX disruptiva:**
  - S30P.T01 (Catálogo inteligente — 🔥, depende S28.T01 stress signal).
  - S31P.T01 (Pedalboard chain).
  - S26P.T02 (JIT integration — condicional ao Go-decision do Sprint #3).
- **Sprint #5 (semana 9–10) — Qualidade Sonora & Remote:**
  - S31P.T02 (Oversampling).
  - S31P.T03 (True-peak limiter).
  - S32P.T01 (Web remote — paralelo, sem deps internas).
  - S34P.T02 (Sandbox loader — paralelo, depende S33P.T01).
- **Sprint #6 (semana 11–12) — Polimento & Distribuição:**
  - S34P.T01 (PW fast-lane — estudo + decisão).
  - S35P.T01 (Flatpak).
  - S35P.T02 (AppImage CLAP).
  - S35P.T03 (NixOS module).

### Pré-condições de início da continuação

1. **Épicos 1–13 da Parte II original CONCLUÍDOS** (especialmente Épico 9 Quantization, Épico 10 RT-OS, Épico 11 UX, Épico 12 Observabilidade).
2. **Épicos 14–15 da Parte I (TODO-sprints.md) CONCLUÍDOS** — entregam `compute_esr`, `compute_mr_stft`, stress signal v2, HDR Histograms e DiagnosticBundle/RuntimeSnapshot que esta continuação **consome amplamente**.
3. **Baseline `cargo bench inference_bench` salvo** em `target/criterion/baseline_post_e15_post_e13/`.
4. **Hardware de validação disponível:**

   - x86_64 com AVX-512 (qualquer Sapphire Rapids EC2 c7i ou Zen 4).

   - **Opcional desejável:** sistema dual-socket Xeon ou Threadripper Pro para S28P.T01 (NUMA validation).

   - PipeWire 0.11+ (Ubuntu 25.10 já entrega) para S34P.T01.

### Gate de saída de cada sprint (estende gate do Épico 14)

1. `bash utils/lints.sh` (clippy strict + fmt).
2. `bash utils/tests-cargo.sh` (unit + integration).
3. `cargo bench inference_bench` — **sem regressão > 1%** vs baseline na configuração `default` (sem features experimentais ativas); **com features experimentais** (jit-cranelift, amx-nightly, etc.), regressão até +5% **aceita se acompanhada de ganho documentado** em path opt-in.
4. `cargo test --test cpp_parity -- --ignored --nocapture` — 20/20 PASS (após Épico 15).
5. `cargo test --features heap-audit` — zero alloc em RT (extensão de S2.T01).
6. Para tasks ✨ (todas desta continuação): documento de inovação em `docs/innovation/<area>.md` com benchmark empírico e comparação contra baseline.

### Validação final da continuação

- **Auditoria por `revisor-auditor`** ao final do Épico 19 com relatório de impacto cumulativo nos 4 pilares: latência RT (p99 sob stress), eficiência inferência (TFLOPS/W em modelos canônicos), UX (workflow comparado vs Neural DSP Quad Cortex desktop sw), surface de plataforma (Flatpak/AppImage/Nix install success rate em 5 distros).
- **Cross-validation `pesquisador-inovador`:** confirmar que cada inovação ✨ entrega o speedup/feature prometido com erro ≤ 20% vs estimativa inicial; ajustes de escopo retroativos se métrica não bate.
- **Documentation freeze:** `docs/architecture.md`, `docs/innovation/index.md` (TOC), `README.md` (changelog 2.0.0?) atualizados pela skill `documentador` antes do release.

> **Nota PO:** Esta continuação eleva nam-rs do estado "engine NAM open-source competitivo" para **referência avant-garde do ecossistema NAM em 2026** — combinando o que **só Linux permite** (memfd weight sharing, landlock sandbox, PREEMPT_RT + TSC-deadline, AMX-sparse 2:4) com **UX inovador** (catálogo perceptual, pedalboard, web remote, adaptive compute) e **forensics de campo** (black-box recorder + DiagnosticBundle). O resultado é um produto que **não tem equivalente** em macOS/Windows até porque depende intrinsecamente do stack Linux/PipeWire.
