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

#### Tarefa S21.T02 — Métricas perceptuais (MR-STFT, ESR) em CI ✨⚠️

- **Onde:** `tests/perceptual_metrics.rs`.
- **Problema/Oportunidade:** MSE é métrica fraca para áudio (não captura percepção). MR-STFT (multi-resolution STFT loss, Yamamoto et al. 2020) e ESR (Error-to-Signal Ratio) são padrões da pesquisa NAM. CI deve falhar se ESR > threshold de regressão.
- **Solução técnica:**
  1. Implementar MR-STFT com window sizes {256, 1024, 4096} log magnitude L1+L2.
  2. ESR: `10*log10(sum(diff²) / sum(target²))`.
  3. Test runner compara saída atual vs golden (último known-good); fail se ESR delta > 1 dB.
- **Critérios de aceitação:** Tests passam para todos os modelos de catálogo; regressões numéricas pegas automaticamente.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

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
