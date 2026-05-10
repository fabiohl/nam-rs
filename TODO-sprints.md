<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# TODO Sprints — NAM-rs

> Gerado pelo `pesquisador-inovador` em 2026-05-10.  
> Baseline: commit `HEAD`, versão 1.4.2, target `x86-64-v3`.

---

## Sprint 1 — RT-Safety Hardening & Bug Fixes

**Objetivo:** Corrigir defeitos de correção e aderência operacional que afetam a confiabilidade em produção. As tarefas são atômicas e independentes.

### Tarefa 1.1 — Corrigir contador `dropped_frames` órfão no `RtStatusFlags`

- **Problema:** `RtStatusFlags.dropped_frames` (`src/spsc.rs:101`) é declarado, inicializado e lido por `poll_rt_status()` (`src/rt_setup.rs:195`) — mas **nenhum código escreve nele**. O contador real está em `DspBridge.dropped_frames` (`src/dsp/pipeline.rs:98`), que é incrementado em `write_bridge()` e `handle_silence_bypass()` mas **nunca lido em produção** (só em testes).
- **Solução:** Remover `dropped_frames` do `RtStatusFlags`. Adicionar um método `DspBridge::drain_dropped_frames() -> u32` que faz `swap(0)`. Fazer `poll_rt_status()` receber `&DspBridge` (via Arc ou ponteiro) e usar esse método. Atualizar `pipeline_test.rs` para testar o novo caminho.
- **Critério de aceitação:** `poll_rt_status` loga "Drifting detectado" com valor > 0 após simular drop no teste `test_hotpath_dropped_frames`.
- **Estimativa:** 30 min
- **Arquivos:** `src/spsc.rs`, `src/dsp/pipeline.rs`, `src/rt_setup.rs`, `src/pw_host.rs`, `src/dsp/pipeline_test.rs`

### Tarefa 1.2 — Remover `max(0.0)` numérico do hot-path do resampler

- **Problema:** `src/dsp/resampler.rs:140-141` executa `self.phase_accum = self.phase_accum.max(0.0)` no loop interno de `process_internal()`. É um branch em float no caminho mais quente do resampler. O `debug_assert!` já confirma que `phase_accum >= -1e-12`, então `max(0.0)` é redundante em release. Além disso, `max()` em float gera `MAXSS`/`VMAXSS` que têm latência de 3-4 ciclos.
- **Solução:** Mover o `max(0.0)` para dentro de `#[cfg(debug_assertions)]` ou substituir a lógica de subtração condicional `while phase_accum >= NUM_PHASES` por um acumulador de ponto fixo `u64` com `phase_accum_fixed: u64` e `phase_step_fixed: u64`, eliminando completamente aritmética float na máquina de estados do resampler. A conversão `f64→u64` com 32 bits fracionários mantém precisão de ~0.23 ns e remove todos os branches de underflow.
- **Critério de aceitação:** Testes `resampler_test.rs` passam; benchmark não mostra regressão; SNR > 120 dB mantido.
- **Estimativa:** 1h30
- **Arquivos:** `src/dsp/resampler.rs`, `src/dsp/sinc_kernel.rs`

### Tarefa 1.3 — Otimizar `configure_realtime_thread()` para minimizar jitter no primeiro frame

- **Problema:** `configure_realtime_thread()` (`src/rt_setup.rs:336`) executa `mlockall`, `prctl`, `pthread_setaffinity_np`, `pthread_setschedparam` no primeiro frame do callback DSP. Todas são syscalls que podem causar page faults ou migração de página no momento crítico.
- **Solução:** Dividir em duas fases:  
  (a) **Pre-DSP**: `set_daz_ftz()` + `prctl(PR_SET_THP_DISABLE)` permanecem no cold-path do primeiro frame (são rápidos, apenas registradores).  
  (b) **Post-alloc**: `mlockall(MCL_CURRENT | MCL_FUTURE)` e `pthread_setaffinity_np` são movidos para logo após `Box::leak` do `DspBridge` em `run_pipewire_host()`, ANTES de conectar as streams PW. Para `pthread_setaffinity_np`, usar `pthread_self()` na thread principal + `libc::sched_setaffinity(0, ...)` que é equivalente.  
  (c) `pthread_setschedparam` permanece no cold-path se PipeWire/rtkit não elevou; isso é aceitável pois é uma syscall leve (~1µs).
- **Critério de aceitação:** Primeiro frame de áudio não excede 50% do budget; log confirma que `mlockall` rodou antes das streams.
- **Estimativa:** 1h
- **Arquivos:** `src/rt_setup.rs`, `src/pw_host.rs`, `src/main.rs`

### Tarefa 1.4 — Verificar e corrigir `SystemSnapshot` enviado ao `NamDiagnostic` na thread CLI

- **Problema:** A thread CLI interativa (`cli.rs`) cria diagnósticos com `SystemSnapshot` clonado do `main()`. Se o `SystemSnapshot` for capturado antes de `pipewire::init()`, pode faltar informação de runtime. Verificar se o snapshot inclui todos os campos relevantes (CPU features, kernel, versão PW).
- **Solução:** Adicionar chamada `SystemSnapshot::refresh()` que atualiza campos dinâmicos (ex: versão PW) sem realocar. Chamar após `pipewire::init()` e antes de spawnar a CLI.
- **Critério de aceitação:** `NamDiagnostic::support_block()` inclui versão do PipeWire quando chamado da CLI.
- **Estimativa:** 30 min
- **Arquivos:** `src/diagnostics.rs`, `src/main.rs`

---

## Sprint 2 — Microarchitectural Performance Wins

**Objetivo:** Reduzir ciclos de CPU no hot-path DSP através de otimizações microarquiteturais de alto impacto. Cada tarefa elimina branches ou melhora o throughput de instruções.

### Tarefa 2.1 — Pré-resolver function pointers SIMD no `DspPipelineContext`

- **Problema:** `apply_output_stage()` (`src/dsp/pipeline.rs:323`) usa `dispatch_simd!` Mode 2 que faz `match SIMD_MATH.instruction_set { ... }` a cada frame. Mesmo sendo um jump table, são ~2-3 ciclos de branch + load da vtable. Multiplicado por ~1000 frames/s, é desperdício evitável.
- **Solução:** Adicionar campo `simd_apply_gain_rt_stereo: unsafe fn(&mut DynamicHysteresis, &mut [f32], &mut [f32], usize)` ao `DspPipelineContext`. Preencher em `run_pipewire_host()` durante a inicialização com o function pointer correto para o hardware. No hot-path, chamar diretamente: `(ctx.simd_apply_gain_rt_stereo)(ctx.silence_hysteresis, ...)`.  
  **Nota:** O Mode 3 (v-table global) já é branch-free para `apply_gain_and_detect_clipping_stereo`. Mode 2 é o alvo.
- **Critério de aceitação:** `cargo bench` mostra redução mensurável (~0.5-1%) no benchmark `WaveNet_Standard`; testes de pipeline passam.
- **Estimativa:** 1h
- **Arquivos:** `src/dsp/pipeline.rs`, `src/pw_host.rs`

### Tarefa 2.2 — Eliminar branch `dispatch_simd!` Mode 1 no `ResamplerCore`

- **Problema:** `NamResampler::process_input()` e `process_output()` chamam `dispatch_simd!(core, process_internal, ...)` que faz o match de 6 vias a cada chamada de resampling. No pior caso (48kHz→44.1kHz), isso ocorre múltiplas vezes por frame.
- **Solução:** Adicionar campo `process_fn: unsafe fn(&mut ResamplerCore, &[f32], &[f32], &mut [f32], &mut [f32]) -> usize` ao `ResamplerCore`, preenchido no construtor via `detect_best_simd()`. `NamResampler::process_input/output` chamam `(core.process_fn)(core, in_l, in_r, out_l, out_r)`.
- **Critério de aceitação:** Testes do resampler passam; benchmark de resampling sem regressão.
- **Estimativa:** 45 min
- **Arquivos:** `src/dsp/resampler.rs`

### Tarefa 2.3 — Adicionar prefetch lookahead no `Conv1d` da WaveNet

- **Problema:** O loop `Conv1d::process_internal()` em `src/models/wavenet.rs` processa dilatações sequencialmente. Cada dilatação acessa um slice de peso F16C em posição `dilation * kernel_size` distante. Sem prefetching, cada primeira iteração de dilatação sofre L1 cache miss (~4-5 ciclos) nos pesos.
- **Solução:** No início de cada dilatação `d`, emitir `_mm_prefetch::<_MM_HINT_T0>` para o endereço base dos pesos da dilatação `d+1`. Como são pesos F16C (2 bytes/elemento), uma cache line (64B) cobre 32 pesos — suficiente para kernels de tamanho 3. O custo é 1 instrução por dilatação (~0.5 ciclo), benefício é evitar 1 L1 miss (~4-5 ciclos).
- **Critério de aceitação:** `perf stat -e L1-dcache-load-misses` mostra redução nos misses do benchmark `WaveNet_Standard`.
- **Estimativa:** 45 min
- **Arquivos:** `src/models/wavenet.rs`

### Tarefa 2.4 — SIMD-ify `HardSwish` e `LeakyHardTanh` activation functions

- **Problema:** `ActivationType::HardSwish` (`src/models/activations.rs:142-147`) e `LeakyHardTanh` (`:154-161`) são puramente escalares. `HardSwish` é usada em arquiteturas modernas MobileNetV3/EfficientNet e `LeakyHardTanh` em variantes A2. Ambas são trivialmente vetorizáveis.

- **Solução (HardSwish AVX2):**  
  
  ```rust
  // x * clamp(x + 3, 0, 6) / 6
  let v3 = _mm256_set1_ps(3.0);
  let v0 = _mm256_setzero_ps();
  let v6 = _mm256_set1_ps(6.0);
  let v_inv6 = _mm256_set1_ps(1.0 / 6.0);
  for chunk in data.chunks_exact_mut(8) {
      let x = _mm256_loadu_ps(chunk.as_ptr());
      let t = _mm256_add_ps(x, v3);
      let clamped = _mm256_max_ps(v0, _mm256_min_ps(t, v6));
      let result = _mm256_mul_ps(x, _mm256_mul_ps(clamped, v_inv6));
      _mm256_storeu_ps(chunk.as_mut_ptr(), result);
  }
  ```
  
  Adicionar `hard_swish_slice_avx2()` e `hard_swish_slice_avx512()` ao `fastmath.rs`. Despachar via `InstructionSet` no `ActivationType::HardSwish`.

- **Solução (LeakyHardTanh AVX2):** Similar com `_mm256_cmp_ps` + `_mm256_blendv_ps` para branching sem branches escalares.

- **Critério de aceitação:** Testes em `activations.rs` passam; benchmark de ativação mostra speedup 3-4x.

- **Estimativa:** 1h30

- **Arquivos:** `src/math/fastmath.rs`, `src/models/activations.rs`

### Tarefa 2.5 — Flag de bypass estático em `apply_gain_simd`

- **Problema:** `apply_gain_simd()` (`src/dsp/gain.rs:17`) faz `(gain_linear - 1.0).abs() < 1e-6` toda chamada. O ganho só muda via mensagem SPSC (rara). O branch é bem predito (always taken quando gain=1.0), mas ainda consome recursos de branch predictor.
- **Solução:** Adicionar `input_gain_is_unity: bool` e `output_gain_is_unity: bool` ao `DspPipelineContext`. Quando `true`, pular completamente a chamada a `apply_gain_simd` no `apply_input_stage` e `apply_output_stage`. Atualizar as flags no consumer do SPSC quando `InputGain`/`OutputGain` chega.
- **Critério de aceitação:** `perf stat` mostra redução de branches no hot-path; testes de ganho passam.
- **Estimativa:** 30 min
- **Arquivos:** `src/dsp/pipeline.rs`, `src/dsp/gain.rs`, `src/pw_host.rs`

---

## Sprint 3 — Code Quality & Maintainability

**Objetivo:** Resolver débito técnico estrutural que afeta segurança de código e aderência às regras do projeto.

### Tarefa 3.1 — Substituir `*mut DspBridge` por abstração segura

- **Problema:** `handle_silence_bypass()`, `write_bridge()`, `playback_dsp_cycle()` e `capture_dsp_pipeline()` recebem `*mut DspBridge` como parâmetro. As funções usam `#[allow(clippy::not_unsafe_ptr_arg_deref)]` para suprimir warnings. Isso mascara a insegurança real: competição entre capture e playback no mesmo `DspBridge` (mitigada por double-buffering + atomics, mas não expressa no tipo).

- **Solução:** Criar struct `SharedBridge` com:
  
  ```rust
  #[repr(align(128))]
  pub struct SharedBridge {
      inner: std::cell::UnsafeCell<DspBridge>,
  }
  impl SharedBridge {
      pub fn write(&self, ...) { /* capture side */ }
      pub fn read(&self, ...) -> Option<...> { /* playback side */ }
  }
  ```
  
  O `UnsafeCell` comunica a mutabilidade interior. Os métodos `write`/`read` encapsulam o protocolo double-buffer e os atomics. Substituir `bridge_ptr: *mut DspBridge` por `bridge: &'static SharedBridge`.

- **Critério de aceitação:** Compila sem `#[allow(clippy::not_unsafe_ptr_arg_deref)]` nos arquivos afetados; testes de pipeline e PW passam.

- **Estimativa:** 2h

- **Arquivos:** `src/dsp/pipeline.rs`, `src/pw_host.rs`, `src/main.rs`, `src/dsp/pipeline_test.rs`, `src/pw_host_test.rs`

### Tarefa 3.2 — Corrigir referência incorreta a `std::simd` nas regras

- **Problema:** `.agents/rules/rust.md:12` instrui "Use `std::simd`", mas o código usa exclusivamente `core::arch::x86_64::*` (intrinsics). `std::simd` é a API portável `std::simd::f32x8` que ainda não é estável e não é usada no projeto. A regra induz futuros contribuidores ao erro.
- **Solução:** Substituir "Use `std::simd` e FastMath Minimax" por "Use `core::arch::x86_64::*` (AVX2/AVX-512 intrinsics) e FastMath Minimax via polinômios Padé/Minimax".
- **Critério de aceitação:** Regra atualizada; revisão de `grep std::simd` retorna zero resultados no código.
- **Estimativa:** 5 min
- **Arquivos:** `.agents/rules/rust.md`

### Tarefa 3.3 — Consolidar documentação multilíngue nos `docs/`

- **Problema:** `docs/architecture.md` é primariamente em português; `docs/benchmarks.md` e `docs/dependencies.md` são em inglês. `README.md` é bilíngue (bom). Mistura de idiomas dificulta contribuidores internacionais.
- **Solução:** Converter `docs/architecture.md` e `docs/f16c_compression_analysis.md` para o padrão bilíngue (PT-BR + EN), cada seção com versão nos dois idiomas OU padronizar tudo em inglês (língua franca de sistemas). Recomendação: inglês para docs técnicos, mantendo README bilíngue para usuários finais brasileiros.
- **Critério de aceitação:** Todos os `docs/*.md` no mesmo idioma ou bilíngues consistentes.
- **Estimativa:** 1h
- **Arquivos:** `docs/architecture.md`, `docs/f16c_compression_analysis.md`, `docs/benchmarks.md`, `docs/clap_integration.md`

### Tarefa 3.4 — Adicionar testes de integração para o caminho `poll_rt_status` completo

- **Problema:** `poll_rt_status()` (`src/rt_setup.rs:124`) não tem teste dedicado que cobre todas as flags (HAS_CLIPPED, GC_OVERFLOW, IS_SILENT, IS_FADING, rate notifications, overloads, drops, latency telemetry). A cobertura atual é apenas indireta via testes de pipeline.
- **Solução:** Criar `rt_setup_test.rs` com teste que:
  1. Cria `RtStatusFlags`, seta cada flag individualmente
  2. Chama `poll_rt_status()` e verifica transições de estado (silent→active, fading edges)
  3. Verifica que flags one-shot (HAS_CLIPPED, GC_OVERFLOW) são limpas após leitura
  4. Verifica throttle de telemetria (a cada 100 chamadas)
- **Critério de aceitação:** `cargo test rt_setup` cobre > 90% das linhas de `poll_rt_status`.
- **Estimativa:** 1h
- **Arquivos:** `src/rt_setup_test.rs`, `src/spsc.rs`

---

## Sprint 4 — Observabilidade & Diagnóstico Avançado

**Objetivo:** Melhorar a capacidade de diagnosticar problemas de performance em produção e desenvolvimento.

### Tarefa 4.1 — Telemetria de latência por estágio do pipeline

- **Problema:** Atualmente `dsp_cycle_time` mede o tempo total do pipeline. Não é possível saber se o gargalo está no gate, na inferência, no resampling ou no output gain.
- **Solução:** Adicionar campos `stage_times: [AtomicU64; 4]` ao `RtStatusFlags`. Medir RDTSC antes/depois de cada estágio no `capture_dsp_pipeline()` e armazenar a diferença. No `poll_rt_status()`, reportar percentual de cada estágio quando a telemetria é emitida (a cada 100 ciclos). Exemplo de log: `"DSP: Gate=2% Inf=78% OutGain=18% Bridge=2%"`
- **Critério de aceitação:** Log de telemetria mostra breakdown por estágio; valores somam ~100%.
- **Estimativa:** 1h
- **Arquivos:** `src/spsc.rs`, `src/dsp/pipeline.rs`, `src/rt_setup.rs`

### Tarefa 4.2 — Adicionar métricas de cache miss via `perf_event_open` no cold-path

- **Problema:** Não há visibilidade de cache misses em produção. Em máquinas de usuário, `perf stat` não está disponível ou requer root.
- **Solução:** No `configure_realtime_thread()`, abrir contadores `PERF_COUNT_HW_CACHE_MISSES` via `perf_event_open` com `pid=0` e `cpu=target_cpu`. Armazenar os FDs no `RtStatusFlags`. Expor leitura acumulada no `poll_rt_status()` a cada ~10s. Isso permite detectar regressões de acesso a memória sem ferramentas externas.  
  **Nota:** Requer kernel Linux com `perf_event_paranoid <= 2` (comum em distros de áudio). Fazer fallback silencioso se `perf_event_open` falhar.
- **Critério de aceitação:** Log mostra "Cache: L1D={} L2={}" quando suportado; sem crash quando não suportado.
- **Estimativa:** 2h
- **Arquivos:** `src/rt_setup.rs`, `src/spsc.rs`, `Cargo.toml` (adicionar dependência `perf-event-open` ou syscall direta via `libc`)

---

## Resumo

| Sprint              | Tarefas                 | Impacto     | Esforço Total |
| ------------------- | ----------------------- | ----------- | ------------- |
| 1 — RT-Safety       | 1.1, 1.2, 1.3, 1.4      | CRÍTICO     | ~3h30         |
| 2 — Microarch       | 2.1, 2.2, 2.3, 2.4, 2.5 | ALTO        | ~4h30         |
| 3 — Code Quality    | 3.1, 3.2, 3.3, 3.4      | MÉDIO       | ~4h           |
| 4 — Observabilidade | 4.1, 4.2                | BAIXO-MÉDIO | ~3h           |

**Ordem recomendada:** Sprint 1 → Sprint 2 → Sprint 3 → Sprint 4.  
Sprints 1 e 2 podem ser paralelizadas (arquivos majoritariamente independentes).
