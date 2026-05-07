# TODO Sprints — Backlog Técnico NAM-rs

> Backlog derivado das auditorias **Cold Review** (infra), **Hot Review** (DSP) e **Test Review** (QA).
> Use `/tarefa <ID>` (ex: `/tarefa 1.1`) para iniciar a implementação de uma tarefa.

---

## Épico A — Corretude e Segurança do Hot-Path

Tarefas que corrigem bugs que afetam o comportamento audível ou a estabilidade do sistema em tempo real.

### Sprint 1: True-Bypass — Sinal Limpo sem Modelo

**Objetivo:** Quando nenhum modelo estiver carregado, a saída deve ser o sinal de entrada inalterado (dry pass-through), em vez de silêncio ou lixo de memória.

#### Tarefa 1.1 · Implementar True-Bypass em `run_inference()` *(Bug — Severidade Alta)* [x]

- **Arquivo:** `src/dsp/pipeline.rs`, função `run_inference()`.
- [x] No caminho **sem resampler** (L200–220): quando `active_model_l` é `None`, copiar `model_in_l` → `model_out_l`.
- [x] Aplicar lógica análoga para `active_model_r` no modo estéreo.
- [x] No caminho **com resampler** (L221–257): quando `active_model_l` é `None`, copiar `model_in_l` → `temp_out_l` (atualmente `temp_out_l` fica zerado, matando o sinal dry).
- [x] Validar que o bypass funciona independentemente do estado do resampler e do modo mono/estéreo.

#### Tarefa 1.2 · Testes de Integração para Bypass *(Cobertura)* [x]

- [x] Criar testes que validem: `output == input` (bit-exact) quando nenhum modelo está carregado.
- [x] Cobrir os 4 cenários: {resampler bypass, resampler ativo} × {mono, estéreo}.

---

### Sprint 2: Eliminação de Vazamentos de Memória no GC

**Objetivo:** Substituir `Box::leak` e `mem::forget` por um mecanismo lock-free que previna vazamentos progressivos de memória sob estresse.

#### Tarefa 2.1 · Overwrite Ring Buffer para GC de Modelos e Resamplers *(Bug — Severidade Média)*

- **Arquivo:** `src/pw_host.rs`.
- [x] Projetar um mecanismo de overwrite ring buffer (ou parking lot com tamanho fixo) que substitua o fallback `Box::leak`.
- [x] Remover `Box::leak(still_here)` nas linhas 374 e 392 (modelos L e R).
- [x] Remover `std::mem::forget(still_here)` na linha 333 (resampler) — atualmente vaza `NamResampler` de forma irrecuperável.
- [x] Atualizar o `gc_producer` para acomodar tanto modelos quanto resamplers.

#### Tarefa 2.2 · Testes de Estresse para GC *(Cobertura)*

- [x] Criar um teste que force o GC a encher o buffer de sobrescrita e validar que o mais antigo é sobrescrito.
- [x] Validar que o GC não vaza memória sob carga contínua de modelos/resamplers.

---

### Sprint 3: Correções Pontuais de Infraestrutura

**Objetivo:** Corrigir bugs sutis na camada de infra que produzem comportamento incorreto em cenários específicos.

#### Tarefa 3.1 · Corrigir Fallback de `rdtsc_nanos()` *(Bug B1 — Severidade Média)*

- **Arquivo:** `src/rt_setup.rs`, linhas 30 e 40.
- **Diagnóstico:** `Instant::now().elapsed()` retorna ~0 ns (cria o instante e mede imediatamente).
- [ ] Substituir por uma referência temporal estática (ex: `minstant::Instant` com âncora cold-path one-shot) para que o fallback produza valores monotônicos significativos.
- [ ] Validar que a telemetria DSP reporta valores corretos quando o TSC não está calibrado.

#### Tarefa 3.2 · Guardar Overflow de Shift em `emit_irq_advisory()` *(Bug B5 — Severidade Baixa)*

- **Arquivo:** `src/diagnostics.rs`, linha 207.
- **Diagnóstico:** `1u32 << core` causa panic se `core >= 32`.
- [ ] Adicionar guarda `if core >= 32 { return; }` antes do cálculo, ou migrar para `u64` com `1u64 << core`.

#### Tarefa 3.3 · Feedback para Comandos CLI Incompletos *(Bug B3 — Severidade Baixa)*

- **Arquivo:** `src/cli.rs`, blocos `"gain"`, `"out"` e `"load"` (linhas 203–252).
- [ ] Adicionar `else` ao `if let Some(...)` com mensagem de uso quando o argumento obrigatório estiver ausente.

---

## Épico B — Otimizações de Performance

Tarefas que melhoram throughput, latência ou eficiência de cache sem alterar o comportamento audível.

### Sprint 4: Histerese de Gate com Granularidade Sub-Bloco

**Objetivo:** Eliminar "zipper noises" no final do fade-out/in do Noise Gate.

#### Tarefa 4.1 · Interpolação Sub-Bloco no Gate *(Perf — Qualidade de Áudio)*

- **Arquivo:** `src/dsp/gate.rs`, função `apply_gain_rt`.
- [ ] Calcular a rampa estritamente até o sample de encerramento do fade.
- [ ] Preencher o restante do buffer com a constante final (`0.0` ou `1.0`).

#### Tarefa 4.2 · Testes de Granularidade *(Cobertura)*

- **Arquivo:** `src/dsp/gate_test.rs`.
- [ ] Adicionar cenários com buffers grandes (`n_samples = 4096`) e fades curtos (`fade_frames = 256`).
- [ ] Assertir que não há saltos de transição não interpolados.

---

### Sprint 5: Alinhamento SIMD para Alocações Dinâmicas

**Objetivo:** Garantir que todos os buffers dinâmicos no hot-path SIMD respeitem alinhamento de 32/64 bytes, evitando degradação de cache por `unaligned loads`.

#### Tarefa 5.1 · Auditoria e Alinhamento de `WaveNetDyn` e `Conv1dDyn` *(Perf — Cache)*

- **Arquivos:** `src/models/wavenet_common.rs`, `src/models/wavenet_dyn.rs`.
- [ ] Revisar todos os buffers dinâmicos que atingem o hot-path SIMD e impor alinhamento de 32 bytes (AVX2) ou 64 bytes (AVX-512).
- [ ] Substituir `debug_assert!(mixin_len <= 4096)` por `assert!` em `WaveNetLayerDyn::process_block_internal` (L417) — evita buffer overrun silencioso em release para topologias A2 com `bottleneck > 64`.
- [ ] Auditar o remainder loop escalar de `Conv1dDyn::process_block` (L139–156) para topologias com `out_ch % 4 != 0`.

---

### Sprint 6: Vetorização SIMD de Ativações A2

**Objetivo:** Eliminar gargalos escalares em funções de ativação antes da implantação WaveNet A2.

#### Tarefa 6.1 · Implementar Versões FMA/AVX2 das Ativações *(Perf — A2 Prep)*

- **Arquivo:** `src/models/activations.rs` (implementações escalares atuais).
- **Ativações-alvo:**
  - `ReLU` / `PReLU` — branch escalar, trivial de vetorizar com `_mm256_max_ps` / `_mm256_blendv_ps`.
  - `Softsign` (`x / (1 + |x|)`) — contém divisão escalar lenta sem aproximação SIMD.
  - `SiLU` (`x * sigmoid(x)`) — usa `(-x).exp()` da libm (~20-60 ciclos vs ~4-6 ciclos do FastMath).
  - `Sigmoid` — mesma issue da `SiLU`.
- [ ] Implementar versões intrinsics em `src/math/fastmath.rs` para cada ativação acima.
- [ ] Escrever testes de paridade RMSE contra a versão escalar e a referência C++.

---

### Sprint 7: Otimizações LSTM — Fusão e BF16

**Objetivo:** Desbloquear ganhos de ILP e throughput no hot-path LSTM.

#### Tarefa 7.1 · Fusão ILP para Gates LSTM — `fused_tanh_sigmoid` *(Perf HP1)*

- **Arquivo:** `src/math/fastmath.rs`.
- [ ] Criar `fused_tanh_sigmoid_avx2(tanh_in, sig_in) -> (__m256, __m256)` intercalando instruções dos polinômios de tanh e sigmoid para maximizar ILP.
- [ ] Atualizar `fused_lstm_gates_avx2` (L470–490) para usar a nova fusão.
- [ ] Criar versão AVX-512 análoga e atualizar `fused_lstm_gates_avx512` (L497–518).
- [ ] Validar paridade numérica e medir ganho via `cargo bench`.

#### Tarefa 7.2 · Vetorizar Conversão BF16 do Hidden State *(Perf HP2)*

- **Arquivo:** `src/models/lstm.rs`, macro `define_lstm_process!` (L91–97 e L124–129).
- **Diagnóstico:** Loop escalar `(h_s_arr[j].to_bits() >> 16) as u16` que pode ser substituído por `M::f32_to_bf16()`.
- [ ] Substituir os loops escalares de conversão BF16 por chamada ao trait `SimdMath`.
- [ ] Verificar via `cargo asm` se o compilador já auto-vetorizava antes de declarar ganho.

#### Tarefa 7.3 · Unificar 4 GEMV do `LstmDynLayer` em Travessia Única *(Perf HP4)*

- **Arquivo:** `src/models/lstm_dyn.rs`, `LstmDynLayer::process_sample` (L66–133).
- [ ] Substituir as 4 chamadas `M::gemv_overwrite` separadas por uma travessia única de `state`.
- [ ] Opção A: `gemv_4gate_dyn` no trait `SimdMath` com slices dinâmicos.
- [ ] Opção B: Reutilizar `gemv_4gate_avx2`/`avx512` existentes com pesos fatiados em 4 chunks.
- [ ] Ganho esperado: ~20-30% pela eliminação de 3 travessias redundantes.

---

### Sprint 8: Otimizações WaveNet Dinâmico — BF16

**Objetivo:** Ativar paths BF16 no WaveNet dinâmico para CPUs com suporte nativo.

#### Tarefa 8.1 · Paths BF16 no `WaveNetLayerDyn` *(Perf HS2)*

- **Arquivo:** `src/models/wavenet_common.rs`, `process_block_internal` (L399–471).
- [ ] Implementar uso de `condition_bf16` e `layer_buffer_bf16` quando `M::IS_BF16 == true`.
- [ ] Adicionar conversão BF16 no output inter-camada via `M::f32_to_bf16()` (análogo ao WaveNet estático em `wavenet.rs` L1016–1018).
- [ ] Utilizar `self.input_mixin.process_block_bf16::<M>()` no path BF16.
- [ ] Ganho esperado: ~15-25% em CPUs com AVX-512 BF16 nativo.

---

### Sprint 9: Streaming I/O na Inicialização

**Objetivo:** Tornar a varredura de `/proc/interrupts` escalável para sistemas com muitos núcleos.

#### Tarefa 9.1 · Refatorar `parse_interrupts_per_cpu()` para Streaming *(Perf — Cold Path)*

- **Arquivo:** `src/rt_setup.rs`, linha 548.
- [ ] Substituir `fs::read_to_string("/proc/interrupts")` por `BufReader::new(File::open(...)).lines()` para evitar alocação monolítica em sistemas com alto número de CPUs.

---

## Épico C — Corretude do Modelo Neural

Tarefas que corrigem erros de interpretação numérica nos modelos de inferência.

### Sprint 10: Correções BF16/f16 no LSTM

**Objetivo:** Corrigir confusões entre formatos BF16 e IEEE f16 que produzem resultados de inferência incorretos.

#### Tarefa 10.1 · Ativar Path BF16 nas Variantes VNNI do LSTM *(Bug HB1 — Severidade Média)*

- **Arquivo:** `src/models/lstm.rs`.
- **Diagnóstico:** As variantes `process_sample_avx2vnni` (L255–269) e `process_sample_avx512vnni` (L270–284) passam `$is_bf16: false`, desperdiçando o potencial de aceleração VNNI. Já existe uma variante `process_sample_avx512_vnni_bf16` (L285–299) com `$is_bf16: true`, mas requer feature `avx512bf16`.
- [ ] Avaliar a ativação de BF16 nas variantes VNNI padrão (sem exigir `avx512bf16`).
- [ ] Validar paridade numérica contra golden vectors C++ — BF16 truncation altera resultados.
- [ ] Confirmar que `gemv_4gate_bf16_fallback` é funcional para hidden sizes padrão (H=8, 12, 16, 24).
- [ ] Atualizar `lstm_test.rs` com tolerâncias ajustadas.

#### Tarefa 10.2 · Corrigir Fallback Escalar — Interpretação f16 vs BF16 *(Bug HB2 — Severidade Alta)*

- **Arquivo:** `src/models/lstm.rs`.
- **Diagnóstico:** Em `process_sample_scalar` (L312), `LstmModel1::process_scalar` (L429) e `LstmModel2::process_scalar` (L533), os pesos `u16` são interpretados como IEEE f16 via `half::f16::from_bits(w).to_f32()`, mas são na verdade BF16 (bits superiores de f32). A conversão correta é `f32::from_bits((w as u32) << 16)`.
- [ ] Substituir todas as 3 ocorrências pela conversão BF16 correta.
- [ ] Recalibrar os golden vectors dos testes que usam `process_scalar` como referência.
- [ ] Considerar `#[deprecated]` no path escalar (x86-64-v3 garante AVX2 mínimo).

---

## Épico D — Refatoração e Higiene de Código

Tarefas que eliminam duplicação, melhoram legibilidade e reduzem risco de divergência entre cópias.

### Sprint 11: Refatoração da Infraestrutura

**Objetivo:** Eliminar código duplicado e melhorar a manutenibilidade dos módulos de infra.

#### Tarefa 11.1 · Extrair Helper Genérico de GC Push *(Refator P2)*

- **Arquivo:** `src/pw_host.rs`.
- **Diagnóstico:** O pattern `push no gc_producer → se cheio, parking lot → se cheio, leak/forget + flag` se repete 4 vezes.
- [ ] Extrair para função genérica:

  ```rust
  fn try_gc_push<T>(producer: &mut Producer<T>, item: T,
      parking: &mut [Option<T>; 8], rt_status: &RtStatusFlags) -> bool
  ```

- [ ] Substituir as 4 ocorrências (resamplers L295–336, model_l L362–378, model_r L380–397).
- [ ] Redução esperada: ~40 linhas.

#### Tarefa 11.2 · Extrair Helper de Parsing de Ganho na CLI *(Refator P1)*

- **Arquivo:** `src/cli.rs`.
- [ ] Extrair a lógica duplicada dos blocos `"gain"` e `"out"` (L203–231) para `parse_and_send_gain()`.
- [ ] Redução esperada: ~15 linhas.

#### Tarefa 11.3 · Agrupar Campos de Gate em `GateContext` *(Refator H3)*

- **Arquivo:** `src/dsp/pipeline.rs`.
- [ ] Criar `GateContext<'a>` agrupando: `gate_params`, `silence_hysteresis`, `mono_hysteresis`, `threshold_open_sq`, `threshold_close_sq`, `process_mono`.
- [ ] Reduzir `DspPipelineContext` de 14 para ~9 campos.
- [ ] Atualizar `apply_input_stage()`, `apply_output_stage()` e `capture_dsp_pipeline()`.

#### Tarefa 11.4 · Eliminar Clone de Metadata no Loader *(Refator S3)*

- **Arquivo:** `src/loader/mod.rs`, linha 83.
- [ ] Substituir `.clone()` por extração direta dos campos via `as_ref()`.

#### Tarefa 11.5 · Documentação e Limpeza de Boilerplate *(Higiene P5/P6/P8)*

- [ ] `src/spsc.rs`: Adicionar `///` documentando a partição entre flags no bitmask e campos atômicos dedicados.
- [ ] `src/diagnostics.rs` (L229–251): Considerar macro `detect_feature!` para os 6 blocos repetidos de `is_x86_feature_detected!`.
- [ ] `src/main.rs` (L64–73): Usar destructuring `let SpscChannels { .. } = channels;` em vez de 10 bindings individuais.

---

### Sprint 12: Legibilidade do Hot-Path DSP

**Objetivo:** Melhorar a legibilidade e documentação do código crítico de inferência neural.

#### Tarefa 12.1 · Refatorar Macro `define_lstm_process!` *(Higiene HS3)*

- **Arquivo:** `src/models/lstm.rs`.
- [ ] Documentar inline os 13 argumentos posicionais da macro (L24–38) com comentários descritivos nos call-sites (L225–299).
- [ ] Considerar struct `LstmProcessConfig` com campos nomeados, ou ao mínimo agrupar argumentos em blocos comentados.
- [ ] Remover blocos `unsafe` redundantes internos (a função gerada já é `unsafe`).

#### Tarefa 12.2 · Documentar Jitter de Offset no WaveNet Ring Buffer *(Higiene HH3)*

- **Arquivo:** `src/models/wavenet_common.rs` (L350–353).
- [ ] Adicionar comentário vinculando o cálculo de `jitter` ao `#[repr(align(64))]` de `WaveNetLayerState`, explicando que evita aliasing de cache lines entre estados de camadas adjacentes.

---

## Épico E — Blindagem da Suite de QA

Tarefas que expandem cobertura de testes, corrigem fragilidades e fortalecem fuzzing e benchmarks.

### Sprint 13: Infraestrutura de Testes — Unificação de Helpers

**Objetivo:** Eliminar duplicação entre test files e corrigir acumulação numérica imprecisa.

#### Tarefa 13.1 · Criar Módulo de Helpers Compartilhados *(Higiene TH1 + Bug TB2)*

- [ ] Criar `tests/common/mod.rs` com funções unificadas:
  - `generate_sine_440hz()`, `compute_mse()` (acumulação `f64`), `compute_max_abs_error()`, `model_path()`, `process_in_blocks()`.
- [ ] Atualizar `nam_infer_test.rs`, `dynamic_parity.rs`, `regression_goldens.rs` e `namb_v2_validation.rs` para importar do módulo.
- [ ] **Garantir que todos os testes usem acumulação `f64` para MSE** — corrige o bug TB2 onde `dynamic_parity.rs` (L19–31) e `regression_goldens.rs` (L29–37) acumulam em `f32`.
- [ ] Mover ou remover `compute_snr()` standalone com `#[allow(dead_code)]` (TH3).
- [ ] Redução esperada: ~80 linhas de duplicação.

---

### Sprint 14: Correções de Bugs em Testes

**Objetivo:** Corrigir bugs nos próprios testes que invalidam o que eles medem.

#### Tarefa 14.1 · Corrigir Threshold de Gate no Soak Test — dB→Linear *(Bug TB1)*

- **Arquivo:** `tests/soak_test.rs` (L502–503).
- **Diagnóstico:** A conversão está invertida — `(-60.0f32).powf(10.0/20.0)` tenta elevar uma base negativa a um expoente fracionário, produzindo NaN.
- [ ] Corrigir para: `10.0f32.powf(-60.0 / 20.0)` e `10.0f32.powf(-80.0 / 20.0)`.
- [ ] Verificar se a correção expõe bugs no Gate (o teste passará a exercitar transições `Open→Hold→Release` de fato).

#### Tarefa 14.2 · Unificar Threshold AVX-512 Dot Product no Proptest *(Bug TB5)*

- **Arquivo:** `tests/proptest_math.rs` (L161).
- **Diagnóstico:** A fórmula AVX-512 usa `scalar_result.abs()` que é frágil quando ≈0.
- [ ] Substituir pela fórmula baseada em `l1_norm` (consistente com AVX2 em L139):

  ```rust
  let l1_norm: f64 = vec_a.iter().zip(vec_b.iter())
      .map(|(&x, &y)| ((x as f64) * (y as f64)).abs()).sum();
  let threshold = 1e-2 * l1_norm.max(1.0);
  ```

---

### Sprint 15: Expansão de Cobertura de Testes

**Objetivo:** Fechar lacunas de cobertura identificadas na auditoria Test Review.

#### Tarefa 15.1 · Testes Unitários do DSP Pipeline *(Cobertura — Alta Prioridade)*

- [ ] Criar `src/dsp/pipeline_test.rs` (arquivo >300 linhas → testes isolados).
- [ ] Cobrir: Pipeline com Resampler vs Nativo, Mono vs Estéreo, Neural Ativado vs Bypass.
- [ ] Validar processamento isolado sem depender do daemon PipeWire.

#### Tarefa 15.2 · Golden Vectors LSTM 2×16 e 1×8 *(Lacuna TC5)*

- [ ] Estender `tests/fixtures/golden_gen_build.sh` para gerar `golden_lstm_2x16.bin` e `golden_lstm_1x8.bin`.
- [ ] Adicionar testes `test_golden_vectors_lstm_2x16()` e `test_golden_vectors_lstm_1x8()` em `nam_infer_test.rs`.
- [ ] Calibrar thresholds MSE/SNR e documentar em `tests/fixtures/README.md`.

#### Tarefa 15.3 · Proptest para `fused_lstm_gates` e Variantes Dual *(Lacuna TC6)*

- **Arquivo:** `tests/proptest_math.rs`.
- [ ] Property test para `fused_lstm_gates_avx2` validando contra referência escalar (4 gates).
- [ ] Property test para variantes `_dual_` de `simd_tanh` e `simd_sigmoid`, validando paridade.
- [ ] Thresholds: 5e-3 (tanh), 2e-5 (sigmoid).

#### Tarefa 15.4 · Zero-Allocation Test para LSTM Dinâmico *(Lacuna TC7)*

- **Arquivo:** `tests/nam_infer_test.rs`.
- [ ] Adicionar `test_zero_alloc_process_lstm_dynamic` via `build_lstm_dynamic`.
- [ ] Seguir padrão de `test_zero_alloc_process_wavenet_dynamic` — documentar com aviso se alocar.

---

### Sprint 16: Fuzzing do Hot-Path DSP

**Objetivo:** Assegurar resiliência das instruções SIMD e primitivas SPSC contra entradas adversariais.

#### Tarefa 16.1 · Novo Fuzz Target para Motor DSP *(Cobertura — Alta Prioridade)*

- [ ] Criar `fuzz/fuzz_targets/fuzz_dsp_process.rs`.
- [ ] Converter payload do libFuzzer em blocos `f32` (incluindo NaN, Inf, Denormals) e injetar em topologia WaveNet/LSTM padrão.
- [ ] Configurar timeout adequado no `fuzz/Cargo.toml`.
- [ ] Vetores de ataque obrigatórios: NaN nos pesos, dilation=0, CH=1 (undersize), CH=512 (oversize), hidden_size=1 (LSTM degenerado).

#### Tarefa 16.2 · Estender Fuzz Targets Existentes com `process()` *(Lacuna TF1/TF2)*

- **Arquivos:** `fuzz/fuzz_targets/fuzz_nam_json.rs`, `fuzz/fuzz_targets/fuzz_namb.rs`.
- **Diagnóstico:** Atualmente os fuzzers constroem o modelo (`build_model()`) mas nunca processam áudio, deixando o hot-path SIMD sem cobertura adversarial.
- [ ] Após `build_model()`, executar `model.prewarm()` e `model.process()` com buffer de teste.

---

### Sprint 17: Benchmarks — Expansão e Robustez

**Objetivo:** Expandir cobertura de benchmarks e prevenir otimizações espúrias do compilador.

#### Tarefa 17.1 · Benchmark do Pipeline Completo *(Lacuna)*

- **Arquivo:** `benches/inference_bench.rs`.
- [ ] Adicionar `bench_capture_dsp_pipeline_full` com blocos de 64 amostras englobando Gain, Gate e RingBuffer.
- [ ] Permite isolar o overhead da camada de orquestração vs o núcleo neural.

#### Tarefa 17.2 · Benchmarks para WaveNet Feather e Nano *(Lacuna TC10)*

- **Arquivo:** `benches/inference_bench.rs`.
- [ ] Adicionar `bench_wavenet_feather_process` (BossWN-feather.nam, 64 amostras).
- [ ] Adicionar `bench_wavenet_nano_process` (BossWN-nano.nam, 64 amostras).
- [ ] Registrar no `criterion_group!`. Detecta regressões de topologia (CH=4/8 vs CH=16).

#### Tarefa 17.3 · Adicionar `black_box` nos Benchmarks de Inferência *(Robustez TF3)*

- **Arquivo:** `benches/inference_bench.rs`.
- **Diagnóstico:** Os benchmarks de inferência (L97–103, L116–122, L233–238, L257–261) não usam `black_box` nos buffers de I/O, diferente dos soak tests e benchmarks de math que já o fazem.
- [ ] Envolver buffers `input` e `output` com `std::hint::black_box()` nos 4 benchmarks listados.

---

### Sprint 18: Limpeza de Código Vestigial em Testes

**Objetivo:** Remover artefatos de debug e código morto que poluem a suite de testes.

#### Tarefa 18.1 · Remover Código Vestigial *(Higiene TH5/TB3)*

- [ ] `tests/nam_infer_test.rs` (L46–83): Avaliar remoção do `TRACKING_ENABLED` vestigial do `CountingAllocator` — a guarda real é feita por `TRACKING_THREAD`.
- [ ] `tests/regression_goldens.rs` (L262–265): Substituir `.to_vec()` intermediário por iteração direta em `save_golden()`.
