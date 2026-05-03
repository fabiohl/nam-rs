# 🚀 Backlog do Produto e Planejamento de Sprints Técnicas

**Modelo de prompt web:** Você é uma equipe de arquitetos sêniores e desenvolvedores especialistas na linguagem Rust e no subsistema de áudio Linux. Em anexo temos o aglutinado repomix do github atualizado do NAM-rs (vide subanexo README.md). Assumindo a persona/workflow .agents/workflows/diagnostico.md e .agents/skills/planejador-arquiteto/SKILL.md você vai detalhar ao máximo a implementação da tarefa técnica demandada logo abaixo. Sua resposta será concisa e direta, usando toda a sua janela de processamento e contexto para orientar o trabalho do(s) implementador(es).

---

## Épico 6: Otimização Microarquitetural de Próxima Geração

> **Objetivo:** Extrair mais ciclos de clock por amostra, estreitando o gap entre a implementação atual e o limite teórico de throughput da microarquitetura x86-64-v3/v4.

### T18 — LSTM: Weight Layout Transposição para Dot Product Contíguo ✅

**Racional Científico:** O loop mais quente do LSTM (`define_lstm_process!`, linhas 76-88 de `lstm.rs`) itera `H` vezes com dot products de `IH` elementos cada. O layout interleaved `[H][IH][4]` impõe que cada iteração salte `IH×4×2 = IH×8 bytes` na memória — um stride que gera cache line splits em modelos `H≥16`. Uma transposição para um layout **"gate-major"** `[4][H][IH]` (4 gates × H neurons × IH weights), processado com um único dot product vectorizado `4H`-dimensonal por gate, eliminaria os H saltos discretos e permitiria streaming contíguo de pesos pela cache.

**Arquivo-alvo:** `src/models/lstm.rs`

**Implementação:**

1. Criar uma função `transpose_interleaved_to_gate_major()` em `src/loader/` que permuta o layout de pesos no cold-path (carregamento do modelo), de `[H][IH][4]` para `[gate][H×IH]` — 4 blocos contíguos de `H×IH` elementos.
2. Refatorar `define_lstm_process!` para consumir 4 slices lineares (`w_i`, `w_f`, `w_g`, `w_o`), cada um de comprimento `H×IH`, e computar o dot product completo de cada gate com um único `dot_product_batch` em vez de `H` chamadas unitárias.
3. Para H=16, IH=17: cada gate tem `16×17=272` elementos ≈ 17 vetores AVX2 — throughput teórico de ~34 ciclos FMA contra ~128 ciclos atuais (redução de ~4× no loop).
4. Atualizar benchmarks LSTM para medir melhoria e validar golden vectors inalterados.

**Critérios de aceite:** MSE dos golden vectors ≤ threshold atual. Benchmark LSTM_2x16_64samp ≤ 16 µs (atualmente ~20 µs).

---

### T19 — WaveNet Conv1D: Pesos Interleaved 4-Wide para Eliminação de Stride ✅

**Racional Científico:** O inner loop de `Conv1d::process_single_frame` (linhas 80-101 de `wavenet.rs`) calcula 4 dot products independentes (`dot_product_4x`). Porém, os pesos estão em layout `[OUT×K][IN]` e o loop precisa calcular 4 offsets `w0_start..w3_start` a cada iteração `out_c`. Um layout **interleaved** `[OUT/4][K][IN][4]` permitiria carregar os 4 pesos contiguamente com um único load (128 bits → 4 × f16), eliminando os 4 cálculos de offset e melhorando a localidade de cache.

**Arquivo-alvo:** `src/models/wavenet.rs`, `src/loader/`

**Implementação:**

1. Criar `interleave_conv1d_weights()` no cold-path (loader) que reorganiza o tensor `[OUT*K*IN]` para `[(OUT/4)*K*IN*4]`, colocando os pesos de 4 canais de saída adjacentes contiguamente.
2. Reescrever o inner loop de `process_single_frame` para iterar linearmente sobre blocos de `IN×4` pesos contíguos, extraindo 4 dot products com um único stream de leitura.
3. Tratar restos quando `OUT % 4 != 0` (Nano: CH=4 divide perfeitamente; Standard: CH=16 divide perfeitamente; Feather: CH=8 divide perfeitamente).
4. Aplicar a mesma estratégia para `process_single_frame_bf16`.

**Critérios de aceite:** Golden vectors inalterados. Benchmark WaveNet_Standard_64samp ≤ 145 µs (atualmente ~170 µs).

---

### T20 — WaveNet Mixin+Conv1D Fusão Temporal: "Ahead-of-Time Conditioning" ✅

**Racional Científico:** No `process_block_internal` de `WaveNetLayer` (linhas 449-494 de `wavenet.rs`), o loop `for i in 0..num_frames` executa para **cada frame**: (1) Conv1D, (2) soma Mixin, (3) Tanh, (4) Head update, (5) 1×1 residual. Passos 3-5 somam apenas `CH` operações escalares (~16 ops). Mas o Conv1D com dilatação alta executa dot products contra dados potencialmente frios na cache.

Uma inovação: pré-computar a **soma Conv1D + Mixin** em batch para múltiplos frames antes de aplicar a ativação. Isso melhora a reutilização do banco de coeficientes do Conv1D na cache ao processar frames sequenciais antes de acionar as ativações.

**Arquivo-alvo:** `src/models/wavenet.rs`

**Implementação:**

1. Criar buffer stack `conv_plus_mixin: [f32; 1024]` (MAX_FRAMES × CH).
2. Primeiro loop: calcular `conv1d + mixin` para todos os frames, escrevendo no buffer intermediário.
3. Segundo loop: aplicar `activation_tanh_block` em batch sobre todo o buffer (melhor vetorização — 64×16 = 1024 elementos contíguos vs. 16 por chamada atual).
4. Terceiro loop: head update + 1×1 residual (sem mudança).
5. **Benchmark especial:** Medir a contagem de L1 cache misses via `perf stat` antes e depois.

**Critérios de aceite:** Golden vectors inalterados. Cache miss ratio reduzido ≥15% para WaveNet Standard em `perf stat -e L1-dcache-load-misses`.

---

### T21 — GEMV AVX-512: Convolução 1×1 com Registradores ZMM (16-wide) ✅

**Racional Científico:** As funções `fused_add_gemv_avx512` e `gemv_overwrite_avx512` em `simd.rs` (linhas 2338-2434) processam 16 floats por iteração — porém suas invocações pela WaveNet Dense Layer (`CH=16`) resultam em exatamente **1 iteração** do loop SIMD + 0 tail. Isso é ótimo, mas o overhead de setup (broadcast de `in_frame[in_c]` para ZMM, carga do ponteiro de pesos) domina. Uma versão **especializada para CH≤16** que desdobra o loop `in_len` com register renaming explícito (2 acumuladores ZMM alternados) pode eliminar ~30% do overhead.

**Arquivo-alvo:** `src/math/simd.rs`

**Implementação:**

1. Criar `gemv_overwrite_avx512_small<const OUT: usize>()` onde `OUT <= 16`, que mantém o acumulador permanentemente em ZMM0 sem stores intermediários.
2. Loop `in_c` desenrolado ×4: carregar 4 vetores de pesos consecutivos e 4 broadcasts de `in_frame[in_c..in_c+4]` por iteração, explorando o rename file (32 ZMM disponíveis em AVX-512).
3. Despachar via `SimdMath::gemv_overwrite` quando `OUT <= 16` detectado em compile-time via const generics.
4. Benchmark isolado do kernel GEMV antes/depois.

**Critérios de aceite:** Benchmark GEMV `16×16` ≤ 15 ns (estimar baseline atual). Golden vectors inalterados.

---

### T22 — Resampler: Convolução AVX-512 com FMA Interleaved Stereo ✅

**Racional Científico:** O resampler em `resampler.rs` processa L e R sequencialmente — `convolve_fir` é chamado 2× por amostra (uma vez por canal). Os coeficientes do filtro são idênticos para ambos os canais. Uma convolução **stereo interleaved** carregaria os coeficientes uma única vez e aplicaria FMA com 2 acumuladores independentes (L, R), cortando pela metade o bandwidth de coeficientes do L1 e eliminando a redundância de loads.

**Arquivo-alvo:** `src/dsp/resampler.rs`

**Implementação:**

1. Criar `convolve_fir_stereo_avx2()` que aceita ponteiros de delay line de L e R.
2. Loop principal: `_mm256_loadu_ps(coeffs)` uma vez, `_mm256_loadu_ps(state_l)` + FMA, `_mm256_loadu_ps(state_r)` + FMA, compartilhando o vetor de coeficientes.
3. Variante AVX-512: `convolve_fir_stereo_avx512()` processando 16 taps por iteração.
4. Integrar em `ResamplerCore` alterando `process_input`/`process_output` para processar ambos os canais em chamada única.
5. Benchmark: `Resampler_44100_to_48k_1024samp` deve cair ~35-40%.

**Critérios de aceite:** Testes de resampling com impulso e roundtrip passam. Benchmark resampler ≤ 14 µs (atualmente ~21 µs).

---

## Épico 7: Arquitetura WaveNet de Segunda Geração

> **Objetivo:** Evolução arquitetural do modelo WaveNet para eliminar gargalos estruturais herdados do C++ e explorar oportunidades únicas do Rust.

### T23 — WaveNet Ring Buffer: Eliminação do Rewind via Mapeamento Virtual ✅

**Racional Científico:** O `WaveNetLayerState::rewind_buffer()` (linhas 571-587 de `wavenet.rs`) executa `copy_within` de `receptive_field_size × CH` floats a cada ~1536 amostras. Para WaveNet Standard (RF=1024, CH=16), isso significa copiar `1024×16×4 = 64 KB` a cada ~32 ms — um spike de ~6-10 µs que pode coincidir com um callback curto (32 samples = ~667 µs budget). A técnica de **virtual ring buffer** via `mmap` com mirror (2 mapeamentos virtuais contíguos do mesmo backing store físico) elimina o rewind completamente.

**Arquivo-alvo:** `src/dsp/`, novo arquivo `src/dsp/vring.rs`

**Implementação:**

1. Criar `VirtualRingBuffer<T>` em `src/dsp/vring.rs` usando `mmap` + `MAP_FIXED|MAP_SHARED` para mapear 2 janelas virtuais contíguos do mesmo `memfd_create` (ou `shm_open`).
2. O buffer é visto como 2×N contíguo; o ponteiro avança linearmente e, ao atingir N, o acesso "wraparound" lê a mesma memória física via o segundo mapeamento.
3. Migrar `WaveNetLayerState::layer_buffer` de `Vec<f32>` para `VirtualRingBuffer<f32>`.
4. Remover `rewind_buffer()` — o avanço é puramente linear sem cópia.
5. **ATENÇÃO RT-SAFETY:** O `mmap` é feito no cold-path (`new()`). No hot-path, é apenas aritmética de ponteiro.

**Critérios de aceite:** Testes WaveNet passam. Benchmark `WaveNet_Standard_CH16_512samp` sem outliers de rewind. Zero syscalls no hot-path.

---

### T24 — WaveNet: Skip-Connection Accumulator em Registrador ✅

**Racional Científico:** O head accumulator (`head_input`) é atualizado amostra-a-amostra com `head_input[i*CH + j] += temp[j]` (linhas 480-483 de `wavenet.rs`). Para CH=16, são 16 stores para memória a cada frame. Se o acumulador fosse mantido em **registradores YMM/ZMM** (2 registradores YMM para 16 floats), os 16 stores/loads seriam eliminados, mantendo o acumulador "in-flight" entre camadas.

**Arquivo-alvo:** `src/models/wavenet.rs`

**Implementação:**

1. Alterar `process_block_internal` para usar variáveis locais `head_accum_0: __m256` e `head_accum_1: __m256` (ou `__m512` para AVX-512) como acumuladores em vez de ponteiro para `head_input`.
2. Carregar o acumulador em registradores **antes** do loop de camadas. Acumular via FMA in-register durante as camadas. Store final único para `head_input` após a última camada.
3. Para `num_frames > 1`, manter array de acumuladores na stack: `[[__m256; 2]; MAX_FRAMES]` — 64 × 32B = 2 KB na stack, cabe no L1.

**Critérios de aceite:** Golden vectors inalterados. Benchmark WaveNet com `num_frames=1` (bloco unitário) melhora ≥10%.

---

### T25 — WaveNet BF16 Pipeline: Conversão Lazy com Dirty Flag ✅

**Racional Científico:** Atualmente, `f32_to_bf16` é chamado incondicionalmente para cada chunk escrito no `layer_buffer` (linhas 659-663, 698-703 de `wavenet.rs`), mesmo quando o dispatch ativo é AVX2 (que não usa BF16). Isso desperdiça ~2-4 µs por callback em CPUs sem AVX-512-BF16. Uma flag `is_bf16_active` resolvida no startup (já temos `SimdMathConfig`) eliminaria esta conversão redundante em CPUs v3.

Adicionalmente, para CPUs BF16: a conversão poderia ser **fundida** com o store ao `layer_buffer`, usando uma instrução `_mm512_cvtneps_pbh` inline, evitando o loop separado.

**Arquivo-alvo:** `src/models/wavenet.rs`

**Implementação:**

1. Adicionar `const IS_BF16_ACTIVE: bool = M::IS_BF16` ao contexto de chamada. Encapsular as chamadas `f32_to_bf16` em `if M::IS_BF16 { ... }` — o compilador eliminará o código morto via monomorfização.
2. **Verificar:** Este padrão já existe parcialmente (linhas 659-663 usam `if M::IS_BF16`). Confirmar que **todas** as conversões BF16 no `process_block_internal` e `prewarm_internal` estão gateadas.
3. Fundir a conversão BF16 com o store no mesmo loop que escreve em `layer_buffer` — evitando segunda passada sobre os dados.

**Critérios de aceite:** Em CPUs sem AVX-512-BF16, zero ciclos gastos em conversão BF16. Benchmark em CPU v3 sem regressão.

---

## Épico 8: Excelência Operacional e Observabilidade RT

> **Objetivo:** Hardening do runtime, melhor telemetria e experiência de operação.

### T26 — Telemetria RT: Histograma de Latência per-Callback via RDTSC ✅

**Racional Científico:** O sistema atual armazena apenas o **último** `dsp_cycle_time` e um contador de overloads. Isso é insuficiente para detectar jitter intermitente (ex: spikes de 2× que não excedem o budget mas degradam a experiência). Um histograma exponencial (~256 bytes) mantido in-place no callback RT capturaria a distribuição completa de latências sem I/O.

**Arquivo-alvo:** `src/pw_host.rs`, `src/rt_setup.rs`

**Implementação:**

1. Criar `LatencyHistogram` com 32 bins exponenciais (2^5 a 2^37 ns), cada bin um `AtomicU32`. Tamanho total: 128 bytes (2 cache lines).
2. No callback RT, `RDTSC` start/end já existe. Converter `elapsed_nanos` em bin index via `63 - leading_zeros(elapsed_nanos)` (1 instrução BSR/LZCNT) e incrementar atomicamente.
3. No `poll_rt_status` (main thread), ler o histograma e calcular P50, P95, P99, Max.
4. Exibir no status periódico: `DSP: P50=45µs P95=62µs P99=110µs Max=180µs`.
5. **RT-SAFETY:** Apenas `AtomicU32::fetch_add` (lock-free) e uma instrução de bitshift. Zero alocação.

**Critérios de aceite:** Histograma funcional. Testes unitários para cálculo de percentis. Zero impacto mensurável no benchmark.

---

### T27 — SPSC GC Channel: Eliminação do Drop no Hot-Path ✅

**Racional Científico:** Em `pw_host.rs` linhas 443-448, quando um modelo é swapped via `std::mem::replace`, o modelo antigo é enviado ao `gc_producer`. Porém, se o GC SPSC estiver cheio, o drop implícito causa `free()` no RT. A solução implementa um "Parking Lot" na stack e um fallback de leak intencional em casos patológicos.

**Arquivo-alvo:** `src/pw_host.rs`, `src/spsc.rs`

**Implementação:**

1. [x] Implementação de GC para `NamResampler` (evita `free()` no swap de rate).
2. [x] Implementação de "Parking Lot" na stack do callback RT (buffer de 8 slots).
3. [x] Implementação de Fallback Determinístico: `Box::leak` / `std::mem::forget` em overflow crítico.
4. [x] Adição de flag `gc_overflow` em `RtStatusFlags` para observabilidade.

5. Aumentar o GC channel de 4 para 8 slots (ou ajustar dinamicamente).
6. **Alternativa superior:** Criar um "GC parking lot" — um array fixo de `Option<Box<dyn NamModel>>` de tamanho 8 na stack do closure. Se o SPSC GC estiver cheio, estacionar o modelo no parking lot. Na próxima iteração do `process()`, tentar drenar o parking lot para o GC.
7. Se o parking lot também estiver cheio (situação patológica: 8 swaps consecutivos sem dreno), leak o modelo — preferível a invocar `free()` no hot-path.
8. Sinalizar via `RtStatusFlags::gc_overflow` para diagnóstico.

**Critérios de aceite:** Zero chamadas `free()` no callback RT mesmo sob stress-test de hot-swap rápido (100 swaps/segundo).

---

### T28 — Carregamento de Modelo: Formato `.namb` com Pesos Já Transpostos ✅

**Racional Científico:** Atualmente, o `.namb` armazena os pesos no layout original do `.nam` (JSON). As transposições propostas em T18/T19 seriam executadas a cada carregamento. Evoluir o formato `.namb` para armazenar os pesos **já no layout otimizado para o kernel** eliminaria ~10-50 ms de CPU no cold-path de carregamento, tornando o hot-swap de modelos imperceptível.

**Arquivo-alvo:** `src/loader/namb.rs`

**Implementação:**

1. Incrementar a versão do formato `.namb` de `1` para `2`.
2. Adicionar flag no header indicando layout dos pesos: `0 = original`, `1 = gate-major (LSTM)`, `2 = interleaved-4 (WaveNet)`.
3. O loader v2 detecta o layout e pula a transposição se já otimizado.
4. Ferramenta de conversão (teste `#[ignore]` em `regression_goldens.rs`) para re-gerar `.namb` com layout otimizado.
5. Manter backward-compatibility: loader v2 aceita `.namb` v1 (transpondo on-the-fly).

**Critérios de aceite:** `.namb` v2 carrega sem transposição. `.namb` v1 continua funcionando.

---

### T29 — `Instant::now()` → `RDTSC` Direto no Callback RT ✅

**Racional Científico:** `Instant::now()` (linha 568 de `pw_host.rs`) invoca `clock_gettime(CLOCK_MONOTONIC)`, que é uma syscall vDSO (~15-25 ns). Em callbacks de 32 samples (~667 µs budget), 25 ns × 2 (start+end) = 50 ns é ~0.007% do budget — aceitável mas evitável. `RDTSC` é uma instrução de 1 ciclo (~0.3 ns) com resolução de ~1 ns via calibração.

**Arquivo-alvo:** `src/pw_host.rs`, `src/rt_setup.rs`

**Implementação:**

1. [x] Criar `rdtsc_nanos()` inline: `_rdtsc()` + divisão por `tsc_freq_ghz` (constante calibrada no startup via `/proc/cpuinfo` ou `calibrated_tsc_freq`).
2. [x] Calibrar `tsc_freq_ghz` no cold-path: ler `CPUID.15H` (TSC frequency) ou medir com `Instant::now()` + `_rdtsc()` em loop calibrado.
3. [x] Substituir `Instant::now()` no callback por `rdtsc_nanos()`.
4. [x] Fallback: se TSC invariante não disponível (`!is_x86_feature_detected!("tsc")`), manter `Instant::now()`.

**Critérios de aceite:** Zero syscalls no callback RT para medição de tempo. Erro de calibração < 1%.

---

### T30 — `pw_host.rs` Modularização: Extrair `capture_dsp_pipeline` ✅

**Racional Científico:** `pw_host.rs` tem 1169 linhas — o maior arquivo do projeto. A função `capture_dsp_pipeline` e seu `DspPipelineContext` formam uma unidade coesa testável que deveria viver em seu próprio módulo (`src/dsp/pipeline.rs`). Isso melhora legibilidade, habilita testes unitários isolados do pipeline DSP, e reduz o tempo de compilação incremental.

**Arquivo-alvo:** `src/pw_host.rs` → `src/dsp/pipeline.rs`

**Implementação:**

1. Mover `DspPipelineContext`, `capture_dsp_pipeline()` e helpers relacionados (energy computation, mono detection) para `src/dsp/pipeline.rs`.
2. Manter em `pw_host.rs` apenas a orquestração PipeWire (streams, listeners, bridge).
3. Criar testes unitários para `capture_dsp_pipeline` com mocks de modelo e resampler.
4. Garantir que `pub(crate)` mantenha encapsulamento.

**Critérios de aceite:** `pw_host.rs` ≤ 800 linhas. Todos os testes passam. Nenhuma mudança na API pública.

---

### T31 — `main.rs` Refatoração: Extração da Lógica CLI ⬜

**Racional Científico:** `main.rs` (19 KB) mistura parsing de argumentos, lógica de carregamento de modelo, CLI interativa, e orquestração PipeWire. Extrair o CLI interativo para `src/cli.rs` (que já existe com 6 KB) e a lógica de setup/orquestração para funções nomeadas melhora testabilidade e legibilidade.

**Arquivo-alvo:** `src/main.rs` → `src/cli.rs`

**Implementação:**

1. Mover o loop interativo (`handle_cli_input`, parsing de comandos) de `main.rs` para `cli.rs`.
2. Extrair `load_and_build_model()` como função pública em `src/loader/mod.rs`.
3. `main()` deve ficar com ~100-150 linhas: arg parsing → load → pw_host::run → loop CLI.
4. Preservar tratamento de Ctrl-C e shutdown graceful.

**Critérios de aceite:** `main.rs` ≤ 300 linhas. Compilação sem erros. Funcionalidade inalterada.
