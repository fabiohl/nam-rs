<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO — Sprints de Integração CLAP

---

## Sprint 0 — Preparação Crítica: Correções RT-Safety e Fundação CLAP

**Objetivo:** Corrigir vulnerabilidades de segurança RT identificadas pela auditoria (SIGSEGV em blocos < 8 samples), compartilhar o pipeline DSP entre features `standalone` e `clap-plugin` sem duplicação de código, e estabelecer cobertura de testes para edge cases que o modo CLAP vai exercitar.

**Pré-requisito:** Nenhum. Este sprint é executável imediatamente sobre a base de código v1.4.5.

**Origem:** Análise consolidada Pesquisador-Inovador × Revisor-Auditor (2026-05-14).

---

### Épico 0.1 — Correção Crítica de RT-Safety: Fallback Escalar para Blocos < 8 Samples

> **Severidade: CRÍTICA.** Sem esta correção, o modo CLAP não pode ser considerado production-ready. O Bitwig pode enviar blocos de 1, 3 ou 7 samples durante automação sample-accurate.

- [x] **Tarefa 0.1.1** — Fallback Escalar em `compute_energy_stereo` e `compute_max_diff_avx2`

  - **Contexto:** As funções SIMD em `src/math/dsp/stereo` operam em vetores de 8 floats (1 `__m256`). Para `n < 8`, ocorre leitura out-of-bounds que causa SIGSEGV. No modo Standalone com PipeWire, os blocos são tipicamente ≥ 64 samples, mascarando o bug. No CLAP, o host pode subdividir blocos arbitrariamente.
  - **Ações Técnicas:**
    1. Em `compute_energy_stereo`, adicionar branch `if n < 8` com implementação escalar (loop simples sobre `n` amostras).
    2. Em `compute_max_diff_avx2`, adicionar branch `if n < 8` com implementação escalar.
    3. O branch é altamente previsível (blocos < 8 são raros) — impacto zero no caminho principal.
    4. Adicionar `#[cold]` no caminho escalar para ajudar o compilador a priorizar o hot-path SIMD.
  - **Aceite:** `cargo test` passa com blocos de 0, 1, 3, 5, 7, 8 e 9 samples. Zero SIGSEGV. Benchmark confirma zero regressão para blocos ≥ 8.

- [x] **Tarefa 0.1.2** — Validação do Gate FSM com `n_samples = 1`

  - **Contexto:** Na `DynamicHysteresis::update()` (`src/dsp/gate.rs`), para `n_samples = 1`, o cálculo de rampa `step = (end - start) / 1.0` resulta em salto abrupto. Embora um salto de 1 sample (~20.8µs @ 48kHz) seja imperceptível, a corretude formal deve ser documentada.
  - **Ações Técnicas:**
    1. Adicionar testes unitários em `gate_test.rs` exercitando `update()` e `apply_gain_rt_stereo()` com `n_samples = 1` em todas as transições de estado (Open→FadingOut, FadingOut→Closed, Closed→FadingIn, FadingIn→Open).
    2. Verificar que os contadores `hold_counter` e `fade_counter` não sofrem overflow ou underflow.
    3. Documentar no código-fonte que o comportamento de salto abrupto para `n_samples = 1` é **aceito por design** (imperceptível a 48kHz).
  - **Aceite:** Todos os testes de edge case passam. Comentário `///` no código documenta a decisão.

---

### Épico 0.2 — Compartilhamento do Pipeline DSP: Feature Gates e Higiene

> **Desbloqueio.** Este épico é precondição para toda a Sprint 2 CLAP (pipeline DSP vivo).

- [x] **Tarefa 0.2.1** — Refatorar Feature Gates em `src/dsp/pipeline.rs`

  - **Contexto:** O hot-path completo (`apply_input_stage`, `run_inference`, `apply_output_stage`, `DspPipelineContext`, constantes `MAX_*`) está gateado em `#[cfg(any(feature = "standalone", test))]`, excluindo o build CLAP. A solução deve gerar compartilhamento seguro de código sem duplicações.
  - **Ações Técnicas:**
    1. Alterar os atributos `#[cfg(...)]` das funções core (`apply_input_stage`, `run_inference`, `apply_output_stage`) e das structs/constantes compartilhadas (`DspPipelineContext`, `MAX_BRIDGE_BUF`, `MAX_RESAMP_BUF`, `BridgeBuffer`) para `#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]`.
    2. Tornar `bridge_ptr` em `DspPipelineContext` condicional via `#[cfg(feature = "standalone")]`, com uma variante de construção que não exige o campo no CLAP.
    3. A função `write_bridge()` permanece exclusiva do Standalone (`#[cfg(feature = "standalone")]`).
    4. `AppState`, `PipewireHostConfig`, `playback_dsp_cycle`, `build_spa_format_pod` permanecem exclusivos do Standalone — **não alterar** estes gates.
    5. `capture_dsp_pipeline()` permanece Standalone-only (orquestra os 4 estágios + bridge). O CLAP invocará os estágios individuais diretamente no `process()`.
    6. Executar `cargo check --features standalone` E `cargo check --no-default-features --features clap-plugin --lib` — ambos devem compilar sem erros.
  - **Aceite:** Ambos os builds compilam com zero erros e zero warnings. Os 3 estágios do pipeline estão acessíveis sob `feature = "clap-plugin"`.

- [x] **Tarefa 0.2.2** — Eliminação de `#[allow(clippy::not_unsafe_ptr_arg_deref)]` em `pipeline.rs`

  - **Contexto:** As funções `capture_dsp_pipeline` e `handle_silence_bypass` usam `*mut DspBridge` com `#[allow]` para silenciar o lint. Ao refatorar o `bridge_ptr` (Tarefa 0.2.1), aproveitar para encapsular o acesso unsafe.
  - **Ações Técnicas:**
    1. Encapsular `*mut DspBridge` em um newtype `BridgeRef` (ou similar) que garante non-null via `debug_assert!` no construtor.
    2. Remover os `#[allow(clippy::not_unsafe_ptr_arg_deref)]` das funções refatoradas.
    3. Manter a assinatura `unsafe` apenas no construtor do `BridgeRef`.
  - **Aceite:** Zero `#[allow(clippy::not_unsafe_ptr_arg_deref)]` em `pipeline.rs`. `lints.sh` limpo.

---

### Épico 0.3 — Cobertura de Testes para Blocos Irregulares

- [x] **Tarefa 0.3.1** — Suite de Testes com Block Sizes Não-Convencionais

  - **Contexto:** Os testes atuais em `pipeline_test.rs` (26 kB) exercitam apenas block sizes padrão (256, 512). No CLAP, o Bitwig subdivide blocos para automação sample-accurate (1, 7, 17, 33, 53...).
  - **Ações Técnicas:**
    1. Adicionar testes parametrizados em `pipeline_test.rs` (ou novo arquivo `pipeline_block_test.rs` se exceder 300 linhas) exercitando o pipeline completo com a sequência de block sizes: `[1, 3, 7, 8, 9, 17, 33, 53, 64, 128, 256, 512]`.
    2. Para cada block size: processar com modelo LSTM e WaveNet (usando fixtures do `tests/nam_files/`). Verificar: zero panics, zero SIGSEGV, output não é NaN/Inf.
    3. Adicionar cenário com `CountingAllocator` para `n_samples = 1` e `n_samples = max_frames_count` confirmando zero alocações no hot-path.
    4. Integrar com `proptest` para gerar block sizes aleatórios no range `[1, 8192]` com 500 iterações.
  - **Aceite:** `cargo test --features standalone` passa. Todos os block sizes exercitados sem crash. `CountingAllocator::alloc_count() == 0` para todos os cenários.

> **Auditoria:** Sprint 0 revisada com sucesso. Os fallbacks escalares `#[cold]` foram garantidos nas raízes SIMD (`compute_energy_stereo_avx2`, etc.) evitando overhead de redução horizontal em blocos ínfimos. A base de código está com `lints.sh` 100% limpo, sem warnings `dead_code` ou `unsafe_ptr_arg_deref`, e pronta para sustentar a implementação do host CLAP (Sprint 1) sem surpresas de estabilidade.

---

## Sprint 1 — Fundação do Plugin: Build, Conformidade e Bypass Estável

**Objetivo:** Garantir que o esqueleto CLAP compile deterministicamente, passe 100% na suite do `clap-validator` sem levantar UI, e seja detectado/carregado pelo Bitwig Studio 6 com bypass puro RT-safe (zero alocações, zero locks, zero crashes).

**Estado atual do código (`src/clap/`):**

- `descriptor.rs` (23L): funcional — ID reverso-DNS, features CLAP corretas.
- `plugin.rs` (45L): `NamClapShared` e `NamClapMainThread` são structs unit vazias. Sem estado.
- `processor.rs` (47L): `NamClapProcessor` sem campos. `activate()` retorna `Ok(Self)` vazio. `process()` faz bypass cópia simples.
- **Zero** extensões CLAP implementadas (`params`, `audio_ports`, `state`, `gui`, `latency`).

---

### Épico 1.0 — Infraestrutura de Build e Lint Limpo

> **Pré-requisito de toda a Sprint.** Nenhuma tarefa subsequente deve ser executada até que este épico passe com zero warnings.

- [x] **Tarefa 1.0.1** — Verificação de Compilação Limpa da Feature `clap-plugin`

  - **Contexto:** O build da `.so` usa `--no-default-features --features clap-plugin`. Qualquer warning de compilação é sinal de dívida técnica que irá escalar.
  - **Ações Técnicas:**
    1. Executar `cargo check --no-default-features --features clap-plugin --lib 2>&1 | grep -E "^error|warning"`.
    2. Corrigir **todos** os warnings (unused imports, dead_code, etc.) em `src/clap/`.
    3. Confirmar que a compilação do binário standalone (`cargo check --features standalone`) continua sem erros — os dois builds devem coexistir.
    4. Executar `./utils/lints.sh` e confirmar saída limpa para ambas as features.
  - **Aceite:** `cargo check` em ambas as features → 0 errors, 0 warnings. `lints.sh` retorna 0.

- [x] **Tarefa 1.0.2** — Auditoria do `utils/build-clap.sh` e Ciclo de Deploy

  - **Contexto:** O script já existe e faz o build release + cópia para `~/.clap/nam-rs.clap`. Porém, ausência de validação pós-cópia pode ocultar instalações silenciosamente corrompidas.
  - **Ações Técnicas:**
    1. Adicionar ao script: verificação de `readelf -d ~/.clap/nam-rs.clap | grep SONAME` para confirmar que é uma `.so` válida.
    2. Adicionar verificação de símbolo de entrada CLAP: `nm -D ~/.clap/nam-rs.clap | grep clap_entry` deve retornar exatamente 1 linha.
    3. Adicionar verificação `file ~/.clap/nam-rs.clap` confirma `ELF 64-bit LSB shared object, x86-64`.
    4. Adicionar ao script um modo de build debug (`--debug`) para acelerar iterações de desenvolvimento.
  - **Aceite:** Script executa, instala e auto-valida sem intervenção manual. Saída clara indica sucesso ou falha específica.

- [x] **Tarefa 1.0.3** — Garantia de Zero-Alocação no Bypass (`CountingAllocator`)

  - **Contexto:** O `process()` atual faz `copy_from_slice` sobre slices do host — teoricamente zero-alloc. Mas `clack-plugin` pode ocultar alocações internas na deserialização de events. Precisamos de prova formal.
  - **Ações Técnicas:**
    1. Em `src/clap/processor.rs`, adicionar um teste de integração em `#[cfg(test)]` que instancia o plugin via `clack-host` (dev-dependency) e invoca o `process()` com um bloco de 512 amostras usando o `CountingAllocator` já presente no projeto (`tests/nam_infer_test.rs`).
    2. O teste deve falhar se `CountingAllocator::alloc_count()` for > 0 após o `process()`.
    3. Seguir a convenção: arquivo `src/clap/processor_test.rs` (arquivo < 300L → inline em `processor.rs`).
  - **Aceite:** `cargo test --no-default-features --features clap-plugin` passa. O teste de zero-alloc do bypass é verde.

---

### Épico 1.1 — Validação Estrutural com `clap-validator` (Off-Host, Zero-UI)

> **Filosofia:** Nunca leve um plugin ao Bitwig sem antes passar no validador oficial do spec. Economiza horas de debug.

- [x] **Tarefa 1.1.1** — Instalação e Execução do `clap-validator`

  - **Contexto:** `clap-validator` é a ferramenta CLI oficial do ecossistema CLAP que testa conformidade estrutural sem necessitar de uma DAW. Deve ser o portão de qualidade de toda a Sprint 1.
  - **Ações Técnicas:**
    1. Instalar: `cargo install clap-validator` (ou via binário pré-compilado de `github.com/free-audio/clap-validator`).
    2. Após build: `clap-validator validate ~/.clap/nam-rs.clap 2>&1 | tee /tmp/clap-validator-output.txt`.
    3. Analisar cada item `FAIL` ou `WARN` na saída — criar sub-tarefas por item se necessário.
  - **Aceite:** `clap-validator validate` retorna **0 FAILs**. Warnings devem ser avaliados individualmente e documentados se aceitos.

- [x] **Tarefa 1.1.2** — Corrigir Falhas de Conformidade Identificadas pelo Validador

  - **Contexto:** Falhas comuns esperadas no esqueleto atual (sem extensões): ausência de `clap_plugin_audio_ports` (obrigatória pelo spec CLAP 1.x para plugins de efeito), e potencial falha no teste de `process()` com eventos nulos.
  - **Ações Técnicas:**
    1. Implementar a extensão `clap_plugin_audio_ports` mínima em `src/clap/extensions/audio_ports.rs`:
       - Declarar 1 porta de entrada e 1 de saída, ambas stereo (`channel_count = 2`, `port_type = CLAP_PORT_STEREO`).
       - Nomear a porta: `"Main"`. Marcar `is_main = true`. Sinalizar `in_place_pair = 0`.
       - Retornar `true` para `supports_inplace()`.
    2. Registrar a extensão via `impl Plugin for NamClapPlugin { fn extension(&self, id: &CStr) -> ... }`.
    3. Reexecutar `clap-validator validate` — confirmar que a falha de audio_ports foi resolvida.
  - **Referência de implementação:** `clack-extensions` já é dependência. Usar trait `PluginAudioPorts`.
  - **Aceite:** `clap-validator` 0 FAILs após a correção.

- [x] **Tarefa 1.1.3** — Teste Headless de Ciclo de Vida Completo via `clack-host`

  - **Contexto:** `clap-validator` testa o spec. `clack-host` (dev-dep) permite simular o lifecycle completo (init → activate → process → deactivate → destroy) em um teste unitário Rust, sem DAW, com output determinístico.
  - **Ações Técnicas:**
    1. Criar `tests/clap_lifecycle_test.rs` com o seguinte roteiro:
       - Carregar `~/.clap/nam-rs.clap` via `clack-host` `PluginBundle::load()`.
       - Instanciar o plugin com `sample_rate = 48000.0`, `max_frames_count = 512`.
       - Invocar `activate()` → verificar que não panics e retorna `Ok`.
       - Enviar 4 blocos de 512 amostras de silêncio no `process()`.
       - Invocar `deactivate()` → `destroy()`.
    2. Adicionar header SPDX ao arquivo de teste.
    3. Adicionar ao `utils/tests-cargo.sh` (se existir) ou documentar como executar.
  - **Aceite:** `cargo test --test clap_lifecycle_test --no-default-features --features clap-plugin` → verde. Zero panics, zero UB (rodar com `RUSTFLAGS="-Zsanitizer=address"` em build nightly de validação).

> **Auditoria:** Épico 1.1 revisado com sucesso. A infraestrutura básica e a extensão obrigatória `clap_plugin_audio_ports` foram implementadas. O validador oficial (`clap-validator`) retorna 0 FAILs e 0 WARNINGs (8 testes passados, 13 ignorados por não se aplicarem ao momento atual). Além disso, a cobertura com o `clack-host` no `clap_lifecycle_test` assegura as garantias de ciclo de vida e estabilidade sob testes controlados. O plugin CLAP já possui fundação estável e "compliant" para seguir ao Épico 1.2 (Host Real).

---

### Épico 1.2 — Validação no Bitwig Studio 6 (Host Real)

- [x] **Tarefa 1.2.1** — Scan, Categorização e Instanciação no Bitwig 6.0.6

  - **Contexto:** Bitwig escaneia plugins em processo separado (sandbox) e reporta erros na aba *Plug-in Errors*. O plugin deve sobreviver ao scan sem falhas de inicialização.
  - **Pré-requisito:** Épicos 1.0 e 1.1 concluídos com zero FAILs.
  - **Ações Técnicas:**
    1. Executar `./utils/build-clap.sh` → confirmar instalação via saída do script (Tarefa 1.0.2).
    2. No Bitwig: *Settings → Plug-ins → Re-scan* ou *Scan for new plug-ins*.
    3. Verificar: plugin aparece na lista com nome `"NAM-rs Neural Amp Modeler"`, vendor `"Fabio Lima"`, categorias `audio-effect / distortion`.
    4. Verificar ausência de entradas na aba *Plug-in Errors* do Bitwig.
    5. Inserir o plugin em uma Device Chain em uma trilha de instrumentos com sinal de áudio.
  - **Aceite:** Plugin listado corretamente. Inserível sem erros. Aba *Plug-in Errors* vazia.

- [x] **Tarefa 1.2.2** — Bypass Puro Estéreo com Áudio Real

  - Vide chat "Validating NAM-rs Bitwig Integration" para operacional detalhado.
  - **Contexto:** Com bypass, cada amostra de input deve chegar ao output inalterada. Qualquer diferença indica bug no mapeamento `ChannelPair`.
  - **Ações Técnicas:**
    1. No Bitwig, conectar uma fonte de áudio com sinal DC ou tone conhecido (ex: gerador de sine 440Hz via "Test Tone" device).
    2. Comparar entrada e saída do plugin usando o *Spectrum Analyser* e o *Oscilloscope* do Bitwig — a forma de onda deve ser idêntica.
    3. Verificar que `ChannelPair::InPlace(_)` não silencia o sinal (comportamento atual no `processor.rs:40`):
       - **Bug identificado:** O braço `InPlace` do match faz `{}` (noop) — o que é correto para in-place (o buffer é o mesmo), mas deve ser documentado explicitamente com um comentário.
    4. Verificar que `ChannelPair::InputOnly` não propaga lixo (atual: noop — correto, host não lê saída nesse caso).
  - **Aceite:** Sine 440Hz de entrada = sine 440Hz de saída. Latência reportada = 0 ms. Oscilloscope mostra sobreposição perfeita.

- [x] **Tarefa 1.2.3** — Stress de Sandboxing e Modos de Hosting

  - **Contexto:** Bitwig oferece 3 modos de hosting: *Together* (mesmo processo), *Individually* (processo separado por plugin), e *Individually (strict)*. Cada modo usa IPC diferente e expõe falhas distintas de lifetime/segfaults.
  - **Ações Técnicas:**
    1. Testar os 3 modos em *Settings → Plug-ins → Plug-in hosting* com o NAM-rs instanciado.
    2. Em cada modo: iniciar playback de 60 segundos com sinal contínuo, ativar/desativar o plugin (bypass Bitwig) 20x rapidamente.
    3. Monitorar o terminal do Bitwig (`bitwig-studio` via CLI) por `SIGSEGV`, `SIGABRT` ou mensagens `[Plugin crashed]`.
    4. Testar *Bounce in Place* de 10 segundos no modo *Individually* — este modo usa timing diferente de `process()`.
  - **Aceite:** Zero crashes em todos os modos. Zero mensagens de erro no console do Bitwig. Bounce in Place produz arquivo de áudio idêntico ao bypass direto.

---

## Sprint 2 — Motor DSP Vivo: Estado, Pipeline e Parâmetros CLAP

**Objetivo:** Transformar o esqueleto CLAP em um plugin funcionalmente completo para inferência neural. Isso exige: (a) um modelo de estado explícito em `NamClapShared`/`NamClapMainThread`/`NamClapProcessor`, (b) integração real do `DspPipelineContext` no `process()`, (c) GC de modelos via SPSC adaptado, (d) mapeamento de 4 parâmetros CLAP com smoothing sample-accurate, e (e) persistência de sessão.

**Pré-requisito:** Sprint 1 concluída (0 FAILs no `clap-validator`, plugin visível no Bitwig).

**Decisão arquitetural crítica — por que não usar `DspBridge` no CLAP:**

> O `DspBridge` (double-buffer lock-free) existe no modo Standalone para **sincronizar duas streams PipeWire independentes** (capture e playback). No modo CLAP o host chama um único `process()` — input e output já são buffers do mesmo ciclo. Portanto, o CLAP **não usa `DspBridge`**. O `DspPipelineContext` é construído no `activate()` com buffers próprios pré-alocados e executado diretamente no `process()`.

---

### Épico 2.0 — Scaffolding de Estado: `NamClapShared`, `MainThread` e `Processor`

> **Audit (2026-05-15):** Épico 2.0 revisado. Identificado e corrigido risco de pânico em re-ativações múltiplas (canais SPSC agora são devolvidos ao `Shared` no `deactivate()`). Invariante de zero-alocação validado formalmente.

- [x] **Tarefa 2.0.1** — Definir `NamClapShared`: Canal de Parâmetros Main→RT

  - **Contexto:** `NamClapShared` é o único objeto acessível tanto pela `main thread` quanto pela `audio thread` do clack. É o ponto ideal para o canal SPSC de parâmetros (análogo ao `ParamPayload` do Standalone).

  - **Ações Técnicas:**

    1. Em `src/clap/plugin.rs`, adicionar campos a `NamClapShared`:

       ```rust
       pub struct NamClapShared {
           // Canal SPSC lock-free: main thread envia novos NamPluginParams para a audio thread.
           // Capacidade 8 é suficiente (o host não bombeia params mais rápido que o process()).
           pub param_tx: rtrb::Producer<NamPluginParams>,
           pub param_rx: rtrb::Consumer<NamPluginParams>,
           // Canal GC: audio thread devolve modelos obsoletos para drop fora do RT.
           pub gc_tx: rtrb::Producer<Box<DynamicModel>>,
           pub gc_rx: rtrb::Consumer<Box<DynamicModel>>,
       }
       ```

    2. Inicializar os canais em `new_shared()` usando `rtrb::RingBuffer::new(8)`.

    3. Derivar `#[repr(align(128))]` para evitar false sharing entre threads.

  - **Aceite:** `cargo check --features clap-plugin` compila sem erros. A struct deixa de ser unit.

- [x] **Tarefa 2.0.2** — Definir `NamClapMainThread`: Estado da UI/Main

  - **Contexto:** `NamClapMainThread` é o estado exclusivo da main thread — seguro para alocações, I/O de arquivo e logging.

  - **Ações Técnicas:**

    1. Adicionar campos:

       ```rust
       pub struct NamClapMainThread<'a> {
           shared: &'a NamClapShared,
           // Parâmetros atuais conhecidos pela main thread (espelho dos params da audio thread).
           pub params: NamPluginParams,
           // Handle do host para notificações (request_restart, latency_changed, etc.).
           pub host: HostMainThreadHandle<'a>,
       }
       ```

    2. Implementar método `load_model(&mut self, path: &Path) -> Result<(), NamDiagnostic>` que:

       - Carrega e desserializa o modelo via `NamLoader` (cold-path — main thread, pode alocar).
       - Envia o `Box<DynamicModel>` para a audio thread via `shared.param_tx` (envolto em um novo enum `ClapParamPayload`).
       - Não bloqueia: se o canal SPSC estiver cheio, retorna `Err` com `NamDiagnostic`.

  - **Aceite:** Método `load_model` compila. Nenhuma alocação ocorre na audio thread durante a troca.

- [x] **Tarefa 2.0.3** — Definir `NamClapProcessor`: Estado Exclusivo da Audio Thread

  - **Contexto:** `NamClapProcessor` detém os buffers pré-alocados e o estado mutable da inferência. É criado no `activate()` e destruído no `deactivate()`.

  - **Ações Técnicas:**

    1. Adicionar campos:

       ```rust
       pub struct NamClapProcessor {
           // Modelo ativo (None = true-bypass, sinal passa sem modificação).
           model_l: Option<Box<DynamicModel>>,
           model_r: Option<Box<DynamicModel>>,
           // Resampler sinc polifásico (bypass quando sample_rate == 48000).
           resampler: NamResampler,
           // Parâmetros atuais na audio thread (snapshottados do SPSC a cada process()).
           params: NamPluginParams,
           // Buffers intermediários pré-alocados no activate() — ZERO alloc no process().
           // Dimensionados para max_frames_count * 2 (margem para o resampler).
           buf_mid_l: Vec<f32>,  // pós-resampler input
           buf_mid_r: Vec<f32>,
           buf_out_l: Vec<f32>,  // pós-modelo, pré-resampler output
           buf_out_r: Vec<f32>,
           // Gate FSM com histerese temporal (mesmo do Standalone).
           gate: DynamicHysteresis,
           silence_hyst: DynamicHysteresis,
           mono_hyst: DynamicHysteresis,
           process_mono: bool,
       }
       ```

    2. Em `activate()`, instanciar `NamClapProcessor` com os campos acima usando `audio_config.max_frames_count` para dimensionar os `Vec<f32>` (alocação permitida aqui — cold-path).

    3. **Invariante crítico:** Após `activate()`, `process()` jamais aloca heap. Validar com `CountingAllocator` (Tarefa 1.0.3).

  - **Aceite:** `activate()` retorna `Ok(NamClapProcessor { ... })`. Todos os buffers alocados. Zero allocações em `process()`.

---

### Épico 2.1 — Integração Real do `DspPipeline` no `process()`

- [x] **Tarefa 2.1.1** — Extração de Buffers `ChannelPair` e Dispatch para Inferência

  - **Contexto:** O `process()` atual faz bypass simples. Aqui substituímos por inferência real, reutilizando as funções `apply_input_stage`, `run_inference` e `apply_output_stage` de `src/dsp/pipeline.rs`.
  - **Problema de feature flag:** Estas funções estão gateadas em `#[cfg(any(feature = "standalone", test))]`. Precisam ser refatoradas para `#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]`.
  - **Ações Técnicas:**
    1. Em `src/dsp/pipeline.rs`, alterar os atributos `#[cfg(...)]` das funções `apply_input_stage`, `run_inference`, `apply_output_stage` e das structs `DspPipelineContext`, `MAX_RESAMP_BUF` para incluir `feature = "clap-plugin"`. **Não remover** o gate do `DspBridge`, `AppState` e funções exclusivas do PipeWire.
    2. Em `src/clap/processor.rs`, dentro do `process()`:
       - Iterar `audio.iter_mut()` para obter os port pairs.
       - Extrair os slices L e R do `ChannelPair::InputOutput` ou `ChannelPair::InPlace`.
       - Drenar o SPSC `shared.param_rx` para atualizar `self.params` (zero-alloc — apenas leitura do ring buffer).
       - Construir um `DspPipelineContext` on-stack (zero-alloc) apontando para `self.buf_*`.
       - Chamar `apply_input_stage` → `run_inference` → `apply_output_stage`.
       - Copiar `self.buf_out_l/r` de volta para os buffers de saída do host.
    3. Verificar que o `DspPipelineContext` não requer `bridge_ptr` (campo exclusivo do Standalone): criar uma versão sem o campo `bridge_ptr` ou usar `std::ptr::null_mut()` com `#[cfg]` condicional.
  - **Aceite:** Inserir o plugin no Bitwig com um modelo `.nam` carregado → áudio processado com timbre do amplificador neural. Medição de latência no Bitwig ≤ 3ms para buffer de 256 samples @ 48kHz.
  - **NOTA do Product Owner:** Como ainda não temos GUI, será necessário implementar (temporariamente, remover isto quando da implementação da GUI) alguma forma de carregar automaticamente algum modelo padrão que esteja na mesma pasta que o binários do plugin.
  - **NOTA do Product Owner:** Explique-me detalhadamente como obter estas informações de latência do Bitwig.

- [x] **Tarefa 2.1.2** — GC de Modelos: Descarte Seguro Fora da Audio Thread

  - **Contexto:** Quando a main thread envia um novo `Box<DynamicModel>` via SPSC, a audio thread substitui o modelo ativo e precisa descartar o antigo **sem chamar `drop()` na RT** (que pode alocar/liberar memória).
  - **Ações Técnicas:**
    1. Na audio thread (`process()`), ao substituir `self.model_l`, enviar o antigo via `shared.gc_tx.push(old_model)`. Se o canal estiver cheio (improvável com capacidade 8), usar `Box::into_raw` + `AtomicPtr` como fallback (padrão já implementado em `GcOverflowBuffer`).
    2. Na main thread (loop de eventos do host ou `on_main_thread()`), drenar `shared.gc_rx` e fazer `drop()` dos modelos obsoletos.
    3. Documentar com `// SAFETY: drop() ocorre fora da audio thread — RT-safe.`
  - **Aceite:** Trocar de modelo no Bitwig 10x consecutivamente sem XRUNs. `valgrind --tool=massif` mostra crescimento zero de heap durante as trocas.
  - **NOTA do Product Owner:** Como ainda não temos GUI, será necessário implementar (temporariamente, remover isto quando da implementação da GUI) alguma maneira de realizar esta troca para finalidade destes testes de aceite.

- [x] **Tarefa 2.1.3** — Logging RT-Safe via `clap_host_log`

  - **Contexto:** Na audio thread não podemos chamar `println!` ou `log::info!` (I/O). A extensão `clap_host_log` permite que o host exiba mensagens de log no seu console sem envolver I/O direto da audio thread.
  - **Ações Técnicas:**
    1. Em `NamClapShared`, adicionar um `AtomicU64` de flags de status análogo ao `RtStatusFlags` do Standalone (bits: `HAS_CLIPPED`, `IS_SILENT`, `MODEL_LOAD_FAILED`).
    2. Na audio thread, setar flags atômicas ao invés de logar diretamente.
    3. Na main thread (`on_main_thread()` ou timer), ler as flags e chamar `HostHandle::log(severity, message)` via `clack-extensions` `HostLog`.
    4. Em cold-paths fora da RT (como `load_model`), usar `log::error!` diretamente.
  - **Aceite:** Mensagem `"NAM-rs: model loaded"` aparece no *Script Console* do Bitwig após carregar um modelo. Nenhum `println!` ou `log::` na audio thread.
  - **NOTA do Product Owner:** Explique-me detalhadamente como obter estas no Bitwig.

---

### Épico 2.2 — Parâmetros CLAP e Sessão Persistente

- [x] **Tarefa 2.2.1** — Mapeamento de 4 Parâmetros via `clap_plugin_params`

  - **Contexto:** `NamPluginParams` em `src/common/params.rs` já define os 4 controles. Precisamos mapeá-los para IDs CLAP numéricos e implementar a trait `PluginParams` de `clack-extensions`.

  - **Ações Técnicas:**

    1. Criar `src/clap/extensions/params.rs` com as constantes de ID:

       ```rust
       pub const PARAM_INPUT_GAIN:  u32 = 0;
       pub const PARAM_OUTPUT_GAIN: u32 = 1;
       pub const PARAM_GATE_THRESH: u32 = 2;
       pub const PARAM_BYPASS:      u32 = 3;
       ```

    2. Implementar `PluginParams::count()` → `4`.

    3. Implementar `PluginParams::get_info(index)` → `ParamInfo` com:

       - `input_gain`: range `[-24.0, +24.0]` dB, default `0.0`, flags `AUTOMATABLE`.
       - `output_gain`: range `[-24.0, +24.0]` dB, default `0.0`, flags `AUTOMATABLE`.
       - `gate_threshold`: range `[-90.0, -40.0]` dB, default `-70.0`, flags `AUTOMATABLE`.
       - `bypass`: range `[0.0, 1.0]`, default `0.0`, flags `AUTOMATABLE | IS_STEPPED`.

    4. Implementar `PluginParams::get_value(id)` → lê de `self.params` (main thread lê snapshot atômico).

    5. Implementar `PluginParams::value_to_text` e `text_to_value` para exibição em dB (ex: `"-3.5 dB"`).

    6. Registrar a extensão em `impl Plugin for NamClapPlugin`.

  - **Aceite:** Os 4 knobs aparecem no Device Panel do Bitwig com labels corretos, ranges e valores padrão. Automação funciona (drag de knob cria lane de automação).

  - **NOTA do Product Owner:** Entendo que os parâmetros `gate_threshold` e `bypass` são apenas internos do NAM. O que é ajustável pelo usuário é apenas `input_gain`, `output_gain` e `nam_file` (este último onde o usuário indica o arquivo de modelo que desjea usar). O `nam_file` pode carregar por default o primeiro arquivo .nam encontrado na pasta do binário do plugin. Ao menos temporariamente enquanto a GUI não chega (inserir lembrete na Tarefa Técnica futura adequada).

- [x] **Tarefa 2.2.2** — Smoothing Sample-Accurate de Ganhos (Anti-Zipper)

  - **Contexto:** Bitwig pode enviar `CLAP_EVENT_PARAM_VALUE` a cada sample para modulação por LFO. Aplicar o ganho abruptamente causa zipper noise audível (~6dB/sample de descontinuidade). Precisamos de um 1-pole IIR smoother RT-safe.

  - **Ações Técnicas:**

    1. Criar `src/clap/param_smoother.rs` com struct:

       ```rust
       /// 1-pole IIR smoother: y[n] = α·target + (1-α)·y[n-1]
       /// α = 1 - exp(-2π * fc / fs), fc = 20Hz (tempo de resposta ~8ms @ 48kHz).
       pub struct ParamSmoother {
           current: f32,
           target: f32,
           coeff: f32,  // pré-calculado no activate()
       }
       ```

    2. `ParamSmoother::tick() -> f32`: avança um sample, retorna valor suavizado.

    3. Em `NamClapProcessor`, adicionar `smoother_in: ParamSmoother, smoother_out: ParamSmoother`.

    4. No início do `process()`, antes do loop de frames:

       - Iterar `events.input` e processar eventos `CLAP_EVENT_PARAM_VALUE` e `CLAP_EVENT_PARAM_MOD`.
       - Atualizar `smoother_in.target` e `smoother_out.target`.

    5. Dentro do loop de inferência (se `n_frames` for pequeno), aplicar `smoother.tick()` por sample para modulação sample-accurate. Para blocos grandes (> 64 samples), processar em sub-blocos de 8 samples com interpolação linear (trade-off precisão × CPU).

  - **Aceite:** LFO do Bitwig em 10Hz modulando Input Gain → espectro FFT da saída mostra sidebands corretos sem artefatos de descontinuidade. `CountingAllocator` = 0 durante processamento.

- [x] **Tarefa 2.2.3** — Persistência de Sessão via `clap_plugin_state`

  - **Contexto:** Ao salvar um projeto no Bitwig, o host chama `state.save(stream)`. Ao reabrir, chama `state.load(stream)`. Sem isso, o Bitwig loga o erro `Error saving initial state of plug-in for undo history: Plug-in does not support saving its state` no carregamento. O plugin deve restaurar o modelo e os parâmetros exatamente como estavam.

  - **Ações Técnicas:**

    1. Criar `src/clap/extensions/state.rs` implementando `PluginState`.

    2. Formato de serialização: JSON minificado via `serde_json` (já é dependência):

       ```json
       {"v":1,"input_gain":0.0,"output_gain":-3.0,"gate":-70.0,"bypass":false,"model":"/home/user/amps/jcm800.nam"}
       ```

    3. Em `save(stream)`: serializar `NamPluginParams` atual (snapshot da main thread) e escrever via `stream.write()` do clack.

    4. Em `load(stream)`: ler bytes via `stream.read()`, deserializar, atualizar `self.params`, e chamar `self.load_model(path)` se `model_path` for `Some`.

    5. Em `load()`: se o arquivo `.nam` não existir no path salvo, logar via `clap_host_log` com severity `WARNING` (não falhar — o projeto deve abrir mesmo sem o modelo).

    6. Notificar o host via `host.params_rescan(CLAP_PARAM_RESCAN_VALUES)` após load.

  - **Aceite:** Salvar projeto no Bitwig → fechar → reabrir → plugin carrega com mesmos ganhos e modelo. Log no Bitwig confirma carregamento. Arquivo inexistente: warning no log, plugin abre em true-bypass limpo.

> **🔎 AUDITORIA PÓS-SPRINT 2 (2026-05-15)**
> Revisão profunda das Sprints 0-2 para validação da fundação CLAP:
>
> - ✅ Todas as 18 tarefas técnicas concluídas e verificadas contra o código implementado.
> - ✅ `lints.sh` 100% limpo, `cargo check` zero warnings para ambas features.
> - ✅ Zero-alloc no hot-path confirmado via `CountingAllocator` + `clack-host`.
> - ✅ Bitwig 6.0.6: scan, bypass, parâmetros e persistência funcionais.
> - **Achado S2-A1 (corrigido):** `powf()` no `process()` substituído por `GainLut::db_to_linear()` (~20x mais rápido).
> - **Achado S2-A4 (corrigido):** Bypass explícito adicionado ao `process()` para conformidade IS_BYPASS.
> - **Achado S1-A1 (corrigido):** `#[allow(dead_code)]` removido; campo `gate` genuinamente morto removido.
> - **Achado S2-A3 (corrigido):** `docs/architecture.md` atualizado com §8.1 (Arquitetura CLAP completa).
> - **Achado S2-A2:** Campos PW-specific em `RtStatusFlags` — confirmado acesso apenas em `pw_host.rs`. Endereçado na T3.4.1.
> - **Achado S2-A5:** `Box::leak` em `state.rs` — endereçado como sub-item na T6.1.1.

---

## Sprint 3 — Latência, Compliance e Estresse de Host

**Objetivo:** Garantir que o plugin reporta latência exata ao host (ADC precisa), processe blocos de qualquer tamanho sem crash ou alocação, e passe em stress tests automatizados sem necessidade de DAW.

**Pré-requisito:** Sprint 2 concluída (pipeline DSP vivo, parâmetros CLAP funcionais).

---

### Épico 3.1 — Latência Exata e ADC (`clap_host_latency`)

- [x] **Tarefa 3.1.1** — Cálculo de Latência Real do `NamResampler`

  - **Contexto:** O `NamResampler` usa um filtro FIR Sinc Polifásico de Fase Mínima com `TAPS_PER_PHASE` coeficientes por fase e `NUM_PHASES = 256` fases. A latência algorítmica **não é zero** quando o resampling está ativo — ela é determinística e computável a partir de `TAPS_PER_PHASE`.

  - **Fórmula de latência (quando resampling ativo):**

    ```rust
    // src/dsp/sinc_kernel.rs define TAPS_PER_PHASE (ex: 32).
    // A latência do filtro FIR de fase mínima é aproximadamente metade da ordem do filtro.
    // Para o filtro de entrada (pw_rate → nam_rate):
    latency_input_samples_at_pw  = (TAPS_PER_PHASE / 2) * (pw_rate / nam_rate)
    // Para o filtro de saída (nam_rate → pw_rate):
    latency_output_samples_at_pw = (TAPS_PER_PHASE / 2)
    // Latência total reportada ao host em samples @ pw_rate:
    total_latency = latency_input_samples_at_pw + latency_output_samples_at_pw
    // Em bypass (pw_rate == nam_rate): total_latency = 0
    ```

  - **Ações Técnicas:**

    1. Adicionar método `pub fn latency_samples(&self, host_rate: u32) -> u32` em `NamResampler`:
       - Retorna `0` quando `is_bypass()`.
       - Retorna a fórmula acima usando `TAPS_PER_PHASE` e `pw_rate` / `nam_rate`.
    2. Criar teste unitário em `resampler_test.rs` que valida:
       - `latency_samples(48000)` == 0 quando `pw_rate == 48000 && nam_rate == 48000`.
       - `latency_samples(44100)` retorna valor > 0 e consistente com a fórmula para `44100 → 48000`.

  - **Aceite:** Método compilado e testado. Valor correto para as taxas mais comuns (44100, 48000, 96000).

- [x] **Tarefa 3.1.2** — Implementação da Extensão `clap_host_latency`

  - **Contexto:** O host precisa saber a latência introduzida pelo plugin para sincronizar trilhas em paralelo (ADC — Automatic Delay Compensation). Sem isso, uma trilha com NAM-rs ficará adiantada em relação às outras quando o resampling estiver ativo.
  - **Ações Técnicas:**
    1. Criar `src/clap/extensions/latency.rs` implementando `PluginLatency` de `clack-extensions`:
       - `latency() -> u32`: retorna `self.resampler.latency_samples(host_rate)`.
       - O valor é lido da `main thread` — `NamClapProcessor` expõe um `AtomicU32` de latência atual (setado no `activate()`), lido pela main thread via `NamClapShared`.
    2. Registrar a extensão em `impl Plugin for NamClapPlugin`.
    3. Quando o modelo é trocado e a latência muda (ex: modelo 44.1kHz → modelo 96kHz), notificar o host via `host.latency_changed()` na main thread.
    4. Reexecutar `clap-validator` após adicionar a extensão — confirmar 0 FAILs.
  - **Aceite:** No Bitwig, inserir NAM-rs e uma trilha de referência em paralelo → *Align Delay* do Bitwig mostra exatamente `N samples` de offset, onde `N == latency_samples()`. Áudios alinhados soam em fase.

---

### Épico 3.2 — Estabilidade com Blocos Irregulares

- [x] **Tarefa 3.2.1** — Mapeamento de Vulnerabilidades com Block Sizes Não-Potência de 2

  - **Contexto:** O Bitwig subdivide blocos para automação sample-accurate e para rendering offline. É comum receber sequências como `[17, 495, 17, 495, ...]` (automação em grid odd) ou `[1, 1, 1, ...]` (modulação por sample). O `NamResampler` e o gate SIMD assumem certos tamanhos mínimos.
  - **Nota (Sprint 0 — completado):** O Épico 0.1 já corrigiu os fallbacks escalares para `n < 8` nos kernels SIMD e validou o Gate FSM com `n_samples = 1`.
  - **Nota (Auditoria S0-S2):** Os itens 3 e 4 das ações abaixo já foram cobertos pela Sprint 0 (`gate_test.rs` e fallbacks `#[cold]`). O foco residual desta tarefa é exclusivamente nos **edge cases do pipeline CLAP completo com modelo carregado e resampler ativo** (caminhos não cobertos pela Sprint 0, que testava componentes isolados).
  - **Ações Técnicas Residuais:**
    1. Auditar `apply_input_stage` e `run_inference` para localizar qualquer `assert!(n >= 8)` ou alinhamento SIMD implícito que explode com `n < 8` — **com modelo carregado no NamClapProcessor**.
    2. Verificar que `NamResampler::process_input/output` retorna `0` (sem pânico) para `n_in = 0`.
    3. ~~Verificar que o gate (`DynamicHysteresis::update`) lida corretamente com `n_samples = 1`.~~ *(Coberto pela Sprint 0)*
    4. ~~Verificar que `compute_energy_stereo` e `compute_max_diff_avx2` funcionam com `n < 8`.~~ *(Coberto pela Sprint 0)*
    5. Criar teste de integração headless CLAP com blocos `[1, 3, 7, 17, 33]` samples **com modelo `.nam` de teste carregado** via SPSC, validando que `process()` retorna `ProcessStatus::Continue` sem panic.
  - **Aceite:** `cargo test --features clap-plugin` passa. Nenhum SIGSEGV ao processar blocos de 1, 3, 7, 17 e 33 samples com modelo ativo.

- [x] **Tarefa 3.2.2** — Suite de Stress Headless com `clack-host` (CI-Ready)

  - **Contexto:** Complementa a Tarefa 1.1.3. Desta vez focado em stress com blocos irregulares, trocas de modelo em mid-stream e modulação intensa — tudo sem DAW.
  - **Ações Técnicas:**
    1. Expandir `tests/clap_lifecycle_test.rs` com cenários adicionais:
       - **Bloco variável:** Enviar sequência `[1, 7, 17, 33, 53, 128, 256, 512, 1, 1]` de frames com áudio de ruído branco. Sem crash, sem silêncio inesperado.
       - **Troca de modelo mid-stream:** Durante 1000 ciclos de `process()`, trocar o modelo a cada 50 ciclos. Confirmar GC correto via `CountingAllocator` (zero alloc no process()).
       - **Modulação intensa:** Enviar evento `CLAP_EVENT_PARAM_VALUE` para `input_gain` em cada sample de um bloco de 512 frames. Confirmar zero zipper noise (testar via análise espectral simples dos samples de saída: RMS do diff entre saída suavizada e esperada < -60 dB).
    2. Adicionar ao `Makefile` ou `utils/tests-cargo.sh` target `test-clap-stress`.
  - **Aceite:** Suite executa em < 30s em CI. Zero FAILs, zero panics, zero allocs no `process()`.

- [x] **Tarefa 3.2.3** — Guard de Zero-Alocação no `process()` com Blocos Irregulares

  - **Contexto:** A Tarefa 1.0.3 validou zero-alloc para bloco padrão de 512. Esta tarefa valida os edge cases: bloco de 1 sample (pior caso) e bloco de `max_frames_count` (caso máximo).
  - **Ações Técnicas:**
    1. Adicionar ao teste de `CountingAllocator` dois cenários extras:
       - `n_frames = 1`: 1000 invocações consecutivas. `alloc_count` deve ser 0 ao final.
       - `n_frames = max_frames_count`: 100 invocações. `alloc_count` deve ser 0.
    2. Se alguma alocação for detectada, usar `backtrace` para localizar a origem e corrigir.
  - **Aceite:** `CountingAllocator::alloc_count()` == 0 para todos os cenários. Resultado documentado no comentário do teste.

---

### Épico 3.3 — Validação em Host Secundário (REAPER) [CANCELADO]

---

### Épico 3.4 — Consolidação `RtStatusFlags` para Multi-Host

> **Origem:** Auditoria Revisor-Auditor (2026-05-14, item B3). Preparar a infraestrutura de flags atômicas para operação dual-backend (Standalone + CLAP).

- [x] **Tarefa 3.4.1** — Separar Campos PipeWire-Specific de `RtStatusFlags`

  - **Contexto:** `RtStatusFlags` em `src/common/spsc.rs` contém campos como `requested_pw_rate`, `active_rate_changed` e `resampler_consumer` que são específicos do backend PipeWire. No modo CLAP, o host controla o sample rate via `activate()` — esses campos são irrelevantes e poluem o namespace.
  - **Ações Técnicas:**
    1. Identificar campos em `RtStatusFlags` que são exclusivos do Standalone: `requested_pw_rate`, `active_rate_changed`, `requested_nam_rate`.
    2. Gatear estes campos com `#[cfg(feature = "standalone")]` para que não existam no build CLAP.
    3. Os campos universais permanecem: `status_bits`, `dsp_cycle_time`, `dsp_overloads`, `last_n_samples`, `latency_hist`, `rt_priority`, `active_rate`.
    4. Ajustar `RtStatusFlags::new()` e todos os consumidores em `pw_host.rs` e `cli.rs` para compilar com os gates condicionais.
    5. Verificar que `cargo check` compila sem erros para ambas as features.
  - **Nota (Auditoria S0-S2):** Confirmado via `grep` que os 3 campos PW-specific (`requested_pw_rate`, `active_rate_changed`, `requested_nam_rate`) são acessados exclusivamente em `src/standalone/pw_host.rs`. Nenhum código em `src/clap/` referencia esses campos. A refatoração é straightforward.
  - **Aceite:** Build `clap-plugin` e `standalone` ambos compilam limpos. `RtStatusFlags` no CLAP contém apenas campos universais.

> **🔎 AUDITORIA PÓS-SPRINT 3 (2026-05-16)**
> Revisão profunda das Sprints 0-3 (Painel Revisor-Auditor × Planejador-Arquiteto):
>
> - ✅ Todas as 4 Sprints aprovadas — nenhum achado bloqueante.
> - ✅ Latência `NamResampler` implementada e reportada via `clap_host_latency`.
> - ✅ Blocos irregulares `[1,7,17,33,53,128,256,512,1,1]` testados com modelo ativo.
> - ✅ Zero-alloc validado para `n_frames=1` (1000x) e `n_frames=max` (100x).
> - ✅ `RtStatusFlags` desacopladas — campos PW-specific com `#[cfg(feature = "standalone")]`.
> - **Achado S2-D1 (registrado):** `Box::leak` em `state.rs` — debt técnico aceito, remediação na T7.1.1.
> - **Achado S2-D2 (aceito):** Auto-load indeterminístico — temporário, removido na Sprint 4.
> - **Achado S2-D3 (aceito):** `powi(2)` otimizado pelo compilador como `x * x` — overhead zero.
> - **Achado S3-D1 (registrado):** Falta modelo LSTM no `test_model_switching_stress` — T7.1.1.
> - **Achado S1-F1 (aceito):** `clap_lifecycle_test` será integrado ao `build-clap.sh`.
> - Decisões técnicas documentadas nos locais do código via comentários.
> - Fundação sólida e pronta para a construção da GUI (Sprint 4).

---

## Sprint 4 — Interface Gráfica Responsiva (egui + baseview)

**Objetivo:** Painel customizado nativo embutido no host via X11/XWayland, isolado da audio thread, renderizando a 60fps com zero-interferência no DSP. Controles bidirecionais de ganho, file picker de modelo, medidores de nível e feedback visual de automação. Validação no Bitwig Studio e Fender Studio Pro.

**Pré-requisito:** Sprint 3 concluída (latência reportada, compliance validado, stress tests verdes).

**Decisões arquiteturais:**

> - **Estratégia Wayland (Pesquisa Lab 2026-05-19):** Pesquisa profunda concluída em `docs/lab-gui-wayland.md`. A estratégia aprovada é uma **arquitetura em camadas**:
>   - **v1.0 (Sprint 4):** Backend X11 via `baseview` + XWayland. Funcional em todas as DAWs. Estável e comprovado.
>   - **v1.1 (Futuro — Exp 2 do Lab):** Backend Wayland floating via `xdg-foreign`. Experimento isolado.
>   - **v2.0 (Futuro — Exp 3 do Lab):** Backend Wayland embedded via nested compositor (quando CLAP padronizar `CLAP_EXT_GUI_WAYLAND_HOST` e Bitwig implementar).
> - **`baseview`**: biblioteca Rust para embeder janelas nativas em hosts (X11 no Linux). Fornece `RawWindowHandle` compatível com CLAP sem depender de GTK/Qt. Issue #135 (Wayland) permanece aberta — XWayland é o caminho estável.
> - **`egui`**: Immediate Mode GUI — sem estado de widget persistente, sem GC, sem alocações no render loop. Ideal para plugins de áudio.
> - **`egui_glow`**: Integração manual do egui via OpenGL 3.3 (`egui_glow`). O crate `egui-baseview` está abandonado (~2021); integração direta é necessária.
> - **`rfd`** (Rusty File Dialog): file picker nativo assíncrono que delega ao `zenity`/`xdg-open` no Linux — nunca bloqueia a UI thread.
> - **Canal SPSC UI↔RT**: a UI nunca acessa diretamente os campos de `NamClapProcessor`. Tudo via `AtomicF32` / `AtomicU64` em `NamClapShared` para leitura de telemetria, e via SPSC de parâmetros para envio de comandos.
> - **Renderizador:** `egui_glow` (OpenGL 3.3) para máxima compatibilidade com hosts sandboxed. Revisável para `egui_wgpu` se necessário no futuro.
> - **Tamanho fixo:** `can_resize() = false` na v1 — simplifica o layout. Tamanho: 600×280px.

---

### Épico 4.0 — Pesquisa Preparatória e Prototipagem

> **Natureza:** Pesquisa técnica, validação de viabilidade e definição de identidade visual antes de mergulhar na implementação.
> **Referência:** `docs/lab-gui-wayland.md` — Laboratório Exploratório (documento vivo).

- [x] **Tarefa 4.0.1** — Pesquisa de Compatibilidade `egui` + `baseview` no Ecossistema CLAP

  - **Contexto:** Validar a viabilidade técnica real do stack no ambiente-alvo (Ubuntu 25.10, Bitwig 6.0.6, Wayland/X11).

  - **Achados (Lab 2026-05-19):**

    1. **Wayland vs X11:** O spec CLAP (`gui.h` L64-66) declara `CLAP_WINDOW_API_WAYLAND` como API oficial, mas com nota *"embed is currently not supported, use floating windows"*. O Wayland não suporta embedding cross-client por design de segurança. O `baseview` (Issue #135, Nov/2022) **não tem e não terá suporte Wayland** no horizonte previsível.
    2. **`egui-baseview`:** O crate no crates.io está **abandonado** (~2021). Incompatível com `egui 0.31+`. Integração manual `baseview` + `egui_glow` é o caminho correto.
    3. **4 hipóteses avaliadas em profundidade:**
       - **H1 (XWayland Fallback):** ✅ Aprovada para v1. Esforço zero, estabilidade comprovada. Todo o ecossistema usa isto.
       - **H2 (xdg-foreign floating):** Viável como experimento futuro. Requer backend Wayland no baseview (~1500-2500 LOC), **NÃO** reescrever o egui. Resultado: janela flutuante com decorações de sistema.
       - **H3 (IWaylandHost nested compositor):** Solução definitiva do futuro. Esforço no plugin ~500-800 LOC, **MAS** depende de Bitwig/CLAP implementarem a extensão. Studio One 7+ já faz isto para VST3 (SDK 3.8.0, MIT).
       - **H4 (Substituir baseview por winit):** Rejeitada. Winit não suporta paradigma de janela filha parasitária.
    4. **Decisão:** X11 via XWayland como estratégia v1. `is_api_supported()` retorna `true` APENAS para `CLAP_WINDOW_API_X11`. Toda lógica de UI (egui) é agnóstica ao backend de janela — trocar backend no futuro não requer reescrever widgets.
    5. **Documentação completa:** `docs/lab-gui-wayland.md` com análise detalhada, referências e log de experimentos.

  - **Aceite:** ~~Relatório técnico com decisão fundamentada.~~ ✅ `docs/lab-gui-wayland.md` publicado e aprovado pelo PO.

- [x] **Tarefa 4.0.2** — Prototipagem de Layout e UX do Plugin

  - **Contexto:** Definir o layout visual, hierarquia de controles e estética premium antes de codificar.

  - **Achados (Lab 2026-05-19):**

    1. Mockup 1 (`mockup_ui.jpg`) aprovado pelo PO como base visual.
    2. Layout 600×280px com 5 zonas:
       - **Identidade** (esquerda): Logo NAM-rs⚡, versão, badge SIMD, model picker.
       - **Controles** (centro): 3 knobs (Input Gain, Output Gain, Noise Gate) com labels dB.
       - **Medidores VU** (direita): L/R com peak hold 2s, gradiente tricolor.
       - **Bypass** (extrema direita): Toggle LED verde/cinza.
       - **Status Bar** (rodapé): Sample rate, latência, tipo de modelo, ISA, botão [📊] telemetria.
    3. **Paleta de cores (Dark Mode):**
       - Background: `#1A1D23` | Painéis: `#232830` | Borda: `#2E3440`
       - Accent knobs: `#00D4AA` (turquesa) | Gate accent: `#F5A623` (amber)
       - VU: `#43E97B` → `#F5CE62` → `#F74E4E` | Texto: `#E5E9F0` / `#8B95A5`
    4. **Tipografia:** Inter (Google Fonts).
    5. Painel de telemetria expansível (P50/P95/P99) acessível via botão [📊], colapsado por default.

  - **Aceite:** ~~2-3 mockups aprovados pelo PO.~~ ✅ Mockup 1 aprovado com ajustes documentados no Lab.

---

### Épico 4.1 — Dependências e Feature Flag de GUI

- [x] **Tarefa 4.1.0** — Avaliação de Impacto nos Feature Flags e Build System

  - **Contexto:** A GUI introduz feature flag `clap-plugin-gui`. Garantir 3 perfis limpos: `standalone`, `clap-plugin` (headless), `clap-plugin-gui` (com janela).

  - **Ações Técnicas:**

    1. Revisar `Cargo.toml` e mapear dependências opcionais necessárias: `egui`, `egui_glow`, `baseview`, `rfd`, `glow`.
    2. Verificar que `egui`, `baseview` e `rfd` podem ser `optional = true` sem conflito.
    3. Planejar estrutura: `src/clap/gui/` com `mod.rs`, `window.rs`, `ui.rs`.
    4. Verificar impacto no `lints.sh` e `build-clap.sh` — precisam de variantes?
    5. Integrar `clap_lifecycle_test` como passo final do `build-clap.sh` (achado S1-F1).

  - **Aceite:** Plano documentado. Zero ambiguidade sobre como compilar cada perfil.

- [x] **Tarefa 4.1.1** — Adicionar Dependências sob Feature `clap-plugin-gui`

  - **Contexto:** A GUI é opcional — um build "headless" (sem GUI) deve ser possível para hosts que não suportam janelas embutidas ou para uso em servidores. Criar uma sub-feature `clap-plugin-gui` que ativa apenas quando o host solicitar.

  - **Ações Técnicas:**

    1. Em `Cargo.toml`, adicionar feature `clap-plugin-gui` que depende de `clap-plugin`:

       ```toml
       [features]
       clap-plugin-gui = ["clap-plugin", "dep:egui", "dep:egui_glow", "dep:baseview", "dep:rfd", "dep:glow"]

       [dependencies]
       egui      = { version = "0.31",  optional = true }
       egui_glow = { version = "0.31",  optional = true }
       glow      = { version = "0.16",  optional = true }
       baseview  = { version = "0.5",   optional = true, git = "https://github.com/RustAudio/baseview" }
       rfd       = { version = "0.15",  optional = true }
       ```

    2. Gatear todo código de UI em `#[cfg(feature = "clap-plugin-gui")]`.

    3. Verificar que `cargo check --no-default-features --features clap-plugin` (sem GUI) compila sem erros — o plugin deve funcionar headless.

    4. Verificar que `cargo check --no-default-features --features clap-plugin-gui` também compila.

    5. Executar `./utils/lints.sh` para ambas as features.

  - **Aceite:** Dois builds limpos: com e sem GUI. Zero warnings.

---

### Épico 4.2 — Ciclo de Vida da Janela CLAP (`clap_plugin_gui`)

- [x] **Tarefa 4.2.1** — Implementação de `PluginGui` para Linux (X11 via XWayland)

  - **Contexto:** No Ubuntu 25.10 com Wayland como compositor padrão, o Bitwig fornece handles X11 via XWayland quando o plugin declara suporte exclusivo a X11. Conforme decisão do Lab (H1 aprovada), o plugin declara suporte **APENAS** a `CLAP_WINDOW_API_X11`, forçando a DAW a usar o path XWayland estável.
  - **Ações Técnicas:**
    1. Criar `src/clap/extensions/gui.rs` implementando `PluginGui` de `clack-extensions`:
       - `is_api_supported(api, is_floating)`: retornar `true` **APENAS** para `CLAP_WINDOW_API_X11` (embedded, `is_floating = false`). Rejeitar `CLAP_WINDOW_API_WAYLAND`, `CLAP_WINDOW_API_WIN32` e `CLAP_WINDOW_API_COCOA`.
       - `get_preferred_api()`: retornar `CLAP_WINDOW_API_X11`, `is_floating = false`.
       - `get_size()`: retornar tamanho fixo `(600, 280)` pixels.
       - `set_size()`: aceitar apenas `(600, 280)` — rejeitar outros tamanhos.
       - `can_resize()`: retornar `false` (tamanho fixo na v1).
    2. Registrar a extensão em `impl Plugin for NamClapPlugin`.
    3. Reexecutar `clap-validator` — 0 FAILs.
  - **Aceite:** Ao clicar no ícone da janela do plugin no Bitwig (Wayland com XWayland), a janela abre embedded. `clap-validator` 0 FAILs.

- [x] **Tarefa 4.2.2** — Embed da Janela `baseview` e Inicialização do `egui`

  - **Contexto:** O `create_window()` do CLAP fornece um `RawWindowHandle` do host (o "parent" onde a janela do plugin deve ser embutida). `baseview::Window::open_parented()` aceita este handle e cria a janela filha.

  - **Ações Técnicas:**

    1. Em `src/clap/gui/window.rs`, criar struct `NamPluginWindow`:

       ```rust
       pub struct NamPluginWindow {
           // Handle da janela baseview (mantém a janela viva via RAII).
           _window: baseview::WindowHandle,
           // Contexto de renderização egui.
           ctx: egui::Context,
           // Referência à área compartilhada de telemetria (lida via atômicos — zero locks).
           shared: Arc<NamClapShared>,
       }
       ```

    2. Implementar `baseview::WindowHandler` para `NamPluginWindow`:

       - `on_frame()`: chamar `ctx.run(input, ui_fn)` e submeter o frame renderizado via `egui_glow::Painter` (decisão D1 — `egui_glow` sobre OpenGL 3.3, não `wgpu`).
       - `on_event()`: encaminhar eventos de mouse/teclado ao `egui` via `egui::RawInput`.

    3. Em `PluginGui::create_window()`: chamar `baseview::Window::open_parented(parent_handle, NamPluginWindow::new(...))`.

    4. Em `PluginGui::destroy()`: dropar o `WindowHandle` para fechar a janela.

  - **Aceite:** Janela abre com fundo preto (antes dos controles). Fecha sem hang. Zero panics ao abrir/fechar 10× consecutivamente.

- [x] **Tarefa 4.2.3** — Thread Safety: UI Thread vs Audio Thread vs Main Thread

  - **Contexto:** O CLAP especifica 3 threads: `audio thread` (process), `main thread` (plugin lifecycle) e `UI thread` (janela). `baseview` no Linux cria sua própria thread para o event loop da janela. A UI **nunca** pode chamar métodos da audio thread diretamente.

  - **Ações Técnicas:**

    1. Definir em `NamClapShared` campos atômicos de telemetria para a UI (leitura lock-free):

       ```rust
       // Nível True Peak L/R — setados pela audio thread, lidos pela UI thread.
       pub ui_peak_l: AtomicU32,  // bits de f32 via f32::to_bits()
       pub ui_peak_r: AtomicU32,
       // Flag: clipping ocorreu desde o último frame da UI.
       pub ui_clipped: AtomicBool,
       // Nome do modelo carregado (path basename). Escrito pela main thread.
       // Lido pela UI thread. Usar AtomicPtr<String> ou um Mutex<String> — UI thread pode bloquear.
       pub ui_model_name: Mutex<String>,
       ```

    2. Na audio thread (`process()`): atualizar `ui_peak_l/r` com `Ordering::Relaxed` (UI pode ver valores levemente atrasados — aceitável para medidores visuais).

    3. Na UI thread (`on_frame()`): ler `ui_peak_l/r` com `Ordering::Relaxed`. Nunca escrever nos campos de `NamClapProcessor`.

    4. Documentar claramente com comentários quais campos são escritos por qual thread.

  - **Aceite:** `cargo test` passa sem `ThreadSanitizer` errors (`RUSTFLAGS="-Zsanitizer=thread" cargo +nightly test`).

---

### Épico 4.3 — Controles, Automação e Telemetria

- [x] **Tarefa 4.3.1** — Layout e Controles Principais (Mockup: `mockup_ui.jpg`)

  - **Contexto:** A UI deve ser funcional e esteticamente coerente com o ecossistema de plugins profissionais. Layout baseado no Mockup aprovado (`mockup_ui.jpg`). Dark mode, compacto (600×280px), tipografia Inter.
  - **Referência visual:** `mockup_ui.jpg` aprovado pelo PO. Paleta de cores e layout detalhados na T4.0.2.
  - **Ações Técnicas:**
    1. Implementar a função `ui_fn(ui: &mut egui::Ui, shared: &NamClapShared, host: &HostHandle)` em `src/clap/gui/ui.rs` com layout de 5 zonas:
       - **Zona 1 — Identidade (esquerda):** Logo "NAM-rs⚡" em `egui::RichText` bold + versão + badge SIMD. Botão `[📂 Load Model]` + label com basename do `.nam` carregado.
       - **Zona 2 — Controles (centro):** 3 knobs customizados (INPUT, OUTPUT, GATE) com arcos de progresso. INPUT/OUTPUT: accent turquesa `#00D4AA`, range −24 a +24 dB. GATE: accent amber `#F5A623`, range −90 a −40 dB. Labels em dB abaixo de cada knob.
       - **Zona 3 — Medidores VU (direita):** Barras verticais L/R com gradiente: verde `#43E97B` < −12dBFS, amarelo `#F5CE62` < −3dBFS, vermelho `#F74E4E` ≥ −3dBFS. Peak hold de 2s com seta indicadora.
       - **Zona 4 — Bypass (extrema direita):** Toggle LED — verde brilhante `#00D4AA` quando ativo, cinza `#4A4F5A` quando bypass.
       - **Zona 5 — Status Bar (rodapé):** Sample rate | Latência em samples | ISA (AVX2/AVX-512) | Tipo de modelo (WaveNet/LSTM) | Botão [📊] telemetria.
    2. Paleta de cores: Background `#1A1D23`, painéis `#232830`, borda `#2E3440`, texto `#E5E9F0` / `#8B95A5`.
    3. Todo controle bidirecional: arrastar atualiza o valor via SPSC de parâmetros (Main→RT) E notifica o host via `host.begin_gesture(id)` / `host.end_gesture(id)` para automação.
  - **Aceite:** UI renderiza a 60fps com CPU < 2% em idle (sem áudio). Todos os controles respondem ao arrastar. Medidores atualizam com o áudio em playback. Visual coerente com o `mockup_ui.jpg`.
  - **NOTA do Product Owner:** Já pode remover o "// TODO: REMOVER quando a GUI estiver funcional (Tarefa 4.3.1)." no `src/clap/plugin.rs`.

- [x] **Tarefa 4.3.2** — Sincronia de Gestos de Automação (`CLAP_EVENT_PARAM_GESTURE_BEGIN/END`)

  - **Contexto:** Sem os eventos de gesto, o Bitwig não registra a trilha de automação ao arrastar um knob na UI do plugin. O host precisa saber exatamente quando o usuário começa e termina o gesto.
  - **Ações Técnicas:**
    1. Ao receber `egui::Response::drag_started()` em um knob: chamar `host.param_gesture_begin(param_id)` na main thread.
    2. Ao receber `egui::Response::drag_released()`: chamar `host.param_gesture_end(param_id)`.
    3. Durante o drag: enviar `host.param_value_changed(param_id, new_value)` a cada frame para que o Bitwig atualize o valor em tempo real (sem esperar o `drag_released`).
    4. Testar: criar uma automação no Bitwig arrastando o knob INPUT → a trilha de automação deve aparecer automaticamente.
  - **Aceite:** Arrastar qualquer knob no plugin → trilha de automação aparece no Bitwig. Valor da automação corresponde exatamente ao valor do knob.

- [X] **Tarefa 4.3.3** — File Picker Assíncrono e Loading Feedback

  - **Contexto:** `rfd::FileDialog::pick_file()` é bloqueante se chamado na UI thread. No Linux, deve ser chamado em thread separada via `std::thread::spawn` ou `rfd::AsyncFileDialog` com um executor async minimalista.
  - **Ações Técnicas:**
    1. Ao clicar `[📂 Load Model]`:
       - Setar um `AtomicBool` `ui_loading = true` em `NamClapShared`.
       - Lançar `std::thread::spawn(|| rfd::FileDialog::new()...pick_file())`.
       - A thread spawned envia o path via um `Mutex<Option<PathBuf>>` de resultado em `NamClapShared`.
    2. A cada frame da UI: verificar se o resultado chegou. Se sim, chamar `main_thread.load_model(path)` (que envia para a audio thread via SPSC).
    3. Enquanto `ui_loading == true`: mostrar spinner animado no label do modelo (ex: `"⏳ Loading..."` alternando a cada 500ms via `egui::Context::request_repaint_after`).
    4. Após carregamento: `ui_loading = false`, atualizar `ui_model_name`.
  - **Aceite:** Clicar `[📂 Load Model]` abre o file picker nativo do GNOME/KDE sem bloquear a UI. Após selecionar, o spinner aparece e desaparece quando o modelo está pronto. Audio thread não é interrompida durante o carregamento.

---

> **🔎 AUDITORIA PARCIAL PÓS-SPRINT 4.3 (2026-05-20)**
> Revisão dos épicos de engenharia da GUI (4.0 a 4.3):
>
> - ✅ Arquitetura X11/XWayland implementada com sucesso no Linux, isolando a UI da thread de áudio.
> - ✅ Build limpo com a feature `clap-plugin-gui` sem quebrar o baseline.
> - ✅ Controles bidirecionais comunicando via canais lock-free sem alocações RT.
> - ✅ Ausência de códigos não tratados (`unwrap`, `todo!`) atestada por validação estática e testes de compilação.
> - A infraestrutura está madura e pronta para a bateria de testes manuais intensivos de UX/UI nas DAWs alvo.

### Épico 4.4 — Validação Humana e Cross-Host (Bitwig & Fender Studio Pro)

> **Atenção (QA/Testador Humano):** As tarefas a seguir não envolvem codificação imediata (exceto ajustes de bugs estéticos/funcionais decorrentes do teste). Elas devem ser executadas manualmente nas DAWs instaladas para aferir a robustez, estabilidade e aderência visual do plugin no mundo real.

- [ ] **Tarefa 4.4.1** — Certificação Visual e Ajuste de Pixel-Perfect vs Mockup

  - **Contexto:** A UI foi concebida a partir do `mockup_ui.jpg`. Na transição do design para a tela (via `egui`), proporções, tamanhos de fonte, espaçamentos e nuances do Dark Mode podem divergir.
  - **Ações do Testador:**
    1. A DAW deve concluir o scan com sucesso e exibir o plugin na categoria correta.
    2. Inserir o plugin: Abrir o NAM-rs e colocar lado-a-lado com a imagem `mockup_ui.jpg`.
    3. Renderização da UI: A janela precisa ser instanciada com estabilidade, sem redimensionamento indesejado ou comportamentos bizarros de mouse (offset de clique).
    4. Comparar criticamente as 5 zonas de interface: margens, cores exatas dos LEDs/knobs (`#00D4AA`, `#F5A623`), tipografia Inter, e o layout e alinhamento dos medidores VU.
    5. Anotar todas as discrepâncias estéticas e de usabilidade encontradas.
  - **Ação Técnica (Engenharia):** Se houver divergências substanciais, ajustar o código do `egui` em `src/clap/gui/ui.rs` para calibrar o visual.
  - **Aceite:** A janela em execução deve prover uma "sensação visual" extremamente próxima ao mockup original, atestando caráter premium.

- [ ] **Tarefa 4.4.2** — Certificação Funcional e de UX

  - **Contexto:** É preciso validar a experiência do usuário ponta-a-ponta na DAW.
  - **Ações do Testador:**
    1. Testar File Picker: O dialog (`rfd`) não pode travar a interface da DAW nem a thread de áudio. O feedback "Loading..." tem que aparecer transitoriamente.
    2. Testar Controles: Mover todos os knobs e atestar que a resposta do áudio e as labels numéricas (dB) mudam instantaneamente.
    3. Testar Medidores (VU): Inserir áudio com forte dinâmica. Os VUs L/R reagem bem? O "peak hold" dura exatos 2 segundos conforme projetado? A cor clipando no topo acende corretamente?
    4. Testar Automações: O clique/arrasto nos knobs deve gravar trilhas de automação de forma suave no grid (gestures begin/end bem comportados).
    5. Persistência de Sessão: Criar um projeto, alterar parâmetros/modelo, salvar e fechar. Ao reabrir, certifique-se de que o estado é idêntico.
  - **Aceite:** Tudo suave, fluido, responsivo a 60fps. Nenhum XRUN (engasgo no áudio) durante o manuseio da interface.

- [ ] **Tarefa 4.4.4** — Bateria de Stress Test (Smoke Test Cross-Host)

  - **Contexto:** A janela do plugin deve sobreviver a um músico de dedos rápidos e DAWs processando operações intensas.
  - **Ações do Testador:**
    1. Alternar o Bypass alucinadamente e abrir/fechar a GUI mais de 20 vezes consecutivamente.
    2. Tentar carregar modelos 10x rapidamente enquanto a GUI processa áudio.
    3. Modulação Estressante: Aplicar uma modulação via LFO pesada nos knobs e deixar operando por > 5 minutos.
  - **Aceite:** O plugin não pode causar panics na DAW. Zero vazamento de RAM visível monitorando via `htop`.

- [ ] **Tarefa 4.4.5** — Validar viabilidade prática do Modo Headless

  - **Contexto:** Ante o esforço de manter um build headless, precisamos debater a real necessidade do usuário.
  - **Ações do Testador:** Refletir: A ausência da interface egui afeta cenários profissionais essenciais onde o Bitwig rodaria o DSP bruto?
  - **Aceite:** Decisão do PO anotada junto com as justificativas pertinentes.

---

## Sprint 5 — Integrações Premium e Otimizações Multi-Instância

**Objetivo:** Elevar o NAM-rs de "plugin funcional" para "first-class citizen" no Bitwig Studio: mapeamento automático para controladores hardware, economia de CPU em silêncio, suporte à modulação delta (`PARAM_MOD`), compartilhamento de pesos entre instâncias e telemetria ao vivo na UI.
> Nota: Fazer o mesmo no Fender Studio depois!

**Pré-requisito:** Sprint 4 concluída (GUI funcional com parâmetros bidirecionais).

---

### Épico 5.1 — Integrações Nativas Bitwig (First-Class Citizen)

- [ ] **Tarefa 5.1.1** — Remote Controls para Controladores Hardware (`clap_plugin_remote_controls`)

  - **Contexto:** Bitwig usa "Remote Controls" para mapear plugins instantaneamente para Push 3, Maschine e superfícies MIDI sem configuração manual. O plugin deve declarar páginas lógicas de parâmetros.
  - **Ações Técnicas:**
    1. Criar `src/clap/extensions/remote_controls.rs` implementando `PluginRemoteControls`:
       - **Página 0 — "Main":** `input_gain_db` (knob 1), `output_gain_db` (knob 2), `bypass` (botão 3). Knobs 4–8: vazios (reserva futura para EQ pós-amp).
       - **Página 1 — "Gate":** `gate_threshold_db` (knob 1). Knobs 2–8: vazios.
    2. `count()` → `2` páginas.
    3. `get_page(index)` → `RemoteControlsPage` com `name`, `plugin_channel = -1` (sem canal específico) e array de 8 `param_id`.
    4. Registrar a extensão. Reexecutar `clap-validator` — 0 FAILs.
  - **Aceite:** Inserir NAM-rs no Bitwig e conectar um Push 3 → os 4 parâmetros aparecem mapeados automaticamente nos encoders físicos sem nenhum MIDI Learn.

- [ ] **Tarefa 5.1.2** — Accent Color Dinâmico via `clap_plugin_context_menu` / `clap_plugin_track_info`

  - **Contexto:** Bitwig informa ao plugin a cor ARGB da trilha em que está inserido. Podemos usar esta cor como *accent color* da UI, criando integração visual perfeita com o projeto do usuário.
  - **Ações Técnicas:**
    1. Criar `src/clap/extensions/track_info.rs` implementando `PluginTrackInfo`:
       - `on_track_info_changed()`: ler `TrackInfo::color` (ARGB u32) e armazenar em `NamClapShared::ui_accent_color: AtomicU32`.
    2. Na UI thread (`on_frame()`): ler `ui_accent_color`, converter ARGB → `egui::Color32`, usar como cor de destaque nos sliders e nos medidores VU.
    3. Fallback: se cor for `0x00000000` (não definida), usar accent turquesa padrão `#00D4AA` (paleta aprovada na T4.0.2).
  - **Aceite:** Trocar a cor da trilha no Bitwig → UI do NAM-rs muda de cor em < 100ms. Trilha vermelha → accent vermelho. Trilha verde → accent verde.

- [ ] **Tarefa 5.1.3** — Auto-Sleep com Tail Preciso (`clap_plugin_tail`)

  - **Contexto:** Sem esta extensão, o Bitwig continua chamando `process()` eternamente mesmo sem áudio, desperdiçando CPU. Com ela, o host "adormece" o plugin e para de chamar `process()` quando o sinal é silêncio por N samples.
  - **Ações Técnicas:**
    1. Criar `src/clap/extensions/tail.rs` implementando `PluginTail`:
       - `get_tail()`: retornar 0 samples (LSTM/WaveNet não têm tail — são stateful mas não produzem sinal após o silêncio de entrada). **Exceção:** se houver reverb ou delay no modelo (improvável mas possível), permitir configuração via `NamPluginParams`.
    2. Na audio thread: quando o Gate FSM entra em `GateState::Closed`, retornar `ProcessStatus::Sleep` ao invés de `ProcessStatus::Continue` — isto sinaliza ao host que o plugin pode dormir.
    3. Verificar que ao sair do silêncio, o host acorda o plugin chamando `process()` novamente — confirmar via log no Bitwig.
  - **Aceite:** Bitwig para de chamar `process()` em silêncio total (verificável via `dsp_cycle_time` da `RtStatusFlags` zerar). CPU cai para ~0% quando a trilha está muda.

- [ ] **Tarefa 5.1.4** — Modulação Não-Destrutiva (`CLAP_EVENT_PARAM_MOD`)

  - **Contexto:** O sistema The Grid do Bitwig envia `CLAP_EVENT_PARAM_MOD` (um delta que soma ao valor base do parâmetro) ao invés de `CLAP_EVENT_PARAM_VALUE` (que substitui o valor). Isso permite que a automação e a modulação coexistam sem conflito.
  - **Ações Técnicas:**
    1. Em `src/clap/processor.rs`, dentro do loop de eventos do `process()`, adicionar braço para `EventType::ParamMod`:
       - Calcular `effective_value = base_param_value + mod_delta` (clampado aos bounds do parâmetro).
       - Passar `effective_value` para o `ParamSmoother` sem alterar `base_param_value`.
    2. Na UI, ao receber `CLAP_EVENT_PARAM_MOD` via notificação do host: desenhar o "anel de modulação" (arco externo ao slider) mostrando o delta de modulação em relação ao valor base. Usar `egui::Painter::arc()` com cor do accent.
    3. **Invariante:** `CLAP_EVENT_PARAM_MOD` nunca modifica o valor persistido — apenas a saída final do smoother.
  - **Aceite:** Conectar um LFO do Grid ao `input_gain` → o anel de modulação na UI oscila. O valor do knob permanece fixo (apenas o anel se move). Modulação cessa sem artefatos ao desconectar o LFO.

---

### Épico 5.2 — Otimizações Multi-Instância e Telemetria

- [ ] **Tarefa 5.2.1** — Compartilhamento de Pesos (Flyweight Pattern com `Arc`)

  - **Contexto:** Se o usuário inserir 8 instâncias do NAM-rs com o mesmo modelo (comum em backing tracks em paralelo), cada instância carrega seus próprios pesos — N × (tamanho do modelo) de RAM. Com `Arc<DynamicModel>` imutável, **apenas uma cópia** dos pesos existe na RAM para todas as instâncias.

  - **Problema RT-safety:** `Arc::drop()` chama o `Drop` do modelo (que desaloca), o que pode acontecer na audio thread quando a última instância é destruída. Solução: o GC SPSC existente.

  - **Ações Técnicas:**

    1. Criar um `ModelCache` global thread-safe em `src/clap/model_cache.rs`:

       ```rust
       // Cache global de modelos compartilhados. Chave: PathBuf canônico.
       // Usa Mutex<HashMap> — acesso apenas na main thread (cold-path de carregamento).
       static MODEL_CACHE: Lazy<Mutex<HashMap<PathBuf, Weak<DynamicModel>>>> = Lazy::new(Default::default);
       ```

    2. Em `NamClapMainThread::load_model()`: antes de carregar, checar o cache. Se existir um `Arc` forte, reutilizá-lo. Se `Weak::upgrade()` falhar (todas as instâncias foram destruídas), recarregar.

    3. Enviar `Arc<DynamicModel>` (clone do Arc — não do modelo) via SPSC para a audio thread. O `NamClapProcessor` detém um `Arc<DynamicModel>` ao invés de `Box<DynamicModel>`.

    4. O GC ainda é necessário: ao trocar de modelo, a audio thread envia o `Arc` antigo via `gc_tx` para drop na main thread.

    5. Validar com 8 instâncias no Bitwig: `htop` mostra uso de RAM estável (não multiplicado por 8).

  - **Abordagem PoC:** Implementar como **prova de conceito** para validar se o compartilhamento de pesos via `Arc` é viável sem impacto RT. A inferência LSTM/WaveNet é stateful (`h`, `c`, ring buffer) — os pesos são readonly mas o estado é per-instance. A separação pesos/estado pode exigir refatoração nos models. A decisão de permanência será baseada nos resultados de benchmark e na complexidade real da refatoração.

  - **Nota (Auditoria S0-S2):** A migração de `Box<DynamicModel>` para `Arc<DynamicModel>` impacta diretamente `GcItem` (enum em `src/common/spsc.rs`, variante `Model(Box<DynamicModel>)`) e o método `NamClapProcessor::push_to_gc()`. Ambos precisarão ser adaptados para aceitar `Arc` ao invés de `Box`. O `GcOverflowBuffer` com `AtomicPtr` também deverá ser revisado.

  - **Aceite:** 8 instâncias com mesmo modelo: RAM = 1× modelo + overhead mínimo por instância. `valgrind --tool=massif` confirma.

- [ ] **Tarefa 5.2.2** — Telemetria ao Vivo na UI: Histograma de Latência DSP

  - **Contexto:** `src/dsp/telemetry.rs` já possui `LatencyHistogram` com P50/P95/P99 e `get_percentile()`. No Standalone, esses dados são exibidos no terminal. No plugin, a UI pode exibir esses dados em tempo real — uma funcionalidade única e premium para o músico profissional.
  - **Dependência:** A T4.3.1 define o botão [📊] na Zona 5 (Status Bar) e o painel de telemetria colapsado por padrão. Esta tarefa implementa o **conteúdo** desse painel.
  - **Ações Técnicas:**
    1. Em `NamClapShared`, adicionar um `Arc<LatencyHistogram>` compartilhado entre audio thread (que chama `record()`) e UI thread (que chama `get_percentile()`).
    2. Na audio thread (`process()`): medir o tempo de execução com `std::time::Instant` (ou `rdtsc` como no Standalone) e chamar `hist.record(duration_ns)`. `LatencyHistogram::record()` já é RT-safe (zero-alloc, atômico).
    3. Na UI thread (`on_frame()`), implementar o painel expansível acionado pelo botão [📊] da Zona 5:
       - P50: `hist.get_percentile(0.50)` → exibir em µs.
       - P95: `hist.get_percentile(0.95)` → exibir em µs.
       - P99: `hist.get_percentile(0.99)` → **vermelho se > 70% do budget** do buffer (budget = `n_frames / sample_rate * 1e6` µs).
       - Botão `[Reset]`: chamar `hist.reset()`.
  - **Aceite:** Com um modelo LSTM carregado, a UI mostra P95 < 1ms para buffer de 256 samples @ 48kHz. P99 visível em vermelho se houver spike de CPU.

- [ ] **Tarefa 5.2.3** — Investigação de `clap_host_thread_pool` para GEMV Paralelo

  - **Contexto:** Bitwig pode prover sua própria infraestrutura de thread pool via `clap_host_thread_pool`. Se disponível, podemos paralelizar o GEMV das camadas LSTM (cálculo das 4 gates em paralelo) sem gerenciar threads próprias — reutilizando o pool já priorizado do host.
  - **Status:** Investigação e prova de conceito. Implementar apenas se o Bitwig confirmar suporte à extensão.
  - **Ações Técnicas:**
    1. Em `new_shared()`, checar `host.get_extension::<HostThreadPool>()`. Logar resultado via `log::info!`.
    2. Se disponível: prototipar paralelização das 4 gates LSTM (`f_gate`, `i_gate`, `o_gate`, `g_gate`) como tasks independentes enviadas ao thread pool.
    3. Medir via `LatencyHistogram`: comparar P95 antes e depois da paralelização.
    4. **Critério de abandono:** Se P95 aumentar (overhead de sincronização > ganho de paralelismo), abandonar e documentar o resultado no `docs/architecture.md`.
  - **Aceite (se viável):** P95 reduz ≥ 15% para modelos LSTM de 2+ camadas em buffers de 256+ samples. Caso contrário: decisão documentada de não-implementação com justificativa de benchmark.

---

### Épico 5.3 — Unificação do Sistema de Dispatch SIMD (Eliminar Dual Dispatch)

> **Origem:** Pesquisador-Inovador (2026-05-14, item A1). Dívida técnica documentada em `src/math/common/dispatch.rs` desde 2026-05-12.

- [ ] **Tarefa 5.3.1** — Migrar Consumidores da V-Table `SimdMathConfig` para a Trait `SimdMath`

  - **Contexto:** O codebase usa dois mecanismos de dispatch SIMD independentes: (1) trait `SimdMath` com monomorphization estática via `dispatch_simd!` e (2) v-table `SimdMathConfig` com 13 ponteiros de função. O Mecanismo 2 impede inline e adiciona ~1-2 ciclos de indireção por chamada. O principal benefício é **higiene arquitetural** — unificar em uma única API. (Dívida documentada em `src/math/common/dispatch.rs` e `docs/architecture.md` §2 "Design Debt".)
  - **Ações Técnicas:**
    1. Refatorar `pipeline.rs`: substituir chamadas via `SIMD_MATH.apply_gain(...)` por `dispatch_simd!(apply_gain, ...)` (Mecanismo 1).
    2. Refatorar `gain.rs`: migrar `apply_gain_simd` e `apply_ramp_simd` para usar `dispatch_simd!`.
    3. Refatorar `rt_setup.rs` e `cli.rs`: migrar referências a `SimdMathConfig::current()`.
    4. Após migração completa: remover a struct `SimdMathConfig` e os ponteiros de função, mantendo apenas `InstructionSet` para consultas de capabilities (`is_avx512`).
    5. Executar `cargo bench` para confirmar zero regressão de performance.
  - **Aceite:** Zero referências a `SimdMathConfig` no codebase. Apenas `dispatch_simd!` e trait `SimdMath` em uso. Benchmarks dentro de ±2% do baseline.

---

### Épico 5.4 — LSTM Layer Overlap Pipelining (2-Layer Interleaving Real)

> **Origem:** Pesquisador-Inovador (2026-05-14, item A3). A arquitetura documenta o conceito em `docs/architecture.md` §2, mas a implementação atual em `LstmModel2` processa as camadas sequencialmente.

- [ ] **Tarefa 5.4.1** — Implementar Pipelining Real para `LstmModel2`

  - **Contexto:** O verdadeiro pipelining faz a Camada 2 processar o frame N-1 enquanto a Camada 1 processa o frame N, ocultando a latência da Camada 2 dentro do tempo de processamento da Camada 1. Isso é ILP puro (single-thread, zero overhead de sincronização).
  - **Decisão aprovada:** A latência adicional de 1 frame (~20.8µs @ 48kHz) é aceita. Os golden vectors existentes **não serão alterados** — a validação usará golden vectors dedicados ao modo pipelined.
  - **Ações Técnicas:**
    1. Em `src/models/lstm/layer.rs`, adicionar campo `pipeline_buf: [f32; H]` ao `LstmModel2` para armazenar a saída da Camada 1 do frame anterior.
    2. Reestruturar o loop `process()` do `LstmModel2`:
       - Frame 0 (priming): processar apenas Camada 1, armazenar saída em `pipeline_buf`.
       - Frames 1..N: Camada 1 processa frame N → `pipeline_buf`; Camada 2 processa `pipeline_buf` do frame N-1 → output.
       - Frame final: Camada 2 processa o último `pipeline_buf` → output.
    3. Atualizar `reset_states()` para zerar `pipeline_buf`.
    4. Criar golden vectors dedicados para modelos `Lstm2x*` com pipelining ativo.
    5. `cargo bench` para medir redução de P95 em modelos 2×16.
  - **Aceite:** P95 reduz ≥ 20% para `Lstm2x16` em buffers de 256 samples. Golden vectors pipelined passam. Modelos 1-layer inalterados.

---

### Épico 5.5 — Prefetch Hints no WaveNet Conv1D (Software Prefetching)

> **Origem:** Pesquisador-Inovador (2026-05-14, item A4). Aprovado com validação obrigatória por benchmarks.

- [ ] **Tarefa 5.5.1** — Investigar e Implementar `_mm_prefetch` na Conv1D Dilatada

  - **Contexto:** O WaveNet itera sobre camadas com dilatação crescente (1, 2, 4, 8...). Dilatações profundas (≥ 64) acessam posições distantes no ring buffer, potencialmente fora da L1. Um prefetch NTA na iteração N para os dados da iteração N+1 pode esconder ~3-4 ciclos de latência L2.
  - **Critério de abandono:** Se o benchmark não mostrar melhoria ≥ 3%, reverter e documentar.
  - **Ações Técnicas:**
    1. Em `src/models/wavenet/conv1d.rs`, adicionar `_mm_prefetch(..., _MM_HINT_NTA)` no início de cada iteração de camada, prefetchando os taps da próxima camada.
    2. Executar `cargo bench` com modelos Standard (16ch) e Lite (12ch) comparando com/sem prefetch.
    3. Se melhoria < 3%: reverter, documentar resultado no código e no `docs/architecture.md`.
    4. Se melhoria ≥ 3%: manter, adicionar comentário explicativo no código.
  - **Aceite (se viável):** Melhoria ≥ 3% no P95 para modelos WaveNet Standard. Caso contrário: decisão documentada de não-implementação com dados de benchmark.

---

## Sprint 6 — Validação Final e Alpha Release

**Objetivo:** Certificar que o NAM-rs v2.0.0-alpha é estável, conformante com o spec CLAP, documentado e publicável. Zero regressões em relação ao modo Standalone.

**Pré-requisito:** Sprints 1–5 concluídas. Todos os testes automatizados verdes.

---

### Épico 6.1 — Auditoria Final de Código

- [ ] **Tarefa 6.1.1** — Revisão Crítica de Todo Código CLAP Adicionado (Dev Humano)

  - **Contexto:**
    Ao longo de 5 sprints de desenvolvimento rápido, múltiplos módulos foram criados em `src/clap/` (processamento de áudio, suavização de parâmetros, extensões CLAP e interface egui). É imperativo realizar uma auditoria rigorosa baseada nas Diretrizes Técnicas e de Performance para garantir coesão arquitetural, nomenclatura aderente à convenção do Rust, ausência de vazamento de memória e que todas as operações no hot-path obedeçam às garantias de RT-Safety (ex: zero blocos de lock, zero heap drops indiretos e tratamento seguro de concorrência).

  - **Ações Humanas:**
    1. Skill `revisor-auditor` focado na pasta `src/clap/`. Demais pastas apenas nas suas realações com `src/clap/`.
    2. Ler o código em `src/clap/` e assegurar completa cobertura de docsys e comentários inline no código fonte.

- [ ] **Tarefa 6.1.2** — Auditoria Completa com `clap-validator`

  - **Contexto:**
    O `clap-validator` da CLAP-SDK é a ferramenta oficial de conformidade para atestar que o plugin obedece de forma irrestrita às especificações e ciclos de vida do padrão CLAP (inclusão de portas, transição de estados atômicos, ativação/desativação dinâmica e carregamento de parâmetros). Esta é a auditoria final que executará toda a suíte de testes de estresse estático contra o binário unificado com todas as extensões ativas.

  - **Ações Técnicas:**
    1. Compilar o plugin em perfil de release otimizado executando `./utils/build-clap.sh`.
    2. Instalar o `clap-validator` mais recente compatível com o spec CLAP v1.2+.
    3. Rodar a validação do binário exportado em formato detalhado e gravar a saída estructurada:

       ```bash
       clap-validator validate ~/.clap/nam-rs.clap --output json > /tmp/final-validation.json
       ```

    4. Validar o arquivo JSON de saída usando `jq` para inspecionar falhas ou avisos em extensões de parâmetros, latência, estado persistente e interface gráfica.
    5. Caso algum teste dinâmico falhe (como pânicos de ciclo de vida induzidos por ativações consecutivas no validador), depurar usando `gdb` ou a skill `debugger` e corrigir o código de inicialização do plugin.

  - **Aceite:**
    - `clap-validator` reporta conformidade total com score de 100% de passes (zero FAILs na suíte oficial de validação).
    - Execução do validador em modo de estresse não gera erros de concorrência ou crashes no arquivo de dump.

- [ ] **Tarefa 6.1.3** — Modo Bilingue

  - **Contexto:**
    Detecta automaticamente o idioma do sistema operacional do usuário e traduz dinamicamente a interface gráfica (CLAP GUI) e as mensagens exibidas no terminal (CLI Standalone). O suporte abrange o Inglês Internacional (idioma padrão) e o Português Brasileiro. Como o NAM-rs é projetado sob restrições estritas de tempo real (RT-Safety), o sistema de tradução deve operar sem alocações dinâmicas de memória (zero heap allocations no lookup de strings) para garantir que threads de alta prioridade de áudio ou loops de renderização da UI a 60fps não sofram interrupções induzidas pelo alocador de heap.

  - **Ações Técnicas:**
    1. Criar o módulo de internacionalização em `src/common/l10n.rs` contendo:
       - O enum `Lang` com variantes `En` (default) e `PtBr`.
       - Uma função RT-safe de detecção do idioma do sistema `detect_system_language() -> Lang` que lê as variáveis de ambiente `LANG`, `LC_ALL` e `LC_MESSAGES` no Linux, retornando `PtBr` se contiverem "pt", caso contrário, retornando `En` como fallback.
       - Um enum central `MsgId` representando todas as chaves de string estáticas traduzíveis do projeto.
       - Um método/função associada `MsgId::tr(&self, lang: Lang) -> &'static str` que mapeia de forma direta e sem alocações cada chave para a sua respectiva tradução estática.
    2. Modificar o módulo CLI em `src/standalone/cli.rs`:
       - Executar a detecção do idioma ao inicializar o CLI (como no `print_help()` e no tratamento de erros do `parse_args()`).
       - Traduzir a tela de ajuda de uso, as opções da CLI e todas as mensagens de erro críticas.
    3. Modificar o desenho da interface gráfica em `src/clap/gui/ui.rs`:
       - Inicializar o idioma detectado uma única vez na criação do estado compartilhado ou na inicialização da janela.
       - Substituir todos os literais de string na interface egui (ex: `"📂 Load Model"`, `"No model loaded"`, `"INPUT"`, `"GATE"`, `"OUTPUT"`, `"BYPASS"`, status de latência, badges de ISA e telemetria expandida) por lookups através de `MsgId::tr()`.
    4. Adicionar testes unitários dedicados em `src/common/l10n_test.rs` (ou inline no próprio arquivo `l10n.rs` se possuir menos de 300 linhas, em conformidade com as Convenções de Testes):
       - Validar a corretude da detecção de idioma via variáveis de ambiente injetadas em ambiente de teste controlado.
       - Garantir que todas as chaves de tradução mapeadas em `MsgId` possuam correspondentes válidos nos dicionários de ambos os idiomas.
       - Certificar a ausência de alocações no heap durante a invocação da função de tradução.
    5. Assegurar a presença do cabeçalho de Copyright/Licença SPDX no topo de todos os novos arquivos de código criados.

  - **Aceite:**
    - Executar `LANG=pt_BR.UTF-8 cargo run --features standalone -- -h` exibe o manual da CLI inteiramente traduzido em Português do Brasil.
    - Executar `LANG=en_US.UTF-8 cargo run --features standalone -- -h` exibe o manual da CLI em Inglês.
    - Ao instanciar a CLAP GUI em hosts configurados com localidade em português, a janela exibe controles ("ENTRADA", "SAÍDA", "PORTÃO" ou equivalentes), status e botões em português.
    - `cargo test` passa com sucesso em todas as suítes e `./utils/lints.sh` é executado com zero warnings/erros.

### Épico 6.2 — Sessão de Estresse Final em DAWs

- [ ] **Tarefa 6.2.1** — Endurance de 1 Hora com Carga Máxima

  - **Contexto:**
    Antes do lançamento Alpha, é crucial assegurar que o plugin não apresenta vazamentos de recursos (memória, file descriptors) ou degradação de performance ao longo de sessões prolongadas sob alta carga no Bitwig Studio e no Fender Studio Pro.
  - **Ações Técnicas:**
    1. Montar um projeto de teste de endurance na DAW contendo 4 instâncias concorrentes do NAM-rs utilizando modelos de redes neurais distintos (WaveNet e LSTM) e com 2 moduladores LFO associados a cada parâmetro.
    2. Adicionar uma trilha de sinal idêntica em paralelo com fase invertida e compensação automática de atraso (ADC) ativa para confirmar que o alinhamento de fase é mantido perfeitamente no tempo.
    3. Iniciar um script de monitoramento em segundo plano que extrai informações de `/proc/$(pgrep -f "bitwig-studio" | head -n 1)/status` e monitora uso de memória residente (RSS), virtual (VSZ) e o contador de XRUNs do PipeWire/JACK.
    4. Inspecionar o log de kernel (`dmesg -T`) e o arquivo de logs do Host à procura de violações de thread access ou pânicos silenciosos.
  - **Aceite:**
    - Zero crashes ou travamentos nos hosts ao longo dos 60 minutos de reprodução contínua.
    - O consumo de memória RSS do processo do plugin/host deve estabilizar (variação menor que 5MB após os primeiros 5 minutos de aquecimento, confirmando ausência de vazamentos no loop da GUI e nos buffers circulares).
    - Latência de áudio estável com zero XRUNs acumulados decorrentes do NAM-rs.

- [ ] **Tarefa 6.2.2** — Regressão do Modo Standalone

  - **Contexto:**
    A introdução de novos feature flags (`clap-plugin` e `clap-plugin-gui`) e a refatoração do pipeline DSP principal podem ter afetado o modo standalone (PipeWire/JACK nativo). É vital atestar que o NAM-rs rodando em modo CLI standalone preserva suas propriedades de baixa latência e compatibilidade.
  - **Ações Técnicas:**
    1. Executar a suíte de testes de integração com a flag de standalone ativa (`cargo test --features standalone`).
    2. Rodar benchmarks de performance DSP no hot-path (`cargo bench --features standalone`) e confrontar os resultados P50 e P95 contra o baseline histórico pré-Sprint-1.
    3. Testar a inicialização do executável standalone conectando-o dinamicamente ao PipeWire sob taxas de amostragem comuns (44.1kHz e 48kHz), verificando se o escalonamento em tempo real (RT) e as prioridades do kernel são atribuídos corretamente.
    4. Garantir que as diretrizes de linting de qualidade e higiene de código se aplicam igualmente (`./utils/lints.sh`).
  - **Aceite:**
    - Executável standalone compila e todos os testes unitários/integração integrados passam com status de sucesso.
    - Desempenho do benchmark DSP de inferência e resampling standalone mantido com variação máxima tolerada de ±2% em comparação ao baseline pré-sprint.

- [ ] **Tarefa 6.2.3** — Testes A/B de Fidelidade e Integridade de Inferência

  - **Contexto:**
    Para garantir a acurácia matemática e fidelidade do processamento DSP e das aproximações matemáticas de funções de ativação do NAM-rs (Wavenet/LSTM) em relação à implementação original de referência em Python/C++ (sdatkinson), devemos realizar testes de verificação de sinal.
  - **Ações Técnicas:**
    1. Coletar um conjunto representativo de arquivos de modelo (.nam e .namb) do repositório público (como Tone3000) cobrindo diferentes arquiteturas e tamanhos (WaveNet Standard, LSTM 1 camada, LSTM 2 camadas).
    2. Instale o NAM-rs e a implementação oficial de referência sobre faixatracks com áudio real e também sobre sinais de entrada de varredura (sine sweep e ruído branco), garantindo que os dois estejam perfeitamente idênticos.
    3. Computar a diferença RMS ponto a ponto entre os sinais gerados (sinal de erro) e validar que a diferença de magnitude não ultrapassa o limite de ruído de quantização exigido (-80 dBFS).
    4. Corrigir discrepâncias de amostragem, coeficientes de filtro FIR de resampling ou tratamento de ganhos caso o desvio seja maior que o tolerável.
  - **Aceite:**
    - Erro RMS de discrepância tonal de inferência em relação à implementação de referência é estritamente inferior a -80 dB em todos os modelos testados.
    - Zero problemas de pânico ou silenciamento nos blocos processados.

### Épico 6.3 — Documentação e Alpha Release

- [ ] **Tarefa 6.3.1** — Atualizar `docs/architecture.md`

  - **Contexto:**
    Toda a arquitetura e detalhes técnicos consolidados nas Sprints 4 e 5 (GUI em egui/baseview, modo bilingue, compartilhamento de pesos base via Arc e histograma de latência DSP) devem ser documentados de maneira detalhada para manter o repositório organizado e alinhado com o estado real do código.
  - **Ações Técnicas:**
    1. Atualizar a seção §8.1 de `docs/architecture.md` detalhando as threads (Main, Audio, UI), a comunicação lock-free SPSC implementada e as restrições de RT-Safety.
    2. Adicionar diagramas de arquitetura (formato Mermaid) que expliquem o fluxo de eventos e dados de telemetria.
    3. Documentar a matriz de suporte de Feature Flags atualizada com os perfis `standalone`, `clap-plugin` e `clap-plugin-gui` e as diretrizes para empacotamento.
    4. Adicionar capturas de tela/mockups de homologação da UI em execução nas DAWs (Bitwig e Fender Studio Pro).
  - **Aceite:**
    - Documento de arquitetura sincronizado com o estado final da branch principal.
    - Diagramas em Mermaid renderizados perfeitamente e sem erros sintáticos.

- [ ] **Tarefa 6.3.2** — Checklist Pré-Release e Tag Git

  - **Ações Técnicas:**
    1. Realizar uma varredura final no repositório com o script `./utils/lints.sh` e auditoria manual para apagar logs temporários, arquivos `.tmp`, dumps e artefatos que não estejam listados no `.gitignore`.
    2. Incrementar a versão no arquivo `Cargo.toml` para `2.0.0` e atualizar o histórico de alterações (`CHANGELOG.md` ou similar).
    3. Criar a Tag git correspondente à versão Alpha e submetê-la ao repositório remoto.
    4. Preparar o manifesto de compilação da release e estruturar os metadados para distribuição preliminar (ex: empacotamento em tarball, pacotes para Flathub ou instaladores apropriados).
  - **Aceite:**
    - Tag de release criada e publicada.
    - Binários finais gerados no CI sem warnings e prontos para teste público da comunidade.

> **📋 FINALIZAÇÃO — Sprint 6 concluída**
> O NAM-rs v2.0.0 estará certificado: 100% no `clap-validator`, 60min de endurance sem falhas em Bitwig E Fender Studio Pro, modo Standalone sem regressões, e documentação sincronizada. Pronto para uso em produção no Bitwig Studio e Fender Studio Pro.
