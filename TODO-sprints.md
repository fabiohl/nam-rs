<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->
# TODO-sprints — Plano de Sprints (Auditoria CLAP: Threading, Features, GUI)

---

## Sprint G1 — Threading CLAP (RT, latência e responsividade)

### Tarefa G1.T01 — Eliminar False Sharing em `NamClapShared` (segregação por padrão de acesso) 🔥⚠️ ✅ CONCLUÍDA

- **Status:** Implementada. `NamClapShared` agora contém 3 sub-structs com `#[repr(align(128))]` isoladas:
  - `rt_to_ui: RtToUi` — `ui_peak_l`, `ui_peak_r`, `ui_clipped`, `current_latency`, `active_channel_count`
  - `ui_to_rt: UiToRt` — `param_input_gain`, `param_output_gain`, `param_gate_thresh`, `param_bypass`, `param_adaptive_compute`, `gesture_flags`
  - `cold: ColdShared` — demais campos (SPSC, rt_status, model info, etc.)
- **Teste de layout:** `offset_of!(NamClapShared, ui_to_rt) - offset_of!(NamClapShared, rt_to_ui) >= 128`, idem para cold.
- **Nota para G1.T02:** O sub-struct `UiToRt` está pronto para receber `gui_param_generation: AtomicU32`.
- **Nota para G1.T03/G1.T04:** Sem alterações necessárias nos acessos de `current_latency`/`sample_rate` (agora via `rt_to_ui`/`cold`).

- **Onde:** `src/clap/plugin/shared.rs:57-129` (struct `NamClapShared`).
- **Problema:** A struct é anotada com `#[repr(align(128))]`, mas isso só alinha o **início** da struct — os campos atômicos individuais permanecem empacotados, compartilhando linhas de cache. No hotpath, o audio thread **escreve a cada bloco** em `ui_peak_l`, `ui_peak_r`, `ui_clipped` (`dsp.rs:98-112,506-520`) e **lê a cada bloco** `param_input_gain`, `param_output_gain`, `param_gate_thresh`, `param_bypass`, `param_adaptive_compute` (`events.rs:143-176`). A GUI/Main escreve nesses mesmos `param_*`. Com os campos misturados na mesma(s) linha(s) de cache, ocorre **cache-line bouncing** (a escrita de peak pelo RT invalida a linha onde estão params lidos no mesmo bloco e escritos pela GUI), violando a diretriz §4 de `.agents/rules/rust.md` ("False Sharing: isolar linhas de cache").
- **Solução técnica:**
  1. Agrupar os atômicos por **direção e frequência de acesso** em sub-structs cada um `#[repr(align(128))]`:
     - `RtToUi` (escrito-a-cada-bloco pelo RT, lido pela UI): `ui_peak_l`, `ui_peak_r`, `ui_clipped`, `current_latency`, `active_channel_count`.
     - `UiToRt` (escrito pela UI/Main, lido-a-cada-bloco pelo RT): `param_input_gain`, `param_output_gain`, `param_gate_thresh`, `param_bypass`, `param_adaptive_compute`, `gesture_flags`.
     - `ColdShared` (baixa frequência): `model_sample_rate`, `sample_rate`, `buffer_size`, `track_accent_color`, `param_indication*`, `model_load_counter`, mutexes de UI/metadata.
  2. Manter a API pública estável (métodos `set_gesture`/`take_gesture`/`write_gui_events` delegam aos sub-structs) para minimizar churn nos call-sites.
  3. Validar o layout com `std::mem::offset_of!` em teste, assegurando que `RtToUi` e `UiToRt` não compartilham linha de 128 bytes.
- **Critérios de aceitação:**
  - Teste de layout confirma offsets de `RtToUi` e `UiToRt` em linhas de cache distintas (≥128 B de separação).
  - `inference_bench`/microbench de `process()` não regride; sob carga estéreo com automação simultânea de gain, perf de bloco melhora ou mantém-se (medir P99 via `LatencyHistogram`).
  - `cargo test --features clap-plugin` verde; nenhuma mudança de comportamento observável.
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 1.5 dia.

### Tarefa G1.T02 — Sincronização GUI→RT por geração única (substituir 5 loads+compares por bloco) 🔥✅ CONCLUÍDA

- **Onde:** `src/clap/processor/events.rs:142-176` e bloco espelhado em `src/clap/extensions/params.rs:379-427` (`PluginAudioProcessorParams::flush`).
- **Problema:** A cada `process()` o RT executa **cinco** pares load-atômico + comparação contra o mirror local (`shared_in_db != self.params.input_gain_db`, etc.) apenas para detectar mudanças vindas da GUI que o host não ecoou como evento. No caso comum (nenhuma mudança de GUI), isso é trabalho 100% desperdiçado no hotpath, repetido por bloco.
- **Solução técnica:**
  1. Introduzir um `AtomicU32 gui_param_generation` em `UiToRt` (ver G1.T01). Todo write de parâmetro pela GUI faz `fetch_add(1)` após gravar o valor (Release).
  2. No RT, manter `last_seen_generation: u32`. No início de `process_events`, ler a geração (Acquire) uma única vez; **só** entrar no bloco de reconciliação dos 5 parâmetros quando `generation != last_seen_generation`.
  3. Aplicar a mesma guarda em `params.rs::flush`.
- **Critérios de aceitação:**
  - No caminho sem mudança de GUI, `process_events` realiza no máximo 1 load atômico relacionado à sincronização de params (geração), comprovado por inspeção/contadores em teste.
  - Mudanças de knob/bypass via GUI continuam refletidas no próximo bloco (latência de resposta inalterada; teste de integração de gesto→valor).
  - Sem `SeqCst`; usar `Acquire`/`Release` apenas no par geração↔valores (conforme §4 de `rust.md`).
- **Especialista:** `implementador`.
- **Esforço:** 1 dia.

### Tarefa G1.T03 — Evitar captura de `Instant::now()` em blocos sem medição (decimação completa) 🧹

- **Onde:** `src/clap/processor/mod.rs:295` (`let start_time = minstant::Instant::now();`) e `src/clap/processor/dsp.rs:523-552`.
- **Problema:** `minstant::Instant::now()` (RDTSC) é capturado em **todo** bloco, embora a telemetria só seja registrada 1-em-16 (`cycles_since_telemetry & 0xF == 0`). É barato, mas é trabalho determinístico desnecessário no hotpath em 15 de cada 16 blocos.
- **Solução técnica:**
  1. Promover o contador de decimação para o `NamClapProcessor` e computar `should_measure` **antes** de `process()` chamar a medição.
  2. Capturar `start_time` apenas quando `should_measure` for verdadeiro (`Option<Instant>`), passando-o adiante; quando `None`, pular o bloco de telemetria inteiro.
  3. Manter a invariante de que o feed do `AdaptiveCompute` continua recebendo amostras representativas (1-em-16 já é a cadência atual).
- **Critérios de aceitação:**
  - Em blocos não medidos, nenhuma chamada a `Instant::now()`/`elapsed()` ocorre (inspeção + teste com mock de relógio, se viável).
  - Telemetria (`dsp_cycle_time`, `latency_hist`, `dsp_overloads`) e adaptive compute mantêm comportamento equivalente (tolerância de cadência idêntica).
- **Especialista:** `implementador`.
- **Esforço:** 0.5 dia.

### Tarefa G1.T04 — Vetorizar o caminho "parâmetro alterado" do ganho (remover loop escalar por amostra) 🧹

- **Onde:** `src/clap/processor/dsp.rs:175-183` e `:221-228` (ramo `param_*_changed` aplicando `smoother.tick()` amostra-a-amostra).
- **Problema:** Quando um parâmetro de ganho muda no bloco, o código cai num **loop escalar por amostra** (`self.smoother_in.tick()`), enquanto o caminho estacionário usa kernels SIMD (`apply_gain_simd`/`apply_ramp_simd`/`*_stereo`). O ramo escalar contraria a diretriz §3 (auto-vetorização) e é o pior caso justamente sob automação intensa.
- **Solução técnica:**
  1. Como o `ParamSmoother` é IIR 1-polo, pré-computar o destino e converter o trecho de convergência num **ramp linear por bloco** já suportado por `apply_ramp_simd`/`apply_ramp_stereo` (o caminho `else` já faz isso). Unificar: ao detectar `param_changed`, calcular `start`/`step` e delegar ao kernel SIMD de ramp, atualizando o estado do smoother ao final.
  2. Validar que a diferença audível (zipper/click) permanece imperceptível (a rampa linear por bloco com cutoff equivalente já é usada no caminho não-alterado).
- **Critérios de aceitação:**
  - Nenhum loop escalar por amostra de ganho remanescente no hotpath (mono e estéreo).
  - Teste de continuidade: varredura de gain de −60→+12 dB em blocos consecutivos sem descontinuidade > limiar de click; paridade com baseline dentro de 1e-4.
- **Especialista:** `implementador` + `pesquisador-inovador` (validação DSP).
- **Esforço:** 1 dia.

---

## Sprint G2 — CLAP Features (extensões corretas e idiomáticas)

### Tarefa G2.T01 — Preset Discovery Factory + `clap.preset-load` (modelos .nam/.namb como presets) 🔥✨

- **Onde:** novo módulo `src/clap/factory/preset_discovery.rs` e `src/clap/extensions/preset_load.rs`; registro em `src/clap/mod.rs` e `src/clap/plugin/mod.rs` (`declare_extensions`).
- **Problema/Oportunidade:** Para um amp modeler, os "presets" **são** os arquivos de modelo (`.nam`/`.namb`). Hoje o carregamento só ocorre via diálogo próprio/drag-drop (`gui/mod.rs`). Implementar a **Preset Discovery Factory** (`clap_preset_discovery_factory`) + a extensão **`clap.preset-load`** permite que o browser de presets do host (Bitwig, Reaper) indexe diretórios de modelos e dispare carregamento — fluxo idiomático e de altíssimo valor percebido.
- **Solução técnica:**
  1. **Preset Discovery Provider:** declarar location(s) configuráveis (ex.: `~/.nam/models`, mais os `model_search_paths` persistidos) e filetypes `nam`/`namb`; emitir metadados (nome, autor, gear) reusando o parsing já existente em `loader` + `NamModelMetadata`.
  2. **`clap.preset-load`:** implementar `from_location(location_kind, location, load_key)` que enfileira o path em `ui_pending_model` e dispara `request_callback()` — reusando o pipeline de `NamClapMainThread::load_model` (Main Thread, RT-safe via SPSC).
  3. Garantir compatibilidade com a persistência de estado (basename + search paths) já implementada em `extensions/state.rs`.
  4. Cobrir indexação e load via `clack-host` em teste de integração.
  5. Atualizar o `docs/functional-tests.md` adequadamente para que esta funcionalidade seja auditada.
- **Critérios de aceitação:**
  - Host com suporte a preset-discovery lista modelos `.nam`/`.namb` dos diretórios declarados com metadados corretos.
  - Selecionar um preset no host carrega o modelo no plugin (verificado por `model_load_counter` incrementado e parâmetro `Active Model` atualizado).
  - `cargo test --features clap-plugin` cobre discovery + load; sem alocação/IO no RT (load roda no Main Thread).
- **Especialista:** `implementador` + `pesquisador-inovador`; revisão `revisor-auditor`.
- **Esforço:** 3–4 dias.

### Tarefa G2.T02 — Fallback de GUI flutuante e robustez de API gráfica ✨

- **Onde:** `src/clap/extensions/gui.rs:17-27` (`is_api_supported`/`get_preferred_api`).
- **Problema:** A GUI aceita **somente** X11 embutido e **rejeita** modo flutuante (`!configuration.is_floating`). Hosts que só oferecem janelas flutuantes (ou ambientes onde o embedding X11 falha) ficam sem GUI. Embedding continua sendo o preferido; falta apenas um fallback.
- **Solução técnica:**
  1. Em `is_api_supported`, aceitar `X11` tanto embutido quanto flutuante; manter `get_preferred_api` retornando embutido (preferência).
  2. Implementar o caminho flutuante em `set_transient`/`create` quando `is_floating == true`, abrindo a janela baseview sem parent (top-level) e respeitando `set_transient` para empilhar sobre a janela do host.
  3. Logar via `HostLog` o modo efetivamente selecionado (já há precedente em `gui/window/mod.rs:85-93`).
- **Critérios de aceitação:**
  - Host que só suporta GUI flutuante exibe a interface corretamente; host com embedding continua usando embedding.
  - Sem regressão no fluxo X11 embutido atual; `cargo build --features clap-plugin` ok.
- **Especialista:** `implementador`.
- **Esforço:** 1.5 dia.

### Tarefa G2.T03 — Revisão das feature strings do descriptor 🧹

- **Onde:** `src/clap/descriptor.rs:15-21`.
- **Problema:** A lista declara `c"simulator"`, que **não** é uma feature padrão definida em `clap/plugin-features.h` (as padronizadas relevantes são `audio-effect`, `distortion`, `gate`, `mono`). Strings não-padrão são ignoradas pela maioria dos hosts e podem prejudicar categorização/busca.
- **Solução técnica:**
  1. Validar cada string contra a versão de `plugin-features.h` correspondente ao `clack`/CLAP em uso.
  2. Remover/ajustar `simulator`; manter o conjunto padrão (`audio-effect`, `distortion`, `gate`, `mono`). Avaliar incluir `mastering`/`utility` apenas se fizer sentido editorial (opinativo — provavelmente não).
  3. Comentar no código a fonte (arquivo/versão) das constantes para auditoria futura.
- **Critérios de aceitação:**
  - Todas as features declaradas existem no header CLAP da versão alvo (citado em comentário).
  - Plugin aparece nas categorias corretas no host de referência (Bitwig).
- **Especialista:** `revisor-auditor`.
- **Esforço:** 0.25 dia.

### Tarefa G2.T04 — Extensões `render` (offline determinístico) e `state-context` ✨

- **Onde:** novos `src/clap/extensions/render.rs` e `src/clap/extensions/state_context.rs`; registro em `plugin/mod.rs`.
- **Problema/Oportunidade:** (a) NAM é **determinístico e causal** — declarar `clap.render` permite ao host sinalizar render offline/realtime, útil para bounce/export consistente. (b) `clap.state-context` permite distinguir save para **preset** vs **projeto** vs **duplicação**, evitando, por exemplo, persistir paths absolutos em contexto de preset portável.
- **Solução técnica:**
  1. `render`: implementar `has_hard_realtime_requirement() -> false` e `set(mode)`; no modo offline, o `AdaptiveCompute` pode ser forçado a `Off` (qualidade máxima, sem soft-degrade) já que não há deadline RT.
  2. `state-context`: rotear `save/load` por `context_type`; em contexto de preset, priorizar persistência por basename + search paths (já existe a infra em `state.rs`).
  3. Testes com `clack-host` para ambos os modos.
- **Critérios de aceitação:**
  - Em render offline, `AdaptiveCompute` não degrada (flag `RT_STATUS_DEGRADE_*` permanece limpa); saída bit-idêntica entre execuções offline.
  - `state-context` salva/restaura corretamente em cada contexto; preset permanece portável.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 2 dias.

---

## Sprint G3 — GUI (eficiência, limpeza e robustez)

### Tarefa G3.T01 — Renderização condicional / idle (honrar o agendamento de repaint do egui) 🔥

- **Onde:** `src/clap/gui/window/mod.rs:202-256` (`on_frame`) e `src/clap/gui/ui/mod.rs:204` (`request_repaint_after(30ms)`).
- **Problema:** Em integração manual baseview, `on_frame` é dirigido pelo loop do baseview e roda **incondicionalmente** a cada frame; os `ui.ctx().request_repaint_after(...)` espalhados (ex.: `ui/mod.rs:204`) **não estão conectados** a nenhum sinal de repaint e são efetivamente *no-ops*. Resultado: o plugin executa `run_ui` + `tessellate` + paint GL + `swap_buffers` **todo frame** (tipicamente 60 fps com VSync) mesmo ocioso, desperdiçando CPU/GPU enquanto o editor está aberto.
- **Solução técnica:**
  1. Capturar o `full_output.viewport_output[ROOT].repaint_delay` retornado por `egui_ctx.run_ui(...)`.
  2. Manter um acumulador de tempo desde o último paint; **pular** `tessellate`/`paint_and_update_textures`/`swap_buffers` quando: (a) egui pedir `repaint_delay` longo (sem animação pendente), **e** (b) não houver eventos de input acumulados desde o último frame, **e** (c) os medidores não estiverem em decaimento ativo (peak-hold/clip animando).
  3. Garantir que qualquer evento (`on_event`) marca o estado como "dirty" para forçar o próximo `on_frame` a pintar.
  4. Opcional: throttle alvo (~30–45 fps) quando ativo, alinhado à intenção original dos 30 ms.
- **Critérios de aceitação:**
  - Com editor aberto e sinal em silêncio/sem interação, uso de CPU do thread de GUI cai significativamente (medir antes/depois; meta ≥ 60% de redução em idle).
  - Animações (pulso de automação, peak-hold, toasts, "Loading...") permanecem fluidas; nenhum input "engole" frame.
  - Sem flicker ao alternar idle↔ativo.
- **Especialista:** `implementador` + UX (revisão de fluidez).
- **Esforço:** 2 dias.

### Tarefa G3.T02 — Cleanup determinístico de recursos OpenGL no fechamento da janela 🧹⚠️

- **Onde:** `src/clap/gui/window/mod.rs` (ausência de `impl Drop`/hook de fechamento) e `src/clap/extensions/gui.rs:38-43` (`destroy`).
- **Problema:** `NamPluginWindow` cria `egui_glow::Painter`, `vu_program` e `vu_vao` (`window/mod.rs:109-126`) mas **nunca** os libera explicitamente. `egui_glow::Painter` exige `destroy()` com o contexto GL *current* (caso contrário emite warning "Resources will be leaked!"); o `program`/`vao` customizados também ficam sem `delete_program`/`delete_vertex_array`. Embora o teardown do contexto pelo baseview mitigue vazamento de processo, o padrão é incorreto, gera ruído de log e é frágil se o contexto sobreviver à janela.
- **Solução técnica:**
  1. Implementar liberação no ponto de fechamento (hook `on_will_close` do `WindowHandler` se disponível na versão do baseview, ou `impl Drop for NamPluginWindow`) tornando o contexto *current* e chamando `painter.destroy()`, `gl.delete_program(vu_program)`, `gl.delete_vertex_array(vu_vao)`.
  2. Assegurar idempotência (não liberar duas vezes) e robustez FFI (sem panics atravessando a fronteira C — seguir o padrão de early-return já usado em `on_frame`).
- **Critérios de aceitação:**
  - Abrir/fechar o editor repetidamente não emite warnings de leak do `egui_glow` nem acumula recursos GL (validar com log/apitrace em sessão de stress de open/close).
  - Sem UB/panic no teardown (cobertura em `gui/window/test.rs`).
- **Especialista:** `implementador` + revisão `revisor-auditor`.
- **Esforço:** 1 dia.

### Tarefa G3.T03 — HiDPI: obter o scale factor inicial correto ✨

- **Onde:** `src/clap/gui/window/mod.rs:148-151` (TODO já documentado; `scale = 1.0` fixo até o primeiro `Resized`).
- **Problema:** O fator de escala inicia em `1.0` e só é corrigido no primeiro `WindowEvent::Resized`. Em telas HiDPI, os primeiros frames renderizam em escala errada (UI pequena/borrada até o resize).
- **Solução técnica:**
  1. Obter o scale via `host` (CLAP `gui.set_scale` já é recebido em `extensions/gui.rs:46-48` — propagar esse valor ao `NamPluginWindow` em vez de descartá-lo).
  2. Encadear `set_scale` → estado compartilhado/curto-circuito para o window handler aplicar `native_pixels_per_point` antes do primeiro paint.
  3. Manter o ajuste via `Resized` como reforço.
- **Critérios de aceitação:**
  - Em host HiDPI (scale 1.5/2.0), o primeiro frame já renderiza na escala correta.
  - Sem regressão em telas 1.0x.
- **Especialista:** `implementador`.
- **Esforço:** 1 dia.

### Tarefa G3.T04 — Reduzir alocações e medições por frame na GUI 🧹

- **Onde:** `src/clap/gui/ui/knob.rs:148-165` (`points_buf[..n].to_vec()` por frame, 2× ao arrastar); `src/clap/gui/ui/mod.rs:760-774` e `:976-987` (`layout_no_wrap` + `text.clone()` por frame para auto-ajuste de fonte da status bar e metadata).
- **Problema:** A cada frame a GUI: (a) aloca um `Vec<Pos2>` por knob (até 6 allocs/frame) para `egui::Shape::line`; (b) refaz o *layout* de 8 strings de telemetria + separador (com `String::clone`) só para medir larguras e calcular o tamanho de fonte; (c) idem para metadata. Não é RT, mas pressiona o alocador e a CPU continuamente (ainda mais relevante após G3.T01, que torna idle barato).
- **Solução técnica:**
  1. **Knob:** reaproveitar buffers persistentes em `UiState` (ou usar API de path do egui que aceite slice/iterador sem `to_vec`), evitando alocação por frame.
  2. **Status/metadata:** memoizar o tamanho de fonte calculado e as larguras dos galleys, recomputando apenas quando o texto (já cacheado em `status_strings`/`metadata_display`) ou a largura disponível mudarem. Já existe cache de strings em `update_telemetry_state` (1 Hz) — estender o cache ao resultado de medição.
- **Critérios de aceitação:**
  - Zero alocações de `Vec` por frame nos knobs (verificável via heap profiler no thread de GUI).
  - Cálculo de fonte/medição da status bar e metadata ocorre apenas em mudança de conteúdo/tamanho, não por frame.
  - Aparência idêntica (pixel-diff dentro de tolerância de antialiasing).
- **Especialista:** `implementador`.
- **Esforço:** 1 dia.
