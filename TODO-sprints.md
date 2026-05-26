<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Dívida Técnica X11 (utils/mod-update.sh)

---

## 🚀 Épico 1: Consolidação da Interface Gráfica via X11 Puro

**Objetivo:** Estabelecer uma fundação gráfica extremamente robusta e simplificada rodando estritamente sob X11 (e sob Wayland via compatibilidade estável do XWayland), permitindo atualizar a stack de dependências (`egui v0.34`, `glow v0.17`) e sanar o débito técnico acumulado de janelas.

- [x] **Tarefa 1.1: Atualização de Dependências e Higiene do `Cargo.toml`**
  - **Descrição:** Elevação coordenada do `egui` e `egui_glow` para a versão `0.34.0`, e do `glow` para a versão `0.17.0`. Saneamento de dependências e eliminação de flags remanescentes dos experimentos in-process de Wayland Nativo.
  - **Ações Executadas:**
    - Atualizados os crates `egui`, `egui_glow` (v0.34) e `glow` (v0.17) nas dependências opcionais.
    - Configurada a feature flag `clap-plugin` em [Cargo.toml](file:///home/fabio/nam-rs/Cargo.toml) para agrupar as dependências gráficas corretas (`egui`, `egui_glow`, `glow`, `baseview`, `rfd`, `keyboard-types` e a flag `baseview/opengl`).
    - Isoladas dependências específicas do PipeWire no escopo da feature `standalone`.
  - **Validação de Não-Regressão:** Executar `cargo check --no-default-features --features clap-plugin --lib` para certificar que o build compila sem arrastar dependências do PipeWire.

- [x] **Tarefa 1.2: Mapeamento de Eventos e Refatoração do `egui v0.34`**
  - **Descrição:** Adaptação da camada de eventos e renderização do plugin em [src/clap/gui/window.rs](file:///home/fabio/nam-rs/src/clap/gui/window.rs) para a nova API de viewports e eventos do `egui v0.34`.
  - **Ações Executadas:**
    - Substituído o método depreciado `Context::run` pelo novo fluxo idiomático `Context::run_ui` em `NamPluginWindow::on_frame`.
    - Remapeada a detecção e o redimensionamento do `screen_rect` no `RawInput` utilizando o fator de escala (`scale`) físico/lógico da viewport raiz.
    - Corrigido o evento de scroll do mouse (`MouseWheel`) em `on_event` para incluir o campo obrigatório `phase: egui::TouchPhase::Move`.
    - Substituído o uso de `raw_scroll_delta` por `smooth_scroll_delta` no arquivo de desenho da interface [src/clap/gui/ui.rs](file:///home/fabio/nam-rs/src/clap/gui/ui.rs#L257).
    - Removida a necessidade de silenciar deprecations em arquivos legados e de teste, mantendo os logs de compilação 100% limpos e livres de warnings.
  - **Validação de Não-Regressão:** Verificar se comandos de rolagem do mouse e redimensionamento de viewport respondem sem lag ou travamentos.

- [x] **Tarefa 1.3: Integração do Raw Window Handle e X11 Puro**
  - **Descrição:** Configuração da ponte de conversão do handle de janela X11 (`CLAP_WINDOW_API_X11`) exposto na versão `0.5` pelo host para o formato esperado na versão `0.6` pelo `egui` e `baseview` em [src/clap/extensions/gui.rs](file:///home/fabio/nam-rs/src/clap/extensions/gui.rs).
  - **Ações Executadas:**
    - Restrito o suporte de API gráfica em `PluginGuiImpl::is_api_supported` e `get_preferred_api` para aceitar unicamente `GuiApiType::X11` e rejeitar janelas flutuantes (`is_floating = false`).
    - Configurada a flag `raw-window-handle_05` no crate `clack-extensions` para permitir a recepção nativa do descritor de janela pelo host.
    - Implementada a destruição limpa da janela no encerramento (`PluginGuiImpl::destroy`) chamando `window_handle.close()`.
  - **Validação de Não-Regressão:** Verificar se a janela de renderização é liberada sem vazamentos ou falhas de segmentação no host DAW durante a remoção ou desativação do plugin.

- [x] **Tarefa 1.4: Suíte de Validação Final e Proteção contra Regressões**
  - **Descrição:** Executar o pipeline de testes locais e scripts de linting do NAM-rs para validar a corretude e garantir que nenhuma regressão de estilo ou de lógica de concorrência foi introduzida.
  - **Ações Executadas:**
    - Executado o script `./utils/lints.sh` com sucesso, assegurando formatação (`cargo fmt`) correta, checagens de compilação sem warnings e clippy limpo (`-D warnings`).
    - Executados os testes unitários, testes de integração, auditoria estrutural do plugin e ciclo de vida do CLAP via `./utils/tests-cargo.sh`, todos passando com 100% de sucesso.
    - Executada a validação formal do plugin CLAP via `clap-validator`, confirmando a conformidade com a especificação com 16 testes aprovados e 0 falhas.
    - Executados os benchmarks de performance via `cargo bench` para proteção contra regressões no hot-path DSP.
  - **Validação de Não-Regressão:** O script `./utils/lints.sh` e o pipeline `./utils/tests-cargo.sh` passaram com status de sucesso absoluto.

---

## 🔧 Épico 1.5: Refinamentos Pós-Auditoria do Subsistema Gráfico

**Objetivo:** Sanar os pontos identificados pela auditoria pós-sprint do Épico 1. Zero dívida técnica no subsistema de GUI antes de avançar para o Épico 2.

- [x] **Tarefa 1.5.1: Cache de `UniformLocation` do VU Meter Shader**
  - **Descrição:** `get_uniform_location` é chamado **10 vezes por frame** (5 queries × 2 medidores L+R) dentro do `CallbackFn` de renderização do VU meter em [src/clap/gui/ui.rs](file:///home/fabio/nam-rs/src/clap/gui/ui.rs). Em OpenGL, essa é uma query de estado do driver que pode causar pipeline stalls. As locations são constantes para um programa compilado e devem ser consultadas **uma única vez, na inicialização do shader**, sendo reutilizadas por todos os frames subsequentes.
  - **Passo a Passo:**
    - Criar struct `VuUniforms { loc_viewport, loc_meter_rect, loc_peak_frac, loc_hold_frac, loc_hold_color_type }` encapsulando os `glow::UniformLocation`.
    - Adicionar campo `vu_l_uniforms: Option<VuUniforms>` e `vu_r_uniforms: Option<VuUniforms>` ao `UiState` (ou um único `vu_uniforms: Option<VuUniforms>` compartilhado, já que o program é o mesmo para ambos os medidores).
    - No `CallbackFn`, popular o `VuUniforms` via `get_or_insert_with` na **primeira execução** após a compilação do programa, usando `gl.get_uniform_location`.
    - Substituir as 5 chamadas `get_uniform_location` por leituras diretas das locations em cache.
  - **Validação de Não-Regressão:** `cargo test --no-default-features --features clap-plugin` e inspecionar visualmente o VU meter na DAW.

- [x] **Tarefa 1.5.2: Eliminar `transmute` Desnecessário de Lifetime em `window.rs`**
  - **Descrição:** Em `NamPluginWindow::on_frame` ([src/clap/gui/window.rs:346](file:///home/fabio/nam-rs/src/clap/gui/window.rs#L346)), um `std::mem::transmute` eleva o lifetime de `&self.host` para `'static` apenas para passar a referência a `draw_ui`. Este transmute é desnecessário e mascara o modelo de propriedade: `self.host` já vive o tempo suficiente para a chamada. Além disso, este transmute é diferente e menos justificado do que o `transmute` em `gui.rs` (que genuinamente precisa cruzar thread boundary). O unsafe desnecessário deve ser eliminado.
  - **Passo a Passo:**
    - Verificar a assinatura de `draw_ui` em `ui.rs`: aceita `&HostSharedHandle<'_>` ou `&HostSharedHandle<'static>`?
    - Se `'static` for necessário por limitação do `egui_glow` ou `clack`, adicionar `// SAFETY:` com justificativa clara. Caso contrário, remover o transmute e passar `&self.host` diretamente.
    - Garantir que o compilador valide sem unsafe adicional.
  - **Validação de Não-Regressão:** `cargo check --no-default-features --features clap-plugin` e `cargo clippy`.

- [x] **Tarefa 1.5.3: Centralizar Constantes de Dimensão da GUI**
  - **Descrição:** Os valores `600` (largura) e `260` (altura) da janela do plugin estão hardcoded em literais espalhados em [gui.rs](file:///home/fabio/nam-rs/src/clap/extensions/gui.rs) (×3 ocorrências) e [window.rs](file:///home/fabio/nam-rs/src/clap/gui/window.rs) (×2 ocorrências). Uma divergência silenciosa entre esses valores pode causar janelas com tamanho inconsistente ou comportamento inesperado nos hosts CLAP.
  - **Passo a Passo:**
    - Adicionar `pub const GUI_WIDTH: u32 = 600;` e `pub const GUI_HEIGHT: u32 = 260;` em [src/clap/gui/mod.rs](file:///home/fabio/nam-rs/src/clap/gui/mod.rs).
    - Substituir todos os literais em `gui.rs` e `window.rs` pelos constantes importados.
  - **Validação de Não-Regressão:** `cargo check --all-features` e `./utils/lints.sh`.

- [x] **Tarefa 1.5.4: Correção do Scale HiDPI na Inicialização da Janela**
  - **Descrição:** Em `NamPluginWindow::new` ([window.rs:287](file:///home/fabio/nam-rs/src/clap/gui/window.rs#L287)), o fator de escala é inicializado como `1.0f32` hardcoded, enquanto `baseview` é configurado com `WindowScalePolicy::SystemScaleFactor`. Em monitores HiDPI (escala ≠ 1.0), o primeiro frame pode ser renderizado com `pixels_per_point` incorreto, causando UI desfocada ou com dimensionamento errado até que o host envie o primeiro evento `Resized`.
  - **Passo a Passo:**
    - Verificar se `baseview::Window` expõe `scale_factor()` ou método equivalente acessível no construtor.
    - Substituir `let scale = 1.0f32` por `let scale = window.scale_factor() as f32` (ou equivalente `baseview`).
    - Se `baseview` não expuser a escala no construtor, documentar a limitação com `// TODO(HiDPI)` e um issue de acompanhamento.
  - **Validação de Não-Regressão:** `cargo check --no-default-features --features clap-plugin` e validar visualmente em ambiente HiDPI se disponível.

- [x] **Tarefa 1.5.5: Constante Nomeada para Fator de Scroll (Opcional/Cosmético)**
  - **Descrição:** O multiplicador `10.0` em `ScrollDelta::Lines` ([window.rs:426](file:///home/fabio/nam-rs/src/clap/gui/window.rs#L426)) é uma heurística sem documentação. Deve ser extraído como constante nomeada para deixar claro o racional de conversão de linhas para pontos.
  - **Passo a Passo:**
    - Adicionar `const SCROLL_LINES_TO_POINTS: f32 = 10.0; // ~10 pixels/linha (heurística baseview→egui)` no topo de `window.rs`.
    - Substituir o literal `10.0` pelo constante.
  - **Validação de Não-Regressão:** `./utils/lints.sh`.

- [x] **Auditoria Pós-Sprint (Revisor-Auditor):** Revisão sistemática de todos os arquivos do Épico 1.5 para garantir zero dívida técnica.
  - **Achados e Correções:**
    - **`gui.rs` (CRÍTICO → CORRIGIDO):** `transmute` de lifetime sem comentário `// SAFETY:` — adicionado comentário completo documentando o invariante: handle do host tem lifetime real da instância do plugin, garantido válido durante toda a execução da janela graças ao `destroy()`.
    - **`window.rs` (MÉDIO → CORRIGIDO):** `.unwrap()` sem fallback em `chars().next()` — substituído por `.unwrap_or('\0')` com comentário explicativo. A guarda `len() == 1` torna o fallback inalcançável em produção, mas o código agora expressa isso explicitamente.
    - **Unsafe residuais auditados:** 3 blocos `unsafe` legítimos permanecem (`transmute` de lifetime em `gui.rs`, `make_current()`/`make_not_current()` da API GL, e deref do raw pointer CLAP shared), todos com justificativa clara.
  - **Validação Final:** `cargo test` (100% passando) e `./utils/lints.sh` (limpo, sem warnings).

---

## 🚀 Épico 2: Integração de Alta Fidelidade com Bitwig Studio (CLAP)

**Objetivo:** Elevar o nível de integração do NAM-rs como cidadão de primeira classe no Bitwig Studio, implementando recursos modernos da especificação CLAP que otimizam a automação, modulação e latência sem introduzir incompatibilidades em DAWs simplificadas.

- [ ] **Tarefa 2.1: Suporte a Eventos de Modulação Monofônica (CLAP Parameter Modulation)**
  - **Descrição:** Habilitar a recepção e aplicação de offsets de modulação em tempo real (`CLAP_EVENT_PARAM_MOD`) na thread de áudio. Permite que moduladores nativos do Bitwig (LFOs, Envelopes) modulem parâmetros do NAM-rs (como os ganhos e o gate) sem alterar seus valores base de automação.
  - **Passo a Passo:**
    - Configurar a flag `CLAP_PARAM_IS_MODULATABLE` nas definições de parâmetros em [src/clap/extensions/params.rs](file:///home/fabio/nam-rs/src/clap/extensions/params.rs).
    - No loop de eventos do processador em [src/clap/processor.rs](file:///home/fabio/nam-rs/src/clap/processor.rs), interceptar eventos de modulação e aplicar o offset aditivo correspondente ao cálculo do ganho ou threshold.
    - Garantir que o valor modulado seja suavizado corretamente pelo `ParamSmoother` para eliminar qualquer zipper noise (cliques analógicos) durante modulações rápidas.
  - **Validação de Não-Regressão:** Executar a suíte de testes do processador. Modular o Input Gain no Bitwig Studio usando um LFO e certificar que o áudio seja modulado com suavidade e sem artefatos sonoros.

- [ ] **Tarefa 2.2: Compensação Dinâmica de Latência (Dynamic Latency PDC Sync)**
  - **Descrição:** Reportar à DAW atualizações de latência em tempo real caso ocorra mudança de amostragem no projeto ou swap de modelos com diferentes latências internas de filtragem no resampler, forçando o Bitwig a atualizar o PDC (Plugin Delay Compensation) imediatamente.
  - **Passo a Passo:**
    - Na audio thread, monitorar se a latência efetiva calculada pelo resampler diverge da latência atômica reportada ao host.
    - Sinalizar atomicamente a alteração de latência para a thread principal (main thread).
    - Na thread principal, invocar o callback `host.request_restart(CLAP_PLUGIN_RESTART_LATENCY)` para que o host solicite a nova latência via extensão correspondente.
  - **Validação de Não-Regressão:** Mudar a taxa de amostragem do Bitwig durante a execução e certificar-se de que a DAW ajusta a compensação de delay do canal do plugin sem desfasamentos ou silêncio temporário.

- [ ] **Tarefa 2.3: Atualização Visual da UI e Exposição de Metadados de Modelo**
  - **Descrição:** Expor de forma simplificada o nome do modelo carregado ativamente para a DAW, facilitando a legibilidade do projeto no painel de dispositivos (Device Panel) do Bitwig.
  - **Passo a Passo:**
    - Adicionar um parâmetro somente leitura (ou atualizar metadados do estado) indicando o nome do arquivo do modelo NAM.
    - Configurar o layout e as páginas de Remote Controls para exibir a string abreviada do modelo.
  - **Validação de Não-Regressão:** Verificar se o `clap-validator` valida o plugin com sucesso e inspecionar se a barra de controle remoto do Bitwig exibe o nome do modelo ativado corretamente.
