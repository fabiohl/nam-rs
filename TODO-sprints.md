<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Dívida Técnica X11 (utils/mod-update.sh)

---

## 🚀 Épico 1: Consolidação da Interface Gráfica via X11 Puro

**Objetivo:** Estabelecer uma fundação gráfica extremamente robusta e simplificada rodando estritamente sob X11 (e sob Wayland via compatibilidade estável do XWayland), permitindo atualizar a stack de dependências (`egui v0.34`, `glow v0.17`) e sanar o débito técnico acumulado de janelas.

- [ ] **Tarefa 1.1: Atualização de Dependências e Higiene do `Cargo.toml`**
  - **Descrição:** Elevação coordenada do `egui` e `egui_glow` para a versão `0.34.0`, e do `glow` para a versão `0.17.0`. Saneamento de dependências e eliminação de flags remanescentes dos experimentos in-process de Wayland Nativo.
  - **Ações Executadas:**
    - Atualizados os crates `egui`, `egui_glow` (v0.34) e `glow` (v0.17) nas dependências opcionais.
    - Configurada a feature flag `clap-plugin` em [Cargo.toml](file:///home/fabio/nam-rs/Cargo.toml) para agrupar as dependências gráficas corretas (`egui`, `egui_glow`, `glow`, `baseview`, `rfd`, `keyboard-types` e a flag `baseview/opengl`).
    - Isoladas dependências específicas do PipeWire no escopo da feature `standalone`.
  - **Validação de Não-Regressão:** Executar `cargo check --no-default-features --features clap-plugin --lib` para certificar que o build compila sem arrastar dependências do PipeWire.

- [ ] **Tarefa 1.2: Mapeamento de Eventos e Refatoração do `egui v0.34`**
  - **Descrição:** Adaptação da camada de eventos e renderização do plugin em [src/clap/gui/window.rs](file:///home/fabio/nam-rs/src/clap/gui/window.rs) para a nova API de viewports e eventos do `egui v0.34`.
  - **Ações Executadas:**
    - Substituído o método depreciado `Context::run` pelo novo fluxo idiomático `Context::run_ui` em `NamPluginWindow::on_frame`.
    - Remapeada a detecção e o redimensionamento do `screen_rect` no `RawInput` utilizando o fator de escala (`scale`) físico/lógico da viewport raiz.
    - Corrigido o evento de scroll do mouse (`MouseWheel`) em `on_event` para incluir o campo obrigatório `phase: egui::TouchPhase::Move`.
    - Substituído o uso de `raw_scroll_delta` por `smooth_scroll_delta` no arquivo de desenho da interface [src/clap/gui/ui.rs](file:///home/fabio/nam-rs/src/clap/gui/ui.rs#L257).
    - Adicionada a diretiva local `#![allow(deprecated)]` para silenciar de forma cirúrgica avisos de APIs em transição em arquivos legados e de teste, mantendo os logs de compilação limpos.
  - **Validação de Não-Regressão:** Verificar se comandos de rolagem do mouse e redimensionamento de viewport respondem sem lag ou travamentos.

- [ ] **Tarefa 1.3: Integração do Raw Window Handle e X11 Puro**
  - **Descrição:** Configuração da ponte de conversão do handle de janela X11 (`CLAP_WINDOW_API_X11`) exposto na versão `0.5` pelo host para o formato esperado na versão `0.6` pelo `egui` e `baseview` em [src/clap/extensions/gui.rs](file:///home/fabio/nam-rs/src/clap/extensions/gui.rs).
  - **Ações Executadas:**
    - Restrito o suporte de API gráfica em `PluginGuiImpl::is_api_supported` e `get_preferred_api` para aceitar unicamente `GuiApiType::X11` e rejeitar janelas flutuantes (`is_floating = false`).
    - Configurada a flag `raw-window-handle_05` no crate `clack-extensions` para permitir a recepção nativa do descritor de janela pelo host.
    - Implementada a destruição limpa da janela no encerramento (`PluginGuiImpl::destroy`) chamando `window_handle.close()`.
  - **Validação de Não-Regressão:** Verificar se a janela de renderização é liberada sem vazamentos ou falhas de segmentação no host DAW durante a remoção ou desativação do plugin.

- [ ] **Tarefa 1.4: Suíte de Validação Final e Proteção contra Regressões**
  - **Descrição:** Executar o pipeline de testes locais e scripts de linting do NAM-rs para validar a corretude e garantir que nenhuma regressão de estilo ou de lógica de concorrência foi introduzida.
  - **Ações Executadas:**
    - Rodar o formatador e clippy via `./utils/lints.sh` (garantindo zero erros e warnings no compilador).
    - Executar os testes unitários e de integração (`./utils/tests-cargo.sh`).
    - Validar o binário gerado através do `clap-validator` para assegurar conformidade com a especificação CLAP.
  - **Validação de Não-Regressão:** O script `./utils/lints.sh` deve passar com status de sucesso absoluto.

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
