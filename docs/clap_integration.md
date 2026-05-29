<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Estratégia de Integração CLAP (Clever Audio Plug-in)

Este documento descreve a arquitetura e a estratégia para transformar o motor DSP do NAM-rs em um plugin de áudio compatível com o padrão CLAP.

## 1. Thread Model

A integração CLAP deve respeitar estritamente a segregação de threads já existente no NAM-rs, mapeando-as para o modelo do host (DAW):

- **Main Thread (Host)**:
  - Responsável pela inicialização do plugin, escaneamento de parâmetros e gerenciamento de estado.
  - No NAM-rs, esta thread substituirá o loop principal em [src/main.rs](file:///home/fabio/nam-rs/src/main.rs).
  - Gerencia o carregamento de arquivos `.nam`/`.namb` via [src/loader/](file:///home/fabio/nam-rs/src/loader/).
- **Audio Thread (Real-time)**:
  - Chamada pelo host via callback `process()`.
  - **Requisito Crítico**: Deve manter a política de **ZERO alocações** e **ZERO locks**.
  - Utilizará [src/dsp/pipeline.rs](file:///home/fabio/nam-rs/src/dsp/pipeline.rs) para o processamento, adaptando os buffers do CLAP para o formato interno.
  - Diferente do PipeWire (que é dual-stream), o CLAP fornece buffers de entrada e saída em um único contexto, eliminando a necessidade do `DspBridge`.

## 2. Mapeamento de Parâmetros

Os parâmetros expostos ao host serão mapeados a partir da estrutura `NamPluginParams` (veja [src/common/params.rs](file:///home/fabio/nam-rs/src/common/params.rs)):

| Parâmetro CLAP     | ID                  | Unidade | Descrição                                        |
|:------------------ |:------------------- |:------- |:------------------------------------------------ |
| **Input Gain**     | `input_gain_db`     | dB      | Ganho aplicado antes da inferência neural.       |
| **Output Gain**    | `output_gain_db`    | dB      | Ganho aplicado após a inferência neural.         |
| **Gate Threshold** | `gate_threshold_db` | dB      | Limiar de abertura do Noise Gate.                |
| **Bypass**         | `bypass`            | binário | Desativa o processamento neural (Dry/Wet 0/100). |
| **Active Model**   | `active_model`      | —       | Nome do modelo carregado (somente leitura).      |

O caminho do modelo (`model_path`) será tratado como um **State Property**, permitindo que a DAW salve e carregue o modelo correto no projeto.

## 3. Estratégia de Compilação

O projeto utiliza *feature flags* para permitir múltiplos alvos de build:

- `cargo build --features standalone`: Binário executável com backend PipeWire (padrão).
- `cargo build --no-default-features --features clap-plugin --lib`: Biblioteca dinâmica (`.clap`) com GUI completa.

A feature `clap-plugin` omitirá os módulos `pw_host.rs` e `rt_setup.rs`, mantendo o binário final sem dependências de PipeWire.

## 4. Framework: `clack-plugin`

- **Motivo**: Oferece controle granular sobre a implementação sem adicionar overhead desnecessário, permitindo uma integração direta com as estruturas RT-safe do NAM-rs. Ao contrário de frameworks mais opinativos, o `clack` mapeia-se quase 1:1 ao spec CLAP enquanto provê segurança de tipos em Rust.
- **Frameworks de alto nível**: Descartados por adicionarem suporte forçado a VST3, uma camada de GUI embutida que conflita com nossa escolha de `egui` puro, e abstrações que poderiam mascarar o determinismo temporal exigido pelo motor DSP do NAM-rs.
- **Link**: [https://github.com/prokopyl/clack](https://github.com/prokopyl/clack)

## 5. Extensões CLAP Implementadas

A integração utiliza o crate `clack-extensions` para implementar as seguintes extensões do protocolo CLAP:

| Extensão                       | Arquivo                                                                                  | Finalidade                                                                                                              |
|:------------------------------ |:---------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------- |
| `clap_plugin_audio_ports`      | [audio_ports.rs](file:///home/fabio/nam-rs/src/clap/extensions/audio_ports.rs)           | Declaração explícita de portas de entrada/saída mono e suporte a processamento in-place                                 |
| `clap_plugin_params`           | [params.rs](file:///home/fabio/nam-rs/src/clap/extensions/params.rs)                     | Mapeamento e automação de parâmetros (`input_gain`, `output_gain`, `gate`, `bypass`) com suporte a gesture e `flush()`  |
| `clap_plugin_state`            | [state.rs](file:///home/fabio/nam-rs/src/clap/extensions/state.rs)                       | Persistência do estado do plugin (parâmetros e caminho do modelo) no projeto da DAW                                     |
| `clap_plugin_latency`          | [latency.rs](file:///home/fabio/nam-rs/src/clap/extensions/latency.rs)                   | Reporte dinâmico de latência induzida pelo processamento e resampling ao host                                           |
| `clap_plugin_track_info`       | [track_info.rs](file:///home/fabio/nam-rs/src/clap/extensions/track_info.rs)             | Suporte à cor da track do host para adaptar dinamicamente o accent color da GUI                                         |
| `clap_plugin_remote_controls`  | [remote_controls.rs](file:///home/fabio/nam-rs/src/clap/extensions/remote_controls.rs)   | Páginas de controle pré-configuradas ("Main" e "Gate") para integração com controladores de hardware e Device Panel     |
| `clap_plugin_param_indication` | [param_indication.rs](file:///home/fabio/nam-rs/src/clap/extensions/param_indication.rs) | Feedback visual na GUI para indicar parâmetros mapeados, automatizados ou sob override temporário                       |
| `clap_plugin_gui`              | [gui.rs](file:///home/fabio/nam-rs/src/clap/extensions/gui.rs)                           | Interface gráfica nativa baseada em `egui` v0.34 embutida via `baseview` e backend X11/XWayland (`CLAP_WINDOW_API_X11`) |

## 6. Plugin Descriptor

O descritor de metadados do plugin seguirá o seguinte padrão:

- **Plugin ID**: `br.eti.fabiolima.nam-rs`
- **Nome**: `NAM-rs`
- **Vendor**: `Fabio Lima`
- **URL**: [https://github.com/fabiohl/nam-rs](https://github.com/fabiohl/nam-rs)
- **Features**: `["audio-effect", "distortion", "gate", "simulator", "mono"]`

> [!NOTE]
> O padrão NAM é, por definição, mono. O plugin CLAP funciona estritamente como mono (mono-in/mono-out) para se alinhar aos workflows tradicionais das DAWs, onde o roteamento de canais é gerenciado externamente pelo host. Já no executável Standalone/Pipewire, o processamento estéreo é fornecido como uma conveniência para sinais estéreo nativos.

## 7. DAWs Alvo de Validação

- **Bitwig Studio**: Plataforma de referência absoluta para conformidade CLAP (co-autora do padrão). Essencial para validar o comportamento de sandboxing e automação sample-accurate.
- **REAPER**: Validação de compatibilidade com hosts de baixo custo e testes de buffers irregulares.
  - NOTA: *Descartado* por estar buggy na minha máquina ubuntu linux.
- **Fender Studio Pro**: Objetivo futuro por exigir modo wayland nativo.
- **CLAP-info / CLAP-host**: Ferramentas de linha de comando para validação técnica rigorosa do spec.

## 8. Interface Gráfica: Estratégia de Windowing e Stack

A GUI do plugin CLAP opera em uma thread dedicada (`UI thread`), completamente isolada da `audio thread`. A arquitetura é unificada no backend X11.

### Estratégia de Windowing Unificada (X11 Puro)

```text
┌────────────────────────────────────────────────┐
│                  NAM-rs GUI                    │
│              (egui + egui_glow)                │
│    draw_ui() — Lógica de UI 100% agnóstica    │
├────────────────────────────────────────────────┤
│       NamPluginWindow (WindowHandler)          │
│   Tradução baseview events → egui::RawInput   │
│   Renderização via egui_glow::Painter + glow  │
├────────────────────────────────────────────────┤
│                  Backend X11                   │
│   (baseview - raw-window-handle 0.5 → 0.6)    │
│           X11 Puro / XWayland nativo           │
└────────────────────────────────────────────────┘
```

- **Backend X11:** O plugin declara suporte exclusivo a `CLAP_WINDOW_API_X11`.
- **Stack:** `egui v0.34` + `glow v0.17`, com tradução de handles de janela (`raw-window-handle 0.5` do host para `0.6` do `egui`/`baseview`).
- **Implementação:** `NamPluginWindow` implementa `baseview::WindowHandler`, traduzindo eventos para `egui::RawInput` sem camada intermediária.

### Stack Tecnológico

| Componente    | Crate/Tecnologia | Papel                                                                                          |
|:------------- |:---------------- |:---------------------------------------------------------------------------------------------- |
| GUI Framework | `egui`           | Immediate Mode GUI — sem estado persistente, sem GC, sem alocações no render loop              |
| Renderizador  | `egui_glow`      | Bridge egui → OpenGL 3.3 via `glow`. Integração manual (sem `egui-baseview`, abandonado ~2021) |
| Windowing     | `baseview`       | Janela nativa embutida X11 via `RawWindowHandle`. Event loop dedicado                          |
| File Picker   | `rfd`            | File dialog nativo assíncrono (zenity/xdg-portal). Nunca bloqueia a UI thread                  |

Todo código de GUI vive em `src/clap/gui/` e é gateado por `#[cfg(feature = "clap-plugin")]`.

### Isolamento de Threads (UI ↔ Audio)

A UI thread **nunca** acessa diretamente os campos de `NamClapProcessor`. A comunicação é estritamente via:

- **Leitura de telemetria (Audio → UI):** Campos atômicos em `NamClapShared` (`AtomicU32` para peaks, `AtomicBool` para clipping), lidos com `Ordering::Relaxed`.
- **Envio de comandos (UI → Audio):** Canal SPSC de parâmetros (`ClapParamPayload`) via `param_tx`, drenado no início de cada `process()`.
- **Metadados (Main → UI):** `Mutex<String>` para nome do modelo — acessado pela UI thread em intervalos de 500ms.
