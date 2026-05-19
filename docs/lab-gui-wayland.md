<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# 🔬 Laboratório: GUI Wayland Nativo para NAM-rs

> **Documento mestre** para acompanhamento da evolução do suporte Wayland no ecossistema de plugins de áudio (CLAP, VST3, DAWs, frameworks) e planejamento técnico dos backends Wayland futuros do NAM-rs.
> **Escopo:** Exclusivamente Wayland nativo (H2 floating e H3 embedded). O backend X11/XWayland (v1.0) é tratado no `TODO-sprints.md` Sprint 4.
> **Decisões arquiteturais:** Documentadas em `docs/architecture.md` §8.1 "Interface Gráfica".

---

## 1. Radar do Ecossistema Wayland

Monitorar a evolução destes indicadores é essencial para saber **quando** investir em backends Wayland nativos.

### 1.1 Estado das DAWs (Última Atualização: 2026-05-19)

| DAW               | Versão | CLAP GUI         | Wayland Nativo                | Nested Compositor  | Notas                                  |
| ----------------- | ------ | ---------------- | ----------------------------- | ------------------ | -------------------------------------- |
| **Bitwig Studio** | 6.0.6  | X11 via XWayland | ❌ Não                        | ❌ Não             | Co-autor do CLAP. Acompanhar releases. |
| **Studio One**    | 7.2+   | X11 via XWayland | ✅ VST3 only (`IWaylandHost`) | ✅ VST3 only       | Contribuiu `IWaylandHost` ao VST3 SDK. |
| **REAPER**        | 7.x    | X11 via XWayland | ❌ Não                        | ❌ Não             | Linux é plataforma secundária.         |
| **Ardour**        | 8.x    | N/A (LV2)        | ❌ Não                        | ❌ Não             | Foco em LV2, não CLAP.                 |
| **Carla**         | 2.6+   | X11 via XWayland | ❌ Não                        | ❌ Não             | Host genérico; depende dos plugins.    |
| **Qtractor**      | 1.x    | N/A              | ❌ Não                        | ❌ Não             | LV2 focus.                             |

### 1.2 Estado dos Padrões

| Padrão / Spec                     | Status                 | Indicador para NAM-rs                                         |
| --------------------------------- | ---------------------- | ------------------------------------------------------------- |
| **CLAP `gui.h` Wayland**          | "embed not supported"  | 🔴 Sem embedding. Apenas floating.                            |
| **CLAP Issue 474**                | Aberta (190+ comments) | 🟡 Discussão ativa, sem consenso. Acompanhar.                 |
| **VST3 SDK 3.8.0 `IWaylandHost`** | Preview (MIT)          | 🟢 Precedente técnico. Modelo a seguir quando CLAP adotar.    |
| **`xdg-foreign-unstable-v2`**     | Instável (prefixo `z`) | 🟡 Funcional em GNOME/KDE/Sway. Pode ser promovido a estável. |
| **`wl_subsurface`**               | Estável (core Wayland) | 🟢 Protocolo maduro. Base do H3.                              |

### 1.3 Estado dos Frameworks Rust

| Framework / Crate     | Wayland Nativo  | Notas                                                      |
| --------------------- | --------------- | ---------------------------------------------------------- |
| **baseview**          | ❌ Issue 135    | Aberta desde Nov/2022. Sem atividade. XWayland apenas.     |
| **egui / egui_glow**  | N/A             | Agnóstico — recebe `glow::Context`. Backend não importa.   |
| **winit**             | ✅ (standalone) | Suporta Wayland, mas **não** suporta janela filha (#1161). |
| **wayland-client**    | ✅ 0.31+        | Cliente Wayland puro Rust. Maduro.                         |
| **wayland-protocols** | ✅              | Bindings para xdg-shell, xdg-foreign, etc.                 |
| **wayland-egl**       | ✅              | Bridge `wl_surface` → `wl_egl_window` para EGL/OpenGL.     |
| **smithay**           | ✅              | Compositor Wayland em Rust. Útil para PoC do H3.           |

> [!TIP]
> **Gatilho para ação:** Quando QUALQUER das seguintes condições for verdadeira, reavaliar prioridade dos experimentos:
>
> 1. Bitwig anunciar suporte a nested compositor para CLAP
> 2. CLAP Issue 474 resultar em uma extensão padronizada (`CLAP_EXT_GUI_WAYLAND_HOST` ou equivalente)
> 3. `xdg-foreign` ser promovido a protocolo estável (sem prefixo `z`)
> 4. `baseview` receber PR de Wayland mergeado

---

## 2. Experimento H2 — Floating Window via `xdg-foreign`

### 2.1 Objetivo

Criar uma janela `xdg_toplevel` Wayland nativa que renderiza egui via OpenGL, posicionada como transiente da janela da DAW usando o protocolo `xdg-foreign-unstable-v2`.

**Resultado esperado:** Janela flutuante com decorações do compositor (server-side decorations — SSD). NÃO é embedded. A UX é aceitável mas não premium (a janela pode ser movida, minimizada e ocultada independentemente da DAW).

### 2.2 Arquitetura do Experimento

```text
┌─────────────────────────────────────────────────────────┐
│                    DAW (Bitwig)                         │
│                                                         │
│  ┌─────────────────────────────────────────┐            │
│  │ Janela Principal (xdg_toplevel)         │            │
│  │                                         │            │
│  │  ┌─ Plugin Strip ─────────────────────┐ │            │
│  │  │  NAM-rs (sem GUI embutida)         │ │            │
│  │  └────────────────────────────────────┘ │            │
│  └─────────────────────────────────────────┘            │
│                                                         │
│  export_toplevel() → handle string ─────────────────┐   │
└─────────────────────────────────────────────────────│───┘
                                                      │
┌─────────────────────────────────────────────────────│───┐
│                 NAM-rs Plugin (Floating UI)          │   │
│                                                     ▼   │
│  import_toplevel(handle) → zxdg_imported_v2             │
│  set_parent_of(my_xdg_toplevel)                         │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │ wl_surface → xdg_toplevel (floating, SSD)        │   │
│  │ wl_egl_window → EGLSurface → glow::Context      │   │
│  │ egui_glow::Painter → egui UI                     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Stack de Dependências

```toml
# Cargo.toml (sob feature flag `wayland-floating`)
[dependencies]
wayland-client    = { version = "0.31", optional = true }
wayland-protocols = { version = "0.32", optional = true, features = ["client", "unstable"] }
wayland-egl       = { version = "0.32", optional = true }
khronos-egl       = { version = "6.0",  optional = true, features = ["dynamic"] }
glow              = { version = "0.16", optional = true }
egui              = { version = "0.31", optional = true }
egui_glow         = { version = "0.31", optional = true }
```

### 2.4 Implementação Passo-a-Passo

#### Passo 1 — Conexão Wayland e Registry

```rust
use wayland_client::{Connection, Dispatch, QueueHandle, globals::registry_queue_init};

let conn = Connection::connect_to_env()?;
let (globals, mut event_queue) = registry_queue_init::<State>(&conn)?;

// Bind globals necessários:
let compositor: WlCompositor = globals.bind(&qh, 6..=6, ())?;
let xdg_wm_base: XdgWmBase = globals.bind(&qh, 6..=6, ())?;
let subcompositor: WlSubcompositor = globals.bind(&qh, 1..=1, ())?; // Para H3
let exporter: ZxdgExporterV2 = globals.bind(&qh, 1..=1, ())?;       // Host exporta
let importer: ZxdgImporterV2 = globals.bind(&qh, 1..=1, ())?;       // Plugin importa
```

#### Passo 2 — Criação da Superfície e Toplevel

```rust
let surface = compositor.create_surface(&qh, ());
let xdg_surface = xdg_wm_base.get_xdg_surface(&surface, &qh, ());
let toplevel = xdg_surface.get_toplevel(&qh, ());
toplevel.set_title("NAM-rs".into());
toplevel.set_app_id("com.namrs.plugin".into());
surface.commit(); // Commit inicial (sem buffer — handshake)
```

#### Passo 3 — xdg-foreign: Declarar Parentesco Transiente

```rust
// O host (DAW) exporta sua toplevel e passa o handle via CLAP set_transient():
// clap_window_t.ptr → handle string (implementação DAW-specific)

// O plugin importa o handle e se declara filho:
let imported = importer.import_toplevel(handle_string, &qh, ());
imported.set_parent_of(&surface);
```

> [!WARNING]
> **Problema em aberto:** O CLAP `set_transient()` recebe um `clap_window_t`, mas para Wayland o membro `ptr` deveria conter o handle string do `xdg-foreign`. **Não há consenso** sobre como o host deve preencher este campo. Issue #474 do CLAP discute exatamente isso.

#### Passo 4 — EGL + OpenGL sobre Wayland

```rust
use wayland_egl::WlEglSurface;

// Criar wl_egl_window a partir da wl_surface
let egl_window = WlEglSurface::new(surface.id(), width as i32, height as i32);

// Inicializar EGL
let egl_display = egl::get_display(conn.display().id().as_ptr())?;
egl::initialize(egl_display)?;
let config = egl::choose_config(egl_display, &attribs)?;
let egl_context = egl::create_context(egl_display, config, None, &ctx_attribs)?;
let egl_surface = egl::create_window_surface(egl_display, config, egl_window.ptr(), None)?;
egl::make_current(egl_display, egl_surface, egl_surface, egl_context)?;

// Inicializar glow
let gl = unsafe {
    glow::Context::from_loader_function(|s| egl::get_proc_address(s) as *const _)
};

// Inicializar egui_glow painter
let painter = egui_glow::Painter::new(Arc::new(gl), "", None)
    .expect("Failed to create egui_glow painter");
```

#### Passo 5 — Event Loop + egui Render

```rust
let egui_ctx = egui::Context::default();

loop {
    // 1. Dispatch Wayland events
    event_queue.dispatch_pending(&mut state)?;

    // 2. Converter Wayland events → egui::RawInput
    let raw_input = state.drain_events_as_egui_input();

    // 3. Construir UI
    let full_output = egui_ctx.run(raw_input, |ctx| {
        nam_ui::draw(ctx, &shared); // Mesma função de UI do backend X11!
    });

    // 4. Renderizar
    let prims = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
    painter.paint_and_update_textures(
        [width as u32, height as u32],
        full_output.pixels_per_point,
        &prims,
        &full_output.textures_delta,
    );
    egl::swap_buffers(egl_display, egl_surface)?;

    // 5. Commit
    surface.commit();
}
```

### 2.5 Compatibilidade de Compositores

| Compositor         | `xdg-foreign-v2` | `set_parent_of` | Notas                              |
| ------------------ | ---------------- | --------------- | ---------------------------------- |
| **Mutter** (GNOME) | ✅               | ✅              | Referência                         |
| **KWin** (KDE)     | ✅               | ✅              | Referência                         |
| **Sway**           | ✅               | ✅              | wlroots-based                      |
| **Hyprland**       | ✅               | ✅              | wlroots-based                      |
| **Weston**         | ⚠️ Parcial       | ⚠️ Parcial      | Compositor de referência, mínimo   |
| **Mir**            | ❌               | ❌              | Ubuntu/Canonical — sem xdg-foreign |

### 2.6 Critérios de Sucesso

- [ ] Janela Wayland nativa renderiza egui com knobs e medidores
- [ ] `xdg-foreign` funciona no GNOME (Mutter) e KDE (KWin)
- [ ] Janela se posiciona como transiente da DAW (acima, segue foco)
- [ ] Input (mouse, teclado) chega corretamente ao egui
- [ ] DPI scaling funciona (leitura de `wl_output.scale`)
- [ ] Fechar a janela não crasheia o plugin

### 2.7 Riscos e Mitigações

| Risco                                       | Probabilidade | Mitigação                                          |
| ------------------------------------------- | ------------- | -------------------------------------------------- |
| DAW não exporta handle via `set_transient`  | Alta          | Janela floating sem parent (funcional, mas solta)  |
| `xdg-foreign` removido antes de estabilizar | Baixa         | Fallback para janela toplevel sem parent           |
| Scaling fracionado (1.25x, 1.5x)            | Média         | Ler `preferred_buffer_scale` do `wl_output`        |
| Clipboard/IME não funciona                  | Média         | Implementar `zwp_text_input_v3` e `wl_data_device` |

---

## 3. Experimento H3 — Embedded via Nested Compositor (`wl_subsurface`)

### 3.1 Objetivo

Validar que é possível criar uma `wl_subsurface` dentro de uma `wl_surface` pai fornecida por outro processo (simulando o modelo `IWaylandHost` do VST3 SDK 3.8.0). Este é o "padrão ouro" do embedding Wayland nativo.

**Resultado esperado:** A janela do plugin é pixel-perfect embedded dentro da janela da DAW, sem decorações, sem bordas, exatamente como no X11/XEmbed — mas nativo Wayland.

### 3.2 Como Funciona o Modelo IWaylandHost

Documentação de referência: [VST3 SDK 3.8.0 IWaylandHost](https://steinbergmedia.github.io/vst3_dev_portal/pages//Technical+Documentation/Change+History/3.8.0/IWaylandHost.html)

```text
┌──────────────────────────────────────────────────────────────┐
│                 DAW (Host = Nested Compositor)               │
│                                                              │
│  wl_display (sistema) ← conecta ao compositor de sistema     │
│       │                                                      │
│       ├─ wl_surface (janela principal da DAW)                │
│       │       │                                              │
│       │       ├─ wl_subsurface (plugin strip UI)             │
│       │       │       │                                      │
│       │       │       └─ wl_surface (plugin A) ← AQUI        │
│       │       │                                              │
│       │       └─ wl_subsurface (outro conteúdo)              │
│       │                                                      │
│  Socket Unix privado ← DAW expõe como "mini compositor"      │
│       │                                                      │
│       └─ Plugin conecta aqui (NÃO ao compositor de sistema)  │
│           via IWaylandHost::openWaylandConnection()          │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                  Plugin (Cliente Wayland)                    │
│                                                              │
│  1. IWaylandHost::openWaylandConnection() → fd               │
│  2. wl_display_connect_to_fd(fd)                             │
│  3. wl_compositor::create_surface() → child_surface          │
│  4. wl_subcompositor::get_subsurface(child, parent) → sub    │
│  5. WlEglSurface::new(child, w, h)                           │
│  6. EGL init → glow::Context → egui_glow::Painter            │
│  7. Render loop (egui agnostic ao backend!)                  │
└──────────────────────────────────────────────────────────────┘
```

### 3.3 O que Falta no Ecossistema CLAP

Para que H3 funcione em produção, são necessárias **3 peças** que não existem:

| Peça                        | Status                         | Quem resolve       |
| --------------------------- | ------------------------------ | ------------------ |
| Extensão CLAP Wayland Host  | ❌ Não existe                  | Comitê CLAP (#474) |
| Bitwig nested compositor    | ❌ Não implementado            | Equipe Bitwig      |
| Plugin-side subsurface code | 🟡 Prototipável (~500-800 LOC) | NAM-rs (este lab)  |

> [!IMPORTANT]
> O NAM-rs só controla a peça 3. As peças 1 e 2 são **bloqueadores externos**. Este experimento valida a peça 3 isoladamente usando um mini-compositor Rust (smithay) como simulador de host.

### 3.4 Plano do Experimento (PoC)

**Dois binários Rust:**

1. **`lab-host`** — Mini-compositor Wayland usando `smithay`:

   - Cria janela no compositor do sistema
   - Expõe socket Unix privado como "nested compositor"
   - Fornece `wl_surface` pai para o plugin
   - Implementa `wl_subcompositor` para o plugin criar subsurface

2. **`lab-plugin`** — Cliente Wayland simulando o plugin:

   - Conecta ao socket do `lab-host` (não ao compositor de sistema)
   - Cria `wl_surface` → `wl_subsurface` usando a `wl_surface` pai
   - Inicializa EGL/OpenGL, renderiza egui
   - Valida: o conteúdo aparece embutido na janela do host?

**Dependências extras (lab only):**

```toml
[dev-dependencies]
smithay = { version = "0.4", features = ["backend_winit", "wayland_frontend"] }
```

### 3.5 Implementação do Plugin-Side (Código Alvo para Produção)

Quando a extensão CLAP existir, o código no plugin será aproximadamente:

```rust
// Em src/clap/gui/wayland_embedded.rs (futuro)

/// Backend Wayland embedded via nested compositor.
/// Ativado quando o host implementa a extensão CLAP_EXT_GUI_WAYLAND_HOST.
pub struct WaylandEmbeddedWindow {
    conn: wayland_client::Connection,
    surface: WlSurface,
    subsurface: WlSubsurface,
    egl_window: WlEglSurface,
    egl_display: EglDisplay,
    egl_surface: EglSurface,
    egl_context: EglContext,
    gl: Arc<glow::Context>,
    painter: egui_glow::Painter,
}

impl WaylandEmbeddedWindow {
    pub fn new(
        host_wayland_fd: RawFd,      // Do host via extensão CLAP
        parent_surface: *mut c_void,  // wl_surface* do host (pai)
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        // 1. Conectar ao "compositor" do host (não ao sistema)
        let conn = unsafe { Connection::from_fd(host_wayland_fd)? };
        let (globals, mut queue) = registry_queue_init::<State>(&conn)?;

        // 2. Bind globals
        let compositor: WlCompositor = globals.bind(&qh, 6..=6, ())?;
        let subcompositor: WlSubcompositor = globals.bind(&qh, 1..=1, ())?;

        // 3. Criar superfície e subsuperfície
        let surface = compositor.create_surface(&qh, ());
        let parent = unsafe { WlSurface::from_ptr(parent_surface) };
        let subsurface = subcompositor.get_subsurface(&surface, &parent, &qh, ());
        subsurface.set_desync(); // Atualizações independentes do pai
        subsurface.set_position(0, 0);

        // 4. EGL + OpenGL (idêntico ao H2, passo 4)
        let egl_window = WlEglSurface::new(surface.id(), width as i32, height as i32);
        // ... (EGL init, glow init, egui_glow::Painter init) ...

        Ok(Self { /* ... */ })
    }
}
```

### 3.6 Critérios de Sucesso

- [ ] `lab-host` (smithay) abre janela e expõe socket de nested compositor
- [ ] `lab-plugin` conecta ao socket, cria subsurface, renderiza egui
- [ ] Conteúdo do plugin aparece embutido na janela do host
- [ ] Resize do host propaga corretamente para o plugin
- [ ] Mouse/keyboard events do host chegam ao plugin

### 3.7 Questões Técnicas em Aberto

| #   | Questão                                   | Impacto | Investigar quando         |
| --- | ----------------------------------------- | ------- | ------------------------- |
| Q1  | Como o host passa o fd do wl_display?     | Crítico | Quando CLAP padronizar    |
| Q2  | `set_desync` vs `set_sync` para o plugin? | UX      | No PoC                    |
| Q3  | Quem gerencia o cursor do mouse?          | UX      | No PoC                    |
| Q4  | Menus popup: `xdg_popup` ou subsurface?   | UX      | Após PoC básico funcionar |
| Q5  | DPI: quem informa o scale factor?         | Visual  | Depende da extensão CLAP  |

---

## 4. Referências Técnicas

### 4.1 Especificações e Issues

| Recurso                         | URL                                                                                                                                                | Relevância                                         |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| CLAP `gui.h` (Wayland L64-66)   | [github.com/free-audio/clap](https://github.com/free-audio/clap/blob/master/include/clap/ext/gui.h)                                                | Spec oficial                                       |
| CLAP Issue 474 "GUI on Wayland" | [github.com/free-audio/clap/issues/474](https://github.com/free-audio/clap/issues/474)                                                             | Discussão central (190+ comments)                  |
| VST3 SDK 3.8.0 IWaylandHost     | [steinbergmedia.github.io](https://steinbergmedia.github.io/vst3_dev_portal/pages//Technical+Documentation/Change+History/3.8.0/IWaylandHost.html) | Modelo de referência para H3                       |
| baseview Issue 135              | [github.com/RustAudio/baseview/issues/135](https://github.com/RustAudio/baseview/issues/135)                                                       | Wayland support (sem progresso)                    |
| xdg-foreign-unstable-v2 spec    | [wayland.app/protocols/xdg-foreign-unstable-v2](https://wayland.app/protocols/xdg-foreign-unstable-v2)                                             | Protocolo do H2                                    |
| Wayland subsurface spec         | [wayland-book.com/surfaces/subsurfaces](https://wayland-book.com/surfaces/subsurfaces.html)                                                        | Protocolo do H3                                    |
| winit Issue 1161                | [github.com/rust-windowing/winit/issues/1161](https://github.com/rust-windowing/winit/issues/1161)                                                 | Wrap existing window (sem solução — H4 descartada) |

### 4.2 Crates Rust Relevantes

| Crate               | Versão | Papel                                            |
| ------------------- | ------ | ------------------------------------------------ |
| `wayland-client`    | 0.31+  | Cliente Wayland puro Rust                        |
| `wayland-protocols` | 0.32+  | Bindings xdg-shell, xdg-foreign, etc.            |
| `wayland-egl`       | 0.32+  | Bridge wl_surface → wl_egl_window                |
| `khronos-egl`       | 6.0+   | EGL loader dinâmico                              |
| `glow`              | 0.16+  | OpenGL wrapper                                   |
| `egui_glow`         | 0.31+  | Renderer egui sobre glow                         |
| `smithay`           | 0.4+   | Compositor Wayland em Rust (para lab-host do H3) |

---

## 5. Log de Evolução

> Registro cronológico de mudanças no ecossistema e resultados de experimentos.

| Data       | Evento                                                      | Impacto                               |
| ---------- | ----------------------------------------------------------- | ------------------------------------- |
| 2022-11-xx | baseview Issue 135 aberta                                   | Wayland no baseview: sem previsão     |
| 2025-10-05 | CLAP Issue 474 aberta (dekrain)                             | Discussão oficial sobre GUI Wayland   |
| 2025-10-20 | VST3 SDK 3.8.0 lançado com `IWaylandHost` (MIT)             | Precedente para nested compositor     |
| 2025-10-20 | Studio One 7.2 com Wayland nativo (VST3 only)               | Primeiro host com embedding real      |
| 2026-05-19 | NAM-rs: pesquisa Lab concluída. H1 (X11) aprovada para v1.0 | Decisão registrada em architecture.md |
| —          | *(Próximos eventos serão registrados aqui)*                 | —                                     |
