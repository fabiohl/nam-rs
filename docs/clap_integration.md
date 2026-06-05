<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# CLAP (Clever Audio Plug-in) Integration Strategy

This document describes the architecture and strategy for transforming the NAM-rs DSP engine into an audio plugin compatible with the CLAP standard.

## 1. Thread Model

The CLAP integration must strictly respect the thread segregation already existing in NAM-rs, mapping them to the host's (DAW) model:

- **Main Thread (Host)**:
  - Responsible for plugin initialization, parameter scanning, and state management.
  - In NAM-rs, this thread will replace the main loop in [src/main.rs](file:///home/fabio/nam-rs/src/main.rs).
  - Manages the loading of `.nam`/`.namb` files via [src/loader/](file:///home/fabio/nam-rs/src/loader/).
- **Audio Thread (Real-time)**:
  - Called by the host via the `process()` callback.
  - **Critical Requirement**: Must maintain a policy of **ZERO allocations** and **ZERO locks**.
  - Will utilize [src/dsp/pipeline.rs](file:///home/fabio/nam-rs/src/dsp/pipeline.rs) for processing, adapting CLAP buffers to the internal format.
  - Unlike PipeWire (which is dual-stream), CLAP provides input and output buffers in a single context, eliminating the need for `DspBridge`.

## 2. Parameter Mapping

Parameters exposed to the host will be mapped from the `NamPluginParams` structure (see [src/common/params.rs](file:///home/fabio/nam-rs/src/common/params.rs)):

| CLAP Parameter     | ID                  | Unit   | Description                                 |
|:------------------ |:------------------- |:------ |:------------------------------------------- |
| **Input Gain**     | `input_gain_db`     | dB     | Gain applied before neural inference.       |
| **Output Gain**    | `output_gain_db`    | dB     | Gain applied after neural inference.        |
| **Gate Threshold** | `gate_threshold_db` | dB     | Noise Gate opening threshold.               |
| **Bypass**         | `bypass`            | binary | Disables neural processing (Dry/Wet 0/100). |
| **Active Model**   | `active_model`      | —      | Loaded model name (read-only).              |

The model path (`model_path`) will be treated as a **State Property**, allowing the DAW to save and load the correct model in the project.

## 3. Compilation Strategy

The project uses *feature flags* to allow multiple build targets:

- `cargo build --features standalone`: Executable binary with PipeWire backend (default).
- `cargo build --no-default-features --features clap-plugin --lib`: Dynamic library (`.clap`) with a complete GUI.

The `clap-plugin` feature will omit the `pw_host.rs` and `rt_setup.rs` modules, keeping the final binary free of PipeWire dependencies.

## 4. Framework: `clack-plugin`

- **Reason**: Offers granular control over the implementation without adding unnecessary overhead, allowing direct integration with the RT-safe structures of NAM-rs. Unlike more opinionated frameworks, `clack` maps almost 1:1 to the CLAP spec while providing type safety in Rust.
- **High-level Frameworks**: Discarded as they force support for VST3, introduce an embedded GUI layer that conflicts with our choice of pure `egui`, and add abstractions that could mask the temporal determinism required by the NAM-rs DSP engine.
- **Link**: [https://github.com/prokopyl/clack](https://github.com/prokopyl/clack)

## 5. Implemented CLAP Extensions

The integration uses the `clack-extensions` crate to implement the following extensions of the CLAP protocol:

| Extension                      | File                                                                                     | Purpose                                                                                                                 |
|:------------------------------ |:---------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------- |
| `clap_plugin_audio_ports`      | [audio_ports.rs](file:///home/fabio/nam-rs/src/clap/extensions/audio_ports.rs)           | Explicit declaration of mono input/output ports and support for in-place processing                                     |
| `clap_plugin_params`           | [params.rs](file:///home/fabio/nam-rs/src/clap/extensions/params.rs)                     | Mapping and automation of parameters (`input_gain`, `output_gain`, `gate`, `bypass`) with gesture and `flush()` support |
| `clap_plugin_state`            | [state.rs](file:///home/fabio/nam-rs/src/clap/extensions/state.rs)                       | Persistence of plugin state (parameters and model path) in the DAW project                                              |
| `clap_plugin_latency`          | [latency.rs](file:///home/fabio/nam-rs/src/clap/extensions/latency.rs)                   | Dynamic reporting of latency induced by processing and resampling to the host                                           |
| `clap_plugin_track_info`       | [track_info.rs](file:///home/fabio/nam-rs/src/clap/extensions/track_info.rs)             | Support for host track color to dynamically adapt the GUI's accent color                                                |
| `clap_plugin_remote_controls`  | [remote_controls.rs](file:///home/fabio/nam-rs/src/clap/extensions/remote_controls.rs)   | Pre-configured control pages ("Main" and "Gate") for hardware controller and Device Panel integration                   |
| `clap_plugin_param_indication` | [param_indication.rs](file:///home/fabio/nam-rs/src/clap/extensions/param_indication.rs) | Visual feedback in the GUI to indicate parameters that are mapped, automated, or under temporary override               |
| `clap_plugin_gui`              | [gui.rs](file:///home/fabio/nam-rs/src/clap/extensions/gui.rs)                           | Native GUI based on `egui` v0.34 embedded via `baseview` and X11/XWayland backend (`CLAP_WINDOW_API_X11`)               |

## 6. Plugin Descriptor

The plugin metadata descriptor will follow this pattern:

- **Plugin ID**: `br.eti.fabiolima.nam-rs`
- **Name**: `NAM-rs`
- **Vendor**: `Fabio Lima`
- **URL**: [https://github.com/fabiohl/nam-rs](https://github.com/fabiohl/nam-rs)
- **Features**: `["audio-effect", "distortion", "gate", "simulator", "mono"]`

> [!NOTE]
> The NAM standard is, by definition, mono. The CLAP plugin works strictly as mono (mono-in/mono-out) to align with traditional DAW workflows where channel routing is managed externally by the host. In contrast, in the Standalone/Pipewire executable, stereo processing is provided as a convenience for native stereo signals.

## 7. Target DAWs for Validation

- **Bitwig Studio**: Absolute reference platform for CLAP compliance (co-author of the standard). Essential for validating sandboxing behavior and sample-accurate automation.
- **REAPER**: Validation of compatibility with low-cost hosts and tests of irregular buffer sizes.
  - NOTE: *Discarded* as it is buggy on my Ubuntu Linux machine.
- **Fender Studio Pro**: Future target requiring Wayland native mode.
- **CLAP-info / CLAP-host**: Command-line tools for rigorous technical validation of the spec.

## 8. Graphical Interface: Windowing Strategy and Stack

The CLAP plugin GUI operates on a dedicated thread (`UI thread`), completely isolated from the `audio thread`. The architecture is unified on the X11 backend.

### Unified Windowing Strategy (Pure X11)

```text
┌────────────────────────────────────────────────┐
│                  NAM-rs GUI                    │
│              (egui + egui_glow)                │
│    draw_ui() — 100% agnostic UI logic          │
├────────────────────────────────────────────────┤
│       NamPluginWindow (WindowHandler)          │
│   baseview events → egui::RawInput translation │
│   Rendering via egui_glow::Painter + glow      │
├────────────────────────────────────────────────┤
│                  Backend X11                   │
│   (baseview - raw-window-handle 0.5 → 0.6)    │
│           Pure X11 / native XWayland           │
└────────────────────────────────────────────────┘
```

- **X11 Backend:** The plugin declares exclusive support for `CLAP_WINDOW_API_X11`.
- **Stack:** `egui v0.34` + `glow v0.17`, with window handle translation (`raw-window-handle 0.5` from host to `0.6` for `egui`/`baseview`).
- **Implementation:** `NamPluginWindow` implements `baseview::WindowHandler`, translating events to `egui::RawInput` without an intermediate layer.

### Technology Stack

| Component     | Crate/Technology | Role                                                                                          |
|:------------- |:---------------- |:--------------------------------------------------------------------------------------------- |
| GUI Framework | `egui`           | Immediate Mode GUI — no persistent state, no GC, no allocations in the render loop            |
| Renderer      | `egui_glow`      | Bridge egui → OpenGL 3.3 via `glow`. Manual integration (no `egui-baseview`, abandoned ~2021) |
| Windowing     | `baseview`       | Native embedded X11 window via `RawWindowHandle`. Dedicated event loop                        |
| File Picker   | `rfd`            | Native asynchronous file dialog (zenity/xdg-portal). Never blocks the UI thread               |

All GUI code lives under `src/clap/gui/` and is gated by `#[cfg(feature = "clap-plugin")]`.

### Thread Isolation (UI ↔ Audio)

The UI thread **never** directly accesses the fields of `NamClapProcessor`. Communication is strictly via:

- **Telemetry Read (Audio → UI):** Atomic fields in `NamClapShared` (`AtomicU32` for peaks, `AtomicBool` for clipping), read with `Ordering::Relaxed`.
- **Command Dispatch (UI → Audio):** SPSC parameter channel (`ClapParamPayload`) via `param_tx`, drained at the start of each `process()`.
- **Metadata (Main → UI):** `Mutex<String>` for the model name — accessed by the UI thread at 500ms intervals.
