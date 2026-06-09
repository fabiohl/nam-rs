<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# CLAP (Clever Audio Plug-in) Integration Strategy

This document describes the architecture and strategy for transforming the NAM-rs DSP engine into an audio plugin compatible with the CLAP standard.

## 1. Thread Model

The CLAP integration must strictly respect the thread segregation already existing in NAM-rs, mapping them to the host's (DAW) model:

- **Main Thread (Host)**:
  - Responsible for plugin initialization, parameter scanning, and state management.
  - In NAM-rs, this thread manages the CLAP lifecycle via `src/clap/plugin/main_thread.rs`.
  - Manages the loading of `.nam`/`.namb` files via `src/loader/`.
- **Audio Thread (Real-time)**:
  - Called by the host via the `process()` callback.
  - **Critical Requirement**: Must maintain a policy of **ZERO allocations** and **ZERO locks**.
  - Uses `src/dsp/pipeline/` for processing, adapting CLAP buffers to the internal format.
  - Unlike PipeWire (which is dual-stream), CLAP provides input and output buffers in a single context, eliminating the need for `DspBridge`.

## 2. Parameter Mapping

Parameters exposed to the host will be mapped from the `NamPluginParams` structure (see `src/common/params.rs`):

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

The `clap-plugin` feature will omit the entire `src/standalone/` module (PipeWire host, RT setup, and CLI), keeping the final binary free of PipeWire dependencies.

## 4. Framework: `clack-plugin`

- **Reason**: Offers granular control over the implementation without adding unnecessary overhead, allowing direct integration with the RT-safe structures of NAM-rs. Unlike more opinionated frameworks, `clack` maps almost 1:1 to the CLAP spec while providing type safety in Rust.
- **High-level Frameworks**: Discarded as they force support for VST3, introduce an embedded GUI layer that conflicts with our choice of pure `egui`, and add abstractions that could mask the temporal determinism required by the NAM-rs DSP engine.
- **Link**: [https://github.com/prokopyl/clack](https://github.com/prokopyl/clack)

## 5. Implemented CLAP Extensions

The integration uses the `clack-extensions` crate to implement the following extensions of the CLAP protocol:

| Extension                      | File                                               | Purpose                                                                                                                 |
|:------------------------------ |:-------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------- |
| `clap_plugin_audio_ports`      | `src/clap/extensions/audio_ports.rs`               | Explicit declaration of mono input/output ports and support for in-place processing                                     |
| `clap_plugin_params`           | `src/clap/extensions/params.rs`                    | Mapping and automation of parameters (`input_gain`, `output_gain`, `gate`, `bypass`) with gesture and `flush()` support |
| `clap_plugin_state`            | `src/clap/extensions/state.rs`                     | Persistence of plugin state (parameters and model path) in the DAW project                                              |
| `clap_plugin_latency`          | `src/clap/extensions/latency.rs`                   | Dynamic reporting of latency induced by processing and resampling to the host                                           |
| `clap_plugin_track_info`       | `src/clap/extensions/track_info.rs`                | Support for host track color to dynamically adapt the GUI's accent color                                                |
| `clap_plugin_remote_controls`  | `src/clap/extensions/remote_controls.rs`           | Pre-configured control pages ("Main" and "Gate") for hardware controller and Device Panel integration                   |
| `clap_plugin_param_indication` | `src/clap/extensions/param_indication.rs`          | Visual feedback in the GUI to indicate parameters that are mapped, automated, or under temporary override               |
| `clap_plugin_gui`              | `src/clap/extensions/gui.rs`                       | Native GUI based on `egui` v0.34 embedded via `baseview` and X11/XWayland backend (`CLAP_WINDOW_API_X11`)               |

> [!NOTE]
> **Adaptive VU Metering Contract:** While the core DSP processing (neural network inference and noise gate) is mono, the VU meter dynamically adapts to the host's track channel configuration.
>
> - **Host Configuration & Dynamic Detection:** During the FFI `process()` callback, the plugin inspects the host's `Audio` port buffers inside the real-time thread (`src/clap/processor/dsp/channels.rs`).
> - **Shared State Update:** If the host provides $\ge 2$ channel buffers (indicating a stereo routing context), the audio thread dynamically stores `2` in the atomic field `shared.rt_to_ui.active_channel_count` (with `Ordering::Relaxed`). Otherwise, it stores `1`.
> - **UI Layout Adaptation:** The GUI/UI thread reads this atomic variable inside the layout logic (`src/clap/gui/ui/zones/meters.rs`) and adapts the VU meter rendering at runtime: showing two separate L and R bars (36px wide each) or a single centered bar (76px wide).

## 6. Plugin Descriptor

The plugin metadata descriptor will follow this pattern:

- **Plugin ID**: `br.eti.fabiolima.nam-rs`
- **Name**: `NAM-rs`
- **Vendor**: `Fabio Lima`
- **URL**: [https://github.com/fabiohl/nam-rs](https://github.com/fabiohl/nam-rs)
- **Features**: `["audio-effect", "distortion", "gate", "mono"]` (CLAP 1.2.2 standard features — validated against `include/clap/plugin-features.h`)

> [!NOTE]
> The NAM standard is, by definition, mono. The CLAP plugin's core DSP processing works strictly as mono (mono-in/mono-out) to align with traditional DAW workflows where channel routing is managed externally by the host. However, the VU meter in Zone 3 is adaptive: it displays a single centered bar when running in a mono track configuration, or two independent L/R bars when the host configures a stereo track (providing $\ge 2$ channels). In contrast, in the Standalone/Pipewire executable, full stereo processing is provided as a convenience for native stereo signals.

## 7. Target DAWs for Validation

- **Bitwig Studio**: Active validation target. The absolute reference platform for CLAP compliance (co-author of the standard). Essential for validating sandboxing behavior, dynamic parameters, state persistence, and sample-accurate automation.
- **REAPER**: *Not actively tested* — known issues with the Linux PipeWire backend on Debian/Ubuntu-based systems make reproducible validation impractical. Bitwig Studio remains the primary CI target.
- **Fender Studio Pro 8+**: Active validation target. Since the host runs as a native Wayland client on Linux and the plugin GUI is currently built exclusively for X11 (`CLAP_WINDOW_API_X11`), native embedding is not supported. The plugin instead runs in floating fallback mode (opening as an independent top-level window) or via host-provided generic sliders, verifying DSP, parameter automation sync, and host stability without freezing. Native Wayland GUI support (`CLAP_WINDOW_API_WAYLAND`) is planned for a future release.
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

## 9. Lock-Free Communication & Cache-Line Optimization

To achieve absolute real-time safety, high throughput, and low latency on the audio thread, the CLAP integration avoids mutexes or any blocking operations in the processing hot-path. Communication and synchronization between the GUI/Main thread and the Audio (RT) thread rely on cache-aligned atomic structures and Single-Producer Single-Consumer (SPSC) channels.

### Cache-Line Isolation (False Sharing Prevention)

Modern CPUs transfer data between cores in cache lines (typically 64 or 128 bytes). If two threads on different cores frequently write to different variables that reside on the same cache line, the cache line bounces between cores (False Sharing), degrading performance.

To prevent this, `NamClapShared` segregates fields into three sub-structs based on access pattern, each isolated using `#[repr(align(128))]` to ensure they never share a cache line:

1. **`RtToUi` (`#[repr(align(128))]`)**:
   - **Access Pattern**: Written every audio block by the RT thread, read at the GUI refresh rate by the UI thread.
   - **Data**: Peak levels (`ui_peak_l`, `ui_peak_r`), clipping flag (`ui_clipped`), reported latency (`current_latency`), and active channel count.
2. **`UiToRt` (`#[repr(align(128))]`)**:
   - **Access Pattern**: Written by the UI thread when controls are adjusted, read every block by the RT thread.
   - **Data**: Target parameters (`param_input_gain`, `param_output_gain`, `param_gate_thresh`, `param_bypass`, `param_adaptive_compute`), gesture modification flags, and the synchronization counter (`gui_param_generation`).
3. **`ColdShared` (`#[repr(align(128))]`)**:
   - **Access Pattern**: Low-frequency access by both threads (e.g., initialization, model loading, DAW track changes).
   - **Data**: SPSC queues (`param_rx`/`param_tx`, `gc_rx`/`gc_tx`), sample rates, model metadata, track accent colors, parameter indications, and UI loading states.

### Parameter Synchronization Protocol (`gui_param_generation`)

During standard processing, loading multiple atomic parameters from `UiToRt` (such as gains, gate, and bypass) in every single audio block introduces unnecessary atomic read overhead, even when parameters are stationary.

To minimize hot-path atomic overhead, the synchronization uses a generation-counter protocol:

- **UI Thread Update**: When a GUI control (e.g., a knob) is modified, the GUI thread writes the new value to the corresponding atomic parameter in `UiToRt` and increments `gui_param_generation` using `Release` ordering.
- **RT Thread Check**: At the start of `process_events()`, the RT thread reads `gui_param_generation` using a single `Acquire` load.
  - If the loaded value matches the cached `last_seen_generation`, no GUI parameters have changed. The RT thread skips reading the individual atomic parameter fields, reducing the block overhead to a single atomic check.
  - If the value differs, the RT thread updates its `last_seen_generation` and performs `Relaxed` loads on each parameter inside `UiToRt` to synchronize its internal state.

### SPSC Queues & RT-Safe Resource Management

Since loading neural network models and resamplers requires disk access, parsing, and heap allocation, these tasks are offloaded to the Main/UI thread. The CLAP plugin uses lock-free SPSC queues (implemented via `rtrb`) for cross-thread transfers:

1. **Model/Parameter Transfer (`param_tx` / `param_rx`)**:
   - The Main thread packages new parameters or fully loaded models (and resamplers) into a `ClapParamPayload` enum and sends them via the queue.
   - The RT thread drains this queue non-blockingly at the start of `process_events()`.
2. **RT-Safe Garbage Collection (`gc_tx` / `gc_rx`)**:
   - The RT thread must never drop heap-allocated objects (such as `Box<DynamicModel>` or `Box<NamResampler>`), as dropping can trigger system deallocations and block the audio thread.
   - When a new model is loaded, the RT thread replaces the active model/resampler and pushes the obsolete instances into `gc_tx` as `GcItem` variants.
   - The Main thread periodically drains `gc_rx` and safely drops the resources.
   - If `gc_tx` is full during a burst of swaps, the RT thread places the items in a fixed-capacity `parking_lot` array (capacity of 16), which is subsequently drained to `gc_tx` in later blocks.
