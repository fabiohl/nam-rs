<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# CLAP Integration & GUI Architecture

Architecture, real-time safety model, and graphical interface of the NAM-rs CLAP plugin.

## 1. Thread Model

The plugin maps NAM-rs's existing thread segregation onto the host (DAW) model:

- **Main Thread (Host)** — plugin lifecycle, parameter scanning, state
  save/load, model loading (`src/clap/plugin/main_thread/`), and GC disposal.
  Loads `.nam`/`.namb` via `src/loader/`.
- **Audio Thread (RT)** — driven by the host `process()` callback.
  **Hard contract: zero allocations, zero locks, zero blocking I/O.** Runs the
  DSP pipeline (`src/clap/processor/dsp/`). Unlike PipeWire's dual-stream model,
  CLAP delivers input and output in a single callback, so no `DspBridge` is used.
- **GUI Thread** — dedicated baseview event loop, fully isolated from the audio
  thread (see §7).

## 2. Compilation Strategy

- `cargo build` — default standalone executable (PipeWire backend, `standalone`
  feature). Omits `src/clap/gui/`.
- `cargo build --no-default-features --features clap-plugin --lib` — `.clap`
  dynamic library with GUI. Omits `src/standalone/` (PipeWire host, RT setup,
  CLI), keeping the binary free of PipeWire dependencies.

All GUI code lives under `src/clap/gui/` and is gated by `#[cfg(feature = "clap-plugin")]`.

## 3. Plugin Descriptor

| Field    | Value                                                         |
|:-------- |:------------------------------------------------------------- |
| ID       | `br.eti.fabiolima.nam-rs`                                     |
| Name     | `NAM-rs`                                                      |
| Vendor   | `Fabio Lima`                                                  |
| URL      | <https://github.com/fabiohl/nam-rs>                           |
| Features | `["audio-effect", "distortion", "gate", "mono"]` (CLAP 1.2.2) |

The descriptor is returned by `nam_descriptor()` (`src/clap/descriptor.rs`) and
is allocation-free, as it is read during host scan.

> NAM is mono by definition. Core DSP is strictly mono in/mono out; channel
> routing is managed by the host. The VU meter is adaptive (see §7.4).

## 4. Parameters

Exposed via `NamPluginParams` (`src/common/params.rs`) and mapped in
`src/clap/extensions/params/`. IDs are `u32` constants `PARAM_*` (0–8).

| Parameter            | ID                     | Type    | Notes                                                                    |
|:-------------------- |:---------------------- |:------- |:------------------------------------------------------------------------ |
| Input Gain           | `input_gain_db`        | dB      | Pre-inference gain, sample-accurate smoothed.                            |
| Output Gain          | `output_gain_db`       | dB      | Post-inference gain, sample-accurate smoothed.                           |
| Gate Threshold       | `gate_threshold_db`    | dB      | Noise-gate opening threshold.                                            |
| Bypass               | `bypass`               | binary  | Disables neural processing (dry = wet).                                  |
| Active Model         | `active_model`         | —       | Loaded model name (read-only).                                           |
| Adaptive Compute     | `adaptive_compute`     | binary  | CPU-based quality fallback (FSM).                                        |
| Slim Override        | `slim_override`        | stepped | Auto / ForceFull / ForceLite.                                            |
| Oversampling Factor  | `oversample`           | stepped | Off / 2× / 4×.                                                           |
| Activation Precision | `activation_precision` | stepped | Standard (exact-grade, universal default) / Fast (Padé/minimax, opt-in). |

The model path is a **State Property**, letting the DAW persist/restore the
correct model in a project.

## 5. CLAP Extensions

Registered in `declare_extensions()` (`src/clap/plugin/mod.rs`) via
`clack-extensions`:

| Extension                      | File                                         | Purpose                                                                                                                                                                         |
|:------------------------------ |:-------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `clap_plugin_audio_ports`      | `src/clap/extensions/audio_ports.rs`         | Mono in/out ports, in-place pair enabled.                                                                                                                                       |
| `clap_plugin_params`           | `src/clap/extensions/params/`                | Parameter mapping/automation with gesture + `flush()` support.                                                                                                                  |
| `clap_plugin_state`            | `src/clap/extensions/state.rs`               | Persist parameters + model path in the DAW project.                                                                                                                             |
| `clap_plugin_state_context`    | `src/clap/extensions/state_context.rs`       | Distinguish preset save (portable, no abs path) vs project/duplicate.                                                                                                           |
| `clap_plugin_latency`          | `src/clap/extensions/latency.rs`             | Dynamic latency reporting (resampler + oversample + cabsim).                                                                                                                    |
| `clap_plugin_track_info`       | `src/clap/extensions/track_info.rs`          | Host track color → GUI accent color.                                                                                                                                            |
| `clap_plugin_remote_controls`  | `src/clap/extensions/remote_controls.rs`     | "Main" / "Gate" control pages for HW controllers / Device Panel.                                                                                                                |
| `clap_plugin_param_indication` | `src/clap/extensions/param_indication.rs`    | GUI feedback for mapped/automated/overridden parameters.                                                                                                                        |
| `clap_plugin_preset_load`      | `src/clap/extensions/preset_load.rs`         | Load `.nam`/`.namb` from the host preset browser.                                                                                                                               |
| `clap_plugin_render`           | `src/clap/extensions/render.rs`              | Offline mode forces `AdaptiveCompute::Off` + `Standard` activation precision (max quality, exact-grade). `has_hard_realtime_requirement = false` (NAM is deterministic/causal). |
| `clap_plugin_gui`              | `src/clap/extensions/gui.rs` *(clap-plugin)* | Native `egui` GUI via `baseview`, X11/XWayland (`CLAP_WINDOW_API_X11`).                                                                                                         |

A separate **Preset Discovery Factory** (`src/clap/factory/preset_discovery.rs`)
indexes `.nam`/`.namb` files from `~/.nam/models` so hosts can list them as
presets with extracted metadata.

## 6. Lock-Free Communication & RT Safety

Cross-thread state lives in `NamClapShared` (`src/clap/plugin/shared.rs`). The
hot-path uses **no mutexes, no locks, no allocations** — only atomics and SPSC
queues. Mutexes wrap SPSC channel endpoints solely to allow ownership transfer
during `activate()`/`deactivate()` and to satisfy the `Sync` bound.

### 6.1 Cache-Line Isolation

`NamClapShared` is split into three `#[repr(align(128))]` sub-structs so no two
share a cache line (false-sharing prevention):

| Sub-struct   | Writer         | Reader         | Contents                                                                |
|:------------ |:-------------- |:-------------- |:----------------------------------------------------------------------- |
| `RtToUi`     | RT (every blk) | UI (refresh)   | `ui_peak_l/r`, `ui_clipped`, `current_latency`, `active_channel_count`. |
| `UiToRt`     | UI/Main        | RT (every blk) | 8 param atomics, `gesture_flags`, `gui_param_generation`.               |
| `ColdShared` | both (rare)    | both (rare)    | SPSC queues, sample/buffer size, model metadata, IR state, indications. |

### 6.2 Generation-Counter Parameter Sync

Loading 8 atomic params every block wastes cycles when params are stationary.
Protocol:

1. UI writes the new value to the `UiToRt` atomic and `fetch_add(1, Release)` on
   `gui_param_generation`.
2. RT does a single `Acquire` load of `gui_param_generation`. If unchanged, it
   skips all per-param loads. If changed, it does `Relaxed` loads and feeds
   targets to `ParamSmoother`.

### 6.3 Three-Tier RT-Safe Garbage Collection

Dropping `Box<StaticModel>`/`Box<NamResampler>`/`ConvEngine`/`OversampleEngine`
on the RT thread would block (system dealloc). Disposal cascades through three
tiers, all lock-/alloc-free on the RT side (`gc_cascade` in
`src/common/spsc/gc.rs`):

1. **SPSC GC channel** (`gc_tx`, 32 slots) → drained by the main thread
   (`drain_gc_channels`) in `housekeeping()`.
2. **16-slot `parking_lot`** (`[Option<GcItem>; 16]` on the processor) — retried
   every block via `drain_parking_lot()`. Capacity is sized for 3 items/swap at
   ~0.3–1.5 ms/block ⇒ >96 slots/s throughput.
3. **`GcOverflowBuffer`** ring (`SPSC_CAPACITY`, packed `AtomicU64` slots) — last
   resort; overwrites the oldest slot (controlled leak). Sets
   `RT_STATUS_GC_OVERFLOW`; unknown type-ids on drain are intentionally leaked
   and set `RT_STATUS_GC_CORRUPTED`.

Telemetry peaks (`ui_peak_l/r`) use `Relaxed` stores; the UI reads them with
`swap(0)` so silence decays correctly when no audio runs.

### 6.4 FFI Lifetime Safety

- **`alive_fence`** (`Arc<AtomicBool>` in `ColdShared`, cleared in `Drop`): the
  async file-picker thread and `NamPluginWindow::safe_shared()` check it before
  dereferencing the shared pointer, preventing use-after-free if the plugin is
  destroyed while a background thread still holds a reference.
- **`extend_host_lifetime`** (`src/clap/gui/mod.rs`): `unsafe` transmute of the
  `HostSharedHandle` to `'static`, documented with the invariant that the GUI
  window is destroyed before the plugin (guaranteed by the CLAP lifecycle).
- **Panic hook**: `install_panic_hook("clap")` runs once in `new_shared()`;
  `Drop` calls `set_shutdown_in_progress()`. A panic crossing the C-ABI FFI
  boundary is UB, so `on_frame()`/`on_event()` use silent early-returns instead
  of unwinding.

### 6.5 Deferred Model Load (F3)

State restore can arrive before `activate()` (host buffer size still 0), where
building the resampler/buffers would allocate on a context without a known
`max_frames_count`. The model payload is stored in `ColdShared::pending_model`
and flushed by `flush_pending_model()` — called from `activate()` (primary) and
`housekeeping()` (fallback for hosts that load state between `activate()` and
the first `process()`). `set_max_buffer_size` is **never** called on the audio
thread; the `heap-audit` CI lane enforces this.

### 6.6 Channel Return on `deactivate()`

`deactivate()` returns the `param_rx`, `gc_tx`, and `slimmable_rx` consumers
back into `ColdShared`, so a host that deactivates/reactivates the processor
(without recreating the plugin) keeps working.

### 6.7 Render Mode

`clap.render.set()` stores the mode (`Release`). On transition the RT thread
forces `AdaptiveCompute::Off` + `ActivationPrecision::Standard` (exact-grade)
in offline mode for deterministic max-quality bounce/export, and restores
user settings on return to realtime.

## 7. GUI Architecture

### 7.1 Windowing Stack (Unified X11)

```text
┌────────────────────────────────────────────┐
│            NAM-rs GUI (egui v0.34)         │
│         draw_ui() — agnostic UI logic      │
├────────────────────────────────────────────┤
│     NamPluginWindow (WindowHandler)        │
│   baseview events → egui::RawInput         │
│   egui_glow::Painter + glow v0.17          │
├────────────────────────────────────────────┤
│             Backend X11 (baseview)         │
│   raw-window-handle 0.5 → 0.6 translation  │
│        Pure X11 / native XWayland          │
└────────────────────────────────────────────┘
```

| Component   | Crate       | Role                                                                   |
|:----------- |:----------- |:---------------------------------------------------------------------- |
| GUI         | `egui`      | Immediate-mode; no persistent state, no GC, no render-loop alloc.      |
| Renderer    | `egui_glow` | egui → OpenGL 3.3 via `glow`. Manual integration (no `egui-baseview`). |
| Windowing   | `baseview`  | Native embedded X11 window via `RawWindowHandle`.                      |
| File Picker | `rfd`       | Native async dialog (zenity/xdg-portal); never blocks UI.              |

Only `CLAP_WINDOW_API_X11` is declared. Native Wayland embedding is planned.

### 7.2 Module Map

- `src/clap/gui/mod.rs` — entryway; `GUI_WIDTH=600`/`GUI_HEIGHT=275`, `extend_host_lifetime`.
- `src/clap/gui/window/state.rs` — `NamPluginWindow`: GL context init, `egui_glow` painter,
  shader compile, theme, teardown. `safe_shared()` guards `alive_fence`.
- `src/clap/gui/window/handler.rs` — `WindowHandler`: `on_frame`/`on_event`, event →
  `egui::RawInput`, drag-and-drop, conditional paint loop.
- `src/clap/gui/window/shaders.rs` — VU GLSL (vertex + fragment).
- `src/clap/gui/window/{drag_drop,input_map}.rs` — model-file validation, key/mouse mapping.
- `src/clap/gui/ui/mod.rs` — 5-zone layout (`draw_ui`).
- `src/clap/gui/ui/zones/` — `identity` (Z1), `controls` (Z2), `meters` (Z3), `bypass_zone` (Z4).
- `src/clap/gui/ui/status_bar/` (Z5) — `orchestrator`, `telemetry`, `metadata`.
- `src/clap/gui/ui/meter/` — `orchestrator`, `glow` (GPU), `cpu` (fallback), `readout`.
- `src/clap/gui/ui/{knob,focus,colors,simd,vsep,bypass,state}.rs` — widgets, a11y, theme.

### 7.3 Frame Lifecycle & Idle Skip

`on_frame()` makes the GL context current, runs `egui_ctx.run_ui(draw_ui)`,
tessellates, paints, and swaps buffers — unless a skip is decided:

```rust
let should_skip = !self.dirty
    && !has_short_repaint
    && !hold_changed
    && time_since_paint < Duration::from_millis(22);
```

- `!dirty` — no input since last paint.
- `!has_short_repaint` — egui requested a short (<50 ms) repaint (toasts/spinners).
- `!hold_changed` — VU peak-hold is not decaying.
- 22 ms throttle — caps active repaint at ~45 FPS, lowering idle CPU with many
  instances. On skip the GL context is released and the frame exits early.

`on_event` always sets `dirty = true`. Close is signalled via `close_signal`:
the next `on_frame` destroys GL resources idempotently and closes the window.

### 7.4 Adaptive VU Metering

The core DSP is mono, but the VU meter adapts to the host's track config. During
`process()`, the RT thread counts channel buffers and stores `1` or `2` in
`RtToUi::active_channel_count` (`Relaxed`). The UI reads it and renders one
centered 76 px bar (mono) or two 36 px L/R bars (stereo).

### 7.5 GPU Shaders & CPU Fallback

- **Vertex** (`shaders.rs`): generates a quad from `gl_VertexID` (no VBO); NDC
  from `u_meter_rect` × `u_viewport`.
- **Fragment**: 3-color dB gradient (green ≤ −12, yellow −12→−3, red −3→+6 dBFS),
  distance-field rounded corners, 1.5 px peak-hold line that recolors by range.
- **Fallback** (`meter/cpu.rs`): if shaders fail to compile or GL features are
  missing, `vu_program` is `None` and flat rectangles are drawn on the egui
  painter mesh, keeping the UI usable.

### 7.6 Async File Dialog & Error Toasts

"Load Model" spawns an `rfd` background thread (X11 space, no DAW freeze). While
open, `ui_loading` shows a loading state. On selection the path lands in
`ui_pending_model` and `host.request_callback()` schedules a main-thread load.
On failure, `ui_load_error` + `ui_load_error_msg` drive a red `⚠ Load failed`
status with a hover tooltip (3 s expiry). Drag-and-drop follows the same path.

### 7.7 Keyboard Accessibility

Focus cycles `Input Gain → Output Gain → Gate → Bypass → Load`
(`gui/ui/focus.rs`). Tab forward, Shift+Tab backward; focused controls show an
accent ring; Space/Enter activate the focused Load/Bypass control.

## 8. Model Gain Calibration

NAM models carry `input_level_dbu` and `loudness` metadata. The loader computes
`input_mult_adj`/`output_mult_adj` (`src/loader/build.rs`) to normalize to
12.0 dBu / −18 dB. **Calibration is always applied** (parity with the standalone
binary and the reference C++ `calibrated_loudness`).

Flow: `load_and_build_model()` → `ClapParamPayload::LoadModel` → RT
`cold_load_model()` stores them in `model_input_mult_adj`/`model_output_mult_adj`
→ the pipeline context carries them as `input_gain_mult`/`output_gain_mult`,
while `smoother_in`/`smoother_out` carry only user gain. Unlike the standalone
(which fuses user + model into one value), CLAP keeps them separate so
sample-accurate user-gain smoothing never touches static model calibration.

## 9. Target DAWs for Validation

- **Bitwig Studio** — primary CI target; CLAP co-author. Validates sandboxing,
  dynamic params, state persistence, sample-accurate automation.
- **REAPER** — not actively tested (Linux PipeWire issues on Debian/Ubuntu make
  reproducible validation impractical).
- **Fender Studio Pro 8+** — native Wayland host; GUI is X11-only, so it runs in
  floating fallback mode or via host generic sliders. Native Wayland GUI support
  is planned.
- **CLAP-info / CLAP-host** — CLI tools for spec validation.
