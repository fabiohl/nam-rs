<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Master Roadmap of Functional Tests (by humans) — NAM-rs (CLAP Plugin)

**Audience:** UI/UX specialists and end users.
**Target DAWs:** Bitwig Studio 6+ and Fender Studio Pro 8+ (Linux, Flatpak).
**Preparation:** `~/.clap/nam-rs.clap` installed (Release build), ≥2 `.nam` models available (one of them being a bogus/invalid file), Guitar DI or signal generator on the track, `pw-top` open to monitor XRUNs. (Recommended: Initial buffer of 128 samples @ 48 kHz).

---

## Block 1 — First Session (Quick Wins, ~10 min)

Objective: first impression — layout, loading, sound, basic controls.

### 1.1 Layout and identity

- [ ] **1.1.1** Open NAM-rs GUI → fixed window **600×275 px** (no host decoration).
- [ ] **1.1.2** Zone 1 (left): turquoise logo `"NAM-rs⚡"`, subtitle `"Neural Amp Modeler"`, version + SIMD badge, `"MODEL"` section header, `[📂 Load Model]` button, model box with dark background; below it (separated by 12px): `"CAB SIM IR"` section header (9pt), `[📂 Load IR]` button, `[🗑 Clear IR]` button (visible only if IR loaded), IR file name display frame (120px wide, dark background).
- [ ] **1.1.3** Zone 2 (center): 3 knobs — **INPUT** (70px, turquoise), **OUTPUT** (70px, turquoise), **GATE** (42px, amber).
- [ ] **1.1.4** Zone 3 (right): **adaptive** VU meter — 1 centered bar (no label) 76px wide when plugin is on a mono track; 2 bars labeled **L** / **R** (36px each) when on a stereo track.
- [ ] **1.1.5** Zone 4 (far right): **BYPASS** toggle with LED and `"ACTIVE"`/`"BYPASSED"` label.
- [ ] **1.1.6** Zone 5 (footer): status bar with RT telemetry (sample rate, latency, DSP load, CPU cycles, last N samples, RT priority, overruns/overloads, flags) and bottom line with model metadata (if loaded).
- [ ] **1.1.7** Zone 5 (footer) far right: small "ℹ" button/icon on the telemetry line for copying diagnostics.
- [ ] **1.1.8** 3 thin vertical separators visible between zones 1–4.

### 1.2 First load and sound

- [ ] **1.2.1** `[📂 Load Model]` → system picker opens without freezing the DAW.
- [ ] **1.2.2** Select `.nam` model → `"Loading"` → `"Loading."` → `"Loading.."` → `"Loading..."` animation → model name. Processed audio audible immediately.
- [ ] **1.2.3** Cancel picker → returns to previous state, button remains clickable.
- [ ] **1.2.4** Drag INPUT knob → volume changes without clicks (zipper noise). Turquoise arc follows smoothly.
- [ ] **1.2.5** Bypass ON → gray LED, `"BYPASSED"` label, audio = clean signal (dry). Bypass OFF → processing resumes without click.

✅ **Quick PASS:** nice GUI, loads model, makes sound, knobs and bypass work.

---

## Block 2 — Validation by Feature

Each section testable after touching the corresponding feature. Self-contained, ~5–15 min each.

---

### 2A — File Picker & Thread Safety

- [ ] **2A.1** Picker open → DAW responsive (drag window, move faders on another track). Playback does not stop.
- [ ] **2A.2** Load a different model on top → name updates, audio changes without stopping playback.
- [ ] **2A.3** Invalid file (`invalid.nam`) → red `"⚠ Load failed"` for ~3s, then returns to previous state. No crash.
- [ ] **2A.4** 0-byte `.nam` file → same error handling.
- [ ] **2A.5** Invalid model with valid model already loaded → previous model preserved, audio not interrupted.
- [ ] **2A.6** (Fender Studio Pro) Confirm no freeze despite limited GUI — generic host parameters work.

---

### 2B — Knobs: Range, Fine-Tune, Reset, Glow

| Knob       | Range             | Default  |
| ---------- | ----------------- | -------- |
| **INPUT**  | −96.0 to +30.0 dB | 0.0 dB   |
| **OUTPUT** | −96.0 to +30.0 dB | 0.0 dB   |
| **GATE**   | −90.0 to −40.0 dB | −70.0 dB |

- [ ] **2B.1** Drag each knob to extremes → tooltip (hover) shows correct value, limits respected.
- [ ] **2B.2** Hover over knob → tooltip with 2 decimal places (e.g.: `"3.50 dB"`). INPUT/OUTPUT: `"X.XX dB"`. GATE: `"X.XX dB (Threshold)"`.
- [ ] **2B.3** **Ctrl+Drag (fine-tune):** same drag distance = ~10× less variation. Ctrl+scroll also 10× slower.
- [ ] **2B.4** **Double-click** on knob → resets to default immediately (INPUT/OUTPUT → 0.0, GATE → −70.0).
- [ ] **2B.5** While dragging → glow (semi-transparent halo) visible on the arc. Disappears on release.
- [ ] **2B.6** Bypass toggle: LED + label toggle instantly. Dry/processed audio without click.

---

### 2C — VU Metering, Peak Hold & Clipping (Mono/Stereo Adaptive)

> **Adaptive meter behavior:** The CLAP plugin processes audio in mono (DSP is always mono, duplicating the Left processed output to the Right channel when inserted on a stereo track). Zone 3 dynamically adapts to the *host track channel configuration* at runtime: on a mono track, it shows **1 centered bar (no label, 76px wide)**; on a stereo track, it shows **2 bars labeled L and R (36px each, separated by 4px)**. The active layout is determined by the presence of a right channel buffer in the audio ports passed by the host during processing. VU meters do **not** change color when the accent color changes.

- [ ] **2C.1** Insert NAM-rs on a **mono DAW track** → Zone 3 displays **1 centered bar without a label** (~76px wide).
- [ ] **2C.2** Insert NAM-rs on a **stereo DAW track** → Zone 3 displays **2 bars labeled L and R** (~36px each, separated by 4px).
- [ ] **2C.3** Feed with dynamic signal → VU bar(s) show tricolor gradient: green (−60 to −12 dB) → yellow (−12 to −3 dB) → red (−3 to +6 dB).
- [ ] **2C.4** Fast transients (pick attack) → bar responds without visual delay (~33 fps).
- [ ] **2C.5** Cause a peak and stop signal → peak hold mark stays ~2s, then decays smoothly.
- [ ] **2C.6** Saturate output (>0 dBFS) → red LED at the top of the meter **persists**. Click on the LED or bar → resets.
- [ ] **2C.7** Feed signal **only to the Left (L) channel** (e.g. hard-panned L or L-only generator) → L meter moves dynamically, R meter remains completely at minimum.
- [ ] **2C.8** Feed signal **only to the Right (R) channel** (e.g. hard-panned R or R-only generator) → R meter moves dynamically, L meter remains completely at minimum.
- [ ] **2C.9** Feed a **symmetric / in-phase signal** (equal level on L and R) → L and R meters move symmetrically (equal peaks and decay).
- [ ] **2C.10** Feed an **asymmetric / panned signal** (different levels on L and R) → L and R meters move independently.
- [ ] **2C.11** **How to fail (reprovação):** The test is failed if meters show identical activity even with asymmetric input (e.g. signal on L-only still moves both meters), or if layout fails to adapt to track configuration.

---

### 2D — Automation & Remote Controls

> **Primary host:** Bitwig Studio.

- [ ] **2D.1** Track in Write/Latch mode → drag INPUT knob in the GUI for ~3s → stop. Automation grid shows smooth curve, no jumps, with anchor points at start/end.
- [ ] **2D.2** Repeat for OUTPUT, GATE, and BYPASS.
- [ ] **2D.3** Draw manual automation ramp for `output_gain_db` → playback: OUTPUT knob in GUI moves smoothly, audio follows without zipper noise.
- [ ] **2D.4** Bitwig Device Panel → 2 pages: **"Main"** (INPUT, OUTPUT, BYPASS) and **"Gate"** (GATE). Bidirectional sync: GUI ↔ Device Panel.
- [ ] **2D.5** (Fender Studio Pro) Move parameters in the host mixer → GUI reflects. Move in GUI → host reflects.

---

### 2E — Dynamic Accent Color

> **Host:** Bitwig Studio (requires `track_info`).

- [ ] **2E.1** Track color changed to red → INPUT/OUTPUT knobs + bypass LED change to red in <100ms. VU meters do **not** change.
- [ ] **2E.2** Change to blue, green → follows.
- [ ] **2E.3** Remove track color → returns to default turquoise (`#00D4AA`).
- [ ] **2E.4** (Fender Studio Pro) Without `track_info` → stays turquoise, no errors.

---

### 2F — Persistence (Save/Reload)

- [ ] **2F.1** Set INPUT=+3.5, OUTPUT=−6.0, GATE=−55.0, model loaded.
- [ ] **2F.2** Save project → close DAW completely → reopen → load project.
- [ ] **2F.3** All parameters preserved at exact values. Model reloaded (name visible). Audio identical.
- [ ] **2F.4** Repeat in Fender Studio Pro.
- [ ] **2F.5** Move model file from its location → reopen project → `"No model loaded"` without crash.

---

### 2G — Drag & Drop + DSP Load Meter

> ⚠️ **Note about Linux (X11):** Drag & Drop support (dragging and dropping files onto the plugin) is currently **unavailable on Linux** due to limitations of the `baseview` windowing library (X11 backend), which does not implement the XDND protocol. There is an open Pull Request ([RustAudio/baseview PR #187](https://github.com/RustAudio/baseview/pull/187)) awaiting acceptance to integrate this functionality in the future. For now, use the `[📂 Load Model]` button on Linux. The drag & drop behavior below should be tested only on supported platforms (e.g.: Windows).

- [ ] **2G.1** Drag `.nam` from file manager onto plugin → overlay `"Drop NAM Model Here ⬇️"` appears. Drop → loads model. (Windows only).
- [ ] **2G.2** Drag `.wav` → overlay appears but ignored on drop. (Windows only).
- [ ] **2G.3** Status bar: `"DSP: XX.X%"` indicator present in telemetry (green <50%, amber 50-80%, red >80%).
- [ ] **2G.4** Hover on DSP Load → tooltip describes real-time usage percentage.
- [ ] **2G.5** **Telemetry cadence:** With steady signal, observe status bar for ≥20s. Telemetry values (sample rate, latency, DSP load) update at a visible interval (~1 Hz) — they do not flicker or update every frame. (G1.T03)

---

### 2H — Parameter Indication

> **Host:** Bitwig Studio.

- [ ] **2H.1** MIDI Learn on INPUT knob → dotted halo with 6 blue dots (`#5e81ac`) around the knob.
- [ ] **2H.2** Automation playback on `output_gain_db` → OUTPUT arc pulses smoothly (alpha 0.3→1.0, ~1s cycle).
- [ ] **2H.3** Manual override (move knob in GUI) during active automation → arc turns amber (`#F5A623`) temporarily. Returns to normal on release.

---

### 2I — Accessibility (Keyboard)

- [ ] **2I.1** Tab → focus cycles: INPUT → OUTPUT → GATE → BYPASS → Load Model → Load IR → INPUT. Focus ring visible.
- [ ] **2I.2** Shift+Tab → reverse order.
- [ ] **2I.3** Focused knob: ↑/→ = +1.0 dB, ↓/← = −1.0 dB. Ctrl+↑ = +0.1 dB, Ctrl+↓ = −0.1 dB. Limits respected.
- [ ] **2I.4** Load Model focused → Space/Enter opens picker.
- [ ] **2I.5** BYPASS focused → Space/Enter toggles bypass.
- [ ] **2I.6** Text contrast OK: `COL_MUTED` readable on `COL_PANEL`, `COL_TEXT` readable on `COL_BG`.

---

### 2J — Real-Time LFO Modulation

> **Host:** Bitwig Studio.

- [ ] **2J.1** LFO on `input_gain_db` at 1–5 Hz, ±6 dB → arc oscillates smoothly, audio without zipper noise.
- [ ] **2J.2** LFO at 20 Hz → audio modulates like a fast tremolo without artifacts or CPU spikes.
- [ ] **2J.3** **Manual gain sweep:** Draw an automation ramp for `input_gain_db` from −60 dB to +12 dB over ~1s → audio transition is smooth, no audible click or zipper noise. (G1.T04)
- [ ] **2J.4** **Repeat gain sweep** with `output_gain_db` — same smoothness, no zipper noise. (G1.T04)

---

### 2K — Dynamic Latency Compensation

- [ ] **2K.1** Change project sample rate (44.1→96 kHz) → status bar updates (`"96kHz"`), latency updates. Bitwig recalculates PDC without desync.
- [ ] **2K.2** Toggle bypass or switch model with different resampling → reported latency changes, Bitwig updates PDC immediately.

---

### 2L — Diagnostics & Copy Support Block

- [ ] **2L.1** Status bar displays a small info button `"ℹ"` on the far right of the telemetry line.
- [ ] **2L.2** Hover over `"ℹ"` button → shows tooltip: `"Copy Diagnostic info to clipboard and ~/.cache/nam-rs/"`.
- [ ] **2L.3** Click `"ℹ"` button → visual toast confirmation `"Diagnostic copied · file in ~/.cache/nam-rs/"` appears in the status bar for 3 seconds.
- [ ] **2L.4** Paste (Ctrl+V) anywhere → diagnostic support block is successfully pasted, containing system info (version, arch, os, kernel, features) and runtime state (model, sample rate).
- [ ] **2L.5** Verify that a diagnostic file was created under `~/.cache/nam-rs/diagnostic-<unix_ts>.txt` with exact permission `0o600` (read/write by owner only).
- [ ] **2L.6** While toast is visible, click `"Open Folder"` button next to it → file manager opens at `~/.cache/nam-rs/` via `xdg-open`.
- [ ] **2L.7** **Failure Scenario (`xdg-open` missing / `HOME` unset):** Simulate absence of `xdg-open` (e.g. temporary PATH modification) or unset `HOME` env variable → clicking `"Open Folder"` degrades gracefully without crashing the plugin or host (silent fallback or friendly warning).
- [ ] **2L.8** **Headless / Server Environment:** In a headless desktop environment (e.g. running the DAW on a Linux server via SSH without an active X11/Wayland display server or D-Bus session) → clicking the button does not cause a crash or UI freeze, failing gracefully.

---

### 2M — Preset Discovery Browser

> **Host:** Bitwig Studio, Reaper (or any host with CLAP preset browser support).
> **Preparation:** At least 2 `.nam` files present in `~/.nam/models/` (or the directories declared by the plugin).

- [ ] **2M.1** Open the host's preset browser for NAM-rs → the plugin's preset entries appear, listing `.nam`/`.namb` files by name (basename).
- [ ] **2M.2** Preset metadata: name, creator (modeled_by), and gear model are displayed (if available in the model file's metadata).
- [ ] **2M.3** Select a preset in the host's browser → the model is loaded into the plugin (verified by `Active Model` parameter update and audible model change).
- [ ] **2M.4** The `model_load_counter` increments on each successful preset load.
- [ ] **2M.5** Loading an invalid/corrupt `.nam` file via the preset browser → error is reported in the host log without crashing.
- [ ] **2M.6** Preset-load is RT-safe: model I/O and building happen on the Main Thread (no allocations in the audio processing thread).

---

### 2N — Floating GUI Fallback Window

> **Host:** Bitwig Studio and Fender Studio Pro.
> **Context:** If the host does not support GUI embedding (X11 parented), the plugin should open as a floating top-level window.

- [ ] **2N.1** **Embedding preference:** On hosts that support X11 parented embedding (Bitwig, Reaper), NAM-rs opens GUI embedded in the host window — not as a separate floating window.
- [ ] **2N.2** **Floating fallback:** On a host that requests floating mode → NAM-rs opens as an independent top-level window. GUI is fully interactive (knobs, bypass, load button all work).
- [ ] **2N.3** **Repeated open/close:** Open and close the floating GUI 10+ times → no crash, no zombie windows, no growing memory.
- [ ] **2N.4** **Playback stability:** Open/close the floating GUI while audio is playing → playback continues, no XRUNs.
- [ ] **2N.5** **Mode logged:** After opening the GUI, check the host log (e.g. `~/.bitwig-studio/log/engine.log`) → message `"NAM-rs: GUI mode selected = embedded"` (or `"floating"`) appears at Info level.
- [ ] **2N.6** **No regression:** On hosts that always used embedded GUI, behavior is identical to before — no visual change, sizing unchanged.

---

### 2O — Plugin Categorization in Host Browser

> **Host:** Bitwig Studio (browser sidebar).
> **Context:** The CLAP feature strings determine where the plugin appears in the host's category tree.

- [ ] **2O.1** Open Bitwig's plugin browser → NAM-rs appears under **Audio FX** → **Distortion** category.
- [ ] **2O.2** In the search/filter bar, typing `"distortion"`, `"gate"`, or `"mono"` finds NAM-rs.
- [ ] **2O.3** The plugin does **not** show up under categories like "Instrument", "Synth", "Delay", or "Reverb".
- [ ] **2O.4** Check: no non-standard category such as `"simulator"` appears (if the host exposes raw features).

---

### 2P — Preset Portability via State-Context

> **Host:** Bitwig Studio (supports `clap.state-context`). Reaper may also work.
> **Preparation:** A `.nam` model loaded, parameters set to non-default values.

- [ ] **2P.1** **Save as device preset:** In Bitwig, right-click NAM-rs → *"Save as Device Preset..."* → enter name → save.
- [ ] **2P.2** **Preset portability:** Move the `.nam` model file to a **different directory**. Delete it from the original location.
- [ ] **2P.3** **Reload preset:** Insert a new NAM-rs instance → load the saved device preset. The model loads from the **new location** (via search paths + basename). Audio parameters (INPUT, OUTPUT, GATE, BYPASS) are restored.
- [ ] **2P.4** **Save project:** Set parameters + model → save DAW project → close → reopen. All state preserved (model loads from absolute path).
- [ ] **2P.5** **Model missing on project load:** Move the model file from its absolute path → reopen project → `"No model loaded"` appears in status bar. No crash.
- [ ] **2P.6** **DSP load meter and telemetry status bar** continue updating correctly after preset/project load.

---

### 2Q — IR CabSim (Impulse Response)

> **Prerequisites:** At least one `.wav` IR file available (mono, 16/24/32-bit PCM or float, commonly 44.1–96 kHz). A second `.wav` IR file for load-change tests. A bogus/invalid `.wav` (truncated or non-WAV data renamed to `.wav`). A `.nam` model already loaded (IR cab sim operates post-inference on the processed signal).
> **Position in pipeline:** Input Gain → Model Inference → Output Gain → **CabSim IR** → Limiter → Bypass/Output.

- [ ] **2Q.1** Zone 1 displays `"CAB SIM IR"` section header (9pt, strong, muted color) below the model section, separated by 12px.
- [ ] **2Q.2** `[📂 Load IR]` button opens system file picker filtered to `.wav` files. DAW remains responsive during file dialog (drag window, move faders).
- [ ] **2Q.3** Select valid mono `.wav` IR → loading animation `"Loading"` / `"Loading."` / `"Loading.."` / `"Loading..."` plays in the IR display frame. After load completes: IR file name (basename) appears, audio undergoes cab-sim convolution (audible difference vs dry signal).
- [ ] **2Q.4** Cancel system picker for IR → returns to previous state. Button remains clickable.
- [ ] **2Q.5** With IR loaded, `[🗑 Clear IR]` button appears below `[📂 Load IR]`. Click it → IR display changes to `"No IR loaded"`, audio returns to post-model without cab sim. Clear IR button disappears.
- [ ] **2Q.6** Load an invalid/corrupt `.wav` → red `"⚠ IR load failed"` displayed in the IR frame for ~3s, then returns to previous state (previous IR preserved if one was loaded). No crash.
- [ ] **2Q.7** Hover over IR name frame → tooltip shows full path. On error, hover shows detailed error message.
- [ ] **2Q.8** Load IR → save project → close DAW completely → reopen → load project. IR path preserved, IR reloaded automatically. Audio identical. Model and IR both restored.
- [ ] **2Q.9** Move `.wav` IR file from its location → reopen project → `"⚠ IR load failed"` appears. Model still loads, audio processed without cab sim.
- [ ] **2Q.10** Load a different IR on top of an existing one → IR name updates, cab-sim audio changes without stopping playback or causing XRUNs.
- [ ] **2Q.11** Load IR, check status bar latency → reported latency includes cab-sim partition size (= buffer size). Toggle bypass or clear IR → latency updates in host PDC.
- [ ] **2Q.12** Change project sample rate (44.1→96 kHz) with IR loaded → IR engine rebuilds for new buffer size; latency updates in status bar; audio continues without desync.
- [ ] **2Q.13** (Fender Studio Pro) Load IR → host parameters reflect IR path. Move IR in host generic params → GUI updates.
- [ ] **2Q.14** (Bitwig) Automation on cab-sim parameters (IR load/clear via host generic params): automation curve shows smooth transitions.
- [ ] **2Q.15** Tab to `[📂 Load IR]` button → Space/Enter opens file picker. Tab/Shift+Tab includes IR button in focus cycle.
- [ ] **2Q.16** Load IR → switch to a different `.nam` model → IR preserved, cab sim active on new model's output.
- [ ] **2Q.17** **Multi-instance:** 2 NAM-rs instances on different tracks. Load IR in instance 1, load different IR in instance 2 → each processes independently.
- [ ] **2Q.18** 3 instances: IR on 1st, no IR on 2nd, load IR on 3rd during playback → no XRUNs, all independent.
- [ ] **2Q.19** **Stress:** Load 10 different IRs in <1 minute with audio running → no crash, no freeze, RSS stable (growth < 2 MB after 10 reloads).
- [ ] **2Q.20** **Bypass null test with IR active:** NAM-rs in bypass + identical signal in parallel with inverted phase + active ADC. Result = silence (<−120 dBFS). Bypass is bit-transparent regardless of cab-sim state.

---

### 2R — Slimmable A2 Container & FSM Degradation

> **Prerequisites:** A `.nam` model packaged as a `"SlimmableContainer"` (e.g. bundling A2-Full and A2-Lite submodels) available. The host DAW is open, and NAM-rs is inserted on a track with audio processing active.
> **Adaptive behavior:** In "Auto" mode, quality switches dynamically between A2-Full and A2-Lite based on CPU headroom/block budget. In manual overrides ("Force Full" / "Force Lite"), the selection is static.

- [ ] **2R.1** Load the slimmable container `.nam` model → loads successfully without errors.
- [ ] **2R.2** Locate the `"Slim Override"` parameter (ID 6) in the host DAW's generic parameter panel.
- [ ] **2R.3** Set `"Slim Override"` to `"Force Full"` → the A2-Full (8-channel) submodel is selected. Audio remains clean.
- [ ] **2R.4** Set `"Slim Override"` to `"Force Lite"` → the submodel swaps to A2-Lite (3-channel) instantly.
- [ ] **2R.5** Verify transition smoothness: Toggle between `"Force Full"` and `"Force Lite"` multiple times during playback → the transition is perfectly seamless and click-free (smoothed by a 32 ms crossfade).
- [ ] **2R.6** Set `"Slim Override"` to `"Auto"` → the control returns to the Adaptive Compute FSM.
- [ ] **2R.7** Simulate CPU pressure (e.g., lower buffer size, run high load) → if the FSM triggers degradation, the status bar displays corresponding degradation flags (`DEGRADE` status).
- [ ] **2R.8** Save the DAW project with `"Slim Override"` set to `"Force Lite"` → close and reopen → parameter value is preserved at `"Force Lite"` and the A2-Lite submodel is active.
- [ ] **2R.9** Render/offline bounce the track → the engine forces full quality, bypasses FSM degradation, and returns a clean render without any active degradation flags.

---

## Block 3 — Stress & Pedantry

Run **after** Blocks 1 and 2 pass. 128 sample buffer @ 48 kHz.

---

### 3.1 Interface Spam

- [ ] **3.1.1** Toggle bypass 20+ consecutive times via GUI. No crash, no artifact.
- [ ] **3.1.2** Open/close GUI 20+ times in <30s with active playback.
- [ ] **3.1.3** (Bitwig) Switch hosting modes (*Together*, *Individually*, *Individually strict*) and repeat spam.

---

### 3.2 Concurrent Fast Load

- [ ] **3.2.1** Load 10 different models in <1 minute, with audio running.
- [ ] **3.2.2** Stable RSS memory (growth <2 MB after 10 reloads). No visible leak.

---

### 3.3 Extreme Modulation

- [ ] **3.3.1** (Bitwig) LFO at 20–100 Hz modulating `input_gain_db` for ≥5 min with 128 sample buffer. Zero zipper noise, zero XRUNs.
- [ ] **3.3.2** (Fender) Channel envelopes/LFOs modulating parameters for ≥5 min.
- [ ] **3.3.3** **Simultaneous automation:** Modulate `input_gain_db` AND `output_gain_db` with independent LFOs (different rates, e.g. 5 Hz and 13 Hz) for ≥2 min at 128 sample buffer. Zero XRUNs in `pw-top`. Audio remains clean. (G1.T01)

---

### 3.4 Gate FSM in Silence

- [ ] **3.4.1** Stop playback for 10s → output = clean silence (no residual noise, no denormals).
- [ ] **3.4.2** Resume playback → audio returns without click, no transient loss.

---

### 3.5 Multi-Instance

- [ ] **3.5.1** With 2 instances processing, add a 3rd during playback → the first 2 are not interrupted.
- [ ] **3.5.2** Delete 3rd instance during playback → the remaining 2 continue.
- [ ] **3.5.3** Open File Picker in 2 instances simultaneously → both work independently.
- [ ] **3.5.4** Close GUI of one instance, keep another open → audio of both continues normally. Reopen GUI → state preserved.
- [ ] **3.5.5** 3 instances: bypass on 1st, active on 2nd, load model on 3rd → each independent.

---

### 3.6 Endurance 1 Hour

- [ ] **3.6.1** Project with 4 instances (2× WaveNet + 2× LSTM), 2 LFOs each, continuous playback for **60 min**.
- [ ] **3.6.2** Monitor every 30s: RSS, file descriptors, threads, XRUNs.
- [ ] **3.6.3** **Acceptance:** zero crashes, RSS stabilizes (variation <5 MB after warmup), zero FD/thread leaks, zero XRUNs.

---

### 3.7 Bypass Null Test

- [ ] **3.7.1** Extra track: NAM-rs in bypass + identical signal in parallel with inverted phase + active ADC.
- [ ] **3.7.2** Result = absolute silence (<−120 dBFS). Bypass is bit-transparent.

---

### 3.8 Offline Bounce Determinism

- [ ] **3.8.1** With active processing, offline bounce 2 consecutive times.
- [ ] **3.8.2** WAV files identical bit-by-bit (`cmp`). **Use offline bounce, not real-time.**
- [ ] **3.8.3** Status bar during offline bounce: DSP flags (in diagnostic info) remain clean — no `DEGRADE` warning. Adaptive compute stays at maximum quality.

---

### 3.9 GUI Idle CPU Reduction — Conditional Render (G3.T01)

> **Prerequisite:** Close all other DAW plugin GUIs. Monitor CPU usage (e.g., `htop` or system monitor) for the DAW process.

- [ ] **3.9.1** **Baseline:** Open NAM-rs editor → play audio for 30s → stop audio. After 5s of silence, note CPU% of the DAW process with editor open and idle.
- [ ] **3.9.2** **Idle behavior:** Wait another 30s with editor open, no audio, no mouse/keyboard interaction. CPU% drops noticeably from active state.
- [ ] **3.9.3** **Interaction resumes rendering:** Move the mouse over the editor, click any control → GUI updates immediately. No "frozen frame" or delay.
- [ ] **3.9.4** **Peak-hold animation:** Play a loud transient, then stop → VU meter peak-hold decays smoothly even in idle mode. Does not "freeze" the peak dot.
- [ ] **3.9.5** **Automation pulse:** With active LFO modulation (see 2J), the knob arc animation pulses continuously without drops.
- [ ] **3.9.6** **Toast/loading animation:** Trigger a model load → `"Loading..."` animation runs smoothly frame-by-frame despite idle-capable render.
- [ ] **3.9.7** **No flicker:** Alternate between moving knobs rapidly and stopping → no screen flicker or tearing when transitioning between active/idle render.

---

### 3.10 GUI Open/Close Stress — No OpenGL Leaks (G3.T02)

> **Prerequisite:** Build NAM-rs with debug logging visible (check DAW logs or run from terminal to see stderr).

- [ ] **3.10.1** **Rapid open/close:** Open and close the NAM-rs editor 30+ times in <60s, with audio playing continuously.
- [ ] **3.10.2** **No leak warnings:** Check DAW logs/terminal output → no `egui_glow` messages containing `"Resources will be leaked!"` or `"leaked"`.
- [ ] **3.10.3** **Memory stability:** After 30 cycles of open/close, DAW process RSS memory is stable (growth < 5 MB from before the test).
- [ ] **3.10.4** **GL resource check:** Open editor → close it → open again → VU meter, knob arcs, and text all render correctly. No "black window" or missing graphics.
- [ ] **3.10.5** **No crash/panic:** After the 30-cycle stress, continue using the plugin normally (load model, adjust knobs) → everything works, no crash.

---

### 3.11 HiDPI First-Frame Scale Correctness (G3.T03)

> **Host:** Test on a HiDPI display (scale factor 1.5 or 2.0, e.g. 4K monitor with 150% or 200% scaling).
> **Also test on a 1.0x display (standard 1080p).**

- [ ] **3.11.1** **HiDPI first frame:** Set system/host to HiDPI scale (≥1.5). Insert NAM-rs → open editor for the **first time**. The GUI renders at correct size — not tiny/blurry. Text is sharp, knobs are proportionate.
- [ ] **3.11.2** **No resize artifact:** The GUI does **not** visibly "jump" or resize itself moments after opening (no late scale correction).
- [ ] **3.11.3** **1.0x regression check:** On a standard 1080p display (scale 1.0), the GUI appears identical to before — same size, same layout, no distortion.
- [ ] **3.11.4** **Manual resize:** After opening, resize the host window (drag corner) → GUI adapts to new size without artifacts.

---

## Release Criteria

- [ ] **RC.1** Zero crashes, panics, or freezes in any operation.
- [ ] **RC.2** Zero XRUNs recorded in `pw-top` during the entire session.
- [ ] **RC.3** Zero audible zipper noise on knobs, automation, or modulation.
- [ ] **RC.4** Stable visual rendering at ~33 fps, no flicker or artifacts.
- [ ] **RC.5** Full workflow: instantiate → load → adjust → save → close → reopen → state preserved.

---

## Bug Report Template

```text
**Test:** <ID, e.g.: 2C.4>
**OS/Kernel:** <e.g.: Ubuntu 24.04, Linux 6.8-lowlatency>
**DAW:** <name and version, e.g.: Bitwig Studio 6.0.6 Flatpak>
**Model:** <.nam file, e.g.: jcm800.nam>
**Buffer/Sample Rate:** <e.g.: 128 samples @ 48 kHz>
**Expected:** <behavior described in the roadmap>
**Observed:** <what actually happened>
**Attachments:** GUI screenshot/video, DAW log, XRUNs from pw-top, Diagnostic support block (copied via "ℹ" button in the status bar).
```
