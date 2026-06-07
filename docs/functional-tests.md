<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Master Roadmap of Functional Tests (by humans) — NAM-rs (CLAP Plugin)

**Audience:** UI/UX specialists and end users.
**Target DAWs:** Bitwig Studio 6+ and Fender Studio Pro 8+ (Linux, Flatpak).
**Preparation:** `~/.clap/nam-rs.clap` installed (Release build), ≥2 `.nam` models available (one of them being a bogus/invalid file), Guitar DI or signal generator on the track, `pw-top` open to monitor XRUNs. (Recommended: Initial buffer of 128 samples @ 48 kHz).

---

## Block 1 — First Session (Quick Wins, ~10 min)

Objective: first impression — layout, loading, sound, basic controls.

### 1.1 Layout and identity

- [ ] Open NAM-rs GUI → fixed window **600×275 px** (no host decoration).
- [ ] Zone 1 (left): turquoise logo `"NAM-rs⚡"`, subtitle `"Neural Amp Modeler"`, version + SIMD badge, `[📂 Load Model]` button, model box with dark background.
- [ ] Zone 2 (center): 3 knobs — **INPUT** (70px, turquoise), **OUTPUT** (70px, turquoise), **GATE** (42px, amber).
- [ ] Zone 3 (right): **adaptive** VU meter — 1 centered bar (no label) 16px wide (mono).
- [ ] Zone 4 (far right): **BYPASS** toggle with LED and `"ACTIVE"`/`"BYPASSED"` label.
- [ ] Zone 5 (footer): status bar with RT telemetry (sample rate, latency, DSP load, CPU cycles, last N samples, RT priority, overruns/overloads, flags) and bottom line with model metadata (if loaded).
- [ ] Zone 5 (footer) far right: small "ℹ" button/icon on the telemetry line for copying diagnostics.
- [ ] 3 thin vertical separators visible between zones 1–4.

### 1.2 First load and sound

- [ ] `[📂 Load Model]` → system picker opens without freezing the DAW.
- [ ] Select `.nam` model → `"Loading"` → `"Loading."` → `"Loading.."` → `"Loading..."` animation → model name. Processed audio audible immediately.
- [ ] Cancel picker → returns to previous state, button remains clickable.
- [ ] Drag INPUT knob → volume changes without clicks (zipper noise). Turquoise arc follows smoothly.
- [ ] Bypass ON → gray LED, `"BYPASSED"` label, audio = clean signal (dry). Bypass OFF → processing resumes without click.

✅ **Quick PASS:** nice GUI, loads model, makes sound, knobs and bypass work.

---

## Block 2 — Validation by Feature

Each section testable after touching the corresponding feature. Self-contained, ~5–15 min each.

---

### 2A — File Picker & Thread Safety

- [ ] Picker open → DAW responsive (drag window, move faders on another track). Playback does not stop.
- [ ] Load a different model on top → name updates, audio changes without stopping playback.
- [ ] Invalid file (`invalid.nam`) → red `"⚠ Load failed"` for ~3s, then returns to previous state. No crash.
- [ ] 0-byte `.nam` file → same error handling.
- [ ] Invalid model with valid model already loaded → previous model preserved, audio not interrupted.
- [ ] (Fender Studio Pro) Confirm no freeze despite limited GUI — generic host parameters work.

---

### 2B — Knobs: Range, Fine-Tune, Reset, Glow

| Knob       | Range             | Default  |
| ---------- | ----------------- | -------- |
| **INPUT**  | −96.0 to +30.0 dB | 0.0 dB   |
| **OUTPUT** | −96.0 to +30.0 dB | 0.0 dB   |
| **GATE**   | −90.0 to −40.0 dB | −70.0 dB |

- [ ] Drag each knob to extremes → tooltip (hover) shows correct value, limits respected.
- [ ] Hover over knob → tooltip with 2 decimal places (e.g.: `"3.50 dB"`). INPUT/OUTPUT: `"X.XX dB"`. GATE: `"X.XX dB (Threshold)"`.
- [ ] **Ctrl+Drag (fine-tune):** same drag distance = ~10× less variation. Ctrl+scroll also 10× slower.
- [ ] **Double-click** on knob → resets to default immediately (INPUT/OUTPUT → 0.0, GATE → −70.0).
- [ ] While dragging → glow (semi-transparent halo) visible on the arc. Disappears on release.
- [ ] Bypass toggle: LED + label toggle instantly. Dry/processed audio without click.

---

### 2C — VU Metering, Peak Hold & Clipping (Mono)

> **Plugin mono behavior:** The CLAP plugin operates strictly in mono (1 channel). Consequently, Zone 3 always displays a single centered meter without a label, regardless of whether it is inserted on a mono or stereo DAW track (where stereo routing/processing is managed by the host).

- [ ] Insert NAM-rs in the DAW → Zone 3 displays **1 centered meter without a label** (16px wide) in an ~76px zone.
- [ ] Feed with dynamic signal → single VU bar with tricolor gradient: green (−60 to −12 dB) → yellow (−12 to −3 dB) → red (−3 to +6 dB).
- [ ] Fast transients (pick attack) → bar responds without visual delay (~33 fps).
- [ ] Cause a peak and stop signal → peak hold mark stays ~2s, then decays smoothly.
- [ ] Saturate output (>0 dBFS) → red LED at the top of the single meter **persists**. Click on the LED or bar → resets.

---

### 2D — Automation & Remote Controls

> **Primary host:** Bitwig Studio.

- [ ] Track in Write/Latch mode → drag INPUT knob in the GUI for ~3s → stop. Automation grid shows smooth curve, no jumps, with anchor points at start/end.
- [ ] Repeat for OUTPUT, GATE, and BYPASS.
- [ ] Draw manual automation ramp for `output_gain_db` → playback: OUTPUT knob in GUI moves smoothly, audio follows without zipper noise.
- [ ] Bitwig Device Panel → 2 pages: **"Main"** (INPUT, OUTPUT, BYPASS) and **"Gate"** (GATE). Bidirectional sync: GUI ↔ Device Panel.
- [ ] (Fender Studio Pro) Move parameters in the host mixer → GUI reflects. Move in GUI → host reflects.

---

### 2E — Dynamic Accent Color

> **Host:** Bitwig Studio (requires `track_info`).

- [ ] Track color changed to red → INPUT/OUTPUT knobs + bypass LED change to red in <100ms. VU meters do **not** change.
- [ ] Change to blue, green → follows.
- [ ] Remove track color → returns to default turquoise (`#00D4AA`).
- [ ] (Fender Studio Pro) Without `track_info` → stays turquoise, no errors.

---

### 2F — Persistence (Save/Reload)

- [ ] Set INPUT=+3.5, OUTPUT=−6.0, GATE=−55.0, model loaded.
- [ ] Save project → close DAW completely → reopen → load project.
- [ ] All parameters preserved at exact values. Model reloaded (name visible). Audio identical.
- [ ] Repeat in Fender Studio Pro.
- [ ] Move model file from its location → reopen project → `"No model loaded"` without crash.

---

### 2G — Drag & Drop + DSP Load Meter

> ⚠️ **Note about Linux (X11):** Drag & Drop support (dragging and dropping files onto the plugin) is currently **unavailable on Linux** due to limitations of the `baseview` windowing library (X11 backend), which does not implement the XDND protocol. There is an open Pull Request ([RustAudio/baseview PR #187](https://github.com/RustAudio/baseview/pull/187)) awaiting acceptance to integrate this functionality in the future. For now, use the `[📂 Load Model]` button on Linux. The drag & drop behavior below should be tested only on supported platforms (e.g.: Windows).

- [ ] Drag `.nam` from file manager onto plugin → overlay `"Drop NAM Model Here ⬇️"` appears. Drop → loads model. (Windows only).
- [ ] Drag `.wav` → overlay appears but ignored on drop. (Windows only).
- [ ] Status bar: `"DSP: XX.X%"` indicator present in telemetry (green <50%, amber 50-80%, red >80%).
- [ ] Hover on DSP Load → tooltip describes real-time usage percentage.
- [ ] **Telemetry cadence:** With steady signal, observe status bar for ≥20s. Telemetry values (sample rate, latency, DSP load) update at a visible interval (~1 Hz) — they do not flicker or update every frame. (G1.T03)

---

### 2H — Parameter Indication

> **Host:** Bitwig Studio.

- [ ] MIDI Learn on INPUT knob → dotted halo with 6 blue dots (`#5e81ac`) around the knob.
- [ ] Automation playback on `output_gain_db` → OUTPUT arc pulses smoothly (alpha 0.3→1.0, ~1s cycle).
- [ ] Manual override (move knob in GUI) during active automation → arc turns amber (`#F5A623`) temporarily. Returns to normal on release.

---

### 2I — Accessibility (Keyboard)

- [ ] Tab → focus cycles: INPUT → OUTPUT → GATE → BYPASS → Load Model → INPUT. Focus ring visible.
- [ ] Shift+Tab → reverse order.
- [ ] Focused knob: ↑/→ = +1.0 dB, ↓/← = −1.0 dB. Ctrl+↑ = +0.1 dB, Ctrl+↓ = −0.1 dB. Limits respected.
- [ ] Load Model focused → Space/Enter opens picker.
- [ ] BYPASS focused → Space/Enter toggles bypass.
- [ ] Text contrast OK: `COL_MUTED` readable on `COL_PANEL`, `COL_TEXT` readable on `COL_BG`.

---

### 2J — Real-Time LFO Modulation

> **Host:** Bitwig Studio.

- [ ] LFO on `input_gain_db` at 1–5 Hz, ±6 dB → arc oscillates smoothly, audio without zipper noise.
- [ ] LFO at 20 Hz → audio modulates like a fast tremolo without artifacts or CPU spikes.
- [ ] **Manual gain sweep:** Draw an automation ramp for `input_gain_db` from −60 dB to +12 dB over ~1s → audio transition is smooth, no audible click or zipper noise. (G1.T04)
- [ ] **Repeat gain sweep** with `output_gain_db` — same smoothness, no zipper noise. (G1.T04)

---

### 2K — Dynamic Latency Compensation

- [ ] Change project sample rate (44.1→96 kHz) → status bar updates (`"96kHz"`), latency updates. Bitwig recalculates PDC without desync.
- [ ] Toggle bypass or switch model with different resampling → reported latency changes, Bitwig updates PDC immediately.

---

### 2L — Diagnostics & Copy Support Block

- [ ] Status bar displays a small info button `"ℹ"` on the far right of the telemetry line.
- [ ] Hover over `"ℹ"` button → shows tooltip: `"Copy Diagnostic info to clipboard and ~/.cache/nam-rs/"`.
- [ ] Click `"ℹ"` button → visual toast confirmation `"Diagnostic copiado · arquivo em ~/.cache/nam-rs/"` appears in the status bar for 3 seconds.
- [ ] Paste (Ctrl+V) anywhere → diagnostic support block is successfully pasted, containing system info (version, arch, os, kernel, features) and runtime state (model, sample rate).
- [ ] Verify that a diagnostic file was created under `~/.cache/nam-rs/diagnostic-<unix_ts>.txt` with exact permission `0o600` (read/write by owner only).
- [ ] While toast is visible, click `"Open Folder"` button next to it → file manager opens at `~/.cache/nam-rs/` via `xdg-open`.

---

### 2M — Preset Discovery Browser

> **Host:** Bitwig Studio, Reaper (or any host with CLAP preset browser support).
> **Preparation:** At least 2 `.nam` files present in `~/.nam/models/` (or the directories declared by the plugin).

- [ ] Open the host's preset browser for NAM-rs → the plugin's preset entries appear, listing `.nam`/`.namb` files by name (basename).
- [ ] Preset metadata: name, creator (modeled_by), and gear model are displayed (if available in the model file's metadata).
- [ ] Select a preset in the host's browser → the model is loaded into the plugin (verified by `Active Model` parameter update and audible model change).
- [ ] The `model_load_counter` increments on each successful preset load.
- [ ] Loading an invalid/corrupt `.nam` file via the preset browser → error is reported in the host log without crashing.
- [ ] Preset-load is RT-safe: model I/O and building happen on the Main Thread (no allocations in the audio processing thread).

---

### 2N — Floating GUI Fallback Window (G2.T02)

> **Host:** Bitwig Studio and Fender Studio Pro.
> **Context:** If the host does not support GUI embedding (X11 parented), the plugin should open as a floating top-level window.

- [ ] **Embedding preference:** On hosts that support X11 parented embedding (Bitwig, Reaper), NAM-rs opens GUI embedded in the host window — not as a separate floating window.
- [ ] **Floating fallback:** On a host that requests floating mode → NAM-rs opens as an independent top-level window. GUI is fully interactive (knobs, bypass, load button all work).
- [ ] **Repeated open/close:** Open and close the floating GUI 10+ times → no crash, no zombie windows, no growing memory.
- [ ] **Playback stability:** Open/close the floating GUI while audio is playing → playback continues, no XRUNs.
- [ ] **Mode logged:** After opening the GUI, check the host log (e.g. `~/.bitwig-studio/log/engine.log`) → message `"NAM-rs: GUI mode selected = embedded"` (or `"floating"`) appears at Info level.
- [ ] **No regression:** On hosts that always used embedded GUI, behavior is identical to before — no visual change, sizing unchanged.

---

### 2O — Plugin Categorization in Host Browser (G2.T03)

> **Host:** Bitwig Studio (browser sidebar).
> **Context:** The CLAP feature strings determine where the plugin appears in the host's category tree.

- [ ] Open Bitwig's plugin browser → NAM-rs appears under **Audio FX** → **Distortion** category.
- [ ] In the search/filter bar, typing `"distortion"`, `"gate"`, or `"mono"` finds NAM-rs.
- [ ] The plugin does **not** show up under categories like "Instrument", "Synth", "Delay", or "Reverb".
- [ ] Check: no non-standard category such as `"simulator"` appears (if the host exposes raw features).

---

### 2P — Preset Portability via State-Context (G2.T04)

> **Host:** Bitwig Studio (supports `clap.state-context`). Reaper may also work.
> **Preparation:** A `.nam` model loaded, parameters set to non-default values.

- [ ] **Save as device preset:** In Bitwig, right-click NAM-rs → *"Save as Device Preset..."* → enter name → save.
- [ ] **Preset portability:** Move the `.nam` model file to a **different directory**. Delete it from the original location.
- [ ] **Reload preset:** Insert a new NAM-rs instance → load the saved device preset. The model loads from the **new location** (via search paths + basename). Audio parameters (INPUT, OUTPUT, GATE, BYPASS) are restored.
- [ ] **Save project:** Set parameters + model → save DAW project → close → reopen. All state preserved (model loads from absolute path).
- [ ] **Model missing on project load:** Move the model file from its absolute path → reopen project → `"No model loaded"` appears in status bar. No crash.
- [ ] **DSP load meter and telemetry status bar** continue updating correctly after preset/project load.

---

## Block 3 — Stress & Pedantry

Run **after** Blocks 1 and 2 pass. 128 sample buffer @ 48 kHz.

---

### 3.1 Interface Spam

- [ ] Toggle bypass 20+ consecutive times via GUI. No crash, no artifact.
- [ ] Open/close GUI 20+ times in <30s with active playback.
- [ ] (Bitwig) Switch hosting modes (*Together*, *Individually*, *Individually strict*) and repeat spam.

---

### 3.2 Concurrent Fast Load

- [ ] Load 10 different models in <1 minute, with audio running.
- [ ] Stable RSS memory (growth <2 MB after 10 reloads). No visible leak.

---

### 3.3 Extreme Modulation

- [ ] (Bitwig) LFO at 20–100 Hz modulating `input_gain_db` for ≥5 min with 128 sample buffer. Zero zipper noise, zero XRUNs.
- [ ] (Fender) Channel envelopes/LFOs modulating parameters for ≥5 min.
- [ ] **Simultaneous automation:** Modulate `input_gain_db` AND `output_gain_db` with independent LFOs (different rates, e.g. 5 Hz and 13 Hz) for ≥2 min at 128 sample buffer. Zero XRUNs in `pw-top`. Audio remains clean. (G1.T01)

---

### 3.4 Gate FSM in Silence

- [ ] Stop playback for 10s → output = clean silence (no residual noise, no denormals).
- [ ] Resume playback → audio returns without click, no transient loss.

---

### 3.5 Multi-Instance

- [ ] With 2 instances processing, add a 3rd during playback → the first 2 are not interrupted.
- [ ] Delete 3rd instance during playback → the remaining 2 continue.
- [ ] Open File Picker in 2 instances simultaneously → both work independently.
- [ ] Close GUI of one instance, keep another open → audio of both continues normally. Reopen GUI → state preserved.
- [ ] 3 instances: bypass on 1st, active on 2nd, load model on 3rd → each independent.

---

### 3.6 Endurance 1 Hour

- [ ] Project with 4 instances (2× WaveNet + 2× LSTM), 2 LFOs each, continuous playback for **60 min**.
- [ ] Monitor every 30s: RSS, file descriptors, threads, XRUNs.
- [ ] **Acceptance:** zero crashes, RSS stabilizes (variation <5 MB after warmup), zero FD/thread leaks, zero XRUNs.

---

### 3.7 Bypass Null Test

- [ ] Extra track: NAM-rs in bypass + identical signal in parallel with inverted phase + active ADC.
- [ ] Result = absolute silence (<−120 dBFS). Bypass is bit-transparent.

---

### 3.8 Offline Bounce Determinism

- [ ] With active processing, offline bounce 2 consecutive times.
- [ ] WAV files identical bit-by-bit (`cmp`). **Use offline bounce, not real-time.**
- [ ] Status bar during offline bounce: DSP flags (in diagnostic info) remain clean — no `DEGRADE` warning. Adaptive compute stays at maximum quality.

---

### 3.9 GUI Idle CPU Reduction — Conditional Render (G3.T01)

> **Prerequisite:** Close all other DAW plugin GUIs. Monitor CPU usage (e.g., `htop` or system monitor) for the DAW process.

- [ ] **Baseline:** Open NAM-rs editor → play audio for 30s → stop audio. After 5s of silence, note CPU% of the DAW process with editor open and idle.
- [ ] **Idle behavior:** Wait another 30s with editor open, no audio, no mouse/keyboard interaction. CPU% drops noticeably from active state.
- [ ] **Interaction resumes rendering:** Move the mouse over the editor, click any control → GUI updates immediately. No "frozen frame" or delay.
- [ ] **Peak-hold animation:** Play a loud transient, then stop → VU meter peak-hold decays smoothly even in idle mode. Does not "freeze" the peak dot.
- [ ] **Automation pulse:** With active LFO modulation (see 2J), the knob arc animation pulses continuously without drops.
- [ ] **Toast/loading animation:** Trigger a model load → `"Loading..."` animation runs smoothly frame-by-frame despite idle-capable render.
- [ ] **No flicker:** Alternate between moving knobs rapidly and stopping → no screen flicker or tearing when transitioning between active/idle render.

---

### 3.10 GUI Open/Close Stress — No OpenGL Leaks (G3.T02)

> **Prerequisite:** Build NAM-rs with debug logging visible (check DAW logs or run from terminal to see stderr).

- [ ] **Rapid open/close:** Open and close the NAM-rs editor 30+ times in <60s, with audio playing continuously.
- [ ] **No leak warnings:** Check DAW logs/terminal output → no `egui_glow` messages containing `"Resources will be leaked!"` or `"leaked"`.
- [ ] **Memory stability:** After 30 cycles of open/close, DAW process RSS memory is stable (growth < 5 MB from before the test).
- [ ] **GL resource check:** Open editor → close it → open again → VU meter, knob arcs, and text all render correctly. No "black window" or missing graphics.
- [ ] **No crash/panic:** After the 30-cycle stress, continue using the plugin normally (load model, adjust knobs) → everything works, no crash.

---

### 3.11 HiDPI First-Frame Scale Correctness (G3.T03)

> **Host:** Test on a HiDPI display (scale factor 1.5 or 2.0, e.g. 4K monitor with 150% or 200% scaling).
> **Also test on a 1.0x display (standard 1080p).**

- [ ] **HiDPI first frame:** Set system/host to HiDPI scale (≥1.5). Insert NAM-rs → open editor for the **first time**. The GUI renders at correct size — not tiny/blurry. Text is sharp, knobs are proportionate.
- [ ] **No resize artifact:** The GUI does **not** visibly "jump" or resize itself moments after opening (no late scale correction).
- [ ] **1.0x regression check:** On a standard 1080p display (scale 1.0), the GUI appears identical to before — same size, same layout, no distortion.
- [ ] **Manual resize:** After opening, resize the host window (drag corner) → GUI adapts to new size without artifacts.

---

## Release Criteria

- [ ] Zero crashes, panics, or freezes in any operation.
- [ ] Zero XRUNs recorded in `pw-top` during the entire session.
- [ ] Zero audible zipper noise on knobs, automation, or modulation.
- [ ] Stable visual rendering at ~33 fps, no flicker or artifacts.
- [ ] Full workflow: instantiate → load → adjust → save → close → reopen → state preserved.

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
