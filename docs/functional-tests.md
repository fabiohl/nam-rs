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
