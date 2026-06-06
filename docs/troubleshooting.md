<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Troubleshooting & Diagnostics Guide

This document describes how to generate diagnostic information for NAM-rs when experiencing issues (e.g. dropouts, audio glitches, loading issues, or crashes) and how to share it for support.

## 🔍 How to Obtain Support Information

NAM-rs provides multiple ways to capture a **Diagnostic Support Bundle** depending on how you are running it:

### 1. Standalone Mode (PipeWire)

- **Immediate Command Line Flag:**
  Run the binary with the `--diagnose` or `--diagnose-full` flags. This prints the diagnostic support block immediately to `stdout` and exits:

  ```bash
  target/release/nam-rs --diagnose
  ```

- **Interactive Shell Session:**
  If you are running an active audio session, type `:diag` (or alias `:support`) in the interactive console. To print raw absolute paths, type `:diag --full` (or `:support --full`):

  ```text
  💡 Interactive console started. Type ':diag' or ':support' for diagnostics.
  :diag
  ```

### 2. CLAP Plugin Mode (DAW)

- Click the **"Copy Diagnostic"** button or information icon (`ℹ`) in the GUI status bar/About zone.
- This copies the bundle to your system clipboard and also writes it to `~/.cache/nam-rs/diagnostic-<timestamp>.txt` for persistence.

### 3. Application Crash Reports

- If the standalone binary or CLAP plugin encounters a panic, a panic hook automatically intercepts it, captures a diagnostic bundle, and saves it to:
  `~/.cache/nam-rs/crash-<timestamp>-<component>.txt`
- These files are saved with strict owner-only permissions (`0o600`).

---

## 🔒 Redaction & Privacy Policy

To protect your privacy when sharing logs publicly (such as on GitHub Issues), NAM-rs follows a strict path sanitization policy by default:

### What is REDACTED by default

- Absolute home paths (e.g., `/home/username/...` is replaced with `~/...`).
- XDG runtime directory paths (replaced with `$XDG_RUNTIME_DIR`).
- Model file paths are shortened to their filename basename only (e.g., `/home/user/my_secret_path/model.nam` becomes `model.nam`).
- No audio content, neural network weights, or system user/hostnames are ever recorded.

### What is UNREDACTED (using `--diagnose-full` or `:diag --full`)

- Full absolute file paths for models and environment directories are printed without replacements. Useful for resolving complex path/permission issues.

---

## 🤝 How to Report Issues

If you encounter issues and need support:

1. Generate the diagnostic bundle using one of the methods above.
2. **For GitHub Issues:** Copy and paste the entire block (including the `──── NAM-rs Diagnostic ...` headers) into your issue description.
3. **For Automated Support (AI agents):** Copy and paste the support block into the chat to trigger the automated `diagnostico` triage skill (linked with `.agents/workflows/diagnostico.md`).

---

## 📄 Example Diagnostic Bundle

A nominal diagnostic block looks like this:

```text
──── NAM-rs Diagnostic ────────────────────────────────────────────────
nam-rs v1.6.0
──── Runtime State ─────────────────────────────────────────────
model=NEVE1073-Standard.nam
sample_rate=48000
arch=x86_64
os=linux kernel=7.0.0-22-generic
pipewire=1.6.2
features=none (baseline x86-64-v3 only)
timestamp=2026-06-06T19:24:27Z
────────────────────────────────────────────────
Copy the block above when opening a support ticket.
```
