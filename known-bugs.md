<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Known Bugs — Pending Triage

Findings collected during the `utils/tests-long.sh` audit run of **2026-07-02**
(`testes.log` + `target/logs/phase*.log`). These are **not fixed** — each needs a
domain-owner decision (recalibrate a threshold vs. fix a real regression vs.
deeper investigation) before any code change. Do not "fix" any of these by
relaxing a gate threshold without first confirming the failure is a calibration
gap and not a genuine regression.

Legend: 🔴 correctness/stability — 🟠 quality/calibration — ⚠️ system-safety.

---

## BUG-1 🔴 — `inference_bench` A2 Dynamic benchmark fixture rejected by the loader

- **Component:** `benches/inference_bench.rs` (benchmark harness only — **not** production code)
- **Status:** Blocked Phase 6 siblings until 2026-07-02 (now mitigated at the script level, see below); fixture itself still broken.

**Symptom** (`target/logs/phase6-benchmarks.log:1232`):

```text
thread 'main' panicked at benches/inference_bench.rs:2140:40:
Dispatcher failed for WaveNet A2 Dynamic benchmark: WaveNet model rejected:
Layer 0 is missing or has invalid 'kernel_size' — required for free geometry
WaveNet A1.. Detected: 1 layer(s) with geometry [(4, 0)]
error: bench failed, to rerun pass `--bench inference_bench`
```

**Root cause:** `make_wavenet_a2_dyn_data()` (`benches/inference_bench.rs:749`)
builds a `NamModelData`/`NamLayerConfig` **directly as a Rust struct literal**,
bypassing the JSON parser entirely. The A1-vs-A2 disambiguation guardrail in
`src/loader/nam_json/topology/wavenet.rs` (~line 320–345, "Reject A1 models
with A2-specific features") only inspects `layer.layer_raw` — the **raw JSON
map** — for A2-marking keys (`gating_mode`, `head1x1`, `layer1x1`, FiLM keys).
Since the fixture never populates `layer_raw`, the guardrail sees nothing and
the model falls through to the "free geometry WaveNet A1" path
(`wavenet.rs:~440`), which then rejects it for missing `kernel_size` (singular
— the fixture only sets `kernel_sizes`, plural, an A2-only field).

**Impact:** Benchmark-only. Real `.nam` A2 models loaded from actual JSON files
are unaffected (they go through the real parser, which populates `layer_raw`
correctly). Consumed by `bench_wavenet_a2_dyn_gated_process` →
`A2Dyn_Gated_64samp_48kHz` group only.

**Side effect (mitigated 2026-07-02):** `cargo bench` aborts the entire
invocation on the first panicking bench binary. Because Phase 6 combined 4
benches (`inference_bench`, `dot_4x_bench`, `kahan_conv1d_bench`,
`regression_gate`) into one `cargo bench` command, this panic also silently
prevented `kahan_conv1d_bench` and `regression_gate` from ever running or
being recorded that night. `utils/tests-long.sh` Phase 6 now runs each bench
target as its own isolated `cargo bench` invocation — this specific fixture
bug can no longer take down its siblings, but the fixture itself is still
broken and `A2Dyn_Gated_64samp_48kHz` still won't produce a number.

**Suggested fix direction (not applied):** either (a) populate
`layer.layer_raw` on the synthetic `NamLayerConfig` with the same
A2-signaling keys the real JSON would have (e.g. a `gating_mode` array), or
(b) build the fixture from an in-memory JSON string through the real parser
(`parse_nam_json`) instead of a hand-rolled struct, mirroring how
`bench_a2_full_process` already loads `tests/fixtures/models/wavenet_a2_full.nam`.

**Repro:** `cargo bench --features long_bench --bench inference_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1`

---

## BUG-2 🟠 — `cpp_parity` v2 multi-SR live parity failures (3 models)

**Component:** `tests/cpp_parity.rs` — the full `--ignored` cross-validation
matrix against a live `NeuralAmpModelerCore` render.

**Discovery context:** this exact matrix had never actually executed before
2026-07-02 (the invocation was orphaned in `utils/tests-long.sh` until that
session's fix). These are the *first* real signal ever collected from it —
treat as newly-surfaced, not a regression from a known-good baseline.

Driver logic: `run_v2_multi_sr_impl` (`tests/cpp_parity.rs:580`) attempts each
of `{44100, 48000, 88200, 96000, 192000}` Hz independently via
`catch_unwind` (line 600), then asserts the completed-SR set equals the
expected set (line 651) or that at least one SR completed (line ~638).

### 2a — `live_cross_validation_v2_a2_dynamic_gated` (CH=8)

Fails **only** at 88200/96000/192000 Hz; **passes** at 44100/48000 Hz.

```text
assertion `left == right` failed: Parity validation for 'Live A2 Dynamic
Gated (CH=8) (v2)': completed SRs ({44100, 48000}) != expected SRs
({44100, 48000, 88200, 96000, 192000})
```

Pattern is *consistent* with the documented resampler-inherent spectral-error
growth at high SR (`docs/testing.md` §7 MR-STFT soft-gate rationale) — but the
gate that fails here is **ESR** (Tier 1, hard), which is not currently
SR-relaxed the way MR-STFT is. Needs a domain decision: extend the
high-SR soft-gate treatment to ESR for this model, recalibrate its baseline,
or treat as a real regression in the gated-dynamic + resampler interaction.

### 2b — `live_cross_validation_v2_wavenet_a2_film_lite` (CH=3)

Fails at **all 5** SRs with an almost SR-**invariant** ESR:

| SR (Hz) | ESR         | Threshold |
| -------:| -----------:| ---------:|
| 44100   | 3.065217e-2 | 6.2e-3    |
| 48000   | 3.065188e-2 | 6.2e-3    |
| 88200   | 3.065522e-2 | 6.2e-3    |
| 96000   | 3.065527e-2 | 6.2e-3    |
| 192000  | 3.065540e-2 | 6.2e-3    |

The near-zero variance across SR (Δ ≈ 3e-7) is **not** consistent with a
resampler-artifact explanation (which would scale with SR). This shape
suggests a systematic, SR-independent bug or a threshold that was never
actually calibrated for the FiLM-Lite model (`topology_thresholds()` may be
falling back to a generic A2 baseline that doesn't fit FiLM). Recommend
checking `topology_thresholds()` calibration for this model before assuming
a real fidelity defect.

### 2c — `live_cross_validation_v2_wavenet_a2_film_full` (CH=8)

Fails **only** at 48000 Hz — the *one* SR requiring no resampling — while
44100/88200/96000/192000 (all resampled) **pass**:

```text
assertion `left == right` failed: Parity validation for 'Live WaveNet
A2-FiLM-Full (CH=8) (v2)': completed SRs ({44100, 88200, 96000, 192000})
!= expected SRs ({44100, 48000, 88200, 96000, 192000})
```

This is the inverse of the usual "high-SR-only" pattern and the least
explainable of the three — worth a re-run in isolation
(`cargo test --release --test cpp_parity live_cross_validation_v2_wavenet_a2_film_full -- --ignored --nocapture --test-threads=1`)
before deep investigation, to rule out one-off flakiness (borderline
threshold value flipping pass/fail) versus a deterministic native-SR-only defect.

---

## BUG-3 ⚠️ — `test_x2_aliasing_rejection` hangs indefinitely; reported to have caused a desktop session reset

- **Component:** `src/dsp/oversample_test.rs` (test), exercises
  `OversampleEngine` in `src/dsp/oversample.rs`
- **Status:** Excluded from every automated suite (`docs/testing.md` §4 warning
  box, lines 227–231). **Do not re-enable without resolving this entry first.**
- **Severity: elevated to system-safety** — confirmed hang (>30 s, `timeout`
  killed with exit 124) during the 2026-07-02 audit session; separately, the
  human operator reported that attempting to run this test **caused the GNOME
  desktop session to reset** while investigating this very report.

**What is confirmed:**

- `cargo test --release --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" --ignored --nocapture --test-threads=1` hangs with no output for at least 30 s (never observed to complete).
- The test's own input is trivial: 128 samples, `OversampleFactor::X2`, no loops depend on it beyond `input.len()`.

**What is NOT confirmed (needs dynamic investigation, not yet done):**

- Static review of `OversampleEngine::upsample`/`downsample` and `X2Stage::upsample`/`downsample` (`src/dsp/oversample.rs:192-269`, `339-380`) shows only `for` loops bounded by `input.len()` or small fixed constants (`HB_DELAY=12`, `HB_TAPS=25`, `HB_ODD_COUNT`) over pre-allocated fixed-size ring buffers — **no unbounded loop, recursion, or dynamic allocation is evident in this function's own logic.** The hang's root cause is therefore *not* obviously in the DSP math itself, which raises the possibility of an external interaction (global lock/static init contention, a debug-assertion combinatorial blowup, or an environment-specific issue) rather than a simple algorithmic infinite loop.
- Whether the desktop reset is causally linked to this specific test (e.g. runaway memory/CPU triggering an OOM/compositor crash) or coincidental (concurrent system load) has not been established.

**⚠️ Safety instruction for whoever investigates next:** do **not** run this
test directly on a primary workstation again. Given the reported desktop
reset, wrap any reproduction attempt in hard resource isolation first:
a container/VM or a cgroup with an explicit memory cap, plus
`timeout -s KILL 10 …` at the shell level (not just cargo's own test timeout,
which is what already failed to bound this). Observe with `perf top` /
`strace -f -p <pid>` from a **separate**, already-running terminal so you can
tell CPU-spin apart from a blocked syscall before the primary session is put
at risk again.

---

## Summary Table

| ID     | Severity | Component                                      | Blocks                                                        | Action needed                                                                         |
| ------ | -------- | ---------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| BUG-1  | 🔴       | `benches/inference_bench.rs`                   | `A2Dyn_Gated_64samp_48kHz` bench only (siblings now isolated) | Fix fixture to signal A2 via `layer_raw`, or build from real JSON                     |
| BUG-2a | 🟠       | `tests/cpp_parity.rs`                          | `live_cross_validation_v2_a2_dynamic_gated` (high-SR)         | Decide: extend soft-gate to ESR at high SR, or fix resampler+gated interaction        |
| BUG-2b | 🟠       | `tests/cpp_parity.rs`                          | `live_cross_validation_v2_wavenet_a2_film_lite` (all SR)      | Check `topology_thresholds()` calibration for FiLM-Lite before assuming a real defect |
| BUG-2c | 🟠       | `tests/cpp_parity.rs`                          | `live_cross_validation_v2_wavenet_a2_film_full` (48 kHz only) | Re-run isolated first to rule out flakiness, then investigate native-SR-only path     |
| BUG-3  | ⚠️       | `src/dsp/oversample_test.rs` / `oversample.rs` | Excluded from all suites                                      | Reproduce only under hard resource isolation; root cause not yet in DSP math itself   |
