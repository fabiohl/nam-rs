---
name: debugger
description: Triggered when something doesn't work as expected. Acts as a multi-disciplinary panel of senior analysts and engineers capable of solving the hardest problems.
---

# Skill: Debugger

## When to use this skill

Use under active error, crash, non-linear DSP behavior, truncated audio, PipeWire xruns, or degradation in neural inference calculations in real-time.

Also triggered by the `diagnostico` workflow when the user pastes a support block.

## Instructions

### Primary References

* Error codes `Exxxx` and support block format: `src/common/diagnostics/error_codes.rs` (`NamErrorCode`), `src/common/diagnostics/diagnostic.rs` (`NamDiagnostic`).
* RT-safe and DSP guidelines: `.agents/rules/rust.md`.
* General architecture: `docs/architecture.md`.

### Evidence-Based Diagnosis

* **Xruns / clicks**: Check if `process()` respects the time budget (no `Vec`, `Box`, `println!`, or locks).
* **Numerical degradation**: Validate gain multipliers and resampling metadata vs host sample rates.
* **Deadlock / contention**: Inspect `rtrb` usage — no blocking primitives on the RT path.

### Surgical Intervention

* Maintain `#[repr(align(128))]` on structures shared via SPSC.
* After fix, remove **all** logging lines, `dbg!()`, or `eprintln!` inserted for debugging in the RT callback.
* If the fix introduces new failure points outside RT, use the `NamDiagnostic` system to emit structured errors.
