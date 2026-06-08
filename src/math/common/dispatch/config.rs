// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::detect::SIMD_MATH;
use super::instruction_set::InstructionSet;

// ══════════════════════════════════════════════════════════════════════════════
// DESIGN DEBT: Coexistence of two dispatch mechanisms
// ══════════════════════════════════════════════════════════════════════════════
//
// The NAM-rs project uses TWO independent mechanisms for SIMD dispatch:
//
// 1. Generic trait `SimdMath` (defined in `common/traits.rs`)
//    - Static dispatch (monomorphization) via `dispatch_simd!` Modes 1 and 2
//    - Used by: WaveNet (`wavenet.rs`, `wavenet_dyn.rs`), LSTM (`lstm_dyn.rs`),
//      DSP (`gate.rs`, `resampler.rs`)
//    - Example: `self.process::<Avx2Math>(args)` → monomorphized at compile time
//    - Advantage: zero v-table overhead, aggressive inlining
//    - Disadvantage: generates duplicate code for each ISA (Avx2, Avx512, Avx512Vnni...)
//
// 2. V-table `SimdMathConfig` (this struct)
//    - Dynamic dispatch via function pointers
//    - Used by: DSP operations in the pipeline (`dsp/pipeline.rs`, `dsp/gain.rs`),
//      standalone host (`standalone/rt_setup.rs`), `dispatch_simd!` Mode 3
//    - Example: `(SIMD_MATH.apply_gain)(data, gain)` → indirect call via pointer
//    - Advantage: single code path, no duplication
//    - Disadvantage: prevents inlining, indirection cost (~1-2 cycles)
//
// Consumers by mechanism:
//   Mechanism 1 (trait):  wavenet.rs, wavenet_dyn.rs, lstm_dyn.rs, gate.rs, resampler.rs
//   Mechanism 2 (v-table): pipeline.rs, rt_setup.rs, cli.rs, ops.rs (compute_energy_stereo)
//   Both (hybrid):        lstm.rs (uses dispatch_simd! Mode 2 for gemv_4gate,
//                          but also calls simd_tanh/simd_sigmoid directly)
//
// Unification plan (future):
//   - Move ALL consumers to the `SimdMath` trait (Mechanism 1)
//   - Replace v-table `SimdMathConfig` with a single trait-based dispatch
//   - Remove function pointers from the `SimdMathConfig` struct
//   - Keep `InstructionSet` for capability queries (e.g.: `is_avx512`)
//   - This will eliminate ~50 lines of boilerplate in `detect_best_simd()`
//
// Debt date: 2026-05-12 (Epics 1-5 refactoring)
// Priority: Medium (does not affect performance on hot paths,
//             which already use Mechanism 1 with monomorphization)
// ══════════════════════════════════════════════════════════════════════════════
/// Dynamic dispatch table (v-table) for SIMD operations.
#[derive(Clone, Copy)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub struct SimdMathConfig {
    /// Active instruction set.
    pub instruction_set: InstructionSet,
    /// Friendly backend name.
    pub name: &'static str,
    /// Whether the backend is AVX-512.
    pub is_avx512: bool,
    /// Fused add + GEMV kernel.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub fused_add_gemv: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], bool),
    /// Fused add + batch GEMM kernel.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub fused_add_gemm_batch: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], usize, bool),
    /// Fused residual batch GEMM kernel.
    pub fused_gemm_residual_batch:
        unsafe fn(&[f32], &[u16], &[f32], &[f32], &mut [f32], usize, bool),
    /// GEMV kernel with overwrite.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub gemv_overwrite: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], bool),
    /// Head accumulation kernel.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub accumulate_head: unsafe fn(&mut [f32], &[f32]),
    /// Horizontal sum of a buffer.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub horizontal_sum: unsafe fn(*const f32, usize) -> f32,
    /// Applies gain and detects clipping in stereo.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub apply_gain_and_detect_clipping_stereo: unsafe fn(&mut [f32], &mut [f32], f32) -> bool,
    /// Applies constant gain in stereo (without clipping detection).
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub apply_gain_stereo: unsafe fn(&mut [f32], &mut [f32], f32),
    /// Applies constant gain to a mono buffer.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub apply_gain: unsafe fn(&mut [f32], f32),
    /// Applies a linear gain ramp to a mono buffer.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub apply_ramp: unsafe fn(&mut [f32], f32, f32),
    /// Applies a linear gain ramp in stereo.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub apply_ramp_stereo: unsafe fn(&mut [f32], &mut [f32], f32, f32),
    /// Stereo convolution (used in the resampler).
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub convolve_stereo: unsafe fn(*const f32, *const f32, *const f32, usize) -> (f32, f32),
    /// Mono convolution (used in the resampler).
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub convolve_mono: unsafe fn(*const f32, *const f32, usize) -> f32,
    /// Dual mono convolution (used in the resampler).
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub convolve_mono_dual: unsafe fn(*const f32, *const f32, *const f32, usize) -> (f32, f32),
    /// Computes the maximum energy between two channels.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub compute_energy_stereo: unsafe fn(&[f32], &[f32]) -> f32,
    /// Computes the energy of a block.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub compute_energy: unsafe fn(&[f32]) -> f32,
    /// Computes the maximum absolute difference between two blocks.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub compute_max_diff: unsafe fn(&[f32], &[f32]) -> f32,
    /// Computes the peak absolute value of both stereo channels.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub compute_peak_abs_stereo: unsafe fn(&[f32], &[f32]) -> (f32, f32),
    /// Applies Tanh activation to a slice.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub tanh_slice: unsafe fn(&mut [f32]),
    /// Applies Sigmoid activation to a slice.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub sigmoid_slice: unsafe fn(&mut [f32]),
    /// Applies ReLU activation to a slice.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub relu_slice: unsafe fn(&mut [f32]),
    /// Applies PReLU activation to a slice.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub prelu_slice: unsafe fn(&mut [f32], &[f32]),
    /// Applies Softsign activation to a slice.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub softsign_slice: unsafe fn(&mut [f32]),
    /// Applies SiLU activation to a slice.
    // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
    pub silu_slice: unsafe fn(&mut [f32]),
}

impl SimdMathConfig {
    /// Returns the active global SIMD configuration.
    pub fn current() -> Self {
        *SIMD_MATH
    }

    /// Alias for current().
    pub fn get() -> Self {
        Self::current()
    }
}
