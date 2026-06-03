// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic dispatch system for SIMD kernels.

use super::traits::SimdMath;
use crate::math::common::{Avx2Math, Avx2VnniMath, Avx512Math, Avx512VnniBf16Math, Avx512VnniMath};
use std::sync::LazyLock;

/// Enumerates the supported instruction sets.
///
/// Note: There is no scalar `Fallback` variant in this enum. The project targets
/// x86-64-v3 (AVX2+FMA) as mandatory. If AVX2 is not detected,
/// `detect_best_simd()` panics at boot (fail-fast).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum InstructionSet {
    /// AVX2 + FMA (x86-64-v3).
    Avx2,
    /// AVX2 + VNNI (Alder Lake+, Zen 4+).
    Avx2Vnni,
    /// AVX-512 Foundation (Skylake-X+, Zen 4+).
    Avx512,
    /// AVX-512 VNNI.
    Avx512Vnni,
    /// AVX-512 VNNI + BF16.
    Avx512VnniBf16,
}

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
    pub fused_add_gemv: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], bool),
    /// Fused add + batch GEMM kernel.
    pub fused_add_gemm_batch: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], usize, bool),
    /// Fused residual batch GEMM kernel.
    pub fused_gemm_residual_batch:
        unsafe fn(&[f32], &[u16], &[f32], &[f32], &mut [f32], usize, bool),
    /// GEMV kernel with overwrite.
    pub gemv_overwrite: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], bool),
    /// Head accumulation kernel.
    pub accumulate_head: unsafe fn(&mut [f32], &[f32]),
    /// Horizontal sum of a buffer.
    pub horizontal_sum: unsafe fn(*const f32, usize) -> f32,
    /// Applies gain and detects clipping in stereo.
    pub apply_gain_and_detect_clipping_stereo: unsafe fn(&mut [f32], &mut [f32], f32) -> bool,
    /// Applies constant gain in stereo (without clipping detection).
    pub apply_gain_stereo: unsafe fn(&mut [f32], &mut [f32], f32),
    /// Applies constant gain to a mono buffer.
    pub apply_gain: unsafe fn(&mut [f32], f32),
    /// Applies a linear gain ramp to a mono buffer.
    pub apply_ramp: unsafe fn(&mut [f32], f32, f32),
    /// Applies a linear gain ramp in stereo.
    pub apply_ramp_stereo: unsafe fn(&mut [f32], &mut [f32], f32, f32),
    /// Stereo convolution (used in the resampler).
    pub convolve_stereo: unsafe fn(*const f32, *const f32, *const f32, usize) -> (f32, f32),
    /// Mono convolution (used in the resampler).
    pub convolve_mono: unsafe fn(*const f32, *const f32, usize) -> f32,
    /// Computes the maximum energy between two channels.
    pub compute_energy_stereo: unsafe fn(&[f32], &[f32]) -> f32,
    /// Computes the energy of a block.
    pub compute_energy: unsafe fn(&[f32]) -> f32,
    /// Computes the maximum absolute difference between two blocks.
    pub compute_max_diff: unsafe fn(&[f32], &[f32]) -> f32,
    /// Computes the peak absolute value of both stereo channels.
    pub compute_peak_abs_stereo: unsafe fn(&[f32], &[f32]) -> (f32, f32),
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

/// Global SIMD configuration instance, detected at system boot.
///
/// Using `LazyLock` ensures the user's CPU is inspected only once,
/// at the moment the first DSP mathematical operation is invoked. After that, the
/// corresponding SIMD configuration struct (with optimized kernel function pointers)
/// is cached in memory and accessed immediately (no real-time checking cost).
pub static SIMD_MATH: LazyLock<SimdMathConfig> = LazyLock::new(detect_best_simd);

/// Inspects the CPU hardware capabilities at runtime and returns the best
/// compatible math function pointer table (SIMD).
///
/// Detection checks supported hardware features using the compiler macro `is_x86_feature_detected!`.
/// We follow a descending priority order, choosing the most advanced instruction set
/// available on the processor where the software is running.
fn detect_best_simd() -> SimdMathConfig {
    #[cfg(target_arch = "x86_64")]
    {
        // 1. AVX-512 with VNNI (Vector Neural Network Instructions) and BF16 (Bfloat16) support.
        // Represents the pinnacle of optimization for latest-generation Intel/AMD processors (e.g.: Xeon Cooper Lake/Sapphire Rapids, AMD EPYC Zen4).
        if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vnni") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx512VnniBf16,
                name: "AVX-512 (VNNI+BF16)",
                is_avx512: true,
                fused_add_gemv: Avx512VnniBf16Math::fused_add_gemv,
                fused_add_gemm_batch: Avx512VnniBf16Math::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx512VnniBf16Math::fused_gemm_residual_batch,
                gemv_overwrite: Avx512VnniBf16Math::gemv_overwrite,
                accumulate_head: Avx512VnniBf16Math::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx512VnniBf16Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512VnniBf16Math::apply_gain_stereo,
                apply_gain: Avx512VnniBf16Math::apply_gain,
                apply_ramp: Avx512VnniBf16Math::apply_ramp,
                apply_ramp_stereo: Avx512VnniBf16Math::apply_ramp_stereo,
                convolve_stereo: Avx512VnniBf16Math::convolve_stereo,
                convolve_mono: Avx512VnniBf16Math::convolve_mono,
                compute_energy_stereo: Avx512VnniBf16Math::compute_energy_stereo,
                compute_energy: Avx512VnniBf16Math::compute_energy,
                compute_max_diff: Avx512VnniBf16Math::compute_max_diff,
                compute_peak_abs_stereo: Avx512VnniBf16Math::compute_peak_abs_stereo,
            };
        }
        // 2. AVX-512 with VNNI support only.
        // Intel Cascade Lake/Ice Lake processors and similar. Accelerates integer/float multiplication and accumulation.
        if is_x86_feature_detected!("avx512vnni") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx512Vnni,
                name: "AVX-512 (VNNI)",
                is_avx512: true,
                fused_add_gemv: Avx512VnniMath::fused_add_gemv,
                fused_add_gemm_batch: Avx512VnniMath::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx512VnniMath::fused_gemm_residual_batch,
                gemv_overwrite: Avx512VnniMath::gemv_overwrite,
                accumulate_head: Avx512VnniMath::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx512VnniMath::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512VnniMath::apply_gain_stereo,
                apply_gain: Avx512VnniMath::apply_gain,
                apply_ramp: Avx512VnniMath::apply_ramp,
                apply_ramp_stereo: Avx512VnniMath::apply_ramp_stereo,
                convolve_stereo: Avx512VnniMath::convolve_stereo,
                convolve_mono: Avx512VnniMath::convolve_mono,
                compute_energy_stereo: Avx512VnniMath::compute_energy_stereo,
                compute_energy: Avx512VnniMath::compute_energy,
                compute_max_diff: Avx512VnniMath::compute_max_diff,
                compute_peak_abs_stereo: Avx512VnniMath::compute_peak_abs_stereo,
            };
        }
        // 3. Basic AVX-512 Foundation (512-bit).
        // Common Intel Skylake-X/Zen4 CPUs. Allows processing 16 32-bit floats in a single CPU instruction.
        if is_x86_feature_detected!("avx512f") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx512,
                name: "AVX-512",
                is_avx512: true,
                fused_add_gemv: Avx512Math::fused_add_gemv,
                fused_add_gemm_batch: Avx512Math::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx512Math::fused_gemm_residual_batch,
                gemv_overwrite: Avx512Math::gemv_overwrite,
                accumulate_head: Avx512Math::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx512Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512Math::apply_gain_stereo,
                apply_gain: Avx512Math::apply_gain,
                apply_ramp: Avx512Math::apply_ramp,
                apply_ramp_stereo: Avx512Math::apply_ramp_stereo,
                convolve_stereo: Avx512Math::convolve_stereo,
                convolve_mono: Avx512Math::convolve_mono,
                compute_energy_stereo: Avx512Math::compute_energy_stereo,
                compute_energy: Avx512Math::compute_energy,
                compute_max_diff: Avx512Math::compute_max_diff,
                compute_peak_abs_stereo: Avx512Math::compute_peak_abs_stereo,
            };
        }
        // 4. AVX2 with VNNI support (AVX-VNNI).
        // Modern CPUs that support VNNI neural network acceleration on 256-bit YMM registers (e.g.: Intel Alder Lake).
        if is_x86_feature_detected!("avxvnni") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx2Vnni,
                name: "AVX2 (VNNI)",
                is_avx512: false,
                fused_add_gemv: Avx2VnniMath::fused_add_gemv,
                fused_add_gemm_batch: Avx2VnniMath::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx2VnniMath::fused_gemm_residual_batch,
                gemv_overwrite: Avx2VnniMath::gemv_overwrite,
                accumulate_head: Avx2VnniMath::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx2(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx2VnniMath::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx2VnniMath::apply_gain_stereo,
                apply_gain: Avx2VnniMath::apply_gain,
                apply_ramp: Avx2VnniMath::apply_ramp,
                apply_ramp_stereo: Avx2VnniMath::apply_ramp_stereo,
                convolve_stereo: Avx2VnniMath::convolve_stereo,
                convolve_mono: Avx2VnniMath::convolve_mono,
                compute_energy_stereo: Avx2VnniMath::compute_energy_stereo,
                compute_energy: Avx2VnniMath::compute_energy,
                compute_max_diff: Avx2VnniMath::compute_max_diff,
                compute_peak_abs_stereo: Avx2VnniMath::compute_peak_abs_stereo,
            };
        }
        // 5. Standard 256-bit AVX2 with FMA (Floating-Point Multiply-Add).
        // The absolute minimum required to run NAM-rs (x86-64-v3 specification).
        // Intel Haswell (2013) or AMD Excavator (2015) processors and newer.
        if is_x86_feature_detected!("avx2") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx2,
                name: "AVX2",
                is_avx512: false,
                fused_add_gemv: Avx2Math::fused_add_gemv,
                fused_add_gemm_batch: Avx2Math::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx2Math::fused_gemm_residual_batch,
                gemv_overwrite: Avx2Math::gemv_overwrite,
                accumulate_head: Avx2Math::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx2(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx2Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx2Math::apply_gain_stereo,
                apply_gain: Avx2Math::apply_gain,
                apply_ramp: Avx2Math::apply_ramp,
                apply_ramp_stereo: Avx2Math::apply_ramp_stereo,
                convolve_stereo: Avx2Math::convolve_stereo,
                convolve_mono: Avx2Math::convolve_mono,
                compute_energy_stereo: Avx2Math::compute_energy_stereo,
                compute_energy: Avx2Math::compute_energy,
                compute_max_diff: Avx2Math::compute_max_diff,
                compute_peak_abs_stereo: Avx2Math::compute_peak_abs_stereo,
            };
        }
    }

    // No compatible instruction set was detected.
    // The project requires x86-64-v3 (AVX2+FMA) as the absolute minimum.
    // Panicking here is intentional: it's better to fail fast at boot
    // than to produce corrupted audio or invoke undefined behavior in the DSP.
    panic!(
        "[NAM-rs] Incompatible CPU: AVX2 not detected.\n\
         This binary requires x86-64-v3 (AVX2 + FMA).\n\
         Run on a processor released after ~2013 (Intel Haswell / AMD Ryzen)."
    );
}
