// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic dispatch system for SIMD kernels.

/// V-table `SimdMathConfig` struct and its methods.
pub mod config;
/// CPU feature detection and global `SIMD_MATH` initialization.
pub mod detect;
/// `InstructionSet` enum for supported x86-64 ISA variants.
pub mod instruction_set;

pub use config::SimdMathConfig;
pub use detect::SIMD_MATH;
pub use instruction_set::InstructionSet;

/// Generates a `SimdMathConfig` v-table for a given ISA.
///
/// Prevents duplication of the ~35-field struct literal across multiple
/// `InstructionSet` variants in `detect_best_simd()`.
///
/// # Parameters
/// - `$type`: The SIMD math type implementing `SimdMath` (e.g., `Avx2Math`).
/// - `$isa`: The `InstructionSet` variant.
/// - `$name`: Friendly backend name.
/// - `$avx512`: Whether the backend is AVX-512 (`true`/`false`).
/// - `$hs`: Horizontal sum utility function name (e.g., `horizontal_sum_avx2`).
/// - `{ $field: $val, ... }`: Activation and other variant-specific function pointers.
#[macro_export]
macro_rules! config_table {
    ($type:ty, $isa:expr, $name:expr, $avx512:expr,
     $hs:ident,
     {$($field:ident : $val:expr),* $(,)?}
    ) => {
        $crate::math::common::dispatch::config::SimdMathConfig {
            instruction_set: $isa,
            name: $name,
            is_avx512: $avx512,
            fused_add_gemv: <$type>::fused_add_gemv,
            fused_add_gemm_batch: <$type>::fused_add_gemm_batch,
            fused_gemm_residual_batch: <$type>::fused_gemm_residual_batch,
            gemv_overwrite: <$type>::gemv_overwrite,
            accumulate_head: <$type>::accumulate_head,
            horizontal_sum: |ptr, len| {
                // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
                unsafe { $crate::math::common::utility::$hs(ptr, len) }
            },
            apply_gain_and_detect_clipping_mono: <$type>::apply_gain_and_detect_clipping_mono,
            apply_gain_and_detect_clipping_stereo: <$type>::apply_gain_and_detect_clipping_stereo,
            apply_gain_stereo: <$type>::apply_gain_stereo,
            apply_gain: <$type>::apply_gain,
            apply_ramp: <$type>::apply_ramp,
            apply_ramp_stereo: <$type>::apply_ramp_stereo,
            crossfade_blend_mono: <$type>::crossfade_blend_mono,
            apply_dither_add: <$type>::apply_dither_add,
            convolve_stereo: <$type>::convolve_stereo,
            convolve_mono: <$type>::convolve_mono,
            convolve_mono_dual: <$type>::convolve_mono_dual,
            compute_energy_stereo: <$type>::compute_energy_stereo,
            compute_energy: <$type>::compute_energy,
            compute_max_diff: <$type>::compute_max_diff,
            compute_peak_abs_stereo: <$type>::compute_peak_abs_stereo,
            compute_peak_abs_mono: <$type>::compute_peak_abs_mono,
            $($field: $val),*
        }
    };
}
