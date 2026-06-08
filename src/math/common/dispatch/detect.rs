// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::config::SimdMathConfig;
use super::instruction_set::InstructionSet;
use crate::math::common::traits::SimdMath;
use crate::math::common::{Avx2Math, Avx2VnniMath, Avx512Math, Avx512VnniBf16Math, Avx512VnniMath};
use std::sync::LazyLock;

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
                // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_mono:
                    Avx512VnniBf16Math::apply_gain_and_detect_clipping_mono,
                apply_gain_and_detect_clipping_stereo:
                    Avx512VnniBf16Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512VnniBf16Math::apply_gain_stereo,
                apply_gain: Avx512VnniBf16Math::apply_gain,
                apply_ramp: Avx512VnniBf16Math::apply_ramp,
                apply_ramp_stereo: Avx512VnniBf16Math::apply_ramp_stereo,
                apply_dither_add: Avx512VnniBf16Math::apply_dither_add,
                convolve_stereo: Avx512VnniBf16Math::convolve_stereo,
                convolve_mono: Avx512VnniBf16Math::convolve_mono,
                convolve_mono_dual: Avx512VnniBf16Math::convolve_mono_dual,
                compute_energy_stereo: Avx512VnniBf16Math::compute_energy_stereo,
                compute_energy: Avx512VnniBf16Math::compute_energy,
                compute_max_diff: Avx512VnniBf16Math::compute_max_diff,
                compute_peak_abs_stereo: Avx512VnniBf16Math::compute_peak_abs_stereo,
                tanh_slice: crate::math::activations::tanh_slice_avx512,
                sigmoid_slice: crate::math::activations::sigmoid_slice_avx512,
                relu_slice: crate::math::activations::relu_slice_avx512,
                prelu_slice: crate::math::activations::prelu_slice_avx512,
                softsign_slice: crate::math::activations::softsign_slice_avx512,
                silu_slice: crate::math::activations::silu_slice_avx512,
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
                // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_mono:
                    Avx512VnniMath::apply_gain_and_detect_clipping_mono,
                apply_gain_and_detect_clipping_stereo:
                    Avx512VnniMath::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512VnniMath::apply_gain_stereo,
                apply_gain: Avx512VnniMath::apply_gain,
                apply_ramp: Avx512VnniMath::apply_ramp,
                apply_ramp_stereo: Avx512VnniMath::apply_ramp_stereo,
                apply_dither_add: Avx512VnniMath::apply_dither_add,
                convolve_stereo: Avx512VnniMath::convolve_stereo,
                convolve_mono: Avx512VnniMath::convolve_mono,
                convolve_mono_dual: Avx512VnniMath::convolve_mono_dual,
                compute_energy_stereo: Avx512VnniMath::compute_energy_stereo,
                compute_energy: Avx512VnniMath::compute_energy,
                compute_max_diff: Avx512VnniMath::compute_max_diff,
                compute_peak_abs_stereo: Avx512VnniMath::compute_peak_abs_stereo,
                tanh_slice: crate::math::activations::tanh_slice_avx512,
                sigmoid_slice: crate::math::activations::sigmoid_slice_avx512,
                relu_slice: crate::math::activations::relu_slice_avx512,
                prelu_slice: crate::math::activations::prelu_slice_avx512,
                softsign_slice: crate::math::activations::softsign_slice_avx512,
                silu_slice: crate::math::activations::silu_slice_avx512,
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
                // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_mono:
                    Avx512Math::apply_gain_and_detect_clipping_mono,
                apply_gain_and_detect_clipping_stereo:
                    Avx512Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512Math::apply_gain_stereo,
                apply_gain: Avx512Math::apply_gain,
                apply_ramp: Avx512Math::apply_ramp,
                apply_ramp_stereo: Avx512Math::apply_ramp_stereo,
                apply_dither_add: Avx512Math::apply_dither_add,
                convolve_stereo: Avx512Math::convolve_stereo,
                convolve_mono: Avx512Math::convolve_mono,
                convolve_mono_dual: Avx512Math::convolve_mono_dual,
                compute_energy_stereo: Avx512Math::compute_energy_stereo,
                compute_energy: Avx512Math::compute_energy,
                compute_max_diff: Avx512Math::compute_max_diff,
                compute_peak_abs_stereo: Avx512Math::compute_peak_abs_stereo,
                tanh_slice: crate::math::activations::tanh_slice_avx512,
                sigmoid_slice: crate::math::activations::sigmoid_slice_avx512,
                relu_slice: crate::math::activations::relu_slice_avx512,
                prelu_slice: crate::math::activations::prelu_slice_avx512,
                softsign_slice: crate::math::activations::softsign_slice_avx512,
                silu_slice: crate::math::activations::silu_slice_avx512,
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
                // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx2(ptr, len)
                },
                apply_gain_and_detect_clipping_mono:
                    Avx2VnniMath::apply_gain_and_detect_clipping_mono,
                apply_gain_and_detect_clipping_stereo:
                    Avx2VnniMath::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx2VnniMath::apply_gain_stereo,
                apply_gain: Avx2VnniMath::apply_gain,
                apply_ramp: Avx2VnniMath::apply_ramp,
                apply_ramp_stereo: Avx2VnniMath::apply_ramp_stereo,
                apply_dither_add: Avx2VnniMath::apply_dither_add,
                convolve_stereo: Avx2VnniMath::convolve_stereo,
                convolve_mono: Avx2VnniMath::convolve_mono,
                convolve_mono_dual: Avx2VnniMath::convolve_mono_dual,
                compute_energy_stereo: Avx2VnniMath::compute_energy_stereo,
                compute_energy: Avx2VnniMath::compute_energy,
                compute_max_diff: Avx2VnniMath::compute_max_diff,
                compute_peak_abs_stereo: Avx2VnniMath::compute_peak_abs_stereo,
                tanh_slice: crate::math::activations::tanh_slice_avx2,
                sigmoid_slice: crate::math::activations::sigmoid_slice_avx2,
                relu_slice: crate::math::activations::relu_slice_avx2,
                prelu_slice: crate::math::activations::prelu_slice_avx2,
                softsign_slice: crate::math::activations::softsign_slice_avx2,
                silu_slice: crate::math::activations::silu_slice_avx2,
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
                // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx2(ptr, len)
                },
                apply_gain_and_detect_clipping_mono: Avx2Math::apply_gain_and_detect_clipping_mono,
                apply_gain_and_detect_clipping_stereo:
                    Avx2Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx2Math::apply_gain_stereo,
                apply_gain: Avx2Math::apply_gain,
                apply_ramp: Avx2Math::apply_ramp,
                apply_ramp_stereo: Avx2Math::apply_ramp_stereo,
                apply_dither_add: Avx2Math::apply_dither_add,
                convolve_stereo: Avx2Math::convolve_stereo,
                convolve_mono: Avx2Math::convolve_mono,
                convolve_mono_dual: Avx2Math::convolve_mono_dual,
                compute_energy_stereo: Avx2Math::compute_energy_stereo,
                compute_energy: Avx2Math::compute_energy,
                compute_max_diff: Avx2Math::compute_max_diff,
                compute_peak_abs_stereo: Avx2Math::compute_peak_abs_stereo,
                tanh_slice: crate::math::activations::tanh_slice_avx2,
                sigmoid_slice: crate::math::activations::sigmoid_slice_avx2,
                relu_slice: crate::math::activations::relu_slice_avx2,
                prelu_slice: crate::math::activations::prelu_slice_avx2,
                softsign_slice: crate::math::activations::softsign_slice_avx2,
                silu_slice: crate::math::activations::silu_slice_avx2,
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
