// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::config::SimdMathConfig;
use super::instruction_set::InstructionSet;
use crate::config_table;
use crate::math::common::traits::SimdMath;
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
            return config_table!(
                crate::math::common::Avx512VnniBf16Math,
                InstructionSet::Avx512VnniBf16,
                "AVX-512 (VNNI+BF16)",
                true,
                horizontal_sum_avx512,
                {
                    tanh_slice: crate::math::activations::tanh_slice_avx512,
                    sigmoid_slice: crate::math::activations::sigmoid_slice_avx512,
                    relu_slice: crate::math::activations::relu_slice_avx512,
                    prelu_slice: crate::math::activations::prelu_slice_avx512,
                    softsign_slice: crate::math::activations::softsign_slice_avx512,
                    silu_slice: crate::math::activations::silu_slice_avx512,
                    hard_tanh_slice: crate::math::activations::hard_tanh_slice_avx512,
                    hard_swish_slice: crate::math::activations::hard_swish_slice_avx512,
                    fast_tanh_slice: crate::math::activations::fast_tanh_slice_avx512,
                    leaky_hard_tanh_slice: crate::math::activations::leaky_hard_tanh_slice_avx512,
                }
            );
        }
        // 2. Basic AVX-512 Foundation (512-bit).
        // Common Intel Skylake-X/Zen4 CPUs. Allows processing 16 32-bit floats in a single CPU instruction.
        if is_x86_feature_detected!("avx512f") {
            return config_table!(
                crate::math::common::Avx512Math,
                InstructionSet::Avx512,
                "AVX-512",
                true,
                horizontal_sum_avx512,
                {
                    tanh_slice: crate::math::activations::tanh_slice_avx512,
                    sigmoid_slice: crate::math::activations::sigmoid_slice_avx512,
                    relu_slice: crate::math::activations::relu_slice_avx512,
                    prelu_slice: crate::math::activations::prelu_slice_avx512,
                    softsign_slice: crate::math::activations::softsign_slice_avx512,
                    silu_slice: crate::math::activations::silu_slice_avx512,
                    hard_tanh_slice: crate::math::activations::hard_tanh_slice_avx512,
                    hard_swish_slice: crate::math::activations::hard_swish_slice_avx512,
                    fast_tanh_slice: crate::math::activations::fast_tanh_slice_avx512,
                    leaky_hard_tanh_slice: crate::math::activations::leaky_hard_tanh_slice_avx512,
                }
            );
        }
        // 3. Standard 256-bit AVX2 with FMA (Floating-Point Multiply-Add).
        // The absolute minimum required to run NAM-rs (x86-64-v3 specification).
        // Intel Haswell (2013) or AMD Excavator (2015) processors and newer.
        if is_x86_feature_detected!("avx2") {
            return config_table!(
                crate::math::common::Avx2Math,
                InstructionSet::Avx2,
                "AVX2",
                false,
                horizontal_sum_avx2,
                {
                    tanh_slice: crate::math::activations::tanh_slice_avx2,
                    sigmoid_slice: crate::math::activations::sigmoid_slice_avx2,
                    relu_slice: crate::math::activations::relu_slice_avx2,
                    prelu_slice: crate::math::activations::prelu_slice_avx2,
                    softsign_slice: crate::math::activations::softsign_slice_avx2,
                    silu_slice: crate::math::activations::silu_slice_avx2,
                    hard_tanh_slice: crate::math::activations::hard_tanh_slice_avx2,
                    hard_swish_slice: crate::math::activations::hard_swish_slice_avx2,
                    fast_tanh_slice: crate::math::activations::fast_tanh_slice_avx2,
                    leaky_hard_tanh_slice: crate::math::activations::leaky_hard_tanh_slice_avx2,
                }
            );
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
