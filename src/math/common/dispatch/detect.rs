// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::config::SimdMathConfig;
use super::instruction_set::InstructionSet;
use crate::config_table;
use std::sync::LazyLock;

/// Global SIMD configuration instance, detected at system boot.
///
/// Using `LazyLock` ensures the user's CPU is inspected only once,
/// at the moment the first DSP mathematical operation is invoked. After that, the
/// corresponding SIMD configuration struct (with instruction set and name)
/// is cached in memory and accessed immediately (no real-time checking cost).
pub static SIMD_MATH: LazyLock<SimdMathConfig> = LazyLock::new(detect_best_simd);

/// Inspects the CPU hardware capabilities at runtime and returns the best
/// compatible SIMD configuration.
///
/// Detection checks supported hardware features using the compiler macro `is_x86_feature_detected!`.
/// We follow a descending priority order, choosing the most advanced instruction set
/// available on the processor where the software is running.
fn detect_best_simd() -> SimdMathConfig {
    if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vnni") {
        return config_table!(InstructionSet::Avx512VnniBf16, "AVX-512 (VNNI+BF16)", true);
    }
    if is_x86_feature_detected!("avx512f") {
        return config_table!(InstructionSet::Avx512, "AVX-512", true);
    }
    config_table!(InstructionSet::Avx2, "AVX2", false)
}
