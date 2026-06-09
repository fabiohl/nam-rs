// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Active SIMD backend identification at runtime.

/// Returns the SIMD badge name based on the detected instruction set.
pub fn get_simd_badge() -> &'static str {
    use crate::math::common::{InstructionSet, SIMD_MATH};
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "AVX-512-BF16",
    }
}
