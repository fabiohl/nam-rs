// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Identificação do backend SIMD ativo em tempo de execução.

/// Retorna o nome do badge SIMD conforme o instruction set detectado.
pub fn get_simd_badge() -> &'static str {
    use crate::math::common::{InstructionSet, SIMD_MATH};
    match SIMD_MATH.instruction_set {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx2Vnni => "AVX2-VNNI",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512Vnni => "AVX-512-VNNI",
        InstructionSet::Avx512VnniBf16 => "AVX-512-BF16",
    }
}
