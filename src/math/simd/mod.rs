// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! [T21] Módulo principal de abstração SIMD.
//!
//! Fornece o macro `dispatch_simd!` e re-exporta os backends de hardware.

pub mod aligned;
pub mod avx2;
pub mod avx512;
pub mod dispatch;
pub mod ops;
pub mod scalar_ref;
pub mod traits;
pub mod utility;

pub use aligned::AlignedVec;
pub use avx2::*;
pub use avx512::*;
pub use dispatch::{InstructionSet, SIMD_MATH, SimdMathConfig};
pub use ops::*;
pub use scalar_ref::*;
pub use traits::SimdMath;

/// Macro para despacho dinâmico SIMD baseado na configuração global.
///
/// # Safety
/// O chamador deve garantir que os métodos despachados (`$method`, etc.) sejam executados apenas
/// em hardware compatível com a instrução SIMD selecionada (`InstructionSet`). A execução de
/// instruções não suportadas pelo CPU causará comportamento indefinido (undefined behavior).
///
/// Suporta três modos:
/// 1. `dispatch_simd!(self, method, args...)`: Despacha para um método genérico `<M: SimdMath>`.
/// 2. `dispatch_simd!(self, m512bf16, m512vnni, m512, m256vnni, m256, args...)`: Despacha para um dos métodos específicos.
/// 3. `dispatch_simd!(method(args...))`: Despacha diretamente para uma função da v-table global.
#[macro_export]
macro_rules! dispatch_simd {
    // Modo 2: Despacho para métodos específicos de um objeto (ex: lstm.rs)
    ($target:expr, $m512bf16:ident, $m512vnni:ident, $m512:ident, $m256vnni:ident, $m256:ident $(, $arg:expr)*) => {
        {
            use $crate::math::simd::{SIMD_MATH, InstructionSet};
            match SIMD_MATH.instruction_set {
                InstructionSet::Avx512VnniBf16 => $target.$m512bf16($($arg),*),
                InstructionSet::Avx512Vnni => $target.$m512vnni($($arg),*),
                InstructionSet::Avx512 => $target.$m512($($arg),*),
                InstructionSet::Avx2Vnni => $target.$m256vnni($($arg),*),
                InstructionSet::Avx2 => $target.$m256($($arg),*),
            }
        }
    };

    // Modo 1: Despacho para método genérico de um objeto (ex: wavenet.rs)
    ($target:expr, $method:ident $(, $arg:expr)*) => {
        {
            use $crate::math::simd::{SIMD_MATH, InstructionSet};
            match SIMD_MATH.instruction_set {
                InstructionSet::Avx512VnniBf16 => {
                    $target.$method::<$crate::math::simd::avx512::Avx512VnniBf16Math>($($arg),*)
                }
                InstructionSet::Avx512Vnni => {
                    $target.$method::<$crate::math::simd::avx512::Avx512VnniMath>($($arg),*)
                }
                InstructionSet::Avx512 => {
                    $target.$method::<$crate::math::simd::avx512::Avx512Math>($($arg),*)
                }
                InstructionSet::Avx2Vnni => {
                    $target.$method::<$crate::math::simd::avx2::Avx2VnniMath>($($arg),*)
                }
                InstructionSet::Avx2 => {
                    $target.$method::<$crate::math::simd::avx2::Avx2Math>($($arg),*)
                }
            }
        }
    };

    // Modo 3: Chamada direta à v-table (legado/específico)
    ($method:ident ($($arg:expr),*)) => {
        {
            #[allow(clippy::macro_metavars_in_unsafe)]
            {
                use $crate::math::simd::SIMD_MATH;
                unsafe { (SIMD_MATH.$method)($($arg),*) }
            }
        }
    };
}

// Re-export do macro para ser visível no caminho crate::math::simd::dispatch_simd
pub use dispatch_simd;

#[cfg(test)]
#[path = "simd_test.rs"]
mod simd_test;
