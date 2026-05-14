// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fundação comum para operações matemáticas e SIMD.
//!
//! Este módulo contém as definições estruturais que permitem ao NAM-rs ser
//! agnóstico ao hardware enquanto mantém performance nativa.
//!
//! # Componentes
//! - `traits`: A trait `SimdMath` que define a interface de todos os kernels.
//! - `dispatch`: Mecanismo de seleção dinâmica de arquitetura (AVX2, AVX-512, etc.).
//! - `avx2_impl` / `avx512_impl`: Implementações concretas dos kernels para x86-64.
//! - `scalar_ref`: Implementações de fallback para compatibilidade e testes.
//! - `aligned`: Estruturas para garantir alinhamento de memória (RT-Safety).

pub mod aligned;
pub mod avx2_impl;
pub mod avx512_impl;
pub mod dispatch;
pub mod ops;
pub mod scalar_ref;
/// Testes de unidade para a infraestrutura matemática comum.
#[cfg(test)]
pub mod tests;
pub mod traits;
pub mod utility;

pub use aligned::AlignedVec;
pub use avx2_impl::{Avx2Math, Avx2VnniMath};
pub use avx512_impl::{Avx512Math, Avx512VnniBf16Math, Avx512VnniMath};
pub use dispatch::{InstructionSet, SIMD_MATH, SimdMathConfig};
pub use ops::*;
pub use scalar_ref::*;
pub use traits::SimdMath;
pub use utility::*;

/// Macro para despacho dinâmico SIMD baseado na configuração global.
#[macro_export]
macro_rules! dispatch_simd {
    // Modo 2: Despacho para métodos específicos de um objeto (ex: lstm.rs)
    ($target:expr, $m512bf16:ident, $m512vnni:ident, $m512:ident, $m256vnni:ident, $m256:ident $(, $arg:expr)*) => {
        {
            use $crate::math::common::{SIMD_MATH, InstructionSet};
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
            use $crate::math::common::{SIMD_MATH, InstructionSet};
            match SIMD_MATH.instruction_set {
                InstructionSet::Avx512VnniBf16 => {
                    $target.$method::<$crate::math::common::Avx512VnniBf16Math>($($arg),*)
                }
                InstructionSet::Avx512Vnni => {
                    $target.$method::<$crate::math::common::Avx512VnniMath>($($arg),*)
                }
                InstructionSet::Avx512 => {
                    $target.$method::<$crate::math::common::Avx512Math>($($arg),*)
                }
                InstructionSet::Avx2Vnni => {
                    $target.$method::<$crate::math::common::Avx2VnniMath>($($arg),*)
                }
                InstructionSet::Avx2 => {
                    $target.$method::<$crate::math::common::Avx2Math>($($arg),*)
                }
            }
        }
    };

    // Modo 3: Chamada direta à v-table
    ($method:ident ($($arg:expr),*)) => {
        {
            #[allow(clippy::macro_metavars_in_unsafe)]
            {
                use $crate::math::common::SIMD_MATH;
                unsafe { (SIMD_MATH.$method)($($arg),*) }
            }
        }
    };
}

pub use dispatch_simd;
