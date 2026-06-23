// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::detect::SIMD_MATH;
use super::instruction_set::InstructionSet;

/// Static SIMD configuration — descriptive data only, no v-table.
///
/// All dispatch is handled via the `dispatch_simd!` macro, which matches
/// on `instruction_set` and monomorphizes calls via the `SimdMath` trait
/// (e.g. `<Avx2Math>::apply_gain(...)`). Function pointers have been removed;
/// all consumers use trait-based static dispatch.
#[derive(Clone, Copy)]
pub struct SimdMathConfig {
    /// Active instruction set.
    pub instruction_set: InstructionSet,
    /// Friendly backend name.
    pub name: &'static str,
    /// Whether the backend is AVX-512.
    pub is_avx512: bool,
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
