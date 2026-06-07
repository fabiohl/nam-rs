// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels Dot Product 4x — AVX-512 BF16 (f32 accumulation via `_mm512_fmadd_ps`).
//!
//! Converts BF16→f32 using `_mm512_slli_epi32` and accumulates strictly
//! in f32 SIMD registers via `_mm512_fmadd_ps`, maintaining full 24-bit
//! mantissa precision throughout the dot product chain.

mod dual;
mod helpers;
mod single;

pub use dual::dot_product_4x_interleaved_dual_frame_avx512_bf16;
pub use single::dot_product_4x_interleaved_avx512_bf16;
