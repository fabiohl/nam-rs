// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels Dot Product 4x — AVX-512 VNNI BF16 (`vdpbf16ps`).
//!
//! Placeholder para implementação futura com VNNI BF16.
//! Atualmente delega para as implementações escalares de referência.

pub use crate::math::common::scalar_ref::{
    dot_product_4x_interleaved_bf16_fallback as dot_product_4x_interleaved_avx512_bf16,
    dot_product_4x_interleaved_dual_frame_bf16_fallback as dot_product_4x_interleaved_dual_frame_avx512_bf16,
};
