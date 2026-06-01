// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementações escalares de referência para Dot Product 4x.
//!
//! Re-exporta as implementações escalares centralizadas em `crate::math::common::scalar_ref`.

pub use crate::math::common::scalar_ref::{
    dot_product_4x_interleaved_bf16_fallback, dot_product_4x_interleaved_dual_frame_bf16_fallback,
    dot_product_4x_interleaved_dual_frame_fallback, dot_product_4x_interleaved_fallback,
};
