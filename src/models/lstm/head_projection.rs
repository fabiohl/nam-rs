// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared head projection logic for LSTM models.
//!
//! The head projection is the final step that converts the LSTM hidden state
//! into an audio sample value via a dot product with the head weights plus bias.
//! Both 1-layer and 2-layer models share the same logic; this module avoids
//! duplication between `model1.rs` and `model2.rs`.

/// Computes the LSTM head output for SIMD-dispatched processing kernels.
///
/// Returns `dot_product(hidden, head_weights) + head_bias`, using full-precision
/// F32 weights when `self.use_f32_head` is true, otherwise using the quantized
/// SIMD dot_product function.
///
/// ## Parameters
/// - `$self`: Reference to the model (provides `use_f32_head`, `head_weights_f32`,
///   `head_weights`, `head_bias`).
/// - `$h_quant`: The hidden state accessible to the quantized dot_product function
///   (e.g. `&[f32]` or `&[u16]` depending on the data type).
/// - `$get_f32_hidden`: Expression returning the hidden state as `&[f32]`
///   (only evaluated when `use_f32_head == true`).
/// - `$dot_prod`: Path to the quantized dot_product function.
#[macro_export]
macro_rules! compute_lstm_head_simd {
    ($self:expr, $h_quant:expr, $get_f32_hidden:expr, $dot_prod:path) => {{
        (if $self.use_f32_head {
            let h_f32 = $get_f32_hidden;
            $crate::math::common::scalar_ref::dot_product_f32_native(h_f32, &$self.head_weights_f32)
        } else {
            $dot_prod($h_quant, &$self.head_weights)
        }) + $self.head_bias
    }};
}
