// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

macro_rules! impl_avx512_activations {
    () => {
        #[inline(always)]
        unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
            accumulate_head_fallback(dest, src)
        }

        #[inline(always)]
        unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            tanh_and_accumulate_block_fallback(head_input, block)
        }

        #[inline(always)]
        unsafe fn gated_activation_and_accumulate_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            unsafe {
                crate::math::wavenet::accumulate::gated_activation_and_accumulate_block_avx512(
                    head_input, block, ch,
                )
            }
        }

        #[inline(always)]
        unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            tanh_and_overwrite_block_fallback(head_input, block)
        }

        #[inline(always)]
        unsafe fn gated_activation_and_overwrite_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            unsafe {
                crate::math::wavenet::accumulate::gated_activation_and_overwrite_block_avx512(
                    head_input, block, ch,
                )
            }
        }

        #[inline(always)]
        unsafe fn tanh_slice(slice: &mut [f32]) {
            crate::math::activations::tanh_slice_avx512(slice)
        }

        #[inline(always)]
        unsafe fn sigmoid_slice(slice: &mut [f32]) {
            crate::math::activations::sigmoid_slice_avx512(slice)
        }

        #[inline(always)]
        unsafe fn activation_tanh_block(buf: &mut [f32]) {
            crate::math::activations::tanh_slice_avx512(buf)
        }

        #[inline(always)]
        unsafe fn fused_lstm_gates_dyn(
            gates: &mut [f32],
            cell_state: &mut [f32],
            hidden_state: &mut [f32],
            hidden_size: usize,
        ) {
            unsafe {
                crate::math::lstm::fused_lstm_gates_dyn_avx512(
                    gates,
                    cell_state,
                    hidden_state,
                    hidden_size,
                )
            }
        }
    };
}

macro_rules! impl_avx512vnni_activations {
    () => {
        #[inline(always)]
        unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
            Avx512Math::accumulate_head(dest, src)
        }

        #[inline(always)]
        unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::tanh_and_accumulate_block(head_input, block)
        }

        #[inline(always)]
        unsafe fn gated_activation_and_accumulate_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            Avx512Math::gated_activation_and_accumulate_block(head_input, block, ch)
        }

        #[inline(always)]
        unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::tanh_and_overwrite_block(head_input, block)
        }

        #[inline(always)]
        unsafe fn gated_activation_and_overwrite_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            Avx512Math::gated_activation_and_overwrite_block(head_input, block, ch)
        }

        #[inline(always)]
        unsafe fn tanh_slice(slice: &mut [f32]) {
            Avx512Math::tanh_slice(slice)
        }

        #[inline(always)]
        unsafe fn sigmoid_slice(slice: &mut [f32]) {
            Avx512Math::sigmoid_slice(slice)
        }

        #[inline(always)]
        unsafe fn activation_tanh_block(buf: &mut [f32]) {
            Avx512Math::activation_tanh_block(buf)
        }

        #[inline(always)]
        unsafe fn fused_lstm_gates_dyn(
            gates: &mut [f32],
            cell_state: &mut [f32],
            hidden_state: &mut [f32],
            hidden_size: usize,
        ) {
            Avx512Math::fused_lstm_gates_dyn(gates, cell_state, hidden_state, hidden_size)
        }
    };
}

macro_rules! impl_avx512vnni_bf16_activations {
    () => {
        #[inline(always)]
        unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
            Avx512Math::accumulate_head(dest, src)
        }

        #[inline(always)]
        unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::tanh_and_accumulate_block(head_input, block)
        }

        #[inline(always)]
        unsafe fn gated_activation_and_accumulate_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            Avx512Math::gated_activation_and_accumulate_block(head_input, block, ch)
        }

        #[inline(always)]
        unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            Avx512Math::tanh_and_overwrite_block(head_input, block)
        }

        #[inline(always)]
        unsafe fn gated_activation_and_overwrite_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            Avx512Math::gated_activation_and_overwrite_block(head_input, block, ch)
        }

        #[inline(always)]
        unsafe fn tanh_slice(slice: &mut [f32]) {
            Avx512Math::tanh_slice(slice)
        }

        #[inline(always)]
        unsafe fn sigmoid_slice(slice: &mut [f32]) {
            Avx512Math::sigmoid_slice(slice)
        }

        #[inline(always)]
        unsafe fn activation_tanh_block(buf: &mut [f32]) {
            Avx512Math::activation_tanh_block(buf)
        }

        #[inline(always)]
        unsafe fn fused_lstm_gates_dyn(
            gates: &mut [f32],
            cell_state: &mut [f32],
            hidden_state: &mut [f32],
            hidden_size: usize,
        ) {
            Avx512Math::fused_lstm_gates_dyn(gates, cell_state, hidden_state, hidden_size)
        }
    };
}


pub(super) use impl_avx512_activations;
pub(super) use impl_avx512vnni_activations;
pub(super) use impl_avx512vnni_bf16_activations;
