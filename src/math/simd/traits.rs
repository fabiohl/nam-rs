// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! [T21] Definição da interface abstrata para operações SIMD.

/// Trait de abstração para despacho estático de operações matemáticas SIMD.
///
/// # Safety
/// Todas as implementações deste trait utilizam intrinsics SIMD x86-64 que requerem
/// features de CPU específicas (AVX2/FMA mínimo). O chamador deve garantir que a CPU
/// suporte as features declaradas via `#[target_feature]` na implementação concreta.
/// Os slices passados devem ser válidos e acessíveis para leitura/escrita conforme indicado.
pub trait SimdMath {
    /// Tipo de registrador SIMD utilizado (ex: __m256 ou __m512).
    type V: Copy;

    /// Indica se esta implementação utiliza pesos e sinais em formato BF16.
    const IS_BF16: bool = false;

    /// Calcula o produto escalar entre dois vetores f32.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32;

    /// Calcula o produto escalar entre dois vetores BF16.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32;

    /// Calcula 4 produtos escalares BF16 simultaneamente (interleaved) com entrada f32.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4];

    /// Calcula 4 produtos escalares BF16 simultaneamente (interleaved) com entrada BF16.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4];

    /// Calcula 4 produtos escalares BF16 simultaneamente (interleaved) para 2 frames paralelos.
    /// Retorna uma tupla com os 4 resultados do frame 0 e os 4 do frame 1.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]);

    /// Calcula 4 produtos escalares BF16 simultaneamente (interleaved) para 2 frames paralelos (entrada BF16).
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]);

    /// Calcula 4 produtos escalares BF16 simultaneamente.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4];

    /// Kernel fundido de adição e GEMV.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Kernel fundido de adição e GEMM em lote.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// Kernel fundido de GEMM residual em lote.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// Kernel de GEMV com sobrescrita.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Kernel de GEMV com sobrescrita (entrada BF16).
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    );

    /// Acumula o conteúdo de um vetor em outro.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]);

    /// Fusão de Tanh + Acúmulo de Head.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]);

    /// Fusão de Gated Activation + Acúmulo de Head.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    );

    /// Conversão de F32 para BF16.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]);

    /// Aplica Tanh em um slice.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn tanh_slice(slice: &mut [f32]);

    /// Aplica Sigmoid em um slice.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn sigmoid_slice(slice: &mut [f32]);

    /// Soma horizontal de um buffer.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32;

    /// Ativação Tanh em bloco.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn activation_tanh_block(buf: &mut [f32]);
}
