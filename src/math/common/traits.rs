// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! [Definição da interface abstrata para operações SIMD.

/// Trait de abstração para despacho estático de operações matemáticas SIMD.
///
/// # Safety
/// Todas as implementações deste trait utilizam intrinsics SIMD x86-64 que requerem
/// features de CPU específicas (AVX2/FMA mínimo). O chamador deve garantir que a CPU
/// suporte as features declaradas via `#[target_feature]` na implementação concreta.
/// Os slices passados devem ser válidos e acessíveis para leitura/escrita conforme indicado.
///
/// # Grupos de Operações
///
/// As operações deste trait estão organizadas nos seguintes grupos:
/// - **(A) Dot Products**: Produtos escalares escalar/4x/dual-frame (e.g., `dot_product`).
/// - **(B) GEMV/GEMM Fused**: Kernels fundidos de matriz-vetor/matriz-matriz (e.g., `fused_add_gemv`).
/// - **(C) Activations**: Funções de ativação tanh/sigmoid e fusões gated (e.g., `tanh_slice`).
/// - **(D) Conversions**: Utilitários de conversão f32 ↔ bf16 (e.g., `f32_to_bf16`).
/// - **(E) LSTM Gates**: Kernels específicos para portas de células LSTM (e.g., `fused_lstm_gates_dyn`).
pub trait SimdMath {
    /// Tipo de registrador SIMD utilizado (ex: __m256 ou __m512).
    type V: Copy;

    /// Indica se esta implementação utiliza pesos e sinais em formato BF16.
    const IS_BF16: bool = false;

    // --- (A) Dot Products ---

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

    /// Soma horizontal de um buffer.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32;

    // --- (B) GEMV/GEMM Fused ---

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

    /// Kernel de GEMV com sobrescrita em lote.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
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

    /// Kernel de GEMV com sobrescrita em lote (entrada BF16).
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn gemv_overwrite_batch_bf16(
        in_frames: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    );

    /// Kernel de GEMV com sobrescrita para 4 portas simultâneas.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    );

    /// Kernel de GEMV com sobrescrita para 4 portas simultâneas (entrada BF16).
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn gemv_overwrite_bf16_4gate(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    );

    // --- (C) Activations ---

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

    /// Calcula o máximo da energia entre dois canais (Estéreo).
    /// Retorna `max(energy_l, energy_r)`.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32;

    /// Calcula a energia (Mean Square) de um bloco.
    /// $E = \frac{1}{N} \sum x_i^2$
    ///
    /// # Safety
    /// O buffer deve ser válido.
    unsafe fn compute_energy(data: &[f32]) -> f32;

    /// Calcula a diferença absoluta máxima entre dois blocos.
    /// $\max(|a_i - b_i|)$
    ///
    /// # Safety
    /// Os buffers devem ser válidos e ter o mesmo tamanho.
    unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32;

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

    /// Ativação Tanh em bloco.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn activation_tanh_block(buf: &mut [f32]);

    // --- (D) Conversions ---

    /// Conversão de F32 para BF16.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]);

    /// Armazena o conteúdo de um registrador SIMD como BF16 (truncado).
    ///
    /// # Safety
    /// O ponteiro deve ser válido e ter espaço suficiente.
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V);

    // --- (E) LSTM Gates ---

    /// Kernel fundido para processamento de portas LSTM dinâmicas.
    /// Realiza ativações (sigmoid/tanh) e atualização de estado (cell/hidden) em um único passo.
    ///
    /// # Safety
    /// Buffers devem ter tamanhos compatíveis com `hidden_size`.
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    );

    /// Convolução estéreo (usada no resampler).
    /// Realiza o produto escalar entre um banco de coeficientes e dois buffers de entrada (L/R).
    ///
    /// # Safety
    /// `coeffs`, `input_l` e `input_r` devem ser ponteiros válidos para pelo menos `taps` elementos.
    /// `coeffs` deve estar alinhado conforme o registrador SIMD.
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32);

    /// Convolução mono (usada no resampler).
    /// Realiza o produto escalar entre um banco de coeficientes e um buffer de entrada.
    ///
    /// # Safety
    /// `coeffs` e `input` devem ser ponteiros válidos para pelo menos `taps` elementos.
    /// `coeffs` deve estar alinhado conforme o registrador SIMD.
    unsafe fn convolve_mono(
        coeffs: *const f32,
        input: *const f32,
        taps: usize,
    ) -> f32;

    /// Aplica ganho e detecta clipping em estéreo em uma única passagem.
    /// Retorna `true` se qualquer amostra resultante possuir `|x| > 1.0`.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool;

    /// Aplica ganho constante em estéreo (sem detecção de clipping).
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32);

    /// Aplica ganho constante em um buffer mono.
    ///
    /// # Safety
    /// O buffer deve ser válido.
    unsafe fn apply_gain(data: &mut [f32], gain: f32);

    /// Aplica rampa linear de ganho em um buffer mono.
    ///
    /// # Safety
    /// O buffer deve ser válido.
    unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32);

    /// Aplica rampa linear de ganho em estéreo.
    ///
    /// # Safety
    /// Buffers devem ser válidos.
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32);

    /// Kernel especializado para soma Head do WaveNet.
    /// Vetoriza o somatório horizontal das projeções head1 (batch), soma head2 e escala final.
    ///
    /// # Safety
    /// Buffers devem ser válidos e ter tamanhos compatíveis com HEAD e num_frames.
    /// Kernel especializado para soma Head do WaveNet.
    /// Vetoriza o somatório horizontal das projeções head1 (batch), soma head2 e escala final.
    ///
    /// # Safety
    /// Buffers devem ser válidos e ter tamanhos compatíveis com HEAD e num_frames.
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    );

    /// Kernel especializado para soma Head do WaveNet (versão dinâmica).
    ///
    /// # Safety
    /// Buffers devem ser válidos e ter tamanhos compatíveis com head e num_frames.
    unsafe fn batch_wavenet_head_sum_dyn(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        head: usize,
        scale: f32,
    );
}
