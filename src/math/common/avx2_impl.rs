// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Implementações AVX2 da trait `SimdMath`.
//!
//! Este módulo contém as structs `Avx2Math` e `Avx2VnniMath` (type alias)
//! que implementam a trait `SimdMath` usando instruções AVX2/FMA.
//! Os métodos delegam para funções-kernel em `math::gemm`, `math::wavenet`,
//! `math::lstm`, `math::dsp` e `math::common::utility`.

use crate::math::common::scalar_ref::*;
use crate::math::common::traits::SimdMath;
use core::arch::x86_64::*;

/// Implementação concreta da trait SimdMath para processadores com suporte a AVX2 e FMA.
///
/// Aqui é onde "ligamos os fios": conectamos as operações matemáticas abstratas do sistema
/// às funções ultra-rápidas que documentamos acima. Esta estrutura garante que o NAM-rs
/// aproveite a força total do hardware moderno para processar áudio em tempo real.
pub struct Avx2Math;

impl SimdMath for Avx2Math {
    type V = __m256;

    // Dot Product: Multiplica pesos por sinal e soma o resultado (o "DNA" das redes neurais).
    // No AVX2, usamos registradores de 256 bits que processam 8 números de uma vez.
    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { super::super::gemm::dot::dot_product_avx2(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        // Como o AVX2 puro não tem aceleração nativa para BF16 ( Brain Float), usamos a versão comum de reserva.
        unsafe { dot_product_bf16_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { super::super::gemm::dot_4x::dot_product_4x_interleaved_avx2(weights, state) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_bf16(weights: &[[u16; 4]], state: &[u16]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_bf16_fallback(weights, state) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame(
        weights: &[[u16; 4]],
        state_f0: &[f32],
        state_f1: &[f32],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe {
            super::super::gemm::dot_4x::dot_product_4x_interleaved_dual_frame_avx2(
                weights, state_f0, state_f1,
            )
        }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved_dual_frame_bf16(
        weights: &[[u16; 4]],
        state_f0: &[u16],
        state_f1: &[u16],
    ) -> ([f32; 4], [f32; 4]) {
        unsafe { dot_product_4x_interleaved_dual_frame_bf16_fallback(weights, state_f0, state_f1) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16_4x(
        w0: &[u16],
        w1: &[u16],
        w2: &[u16],
        w3: &[u16],
        in_frame: &[u16],
    ) -> [f32; 4] {
        unsafe { dot_product_bf16_4x_fallback(w0, w1, w2, w3, in_frame) }
    }

    // Operações GEMV: Multiplicação de matriz por vetor, usada em quase todas as camadas do modelo.
    // O prefixo "fused" indica que a adição do vetor de Bias é combinada (fundida) com a multiplicação
    // para economizar acessos de memória e instruções do processador.
    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe {
            // Delega o cálculo para o kernel otimizado de multiplicação matriz-vetor AVX2.
            super::super::gemm::gemv::fused_add_gemv_avx2(
                in_frame, weights, bias, out_frame, do_bias,
            )
        }
    }

    /// Executa multiplicação de matriz por um lote (batch) de vetores via AVX2.
    /// Útil quando processamos múltiplos quadros de áudio concorrentemente para reduzir overheads.
    #[inline(always)]
    unsafe fn fused_add_gemm_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        unsafe {
            // Delega o cálculo do batch de multiplicação matriz-matriz (GEMM) para o kernel AVX2.
            super::super::gemm::gemm_batch::fused_add_gemm_batch_avx2(
                in_frames, weights, bias, out_frames, num_frames, do_bias,
            )
        }
    }

    /// Executa multiplicação de matriz por vetor adicionando também a conexão residual (skip connection)
    /// da camada anterior. Muito utilizado na arquitetura de blocos residuais WaveNet.
    #[inline(always)]
    unsafe fn fused_gemm_residual_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        residual: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        unsafe {
            // Delega a multiplicação com soma residual e bias integrada para o kernel AVX2.
            super::super::gemm::gemm_batch::fused_gemm_residual_batch_avx2(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
    }

    /// Versão que sobrescreve o buffer de saída diretamente com o resultado da multiplicação matriz-vetor,
    /// sem acumular com valores preexistentes no buffer.
    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe {
            super::super::gemm::gemv::gemv_overwrite_avx2(
                in_frame, weights, bias, out_frame, do_bias,
            )
        }
    }

    /// Versão que sobrescreve o buffer de saída aceitando dados de entrada representados em BF16 (16 bits)
    /// e pesos em BF16, realizando a acumulação em f32 para preservar fidelidade.
    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        // Como a arquitetura AVX2 clássica não possui suporte nativo a instruções de dot-product BF16,
        // recorremos a uma função de fallback (conversão em tempo de execução para f32).
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

    // Portas LSTM (4-gate): Calcula simultaneamente os 4 controles de memória da rede LSTM.
    // O cálculo das portas (Input, Forget, Cell Candidate e Output) compartilha os mesmos estados
    // de entrada. Projetar em paralelo reduz drasticamente os saltos de cache.
    #[inline(always)]
    unsafe fn gemv_overwrite_4gate(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        let ih = in_frame.len();
        let stride = ih * hidden_size;
        unsafe {
            // Dividimos a matriz de pesos única contínua em 4 blocos (strides) correspondentes a cada porta:
            // - weights[0..stride]: Pesos da porta de Entrada (Input Gate).
            // - weights[stride..2*stride]: Pesos da porta de Esquecimento (Forget Gate).
            // - weights[2*stride..3*stride]: Pesos da porta Candidata (Cell Candidate).
            // - weights[3*stride..4*stride]: Pesos da porta de Saída (Output Gate).
            super::super::gemm::gemv_4gate::gemv_4gate_avx2(
                in_frame,
                &weights[0..stride],
                &weights[stride..2 * stride],
                &weights[2 * stride..3 * stride],
                &weights[3 * stride..4 * stride],
                bias,
                out_gates,
                do_bias,
            )
        }
    }

    /// Equivalente ao `gemv_overwrite_4gate` porém processando dados de entrada representados
    /// no formato de precisão reduzida BF16.
    #[inline(always)]
    unsafe fn gemv_overwrite_bf16_4gate(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_gates: &mut [f32],
        hidden_size: usize,
        do_bias: bool,
    ) {
        let ih = in_frame.len();
        let stride = ih * hidden_size;
        unsafe {
            // Como AVX2 não possui VNNI/BF16 nativo com acumulação direta na CPU, usamos o fallback.
            gemv_4gate_bf16_fallback(
                in_frame,
                &weights[0..stride],
                &weights[stride..2 * stride],
                &weights[2 * stride..3 * stride],
                &weights[3 * stride..4 * stride],
                bias,
                out_gates,
                do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
        unsafe { super::super::wavenet::accumulate::accumulate_head_avx2(dest, src) }
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        unsafe {
            super::super::wavenet::accumulate::tanh_and_accumulate_block_avx2(head_input, block)
        }
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        unsafe {
            super::super::wavenet::accumulate::gated_activation_and_accumulate_block_avx2(
                head_input, block, ch,
            )
        }
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        unsafe { f32_to_bf16_fallback(src, dest) }
    }

    /// Converte um registrador contendo 8 floats de 32 bits (Self::V) para o formato compactado
    /// BF16 (16 bits) e armazena os resultados na memória.
    ///
    /// ## Detalhes da Mágica de Bits (Bitwise SIMD):
    /// Para converter f32 em BF16 sem gastar muitos ciclos de CPU em conversões matemáticas lentas,
    /// a técnica aproveita a semelhança estrutural entre float de precisão simples IEEE 754 e BF16:
    /// Ambos compartilham a mesma faixa dinâmica (8 bits de expoente), porém o BF16 descarta os
    /// 16 bits menos significativos da mantissa (truncamento/arredondamento).
    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, v: Self::V) {
        unsafe {
            // 1. Reinterpreta o registrador f32 (__m256) como inteiros de 32 bits (__m256i). Custo: 0 ciclos.
            let v_i = _mm256_castps_si256(v);
            // 2. Desloca os inteiros 16 bits para a direita (Shift Right). A metade superior (mantissa BF16)
            // se move para a metade inferior de cada elemento de 32 bits.
            let v_shifted = _mm256_srli_epi32(v_i, 16);
            // 3. Empacota com saturação não-sinalizada (Packus) elementos de 32 bits em elementos de 16 bits.
            // Isso consolida os dados de 16 bits úteis.
            let packed = _mm256_packus_epi32(v_shifted, v_shifted);
            // 4. Permuta os canais de 64 bits usando o padrão (8/0x08) para reagrupar os resultados válidos
            // que foram misturados devido ao comportamento inerente da instrução _mm256_packus_epi32.
            let permuted = _mm256_permute4x64_epi64(packed, 8);
            // 5. Extrai a metade inferior (128 bits de um registrador de 256 bits), contendo os 8 valores BF16.
            let v_low = _mm256_castsi256_si128(permuted);
            // 6. Grava os 8 BF16 compactados diretamente no destino da memória RAM.
            _mm_storeu_si128(ptr as *mut __m128i, v_low);
        }
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { crate::math::activations::tanh_slice_avx2(slice) }
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { crate::math::activations::sigmoid_slice_avx2(slice) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { super::utility::horizontal_sum_avx2(ptr, N) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { crate::math::activations::tanh_slice_avx2(buf) }
    }

    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        hidden_state: &mut [f32],
        hidden_size: usize,
    ) {
        unsafe {
            super::super::lstm::fused_lstm_gates_dyn_avx2(
                gates,
                cell_state,
                hidden_state,
                hidden_size,
            )
        }
    }

    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        unsafe { super::super::dsp::stereo::compute_energy_stereo_avx2(l, r) }
    }

    #[inline(always)]
    unsafe fn compute_energy(data: &[f32]) -> f32 {
        unsafe { super::super::dsp::stereo::compute_energy_avx2(data) }
    }

    #[inline(always)]
    unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
        unsafe { super::super::dsp::stereo::compute_max_diff_avx2(a, b) }
    }

    // Convolve Stereo: Aplica filtragem (equalização) nos dois canais (esquerdo e direito) simultaneamente.
    //
    // ## Otimização de Vazão (Throughput SIMD):
    // Como a resposta ao impulso (coeficientes de filtro FIR) é idêntica para ambos os canais em um
    // processamento estéreo padrão, carregamos os coeficientes uma única vez nos registradores AVX2 e
    // os multiplicamos concorrentemente pelas amostras de áudio esquerda (input_l) e direita (input_r).
    // Isso dobra a eficiência do cálculo em relação a filtrar cada canal sequencialmente.
    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { super::super::dsp::stereo::convolve_stereo_avx2(coeffs, input_l, input_r, taps) }
    }

    #[inline(always)]
    unsafe fn convolve_stereo_dual(
        coeffs0: *const f32,
        coeffs1: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> ((f32, f32), (f32, f32)) {
        unsafe {
            super::super::dsp::stereo::convolve_stereo_dual_avx2(
                coeffs0, coeffs1, input_l, input_r, taps,
            )
        }
    }

    #[inline(always)]
    unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
        unsafe { super::super::dsp::stereo::convolve_mono_avx2(coeffs, input, taps) }
    }

    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        unsafe {
            super::super::dsp::gain::apply_gain_and_detect_clipping_stereo_avx2(left, right, gain)
        }
    }

    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        unsafe { super::super::dsp::gain::apply_gain_stereo_avx2(left, right, gain) }
    }

    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        unsafe { super::super::dsp::gain::apply_gain_avx2(data, gain) }
    }

    #[inline(always)]
    unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
        unsafe { super::super::dsp::gain::apply_ramp_avx2(data, start, step) }
    }

    #[inline(always)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        unsafe {
            super::super::wavenet::head::batch_wavenet_head_sum_avx2::<HEAD>(
                head1, head2, output, scale,
            )
        }
    }

    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        unsafe { super::super::dsp::gain::apply_ramp_stereo_avx2(left, right, start, step) }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch(
        in_frames: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        let in_len = in_frames.len() / num_frames;
        let out_len = out_frames.len() / num_frames;
        for i in 0..num_frames {
            let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
            let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
            unsafe {
                super::super::gemm::gemv::gemv_overwrite_avx2(
                    in_slice, weights, bias, out_slice, do_bias,
                )
            };
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_batch_bf16(
        in_frames: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frames: &mut [f32],
        num_frames: usize,
        do_bias: bool,
    ) {
        let in_len = in_frames.len() / num_frames;
        let out_len = out_frames.len() / num_frames;
        for i in 0..num_frames {
            let in_slice = &in_frames[i * in_len..(i + 1) * in_len];
            let out_slice = &mut out_frames[i * out_len..(i + 1) * out_len];
            unsafe { gemv_overwrite_bf16_fallback(in_slice, weights, bias, out_slice, do_bias) };
        }
    }

    // Etapa final do processamento WaveNet: soma as saídas para gerar o áudio final.
    #[inline(always)]
    unsafe fn batch_wavenet_head_sum_dyn(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        head: usize,
        scale: f32,
    ) {
        unsafe {
            super::super::wavenet::head::batch_wavenet_head_sum_dyn_avx2(
                head1, head2, output, head, scale,
            )
        }
    }
}

/// AVX2 + VNNI: a instrução `VPDPBUSD` opera sobre inteiros de 8 bits,
/// sem benefício mensurável para kernels float do NAM-rs.
/// Delegação total para `Avx2Math` — type alias elimina ~300 linhas mortas.
///
/// Mantido como alias (não removido) para preservar compatibilidade com
/// `InstructionSet::Avx2Vnni` e o macro `dispatch_simd!`.
/// Futuro: remover também do enum quando a v-table for unificada.
pub type Avx2VnniMath = Avx2Math;
