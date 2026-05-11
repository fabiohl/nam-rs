// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Implementações Escalares de Referência para kernels matemáticos DSP.
//!
//! # Propósito
//!
//! Este módulo **não é um fallback de hardware**. O projeto tem como alvo mandatório a
//! microarquitetura x86-64-v3, que garante a presença de AVX2+FMA em todo ambiente de
//! produção. Se AVX2 não for detectado em runtime, o sistema entra em pânico no boot
//! (fail-fast), portanto este código nunca é acionado no caminho de produção.
//!
//! O papel real deste módulo é ser a **implementação escalar de referência**: algoritmos
//! simples, corretos e sem otimizações, que servem como oráculo nos testes de paridade
//! dos kernels AVX2 e AVX-512. Todo kernel vetorial deve produzir o mesmo resultado
//! numérico (dentro da tolerância de ponto-flutuante) que a sua contraparte escalar aqui.
//!
//! # Fonte de Verdade
//!
//! A especificação matemática definitiva de cada operação é o NeuralAmpModelerCore (C++).
//! As implementações aqui são traduções escalares fiéis desse código de referência,
//! usadas exclusivamente para validação interna.

use super::traits::SimdMath;

/// Faz o "Produto Escalar" entre dois conjuntos de números.
/// Imagine multiplicar cada item de uma lista pelo item correspondente de outra lista
/// e, no final, somar tudo o que deu.
///
/// - `a`: Lista de números decimais (f32).
/// - `b`: Lista de "pesos" que estão guardados de forma compacta (u16/f16).
pub unsafe fn dot_product_fallback(a: &[f32], b: &[u16]) -> f32 {
    // Escolhemos o tamanho da menor lista para garantir que não vamos "atropelar" a memória.
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32; // Começamos a soma com zero.

    for i in 0..len {
        unsafe {
            // O peso 'b' está "encolhido" (f16). Aqui nós transformamos ele de volta para
            // um número decimal normal (f32) para poder fazer a conta.
            let fb = half::f16::from_bits(*b.get_unchecked(i)).to_f32();

            // Multiplicamos o valor da entrada 'a' pelo peso 'fb' e somamos ao total.
            // O 'get_unchecked' é como dizer ao computador: "Pode ir direto nesse endereço,
            // eu garanto que ele existe", o que ganha um pouquinho de velocidade.
            sum += *a.get_unchecked(i) * fb;
        }
    }
    sum // Devolve o resultado final da soma.
}

/// Versão do Produto Escalar para o formato "BF16" (Brain Floating Point).
/// É um formato de número decimal muito usado em Inteligência Artificial porque
/// ocupa metade do espaço mas mantém a "escala" dos números grandes.
pub unsafe fn dot_product_bf16_fallback(a: &[u16], b: &[u16]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    let mut sum = 0.0f32;

    for i in 0..len {
        unsafe {
            // Aqui fazemos uma "mágica" com bits: movemos o número 16 casas para a esquerda.
            // Isso transforma o formato compacto BF16 de volta para o decimal f32 padrão.
            let fa = f32::from_bits((*a.get_unchecked(i) as u32) << 16);
            let fb = f32::from_bits((*b.get_unchecked(i) as u32) << 16);

            // Multiplica e acumula na soma.
            sum += fa * fb;
        }
    }
    sum
}

/// Produto Escalar Intercalado (4x).
/// Em vez de calcular uma única soma, esta função calcula 4 somas ao mesmo tempo
/// usando os mesmos dados de entrada mas pesos diferentes.
/// Útil quando um som (estado) afeta 4 "canais" ou "neurônios" diferentes.
pub unsafe fn dot_product_4x_interleaved_fallback(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4]; // Criamos um "baldinho" para cada uma das 4 somas.

    for i in 0..len {
        unsafe {
            let s = *state.get_unchecked(i); // Pegamos um valor da entrada.
            let w = weights.get_unchecked(i); // Pegamos 4 pesos correspondentes.

            // Cada peso multiplica o mesmo valor de entrada e vai para sua respectiva soma.
            sum[0] += half::f16::from_bits(w[0]).to_f32() * s;
            sum[1] += half::f16::from_bits(w[1]).to_f32() * s;
            sum[2] += half::f16::from_bits(w[2]).to_f32() * s;
            sum[3] += half::f16::from_bits(w[3]).to_f32() * s;
        }
    }
    sum // Retorna os 4 resultados.
}

/// Mesma lógica de cima (4 somas ao mesmo tempo), mas usando o formato compacto BF16.
pub unsafe fn dot_product_4x_interleaved_bf16_fallback(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];

    for i in 0..len {
        unsafe {
            // Descompacta a entrada BF16 -> f32.
            let s = f32::from_bits((*state.get_unchecked(i) as u32) << 16);
            let w = weights.get_unchecked(i);

            // Descompacta cada um dos 4 pesos e multiplica pela entrada.
            sum[0] += f32::from_bits((w[0] as u32) << 16) * s;
            sum[1] += f32::from_bits((w[1] as u32) << 16) * s;
            sum[2] += f32::from_bits((w[2] as u32) << 16) * s;
            sum[3] += f32::from_bits((w[3] as u32) << 16) * s;
        }
    }
    sum
}

/// Produto Escalar Intercalado para "Dual Frame" (Dois quadros de áudio).
/// Esta função é ainda mais trabalhadora: ela calcula 4 somas para o primeiro
/// quadro de áudio E 4 somas para o segundo quadro, tudo num loop só.
/// Isso economiza tempo porque lemos os pesos da memória apenas uma vez.
pub unsafe fn dot_product_4x_interleaved_dual_frame_fallback(
    weights: &[[u16; 4]],
    state_f0: &[f32], // Primeiro quadro de áudio (Ex: amostra atual)
    state_f1: &[f32], // Segundo quadro (Ex: amostra anterior ou próxima)
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut sum_f0 = [0.0f32; 4];
    let mut sum_f1 = [0.0f32; 4];

    for i in 0..len {
        unsafe {
            let s0 = *state_f0.get_unchecked(i);
            let s1 = *state_f1.get_unchecked(i);
            let w = weights.get_unchecked(i);

            // Descompactamos os 4 pesos uma única vez para usar nos dois quadros.
            let w0 = half::f16::from_bits(w[0]).to_f32();
            let w1 = half::f16::from_bits(w[1]).to_f32();
            let w2 = half::f16::from_bits(w[2]).to_f32();
            let w3 = half::f16::from_bits(w[3]).to_f32();

            // Somas para o quadro 0.
            sum_f0[0] += w0 * s0;
            sum_f0[1] += w1 * s0;
            sum_f0[2] += w2 * s0;
            sum_f0[3] += w3 * s0;

            // Somas para o quadro 1.
            sum_f1[0] += w0 * s1;
            sum_f1[1] += w1 * s1;
            sum_f1[2] += w2 * s1;
            sum_f1[3] += w3 * s1;
        }
    }
    (sum_f0, sum_f1)
}

/// Mesma lógica do "Dual Frame" acima, mas tudo no formato BF16.
pub unsafe fn dot_product_4x_interleaved_dual_frame_bf16_fallback(
    weights: &[[u16; 4]],
    state_f0: &[u16],
    state_f1: &[u16],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut sum_f0 = [0.0f32; 4];
    let mut sum_f1 = [0.0f32; 4];

    for i in 0..len {
        unsafe {
            // Converte entradas de BF16 para f32.
            let s0 = f32::from_bits((*state_f0.get_unchecked(i) as u32) << 16);
            let s1 = f32::from_bits((*state_f1.get_unchecked(i) as u32) << 16);
            let w = weights.get_unchecked(i);

            // Converte pesos de BF16 para f32.
            let w0 = f32::from_bits((w[0] as u32) << 16);
            let w1 = f32::from_bits((w[1] as u32) << 16);
            let w2 = f32::from_bits((w[2] as u32) << 16);
            let w3 = f32::from_bits((w[3] as u32) << 16);

            // Acumula os resultados.
            sum_f0[0] += w0 * s0;
            sum_f0[1] += w1 * s0;
            sum_f0[2] += w2 * s0;
            sum_f0[3] += w3 * s0;

            sum_f1[0] += w0 * s1;
            sum_f1[1] += w1 * s1;
            sum_f1[2] += w2 * s1;
            sum_f1[3] += w3 * s1;
        }
    }
    (sum_f0, sum_f1)
}

/// Calcula 4 produtos escalares de uma vez para BF16.
/// É um atalho para chamar a função de uma única linha 4 vezes.
pub unsafe fn dot_product_bf16_4x_fallback(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    in_frame: &[u16],
) -> [f32; 4] {
    unsafe {
        [
            dot_product_bf16_fallback(in_frame, w0),
            dot_product_bf16_fallback(in_frame, w1),
            dot_product_bf16_fallback(in_frame, w2),
            dot_product_bf16_fallback(in_frame, w3),
        ]
    }
}

/// Processamento de Matriz (GEMM) em lote.
/// GEMM significa "Multiplicação de Matriz Geral". É o coração das redes neurais.
/// Esta função processa vários "quadros" de áudio de uma vez só.
pub unsafe fn fused_add_gemm_batch_fallback(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32], // "Bias" é um ajuste fixo somado ao final (como o 'b' em y = ax + b).
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    // Descobre quanto espaço cada quadro ocupa na memória.
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    // Para cada quadro no lote...
    for f in 0..num_frames {
        unsafe {
            // ...chama a função que processa um único vetor (GEMV).
            fused_add_gemv_fallback(
                in_frames.get_unchecked(f * in_len..(f + 1) * in_len),
                weights,
                bias,
                out_frames.get_unchecked_mut(f * out_len..(f + 1) * out_len),
                do_bias,
            );
        }
    }
}

/// Multiplicação Matriz-Vetor (GEMV) que SOMA ao resultado existente.
/// Pense nisso como injetar uma nova camada de processamento por cima do que já estava lá.
pub unsafe fn fused_add_gemv_fallback(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    // Para cada "neurônio" de saída...
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        // Começamos com o 'bias' (ajuste) ou com zero.
        let mut sum = if do_bias { b } else { 0.0 };

        // Passamos por todas as entradas e pesos correspondentes.
        for in_c in 0..in_len {
            unsafe {
                // Pega o peso comprimido e descompacta.
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                // Multiplica a entrada pelo peso e acumula.
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        unsafe {
            // IMPORTANTE: Aqui usamos '+=' para SOMAR ao que já estava no buffer de saída.
            *out_frame.get_unchecked_mut(out_c) += sum;
        }
    }
}

/// Multiplicação Matriz-Vetor (GEMV) que SOBRESCREVE o resultado.
/// Diferente da anterior, esta apaga o que estava na saída e coloca o novo valor.
pub unsafe fn gemv_overwrite_fallback(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            unsafe {
                let w =
                    half::f16::from_bits(*weights.get_unchecked(in_c * out_len + out_c)).to_f32();
                sum += *in_frame.get_unchecked(in_c) * w;
            }
        }
        unsafe {
            // IMPORTANTE: Aqui usamos '=' para LIMPAR e definir o novo valor.
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Multiplicação de Matriz Residual em lote.
/// Em redes neurais "Residuais", nós somamos o resultado do processamento ao sinal original.
/// É como dizer: "Mude o som apenas um pouquinho em relação ao que ele era".
pub unsafe fn fused_gemm_residual_batch_fallback(
    in_frames: &[f32],
    weights: &[u16],
    bias: &[f32],
    residual: &[f32], // Este é o sinal "limpo" que será somado ao final.
    out_frames: &mut [f32],
    num_frames: usize,
    do_bias: bool,
) {
    if num_frames == 0 {
        return;
    }
    let in_len = in_frames.len() / num_frames;
    let out_len = out_frames.len() / num_frames;

    for frame_idx in 0..num_frames {
        unsafe {
            for (out_c, &b) in bias.iter().enumerate().take(out_len) {
                let mut sum = if do_bias { b } else { 0.0 };
                for in_c in 0..in_len {
                    let w_bits = *weights.get_unchecked(in_c * out_len + out_c);
                    let w = half::f16::from_bits(w_bits).to_f32();
                    sum += *in_frames.get_unchecked(frame_idx * in_len + in_c) * w;
                }
                // Resultado final = (Processamento da Matriz) + (Sinal Residual original).
                *out_frames.get_unchecked_mut(frame_idx * out_len + out_c) =
                    sum + *residual.get_unchecked(frame_idx * out_len + out_c);
            }
        }
    }
}

/// Multiplicação Matriz-Vetor (Sobrescrita) usando entrada e pesos BF16.
pub unsafe fn gemv_overwrite_bf16_fallback(
    in_frame: &[u16],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for (out_c, &b) in bias.iter().enumerate().take(out_len) {
        let mut sum = if do_bias { b } else { 0.0 };
        for in_c in 0..in_len {
            unsafe {
                // Descompacta entrada BF16 -> f32.
                let s = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
                // Descompacta peso BF16 -> f32 e multiplica.
                sum += s * f32::from_bits(
                    (*weights.get_unchecked(in_c * out_len + out_c) as u32) << 16,
                );
            }
        }
        unsafe {
            *out_frame.get_unchecked_mut(out_c) = sum;
        }
    }
}

/// Acumulação da "Cabeça" (Head) da rede.
/// Nas arquiteturas tipo WaveNet, as várias camadas (blocos) da rede contribuem
/// para um resultado comum chamado "head". Esta função apenas soma essas contribuições.
pub unsafe fn accumulate_head_fallback(dest: &mut [f32], src: &[f32]) {
    let len = core::cmp::min(dest.len(), src.len());
    for i in 0..len {
        unsafe {
            // Soma o conteúdo de 'src' (origem) no 'dest' (destino).
            *dest.get_unchecked_mut(i) += *src.get_unchecked(i);
        }
    }
}

/// Aplica a ativação 'Tanh' e já acumula na saída principal.
/// Tanh (Tangente Hiperbólica) é uma função que "esmaga" qualquer número para
/// ficar entre -1.0 e 1.0. É muito comum em modelagem de amplificadores de guitarra.
pub unsafe fn tanh_and_accumulate_block_fallback(head_input: &mut [f32], block: &mut [f32]) {
    let len = head_input.len();
    for i in 0..len {
        let v = block[i];
        let activated = v.tanh(); // Aplica o "esmagamento".
        block[i] = activated; // Guarda o valor esmagado no bloco.
        head_input[i] += activated; // Soma o mesmo valor no acumulador da "cabeça".
    }
}

/// Ativação Gated (Com "Portão") + Acumulação.
/// Esta técnica usa dois sinais (z1 e z2):
/// 1. z1 passa por um Tanh (contém a "informação").
/// 2. z2 passa por um Sigmoid (serve como um "volume" ou "portão" para o z1).
///
/// No final, multiplicamos os dois. É como se z2 decidisse quanto de z1 vai passar.
pub unsafe fn gated_activation_and_accumulate_block_fallback(
    head_input: &mut [f32],
    block: &mut [f32],
    ch: usize, // Número de canais.
) {
    let num_frames = head_input.len() / ch;
    for f in 0..num_frames {
        let block_offset = f * 2 * ch;
        let head_offset = f * ch;
        for c in 0..ch {
            // O bloco contém z1 e z2 um do lado do outro.
            let z1 = block[block_offset + c];
            let z2 = block[block_offset + ch + c];

            // Ativação complexa: tanh(z1) * sigmoid(z2).
            let activated = z1.tanh() * (1.0 / (1.0 + (-z2).exp()));

            block[block_offset + c] = activated;
            head_input[head_offset + c] += activated;
        }
    }
}

/// Converte números decimais de alta precisão (f32) para o formato compacto (BF16).
/// Isso economiza muita memória e é usado para guardar os "pesos" da rede neural.
pub unsafe fn f32_to_bf16_fallback(src: &[f32], dest: &mut [u16]) {
    for (s, d) in src.iter().zip(dest.iter_mut()) {
        // Pega os 16 bits mais importantes do número e descarta o resto.
        *d = (s.to_bits() >> 16) as u16;
    }
}

/// Aplica o "esmagamento" Tanh em toda uma lista de números.
pub unsafe fn tanh_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = v.tanh();
    }
}

/// Aplica a função "Sigmoid" em toda uma lista de números.
/// A Sigmoid esmaga os números para ficarem entre 0.0 e 1.0.
/// É ótima para criar "portões" ou controles de volume automáticos.
pub unsafe fn sigmoid_slice_fallback(slice: &mut [f32]) {
    for v in slice.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Soma todos os números de uma lista e devolve um único valor final.
pub unsafe fn horizontal_sum_fallback(ptr: *const f32, len: usize) -> f32 {
    let slice = unsafe { core::slice::from_raw_parts(ptr, len) };
    slice.iter().sum()
}

/// Convolução Estéreo (usada no Resampler).
/// Convolução é como aplicar um filtro (como um equalizador).
/// Aqui fazemos isso para os canais Esquerdo (L) e Direito (R) simultaneamente.
pub unsafe fn convolve_stereo_fallback(
    coeffs: *const f32,  // Coeficientes do filtro.
    input_l: *const f32, // Entrada canal esquerdo.
    input_r: *const f32, // Entrada canal direito.
    taps: usize,         // "Tamanho" do filtro.
) -> (f32, f32) {
    let mut sum_l = 0.0f32;
    let mut sum_r = 0.0f32;
    for i in 0..taps {
        let h = *coeffs.add(i);
        sum_l += h * *input_l.add(i);
        sum_r += h * *input_r.add(i);
    }
    (sum_l, sum_r)
}

/// Esta estrutura organiza todas as funções acima para que o resto do programa
/// possa usá-las de forma padronizada através do `SimdMath`.
pub struct ScalarRefMath;

impl SimdMath for ScalarRefMath {
    type V = f32; // No fallback, o "vetor" é apenas um número simples.

    // As funções abaixo apenas "chamam" as versões de fallback que explicamos lá em cima.
    // O comando `#[inline(always)]` é um pedido ao compilador para "colar" o código
    // diretamente onde ele for usado, para ser mais rápido.

    #[inline(always)]
    unsafe fn dot_product(a: &[f32], b: &[u16]) -> f32 {
        unsafe { dot_product_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_bf16(a: &[u16], b: &[u16]) -> f32 {
        unsafe { dot_product_bf16_fallback(a, b) }
    }

    #[inline(always)]
    unsafe fn dot_product_4x_interleaved(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
        unsafe { dot_product_4x_interleaved_fallback(weights, state) }
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
        unsafe { dot_product_4x_interleaved_dual_frame_fallback(weights, state_f0, state_f1) }
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

    #[inline(always)]
    unsafe fn fused_add_gemv(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { fused_add_gemv_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

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
            fused_add_gemm_batch_fallback(in_frames, weights, bias, out_frames, num_frames, do_bias)
        }
    }

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
            fused_gemm_residual_batch_fallback(
                in_frames, weights, bias, residual, out_frames, num_frames, do_bias,
            )
        }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite(
        in_frame: &[f32],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

    #[inline(always)]
    unsafe fn gemv_overwrite_bf16(
        in_frame: &[u16],
        weights: &[u16],
        bias: &[f32],
        out_frame: &mut [f32],
        do_bias: bool,
    ) {
        unsafe { gemv_overwrite_bf16_fallback(in_frame, weights, bias, out_frame, do_bias) }
    }

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
        // Divide o bloco gigante de pesos em 4 partes (os "portões" da LSTM).
        unsafe {
            gemv_4gate_fallback(
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
        unsafe { accumulate_head_fallback(dest, src) }
    }

    #[inline(always)]
    unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
        unsafe { tanh_and_accumulate_block_fallback(head_input, block) }
    }

    #[inline(always)]
    unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
        let len = l.len().min(r.len());
        if len == 0 {
            return 0.0;
        }
        let mut sum_l = 0.0f32;
        let mut sum_r = 0.0f32;
        for i in 0..len {
            // Calcula o quadrado de cada amostra (energia).
            sum_l += l[i] * l[i];
            sum_r += r[i] * r[i];
        }
        // Retorna a média da energia do canal mais forte.
        (sum_l / len as f32).max(sum_r / len as f32)
    }

    #[inline(always)]
    unsafe fn gated_activation_and_accumulate_block(
        head_input: &mut [f32],
        block: &mut [f32],
        ch: usize,
    ) {
        unsafe { gated_activation_and_accumulate_block_fallback(head_input, block, ch) }
    }

    #[inline(always)]
    unsafe fn f32_to_bf16(src: &[f32], dest: &mut [u16]) {
        unsafe { f32_to_bf16_fallback(src, dest) }
    }

    #[inline(always)]
    unsafe fn store_bf16(ptr: *mut u16, _v: Self::V) {
        // Guarda um único valor BF16 na memória.
        *ptr = (_v.to_bits() >> 16) as u16;
    }

    #[inline(always)]
    unsafe fn tanh_slice(slice: &mut [f32]) {
        unsafe { tanh_slice_fallback(slice) }
    }

    #[inline(always)]
    unsafe fn sigmoid_slice(slice: &mut [f32]) {
        unsafe { sigmoid_slice_fallback(slice) }
    }

    #[inline(always)]
    unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
        unsafe { horizontal_sum_fallback(ptr, N) }
    }

    #[inline(always)]
    unsafe fn activation_tanh_block(buf: &mut [f32]) {
        unsafe { tanh_slice_fallback(buf) }
    }

    /// Implementação fundida dos portões de uma LSTM (Long Short-Term Memory).
    /// Esta é uma das partes mais complexas do processamento de áudio inteligente.
    /// Ela controla o que a rede deve "lembrar" ou "esquecer" do som passado.
    #[inline(always)]
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],        // Saídas dos cálculos de matriz.
        cell_state: &mut [f32],   // A "memória" interna da rede.
        hidden_state: &mut [f32], // A "saída" que vai para a próxima camada.
        hidden_size: usize,
    ) {
        for j in 0..hidden_size {
            // Calcula os 4 controles da LSTM:
            // - Input (entrada)
            // - Forget (esquecer)
            // - G (novo conteúdo)
            // - Output (saída)
            let sig_i = 1.0 / (1.0 + (-gates[j]).exp());
            let sig_f = 1.0 / (1.0 + (-gates[j + hidden_size]).exp());
            let tanh_g = gates[j + 2 * hidden_size].tanh();
            let sig_o = 1.0 / (1.0 + (-gates[j + 3 * hidden_size]).exp());

            // Atualiza a memória interna: decide o que manter e o que adicionar de novo.
            let new_cs = sig_f * cell_state[j] + sig_i * tanh_g;
            cell_state[j] = new_cs;

            // Gera o estado oculto final aplicando Tanh na memória e multiplicando pela porta de saída.
            hidden_state[j] = sig_o * new_cs.tanh();
        }
    }

    #[inline(always)]
    unsafe fn convolve_stereo(
        coeffs: *const f32,
        input_l: *const f32,
        input_r: *const f32,
        taps: usize,
    ) -> (f32, f32) {
        unsafe { convolve_stereo_fallback(coeffs, input_l, input_r, taps) }
    }

    /// Aplica volume (ganho) e avisa se o som "estourou" (clipping).
    /// Clipping acontece quando o som fica alto demais para o computador representar,
    /// gerando aquela distorção digital desagradável.
    #[inline(always)]
    unsafe fn apply_gain_and_detect_clipping_stereo(
        left: &mut [f32],
        right: &mut [f32],
        gain: f32,
    ) -> bool {
        let n = core::cmp::min(left.len(), right.len());
        let mut clipped = false;
        for i in 0..n {
            let vl = *left.get_unchecked(i) * gain;
            let vr = *right.get_unchecked(i) * gain;

            *left.get_unchecked_mut(i) = vl;
            *right.get_unchecked_mut(i) = vr;

            // Verifica se o sinal passou do limite de 1.0 (ou -1.0).
            if !clipped && (vl.abs() > 1.0 || vr.abs() > 1.0) {
                clipped = true;
            }
        }
        clipped
    }

    /// Aplica volume constante nos dois canais.
    #[inline(always)]
    unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
        let n = core::cmp::min(left.len(), right.len());
        for i in 0..n {
            *left.get_unchecked_mut(i) *= gain;
            *right.get_unchecked_mut(i) *= gain;
        }
    }

    /// Aplica volume constante em um buffer simples.
    #[inline(always)]
    unsafe fn apply_gain(data: &mut [f32], gain: f32) {
        unsafe { apply_gain_fallback(data, gain) }
    }

    /// Aplica um "Fade" (Aumento ou diminuição gradual de volume).
    /// Isso evita estalos (clicks) quando mudamos o volume de repente.
    #[inline(always)]
    unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
        let n = core::cmp::min(left.len(), right.len());
        let mut g = start;
        for i in 0..n {
            *left.get_unchecked_mut(i) *= g;
            *right.get_unchecked_mut(i) *= g;
            g += step; // Aumenta o volume um passinho por vez.
        }
    }

    /// Soma final da arquitetura WaveNet.
    /// Pega as contribuições de todas as camadas e aplica uma escala final.
    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn batch_wavenet_head_sum<const HEAD: usize>(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        scale: f32,
    ) {
        let num_frames = output.len();
        for i in 0..num_frames {
            let start = i * HEAD;
            let mut sum = 0.0;
            for j in 0..HEAD {
                sum += *head1.get_unchecked(start + j);
            }
            // Soma a segunda cabeça e aplica o volume final.
            output[i] = (sum + *head2.get_unchecked(i)) * scale;
        }
    }

    /// Processamento de Matriz em lote que sobrescreve a saída.
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
            gemv_overwrite_fallback(in_slice, weights, bias, out_slice, do_bias);
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
            gemv_overwrite_bf16_fallback(in_slice, weights, bias, out_slice, do_bias);
        }
    }

    /// Soma final da WaveNet com tamanho de cabeça dinâmico.
    #[inline(always)]
    unsafe fn batch_wavenet_head_sum_dyn(
        head1: &[f32],
        head2: &[f32],
        output: &mut [f32],
        head: usize,
        scale: f32,
    ) {
        let num_frames = output.len();
        for i in 0..num_frames {
            let mut sum = 0.0;
            let start = i * head;
            for j in 0..head {
                sum += head1[start + j];
            }
            output[i] = (sum + head2[i]) * scale;
        }
    }
}

/// Fallback para os 4 "portões" da LSTM.
/// Cada portão decide uma coisa diferente: o que entra, o que sai, o que apaga...
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_fallback(
    in_frame: &[f32],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    unsafe {
        // Processa cada um dos 4 portões separadamente.
        gemv_overwrite_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}

/// Versão BF16 para os 4 portões da LSTM.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_4gate_bf16_fallback(
    in_frame: &[u16],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    unsafe {
        gemv_overwrite_bf16_fallback(
            in_frame,
            w0,
            &bias[0..out_len],
            &mut out[0..out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w1,
            &bias[out_len..2 * out_len],
            &mut out[out_len..2 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w2,
            &bias[2 * out_len..3 * out_len],
            &mut out[2 * out_len..3 * out_len],
            do_bias,
        );
        gemv_overwrite_bf16_fallback(
            in_frame,
            w3,
            &bias[3 * out_len..4 * out_len],
            &mut out[3 * out_len..4 * out_len],
            do_bias,
        );
    }
}

/// Aplica um volume (ganho) multiplicando cada amostra pelo valor desejado.
pub unsafe fn apply_gain_fallback(data: &mut [f32], gain: f32) {
    for x in data.iter_mut() {
        *x *= gain;
    }
}

// Inclusão dos testes para garantir que tudo o que explicamos acima funciona de verdade!
#[cfg(test)]
#[path = "scalar_ref_test.rs"]
mod scalar_ref_test;
