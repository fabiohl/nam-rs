// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Operações matemáticas básicas e utilitários SIMD de baixo nível.

use core::arch::x86_64::*;

/// Converte F32 para os bits de um BF16 (truncamento simples).
#[inline(always)]
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// Converte um vetor de números f32 (normais) para bf16 (compactos) via AVX-512.
/// O formato bf16 ocupa metade do espaço, mas mantém o alcance dos números f32,
/// sendo ideal para modelos de inteligência artificial rápidos.
///
/// # Safety
/// `src` e `dest` devem ser slices válidos. `dest` deve ter pelo menos `src.len()` elementos.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn f32_to_bf16_avx512(src: &[f32], dest: &mut [u16]) {
    let n = core::cmp::min(src.len(), dest.len());
    let mut i = 0;
    // Processa 16 conversões de uma vez.
    while i + 16 <= n {
        unsafe {
            let v = _mm512_loadu_ps(src.as_ptr().add(i)); // Carrega 16 números f32.
            let v_i = _mm512_castps_si512(v); // Trata como inteiros para manipular os bits.
            let v_shifted = _mm512_srli_epi32(v_i, 16); // Descarta a parte menos importante (precisão extra).
            let packed = _mm512_cvtepi32_epi16(v_shifted); // Compacta para 16 bits cada.
            _mm256_storeu_si256(dest.as_mut_ptr().add(i) as *mut __m256i, packed); // Salva 16 números bf16.
        }
        i += 16;
    }
    // Converte o resto manualmente.
    while i < n {
        unsafe {
            *dest.get_unchecked_mut(i) = (*src.get_unchecked(i)).to_bits() as u16;
        }
        i += 1;
    }
}

/// Prefetch adaptativo baseado na dilatação (Causal Conv1D).
///
/// # Safety
/// O ponteiro `ptr` deve ser válido ou estar dentro da margem de segurança do buffer.
#[inline(always)]
pub unsafe fn adaptive_prefetch_f32(ptr: *const f32, dilation: usize) {
    if dilation <= 8 {
        // Hardware prefetcher domina.
    } else if dilation <= 64 {
        // Traz para L1 (reuso iminente).
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
        }
    } else {
        // Dilatações massivas: traz para L2 para poupar o L1 de evicção agressiva.
        unsafe {
            _mm_prefetch::<_MM_HINT_T1>(ptr as *const i8);
        }
    }
}

/// Prefetch adaptativo de 2 estágios para dilatações extremas (Causal Conv1D).
///
/// # Safety
/// Os ponteiros devem ser válidos ou estar dentro da margem de segurança do buffer.
#[inline(always)]
pub unsafe fn adaptive_prefetch_2stage_f32(
    ptr_next: *const f32,
    ptr_next_next: *const f32,
    dilation: usize,
) {
    if dilation >= 128 {
        unsafe {
            // Traz o próximo tap para L1 (uso imediato no próximo k)
            _mm_prefetch::<_MM_HINT_T0>(ptr_next as *const i8);
            // Traz o tap subsequente para L2 (prepara para k+2)
            _mm_prefetch::<_MM_HINT_T1>(ptr_next_next as *const i8);
        }
    } else {
        // Fallback para prefetch simples para dilatações menores
        unsafe {
            adaptive_prefetch_f32(ptr_next, dilation);
        }
    }
}

/// Assinatura unificada para estratégias de prefetch.
pub type PrefetchFn =
    unsafe fn(base_ptr: *const f32, step: usize, k: usize, k_limit: usize, dilation: usize);

/// Estratégia de prefetch simples para dilatações pequenas/médias.
///
/// # Safety
/// O ponteiro base deve ser válido.
pub unsafe fn prefetch_strategy_simple(
    base_ptr: *const f32,
    _step: usize,
    _k: usize,
    _k_limit: usize,
    _dilation: usize,
) {
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
            base_ptr.add(16).cast(),
        );
    }
}

/// Estratégia de prefetch de 2 estágios para dilatações extremas.
///
/// # Safety
/// O ponteiro base e os saltos calculados devem ser válidos.
pub unsafe fn prefetch_strategy_2stage(
    base_ptr: *const f32,
    step: usize,
    k: usize,
    k_limit: usize,
    _dilation: usize,
) {
    if k + 1 < k_limit {
        unsafe {
            let ptr_n1 = base_ptr.add(step);
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr_n1.cast());

            if k + 2 < k_limit {
                let ptr_n2 = base_ptr.add(2 * step);
                core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T1 }>(
                    ptr_n2.cast(),
                );
            }
        }
    }
}

/// Habilita DAZ (Denormals-Are-Zero) e FTZ (Flush-To-Zero) no registrador MXCSR.
///
/// # Safety
/// SSE2 é implícito em x86-64.
pub unsafe fn set_daz_ftz() {
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut mxcsr);
        mxcsr |= 0x8040;
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &mxcsr);
    }
}
