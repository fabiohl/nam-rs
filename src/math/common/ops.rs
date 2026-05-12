// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Operações matemáticas básicas e utilitários SIMD de baixo nível.

use core::arch::x86_64::*;

/// Converte F32 para os bits de um BF16 (truncamento simples).
#[inline(always)]
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
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
