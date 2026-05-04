// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! [T21] Operações matemáticas básicas e utilitários SIMD.

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

/// Habilita DAZ (Denormals-Are-Zero) e FTZ (Flush-To-Zero) no registrador MXCSR.
///
/// # Safety
/// O chamador deve garantir que a CPU suporte SSE2 (implícito em x86-64).
pub unsafe fn set_daz_ftz() {
    // 0x8040 = bit 15 (FTZ) | bit 6 (DAZ)
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut mxcsr);
        mxcsr |= 0x8040;
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &mxcsr);
    }
}

/// Calcula a energia (Mean Square) de um bloco via AVX2.
/// $E = \frac{1}{N} \sum x_i^2$
///
/// # Safety
/// O slice `data` deve ser válido.
pub unsafe fn compute_energy_avx2(data: &[f32]) -> f32 {
    let len = data.len();
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();

        while i + 16 <= len {
            let v0 = _mm256_loadu_ps(data.as_ptr().add(i));
            let v1 = _mm256_loadu_ps(data.as_ptr().add(i + 8));

            sum0 = _mm256_fmadd_ps(v0, v0, sum0);
            sum1 = _mm256_fmadd_ps(v1, v1, sum1);
            i += 16;
        }

        while i + 8 <= len {
            let v = _mm256_loadu_ps(data.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(v, v, sum0);
            i += 8;
        }

        let sum = _mm256_add_ps(sum0, sum1);
        let hi = _mm256_extractf128_ps(sum, 1);
        let lo = _mm256_castps256_ps128(sum);
        let s128 = _mm_add_ps(lo, hi);

        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let r = _mm_add_ss(sums, shuf2);

        let mut total_sum = 0.0f32;
        _mm_store_ss(&mut total_sum, r);

        while i < len {
            total_sum += data[i] * data[i];
            i += 1;
        }
        total_sum / (len as f32)
    }
}

/// Calcula a diferença absoluta máxima entre dois blocos via AVX2.
/// $\max(|L_i - R_i|)$
///
/// # Safety
/// Os slices `a` e `b` devem ter o mesmo tamanho.
pub unsafe fn compute_max_diff_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = core::cmp::min(a.len(), b.len());
    if len == 0 {
        return 0.0;
    }
    let mut i = 0;
    unsafe {
        let mut max_v = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0f32);

        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            max_v = _mm256_max_ps(max_v, abs_diff);
            i += 8;
        }

        let hi = _mm256_extractf128_ps(max_v, 1);
        let lo = _mm256_castps256_ps128(max_v);
        let m128 = _mm_max_ps(lo, hi);

        let shuf = _mm_shuffle_ps(m128, m128, 0xEE);
        let m64 = _mm_max_ps(m128, shuf);
        let shuf2 = _mm_shuffle_ps(m64, m64, 0x55);
        let m32 = _mm_max_ps(m64, shuf2);

        let mut max_diff = 0.0f32;
        _mm_store_ss(&mut max_diff, m32);

        while i < len {
            let d = (a[i] - b[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            i += 1;
        }

        max_diff
    }
}
