use super::helpers::{bf16x4_to_f32x4, bf16x16_to_f32x16};
use core::arch::x86_64::*;

/// Dot product interleaved 4x with BF16 state — AVX-512 native accumulation.
///
/// Weights are stored in `[[u16; 4]]` interleaved format (4 channels grouped).
/// State is `&[u16]` in BF16. Accumulation happens entirely in f32 ZMM registers.
///
/// Uses 8 alternating ZMM accumulators to break FMA dependency chains,
/// and permutexvar to broadcast each state value to all 4 channels.
#[target_feature(enable = "avx512f,avx512vl")]
#[inline]
pub unsafe fn dot_product_4x_interleaved_avx512_bf16(
    weights: &[[u16; 4]],
    state: &[u16],
) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut i = 0;

    unsafe {
        let perm_idx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

        let mut sum_a0 = _mm512_setzero_ps();
        let mut sum_a1 = _mm512_setzero_ps();
        let mut sum_a2 = _mm512_setzero_ps();
        let mut sum_a3 = _mm512_setzero_ps();
        let mut sum_b0 = _mm512_setzero_ps();
        let mut sum_b1 = _mm512_setzero_ps();
        let mut sum_b2 = _mm512_setzero_ps();
        let mut sum_b3 = _mm512_setzero_ps();

        while i + 32 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().add(i + 48) as *const i8);

            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i))),
            );
            let w_a = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 4))),
            );
            let w_b = bf16x16_to_f32x16(weights.as_ptr().add(i + 4) as *const u16);
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 8))),
            );
            let w_c = bf16x16_to_f32x16(weights.as_ptr().add(i + 8) as *const u16);
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 12))),
            );
            let w_d = bf16x16_to_f32x16(weights.as_ptr().add(i + 12) as *const u16);
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            let s_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 16))),
            );
            let w_e = bf16x16_to_f32x16(weights.as_ptr().add(i + 16) as *const u16);
            sum_b0 = _mm512_fmadd_ps(w_e, s_e, sum_b0);

            let s_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 20))),
            );
            let w_f = bf16x16_to_f32x16(weights.as_ptr().add(i + 20) as *const u16);
            sum_b1 = _mm512_fmadd_ps(w_f, s_f, sum_b1);

            let s_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 24))),
            );
            let w_g = bf16x16_to_f32x16(weights.as_ptr().add(i + 24) as *const u16);
            sum_b2 = _mm512_fmadd_ps(w_g, s_g, sum_b2);

            let s_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 28))),
            );
            let w_h = bf16x16_to_f32x16(weights.as_ptr().add(i + 28) as *const u16);
            sum_b3 = _mm512_fmadd_ps(w_h, s_h, sum_b3);

            i += 32;
        }

        while i + 16 <= len {
            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i))),
            );
            let w_a = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 4))),
            );
            let w_b = bf16x16_to_f32x16(weights.as_ptr().add(i + 4) as *const u16);
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 8))),
            );
            let w_c = bf16x16_to_f32x16(weights.as_ptr().add(i + 8) as *const u16);
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i + 12))),
            );
            let w_d = bf16x16_to_f32x16(weights.as_ptr().add(i + 12) as *const u16);
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            i += 16;
        }

        while i + 4 <= len {
            let s = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(bf16x4_to_f32x4(state.as_ptr().add(i))),
            );
            let w = bf16x16_to_f32x16(weights.as_ptr().add(i) as *const u16);
            sum_a0 = _mm512_fmadd_ps(w, s, sum_a0);
            i += 4;
        }

        let s_ab = _mm512_add_ps(_mm512_add_ps(sum_a0, sum_a1), _mm512_add_ps(sum_a2, sum_a3));
        let s_all = _mm512_add_ps(
            s_ab,
            _mm512_add_ps(_mm512_add_ps(sum_b0, sum_b1), _mm512_add_ps(sum_b2, sum_b3)),
        );

        let lo = _mm512_extractf32x4_ps(s_all, 0);
        let hi0 = _mm512_extractf32x4_ps(s_all, 1);
        let hi1 = _mm512_extractf32x4_ps(s_all, 2);
        let hi2 = _mm512_extractf32x4_ps(s_all, 3);
        let mut sum128 = _mm_add_ps(_mm_add_ps(lo, hi0), _mm_add_ps(hi1, hi2));

        // Scalar tail: convert remaining BF16 state values to f32 and accumulate.
        // Each weight group has 4 interleaved channels.
        while i < len {
            let si = f32::from_bits((*state.get_unchecked(i) as u32) << 16);
            let w_ptr = weights.as_ptr().add(i) as *const u16;
            let w0 = f32::from_bits((*w_ptr as u32) << 16);
            let w1 = f32::from_bits((*w_ptr.add(1) as u32) << 16);
            let w2 = f32::from_bits((*w_ptr.add(2) as u32) << 16);
            let w3 = f32::from_bits((*w_ptr.add(3) as u32) << 16);
            let s_vec = _mm_set_ps(w3 * si, w2 * si, w1 * si, w0 * si);
            sum128 = _mm_add_ps(sum128, s_vec);
            i += 1;
        }

        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum128);
        out
    }
}
