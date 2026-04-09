#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn test() -> __m512 {
    let zeros = _mm512_setzero_ps();
    let ones = _mm512_set1_ps(1.0);
    let rsq = _mm512_rsqrt14_ps(ones);
    let m = _mm512_mul_ps(zeros, ones);
    let f = _mm512_fmadd_ps(ones, ones, zeros);
    let s = _mm512_sub_ps(ones, zeros);
    let a = _mm512_add_ps(ones, zeros);
    _mm512_mul_ps(rsq, _mm512_add_ps(m, _mm512_add_ps(f, _mm512_add_ps(s, a))))
}
fn main() {}
