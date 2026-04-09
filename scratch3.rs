#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512f")]
unsafe fn test() -> __m512 {
    let zeros = _mm512_setzero_ps();
    let ones = _mm512_set1_ps(1.0);
    _mm512_add_ps(zeros, ones)
}
fn main() {}
