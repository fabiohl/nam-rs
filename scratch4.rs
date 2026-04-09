use std::arch::x86_64::*;

#[target_feature(enable = "avx512f")]
unsafe fn test() -> f32 {
    let zeros = _mm512_setzero_ps();
    _mm512_reduce_add_ps(zeros)
}
fn main() {}
