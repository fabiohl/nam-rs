#[inline(always)]
unsafe fn print_feature() {
    if cfg!(target_feature = "avx512f") {
        println!("AVX512");
    } else {
        println!("AVX2");
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn test_avx512() {
    print_feature();
}

unsafe fn test_avx2() {
    print_feature();
}

fn main() {
    unsafe {
        test_avx512();
        test_avx2();
    }
}
