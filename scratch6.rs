#[inline(always)]
unsafe fn print_feature() {
    if std::is_x86_feature_detected!("avx512f") {
        println!("Has AVX512");
    } else {
        println!("No AVX512");
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn test_avx512() {
    print_feature();
}

fn main() {
    unsafe { test_avx512(); }
}
