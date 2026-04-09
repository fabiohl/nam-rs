#[inline(never)]
unsafe fn dot_product_direct(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..a.len() { sum += a[i] * b[i]; }
    sum
}

#[inline(always)]
unsafe fn run_direct(a: &[f32], b: &[f32], iters: usize) -> f32 {
    let mut s = 0.0;
    for _ in 0..iters { s += dot_product_direct(a, b); }
    s
}

unsafe fn run_ptr(a: &[f32], b: &[f32], iters: usize, f: unsafe fn(&[f32],&[f32])->f32) -> f32 {
    let mut s = 0.0;
    for _ in 0..iters { s += f(a, b); }
    s
}

fn main() {
    let a = vec![1.0; 16];
    let b = vec![2.0; 16];
    let iters = 10_000_000;
    
    let t0 = std::time::Instant::now();
    let s1 = unsafe { run_direct(&a, &b, iters) };
    let d1 = t0.elapsed();

    let t1 = std::time::Instant::now();
    let s2 = unsafe { run_ptr(&a, &b, iters, dot_product_direct) };
    let d2 = t1.elapsed();

    println!("Direct: {:?} (s1={}), Ptr: {:?} (s1={})", d1, s1, d2, s2);
}
