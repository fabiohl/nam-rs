// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration Test for WaveNetA2 Heap-Audit Coverage (RT-Safety).
//!
//! Validates zero heap allocations on the A2 hot-path using CountingAllocator.

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use nam_rs::models::a2::{A2_KERNEL_SIZES, WaveNetA2};
    #[cfg(not(feature = "clap-plugin"))]
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::Ordering;
    #[cfg(not(feature = "clap-plugin"))]
    use std::sync::atomic::{AtomicI32, AtomicUsize};

    // =============================================================================
    // Counting Allocator for Zero-Allocation Verification
    // =============================================================================

    #[cfg(not(feature = "clap-plugin"))]
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    #[cfg(not(feature = "clap-plugin"))]
    static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0);

    #[cfg(not(feature = "clap-plugin"))]
    struct CountingAllocator;

    #[cfg(not(feature = "clap-plugin"))]
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            if tid == TRACKING_THREAD.load(Ordering::Relaxed) {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[cfg(not(feature = "clap-plugin"))]
    #[global_allocator]
    static GLOBAL: CountingAllocator = CountingAllocator;

    struct TrackingGuard {
        #[cfg(feature = "clap-plugin")]
        _inner: nam_rs::common::alloc_audit::TrackingGuard,
    }

    impl TrackingGuard {
        fn new() -> Self {
            #[cfg(feature = "clap-plugin")]
            {
                Self {
                    _inner: nam_rs::common::alloc_audit::TrackingGuard::new(),
                }
            }
            #[cfg(not(feature = "clap-plugin"))]
            {
                let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
                TRACKING_THREAD.store(tid, Ordering::Relaxed);
                ALLOC_COUNT.store(0, Ordering::Relaxed);
                Self {}
            }
        }
    }

    impl Drop for TrackingGuard {
        fn drop(&mut self) {
            #[cfg(not(feature = "clap-plugin"))]
            {
                TRACKING_THREAD.store(0, Ordering::Relaxed);
            }
        }
    }

    fn get_alloc_count() -> usize {
        #[cfg(feature = "clap-plugin")]
        {
            nam_rs::common::alloc_audit::ALLOC_COUNT.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "clap-plugin"))]
        {
            ALLOC_COUNT.load(Ordering::Relaxed)
        }
    }

    // =============================================================================
    // A2 Helper — Synthetic Weights
    // =============================================================================

    /// Computes total A2 weight count for a given CH.
    const fn a2_total_weights<const CH: usize>() -> usize {
        let mut total = CH;
        let mut layer_idx = 0;
        while layer_idx < 23 {
            let k = A2_KERNEL_SIZES[layer_idx];
            total += CH * CH * k + CH + CH + CH * CH + CH;
            layer_idx += 1;
        }
        total += 16 * CH + 2;
        total
    }

    /// Assembles synthetic A2 weight stream.
    fn a2_synth_weights<const CH: usize>(weight_val: f32) -> Vec<f32> {
        let num_weights = a2_total_weights::<CH>();
        let mut w = Vec::with_capacity(num_weights);

        w.extend(std::iter::repeat_n(weight_val, CH));
        for &k in &A2_KERNEL_SIZES {
            w.extend(std::iter::repeat_n(weight_val, CH * CH * k));
            w.extend(std::iter::repeat_n(0.0f32, CH));
            w.extend(std::iter::repeat_n(weight_val, CH));
            w.extend(std::iter::repeat_n(weight_val, CH * CH));
            w.extend(std::iter::repeat_n(0.0f32, CH));
        }
        w.extend(std::iter::repeat_n(weight_val, 16 * CH));
        w.push(0.0);
        w.push(0.02);

        assert_eq!(w.len(), num_weights);
        w
    }

    /// Builds a synthetically-weighted A2 model.
    fn build_a2<const CH: usize>(weight_val: f32) -> WaveNetA2<CH> {
        let weights = a2_synth_weights::<CH>(weight_val);
        let mut model = WaveNetA2::<CH>::new();
        model.set_weights(&weights).expect("A2 set_weights failed");
        model
    }

    // =============================================================================
    // Audit Tests
    // =============================================================================

    fn run_a2_audit<const CH: usize>(label: &str) {
        let mut model = build_a2::<CH>(0.01);
        model.prewarm();

        let block_sizes = [1usize, 16, 32, 48, 64];

        // Pre-allocate buffers outside the audit guard
        let mut inputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();
        let mut outputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();

        let mut sample_offset = 0usize;
        for (bi, &block_size) in block_sizes.iter().enumerate() {
            for (i, v) in inputs[bi].iter_mut().enumerate().take(block_size) {
                let t = (sample_offset + i) as f32;
                *v = (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin();
            }
            model.process(&inputs[bi], &mut outputs[bi]);
            sample_offset += block_size;
        }

        // Audit — must have zero allocations on hot-path
        let iters = 1000;
        let count = {
            let _guard = TrackingGuard::new();
            for _ in 0..iters {
                for (bi, &block_size) in block_sizes.iter().enumerate() {
                    for (i, v) in inputs[bi].iter_mut().enumerate().take(block_size) {
                        let t = (sample_offset + i) as f32;
                        *v = (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin();
                    }
                    model.process(
                        std::hint::black_box(&inputs[bi]),
                        std::hint::black_box(&mut outputs[bi]),
                    );
                    sample_offset += block_size;
                }
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Heap allocations detected on A2-{label} hot-path! count={}",
            count
        );
    }

    #[test]
    fn test_a2_full_heap_audit() {
        run_a2_audit::<8>("Full");
    }

    #[test]
    fn test_a2_lite_heap_audit() {
        run_a2_audit::<3>("Lite");
    }
}
