// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration Test for IR Cabsim Heap-Audit Coverage (RT-Safety).
//!
//! Validates zero heap allocations on the UPOLS convolution hot-path
//! using CountingAllocator.

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use nam_rs::dsp::cabsim::conv::ConvEngine;
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
    // Synthetic IR helpers
    // =============================================================================

    fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
        let sample_rate = 48000u32;
        (0..len)
            .map(|n| {
                let t = n as f32 / sample_rate as f32;
                (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
            })
            .collect()
    }

    // =============================================================================
    // Audit
    // =============================================================================

    fn run_cabsim_audit(ir_len: usize, partition_size: usize, label: &str) {
        let ir = synth_ir(ir_len, 440.0, 10.0);

        let mut engine = ConvEngine::new(&ir, partition_size);

        let iters = 2000;
        let mut input = vec![0.0f32; partition_size];
        let mut output = vec![0.0f32; partition_size];

        // Warm-up: fill FDL
        for i in 0..engine.num_partitions().max(1) {
            for (j, v) in input.iter_mut().enumerate() {
                *v = ((i * partition_size + j) as f32 * 0.001).sin();
            }
            engine.process(&input, &mut output);
        }

        let count = {
            let _guard = TrackingGuard::new();
            for iter in 0..iters {
                for (j, v) in input.iter_mut().enumerate() {
                    *v = ((iter * partition_size + j) as f32 * 0.001).sin();
                }
                engine.process(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                );
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Heap allocations detected on cabsim-{label} hot-path! count={count}",
        );
    }

    #[test]
    fn test_cabsim_short_ir_heap_audit() {
        run_cabsim_audit(64, 64, "short");
    }

    #[test]
    fn test_cabsim_medium_ir_heap_audit() {
        run_cabsim_audit(512, 64, "medium");
    }

    #[test]
    fn test_cabsim_long_ir_heap_audit() {
        run_cabsim_audit(4096, 128, "long");
    }

    #[test]
    fn test_cabsim_passthrough_heap_audit() {
        let mut engine = ConvEngine::new(&[], 64);
        assert!(engine.is_passthrough());

        let iters = 2000;
        let mut input = vec![0.0f32; 64];
        let mut output = vec![0.0f32; 64];

        for (j, v) in input.iter_mut().enumerate() {
            *v = (j as f32 * 0.01).sin();
        }
        engine.process(&input, &mut output);

        let count = {
            let _guard = TrackingGuard::new();
            for iter in 0..iters {
                for (j, v) in input.iter_mut().enumerate() {
                    *v = ((iter * 64 + j) as f32 * 0.01).sin();
                }
                engine.process(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                );
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Heap allocations detected on cabsim-passthrough hot-path! count={count}",
        );
    }
}
