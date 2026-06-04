// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration Test for NamResampler Heap-Audit Coverage.

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use nam_rs::dsp::resampler::NamResampler;
    #[cfg(not(feature = "clap-plugin"))]
    use std::alloc::{GlobalAlloc, Layout, System};
    #[cfg(not(feature = "clap-plugin"))]
    use std::sync::atomic::{AtomicI32, AtomicUsize};
    use std::sync::atomic::Ordering;

    // =============================================================================
    // Counting Allocator for Zero-Allocation Verification
    // =============================================================================
    // Only registered if the library itself doesn't register a global allocator.
    // The library registers it when (clap-plugin + heap-audit) are enabled.
    
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

    // Guard to safely enable/disable counting (even in panics).
    struct TrackingGuard {
        #[cfg(feature = "clap-plugin")]
        _inner: nam_rs::clap::heap_audit::TrackingGuard,
    }

    impl TrackingGuard {
        fn new() -> Self {
            #[cfg(feature = "clap-plugin")]
            {
                Self {
                    _inner: nam_rs::clap::heap_audit::TrackingGuard::new(),
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
            nam_rs::clap::heap_audit::ALLOC_COUNT.load(Ordering::Relaxed)
        }
        #[cfg(not(feature = "clap-plugin"))]
        {
            ALLOC_COUNT.load(Ordering::Relaxed)
        }
    }

    fn generate_noise(samples: usize) -> Vec<f32> {
        (0..samples)
            .map(|i| (i as f32 * 0.1234).sin())
            .collect()
    }

    fn run_resampler_audit(pw_rate: u32, nam_rate: u32, chunk_size: usize, is_mono: bool) {
        let mut resampler = NamResampler::new(pw_rate, nam_rate, chunk_size)
            .expect("Failed to create resampler");

        let in_l = generate_noise(chunk_size);
        let in_r = generate_noise(chunk_size);
        let mut out_l = vec![0.0f32; chunk_size * 4];
        let mut out_r = vec![0.0f32; chunk_size * 4];

        // Warmup
        for _ in 0..5 {
            if is_mono {
                resampler.process_input_mono(&in_l, &mut out_l, &mut out_r);
                resampler.process_output_mono(&in_l, &mut out_l, &mut out_r);
            } else {
                resampler.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
                resampler.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
            }
        }

        // Audit
        let count = {
            let _guard = TrackingGuard::new();
            for _ in 0..1000 {
                if is_mono {
                    resampler.process_input_mono(&in_l, &mut out_l, &mut out_r);
                    resampler.process_output_mono(&in_l, &mut out_l, &mut out_r);
                } else {
                    resampler.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
                    resampler.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
                }
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Allocations detected during resampling! (pw_rate={}, nam_rate={}, mono={})",
            pw_rate, nam_rate, is_mono
        );
    }

    #[test]
    fn test_resampler_heap_audit_all_scenarios() {
        let scenarios = [
            // ratio 1:1 (passthrough)
            (48000, 48000, 32, true),
            (48000, 48000, 32, false),
            (48000, 48000, 1024, true),
            (48000, 48000, 1024, false),
            
            // 96k -> 48k (downsample)
            (96000, 48000, 32, true),
            (96000, 48000, 32, false),
            (96000, 48000, 1024, true),
            (96000, 48000, 1024, false),
            
            // 44.1k -> 48k (upsample)
            (44100, 48000, 32, true),
            (44100, 48000, 32, false),
            (44100, 48000, 1024, true),
            (44100, 48000, 1024, false),
        ];

        for &(pw, nam, chunk, is_mono) in &scenarios {
            run_resampler_audit(pw, nam, chunk, is_mono);
        }
    }
}
