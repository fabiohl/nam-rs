// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
pub(crate) mod infra {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

    /// Global counter of how many times memory was requested.
    pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    /// Stores which "processing line" (thread) we are currently watching.
    pub static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0);

    /// The "Memory Watchdog": intercepts all memory requests from the program.
    pub struct CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // Finds out who is requesting memory right now.
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            // If it's the processing line we are watching, increment the counter.
            if tid == TRACKING_THREAD.load(Ordering::Relaxed) {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            // Makes the memory request normally through the system.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // Returns the memory to the system.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    /// The "Switch": turns on the watchdog when created, turns it off when destroyed.
    pub struct TrackingGuard;
    impl TrackingGuard {
        /// Starts watching the current processing line.
        pub fn new() -> Self {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            TRACKING_THREAD.store(tid, Ordering::Relaxed);
            ALLOC_COUNT.store(0, Ordering::Relaxed);
            Self
        }
    }
    impl Drop for TrackingGuard {
        /// Stops watching when the test finishes or goes out of scope.
        fn drop(&mut self) {
            TRACKING_THREAD.store(0, Ordering::Relaxed);
        }
    }
}
