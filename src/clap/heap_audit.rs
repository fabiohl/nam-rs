// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time heap allocation auditing (RT-Safety).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};

/// Flag controlling whether heap allocation tracking is enabled.
pub static AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
/// Global counter of allocations performed on the watched thread.
pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// ID of the thread (tid) we are watching.
pub static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0);
/// ID of the thread authorized to perform the audit (used to isolate parallel tests).
pub static AUDIT_THREAD: AtomicI32 = AtomicI32::new(0);

/// The "Memory Watchdog": intercepts all memory requests from the program.
pub struct CountingAllocator;

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

#[cfg(all(feature = "clap-plugin", feature = "heap-audit", not(test)))]
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// The "Switch": turns on the watchdog when created and turns it off when destroyed.
pub struct TrackingGuard;

impl TrackingGuard {
    /// Starts watching the current thread.
    pub fn new() -> Self {
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
        TRACKING_THREAD.store(tid, Ordering::Relaxed);
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        Self
    }
}

impl Default for TrackingGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TrackingGuard {
    fn drop(&mut self) {
        TRACKING_THREAD.store(0, Ordering::Relaxed);
    }
}
