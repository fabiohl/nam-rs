// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared allocation audit infrastructure for RT-Safety verification.
//!
//! Provides `CountingAllocator` (the "Memory Watchdog") and `TrackingGuard`
//! used to prove that hot-path DSP code performs zero heap allocations.

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
///
/// Implements `GlobalAlloc` directly — register as `#[global_allocator]`
/// with `static GLOBAL: CountingAllocator = CountingAllocator;`.
pub struct CountingAllocator;

impl CountingAllocator {
    /// Intercepts allocation: increments `ALLOC_COUNT` if on the watched thread.
    ///
    /// # Safety
    ///
    /// The caller must ensure `layout` is valid (non-zero size, non-ZST
    /// with alignment ≤ size). This delegates to the system allocator.
    pub unsafe fn alloc(layout: Layout) -> *mut u8 {
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
        if tid == TRACKING_THREAD.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    /// Delegates deallocation to the system allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must have been previously allocated via `CountingAllocator::alloc`
    /// with the same `layout`.
    pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { Self::alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { Self::dealloc(ptr, layout) }
    }
}

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
