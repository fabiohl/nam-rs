// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared CountingAllocator infrastructure for heap-audit integration tests.
//!
//! Provides a local `CountingAllocator` (used when `clap-plugin` is disabled),
//! `TrackingGuard` (RAII gate that starts/stops allocation counting), and
//! `get_alloc_count()`. When `clap-plugin` is enabled, the guard and counter
//! delegate to [`nam_rs::common::alloc_audit`].
//!
//! Each test binary registers its own `#[global_allocator]` referencing
//! [`CountingAllocator`]; this module only provides the shared type.

use std::sync::atomic::Ordering;

#[cfg(not(feature = "clap-plugin"))]
use std::alloc::{GlobalAlloc, Layout, System};
#[cfg(not(feature = "clap-plugin"))]
use std::sync::atomic::{AtomicI32, AtomicUsize};

#[cfg(not(feature = "clap-plugin"))]
pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(feature = "clap-plugin"))]
pub static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0);

#[cfg(not(feature = "clap-plugin"))]
pub struct CountingAllocator;

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

pub struct TrackingGuard {
    #[cfg(feature = "clap-plugin")]
    _inner: nam_rs::common::alloc_audit::TrackingGuard,
}

impl TrackingGuard {
    pub fn new() -> Self {
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

pub fn get_alloc_count() -> usize {
    #[cfg(feature = "clap-plugin")]
    {
        nam_rs::common::alloc_audit::ALLOC_COUNT.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "clap-plugin"))]
    {
        ALLOC_COUNT.load(Ordering::Relaxed)
    }
}
