// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time heap allocation auditing (RT-Safety).

use std::alloc::{GlobalAlloc, Layout};

pub use crate::common::alloc_audit::{
    ALLOC_COUNT, AUDIT_ENABLED, AUDIT_THREAD, TRACKING_THREAD, TrackingGuard,
};

/// Local `GlobalAlloc` wrapper that delegates to the shared `CountingAllocator`.
#[allow(dead_code)]
struct ClapAlloc;

unsafe impl GlobalAlloc for ClapAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { crate::common::alloc_audit::CountingAllocator::alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { crate::common::alloc_audit::CountingAllocator::dealloc(ptr, layout) }
    }
}

#[cfg(all(feature = "clap-plugin", feature = "heap-audit", not(test)))]
#[global_allocator]
static GLOBAL: ClapAlloc = ClapAlloc;
