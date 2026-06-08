// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
pub(crate) mod infra {
    use std::alloc::{GlobalAlloc, Layout};

    pub use crate::common::alloc_audit::{ALLOC_COUNT, TrackingGuard};

    pub struct CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            unsafe { crate::common::alloc_audit::CountingAllocator::alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { crate::common::alloc_audit::CountingAllocator::dealloc(ptr, layout) }
        }
    }
}
