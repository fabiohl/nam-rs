// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
#![warn(clippy::undocumented_unsafe_blocks)]

//! # Mirrored Buffer (MirroredBuffer) via Mirrored Memory Mapping
//!
//! The `MirroredBuffer` is an advanced memory management technique that solves
//! the classic "contiguity break" problem in circular/delay-line buffers.
//!
//! ## The Problem: The Buffer Boundary
//! In traditional circular buffers, upon reaching the end of allocated space, the pointer
//! wraps back to the start. If a DSP algorithm (such as a Convolution or FFT) needs to read
//! a window of 1024 samples, but the pointer is only 500 samples from the end, the developer
//! would have to handle the read in two parts or perform an expensive copy (`copy_within`).
//! This introduces complex logic (`if/else`) and hurts performance in the "hot-path".
//!
//! ## The Solution: The Mirroring "Trick"
//! This structure leverages the processor's Memory Management Unit (MMU) features
//! to map the **same physical memory block** twice consecutively in the virtual
//! address space:
//!
//! ```text
//! Virtual Space: [ Physical Block (Page 0..N) ] [ Physical Block (Page 0..N) ]
//!                 ^                             ^
//!                 |                             |
//!          Buffer Start                    Mirror of the Start
//! ```
//!
//! Thanks to this mapping, any access that "goes past" the end of the first block will
//! automatically fall into the start of the second block — which is, physically, the buffer
//! start itself.
//!
//! ## Benefits for Real-Time Audio
//! 1. **Linear Access**: Algorithms can read contiguous windows of any size (up to the full buffer size) without worrying about "wrap".
//! 2. **Zero-Copy**: Eliminates the need to copy data to temporary buffers to linearize them.
//! 3. **SIMD Performance**: Enables vector instructions (AVX/SSE) to process data across the buffer boundary without logic interruptions.
//! 4. **Branch-Free**: Removes modulo (`%`) operations and `if` conditions, optimizing processor branch prediction.
//!
//! ## Huge Page Support
//! Attempts 2 MB huge pages (MAP_HUGETLB / MFD_HUGETLB) for the mirror buffer to reduce
//! TLB pressure in the DSP hot-path. Falls back to regular pages + `madvise(MADV_HUGEPAGE)`.
//! Status is tracked via `MIRROR_BUF_HUGEPAGE_ACTIVE` global to avoid inflating the
//! struct layout (which would affect hot-path cache performance).
use libc::{c_void, munmap};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::AtomicBool;

mod alloc;

thread_local! {
    pub(crate) static SIMULATE_FAIL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Global flag: set to `true` when any MirroredBuffer successfully uses huge pages.
/// The main thread reads this to set `RT_STATUS_HUGEPAGE_OK`.
static MIRROR_BUF_HUGEPAGE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Synchronizes the mirror buffer huge-page status to the RT status flag.
/// Call once during main-thread initialization.
pub fn sync_huge_page_flag(rt_status: &crate::common::spsc::RtStatusFlags) {
    if MIRROR_BUF_HUGEPAGE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
        rt_status.set_flag(crate::common::spsc::RT_STATUS_HUGEPAGE_OK);
    }
}

/// Tracks whether huge pages were successfully activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorHugePageStatus {
    /// Explicit 2 MB huge pages via MAP_HUGETLB | MFD_HUGETLB.
    Explicit2MB,
    /// Transparent huge pages via madvise(MADV_HUGEPAGE) hint.
    Transparent,
    /// Standard 4 KB pages (fallback).
    Standard,
}

/// Returns whether any mirror buffer has huge pages active.
pub fn is_huge_page_active() -> bool {
    MIRROR_BUF_HUGEPAGE_ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
}

/// A Mirrored Buffer that uses mirrored memory mapping.
///
/// This structure maps the same physical content twice consecutively in the virtual
/// address space. This allows accesses that would "cross" the end of the buffer
/// to be performed linearly and contiguously, eliminating the need for "rewind"
/// or "copy_within" operations in the DSP hot-path.
///
/// Struct layout (16 bytes): optimized for cache — the hot-path Deref
/// accesses only the first two fields.
pub struct MirroredBuffer<T> {
    ptr: *mut T,
    size_elements: usize,
    _marker: PhantomData<T>,
}

/// Sets whether the next `MirroredBuffer` creation calls should simulate
/// virtual memory allocation failure.
pub fn set_simulate_fail(fail: bool) {
    SIMULATE_FAIL.with(|f| f.set(fail));
}

impl<T> MirroredBuffer<T> {
    /// Returns the physical buffer size (before mirroring) in elements.
    pub fn size(&self) -> usize {
        self.size_elements
    }
}

impl<T> std::fmt::Debug for MirroredBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MirroredBuffer")
            .field("ptr", &self.ptr)
            .field("size_elements", &self.size_elements)
            .field("capacity_virtual", &(self.size_elements * 2))
            .finish()
    }
}

impl<T> Deref for MirroredBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // Returns a slice that covers both halves (2x size)
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_elements * 2) }
    }
}

impl<T> DerefMut for MirroredBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_elements * 2) }
    }
}

impl<T> Drop for MirroredBuffer<T> {
    fn drop(&mut self) {
        let element_size = std::mem::size_of::<T>();
        let size_bytes = self.size_elements * element_size;
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe {
            munmap(self.ptr as *mut c_void, size_bytes * 2);
        }
    }
}

impl<T: Clone> Clone for MirroredBuffer<T> {
    #[cold]
    fn clone(&self) -> Self {
        match Self::new(self.size_elements) {
            Ok(mut new_buf) => {
                new_buf[..self.size_elements].clone_from_slice(&self[..self.size_elements]);
                new_buf
            }
            Err(err) => {
                std::panic::panic_any(format!("Failed to clone MirroredBuffer: {:?}", err));
            }
        }
    }
}

// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
unsafe impl<T: Send> Send for MirroredBuffer<T> {}
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
unsafe impl<T: Sync> Sync for MirroredBuffer<T> {}

#[cfg(all(test, target_os = "linux"))]
#[path = "mirror_buf_test.rs"]
mod mirror_buf_test;
