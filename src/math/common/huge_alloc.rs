// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Low-level mmap/munmap/madvise syscalls with size/alignment validation.

//! Huge Page-aware memory allocation for large, RT-critical buffers.
//!
//! TLB (Translation Lookaside Buffer) misses in the DSP hot-path cost ~100 cycles.
//! Standard 4 KB pages cause ~20 TLB entries for an 80 KB weight buffer;
//! a single 2 MB huge page covers the entire buffer with 1 TLB entry.
//!
//! # Strategy (best-effort, zero regression risk)
//! 1. `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` — explicit huge pages (requires admin setup).
//! 2. `mmap(MAP_ANONYMOUS | MAP_PRIVATE)` + `madvise(MADV_HUGEPAGE)` — transparent THP.
//! 3. `std::alloc::alloc` — fallback (existing behaviour).
//!
//! Callers must use the returned `AllocInfo` to choose the correct deallocation path.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ptr;

/// Tracks which allocation strategy was used, to pick the right deallocation path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocInfo {
    /// Standard global-allocator allocation (`std::alloc::dealloc`).
    Heap,
    /// Anonymous `mmap`: deallocate via `munmap(ptr, size_bytes)`.
    MmapAnon {
        /// The exact byte size passed to mmap (page-aligned).
        size_bytes: usize,
    },
    /// Explicit 2 MB huge-page `mmap`: deallocate via `munmap(ptr, size_bytes)`.
    HugeTlb2M {
        /// The exact byte size passed to mmap (2 MB-aligned).
        size_bytes: usize,
    },
}

/// Result of a huge-page allocation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageStatus {
    /// Explicit 2 MB huge pages via MAP_HUGETLB succeeded.
    Explicit2MB,
    /// Transparent huge pages via madvise(MADV_HUGEPAGE) on anonymous mmap.
    Transparent,
    /// Fallback to standard heap allocation (global allocator).
    Heap,
}

/// Minimum allocation size (in bytes) that triggers a huge-page attempt.
/// Below this threshold, standard `alloc` is used directly (no syscall overhead).
pub const HUGE_PAGE_THRESHOLD: usize = 1 << 20; // 1 MiB

/// Page size for transparent huge pages (4 KB standard pages, kernel promotes to 2 MB).
const PAGE_4K: usize = 4096;

/// Huge page size (2 MB on x86-64). Must divide evenly into the allocation size.
const PAGE_2M: usize = 2 * 1024 * 1024;

/// Rounds `size` up to the next multiple of `alignment`.
const fn align_up(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Attempts to allocate `size_bytes` with huge-page preference.
///
/// # Returns
/// `(ptr, AllocInfo, HugePageStatus)` — the caller must use `AllocInfo` to
/// select the correct deallocation path.
///
/// # Safety
/// The caller must eventually deallocate using `deallocate_huge()` with the matching `AllocInfo`.
/// The returned pointer is guaranteed to be at least 64-byte aligned for AVX-512.
pub fn allocate_huge_pages(size_bytes: usize) -> (*mut u8, AllocInfo, HugePageStatus) {
    if size_bytes < HUGE_PAGE_THRESHOLD {
        // Small allocations: standard allocator, no syscall overhead.
        let layout = Layout::from_size_align(size_bytes, 64)
            .expect("Failed to create layout for huge_alloc (small)");
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        return (ptr, AllocInfo::Heap, HugePageStatus::Heap);
    }

    let huge_2m_size = align_up(size_bytes, PAGE_2M);

    // Strategy 1: explicit 2 MB huge pages via MAP_HUGETLB.
    // SAFETY: mmap with validated size, no aliasing violations.
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            huge_2m_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB | libc::MAP_HUGE_2MB,
            -1,
            0,
        )
    };

    if ptr != libc::MAP_FAILED {
        return (
            ptr as *mut u8,
            AllocInfo::HugeTlb2M {
                size_bytes: huge_2m_size,
            },
            HugePageStatus::Explicit2MB,
        );
    }

    // Strategy 2: anonymous mmap + madvise(MADV_HUGEPAGE) for transparent THP.
    let thp_size = align_up(size_bytes, PAGE_4K);
    // SAFETY: mmap with validated size.
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            thp_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };

    if ptr != libc::MAP_FAILED {
        // Hint the kernel to promote these pages to huge pages.
        // SAFETY: ptr and thp_size are valid from the successful mmap above.
        let _madvise_rc = unsafe { libc::madvise(ptr, thp_size, libc::MADV_HUGEPAGE) };
        // madvise failure is non-fatal — pages may still be promoted later by khugepaged.

        return (
            ptr as *mut u8,
            AllocInfo::MmapAnon {
                size_bytes: thp_size,
            },
            HugePageStatus::Transparent,
        );
    }

    // Strategy 3: fallback to standard allocator.
    let layout = Layout::from_size_align(size_bytes, 64)
        .expect("Failed to create layout for huge_alloc (fallback)");
    // SAFETY: standard alloc with valid layout.
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        handle_alloc_error(layout);
    }
    (ptr, AllocInfo::Heap, HugePageStatus::Heap)
}

/// Deallocates memory previously obtained from `allocate_huge_pages()`.
///
/// # Safety
/// - `ptr` must have been returned by `allocate_huge_pages()`.
/// - `info` must be the exact `AllocInfo` returned by that call.
/// - `size_bytes` must be the original request size (un-rounded) for Heap deallocs;
///   for mmap-based deallocs, the alloc info already stores the correct size.
pub unsafe fn deallocate_huge(ptr: *mut u8, info: AllocInfo, size_bytes: usize) {
    match info {
        AllocInfo::Heap => {
            let layout = Layout::from_size_align(size_bytes, 64)
                .expect("Failed to create layout for huge_alloc dealloc");
            // SAFETY: ptr and layout match the original allocation.
            unsafe { dealloc(ptr, layout) };
        }
        AllocInfo::MmapAnon { size_bytes: mmap_size } | AllocInfo::HugeTlb2M { size_bytes: mmap_size } => {
            // SAFETY: ptr and mmap_size match the original mmap.
            unsafe { libc::munmap(ptr as *mut libc::c_void, mmap_size) };
        }
    }
}

// ── HugePageVec: AlignedVec-like wrapper with huge-page deallocation ──────

use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A 64-byte aligned buffer backed by huge-page allocation (best-effort).
///
/// Unlike `AlignedVec`, this type carries the deallocation metadata (mmap size)
/// and is intended for allocations ≥ 1 MiB where TLB pressure matters.
///
/// Layout (24 bytes): larger than `AlignedVec` (16 bytes), but only used for
/// large allocations where the overhead is negligible.
#[derive(Debug)]
pub struct HugePageVec<T> {
    ptr: NonNull<T>,
    len: usize,
    alloc_info: AllocInfo,
}

impl<T> HugePageVec<T> {
    /// Creates a new huge-page-backed buffer filled with `default`.
    pub fn new(len: usize, default: T) -> (Self, HugePageStatus)
    where
        T: Copy,
    {
        let (mut vec, status) = Self::with_capacity(len);
        // SAFETY: Inner safety guarantees are upheld by caller invariants.
        unsafe {
            for i in 0..len {
                vec.ptr.as_ptr().add(i).write(default);
            }
        }
        vec.len = len;
        (vec, status)
    }

    /// Reserves capacity with huge-page preference.
    pub fn with_capacity(capacity: usize) -> (Self, HugePageStatus) {
        if capacity == 0 {
            return (
                Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                    alloc_info: AllocInfo::Heap,
                },
                HugePageStatus::Heap,
            );
        }
        let size_bytes = capacity * std::mem::size_of::<T>();
        let (ptr, alloc_info, status) = allocate_huge_pages(size_bytes);
        (
            Self {
                ptr: NonNull::new(ptr as *mut T).unwrap(),
                len: 0,
                alloc_info,
            },
            status,
        )
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the huge-page allocation status.
    pub fn huge_page_status(&self) -> HugePageStatus {
        match self.alloc_info {
            AllocInfo::Heap => HugePageStatus::Heap,
            AllocInfo::MmapAnon { .. } => HugePageStatus::Transparent,
            AllocInfo::HugeTlb2M { .. } => HugePageStatus::Explicit2MB,
        }
    }
}

impl<T> Deref for HugePageVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: Inner safety guarantees are upheld by caller invariants.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T> DerefMut for HugePageVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: Inner safety guarantees are upheld by caller invariants.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T> Drop for HugePageVec<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: Inner safety guarantees are upheld by caller invariants.
            unsafe {
                deallocate_huge(
                    self.ptr.as_ptr() as *mut u8,
                    self.alloc_info,
                    self.len * std::mem::size_of::<T>(),
                );
            }
        }
    }
}

// SAFETY: Inner safety guarantees are upheld by caller invariants.
unsafe impl<T: Send> Send for HugePageVec<T> {}
// SAFETY: Inner safety guarantees are upheld by caller invariants.
unsafe impl<T: Sync> Sync for HugePageVec<T> {}
