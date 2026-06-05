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
use libc::{
    MAP_ANONYMOUS, MAP_FAILED, MAP_FIXED, MAP_HUGETLB, MAP_HUGE_2MB, MAP_PRIVATE, MAP_SHARED,
    MADV_HUGEPAGE, PROT_NONE, PROT_READ, PROT_WRITE, c_void, ftruncate, mmap, munmap, sysconf,
};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::sync::atomic::AtomicBool;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(not(target_os = "linux"))]
mod fallback;

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
    /// Creates a new mirrored buffer with huge-page preference.
    ///
    /// The `requested_size` (in elements) will be rounded up to the next
    /// multiple of the system page size (2 MB for huge pages, 4 KB for standard).
    #[cold]
    pub fn new(requested_size: usize) -> std::io::Result<Self> {
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let page_size = unsafe { sysconf(libc::_SC_PAGESIZE) } as usize;
        let element_size = std::mem::size_of::<T>();

        // Ensure the element size is not zero (e.g., ZST)
        if element_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "MirroredBuffer does not support Zero Sized Types",
            ));
        }

        if requested_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "requested_size must be greater than zero",
            ));
        }

        let requested_bytes = match requested_size.checked_mul(element_size) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "requested_size * element_size overflowed",
                ));
            }
        };

        // Page size for huge-page alignment (2 MB on x86-64).
        const HUGE_PAGE_2M: usize = 2 * 1024 * 1024;

        // Try huge-page path if the total size (2x) is at least 2 MB.
        let total_chunk = match requested_bytes.checked_mul(2) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "requested_bytes * 2 overflowed",
                ));
            }
        };

        if total_chunk >= HUGE_PAGE_2M {
            if let Ok(buf) = Self::try_new_huge(requested_bytes, HUGE_PAGE_2M) {
                MIRROR_BUF_HUGEPAGE_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);
                return Ok(buf);
            }
        }

        // Standard path (4 KB pages)
        let page_mask = page_size - 1;
        let size_bytes = match requested_bytes.checked_add(page_mask) {
            Some(val) => val & !page_mask,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "size_bytes calculation overflowed",
                ));
            }
        };
        let size_elements = size_bytes / element_size;

        // 1. Create backing store (memfd on Linux, stub fallback on other platforms)
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let fd = unsafe {
            #[cfg(target_os = "linux")]
            {
                linux::create_backing_fd()?
            }
            #[cfg(not(target_os = "linux"))]
            {
                fallback::create_backing_fd()?
            }
        };

        // 2. Set file size
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        if unsafe { ftruncate(fd, size_bytes as libc::off_t) } == -1 {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // 3. Reserve contiguous virtual space (2x size)
        let total_size = size_bytes * 2;

        // Ensure required invariant before mmap
        assert!(
            requested_size > 0,
            "requested_size must be greater than zero"
        );

        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let base_ptr = unsafe {
            if SIMULATE_FAIL.with(|f| f.get()) {
                *libc::__errno_location() = libc::ENOMEM;
                MAP_FAILED
            } else {
                mmap(
                    ptr::null_mut(),
                    total_size,
                    PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            }
        };
        if base_ptr == MAP_FAILED {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // 4. Map the first half
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let ptr1 = unsafe {
            mmap(
                base_ptr,
                size_bytes,
                PROT_READ | PROT_WRITE,
                MAP_FIXED | MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr1 != base_ptr {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe {
                munmap(base_ptr, total_size);
                libc::close(fd);
            }
            return Err(err);
        }

        // 5. Map the second half (mirror)
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let ptr2 = unsafe {
            mmap(
                (base_ptr as *mut u8).add(size_bytes) as *mut c_void,
                size_bytes,
                PROT_READ | PROT_WRITE,
                MAP_FIXED | MAP_SHARED,
                fd,
                0,
            )
        };
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        if ptr2 != unsafe { (base_ptr as *mut u8).add(size_bytes) as *mut c_void } {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe {
                munmap(base_ptr, total_size);
                libc::close(fd);
            }
            return Err(err);
        }

        // Hint THP promotion for the data regions.
        // SAFETY: base_ptr and size_bytes are valid mapped regions.
        unsafe {
            libc::madvise(base_ptr, size_bytes, MADV_HUGEPAGE);
        }
        MIRROR_BUF_HUGEPAGE_ACTIVE.store(true, std::sync::atomic::Ordering::Relaxed);

        // The FD is no longer needed after mmap (it holds a reference to the file)
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe { libc::close(fd) };

        Ok(Self {
            ptr: base_ptr as *mut T,
            size_elements,
            _marker: PhantomData,
        })
    }

    /// Attempts creation with explicit 2 MB huge pages.
    fn try_new_huge(requested_bytes: usize, huge_page_size: usize) -> std::io::Result<Self> {
        let element_size = std::mem::size_of::<T>();

        let huge_mask = huge_page_size - 1;
        let size_bytes = (requested_bytes + huge_mask) & !huge_mask;
        let size_elements = size_bytes / element_size;

        // 1. Create HugeTLB-backed memfd (falls back to regular memfd)
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let (fd, _fd_kind) = unsafe { linux::create_huge_backing_fd()? };

        // 2. Set file size (must be huge-page-aligned for HugeTLB memfd).
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        if unsafe { ftruncate(fd, size_bytes as libc::off_t) } == -1 {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe { libc::close(fd) };
            return Err(err);
        }

        let total_size = size_bytes * 2;

        // 3. Reserve 2x virtual space with MAP_HUGETLB
        let mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB;
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let base_ptr = unsafe {
            mmap(
                ptr::null_mut(),
                total_size,
                PROT_NONE,
                mmap_flags,
                -1,
                0,
            )
        };
        if base_ptr == MAP_FAILED {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // 4. Map the first half
        let map_flags = MAP_FIXED | MAP_SHARED;
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let ptr1 = unsafe {
            mmap(base_ptr, size_bytes, PROT_READ | PROT_WRITE, map_flags, fd, 0)
        };
        if ptr1 != base_ptr {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe {
                munmap(base_ptr, total_size);
                libc::close(fd);
            }
            return Err(err);
        }

        // 5. Map the second half (mirror)
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let ptr2 = unsafe {
            mmap(
                (base_ptr as *mut u8).add(size_bytes) as *mut c_void,
                size_bytes,
                PROT_READ | PROT_WRITE,
                map_flags,
                fd,
                0,
            )
        };
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        if ptr2 != unsafe { (base_ptr as *mut u8).add(size_bytes) as *mut c_void } {
            let err = std::io::Error::last_os_error();
            // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
            unsafe {
                munmap(base_ptr, total_size);
                libc::close(fd);
            }
            return Err(err);
        }

        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe { libc::close(fd) };

        Ok(Self {
            ptr: base_ptr as *mut T,
            size_elements,
            _marker: PhantomData,
        })
    }

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
