// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
#![warn(clippy::undocumented_unsafe_blocks)]

use super::{MIRROR_BUF_HUGEPAGE_ACTIVE, MirroredBuffer, SIMULATE_FAIL};
use crate::math::common::huge_alloc::HUGE_PAGE_2M;
use libc::{
    MADV_HUGEPAGE, MAP_ANONYMOUS, MAP_FAILED, MAP_FIXED, MAP_HUGE_2MB, MAP_HUGETLB, MAP_PRIVATE,
    MAP_SHARED, PROT_NONE, PROT_READ, PROT_WRITE, c_void, ftruncate, mmap, munmap, sysconf,
};
use std::marker::PhantomData;
use std::ptr;

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
            let huge_res = Self::try_new_huge(requested_bytes, HUGE_PAGE_2M);
            if let Ok(buf) = huge_res {
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
                super::linux::create_backing_fd()?
            }
            #[cfg(not(target_os = "linux"))]
            {
                super::fallback::create_backing_fd()?
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
    #[cold]
    fn try_new_huge(requested_bytes: usize, huge_page_size: usize) -> std::io::Result<Self> {
        let element_size = std::mem::size_of::<T>();

        let huge_mask = huge_page_size - 1;
        let size_bytes = (requested_bytes + huge_mask) & !huge_mask;
        let size_elements = size_bytes / element_size;

        // 1. Create HugeTLB-backed memfd (falls back to regular memfd)
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        let (fd, _fd_kind) = unsafe { super::linux::create_huge_backing_fd()? };

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
        let base_ptr = unsafe { mmap(ptr::null_mut(), total_size, PROT_NONE, mmap_flags, -1, 0) };
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
            mmap(
                base_ptr,
                size_bytes,
                PROT_READ | PROT_WRITE,
                map_flags,
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
}
