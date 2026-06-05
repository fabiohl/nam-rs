// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::SIMULATE_FAIL;

/// Tracks whether the backing store was created with huge-page support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackingFdKind {
    /// Standard anonymous memfd (4 KB pages).
    Memfd,
    /// HugeTLB-backed memfd (2 MB pages, requires Linux 5.14+).
    HugetlbMemfd,
}

/// Creates a regular memfd backing store (4 KB pages, no huge-page attempt).
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
pub(crate) unsafe fn create_backing_fd() -> std::io::Result<libc::c_int> {
    if SIMULATE_FAIL.with(|f| f.get()) {
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe {
            *libc::__errno_location() = libc::ENOMEM;
        }
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
    let fd = unsafe { libc::memfd_create(c"mirror_buf".as_ptr(), libc::MFD_CLOEXEC) };
    if fd == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(fd)
    }
}

/// Creates a huge-page-backed memfd for the mirror buffer.
///
/// Tries `MFD_HUGETLB` on Linux 5.14+ for explicit 2 MB huge-page backing first,
/// falls back to regular `memfd_create` if huge pages are unavailable.
///
/// Returns the FD together with its kind for telemetry.
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
pub(crate) unsafe fn create_huge_backing_fd() -> std::io::Result<(libc::c_int, BackingFdKind)> {
    if SIMULATE_FAIL.with(|f| f.get()) {
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe {
            *libc::__errno_location() = libc::ENOMEM;
        }
        return Err(std::io::Error::last_os_error());
    }

    // Strategy 1: try MFD_HUGETLB for explicit 2 MB huge pages (Linux 5.14+).
    const MFD_HUGETLB: libc::c_uint = 0x0004;
    // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
    let fd = unsafe {
        libc::memfd_create(c"mirror_buf_huge".as_ptr(), libc::MFD_CLOEXEC | MFD_HUGETLB)
    };
    if fd != -1 {
        return Ok((fd, BackingFdKind::HugetlbMemfd));
    }

    // Strategy 2: regular memfd (THP promotion via madvise downstream).
    // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
    let fd = unsafe { libc::memfd_create(c"mirror_buf".as_ptr(), libc::MFD_CLOEXEC) };
    if fd == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok((fd, BackingFdKind::Memfd))
    }
}
