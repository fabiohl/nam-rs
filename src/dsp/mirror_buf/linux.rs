// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::SIMULATE_FAIL;

/// Creates the Linux-specific backing store (memfd).
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
pub(crate) unsafe fn create_backing_fd() -> std::io::Result<libc::c_int> {
    let fd = if SIMULATE_FAIL.with(|f| f.get()) {
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe {
            *libc::__errno_location() = libc::ENOMEM;
        }
        -1
    } else {
        // SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
        unsafe { libc::memfd_create(c"mirror_buf".as_ptr(), libc::MFD_CLOEXEC) }
    };

    if fd == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(fd)
    }
}
