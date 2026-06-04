// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::io;

/// Backing store creation stub for non-Linux platforms.
// SAFETY: Low-level virtual memory manipulation (mmap/ftruncate) with checked parameters.
pub(crate) unsafe fn create_backing_fd() -> io::Result<libc::c_int> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "MirroredBuffer is only supported on Linux",
    ))
}
