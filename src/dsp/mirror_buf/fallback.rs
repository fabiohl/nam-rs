// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::io;

/// Stub de criação do backing store para plataformas não-Linux.
pub(crate) unsafe fn create_backing_fd() -> io::Result<libc::c_int> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "MirroredBuffer is only supported on Linux",
    ))
}
