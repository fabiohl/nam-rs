// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed errors for `.namb` binary file parsing.

/// Typed error for `.namb` file parsing.
///
/// Each variant corresponds to a specific integrity or format failure
/// of the binary file, enabling precise diagnosis via
/// `downcast_ref` in the `loader` module.
#[derive(Debug, thiserror::Error)]
pub enum NambError {
    /// Truncated file: insufficient bytes for the minimum header.
    #[error("file truncated: got {got} bytes, need at least {need}")]
    Truncated {
        /// Bytes available in the file.
        got: usize,
        /// Minimum bytes needed.
        need: usize,
    },

    /// Invalid magic number (not 0x4E414D42).
    #[error("invalid magic number: 0x{0:08X} (expected 0x4E414D42)")]
    InvalidMagic(u32),

    /// Unsupported `.namb` format version.
    #[error("unsupported .namb version: {0}")]
    InvalidVersion(u16),

    /// Weight section offset beyond file size.
    #[error("weights offset {offset} out of file bounds (file size: {file_len})")]
    WeightsOffsetOutOfBounds {
        /// Offset declared in the header.
        offset: usize,
        /// Total file size in bytes.
        file_len: usize,
    },

    /// Weight section offset smaller than the header size.
    #[error("invalid weights offset {offset} (smaller than header size {header_size})")]
    InvalidWeightsOffset {
        /// Offset declared in the header.
        offset: usize,
        /// Expected header size.
        header_size: usize,
    },

    /// CRC32 checksum of the weight section does not match.
    #[error("CRC32 mismatch: got 0x{got:08X}, expected 0x{expected:08X}")]
    CrcMismatch {
        /// CRC calculated from the data.
        got: u32,
        /// CRC declared in the header.
        expected: u32,
    },

    /// CRC32 missing in NAMB v2+ file (FLAG_HAS_CRC32 flag not set).
    #[error("CRC32 flag missing in NAMB v{version} file (FLAG_HAS_CRC32 not set)")]
    CrcMissing {
        /// NAMB file version.
        version: u16,
    },
}
