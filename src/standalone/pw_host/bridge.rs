// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Allocation of the `DspBridge` shared buffer for lock-free communication
//! between capture and playback streams.

use crate::dsp::pipeline::{BridgeBuffer, BridgeRef, DspBridge, MAX_BRIDGE_BUF};

/// Allocates `DspBridge` with double-buffering via `Box::leak` (`'static` lifetime).
///
/// **Standalone-only**: This function lives under `#[cfg(feature = "standalone")]`
/// because `Box::leak` is only acceptable when the DspBridge lifetime equals the
/// process lifetime. In the CLAP plugin, no DspBridge is created — audio flows
/// synchronously within a single `process()` callback.
///
/// The buffer is aligned to 128 bytes (`repr(align(128))`) to avoid false-sharing
/// between the two RT callbacks.
///
/// Applies `madvise(MADV_DONTFORK | MADV_DONTDUMP)` to avoid Copy-on-Write
/// overhead on forks and to exclude the buffers from core dumps.
pub fn allocate_dsp_bridge() -> BridgeRef {
    let bridge: &'static DspBridge = Box::leak(Box::new(DspBridge {
        buffers: [
            BridgeBuffer {
                buf_l: [0.0f32; MAX_BRIDGE_BUF],
                buf_r: [0.0f32; MAX_BRIDGE_BUF],
                n_samples: 0,
            },
            BridgeBuffer {
                buf_l: [0.0f32; MAX_BRIDGE_BUF],
                buf_r: [0.0f32; MAX_BRIDGE_BUF],
                n_samples: 0,
            },
        ],
        active_read_idx: std::sync::atomic::AtomicUsize::new(0),
        generation: std::sync::atomic::AtomicU64::new(0),
        consumed_gen: std::sync::atomic::AtomicU64::new(0),
        dropped_frames: std::sync::atomic::AtomicU32::new(0),
    }));
    let bridge_ptr = unsafe { BridgeRef::new(bridge as *const DspBridge as *mut DspBridge) };

    // SAFETY: `bridge` points to a valid `DspBridge` allocation produced
    // by `Box::leak` immediately above — the pointer and size are correct.
    // `MADV_DONTFORK | MADV_DONTDUMP` are advisory hints; they cannot
    // corrupt memory, crash, or invalidate the allocation. The kernel may
    // ignore the hint on older versions (pre-4.4 for DONTFORK, pre-3.4 for
    // DONTDUMP), which is benign.
    let ret = unsafe {
        libc::madvise(
            bridge as *const DspBridge as *mut libc::c_void,
            std::mem::size_of::<DspBridge>(),
            libc::MADV_DONTFORK | libc::MADV_DONTDUMP,
        )
    };
    if ret != 0 {
        log::warn!(
            "madvise(MADV_DONTFORK|MADV_DONTDUMP) returned {} (errno: {}). \
             Buffers may be included in forks/core-dumps.",
            ret,
            std::io::Error::last_os_error()
        );
    }

    bridge_ptr
}
