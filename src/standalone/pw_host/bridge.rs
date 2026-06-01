// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Alocação do buffer compartilhado `DspBridge` para comunicação lock-free
//! entre as streams de captura e playback.

use crate::dsp::pipeline::{BridgeBuffer, BridgeRef, DspBridge, MAX_BRIDGE_BUF};

/// Aloca o `DspBridge` com double-buffering via `Box::leak` (vida `'static`).
///
/// O buffer é alinhado a 128 bytes (`repr(align(128))`) para evitar false-sharing
/// entre os dois callbacks RT.
///
/// Aplica `madvise(MADV_DONTFORK | MADV_DONTDUMP)` para evitar overhead de
/// Copy-on-Write em forks e excluir os buffers de core dumps.
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

    unsafe {
        libc::madvise(
            bridge as *const DspBridge as *mut libc::c_void,
            std::mem::size_of::<DspBridge>(),
            libc::MADV_DONTFORK | libc::MADV_DONTDUMP,
        );
    }

    bridge_ptr
}
