// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! PipeWire output DSP pipeline and host configuration (standalone).

use super::bridge::DspBridgeReader;

#[cfg(feature = "standalone")]
use crate::common::diagnostics::SystemSnapshot;
#[cfg(feature = "standalone")]
#[cfg(feature = "standalone")]
use pipewire as pw;

/// Holds essential PipeWire instances (`StreamBox` and `Listener`).
/// Fields exist solely to keep streams and listeners alive via RAII;
/// they are never read directly — only held for their drop semantics.
#[cfg(feature = "standalone")]
#[allow(dead_code)]
pub(crate) struct AppState<S1, L1, S2, L2> {
    pub capture_stream: S1,
    pub capture_listener: L1,
    pub playback_stream: S2,
    pub playback_listener: L2,
}

/// Configuration for PipeWire host initialization.
#[cfg(feature = "standalone")]
pub struct PipewireHostConfig {
    /// Requested audio buffer size.
    pub buffer_size: u32,
    /// System snapshot for diagnostics.
    pub sys: SystemSnapshot,
    /// Raw IR samples for adaptive partition rebuild (None if no IR loaded).
    pub ir_raw_samples: Option<Vec<f32>>,
}

/// Playback DSP Pipeline (Bridge → Hardware).
#[cfg(feature = "standalone")]
#[inline(always)]
pub(crate) fn playback_dsp_cycle(
    stream: &pw::stream::Stream,
    bridge: DspBridgeReader,
    last_bridge_gen: &mut u64,
) {
    bridge.read_block(last_bridge_gen, |buf_l, buf_r| {
        let n_samples = buf_l.len();
        if n_samples == 0 {
            return;
        }

        // Requests an empty space from the sound system (PipeWire) to place the audio.
        let mut buf = match stream.dequeue_buffer() {
            Some(b) => b,
            None => return,
        };

        let datas = buf.datas_mut();
        if datas.len() < 2 {
            return;
        }

        // Splits the Left and Right channels for final delivery.
        let (datas_left, datas_right) = datas.split_at_mut(1);
        let data_l = &mut datas_left[0];
        let data_r = &mut datas_right[0];

        let max_l = data_l.as_raw().maxsize as usize / std::mem::size_of::<f32>();
        let max_r = data_r.as_raw().maxsize as usize / std::mem::size_of::<f32>();
        let n_out = n_samples.min(max_l).min(max_r);
        if n_out == 0 {
            return;
        }

        // Copies the processed sound directly to your sound card outputs.
        if let Some(raw_l) = data_l.data() {
            let out_l =
                unsafe { std::slice::from_raw_parts_mut(raw_l.as_mut_ptr().cast::<f32>(), n_out) };
            out_l.copy_from_slice(&buf_l[..n_out]);
        }
        if let Some(raw_r) = data_r.data() {
            let out_r =
                unsafe { std::slice::from_raw_parts_mut(raw_r.as_mut_ptr().cast::<f32>(), n_out) };
            out_r.copy_from_slice(&buf_r[..n_out]);
        }

        // Informs the hardware exactly how much sound was delivered this time.
        {
            let chunk = data_l.chunk_mut();
            *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
        }
        {
            let chunk = data_r.chunk_mut();
            *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
            *chunk.offset_mut() = 0;
            *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
        }
    });
}

/// Builds an F32P stereo audio format SPA Pod for PipeWire negotiation.
///
/// # Safety
/// The returned binary pod points directly to the provided `format_buf`.
#[cfg(feature = "standalone")]
pub(crate) unsafe fn build_spa_format_pod<'a>(
    audio_info: &pw::spa::param::audio::AudioInfoRaw,
    format_buf: &'a mut [u8; 1024],
) -> anyhow::Result<&'a pw::spa::pod::Pod> {
    unsafe {
        let mut builder: pw::spa::sys::spa_pod_builder = std::mem::zeroed();
        // Prepares a "builder" to create the audio format contract.
        pw::spa::sys::spa_pod_builder_init(
            &mut builder,
            format_buf.as_mut_ptr().cast(),
            format_buf.len() as u32,
        );

        // Builds the binary document (SPA Pod) describing the audio (e.g.: 48kHz, Stereo).
        // This document is what PipeWire uses to understand how to send sound to us.
        let pod_ptr = pw::spa::sys::spa_format_audio_raw_build(
            &mut builder,
            pw::spa::param::ParamType::EnumFormat.as_raw(),
            &audio_info.as_raw(),
        );

        if pod_ptr.is_null() {
            // If failure, the system won't know how to negotiate sound with your card.
            return Err(anyhow::anyhow!(
                "Failed to build the audio negotiation document (SPA Pod)"
            ));
        }

        // Returns the contract ready to be signed and used by the system.
        Ok(&*(pod_ptr as *const pw::spa::pod::Pod))
    }
}
