// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! PipeWire playback stream configuration (`Stream/Output/Audio`) — reads from
//! `DspBridge` and delivers processed audio to the hardware.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::pipeline::{BridgeRef, DspBridgeReader, build_spa_format_pod, playback_dsp_cycle};
use crate::standalone::colors::Colorize;

use pipewire as pw;
use pw::properties::properties;
use std::sync::Arc;

/// Configures the playback stream and its RT listener.
///
/// The `process()` closure reads from `DspBridge` (filled by the capture stream)
/// e entrega ao hardware via `playback_dsp_cycle`.
pub fn setup_playback_stream<'c>(
    core: &'c pw::core::Core,
    bridge_ptr: BridgeRef,
    buffer_size: u32,
    latency_str: &str,
    rt_status: Arc<RtStatusFlags>,
) -> anyhow::Result<(pw::stream::StreamBox<'c>, pw::stream::StreamListener<()>)> {
    let bridge_ptr_playback = unsafe { DspBridgeReader::new(bridge_ptr.as_ptr()) };
    let rt_status_playback = rt_status.clone();

    let mut playback_props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Playback",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::MEDIA_CLASS => "Stream/Output/Audio",
        *pw::keys::NODE_NAME => "NAM-rs-playback",
        *pw::keys::NODE_DESCRIPTION => "NAM-rs Processed Output",
        "audio.position" => "FL,FR",
        "node.group" => "nam-rs-dsp",
        "node.link-group" => "nam-rs-link-group",
    };

    if buffer_size > 0 {
        playback_props.insert("node.latency", latency_str);
    }

    let playback_stream = pw::stream::StreamBox::new(core, "NAM-rs-Output", playback_props)?;

    let mut last_bridge_gen: u64 = 0;

    let playback_listener = playback_stream
        .add_local_listener::<()>()
        .process(move |stream: &pw::stream::Stream, _info| {
            playback_dsp_cycle(
                stream,
                bridge_ptr_playback,
                &mut last_bridge_gen,
                &rt_status_playback,
            );
        })
        .register()?;

    let mut playback_audio_info = pw::spa::param::audio::AudioInfoRaw::new();
    playback_audio_info.set_format(pw::spa::param::audio::AudioFormat::F32P);
    playback_audio_info.set_channels(2);

    let mut playback_format_buf = [0u8; 1024];
    let playback_format_pod =
        unsafe { build_spa_format_pod(&playback_audio_info, &mut playback_format_buf)? };

    playback_stream.connect(
        pw::spa::utils::Direction::Output,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut [playback_format_pod],
    )?;

    log::info!(
        "{} Playback stream connected to PipeWire (Stream/Output/Audio, F32P Planar Stereo).",
        "🔊".bright_blue()
    );

    Ok((playback_stream, playback_listener))
}
