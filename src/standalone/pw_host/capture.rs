// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Configuration of the PipeWire capture stream (`Audio/Sink`) — Virtual Sink that
//! receives audio from apps, applies the DSP chain and writes to `DspBridge`.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode};
use crate::common::spsc::{GcItem, GcOverflowBuffer, ParamPayload, RtStatusFlags};
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::pipeline::{
    BridgeRef, DspBridgeWriter, DspBuffers, DspPipelineContext, MAX_RESAMP_BUF,
    build_spa_format_pod,
};
use crate::dsp::resampler::NamResampler;
use crate::standalone::colors::Colorize;
use crate::standalone::rt_setup;

use super::rt_callback;

use pipewire as pw;
use pw::properties::properties;
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Configures the capture stream (Virtual Sink) and its RT listener.
///
/// The `process()` closure captures all DSP state by `move` and executes
/// the full pipeline: resampler draining, command reception,
/// rate synchronization, and DSP processing via `capture_dsp_pipeline`.
#[allow(clippy::too_many_arguments)]
pub fn setup_capture_stream<'c>(
    core: &'c pw::core::Core,
    bridge_ptr: BridgeRef,
    buffer_size: u32,
    sys: &crate::common::diagnostics::SystemSnapshot,
    target_cpu: usize,
    mut consumer: Consumer<ParamPayload>,
    mut gc_producer: rtrb::Producer<GcItem>,
    gc_overflow: Arc<GcOverflowBuffer>,
    mut resampler_consumer: Consumer<NamResampler>,
    rt_status: Arc<RtStatusFlags>,
) -> anyhow::Result<(pw::stream::StreamBox<'c>, pw::stream::StreamListener<()>)> {
    let mut capture_props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Duplex",
        *pw::keys::MEDIA_ROLE => "DSP",
        *pw::keys::MEDIA_CLASS => "Audio/Sink",
        *pw::keys::NODE_NAME => "NAM-rs-input",
        *pw::keys::NODE_DESCRIPTION => "NAM-rs Input",
        *pw::keys::NODE_VIRTUAL => "true",
        *pw::keys::PRIORITY_SESSION => "2000",
        *pw::keys::PRIORITY_DRIVER => "2000",
        "audio.position" => "FL,FR",
        "node.group" => "nam-rs-dsp",
        "node.link-group" => "nam-rs-link-group",
    };

    let latency_str = format!("{}/48000", buffer_size);
    if buffer_size > 0 {
        capture_props.insert("node.latency", latency_str.as_str());
    }

    let capture_stream = pw::stream::StreamBox::new(core, "NAM-rs", capture_props)?;

    let mut active_model_l: Option<Box<crate::models::DynamicModel>> = None;
    let mut active_model_r: Option<Box<crate::models::DynamicModel>> = None;

    let resampler = NamResampler::new(48_000, 48_000, 2048).unwrap_or_else(|e| {
        NamDiagnostic::new(NamErrorCode::ResamplerBuildFailed, sys)
            .message("Failed to create initial NamResampler (using 48k bypass).")
            .hint("The engine remains in bypass mode. The resampler will be recreated upon receiving the actual rate from PipeWire.")
            .param("initial_rate", 48_000_u32)
            .param("detail", &e)
            .emit_warning();
        NamResampler::new(48_000, 48_000, 2048).expect("bypass cannot fail")
    });
    let mut resampler = Box::new(resampler);

    let mut current_nam_rate: u32 = 48_000;

    let mut resamp_mid_l = [0.0f32; MAX_RESAMP_BUF];
    let mut resamp_out_l = [0.0f32; MAX_RESAMP_BUF];
    let mut resamp_mid_r = [0.0f32; MAX_RESAMP_BUF];
    let mut resamp_out_r = [0.0f32; MAX_RESAMP_BUF];
    let mut model_out_l = [0.0f32; MAX_RESAMP_BUF];
    let mut model_out_r = [0.0f32; MAX_RESAMP_BUF];

    let mut user_input_gain_mult: f32 = 1.0;
    let mut user_output_gain_mult: f32 = 1.0;
    let mut model_input_mult_adj: f32 = 1.0;
    let mut model_output_mult_adj: f32 = 1.0;

    let mut input_gain_mult: f32 = 1.0;
    let mut output_gain_mult: f32 = 1.0;

    let mut gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = false;

    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let open_lin = lut.db_to_linear(gate_params.threshold_open_db);
    let close_lin = lut.db_to_linear(gate_params.threshold_close_db);
    let mut threshold_open_sq: f32 = open_lin * open_lin;
    let mut threshold_close_sq: f32 = close_lin * close_lin;

    let shared_target_rate = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let rate_for_param = shared_target_rate.clone();
    let rate_for_process = shared_target_rate.clone();

    let mut parking_lot: [Option<GcItem>; 16] = Default::default();

    let rt_status_for_process = rt_status.clone();
    let gc_overflow_for_process = gc_overflow.clone();
    let mut frame_count: u32 = 0;

    let mut thread_configured = false;

    let capture_listener = capture_stream
        .add_local_listener::<()>()
        .state_changed(move |_stream, _user_data, old, new| match new {
            pw::stream::StreamState::Error(err) => {
                log::error!("{} Critical PW audio stream failure: {}", "💥".red(), err);
            }
            pw::stream::StreamState::Paused if old == pw::stream::StreamState::Streaming => {
                log::info!("{} Audio disconnected or node switch.", "⏸️".yellow());
            }
            pw::stream::StreamState::Streaming if old == pw::stream::StreamState::Paused => {
                log::info!("{} Audio captured (connection established)", "▶️".green());
            }
            _ => {}
        })
        .param_changed(move |_stream, _user_data, id, param| {
            let Some(param) = param else { return };
            if id != pw::spa::param::ParamType::Format.as_raw() {
                return;
            }

            let (media_type, media_subtype) =
                match pw::spa::param::format_utils::parse_format(param) {
                    Ok(v) => v,
                    Err(_) => return,
                };

            if media_type != pw::spa::param::format::MediaType::Audio
                || media_subtype != pw::spa::param::format::MediaSubtype::Raw
            {
                return;
            }

            let mut format = pw::spa::param::audio::AudioInfoRaw::default();
            if format.parse(param).is_ok() {
                let rate = format.rate();
                rate_for_param.store(rate, Ordering::Relaxed);
            }
        })
        .process(move |stream: &pw::stream::Stream, _info| {
            if !thread_configured {
                rt_setup::configure_realtime_thread(target_cpu, rt_status_for_process.clone());
                thread_configured = true;
            }

            for slot in parking_lot.iter_mut() {
                let Some(old) = slot.take() else { continue };
                if let Err(rtrb::PushError::Full(old_back)) = gc_producer.push(old) {
                    *slot = Some(old_back);
                    break;
                }
            }

            rt_callback::drain_resamplers(
                &mut resampler_consumer,
                &mut resampler,
                &mut gc_producer,
                &mut parking_lot,
                &gc_overflow_for_process,
                &rt_status_for_process,
            );

            let param_changed = rt_callback::receive_commands(
                &mut consumer,
                &mut model_input_mult_adj,
                &mut model_output_mult_adj,
                &mut current_nam_rate,
                &mut active_model_l,
                &mut active_model_r,
                &mut gc_producer,
                &mut parking_lot,
                &gc_overflow_for_process,
                &rt_status_for_process,
                &mut user_input_gain_mult,
                &mut user_output_gain_mult,
                &mut gate_params,
                &mut threshold_open_sq,
                &mut threshold_close_sq,
                lut,
            );

            let current_pw_rate = rt_callback::sync_rate(
                &rate_for_process,
                &resampler,
                current_nam_rate,
                &rt_status_for_process,
            );

            if param_changed {
                rt_setup::compute_gain_multipliers(
                    user_input_gain_mult,
                    user_output_gain_mult,
                    model_input_mult_adj,
                    model_output_mult_adj,
                    &mut input_gain_mult,
                    &mut output_gain_mult,
                );
            }

            if rt_status_for_process.check_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING)
            {
                if rt_status_for_process
                    .check_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED)
                {
                    rt_status_for_process
                        .clear_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING);
                } else {
                    let _ = stream.dequeue_buffer();
                    return;
                }
            }

            rt_callback::process_dsp_buffer(
                stream,
                DspPipelineContext {
                    resampler: &mut resampler,
                    active_model_l: &mut active_model_l,
                    active_model_r: &mut active_model_r,
                    input_gain_mult,
                    output_gain_mult,
                    gate_params: &gate_params,
                    silence_hysteresis: &mut silence_hysteresis,
                    mono_hysteresis: &mut mono_hysteresis,
                    threshold_open_sq,
                    threshold_close_sq,
                    process_mono: &mut process_mono,
                    rt_status: &rt_status_for_process,
                    bridge_writer: DspBridgeWriter::from_ref(bridge_ptr),
                },
                DspBuffers {
                    resamp_mid_l: &mut resamp_mid_l,
                    resamp_mid_r: &mut resamp_mid_r,
                    resamp_out_l: &mut resamp_out_l,
                    resamp_out_r: &mut resamp_out_r,
                    model_out_l: &mut model_out_l,
                    model_out_r: &mut model_out_r,
                },
                current_pw_rate,
                &mut frame_count,
                &rt_status_for_process,
            );
        })
        .register()?;

    let mut audio_info = pw::spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(pw::spa::param::audio::AudioFormat::F32P);
    audio_info.set_channels(2);

    let mut format_buf = [0u8; 1024];
    let format_pod = unsafe { build_spa_format_pod(&audio_info, &mut format_buf)? };

    capture_stream.connect(
        pw::spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS
            | pw::stream::StreamFlags::EXCLUSIVE,
        &mut [format_pod],
    )?;

    log::info!(
        "{} Capture stream connected to PipeWire (Audio/Sink, F32P Planar Stereo).",
        "🎼".bright_blue()
    );

    Ok((capture_stream, capture_listener))
}
