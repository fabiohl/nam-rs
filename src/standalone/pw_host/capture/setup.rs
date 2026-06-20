// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `setup_capture_stream` — configures the PipeWire capture stream (Virtual Sink)
//! and its RT listener, with process, param_changed, and state_changed closures.

use super::super::rt_callback;
use super::state::CaptureState;
use crate::common::spsc::{GcItem, GcOverflowBuffer, ParamPayload, RtStatusFlags};
use crate::dsp::pipeline::{
    BridgeRef, DspBridgeWriter, DspBuffers, DspPipelineContext, build_spa_format_pod,
};
use crate::standalone::colors::Colorize;
use crate::standalone::rt_setup;

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
    ir_raw_samples: Option<Vec<f32>>,
    sys: &crate::common::diagnostics::SystemSnapshot,
    target_cpu: usize,
    mut consumer: Consumer<ParamPayload>,
    mut gc_producer: rtrb::Producer<GcItem>,
    gc_overflow: Arc<GcOverflowBuffer>,
    mut resampler_consumer: Consumer<Box<crate::dsp::resampler::NamResampler>>,
    mut cabsim_consumer: Consumer<Option<Box<crate::dsp::cabsim::conv::ConvEngine>>>,
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

    let mut state = CaptureState::init(sys);
    state.ir_raw_samples = ir_raw_samples;
    let rate_for_param = state.shared_target_rate.clone();
    let rate_for_process = state.shared_target_rate.clone();

    let rt_status_for_process = rt_status.clone();
    let gc_overflow_for_process = gc_overflow.clone();

    let lut = crate::math::dsp::gain_lut::get_gain_lut();

    let capture_listener = capture_stream
        .add_local_listener::<()>()
        .state_changed(super::listeners::state_changed_handler)
        .param_changed(move |stream, user_data, id, param| {
            super::listeners::param_changed_handler(stream, user_data, id, param, &rate_for_param)
        })
        .process(move |stream: &pw::stream::Stream, _info| {
            if !state.thread_configured {
                rt_setup::configure_realtime_thread(target_cpu, rt_status_for_process.clone());
                state.thread_configured = true;
            }

            for slot in state.parking_lot.iter_mut() {
                let Some(old) = slot.take() else { continue };
                if let Err(rtrb::PushError::Full(old_back)) = gc_producer.push(old) {
                    *slot = Some(old_back);
                    break;
                }
            }

            rt_callback::drain_resamplers(
                &mut resampler_consumer,
                &mut state.resampler,
                &mut gc_producer,
                &mut state.parking_lot,
                &gc_overflow_for_process,
                &rt_status_for_process,
            );

            rt_callback::drain_cabsims(
                &mut cabsim_consumer,
                &mut state.active_cabsim,
                &mut gc_producer,
                &mut state.parking_lot,
                &gc_overflow_for_process,
                &rt_status_for_process,
            );

            let param_changed = rt_callback::receive_commands(
                &mut consumer,
                &mut state.model_input_mult_adj,
                &mut state.model_output_mult_adj,
                &mut state.current_nam_rate,
                &mut state.active_model_l,
                &mut state.active_model_r,
                &mut gc_producer,
                &mut state.parking_lot,
                &gc_overflow_for_process,
                &rt_status_for_process,
                &mut state.user_input_gain_mult,
                &mut state.user_output_gain_mult,
                &mut state.gate_params,
                &mut state.threshold_open_sq,
                &mut state.threshold_close_sq,
                lut,
                &mut state.adaptive_compute,
            );

            rt_callback::try_slimmable_rebuild(
                &mut state.active_model_l,
                &mut state.active_model_r,
                &mut gc_producer,
                &mut state.parking_lot,
                &gc_overflow_for_process,
                &rt_status_for_process,
                &mut state.adaptive_compute,
            );

            let current_pw_rate = rt_callback::sync_rate(
                &rate_for_process,
                &state.resampler,
                state.current_nam_rate,
                &rt_status_for_process,
            );

            if param_changed {
                rt_setup::compute_gain_multipliers(
                    state.user_input_gain_mult,
                    state.user_output_gain_mult,
                    state.model_input_mult_adj,
                    state.model_output_mult_adj,
                    &mut state.input_gain_mult,
                    &mut state.output_gain_mult,
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
                    resampler: &mut state.resampler,
                    active_model_l: &mut state.active_model_l,
                    active_model_r: &mut state.active_model_r,
                    input_gain_mult: state.input_gain_mult,
                    output_gain_mult: state.output_gain_mult,
                    gate_params: &state.gate_params,
                    silence_hysteresis: &mut state.silence_hysteresis,
                    mono_hysteresis: &mut state.mono_hysteresis,
                    threshold_open_sq: state.threshold_open_sq,
                    threshold_close_sq: state.threshold_close_sq,
                    process_mono: &mut state.process_mono,
                    rt_status: &rt_status_for_process,
                    adaptive: &mut state.adaptive_compute,
                    bridge_writer: DspBridgeWriter::from_ref(bridge_ptr),
                    conv: state.active_cabsim.as_mut().map(|e| e.as_mut()),
                },
                DspBuffers {
                    resamp_mid_l: &mut state.resamp_mid_l,
                    resamp_mid_r: &mut state.resamp_mid_r,
                    resamp_out_l: &mut state.resamp_out_l,
                    resamp_out_r: &mut state.resamp_out_r,
                    model_out_l: &mut state.model_out_l,
                    model_out_r: &mut state.model_out_r,
                },
                current_pw_rate,
                &mut state.frame_count,
                &rt_status_for_process,
            );

            if (state.frame_count.wrapping_sub(1) & 0xF) == 0 {
                let elapsed_nanos = rt_status_for_process.dsp_cycle_time.load(Ordering::Relaxed);
                if elapsed_nanos > 0 {
                    let n_samples =
                        rt_status_for_process.last_n_samples.load(Ordering::Relaxed) as u64;
                    if n_samples > 0 && current_pw_rate > 0 {
                        let budget_ns = n_samples * 1_000_000_000 / current_pw_rate as u64;
                        let latency_us = elapsed_nanos / 1000;
                        let budget_us = budget_ns / 1000;
                        state.adaptive_compute.update(
                            latency_us,
                            budget_us,
                            current_pw_rate,
                            &rt_status_for_process,
                        );
                    }
                }
            }

            // Detect cabsim partition mismatch and signal rebuild
            if let Some(ref cabsim) = state.active_cabsim {
                let n_samples =
                    rt_status_for_process.last_n_samples.load(Ordering::Relaxed) as usize;
                if n_samples > 0
                    && cabsim.partition_size() != n_samples
                    && state.ir_raw_samples.is_some()
                {
                    rt_status_for_process
                        .set_flag(crate::common::spsc::RT_STATUS_NEEDS_CABSIM_REBUILD);
                    rt_status_for_process
                        .requested_cabsim_partition_size
                        .store(n_samples as u32, Ordering::Relaxed);
                }
            }
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
        "\u{1f3bc}".bright_blue()
    );

    Ok((capture_stream, capture_listener))
}
