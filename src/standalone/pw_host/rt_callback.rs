// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Helper functions executed inside the capture stream's `process()` RT callback.
//!
//! All functions in this module follow the absolute callback rules:
//! - Zero heap allocation
//! - Zero I/O
//! - Zero mutexes

use crate::common::spsc::{GcItem, GcOverflowBuffer, ParamPayload, RtStatusFlags};
use crate::dsp::gate::GateParams;
use crate::dsp::pipeline::{DspBuffers, DspPipelineContext, capture_dsp_pipeline};
use crate::dsp::resampler::NamResampler;
use crate::standalone::rt_setup;

use pipewire as pw;
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// 5.1.1. Resampler Draining (Zero-Alloc Swap)
/// Replaces resamplers without using memory allocation in the critical path.
#[inline(always)]
pub fn drain_resamplers(
    resampler_consumer: &mut Consumer<Box<NamResampler>>,
    resampler: &mut Box<NamResampler>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow_for_process: &GcOverflowBuffer,
    rt_status_for_process: &RtStatusFlags,
) {
    while let Ok(new_rs) = resampler_consumer.pop() {
        let new_rs = new_rs;
        rt_status_for_process
            .active_rate
            .store(new_rs.pw_rate(), Ordering::Relaxed);
        rt_status_for_process
            .active_rate_changed
            .store(new_rs.pw_rate(), Ordering::Relaxed);

        let old_rs = std::mem::replace(resampler, new_rs);

        rt_status_for_process.clear_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING);

        let mut item = Some(GcItem::Resampler(old_rs));

        if let Some(i) = item.take() {
            if let Err(rtrb::PushError::Full(returned)) = gc_producer.push(i) {
                item = Some(returned);
            } else {
                continue;
            }
        }

        if let Some(i) = item.take() {
            let mut parked = false;
            let mut i_opt = Some(i);
            for slot in parking_lot.iter_mut() {
                if slot.is_none() {
                    *slot = i_opt.take();
                    parked = true;
                    break;
                }
            }
            if !parked {
                item = i_opt;
            } else {
                continue;
            }
        }

        if let Some(i) = item.take() {
            rt_status_for_process.set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
            gc_overflow_for_process.push(i);
        }
    }
}

/// 5.1.2. COMMAND RECEPTION (SPSC Channel)
/// Processes commands from the command-line interface or control system (volume, model, noise gate).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn receive_commands(
    consumer: &mut Consumer<ParamPayload>,
    model_input_mult_adj: &mut f32,
    model_output_mult_adj: &mut f32,
    current_nam_rate: &mut u32,
    active_model_l: &mut Option<Box<crate::models::DynamicModel>>,
    active_model_r: &mut Option<Box<crate::models::DynamicModel>>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow_for_process: &GcOverflowBuffer,
    rt_status_for_process: &Arc<RtStatusFlags>,
    user_input_gain_mult: &mut f32,
    user_output_gain_mult: &mut f32,
    gate_params: &mut GateParams,
    threshold_open_sq: &mut f32,
    threshold_close_sq: &mut f32,
    lut: &crate::math::dsp::gain_lut::GainLUT,
) -> bool {
    let mut param_changed = false;

    while let Ok(payload) = consumer.pop() {
        match payload {
            ParamPayload::LoadModel {
                model_l,
                model_r,
                input_mult_adj,
                output_mult_adj,
                sample_rate,
            } => {
                let new_model_l = model_l;
                let new_model_r = model_r;

                if new_model_l.is_some() || new_model_r.is_some() {
                    *model_input_mult_adj = input_mult_adj;
                    *model_output_mult_adj = output_mult_adj;
                    *current_nam_rate = sample_rate;
                } else {
                    *model_input_mult_adj = 1.0;
                    *model_output_mult_adj = 1.0;
                    *current_nam_rate = 48_000;
                }

                let mut old_models: [Option<Box<crate::models::DynamicModel>>; 2] = [None, None];
                rt_status_for_process.clear_flag(crate::common::spsc::RT_STATUS_A2_PLACEHOLDER);
                if let Some(old) = std::mem::replace(active_model_l, new_model_l) {
                    old_models[0] = Some(old);
                }
                if let Some(model) = active_model_l {
                    model.inject_rt_status(Arc::clone(rt_status_for_process));
                }
                if let Some(old) = std::mem::replace(active_model_r, new_model_r) {
                    old_models[1] = Some(old);
                }
                if let Some(model) = active_model_r {
                    model.inject_rt_status(Arc::clone(rt_status_for_process));
                }

                for m_opt in &mut old_models {
                    if let Some(m) = m_opt.take() {
                        let mut item = Some(GcItem::Model(m));

                        if let Some(v) = item.take() {
                            if let Err(rtrb::PushError::Full(returned)) = gc_producer.push(v) {
                                item = Some(returned);
                            } else {
                                continue;
                            }
                        }

                        if let Some(v) = item.take() {
                            let mut parked = false;
                            let mut v_opt = Some(v);
                            for slot in parking_lot.iter_mut() {
                                if slot.is_none() {
                                    *slot = v_opt.take();
                                    parked = true;
                                    break;
                                }
                            }
                            if !parked {
                                item = v_opt;
                            }
                        }

                        if let Some(v) = item.take() {
                            rt_status_for_process
                                .set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
                            gc_overflow_for_process.push(v);
                        }
                    }
                }
                param_changed = true;
            }
            ParamPayload::InputGain(mult) => {
                *user_input_gain_mult = mult;
                param_changed = true;
            }
            ParamPayload::OutputGain(mult) => {
                *user_output_gain_mult = mult;
                param_changed = true;
            }
            ParamPayload::GateConfig(params) => {
                let open_lin = lut.db_to_linear(params.threshold_open_db);
                let close_lin = lut.db_to_linear(params.threshold_close_db);
                *threshold_open_sq = open_lin * open_lin;
                *threshold_close_sq = close_lin * close_lin;
                *gate_params = params;
            }
        }
    }
    param_changed
}

/// 5.1.3. RATE SYNCHRONIZATION (Clock Tracking)
/// Checks for frequency discrepancy and sends a request to the Main Thread.
#[inline(always)]
pub fn sync_rate(
    rate_for_process: &std::sync::atomic::AtomicU32,
    resampler: &NamResampler,
    current_nam_rate: u32,
    rt_status_for_process: &RtStatusFlags,
) -> u32 {
    let detected_pw_rate = rate_for_process.swap(0, Ordering::Relaxed);
    let current_pw_rate = resampler.pw_rate();

    let mut pw_rate_to_request = current_pw_rate;
    let mut requires_rebuild = false;

    if detected_pw_rate != 0 && detected_pw_rate != current_pw_rate {
        pw_rate_to_request = detected_pw_rate;
        requires_rebuild = true;
    }

    if current_nam_rate != resampler.nam_rate() {
        requires_rebuild = true;
    }

    if requires_rebuild && pw_rate_to_request != 0 {
        rt_status_for_process
            .requested_pw_rate
            .store(pw_rate_to_request, Ordering::Relaxed);
        rt_status_for_process
            .requested_nam_rate
            .store(current_nam_rate, Ordering::Relaxed);
        rt_status_for_process.set_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
        rt_status_for_process.set_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING);
    }

    current_pw_rate
}

/// 5.1.4. REAL-TIME DSP LOGIC
/// Acquires the raw system buffer and delegates the heavy lifting to the Audio Factory (pipeline).
#[inline(always)]
pub fn process_dsp_buffer(
    stream: &pw::stream::Stream,
    context: DspPipelineContext,
    buffers: DspBuffers,
    current_pw_rate: u32,
    frame_count: &mut u32,
    rt_status_for_process: &RtStatusFlags,
) {
    let mut _buf = match stream.dequeue_buffer() {
        Some(b) => b,
        None => return,
    };

    let datas = _buf.datas_mut();
    if datas.len() >= 2 {
        let (left_datas, right_datas) = datas.split_at_mut(1);
        if let (Some(d_l), Some(d_r)) = (left_datas.first_mut(), right_datas.first_mut()) {
            let chunk_l = d_l.chunk();
            let chunk_r = d_r.chunk();
            let offset_l = chunk_l.offset() as usize;
            let size_l = chunk_l.size() as usize;
            let offset_r = chunk_r.offset() as usize;
            let size_r = chunk_r.size() as usize;

            if let (Some(raw_l), Some(raw_r)) = (d_l.data(), d_r.data()) {
                let n_bytes_l = size_l.min(raw_l.len().saturating_sub(offset_l));
                let n_bytes_r = size_r.min(raw_r.len().saturating_sub(offset_r));
                let n_samples_l = n_bytes_l / std::mem::size_of::<f32>();
                let n_samples_r = n_bytes_r / std::mem::size_of::<f32>();

                let n_samples = n_samples_l.min(n_samples_r);

                if n_samples > 0 {
                    let samples_l = unsafe {
                        std::slice::from_raw_parts_mut(
                            raw_l.as_mut_ptr().add(offset_l).cast::<f32>(),
                            n_samples,
                        )
                    };
                    let samples_r = unsafe {
                        std::slice::from_raw_parts_mut(
                            raw_r.as_mut_ptr().add(offset_r).cast::<f32>(),
                            n_samples,
                        )
                    };

                    let should_measure = (*frame_count & 0xF) == 0;
                    *frame_count = frame_count.wrapping_add(1);

                    let start_nanos = if should_measure {
                        rt_setup::rdtsc_nanos()
                    } else {
                        0
                    };

                    capture_dsp_pipeline(samples_l, samples_r, n_samples, context, buffers);

                    if should_measure {
                        let elapsed_nanos = rt_setup::rdtsc_nanos().wrapping_sub(start_nanos);
                        rt_status_for_process
                            .dsp_cycle_time
                            .store(elapsed_nanos, Ordering::Relaxed);
                        rt_status_for_process.latency_hist.record(elapsed_nanos);

                        let elapsed_secs = elapsed_nanos as f64 / 1_000_000_000.0;
                        let budget_secs = (n_samples as f64 / current_pw_rate as f64) * 0.85;
                        if elapsed_secs > budget_secs {
                            rt_status_for_process
                                .dsp_overloads
                                .fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    rt_status_for_process
                        .last_n_samples
                        .store(n_samples as u32, Ordering::Relaxed);
                }
            }
        }
    }
}
