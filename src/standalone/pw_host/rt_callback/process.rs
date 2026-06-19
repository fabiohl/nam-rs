// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 5.1.4. REAL-TIME DSP LOGIC
//! Acquires the raw system buffer and delegates the heavy lifting to the Audio Factory (pipeline).

use crate::common::spsc::RtStatusFlags;
use crate::dsp::pipeline::{DspBuffers, DspPipelineContext, capture_dsp_pipeline};
use crate::standalone::rt_setup;

use pipewire as pw;
use std::sync::atomic::Ordering;

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

                    if (*frame_count & 0x3FF) == 0 {
                        unsafe {
                            crate::math::common::set_daz_ftz();
                        }
                    }

                    capture_dsp_pipeline(
                        samples_l,
                        samples_r,
                        n_samples,
                        context,
                        buffers,
                        current_pw_rate,
                    );

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
