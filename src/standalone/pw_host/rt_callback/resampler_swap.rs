// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 5.1.1. Resampler Draining (Zero-Alloc Swap)
//! Replaces resamplers without using memory allocation in the critical path.

use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags, gc_cascade};
use crate::dsp::resampler::NamResampler;

use rtrb::Consumer;
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
        rt_status_for_process
            .active_rate
            .store(new_rs.pw_rate(), Ordering::Relaxed);
        rt_status_for_process
            .active_rate_changed
            .store(new_rs.pw_rate(), Ordering::Relaxed);

        let old_rs = std::mem::replace(resampler, new_rs);

        rt_status_for_process.clear_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING);

        gc_cascade(
            Some(GcItem::Resampler(old_rs)),
            gc_producer,
            parking_lot,
            gc_overflow_for_process,
            rt_status_for_process,
        );
    }
}
