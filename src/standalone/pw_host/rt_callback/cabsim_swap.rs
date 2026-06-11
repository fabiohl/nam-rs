// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cab-sim Convolution Engine Draining (Zero-Alloc Swap)
//! Replaces the active convolution engine without using memory allocation in the critical path.

use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::cabsim::conv::ConvEngine;

use rtrb::Consumer;

/// Drains the cab-sim convolution engine SPSC channel and swaps the active engine atomically.
///
/// Follows the same cascade pattern as `drain_resamplers`:
/// GC channel → parking_lot → overflow buffer.
///
/// An `Option` is used so that `None` can be sent to clear/bypass the convolution.
#[inline(always)]
pub fn drain_cabsims(
    cabsim_consumer: &mut Consumer<Option<Box<ConvEngine>>>,
    active_cabsim: &mut Option<Box<ConvEngine>>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow_for_process: &GcOverflowBuffer,
    rt_status_for_process: &RtStatusFlags,
) {
    while let Ok(new_engine) = cabsim_consumer.pop() {
        let old_engine = std::mem::replace(active_cabsim, new_engine);

        if let Some(old) = old_engine {
            crate::common::spsc::gc_cascade(
                Some(GcItem::CabConvEngine(old)),
                gc_producer,
                parking_lot,
                gc_overflow_for_process,
                rt_status_for_process,
            );
        }
    }
}
