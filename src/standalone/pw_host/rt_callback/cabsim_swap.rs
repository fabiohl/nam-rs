// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Cab-sim IR Draining (Zero-Alloc Swap)
//! Replaces the active IR without using memory allocation in the critical path.

use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags};
use crate::dsp::cabsim::loader::CabSimIr;

use rtrb::Consumer;

/// Drains the cab-sim IR SPSC channel and swaps the active IR atomically.
///
/// Follows the same cascade pattern as `drain_resamplers`:
/// GC channel → parking_lot → overflow buffer.
///
/// An `Option` is used so that `None` can be sent to clear/bypass the IR.
#[inline(always)]
pub fn drain_cabsims(
    cabsim_consumer: &mut Consumer<Option<Box<CabSimIr>>>,
    active_cabsim: &mut Option<Box<CabSimIr>>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow_for_process: &GcOverflowBuffer,
    rt_status_for_process: &RtStatusFlags,
) {
    while let Ok(new_ir) = cabsim_consumer.pop() {
        let old_ir = std::mem::replace(active_cabsim, new_ir);

        if let Some(old) = old_ir {
            crate::common::spsc::gc_cascade(
                Some(GcItem::CabSimIr(old)),
                gc_producer,
                parking_lot,
                gc_overflow_for_process,
                rt_status_for_process,
            );
        }
    }
}
