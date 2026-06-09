// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Garbage collection: safe disposal of heap-allocated resources from the audio thread.
//! Must remain alloc/lock-free (array parking-lot + overflow buffer).

use super::NamClapProcessor;
use crate::common::spsc::{GcItem, RT_STATUS_GC_OVERFLOW};

impl<'a> NamClapProcessor<'a> {
    /// Attempts to drain items parked in the parking lot back to the SPSC GC channel.
    /// Called at the start of each process() block so that items parked during a previous
    /// swap (when the GC channel was full) are retried every audio cycle.
    /// Capacity of 16 is safe: 3 slots per swap (2 channels + 1 resampler), and draining
    /// happens on every block (~0.3–1.5ms @ 48kHz), yielding >96 slots/s throughput.
    #[inline]
    pub(super) fn drain_parking_lot(&mut self) {
        for slot in self.parking_lot.iter_mut() {
            let Some(item) = slot.take() else { continue };
            if let Err(rtrb::PushError::Full(returned)) = self.gc_tx.push(item) {
                *slot = Some(returned);
                break;
            }
        }
    }

    /// Attempts to send an item for safe disposal (GC).
    /// If the main channel is full, falls back to the parking lot and then the overflow buffer.
    pub(super) fn push_to_gc(&mut self, item: GcItem) {
        let mut item = Some(item);

        // 1. Try the main channel (SPSC)
        if let Some(i) = item.take() {
            if let Err(rtrb::PushError::Full(returned)) = self.gc_tx.push(i) {
                item = Some(returned);
            } else {
                return; // Success!
            }
        }

        // 2. If that failed, try the Parking Lot (Array stack-based)
        //    With drain_parking_lot() called on every block, 16 slots are more than
        //    sufficient. Worst case: 3 slots/swap (2 channels + 1 resampler) with
        //    draining at ~0.3–1.5ms intervals => >96 slots/s throughput.
        if let Some(i) = item.take() {
            let mut i_opt = Some(i);
            for slot in self.parking_lot.iter_mut() {
                if slot.is_none() {
                    *slot = i_opt.take();
                    return; // Successfully parked!
                }
            }
            item = i_opt;
        }

        // 3. If even the Parking Lot failed, use the Overflow Buffer (overwrite/controlled leak)
        if let Some(i) = item.take() {
            self.rt_status.set_flag(RT_STATUS_GC_OVERFLOW);
            self.gc_overflow.push(i);
        }
    }
}
