// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 5.1.3. RATE SYNCHRONIZATION (Clock Tracking)
//! Checks for frequency discrepancy and sends a request to the Main Thread.

use crate::common::spsc::RtStatusFlags;
use crate::dsp::resampler::NamResampler;

use std::sync::atomic::Ordering;

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
