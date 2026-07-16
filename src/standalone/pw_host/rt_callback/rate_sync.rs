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
    let detected_pw_rate = rate_for_process.swap(0, Ordering::Acquire); // pairs with Release store em capture/listeners.rs:59
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
        rt_status_for_process
            .set_flag_release(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
        rt_status_for_process.set_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING);
    }

    current_pw_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::spsc::{RT_STATUS_NEEDS_RESAMPLER_REBUILD, RT_STATUS_RESAMP_SWAP_PENDING};
    use std::sync::atomic::AtomicU32;

    fn make_resampler(pw: u32, nam: u32) -> NamResampler {
        NamResampler::new(pw, nam, 64).unwrap()
    }

    #[test]
    fn no_rate_change_no_flags() {
        let rate = AtomicU32::new(0);
        let rs = make_resampler(48000, 48000);
        let flags = RtStatusFlags::new();

        let result = sync_rate(&rate, &rs, 48000, &flags);

        assert_eq!(result, 48000);
        assert!(!flags.check_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD));
        assert!(!flags.check_flag(RT_STATUS_RESAMP_SWAP_PENDING));
    }

    #[test]
    fn pw_rate_change_sets_flags() {
        let rate = AtomicU32::new(44100);
        let rs = make_resampler(48000, 48000);
        let flags = RtStatusFlags::new();

        let result = sync_rate(&rate, &rs, 48000, &flags);

        assert_eq!(result, 48000);
        assert!(flags.check_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD));
        assert!(flags.check_flag(RT_STATUS_RESAMP_SWAP_PENDING));
        assert_eq!(flags.requested_pw_rate.load(Ordering::Relaxed), 44100);
    }

    #[test]
    fn nam_rate_change_sets_flags() {
        let rate = AtomicU32::new(0);
        let rs = make_resampler(48000, 48000);
        let flags = RtStatusFlags::new();

        let result = sync_rate(&rate, &rs, 44100, &flags);

        assert_eq!(result, 48000);
        assert!(flags.check_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD));
        assert!(flags.check_flag(RT_STATUS_RESAMP_SWAP_PENDING));
    }

    #[test]
    fn zero_detected_rate_ignored() {
        let rate = AtomicU32::new(0);
        let rs = make_resampler(48000, 44100);
        let flags = RtStatusFlags::new();

        let result = sync_rate(&rate, &rs, 44100, &flags);

        assert_eq!(result, 48000);
        assert!(!flags.check_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD));
    }

    #[test]
    fn both_rates_unchanged_no_flags() {
        let rate = AtomicU32::new(48000);
        let rs = make_resampler(48000, 44100);
        let flags = RtStatusFlags::new();

        let result = sync_rate(&rate, &rs, 44100, &flags);

        assert_eq!(result, 48000);
        assert!(!flags.check_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD));
    }

    #[test]
    fn rate_sync_returns_current_pw_rate() {
        let rate = AtomicU32::new(0);
        let rs = make_resampler(96000, 48000);
        let flags = RtStatusFlags::new();

        let result = sync_rate(&rate, &rs, 48000, &flags);

        assert_eq!(result, 96000);
    }

    #[test]
    fn nam_rate_mismatch_with_zero_detected_pw_rate_sets_flags() {
        let rate = AtomicU32::new(0);
        let rs = make_resampler(48000, 44100);
        let flags = RtStatusFlags::new();

        let result = sync_rate(&rate, &rs, 48000, &flags);

        assert_eq!(result, 48000);
        assert!(flags.check_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD));
        assert!(flags.check_flag(RT_STATUS_RESAMP_SWAP_PENDING));
        assert_eq!(flags.requested_nam_rate.load(Ordering::Relaxed), 48000);
    }
}
