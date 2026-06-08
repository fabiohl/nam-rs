// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use minstant::Instant;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    #[inline(always)]
    pub(super) fn process_telemetry(&mut self, start_time: Option<Instant>) {
        if let Some(start_time) = start_time {
            let elapsed_nanos = start_time.elapsed().as_nanos() as u64;
            self.rt_status
                .dsp_cycle_time
                .store(elapsed_nanos, Ordering::Relaxed);
            self.rt_status.latency_hist.record(elapsed_nanos);

            let sample_rate = self.shared.cold.sample_rate.load(Ordering::Relaxed);
            let last_n_samples = self.rt_status.last_n_samples.load(Ordering::Relaxed);
            if sample_rate > 0 && last_n_samples > 0 {
                let budget_ns = (last_n_samples as u64 * 1_000_000_000) / sample_rate as u64;
                let threshold_ns = (budget_ns * 85) / 100;
                if elapsed_nanos > threshold_ns {
                    self.rt_status.dsp_overloads.fetch_add(1, Ordering::Relaxed);
                }

                let latency_us = elapsed_nanos / 1000;
                let budget_us = budget_ns / 1000;
                self.adaptive_compute
                    .update(latency_us, budget_us, sample_rate, &self.rt_status);
            }
        }
    }
}
