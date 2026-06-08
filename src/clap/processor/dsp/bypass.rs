// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use crate::clap::processor::dsp::peaks;
use clack_plugin::{prelude::*, process::audio::PortPair};

impl<'a> NamClapProcessor<'a> {
    #[inline(always)]
    pub(super) fn process_bypass(
        &self,
        port_pair: &mut PortPair<'_>,
        n_samples: usize,
    ) -> Result<bool, PluginError> {
        if !self.params.bypass {
            return Ok(false);
        }
        self.process_bypass_cold(port_pair, n_samples)
    }

    #[cold]
    #[inline(never)]
    fn process_bypass_cold(
        &self,
        port_pair: &mut PortPair<'_>,
        n_samples: usize,
    ) -> Result<bool, PluginError> {
        let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
            return Ok(true);
        };
        let mut peak_l = 0.0f32;
        let mut peak_r = 0.0f32;
        let mut channel_iter = channel_pairs.into_iter();

        if let Some(pair) = channel_iter.next() {
            match pair {
                ChannelPair::InputOutput(i, o) => {
                    let n = n_samples.min(o.len());
                    o[..n].copy_from_slice(&i[..n]);
                    for &sample in &o[..n] {
                        let abs_val = sample.abs();
                        if abs_val > peak_l {
                            peak_l = abs_val;
                        }
                    }
                }
                ChannelPair::InPlace(io) => {
                    let n = n_samples.min(io.len());
                    for &sample in &io[..n] {
                        let abs_val = sample.abs();
                        if abs_val > peak_l {
                            peak_l = abs_val;
                        }
                    }
                }
                ChannelPair::InputOnly(_) | ChannelPair::OutputOnly(_) => {}
            }
        }
        if let Some(pair) = channel_iter.next() {
            match pair {
                ChannelPair::InputOutput(i, o) => {
                    let n = n_samples.min(o.len());
                    o[..n].copy_from_slice(&i[..n]);
                    for &sample in &o[..n] {
                        let abs_val = sample.abs();
                        if abs_val > peak_r {
                            peak_r = abs_val;
                        }
                    }
                }
                ChannelPair::InPlace(io) => {
                    let n = n_samples.min(io.len());
                    for &sample in &io[..n] {
                        let abs_val = sample.abs();
                        if abs_val > peak_r {
                            peak_r = abs_val;
                        }
                    }
                }
                ChannelPair::InputOnly(_) | ChannelPair::OutputOnly(_) => {}
            }
        }

        peaks::store_peaks(self.shared, peak_l, peak_r);
        Ok(true)
    }
}
