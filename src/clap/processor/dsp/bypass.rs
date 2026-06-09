// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use crate::clap::processor::dsp::peaks;
use crate::math::dsp::stereo::compute_peak_abs_stereo;
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
                    // SAFETY: both slices reference the same valid buffer `o`
                    // with identical length `n`, satisfying the equal-length
                    // requirement of `compute_peak_abs_stereo`.
                    let (p, _) = unsafe { compute_peak_abs_stereo(&o[..n], &o[..n]) };
                    peak_l = p;
                }
                ChannelPair::InPlace(io) => {
                    let n = n_samples.min(io.len());
                    // SAFETY: both slices reference the same valid buffer `io`
                    // with identical length `n`, satisfying the equal-length
                    // requirement of `compute_peak_abs_stereo`.
                    let (p, _) = unsafe { compute_peak_abs_stereo(&io[..n], &io[..n]) };
                    peak_l = p;
                }
                ChannelPair::InputOnly(_) | ChannelPair::OutputOnly(_) => {}
            }
        }
        if let Some(pair) = channel_iter.next() {
            match pair {
                ChannelPair::InputOutput(i, o) => {
                    let n = n_samples.min(o.len());
                    o[..n].copy_from_slice(&i[..n]);
                    // SAFETY: both slices reference the same valid buffer `o`
                    // with identical length `n`, satisfying the equal-length
                    // requirement of `compute_peak_abs_stereo`.
                    let (p, _) = unsafe { compute_peak_abs_stereo(&o[..n], &o[..n]) };
                    peak_r = p;
                }
                ChannelPair::InPlace(io) => {
                    let n = n_samples.min(io.len());
                    // SAFETY: both slices reference the same valid buffer `io`
                    // with identical length `n`, satisfying the equal-length
                    // requirement of `compute_peak_abs_stereo`.
                    let (p, _) = unsafe { compute_peak_abs_stereo(&io[..n], &io[..n]) };
                    peak_r = p;
                }
                ChannelPair::InputOnly(_) | ChannelPair::OutputOnly(_) => {}
            }
        }

        peaks::store_peaks(self.shared, peak_l, peak_r);
        Ok(true)
    }
}
