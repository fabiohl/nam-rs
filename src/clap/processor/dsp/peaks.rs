// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use crate::clap::plugin::NamClapShared;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    #[inline(always)]
    pub(super) fn compute_output_peaks(
        &mut self,
        out_l: &mut Option<&mut [f32]>,
        out_r: &mut Option<&mut [f32]>,
        n_out: usize,
    ) -> (f32, f32) {
        let mut peak_l = 0.0f32;
        let mut peak_r = 0.0f32;
        let mut len_l = 0;
        if let Some(o_l) = out_l {
            let n = n_out.min(o_l.len());
            o_l[..n].copy_from_slice(&self.buf_out_l[..n]);
            len_l = n;
        }
        let mut len_r = 0;
        if let Some(o_r) = out_r {
            let n = n_out.min(o_r.len());
            #[cfg(feature = "stereo")]
            o_r[..n].copy_from_slice(&self.buf_out_r[..n]);
            #[cfg(not(feature = "stereo"))]
            o_r[..n].copy_from_slice(&self.buf_out_l[..n]);
            len_r = n;
        }

        #[cfg(feature = "stereo")]
        {
            if len_l > 0 && len_r > 0 {
                let n = len_l.min(len_r);
                let (p_l, p_r) = unsafe {
                    crate::math::dsp::stereo::compute_peak_abs_stereo(
                        &self.buf_out_l[..n],
                        &self.buf_out_r[..n],
                    )
                };
                peak_l = p_l;
                peak_r = p_r;
            } else if len_l > 0 {
                let (p_l, _) = unsafe {
                    crate::math::dsp::stereo::compute_peak_abs_stereo(
                        &self.buf_out_l[..len_l],
                        &self.buf_out_l[..len_l],
                    )
                };
                peak_l = p_l;
            } else if len_r > 0 {
                let (_, p_r) = unsafe {
                    crate::math::dsp::stereo::compute_peak_abs_stereo(
                        &self.buf_out_r[..len_r],
                        &self.buf_out_r[..len_r],
                    )
                };
                peak_r = p_r;
            }
        }
        #[cfg(not(feature = "stereo"))]
        {
            if len_l > 0 {
                let (p_l, _) = unsafe {
                    crate::math::dsp::stereo::compute_peak_abs_stereo(
                        &self.buf_out_l[..len_l],
                        &self.buf_out_l[..len_l],
                    )
                };
                peak_l = p_l;
                peak_r = p_l;
            } else if len_r > 0 {
                let (p_l, _) = unsafe {
                    crate::math::dsp::stereo::compute_peak_abs_stereo(
                        &self.buf_out_l[..len_r],
                        &self.buf_out_l[..len_r],
                    )
                };
                peak_l = p_l;
                peak_r = p_l;
            }
        }

        (peak_l, peak_r)
    }
}

#[inline(always)]
pub(super) fn store_peaks(shared: &NamClapShared, peak_l: f32, peak_r: f32) {
    let current_peak_l = f32::from_bits(shared.rt_to_ui.ui_peak_l.load(Ordering::Relaxed));
    if peak_l > current_peak_l {
        shared
            .rt_to_ui
            .ui_peak_l
            .store(peak_l.to_bits(), Ordering::Relaxed);
    }
    let current_peak_r = f32::from_bits(shared.rt_to_ui.ui_peak_r.load(Ordering::Relaxed));
    if peak_r > current_peak_r {
        shared
            .rt_to_ui
            .ui_peak_r
            .store(peak_r.to_bits(), Ordering::Relaxed);
    }
    if peak_l > 1.0 || peak_r > 1.0 {
        shared.rt_to_ui.ui_clipped.store(true, Ordering::Relaxed);
    }
}
