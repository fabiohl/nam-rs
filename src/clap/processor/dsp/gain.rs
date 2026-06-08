// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::NamClapProcessor;
use std::sync::atomic::Ordering;

impl<'a> NamClapProcessor<'a> {
    #[inline(always)]
    pub(super) fn apply_input_gain(&mut self, n_samples: usize) {
        let mut input_has_clipped = false;
        #[cfg(feature = "stereo")]
        {
            let start = self.smoother_in.peek();
            let target = self.smoother_in.target_value();
            if (start - target).abs() < 1e-9 {
                input_has_clipped = unsafe {
                    crate::math::dsp::gain::apply_gain_and_detect_clipping_stereo(
                        &mut self.buf_host_l[..n_samples],
                        &mut self.buf_host_r[..n_samples],
                        start,
                    )
                };
            } else {
                let step = (target - start) / n_samples as f32;
                unsafe {
                    crate::math::dsp::gain::apply_ramp_stereo(
                        &mut self.buf_host_l[..n_samples],
                        &mut self.buf_host_r[..n_samples],
                        start,
                        step,
                    );
                }
                self.smoother_in.set(target);
                let (peak_l, peak_r) = unsafe {
                    crate::math::dsp::stereo::compute_peak_abs_stereo(
                        &self.buf_host_l[..n_samples],
                        &self.buf_host_r[..n_samples],
                    )
                };
                if peak_l > 1.0 || peak_r > 1.0 {
                    input_has_clipped = true;
                }
            }
        }
        #[cfg(not(feature = "stereo"))]
        {
            let start = self.smoother_in.peek();
            let target = self.smoother_in.target_value();
            if (start - target).abs() < 1e-9 {
                crate::math::dsp::gain::apply_gain_simd(&mut self.buf_host_l[..n_samples], start);
            } else {
                let step = (target - start) / n_samples as f32;
                crate::math::dsp::gain::apply_ramp_simd(
                    &mut self.buf_host_l[..n_samples],
                    start,
                    step,
                );
                self.smoother_in.set(target);
            }
            for &sample in &self.buf_host_l[..n_samples] {
                if sample.abs() > 1.0 {
                    input_has_clipped = true;
                    break;
                }
            }
        }
        if input_has_clipped {
            self.shared
                .rt_to_ui
                .ui_clipped
                .store(true, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub(super) fn apply_output_gain(&mut self, n_out: usize) {
        #[cfg(feature = "stereo")]
        {
            let start = self.smoother_out.peek();
            let target = self.smoother_out.target_value();
            if (start - target).abs() < 1e-9 {
                unsafe {
                    crate::math::dsp::gain::apply_gain_stereo(
                        &mut self.buf_out_l[..n_out],
                        &mut self.buf_out_r[..n_out],
                        start,
                    );
                }
            } else {
                let step = (target - start) / n_out as f32;
                unsafe {
                    crate::math::dsp::gain::apply_ramp_stereo(
                        &mut self.buf_out_l[..n_out],
                        &mut self.buf_out_r[..n_out],
                        start,
                        step,
                    );
                }
                self.smoother_out.set(target);
            }
        }
        #[cfg(not(feature = "stereo"))]
        {
            let start = self.smoother_out.peek();
            let target = self.smoother_out.target_value();
            if (start - target).abs() < 1e-9 {
                crate::math::dsp::gain::apply_gain_simd(&mut self.buf_out_l[..n_out], start);
            } else {
                let step = (target - start) / n_out as f32;
                crate::math::dsp::gain::apply_ramp_simd(&mut self.buf_out_l[..n_out], start, step);
                self.smoother_out.set(target);
            }
        }
    }
}
