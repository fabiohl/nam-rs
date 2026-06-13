// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Smoothing filter for audio parameters.
//!
//! Implements a 1-pole IIR filter (Low-pass) to avoid clicks and zipper noise
//! when changing gains during real-time processing.

/// Parameter smoother based on a 1-pole IIR filter.
/// y[n] = α * target + (1 - α) * y[n-1]
#[derive(Debug, Clone, Copy)]
pub struct ParamSmoother {
    current: f32,
    target: f32,
    alpha: f32,
}

impl ParamSmoother {
    /// Creates a new smoother with initial value and alpha coefficient.
    ///
    /// # Parameters
    /// * `initial_value`: Initial value (and initial target).
    /// * `sample_rate`: Sampling rate (fs).
    /// * `cutoff_hz`: Cutoff frequency (fc). Recommended ~20Hz for gains.
    pub fn new(initial_value: f32, sample_rate: f32, cutoff_hz: f32) -> Self {
        let alpha = if sample_rate > 0.0 {
            // α = 1 - exp(-2π * fc / fs)
            1.0 - (-(2.0 * std::f32::consts::PI * cutoff_hz) / sample_rate).exp()
        } else {
            1.0
        };

        Self {
            current: initial_value,
            target: initial_value,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Updates the target value of the parameter.
    #[inline]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Jumps immediately to the target value (without smoothing).
    #[inline]
    pub fn snap_to_target(&mut self) {
        self.current = self.target;
    }

    /// Advances one sample and returns the smoothed value.
    ///
    /// Called per-sample in the output gain smoothing path.
    /// Micro-opt [T18.5d]: `#[inline]` eliminates the function-call
    /// overhead for this hot-path 1-pole IIR tick.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let diff = self.current - self.target;
        // Threshold proportional to the target: 2-5x faster convergence for higher values.
        let threshold = 1e-6 * self.target.abs().max(1.0);
        if diff.abs() < threshold {
            self.current = self.target;
        } else {
            let next = self.alpha * self.target + (1.0 - self.alpha) * self.current;
            if next == self.current {
                // Precision stall detection in f32: if the step is smaller than
                // the smallest representable variation, forces a snap to the target.
                self.current = self.target;
            } else {
                self.current = next;
                // Fade-to-zero guard (RT-Safety §2.1).
                //
                // With DAZ/FTZ active in MXCSR (set at boot and periodically
                // reaffirmed in the CLAP processor), actual f32 subnormals
                // (abs < ~1.18e-38) are never created — FPU hardware flushes
                // them to zero automatically.  Therefore this check is not
                // about denormal protection (which DAZ/FTZ already provides).
                //
                // The threshold 1e-15 is ~17 orders of magnitude above the
                // subnormal boundary.  It serves a *sonic* purpose: kill the
                // inaudible tail of the smoother to prevent a theoretically
                // infinite decay when values get so small they no longer
                // produce any audible output (< -300 dBFS).
                if self.current.abs() < 1e-15 {
                    self.current = 0.0;
                }
            }
        }
        self.current
    }

    /// Returns the current value (last computed).
    #[inline]
    pub fn current_value(&self) -> f32 {
        self.current
    }

    /// Returns the target value.
    #[inline]
    pub fn target_value(&self) -> f32 {
        self.target
    }

    /// Returns the current value (peek).
    #[inline]
    pub fn peek(&self) -> f32 {
        self.current
    }

    /// Sets the current value.
    #[inline]
    pub fn set(&mut self, val: f32) {
        self.current = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoother_convergence() {
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
        smoother.set_target(1.0);

        // Should converge gradually
        let mut last_val = 0.0;
        for _ in 0..1000 {
            let current = smoother.tick();
            assert!(current >= last_val);
            last_val = current;
        }

        assert!(last_val > 0.5); // At 1000 samples @ 48k with 20Hz (~20ms), should already be pretty far along
    }

    #[test]
    fn test_smoother_snap() {
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
        smoother.set_target(1.0);
        smoother.snap_to_target();
        assert_eq!(smoother.tick(), 1.0);
    }

    #[test]
    fn test_smoother_convergence_high_gain() {
        // Verify that for target = 3.98 (≈ +12dB), the smoother converges within ≤ 2400 samples at 48kHz (50ms).
        // Note: The 45Hz cutoff perfectly illustrates the benefit of the relative threshold,
        // since with a fixed threshold (1e-6) convergence would take 2581 samples (exceeding 2400),
        // while the relative threshold allows convergence in 2347 samples.
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 45.0);
        smoother.set_target(3.98);

        let mut samples = 0;
        for _ in 0..5000 {
            let current = smoother.tick();
            samples += 1;
            if current == 3.98 {
                break;
            }
        }
        assert!(
            samples <= 2400,
            "Convergence took {} samples (expected <= 2400)",
            samples
        );
    }

    #[test]
    fn test_smoother_denormal_prevention() {
        // Verify that for target = 0.0 and initial = 1e-20, tick() returns exactly 0.0 after ≤ 10 iterations.
        let mut smoother = ParamSmoother::new(1e-20, 48000.0, 20.0);
        smoother.set_target(0.0);

        let mut converged = false;
        for _ in 0..10 {
            if smoother.tick() == 0.0 {
                converged = true;
                break;
            }
        }
        assert!(converged, "Did not converge to 0.0 in 10 iterations");
    }

    #[test]
    fn test_smoother_relative_threshold() {
        // Verify that target = 0.001 still converges correctly (no premature snap).
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
        smoother.set_target(0.001);

        // The first tick should not hit 0.001 immediately (premature snap).
        let val1 = smoother.tick();
        assert!(val1 > 0.0);
        assert!(val1 < 0.001);

        // Should eventually converge
        let mut converged = false;
        for _ in 0..5000 {
            if smoother.tick() == 0.001 {
                converged = true;
                break;
            }
        }
        assert!(converged, "Should converge to 0.001");
    }
}
