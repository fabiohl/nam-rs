// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kahan compensated summation (Kahan–Babuška algorithm) for f32 accumulators.
//!
//! Kahan summation reduces the floating-point accumulation error from O(N·ε)
//! (linear growth with the number of terms) to O(ε) (bounded), at the cost of
//! 2 extra f32 operations per addition.
//!
//! # When to use
//! - Loops with > 32 sequential f32 adds into the same accumulator
//! - High-channel-count convolution accumulators (IN ≥ 64)
//! - Deep layer stacks where drift multiplies across 10+ layers
//!
//! # When NOT to use
//! - SIMD inner FMA loops (dependency chains break ILP)
//! - Single-digit additions (K ≤ 3 taps)
//!
//! # Reference
//! Kahan, W. "Further remarks on reducing truncation errors."
//! Communications of the ACM, 1965.

/// Kahan compensated accumulator for a single f32 value.
///
/// Tracks the sum and the lost low-order bits (compensation) so that
/// the total accumulated error stays bounded.
#[derive(Clone, Copy, Debug, Default)]
pub struct KahanF32 {
    /// Accumulated sum.
    pub sum: f32,
    /// Lost low-order bits from previous additions.
    compensation: f32,
}

impl KahanF32 {
    /// Creates a new accumulator initialized to a given starting value.
    #[inline(always)]
    pub fn new(initial: f32) -> Self {
        Self {
            sum: initial,
            compensation: 0.0,
        }
    }

    /// Adds `x` to the accumulator using Kahan compensated summation.
    ///
    /// Algorithm:
    ///   y = x - c          // compensate for previous error
    ///   t = s + y          // tentative sum
    ///   c = (t - s) - y    // recover low-order bits lost in t
    ///   s = t
    #[inline(always)]
    pub fn add(&mut self, x: f32) {
        let y = x - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Returns the accumulated sum.
    #[inline(always)]
    pub fn value(self) -> f32 {
        self.sum
    }
}

/// Kahan compensated accumulator for 4 independent f32 channels (SoA layout).
///
/// Used when accumulating 4 output channels simultaneously, as in
/// the interleaved convolution kernel.
#[derive(Clone, Copy, Debug)]
pub struct Kahan4F32 {
    /// Accumulated sums for each of the 4 channels.
    pub sum: [f32; 4],
    /// Lost low-order bits per channel.
    compensation: [f32; 4],
}

impl Kahan4F32 {
    /// Creates a 4-channel accumulator initialized to the given starting values.
    #[inline(always)]
    pub fn new(initial: [f32; 4]) -> Self {
        Self {
            sum: initial,
            compensation: [0.0; 4],
        }
    }

    /// Adds `x[i]` to channel `i` using Kahan compensated summation.
    #[inline(always)]
    pub fn add(&mut self, x: [f32; 4]) {
        for (ch, &x_ch) in x.iter().enumerate() {
            let y = x_ch - self.compensation[ch];
            let t = self.sum[ch] + y;
            self.compensation[ch] = (t - self.sum[ch]) - y;
            self.sum[ch] = t;
        }
    }

    /// Adds `x` to a single channel `ch` using Kahan compensated summation.
    #[inline(always)]
    pub fn add_channel(&mut self, ch: usize, x: f32) {
        let y = x - self.compensation[ch];
        let t = self.sum[ch] + y;
        self.compensation[ch] = (t - self.sum[ch]) - y;
        self.sum[ch] = t;
    }

    /// Returns the accumulated sum for all 4 channels.
    #[inline(always)]
    pub fn value(self) -> [f32; 4] {
        self.sum
    }
}

/// Inline Kahan addition for use in tight loops where struct overhead is undesirable.
///
/// Returns the updated `(sum, compensation)` tuple.
#[inline(always)]
pub fn kahan_add(sum: f32, compensation: f32, x: f32) -> (f32, f32) {
    let y = x - compensation;
    let t = sum + y;
    (t, (t - sum) - y)
}

#[cfg(test)]
#[path = "kahan_test.rs"]
mod kahan_test;
