// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Adaptive compute: soft-degrade sob CPU pressure (graceful fallback).
//!
//! Monitors DSP block latency and gracefully reduces model complexity
//! when CPU pressure is detected, preventing audible xruns.
//!
//! ## Hysteresis FSM
//!
//! - **Full → Reduced**: 3 consecutive blocks with `latency > 0.70 * budget` (Conservative)
//!   or `latency > 0.55 * budget` (Aggressive).
//! - **Reduced → Minimal**: 3 consecutive blocks with `latency > 0.85 * budget` (Conservative)
//!   or `latency > 0.70 * budget` (Aggressive).
//! - **Reverse path**: 5 consecutive blocks below recovery threshold.
//!
//! ## Crossfade
//!
//! 32 ms linear ramp between states to avoid audible artifacts.

use crate::common::spsc::RtStatusFlags;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

/// Adaptive compute mode — user-facing parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum AdaptiveComputeMode {
    /// Adaptive compute disabled (passthrough).
    #[default]
    Off = 0,
    /// Conservative threshold — late degradation.
    Conservative = 1,
    /// Aggressive threshold — early degradation.
    Aggressive = 2,
}

impl AdaptiveComputeMode {
    /// Creates an `AdaptiveComputeMode` from a floating-point value.
    pub fn from_f32(val: f32) -> Self {
        match val.round() as i32 {
            1 => Self::Conservative,
            2 => Self::Aggressive,
            _ => Self::Off,
        }
    }

    /// Converts the mode to its corresponding floating-point representation.
    pub fn to_f32(self) -> f32 {
        self as u8 as f32
    }
}

/// Degradation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdaptiveState {
    /// Full processing — all model layers active.
    Full = 0,
    /// Reduced processing — fewer layers.
    Reduced = 1,
    /// Minimal processing — maximum reduction.
    Minimal = 2,
}

/// Crossfade phase between degradation stages.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CrossfadePhase {
    /// No crossfade in progress.
    Idle,
    /// Actively crossfading.
    Active { remaining_samples: usize },
}

/// Hysteresis-driven adaptive compute FSM.
///
/// Uses asymmetric hysteresis:
/// - Upgrade (degrade): 3 consecutive blocks above threshold.
/// - Downgrade (recover): 5 consecutive blocks below threshold.
pub struct AdaptiveCompute {
    /// Current degradation state.
    state: AdaptiveState,
    /// Consecutive overload counter for degradation.
    overload_counter: u32,
    /// Consecutive underload counter for recovery.
    recovery_counter: u32,
    /// Crossfade phase.
    crossfade: CrossfadePhase,
    /// Current user mode.
    mode: AdaptiveComputeMode,
    /// Crossfade elapsed samples.
    crossfade_elapsed: usize,
}

const DEGRADE_CONSECUTIVE: u32 = 3;
const RECOVER_CONSECUTIVE: u32 = 5;

const THRESHOLD_FULL_TO_REDUCED_CONSERVATIVE: f32 = 0.70;
const THRESHOLD_REDUCED_TO_MINIMAL_CONSERVATIVE: f32 = 0.85;
const THRESHOLD_FULL_TO_REDUCED_AGGRESSIVE: f32 = 0.55;
const THRESHOLD_REDUCED_TO_MINIMAL_AGGRESSIVE: f32 = 0.70;

/// Crossfade duration in samples at 48 kHz = 32 ms.
/// At other sample rates, this is scaled proportionally.
const CROSSFADE_DURATION_MS: f32 = 32.0;

impl AdaptiveCompute {
    /// Creates a new `AdaptiveCompute` instance with the given mode.
    #[cold]
    pub fn new(mode: AdaptiveComputeMode) -> Self {
        Self {
            state: AdaptiveState::Full,
            overload_counter: 0,
            recovery_counter: 0,
            crossfade: CrossfadePhase::Idle,
            mode,
            crossfade_elapsed: 0,
        }
    }

    /// Sets the user-facing adaptive compute mode.
    pub fn set_mode(&mut self, mode: AdaptiveComputeMode) {
        self.mode = mode;
        if mode == AdaptiveComputeMode::Off {
            self.state = AdaptiveState::Full;
            self.overload_counter = 0;
            self.recovery_counter = 0;
            self.crossfade = CrossfadePhase::Idle;
            self.crossfade_elapsed = 0;
        }
    }

    /// Returns the current adaptive compute mode.
    pub fn mode(&self) -> AdaptiveComputeMode {
        self.mode
    }

    /// Returns the current degradation state of the FSM.
    pub fn state(&self) -> AdaptiveState {
        self.state
    }

    /// Returns the crossfade gain (`crossfade_elapsed / total`).
    /// `sample_rate` is used to compute the crossfade duration in samples.
    /// Returns the linear multiplier applied to the degradation path.
    /// Call this during the output stage.
    #[inline(always)]
    pub fn crossfade_multiplier(&mut self, _sample_rate: u32, frame_advance: usize) -> f32 {
        match self.crossfade {
            CrossfadePhase::Idle => {
                if self.state == AdaptiveState::Full {
                    0.0
                } else {
                    1.0
                }
            }
            CrossfadePhase::Active { remaining_samples } => {
                let total = remaining_samples + self.crossfade_elapsed;
                let progress = if total == 0 {
                    1.0
                } else {
                    (self.crossfade_elapsed as f32 / total as f32).min(1.0)
                };

                if self.crossfade_elapsed + frame_advance >= remaining_samples {
                    self.crossfade = CrossfadePhase::Idle;
                    self.crossfade_elapsed = 0;
                    1.0
                } else {
                    self.crossfade_elapsed += frame_advance;
                    progress
                }
            }
        }
    }

    /// Returns `true` if a crossfade is in progress.
    #[inline(always)]
    pub fn is_crossfading(&self) -> bool {
        !matches!(self.crossfade, CrossfadePhase::Idle)
    }

    /// Feeds the FSM with the latest latency measurement.
    ///
    /// `latency_us` — processing time of this block in microseconds.
    /// `budget_us` — budget for this block in microseconds.
    /// `sample_rate` — sample rate for crossfade duration.
    pub fn update(
        &mut self,
        latency_us: u64,
        budget_us: u64,
        sample_rate: u32,
        rt_status: &RtStatusFlags,
    ) {
        if self.mode == AdaptiveComputeMode::Off || budget_us == 0 {
            return;
        }

        let ratio = latency_us as f32 / budget_us as f32;
        let (full_to_reduced, reduced_to_minimal) = match self.mode {
            AdaptiveComputeMode::Conservative => (
                THRESHOLD_FULL_TO_REDUCED_CONSERVATIVE,
                THRESHOLD_REDUCED_TO_MINIMAL_CONSERVATIVE,
            ),
            AdaptiveComputeMode::Aggressive => (
                THRESHOLD_FULL_TO_REDUCED_AGGRESSIVE,
                THRESHOLD_REDUCED_TO_MINIMAL_AGGRESSIVE,
            ),
            AdaptiveComputeMode::Off => return,
        };

        let recovery_reduced = full_to_reduced * 0.5;
        let recovery_minimal = reduced_to_minimal * 0.5;

        match self.state {
            AdaptiveState::Full => {
                if ratio > full_to_reduced {
                    self.overload_counter = self.overload_counter.saturating_add(1);
                    self.recovery_counter = 0;
                    if self.overload_counter >= DEGRADE_CONSECUTIVE {
                        self.transition_to(AdaptiveState::Reduced, sample_rate, rt_status);
                    }
                } else {
                    self.overload_counter = 0;
                }
            }
            AdaptiveState::Reduced => {
                if ratio > reduced_to_minimal {
                    self.overload_counter = self.overload_counter.saturating_add(1);
                    self.recovery_counter = 0;
                    if self.overload_counter >= DEGRADE_CONSECUTIVE {
                        self.transition_to(AdaptiveState::Minimal, sample_rate, rt_status);
                    }
                } else if ratio < recovery_reduced {
                    self.recovery_counter = self.recovery_counter.saturating_add(1);
                    self.overload_counter = 0;
                    if self.recovery_counter >= RECOVER_CONSECUTIVE {
                        self.transition_to(AdaptiveState::Full, sample_rate, rt_status);
                        self.recovery_counter = 0;
                    }
                } else {
                    self.overload_counter = 0;
                    self.recovery_counter = 0;
                }
            }
            AdaptiveState::Minimal => {
                if ratio < recovery_minimal {
                    self.recovery_counter = self.recovery_counter.saturating_add(1);
                    if self.recovery_counter >= RECOVER_CONSECUTIVE {
                        self.transition_to(AdaptiveState::Reduced, sample_rate, rt_status);
                        self.recovery_counter = 0;
                    }
                } else {
                    self.recovery_counter = 0;
                }
            }
        }
    }

    fn transition_to(
        &mut self,
        new_state: AdaptiveState,
        sample_rate: u32,
        rt_status: &RtStatusFlags,
    ) {
        if self.state == new_state {
            return;
        }

        self.state = new_state;
        self.overload_counter = 0;
        self.recovery_counter = 0;

        let crossfade_samples =
            (CROSSFADE_DURATION_MS / 1000.0 * sample_rate as f32).round() as usize;
        self.crossfade = CrossfadePhase::Active {
            remaining_samples: crossfade_samples.max(1),
        };
        self.crossfade_elapsed = 0;

        rt_status
            .degrade_transitions_total
            .fetch_add(1, Ordering::Relaxed);

        match new_state {
            AdaptiveState::Full => {
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
            }
            AdaptiveState::Reduced => {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
            }
            AdaptiveState::Minimal => {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
                rt_status.set_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
            }
        }
    }

    /// Fraction of WaveNet layers to skip: 0.0 (Full), 0.25 (Reduced), 0.50 (Minimal).
    pub fn wavenet_skip_fraction(&self) -> f32 {
        match self.state {
            AdaptiveState::Full => 0.0,
            AdaptiveState::Reduced => 0.25,
            AdaptiveState::Minimal => 0.50,
        }
    }

    /// Effective number of WaveNet layers after reduction.
    #[inline(always)]
    pub fn wavenet_effective_layers(&self, total_layers: usize) -> usize {
        let skip = (total_layers as f32 * self.wavenet_skip_fraction()).round() as usize;
        total_layers.saturating_sub(skip).max(1)
    }

    /// Effective LSTM layers: 2 (Full), 1 (Reduced), 0 = passthrough (Minimal).
    pub fn lstm_effective_layers(&self, total_layers: usize) -> usize {
        match self.state {
            AdaptiveState::Full => total_layers,
            AdaptiveState::Reduced => total_layers.min(1), // Keep at most 1
            AdaptiveState::Minimal => 0,                   // Passthrough
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::spsc::{RT_STATUS_DEGRADE_MINIMAL, RT_STATUS_DEGRADE_REDUCED};

    fn rt_flags() -> RtStatusFlags {
        RtStatusFlags::default()
    }

    fn above_threshold(budget_us: u64, ratio: f32) -> u64 {
        ((budget_us as f32 * ratio).ceil() as u64) + 1
    }

    fn below_threshold(budget_us: u64, ratio: f32) -> u64 {
        ((budget_us as f32 * ratio).floor() as u64).saturating_sub(1)
    }

    #[test]
    fn mode_off_no_transitions() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);
        let budget = 1000;

        for _ in 0..10 {
            adaptive.update(above_threshold(budget, 0.99), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Full);
        assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn conservative_full_to_reduced() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        // 3 consecutive overloads needed
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Reduced);
        assert!(flags.check_flag(RT_STATUS_DEGRADE_REDUCED));
        assert!(!flags.check_flag(RT_STATUS_DEGRADE_MINIMAL));
        assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn conservative_full_stays_full_if_not_consecutive() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        adaptive.update(below_threshold(budget, 0.69), budget, 48000, &flags); // resets counter
        adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);

        assert_eq!(adaptive.state(), AdaptiveState::Full); // only 2 consecutive after reset
        assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn conservative_reduced_to_minimal() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        // First, go to Reduced
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Reduced);

        // Then to Minimal: need > 0.85 * budget for 3 consecutive blocks
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.86), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Minimal);
        assert!(flags.check_flag(RT_STATUS_DEGRADE_REDUCED));
        assert!(flags.check_flag(RT_STATUS_DEGRADE_MINIMAL));
        assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn conservative_reduced_recovers_to_full() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        // Go to Reduced
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Reduced);

        // Recovery: need 5 consecutive blocks below 0.35 * budget
        for _ in 0..5 {
            adaptive.update(below_threshold(budget, 0.34), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Full);
        assert!(!flags.check_flag(RT_STATUS_DEGRADE_REDUCED));
        assert!(!flags.check_flag(RT_STATUS_DEGRADE_MINIMAL));
        assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn conservative_reduced_recovery_resets_on_intermediate() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        // Go to Reduced
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Reduced);

        // 3 recovery blocks, then 1 block in the dead zone → resets counter
        for _ in 0..3 {
            adaptive.update(below_threshold(budget, 0.34), budget, 48000, &flags);
        }
        // Dead zone: between 0.35 and 0.70 → resets both counters
        adaptive.update(budget / 2, budget, 48000, &flags);
        // Need 5 more recovery blocks
        for _ in 0..5 {
            adaptive.update(below_threshold(budget, 0.34), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Full);
    }

    #[test]
    fn minimal_recovers_to_reduced() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        // Full → Reduced
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        // Reduced → Minimal
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.86), budget, 48000, &flags);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Minimal);
        assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 2);

        // Recovery Minimal → Reduced: 5 blocks below 0.425 * budget
        for _ in 0..5 {
            adaptive.update(below_threshold(budget, 0.42), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Reduced);
        assert!(flags.check_flag(RT_STATUS_DEGRADE_REDUCED));
        assert!(!flags.check_flag(RT_STATUS_DEGRADE_MINIMAL));
        assert_eq!(flags.degrade_transitions_total.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn aggressive_full_to_reduced_lower_threshold() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Aggressive);
        let budget = 1000;

        // Aggressive: Full→Reduced at > 0.55
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.56), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Reduced);
    }

    #[test]
    fn aggressive_reduced_to_minimal() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Aggressive);
        let budget = 1000;

        // Full → Reduced at > 0.55
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.56), budget, 48000, &flags);
        }
        // Reduced → Minimal at > 0.70
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }

        assert_eq!(adaptive.state(), AdaptiveState::Minimal);
    }

    #[test]
    fn set_mode_resets_state() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        // Degrade to Minimal
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.86), budget, 48000, &flags);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Minimal);

        // Set mode to Off → resets
        adaptive.set_mode(AdaptiveComputeMode::Off);
        assert_eq!(adaptive.state(), AdaptiveState::Full);
        assert!(!adaptive.is_crossfading());
    }

    #[test]
    fn wavenet_effective_layers_full() {
        let adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        assert_eq!(adaptive.wavenet_effective_layers(8), 8);
        assert_eq!(adaptive.wavenet_effective_layers(4), 4);
    }

    #[test]
    fn wavenet_effective_layers_reduced() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Reduced);

        // 25% reduction: 8 layers → 6
        assert_eq!(adaptive.wavenet_effective_layers(8), 6);
        // 25% reduction: 4 layers → 3
        assert_eq!(adaptive.wavenet_effective_layers(4), 3);
        // Minimum 1 layer
        assert_eq!(adaptive.wavenet_effective_layers(1), 1);
    }

    #[test]
    fn wavenet_effective_layers_minimal() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.86), budget, 48000, &flags);
        }
        assert_eq!(adaptive.state(), AdaptiveState::Minimal);

        // 50% reduction: 8 → 4
        assert_eq!(adaptive.wavenet_effective_layers(8), 4);
        // Minimum 1
        assert_eq!(adaptive.wavenet_effective_layers(1), 1);
    }

    #[test]
    fn lstm_effective_layers() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        // Full: 2 layers
        assert_eq!(adaptive.lstm_effective_layers(2), 2);
        assert_eq!(adaptive.lstm_effective_layers(1), 1);

        // Reduced: at most 1
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        assert_eq!(adaptive.lstm_effective_layers(2), 1);
        assert_eq!(adaptive.lstm_effective_layers(1), 1);

        // Minimal: passthrough
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.86), budget, 48000, &flags);
        }
        assert_eq!(adaptive.lstm_effective_layers(2), 0);
    }

    #[test]
    fn crossfade_starts_on_transition() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        assert!(!adaptive.is_crossfading());

        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }

        assert!(adaptive.is_crossfading());
    }

    #[test]
    fn crossfade_completes() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        assert!(adaptive.is_crossfading());

        // Crossfade: 32ms * 48kHz = 1536 samples
        // Advance in 64-sample frames
        for _ in 0..25 {
            adaptive.crossfade_multiplier(48000, 64);
        }

        assert!(!adaptive.is_crossfading());
    }
}
