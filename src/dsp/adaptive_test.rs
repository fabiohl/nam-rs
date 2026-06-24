// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::common::spsc::{
        RT_STATUS_DEGRADE_MINIMAL, RT_STATUS_DEGRADE_REDUCED, RtStatusFlags,
    };
    use crate::dsp::adaptive::*;
    use std::sync::atomic::Ordering;

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
        adaptive.set_mode(AdaptiveComputeMode::Off, &flags);
        assert_eq!(adaptive.state(), AdaptiveState::Full);
        assert!(!adaptive.is_crossfading());
        assert!(!flags.check_flag(RT_STATUS_DEGRADE_REDUCED));
        assert!(!flags.check_flag(RT_STATUS_DEGRADE_MINIMAL));
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

    #[test]
    fn test_crossfade_prev_state_and_multipliers() {
        let flags = rt_flags();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Conservative);
        let budget = 1000;

        assert_eq!(adaptive.prev_state(), AdaptiveState::Full);
        assert_eq!(adaptive.state(), AdaptiveState::Full);
        assert_eq!(adaptive.current_crossfade_multiplier(), 0.0);

        // Go to Reduced (degrade)
        for _ in 0..3 {
            adaptive.update(above_threshold(budget, 0.71), budget, 48000, &flags);
        }
        assert_eq!(adaptive.prev_state(), AdaptiveState::Full);
        assert_eq!(adaptive.state(), AdaptiveState::Reduced);
        assert!(adaptive.is_crossfading());

        // Check helper function for layer calculation
        let total_layers = 16;
        let old_layers =
            adaptive.wavenet_effective_layers_for_state(adaptive.prev_state(), total_layers);
        let new_layers =
            adaptive.wavenet_effective_layers_for_state(adaptive.state(), total_layers);
        assert_eq!(old_layers, 16); // Full: 0% skipped -> 16 layers
        assert_eq!(new_layers, 12); // Reduced: 25% skipped -> 12 layers

        // Read progress multiplier without advancing clock
        let m0 = adaptive.current_crossfade_multiplier();
        assert_eq!(m0, 0.0);
        let m0_again = adaptive.current_crossfade_multiplier();
        assert_eq!(m0_again, 0.0); // should not change

        // Advance a bit
        adaptive.crossfade_multiplier(48000, 512);
        let m1 = adaptive.current_crossfade_multiplier();
        assert!(m1 > 0.0 && m1 < 1.0);

        // Complete crossfade
        adaptive.crossfade_multiplier(48000, 1500);
        assert!(!adaptive.is_crossfading());
        assert_eq!(adaptive.current_crossfade_multiplier(), 1.0);
    }
}
