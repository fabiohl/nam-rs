// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

use nam_rs::testing::stress::generate_stress_signal_v1;

#[deprecated(since = "1.5.0", note = "Use `generate_stress_signal_v1()` directly")]
pub fn generate_stress_signal() -> Vec<f32> {
    generate_stress_signal_v1()
}

pub fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect()
}
