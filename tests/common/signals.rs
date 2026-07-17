// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

use nam_rs::testing::stress::generate_stress_signal_v1;

#[deprecated(since = "1.5.0", note = "Use `generate_stress_signal_v1()` directly")]
pub fn generate_stress_signal() -> Vec<f32> {
    generate_stress_signal_v1()
}

pub use nam_rs::testing::aliasing::generate_sine_440hz;
