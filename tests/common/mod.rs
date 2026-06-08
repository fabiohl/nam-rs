// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared helper module for NAM-rs integration tests.
//!
//! Centralizes signal generation functions, error metrics, and DSP validation
//! to avoid duplication across integration test files.

#![allow(dead_code)]
#![allow(unused_imports)]

pub mod constants;
pub mod io_helpers;
pub mod metrics;
pub mod mushra_primitives;
pub mod perceptual;
pub mod signals;
pub mod validation;
pub mod wav;

pub use constants::*;
pub use io_helpers::*;
pub use metrics::*;
pub use nam_rs::testing::stress::SUPPORTED_SAMPLE_RATES;
pub use nam_rs::testing::stress::generate_stress_signal_v1;
pub use nam_rs::testing::stress::generate_stress_signal_v2_default as generate_stress_signal_v2;
pub use signals::*;
pub use validation::*;
