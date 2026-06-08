// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the CLAP parameters extension for NAM-rs.

pub mod audio;
pub mod main;

/// Input gain parameter ID.
pub const PARAM_INPUT_GAIN: u32 = 0;
/// Output gain parameter ID.
pub const PARAM_OUTPUT_GAIN: u32 = 1;
/// Noise gate threshold parameter ID.
pub const PARAM_GATE_THRESH: u32 = 2;
/// Plugin bypass parameter ID.
pub const PARAM_BYPASS: u32 = 3;
/// Loaded model name parameter ID (read-only).
pub const PARAM_ACTIVE_MODEL: u32 = 4;
/// Adaptive compute mode parameter ID.
pub const PARAM_ADAPTIVE_COMPUTE: u32 = 5;
