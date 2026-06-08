// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP block proper: channel extraction, gate, inference,
//! resampling, output gain, peaks, and telemetry.

pub mod bypass;
pub mod channels;
pub mod gain;
pub mod gate_flags;
pub mod orchestrator;
pub mod peaks;
pub mod telemetry;
