// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Digital Signal Processing (DSP) module for generic operations
//! before and after the neural engine in NAM-rs.

pub mod adaptive;
pub mod gate;
pub mod gate_flags;
pub mod mirror_buf;
pub mod pipeline;
pub mod resampler;
pub mod sinc_kernel;
pub mod smoother;
pub mod telemetry;
