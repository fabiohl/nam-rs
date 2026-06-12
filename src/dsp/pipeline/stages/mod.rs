// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP pipeline stages (1–4): Input → Inference → Output → Bridge.
//!
//! All functions in this module follow the RT callback rules:
//! - Zero heap allocation
//! - Zero I/O
//! - Zero mutexes

mod bridge;
mod inference;
mod input;
mod output;

pub use bridge::write_bridge;
pub(crate) use inference::run_inference;
pub(crate) use input::DENORMAL_DITHER_OFFSET;
pub(crate) use input::apply_input_stage;
pub use input::{DISABLE_GATE, handle_silence_bypass};
pub(crate) use output::apply_output_stage;
