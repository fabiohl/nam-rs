// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLAP extensions implemented by NAM-rs.

pub mod audio_ports;
pub mod audio_ports_activation;
pub mod latency;
pub mod param_indication;
pub mod params;
pub mod preset_load;
pub mod remote_controls;
pub mod render;
pub mod state;
pub mod state_context;
pub(crate) mod state_transaction;
pub mod tail;
pub mod track_info;

#[cfg(feature = "clap-plugin")]
pub mod gui;
