// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Extensões CLAP implementadas pelo NAM-rs.

pub mod audio_ports;
pub mod latency;
pub mod param_indication;
pub mod params;
pub mod remote_controls;
pub mod state;
pub mod track_info;

#[cfg(feature = "clap-plugin-gui")]
pub mod gui;
