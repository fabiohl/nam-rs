// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration of NAM-rs as a plugin in CLAP (CLever Audio Plug-in) format.
//!
//! Activated via the `clap-plugin` feature flag. Completely isolated from PipeWire.

pub mod descriptor;
pub mod entry;
pub mod extensions;
pub mod factory;
#[cfg(feature = "heap-audit")]
pub mod heap_audit;
pub mod param_smoother;
pub mod plugin;
pub mod processor;

#[cfg(feature = "clap-plugin")]
pub mod gui;

pub use plugin::NamClapPlugin;

use clack_plugin::clack_export_entry;

clack_export_entry!(entry::NamEntry);

#[cfg(test)]
#[path = "preset_discovery_test.rs"]
mod preset_discovery_test;
