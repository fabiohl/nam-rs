// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration of NAM-rs as a plugin in CLAP (CLever Audio Plug-in) format.
//!
//! Activated via the `clap-plugin` feature flag. Completely isolated from PipeWire.

pub mod descriptor;
pub mod extensions;
#[cfg(feature = "heap-audit")]
pub mod heap_audit;
pub mod param_smoother;
pub mod plugin;
pub mod processor;

/// Module containing the graphical user interface (GUI) implementation.
#[cfg(feature = "clap-plugin")]
pub mod gui;

use clack_plugin::prelude::*;
use plugin::NamClapPlugin;

clack_export_entry!(SinglePluginEntry<NamClapPlugin>);
