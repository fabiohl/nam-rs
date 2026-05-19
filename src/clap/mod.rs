// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integração do NAM-rs como plugin no formato CLAP (CLever Audio Plug-in).
//!
//! Ativado via feature flag `clap-plugin`. Totalmente isolado do PipeWire.

pub mod descriptor;
pub mod extensions;
pub mod param_smoother;
pub mod plugin;
pub mod processor;

/// Módulo contendo a implementação da interface gráfica (GUI).
#[cfg(feature = "clap-plugin-gui")]
pub mod gui;

use clack_plugin::prelude::*;
use plugin::NamClapPlugin;

clack_export_entry!(SinglePluginEntry<NamClapPlugin>);
