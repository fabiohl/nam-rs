// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Descriptor de identidade do plugin NAM-rs no formato CLAP.

use clack_plugin::prelude::*;

/// Retorna o descritor imutável do plugin.
/// Lido pelo host durante scan — deve ser determinístico e sem alocações.
pub fn nam_descriptor() -> PluginDescriptor {
    PluginDescriptor::new("br.eti.fabiolima.nam-rs", "NAM-rs Neural Amp Modeler")
        .with_vendor("Fabio Lima")
        .with_url("https://github.com/fabiohl/nam-rs")
        .with_description("Real-time Neural Amp Modeler plugin (CLAP)")
        .with_features([
            c"audio-effect",
            c"distortion",
            c"gate",
            c"simulator",
            c"stereo",
        ])
}
