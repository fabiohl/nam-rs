// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Identity descriptor of the NAM-rs plugin in CLAP format.

use clack_plugin::prelude::*;

/// Returns the immutable plugin descriptor.
/// Read by the host during scan — must be deterministic and without allocations.
pub fn nam_descriptor() -> PluginDescriptor {
    PluginDescriptor::new("br.eti.fabiolima.nam-rs", "NAM-rs")
        .with_vendor("Fabio Lima")
        .with_url("https://github.com/fabiohl/nam-rs")
        .with_description("Real-time Neural Amp Modeler plugin (CLAP)")
        .with_features([
            c"audio-effect",
            c"distortion",
            c"gate",
            c"simulator",
            c"mono",
        ])
}
