// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementation of the CLAP render extension.
//!
//! NAM is deterministic and causal, so it has no hard realtime requirement.
//! In offline mode (bounce/export), `AdaptiveCompute` is forced to `Off`
//! so the model always processes at maximum quality without soft-degrade.
//!
//! See: `clap/ext/render.h`

use crate::clap::plugin::{NamClapMainThread, RENDER_MODE_OFFLINE, RENDER_MODE_REALTIME};
use clack_extensions::render::{PluginRender, PluginRenderImpl, RenderMode};
use clack_plugin::prelude::*;
use std::sync::atomic::Ordering;

/// Type alias for the CLAP render extension registration.
pub type NamPluginRender = PluginRender;

impl<'a> PluginRenderImpl for NamClapMainThread<'a> {
    fn has_hard_realtime_requirement(&self) -> bool {
        false
    }

    fn set(&mut self, mode: RenderMode) -> Result<(), PluginError> {
        let val = match mode {
            RenderMode::Realtime => RENDER_MODE_REALTIME,
            RenderMode::Offline => RENDER_MODE_OFFLINE,
        };
        self.shared.cold.render_mode.store(val, Ordering::Release);
        Ok(())
    }
}
