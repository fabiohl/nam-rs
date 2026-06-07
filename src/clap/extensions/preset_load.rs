// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLAP `clap.preset-load` extension implementation.
//!
//! Enables hosts to load NAM model files as presets via the host's preset browser.

use clack_extensions::preset_discovery::prelude::*;
use clack_plugin::prelude::*;
use std::ffi::{CStr, CString};
use std::path::PathBuf;

use crate::clap::plugin::NamClapMainThread;

impl PluginPresetLoadImpl for NamClapMainThread<'_> {
    fn load_from_location(
        &mut self,
        location: Location,
        _load_key: Option<&CStr>,
    ) -> Result<(), PluginError> {
        let path = match location {
            Location::File { path } => path,
            Location::Plugin => {
                return Err(PluginError::Message(
                    "Cannot load NAM model from plugin container",
                ));
            }
        };

        let path_str = path
            .to_str()
            .map_err(|_| PluginError::Message("Invalid UTF-8 in model path"))?;
        let path_buf = PathBuf::from(path_str);

        // Enqueue the model for loading via the existing pipeline (same as GUI/drag-drop).
        if let Ok(mut pending_guard) = self.shared.cold.ui_pending_model.lock() {
            *pending_guard = Some(path_buf);
            self.shared
                .cold
                .ui_loading
                .store(true, std::sync::atomic::Ordering::Relaxed);
        } else {
            return Err(PluginError::Message(
                "Failed to acquire ui_pending_model lock",
            ));
        }

        self.host.shared().request_callback();

        // Log the preset load request.
        if let Some(log) = self.host.get_extension::<clack_extensions::log::HostLog>() {
            let msg = CString::new(format!("NAM-rs: Loading preset from {:?}", path_str))
                .unwrap_or_default();
            log.log(
                &self.host.shared(),
                clack_extensions::log::LogSeverity::Info,
                &msg,
            );
        }

        Ok(())
    }
}
