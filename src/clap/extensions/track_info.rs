// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão `clap_plugin_track_info` para o NAM-rs.

use crate::clap::plugin::NamClapMainThread;
use clack_extensions::track_info::{PluginTrackInfo, PluginTrackInfoImpl};
use std::sync::atomic::Ordering;

impl<'a> PluginTrackInfoImpl for NamClapMainThread<'a> {
    fn changed(&self) {
        if let Some(track_info_ext) = self
            .host
            .get_extension::<clack_extensions::track_info::HostTrackInfo>()
        {
            let mut buffer = clack_extensions::track_info::TrackInfoBuffer::new();
            // SAFETY: A thread principal do CLAP executa este callback de forma síncrona.
            // É seguro criar um HostMainThreadHandle temporário pois estamos na thread principal.
            let mut host_mut = unsafe { self.host.shared().as_main_thread_unchecked() };
            if let Some(info) = track_info_ext.get(&mut host_mut, &mut buffer) {
                if let Some(color) = info.color() {
                    let packed = ((color.alpha as u32) << 24)
                        | ((color.red as u32) << 16)
                        | ((color.green as u32) << 8)
                        | (color.blue as u32);
                    self.shared
                        .track_accent_color
                        .store(packed, Ordering::Relaxed);
                } else {
                    self.shared.track_accent_color.store(0, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Tipo marcador para registro da extensão.
pub type NamPluginTrackInfo = PluginTrackInfo;
