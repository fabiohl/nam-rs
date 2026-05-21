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
                    let packed = pack_argb(color.alpha, color.red, color.green, color.blue);
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

/// Empacota componentes de cor ARGB em um `u32` para armazenamento atômico.
///
/// Formato: `(alpha << 24) | (red << 16) | (green << 8) | blue`.
/// O valor zero (`pack_argb(0, 0, 0, 0)`) é tratado como sentinela de "sem cor".
/// A função `resolve_accent()` na GUI usa `alpha == 0` como fallback para `COL_ACCENT`.
pub fn pack_argb(alpha: u8, red: u8, green: u8, blue: u8) -> u32 {
    ((alpha as u32) << 24) | ((red as u32) << 16) | ((green as u32) << 8) | (blue as u32)
}

/// Tipo marcador para registro da extensão.
pub type NamPluginTrackInfo = PluginTrackInfo;

#[cfg(test)]
mod tests {
    use super::*;

    /// Valida o empacotamento ARGB para as 5 cores representativas exigidas pela spec A.1.
    #[test]
    fn test_pack_argb_representative_colors() {
        // Branco opaco: ARGB = (FF, FF, FF, FF)
        assert_eq!(pack_argb(0xFF, 0xFF, 0xFF, 0xFF), 0xFFFF_FFFF);

        // Preto opaco: ARGB = (FF, 00, 00, 00)
        assert_eq!(pack_argb(0xFF, 0x00, 0x00, 0x00), 0xFF00_0000);

        // Vermelho puro: ARGB = (FF, FF, 00, 00)
        assert_eq!(pack_argb(0xFF, 0xFF, 0x00, 0x00), 0xFFFF_0000);

        // Verde puro: ARGB = (FF, 00, FF, 00)
        assert_eq!(pack_argb(0xFF, 0x00, 0xFF, 0x00), 0xFF00_FF00);

        // Azul Bitwig #5e81ac: ARGB = (FF, 5E, 81, AC)
        assert_eq!(pack_argb(0xFF, 0x5E, 0x81, 0xAC), 0xFF5E_81AC);
    }

    /// Valida que alpha == 0 (sentinela) resulta em packed == 0 quando RGB também é zero.
    #[test]
    fn test_pack_argb_zero_is_sentinel() {
        assert_eq!(pack_argb(0, 0, 0, 0), 0);
    }

    /// Valida consistência bidirecional: pack e depois unpack deve recuperar os componentes.
    #[test]
    fn test_pack_argb_roundtrip() {
        let colors = [
            (0xFFu8, 0x00u8, 0xD4u8, 0xAAu8), // Turquesa #00D4AA (fallback)
            (0xFF, 0x5E, 0x81, 0xAC),         // Azul Bitwig
            (0xFF, 0xF5, 0xA6, 0x23),         // Amber
        ];
        for (alpha, red, green, blue) in colors {
            let packed = pack_argb(alpha, red, green, blue);
            assert_eq!((packed >> 24) as u8, alpha, "alpha mismatch");
            assert_eq!(((packed >> 16) & 0xFF) as u8, red, "red mismatch");
            assert_eq!(((packed >> 8) & 0xFF) as u8, green, "green mismatch");
            assert_eq!((packed & 0xFF) as u8, blue, "blue mismatch");
        }
    }
}
