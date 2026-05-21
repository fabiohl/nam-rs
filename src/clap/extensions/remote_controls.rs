// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão `clap_plugin_remote_controls` para o NAM-rs.

use crate::clap::extensions::params::{
    PARAM_BYPASS, PARAM_GATE_THRESH, PARAM_INPUT_GAIN, PARAM_OUTPUT_GAIN,
};
use crate::clap::plugin::NamClapMainThread;
use clack_common::utils::ClapId;
use clack_extensions::remote_controls::{
    PluginRemoteControls, PluginRemoteControlsImpl, RemoteControlsPage, RemoteControlsPageWriter,
};

/// Retorna a quantidade de páginas de controle remoto de hardware.
pub fn get_page_count() -> u32 {
    2
}

/// Preenche a página de controle remoto especificada pelo índice.
pub fn fill_remote_controls_page(index: u32, writer: &mut RemoteControlsPageWriter) {
    match index {
        0 => {
            let mut param_ids = [None; 8];
            param_ids[0] = Some(ClapId::new(PARAM_INPUT_GAIN));
            param_ids[1] = Some(ClapId::new(PARAM_OUTPUT_GAIN));
            param_ids[2] = Some(ClapId::new(PARAM_BYPASS));

            let page = RemoteControlsPage {
                section_name: b"NAM-rs",
                page_id: ClapId::new(0),
                page_name: b"Main",
                param_ids,
                is_for_preset: false,
            };
            writer.set(&page);
        }
        1 => {
            let mut param_ids = [None; 8];
            param_ids[0] = Some(ClapId::new(PARAM_GATE_THRESH));

            let page = RemoteControlsPage {
                section_name: b"NAM-rs",
                page_id: ClapId::new(1),
                page_name: b"Gate",
                param_ids,
                is_for_preset: false,
            };
            writer.set(&page);
        }
        _ => {}
    }
}

impl<'a> PluginRemoteControlsImpl for NamClapMainThread<'a> {
    fn count(&mut self) -> u32 {
        get_page_count()
    }

    fn get(&mut self, index: u32, writer: &mut RemoteControlsPageWriter) {
        fill_remote_controls_page(index, writer);
    }
}

/// Tipo marcador para registro da extensão.
pub type NamPluginRemoteControls = PluginRemoteControls;

#[cfg(test)]
mod tests {
    use super::*;
    use clap_sys::ext::remote_controls::clap_remote_controls_page;
    use std::mem::MaybeUninit;

    #[test]
    fn test_page_count() {
        assert_eq!(get_page_count(), 2);
    }

    #[test]
    fn test_fill_pages() {
        // Página 0: Main
        let mut raw_page = MaybeUninit::<clap_remote_controls_page>::zeroed();
        unsafe {
            let mut writer = RemoteControlsPageWriter::from_raw(raw_page.as_mut_ptr());
            fill_remote_controls_page(0, &mut writer);
            let raw_ref = raw_page.assume_init_ref();
            let page = RemoteControlsPage::from_raw(raw_ref).unwrap();

            assert_eq!(page.section_name, b"NAM-rs");
            assert_eq!(page.page_name, b"Main");
            assert_eq!(page.page_id.get(), 0);
            assert_eq!(page.param_ids[0].map(|id| id.get()), Some(PARAM_INPUT_GAIN));
            assert_eq!(page.param_ids[1].map(|id| id.get()), Some(PARAM_OUTPUT_GAIN));
            assert_eq!(page.param_ids[2].map(|id| id.get()), Some(PARAM_BYPASS));
            for i in 3..8 {
                assert!(page.param_ids[i].is_none());
            }
        }

        // Página 1: Gate
        let mut raw_page = MaybeUninit::<clap_remote_controls_page>::zeroed();
        unsafe {
            let mut writer = RemoteControlsPageWriter::from_raw(raw_page.as_mut_ptr());
            fill_remote_controls_page(1, &mut writer);
            let raw_ref = raw_page.assume_init_ref();
            let page = RemoteControlsPage::from_raw(raw_ref).unwrap();

            assert_eq!(page.section_name, b"NAM-rs");
            assert_eq!(page.page_name, b"Gate");
            assert_eq!(page.page_id.get(), 1);
            assert_eq!(page.param_ids[0].map(|id| id.get()), Some(PARAM_GATE_THRESH));
            for i in 1..8 {
                assert!(page.param_ids[i].is_none());
            }
        }
    }
}
