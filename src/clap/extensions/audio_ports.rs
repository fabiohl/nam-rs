// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão de portas de áudio CLAP para o NAM-rs.

use crate::clap::plugin::NamClapMainThread;
use clack_extensions::audio_ports::{
    AudioPortFlags, AudioPortInfo, AudioPortInfoWriter, AudioPortType, PluginAudioPortsImpl,
};
use clack_plugin::prelude::ClapId;

impl PluginAudioPortsImpl for NamClapMainThread<'_> {
    /// Retorna o número de portas de áudio (entrada ou saída).
    ///
    /// O NAM-rs é um plugin stereo simples com exatamente 1 porta de entrada e 1 de saída.
    fn count(&mut self, _is_input: bool) -> u32 {
        1
    }

    /// Preenche as informações da porta de áudio no índice especificado.
    ///
    /// Configura uma porta stereo (2 canais) com in-place pair habilitado
    /// (permite que o host use o mesmo buffer para entrada e saída).
    fn get(&mut self, index: u32, is_input: bool, writer: &mut AudioPortInfoWriter) {
        if index == 0 {
            writer.set(&AudioPortInfo {
                id: ClapId::new(0),
                name: if is_input {
                    b"Main Input"
                } else {
                    b"Main Output"
                },
                channel_count: 2,
                flags: AudioPortFlags::IS_MAIN,
                port_type: Some(AudioPortType::STEREO),
                in_place_pair: Some(ClapId::new(0)),
            });
        }
    }
}
