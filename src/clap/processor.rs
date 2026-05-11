// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Processador de áudio CLAP — Sprint 1: bypass puro (input → output).

use crate::clap::plugin::{NamClapMainThread, NamClapShared};
use clack_plugin::prelude::*;

/// Processador de áudio RT-safe. Executa na audio thread do host.
/// Sprint 1: bypass — copia cada amostra de input para output sem processar.
pub struct NamClapProcessor;

impl<'a> PluginAudioProcessor<'a, NamClapShared, NamClapMainThread> for NamClapProcessor {
    fn activate(
        _host: HostAudioProcessorHandle<'a>,
        _main_thread: &mut NamClapMainThread,
        _shared: &'a NamClapShared,
        _audio_config: PluginAudioConfiguration,
    ) -> Result<Self, PluginError> {
        Ok(Self)
    }

    fn process(
        &mut self,
        _process: Process,
        mut audio: Audio,
        _events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        for mut port_pair in &mut audio {
            let Some(channel_pairs) = port_pair.channels()?.into_f32() else {
                continue;
            };
            for channel_pair in channel_pairs {
                match channel_pair {
                    ChannelPair::InputOnly(_) => {}
                    ChannelPair::OutputOnly(buf) => buf.fill(0.0),
                    ChannelPair::InputOutput(input, output) => {
                        output.copy_from_slice(input);
                    }
                    ChannelPair::InPlace(_) => {} // in-place: nada a fazer no bypass
                }
            }
        }
        Ok(ProcessStatus::Continue)
    }
}
