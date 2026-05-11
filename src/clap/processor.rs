// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Processador de áudio CLAP — Stub para permitir compilação da Tarefa 1.2.2.

use crate::clap::plugin::{NamClapMainThread, NamClapShared};
use clack_plugin::prelude::*;

/// Processador de áudio RT-safe. Executa na audio thread do host.
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
        _audio: Audio,
        _events: Events,
    ) -> Result<ProcessStatus, PluginError> {
        Ok(ProcessStatus::Continue)
    }
}
