// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Definição do plugin NAM-rs e seus componentes de ciclo de vida CLAP.

use crate::clap::descriptor::nam_descriptor;
use crate::clap::processor::NamClapProcessor;
use clack_plugin::prelude::*;

/// Estado compartilhado entre a audio thread e a main thread (lock-free).
pub struct NamClapShared;

impl<'a> PluginShared<'a> for NamClapShared {}

/// Estado exclusivo da main thread (carregamento de modelos, state save/load).
pub struct NamClapMainThread;

impl<'a> PluginMainThread<'a, NamClapShared> for NamClapMainThread {}

/// Plugin NAM-rs: ponto de entrada principal do ciclo de vida CLAP.
pub struct NamClapPlugin;

impl Plugin for NamClapPlugin {
    type AudioProcessor<'a> = NamClapProcessor;
    type Shared<'a> = NamClapShared;
    type MainThread<'a> = NamClapMainThread;
}

impl DefaultPluginFactory for NamClapPlugin {
    fn get_descriptor() -> PluginDescriptor {
        nam_descriptor()
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        Ok(NamClapShared)
    }

    fn new_main_thread<'a>(
        _host: HostMainThreadHandle<'a>,
        _shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        Ok(NamClapMainThread)
    }
}
