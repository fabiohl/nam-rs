// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Definição do plugin NAM-rs e seus componentes de ciclo de vida CLAP.

use crate::clap::descriptor::nam_descriptor;
use crate::clap::processor::NamClapProcessor;
use crate::common::params::NamPluginParams;
use crate::models::DynamicModel;
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Mutex;

/// Estado compartilhado entre a audio thread e a main thread (lock-free).
///
/// Adota alinhamento a 128 bytes para mitigar False Sharing.
/// Os canais SPSC são envolvidos em Mutex<Option<...>> apenas para permitir
/// que sejam "extraídos" pelas respectivas threads durante a inicialização,
/// satisfazendo o requisito `Sync` da trait `PluginShared`.
#[repr(align(128))]
pub struct NamClapShared {
    /// Canal SPSC: Main Thread -> Audio Thread (Novos parâmetros).
    pub param_tx: Mutex<Option<Producer<NamPluginParams>>>,
    /// Canal SPSC: Main Thread -> Audio Thread (Consumidor).
    pub param_rx: Mutex<Option<Consumer<NamPluginParams>>>,
    /// Canal GC: Audio Thread -> Main Thread (Modelos obsoletos para descarte).
    pub gc_tx: Mutex<Option<Producer<Box<DynamicModel>>>>,
    /// Canal GC: Audio Thread -> Main Thread (Consumidor).
    pub gc_rx: Mutex<Option<Consumer<Box<DynamicModel>>>>,
}

impl<'a> PluginShared<'a> for NamClapShared {}

/// Estado exclusivo da main thread (carregamento de modelos, state save/load).
pub struct NamClapMainThread;

impl<'a> PluginMainThread<'a, NamClapShared> for NamClapMainThread {}

/// Plugin NAM-rs: ponto de entrada principal do ciclo de vida CLAP.
pub struct NamClapPlugin;

use clack_extensions::audio_ports::PluginAudioPorts;

impl Plugin for NamClapPlugin {
    type AudioProcessor<'a> = NamClapProcessor;
    type Shared<'a> = NamClapShared;
    type MainThread<'a> = NamClapMainThread;

    fn declare_extensions(
        builder: &mut PluginExtensions<Self>,
        _shared: Option<&Self::Shared<'_>>,
    ) {
        builder.register::<PluginAudioPorts>();
    }
}

impl DefaultPluginFactory for NamClapPlugin {
    fn get_descriptor() -> PluginDescriptor {
        nam_descriptor()
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        let (param_tx, param_rx) = RingBuffer::new(8);
        let (gc_tx, gc_rx) = RingBuffer::new(8);

        Ok(NamClapShared {
            param_tx: Mutex::new(Some(param_tx)),
            param_rx: Mutex::new(Some(param_rx)),
            gc_tx: Mutex::new(Some(gc_tx)),
            gc_rx: Mutex::new(Some(gc_rx)),
        })
    }

    fn new_main_thread<'a>(
        _host: HostMainThreadHandle<'a>,
        _shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        Ok(NamClapMainThread)
    }
}
