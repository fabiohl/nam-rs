// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão `clap_plugin_latency` para o NAM-rs.

use crate::clap::plugin::NamClapMainThread;
use clack_extensions::latency::{PluginLatency, PluginLatencyImpl};
use std::sync::atomic::Ordering;

/// Implementação da trait `PluginLatencyImpl` para o plugin NAM-rs.
/// A trait é implementada na `MainThread`.
impl<'a> PluginLatencyImpl for NamClapMainThread<'a> {
    /// Retorna a latência atual do plugin em amostras.
    ///
    /// O valor é lido do estado compartilhado, que é atualizado pelo `NamClapProcessor`
    /// sempre que a latência algorítmica muda (ex: ativação ou troca de modelo).
    fn get(&mut self) -> u32 {
        self.shared.current_latency.load(Ordering::Relaxed)
    }
}

/// Tipo marcador para registro da extensão.
pub type NamPluginLatency = PluginLatency;
