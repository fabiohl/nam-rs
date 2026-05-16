// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Implementação da extensão de estado (state) do CLAP.
//! Permite que o plugin salve e carregue sua configuração atual (parâmetros e modelo).

use crate::clap::plugin::{ClapParamPayload, NamClapMainThread};
use clack_common::stream::{InputStream, OutputStream};
use clack_extensions::log::{HostLog, LogSeverity};
use clack_extensions::params::{HostParams, ParamRescanFlags};
use clack_extensions::state::PluginStateImpl;
use clack_plugin::prelude::*;
use std::ffi::CString;
use std::io::{Read, Write};

impl<'a> PluginStateImpl for NamClapMainThread<'a> {
    fn save(&mut self, output: &mut OutputStream) -> Result<(), PluginError> {
        // 1. Sincroniza os parâmetros atuais com os valores atômicos mais recentes da audio thread
        self.params.input_gain_db = f32::from_bits(
            self.shared
                .param_input_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.output_gain_db = f32::from_bits(
            self.shared
                .param_output_gain
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.gate_threshold_db = f32::from_bits(
            self.shared
                .param_gate_thresh
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        self.params.bypass = self
            .shared
            .param_bypass
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0;

        // 2. Serializa os parâmetros atuais (incluindo o caminho do modelo) em JSON
        // Decisão técnica: Box::leak converte String em &'static str
        // para satisfazer PluginError::Message. É um memory leak intencional — aceitável
        // pois erros de serialização são raros em operação normal. Plano de remediação
        // registrado na Sprint 6 (T6.1.1) para uso de pool estático ou API owned.
        let serialized = serde_json::to_vec(&self.params).map_err(|e| {
            PluginError::Message(Box::leak(
                format!("Falha ao serializar estado: {}", e).into_boxed_str(),
            ))
        })?;

        // Escreve no stream do CLAP
        output.write_all(&serialized).map_err(|e| {
            PluginError::Message(Box::leak(
                format!("Falha ao escrever no stream de estado: {}", e).into_boxed_str(),
            ))
        })?;

        Ok(())
    }

    /// Carrega o estado do plugin a partir do stream fornecido pelo host.
    fn load(&mut self, input: &mut InputStream) -> Result<(), PluginError> {
        let mut buffer = Vec::new();
        input.read_to_end(&mut buffer).map_err(|e| {
            PluginError::Message(Box::leak(
                format!("Falha ao ler do stream de estado: {}", e).into_boxed_str(),
            ))
        })?;

        // Deserializa o JSON para os parâmetros locais
        let new_params: crate::common::params::NamPluginParams = serde_json::from_slice(&buffer)
            .map_err(|e| {
                PluginError::Message(Box::leak(
                    format!("Falha ao deserializar estado: {}", e).into_boxed_str(),
                ))
            })?;

        // 1. Atualiza os parâmetros na Main Thread e nos atomics
        self.params = new_params;
        self.shared.param_input_gain.store(
            self.params.input_gain_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.param_output_gain.store(
            self.params.output_gain_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.param_gate_thresh.store(
            self.params.gate_threshold_db.to_bits(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.shared.param_bypass.store(
            if self.params.bypass { 1 } else { 0 },
            std::sync::atomic::Ordering::Relaxed,
        );

        // 2. Tenta carregar o modelo se houver um caminho salvo
        if let Some(path) = self.params.model_path.clone() {
            if path.exists() {
                if let Err(e) = self.load_model(&path) {
                    // Logamos o erro mas não impedimos o carregamento do restante do estado
                    if let Some(log) = self.host.get_extension::<HostLog>() {
                        let msg = format!(
                            "NAM-rs: Falha ao restaurar modelo salvo ({:?}): {}",
                            path, e
                        );
                        if let Ok(c_msg) = CString::new(msg) {
                            log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                        }
                    }
                }
            } else if let Some(log) = self.host.get_extension::<HostLog>() {
                // Arquivo não existe no path salvo
                let msg = format!("NAM-rs: Modelo salvo não encontrado no caminho: {:?}", path);
                if let Ok(c_msg) = CString::new(msg) {
                    log.log(&self.host.shared(), LogSeverity::Warning, &c_msg);
                }
            }
        }

        // 3. Sincroniza os parâmetros (ganhos, gate, bypass) com a Audio Thread
        let _ = self
            .param_tx
            .push(ClapParamPayload::Params(self.params.clone()));

        // 4. Notifica o host que os valores dos parâmetros mudaram
        if let Some(params_ext) = self.host.get_extension::<HostParams>() {
            params_ext.rescan(&mut self.host, ParamRescanFlags::VALUES);
        }

        Ok(())
    }
}
