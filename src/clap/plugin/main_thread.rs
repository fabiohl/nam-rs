// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Estado exclusivo da main thread (carregamento de modelos, state save/load).

use super::shared::{ClapParamPayload, NamClapShared, NamModelMetadata};
use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, drain_gc_channels};
use crate::dsp::resampler::NamResampler;
use crate::loader::load_and_build_model;
use clack_extensions::log::{HostLog, LogSeverity};
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer};
use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::Ordering;

/// Estado exclusivo da main thread (carregamento de modelos, state save/load).
pub struct NamClapMainThread<'a> {
    pub(crate) shared: &'a NamClapShared,
    /// Parâmetros atuais conhecidos pela main thread (espelho dos params da audio thread).
    pub params: NamPluginParams,
    /// Handle do host para notificações (request_restart, latency_changed, etc.).
    pub host: HostMainThreadHandle<'a>,
    /// Snapshot do sistema para emissão de diagnósticos.
    pub sys: SystemSnapshot,
    /// Produtor para enviar atualizações para a audio thread.
    pub param_tx: Producer<ClapParamPayload>,
    /// Consumidor para coletar lixo (modelos obsoletos) da audio thread.
    pub gc_rx: Consumer<GcItem>,
    /// Cache da última latência reportada ao host para evitar notificações redundantes.
    pub last_reported_latency: u32,
    /// Handle da janela baseview para controle de ciclo de vida da GUI.
    #[cfg(feature = "clap-plugin")]
    pub window_handle: Option<baseview::WindowHandle>,
}

impl<'a> PluginMainThread<'a, NamClapShared> for NamClapMainThread<'a> {
    /// Chamado periodicamente ou em resposta a eventos do host.
    /// Aqui podemos aproveitar para drenar o canal de GC.
    fn on_main_thread(&mut self) {
        // Drenar modelos obsoletos para liberar memória fora do RT.
        drain_gc_channels(&mut self.gc_rx, &self.shared.gc_overflow);

        // Verifica se há um modelo pendente enviado pela UI
        let pending_model = if let Ok(mut pending_guard) = self.shared.ui_pending_model.lock() {
            pending_guard.take()
        } else {
            None
        };
        if let Some(path) = pending_model {
            let res = self.load_model(&path);
            self.shared.ui_loading.store(false, Ordering::Relaxed);
            match res {
                Ok(_) => {}
                Err(e) => {
                    let err_msg = match e.error_code() {
                        NamErrorCode::FileNotFound => "File not found",
                        NamErrorCode::FileReadError => "File read error",
                        NamErrorCode::UnknownExtension => "Unknown extension",
                        NamErrorCode::NamJsonParseError => "Invalid JSON format",
                        NamErrorCode::NambCrc32Mismatch => "CRC32 checksum mismatch",
                        NamErrorCode::NambCrc32Missing => "CRC32 integrity flag missing (v2+)",
                        NamErrorCode::NambInvalidMagic => "Invalid signature",
                        NamErrorCode::NambUnsupportedVersion => "Unsupported version",
                        NamErrorCode::NambTruncated => "Corrupted/truncated file",
                        NamErrorCode::UnsupportedArchitecture => "Unsupported architecture",
                        NamErrorCode::TopologyDetectionFailed => "Topology detection failed",
                        NamErrorCode::WeightCountMismatch => "Weight count mismatch",
                        NamErrorCode::ModelBuildFailed => "Model build failed",
                        NamErrorCode::ModelTooLarge => "Model file too large",
                        _ => "Internal error",
                    };
                    if let Ok(mut msg_guard) = self.shared.ui_load_error_msg.lock() {
                        *msg_guard = err_msg.to_string();
                    }
                    self.shared.ui_load_error.store(true, Ordering::Relaxed);

                    if let Some(log) = self.host.get_extension::<HostLog>() {
                        let shared = self.host.shared();
                        let err_str = format!("NAM-rs: Failed to load model from GUI: {:?}", e);
                        let sanitized_err = err_str.replace('\0', " ");
                        let msg = CString::new(sanitized_err).unwrap_or_else(|_| {
                            CString::new(
                                "NAM-rs: Failed to load model from GUI due to invalid characters",
                            )
                            .unwrap()
                        });
                        log.log(&shared, LogSeverity::Error, &msg);
                    }
                }
            }
        }

        // Logging RT-Safe via flags atômicas (consome eventos transientes)
        if let Some(log) = self.host.get_extension::<HostLog>() {
            let shared = self.host.shared();

            // WHITELIST: CString::new(…).expect() nas linhas abaixo é seguro porque:
            // - Todas as strings são literais ASCII estáticos sem bytes nulos internos.
            // - O compilador garante que tais strings não contêm '\0' antes do terminador.
            // - Estas chamadas estão na main thread, fora de qualquer hotpath RT ou FFI de áudio.
            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED)
            {
                let msg = CString::new("NAM-rs: Output clipping detected!")
                    .expect("Failed to create CString");
                log.log(&shared, LogSeverity::Warning, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW)
            {
                let msg = CString::new("NAM-rs: GC channel overflow! Possible memory leak.")
                    .expect("Failed to create CString");
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED)
            {
                let msg = CString::new("NAM-rs: Critical failure! No active model for processing.")
                    .expect("Failed to create CString");
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HEAP_ALLOC)
            {
                let msg = CString::new(
                    "NAM-rs: Heap allocation detected in audio thread during process()!",
                )
                .expect("Failed to create CString");
                log.log(&shared, LogSeverity::Error, &msg);
            }
        }

        // 2. Monitoramento de Latência: Notifica o host se o valor mudou
        let current_latency = self.shared.current_latency.load(Ordering::Relaxed);
        if current_latency != self.last_reported_latency {
            self.last_reported_latency = current_latency;
            if let Some(latency_ext) = self
                .host
                .get_extension::<clack_extensions::latency::HostLatency>()
            {
                latency_ext.changed(&mut self.host);
            }
            self.host.request_restart();
        }
    }
}

impl<'a> NamClapMainThread<'a> {
    /// Carrega um novo modelo neural a partir do caminho especificado.
    ///
    /// Este método realiza I/O e alocações de memória, sendo seguro para execução
    /// apenas na thread principal. O modelo carregado é enviado para a thread RT
    /// via canal lock-free.
    pub fn load_model(&mut self, path: &Path) -> Result<(), Box<NamDiagnostic>> {
        // 1. Carregamento e build do par de modelos (L+R)
        let model_pair = load_and_build_model(path, &self.sys).map_err(|e| {
            Box::new(
                NamDiagnostic::new(NamErrorCode::ModelBuildFailed, &self.sys)
                    .message(format!("Failed to load model: {:?}", path))
                    .param("error", e.to_string()),
            )
        })?;

        // Construção do novo resampler para a taxa do host e do modelo
        let host_rate = self.shared.sample_rate.load(Ordering::Relaxed);
        let host_rate = if host_rate == 0 { 48000 } else { host_rate };
        let new_resampler = Box::new(
            NamResampler::new(host_rate, model_pair.sample_rate, 0).map_err(|e| {
                Box::new(
                    NamDiagnostic::new(NamErrorCode::ModelBuildFailed, &self.sys)
                        .message("Failed to build resampler")
                        .param("error", e.to_string()),
                )
            })?,
        );

        // 3. Atualiza o path nos parâmetros locais (espelhados)
        self.params.model_path = Some(path.to_path_buf());

        // Extrai e atualiza os metadados do modelo para a GUI
        let metadata = model_pair.metadata.clone();
        let architecture = model_pair.architecture.clone();
        let topology = model_pair.topology.clone();
        if let Ok(mut meta_guard) = self.shared.ui_model_metadata.lock() {
            *meta_guard = Some(NamModelMetadata {
                architecture,
                topology,
                modeled_by: metadata.as_ref().and_then(|m| m.modeled_by.clone()),
                gear_make: metadata.as_ref().and_then(|m| m.gear_make.clone()),
                gear_model: metadata.as_ref().and_then(|m| m.gear_model.clone()),
                gear_type: metadata.as_ref().and_then(|m| m.gear_type.clone()),
                tone_type: metadata.as_ref().and_then(|m| m.tone_type.clone()),
                date: metadata
                    .as_ref()
                    .and_then(|m| m.date.as_ref())
                    .map(|d| match (d.year, d.month, d.day) {
                        (Some(y), Some(m), Some(d)) => format!("{:04}-{:02}-{:02}", y, m, d),
                        (Some(y), Some(m), None) => format!("{:04}-{:02}", y, m),
                        (Some(y), None, None) => format!("{:04}", y),
                        _ => String::new(),
                    })
                    .filter(|s| !s.is_empty()),
            });
        }

        // Atualiza o sample rate do modelo
        self.shared
            .model_sample_rate
            .store(model_pair.sample_rate, Ordering::Relaxed);

        // 2. Envio para a thread RT via canal SPSC
        self.param_tx
            .push(ClapParamPayload::LoadModel(
                Box::new(model_pair),
                new_resampler,
            ))
            .map_err(|_| {
                Box::new(
                    NamDiagnostic::new(NamErrorCode::ParamChannelFull, &self.sys)
                        .message("The communication channel with the audio thread is full.")
                        .hint("Please try loading the model again in a few moments."),
                )
            })?;

        // Atualiza o nome do modelo carregado para exibição na UI (basename)
        let basename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if let Ok(mut name_guard) = self.shared.ui_model_name.lock() {
            *name_guard = basename;
        }
        self.shared
            .model_load_counter
            .fetch_add(1, Ordering::Relaxed);

        // Notifica o host que o valor do parâmetro de modelo ativo mudou
        if let Some(params_ext) = self
            .host
            .get_extension::<clack_extensions::params::HostParams>()
        {
            params_ext.rescan(
                &mut self.host,
                clack_extensions::params::ParamRescanFlags::VALUES,
            );
        }

        // 4. Notifica o host sobre o carregamento bem-sucedido (Cold Path)
        if let Some(log) = self.host.get_extension::<HostLog>() {
            let msg = format!("NAM-rs: model loaded ({:?})", path);
            if let Ok(c_msg) = CString::new(msg) {
                log.log(&self.host.shared(), LogSeverity::Info, &c_msg);
            }
        }

        // 5. Notifica o host que o estado mudou (dirty)
        if let Some(mut state_ext) = self
            .host
            .get_extension::<clack_extensions::state::HostState>()
        {
            state_ext.mark_dirty(&self.host);
        }

        Ok(())
    }
}
