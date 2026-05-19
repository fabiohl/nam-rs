// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Definição do plugin NAM-rs e seus componentes de ciclo de vida CLAP.

use crate::clap::descriptor::nam_descriptor;
use crate::clap::processor::NamClapProcessor;
use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::common::params::NamPluginParams;
use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags, drain_gc_channels};
use crate::loader::{LoadedModelPair, load_and_build_model};
use clack_extensions::log::{HostLog, LogSeverity};
use clack_plugin::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};
use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Payload de comunicação Main -> RT para o plugin CLAP.
pub enum ClapParamPayload {
    /// Atualização de parâmetros (ganho, gate, bypass).
    Params(NamPluginParams),
    /// Carregamento de um novo par de modelos com metadados de calibração.
    LoadModel(Box<LoadedModelPair>),
}

/// Estado compartilhado entre a audio thread e a main thread (lock-free).
///
/// Adota alinhamento a 128 bytes para mitigar False Sharing.
/// Os canais SPSC são envolvidos em Mutex<Option<...>> apenas para permitir
/// que sejam "extraídos" pelas respectivas threads durante a inicialização,
/// satisfazendo o requisito `Sync` da trait `PluginShared`.
#[repr(align(128))]
pub struct NamClapShared {
    /// Canal SPSC: Main Thread -> Audio Thread (Novos parâmetros/modelos).
    pub param_tx: Mutex<Option<Producer<ClapParamPayload>>>,
    /// Canal SPSC: Main Thread -> Audio Thread (Consumidor).
    pub param_rx: Mutex<Option<Consumer<ClapParamPayload>>>,
    /// Canal GC: Audio Thread -> Main Thread (Modelos obsoletos para descarte).
    pub gc_tx: Mutex<Option<Producer<GcItem>>>,
    /// Canal GC: Audio Thread -> Main Thread (Consumidor).
    pub gc_rx: Mutex<Option<Consumer<GcItem>>>,
    /// Buffer de fallback para overflow de GC (overwrite).
    pub gc_overflow: Arc<GcOverflowBuffer>,
    /// Flags atômicas de status (telemetria RT->Main).
    pub rt_status: Arc<RtStatusFlags>,
    /// Latência atual reportada ao host (em samples).
    pub current_latency: AtomicU32,
    /// Último valor do parâmetro Input Gain (f32 como bits).
    pub param_input_gain: AtomicU32,
    /// Último valor do parâmetro Output Gain (f32 como bits).
    pub param_output_gain: AtomicU32,
    /// Último valor do parâmetro Gate Threshold (f32 como bits).
    pub param_gate_thresh: AtomicU32,
    /// Último valor do parâmetro Bypass (0 = false, 1 = true).
    pub param_bypass: AtomicU32,
}

impl<'a> PluginShared<'a> for NamClapShared {}

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
}

impl<'a> PluginMainThread<'a, NamClapShared> for NamClapMainThread<'a> {
    /// Chamado periodicamente ou em resposta a eventos do host.
    /// Aqui podemos aproveitar para drenar o canal de GC.
    fn on_main_thread(&mut self) {
        // Drenar modelos obsoletos para liberar memória fora do RT.
        drain_gc_channels(&mut self.gc_rx, &self.shared.gc_overflow);

        // Logging RT-Safe via flags atômicas (consome eventos transientes)
        if let Some(log) = self.host.get_extension::<HostLog>() {
            let shared = self.host.shared();

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED)
            {
                let msg = CString::new("NAM-rs: Saturação (clipping) detectada na saída!")
                    .expect("Falha ao criar CString");
                log.log(&shared, LogSeverity::Warning, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW)
            {
                let msg =
                    CString::new("NAM-rs: Overflow no canal de GC! Possível leak de memória.")
                        .expect("Falha ao criar CString");
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED)
            {
                let msg =
                    CString::new("NAM-rs: Falha crítica! Nenhum modelo ativo para processamento.")
                        .expect("Falha ao criar CString");
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
                    .message(format!("Falha ao carregar o modelo: {:?}", path))
                    .param("error", e.to_string()),
            )
        })?;

        // 2. Envio para a thread RT via canal SPSC
        self.param_tx
            .push(ClapParamPayload::LoadModel(Box::new(model_pair)))
            .map_err(|_| {
                Box::new(
                    NamDiagnostic::new(NamErrorCode::ParamChannelFull, &self.sys)
                        .message("O canal de comunicação com a thread de áudio está cheio.")
                        .hint("Tente carregar o modelo novamente em alguns instantes."),
                )
            })?;

        // 3. Atualiza o path nos parâmetros locais (espelhados)
        self.params.model_path = Some(path.to_path_buf());

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

/// Plugin NAM-rs: ponto de entrada principal do ciclo de vida CLAP.
pub struct NamClapPlugin;

impl Plugin for NamClapPlugin {
    type AudioProcessor<'a> = NamClapProcessor<'a>;
    type Shared<'a> = NamClapShared;
    type MainThread<'a> = NamClapMainThread<'a>;

    fn declare_extensions(
        builder: &mut PluginExtensions<Self>,
        _shared: Option<&Self::Shared<'_>>,
    ) {
        builder.register::<clack_extensions::audio_ports::PluginAudioPorts>();
        builder.register::<clack_extensions::params::PluginParams>();
        builder.register::<clack_extensions::state::PluginState>();
        builder.register::<crate::clap::extensions::latency::NamPluginLatency>();

        #[cfg(feature = "clap-plugin-gui")]
        builder.register::<crate::clap::extensions::gui::NamPluginGui>();
    }
}

impl DefaultPluginFactory for NamClapPlugin {
    fn get_descriptor() -> PluginDescriptor {
        nam_descriptor()
    }

    fn new_shared(_host: HostSharedHandle<'_>) -> Result<Self::Shared<'_>, PluginError> {
        let (param_tx, param_rx) = RingBuffer::new(8);
        let (gc_tx, gc_rx) = RingBuffer::new(32); // Capacidade ampliada para o plugin

        Ok(NamClapShared {
            param_tx: Mutex::new(Some(param_tx)),
            param_rx: Mutex::new(Some(param_rx)),
            gc_tx: Mutex::new(Some(gc_tx)),
            gc_rx: Mutex::new(Some(gc_rx)),
            gc_overflow: Arc::new(GcOverflowBuffer::new(64)),
            rt_status: Arc::new(RtStatusFlags::new()),
            current_latency: AtomicU32::new(0),
            param_input_gain: AtomicU32::new(0.0f32.to_bits()),
            param_output_gain: AtomicU32::new(0.0f32.to_bits()),
            param_gate_thresh: AtomicU32::new((-70.0f32).to_bits()),
            param_bypass: AtomicU32::new(0),
        })
    }

    fn new_main_thread<'a>(
        host: HostMainThreadHandle<'a>,
        shared: &'a Self::Shared<'a>,
    ) -> Result<Self::MainThread<'a>, PluginError> {
        // Extrai os canais exclusivos da Main Thread do estado compartilhado
        // O uso de expect() é seguro aqui pois estas instâncias DEVEM estar presentes
        // e são inicializadas no new_shared().
        let param_tx = shared
            .param_tx
            .lock()
            .expect("Falha ao travar o Mutex do param_tx")
            .take()
            .expect("Produtor param_tx já foi extraído");

        let gc_rx = shared
            .gc_rx
            .lock()
            .expect("Falha ao travar o Mutex do gc_rx")
            .take()
            .expect("Consumidor gc_rx já foi extraído");

        #[cfg_attr(test, allow(unused_mut))]
        let mut main_thread = NamClapMainThread {
            shared,
            params: NamPluginParams::default(),
            host,
            sys: SystemSnapshot::capture(),
            param_tx,
            gc_rx,
            last_reported_latency: 0,
        };

        // TODO: REMOVER quando a GUI estiver funcional (Tarefa 4.3.1).
        // Busca automática de modelo para facilitar testes iniciais sem GUI.
        // Decisão técnica: O modelo carregado é indeterminístico
        // (depende da ordem de `readdir` do filesystem). Isso é aceito por design
        // pois este bloco inteiro é temporário e será removido na Sprint 4.
        #[cfg(not(test))]
        if let Ok(home) = std::env::var("HOME") {
            let clap_dir = std::path::Path::new(&home).join(".clap");
            if let Ok(entries) = std::fs::read_dir(&clap_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
                    if ext == "nam" || ext == "namb" {
                        #[allow(clippy::collapsible_if)]
                        if let Ok(()) = main_thread.load_model(&path) {
                            if let Some(log) = main_thread.host.get_extension::<HostLog>() {
                                let msg = format!(
                                    "NAM-rs: Carregado automaticamente o modelo {:?}",
                                    path
                                );
                                if let Ok(c_msg) = CString::new(msg) {
                                    log.log(&main_thread.host.shared(), LogSeverity::Info, &c_msg);
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }

        Ok(main_thread)
    }
}
