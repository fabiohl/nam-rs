// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Núcleo de processamento de áudio DSP usando `pipewire-rs`.
//!
//! Este é o "coração" do NAM-rs: o módulo que de fato processa o sinal de áudio em
//! tempo real. Ele recebe amostras brutas de áudio do PipeWire (o servidor de som
//! do Linux), passa pelo "motor neural", e entrega o resultado final processado
//! ao hardware via arquitetura dual-stream.
//!
//! ## Arquitetura Dual-Stream com DspBridge
//!
//! O PipeWire copia os buffers para o monitor port **antes** de chamar o `process()`.
//! Portanto, modificações in-place numa única stream `Audio/Sink` seriam invisíveis
//! ao hardware. A solução usa duas streams:
//!
//! 1. **Capture stream** (`Audio/Sink`, `Direction::Input`) — atua como um Virtual Sink
//!    que recebe áudio dos apps, aplica a cadeia DSP (ganho + inferência neural)
//!    e escreve o resultado no [`DspBridge`].
//! 2. **Playback stream** (`Stream/Output/Audio`, `Direction::Output`) — atua como um
//!    cliente de reprodução que lê do [`DspBridge`] e entrega ao hardware.
//!
//! O [`DspBridge`] é um buffer `#[repr(align(128))]` compartilhado entre as duas
//! closures via ponteiro raw, com sincronização lock-free via `fence(Release/Acquire)`
//! e contador de geração atômico.
//!
//! ## Regras absolutas deste módulo (por que são tão rigorosas?)
//!
//! No `process()` callback (a função chamada centenas de vezes por segundo pelo PipeWire):
//! - **Zero alocação na heap** — nunca pedimos memória nova ao sistema durante o processamento.
//! - **Zero I/O** — nunca escrevemos no terminal ou em arquivos; status é reportado via flags atômicas.
//! - **Zero mutexes** — nunca travamos/esperamos por outras threads.
//!
//! Essas regras existem porque qualquer pausa, por menor que seja, causaria estalos e cortes no
//! áudio — inaceitável para um músico tocando ao vivo.
//!
//! ## Fluxo de processamento (Capture callback)
//!
//! O callback `process()` segue esta sequência para cada bloco de áudio:
//! 1. Aplica ganho de entrada (ajuste de volume pré-processamento): uma conveniência oferecida ao usuário
//! 2. `NamResampler::process_input()` — converte o sample rate para a taxa compatível (geralmente 48 kHz)
//! 3. Inferência neural WaveNet/LSTM — o motor neural que processa o sinal sonoro
//! 4. `NamResampler::process_output()` — converte de volta para o sample rate original
//! 5. Aplica ganho de saída (ajuste de volume pós-processamento): uma conveniência oferecida ao usuário
//! 6. Escreve resultado no [`DspBridge`] com `fence(Release)` para o playback callback
//!
//! Quando nenhum modelo está carregado, o motor emite silêncio (prevenindo ruídos inesperados).
//! Quando o sample rate do PipeWire é o mesmo do modelo nam, o resampler opera em bypass sem overhead.

use crate::colors::Colorize;
use crate::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::dsp::gate::{DynamicHysteresis, GateParams, GateState};
use crate::dsp::resampler::NamResampler;
use crate::math::simd::{compute_energy_avx2, compute_max_diff_avx2};
use crate::models::NamModel;
use crate::rt_setup;
use crate::spsc::{ParamPayload, RtStatusFlags, SHUTDOWN};
use minstant::{Anchor, Instant};
use pipewire as pw;
use pw::properties::properties;
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Tamanho máximo do buffer intermediário entre as duas streams (capture → playback).
/// Dimensionado para o quantum máximo do PipeWire (`max-quantum = 8192`).
const MAX_BRIDGE_BUF: usize = 8192;
const MAX_RESAMP_BUF: usize = 4096;

/// Buffer individual de áudio para o DspBridge (double-buffer).
#[repr(align(128))]
struct BridgeBuffer {
    /// Buffer de saída processada, canal esquerdo.
    buf_l: [f32; MAX_BRIDGE_BUF],
    /// Buffer de saída processada, canal direito.
    buf_r: [f32; MAX_BRIDGE_BUF],
    /// Número de amostras válidas no buffer atual.
    n_samples: u32,
}

/// Buffer compartilhado entre o callback de captura (DSP) e o callback de playback.
///
/// O capture callback escreve o resultado processado aqui com `fence(Release)`;
/// o playback callback lê com `fence(Acquire)`. A `generation` atômica permite
/// ao playback detectar se há dados novos disponíveis sem spin-lock.
///
/// Alinhado a 128 bytes para evitar false-sharing entre os dois callbacks RT.
#[repr(align(128))]
pub(crate) struct DspBridge {
    /// Os dois buffers físicos (front / back) para o double-buffering.
    buffers: [BridgeBuffer; 2],
    /// Índice do buffer ativo para LEITURA (0 ou 1). O capture sempre escreve no (1 - ativo).
    active_read_idx: std::sync::atomic::AtomicUsize,
    /// Contador de geração — incrementado a cada escrita pelo capture callback.
    /// O playback compara com sua cópia local para detectar novos dados.
    generation: std::sync::atomic::AtomicU64,
}

/// Estrutura de posse explícita para evitar o leak de memória
/// das instâncias essenciais do PipeWire (`StreamBox` e `Listener`).
/// Mantém ownership das 4 instâncias (2 streams + 2 listeners) da arquitetura dual-stream.
struct AppState<S1, L1, S2, L2> {
    #[allow(dead_code)]
    capture_stream: S1,
    #[allow(dead_code)]
    capture_listener: L1,
    #[allow(dead_code)]
    playback_stream: S2,
    #[allow(dead_code)]
    playback_listener: L2,
}

/// Configurações para inicialização do host PipeWire.
pub struct PipewireHostConfig {
    /// Tamanho do buffer de áudio solicitado.
    pub buffer_size: u32,
    /// Âncora de tempo para telemetria RDTSC.
    pub tsc_anchor: Anchor,
    /// Snapshot do sistema para diagnósticos.
    pub sys: SystemSnapshot,
}

/// Contexto de dados para a pipeline DSP hot-path.
pub(crate) struct DspPipelineContext<'a> {
    /// Resampler ativo para conversão de sample rate.
    pub resampler: &'a mut NamResampler,
    /// Modelo ativo para o canal esquerdo.
    pub active_model_l: &'a mut Option<Box<crate::models::DynamicModel>>,
    /// Modelo ativo para o canal direito.
    pub active_model_r: &'a mut Option<Box<crate::models::DynamicModel>>,
    /// Multiplicador de ganho de entrada.
    pub input_gain_mult: f32,
    /// Multiplicador de ganho de saída.
    pub output_gain_mult: f32,
    /// Parâmetros do Noise Gate.
    pub gate_params: &'a GateParams,
    /// Histerese de silêncio global.
    pub silence_hysteresis: &'a mut DynamicHysteresis,
    /// Histerese para detecção de sinal mono.
    pub mono_hysteresis: &'a mut DynamicHysteresis,
    /// Threshold de abertura do gate (quadrático).
    pub threshold_open_sq: f32,
    /// Threshold de fechamento do gate (quadrático).
    pub threshold_close_sq: f32,
    /// Flag indicando se o processamento deve ser mono.
    pub process_mono: &'a mut bool,
    /// Flags de status em tempo real.
    pub rt_status: &'a RtStatusFlags,
    /// Ponteiro para a ponte de memória compartilhada.
    pub bridge_ptr: *mut DspBridge,
    /// Buffer intermediário L (pós-resampler input).
    pub resamp_mid_l: &'a mut [f32],
    /// Buffer intermediário R (pós-resampler input).
    pub resamp_mid_r: &'a mut [f32],
    /// Buffer de saída L (pré-resampler output).
    pub resamp_out_l: &'a mut [f32],
    /// Buffer de saída R (pré-resampler output).
    pub resamp_out_r: &'a mut [f32],
}

/// Silence Bypass: sinaliza silêncio e zera o bridge para que o playback emita silêncio.
///
/// Marcada como `#[cold]` e `#[inline(never)]` para sinalizar ao LLVM que este é o path improvável.
/// Isso permite ao compilador posicionar este código fora da região quente de I-Cache do
/// `process()`, reduzindo a "I-Cache pressure" e otimizando o Branch Prediction para o path
/// onde o áudio está presente. O código de tratamento de silêncio é movido para uma região
/// distante da memória, mantendo o loop DSP principal compacto e eficiente.
///
/// ## Safety
/// - `bridge_ptr` deve apontar para um `DspBridge` válido (garantido pelo `Box::leak`).
/// - Deve ser chamada apenas do capture callback (único escritor do bridge).
#[cold]
#[inline(never)]
fn handle_silence_bypass(bridge_ptr: *mut DspBridge, rt_status: &RtStatusFlags) {
    // Sinaliza silêncio via flag atômica (zero I/O no callback RT)
    rt_status.is_silent.store(true, Ordering::Relaxed);

    // Zera o bridge para que o playback emita silêncio
    // em vez de repetir o último frame processado.
    // SAFETY: capture callback é o único escritor.
    let bridge_ref = unsafe { &mut *bridge_ptr };
    let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
    bridge_ref.buffers[back_idx].n_samples = 0;
    std::sync::atomic::fence(Ordering::Release);
    bridge_ref
        .active_read_idx
        .store(back_idx, Ordering::Relaxed);
    bridge_ref.generation.fetch_add(1, Ordering::Relaxed);
}

/// Inicializa a topologia PipeWire dual-stream (Capture + Playback).
///
/// Arquitetura: Apps → [Capture: Audio/Sink] → process(DSP) → DspBridge → [Playback: Stream/Output] → Hardware.
/// O monitor port do `Audio/Sink` copia o buffer *antes* do `process()` — portanto, a única
/// forma de entregar o áudio processado ao hardware é via uma segunda stream de playback
/// que lê do `DspBridge` pós-DSP.
///
/// ## Parâmetros de canais SPSC
///
/// - `consumer`: Consumidor do canal de parâmetros CLI→DSP (gain, modelo, etc.).
/// - `gc_producer`: Produtor do canal GC para drop-delegation de modelos obsoletos.
/// - `resampler_consumer`: Canal dedicado para receber resamplers pré-construídos
///   pela thread principal — **zero alocações no callback RT**.
/// - `resampler_producer`: Produtor do canal de resamplers — a thread principal
///   constrói `NamResampler::new()` aqui (alocação fora do RT) e envia para o callback.
/// - `rt_status`: Flags atômicas para comunicação silenciosa RT→Main.
pub fn run_pipewire_host(
    mut consumer: Consumer<ParamPayload>,
    mut gc_producer: rtrb::Producer<Box<crate::models::DynamicModel>>,
    mut resampler_consumer: Consumer<NamResampler>,
    mut resampler_producer: rtrb::Producer<NamResampler>,
    rt_status: Arc<RtStatusFlags>,
    config: PipewireHostConfig,
) -> anyhow::Result<()> {
    let PipewireHostConfig {
        buffer_size,
        tsc_anchor,
        sys,
    } = config;

    // 1. Cria a thread assíncrona gerenciada nativamente pelo PipeWire
    let thread_loop = unsafe { pw::thread_loop::ThreadLoopBox::new(Some("nam-rs-loop"), None) }?;
    let context = pw::context::ContextBox::new(thread_loop.loop_(), None)?;
    let core = context.connect(None)?;

    let capture_stream;
    let capture_listener;
    let playback_stream;
    let playback_listener;
    let mut thread_configured = false;

    // Aloca o buffer compartilhado entre capture e playback via leak intencional.
    // O DspBridge vive até o shutdown do processo — nunca é dropado em runtime.
    let bridge: &'static DspBridge = Box::leak(Box::new(DspBridge {
        buffers: [
            BridgeBuffer {
                buf_l: [0.0f32; MAX_BRIDGE_BUF],
                buf_r: [0.0f32; MAX_BRIDGE_BUF],
                n_samples: 0,
            },
            BridgeBuffer {
                buf_l: [0.0f32; MAX_BRIDGE_BUF],
                buf_r: [0.0f32; MAX_BRIDGE_BUF],
                n_samples: 0,
            },
        ],
        active_read_idx: std::sync::atomic::AtomicUsize::new(0),
        generation: std::sync::atomic::AtomicU64::new(0),
    }));
    // Ponteiro raw para compartilhamento seguro entre closures (ambos RT, mesmo context PW).
    let bridge_ptr = bridge as *const DspBridge as *mut DspBridge;

    // Otimiza o gerenciamento de memória do bridge no Kernel:
    // - MADV_DONTFORK: Evita overhead de Copy-on-Write (COW) se o processo sofrer fork.
    // - MADV_DONTDUMP: Exclui os grandes buffers de áudio de core dumps, reduzindo latência de escrita em falhas.
    unsafe {
        libc::madvise(
            bridge_ptr as *mut libc::c_void,
            std::mem::size_of::<DspBridge>(),
            libc::MADV_DONTFORK | libc::MADV_DONTDUMP,
        );
    }

    // Seleciona o núcleo ideal para a thread RT (capacidade + menor carga IRQ).
    // Computado antes do lock para que o valor seja acessível após o escopo.
    let target_cpu = rt_setup::select_optimal_cpu().unwrap_or(0);

    // Escopo de configuração protegida: o lock é liberado automaticamente no '}' final.
    {
        // Obtém o lock para o loop, necessário para criar e configurar streams do PW com segurança.
        // O lock é automaticamente liberado ao final deste escopo (_lock is dropped).
        let _lock = thread_loop.lock();

        let mut capture_props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Duplex",
            *pw::keys::MEDIA_ROLE => "DSP",
            *pw::keys::MEDIA_CLASS => "Audio/Sink", // Virtual Sink — recebe áudio dos apps
            *pw::keys::NODE_NAME => "NAM-rs-input",
            *pw::keys::NODE_DESCRIPTION => "NAM-rs Input",
            *pw::keys::NODE_VIRTUAL => "true",
            *pw::keys::PRIORITY_SESSION => "2000",
            *pw::keys::PRIORITY_DRIVER => "2000",
            "audio.position" => "FL,FR",
            "node.group" => "nam-rs-dsp", // Sincroniza clock com playback stream
            "node.link-group" => "nam-rs-link-group", // Evita feedback loop (WirePlumber não linka nós do mesmo link-group)
        };

        // Configura a latência desejada na stream (se especificada via CLI).
        let latency_str = format!("{}/48000", buffer_size);
        if buffer_size > 0 {
            capture_props.insert("node.latency", latency_str.as_str());
        }

        // Inicializa a stream de captura (Virtual Sink) que receberá áudio das aplicações.
        capture_stream = pw::stream::StreamBox::new(&core, "NAM-rs", capture_props)?;

        // Estado interno dos modelos ativos: um para cada canal, permitindo processamento stereo.
        let mut active_model_l: Option<Box<crate::models::DynamicModel>> = None;
        let mut active_model_r: Option<Box<crate::models::DynamicModel>> = None;

        // NamResampler bidirecional: converte o rate do PipeWire para o rate suportado pelo NAM carregado (geralmente 48khz).
        // Inicializado com 48k (bypass) — rate real será atualizado via canal SPSC de resamplers
        // quando a thread principal construir e enviar um novo resampler.
        let mut resampler = NamResampler::new(48_000, 48_000, 2048).unwrap_or_else(|e| {
            NamDiagnostic::new(NamErrorCode::ResamplerBuildFailed, &sys)
                .message("Falha ao criar NamResampler inicial (usando bypass 48k).")
                .hint("O engine contínua em modo bypass. O resampler será recriado ao receber o rate real do PipeWire.")
                .param("initial_rate", 48_000_u32)
                .param("detail", &e)
                .emit_warning();
            // Fallback: bypass 48k nunca falha (rate == NAM_RATE)
            NamResampler::new(48_000, 48_000, 2048).expect("bypass não pode falhar")
        });

        // Trackeia a taxa de amostragem alvo do loop RT
        let mut current_nam_rate: u32 = 48_000;

        // Buffer intermediário 48k para saída do resampler (stack-allocated).
        // MAX_RESAMP_BUF comporta a expansão do resampler (ex: 44100→48000 = ~1.088x).
        let mut resamp_mid_l = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_l = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_mid_r = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0f32; MAX_RESAMP_BUF];

        // Variáveis de ganho como multiplicadores lineares
        let mut user_input_gain_mult: f32 = 1.0;
        let mut user_output_gain_mult: f32 = 1.0;
        let mut model_input_mult_adj: f32 = 1.0;
        let mut model_output_mult_adj: f32 = 1.0;

        // Multiplicadores pré-calculados lineares para inserção no DSP FMA Simd
        let mut input_gain_mult: f32 = 1.0;
        let mut output_gain_mult: f32 = 1.0;

        // Histerese Dinâmica para otimizações (Silêncio/Mono)
        let mut gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        // Thresholds pré-calculados em valor linear² (MS) — cold-path only.
        // Evita `powf` no callback RT (T16). Atualizados ao receber GateConfig.
        let lut = crate::math::fastmath::get_gain_lut();
        let open_lin = lut.db_to_linear(gate_params.threshold_open_db);
        let close_lin = lut.db_to_linear(gate_params.threshold_close_db);
        let mut threshold_open_sq: f32 = open_lin * open_lin;
        let mut threshold_close_sq: f32 = close_lin * close_lin;

        let shared_target_rate = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let rate_for_param = shared_target_rate.clone();
        let rate_for_process = shared_target_rate.clone();

        // Clonar Arc das flags para uso dentro do callback
        let rt_status_for_process = rt_status.clone();

        // Oculta warnings do compilador de lifetime ou clonagem de referências do pw-stream
        capture_listener = capture_stream
            .add_local_listener::<()>()
            .state_changed(move |_stream, _user_data, old, new| match new {
                pw::stream::StreamState::Error(err) => {
                    log::error!("{} Falha grave na stream de áudio PW: {}", "💥".red(), err);
                }
                pw::stream::StreamState::Paused if old == pw::stream::StreamState::Streaming => {
                    log::info!("{} Áudio desconectado ou troca de nó.", "⏸️".yellow());
                }
                pw::stream::StreamState::Streaming if old == pw::stream::StreamState::Paused => {
                    log::info!("{} Áudio capturado (conexão estabelecida)", "▶️".green());
                }
                _ => {}
            })
            .param_changed(move |_stream, _user_data, id, param| {
                let Some(param) = param else { return };
                if id != pw::spa::param::ParamType::Format.as_raw() {
                    return;
                }

                let (media_type, media_subtype) =
                    match pw::spa::param::format_utils::parse_format(param) {
                        Ok(v) => v,
                        Err(_) => return,
                    };

                if media_type != pw::spa::param::format::MediaType::Audio
                    || media_subtype != pw::spa::param::format::MediaSubtype::Raw
                {
                    return;
                }

                let mut format = pw::spa::param::audio::AudioInfoRaw::default();
                if format.parse(param).is_ok() {
                    let rate = format.rate();
                    rate_for_param.store(rate, std::sync::atomic::Ordering::Relaxed);
                }
            })
            .process(move |stream: &pw::stream::Stream, _info| {
                // =========================================================
                // CALLBACK RT — ZERO ALOCAÇÕES, ZERO I/O
                // =========================================================
                // Toda alocação (NamResampler::new, Vec, Box) e toda operação
                // de I/O (println!, eprintln!, write!) são PROIBIDAS neste escopo.
                // Status comunicado via flags atômicas em `rt_status_for_process`.

                // Executa no kernel da thread RT (Data Thread do Pipewire)
                if !thread_configured {
                    rt_setup::configure_realtime_thread(target_cpu, rt_status_for_process.clone());
                    thread_configured = true;
                }

                // Drena resamplers pré-construídos pela thread principal (zero-alloc swap).
                // O Drop do resampler antigo é aceitável aqui (~50ns free(), evento raro).
                while let Ok(new_rs) = resampler_consumer.pop() {
                    // Reporta rate ativo via flag atômica (sem println!)
                    rt_status_for_process
                        .active_rate
                        .store(new_rs.pw_rate(), Ordering::Relaxed);
                    resampler = new_rs;
                }

                // Drena parâmetros guiados da thread CLI (Lock-Free) via SPSC Ring Buffer
                let mut param_changed = false;
                while let Ok(payload) = consumer.pop() {
                    match payload {
                        ParamPayload::LoadModel {
                            model_l,
                            model_r,
                            input_mult_adj,
                            output_mult_adj,
                            sample_rate,
                        } => {
                            let new_model_l = model_l;
                            let new_model_r = model_r;
                            if new_model_l.is_some() || new_model_r.is_some() {
                                model_input_mult_adj = input_mult_adj;
                                model_output_mult_adj = output_mult_adj;
                                current_nam_rate = sample_rate;
                            } else {
                                model_input_mult_adj = 1.0;
                                model_output_mult_adj = 1.0;
                                current_nam_rate = 48_000;
                            }

                            if let Some(old) = std::mem::replace(&mut active_model_l, new_model_l) {
                                let _ = gc_producer.push(old);
                            }
                            if let Some(old) = std::mem::replace(&mut active_model_r, new_model_r) {
                                let _ = gc_producer.push(old);
                            }
                            param_changed = true;
                        }
                        ParamPayload::InputGain(mult) => {
                            user_input_gain_mult = mult;
                            param_changed = true;
                        }
                        ParamPayload::OutputGain(mult) => {
                            user_output_gain_mult = mult;
                            param_changed = true;
                        }
                        ParamPayload::GateConfig(params) => {
                            // Pré-calcula thresholds lineares² (cold-path, evento raro).
                            // Usa GainLUT interpolada (~4 ciclos) em vez de powf (~200+ ciclos).
                            let open_lin = lut.db_to_linear(params.threshold_open_db);
                            let close_lin = lut.db_to_linear(params.threshold_close_db);
                            threshold_open_sq = open_lin * open_lin;
                            threshold_close_sq = close_lin * close_lin;
                            gate_params = params;
                        }
                    }
                }

                // Verifica se o PW ou a topologia solicitou mudança na taxa de amostragem.
                let detected_pw_rate = rate_for_process.swap(0, Ordering::Relaxed);
                let current_pw_rate = resampler.pw_rate();

                let mut pw_rate_to_request = current_pw_rate;
                let mut requires_rebuild = false;

                if detected_pw_rate != 0 && detected_pw_rate != current_pw_rate {
                    pw_rate_to_request = detected_pw_rate;
                    requires_rebuild = true;
                }

                if current_nam_rate != resampler.nam_rate() {
                    requires_rebuild = true;
                }

                if requires_rebuild && pw_rate_to_request != 0 {
                    rt_status_for_process
                        .requested_pw_rate
                        .store(pw_rate_to_request, Ordering::Relaxed);
                    rt_status_for_process
                        .requested_nam_rate
                        .store(current_nam_rate, Ordering::Relaxed);
                    rt_status_for_process
                        .needs_resampler_rebuild
                        .store(true, Ordering::Relaxed);
                }

                if param_changed {
                    rt_setup::compute_gain_multipliers(
                        user_input_gain_mult,
                        user_output_gain_mult,
                        model_input_mult_adj,
                        model_output_mult_adj,
                        &mut input_gain_mult,
                        &mut output_gain_mult,
                    );
                }

                // =========================================================
                // LÓGICA DSP - TEMPO REAL
                // =========================================================
                // Recupera o Buffer do PipeWire alocado internamente contendo os canais.
                // Recupera o Buffer do PipeWire contendo as amostras de áudio.
                let mut _buf = match stream.dequeue_buffer() {
                    Some(b) => b,
                    None => return,
                };

                // 1. Extração dos buffers de áudio do PipeWire
                // O PipeWire fornece buffers planares (F32P). Cada canal tem seu próprio array de dados.
                let datas = _buf.datas_mut();
                if datas.len() >= 2 {
                    // Dividimos o slice de metadados em Esquerdo e Direito para obter referências mutáveis únicas.
                    let (left_datas, right_datas) = datas.split_at_mut(1);
                    if let (Some(d_l), Some(d_r)) =
                        (left_datas.first_mut(), right_datas.first_mut())
                    {
                        // Cada canal possui um 'chunk' que descreve a porção válida do buffer (offset e size).
                        let chunk_l = d_l.chunk();
                        let chunk_r = d_r.chunk();
                        let offset_l = chunk_l.offset() as usize;
                        let size_l = chunk_l.size() as usize;
                        let offset_r = chunk_r.offset() as usize;
                        let size_r = chunk_r.size() as usize;

                        // d.data() retorna o buffer bruto de bytes mapeado pelo kernel.
                        if let (Some(raw_l), Some(raw_r)) = (d_l.data(), d_r.data()) {
                            // Calculamos o número de amostras (f32) disponíveis, respeitando os limites
                            // físicos do buffer e a saturação para evitar overflows em cálculos de índice.
                            let n_bytes_l = size_l.min(raw_l.len().saturating_sub(offset_l));
                            let n_bytes_r = size_r.min(raw_r.len().saturating_sub(offset_r));
                            let n_samples_l = n_bytes_l / std::mem::size_of::<f32>();
                            let n_samples_r = n_bytes_r / std::mem::size_of::<f32>();

                            // Sincronizamos o tamanho do processamento pelo menor buffer disponível.
                            let n_samples = n_samples_l.min(n_samples_r);

                            if n_samples > 0 {
                                // SAFETY: O PipeWire garante que os buffers mmap permanecem válidos durante
                                // a execução do callback RT. Castamos para f32 pois negociamos F32P (float planar).
                                // Criamos slices diretamente sobre a memória do PipeWire para evitar cópias extras (zero-copy).
                                let samples_l = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        raw_l.as_mut_ptr().add(offset_l).cast::<f32>(),
                                        n_samples,
                                    )
                                };
                                let samples_r = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        raw_r.as_mut_ptr().add(offset_r).cast::<f32>(),
                                        n_samples,
                                    )
                                };

                                // Iniciamos a medição do tempo de processamento DSP puro (RDTSC).
                                // Usado para calcular o 'DSP Load' e detectar XRuns iminentes.
                                let start_time = Instant::now();

                                capture_dsp_pipeline(
                                    samples_l,
                                    samples_r,
                                    n_samples,
                                    DspPipelineContext {
                                        resampler: &mut resampler,
                                        active_model_l: &mut active_model_l,
                                        active_model_r: &mut active_model_r,
                                        input_gain_mult,
                                        output_gain_mult,
                                        gate_params: &gate_params,
                                        silence_hysteresis: &mut silence_hysteresis,
                                        mono_hysteresis: &mut mono_hysteresis,
                                        threshold_open_sq,
                                        threshold_close_sq,
                                        process_mono: &mut process_mono,
                                        rt_status: &rt_status_for_process,
                                        bridge_ptr,
                                        resamp_mid_l: &mut resamp_mid_l,
                                        resamp_mid_r: &mut resamp_mid_r,
                                        resamp_out_l: &mut resamp_out_l,
                                        resamp_out_r: &mut resamp_out_r,
                                    },
                                );

                                // Monitoramento de carga de DSP (DSP Load Monitoring via RDTSC)
                                // `start_time` é inicializado no início da pipeline DSP (acima).
                                // Reportamos o tempo bruto em nanos para a thread de controle.
                                let elapsed = start_time.elapsed();
                                rt_status_for_process
                                    .dsp_cycle_time
                                    .store(elapsed.as_nanos() as u64, Ordering::Relaxed);
                                rt_status_for_process
                                    .last_n_samples
                                    .store(n_samples as u32, Ordering::Relaxed);

                                // Calcula o budget máximo tolerável (85% do tempo real do buffer)
                                // Ainda usamos uma aproximação escalar aqui para detecção rápida de overload,
                                // mas a telemetria precisa via Anchor ocorre no poll_rt_status.
                                let elapsed_secs = elapsed.as_secs_f64();
                                let budget_secs =
                                    (n_samples as f64 / current_pw_rate as f64) * 0.85;
                                if elapsed_secs > budget_secs {
                                    rt_status_for_process
                                        .dsp_overloads
                                        .fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                    }
                }
                // Buffer devolvido automaticamente ao PipeWire no `Drop`.
            })
            .register()?;

        // Constrói o SPA Pod F32P stereo para negociação de formato com PipeWire.
        let mut audio_info = pw::spa::param::audio::AudioInfoRaw::new();
        audio_info.set_format(pw::spa::param::audio::AudioFormat::F32P);
        audio_info.set_channels(2);

        let mut format_buf = [0u8; 1024];
        let format_pod = unsafe { build_spa_format_pod(&audio_info, &mut format_buf)? };

        // Ativa a capture stream com formato de áudio declarado.
        // Direction::Input — recebe áudio dos apps (Virtual Sink).
        capture_stream.connect(
            pw::spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS
                | pw::stream::StreamFlags::EXCLUSIVE,
            &mut [format_pod],
        )?;

        log::info!(
            "{} Capture stream conectado ao PipeWire (Audio/Sink, F32P Planar Stereo).",
            "🎼".bright_blue()
        );

        // =========================================================
        // PLAYBACK STREAM: envia áudio processado para o hardware
        // =========================================================
        // Ponteiro raw para o bridge, compartilhado com o playback callback.
        let bridge_ptr_playback = bridge_ptr;

        let hardware_target = rt_setup::detect_hardware_sink();

        let mut playback_props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::MEDIA_CLASS => "Stream/Output/Audio",
            *pw::keys::NODE_NAME => "NAM-rs-playback",
            *pw::keys::NODE_DESCRIPTION => "NAM-rs Processed Output",
            "audio.position" => "FL,FR",
            "node.group" => "nam-rs-dsp", // Clock sync com capture stream
            "node.link-group" => "nam-rs-link-group", // Evita feedback loop (WirePlumber não linka nós do mesmo link-group)
        };

        if let Some(target) = hardware_target {
            playback_props.insert("node.target", target.as_str());
            log::info!(
                "{} Saída roteada automaticamente para o hardware padrão: {}",
                "🔌".green(),
                target
            );
        }

        if buffer_size > 0 {
            playback_props.insert("node.latency", latency_str.as_str());
        }

        playback_stream = pw::stream::StreamBox::new(&core, "NAM-rs-Output", playback_props)?;

        // Contador de geração local para detectar novos dados do bridge.
        let mut last_bridge_gen: u64 = 0;

        playback_listener = playback_stream
            .add_local_listener::<()>()
            .process(move |stream: &pw::stream::Stream, _info| {
                // Callback RT do playback: lê do DspBridge e copia para o output buffer.
                // SAFETY: `bridge_ptr_playback` aponta para `DspBridge` alocado via
                // `Box::leak` (vive até o shutdown). O playback callback é o único
                // leitor dos arrays `buf_l`/`buf_r` após `fence(Acquire)`.
                let bridge_ref = unsafe { &*bridge_ptr_playback };

                // Verifica se há dados novos do capture callback.
                let current_gen = bridge_ref.generation.load(Ordering::Relaxed);
                if current_gen == last_bridge_gen {
                    return; // Nenhum dado novo — pula este ciclo.
                }
                // 1. Sincronização e Acesso ao Bridge
                // Utilizamos fence(Acquire) para garantir que todas as escritas realizadas pelo
                // capture callback (Release) sejam visíveis nesta thread.
                last_bridge_gen = current_gen;
                std::sync::atomic::fence(Ordering::Acquire);

                // Obtemos o índice do buffer que acabou de ser preenchido pelo DSP.
                let read_idx = bridge_ref.active_read_idx.load(Ordering::Relaxed);
                let front_buf = &bridge_ref.buffers[read_idx];

                let n_samples = front_buf.n_samples as usize;
                if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
                    return;
                }

                // 2. Recuperação do Buffer de Saída do PipeWire
                let mut buf = match stream.dequeue_buffer() {
                    Some(b) => b,
                    None => return,
                };

                let datas = buf.datas_mut();
                if datas.len() < 2 {
                    return;
                }

                // Dividimos o slice para satisfazer as regras de mutabilidade única do Rust (borrow checker).
                let (datas_left, datas_right) = datas.split_at_mut(1);
                let data_l = &mut datas_left[0];
                let data_r = &mut datas_right[0];

                // Calculamos a capacidade máxima do buffer do hardware para evitar overflow.
                let max_l = data_l.as_raw().maxsize as usize / std::mem::size_of::<f32>();
                let max_r = data_r.as_raw().maxsize as usize / std::mem::size_of::<f32>();
                let n_out = n_samples.min(max_l).min(max_r);
                if n_out == 0 {
                    return;
                }

                // 3. Transferência de Áudio (Bridge → Hardware)
                // Copiamos os dados processados do DspBridge diretamente para os buffers de saída.
                if let Some(raw_l) = data_l.data() {
                    // SAFETY: `raw_l` é memória mapeada válida durante o ciclo do callback.
                    // O formato F32P garante que os dados são floats de 32 bits planares.
                    let out_l = unsafe {
                        std::slice::from_raw_parts_mut(raw_l.as_mut_ptr().cast::<f32>(), n_out)
                    };
                    out_l.copy_from_slice(&front_buf.buf_l[..n_out]);
                }
                if let Some(raw_r) = data_r.data() {
                    // SAFETY: Idem para o canal direito.
                    let out_r = unsafe {
                        std::slice::from_raw_parts_mut(raw_r.as_mut_ptr().cast::<f32>(), n_out)
                    };
                    out_r.copy_from_slice(&front_buf.buf_r[..n_out]);
                }

                // 4. Atualização de Metadados (Chunks)
                // Informamos ao PipeWire a quantidade exata de dados escritos e a estrutura do buffer.
                {
                    let chunk = data_l.chunk_mut();
                    *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
                    *chunk.offset_mut() = 0;
                    *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
                }
                {
                    let chunk = data_r.chunk_mut();
                    *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
                    *chunk.offset_mut() = 0;
                    *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
                }
            })
            .register()?;

        // Constrói SPA Pod F32P stereo para a playback stream (mesmo formato da capture).
        let mut playback_audio_info = pw::spa::param::audio::AudioInfoRaw::new();
        playback_audio_info.set_format(pw::spa::param::audio::AudioFormat::F32P);
        playback_audio_info.set_channels(2);

        let mut playback_format_buf = [0u8; 1024];
        let playback_format_pod =
            unsafe { build_spa_format_pod(&playback_audio_info, &mut playback_format_buf)? };

        // Ativa a playback stream — Direction::Output envia para o hardware.
        playback_stream.connect(
            pw::spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut [playback_format_pod],
        )?;

        log::info!(
            "{} Playback stream conectado ao PipeWire (Stream/Output/Audio, F32P Planar Stereo).",
            "🔊".bright_blue()
        );
    }

    let _app_state = AppState {
        capture_stream,
        capture_listener,
        playback_stream,
        playback_listener,
    };

    let _cpu_dma_lock = rt_setup::lock_cpu_c_states();

    // Emitido aqui (após PM QoS e antes do thread_loop.start()) para agrupar
    // todos os diagnósticos de hardening RT na saída do console.
    sys.emit_irq_advisory(target_cpu);

    // Inicia a execução da ThreadLoop em background, com PipeWire assumindo a Thread de RT
    thread_loop.start();

    // =========================================================
    // LOOP DE CONTROLE (NON-RT)
    // =========================================================
    // Este loop executa na thread principal da aplicação. Sua função é gerenciar
    // tarefas pesadas (alocações, I/O) que são proibidas dentro dos callbacks RT.
    let mut was_silent = false;
    while !SHUTDOWN.load(Ordering::Relaxed) {
        // 1. Gestão Dinâmica de Resampling
        // O callback RT sinaliza via flag atômica se houve mudança na taxa de amostragem
        // do PipeWire ou do modelo carregado.
        if rt_status.needs_resampler_rebuild.load(Ordering::Relaxed) {
            let target_pw_rate = rt_status.requested_pw_rate.load(Ordering::Relaxed);
            let target_nam_rate = rt_status.requested_nam_rate.load(Ordering::Relaxed);

            if target_pw_rate != 0 && target_nam_rate != 0 {
                // Reconstruímos o resampler aqui (Non-RT). A alocação de heap é permitida.
                // Isso garante que o áudio não sofra dropouts durante a troca de formato.
                match NamResampler::new(target_pw_rate, target_nam_rate, 2048) {
                    Ok(new_rs) => {
                        rt_status
                            .resampler_rebuild_failed
                            .store(false, Ordering::Relaxed);

                        log::info!(
                            "{} Sample rate atualizado: PW={} Hz, NAM={} Hz (bypass={})",
                            "🔄".cyan(),
                            target_pw_rate,
                            target_nam_rate,
                            new_rs.is_bypass()
                        );

                        // Enviamos o novo objeto para a thread RT via canal SPSC (Lock-Free).
                        if resampler_producer.push(new_rs).is_err() {
                            NamDiagnostic::new(NamErrorCode::ResamplerChannelFull, &sys)
                                .message("Canal de resampler cheio. Rebuild descartado.")
                                .hint(
                                    "O motor de áudio está sobrecarregado. \
                                     Se o problema persistir, reinicie o NAM-rs.",
                                )
                                .param("target_pw_rate", target_pw_rate)
                                .param("target_nam_rate", target_nam_rate)
                                .emit_warning();
                        }
                    }
                    Err(e) => {
                        // Se o rebuild falhar, emitimos um diagnóstico e mantemos o resampler anterior.
                        NamDiagnostic::new(NamErrorCode::ResamplerBuildFailed, &sys)
                            .message(format!(
                                "Falha ao reconstruir o resampler para PW={} Hz e NAM={} Hz.",
                                target_pw_rate, target_nam_rate
                            ))
                            .hint(
                                "O áudio continuará com o resampler anterior. \
                                 Se a taxa de amostragem estiver errada, reinicie o NAM-rs.",
                            )
                            .param("target_pw_rate", target_pw_rate)
                            .param("target_nam_rate", target_nam_rate)
                            .param("detail", &e)
                            .emit();

                        rt_status
                            .resampler_rebuild_failed
                            .store(true, Ordering::Relaxed);
                    }
                }
                // Limpamos a solicitação após o processamento.
                rt_status
                    .needs_resampler_rebuild
                    .store(false, Ordering::Relaxed);
            }
        }

        // 2. Monitoramento de Status
        // Consultamos as flags atômicas do callback RT para atualizar a UI/Logs (clipping, silêncio, timing).
        was_silent = rt_setup::poll_rt_status(&rt_status, &sys, was_silent, &tsc_anchor);

        // Baixa frequência de polling para economizar energia, já que estas são tarefas de controle.
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // 3. Graceful Shutdown
    // Paramos o loop do PipeWire antes de encerrar a função para garantir que
    // os recursos sejam liberados corretamente.
    thread_loop.stop();

    Ok(())
}

// =============================================================================
// Helpers Extraídos — Redução de Complexidade do `run_pipewire_host`
// =============================================================================

/// Constrói um SPA Pod de formato de áudio F32P stereo para negociação PipeWire.
///
/// O PipeWire utiliza o sistema SPA (Signal Processing Algorithms) para descrever formatos.
/// Esta função serializa a estrutura `audio_info` em um pod binário usando um builder manual.
/// Isso é necessário para garantir a negociação correta do formato planar (F32P) exigido pelo motor.
///
/// # Safety
///
/// O pod binário retornado aponta diretamente para o `format_buf` fornecido (zero-copy).
/// Portanto, o `format_buf` DEVE permanecer válido e não ser modificado enquanto o pod
/// estiver sendo utilizado para configurar a stream.
unsafe fn build_spa_format_pod<'a>(
    audio_info: &pw::spa::param::audio::AudioInfoRaw,
    format_buf: &'a mut [u8; 1024],
) -> anyhow::Result<&'a pw::spa::pod::Pod> {
    unsafe {
        // Inicializa a estrutura do builder com zeros (FFI requirement)
        let mut builder: pw::spa::sys::spa_pod_builder = std::mem::zeroed();

        // Vincula o builder ao nosso buffer de memória pré-alocado
        pw::spa::sys::spa_pod_builder_init(
            &mut builder,
            format_buf.as_mut_ptr().cast(),
            format_buf.len() as u32,
        );

        // Serializa a info de áudio no formato binário SPA Pod (F32P Stereo)
        let pod_ptr = pw::spa::sys::spa_format_audio_raw_build(
            &mut builder,
            pw::spa::param::ParamType::EnumFormat.as_raw(),
            &audio_info.as_raw(),
        );

        // Verifica se a serialização falhou (ex: buffer muito pequeno)
        if pod_ptr.is_null() {
            return Err(anyhow::anyhow!(
                "Falha ao construir SPA Pod de formato de áudio"
            ));
        }

        // Converte o ponteiro bruto de volta para uma referência de alto nível do pipewire-rs
        Ok(&*(pod_ptr as *const pw::spa::pod::Pod))
    }
}

// =============================================================================
// Estágios do Pipeline DSP (Modularizados e Inlinados)
// =============================================================================

/// Estágio 1: Gate, Ganhos de Entrada e Detecção de Mono.
#[inline(always)]
fn apply_input_stage(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> GateState {
    // 1. Detecção de Silêncio com Histerese (Silence Bypass)
    let energy_ms = unsafe { compute_energy_avx2(&samples_l[..n_samples]) };

    ctx.silence_hysteresis.update(
        energy_ms,
        ctx.threshold_open_sq,
        ctx.threshold_close_sq,
        ctx.gate_params,
        n_samples,
    );

    if ctx.silence_hysteresis.state() == GateState::Closed {
        return GateState::Closed;
    }

    // 2. Mitigação Mono com Histerese (Otimização Dinâmica)
    let max_diff =
        unsafe { compute_max_diff_avx2(&samples_l[..n_samples], &samples_r[..n_samples]) };

    ctx.mono_hysteresis.update(
        max_diff,
        ctx.gate_params.mono_epsilon,
        ctx.gate_params.mono_epsilon * 0.9,
        ctx.gate_params,
        n_samples,
    );

    *ctx.process_mono = ctx.mono_hysteresis.state() == GateState::Closed
        || ctx.mono_hysteresis.state() == GateState::FadingOut;

    // 3. Pipeline DSP: Ganho de Entrada
    crate::dsp::gain::apply_gain_simd(&mut samples_l[..n_samples], ctx.input_gain_mult);
    if !*ctx.process_mono {
        crate::dsp::gain::apply_gain_simd(&mut samples_r[..n_samples], ctx.input_gain_mult);
    }

    ctx.silence_hysteresis.state()
}

/// Estágio 2: Inferência Neural e Resampling.
#[inline(always)]
fn run_inference(
    samples_l: &[f32],
    samples_r: &[f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> usize {
    let is_resamp_bypass = ctx.resampler.is_bypass();
    let n = n_samples.min(MAX_RESAMP_BUF);

    if is_resamp_bypass {
        let model_in_l = &samples_l[..n];
        let model_in_r = if *ctx.process_mono {
            &samples_l[..n]
        } else {
            &samples_r[..n]
        };
        let model_out_l = &mut ctx.resamp_out_l[..n];
        let model_out_r = &mut ctx.resamp_out_r[..n];

        if let Some(model_l) = ctx.active_model_l {
            model_l.process(model_in_l, model_out_l);
        }

        if *ctx.process_mono {
            model_out_r.copy_from_slice(model_out_l);
        } else if let Some(model_r) = ctx.active_model_r {
            model_r.process(model_in_r, model_out_r);
        }

        n
    } else {
        let n_48k = ctx.resampler.process_input(
            &samples_l[..n],
            if *ctx.process_mono {
                &samples_l[..n]
            } else {
                &samples_r[..n]
            },
            &mut ctx.resamp_mid_l[..MAX_RESAMP_BUF],
            &mut ctx.resamp_mid_r[..MAX_RESAMP_BUF],
        );

        let mut temp_out_l = [0.0f32; MAX_RESAMP_BUF];
        let mut temp_out_r = [0.0f32; MAX_RESAMP_BUF];

        let model_in_l = &ctx.resamp_mid_l[..n_48k];
        let model_in_r = &ctx.resamp_mid_r[..n_48k];
        let model_out_l = &mut temp_out_l[..n_48k];
        let model_out_r = &mut temp_out_r[..n_48k];

        if let Some(model_l) = ctx.active_model_l {
            model_l.process(model_in_l, model_out_l);
        }

        if *ctx.process_mono {
            model_out_r.copy_from_slice(model_out_l);
        } else if let Some(model_r) = ctx.active_model_r {
            model_r.process(model_in_r, model_out_r);
        }

        ctx.resampler.process_output(
            model_out_l,
            model_out_r,
            &mut ctx.resamp_out_l[..MAX_RESAMP_BUF],
            &mut ctx.resamp_out_r[..MAX_RESAMP_BUF],
        )
    }
}

/// Estágio 3: Ganho de Saída, Fading e Detecção de Clipping.
#[inline(always)]
fn apply_output_stage(
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    n_pw: usize,
    output_gain_mult: f32,
    gate_params: &GateParams,
    silence_hysteresis: &mut DynamicHysteresis,
    rt_status: &RtStatusFlags,
) {
    crate::dsp::gain::apply_gain_simd(&mut resamp_out_l[..n_pw], output_gain_mult);
    crate::dsp::gain::apply_gain_simd(&mut resamp_out_r[..n_pw], output_gain_mult);

    silence_hysteresis.apply_gain_rt(&mut resamp_out_l[..n_pw], gate_params, n_pw);
    silence_hysteresis.apply_gain_rt(&mut resamp_out_r[..n_pw], gate_params, n_pw);

    if crate::dsp::gain::detect_clipping_stereo_simd(&resamp_out_l[..n_pw], &resamp_out_r[..n_pw]) {
        rt_status.has_clipped.store(true, Ordering::Relaxed);
    }
}

/// Estágio 4: Escrita no DspBridge.
#[inline(always)]
fn write_bridge(
    resamp_out_l: &[f32],
    resamp_out_r: &[f32],
    n_pw: usize,
    bridge_ptr: *mut DspBridge,
) {
    let bridge_ref = unsafe { &mut *bridge_ptr };
    let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
    let back_buf = &mut bridge_ref.buffers[back_idx];

    let n_bridge = n_pw.min(MAX_BRIDGE_BUF);
    unsafe {
        core::ptr::copy_nonoverlapping(
            resamp_out_l.as_ptr(),
            back_buf.buf_l.as_mut_ptr(),
            n_bridge,
        );
        core::ptr::copy_nonoverlapping(
            resamp_out_r.as_ptr(),
            back_buf.buf_r.as_mut_ptr(),
            n_bridge,
        );
    }
    back_buf.n_samples = n_bridge as u32;

    std::sync::atomic::fence(Ordering::Release);
    bridge_ref
        .active_read_idx
        .store(back_idx, Ordering::Relaxed);
    bridge_ref.generation.fetch_add(1, Ordering::Relaxed);
}

/// Pipeline DSP Completo (Agregador).
#[inline(always)]
fn capture_dsp_pipeline(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    mut ctx: DspPipelineContext<'_>,
) {
    let gate_state = apply_input_stage(samples_l, samples_r, n_samples, &mut ctx);

    if gate_state == GateState::Closed {
        handle_silence_bypass(ctx.bridge_ptr, ctx.rt_status);
        return;
    }

    ctx.rt_status
        .is_silent
        .store(gate_state != GateState::Open, Ordering::Relaxed);

    let n_pw = run_inference(samples_l, samples_r, n_samples, &mut ctx);

    apply_output_stage(
        ctx.resamp_out_l,
        ctx.resamp_out_r,
        n_pw,
        ctx.output_gain_mult,
        ctx.gate_params,
        ctx.silence_hysteresis,
        ctx.rt_status,
    );

    write_bridge(ctx.resamp_out_l, ctx.resamp_out_r, n_pw, ctx.bridge_ptr);
}

#[cfg(test)]
#[path = "pw_host_test.rs"]
mod pw_host_test;
