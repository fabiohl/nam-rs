// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Núcleo de processamento de áudio DSP usando `pipewire-rs`.
//!
//! Este é o "coração" do NAM-rs: o módulo que de fato processa o sinal de áudio em
//! tempo real. Ele recebe amostras brutas de áudio do PipeWire (o servidor de som
//! do Linux), passa pelo amplificador neural, e entrega o resultado processado
//! ao hardware via arquitetura dual-stream.
//!
//! ## Arquitetura Dual-Stream com DspBridge
//!
//! O PipeWire copia os buffers para o monitor port **antes** de chamar o `process()`.
//! Portanto, modificações in-place numa única stream `Audio/Sink` seriam invisíveis
//! ao hardware. A solução usa duas streams:
//!
//! 1. **Capture stream** (`Audio/Sink`, Direction::Input) — recebe áudio dos apps,
//!    aplica DSP (ganho + inferência neural), escreve no [`DspBridge`].
//! 2. **Playback stream** (`Stream/Output/Audio`, Direction::Output) — lê do
//!    [`DspBridge`] e entrega ao hardware.
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
//! 1. Aplica ganho de entrada SIMD (ajuste de volume pré-amplificador)
//! 2. `NamResampler::process_input()` — converte o sample rate do PipeWire para 48 kHz
//! 3. Inferência neural WaveNet/LSTM — o "amplificador virtual" que processa o som
//! 4. `NamResampler::process_output()` — converte de volta para o sample rate original
//! 5. Aplica ganho de saída SIMD (ajuste de volume pós-amplificador)
//! 6. Escreve resultado no [`DspBridge`] com `fence(Release)` para o playback callback
//!
//! Quando nenhum modelo está carregado, opera em pass-through (o som entra e sai sem alteração).
//! Quando o sample rate do PipeWire já é 48 kHz, o resampler opera em bypass sem overhead.

use crate::colors::Colorize;
use crate::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::dsp::gain::{apply_gain_simd, is_buffer_mono_simd, is_buffer_silent_stereo_simd};
use crate::dsp::resampler::NamResampler;
use crate::spsc::{ParamPayload, RtStatusFlags, SHUTDOWN};
use pipewire as pw;
use pw::properties::properties;
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Tamanho máximo do buffer intermediário entre as duas streams (capture → playback).
/// Dimensionado para o quantum máximo do PipeWire (`max-quantum = 8192`).
const MAX_BRIDGE_BUF: usize = 8192;

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
struct DspBridge {
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

/// Silence Bypass: sinaliza silêncio e zera o bridge para que o playback emita silêncio.
///
/// Marcada como `#[cold]` para sinalizar ao LLVM que este é o path improvável (o músico
/// está tocando, portanto há sinal na maioria dos callbacks). Isso permite ao compilador
/// posicionar este código fora da região quente de I-Cache do `process()`.
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
/// Arquitetura: Apps → [Capture Stream: Audio/Sink] → process(DSP) → DspBridge → [Playback Stream] → Hardware.
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
    sys: SystemSnapshot,
    buffer_size: u32,
) -> anyhow::Result<()> {
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

    unsafe {
        libc::madvise(
            bridge_ptr as *mut libc::c_void,
            std::mem::size_of::<DspBridge>(),
            libc::MADV_DONTFORK | libc::MADV_DONTDUMP,
        );
    }

    // Seleciona o núcleo ideal para a thread RT (capacidade + menor carga IRQ).
    // Computado antes do lock para que o valor seja acessível após o escopo.
    let target_cpu = select_optimal_cpu().unwrap_or(0);

    // Obtém o lock para o loop, necessário para criar e configurar streams do PW com segurança
    // O lock é automaticamente liberado ao final deste escopo (_lock is dropped).
    {
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

        let latency_str = format!("{}/48000", buffer_size);
        if buffer_size > 0 {
            capture_props.insert("node.latency", latency_str.as_str());
        }

        capture_stream = pw::stream::StreamBox::new(&core, "NAM-rs", capture_props)?;

        let mut active_model_l: Option<Box<crate::models::DynamicModel>> = None;
        let mut active_model_r: Option<Box<crate::models::DynamicModel>> = None;

        // NamResampler bidirecional: converte entre o rate do PipeWire e os 48 kHz do NAM.
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

        // Trackeia a taxa de amostragem alvo do amplificador virtual no loop RT
        let mut current_nam_rate: u32 = 48_000;

        // Buffer intermediário 48k para saída do resampler (stack-allocated).
        // MAX_RESAMP_BUF comporta a expansão do resampler (ex: 44100→48000 = ~1.088x).
        const MAX_RESAMP_BUF: usize = 4096;
        let mut resamp_mid_l = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_l = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_mid_r = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0f32; MAX_RESAMP_BUF];

        // Variáveis de ganho em decibéis
        let mut user_input_gain_db: f32 = 0.0;
        let mut user_output_gain_db: f32 = 0.0;
        let mut model_input_db_adj: f32 = 0.0;
        let mut model_output_db_adj: f32 = 0.0;

        // Multiplicadores pré-calculados lineares para inserção no DSP FMA Simd
        let mut input_gain_mult: f32 = 1.0;
        let mut output_gain_mult: f32 = 1.0;

        // Função local para recomputar os fatores lineares
        let update_gain_multipliers =
            |u_in: f32,
             u_out: f32,
             m_in: f32,
             m_out: f32,
             out_in_mult: &mut f32,
             out_out_mult: &mut f32| {
                let total_in_db = u_in + m_in;
                *out_in_mult = 10.0f32.powf(total_in_db / 20.0);

                let total_out_db = u_out + m_out;
                *out_out_mult = 10.0f32.powf(total_out_db / 20.0);
            };

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
                    configure_realtime_thread(target_cpu, rt_status_for_process.clone());
                    thread_configured = true;
                }

                let start_time = std::time::Instant::now();

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
                            input_db_adj,
                            output_db_adj,
                            sample_rate,
                        } => {
                            let new_model_l = model_l;
                            let new_model_r = model_r;
                            if new_model_l.is_some() || new_model_r.is_some() {
                                model_input_db_adj = input_db_adj;
                                model_output_db_adj = output_db_adj;
                                current_nam_rate = sample_rate;
                            } else {
                                model_input_db_adj = 0.0;
                                model_output_db_adj = 0.0;
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
                        ParamPayload::InputGain(gain_db) => {
                            user_input_gain_db = gain_db;
                            param_changed = true;
                        }
                        ParamPayload::OutputGain(gain_db) => {
                            user_output_gain_db = gain_db;
                            param_changed = true;
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
                    update_gain_multipliers(
                        user_input_gain_db,
                        user_output_gain_db,
                        model_input_db_adj,
                        model_output_db_adj,
                        &mut input_gain_mult,
                        &mut output_gain_mult,
                    );
                }

                // =========================================================
                // LÓGICA DSP - TEMPO REAL
                // =========================================================
                // Recupera o Buffer do PipeWire alocado internamente contendo os canais.
                // Numa stream Filter, um `dequeue_buffer` entrega dados de In e Out conjugados.
                let mut _buf = match stream.dequeue_buffer() {
                    Some(b) => b,
                    None => return,
                };

                let datas = _buf.datas_mut();
                if datas.len() >= 2 {
                    let (left_datas, right_datas) = datas.split_at_mut(1);
                    if let (Some(d_l), Some(d_r)) =
                        (left_datas.first_mut(), right_datas.first_mut())
                    {
                        let chunk_l = d_l.chunk();
                        let chunk_r = d_r.chunk();
                        let offset_l = chunk_l.offset() as usize;
                        let size_l = chunk_l.size() as usize;
                        let offset_r = chunk_r.offset() as usize;
                        let size_r = chunk_r.size() as usize;

                        if let (Some(raw_l), Some(raw_r)) = (d_l.data(), d_r.data()) {
                            let n_bytes_l = size_l.min(raw_l.len().saturating_sub(offset_l));
                            let n_bytes_r = size_r.min(raw_r.len().saturating_sub(offset_r));
                            let n_samples_l = n_bytes_l / std::mem::size_of::<f32>();
                            let n_samples_r = n_bytes_r / std::mem::size_of::<f32>();
                            let n_samples = n_samples_l.min(n_samples_r);

                            if n_samples > 0 {
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

                                if is_buffer_silent_stereo_simd(
                                    &samples_l[..n_samples],
                                    &samples_r[..n_samples],
                                ) {
                                    handle_silence_bypass(bridge_ptr, &rt_status_for_process);
                                    return;
                                }
                                // Se chegou aqui, há sinal — reseta flag de silêncio
                                rt_status_for_process
                                    .is_silent
                                    .store(false, Ordering::Relaxed);

                                // Mitigação: se o R for puramente zero ou exatamente igual ao L,
                                // processamos no modo Mono economizando 50% de CPU.
                                let process_mono = is_buffer_mono_simd(
                                    &samples_l[..n_samples],
                                    &samples_r[..n_samples],
                                );

                                // O resampler de entrada pode expandir amostras
                                // (ex: 44100→48000 = ratio ~1.088x), portanto os buffers
                                // intermediários de inferência usam MAX_RESAMP_BUF.
                                let mut temp_out_l = [0.0f32; MAX_RESAMP_BUF];
                                let mut temp_out_r = [0.0f32; MAX_RESAMP_BUF];
                                let n = n_samples.min(MAX_RESAMP_BUF);

                                // 1. Aplica ganho de entrada SIMD antes do resampling
                                apply_gain_simd(&mut samples_l[..n], input_gain_mult);
                                if !process_mono {
                                    apply_gain_simd(&mut samples_r[..n], input_gain_mult);
                                }

                                // 2. Downsample: PW_rate → 48 kHz
                                let n_48k = resampler.process_input(
                                    &samples_l[..n],
                                    if process_mono {
                                        &samples_l[..n]
                                    } else {
                                        &samples_r[..n]
                                    },
                                    &mut resamp_mid_l[..MAX_RESAMP_BUF],
                                    &mut resamp_mid_r[..MAX_RESAMP_BUF],
                                );

                                // 3. Inferência NAM a 48 kHz
                                if let Some(ref mut model_l) = active_model_l {
                                    model_l
                                        .0
                                        .process(&resamp_mid_l[..n_48k], &mut temp_out_l[..n_48k]);
                                }

                                if process_mono {
                                    // Copia a saída L para R
                                    temp_out_r[..n_48k].copy_from_slice(&temp_out_l[..n_48k]);
                                } else if let Some(ref mut model_r) = active_model_r {
                                    model_r
                                        .0
                                        .process(&resamp_mid_r[..n_48k], &mut temp_out_r[..n_48k]);
                                }

                                // 4. Upsample: 48 kHz → PW_rate
                                let n_pw = resampler.process_output(
                                    &temp_out_l[..n_48k],
                                    &temp_out_r[..n_48k],
                                    &mut resamp_out_l[..MAX_RESAMP_BUF],
                                    &mut resamp_out_r[..MAX_RESAMP_BUF],
                                );

                                // 5. Aplica ganho de saída SIMD após conversão de rate
                                apply_gain_simd(&mut resamp_out_l[..n_pw], output_gain_mult);
                                apply_gain_simd(&mut resamp_out_r[..n_pw], output_gain_mult);

                                // Copia resultado e detecta saturação
                                let n_copy = n_pw.min(n);
                                let out_slice_l = &resamp_out_l[..n_copy];
                                let out_slice_r = &resamp_out_r[..n_copy];

                                // Detecção de clipping via AVX2 vetorial (Item 6):
                                // 8 samples/iteração vs loop escalar (128 comparações).
                                if crate::dsp::gain::detect_clipping_stereo_simd(
                                    out_slice_l,
                                    out_slice_r,
                                ) {
                                    rt_status_for_process
                                        .has_clipped
                                        .store(true, Ordering::Relaxed);
                                }

                                // C.4: A cópia para samples_l/samples_r foi eliminada.
                                // Na arquitetura dual-stream, o monitor port do Audio/Sink copia
                                // o buffer *antes* do process() — portanto, escrever de volta em
                                // samples_l/samples_r é redundante. O áudio processado chega ao
                                // hardware exclusivamente via playback stream ← DspBridge.
                                // Ver TODO.txt para instruções de validação e reversão.

                                // =========================================================
                                // BRIDGE WRITE: copia resultado pós-DSP para o DspBridge
                                // O playback callback lê daqui via fence(Acquire).
                                // =========================================================
                                // SAFETY: `bridge_ptr` aponta para `DspBridge` alocado via
                                // `Box::leak` (vive até o shutdown). O capture callback é
                                // o único escritor dos arrays `buf_l`/`buf_r` — não há
                                // data race. Os campos atômicos são acessados via Ordering.
                                let bridge_ref = unsafe { &mut *bridge_ptr };
                                let back_idx =
                                    1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
                                let back_buf = &mut bridge_ref.buffers[back_idx];

                                let n_bridge = n_copy.min(MAX_BRIDGE_BUF);
                                back_buf.buf_l[..n_bridge]
                                    .copy_from_slice(&resamp_out_l[..n_bridge]);
                                back_buf.buf_r[..n_bridge]
                                    .copy_from_slice(&resamp_out_r[..n_bridge]);
                                back_buf.n_samples = n_bridge as u32;

                                std::sync::atomic::fence(Ordering::Release);
                                bridge_ref
                                    .active_read_idx
                                    .store(back_idx, Ordering::Relaxed);
                                bridge_ref.generation.fetch_add(1, Ordering::Relaxed);

                                // Monitoramento de carga de DSP (DSP Load Monitoring)
                                let elapsed = start_time.elapsed();
                                // Calcula o budget máximo tolerável (85% do tempo real do buffer)
                                let budget_secs =
                                    (n_samples as f64 / current_pw_rate as f64) * 0.85;
                                if elapsed.as_secs_f64() > budget_secs {
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

        let hardware_target = detect_hardware_sink();

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
                last_bridge_gen = current_gen;
                std::sync::atomic::fence(Ordering::Acquire);

                let read_idx = bridge_ref.active_read_idx.load(Ordering::Relaxed);
                let front_buf = &bridge_ref.buffers[read_idx];

                let n_samples = front_buf.n_samples as usize;
                if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
                    return;
                }

                let mut buf = match stream.dequeue_buffer() {
                    Some(b) => b,
                    None => return,
                };

                let datas = buf.datas_mut();
                if datas.len() < 2 {
                    return;
                }

                // split_at_mut para satisfazer o borrow checker
                let (datas_left, datas_right) = datas.split_at_mut(1);
                let data_l = &mut datas_left[0];
                let data_r = &mut datas_right[0];

                let max_l = data_l.as_raw().maxsize as usize / std::mem::size_of::<f32>();
                let max_r = data_r.as_raw().maxsize as usize / std::mem::size_of::<f32>();
                let n_out = n_samples.min(max_l).min(max_r);
                if n_out == 0 {
                    return;
                }

                if let Some(raw_l) = data_l.data() {
                    // SAFETY: `raw_l` é um mmap do PipeWire válido enquanto o buffer
                    // existir. Castamos para `f32` pois negociamos F32P.
                    let out_l = unsafe {
                        std::slice::from_raw_parts_mut(raw_l.as_mut_ptr().cast::<f32>(), n_out)
                    };
                    out_l.copy_from_slice(&front_buf.buf_l[..n_out]);
                }
                if let Some(raw_r) = data_r.data() {
                    // SAFETY: Idem canal direito.
                    let out_r = unsafe {
                        std::slice::from_raw_parts_mut(raw_r.as_mut_ptr().cast::<f32>(), n_out)
                    };
                    out_r.copy_from_slice(&front_buf.buf_r[..n_out]);
                }

                // Informa ao PipeWire quantos bytes foram escritos
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

    let _cpu_dma_lock = lock_cpu_c_states();

    // Emitido aqui (após PM QoS e antes do thread_loop.start()) para agrupar
    // todos os diagnósticos de hardening RT na saída do console.
    sys.emit_irq_advisory(target_cpu);

    // Inicia a execução da ThreadLoop em background, com PipeWire assumindo a Thread de RT
    thread_loop.start();

    // Loop principal da nossa aplicação, monitorando a flag SHUTDOWN e
    // gerenciando a construção de resamplers fora do callback RT.
    let mut was_silent = false;
    while !SHUTDOWN.load(Ordering::Relaxed) {
        // Verifica se o callback RT solicitou rebuild do resampler via flag atômica
        if rt_status.needs_resampler_rebuild.load(Ordering::Relaxed) {
            let target_pw_rate = rt_status.requested_pw_rate.load(Ordering::Relaxed);
            let target_nam_rate = rt_status.requested_nam_rate.load(Ordering::Relaxed);
            if target_pw_rate != 0 && target_nam_rate != 0 {
                // Constrói novo resampler FORA do callback RT (alocação de heap permitida aqui)
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
                        // Envia para o callback RT via canal SPSC dedicado
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
                rt_status
                    .needs_resampler_rebuild
                    .store(false, Ordering::Relaxed);
            }
        }

        was_silent = poll_rt_status(&rt_status, &sys, was_silent);

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    thread_loop.stop();

    Ok(())
}

// =============================================================================
// Helpers Extraídos — Redução de Complexidade do `run_pipewire_host`
// =============================================================================

/// Constrói um SPA Pod de formato de áudio F32P stereo para negociação PipeWire.
///
/// Serializa `audio_info` em um pod binário via FFI (`spa_format_audio_raw_build`).
/// O buffer `format_buf` deve viver pelo menos tanto quanto o pod retornado,
/// pois o pod aponta diretamente para dentro dele (zero-copy FFI).
///
/// # Safety
///
/// Requer que `audio_info` esteja corretamente inicializado (format + channels).
/// O pod retornado é válido **somente** enquanto `format_buf` existir em escopo.
unsafe fn build_spa_format_pod<'a>(
    audio_info: &pw::spa::param::audio::AudioInfoRaw,
    format_buf: &'a mut [u8; 1024],
) -> anyhow::Result<&'a pw::spa::pod::Pod> {
    unsafe {
        let mut builder: pw::spa::sys::spa_pod_builder = std::mem::zeroed();
        pw::spa::sys::spa_pod_builder_init(
            &mut builder,
            format_buf.as_mut_ptr().cast(),
            format_buf.len() as u32,
        );
        let pod_ptr = pw::spa::sys::spa_format_audio_raw_build(
            &mut builder,
            pw::spa::param::ParamType::EnumFormat.as_raw(),
            &audio_info.as_raw(),
        );
        if pod_ptr.is_null() {
            return Err(anyhow::anyhow!(
                "Falha ao construir SPA Pod de formato de áudio"
            ));
        }
        // O pod aponta dentro de format_buf — válido enquanto format_buf existir
        Ok(&*(pod_ptr as *const pw::spa::pod::Pod))
    }
}

/// Detecta o sink de hardware padrão do PipeWire via `pw-metadata`.
///
/// Retorna `None` se não for possível determinar (falha de comando,
/// o default é o próprio NAM-rs, ou parsing falhou).
fn detect_hardware_sink() -> Option<String> {
    let out = std::process::Command::new("pw-metadata")
        .args(["-n", "default", "0", "default.audio.sink"])
        .output()
        .ok()?;

    let s = String::from_utf8_lossy(&out.stdout);
    let start = s.find("\"name\":\"")?;
    let rest = &s[start + 8..];
    let end = rest.find('"')?;
    let name = &rest[..end];

    // Ignora se o default detectado for o próprio NAM-rs
    // (pode ocorrer se foi salvo pelo WirePlumber)
    if name == "NAM-rs-input" || name == "NAM-rs-standalone" {
        None
    } else {
        Some(name.to_string())
    }
}

/// Lê flags atômicas de status RT e emite logs de monitoramento.
///
/// Chamada a cada iteração do loop principal (~100ms). Consome flags
/// one-shot (active_rate, has_clipped, rt_priority, dsp_overloads) e
/// faz edge-detection no estado de silêncio.
///
/// Retorna o novo estado de silêncio (para edge-detection no caller).
fn poll_rt_status(rt_status: &RtStatusFlags, sys: &SystemSnapshot, was_silent: bool) -> bool {
    // Rate do resampler ativado pelo callback RT
    let active_rate = rt_status.active_rate.swap(0, Ordering::Relaxed);
    if active_rate != 0 {
        log::info!(
            "{} Callback RT ativou resampler com rate = {} Hz",
            "✅".green(),
            active_rate
        );
    }

    // Detecção de clipping na saída
    if rt_status.has_clipped.swap(false, Ordering::Relaxed) {
        log::warn!(
            "{} Saturação detectada (Clipping)! Considere reduzir o ganho de entrada e/ou saída.",
            "🔥".bright_red().bold()
        );
    }

    // Confirmação programática de SCHED_FIFO — sentinela -1 significa "ainda não setado".
    // A thread DSP publica este valor uma única vez no cold-path do primeiro frame.
    let prio = rt_status.rt_priority.load(Ordering::Relaxed);
    if prio != -1 {
        let is_fifo = rt_status.rt_is_fifo.load(Ordering::Relaxed);
        // Rearmamos sentinela para não logar novamente nas iterações seguintes.
        rt_status.rt_priority.store(-1, Ordering::Relaxed);

        if is_fifo {
            log::info!(
                "{} Thread DSP confirmada: SCHED_FIFO ativo, prioridade RT = {}",
                "✅".green(),
                prio
            );
        } else {
            NamDiagnostic::new(NamErrorCode::SchedFifoDenied, sys)
                .message(format!(
                    "Thread DSP NÃO está em SCHED_FIFO (prioridade = {}). \
                     O áudio pode sofrer jitter e xruns.",
                    prio
                ))
                .hint(
                    "Verifique se o usuário possui permissão RT (ulimit -r) \
                     ou se o sistema tem rtkit/PipeWire configurados corretamente.",
                )
                .emit_warning();
        }
    }

    // Contador de overloads de CPU
    let overloads = rt_status.dsp_overloads.swap(0, Ordering::Relaxed);
    if overloads > 0 {
        log::warn!(
            "{} Sobrecarga de CPU ({} buffers). Considere usar um modelo mais leve ou processador mais poderoso.",
            "🚨".red(),
            overloads
        );
    }

    // Detecção de transição de silêncio (edge-detect: loga apenas na mudança de estado)
    let current_silent = rt_status.is_silent.load(Ordering::Relaxed);
    if current_silent != was_silent {
        if current_silent {
            log::info!(
                "{} Silence bypass ativado (entrada < −80 dBFS). DSP em idle.",
                "🔇".blue()
            );
        } else {
            log::info!(
                "{} Sinal detectado. Processamento DSP retomado.",
                "🔊".green()
            );
        }
    }

    current_silent
}

/// Configura a thread DSP atual para operação em tempo real.
///
/// Executada **uma única vez** no cold-path do primeiro frame do callback `process()`,
/// antes do fluxo de dados começar de fato. Aplica:
///
/// 1. **DAZ/FTZ** — Habilita Denormals-Are-Zero e Flush-To-Zero no registro MXCSR
///    para evitar penalidades de FPU em blocos de silêncio ("espiral da morte").
/// 2. **Core Affinity** — Fixa a thread no núcleo físico ideal via
///    `pthread_setaffinity_np`, evitando migração de core e cache misses L1/L2.
/// 3. **SCHED_FIFO** — Eleva a prioridade para agendamento de tempo real (prio 90).
///
/// Após configurar, publica o resultado via `rt_status` (flags atômicas):
/// - `rt_is_fifo`: `true` se `SCHED_FIFO` foi confirmado por `pthread_getschedparam`.
/// - `rt_priority`: prioridade efetiva concedida pelo kernel (ou `0` se FIFO não obtido).
///
/// O loop principal em `run_pipewire_host` lê essas flags e emite log de confirmação
/// (ou aviso) de forma auditável — **zero I/O adicional dentro do callback RT**.
///
/// # Diagnósticos
///
/// Esta função usa macros de `log` diretas (`log::error!`, etc) ao invés de `NamDiagnostic` porque:
/// - É chamada dentro do callback RT (thread PipeWire), onde o `SystemSnapshot`
///   não está disponível sem adicionar `Arc` ou parâmetro extra ao closure.
/// - É executada apenas **uma vez** no cold-path inicial — não afeta latência RT.
/// - O formato simula manualmente o padrão de diagnóstico (códigos E2301–E2303) para
///   permitir triagem via `/diagnostico` mesmo sem o bloco de suporte completo.
fn configure_realtime_thread(target_cpu: usize, rt_status: Arc<RtStatusFlags>) {
    // Proteção contra denormals (números subnormalizados que travam o processador):
    // Habilita DAZ (Denormals-Are-Zero) e FTZ (Flush-To-Zero) via registro MXCSR.
    // Sem isso, blocos de silêncio poderiam causar lentidão extrema na FPU ("espiral da morte").
    unsafe {
        crate::math::simd::set_daz_ftz();
    }

    // Desabilita Transparent Huge Pages (THP) para este processo antes do mlockall,
    // evitando latências de compactação em background pelo khugepaged.
    unsafe {
        libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0);
    }

    // Bloqueia a memória atual e futura na RAM física, prevenindo page faults.
    let ret_mlock = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };

    if ret_mlock != 0 {
        let err = std::io::Error::last_os_error();
        log::warn!(
            "⚠️ mlockall() falhou ({}). O áudio pode sofrer engasgos se o sistema usar swap.\n  Dica: Verifique o limite de 'memlock' no ulimits.",
            err
        );
    } else {
        log::info!(
            "🔒 Memória do processo travada na RAM física (mlockall). Page faults prevenidos."
        );
    }

    unsafe {
        let thread_id = libc::pthread_self();

        let name = b"nam_rs_dsp\0";
        libc::pthread_setname_np(thread_id, name.as_ptr() as *const libc::c_char);

        // 1. Afinidade de Núcleo: evita migração de core e invalidação de cache L1/L2
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(target_cpu, &mut cpuset);

        let ret_aff = libc::pthread_setaffinity_np(
            thread_id,
            std::mem::size_of::<libc::cpu_set_t>(),
            &cpuset,
        );

        if ret_aff != 0 {
            log::error!(
                "\n  ⚡ Falha ao definir afinidade de CPU para núcleo {} (errno={}).\n  💡 O NAM-rs continuará funcionando, mas pode sofrer jitter por Core Migration.\n  [E2301 | CPU_AFFINITY_FAILED] cpu={} errno={}\n",
                target_cpu,
                ret_aff,
                target_cpu,
                ret_aff
            );
        }

        // 2. Verificação programática prévia: lê policy/prio concedidos pelo kernel via rtkit/PipeWire
        let mut actual_policy = 0i32;
        let mut actual_param: libc::sched_param = std::mem::zeroed();
        let ret_getsched =
            libc::pthread_getschedparam(thread_id, &mut actual_policy, &mut actual_param);

        let actual_cpu = libc::sched_getcpu();

        if ret_getsched == 0 {
            let reset_on_fork_flag = 0x40000000i32;
            let mut has_reset_on_fork = (actual_policy & reset_on_fork_flag) != 0;
            let mut base_policy = actual_policy & !reset_on_fork_flag;

            // 3. Se PipeWire não elevou para SCHED_FIFO via rtkit, tentamos forçar manualmente
            if base_policy != libc::SCHED_FIFO {
                let mut param: libc::sched_param = std::mem::zeroed();
                param.sched_priority = 90;

                let ret_sched = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);
                if ret_sched != 0 {
                    log::error!(
                        "⚠️ pthread_setschedparam(SCHED_FIFO, 90) falhou (errno={}).\n  [E2302 | RT_SCHED_FAILED] Verifique ulimit -r e permissões rtkit.\n",
                        ret_sched
                    );
                } else {
                    // Atualiza as variáveis refletindo a aplicação bem-sucedida
                    base_policy = libc::SCHED_FIFO;
                    actual_param.sched_priority = 90;
                    has_reset_on_fork = false; // Nós não setamos a flag de reset on fork
                }
            }

            let policy_str = match base_policy {
                libc::SCHED_FIFO => "SCHED_FIFO",
                libc::SCHED_RR => "SCHED_RR",
                libc::SCHED_OTHER => "SCHED_OTHER",
                libc::SCHED_BATCH => "SCHED_BATCH",
                libc::SCHED_IDLE => "SCHED_IDLE",
                _ => "UNKNOWN",
            };

            let confirmed_fifo = base_policy == libc::SCHED_FIFO;

            // Publica resultado real via flags atômicas — zero I/O no caminho quente
            rt_status
                .rt_is_fifo
                .store(confirmed_fifo, Ordering::Relaxed);
            rt_status
                .rt_priority
                .store(actual_param.sched_priority, Ordering::Relaxed);

            // Log inline no cold-path (uma única vez, antes do deadline RT) — aceitável.
            if has_reset_on_fork {
                log::info!(
                    "{} Data/RT Thread PW: CPU Core = {}, Policy = {} | SCHED_RESET_ON_FORK, Priority = {}",
                    "🔍".blue(),
                    actual_cpu.to_string().cyan(),
                    policy_str.cyan(),
                    actual_param.sched_priority.to_string().green()
                );
            } else {
                log::info!(
                    "{} Data/RT Thread PW: CPU Core = {}, Policy = {}, Priority = {}",
                    "🔍".blue(),
                    actual_cpu.to_string().cyan(),
                    policy_str.cyan(),
                    actual_param.sched_priority.to_string().green()
                );
            }
        } else {
            // Publica sentinela de falha de verificação
            rt_status.rt_is_fifo.store(false, Ordering::Relaxed);
            rt_status.rt_priority.store(0, Ordering::Relaxed);

            log::error!(
                "  [E2303 | RT_GETSCHED_FAILED] pthread_getschedparam falhou (ret={}).\n",
                ret_getsched
            );
        }
    }
}

/// Seleciona o núcleo de CPU ideal para fixar a thread RT (core affinity).
fn select_optimal_cpu() -> Option<usize> {
    use std::fs;

    let cpu_dir = fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut cpus: Vec<usize> = cpu_dir
        .filter_map(|entry| {
            let name = entry.ok()?.file_name();
            let name = name.to_str()?;
            name.strip_prefix("cpu")?.parse::<usize>().ok()
        })
        .collect();

    if cpus.is_empty() {
        return None;
    }
    cpus.sort_unstable();

    let capacities: Vec<(usize, u64)> = cpus
        .iter()
        .map(|&cpu| {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
            let cap = fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(1024);
            (cpu, cap)
        })
        .collect();

    let irq_totals = parse_interrupts_per_cpu(cpus.len());

    capacities
        .iter()
        .map(|&(cpu, cap)| {
            let irqs = irq_totals.get(cpu).copied().unwrap_or(u64::MAX);
            (cpu, cap, irqs)
        })
        .max_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| a.0.cmp(&b.0))
        })
        .map(|(cpu, _, _)| cpu)
}

fn parse_interrupts_per_cpu(num_cpus: usize) -> Vec<u64> {
    use std::fs;

    let mut totals = vec![0u64; num_cpus];

    let content = match fs::read_to_string("/proc/interrupts") {
        Ok(c) => c,
        Err(_) => return totals,
    };

    for line in content.lines().skip(1) {
        let trimmed = line.trim_start();
        let irq_end = trimmed.find(':').unwrap_or(0);
        if irq_end == 0 {
            continue;
        }
        if !trimmed[..irq_end]
            .trim()
            .bytes()
            .all(|b| b.is_ascii_digit())
        {
            continue;
        }

        let after_colon = match trimmed.get(irq_end + 1..) {
            Some(s) => s,
            None => continue,
        };

        for (cpu_idx, token) in after_colon.split_whitespace().enumerate() {
            if cpu_idx >= num_cpus {
                break;
            }
            if let Ok(count) = token.parse::<u64>() {
                totals[cpu_idx] += count;
            } else {
                break;
            }
        }
    }

    totals
}

/// Impede que o processador entre em C-States de economia de energia,
/// garantindo latência de despertar de 0ms para processamento de áudio RT.
///
/// RETORNO: O arquivo `File`. Ele DEVE ser mantido vivo no escopo principal.
/// Se for "dropado", o kernel anula a proteção.
pub fn lock_cpu_c_states() -> Option<std::fs::File> {
    match std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/cpu_dma_latency")
    {
        Ok(mut file) => {
            let zero: i32 = 0;
            if std::io::Write::write_all(&mut file, &zero.to_ne_bytes()).is_ok() {
                log::info!(
                    "⚡ PM QoS Lock: C-States profundos da CPU desativados (Zero DMA Latency)."
                );
                return Some(file);
            }
            log::warn!("PM QoS: Falha ao escrever em /dev/cpu_dma_latency.");
            None
        }
        Err(e) => {
            log::warn!(
                "PM QoS: Acesso negado a /dev/cpu_dma_latency ({}). \
                 Considere criar uma regra de udev para o grupo 'audio'.",
                e
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    #[test]
    fn test_dsp_bridge_concurrent_access() {
        let bridge: &'static DspBridge = Box::leak(Box::new(DspBridge {
            buffers: [
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
            ],
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
        }));

        let bridge_ptr_writer = bridge as *const DspBridge as *mut DspBridge as usize;
        let bridge_ptr_reader = bridge as *const DspBridge;

        let writer_handle = std::thread::spawn(move || {
            let mut counter = 0.0f32;
            let bridge_ptr_writer = bridge_ptr_writer as *mut DspBridge;
            for _ in 0..1000 {
                let bridge_ref = unsafe { &mut *bridge_ptr_writer };
                let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
                let back_buf = &mut bridge_ref.buffers[back_idx];

                // Write 64 samples
                for i in 0..64 {
                    back_buf.buf_l[i] = counter;
                    back_buf.buf_r[i] = counter;
                    counter += 1.0;
                }
                back_buf.n_samples = 64;

                std::sync::atomic::fence(Ordering::Release);
                bridge_ref
                    .active_read_idx
                    .store(back_idx, Ordering::Relaxed);
                bridge_ref.generation.fetch_add(1, Ordering::Relaxed);

                // Small sleep to simulate DSP time and allow reader to catch some frames
                std::thread::sleep(Duration::from_micros(10));
            }
        });

        let start = Instant::now();
        let mut last_gen = 0;
        let mut reads = 0;
        let mut last_val_read = -1.0;

        while reads < 1000 && start.elapsed() < Duration::from_millis(100) {
            let bridge_ref = unsafe { &*bridge_ptr_reader };
            let current_gen = bridge_ref.generation.load(Ordering::Relaxed);
            if current_gen != last_gen {
                std::sync::atomic::fence(Ordering::Acquire);
                let read_idx = bridge_ref.active_read_idx.load(Ordering::Relaxed);
                let front_buf = &bridge_ref.buffers[read_idx];

                assert_eq!(front_buf.n_samples, 64);

                let first_val = front_buf.buf_l[0];
                for i in 0..64 {
                    assert_eq!(
                        front_buf.buf_l[i],
                        first_val + i as f32,
                        "Buffer mixing detected in L channel"
                    );
                    assert_eq!(
                        front_buf.buf_r[i],
                        first_val + i as f32,
                        "Buffer mixing detected in R channel"
                    );
                }

                // Values should be monotonically increasing (we might drop frames, but never go backward)
                assert!(
                    first_val > last_val_read,
                    "Read older data than previously seen!"
                );
                last_val_read = front_buf.buf_l[63];

                last_gen = current_gen;
                reads += 1;
            }

            if last_gen == 1000 {
                break;
            }
        }

        writer_handle.join().unwrap();

        // Assert execution time
        assert!(
            start.elapsed() < Duration::from_millis(100),
            "Test took too long to execute"
        );
    }
}
