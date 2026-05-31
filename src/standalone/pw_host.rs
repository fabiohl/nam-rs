// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Núcleo de processamento de áudio DSP usando `pipewire-rs`.
//!
//! Este é o "coração" do modo Standalone do NAM-rs: o módulo que processa o áudio em
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
//! closures via ponteiro raw, com sincronização lock-free via `Ordering::Release/Acquire`
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
//! 1. **Noise Gate e Ganho de Entrada** — Avalia a energia do sinal e aplica o ganho inicial (pré-DSP).
//! 2. `NamResampler::process_input()` — Converte o sample rate para a taxa compatível (geralmente 48 kHz).
//! 3. **Inferência neural WaveNet/LSTM** — O motor neural que processa o sinal sonoro.
//! 4. `NamResampler::process_output()` — Converte de volta para o sample rate original do host.
//! 5. **Ganho de Saída e Clipping** — Aplica o volume final e detecta saturação digital.
//! 6. **Escrita no [`DspBridge`]** — Publica o resultado com `Ordering::Release` para o playback callback.
//!
//! Quando nenhum modelo está carregado, o motor opera em **True-Bypass** (o sinal de entrada passa limpo).
//! Quando o sample rate do PipeWire é o mesmo do modelo nam, o resampler opera em bypass sem overhead.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode};
use crate::common::spsc::{GcItem, GcOverflowBuffer, ParamPayload, RtStatusFlags, SHUTDOWN};
use crate::dsp::gate::{DynamicHysteresis, GateParams};
pub use crate::dsp::pipeline::PipewireHostConfig;
use crate::dsp::pipeline::{
    AppState, BridgeBuffer, BridgeRef, DspBridge, DspBridgeReader, DspBridgeWriter, DspBuffers,
    DspPipelineContext, MAX_BRIDGE_BUF, MAX_RESAMP_BUF, build_spa_format_pod, capture_dsp_pipeline,
    playback_dsp_cycle,
};
use crate::dsp::resampler::NamResampler;
use crate::standalone::colors::Colorize;
use crate::standalone::rt_setup;

use pipewire as pw;
use pw::properties::properties;
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

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
#[allow(clippy::too_many_arguments)]
pub fn run_pipewire_host(
    mut consumer: Consumer<ParamPayload>,
    mut gc_producer: rtrb::Producer<GcItem>,
    gc_overflow: Arc<GcOverflowBuffer>,
    mut resampler_consumer: Consumer<NamResampler>,
    mut resampler_producer: rtrb::Producer<NamResampler>,
    rt_status: Arc<RtStatusFlags>,
    config: PipewireHostConfig,
    mut gc_consumer: Consumer<GcItem>,
) -> anyhow::Result<()> {
    let PipewireHostConfig {
        buffer_size,
        tsc_anchor,
        sys,
    } = config;

    // =========================================================
    // 1. INICIALIZAÇÃO DO LOOP PIPEWIRE
    // =========================================================
    // Cria a thread assíncrona (ThreadLoop) nativa do PipeWire e conecta ao contexto/core.
    // O ThreadLoop é o "motor" que despachará os eventos de áudio.
    let thread_loop = unsafe { pw::thread_loop::ThreadLoopBox::new(Some("nam-rs-loop"), None) }?;
    let context = pw::context::ContextBox::new(thread_loop.loop_(), None)?;
    let core = context.connect(None)?;

    let capture_stream;
    let capture_listener;
    let playback_stream;
    let playback_listener;
    let mut thread_configured = false;

    // =========================================================
    // 2. ALOCAÇÃO DO DSP BRIDGE (Comunicação Lock-Free)
    // =========================================================
    // Aloca o buffer compartilhado entre as streams de capture e playback via leak intencional.
    // O DspBridge vive até o término do processo — evitando completamente a necessidade
    // de locks (mutexes) entre as threads no caminho crítico (hot-path).
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
        consumed_gen: std::sync::atomic::AtomicU64::new(0),
        dropped_frames: std::sync::atomic::AtomicU32::new(0),
    }));
    // Ponteiro mutável bruto encapsulado em BridgeRef para segurança RT.
    // Ambas rodam no mesmo contexto RT do PipeWire, garantindo acesso exclusivo via atomics no Bridge.
    let bridge_ptr = unsafe { BridgeRef::new(bridge as *const DspBridge as *mut DspBridge) };

    // Otimiza o gerenciamento de memória do bridge no Kernel:
    // - MADV_DONTFORK: Evita overhead de Copy-on-Write (COW) se o processo sofrer fork.
    // - MADV_DONTDUMP: Exclui os grandes buffers de áudio de core dumps, reduzindo latência de escrita em falhas.
    unsafe {
        libc::madvise(
            bridge as *const DspBridge as *mut libc::c_void,
            std::mem::size_of::<DspBridge>(),
            libc::MADV_DONTFORK | libc::MADV_DONTDUMP,
        );
    }

    // =========================================================
    // 3. OTIMIZAÇÃO DE NÚCLEO (CPU Affinity)
    // =========================================================
    // Seleciona o núcleo físico ideal para a thread RT (maior capacidade + menor carga IRQ).
    // Computado na thread main, antes do lock, pois a leitura do /sys/ é uma I/O lenta.
    let target_cpu = rt_setup::select_optimal_cpu().unwrap_or(0);

    // =========================================================
    // 4. ESCOPO DE CONFIGURAÇÃO PROTEGIDA (RAII)
    // =========================================================
    // Tudo dentro deste bloco `{ ... }` executa com o loop do PipeWire "congelado".
    //
    // O `thread_loop.lock()` adquire o mutex global do PipeWire. Isso previne que a
    // API C despache eventos de áudio enquanto as streams ainda estão sendo construídas,
    // evitando race conditions e segfaults.
    //
    // O Rust garante que a variável `_lock` chame a função `Drop` ao atingir a chave `}` final,
    // liberando o PipeWire de forma infalível — mesmo que ocorram erros precoces via `?`.
    // Isso evita deadlocks antes do posterior `thread_loop.start()`.
    //
    // POR QUE AQUI? (Posicionamento do Escopo)
    // - Não antes: Alocações na Heap (Box::leak), syscalls (madvise) e leitura de arquivos
    //   (/sys/ para descobrir a CPU ideal) são lentos. Se fizéssemos isso com o lock ativo,
    //   congelaríamos todo o servidor de áudio do sistema (PipeWire) por tempo desnecessário,
    //   podendo causar estalos (xruns) em outros apps.
    // - Não depois: A partir daqui, criaremos as Streams e os Listeners. Assim que uma stream
    //   é conectada ao Core, o PipeWire já pode tentar enviar eventos. Precisamos do lock
    //   para garantir que os callbacks e o estado interno estejam 100% prontos antes de soltar a trava.
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

        // Configura a latência desejada na stream (se especificada via CLI).
        let latency_str = format!("{}/48000", buffer_size);
        if buffer_size > 0 {
            capture_props.insert("node.latency", latency_str.as_str());
        }

        // 4.1. Inicializa a stream de captura (Virtual Sink) que receberá áudio das aplicações.
        capture_stream = pw::stream::StreamBox::new(&core, "NAM-rs", capture_props)?;

        // 4.2. Estado interno dos modelos ativos: um para cada canal, permitindo processamento stereo.
        let mut active_model_l: Option<Box<crate::models::DynamicModel>> = None;
        let mut active_model_r: Option<Box<crate::models::DynamicModel>> = None;

        // 4.3. NamResampler bidirecional: converte o rate do PipeWire para o rate suportado pelo NAM carregado (geralmente 48khz).
        // Inicializado com 48k (bypass) — rate real será atualizado via canal SPSC de resamplers
        // quando a thread principal construir e enviar um novo resampler.
        let resampler = NamResampler::new(48_000, 48_000, 2048).unwrap_or_else(|e| {
            NamDiagnostic::new(NamErrorCode::ResamplerBuildFailed, &sys)
                .message("Failed to create initial NamResampler (using 48k bypass).")
                .hint("The engine remains in bypass mode. The resampler will be recreated upon receiving the actual rate from PipeWire.")
                .param("initial_rate", 48_000_u32)
                .param("detail", &e)
                .emit_warning();
            // Fallback: bypass 48k nunca falha (rate == NAM_RATE)
            NamResampler::new(48_000, 48_000, 2048).expect("bypass não pode falhar")
        });
        let mut resampler = Box::new(resampler);

        // Trackeia a taxa de amostragem alvo do loop RT
        let mut current_nam_rate: u32 = 48_000;

        // Buffer intermediário 48k para saída do resampler (stack-allocated).
        // MAX_RESAMP_BUF comporta a expansão do resampler (ex: 44100→48000 = ~1.088x).
        let mut resamp_mid_l = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_l = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_mid_r = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0f32; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0f32; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0f32; MAX_RESAMP_BUF];

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
        let lut = crate::math::dsp::gain_lut::get_gain_lut();
        let open_lin = lut.db_to_linear(gate_params.threshold_open_db);
        let close_lin = lut.db_to_linear(gate_params.threshold_close_db);
        let mut threshold_open_sq: f32 = open_lin * open_lin;
        let mut threshold_close_sq: f32 = close_lin * close_lin;

        let shared_target_rate = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let rate_for_param = shared_target_rate.clone();
        let rate_for_process = shared_target_rate.clone();

        // Parking Lot para garantir zero-drop mesmo se o canal GC estiver cheio.
        // Unificado para modelos e resamplers via GcItem.
        let mut parking_lot: [Option<GcItem>; 16] = Default::default();

        // Clonar Arc das flags para uso dentro do callback
        let rt_status_for_process = rt_status.clone();
        let gc_overflow_for_process = gc_overflow.clone();
        let mut frame_count: u32 = 0;

        // =========================================================
        // 5. REGISTRO DOS CALLBACKS (Listeners) E ESTADO RT
        // =========================================================
        // Captura o contexto por valor ('move') para a closure do listener.
        // As variáveis locais criadas até aqui (modelos, resamplers, buffers)
        // tornam-se o estado persistente da stream RT, evitando alocações futuras.
        capture_listener = capture_stream
            .add_local_listener::<()>()
            // Callback: Acionado quando o estado geral da stream muda (ex: conectando, pausado, erro).
            .state_changed(move |_stream, _user_data, old, new| match new {
                pw::stream::StreamState::Error(err) => {
                    log::error!("{} Critical PW audio stream failure: {}", "💥".red(), err);
                }
                pw::stream::StreamState::Paused if old == pw::stream::StreamState::Streaming => {
                    log::info!("{} Audio disconnected or node switch.", "⏸️".yellow());
                }
                pw::stream::StreamState::Streaming if old == pw::stream::StreamState::Paused => {
                    log::info!("{} Audio captured (connection established)", "▶️".green());
                }
                _ => {}
            })
            // Callback: Acionado quando os parâmetros de formato do áudio são alterados.
            .param_changed(move |_stream, _user_data, id, param| {
                let Some(param) = param else { return };
                // Apenas processa mudanças relacionadas ao tipo de Formato de Áudio.
                if id != pw::spa::param::ParamType::Format.as_raw() {
                    return;
                }

                // Tenta extrair as características do formato (tipo de mídia e subtipo).
                let (media_type, media_subtype) =
                    match pw::spa::param::format_utils::parse_format(param) {
                        Ok(v) => v,
                        Err(_) => return,
                    };

                // Nós só nos importamos com áudio bruto (Raw Audio). Ignora MIDI ou vídeos.
                if media_type != pw::spa::param::format::MediaType::Audio
                    || media_subtype != pw::spa::param::format::MediaSubtype::Raw
                {
                    return;
                }

                // Efetua o parse final da estrutura de formato de áudio fornecida pelo PipeWire.
                let mut format = pw::spa::param::audio::AudioInfoRaw::default();
                if format.parse(param).is_ok() {
                    let rate = format.rate();
                    // Salva a nova taxa de amostragem na flag atômica (comunicação thread-safe para a thread principal).
                    rate_for_param.store(rate, std::sync::atomic::Ordering::Relaxed);
                }
            })
            .process(move |stream: &pw::stream::Stream, _info| {
                // =========================================================
                // 5.1. CALLBACK RT (Capture) — ZERO ALOCAÇÕES, ZERO I/O
                // =========================================================
                // Toda alocação (NamResampler::new, Vec, Box) e operação de I/O
                // são PROIBIDAS aqui. Comunicação com a main thread ocorre apenas
                // via flags atômicas e canais SPSC.
                // Status comunicado via flags atômicas em `rt_status_for_process`.

                // Configuração da Thread em si (executada no primeiro ciclo RT).
                if !thread_configured {
                    // Configura Scheduler (FIFO), Afinidade de CPU e PM QoS para garantir RT estrito.
                    rt_setup::configure_realtime_thread(target_cpu, rt_status_for_process.clone());
                    thread_configured = true;
                }

                // Tentativa de esvaziar o 'Parking Lot' de Garbage Collection.
                // Itens aqui foram estacionados temporariamente porque a fila principal do GC estava cheia no passado.
                for slot in parking_lot.iter_mut() {
                    let Some(old) = slot.take() else { continue };
                    // Tenta empurrar para o coletor da thread principal.
                    if let Err(rtrb::PushError::Full(old_back)) = gc_producer.push(old) {
                        // Se falhou novamente, devolvemos a variável para o estacionamento e paramos a iteração.
                        *slot = Some(old_back);
                        break; // Fila cheia, nem adianta tentar os próximos
                    }
                }

                // 5.1.1. Drenagem de Resamplers (Zero-Alloc Swap)
                drain_resamplers(
                    &mut resampler_consumer,
                    &mut resampler,
                    &mut gc_producer,
                    &mut parking_lot,
                    &gc_overflow_for_process,
                    &rt_status_for_process,
                );

                // 5.1.2. RECEPÇÃO DE COMANDOS (Canal SPSC)
                let param_changed = receive_commands(
                    &mut consumer,
                    &mut model_input_mult_adj,
                    &mut model_output_mult_adj,
                    &mut current_nam_rate,
                    &mut active_model_l,
                    &mut active_model_r,
                    &mut gc_producer,
                    &mut parking_lot,
                    &gc_overflow_for_process,
                    &rt_status_for_process,
                    &mut user_input_gain_mult,
                    &mut user_output_gain_mult,
                    &mut gate_params,
                    &mut threshold_open_sq,
                    &mut threshold_close_sq,
                    lut,
                );

                // 5.1.3. SINCRONIZAÇÃO DE RATE (Clock Tracking)
                let current_pw_rate = sync_rate(
                    &rate_for_process,
                    &resampler,
                    current_nam_rate,
                    &rt_status_for_process,
                );

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

                // 5.1.4. LÓGICA DSP DE TEMPO REAL
                process_dsp_buffer(
                    stream,
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
                        bridge_writer: DspBridgeWriter::from_ref(bridge_ptr),
                    },
                    DspBuffers {
                        resamp_mid_l: &mut resamp_mid_l,
                        resamp_mid_r: &mut resamp_mid_r,
                        resamp_out_l: &mut resamp_out_l,
                        resamp_out_r: &mut resamp_out_r,
                        model_out_l: &mut model_out_l,
                        model_out_r: &mut model_out_r,
                    },
                    current_pw_rate,
                    &mut frame_count,
                    &rt_status_for_process,
                );
                // Buffer devolvido automaticamente ao PipeWire no `Drop`.
            })
            .register()?;

        // =========================================================
        // 6. NEGOCIAÇÃO DE FORMATO (Capture Stream)
        // =========================================================
        // Constrói o "SPA Pod" contendo a negociação do formato Float 32-bit Planar (F32P).
        let mut audio_info = pw::spa::param::audio::AudioInfoRaw::new();
        audio_info.set_format(pw::spa::param::audio::AudioFormat::F32P);
        audio_info.set_channels(2);

        let mut format_buf = [0u8; 1024];
        let format_pod = unsafe { build_spa_format_pod(&audio_info, &mut format_buf)? };

        // 6.1. Ativação (Connect) da Stream de Captura
        // Direction::Input orienta a porta lógica do PipeWire a receber áudio de outros apps.
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
            "{} Capture stream connected to PipeWire (Audio/Sink, F32P Planar Stereo).",
            "🎼".bright_blue()
        );

        // =========================================================
        // 7. PLAYBACK STREAM: Entrega final do áudio ao Hardware
        // =========================================================
        // A stream de captura (acima) processa o áudio em tempo real, porém, por ser um "Sink" virtual,
        // o PipeWire apenas lê seu estado de entrada e ignora modificações in-place para reprodução.
        // A Playback Stream atua como a perna final do pipeline, lendo do `bridge`
        // e efetivamente tocando o som nos alto-falantes/placa de som do sistema.

        // 7.1. Configuração do Ponteiro Compartilhado
        // O bridge_ptr_playback é enviado para o callback da playback stream, de forma que ele
        // saiba de onde puxar as amostras de áudio que foram preenchidas pela stream de captura.
        let bridge_ptr_playback = unsafe { DspBridgeReader::new(bridge_ptr.as_ptr()) };

        // 7.2. Detecção Automática de Placa de Som
        // Verifica as interfaces de saída física ativas (ALSA/USB). Isso evita que o
        // WirePlumber envie o som processado de volta para um nó virtual, criando feedback.
        let hardware_target = rt_setup::detect_hardware_sink();

        // 7.3. Propriedades do Nó de Reprodução (Playback) no PipeWire
        let mut playback_props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Playback",
            *pw::keys::MEDIA_ROLE => "Music", // Classifica no SO que é fluxo musical prioritário
            *pw::keys::MEDIA_CLASS => "Stream/Output/Audio",
            *pw::keys::NODE_NAME => "NAM-rs-playback",
            *pw::keys::NODE_DESCRIPTION => "NAM-rs Processed Output",
            "audio.position" => "FL,FR",
            "node.group" => "nam-rs-dsp", // Sincroniza o relógio (clock) com a capture stream
            "node.link-group" => "nam-rs-link-group", // Evita feedback loop (WirePlumber não conecta nós do mesmo grupo)
        };

        if let Some(target) = hardware_target {
            // Força o envio direto para o nó de hardware detectado
            playback_props.insert("node.target", target.as_str());
            log::info!(
                "{} Output automatically routed to default hardware: {}",
                "🔌".green(),
                target
            );
        }

        if buffer_size > 0 {
            // Repassa a latência (tamanho de bloco) solicitada para a playback stream
            playback_props.insert("node.latency", latency_str.as_str());
        }

        // 7.4. Inicialização da Stream de Playback
        playback_stream = pw::stream::StreamBox::new(&core, "NAM-rs-Output", playback_props)?;

        // 7.5. Configuração do Callback de Leitura (RT)
        // Controla a 'geração' (generation) dos dados para não ler o mesmo buffer duas vezes
        let mut last_bridge_gen: u64 = 0;

        playback_listener = playback_stream
            .add_local_listener::<()>()
            .process(move |stream: &pw::stream::Stream, _info| {
                // O loop RT que lê as amostras prontas no Bridge e transfere para o PipeWire.
                playback_dsp_cycle(stream, bridge_ptr_playback, &mut last_bridge_gen);
            })
            .register()?;

        // 7.6. Negociação de Formato (F32P)
        // O formato precisa ser estritamente igual ao da captura (Float 32bits Planar) para que
        // a transferência não exija cópias ou conversões extras no buffer bridge (Zero I/O, Zero Alloc).
        let mut playback_audio_info = pw::spa::param::audio::AudioInfoRaw::new();
        playback_audio_info.set_format(pw::spa::param::audio::AudioFormat::F32P);
        playback_audio_info.set_channels(2);

        let mut playback_format_buf = [0u8; 1024];
        let playback_format_pod =
            unsafe { build_spa_format_pod(&playback_audio_info, &mut playback_format_buf)? };

        // 7.7. Conexão Final da Stream (Direction::Output)
        // Direction::Output sinaliza que nós estamos enviando dados para o servidor de áudio
        playback_stream.connect(
            pw::spa::utils::Direction::Output,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut [playback_format_pod],
        )?;

        log::info!(
            "{} Playback stream connected to PipeWire (Stream/Output/Audio, F32P Planar Stereo).",
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

    // =========================================================
    // 8. INÍCIO DA THREAD RT (Background)
    // =========================================================
    // O PipeWire assume oficialmente o controle da thread de áudio a partir daqui.
    thread_loop.start();

    // =========================================================
    // 9. LOOP DE CONTROLE PRINCIPAL (Thread Main, Non-RT)
    // =========================================================
    // Enquanto o DSP roda em background, a thread principal (esta) acorda periodicamente
    // para tratar operações lentas, gerenciamento de memória e reconstrução de objetos.
    let mut was_silent = false;
    let mut was_fading = false;
    while !SHUTDOWN.load(Ordering::Relaxed) {
        // 9.1. Gestão Dinâmica de Resampling
        // O callback RT sinaliza via flag atômica se houve mudança na taxa de amostragem
        // do PipeWire ou do modelo carregado.
        if rt_status.check_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD) {
            let target_pw_rate = rt_status.requested_pw_rate.load(Ordering::Relaxed);
            let target_nam_rate = rt_status.requested_nam_rate.load(Ordering::Relaxed);

            if target_pw_rate != 0 && target_nam_rate != 0 {
                // Reconstruímos o resampler aqui (Non-RT). A alocação de heap é permitida.
                // Isso garante que o áudio não sofra dropouts durante a troca de formato.
                match NamResampler::new(target_pw_rate, target_nam_rate, 2048) {
                    Ok(new_rs) => {
                        rt_status
                            .clear_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED);

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
                                .message("Resampler channel full. Rebuild discarded.")
                                .hint(
                                    "The audio engine is overloaded. \
                                     If the problem persists, restart NAM-rs.",
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
                                "Failed to rebuild resampler for PW={} Hz and NAM={} Hz.",
                                target_pw_rate, target_nam_rate
                            ))
                            .hint(
                                "Audio will continue with the previous resampler. \
                                 If the sample rate is incorrect, restart NAM-rs.",
                            )
                            .param("target_pw_rate", target_pw_rate)
                            .param("target_nam_rate", target_nam_rate)
                            .param("detail", &e)
                            .emit();

                        rt_status.set_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED);
                    }
                }
                // Limpamos a solicitação após o processamento.
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
            }
        }

        // 9.2. Telemetria e Coleta de Lixo (GC)
        // Faz "polling" do status RT e envia modelos velhos de volta ao sistema para desalocação segura.
        (was_silent, was_fading) = rt_setup::poll_rt_status(
            &rt_status,
            &sys,
            was_silent,
            was_fading,
            &tsc_anchor,
            bridge,
        );

        // Executa a drenagem de modelos e resamplers obsoletos (Drop-Delegation).
        crate::common::spsc::drain_gc_channels(&mut gc_consumer, &gc_overflow);

        // Baixa frequência de polling para economizar energia, já que estas são tarefas de controle.
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // =========================================================
    // 10. GRACEFUL SHUTDOWN
    // =========================================================
    // Para o loop e encerra streams antes que o processo principal caia, liberando recursos no kernel.
    // os recursos sejam liberados corretamente.
    thread_loop.stop();

    Ok(())
}

#[cfg(test)]
#[path = "pw_host_test.rs"]
mod pw_host_test;

// =========================================================
// FUNÇÕES AUXILIARES DE PROCESSAMENTO RT
// =========================================================

/// 5.1.1. Drenagem de Resamplers (Zero-Alloc Swap)
/// Substitui os resamplers sem usar alocação de memória no caminho crítico.
#[inline(always)]
fn drain_resamplers(
    resampler_consumer: &mut rtrb::Consumer<NamResampler>,
    resampler: &mut Box<NamResampler>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow_for_process: &crate::common::spsc::GcOverflowBuffer,
    rt_status_for_process: &RtStatusFlags,
) {
    // Puxa da fila o resampler atualizado criado na thread de controle (que pode usar a heap).
    while let Ok(new_rs) = resampler_consumer.pop() {
        let new_rs = Box::new(new_rs);
        // Atualiza atômicos de diagnóstico para o painel de status sem interrupção.
        rt_status_for_process
            .active_rate
            .store(new_rs.pw_rate(), std::sync::atomic::Ordering::Relaxed);
        rt_status_for_process
            .active_rate_changed
            .store(new_rs.pw_rate(), std::sync::atomic::Ordering::Relaxed);

        // Swap ultra-rápido: Troca a variável velha pela nova.
        let old_rs = std::mem::replace(resampler, new_rs);

        // Envia para GC
        let mut item = Some(GcItem::Resampler(old_rs));

        // 1. Canal principal
        if let Some(i) = item.take() {
            if let Err(rtrb::PushError::Full(returned)) = gc_producer.push(i) {
                item = Some(returned);
            } else {
                continue;
            }
        }

        // 2. Parking lot
        if let Some(i) = item.take() {
            let mut parked = false;
            let mut i_opt = Some(i);
            for slot in parking_lot.iter_mut() {
                if slot.is_none() {
                    *slot = i_opt.take();
                    parked = true;
                    break;
                }
            }
            if !parked {
                item = i_opt;
            } else {
                continue;
            }
        }

        // 3. Overflow
        if let Some(i) = item.take() {
            rt_status_for_process.set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
            gc_overflow_for_process.push(i);
        }
    }
}

/// 5.1.2. RECEPÇÃO DE COMANDOS (Canal SPSC)
/// Esta função processa comandos da interface de linha de comando ou sistema de controle (volume, modelo, noise gate).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn receive_commands(
    consumer: &mut rtrb::Consumer<ParamPayload>,
    model_input_mult_adj: &mut f32,
    model_output_mult_adj: &mut f32,
    current_nam_rate: &mut u32,
    active_model_l: &mut Option<Box<crate::models::DynamicModel>>,
    active_model_r: &mut Option<Box<crate::models::DynamicModel>>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow_for_process: &crate::common::spsc::GcOverflowBuffer,
    rt_status_for_process: &RtStatusFlags,
    user_input_gain_mult: &mut f32,
    user_output_gain_mult: &mut f32,
    gate_params: &mut GateParams,
    threshold_open_sq: &mut f32,
    threshold_close_sq: &mut f32,
    lut: &crate::math::dsp::gain_lut::GainLUT,
) -> bool {
    let mut param_changed = false;

    // Processa toda a fila de parâmetros até ela esvaziar. Zero travamentos lock-free.
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

                // Caso haja um modelo novo, absorve os ganhos específicos e taxa alvo.
                if new_model_l.is_some() || new_model_r.is_some() {
                    *model_input_mult_adj = input_mult_adj;
                    *model_output_mult_adj = output_mult_adj;
                    *current_nam_rate = sample_rate;
                } else {
                    // Bypass total. Restaura fatores para 1.0 e taxa default de 48k.
                    *model_input_mult_adj = 1.0;
                    *model_output_mult_adj = 1.0;
                    *current_nam_rate = 48_000;
                }

                // Troca nos canais L e R.
                let mut old_models = Vec::with_capacity(2);
                if let Some(old) = std::mem::replace(active_model_l, new_model_l) {
                    old_models.push(old);
                }
                if let Some(old) = std::mem::replace(active_model_r, new_model_r) {
                    old_models.push(old);
                }

                for m in old_models {
                    let mut item = Some(GcItem::Model(m));

                    // 1. Canal principal
                    if let Some(v) = item.take() {
                        if let Err(rtrb::PushError::Full(returned)) = gc_producer.push(v) {
                            item = Some(returned);
                        } else {
                            continue;
                        }
                    }

                    // 2. Parking Lot
                    if let Some(v) = item.take() {
                        let mut parked = false;
                        let mut v_opt = Some(v);
                        for slot in parking_lot.iter_mut() {
                            if slot.is_none() {
                                *slot = v_opt.take();
                                parked = true;
                                break;
                            }
                        }
                        if !parked {
                            item = v_opt;
                        }
                    }

                    // 3. Overflow
                    if let Some(v) = item.take() {
                        rt_status_for_process.set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
                        gc_overflow_for_process.push(v);
                    }
                }
                param_changed = true;
            }
            ParamPayload::InputGain(mult) => {
                // Altera o volume da entrada de áudio do usuário.
                *user_input_gain_mult = mult;
                param_changed = true;
            }
            ParamPayload::OutputGain(mult) => {
                // Altera o volume final.
                *user_output_gain_mult = mult;
                param_changed = true;
            }
            ParamPayload::GateConfig(params) => {
                // Converte dB em escalas Lineares via tabela de busca LUT instantânea.
                let open_lin = lut.db_to_linear(params.threshold_open_db);
                let close_lin = lut.db_to_linear(params.threshold_close_db);
                // Já trabalha valores quadrados em cold-path para que no callback T16 (per-sample)
                // nós possamos comparar a energia do sinal ao invés de sua raiz (evitando raiz quadrada no DSP).
                *threshold_open_sq = open_lin * open_lin;
                *threshold_close_sq = close_lin * close_lin;
                *gate_params = params;
            }
        }
    }
    // Retorna flag informando se os coeficientes de Volume geral precisarão ser recalculados.
    param_changed
}

/// 5.1.3. SINCRONIZAÇÃO DE RATE (Clock Tracking)
/// Avalia se houve discrepância de frequência e envia pedido para a Thread Main.
#[inline(always)]
fn sync_rate(
    rate_for_process: &std::sync::atomic::AtomicU32,
    resampler: &NamResampler,
    current_nam_rate: u32,
    rt_status_for_process: &RtStatusFlags,
) -> u32 {
    // Coleta a taxa de amostragem detectada pelo sistema PipeWire (callback de parâmetro format).
    let detected_pw_rate = rate_for_process.swap(0, std::sync::atomic::Ordering::Relaxed);
    // Coleta a taxa de amostragem com a qual o Resampler foi programado inicialmente.
    let current_pw_rate = resampler.pw_rate();

    let mut pw_rate_to_request = current_pw_rate;
    let mut requires_rebuild = false;

    // Se a placa de som mudou a amostragem, precisaremos de um novo tradutor.
    if detected_pw_rate != 0 && detected_pw_rate != current_pw_rate {
        pw_rate_to_request = detected_pw_rate;
        requires_rebuild = true;
    }

    // Se o modelo neural exige um "ritmo" diferente do que está configurado.
    if current_nam_rate != resampler.nam_rate() {
        requires_rebuild = true;
    }

    // Pede gentilmente (via flag lock-free) para a Main Thread instanciar um novo Resampler e enviar.
    if requires_rebuild && pw_rate_to_request != 0 {
        rt_status_for_process
            .requested_pw_rate
            .store(pw_rate_to_request, std::sync::atomic::Ordering::Relaxed);
        rt_status_for_process
            .requested_nam_rate
            .store(current_nam_rate, std::sync::atomic::Ordering::Relaxed);
        rt_status_for_process.set_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
    }

    // Retorna o rate atual com a qual o DSP deve lidar nesse mesmo exato quadro de áudio.
    current_pw_rate
}

/// 5.1.4. LÓGICA DSP DE TEMPO REAL
/// Adquire o Buffer bruto do sistema e delega à Fãbrica de Áudio (pipeline) o trabalho pesado.
#[inline(always)]
fn process_dsp_buffer(
    stream: &pw::stream::Stream,
    context: DspPipelineContext,
    buffers: DspBuffers,
    current_pw_rate: u32,
    frame_count: &mut u32,
    rt_status_for_process: &RtStatusFlags,
) {
    // 1. Tira o próximo bloco da fila do PipeWire. Se falhar, retorna cedo (early exit).
    let mut _buf = match stream.dequeue_buffer() {
        Some(b) => b,
        None => return,
    };

    let datas = _buf.datas_mut();
    // 2. Garante que existem ao menos 2 canais (Esquerdo e Direito).
    if datas.len() >= 2 {
        let (left_datas, right_datas) = datas.split_at_mut(1);
        if let (Some(d_l), Some(d_r)) = (left_datas.first_mut(), right_datas.first_mut()) {
            // 3. Os "chunks" representam os limites da memória que o Kernel liberou para usarmos.
            let chunk_l = d_l.chunk();
            let chunk_r = d_r.chunk();
            let offset_l = chunk_l.offset() as usize;
            let size_l = chunk_l.size() as usize;
            let offset_r = chunk_r.offset() as usize;
            let size_r = chunk_r.size() as usize;

            if let (Some(raw_l), Some(raw_r)) = (d_l.data(), d_r.data()) {
                // 4. Prevenção absoluta de overflow (Crash de Kernel). Conta exata de amostras F32.
                let n_bytes_l = size_l.min(raw_l.len().saturating_sub(offset_l));
                let n_bytes_r = size_r.min(raw_r.len().saturating_sub(offset_r));
                let n_samples_l = n_bytes_l / std::mem::size_of::<f32>();
                let n_samples_r = n_bytes_r / std::mem::size_of::<f32>();

                // Nivelamento pelo menor bloco, para não tocar num ponteiro além do canal vizinho.
                let n_samples = n_samples_l.min(n_samples_r);

                if n_samples > 0 {
                    // 5. Zero-Copy magic! Mapeia memória do kernel direto num array tipado Rust F32.
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

                    // 6. Decimação de Telemetria: Economiza ciclos só medindo tempo em 1 de cada 16 quadros.
                    let should_measure = (*frame_count & 0xF) == 0;
                    *frame_count = frame_count.wrapping_add(1);

                    let start_nanos = if should_measure {
                        rt_setup::rdtsc_nanos() // CPU cycles (timestamp via hardware)
                    } else {
                        0
                    };

                    // 7. O CORAÇÃO DO SISTEMA! Manda tudo para o motor SIMD neural de áudio.
                    capture_dsp_pipeline(samples_l, samples_r, n_samples, context, buffers);

                    // 8. Finalizando diagnóstico. Subtrai tempo de CPU final pelo inicial.
                    if should_measure {
                        let elapsed_nanos = rt_setup::rdtsc_nanos().wrapping_sub(start_nanos);
                        rt_status_for_process
                            .dsp_cycle_time
                            .store(elapsed_nanos, std::sync::atomic::Ordering::Relaxed);
                        rt_status_for_process.latency_hist.record(elapsed_nanos);

                        // 9. Se o processamento custou 85% do tempo disponível, contamos +1 sobrecarga.
                        let elapsed_secs = elapsed_nanos as f64 / 1_000_000_000.0;
                        let budget_secs = (n_samples as f64 / current_pw_rate as f64) * 0.85;
                        if elapsed_secs > budget_secs {
                            rt_status_for_process
                                .dsp_overloads
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }

                    // Por último, grava quantos frames tocamos e fecha a cortina do áudio RT.
                    rt_status_for_process
                        .last_n_samples
                        .store(n_samples as u32, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }
}
