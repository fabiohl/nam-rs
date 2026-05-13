// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

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
    AppState, BridgeBuffer, DspBridge, DspPipelineContext, MAX_BRIDGE_BUF, MAX_RESAMP_BUF,
    build_spa_format_pod, capture_dsp_pipeline, playback_dsp_cycle,
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

    // 1. Cria a thread assíncrona gerenciada nativamente pelo PipeWire
    let thread_loop = unsafe { pw::thread_loop::ThreadLoopBox::new(Some("nam-rs-loop"), None) }?;
    let context = pw::context::ContextBox::new(thread_loop.loop_(), None)?;
    let core = context.connect(None)?;

    let capture_stream;
    let capture_listener;
    let playback_stream;
    let playback_listener;
    let mut thread_configured = false;

    // 2. Aloca o buffer compartilhado entre capture e playback via leak intencional.
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
        consumed_gen: std::sync::atomic::AtomicU64::new(0),
        dropped_frames: std::sync::atomic::AtomicU32::new(0),
    }));
    // Ponteiro mutável bruto para compartilhamento entre as closures de Capture e Playback.
    // Ambas rodam no mesmo contexto RT do PipeWire, garantindo acesso exclusivo via atomics no Bridge.
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

    // 3. Seleciona o núcleo ideal para a thread RT (capacidade + menor carga IRQ).
    // Computado antes do lock para que o valor seja acessível após o escopo.
    let target_cpu = rt_setup::select_optimal_cpu().unwrap_or(0);

    // 4. Escopo de configuração protegida (RAII)
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

        // 5. Captura o contexto por valor ('move') para a closure do listener.
        // As variáveis locais (modelos, resamplers, buffers) tornam-se o estado persistente da stream RT.
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
                // 5.1. CALLBACK RT — ZERO ALOCAÇÕES, ZERO I/O
                // =========================================================
                // Toda alocação (NamResampler::new, Vec, Box) e toda operação
                // de I/O (println!, eprintln!, write!) são PROIBIDAS neste escopo.
                // Status comunicado via flags atômicas em `rt_status_for_process`.

                // Executa no kernel da thread RT (Data Thread do Pipewire)
                if !thread_configured {
                    rt_setup::configure_realtime_thread(target_cpu, rt_status_for_process.clone());
                    thread_configured = true;
                }

                for slot in parking_lot.iter_mut() {
                    let Some(old) = slot.take() else { continue };
                    if let Err(rtrb::PushError::Full(old_back)) = gc_producer.push(old) {
                        *slot = Some(old_back);
                        break; // Canal ainda cheio
                    }
                }

                // 5.1.1. Drena resamplers pré-construídos pela thread principal (zero-alloc swap).
                while let Ok(new_rs) = resampler_consumer.pop() {
                    // Reporta rate ativo via flag atômica (sem println!)
                    rt_status_for_process
                        .active_rate
                        .store(new_rs.pw_rate(), Ordering::Relaxed);
                    rt_status_for_process
                        .active_rate_changed
                        .store(new_rs.pw_rate(), Ordering::Relaxed);

                    let old_rs = std::mem::replace(&mut resampler, new_rs);
                    if let Err(rtrb::PushError::Full(old_rs)) =
                        gc_producer.push(GcItem::Resampler(old_rs))
                    {
                        // Se falhar o push, tenta o parking lot
                        let mut to_park = Some(old_rs);
                        for slot in parking_lot.iter_mut() {
                            if slot.is_none() {
                                *slot = to_park.take();
                                break;
                            }
                        }
                        if let Some(still_here) = to_park {
                            // Pathológico: Usa o overwrite buffer compartilhado para evitar leak.
                            // Convertemos para Box<GcItem> e passamos o ponteiro raw.
                            let ptr = Box::into_raw(Box::new(still_here));
                            if let Some(leaked_ptr) = gc_overflow_for_process.push_raw(ptr) {
                                // Se o overflow também encheu, o swap retornou o ponteiro antigo.
                                // Como último recurso, vazamos para não dar drop no RT.
                                unsafe {
                                    std::mem::forget(Box::from_raw(leaked_ptr));
                                }
                            }
                            rt_status_for_process
                                .set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
                        }
                    }
                }

                // ---------------------------------------------------------
                // 5.1.2. RECEPÇÃO DE COMANDOS (O "Correio" entre Usuário e Áudio)
                // ---------------------------------------------------------
                // Aqui o motor de áudio verifica se o usuário enviou novos comandos pela CLI
                // (ex: trocar de pedal, mudar o volume). Usamos um canal "SPSC" que permite
                // receber essas ordens sem nunca parar ou atrasar o som.
                let mut param_changed = false;
                while let Ok(payload) = consumer.pop() {
                    match payload {
                        // Comando: Trocar o "Cérebro" Neural (Modelo .nam)
                        ParamPayload::LoadModel {
                            model_l,
                            model_r,
                            input_mult_adj,
                            output_mult_adj,
                            sample_rate,
                        } => {
                            let new_model_l = model_l;
                            let new_model_r = model_r;

                            // Se recebemos modelos válidos, ajustamos os volumes internos
                            // específicos desse modelo (cada captura NAM tem sua própria calibração).
                            if new_model_l.is_some() || new_model_r.is_some() {
                                model_input_mult_adj = input_mult_adj;
                                model_output_mult_adj = output_mult_adj;
                                current_nam_rate = sample_rate;
                            } else {
                                // Caso o modelo seja removido (bypass), resetamos os ganhos.
                                model_input_mult_adj = 1.0;
                                model_output_mult_adj = 1.0;
                                current_nam_rate = 48_000;
                            }

                            // --- DESCARTE SEGURO DO MODELO ANTIGO (Garbage Collection) ---
                            // No mundo do áudio de alta performance, não podemos simplesmente
                            // "apagar" um modelo velho aqui dentro, pois isso causaria um
                            // "soluço" (drop) no som. Em vez disso, trocamos o modelo antigo
                            // pelo novo instantaneamente e enviamos o antigo para uma
                            // "lixeira" (gc_producer) que será esvaziada por outra thread
                            // fora do caminho crítico do áudio.

                            // Troca o modelo do canal Esquerdo
                            if let Some(old) = std::mem::replace(&mut active_model_l, new_model_l) {
                                #[allow(clippy::collapsible_if)]
                                if let Err(rtrb::PushError::Full(old_item)) =
                                    gc_producer.push(GcItem::Model(old))
                                {
                                    // Se a lixeira estiver cheia, usamos um "estacionamento" temporário
                                    let mut to_park = Some(old_item);
                                    for slot in parking_lot.iter_mut() {
                                        if slot.is_none() {
                                            *slot = to_park.take();
                                            break;
                                        }
                                    }
                                    // Caso extremo: se até o estacionamento lotar, usamos um buffer de overflow
                                    // para garantir que a memória não vaze nem cause travamentos.
                                    if let Some(still_here) = to_park {
                                        let ptr = Box::into_raw(Box::new(still_here));
                                        if let Some(leaked_ptr) =
                                            gc_overflow_for_process.push_raw(ptr)
                                        {
                                            unsafe {
                                                std::mem::forget(Box::from_raw(leaked_ptr));
                                            }
                                        }
                                        rt_status_for_process
                                            .set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
                                    }
                                }
                            }

                            // Troca o modelo do canal Direito (mesma lógica do Esquerdo)
                            if let Some(old) = std::mem::replace(&mut active_model_r, new_model_r) {
                                #[allow(clippy::collapsible_if)]
                                if let Err(rtrb::PushError::Full(old_item)) =
                                    gc_producer.push(GcItem::Model(old))
                                {
                                    let mut to_park = Some(old_item);
                                    for slot in parking_lot.iter_mut() {
                                        if slot.is_none() {
                                            *slot = to_park.take();
                                            break;
                                        }
                                    }
                                    if let Some(still_here) = to_park {
                                        let ptr = Box::into_raw(Box::new(still_here));
                                        if let Some(leaked_ptr) =
                                            gc_overflow_for_process.push_raw(ptr)
                                        {
                                            unsafe {
                                                std::mem::forget(Box::from_raw(leaked_ptr));
                                            }
                                        }
                                        rt_status_for_process
                                            .set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
                                    }
                                }
                            }
                            param_changed = true;
                        }

                        // Comando: Ajustar o Ganho de Entrada
                        ParamPayload::InputGain(mult) => {
                            user_input_gain_mult = mult;
                            param_changed = true;
                        }

                        // Comando: Ajustar o Ganho de Saída
                        ParamPayload::OutputGain(mult) => {
                            user_output_gain_mult = mult;
                            param_changed = true;
                        }

                        // Comando: Configurar o Noise Gate (Filtro de Ruído)
                        ParamPayload::GateConfig(params) => {
                            // O Gate impede que ruídos de fundo (chiado) passem quando o áudio não está tocando.
                            // Aqui transformamos os decibéis (dB) do usuário em valores matemáticos lineares
                            // para que o processador possa compará-los rapidamente em cada amostra de som.
                            let open_lin = lut.db_to_linear(params.threshold_open_db);
                            let close_lin = lut.db_to_linear(params.threshold_close_db);
                            threshold_open_sq = open_lin * open_lin;
                            threshold_close_sq = close_lin * close_lin;
                            gate_params = params;
                        }
                    }
                }

                // ---------------------------------------------------------
                // 5.1.3. SINCRONIZAÇÃO DE RITMO (Taxa de Amostragem / Sample Rate)
                // ---------------------------------------------------------
                // O áudio digital tem um "ritmo" (ex: 48.000 amostras por segundo).
                // Se o sistema (PipeWire) mudar esse ritmo, ou se carregarmos um efeito
                // que trabalha em outro ritmo, precisamos de um "tradutor" (Resampler).
                let detected_pw_rate = rate_for_process.swap(0, Ordering::Relaxed);
                let current_pw_rate = resampler.pw_rate();

                let mut pw_rate_to_request = current_pw_rate;
                let mut requires_rebuild = false;

                // Verifica se o ritmo do sistema mudou
                if detected_pw_rate != 0 && detected_pw_rate != current_pw_rate {
                    pw_rate_to_request = detected_pw_rate;
                    requires_rebuild = true;
                }

                // Verifica se o ritmo do modelo neural é diferente do atual
                if current_nam_rate != resampler.nam_rate() {
                    requires_rebuild = true;
                }

                // Se os ritmos não batem, pedimos para a thread principal construir
                // um novo "tradutor" (Resampler). Não fazemos isso aqui para evitar atrasos.
                if requires_rebuild && pw_rate_to_request != 0 {
                    rt_status_for_process
                        .requested_pw_rate
                        .store(pw_rate_to_request, Ordering::Relaxed);
                    rt_status_for_process
                        .requested_nam_rate
                        .store(current_nam_rate, Ordering::Relaxed);
                    rt_status_for_process
                        .set_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
                }

                // Se o usuário mexeu no volume ou trocou o modelo, recalculamos os
                // multiplicadores de ganho. Isso deixa a conta matemática pronta para
                // o processador usar no caminho do áudio, economizando tempo.
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
                // 5.1.4. LÓGICA DSP - TEMPO REAL
                // =========================================================
                // Adquire o próximo buffer disponível do servidor PipeWire.
                let mut _buf = match stream.dequeue_buffer() {
                    Some(b) => b,
                    None => return,
                };

                // 5.1.4.1. Extração dos buffers de áudio do PipeWire
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

                                // Decimação de Telemetria: Medimos o tempo apenas a cada 16 frames
                                // para reduzir o overhead de RDTSC no hot-path.
                                let should_measure = (frame_count & 0xF) == 0;
                                frame_count = frame_count.wrapping_add(1);

                                let start_nanos = if should_measure {
                                    rt_setup::rdtsc_nanos()
                                } else {
                                    0
                                };

                                // ---------------------------------------------------------
                                // 5.1.4.2. EXECUÇÃO DO PROCESSAMENTO (A Fábrica de Som)
                                // ---------------------------------------------------------
                                // Aqui é onde a mágica acontece. Chamamos a "fábrica" (pipeline)
                                // passando o som bruto da guitarra e um "kit de ferramentas"
                                // (DspPipelineContext) que contém os modelos neurais, volumes
                                // e filtros que preparamos nos passos anteriores.
                                capture_dsp_pipeline(
                                    samples_l,
                                    samples_r,
                                    n_samples,
                                    DspPipelineContext {
                                        resampler: &mut resampler,           // O tradutor de ritmo
                                        active_model_l: &mut active_model_l, // O cérebro esquerdo
                                        active_model_r: &mut active_model_r, // O cérebro direito
                                        input_gain_mult,                     // Volume de entrada
                                        output_gain_mult,                    // Volume de saída
                                        gate_params: &gate_params, // Ajustes do filtro de ruído
                                        silence_hysteresis: &mut silence_hysteresis,
                                        mono_hysteresis: &mut mono_hysteresis,
                                        threshold_open_sq,
                                        threshold_close_sq,
                                        process_mono: &mut process_mono,
                                        rt_status: &rt_status_for_process,
                                        bridge_ptr, // Ponte para a saída
                                        resamp_mid_l: &mut resamp_mid_l, // Buffers de trabalho
                                        resamp_mid_r: &mut resamp_mid_r,
                                        resamp_out_l: &mut resamp_out_l,
                                        resamp_out_r: &mut resamp_out_r,
                                        model_out_l: &mut model_out_l,
                                        model_out_r: &mut model_out_r,
                                    },
                                );

                                if should_measure {
                                    // Monitoramento de carga de DSP (DSP Load Monitoring via RDTSC)
                                    // `start_nanos` é capturado no início da pipeline DSP (acima).
                                    // Reportamos o tempo bruto em nanos para a thread de controle.
                                    let elapsed_nanos =
                                        rt_setup::rdtsc_nanos().wrapping_sub(start_nanos);
                                    rt_status_for_process
                                        .dsp_cycle_time
                                        .store(elapsed_nanos, Ordering::Relaxed);
                                    rt_status_for_process.latency_hist.record(elapsed_nanos);

                                    // Calcula o budget máximo tolerável (85% do tempo real do buffer)
                                    // Ainda usamos uma aproximação escalar aqui para detecção rápida de overload,
                                    // mas a telemetria precisa via Anchor ocorre no poll_rt_status.
                                    let elapsed_secs = elapsed_nanos as f64 / 1_000_000_000.0;
                                    let budget_secs =
                                        (n_samples as f64 / current_pw_rate as f64) * 0.85;
                                    if elapsed_secs > budget_secs {
                                        rt_status_for_process
                                            .dsp_overloads
                                            .fetch_add(1, Ordering::Relaxed);
                                    }
                                }

                                rt_status_for_process
                                    .last_n_samples
                                    .store(n_samples as u32, Ordering::Relaxed);
                            }
                        }
                    }
                }
                // Buffer devolvido automaticamente ao PipeWire no `Drop`.
            })
            .register()?;

        // 6. Constrói o SPA Pod F32P stereo para negociação de formato com PipeWire.
        let mut audio_info = pw::spa::param::audio::AudioInfoRaw::new();
        audio_info.set_format(pw::spa::param::audio::AudioFormat::F32P);
        audio_info.set_channels(2);

        let mut format_buf = [0u8; 1024];
        let format_pod = unsafe { build_spa_format_pod(&audio_info, &mut format_buf)? };

        // 6.1. Ativa a capture stream com formato de áudio declarado.
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

        // 7. PLAYBACK STREAM: envia áudio processado para o hardware
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
                playback_dsp_cycle(stream, bridge_ptr_playback, &mut last_bridge_gen);
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

    // 8. Inicia a execução da ThreadLoop em background, com PipeWire assumindo a Thread de RT
    thread_loop.start();

    // =========================================================
    // 9. LOOP DE CONTROLE (NON-RT)
    // =========================================================
    // Este loop executa na thread principal da aplicação. Sua função é gerenciar
    // tarefas pesadas (alocações, I/O) que são proibidas dentro dos callbacks RT.
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

                        rt_status.set_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED);
                    }
                }
                // Limpamos a solicitação após o processamento.
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_NEEDS_RESAMPLER_REBUILD);
            }
        }

        // 9.2. Monitoramento de Status e Limpeza de Memória (GC)
        // Consultamos as flags atômicas do callback RT para atualizar a UI/Logs (clipping, silêncio, timing).
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

    // 10. Graceful Shutdown
    // Paramos o loop do PipeWire antes de encerrar a função para garantir que
    // os recursos sejam liberados corretamente.
    thread_loop.stop();

    Ok(())
}

#[cfg(test)]
#[path = "pw_host_test.rs"]
mod pw_host_test;
