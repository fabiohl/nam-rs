// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Núcleo de processamento de áudio DSP usando `pipewire-rs`.
//!
//! Este é o "coração" do NAM-rs: o módulo que de fato processa o som da guitarra
//! em tempo real. Ele recebe amostras brutas de áudio do PipeWire (o servidor de som
//! do Linux), passa pelo amplificador neural, e devolve o resultado processado —
//! tudo sem nenhum engasgo ou atraso perceptível.
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
//! ## Fluxo de processamento
//!
//! O callback `process()` segue esta sequência para cada bloco de áudio:
//! 1. Aplica ganho de entrada SIMD (ajuste de volume pré-amplificador)
//! 2. `NamResampler::process_input()` — converte o sample rate do PipeWire para 48 kHz
//! 3. Inferência neural WaveNet/LSTM — o "amplificador virtual" processa o som
//! 4. `NamResampler::process_output()` — converte de volta para o sample rate original
//! 5. Aplica ganho de saída SIMD (ajuste de volume pós-amplificador)
//!
//! Quando nenhum modelo está carregado, opera em pass-through (o som entra e sai sem alteração).
//! Quando o sample rate do PipeWire já é 48 kHz, o resampler opera em bypass sem overhead.

use crate::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::dsp::gain::apply_gain_simd;
use crate::dsp::resampler::NamResampler;
use crate::spsc::{ParamPayload, RtStatusFlags, SHUTDOWN};
use pipewire as pw;
use pw::properties::properties;
use rtrb::Consumer;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Estrutura de posse explícita para evitar o leak de memória
/// das instâncias essenciais do PipeWire (`StreamBox` e `Listener`).
struct AppState<S, L> {
    #[allow(dead_code)]
    stream: S,
    #[allow(dead_code)]
    listener: L,
}

/// Inicializa a topologia PipeWire, configurando o nó como um filtro bidirecional
/// (consumo e produção) na API do WirePlumber, executando o MainLoop até SHUTDOWN.
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
) -> anyhow::Result<()> {
    // 1. Cria a thread assíncrona gerenciada nativamente pelo PipeWire
    let thread_loop = unsafe { pw::thread_loop::ThreadLoopBox::new(Some("nam-rs-loop"), None) }?;
    let context = pw::context::ContextBox::new(thread_loop.loop_(), None)?;
    let core = context.connect(None)?;

    let stream;
    let listener;
    let mut thread_configured = false;

    // Obtém o lock para o loop, necessário para criar e configurar streams do PW com segurança
    // O lock é automaticamente liberado ao final deste escopo (_lock is dropped).
    {
        let _lock = thread_loop.lock();

        // Configurando Stream DSP Ativo. "Audio/Filter" cria um nó que expõe
        // portas de Capture e Playback dinamicamente, permitindo ligação na cadeia sem resampling forçado.
        stream = pw::stream::StreamBox::new(
            &core,
            "NAM-rs",
            properties! {
                *pw::keys::MEDIA_TYPE => "Audio",
                *pw::keys::MEDIA_CATEGORY => "Filter",
                *pw::keys::MEDIA_ROLE => "DSP",
                *pw::keys::MEDIA_CLASS => "Audio/Filter", // Nó DSP
                *pw::keys::NODE_NAME => "NAM-rs-standalone",
                *pw::keys::NODE_DESCRIPTION => "Neural Amp Modeler (Rust PipeWire native)",
                // Extensões para bit-perfect: PipeWire tentará acoplar com as cfg da fonte
            },
        )?;

        let target_cpu = select_optimal_cpu().unwrap_or(0);
        let mut active_model: Option<Box<crate::models::DynamicModel>> = None;

        // NamResampler bidirecional: converte entre o rate do PipeWire e os 48 kHz do NAM.
        // Inicializado com 48k (bypass) — rate real será atualizado via canal SPSC de resamplers
        // quando a thread principal construir e enviar um novo resampler.
        let mut resampler = NamResampler::new(48_000, 2048).unwrap_or_else(|e| {
            NamDiagnostic::new(NamErrorCode::ResamplerBuildFailed, &sys)
                .message("Falha ao criar NamResampler inicial (usando bypass 48k).")
                .hint("O engine contínua em modo bypass. O resampler será recriado ao receber o rate real do PipeWire.")
                .param("initial_rate", 48_000_u32)
                .param("detail", &e)
                .emit_warning();
            // Fallback: bypass 48k nunca falha (rate == NAM_RATE)
            NamResampler::new(48_000, 2048).expect("bypass 48k não pode falhar")
        });

        // Buffer intermediário 48k para saída do process_input e entrada do process_output (stack).
        // Dimensionado para MAX_DSP_BUF — o resampler pode expandir/contrair dentro deste limite.
        const MAX_RESAMP_BUF: usize = 4096;
        let mut resamp_mid = [0.0f32; MAX_RESAMP_BUF];
        let mut resamp_out = [0.0f32; MAX_RESAMP_BUF];

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
        listener = stream
            .add_local_listener::<()>()
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
                            model,
                            input_db_adj,
                            output_db_adj,
                        } => {
                            let mut new_model = model;
                            if let Some(ref mut _m) = new_model {
                                model_input_db_adj = input_db_adj;
                                model_output_db_adj = output_db_adj;
                            } else {
                                model_input_db_adj = 0.0;
                                model_output_db_adj = 0.0;
                            }

                            if let Some(old) = std::mem::replace(&mut active_model, new_model) {
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

                // Verifica se a Thread Main do PipeWire reportou alguma alteração via param_changed.
                // Em vez de alocar NamResampler::new() aqui, setamos uma flag atômica para que
                // a thread principal construa o resampler fora do callback RT.
                let detected_rate = rate_for_process.swap(0, Ordering::Relaxed);
                if detected_rate != 0 && detected_rate != resampler.pw_rate() {
                    rt_status_for_process
                        .requested_rate
                        .store(detected_rate, Ordering::Relaxed);
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

                // Extrai a primeira região de dados do buffer (mono, canal 0).
                let datas = _buf.datas_mut();
                if let Some(d) = datas.first_mut() {
                    let chunk = d.chunk();
                    let offset = chunk.offset() as usize;
                    let size = chunk.size() as usize;

                    if let Some(ref mut model_box) = active_model
                        && size > 0
                        && let Some(raw_bytes) = d.data()
                    {
                        let n_bytes = size.min(raw_bytes.len().saturating_sub(offset));
                        let n_samples = n_bytes / std::mem::size_of::<f32>();

                        if n_samples > 0 {
                            let samples = unsafe {
                                std::slice::from_raw_parts_mut(
                                    raw_bytes.as_mut_ptr().add(offset).cast::<f32>(),
                                    n_samples,
                                )
                            };

                            // Buffer temporal na stack para saída (Zero-Alloc).
                            // Máximo razoável para callbacks PipeWire: 2048 amostras.
                            const MAX_DSP_BUF: usize = 2048;
                            let mut temp_out = [0.0f32; MAX_DSP_BUF];
                            let n = n_samples.min(MAX_DSP_BUF);

                            // 1. Aplica ganho de entrada SIMD antes do resampling e inferência
                            apply_gain_simd(&mut samples[..n], input_gain_mult);

                            // 2. Downsample: PW_rate → 48 kHz (ou bypass se já em 48k)
                            let n_48k = resampler
                                .process_input(&samples[..n], &mut resamp_mid[..MAX_RESAMP_BUF]);

                            // 3. Inferência NAM a 48 kHz
                            model_box
                                .0
                                .process(&resamp_mid[..n_48k], &mut temp_out[..n_48k]);

                            // 4. Upsample: 48 kHz → PW_rate (ou bypass se já em 48k)
                            let n_pw = resampler.process_output(
                                &temp_out[..n_48k],
                                &mut resamp_out[..MAX_RESAMP_BUF],
                            );

                            // 5. Aplica ganho de saída SIMD após conversão de rate
                            apply_gain_simd(&mut resamp_out[..n_pw], output_gain_mult);

                            // Copia resultado de volta para o buffer do PipeWire
                            let n_copy = n_pw.min(n);
                            samples[..n_copy].copy_from_slice(&resamp_out[..n_copy]);
                        }
                    }
                }
                // Buffer devolvido automaticamente ao PipeWire no `Drop`.
            })
            .register()?;

        // Ativa o stream pedindo auto-conexão pelo daemon (WirePlumber),
        // com buffers mapeados pelo DMA/Mem no host, na cadência RT sem especificar Direction
        // pois filtros configuram in & out.
        stream.connect(
            pw::spa::utils::Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut [],
        )?;
    }

    let _app_state = AppState { stream, listener };

    // Inicia a execução da ThreadLoop em background, com PipeWire assumindo a Thread de RT
    thread_loop.start();

    // Loop principal da nossa aplicação, monitorando a flag SHUTDOWN e
    // gerenciando a construção de resamplers fora do callback RT.
    while !SHUTDOWN.load(Ordering::Relaxed) {
        // Verifica se o callback RT solicitou rebuild do resampler via flag atômica
        if rt_status.needs_resampler_rebuild.load(Ordering::Relaxed) {
            let target_rate = rt_status.requested_rate.load(Ordering::Relaxed);
            if target_rate != 0 {
                // Constrói novo resampler FORA do callback RT (alocação de heap permitida aqui)
                match NamResampler::new(target_rate, 2048) {
                    Ok(new_rs) => {
                        rt_status
                            .resampler_rebuild_failed
                            .store(false, Ordering::Relaxed);
                        // Mensagem mundana informativa
                        println!(
                            "[NAM-rs] 🔄 Sample rate atualizado: {} Hz (bypass={})",
                            target_rate,
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
                                .param("target_rate", target_rate)
                                .emit_warning();
                        }
                    }
                    Err(e) => {
                        NamDiagnostic::new(NamErrorCode::ResamplerBuildFailed, &sys)
                            .message(format!(
                                "Falha ao reconstruir o resampler para {} Hz.",
                                target_rate
                            ))
                            .hint(
                                "O áudio continuará com o resampler anterior. \
                                 Se a taxa de amostragem estiver errada, reinicie o NAM-rs.",
                            )
                            .param("target_rate", target_rate)
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

        // Lê e imprime status do callback RT (comunicação silenciosa via flags atômicas)
        let active_rate = rt_status.active_rate.swap(0, Ordering::Relaxed);
        if active_rate != 0 {
            println!(
                "[NAM-rs] ✅ Callback RT ativou resampler com rate = {} Hz",
                active_rate
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
                println!(
                    "[NAM-rs] ✅ Thread DSP confirmada: SCHED_FIFO ativo, prioridade RT = {}",
                    prio
                );
            } else {
                NamDiagnostic::new(NamErrorCode::SchedFifoDenied, &sys)
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

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    thread_loop.stop();

    Ok(())
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
/// Esta função usa `eprintln!` direto (ao invés de `NamDiagnostic`) porque:
/// - É chamada dentro do callback RT (thread PipeWire), onde o `SystemSnapshot`
///   não está disponível sem adicionar `Arc` ou parâmetro extra ao closure.
/// - É executada apenas **uma vez** no cold-path inicial — não afeta latência RT.
/// - O formato simula manualmente o padrão de diagnóstico (códigos E2301–E2303) para
///   permitir triagem via `/diagnostico` mesmo sem o bloco de suporte completo.
fn configure_realtime_thread(target_cpu: usize, rt_status: Arc<RtStatusFlags>) {
    // Proteção contra denormals (números subnormalizados que travam o processador):
    // Habilita DAZ (Denormals-Are-Zero) e FTZ (Flush-To-Zero) via registro MXCSR.
    // Sem isso, blocos de silêncio poderiam causar lentidão extrema na FPU ("espiral da morte").
    #[cfg(target_arch = "x86_64")]
    unsafe {
        crate::math::simd::set_daz_ftz();
    }

    #[cfg(target_os = "linux")]
    {
        unsafe {
            let thread_id = libc::pthread_self();

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
                eprintln!(
                    "\n  ⚡ Falha ao definir afinidade de CPU para núcleo {} (errno={}).",
                    target_cpu, ret_aff
                );
                eprintln!(
                    "  💡 O NAM-rs continuará funcionando, mas pode sofrer jitter por Core Migration."
                );
                eprintln!(
                    "  [E2301 | CPU_AFFINITY_FAILED] cpu={} errno={}\n",
                    target_cpu, ret_aff
                );
            }

            // 2. Tempo Real: SCHED_FIFO prio 90
            let mut param: libc::sched_param = std::mem::zeroed();
            param.sched_priority = 90;

            let ret_sched = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);
            if ret_sched != 0 {
                eprintln!(
                    "  ⚠️  pthread_setschedparam(SCHED_FIFO, 90) falhou (errno={}).",
                    ret_sched
                );
                eprintln!("  [E2302 | RT_SCHED_FAILED] Verifique ulimit -r e permissões rtkit.\n");
            }

            // 3. Verificação programática: lê policy/prio *reais* concedidos pelo kernel.
            //    Publicados via RtStatusFlags — o loop principal (fora do RT) fará o log definitivo.
            let mut actual_policy = 0i32;
            let mut actual_param: libc::sched_param = std::mem::zeroed();
            let ret_getsched =
                libc::pthread_getschedparam(thread_id, &mut actual_policy, &mut actual_param);

            let actual_cpu = libc::sched_getcpu();

            if ret_getsched == 0 {
                let reset_on_fork_flag = 0x40000000i32;
                let has_reset_on_fork = (actual_policy & reset_on_fork_flag) != 0;
                let base_policy = actual_policy & !reset_on_fork_flag;

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
                    println!(
                        "[NAM-rs] \u{1F50D} Data/RT Thread PW: CPU Core = {}, Policy = {} | SCHED_RESET_ON_FORK, Priority = {}",
                        actual_cpu, policy_str, actual_param.sched_priority
                    );
                } else {
                    println!(
                        "[NAM-rs] \u{1F50D} Data/RT Thread PW: CPU Core = {}, Policy = {}, Priority = {}",
                        actual_cpu, policy_str, actual_param.sched_priority
                    );
                }
            } else {
                // Publica sentinela de falha de verificação
                rt_status.rt_is_fifo.store(false, Ordering::Relaxed);
                rt_status.rt_priority.store(0, Ordering::Relaxed);

                eprintln!(
                    "  [E2303 | RT_GETSCHED_FAILED] pthread_getschedparam falhou (ret={}).\n",
                    ret_getsched
                );
            }
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
