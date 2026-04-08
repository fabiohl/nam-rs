// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Núcleo de processamento de áudio DSP usando `pipewire-rs`.
//! Contém o callback estrito da thread de Tempo-Real/DSP responsável pelo processamento
//! de inferência neural NAM. Recebe amostras brutas do host PipeWire via backend
//! standalone nativo e as processa em tempo real.
//! Zero alocação na heap, zero I/O, zero mutexes durante o `process()`.
//!
//! # Estado Atual (Sprint 4)
//! O motor de inferência neural está integrado: o callback `process()` extrai amostras
//! do buffer PipeWire e despacha para o modelo ativo (WaveNet/LSTM) via trait `NamModel`.
//! Gain staging SIMD (input/output) é aplicado antes e após a inferência.
//! Quando nenhum modelo está carregado, opera em pass-through nativo (Injeção Nula).

use crate::dsp::gain::apply_gain_simd;
use crate::spsc::{ParamPayload, SHUTDOWN};
use pipewire as pw;
use pw::properties::properties;
use rtrb::Consumer;
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
pub fn run_pipewire_host(mut consumer: Consumer<ParamPayload>) -> anyhow::Result<()> {
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
        let mut active_model: *mut crate::models::DynamicModel = std::ptr::null_mut();

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

        // Oculta warnings do compilador de lifetime ou clonagem de referências do pw-stream
        listener = stream
            .add_local_listener::<()>()
            .process(move |stream: &pw::stream::Stream, _info| {
                // Executa no kernel da thread RT (Data Thread do Pipewire)
                if !thread_configured {
                    configure_realtime_thread(target_cpu);
                    thread_configured = true;
                }

                // Drena parâmetros guiados da thread CLI (Lock-Free) via SPSC Ring Buffer
                let mut param_changed = false;
                while let Ok(payload) = consumer.pop() {
                    match payload {
                        ParamPayload::LoadModel {
                            ptr,
                            input_db_adj,
                            output_db_adj,
                        } => {
                            if !ptr.is_null() {
                                active_model = ptr as *mut crate::models::DynamicModel;
                                let model = unsafe { &mut *active_model };
                                model.0.prewarm(2048);
                                model_input_db_adj = input_db_adj;
                                model_output_db_adj = output_db_adj;
                            } else {
                                active_model = std::ptr::null_mut();
                                model_input_db_adj = 0.0;
                                model_output_db_adj = 0.0;
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

                    if !active_model.is_null()
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

                            // Multiplica buffer SIMD entrada pelo reescalonamento dinâmico do Stage de Ganho
                            apply_gain_simd(&mut samples[..n], input_gain_mult);

                            let model = unsafe { &mut *active_model };
                            model.0.process(&samples[..n], &mut temp_out[..n]);

                            // Multiplica buffer SIMD saída garantindo ausência de clipping final
                            apply_gain_simd(&mut temp_out[..n], output_gain_mult);

                            samples[..n].copy_from_slice(&temp_out[..n]);
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

    // Loop principal da nossa aplicação, monitorando a flag SHUTDOWN.
    // Assim não paralisamos a main mantendo ela limpa.
    while !SHUTDOWN.load(Ordering::Relaxed) {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    thread_loop.stop();

    Ok(())
}

/// Tenta fixar a thread DSP no núcleo físico ideal e aplicar SCHED_FIFO para
/// agendamento de tempo real. Chamada apenas uma vez na primeira invocação do `process()`,
/// antes do fluxo de dados começar.
fn configure_realtime_thread(target_cpu: usize) {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let thread_id = libc::pthread_self();

            // 1. Afinidade de Núcleo: Foco L1/L2
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
                    "Aviso: Falha ao definir afinidade de CPU para núcleo {} (erro {}).",
                    target_cpu, ret_aff
                );
            }

            // 2. Tempo Real: SCHED_FIFO
            let mut param: libc::sched_param = std::mem::zeroed();
            param.sched_priority = 90;

            let _ret_sched = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);

            let mut actual_policy = 0;
            let mut actual_param: libc::sched_param = std::mem::zeroed();
            let ret_getsched =
                libc::pthread_getschedparam(thread_id, &mut actual_policy, &mut actual_param);

            let actual_cpu = libc::sched_getcpu();

            if ret_getsched == 0 {
                let reset_on_fork_flag = 0x40000000;
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

                if has_reset_on_fork {
                    println!(
                        "[NAM-rs] 🔍 Data/RT Thread PW: CPU Core = {}, Policy = {} | SCHED_RESET_ON_FORK, Priority = {}",
                        actual_cpu, policy_str, actual_param.sched_priority
                    );
                } else {
                    println!(
                        "[NAM-rs] 🔍 Data/RT Thread PW: CPU Core = {}, Policy = {}, Priority = {}",
                        actual_cpu, policy_str, actual_param.sched_priority
                    );
                }
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
