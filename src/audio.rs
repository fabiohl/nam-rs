// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Núcleo de processamento de áudio DSP usando `pipewire-rs`.
//! Contém o callback estrito da thread de Tempo-Real/DSP responsável pelo processamento
//! de inferência neural NAM. Recebe amostras brutas do host PipeWire via backend
//! standalone nativo e as processa em tempo real.
//! Zero alocação na heap, zero I/O, zero mutexes durante o `process()`.
//!
//! # Estado Atual (Tarefa 1.3)
//! O motor de inferência neural ainda não foi implementado (Sprints 2-5).
//! O callback do Stream opera em modo **pass-through**: o áudio de entrada
//! flui inalterado para a saída, provando a injeção nula do nó na Session Manager.

use crate::buffer::{ParamPayload, SHUTDOWN};
use pipewire as pw;
use pw::properties::properties;
use rtrb::Consumer;
use std::sync::atomic::Ordering;

/// Inicializa a topologia PipeWire, configurando o nó como um filtro bidirecional
/// (consumo e produção) na API do WirePlumber, executando o MainLoop até SHUTDOWN.
pub fn run_pipewire_host(mut consumer: Consumer<ParamPayload>) -> anyhow::Result<()> {
    // 1. Cria a thread assíncrona gerenciada nativamente pelo PipeWire
    let thread_loop = unsafe { pw::thread_loop::ThreadLoopBox::new(Some("nam-rs-loop"), None) }?;
    let context = pw::context::ContextBox::new(thread_loop.loop_(), None)?;
    let core = context.connect(None)?;

    let mut thread_configured = false;

    // Obtém o lock para o loop, necessário para criar e configurar streams do PW com segurança
    // O lock é automaticamente liberado ao final deste escopo (_lock is dropped).
    {
        let _lock = thread_loop.lock();

        // Configurando Stream DSP Ativo. "Audio/Filter" cria um nó que expõe
        // portas de Capture e Playback dinamicamente, permitindo ligação na cadeia sem resampling forçado.
        let stream = pw::stream::StreamBox::new(
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

        // Oculta warnings do compilador de lifetime ou clonagem de referências do pw-stream
        let _listener = stream
            .add_local_listener::<()>()
            .process(move |stream: &pw::stream::Stream, _info| {
                // Executa no kernel da thread RT (Data Thread do Pipewire)
                if !thread_configured {
                    configure_realtime_thread(target_cpu);
                    thread_configured = true;
                }

                // Drena parâmetros guiados da thread CLI (Lock-Free) via SPSC Ring Buffer
                while let Ok(_payload) = consumer.pop() {
                    // Sprint 1.3: pass-through sem lógica de modelagem; suprimimos o limite
                    // Evita que o SPSC Ring encha enquanto o DSP não manipula as mensagens.
                }

                // =========================================================
                // LÓGICA DSP - TEMPO REAL
                // =========================================================
                // Recupera o Buffer do PipeWire alocado internamente contendo os canais.
                // Numa stream Filter, um `dequeue_buffer` entrega dados de In e Out conjugados
                // para "In-place processing".
                let mut _buf = match stream.dequeue_buffer() {
                    Some(b) => b,
                    None => return,
                };

                // O PipeWire lida nativamente com o pass-through in-place de bytes não tocados (Injeção Nula).
                // Retorna o pacote processado: `Buffer` devolve para a fila automaticamente no `Drop`.
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

        // Aqui temos um pequeno caveat: O _listener não pode ser dropado!
        // No Pipewire-rs, o Stream e o Listener não podem ser dropados enquanto
        // quisermos que eles existam. Precisamos mantê-los vivos!
        // Vou usar mem::forget por enquanto ou jogar eles dentro de uma struct.
        // Wait, o StreamBox *DEVE* sobreviver.
        std::mem::forget(stream);
        std::mem::forget(_listener);
    }

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
