// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Utilitários para configuração e monitoramento de tempo real e hardware.
//!
//! Este módulo contém funções auxiliares para gerenciar afinidade de CPU,
//! prioridades SCHED_FIFO, travamento de memória (mlockall), e telemetria
//! de status RT do motor de áudio.

use crate::colors::Colorize;
use crate::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::spsc::RtStatusFlags;
use minstant::Anchor;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Frequência calibrada do TSC em GHz (ciclos por nanosegundo).
/// Armazenado como ponto fixo (valor * 1000) para evitar floats no hot-path.
static TSC_FREQ_GHZ_X1000: AtomicU64 = AtomicU64::new(0);

/// Retorna o tempo atual em nanosegundos usando a instrução RDTSC.
///
/// Oferece precisão de ~1ns com custo de ~1 ciclo, evitando a syscall vDSO clock_gettime.
/// Se o TSC não estiver calibrado ou disponível, faz fallback para Instant::now().
#[inline(always)]
pub fn rdtsc_nanos() -> u64 {
    let freq_x1000 = TSC_FREQ_GHZ_X1000.load(Ordering::Relaxed);
    if freq_x1000 == 0 {
        return Instant::now().elapsed().as_nanos() as u64; // Fallback (aproximado)
    }

    #[cfg(target_arch = "x86_64")]
    {
        let cycles = unsafe { core::arch::x86_64::_rdtsc() };
        (cycles * 1000) / freq_x1000
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Instant::now().elapsed().as_nanos() as u64
    }
}

/// Calibra a frequência do TSC em relação ao clock do sistema.
///
/// Executada no startup (cold-path). Mede a variação do RDTSC durante 50ms
/// para determinar a taxa de ciclos por nanosegundo.
#[cold]
pub fn calibrate_tsc() {
    #[cfg(target_arch = "x86_64")]
    {
        use std::thread;

        // Aquecimento (Ignora o primeiro intervalo para estabilizar caches)
        let _ = unsafe { core::arch::x86_64::_rdtsc() };
        thread::sleep(Duration::from_millis(10));

        let start_inst = Instant::now();
        let start_tsc = unsafe { core::arch::x86_64::_rdtsc() };

        thread::sleep(Duration::from_millis(50));

        let end_inst = Instant::now();
        let end_tsc = unsafe { core::arch::x86_64::_rdtsc() };

        let elapsed_nanos = end_inst.duration_since(start_inst).as_nanos() as u64;
        let elapsed_cycles = end_tsc.wrapping_sub(start_tsc);

        if let Some(freq_x1000) = (elapsed_cycles * 1000).checked_div(elapsed_nanos) {
            TSC_FREQ_GHZ_X1000.store(freq_x1000, Ordering::SeqCst);

            log::info!(
                "{} Relógio de Alta Precisão (TSC) calibrado em {:.3} GHz",
                "⏱️".bright_blue(),
                freq_x1000 as f64 / 1000.0
            );
        }
    }
}

/// Detecta dinamicamente o sink de hardware padrão do sistema via `pw-metadata`.
///
/// Esta função tenta identificar para qual dispositivo físico o áudio deve ser enviado
/// por padrão. Ela faz o parsing da saída do utilitário `pw-metadata` do PipeWire.
///
/// Retorna `Some(name)` se encontrar um sink válido que não seja o próprio NAM-rs,
/// ou `None` caso contrário (permitindo que o roteamento seja decidido pelo WirePlumber).
pub fn detect_hardware_sink() -> Option<String> {
    // Executa o comando externo para ler metadados do servidor PipeWire
    let out = std::process::Command::new("pw-metadata")
        .args(["-n", "default", "0", "default.audio.sink"])
        .output()
        .ok()?;

    // Converte a saída bruta para string (UTF-8 com perdas)
    let s = String::from_utf8_lossy(&out.stdout);

    // Parsing manual: localiza a chave "name" na saída JSON-like
    let start = s.find("\"name\":\"")?;
    let rest = &s[start + 8..];
    let end = rest.find('"')?;
    let name = &rest[..end];

    // Evitamos o "loop infinito" de roteamento se o default detectado for o próprio input do NAM-rs.
    if name == "NAM-rs-input" || name == "NAM-rs-standalone" {
        None
    } else {
        // Retorna o nome do hardware real (ex: 'alsa_output.pci-0000_00_1f.3.analog-stereo')
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
pub fn poll_rt_status(
    rt_status: &RtStatusFlags,
    sys: &SystemSnapshot,
    was_silent: bool,
    _tsc_anchor: &Anchor,
) -> bool {
    // Verificação de overflow no GC (vazamento de memória para proteger RT)
    if rt_status.gc_overflow.swap(false, Ordering::Relaxed) {
        NamDiagnostic::new(NamErrorCode::GcOverflow, sys)
            .message("Overflow detectado no canal de Garbage Collection (GC).")
            .hint(
                "A thread de áudio teve que 'vazar' memória para evitar drop no hot-path. \
                   Isso pode ocorrer durante trocas ultra-rápidas de modelo. \
                   O NAM-rs drenará o buffer agressivamente agora.",
            )
            .emit_warning();
    }

    // Rate do resampler ativado pelo callback RT (notificação de mudança)
    let rate_notif = rt_status.active_rate_changed.swap(0, Ordering::Relaxed);
    if rate_notif != 0 {
        log::info!(
            "{} Callback RT ativou resampler com rate = {} Hz",
            "✅".green(),
            rate_notif
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
                    "Verifique se o seu usuário possui permissão RT (ulimit -r) \
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

    // Telemetria de timing via RDTSC (minstant)
    let nanos = rt_status.dsp_cycle_time.load(Ordering::Relaxed);
    if nanos > 0 {
        let duration = Duration::from_nanos(nanos);

        // [T26] Exibe percentis acumulados e reseta o histograma a cada ~10 segundos (100 ciclos de poll).
        // Isso reduz drasticamente a verbosidade do terminal mantendo a observabilidade.
        static TELEMETRY_THROTTLE: AtomicU32 = AtomicU32::new(0);
        if TELEMETRY_THROTTLE
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(100)
        {
            let p50 = rt_status.latency_hist.get_percentile(0.50) / 1000;
            let p99 = rt_status.latency_hist.get_percentile(0.99) / 1000;
            let max = rt_status.latency_hist.get_max() / 1000;
            let total_calls = rt_status.latency_hist.total_count();

            log::info!(
                "{} Telemetria DSP (10s): {}µs (Mediana) | {}µs (P99) | {}µs (Máx) [{} blocos]",
                "📊".bright_blue(),
                p50,
                p99,
                max,
                total_calls
            );
            rt_status.latency_hist.reset();
        }

        // Carregamos os valores estáveis uma única vez para evitar race conditions na divisão.
        let rate_val = rt_status.active_rate.load(Ordering::Relaxed);
        let samples_val = rt_status.last_n_samples.load(Ordering::Relaxed);

        if rate_val > 0 && samples_val > 0 {
            let budget_us = (samples_val as f64 / rate_val as f64) * 1_000_000.0;
            let elapsed_us = duration.as_micros() as f64;

            // Se o tempo de execução exceder o budget (100% do tempo do buffer),
            // emitimos um diagnóstico crítico. O dsp_overloads já cuida de avisos a 85%.
            if elapsed_us > budget_us {
                NamDiagnostic::new(NamErrorCode::DeadlineExceeded, sys)
                    .message("Deadline do PipeWire estourado (Possível Xrun detectado)")
                    .hint("Verifique a topologia do modelo ou diminua a carga do sistema.")
                    .param("exec_time_us", elapsed_us as u64)
                    .param("budget_us", budget_us as u64)
                    .param("n_samples", samples_val)
                    .param("rate", rate_val)
                    .emit();
            }
        }
    }

    // Detecção de transição de silêncio (edge-detect: loga apenas na mudança de estado)
    let current_silent = rt_status.is_silent.load(Ordering::Relaxed);
    if current_silent != was_silent {
        if current_silent {
            log::info!(
                "{} Modo Silencioso: Entrada abaixo de −80 dBFS (DSP em espera).",
                "🔇".blue()
            );
        } else {
            log::info!(
                "{} Sinal de Áudio Detectado: Processamento DSP retomado.",
                "🔊".green()
            );
        }
    }

    current_silent
}

/// Drena agressivamente os canais de Garbage Collection para liberar memória.
///
/// Esta função deve ser chamada periodicamente pela thread principal (CLI/UI)
/// ou pelo loop de eventos do host (PipeWire, CLAP). Ela executa o `drop()`
/// dos objetos obsoletos (modelos, resamplers) fora da thread RT.
pub fn drain_gc_channels(
    gc_model: &mut rtrb::Consumer<Box<crate::models::DynamicModel>>,
    gc_rs: &mut rtrb::Consumer<crate::dsp::resampler::NamResampler>,
) {
    // Drena até esvaziar. O drop() pode ser pesado (liberação de heap).
    while let Ok(model) = gc_model.pop() {
        drop(model);
    }
    while let Ok(rs) = gc_rs.pop() {
        drop(rs);
    }
}

/// Calcula os multiplicadores finais combinando ganho do usuário e ajustes do modelo.
/// Operação ultra-rápida (apenas multiplicações lineares) para manter o callback RT leve.
pub fn compute_gain_multipliers(
    u_in_mult: f32,
    u_out_mult: f32,
    m_in_mult: f32,
    m_out_mult: f32,
    out_in_mult: &mut f32,
    out_out_mult: &mut f32,
) {
    *out_in_mult = u_in_mult * m_in_mult;
    *out_out_mult = u_out_mult * m_out_mult;
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
#[cold]
#[inline(never)]
pub fn configure_realtime_thread(target_cpu: usize, rt_status: Arc<RtStatusFlags>) {
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
            "🔒 Proteção de Memória: Travada na RAM física para evitar engasgos (mlockall)."
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

            let confirmed_fifo = base_policy == libc::SCHED_FIFO;

            // Publica resultado real via flags atômicas — zero I/O no caminho quente
            rt_status
                .rt_is_fifo
                .store(confirmed_fifo, Ordering::Relaxed);
            rt_status
                .rt_priority
                .store(actual_param.sched_priority, Ordering::Relaxed);

            // Log inline no cold-path (uma única vez, antes do deadline RT) — aceitável.
            let reset_info = if has_reset_on_fork {
                " | Reset-on-Fork"
            } else {
                ""
            };
            log::info!(
                "{} Otimização de Thread: Núcleo {} dedicado com prioridade Real-Time (FIFO{}, Prio={})",
                "🔍".blue(),
                actual_cpu.to_string().cyan(),
                reset_info,
                actual_param.sched_priority.to_string().green()
            );
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
///
/// A heurística prioriza:
/// 1. **Capacidade Máxima** — Em arquiteturas híbridas (ex: big.LITTLE), prefere núcleos de alta performance.
/// 2. **Menor Carga de IRQ** — Minimiza jitter causado por interrupções de hardware (rede, disco, etc).
/// 3. **Índice Numérico** — Desempate determinístico.
pub fn select_optimal_cpu() -> Option<usize> {
    use std::fs;

    // Varre /sys para encontrar todos os núcleos lógicos disponíveis.
    let cpu_dir = fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut cpus: Vec<usize> = cpu_dir
        .filter_map(|entry| {
            let name = entry.ok()?.file_name();
            let name = name.to_str()?;
            // Filtra apenas diretórios no formato 'cpuX'.
            name.strip_prefix("cpu")?.parse::<usize>().ok()
        })
        .collect();

    if cpus.is_empty() {
        return None;
    }
    cpus.sort_unstable();

    // Obtém a capacidade de processamento de cada núcleo (comum em sistemas ARM ou x86 modernos).
    let capacities: Vec<(usize, u64)> = cpus
        .iter()
        .map(|&cpu| {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
            let cap = fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(1024); // Fallback padrão do kernel.
            (cpu, cap)
        })
        .collect();

    // Totaliza as interrupções processadas por cada core para detectar "vizinhos barulhentos".
    let irq_totals = parse_interrupts_per_cpu(cpus.len());

    capacities
        .iter()
        .map(|&(cpu, cap)| {
            let irqs = irq_totals.get(cpu).copied().unwrap_or(u64::MAX);
            (cpu, cap, irqs)
        })
        .max_by(|a, b| {
            // Heurística de Ordenação:
            // 1. Maior capacidade (cap)
            // 2. Menor número de interrupções (irqs) — invertido no cmp
            // 3. Maior índice de CPU (cpu)
            a.1.cmp(&b.1)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| a.0.cmp(&b.0))
        })
        .map(|(cpu, _, _)| cpu)
}

/// Analisa /proc/interrupts para extrair a carga de interrupções por núcleo.
///
/// Esta função realiza o parsing do formato tabular do kernel, onde cada coluna
/// (após a primeira) representa um contador para um núcleo específico.
pub fn parse_interrupts_per_cpu(num_cpus: usize) -> Vec<u64> {
    use std::fs;

    let mut totals = vec![0u64; num_cpus];

    let content = match fs::read_to_string("/proc/interrupts") {
        Ok(c) => c,
        Err(_) => return totals,
    };

    // Pula o cabeçalho (CPU0 CPU1 ...)
    for line in content.lines().skip(1) {
        let trimmed = line.trim_start();
        // Localiza o delimitador do nome da interrupção (ex: "7: ...")
        let irq_end = trimmed.find(':').unwrap_or(0);
        if irq_end == 0 {
            continue;
        }

        // Filtra apenas interrupções numéricas (ignora NMI, LOC, etc por simplicidade).
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

        // Acumula os contadores para cada core presente na linha.
        for (cpu_idx, token) in after_colon.split_whitespace().enumerate() {
            if cpu_idx >= num_cpus {
                break;
            }
            if let Ok(count) = token.parse::<u64>() {
                totals[cpu_idx] += count;
            } else {
                // Para no primeiro token não numérico (geralmente o nome do driver/dispositivo).
                break;
            }
        }
    }

    totals
}

/// Impede que o processador entre em C-States de economia de energia,
/// garantindo latência de despertar de 0ms para processamento de áudio RT.
///
/// Utiliza a interface PM QoS do kernel Linux para solicitar latência zero.
///
/// RETORNO: O arquivo `File`. Ele DEVE ser mantido vivo no escopo principal.
/// Se o descritor de arquivo for fechado (drop), o kernel anula a proteção.
pub fn lock_cpu_c_states() -> Option<std::fs::File> {
    match std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/cpu_dma_latency")
    {
        Ok(mut file) => {
            // Valor 0 indica tolerância zero a latência de transição de energia.
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
            // Frequentemente falha se não houver permissão de escrita ou se o arquivo não existir.
            log::warn!(
                "PM QoS: Acesso negado a /dev/cpu_dma_latency ({}). \
                 Considere criar uma regra de udev para o grupo 'audio'.",
                e
            );
            None
        }
    }
}
