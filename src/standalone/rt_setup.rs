// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![cfg(target_arch = "x86_64")]

//! Utilitários para configuração e monitoramento de tempo real e hardware.
//!
//! Este módulo contém funções auxiliares para gerenciar afinidade de CPU,
//! prioridades SCHED_FIFO, travamento de memória (mlockall), e telemetria
//! de status RT do motor de áudio.

use crate::common::diagnostics::{NamDiagnostic, NamErrorCode, SystemSnapshot};
use crate::common::spsc::RtStatusFlags;
use crate::standalone::colors::Colorize;

use minstant::Anchor;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Frequência calibrada do TSC em GHz (ciclos por nanosegundo).
/// Armazenado como ponto fixo (valor * 1000) para evitar floats no hot-path.
static TSC_FREQ_GHZ_X1000: AtomicU64 = AtomicU64::new(0);
/// Âncora temporal para o fallback de rdtsc (monotônico).
static BOOT_TIME: OnceLock<Instant> = OnceLock::new();

/// Retorna o tempo atual em nanosegundos usando a instrução RDTSC.
///
/// Oferece precisão de ~1ns com custo de ~1 ciclo, evitando a syscall vDSO clock_gettime.
/// Se o TSC não estiver calibrado ou disponível, faz fallback para Instant::now().
#[inline(always)]
pub fn rdtsc_nanos() -> u64 {
    let freq_x1000 = TSC_FREQ_GHZ_X1000.load(Ordering::Relaxed);

    // SAFETY: Divisão por zero no hot-path é fatal. Se a calibração falhou
    // no boot, usamos o relógio do sistema como rede de segurança.
    #[allow(clippy::manual_checked_ops)]
    if freq_x1000 != 0 {
        let cycles = unsafe { core::arch::x86_64::_rdtsc() };
        (cycles * 1000) / freq_x1000
    } else {
        BOOT_TIME.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// Calibra a frequência do TSC (Time Stamp Counter) em relação ao relógio do sistema.
///
/// Imagine que a CPU tem um "odômetro" interno que conta cada batida do seu coração (ciclo).
/// Como a velocidade da CPU pode variar, precisamos descobrir quantos "batimentos"
/// equivalem a 1 nanosegundo real para podermos medir o tempo com precisão cirúrgica.
///
/// Esta função é executada apenas uma vez no início do programa (cold-path).
#[cold]
pub fn calibrate_tsc() {
    use std::thread;

    // 1. AQUECIMENTO:
    // Chamamos a instrução uma vez e esperamos um pouco. Isso garante que a CPU
    // "acorde" de estados de baixo consumo e que os dados estejam prontos nos caches.
    let _ = unsafe { core::arch::x86_64::_rdtsc() };
    thread::sleep(Duration::from_millis(10));

    // 2. MARCO ZERO (Início da Medição):
    // Capturamos simultaneamente o tempo do relógio do sistema (lento mas confiável)
    // e o valor do contador de ciclos da CPU (ultra-rápido).
    let start_inst = Instant::now();
    let start_tsc = unsafe { core::arch::x86_64::_rdtsc() };

    // 3. ESPERA CONTROLADA:
    // Aguardamos 50 milissegundos. É um tempo curto para o humano, mas permite que
    // a CPU realize milhões de ciclos, reduzindo erros de amostragem.
    thread::sleep(Duration::from_millis(50));

    // 4. MARCO FINAL:
    // Capturamos novamente os dois valores para calcular quanto cada um avançou.
    let end_inst = Instant::now();
    let end_tsc = unsafe { core::arch::x86_64::_rdtsc() };

    let elapsed_nanos = end_inst.duration_since(start_inst).as_nanos() as u64;
    let elapsed_cycles = end_tsc.wrapping_sub(start_tsc);

    // 5. CÁLCULO DA TAXA DE CONVERSÃO:
    // Calculamos a relação: (Ciclos * 1000) / Nanosegundos.
    // Usamos o multiplicador 1000 para guardar o resultado como um número inteiro
    // mantendo 3 casas decimais de precisão sem precisar de ponto flutuante (floats).
    if let Some(freq_x1000) = (elapsed_cycles * 1000).checked_div(elapsed_nanos) {
        TSC_FREQ_GHZ_X1000.store(freq_x1000, Ordering::SeqCst);

        log::info!(
            "{} Relógio de Alta Precisão (TSC) calibrado em {:.3} GHz",
            "⏱️".bright_blue(),
            freq_x1000 as f64 / 1000.0
        );
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

/// Lê flags atômicas de status RT e emite logs de monitoramento para o usuário.
///
/// Esta função atua como o "painel de instrumentos" do NAM-rs. Ela é chamada periodicamente
/// para traduzir os sinais técnicos vindos da thread de áudio (que é silenciosa e ultra-veloz)
/// em mensagens compreensíveis, avisos de performance e telemetria de latência.
///
/// Retorna uma tupla (current_silent, current_fading) para controle de estado no loop principal.
pub fn poll_rt_status(
    rt_status: &RtStatusFlags,
    sys: &SystemSnapshot,
    was_silent: bool,
    was_fading: bool,
    _tsc_anchor: &Anchor,
    bridge: &crate::dsp::pipeline::DspBridge,
) -> (bool, bool) {
    // 1. GERENCIAMENTO DE MEMÓRIA (Garbage Collection):
    // Se o canal de limpeza estiver cheio, significa que estamos trocando modelos neurais
    // mais rápido do que o sistema consegue descartar os antigos. Priorizamos o áudio
    // "vazando" memória temporariamente para evitar estalos (drops) no som.
    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW) {
        NamDiagnostic::new(NamErrorCode::GcOverflow, sys)
            .message("Overflow detectado no canal de Garbage Collection (GC).")
            .hint(
                "A thread de áudio teve que 'vazar' memória para evitar drop no hot-path. \
                   Isso pode ocorrer durante trocas ultra-rápidas de modelo. \
                   O NAM-rs drenará o buffer agressivamente agora.",
            )
            .emit_warning();
    }

    // 2. MUDANÇA DE RITMO (Sample Rate):
    // Avisa quando o servidor de áudio (PipeWire) altera a frequência de amostragem
    // (ex: mudou de 44.100 para 48.000 batidas por segundo).
    let rate_notif = rt_status.active_rate_changed.swap(0, Ordering::Relaxed);
    if rate_notif != 0 {
        log::info!(
            "{} Callback RT ativou resampler com rate = {} Hz",
            "✅".green(),
            rate_notif
        );
    }

    // 3. DISTORÇÃO DIGITAL (Clipping):
    // O equivalente ao "LED vermelho" em mesas de som. Indica que o volume do sinal
    // ultrapassou o limite máximo do processamento digital.
    if rt_status.check_and_clear_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED) {
        log::warn!(
            "{} Saturação detectada (Clipping)! Considere reduzir o ganho de entrada e/ou saída.",
            "🔥".bright_red().bold()
        );
    }

    // 4. PRIORIDADE DE TEMPO REAL (Real-Time Priority):
    // Verifica se o Linux permitiu que o NAM-rs rode com "prioridade máxima".
    // Isso impede que outros programas (como o navegador) causem interrupções no áudio.
    let prio = rt_status.rt_priority.load(Ordering::Relaxed);
    if prio != -1 {
        let is_fifo = rt_status.check_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);

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

    // 5. SOBRECARGA DE PROCESSAMENTO (Overloads):
    // Avisa se o processador (CPU) não está sendo rápido o suficiente para calcular
    // a rede neural antes do próximo bloco de áudio ser necessário.
    let overloads = rt_status.dsp_overloads.swap(0, Ordering::Relaxed);
    if overloads > 0 {
        log::warn!(
            "{} Sobrecarga de CPU ({} buffers). Considere usar um modelo mais leve ou processador mais poderoso.",
            "🚨".red(),
            overloads
        );
    }

    // 6. SINCRONIA DE PLACAS (Clock Drifting):
    // Ocorre quando você usa dispositivos diferentes para entrada e saída (ex: Microfone USB
    // e Fone P2). Se um for levemente mais rápido que o outro, o sistema precisa descartar
    // alguns pequenos pedaços de áudio para manter a sincronia.
    let drops = bridge.drain_dropped_frames();
    if drops > 0 {
        log::warn!(
            "{} Drifting detectado: {} blocos de áudio descartados (capture > playback).",
            "⚠️".yellow(),
            drops
        );
    }

    // 7. TELEMETRIA DE PERFORMANCE (Latência):
    // Mostra estatísticas de quanto tempo a CPU leva para processar cada bloco.
    // - Mediana (P50): O tempo "comum" de processamento.
    // - P99: O pior caso em 99% das vezes (indica estabilidade).
    // - Máx: O maior pico de atraso já registrado.
    let nanos = rt_status.dsp_cycle_time.load(Ordering::Relaxed);
    if nanos > 0 {
        let duration = Duration::from_nanos(nanos);
        static TELEMETRY_THROTTLE: AtomicU32 = AtomicU32::new(0);
        if TELEMETRY_THROTTLE
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_rem(100)
            == 0
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

        // 8. VERIFICAÇÃO DE PRAZO (Deadline):
        // Se o tempo de execução ultrapassar o "orçamento" dado pelo sistema de áudio,
        // geramos um erro de diagnóstico explicando o que falhou.
        let rate_val = rt_status.active_rate.load(Ordering::Relaxed);
        let samples_val = rt_status.last_n_samples.load(Ordering::Relaxed);

        if rate_val > 0 && samples_val > 0 {
            let budget_us = (samples_val as f64 / rate_val as f64) * 1_000_000.0;
            let elapsed_us = duration.as_micros() as f64;

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

    // Detecção de transição de silêncio
    let current_silent = rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT);

    if current_silent != was_silent {
        if current_silent {
            log::info!(
                "{} Modo Silencioso: Entrada abaixo do limiar (Gate Fechado).",
                "🔇".blue()
            );
        } else {
            log::info!(
                "{} Sinal de Áudio Detectado: Processamento DSP retomado.",
                "🔊".green()
            );
        }
    }

    // Detecção de transição de fading
    let current_fading = rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING);

    if current_fading != was_fading && current_fading {
        log::info!("{} Transição de Sinal: Gate em Fade-In/Out.", "🌓".yellow());
    }

    (current_silent, current_fading)
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

/// Configura o processo para operação de tempo real (process-wide).
///
/// Deve ser chamada de `main()` **após** todas as alocações heap principais e
/// **antes** de iniciar a thread DSP do PipeWire. Executa:
///
/// 1. **THP disable** — Desabilita Transparent Huge Pages via `prctl`, evitando
///    latências de compactação em background pelo khugepaged.
/// 2. **mlockall** — Bloqueia a memória atual e futura na RAM física, prevenindo
///    page faults na thread DSP.
///
/// Estas operações eram originalmente executadas no cold-path do primeiro frame DSP,
/// mas foram movidas para cá para reduzir o jitter no momento crítico da primeira
/// entrega de áudio.
pub fn configure_process_wide() {
    unsafe {
        libc::prctl(libc::PR_SET_THP_DISABLE, 1, 0, 0, 0);
    }

    let ret_mlock = unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) };

    if ret_mlock != 0 {
        let err = std::io::Error::last_os_error();
        log::warn!(
            "mlockall() falhou ({}). O áudio pode sofrer engasgos se o sistema usar swap.\n  Dica: Verifique o limite de 'memlock' no ulimits.",
            err
        );
    } else {
        log::info!(
            "🔒 Proteção de Memória: Travada na RAM física para evitar engasgos (mlockall)."
        );
    }
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
/// As operações process-wide (THP disable, mlockall) foram movidas para
/// `configure_process_wide()`, chamada de `main()` antes do PipeWire.
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
        crate::math::common::set_daz_ftz();
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
            if confirmed_fifo {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
            } else {
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
            }

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
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
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
/// 1. **Filtro de Afinidade** — Respeita os núcleos autorizados pelo SO (isolcpus, cgroups).
/// 2. **Capacidade Máxima** — Em arquiteturas híbridas (ex: big.LITTLE), prefere núcleos de alta performance.
/// 3. **Menor Carga de IRQ** — Minimiza jitter causado por interrupções de hardware (rede, disco, etc).
/// 4. **Índice Numérico** — Desempate determinístico.
pub fn select_optimal_cpu() -> Option<usize> {
    use std::fs;

    // 1. Obtém os núcleos que o sistema operacional permitiu para este processo.
    let allowed_cpus = get_allowed_cpus();
    if allowed_cpus.is_empty() {
        log::warn!(
            "Não foi possível detectar CPUs permitidas via sched_getaffinity. Usando fallback total."
        );
    }

    // Varre /sys para encontrar todos os núcleos lógicos disponíveis.
    let cpu_dir = fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut cpus: Vec<usize> = cpu_dir
        .filter_map(|entry| {
            let name = entry.ok()?.file_name();
            let name = name.to_str()?;
            // Filtra apenas diretórios no formato 'cpuX'.
            let cpu_idx = name.strip_prefix("cpu")?.parse::<usize>().ok()?;

            // Se tivermos a lista de permitidos, filtramos por ela.
            if !allowed_cpus.is_empty() && !allowed_cpus.contains(&cpu_idx) {
                return None;
            }

            Some(cpu_idx)
        })
        .collect();

    if cpus.is_empty() {
        return None;
    }
    cpus.sort_unstable();

    // Obtém a capacidade de processamento de cada núcleo.
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

/// Obtém a lista de CPUs permitidas para o processo atual via `sched_getaffinity`.
///
/// Respeita isolamento de CPUs (isolcpus), cgroups e máscaras de afinidade
/// impostas pelo sistema operacional ou pelo usuário (ex: taskset).
fn get_allowed_cpus() -> Vec<usize> {
    let mut allowed = Vec::new();
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);

        // Chamada ao kernel para obter a máscara de afinidade do processo atual (pid 0)
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut cpuset) == 0 {
            for i in 0..libc::CPU_SETSIZE as usize {
                if libc::CPU_ISSET(i, &cpuset) {
                    allowed.push(i);
                }
            }
        }
    }
    allowed
}

/// Analisa /proc/interrupts para extrair a carga de interrupções por núcleo.
///
/// Esta função realiza o parsing do formato tabular do kernel, onde cada coluna
/// (após a primeira) representa um contador para um núcleo específico.
pub fn parse_interrupts_per_cpu(num_cpus: usize) -> Vec<u64> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let mut totals = vec![0u64; num_cpus];

    // Refatoração para Streaming: Evita alocação monolítica de string para /proc/interrupts.
    // Essencial para sistemas com alto número de CPUs onde o arquivo pode ser grande.
    let file = match File::open("/proc/interrupts") {
        Ok(f) => f,
        Err(_) => return totals,
    };
    let reader = BufReader::new(file);

    // Pula o cabeçalho (CPU0 CPU1 ...)
    for line_result in reader.lines().skip(1) {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => break,
        };
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

#[cfg(test)]
#[path = "rt_setup_test.rs"]
mod rt_setup_test;
