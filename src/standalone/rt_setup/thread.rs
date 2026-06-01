// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Configuração de thread e processo para operação de tempo real.
//!
//! Aplica afinidade de CPU, SCHED_FIFO, mlockall, DAZ/FTZ e desabilitação
//! de THP para garantir execução determinística da thread DSP.

use crate::common::spsc::RtStatusFlags;
use crate::standalone::colors::Colorize;
use std::sync::Arc;
use std::sync::atomic::Ordering;

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
            "mlockall() failed ({}). Audio may experience dropouts if the system swaps.\n  Hint: Verify the 'memlock' limit in ulimits.",
            err
        );
    } else {
        log::info!("🔒 Memory Protection: Locked in physical RAM to prevent dropouts (mlockall).");
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
                "\n  ⚡ Failed to set CPU affinity to core {} (errno={}).\n  💡 NAM-rs will continue running, but may suffer jitter due to Core Migration.\n  [E2301 | CPU_AFFINITY_FAILED] cpu={} errno={}\n",
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
                        "⚠️ pthread_setschedparam(SCHED_FIFO, 90) failed (errno={}).\n  [E2302 | RT_SCHED_FAILED] Check ulimit -r and rtkit permissions.\n",
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
                "{} Thread Optimization: Dedicated core {} with Real-Time priority (FIFO{}, Prio={})",
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
                "  [E2303 | RT_GETSCHED_FAILED] pthread_getschedparam failed (ret={}).\n",
                ret_getsched
            );
        }
    }
}
