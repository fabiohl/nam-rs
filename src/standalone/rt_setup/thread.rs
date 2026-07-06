// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Thread and process configuration for real-time operation.
//!
//! Applies CPU affinity, SCHED_FIFO, mlockall, DAZ/FTZ and THP disabling
//! to ensure deterministic execution of the DSP thread.

use crate::common::spsc::RtStatusFlags;
use crate::standalone::colors::Colorize;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Configures the process for real-time operation (process-wide).
///
/// Must be called from `main()` **after** all major heap allocations and
/// **before** starting the PipeWire DSP thread. Runs:
///
/// 1. **THP disable** — Disables Transparent Huge Pages via `prctl`, avoiding
///    background compaction latencies from khugepaged.
/// 2. **mlockall** — Locks current and future memory in physical RAM, preventing
///    page faults in the DSP thread.
///
/// These operations were originally executed in the cold-path of the first DSP frame,
/// but were moved here to reduce jitter at the critical moment of the first
/// audio delivery.
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

/// Configures the current DSP thread for real-time operation.
///
/// Executed **only once** in the cold-path of the first `process()` callback frame,
/// before the data flow actually begins. Applies:
///
/// 1. **DAZ/FTZ** — Enables Denormals-Are-Zero and Flush-To-Zero in the MXCSR register
///    to avoid FPU penalties on silence blocks ("death spiral").
/// 2. **Core Affinity** — Pins the thread to the ideal physical core via
///    `pthread_setaffinity_np`, avoiding core migration and L1/L2 cache misses.
/// 3. **SCHED_FIFO** — Elevates priority to real-time scheduling (prio 90).
///
/// The process-wide operations (THP disable, mlockall) have been moved to
/// `configure_process_wide()`, called from `main()` before PipeWire.
///
/// After configuring, publishes the result via `rt_status` (atomic flags):
/// - `rt_is_fifo`: `true` if `SCHED_FIFO` was confirmed by `pthread_getschedparam`.
/// - `rt_priority`: effective priority granted by the kernel (or `0` if FIFO not obtained).
///
/// The main loop in `run_pipewire_host` reads these flags and emits an auditable
/// confirmation log (or warning) — **zero additional I/O inside the RT callback**.
#[cold]
#[inline(never)]
pub fn configure_realtime_thread(target_cpu: usize, rt_status: Arc<RtStatusFlags>) {
    let cpu_setsize = libc::CPU_SETSIZE as usize;
    let cpu_in_bounds = target_cpu < cpu_setsize;

    if !cpu_in_bounds {
        log::error!(
            "CPU {} is out of bounds (CPU_SETSIZE={}). NAM-rs will continue running without CPU affinity.\n[E2301 | CPU_OUT_OF_BOUNDS] cpu={} max={}",
            target_cpu,
            cpu_setsize,
            target_cpu,
            cpu_setsize - 1,
        );
    }

    unsafe {
        crate::math::common::set_daz_ftz();
    }

    let thread_id = unsafe { libc::pthread_self() };

    unsafe {
        let name = b"nam_rs_dsp\0";
        libc::pthread_setname_np(thread_id, name.as_ptr() as *const libc::c_char);
    }

    if cpu_in_bounds {
        let mut cpuset: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::CPU_ZERO(&mut cpuset);
            libc::CPU_SET(target_cpu, &mut cpuset);
        }

        let ret_aff = unsafe {
            libc::pthread_setaffinity_np(thread_id, std::mem::size_of::<libc::cpu_set_t>(), &cpuset)
        };

        if ret_aff != 0 {
            log::error!(
                "\n  ⚡ Failed to set CPU affinity to core {} (errno={}).\n  💡 NAM-rs will continue running, but may suffer jitter due to Core Migration.\n  [E2301 | CPU_AFFINITY_FAILED] cpu={} errno={}\n",
                target_cpu,
                ret_aff,
                target_cpu,
                ret_aff
            );
        }
    }

    let mut actual_policy = 0i32;
    let mut actual_param: libc::sched_param = unsafe { std::mem::zeroed() };
    let ret_getsched =
        unsafe { libc::pthread_getschedparam(thread_id, &mut actual_policy, &mut actual_param) };

    let actual_cpu = unsafe { libc::sched_getcpu() };
    rt_status.rt_cpu.store(actual_cpu, Ordering::Relaxed);

    if ret_getsched == 0 {
        let reset_on_fork_flag = 0x40000000i32;
        let mut has_reset_on_fork = (actual_policy & reset_on_fork_flag) != 0;
        let mut base_policy = actual_policy & !reset_on_fork_flag;

        if base_policy != libc::SCHED_FIFO {
            let mut param: libc::sched_param = unsafe { std::mem::zeroed() };
            param.sched_priority = 90;

            let ret_sched =
                unsafe { libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param) };

            if ret_sched != 0 {
                log::error!(
                    "⚠️ pthread_setschedparam(SCHED_FIFO, 90) failed (errno={}).\n  [E2302 | RT_SCHED_FAILED] Check ulimit -r and rtkit permissions.\n",
                    ret_sched
                );
            } else {
                base_policy = libc::SCHED_FIFO;
                actual_param.sched_priority = 90;
                has_reset_on_fork = false;
            }
        }

        let confirmed_fifo = base_policy == libc::SCHED_FIFO;

        if confirmed_fifo {
            rt_status.set_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
        } else {
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
        }

        rt_status
            .rt_priority
            .store(actual_param.sched_priority, Ordering::Relaxed);
        rt_status
            .confirmed_priority
            .store(actual_param.sched_priority, Ordering::Relaxed);
        rt_status.rt_policy.store(base_policy, Ordering::Relaxed);

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
        rt_status.clear_flag(crate::common::spsc::RT_STATUS_RT_IS_FIFO);
        rt_status.rt_priority.store(0, Ordering::Relaxed);
        rt_status.confirmed_priority.store(-1, Ordering::Relaxed);
        rt_status.rt_policy.store(-1, Ordering::Relaxed);

        log::error!(
            "  [E2303 | RT_GETSCHED_FAILED] pthread_getschedparam failed (ret={}).\n",
            ret_getsched
        );
    }
}
