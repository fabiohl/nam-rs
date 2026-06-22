// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};

/// Global flag for coordinated graceful shutdown across all threads.
/// Set to `true` by the CTRL+C handler.
pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Flag indicating that the DSP thread needs a new `NamResampler`.
pub const RT_STATUS_NEEDS_RESAMPLER_REBUILD: u64 = 1 << 0;
/// Indicates whether the last resampler rebuild attempt by the main thread failed.
pub const RT_STATUS_RESAMPLER_REBUILD_FAILED: u64 = 1 << 1;
/// `true` if the DSP thread confirmed operation under `SCHED_FIFO`.
pub const RT_STATUS_RT_IS_FIFO: u64 = 1 << 2;
/// Flag indicating that saturation (clipping) occurred on the output audio.
pub const RT_STATUS_HAS_CLIPPED: u64 = 1 << 3;
/// Flag indicating that the current buffer is completely silent (Gate closed).
pub const RT_STATUS_IS_SILENT: u64 = 1 << 4;
/// Flag indicating that the GC channel overflowed.
pub const RT_STATUS_GC_OVERFLOW: u64 = 1 << 5;
/// Flag indicating that the gate is transitioning (Fade-In or Fade-Out).
pub const RT_STATUS_IS_FADING: u64 = 1 << 6;
/// Flag indicating that a critical model load failure occurred on the RT thread.
pub const RT_STATUS_MODEL_LOAD_FAILED: u64 = 1 << 7;
/// Flag indicating that a heap allocation occurred on the RT thread (detected by heap-audit).
pub const RT_STATUS_HEAP_ALLOC: u64 = 1 << 8;
/// Flag indicating that the RT callback should pause DSP processing until
/// the resampler is replaced (during hot-plug or sample rate change).
pub const RT_STATUS_RESAMP_SWAP_PENDING: u64 = 1 << 10;
/// Flag indicating that at least one huge-page allocation succeeded
/// (set by main thread after alloc, checked by telemetry for logging).
pub const RT_STATUS_HUGEPAGE_OK: u64 = 1 << 11;
/// Soft-degrade: model running in Reduced mode
/// (fewer WaveNet layers or LSTM single-layer).
pub const RT_STATUS_DEGRADE_REDUCED: u64 = 1 << 12;
/// Soft-degrade: model running in Minimal mode
/// (maximum reduction — passthrough for LSTM, half-WaveNet).
pub const RT_STATUS_DEGRADE_MINIMAL: u64 = 1 << 13;
/// Flag indicating that a cabsim rebuild is needed
/// (partition_size no longer matches current buffer size).
pub const RT_STATUS_NEEDS_CABSIM_REBUILD: u64 = 1 << 9;
/// Flag indicating that a corrupted/malformed GC item was detected
/// (unknown type_id or inconsistent type+ptr in overflow buffer).
pub const RT_STATUS_GC_CORRUPTED: u64 = 1 << 14;
/// Flag indicating that the A2 static variant triggered the scalar fallback path
/// (no CH=3 or CH=8 conv available). Set by the RT thread for telemetry polling.
pub const RT_STATUS_A2_FALLBACK_TRIGGERED: u64 = 1 << 15;
/// Flag indicating that a WaveNet slimmable slice_channels rebuild failed on the RT thread.
/// Replaces `log::error!` for RT-zero-IO compliance.
pub const RT_STATUS_SLIMMABLE_SLICE_FAILED: u64 = 1 << 16;
/// Flag indicating that a WaveNet slimmable rebuild is needed (set by RT, cleared by main).
pub const RT_STATUS_NEEDS_SLIMMABLE_REBUILD: u64 = 1 << 17;

/// Atomic status flags for silent RT→Main communication.
///
/// The DSP thread sets atomic flags instead of calling `println!`/`eprintln!`.
/// The main thread reads these flags periodically and prints logs to the user.
/// This ensures **zero I/O** occurs in the RT callback.
///
/// ### Bitmask Map (`status_bits`)
///
/// | Bit | Constant | Description |
/// | :--- | :--- | :--- |
/// | 0 | `NEEDS_RESAMPLER_REBUILD` | DSP thread requests new resampler |
/// | 1 | `RESAMPLER_REBUILD_FAILED` | Resampler rebuild failed |
/// | 2 | `RT_IS_FIFO` | SCHED_FIFO active confirmed |
/// | 3 | `HAS_CLIPPED` | Output saturation (clipping) |
/// | 4 | `IS_SILENT` | Buffer completely silent (Gate Closed) |
/// | 5 | `GC_OVERFLOW` | Garbage Collection channel overflow |
/// | 6 | `IS_FADING` | Gate transitioning (Fading In/Out) |
/// | 7 | `MODEL_LOAD_FAILED` | Model load failure on RT thread |
/// | 8 | `HEAP_ALLOC` | Heap allocation detected on RT thread |
/// | 9 | `NEEDS_CABSIM_REBUILD` | DSP thread requests cabsim engine rebuild |
/// | 10 | `RESAMP_SWAP_PENDING` | RT callback paused awaiting resampler swap |
/// | 11 | `HUGEPAGE_OK` | Huge-page allocation confirmed active |
/// | 12 | `DEGRADE_REDUCED` | Soft-degrade active — Reduced mode |
/// | 13 | `DEGRADE_MINIMAL` | Soft-degrade active — Minimal mode |
/// | 14 | `GC_CORRUPTED` | GC overflow buffer corrupted (unknown type/ptr) |
/// | 15 | `A2_FALLBACK_TRIGGERED` | A2 static variant fell back to scalar zero-output path |
/// | 16 | `SLIMMABLE_SLICE_FAILED` | WaveNet slimmable slice_channels rebuild failed |
/// | 17 | `NEEDS_SLIMMABLE_REBUILD` | DSP thread requests slimmable model rebuild |
#[repr(align(128))]
pub struct RtStatusFlags {
    /// Effective sample rate active on the DSP thread after resampler rebuild.
    /// Set by the DSP thread upon consuming a new `NamResampler` from the SPSC channel.
    /// Value `0` indicates no pending update.
    pub active_rate: AtomicU32,
    /// Rate change notification for logging purposes.
    /// Value `0` indicates no change since the last poll.
    pub active_rate_changed: AtomicU32,

    /// Target rate detected by the DSP thread from PipeWire but not yet applied (awaiting rebuild).
    /// The main thread reads this value to know which rate to build.
    /// Value `0` indicates no pending request.
    pub requested_pw_rate: AtomicU32,

    /// Target rate of the loaded model (NAM). The usual default is 48000.
    pub requested_nam_rate: AtomicU32,

    /// Effective RT priority confirmed by `pthread_getschedparam`.
    /// Value `-1` indicates the check has not yet been performed.
    /// Set on the cold-path of the DSP thread's first frame.
    pub rt_priority: AtomicI32,

    /// Atomic counter of DSP overloads (virtual XRUNs).
    /// Incremented by the RT callback if processing exceeds 85% of the time budget.
    pub dsp_overloads: AtomicU32,

    /// Processing time of the last DSP cycle in ticks (RDTSC).
    /// Read by the main thread and converted to Duration via Anchor.
    pub dsp_cycle_time: AtomicU64,

    /// Number of samples processed in the last cycle (for budget calculation).
    pub last_n_samples: AtomicU32,

    /// Latency histogram for statistical analysis (P50, P95, P99).
    pub latency_hist: crate::dsp::telemetry::LatencyHistogram,

    /// Total degradation transitions that have occurred (Full↔Reduced↔Minimal).
    pub degrade_transitions_total: AtomicU32,

    /// Atomic bitmask containing binary states (needs_rebuild, clipped, silent, etc).
    /// Reduces Cache Bouncing by condensing multiple states into a single cache line.
    pub status_bits: AtomicU64,

    /// Confirmed RT priority.
    pub confirmed_priority: AtomicI32,
    /// Confirmed RT scheduling policy.
    pub rt_policy: AtomicI32,
    /// Pinned physical CPU core (or -1 if not pinned).
    pub rt_cpu: AtomicI32,
    /// Accumulated OR of all RT_STATUS_* flags ever seen since startup.
    pub flags_seen: AtomicU64,
    /// Total count of virtual XRUNs/overloads.
    pub xruns: AtomicU32,
    /// Total count of GC items successfully drained.
    pub drains: AtomicU32,
    /// Requested partition size for cabsim rebuild (set by RT thread).
    pub requested_cabsim_partition_size: AtomicU32,
    /// Requested slimmable channel count (set by RT thread, read by main thread).
    /// Value `0` indicates no pending request.
    pub requested_slimmable_ch: AtomicU32,
}

impl RtStatusFlags {
    /// Creates a new instance with zero/sentinel initial values.
    #[cold]
    pub fn new() -> Self {
        Self {
            active_rate: AtomicU32::new(0),
            active_rate_changed: AtomicU32::new(0),
            requested_pw_rate: AtomicU32::new(0),
            requested_nam_rate: AtomicU32::new(48_000),
            rt_priority: AtomicI32::new(-1),
            dsp_overloads: AtomicU32::new(0),
            dsp_cycle_time: AtomicU64::new(0),
            last_n_samples: AtomicU32::new(0),
            latency_hist: crate::dsp::telemetry::LatencyHistogram::new(),
            degrade_transitions_total: AtomicU32::new(0),
            status_bits: AtomicU64::new(0),
            confirmed_priority: AtomicI32::new(-1),
            rt_policy: AtomicI32::new(-1),
            rt_cpu: AtomicI32::new(-1),
            flags_seen: AtomicU64::new(0),
            xruns: AtomicU32::new(0),
            drains: AtomicU32::new(0),
            requested_cabsim_partition_size: AtomicU32::new(0),
            requested_slimmable_ch: AtomicU32::new(0),
        }
    }

    /// Sets one or more flags in the bitmask.
    #[inline(always)]
    pub fn set_flag(&self, flag: u64) {
        self.status_bits.fetch_or(flag, Ordering::Relaxed);
    }

    /// Clears one or more flags in the bitmask.
    #[inline(always)]
    pub fn clear_flag(&self, flag: u64) {
        self.status_bits.fetch_and(!flag, Ordering::Relaxed);
    }

    /// Checks whether a flag is active.
    #[inline(always)]
    pub fn check_flag(&self, flag: u64) -> bool {
        (self.status_bits.load(Ordering::Relaxed) & flag) != 0
    }

    /// Checks whether a flag is active and clears it atomically in a single operation.
    /// Returns `true` if the flag was active.
    #[inline(always)]
    pub fn check_and_clear_flag(&self, flag: u64) -> bool {
        let old = self.status_bits.fetch_and(!flag, Ordering::Relaxed);
        let active = (old & flag) != 0;
        if active {
            self.flags_seen.fetch_or(flag, Ordering::Relaxed);
        }
        active
    }
}

impl Default for RtStatusFlags {
    fn default() -> Self {
        Self::new()
    }
}
