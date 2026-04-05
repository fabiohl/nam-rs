// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Audio DSP core using `nih_plug`.
//! Contains the strict Real-Time/DSP thread callback responsible for Bit-Perfect capture.
//! Receives raw f32 samples from the PipeWire host via the `nih_plug` standalone backend,
//! interleaves them into cache-aligned blocks, and pushes them to the lock-free SPSC ring buffer
//! for consumption by the I/O thread. Zero heap allocation, zero I/O, zero mutexes
//! during `process()`.

use nih_plug::prelude::*;
use rtrb::Producer;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use crate::buffer::{AlignedBlock, AudioMetadata, MAX_BLOCK_SIZE, OVERRUN_COUNT, RingPayload};

/// Tolerance threshold for considering an audio signal as absolute silence.
/// Used by the Zero-Overhead Noise Gate to suppress recording of empty blocks.
const SILENCE_THRESHOLD: f32 = 1e-6;

/// Global storage to inject the Producer into the plugin instance created by `nih_plug` standalone.
/// The `OnceLock<Mutex<Option<...>>>` pattern is used because `nih_plug` instantiates the plugin
/// internally via `Default::default()`, making it necessary to inject the producer through a global.
/// The Mutex is locked only once during `Default::default()`, never during `process()`.
pub static PRODUCER: OnceLock<Mutex<Option<Producer<RingPayload<MAX_BLOCK_SIZE>>>>> =
    OnceLock::new();

/// Checks whether the audio block is considered silent.
/// Using simple iterators on contiguous memory slices allows the LLVM to apply
/// auto-vectorization (SIMD/AVX2) without requiring unsafe code or extra crates.
#[inline(always)]
fn is_silent(data: &[f32]) -> bool {
    data.iter().all(|&sample| sample.abs() < SILENCE_THRESHOLD)
}

/// AudioRip plugin implementation.
/// Holds the ring buffer producer and tracks whether stream metadata has been sent.
pub struct AudioRipPlugin {
    params: Arc<AudioRipParams>,
    /// Ring buffer producer, injected via the global `PRODUCER` during `Default::default()`.
    pub producer: Option<Producer<RingPayload<MAX_BLOCK_SIZE>>>,
    /// Indicates whether metadata (sample rate, bit depth, channels) has been sent in the current session.
    metadata_sent: bool,
    /// Indicates whether the RT thread configuration (core affinity, scheduler) has been applied.
    thread_configured: bool,
}

/// AudioRip plugin parameters.
/// Empty struct because the passive ripper has no user-exposed parameters,
/// but the `nih_plug` `Params` trait requires a parameters struct.
#[derive(Params)]
pub struct AudioRipParams {}

impl Default for AudioRipPlugin {
    fn default() -> Self {
        let producer = if let Some(mutex) = PRODUCER.get() {
            if let Ok(mut guard) = mutex.lock() {
                guard.take()
            } else {
                None
            }
        } else {
            None
        };

        Self {
            params: Arc::new(AudioRipParams {}),
            producer,
            metadata_sent: false,
            thread_configured: false,
        }
    }
}

impl Plugin for AudioRipPlugin {
    const NAME: &'static str = "AudioRip";
    const VENDOR: &'static str = "Fabio Lima";
    const URL: &'static str = "";
    const EMAIL: &'static str = "fabio.henrique.lima.silva@gmail.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: std::num::NonZeroU32::new(2),
        main_output_channels: std::num::NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = false;
    type SysExMessage = ();

    // Opaque background task type — unused; required by nih_plug.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        _buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Reset state for a new capture session
        self.metadata_sent = false;
        true
    }

    fn reset(&mut self) {
        self.metadata_sent = false;

        // Push the stop signal to the Consumer without allocating memory on the DSP thread
        if let Some(producer) = &mut self.producer {
            let _ = producer.push(RingPayload::StreamStop);
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Apply the RT thread configuration only once, on the first callback invocation.
        // Runs before any audio data processing.
        if !self.thread_configured {
            self.configure_realtime_thread();
            self.thread_configured = true;
        }

        let samples = buffer.samples();
        let channels = buffer.channels();
        let sample_rate = context.transport().sample_rate;

        // Prevent downstream panics if the host passes an empty/dead buffer momentarily
        if samples == 0 || channels == 0 || sample_rate <= 0.0 {
            return ProcessStatus::Normal;
        }

        if let Some(producer) = &mut self.producer {
            // Send stream metadata to the I/O thread on the first buffer or after reset.
            // Allows the I/O thread to create a correctly formatted WAV header.
            if !self.metadata_sent {
                // nih_plug delivers f32 natively; PipeWire translates transparently.
                let bit_depth = 32;
                let channels = channels as u16;

                let meta = AudioMetadata {
                    sample_rate,
                    bit_depth,
                    channels,
                };

                if producer.push(RingPayload::Metadata(meta)).is_err() {
                    OVERRUN_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                self.metadata_sent = true;
            }

            // ⚡ HOT PATH: Audio Interleaving ⚡
            // Transposes non-interleaved arrays from `nih_plug` natively into a
            // 128-byte-aligned lock-free block structure. This prevents cache
            // bouncing and never touches the default memory allocators (`Box`, `Vec`).
            let mut block = AlignedBlock::<MAX_BLOCK_SIZE>::new();
            let mut block_idx = 0;

            for sample_idx in 0..samples {
                for ch in 0..channels {
                    if block_idx >= MAX_BLOCK_SIZE {
                        // Block full — push with exact valid_len and start a new one.
                        block.valid_len = MAX_BLOCK_SIZE;

                        // Zero-Overhead Noise Gate: submit audio only if not absolute silence.
                        if !is_silent(&block.data[..block.valid_len])
                            && producer.push(RingPayload::Audio(block)).is_err()
                        {
                            OVERRUN_COUNT.fetch_add(1, Ordering::Relaxed);
                        }

                        block = AlignedBlock::<MAX_BLOCK_SIZE>::new();
                        block_idx = 0;
                    }

                    // Native memory read, bit-perfect pass-through to the ring buffer.
                    block.data[block_idx] = buffer.as_slice()[ch][sample_idx];
                    block_idx += 1;
                }
            }

            // Push the residual block with the precise count of valid samples.
            // Only `valid_len` samples will be written to the WAV file by the I/O thread.
            if block_idx > 0 {
                block.valid_len = block_idx;

                // Zero-Overhead Noise Gate applied to the residual block.
                if !is_silent(&block.data[..block.valid_len])
                    && producer.push(RingPayload::Audio(block)).is_err()
                {
                    OVERRUN_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Silence the active buffer to prevent unwanted feedback in the user's headphones,
            // since nih_plug processes strictly in-place.
            for ch_slice in buffer.as_slice() {
                ch_slice.fill(0.0);
            }
        }

        ProcessStatus::Normal
    }
}

impl AudioRipPlugin {
    /// Selects the optimal CPU core for RT thread pinning.
    ///
    /// Selection criteria (in tiebreaker order):
    /// 1. Highest `cpu_capacity` from `/sys/devices/system/cpu/cpuN/cpu_capacity`
    /// 2. Fewest total interrupts from `/proc/interrupts`
    /// 3. Highest CPU index number (final tiebreaker)
    ///
    /// Returns the selected CPU index, or `None` if detection fails entirely.
    fn select_optimal_cpu() -> Option<usize> {
        use std::fs;

        // Discover available logical CPUs from sysfs
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

        // 1. Read cpu_capacity for each CPU (default 1024 if missing — ARM DynamIQ / EAS value)
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

        // 2. Parse total interrupts per CPU from /proc/interrupts
        let irq_totals = Self::parse_interrupts_per_cpu(cpus.len());

        // 3. Build composite score: (capacity DESC, -total_interrupts, cpu_index DESC)
        //    We maximize capacity and cpu_index, minimize total_interrupts.
        capacities
            .iter()
            .map(|&(cpu, cap)| {
                let irqs = irq_totals.get(cpu).copied().unwrap_or(u64::MAX);
                (cpu, cap, irqs)
            })
            .max_by(|a, b| {
                // Primary: highest capacity
                a.1.cmp(&b.1)
                    // Secondary: fewest interrupts (reverse comparison)
                    .then_with(|| b.2.cmp(&a.2))
                    // Tertiary: highest CPU index
                    .then_with(|| a.0.cmp(&b.0))
            })
            .map(|(cpu, _, _)| cpu)
    }

    /// Parses `/proc/interrupts` and returns a `Vec` indexed by CPU number
    /// containing the total interrupt count across all IRQ lines for each CPU.
    ///
    /// Only lines with numeric IRQ identifiers are counted (hardware IRQs).
    /// System-internal counters (LOC, NMI, RES, CAL, TLB, etc.) are excluded
    /// because they are inherent to the scheduler and do not represent
    /// external device load that would interfere with DSP processing.
    fn parse_interrupts_per_cpu(num_cpus: usize) -> Vec<u64> {
        use std::fs;

        let mut totals = vec![0u64; num_cpus];

        let content = match fs::read_to_string("/proc/interrupts") {
            Ok(c) => c,
            Err(_) => return totals,
        };

        for line in content.lines().skip(1) {
            // Skip lines that don't start with a numeric IRQ number
            let trimmed = line.trim_start();
            let irq_end = trimmed.find(':').unwrap_or(0);
            if irq_end == 0 {
                continue;
            }
            // Only count hardware IRQ lines (numeric identifiers)
            if !trimmed[..irq_end]
                .trim()
                .bytes()
                .all(|b| b.is_ascii_digit())
            {
                continue;
            }

            // Parse per-CPU counts after the colon
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
                    // Hit the device description text — stop parsing this line
                    break;
                }
            }
        }

        totals
    }

    /// Attempts to pin the DSP thread to the optimal physical core
    /// and apply SCHED_FIFO for real-time scheduling.
    /// Called only once on the first invocation of `process()`, before the data flow begins.
    /// NOTE: On modern Linux, the kernel refuses to grant SCHED_FIFO — even on demand. In practice, this request is silently ignored.
    /// However, we are not left unprotected. PipeWire, via RTkit, automatically grants a high priority within CFS.
    ///
    /// # Documented I/O Exception
    /// This function uses `println!`/`eprintln!` for one-time diagnostic logging
    /// and reads `/sys` + `/proc` for CPU topology detection.
    /// Although this involves I/O syscalls, it runs only once and before any audio data flows,
    /// so it does not compromise steady-state processing latency.
    fn configure_realtime_thread(&self) {
        #[cfg(target_os = "linux")]
        {
            // Select the optimal CPU core dynamically (fallback to CPU 0 if detection fails)
            let target_cpu = Self::select_optimal_cpu().unwrap_or(0);

            unsafe {
                let thread_id = libc::pthread_self();

                // 1. Core Affinity: Pin the thread to the optimal core to protect L1/L2 Cache
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
                        "Warning: Failed to set CPU affinity to core {} (error {}).",
                        target_cpu, ret_aff
                    );
                }

                // 2. Real-Time: Apply SCHED_FIFO for deterministic preemption
                let mut param: libc::sched_param = std::mem::zeroed();
                param.sched_priority = 90; // High priority (requires rtprio >= 90 in limits.conf)

                let _ret_sched = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);

                // Verify the actual thread state after the configuration attempts
                let mut actual_policy = 0;
                let mut actual_param: libc::sched_param = std::mem::zeroed();
                let ret_getsched =
                    libc::pthread_getschedparam(thread_id, &mut actual_policy, &mut actual_param);

                let actual_cpu = libc::sched_getcpu();

                if ret_getsched == 0 {
                    let reset_on_fork_flag = 0x40000000;
                    let has_reset_on_fork = (actual_policy & reset_on_fork_flag) != 0;
                    let base_policy = actual_policy & !reset_on_fork_flag;

                    let mut policy_str = match base_policy {
                        libc::SCHED_FIFO => "SCHED_FIFO".to_string(),
                        libc::SCHED_RR => "SCHED_RR".to_string(),
                        libc::SCHED_OTHER => "SCHED_OTHER".to_string(),
                        libc::SCHED_BATCH => "SCHED_BATCH".to_string(),
                        libc::SCHED_IDLE => "SCHED_IDLE".to_string(),
                        other => format!("UNKNOWN: {}", other),
                    };

                    if has_reset_on_fork {
                        policy_str.push_str(" | SCHED_RESET_ON_FORK");
                    }
                    println!(
                        "[AudioRip] 🔍 Audio/DSP Thread: CPU Core = {}, Policy = {}, Priority = {}",
                        actual_cpu, policy_str, actual_param.sched_priority
                    );
                } else {
                    eprintln!(
                        "[AudioRip] ⚠️ Failed to verify thread parameters (error {}).",
                        ret_getsched
                    );
                }
            }
        }
    }
}
