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
    /// Attempts to pin the DSP thread to a physical high-priority core
    /// and apply SCHED_FIFO for real-time scheduling.
    /// Called only once on the first invocation of `process()`, before the data flow begins.
    /// NOTE: On modern Linux, the kernel refuses to grant SCHED_FIFO — even on demand. In practice, this request is silently ignored.
    /// However, we are not left unprotected. PipeWire, via RTkit, automatically grants a high priority within CFS.
    ///
    /// # Documented I/O Exception
    /// This function uses `println!`/`eprintln!` for one-time diagnostic logging.
    /// Although this involves locks on stdout/stderr and potentially `write()` syscalls,
    /// it runs only once and before any audio data flows,
    /// so it does not compromise steady-state processing latency.
    fn configure_realtime_thread(&self) {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let thread_id = libc::pthread_self();

                // 1. Core Affinity: Pin the thread to a physical core to protect L1/L2 Cache
                let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut cpuset);
                libc::CPU_SET(1, &mut cpuset); // Assuming Core 1 as performance core (refine as needed — this is not a universal rule!)

                let ret_aff = libc::pthread_setaffinity_np(
                    thread_id,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &cpuset,
                );

                if ret_aff != 0 {
                    eprintln!("Warning: Failed to set CPU affinity (error {}).", ret_aff);
                }

                // 2. Real-Time: Apply SCHED_FIFO for deterministic preemption
                let mut param: libc::sched_param = std::mem::zeroed();
                param.sched_priority = 90; // High priority (requires rtprio >= 90 in limits.conf)

                let ret_sched = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);

                if ret_sched == 0 {
                    println!(
                        "[AudioRip] ⚡ DSP Thread pinned to Core 1 running under SCHED_FIFO (Priority 90)."
                    );
                }

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
                        "[AudioRip] 🔍 DSP Verification: Current core = {}, Policy = {}, Priority = {}",
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
