// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared data structures and flags for lock-free thread coordination.
//!
//! This module defines all the "invisible wiring" that connects the CLI interface (the
//! musician's remote control) with the DSP audio engine (the neural amplifier running in real-time).
//!
//! ## Main components
//!
//! - **`SHUTDOWN`**: global flag signaling all threads that the program should terminate.
//! - **`ParamPayload`**: parameter "packets" (input/output gain, model swap)
//!   that the CLI sends to the DSP without locking any thread (lock-free via SPSC ring buffer).
//! - **`RtStatusFlags`**: atomic flags for silent RT→Main communication (no I/O in callback).
//!   Allows the DSP engine to report its status without ever calling `println!`.
//! - **`NamResampler` SPSC channel**: the main thread builds the resampler (with memory
//!   allocations) outside real-time and sends it to the DSP callback via a lock-free channel.

use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Arc;
use std::sync::atomic::{
    AtomicBool, AtomicI32, AtomicPtr, AtomicU8, AtomicU32, AtomicU64, Ordering,
};

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
/// Flag indicating that an A2 placeholder model is active (silent bypass).
pub const RT_STATUS_A2_PLACEHOLDER: u64 = 1 << 9;
/// Flag indicating that the RT callback should pause DSP processing until
/// the resampler is replaced (during hot-plug or sample rate change).
pub const RT_STATUS_RESAMP_SWAP_PENDING: u64 = 1 << 10;

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
/// | 9 | `A2_PLACEHOLDER` | A2 placeholder model active (silent bypass) |
/// | 10 | `RESAMP_SWAP_PENDING` | RT callback paused awaiting resampler swap |
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

    /// Atomic bitmask containing binary states (needs_rebuild, clipped, silent, etc).
    /// Reduces Cache Bouncing by condensing multiple states into a single cache line.
    pub status_bits: AtomicU64,
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
            status_bits: AtomicU64::new(0),
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
        (self.status_bits.fetch_and(!flag, Ordering::Relaxed) & flag) != 0
    }
}

impl Default for RtStatusFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// SPSC payload sent from the Host (CLI/UI) to the DSP Thread.
/// Aligned to 128 bytes to mitigate False Sharing.
#[repr(align(128))]
pub enum ParamPayload {
    /// Injects the input gain as a linear multiplier.
    InputGain(f32),
    /// Injects the output gain as a linear multiplier.
    OutputGain(f32),
    /// Loads the decoded mathematical topology, also informing the thresholds
    /// expected by the model creator (resolved from input_level_dbu and loudness tags).
    /// The pointer ensures zero-allocation (no-heap) and deterministic initialization.
    LoadModel {
        /// The encapsulated model for neural inference (Left Channel)
        model_l: Option<Box<crate::models::DynamicModel>>,
        /// The encapsulated model for neural inference (Right Channel)
        model_r: Option<Box<crate::models::DynamicModel>>,
        /// Expected input gain adjustment as a linear multiplier.
        input_mult_adj: f32,
        /// Expected output gain adjustment as a linear multiplier.
        output_mult_adj: f32,
        /// Sample rate required by the model (usually 48000).
        sample_rate: u32,
    },
    /// Injects the Silence/Mono Gate settings.
    GateConfig(crate::dsp::gate::GateParams),
}

#[allow(clippy::large_enum_variant)]
/// Represents an item that should be safely disposed outside the audio thread.
/// Dropping these items may involve heavy memory deallocations.
pub enum GcItem {
    /// A dynamic model (LSTM or WaveNet).
    Model(Box<crate::models::DynamicModel>),
    /// A resampler (boxed to ensure RT-safety on drop).
    Resampler(Box<crate::dsp::resampler::NamResampler>),
    /// Test variant for integrity and stress validation.
    #[cfg(test)]
    Test(Box<std::sync::Arc<std::sync::atomic::AtomicU32>>),
}

impl GcItem {
    /// Returns the type ID for the overflow buffer.
    fn type_id(&self) -> u8 {
        match self {
            GcItem::Model(_) => 1,
            GcItem::Resampler(_) => 2,
            #[cfg(test)]
            GcItem::Test(_) => 255,
        }
    }

    /// Reconstructs a GcItem from a raw pointer and a type ID.
    ///
    /// # Safety
    /// The pointer must have been generated via `Box::into_raw` of an object of the corresponding type.
    unsafe fn from_raw_parts(ptr: *mut std::ffi::c_void, type_id: u8) -> Self {
        match type_id {
            1 => GcItem::Model(unsafe { Box::from_raw(ptr as *mut crate::models::DynamicModel) }),
            2 => GcItem::Resampler(unsafe {
                Box::from_raw(ptr as *mut crate::dsp::resampler::NamResampler)
            }),
            #[cfg(test)]
            255 => GcItem::Test(unsafe {
                Box::from_raw(ptr as *mut std::sync::Arc<std::sync::atomic::AtomicU32>)
            }),
            _ => panic!("GcItem: tipo desconhecido {}", type_id),
        }
    }
}

/// Circular "final parking" buffer for GC items.
/// Used when the main SPSC channel and the thread's parking lot are both full.
/// Ensures no object is dropped on the audio thread at the cost of a controlled leak
/// or overwrite in extreme stress scenarios.
pub struct GcOverflowBuffer {
    slots: Box<[AtomicPtr<std::ffi::c_void>]>,
    types: Box<[AtomicU8]>,
    write_idx: AtomicU64,
}

impl GcOverflowBuffer {
    #[cold]
    /// Creates a new overflow buffer with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "GcOverflowBuffer: capacity must be greater than 0 to avoid division by zero panic."
        );
        let mut slots = Vec::with_capacity(capacity);
        let mut types = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(AtomicPtr::new(std::ptr::null_mut()));
            types.push(AtomicU8::new(0));
        }
        Self {
            slots: slots.into_boxed_slice(),
            types: types.into_boxed_slice(),
            write_idx: AtomicU64::new(0),
        }
    }

    /// Attempts to park an item in the overflow buffer without RT allocations.
    ///
    /// If the buffer is full, the oldest item is overwritten (leak).
    pub fn push(&self, item: GcItem) -> bool {
        let type_id = item.type_id();
        let ptr = match item {
            GcItem::Model(b) => Box::into_raw(b) as *mut std::ffi::c_void,
            GcItem::Resampler(b) => Box::into_raw(b) as *mut std::ffi::c_void,
            #[cfg(test)]
            GcItem::Test(b) => Box::into_raw(b) as *mut std::ffi::c_void,
        };

        let len = self.slots.len() as u64;
        let idx = (self.write_idx.fetch_add(1, Ordering::Relaxed) % len) as usize;

        // Swap the type first, then the pointer.
        let old_type = self.types[idx].swap(type_id, Ordering::Release);
        let old_ptr = self.slots[idx].swap(ptr, Ordering::Acquire);

        if !old_ptr.is_null() && old_type != 0 {
            // OVERWRITE: Intentional leak to avoid Drop in RT.
            // The user will be notified via the status flag.
            true // An overwrite occurred
        } else {
            false
        }
    }

    /// Drains all accumulated items from the overflow buffer.
    /// Should be called periodically by the main thread (Cold Path).
    pub fn drain(&self) -> Vec<GcItem> {
        let mut items = Vec::with_capacity(self.slots.len());
        for i in 0..self.slots.len() {
            let ptr = self.slots[i].swap(std::ptr::null_mut(), Ordering::Release);
            let type_id = self.types[i].swap(0, Ordering::Acquire);
            if !ptr.is_null() && type_id != 0 {
                unsafe {
                    items.push(GcItem::from_raw_parts(ptr, type_id));
                }
            }
        }
        items
    }
}

impl Default for GcOverflowBuffer {
    fn default() -> Self {
        Self::new(64)
    }
}

/// SPSC initialization result: parameter channels, model GC,
/// RT-safe resampler channel, and atomic status flags.
pub struct SpscChannels {
    /// CLI→DSP parameter producer.
    pub param_producer: Producer<ParamPayload>,
    /// CLI→DSP parameter consumer (moved to the RT callback).
    pub param_consumer: Consumer<ParamPayload>,
    /// GC producer: DSP thread sends obsolete items for drop outside RT.
    pub gc_producer: Producer<GcItem>,
    /// GC consumer: background thread executes `drop()`.
    pub gc_consumer: Consumer<GcItem>,
    /// Fallback buffer for GC overflow (overwrite).
    pub gc_overflow: Arc<GcOverflowBuffer>,
    /// Resampler producer: main thread builds and sends to the RT callback.
    pub resampler_producer: Producer<Box<crate::dsp::resampler::NamResampler>>,
    /// Resampler consumer: RT callback drains to replace the active resampler.
    pub resampler_consumer: Consumer<Box<crate::dsp::resampler::NamResampler>>,
    /// Atomic status flags shared between RT and Main (zero I/O in callback).
    pub rt_status: Arc<RtStatusFlags>,
}

/// Creates and returns the complete lock-free SPSC mesh for the pipeline.
///
/// Includes channels for:
/// - CLI→DSP parameters (`ParamPayload`)
/// - Obsolete model GC (Drop-Delegation)
/// - Pre-built resamplers (Main→RT, zero allocation in callback)
/// - RT→Main atomic status flags
///
/// `capacity` should preferably be a power of 2.
pub fn setup_spsc(capacity: usize) -> SpscChannels {
    let (param_prod, param_cons) = RingBuffer::new(capacity);
    let (gc_prod, gc_cons) = RingBuffer::new(capacity * 4); // Quadrupled capacity for safe garbage collection
    // Resampler channel: small capacity (only 1 in transit at a time, typically)
    let (rs_prod, rs_cons) = RingBuffer::new(4);
    let rt_status = Arc::new(RtStatusFlags::new());
    // The overflow buffer should be large enough to accommodate model swap spikes.
    // We use 64 as a base, or the requested capacity if higher.
    let gc_overflow = Arc::new(GcOverflowBuffer::new(capacity.max(64)));

    SpscChannels {
        param_producer: param_prod,
        param_consumer: param_cons,
        gc_producer: gc_prod,
        gc_consumer: gc_cons,
        gc_overflow,
        resampler_producer: rs_prod,
        resampler_consumer: rs_cons,
        rt_status,
    }
}

/// Aggressively drains the Garbage Collection channels to free memory.
///
/// This function should be called periodically by the main thread (CLI/UI)
/// or by the host event loop (PipeWire, CLAP). It executes `drop()`
/// on obsolete objects (models, resamplers) outside the RT thread.
pub fn drain_gc_channels(gc_consumer: &mut Consumer<GcItem>, gc_overflow: &GcOverflowBuffer) {
    // 1. Drain the main SPSC channel (Drop-Delegation)
    while let Ok(item) = gc_consumer.pop() {
        drop(item);
    }

    // 2. Drain the overflow buffer (overwrite ring buffer)
    for item in gc_overflow.drain() {
        drop(item);
    }
}

#[cfg(test)]
#[path = "spsc_test.rs"]
mod spsc_test;
