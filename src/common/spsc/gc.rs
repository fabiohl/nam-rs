// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use rtrb::Consumer;
use std::sync::atomic::{AtomicU64, Ordering};

#[allow(clippy::large_enum_variant)]
/// Represents an item that should be safely disposed outside the audio thread.
/// Dropping these items may involve heavy memory deallocations.
pub enum GcItem {
    /// A model instance (LSTM or WaveNet).
    Model(Box<crate::models::StaticModel>),
    /// A resampler (boxed to ensure RT-safety on drop).
    Resampler(Box<crate::dsp::resampler::NamResampler>),
    /// A cab-sim impulse response (boxed to ensure RT-safety on drop).
    #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
    CabSimIr(Box<crate::dsp::cabsim::loader::CabSimIr>),
    /// A cab-sim convolution engine (boxed to ensure RT-safety on drop).
    #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
    CabConvEngine(Box<crate::dsp::cabsim::conv::ConvEngine>),
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
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            GcItem::CabSimIr(_) => 3,
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            GcItem::CabConvEngine(_) => 4,
            #[cfg(test)]
            GcItem::Test(_) => 255,
        }
    }

    /// Reconstructs a GcItem from a raw pointer and a type ID.
    ///
    /// Returns `None` if the type ID is unknown. In that case, the caller
    /// must intentionally leak the pointer to avoid UB from `Box::from_raw`
    /// with a mismatched type.
    ///
    /// # Safety
    /// The pointer must have been generated via `Box::into_raw` of an object
    /// of the corresponding type, validated by the caller against type_id.
    unsafe fn from_raw_parts(ptr: *mut std::ffi::c_void, type_id: u8) -> Option<Self> {
        match type_id {
            1 => Some(GcItem::Model(unsafe {
                Box::from_raw(ptr as *mut crate::models::StaticModel)
            })),
            2 => Some(GcItem::Resampler(unsafe {
                Box::from_raw(ptr as *mut crate::dsp::resampler::NamResampler)
            })),
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            3 => Some(GcItem::CabSimIr(unsafe {
                Box::from_raw(ptr as *mut crate::dsp::cabsim::loader::CabSimIr)
            })),
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            4 => Some(GcItem::CabConvEngine(unsafe {
                Box::from_raw(ptr as *mut crate::dsp::cabsim::conv::ConvEngine)
            })),
            #[cfg(test)]
            255 => Some(GcItem::Test(unsafe {
                Box::from_raw(ptr as *mut std::sync::Arc<std::sync::atomic::AtomicU32>)
            })),
            _ => None,
        }
    }

    /// Converts this GcItem into a packed 64-bit representation for the
    /// overflow buffer.
    ///
    /// Bits 0-55: user-space pointer (≤ 56 bits on all x86-64 Linux configs).
    /// Bits 56-63: type ID.
    pub(crate) fn into_packed(self) -> u64 {
        let type_id = self.type_id();
        let ptr = match self {
            GcItem::Model(b) => Box::into_raw(b) as *mut std::ffi::c_void,
            GcItem::Resampler(b) => Box::into_raw(b) as *mut std::ffi::c_void,
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            GcItem::CabSimIr(b) => Box::into_raw(b) as *mut std::ffi::c_void,
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            GcItem::CabConvEngine(b) => Box::into_raw(b) as *mut std::ffi::c_void,
            #[cfg(test)]
            GcItem::Test(b) => Box::into_raw(b) as *mut std::ffi::c_void,
        };
        ((type_id as u64) << 56) | (ptr as u64 & 0x00FF_FFFF_FFFF_FFFF)
    }

    /// Reconstructs a GcItem from a packed 64-bit value.
    ///
    /// Returns `None` if the packed value is zero (empty slot) or the type
    /// ID is unknown. On unknown type ID, the caller must leak the pointer.
    ///
    /// # Safety
    /// The pointer embedded in `packed` must be valid for the type encoded in bits 56-63.
    unsafe fn from_packed(packed: u64) -> Option<Self> {
        if packed == 0 {
            return None;
        }
        let type_id = ((packed >> 56) & 0xFF) as u8;
        let ptr = (packed & 0x00FF_FFFF_FFFF_FFFF) as *mut std::ffi::c_void;
        if type_id == 0 {
            return None;
        }
        unsafe { Self::from_raw_parts(ptr, type_id) }
    }
}

/// Circular "final parking" buffer for GC items.
///
/// Each slot is a single `AtomicU64` packing type_id (bits 56-63) and
/// pointer (bits 0-55) into one atomic word, eliminating the torn-read
/// window that existed when type and pointer were swapped independently.
///
/// Used when the main SPSC channel and the thread's parking lot are both full.
/// Ensures no object is dropped on the audio thread at the cost of a controlled leak
/// or overwrite in extreme stress scenarios.
pub struct GcOverflowBuffer {
    pub(crate) slots: Box<[AtomicU64]>,
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
        for _ in 0..capacity {
            slots.push(AtomicU64::new(0));
        }
        Self {
            slots: slots.into_boxed_slice(),
            write_idx: AtomicU64::new(0),
        }
    }

    /// Attempts to park an item in the overflow buffer without RT allocations.
    ///
    /// If the buffer is full, the oldest item is overwritten (leak).
    /// Returns `true` if an overwrite (controlled leak) occurred.
    pub fn push(&self, item: GcItem) -> bool {
        let packed = item.into_packed();

        let len = self.slots.len() as u64;
        let idx = (self.write_idx.fetch_add(1, Ordering::Relaxed) % len) as usize;

        let old = self.slots[idx].swap(packed, Ordering::AcqRel);

        old != 0
    }

    /// Drains all accumulated items from the overflow buffer.
    /// Should be called periodically by the main thread (Cold Path).
    ///
    /// On corrupted slots (unknown type_id), the pointer is intentionally leaked
    /// and `RT_STATUS_GC_CORRUPTED` is set via the `rt_status` parameter.
    /// Returns ownership of the drained items so the caller can drop them.
    pub fn drain(&self, rt_status: &super::RtStatusFlags) -> Vec<GcItem> {
        let mut items = Vec::with_capacity(self.slots.len());
        for i in 0..self.slots.len() {
            let packed = self.slots[i].swap(0, Ordering::AcqRel);
            if packed == 0 {
                continue;
            }
            unsafe {
                match GcItem::from_packed(packed) {
                    Some(item) => items.push(item),
                    None => {
                        rt_status.set_flag(super::RT_STATUS_GC_CORRUPTED);
                    }
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

/// RT-safe GC cascade: tries the SPSC channel, then a 16-slot parking lot,
/// then the overflow buffer as a last resort.
///
/// # Parameters
/// - `item`: The `GcItem` to dispose of outside the RT thread.
/// - `gc_producer`: SPSC GC channel.
/// - `parking_lot`: 16-slot fallback array shared across all drainers.
/// - `gc_overflow`: Overflow ring buffer.
/// - `rt_status`: Status flags (sets `RT_STATUS_GC_OVERFLOW` on overflow).
#[inline(always)]
pub fn gc_cascade(
    mut item: Option<GcItem>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow: &GcOverflowBuffer,
    rt_status: &super::RtStatusFlags,
) {
    if let Some(i) = item.take() {
        if let Err(rtrb::PushError::Full(returned)) = gc_producer.push(i) {
            item = Some(returned);
        } else {
            return;
        }
    }

    if let Some(i) = item.take() {
        let mut i_opt = Some(i);
        for slot in parking_lot.iter_mut() {
            if slot.is_none() {
                *slot = i_opt.take();
                return;
            }
        }
        item = i_opt;
    }

    if let Some(i) = item.take() {
        rt_status.set_flag(super::RT_STATUS_GC_OVERFLOW);
        gc_overflow.push(i);
    }
}

/// Aggressively drains the Garbage Collection channels to free memory.
///
/// This function should be called periodically by the main thread (CLI/UI)
/// or by the host event loop (PipeWire, CLAP). It executes `drop()`
/// on obsolete objects (models, resamplers) outside the RT thread.
///
/// Returns the total number of GC items dropped during this call.
pub fn drain_gc_channels(
    gc_consumer: &mut Consumer<GcItem>,
    gc_overflow: &GcOverflowBuffer,
    rt_status: &super::RtStatusFlags,
) -> usize {
    let mut count = 0;
    // 1. Drain the main SPSC channel (Drop-Delegation)
    while let Ok(item) = gc_consumer.pop() {
        drop(item);
        count += 1;
    }

    // 2. Drain the overflow buffer (overwrite ring buffer)
    for item in gc_overflow.drain(rt_status) {
        drop(item);
        count += 1;
    }
    count
}
