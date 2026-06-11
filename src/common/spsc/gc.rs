// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use rtrb::Consumer;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicU64, Ordering};

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
            1 => GcItem::Model(unsafe { Box::from_raw(ptr as *mut crate::models::StaticModel) }),
            2 => GcItem::Resampler(unsafe {
                Box::from_raw(ptr as *mut crate::dsp::resampler::NamResampler)
            }),
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            3 => GcItem::CabSimIr(unsafe {
                Box::from_raw(ptr as *mut crate::dsp::cabsim::loader::CabSimIr)
            }),
            #[cfg(test)]
            255 => GcItem::Test(unsafe {
                Box::from_raw(ptr as *mut std::sync::Arc<std::sync::atomic::AtomicU32>)
            }),
            _ => panic!("GcItem: unknown type {}", type_id),
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
            #[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
            GcItem::CabSimIr(b) => Box::into_raw(b) as *mut std::ffi::c_void,
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
) -> usize {
    let mut count = 0;
    // 1. Drain the main SPSC channel (Drop-Delegation)
    while let Ok(item) = gc_consumer.pop() {
        drop(item);
        count += 1;
    }

    // 2. Drain the overflow buffer (overwrite ring buffer)
    for item in gc_overflow.drain() {
        drop(item);
        count += 1;
    }
    count
}
