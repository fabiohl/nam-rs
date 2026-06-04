// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Utility for 64-byte aligned memory allocation.
//!
//! Ensures that dynamic buffers (weights, accumulators) respect cache line
//! boundaries and AVX2/AVX-512 requirements, avoiding unaligned load/store
//! penalties.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A 64-byte aligned buffer (Cache Line / AVX-512).
#[derive(Debug)]
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T> AlignedVec<T> {
    /// The default guaranteed alignment (64 bytes).
    pub const ALIGN: usize = 64;

    /// Creates a new aligned buffer and fills it with an initial value.
    ///
    /// Think of a shelf where each compartment has exactly the size the processor
    /// prefers to read (64 bytes). This function reserves the space and places a "default value"
    /// in each spot, leaving everything ready for immediate use.
    pub fn new(len: usize, default: T) -> Self
    where
        T: Copy,
    {
        let mut vec = Self::with_capacity(len);
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            for i in 0..len {
                vec.ptr.as_ptr().add(i).write(default);
            }
        }
        vec.len = len;
        vec
    }

    /// Reserves memory space with the required alignment, but without filling the data yet.
    ///
    /// It's like reserving an exclusive parking lot: the space is there, guaranteed and
    /// organized in 64-byte blocks, but the "slots" are still empty.
    /// This special alignment allows the processor to read data at maximum
    /// speed, without needing to "adjust" the memory position.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }

        let layout = Layout::from_size_align(capacity * std::mem::size_of::<T>(), Self::ALIGN)
            .expect("Failed to create layout for AlignedVec");

        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        Self {
            ptr: NonNull::new(ptr as *mut T).unwrap(),
            len: 0,
        }
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resizes the buffer to a new length, filling new elements with `default`.
    /// If `new_len <= self.len`, this is a no-op (never shrinks).
    pub fn resize(&mut self, new_len: usize, default: T)
    where
        T: Copy,
    {
        if new_len <= self.len {
            return;
        }
        let mut new_vec = Self::with_capacity(new_len);
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            let old_ptr = self.ptr.as_ptr();
            for i in 0..self.len {
                new_vec.ptr.as_ptr().add(i).write(old_ptr.add(i).read());
            }
            for i in self.len..new_len {
                new_vec.ptr.as_ptr().add(i).write(default);
            }
        }
        new_vec.len = new_len;
        let old = std::mem::replace(self, new_vec);
        drop(old);
    }
}

/// Allows accessing the buffer data as if it were a regular Rust slice.
///
/// This simplifies usage, as you can use functions that expect a normal slice
/// without needing complicated conversions.
impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

/// Allows accessing and modifying buffer data as a regular list.
///
/// Ensures freedom to read and write content in a simple and direct way.
impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

/// Ensures memory is returned to the system when the object is destroyed.
///
/// It's the "cleaner" that frees the reserved space as soon as you finish using
/// the buffer, preventing memory waste (so-called "memory leaks").
impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout =
                Layout::from_size_align(self.len * std::mem::size_of::<T>(), Self::ALIGN).unwrap();
            // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

/// Creates an identical copy of the buffer and its entire contents.
///
/// Reserves new memory space with the same alignment and copies each item
/// one by one, ensuring the new "shelf" is a faithful replica of the original.
impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity(self.len);
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            for i in 0..self.len {
                let val = (*self.ptr.as_ptr().add(i)).clone();
                new_vec.ptr.as_ptr().add(i).write(val);
            }
        }
        new_vec.len = self.len;
        new_vec
    }
}

/// Allows transforming a regular vector (`Vec`) into an aligned buffer.
///
/// It's like transferring items from a regular plastic bag into a case with
/// custom-fit dividers (64-byte alignment), preparing the data for
/// high-speed processing.
impl<T: Copy> From<Vec<T>> for AlignedVec<T> {
    fn from(v: Vec<T>) -> Self {
        let mut aligned = Self::new(v.len(), v[0]); // v[0] works if len > 0
        if v.is_empty() {
            return Self::with_capacity(0);
        }
        aligned.copy_from_slice(&v);
        aligned
    }
}

// Safe implementation for empty vec
impl<T: Copy> AlignedVec<T> {
    /// An optimized version of the conversion from regular vector to aligned buffer.
    ///
    /// Uses a direct memory copy to ensure the transfer is as fast as possible,
    /// maintaining the 64-byte alignment.
    pub fn from_vec(v: Vec<T>) -> Self {
        if v.is_empty() {
            return Self::with_capacity(0);
        }
        let mut aligned = Self::with_capacity(v.len());
        // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), aligned.ptr.as_ptr(), v.len());
        }
        aligned.len = v.len();
        aligned
    }
}

/// Tells Rust that it is safe to send and share this buffer between different threads.
///
/// This is essential so that audio processing can be distributed across
/// multiple processor cores with full safety.
// SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
unsafe impl<T: Send> Send for AlignedVec<T> {}
// SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
unsafe impl<T: Sync> Sync for AlignedVec<T> {}
