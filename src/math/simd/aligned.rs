// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! [TA5] Utilitário para alocação de memória alinhada a 64 bytes.
//!
//! Garante que buffers dinâmicos (pesos, acumuladores) respeitem os limites
//! de cache line e requisitos de AVX2/AVX-512, evitando penalidades de
//! unaligned load/store.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Um buffer alinhado a 64 bytes (Cache Line / AVX-512).
#[derive(Debug)]
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
}

impl<T> AlignedVec<T> {
    /// O alinhamento padrão garantido (64 bytes).
    pub const ALIGN: usize = 64;

    /// Cria um novo AlignedVec preenchido com o valor padrão.
    pub fn new(len: usize, default: T) -> Self
    where
        T: Copy,
    {
        let mut vec = Self::with_capacity(len);
        unsafe {
            for i in 0..len {
                vec.ptr.as_ptr().add(i).write(default);
            }
        }
        vec.len = len;
        vec
    }

    /// Cria um AlignedVec com a capacidade especificada, mas comprimento zero.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
            };
        }

        let layout = Layout::from_size_align(capacity * std::mem::size_of::<T>(), Self::ALIGN)
            .expect("Falha ao criar layout para AlignedVec");

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        Self {
            ptr: NonNull::new(ptr as *mut T).unwrap(),
            len: 0,
        }
    }

    /// Retorna o número de elementos no buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Retorna true se o buffer estiver vazio.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout =
                Layout::from_size_align(self.len * std::mem::size_of::<T>(), Self::ALIGN).unwrap();
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity(self.len);
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

// Implementação segura para empty vec
impl<T: Copy> AlignedVec<T> {
    /// Converte um Vec comum em AlignedVec (alinhamento 64).
    pub fn from_vec(v: Vec<T>) -> Self {
        if v.is_empty() {
            return Self::with_capacity(0);
        }
        let mut aligned = Self::with_capacity(v.len());
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), aligned.ptr.as_ptr(), v.len());
        }
        aligned.len = v.len();
        aligned
    }
}

unsafe impl<T: Send> Send for AlignedVec<T> {}
unsafe impl<T: Sync> Sync for AlignedVec<T> {}
