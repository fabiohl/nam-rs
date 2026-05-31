// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Auditoria de alocações no heap em tempo real (RT-Safety).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};

/// Flag controlando se a verificação de alocação de heap está ativada.
pub static AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);
/// Contador global de alocações realizadas na thread vigiada.
pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// ID da thread (tid) que estamos vigiando.
pub static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0);
/// ID da thread que está autorizada a realizar a auditoria (usada para isolar testes paralelos).
pub static AUDIT_THREAD: AtomicI32 = AtomicI32::new(0);

/// O "Vigia de Memória": intercepta todos os pedidos de memória do programa.
pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
        if tid == TRACKING_THREAD.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(all(feature = "clap-plugin", feature = "heap-audit", not(test)))]
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// O "Interruptor": liga o vigia quando criado e o desliga quando destruído.
pub struct TrackingGuard;

impl TrackingGuard {
    /// Começa a vigiar a thread atual.
    pub fn new() -> Self {
        let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
        TRACKING_THREAD.store(tid, Ordering::Relaxed);
        ALLOC_COUNT.store(0, Ordering::Relaxed);
        Self
    }
}

impl Default for TrackingGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TrackingGuard {
    fn drop(&mut self) {
        TRACKING_THREAD.store(0, Ordering::Relaxed);
    }
}
