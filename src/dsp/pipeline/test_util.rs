// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
pub(crate) mod infra {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

    /// Contador global de quantas vezes a memória foi pedida.
    pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    /// Guarda qual "linha de processamento" (thread) estamos vigiando no momento.
    pub static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0);

    /// O "Vigia de Memória": intercepta todos os pedidos de memória do programa.
    pub struct CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // Descobre quem está pedindo memória agora.
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            // Se for a linha de processamento que estamos vigiando, aumenta o contador.
            if tid == TRACKING_THREAD.load(Ordering::Relaxed) {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            // Realiza o pedido de memória normalmente através do sistema.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // Devolve a memória ao sistema.
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    /// O "Interruptor": liga o vigia quando criado e o desliga quando destruído.
    pub struct TrackingGuard;
    impl TrackingGuard {
        /// Começa a vigiar a linha de processamento atual.
        pub fn new() -> Self {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            TRACKING_THREAD.store(tid, Ordering::Relaxed);
            ALLOC_COUNT.store(0, Ordering::Relaxed);
            Self
        }
    }
    impl Drop for TrackingGuard {
        /// Para de vigiar quando o teste termina ou sai do escopo.
        fn drop(&mut self) {
            TRACKING_THREAD.store(0, Ordering::Relaxed);
        }
    }
}
