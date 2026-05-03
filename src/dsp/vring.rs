// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de Buffer Circular Virtual via Mapeamento de Memória Espelhada.
//!
//! Este módulo fornece a estrutura `VirtualRingBuffer<T>`, que utiliza a técnica
//! de mapear a mesma memória física duas vezes consecutivas no espaço de endereçamento
//! virtual. Isso elimina a necessidade de operações de rewind (copy_within) em buffers
//! circulares, permitindo acessos lineares contíguos mesmo através da fronteira do buffer.
use libc::{
    _SC_PAGESIZE, MAP_ANONYMOUS, MAP_FAILED, MAP_FIXED, MAP_PRIVATE, MAP_SHARED, PROT_NONE,
    PROT_READ, PROT_WRITE, c_void, ftruncate, mmap, munmap, sysconf,
};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr;

/// Um Buffer Circular Virtual que usa mapeamento de memória espelhado.
///
/// Esta estrutura mapeia o mesmo conteúdo físico duas vezes consecutivas no espaço
/// de endereçamento virtual. Isso permite que acessos que "atravessariam" o fim do
/// buffer sejam feitos de forma linear e contígua, eliminando a necessidade de
/// operações de "rewind" ou "copy_within" no hot-path do DSP.
pub struct VirtualRingBuffer<T> {
    ptr: *mut T,
    size_elements: usize,
    _marker: PhantomData<T>,
}

impl<T> VirtualRingBuffer<T> {
    /// Cria um novo buffer circular virtual.
    ///
    /// O tamanho `requested_size` (em elementos) será arredondado para cima para
    /// o próximo múltiplo do tamanho da página do sistema.
    pub fn new(requested_size: usize) -> Self {
        let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
        let element_size = std::mem::size_of::<T>();

        // Garantir que o tamanho do elemento não seja zero (ex: ZST)
        assert!(
            element_size > 0,
            "VirtualRingBuffer não suporta Zero Sized Types"
        );

        let requested_bytes = requested_size * element_size;

        // Arredonda para múltiplo da página
        let size_bytes = (requested_bytes + page_size - 1) & !(page_size - 1);
        let size_elements = size_bytes / element_size;

        // 1. Criar backing store (memfd no Linux)
        // MFD_CLOEXEC evita que o FD seja herdado por processos filhos.
        let fd = unsafe { libc::memfd_create(c"vring".as_ptr(), libc::MFD_CLOEXEC) };
        if fd == -1 {
            panic!(
                "Falha ao criar memfd para VirtualRingBuffer: {}",
                std::io::Error::last_os_error()
            );
        }

        // 2. Definir o tamanho do arquivo
        if unsafe { ftruncate(fd, size_bytes as libc::off_t) } == -1 {
            let err = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            panic!("Falha ao truncar memfd para VirtualRingBuffer: {}", err);
        }

        // 3. Reservar espaço virtual contíguo (2x tamanho)
        let total_size = size_bytes * 2;
        let base_ptr = unsafe {
            mmap(
                ptr::null_mut(),
                total_size,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if base_ptr == MAP_FAILED {
            let err = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            panic!(
                "Falha ao reservar memória virtual para VirtualRingBuffer: {}",
                err
            );
        }

        // 4. Mapear a primeira metade
        let ptr1 = unsafe {
            mmap(
                base_ptr,
                size_bytes,
                PROT_READ | PROT_WRITE,
                MAP_FIXED | MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr1 != base_ptr {
            let err = std::io::Error::last_os_error();
            unsafe {
                munmap(base_ptr, total_size);
                libc::close(fd);
            }
            panic!(
                "Falha ao mapear primeira metade do VirtualRingBuffer: {}",
                err
            );
        }

        // 5. Mapear a segunda metade (espelho)
        let ptr2 = unsafe {
            mmap(
                (base_ptr as *mut u8).add(size_bytes) as *mut c_void,
                size_bytes,
                PROT_READ | PROT_WRITE,
                MAP_FIXED | MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr2 != unsafe { (base_ptr as *mut u8).add(size_bytes) as *mut c_void } {
            let err = std::io::Error::last_os_error();
            unsafe {
                munmap(base_ptr, total_size);
                libc::close(fd);
            }
            panic!(
                "Falha ao mapear segunda metade do VirtualRingBuffer: {}",
                err
            );
        }

        // O FD não é mais necessário após o mmap (ele mantém uma referência ao arquivo)
        unsafe { libc::close(fd) };

        Self {
            ptr: base_ptr as *mut T,
            size_elements,
            _marker: PhantomData,
        }
    }

    /// Retorna o tamanho físico do buffer (antes do espelhamento) em elementos.
    pub fn size(&self) -> usize {
        self.size_elements
    }
}

impl<T> Deref for VirtualRingBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // Retornamos um slice que cobre as duas metades (tamanho 2x)
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_elements * 2) }
    }
}

impl<T> DerefMut for VirtualRingBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_elements * 2) }
    }
}

impl<T> Drop for VirtualRingBuffer<T> {
    fn drop(&mut self) {
        let element_size = std::mem::size_of::<T>();
        let size_bytes = self.size_elements * element_size;
        unsafe {
            munmap(self.ptr as *mut c_void, size_bytes * 2);
        }
    }
}

impl<T: Clone> Clone for VirtualRingBuffer<T> {
    fn clone(&self) -> Self {
        let mut new_vring = Self::new(self.size_elements);
        new_vring[..self.size_elements].clone_from_slice(&self[..self.size_elements]);
        new_vring
    }
}

unsafe impl<T: Send> Send for VirtualRingBuffer<T> {}
unsafe impl<T: Sync> Sync for VirtualRingBuffer<T> {}
