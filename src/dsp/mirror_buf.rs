// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Buffer Espelhado (MirroredBuffer) via Mapeamento de Memória Espelhada
//!
//! O `MirroredBuffer` é uma técnica avançada de gerenciamento de memória que resolve
//! o problema clássico da "quebra de contiguidade" em buffers circulares/fita de atraso.
//!
//! ## O Problema: A Fronteira do Buffer
//! Em buffers circulares tradicionais, ao atingir o fim do espaço alocado, o ponteiro volta
//! para o início. Se um algoritmo DSP (como uma Convolução ou FFT) precisar ler uma janela
//! de 1024 amostras, mas o ponteiro estiver a apenas 500 amostras do fim, o programador
//! teria que lidar com a leitura em duas partes ou realizar uma cópia cara (`copy_within`).
//! Isso introduz lógica complexa (`if/else`) e prejudica a performance no "hot-path".
//!
//! ## A Solução: O "Truque" do Espelhamento
//! Esta estrutura utiliza recursos da Unidade de Gerenciamento de Memória (MMU) do processador
//! para mapear o **mesmo bloco de memória física** duas vezes consecutivas no espaço de
//! endereçamento virtual:
//!
//! ```text
//! Espaço Virtual: [ Bloco Físico (Página 0..N) ] [ Bloco Físico (Página 0..N) ]
//!                 ^                             ^
//!                 |                             |
//!           Início do Buffer              Espelhamento do Início
//! ```
//!
//! Graças a este mapeamento, qualquer acesso que "ultrapasse" o fim do primeiro bloco cairá
//! automaticamente no início do segundo bloco — que é, fisicamente, o próprio início do buffer.
//!
//! ## Benefícios para Áudio Real-Time
//! 1. **Acesso Linear**: Algoritmos podem ler janelas contíguas de qualquer tamanho (até o tamanho total do buffer) sem se preocupar com o "wrap".
//! 2. **Zero-Copy**: Elimina a necessidade de copiar dados para buffers temporários para linearizá-los.
//! 3. **Performance SIMD**: Permite que instruções vetoriais (AVX/SSE) processem dados através da fronteira do buffer sem interrupções de lógica.
//! 4. **Sem Branches**: Remove operações de módulo (`%`) e condições `if`, otimizando a predição de desvios do processador.
use libc::{
    _SC_PAGESIZE, MAP_ANONYMOUS, MAP_FAILED, MAP_FIXED, MAP_PRIVATE, MAP_SHARED, PROT_NONE,
    PROT_READ, PROT_WRITE, c_void, ftruncate, mmap, munmap, sysconf,
};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(not(target_os = "linux"))]
mod fallback;

/// Um Buffer Espelhado que usa mapeamento de memória espelhado.
///
/// Esta estrutura mapeia o mesmo conteúdo físico duas vezes consecutivas no espaço
/// de endereçamento virtual. Isso permite que acessos que "atravessariam" o fim do
/// buffer sejam feitos de forma linear e contígua, eliminando a necessidade de
/// operações de "rewind" ou "copy_within" no hot-path do DSP.
pub struct MirroredBuffer<T> {
    ptr: *mut T,
    size_elements: usize,
    _marker: PhantomData<T>,
}

thread_local! {
    pub(crate) static SIMULATE_FAIL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Define se as próximas chamadas de criação do `MirroredBuffer` devem simular
/// falha de alocação de memória virtual.
pub fn set_simulate_fail(fail: bool) {
    SIMULATE_FAIL.with(|f| f.set(fail));
}

impl<T> MirroredBuffer<T> {
    /// Cria um novo buffer espelhado.
    ///
    /// O tamanho `requested_size` (em elementos) será arredondado para cima para
    /// o próximo múltiplo do tamanho da página do sistema.
    #[cold]
    pub fn new(requested_size: usize) -> std::io::Result<Self> {
        let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
        let element_size = std::mem::size_of::<T>();

        // Garantir que o tamanho do elemento não seja zero (ex: ZST)
        if element_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "MirroredBuffer does not support Zero Sized Types",
            ));
        }

        if requested_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "requested_size must be greater than zero",
            ));
        }

        let requested_bytes = match requested_size.checked_mul(element_size) {
            Some(val) => val,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "requested_size * element_size overflowed",
                ));
            }
        };

        // Arredonda para múltiplo da página
        let page_mask = page_size - 1;
        let size_bytes = match requested_bytes.checked_add(page_mask) {
            Some(val) => val & !page_mask,
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "size_bytes calculation overflowed",
                ));
            }
        };
        let size_elements = size_bytes / element_size;

        // 1. Criar backing store (memfd no Linux, stub fallback em outras plataformas)
        let fd = unsafe {
            #[cfg(target_os = "linux")]
            {
                linux::create_backing_fd()?
            }
            #[cfg(not(target_os = "linux"))]
            {
                fallback::create_backing_fd()?
            }
        };

        // 2. Definir o tamanho do arquivo
        if unsafe { ftruncate(fd, size_bytes as libc::off_t) } == -1 {
            let err = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
        }

        // 3. Reservar espaço virtual contíguo (2x tamanho)
        let total_size = match size_bytes.checked_mul(2) {
            Some(val) => val,
            None => {
                unsafe { libc::close(fd) };
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "total_size (size_bytes * 2) overflowed",
                ));
            }
        };

        // Assegurar invariante requerida antes de mmap
        assert!(
            requested_size > 0,
            "requested_size must be greater than zero"
        );

        let base_ptr = unsafe {
            if SIMULATE_FAIL.with(|f| f.get()) {
                *libc::__errno_location() = libc::ENOMEM;
                MAP_FAILED
            } else {
                mmap(
                    ptr::null_mut(),
                    total_size,
                    PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                )
            }
        };
        if base_ptr == MAP_FAILED {
            let err = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(err);
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
            return Err(err);
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
            return Err(err);
        }

        // O FD não é mais necessário após o mmap (ele mantém uma referência ao arquivo)
        unsafe { libc::close(fd) };

        Ok(Self {
            ptr: base_ptr as *mut T,
            size_elements,
            _marker: PhantomData,
        })
    }

    /// Retorna o tamanho físico do buffer (antes do espelhamento) em elementos.
    pub fn size(&self) -> usize {
        self.size_elements
    }
}

impl<T> std::fmt::Debug for MirroredBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MirroredBuffer")
            .field("ptr", &self.ptr)
            .field("size_elements", &self.size_elements)
            .field("capacity_virtual", &(self.size_elements * 2))
            .finish()
    }
}

impl<T> Deref for MirroredBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // Retornamos um slice que cobre as duas metades (tamanho 2x)
        unsafe { std::slice::from_raw_parts(self.ptr, self.size_elements * 2) }
    }
}

impl<T> DerefMut for MirroredBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size_elements * 2) }
    }
}

impl<T> Drop for MirroredBuffer<T> {
    fn drop(&mut self) {
        let element_size = std::mem::size_of::<T>();
        let size_bytes = self.size_elements * element_size;
        unsafe {
            munmap(self.ptr as *mut c_void, size_bytes * 2);
        }
    }
}

impl<T: Clone> Clone for MirroredBuffer<T> {
    #[cold]
    fn clone(&self) -> Self {
        match Self::new(self.size_elements) {
            Ok(mut new_buf) => {
                new_buf[..self.size_elements].clone_from_slice(&self[..self.size_elements]);
                new_buf
            }
            Err(err) => {
                std::panic::panic_any(format!("Failed to clone MirroredBuffer: {:?}", err));
            }
        }
    }
}

unsafe impl<T: Send> Send for MirroredBuffer<T> {}
unsafe impl<T: Sync> Sync for MirroredBuffer<T> {}

#[cfg(all(test, target_os = "linux"))]
#[path = "mirror_buf_test.rs"]
mod mirror_buf_test;
