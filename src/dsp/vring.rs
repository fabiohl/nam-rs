// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Buffer Circular Virtual (VRing) via Mapeamento de Memória Espelhada
//!
//! O `VirtualRingBuffer` é uma técnica avançada de gerenciamento de memória que resolve
//! o problema clássico da "quebra de contiguidade" em buffers circulares.
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

thread_local! {
    pub(crate) static SIMULATE_FAIL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Define se as próximas chamadas de criação do `VirtualRingBuffer` devem simular
/// falha de alocação de memória virtual.
pub fn set_simulate_fail(fail: bool) {
    SIMULATE_FAIL.with(|f| f.set(fail));
}

impl<T> VirtualRingBuffer<T> {
    /// Cria um novo buffer circular virtual.
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
                "VirtualRingBuffer does not support Zero Sized Types",
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

        // 1. Criar backing store (memfd no Linux)
        // MFD_CLOEXEC evita que o FD seja herdado por processos filhos.
        let fd = unsafe {
            if SIMULATE_FAIL.with(|f| f.get()) {
                *libc::__errno_location() = libc::ENOMEM;
                -1
            } else {
                libc::memfd_create(c"vring".as_ptr(), libc::MFD_CLOEXEC)
            }
        };
        if fd == -1 {
            return Err(std::io::Error::last_os_error());
        }

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

impl<T> std::fmt::Debug for VirtualRingBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualRingBuffer")
            .field("ptr", &self.ptr)
            .field("size_elements", &self.size_elements)
            .field("capacity_virtual", &(self.size_elements * 2))
            .finish()
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
    #[cold]
    fn clone(&self) -> Self {
        match Self::new(self.size_elements) {
            Ok(mut new_vring) => {
                new_vring[..self.size_elements].clone_from_slice(&self[..self.size_elements]);
                new_vring
            }
            Err(err) => {
                std::panic::panic_any(format!("Failed to clone VirtualRingBuffer: {:?}", err));
            }
        }
    }
}

unsafe impl<T: Send> Send for VirtualRingBuffer<T> {}
unsafe impl<T: Sync> Sync for VirtualRingBuffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vring_page_alignment() -> Result<(), Box<dyn std::error::Error>> {
        let page_size = unsafe { sysconf(_SC_PAGESIZE) } as usize;
        let element_size = std::mem::size_of::<f32>();

        // Pede 1 elemento, deve arredondar para 1 página
        let vring = VirtualRingBuffer::<f32>::new(1)?;
        let expected_elements = page_size / element_size;

        assert_eq!(vring.size(), expected_elements);
        assert_eq!(vring.len(), expected_elements * 2);
        Ok(())
    }

    #[test]
    fn test_vring_mirroring() -> Result<(), Box<dyn std::error::Error>> {
        // Teste do Espelhamento: Verifica se o "truque" da memória virtual está funcionando.
        // Qualquer valor escrito na primeira metade deve aparecer instantaneamente na
        // segunda metade (o espelho), pois ambas as janelas apontam para o mesmo lugar físico.
        // Cria um buffer pequeno (será arredondado para 1 página)
        let mut vring = VirtualRingBuffer::<u32>::new(1)?;
        let size = vring.size();

        // 1. Escrita no início da primeira metade
        vring[0] = 0x12345678;
        // Deve ser visível no início da segunda metade (espelho)
        assert_eq!(vring[size], 0x12345678);

        // 2. Escrita no final da primeira metade
        vring[size - 1] = 0xDEADBEEF;
        // Deve ser visível no final da segunda metade
        assert_eq!(vring[2 * size - 1], 0xDEADBEEF);

        // 3. Acesso contíguo cruzando a fronteira (o "Pulo do Gato")
        // Vamos escrever uma sequência de valores que atravessa exatamente o meio do buffer.
        // Em um buffer comum isso exigiria dois loops ou um 'if', mas aqui é linear.
        let middle = size;
        let start = middle - 8;
        for i in 0..16 {
            vring[start + i] = i as u32;
        }

        // Verifica se a primeira metade (original) reflete as mudanças
        // Os primeiros 8 valores foram escritos no final da primeira metade
        for i in 0..8 {
            assert_eq!(vring[size - 8 + i], i as u32);
        }
        // Os próximos 8 valores foram escritos no início da segunda metade,
        // o que deve ter modificado o início da PRIMEIRA metade física.
        for i in 8..16 {
            assert_eq!(vring[i - 8], i as u32);
        }
        Ok(())
    }

    #[test]
    fn test_vring_clone() -> Result<(), Box<dyn std::error::Error>> {
        let mut vring = VirtualRingBuffer::<i32>::new(100)?;
        vring[0] = 42;

        let vring2 = vring.clone();
        assert_eq!(vring2[0], 42);
        assert_eq!(vring2.size(), vring.size());

        // Modifica o original, o clone deve permanecer inalterado
        vring[0] = 99;
        assert_eq!(
            vring2[0], 42,
            "Clones de VirtualRingBuffer devem ser independentes"
        );
        Ok(())
    }

    #[test]
    fn test_vring_zst_error() {
        assert!(VirtualRingBuffer::<()>::new(1024).is_err());
    }

    #[test]
    fn test_vring_large_allocation() -> Result<(), Box<dyn std::error::Error>> {
        // Testa alocação de ~1MB
        let size = 1024 * 1024 / 4;
        let vring = VirtualRingBuffer::<f32>::new(size)?;
        assert!(vring.size() >= size);
        // Apenas garante que não deu panic e o mmap foi bem sucedido
        Ok(())
    }

    #[test]
    fn test_vring_debug() -> Result<(), Box<dyn std::error::Error>> {
        let vring = VirtualRingBuffer::<f32>::new(1024)?;
        let debug_str = format!("{:?}", vring);
        assert!(debug_str.contains("VirtualRingBuffer"));
        assert!(debug_str.contains("ptr"));
        assert!(debug_str.contains("size_elements"));
        Ok(())
    }
}
