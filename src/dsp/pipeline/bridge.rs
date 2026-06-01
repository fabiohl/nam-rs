// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tipos de ponte (bridge) para comunicação lock-free entre capture e playback.
//!
//! Contém `DspBridge`, `BridgeBuffer`, `BridgeRef`, `DspBridgeWriter`,
//! `DspBridgeReader` e as constantes `MAX_BRIDGE_BUF` / `MAX_RESAMP_BUF`.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use std::sync::atomic::Ordering;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Tamanho máximo do buffer intermediário entre as duas streams (capture → playback).
/// Dimensionado para o quantum máximo do PipeWire (`max-quantum = 8192`).
pub const MAX_BRIDGE_BUF: usize = 8192;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Tamanho máximo do buffer para resampling.
///
/// **Contrato de Segurança RT**: Este valor determina o tamanho dos buffers pré-alocados
/// no `DspPipelineContext`. Aumentar este valor impacta o tamanho do objeto da closure
/// de processamento (que deve caber na stack da thread RT ou ser movido para o heap).
/// Atualmente fixado em 4096 amostras (16 KiB por canal).
pub const MAX_RESAMP_BUF: usize = 8192;

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Buffer individual de áudio para o DspBridge (double-buffer).
#[repr(align(128))]
pub struct BridgeBuffer {
    /// Buffer de saída processada, canal esquerdo.
    pub buf_l: [f32; MAX_BRIDGE_BUF],
    /// Buffer de saída processada, canal direito.
    pub buf_r: [f32; MAX_BRIDGE_BUF],
    /// Número de amostras válidas no buffer atual.
    pub n_samples: u32,
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Buffer compartilhado entre o callback de captura (DSP) e o callback de playback.
///
/// O capture callback escreve o resultado processado aqui com `fence(Release)`;
/// o playback callback lê com `fence(Acquire)`. A `generation` atômica permite
/// ao playback detectar se há dados novos disponíveis sem spin-lock.
///
/// Alinhado a 128 bytes para evitar false-sharing entre os dois callbacks RT.
#[repr(align(128))]
pub struct DspBridge {
    /// Os dois buffers físicos (front / back) para o double-buffering.
    pub buffers: [BridgeBuffer; 2],
    /// Índice do buffer ativo para LEITURA (0 ou 1). O capture sempre escreve no (1 - ativo).
    pub active_read_idx: std::sync::atomic::AtomicUsize,
    /// Contador de geração — incrementado a cada escrita pelo capture callback.
    /// O playback compara com sua cópia local para detectar novos dados.
    pub generation: std::sync::atomic::AtomicU64,
    /// Contador de geração consumida — atualizado pelo playback callback.
    pub consumed_gen: std::sync::atomic::AtomicU64,
    /// Contador de frames descartados (sobrescritos sem consumo).
    /// Incrementado pelos callbacks RT, drenado via `drain_dropped_frames()` pelo loop principal.
    pub dropped_frames: std::sync::atomic::AtomicU32,
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
impl DspBridge {
    /// Drena o contador de frames descartados, retornando o valor acumulado e zerando-o.
    ///
    /// RT-Safe para o leitor: usa `swap` atômico sem locks.
    pub fn drain_dropped_frames(&self) -> u32 {
        self.dropped_frames.swap(0, Ordering::Relaxed)
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[derive(Clone, Copy)]
/// Referência segura para o DspBridge (compartilhado entre threads via ponteiro).
pub struct BridgeRef(*mut DspBridge);

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
impl BridgeRef {
    /// Cria um novo BridgeRef.
    /// # Safety
    /// O ponteiro deve ser válido e não nulo.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        debug_assert!(!ptr.is_null());
        Self(ptr)
    }

    /// Cria um BridgeRef nulo (para quando a ponte não é necessária).
    #[inline(always)]
    pub fn null() -> Self {
        Self(std::ptr::null_mut())
    }

    /// Verifica se o BridgeRef é nulo.
    #[inline(always)]
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }

    /// Retorna o ponteiro bruto interno.
    /// # Safety
    /// O chamador deve garantir que o ponteiro é válido se for desreferenciado.
    #[inline(always)]
    pub unsafe fn as_ptr(self) -> *mut DspBridge {
        self.0
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[derive(Clone, Copy)]
/// Face de escrita do `DspBridge` exposta à capture thread.
pub struct DspBridgeWriter(std::ptr::NonNull<DspBridge>);

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// SAFETY: DspBridgeWriter é seguro de enviar entre threads para inicialização.
/// O acesso em tempo real ocorre de forma exclusiva/síncrona na capture thread.
unsafe impl Send for DspBridgeWriter {}
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// SAFETY: DspBridgeWriter é seguro de compartilhar entre threads pois o acesso é internamente atômico/síncrono.
unsafe impl Sync for DspBridgeWriter {}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
impl DspBridgeWriter {
    /// Cria um `DspBridgeWriter` a partir de um ponteiro bruto para `DspBridge`.
    /// # Safety
    /// O ponteiro deve ser válido e não nulo.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        debug_assert!(!ptr.is_null());
        Self(unsafe { std::ptr::NonNull::new_unchecked(ptr) })
    }

    /// Cria um `DspBridgeWriter` a partir de um `BridgeRef`.
    /// Retorna `None` se a referência for nula.
    #[inline(always)]
    pub fn from_ref(r: BridgeRef) -> Option<Self> {
        std::ptr::NonNull::new(r.0).map(Self)
    }

    /// Escreve um bloco de amostras estéreo no buffer back-buffer ativo do bridge.
    pub fn write_block(&self, resamp_out_l: &[f32], resamp_out_r: &[f32], n_pw: usize) {
        // SAFETY: O ponteiro subjacente é válido e o acesso ao back-buffer é exclusivo e atômico.
        unsafe {
            let bridge = self.0.as_ref();
            let back_idx = 1 - bridge.active_read_idx.load(Ordering::Relaxed);
            let back_buf = &mut (*self.0.as_ptr()).buffers[back_idx];

            let n_bridge = n_pw.min(MAX_BRIDGE_BUF);
            core::ptr::copy_nonoverlapping(
                resamp_out_l.as_ptr(),
                back_buf.buf_l.as_mut_ptr(),
                n_bridge,
            );
            core::ptr::copy_nonoverlapping(
                resamp_out_r.as_ptr(),
                back_buf.buf_r.as_mut_ptr(),
                n_bridge,
            );
            back_buf.n_samples = n_bridge as u32;

            bridge.active_read_idx.store(back_idx, Ordering::Release);

            let current_gen = bridge.generation.load(Ordering::Relaxed);
            let consumed_gen = bridge.consumed_gen.load(Ordering::Relaxed);
            if current_gen > consumed_gen {
                let _ =
                    bridge
                        .dropped_frames
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                            Some(v.saturating_add(1))
                        });
            }

            bridge.generation.store(current_gen + 1, Ordering::Release);
        }
    }

    /// Reseta o buffer back-buffer ativo para indicar silêncio (0 amostras).
    pub fn write_silence(&self) {
        // SAFETY: O ponteiro subjacente é válido e o acesso ao back-buffer é exclusivo e atômico.
        unsafe {
            let bridge = self.0.as_ref();
            let back_idx = 1 - bridge.active_read_idx.load(Ordering::Relaxed);
            let back_buf = &mut (*self.0.as_ptr()).buffers[back_idx];
            back_buf.n_samples = 0;

            bridge.active_read_idx.store(back_idx, Ordering::Release);

            let current_gen = bridge.generation.load(Ordering::Relaxed);
            let consumed_gen = bridge.consumed_gen.load(Ordering::Relaxed);
            if current_gen > consumed_gen {
                let _ =
                    bridge
                        .dropped_frames
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                            Some(v.saturating_add(1))
                        });
            }

            bridge.generation.store(current_gen + 1, Ordering::Release);
        }
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[derive(Clone, Copy)]
/// Face de leitura do `DspBridge` exposta à playback thread.
pub struct DspBridgeReader(std::ptr::NonNull<DspBridge>);

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// SAFETY: DspBridgeReader é seguro de enviar entre threads para inicialização.
/// O acesso em tempo real ocorre de forma exclusiva/síncrona na playback thread.
unsafe impl Send for DspBridgeReader {}
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// SAFETY: DspBridgeReader é seguro de compartilhar entre threads pois o acesso é internamente atômico/síncrono.
unsafe impl Sync for DspBridgeReader {}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
impl DspBridgeReader {
    /// Cria um `DspBridgeReader` a partir de um ponteiro bruto para `DspBridge`.
    /// # Safety
    /// O ponteiro deve ser válido e não nulo.
    #[inline(always)]
    pub unsafe fn new(ptr: *mut DspBridge) -> Self {
        debug_assert!(!ptr.is_null());
        Self(unsafe { std::ptr::NonNull::new_unchecked(ptr) })
    }

    /// Cria um `DspBridgeReader` a partir de um `BridgeRef`.
    /// Retorna `None` se a referência for nula.
    #[inline(always)]
    pub fn from_ref(r: BridgeRef) -> Option<Self> {
        std::ptr::NonNull::new(r.0).map(Self)
    }

    /// Tenta ler um bloco de áudio do bridge, passando referências aos canais L e R para um closure.
    ///
    /// Retorna `Some(R)` se houver um bloco novo e válido disponível.
    /// Caso contrário, retorna `None`.
    pub fn read_block<F, R>(&self, last_bridge_gen: &mut u64, f: F) -> Option<R>
    where
        F: FnOnce(&[f32], &[f32]) -> R,
    {
        // SAFETY: O ponteiro é válido e o acesso ao buffer ativo é exclusivo e atômico.
        unsafe {
            let bridge = self.0.as_ref();
            let current_gen = bridge.generation.load(Ordering::Acquire);
            if current_gen == *last_bridge_gen {
                return None;
            }
            *last_bridge_gen = current_gen;
            bridge.consumed_gen.store(current_gen, Ordering::Release);

            let read_idx = bridge.active_read_idx.load(Ordering::Relaxed);
            let front_buf = &bridge.buffers[read_idx];
            let n_samples = front_buf.n_samples as usize;
            if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
                return None;
            }

            Some(f(
                &front_buf.buf_l[..n_samples],
                &front_buf.buf_r[..n_samples],
            ))
        }
    }
}
