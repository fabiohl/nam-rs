// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Estruturas de dados e flags compartilhadas para coordenação lock-free entre threads.
//!
//! Este módulo define toda a "fiação invisível" que conecta a interface CLI (o controle
//! remoto do músico) com o motor de áudio DSP (o amplificador neural rodando em tempo real).
//!
//! ## Componentes principais
//!
//! - **`SHUTDOWN`**: flag global que sinaliza a todas as threads que o programa deve encerrar.
//! - **`ParamPayload`**: "pacotes" de parâmetros (ganho de entrada/saída, troca de modelo)
//!   que a CLI envia para o DSP sem travar nenhuma thread (lock-free via ring buffer SPSC).
//! - **`RtStatusFlags`**: flags atômicas para comunicação silenciosa RT→Main (sem I/O no callback).
//!   Permitem que o motor DSP reporte seu estado sem jamais chamar `println!`.
//! - Canal SPSC de `NamResampler`: a thread principal constrói o resampler (com alocações
//!   de memória) fora do tempo real e o envia para o callback DSP via canal lock-free.

use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU64, Ordering};

/// Flag global para shutdown coordenado e gracioso entre todas as threads.
/// Definida como `true` pelo handler de CTRL+C.
pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Flag indicando que a thread DSP precisa de um novo `NamResampler`.
pub const RT_STATUS_NEEDS_RESAMPLER_REBUILD: u64 = 1 << 0;
/// Indica se a última tentativa de rebuild do resampler pela thread principal falhou.
pub const RT_STATUS_RESAMPLER_REBUILD_FAILED: u64 = 1 << 1;
/// `true` se a thread DSP confirmou operação em `SCHED_FIFO`.
pub const RT_STATUS_RT_IS_FIFO: u64 = 1 << 2;
/// Flag indicando que houve saturação (clipping) no áudio de saída.
pub const RT_STATUS_HAS_CLIPPED: u64 = 1 << 3;
/// Flag indicando que o buffer atual está em silêncio (abaixo de −80 dBFS).
pub const RT_STATUS_IS_SILENT: u64 = 1 << 4;
/// Flag indicando que houve overflow no canal de GC.
pub const RT_STATUS_GC_OVERFLOW: u64 = 1 << 5;

/// Flags atômicas de status para comunicação silenciosa RT→Main.
///
/// A thread DSP seta flags atômicas ao invés de chamar `println!`/`eprintln!`.
/// A thread principal lê estas flags periodicamente e imprime logs ao usuário.
/// Isso garante que **zero I/O** ocorra no callback RT.
#[repr(align(128))]
pub struct RtStatusFlags {
    /// Sample rate efetivamente ativo na thread DSP após reconstrução do resampler.
    /// Setado pela thread DSP ao consumir um novo `NamResampler` do canal SPSC.
    /// Valor `0` indica ausência de atualização pendente.
    pub active_rate: AtomicU32,
    /// Notificação de mudança de rate para fins de log.
    /// Valor `0` indica nenhuma mudança desde o último poll.
    pub active_rate_changed: AtomicU32,

    /// Rate alvo que a thread DSP detectou do pw mas não conseguiu aplicar (aguardando rebuild).
    /// A thread principal lê este valor para saber qual rate construir.
    /// Valor `0` indica nenhuma requisição pendente.
    pub requested_pw_rate: AtomicU32,

    /// Rate alvo do modelo carregado (NAM). O padrão usual é 48000.
    pub requested_nam_rate: AtomicU32,

    /// Prioridade RT efetiva confirmada por `pthread_getschedparam`.
    /// Valor `-1` indica que a verificação ainda não foi realizada.
    /// Setado no cold-path do primeiro frame da thread DSP.
    pub rt_priority: AtomicI32,

    /// Contador atômico de overloads de DSP (XRUNs virtuais).
    /// Incrementado pelo callback RT se o processamento exceder 85% do tempo limite (budget).
    pub dsp_overloads: AtomicU32,

    /// Tempo de processamento do último ciclo DSP em ticks (RDTSC).
    /// Lido pela thread principal e convertido para Duration via Anchor.
    pub dsp_cycle_time: AtomicU64,

    /// Número de amostras processadas no último ciclo (para cálculo de budget).
    pub last_n_samples: AtomicU32,

    /// Histograma de latência para análise estatística (P50, P95, P99).
    pub latency_hist: crate::dsp::telemetry::LatencyHistogram,

    /// Bitmask atômico contendo os estados binários (needs_rebuild, clipped, silent, etc).
    /// Reduz Cache Bouncing ao condensar múltiplos estados em uma única linha de cache.
    pub status_bits: AtomicU64,
}

impl RtStatusFlags {
    /// Cria uma nova instância com valores iniciais zerados/sentinela.
    pub fn new() -> Self {
        Self {
            active_rate: AtomicU32::new(0),
            active_rate_changed: AtomicU32::new(0),
            requested_pw_rate: AtomicU32::new(0),
            requested_nam_rate: AtomicU32::new(48_000),
            rt_priority: AtomicI32::new(-1),
            dsp_overloads: AtomicU32::new(0),
            dsp_cycle_time: AtomicU64::new(0),
            last_n_samples: AtomicU32::new(0),
            latency_hist: crate::dsp::telemetry::LatencyHistogram::new(),
            status_bits: AtomicU64::new(0),
        }
    }

    /// Seta uma ou mais flags no bitmask.
    #[inline(always)]
    pub fn set_flag(&self, flag: u64) {
        self.status_bits.fetch_or(flag, Ordering::Relaxed);
    }

    /// Limpa uma ou mais flags no bitmask.
    #[inline(always)]
    pub fn clear_flag(&self, flag: u64) {
        self.status_bits.fetch_and(!flag, Ordering::Relaxed);
    }

    /// Verifica se uma flag está ativa.
    #[inline(always)]
    pub fn check_flag(&self, flag: u64) -> bool {
        (self.status_bits.load(Ordering::Relaxed) & flag) != 0
    }

    /// Verifica se uma flag está ativa e a limpa atomicamente em uma única operação.
    /// Retorna `true` se a flag estava ativa.
    #[inline(always)]
    pub fn check_and_clear_flag(&self, flag: u64) -> bool {
        (self.status_bits.fetch_and(!flag, Ordering::Relaxed) & flag) != 0
    }
}

impl Default for RtStatusFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// Payload SPSC enviado do Host (CLI/UI) para a Thread DSP.
/// Adota alinhamento a 128 bytes para mitigar False Sharing.
#[repr(align(128))]
pub enum ParamPayload {
    /// Injeta o ganho de entrada (Input Gain) como multiplicador linear.
    InputGain(f32),
    /// Injeta o ganho de saída (Output Gain) como multiplicador linear.
    OutputGain(f32),
    /// Carrega a topologia matemática decodificada informando, simultaneamente, os limiares
    /// esperados pelo criador do modelo (resolvidos da tag input_level_dbu e loudness).
    /// O ponteiro garante alocação-zero (no-heap) e inicialização determinística.
    LoadModel {
        /// O modelo encapsulado para a inferência neural (Canal Esquerdo)
        model_l: Option<Box<crate::models::DynamicModel>>,
        /// O modelo encapsulado para a inferência neural (Canal Direito)
        model_r: Option<Box<crate::models::DynamicModel>>,
        /// Ajuste de ganho esperado na entrada como multiplicador linear.
        input_mult_adj: f32,
        /// Ajuste de ganho esperado na saída como multiplicador linear.
        output_mult_adj: f32,
        /// Sample rate exigido pelo modelo (geralmente 48000).
        sample_rate: u32,
    },
    /// Injeta as configurações do Gate de Silêncio/Mono.
    GateConfig(crate::dsp::gate::GateParams),
}

/// Item a ser coletado pelo Garbage Collector fora da thread de tempo real.
///
/// Encapsula tanto modelos neurais quanto resamplers obsoletos em uma única
/// estrutura para simplificar a gestão de memória (Drop-Delegation).
#[allow(clippy::large_enum_variant)]
pub enum GcItem {
    /// Um modelo dinâmico (LSTM ou WaveNet).
    Model(Box<crate::models::DynamicModel>),
    /// Um resampler.
    Resampler(crate::dsp::resampler::NamResampler),
    /// Variante de teste para validação de integridade e stress.
    #[cfg(test)]
    Test(Box<std::sync::Arc<std::sync::atomic::AtomicU32>>),
}

/// Um buffer circular de sobrescrita (overwrite ring buffer) para ponteiros de GC.
///
/// Atua como "parking lot" compartilhado entre RT e Main. Se o canal SPSC principal
/// estiver cheio, o RT estaciona o ponteiro aqui. Se este buffer também encher,
/// ele sobrescreve o mais antigo (causando leak, mas apenas em cenários extremos).
/// Projetado para suportar apenas `Box<GcItem>` via ponteiro raw para garantir RT-safety.
pub struct GcOverflowBuffer {
    slots: Box<[AtomicPtr<GcItem>]>,
    write_idx: AtomicU64,
}

impl GcOverflowBuffer {
    /// Cria um novo buffer de sobrescrita vazio com a capacidade especificada.
    ///
    /// # Panics
    /// Panica se `capacity` for 0.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "GcOverflowBuffer: capacity deve ser maior que 0 para evitar panic por divisão por zero."
        );
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        Self {
            slots: slots.into_boxed_slice(),
            write_idx: AtomicU64::new(0),
        }
    }

    /// Tenta estacionar um item no buffer.
    ///
    /// # Safety
    /// O item deve ser convertido para `Box<GcItem>` antes de ser passado via ponteiro.
    /// RT-Safe: usa `swap` atômico sem locks.
    pub fn push_raw(&self, ptr: *mut GcItem) -> Option<*mut GcItem> {
        let len = self.slots.len() as u64;
        let idx = (self.write_idx.fetch_add(1, Ordering::Relaxed) % len) as usize;
        let old_ptr = self.slots[idx].swap(ptr, Ordering::Acquire);
        if old_ptr.is_null() {
            None
        } else {
            Some(old_ptr)
        }
    }

    /// Drena todos os itens presentes no buffer.
    /// Chamado pela thread de controle (Non-RT).
    pub fn drain(&self) -> Vec<GcItem> {
        let mut items = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let ptr = slot.swap(std::ptr::null_mut(), Ordering::Release);
            if !ptr.is_null() {
                unsafe {
                    let item = *Box::from_raw(ptr);
                    items.push(item);
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

/// Resultado da inicialização SPSC: canais de parâmetros, GC de modelos,
/// canal de resamplers RT-safe, e flags de status atômicas.
pub struct SpscChannels {
    /// Produtor de parâmetros CLI→DSP.
    pub param_producer: Producer<ParamPayload>,
    /// Consumidor de parâmetros CLI→DSP (movido para o callback RT).
    pub param_consumer: Consumer<ParamPayload>,
    /// Produtor GC: thread DSP envia itens obsoletos para drop fora do RT.
    pub gc_producer: Producer<GcItem>,
    /// Consumidor GC: thread background executa `drop()`.
    pub gc_consumer: Consumer<GcItem>,
    /// Buffer de fallback para overflow de GC (overwrite).
    pub gc_overflow: Arc<GcOverflowBuffer>,
    /// Produtor de resamplers: thread principal constrói e envia para o callback RT.
    pub resampler_producer: Producer<crate::dsp::resampler::NamResampler>,
    /// Consumidor de resamplers: callback RT drena para substituir o resampler ativo.
    pub resampler_consumer: Consumer<crate::dsp::resampler::NamResampler>,
    /// Flags atômicas de status compartilhadas entre RT e Main (zero I/O no callback).
    pub rt_status: Arc<RtStatusFlags>,
}

/// Cria e retorna a malha SPSC lock-free completa para o pipeline.
///
/// Inclui canais para:
/// - Parâmetros CLI→DSP (`ParamPayload`)
/// - GC de modelos obsoletos (Drop-Delegation)
/// - Resamplers pré-construídos (Main→RT, zero alocação no callback)
/// - Flags atômicas de status RT→Main
///
/// `capacity` deve ser preferencialmente potência de 2.
pub fn setup_spsc(capacity: usize) -> SpscChannels {
    let (param_prod, param_cons) = RingBuffer::new(capacity);
    let (gc_prod, gc_cons) = RingBuffer::new(capacity * 4); // Capacidade quadruplicada para garbage collection segura
    // Canal de resampler: capacidade pequena (apenas 1 em trânsito por vez, tipicamente)
    let (rs_prod, rs_cons) = RingBuffer::new(4);
    let rt_status = Arc::new(RtStatusFlags::new());
    // O buffer de overflow deve ser grande o suficiente para acomodar picos de trocas de modelos.
    // Usamos 64 como base, ou a capacidade solicitada se for maior.
    let gc_overflow = Arc::new(GcOverflowBuffer::new(capacity.max(64)));

    SpscChannels {
        param_producer: param_prod,
        param_consumer: param_cons,
        gc_producer: gc_prod,
        gc_consumer: gc_cons,
        gc_overflow,
        resampler_producer: rs_prod,
        resampler_consumer: rs_cons,
        rt_status,
    }
}

#[cfg(test)]
#[path = "spsc_test.rs"]
mod spsc_test;
