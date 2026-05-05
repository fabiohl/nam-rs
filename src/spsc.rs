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
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64};

/// Flag global para shutdown coordenado e gracioso entre todas as threads.
/// Definida como `true` pelo handler de CTRL+C.
pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

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

    /// Flag indicando que a thread DSP precisa de um novo `NamResampler`.
    /// Setada pelo callback quando detecta mudança de rate via `AtomicU32`.
    /// Lida e consumida pela thread principal.
    pub needs_resampler_rebuild: AtomicBool,

    /// Indica se a última tentativa de rebuild do resampler pela thread principal falhou.
    /// A thread principal seta `true` ao falhar e o callback pode ler para decidir
    /// se continua com o resampler anterior.
    pub resampler_rebuild_failed: AtomicBool,

    /// `true` se a thread DSP confirmou operação em `SCHED_FIFO` via
    /// `pthread_getschedparam`. Setado no cold-path do primeiro frame.
    /// Lido pelo loop principal para confirmação auditável no log.
    pub rt_is_fifo: AtomicBool,

    /// Prioridade RT efetiva confirmada por `pthread_getschedparam`.
    /// Valor `-1` indica que a verificação ainda não foi realizada.
    /// Setado no cold-path do primeiro frame da thread DSP.
    pub rt_priority: AtomicI32,

    /// Flag atômica indicando que houve saturação (clipping) no áudio de saída.
    /// Setada pelo callback RT ao detectar samples fora do intervalo [-1.0, 1.0].
    /// A thread principal exibe um alerta e a reseta.
    pub has_clipped: AtomicBool,

    /// Contador atômico de overloads de DSP (XRUNs virtuais).
    /// Incrementado pelo callback RT se o processamento exceder 85% do tempo limite (budget).
    pub dsp_overloads: AtomicU32,

    /// Flag atômica indicando que o buffer atual está em silêncio (abaixo de −80 dBFS).
    /// O callback RT seta `true` ao detectar silêncio e pular o pipeline DSP pesado.
    /// A thread principal lê para exibir/ocultar o status de "Silence Bypass" ativo.
    pub is_silent: AtomicBool,

    /// Tempo de processamento do último ciclo DSP em ticks (RDTSC).
    /// Lido pela thread principal e convertido para Duration via Anchor.
    pub dsp_cycle_time: AtomicU64,

    /// Número de amostras processadas no último ciclo (para cálculo de budget).
    pub last_n_samples: AtomicU32,

    /// Histograma de latência para análise estatística (P50, P95, P99).
    pub latency_hist: crate::dsp::telemetry::LatencyHistogram,

    /// Flag indicando que houve overflow no canal de GC.
    /// Se `true`, modelos ou resamplers obsoletos foram "vazados" (leaked)
    /// para evitar um `drop()` no hot-path.
    pub gc_overflow: AtomicBool,
}

impl RtStatusFlags {
    /// Cria uma nova instância com valores iniciais zerados/sentinela.
    pub fn new() -> Self {
        Self {
            active_rate: AtomicU32::new(0),
            active_rate_changed: AtomicU32::new(0),
            requested_pw_rate: AtomicU32::new(0),
            requested_nam_rate: AtomicU32::new(48_000),
            needs_resampler_rebuild: AtomicBool::new(false),
            resampler_rebuild_failed: AtomicBool::new(false),
            rt_is_fifo: AtomicBool::new(false),
            rt_priority: AtomicI32::new(-1),
            has_clipped: AtomicBool::new(false),
            dsp_overloads: AtomicU32::new(0),
            is_silent: AtomicBool::new(false),
            dsp_cycle_time: AtomicU64::new(0),
            last_n_samples: AtomicU32::new(0),
            latency_hist: crate::dsp::telemetry::LatencyHistogram::new(),
            gc_overflow: AtomicBool::new(false),
        }
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

/// Resultado da inicialização SPSC: canais de parâmetros, GC de modelos,
/// canal de resamplers RT-safe, e flags de status atômicas.
pub struct SpscChannels {
    /// Produtor de parâmetros CLI→DSP.
    pub param_producer: Producer<ParamPayload>,
    /// Consumidor de parâmetros CLI→DSP (movido para o callback RT).
    pub param_consumer: Consumer<ParamPayload>,
    /// Produtor GC: thread DSP envia modelos obsoletos para drop fora do RT.
    pub gc_producer: Producer<Box<crate::models::DynamicModel>>,
    /// Consumidor GC: thread background executa `drop()` dos modelos.
    pub gc_consumer: Consumer<Box<crate::models::DynamicModel>>,
    /// Produtor GC de resamplers: thread DSP envia resamplers obsoletos para drop fora do RT.
    pub gc_resampler_producer: Producer<crate::dsp::resampler::NamResampler>,
    /// Consumidor GC de resamplers: thread background executa `drop()`.
    pub gc_resampler_consumer: Consumer<crate::dsp::resampler::NamResampler>,
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
    let (gc_prod, gc_cons) = RingBuffer::new(capacity * 4); // Capacidade quadruplicada para garbage collection segura (L+R + folga)
    let (gc_rs_prod, gc_rs_cons) = RingBuffer::new(16); // GC de resamplers (raro)
    // Canal de resampler: capacidade pequena (apenas 1 em trânsito por vez, tipicamente)
    let (rs_prod, rs_cons) = RingBuffer::new(4);
    let rt_status = Arc::new(RtStatusFlags::new());

    SpscChannels {
        param_producer: param_prod,
        param_consumer: param_cons,
        gc_producer: gc_prod,
        gc_consumer: gc_cons,
        gc_resampler_producer: gc_rs_prod,
        gc_resampler_consumer: gc_rs_cons,
        resampler_producer: rs_prod,
        resampler_consumer: rs_cons,
        rt_status,
    }
}

#[cfg(test)]
#[path = "spsc_test.rs"]
mod spsc_test;
