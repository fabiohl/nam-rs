// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pipeline de processamento DSP (Capture → DSP → Bridge).
//!
//! Este módulo isola a lógica de processamento de áudio da orquestração do PipeWire.
//! Ele contém o "hot-path" que é executado a cada ciclo de áudio na thread de tempo real.

#[cfg(feature = "standalone")]
use crate::common::diagnostics::SystemSnapshot;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::{DynamicHysteresis, GateParams, GateState};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::resampler::NamResampler;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::common::dispatch_simd;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::dsp::stereo::{compute_energy_stereo, compute_max_diff};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::models::{DynamicModel, NamModel};

#[cfg(feature = "standalone")]
use minstant::Anchor;
#[cfg(feature = "standalone")]
use pipewire as pw;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use std::sync::atomic::Ordering;
/// das instâncias essenciais do PipeWire (`StreamBox` e `Listener`).
#[cfg(feature = "standalone")]
pub(crate) struct AppState<S1, L1, S2, L2> {
    #[allow(dead_code)]
    pub capture_stream: S1,
    #[allow(dead_code)]
    pub capture_listener: L1,
    #[allow(dead_code)]
    pub playback_stream: S2,
    #[allow(dead_code)]
    pub playback_listener: L2,
}

/// Configurações para inicialização do host PipeWire.
#[cfg(feature = "standalone")]
pub struct PipewireHostConfig {
    /// Tamanho do buffer de áudio solicitado.
    pub buffer_size: u32,
    /// Âncora de tempo para telemetria RDTSC.
    pub tsc_anchor: Anchor,
    /// Snapshot do sistema para diagnósticos.
    pub sys: SystemSnapshot,
}

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

    /// Dereferencia o ponteiro para acesso mutável.
    #[inline(always)]
    pub fn as_mut(self) -> &'static mut DspBridge {
        unsafe { &mut *self.0 }
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Contexto de dados para a pipeline DSP hot-path.
pub struct DspPipelineContext<'a> {
    /// Resampler ativo para conversão de sample rate.
    pub resampler: &'a mut NamResampler,
    /// Modelo ativo para o canal esquerdo.
    pub active_model_l: &'a mut Option<Box<DynamicModel>>,
    /// Modelo ativo para o canal direito.
    pub active_model_r: &'a mut Option<Box<DynamicModel>>,
    /// Multiplicador de ganho de entrada.
    pub input_gain_mult: f32,
    /// Multiplicador de ganho de saída.
    pub output_gain_mult: f32,
    /// Parâmetros do Noise Gate.
    pub gate_params: &'a GateParams,
    /// Histerese para detecção de silêncio.
    pub silence_hysteresis: &'a mut DynamicHysteresis,
    /// Histerese para detecção de sinal mono.
    pub mono_hysteresis: &'a mut DynamicHysteresis,
    /// Limiar de abertura (ao quadrado).
    pub threshold_open_sq: f32,
    /// Limiar de fechamento (ao quadrado).
    pub threshold_close_sq: f32,
    /// Flag indicando processamento em mono.
    pub process_mono: &'a mut bool,
    /// Flags de status RT.
    pub rt_status: &'a RtStatusFlags,
    /// Referência para a ponte de monitoração de áudio (opcional).
    pub bridge_ptr: BridgeRef,
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Conjunto de buffers de trabalho para o pipeline DSP.
/// Buffers de trabalho intermediários para o pipeline DSP.
pub struct DspBuffers<'a> {
    /// Buffer intermediário pós-resampler L.
    pub resamp_mid_l: &'a mut [f32],
    /// Buffer intermediário pós-resampler R.
    pub resamp_mid_r: &'a mut [f32],
    /// Buffer de saída do resampler L.
    pub resamp_out_l: &'a mut [f32],
    /// Buffer de saída do resampler R.
    pub resamp_out_r: &'a mut [f32],
    /// Buffer de saída do modelo L.
    pub model_out_l: &'a mut [f32],
    /// Buffer de saída do modelo R.
    pub model_out_r: &'a mut [f32],
}

/// Silence Bypass: sinaliza silêncio e zera o bridge para que o playback emita silêncio.
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[cold]
#[inline(never)]
pub fn handle_silence_bypass(bridge: BridgeRef, rt_status: &RtStatusFlags) {
    rt_status.set_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
    rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);

    let bridge_ref = bridge.as_mut();
    let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
    bridge_ref.buffers[back_idx].n_samples = 0;
    bridge_ref
        .active_read_idx
        .store(back_idx, Ordering::Release);
    // Detecção de Drop: se a geração atual ainda não foi consumida pelo playback,
    // a troca atual para silêncio irá 'pular' o bloco anterior.
    let current_gen = bridge_ref.generation.load(Ordering::Relaxed);
    let consumed_gen = bridge_ref.consumed_gen.load(Ordering::Relaxed);
    if current_gen > consumed_gen {
        let _ = bridge_ref
            .dropped_frames
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_add(1))
            });
    }

    bridge_ref
        .generation
        .store(current_gen + 1, Ordering::Release);
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Estágio 1: Gate, Ganhos de Entrada e Detecção de Mono.
#[inline(always)]
pub(crate) fn apply_input_stage(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> GateState {
    // Usa o máximo das energias de ambos os canais: qualquer canal com sinal ativo
    // deve manter o gate aberto. Usando o kernel fundido para reduzir o tráfego de cache.
    let energy_ms =
        unsafe { compute_energy_stereo(&samples_l[..n_samples], &samples_r[..n_samples]) };

    // 1. ATUALIZA O PORTÃO DE SILÊNCIO (NOISE GATE)
    // Decide se o som é forte o suficiente para passar ou se deve ser silenciado para economizar processamento.
    ctx.silence_hysteresis.update(
        energy_ms,
        ctx.threshold_open_sq,
        ctx.threshold_close_sq,
        ctx.gate_params,
        n_samples,
    );

    // Se o portão estiver totalmente fechado (silêncio absoluto), paramos por aqui para poupar bateria/CPU.
    if ctx.silence_hysteresis.state() == GateState::Closed {
        return GateState::Closed;
    }

    // 2. DETECÇÃO DE SOM MONO (IGUAL NOS DOIS LADOS)
    // Calcula a diferença entre o lado esquerdo e direito para ver se o som é o mesmo.
    let max_diff = unsafe { compute_max_diff(&samples_l[..n_samples], &samples_r[..n_samples]) };

    ctx.mono_hysteresis.update(
        max_diff,
        ctx.gate_params.mono_epsilon,
        ctx.gate_params.mono_epsilon * 0.9,
        ctx.gate_params,
        n_samples,
    );

    // Se o som for igual nos dois lados (mono), avisamos o sistema para processar apenas um lado.
    // Isso corta o trabalho pela metade sem perder qualidade!
    *ctx.process_mono = ctx.mono_hysteresis.state() == GateState::Closed
        || ctx.mono_hysteresis.state() == GateState::FadingOut;

    // 3. AJUSTE DE VOLUME DE ENTRADA (GAIN)
    // Aplica o ganho (volume) inicial definido pelo usuário.
    crate::dsp::gain::apply_gain_simd(&mut samples_l[..n_samples], ctx.input_gain_mult);

    // Só ajustamos o lado direito se o som NÃO for mono (para economizar processamento).
    if !*ctx.process_mono {
        crate::dsp::gain::apply_gain_simd(&mut samples_r[..n_samples], ctx.input_gain_mult);
    }

    ctx.silence_hysteresis.state()
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Estágio 2: Inferência Neural e Resampling.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_inference(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
    resamp_mid_l: &mut [f32],
    resamp_mid_r: &mut [f32],
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    model_out_l: &mut [f32],
    model_out_r: &mut [f32],
) -> usize {
    let is_resamp_bypass = ctx.resampler.is_bypass();
    let n = n_samples.min(MAX_RESAMP_BUF);

    // CAMINHO A: Ajuste de Qualidade desligado (Resampler em Bypass).
    if is_resamp_bypass {
        let model_in_l = &samples_l[..n];
        let model_in_r = if *ctx.process_mono {
            &samples_l[..n]
        } else {
            &samples_r[..n]
        };
        let m_out_l = &mut resamp_out_l[..n];
        let m_out_r = &mut resamp_out_r[..n];

        // Processa o som através do "Cérebro" (Modelo Neural) do lado esquerdo.
        if let Some(model_l) = ctx.active_model_l {
            model_l.process(model_in_l, m_out_l);
        } else {
            m_out_l.copy_from_slice(model_in_l);
        }

        // No modo mono, apenas copiamos o resultado do lado esquerdo para o direito.
        if *ctx.process_mono {
            m_out_r.copy_from_slice(m_out_l);
        } else if let Some(model_r) = ctx.active_model_r {
            // Se for estéreo, processamos o lado direito de forma independente.
            model_r.process(model_in_r, m_out_r);
        } else {
            // Caminho Limpo para o lado direito.
            m_out_r.copy_from_slice(model_in_r);
        }

        n
    } else {
        // CAMINHO B: Ajuste de Qualidade ligado (Resampler Ativo).

        // 1. Traduz o som para a frequência que o "Cérebro" neural entende (geralmente 48kHz).
        let n_48k = ctx.resampler.process_input(
            &samples_l[..n],
            if *ctx.process_mono {
                &samples_l[..n]
            } else {
                &samples_r[..n]
            },
            &mut resamp_mid_l[..MAX_RESAMP_BUF],
            &mut resamp_mid_r[..MAX_RESAMP_BUF],
        );

        let model_in_l = &resamp_mid_l[..n_48k];
        let model_in_r = &resamp_mid_r[..n_48k];
        let m_out_l = &mut model_out_l[..n_48k];
        let m_out_r = &mut model_out_r[..n_48k];

        // 2. Aplica a simulação do amplificador (Modelo Neural) lado esquerdo.
        if let Some(model_l) = ctx.active_model_l {
            model_l.process(model_in_l, m_out_l);
        } else {
            m_out_l.copy_from_slice(model_in_l);
        }

        // Processa o lado direito (Stereo ou cópia do Mono).
        if *ctx.process_mono {
            m_out_r.copy_from_slice(m_out_l);
        } else if let Some(model_r) = ctx.active_model_r {
            model_r.process(model_in_r, m_out_r);
        } else {
            m_out_r.copy_from_slice(model_in_r);
        }

        // 3. Traduz o som de volta para a frequência original da sua placa de som.
        ctx.resampler.process_output(
            m_out_l,
            m_out_r,
            &mut resamp_out_l[..MAX_RESAMP_BUF],
            &mut resamp_out_r[..MAX_RESAMP_BUF],
        )
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Estágio 3: Ganho de Saída, Fading e Detecção de Clipping.
#[inline(always)]
pub(crate) fn apply_output_stage(
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    n_pw: usize,
    output_gain_mult: f32,
    silence_hysteresis: &mut DynamicHysteresis,
    rt_status: &RtStatusFlags,
) {
    // 1. AJUSTE DE VOLUME FINAL E PROTEÇÃO CONTRA DISTORÇÃO (CLIPPING)
    // Aplica o volume de saída e verifica se o som não "estourou" o limite digital.
    let has_clipped = crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_stereo(
        &mut resamp_out_l[..n_pw],
        &mut resamp_out_r[..n_pw],
        output_gain_mult
    ));

    // 2. SUAVIZAÇÃO DO PORTÃO DE SILÊNCIO (FADING)
    // Aplica o fechamento ou abertura suave do som (fade) para evitar estalos ou cliques.
    dispatch_simd!(
        silence_hysteresis,
        apply_gain_rt_stereo,
        &mut resamp_out_l[..n_pw],
        &mut resamp_out_r[..n_pw],
        n_pw
    );

    // Se o som "estourou" o limite em qualquer momento, acendemos um aviso (flag) no sistema.
    if has_clipped {
        rt_status.set_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED);
    }
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Estágio 4: Escrita no DspBridge.
#[inline(always)]
pub fn write_bridge(resamp_out_l: &[f32], resamp_out_r: &[f32], n_pw: usize, bridge: BridgeRef) {
    let bridge_ref = bridge.as_mut();
    // Identifica qual "gaveta" da ponte (bridge) está vazia para podermos escrever o novo som.
    let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
    let back_buf = &mut bridge_ref.buffers[back_idx];

    let n_bridge = n_pw.min(MAX_BRIDGE_BUF);
    unsafe {
        // Copia o som processado para a gaveta vazia de forma ultra-rápida.
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
    }
    back_buf.n_samples = n_bridge as u32;
    // Avisa o sistema de som (PipeWire) que esta gaveta está pronta para ser "ouvida".
    bridge_ref
        .active_read_idx
        .store(back_idx, Ordering::Release);

    // DETECÇÃO DE ATRASO (DROP):
    // Se o sistema ainda não tocou o som anterior e já estamos entregando um novo,
    // significa que o computador atrasou. Contamos isso como um "pacote perdido".
    let current_gen = bridge_ref.generation.load(Ordering::Relaxed);
    let consumed_gen = bridge_ref.consumed_gen.load(Ordering::Relaxed);
    if current_gen > consumed_gen {
        let _ = bridge_ref
            .dropped_frames
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_add(1))
            });
    }

    // Atualiza o contador de entregas (Geração) para manter o sistema sincronizado.
    bridge_ref
        .generation
        .store(current_gen + 1, Ordering::Release);
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Pipeline DSP Completo (Agregador).
#[inline(always)]
pub fn capture_dsp_pipeline(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    mut ctx: DspPipelineContext<'_>,
    bufs: DspBuffers<'_>,
) {
    if ctx.bridge_ptr.is_null() {
        return;
    }
    // ESTÁGIO 1: ENTRADA E LIMPEZA
    // Prepara o som e verifica se há silêncio para economizar energia.
    let gate_state = apply_input_stage(samples_l, samples_r, n_samples, &mut ctx);

    // GERENCIAMENTO DE ESTADO (SILÊNCIO vs SOM)
    match gate_state {
        GateState::Closed => {
            // Se o portão está fechado (silêncio total), avisamos o sistema para economizar CPU.
            handle_silence_bypass(ctx.bridge_ptr, ctx.rt_status);
            return;
        }
        GateState::FadingIn | GateState::FadingOut => {
            // Indica que o som está surgindo ou sumindo de forma suave.
            ctx.rt_status
                .clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            ctx.rt_status
                .set_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
        GateState::Open => {
            // O som está passando normalmente com volume total.
            ctx.rt_status
                .clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            ctx.rt_status
                .clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
    }

    // ESTÁGIO 2: O "CÉREBRO" (SIMULAÇÃO DO AMP/PEDAL)
    // Aqui acontece a mágica: a rede neural simula o timbre desejado.
    let n_pw = run_inference(
        samples_l,
        samples_r,
        n_samples,
        &mut ctx,
        bufs.resamp_mid_l,
        bufs.resamp_mid_r,
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        bufs.model_out_l,
        bufs.model_out_r,
    );

    // ESTÁGIO 3: AJUSTE FINAL E PROTEÇÃO
    // Controla o volume de saída e garante que o som não "estoure" (distorção).
    apply_output_stage(
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        n_pw,
        ctx.output_gain_mult,
        ctx.silence_hysteresis,
        ctx.rt_status,
    );

    // ESTÁGIO 4: ENTREGA FINAL (A PONTE)
    // Envia o resultado processado para os seus alto-falantes através da ponte (bridge).
    write_bridge(bufs.resamp_out_l, bufs.resamp_out_r, n_pw, ctx.bridge_ptr);
}

/// Pipeline DSP de Reprodução (Bridge → Hardware).
#[cfg(feature = "standalone")]
#[inline(always)]
pub(crate) fn playback_dsp_cycle(
    stream: &pw::stream::Stream,
    bridge: BridgeRef,
    last_bridge_gen: &mut u64,
) {
    let bridge_ref = bridge.as_mut();
    // Verifica se há som novo na "ponte" (bridge). Se não houver nada novo, apenas esperamos.
    let current_gen = bridge_ref.generation.load(Ordering::Acquire);
    if current_gen == *last_bridge_gen {
        return;
    }
    *last_bridge_gen = current_gen;
    // Marca que este som já foi "consumido" e pode ser retirado da fila.
    bridge_ref
        .consumed_gen
        .store(current_gen, Ordering::Release);

    // Pega o som que está na "gaveta" ativa para ser tocado.
    let read_idx = bridge_ref.active_read_idx.load(Ordering::Relaxed);
    let front_buf = &bridge_ref.buffers[read_idx];

    let n_samples = front_buf.n_samples as usize;
    if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
        return;
    }

    // Pede ao sistema de som (PipeWire) um espaço vazio para colocar o áudio.
    let mut buf = match stream.dequeue_buffer() {
        Some(b) => b,
        None => return,
    };

    let datas = buf.datas_mut();
    if datas.len() < 2 {
        return;
    }

    // Separa os canais Esquerdo e Direito para a entrega final.
    let (datas_left, datas_right) = datas.split_at_mut(1);
    let data_l = &mut datas_left[0];
    let data_r = &mut datas_right[0];

    let max_l = data_l.as_raw().maxsize as usize / std::mem::size_of::<f32>();
    let max_r = data_r.as_raw().maxsize as usize / std::mem::size_of::<f32>();
    let n_out = n_samples.min(max_l).min(max_r);
    if n_out == 0 {
        return;
    }

    // Copia o som processado diretamente para as saídas da sua placa de som.
    if let Some(raw_l) = data_l.data() {
        let out_l =
            unsafe { std::slice::from_raw_parts_mut(raw_l.as_mut_ptr().cast::<f32>(), n_out) };
        out_l.copy_from_slice(&front_buf.buf_l[..n_out]);
    }
    if let Some(raw_r) = data_r.data() {
        let out_r =
            unsafe { std::slice::from_raw_parts_mut(raw_r.as_mut_ptr().cast::<f32>(), n_out) };
        out_r.copy_from_slice(&front_buf.buf_r[..n_out]);
    }

    // Informa ao hardware exatamente quanto de som foi entregue agora.
    {
        let chunk = data_l.chunk_mut();
        *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
        *chunk.offset_mut() = 0;
        *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
    }
    {
        let chunk = data_r.chunk_mut();
        *chunk.size_mut() = (n_out * std::mem::size_of::<f32>()) as u32;
        *chunk.offset_mut() = 0;
        *chunk.stride_mut() = std::mem::size_of::<f32>() as i32;
    }
}

/// Constrói um SPA Pod de formato de áudio F32P stereo para negociação PipeWire.
///
/// # Safety
/// O pod binário retornado aponta diretamente para o `format_buf` fornecido.
#[cfg(feature = "standalone")]
pub(crate) unsafe fn build_spa_format_pod<'a>(
    audio_info: &pw::spa::param::audio::AudioInfoRaw,
    format_buf: &'a mut [u8; 1024],
) -> anyhow::Result<&'a pw::spa::pod::Pod> {
    unsafe {
        let mut builder: pw::spa::sys::spa_pod_builder = std::mem::zeroed();
        // Prepara um "construtor" para criar o contrato de formato de áudio.
        pw::spa::sys::spa_pod_builder_init(
            &mut builder,
            format_buf.as_mut_ptr().cast(),
            format_buf.len() as u32,
        );

        // Constrói o documento binário (SPA Pod) que descreve o áudio (Ex: 48kHz, Estéreo).
        // Esse documento é o que o PipeWire usa para entender como enviar som para nós.
        let pod_ptr = pw::spa::sys::spa_format_audio_raw_build(
            &mut builder,
            pw::spa::param::ParamType::EnumFormat.as_raw(),
            &audio_info.as_raw(),
        );

        if pod_ptr.is_null() {
            // Se falhar, o sistema não saberá como negociar o som com a sua placa.
            return Err(anyhow::anyhow!(
                "Failed to build the audio negotiation document (SPA Pod)"
            ));
        }

        // Retorna o contrato pronto para ser assinado e usado pelo sistema.
        Ok(&*(pod_ptr as *const pw::spa::pod::Pod))
    }
}

#[cfg(test)]
pub(crate) mod test_util;

#[cfg(test)]
#[global_allocator]
static GLOBAL: test_util::infra::CountingAllocator = test_util::infra::CountingAllocator;

#[cfg(test)]
mod pipeline_test;

#[cfg(test)]
mod pipeline_block_test;
