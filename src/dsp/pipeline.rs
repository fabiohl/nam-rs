// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Pipeline de processamento DSP (Capture → DSP → Bridge).
//!
//! Este módulo isola a lógica de processamento de áudio da orquestração do PipeWire.
//! Ele contém o "hot-path" que é executado a cada ciclo de áudio na thread de tempo real.

#[cfg(feature = "standalone")]
use crate::diagnostics::SystemSnapshot;
#[cfg(any(feature = "standalone", test))]
use crate::dsp::gate::{DynamicHysteresis, GateParams, GateState};
#[cfg(any(feature = "standalone", test))]
use crate::dsp::resampler::NamResampler;
#[cfg(any(feature = "standalone", test))]
use crate::math::simd::{compute_energy_avx2, compute_max_diff_avx2};
#[cfg(any(feature = "standalone", test))]
use crate::models::{DynamicModel, NamModel};
#[cfg(any(feature = "standalone", test))]
use crate::spsc::RtStatusFlags;
#[cfg(feature = "standalone")]
use minstant::Anchor;
#[cfg(feature = "standalone")]
use pipewire as pw;
#[cfg(any(feature = "standalone", test))]
use std::sync::atomic::Ordering;

/// Estrutura de posse explícita para evitar o leak de memória
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

#[cfg(any(feature = "standalone", test))]
/// Tamanho máximo do buffer intermediário entre as duas streams (capture → playback).
/// Dimensionado para o quantum máximo do PipeWire (`max-quantum = 8192`).
pub(crate) const MAX_BRIDGE_BUF: usize = 8192;
#[cfg(any(feature = "standalone", test))]
/// Tamanho máximo do buffer para resampling.
pub(crate) const MAX_RESAMP_BUF: usize = 4096;

#[cfg(any(feature = "standalone", test))]
/// Buffer individual de áudio para o DspBridge (double-buffer).
#[repr(align(128))]
pub(crate) struct BridgeBuffer {
    /// Buffer de saída processada, canal esquerdo.
    pub buf_l: [f32; MAX_BRIDGE_BUF],
    /// Buffer de saída processada, canal direito.
    pub buf_r: [f32; MAX_BRIDGE_BUF],
    /// Número de amostras válidas no buffer atual.
    pub n_samples: u32,
}

#[cfg(any(feature = "standalone", test))]
/// Buffer compartilhado entre o callback de captura (DSP) e o callback de playback.
///
/// O capture callback escreve o resultado processado aqui com `fence(Release)`;
/// o playback callback lê com `fence(Acquire)`. A `generation` atômica permite
/// ao playback detectar se há dados novos disponíveis sem spin-lock.
///
/// Alinhado a 128 bytes para evitar false-sharing entre os dois callbacks RT.
#[repr(align(128))]
pub(crate) struct DspBridge {
    /// Os dois buffers físicos (front / back) para o double-buffering.
    pub buffers: [BridgeBuffer; 2],
    /// Índice do buffer ativo para LEITURA (0 ou 1). O capture sempre escreve no (1 - ativo).
    pub active_read_idx: std::sync::atomic::AtomicUsize,
    /// Contador de geração — incrementado a cada escrita pelo capture callback.
    /// O playback compara com sua cópia local para detectar novos dados.
    pub generation: std::sync::atomic::AtomicU64,
}

#[cfg(any(feature = "standalone", test))]
/// Contexto de dados para a pipeline DSP hot-path.
pub(crate) struct DspPipelineContext<'a> {
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
    /// Histerese de silêncio global.
    pub silence_hysteresis: &'a mut DynamicHysteresis,
    /// Histerese para detecção de sinal mono.
    pub mono_hysteresis: &'a mut DynamicHysteresis,
    /// Threshold de abertura do gate (quadrático).
    pub threshold_open_sq: f32,
    /// Threshold de fechamento do gate (quadrático).
    pub threshold_close_sq: f32,
    /// Flag indicando se o processamento deve ser mono.
    pub process_mono: &'a mut bool,
    /// Flags de status em tempo real.
    pub rt_status: &'a RtStatusFlags,
    /// Ponteiro para a ponte de memória compartilhada.
    pub bridge_ptr: *mut DspBridge,
    /// Buffer intermediário L (pós-resampler input).
    pub resamp_mid_l: &'a mut [f32],
    /// Buffer intermediário R (pós-resampler input).
    pub resamp_mid_r: &'a mut [f32],
    /// Buffer de saída L (pré-resampler output).
    pub resamp_out_l: &'a mut [f32],
    /// Buffer de saída R (pré-resampler output).
    pub resamp_out_r: &'a mut [f32],
}

/// Silence Bypass: sinaliza silêncio e zera o bridge para que o playback emita silêncio.
#[cfg(any(feature = "standalone", test))]
#[cold]
#[inline(never)]
pub(crate) fn handle_silence_bypass(bridge_ptr: *mut DspBridge, rt_status: &RtStatusFlags) {
    rt_status.set_flag(crate::spsc::RT_STATUS_IS_SILENT);

    let bridge_ref = unsafe { &mut *bridge_ptr };
    let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
    bridge_ref.buffers[back_idx].n_samples = 0;
    std::sync::atomic::fence(Ordering::Release);
    bridge_ref
        .active_read_idx
        .store(back_idx, Ordering::Relaxed);
    bridge_ref.generation.fetch_add(1, Ordering::Relaxed);
}

#[cfg(any(feature = "standalone", test))]
/// Estágio 1: Gate, Ganhos de Entrada e Detecção de Mono.
#[inline(always)]
fn apply_input_stage(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> GateState {
    // Usa o máximo das energias de ambos os canais: qualquer canal com sinal ativo
    // deve manter o gate aberto. Usando apenas L anteriormente, um sinal "Somente Direita"
    // (L=0, R≠0) era erroneamente classificado como silêncio → bug de mudo no canal R.
    let energy_l = unsafe { compute_energy_avx2(&samples_l[..n_samples]) };
    let energy_r = unsafe { compute_energy_avx2(&samples_r[..n_samples]) };
    let energy_ms = if energy_l >= energy_r {
        energy_l
    } else {
        energy_r
    };

    ctx.silence_hysteresis.update(
        energy_ms,
        ctx.threshold_open_sq,
        ctx.threshold_close_sq,
        ctx.gate_params,
        n_samples,
    );

    if ctx.silence_hysteresis.state() == GateState::Closed {
        return GateState::Closed;
    }

    let max_diff =
        unsafe { compute_max_diff_avx2(&samples_l[..n_samples], &samples_r[..n_samples]) };

    ctx.mono_hysteresis.update(
        max_diff,
        ctx.gate_params.mono_epsilon,
        ctx.gate_params.mono_epsilon * 0.9,
        ctx.gate_params,
        n_samples,
    );

    *ctx.process_mono = ctx.mono_hysteresis.state() == GateState::Closed
        || ctx.mono_hysteresis.state() == GateState::FadingOut;

    crate::dsp::gain::apply_gain_simd(&mut samples_l[..n_samples], ctx.input_gain_mult);
    if !*ctx.process_mono {
        crate::dsp::gain::apply_gain_simd(&mut samples_r[..n_samples], ctx.input_gain_mult);
    }

    ctx.silence_hysteresis.state()
}

#[cfg(any(feature = "standalone", test))]
/// Estágio 2: Inferência Neural e Resampling.
#[inline(always)]
fn run_inference(
    samples_l: &[f32],
    samples_r: &[f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> usize {
    let is_resamp_bypass = ctx.resampler.is_bypass();
    let n = n_samples.min(MAX_RESAMP_BUF);

    if is_resamp_bypass {
        let model_in_l = &samples_l[..n];
        let model_in_r = if *ctx.process_mono {
            &samples_l[..n]
        } else {
            &samples_r[..n]
        };
        let model_out_l = &mut ctx.resamp_out_l[..n];
        let model_out_r = &mut ctx.resamp_out_r[..n];

        if let Some(model_l) = ctx.active_model_l {
            model_l.process(model_in_l, model_out_l);
        } else {
            // True-Bypass: se não há modelo carregado, o sinal passa limpo (dry pass-through)
            model_out_l.copy_from_slice(model_in_l);
        }

        if *ctx.process_mono {
            // No modo mono, o canal direito é uma cópia do esquerdo processado
            model_out_r.copy_from_slice(model_out_l);
        } else if let Some(model_r) = ctx.active_model_r {
            model_r.process(model_in_r, model_out_r);
        } else {
            // True-Bypass para o canal R em modo estéreo
            model_out_r.copy_from_slice(model_in_r);
        }

        n
    } else {
        let n_48k = ctx.resampler.process_input(
            &samples_l[..n],
            if *ctx.process_mono {
                &samples_l[..n]
            } else {
                &samples_r[..n]
            },
            &mut ctx.resamp_mid_l[..MAX_RESAMP_BUF],
            &mut ctx.resamp_mid_r[..MAX_RESAMP_BUF],
        );

        let mut temp_out_l = [0.0f32; MAX_RESAMP_BUF];
        let mut temp_out_r = [0.0f32; MAX_RESAMP_BUF];

        let model_in_l = &ctx.resamp_mid_l[..n_48k];
        let model_in_r = &ctx.resamp_mid_r[..n_48k];
        let model_out_l = &mut temp_out_l[..n_48k];
        let model_out_r = &mut temp_out_r[..n_48k];

        if let Some(model_l) = ctx.active_model_l {
            model_l.process(model_in_l, model_out_l);
        } else {
            // True-Bypass: sinal resampleado passa limpo para o output temporário
            model_out_l.copy_from_slice(model_in_l);
        }

        if *ctx.process_mono {
            model_out_r.copy_from_slice(model_out_l);
        } else if let Some(model_r) = ctx.active_model_r {
            model_r.process(model_in_r, model_out_r);
        } else {
            // True-Bypass R
            model_out_r.copy_from_slice(model_in_r);
        }

        ctx.resampler.process_output(
            model_out_l,
            model_out_r,
            &mut ctx.resamp_out_l[..MAX_RESAMP_BUF],
            &mut ctx.resamp_out_r[..MAX_RESAMP_BUF],
        )
    }
}

#[cfg(any(feature = "standalone", test))]
/// Estágio 3: Ganho de Saída, Fading e Detecção de Clipping.
#[inline(always)]
fn apply_output_stage(
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    n_pw: usize,
    output_gain_mult: f32,
    gate_params: &GateParams,
    silence_hysteresis: &mut DynamicHysteresis,
    rt_status: &RtStatusFlags,
) {
    crate::dsp::gain::apply_gain_simd(&mut resamp_out_l[..n_pw], output_gain_mult);
    crate::dsp::gain::apply_gain_simd(&mut resamp_out_r[..n_pw], output_gain_mult);

    silence_hysteresis.apply_gain_rt(&mut resamp_out_l[..n_pw], gate_params, n_pw);
    silence_hysteresis.apply_gain_rt(&mut resamp_out_r[..n_pw], gate_params, n_pw);

    if crate::dsp::gain::detect_clipping_stereo_simd(&resamp_out_l[..n_pw], &resamp_out_r[..n_pw]) {
        rt_status.set_flag(crate::spsc::RT_STATUS_HAS_CLIPPED);
    }
}

#[cfg(any(feature = "standalone", test))]
/// Estágio 4: Escrita no DspBridge.
#[inline(always)]
fn write_bridge(
    resamp_out_l: &[f32],
    resamp_out_r: &[f32],
    n_pw: usize,
    bridge_ptr: *mut DspBridge,
) {
    let bridge_ref = unsafe { &mut *bridge_ptr };
    let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
    let back_buf = &mut bridge_ref.buffers[back_idx];

    let n_bridge = n_pw.min(MAX_BRIDGE_BUF);
    unsafe {
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

    std::sync::atomic::fence(Ordering::Release);
    bridge_ref
        .active_read_idx
        .store(back_idx, Ordering::Relaxed);
    bridge_ref.generation.fetch_add(1, Ordering::Relaxed);
}

#[cfg(any(feature = "standalone", test))]
/// Pipeline DSP Completo (Agregador).
#[inline(always)]
pub(crate) fn capture_dsp_pipeline(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    mut ctx: DspPipelineContext<'_>,
) {
    let gate_state = apply_input_stage(samples_l, samples_r, n_samples, &mut ctx);

    if gate_state == GateState::Closed {
        handle_silence_bypass(ctx.bridge_ptr, ctx.rt_status);
        return;
    }

    if gate_state != GateState::Open {
        ctx.rt_status.set_flag(crate::spsc::RT_STATUS_IS_SILENT);
    } else {
        ctx.rt_status.clear_flag(crate::spsc::RT_STATUS_IS_SILENT);
    }

    let n_pw = run_inference(samples_l, samples_r, n_samples, &mut ctx);

    apply_output_stage(
        ctx.resamp_out_l,
        ctx.resamp_out_r,
        n_pw,
        ctx.output_gain_mult,
        ctx.gate_params,
        ctx.silence_hysteresis,
        ctx.rt_status,
    );

    write_bridge(ctx.resamp_out_l, ctx.resamp_out_r, n_pw, ctx.bridge_ptr);
}

/// Pipeline DSP de Reprodução (Bridge → Hardware).
#[cfg(feature = "standalone")]
#[inline(always)]
pub(crate) fn playback_dsp_cycle(
    stream: &pw::stream::Stream,
    bridge_ptr: *mut DspBridge,
    last_bridge_gen: &mut u64,
) {
    let bridge_ref = unsafe { &*bridge_ptr };
    let current_gen = bridge_ref.generation.load(Ordering::Relaxed);
    if current_gen == *last_bridge_gen {
        return;
    }
    *last_bridge_gen = current_gen;
    std::sync::atomic::fence(Ordering::Acquire);

    let read_idx = bridge_ref.active_read_idx.load(Ordering::Relaxed);
    let front_buf = &bridge_ref.buffers[read_idx];

    let n_samples = front_buf.n_samples as usize;
    if n_samples == 0 || n_samples > MAX_BRIDGE_BUF {
        return;
    }

    let mut buf = match stream.dequeue_buffer() {
        Some(b) => b,
        None => return,
    };

    let datas = buf.datas_mut();
    if datas.len() < 2 {
        return;
    }

    let (datas_left, datas_right) = datas.split_at_mut(1);
    let data_l = &mut datas_left[0];
    let data_r = &mut datas_right[0];

    let max_l = data_l.as_raw().maxsize as usize / std::mem::size_of::<f32>();
    let max_r = data_r.as_raw().maxsize as usize / std::mem::size_of::<f32>();
    let n_out = n_samples.min(max_l).min(max_r);
    if n_out == 0 {
        return;
    }

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
        pw::spa::sys::spa_pod_builder_init(
            &mut builder,
            format_buf.as_mut_ptr().cast(),
            format_buf.len() as u32,
        );

        let pod_ptr = pw::spa::sys::spa_format_audio_raw_build(
            &mut builder,
            pw::spa::param::ParamType::EnumFormat.as_raw(),
            &audio_info.as_raw(),
        );

        if pod_ptr.is_null() {
            return Err(anyhow::anyhow!(
                "Falha ao construir SPA Pod de formato de áudio"
            ));
        }

        Ok(&*(pod_ptr as *const pw::spa::pod::Pod))
    }
}

#[cfg(test)]
#[path = "pipeline_test.rs"]
mod pipeline_test;
