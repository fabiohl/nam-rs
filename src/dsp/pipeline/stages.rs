// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Estágios individuais do pipeline DSP: gate/ganho de entrada,
//! inferência neural, ganho de saída/clipping e escrita no bridge.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::common::spsc::RtStatusFlags;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::{DynamicHysteresis, GateState};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::common::dispatch_simd;
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::math::dsp::stereo::{compute_energy_stereo, compute_max_diff};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::models::{DynamicModel, NamModel};

use super::bridge::{DspBridgeWriter, MAX_RESAMP_BUF};
use super::context::DspPipelineContext;

/// Silence Bypass: sinaliza silêncio e zera o bridge para que o playback emita silêncio.
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
#[cold]
#[inline(never)]
pub fn handle_silence_bypass(bridge: Option<DspBridgeWriter>, rt_status: &RtStatusFlags) {
    rt_status.set_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
    rt_status.clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);

    if let Some(writer) = bridge {
        writer.write_silence();
    }
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
    crate::math::dsp::gain::apply_gain_simd(&mut samples_l[..n_samples], ctx.input_gain_mult);

    // Só ajustamos o lado direito se o som NÃO for mono (para economizar processamento).
    if !*ctx.process_mono {
        crate::math::dsp::gain::apply_gain_simd(&mut samples_r[..n_samples], ctx.input_gain_mult);
    }

    ctx.silence_hysteresis.state()
}

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Helper unificado para processamento mono/stereo de modelos neurais.
///
/// Processa o modelo do canal L (_always_) e decide se o canal R é cópia mono
/// ou processamento independente via modelo R ativo.
#[inline(always)]
fn run_stereo_or_mono(
    active_model_l: &mut Option<Box<DynamicModel>>,
    active_model_r: &mut Option<Box<DynamicModel>>,
    model_in_l: &[f32],
    model_in_r: &[f32],
    m_out_l: &mut [f32],
    m_out_r: &mut [f32],
    process_mono: bool,
) {
    if let Some(model_l) = active_model_l {
        model_l.process(model_in_l, m_out_l);
    } else {
        m_out_l.copy_from_slice(model_in_l);
    }

    if process_mono {
        m_out_r.copy_from_slice(m_out_l);
    } else if let Some(model_r) = active_model_r {
        model_r.process(model_in_r, m_out_r);
    } else {
        m_out_r.copy_from_slice(model_in_r);
    }
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

        run_stereo_or_mono(
            ctx.active_model_l,
            ctx.active_model_r,
            model_in_l,
            model_in_r,
            m_out_l,
            m_out_r,
            *ctx.process_mono,
        );

        n
    } else {
        // CAMINHO B: Ajuste de Qualidade ligado (Resampler Ativo).

        // 1. Traduz o som para a frequência que o "Cérebro" neural entende (geralmente 48kHz).
        let n_48k = if *ctx.process_mono {
            ctx.resampler.process_input_mono(
                &samples_l[..n],
                &mut resamp_mid_l[..MAX_RESAMP_BUF],
                &mut resamp_mid_r[..MAX_RESAMP_BUF],
            )
        } else {
            ctx.resampler.process_input(
                &samples_l[..n],
                &samples_r[..n],
                &mut resamp_mid_l[..MAX_RESAMP_BUF],
                &mut resamp_mid_r[..MAX_RESAMP_BUF],
            )
        };

        let model_in_l = &resamp_mid_l[..n_48k];
        let model_in_r = &resamp_mid_r[..n_48k];
        let m_out_l = &mut model_out_l[..n_48k];
        let m_out_r = &mut model_out_r[..n_48k];

        // 2. Aplica a simulação do amplificador (Modelo Neural).
        run_stereo_or_mono(
            ctx.active_model_l,
            ctx.active_model_r,
            model_in_l,
            model_in_r,
            m_out_l,
            m_out_r,
            *ctx.process_mono,
        );

        // 3. Traduz o som de volta para a frequência original da sua placa de som.
        if *ctx.process_mono {
            ctx.resampler.process_output_mono(
                m_out_l,
                &mut resamp_out_l[..MAX_RESAMP_BUF],
                &mut resamp_out_r[..MAX_RESAMP_BUF],
            )
        } else {
            ctx.resampler.process_output(
                m_out_l,
                m_out_r,
                &mut resamp_out_l[..MAX_RESAMP_BUF],
                &mut resamp_out_r[..MAX_RESAMP_BUF],
            )
        }
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
pub fn write_bridge(
    resamp_out_l: &[f32],
    resamp_out_r: &[f32],
    n_pw: usize,
    bridge: Option<DspBridgeWriter>,
) {
    if let Some(writer) = bridge {
        writer.write_block(resamp_out_l, resamp_out_r, n_pw);
    }
}
