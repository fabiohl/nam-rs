// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pipeline DSP de captura completo — agrega todos os estágios.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::GateState;

use super::context::{DspBuffers, DspPipelineContext};
#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use super::stages::{
    apply_input_stage, apply_output_stage, handle_silence_bypass, run_inference, write_bridge,
};

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
    if ctx.bridge_writer.is_none() {
        return;
    }
    // ESTÁGIO 1: ENTRADA E LIMPEZA
    // Prepara o som e verifica se há silêncio para economizar energia.
    let gate_state = apply_input_stage(samples_l, samples_r, n_samples, &mut ctx);

    // GERENCIAMENTO DE ESTADO (SILÊNCIO vs SOM)
    match gate_state {
        GateState::Closed => {
            // Se o portão está fechado (silêncio total), avisamos o sistema para economizar CPU.
            handle_silence_bypass(ctx.bridge_writer, ctx.rt_status);
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
    write_bridge(
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        n_pw,
        ctx.bridge_writer,
    );
}
