// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Full capture DSP pipeline — aggregates all stages.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::GateState;

use super::context::{DspBuffers, DspPipelineContext};
use super::stages::{
    apply_input_stage, apply_output_stage, handle_silence_bypass, run_inference, write_bridge,
};

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
/// Full DSP Pipeline (Aggregator).
#[inline(always)]
pub fn capture_dsp_pipeline(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    mut ctx: DspPipelineContext<'_>,
    bufs: DspBuffers<'_>,
    sample_rate: u32,
) {
    if ctx.bridge_writer.is_none() {
        return;
    }
    // STAGE 1: INPUT AND CLEANUP
    // Prepare the sound and check for silence to save energy.
    let gate_state = apply_input_stage(samples_l, samples_r, n_samples, &mut ctx);

    // STATE MANAGEMENT (SILENCE vs SOUND)
    match gate_state {
        GateState::Closed => {
            // If the gate is closed (total silence), notify the system to save CPU.
            handle_silence_bypass(ctx.bridge_writer, ctx.rt_status);
            return;
        }
        GateState::FadingIn | GateState::FadingOut => {
            // Indicates that the sound is smoothly appearing or vanishing.
            ctx.rt_status
                .clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            ctx.rt_status
                .set_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
        GateState::Open => {
            // Sound is passing normally at full volume.
            ctx.rt_status
                .clear_flag(crate::common::spsc::RT_STATUS_IS_SILENT);
            ctx.rt_status
                .clear_flag(crate::common::spsc::RT_STATUS_IS_FADING);
        }
    }

    // STAGE 2: THE "BRAIN" (AMP/PEDAL SIMULATION)
    // This is where the magic happens: the neural network simulates the desired timbre.
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

    // STAGE 3: FINAL ADJUSTMENT AND PROTECTION
    // Controls output volume and ensures the sound does not "blow up" (distortion).
    apply_output_stage(
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        n_pw,
        ctx.output_gain_mult,
        ctx.silence_hysteresis,
        ctx.rt_status,
        *ctx.process_mono,
        ctx.adaptive,
        sample_rate,
    );

    // STAGE 4: FINAL DELIVERY (THE BRIDGE)
    // Sends the processed result to your speakers via the bridge.
    write_bridge(
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        n_pw,
        ctx.bridge_writer,
    );
}
