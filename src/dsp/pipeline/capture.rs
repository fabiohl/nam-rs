// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Full capture DSP pipeline — aggregates all stages.

#[cfg(any(feature = "standalone", feature = "clap-plugin", test))]
use crate::dsp::gate::GateState;

use super::context::{DspBuffers, DspPipelineContext};
use super::stages::{apply_input_stage, apply_output_stage, run_inference, write_bridge};

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
    // Report gate state to real-time status flags via the canonical function.
    crate::dsp::gate_flags::report_gate_flags(ctx.rt_status, gate_state);

    if gate_state == GateState::Closed {
        // If the gate is closed (total silence), zero the bridge to save CPU.
        if let Some(writer) = ctx.bridge_writer {
            writer.write_silence();
        }
        return;
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

    // STAGE 3: CAB-SIM (OPTIONAL IR CONVOLUTION)
    // Post-inference speaker/cabinet simulation via impulse response.
    // When no IR is loaded (conv == None), this stage is a single branch — zero cost.
    if let Some(ref mut conv) = ctx.conv {
        let partition = conv.partition_size();
        if n_pw == partition {
            conv.process(&bufs.resamp_out_l[..n_pw], &mut bufs.model_out_l[..n_pw]);
            bufs.resamp_out_l[..n_pw].copy_from_slice(&bufs.model_out_l[..n_pw]);
            if !*ctx.process_mono {
                conv.process(&bufs.resamp_out_r[..n_pw], &mut bufs.model_out_r[..n_pw]);
                bufs.resamp_out_r[..n_pw].copy_from_slice(&bufs.model_out_r[..n_pw]);
            } else {
                bufs.resamp_out_r[..n_pw].copy_from_slice(&bufs.resamp_out_l[..n_pw]);
            }
        }
    }

    // STAGE 4: FINAL ADJUSTMENT AND PROTECTION
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

    // STAGE 5: FINAL DELIVERY (THE BRIDGE)
    // Sends the processed result to your speakers via the bridge.
    write_bridge(
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        n_pw,
        ctx.bridge_writer,
    );
}
