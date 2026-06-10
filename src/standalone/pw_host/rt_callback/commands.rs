// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 5.1.2. COMMAND RECEPTION (SPSC Channel)
//! Processes commands from the command-line interface or control system (volume, model, noise gate).

use crate::common::spsc::{GcItem, GcOverflowBuffer, ParamPayload, RtStatusFlags};
use crate::dsp::gate::GateParams;

use std::sync::Arc;

/// 5.1.2. COMMAND RECEPTION (SPSC Channel)
/// Processes commands from the command-line interface or control system (volume, model, noise gate).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn receive_commands(
    consumer: &mut rtrb::Consumer<ParamPayload>,
    model_input_mult_adj: &mut f32,
    model_output_mult_adj: &mut f32,
    current_nam_rate: &mut u32,
    active_model_l: &mut Option<Box<crate::models::DynamicModel>>,
    active_model_r: &mut Option<Box<crate::models::DynamicModel>>,
    gc_producer: &mut rtrb::Producer<GcItem>,
    parking_lot: &mut [Option<GcItem>; 16],
    gc_overflow_for_process: &GcOverflowBuffer,
    rt_status_for_process: &Arc<RtStatusFlags>,
    user_input_gain_mult: &mut f32,
    user_output_gain_mult: &mut f32,
    gate_params: &mut GateParams,
    threshold_open_sq: &mut f32,
    threshold_close_sq: &mut f32,
    lut: &crate::math::dsp::gain_lut::GainLUT,
) -> bool {
    let mut param_changed = false;

    while let Ok(payload) = consumer.pop() {
        match payload {
            ParamPayload::LoadModel {
                model_l,
                model_r,
                input_mult_adj,
                output_mult_adj,
                sample_rate,
            } => {
                if model_l.is_some() || model_r.is_some() {
                    *model_input_mult_adj = input_mult_adj;
                    *model_output_mult_adj = output_mult_adj;
                    *current_nam_rate = sample_rate;
                } else {
                    *model_input_mult_adj = 1.0;
                    *model_output_mult_adj = 1.0;
                    *current_nam_rate = 48_000;
                }

                let mut old_models: [Option<Box<crate::models::DynamicModel>>; 2] = [None, None];
                if let Some(old) = std::mem::replace(active_model_l, model_l) {
                    old_models[0] = Some(old);
                }
                if let Some(model) = active_model_l {
                    model.inject_rt_status(Arc::clone(rt_status_for_process));
                }
                if let Some(old) = std::mem::replace(active_model_r, model_r) {
                    old_models[1] = Some(old);
                }
                if let Some(model) = active_model_r {
                    model.inject_rt_status(Arc::clone(rt_status_for_process));
                }

                for m_opt in &mut old_models {
                    if let Some(m) = m_opt.take() {
                        let mut item = Some(GcItem::Model(m));

                        if let Some(v) = item.take() {
                            if let Err(rtrb::PushError::Full(returned)) = gc_producer.push(v) {
                                item = Some(returned);
                            } else {
                                continue;
                            }
                        }

                        if let Some(v) = item.take() {
                            let mut parked = false;
                            let mut v_opt = Some(v);
                            for slot in parking_lot.iter_mut() {
                                if slot.is_none() {
                                    *slot = v_opt.take();
                                    parked = true;
                                    break;
                                }
                            }
                            if !parked {
                                item = v_opt;
                            }
                        }

                        if let Some(v) = item.take() {
                            rt_status_for_process
                                .set_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW);
                            gc_overflow_for_process.push(v);
                        }
                    }
                }
                param_changed = true;
            }
            ParamPayload::InputGain(mult) => {
                *user_input_gain_mult = mult;
                param_changed = true;
            }
            ParamPayload::OutputGain(mult) => {
                *user_output_gain_mult = mult;
                param_changed = true;
            }
            ParamPayload::GateConfig(params) => {
                let open_lin = lut.db_to_linear(params.threshold_open_db);
                let close_lin = lut.db_to_linear(params.threshold_close_db);
                *threshold_open_sq = open_lin * open_lin;
                *threshold_close_sq = close_lin * close_lin;
                *gate_params = params;
            }
        }
    }
    param_changed
}
