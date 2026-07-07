// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(missing_docs)]

use crate::loader::nam_json::model::{NamLayerConfig, NamModelData};
use crate::models::a2::weights_layout::{
    FILM_KEYS, film_bias_count, film_bias_count_generic, film_weight_count,
    film_weight_count_generic,
};

use super::*;

const A2_HEAD_KERNEL: usize = 16;

#[derive(Clone)]
struct FiLMOracleSlot {
    shift: bool,
    groups: u32,
    weights: Vec<f64>,
    bias: Vec<f64>,
    buf: Vec<f64>,
}

impl FiLMOracleSlot {
    fn new(shift: bool, groups: u32, weights: Vec<f64>, bias: Vec<f64>, channels: usize) -> Self {
        let expected_bias = if shift { channels * 2 } else { channels };
        let mut padded_bias = bias;
        padded_bias.resize(expected_bias, 0.0);
        Self {
            shift,
            groups,
            weights,
            bias: padded_bias,
            buf: vec![0.0f64; channels * 2],
        }
    }

    fn apply(&mut self, input: &mut [f64], condition: &[f64]) {
        // cond_to_scale_shift uses constructed channels (weights/bias are
        // laid out for self.channels), but modulation only applies to
        // min(input.len(), self.channels) elements.
        let constructed_ch = self.buf.len() / 2;
        let g = self.groups as usize;
        let ch_per_group = constructed_ch / g;
        let cond_per_group = condition.len().checked_div(g).unwrap_or(0);
        let out_per_group = if self.shift {
            ch_per_group * 2
        } else {
            ch_per_group
        };

        self.buf.fill(0.0);
        let buf = &mut self.buf;

        for grp in 0..g {
            let cond_off = grp * cond_per_group;
            let row_off = grp * out_per_group;
            let w_off = row_off * cond_per_group;
            for row in 0..out_per_group {
                let global_out = if row < ch_per_group {
                    grp * ch_per_group + row
                } else {
                    constructed_ch + grp * ch_per_group + (row - ch_per_group)
                };
                let mut sum = self.bias[global_out];
                for k in 0..cond_per_group {
                    sum += self.weights[w_off + row * cond_per_group + k] * condition[cond_off + k];
                }
                buf[global_out] = sum;
            }
        }

        let apply_len = input.len().min(constructed_ch);
        for c in 0..apply_len {
            let scale = buf[c];
            let shift = if self.shift {
                buf[c + constructed_ch]
            } else {
                0.0
            };
            input[c] = input[c] * scale + shift;
        }
    }
}

// ── Architecture parameter extraction ─────────────────────────────────────

fn a2_read_topology(layer_cfg: &NamLayerConfig) -> Option<(Vec<usize>, Vec<usize>, usize, usize)> {
    let dil = layer_cfg.dilations.clone()?;
    let nlayers = dil.len();
    if nlayers == 0 {
        return None;
    }
    // kernel_sizes may be None when the model uses a single scalar kernel_size
    // applied to all layers (e.g. wavenet_a2_max.nam with kernel_size=4).
    let ks = if let Some(ks_vec) = layer_cfg.kernel_sizes.clone() {
        if ks_vec.len() != nlayers {
            return None;
        }
        ks_vec
    } else if let Some(ks_scalar) = layer_cfg.kernel_size {
        vec![ks_scalar; nlayers]
    } else {
        return None;
    };
    let bn = layer_cfg
        .bottleneck
        .unwrap_or(layer_cfg.channels.unwrap_or(8));
    Some((ks, dil, nlayers, bn))
}

fn a2_read_activation(raw: &serde_json::Value, li: usize, _num_layers: usize) -> ActivationConfig {
    let arr = raw.get("activation").and_then(|v| v.as_array());
    if let Some(arr) = arr
        && li < arr.len()
    {
        return ActivationConfig::from_json(&arr[li]);
    }
    // Fallback: single-object activation (e.g. {"type":"Softsign"})
    if let Some(obj) = raw.get("activation").and_then(|v| v.as_object()) {
        return ActivationConfig::from_json_obj(obj);
    }
    // Default: LeakyReLU(0.01)
    ActivationConfig::LeakyReLU {
        negative_slope: 0.01,
    }
}

fn a2_read_gating_mode(raw: &serde_json::Value, li: usize) -> GatingModeOracle {
    let arr = raw.get("gating_mode").and_then(|v| v.as_array());
    if let Some(arr) = arr
        && li < arr.len()
        && let Some(s) = arr[li].as_str()
    {
        return match s {
            "gated" => GatingModeOracle::Gated,
            "blended" => GatingModeOracle::Blended,
            _ => GatingModeOracle::None,
        };
    }
    GatingModeOracle::None
}

fn a2_read_head1x1_active(raw: &serde_json::Value) -> bool {
    raw.get("head1x1")
        .and_then(|v| v.get("active"))
        .and_then(|a| a.as_bool())
        .unwrap_or(false)
}

#[derive(Clone, Copy, PartialEq)]
enum GatingModeOracle {
    None,
    Gated,
    Blended,
}

#[derive(Clone)]
enum ActivationConfig {
    Tanh,
    HardTanh,
    FastTanh,
    ReLU,
    LeakyReLU { negative_slope: f64 },
    Sigmoid,
    SiLU,
    HardSwish,
    Softsign,
}

impl ActivationConfig {
    fn from_json(v: &serde_json::Value) -> Self {
        let obj = v.as_object();
        if let Some(obj) = obj {
            return Self::from_json_obj(obj);
        }
        if let Some(s) = v.as_str() {
            return match s {
                "Tanh" => Self::Tanh,
                "HardTanh" => Self::HardTanh,
                "FastTanh" => Self::FastTanh,
                "ReLU" => Self::ReLU,
                "Sigmoid" => Self::Sigmoid,
                "SiLU" => Self::SiLU,
                "HardSwish" => Self::HardSwish,
                "Softsign" => Self::Softsign,
                _ => Self::Tanh,
            };
        }
        Self::Tanh
    }

    fn from_json_obj(obj: &serde_json::Map<String, serde_json::Value>) -> Self {
        let t = obj.get("type").and_then(|v| v.as_str()).unwrap_or("Tanh");
        match t {
            "Tanh" => Self::Tanh,
            "HardTanh" => Self::HardTanh,
            "FastTanh" => Self::FastTanh,
            "ReLU" => Self::ReLU,
            "LeakyReLU" => {
                let slope = obj
                    .get("negative_slope")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.01);
                Self::LeakyReLU {
                    negative_slope: slope,
                }
            }
            "Sigmoid" => Self::Sigmoid,
            "SiLU" => Self::SiLU,
            "HardSwish" => Self::HardSwish,
            "Softsign" => Self::Softsign,
            _ => Self::Tanh,
        }
    }

    fn apply(&self, z: &mut [f64], activation_mode: ActivationMode) {
        match self {
            Self::Tanh => {
                for v in z.iter_mut() {
                    *v = oracle_tanh(*v, activation_mode);
                }
            }
            Self::HardTanh => {
                for v in z.iter_mut() {
                    *v = v.clamp(-1.0, 1.0);
                }
            }
            Self::FastTanh => {
                for v in z.iter_mut() {
                    *v = oracle_tanh(*v, activation_mode);
                }
            }
            Self::ReLU => {
                for v in z.iter_mut() {
                    *v = v.max(0.0);
                }
            }
            Self::LeakyReLU { negative_slope } => {
                let s = *negative_slope;
                for v in z.iter_mut() {
                    if *v < 0.0 {
                        *v *= s;
                    }
                }
            }
            Self::Sigmoid => {
                for v in z.iter_mut() {
                    *v = oracle_sigmoid(*v, activation_mode);
                }
            }
            Self::SiLU => {
                for v in z.iter_mut() {
                    let s = oracle_sigmoid(*v, activation_mode);
                    *v *= s;
                }
            }
            Self::HardSwish => {
                for v in z.iter_mut() {
                    let relu6 = (*v + 3.0).clamp(0.0, 6.0);
                    *v = *v * relu6 / 6.0;
                }
            }
            Self::Softsign => {
                for v in z.iter_mut() {
                    *v /= 1.0 + v.abs();
                }
            }
        }
    }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn oracle_a2_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let num_frames = input.len();
    if num_frames == 0 {
        return vec![];
    }

    let layers = &model_data.config.layers;
    if layers.is_empty() {
        return vec![0.0; num_frames];
    }

    // S14.2 (PM-15): Process condition_dsp sub-model to obtain per-frame condition
    // vectors. The sub-model processes the raw input and produces condition_size
    // samples per frame (the head_size of the condition_dsp's last array).
    let cond_output: Option<Vec<f64>> = model_data.config.condition_dsp.as_ref().map(|json| {
        let cond_model: NamModelData =
            serde_json::from_value(json.clone()).expect("Failed to parse condition_dsp JSON");
        oracle_forward(&cond_model, input, config)
    });

    let head_scale = model_data.config.head_scale.unwrap_or(1.0) as f64;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let acc_mode = config.accumulation;

    let num_arrays = layers.len();

    // Read all array configurations
    #[allow(clippy::type_complexity)]
    struct ArrayState {
        ch: usize,
        head_accum_size: usize,
        bottleneck: usize,
        cond_size: usize,
        head_size: usize,
        rechannel_w: Vec<f64>,
        lws: Vec<A2OracleLayerWeights>,
        head1x1_active: bool,
        h1_groups: usize,
        h1_in_size: usize,
        head1x1_w: Vec<f64>,
        head1x1_b: Vec<f64>,
        head_w: Vec<f64>,
        head_b: f64,
        fwd_bufs: Vec<Vec<f64>>,
    }

    let mut arrays: Vec<ArrayState> = Vec::with_capacity(num_arrays);

    for (ai, layer_cfg) in layers.iter().enumerate() {
        let ch = layer_cfg.channels.unwrap_or(8);
        let layer_raw = layer_cfg.layer_raw.clone();
        let cond_size = layer_cfg.condition_size.unwrap_or(1);

        let (kernel_sizes, dilations, num_layers, bottleneck) =
            a2_read_topology(layer_cfg).unwrap_or_else(|| (vec![], vec![], 0, ch));

        if num_layers == 0 {
            return vec![0.0; num_frames];
        }

        let head1x1_active = layer_raw
            .as_ref()
            .map(a2_read_head1x1_active)
            .unwrap_or(false);

        let pre_gating_modes: Vec<GatingModeOracle> = if let Some(ref raw) = layer_raw {
            (0..num_layers)
                .map(|li| a2_read_gating_mode(raw, li))
                .collect()
        } else {
            vec![GatingModeOracle::None; num_layers]
        };

        let pre_activations: Vec<ActivationConfig> = if let Some(ref raw) = layer_raw {
            (0..num_layers)
                .map(|li| a2_read_activation(raw, li, num_layers))
                .collect()
        } else {
            vec![
                ActivationConfig::LeakyReLU {
                    negative_slope: 0.01,
                };
                num_layers
            ]
        };

        let film_configs: [bool; 8] = if let Some(ref raw) = layer_raw {
            let mut active = [false; 8];
            for &(key, idx) in FILM_KEYS {
                let cfg = raw.get(key).and_then(|v| v.as_object());
                if let Some(obj) = cfg {
                    let a = obj.get("active").and_then(|a| a.as_bool()).unwrap_or(false);
                    if a {
                        active[idx] = true;
                    }
                }
            }
            active
        } else {
            [false; 8]
        };

        // Rechannel weights: 1×ch for array 0, prev_ch×ch for cascade.
        let in_ch: usize = if ai == 0 {
            1
        } else {
            layers[ai - 1].channels.unwrap_or(8)
        };
        let rechannel_w = cursor.read_f64(in_ch * ch);

        // Per-layer weights
        let mut lws: Vec<A2OracleLayerWeights> = Vec::new();
        for li in 0..num_layers {
            let ks = kernel_sizes[li];
            let dil = dilations[li];
            let gmode = pre_gating_modes[li];
            let use_gating = gmode == GatingModeOracle::Gated || gmode == GatingModeOracle::Blended;
            let conv_out = if use_gating {
                bottleneck * 2
            } else {
                bottleneck
            };

            let conv_w = cursor.read_f64(ch * conv_out * ks);
            let conv_b = cursor.read_f64(conv_out);
            let mixin_w = cursor.read_f64(conv_out * cond_size);
            let l1x1_w = cursor.read_f64(bottleneck * ch);
            let l1x1_b = cursor.read_f64(ch);

            let mut film_slots: Vec<Option<FiLMOracleSlot>> = vec![None; 8];
            for slot_idx in 0..8 {
                if !film_configs[slot_idx] {
                    continue;
                }
                let g = layer_raw
                    .as_ref()
                    .and_then(|raw| {
                        let key = FILM_KEYS.iter().find(|(_, idx)| *idx == slot_idx)?.0;
                        raw.get(key)
                    })
                    .and_then(|v| v.get("groups"))
                    .and_then(|g| g.as_u64())
                    .unwrap_or(1) as u32;
                let shift = layer_raw
                    .as_ref()
                    .and_then(|raw| {
                        let key = FILM_KEYS.iter().find(|(_, idx)| *idx == slot_idx)?.0;
                        raw.get(key)
                    })
                    .and_then(|v| v.get("shift"))
                    .and_then(|s| s.as_bool())
                    .unwrap_or(true);

                let (w_count, b_count) = if cond_size > 1 {
                    (
                        film_weight_count_generic(g, cond_size, ch, shift),
                        film_bias_count_generic(ch),
                    )
                } else {
                    (
                        film_weight_count(g, cond_size, ch, shift),
                        film_bias_count(ch, shift),
                    )
                };
                let weights = cursor.read_f64(w_count);
                let bias = cursor.read_f64(b_count);
                film_slots[slot_idx] = Some(FiLMOracleSlot::new(shift, g, weights, bias, ch));
            }

            lws.push(A2OracleLayerWeights {
                conv_w,
                conv_b,
                mixin_w,
                l1x1_w,
                l1x1_b,
                ks,
                dil,
                film: film_slots,
                gating_mode: gmode,
                activation: pre_activations[li].clone(),
                conv_out,
            });
        }

        // Head1x1 weights
        let h1_groups = layer_raw
            .as_ref()
            .and_then(|raw| raw.get("head1x1"))
            .and_then(|h| h.get("groups"))
            .and_then(|g| g.as_u64())
            .unwrap_or(1) as usize;
        let h1_in_size = if head1x1_active {
            bottleneck / h1_groups
        } else {
            0
        };
        let head_accum_size = if head1x1_active {
            layer_raw
                .as_ref()
                .and_then(|raw| raw.get("head1x1"))
                .and_then(|h| h.get("out_channels"))
                .and_then(|a| a.as_u64())
                .unwrap_or(bottleneck as u64) as usize
        } else {
            bottleneck
        };
        let head1x1_w: Vec<f64> = if head1x1_active {
            cursor.read_f64(ch * h1_in_size)
        } else {
            vec![]
        };
        let head1x1_b: Vec<f64> = if head1x1_active {
            cursor.read_f64(ch)
        } else {
            vec![]
        };

        let head_size = layer_cfg.head_size.unwrap_or(1);
        let (head_w, head_b) = if head_size == 1 {
            let head_w_raw = cursor.read_f64(A2_HEAD_KERNEL * head_accum_size);
            let mut head_w = vec![0.0f64; A2_HEAD_KERNEL * head_accum_size];
            for tap in 0..A2_HEAD_KERNEL {
                for c in 0..head_accum_size {
                    head_w[tap * head_accum_size + c] = head_w_raw[c * A2_HEAD_KERNEL + tap];
                }
            }
            let head_b = cursor.read_one_f64();
            let _head_scale_val = cursor.read_one_f64();
            (head_w, head_b)
        } else {
            let hw_count = head_accum_size * head_size;
            let head_w = cursor.read_f64(hw_count);
            (head_w, 0.0f64)
        };

        // Pre-compute per-array max RF for buffer sizing
        let max_dil: usize = *dilations.iter().max().unwrap_or(&1);
        let max_ks: usize = *kernel_sizes.iter().max().unwrap_or(&6);
        let _max_rf = (max_ks - 1) * max_dil + 64;

        arrays.push(ArrayState {
            ch,
            head_accum_size,
            bottleneck,
            cond_size,
            head_size,
            rechannel_w,
            lws,
            head1x1_active,
            h1_groups,
            h1_in_size,
            head1x1_w,
            head1x1_b,
            head_w,
            head_b,
            fwd_bufs: vec![],
        });
    }

    // Allocate history buffers per array (largest across arrays).
    let mut max_rf: usize = 0;
    for arr in &arrays {
        let max_dil = arr.lws.iter().map(|lw| lw.dil).max().unwrap_or(1);
        let max_ks_a = arr.lws.iter().map(|lw| lw.ks).max().unwrap_or(6);
        max_rf = max_rf.max((max_ks_a - 1) * max_dil + 64);
    }
    let hist_size = max_rf + num_frames + 64;
    let bs = max_rf;

    for arr in &mut arrays {
        let num_layers = arr.lws.len();
        let ch = arr.ch;
        arr.fwd_bufs = (0..num_layers)
            .map(|_| vec![0.0f64; hist_size * ch])
            .collect();
    }

    // Head accumulator (shared across arrays, per-channel).
    let hr_len = (max_rf + num_frames + 64).next_power_of_two();
    let ring_mask = hr_len - 1;
    let max_ch = arrays.iter().map(|a| a.ch).max().unwrap_or(8);
    let mut head_acc = vec![0.0f64; hr_len * max_ch];
    let mut head_wp = 0usize;

    // Pre-compute channel counts for cascade residual flow.
    let array_channels: Vec<usize> = arrays.iter().map(|a| a.ch).collect();

    // Reserve cascade residual buffer (multi-channel between arrays).
    let mut cascade_residual = vec![0.0f64; hist_size * max_ch];

    let mut output = vec![0.0f64; num_frames];

    #[allow(clippy::explicit_counter_loop)]
    for (f, out_val) in output.iter_mut().enumerate() {
        let fi = bs + f;
        let x = input[f];
        let head_col = head_wp;
        head_wp += 1;

        // ── Cascade: process each array ──
        for (ai, arr) in arrays.iter_mut().enumerate() {
            let ch = arr.ch;
            let bottleneck = arr.bottleneck;
            let cond_size = arr.cond_size;

            // Condition vector: from condition_dsp or raw input.
            let condition: &[f64] = if cond_size == 1 {
                // S14.2 (PM-15): For cascade arrays after the first, the
                // condition may be per-frame from the cascade residual — but
                // for condition_size==1 we use the raw audio directly.
                std::slice::from_ref(&x)
            } else if let Some(ref cond_out) = cond_output {
                // condition_dsp output: cond_out[f * cond_size..(f+1)*cond_size]
                let off = f * cond_size;
                if off + cond_size <= cond_out.len() {
                    &cond_out[off..off + cond_size]
                } else {
                    &[]
                }
            } else {
                // No condition_dsp: zero condition (will produce zero FiLM/mixin).
                &[]
            };

            // Per-array history buffers.
            let num_layers = arr.lws.len();
            let mut head1x1_scratch = if arr.head1x1_active {
                vec![0.0f64; arr.head_accum_size]
            } else {
                vec![]
            };
            let mut z_scratch = vec![0.0f64; bottleneck * 2];

            // Input to this array: mono for array 0, cascade residual for others.
            let mut layer_in = vec![0.0f64; ch];
            if ai == 0 {
                for c in 0..ch {
                    layer_in[c] = x * arr.rechannel_w[c];
                }
            } else {
                // Rechannel from previous array's residual (saved per-frame).
                let prev_ch = array_channels[ai - 1];
                let rw = &arr.rechannel_w;
                for nc in 0..ch {
                    let mut sum = 0.0;
                    for ic in 0..prev_ch {
                        sum += cascade_residual[fi * max_ch + ic] * rw[ic * ch + nc];
                    }
                    layer_in[nc] = sum;
                }
            }

            // Per-layer history buffers
            let fwd_bufs = &mut arr.fwd_bufs;

            // Write input to first layer's history
            for c in 0..ch {
                fwd_bufs[0][fi * ch + c] = layer_in[c];
            }

            for (li, lw) in arr.lws.iter_mut().enumerate() {
                let z_out_ch = lw.conv_out;
                let use_gating = lw.gating_mode == GatingModeOracle::Gated;
                let use_blending = lw.gating_mode == GatingModeOracle::Blended;

                // conv_pre_film (slot 0)
                if let Some(ref mut film) = lw.film[0] {
                    film.apply(&mut fwd_bufs[li][fi * ch..fi * ch + ch], condition);
                }

                // Conv1d
                z_scratch.fill(0.0);
                for oc in 0..z_out_ch {
                    let mut sum = lw.conv_b[oc];
                    let wb = oc * ch * lw.ks;
                    for kt in 0..lw.ks {
                        let off = (lw.dil as isize) * ((kt as isize) + 1 - (lw.ks as isize));
                        let ins = ((fi as isize) + off) as usize * ch;
                        for ic in 0..ch {
                            if ins + ic < fwd_bufs[li].len() {
                                sum = mul_add_f64(
                                    fwd_bufs[li][ins + ic],
                                    lw.conv_w[wb + ic * lw.ks + kt],
                                    sum,
                                    acc_mode,
                                );
                            }
                        }
                    }
                    z_scratch[oc] = sum;
                }

                // conv_post_film (slot 1) + input_mixin_pre_film (slot 2)
                if let Some(ref mut film) = lw.film[1] {
                    film.apply(&mut z_scratch[..z_out_ch], condition);
                }
                if let Some(ref mut film) = lw.film[2] {
                    film.apply(&mut z_scratch[..z_out_ch], condition);
                }

                // Mixin
                if !condition.is_empty() {
                    for c in 0..z_out_ch.min(bottleneck) {
                        let mut sum = 0.0;
                        for k in 0..cond_size.min(condition.len()) {
                            sum += lw.mixin_w[c * cond_size + k] * condition[k];
                        }
                        z_scratch[c] += sum;
                    }
                }

                // input_mixin_post_film (slot 3) + activation_pre_film (slot 4)
                if let Some(ref mut film) = lw.film[3] {
                    film.apply(&mut z_scratch[..z_out_ch], condition);
                }
                if let Some(ref mut film) = lw.film[4] {
                    film.apply(&mut z_scratch[..z_out_ch], condition);
                }

                // Activation or Gating/Blending
                let z_len = if use_gating {
                    let half = bottleneck;
                    lw.activation
                        .apply(&mut z_scratch[..half], config.activation);
                    for i in 0..half {
                        let gate = exact_sigmoid_f64(z_scratch[half + i]);
                        z_scratch[i] *= gate;
                    }
                    half
                } else if use_blending {
                    let half = bottleneck;
                    lw.activation
                        .apply(&mut z_scratch[..half], config.activation);
                    for i in 0..half {
                        let alpha = exact_sigmoid_f64(z_scratch[half + i]);
                        z_scratch[i] = alpha * z_scratch[i] + (1.0 - alpha) * z_scratch[half + i];
                    }
                    half
                } else {
                    lw.activation
                        .apply(&mut z_scratch[..bottleneck], config.activation);
                    bottleneck
                };

                // activation_post_film (slot 5)
                if let Some(ref mut film) = lw.film[5] {
                    film.apply(&mut z_scratch[..z_len], condition);
                }

                // Head accumulate
                let head_off = head_col * max_ch;
                if arr.head1x1_active {
                    let h1_groups = arr.h1_groups;
                    let h1_in_size = arr.h1_in_size;
                    let ch_per_group = ch / h1_groups;
                    head1x1_scratch.fill(0.0);
                    for grp in 0..h1_groups {
                        for oc in grp * ch_per_group..(grp + 1) * ch_per_group {
                            let mut sum = arr.head1x1_b[oc];
                            for ic in 0..h1_in_size {
                                sum = mul_add_f64(
                                    z_scratch[grp * h1_in_size + ic],
                                    arr.head1x1_w[oc * h1_in_size + ic],
                                    sum,
                                    acc_mode,
                                );
                            }
                            head1x1_scratch[oc] = sum;
                        }
                    }
                    if let Some(ref mut film) = lw.film[7] {
                        film.apply(&mut head1x1_scratch, condition);
                    }
                    if li == 0 && ai == 0 {
                        head_acc[head_off..head_off + arr.head_accum_size]
                            .copy_from_slice(&head1x1_scratch[..arr.head_accum_size]);
                    } else {
                        for c in 0..arr.head_accum_size {
                            head_acc[head_off + c] =
                                accum_f64(head_acc[head_off + c], head1x1_scratch[c], acc_mode);
                        }
                    }
                } else {
                    if li == 0 && ai == 0 {
                        head_acc[head_off..head_off + z_len].copy_from_slice(&z_scratch[..z_len]);
                    } else {
                        for c in 0..z_len {
                            head_acc[head_off + c] =
                                accum_f64(head_acc[head_off + c], z_scratch[c], acc_mode);
                        }
                    }
                }

                // L1x1 residual
                if li < num_layers - 1 {
                    let mut next = vec![0.0f64; ch];
                    for oc in 0..ch {
                        let mut sum = lw.l1x1_b[oc];
                        for ic in 0..bottleneck {
                            sum = mul_add_f64(
                                z_scratch[ic],
                                lw.l1x1_w[oc * bottleneck + ic],
                                sum,
                                acc_mode,
                            );
                        }
                        next[oc] = accum_f64(layer_in[oc], sum, acc_mode);
                    }
                    if let Some(ref mut film) = lw.film[6] {
                        film.apply(&mut next, condition);
                    }
                    for c in 0..ch {
                        fwd_bufs[li + 1][fi * ch + c] = next[c];
                    }
                    layer_in = next;
                }
            }

            // Save residual for next array (cascade_input reads from cascade_residual).
            if ai + 1 < num_arrays {
                for c in 0..ch {
                    cascade_residual[fi * max_ch + c] = layer_in[c];
                }
            }
        }

        // ── Head finalize (last array only) ──
        let last_arr = &arrays[num_arrays - 1];
        let lch = last_arr.head_accum_size;
        let k = if last_arr.head_size == 1 {
            A2_HEAD_KERNEL
        } else {
            last_arr.head_size
        };
        let cb = head_col.wrapping_sub(k - 1);
        let mut y = last_arr.head_b;
        for t in 0..k {
            let col = cb.wrapping_add(t) & ring_mask;
            let so = col * max_ch;
            let wo = t * lch;
            for c in 0..last_arr.head_accum_size {
                y = mul_add_f64(last_arr.head_w[wo + c], head_acc[so + c], y, acc_mode);
            }
        }
        *out_val = y * head_scale;
    }

    output
}

// ── A2 Oracle Layer Weights (used by oracle_a2_forward) ─────────────────

struct A2OracleLayerWeights {
    conv_w: Vec<f64>,
    conv_b: Vec<f64>,
    mixin_w: Vec<f64>,
    l1x1_w: Vec<f64>,
    l1x1_b: Vec<f64>,
    ks: usize,
    dil: usize,
    film: Vec<Option<FiLMOracleSlot>>,
    gating_mode: GatingModeOracle,
    activation: ActivationConfig,
    conv_out: usize,
}

// =============================================================================
// LSTM Oracle
// =============================================================================
