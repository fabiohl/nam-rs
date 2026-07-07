// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Reference oracle in f64 — absolute ground truth for the NAM mathematical model.
#![allow(missing_docs)]
//!
//! Computes the **ideal forward pass** of WaveNet, LSTM, and A2 topologies
//! using double-precision (f64) arithmetic, exact activation functions
//! (`f64::tanh`, `f64::exp`), and Kahan/Neumaier compensated accumulation.
//!
//! # Purpose
//! The production path (f32 + Padé tanh + minimax sigmoid + FMA accumulation)
//! shares the same limitations as the C++ NAMCore (also f32). The oracle
//! provides an **independent** high-precision reference that:
//! - Measures the **absolute error floor** of the f32 production path.
//! - Permits **source decomposition** — isolating the contribution of each
//!   approximation (weight quantization, activation, accumulation) to total error.

use crate::loader::nam_json::WeightsLayout;
use crate::loader::nam_json::model::{NamLayerConfig, NamModelData};
use crate::models::a2::weights_layout::{
    FILM_KEYS, film_bias_count, film_bias_count_generic, film_weight_count,
    film_weight_count_generic,
};

// =============================================================================
// Precision Configuration
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightPrecision {
    /// Full f64 precision (weights converted from f32).
    F64Exact,
    /// Weights quantized to f16c (binary16) then converted to f64.
    F16C,
    /// Weights quantized to bf16 then converted to f64.
    BF16,
    /// Weights kept in f32 precision (cast to f64 at compute time).
    F32,
}

/// Activation function precision mode for source decomposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationMode {
    /// Exact: `f64::tanh`, `f64` exp-based sigmoid.
    Exact,
    /// Approximate: Padé [5,4] tanh, minimax degree-17 sigmoid.
    PadeMinimax,
}

/// Accumulation mode for dot products and residual sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccumulationMode {
    /// Plain f64 accumulation (no compensation).
    F64Plain,
    /// Kahan compensated summation.
    Kahan,
    /// Neumaier compensated summation.
    Neumaier,
    /// Accumulation in f32 to simulate f32 error.
    F32Plain,
}

/// Precision configuration for one oracle run.
#[derive(Clone, Copy, Debug)]
pub struct PrecisionConfig {
    /// Weight representation precision.
    pub weight_precision: WeightPrecision,
    /// Activation function approximation mode.
    pub activation: ActivationMode,
    /// Accumulation algorithm.
    pub accumulation: AccumulationMode,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            weight_precision: WeightPrecision::F64Exact,
            activation: ActivationMode::Exact,
            accumulation: AccumulationMode::Neumaier,
        }
    }
}

// =============================================================================
// f64 Activation Functions
// =============================================================================

/// Padé [5,4] rational approximant for tanh in f64.
#[inline]
pub fn pade_tanh_f64(x: f64) -> f64 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let num = x * (x2 + 105.0).mul_add(x2, 945.0);
    let den = (15.0f64.mul_add(x2, 420.0)).mul_add(x2, 945.0);
    (num / den).clamp(-1.0, 1.0)
}

#[inline]
pub fn exact_tanh_f64(x: f64) -> f64 {
    x.tanh()
}

#[inline]
pub fn oracle_tanh(x: f64, mode: ActivationMode) -> f64 {
    match mode {
        ActivationMode::Exact => exact_tanh_f64(x),
        ActivationMode::PadeMinimax => pade_tanh_f64(x),
    }
}

#[inline]
pub fn exact_sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn minimax_sigmoid_f64(x: f64) -> f64 {
    let c0 = 2.4885319190e-01_f64;
    let c1 = -1.9318685012e-02_f64;
    let c2 = 1.4623214305e-03_f64;
    let c3 = -7.9953400187e-05_f64;
    let c4 = 2.9140652422e-06_f64;
    let c5 = -6.8000246432e-08_f64;
    let c6 = 9.6897239158e-10_f64;
    let c7 = -7.6498626314e-12_f64;
    let c8 = 2.5585471676e-14_f64;
    let xc = x.clamp(-8.0, 8.0);
    let x2 = xc * xc;
    let p = c8.mul_add(x2, c7);
    let p = p.mul_add(x2, c6);
    let p = p.mul_add(x2, c5);
    let p = p.mul_add(x2, c4);
    let p = p.mul_add(x2, c3);
    let p = p.mul_add(x2, c2);
    let p = p.mul_add(x2, c1);
    let p = p.mul_add(x2, c0);
    (xc.mul_add(p, 0.5)).clamp(0.0, 1.0)
}

#[inline]
pub fn oracle_sigmoid(x: f64, mode: ActivationMode) -> f64 {
    match mode {
        ActivationMode::Exact => exact_sigmoid_f64(x),
        ActivationMode::PadeMinimax => minimax_sigmoid_f64(x),
    }
}

// =============================================================================
// Weight Conversion
// =============================================================================

#[inline]
fn weight_f32_to_f64(w: f32, mode: WeightPrecision) -> f64 {
    match mode {
        WeightPrecision::F64Exact | WeightPrecision::F32 => w as f64,
        WeightPrecision::F16C => {
            let bits = crate::math::common::half::f32_to_f16_bits(w);
            crate::math::common::half::f16_bits_to_f32(bits) as f64
        }
        WeightPrecision::BF16 => {
            let bits = w.to_bits();
            let bf16_bits = (bits >> 16) as u16;
            f32::from_bits((bf16_bits as u32) << 16) as f64
        }
    }
}

/// Simple weight stream cursor with optional weight-precision quantization.
struct Cursor<'a> {
    data: &'a [f32],
    pos: usize,
    weight_mode: WeightPrecision,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [f32], weight_mode: WeightPrecision) -> Self {
        Self {
            data,
            pos: 0,
            weight_mode,
        }
    }

    fn read_f64(&mut self, count: usize) -> Vec<f64> {
        let end = self.pos + count;
        let out: Vec<f64> = self.data[self.pos..end]
            .iter()
            .map(|&x| weight_f32_to_f64(x, self.weight_mode))
            .collect();
        self.pos = end;
        out
    }

    fn read_one_f64(&mut self) -> f64 {
        let v = weight_f32_to_f64(self.data[self.pos], self.weight_mode);
        self.pos += 1;
        v
    }
}

/// Helper for accumulation mode: optionally casts through f32 to simulate f32 error.
#[inline]
fn accum_f64(a: f64, b: f64, mode: AccumulationMode) -> f64 {
    match mode {
        AccumulationMode::F64Plain | AccumulationMode::Kahan | AccumulationMode::Neumaier => a + b,
        AccumulationMode::F32Plain => {
            let s = (a as f32) + (b as f32);
            s as f64
        }
    }
}

/// Multiply-add with optional f32 accumulation.
#[inline]
fn mul_add_f64(a: f64, b: f64, c: f64, mode: AccumulationMode) -> f64 {
    match mode {
        AccumulationMode::F64Plain | AccumulationMode::Kahan | AccumulationMode::Neumaier => {
            a.mul_add(b, c)
        }
        AccumulationMode::F32Plain => {
            let p = (a as f32) * (b as f32);
            let s = p + (c as f32);
            s as f64
        }
    }
}

// =============================================================================
// ESR
// =============================================================================

pub fn compute_esr_f64(reference: &[f64], test: &[f64]) -> f64 {
    let len = reference.len().min(test.len());
    if len == 0 {
        return 0.0;
    }
    let mut noise = 0.0f64;
    let mut signal = 0.0f64;
    for i in 0..len {
        let diff = reference[i] - test[i];
        noise += diff * diff;
        signal += reference[i] * reference[i];
    }
    if signal == 0.0 {
        if noise == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        noise / signal
    }
}

pub fn esr_to_db_f64(esr: f64) -> f64 {
    if esr <= 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * esr.log10()
    }
}

// =============================================================================
// Top-level Oracle Dispatcher
// =============================================================================

pub fn oracle_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    match model_data.architecture.as_str() {
        "WaveNet" => {
            if is_a2_model(model_data) {
                oracle_a2_forward(model_data, input, config)
            } else {
                oracle_wavenet_forward(model_data, input, config)
            }
        }
        "LSTM" => oracle_lstm_forward(model_data, input, config),
        "ConvNet" => oracle_convnet_forward(model_data, input, config),
        _ => vec![0.0; input.len()],
    }
}

fn is_a2_model(model_data: &NamModelData) -> bool {
    let layers = &model_data.config.layers;
    if layers.is_empty() {
        return false;
    }
    // S14.2 (PM-15): A2 detection requires:
    // 1. At least one layer array with dilations+channels.
    // 2. head_scale is present (standard WaveNets don't have it).
    // 3. No post-stack head.
    // 4. For multi-array (> 1) models: at least one A2-specific feature.
    let l0 = &layers[0];
    if l0.dilations.is_none() || l0.channels.is_none() {
        return false;
    }
    if model_data.config.head_scale.is_none() {
        return false;
    }
    if let Some(ref head) = model_data.config.head
        && !head.is_null()
    {
        return false;
    }
    // Multi-array: require A2-specific features to avoid misclassifying
    // standard WaveNet models (which also have head_scale + dilations).
    if layers.len() > 1 {
        for l in layers.iter() {
            if let Some(ref raw) = l.layer_raw {
                let has_head1x1 = raw
                    .get("head1x1")
                    .and_then(|h| h.get("active"))
                    .and_then(|a| a.as_bool())
                    .unwrap_or(false);
                if has_head1x1 {
                    return true;
                }
                for key in FILM_KEYS.iter().map(|(k, _)| *k) {
                    if raw
                        .get(key)
                        .and_then(|v| v.get("active"))
                        .and_then(|a| a.as_bool())
                        .unwrap_or(false)
                    {
                        return true;
                    }
                }
                if let Some(act) = raw.get("activation")
                    && (act.is_array() || act.is_object())
                {
                    return true;
                }
            }
        }
        return false;
    }
    true
}

// =============================================================================
// WaveNet Oracle
// =============================================================================

fn oracle_wavenet_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let layers = &model_data.config.layers;
    let head_scale = model_data.config.head_scale.unwrap_or(1.0) as f64;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let num_frames = input.len();

    if layers.len() < 2 {
        return vec![0.0; num_frames];
    }

    let l0 = &layers[0];
    let l1 = &layers[1];
    let a0_ch = l0.channels.unwrap_or(16);
    let a0_head = l0.head_size.unwrap_or(8);
    let a0_k = l0.kernel_size.unwrap_or(3);
    let a0_dilations = l0.dilations.clone().unwrap_or_else(|| vec![1, 2, 4, 8]);
    let a0_cond = l0.condition_size.unwrap_or(1);
    let a1_ch = a0_head;
    let a1_head = l1.head_size.unwrap_or(1);
    let a1_k = l1.kernel_size.unwrap_or(3);
    let a1_dilations = l1.dilations.clone().unwrap_or_else(|| vec![1, 2, 4, 8]);
    let a1_cond = l1.condition_size.unwrap_or(1);

    // Receptive field per array
    let a0_rf: usize = a0_dilations.iter().map(|&d| (a0_k - 1) * d).sum();
    let a1_rf: usize = a1_dilations.iter().map(|&d| (a1_k - 1) * d).sum();
    let max_rf = a0_rf.max(a1_rf) + 64;
    let acc_mode = config.accumulation;

    let mut output = vec![0.0f64; num_frames];

    struct LayerW {
        conv_w: Vec<f64>,
        conv_b: Vec<f64>,
        mixin_w: Vec<f64>,
        l1x1_w: Vec<f64>,
        l1x1_b: Vec<f64>,
        dilation: usize,
    }

    // ── Read Array0 weights ────────────────────────────────────────────────
    let a0_rechannel_w = cursor.read_f64(a0_ch);
    let a0_num_layers = a0_dilations.len();
    let mut a0_lws: Vec<LayerW> = Vec::new();
    for &dil in &a0_dilations {
        let conv_w = cursor.read_f64(a0_ch * a0_ch * a0_k);
        let conv_b = cursor.read_f64(a0_ch);
        let mixin_w = cursor.read_f64(a0_cond * a0_ch);
        let l1x1_w = cursor.read_f64(a0_ch * a0_ch);
        let l1x1_b = cursor.read_f64(a0_ch);
        a0_lws.push(LayerW {
            conv_w,
            conv_b,
            mixin_w,
            l1x1_w,
            l1x1_b,
            dilation: dil,
        });
    }
    let a0_head_w = cursor.read_f64(a0_ch * a0_head);

    // ── Read Array1 weights ────────────────────────────────────────────────
    let a1_rechannel_w = cursor.read_f64(a0_ch * a1_ch);
    let a1_num_layers = a1_dilations.len();
    let mut a1_lws: Vec<LayerW> = Vec::new();
    for &dil in &a1_dilations {
        let conv_w = cursor.read_f64(a1_ch * a1_ch * a1_k);
        let conv_b = cursor.read_f64(a1_ch);
        let mixin_w = cursor.read_f64(a1_cond * a1_ch);
        let l1x1_w = cursor.read_f64(a1_ch * a1_ch);
        let l1x1_b = cursor.read_f64(a1_ch);
        a1_lws.push(LayerW {
            conv_w,
            conv_b,
            mixin_w,
            l1x1_w,
            l1x1_b,
            dilation: dil,
        });
    }
    let a1_head_w = cursor.read_f64(a1_ch * a1_head);
    let a1_head_b = cursor.read_f64(a1_head);

    // ── Array0 per-layer buffers ───────────────────────────────────────────
    // a0_bufs[0..N] where N = num_layers + 1 (buf[0] for rechannel, buf[i+1] for layer i output)
    let buf_size = max_rf + num_frames + 64;
    let bs = max_rf;
    let a0_buf_count = a0_num_layers + 1;
    let mut a0_bufs: Vec<Vec<f64>> = (0..a0_buf_count)
        .map(|_| vec![0.0f64; buf_size * a0_ch])
        .collect();
    let mut a0_ch_out = vec![0.0f64; num_frames * a0_ch];
    let mut a0_out = vec![0.0f64; num_frames * a0_head];
    let mut a0_head_accum = vec![0.0f64; num_frames * a0_ch];

    // Rechannel → a0_bufs[0]
    for (f, &inp) in input.iter().enumerate() {
        let idx = bs + f;
        for (c, rec_w) in a0_rechannel_w.iter().enumerate() {
            a0_bufs[0][idx * a0_ch + c] = inp * *rec_w;
        }
    }

    // Array0 layer cascade
    for (li, lw) in a0_lws.iter().enumerate() {
        let is_first = li == 0;
        for (f, &inp) in input.iter().enumerate() {
            let idx = bs + f;

            // Conv1d + mixin (reads from a0_bufs[li])
            let conv_out = {
                let hist = &a0_bufs[li];
                let mut conv_out = vec![0.0f64; a0_ch];
                for (oc, cv) in conv_out.iter_mut().enumerate() {
                    let mut sum = lw.conv_b[oc];
                    let wb = oc * a0_ch * a0_k;
                    for kt in 0..a0_k {
                        let off = (lw.dilation as isize) * ((kt as isize) + 1 - (a0_k as isize));
                        let ins = ((idx as isize) + off) as usize * a0_ch;
                        for ic in 0..a0_ch {
                            if ins + ic < hist.len() {
                                sum = mul_add_f64(
                                    hist[ins + ic],
                                    lw.conv_w[wb + ic * a0_k + kt],
                                    sum,
                                    acc_mode,
                                );
                            }
                        }
                    }
                    *cv = sum;
                }
                for (c, co) in conv_out.iter_mut().enumerate() {
                    *co = mul_add_f64(inp, lw.mixin_w[c], *co, acc_mode);
                }
                for cv in conv_out.iter_mut() {
                    *cv = oracle_tanh(*cv, config.activation);
                }
                conv_out
            };

            if is_first {
                for c in 0..a0_ch {
                    a0_head_accum[f * a0_ch + c] = conv_out[c];
                }
            } else {
                for c in 0..a0_ch {
                    a0_head_accum[f * a0_ch + c] =
                        accum_f64(a0_head_accum[f * a0_ch + c], conv_out[c], acc_mode);
                }
            }

            // L1x1 residual → next layer's buffer (reads a0_bufs[li] and writes a0_bufs[li+1])
            for oc in 0..a0_ch {
                let mut sum = lw.l1x1_b[oc];
                for (ic, co) in conv_out.iter().enumerate() {
                    sum = mul_add_f64(*co, lw.l1x1_w[oc * a0_ch + ic], sum, acc_mode);
                }
                a0_bufs[li + 1][idx * a0_ch + oc] =
                    accum_f64(a0_bufs[li][idx * a0_ch + oc], sum, acc_mode);
            }
        }
    }

    // Array0 head rechannel
    for f in 0..num_frames {
        for hc in 0..a0_head {
            let mut sum = 0.0f64;
            for c in 0..a0_ch {
                sum = mul_add_f64(
                    a0_head_accum[f * a0_ch + c],
                    a0_head_w[hc * a0_ch + c],
                    sum,
                    acc_mode,
                );
            }
            a0_out[f * a0_head + hc] = sum;
        }
    }

    // Save Array0 channel output for Array1's rechannel
    for f in 0..num_frames {
        let idx = bs + f;
        a0_ch_out[f * a0_ch..f * a0_ch + a0_ch]
            .copy_from_slice(&a0_bufs[a0_num_layers][idx * a0_ch..idx * a0_ch + a0_ch]);
    }

    // ── Array1 per-layer buffers ───────────────────────────────────────────
    let a1_buf_count = a1_num_layers + 1;
    let mut a1_bufs: Vec<Vec<f64>> = (0..a1_buf_count)
        .map(|_| vec![0.0f64; buf_size * a1_ch])
        .collect();
    let mut a1_head_accum = vec![0.0f64; num_frames * a1_ch];

    // Array1 rechannel from a0_ch_out → a1_bufs[0]
    for f in 0..num_frames {
        let idx = bs + f;
        for c in 0..a1_ch {
            let mut sum = 0.0f64;
            for ic in 0..a0_ch {
                sum = mul_add_f64(
                    a0_ch_out[f * a0_ch + ic],
                    a1_rechannel_w[c * a0_ch + ic],
                    sum,
                    acc_mode,
                );
            }
            a1_bufs[0][idx * a1_ch + c] = sum;
        }
    }

    // Array1 layer cascade
    for (li, lw) in a1_lws.iter().enumerate() {
        let is_first = li == 0;
        for (f, &inp) in input.iter().enumerate() {
            let idx = bs + f;

            let conv_out = {
                let hist = &a1_bufs[li];
                let mut conv_out = vec![0.0f64; a1_ch];
                for (oc, cv) in conv_out.iter_mut().enumerate() {
                    let mut sum = lw.conv_b[oc];
                    let wb = oc * a1_ch * a1_k;
                    for kt in 0..a1_k {
                        let off = (lw.dilation as isize) * ((kt as isize) + 1 - (a1_k as isize));
                        let ins = ((idx as isize) + off) as usize * a1_ch;
                        for ic in 0..a1_ch {
                            if ins + ic < hist.len() {
                                sum = mul_add_f64(
                                    hist[ins + ic],
                                    lw.conv_w[wb + ic * a1_k + kt],
                                    sum,
                                    acc_mode,
                                );
                            }
                        }
                    }
                    *cv = sum;
                }
                for (c, co) in conv_out.iter_mut().enumerate() {
                    *co = mul_add_f64(inp, lw.mixin_w[c], *co, acc_mode);
                }
                for cv in conv_out.iter_mut() {
                    *cv = oracle_tanh(*cv, config.activation);
                }
                conv_out
            };

            if is_first {
                for c in 0..a1_ch {
                    a1_head_accum[f * a1_ch + c] =
                        accum_f64(a0_out[f * a1_ch + c], conv_out[c], acc_mode);
                }
            } else {
                for c in 0..a1_ch {
                    a1_head_accum[f * a1_ch + c] =
                        accum_f64(a1_head_accum[f * a1_ch + c], conv_out[c], acc_mode);
                }
            }

            // L1x1 residual → next layer's buffer
            for oc in 0..a1_ch {
                let mut sum = lw.l1x1_b[oc];
                for (ic, co) in conv_out.iter().enumerate() {
                    sum = mul_add_f64(*co, lw.l1x1_w[oc * a1_ch + ic], sum, acc_mode);
                }
                a1_bufs[li + 1][idx * a1_ch + oc] =
                    accum_f64(a1_bufs[li][idx * a1_ch + oc], sum, acc_mode);
            }
        }
    }

    // Array1 head rechannel → 1-channel output × head_scale
    for f in 0..num_frames {
        let mut y = a1_head_b[0];
        for c in 0..a1_ch {
            y = mul_add_f64(
                a1_head_accum[f * a1_ch + c],
                a1_head_w[c * a1_head],
                y,
                acc_mode,
            );
        }
        output[f] = y * head_scale;
    }

    output
}

// =============================================================================
// A2 Oracle — Generic topology support (S13.2)
// =============================================================================
// Supports arbitrary channel counts, kernel sizes, dilations, bottleneck≠channels,
// heterogeneous activations, gating/blending, head1x1, condition_size>1, condition_dsp,
// and all 8 FiLM insertion slots including head1x1_post_film (slot 7).
// Backward-compatible with the legacy 23-layer A2 fast-path models.

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
fn oracle_a2_forward(
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

fn oracle_lstm_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let h = model_data.config.hidden_size.unwrap_or(16);
    let nlayers = model_data.config.num_layers.unwrap_or(1);
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);

    struct LstmLW {
        ih_w: Vec<Vec<Vec<f64>>>, // [gate=4][row=ih][col=h]
        bias: Vec<f64>,
        hidden: Vec<f64>,
        cell: Vec<f64>,
        in_size: usize,
    }

    // Flattened weights: depends on weights_layout
    // Original: [gate][H][IH] = row-major within each gate
    // GateMajor: [gate][IH][H] = row-major within each gate
    let is_gate_major = model_data.weights_layout == WeightsLayout::GateMajorLstm;
    let mut ll: Vec<LstmLW> = Vec::new();
    for l in 0..nlayers {
        let ins = if l == 0 { 1 } else { h };
        let ih = ins + h;
        let raw = cursor.read_f64(4 * ih * h);
        let bias = cursor.read_f64(4 * h);
        let hidden = cursor.read_f64(h);
        let cell = cursor.read_f64(h);

        // Build weight matrix [gate][row][col]
        let mut wh = vec![vec![vec![0.0f64; h]; ih]; 4];
        for g in 0..4 {
            for r in 0..ih {
                for c in 0..h {
                    wh[g][r][c] = if is_gate_major {
                        // GateMajor: raw[gate][row=IH][col=H]
                        raw[g * ih * h + r * h + c]
                    } else {
                        // Original: raw[gate][col=H][row=IH]
                        raw[g * h * ih + c * ih + r]
                    };
                }
            }
        }

        ll.push(LstmLW {
            ih_w: wh,
            bias,
            hidden,
            cell,
            in_size: ins,
        });
    }

    let head_w = cursor.read_f64(h);
    let head_b = cursor.read_one_f64();

    let num_frames = input.len();
    let mut output = vec![0.0f64; num_frames];
    let acc_mode = config.accumulation;

    // Clone cell states for independent runs (decomposition needs fresh state)
    let _orig_cell: Vec<Vec<f64>> = ll.iter().map(|l| l.cell.clone()).collect();
    let _orig_hidden: Vec<Vec<f64>> = ll.iter().map(|l| l.hidden.clone()).collect();

    #[allow(clippy::needless_range_loop)]
    for f in 0..num_frames {
        let x = input[f];

        // Reset states to initial for each frame? No — LSTM is recurrent.
        // We process sequentially.

        for l in 0..nlayers {
            let ins = ll[l].in_size;
            let ih = ins + h;

            // Build state: [input_part; hidden]
            let mut state = vec![0.0f64; ih];
            if l == 0 {
                state[0] = x;
            } else {
                state[..ins].copy_from_slice(&ll[l - 1].hidden[..ins]);
            }
            state[ins..ins + h].copy_from_slice(&ll[l].hidden[..h]);

            // GEMV: gates[g*h + i] = bias[g*h + i] + Σ_j state[j] * wh[g][j][i]
            let mut gates = vec![0.0f64; 4 * h];
            for g in 0..4 {
                for i in 0..h {
                    let mut sum = ll[l].bias[g * h + i];
                    for j in 0..ih {
                        sum = mul_add_f64(state[j], ll[l].ih_w[g][j][i], sum, acc_mode);
                    }
                    gates[g * h + i] = sum;
                }
            }

            // Fused gates
            for i in 0..h {
                let gi = gates[i];
                let gf = gates[h + i];
                let gg = gates[2 * h + i];
                let go = gates[3 * h + i];

                let fg = oracle_sigmoid(gf, config.activation);
                let ig = oracle_sigmoid(gi, config.activation);
                let gv = oracle_tanh(gg, config.activation);
                let og = oracle_sigmoid(go, config.activation);

                let nc = fg * ll[l].cell[i] + ig * gv;
                let hv = og * oracle_tanh(nc, config.activation);

                ll[l].cell[i] = nc;
                ll[l].hidden[i] = hv;
            }
        }

        // Head
        let last_h = &ll.last().unwrap().hidden;
        let mut y = head_b;
        for i in 0..h {
            y = mul_add_f64(last_h[i], head_w[i], y, acc_mode);
        }
        output[f] = y;
    }

    output
}

// =============================================================================
// ConvNet Oracle
// =============================================================================

fn oracle_convnet_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let layers = &model_data.config.layers;
    let head_scale = model_data.config.head_scale.unwrap_or(1.0) as f64;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let num_frames = input.len();
    let acc_mode = config.accumulation;

    if layers.is_empty() {
        return vec![0.0; num_frames];
    }

    struct BlockW {
        conv_w: Vec<f64>,
        conv_b: Vec<f64>,
        bn_scale: Vec<f64>,
        bn_offset: Vec<f64>,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        activation: String,
    }

    let mut blocks: Vec<BlockW> = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        let out_ch = layer.channels.unwrap_or(8);
        let in_ch = if i == 0 {
            1
        } else {
            layers[i - 1].channels.unwrap_or(out_ch)
        };
        let kernel = layer.kernel_size.unwrap_or(3);
        let dilation = layer
            .dilations
            .as_ref()
            .and_then(|d| d.first().copied())
            .unwrap_or(1);
        let activation = layer
            .activation
            .clone()
            .unwrap_or_else(|| "Tanh".to_string());

        let conv_w = cursor.read_f64(in_ch * out_ch * kernel);
        let conv_b = cursor.read_f64(out_ch);
        let bn_scale = cursor.read_f64(out_ch);
        let bn_offset = cursor.read_f64(out_ch);

        blocks.push(BlockW {
            conv_w,
            conv_b,
            bn_scale,
            bn_offset,
            in_ch,
            out_ch,
            kernel,
            dilation,
            activation,
        });
    }

    let head = {
        let head_config = model_data.config.parse_head();
        head_config.map(|hc| {
            let last_out_ch = blocks.last().map(|b| b.out_ch).unwrap_or(1);
            let h_in_ch = hc.channels.unwrap_or(last_out_ch);
            let h_out_ch = hc.out_channels.unwrap_or(1);
            let h_kernel = hc.kernel_size.unwrap_or(1);
            let h_has_bias = hc.bias.unwrap_or(true);
            let h_activation = hc.activation.unwrap_or_else(|| "Tanh".to_string());

            let h_w = cursor.read_f64(h_in_ch * h_out_ch * h_kernel);
            let h_b = if h_has_bias {
                cursor.read_f64(h_out_ch)
            } else {
                vec![0.0; h_out_ch]
            };

            (h_w, h_b, h_in_ch, h_out_ch, h_kernel, h_activation)
        })
    };

    let max_rf: usize = blocks
        .iter()
        .map(|b| (b.kernel - 1) * b.dilation)
        .max()
        .unwrap_or(0)
        + 64;
    let hist_size = max_rf + num_frames + 64;

    let mut block_hists: Vec<Vec<f64>> = blocks
        .iter()
        .map(|b| vec![0.0f64; hist_size * b.in_ch])
        .collect();

    let mut output = vec![0.0f64; num_frames];

    for f in 0..num_frames {
        let hist_i = max_rf + f;

        block_hists[0][hist_i * blocks[0].in_ch] = input[f];

        let mut last_out: Option<Vec<f64>> = None;

        for (bi, block) in blocks.iter().enumerate() {
            let in_ch = block.in_ch;
            let out_ch = block.out_ch;
            let kernel = block.kernel;
            let dilation = block.dilation;

            let mut conv_out = vec![0.0f64; out_ch];
            let hist = &block_hists[bi];

            for (oc, cv) in conv_out.iter_mut().enumerate() {
                let mut sum = block.conv_b[oc];
                let wb = oc * in_ch * kernel;
                for kt in 0..kernel {
                    let off = (dilation as isize) * ((kt as isize) + 1 - (kernel as isize));
                    let ins = ((hist_i as isize) + off) as usize;
                    if ins < hist_size {
                        for ic in 0..in_ch {
                            sum = mul_add_f64(
                                hist[ins * in_ch + ic],
                                block.conv_w[wb + ic * kernel + kt],
                                sum,
                                acc_mode,
                            );
                        }
                    }
                }
                *cv = sum;
            }

            for (c, cv) in conv_out.iter_mut().enumerate() {
                *cv = cv.mul_add(block.bn_scale[c], block.bn_offset[c]);
            }

            oracle_apply_activation(&mut conv_out, &block.activation, config);

            if bi + 1 < blocks.len() {
                let next_in_ch = blocks[bi + 1].in_ch;
                for c in 0..out_ch.min(next_in_ch) {
                    block_hists[bi + 1][hist_i * next_in_ch + c] = conv_out[c];
                }
            }

            if bi == blocks.len() - 1 {
                last_out = Some(conv_out);
            }
        }

        let block_out = last_out.unwrap();

        let y = if let Some((ref hw, ref hb, h_in_ch, h_out_ch, h_kernel, ref h_act)) = head {
            let mut h_out = vec![0.0f64; h_out_ch];
            for oc in 0..h_out_ch {
                let mut sum = hb[oc];
                let wb = oc * h_in_ch * h_kernel;
                for kt in 0..h_kernel {
                    for ic in 0..h_in_ch {
                        sum =
                            mul_add_f64(block_out[ic], hw[wb + ic * h_kernel + kt], sum, acc_mode);
                    }
                }
                h_out[oc] = sum;
            }
            oracle_apply_activation(&mut h_out, h_act, config);
            h_out[0]
        } else {
            block_out[0]
        };

        output[f] = y * head_scale;
    }

    output
}

fn oracle_apply_activation(data: &mut [f64], activation: &str, config: &PrecisionConfig) {
    match activation {
        "Tanh" => {
            for v in data.iter_mut() {
                *v = oracle_tanh(*v, config.activation);
            }
        }
        "HardTanh" => {
            for v in data.iter_mut() {
                *v = v.clamp(-1.0, 1.0);
            }
        }
        "FastTanh" => {
            for v in data.iter_mut() {
                *v = oracle_tanh(*v, config.activation);
            }
        }
        "ReLU" => {
            for v in data.iter_mut() {
                *v = v.max(0.0);
            }
        }
        "Sigmoid" => {
            for v in data.iter_mut() {
                *v = oracle_sigmoid(*v, config.activation);
            }
        }
        "SiLU" => {
            for v in data.iter_mut() {
                let s = oracle_sigmoid(*v, config.activation);
                *v *= s;
            }
        }
        "HardSwish" => {
            for v in data.iter_mut() {
                let relu6 = (*v + 3.0).clamp(0.0, 6.0);
                *v = *v * relu6 / 6.0;
            }
        }
        "Softsign" => {
            for v in data.iter_mut() {
                *v /= 1.0 + v.abs();
            }
        }
        _ => {
            for v in data.iter_mut() {
                *v = oracle_tanh(*v, config.activation);
            }
        }
    }
}

// =============================================================================
// Decomposition
// =============================================================================

#[derive(Debug)]
pub struct DecompositionResult {
    pub label: String,
    pub architecture: String,
    pub esr_f32_vs_f64: f64,
    pub esr_quant_f16c: Option<f64>,
    pub esr_quant_bf16: Option<f64>,
    pub esr_activation: Option<f64>,
    pub esr_accumulation: Option<f64>,
    pub esr_combined: Option<f64>,
}

impl DecompositionResult {
    pub fn esr_quant_f16c_display(&self) -> f64 {
        self.esr_quant_f16c.unwrap_or(0.0)
    }
    pub fn esr_quant_bf16_display(&self) -> f64 {
        self.esr_quant_bf16.unwrap_or(0.0)
    }
    pub fn esr_activation_display(&self) -> f64 {
        self.esr_activation.unwrap_or(0.0)
    }
    pub fn esr_accumulation_display(&self) -> f64 {
        self.esr_accumulation.unwrap_or(0.0)
    }
    pub fn esr_combined_display(&self) -> f64 {
        self.esr_combined.unwrap_or(0.0)
    }
}

pub fn run_decomposition(
    label: &str,
    architecture: &str,
    model_data: &NamModelData,
    production_output: &[f64],
    input_signal: &[f64],
) -> DecompositionResult {
    let oracle_cfg = PrecisionConfig::default();

    let oracle_out = oracle_forward(model_data, input_signal, &oracle_cfg);
    let esr_f32_vs_f64 = compute_esr_f64(&oracle_out, production_output);

    let mut cfg_f16c = oracle_cfg;
    cfg_f16c.weight_precision = WeightPrecision::F16C;
    let out_f16c = oracle_forward(model_data, input_signal, &cfg_f16c);
    let esr_f16c = compute_esr_f64(&oracle_out, &out_f16c);

    let mut cfg_bf16 = oracle_cfg;
    cfg_bf16.weight_precision = WeightPrecision::BF16;
    let out_bf16 = oracle_forward(model_data, input_signal, &cfg_bf16);
    let esr_bf16 = compute_esr_f64(&oracle_out, &out_bf16);

    let mut cfg_act = oracle_cfg;
    cfg_act.activation = ActivationMode::PadeMinimax;
    let out_act = oracle_forward(model_data, input_signal, &cfg_act);
    let esr_act = compute_esr_f64(&oracle_out, &out_act);

    let mut cfg_acc = oracle_cfg;
    cfg_acc.accumulation = AccumulationMode::F32Plain;
    let out_acc = oracle_forward(model_data, input_signal, &cfg_acc);
    let esr_acc = compute_esr_f64(&oracle_out, &out_acc);

    let combined_cfg = PrecisionConfig {
        weight_precision: WeightPrecision::F16C,
        activation: ActivationMode::PadeMinimax,
        accumulation: AccumulationMode::F32Plain,
    };
    let out_combined = oracle_forward(model_data, input_signal, &combined_cfg);
    let esr_combined = compute_esr_f64(&oracle_out, &out_combined);

    DecompositionResult {
        label: label.to_string(),
        architecture: architecture.to_string(),
        esr_f32_vs_f64,
        esr_quant_f16c: Some(esr_f16c),
        esr_quant_bf16: Some(esr_bf16),
        esr_activation: Some(esr_act),
        esr_accumulation: Some(esr_acc),
        esr_combined: Some(esr_combined),
    }
}
