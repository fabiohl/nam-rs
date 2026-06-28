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

use crate::loader::nam_json::model::NamModelData;
use crate::math::common::half::f16_bits_to_f32;

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

/// Converts a u16 f16c weight to f64 (for binary-format models with pre-quantized weights).
#[allow(dead_code)]
#[inline]
fn weight_f16c_to_f64(w: u16, _mode: WeightPrecision) -> f64 {
    f16_bits_to_f32(w) as f64
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
        _ => vec![0.0; input.len()],
    }
}

fn is_a2_model(model_data: &NamModelData) -> bool {
    let layers = &model_data.config.layers;
    if layers.len() != 1 {
        return false;
    }
    let l0 = &layers[0];
    l0.kernel_size.is_none() && l0.channels.is_some()
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

    let a0_rf: usize = a0_dilations.iter().map(|&d| (a0_k - 1) * d).sum();
    let a1_rf: usize = a1_dilations.iter().map(|&d| (a1_k - 1) * d).sum();
    let max_rf = a0_rf.max(a1_rf) + 64;
    let max_ch = a0_ch.max(a1_ch);
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
    // skip head_scale — already read from JSON config

    // ── Buffer allocation ──────────────────────────────────────────────────
    let buf_len = max_rf * max_ch * 2 + num_frames * max_ch * 2 + 4096;
    let mut layer_buffer = vec![0.0f64; buf_len];
    let bs = max_rf;

    // ── Array0 forward ─────────────────────────────────────────────────────
    let mut a0_head_accum = vec![0.0f64; num_frames * a0_ch];
    let mut a0_ch_out = vec![0.0f64; num_frames * a0_ch];
    let mut a0_out = vec![0.0f64; num_frames * a0_head];

    // Rechannel input → layer_buffer
    for (f, inp) in input.iter().enumerate() {
        let idx = bs + f;
        for (c, rec_w) in a0_rechannel_w.iter().enumerate() {
            layer_buffer[idx * a0_ch + c] = *inp * *rec_w;
        }
    }

    for (li, lw) in a0_lws.iter().enumerate() {
        let is_first = li == 0;
        for (f, inp) in input.iter().enumerate() {
            let idx = bs + f;

            // Conv1d + bias
            let mut conv_out = vec![0.0f64; a0_ch];
            for (oc, cv) in conv_out.iter_mut().enumerate() {
                let mut sum = lw.conv_b[oc];
                let wb = oc * a0_ch * a0_k;
                for kt in 0..a0_k {
                    let off = (lw.dilation as isize) * ((kt as isize) + 1 - (a0_k as isize));
                    let ins = ((idx as isize) + off) as usize * a0_ch;
                    for ic in 0..a0_ch {
                        if ins + ic < layer_buffer.len() {
                            sum = mul_add_f64(
                                layer_buffer[ins + ic],
                                lw.conv_w[wb + ic * a0_k + kt],
                                sum,
                                acc_mode,
                            );
                        }
                    }
                }
                *cv = sum;
            }

            // Mixin
            let cond_in = *inp; // condition is the raw input (cond_size=1)
            for (c, co) in conv_out.iter_mut().enumerate() {
                *co = mul_add_f64(cond_in, lw.mixin_w[c], *co, acc_mode);
            }

            // Tanh
            for cv in conv_out.iter_mut() {
                *cv = oracle_tanh(*cv, config.activation);
            }

            // Head accumulate (skip connections)
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

            // 1x1 residual
            for oc in 0..a0_ch {
                let mut sum = lw.l1x1_b[oc];
                for (ic, co) in conv_out.iter().enumerate() {
                    sum = mul_add_f64(*co, lw.l1x1_w[oc * a0_ch + ic], sum, acc_mode);
                }
                layer_buffer[idx * a0_ch + oc] =
                    accum_f64(layer_buffer[idx * a0_ch + oc], sum, acc_mode);
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
                    a0_head_w[c * a0_head + hc],
                    sum,
                    acc_mode,
                );
            }
            a0_out[f * a0_head + hc] = sum;
        }
    }

    // Save array0's CH-channel output (after last layer's residual)
    for f in 0..num_frames {
        let idx = bs + f;
        a0_ch_out[f * a0_ch..f * a0_ch + a0_ch]
            .copy_from_slice(&layer_buffer[idx * a0_ch..idx * a0_ch + a0_ch]);
    }

    // ── Array1 forward ─────────────────────────────────────────────────────
    let mut a1_head_accum = vec![0.0f64; num_frames * a1_ch];

    // Rechannel: a0_ch_out → a1_ch channels into layer_buffer
    for f in 0..num_frames {
        let idx = bs + f;
        for c in 0..a1_ch {
            let mut sum = 0.0f64;
            for ic in 0..a0_ch {
                sum = mul_add_f64(
                    a0_ch_out[f * a0_ch + ic],
                    a1_rechannel_w[ic * a1_ch + c],
                    sum,
                    acc_mode,
                );
            }
            layer_buffer[idx * a1_ch + c] = sum;
        }
    }

    for (li, lw) in a1_lws.iter().enumerate() {
        let is_first = li == 0;
        for f in 0..num_frames {
            let idx = bs + f;

            let mut conv_out = vec![0.0f64; a1_ch];
            for (oc, cv) in conv_out.iter_mut().enumerate() {
                let mut sum = lw.conv_b[oc];
                let wb = oc * a1_ch * a1_k;
                for kt in 0..a1_k {
                    let off = (lw.dilation as isize) * ((kt as isize) + 1 - (a1_k as isize));
                    let ins = ((idx as isize) + off) as usize * a1_ch;
                    for ic in 0..a1_ch {
                        if ins + ic < layer_buffer.len() {
                            sum = mul_add_f64(
                                layer_buffer[ins + ic],
                                lw.conv_w[wb + ic * a1_k + kt],
                                sum,
                                acc_mode,
                            );
                        }
                    }
                }
                *cv = sum;
            }

            // Mixin
            let cond_in = input[f]; // condition is the raw input for all arrays
            for (c, co) in conv_out.iter_mut().enumerate() {
                *co = mul_add_f64(cond_in, lw.mixin_w[c], *co, acc_mode);
            }

            // Tanh
            for cv in conv_out.iter_mut() {
                *cv = oracle_tanh(*cv, config.activation);
            }

            // Head accumulate — seed a0_out on first layer
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

            // 1x1 residual
            for oc in 0..a1_ch {
                let mut sum = lw.l1x1_b[oc];
                for (ic, co) in conv_out.iter().enumerate() {
                    sum = mul_add_f64(*co, lw.l1x1_w[oc * a1_ch + ic], sum, acc_mode);
                }
                layer_buffer[idx * a1_ch + oc] =
                    accum_f64(layer_buffer[idx * a1_ch + oc], sum, acc_mode);
            }
        }
    }

    // Array1 head rechannel → 1-channel output
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
// A2 Oracle
// =============================================================================

const A2_NUM_LAYERS: usize = 23;
const A2_HEAD_KERNEL: usize = 16;
const A2_KS: [usize; 23] = [
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
];
const A2_DIL: [usize; 23] = [
    1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
];

fn oracle_a2_forward(
    model_data: &NamModelData,
    input: &[f64],
    config: &PrecisionConfig,
) -> Vec<f64> {
    let ch = model_data
        .config
        .layers
        .first()
        .and_then(|l| l.channels)
        .unwrap_or(8);
    let head_scale = model_data.config.head_scale.unwrap_or(1.0) as f64;
    let mut cursor = Cursor::new(&model_data.weights, config.weight_precision);
    let num_frames = input.len();

    // Rechannel: CH f32
    let rechannel_w = cursor.read_f64(ch);

    // Per-layer weights
    struct A2LW {
        conv_w: Vec<f64>,
        conv_b: Vec<f64>,
        mixin_w: Vec<f64>,
        l1x1_w: Vec<f64>,
        l1x1_b: Vec<f64>,
        ks: usize,
        dil: usize,
    }
    let mut lws: Vec<A2LW> = Vec::new();
    for li in 0..A2_NUM_LAYERS {
        let ks = A2_KS[li];
        let dil = A2_DIL[li];
        let conv_w = cursor.read_f64(ch * ch * ks);
        let conv_b = cursor.read_f64(ch);
        let mixin_w = cursor.read_f64(ch);
        let l1x1_w = cursor.read_f64(ch * ch);
        let l1x1_b = cursor.read_f64(ch);
        lws.push(A2LW {
            conv_w,
            conv_b,
            mixin_w,
            l1x1_w,
            l1x1_b,
            ks,
            dil,
        });
    }

    // Head: 16*CH + 1 weight
    // Raw NAM JSON stores [channel][tap]; production transposes to [tap][channel].
    let head_w_raw = cursor.read_f64(A2_HEAD_KERNEL * ch);
    let mut head_w = vec![0.0f64; A2_HEAD_KERNEL * ch];
    for tap in 0..A2_HEAD_KERNEL {
        for c in 0..ch {
            head_w[tap * ch + c] = head_w_raw[c * A2_HEAD_KERNEL + tap];
        }
    }
    let head_b = cursor.read_one_f64();

    let acc_mode = config.accumulation;

    // Buffer allocation
    let max_dil: usize = *A2_DIL.iter().max().unwrap_or(&1);
    let max_ks: usize = *A2_KS.iter().max().unwrap_or(&6);
    let max_rf = (max_ks - 1) * max_dil + 64;
    let hist_size = max_rf + num_frames + 64;
    let mut history = vec![0.0f64; hist_size * ch];
    let bs = max_rf;

    // Head accumulator ring
    let hr_len = (max_rf + num_frames + 64).next_power_of_two();
    let mut head_acc = vec![0.0f64; hr_len * ch];
    let ring_mask = hr_len - 1;
    let mut head_wp = 0usize;

    let mut output = vec![0.0f64; num_frames];

    #[allow(clippy::explicit_counter_loop)]
    for (f, out_val) in output.iter_mut().enumerate() {
        let fi = bs + f;
        let x = input[f];

        // Rechannel: layer_in[c] = x * rechannel_w[c]
        let mut layer_in = vec![0.0f64; ch];
        for (c, li) in layer_in.iter_mut().enumerate() {
            *li = x * rechannel_w[c];
            history[fi * ch + c] = *li;
        }

        let head_col = head_wp;
        head_wp += 1;

        for (li, lw) in lws.iter().enumerate() {
            // Conv1d
            let mut z = vec![0.0f64; ch];
            for (oc, zv) in z.iter_mut().enumerate() {
                let mut sum = lw.conv_b[oc];
                let wb = oc * ch * lw.ks;
                for kt in 0..lw.ks {
                    let off = (lw.dil as isize) * ((kt as isize) + 1 - (lw.ks as isize));
                    let ins = ((fi as isize) + off) as usize * ch;
                    for ic in 0..ch {
                        if ins + ic < history.len() {
                            sum = mul_add_f64(
                                history[ins + ic],
                                lw.conv_w[wb + ic * lw.ks + kt],
                                sum,
                                acc_mode,
                            );
                        }
                    }
                }
                *zv = sum;
            }

            // Mixin
            for (c, zv) in z.iter_mut().enumerate() {
                *zv = mul_add_f64(lw.mixin_w[c], x, *zv, acc_mode);
            }

            // LeakyReLU(0.01)
            for zv in z.iter_mut().take(ch) {
                if *zv < 0.0 {
                    *zv *= 0.01;
                }
            }

            // Head accumulate
            let ho = head_col * ch;
            if li == 0 {
                head_acc[ho..ho + ch].copy_from_slice(&z[..ch]);
            } else {
                for (c, &zv) in z.iter().enumerate() {
                    head_acc[ho + c] = accum_f64(head_acc[ho + c], zv, acc_mode);
                }
            }

            // L1x1 residual (skip last)
            if li < A2_NUM_LAYERS - 1 {
                let mut next = vec![0.0f64; ch];
                for (oc, nv) in next.iter_mut().enumerate() {
                    let mut sum = lw.l1x1_b[oc];
                    for (ic, &zv) in z.iter().enumerate() {
                        sum = mul_add_f64(zv, lw.l1x1_w[oc * ch + ic], sum, acc_mode);
                    }
                    *nv = accum_f64(layer_in[oc], sum, acc_mode);
                }
                layer_in = next;
            }
        }

        // Head finalize
        let k = A2_HEAD_KERNEL;
        let cb = head_col.wrapping_sub(k - 1);
        let mut y = head_b;
        for t in 0..k {
            let col = cb.wrapping_add(t) & ring_mask;
            let so = col * ch;
            let wo = t * ch;
            for c in 0..ch {
                y = mul_add_f64(head_w[wo + c], head_acc[so + c], y, acc_mode);
            }
        }
        *out_val = y * head_scale;
    }

    output
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

    // Flattened: [gate=4][row=ih][col=h] = 4 * IH * H f32s
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
                    wh[g][r][c] = raw[g * ih * h + r * h + c];
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

    DecompositionResult {
        label: label.to_string(),
        architecture: architecture.to_string(),
        esr_f32_vs_f64,
        esr_quant_f16c: Some(esr_f16c),
        esr_quant_bf16: Some(esr_bf16),
        esr_activation: Some(esr_act),
        esr_accumulation: Some(esr_acc),
    }
}
