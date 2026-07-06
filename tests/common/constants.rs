// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

pub const GOLDEN_NUM_SAMPLES: usize = 2048;
pub const GOLDEN_BLOCK_SIZE: usize = 64;
pub const TEST_BLOCK_SIZE: usize = 64;
pub const TEST_NUM_BLOCKS: usize = 4096;
pub const STRESS_SAMPLE_RATE: u32 = 48000;

pub const V2_PREWARM_SAMPLES: usize = 2048;
pub const V2_TEST_BLOCK_SIZE: usize = 64;
pub const V2_STRESS_DURATION_SECS: f64 = 5.0;

// ── T8.3 + SQ5.5: Re-derived fidelity gates (post-weight-dequantization) ──
// Moved to common module for cross-test access (Tarefa 8.6).
// SQ5.5: Recalibrated post-SQ5 removal of f16c weight quantization.
//
// Methodology: 24k prewarm + 256-sample sweep @ 48 kHz; both production (f32)
// and oracle (f64 ideal) fed the same signal; ESR measured on last 256 samples.
// Limit = measured ESR × 2 (conservative margin).
//
// Measured (post-SQ5, f32 weights, prewarm-paired, 256-sample sweep @ 48 kHz):
//   WaveNet: ESR = 6.13e-14  →  WAVENET_ESR_LIMIT = 6.13e-14 × 2  →  1e-12 (numerical floor)
//   LSTM:    ESR = 3.41e-3   →  LSTM_ESR_LIMIT    = 3.41e-3 × 2   →  7.0e-3
//   A2:      ESR = 2.22e-14  →  A2_ESR_LIMIT      = 2.22e-14 × 2  →  1e-12 (numerical floor, SQ5.5)
//   ConvNet: ESR = 1.83e-14  →  CONVNET_ESR_LIMIT = 1e-12 (numerical floor)
//   A2-FiLM-Lite:  ESR = 9.52e-15  →  A2_FILM_ESR_LIMIT = 1e-12 (numerical floor, S10.3)
//   A2-FiLM-Full:  ESR = 1.15e-14  →  A2_FILM_ESR_LIMIT = 1e-12 (numerical floor, S10.3)

pub const WAVENET_ESR_LIMIT: f64 = 1e-12;
pub const LSTM_ESR_LIMIT: f64 = 7.0e-3;
pub const A2_ESR_LIMIT: f64 = 1e-12;
pub const CONVNET_ESR_LIMIT: f64 = 1e-12;
pub const A2_FILM_ESR_LIMIT: f64 = 1e-9;
// SQ5.5: A2 Generic (wavenet_a2_max.nam) — disabled §7.1, dead threshold retained for meta-test calibration
pub const A2_GENERIC_ESR_LIMIT: f64 = 1e-9;
