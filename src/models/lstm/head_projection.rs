// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Head projection logic for LSTM models (previously compute_lstm_head_simd!).
//!
//! The macro was removed during loop-unswitching (Tarefa C1 — P6).
//! Head computation is now inlined directly in each model's processing kernel,
//! with the `use_f32_head` branch hoisted outside the per-sample loop.
