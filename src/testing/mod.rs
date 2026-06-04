// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Testing utilities for NAM-rs cross-validation and signal generation.
//!
//! This module provides:
//! - Deterministic stress signal generators (v1 for fast CI, v2 for comprehensive validation)
//! - Audio primitives ported from t3k-mushra (MIT-licensed)
//! - Perceptual metrics (ESR, LUFS) calibrated against published baselines
//! - WAV I/O helpers for test fixtures

pub mod mushra;
pub mod perceptual;
pub mod stress;
pub mod wav;
