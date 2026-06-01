// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Constantes matemáticas, coeficientes de polinômios Minimax e parâmetros de LUT.
//! Centralizadas para garantir consistência entre implementações escalares, AVX2 e AVX-512.

// --- Parâmetros de LUT de Ganho ---

/// Tamanho da tabela LUT para ganho. 4096 pontos fornecem precisão sub-0.001 dB.
pub const GAIN_LUT_SIZE: usize = 4096;
/// Limite inferior em dB para a LUT (Piso de silêncio prático -96dB).
pub const GAIN_MIN_DB: f32 = -96.0;
/// Limite superior em dB para a LUT (+30dB é um boost extremo).
pub const GAIN_MAX_DB: f32 = 30.0;
/// Intervalo total de ganho em dB suportado pela LUT.
pub const GAIN_DB_RANGE: f32 = GAIN_MAX_DB - GAIN_MIN_DB;
/// Passo em dB entre cada entrada da LUT.
pub const GAIN_DB_STEP: f32 = GAIN_DB_RANGE / (GAIN_LUT_SIZE as f32 - 1.0);
/// Inverso do passo de ganho (otimização para evitar divisões no hot-path).
pub const INV_GAIN_DB_STEP: f32 = 1.0 / GAIN_DB_STEP;

// --- Coeficientes Padé para Tanh (Rational [5,4]) ---

/// Numerador constante: x⁴ + 105·x² + 945.
pub const PADE_TANH_NUM_A: f32 = 105.0;
/// Numerador constante.
pub const PADE_TANH_NUM_B: f32 = 945.0;
/// Denominador coeficiente x⁴: 15.
pub const PADE_TANH_DEN_C4: f32 = 15.0;
/// Denominador coeficiente x²: 420.
pub const PADE_TANH_DEN_C2: f32 = 420.0;
/// Denominador constante.
pub const PADE_TANH_DEN_A: f32 = 945.0;
/// Limite de clamp para o domínio do Padé tanh (|x| < 4).
pub const PADE_TANH_CLAMP: f32 = 4.0;
