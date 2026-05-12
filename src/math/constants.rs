// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

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

// --- Limites de Clamp para Ativações ---

/// Limite de segurança para evitar overflow no polinômio de tanh (evita NaN).
pub const TANH_CLAMP_LIMIT: f32 = 15.0;
/// Limite de segurança para evitar overflow no polinômio de sigmoid.
pub const SIGMOID_CLAMP_LIMIT: f32 = 12.0;

// --- Coeficientes Tanh (Minimax Grau 7) ---

/// Coeficiente de grau 0 para o polinômio Minimax de tanh.
pub const TANH_C0: f32 = 0.166_814_34;
/// Coeficiente de grau 1 para o polinômio Minimax de tanh.
pub const TANH_C1: f32 = 0.008_153_17;
/// Coeficiente de grau 2 para o polinômio Minimax de tanh.
pub const TANH_C2: f32 = 0.000_246_32;

// --- Coeficientes Sigmoid/Exp (Minimax Grau 6) ---

/// Logaritmo de 2 na base e (usado para escalonamento de expoente).
pub const EXP_LOG2E: f32 = 1.442_695_1;
/// Parte alta de ln(2) para precisão estendida.
pub const EXP_LN2_HI: f32 = -0.693_145_75;
/// Parte baixa de ln(2) para precisão estendida.
pub const EXP_LN2_LO: f32 = -0.000_001_428_606_8;

/// Coeficiente de grau 6 para o polinômio de Exp.
pub const EXP_C6: f32 = 0.001_388_888_9;
/// Coeficiente de grau 5 para o polinômio de Exp.
pub const EXP_C5: f32 = 0.008_333_333;
/// Coeficiente de grau 4 para o polinômio de Exp.
pub const EXP_C4: f32 = 0.041_666_668;
/// Coeficiente de grau 3 para o polinômio de Exp.
pub const EXP_C3: f32 = 0.166_666_67;
/// Coeficiente de grau 2 para o polinômio de Exp.
pub const EXP_C2: f32 = 0.5;
