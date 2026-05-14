// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Módulo para operações de processamento de sinal digital (DSP).
//!
//! Contém utilitários para manipulação de áudio em buffers, otimizados para
//! operação em estéreo e segurança contra clipping.
//!
//! # Funcionalidades
//! - **Ganho e Rampa**: Aplicação de ganho linear e rampas SIMD para transições suaves.
//! - **Clipping Detection**: Detecção de picos acima de 0dBFS integrada à aplicação de ganho.
//! - **Stereo Processing**: Kernels que operam simultaneamente em canais L/R para melhor localidade de cache.
//! - **Energy Computation**: Cálculo de energia RMS/Peak para telemetria de sinal.
//!
//! Contém implementações de algoritmos de áudio otimizados, incluindo
//! cálculos de energia, correlações e filtros.

pub mod gain;
pub mod gain_lut;
pub mod stereo;
