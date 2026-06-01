// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![cfg(target_arch = "x86_64")]

//! Utilitários para configuração e monitoramento de tempo real e hardware.
//!
//! Este módulo contém funções auxiliares para gerenciar afinidade de CPU,
//! prioridades SCHED_FIFO, travamento de memória (mlockall), e telemetria
//! de status RT do motor de áudio.

pub mod affinity;
pub mod pm_qos;
pub mod telemetry;
pub mod thread;
pub mod tsc;

pub use affinity::*;
pub use pm_qos::*;
pub use telemetry::*;
pub use thread::*;
pub use tsc::*;

/// Calcula os multiplicadores finais combinando ganho do usuário e ajustes do modelo.
/// Operação ultra-rápida (apenas multiplicações lineares) para manter o callback RT leve.
pub fn compute_gain_multipliers(
    u_in_mult: f32,
    u_out_mult: f32,
    m_in_mult: f32,
    m_out_mult: f32,
    out_in_mult: &mut f32,
    out_out_mult: &mut f32,
) {
    *out_in_mult = u_in_mult * m_in_mult;
    *out_out_mult = u_out_mult * m_out_mult;
}

#[cfg(test)]
#[path = "rt_setup_test.rs"]
mod rt_setup_test;
