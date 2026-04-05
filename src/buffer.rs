// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Estruturas de dados e flags compartilhadas para coordenação entre threads.
//! Na Tarefa 1.2, este módulo será expandido com as novas estruturas de comunicação
//! lock-free SPSC para parametrização CLI → DSP (input/output gain, troca de .namb).

use std::sync::atomic::AtomicBool;

/// Flag global para shutdown coordenado e gracioso entre todas as threads.
/// Definida como `true` pelo handler de CTRL+C.
pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);
