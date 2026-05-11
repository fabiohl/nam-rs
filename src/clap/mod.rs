// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Integração do NAM-rs como plugin no formato CLAP (CLever Audio Plug-in).
//!
//! Este módulo contém a implementação do plugin utilizando o framework `clack`.
//! O suporte é ativado através da feature flag `clap-plugin`.

#![cfg(feature = "clap-plugin")]

pub mod descriptor;

// TODO: Implementar plugin, processor e main_thread conforme Sprint 1.
