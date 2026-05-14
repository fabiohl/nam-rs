// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![warn(missing_docs)]

//! Biblioteca do motor de inferência Neural Amp Modeler (NAM-rs).
//!
//! Este arquivo define o "coração" do projeto. Ele é uma biblioteca compartilhada que:
//! 1. Serve de base para o executável CLI (`main.rs`).
//! 2. Serve de base para os plugins (ex: CLAP).
//!
//! O NAM-rs é organizado em uma estrutura modular para suportar múltiplos alvos de build:
//!
//! - **Common**: Infraestrutura agnóstica de host (diagnósticos, comunicação SPSC, parâmetros).
//! - **Standalone**: Implementação nativa para PipeWire e utilitários de CLI.
//!   Ativado via feature `standalone`. Utilizado por `main.rs`.
//! - **CLAP Plugin**: Integração como plugin no formato CLAP (CLever Audio Plug-in).
//!   Ativado via feature `clap-plugin`.
//!
//! O motor de inferência, os kernels de processamento (SIMD) e a lógica de carregamento
//! de modelos residem aqui para garantir paridade matemática entre o CLI e o Plugin.

/// Camada de infraestrutura comum e agnóstica de host.
pub mod common;
pub use common::*;

/// Infraestrutura para execução standalone (PipeWire + CLI).
#[cfg(feature = "standalone")]
pub mod standalone;
#[cfg(feature = "standalone")]
pub use standalone::*;

/// Integração no formato de plugin CLAP.
#[cfg(feature = "clap-plugin")]
pub mod clap;

/// Motor de processamento digital de sinais (DSP).
pub mod dsp;
/// Carregamento e construção de modelos (.nam, .namb).
pub mod loader;
/// Primitivas matemáticas e kernels SIMD otimizados.
pub mod math;
/// Definições de tensores e topologias de redes neurais.
pub mod models;
