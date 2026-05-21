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

// Retrocompatibilidade com versões mais antigas do GLIBC (ex: para Flatpak/Bitwig).
// Redireciona símbolos matemáticos para a versão estável do GLIBC_2.2.5.
// Como dependências externas usam esses símbolos, declaramos wrappers globais
// que interceptam as chamadas e saltam (jmp) via PLT para as versões compatíveis.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
core::arch::global_asm!(
    ".global log10f",
    ".type log10f, @function",
    "log10f:",
    "    jmp log10f_compat@PLT",
    ".symver log10f_compat, log10f@GLIBC_2.2.5",
    ".global atan2f",
    ".type atan2f, @function",
    "atan2f:",
    "    jmp atan2f_compat@PLT",
    ".symver atan2f_compat, atan2f@GLIBC_2.2.5",
    ".global acosf",
    ".type acosf, @function",
    "acosf:",
    "    jmp acosf_compat@PLT",
    ".symver acosf_compat, acosf@GLIBC_2.2.5"
);
