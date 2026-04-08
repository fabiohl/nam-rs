// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![warn(missing_docs)]

//! Biblioteca do motor de inferência Neural Amp Modeler (NAM-rs).
//!
//! Contém a infraestrutura matemática, arquitetura de inferência neural
//! sob Const Generics, utilitários Lock-Free (SPSC) e host Pipeline.

pub mod dsp;
pub mod loader;
pub mod math;
pub mod models;
pub mod pw_host;
pub mod spsc;
