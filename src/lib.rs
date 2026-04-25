// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#![warn(missing_docs)]

//! Biblioteca do motor de inferência Neural Amp Modeler (NAM-rs).
//!
//! Este projeto segue uma arquitetura híbrida (Library + Binary):
//! 1. **Core Engine (Library)**: Este arquivo (`lib.rs`) define o motor de inferência,
//!    infraestrutura matemática (SIMD), utilitários Lock-Free e o host PipeWire.
//! 2. **Interface (Binary)**: O arquivo `main.rs` consome esta biblioteca para prover
//!    a interface de linha de comando (CLI).
//!
//! Esta separação permite:
//! - **Testabilidade**: Facilita a execução de testes de integração e benchmarks.
//! - **Modularidade**: Isola a lógica de processamento de áudio em tempo real da interface.
//! - **Reusabilidade**: Permite que o motor seja futuramente incorporado em outros
//!   front-ends (GUIs) ou formatos de plugins (VST/CLAP).

pub mod colors;
pub mod diagnostics;
pub mod dsp;
pub mod loader;
pub mod math;
pub mod models;
pub mod pw_host;
pub mod spsc;
