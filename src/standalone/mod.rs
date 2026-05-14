// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Módulos específicos para a execução standalone (PipeWire, CLI).

#![cfg(feature = "standalone")]

pub mod cli;
pub mod colors;
pub mod pw_host;
pub mod rt_setup;

pub use cli::*;
pub use colors::*;
pub use pw_host::*;
pub use rt_setup::*;
