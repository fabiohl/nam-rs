// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulos compartilhados entre a versão standalone e plugin.

pub mod audio_host;
pub mod diagnostics;
pub mod params;
pub mod spsc;

pub use audio_host::*;
pub use diagnostics::*;
pub use params::*;
pub use spsc::*;
