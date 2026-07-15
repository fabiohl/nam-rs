// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

pub(crate) struct DialogSharedState {
    #[allow(dead_code)]
    pub alive: AtomicBool,
    pub pending_model: Mutex<Option<PathBuf>>,
    pub loading: AtomicBool,
}

pub(crate) struct IrDialogSharedState {
    #[allow(dead_code)]
    pub alive: AtomicBool,
    pub pending_ir: Mutex<Option<PathBuf>>,
    pub ir_loading: AtomicBool,
}
