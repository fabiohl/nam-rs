// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::sync::{Mutex, MutexGuard};

use nam_rs::math::activations::{
    ActivationPrecision, activation_precision, set_activation_precision,
};

static PRECISION_MUTEX: Mutex<()> = Mutex::new(());

/// RAII guard that sets the global activation precision mode on construction
/// and restores the original mode on drop.
///
/// Holds the global PRECISION_MUTEX for the guard's lifetime, serializing
/// concurrent tests that switch activation precision.
pub struct PrecisionGuard {
    original_mode: ActivationPrecision,
    _lock: MutexGuard<'static, ()>,
}

impl PrecisionGuard {
    /// Acquires the global precision mutex and sets `ActivationPrecision` to `mode`.
    ///
    /// Returns a guard that restores the previous precision mode on drop.
    #[must_use]
    pub fn new(mode: ActivationPrecision) -> Self {
        let _lock = PRECISION_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let original_mode = activation_precision();
        set_activation_precision(mode);
        Self {
            original_mode,
            _lock,
        }
    }
}

impl Drop for PrecisionGuard {
    fn drop(&mut self) {
        set_activation_precision(self.original_mode);
    }
}
