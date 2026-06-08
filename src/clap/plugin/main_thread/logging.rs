// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! RT-Safe logging via atomic flags — consumes transient events on the main thread.

use super::NamClapMainThread;
use clack_extensions::log::{HostLog, LogSeverity};
use std::ffi::CString;

impl<'a> NamClapMainThread<'a> {
    /// Emits host log messages for RT telemetry flags that were set.
    ///
    /// RT-Safe: all `CString::new(…).unwrap_or_default()` calls use static
    /// ASCII literals with no internal null bytes — guaranteed non-panicking.
    pub(crate) fn emit_pending_logs(&mut self) {
        if let Some(log) = self.host.get_extension::<HostLog>() {
            let shared = self.host.shared();

            if self
                .shared
                .cold
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED)
            {
                let msg = CString::new("NAM-rs: Output clipping detected!").unwrap_or_default();
                log.log(&shared, LogSeverity::Warning, &msg);
            }

            if self
                .shared
                .cold
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_GC_OVERFLOW)
            {
                let msg = CString::new("NAM-rs: GC channel overflow! Possible memory leak.")
                    .unwrap_or_default();
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .cold
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_MODEL_LOAD_FAILED)
            {
                let msg = CString::new("NAM-rs: Critical failure! No active model for processing.")
                    .unwrap_or_default();
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .cold
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HEAP_ALLOC)
            {
                let msg = CString::new(
                    "NAM-rs: Heap allocation detected in audio thread during process()!",
                )
                .unwrap_or_default();
                log.log(&shared, LogSeverity::Error, &msg);
            }

            if self
                .shared
                .cold
                .rt_status
                .check_and_clear_flag(crate::common::spsc::RT_STATUS_HUGEPAGE_OK)
            {
                let msg = CString::new(
                    "NAM-rs: Huge pages (2MB) active — reduced TLB pressure on DSP thread.",
                )
                .unwrap_or_default();
                log.log(&shared, LogSeverity::Info, &msg);
            }
        }
    }
}
