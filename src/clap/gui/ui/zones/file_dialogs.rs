// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::dialog_state::{DialogSharedState, IrDialogSharedState};
use clack_plugin::host::HostSharedHandle;
use std::sync::Arc;

pub(crate) fn spawn_file_dialog(
    state: Arc<DialogSharedState>,
    host_static: HostSharedHandle<'static>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let path_opt = rfd::FileDialog::new()
                .add_filter("NAM Model", &["nam", "namb"])
                .pick_file();
            let _ = tx.send(path_opt);
        });
        match rx.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(Some(path)) => {
                if let Ok(mut guard) = state.pending_model.lock() {
                    *guard = Some(path);
                    host_static.request_callback();
                }
            }
            Ok(None) => {
                state
                    .loading
                    .store(false, std::sync::atomic::Ordering::Release);
            }
            Err(_) => {
                state
                    .loading
                    .store(false, std::sync::atomic::Ordering::Release);
                if let (Some(log), Ok(c_msg)) = (
                    host_static.get_extension::<clack_extensions::log::HostLog>(),
                    std::ffi::CString::new("NAM-rs: File dialog portal timed out after 120s"),
                ) {
                    log.log(
                        &host_static,
                        clack_extensions::log::LogSeverity::Warning,
                        &c_msg,
                    );
                }
            }
        }
    })
}

pub(crate) fn spawn_ir_file_dialog(
    state: Arc<IrDialogSharedState>,
    host_static: HostSharedHandle<'static>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let path_opt = rfd::FileDialog::new()
                .add_filter("WAV Impulse Response", &["wav"])
                .pick_file();
            let _ = tx.send(path_opt);
        });
        match rx.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(Some(path)) => {
                if let Ok(mut guard) = state.pending_ir.lock() {
                    *guard = Some(path);
                    host_static.request_callback();
                }
            }
            Ok(None) => {
                state
                    .ir_loading
                    .store(false, std::sync::atomic::Ordering::Release);
            }
            Err(_) => {
                state
                    .ir_loading
                    .store(false, std::sync::atomic::Ordering::Release);
                if let (Some(log), Ok(c_msg)) = (
                    host_static.get_extension::<clack_extensions::log::HostLog>(),
                    std::ffi::CString::new("NAM-rs: IR file dialog portal timed out after 120s"),
                ) {
                    log.log(
                        &host_static,
                        clack_extensions::log::LogSeverity::Warning,
                        &c_msg,
                    );
                }
            }
        }
    })
}

#[cfg(test)]
mod file_dialogs_test {
    #[allow(unused_imports)]
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn test_dialog_state_outlives_plugin_drop() {
        let state = Arc::new(super::super::dialog_state::DialogSharedState {
            pending_model: std::sync::Mutex::new(None),
            loading: AtomicBool::new(true),
        });
        let state_clone = Arc::clone(&state);

        drop(state);

        state_clone
            .loading
            .store(false, std::sync::atomic::Ordering::Release);
    }

    #[test]
    fn test_ir_dialog_state_outlives_plugin_drop() {
        let state = Arc::new(super::super::dialog_state::IrDialogSharedState {
            pending_ir: std::sync::Mutex::new(None),
            ir_loading: AtomicBool::new(true),
        });
        let state_clone = Arc::clone(&state);

        drop(state);

        state_clone
            .ir_loading
            .store(false, std::sync::atomic::Ordering::Release);
    }

    #[test]
    #[ignore = "rfd not available in headless CI; covered by clap-validator integration tests"]
    fn test_spawn_returns_joinhandle() {}
}
