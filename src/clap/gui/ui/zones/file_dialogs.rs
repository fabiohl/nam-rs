// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//! Native file dialog helpers for model and IR loading (asynchronous, off-GUI-thread).

use crate::clap::plugin::NamClapShared;
use clack_plugin::host::HostSharedHandle;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) fn spawn_file_dialog(
    shared_addr: usize,
    host_static: HostSharedHandle<'static>,
    alive_fence: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let path_opt = rfd::FileDialog::new()
                .add_filter("NAM Model", &["nam", "namb"])
                .pick_file();
            let _ = tx.send(path_opt);
        });

        match rx.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(path_opt) => {
                if alive_fence.load(Ordering::Relaxed) {
                    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                    if let Some(path) = path_opt {
                        if let Ok(mut pending_guard) = shared.cold.ui_pending_model.lock() {
                            *pending_guard = Some(path);
                            host_static.request_callback();
                        }
                    } else {
                        shared.cold.ui_loading.store(false, Ordering::Relaxed);
                    }
                }
            }
            Err(_) => {
                if alive_fence.load(Ordering::Relaxed) {
                    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                    shared.cold.ui_loading.store(false, Ordering::Relaxed);

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
        }
    });
}

pub(crate) fn spawn_ir_file_dialog(
    shared_addr: usize,
    host_static: HostSharedHandle<'static>,
    alive_fence: Arc<AtomicBool>,
) {
    std::thread::spawn(move || {
        let (tx, rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let path_opt = rfd::FileDialog::new()
                .add_filter("WAV Impulse Response", &["wav"])
                .pick_file();
            let _ = tx.send(path_opt);
        });

        match rx.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(path_opt) => {
                if alive_fence.load(Ordering::Relaxed) {
                    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                    if let Some(path) = path_opt {
                        if let Ok(mut pending_guard) = shared.cold.ui_pending_ir.lock() {
                            *pending_guard = Some(path);
                            host_static.request_callback();
                        }
                    } else {
                        shared.cold.ui_ir_loading.store(false, Ordering::Relaxed);
                    }
                }
            }
            Err(_) => {
                if alive_fence.load(Ordering::Relaxed) {
                    // SAFETY: FFI call, host pointer transmute, or raw graphics context access with verified lifetimes.
                    let shared = unsafe { &*(shared_addr as *const NamClapShared) };
                    shared.cold.ui_ir_loading.store(false, Ordering::Relaxed);

                    if let (Some(log), Ok(c_msg)) = (
                        host_static.get_extension::<clack_extensions::log::HostLog>(),
                        std::ffi::CString::new(
                            "NAM-rs: IR file dialog portal timed out after 120s",
                        ),
                    ) {
                        log.log(
                            &host_static,
                            clack_extensions::log::LogSeverity::Warning,
                            &c_msg,
                        );
                    }
                }
            }
        }
    });
}
