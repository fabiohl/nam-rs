// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Panic hook facility to capture and write diagnostic bundles to disk on panic.

use crate::common::diagnostics::DiagnosticBundle;
use std::cell::Cell;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::SystemTime;

static SHUTDOWN_IN_PROGRESS: OnceLock<bool> = OnceLock::new();

thread_local! {
    static PANIC_HOOK_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Sets the global flag signaling that shutdown is in progress.
///
/// This bypasses the panic hook to avoid race conditions or deadlock
/// during host-initiated cleanup.
pub fn set_shutdown_in_progress() {
    let _ = SHUTDOWN_IN_PROGRESS.set(true);
}

/// Checks if a shutdown (host-initiated or signal-initiated) is in progress.
pub fn is_shutdown_in_progress() -> bool {
    crate::common::spsc::SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed)
        || *SHUTDOWN_IN_PROGRESS.get().unwrap_or(&false)
}

/// Installs a panic hook that logs a full crash report containing a
/// `DiagnosticBundle` to `~/.cache/nam-rs/crash-<unix_ts>-<component>.txt`.
pub fn install_panic_hook(component: &'static str) {
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if is_shutdown_in_progress() {
            prev_hook(info);
            return;
        }

        if PANIC_HOOK_ACTIVE.replace(true) {
            prev_hook(info);
            return;
        }

        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("unknown");

        let payload = info.payload();
        let payload_str = if let Some(s) = payload.downcast_ref::<&str>() {
            *s
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.as_str()
        } else {
            "Box<dyn Any>"
        };

        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "unknown".to_string());

        let bundle_rendered = std::panic::catch_unwind(|| DiagnosticBundle::capture().render());

        let mut report = format!(
            "================================================================\n\
             NAM-rs CRASH REPORT\n\
             ================================================================\n\
             Component: {}\n\
             Thread: {}\n\
             Location: {}\n\
             Message: {}\n\n",
            component, thread_name, location, payload_str
        );

        match bundle_rendered {
            Ok(rendered) => report.push_str(&rendered),
            Err(_) => {
                report.push_str("Failed to capture DiagnosticBundle (panic during capture).\n")
            }
        }

        if let Some(home_dir) = std::env::var_os("HOME") {
            let mut cache_dir = PathBuf::from(home_dir);
            cache_dir.push(".cache/nam-rs");

            if std::fs::create_dir_all(&cache_dir).is_ok() {
                #[cfg(unix)]
                {
                    use std::fs::Permissions;
                    use std::os::unix::fs::PermissionsExt;
                    let _ = std::fs::set_permissions(&cache_dir, Permissions::from_mode(0o700));
                }

                let unix_ts = SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                let filename = format!("crash-{}-{}.txt", unix_ts, component);
                let tmp_filename = format!("crash-{}-{}.tmp", unix_ts, component);

                let file_path = cache_dir.join(filename);
                let tmp_file_path = cache_dir.join(tmp_filename);

                let mut options = OpenOptions::new();
                options.write(true).create(true).truncate(true);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::OpenOptionsExt;
                    options.mode(0o600);
                }

                if let Ok(mut file) = options.open(&tmp_file_path) {
                    if file.write_all(report.as_bytes()).is_ok() {
                        let _ = file.sync_all();
                        drop(file);
                        let _ = std::fs::rename(&tmp_file_path, &file_path);
                    } else {
                        let _ = std::fs::remove_file(&tmp_file_path);
                    }
                }
            }
        }

        PANIC_HOOK_ACTIVE.set(false);
        prev_hook(info);
    }));
}
