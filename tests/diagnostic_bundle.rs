// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration tests for DiagnosticBundle and RuntimeSnapshot.

use nam_rs::common::diagnostics::{
    AudioInfo, DiagnosticBundle, HasRuntimeSnapshot, ModelInfo, RtInfo, RuntimeSnapshot,
    TelemetrySnapshot,
};
use nam_rs::common::spsc::RtStatusFlags;
use std::sync::atomic::Ordering;

struct MockSnapshotProvider {
    model: Option<ModelInfo>,
    audio: AudioInfo,
    rt: RtInfo,
    telemetry: TelemetrySnapshot,
    flags: u64,
}

impl HasRuntimeSnapshot for MockSnapshotProvider {
    fn model_info(&self) -> Option<ModelInfo> {
        self.model.clone()
    }

    fn audio_info(&self) -> AudioInfo {
        self.audio.clone()
    }

    fn rt_info(&self) -> RtInfo {
        self.rt.clone()
    }

    fn telemetry_snapshot(&self) -> TelemetrySnapshot {
        self.telemetry.clone()
    }

    fn flags_seen(&self) -> u64 {
        self.flags
    }
}

#[test]
fn test_diagnostic_bundle_with_mock_provider() {
    let provider = MockSnapshotProvider {
        model: Some(ModelInfo {
            arch_label: "WaveNet".to_string(),
            channels: 16,
            receptive_field: 2048,
            weights_layout: "Interleaved4WaveNet".to_string(),
            path_basename: "test_model.nam".to_string(),
        }),
        audio: AudioInfo {
            sample_rate: 48000,
            buffer_size: 256,
            channel_count: 2,
            host_name: "CLAP".to_string(),
        },
        rt: RtInfo {
            thread_priority: 90,
            scheduler: "FIFO".to_string(),
            cpu_pinned: Some(3),
            huge_pages_active: true,
        },
        telemetry: TelemetrySnapshot {
            p50_us: 120,
            p99_us: 250,
            p999_us: 400,
            max_us: 850,
            total_blocks: 10000,
            xruns: 2,
            drains: 5,
        },
        flags: 0x1a,
    };

    let bundle = DiagnosticBundle::capture_with_runtime(&provider);
    let rendered = bundle.render();

    // Verify model info
    assert!(rendered.contains("model.arch=WaveNet"));
    assert!(rendered.contains("model.channels=16"));
    assert!(rendered.contains("model.receptive_field=2048"));
    assert!(rendered.contains("model.weights_layout=Interleaved4WaveNet"));
    assert!(rendered.contains("model.path_basename=test_model.nam"));

    // Verify audio info
    assert!(rendered.contains("audio.sr=48000"));
    assert!(rendered.contains("audio.buffer_size=256"));
    assert!(rendered.contains("audio.channel_count=2"));
    assert!(rendered.contains("audio.host_name=CLAP"));

    // Verify RT info
    assert!(rendered.contains("rt.prio=90"));
    assert!(rendered.contains("rt.scheduler=FIFO"));
    assert!(rendered.contains("rt.cpu_pinned=3"));
    assert!(rendered.contains("rt.huge_pages_active=true"));

    // Verify Telemetry info
    assert!(rendered.contains("telemetry.p50_us=120"));
    assert!(rendered.contains("telemetry.p99_us=250"));
    assert!(rendered.contains("telemetry.p999_us=400"));
    assert!(rendered.contains("telemetry.max_us=850"));
    assert!(rendered.contains("telemetry.total_blocks=10000"));
    assert!(rendered.contains("telemetry.xruns=2"));
    assert!(rendered.contains("telemetry.drains=5"));

    // Verify flags seen in hex format
    assert!(rendered.contains("flags_seen=0x1a"));
}

#[test]
fn test_rt_status_flags_provider_defaults() {
    let rt_status = RtStatusFlags::new();
    let snapshot = RuntimeSnapshot::capture(&rt_status);

    assert_eq!(snapshot.audio.sample_rate, 0);
    assert_eq!(snapshot.audio.buffer_size, 0);
    assert_eq!(snapshot.audio.host_name, "PipeWire");
    assert_eq!(snapshot.rt.thread_priority, -1);
    assert_eq!(snapshot.rt.scheduler, "UNKNOWN");
    assert_eq!(snapshot.rt.cpu_pinned, None);
    assert!(!snapshot.rt.huge_pages_active);
    assert_eq!(snapshot.telemetry.xruns, 0);
    assert_eq!(snapshot.telemetry.drains, 0);
    assert_eq!(snapshot.flags_seen, 0);
}

#[test]
fn test_rt_status_flags_provider_populated() {
    let rt_status = RtStatusFlags::new();
    rt_status.active_rate.store(44100, Ordering::Relaxed);
    rt_status.last_n_samples.store(512, Ordering::Relaxed);
    rt_status.confirmed_priority.store(85, Ordering::Relaxed);
    rt_status
        .rt_policy
        .store(libc::SCHED_FIFO, Ordering::Relaxed);
    rt_status.rt_cpu.store(2, Ordering::Relaxed);
    rt_status.xruns.store(3, Ordering::Relaxed);
    rt_status.drains.store(7, Ordering::Relaxed);
    rt_status.flags_seen.store(0x4b, Ordering::Relaxed);

    let snapshot = RuntimeSnapshot::capture(&rt_status);

    assert_eq!(snapshot.audio.sample_rate, 44100);
    assert_eq!(snapshot.audio.buffer_size, 512);
    assert_eq!(snapshot.rt.thread_priority, 85);
    assert_eq!(snapshot.rt.scheduler, "FIFO");
    assert_eq!(snapshot.rt.cpu_pinned, Some(2));
    assert_eq!(snapshot.telemetry.xruns, 3);
    assert_eq!(snapshot.telemetry.drains, 7);
    assert_eq!(snapshot.flags_seen, 0x4b);
}

#[test]
fn test_panic_hook_behavior() {
    use std::fs;
    use std::path::PathBuf;

    let home = match std::env::var_os("HOME") {
        Some(h) => PathBuf::from(h),
        None => return,
    };
    let cache_dir = home.join(".cache/nam-rs");
    let _ = fs::create_dir_all(&cache_dir);

    // Part 1: Test persistence when shutdown is NOT in progress.
    let component_name = "test-panic-persistence";

    // Clear old files
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_name))
                    .unwrap_or(false);
            if is_target {
                let _ = fs::remove_file(path);
            }
        }
    }

    let prev_hook = std::panic::take_hook();
    nam_rs::common::panic_hook::install_panic_hook(component_name);

    let result = std::panic::catch_unwind(|| {
        panic!("Controlled testing panic message");
    });
    assert!(result.is_err());

    std::panic::set_hook(prev_hook);

    // Verify report
    let mut found_report = None;
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_name) && name.ends_with(".txt"))
                    .unwrap_or(false);
            if is_target {
                found_report = Some(path);
                break;
            }
        }
    }

    let report_path = found_report.expect("Crash report file should be created");
    let content = fs::read_to_string(&report_path).expect("Should read report content");

    assert!(content.contains("NAM-rs CRASH REPORT"));
    assert!(content.contains(&format!("Component: {}", component_name)));
    assert!(content.contains("Location:"));
    assert!(content.contains("Message: Controlled testing panic message"));
    assert!(content.contains("──── Runtime State ─────────────────────────────"));
    assert!(content.contains("arch="));
    assert!(content.contains("os="));

    let _ = fs::remove_file(report_path);

    // Part 2: Test bypass when shutdown IS in progress.
    let component_bypass = "test-panic-bypass";

    // Clear old files
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_bypass))
                    .unwrap_or(false);
            if is_target {
                let _ = fs::remove_file(path);
            }
        }
    }

    nam_rs::common::panic_hook::set_shutdown_in_progress();
    assert!(nam_rs::common::panic_hook::is_shutdown_in_progress());

    let prev_hook = std::panic::take_hook();
    nam_rs::common::panic_hook::install_panic_hook(component_bypass);

    let result = std::panic::catch_unwind(|| {
        panic!("Controlled panic during shutdown");
    });
    assert!(result.is_err());

    std::panic::set_hook(prev_hook);

    // Verify NO report was created
    let mut found_bypass_report = false;
    if let Ok(entries) = fs::read_dir(&cache_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_target = path.is_file()
                && path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|name| name.contains(component_bypass))
                    .unwrap_or(false);
            if is_target {
                found_bypass_report = true;
                let _ = fs::remove_file(path);
            }
        }
    }

    assert!(
        !found_bypass_report,
        "No report should be written if shutdown is in progress"
    );
}

#[test]
fn test_diagnostic_bundle_path_redaction() {
    // Save current env vars to restore later if modified, or just use them.
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/mockuser".to_string());
    let xdg = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/run/user/1000".to_string());

    // We override standard env vars if they are empty for the test
    if std::env::var("HOME").is_err() {
        unsafe {
            std::env::set_var("HOME", &home);
        }
    }
    if std::env::var("XDG_RUNTIME_DIR").is_err() {
        unsafe {
            std::env::set_var("XDG_RUNTIME_DIR", &xdg);
        }
    }

    let home_path = format!("{}/nam-rs/models/test_model.nam", home);
    let xdg_path = format!("{}/pipewire-0", xdg);

    // Mock provider with a full path in model.path_basename
    let provider = MockSnapshotProvider {
        model: Some(ModelInfo {
            arch_label: "WaveNet".to_string(),
            channels: 16,
            receptive_field: 2048,
            weights_layout: "Interleaved4WaveNet".to_string(),
            path_basename: home_path.clone(),
        }),
        audio: AudioInfo {
            sample_rate: 48000,
            buffer_size: 256,
            channel_count: 2,
            host_name: "CLAP".to_string(),
        },
        rt: RtInfo {
            thread_priority: 90,
            scheduler: "FIFO".to_string(),
            cpu_pinned: Some(3),
            huge_pages_active: true,
        },
        telemetry: TelemetrySnapshot::default(),
        flags: 0,
    };

    // Case A: Default capture (redacted)
    let bundle_default = DiagnosticBundle::capture_with_runtime(&provider);
    let rendered_default = bundle_default.render();

    // Verifications for default capture
    // 1. Should NOT contain raw HOME path
    assert!(!rendered_default.contains(&home_path));
    // 2. model.path_basename should be formatted to just the basename
    assert!(rendered_default.contains("model.path_basename=test_model.nam"));
    // 3. model should also print only the basename
    assert!(rendered_default.contains("model=test_model.nam"));

    // Now test parameter redaction. Let's capture with an error containing paths
    let params = vec![
        ("model_path", home_path.clone()),
        ("socket_path", xdg_path.clone()),
    ];
    let bundle_err = DiagnosticBundle::capture_with_error(
        nam_rs::common::diagnostics::NamErrorCode::ModelBuildFailed,
        params,
    );
    let rendered_err_default = bundle_err.render();

    // Verifications for default error parameters
    assert!(rendered_err_default.contains("model_path=~/nam-rs/models/test_model.nam"));
    assert!(rendered_err_default.contains("socket_path=$XDG_RUNTIME_DIR/pipewire-0"));
    assert!(!rendered_err_default.contains(&home_path));
    assert!(!rendered_err_default.contains(&xdg_path));

    // Case B: Full capture (unredacted / bruto)
    let bundle_full = DiagnosticBundle::capture_with_runtime(&provider).with_full(true);
    let rendered_full = bundle_full.render();

    // Verifications for full capture
    // 1. Should contain raw HOME path
    assert!(rendered_full.contains(&home_path));
    // 2. model.path_basename should print full path
    assert!(rendered_full.contains(&format!("model.path_basename={}", home_path)));
    // 3. model should print full path
    assert!(rendered_full.contains(&format!("model={}", home_path)));

    // Full error parameters
    let bundle_err_full = DiagnosticBundle::capture_with_error(
        nam_rs::common::diagnostics::NamErrorCode::ModelBuildFailed,
        vec![
            ("model_path", home_path.clone()),
            ("socket_path", xdg_path.clone()),
        ],
    )
    .with_full(true);
    let rendered_err_full = bundle_err_full.render();

    assert!(rendered_err_full.contains(&format!("model_path={}", home_path)));
    assert!(rendered_err_full.contains(&format!("socket_path={}", xdg_path)));
}
