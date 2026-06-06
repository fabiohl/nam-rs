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
