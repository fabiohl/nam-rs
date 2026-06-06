// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Validates the visual formatting and consistency of the error code and its mnemonic.
#[test]
fn test_error_code_display() {
    let code = NamErrorCode::NambCrc32Mismatch;
    assert_eq!(code.code(), "E1201");
    assert_eq!(code.mnemonic(), "NAMB_CRC32_MISMATCH");
    assert_eq!(format!("{}", code), "E1201 | NAMB_CRC32_MISMATCH");
}

/// Ensures the integrity of the error registry, preventing duplicate numeric codes (E####).
/// Essential for maintaining log stability and support protocols.
#[test]
fn test_all_codes_have_unique_numeric() {
    use NamErrorCode::*;
    let all = [
        FileNotFound,
        FileReadError,
        UnknownExtension,
        NamJsonParseError,
        NamJsonWeightsExceedLimit,
        NamJsonTrainingTooLarge,
        NamJsonTrainingTooDeep,
        NambCrc32Mismatch,
        NambCrc32Missing,
        NambInvalidMagic,
        NambUnsupportedVersion,
        NambTruncated,
        UnsupportedArchitecture,
        TopologyDetectionFailed,
        WeightCountMismatch,
        ModelBuildFailed,
        PipewireInitFailed,
        StreamConnectFailed,
        ResamplerBuildFailed,
        ResamplerChannelFull,
        SchedFifoDenied,
        CpuAffinityFailed,
        ParamChannelFull,
        InvalidGainValue,
        UnknownCommand,
        CtrlCHandlerFailed,
        DeadlineExceeded, // Added as per T17
    ];

    let codes: Vec<&str> = all.iter().map(|c| c.code()).collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        codes.len(),
        sorted.len(),
        "Duplicate numeric codes detected"
    );
}

/// Verifies that system information capture (OS, Arch, Version) is active and returning valid data.
#[test]
fn test_system_snapshot_capture() {
    let snap = SystemSnapshot::capture();
    assert!(!snap.version.is_empty());
    assert!(!snap.arch.is_empty());
    assert!(!snap.os.is_empty());
}

/// Validates the generation of the "support block" for the user, ensuring it contains the error,
/// mnemonic, contextual parameters, and copy instructions.
#[test]
fn test_diagnostic_support_block_contains_code() {
    let snap = SystemSnapshot::capture();
    let diag = NamDiagnostic::new(NamErrorCode::NambCrc32Mismatch, &snap)
        .message("Corrupted file")
        .hint("Download it again")
        .param("expected", "0xDEADBEEF")
        .param("computed", "0x12345678");

    let block = diag.support_block();
    assert!(block.contains("E1201"), "Block must contain the code");
    assert!(
        block.contains("NAMB_CRC32_MISMATCH"),
        "Block must contain the mnemonic"
    );
    assert!(
        block.contains("expected=0xDEADBEEF"),
        "Block must contain parameters"
    );
    assert!(
        block.contains("computed=0x12345678"),
        "Block must contain parameters"
    );
    assert!(
        block.contains("Copy the block above"),
        "Block must contain copy instructions"
    );
}

/// Tests the simplified diagnostic representation for display via the Display trait.
#[test]
fn test_diagnostic_display() {
    let snap = SystemSnapshot::capture();
    let diag = NamDiagnostic::new(NamErrorCode::FileNotFound, &snap).message("Model not found");
    assert_eq!(format!("{}", diag), "[E1100] Model not found");
}

/// Validates the manual algorithm for converting days since epoch to a date (Year, Month, Day),
/// keeping the binary lightweight without external time dependencies (like chrono).
#[test]
fn test_days_to_date_epoch() {
    let (y, m, d) = days_to_date(0);
    assert_eq!((y, m, d), (1970, 1, 1));
}

/// Tests the accuracy of the date algorithm with a known date (2026-04-10).
#[test]
fn test_days_to_date_known() {
    // 2026-04-10 = ~20553 days since epoch
    // (1970-01-01 + 20553 = 2026-04-10)
    let (y, m, d) = days_to_date(20553);
    assert_eq!(y, 2026);
    assert_eq!(m, 4);
    assert_eq!(d, 10);
}

/// Ensures the generated timestamp strictly follows the ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
#[test]
fn test_timestamp_format() {
    let ts = NamDiagnostic::timestamp();
    // Verify ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
    assert_eq!(ts.len(), 20, "Timestamp must be 20 characters long");
    assert!(ts.ends_with('Z'), "Timestamp must end with Z");
    assert_eq!(&ts[4..5], "-", "Date separator");
    assert_eq!(&ts[10..11], "T", "T separator");
}

/// Ensures the IRQ mask calculation does not panic on systems with many cores (Bug B5).
#[test]
fn test_emit_irq_advisory_safety() {
    let snap = SystemSnapshot::capture();
    // Test low cores, u32 boundary, and u64 boundary
    snap.emit_irq_advisory(0);
    snap.emit_irq_advisory(31);
    snap.emit_irq_advisory(32);
    snap.emit_irq_advisory(63);
    snap.emit_irq_advisory(64); // Should return via guard without panic
    snap.emit_irq_advisory(128); // Extreme case
}

/// Verifies that DiagnosticBundle::capture().render() yields a nominal block without error fields.
#[test]
fn test_diagnostic_bundle_nominal() {
    let bundle = DiagnosticBundle::capture();
    let rendered = bundle.render();

    assert!(rendered.contains("NAM-rs Diagnostic"), "Must contain standard header");
    assert!(rendered.contains(&format!("nam-rs v{}", bundle.system.version)), "Must contain version");
    assert!(!rendered.contains("E1"), "Must not contain error codes");
    assert!(!rendered.contains("CRC32"), "Must not contain error mnemonics");
    assert!(rendered.contains("arch="), "Must contain system info");
    assert!(rendered.contains("os="), "Must contain system info");
}

/// Verifies that DiagnosticBundle::capture_with_error() matches NamDiagnostic::support_block() output.
#[test]
fn test_diagnostic_bundle_with_error_matches() {
    let snap = SystemSnapshot::capture();
    let code = NamErrorCode::NambCrc32Mismatch;
    let params = vec![("expected", "0xDEADBEEF".to_string()), ("computed", "0x12345678".to_string())];

    let diag = NamDiagnostic::new(code, &snap)
        .param("expected", "0xDEADBEEF")
        .param("computed", "0x12345678");

    let bundle = DiagnosticBundle::capture_with_error(code, params);

    let block_diag = diag.support_block();
    let block_bundle = bundle.render();

    // Strip timestamp lines to prevent flaky failures on second boundaries
    let lines_diag: Vec<&str> = block_diag.lines().filter(|l| !l.starts_with("timestamp=")).collect();
    let lines_bundle: Vec<&str> = block_bundle.lines().filter(|l| !l.starts_with("timestamp=")).collect();

    assert_eq!(lines_diag, lines_bundle);
}
