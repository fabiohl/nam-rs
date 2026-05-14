// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Valida a formatação visual e consistência do código de erro e seu mnemônico.
#[test]
fn test_error_code_display() {
    let code = NamErrorCode::NambCrc32Mismatch;
    assert_eq!(code.code(), "E1201");
    assert_eq!(code.mnemonic(), "NAMB_CRC32_MISMATCH");
    assert_eq!(format!("{}", code), "E1201 | NAMB_CRC32_MISMATCH");
}

/// Garante a integridade do registro de erros, impedindo duplicidade de códigos numéricos (E####).
/// Essencial para manter a estabilidade de logs e protocolos de suporte.
#[test]
fn test_all_codes_have_unique_numeric() {
    use NamErrorCode::*;
    let all = [
        FileNotFound,
        FileReadError,
        UnknownExtension,
        NamJsonParseError,
        NambCrc32Mismatch,
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
        DeadlineExceeded, // Adicionado conforme T17
    ];

    let codes: Vec<&str> = all.iter().map(|c| c.code()).collect();
    let mut sorted = codes.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        codes.len(),
        sorted.len(),
        "Códigos numéricos duplicados detectados"
    );
}

/// Verifica se a captura de informações do sistema (OS, Arch, Versão) está ativa e retornando dados válidos.
#[test]
fn test_system_snapshot_capture() {
    let snap = SystemSnapshot::capture();
    assert!(!snap.version.is_empty());
    assert!(!snap.arch.is_empty());
    assert!(!snap.os.is_empty());
}

/// Valida a geração do "bloco de suporte" para o usuário, garantindo que contenha o erro,
/// mnemônico, parâmetros contextuais e instruções de cópia.
#[test]
fn test_diagnostic_support_block_contains_code() {
    let snap = SystemSnapshot::capture();
    let diag = NamDiagnostic::new(NamErrorCode::NambCrc32Mismatch, &snap)
        .message("Arquivo corrompido")
        .hint("Baixe novamente")
        .param("expected", "0xDEADBEEF")
        .param("computed", "0x12345678");

    let block = diag.support_block();
    assert!(block.contains("E1201"), "Bloco deve conter o código");
    assert!(
        block.contains("NAMB_CRC32_MISMATCH"),
        "Bloco deve conter o mnemônico"
    );
    assert!(
        block.contains("expected=0xDEADBEEF"),
        "Bloco deve conter parâmetros"
    );
    assert!(
        block.contains("computed=0x12345678"),
        "Bloco deve conter parâmetros"
    );
    assert!(
        block.contains("Copie o bloco acima"),
        "Bloco deve conter instrução de cópia"
    );
}

/// Testa a representação simplificada do diagnóstico para exibição direta via trait Display.
#[test]
fn test_diagnostic_display() {
    let snap = SystemSnapshot::capture();
    let diag =
        NamDiagnostic::new(NamErrorCode::FileNotFound, &snap).message("Modelo não encontrado");
    assert_eq!(format!("{}", diag), "[E1100] Modelo não encontrado");
}

/// Valida o algoritmo manual de conversão de dias desde o epoch para data (Ano, Mês, Dia),
/// permitindo manter o binário leve sem dependências externas de tempo (como chrono).
#[test]
fn test_days_to_date_epoch() {
    let (y, m, d) = days_to_date(0);
    assert_eq!((y, m, d), (1970, 1, 1));
}

/// Testa a precisão do algoritmo de data com uma data conhecida (2026-04-10).
#[test]
fn test_days_to_date_known() {
    // 2026-04-10 = ~20553 days since epoch
    // (1970-01-01 + 20553 = 2026-04-10)
    let (y, m, d) = days_to_date(20553);
    assert_eq!(y, 2026);
    assert_eq!(m, 4);
    assert_eq!(d, 10);
}

/// Garante que o timestamp gerado segue rigorosamente o padrão ISO 8601 (YYYY-MM-DDTHH:MM:SSZ).
#[test]
fn test_timestamp_format() {
    let ts = NamDiagnostic::timestamp();
    // Verifica formato ISO 8601: YYYY-MM-DDTHH:MM:SSZ
    assert_eq!(ts.len(), 20, "Timestamp deve ter 20 caracteres");
    assert!(ts.ends_with('Z'), "Timestamp deve terminar em Z");
    assert_eq!(&ts[4..5], "-", "Separador de data");
    assert_eq!(&ts[10..11], "T", "Separador T");
}

/// Garante que o cálculo da máscara de IRQ não causa panic em sistemas com muitos cores (Bug B5).
#[test]
fn test_emit_irq_advisory_safety() {
    let snap = SystemSnapshot::capture();
    // Teste de cores baixos, limite de u32, e limite de u64
    snap.emit_irq_advisory(0);
    snap.emit_irq_advisory(31);
    snap.emit_irq_advisory(32);
    snap.emit_irq_advisory(63);
    snap.emit_irq_advisory(64); // Deve retornar via guarda sem panic
    snap.emit_irq_advisory(128); // Caso extremo
}
