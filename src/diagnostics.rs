// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Sistema de diagnósticos estruturados do NAM-rs.
//!
//! Fornece mensagens de erro em duas camadas:
//! 1. **Mensagem amigável** — texto legível para o usuário final (pt-BR).
//! 2. **Bloco de suporte** — código de erro tipado, parâmetros contextuais,
//!    versão, arquitetura e timestamp para triagem precisa por devs/IA.
//!
//! ## Uso fora da thread RT
//!
//! Toda formatação e impressão de diagnósticos ocorre **exclusivamente** em
//! threads não-RT (CLI, loop principal do PipeWire). O callback `process()`
//! continua usando flags atômicas (`RtStatusFlags`) para sinalização silenciosa.

use colored::Colorize;
use std::fmt;
use std::time::SystemTime;

/// Versão do binário, injetada em tempo de compilação.
const NAM_VERSION: &str = env!("CARGO_PKG_VERSION");

// =============================================================================
// Códigos de Erro — Catálogo Tipado
// =============================================================================

/// Código de erro estruturado para triagem precisa de falhas.
///
/// Cada variante mapeia a exatamente um cenário de falha. O código numérico
/// (usado no bloco de suporte) segue a convenção:
/// - **E1xxx**: Carregamento de modelo (I/O, parse, build)
/// - **E2xxx**: PipeWire / áudio (init, stream, resampler, RT)
/// - **E3xxx**: SPSC / comunicação inter-thread
/// - **E4xxx**: Runtime / CLI (parsing, comandos)
/// - **E5xxx**: Sistema / hardware (CPU features, memória)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamErrorCode {
    // E1xxx — Carregamento de Modelo
    /// Arquivo de modelo não encontrado no sistema de arquivos.
    FileNotFound,
    /// Erro de leitura de I/O ao acessar o arquivo de modelo.
    FileReadError,
    /// Extensão de arquivo desconhecida (esperado .nam ou .namb).
    UnknownExtension,
    /// Falha ao fazer parse do JSON no formato .nam.
    NamJsonParseError,
    /// Checksum CRC32 do .namb não confere com o esperado.
    NambCrc32Mismatch,
    /// Assinatura mágica do .namb inválida.
    NambInvalidMagic,
    /// Versão do formato .namb não suportada.
    NambUnsupportedVersion,
    /// Arquivo .namb truncado (menor que o cabeçalho mínimo).
    NambTruncated,
    /// Arquitetura neural declarada não é suportada (nem WaveNet nem LSTM).
    UnsupportedArchitecture,
    /// Topologia WaveNet/LSTM não detectável a partir da configuração.
    TopologyDetectionFailed,
    /// Quantidade de pesos inconsistente com a geometria da rede.
    WeightCountMismatch,
    /// Falha genérica na construção do modelo pelo dispatcher.
    ModelBuildFailed,

    // E2xxx — PipeWire / Áudio
    /// Falha ao inicializar o PipeWire ou criar o contexto/core.
    PipewireInitFailed,
    /// Falha ao conectar o stream do PipeWire.
    StreamConnectFailed,
    /// Falha ao construir o NamResampler (rubato).
    ResamplerBuildFailed,
    /// Canal SPSC de resamplers cheio (rebuild descartado).
    ResamplerChannelFull,
    /// Permissão negada para SCHED_FIFO (sem RT privileges).
    SchedFifoDenied,
    /// Falha ao definir afinidade de CPU na thread DSP.
    CpuAffinityFailed,

    // E3xxx — SPSC / Comunicação
    /// Canal SPSC de parâmetros CLI→DSP cheio.
    ParamChannelFull,

    // E4xxx — Runtime / CLI
    /// Valor de ganho inválido (não é um número f32 válido).
    InvalidGainValue,
    /// Comando CLI desconhecido.
    UnknownCommand,
    /// Falha ao configurar o handler de Ctrl-C.
    CtrlCHandlerFailed,
}

impl NamErrorCode {
    /// Retorna o código numérico no formato "Exxxx".
    pub fn code(self) -> &'static str {
        match self {
            Self::FileNotFound => "E1100",
            Self::FileReadError => "E1101",
            Self::UnknownExtension => "E1102",
            Self::NamJsonParseError => "E1200",
            Self::NambCrc32Mismatch => "E1201",
            Self::NambInvalidMagic => "E1202",
            Self::NambUnsupportedVersion => "E1203",
            Self::NambTruncated => "E1204",
            Self::UnsupportedArchitecture => "E1300",
            Self::TopologyDetectionFailed => "E1301",
            Self::WeightCountMismatch => "E1302",
            Self::ModelBuildFailed => "E1303",
            Self::PipewireInitFailed => "E2100",
            Self::StreamConnectFailed => "E2101",
            Self::ResamplerBuildFailed => "E2200",
            Self::ResamplerChannelFull => "E2201",
            Self::SchedFifoDenied => "E2300",
            Self::CpuAffinityFailed => "E2301",
            Self::ParamChannelFull => "E3100",
            Self::InvalidGainValue => "E4100",
            Self::UnknownCommand => "E4101",
            Self::CtrlCHandlerFailed => "E4102",
        }
    }

    /// Retorna o nome mnemônico legível (SCREAMING_SNAKE_CASE) do código.
    pub fn mnemonic(self) -> &'static str {
        match self {
            Self::FileNotFound => "FILE_NOT_FOUND",
            Self::FileReadError => "FILE_READ_ERROR",
            Self::UnknownExtension => "UNKNOWN_EXTENSION",
            Self::NamJsonParseError => "NAM_JSON_PARSE_ERROR",
            Self::NambCrc32Mismatch => "NAMB_CRC32_MISMATCH",
            Self::NambInvalidMagic => "NAMB_INVALID_MAGIC",
            Self::NambUnsupportedVersion => "NAMB_UNSUPPORTED_VERSION",
            Self::NambTruncated => "NAMB_TRUNCATED",
            Self::UnsupportedArchitecture => "UNSUPPORTED_ARCHITECTURE",
            Self::TopologyDetectionFailed => "TOPOLOGY_DETECTION_FAILED",
            Self::WeightCountMismatch => "WEIGHT_COUNT_MISMATCH",
            Self::ModelBuildFailed => "MODEL_BUILD_FAILED",
            Self::PipewireInitFailed => "PIPEWIRE_INIT_FAILED",
            Self::StreamConnectFailed => "STREAM_CONNECT_FAILED",
            Self::ResamplerBuildFailed => "RESAMPLER_BUILD_FAILED",
            Self::ResamplerChannelFull => "RESAMPLER_CHANNEL_FULL",
            Self::SchedFifoDenied => "SCHED_FIFO_DENIED",
            Self::CpuAffinityFailed => "CPU_AFFINITY_FAILED",
            Self::ParamChannelFull => "PARAM_CHANNEL_FULL",
            Self::InvalidGainValue => "INVALID_GAIN_VALUE",
            Self::UnknownCommand => "UNKNOWN_COMMAND",
            Self::CtrlCHandlerFailed => "CTRL_C_HANDLER_FAILED",
        }
    }
}

impl fmt::Display for NamErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} | {}", self.code(), self.mnemonic())
    }
}

// =============================================================================
// SystemSnapshot — Informações de ambiente capturadas uma vez no startup
// =============================================================================

/// Snapshot do ambiente de execução, capturado uma vez no startup.
///
/// Inclui informações estáticas do sistema que são úteis para triagem:
/// versão do NAM-rs, arquitetura, OS, features de CPU, versão do kernel.
#[derive(Debug, Clone)]
pub struct SystemSnapshot {
    /// Versão do NAM-rs (e.g. "0.7.0").
    pub version: &'static str,
    /// Arquitetura do CPU (e.g. "x86_64").
    pub arch: &'static str,
    /// Sistema operacional (e.g. "linux").
    pub os: &'static str,
    /// Versão do kernel Linux (lida de /proc/version).
    pub kernel: String,
}

impl SystemSnapshot {
    /// Captura as informações do sistema no momento da chamada.
    ///
    /// Deve ser chamada uma vez no `main()` e o resultado propagado
    /// para as funções que emitem diagnósticos.
    pub fn capture() -> Self {
        let kernel = read_kernel_version();

        Self {
            version: NAM_VERSION,
            arch: std::env::consts::ARCH,
            os: std::env::consts::OS,
            kernel,
        }
    }

    /// Emite aviso informativo recomendando o isolamento de IRQs do núcleo afixado
    /// pelo PipeWire, de modo a minimizar a preempção em tempo real.
    pub fn emit_irq_advisory(&self, core: usize) {
        let mask = 0xFFFFFFFFu32 ^ (1u32 << core);
        log::info!(
            "{} Para latência mínima, isole o core {} das IRQs do sistema (como sudo): echo {:X} > /proc/irq/default_smp_affinity",
            "💡".yellow(),
            core,
            mask
        );
    }
}

/// Lê a versão do kernel Linux a partir de `/proc/version`.
fn read_kernel_version() -> String {
    std::fs::read_to_string("/proc/version")
        .ok()
        .and_then(|v| {
            // Formato: "Linux version 6.8.0-generic (...)"
            v.split_whitespace().nth(2).map(String::from)
        })
        .unwrap_or_else(|| "unknown".to_string())
}

// =============================================================================
// NamDiagnostic — Diagnóstico Estruturado
// =============================================================================

/// Diagnóstico estruturado para erros e avisos do NAM-rs.
///
/// Combina uma mensagem amigável para o usuário com um bloco técnico
/// copiável para suporte. Formatado para triagem precisa.
pub struct NamDiagnostic {
    /// Código de erro tipado.
    code: NamErrorCode,
    /// Mensagem amigável para o usuário (pt-BR).
    user_message: String,
    /// Sugestão de ação para o usuário (pt-BR).
    user_hint: String,
    /// Parâmetros contextuais para o bloco de suporte (chave=valor).
    params: Vec<(&'static str, String)>,
    /// Snapshot do sistema (capturado no startup).
    system: SystemSnapshot,
}

impl NamDiagnostic {
    /// Cria um novo diagnóstico com o código de erro e snapshot do sistema.
    pub fn new(code: NamErrorCode, system: &SystemSnapshot) -> Self {
        Self {
            code,
            user_message: String::new(),
            user_hint: String::new(),
            params: Vec::new(),
            system: system.clone(),
        }
    }

    /// Define a mensagem amigável para o usuário.
    pub fn message(mut self, msg: impl Into<String>) -> Self {
        self.user_message = msg.into();
        self
    }

    /// Define a sugestão de ação para o usuário.
    pub fn hint(mut self, hint: impl Into<String>) -> Self {
        self.user_hint = hint.into();
        self
    }

    /// Adiciona um parâmetro contextual ao bloco de suporte.
    pub fn param(mut self, key: &'static str, value: impl fmt::Display) -> Self {
        self.params.push((key, value.to_string()));
        self
    }

    /// Gera o timestamp ISO 8601 do momento atual.
    fn timestamp() -> String {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();

        // Formatação manual ISO 8601 sem dependências externas
        let secs = now.as_secs();
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Cálculo da data a partir de dias desde epoch (1970-01-01)
        let (year, month, day) = days_to_date(days);

        format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
    }

    /// Gera o bloco de suporte técnico formatado para cópia.
    fn support_block(&self) -> String {
        let separator = "────────────────────────────────────────────────";
        let mut block = format!(
            "──── NAM-rs Diagnostic {separator}\n\
             nam-rs v{} | {} | {}\n",
            self.system.version,
            self.code.code(),
            self.code.mnemonic(),
        );

        // Parâmetros contextuais
        if !self.params.is_empty() {
            let param_line: String = self
                .params
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ");
            block.push_str(&param_line);
            block.push('\n');
        }

        // Informações do sistema
        block.push_str(&format!(
            "arch={}\n\
             os={} kernel={}\n\
             timestamp={}\n\
             {separator}\n\
             Copie o bloco acima ao abrir um ticket de suporte.",
            self.system.arch,
            self.system.os,
            self.system.kernel,
            Self::timestamp(),
        ));

        block
    }

    /// Imprime o diagnóstico completo no stderr (mensagem amigável + bloco de suporte).
    pub fn emit(&self) {
        log::error!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }

    /// Imprime como aviso (não-fatal) com prefixo visual diferenciado.
    pub fn emit_warning(&self) {
        log::warn!(
            "{}\n\n{}\n\n{}",
            self.user_message,
            if self.user_hint.is_empty() {
                ""
            } else {
                &self.user_hint
            },
            self.support_block()
        );
    }
}

impl fmt::Display for NamDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code.code(), self.user_message)
    }
}

// =============================================================================
// Helpers — Conversão de dias epoch → data gregoriana
// =============================================================================

/// Converte dias desde 1970-01-01 em (ano, mês, dia) gregoriano.
///
/// Implementação mínima do algoritmo civil de Howard Hinnant,
/// sem dependências externas.
fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algoritmo civil: https://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

// =============================================================================
// Testes Unitários
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_display() {
        let code = NamErrorCode::NambCrc32Mismatch;
        assert_eq!(code.code(), "E1201");
        assert_eq!(code.mnemonic(), "NAMB_CRC32_MISMATCH");
        assert_eq!(format!("{}", code), "E1201 | NAMB_CRC32_MISMATCH");
    }

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

    #[test]
    fn test_system_snapshot_capture() {
        let snap = SystemSnapshot::capture();
        assert!(!snap.version.is_empty());
        assert!(!snap.arch.is_empty());
        assert!(!snap.os.is_empty());
    }

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

    #[test]
    fn test_diagnostic_display() {
        let snap = SystemSnapshot::capture();
        let diag =
            NamDiagnostic::new(NamErrorCode::FileNotFound, &snap).message("Modelo não encontrado");
        assert_eq!(format!("{}", diag), "[E1100] Modelo não encontrado");
    }

    #[test]
    fn test_days_to_date_epoch() {
        let (y, m, d) = days_to_date(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn test_days_to_date_known() {
        // 2026-04-10 = ~20553 days since epoch
        // (1970-01-01 + 20553 = 2026-04-10)
        let (y, m, d) = days_to_date(20553);
        assert_eq!(y, 2026);
        assert_eq!(m, 4);
        assert_eq!(d, 10);
    }

    #[test]
    fn test_timestamp_format() {
        let ts = NamDiagnostic::timestamp();
        // Verifica formato ISO 8601: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(ts.len(), 20, "Timestamp deve ter 20 caracteres");
        assert!(ts.ends_with('Z'), "Timestamp deve terminar em Z");
        assert_eq!(&ts[4..5], "-", "Separador de data");
        assert_eq!(&ts[10..11], "T", "Separador T");
    }
}
