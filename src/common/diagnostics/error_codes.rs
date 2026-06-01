// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Códigos de erro tipados para triagem precisa de falhas.
//!
//! A convenção numérica dos códigos:
//! - **E1xxx**: Carregamento de modelo (I/O, parse, build)
//! - **E2xxx**: PipeWire / áudio (init, stream, resampler, RT)
//! - **E3xxx**: SPSC / comunicação inter-thread
//! - **E4xxx**: Runtime / CLI (parsing, comandos)
//! - **E5xxx**: Sistema / hardware (CPU features, memória)

use std::fmt;

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
    /// Array `weights` do JSON excede o limite de floats (MAX_WEIGHTS).
    NamJsonWeightsExceedLimit,
    /// Campo `metadata.training` do JSON excede limite de tamanho (1 MiB).
    NamJsonTrainingTooLarge,
    /// Campo `metadata.training` do JSON excede limite de profundidade de árvore.
    NamJsonTrainingTooDeep,
    /// Checksum CRC32 do .namb não confere com o esperado.
    NambCrc32Mismatch,
    /// Flag CRC32 ausente em arquivo .namb v2+ (obrigatório).
    NambCrc32Missing,
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
    /// Arquivo de modelo excede o tamanho máximo permitido (256 MiB).
    ModelTooLarge,

    // E2xxx — PipeWire / Áudio
    /// Falha ao inicializar o PipeWire ou criar o contexto/core.
    PipewireInitFailed,
    /// Falha ao conectar o stream do PipeWire.
    StreamConnectFailed,
    /// Falha ao construir o NamResampler (Sinc Polifásico nativo).
    ResamplerBuildFailed,
    /// Canal SPSC de resamplers cheio (rebuild descartado).
    ResamplerChannelFull,
    /// Permissão negada para SCHED_FIFO (sem RT privileges).
    SchedFifoDenied,
    /// Falha ao definir afinidade de CPU na thread DSP.
    CpuAffinityFailed,
    /// Deadline do PipeWire estourado (Processamento DSP excedeu budget).
    DeadlineExceeded,

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
    /// Overflow detectado no canal de Garbage Collection (GC).
    GcOverflow,
}

impl NamErrorCode {
    /// Retorna o código numérico no formato "Exxxx".
    pub fn code(self) -> &'static str {
        match self {
            Self::FileNotFound => "E1100",
            Self::FileReadError => "E1101",
            Self::UnknownExtension => "E1102",
            Self::NamJsonParseError => "E1200",
            Self::NamJsonWeightsExceedLimit => "E1206",
            Self::NamJsonTrainingTooLarge => "E1207",
            Self::NamJsonTrainingTooDeep => "E1208",
            Self::NambCrc32Mismatch => "E1201",
            Self::NambCrc32Missing => "E1205",
            Self::NambInvalidMagic => "E1202",
            Self::NambUnsupportedVersion => "E1203",
            Self::NambTruncated => "E1204",
            Self::UnsupportedArchitecture => "E1300",
            Self::TopologyDetectionFailed => "E1301",
            Self::WeightCountMismatch => "E1302",
            Self::ModelBuildFailed => "E1303",
            Self::ModelTooLarge => "E1304",
            Self::PipewireInitFailed => "E2100",
            Self::StreamConnectFailed => "E2101",
            Self::ResamplerBuildFailed => "E2200",
            Self::ResamplerChannelFull => "E2201",
            Self::SchedFifoDenied => "E2300",
            Self::CpuAffinityFailed => "E2301",
            Self::DeadlineExceeded => "E2001",
            Self::ParamChannelFull => "E3100",
            Self::GcOverflow => "E3101",
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
            Self::NamJsonWeightsExceedLimit => "NAM_JSON_WEIGHTS_EXCEED_LIMIT",
            Self::NamJsonTrainingTooLarge => "NAM_JSON_TRAINING_TOO_LARGE",
            Self::NamJsonTrainingTooDeep => "NAM_JSON_TRAINING_TOO_DEEP",
            Self::NambCrc32Mismatch => "NAMB_CRC32_MISMATCH",
            Self::NambCrc32Missing => "NAMB_CRC32_MISSING",
            Self::NambInvalidMagic => "NAMB_INVALID_MAGIC",
            Self::NambUnsupportedVersion => "NAMB_UNSUPPORTED_VERSION",
            Self::NambTruncated => "NAMB_TRUNCATED",
            Self::UnsupportedArchitecture => "UNSUPPORTED_ARCHITECTURE",
            Self::TopologyDetectionFailed => "TOPOLOGY_DETECTION_FAILED",
            Self::WeightCountMismatch => "WEIGHT_COUNT_MISMATCH",
            Self::ModelBuildFailed => "MODEL_BUILD_FAILED",
            Self::ModelTooLarge => "MODEL_TOO_LARGE",
            Self::PipewireInitFailed => "PIPEWIRE_INIT_FAILED",
            Self::StreamConnectFailed => "STREAM_CONNECT_FAILED",
            Self::ResamplerBuildFailed => "RESAMPLER_BUILD_FAILED",
            Self::ResamplerChannelFull => "RESAMPLER_CHANNEL_FULL",
            Self::SchedFifoDenied => "SCHED_FIFO_DENIED",
            Self::CpuAffinityFailed => "CPU_AFFINITY_FAILED",
            Self::DeadlineExceeded => "DEADLINE_EXCEEDED",
            Self::ParamChannelFull => "PARAM_CHANNEL_FULL",
            Self::GcOverflow => "GC_OVERFLOW",
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
