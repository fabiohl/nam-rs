// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed errors of the `.nam` JSON parser.

/// Typed errors of the `.nam` JSON parser.
#[derive(Debug)]
pub enum JsonError {
    /// The `weights` array exceeds the float limit.
    WeightsExceedLimit {
        /// Number of floats received.
        got: usize,
        /// Maximum configured limit.
        max: usize,
    },
    /// The `metadata.training` field exceeds the JSON tree depth limit.
    TrainingTooDeep {
        /// Depth found.
        depth: usize,
        /// Maximum allowed depth.
        max_depth: usize,
    },
    /// The `metadata.training` field exceeds the size limit.
    TrainingTooLarge {
        /// Approximate size in bytes.
        size: usize,
        /// Maximum allowed size.
        max_size: usize,
    },
    /// Generic serde_json parse error.
    Serde(String),
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightsExceedLimit { got, max } => {
                write!(
                    f,
                    "weights array exceeds limit ({} floats, max is {})",
                    got, max
                )
            }
            Self::TrainingTooDeep { depth, max_depth } => {
                write!(
                    f,
                    "metadata.training JSON tree too deep (depth {}, max is {})",
                    depth, max_depth
                )
            }
            Self::TrainingTooLarge { size, max_size } => {
                write!(
                    f,
                    "metadata.training JSON too large ({} bytes, max is {} bytes)",
                    size, max_size
                )
            }
            Self::Serde(msg) => write!(f, "JSON parse error: {}", msg),
        }
    }
}

impl std::error::Error for JsonError {}

impl From<serde_json::Error> for JsonError {
    fn from(e: serde_json::Error) -> Self {
        JsonError::Serde(e.to_string())
    }
}
