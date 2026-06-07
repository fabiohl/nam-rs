// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Binary loader for `.namb` models.
//!
//! Performs direct, deterministic, lock-free analysis and deserialization
//! from a binary block into the `NamModelData` structure.

use super::nam_json::{NamConfig, NamLayerConfig, NamMetadata, NamModelData, WeightsLayout};
use anyhow::Result;
use log::info;

/// Typed error for `.namb` file parsing.
///
/// Each variant corresponds to a specific integrity or format failure
/// of the binary file, enabling precise diagnosis via
/// `downcast_ref` in the `loader` module.
#[derive(Debug, thiserror::Error)]
pub enum NambError {
    /// Truncated file: insufficient bytes for the minimum header.
    #[error("file truncated: got {got} bytes, need at least {need}")]
    Truncated {
        /// Bytes available in the file.
        got: usize,
        /// Minimum bytes needed.
        need: usize,
    },

    /// Invalid magic number (not 0x4E414D42).
    #[error("invalid magic number: 0x{0:08X} (expected 0x4E414D42)")]
    InvalidMagic(u32),

    /// Unsupported `.namb` format version.
    #[error("unsupported .namb version: {0}")]
    InvalidVersion(u16),

    /// Weight section offset beyond file size.
    #[error("weights offset {offset} out of file bounds (file size: {file_len})")]
    WeightsOffsetOutOfBounds {
        /// Offset declared in the header.
        offset: usize,
        /// Total file size in bytes.
        file_len: usize,
    },

    /// Weight section offset smaller than the header size.
    #[error("invalid weights offset {offset} (smaller than header size {header_size})")]
    InvalidWeightsOffset {
        /// Offset declared in the header.
        offset: usize,
        /// Expected header size.
        header_size: usize,
    },

    /// CRC32 checksum of the weight section does not match.
    #[error("CRC32 mismatch: got 0x{got:08X}, expected 0x{expected:08X}")]
    CrcMismatch {
        /// CRC calculated from the data.
        got: u32,
        /// CRC declared in the header.
        expected: u32,
    },

    /// CRC32 missing in NAMB v2+ file (FLAG_HAS_CRC32 flag not set).
    #[error("CRC32 flag missing in NAMB v{version} file (FLAG_HAS_CRC32 not set)")]
    CrcMissing {
        /// NAMB file version.
        version: u16,
    },
}

/// Computes the CRC32 (IEEE 802.3) of a byte slice.
/// Replaces the external `crc32fast` dependency with a lightweight software version.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB88320u32 & mask);
        }
    }
    crc ^ 0xFFFFFFFFu32
}

fn check_crc(data: &[u8], weights_offset: usize, expected: u32) -> Result<(), NambError> {
    let calculated = crc32_ieee(&data[weights_offset..]);
    if calculated != expected {
        return Err(NambError::CrcMismatch {
            got: calculated,
            expected,
        });
    }
    Ok(())
}

/// Flag bitmask for the NAMB header `flags` field.
pub const FLAG_HAS_CRC32: u8 = 0x01;

/// Fixed binary header of the `.namb` format.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct NambHeader {
    /// Magic number `0x4E414D42` ("NAMB" in ASCII).
    pub magic: u32,
    /// Format version (1 = legacy, 2 = with pre-transposed layout).
    pub version: u16,
    /// Weight layout (only if version >= 2). Offset: 6.
    pub layout_type: u8,
    /// Feature flags (bit 0 = FLAG_HAS_CRC32). Offset: 7.
    pub flags: u8,
    /// Reserved for future expansion. Offset: 8.
    pub reserved_v2: [u8; 4],
    /// Offset (in bytes) from the beginning of the file to the start of the weight section.
    pub weights_offset: u32,
    /// Reserved for future expansion.
    pub reserved1: [u32; 2],
    /// CRC32 checksum of the weight block (optional).
    pub crc32: u32,
    /// Reserved for future expansion.
    pub reserved2: u32,
    /// Informational version string (e.g. "NAMB 2.0.0").
    pub version_str: [u8; 32],
    /// Default sample rate (e.g. 48000.0).
    pub sample_rate: f32,
    /// Default input level dBu (e.g. 12.0).
    pub input_level_dbu: f32,
    /// Default output level dBu (e.g. 12.0).
    pub output_level_dbu: f32,
    /// Reserved (total header size must be at least 80 bytes).
    pub reserved3: [u32; 1],
}

impl NambHeader {
    /// Validates whether the header has the magic number and a supported version.
    pub fn validate(&self) -> Result<(), NambError> {
        let magic = self.magic;
        let version = self.version;
        if magic != 0x4E414D42 {
            return Err(NambError::InvalidMagic(magic));
        }
        if version != 1 && version != 2 {
            return Err(NambError::InvalidVersion(version));
        }
        Ok(())
    }

    /// Returns the weight layout based on the version and the flag.
    pub fn get_layout(&self) -> WeightsLayout {
        let version = self.version;
        if version < 2 {
            return WeightsLayout::Original;
        }
        match self.layout_type {
            1 => WeightsLayout::GateMajorLstm,
            2 => WeightsLayout::Interleaved4WaveNet,
            _ => WeightsLayout::Original,
        }
    }
}

/// Loads a model in the `.namb` binary format.
pub fn parse_namb(data: &[u8]) -> Result<NamModelData> {
    let header_size = std::mem::size_of::<NambHeader>();
    if data.len() < header_size {
        return Err(NambError::Truncated {
            got: data.len(),
            need: header_size,
        }
        .into());
    }

    // 1. Reads the header
    let header = unsafe { &*data.as_ptr().cast::<NambHeader>() };
    header.validate()?;

    // 2. Reads the JSON metadata section (optional in .namb, but common)
    // If weights_offset > header_size, there is a JSON between them.
    let weights_offset = header.weights_offset as usize;
    if weights_offset > data.len() {
        return Err(NambError::WeightsOffsetOutOfBounds {
            offset: weights_offset,
            file_len: data.len(),
        }
        .into());
    }
    if weights_offset < header_size {
        return Err(NambError::InvalidWeightsOffset {
            offset: weights_offset,
            header_size,
        }
        .into());
    }

    let mut model_data = if weights_offset > header_size {
        let json_bytes = &data[header_size..weights_offset];
        // Truncate nulls if present (the NAMB buffer is usually padded)
        let actual_json = if let Some(pos) = json_bytes.iter().position(|&b| b == 0) {
            &json_bytes[..pos]
        } else {
            json_bytes
        };

        if !actual_json.is_empty() {
            crate::loader::nam_json::parse_nam_json(std::str::from_utf8(actual_json)?)?
        } else {
            make_fallback_model_data()
        }
    } else {
        make_fallback_model_data()
    };

    // 3. Integrity Validation (CRC32)
    let version = header.version;
    let crc32_header = header.crc32;
    if version >= 2 {
        if header.flags & FLAG_HAS_CRC32 == 0 {
            return Err(NambError::CrcMissing { version }.into());
        }
        check_crc(data, weights_offset, crc32_header)?;
    } else if crc32_header != 0 {
        check_crc(data, weights_offset, crc32_header)?;
    } else {
        log::warn!("CRC32 missing in NAMB v1 file (crc32=0 sentinel) — skipping integrity check");
    }

    // 4. Reads the binary weights
    let pesos_raw = &data[weights_offset..];
    let float_count = pesos_raw.len() / 4;
    let mut weights = Vec::with_capacity(float_count);

    for chunk in pesos_raw.chunks_exact(4) {
        weights.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    // Populates header metadata into the final NamModelData
    let sample_rate_header = header.sample_rate;
    let input_level_header = header.input_level_dbu;
    let output_level_header = header.output_level_dbu;
    let version_header = header.version;

    model_data.weights = weights;
    model_data.sample_rate = Some(sample_rate_header);
    model_data.weights_layout = header.get_layout();

    // Updates metadata if it exists
    if let Some(ref mut metadata) = model_data.metadata {
        metadata.input_level_dbu = Some(input_level_header);
        metadata.output_level_dbu = Some(output_level_header);
    } else {
        model_data.metadata = Some(NamMetadata {
            date: None,
            name: None,
            modeled_by: None,
            gear_make: None,
            gear_model: None,
            gear_type: None,
            tone_type: None,
            training: None,
            input_level_dbu: Some(input_level_header),
            output_level_dbu: Some(output_level_header),
            loudness: Some(-18.0),
        });
    }

    // If the version is null (fallback), gets it from the header string
    if model_data.version.is_none() {
        let end_idx = header
            .version_str
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(32);
        let version_str = String::from_utf8_lossy(&header.version_str[..end_idx]).into_owned();
        model_data.version = Some(version_str);
    }

    info!(
        "[Loader] .namb v{} loaded ({} weights, layout={:?})",
        version_header, float_count, model_data.weights_layout
    );

    Ok(model_data)
}

/// Creates a "fallback" dataset.
/// Useful for old .namb files that do not describe their own structure.
fn make_fallback_model_data() -> NamModelData {
    NamModelData {
        version: None,
        architecture: "WaveNet".to_string(), // legacy .namb files are always WaveNet Standard
        config: make_standard_wavenet_config(),
        weights: Vec::new(),
        sample_rate: None,
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

/// Defines the standard "template" for the WaveNet algorithm.
/// It's like defining the number of neurons and connections of a standard digital brain.
fn make_standard_wavenet_config() -> NamConfig {
    // Dilations: defines the "reach" of the algorithm's memory (essential for capturing timbre).
    let std_dilations = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    // First processing layer.
    let l0 = NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        head_size: Some(8),
        channels: Some(16),   // "Width" of the internal processing.
        kernel_size: Some(3), // Number of neighboring samples analyzed at each step.
        dilations: Some(std_dilations.clone()),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(false),
    };

    // Second layer (usually identical to the first in Standard models).
    let l1 = NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        head_size: Some(8),
        channels: Some(16),
        kernel_size: Some(3),
        dilations: Some(std_dilations),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(true),
    };

    NamConfig {
        layers: vec![l0, l1],
        head: Some(None),
        head_scale: Some(0.02), // Final volume adjustment to ensure consistency.
        num_layers: None,
        hidden_size: None,
    }
}

#[cfg(test)]
#[path = "namb_test.rs"]
mod namb_test;
