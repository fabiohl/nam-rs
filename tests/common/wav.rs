// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Helpers minimalistas para WAV mono float32 IEEE.
//!
//! Formato fixo: 1 canal, 32-bit IEEE float, little-endian.
//! Sem crate externo — apenas `std::fs` e `std::io`.

use std::fs;
use std::io;
use std::path::Path;

const WAV_HEADER_SIZE: usize = 44;
const FMT_CHUNK_SIZE: u32 = 16;
const AUDIO_FORMAT_IEEE_FLOAT: u16 = 3;

pub fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) -> io::Result<()> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 4; // 32-bit float = 4 bytes
    let file_size = WAV_HEADER_SIZE as u32 + data_size;
    let byte_rate = sample_rate * 4; // 1 channel × 4 bytes
    let block_align: u16 = 4;
    let bits_per_sample: u16 = 32;

    let mut buf = Vec::with_capacity(file_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt subchunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&FMT_CHUNK_SIZE.to_le_bytes());
    buf.extend_from_slice(&AUDIO_FORMAT_IEEE_FLOAT.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // 1 channel (mono)
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data subchunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }

    fs::write(path, &buf)
}

pub fn read_wav_f32(path: &Path) -> io::Result<(Vec<f32>, u32)> {
    let data = fs::read(path)?;

    if data.len() < WAV_HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Arquivo WAV muito pequeno: {} bytes", data.len()),
        ));
    }

    // RIFF header
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Header RIFF/WAVE inválido",
        ));
    }

    // fmt subchunk
    let audio_format = u16::from_le_bytes([data[20], data[21]]);
    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    let sr = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);

    if audio_format != AUDIO_FORMAT_IEEE_FLOAT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Formato de áudio não é float32 IEEE: {}", audio_format),
        ));
    }
    if num_channels != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAV não é mono (canais={})", num_channels),
        ));
    }

    // data subchunk (skip non-data chunks if present)
    let mut pos = 36;
    let mut data_size: u32 = 0;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        if chunk_id == b"data" {
            data_size = chunk_size;
            pos += 8;
            break;
        }
        pos += 8 + chunk_size as usize;
    }
    if data_size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Chunk 'data' não encontrado no WAV",
        ));
    }

    let num_samples = (data_size as usize) / 4;
    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let offset = pos + i * 4;
        if offset + 4 > data.len() {
            break;
        }
        let f = f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        samples.push(f);
    }

    Ok((samples, sr))
}
