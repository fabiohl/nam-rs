// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Minimalist helpers for IEEE float32 mono WAV read/write.
//!
//! Fixed format: 1 channel, 32-bit IEEE float, little-endian.
//! No external crate — just `std::fs` and `std::io`.

#![allow(dead_code)]

use std::fs;
use std::io;
use std::path::Path;

const WAV_HEADER_SIZE: usize = 44;
const FMT_CHUNK_SIZE: u32 = 16;
const AUDIO_FORMAT_IEEE_FLOAT: u16 = 3;

/// Writes a mono IEEE float32 WAV file.
///
/// Uses fast bulk-copy on little-endian systems instead of per-sample iteration.
pub fn write_wav_f32(path: &Path, samples: &[f32], sample_rate: u32) -> io::Result<()> {
    let num_samples = samples.len() as u32;
    let data_size = num_samples * 4;
    let file_size = WAV_HEADER_SIZE as u32 + data_size;
    let byte_rate = sample_rate * 4;
    let block_align: u16 = 4;
    let bits_per_sample: u16 = 32;

    let mut buf = Vec::with_capacity(file_size as usize);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&FMT_CHUNK_SIZE.to_le_bytes());
    buf.extend_from_slice(&AUDIO_FORMAT_IEEE_FLOAT.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());

    #[cfg(target_endian = "little")]
    {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(samples.as_ptr() as *const u8, samples.len() * 4) };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &s in samples {
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }

    fs::write(path, &buf)
}

/// Reads a mono IEEE float32 WAV file.
///
/// Returns `(samples, sample_rate)`.
pub fn read_wav_f32(path: &Path) -> io::Result<(Vec<f32>, u32)> {
    let data = fs::read(path)?;

    if data.len() < WAV_HEADER_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAV file too small: {} bytes", data.len()),
        ));
    }

    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid RIFF/WAVE header",
        ));
    }

    let audio_format = u16::from_le_bytes([data[20], data[21]]);
    let num_channels = u16::from_le_bytes([data[22], data[23]]);
    let sr = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);

    if audio_format != AUDIO_FORMAT_IEEE_FLOAT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Audio format is not IEEE float32: {audio_format}"),
        ));
    }
    if num_channels != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("WAV is not mono (channels={num_channels})"),
        ));
    }

    let mut pos = 36;
    let mut data_size: u32 = 0;
    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
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
            "'data' chunk not found in WAV",
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
