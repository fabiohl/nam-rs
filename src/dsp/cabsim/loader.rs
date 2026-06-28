// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Robust WAV IR loader: mono PCM16, PCM24, and IEEE float32.
//!
//! Loading, resampling, and normalization happen **outside** the audio thread.
//! The resulting `CabSimIr` is transferred to the RT callback via lock-free SPSC,
//! following the same pattern as `NamResampler` swapping.

use crate::dsp::pipeline::MAX_RESAMP_BUF;
use crate::dsp::resampler::NamResampler;
use crate::dsp::sinc_kernel::TAPS_PER_PHASE;

use std::io;
use std::io::Read;
use std::path::Path;

/// Maximum IR file size to guard against malformed/malicious WAVs (1 GiB).
const MAX_IR_FILE_SIZE: u64 = 1_073_741_824;

/// Minimum size for a valid WAV header (44 bytes: RIFF + fmt + data).
const WAV_HEADER_MIN: usize = 44;

// RIFF chunk IDs.
const RIFF_ID: [u8; 4] = *b"RIFF";
const WAVE_ID: [u8; 4] = *b"WAVE";
const FMT_ID: [u8; 4] = *b"fmt ";
const DATA_ID: [u8; 4] = *b"data";

// WAV format tags.
const WAV_FORMAT_PCM: u16 = 1;
const WAV_FORMAT_IEEE_FLOAT: u16 = 3;

/// A loaded impulse response ready for convolution.
///
/// All memory is pre-allocated at load time (outside the audio thread).
/// The struct holds the resampled, normalized mono samples alongside
/// the original metadata needed for convolution setup.
pub struct CabSimIr {
    /// Mono IR samples (f32).
    pub samples: Vec<f32>,
    /// Sample rate of the IR after resampling (i.e., the effective rate of `samples`).
    pub sample_rate: u32,
    /// Sample rate of the original WAV file (before resampling, if applicable).
    pub original_rate: u32,
    /// Whether the IR was normalized (peak = 1.0).
    pub normalized: bool,
}

impl CabSimIr {
    /// Loads a mono WAV impulse response from `path`.
    ///
    /// Supported formats: PCM16, PCM24, IEEE float32.
    ///
    /// # Parameters
    /// - `path`: path to the `.wav` file.
    /// - `target_rate`: desired sample rate after resampling. If 0, no resampling.
    /// - `normalize`: if true, normalize samples so peak amplitude = 1.0.
    ///
    /// # Errors
    /// Returns `io::Error` on I/O failures, invalid headers, unsupported formats,
    /// or resampling errors. Never panics.
    #[cold]
    pub fn load(path: &Path, target_rate: u32, normalize: bool) -> io::Result<Box<Self>> {
        let data = Self::read_file(path)?;
        let (samples, original_rate) = Self::parse_wav(&data)?;

        let mut samples = if target_rate != 0 && target_rate != original_rate {
            Self::resample(&samples, original_rate, target_rate)?
        } else {
            samples
        };

        let effective_rate = if target_rate != 0 && target_rate != original_rate {
            target_rate
        } else {
            original_rate
        };

        let normalized = if normalize {
            Self::normalize_in_place(&mut samples)
        } else {
            false
        };

        Ok(Box::new(Self {
            samples,
            sample_rate: effective_rate,
            original_rate,
            normalized,
        }))
    }

    /// Reads and validates the file, guarding against oversized inputs.
    ///
    /// Uses a single file-open to avoid TOCTOU races between metadata check and read.
    fn read_file(path: &Path) -> io::Result<Vec<u8>> {
        let mut file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        if file_size == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("IR WAV file is empty: {}", path.display()),
            ));
        }

        if file_size > MAX_IR_FILE_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IR WAV file too large ({} bytes, max {})",
                    file_size, MAX_IR_FILE_SIZE
                ),
            ));
        }

        if file_size < WAV_HEADER_MIN as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "IR WAV file too small ({} bytes, min {})",
                    file_size, WAV_HEADER_MIN
                ),
            ));
        }

        let mut data = Vec::with_capacity(file_size as usize);
        file.read_to_end(&mut data)?;
        Ok(data)
    }

    /// Parses a WAV byte buffer, returning (samples, sample_rate).
    ///
    /// Handles PCM16, PCM24, and IEEE float32 in mono.
    /// Scans for `"fmt "` and `"data"` chunks rather than assuming fixed offsets.
    fn parse_wav(data: &[u8]) -> io::Result<(Vec<f32>, u32)> {
        if data[0..4] != RIFF_ID || data[8..12] != WAVE_ID {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: invalid RIFF/WAVE header",
            ));
        }

        let (fmt_offset, _fmt_size) = Self::find_chunk(data, &FMT_ID, 12).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IR WAV: 'fmt ' chunk not found")
        })?;

        if fmt_offset + 16 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: 'fmt ' chunk too small",
            ));
        }

        let audio_format = u16::from_le_bytes([data[fmt_offset], data[fmt_offset + 1]]);
        let num_channels = u16::from_le_bytes([data[fmt_offset + 2], data[fmt_offset + 3]]);
        let sample_rate = u32::from_le_bytes([
            data[fmt_offset + 4],
            data[fmt_offset + 5],
            data[fmt_offset + 6],
            data[fmt_offset + 7],
        ]);
        let bits_per_sample = u16::from_le_bytes([data[fmt_offset + 14], data[fmt_offset + 15]]);

        if num_channels != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("IR WAV must be mono (found {} channels)", num_channels),
            ));
        }

        if sample_rate == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: sample rate is zero",
            ));
        }

        let (data_start, data_size) = Self::find_data_chunk(data)?;

        let samples = match audio_format {
            WAV_FORMAT_PCM => match bits_per_sample {
                16 => Self::read_pcm16(&data[data_start..], data_size),
                24 => Self::read_pcm24(&data[data_start..], data_size),
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "IR WAV: unsupported PCM bit depth {} (only 16 and 24 supported)",
                            bits_per_sample
                        ),
                    ));
                }
            },
            WAV_FORMAT_IEEE_FLOAT if bits_per_sample == 32 => {
                Self::read_float32(&data[data_start..], data_size)
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "IR WAV: unsupported format (audio_format={}, bits={})",
                        audio_format, bits_per_sample
                    ),
                ));
            }
        };

        if samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: no audio samples found",
            ));
        }

        Self::validate_samples(&samples)?;

        Ok((samples, sample_rate))
    }

    /// Locates a RIFF chunk by its 4-byte ID, starting search at `start_offset`.
    ///
    /// Returns `Some((data_offset, data_size))` where `data_offset` points to the
    /// chunk's payload, or `None` if not found.
    fn find_chunk(data: &[u8], chunk_id: &[u8; 4], start_offset: usize) -> Option<(usize, u32)> {
        let mut pos = start_offset;
        while pos + 8 <= data.len() {
            let id = &data[pos..pos + 4];
            let size =
                u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);
            if id == chunk_id {
                return Some((pos + 8, size));
            }
            pos = pos.saturating_add(8 + size as usize);
        }
        None
    }

    /// Locates the "data" chunk in a WAV file, returning (offset, size_in_bytes).
    fn find_data_chunk(data: &[u8]) -> io::Result<(usize, u32)> {
        Self::find_chunk(data, &DATA_ID, 12).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "IR WAV: 'data' chunk not found")
        })
    }

    /// Validates that all samples are finite (no NaN, +Inf, -Inf).
    ///
    /// Catches corrupt float32 WAVs that would otherwise propagate NaN
    /// through the entire DSP pipeline.
    fn validate_samples(samples: &[f32]) -> io::Result<()> {
        if samples.iter().any(|&s| !s.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IR WAV: samples contain NaN or Infinity",
            ));
        }
        Ok(())
    }

    /// Reads PCM16 mono samples as f32 in [-1.0, 1.0).
    fn read_pcm16(data: &[u8], data_size: u32) -> Vec<f32> {
        let num_samples = (data_size as usize) / 2;
        let available = data.len() / 2;
        let n = num_samples.min(available);
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 2;
            let raw = i16::from_le_bytes([data[offset], data[offset + 1]]);
            // Normalize to [-1.0, 1.0)
            let f = if raw >= 0 {
                raw as f32 / 32767.0
            } else {
                raw as f32 / 32768.0
            };
            samples.push(f);
        }
        samples
    }

    /// Reads PCM24 mono samples as f32 in [-1.0, 1.0).
    fn read_pcm24(data: &[u8], data_size: u32) -> Vec<f32> {
        let num_samples = (data_size as usize) / 3;
        let available = data.len() / 3;
        let n = num_samples.min(available);
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 3;
            // Little-endian 24-bit signed -> i32
            let b0 = data[offset] as i32;
            let b1 = data[offset + 1] as i32;
            let b2 = data[offset + 2] as i32;
            let raw = (b2 << 16) | (b1 << 8) | b0;
            // Sign-extend from 24 bits
            let raw = if raw & 0x0080_0000 != 0 {
                raw | 0xFF00_0000u32 as i32
            } else {
                raw
            };
            // Normalize to [-1.0, 1.0)
            let f = if raw >= 0 {
                raw as f32 / 8_388_607.0
            } else {
                raw as f32 / 8_388_608.0
            };
            samples.push(f);
        }
        samples
    }

    /// Reads IEEE float32 mono samples directly (no conversion).
    fn read_float32(data: &[u8], data_size: u32) -> Vec<f32> {
        let num_samples = (data_size as usize) / 4;
        let available = data.len() / 4;
        let n = num_samples.min(available);
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 4;
            let f = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            samples.push(f);
        }
        samples
    }

    /// Resamples `input` from `input_rate` to `output_rate` using the polyphase resampler.
    ///
    /// This is a batch/offline operation: feeds the entire IR through `NamResampler`
    /// and pads with zeros to flush the filter’s delay line.
    fn resample(input: &[f32], input_rate: u32, output_rate: u32) -> io::Result<Vec<f32>> {
        let mut resampler =
            NamResampler::new(input_rate, output_rate, MAX_RESAMP_BUF).map_err(|e| {
                io::Error::other(format!(
                    "IR resample failed ({} Hz → {} Hz): {:#}",
                    input_rate, output_rate, e
                ))
            })?;

        // Estimate output length: input_len * (output_rate / input_rate) + safety margin
        let est_len =
            ((input.len() as f64 * output_rate as f64 / input_rate as f64).ceil() as usize) + 256;
        let mut output = Vec::with_capacity(est_len);

        let mut in_buf = vec![0.0f32; MAX_RESAMP_BUF];
        let mut out_l = vec![0.0f32; MAX_RESAMP_BUF];
        let mut out_r = vec![0.0f32; MAX_RESAMP_BUF];

        let mut pos = 0usize;
        while pos < input.len() {
            let chunk = (input.len() - pos).min(MAX_RESAMP_BUF);
            in_buf[..chunk].copy_from_slice(&input[pos..pos + chunk]);
            let written = resampler.process_input_mono(&in_buf[..chunk], &mut out_l, &mut out_r);
            output.extend_from_slice(&out_l[..written]);
            pos += chunk;
        }

        // Flush the resampler's delay line with zeros. Each iteration feeds
        // MAX_RESAMP_BUF zeros; TAPS_PER_PHASE × 2 iterations ensure the longer
        // ringdown with the increased tap count is fully drained.
        in_buf.fill(0.0);
        let flush_iters = TAPS_PER_PHASE * 2;
        for _ in 0..flush_iters {
            let written = resampler.process_input_mono(&in_buf, &mut out_l, &mut out_r);
            if written == 0 {
                break;
            }
            output.extend_from_slice(&out_l[..written]);
        }

        if output.is_empty() {
            return Err(io::Error::other("IR resample produced no output samples"));
        }

        Ok(output)
    }

    /// Normalizes samples in-place so the absolute peak becomes 1.0.
    ///
    /// Returns `true` if normalization was applied (peak > 0).
    /// If all samples are zero, they are left unchanged and returns `false`.
    fn normalize_in_place(samples: &mut [f32]) -> bool {
        let peak = samples.iter().fold(0.0f32, |acc, &s| acc.max(s.abs()));

        if peak <= 0.0 || peak >= 1.0 && (peak - 1.0).abs() < f32::EPSILON {
            return false;
        }

        let gain = 1.0 / peak;
        for s in samples.iter_mut() {
            *s *= gain;
        }
        true
    }
}

#[cfg(test)]
#[path = "loader_test.rs"]
mod loader_test;
