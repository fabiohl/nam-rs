// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::loader::namb::*;
    use anyhow::Result;

    /// Builds a .namb v1 file (binary format) in memory for testing purposes.
    /// It's like "fabricating" a fake file to see if the program can read it.
    fn build_valid_namb_v1(w_floats: &[f32]) -> Vec<u8> {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size + w_floats.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        // We fill the "header" (the file's identification label).
        header.magic = 0x4E414D42; // "NAMB" in hexadecimal code.
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        // Converts the decimal numbers (weights) into raw bytes.
        for (i, &f) in w_floats.iter().enumerate() {
            let offset = header_size + i * 4;
            data[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
        }

        // Generates a "security seal" (CRC32) to ensure the data has not been altered.
        header.crc32 = crc32_ieee(&data[header_size..]);
        data
    }

    #[test]
    fn test_parse_namb_v1() -> Result<()> {
        // Tests whether the loader can correctly read a basic v1 file.
        let w = [0.1f32, -0.5f32, 1.0f32];
        let data = build_valid_namb_v1(&w);
        let parsed = parse_namb(&data)?;

        // Checks whether the read values are identical to the ones we wrote.
        assert_eq!(parsed.weights, w);
        assert_eq!(parsed.weights_layout, WeightsLayout::Original);
        assert_eq!(parsed.sample_rate, Some(48000.0));
        Ok(())
    }

    #[test]
    fn test_parse_namb_v2_gate_major() -> Result<()> {
        // Tests whether the loader recognizes the new v2 format (with optimized layout).
        let header_size = std::mem::size_of::<NambHeader>();
        let w = [0.0f32; 4];
        let mut data = vec![0u8; header_size + w.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 2;
        header.layout_type = 1; // 1 indicates "GateMajorLstm" (layout optimized for LSTM)
        header.flags = FLAG_HAS_CRC32;
        header.weights_offset = header_size as u32;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;

        // Writes the weights into the buffer
        for (i, &f) in w.iter().enumerate() {
            let offset = header_size + i * 4;
            data[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
        }
        header.crc32 = crc32_ieee(&data[header_size..]);

        let parsed = parse_namb(&data)?;
        // Ensures the program understood that this file needs a special reorganization.
        assert_eq!(parsed.weights_layout, WeightsLayout::GateMajorLstm);
        Ok(())
    }

    #[test]
    fn test_v2_missing_crc32_flag_rejected() {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 2;
        header.layout_type = 1;
        header.flags = 0; // FLAG_HAS_CRC32 NOT set
        header.weights_offset = header_size as u32;
        header.crc32 = 0xDEADBEEF;

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::CrcMissing");
        assert!(
            matches!(namb_err, NambError::CrcMissing { version: 2 }),
            "Expected CrcMissing, got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_v2_crc32_zero_legitimate_passes() -> Result<()> {
        // CRC32 of an empty slice is 0 (property of the IEEE 802.3 algorithm).
        // With FLAG_HAS_CRC32 set and crc32=0 legitimate, the parser should accept.
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size]; // No weights → crc32_ieee(&[]) == 0
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 2;
        header.layout_type = 1;
        header.flags = FLAG_HAS_CRC32;
        header.weights_offset = header_size as u32;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.crc32 = 0; // Legitimate CRC32 for empty slice

        let parsed = parse_namb(&data)?;
        assert!(parsed.weights.is_empty());
        Ok(())
    }

    #[test]
    fn test_v1_crc32_zero_warns_but_passes() -> Result<()> {
        // v1 with crc==0 (sentinel) should pass with warning, not block.
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size + 4]; // 1 float dummy
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.crc32 = 0; // Sentinel: CRC absent in v1

        // Writes a dummy weight
        let w = 0.5f32;
        data[header_size..header_size + 4].copy_from_slice(&w.to_le_bytes());

        let parsed = parse_namb(&data)?;
        assert_eq!(parsed.weights, vec![0.5f32]);
        Ok(())
    }

    #[test]
    fn test_reject_magic_bman() {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x424D414E; // "BMAN" — no longer accepted (S5.T09)
        header.version = 1;
        header.weights_offset = header_size as u32;

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::InvalidMagic");
        assert!(
            matches!(namb_err, NambError::InvalidMagic(m) if *m == 0x424D414E),
            "Expected InvalidMagic(0x424D414E), got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_truncated_header() {
        // File smaller than the minimum header (80 bytes) must be rejected.
        for len in 0..80 {
            let data = vec![0u8; len];
            let err = parse_namb(&data).unwrap_err();
            let namb_err = err
                .downcast_ref::<NambError>()
                .expect("Error should be NambError::Truncated");
            assert!(
                matches!(namb_err, NambError::Truncated { got: _, need: _ }),
                "Expected Truncated, got: {:?}",
                namb_err
            );
        }
    }

    #[test]
    fn test_weight_residue_rejected() {
        // Weights section with trailing 1-3 bytes must be rejected.
        for residue in 1..=3 {
            let header_size = std::mem::size_of::<NambHeader>();
            let weights_bytes = residue + 4; // 1 complete f32 + residue
            let mut data = vec![0u8; header_size + weights_bytes];
            let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

            header.magic = 0x4E414D42;
            header.version = 1;
            header.weights_offset = header_size as u32;
            header.crc32 = 0; // v1 sentinel: no CRC
            // Fill version_str with something
            header.version_str[0..5].copy_from_slice(b"1.0.0");

            // Write one valid f32
            let weight = 0.5f32;
            data[header_size..header_size + 4].copy_from_slice(&weight.to_le_bytes());
            // Trailing 1-3 bytes are left as zeros (residue)

            let err = parse_namb(&data).unwrap_err();
            let namb_err = err
                .downcast_ref::<NambError>()
                .expect("Error should be NambError::Truncated");
            assert!(
                matches!(namb_err, NambError::Truncated { got: _, need: _ }),
                "Residue of {} byte(s) should be rejected as Truncated, got: {:?}",
                residue,
                namb_err
            );
        }
    }

    // ── CRC32 KAT (known-answer tests) ──────────────────────────────────

    #[test]
    fn test_crc32_kat_canonical() {
        // Canonical check vector: CRC32/IEEE 802.3 of "123456789" == 0xCBF43926
        assert_eq!(
            crc32_ieee(b"123456789"),
            0xCBF43926,
            "CRC32 canonical KAT failed"
        );
    }

    #[test]
    fn test_crc32_kat_empty() {
        // CRC32 of empty slice is 0x00000000 (property of the IEEE 802.3 algorithm)
        assert_eq!(crc32_ieee(b""), 0x00000000, "CRC32 of empty slice != 0");
    }

    #[test]
    fn test_crc32_kat_single_zero() {
        assert_eq!(
            crc32_ieee(&[0u8; 1]),
            0xD202EF8D,
            "CRC32 of single 0x00 byte"
        );
    }

    #[test]
    fn test_crc32_kat_four_zeros() {
        assert_eq!(
            crc32_ieee(&[0u8; 4]),
            0x2144DF1C,
            "CRC32 of four zero bytes"
        );
    }

    #[test]
    fn test_crc32_kat_thirtytwo_zeros() {
        assert_eq!(crc32_ieee(&[0u8; 32]), 0x190A55AD, "CRC32 of 32 zero bytes");
    }

    #[test]
    fn test_crc32_kat_sequential() {
        // 0x00..0x1F (32 sequential bytes)
        let data: Vec<u8> = (0u8..32).collect();
        assert_eq!(crc32_ieee(&data), 0x91267E8A, "CRC32 of 0x00..0x1F");
    }

    #[test]
    fn test_crc32_kat_all_ff() {
        assert_eq!(
            crc32_ieee(&[0xFFu8; 32]),
            0xFF6CAB0B,
            "CRC32 of 32 bytes of 0xFF"
        );
    }

    #[test]
    fn test_crc32_kat_alternating() {
        // Alternating 0x55, 0xAA × 16 (32 bytes)
        let pattern: Vec<u8> = std::iter::repeat_n([0x55u8, 0xAAu8], 16)
            .flatten()
            .collect();
        assert_eq!(crc32_ieee(&pattern), 0x8BA7B8B6, "CRC32 of 0x55/0xAA ×16");
    }

    // ── Truncation at header boundary offsets ──────────────────────────

    /// Builds a .namb v1 without CRC (crc32=0 sentinel) for truncation testing.
    /// CRC is intentionally 0 so truncation doesn't trigger CRC mismatch.
    fn build_namb_v1_no_crc(w_floats: &[f32]) -> Vec<u8> {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size + w_floats.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0; // v1 sentinel: no CRC validation
        header.version_str[0..5].copy_from_slice(b"1.0.0");
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;

        for (i, &f) in w_floats.iter().enumerate() {
            let offset = header_size + i * 4;
            data[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
        }
        data
    }

    #[test]
    fn test_truncation_header_boundaries() {
        // Systematic truncation at every byte offset 0..79 (header boundaries).
        // All must return NambError::Truncated because the header is incomplete.
        let weights = [0.1f32, 0.2f32, 0.3f32];
        let full = build_namb_v1_no_crc(&weights);
        let header_size = std::mem::size_of::<NambHeader>();

        for len in 0..header_size {
            let truncated = &full[..len];
            let err = parse_namb(truncated).unwrap_err();
            let namb_err = err
                .downcast_ref::<NambError>()
                .expect("Error should be NambError::Truncated");
            assert!(
                matches!(namb_err, NambError::Truncated { got: _, need: _ }),
                "Truncation at {} bytes: expected Truncated, got {:?}",
                len,
                namb_err
            );
        }
    }

    #[test]
    fn test_truncation_weight_boundaries() -> Result<()> {
        // Creates a valid namb with 16 weights (no CRC) and truncates
        // at every 4-byte boundary within the weight section.
        // Each truncation at a 4-byte boundary must parse correctly.
        let weights: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let full = build_namb_v1_no_crc(&weights);
        let header_size = std::mem::size_of::<NambHeader>();

        for num_weights in 0..=weights.len() {
            let truncate_at = header_size + num_weights * 4;
            let truncated = &full[..truncate_at];
            let parsed = parse_namb(truncated)?;
            assert_eq!(
                parsed.weights.len(),
                num_weights,
                "Expected {} weights at truncation offset {}, got {}",
                num_weights,
                truncate_at,
                parsed.weights.len()
            );
            // Verify weight values match the original
            for (i, &expected) in weights[..num_weights].iter().enumerate() {
                assert!(
                    (parsed.weights[i] - expected).abs() < f32::EPSILON,
                    "Weight {} mismatch at truncation {} bytes: expected {}, got {}",
                    i,
                    truncate_at,
                    expected,
                    parsed.weights[i]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_truncation_residue_at_every_boundary() {
        // For every 4-byte boundary in the weights section, adding 1-3
        // residue bytes must be rejected as Truncated.
        let weights: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let full = build_namb_v1_no_crc(&weights);
        let header_size = std::mem::size_of::<NambHeader>();

        for num_weights in 0..=weights.len() {
            let base = header_size + num_weights * 4;
            for residue in 1..=3 {
                let truncate_at = base + residue;
                if truncate_at > full.len() {
                    continue;
                }
                let truncated = &full[..truncate_at];
                let err = parse_namb(truncated).unwrap_err();
                let namb_err = err
                    .downcast_ref::<NambError>()
                    .expect("Error should be NambError::Truncated");
                assert!(
                    matches!(namb_err, NambError::Truncated { .. }),
                    "Residue {} after {} weights (offset {}): expected Truncated, got {:?}",
                    residue,
                    num_weights,
                    truncate_at,
                    namb_err
                );
            }
        }
    }

    #[test]
    fn test_truncation_v2_header_boundary() {
        // v2 file truncated exactly at the header boundary (80 bytes),
        // with FLAG_HAS_CRC32 set and valid CRC (empty weights → crc=0).
        // Must parse successfully with 0 weights.
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 2;
        header.flags = FLAG_HAS_CRC32;
        header.weights_offset = header_size as u32;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.crc32 = 0; // crc32_ieee(&[]) == 0

        let parsed = parse_namb(&data).unwrap();
        assert!(parsed.weights.is_empty());
    }

    #[test]
    fn test_truncation_v2_post_header_residue() {
        // v2 file with 1 byte of residue after header (81 bytes total).
        // Must be rejected because weights section is not multiple of 4.
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size + 1];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 2;
        header.flags = FLAG_HAS_CRC32;
        header.weights_offset = header_size as u32;
        header.crc32 = crc32_ieee(&data[header_size..]);

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::Truncated");
        assert!(
            matches!(namb_err, NambError::Truncated { .. }),
            "v2 residue after header: expected Truncated, got {:?}",
            namb_err
        );
    }

    // ── Non-finite weight rejection ────────────────────────────────────

    #[test]
    fn test_non_finite_weight_nan_rejected() {
        let header_size = std::mem::size_of::<NambHeader>();
        let w = [f32::NAN];
        let mut data = vec![0u8; header_size + w.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0; // v1 sentinel: no CRC
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        data[header_size..header_size + 4].copy_from_slice(&w[0].to_le_bytes());

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::NonFiniteWeight");
        assert!(
            matches!(namb_err, NambError::NonFiniteWeight { index: 0, .. }),
            "Expected NonFiniteWeight at index 0, got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_non_finite_weight_inf_rejected() {
        let header_size = std::mem::size_of::<NambHeader>();
        let w = [0.5f32, f32::INFINITY];
        let mut data = vec![0u8; header_size + w.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        for (i, &f) in w.iter().enumerate() {
            let offset = header_size + i * 4;
            data[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
        }

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::NonFiniteWeight");
        assert!(
            matches!(namb_err, NambError::NonFiniteWeight { index: 1, .. }),
            "Expected NonFiniteWeight at index 1, got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_non_finite_weight_neg_inf_rejected() {
        let header_size = std::mem::size_of::<NambHeader>();
        let w = [1.0f32, 2.0f32, f32::NEG_INFINITY];
        let mut data = vec![0u8; header_size + w.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        for (i, &f) in w.iter().enumerate() {
            let offset = header_size + i * 4;
            data[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
        }

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::NonFiniteWeight");
        assert!(
            matches!(namb_err, NambError::NonFiniteWeight { index: 2, .. }),
            "Expected NonFiniteWeight at index 2, got: {:?}",
            namb_err
        );
    }

    // ── Invalid header field rejection ─────────────────────────────────

    #[test]
    fn test_invalid_header_sample_rate_nan() {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0;
        header.sample_rate = f32::NAN;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::InvalidHeaderField");
        assert!(
            matches!(
                namb_err,
                NambError::InvalidHeaderField {
                    field: "sample_rate",
                    ..
                }
            ),
            "Expected InvalidHeaderField(sample_rate), got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_invalid_header_sample_rate_negative() {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0;
        header.sample_rate = -44100.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::InvalidHeaderField");
        assert!(
            matches!(
                namb_err,
                NambError::InvalidHeaderField {
                    field: "sample_rate",
                    ..
                }
            ),
            "Expected InvalidHeaderField(sample_rate), got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_invalid_header_sample_rate_zero() {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0;
        header.sample_rate = 0.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::InvalidHeaderField");
        assert!(
            matches!(
                namb_err,
                NambError::InvalidHeaderField {
                    field: "sample_rate",
                    ..
                }
            ),
            "Expected InvalidHeaderField(sample_rate), got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_invalid_header_input_level_inf() {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0;
        header.sample_rate = 48000.0;
        header.input_level_dbu = f32::INFINITY;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::InvalidHeaderField");
        assert!(
            matches!(
                namb_err,
                NambError::InvalidHeaderField {
                    field: "input_level_dbu",
                    ..
                }
            ),
            "Expected InvalidHeaderField(input_level_dbu), got: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_invalid_header_output_level_neg_inf() {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = f32::NEG_INFINITY;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Error should be NambError::InvalidHeaderField");
        assert!(
            matches!(
                namb_err,
                NambError::InvalidHeaderField {
                    field: "output_level_dbu",
                    ..
                }
            ),
            "Expected InvalidHeaderField(output_level_dbu), got: {:?}",
            namb_err
        );
    }
}
