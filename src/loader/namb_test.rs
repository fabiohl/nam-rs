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
}
