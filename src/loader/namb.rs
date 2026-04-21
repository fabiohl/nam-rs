// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Carregador Binário para modelos `.namb`.
//!
//! Realiza a análise e deserialização direta, determinística e lock-free
//! a partir de um bloco binário para a estrutura `NamModelData`.

use super::nam_json::{NamConfig, NamLayerConfig, NamMetadata, NamModelData};
use anyhow::{Result, bail};
use crc32fast::Hasher;

/// Lê um `f32` little-endian do slice nos offsets informados.
fn read_f32_le(data: &[u8], offset: usize) -> Result<f32> {
    if offset + 4 > data.len() {
        bail!("Fim inesperado do arquivo ao ler f32");
    }
    let bytes = data[offset..offset + 4].try_into()?;
    Ok(f32::from_le_bytes(bytes))
}

/// Carrega um modelo no formato binário `.namb`.
pub fn parse_namb(data: &[u8]) -> Result<NamModelData> {
    if data.len() < 80 {
        bail!("O arquivo fornecido é muito pequeno para conter o cabeçalho .namb");
    }

    // 0: Magic Number (b"NAMB" ou b"BMAN". A especificação diz 0x4E414D42)
    // Se convertermos 0x4E414D42 para `[u8; 4]` local, little-endian será `[0x42, 0x4D, 0x41, 0x4E]` ("BMAN")
    // Vamos checar diretamente o array fornecido.
    let magic_le = u32::from_le_bytes(data[0..4].try_into()?);
    if magic_le != 0x4E414D42 {
        // Fallback: Vamos checar se o Magic Number for string direta "NAMB"
        if &data[0..4] != b"NAMB" && &data[0..4] != b"BMAN" {
            bail!("Assinatura mágica inválida. O arquivo não parece ser um modelo .namb");
        }
    }

    // 4: Versão Lógica (Estritamente Versão 1)
    let version_le = u16::from_le_bytes(data[4..6].try_into()?);
    if version_le != 1 {
        bail!("Versão de arquivo não suportada: {}", version_le);
    }

    // 12: Offset de Início do Conjunto de Pesos Neurais
    let weights_offset = u32::from_le_bytes(data[12..16].try_into()?) as usize;
    if weights_offset > data.len() {
        bail!("Offset de pesos aponta além dos limites do arquivo.");
    }

    // 24: CRC32
    let crc_expected = u32::from_le_bytes(data[24..28].try_into()?);

    // Verificação CRC32 base IEEE 802.3 sobre o bloco de pesos.
    let mut hasher = Hasher::new();
    hasher.update(&data[weights_offset..]);
    let crc_calculated = hasher.finalize();

    if crc_calculated != crc_expected {
        bail!(
            "Falha na verificação de integridade CRC32: esperado 0x{:08X}, calculado 0x{:08X}. Arquivo possivelmente corrompido.",
            crc_expected,
            crc_calculated
        );
    }

    // 32: Geometria de Referência de Estúdio (48 bytes)
    // Assumimos que o cabeçalho seja estruturado conforme:
    // 32..64: version string / null-terminated string (32 bytes)
    // 64..68: amostra / sample_rate (f32)
    // 68..72: input_level_dbu (f32)
    // 72..76: output_level_dbu (f32)

    // Ler string truncando os nulos no fim.
    let version_str_bytes = &data[32..64];
    let end_idx = version_str_bytes.iter().position(|&b| b == 0).unwrap_or(32);
    let version_str = String::from_utf8_lossy(&version_str_bytes[..end_idx]).into_owned();

    let sample_rate = read_f32_le(data, 64).unwrap_or(48000.0);
    let input_level_dbu = read_f32_le(data, 68).unwrap_or(0.0);
    let output_level_dbu = read_f32_le(data, 72).unwrap_or(0.0);

    let metadata = NamMetadata {
        date: None,
        name: None,
        modeled_by: None,
        gear_make: None,
        gear_model: None,
        gear_type: None,
        tone_type: None,
        training: None,
        input_level_dbu: Some(input_level_dbu),
        output_level_dbu: Some(output_level_dbu),
        loudness: Some(-18.0), // Fixo para fallback de ganho, se não fornecido
    };

    // Obter array unidimensional de f32
    let mut weights = Vec::new();
    let pesos_raw = &data[weights_offset..];

    let float_count = pesos_raw.len() / 4;
    weights.reserve_exact(float_count);

    for i in 0..float_count {
        let chunk = &pesos_raw[i * 4..(i + 1) * 4];
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        weights.push(f32::from_le_bytes(bytes));
    }

    // População do padrão de arquitetura exigido pela compatibilidade local
    // Como arquivos ".namb baseados em emulação densa 'Standard'" são a premissa para
    // validação inter-compatível, fabricaremos uma simetria estática idêntica.
    let config = make_standard_wavenet_config();

    Ok(NamModelData {
        version: Some(version_str),
        architecture: "WaveNet".to_string(),
        config,
        weights,
        sample_rate: Some(sample_rate),
        metadata: Some(metadata),
    })
}

/// Cria o arranjo compatível com a simetria Standard de WaveNet da inferência.
///
/// **Decisão arquitetural:** Arquivos `.namb` são predominantemente
/// modelos WaveNet "Standard" (16ch, dilatações `[1..512]`×2). A especificação binária
/// não transporta explicitamente a topologia no cabeçalho — os pesos são interpretados
/// diretamente pela geometria fixa. Por isso, hardcodamos a configuração Standard
/// para garantir intercambiabilidade com o parser JSON.
fn make_standard_wavenet_config() -> NamConfig {
    let std_dilations = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    let l0 = NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        head_size: Some(8),
        channels: Some(16),
        kernel_size: Some(3),
        dilations: Some(std_dilations.clone()),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(false),
    };

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
        head_scale: Some(0.02),
        num_layers: None,
        hidden_size: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    /// Constrói um buffer NAMB válido com pesos e CRC32 corretos.
    fn build_valid_namb(w_floats: &[f32]) -> Vec<u8> {
        let weights_offset: usize = 80;
        let total_size = weights_offset + w_floats.len() * 4;
        let mut sim_data = vec![0u8; total_size];

        // Magic Number
        sim_data[0..4].copy_from_slice(&0x4E414D42u32.to_le_bytes());
        // Version = 1
        sim_data[4..6].copy_from_slice(&1u16.to_le_bytes());
        // Offset de pesos
        sim_data[12..16].copy_from_slice(&(weights_offset as u32).to_le_bytes());
        // Version String @32
        sim_data[32..37].copy_from_slice(b"1.0.0");
        // Frequência
        sim_data[64..68].copy_from_slice(&48000.0f32.to_le_bytes());
        // Input DBU = 12.0
        sim_data[68..72].copy_from_slice(&12.0f32.to_le_bytes());
        // Output DBU = -6.0
        sim_data[72..76].copy_from_slice(&(-6.0f32).to_le_bytes());

        // Pesos
        for (i, float_val) in w_floats.iter().enumerate() {
            let off = weights_offset + i * 4;
            sim_data[off..off + 4].copy_from_slice(&float_val.to_le_bytes());
        }

        // CRC32 IEEE 802.3 sobre os bytes de pesos
        let mut hasher = Hasher::new();
        hasher.update(&sim_data[weights_offset..]);
        let crc = hasher.finalize();
        sim_data[24..28].copy_from_slice(&crc.to_le_bytes());

        sim_data
    }

    #[test]
    fn test_parse_namb_standard() -> Result<()> {
        let w_floats = [0.1f32, -0.2f32, 2.5f32, 10.0f32];
        let sim_data = build_valid_namb(&w_floats);

        let parsed = parse_namb(&sim_data)?;

        // Validação idêntica:
        assert_eq!(parsed.architecture, "WaveNet");
        assert_eq!(parsed.config.layers.len(), 2);
        assert_eq!(parsed.config.layers[0].channels.unwrap(), 16);
        assert_eq!(parsed.weights.len(), 4);
        assert_eq!(parsed.weights, w_floats);

        assert_eq!(parsed.sample_rate.unwrap(), 48000.0);
        let meta = parsed.metadata.unwrap();
        assert_eq!(meta.input_level_dbu.unwrap(), 12.0);
        assert_eq!(meta.output_level_dbu.unwrap(), -6.0);
        assert_eq!(parsed.version.as_deref(), Some("1.0.0"));

        Ok(())
    }

    #[test]
    fn test_reject_truncated_file() {
        let data = vec![0u8; 40]; // Menor que o cabeçalho mínimo de 80 bytes
        let result = parse_namb(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("muito pequeno"), "Mensagem inesperada: {msg}");
    }

    #[test]
    fn test_reject_invalid_magic() {
        let w = [1.0f32];
        let mut data = build_valid_namb(&w);
        // Corrompe o magic number
        data[0..4].copy_from_slice(b"XXXX");
        let result = parse_namb(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("mágica inválida") || msg.contains("Assinatura"),
            "Mensagem inesperada: {msg}"
        );
    }

    #[test]
    fn test_reject_unsupported_version() {
        let w = [1.0f32];
        let mut data = build_valid_namb(&w);
        // Define versão 99 (não suportada)
        data[4..6].copy_from_slice(&99u16.to_le_bytes());
        let result = parse_namb(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Versão") || msg.contains("não suportada"),
            "Mensagem inesperada: {msg}"
        );
    }

    #[test]
    fn test_reject_corrupted_crc32() {
        let w = [1.0f32, 2.0, 3.0, 4.0];
        let mut data = build_valid_namb(&w);
        // Corrompe o CRC32 deliberadamente
        data[24..28].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let result = parse_namb(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("CRC32") || msg.contains("integridade"),
            "Mensagem inesperada: {msg}"
        );
    }
}
