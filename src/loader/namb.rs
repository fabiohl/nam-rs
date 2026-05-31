// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Carregador Binário para modelos `.namb`.
//!
//! Realiza a análise e deserialização direta, determinística e lock-free
//! a partir de um bloco binário para a estrutura `NamModelData`.

use super::nam_json::{NamConfig, NamLayerConfig, NamMetadata, NamModelData, WeightsLayout};
use anyhow::Result;
use log::info;

/// Erro tipado para parsing de arquivos `.namb`.
///
/// Cada variante corresponde a uma falha específica de integridade ou
/// formato do arquivo binário, permitindo diagnóstico preciso via
/// `downcast_ref` no módulo `loader`.
#[derive(Debug, thiserror::Error)]
pub enum NambError {
    /// Arquivo truncado: bytes insuficientes para o cabeçalho mínimo.
    #[error("file truncated: got {got} bytes, need at least {need}")]
    Truncated {
        /// Bytes disponíveis no arquivo.
        got: usize,
        /// Bytes mínimos necessários.
        need: usize,
    },

    /// Número mágico inválido (não é 0x4E414D42).
    #[error("invalid magic number: 0x{0:08X} (expected 0x4E414D42)")]
    InvalidMagic(u32),

    /// Versão do formato `.namb` não suportada.
    #[error("unsupported .namb version: {0}")]
    InvalidVersion(u16),

    /// Offset da seção de pesos além do tamanho do arquivo.
    #[error("weights offset {offset} out of file bounds (file size: {file_len})")]
    WeightsOffsetOutOfBounds {
        /// Offset declarado no cabeçalho.
        offset: usize,
        /// Tamanho total do arquivo em bytes.
        file_len: usize,
    },

    /// Offset da seção de pesos menor que o tamanho do cabeçalho.
    #[error("invalid weights offset {offset} (smaller than header size {header_size})")]
    InvalidWeightsOffset {
        /// Offset declarado no cabeçalho.
        offset: usize,
        /// Tamanho esperado do cabeçalho.
        header_size: usize,
    },

    /// Checksum CRC32 da seção de pesos não confere.
    #[error("CRC32 mismatch: got 0x{got:08X}, expected 0x{expected:08X}")]
    CrcMismatch {
        /// CRC calculado a partir dos dados.
        got: u32,
        /// CRC declarado no cabeçalho.
        expected: u32,
    },

    /// CRC32 ausente em arquivo NAMB v2+ (flag FLAG_HAS_CRC32 não setado).
    #[error("CRC32 flag missing in NAMB v{version} file (FLAG_HAS_CRC32 not set)")]
    CrcMissing {
        /// Versão do arquivo NAMB.
        version: u16,
    },
}

/// Calcula o CRC32 (IEEE 802.3) de um slice de bytes.
/// Substitui a dependência externa `crc32fast` por uma versão leve em software.
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

/// Flag bitmask para o campo `flags` do header NAMB.
pub const FLAG_HAS_CRC32: u8 = 0x01;

/// Cabeçalho binário fixo do formato `.namb`.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct NambHeader {
    /// Número mágico `0x4E414D42` ("NAMB" em ASCII).
    pub magic: u32,
    /// Versão do formato (1 = legada, 2 = com layout pre-transposto).
    pub version: u16,
    /// Layout dos pesos (apenas se version >= 2). Offset: 6.
    pub layout_type: u8,
    /// Flags de feature (bit 0 = FLAG_HAS_CRC32). Offset: 7.
    pub flags: u8,
    /// Reservado para expansão futura. Offset: 8.
    pub reserved_v2: [u8; 4],
    /// Offset (em bytes) do início da seção de pesos em relação ao início do arquivo.
    pub weights_offset: u32,
    /// Reservado para expansão futura.
    pub reserved1: [u32; 2],
    /// Soma de verificação CRC32 do bloco de pesos (opcional).
    pub crc32: u32,
    /// Reservado para expansão futura.
    pub reserved2: u32,
    /// String de versão informativa (Ex: "NAMB 2.0.0").
    pub version_str: [u8; 32],
    /// Frequência de amostragem padrão (Ex: 48000.0).
    pub sample_rate: f32,
    /// Nível de entrada dBu padrão (Ex: 12.0).
    pub input_level_dbu: f32,
    /// Nível de saída dBu padrão (Ex: 12.0).
    pub output_level_dbu: f32,
    /// Reservado (tamanho total do header deve ser pelo menos 80 bytes).
    pub reserved3: [u32; 1],
}

impl NambHeader {
    /// Valida se o cabeçalho possui o número mágico e versão suportada.
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

    /// Retorna o layout dos pesos conforme a versão e o flag.
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

/// Carrega um modelo no formato binário `.namb`.
pub fn parse_namb(data: &[u8]) -> Result<NamModelData> {
    let header_size = std::mem::size_of::<NambHeader>();
    if data.len() < header_size {
        return Err(NambError::Truncated {
            got: data.len(),
            need: header_size,
        }
        .into());
    }

    // 1. Lê o cabeçalho (Header)
    let header = unsafe { &*data.as_ptr().cast::<NambHeader>() };
    header.validate()?;

    // 2. Lê a seção de metadados JSON (opcional em .namb, mas comum)
    // Se weights_offset > header_size, há um JSON entre eles.
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
        // Truncar nulos se houver (o buffer NAMB costuma ser padded)
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

    // 3. Validação de Integridade (CRC32)
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

    // 4. Lê os pesos binários
    let pesos_raw = &data[weights_offset..];
    let float_count = pesos_raw.len() / 4;
    let mut weights = Vec::with_capacity(float_count);

    for chunk in pesos_raw.chunks_exact(4) {
        weights.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    // Popula metadados do header no NamModelData final
    let sample_rate_header = header.sample_rate;
    let input_level_header = header.input_level_dbu;
    let output_level_header = header.output_level_dbu;
    let version_header = header.version;

    model_data.weights = weights;
    model_data.sample_rate = Some(sample_rate_header);
    model_data.weights_layout = header.get_layout();

    // Atualiza metadados se existirem
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

    // Se a versão for nula (fallback), pega do header string
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

/// Cria um conjunto de dados "de reserva" (fallback).
/// Útil para arquivos .namb antigos que não descrevem sua própria estrutura.
fn make_fallback_model_data() -> NamModelData {
    NamModelData {
        version: None,
        architecture: "WaveNet".to_string(), // .namb legados são sempre WaveNet Standard
        config: make_standard_wavenet_config(),
        weights: Vec::new(),
        sample_rate: None,
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

/// Define o "gabarito" padrão para o algoritmo WaveNet.
/// É como definir o número de neurônios e conexões de um cérebro digital padrão.
fn make_standard_wavenet_config() -> NamConfig {
    // Dilatações: define o "alcance" da memória do algoritmo (essencial para capturar o timbre).
    let std_dilations = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    // Primeira camada do processamento.
    let l0 = NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        head_size: Some(8),
        channels: Some(16),   // "Largura" do processamento interno.
        kernel_size: Some(3), // Quantidade de amostras vizinhas analisadas de cada vez.
        dilations: Some(std_dilations.clone()),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(false),
    };

    // Segunda camada (geralmente idêntica à primeira em modelos Standard).
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
        head_scale: Some(0.02), // Ajuste final de volume para garantir consistência.
        num_layers: None,
        hidden_size: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constrói um arquivo .namb v1 (formato binário) em memória para fins de teste.
    /// É como "fabricar" um arquivo de mentira para ver se o programa consegue ler.
    fn build_valid_namb_v1(w_floats: &[f32]) -> Vec<u8> {
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size + w_floats.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        // Preenchemos o "cabeçalho" (a etiqueta de identificação do arquivo).
        header.magic = 0x4E414D42; // "NAMB" em código hexadecimal.
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.sample_rate = 48000.0;
        header.input_level_dbu = 12.0;
        header.output_level_dbu = -6.0;
        header.version_str[0..5].copy_from_slice(b"1.0.0");

        // Converte os números decimais (pesos) em bytes brutos.
        for (i, &f) in w_floats.iter().enumerate() {
            let offset = header_size + i * 4;
            data[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
        }

        // Gera um "lacre de segurança" (CRC32) para garantir que os dados não foram alterados.
        header.crc32 = crc32_ieee(&data[header_size..]);
        data
    }

    #[test]
    fn test_parse_namb_v1() -> Result<()> {
        // Testamos se o carregador consegue ler corretamente um arquivo v1 básico.
        let w = [0.1f32, -0.5f32, 1.0f32];
        let data = build_valid_namb_v1(&w);
        let parsed = parse_namb(&data)?;

        // Verificamos se os valores lidos são idênticos aos que gravamos.
        assert_eq!(parsed.weights, w);
        assert_eq!(parsed.weights_layout, WeightsLayout::Original);
        assert_eq!(parsed.sample_rate, Some(48000.0));
        Ok(())
    }

    #[test]
    fn test_parse_namb_v2_gate_major() -> Result<()> {
        // Testamos se o carregador reconhece o novo formato v2 (com layout otimizado).
        let header_size = std::mem::size_of::<NambHeader>();
        let w = [0.0f32; 4];
        let mut data = vec![0u8; header_size + w.len() * 4];
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 2;
        header.layout_type = 1; // 1 indica "GateMajorLstm" (layout otimizado para LSTM)
        header.flags = FLAG_HAS_CRC32;
        header.weights_offset = header_size as u32;

        // Escreve os pesos no buffer
        for (i, &f) in w.iter().enumerate() {
            let offset = header_size + i * 4;
            data[offset..offset + 4].copy_from_slice(&f.to_le_bytes());
        }
        header.crc32 = crc32_ieee(&data[header_size..]);

        let parsed = parse_namb(&data)?;
        // Garantimos que o programa entendeu que este arquivo precisa de uma reorganização especial.
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
        header.flags = 0; // FLAG_HAS_CRC32 NÃO setado
        header.weights_offset = header_size as u32;
        header.crc32 = 0xDEADBEEF;

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Erro deveria ser NambError::CrcMissing");
        assert!(
            matches!(namb_err, NambError::CrcMissing { version: 2 }),
            "Esperado CrcMissing, obtido: {:?}",
            namb_err
        );
    }

    #[test]
    fn test_v2_crc32_zero_legitimate_passes() -> Result<()> {
        // CRC32 de um slice vazio é 0 (propriedade do algoritmo IEEE 802.3).
        // Com FLAG_HAS_CRC32 setado e crc32=0 legítimo, o parser deve aceitar.
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size]; // Sem pesos → crc32_ieee(&[]) == 0
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 2;
        header.layout_type = 1;
        header.flags = FLAG_HAS_CRC32;
        header.weights_offset = header_size as u32;
        header.crc32 = 0; // CRC32 legítimo para slice vazio

        let parsed = parse_namb(&data)?;
        assert!(parsed.weights.is_empty());
        Ok(())
    }

    #[test]
    fn test_v1_crc32_zero_warns_but_passes() -> Result<()> {
        // v1 com crc==0 (sentinel) deve passar com warning, não bloquear.
        let header_size = std::mem::size_of::<NambHeader>();
        let mut data = vec![0u8; header_size + 4]; // 1 float dummy
        let header = unsafe { &mut *data.as_mut_ptr().cast::<NambHeader>() };

        header.magic = 0x4E414D42;
        header.version = 1;
        header.weights_offset = header_size as u32;
        header.crc32 = 0; // Sentinel: CRC ausente em v1

        // Escreve um peso dummy
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

        header.magic = 0x424D414E; // "BMAN" — não mais aceito (S5.T09)
        header.version = 1;
        header.weights_offset = header_size as u32;

        let err = parse_namb(&data).unwrap_err();
        let namb_err = err
            .downcast_ref::<NambError>()
            .expect("Erro deveria ser NambError::InvalidMagic");
        assert!(
            matches!(namb_err, NambError::InvalidMagic(m) if *m == 0x424D414E),
            "Esperado InvalidMagic(0x424D414E), obtido: {:?}",
            namb_err
        );
    }
}
