<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# NAMB Binary Format Specification

Especificação do formato binário `.namb` para distribuição otimizada de
modelos NAM (Neural Amp Modeler).

## Header (80 bytes, packed)

| Offset | Tamanho | Campo            | Descrição                                    |
|--------|---------|------------------|----------------------------------------------|
| 0      | 4       | magic            | `0x4E414D42` ("NAMB" LE)                     |
| 4      | 2       | version          | 1 (legado) ou 2 (pré-transposto)             |
| 6      | 1       | layout_type      | 0=Original, 1=GateMajorLstm, 2=Interleaved4WaveNet |
| 7      | 1       | flags            | Bit 0 = `FLAG_HAS_CRC32` (0x01)              |
| 8      | 4       | reserved_v2      | Reservado para expansão futura               |
| 12     | 4       | weights_offset   | Offset do bloco de pesos (LE)                |
| 16     | 8       | reserved1        | Reservado                                    |
| 24     | 4       | crc32            | CRC32 IEEE 802.3 do bloco de pesos (LE)      |
| 28     | 4       | reserved2        | Reservado                                    |
| 32     | 32      | version_str      | String informativa (ex: "NAMB v2 (... )")   |
| 64     | 4       | sample_rate      | Taxa de amostragem (f32 LE)                  |
| 68     | 4       | input_level_dbu  | Nível de entrada dBu (f32 LE)                |
| 72     | 4       | output_level_dbu | Nível de saída dBu (f32 LE)                  |
| 76     | 4       | reserved3        | Reservado                                    |

## Flags

| Bit | Nome             | Descrição                                    |
|-----|------------------|----------------------------------------------|
| 0   | FLAG_HAS_CRC32   | CRC32 presente e deve ser validado           |
| 1-7 | —                | Reservado                                    |

## Seções pós-header

1. **JSON de metadados**: UTF-8 entre offset 80 e `weights_offset`.
   Pode ser omitido (nesse caso `weights_offset == header_size`).

2. **Bloco de pesos**: `f32` little-endian contíguos a partir de `weights_offset`.

## Validação de Integridade (CRC32)

### v2+ (version >= 2)

- `FLAG_HAS_CRC32` **obrigatório**: decodificador rejeita com `NambError::CrcMissing`.
- CRC32 validado **sempre** (mesmo se `crc32 == 0`, caso legítimo de coincidência).
- Algoritmo: CRC32 IEEE 802.3 (polinômio 0xEDB88320, init 0xFFFFFFFF, xorout 0xFFFFFFFF).

### v1 (legado)

- `crc32 == 0` atua como sentinel "CRC ausente": validação é pulada com `log::warn!`
  (compatibilidade com arquivos pré-flag).
- `crc32 != 0`: validação normal por CRC32 IEEE 802.3.

## Evolução do Header

| Versão | Mudança                                         |
|--------|-------------------------------------------------|
| v1     | Header original, `reserved_v2: [u8; 5]` no offset 7. CRC opcional via sentinel `crc32==0`. |
| v2     | Offset 7 renomeado para `flags: u8`, `reserved_v2` reduzido a 4 bytes. `FLAG_HAS_CRC32` obrigatório. `layout_type` ativo para pesos pré-transpostos. |
