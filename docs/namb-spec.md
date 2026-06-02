<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# NAMB Binary Format Specification

Especificação do formato binário `.namb` para distribuição otimizada de
modelos NAM (Neural Amp Modeler).

## 1. Header (80 bytes, packed)

O cabeçalho é uma struct `#[repr(C, packed)]` — **sem padding implícito**.
Todos os inteiros multi-byte estão em **little-endian**.

| Offset | Tamanho | Campo              | Tipo      | Descrição                                                                                                 |
| ------ | ------- | ------------------ | --------- | --------------------------------------------------------------------------------------------------------- |
| 0x00   | 4       | `magic`            | `u32`     | Número mágico `0x4E414D42` ("NAMB" em ASCII LE). Valores diferentes são rejeitados com `InvalidMagic`.    |
| 0x04   | 2       | `version`          | `u16`     | Versão do formato: `1` = legado, `2` = pré-transposto                                                     |
| 0x06   | 1       | `layout_type`      | `u8`      | Layout dos pesos (ativo p/ `version >= 2`): `0`=Original, `1`=GateMajorLstm, `2`=Interleaved4WaveNet      |
| 0x07   | 1       | `flags`            | `u8`      | Bitmask de flags de funcionalidade (ver §2)                                                               |
| 0x08   | 4       | `reserved_v2`      | `[u8;4]`  | Reservado para expansão futura                                                                            |
| 0x0C   | 4       | `weights_offset`   | `u32`     | Offset do início do bloco de pesos a partir do início do arquivo. Deve ser `>= 0x50` (80).                |
| 0x10   | 8       | `reserved1`        | `[u32;2]` | Reservado para expansão futura                                                                            |
| 0x18   | 4       | `crc32`            | `u32`     | CRC32 IEEE 802.3 do bloco de pesos (faixa: `[weights_offset..EOF]`). Semântica varia por versão (ver §6). |
| 0x1C   | 4       | `reserved2`        | `u32`     | Reservado para expansão futura                                                                            |
| 0x20   | 32      | `version_str`      | `[u8;32]` | String de versão informativa, null-terminated (ex: `"NAMB v2 (GateMajorLstm)\0"`).                        |
| 0x40   | 4       | `sample_rate`      | `f32`     | Taxa de amostragem padrão (ex: `48000.0`)                                                                 |
| 0x44   | 4       | `input_level_dbu`  | `f32`     | Nível de entrada em dBu (ex: `12.0`)                                                                      |
| 0x48   | 4       | `output_level_dbu` | `f32`     | Nível de saída em dBu (ex: `-6.0`)                                                                        |
| 0x4C   | 4       | `reserved3`        | `[u32;1]` | Reservado para expansão futura                                                                            |

**Tamanho total do cabeçalho: 80 bytes (0x50).**

### 1.1 Número mágico

O valor canônico é `0x4E414D42` (bytes `42 4D 41 4E` = `"NAMB"` em little-endian).
Qualquer outro valor é rejeitado com `NambError::InvalidMagic` (E1202).
O encoder **sempre** grava `0x4E414D42`.

### 1.2 Versionamento

| Versão | Descrição                                                                                                                                                                                                      |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`    | Formato legado. Não há campo `flags` — o byte no offset 0x07 é parte de `reserved_v2: [u8;5]`. `layout_type` era parte da região reservada, tratado como `0` (Original). CRC opcional via sentinel `crc32==0`. |
| `2`    | Formato com pré-transposição. Byte 0x07 é `flags: u8`, `reserved_v2` reduzido para 4 bytes. `layout_type` ativo. `FLAG_HAS_CRC32` obrigatório.                                                                 |

Valores de `version` diferentes de `1` e `2` são rejeitados com `NambError::InvalidVersion`.

## 2. Flags

| Bit | Nome             | Valor  | Descrição                                                                                        |
| --- | ---------------- | ------ | ------------------------------------------------------------------------------------------------ |
| 0   | `FLAG_HAS_CRC32` | `0x01` | Indica que o campo `crc32` contém um CRC válido que **deve** ser verificado (obrigatório em v2+) |
| 1-7 | —                | —      | Reservado para uso futuro. **Devem ser 0.**                                                      |

## 3. Layouts de Pesos

### 3.1 `Original` (`layout_type = 0`)

Formato herdado do ecossistema NAM em C++. Os pesos são armazenados exatamente
como aparecem no JSON `.nam`, sem transposição.

#### LSTM — Ordem Original

```text
Para cada camada l (0..num_layers-1):
  input_hidden_weights:  [4*H][input+hidden]  f32 row-major  (ex: [gate][hidden][IH])
  bias:                  [4*H]                f32
  hidden_init:           [H]                  f32
  cell_init:             [H]                  f32

Após todas as camadas:
  head_weights:  [H]  f32
  head_bias:     1    f32
```

Onde:

- `H` = `hidden_size`
- `input` = `1` (camada 1) ou `H` (camadas subsequentes)
- `IH` = `input + H`
- `4*H` corresponde às 4 portas do LSTM (input, forget, cell, output)

A convenção C++ armazena `[gate][hidden][IH]` row-major.

#### WaveNet — Ordem Original

```text
Array 1 (layer 0):
  Rechannel:      [CH][IN=1]          f32 row-major  [OUT][IN]
  Para cada dilatação d:
    Conv1d:       [CO][CH][K]         f32 row-major  [OUT][IN][K]
    Conv1d bias:  [CO]                f32
    Input Mixin:  [CH][COND=1]        f32 row-major  [OUT][IN]
    1x1:          [CH][CH]            f32 row-major  [OUT][IN]
    1x1 bias:     [CH]                f32
  Head Rechannel: [HEAD][CH]          f32 row-major  [OUT][IN]  (sem bias no array 1)

Array 2 (layer 1):
  Rechannel:      [HEAD][IN=1]        f32 row-major
  Para cada dilatação d:
    Conv1d:       [CO][HEAD][K]       f32 row-major
    Conv1d bias:  [CO]                f32
    Input Mixin:  [HEAD][COND=1]      f32 row-major
    1x1:          [HEAD][HEAD]        f32 row-major
    1x1 bias:     [HEAD]              f32
  Head Rechannel: [1][HEAD]           f32 row-major  (com bias no array 2)

Head Scale: 1 f32
```

Onde:

- `CH` = canais internos da camada
- `CO` = canais de saída da convolução (`CH*2` se gated, senão `CH`)
- `HEAD` = canais de saída da camada (head_size)
- `K` = tamanho do kernel de convolução

### 3.2 `GateMajorLstm` (`layout_type = 1`)

Layout otimizado para LSTM com transposição `[Gate][Hidden][IH]` → `[Gate][IH][Hidden]`.
Permite que o decodificador leia os pesos diretamente sem transposição em tempo de carga.

**Encoder** (S3.T03): os pesos de cada camada são intercalados na ordem correta
— após escrever `W_l, bias_l`, os estados `hidden_init_l, cell_init_l` são
escritos imediatamente, garantindo que o decodificador leia a sequência completa
por camada.

```text
Para cada camada l (0..num_layers-1):
  gate_major_weights:  [4*H][IH]  →  transposto para [Gate][IH][H]
    Original: raw[k * H * IH + i * IH + j]  →  [gate=k][hidden=i][input=j]
    Transposto: out[k * H * IH + j * H + i]  →  [gate=k][input=j][hidden=i]
  bias:                [4*H]      f32  (não transposto)
  hidden_init:         [H]        f32  (não transposto)
  cell_init:           [H]        f32  (não transposto)

Após todas as camadas:
  head_weights:  [H]  f32  (não transposto)
  head_bias:     1    f32  (não transposto)
```

O decodificador, ao detectar `layout_type = 1`, faz a leitura direta dos pesos
sem transposição adicional, pois estes já estão no formato de consumo do runtime.

### 3.3 `Interleaved4WaveNet` (`layout_type = 2`)

Layout otimizado para WaveNet com duas transformações:

1. **Conv1d interleaved-4**: pesos rearranjados para `[BLOCK][K][IN][4]`
   permitindo processamento SIMD de 4 canais por vez.
2. **Camadas densas transpostas**: `[OUT][IN]` → `[IN][OUT]` para acesso
   sequencial eficiente no decodificador.

```text
Array 1 (layer 0):
  Rechannel:      [IN=1][CH]          f32  transposto [OUT][IN]→[IN][OUT]
  Para cada dilatação d:
    Conv1d:       [BLOCK][K][CH][4]   f32  interleaved-4 (ver abaixo)
    Conv1d bias:  [CO]                f32  (não transposto)
    Input Mixin:  [COND=1][CH]        f32  transposto [OUT][IN]→[IN][OUT]
    1x1:          [CH][CH]            f32  transposto [OUT][IN]→[IN][OUT]
    1x1 bias:     [CH]                f32  (não transposto)
  Head Rechannel: [CH][HEAD]          f32  transposto [OUT][IN]→[IN][OUT]
  Head bias:      [HEAD]              f32  (se head_bias=true)

Array 2 (layer 1):
  ... (mesma estrutura com IN/OUT adaptados)
```

#### Conv1d Interleaved-4

Dado `CO` = canais de saída, `K` = kernel, `IN` = canais de entrada:

```text
num_blocks = ceil(CO / 4)
total_padded = num_blocks * 4 * IN * K  f32

Layout: [BLOCK][K][IN][LANE]  (innermost = lane de 4)

Para bloco b em 0..num_blocks-1:
  Para kernel ki em 0..K-1:
    Para canal de entrada in_c em 0..IN-1:
      Para lane em 0..3:
        out_c = b * 4 + lane
        Se out_c < CO:
          valor = original[(out_c * IN + in_c) * K + ki]
        Senão:
          valor = 0.0  (zero-padding)
```

**Padding implícito** (S3.T04): quando `CO % 4 != 0`, o encoder preenche os
slots excedentes com `0.0` para manter o formato interleaved uniforme. O
decodificador sempre lê o tamanho preenchido (`num_blocks * 4 * IN * K`) mas só
acessa `CO` canais lógicos. Isto garante que geometrias futuras com `CO % 4 != 0`
funcionem corretamente sem código condicional no decoder.

## 4. Seções Pós-Header

### 4.1 JSON de Metadados (opcional)

UTF-8 entre offset 80 (0x50) e `weights_offset`. Se `weights_offset > 80`,
os bytes intermediários contêm a serialização JSON de `NamModelData` **sem o
campo `weights`** (serializado via `serde_json`).

O JSON é null-terminated; bytes nulos após o terminador são ignorados. Se o JSON
estiver vazio ou `weights_offset == 80`, o decodificador sintetiza uma
configuração fallback (WaveNet Standard).

### 4.2 Bloco de Pesos

Array contíguo de `f32` little-endian a partir de `weights_offset` até o fim
do arquivo. Cada `f32` ocupa 4 bytes. O número de floats é
`(file_len - weights_offset) / 4`.

Bytes residuais que não completem 4 bytes são **descartados silenciosamente**
(não constituem erro).

Os `f32` são lidos via `f32::from_le_bytes()` em chunks de 4 bytes. Não há
alinhamento ou padding dentro do bloco — é um stream contíguo puro.

## 5. Estrutura do Arquivo

```text
┌──────────────────────┐
│   NambHeader         │  offset 0x0000 .. 0x004F  (80 bytes)
│   (80 bytes)         │
├──────────────────────┤
│   JSON Metadata      │  offset 0x0050 .. (weights_offset - 1)  OPCIONAL
│   (UTF-8, null-term) │
├──────────────────────┤
│   Weights Blob       │  offset weights_offset .. EOF  OBRIGATÓRIO
│   (f32 LE contíguo) │
└──────────────────────┘
```

## 6. Validação de Integridade (CRC32)

### 6.1 Algoritmo

- **Padrão:** CRC32 IEEE 802.3 (CRC-32/ISO-HDLC, PKZIP, Ethernet)
- **Polinômio:** `0xEDB88320` (refletido)
- **Init:** `0xFFFFFFFF`
- **XOR out:** `0xFFFFFFFF`
- **Implementação:** software puro (sem dependência externa), 8 bits por iteração

### 6.2 Faixa de Cobertura

O CRC32 cobre **apenas o bloco de pesos** (`data[weights_offset..]`).
Cabeçalho e JSON de metadados **não** são cobertos pelo CRC.

### 6.3 Semântica por Versão

| Condição                         | Comportamento                                                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------- |
| v1, `crc32 == 0`                 | CRC **ignorado** (sentinel). Emite `log::warn!`. Compatibilidade com arquivos pré-flag.     |
| v1, `crc32 != 0`                 | CRC **validado** normalmente.                                                               |
| v2+, `FLAG_HAS_CRC32` não setado | **Rejeitado** com `NambError::CrcMissing { version }`.                                      |
| v2+, `FLAG_HAS_CRC32` setado     | CRC **validado** sempre (incluindo `crc32 == 0` como caso legítimo de coincidência ~1/2³²). |

### 6.4 Comportamento do Encoder

- v1: `flags = 0`, CRC sempre calculado e gravado.
- v2+: `flags = FLAG_HAS_CRC32` (0x01), CRC sempre calculado e gravado.

## 7. Exemplos Hexadecimais

### 7.1 Header mínimo (v2, Original, sem JSON)

```text
Offset  Hex                                           Decodificação
------  --------------------------------------------  ------------------------------
0x00    42 4D 41 4E                                   magic = "NAMB" (0x4E414D42) LE
0x04    02 00                                         version = 2
0x06    00                                            layout_type = Original (0)
0x07    01                                            flags = FLAG_HAS_CRC32 (0x01)
0x08    00 00 00 00                                   reserved_v2
0x0C    50 00 00 00                                   weights_offset = 80 (0x50)
0x10    00 00 00 00 00 00 00 00                       reserved1
0x18    XX XX XX XX                                   crc32 (calculado)
0x1C    00 00 00 00                                   reserved2
0x20    4E 41 4D 42 20 76 32 20 28 4F 72 69 67 69     version_str = "NAMB v2 (Original)\0..."
        6E 61 6C 29 00 00 00 00 00 00 00 00 00 00
        00 00 00 00 00 00
0x40    00 00 7F 47                                   sample_rate = 48000.0 (f32 LE: 0x477F0000)
0x44    00 00 40 41                                   input_level_dbu = 12.0 (f32 LE: 0x41400000)
0x48    00 00 C0 C0                                   output_level_dbu = -6.0 (f32 LE: 0xC0C00000)
0x4C    00 00 00 00                                   reserved3
```

### 7.2 GateMajorLstm — Exemplo microscópico (H=2, 1-layer)

Modelo com `hidden_size=2`, 1 camada, `input=1`:

```text
Original (Layout=0):
  pesos [4*H][IH=3] = 24 f32: [g0h0i0, g0h0i1, g0h0i2, g0h1i0, g0h1i1, g0h1i2,
                                 g1h0i0, g1h0i1, g1h0i2, g1h1i0, g1h1i1, g1h1i2,
                                 g2h0i0, g2h0i1, g2h0i2, g2h1i0, g2h1i1, g2h1i2,
                                 g3h0i0, g3h0i1, g3h0i2, g3h1i0, g3h1i1, g3h1i2]

GateMajorLstm (Layout=1):
  pesos transpostos [Gate][IH][H] = 24 f32: [g0i0h0, g0i0h1, g0i1h0, g0i1h1, g0i2h0, g0i2h1,
                                              g1i0h0, g1i0h1, g1i1h0, g1i1h1, g1i2h0, g1i2h1,
                                              g2i0h0, g2i0h1, g2i1h0, g2i1h1, g2i2h0, g2i2h1,
                                              g3i0h0, g3i0h1, g3i1h0, g3i1h1, g3i2h0, g3i2h1]
```

Se os pesos originais fossem `1..24` (f32), a leitura do original seria por
linha: `1,2,3,4,5,6, 7,8,...`. O GateMajorLstm seria: `1,4,2,5,3,6, 7,10,...`.

**Ordem completa no bloco de pesos (1-layer, H=2):**

```text
Offset no bloco  Conteúdo
---------------  --------
+0               gate_major_weights[24 f32]
+96              bias[8 f32]
+128             hidden_init[2 f32]
+136             cell_init[2 f32]
+144             head_weights[2 f32]
+152             head_bias[1 f32]
```

### 7.3 Interleaved4WaveNet — Exemplo de Conv1d com padding (CO=6, K=3, IN=2)

`CO=6` canais de saída não são múltiplos de 4. `num_blocks = ceil(6/4) = 2`.

```text
num_blocks = 2, padded_total = 2 * 4 * 2 * 3 = 48 f32

Original (Layout=0): [OUT][IN][K] = 36 f32
  out0: in0k0, in0k1, in0k2, in1k0, in1k1, in1k2
  out1: in0k0, in0k1, in0k2, in1k0, in1k1, in1k2
  ...
  out5: in0k0, in0k1, in0k2, in1k0, in1k1, in1k2

Interleaved4WaveNet (Layout=2): [BLOCK][K][IN][LANE] = 48 f32

  Bloco 0 (cobre out0..out3):
    k=0, in0: out0in0k0, out1in0k0, out2in0k0, out3in0k0  (lane 0..3)
    k=0, in1: out0in1k0, out1in1k0, out2in1k0, out3in1k0
    k=1, in0: out0in0k1, out1in0k1, out2in0k1, out3in0k1
    k=1, in1: out0in1k1, out1in1k1, out2in1k1, out3in1k1
    k=2, in0: out0in0k2, out1in0k2, out2in0k2, out3in0k2
    k=2, in1: out0in1k2, out1in1k2, out2in1k2, out3in1k2

  Bloco 1 (cobre out4..out5, lanes 2-3 são zero-padding):
    k=0, in0: out4in0k0, out5in0k0, 0.0, 0.0
    k=0, in1: out4in1k0, out5in1k0, 0.0, 0.0
    k=1, in0: out4in0k1, out5in0k1, 0.0, 0.0
    k=1, in1: out4in1k1, out5in1k1, 0.0, 0.0
    k=2, in0: out4in0k2, out5in0k2, 0.0, 0.0
    k=2, in1: out4in1k2, out5in1k2, 0.0, 0.0
```

## 8. Erros Tipados (`NambError`)

O parsing NAMB utiliza erros tipados via `thiserror` (S5.T02) para
diagnóstico preciso. O módulo `loader` faz downcast via `downcast_ref` para
mapear cada variante ao código `NamErrorCode` correspondente.

| Variante                                        | Código | Mnemônico                  | Condição                                         |
| ----------------------------------------------- | ------ | -------------------------- | ------------------------------------------------ |
| `Truncated { got, need }`                       | E1204  | `NAMB_TRUNCATED`           | Arquivo < 80 bytes (tamanho mínimo do cabeçalho) |
| `InvalidMagic(u32)`                             | E1202  | `NAMB_INVALID_MAGIC`       | Magic não é `0x4E414D42`                         |
| `InvalidVersion(u16)`                           | E1203  | `NAMB_UNSUPPORTED_VERSION` | Versão diferente de 1 e 2                        |
| `WeightsOffsetOutOfBounds { offset, file_len }` | E1204  | `NAMB_TRUNCATED`           | `weights_offset` além do tamanho do arquivo      |
| `InvalidWeightsOffset { offset, header_size }`  | E1204  | `NAMB_TRUNCATED`           | `weights_offset` < 80 (menor que o cabeçalho)    |
| `CrcMismatch { got, expected }`                 | E1201  | `NAMB_CRC32_MISMATCH`      | CRC calculado ≠ CRC do cabeçalho                 |
| `CrcMissing { version }`                        | E1205  | `NAMB_CRC32_MISSING`       | Arquivo v2+ sem `FLAG_HAS_CRC32` setado          |

Limite global: `MAX_MODEL_BYTES = 256 MiB`. Arquivos maiores que este limite
são rejeitados com `NamErrorCode::ModelTooLarge` (E1304).

## 9. Política de Evolução e Compatibilidade

### 9.1 Compatibilidade Retroativa

- **v1**: mantido para leitura de arquivos legados. CRC opcional via sentinel `crc32==0`.
- **v2**: `FLAG_HAS_CRC32` obrigatório. `layout_type` ativo.
- **Padding zero Interleaved-4** (S3.T04): alteração transparente. Modelos NAMB v2
  produzidos antes do fix permanecem válidos (afetam apenas geometrias `CO % 4 != 0`,
  inexistentes no catálogo atual). Documentado como bump implícito sem mudança de versão.

### 9.2 Expansão Futura

- **Novos `layout_type`**: valores `3..=255` reservados para layouts futuros.
  Valores desconhecidos são tratados como `Original` pelo decodificador.
- **Novos flags**: bits `1..=7` do campo `flags` reservados. Devem ser `0` em
  arquivos atuais. Decodificadores futuros podem definir novos bits com
  fallback para comportamento atual quando não setados.
- **Campos `reserved_*`**: devem ser `0`. Reservados para adição de campos
  sem quebra de compatibilidade binária do cabeçalho.
- **Novas versões**: versões `>= 3` são reservadas para mudanças estruturais
  que exijam revisão do formato de cabeçalho ou bloco de pesos.

## 10. Constantes de Referência

| Constante           | Valor                   | Local                               |
| ------------------- | ----------------------- | ----------------------------------- |
| `FLAG_HAS_CRC32`    | `0x01` (u8)             | `src/loader/namb.rs:98`             |
| `MAX_MODEL_BYTES`   | `268_435_456` (256 MiB) | `src/loader/mod.rs:27`              |
| Header size         | `80` bytes (0x50)       | `std::mem::size_of::<NambHeader>()` |
| CRC polynomial      | `0xEDB88320`            | `src/loader/namb.rs:80`             |
| CRC init            | `0xFFFFFFFF`            | `src/loader/namb.rs:75`             |
| CRC xorout          | `0xFFFFFFFF`            | `src/loader/namb.rs:83`             |
| Magic LE            | `0x4E414D42`            | `src/loader/namb.rs:104`            |
| Default sample rate | `48000.0` (f32)         | `src/loader/mod.rs:25`              |
| Default input dBu   | `12.0` (f32)            | `src/loader/mod.rs:21`              |
| Default output dBu  | `-6.0` (f32)            | `src/loader/namb_encoder.rs:64`     |
| Default loudness    | `-18.0` (f32)           | `src/loader/mod.rs:23`              |

## 11. Referências

- Decoder NAMB: `src/loader/namb.rs`
- Encoder NAMB: `src/loader/namb_encoder.rs`
- Definição de layouts: `src/loader/nam_json/data.rs` (`WeightsLayout`)
- Decoder LSTM: `src/loader/dispatcher/lstm.rs`
- Decoder WaveNet: `src/loader/dispatcher/wavenet/mod.rs`
- Mapeamento de erros: `src/loader/mod.rs`
- Códigos de erro: `src/common/diagnostics/error_codes.rs` (`NamErrorCode`)
- Testes round-trip: `tests/namb_v2_roundtrip.rs`, `tests/namb_v2_validation.rs`
- Arquitetura: `docs/architecture.md`
- Planejamento: `TODO-sprints.md` (S3.T03, S3.T04, S5.T02, S5.T03)
