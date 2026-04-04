// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Estruturas de dados para comunicação lock-free entre a thread DSP e a thread de I/O.
//! Contém buffers alinhados a linhas de cache via const generics para mitigar False Sharing
//! nas caches L1/L2, além de flags atômicas compartilhadas para coordenação entre threads.

use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::atomic::{AtomicBool, AtomicU64};

/// Número máximo de amostras f32 intercaladas por bloco de áudio, definido em tempo de compilação.
pub const MAX_BLOCK_SIZE: usize = 8192;

/// Número de slots no ring buffer SPSC.
/// Cada slot contém um `RingPayload<MAX_BLOCK_SIZE>` (~32 KiB).
pub const RING_CAPACITY: usize = 8192;

/// Flag global para shutdown coordenado e gracioso entre todas as threads.
/// Definida como `true` pelo handler de CTRL+C.
pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Contador atômico de overruns do ring buffer (falhas de push no lado do produtor DSP).
/// Reportado ao usuário no shutdown para informar possível perda de dados de áudio.
pub static OVERRUN_COUNT: AtomicU64 = AtomicU64::new(0);

/// Bloco de áudio alinhado para prevenir bouncing de cache line (False Sharing)
/// entre as threads DSP e I/O. As linhas de cache padrão de CPU têm 64 bytes,
/// mas o alinhamento a 128 bytes (`align(128)`) impede que o prefetcher espacial
/// do hardware puxe variáveis adjacentes intensamente mutadas da thread de I/O
/// para dentro do perímetro da cache da thread DSP.
///
/// `valid_len` rastreia quantas amostras em `data` contêm áudio real — o restante
/// é preenchimento zero e não deve ser escrito no arquivo WAV.
#[repr(align(128))]
pub struct AlignedBlock<const SIZE: usize> {
    /// Amostras de áudio f32 intercaladas (canal L, canal R, L, R, ...).
    pub data: [f32; SIZE],
    /// Quantidade de amostras f32 válidas em `data[0..valid_len]`.
    pub valid_len: usize,
}

impl<const SIZE: usize> AlignedBlock<SIZE> {
    /// Cria um novo bloco zerado sem alocação na heap.
    pub const fn new() -> Self {
        Self {
            data: [0.0; SIZE],
            valid_len: 0,
        }
    }
}

/// Payload de metadados enviado quando o formato do stream é inicializado ou alterado.
/// Alinhado a 128 bytes para conformar-se ao layout anti-false-sharing do ring buffer.
#[repr(align(128))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioMetadata {
    /// Taxa de amostragem do stream de áudio (ex: 44100.0, 48000.0).
    pub sample_rate: f32,
    /// Profundidade de bits por amostra (ex: 32 para float IEEE 754).
    pub bit_depth: u16,
    /// Número de canais de áudio (ex: 2 para estéreo).
    pub channels: u16,
}

/// Payload principal trocado via ring buffer SPSC.
/// Alinhado a 128 bytes para garantir que cada slot ocupe linhas de cache distintas,
/// prevenindo false sharing entre os núcleos do produtor (DSP) e consumidor (I/O).
#[repr(align(128))]
pub enum RingPayload<const SIZE: usize> {
    /// Bloco de áudio contendo amostras f32 intercaladas prontas para gravação.
    Audio(AlignedBlock<SIZE>),
    /// Metadados do stream (sample rate, bit depth, canais) para configurar o arquivo WAV.
    Metadata(AudioMetadata),
    /// Sinal de encerramento do stream — instrui a thread I/O a fechar o WAV atual.
    StreamStop,
}

/// Cria um ring buffer SPSC (produtor-único/consumidor-único) estritamente dimensionado
/// para estruturas `RingPayload`. O parâmetro `capacity` define o número de slots.
pub fn create_audio_ring_buffer<const BLOCK_SIZE: usize>(
    capacity: usize,
) -> (
    Producer<RingPayload<BLOCK_SIZE>>,
    Consumer<RingPayload<BLOCK_SIZE>>,
) {
    RingBuffer::new(capacity)
}
