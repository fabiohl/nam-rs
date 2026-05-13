// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Tipos comuns, constantes e estado de camada para módulos WaveNet.

use crate::dsp::vring::VirtualRingBuffer;

/// Máximo de frames a processar em um pulso do callback.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Padding temporal circular das memórias no framework de Ring Buffers.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;

/// Contexto de processamento para otimizar a passagem de parâmetros no hot-path da WaveNet.
/// Unifica as necessidades dos modelos estáticos (const generic) e dinâmicos.
pub struct WavenetProcessContext<'a> {
    /// Buffer de condicionamento (sidechain).
    pub condition: &'a [f32],
    /// Buffer de condicionamento pré-convertido em BF16.
    pub condition_bf16: &'a [u16],
    /// Acumulador Head (Skip-Connection).
    pub head_input: &'a mut [f32],
    /// Buffer de saída da camada (para a próxima camada ou output final).
    pub output: &'a mut [f32],
    /// Buffer de saída opcional em BF16 (para a próxima camada em CPUs BF16).
    pub output_bf16: Option<&'a mut [u16]>,
    /// Buffer circular da camada corrente (delay line).
    pub layer_buffer: &'a [f32],
    /// Buffer circular (fita de retardo) em BF16.
    pub layer_buffer_bf16: &'a [u16],
    /// Índice inicial no buffer circular.
    pub buffer_start: usize,
    /// Número de frames a processar.
    pub num_frames: usize,
    /// Buffer temporário na stack para cálculos intermediários.
    pub block: &'a mut [f32],
}

/// Gerencia a memória buffer de uma célula WaveNet.
///
/// Alinhamento de 64B (cache line) é suficiente pois esta struct vive exclusivamente
/// na thread DSP — não há compartilhamento inter-thread que exija 128B anti-false-sharing.
#[repr(align(64))]
#[derive(Clone)]
pub struct WaveNetLayerState {
    /// Buffer Circular Virtual (zero alocações em contexto DSP, eliminação de rewind).
    pub layer_buffer: VirtualRingBuffer<f32>,
    /// Buffer Circular Virtual em BF16 para processamento VNNI.
    pub layer_buffer_bf16: VirtualRingBuffer<u16>,
    /// Ponteiro numérico do frame atual (avança a cada frame processado).
    pub buffer_start: usize,
    /// Dimensão física do espaço vetorial receptivo (tamanho do histórico de dilatação).
    pub receptive_field_size: usize,
}

impl WaveNetLayerState {
    /// Construtor alocador estático do Estado (executar antes do Thread DSP).
    pub fn new(channels: usize, receptive_field_size: usize, alloc_num: usize) -> Self {
        // [PASSO 1: Cálculo do Tamanho do Buffer Temporal]
        // O buffer precisa acomodar o campo receptivo e o padding de blocos.
        // Arredondamento para página é feito internamente pelo VirtualRingBuffer.
        let min_buffer_frames =
            receptive_field_size + (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES;

        let buffer = VirtualRingBuffer::<f32>::new(min_buffer_frames * channels);
        let buffer_bf16 = VirtualRingBuffer::<u16>::new(min_buffer_frames * channels);

        let actual_buffer_frames = buffer.size() / channels;

        // [PASSO 2: Offset Inicial (Jittering Alocado)]
        // Posicionamos o ponteiro inicial na segunda metade do mapeamento virtual (offset N).
        // Isso permite olhar para trás (receptive field) sem cruzar o início do buffer virtual.
        let jitter = (alloc_num % LAYER_ARRAY_BUFFER_PADDING) + 1;
        let start = actual_buffer_frames * 2 - (WAVENET_MAX_NUM_FRAMES * jitter);

        Self {
            layer_buffer: buffer,
            layer_buffer_bf16: buffer_bf16,
            buffer_start: start,
            receptive_field_size,
        }
    }

    /// Executa um passo do ponteiro do Ring Buffer. Se chegar na margem, volta para o início.
    pub fn advance_frames(&mut self, num_frames: usize, channels: usize) {
        self.buffer_start += num_frames;
        let buffer_frames = self.layer_buffer.size() / channels;

        // [VIRTUAL RING BUFFER]
        // Se o próximo bloco de tamanho máximo (64) puder ultrapassar o limite do mapeamento 2N,
        // retrocedemos o ponteiro para a primeira metade (mantendo a paridade de endereço virtual).
        // Isso garante que [buffer_start .. buffer_start + 64] seja sempre um acesso seguro.
        if self.buffer_start + WAVENET_MAX_NUM_FRAMES > buffer_frames * 2 {
            self.buffer_start -= buffer_frames;
        }
    }
}
