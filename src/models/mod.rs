// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de Motores Cerebrais (Arquiteturas Inferenciais Neurais) para o NAM-rs.
//!
//! Este módulo contém os cérebros acústicos do programa: redes neurais que aprenderam como,
//! por exemplo, um amplificador ou pedal verdadeiro distorce e colora o som de uma guitarra.
//! Aqui temos implementações super rápidas ("Zero-Allocation") baseadas no
//! comportamento exato treinado pelo Neural Amp Modeler (NAM) original.

// =============================================================================
// Sub-módulos
// =============================================================================

pub mod lstm;
pub mod lstm_dyn;
pub mod wavenet;
pub mod wavenet_dyn;

// =============================================================================
// Type Aliases — Perfis LSTM NAM Comuns
// =============================================================================

/// LSTM 1 camada × 8 unidades ocultas (Nano/Feather).
pub type Lstm1x8 = lstm::LstmModel1<8, 9, 32>;
/// LSTM 1 camada × 12 unidades ocultas (Lite).
pub type Lstm1x12 = lstm::LstmModel1<12, 13, 48>;
/// LSTM 1 camada × 16 unidades ocultas (Standard).
pub type Lstm1x16 = lstm::LstmModel1<16, 17, 64>;
/// LSTM 1 camada × 24 unidades ocultas (Heavy Standard).
pub type Lstm1x24 = lstm::LstmModel1<24, 25, 96>;

/// LSTM 2 camadas × 8 unidades ocultas.
pub type Lstm2x8 = lstm::LstmModel2<8, 9, 16, 32>;
/// LSTM 2 camadas × 12 unidades ocultas.
pub type Lstm2x12 = lstm::LstmModel2<12, 13, 24, 48>;
/// LSTM 2 camadas × 16 unidades ocultas.
pub type Lstm2x16 = lstm::LstmModel2<16, 17, 32, 64>;

// =============================================================================
// Trait NamModel — Contrato Público
// =============================================================================

/// A interface (o conector padrão) para qualquer modelo neural (amplificadores, pedais, etc.).
///
/// Pense nisso como um cabo de áudio padrão: não importa a marca do pedal interno (WaveNet ou LSTM),
/// o sistema de áudio (host principal) sabe enviar som para ele e receber de volta através deste contrato/trait.
pub trait NamModel: Send + Sync {
    /// Invocado pelo DSP RT-Thread para processar blocos de amostragem acústica (Float32).
    /// O áudio deverá ser processado *in-place* ou lido de input para output dependendo da geometria vetorial.
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// "Aquece" as válvulas virtuais do motor neural (`prewarm`).
    /// Injeta um curto zumbido ou silêncio (`num_samples`) por dentro do amplificador antes de expor
    /// uma guitarra de verdade a ele. Isso acalma o "campo magnético virtual", garantindo que não
    /// hajam estalos (pops) estridentes quando você liga ou troca de modelo de súbito.
    fn prewarm(&mut self, num_samples: usize);
}

/// Wrapper para preservar a vtable do trait object. Transita como thin pointer seguro.
pub struct DynamicModel(pub Box<dyn NamModel>);

// =============================================================================
// NamModel para WaveNet (Const Generics)
// =============================================================================

impl<const CH: usize, const K: usize, const HEAD: usize> NamModel
    for wavenet::WaveNetModel<CH, K, HEAD>
{
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Delega ao método inherent WaveNetModel::process (métodos inherent têm prioridade)
        self.process(input, output);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        // WaveNet prewarm é one-shot: preenche o campo receptivo via copy_buffer.
        // O C++ executa `model->Prewarm()` sem parâmetro (diferente do LSTM).
        self.prewarm();
    }
}

// =============================================================================
// NamModel para WaveNet Dinâmico
// =============================================================================

impl NamModel for wavenet_dyn::WaveNetDynModel {
    /// Delega o processamento para a implementação interna do modelo WaveNet dinâmico.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// O "aquecimento" da WaveNet é simplificado pois ela não possui memória infinita
    /// como a LSTM, apenas um buffer de delay (campo receptivo).
    fn prewarm(&mut self, _num_samples: usize) {
        self.prewarm();
    }
}

// =============================================================================
// NamModel para LSTM — 1 Camada
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H_H4: usize> NamModel
    for lstm::LstmModel1<H, H1_IH, H_H4>
{
    /// Executa o processamento de áudio da LSTM.
    /// Note que `self.process` chama o método inerente da struct, que já possui
    /// a lógica de despacho SIMD (AVX2/512) otimizada.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Safety: A verificação de compatibilidade de hardware é feita no início da aplicação.
        self.process(input, output);
    }

    /// O prewarm na LSTM é vital. Como é um modelo recorrente, o estado interno (memória)
    /// precisa de um tempo processando silêncio para "estabilizar" antes do áudio real.
    fn prewarm(&mut self, num_samples: usize) {
        // 1. Limpa qualquer resíduo de processamentos anteriores.
        self.reset_states();

        // 2. Processa amostras de valor zero.
        // Fazemos isso em pedaços (CHUNK) para reaproveitar buffers pequenos na stack.
        const CHUNK: usize = 512;
        let zero_in = [0.0f32; CHUNK];
        let mut zero_out = [0.0f32; CHUNK];
        let mut rem = num_samples;

        while rem > 0 {
            let n = rem.min(CHUNK);
            // Simula o processamento de silêncio para carregar a memória da LSTM.
            self.process(&zero_in[..n], &mut zero_out[..n]);
            rem -= n;
        }
    }
}

// =============================================================================
// NamModel para LSTM — 2 Camadas
// =============================================================================

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> NamModel
    for lstm::LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Processamento idêntico ao modelo de 1 camada, mas operando sobre a cadeia de 2 camadas.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// Prewarm para o modelo empilhado. Ambas as camadas são estabilizadas sequencialmente.
    fn prewarm(&mut self, num_samples: usize) {
        // Zera os estados internos de ambas as camadas.
        self.reset_states();

        const CHUNK: usize = 512;
        let zero_in = [0.0f32; CHUNK];
        let mut zero_out = [0.0f32; CHUNK];
        let mut rem = num_samples;

        while rem > 0 {
            let n = rem.min(CHUNK);
            self.process(&zero_in[..n], &mut zero_out[..n]);
            rem -= n;
        }
    }
}

// =============================================================================
// NamModel para LSTM Dinâmico
// =============================================================================

impl NamModel for lstm_dyn::LstmDynModel {
    /// Implementação para modelos onde o tamanho do hidden state é definido em tempo de execução.
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        self.process(input, output);
    }

    /// O prewarm dinâmico já encapsula internamente a lógica de loop de silêncio.
    fn prewarm(&mut self, num_samples: usize) {
        self.prewarm(num_samples);
    }
}
