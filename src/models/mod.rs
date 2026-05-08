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

pub mod activations;
pub mod film;
pub mod gating;
pub mod lstm;
pub mod lstm_dyn;
pub mod wavenet;
pub mod wavenet_common;
pub mod wavenet_dyn;
pub mod wavenet_params;

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

/// Wrapper enum para as variantes de modelo treinadas.
/// Permite despacho estático das chamadas DSP para a variante concreta evitando overhead de vtable.
pub enum DynamicModel {
    /// WaveNet Standard (16 canais, kernel 3, dilation 8).
    WavenetStandard(Box<wavenet::WaveNetModel<16, 3, 8>>),
    /// WaveNet Lite (12 canais, kernel 3, dilation 6).
    WavenetLite(Box<wavenet::WaveNetModel<12, 3, 6>>),
    /// WaveNet Feather (8 canais, kernel 3, dilation 4).
    WavenetFeather(Box<wavenet::WaveNetModel<8, 3, 4>>),
    /// WaveNet Nano (4 canais, kernel 3, dilation 2).
    WavenetNano(Box<wavenet::WaveNetModel<4, 3, 2>>),
    /// WaveNet Dinâmico (usado como fallback para arquiteturas não-padrão).
    WavenetDyn(Box<wavenet_dyn::WaveNetDynModel>),
    /// WaveNet A2 (Placeholder para arquitetura nova).
    WavenetA2(Box<WavenetA2Placeholder>),
    /// LSTM 1 Camada × 8 unidades ocultas.
    Lstm1x8(Box<Lstm1x8>),
    /// LSTM 1 Camada × 12 unidades ocultas.
    Lstm1x12(Box<Lstm1x12>),
    /// LSTM 1 Camada × 16 unidades ocultas.
    Lstm1x16(Box<Lstm1x16>),
    /// LSTM 1 Camada × 24 unidades ocultas.
    Lstm1x24(Box<Lstm1x24>),
    /// LSTM 2 Camadas × 8 unidades ocultas.
    Lstm2x8(Box<Lstm2x8>),
    /// LSTM 2 Camadas × 12 unidades ocultas.
    Lstm2x12(Box<Lstm2x12>),
    /// LSTM 2 Camadas × 16 unidades ocultas.
    Lstm2x16(Box<Lstm2x16>),
    /// LSTM Dinâmico (usado como fallback).
    LstmDyn(Box<lstm_dyn::LstmDynModel>),
}

impl NamModel for DynamicModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        match self {
            Self::WavenetStandard(m) => m.process(input, output),
            Self::WavenetLite(m) => m.process(input, output),
            Self::WavenetFeather(m) => m.process(input, output),
            Self::WavenetNano(m) => m.process(input, output),
            Self::WavenetDyn(m) => m.process(input, output),
            Self::WavenetA2(m) => m.process(input, output),
            Self::Lstm1x8(m) => m.process(input, output),
            Self::Lstm1x12(m) => m.process(input, output),
            Self::Lstm1x16(m) => m.process(input, output),
            Self::Lstm1x24(m) => m.process(input, output),
            Self::Lstm2x8(m) => m.process(input, output),
            Self::Lstm2x12(m) => m.process(input, output),
            Self::Lstm2x16(m) => m.process(input, output),
            Self::LstmDyn(m) => m.process(input, output),
        }
    }

    fn prewarm(&mut self, num_samples: usize) {
        match self {
            Self::WavenetStandard(m) => m.prewarm(),
            Self::WavenetLite(m) => m.prewarm(),
            Self::WavenetFeather(m) => m.prewarm(),
            Self::WavenetNano(m) => m.prewarm(),
            Self::WavenetDyn(m) => m.prewarm(),
            Self::WavenetA2(m) => m.prewarm(num_samples),
            Self::Lstm1x8(m) => m.prewarm(num_samples),
            Self::Lstm1x12(m) => m.prewarm(num_samples),
            Self::Lstm1x16(m) => m.prewarm(num_samples),
            Self::Lstm1x24(m) => m.prewarm(num_samples),
            Self::Lstm2x8(m) => m.prewarm(num_samples),
            Self::Lstm2x12(m) => m.prewarm(num_samples),
            Self::Lstm2x16(m) => m.prewarm(num_samples),
            Self::LstmDyn(m) => m.prewarm(num_samples),
        }
    }
}

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
        lstm_prewarm_common(self, num_samples);
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
        lstm_prewarm_common(self, num_samples);
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

// =============================================================================
// Helpers Internos — Redução de Boilerplate
// =============================================================================

/// Trait interno para unificar modelos que possuem estado LSTM resetável.
trait LstmLike: NamModel {
    fn reset_states(&mut self);
}

impl<const H: usize, const H1_IH: usize, const H_H4: usize> LstmLike
    for lstm::LstmModel1<H, H1_IH, H_H4>
{
    fn reset_states(&mut self) {
        self.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> LstmLike
    for lstm::LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn reset_states(&mut self) {
        self.reset_states();
    }
}

/// Implementação genérica de aquecimento (prewarm) para modelos baseados em LSTM.
/// Injeta silêncio para estabilizar os estados internos antes do uso real.
fn lstm_prewarm_common(model: &mut impl LstmLike, num_samples: usize) {
    // 1. Limpa qualquer resíduo de processamentos anteriores.
    model.reset_states();

    // 2. Processa amostras de valor zero.
    const CHUNK: usize = 512;
    let zero_in = [0.0f32; CHUNK];
    let mut zero_out = [0.0f32; CHUNK];
    let mut rem = num_samples;

    while rem > 0 {
        let n = rem.min(CHUNK);
        model.process(&zero_in[..n], &mut zero_out[..n]);
        rem -= n;
    }
}

// =============================================================================
// Placeholder para WaveNet A2 (Staging)
// =============================================================================

/// Placeholder para a arquitetura WaveNet A2.
///
/// Este struct permite que o sistema carregue modelos A2 sem falhar, retornando
/// silêncio até que a implementação completa do motor de inferência esteja pronta.
#[derive(Default)]
pub struct WavenetA2Placeholder {
    /// Flag para emitir o aviso de log apenas uma vez por instância.
    warned: bool,
}

impl NamModel for WavenetA2Placeholder {
    fn process(&mut self, _input: &[f32], output: &mut [f32]) {
        if !self.warned {
            log::warn!(
                "Arquitetura WaveNet A2 detectada: Modo Placeholder (Silencioso) ativo. A implementação real está em desenvolvimento."
            );
            self.warned = true;
        }

        // Retorna silêncio absoluto.
        output.fill(0.0);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        // No-op para o placeholder.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenet_a2_placeholder_silence() {
        let mut model = WavenetA2Placeholder::default();
        let input = [1.0f32; 10];
        let mut output = [1.0f32; 10];
        model.process(&input, &mut output);
        for val in output.iter() {
            assert_eq!(*val, 0.0);
        }
    }
}
