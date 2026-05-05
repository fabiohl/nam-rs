// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Lógica de Gate com Histerese Dinâmica para Otimizações de DSP.
//!
//! Este módulo implementa uma Máquina de Estados Finitos (FSM) para detectar
//! silêncio ou sinal mono com histerese temporal e de amplitude (Schmitt Trigger).
//! O objetivo é evitar "chattering" (oscilação rápida de estado) e artefatos
//! audíveis (clicks/zipper noise) ao alternar entre modos de processamento.

/// Parâmetros de configuração para a lógica de Gate e Histerese.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(align(128))] // Evita false sharing no canal SPSC
pub struct GateParams {
    /// Limiar para abrir o gate (transição de Silêncio para Sinal), em dB.
    pub threshold_open_db: f32,
    /// Limiar para fechar o gate (transição de Sinal para Silêncio), em dB.
    pub threshold_close_db: f32,
    /// Número de frames (amostras) a aguardar em silêncio antes de fechar o gate.
    pub hold_frames: usize,
    /// Número de frames (amostras) para realizar o fade-in/fade-out (suavização).
    pub fade_frames: usize,
    /// Tolerância absoluta entre canais L/R para detecção de sinal Mono.
    pub mono_epsilon: f32,
}

impl Default for GateParams {
    fn default() -> Self {
        Self {
            threshold_open_db: -70.0,
            threshold_close_db: -80.0,
            hold_frames: 2048, // ~42ms @ 48kHz
            fade_frames: 256,  // ~5ms @ 48kHz
            mono_epsilon: 1e-4,
        }
    }
}

/// Representa os estados possíveis do gate com histerese.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateState {
    /// Gate aberto: processamento normal.
    Open,
    /// Iniciando transição para fechado (fade-out).
    FadingOut,
    /// Gate fechado: bypass total/silêncio.
    Closed,
    /// Iniciando transição para aberto (fade-in).
    FadingIn,
}

/// Implementação da FSM de Histerese Dinâmica (Schmitt Trigger).
pub struct DynamicHysteresis {
    state: GateState,
    hold_counter: usize,
    fade_counter: usize,
    current_multiplier: f32,
}

impl Default for DynamicHysteresis {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicHysteresis {
    /// Cria uma nova instância no estado inicial Aberto.
    pub fn new() -> Self {
        Self {
            state: GateState::Open,
            hold_counter: 0,
            fade_counter: 0,
            current_multiplier: 1.0,
        }
    }

    /// Retorna o estado atual da FSM.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Retorna o multiplicador de ganho atual (0.0 a 1.0).
    pub fn multiplier(&self) -> f32 {
        self.current_multiplier
    }

    /// Atualiza o estado da FSM baseado em um valor de controle (energia ou diferença).
    ///
    /// # Parâmetros
    /// - `value`: Valor de controle (ex: RMS ou MaxDiff linear).
    /// - `threshold_open`: Limiar linear para abrir.
    /// - `threshold_close`: Limiar linear para fechar.
    /// - `params`: Configurações de tempo (hold/fade).
    /// - `n_samples`: Número de amostras no bloco atual.
    pub fn update(
        &mut self,
        value: f32,
        threshold_open: f32,
        threshold_close: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        match self.state {
            GateState::Open => {
                if value < threshold_close {
                    self.hold_counter += n_samples;
                    if self.hold_counter >= params.hold_frames {
                        self.state = GateState::FadingOut;
                        self.fade_counter = params.fade_frames;
                    }
                } else {
                    self.hold_counter = 0;
                }
            }
            GateState::FadingOut => {
                if value >= threshold_open {
                    self.state = GateState::FadingIn;
                    self.fade_counter = params.fade_frames.saturating_sub(self.fade_counter);
                } else if self.fade_counter > n_samples {
                    self.fade_counter -= n_samples;
                    self.current_multiplier = self.fade_counter as f32 / params.fade_frames as f32;
                } else {
                    self.state = GateState::Closed;
                    self.current_multiplier = 0.0;
                    self.fade_counter = 0;
                }
            }
            GateState::Closed => {
                if value >= threshold_open {
                    self.state = GateState::FadingIn;
                    self.fade_counter = 0;
                    self.hold_counter = 0;
                }
            }
            GateState::FadingIn => {
                if value < threshold_close {
                    self.state = GateState::FadingOut;
                    self.fade_counter = params.fade_frames.saturating_sub(self.fade_counter);
                } else if self.fade_counter + n_samples < params.fade_frames {
                    self.fade_counter += n_samples;
                    self.current_multiplier = self.fade_counter as f32 / params.fade_frames as f32;
                } else {
                    self.state = GateState::Open;
                    self.current_multiplier = 1.0;
                    self.fade_counter = params.fade_frames;
                    self.hold_counter = 0;
                }
            }
        }
    }

    /// Aplica o multiplicador de ganho atual ao buffer.
    /// Se o estado for FadingIn ou FadingOut, aplica uma rampa linear.
    ///
    /// # Safety
    /// Esta função deve ser chamada apenas na thread RT.
    pub fn apply_gain_rt(&self, buffer: &mut [f32], params: &GateParams, n_samples: usize) {
        if self.state == GateState::Closed {
            buffer.fill(0.0);
            return;
        }
        if self.state == GateState::Open {
            // Se o multiplicador for 1.0, não faz nada (bypass).
            if (self.current_multiplier - 1.0).abs() > 1e-6 {
                crate::dsp::gain::apply_gain_simd(buffer, self.current_multiplier);
            }
            return;
        }

        // Caso FadingIn ou FadingOut: aplica rampa linear.
        let start_mult = match self.state {
            GateState::FadingIn => {
                (self.fade_counter.saturating_sub(n_samples) as f32) / (params.fade_frames as f32)
            }
            GateState::FadingOut => {
                ((self.fade_counter + n_samples) as f32 / params.fade_frames as f32).min(1.0)
            }
            _ => self.current_multiplier,
        };
        let end_mult = self.current_multiplier;

        if (start_mult - end_mult).abs() < 1e-6 {
            crate::dsp::gain::apply_gain_simd(buffer, end_mult);
        } else {
            let step = (end_mult - start_mult) / (n_samples as f32);
            crate::dsp::gain::apply_ramp_simd(buffer, start_mult, step);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_params_default() {
        let params = GateParams::default();
        assert_eq!(params.threshold_open_db, -70.0);
        assert_eq!(params.threshold_close_db, -80.0);
        assert_eq!(params.hold_frames, 2048);
        assert_eq!(params.fade_frames, 256);
    }

    #[test]
    fn test_hysteresis_basic_transitions() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams {
            threshold_open_db: -10.0,
            threshold_close_db: -20.0,
            hold_frames: 10,
            fade_frames: 10,
            mono_epsilon: 1e-4,
        };
        // Usamos valores lineares diretos para o teste (como se fossem squared energy)
        let th_open = 1.0;
        let th_close = 0.5;

        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);

        // 1. Sinal cai abaixo do threshold de fechamento
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Open, "Deve permanecer Open durante o hold");
        
        // 2. Passa o tempo de hold
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::FadingOut, "Deve entrar em FadingOut após hold_frames");
        
        // 3. Meio do fade out
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.5); // 5/10

        // 4. Conclui fade out
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Closed);
        assert_eq!(dh.multiplier(), 0.0);

        // 5. Sinal volta (acima do open)
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.0, "Transição inicial de Closed para FadingIn mantém mult=0");

        // 6. Progresso do fade in
        dh.update(2.0, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.5); // 5/10

        // 7. Conclui fade in
        dh.update(2.0, th_open, th_close, &params, 5);
        assert_eq!(dh.state(), GateState::Open);
        assert_eq!(dh.multiplier(), 1.0);
    }

    #[test]
    fn test_hysteresis_interrupted_fade() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams {
            hold_frames: 10,
            fade_frames: 10,
            ..Default::default()
        };
        let th_open = 1.0;
        let th_close = 0.5;

        // Força para FadingOut
        dh.update(0.1, th_open, th_close, &params, 11);
        assert_eq!(dh.state(), GateState::FadingOut);
        
        // Avança fade out até a metade
        dh.update(0.1, th_open, th_close, &params, 5);
        assert_eq!(dh.multiplier(), 0.5);

        // Interrompe com sinal alto -> Deve entrar em FadingIn a partir de onde parou
        dh.update(2.0, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingIn);
        assert_eq!(dh.multiplier(), 0.5);

        // Avança um pouco o fade in
        dh.update(2.0, th_open, th_close, &params, 2);
        assert_eq!(dh.multiplier(), 0.7);

        // Interrompe novamente com silêncio
        dh.update(0.1, th_open, th_close, &params, 1);
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 0.7);
    }

    #[test]
    fn test_hysteresis_apply_gain_ramp() {
        let mut dh = DynamicHysteresis::new();
        let params = GateParams {
            fade_frames: 100,
            ..Default::default()
        };
        let mut buffer = [1.0f32; 10];

        // Caso FadingOut: deve aplicar rampa descendente
        dh.update(0.0, 1.0, 0.5, &params, 2048 + 10); // Passa hold e inicia fade
        // fade_counter era 100, agora deve ser 90 (decrementado na segunda chamada de update se n=10)
        // Mas o update acima foi n=2048+10. 
        // Vamos fazer passo a passo.
        
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2047); // Quase no limite do hold
        assert_eq!(dh.state(), GateState::Open);
        
        dh.update(0.0, 1.0, 0.5, &params, 10); // Passa o hold e inicia FadingOut
        assert_eq!(dh.state(), GateState::FadingOut);
        assert_eq!(dh.multiplier(), 1.0, "No bloco de transição o multiplier ainda é 1.0");
        
        dh.update(0.0, 1.0, 0.5, &params, 10); // Primeiro bloco real de fade
        assert_eq!(dh.multiplier(), 0.9); // 90/100
        
        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, &params, 10);
        // Deve ser rampa de 1.0 a 0.9. O primeiro sample (index 0) recebe o ganho 'start' (1.0).
        assert!((buffer[0] - 1.0).abs() < 1e-3);
        assert!((buffer[9] - 0.91).abs() < 1e-3);

        // Caso FadingIn: deve aplicar rampa ascendente
        let mut dh = DynamicHysteresis::new();
        dh.update(0.0, 1.0, 0.5, &params, 2048); // Passa hold -> FadingOut
        dh.update(0.0, 1.0, 0.5, &params, 101);  // Passa fade -> Closed
        assert_eq!(dh.state(), GateState::Closed);
        
        dh.update(2.0, 1.0, 0.5, &params, 1); // Transição para FadingIn, counter=0
        dh.update(2.0, 1.0, 0.5, &params, 10); // Avança fade in, counter=10, mult=0.1
        assert_eq!(dh.multiplier(), 0.1);
        
        buffer.fill(1.0);
        dh.apply_gain_rt(&mut buffer, &params, 10);
        // Deve ser rampa de 0.0 a 0.1. O primeiro sample (index 0) recebe o ganho 'start' (0.0).
        assert!((buffer[0] - 0.0).abs() < 1e-3);
        assert!((buffer[9] - 0.09).abs() < 1e-3);
    }
}
