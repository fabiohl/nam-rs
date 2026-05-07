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
    ramp_start_multiplier: f32,
    ramp_samples: usize,
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
            ramp_start_multiplier: 1.0,
            ramp_samples: 0,
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
        self.ramp_start_multiplier = self.current_multiplier;

        match self.state {
            GateState::Open => {
                if value < threshold_close {
                    self.hold_counter += n_samples;
                    if self.hold_counter >= params.hold_frames {
                        self.state = GateState::FadingOut;
                        self.fade_counter = params.fade_frames;
                        self.ramp_samples = 0;
                    } else {
                        self.ramp_samples = 0;
                    }
                } else {
                    self.hold_counter = 0;
                    self.ramp_samples = 0;
                }
            }
            GateState::FadingOut => {
                if value >= threshold_open {
                    self.state = GateState::FadingIn;
                    self.fade_counter = params.fade_frames.saturating_sub(self.fade_counter);

                    if self.fade_counter + n_samples < params.fade_frames {
                        self.fade_counter += n_samples;
                        self.current_multiplier =
                            self.fade_counter as f32 / params.fade_frames as f32;
                        self.ramp_samples = n_samples;
                    } else {
                        self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
                        self.state = GateState::Open;
                        self.current_multiplier = 1.0;
                        self.fade_counter = params.fade_frames;
                        self.hold_counter = 0;
                    }
                } else if self.fade_counter > n_samples {
                    self.fade_counter -= n_samples;
                    self.current_multiplier = self.fade_counter as f32 / params.fade_frames as f32;
                    self.ramp_samples = n_samples;
                } else {
                    self.ramp_samples = self.fade_counter;
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

                    if self.fade_counter + n_samples < params.fade_frames {
                        self.fade_counter += n_samples;
                        self.current_multiplier =
                            self.fade_counter as f32 / params.fade_frames as f32;
                        self.ramp_samples = n_samples;
                    } else {
                        self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
                        self.state = GateState::Open;
                        self.current_multiplier = 1.0;
                        self.fade_counter = params.fade_frames;
                    }
                } else {
                    self.ramp_samples = 0;
                }
            }
            GateState::FadingIn => {
                if value < threshold_close {
                    self.state = GateState::FadingOut;
                    self.fade_counter = params.fade_frames.saturating_sub(self.fade_counter);

                    if self.fade_counter > n_samples {
                        self.fade_counter -= n_samples;
                        self.current_multiplier =
                            self.fade_counter as f32 / params.fade_frames as f32;
                        self.ramp_samples = n_samples;
                    } else {
                        self.ramp_samples = self.fade_counter;
                        self.state = GateState::Closed;
                        self.current_multiplier = 0.0;
                        self.fade_counter = 0;
                    }
                } else if self.fade_counter + n_samples < params.fade_frames {
                    self.fade_counter += n_samples;
                    self.current_multiplier = self.fade_counter as f32 / params.fade_frames as f32;
                    self.ramp_samples = n_samples;
                } else {
                    self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
                    self.state = GateState::Open;
                    self.current_multiplier = 1.0;
                    self.fade_counter = params.fade_frames;
                    self.hold_counter = 0;
                }
            }
        }
    }

    /// Aplica o multiplicador de ganho atual ao buffer.
    /// Se o estado for FadingIn ou FadingOut, aplica uma rampa linear estritamente
    /// até o sample onde o fade encerra (Interpolação Sub-Bloco).
    ///
    /// # Safety
    /// Esta função deve ser chamada apenas na thread RT.
    pub fn apply_gain_rt(&self, buffer: &mut [f32], _params: &GateParams, n_samples: usize) {
        if self.ramp_samples == 0 {
            if self.current_multiplier == 0.0 {
                buffer.fill(0.0);
            } else if (self.current_multiplier - 1.0).abs() > 1e-6 {
                crate::dsp::gain::apply_gain_simd(buffer, self.current_multiplier);
            }
            return;
        }

        let start_mult = self.ramp_start_multiplier;
        let end_mult = self.current_multiplier;

        if self.ramp_samples >= n_samples {
            // A rampa ocupa o bloco inteiro
            if (start_mult - end_mult).abs() < 1e-6 {
                crate::dsp::gain::apply_gain_simd(buffer, end_mult);
            } else {
                let step = (end_mult - start_mult) / (n_samples as f32);
                crate::dsp::gain::apply_ramp_simd(buffer, start_mult, step);
            }
        } else {
            // Interpolação Sub-Bloco! A rampa termina antes do fim do bloco.
            let (ramp_part, const_part) = buffer.split_at_mut(self.ramp_samples);

            if (start_mult - end_mult).abs() < 1e-6 {
                crate::dsp::gain::apply_gain_simd(ramp_part, end_mult);
            } else {
                let step = (end_mult - start_mult) / (self.ramp_samples as f32);
                crate::dsp::gain::apply_ramp_simd(ramp_part, start_mult, step);
            }

            // Preenche o resto do buffer com a constante final
            if end_mult == 0.0 {
                const_part.fill(0.0);
            } else if (end_mult - 1.0).abs() > 1e-6 {
                crate::dsp::gain::apply_gain_simd(const_part, end_mult);
            }
        }
    }
}

#[cfg(test)]
#[path = "gate_test.rs"]
mod gate_test;
