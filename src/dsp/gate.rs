// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Lógica de Gate com Histerese Dinâmica para Otimizações de DSP.
//!
//! Este módulo implementa uma Máquina de Estados Finitos (FSM) para detectar
//! silêncio ou sinal mono com histerese temporal e de amplitude (Schmitt Trigger).
//! O objetivo é evitar "chattering" (oscilação rápida de estado) e artefatos
//! audíveis (clicks/zipper noise) ao alternar entre modos de processamento.

use crate::math::common::SimdMath;

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
    /// Pré-computado: `1.0 / fade_frames as f32` para evitar divisão no hotpath.
    pub inv_fade_frames: f32,
    /// Tolerância absoluta entre canais L/R para detecção de sinal Mono.
    pub mono_epsilon: f32,
}

impl GateParams {
    /// Cria novos parâmetros de gate computando `inv_fade_frames = 1.0 / fade_frames`.
    pub fn new(
        threshold_open_db: f32,
        threshold_close_db: f32,
        hold_frames: usize,
        fade_frames: usize,
        mono_epsilon: f32,
    ) -> Self {
        let div = fade_frames.max(1) as f32;
        Self {
            threshold_open_db,
            threshold_close_db,
            hold_frames,
            fade_frames,
            inv_fade_frames: 1.0 / div,
            mono_epsilon,
        }
    }
}

impl Default for GateParams {
    fn default() -> Self {
        Self::new(-70.0, -80.0, 2048, 256, 1e-4)
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
    #[cold]
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

    /// Decide se o portão de ruído deve abrir, fechar ou continuar como está,
    /// baseando-se no volume do áudio atual.
    ///
    /// # Parâmetros
    /// - `value`: O volume atual detectado.
    /// - `threshold_open`: O volume necessário para "abrir" o portão.
    /// - `threshold_close`: O volume abaixo do qual o portão deve começar a "fechar".
    /// - `params`: Configurações de tempo (quanto tempo esperar e quão lento fechar).
    /// - `n_samples`: Quantas amostras de som estamos processando agora.
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
            GateState::Open => self.update_open(value, threshold_close, params, n_samples),
            GateState::FadingOut => {
                self.update_fading_out(value, threshold_open, params, n_samples)
            }
            GateState::Closed => self.update_closed(value, threshold_open, params, n_samples),
            GateState::FadingIn => self.update_fading_in(value, threshold_close, params, n_samples),
        }
    }

    fn update_open(
        &mut self,
        value: f32,
        threshold_close: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
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

    fn update_fading_out(
        &mut self,
        value: f32,
        threshold_open: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        if value >= threshold_open {
            self.state = GateState::FadingIn;
            self.fade_counter = params.fade_frames.saturating_sub(self.fade_counter);
            if self.fade_counter + n_samples < params.fade_frames {
                self.fade_counter += n_samples;
                self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
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
            self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
            self.ramp_samples = n_samples;
        } else {
            self.ramp_samples = self.fade_counter;
            self.state = GateState::Closed;
            self.current_multiplier = 0.0;
            self.fade_counter = 0;
        }
    }

    fn update_closed(
        &mut self,
        value: f32,
        threshold_open: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        if value >= threshold_open {
            self.state = GateState::FadingIn;
            self.fade_counter = 0;
            self.hold_counter = 0;
            if n_samples < params.fade_frames {
                self.fade_counter += n_samples;
                self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
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

    fn update_fading_in(
        &mut self,
        value: f32,
        threshold_close: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        if value < threshold_close {
            self.state = GateState::FadingOut;
            self.fade_counter = params.fade_frames.saturating_sub(self.fade_counter);
            if self.fade_counter > n_samples {
                self.fade_counter -= n_samples;
                self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
                self.ramp_samples = n_samples;
            } else {
                self.ramp_samples = self.fade_counter;
                self.state = GateState::Closed;
                self.current_multiplier = 0.0;
                self.fade_counter = 0;
            }
        } else if self.fade_counter + n_samples < params.fade_frames {
            self.fade_counter += n_samples;
            self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
            self.ramp_samples = n_samples;
        } else {
            self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
            self.state = GateState::Open;
            self.current_multiplier = 1.0;
            self.fade_counter = params.fade_frames;
            self.hold_counter = 0;
        }
    }

    /// Aplica o volume (ganho) atual ao som.
    /// Se o portão estiver abrindo ou fechando, ele faz uma mudança suave (rampa).
    /// Se estiver totalmente aberto ou fechado, ele aplica o volume constante.
    pub fn apply_gain_rt(&self, buffer: &mut [f32], n_samples: usize) {
        if self.ramp_samples == 0 {
            // O volume está estável (não está no meio de um fade).
            if self.current_multiplier == 0.0 {
                // Silêncio total.
                buffer.fill(0.0);
            } else if (self.current_multiplier - 1.0).abs() > 1e-6 {
                // Aplica um volume constante (ex: 50%).
                crate::math::dsp::gain::apply_gain_simd(buffer, self.current_multiplier);
            }
            // Se o volume for 1.0 (100%), não precisamos fazer nada.
            return;
        }

        let start_mult = self.ramp_start_multiplier;
        let end_mult = self.current_multiplier;

        if self.ramp_samples >= n_samples {
            // A mudança suave de volume vai durar o bloco inteiro.
            if (start_mult - end_mult).abs() < 1e-6 {
                crate::math::dsp::gain::apply_gain_simd(buffer, end_mult);
            } else {
                // Calcula o "degrau" de volume para cada amostra de som.
                // NOTA: Se n_samples = 1, step = (end - start) / 1.0.
                // O valor resultante será aplicado à única amostra, o que é o comportamento
                // esperado para mudanças instantâneas (sample-accurate) no CLAP.
                let step = (end_mult - start_mult) / (n_samples as f32);
                crate::math::dsp::gain::apply_ramp_simd(buffer, start_mult, step);
            }
        } else {
            // Caso especial: a mudança de volume termina antes do fim do bloco.
            // Dividimos o bloco em duas partes: a rampa e o volume constante final.
            let (ramp_part, const_part) = buffer.split_at_mut(self.ramp_samples);

            if (start_mult - end_mult).abs() < 1e-6 {
                crate::math::dsp::gain::apply_gain_simd(ramp_part, end_mult);
            } else {
                let step = (end_mult - start_mult) / (self.ramp_samples as f32);
                crate::math::dsp::gain::apply_ramp_simd(ramp_part, start_mult, step);
            }

            // Preenche o restante do bloco com o volume final estabilizado.
            if end_mult == 0.0 {
                const_part.fill(0.0);
            } else if (end_mult - 1.0).abs() > 1e-6 {
                crate::math::dsp::gain::apply_gain_simd(const_part, end_mult);
            }
        }
    }

    /// Faz o mesmo que a função acima, mas para som estéreo (canal esquerdo e direito).
    /// O processamento é feito em conjunto para ser mais rápido.
    pub fn apply_gain_rt_stereo<M: SimdMath>(
        &self,
        left: &mut [f32],
        right: &mut [f32],
        n_samples: usize,
    ) {
        if self.ramp_samples == 0 {
            // Volume estável para ambos os canais.
            if self.current_multiplier == 0.0 {
                left.fill(0.0);
                right.fill(0.0);
            } else if (self.current_multiplier - 1.0).abs() > 1e-6 {
                // Aplica o volume nos dois canais de forma eficiente.
                unsafe { M::apply_gain_stereo(left, right, self.current_multiplier) };
            }
            return;
        }

        let start_mult = self.ramp_start_multiplier;
        let end_mult = self.current_multiplier;

        if self.ramp_samples >= n_samples {
            // Mudança suave nos dois canais durante todo o bloco.
            if (start_mult - end_mult).abs() < 1e-6 {
                unsafe { M::apply_gain_stereo(left, right, end_mult) };
            } else {
                // NOTA: Com n_samples = 1, o "ramp" de 1 sample
                // resulta em um salto direto para o valor alvo, o que é aceito por design.
                let step = (end_mult - start_mult) / (n_samples as f32);
                unsafe { M::apply_ramp_stereo(left, right, start_mult, step) };
            }
        } else {
            // Mudança suave termina antes do fim do bloco para ambos os canais.
            let (ramp_l, const_l) = left.split_at_mut(self.ramp_samples);
            let (ramp_r, const_r) = right.split_at_mut(self.ramp_samples);

            if (start_mult - end_mult).abs() < 1e-6 {
                unsafe { M::apply_gain_stereo(ramp_l, ramp_r, end_mult) };
            } else {
                let step = (end_mult - start_mult) / (self.ramp_samples as f32);
                unsafe { M::apply_ramp_stereo(ramp_l, ramp_r, start_mult, step) };
            }

            // Finaliza o restante do bloco com o volume estável.
            if end_mult == 0.0 {
                const_l.fill(0.0);
                const_r.fill(0.0);
            } else if (end_mult - 1.0).abs() > 1e-6 {
                unsafe { M::apply_gain_stereo(const_l, const_r, end_mult) };
            }
        }
    }
}

#[cfg(test)]
#[path = "gate_test.rs"]
mod gate_test;
