// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Filtro de suavização (smoothing) para parâmetros de áudio.
//!
//! Implementa um filtro IIR de 1-pólo (Low-pass) para evitar cliques e ruídos
//! de zipper ao alterar ganhos durante o processamento em tempo real.

/// Suavizador de parâmetros baseado em filtro IIR de 1-pólo.
/// y[n] = α * target + (1 - α) * y[n-1]
#[derive(Debug, Clone, Copy)]
pub struct ParamSmoother {
    current: f32,
    target: f32,
    alpha: f32,
}

impl ParamSmoother {
    /// Cria um novo suavizador com valor inicial e coeficiente alfa.
    ///
    /// # Parâmetros
    /// * `initial_value`: Valor inicial (e alvo inicial).
    /// * `sample_rate`: Taxa de amostragem (fs).
    /// * `cutoff_hz`: Frequência de corte (fc). Recomendado ~20Hz para ganhos.
    pub fn new(initial_value: f32, sample_rate: f32, cutoff_hz: f32) -> Self {
        let alpha = if sample_rate > 0.0 {
            // α = 1 - exp(-2π * fc / fs)
            1.0 - (-(2.0 * std::f32::consts::PI * cutoff_hz) / sample_rate).exp()
        } else {
            1.0
        };

        Self {
            current: initial_value,
            target: initial_value,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Atualiza o valor alvo do parâmetro.
    #[inline]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Salta imediatamente para o valor alvo (sem suavização).
    #[inline]
    pub fn snap_to_target(&mut self) {
        self.current = self.target;
    }

    /// Avança um sample e retorna o valor suavizado.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        if (self.current - self.target).abs() < 1e-6 {
            self.current = self.target;
        } else {
            self.current = self.alpha * self.target + (1.0 - self.alpha) * self.current;
        }
        self.current
    }

    /// Retorna o valor atual (último calculado).
    #[inline]
    pub fn current_value(&self) -> f32 {
        self.current
    }

    /// Retorna o valor alvo.
    #[inline]
    pub fn target_value(&self) -> f32 {
        self.target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoother_convergence() {
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
        smoother.set_target(1.0);

        // Deve convergir gradualmente
        let mut last_val = 0.0;
        for _ in 0..1000 {
            let current = smoother.tick();
            assert!(current >= last_val);
            last_val = current;
        }

        assert!(last_val > 0.5); // Em 1000 samples @ 48k com 20Hz (~20ms), já deve estar bem avançado
    }

    #[test]
    fn test_smoother_snap() {
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
        smoother.set_target(1.0);
        smoother.snap_to_target();
        assert_eq!(smoother.tick(), 1.0);
    }
}
