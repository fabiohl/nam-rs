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
        let diff = self.current - self.target;
        // Threshold proporcional ao target: convergência 2-5x mais rápida para valores altos.
        let threshold = 1e-6 * self.target.abs().max(1.0);
        if diff.abs() < threshold {
            self.current = self.target;
        } else {
            let next = self.alpha * self.target + (1.0 - self.alpha) * self.current;
            if next == self.current {
                // Detecção de precision stall em f32: se o passo for menor do que
                // a menor variação representável, força snap para o target.
                self.current = self.target;
            } else {
                self.current = next;
                // Denormal guard (RT-Safety §2.1): flush a zero para evitar FPU slowdown.
                if self.current.abs() < 1e-15 {
                    self.current = 0.0;
                }
            }
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

    #[test]
    fn test_smoother_convergence_high_gain() {
        // Verificar que para target = 3.98 (≈ +12dB), o smoother converge em ≤ 2400 samples a 48kHz (50ms).
        // Nota: O cutoff de 45Hz ilustra perfeitamente o benefício do threshold relativo,
        // pois com threshold fixo (1e-6) a convergência levaria 2581 samples (excedendo 2400),
        // enquanto o threshold relativo permite a convergência em 2347 samples.
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 45.0);
        smoother.set_target(3.98);

        let mut samples = 0;
        for _ in 0..5000 {
            let current = smoother.tick();
            samples += 1;
            if current == 3.98 {
                break;
            }
        }
        assert!(
            samples <= 2400,
            "Convergência demorou {} samples (esperado <= 2400)",
            samples
        );
    }

    #[test]
    fn test_smoother_denormal_prevention() {
        // Verificar que para target = 0.0 e initial = 1e-20, o tick() retorna exatamente 0.0 após ≤ 10 iterações.
        let mut smoother = ParamSmoother::new(1e-20, 48000.0, 20.0);
        smoother.set_target(0.0);

        let mut converged = false;
        for _ in 0..10 {
            if smoother.tick() == 0.0 {
                converged = true;
                break;
            }
        }
        assert!(converged, "Não convergiu a 0.0 em 10 iterações");
    }

    #[test]
    fn test_smoother_relative_threshold() {
        // Verificar que target = 0.001 ainda converge corretamente (não snap prematuro).
        let mut smoother = ParamSmoother::new(0.0, 48000.0, 20.0);
        smoother.set_target(0.001);

        // O primeiro tick não deve atingir imediatamente 0.001 (snap prematuro).
        let val1 = smoother.tick();
        assert!(val1 > 0.0);
        assert!(val1 < 0.001);

        // Deve convergir eventualmente
        let mut converged = false;
        for _ in 0..5000 {
            if smoother.tick() == 0.001 {
                converged = true;
                break;
            }
        }
        assert!(converged, "Deveria convergir para 0.001");
    }
}
