// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de funções de ativação para a arquitetura NAM A2.
//!
//! Este módulo fornece implementações de várias funções de ativação não-lineares
//! utilizadas nos modelos Neural Amp Modeler, garantindo paridade com a
//! implementação C++ original.

/// Tipos de ativação suportados pelo NAM A2.
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    /// Tangente Hiperbólica (standard).
    Tanh,
    /// Tangente Hiperbólica com saturação rígida em [-1, 1].
    HardTanh,
    /// Aproximação racional rápida para Tanh.
    FastTanh,
    /// Rectified Linear Unit: max(0, x).
    ReLU,
    /// Leaky ReLU com inclinação negativa fixa.
    LeakyReLU {
        /// Inclinação aplicada quando x < 0.
        negative_slope: f32,
    },
    /// Parametric ReLU com inclinação negativa por canal ou global.
    PReLU {
        /// Vetor de inclinações para valores negativos.
        negative_slopes: Vec<f32>,
    },
    /// Função Sigmoide logística: 1 / (1 + exp(-x)).
    Sigmoid,
    /// Sigmoid Linear Unit: x * sigmoid(x).
    SiLU,
    /// Versão eficiente da função Swish.
    HardSwish,
    /// HardTanh com inclinações configuráveis para as regiões de saturação.
    LeakyHardTanh {
        /// Valor mínimo para saturação linear.
        min_val: f32,
        /// Valor máximo para saturação linear.
        max_val: f32,
        /// Inclinação aplicada abaixo de `min_val`.
        min_slope: f32,
        /// Inclinação aplicada acima de `max_val`.
        max_slope: f32,
    },
    /// Softsign: x / (1 + |x|).
    Softsign,
}

/// Trait para aplicação de funções de ativação em buffers de áudio.
pub trait ActivationFn {
    /// Aplica a função de ativação in-place no buffer fornecido.
    fn apply(&self, data: &mut [f32]);
}

impl ActivationFn for ActivationType {
    fn apply(&self, data: &mut [f32]) {
        match self {
            Self::Tanh => {
                for x in data.iter_mut() {
                    *x = x.tanh();
                }
            }
            Self::HardTanh => {
                for x in data.iter_mut() {
                    *x = x.clamp(-1.0, 1.0);
                }
            }
            Self::FastTanh => {
                for x in data.iter_mut() {
                    *x = fast_tanh(*x);
                }
            }
            Self::ReLU => {
                for x in data.iter_mut() {
                    if *x < 0.0 {
                        *x = 0.0;
                    }
                }
            }
            Self::LeakyReLU { negative_slope } => {
                for x in data.iter_mut() {
                    if *x < 0.0 {
                        *x *= negative_slope;
                    }
                }
            }
            Self::PReLU { negative_slopes } => {
                if negative_slopes.is_empty() {
                    return;
                }
                for (i, x) in data.iter_mut().enumerate() {
                    if *x < 0.0 {
                        *x *= negative_slopes[i % negative_slopes.len()];
                    }
                }
            }
            Self::Sigmoid => {
                for x in data.iter_mut() {
                    *x = 1.0 / (1.0 + (-*x).exp());
                }
            }
            Self::SiLU => {
                for x in data.iter_mut() {
                    let s = 1.0 / (1.0 + (-*x).exp());
                    *x *= s;
                }
            }
            Self::HardSwish => {
                for x in data.iter_mut() {
                    let t = *x + 3.0;
                    let clamped = t.clamp(0.0, 6.0);
                    *x *= clamped * (1.0 / 6.0);
                }
            }
            Self::LeakyHardTanh {
                min_val,
                max_val,
                min_slope,
                max_slope,
            } => {
                for x in data.iter_mut() {
                    if *x < *min_val {
                        *x = (*x - *min_val) * *min_slope + *min_val;
                    } else if *x > *max_val {
                        *x = (*x - *max_val) * *max_slope + *max_val;
                    }
                }
            }
            Self::Softsign => {
                for x in data.iter_mut() {
                    *x /= 1.0 + x.abs();
                }
            }
        }
    }
}

/// Aproximação racional rápida para a função tanh.
/// Paridade com `NAM/activations.h` L111-122.
#[inline(always)]
#[allow(clippy::excessive_precision)]
fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;

    (x * (2.4555075 + 2.4555075 * ax + (0.89322985 + 0.82122667 * ax) * x2))
        / (2.4450663 + (2.4450663 + x2) * (x + 0.81464273 * x * ax).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh() {
        let mut data = [0.0, 1.0, -1.0];
        ActivationType::Tanh.apply(&mut data);
        assert_eq!(data[0], 0.0f32.tanh());
        assert_eq!(data[1], 1.0f32.tanh());
        assert_eq!(data[2], (-1.0f32).tanh());
    }

    #[test]
    fn test_hard_tanh() {
        let mut data = [-2.0, -0.5, 0.0, 0.5, 2.0];
        ActivationType::HardTanh.apply(&mut data);
        assert_eq!(data, [-1.0, -0.5, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_relu() {
        let mut data = [-1.0, 0.0, 1.0];
        ActivationType::ReLU.apply(&mut data);
        assert_eq!(data, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let mut data = [-1.0, 0.0, 1.0];
        ActivationType::LeakyReLU {
            negative_slope: 0.01,
        }
        .apply(&mut data);
        assert_eq!(data, [-0.01, 0.0, 1.0]);
    }

    #[test]
    fn test_prelu() {
        let mut data = [-1.0, -2.0, 1.0];
        ActivationType::PReLU {
            negative_slopes: vec![0.1, 0.2],
        }
        .apply(&mut data);
        // data[0] -> -1.0 * 0.1 = -0.1
        // data[1] -> -2.0 * 0.2 = -0.4
        // data[2] -> 1.0 (positive)
        assert_eq!(data, [-0.1, -0.4, 1.0]);
    }

    #[test]
    fn test_sigmoid() {
        let mut data = [0.0];
        ActivationType::Sigmoid.apply(&mut data);
        assert_eq!(data[0], 0.5);
    }

    #[test]
    fn test_silu() {
        let mut data = [1.0];
        ActivationType::SiLU.apply(&mut data);
        let expected = 1.0 * (1.0 / (1.0 + (-1.0f32).exp()));
        assert!((data[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_hard_swish() {
        let mut data = [-4.0, 0.0, 4.0];
        ActivationType::HardSwish.apply(&mut data);
        // -4.0: t=-1, clamp=0 -> 0
        // 0.0: t=3, clamp=3 -> 0 * 3 / 6 = 0
        // 4.0: t=7, clamp=6 -> 4 * 6 / 6 = 4
        assert_eq!(data, [0.0, 0.0, 4.0]);
    }

    #[test]
    fn test_softsign() {
        let mut data = [1.0, -1.0];
        ActivationType::Softsign.apply(&mut data);
        assert_eq!(data[0], 1.0 / 2.0);
        assert_eq!(data[1], -1.0 / 2.0);
    }

    #[test]
    fn test_leaky_hardtanh() {
        let mut data = [-2.0, 0.0, 2.0];
        ActivationType::LeakyHardTanh {
            min_val: -1.0,
            max_val: 1.0,
            min_slope: 0.1,
            max_slope: 0.2,
        }
        .apply(&mut data);
        // -2.0: < -1.0 -> (-2 - -1)*0.1 + -1 = -1.1
        // 0.0: in range -> 0.0
        // 2.0: > 1.0 -> (2 - 1)*0.2 + 1 = 1.2
        assert!((data[0] - (-1.1)).abs() < 1e-6);
        assert_eq!(data[1], 0.0);
        assert!((data[2] - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_fast_tanh_parity() {
        // Just a smoke test to ensure it runs
        let mut data = [0.5];
        ActivationType::FastTanh.apply(&mut data);
        assert!(data[0] > 0.0 && data[0] < 1.0);
    }
}
