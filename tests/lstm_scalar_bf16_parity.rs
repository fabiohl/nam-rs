// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use nam_rs::models::lstm::LstmModel1;
use proptest::prelude::*;

fn quantize_weight(f: f32, is_bf16: bool) -> u16 {
    if is_bf16 {
        (f.to_bits() >> 16) as u16
    } else {
        half::f16::from_f32(f).to_bits()
    }
}

prop_compose! {
    fn lstm_parity_strategy()(
        inputs in prop::collection::vec(-1.0f32..1.0f32, 1),
        weights in prop::collection::vec(-1.0f32..1.0f32, 288),
        bias in prop::collection::vec(-1.0f32..1.0f32, 32),
        head_weights in prop::collection::vec(-1.0f32..1.0f32, 8),
        head_bias in -1.0f32..1.0f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        (inputs, weights, bias, head_weights, head_bias)
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::Off)),
        .. ProptestConfig::with_cases(10_000)
    })]

    #[test]
    fn test_lstm_scalar_vs_simd_parity((inputs, weights, bias, head_weights, head_bias) in lstm_parity_strategy()) {
        let is_bf16 = nam_rs::math::common::SimdMathConfig::get().instruction_set
            == nam_rs::math::common::InstructionSet::Avx512VnniBf16;

        let scale = if is_bf16 {
            f32::MAX / 4.0
        } else {
            60000.0
        };

        // Força todos os pesos e bias para os limites extremos (+scale / -scale)
        // para garantir saturação total conforme os critérios de aceitação.
        let scaled_weights: Vec<f32> = weights.iter().map(|&w| if w >= 0.0 { scale } else { -scale }).collect();
        let scaled_bias: Vec<f32> = bias.iter().map(|&b| if b >= 0.0 { scale } else { -scale }).collect();
        let scaled_head_weights: Vec<f32> = head_weights.iter().map(|&w| if w >= 0.0 { scale } else { -scale }).collect();
        let scaled_head_bias = if head_bias >= 0.0 { scale } else { -scale };

        let mut model_simd = LstmModel1::<8, 9, 32>::new();
        let mut model_scalar = LstmModel1::<8, 9, 32>::new();

        let mut w_idx = 0;
        for k in 0..4 {
            for j in 0..9 {
                for i in 0..8 {
                    let w_bits = quantize_weight(scaled_weights[w_idx], is_bf16);
                    model_simd.layer.input_hidden_weights[k][j][i] = w_bits;
                    model_scalar.layer.input_hidden_weights[k][j][i] = w_bits;
                    w_idx += 1;
                }
            }
        }

        model_simd.layer.bias.copy_from_slice(&scaled_bias[..32]);
        model_scalar.layer.bias.copy_from_slice(&scaled_bias[..32]);

        for (i, &w) in scaled_head_weights.iter().enumerate().take(8) {
            let hw_bits = quantize_weight(w, is_bf16);
            model_simd.head_weights[i] = hw_bits;
            model_scalar.head_weights[i] = hw_bits;
        }

        model_simd.head_bias = scaled_head_bias;
        model_scalar.head_bias = scaled_head_bias;

        model_simd.reset_states();
        model_scalar.reset_states();

        let mut out_simd = vec![0.0f32; inputs.len()];
        let mut out_scalar = vec![0.0f32; inputs.len()];

        model_simd.process(&inputs, &mut out_simd);
        model_scalar.process_scalar(&inputs, &mut out_scalar);

        for i in 0..inputs.len() {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let max_val = out_simd[i].abs().max(out_scalar[i].abs()).max(1.0);
            let rel_diff = diff / max_val;
            assert!(
                rel_diff < 5.0e-3 || (out_simd[i].is_nan() && out_scalar[i].is_nan()) || (out_simd[i].is_infinite() && out_scalar[i].is_infinite()),
                "LSTM Parity falhou no index {}: SIMD={}, Scalar={}, Delta={}, Relative Delta={}",
                i,
                out_simd[i],
                out_scalar[i],
                diff,
                rel_diff
            );
        }
    }
}
