// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use crate::models::lstm::{LstmModel1, LstmModel2};

    // Verify that internal buffers (gates and state) are allocated with the correct
    // size based on Const Generics parameters.
    #[test]
    fn test_lstm_model1_allocation() {
        let model: LstmModel1<8, 9, 32> = LstmModel1::new();
        assert_eq!(model.layer.gates.len(), 32);
        assert_eq!(model.layer.state.len(), 9);
    }

    // Verify allocation in a two-layer model (LstmModel2).
    #[test]
    fn test_lstm_model2_allocation() {
        let model: LstmModel2<16, 17, 32, 64> = LstmModel2::new();
        assert_eq!(model.layer1.input_hidden_weights.len(), 4);
        assert_eq!(model.layer2.input_hidden_weights[0].len(), 32);
    }

    // Basic sanity test: process some values and ensure the output
    // contains no invalid values (NaN or Infinity).
    #[test]
    fn test_lstm_model1_process_zeros() {
        {
            let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();
            let input = [1.0, -1.0, 0.5, -0.5];
            let mut output = [0.0f32; 4];

            model.process(&input, &mut output);

            for (i, &v) in output.iter().enumerate() {
                assert!(v.is_finite(), "LSTM1×8: output [{}] is NaN/Inf: {}", i, v);
            }
        }
    }

    // Ensure processing is deterministic: the same input must
    // always produce the same output if the initial state is equal.
    #[test]
    fn test_lstm_model2_process_deterministic() {
        {
            let mut model_a: LstmModel2<8, 9, 16, 32> = LstmModel2::new();
            let mut model_b: LstmModel2<8, 9, 16, 32> = LstmModel2::new();

            let input = [1.0, -1.0, 0.5, -0.5];
            let mut out_a = [0.0f32; 4];
            let mut out_b = [0.0f32; 4];

            model_a.process(&input, &mut out_a);
            model_b.process(&input, &mut out_b);

            for i in 0..4 {
                assert!(
                    (out_a[i] - out_b[i]).abs() < 1e-6,
                    "LSTM2×8 non-deterministic at [{}]: {} vs {}",
                    i,
                    out_a[i],
                    out_b[i]
                );
            }
        }
    }

    // Validates whether the LSTM gate memory layout [I, F, C, O] follows the
    // expected order so that SIMD optimizations work correctly.
    #[test]
    fn test_lstm_gate_order_consistency() {
        // Validates that the [i|f|g|o] layout at offsets [0, H, 2H, 3H] is respected.
        let model: LstmModel1<8, 9, 32> = LstmModel1::new();
        assert_eq!(model.layer.gates.len(), 32); // 4 * H = 4 * 8 = 32
        assert_eq!(model.layer.input_hidden_weights.len(), 4); // Gates = 4
        assert_eq!(model.layer.input_hidden_weights[0].len(), 9); // IH = I + H = 1 + 8 = 9
    }

    // Verify that the LSTM's internal state (memory) evolves as we process data.
    // The hidden state after 10 samples should be different from the state after 1 sample.
    #[test]
    fn test_lstm_state_evolution() {
        let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();

        // Set some weights so the state changes significantly
        for i in 0..32 {
            model.layer.bias[i] = 0.1;
        }
        for k in 0..4 {
            for j in 0..9 {
                for i in 0..8 {
                    model.layer.input_hidden_weights[k][j][i] = half::f16::from_f32(0.05).to_bits();
                }
            }
        }

        let input_step = [1.0f32];
        let mut out = [0.0f32];

        model.process(&input_step, &mut out);
        let hidden_1 = model.layer.get_hidden_state().to_vec();

        for _ in 0..9 {
            model.process(&input_step, &mut out);
        }
        let hidden_10 = model.layer.get_hidden_state().to_vec();

        // Verify evolution (hidden states should be different after multiple iterations)
        for i in 0..8 {
            assert!(
                (hidden_1[i] - hidden_10[i]).abs() > 1e-4,
                "Hidden state did not evolve at position {}: {} vs {}",
                i,
                hidden_1[i],
                hidden_10[i]
            );
        }
    }

    // CRITICAL TEST: Ensures the result is the same regardless of the processed
    // block size (e.g., processing 64 samples at once or 64 times one sample).
    // This is vital to ensure audio doesn't change if the driver changes the buffer size.
    #[test]
    fn test_lstm_variable_block_sizes() {
        let mut input = [0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate() {
            *val = ((i as f32) * 0.1).sin(); // Simple sine wave
        }

        let block_sizes = [1, 8, 16, 32, 64];
        let mut reference_out = [0.0f32; 64];

        for (idx, &block_size) in block_sizes.iter().enumerate() {
            let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();

            // Assign deterministic weights to test identical output
            for i in 0..32 {
                model.layer.bias[i] = 0.1;
            }
            for k in 0..4 {
                for j in 0..9 {
                    for i in 0..8 {
                        model.layer.input_hidden_weights[k][j][i] =
                            half::f16::from_f32(0.05).to_bits();
                    }
                }
            }
            for i in 0..8 {
                model.head_weights[i] = half::f16::from_f32(0.5).to_bits();
            }

            let mut out = [0.0f32; 64];
            for chunk in 0..(64 / block_size) {
                let start = chunk * block_size;
                let end = start + block_size;
                model.process(&input[start..end], &mut out[start..end]);
            }

            if idx == 0 {
                reference_out.copy_from_slice(&out);
            } else {
                for i in 0..64 {
                    assert!(
                        (out[i] - reference_out[i]).abs() < 1e-6,
                        "Mismatch at block_size {} at sample {}: {} vs {}",
                        block_size,
                        i,
                        out[i],
                        reference_out[i]
                    );
                }
            }
        }
    }

    // Verify that reset and prewarm functions are clearing states correctly.
    #[test]
    fn test_lstm_reset_on_prewarm() {
        use crate::models::NamModel; // trait
        let mut model: LstmModel1<8, 9, 32> = LstmModel1::new();

        // Set bias and weights to ensure the state shifts away from zero
        for i in 0..32 {
            model.layer.bias[i] = 1.0;
        }

        let input = [1.0f32; 64];
        let mut out = [0.0f32; 64];
        model.process(&input, &mut out);

        // Artificially change the cell state
        model.layer.cell_state.fill(5.0);
        model.layer.state.fill(5.0);

        // Verify the state is not zeroed
        assert!(model.layer.cell_state[0] != 0.0);
        assert!(model.layer.state[0] != 0.0);

        // Calls prewarm — internally calls reset_states() and processes N zeros
        model.prewarm(0);
        model.prewarm(10);

        // Verify that reset_states() effectively zeros hidden and cell states
        model.reset_states();
        for val in model.layer.cell_state.iter() {
            assert_eq!(*val, 0.0);
        }
        for val in model.layer.state.iter() {
            assert_eq!(*val, 0.0);
        }
    }

    // PARITY TEST: Ensures the pipelined version (SIMD) produces the same
    // result as the sequential version (Scalar).
    #[test]
    fn test_lstm_model2_pipelining_parity() {
        let mut model_simd: LstmModel2<8, 9, 17, 32> = LstmModel2::new();
        let mut model_scalar: LstmModel2<8, 9, 17, 32> = LstmModel2::new();

        // Assign deterministic weights
        for i in 0..32 {
            model_simd.layer1.bias[i] = 0.1;
            model_scalar.layer1.bias[i] = 0.1;
            model_simd.layer2.bias[i] = -0.05;
            model_scalar.layer2.bias[i] = -0.05;
        }
        for k in 0..4 {
            for j in 0..17 {
                for i in 0..8 {
                    let w = half::f16::from_f32(0.01 * (i + j + k) as f32).to_bits();
                    if j < 9 {
                        model_simd.layer1.input_hidden_weights[k][j][i] = w;
                        model_scalar.layer1.input_hidden_weights[k][j][i] = w;
                    }
                    model_simd.layer2.input_hidden_weights[k][j][i] = w;
                    model_scalar.layer2.input_hidden_weights[k][j][i] = w;
                }
            }
        }
        for i in 0..8 {
            let w = half::f16::from_f32(0.1 * i as f32).to_bits();
            model_simd.head_weights[i] = w;
            model_scalar.head_weights[i] = w;
        }

        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut out_simd = [0.0f32; 64];
        let mut out_scalar = [0.0f32; 64];

        model_simd.process(&input, &mut out_simd);
        model_scalar.process_scalar(&input, &mut out_scalar);

        for i in 0..64 {
            assert!(
                (out_simd[i] - out_scalar[i]).abs() < 5e-3,
                "SIMD/Pipelined vs Scalar parity failed at [{}]: {} vs {}",
                i,
                out_simd[i],
                out_scalar[i]
            );
        }
    }
}
