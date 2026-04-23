// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use crate::models::wavenet::*;

    /// Constrói um WaveNetModel<4, 3, 2> mínimo para testes com dados zerados.
    /// Array1: IN=1, COND=1, CH=4, K=3, HEAD=2
    /// Array2: IN=4, COND=1, CH=2 (=HEAD), K=3, HEAD2=1
    fn build_tiny_wavenet() -> WaveNetModel<4, 3, 2> {
        let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 3> {
            WaveNetLayer {
                conv1d: Conv1d {
                    weights: vec![0.01; 4 * 3 * 4],
                    bias: vec![0.0; 4],
                    do_bias: false,
                    dilation,
                },
                input_mixin: DenseLayer {
                    weights: vec![0.01; 4],
                    bias: vec![0.0; 4],
                    do_bias: false,
                },
                one_by_one: DenseLayer {
                    weights: vec![0.01; 4 * 4],
                    bias: vec![0.0; 4],
                    do_bias: false,
                },
            }
        };

        // Array2: CH=2 (=HEAD), layers com COND=1, CH=2
        let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 3> {
            WaveNetLayer {
                conv1d: Conv1d {
                    weights: vec![0.01; 2 * 3 * 2],
                    bias: vec![0.0; 2],
                    do_bias: false,
                    dilation,
                },
                input_mixin: DenseLayer {
                    weights: vec![0.01; 2],
                    bias: vec![0.0; 2],
                    do_bias: false,
                },
                one_by_one: DenseLayer {
                    weights: vec![0.01; 2 * 2],
                    bias: vec![0.0; 2],
                    do_bias: false,
                },
            }
        };

        let dilations_1 = [1, 2, 4];
        let dilations_2 = [1, 2, 4];

        let rf1 = *dilations_1.iter().max().unwrap_or(&1) * (3 - 1);
        let rf2 = *dilations_2.iter().max().unwrap_or(&1) * (3 - 1);

        // Construção manual das arrays com const generics explícitos.
        let layers_1: Vec<WaveNetLayer<1, 4, 3>> =
            dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
        let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
            .map(|i| WaveNetLayerState::new(4, rf1, i))
            .collect();

        let array1 = WaveNetLayerArray::<1, 1, 4, 3, 2> {
            layers: layers_1,
            states: states_1,
            rechannel: DenseLayer {
                weights: vec![0.01; 4],
                bias: vec![0.0; 4],
                do_bias: false,
            },
            head_rechannel: DenseLayer {
                weights: vec![0.01; 2 * 4],
                bias: vec![0.0; 2],
                do_bias: false,
            },
            array_outputs: vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES],
            head_accum: vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES],
            head_outputs: vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES],
            receptive_field_size: rf1,
        };

        // Array2: IN=4(=CH), COND=1, CH=2(=HEAD), K=3, HEAD2=1
        let layers_2: Vec<WaveNetLayer<1, 2, 3>> =
            dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
        let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
            .map(|i| WaveNetLayerState::new(2, rf2, i))
            .collect();

        let array2 = WaveNetLayerArray::<4, 1, 2, 3, 1> {
            layers: layers_2,
            states: states_2,
            rechannel: DenseLayer {
                weights: vec![0.01; 4 * 2],
                bias: vec![0.0; 2],
                do_bias: false,
            },
            head_rechannel: DenseLayer {
                weights: vec![0.01; 2],
                bias: vec![0.0; 1],
                do_bias: true, // array2 HasHeadBias=true
            },
            array_outputs: vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES],
            head_accum: vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES],
            head_outputs: vec![0.0; WAVENET_MAX_NUM_FRAMES],
            receptive_field_size: rf2,
        };

        WaveNetModel {
            array1,
            array2,
            head_scale: 0.02,
            receptive_field_size: rf1.max(rf2),
        }
    }

    #[test]
    fn test_wavenet_model_allocation() {
        let model = build_tiny_wavenet();
        assert_eq!(model.array1.layers.len(), 3);
        assert_eq!(model.array2.layers.len(), 3);
        assert_eq!(model.array1.head_outputs.len(), 2 * WAVENET_MAX_NUM_FRAMES); // HEAD1=2
        assert_eq!(model.array2.head_outputs.len(), WAVENET_MAX_NUM_FRAMES); // HEAD2=1 (sempre fixo)
        assert!((model.head_scale - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_wavenet_prewarm_no_nan() {
        let mut model = build_tiny_wavenet();
        model.prewarm();

        // Verificar que os buffers internos não contêm NaN/Inf após prewarm
        for state in &model.array1.states {
            for &v in &state.layer_buffer {
                assert!(v.is_finite(), "NaN/Inf detectado no array1 após prewarm");
            }
        }
        for state in &model.array2.states {
            for &v in &state.layer_buffer {
                assert!(v.is_finite(), "NaN/Inf detectado no array2 após prewarm");
            }
        }
    }

    #[test]
    fn test_wavenet_process_zeros() {
        let mut model = build_tiny_wavenet();
        model.prewarm();

        let input = [0.0f32; 16];
        let mut output = [0.0f32; 16];

        model.process(&input, &mut output);

        for (i, &v) in output.iter().enumerate() {
            assert!(v.is_finite(), "Amostra de saída [{}] é NaN/Inf: {}", i, v);
        }
    }

    #[test]
    fn test_wavenet_process_deterministic() {
        let mut model_a = build_tiny_wavenet();
        let mut model_b = build_tiny_wavenet();

        model_a.prewarm();
        model_b.prewarm();

        let input = [0.1f32; 8];
        let mut out_a = [0.0f32; 8];
        let mut out_b = [0.0f32; 8];

        model_a.process(&input, &mut out_a);
        model_b.process(&input, &mut out_b);

        for i in 0..8 {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-6,
                "Resultado não-determinístico na amostra [{}]: {} vs {}",
                i,
                out_a[i],
                out_b[i]
            );
        }
    }

    #[test]
    fn test_conv1d_identity_kernel() {
        let mut weights = vec![0.0; 16]; // 4 * 1 * 4
        for i in 0..4 {
            weights[i * 4 + i] = 1.0;
        }

        let conv = Conv1d::<4, 4, 1> {
            weights,
            bias: vec![0.0; 4],
            do_bias: false,
            dilation: 1,
        };

        let layer_buffer = vec![1.0, 2.0, 3.0, 4.0];
        let mut block = vec![0.0; 4];

        unsafe {
            conv.process_block::<crate::math::simd::Avx2Math>(&layer_buffer, &mut block, 0, 1);
        }

        assert_eq!(block, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let mut weights = vec![0.0; 16]; // 4 * 1 * 4
        for i in 0..4 {
            weights[i * 4 + i] = 1.0;
        }

        let conv = Conv1d::<4, 4, 1> {
            weights,
            bias: vec![0.5; 4],
            do_bias: true,
            dilation: 1,
        };

        let layer_buffer = vec![1.0, 2.0, 3.0, 4.0];
        let mut block = vec![0.0; 4];

        unsafe {
            conv.process_block::<crate::math::simd::Avx2Math>(&layer_buffer, &mut block, 0, 1);
        }

        assert_eq!(block, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_conv1d_dilation() {
        let mut weights = vec![0.0; 2 * 3 * 2];
        for i in 0..12 {
            weights[i] = 1.0;
        }

        let conv = Conv1d::<2, 2, 3> {
            weights,
            bias: vec![0.0; 2],
            do_bias: false,
            dilation: 2,
        };

        let mut layer_buffer = vec![0.0; 6 * 2];
        layer_buffer[0] = 1.0;
        layer_buffer[1] = 2.0;
        layer_buffer[2] = 10.0;
        layer_buffer[3] = 20.0;
        layer_buffer[4] = 3.0;
        layer_buffer[5] = 4.0;
        layer_buffer[6] = 30.0;
        layer_buffer[7] = 40.0;
        layer_buffer[8] = 5.0;
        layer_buffer[9] = 6.0;

        let mut block = vec![0.0; 2];

        unsafe {
            conv.process_block::<crate::math::simd::Avx2Math>(&layer_buffer, &mut block, 4, 1);
        }

        assert_eq!(block[0], 21.0);
        assert_eq!(block[1], 21.0);
    }

    #[test]
    fn test_conv1d_zero_input() {
        let mut weights = vec![0.0; 2 * 3 * 2];
        for i in 0..12 {
            weights[i] = 100.0;
        }

        let mut conv = Conv1d::<2, 2, 3> {
            weights: weights.clone(),
            bias: vec![0.0; 2],
            do_bias: false,
            dilation: 1,
        };

        let layer_buffer = vec![0.0; 4 * 2];
        let mut block = vec![0.0; 2];

        unsafe {
            conv.process_block::<crate::math::simd::Avx2Math>(&layer_buffer, &mut block, 2, 1);
        }

        assert_eq!(block, vec![0.0, 0.0]);

        conv.do_bias = true;
        conv.bias = vec![7.5, 8.5];

        unsafe {
            conv.process_block::<crate::math::simd::Avx2Math>(&layer_buffer, &mut block, 2, 1);
        }

        assert_eq!(block, vec![7.5, 8.5]);
    }

    #[test]
    fn test_conv1d_known_output() {
        let mut weights = vec![0.0; 2 * 2 * 2];
        weights[0] = 0.5;
        weights[1] = 1.0;
        weights[2] = 1.5;
        weights[3] = 2.0;
        weights[4] = -0.5;
        weights[5] = -1.0;
        weights[6] = -1.5;
        weights[7] = -2.0;

        let conv = Conv1d::<2, 2, 2> {
            weights,
            bias: vec![1.0, -1.0],
            do_bias: true,
            dilation: 1,
        };

        let layer_buffer = vec![2.0, 3.0, 4.0, 5.0];
        let mut block = vec![0.0; 2];

        unsafe {
            conv.process_block::<crate::math::simd::Avx2Math>(&layer_buffer, &mut block, 1, 1);
        }

        assert_eq!(block[0], 21.0);
        assert_eq!(block[1], -21.0);
    }

    #[test]
    fn test_dense_layer_identity() {
        let mut weights = vec![0.0; 16]; // OUT=4 * IN=4
        for out_c in 0..4 {
            weights[out_c * 4 + out_c] = 1.0;
        }

        let dense = DenseLayer::<4, 4> {
            weights,
            bias: vec![0.0; 4],
            do_bias: false,
        };

        let input = vec![1.5, 2.5, 3.5, 4.5];
        let mut output = vec![0.0; 4];

        unsafe {
            dense.process_block::<crate::math::simd::Avx2Math>(&input, &mut output, 1);
        }

        assert_eq!(output, vec![1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_dense_layer_with_bias() {
        let mut weights = vec![0.0; 16];
        for out_c in 0..4 {
            weights[out_c * 4 + out_c] = 1.0;
        }

        let dense = DenseLayer::<4, 4> {
            weights,
            bias: vec![1.0; 4],
            do_bias: true,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0; 4];

        unsafe {
            dense.process_block::<crate::math::simd::Avx2Math>(&input, &mut output, 1);
        }

        assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_dense_layer_rectangular() {
        // IN=8, OUT=4. Pesos conhecidos, output verificado manualmente.
        let mut weights = vec![0.0; 32]; // 4 * 8

        // out_c = 0: in[0]*1 + in[1]*2
        weights[0] = 1.0;
        weights[1] = 2.0;

        // out_c = 1: in[2]*3 + in[3]*4
        weights[10] = 3.0;
        weights[11] = 4.0;

        // out_c = 2: in[4]*0.5
        weights[20] = 0.5;

        // out_c = 3: in[7]*(-1.0)
        weights[31] = -1.0;

        let dense = DenseLayer::<8, 4> {
            weights,
            bias: vec![0.5, -0.5, 1.0, -1.0],
            do_bias: true,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0; 4];

        unsafe {
            dense.process_block::<crate::math::simd::Avx2Math>(&input, &mut output, 1);
        }

        // Expected output:
        // out[0] = 1*1.0 + 2*2.0 + 0.5 = 5.5
        // out[1] = 3*3.0 + 4*4.0 - 0.5 = 9.0 + 16.0 - 0.5 = 24.5
        // out[2] = 5*0.5 + 1.0 = 2.5 + 1.0 = 3.5
        // out[3] = 8*(-1.0) - 1.0 = -8.0 - 1.0 = -9.0

        assert_eq!(output[0], 5.5);
        assert_eq!(output[1], 24.5);
        assert_eq!(output[2], 3.5);
        assert_eq!(output[3], -9.0);
    }
}
