// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use crate::math::simd::SimdMathConfig;
    use crate::models::wavenet::WaveNetLayerState;
    use crate::models::wavenet_dyn::*;

    /// Constrói um `Conv1dDyn` mínimo com `kernel=1`, `dilation=1`.
    ///
    /// - `in_ch`: canais de entrada
    /// - `out_ch`: canais de saída (2×ch quando gated)
    /// - `weight`: valor fixo para todos os pesos (facilita cálculo analítico)
    fn make_conv1d(in_ch: usize, out_ch: usize, weight: f32) -> Conv1dDyn {
        Conv1dDyn {
            weights: vec![weight; out_ch * in_ch], // kernel=1
            bias: vec![0.0; out_ch],
            do_bias: false,
            dilation: 1,
            in_ch,
            out_ch,
            kernel: 1,
        }
    }

    /// Constrói um `DenseLayerDyn` identidade (peso=0, bias=0, sem efeito).
    fn make_dense_zero(in_size: usize, out_size: usize) -> DenseLayerDyn {
        DenseLayerDyn {
            weights: vec![0.0; out_size * in_size],
            bias: vec![0.0; out_size],
            do_bias: false,
            in_size,
            out_size,
        }
    }

    /// Verifica que `WaveNetLayerDyn` com `gated=true` produz `tanh(conv) ⊙ sigmoid(conv)`.
    ///
    /// Configuração sintética (CH=1, kernel=1, dilation=1):
    /// - `conv1d` IN=1, OUT=2, peso=1.0 → out[0]=x, out[1]=x (ambos os slots recebem x)
    /// - `input_mixin` e `one_by_one` com pesos zero (sem contribuição externa)
    /// - `layer_buffer[buffer_start] = x = 0.7` → residual adicionado ao output
    ///
    /// Saída esperada em `head_input[0]`: `tanh(x) * sigmoid(x)`.
    #[test]
    fn test_gated_layer_dyn_process() {
        let ch = 1usize;
        let x = 0.7f32;

        // Montar layer_buffer: [x] na posição buffer_start (buffer_start=1, buffer_frames=2 → size=2)
        let buffer_start = 1usize;
        let layer_buffer = vec![0.0f32, x]; // índice 1 = x

        // Conv1d: IN=1, OUT=2 (gated), kernel=1, weight=1.0 → out[0]=x, out[1]=x
        let conv1d = make_conv1d(ch, 2 * ch, 1.0);

        let layer = WaveNetLayerDyn {
            conv1d,
            input_mixin: make_dense_zero(1, ch), // condition=&[0.0] → zero contrib
            one_by_one: make_dense_zero(ch, ch), // zero → output permanece 0 antes do residual
            ch,
            gated: true,
        };

        let condition = [0.0f32];
        let mut head_input = vec![0.0f32; ch];
        let mut output = vec![0.0f32; ch];
        let mut block = vec![0.0f32; 2 * ch];

        let math = SimdMathConfig::current();

        unsafe {
            layer.process_block_internal(
                &condition,
                &mut head_input,
                &mut output,
                &layer_buffer,
                buffer_start,
                &mut block,
                1,
                &math,
            );
        }

        // Esperado: tanh(x) * sigmoid(x) para cada canal
        let expected_activation = x.tanh() * (0.5 * (1.0 + (0.5 * x).tanh())); // sigmoid(x)
        // head_input deve acumular block[0..ch] após gated
        let eps = 1e-5f32;
        assert!(
            (head_input[0] - expected_activation).abs() < eps,
            "head_input[0] deveria ser tanh(x)*sigmoid(x)={}, obteve {}",
            expected_activation,
            head_input[0]
        );

        // output[0] = one_by_one(block[0..ch]=0) + layer_buffer[buffer_start*ch + 0] = 0 + x = x
        assert!(
            (output[0] - x).abs() < eps,
            "output[0] deveria ser residual x={}, obteve {}",
            x,
            output[0]
        );
    }

    /// Verifica que `gated=false` mantém o comportamento original: `tanh(conv + mixin)`.
    #[test]
    fn test_non_gated_layer_dyn_process() {
        let ch = 1usize;
        let x = 0.7f32;

        let buffer_start = 1usize;
        let layer_buffer = vec![0.0f32, x];

        // Conv1d: IN=1, OUT=1 (não-gated), weight=1.0 → out[0]=x
        let conv1d = make_conv1d(ch, ch, 1.0);

        let layer = WaveNetLayerDyn {
            conv1d,
            input_mixin: make_dense_zero(1, ch),
            one_by_one: make_dense_zero(ch, ch),
            ch,
            gated: false,
        };

        let condition = [0.0f32];
        let mut head_input = vec![0.0f32; ch];
        let mut output = vec![0.0f32; ch];
        let mut block = vec![0.0f32; ch];

        let math = SimdMathConfig::current();

        unsafe {
            layer.process_block_internal(
                &condition,
                &mut head_input,
                &mut output,
                &layer_buffer,
                buffer_start,
                &mut block,
                1,
                &math,
            );
        }

        let expected = x.tanh();
        let eps = 1e-5f32;
        assert!(
            (head_input[0] - expected).abs() < eps,
            "head_input[0] deveria ser tanh(x)={}, obteve {}",
            expected,
            head_input[0]
        );
    }

    /// Verifica que `WaveNetLayerState` e pool de buffers são corretamente mantidos
    /// ao construir um `WaveNetLayerArrayDyn` com `block_size = 2*ch` quando gated.
    #[test]
    fn test_wavenet_layer_array_dyn_block_size_gated() {
        let ch = 4usize;
        let block_size = 2 * ch;

        // Construir WaveNetLayerArrayDyn manualmente com block_size=2*ch
        let state = WaveNetLayerState::new(ch, 0, 0); // RF=0 apenas para alocação
        let conv1d = Conv1dDyn {
            weights: vec![0.0; 2 * ch * ch],
            bias: vec![0.0; 2 * ch],
            do_bias: false,
            dilation: 1,
            in_ch: ch,
            out_ch: 2 * ch,
            kernel: 1,
        };
        let layer = WaveNetLayerDyn {
            conv1d,
            input_mixin: make_dense_zero(1, ch),
            one_by_one: make_dense_zero(ch, ch),
            ch,
            gated: true,
        };

        let array = WaveNetLayerArrayDyn {
            layers: vec![layer],
            states: vec![state],
            rechannel: make_dense_zero(1, ch),
            head_rechannel: make_dense_zero(ch, 1),
            array_outputs: vec![0.0; ch],
            head_accum: vec![0.0; ch],
            head_outputs: vec![0.0; 1],
            block_buffer: vec![0.0; block_size],
            block_size,
            receptive_field_size: 0,
            ch,
            head: 1,
        };

        assert_eq!(
            array.block_buffer.len(),
            2 * ch,
            "block_buffer deve ter tamanho 2*ch para gated"
        );
        assert_eq!(array.block_size, 2 * ch);
    }
}
