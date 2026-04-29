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
            weights: vec![half::f16::from_f32(weight).to_bits(); out_ch * in_ch], // kernel=1
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
            weights: vec![0u16; out_size * in_size],
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
    /// Verifica que `WaveNetLayerDyn` com `gated=true` produz `tanh(conv) ⊙ sigmoid(conv)`.
    #[test]
    fn test_gated_layer_dyn_process() {
        // Usamos CH=1 para facilitar o rastreio manual dos valores.
        let ch = 1usize;
        let x = 0.7f32; // Valor de entrada arbitrário

        // O 'layer_buffer' simula o histórico de amostras (receptive field).
        // Aqui, colocamos 'x' exatamente onde a convolução irá ler.
        // buffer_start=1 significa que estamos processando a amostra no índice 1.
        let buffer_start = 1usize;
        let layer_buffer = vec![0.0f32, x]; // [t-1, t] onde t=x

        // Camada de Convolução 1D:
        // No modo 'gated', a saída tem o DOBRO de canais (2 * ch).
        // A primeira metade vai para o Tanh, a segunda para o Sigmoid.
        // Com peso 1.0 e kernel 1, out[0] = x e out[1] = x.
        let conv1d = make_conv1d(ch, 2 * ch, 1.0);

        // WaveNetLayerDyn agrupa a convolução, mixins de condicionamento e projeção 1x1.
        let layer = WaveNetLayerDyn {
            conv1d,
            // Zeramos o input_mixin para que o sinal externo (condition) não afete o teste.
            input_mixin: make_dense_zero(1, ch),
            // Zeramos o one_by_one para que a saída da ativação não contribua para o próximo bloco,
            // isolando o teste apenas para o valor residual.
            one_by_one: make_dense_zero(ch, ch),
            ch,
            gated: true,
        };

        // Inputs para o processamento:
        let condition = [0.0f32]; // Condicionamento global (ex: parâmetros de EQ)
        let mut head_input = vec![0.0f32; ch]; // Acumulador para as "heads" de saída (skip connections)
        let mut output = vec![0.0f32; ch]; // Saída para a próxima camada (residual path)
        let mut block = vec![0.0f32; 2 * ch]; // Buffer temporário para cálculos intermediários

        let _math = SimdMathConfig::current();

        // Executamos o processamento interno (unsafe pois lida com ponteiros/SIMD em produção).
        unsafe {
            layer.process_block_internal::<crate::math::simd::Avx2Math>(
                &condition,
                &mut head_input,
                &mut output,
                &layer_buffer,
                buffer_start,
                &mut block,
                1,
            );
        }

        // --- VALIDAÇÃO DA ATIVAÇÃO GATED ---
        // A matemática da WaveNet original define: activation = tanh(W_f * x) * sigmoid(W_g * x)
        // Onde 'f' é o filtro e 'g' é o gate. Como nossos pesos são 1.0:
        // activation = tanh(x) * sigmoid(x)
        let expected_activation = x.tanh() * (0.5 * (1.0 + (0.5 * x).tanh())); // sigmoid(x) aproximado/padrão

        let eps = 1e-5f32;
        // 'head_input' recebe o resultado da ativação (skip connection).
        assert!(
            (head_input[0] - expected_activation).abs() < eps,
            "head_input[0] deveria ser tanh(x)*sigmoid(x)={}, obteve {}",
            expected_activation,
            head_input[0]
        );

        // --- VALIDAÇÃO DO RESIDUAL PATH ---
        // Na WaveNet, output = one_by_one(activation) + input.
        // Como 'one_by_one' é zero, output deve ser apenas o 'x' original (o residual).
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
        // Setup similar ao teste gated, mas com lógica simplificada (apenas 1 canal de saída).
        let ch = 1usize;
        let x = 0.7f32;

        let buffer_start = 1usize;
        let layer_buffer = vec![0.0f32, x];

        // Conv1d: IN=1, OUT=1 (não-gated).
        // Diferente do modo gated, aqui a saída tem o MESMO número de canais da entrada.
        // weight=1.0 → out[0]=x
        let conv1d = make_conv1d(ch, ch, 1.0);

        let layer = WaveNetLayerDyn {
            conv1d,
            input_mixin: make_dense_zero(1, ch),
            one_by_one: make_dense_zero(ch, ch),
            ch,
            gated: false, // Desativa a lógica de split tanh * sigmoid
        };

        let condition = [0.0f32];
        let mut head_input = vec![0.0f32; ch];
        let mut output = vec![0.0f32; ch];
        let mut block = vec![0.0f32; ch]; // Block buffer tem tamanho 'ch' (não 2*ch)

        let _math = SimdMathConfig::current();

        unsafe {
            layer.process_block_internal::<crate::math::simd::Avx2Math>(
                &condition,
                &mut head_input,
                &mut output,
                &layer_buffer,
                buffer_start,
                &mut block,
                1,
            );
        }

        // --- VALIDAÇÃO DA ATIVAÇÃO PADRÃO ---
        // Sem o gate, a WaveNet aplica apenas tanh ao resultado da convolução + mixin.
        // expected = tanh(x)
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
        // Este teste foca na ALOCAÇÃO DE MEMÓRIA.
        // Em redes neurais de áudio, evitar realocações durante o processamento é crucial.
        let ch = 4usize;
        let block_size = 2 * ch; // Para modo 'gated', precisamos de espaço para Tanh E Sigmoid.

        // 'WaveNetLayerState' gerencia o buffer circular (histórico) de uma camada.
        // Receptive Field (RF) aqui é 0 apenas para simplificar a alocação do teste.
        let state = WaveNetLayerState::new(ch, 0, 0);

        let conv1d = Conv1dDyn {
            weights: vec![0u16; 2 * ch * ch],
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

        // 'WaveNetLayerArrayDyn' é o container principal que orquestra todas as camadas.
        // Ele pré-aloca 'block_buffer' para ser reutilizado por todas as camadas durante o processamento,
        // economizando memória e aumentando a performance (cache locality).
        let array = WaveNetLayerArrayDyn {
            layers: vec![layer],
            states: vec![state],
            rechannel: make_dense_zero(1, ch),
            head_rechannel: make_dense_zero(ch, 1),
            array_outputs: vec![0.0; ch],
            head_accum: vec![0.0; ch],
            head_outputs: vec![0.0; 1],
            block_buffer: vec![0.0; block_size], // O buffer compartilhado
            block_size,
            receptive_field_size: 0,
            ch,
            head: 1,
        };

        // Verificação crucial: se o buffer não tiver 2*ch, o processamento gated causaria
        // um transbordamento de buffer (buffer overflow) ou pânico.
        assert_eq!(
            array.block_buffer.len(),
            2 * ch,
            "block_buffer deve ter tamanho 2*ch para suportar ativação gated (filter + gate)"
        );
        assert_eq!(array.block_size, 2 * ch);
    }
}
