// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::{AlignedVec, SimdMathConfig};
use crate::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use crate::models::wavenet::conv1d::Conv1d;
use crate::models::wavenet::dense::DenseLayer;
use crate::models::wavenet::model::*;

/// Constrói um WaveNetModel<4, 3, 2> mínimo para testes com dados estáticos e controlados.
/// Esta função serve como um "mock" (modelo simulado) para os testes unitários.
fn build_tiny_wavenet() -> WaveNetModel<4, 3, 2> {
    // Fábrica de camadas para o Array 1 (Main Array).
    // Generics: <COND=1, CH=4, K=3>.
    // No WaveNet, cada camada é uma unidade funcional que processa o sinal dilatado.
    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 3> {
        WaveNetLayer {
            // A Convolução Causal Dilatada permite capturar dependências temporais longas
            // sem aumentar linearmente o número de parâmetros.
            conv1d: Conv1d {
                // Dimensões: OUT * K * IN = 4 * 3 * 4.
                // Aqui, IN=CH pois a camada recebe o sinal das camadas anteriores.
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 3 * 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    crate::math::common::prefetch_strategy_2stage
                } else {
                    crate::math::common::prefetch_strategy_simple
                },
            },
            // O input_mixin injeta o condicionamento (ex: metadados do timbre) no sinal.
            // Dimensões: OUT * IN = 4 * 1.
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
            // A projeção 1x1 (Dense) finaliza a célula, preparando o sinal para o residual.
            // Dimensões: OUT * IN = 4 * 4.
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 4]),
                bias: AlignedVec::from_vec(vec![0.0; 4]),
                do_bias: false,
            },
        }
    };

    // Array2: CH=2 (=HEAD), layers com COND=1, CH=2
    // Fábrica de camadas para o Array 2 (Head Array).
    // Generics: <COND=1, CH=2, K=3>.
    // Este array geralmente possui menos canais e foca no refinamento final do áudio.
    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                // Dimensões: OUT * K * IN = 2 * 3 * 2.
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 3 * 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
                dilation,
                prefetch_fn: if dilation >= 128 {
                    crate::math::common::prefetch_strategy_2stage
                } else {
                    crate::math::common::prefetch_strategy_simple
                },
            },
            input_mixin: DenseLayer {
                // Dimensões: OUT * IN = 2 * 1.
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                // Dimensões: OUT * IN = 2 * 2.
                weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 2]),
                bias: AlignedVec::from_vec(vec![0.0; 2]),
                do_bias: false,
            },
        }
    };

    // Definimos o padrão de dilatação. O crescimento exponencial (1, 2, 4...)
    // é o que permite à WaveNet ter um campo receptivo vasto com poucas camadas.
    let dilations_1 = [1, 2, 4];
    let dilations_2 = [1, 2, 4];

    // Cálculo do Receptive Field (RF): determina quantos samples do passado influenciam o presente.
    // Fórmula simplificada: max_dilation * (kernel_size - 1).
    let rf1 = *dilations_1.iter().max().unwrap_or(&1) * (3 - 1);
    let rf2 = *dilations_2.iter().max().unwrap_or(&1) * (3 - 1);

    // Construção manual das arrays com const generics explícitos.
    // Array1 (Main Receptive Field): Assegura a extração principal de características.
    // Para cada dilatação, construímos uma camada e alocamos seu estado interno (buffer histórico).
    let layers_1: Vec<WaveNetLayer<1, 4, 3>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(4, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array1 = WaveNetLayerArray::<1, 1, 4, 3, 2> {
        layers: layers_1,
        states: states_1,
        // Rechannel: Projeta a entrada bruta (Mono/Stereo) para a dimensão interna (Channels).
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4]),
            bias: AlignedVec::from_vec(vec![0.0; 4]),
            do_bias: false,
        },
        // Head Rechannel: Agrega as "skip connections" de todas as camadas para a saída do array.
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2 * 4]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        // Buffers de saída pré-alocados para garantir RT-Safety (Zero Alloc no loop).
        array_outputs: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf1,
        block_size: 4,
        block_buffer: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    // Array2 (Head Definition): O array secundário atua nas predições refinadas finais.
    // IN=4(=CH de array1), COND=1, CH=2(=HEAD1), K=3, HEAD2=1
    // O `head_rechannel` define a transição final de dados.
    let layers_2: Vec<WaveNetLayer<1, 2, 3>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(2, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array2 = WaveNetLayerArray::<4, 1, 2, 3, 1> {
        layers: layers_2,
        states: states_2,
        // Projeta a saída do Array 1 (HEAD1=2) para a dimensão do Array 2 (CH2=2).
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 4 * 2]),
            bias: AlignedVec::from_vec(vec![0.0; 2]),
            do_bias: false,
        },
        // A projeção final do modelo NAM reduz tudo para 1 canal (áudio mono).
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![half::f16::from_f32(0.01).to_bits(); 2]),
            bias: AlignedVec::from_vec(vec![0.0; 1]),
            do_bias: true, // Habilitamos bias na saída final para DC offset Correction.
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        head_accum: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES]),
        receptive_field_size: rf2,
        block_size: 2,
        block_buffer: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES]),
        last_condition: [0.0; 1],
        last_condition_bf16: [0; 1],
        condition_init: false,
    };

    // O WaveNetModel orquestra a cascata de arrays e aplica o ganho final.
    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        // O RF global é o maior entre os arrays (geralmente o RF do array 1 domina).
        receptive_field_size: rf1.max(rf2),
    }
}

/// Testa se os tamanhos alocados na pilha/heap pelas estruturas do WaveNet
/// coincidem com as especificações matemáticas de canais e frames.
///
/// Como o WaveNet usa vetores e arrays fixos, é crucial garantir
/// que `head_outputs` tenha espaço exato (ex: 2 canais * 64 frames),
/// prevenindo segfaults e "out of bounds" durante o processamento de áudio em tempo real.
#[test]
fn test_wavenet_model_allocation() {
    let model = build_tiny_wavenet();
    assert_eq!(model.array1.layers.len(), 3);
    assert_eq!(model.array2.layers.len(), 3);
    assert_eq!(model.array1.head_outputs.len(), 2 * WAVENET_MAX_NUM_FRAMES); // HEAD1=2
    assert_eq!(model.array2.head_outputs.len(), WAVENET_MAX_NUM_FRAMES); // HEAD2=1 (sempre fixo)
    assert!((model.head_scale - 0.02).abs() < 1e-6);
}

/// Verifica se a inicialização "a quente" (prewarm) do modelo não gera
/// instabilidade numérica como NaN (Not a Number) ou Inf (Infinity).
///
/// O *prewarm* serve para estabilizar a rede (especialmente buffers de delay)
/// iterando repetidas vezes com zeros antes de iniciar a reprodução. Um bug
/// de memória não inicializada rapidamente apareceria aqui.
#[test]
fn test_wavenet_prewarm_no_nan() {
    let mut model = build_tiny_wavenet();
    model.prewarm();

    // Verificar que os buffers internos não contêm NaN/Inf após prewarm
    for state in &model.array1.states {
        for &v in state.layer_buffer.iter() {
            assert!(v.is_finite(), "NaN/Inf detectado no array1 após prewarm");
        }
    }
    for state in &model.array2.states {
        for &v in state.layer_buffer.iter() {
            assert!(v.is_finite(), "NaN/Inf detectado no array2 após prewarm");
        }
    }
}

/// Processa um bloco inteiro de silêncio absoluto (zeros) e garante
/// que o modelo reaja de forma linear e previsível.
///
/// Silêncio na entrada deve, no máximo, extrair os *bias* (vieses) do modelo
/// de maneira estável. Qualquer divergência para NaN ou Inf significa que
/// o modelo sofreu divisão por zero ou corrupção de ponteiro no fluxo SIMD.
#[test]
fn test_wavenet_process_zeros() {
    // Instancia o modelo "tiny" (mock) e estabiliza os Ring Buffers históricos.
    let mut model = build_tiny_wavenet();
    model.prewarm();

    // Prepara um bloco de 16 amostras de silêncio absoluto.
    let input = [0.0f32; 16];
    let mut output = [0.0f32; 16];

    // O processo forward deve converter silêncio em valores estáveis (geralmente pequenos DC offsets).
    model.process(&input, &mut output);

    // Validamos se o sinal de saída é finito. Se houver divisão por zero ou overflow
    // em qualquer camada SIMD, o resultado será NaN ou Inf, o que é inaceitável em DSP.
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Amostra de saída [{}] é NaN/Inf: {}", i, v);
    }
}

/// Teste de integridade determinística.
///
/// Executa duas instâncias idênticas do modelo processando o mesmo estímulo.
/// O motor DSP (neste caso, as rotinas SIMD AVX2) precisa ser matematicamente
/// determinístico. Diferenças pontuais (Flutuantes divergentes) indicariam
/// vazamento de estado de processamentos anteriores (*state bleeding*),
/// o que arruinaria a qualidade de fase do áudio gerado.
#[test]
fn test_wavenet_process_deterministic() {
    // Criamos duas instâncias isoladas e idênticas para garantir que o estado interno
    // não é compartilhado de forma insegura (vazamento de memória ou threads).
    let mut model_a = build_tiny_wavenet();
    let mut model_b = build_tiny_wavenet();

    model_a.prewarm();
    model_b.prewarm();

    // Aplicamos um estímulo constante de 0.1 (impulso DC).
    let input = [0.1f32; 8];
    let mut out_a = [0.0f32; 8];
    let mut out_b = [0.0f32; 8];

    // Processamos o mesmo sinal em ambos os modelos.
    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    // A arquitetura NAM deve ser determinística: o mesmo input deve gerar o mesmo output
    // bit-a-bit ou dentro da tolerância de arredondamento de hardware (1e-6).
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

/// Verifica se uma Convolução 1D atuando como "Kernel Identidade"
/// reproduz exatamente a entrada na saída, sem qualquer modificação.
///
/// A importância pedagógica disto é enorme: usamos isso para garantir que
/// as primitivas SIMD de mais baixo nível não estão transpondo, corrompendo
/// ou descartando canais ao acessar e armazenar valores na memória (SoA/AoS).
#[test]
fn test_conv1d_identity_kernel() {
    // Criamos uma matriz de pesos 4x4 (achatada para 16 floats).
    // Ao preencher apenas a diagonal principal com 1.0, criamos um "Kernel Identidade".
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 16]); // OUT=4 * K=1 * IN=4
    for i in 0..4 {
        weights[i * 4 + i] = half::f16::from_f32(1.0).to_bits();
    }

    // Instanciamos a Convolução 1D sem bias e com dilatação 1 (processamento linear).
    let conv = Conv1d::<4, 4, 1> {
        weights,
        bias: AlignedVec::from_vec(vec![0.0; 4]),
        do_bias: false,
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Simulamos um buffer de camada (input) com valores sequenciais.
    let layer_buffer = vec![1.0, 2.0, 3.0, 4.0];
    let mut block = vec![0.0; 4];

    // Invocamos manualmente a rotina SIMD otimizada para AVX2.
    // O uso do bloco 'unsafe' é necessário pois acessamos primitivas intrínsecas de hardware.
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 0, 1);
    }

    // Como o kernel é identidade, a saída deve ser uma cópia bit-perfect da entrada.
    assert_eq!(block, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Verifica a funcionalidade de adição de *Bias* (viés) na Convolução 1D.
///
/// Uma camada equipada com pesos de matriz Identidade somada a um viés constante
/// deve apenas transladar o eixo numérico dos valores processados. Isso atesta
/// o uso correto de FMA (Fused Multiply-Add) com a flag `do_bias`.
#[test]
fn test_conv1d_with_bias() {
    // Novamente, usamos um kernel identidade para isolar o efeito do Bias.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 16]);
    for i in 0..4 {
        weights[i * 4 + i] = half::f16::from_f32(1.0).to_bits();
    }

    // Configuramos a camada com Bias de 0.5 em todos os canais de saída.
    let conv = Conv1d::<4, 4, 1> {
        weights,
        bias: AlignedVec::from_vec(vec![0.5; 4]),
        do_bias: true, // Habilita a adição do vetor de bias.
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    let layer_buffer = vec![1.0, 2.0, 3.0, 4.0];
    let mut block = vec![0.0; 4];

    // O motor SIMD executa Fused Multiply-Add (FMA): (input * 1.0) + 0.5.
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 0, 1);
    }

    // Verificamos se cada elemento foi transladado corretamente pelo valor do bias.
    assert_eq!(block, vec![1.5, 2.5, 3.5, 4.5]);
}

/// Testa o conceito central do WaveNet: Convoluções Dilatadas.
///
/// A dilatação insere espaçamentos determinísticos na amostragem histórica de dados.
/// Aqui validamos se o deslocamento matemático na `Conv1d` acessa corretamente os
/// frames defasados (com `dilation: 2`). Isso significa ignorar amostras adjacentes
/// e pular janelas dentro do Ring Buffer. O sucesso deste teste garante a
/// construção correta do *Receptive Field* geral da arquitetura.
#[test]
fn test_conv1d_dilation() {
    // Configuramos pesos unitários (1.0) para somar todos os inputs diretamente.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 2 * 3 * 2]); // OUT=2 * K=3 * IN=2
    for w in weights.iter_mut().take(12) {
        *w = half::f16::from_f32(1.0).to_bits();
    }

    // Definimos dilation: 2. Isso fará o kernel "saltar" um frame a cada tap.
    let conv = Conv1d::<2, 2, 3> {
        weights,
        bias: AlignedVec::from_vec(vec![0.0; 2]),
        do_bias: false,
        dilation: 2,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Criamos um histórico de 6 frames (12 floats).
    // Organização: [F0, F1, F2, F3, F4, F5] -> [ (1,2), (10,20), (3,4), (30,40), (5,6), (0,0) ]
    let mut layer_buffer = vec![0.0; 6 * 2];
    layer_buffer[0] = 1.0;
    layer_buffer[1] = 2.0; // F0
    layer_buffer[2] = 10.0;
    layer_buffer[3] = 20.0; // F1 (Será ignorado pela dilatação 2)
    layer_buffer[4] = 3.0;
    layer_buffer[5] = 4.0; // F2
    layer_buffer[6] = 30.0;
    layer_buffer[7] = 40.0; // F3 (Será ignorado pela dilatação 2)
    layer_buffer[8] = 5.0;
    layer_buffer[9] = 6.0; // F4

    let mut block = vec![0.0; 2];

    // Processamos no frame de índice 4 (F4).
    // Com K=3 e Dilation=2, os taps serão:
    // Tap 0: frame 4 - (2 * 2) = frame 0 -> [1.0, 2.0]
    // Tap 1: frame 4 - (2 * 1) = frame 2 -> [3.0, 4.0]
    // Tap 2: frame 4 - (2 * 0) = frame 4 -> [5.0, 6.0]
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 4, 1);
    }

    // Soma total esperada: 1+2 + 3+4 + 5+6 = 21.0.
    assert_eq!(block[0], 21.0);
    assert_eq!(block[1], 21.0);
}

/// Garante que pesos gigantes (100.0) multiplicados por entradas de silêncio (0.0)
/// se manterão absolutamente no zero, provando que o laço de multiplicação não
/// introduz ruído digital (*DC offset* espontâneo).
/// Em seguida, ativa o *bias* no meio da execução para provar que a alternância
/// no tempo de execução é respeitada pela primitiva DSP.
#[test]
fn test_conv1d_zero_input() {
    // Pesos extremamente altos para testar se qualquer ruído residual é amplificado.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 2 * 3 * 2]);
    for w in weights.iter_mut().take(12) {
        *w = half::f16::from_f32(100.0).to_bits();
    }

    let mut conv = Conv1d::<2, 2, 3> {
        weights: weights.clone(),
        bias: AlignedVec::from_vec(vec![0.0; 2]),
        do_bias: false,
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Buffer preenchido com zeros.
    let layer_buffer = vec![0.0; 4 * 2];
    let mut block = vec![0.0; 2];

    // Primeira passagem: Sem bias, saída deve ser 0.0 absoluto.
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 2, 1);
    }

    assert_eq!(block, vec![0.0, 0.0]);

    // Segunda passagem: Ativamos o bias. A saída deve refletir exatamente o bias injetado.
    conv.do_bias = true;
    conv.bias = AlignedVec::from_vec(vec![7.5, 8.5]);

    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 2, 1);
    }

    assert_eq!(block, vec![7.5, 8.5]);
}

/// Teste manual de cruzamento de matrizes (Dot Product)
/// contra um resultado esperado e pré-calculado por um cientista.
///
/// Provê garantia absoluta de que a rotina `process_block` otimizada para
/// hardware x86-64 lida perfeitamente com somatórias que contêm valores e pesos
/// positivos e negativos concorrentes, validando a correção dos cálculos matemáticos.
#[test]
fn test_conv1d_known_output() {
    // Matriz de pesos heterogênea para validar o cruzamento de canais (Dot Product).
    // Estrutura: OUT=2, K=2, IN=2. Total 8 pesos.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 2 * 2 * 2]);
    weights[0] = half::f16::from_f32(0.5).to_bits();
    weights[1] = half::f16::from_f32(1.0).to_bits(); // out0, k0
    weights[2] = half::f16::from_f32(1.5).to_bits();
    weights[3] = half::f16::from_f32(2.0).to_bits(); // out0, k1
    weights[4] = half::f16::from_f32(-0.5).to_bits();
    weights[5] = half::f16::from_f32(-1.0).to_bits(); // out1, k0
    weights[6] = half::f16::from_f32(-1.5).to_bits();
    weights[7] = half::f16::from_f32(-2.0).to_bits(); // out1, k1

    let conv = Conv1d::<2, 2, 2> {
        weights,
        bias: AlignedVec::from_vec(vec![1.0, -1.0]),
        do_bias: true,
        dilation: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    };

    // Layer buffer com 2 frames: F0=(2.0, 3.0), F1=(4.0, 5.0).
    let layer_buffer = vec![2.0, 3.0, 4.0, 5.0];
    let mut block = vec![0.0; 2];

    // Processamos no frame de índice 1 (F1).
    // out0 = bias[0] + dot(F0, w[out0,k0]) + dot(F1, w[out0,k1])
    //      = 1.0 + (2*0.5 + 3*1.0) + (4*1.5 + 5*2.0) = 1.0 + 4.0 + 16.0 = 21.0
    // out1 = bias[1] + dot(F0, w[out1,k0]) + dot(F1, w[out1,k1])
    //      = -1.0 + (2*-0.5 + 3*-1.0) + (4*-1.5 + 5*-2.0) = -1.0 - 4.0 - 16.0 = -21.0
    unsafe {
        conv.process_block::<crate::math::common::Avx2Math>(&layer_buffer, &mut block, 1, 1);
    }

    assert_eq!(block[0], 21.0);
    assert_eq!(block[1], -21.0);
}

/// Verifica a funcionalidade "Identidade" básica de uma camada Densa
/// (Totalmente Conectada). Isso é usado em demasia nas conexões *1x1* e
/// agregações de saída de canal do WaveNet (*Skip Connections*).
#[test]
fn test_dense_layer_identity() {
    // Matriz de pesos 4x4 Identity.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 16]); // OUT=4 * IN=4
    for out_c in 0..4 {
        weights[out_c * 4 + out_c] = half::f16::from_f32(1.0).to_bits();
    }

    let dense = DenseLayer::<4, 4> {
        weights,
        bias: AlignedVec::from_vec(vec![0.0; 4]),
        do_bias: false,
    };

    let input = vec![1.5, 2.5, 3.5, 4.5];
    let mut output = vec![0.0; 4];

    // Camadas densas 1x1 são fundamentais para misturar canais sem olhar para o tempo.
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    assert_eq!(output, vec![1.5, 2.5, 3.5, 4.5]);
}

/// Verifica a correta injeção de tensores de *Bias* (viés) em Camadas Densas
/// através de ponteiros SIMD, garantindo a alteração linear do Output final.
#[test]
fn test_dense_layer_with_bias() {
    // Pesos identidade + Bias de 1.0.
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 16]);
    for out_c in 0..4 {
        weights[out_c * 4 + out_c] = half::f16::from_f32(1.0).to_bits();
    }

    let dense = DenseLayer::<4, 4> {
        weights,
        bias: AlignedVec::from_vec(vec![1.0; 4]),
        do_bias: true,
    };

    let input = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![0.0; 4];

    // O resultado deve ser transladado pelo bias em todos os 4 canais.
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0]);
}

/// Executa uma Camada Densa com dimensionalidade não-quadrada (IN=8, OUT=4).
///
/// Como o WaveNet altera constantemente as matrizes (de CH para HEAD e vice-versa),
/// os motores SIMD jamais devem presumir matrizes perfeitamente simétricas (NxN).
/// Este teste injeta valores heterogêneos para garantir que as paradas
/// dos laços aninhados de FMA calculem tudo até o limite exato de alocação.
#[test]
fn test_dense_layer_rectangular() {
    // Matriz Assimétrica: IN=8, OUT=4.
    // Em modelos reais, isso acontece ao projetar CH (ex: 16) para HEAD (ex: 8).
    let mut weights = AlignedVec::from_vec(vec![half::f16::from_f32(0.0).to_bits(); 32]); // 4 * 8
    // out_c = 0: Soma ponderada de in[0] e in[1]
    // [IN][OUT] -> in_c * OUT + out_c
    weights[0] = half::f16::from_f32(1.0).to_bits(); // in0, out0
    weights[4] = half::f16::from_f32(2.0).to_bits(); // in1, out0

    // out_c = 1: Soma ponderada de in[2] e in[3]
    weights[9] = half::f16::from_f32(3.0).to_bits(); // in2, out1
    weights[13] = half::f16::from_f32(4.0).to_bits(); // in3, out1

    // out_c = 2: Escala simples de in[4]
    weights[18] = half::f16::from_f32(0.5).to_bits(); // in4, out2

    // out_c = 3: Inversão de fase de in[7]
    weights[31] = half::f16::from_f32(-1.0).to_bits(); // in7, out3

    let dense = DenseLayer::<8, 4> {
        weights,
        bias: AlignedVec::from_vec(vec![0.5, -0.5, 1.0, -1.0]),
        do_bias: true,
    };

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = vec![0.0; 4];

    // Validamos se o laço SIMD trata corretamente o fim da linha da matriz (stride).
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut output, 1);
    }

    // Trace do cálculo manual:
    // out[0] = (1.0 * 1.0) + (2.0 * 2.0) + 0.5 = 1.0 + 4.0 + 0.5 = 5.5
    // out[1] = (3.0 * 3.0) + (4.0 * 4.0) - 0.5 = 9.0 + 16.0 - 0.5 = 24.5
    // out[2] = (5.0 * 0.5) + 1.0 = 2.5 + 1.0 = 3.5
    // out[3] = (8.0 * -1.0) - 1.0 = -8.0 - 1.0 = -9.0

    assert_eq!(output[0], 5.5);
    assert_eq!(output[1], 24.5);
    assert_eq!(output[2], 3.5);
    assert_eq!(output[3], -9.0);
}

use crate::models::wavenet::common::WavenetProcessContext;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
use crate::models::wavenet::model_dyn::*;

/// Constrói um `Conv1dDyn` mínimo com `kernel=1`, `dilation=1`.
///
/// - `in_ch`: canais de entrada
/// - `out_ch`: canais de saída (2×ch quando gated)
/// - `weight`: valor fixo para todos os pesos (facilita cálculo analítico)
fn make_conv1d(in_ch: usize, out_ch: usize, weight: f32) -> Conv1dDyn {
    Conv1dDyn {
        weights: AlignedVec::from_vec(vec![
            half::f16::from_f32(weight).to_bits();
            (out_ch * in_ch + 7) & !7
        ]), // kernel=1, padded to 8
        bias: AlignedVec::from_vec(vec![0.0; out_ch]),
        do_bias: false,
        dilation: 1,
        in_ch,
        out_ch,
        kernel: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
    }
}

/// Constrói um `DenseLayerDyn` identidade (peso=0, bias=0, sem efeito).
fn make_dense_zero(in_size: usize, out_size: usize) -> DenseLayerDyn {
    DenseLayerDyn {
        weights: AlignedVec::new(out_size * in_size, 0u16),
        bias: AlignedVec::new(out_size, 0.0),
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
    let condition = AlignedVec::new(ch, 0.0);
    let mut head_input = AlignedVec::new(ch, 0.0);
    let mut output = AlignedVec::new(ch, 0.0);
    let mut block = AlignedVec::new(2 * ch * WAVENET_MAX_NUM_FRAMES, 0.0);

    let _math = SimdMathConfig::current();

    // Executamos o processamento interno (unsafe pois lida com ponteiros/SIMD em produção).
    unsafe {
        layer.process_block_internal::<crate::math::common::Avx2Math>(WavenetProcessContext {
            condition: &condition,
            condition_bf16: &[0u16],
            head_input: &mut head_input,
            output: &mut output,
            output_bf16: None,
            layer_buffer: &layer_buffer,
            layer_buffer_bf16: &[],
            buffer_start,
            block: &mut block,
            num_frames: 1,
        });
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

    let condition = AlignedVec::new(ch, 0.0);
    let mut head_input = AlignedVec::new(ch, 0.0);
    let mut output = AlignedVec::new(ch, 0.0);
    let mut block = AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0);

    let _math = SimdMathConfig::current();

    unsafe {
        layer.process_block_internal::<crate::math::common::Avx2Math>(WavenetProcessContext {
            condition: &condition,
            condition_bf16: &[0u16],
            head_input: &mut head_input,
            output: &mut output,
            output_bf16: None,
            layer_buffer: &layer_buffer,
            layer_buffer_bf16: &[],
            buffer_start,
            block: &mut block,
            num_frames: 1,
        });
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
    let state = WaveNetLayerState::new(ch, 0, 0).expect("Failed to create WaveNetLayerState");

    let conv1d = Conv1dDyn {
        weights: AlignedVec::from_vec(vec![0u16; 2 * ch * ch]),
        bias: AlignedVec::from_vec(vec![0.0; 2 * ch]),
        do_bias: false,
        dilation: 1,
        in_ch: ch,
        out_ch: 2 * ch,
        kernel: 1,
        prefetch_fn: crate::math::common::prefetch_strategy_simple,
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
        array_outputs: AlignedVec::from_vec(vec![0.0; ch]),
        head_accum: AlignedVec::from_vec(vec![0.0; ch]),
        head_outputs: AlignedVec::from_vec(vec![0.0; 1]),
        block_buffer: AlignedVec::from_vec(vec![0.0; block_size]),
        block_size,
        receptive_field_size: 0,
        ch,
        head: 1,
        last_condition: AlignedVec::from_vec(vec![0.0; 1]),
        last_condition_bf16: AlignedVec::from_vec(vec![0u16; 1]),
        condition_init: false,
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
