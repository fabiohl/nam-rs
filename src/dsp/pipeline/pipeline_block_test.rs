// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod block_tests {
    use super::super::test_util::infra::{ALLOC_COUNT, TrackingGuard};
    use super::super::*;
    use crate::common::spsc::RtStatusFlags;
    use crate::dsp::gate::{DynamicHysteresis, GateParams};
    use crate::dsp::resampler::NamResampler;
    use crate::loader::dispatcher::build_model;
    use crate::loader::nam_json::parse_nam_json;
    use crate::models::DynamicModel;
    use proptest::prelude::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::Ordering;

    /// Helper para resolver o caminho dos modelos de teste.
    fn get_test_model_path(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/fixtures/models");
        path.push(name);
        path
    }

    /// Helper para carregar um modelo NAM para testes.
    fn load_test_model(name: &str) -> Box<DynamicModel> {
        let path = get_test_model_path(name);
        let json_data = fs::read_to_string(path).expect("Falha ao ler arquivo do modelo");
        let model_data = parse_nam_json(&json_data).expect("Falha ao processar JSON do modelo");
        build_model(&model_data).expect("Falha ao construir modelo")
    }

    /// Executor principal do teste de pipeline com block size variável.
    fn run_block_size_test(model_name: Option<&str>, block_size: usize) {
        // Se o tamanho do bloco for 0, não faz sentido testar.
        if block_size == 0 {
            return;
        }
        // Se exceder o limite do bridge, limitamos para o teste não dar panic por overflow de buffer fixo.
        let n = block_size.min(MAX_BRIDGE_BUF);

        // Carrega o modelo de simulação de amplificador (LSTM ou WaveNet) se um nome foi fornecido.
        let mut model = model_name.map(load_test_model);
        if let Some(ref mut m) = model {
            // O "prewarm" prepara o estado interno do modelo para processar áudio imediatamente,
            // evitando estalos ou silêncio nas primeiras amostras.
            m.prewarm(2048);
        }

        // Inicializa o reamostrador (Resampler). Aqui usamos 48kHz -> 48kHz (bypass)
        // apenas para testar a infraestrutura de bufferização do resampler com tamanhos de bloco estranhos.
        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();

        // Flags de status em tempo real (indicam se houve clipping ou outros problemas).
        let rt_status = RtStatusFlags::default();

        // O DspBridge é a nossa "ponte" de memória. Ele armazena o áudio processado para ser
        // lido por outra thread (como a interface gráfica ou o gravador).
        // Usamos Box para garantir que ele tenha um endereço de memória fixo (heap).
        let mut bridge = Box::new(DspBridge {
            // Criamos dois buffers para técnica de "Double Buffering" (evita que quem lê atrapalhe quem escreve).
            buffers: [
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
                BridgeBuffer {
                    buf_l: [0.0; MAX_BRIDGE_BUF],
                    buf_r: [0.0; MAX_BRIDGE_BUF],
                    n_samples: 0,
                },
            ],
            // Contadores atômicos para sincronização segura entre threads sem usar travas (locks).
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // Alocamos buffers intermediários necessários para as etapas de processamento.
        // Vec::with_capacity ou vec![...] no início de um teste é aceitável, pois não estamos no "hot-path" ainda.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // Configuração do Noise Gate (supressor de ruído).
        let gate_params = GateParams::default();
        // Histerese controla a suavidade da abertura e fechamento do som para evitar "pipocos".
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        // Criamos amostras de teste (um sinal constante de 0.1) para processar.
        let mut samples_l = vec![0.1; n];
        let mut samples_r = vec![0.1; n];

        // O Contexto (ctx) agrupa todas as ferramentas que o pipeline precisa para trabalhar.
        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut model,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0,
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            // BridgeRef é um ponteiro seguro para a ponte de áudio.
            bridge_ptr: unsafe { BridgeRef::new(&mut *bridge as *mut DspBridge) },
        };

        let bufs = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();

        // Executamos o pipeline principal que orquestra todo o DSP do NAM-rs.
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs);

        // Verificamos quantas alocações ocorreram durante o processamento.
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        // Removemos a vigia.
        drop(_guard);

        assert_eq!(allocs, 0);

        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[read_idx];
        assert_eq!(out_buf.n_samples as usize, n);

        // Verificação de sanidade matemática: o áudio não pode "explodir" (virar NaN ou Infinito).
        for i in 0..n {
            assert!(out_buf.buf_l[i].is_finite());
            assert!(out_buf.buf_r[i].is_finite());
        }
    }

    /// TESTE: Tamanhos de bloco não-convencionais para modelos LSTM.
    /// Hosts como Bitwig Studio podem enviar blocos de qualquer tamanho (ex: 7 ou 17 samples).
    #[test]
    fn test_unconventional_block_sizes_lstm() {
        let sizes = [1, 3, 7, 8, 9, 17, 33, 53, 64, 128, 256, 512];
        for &size in &sizes {
            run_block_size_test(Some("BossLSTM-1x16.nam"), size);
        }
    }

    /// TESTE: Tamanhos de bloco não-convencionais para modelos WaveNet.
    #[test]
    fn test_unconventional_block_sizes_wavenet() {
        let sizes = [1, 3, 7, 8, 9, 17, 33, 53, 64, 128, 256, 512];
        for &size in &sizes {
            run_block_size_test(Some("BossWN-nano.nam"), size);
        }
    }

    /// TESTE: Casos de borda (extremos).
    #[test]
    fn test_zero_alloc_edge_cases() {
        // n_samples = 1 (mínimo possível)
        run_block_size_test(Some("BossWN-nano.nam"), 1);
        // n_samples = MAX_BRIDGE_BUF (máximo suportado pelo nosso buffer interno)
        run_block_size_test(Some("BossWN-nano.nam"), MAX_BRIDGE_BUF);
    }

    // Property-Based Testing (Proptest):
    // Em vez de escolhermos os números, deixamos o computador gerar 500 tamanhos
    // aleatórios entre 1 e 8192 para tentar quebrar o nosso código.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn test_random_block_sizes_proptest(size in 1..8192usize) {
            run_block_size_test(Some("BossWN-nano.nam"), size);
        }
    }
}
