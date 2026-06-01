// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
mod tests {
    use super::super::test_util::infra::{ALLOC_COUNT, TrackingGuard};
    use super::super::*;
    use crate::common::spsc::RtStatusFlags;
    use crate::dsp::gate::{DynamicHysteresis, GateParams};
    use crate::dsp::resampler::NamResampler;

    use std::sync::atomic::Ordering;

    /// Função de ajuda que simula um laboratório para testar o motor de áudio (pipeline).
    /// Ela configura tudo o que é necessário para ver se o som entra e sai corretamente.
    fn run_pipeline_test(
        pw_rate: u32,
        nam_rate: u32,
        input_l: &[f32],
        input_r: &[f32],
        force_hold_zero: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = input_l.len();
        // Prepara o conversor de taxa de amostragem (ex: transformar 44100 em 48000 Hz).
        let mut resampler = NamResampler::new(pw_rate, nam_rate, n).unwrap();
        let rt_status = RtStatusFlags::default();

        // Cria a "ponte" (bridge) que armazena os sons processados para podermos ler depois.
        let mut bridge = Box::new(DspBridge {
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // Prepara "gavetas" temporárias para guardar o som enquanto ele está sendo processado.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // Configura as opções do redutor de ruído (Noise Gate) para o teste.
        let mut gate_params = GateParams::default();
        if force_hold_zero {
            gate_params.hold_frames = 0;
            gate_params.mono_epsilon = 1.0; // Facilita a detecção de som mono (igual nos dois lados).
        }
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.to_vec();
        let mut samples_r = input_r.to_vec();

        // Junta todas as configurações em um único "Manual de Instruções" (Contexto).
        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0, // Mantém o som sempre passando.
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
        };

        let bufs = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        // LIGA O VIGIA DE MEMÓRIA.
        let _guard = TrackingGuard::new();
        // EXECUTA O PROCESSAMENTO DE SOM REAL.
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs);
        // Verifica se o vigia pegou algum pedido de memória proibido.
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // Se o programa "parou para pensar" (pediu memória) enquanto tocava som, o teste falha.
        assert_eq!(
            allocs, 0,
            "Alocação detectada no hot-path! O sistema não pode pedir memória enquanto processa áudio."
        );

        // Recupera o som final que foi parar na nossa "ponte" para conferir o resultado.
        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[read_idx];
        let n_out = out_buf.n_samples as usize;

        (
            out_buf.buf_l[..n_out].to_vec(),
            out_buf.buf_r[..n_out].to_vec(),
        )
    }

    /// TESTE: Som Direto (Bypass) em Estéreo.
    /// Garante que, se não houver efeitos ligados, o som que entra é exatamente igual ao que sai.
    #[test]
    fn test_bypass_no_resampler_stereo() {
        let n = 64;
        // Cria um som de teste para os lados esquerdo (L) e direito (R).
        let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let input_r: Vec<f32> = (0..n).map(|i| (i as f32 + 50.0) * 0.01).collect();

        // Roda o laboratório com a mesma taxa de entrada e saída (sem conversão de Hz).
        let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, false);

        assert_eq!(out_l.len(), n);
        assert_eq!(out_r.len(), n);
        // Verifica se os sons são idênticos, amostra por amostra.
        assert_eq!(out_l, input_l, "O canal L deve ser idêntico ao original");
        assert_eq!(out_r, input_r, "O canal R deve ser idêntico ao original");
    }

    /// TESTE: Som Direto (Bypass) em Mono.
    /// Verifica se o sistema identifica sons iguais e mantém a saída sincronizada.
    #[test]
    fn test_bypass_no_resampler_mono() {
        let n = 64;
        let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        // Para o teste mono, fazemos o lado direito ser uma cópia exata do esquerdo.
        let input_r = input_l.clone();

        let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, true);

        assert_eq!(out_l.len(), n);
        assert_eq!(out_r.len(), n);
        assert_eq!(out_l, input_l);
        // No modo mono, o lado direito (R) deve sair exatamente igual ao esquerdo (L).
        assert_eq!(
            out_r, input_l,
            "No modo mono, o lado R deve ser uma cópia do L"
        );
    }

    /// TESTE: Som Direto com Mudança de Qualidade (Resampling).
    /// Testa se o som continua passando corretamente mesmo quando a "velocidade" (taxa de Hz) muda.
    #[test]
    fn test_bypass_with_resampler_stereo() {
        // Exemplo: Som entra a 44100Hz (CD) e sai a 48000Hz (Vídeo).
        let n = 256;
        let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let input_r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1 + 0.5).sin()).collect();

        let (out_l, _out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, false);

        assert!(!out_l.is_empty());

        // Calcula a "força" (energia) do som original.
        let mut energy_in = 0.0;
        for &x in &input_l {
            energy_in += x * x;
        }

        // Calcula a "força" do som que saiu após a conversão.
        let mut energy_out = 0.0;
        for &x in &out_l {
            energy_out += x * x;
        }

        // O som não precisa ser idêntico (por causa da conversão matemática),
        // mas deve manter um volume similar e não ficar em silêncio.
        assert!(
            energy_out > energy_in * 0.5,
            "O som de saída está muito fraco ou em silêncio após a conversão"
        );
    }

    /// TESTE: Mono com Mudança de Qualidade (Resampling).
    /// Garante que, mesmo após converter a taxa de Hz, os dois canais continuam idênticos.
    #[test]
    fn test_bypass_with_resampler_mono() {
        let n = 256;
        let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let input_r = input_l.clone();

        let (out_l, out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, true);

        assert!(!out_l.is_empty());
        assert_eq!(
            out_l, out_r,
            "Mesmo com conversão de Hz, o som mono deve ser igual nos dois lados (L == R)"
        );
    }

    /// TESTE: Economia de Processamento (Gate Fechado e Silêncio).
    /// Se não há som e o portão está fechado, o sistema não deve perder tempo processando nada.
    #[test]
    fn test_hotpath_gate_closed_and_silence() {
        let n = 64; // Tamanho do bloco de amostras (64 "pedacinhos" de som).
        let input_l = vec![0.0; n]; // Entrada silenciosa no canal esquerdo.
        let input_r = vec![0.0; n]; // Entrada silenciosa no canal direito.

        // Preparamos as ferramentas de áudio (reamostrador e ponte de dados).
        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
        let rt_status = RtStatusFlags::default();
        let mut bridge = Box::new(DspBridge {
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // Buffers de trabalho temporários para os cálculos de DSP.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // Configuração do portão de ruído (Gate).
        let gate_params = GateParams::new(-70.0, -80.0, 0, 0, 1e-4);
        let mut silence_hysteresis = DynamicHysteresis::new();
        // Força o fechamento do portão manualmente para o teste.
        // Simulamos que o som está muito baixo (0.0) para que o portão se feche.
        silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
        silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.clone();
        let mut samples_r = input_r.clone();

        // Agrupamos tudo no "Contexto" para o processamento.
        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.1,
            threshold_close_sq: 0.01,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
        };

        let bufs = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        // Vigia de alocação de memória.
        let _guard = TrackingGuard::new();
        // Executamos a orquestra de áudio (Pipeline).
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // Verificação: Alocar memória no meio do áudio é proibido (causa estalos).
        assert_eq!(allocs, 0, "Alocação no caminho crítico!");

        // Verificação: O sistema deve marcar que o estado atual é de silêncio.
        assert!(
            rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT),
            "O portão deveria estar fechado (silêncio)"
        );
        assert!(!rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING));

        // Verificação final: Como o portão está fechado, não deve ter sido enviada nenhuma amostra para a ponte.
        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[1 - read_idx];
        assert_eq!(
            out_buf.n_samples, 0,
            "Não deve haver amostras processadas quando o portão está em silêncio absoluto"
        );
    }

    /// TESTE: Transição Suave (FadeOut).
    /// Verifica se o sistema detecta corretamente quando o volume está diminuindo aos poucos (esmaecendo).
    #[test]
    fn test_hotpath_gate_fading() {
        let n = 64;
        let mut input_l = vec![0.0; n];
        let mut input_r = vec![0.0; n];
        // Simulamos um sinal muito fraco, que deve ativar o fechamento suave do som.
        for i in 0..n {
            input_l[i] = 0.05; // Este valor está entre os limites de "abrir" e "fechar".
            input_r[i] = 0.05;
        }

        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
        let rt_status = RtStatusFlags::default();
        let mut bridge = Box::new(DspBridge {
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        // Configuramos o FadeOut para durar 100 quadros de som.
        let gate_params = GateParams::new(-70.0, -80.0, 0, 100, 1e-4);
        let mut silence_hysteresis = DynamicHysteresis::new();
        // Primeiro abrimos o som (1.0) e depois injetamos silêncio para iniciar o FadeOut.
        silence_hysteresis.update(1.0, 0.1, 0.0001, &gate_params, 100);

        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = vec![0.0; n];
        let mut samples_r = vec![0.0; n];

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.1,
            threshold_close_sq: 0.01,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
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
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // Verificações: Sem alocação e deve indicar o estado de "FADING".
        assert_eq!(allocs, 0);
        assert!(
            rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_FADING),
            "O sistema deveria indicar que está no meio de um fechamento suave (FadeOut)"
        );
        assert!(!rt_status.check_flag(crate::common::spsc::RT_STATUS_IS_SILENT));
    }

    /// TESTE: Detecção de Distorção (Clipping).
    /// Verifica se o sistema avisa quando o volume ultrapassa o limite digital (1.0), o que causa ruído indesejado.
    #[test]
    fn test_hotpath_clipping_detection() {
        let n = 64;
        let mut input_l = vec![0.0; n];
        let input_r = vec![0.0; n];
        // Forçamos um volume impossível (1.5) em uma amostra específica para causar distorção.
        input_l[10] = 1.5;

        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
        let rt_status = RtStatusFlags::default();
        let mut bridge = Box::new(DspBridge {
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        // Buffers de trabalho temporários para os cálculos de DSP.
        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];

        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.clone();
        let mut samples_r = input_r.clone();

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
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
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
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
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        // Verificação: Deve detectar a flag de Clipping (RT_STATUS_HAS_CLIPPED).
        assert_eq!(allocs, 0);
        assert!(
            rt_status.check_flag(crate::common::spsc::RT_STATUS_HAS_CLIPPED),
            "O sistema deveria ter detectado que o som ultrapassou o limite (clipping)"
        );
    }

    /// TESTE: Detecção de Perda de Som (Dropped Frames).
    /// Verifica se o sistema percebe quando o computador está muito lento e começa a perder pacotes de áudio
    /// porque quem deveria processar o som não está dando conta de ler a tempo.
    #[test]
    fn test_hotpath_dropped_frames() {
        let n = 64;
        let mut resampler = NamResampler::new(48000, 48000, n).unwrap();
        let rt_status = RtStatusFlags::default();
        let mut bridge = Box::new(DspBridge {
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
            active_read_idx: std::sync::atomic::AtomicUsize::new(0),
            generation: std::sync::atomic::AtomicU64::new(0),
            // Simula que quem deveria "ouvir" o som (consumidor) não leu nada ainda.
            consumed_gen: std::sync::atomic::AtomicU64::new(0),
            dropped_frames: std::sync::atomic::AtomicU32::new(0),
        });

        let mut resamp_mid_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_mid_r = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_l = vec![0.0; MAX_RESAMP_BUF];
        let mut resamp_out_r = [0.0; MAX_RESAMP_BUF];
        let mut model_out_l = [0.0; MAX_RESAMP_BUF];
        let mut model_out_r = [0.0; MAX_RESAMP_BUF];
        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();

        // 1ª passada: O pipeline processa o som e guarda na "ponte".
        let mut process_mono = false;
        let mut samples_l = vec![1.0; n];
        let mut samples_r = vec![1.0; n];

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
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
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
        };

        let bufs = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx, bufs);

        // 2ª passada: O sistema tenta processar mais som, mas vê que a ponte ainda está ocupada
        // com o som da passada anterior que ninguém leu.
        let mut process_mono2 = false;
        let mut samples_l2 = vec![1.0; n];
        let mut samples_r2 = vec![1.0; n];
        let ctx2 = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0,
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono2,
            rt_status: &rt_status,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
        };

        let bufs2 = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();
        // Aqui o sistema deve ser obrigado a descartar este novo pacote de som.
        capture_dsp_pipeline(&mut samples_l2, &mut samples_r2, n, ctx2, bufs2);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        assert_eq!(allocs, 0);

        // O sistema deve ter incrementado o contador de pacotes perdidos (dropped).
        let dropped = bridge.dropped_frames.load(Ordering::Relaxed);
        assert_eq!(
            dropped, 1,
            "Deveria ter detectado 1 pacote de áudio perdido (dropado)"
        );
    }
}
