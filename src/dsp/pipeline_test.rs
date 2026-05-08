// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::dsp::gate::{DynamicHysteresis, GateParams};
    use crate::dsp::resampler::NamResampler;
    use crate::spsc::RtStatusFlags;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

    // =============================================================================
    // Counting Allocator para Verificação Zero-Allocation
    // =============================================================================
    static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
    static TRACKING_THREAD: AtomicI32 = AtomicI32::new(0);

    struct CountingAllocator;

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            if tid == TRACKING_THREAD.load(Ordering::Relaxed) {
                ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAllocator = CountingAllocator;

    struct TrackingGuard;
    impl TrackingGuard {
        fn new() -> Self {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            TRACKING_THREAD.store(tid, Ordering::Relaxed);
            ALLOC_COUNT.store(0, Ordering::Relaxed);
            Self
        }
    }
    impl Drop for TrackingGuard {
        fn drop(&mut self) {
            TRACKING_THREAD.store(0, Ordering::Relaxed);
        }
    }

    /// Helper para configurar o ambiente de teste do pipeline.
    fn run_pipeline_test(
        pw_rate: u32,
        nam_rate: u32,
        input_l: &[f32],
        input_r: &[f32],
        force_hold_zero: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = input_l.len();
        let mut resampler = NamResampler::new(pw_rate, nam_rate, n).unwrap();
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

        let mut gate_params = GateParams::default();
        if force_hold_zero {
            gate_params.hold_frames = 0;
            gate_params.mono_epsilon = 1.0; // Facilita detecção de mono
        }
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        let mut samples_l = input_l.to_vec();
        let mut samples_r = input_r.to_vec();

        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0, // Sempre aberto
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            bridge_ptr: &mut *bridge as *mut DspBridge,
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        assert_eq!(
            allocs, 0,
            "Alocação detectada no hot-path (capture_dsp_pipeline)!"
        );

        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        let out_buf = &bridge.buffers[read_idx];
        let n_out = out_buf.n_samples as usize;

        (
            out_buf.buf_l[..n_out].to_vec(),
            out_buf.buf_r[..n_out].to_vec(),
        )
    }

    #[test]
    fn test_bypass_no_resampler_stereo() {
        let n = 64;
        let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let input_r: Vec<f32> = (0..n).map(|i| (i as f32 + 50.0) * 0.01).collect();

        let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, false);

        assert_eq!(out_l.len(), n);
        assert_eq!(out_r.len(), n);
        assert_eq!(out_l, input_l, "Canal L deve ser bit-exact em bypass");
        assert_eq!(out_r, input_r, "Canal R deve ser bit-exact em bypass");
    }

    #[test]
    fn test_bypass_no_resampler_mono() {
        let n = 64;
        let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        // Para detectar mono, os canais devem ser muito próximos
        let input_r = input_l.clone();

        let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, true);

        assert_eq!(out_l.len(), n);
        assert_eq!(out_r.len(), n);
        assert_eq!(out_l, input_l);
        assert_eq!(out_r, input_l, "No modo mono, R deve ser cópia do L");
    }

    #[test]
    fn test_bypass_with_resampler_stereo() {
        // 44100 -> 48000 -> 44100
        let n = 256;
        let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let input_r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1 + 0.5).sin()).collect();

        let (out_l, _out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, false);

        assert!(!out_l.is_empty());

        let mut energy_in = 0.0;
        for &x in &input_l {
            energy_in += x * x;
        }

        let mut energy_out = 0.0;
        for &x in &out_l {
            energy_out += x * x;
        }

        assert!(
            energy_out > energy_in * 0.5,
            "Sinal de saída muito fraco ou silêncio"
        );
    }

    #[test]
    fn test_bypass_with_resampler_mono() {
        let n = 256;
        let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
        let input_r = input_l.clone();

        let (out_l, out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, true);

        assert!(!out_l.is_empty());
        assert_eq!(
            out_l, out_r,
            "Mesmo com resampler, mono deve produzir L == R"
        );
    }

    #[test]
    fn test_hotpath_gate_closed_and_silence() {
        let n = 64;
        let input_l = vec![0.0; n];
        let input_r = vec![0.0; n];

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

        let gate_params = GateParams {
            hold_frames: 0,
            fade_frames: 0,
            ..Default::default()
        };
        let mut silence_hysteresis = DynamicHysteresis::new();
        // Force gate to closed explicitly
        silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
        silence_hysteresis.update(0.0, 0.1, 0.01, &gate_params, 1000);
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
            threshold_open_sq: 0.1,
            threshold_close_sq: 0.01,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            bridge_ptr: &mut *bridge as *mut DspBridge,
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        assert_eq!(allocs, 0, "Alocação no hot-path!");

        assert!(
            rt_status.check_flag(crate::spsc::RT_STATUS_IS_SILENT),
            "Gate deveria fechar"
        );
        assert!(!rt_status.check_flag(crate::spsc::RT_STATUS_IS_FADING));

        let read_idx = bridge.active_read_idx.load(Ordering::Acquire);
        // O buffer traseiro (1 - read_idx) foi escrito.
        let out_buf = &bridge.buffers[1 - read_idx];
        assert_eq!(
            out_buf.n_samples, 0,
            "O bridge deveria ter n_samples = 0 no silence bypass"
        );
    }

    #[test]
    fn test_hotpath_gate_fading() {
        let n = 64;
        let mut input_l = vec![0.0; n];
        let mut input_r = vec![0.0; n];
        // Sinal fraco para disparar fade out
        for i in 0..n {
            input_l[i] = 0.05; // menor que open sqrt(0.1) ~ 0.31, maior que close sqrt(0.0001) = 0.01
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

        let gate_params = GateParams {
            hold_frames: 0,
            fade_frames: 100,
            ..Default::default()
        };
        let mut silence_hysteresis = DynamicHysteresis::new();
        // Abrimos o gate manualmente para que o sinal fraco ative o fading out
        silence_hysteresis.update(1.0, 0.1, 0.0001, &gate_params, 100);

        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;

        // Sinal zero para cair abaixo do threshold_close_sq (0.01) e iniciar fading out
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
            bridge_ptr: &mut *bridge as *mut DspBridge,
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        assert_eq!(allocs, 0);
        assert!(
            rt_status.check_flag(crate::spsc::RT_STATUS_IS_FADING),
            "Deveria estar em FadeOut"
        );
        assert!(!rt_status.check_flag(crate::spsc::RT_STATUS_IS_SILENT));
    }

    #[test]
    fn test_hotpath_clipping_detection() {
        let n = 64;
        let mut input_l = vec![0.0; n];
        let input_r = vec![0.0; n];
        // Satura o sinal
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
            output_gain_mult: 1.0, // Não altera o clipping inserido
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0,
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            bridge_ptr: &mut *bridge as *mut DspBridge,
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        assert_eq!(allocs, 0);
        assert!(
            rt_status.check_flag(crate::spsc::RT_STATUS_HAS_CLIPPED),
            "Clipping deveria ser detectado"
        );
    }

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
            // Simulando que o consumidor não leu nada ainda
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

        // 1ª passada
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
            bridge_ptr: &mut *bridge as *mut DspBridge,
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, n, ctx);
        // Generation deve ser 1

        // 2ª passada sem o consumidor ler
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
            bridge_ptr: &mut *bridge as *mut DspBridge,
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
        };

        let _guard = TrackingGuard::new();
        capture_dsp_pipeline(&mut samples_l2, &mut samples_r2, n, ctx2);
        let allocs = ALLOC_COUNT.load(Ordering::Relaxed);
        drop(_guard);

        assert_eq!(allocs, 0);

        let dropped = bridge.dropped_frames.load(Ordering::Relaxed);
        assert_eq!(dropped, 1, "Deveria ter detectado 1 frame dropado");
    }
}
