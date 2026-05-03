// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

#[test]
fn test_dsp_bridge_concurrent_access() {
    // 1. Setup do DspBridge:
    // Usamos Box::leak para obter uma referência 'static, simulando o comportamento
    // em tempo de execução onde o objeto vive por toda a duração do host PipeWire.
    // Isso permite converter a referência para ponteiros raw (*const/*mut) com segurança.
    let bridge: &'static DspBridge = Box::leak(Box::new(DspBridge {
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
    }));

    // Ponteiros raw para as threads (writer/reader)
    let bridge_ptr_writer = bridge as *const DspBridge as *mut DspBridge as usize;
    let bridge_ptr_reader = bridge as *const DspBridge;

    // 2. Thread Escritora (Simula Capture Callback RT):
    // Esta thread preenche o "back buffer" (o buffer que não está sendo lido)
    // e depois alterna o índice atômico para torná-lo o novo "front buffer".
    let writer_handle = std::thread::spawn(move || {
        let mut counter = 0.0f32;
        let bridge_ptr_writer = bridge_ptr_writer as *mut DspBridge;
        for _ in 0..1000 {
            let bridge_ref = unsafe { &mut *bridge_ptr_writer };

            // Localiza o buffer inativo (back-buffer) para escrita.
            let back_idx = 1 - bridge_ref.active_read_idx.load(Ordering::Relaxed);
            let back_buf = &mut bridge_ref.buffers[back_idx];

            // Preenche com dados sequenciais para verificar integridade no leitor.
            for i in 0..64 {
                back_buf.buf_l[i] = counter;
                back_buf.buf_r[i] = counter;
                counter += 1.0;
            }
            back_buf.n_samples = 64;

            // BARREIRA DE MEMÓRIA (Release):
            // Garante que todas as escritas no buffer acima sejam visíveis por outras threads
            // ANTES que o 'active_read_idx' ou 'generation' sejam atualizados.
            std::sync::atomic::fence(Ordering::Release);

            bridge_ref
                .active_read_idx
                .store(back_idx, Ordering::Relaxed);
            bridge_ref.generation.fetch_add(1, Ordering::Relaxed);

            // Pequeno delay para simular o tempo de processamento DSP e permitir interleave.
            std::thread::sleep(Duration::from_micros(10));
        }
    });

    // 3. Thread Leitora (Simula Playback Callback RT):
    // Esta thread monitora o contador 'generation'. Quando ele muda, ela
    // consome o novo "front buffer".
    let start = Instant::now();
    let mut last_gen = 0;
    let mut reads = 0;
    let mut last_val_read = -1.0;

    while reads < 1000 && start.elapsed() < Duration::from_millis(500) {
        let bridge_ref = unsafe { &*bridge_ptr_reader };
        let current_gen = bridge_ref.generation.load(Ordering::Relaxed);

        if current_gen != last_gen {
            // BARREIRA DE MEMÓRIA (Acquire):
            // Sincroniza com o Release do escritor. Garante que os dados do buffer
            // lidos abaixo sejam a versão mais recente escrita.
            std::sync::atomic::fence(Ordering::Acquire);

            let read_idx = bridge_ref.active_read_idx.load(Ordering::Relaxed);
            let front_buf = &bridge_ref.buffers[read_idx];

            assert_eq!(front_buf.n_samples, 64);

            // Verificação de Integridade: os dados em um buffer devem ser contíguos.
            let first_val = front_buf.buf_l[0];
            for i in 0..64 {
                assert_eq!(
                    front_buf.buf_l[i],
                    first_val + i as f32,
                    "Buffer mixing detectado no canal L"
                );
                assert_eq!(
                    front_buf.buf_r[i],
                    first_val + i as f32,
                    "Buffer mixing detectado no canal R"
                );
            }

            // Verificação de Monotonicidade:
            // Mesmo se pularmos frames (o que pode ocorrer em testes sob carga),
            // nunca devemos ler dados mais antigos do que os lidos anteriormente.
            assert!(
                first_val > last_val_read,
                "Lido dado mais antigo do que o visto anteriormente! (Stale read)"
            );
            last_val_read = front_buf.buf_l[63];

            last_gen = current_gen;
            reads += 1;
        }

        if last_gen == 1000 {
            break;
        }
    }

    writer_handle.join().unwrap();

    // 4. Verificação de Performance:
    // O teste deve completar rapidamente. Se demorar muito, indica deadlocks
    // ou starvation severa (embora o design seja lock-free).
    assert!(
        start.elapsed() < Duration::from_millis(500),
        "O teste demorou demais para executar"
    );
}
