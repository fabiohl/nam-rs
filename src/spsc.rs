// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Estruturas de dados e flags compartilhadas para coordenação entre threads.
//! Na Tarefa 1.2, este módulo foi expandido com as estruturas de comunicação
//! lock-free SPSC para parametrização CLI -> DSP (input/output gain, troca de modelo).

use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::atomic::AtomicBool;

/// Flag global para shutdown coordenado e gracioso entre todas as threads.
/// Definida como `true` pelo handler de CTRL+C.
pub static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Payload SPSC enviado do Host (CLI/UI) para a Thread DSP.
/// Adota alinhamento a 128 bytes para mitigar False Sharing.
#[repr(align(128))]
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum ParamPayload {
    /// Injeta o ganho de entrada (Input Gain) em dB.
    InputGain(f32),
    /// Injeta o ganho de saída (Output Gain) em dB.
    OutputGain(f32),
    /// Aponta para as estruturas matemáticas já decodificadas de um arquivo .namb.
    /// O ponteiro garante zero alocação (no-heap) e O(1) na thread DSP.
    LoadModelPtr(*mut ()), // TODO: Substituir por um container estrito na Tarefa 4.1
}

/// Garantir que podemos enviar o ponteiro bruto entre threads sem erro de compilador (unsafe trait).
/// OBS: Devemos assegurar concorrência segura manualmente ao implementar os carregadores.
unsafe impl Send for ParamPayload {}
unsafe impl Sync for ParamPayload {}

/// Cria e retorna a malha SPSC lock-free para o pipeline.
///
/// `capacity` deve ser preferencialmente potência de 2.
#[allow(dead_code)]
pub fn setup_spsc(capacity: usize) -> (Producer<ParamPayload>, Consumer<ParamPayload>) {
    RingBuffer::new(capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_spsc_concurrency() {
        // Configura o SPSC Ring Buffer para testes
        let (mut producer, mut consumer) = setup_spsc(64);

        // Garante que o shutdown inicie limpo
        SHUTDOWN.store(false, Ordering::SeqCst);

        // Thread DSP (Consumidor sched_fifo simulado)
        let dsp_handle = thread::spawn(move || {
            let mut processed_messages = 0;
            // Laço de super rotação (semelhante ao DSP real)
            while !SHUTDOWN.load(Ordering::Relaxed) {
                if let Ok(payload) = consumer.pop() {
                    match payload {
                        ParamPayload::InputGain(gain) => {
                            assert!((-60.0..=24.0).contains(&gain));
                            processed_messages += 1;
                        }
                        ParamPayload::OutputGain(gain) => {
                            assert!((-60.0..=24.0).contains(&gain));
                            processed_messages += 1;
                        }
                        ParamPayload::LoadModelPtr(_ptr) => {
                            processed_messages += 1;
                        }
                    }
                }
            }
            processed_messages
        });

        // Thread CLI (Produtor paramétrico)
        let cli_handle = thread::spawn(move || {
            // Envia vários pacotes enquanto simula CLI interactions
            let payloads = vec![
                ParamPayload::InputGain(0.0),
                ParamPayload::OutputGain(-12.5),
                ParamPayload::InputGain(12.0),
                ParamPayload::LoadModelPtr(std::ptr::null_mut()),
                ParamPayload::OutputGain(3.5),
            ];

            for payload in payloads {
                while producer.push(payload.clone()).is_err() {
                    thread::yield_now(); // Buf cheio
                }
                // Simula latência do operador CLI
                thread::sleep(Duration::from_millis(5));
            }

            thread::sleep(Duration::from_millis(10));
            // Dispara shutdown
            SHUTDOWN.store(true, Ordering::SeqCst);
        });

        cli_handle.join().expect("CLI thread failed");
        let count = dsp_handle.join().expect("DSP thread failed");

        assert_eq!(
            count, 5,
            "O DSP deveria processar exatamente 5 pacotes lock-free"
        );
    }
}
