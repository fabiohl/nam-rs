// Copyright 2026 Fábio Henrique de Lima Silva. Todos os direitos reservados.
// Este arquivo é confidencial e propriedade de Fábio Henrique de Lima Silva. O uso não autorizado é estritamente proibido.

#![warn(missing_docs)]

//! Ponto de entrada principal do AudioRip.
//! Inicializa recursos globais, estabelece canais lock-free, dispara loops assíncronos de I/O,
//! e inicializa o backend standalone do `nih_plug` para processamento bit-perfect de áudio via PipeWire.
//! Coordena o shutdown gracioso entre todas as threads quando CTRL+C é pressionado.
//!
//! # Regras de Arquitetura para Desenvolvedores
//! - **ZERO LOCKS** na thread DSP (módulo audio).
//! - **ZERO ALOCAÇÕES** na thread DSP (o loop `process()` deve ser zero-allocation).
//! - **Zero Resampling**: Extração bit-perfect adaptando-se dinamicamente aos streams do host.

mod audio;
mod buffer;
mod disk;

use audio::AudioRipPlugin;
use nih_plug::wrapper::standalone::nih_export_standalone;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::time::Duration;

fn main() {
    // ------------------------------------------------------------
    // 1️⃣ Inicializa o ring buffer lock-free SPSC
    // ------------------------------------------------------------
    // O ring buffer garante comunicação ZERO-COPY e ZERO-WAIT entre
    // a estrita thread de Tempo Real e a thread de alto throughput io_uring.
    // As dimensões são const-generic para forçar comportamento alinhado por cache.
    let (producer, consumer) = buffer::create_audio_ring_buffer(buffer::RING_CAPACITY);

    // ------------------------------------------------------------
    // 2️⃣ Compartilha o producer com o plugin
    // ------------------------------------------------------------
    // `nih_export_standalone` cria a instância do plugin internamente via Default.
    // Um OnceLock global é a forma mais limpa e segura de injetar o Producer sem
    // envolver ponteiros dinâmicos ou código unsafe que violaria as restrições RT.
    let _ = audio::PRODUCER.set(Mutex::new(Some(producer)));

    // ------------------------------------------------------------
    // 3️⃣ Configura o tratamento de shutdown gracioso
    // ------------------------------------------------------------
    // Um handler de sinal intercepta a intenção de encerramento. Threads em background
    // leem `SHUTDOWN` passivamente para garantir que ring buffers sejam totalmente drenados
    // antes do fechamento.
    ctrlc::set_handler(|| {
        if buffer::SHUTDOWN.load(Ordering::SeqCst) {
            // Segundo sinal: aborta imediatamente.
            std::process::exit(1);
        }
        buffer::SHUTDOWN.store(true, Ordering::SeqCst);
    })
    .expect("Erro ao configurar handler de Ctrl-C");

    // 4️⃣ Dispara a thread de I/O assíncrona estritamente separada (tokio-uring).
    // Quando SHUTDOWN é ativado, esta thread drena o ring buffer, finaliza o WAV,
    // e então encerra — provocando o término do processo já que o nih_plug bloqueia a main.
    let io_handle = std::thread::spawn(move || {
        tokio_uring::start(async {
            if let Err(e) = disk::disk_writer_loop(consumer).await {
                eprintln!("Erro no Disk Writer: {e:?}");
            }
        });
    });

    // 5️⃣ Dispara thread coordenadora de shutdown.
    // Aguarda o SHUTDOWN, junta a thread I/O (garantindo finalização do WAV),
    // e encerra o processo de forma limpa já que o nih_plug bloqueia a main indefinidamente.
    std::thread::spawn(move || {
        while !buffer::SHUTDOWN.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(100));
        }
        // Aguarda a thread I/O drenar o ring buffer e finalizar o WAV.
        let _ = io_handle.join();
        // Breve pausa para permitir que o buffer de stdout esvazie mensagens de forma limpa.
        std::thread::sleep(Duration::from_millis(50));
        std::process::exit(0);
    });

    // 6️⃣ Inicializa o motor DSP de Tempo Real via emulação de plugin PipeWire.
    // Isso bloqueia a thread main indefinidamente.
    nih_export_standalone::<AudioRipPlugin>();
}
