// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Thread de I/O para gravação em disco.
//! Consome dados de áudio do ring buffer lock-free SPSC e grava arquivos WAV
//! utilizando o subsistema **`io_uring`** para I/O de disco 100% assíncrono e não-bloqueante.
//! Para manter semântica verdadeiramente zero-blocking, os headers WAV são gerados manualmente
//! (header RIFF/WAVE padrão de 44 bytes para Float32) e as amostras são escritas diretamente
//! via `tokio_uring::fs::File::write_at`. Em mudanças de formato do stream ou shutdown gracioso,
//! o header do arquivo é reescrito com a contagem final de bytes e um `fsync` é emitido
//! para garantir integridade do arquivo antes do fechamento.
//!
//! # Limitação Conhecida: Tamanho máximo de arquivo WAV
//! O formato RIFF/WAV utiliza campos `u32` para tamanhos de chunks, limitando o payload de
//! dados a ~4 GiB (~3h de áudio estéreo 32-bit float a 48kHz). Na prática, este limite
//! não é atingível nos casos de uso previstos do AudioRip (capturas curtas via qpwgraph).

use anyhow::{Context, Result};
use rtrb::Consumer;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

use crate::buffer::{
    AlignedBlock, AudioMetadata, MAX_BLOCK_SIZE, OVERRUN_COUNT, RingPayload, SHUTDOWN,
};

/// Obtém um timestamp formatado como string usando `libc::localtime_r` (thread-safe).
/// Retorna `None` se a chamada ao sistema falhar.
fn get_formatted_timestamp() -> Option<String> {
    let mut t: libc::time_t = 0;
    unsafe { libc::time(&mut t) };
    let mut tm_buf: libc::tm = unsafe { std::mem::zeroed() };
    let tm = unsafe { libc::localtime_r(&t, &mut tm_buf) };

    if tm.is_null() {
        return None;
    }

    Some(format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
        tm_buf.tm_year + 1900,
        tm_buf.tm_mon + 1,
        tm_buf.tm_mday,
        tm_buf.tm_hour,
        tm_buf.tm_min,
        tm_buf.tm_sec
    ))
}

/// Imprime uma mensagem no console com timestamp formatado.
/// Utilizada exclusivamente pela thread de I/O (nunca pela thread DSP).
fn log_msg(msg: &str) {
    if let Some(ts) = get_formatted_timestamp() {
        println!("[{}] [AudioRip] {}", ts, msg);
    } else {
        println!("[AudioRip] {}", msg);
    }
}

/// Gera o nome do arquivo WAV baseado no timestamp atual.
/// Formato: `capture_YYYYMMDD_HHMMSS.wav`
fn generate_wav_filename() -> String {
    let mut t: libc::time_t = 0;
    unsafe { libc::time(&mut t) };
    let mut tm_buf: libc::tm = unsafe { std::mem::zeroed() };
    let tm = unsafe { libc::localtime_r(&t, &mut tm_buf) };

    if tm.is_null() {
        "capture_unknown.wav".to_string()
    } else {
        format!(
            "capture_{:04}{:02}{:02}_{:02}{:02}{:02}.wav",
            tm_buf.tm_year + 1900,
            tm_buf.tm_mon + 1,
            tm_buf.tm_mday,
            tm_buf.tm_hour,
            tm_buf.tm_min,
            tm_buf.tm_sec
        )
    }
}

/// Gera um header WAV padrão para IEEE Float 32-bit PCM usando a crate `hound`.
/// `data_bytes` é o tamanho total dos dados brutos de áudio (pode ser 0 inicialmente).
///
/// O processo de geração cria um header válido via `hound` e então aplica patches cirúrgicos
/// nos campos de tamanho (RIFF, data, fact) para refletir a quantidade real de dados escritos.
/// A busca por chunks é feita por scan explícito para robustez contra mudanças internas do `hound`.
fn build_wav_header(meta: &AudioMetadata, data_bytes: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: meta.channels,
        sample_rate: meta.sample_rate as u32,
        bits_per_sample: meta.bit_depth,
        sample_format: hound::SampleFormat::Float,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        // Escrever 0 amostras gera apenas o header.
        let writer = hound::WavWriter::new(&mut cursor, spec)
            .context("Falha ao criar escritor de header WAV")?;
        writer
            .finalize()
            .context("Falha ao finalizar escritor de header WAV")?;
    }
    let mut header = cursor.into_inner();

    // Patcha o tamanho do chunk RIFF (bytes 4..8)
    let file_size = header.len() as u32 - 8 + data_bytes;
    header[4..8].copy_from_slice(&file_size.to_le_bytes());

    // Patcha o tamanho do chunk "data" — busca explícita por `rposition` para robustez,
    // evitando assumir que os últimos 4 bytes do header são necessariamente o campo de tamanho.
    let data_pos = header
        .windows(4)
        .rposition(|w| w == b"data")
        .context("Chunk 'data' não encontrado no header WAV gerado pelo hound")?;
    if data_pos + 8 > header.len() {
        anyhow::bail!("Header WAV malformado: chunk 'data' truncado");
    }
    header[data_pos + 4..data_pos + 8].copy_from_slice(&data_bytes.to_le_bytes());

    // Patcha o campo de contagem de amostras do chunk "fact" (requerido para formato Float).
    // Valida o tamanho do chunk antes de patchear para robustez contra mudanças no `hound`.
    if meta.bit_depth == 32 {
        let samples_per_channel = data_bytes / (meta.channels as u32 * (meta.bit_depth as u32 / 8));
        if let Some(fact_pos) = header.windows(4).position(|w| w == b"fact")
            && fact_pos + 12 <= header.len()
        {
            let chunk_size = u32::from_le_bytes(
                header[fact_pos + 4..fact_pos + 8]
                    .try_into()
                    .expect("Fatia de 4 bytes para u32 deve ser infálivel"),
            );
            if chunk_size >= 4 {
                header[fact_pos + 8..fact_pos + 12]
                    .copy_from_slice(&samples_per_channel.to_le_bytes());
            }
        }
    }

    Ok(header)
}

/// Escritor assíncrono de WAV utilizando `tokio_uring` para I/O de disco puramente zero-blocking.
///
/// Mantém um buffer de I/O reutilizável (`io_buf`) para evitar alocação na heap a cada bloco
/// de áudio recebido do ring buffer, reduzindo significativamente a pressão no alocador.
struct AsyncWavWriter {
    /// Handle do arquivo aberto via io_uring.
    file: tokio_uring::fs::File,
    /// Metadados do stream de áudio atual (sample rate, bit depth, canais).
    metadata: AudioMetadata,
    /// Total de bytes de dados de áudio escritos (exclui o header WAV).
    /// Limitado a u32 pela especificação RIFF/WAV (~4 GiB máximo).
    data_bytes_written: u32,
    /// Offset atual de escrita no arquivo (header + dados já escritos).
    current_offset: u64,
    /// Buffer de I/O reutilizável para conversão f32→bytes antes da escrita via io_uring.
    /// O `tokio_uring` exige ownership do buffer; após a escrita, o buffer é devolvido
    /// e reutilizado no próximo bloco, eliminando alocações repetidas.
    io_buf: Vec<u8>,
}

impl AsyncWavWriter {
    /// Cria um novo arquivo WAV e escreve o header inicial.
    async fn create(path: &PathBuf, metadata: AudioMetadata) -> Result<Self> {
        let file = tokio_uring::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .await
            .context("Falha ao abrir arquivo via io_uring")?;

        let header = build_wav_header(&metadata, 0)?;
        let header_len = header.len() as u64;
        let (res, _): (std::io::Result<usize>, _) = file.write_at(header, 0).submit().await;
        res.context("Falha ao escrever header WAV inicial")?;

        Ok(Self {
            file,
            metadata,
            data_bytes_written: 0,
            current_offset: header_len,
            io_buf: Vec::new(),
        })
    }

    /// Escreve um bloco de áudio bruto de forma totalmente assíncrona.
    /// O buffer de I/O interno é reutilizado entre chamadas para evitar alocações repetidas.
    async fn write_block(&mut self, block: &AlignedBlock<MAX_BLOCK_SIZE>) -> Result<()> {
        let valid_samples = block.valid_len;
        if valid_samples == 0 {
            return Ok(());
        }

        // Prepara o buffer de I/O reutilizável com os bytes do bloco (Little Endian conforme WAV)
        let bytes_len = valid_samples * 4;
        self.io_buf.clear();
        self.io_buf.reserve(bytes_len);

        // Conversão segura de bytes iterativa. O `tokio_uring` exige ownership do buffer;
        // `f32::to_le_bytes()` garante segurança e independência de plataforma.
        for &sample in &block.data[..valid_samples] {
            self.io_buf.extend_from_slice(&sample.to_le_bytes());
        }

        // O tokio_uring toma ownership do buffer para a escrita assíncrona.
        // Após conclusão, o buffer é devolvido e reatribuído para reutilização.
        let buf = std::mem::take(&mut self.io_buf);
        let (res, returned_buf): (std::io::Result<usize>, Vec<u8>) =
            self.file.write_at(buf, self.current_offset).submit().await;
        self.io_buf = returned_buf;

        let written = res.context("Falha ao escrever bloco de áudio via io_uring")?;

        self.data_bytes_written += written as u32;
        self.current_offset += written as u64;

        Ok(())
    }

    /// Finaliza o arquivo WAV reescrevendo o header com o tamanho final dos dados,
    /// e executando um `fsync` explícito para garantir persistência dos dados em disco.
    async fn finalize(self) -> Result<()> {
        let header = build_wav_header(&self.metadata, self.data_bytes_written)?;
        // Reescreve o header na origem (offset 0)
        let (res, _): (std::io::Result<usize>, _) = self.file.write_at(header, 0).submit().await;
        res.context("Falha ao reescrever header WAV na finalização")?;

        // Garante sincronização com o estado do hardware
        self.file
            .sync_all()
            .await
            .context("Falha ao fsync do arquivo WAV")?;

        // `tokio_uring::fs::File` é fechado automaticamente quando descartado (drop)
        Ok(())
    }
}

/// Ponto de entrada principal da thread de I/O de Disco.
/// Consome o ring buffer lock-free e grava arquivos WAV de forma totalmente assíncrona via `io_uring`.
/// Suporta shutdown gracioso: quando `SHUTDOWN` é ativado, todos os dados remanescentes são drenados,
/// e o arquivo WAV é devidamente finalizado (via `fsync`) antes de retornar.
pub async fn disk_writer_loop(mut consumer: Consumer<RingPayload<MAX_BLOCK_SIZE>>) -> Result<()> {
    let mut wav_writer: Option<AsyncWavWriter> = None;

    loop {
        if let Ok(payload) = consumer.pop() {
            match payload {
                RingPayload::Metadata(meta) => {
                    // Finaliza o WAV anterior se o formato mudou no meio do stream.
                    if let Some(existing_writer) = wav_writer.take() {
                        existing_writer
                            .finalize()
                            .await
                            .context("Falha ao finalizar arquivo WAV anterior")?;
                        log_msg("⏹️  Fechei a captura anterior com segurança.");
                    }

                    let filename_str = generate_wav_filename();

                    // Sempre salva no diretório de trabalho atual.
                    let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                    let filename = base_dir.join(&filename_str);

                    println!();
                    log_msg(&format!("🎬 Criei o arquivo: {}", filename.display()));
                    log_msg("🎧 Comecei a escrever áudio estrito de fonte PipeWire...");
                    println!();

                    let writer = AsyncWavWriter::create(&filename, meta).await?;
                    wav_writer = Some(writer);
                }
                RingPayload::Audio(block) => {
                    if let Some(writer) = &mut wav_writer {
                        writer.write_block(&block).await?;
                    }
                }
                RingPayload::StreamStop => {
                    if let Some(writer) = wav_writer.take() {
                        writer
                            .finalize()
                            .await
                            .context("Falha ao finalizar WAV no encerramento do stream")?;
                        log_msg(
                            "⏹️  Fonte de áudio interrompida. Arquivo WAV fechado com segurança e pronto para uso.",
                        );
                    }
                }
            }
        } else if SHUTDOWN.load(Ordering::SeqCst) {
            // Drena itens remanescentes que chegaram entre o último pop e a detecção de shutdown.
            while let Ok(payload) = consumer.pop() {
                if let RingPayload::Audio(block) = payload
                    && let Some(writer) = &mut wav_writer
                {
                    writer.write_block(&block).await?;
                }
            }
            // Finaliza o header WAV e sincroniza com disco para que o arquivo esteja garantidamente válido.
            if let Some(writer) = wav_writer.take() {
                writer
                    .finalize()
                    .await
                    .context("Falha ao finalizar arquivo WAV no shutdown")?;
                log_msg("⏹️  Fechei o arquivo de captura com segurança.");
            }

            // Reporta overruns detectados ao usuário (perda potencial de dados de áudio)
            let overruns = OVERRUN_COUNT.load(Ordering::Relaxed);
            if overruns > 0 {
                log_msg(&format!(
                    "⚠️  Detectados {} overruns no ring buffer — possível perda de dados de áudio.",
                    overruns
                ));
            }

            break;
        } else {
            // Ring buffer vazio e sem shutdown — backoff para evitar uso de 100% da CPU.
            tokio::task::yield_now().await;
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    }

    Ok(())
}
