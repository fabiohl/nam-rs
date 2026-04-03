// Copyright 2026 Fábio Henrique de Lima Silva. Todos os direitos reservados.
// Este arquivo é confidencial e propriedade de Fábio Henrique de Lima Silva. O uso não autorizado é estritamente proibido.

//! Núcleo DSP de áudio utilizando `nih_plug`.
//! Contém o callback estrito da thread de Tempo Real/DSP responsável pela captura Bit-Perfect.
//! Recebe amostras brutas f32 do host PipeWire via backend standalone do `nih_plug`,
//! intercala-as em blocos alinhados a cache e as empurra para o ring buffer lock-free SPSC
//! para consumo pela thread de I/O. Zero alocação na heap, zero I/O, zero mutexes
//! durante o `process()`.

use nih_plug::prelude::*;
use rtrb::Producer;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use crate::buffer::{AlignedBlock, AudioMetadata, MAX_BLOCK_SIZE, OVERRUN_COUNT, RingPayload};

/// Limite de tolerância para considerar um sinal de áudio como silêncio absoluto.
/// Utilizado pelo Noise Gate de Zero-Overhead para suprimir gravação de blocos vazios.
const SILENCE_THRESHOLD: f32 = 1e-6;

/// Armazenamento global para injetar o Producer na instância do plugin criada pelo `nih_plug` standalone.
/// O padrão `OnceLock<Mutex<Option<...>>>` é utilizado porque o `nih_plug` instancia o plugin
/// internamente via `Default::default()`, tornando necessário injetar o producer através de um global.
/// O Mutex é travado apenas uma vez durante `Default::default()`, jamais durante `process()`.
pub static PRODUCER: OnceLock<Mutex<Option<Producer<RingPayload<MAX_BLOCK_SIZE>>>>> =
    OnceLock::new();

/// Verifica se o bloco de áudio é considerado silencioso.
/// O uso de iteradores simples em fatias contíguas de memória permite que o LLVM
/// aplique auto-vetorização (SIMD/AVX2) sem a necessidade de unsafe code ou crates extras.
#[inline(always)]
fn is_silent(data: &[f32]) -> bool {
    data.iter().all(|&sample| sample.abs() < SILENCE_THRESHOLD)
}

/// Implementação do plugin AudioRip.
/// Mantém o producer do ring buffer e rastreia se os metadados do stream já foram enviados.
pub struct AudioRipPlugin {
    params: Arc<AudioRipParams>,
    /// Producer do ring buffer, injetado via global `PRODUCER` durante `Default::default()`.
    pub producer: Option<Producer<RingPayload<MAX_BLOCK_SIZE>>>,
    /// Indica se os metadados (sample rate, bit depth, canais) já foram enviados na sessão atual.
    metadata_sent: bool,
    /// Indica se a configuração RT da thread (core affinity, scheduler) já foi aplicada.
    thread_configured: bool,
}

/// Parâmetros do plugin AudioRip.
/// Struct vazia porque o ripper passivo não possui parâmetros expostos ao usuário,
/// porém o trait `Params` do `nih_plug` exige uma struct de parâmetros.
#[derive(Params)]
pub struct AudioRipParams {}

impl Default for AudioRipPlugin {
    fn default() -> Self {
        let producer = if let Some(mutex) = PRODUCER.get() {
            if let Ok(mut guard) = mutex.lock() {
                guard.take()
            } else {
                None
            }
        } else {
            None
        };

        Self {
            params: Arc::new(AudioRipParams {}),
            producer,
            metadata_sent: false,
            thread_configured: false,
        }
    }
}

impl Plugin for AudioRipPlugin {
    const NAME: &'static str = "AudioRip";
    const VENDOR: &'static str = "Fabio Lima";
    const URL: &'static str = "";
    const EMAIL: &'static str = "fabio.henrique.lima.silva@gmail.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: std::num::NonZeroU32::new(2),
        main_output_channels: std::num::NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = false;
    type SysExMessage = ();

    // Tipo opaco de tarefa em background — não utilizado; requerido pelo nih_plug.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        _buffer_config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        // Reinicia o estado para nova sessão de captura
        self.metadata_sent = false;
        true
    }

    fn reset(&mut self) {
        self.metadata_sent = false;

        // Empurra o sinal de parada para o Consumidor sem alocar memória na thread DSP
        if let Some(producer) = &mut self.producer {
            let _ = producer.push(RingPayload::StreamStop);
        }
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Aplica a configuração RT da thread apenas uma vez, na primeira invocação do callback.
        // Executa antes de qualquer processamento de dados de áudio.
        if !self.thread_configured {
            self.configure_realtime_thread();
            self.thread_configured = true;
        }

        let samples = buffer.samples();
        let channels = buffer.channels();
        let sample_rate = context.transport().sample_rate;

        // Previne panics downstream se o host passar um buffer vazio/morto momentaneamente
        if samples == 0 || channels == 0 || sample_rate <= 0.0 {
            return ProcessStatus::Normal;
        }

        if let Some(producer) = &mut self.producer {
            // Envia metadados do stream para a thread I/O no primeiro buffer ou após reset.
            // Permite que a thread I/O crie um header WAV corretamente formatado.
            if !self.metadata_sent {
                // O nih_plug entrega f32 nativamente; o PipeWire traduz de forma transparente.
                let bit_depth = 32;
                let channels = channels as u16;

                let meta = AudioMetadata {
                    sample_rate,
                    bit_depth,
                    channels,
                };

                if producer.push(RingPayload::Metadata(meta)).is_err() {
                    OVERRUN_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                self.metadata_sent = true;
            }

            // ⚡ CAMINHO EXTREMAMENTE RÁPIDO: Intercalação de Áudio ⚡
            // Transpõe arrays não-intercalados do `nih_plug` nativamente para uma
            // estrutura de blocos lock-free alinhada a 128 bytes. Isso previne bouncing
            // de cache e jamais toca nos alocadores padrão de memória (`Box`, `Vec`).
            let mut block = AlignedBlock::<MAX_BLOCK_SIZE>::new();
            let mut block_idx = 0;

            for sample_idx in 0..samples {
                for ch in 0..channels {
                    if block_idx >= MAX_BLOCK_SIZE {
                        // Bloco cheio — empurra com valid_len exato e inicia um novo.
                        block.valid_len = MAX_BLOCK_SIZE;

                        // Noise Gate de Zero-Overhead: submete áudio apenas se não for silêncio absoluto.
                        if !is_silent(&block.data[..block.valid_len])
                            && producer.push(RingPayload::Audio(block)).is_err()
                        {
                            OVERRUN_COUNT.fetch_add(1, Ordering::Relaxed);
                        }

                        block = AlignedBlock::<MAX_BLOCK_SIZE>::new();
                        block_idx = 0;
                    }

                    // Leitura nativa de memória, pass-through bit-perfect para o ringbuffer.
                    block.data[block_idx] = buffer.as_slice()[ch][sample_idx];
                    block_idx += 1;
                }
            }

            // Empurra o bloco residual com a contagem precisa de amostras válidas.
            // Apenas `valid_len` amostras serão escritas no arquivo WAV pela thread de I/O.
            if block_idx > 0 {
                block.valid_len = block_idx;

                // Noise Gate de Zero-Overhead aplicado ao bloco residual.
                if !is_silent(&block.data[..block.valid_len])
                    && producer.push(RingPayload::Audio(block)).is_err()
                {
                    OVERRUN_COUNT.fetch_add(1, Ordering::Relaxed);
                }
            }

            // Silencia o buffer ativo para evitar feedback indesejado nos fones do usuário,
            // já que o nih_plug processa estritamente in-place.
            for ch_slice in buffer.as_slice() {
                ch_slice.fill(0.0);
            }
        }

        ProcessStatus::Normal
    }
}

impl AudioRipPlugin {
    /// Tenta vincular a thread DSP a um núcleo físico de alta prioridade
    /// e aplicar SCHED_FIFO para escalonamento de tempo real.
    /// Chamada uma única vez na primeira invocação de `process()`, antes do fluxo de dados.
    ///
    /// # Exceção de I/O documentada
    /// Esta função utiliza `println!`/`eprintln!` para log de diagnóstico único.
    /// Embora isso envolva locks no stdout/stderr e potencialmente syscalls `write()`,
    /// a execução ocorre apenas uma vez e antes de qualquer dado de áudio fluir,
    /// portanto não compromete a latência do processamento em regime permanente.
    fn configure_realtime_thread(&self) {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let thread_id = libc::pthread_self();

                // 1. Core Affinity: Trava a thread no núcleo físico para proteger o Cache L1/L2
                let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut cpuset);
                libc::CPU_SET(1, &mut cpuset); // Assumindo Core 1 como núcleo de performance (refinar porque isto não é uma regra imutável!)

                let ret_aff = libc::pthread_setaffinity_np(
                    thread_id,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &cpuset,
                );

                if ret_aff != 0 {
                    eprintln!("Warning: Failed to set CPU affinity (error {}).", ret_aff);
                }

                // 2. Tempo Real: Aplica SCHED_FIFO para preempção determinística
                let mut param: libc::sched_param = std::mem::zeroed();
                param.sched_priority = 90; // Prioridade alta (requer rtprio >= 90 no limits.conf)

                let ret_sched = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);

                if ret_sched == 0 {
                    println!(
                        "[AudioRip] ⚡ DSP Thread pinada ao Core 1 rodando sob SCHED_FIFO (Prioridade 90)."
                    );
                }

                // Verifica o estado real da thread após as tentativas de configuração
                let mut actual_policy = 0;
                let mut actual_param: libc::sched_param = std::mem::zeroed();
                let ret_getsched =
                    libc::pthread_getschedparam(thread_id, &mut actual_policy, &mut actual_param);

                let actual_cpu = libc::sched_getcpu();

                if ret_getsched == 0 {
                    let policy_str = match actual_policy {
                        libc::SCHED_FIFO => "SCHED_FIFO",
                        libc::SCHED_RR => "SCHED_RR",
                        libc::SCHED_OTHER => "SCHED_OTHER",
                        libc::SCHED_BATCH => "SCHED_BATCH",
                        libc::SCHED_IDLE => "SCHED_IDLE",
                        _ => "UNKNOWN",
                    };
                    println!(
                        "[AudioRip] 🔍 Verificação DSP: Core atual = {}, Política = {}, Prioridade = {}",
                        actual_cpu, policy_str, actual_param.sched_priority
                    );
                } else {
                    eprintln!(
                        "[AudioRip] ⚠️ Falha ao verificar parâmetros da thread (error {}).",
                        ret_getsched
                    );
                }
            }
        }
    }
}
