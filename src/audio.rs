// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Núcleo de processamento de áudio DSP usando `nih_plug`.
//! Contém o callback estrito da thread de Tempo-Real/DSP responsável pelo processamento
//! de inferência neural NAM. Recebe amostras f32 brutas do host PipeWire via backend
//! standalone do `nih_plug` e as processa em tempo real.
//! Zero alocação na heap, zero I/O, zero mutexes durante o `process()`.
//!
//! # Estado Atual (pós-Tarefa 1.1)
//! O motor de inferência neural ainda não foi implementado (Sprints 2-5).
//! O `process()` opera em modo **pass-through**: o áudio de entrada flui inalterado
//! para a saída, garantindo que o pipeline PipeWire funcione sem interrupções.

use nih_plug::prelude::*;
use std::sync::Arc;

/// Implementação do plugin NAM-rs.
/// Motor de inferência neural para emulação de amplificadores via PipeWire.
pub struct NamRsPlugin {
    params: Arc<NamRsParams>,
    /// Indica se a configuração da thread RT (afinidade de núcleo, agendador) já foi aplicada.
    thread_configured: bool,
}

/// Parâmetros do plugin NAM-rs.
/// Será expandido nas Sprints seguintes com Input Gain, Output Gain e seleção de modelo .namb.
#[derive(Params)]
pub struct NamRsParams {}

impl Default for NamRsPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(NamRsParams {}),
            thread_configured: false,
        }
    }
}

impl Plugin for NamRsPlugin {
    const NAME: &'static str = "NAM-rs";
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

    // Tipo de tarefa em background opaco — não utilizado; exigido pelo nih_plug.
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
        true
    }

    fn reset(&mut self) {
        // Sem estado interno a resetar nesta fase.
        // Será expandido quando o motor de inferência mantiver estados recorrentes (LSTM).
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Aplica a configuração da thread RT apenas uma vez, na primeira chamada do callback.
        // Executado antes de qualquer processamento de dados de áudio.
        if !self.thread_configured {
            self.configure_realtime_thread();
            self.thread_configured = true;
        }

        // ⚡ PASS-THROUGH: O áudio de entrada flui inalterado para a saída.
        // O nih_plug opera in-place — entrada e saída compartilham o mesmo buffer.
        // Não tocar nos dados do buffer significa que a entrada é copiada diretamente
        // para a saída, comportamento natural de um processador sem efeito aplicado.
        //
        // Quando o motor de inferência neural for implementado (Sprints 2-5),
        // este bloco será substituído pela cadeia de processamento:
        // Input Gain → Resampler → NAM Inference (WaveNet/LSTM) → Output Gain → Resampler
        let _ = buffer;

        ProcessStatus::Normal
    }
}

impl NamRsPlugin {
    /// Seleciona o núcleo de CPU ideal para fixar a thread RT (core affinity).
    ///
    /// Critérios de seleção (em ordem de desempate):
    /// 1. Maior `cpu_capacity` lida de `/sys/devices/system/cpu/cpuN/cpu_capacity`
    /// 2. Menor total de interrupções lido de `/proc/interrupts`
    /// 3. Maior índice de CPU (critério final de desempate)
    ///
    /// Retorna o índice do CPU selecionado, ou `None` se a detecção falhar completamente.
    fn select_optimal_cpu() -> Option<usize> {
        use std::fs;

        // Descobre os CPUs lógicos disponíveis via sysfs
        let cpu_dir = fs::read_dir("/sys/devices/system/cpu").ok()?;
        let mut cpus: Vec<usize> = cpu_dir
            .filter_map(|entry| {
                let name = entry.ok()?.file_name();
                let name = name.to_str()?;
                name.strip_prefix("cpu")?.parse::<usize>().ok()
            })
            .collect();

        if cpus.is_empty() {
            return None;
        }
        cpus.sort_unstable();

        // 1. Lê a capacidade de cada CPU (padrão 1024 se ausente — valor ARM DynamIQ / EAS)
        let capacities: Vec<(usize, u64)> = cpus
            .iter()
            .map(|&cpu| {
                let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
                let cap = fs::read_to_string(path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .unwrap_or(1024);
                (cpu, cap)
            })
            .collect();

        // 2. Analisa o total de interrupções por CPU a partir de /proc/interrupts
        let irq_totals = Self::parse_interrupts_per_cpu(cpus.len());

        // 3. Calcula a pontuação composta: (capacidade DESC, -total_interrupções, índice_cpu DESC)
        //    Maximizamos capacidade e índice_cpu, minimizamos total_interrupções.
        capacities
            .iter()
            .map(|&(cpu, cap)| {
                let irqs = irq_totals.get(cpu).copied().unwrap_or(u64::MAX);
                (cpu, cap, irqs)
            })
            .max_by(|a, b| {
                // Critério primário: maior capacidade
                a.1.cmp(&b.1)
                    // Critério secundário: menos interrupções (comparação invertida)
                    .then_with(|| b.2.cmp(&a.2))
                    // Critério terciário: maior índice de CPU
                    .then_with(|| a.0.cmp(&b.0))
            })
            .map(|(cpu, _, _)| cpu)
    }

    /// Analisa `/proc/interrupts` e retorna um `Vec` indexado pelo número do CPU
    /// contendo a contagem total de interrupções de todas as linhas de IRQ para cada CPU.
    ///
    /// Apenas linhas com identificadores numéricos de IRQ são contadas (IRQs de hardware).
    /// Contadores internos do sistema (LOC, NMI, RES, CAL, TLB, etc.) são excluídos
    /// porque são inerentes ao agendador e não representam carga de dispositivos externos
    /// que interfeririam no processamento DSP.
    fn parse_interrupts_per_cpu(num_cpus: usize) -> Vec<u64> {
        use std::fs;

        let mut totals = vec![0u64; num_cpus];

        let content = match fs::read_to_string("/proc/interrupts") {
            Ok(c) => c,
            Err(_) => return totals,
        };

        for line in content.lines().skip(1) {
            // Pula linhas que não começam com um número de IRQ válido
            let trimmed = line.trim_start();
            let irq_end = trimmed.find(':').unwrap_or(0);
            if irq_end == 0 {
                continue;
            }
            // Conta apenas linhas de IRQ de hardware (identificadores numéricos)
            if !trimmed[..irq_end]
                .trim()
                .bytes()
                .all(|b| b.is_ascii_digit())
            {
                continue;
            }

            // Analisa as contagens por CPU após o dois-pontos
            let after_colon = match trimmed.get(irq_end + 1..) {
                Some(s) => s,
                None => continue,
            };

            for (cpu_idx, token) in after_colon.split_whitespace().enumerate() {
                if cpu_idx >= num_cpus {
                    break;
                }
                if let Ok(count) = token.parse::<u64>() {
                    totals[cpu_idx] += count;
                } else {
                    // Chegou no texto descritivo do dispositivo — para de analisar esta linha
                    break;
                }
            }
        }

        totals
    }

    /// Tenta fixar a thread DSP no núcleo físico ideal e aplicar SCHED_FIFO para
    /// agendamento de tempo real. Chamada apenas uma vez na primeira invocação do `process()`,
    /// antes do fluxo de dados começar.
    ///
    /// NOTA: No Linux moderno, o kernel recusa conceder SCHED_FIFO mesmo sob demanda. Na
    /// prática, essa solicitação é ignorada silenciosamente. Porém, não ficamos sem proteção:
    /// o PipeWire, via RTkit, concede automaticamente alta prioridade dentro do CFS.
    ///
    /// # Exceção de I/O Documentada
    /// Esta função usa `println!`/`eprintln!` para log diagnóstico único de inicialização
    /// e lê `/sys` + `/proc` para detectar a topologia de CPU.
    /// Embora envolva chamadas de I/O, executa apenas uma vez e antes de qualquer dado de áudio
    /// fluir, portanto não compromete a latência do processamento em regime estacionário.
    fn configure_realtime_thread(&self) {
        #[cfg(target_os = "linux")]
        {
            // Seleciona dinamicamente o núcleo de CPU ideal (fallback para CPU 0 se a detecção falhar)
            let target_cpu = Self::select_optimal_cpu().unwrap_or(0);

            unsafe {
                let thread_id = libc::pthread_self();

                // 1. Afinidade de Núcleo: Fixa a thread no núcleo ideal para proteger o Cache L1/L2
                let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut cpuset);
                libc::CPU_SET(target_cpu, &mut cpuset);

                let ret_aff = libc::pthread_setaffinity_np(
                    thread_id,
                    std::mem::size_of::<libc::cpu_set_t>(),
                    &cpuset,
                );

                if ret_aff != 0 {
                    eprintln!(
                        "Aviso: Falha ao definir afinidade de CPU para o núcleo {} (erro {}).",
                        target_cpu, ret_aff
                    );
                }

                // 2. Tempo Real: Aplica SCHED_FIFO para preempção determinística
                let mut param: libc::sched_param = std::mem::zeroed();
                param.sched_priority = 90; // Alta prioridade (requer rtprio >= 90 no limits.conf)

                let _ret_sched = libc::pthread_setschedparam(thread_id, libc::SCHED_FIFO, &param);

                // Verifica o estado real da thread após as tentativas de configuração
                let mut actual_policy = 0;
                let mut actual_param: libc::sched_param = std::mem::zeroed();
                let ret_getsched =
                    libc::pthread_getschedparam(thread_id, &mut actual_policy, &mut actual_param);

                let actual_cpu = libc::sched_getcpu();

                if ret_getsched == 0 {
                    let reset_on_fork_flag = 0x40000000;
                    let has_reset_on_fork = (actual_policy & reset_on_fork_flag) != 0;
                    let base_policy = actual_policy & !reset_on_fork_flag;

                    let mut policy_str = match base_policy {
                        libc::SCHED_FIFO => "SCHED_FIFO".to_string(),
                        libc::SCHED_RR => "SCHED_RR".to_string(),
                        libc::SCHED_OTHER => "SCHED_OTHER".to_string(),
                        libc::SCHED_BATCH => "SCHED_BATCH".to_string(),
                        libc::SCHED_IDLE => "SCHED_IDLE".to_string(),
                        other => format!("UNKNOWN: {}", other),
                    };

                    if has_reset_on_fork {
                        policy_str.push_str(" | SCHED_RESET_ON_FORK");
                    }
                    println!(
                        "[NAM-rs] 🔍 Audio/DSP Thread: CPU Core = {}, Policy = {}, Priority = {}",
                        actual_cpu, policy_str, actual_param.sched_priority
                    );
                } else {
                    eprintln!(
                        "[NAM-rs] ⚠️ Falha ao verificar parâmetros da thread (erro {}).",
                        ret_getsched
                    );
                }
            }
        }
    }
}
