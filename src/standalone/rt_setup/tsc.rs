// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Calibração e leitura do Time Stamp Counter (TSC) via RDTSC.
//!
//! Oferece medição de tempo com precisão de ~1ns e custo de ~1 ciclo,
//! evitando a syscall vDSO clock_gettime no hot-path de áudio.

use crate::standalone::colors::Colorize;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Frequência calibrada do TSC em GHz (ciclos por nanosegundo).
/// Armazenado como ponto fixo (valor * 1000) para evitar floats no hot-path.
static TSC_FREQ_GHZ_X1000: AtomicU64 = AtomicU64::new(0);
/// Âncora temporal para o fallback de rdtsc (monotônico).
static BOOT_TIME: OnceLock<Instant> = OnceLock::new();

/// Retorna o tempo atual em nanosegundos usando a instrução RDTSC.
///
/// Oferece precisão de ~1ns com custo de ~1 ciclo, evitando a syscall vDSO clock_gettime.
/// Se o TSC não estiver calibrado ou disponível, faz fallback para Instant::now().
#[inline(always)]
pub fn rdtsc_nanos() -> u64 {
    let freq_x1000 = TSC_FREQ_GHZ_X1000.load(Ordering::Relaxed);

    // SAFETY: Divisão por zero no hot-path é fatal. Se a calibração falhou
    // no boot, usamos o relógio do sistema como rede de segurança.
    #[allow(clippy::manual_checked_ops)]
    if freq_x1000 != 0 {
        let cycles = unsafe { core::arch::x86_64::_rdtsc() };
        (cycles * 1000) / freq_x1000
    } else {
        BOOT_TIME.get_or_init(Instant::now).elapsed().as_nanos() as u64
    }
}

/// Calibra a frequência do TSC (Time Stamp Counter) em relação ao relógio do sistema.
///
/// Imagine que a CPU tem um "odômetro" interno que conta cada batida do seu coração (ciclo).
/// Como a velocidade da CPU pode variar, precisamos descobrir quantos "batimentos"
/// equivalem a 1 nanosegundo real para podermos medir o tempo com precisão cirúrgica.
///
/// Esta função é executada apenas uma vez no início do programa (cold-path).
#[cold]
pub fn calibrate_tsc() {
    use std::thread;

    // 1. AQUECIMENTO:
    // Chamamos a instrução uma vez e esperamos um pouco. Isso garante que a CPU
    // "acorde" de estados de baixo consumo e que os dados estejam prontos nos caches.
    let _ = unsafe { core::arch::x86_64::_rdtsc() };
    thread::sleep(Duration::from_millis(10));

    // 2. MARCO ZERO (Início da Medição):
    // Capturamos simultaneamente o tempo do relógio do sistema (lento mas confiável)
    // e o valor do contador de ciclos da CPU (ultra-rápido).
    let start_inst = Instant::now();
    let start_tsc = unsafe { core::arch::x86_64::_rdtsc() };

    // 3. ESPERA CONTROLADA:
    // Aguardamos 50 milissegundos. É um tempo curto para o humano, mas permite que
    // a CPU realize milhões de ciclos, reduzindo erros de amostragem.
    thread::sleep(Duration::from_millis(50));

    // 4. MARCO FINAL:
    // Capturamos novamente os dois valores para calcular quanto cada um avançou.
    let end_inst = Instant::now();
    let end_tsc = unsafe { core::arch::x86_64::_rdtsc() };

    let elapsed_nanos = end_inst.duration_since(start_inst).as_nanos() as u64;
    let elapsed_cycles = end_tsc.wrapping_sub(start_tsc);

    // 5. CÁLCULO DA TAXA DE CONVERSÃO:
    // Calculamos a relação: (Ciclos * 1000) / Nanosegundos.
    // Usamos o multiplicador 1000 para guardar o resultado como um número inteiro
    // mantendo 3 casas decimais de precisão sem precisar de ponto flutuante (floats).
    if let Some(freq_x1000) = (elapsed_cycles * 1000).checked_div(elapsed_nanos) {
        TSC_FREQ_GHZ_X1000.store(freq_x1000, Ordering::SeqCst);

        log::info!(
            "{} Relógio de Alta Precisão (TSC) calibrado em {:.3} GHz",
            "⏱️".bright_blue(),
            freq_x1000 as f64 / 1000.0
        );
    }
}
