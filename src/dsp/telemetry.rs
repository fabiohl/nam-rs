// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Telemetria de latência em tempo real para o pipeline DSP.

use std::sync::atomic::{AtomicU64, Ordering};

/// Histograma exponencial para rastrear a distribuição de latências do callback RT.
///
/// Possui 32 bins cobrindo potências de 2 de 2^5 ns (~32 ns) até 2^36 ns (~68 s).
/// Utiliza operações atômicas para garantir RT-safety (lock-free, zero-allocation).
#[repr(align(128))]
pub struct LatencyHistogram {
    /// Bins atômicos do histograma.
    bins: [AtomicU64; 32],
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    /// Cria um novo histograma zerado.
    #[cold]
    pub fn new() -> Self {
        Self {
            bins: [const { AtomicU64::new(0) }; 32],
        }
    }

    /// Registra uma observação de latência em nanosegundos.
    ///
    /// # RT-Safety
    /// Esta função é segura para chamadas no hot-path DSP.
    #[inline(always)]
    pub fn record(&self, duration_ns: u64) {
        if duration_ns == 0 {
            self.bins[0].fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Calcula o índice do bin via log2 aproximado (LZCNT).
        // 2^5 ns (32ns) -> bin 0
        // 2^36 ns (68s) -> bin 31
        let msb = 63 - duration_ns.leading_zeros();
        let index = msb.saturating_sub(5).min(31) as usize;

        self.bins[index].fetch_add(1, Ordering::Relaxed);
    }

    /// Calcula o valor aproximado de um percentil (ex: 0.95 para P95).
    /// Retorna o valor em nanosegundos da borda superior do bin.
    pub fn get_percentile(&self, p: f32) -> u64 {
        let counts: [u64; 32] = core::array::from_fn(|i| self.bins[i].load(Ordering::Relaxed));
        let total: u64 = counts.iter().sum();

        if total == 0 {
            return 0;
        }

        let target = (total as f64 * p as f64) as u64;
        let mut accum: u64 = 0;

        for (i, &count) in counts.iter().enumerate() {
            accum += count;
            if accum >= target {
                // Retorna 2^(i + 5)
                return 1u64 << (i + 5);
            }
        }

        1u64 << 36
    }

    /// Retorna o valor máximo observado (borda superior do bin mais alto não vazio).
    pub fn get_max(&self) -> u64 {
        for i in (0..32).rev() {
            if self.bins[i].load(Ordering::Relaxed) > 0 {
                return 1u64 << (i + 5);
            }
        }
        0
    }

    /// Retorna o número total de observações registradas desde o último reset.
    pub fn total_count(&self) -> u64 {
        self.bins.iter().map(|b| b.load(Ordering::Relaxed)).sum()
    }

    /// Zera todos os bins do histograma.
    pub fn reset(&self) {
        for bin in &self.bins {
            bin.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_mapping() {
        // Teste de Organização: Verifica se o sistema guarda os tempos de processamento
        // nas "gavetas" (bins) corretas, separando o que é ultra-rápido do que é lento.
        let hist = LatencyHistogram::new();

        hist.record(10); // bin 0 (sub-32ns)
        hist.record(40); // bin 0 (2^5 = 32)
        hist.record(100); // bin 1 (2^6 = 64)
        hist.record(1000); // bin 4 (2^9 = 512)

        assert_eq!(hist.bins[0].load(Ordering::Relaxed), 2);
        assert_eq!(hist.bins[1].load(Ordering::Relaxed), 1);
        assert_eq!(hist.bins[4].load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_percentiles() {
        // Teste de Estatística: Verifica se o sistema consegue identificar corretamente
        // a "mediana" (P50) e os "piores casos" (P95 e P99) em um lote de medições.
        let hist = LatencyHistogram::new();

        // 100 amostras
        for _ in 0..50 {
            hist.record(100);
        } // P50 deve estar no bin 1 (2^6 = 64)
        for _ in 0..45 {
            hist.record(1000);
        } // P95 deve estar no bin 4 (2^9 = 512)
        for _ in 0..5 {
            hist.record(10000);
        } // P99 deve estar no bin 8 (2^13 = 8192)

        assert!(hist.get_percentile(0.50) <= 64);
        assert!(hist.get_percentile(0.95) <= 512);
        assert!(hist.get_percentile(0.99) <= 16384);
    }
}
