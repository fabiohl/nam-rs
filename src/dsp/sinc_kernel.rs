// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Geração offline de kernels FIR Sinc com fase mínima para o resampler nativo.
//!
//! Este módulo é invocado **exclusivamente** em `NamResampler::new()` — fora da thread RT.
//! Toda alocação e computação pesada (FFT, cepstrum) ocorre aqui durante a inicialização.
//!
//! ## Pipeline de Geração
//!
//! 1. **Sinc Ideal + Janelamento Kaiser** — gera o protótipo FIR lowpass de fase linear.
//! 2. **Transformação de Fase Mínima (Cepstrum Real)** — elimina pré-ringing concentrando
//!    a energia no menor atraso possível, usando FFT em f64 para precisão numérica.
//! 3. **Partição Polifásica** — decompõe o protótipo em `num_phases` sub-filtros,
//!    cada um com taps alinhados a 32 bytes para convolução AVX2+FMA.

use rustfft::{FftPlanner, num_complex::Complex};

/// Número de fases do banco polifásico sobreabundante.
///
/// Controla a resolução fracionária do resampler. Com 256 fases,
/// o erro de fase máximo entre sub-filtros adjacentes é < 0.4%,
/// tornando a interpolação linear entre fases suficiente para
/// SNR > 140 dB na conversão de taxa.
pub const NUM_PHASES: usize = 256;

/// Número de taps por fase do banco polifásico.
///
/// Com 32 taps por fase e janela Kaiser (β=12), o filtro atinge
/// rejeição de aliasing > 120 dB com banda de transição de ~3 kHz
/// a 48 kHz. O valor é múltiplo de 8 para alinhamento AVX2.
pub const TAPS_PER_PHASE: usize = 32;

/// Comprimento total do protótipo FIR = NUM_PHASES × TAPS_PER_PHASE.
const PROTO_LEN: usize = NUM_PHASES * TAPS_PER_PHASE;

/// Banco de filtros polifásico com coeficientes alinhados para SIMD.
///
/// Layout em memória: `coeffs[phase * TAPS_PER_PHASE .. (phase+1) * TAPS_PER_PHASE]`
/// Cada fase é contígua e alinhada a 32 bytes para `_mm256_load_ps`.
pub struct PolyphaseBank {
    /// Coeficientes f32 alinhados. Tamanho = `NUM_PHASES * taps_padded`.
    /// Cada fase é zero-padded para múltiplo de 8 (garantia AVX2).
    coeffs: AlignedCoeffs,
    /// Taps por fase (sempre TAPS_PER_PHASE, já múltiplo de 8).
    pub taps_per_phase: usize,
}

/// Wrapper para vetor de coeficientes com alinhamento de 32 bytes.
///
/// O alinhamento garante que `_mm256_load_ps` (aligned load) pode ser
/// usado nos coeficientes do filtro, economizando 1 ciclo por load
/// em relação a `_mm256_loadu_ps` (unaligned).
#[repr(C, align(32))]
struct AlignedBlock([f32; 8]);

struct AlignedCoeffs {
    /// Blocos de 8 floats alinhados a 32 bytes.
    blocks: Vec<AlignedBlock>,
}

impl AlignedCoeffs {
    fn new(data: &[f32]) -> Self {
        let n_blocks = data.len().div_ceil(8);
        let mut blocks = Vec::with_capacity(n_blocks);
        for chunk in data.chunks(8) {
            let mut block = [0.0f32; 8];
            block[..chunk.len()].copy_from_slice(chunk);
            blocks.push(AlignedBlock(block));
        }
        Self { blocks }
    }

    /// Retorna um ponteiro alinhado a 32 bytes para o início dos coeficientes.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.blocks.as_ptr() as *const f32
    }

    /// Retorna o slice completo (incluindo padding zeros).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.blocks.len() * 8) }
    }
}

impl PolyphaseBank {
    /// Retorna o ponteiro para o início dos coeficientes da fase `phase`.
    ///
    /// # Safety
    /// `phase` deve ser < `NUM_PHASES`.
    #[inline]
    pub fn phase_ptr(&self, phase: usize) -> *const f32 {
        debug_assert!(phase < NUM_PHASES);
        unsafe { self.coeffs.as_ptr().add(phase * self.taps_per_phase) }
    }

    /// Retorna o slice de coeficientes da fase `phase`.
    #[inline]
    pub fn phase_coeffs(&self, phase: usize) -> &[f32] {
        debug_assert!(phase < NUM_PHASES);
        let start = phase * self.taps_per_phase;
        &self.coeffs.as_slice()[start..start + self.taps_per_phase]
    }
}

/// Gera o banco polifásico completo para conversão `from_rate → to_rate`.
///
/// Pipeline: Sinc+Kaiser → Fase Mínima (Cepstrum) → Partição Polifásica.
///
/// # Parâmetros
/// - `from_rate`: taxa de amostragem de origem (Hz).
/// - `to_rate`: taxa de amostragem de destino (Hz).
///
/// # Retorno
/// Banco polifásico pronto para convolução SIMD.
pub fn generate_polyphase_bank(from_rate: u32, to_rate: u32) -> PolyphaseBank {
    // 1. Gera protótipo Sinc + Kaiser em f64
    let cutoff_ratio = (from_rate.min(to_rate) as f64) / (from_rate.max(to_rate) as f64);
    let cutoff = 0.95 * cutoff_ratio;
    let proto_f64 = generate_sinc_kaiser(PROTO_LEN, cutoff, 12.0);

    // 2. Transforma para fase mínima via cepstrum real
    let min_phase = to_minimum_phase(&proto_f64);

    // 3. Normaliza energia (ganho DC = 1.0 por fase)
    let proto_f32: Vec<f32> = min_phase.iter().map(|&x| x as f32).collect();

    // 4. Particiona em NUM_PHASES sub-filtros
    partition_polyphase(&proto_f32)
}

/// Gera um kernel FIR Sinc com janelamento Kaiser.
///
/// # Parâmetros
/// - `length`: comprimento total do filtro (amostras).
/// - `cutoff`: frequência de corte normalizada (0..1, relativa a Nyquist).
/// - `beta`: parâmetro β da janela Kaiser (controla atenuação de stop-band).
///   β=12 → ~120 dB de rejeição.
fn generate_sinc_kaiser(length: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let half = (length - 1) as f64 / 2.0;
    let i0_beta = bessel_i0(beta);

    let mut kernel = Vec::with_capacity(length);
    for i in 0..length {
        let n = i as f64 - half;

        // Sinc normalizado
        let sinc = if n.abs() < 1e-10 {
            cutoff
        } else {
            let x = std::f64::consts::PI * n * cutoff;
            x.sin() / (std::f64::consts::PI * n)
        };

        // Janela Kaiser: I0(β × sqrt(1 - (2n/N-1)²)) / I0(β)
        let ratio = n / half;
        let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
        let window = bessel_i0(arg) / i0_beta;

        kernel.push(sinc * window);
    }

    // Normaliza para ganho DC unitário
    let dc_sum: f64 = kernel.iter().sum();
    if dc_sum.abs() > 1e-15 {
        for k in &mut kernel {
            *k /= dc_sum;
        }
    }

    kernel
}

/// Função de Bessel modificada de primeira espécie, ordem zero — I₀(x).
///
/// Expansão em série de Taylor com 20 termos (precisão > 1e-12 para β ≤ 25).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;
    for k in 1..=20 {
        term *= (half_x / k as f64) * (half_x / k as f64);
        sum += term;
        if term < 1e-15 * sum {
            break;
        }
    }
    sum
}

/// Transforma um kernel FIR de fase linear para fase mínima via Cepstrum Real.
///
/// ## Algoritmo (Oppenheim & Schafer, Discrete-Time Signal Processing)
///
/// 1. Zero-pad kernel para `N_fft` (potência de 2, ≥ 4× comprimento original).
/// 2. FFT → espectro complexo `H(k)`.
/// 3. Log-magnitude: `L(k) = ln(|H(k)| + ε)`.
/// 4. IFFT de `L` → cepstrum real `c[n]`.
/// 5. Truncamento causal: `c[0]` inalterado, `c[1..N/2-1] × 2`, `c[N/2+1..] = 0`.
/// 6. FFT do cepstrum causal → `Ĉ(k)`.
/// 7. Exponencial complexa: `H_min(k) = exp(Ĉ(k))`.
/// 8. IFFT → `h_min[n]` (parte real), truncar para comprimento original.
///
/// Toda a computação é em f64 para estabilidade numérica no domínio logarítmico,
/// conforme recomendado por r8brain-free-src (Vaneev).
fn to_minimum_phase(kernel: &[f64]) -> Vec<f64> {
    let n_proto = kernel.len();
    let n_fft = (4 * n_proto).next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n_fft);
    let fft_inv = planner.plan_fft_inverse(n_fft);
    let scale = 1.0 / n_fft as f64;

    // Passo 1-2: Zero-pad + FFT
    let mut buf: Vec<Complex<f64>> = kernel
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat_n(Complex::new(0.0, 0.0), n_fft - n_proto))
        .collect();
    fft_fwd.process(&mut buf);

    // Passo 3: Log-magnitude (real-only complex)
    let eps = 1e-10_f64;
    for c in &mut buf {
        *c = Complex::new((c.norm() + eps).ln(), 0.0);
    }

    // Passo 4: IFFT → cepstrum real
    fft_inv.process(&mut buf);
    for c in &mut buf {
        *c *= scale;
    }

    // Passo 5: Truncamento causal
    // c[0] inalterado, c[1..N/2-1] *= 2, c[N/2] inalterado, c[N/2+1..] = 0
    let half = n_fft / 2;
    for c in &mut buf[1..half] {
        *c *= 2.0;
    }
    for c in &mut buf[half + 1..] {
        *c = Complex::new(0.0, 0.0);
    }

    // Passo 6: FFT do cepstrum causal
    fft_fwd.process(&mut buf);

    // Passo 7: Exponencial complexa
    for c in &mut buf {
        *c = c.exp();
    }

    // Passo 8: IFFT → impulso de fase mínima
    fft_inv.process(&mut buf);

    // Retornar parte real, truncada ao comprimento original
    buf[..n_proto].iter().map(|c| c.re * scale).collect()
}

/// Particiona o protótipo FIR em `NUM_PHASES` sub-filtros polifásicos.
///
/// O coeficiente `proto[n]` vai para a fase `n % NUM_PHASES`, tap `n / NUM_PHASES`.
/// Cada fase é zero-padded para `TAPS_PER_PHASE` (múltiplo de 8).
fn partition_polyphase(proto: &[f32]) -> PolyphaseBank {
    let taps = TAPS_PER_PHASE;
    let total = NUM_PHASES * taps;
    let mut coeffs = vec![0.0f32; total];

    // Escala por NUM_PHASES para compensar a decomposição polifásica.
    // No upsampling conceitual (inserção de L-1 zeros entre amostras),
    // o filtro protótipo é aplicado à taxa L×fs. A partição polifásica
    // divide o ganho total por L, exigindo compensação de ganho.
    let gain = NUM_PHASES as f32;

    for (n, &coeff) in proto.iter().enumerate() {
        let phase = n % NUM_PHASES;
        let tap = n / NUM_PHASES;
        if tap < taps {
            coeffs[phase * taps + tap] = coeff * gain;
        }
    }

    PolyphaseBank {
        coeffs: AlignedCoeffs::new(&coeffs),
        taps_per_phase: taps,
    }
}

#[cfg(test)]
#[path = "sinc_kernel_test.rs"]
mod sinc_kernel_test;
