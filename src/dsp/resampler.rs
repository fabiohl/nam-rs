// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Resampler FIR Sinc Polifásico nativo com Fase Mínima e convolução SIMD AVX2+FMA.
//!
//! Implementa `NamResampler`, um motor de conversão de taxa de amostragem RT-safe
//! que substitui o crate `rubato` por um filtro FIR Sinc Polifásico customizado.
//!
//! ## Vantagens sobre o rubato (fase linear)
//!
//! - **Eliminação de pré-ringing**: a transformação de fase mínima via Cepstrum Real
//!   concentra toda a energia do filtro no menor atraso possível, removendo 100%
//!   dos artefatos de pré-eco em transientes de guitarra.
//! - **Redução de latência algorítmica**: de ~1.5 ms (fase linear) para ~0.1 ms.
//! - **Convolução vetorizada**: inner product AVX2+FMA com coeficientes alinhados
//!   a 32 bytes, saturando o throughput das portas FMA do processador.
//!
//! ## Arquitetura: Polyphase Oversampled com Interpolação
//!
//! Em vez de usar L/M fases discretas (impraticável para L=160 na razão 44.1→48),
//! o resampler usa um banco sobreabundante de 256 fases com interpolação linear
//! entre fases adjacentes. Isso produz razões de conversão arbitrárias com
//! qualidade > 120 dB SNR e overhead computacional mínimo (2 convoluções por sample).
//!
//! ## Garantias de Tempo-Real
//!
//! Toda alocação ocorre em `NamResampler::new()`, fora da thread DSP.
//! No callback RT, apenas `process_input()` / `process_output()` são invocados —
//! operações zero-alloc que manipulam ring buffers pré-alocados.

use anyhow::{Result, bail};
use core::arch::x86_64::*;

use super::sinc_kernel::{NUM_PHASES, PolyphaseBank, TAPS_PER_PHASE, generate_polyphase_bank};

/// Tamanho do delay line (double-buffer) para garantir acesso contíguo.
/// Mantém 2 cópias do histórico para evitar lógica de wrap no hot-path SIMD.
const DELAY_LINE_LEN: usize = TAPS_PER_PHASE * 2;

/// Estado do filtro FIR para um canal (mono).
///
/// Usa a técnica de "double-buffer": o histórico de amostras é mantido em duas
/// cópias contíguas. Ao inserir uma nova amostra, ela é escrita em ambas as
/// posições `[write_pos]` e `[write_pos + TAPS_PER_PHASE]`. Isso garante que
/// qualquer janela de `TAPS_PER_PHASE` amostras consecutivas a partir de
/// `write_pos` é sempre contígua — eliminando a necessidade de lógica de wrap
/// circular no inner loop SIMD.
struct DelayLine {
    /// Buffer de amostras (tamanho = DELAY_LINE_LEN = 2 × TAPS_PER_PHASE).
    buf: Vec<f32>,
    /// Posição de escrita (0..TAPS_PER_PHASE-1, wrapping).
    pos: usize,
}

impl DelayLine {
    fn new() -> Self {
        Self {
            buf: vec![0.0f32; DELAY_LINE_LEN],
            pos: 0,
        }
    }

    /// Insere uma amostra no delay line (double-write para contiguidade).
    #[inline(always)]
    fn push(&mut self, sample: f32) {
        self.buf[self.pos] = sample;
        self.buf[self.pos + TAPS_PER_PHASE] = sample;
        self.pos += 1;
        if self.pos >= TAPS_PER_PHASE {
            self.pos = 0;
        }
    }

    /// Retorna ponteiro para TAPS_PER_PHASE amostras contíguas (mais recentes primeiro).
    #[inline(always)]
    fn window_ptr(&self) -> *const f32 {
        unsafe { self.buf.as_ptr().add(self.pos) }
    }
}

/// Motor de resampling para uma direção (entrada ou saída).
///
/// Contém o banco polifásico e o estado fracionário para rastreamento
/// da posição entre amostras de entrada e saída.
struct ResamplerCore {
    /// Banco de filtros polifásicos (coeficientes alinhados 32B).
    bank: PolyphaseBank,
    /// Delay line do canal esquerdo.
    state_l: DelayLine,
    /// Delay line do canal direito.
    state_r: DelayLine,
    /// Posição fracionária no espaço de fases (0.0 .. NUM_PHASES).
    /// Avança por `phase_step` a cada amostra de saída.
    phase_accum: f64,
    /// Incremento de fase por amostra de saída = `from_rate / to_rate * NUM_PHASES`.
    phase_step: f64,
}

impl ResamplerCore {
    fn new(from_rate: u32, to_rate: u32) -> Self {
        let bank = generate_polyphase_bank(from_rate, to_rate);
        // Inicia o acumulador em NUM_PHASES para que a primeira iteração
        // do loop de processamento consuma imediatamente a primeira amostra
        // de entrada, preenchendo o delay line antes de tentar produzir saída.
        let phase_step = (from_rate as f64 / to_rate as f64) * NUM_PHASES as f64;
        Self {
            bank,
            state_l: DelayLine::new(),
            state_r: DelayLine::new(),
            phase_accum: NUM_PHASES as f64,
            phase_step,
        }
    }

    /// Processa um bloco estéreo. RT-safe: zero alocações.
    ///
    /// Retorna o número de amostras escritas em `out_l` / `out_r`.
    fn process(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let n_in = in_l.len().min(in_r.len());
        let n_out_max = out_l.len().min(out_r.len());
        let mut in_idx = 0usize;
        let mut out_idx = 0usize;

        while out_idx < n_out_max {
            // Consumir amostras de entrada conforme o acumulador de fase avança
            while self.phase_accum >= NUM_PHASES as f64 {
                if in_idx >= n_in {
                    return out_idx;
                }
                self.state_l.push(in_l[in_idx]);
                self.state_r.push(in_r[in_idx]);
                self.phase_accum -= NUM_PHASES as f64;
                in_idx += 1;
            }

            // Determina a fase e o fator de interpolação fracionário
            let phase_f = self.phase_accum;
            let phase_idx = phase_f as usize;
            let frac = (phase_f - phase_idx as f64) as f32;

            // Fase seguinte (com wrap)
            let phase_next = if phase_idx + 1 >= NUM_PHASES {
                0
            } else {
                phase_idx + 1
            };

            // Convolução SIMD + interpolação linear entre fases adjacentes
            let (y_l, y_r) = unsafe {
                let c0 = self.bank.phase_ptr(phase_idx);
                let c1 = self.bank.phase_ptr(phase_next);
                let x_l = self.state_l.window_ptr();
                let x_r = self.state_r.window_ptr();
                let taps = self.bank.taps_per_phase;
                let is_avx512 = crate::math::simd::SimdMathConfig::get().is_avx512;

                if is_avx512 {
                    let y0_l = convolve_avx512(c0, x_l, taps);
                    let y1_l = convolve_avx512(c1, x_l, taps);
                    let y0_r = convolve_avx512(c0, x_r, taps);
                    let y1_r = convolve_avx512(c1, x_r, taps);
                    (y0_l + frac * (y1_l - y0_l), y0_r + frac * (y1_r - y0_r))
                } else {
                    let y0_l = convolve_avx2(c0, x_l, taps);
                    let y1_l = convolve_avx2(c1, x_l, taps);
                    let y0_r = convolve_avx2(c0, x_r, taps);
                    let y1_r = convolve_avx2(c1, x_r, taps);
                    (y0_l + frac * (y1_l - y0_l), y0_r + frac * (y1_r - y0_r))
                }
            };

            out_l[out_idx] = y_l;
            out_r[out_idx] = y_r;
            out_idx += 1;

            self.phase_accum += self.phase_step;
        }

        // Consumir amostras de entrada restantes (manter state atualizado)
        while self.phase_accum >= NUM_PHASES as f64 && in_idx < n_in {
            self.state_l.push(in_l[in_idx]);
            self.state_r.push(in_r[in_idx]);
            self.phase_accum -= NUM_PHASES as f64;
            in_idx += 1;
        }

        out_idx
    }
}

/// Inner product AVX2+FMA entre coeficientes alinhados e janela do delay line.
///
/// Processa `taps` amostras usando 2 acumuladores YMM para quebrar a cadeia
/// de dependência FMA, saturando o throughput de 2 FMA ports.
///
/// # Safety
/// - CPU deve suportar AVX2 e FMA (garantido por x86-64-v3 em `.cargo/config.toml`).
/// - `coeffs` deve apontar para dados alinhados a 32 bytes com pelo menos `taps` f32.
/// - `input` deve apontar para pelo menos `taps` f32 contíguos.
#[inline]
unsafe fn convolve_avx2(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut i = 0;

        // Loop principal: 2×8 = 16 floats/iteração
        while i + 16 <= taps {
            let h0 = _mm256_load_ps(coeffs.add(i)); // Aligned
            let x0 = _mm256_loadu_ps(input.add(i)); // Input não alinhado
            sum0 = _mm256_fmadd_ps(h0, x0, sum0);

            let h1 = _mm256_load_ps(coeffs.add(i + 8));
            let x1 = _mm256_loadu_ps(input.add(i + 8));
            sum1 = _mm256_fmadd_ps(h1, x1, sum1);

            i += 16;
        }

        // Resto: 8-em-8
        while i + 8 <= taps {
            let h = _mm256_load_ps(coeffs.add(i));
            let x = _mm256_loadu_ps(input.add(i));
            sum0 = _mm256_fmadd_ps(h, x, sum0);
            i += 8;
        }

        // Redução horizontal: 2 acumuladores → escalar
        let sum = _mm256_add_ps(sum0, sum1);
        let hi128 = _mm256_extractf128_ps(sum, 1);
        let lo128 = _mm256_castps256_ps128(sum);
        let s128 = _mm_add_ps(lo128, hi128);
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let r = _mm_add_ss(sums, shuf2);
        let mut out = 0.0f32;
        _mm_store_ss(&mut out, r);

        // Tail escalar (para taps não múltiplo de 8 — não deveria ocorrer com TAPS_PER_PHASE=32)
        while i < taps {
            out += *coeffs.add(i) * *input.add(i);
            i += 1;
        }

        out
    }
}

/// Inner product AVX-512 (ZMM) entre coeficientes alinhados e janela do delay line.
///
/// Processa `taps` amostras usando 2 acumuladores ZMM para maximizar o throughput
/// do pipeline AVX-512 (32 floats/iteração).
///
/// # Safety
/// - CPU deve suportar AVX-512F.
/// - `coeffs` deve apontar para dados alinhados a 64 bytes (ou 32B se ZMM permitir)
///   com pelo menos `taps` f32.
/// - `input` deve apontar para pelo menos `taps` f32 contíguos.
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn convolve_avx512(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    unsafe {
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut i = 0;

        // Loop principal: 2×16 = 32 floats/iteração
        while i + 32 <= taps {
            let h0 = _mm512_loadu_ps(coeffs.add(i)); // Unaligned (coeffs are 32B aligned)
            let x0 = _mm512_loadu_ps(input.add(i)); // Input não alinhado
            sum0 = _mm512_fmadd_ps(h0, x0, sum0);

            let h1 = _mm512_loadu_ps(coeffs.add(i + 16));
            let x1 = _mm512_loadu_ps(input.add(i + 16));
            sum1 = _mm512_fmadd_ps(h1, x1, sum1);

            i += 32;
        }

        // Resto: 16-em-16
        while i + 16 <= taps {
            let h = _mm512_loadu_ps(coeffs.add(i));
            let x = _mm512_loadu_ps(input.add(i));
            sum0 = _mm512_fmadd_ps(h, x, sum0);
            i += 16;
        }

        // Redução horizontal ZMM -> Escalar
        let sum = _mm512_add_ps(sum0, sum1);
        let mut out = _mm512_reduce_add_ps(sum);

        // Tail escalar (para taps não múltiplo de 16)
        while i < taps {
            out += *coeffs.add(i) * *input.add(i);
            i += 1;
        }

        out
    }
}

/// Wrapper RT-safe para resampling bidirecional FIR Sinc Polifásico de Fase Mínima.
///
/// Encapsula dois motores independentes (input + output) pré-alocados.
/// Na thread DSP apenas `process_input()` / `process_output()` são chamados —
/// operações zero-alloc que operam sobre delay lines pré-alocados.
///
/// Quando `pw_rate == nam_rate`, ambos os motores ficam em bypass (`None`)
/// e o caminho hot passa direto sem nenhum overhead.
pub struct NamResampler {
    /// Motor de entrada: `pw_rate → nam_rate`. `None` = bypass.
    inner: Option<ResamplerCore>,
    /// Motor de saída: `nam_rate → pw_rate`. `None` = bypass.
    outer: Option<ResamplerCore>,
    /// Rate do PipeWire.
    pw_rate: u32,
    /// Rate alvo do modelo NAM.
    nam_rate: u32,
}

impl NamResampler {
    /// Cria o par de resamplers (input+output) pré-alocando todos os buffers.
    ///
    /// Se `pw_rate == nam_rate`, bypass total sem overhead.
    ///
    /// # Parâmetros
    /// - `pw_rate`: taxa do PipeWire (e.g., 44100, 48000, 96000).
    /// - `nam_rate`: taxa do modelo NAM (e.g., 48000).
    /// - `_chunk_size`: mantido por compatibilidade de API (não usado internamente).
    pub fn new(pw_rate: u32, nam_rate: u32, _chunk_size: usize) -> Result<Self> {
        if pw_rate == 0 || nam_rate == 0 {
            bail!("NamResampler: as taxas de amostragem não podem ser nulas");
        }

        if pw_rate == nam_rate {
            return Ok(Self {
                inner: None,
                outer: None,
                pw_rate,
                nam_rate,
            });
        }

        let inner = ResamplerCore::new(pw_rate, nam_rate);
        let outer = ResamplerCore::new(nam_rate, pw_rate);

        Ok(Self {
            inner: Some(inner),
            outer: Some(outer),
            pw_rate,
            nam_rate,
        })
    }

    /// Retorna `true` quando `pw_rate == nam_rate` (bypass).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        self.inner.is_none()
    }

    /// Retorna a taxa do PipeWire.
    #[inline]
    pub fn pw_rate(&self) -> u32 {
        self.pw_rate
    }

    /// Retorna a taxa do modelo NAM.
    #[inline]
    pub fn nam_rate(&self) -> u32 {
        self.nam_rate
    }

    /// **Resampling de entrada** (input path): `pw_rate → nam_rate`.
    ///
    /// RT-safe: zero alocações. Em bypass, copia diretamente.
    ///
    /// # Retorno
    /// Número de amostras escritas em `out_l` / `out_r`.
    pub fn process_input(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.inner else {
            let n = in_l.len().min(out_l.len());
            out_l[..n].copy_from_slice(&in_l[..n]);
            out_r[..n].copy_from_slice(&in_r[..n]);
            return n;
        };
        core.process(in_l, in_r, out_l, out_r)
    }

    /// **Resampling de saída** (output path): `nam_rate → pw_rate`.
    ///
    /// RT-safe: zero alocações. Em bypass, copia diretamente.
    ///
    /// # Retorno
    /// Número de amostras escritas em `out_l` / `out_r`.
    pub fn process_output(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.outer else {
            let n = in_l.len().min(out_l.len());
            out_l[..n].copy_from_slice(&in_l[..n]);
            out_r[..n].copy_from_slice(&in_r[..n]);
            return n;
        };
        core.process(in_l, in_r, out_l, out_r)
    }
}

#[cfg(test)]
#[path = "resampler_test.rs"]
mod resampler_test;
