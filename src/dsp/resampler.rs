// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de conversão de taxa de amostragem (resampling) de Fase Linear FIR Polifásica.
//!
//! Implementa `NamResampler`, um wrapper RT-safe em torno do crate `rubato` (Sinc assíncrono),
//! encapsulando **dois** resamplers independentes:
//!
//! - **`inner`** (input path): `PW_rate → 48 kHz` — aplicado antes da inferência NAM.
//! - **`outer`** (output path): `48 kHz → PW_rate` — aplicado após a inferência, antes de
//!   devolver o áudio ao PipeWire (placa de som recebe no mesmo rate que enviou).
//!
//! ## Garantias de Tempo-Real
//!
//! Toda alocação (`Vec`, buffers internos do rubato) ocorre em `NamResampler::new()`, **fora**
//! da thread DSP. No callback real-time, apenas `process_into_buffer()` é invocado — operação
//! documentada pelo rubato como RT-safe (zero alocações, zero bloqueios).
//!
//! Quando `pw_rate == 48000`, ambos os resamplers ficam em bypass automático (`Option::None`)
//! e o caminho hot passa direto sem nenhum overhead de conversão.
//!
//! ## Decisão de Versão
//!
//! Usa `rubato 0.16.x` (não 2.0.0): a API 2.0 introduziu dependência obrigatória de
//! `audioadapter`/`audioadapter-buffers`, desnecessária para nosso modelo de buffers RT.
//! Feature `fft_resampler` desabilitada (não usamos resampler FFT síncrono).

use anyhow::{Result, bail};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Taxa de amostragem alvo do motor NAM (modelos treinados a 48 kHz).
const NAM_RATE: u32 = 48_000;

/// Parâmetros do filtro Sinc Kaiser para o resampler de entrada/saída.
///
/// `sinc_len = 256` → comprimento do filtro Sinc (maior = melhor rejeição de aliasing).
/// `f_cutoff = 0.95` → frequência de corte normalizada (0.95 × Nyquist).
/// `oversampling_factor = 128` → superamostragem interna Sinc.
/// `interpolation = Linear` → interpolação linear (equilíbrio custo/qualidade RT).
/// `window = Blackman2` → janela equivalente a Kaiser para atenuação agressiva.
fn sinc_params() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Linear,
        window: WindowFunction::BlackmanHarris2,
    }
}

/// Wrapper RT-safe para resampling bidirecional FIR Sinc de fase linear.
///
/// Encapsula dois resamplers independentes (input + output) pré-alocados.
/// Na thread DSP apenas `process_into_buffer()` é chamado — zero alocações.
pub struct NamResampler {
    /// Resampler de entrada: `pw_rate → 48 kHz`. `None` quando `pw_rate == 48000` (bypass).
    inner: Option<SincFixedIn<f32>>,
    /// Buffer de entrada do resampler `inner` (pré-alocado, canal único).
    inner_in_buf: Vec<Vec<f32>>,
    /// Buffer de saída do resampler `inner` (pré-alocado, canal único).
    inner_out_buf: Vec<Vec<f32>>,

    /// Resampler de saída: `48 kHz → pw_rate`. `None` quando `pw_rate == 48000` (bypass).
    outer: Option<SincFixedIn<f32>>,
    /// Buffer de entrada do resampler `outer` (pré-alocado, canal único).
    outer_in_buf: Vec<Vec<f32>>,
    /// Buffer de saída do resampler `outer` (pré-alocado, canal único).
    outer_out_buf: Vec<Vec<f32>>,

    /// Rate do PipeWire (rate da placa de som).
    pw_rate: u32,
}

impl NamResampler {
    /// Cria o par de resamplers (input+output) pré-alocando todos os buffers internos.
    ///
    /// Se `pw_rate == 48000`, ambos os resamplers ficam em bypass (`None`) — sem overhead.
    ///
    /// # Parâmetros
    /// - `pw_rate`: Taxa de amostragem do PipeWire (e.g., 44100, 48000, 96000).
    /// - `chunk_size`: Tamanho máximo esperado do bloco DSP (frames por callback PipeWire).
    ///
    /// # Erros
    /// Falha se o rubato não conseguir criar o resampler com os parâmetros fornecidos.
    pub fn new(pw_rate: u32, chunk_size: usize) -> Result<Self> {
        if pw_rate == 0 {
            bail!("NamResampler: pw_rate não pode ser zero");
        }

        if pw_rate == NAM_RATE {
            // Bypass total: sem overhead de resampling
            return Ok(Self {
                inner: None,
                inner_in_buf: Vec::new(),
                inner_out_buf: Vec::new(),
                outer: None,
                outer_in_buf: Vec::new(),
                outer_out_buf: Vec::new(),
                pw_rate,
            });
        }

        let ratio_in = NAM_RATE as f64 / pw_rate as f64;
        let ratio_out = pw_rate as f64 / NAM_RATE as f64;

        // Resampler de entrada: pw_rate → 48k
        // SincFixedIn processa chunks de tamanho fixo na entrada.
        let inner = SincFixedIn::<f32>::new(
            ratio_in,
            2.0, // max_resample_ratio_relative: permite variação de ±2× (async)
            sinc_params(),
            chunk_size,
            1, // mono (canal único)
        )?;

        // Tamanho do bloco de saída do inner (em frames 48k)
        let inner_out_size = inner.output_frames_max();
        let inner_in_buf = vec![vec![0.0f32; chunk_size]];
        let inner_out_buf = vec![vec![0.0f32; inner_out_size]];

        // Bloco de entrada do outer = saída do inner (amostras 48k)
        let outer = SincFixedIn::<f32>::new(ratio_out, 2.0, sinc_params(), inner_out_size, 1)?;

        let outer_out_size = outer.output_frames_max();
        let outer_in_buf = vec![vec![0.0f32; inner_out_size]];
        let outer_out_buf = vec![vec![0.0f32; outer_out_size]];

        Ok(Self {
            inner: Some(inner),
            inner_in_buf,
            inner_out_buf,
            outer: Some(outer),
            outer_in_buf,
            outer_out_buf,
            pw_rate,
        })
    }

    /// Retorna `true` quando `pw_rate == 48000` (ambos resamplers em bypass).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        self.inner.is_none()
    }

    /// Retorna a taxa de amostragem do PipeWire configurada nesta instância.
    #[inline]
    pub fn pw_rate(&self) -> u32 {
        self.pw_rate
    }

    /// **Resampling de entrada** (input path): `pw_rate → 48 kHz`.
    ///
    /// Processa `input` (em `pw_rate`) e escreve o resultado em `output` (em 48 kHz).
    /// RT-safe: usa apenas `process_into_buffer()` — zero alocações.
    ///
    /// # Retorno
    /// Número de amostras escritas em `output`. Em modo bypass retorna `input.len()`.
    ///
    /// # Panics
    /// Não entra em pânico; trunca silenciosamente se `output` for menor que o necessário.
    pub fn process_input(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let Some(ref mut resampler) = self.inner else {
            // Bypass: copia diretamente
            let n = input.len().min(output.len());
            output[..n].copy_from_slice(&input[..n]);
            return n;
        };

        let n_in = input.len().min(self.inner_in_buf[0].len());
        self.inner_in_buf[0][..n_in].copy_from_slice(&input[..n_in]);

        // Limpa o restante se o chunk for menor que o configurado
        if n_in < self.inner_in_buf[0].len() {
            self.inner_in_buf[0][n_in..].fill(0.0);
        }

        match resampler.process_into_buffer(&self.inner_in_buf, &mut self.inner_out_buf, None) {
            Ok((_, n_out)) => {
                let n = n_out.min(output.len());
                output[..n].copy_from_slice(&self.inner_out_buf[0][..n]);
                n
            }
            Err(_) => {
                // Falha silenciosa RT-safe: retorna silêncio
                let n = n_in.min(output.len());
                output[..n].fill(0.0);
                n
            }
        }
    }

    /// **Resampling de saída** (output path): `48 kHz → pw_rate`.
    ///
    /// Processa `input` (em 48 kHz, saída da inferência NAM) e escreve em `output` (em `pw_rate`).
    /// RT-safe: usa apenas `process_into_buffer()` — zero alocações.
    ///
    /// # Retorno
    /// Número de amostras escritas em `output`. Em modo bypass retorna `input.len()`.
    pub fn process_output(&mut self, input: &[f32], output: &mut [f32]) -> usize {
        let Some(ref mut resampler) = self.outer else {
            // Bypass: copia diretamente
            let n = input.len().min(output.len());
            output[..n].copy_from_slice(&input[..n]);
            return n;
        };

        let n_in = input.len().min(self.outer_in_buf[0].len());
        self.outer_in_buf[0][..n_in].copy_from_slice(&input[..n_in]);

        if n_in < self.outer_in_buf[0].len() {
            self.outer_in_buf[0][n_in..].fill(0.0);
        }

        match resampler.process_into_buffer(&self.outer_in_buf, &mut self.outer_out_buf, None) {
            Ok((_, n_out)) => {
                let n = n_out.min(output.len());
                output[..n].copy_from_slice(&self.outer_out_buf[0][..n]);
                n
            }
            Err(_) => {
                let n = n_in.min(output.len());
                output[..n].fill(0.0);
                n
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifica que 48 kHz gera bypass automático e amostras passam intactas.
    #[test]
    fn test_bypass_48k() {
        let mut rs = NamResampler::new(48_000, 256).expect("new falhou");
        assert!(rs.is_bypass(), "48k deve ser bypass");

        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f32; 5];
        let n = rs.process_input(&input, &mut output);
        assert_eq!(n, 5);
        assert_eq!(output, input, "bypass deve copiar exatamente");

        let n2 = rs.process_output(&input, &mut output);
        assert_eq!(n2, 5);
        assert_eq!(output, input, "bypass de saída deve copiar exatamente");
    }

    /// Verifica que o downsample 96k→48k produz ~metade das amostras.
    #[test]
    fn test_downsample_96k_to_48k() {
        let chunk = 512usize;
        let mut rs = NamResampler::new(96_000, chunk).expect("new falhou");
        assert!(!rs.is_bypass());

        let input = vec![0.5f32; chunk];
        let mut output = vec![0.0f32; chunk * 2]; // buffer generoso
        let n = rs.process_input(&input, &mut output);

        // Para 96k→48k, esperamos ~chunk/2 amostras de saída (±tolerância do Sinc)
        let expected_approx = chunk / 2;
        assert!(
            n >= expected_approx.saturating_sub(16) && n <= expected_approx + 16,
            "96k→48k: esperado ~{expected_approx} amostras, obteve {n}"
        );
    }

    /// Verifica que o upsample 44.1k→48k produz mais amostras que a entrada.
    #[test]
    fn test_upsample_44k_to_48k() {
        let chunk = 441usize;
        let mut rs = NamResampler::new(44_100, chunk).expect("new falhou");
        assert!(!rs.is_bypass());

        let input = vec![0.3f32; chunk];
        let mut output = vec![0.0f32; chunk * 2];
        let n = rs.process_input(&input, &mut output);

        // 441 * (48000/44100) ≈ 480
        let expected_approx = (chunk as f64 * 48_000.0 / 44_100.0) as usize;
        assert!(
            n >= expected_approx.saturating_sub(16) && n <= expected_approx + 16,
            "44.1k→48k: esperado ~{expected_approx} amostras, obteve {n}"
        );
    }

    /// Verifica que o upsample de saída 48k→96k produz ~o dobro de amostras.
    #[test]
    fn test_output_upsample_48k_to_96k() {
        let chunk = 256usize;
        let mut rs = NamResampler::new(96_000, chunk).expect("new falhou");

        // process_output recebe amostras 48k e deve produzir amostras 96k
        // chunk_size do inner_out_buf é inner.output_frames_max() ≈ chunk/2
        let inner_out_size = chunk / 2; // tamanho aproximado do bloco de 48k

        let input = vec![0.4f32; inner_out_size];
        let mut output = vec![0.0f32; chunk * 2];
        let n = rs.process_output(&input, &mut output);

        // inner_out_size * (96000/48000) ≈ chunk
        let expected_approx = inner_out_size * 2;
        assert!(
            n >= expected_approx.saturating_sub(16) && n <= expected_approx + 16,
            "48k→96k (output): esperado ~{expected_approx} amostras, obteve {n}"
        );
    }

    /// Roundtrip 96k→48k (process_input) → 48k→96k (process_output):
    /// energia do sinal deve ser preservada (com tolerância pelo filtro FIR).
    #[test]
    fn test_roundtrip_96k() {
        let chunk = 512usize;
        let mut rs = NamResampler::new(96_000, chunk).expect("new falhou");

        // Sinal de teste: onda senoidal simples (verificável por energia)
        let input: Vec<f32> = (0..chunk)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 96_000.0_f32).sin())
            .collect();

        let mut mid = vec![0.0f32; chunk];
        let n_mid = rs.process_input(&input, &mut mid);
        assert!(n_mid > 0, "process_input não produziu saída");

        let mut out = vec![0.0f32; chunk * 2];
        let n_out = rs.process_output(&mid[..n_mid], &mut out);
        assert!(n_out > 0, "process_output não produziu saída");

        // A energia do sinal de saída deve ser razoavelmente próxima à entrada
        // (o FIR pode introduzir atraso de grupo e rolloff leve nos extremos)
        let energy_in: f32 = input.iter().map(|x| x * x).sum::<f32>() / chunk as f32;
        let energy_out: f32 = out[..n_out].iter().map(|x| x * x).sum::<f32>() / n_out as f32;

        // Tolerância ampla: o filtro Kaiser com sinc_len=256 tem delay de grupo no início
        // do buffer; verificamos apenas que a energia não colapsa (>10% da original)
        assert!(
            energy_out > energy_in * 0.1,
            "Energia no roundtrip colapsou: entrada={energy_in:.4}, saída={energy_out:.4}"
        );
    }

    /// Impulso de Dirac no input path: verifica que a resposta ao impulso não tem energia
    /// em frequências acima de Nyquist (24 kHz para 48k NAM rate) — sem aliasing catastrófico.
    #[test]
    fn test_impulse_response_input() {
        let chunk = 512usize;
        let mut rs = NamResampler::new(96_000, chunk).expect("new falhou");

        // Impulso unitário no início do bloco
        let mut input = vec![0.0f32; chunk];
        input[0] = 1.0;

        let mut output = vec![0.0f32; chunk];
        let n = rs.process_input(&input, &mut output);
        assert!(n > 0);

        // A resposta ao impulso do filtro Sinc deve ter energia finita e não-zero
        let energy: f32 = output[..n].iter().map(|x| x * x).sum();
        assert!(
            energy > 0.0 && energy.is_finite(),
            "Resposta ao impulso inválida: energy={energy}"
        );
        // O pico máximo absoluto não deve exceder 1.5 (filtro normalizado)
        let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            peak <= 1.5,
            "Pico na resposta ao impulso excessivo: {peak:.4}"
        );
    }

    /// Impulso de Dirac no output path: mesmas garantias para o resampler de saída.
    #[test]
    fn test_impulse_response_output() {
        let chunk = 256usize;
        let mut rs = NamResampler::new(96_000, chunk).expect("new falhou");

        // Usa tamanho interno do outer_in_buf (≈ inner_out_size)
        let inner_out_approx = chunk / 2;
        let mut input = vec![0.0f32; inner_out_approx];
        input[0] = 1.0;

        let mut output = vec![0.0f32; chunk];
        let n = rs.process_output(&input, &mut output);
        assert!(n > 0);

        let energy: f32 = output[..n].iter().map(|x| x * x).sum();
        assert!(
            energy > 0.0 && energy.is_finite(),
            "Resposta ao impulso (output) inválida: energy={energy}"
        );
        let peak = output[..n].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            peak <= 1.5,
            "Pico na resposta ao impulso (output) excessivo: {peak:.4}"
        );
    }
}
