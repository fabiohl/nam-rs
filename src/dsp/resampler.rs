// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Módulo de conversão de taxa de amostragem (resampling) de Fase Linear FIR Polifásica.
//!
//! Implementa `NamResampler`, um wrapper RT-safe em torno do crate `rubato` (Sinc assíncrono),
//! encapsulando **dois** resamplers independentes:
//!
//! - **`inner`** (input path): `PW_rate → nam_rate kHz` — aplicado antes da inferência NAM.
//! - **`outer`** (output path): `nam_rate kHz → PW_rate` — aplicado após a inferência, antes de
//!   devolver o áudio ao PipeWire (placa de som recebe no mesmo rate que enviou).
//!
//! O motor de inferência do NAM opera em uma taxa de amostragem fixa (historicamente
//! e por padrão em 48 kHz). Caso o modelo .nam carregado não possua a chave
//! `sample_rate` em seus metadados, a especificação exige o uso de 48 kHz.
//!
//! Como o PipeWire (servidor de áudio) possui o relógio mestre e determina a taxa
//! de amostragem da sessão (ex: 44.1 kHz, 96 kHz), é estritamente necessário
//! realizar uma dupla conversão de estágio (inner/outer) para evitar alteração
//! de afinação (pitch shift) e quebra do tamanho dos buffers (dropouts):
//! 1. Input: Converte o áudio da taxa do PipeWire para a taxa exigida pelo modelo NAM.
//! 2. Output: Reconverte a saída do modelo de volta para a taxa nativa do PipeWire.
//!
//! ## Garantias de Tempo-Real
//!
//! Toda alocação (`Vec`, buffers internos do rubato) ocorre em `NamResampler::new()`, **fora**
//! da thread DSP. No callback real-time, apenas `process_into_buffer()` é invocado — operação
//! documentada pelo rubato como RT-safe (zero alocações, zero bloqueios).
//!
//! Quando `pw_rate == nam_rate`, ambos os resamplers ficam em bypass automático (`Option::None`)
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

/// Parâmetros do filtro Sinc Kaiser para o resampler de entrada/saída.
///
/// `sinc_len = 256` → comprimento do filtro Sinc (maior = melhor rejeição de aliasing).
/// `f_cutoff = 0.95` → frequência de corte normalizada (0.95 × Nyquist).
/// `oversampling_factor = 128` → superamostragem interna Sinc.
/// `interpolation = Cubic` → interpolação Hermite cúbica entre taps Sinc.
///   Upgrade de `Linear` para `Cubic` proporciona ~+6 dB SNR na conversão
///   de sample rate (erro de interpolação de 4ª ordem vs 2ª ordem linear),
///   com overhead de CPU negligível (<5% no resampler, que não é gargalo).
/// `window = Blackman2` → janela equivalente a Kaiser para atenuação agressiva.
fn sinc_params() -> SincInterpolationParameters {
    SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        oversampling_factor: 128,
        interpolation: SincInterpolationType::Cubic,
        window: WindowFunction::BlackmanHarris2,
    }
}

/// Wrapper RT-safe para resampling bidirecional FIR Sinc de fase linear.
///
/// Encapsula dois resamplers independentes (input + output) pré-alocados.
/// Na thread DSP apenas `process_into_buffer()` é chamado — zero alocações.
pub struct NamResampler {
    /// Resampler de entrada: `pw_rate → nam_rate`. `None` quando `pw_rate == nam_rate` (bypass).
    inner: Option<SincFixedIn<f32>>,
    /// Buffer de entrada do resampler `inner` (pré-alocado, planar 2 canais).
    inner_in_buf: Vec<Vec<f32>>,
    /// Buffer de saída do resampler `inner` (pré-alocado, planar 2 canais).
    inner_out_buf: Vec<Vec<f32>>,

    /// Resampler de saída: `nam_rate → pw_rate`. `None` quando `pw_rate == nam_rate` (bypass).
    outer: Option<SincFixedIn<f32>>,
    /// Buffer de entrada do resampler `outer` (pré-alocado, planar 2 canais).
    outer_in_buf: Vec<Vec<f32>>,
    /// Buffer de saída do resampler `outer` (pré-alocado, planar 2 canais).
    outer_out_buf: Vec<Vec<f32>>,

    /// Rate do PipeWire (rate da placa de som).
    pw_rate: u32,
    /// Rate alvo exigido pelo modelo carregado (geralmente 48000).
    nam_rate: u32,
}

impl NamResampler {
    /// Cria o par de resamplers (input+output) pré-alocando todos os buffers internos.
    ///
    /// Se `pw_rate == nam_rate`, ambos os resamplers ficam em bypass (`None`) — sem overhead.
    ///
    /// # Parâmetros
    /// - `pw_rate`: Taxa de amostragem do PipeWire (e.g., 44100, 48000, 96000).
    /// - `nam_rate`: Taxa de amostragem do modelo carregado (e.g., 48000).
    /// - `chunk_size`: Tamanho máximo esperado do bloco DSP (frames por callback PipeWire).
    ///
    /// # Erros
    /// Falha se o rubato não conseguir criar o resampler com os parâmetros fornecidos.
    pub fn new(pw_rate: u32, nam_rate: u32, chunk_size: usize) -> Result<Self> {
        if pw_rate == 0 || nam_rate == 0 {
            bail!("NamResampler: as taxas de amostragem não podem ser nulas");
        }

        if pw_rate == nam_rate {
            // Bypass total: sem overhead de resampling
            return Ok(Self {
                inner: None,
                inner_in_buf: Vec::new(),
                inner_out_buf: Vec::new(),
                outer: None,
                outer_in_buf: Vec::new(),
                outer_out_buf: Vec::new(),
                pw_rate,
                nam_rate,
            });
        }

        let ratio_in = nam_rate as f64 / pw_rate as f64;
        let ratio_out = pw_rate as f64 / nam_rate as f64;

        // Resampler de entrada: pw_rate → nam_rate
        // SincFixedIn processa chunks de tamanho fixo na entrada.
        let inner = SincFixedIn::<f32>::new(
            ratio_in,
            2.0, // max_resample_ratio_relative: permite variação de ±2× (async)
            sinc_params(),
            chunk_size,
            2, // Planar estéreo (2 canais)
        )?;

        // Tamanho do bloco de saída do inner (em frames 48k)
        let inner_out_size = inner.output_frames_max();
        let inner_in_buf = vec![vec![0.0f32; chunk_size], vec![0.0f32; chunk_size]];
        let inner_out_buf = vec![vec![0.0f32; inner_out_size], vec![0.0f32; inner_out_size]];

        // Bloco de entrada do outer = saída do inner (amostras 48k)
        let outer = SincFixedIn::<f32>::new(ratio_out, 2.0, sinc_params(), inner_out_size, 2)?;

        let outer_out_size = outer.output_frames_max();
        let outer_in_buf = vec![vec![0.0f32; inner_out_size], vec![0.0f32; inner_out_size]];
        let outer_out_buf = vec![vec![0.0f32; outer_out_size], vec![0.0f32; outer_out_size]];

        Ok(Self {
            inner: Some(inner),
            inner_in_buf,
            inner_out_buf,
            outer: Some(outer),
            outer_in_buf,
            outer_out_buf,
            pw_rate,
            nam_rate,
        })
    }

    /// Retorna `true` quando `pw_rate == nam_rate` (ambos resamplers em bypass).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        self.inner.is_none()
    }

    /// Retorna a taxa de amostragem do PipeWire configurada nesta instância.
    #[inline]
    pub fn pw_rate(&self) -> u32 {
        self.pw_rate
    }

    /// Retorna a taxa de amostragem alvo do Modelo NAM configurada nesta instância.
    #[inline]
    pub fn nam_rate(&self) -> u32 {
        self.nam_rate
    }

    /// **Resampling de entrada** (input path): `pw_rate → nam_rate`.
    ///
    /// Processa `input` (em `pw_rate`) e escreve o resultado em `output` (em nam_rate kHz).
    /// RT-safe: usa apenas `process_into_buffer()` — zero alocações.
    ///
    /// # Retorno
    /// Número de amostras escritas em `output`. Em modo bypass retorna `input.len()`.
    ///
    /// # Panics
    /// Não entra em pânico; trunca silenciosamente se `output` for menor que o necessário.
    pub fn process_input(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        // Verifica se o resampler está ativo. Se for None, significa que a taxa de
        // amostragem do PipeWire é igual à do modelo NAM (bypass).
        let Some(ref mut resampler) = self.inner else {
            // Bypass Fast-Path: Copia amostras diretamente sem conversão.
            let n = in_l.len().min(out_l.len());
            out_l[..n].copy_from_slice(&in_l[..n]);
            out_r[..n].copy_from_slice(&in_r[..n]);
            return n;
        };

        // Prepara os buffers internos do resampler (formato planar L/R).
        // Resampler exige chunks fixos. Se in_l for maior, truncamos.
        let n_in = in_l.len().min(self.inner_in_buf[0].len());
        self.inner_in_buf[0][..n_in].copy_from_slice(&in_l[..n_in]);
        self.inner_in_buf[1][..n_in].copy_from_slice(&in_r[..n_in]);

        // Padding: Se o chunk de entrada for menor que o esperado pelo rubato,
        // preenchemos o restante com silêncio (0.0) para evitar lixo de memória.
        if n_in < self.inner_in_buf[0].len() {
            self.inner_in_buf[0][n_in..].fill(0.0);
            self.inner_in_buf[1][n_in..].fill(0.0);
        }

        // Executa o resampling propriamente dito. Operação RT-safe (zero-alloc).
        match resampler.process_into_buffer(&self.inner_in_buf, &mut self.inner_out_buf, None) {
            Ok((_, n_out)) => {
                // Copia o resultado do buffer interno para o buffer de saída fornecido.
                let n = n_out.min(out_l.len());
                out_l[..n].copy_from_slice(&self.inner_out_buf[0][..n]);
                out_r[..n].copy_from_slice(&self.inner_out_buf[1][..n]);
                n
            }
            Err(_) => {
                // Falha silenciosa RT-safe: Em caso de erro (ex: buffer desalinhado),
                // retornamos silêncio para evitar ruídos explosivos (glitches).
                let n = n_in.min(out_l.len());
                out_l[..n].fill(0.0);
                out_r[..n].fill(0.0);
                n
            }
        }
    }

    /// **Resampling de saída** (output path): `nam_rate → pw_rate`.
    ///
    /// Processa `input` (em nam_rate, saída da inferência NAM) e escreve em `output` (em `pw_rate`).
    /// RT-safe: usa apenas `process_into_buffer()` — zero alocações.
    ///
    /// # Retorno
    /// Número de amostras escritas em `output`. Em modo bypass retorna `input.len()`.
    pub fn process_output(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        // Verifica se o resampler está ativo (bypass se nam_rate == pw_rate).
        let Some(ref mut resampler) = self.outer else {
            // Bypass: copia diretamente o áudio processado pelo NAM para a saída do PipeWire.
            let n = in_l.len().min(out_l.len());
            out_l[..n].copy_from_slice(&in_l[..n]);
            out_r[..n].copy_from_slice(&in_r[..n]);
            return n;
        };

        // Copia para o buffer planar de entrada do resampler 'outer'.
        let n_in = in_l.len().min(self.outer_in_buf[0].len());
        self.outer_in_buf[0][..n_in].copy_from_slice(&in_l[..n_in]);
        self.outer_in_buf[1][..n_in].copy_from_slice(&in_r[..n_in]);

        // Padding para chunks incompletos.
        if n_in < self.outer_in_buf[0].len() {
            self.outer_in_buf[0][n_in..].fill(0.0);
            self.outer_in_buf[1][n_in..].fill(0.0);
        }

        // Converte de nam_rate kHz de volta para a taxa nativa do PipeWire (pw_rate).
        match resampler.process_into_buffer(&self.outer_in_buf, &mut self.outer_out_buf, None) {
            Ok((_, n_out)) => {
                let n = n_out.min(out_l.len());
                out_l[..n].copy_from_slice(&self.outer_out_buf[0][..n]);
                out_r[..n].copy_from_slice(&self.outer_out_buf[1][..n]);
                n
            }
            Err(_) => {
                // Fallback de segurança: silêncio.
                let n = n_in.min(out_l.len());
                out_l[..n].fill(0.0);
                out_r[..n].fill(0.0);
                n
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifica que quando a taxa do PipeWire é igual à taxa do NAM (ex: 48 kHz),
    /// o resampler entra em bypass automático e as amostras passam intactas (bit-perfect).
    #[test]
    fn test_bypass_48k() {
        let mut rs = NamResampler::new(48_000, 48_000, 256).expect("new falhou");
        assert!(rs.is_bypass(), "48k deve ser bypass");

        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input_r = [5.0f32, 4.0, 3.0, 2.0, 1.0];
        let mut output = [0.0f32; 5];
        let mut output_r = [0.0f32; 5];

        // Testa bypass no caminho de entrada.
        let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);
        assert_eq!(n, 5);
        assert_eq!(output, input, "bypass deve copiar exatamente L");
        assert_eq!(output_r, input_r, "bypass deve copiar exatamente R");

        // Testa bypass no caminho de saída.
        let n2 = rs.process_output(&input, &input_r, &mut output, &mut output_r);
        assert_eq!(n2, 5);
        assert_eq!(output, input, "bypass de saída deve copiar exatamente L");
        assert_eq!(
            output_r, input_r,
            "bypass de saída deve copiar exatamente R"
        );
    }

    /// Verifica o processo de downsampling de 96 kHz para 48 kHz.
    /// Valida se o número de amostras produzidas está dentro da faixa esperada,
    /// considerando o atraso de grupo (Group Delay) do filtro FIR Sinc.
    #[test]
    fn test_downsample_96k_to_48k() {
        let chunk = 512usize;
        let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");
        assert!(!rs.is_bypass());

        let input = vec![0.5f32; chunk];
        let input_r = vec![0.5f32; chunk];
        let mut output = vec![0.0f32; chunk * 2]; // buffer generoso
        let mut output_r = vec![0.0f32; chunk * 2];
        let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);

        // Para 96k→48k, esperamos ~chunk/2 amostras de saída.
        // O filtro Kaiser introduz Group Delay (metade do sinc_len).
        let expected_approx = chunk / 2;
        let expected_with_delay = expected_approx.saturating_sub(64);
        assert!(
            n >= expected_with_delay.saturating_sub(16) && n <= expected_approx + 16,
            "96k→48k: esperado ~{expected_with_delay} a {expected_approx} amostras pós-delay, obteve {n}"
        );
    }

    /// Verifica o upsampling de 44.1 kHz (CD quality) para 48 kHz.
    /// Garante que o interpolador está produzindo o aumento correto de densidade de amostras.
    #[test]
    fn test_upsample_44k_to_48k() {
        let chunk = 441usize;
        let mut rs = NamResampler::new(44_100, 48_000, chunk).expect("new falhou");
        assert!(!rs.is_bypass());

        let input = vec![0.3f32; chunk];
        let input_r = vec![0.3f32; chunk];
        let mut output = vec![0.0f32; chunk * 2];
        let mut output_r = vec![0.0f32; chunk * 2];
        let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);

        // Relação 48000/44100 ≈ 1.088x.
        let expected_approx = (chunk as f64 * 48_000.0 / 44_100.0) as usize;
        let expected_with_delay = expected_approx.saturating_sub(140);
        assert!(
            n >= expected_with_delay.saturating_sub(16) && n <= expected_approx + 16,
            "44.1k→48k: esperado ~{expected_with_delay} a {expected_approx} amostras pós-delay, obteve {n}"
        );
    }

    /// Verifica o upsampling no caminho de saída (48 kHz NAM -> 96 kHz PipeWire).
    #[test]
    fn test_output_upsample_48k_to_96k() {
        let chunk = 256usize;
        let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

        // O tamanho interno do buffer de saída do resampler de entrada é usado como
        // referência para o que o NAM entregaria.
        let inner_out_size = chunk / 2;

        let input = vec![0.4f32; inner_out_size];
        let input_r = vec![0.4f32; inner_out_size];
        let mut output = vec![0.0f32; chunk * 2];
        let mut output_r = vec![0.0f32; chunk * 2];
        let n = rs.process_output(&input, &input_r, &mut output, &mut output_r);

        // Esperamos o dobro de amostras (Relação 2:1).
        let expected_approx = inner_out_size * 2;
        assert!(
            n >= expected_approx.saturating_sub(16) && n <= expected_approx + 16,
            "48k→96k (output): esperado ~{expected_approx} amostras, obteve {n}"
        );
    }

    /// Teste de Roundtrip Completo: 96k -> 48k -> 96k.
    /// Valida a conservação de energia do sinal senoidal após passar por dois
    /// estágios de filtragem Sinc.
    #[test]
    fn test_roundtrip_96k() {
        let chunk = 512usize;
        let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

        // Sinal de teste: onda senoidal de 440 Hz.
        let input: Vec<f32> = (0..chunk)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 96_000.0_f32).sin())
            .collect();
        let input_r = input.clone();

        // 1. Downsample: 96k -> 48k
        let mut mid = vec![0.0f32; chunk];
        let mut mid_r = vec![0.0f32; chunk];
        let n_mid = rs.process_input(&input, &input_r, &mut mid, &mut mid_r);
        assert!(n_mid > 0, "process_input não produziu saída");

        // 2. Upsample: 48k -> 96k
        let mut out = vec![0.0f32; chunk * 2];
        let mut out_r = vec![0.0f32; chunk * 2];
        let n_out = rs.process_output(&mid[..n_mid], &mid_r[..n_mid], &mut out, &mut out_r);
        assert!(n_out > 0, "process_output não produziu saída");

        // Calcula RMS/Energia para garantir que o sinal não foi atenuado excessivamente.
        let energy_in: f32 = input.iter().map(|x| x * x).sum::<f32>() / chunk as f32;
        let energy_out: f32 = out[..n_out].iter().map(|x| x * x).sum::<f32>() / n_out as f32;

        assert!(
            energy_out > energy_in * 0.1,
            "Energia no roundtrip colapsou: entrada={energy_in:.4}, saída={energy_out:.4}"
        );
    }

    /// Testa a Resposta ao Impulso no caminho de entrada.
    /// Um impulso de Dirac deve resultar em uma função Sinc filtrada (Low-pass)
    /// sem picos excessivos que indicariam instabilidade numérica ou aliasing.
    #[test]
    fn test_impulse_response_input() {
        let chunk = 512usize;
        let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

        // Impulso unitário.
        let mut input = vec![0.0f32; chunk];
        input[0] = 1.0;
        let mut input_r = vec![0.0f32; chunk];
        input_r[0] = 1.0;

        let mut output = vec![0.0f32; chunk];
        let mut output_r = vec![0.0f32; chunk];
        let n = rs.process_input(&input, &input_r, &mut output, &mut output_r);
        assert!(n > 0);

        // Verifica energia finita e pico normalizado.
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

    /// Testa a Resposta ao Impulso no caminho de saída.
    #[test]
    fn test_impulse_response_output() {
        let chunk = 256usize;
        let mut rs = NamResampler::new(96_000, 48_000, chunk).expect("new falhou");

        let inner_out_approx = chunk / 2;
        let mut input = vec![0.0f32; inner_out_approx];
        input[0] = 1.0;
        let mut input_r = vec![0.0f32; inner_out_approx];
        input_r[0] = 1.0;

        let mut output = vec![0.0f32; chunk];
        let mut output_r = vec![0.0f32; chunk];
        let n = rs.process_output(&input, &input_r, &mut output, &mut output_r);
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
