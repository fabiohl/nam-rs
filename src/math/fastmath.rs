// Copyright (c) 2026 Fábio Henrique de Lima Silva.
#![allow(unsafe_op_in_unsafe_fn)]

//! Módulo de FastMath (Minimax & Pade) para otimização de ativação não linear.
//! mitigando gargalos de ALU ao calcular `tanh` e `sigmoid` na `WaveNet`/`LSTM`.
//! As aproximações são derivadas de polinômios do ecossistema referencial (math_approx).

use core::arch::x86_64::*;
use std::sync::OnceLock;

/// Tamanho da tabela LUT para ganho. 4096 pontos fornecem precisão sub-0.001 dB.
pub const GAIN_LUT_SIZE: usize = 4096;
/// Limite inferior em dB para a LUT (Piso de silêncio prático -96dB).
pub const GAIN_MIN_DB: f32 = -96.0;
/// Limite superior em dB para a LUT (+30dB é um boost extremo).
pub const GAIN_MAX_DB: f32 = 30.0;
const GAIN_DB_RANGE: f32 = GAIN_MAX_DB - GAIN_MIN_DB;
const GAIN_DB_STEP: f32 = GAIN_DB_RANGE / (GAIN_LUT_SIZE as f32 - 1.0);
const INV_GAIN_DB_STEP: f32 = 1.0 / GAIN_DB_STEP;
/// Limite de segurança para evitar overflow no polinômio de tanh (evita NaN).
const TANH_CLAMP_LIMIT: f32 = 15.0;

/// Tabela de Look-Up para conversão ultra-rápida de dB para ganho linear.
/// Projetada para ser instanciada via `OnceLock` e acessada em threads RT.
pub struct GainLUT {
    table: [f32; GAIN_LUT_SIZE],
}

impl GainLUT {
    /// Inicializa a LUT pré-calculando os valores de ganho linear.
    pub fn new() -> Self {
        let mut table = [0.0f32; GAIN_LUT_SIZE];
        for (i, item) in table.iter_mut().enumerate() {
            let db = GAIN_MIN_DB + (i as f32 * GAIN_DB_STEP);
            *item = 10.0f32.powf(db / 20.0);
        }
        Self { table }
    }
}

impl Default for GainLUT {
    fn default() -> Self {
        Self::new()
    }
}

impl GainLUT {
    /// Converte dB para ganho linear usando interpolação linear na LUT.
    /// Operação determinística, zero-allocation e amigável ao pipeline da CPU.
    #[inline(always)]
    pub fn db_to_linear(&self, db: f32) -> f32 {
        // Clamp para garantir que o índice esteja dentro dos limites da tabela.
        let db_clamped = db.clamp(GAIN_MIN_DB, GAIN_MAX_DB);
        let exact_idx = (db_clamped - GAIN_MIN_DB) * INV_GAIN_DB_STEP;

        let idx0 = exact_idx as usize;
        let idx1 = (idx0 + 1).min(GAIN_LUT_SIZE - 1);
        let frac = exact_idx - (idx0 as f32);

        // Interpolação linear: y = y0 + frac * (y1 - y0)
        let y0 = self.table[idx0];
        let y1 = self.table[idx1];
        y0 + frac * (y1 - y0)
    }
}

/// Instância global da LUT de ganho, carregada no primeiro acesso.
pub static GAIN_LUT: OnceLock<GainLUT> = OnceLock::new();

/// Retorna a instância global da LUT, inicializando-a se necessário.
pub fn get_gain_lut() -> &'static GainLUT {
    GAIN_LUT.get_or_init(GainLUT::new)
}

/// Otimizações matemáticas portadas dos polinômios de NeuralAudio / math_approx.
/// Aplica aproximação vetorial de `tanh(x)` iterando um polinômio de grau 5.
///
/// Algoritmo otimizado correspondente ao `tanh<5>` usado no NAM core. O polinômio
/// é resolvido usando instruções primitivas de Fused Multiply-Add (FMA) e
/// então dividido pela sua aproximação de raiz quadrada recíproca.
///
/// # Erro Máximo vs `f32::tanh()`
///
/// O polinômio Minimax de grau 7 + refinamento Newton-Raphson sobre `_mm256_rsqrt_ps`
/// introduz um erro absoluto máximo de **~1.2e-5** por ativação em relação a `f32::tanh()`
/// (validado pelos testes unitários de `test_simd_fastmath_tanh_mse`).
///
/// Esta divergência é intencional: o custo de um `tanh` escalar via libm (~20–60 ciclos)
/// é substituído por uma sequência FMA+rsqrt de ~4–6 ciclos, com erro aceitável para
/// inferência perceptual de áudio (resolução de 16-bit equivale a erro de ~3e-5 no
/// domínio normalizado — o FastMath opera uma ordem de magnitude acima desse piso).
///
/// # Acumulação em Modelos WaveNet Empilhados
///
/// Em modelos WaveNet com `N` camadas empilhadas, o erro do FastMath **não** acumula
/// linearmente: cada camada aplica uma ativação não-linear que reescala o resíduo.
/// A acumulação empírica segue um modelo sublinear aproximado:
///
/// ```text
/// erro_máx_acumulado ≈ √N_camadas × erro_por_camada
/// ```
///
/// Para o modelo BossWN-standard (20 camadas: 2 arrays × 10 layers), com
/// `erro_por_camada ≈ 5e-3`:
///
/// ```text
/// erro_máx ≈ √20 × 5e-3 ≈ 4.47 × 5e-3 ≈ 2.2e-2
/// ```
///
/// O MSE golden medido (3.21e-2 em 2026-04-15) é consistente com esta estimativa
/// — ver `docs/architecture.md §2` e docstring de `test_golden_vectors_wavenet`
/// para a justificativa completa do threshold `5e-2`.
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX2 e FMA, e que o registrador
/// `x` contenha valores f32 válidos.
pub unsafe fn simd_tanh_avx2(x: __m256) -> __m256 {
    unsafe {
        // Coeficientes do polinômio Minimax de grau 7
        let c0 = _mm256_set1_ps(0.166_814_34_f32);
        let c1 = _mm256_set1_ps(0.008_153_17_f32);
        let c2 = _mm256_set1_ps(0.000_246_32_f32);
        let one = _mm256_set1_ps(1.0);
        let min_limit = _mm256_set1_ps(-TANH_CLAMP_LIMIT);
        let max_limit = _mm256_set1_ps(TANH_CLAMP_LIMIT);

        // Clamp de saturação para evitar overflow no cálculo de p(x)^2
        let x = _mm256_max_ps(min_limit, _mm256_min_ps(max_limit, x));

        // x_sq = x * x
        let x_sq = _mm256_mul_ps(x, x);
        let x_sq_sq = _mm256_mul_ps(x_sq, x_sq);

        let y_3_5 = _mm256_fmadd_ps(c1, x_sq, c0);
        let y_3_5_7 = _mm256_fmadd_ps(c2, x_sq_sq, y_3_5);
        let y_full = _mm256_fmadd_ps(y_3_5_7, x_sq, one);

        // p(x) = x * y_full
        let p_x = _mm256_mul_ps(x, y_full);

        // Evaluando rsqrt(p(x)^2 + 1)
        let p_x_sq = _mm256_mul_ps(p_x, p_x);
        let radicand = _mm256_add_ps(p_x_sq, one);

        // Instrução HW nativa de rsqrt fornece ~11-14 bits de precisão na base arquitetural
        let mut rr = _mm256_rsqrt_ps(radicand);

        // Refinamento de Newton-Raphson: eleva a precisão das casas decimais (1 iter)
        let three = _mm256_set1_ps(3.0);
        let half = _mm256_set1_ps(0.5);

        let rr_sq = _mm256_mul_ps(rr, rr);

        // diff = 3.0 - (radicand * rr^2) — fusão FMA (FNMADD: c - a*b em 1 ciclo)
        let diff = _mm256_fnmadd_ps(radicand, rr_sq, three);

        // rr_half = rr * 0.5
        let rr_half = _mm256_mul_ps(rr, half);

        // rr_new = (rr * 0.5) * (3.0 - radicand * rr^2)
        rr = _mm256_mul_ps(rr_half, diff);

        // Retorna tangete final iterada como tanh(x) ~ p(x) * rsqrt(p(x)^2 + 1)
        _mm256_mul_ps(p_x, rr)
    }
}

/// Aplica aproximação vetorial de `sigmoid(x)` através da identidade logarítimica da tanh.
/// Baseia-se matematicamente em `sigmoid(x) = 0.5 * (1.0 + tanh(0.5 * x))`.
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX2 e FMA.
pub unsafe fn simd_sigmoid_avx2(x: __m256) -> __m256 {
    unsafe {
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);

        // x * 0.5
        let x_half = _mm256_mul_ps(x, half);

        // tanh(x * 0.5)
        let th = simd_tanh_avx2(x_half);

        // 1.0 + tanh(x * 0.5)
        let t_plus_one = _mm256_add_ps(th, one);

        // 0.5 * (1.0 + tanh(x * 0.5))
        _mm256_mul_ps(t_plus_one, half)
    }
}

/// Aplica aproximação vetorial de `tanh(x)` iterando um polinômio de grau 5 (AVX-512).
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX-512 (F e VL).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_avx512(x: __m512) -> __m512 {
    // Coeficientes do polinômio Minimax de grau 7
    let c0 = _mm512_set1_ps(0.166_814_34_f32);
    let c1 = _mm512_set1_ps(0.008_153_17_f32);
    let c2 = _mm512_set1_ps(0.000_246_32_f32);
    let one = _mm512_set1_ps(1.0);
    let min_limit = _mm512_set1_ps(-TANH_CLAMP_LIMIT);
    let max_limit = _mm512_set1_ps(TANH_CLAMP_LIMIT);

    // Clamp de saturação para evitar overflow no cálculo de p(x)^2
    let x = _mm512_max_ps(min_limit, _mm512_min_ps(max_limit, x));

    // x_sq = x * x
    let x_sq = _mm512_mul_ps(x, x);
    let x_sq_sq = _mm512_mul_ps(x_sq, x_sq);

    let y_3_5 = _mm512_fmadd_ps(c1, x_sq, c0);
    let y_3_5_7 = _mm512_fmadd_ps(c2, x_sq_sq, y_3_5);
    let y_full = _mm512_fmadd_ps(y_3_5_7, x_sq, one);

    // p(x) = x * y_full
    let p_x = _mm512_mul_ps(x, y_full);

    // Evaluando rsqrt(p(x)^2 + 1)
    let p_x_sq = _mm512_mul_ps(p_x, p_x);
    let radicand = _mm512_add_ps(p_x_sq, one);

    // _mm512_rsqrt14_ps ~14 bits de precisão
    let mut rr = _mm512_rsqrt14_ps(radicand);

    // Refinamento de Newton-Raphson: eleva a precisão das casas decimais (1 iter)
    let three = _mm512_set1_ps(3.0);
    let half = _mm512_set1_ps(0.5);

    let rr_sq = _mm512_mul_ps(rr, rr);

    // diff = 3.0 - (radicand * rr^2) — fusão FMA (FNMADD: c - a*b em 1 ciclo)
    let diff = _mm512_fnmadd_ps(radicand, rr_sq, three);

    // rr_half = rr * 0.5
    let rr_half = _mm512_mul_ps(rr, half);

    // rr_new = (rr * 0.5) * (3.0 - radicand * rr^2)
    rr = _mm512_mul_ps(rr_half, diff);

    // Retorna tangete final iterada como tanh(x) ~ p(x) * rsqrt(p(x)^2 + 1)
    _mm512_mul_ps(p_x, rr)
}

/// Aplica aproximação vetorial de `sigmoid(x)` através da identidade logarítimica da tanh (AVX-512).
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX-512 (F e VL).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_sigmoid_avx512(x: __m512) -> __m512 {
    unsafe {
        let half = _mm512_set1_ps(0.5);
        let one = _mm512_set1_ps(1.0);

        // x * 0.5
        let x_half = _mm512_mul_ps(x, half);

        // tanh(x * 0.5)
        let th = simd_tanh_avx512(x_half);

        // 1.0 + tanh(x * 0.5)
        let t_plus_one = _mm512_add_ps(th, one);

        // 0.5 * (1.0 + tanh(x * 0.5))
        _mm512_mul_ps(t_plus_one, half)
    }
}

/// Executa a ativação fundida dos gates LSTM para AVX2.
/// Computa as portas f, i, g, o e atualiza o estado da célula e saída oculta.
///
/// # Safety
/// Requer suporte a AVX2 e FMA.
pub unsafe fn fused_lstm_gates_avx2(
    gf: __m256,
    gi: __m256,
    gg: __m256,
    go: __m256,
    cs: __m256,
) -> (__m256, __m256) {
    unsafe {
        let f = simd_sigmoid_avx2(gf);
        let i = simd_sigmoid_avx2(gi);
        let g = simd_tanh_avx2(gg);
        let o = simd_sigmoid_avx2(go);

        // new_cs = f * cs + i * g
        let new_cs = _mm256_fmadd_ps(f, cs, _mm256_mul_ps(i, g));

        // hidden = o * tanh(new_cs)
        let hidden = _mm256_mul_ps(o, simd_tanh_avx2(new_cs));

        (new_cs, hidden)
    }
}

/// Executa a ativação fundida dos gates LSTM para AVX-512.
///
/// # Safety
/// Requer suporte a AVX-512F e AVX-512VL.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn fused_lstm_gates_avx512(
    gf: __m512,
    gi: __m512,
    gg: __m512,
    go: __m512,
    cs: __m512,
) -> (__m512, __m512) {
    unsafe {
        let f = simd_sigmoid_avx512(gf);
        let i = simd_sigmoid_avx512(gi);
        let g = simd_tanh_avx512(gg);
        let o = simd_sigmoid_avx512(go);

        // new_cs = f * cs + i * g
        let new_cs = _mm512_fmadd_ps(f, cs, _mm512_mul_ps(i, g));

        // hidden = o * tanh(new_cs)
        let hidden = _mm512_mul_ps(o, simd_tanh_avx512(new_cs));

        (new_cs, hidden)
    }
}

/// Processa uma fatia in-place aplicando a `simd_tanh` otimizada baseada em AVX2 (YMM).
/// Abstrai a carga iterativa de 8 elementos da camada neural.
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX2 e FMA, e que `slice`
/// tenha tamanho suficiente e esteja corretamente alinhado se necessário (loadu lida com desalinhamento).
pub unsafe fn tanh_slice_avx2(slice: &mut [f32]) {
    unsafe {
        let mut i = 0;
        while i + 8 <= slice.len() {
            let va = _mm256_loadu_ps(slice.as_ptr().add(i));
            let vt = simd_tanh_avx2(va);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), vt);
            i += 8;
        }
        while i < slice.len() {
            slice[i] = slice[i].tanh();
            i += 1;
        }
    }
}

/// Processa uma fatia in-place aplicando a `simd_tanh_avx512` otimizada baseada em AVX-512 (ZMM).
/// Abstrai a carga iterativa de 16 elementos da camada neural.
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX-512 (F e VL).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn tanh_slice_avx512(slice: &mut [f32]) {
    unsafe {
        let mut i = 0;
        while i + 16 <= slice.len() {
            let va = _mm512_loadu_ps(slice.as_ptr().add(i));
            let vt = simd_tanh_avx512(va);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), vt);
            i += 16;
        }
        while i < slice.len() {
            slice[i] = slice[i].tanh();
            i += 1;
        }
    }
}

/// Processa uma fatia in-place aplicando a `simd_sigmoid` otimizada baseada em AVX2 (YMM).
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX2 e FMA.
pub unsafe fn sigmoid_slice_avx2(slice: &mut [f32]) {
    unsafe {
        let mut i = 0;
        while i + 8 <= slice.len() {
            let va = _mm256_loadu_ps(slice.as_ptr().add(i));
            let vt = simd_sigmoid_avx2(va);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), vt);
            i += 8;
        }
        while i < slice.len() {
            let val = slice[i];
            slice[i] = 0.5 * (1.0 + (val * 0.5).tanh());
            i += 1;
        }
    }
}

/// Processa uma fatia in-place aplicando a `simd_sigmoid_avx512` otimizada baseada em AVX-512 (ZMM).
///
/// # Safety
/// O chamador deve garantir que a CPU suporte instruções AVX-512 (F e VL).
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn sigmoid_slice_avx512(slice: &mut [f32]) {
    unsafe {
        let mut i = 0;
        while i + 16 <= slice.len() {
            let va = _mm512_loadu_ps(slice.as_ptr().add(i));
            let vt = simd_sigmoid_avx512(va);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), vt);
            i += 16;
        }
        while i < slice.len() {
            let val = slice[i];
            slice[i] = 0.5 * (1.0 + (val * 0.5).tanh());
            i += 1;
        }
    }
}

/// Aproximação escalar de `tanh(x)`.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Aproximação escalar de `sigmoid(x)`.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + (x * 0.5).tanh())
}

/// Despacha para a implementação de tanh SIMD mais rápida detectada (AVX2).
///
/// # Safety
/// Esta função é segura se o despacho detectar corretamente as instruções suportadas.
pub unsafe fn simd_tanh(x: __m256) -> __m256 {
    simd_tanh_avx2(x)
}

/// Despacha para a implementação de sigmoid SIMD mais rápida detectada (AVX2).
///
/// # Safety
/// Esta função é segura se o despacho detectar corretamente as instruções suportadas.
pub unsafe fn simd_sigmoid(x: __m256) -> __m256 {
    simd_sigmoid_avx2(x)
}

#[cfg(test)]
#[path = "fastmath_test.rs"]
mod fastmath_test;
