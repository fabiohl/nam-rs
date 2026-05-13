// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Sistema de despacho dinâmico para kernels SIMD.

use super::traits::SimdMath;
use crate::math::common::{Avx2Math, Avx2VnniMath, Avx512Math, Avx512VnniBf16Math, Avx512VnniMath};
use std::sync::LazyLock;

/// Enumera os conjuntos de instruções suportados.
///
/// Nota: Não existe variante `Fallback` escalar neste enum. O projeto tem como alvo
/// mandatório a microarquitetura x86-64-v3 (AVX2+FMA). Se AVX2 não for detectado,
/// `detect_best_simd()` entra em pânico no boot (fail-fast).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum InstructionSet {
    /// AVX2 + FMA (x86-64-v3).
    Avx2,
    /// AVX2 + VNNI (Alder Lake+, Zen 4+).
    Avx2Vnni,
    /// AVX-512 Foundation (Skylake-X+, Zen 4+).
    Avx512,
    /// AVX-512 VNNI.
    Avx512Vnni,
    /// AVX-512 VNNI + BF16.
    Avx512VnniBf16,
}

// ══════════════════════════════════════════════════════════════════════════════
// DESIGN DEBT: Coexistência de dois mecanismos de dispatch
// ══════════════════════════════════════════════════════════════════════════════
//
// O projeto NAM-rs usa DOIS mecanismos independentes para despacho SIMD:
//
// 1. Trait genérica `SimdMath` (definida em `common/traits.rs`)
//    - Despacho estático (monomorphization) via `dispatch_simd!` Modos 1 e 2
//    - Usado por: WaveNet (`wavenet.rs`, `wavenet_dyn.rs`), LSTM (`lstm_dyn.rs`),
//      DSP (`gate.rs`, `resampler.rs`)
//    - Exemplo: `self.process::<Avx2Math>(args)` → monomorphized em tempo de compilação
//    - Vantagem: zero overhead de v-table, inline agressivo
//    - Desvantagem: gera código duplicado para cada ISA (Avx2, Avx512, Avx512Vnni...)
//
// 2. V-table `SimdMathConfig` (esta struct)
//    - Despacho dinâmico via ponteiros de função
//    - Usado por: operações DSP no pipeline (`dsp/pipeline.rs`, `dsp/gain.rs`),
//      standalone host (`standalone/rt_setup.rs`), `dispatch_simd!` Modo 3
//    - Exemplo: `(SIMD_MATH.apply_gain)(data, gain)` → chamada indireta via ponteiro
//    - Vantagem: código único, sem duplicação
//    - Desvantagem: impede inline, custo de indireção (~1-2 ciclos)
//
// Consumidores por mecanismo:
//   Mecanismo 1 (trait):  wavenet.rs, wavenet_dyn.rs, lstm_dyn.rs, gate.rs, resampler.rs
//   Mecanismo 2 (v-table): pipeline.rs, rt_setup.rs, cli.rs, ops.rs (compute_energy_stereo)
//   Ambos (híbrido):      lstm.rs (usa dispatch_simd! Modo 2 para gemv_4gate,
//                          mas também chama simd_tanh/simd_sigmoid diretamente)
//
// Plano de unificação (futuro):
//   - Mover TODOS os consumidores para a trait `SimdMath` (Mecanismo 1)
//   - Substituir v-table `SimdMathConfig` por um único despacho baseado na trait
//   - Remover ponteiros de função da struct `SimdMathConfig`
//   - Manter `InstructionSet` para consultas de capabilities (ex: `is_avx512`)
//   - Isso eliminará ~50 linhas de boilerplate em `detect_best_simd()`
//
// Data do debt: 2026-05-12 (refatoração Épicos 1-5)
// Prioridade: Média (não afeta performance em caminhos quentes,
//             que já usam Mecanismo 1 com monomorphization)
// ══════════════════════════════════════════════════════════════════════════════
/// Tabela de despacho dinâmico (v-table) para operações SIMD.
#[derive(Clone, Copy)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub struct SimdMathConfig {
    /// Conjunto de instruções ativo.
    pub instruction_set: InstructionSet,
    /// Nome amigável do backend.
    pub name: &'static str,
    /// Se o backend é AVX-512.
    pub is_avx512: bool,
    /// Kernel fundido de adição e GEMV.
    pub fused_add_gemv: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], bool),
    /// Kernel fundido de adição e GEMM em lote.
    pub fused_add_gemm_batch: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], usize, bool),
    /// Kernel fundido de GEMM residual em lote.
    pub fused_gemm_residual_batch:
        unsafe fn(&[f32], &[u16], &[f32], &[f32], &mut [f32], usize, bool),
    /// Kernel de GEMV com sobrescrita.
    pub gemv_overwrite: unsafe fn(&[f32], &[u16], &[f32], &mut [f32], bool),
    /// Kernel de acumulação de cabeça.
    pub accumulate_head: unsafe fn(&mut [f32], &[f32]),
    /// Soma horizontal de um buffer.
    pub horizontal_sum: unsafe fn(*const f32, usize) -> f32,
    /// Aplica ganho e detecta clipping em estéreo.
    pub apply_gain_and_detect_clipping_stereo: unsafe fn(&mut [f32], &mut [f32], f32) -> bool,
    /// Aplica ganho constante em estéreo (sem detecção de clipping).
    pub apply_gain_stereo: unsafe fn(&mut [f32], &mut [f32], f32),
    /// Aplica ganho constante em um buffer mono.
    pub apply_gain: unsafe fn(&mut [f32], f32),
    /// Aplica rampa linear de ganho em estéreo.
    pub apply_ramp_stereo: unsafe fn(&mut [f32], &mut [f32], f32, f32),
    /// Convolução estéreo (usada no resampler).
    pub convolve_stereo: unsafe fn(*const f32, *const f32, *const f32, usize) -> (f32, f32),
    /// Calcula o máximo da energia entre dois canais.
    pub compute_energy_stereo: unsafe fn(&[f32], &[f32]) -> f32,
}

impl SimdMathConfig {
    /// Retorna a configuração SIMD global ativa.
    pub fn current() -> Self {
        *SIMD_MATH
    }

    /// Alias para current().
    pub fn get() -> Self {
        Self::current()
    }
}

/// Instância global da configuração SIMD, detectada no boot.
pub static SIMD_MATH: LazyLock<SimdMathConfig> = LazyLock::new(detect_best_simd);

fn detect_best_simd() -> SimdMathConfig {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vnni") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx512VnniBf16,
                name: "AVX-512 (VNNI+BF16)",
                is_avx512: true,
                fused_add_gemv: Avx512VnniBf16Math::fused_add_gemv,
                fused_add_gemm_batch: Avx512VnniBf16Math::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx512VnniBf16Math::fused_gemm_residual_batch,
                gemv_overwrite: Avx512VnniBf16Math::gemv_overwrite,
                accumulate_head: Avx512VnniBf16Math::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx512VnniBf16Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512VnniBf16Math::apply_gain_stereo,
                apply_gain: Avx512VnniBf16Math::apply_gain,
                apply_ramp_stereo: Avx512VnniBf16Math::apply_ramp_stereo,
                convolve_stereo: Avx512VnniBf16Math::convolve_stereo,
                compute_energy_stereo: Avx512VnniBf16Math::compute_energy_stereo,
            };
        }
        if is_x86_feature_detected!("avx512vnni") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx512Vnni,
                name: "AVX-512 (VNNI)",
                is_avx512: true,
                fused_add_gemv: Avx512VnniMath::fused_add_gemv,
                fused_add_gemm_batch: Avx512VnniMath::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx512VnniMath::fused_gemm_residual_batch,
                gemv_overwrite: Avx512VnniMath::gemv_overwrite,
                accumulate_head: Avx512VnniMath::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx512VnniMath::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512VnniMath::apply_gain_stereo,
                apply_gain: Avx512VnniMath::apply_gain,
                apply_ramp_stereo: Avx512VnniMath::apply_ramp_stereo,
                convolve_stereo: Avx512VnniMath::convolve_stereo,
                compute_energy_stereo: Avx512VnniMath::compute_energy_stereo,
            };
        }
        if is_x86_feature_detected!("avx512f") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx512,
                name: "AVX-512",
                is_avx512: true,
                fused_add_gemv: Avx512Math::fused_add_gemv,
                fused_add_gemm_batch: Avx512Math::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx512Math::fused_gemm_residual_batch,
                gemv_overwrite: Avx512Math::gemv_overwrite,
                accumulate_head: Avx512Math::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx512(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx512Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx512Math::apply_gain_stereo,
                apply_gain: Avx512Math::apply_gain,
                apply_ramp_stereo: Avx512Math::apply_ramp_stereo,
                convolve_stereo: Avx512Math::convolve_stereo,
                compute_energy_stereo: Avx512Math::compute_energy_stereo,
            };
        }
        if is_x86_feature_detected!("avxvnni") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx2Vnni,
                name: "AVX2 (VNNI)",
                is_avx512: false,
                fused_add_gemv: Avx2VnniMath::fused_add_gemv,
                fused_add_gemm_batch: Avx2VnniMath::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx2VnniMath::fused_gemm_residual_batch,
                gemv_overwrite: Avx2VnniMath::gemv_overwrite,
                accumulate_head: Avx2VnniMath::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx2(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx2VnniMath::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx2VnniMath::apply_gain_stereo,
                apply_gain: Avx2VnniMath::apply_gain,
                apply_ramp_stereo: Avx2VnniMath::apply_ramp_stereo,
                convolve_stereo: Avx2VnniMath::convolve_stereo,
                compute_energy_stereo: Avx2VnniMath::compute_energy_stereo,
            };
        }
        if is_x86_feature_detected!("avx2") {
            return SimdMathConfig {
                instruction_set: InstructionSet::Avx2,
                name: "AVX2",
                is_avx512: false,
                fused_add_gemv: Avx2Math::fused_add_gemv,
                fused_add_gemm_batch: Avx2Math::fused_add_gemm_batch,
                fused_gemm_residual_batch: Avx2Math::fused_gemm_residual_batch,
                gemv_overwrite: Avx2Math::gemv_overwrite,
                accumulate_head: Avx2Math::accumulate_head,
                horizontal_sum: |ptr, len| unsafe {
                    crate::math::common::utility::horizontal_sum_avx2(ptr, len)
                },
                apply_gain_and_detect_clipping_stereo:
                    Avx2Math::apply_gain_and_detect_clipping_stereo,
                apply_gain_stereo: Avx2Math::apply_gain_stereo,
                apply_gain: Avx2Math::apply_gain,
                apply_ramp_stereo: Avx2Math::apply_ramp_stereo,
                convolve_stereo: Avx2Math::convolve_stereo,
                compute_energy_stereo: Avx2Math::compute_energy_stereo,
            };
        }
    }

    // Nenhum conjunto de instruções compatível foi detectado.
    // O projeto exige x86-64-v3 (AVX2+FMA) como mínimo absoluto.
    // Entrar em pânico aqui é intencional: é melhor falhar rápido no boot
    // do que produzir áudio corrompido ou tráfego undefined-behavior no DSP.
    panic!(
        "[NAM-rs] CPU incompatível: AVX2 não detectado.\n\
         Este binário requer x86-64-v3 (AVX2 + FMA).\n\
         Execute em um processador lançado após ~2013 (Intel Haswell / AMD Ryzen)."
    );
}
