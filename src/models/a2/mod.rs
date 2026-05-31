// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Arquitetura A2 (Staging e Placeholder).
//!
//! Este módulo isola os componentes da arquitetura A2 (v0.6+), incluindo
//! stubs para ativações, FiLM, gating e parâmetros.

pub mod activations;
pub mod film;
pub mod gating;
pub mod params;

use crate::models::NamModel;

/// Re-exports públicos para facilitar o acesso.
pub use activations::{ActivationFn, ActivationType};
pub use film::{FiLMConfig, FiLMLayer};
pub use gating::GatingMode;
pub use params::{HeadParams, LayerArrayParamsA2, LayerParamsA2};

// =============================================================================
// Placeholder para WaveNet A2 (Staging)
// =============================================================================

/// Placeholder para a arquitetura WaveNet A2.
///
/// Este struct permite que o sistema carregue modelos A2 sem falhar, retornando
/// silêncio até que a implementação completa do motor de inferência esteja pronta.
#[derive(Default)]
pub struct WavenetA2Placeholder {
    /// Flag para emitir o aviso de log apenas uma vez por instância.
    warned: bool,
}

impl NamModel for WavenetA2Placeholder {
    fn process(&mut self, _input: &[f32], output: &mut [f32]) {
        #[cfg(feature = "heap-audit")]
        if crate::clap::heap_audit::AUDIT_ENABLED.load(std::sync::atomic::Ordering::Relaxed) {
            let tid = unsafe { libc::syscall(libc::SYS_gettid) as i32 };
            let audit_thread =
                crate::clap::heap_audit::AUDIT_THREAD.load(std::sync::atomic::Ordering::Relaxed);
            if audit_thread == 0 || audit_thread == tid {
                crate::clap::heap_audit::ALLOC_COUNT
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }

        if !self.warned {
            log::warn!(
                "Arquitetura WaveNet A2 detectada: Modo Placeholder (Silencioso) ativo. A implementação real está em desenvolvimento."
            );
            self.warned = true;
        }

        // Retorna silêncio absoluto.
        output.fill(0.0);
    }

    fn prewarm(&mut self, _num_samples: usize) {
        // No-op para o placeholder.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavenet_a2_placeholder_silence() {
        let mut model = WavenetA2Placeholder::default();
        let input = [1.0f32; 10];
        let mut output = [1.0f32; 10];
        model.process(&input, &mut output);
        for val in output.iter() {
            assert_eq!(*val, 0.0);
        }
    }
}
