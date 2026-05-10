// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

//! Abstração de host de áudio (PipeWire, CLAP, etc).
//!
//! Este módulo define o contrato que diferentes backends de áudio devem seguir
//! para hospedar o motor DSP do NAM-rs.

use crate::common::diagnostics::NamDiagnostic;

/// Define a interface mínima que um host de áudio deve implementar.
///
/// Esta trait abstrai o ciclo de vida e a configuração do host, permitindo
/// que o motor DSP do NAM-rs funcione tanto como um app standalone (PipeWire)
/// quanto como um plugin (CLAP/VST).
///
/// O `AudioHost` gerencia a infraestrutura necessária para o processamento de áudio,
/// mas não executa diretamente a lógica de processamento no hot-path (que é
/// responsabilidade do [`NamModel`](crate::models::NamModel) e da pipeline DSP).
pub trait AudioHost {
    /// Retorna a taxa de amostragem (Sample Rate) atual do host em Hz.
    fn sample_rate(&self) -> f32;

    /// Retorna o tamanho máximo de buffer (em samples) que o host pode processar.
    fn max_buffer_size(&self) -> usize;

    /// Inicia o loop de processamento do host.
    ///
    /// Esta função é bloqueante para hosts standalone (como o PipeWire) e
    /// retorna apenas quando o host é encerrado ou ocorre um erro fatal.
    /// Para hosts de plugin, a implementação pode variar conforme o framework.
    ///
    /// # Erros
    ///
    /// Retorna um [`NamDiagnostic`] se o host falhar ao inicializar ou
    /// se o loop for interrompido por um erro crítico.
    fn run(&mut self) -> Result<(), Box<NamDiagnostic>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::diagnostics::{NamErrorCode, SystemSnapshot};

    struct MockHost {
        rate: f32,
        buffer_size: usize,
    }

    impl AudioHost for MockHost {
        fn sample_rate(&self) -> f32 {
            self.rate
        }

        fn max_buffer_size(&self) -> usize {
            self.buffer_size
        }

        fn run(&mut self) -> Result<(), Box<NamDiagnostic>> {
            if self.rate <= 0.0 {
                let sys = SystemSnapshot::capture();
                return Err(Box::new(
                    NamDiagnostic::new(NamErrorCode::PipewireInitFailed, &sys)
                        .message("Sample rate inválido no MockHost"),
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn test_mock_host_traits() {
        let mut host = MockHost {
            rate: 48000.0,
            buffer_size: 1024,
        };

        assert_eq!(host.sample_rate(), 48000.0);
        assert_eq!(host.max_buffer_size(), 1024);
        assert!(host.run().is_ok());
    }

    #[test]
    fn test_mock_host_error() {
        let mut host = MockHost {
            rate: 0.0,
            buffer_size: 1024,
        };

        let res = host.run();
        assert!(res.is_err());
        let diag = res.unwrap_err();
        assert_eq!(diag.to_string(), "[E2100] Sample rate inválido no MockHost");
    }
}
