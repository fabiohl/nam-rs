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
    /// Retorna um [`NamDiagnostic`] se o host falhar ao inicializar ou
    /// se o loop for interrompido por um erro crítico.
    fn run(&mut self) -> Result<(), Box<NamDiagnostic>>;
}

#[cfg(test)]
/// Módulo de testes para validar o comportamento da trait AudioHost.
mod tests {
    use super::*;
    use crate::common::diagnostics::{NamErrorCode, SystemSnapshot};

    /// Um "MockHost" é um simulador de sistema de áudio usado apenas para testes.
    /// Ele nos permite testar como o NAM-rs reage a diferentes configurações
    /// (como taxas de amostragem diferentes) sem precisar ligar o PipeWire ou um DAW.
    struct MockHost {
        /// A taxa de amostragem simulada (ex: 44100.0 ou 48000.0 Hz).
        rate: f32,
        /// O tamanho máximo do "pacote" de áudio que o sistema pode processar por vez.
        buffer_size: usize,
    }

    impl AudioHost for MockHost {
        /// Retorna a taxa de amostragem configurada no simulador.
        fn sample_rate(&self) -> f32 {
            self.rate
        }

        /// Retorna o tamanho do buffer configurado no simulador.
        fn max_buffer_size(&self) -> usize {
            self.buffer_size
        }

        /// Simula o início do processamento de áudio.
        /// Se a taxa de amostragem for inválida (zero ou negativa), ele gera um erro simulado.
        fn run(&mut self) -> Result<(), Box<NamDiagnostic>> {
            if self.rate <= 0.0 {
                // Captura o estado do sistema para ajudar no diagnóstico do erro.
                let sys = SystemSnapshot::capture();
                return Err(Box::new(
                    NamDiagnostic::new(NamErrorCode::PipewireInitFailed, &sys)
                        .message("Sample rate inválido no MockHost"),
                ));
            }
            Ok(())
        }
    }

    /// Testa se o simulador reporta corretamente os valores que configuramos nele.
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

    /// Testa se o simulador detecta corretamente uma configuração de erro
    /// (neste caso, uma taxa de amostragem de 0 Hz).
    #[test]
    fn test_mock_host_error() {
        let mut host = MockHost {
            rate: 0.0,
            buffer_size: 1024,
        };

        let res = host.run();
        // Verificamos se o sistema realmente percebeu o erro.
        assert!(res.is_err());
        let diag = res.unwrap_err();
        // Verificamos se a mensagem de erro é a que esperamos.
        assert_eq!(diag.to_string(), "[E2100] Sample rate inválido no MockHost");
    }
}
