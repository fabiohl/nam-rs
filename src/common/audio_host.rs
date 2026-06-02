// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Audio host abstraction (PipeWire, CLAP, etc).
//!
//! This module defines the contract that different audio backends must follow
//! to host the NAM-rs DSP engine.

use crate::common::diagnostics::NamDiagnostic;

/// Defines the minimum interface that an audio host must implement.
///
/// This trait abstracts the host lifecycle and configuration, allowing
/// the NAM-rs DSP engine to function both as a standalone app (PipeWire)
/// and as a plugin (CLAP/VST).
///
/// The `AudioHost` manages the infrastructure needed for audio processing,
/// but does not directly execute processing logic on the hot-path (which is
/// the responsibility of [`NamModel`](crate::models::NamModel) and the DSP pipeline).
pub trait AudioHost {
    /// Returns the current sample rate of the host in Hz.
    fn sample_rate(&self) -> f32;

    /// Returns the maximum buffer size (in samples) that the host can process.
    fn max_buffer_size(&self) -> usize;

    /// Starts the host processing loop.
    ///
    /// This function is blocking for standalone hosts (such as PipeWire) and
    /// returns only when the host is shut down or a fatal error occurs.
    /// For plugin hosts, the implementation may vary depending on the framework.
    ///
    /// Returns a [`NamDiagnostic`] if the host fails to initialize or
    /// if the loop is interrupted by a critical error.
    fn run(&mut self) -> Result<(), Box<NamDiagnostic>>;
}

#[cfg(test)]
/// Test module for validating AudioHost trait behavior.
mod tests {
    use super::*;
    use crate::common::diagnostics::{NamErrorCode, SystemSnapshot};

    /// A "MockHost" is an audio system simulator used only for testing.
    /// It allows testing how NAM-rs reacts to different configurations
    /// (such as different sample rates) without needing to connect PipeWire or a DAW.
    struct MockHost {
        /// The simulated sample rate (e.g. 44100.0 or 48000.0 Hz).
        rate: f32,
        /// The maximum audio "chunk" size the system can process at once.
        buffer_size: usize,
    }

    impl AudioHost for MockHost {
        /// Returns the sample rate configured in the simulator.
        fn sample_rate(&self) -> f32 {
            self.rate
        }

        /// Returns the buffer size configured in the simulator.
        fn max_buffer_size(&self) -> usize {
            self.buffer_size
        }

        /// Simulates the start of audio processing.
        /// If the sample rate is invalid (zero or negative), it generates a simulated error.
        fn run(&mut self) -> Result<(), Box<NamDiagnostic>> {
            if self.rate <= 0.0 {
                // Capture system state to aid in error diagnosis.
                let sys = SystemSnapshot::capture();
                return Err(Box::new(
                    NamDiagnostic::new(NamErrorCode::PipewireInitFailed, &sys)
                        .message("Invalid sample rate in MockHost"),
                ));
            }
            Ok(())
        }
    }

    /// Tests that the simulator correctly reports the values we configured.
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

    /// Tests that the simulator correctly detects an error configuration
    /// (in this case, a sample rate of 0 Hz).
    #[test]
    fn test_mock_host_error() {
        let mut host = MockHost {
            rate: 0.0,
            buffer_size: 1024,
        };

        let res = host.run();
        // Verify that the system detected the error.
        assert!(res.is_err());
        let diag = res.unwrap_err();
        // Verify that the error message matches what we expect.
        assert_eq!(diag.to_string(), "[E2100] Invalid sample rate in MockHost");
    }
}
