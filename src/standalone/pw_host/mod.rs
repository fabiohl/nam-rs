// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP audio processing core using `pipewire-rs`.
//!
//! This is the "heart" of NAM-rs Standalone mode: the module that processes audio in
//! real time. It receives raw audio samples from PipeWire (the Linux sound
//! server), passes them through the "neural engine", and delivers the processed
//! final result to the hardware via a dual-stream architecture.
//!
//! ## Dual-Stream Architecture with DspBridge
//!
//! PipeWire copies buffers to the monitor port **before** calling `process()`.
//! Therefore, in-place modifications on a single `Audio/Sink` stream would be invisible
//! to the hardware. The solution uses two streams:
//!
//! 1. **Capture stream** (`Audio/Sink`, `Direction::Input`) — acts as a Virtual Sink
//!    that receives audio from apps, applies the DSP chain (gain + neural inference)
//!    and writes the result to [`DspBridge`].
//! 2. **Playback stream** (`Stream/Output/Audio`, `Direction::Output`) — acts as a
//!    playback client that reads from [`DspBridge`] and delivers to hardware.
//!
//! The [`DspBridge`] is a `#[repr(align(128))]` buffer shared between the two
//! closures via raw pointer, with lock-free synchronization via `Ordering::Release/Acquire`
//! and an atomic generation counter.
//!
//! ## Absolute rules of this module (why are they so strict?)
//!
//! In the `process()` callback (the function called hundreds of times per second by PipeWire):
//! - **Zero heap allocation** — we never request new memory from the system during processing.
//! - **Zero I/O** — we never write to the terminal or files; status is reported via atomic flags.
//! - **Zero mutexes** — we never lock/wait for other threads.
//!
//! These rules exist because any pause, no matter how small, would cause clicks and glitches in
//! the audio — unacceptable for a musician playing live.
//!
//! ## Processing flow (Capture callback)
//!
//! The `process()` callback follows this sequence for each audio block:
//! 1. **Noise Gate and Input Gain** — Evaluates signal energy and applies the initial gain (pre-DSP).
//! 2. `NamResampler::process_input()` — Converts sample rate to the compatible rate (usually 48 kHz).
//! 3. **WaveNet/LSTM neural inference** — The neural engine that processes the audio signal.
//! 4. `NamResampler::process_output()` — Converts back to the original host sample rate.
//! 5. **Output Gain and Clipping** — Applies the final volume and detects digital saturation.
//! 6. **Write to [`DspBridge`]** — Publishes the result with `Ordering::Release` to the playback callback.
//!
//! When no model is loaded, the engine operates in **True-Bypass** (the input signal passes clean).
//! When the PipeWire sample rate is the same as the nam model, the resampler operates in bypass without overhead.

mod bridge;
mod capture;
mod playback;
mod rt_callback;
mod run;

pub use crate::dsp::pipeline::PipewireHostConfig;
pub use run::run_pipewire_host;

// Re-exports for test module compatibility (pw_host_test.rs).
#[cfg(test)]
pub(crate) use crate::dsp::pipeline::{BridgeBuffer, DspBridge, MAX_BRIDGE_BUF};

#[cfg(test)]
#[path = "pw_host_test.rs"]
mod pw_host_test;
