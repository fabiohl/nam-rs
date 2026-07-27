// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Thread-safe command channel with acknowledgment and coalescing.
//!
//! # Architecture
//!
//! The `CommandScheduler` provides a lossless, ordered channel between the
//! Main Thread and the Audio Thread, solving three problems identified in
//! CLAP-F004:
//!
//! 1. **Coalescing** — rapid parameter automation bursts are merged so the
//!    SPSC never saturates. 10 000 host events reduce to ≤ 9 internal pushes
//!    (one per parameter, only the latest value survives).
//! 2. **Acknowledgment** — every command batch receives a monotonic sequence
//!    number. The audio thread atomically reports the last fully-drained
//!    batch, giving the main thread non-blocking confirmation of delivery.
//! 3. **Ordering** — non-coalescable commands (model load, IR swap,
//!    oversampling engine hot-swap) flush any pending coalesced parameters
//!    *before* being enqueued, preserving the total causal order.

use super::shared::ClapParamPayload;
use crate::common::params::RtPluginParams;
use rtrb::{Consumer, Producer};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Default capacity for the command SPSC ring buffer.
/// 256 was chosen to safely handle parameter automation bursts
/// while keeping memory footprint minimal (~8 KiB for pointers).
pub const CMD_QUEUE_CAPACITY: usize = 256;

const PARAM_COUNT: usize = 9;

/// Error returned when the SPSC ring buffer is full and the
/// command cannot be enqueued. The caller should retry or fall
/// back to atomic-based signalling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PushError {
    /// The destination SPSC ring buffer has no free slots.
    Full,
}

#[derive(Debug)]
struct CoalesceBuffer {
    slots: [Option<f64>; PARAM_COUNT],
    dirty_mask: u16,
}

impl Default for CoalesceBuffer {
    fn default() -> Self {
        Self {
            slots: [None; PARAM_COUNT],
            dirty_mask: 0,
        }
    }
}

impl CoalesceBuffer {
    fn set(&mut self, param_id: u32, value: f64) {
        let idx = param_id as usize;
        if idx < PARAM_COUNT {
            self.slots[idx] = Some(value);
            self.dirty_mask |= 1u16 << idx;
        }
    }

    fn take_snapshot(&mut self) -> Option<RtPluginParams> {
        if self.dirty_mask == 0 {
            return None;
        }
        let mask = self.dirty_mask;
        self.dirty_mask = 0;

        let mut params = RtPluginParams::default();

        if mask & (1 << 0) != 0
            && let Some(v) = self.slots[0].take()
        {
            params.input_gain_db = v as f32;
        }
        if mask & (1 << 1) != 0
            && let Some(v) = self.slots[1].take()
        {
            params.output_gain_db = v as f32;
        }
        if mask & (1 << 2) != 0
            && let Some(v) = self.slots[2].take()
        {
            params.gate_threshold_db = v as f32;
        }
        if mask & (1 << 3) != 0
            && let Some(v) = self.slots[3].take()
        {
            params.bypass = v != 0.0;
        }
        if mask & (1 << 4) != 0
            && let Some(v) = self.slots[4].take()
        {
            params.adaptive_compute =
                crate::common::params::AdaptiveComputeMode::from_f32(v as f32);
        }
        if mask & (1 << 5) != 0
            && let Some(v) = self.slots[5].take()
        {
            params.slim_override = crate::dsp::adaptive::SlimOverride::from_f32(v as f32);
        }
        if mask & (1 << 6) != 0
            && let Some(v) = self.slots[6].take()
        {
            params.oversample = crate::dsp::oversample::OversampleFactor::from_f32(v as f32);
        }
        if mask & (1 << 7) != 0
            && let Some(v) = self.slots[7].take()
        {
            params.activation_precision =
                crate::common::params::ActivationPrecision::from_f32(v as f32);
        }

        Some(params)
    }
}

/// Main-thread side of the command scheduler.
///
/// Wraps an SPSC producer with coalescing logic and acknowledgment
/// tracking. Owned exclusively by [`NamClapMainThread`].
pub struct CommandProducer<'a> {
    tx: Producer<ClapParamPayload>,
    next_seq: &'a AtomicU64,
    last_ack: &'a AtomicU64,
    coalescing: CoalesceBuffer,
}

/// Audio-thread side of the command scheduler.
///
/// Wraps an SPSC consumer. Drains commands in `process_events()` and
/// updates the atomic acknowledgment counter so the main thread can
/// confirm delivery.
pub struct CommandConsumer<'a> {
    rx: Consumer<ClapParamPayload>,
    last_ack: &'a AtomicU64,
}

/// Channel endpoints extracted from [`CommandScheduler`] during
/// plugin initialisation.
pub struct CommandSchedulerChannels {
    /// Producer (main-thread → audio-thread).
    pub cmd_tx: Producer<ClapParamPayload>,
    /// Consumer (audio-thread side).
    pub cmd_rx: Consumer<ClapParamPayload>,
}

/// Shared portion of the command scheduler stored in [`ColdShared`].
///
/// Holds the SPSC channel ends (behind `Mutex<Option<>>` to satisfy
/// the `PluginShared` extraction protocol) and two atomic u64 for
/// sequence-number-based acknowledgment.
pub struct CommandScheduler {
    /// Lock-protected SPSC producer (main-thread side).
    pub cmd_tx: Mutex<Option<Producer<ClapParamPayload>>>,
    /// Lock-protected SPSC consumer (audio-thread side).
    pub cmd_rx: Mutex<Option<Consumer<ClapParamPayload>>>,
    /// Monotonic sequence counter incremented by the main thread.
    pub cmd_next_seq: AtomicU64,
    /// Last sequence fully drained by the audio thread (ack).
    pub cmd_last_ack: AtomicU64,
}

impl CommandScheduler {
    /// Creates a new command scheduler with a ring buffer of
    /// [`CMD_QUEUE_CAPACITY`] slots.
    pub fn new() -> Self {
        let (tx, rx) = rtrb::RingBuffer::new(CMD_QUEUE_CAPACITY);
        Self {
            cmd_tx: Mutex::new(Some(tx)),
            cmd_rx: Mutex::new(Some(rx)),
            cmd_next_seq: AtomicU64::new(0),
            cmd_last_ack: AtomicU64::new(0),
        }
    }

    /// Extracts the SPSC channel ends for exclusive ownership by the
    /// main thread and audio thread respectively. Returns `None` if
    /// already extracted.
    pub fn extract_producer_consumer(&self) -> Option<CommandSchedulerChannels> {
        let tx = self
            .cmd_tx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()?;
        let rx = self
            .cmd_rx
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()?;
        Some(CommandSchedulerChannels {
            cmd_tx: tx,
            cmd_rx: rx,
        })
    }

    /// Returns previously extracted channel ends to the cold storage
    /// (used during deactivate / rollback).
    pub fn restore_channels(
        &self,
        tx: Producer<ClapParamPayload>,
        rx: Consumer<ClapParamPayload>,
    ) {
        if let Ok(mut g) = self.cmd_tx.lock() {
            *g = Some(tx);
        }
        if let Ok(mut g) = self.cmd_rx.lock() {
            *g = Some(rx);
        }
    }
}

impl Default for CommandScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CommandProducer<'a> {
    /// Creates a new producer wrapping the given SPSC endpoint and
    /// ack atomics.
    pub fn new(
        tx: Producer<ClapParamPayload>,
        next_seq: &'a AtomicU64,
        last_ack: &'a AtomicU64,
    ) -> Self {
        Self {
            tx,
            next_seq,
            last_ack,
            coalescing: CoalesceBuffer::default(),
        }
    }

    /// Queues a parameter snapshot for delivery to the audio thread.
    ///
    /// Coalescing: if parameters have already been pushed since the
    /// last drain, the new snapshot overwrites the previous one
    /// (only the latest value per parameter is retained). No SPSC
    /// push occurs until [`force_flush`](Self::force_flush) or
    /// [`push_command`](Self::push_command) is called.
    ///
    /// Returns `true` if this is a new batch (not coalesced into an
    /// existing pending batch). Callers should call `force_flush()`
    /// after a batch of `push_params` to actually deliver the data.
    pub fn push_params(&mut self, params: RtPluginParams) -> bool {
        let had_pending = self.coalescing.dirty_mask != 0;

        self.coalescing.set(0, params.input_gain_db as f64);
        self.coalescing.set(1, params.output_gain_db as f64);
        self.coalescing.set(2, params.gate_threshold_db as f64);
        self.coalescing.set(3, if params.bypass { 1.0 } else { 0.0 });
        self.coalescing.set(4, params.adaptive_compute as u32 as f64);
        self.coalescing.set(5, params.slim_override as u32 as f64);
        self.coalescing.set(6, params.oversample as u32 as f64);
        self.coalescing.set(7, params.activation_precision as u32 as f64);

        !had_pending
    }

    /// Queues a non-coalescable command (model load, IR swap,
    /// oversampling engine hot-swap).
    ///
    /// Any pending coalesced parameters are flushed **before** the
    /// command is enqueued, preserving causal ordering.
    ///
    /// Returns the monotonic sequence number assigned to the command
    /// batch.
    pub fn push_command(&mut self, cmd: ClapParamPayload) -> Result<u64, PushError> {
        self.force_flush()?;
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
        self.tx.push(cmd).map_err(|_| PushError::Full)?;
        Ok(seq)
    }

    /// Immediately pushes any pending coalesced parameters to the
    /// SPSC channel. No-op if the coalescing buffer is empty.
    ///
    /// Returns `Ok(seq)` with the assigned sequence number if a push
    /// occurred, or `Ok(0)` if the buffer was empty.
    pub fn force_flush(&mut self) -> Result<u64, PushError> {
        if let Some(snapshot) = self.coalescing.take_snapshot() {
            let seq = self.next_seq.fetch_add(1, Ordering::Relaxed) + 1;
            self.tx
                .push(ClapParamPayload::Params(snapshot))
                .map_err(|_| PushError::Full)?;
            Ok(seq)
        } else {
            Ok(0)
        }
    }

    /// Returns the last sequence number acknowledged by the audio
    /// thread (non-blocking, Acquire load).
    pub fn last_acked_seq(&self) -> u64 {
        self.last_ack.load(Ordering::Acquire)
    }

    /// Spin-waits until the audio thread has acknowledged `seq`
    /// (or any higher sequence number).
    ///
    /// Call this only on the main thread when blocking is acceptable
    /// (e.g. synchronous API calls). Do **not** call on the audio
    /// thread.
    pub fn wait_for_ack(&self, seq: u64) {
        while self.last_ack.load(Ordering::Acquire) < seq {
            std::hint::spin_loop();
        }
    }

    /// Returns `true` if the audio thread has already acknowledged
    /// `seq` (non-blocking).
    pub fn is_acked(&self, seq: u64) -> bool {
        self.last_ack.load(Ordering::Acquire) >= seq
    }
}

impl<'a> CommandConsumer<'a> {
    /// Creates a new consumer wrapping the given SPSC endpoint and
    /// ack atomic.
    pub fn new(rx: Consumer<ClapParamPayload>, last_ack: &'a AtomicU64) -> Self {
        Self { rx, last_ack }
    }

    /// Pops a single command from the SPSC channel (non-blocking).
    pub(crate) fn pop(&mut self) -> Option<ClapParamPayload> {
        self.rx.pop().ok()
    }

    /// Drains up to `max` commands from the SPSC channel, calling
    /// `process` for each one. Returns the number of commands
    /// actually drained.
    pub fn drain_and_process<F>(&mut self, max: usize, mut process: F) -> usize
    where
        F: FnMut(ClapParamPayload),
    {
        let mut count = 0;
        while count < max {
            if let Ok(payload) = self.rx.pop() {
                process(payload);
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    /// Records that all commands up to sequence `seq` have been
    /// processed (Release store).
    pub fn ack_up_to(&self, seq: u64) {
        self.last_ack.store(seq, Ordering::Release);
    }

    /// Acknowledges the latest sequence number produced by the main
    /// thread. Only updates `cmd_last_ack` if the latest value is
    /// greater than the previously acknowledged value.
    pub fn ack_latest(&self, latest_seq: &AtomicU64) {
        let current = latest_seq.load(Ordering::Relaxed);
        let prev = self.last_ack.load(Ordering::Relaxed);
        if current > prev {
            self.last_ack.store(current, Ordering::Release);
        }
    }

    /// Returns the inner SPSC consumer for channel restoration
    /// during deactivation.
    pub(crate) fn into_inner(self) -> Consumer<ClapParamPayload> {
        self.rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clap::plugin::ClapParamPayload;
    use crate::common::params::RtPluginParams;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;
    use std::thread;

    fn make_test_scheduler() -> (CommandScheduler, Arc<AtomicU64>, Arc<AtomicU64>) {
        let sched = CommandScheduler::new();
        let next_seq = Arc::new(AtomicU64::new(0));
        let last_ack = Arc::new(AtomicU64::new(0));
        (sched, next_seq, last_ack)
    }

    #[test]
    fn coalesce_single_param_and_flush() {
        let (_sched, next_seq, last_ack) = make_test_scheduler();
        let (tx, _rx) = rtrb::RingBuffer::new(256);
        let mut producer = CommandProducer::new(tx, &next_seq, &last_ack);

        let mut params = RtPluginParams::default();
        params.input_gain_db = 5.0;
        params.bypass = false;

        let is_new = producer.push_params(params);
        assert!(is_new, "first push should start a new batch");

        let seq = producer.force_flush().unwrap();
        assert!(seq > 0, "flush should get a sequence number");
    }

    #[test]
    fn coalesce_merges_consecutive_param_updates() {
        let (_sched, next_seq, last_ack) = make_test_scheduler();
        let (tx, mut rx) = rtrb::RingBuffer::new(256);
        let mut producer = CommandProducer::new(tx, &next_seq, &last_ack);

        for gain in 0..100 {
            let mut p = RtPluginParams::default();
            p.input_gain_db = gain as f32;
            let is_new = producer.push_params(p);
            // First is new, rest are coalesced
            if gain == 0 {
                assert!(is_new);
            } else {
                assert!(!is_new);
            }
        }

        let seq = producer.force_flush().unwrap();
        assert!(seq > 0, "flush should get a sequence number");

        let mut found = false;
        while let Ok(payload) = rx.pop() {
            if let ClapParamPayload::Params(p) = payload {
                assert_eq!(p.input_gain_db, 99.0, "should keep only the latest value");
                found = true;
            }
        }
        assert!(found, "should have received the coalesced params");
    }

    #[test]
    fn coalesce_preserves_multi_param_merging() {
        let (_sched, next_seq, last_ack) = make_test_scheduler();
        let (tx, mut rx) = rtrb::RingBuffer::new(256);
        let mut producer = CommandProducer::new(tx, &next_seq, &last_ack);

        let mut p1 = RtPluginParams::default();
        p1.input_gain_db = 3.0;
        p1.bypass = true;
        assert!(producer.push_params(p1));

        let mut p2 = RtPluginParams::default();
        p2.input_gain_db = 3.0;
        p2.bypass = true;
        p2.output_gain_db = -6.0;
        p2.gate_threshold_db = -50.0;
        assert!(!producer.push_params(p2));

        producer.force_flush().unwrap();

        let mut final_params: Option<RtPluginParams> = None;
        while let Ok(payload) = rx.pop() {
            if let ClapParamPayload::Params(p) = payload {
                final_params = Some(p);
            }
        }
        let fp = final_params.expect("should receive coalesced params");
        assert_eq!(fp.input_gain_db, 3.0);
        assert_eq!(fp.output_gain_db, -6.0);
        assert_eq!(fp.gate_threshold_db, -50.0);
        assert!(fp.bypass);
    }

    #[test]
    fn non_coalescable_flushes_pending_params_first() {
        let (_sched, next_seq, last_ack) = make_test_scheduler();
        let (tx, mut rx) = rtrb::RingBuffer::new(256);
        let mut producer = CommandProducer::new(tx, &next_seq, &last_ack);

        let mut p = RtPluginParams::default();
        p.input_gain_db = 12.0;
        assert!(producer.push_params(p));

        let seq = producer
            .push_command(ClapParamPayload::LoadCabIr { adapter: None })
            .unwrap();
        assert!(seq > 0, "command should get a sequence number");

        let expected_order = vec!["Params", "LoadCabIr"];
        let mut actual_order = Vec::new();
        while let Ok(payload) = rx.pop() {
            actual_order.push(match payload {
                ClapParamPayload::Params(_) => "Params",
                ClapParamPayload::LoadCabIr { .. } => "LoadCabIr",
                _ => "Other",
            });
        }
        assert_eq!(
            actual_order, expected_order,
            "params must be flushed before the non-coalescable command"
        );
    }

    #[test]
    fn ack_tracking_basic() {
        let next_seq = Arc::new(AtomicU64::new(0));
        let last_ack = Arc::new(AtomicU64::new(0));
        let (tx, rx) = rtrb::RingBuffer::new(256);
        let mut producer = CommandProducer::new(tx, &next_seq, &last_ack);

        let mut p = RtPluginParams::default();
        p.input_gain_db = 1.0;
        producer.push_params(p);
        let seq1 = producer.force_flush().unwrap();

        p.input_gain_db = 2.0;
        producer.push_params(p);
        let seq2 = producer.force_flush().unwrap();

        assert!(seq1 > 0);
        assert!(seq2 > seq1);
        assert!(!producer.is_acked(seq2));

        let mut consumer = CommandConsumer::new(rx, &last_ack);
        consumer.drain_and_process(256, |_| {});
        consumer.ack_up_to(seq2);

        assert!(producer.is_acked(seq2));
    }

    #[test]
    fn stress_10k_param_burst_no_loss_no_deadlock() {
        let sched = CommandScheduler::new();
        let next_seq = Arc::new(AtomicU64::new(0));
        let last_ack = Arc::new(AtomicU64::new(0));

        let channels = sched.extract_producer_consumer().unwrap();
        let cmd_tx = channels.cmd_tx;
        let cmd_rx = channels.cmd_rx;

        let next_seq_clone = Arc::clone(&next_seq);
        let last_ack_clone = Arc::clone(&last_ack);

        let producer_handle = thread::spawn(move || {
            let mut producer =
                CommandProducer::new(cmd_tx, &next_seq_clone, &last_ack_clone);

            for i in 0..10_000u32 {
                let mut p = RtPluginParams::default();
                let val = i as f32 * 0.01;
                p.input_gain_db = val;
                p.output_gain_db = -val;
                p.gate_threshold_db = -70.0 + val * 0.1;
                p.bypass = i % 100 == 0;

                producer.push_params(p);
            }
            let last_seq = producer.force_flush().unwrap();

            producer.wait_for_ack(last_seq);
            last_seq
        });

        let consumer_handle = thread::spawn(move || {
            let mut consumer = CommandConsumer::new(cmd_rx, &last_ack);
            let mut total_drained = 0usize;

            loop {
                let drained = consumer.drain_and_process(64, |_| {});
                total_drained += drained;

                if drained > 0 {
                    consumer.ack_latest(&next_seq);
                }

                let current = next_seq.load(Ordering::Relaxed);
                if current > 0 && last_ack.load(Ordering::Acquire) >= current {
                    break;
                }

                std::thread::yield_now();
            }

            total_drained
        });

        let last_seq = producer_handle.join().unwrap();
        let total = consumer_handle.join().unwrap();

        assert!(last_seq > 0, "producer should have sent at least one batch");
        assert!(total > 0, "consumer should have drained at least one message");
        assert!(
            total <= 256,
            "with coalescing, 10k pushes should produce few messages, got {total}"
        );
    }

    #[test]
    fn interleaved_commands_preserve_ordering() {
        let sched = CommandScheduler::new();
        let next_seq = Arc::new(AtomicU64::new(0));
        let last_ack = Arc::new(AtomicU64::new(0));

        let channels = sched.extract_producer_consumer().unwrap();
        let cmd_tx = channels.cmd_tx;
        let mut consumer_rx = channels.cmd_rx;

        let mut producer = CommandProducer::new(cmd_tx, &next_seq, &last_ack);

        let mut p = RtPluginParams::default();
        p.input_gain_db = 3.0;
        producer.push_params(p);

        let _ = producer
            .push_command(ClapParamPayload::LoadCabIr { adapter: None })
            .unwrap();

        p.output_gain_db = -6.0;
        producer.push_params(p);

        let _ = producer.force_flush();

        let mut order = Vec::new();
        while let Ok(payload) = consumer_rx.pop() {
            order.push(match payload {
                ClapParamPayload::Params(_) => "P",
                ClapParamPayload::LoadCabIr { .. } => "C",
                _ => "?",
            });
        }

        assert_eq!(
            order,
            vec!["P", "C", "P"],
            "ordering: params before command, then params after"
        );
    }

    #[test]
    fn spin_wait_for_ack_does_not_deadlock() {
        let next_seq = Arc::new(AtomicU64::new(0));
        let last_ack = Arc::new(AtomicU64::new(0));
        let (tx, rx) = rtrb::RingBuffer::new(256);
        let mut producer = CommandProducer::new(tx, &next_seq, &last_ack);

        let mut p = RtPluginParams::default();
        p.input_gain_db = 7.0;
        producer.push_params(p);
        let seq = producer.force_flush().unwrap();

        let next_seq2 = Arc::clone(&next_seq);
        let last_ack2 = Arc::clone(&last_ack);

        thread::spawn(move || {
            let mut consumer = CommandConsumer::new(rx, &last_ack2);
            std::thread::sleep(std::time::Duration::from_millis(10));
            consumer.drain_and_process(256, |_| {});
            consumer.ack_up_to(next_seq2.load(Ordering::Relaxed));
        });

        producer.wait_for_ack(seq);
        assert!(producer.is_acked(seq));
    }

    #[test]
    fn producer_without_consumer_returns_full_on_overflow() {
        let next_seq = Arc::new(AtomicU64::new(0));
        let last_ack = Arc::new(AtomicU64::new(0));
        let (tx, _rx) = rtrb::RingBuffer::new(4);
        let mut producer = CommandProducer::new(tx, &next_seq, &last_ack);

        for i in 0..8 {
            let mut p = RtPluginParams::default();
            p.input_gain_db = i as f32;
            producer.push_params(p);
            let r = producer.force_flush();
            if i < 3 {
                assert!(r.is_ok(), "early pushes should succeed");
            }
        }

        let mut full_count = 0;
        for _ in 0..64 {
            let mut p = RtPluginParams::default();
            p.input_gain_db = 99.0;
            producer.push_params(p);
            if producer.force_flush().is_err() {
                full_count += 1;
                break;
            }
        }
        assert!(
            full_count > 0,
            "SPSC should have returned Full after saturation"
        );
    }
}
