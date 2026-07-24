// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Ring buffer `LogBuffer` — in-memory circular history of formatted log
//! entries, suitable for rendering into diagnostic dumps and crash reports.
//!
//! # RT-Safety
//!
//! The `LogBuffer` is strictly off-RT. RT callers must **never** invoke `log::*`
//! and therefore never touch this buffer from the audio thread. The public API
//! uses `std::sync::Mutex` — ordinary (blocking) `lock()` for off-RT use,
//! `try_lock()` for panic-hook contexts where deadlock must be avoided.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Default maximum number of log records held in the ring buffer.
const DEFAULT_CAPACITY: usize = 256;

/// A single log record stored in the `LogBuffer`.
#[derive(Debug, Clone)]
pub struct LogRecord {
    /// UNIX timestamp (seconds) when the log entry was emitted.
    pub timestamp_secs: u64,
    /// Log level string (`ERROR`, `WARN`, `INFO`, `DEBUG`, `TRACE`).
    pub level: String,
    /// Target / module that produced the log entry.
    pub target: String,
    /// Formatted log message.
    pub message: String,
}

impl LogRecord {
    /// Creates a new `LogRecord` with the current wall-clock timestamp.
    pub fn now(
        level: impl Into<String>,
        target: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            timestamp_secs: ts,
            level: level.into(),
            target: target.into(),
            message: message.into(),
        }
    }
}

/// Thread-safe ring buffer for retaining a bounded history of log entries.
///
/// Uses a `Mutex<VecDeque<LogRecord>>` with a fixed maximum capacity. When
/// capacity is exceeded, the oldest entry is discarded (FIFO). Designed
/// exclusively for off-RT use; the `try_push` variant is safe to call from
/// panic hooks because it falls back gracefully on lock contention.
#[derive(Debug)]
pub struct LogBuffer {
    inner: Mutex<VecDeque<LogRecord>>,
    capacity: usize,
}

impl Default for LogBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl LogBuffer {
    /// Creates a new `LogBuffer` with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LogBuffer capacity must be > 0");
        Self {
            inner: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
        }
    }

    /// Pushes a `LogRecord` into the buffer, discarding the oldest entry if
    /// the buffer has reached its capacity.
    ///
    /// Uses a blocking `lock()` — intended for off-RT callers.
    pub fn push(&self, record: LogRecord) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.len() >= self.capacity {
            inner.pop_front();
        }
        inner.push_back(record);
    }

    /// Attempts to push without blocking. Returns `false` if the lock could
    /// not be acquired (e.g. held by a panicking thread), `true` on success.
    ///
    /// Safe to call from panic hooks and crash-report paths.
    pub fn try_push(&self, record: LogRecord) -> bool {
        let Some(mut inner) = self.inner.try_lock().ok() else {
            return false;
        };
        if inner.len() >= self.capacity {
            inner.pop_front();
        }
        inner.push_back(record);
        true
    }

    /// Returns a snapshot of all currently buffered log records (cloned).
    /// Does not mutate the original state.
    pub fn snapshot(&self) -> Vec<LogRecord> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Attempts to take a snapshot without blocking. Returns `None` if the
    /// lock could not be acquired.
    pub fn try_snapshot(&self) -> Option<Vec<LogRecord>> {
        let inner = self.inner.try_lock().ok()?;
        Some(inner.iter().cloned().collect())
    }

    /// Renders up to `limit` most recent log entries as a formatted string
    /// suitable for embedding in diagnostic dumps and crash reports.
    ///
    /// Each line follows the format:
    /// `[timestamp_secs] LEVEL target: message`
    pub fn render_trace(&self, limit: usize) -> String {
        let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let start = if inner.len() > limit {
            inner.len() - limit
        } else {
            0
        };
        let mut out = String::with_capacity(limit * 128);
        for record in inner.range(start..) {
            out.push_str(&format!(
                "[{}] {} {}: {}\n",
                record.timestamp_secs, record.level, record.target, record.message
            ));
        }
        out
    }

    /// Attempts to render a trace without blocking. Returns `None` if the
    /// lock could not be acquired.
    pub fn try_render_trace(&self, limit: usize) -> Option<String> {
        let inner = self.inner.try_lock().ok()?;
        let start = if inner.len() > limit {
            inner.len() - limit
        } else {
            0
        };
        let mut out = String::with_capacity(limit * 128);
        for record in inner.range(start..) {
            out.push_str(&format!(
                "[{}] {} {}: {}\n",
                record.timestamp_secs, record.level, record.target, record.message
            ));
        }
        Some(out)
    }

    /// Returns the current number of records in the buffer.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Returns `true` if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the fixed capacity of the ring buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
#[path = "logger_test.rs"]
mod logger_test;
