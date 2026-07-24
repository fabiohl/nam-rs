// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::sync::Arc;
use std::thread;

#[test]
fn log_buffer_default_capacity() {
    let buf = LogBuffer::default();
    assert_eq!(buf.capacity(), 256);
    assert!(buf.is_empty());
}

#[test]
fn log_buffer_custom_capacity() {
    let buf = LogBuffer::new(8);
    assert_eq!(buf.capacity(), 8);
    assert!(buf.is_empty());
}

#[test]
#[should_panic(expected = "capacity must be > 0")]
fn log_buffer_zero_capacity_panics() {
    LogBuffer::new(0);
}

#[test]
fn push_single_record() {
    let buf = LogBuffer::new(4);
    buf.push(LogRecord::now("INFO", "test", "hello"));
    assert_eq!(buf.len(), 1);

    let snap = buf.snapshot();
    assert_eq!(snap.len(), 1);
    assert_eq!(snap[0].level, "INFO");
    assert_eq!(snap[0].target, "test");
    assert_eq!(snap[0].message, "hello");
    assert!(snap[0].timestamp_secs > 0);
}

#[test]
fn push_up_to_capacity() {
    let buf = LogBuffer::new(3);
    for i in 0..3 {
        buf.push(LogRecord::now("INFO", "test", format!("msg{}", i)));
    }
    assert_eq!(buf.len(), 3);
}

#[test]
fn overflow_discards_oldest_fifo() {
    let buf = LogBuffer::new(3);
    for i in 0..5 {
        buf.push(LogRecord::now("INFO", "test", format!("msg{}", i)));
    }
    assert_eq!(buf.len(), 3);

    let snap = buf.snapshot();
    assert_eq!(snap[0].message, "msg2");
    assert_eq!(snap[1].message, "msg3");
    assert_eq!(snap[2].message, "msg4");
}

#[test]
fn snapshot_does_not_mutate_original() {
    let buf = LogBuffer::new(3);
    buf.push(LogRecord::now("INFO", "test", "a"));
    buf.push(LogRecord::now("WARN", "test", "b"));

    let snap1 = buf.snapshot();
    assert_eq!(snap1.len(), 2);

    buf.push(LogRecord::now("ERROR", "test", "c"));

    let snap2 = buf.snapshot();
    assert_eq!(snap2.len(), 3);

    assert_eq!(snap1.len(), 2);
    assert_eq!(snap1[0].message, "a");
    assert_eq!(snap1[1].message, "b");
}

#[test]
fn render_trace_within_limit() {
    let buf = LogBuffer::new(10);
    buf.push(LogRecord::now("INFO", "mod", "one"));
    buf.push(LogRecord::now("WARN", "mod", "two"));

    let trace = buf.render_trace(3);
    assert!(trace.contains("[") && trace.contains("INFO mod: one"));
    assert!(trace.contains("WARN mod: two"));
    assert_eq!(trace.matches('\n').count(), 2);
}

#[test]
fn render_trace_clips_to_limit() {
    let buf = LogBuffer::new(10);
    for i in 0..5 {
        buf.push(LogRecord::now("INFO", "t", format!("m{}", i)));
    }
    let trace = buf.render_trace(2);
    assert_eq!(trace.matches('\n').count(), 2);
    assert!(trace.contains("m3"));
    assert!(trace.contains("m4"));
    assert!(!trace.contains("m2"));
}

#[test]
fn render_trace_limit_beyond_capacity_shows_all() {
    let buf = LogBuffer::new(4);
    for i in 0..4 {
        buf.push(LogRecord::now("INFO", "t", format!("m{}", i)));
    }
    let trace = buf.render_trace(100);
    assert_eq!(trace.matches('\n').count(), 4);
}

#[test]
fn concurrent_push() {
    let buf = Arc::new(LogBuffer::new(256));
    let threads: Vec<_> = (0..8)
        .map(|t| {
            let buf = Arc::clone(&buf);
            thread::spawn(move || {
                for i in 0..100 {
                    buf.push(LogRecord::now(
                        "INFO",
                        "concurrent",
                        format!("t{}_msg{}", t, i),
                    ));
                }
            })
        })
        .collect();

    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(buf.len(), 256);
}

#[test]
fn try_push_succeeds_when_uncontested() {
    let buf = LogBuffer::new(8);
    assert!(buf.try_push(LogRecord::now("INFO", "t", "x")));
    assert_eq!(buf.len(), 1);
}

#[test]
fn try_snapshot_returns_some() {
    let buf = LogBuffer::new(4);
    buf.push(LogRecord::now("INFO", "t", "a"));
    let snap = buf.try_snapshot().unwrap();
    assert_eq!(snap.len(), 1);
}

#[test]
fn try_render_trace_returns_some() {
    let buf = LogBuffer::new(4);
    buf.push(LogRecord::now("WARN", "m", "msg"));
    let trace = buf.try_render_trace(10).unwrap();
    assert!(trace.contains("WARN m: msg"));
}

#[test]
fn try_operations_on_contended_lock_during_panic() {
    let buf = Arc::new(LogBuffer::new(32));

    let held = buf.inner.lock().unwrap();

    let buf_clone = Arc::clone(&buf);
    let handle = thread::spawn(move || {
        assert!(!buf_clone.try_push(LogRecord::now("INFO", "t", "should_fail")));
        assert!(buf_clone.try_snapshot().is_none());
        assert!(buf_clone.try_render_trace(10).is_none());
    });
    handle.join().unwrap();
    drop(held);
}

#[test]
fn len_and_is_empty() {
    let buf = LogBuffer::new(16);
    assert!(buf.is_empty());
    buf.push(LogRecord::now("INFO", "t", "x"));
    assert!(!buf.is_empty());
    assert_eq!(buf.len(), 1);

    for _ in 0..20 {
        buf.push(LogRecord::now("DEBUG", "t", "fill"));
    }
    assert_eq!(buf.len(), 16);
}
