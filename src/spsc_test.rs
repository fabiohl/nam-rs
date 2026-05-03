// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva.

use super::*;

#[test]
fn test_spsc_concurrency() {
    let (mut prod, mut cons) = RingBuffer::<i32>::new(64);
    let handle = std::thread::spawn(move || {
        let mut count = 0;
        while count < 1000 {
            if prod.push(count).is_ok() {
                count += 1;
            }
            std::thread::yield_now();
        }
    });

    let mut count = 0;
    while count < 1000 {
        if let Ok(val) = cons.pop() {
            assert_eq!(val, count);
            count += 1;
        }
        std::thread::yield_now();
    }
    handle.join().unwrap();
}

#[test]
fn test_spsc_full_empty() {
    let (mut prod, mut cons) = RingBuffer::<i32>::new(4);
    assert!(cons.pop().is_err());

    assert!(prod.push(1).is_ok());
    assert!(prod.push(2).is_ok());
    assert!(prod.push(3).is_ok());
    assert!(prod.push(4).is_ok());
    assert!(prod.push(5).is_err()); // Full at 4

    assert_eq!(cons.pop(), Ok(1));
    assert!(prod.push(5).is_ok());
    assert!(prod.push(6).is_err());
}
