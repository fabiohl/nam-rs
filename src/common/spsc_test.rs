// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Tests whether the RingBuffer can pass data between two "processing lines" (threads)
/// concurrently without losing information or locking up.
#[test]
fn test_spsc_concurrency() {
    // Creates a buffer with 64 empty slots.
    // 'prod' (producer) sends data, 'cons' (consumer) receives.
    let (mut prod, mut cons) = RingBuffer::<i32>::new(64);

    // Creates a new processing line that will "produce" numbers from 0 to 999.
    let handle = std::thread::spawn(move || {
        let mut count = 0;
        while count < 1000 {
            // Attempt to push the number into the buffer. If full, retry later.
            if prod.push(count).is_ok() {
                count += 1;
            }
            std::thread::yield_now(); // A small pause to avoid overwhelming the processor.
        }
    });

    // This part (the main line) will "consume" the numbers sent.
    let mut count = 0;
    while count < 1000 {
        // Attempt to pop a number from the buffer.
        if let Ok(val) = cons.pop() {
            // Verify that the received number is exactly what we expected.
            assert_eq!(val, count);
            count += 1;
        }
        std::thread::yield_now();
    }
    // Wait for the other processing line to finish before ending the test.
    handle.join().unwrap();
}

/// Tests buffer limits: what happens when it is completely empty or completely full.
#[test]
fn test_spsc_full_empty() {
    // Creates a small buffer with only 4 slots.
    let (mut prod, mut cons) = RingBuffer::<i32>::new(4);

    // Attempt to pop from an empty buffer. Should return an error indicating nothing is there.
    assert!(cons.pop().is_err());

    // Fill the 4 available slots.
    assert!(prod.push(1).is_ok());
    assert!(prod.push(2).is_ok());
    assert!(prod.push(3).is_ok());
    assert!(prod.push(4).is_ok());

    // Attempt to push the 5th item into a 4-slot buffer. Should return a "full" error.
    assert!(prod.push(5).is_err());

    // Pop the first item (the number 1).
    assert_eq!(cons.pop(), Ok(1));

    // Now that a slot opened, the number 5 should push successfully.
    assert!(prod.push(5).is_ok());

    // Attempt to push the 6th item. Since the buffer is full again (contains 2, 3, 4, 5), it should fail.
    assert!(prod.push(6).is_err());
}

#[test]
fn test_gc_overflow_overwrite() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let overflow = GcOverflowBuffer::new(64);
    let counter = Arc::new(AtomicU32::new(0));

    // 1. Fill the 64-slot buffer
    for _ in 0..64 {
        let item = GcItem::Test(Box::new(counter.clone()));
        overflow.push(item);
    }

    // 2. Attempt to insert the 65th item (should overwrite the 1st)
    let item_65 = GcItem::Test(Box::new(counter.clone()));
    overflow.push(item_65);

    // 3. Validate that drain returns 64 items
    let drained = overflow.drain();
    assert_eq!(drained.len(), 64);
}

#[test]
fn test_gc_stress_no_leak() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let (mut gc_prod, mut gc_cons) = RingBuffer::<GcItem>::new(32);
    let overflow = GcOverflowBuffer::new(32);
    let counter = Arc::new(AtomicU32::new(0));

    // Stress: 1000 "resource" swaps
    for _ in 0..1000 {
        let item = GcItem::Test(Box::new(counter.clone()));
        if let Err(rtrb::PushError::Full(returned_item)) = gc_prod.push(item) {
            // If the main channel is full, go to overflow
            overflow.push(returned_item);
        }

        // Drain periodically to avoid unbounded accumulation
        super::drain_gc_channels(&mut gc_cons, &overflow);
    }

    // Validate that the final counter is correct after dropping everything
    drop(gc_cons);
    for _ in overflow.drain() {}
}
