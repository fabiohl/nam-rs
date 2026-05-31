// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use nam_rs::dsp::mirror_buf::{MirroredBuffer, set_simulate_fail};

#[test]
fn test_mirror_buf_mmap_failure_injection() {
    // Activating failure simulation
    set_simulate_fail(true);

    let res = MirroredBuffer::<f32>::new(1024);
    assert!(
        res.is_err(),
        "Expected MirroredBuffer::new to fail when SIMULATE_FAIL is active"
    );

    // Deactivating failure simulation
    set_simulate_fail(false);

    let res = MirroredBuffer::<f32>::new(1024);
    assert!(
        res.is_ok(),
        "Expected MirroredBuffer::new to succeed when SIMULATE_FAIL is inactive"
    );
}
