// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLAP-specific global allocator registration for heap audit.
//!
//! The shared `CountingAllocator` and tracking infrastructure lives in
//! [`crate::common::alloc_audit`].

#[cfg(all(feature = "clap-plugin", feature = "heap-audit", not(test)))]
#[global_allocator]
static GLOBAL: crate::common::alloc_audit::CountingAllocator =
    crate::common::alloc_audit::CountingAllocator;
