// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

mod common;

use common::alloc_audit::CountingAllocator;

#[cfg_attr(
    not(all(feature = "clap-plugin", feature = "heap-audit")),
    global_allocator
)]
#[allow(dead_code, clippy::allow_attributes)]
static GLOBAL: CountingAllocator = CountingAllocator;

#[path = "rt_constraints/a2_heap_audit.rs"]
mod a2_heap_audit;
#[path = "rt_constraints/cabsim_heap_audit.rs"]
mod cabsim_heap_audit;
#[path = "rt_constraints/resampler_heap_audit.rs"]
mod resampler_heap_audit;
#[path = "rt_constraints/rt_deadline.rs"]
mod rt_deadline;
#[path = "rt_constraints/rt_jitter.rs"]
mod rt_jitter;
