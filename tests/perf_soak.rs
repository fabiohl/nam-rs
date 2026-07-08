// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

mod common;

use common::alloc_audit::CountingAllocator;

#[cfg_attr(not(all(feature = "clap-plugin", feature = "heap-audit")), global_allocator)]
static GLOBAL: CountingAllocator = CountingAllocator;

#[path = "perf_soak/concurrency_stress.rs"]
mod concurrency_stress;
#[path = "perf_soak/pipeline_soak.rs"]
mod pipeline_soak;
#[cfg(feature = "standalone")]
#[path = "perf_soak/pw_integration_test.rs"]
mod pw_integration_test;
#[path = "perf_soak/soak_test.rs"]
mod soak_test;
#[path = "perf_soak/spsc_pipeline.rs"]
mod spsc_pipeline;
