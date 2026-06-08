// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(test)]
pub(crate) mod infra {
    pub use crate::common::alloc_audit::{ALLOC_COUNT, CountingAllocator, TrackingGuard};
}
