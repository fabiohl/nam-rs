// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#[cfg(feature = "clap-plugin")]
#[path = "clap/clap_lifecycle_test.rs"]
mod clap_lifecycle_test;
#[cfg(feature = "clap-plugin")]
#[path = "clap/clap_multi_instance.rs"]
mod clap_multi_instance;
#[cfg(feature = "clap-plugin")]
#[path = "clap/clap_state_migration.rs"]
mod clap_state_migration;
#[cfg(feature = "clap-plugin")]
#[path = "clap/tail_semantics.rs"]
mod tail_semantics;
