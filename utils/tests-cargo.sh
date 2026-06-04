#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# This entire script takes about 35-40 seconds to complete (assuming a well-populated target/ directory)
set -xeuo pipefail

# 1. Run the standard unit and integration test suite [Usually takes about 25 seconds]
cargo test

# 2. Build CLAP plugin specifically for tests (in debug mode with heap-audit)
echo "🔨 Building CLAP plugin for local tests..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

# 3. Run plugin lifecycle tests targeting the isolated binary
echo "🧪 Running lifecycle tests with heap-audit..."
CLAP_PLUGIN_PATH="target/clap-test/debug/libnam_rs.so" \
  NAM_HEAP_AUDIT=1 \
  cargo test --test clap_lifecycle_test --features "clap-plugin" --target-dir target/clap-test

# 4. Run the official CLAP validator on the debug binary with heap-audit enabled if available
echo "🔍 Validating plugin with clap-validator and heap-audit..."
if command -v clap-validator >/dev/null 2>&1; then
  CLAP_PLUGIN_PATH="target/clap-test/debug/libnam_rs.so" \
    NAM_HEAP_AUDIT=1 \
    clap-validator validate target/clap-test/debug/libnam_rs.so
else
  echo "⚠️ WARNING: clap-validator not found. Skipping CLAP validation."
fi

# 5. Run basic benchmarks [Usually takes about 9 minutes]
#At this stage it's more useful to only run them via utils/tests-long.sh
#cargo bench inference_bench
