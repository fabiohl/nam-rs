#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

set -xeuo pipefail

echo "🔥 Starting long-duration stress tests..."
echo "⚠️ This operation is intensive and may take several minutes!"
echo "⚠️ It has taken about 20 minutes (assuming a well-populated target/ directory)"
rm -rf target/logs/
mkdir -p target/logs/

echo "==================================================="
echo "🧪 Running Soak Tests (Numerical Stability)..."
cargo test
cargo test --release --features standalone --test soak_test -- --ignored --nocapture --test-threads=1 2>&1 | tee target/logs/soak-test.log
cargo test --release --features standalone --test pipeline_soak -- --ignored --nocapture --test-threads=1 2>&1 | tee target/logs/pipeline-soak.log

echo "==================================================================="
echo "🧪 Running Property-Based and Parity Tests in Release..."
time cargo test --release --test proptest_parsers -- --ignored 2>&1 | tee target/logs/proptest-parsers.log
time cargo test --release --test proptest_math -- --ignored 2>&1 | tee target/logs/proptest-math.log
time cargo test --release --test lstm_gate_bf16_parity -- --ignored 2>&1 | tee target/logs/lstm-gate-bf16-parity.log
time cargo test --release --test lstm_scalar_bf16_parity -- --ignored 2>&1 | tee target/logs/lstm-scalar-bf16-parity.log
time cargo test --release --lib -- dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest --ignored 2>&1 | tee target/logs/pipeline-block-proptest.log

echo "==================================================================="
echo "🌐 Running NAM-rs ↔ NeuralAmpModelerCore Cross-Validation..."
cargo test --test cpp_parity -- --ignored --nocapture 2>&1 | tee target/logs/cpp-parity.log

echo "===================================================================="
echo "🛡️ Running CLAP compliance validation with Heap Alloc Audit..."
echo "🔨 Building in Debug mode with heap-audit..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

echo "========================================================"
echo "🔍 Validating plugin with clap-validator and heap-audit..."
NAM_HEAP_AUDIT=1 \
  clap-validator validate target/clap-test/debug/libnam_rs.so --json 2>target/logs/debug-validation.stderr.log | tee target/logs/debug-validation.json
jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' target/logs/debug-validation.json

echo "=========================================="
echo "🔨 Building in Release mode with heap-audit..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --release --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

echo "==========================================================="
echo "🔍 Validating plugin with clap-validator in release mode..."
NAM_HEAP_AUDIT=1 \
  clap-validator validate target/clap-test/release/libnam_rs.so --json 2>target/logs/release-validation.stderr.log | tee target/logs/release-validation.json
jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' target/logs/release-validation.json

echo "============================================================"
echo "🔬 Running CLAP Multi-Instance Stress Test (10 instances)..."
cargo test --no-default-features --features "clap-plugin" --test clap_multi_instance -- --ignored --nocapture 2>&1 | tee target/logs/clap-multi-instance.log

echo "=============================================="
echo "📊 Running Long Benchmarks (Performance)..."
cargo bench
cargo bench --features "standalone,long_bench" --bench inference_bench 2>&1 | tee target/logs/long-bench.log

echo "==================================="
echo "✅ Audit completed successfully!"
echo "📄 Logs: soak-test.log, pipeline-soak.log, long-bench.log, cpp-parity.log, debug-validation.json, release-validation.json, proptest-parsers.log, proptest-math.log, lstm-gate-bf16-parity.log, lstm-scalar-bf16-parity.log, pipeline-block-proptest.log, clap-multi-instance.log"
