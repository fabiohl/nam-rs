#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

set -xeuo pipefail

echo "🔥 Iniciando testes de estresse de longa duração..."
echo "⚠️ Esta operação é intensiva e pode durar vários minutos!"
echo "⚠️ Tem demorado por volta de 20 minutos (considerando diretório target/ devidamente povoado)"
date

echo "==================================================="
echo "🧪 Executando Soak Tests (Estabilidade Numérica)..."
cargo test
cargo test --release --features standalone --test soak_test -- --ignored --nocapture --test-threads=1 2>&1 | tee soak-test.log

echo "==================================================================="
echo "🧪 Executando Property-Based e Parity Tests em Release..."
time cargo test --release --test proptest_parsers -- --ignored 2>&1 | tee proptest-parsers.log
time cargo test --release --test proptest_math -- --ignored 2>&1 | tee proptest-math.log
time cargo test --release --test lstm_gate_bf16_parity -- --ignored 2>&1 | tee lstm-gate-bf16-parity.log
time cargo test --release --test lstm_scalar_bf16_parity -- --ignored 2>&1 | tee lstm-scalar-bf16-parity.log
time cargo test --release --lib -- dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest --ignored 2>&1 | tee pipeline-block-proptest.log

echo "==================================================================="
echo "🌐 Executando Cross-Validation NAM-rs ↔ NeuralAmpModelerCore..."
cargo test --test cpp_parity -- --ignored --nocapture 2>&1 | tee cpp-parity.log

echo "===================================================================="
echo "🛡️ Executando validação de conformidade CLAP com Heap Alloc Audit..."
echo "🔨 Compilando em Debug com heap-audit..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

echo "========================================================"
echo "🔍 Validando o plugin com clap-validator e heap-audit..."
NAM_HEAP_AUDIT=1 \
  clap-validator validate target/clap-test/debug/libnam_rs.so --json > debug-validation.json
jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' debug-validation.json

echo "=========================================="
echo "🔨 Compilando em Release com heap-audit..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --release --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

echo "==========================================================="
echo "🔍 Validando o plugin com clap-validator no modo release..."
NAM_HEAP_AUDIT=1 \
  clap-validator validate target/clap-test/release/libnam_rs.so --json > release-validation.json
jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' release-validation.json

echo "============================================================"
echo "🔬 Executando Stress Test Multi-Instância CLAP (10 instâncias)..."
cargo test --no-default-features --features "clap-plugin" --test clap_multi_instance -- --ignored --nocapture 2>&1 | tee clap-multi-instance.log

echo "=============================================="
echo "📊 Executando Long Benchmarks (Performance)..."
cargo bench
cargo bench --features "standalone,long_bench" --bench inference_bench 2>&1 | tee long-bench.log

echo "==================================="
echo "✅ Auditoria concluída com sucesso!"
echo "📄 Logs: soak-test.log, long-bench.log, cpp-parity.log, debug-validation.json, release-validation.json, proptest-parsers.log, proptest-math.log, lstm-gate-bf16-parity.log, lstm-scalar-bf16-parity.log, pipeline-block-proptest.log, clap-multi-instance.log"
