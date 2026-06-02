#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

set -xeuo pipefail

echo "🔥 Iniciando testes de estresse de longa duração..."
echo "⚠️ Esta operação é intensiva e pode durar vários minutos!"
date

echo "==================================================="
echo "🧪 Executando Soak Tests (Estabilidade Numérica)..."
time cargo test
time cargo test --release --features standalone --test soak_test -- --ignored --nocapture --test-threads=1 2>&1 | tee soak-test.log

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

echo "=============================================="
echo "📊 Executando Long Benchmarks (Performance)..."
time cargo bench
time cargo bench --features "standalone,long_bench" --bench inference_bench 2>&1 | tee long-bench.log

echo "==================================================================="
echo "🌐 Executando Cross-Validation NAM-rs ↔ NeuralAmpModelerCore..."
cargo test --test cpp_parity -- --ignored --nocapture 2>&1 | tee cpp-parity.log

echo -e "\n==================================="
echo "✅ Auditoria concluída com sucesso!"
echo "📄 Logs: soak-test.log, long-bench.log, cpp-parity.log, debug-validation.json, release-validation.json"
date
