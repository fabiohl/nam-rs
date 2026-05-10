#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

set -euo pipefail

echo "🔥 Iniciando testes de estresse de longa duração..."
echo "⚠️ Esta operação é intensiva e pode durar vários minutos."

echo "🧪 1/2 Executando Soak Tests (Estabilidade Numérica)..."
date
cargo test --release --features standalone --test soak_test -- --ignored --nocapture --test-threads=1 2>&1 | tee soak-test.log

echo "📊 2/2 Executando Long Benchmarks (Performance)..."
date
cargo bench --features "standalone,long_bench" --bench inference_bench 2>&1 | tee long-bench.log


echo -e "\n✅ Auditoria concluída com sucesso!"
echo "📄 Logs: soak-test.log, long-bench.log"
echo "📈 Relatório visual: target/criterion/report/index.html"
