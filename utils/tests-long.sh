#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Intensive long-duration verification suite for nam-rs.
# Serves as a continuation of utils/tests-cargo.sh. First runs standard tests,
# then proceeds with numerical soak tests, proptest fuzzing, NeuralAmpModelerCore parity checks,
# CLAP release compliance, multi-instance stress tests, and long-running performance benchmarks.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}          nam-rs Long-Duration Stress & Audit Suite             ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. Run standard test suite first (continuation pattern)
echo -e "\n${BLUE}${BOLD}[Phase 1/7] Iniciando a suíte de testes padrão (tests-cargo.sh)...${NC}"
./utils/tests-cargo.sh

# 2. Soak/Stress tests setup
echo -e "\n${BLUE}${BOLD}[Phase 2/7] Configurando testes de Soak e Estresse...${NC}"
rm -rf target/logs/
mkdir -p target/logs/

# Auto-clone NeuralAmpModelerCore if not present (helps in fresh clones)
if [ ! -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    echo "🌐 Clonando NeuralAmpModelerCore para testes de paridade..."
    git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git tests/fixtures/NeuralAmpModelerCore
fi

# 3. Soak Tests (Numerical Stability)
echo -e "\n${BLUE}${BOLD}[Phase 3/7] Executando testes de estabilidade numérica (Soak)...${NC}"
cargo test --release --features standalone --test soak_test -- --ignored --nocapture --test-threads=1 2>&1 | tee target/logs/soak-test.log
cargo test --release --features standalone --test pipeline_soak -- --ignored --nocapture --test-threads=1 2>&1 | tee target/logs/pipeline-soak.log

# 4. Property-Based and Parity Tests in Release
echo -e "\n${BLUE}${BOLD}[Phase 4/7] Executando testes baseados em propriedades (Proptests)...${NC}"
cargo test --release --test proptest_parsers -- --ignored 2>&1 | tee target/logs/proptest-parsers.log
cargo test --release --test proptest_math -- --ignored 2>&1 | tee target/logs/proptest-math.log
cargo test --release --test lstm_gate_bf16_parity -- --ignored 2>&1 | tee target/logs/lstm-gate-bf16-parity.log
cargo test --release --test lstm_scalar_bf16_parity -- --ignored 2>&1 | tee target/logs/lstm-scalar-bf16-parity.log
cargo test --release --lib -- dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest --ignored 2>&1 | tee target/logs/pipeline-block-proptest.log
cargo test --release --test gate_fsm_proptest -- --ignored 2>&1 | tee target/logs/gate-fsm-proptest.log

# 5. Resampler Heap-Audit and C++ Parity
echo -e "\n${BLUE}${BOLD}[Phase 5/7] Executando auditoria do resampler e paridade C++...${NC}"
cargo test --release --features heap-audit --test resampler_heap_audit 2>&1 | tee target/logs/resampler-heap-audit.log
cargo test --release --test cpp_parity -- --ignored --nocapture 2>&1 | tee target/logs/cpp-parity.log

# 6. CLAP Release Validation with Heap Alloc Audit
echo -e "\n${BLUE}${BOLD}[Phase 6/7] Validando conformidade CLAP em modo Release...${NC}"
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --release --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

RELEASE_CLAP_BIN="target/clap-test/release/libnam_rs.so"
if [ ! -f "$RELEASE_CLAP_BIN" ]; then
    echo -e "${RED}Erro: Falha ao encontrar biblioteca CLAP Release em $RELEASE_CLAP_BIN!${NC}"
    exit 1
fi

# Audit release binary properties
echo -e "  Auditando propriedades do binário de Release..."
if readelf -d "$RELEASE_CLAP_BIN" | grep -q SONAME; then
    echo -e "    ${GREEN}✓${NC} SONAME presente."
else
    echo -e "${RED}    ❌ Erro: SONAME ausente!${NC}"
    exit 1
fi

if nm -D "$RELEASE_CLAP_BIN" | grep -q "clap_entry"; then
    echo -e "    ${GREEN}✓${NC} Símbolo 'clap_entry' presente."
else
    echo -e "${RED}    ❌ Erro: Símbolo 'clap_entry' ausente!${NC}"
    exit 1
fi

# Run clap-validator on release binary
if command -v clap-validator >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
    NAM_HEAP_AUDIT=1 \
      clap-validator validate "$RELEASE_CLAP_BIN" --json 2>target/logs/release-validation.stderr.log | tee target/logs/release-validation.json
    jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' target/logs/release-validation.json >/dev/null
    echo -e "    ${GREEN}✓${NC} Validação clap-validator sem avisos/falhas."
else
    echo -e "${YELLOW}    Aviso: clap-validator ou jq não instalados. Pulando validação estrita.${NC}"
fi

# Multi-instance stress tests
echo -e "  Executando teste de concorrência com instâncias múltiplas..."
cargo test --no-default-features --features "clap-plugin" --test clap_multi_instance -- --ignored --nocapture 2>&1 | tee target/logs/clap-multi-instance.log

# 7. Long Benchmarks (Performance)
echo -e "\n${BLUE}${BOLD}[Phase 7/7] Executando benchmarks de performance longos...${NC}"
cargo bench
cargo bench --features "standalone,long_bench" --bench inference_bench 2>&1 | tee target/logs/long-bench.log

echo -e "\n${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}          Auditoria de Longa Duração Concluída com Sucesso!     ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "  Logs persistidos em: ${YELLOW}target/logs/${NC}"
