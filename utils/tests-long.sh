#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Intensive long-duration verification suite for nam-rs.
# Performs numerical soak tests, proptest fuzzing, NeuralAmpModelerCore parity checks,
# CLAP release compliance, multi-instance stress tests, and long-running performance benchmarks.

set -uo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Setup defensive error trap
trap 'echo -e "\n${RED}${BOLD}❌ Erro inesperado: Comando \"$BASH_COMMAND\" falhou na linha $LINENO com status $?. Abortando suíte de testes.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}==========================================================================${NC}"
echo -e "${BLUE}${BOLD}    nam-rs Long-Duration Stress & Audit Suite (± 46 minutes - cold run)   ${NC}"
echo -e "${BLUE}${BOLD}==========================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Setup target logs
rm -rf target/logs/
mkdir -p target/logs/

# Cleanup accumulated live-test artifacts from previous runs (41+ MB WAVs)
rm -rf tests/fixtures/.temp_live/

# Auto-clone NeuralAmpModelerCore if not present (helps in fresh clones)
if [ ! -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    echo "🌐 Clonando NeuralAmpModelerCore para testes de paridade..."
    git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git tests/fixtures/NeuralAmpModelerCore
fi

# Trackers for the final summary
declare -a PHASE_NAMES
declare -a PHASE_COMMANDS
declare -a PHASE_STATUS
declare -a PHASE_DURATIONS
PHASE_COUNT=0

run_phase() {
    local name="$1"
    local cmd="$2"
    local log_file="$3"

    echo -e "\n${BLUE}${BOLD}[Phase $((PHASE_COUNT+1))] $name...${NC}"
    echo -e "Executando: ${YELLOW}$cmd${NC}"
    echo -e "Log em: ${YELLOW}target/logs/$log_file${NC}"

    local start_time=$(date +%s)

    # Run command and capture output/status
    eval "$cmd" > "target/logs/$log_file" 2>&1
    local status=$?

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    PHASE_NAMES[$PHASE_COUNT]="$name"
    PHASE_COMMANDS[$PHASE_COUNT]="$cmd"
    PHASE_DURATIONS[$PHASE_COUNT]="$duration"

    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ Sucesso (${duration}s)${NC}"
        PHASE_STATUS[$PHASE_COUNT]="PASSED"
    else
        echo -e "${RED}❌ Falha (${duration}s) - Status: $status${NC}"
        PHASE_STATUS[$PHASE_COUNT]="FAILED"
    fi

    PHASE_COUNT=$((PHASE_COUNT + 1))
    return $status
}

# --- Phase 1: Soak/Stress tests (Numerical Stability) ---
run_phase \
    "Soak Tests (Numerical Stability)" \
    "cargo test --release --features standalone --test soak_test -- --ignored --nocapture --test-threads=1 && cargo test --release --features standalone --test pipeline_soak -- --ignored --nocapture --test-threads=1" \
    "phase1-soak.log"

# --- Phase 2: Property-Based and Parity Tests in Release ---
run_phase \
    "Property-Based & Parity Tests in Release" \
    "cargo test --release --test proptest_parsers -- --ignored && cargo test --release --test proptest_math -- --ignored && cargo test --release --test lstm_gate_bf16_parity -- --ignored && cargo test --release --test lstm_scalar_bf16_parity -- --ignored && cargo test --release --lib -- dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest --ignored && cargo test --release --test gate_fsm_proptest -- --ignored" \
    "phase2-proptests.log"

# --- Phase 3: Resampler Heap-Audit and C++ Parity ---
run_phase \
    "Resampler, Cabsim & A2 Heap-Audit, C++ Parity" \
    "cargo test --release --features heap-audit --test resampler_heap_audit && cargo test --release --features heap-audit --test cabsim_heap_audit && cargo test --release --features heap-audit --test a2_heap_audit && cargo test --release --test cpp_parity -- --ignored --nocapture && cargo test --release --test cabsim_cpp_parity -- --ignored --nocapture" \
    "phase3-parity-audit.log"

# --- Phase 4: CLAP Release Validation & Concurrency (Local helper function) ---
run_clap_audit_local() {
    echo "  Limpando diretório target do CLAP..."
    cargo clean --target-dir target/clap-test

    echo "  Compilando CLAP Plugin em modo Release..."
    RUSTFLAGS="-Clink-arg=-Wl,-soname,nam-rs.clap" \
      cargo build --release --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

    local RELEASE_CLAP_BIN="target/clap-test/release/libnam_rs.so"
    if [ ! -f "$RELEASE_CLAP_BIN" ]; then
        echo "Erro: libnam_rs.so de release não encontrado." >&2
        return 1
    fi

    echo "  Auditando SONAME e símbolos exportados..."
    if ! readelf -d "$RELEASE_CLAP_BIN" | grep -q SONAME; then
        echo "Erro: SONAME ausente no binário de Release!" >&2
        return 1
    fi
    if ! nm -D "$RELEASE_CLAP_BIN" | grep -q "clap_entry"; then
        echo "Erro: Símbolo 'clap_entry' ausente no binário de Release!" >&2
        return 1
    fi

    if command -v clap-validator >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
        echo "  Executando clap-validator estrito..."
        NAM_HEAP_AUDIT=1 clap-validator validate "$RELEASE_CLAP_BIN" --json > target/logs/release-validation.json 2> target/logs/release-validation.stderr
        if ! jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' target/logs/release-validation.json >/dev/null; then
            echo "Erro: Falha ou avisos detectados pelo clap-validator!" >&2
            return 1
        fi
    else
        echo "  Aviso: clap-validator ou jq indisponíveis. Pulando auditoria externa."
    fi

    echo "  Executando testes de concorrência com instâncias múltiplas..."
    cargo test --no-default-features --features "clap-plugin" --test clap_multi_instance -- --ignored --nocapture && \
      echo "  Executando teste de stress do GC com 1000 swaps..." && \
      cargo test --no-default-features --features "clap-plugin" --lib -- clap::processor::processor_test::tests::test_gc_stress_1000_swaps --include-ignored --nocapture && \
      echo "  Executando testes de concorrência dedicados (T8.12, sem --test-threads=1)..." && \
      cargo test --features standalone --test concurrency_stress -- --ignored --nocapture && \
      echo "  Executando testes unitários e de integração em modo Mono..." && \
      cargo test --no-default-features --features "clap-plugin,testing"
}

run_phase \
    "CLAP Release Validation & Concurrency" \
    "run_clap_audit_local" \
    "phase4-clap-validation.log"

# --- Phase 5: Long Benchmarks (Performance) ---
run_phase \
    "Long Performance Benchmarks" \
    "cargo bench && cargo bench --features standalone,long_bench --bench inference_bench" \
    "phase5-benchmarks.log"

# --- Phase 6: PipeWire Integration Test (optional – skipped when daemon is absent) ---
run_pipewire_phase() {
    echo "  Verificando daemon PipeWire..."
    if pw-cli info >/dev/null 2>&1; then
        echo "  PipeWire detectado. Executando teste de integração..."
        cargo test --release --features standalone --test pw_integration_test -- --ignored --nocapture
    else
        echo "  PipeWire indisponível (pw-cli info falhou). Pulando teste de integração."
        return 0
    fi
}

run_phase \
    "PipeWire Integration Test" \
    "run_pipewire_phase" \
    "phase6-pipewire.log"

# --- Print beautifully structured summary ---
echo -e "\n${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                  AUDIT SUMMARY REPORT                          ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"
printf " | %-45s | %-10s | %-10s |\n" "Phase Name" "Duration" "Status"
printf " |-%-45s-|-%-10s-|-%-10s-|\n" "---------------------------------------------" "----------" "----------"

ANY_FAILED=0
for ((i=0; i<PHASE_COUNT; i++)); do
    name="${PHASE_NAMES[$i]}"
    duration="${PHASE_DURATIONS[$i]}s"
    status="${PHASE_STATUS[$i]}"

    if [ "$status" = "PASSED" ]; then
        status_colored="${GREEN}${status}${NC}"
    else
        status_colored="${RED}${status}${NC}"
        ANY_FAILED=1
    fi
    printf " | %-45s | %-10s | %-19b |\n" "$name" "$duration" "$status_colored"
done
echo -e "${BLUE}${BOLD}================================================================${NC}"

if [ $ANY_FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Todos os estágios da auditoria passaram com sucesso!${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}❌ Algum estágio da auditoria falhou. Verifique os logs em target/logs/${NC}"
    exit 1
fi
