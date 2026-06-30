#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and testing script for nam-rs — unified PR gate.
#
# Phases:
#   1. Cargo clippy strict (standalone + CLAP plugin features)
#   2. Unit/integration tests (fast feedback, debug mode)
#   3. Medium validation suite (C++ parity + proptest parsers + proptest math, release mode, ignored)
#   4. Build CLAP plugin (debug + heap-audit)
#   5. CLAP integration and heap-audit tests
#   6. CLAP validator (external)
#
# Phase 3 requires NeuralAmpModelerCore (./utils/mod-update.sh) and golden vectors.
# It is gracefully skipped if not available, emitting a warning.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Re-execute with low CPU and I/O priority (nice and ionice) to prevent overloading the system.
# This can be bypassed by setting NAM_NO_LOW_PRIORITY=1.
if [ "${NAM_LOW_PRIORITY:-0}" != "1" ] && [ "${NAM_NO_LOW_PRIORITY:-0}" != "1" ]; then
    export NAM_LOW_PRIORITY=1
    CMD_PREFIX=""
    if command -v nice >/dev/null 2>&1; then
        CMD_PREFIX="nice -n 19"
    fi
    if command -v ionice >/dev/null 2>&1; then
        CMD_PREFIX="$CMD_PREFIX ionice -c 3"
    fi
    if [ -n "$CMD_PREFIX" ]; then
        echo -e "${YELLOW}ⓘ Reiniciando o script com baixa prioridade (CPU/IO) para evitar travamentos...${NC}"
        exec $CMD_PREFIX "$0" "$@"
    fi
fi

trap 'echo -e "\n${RED}${BOLD}❌ Erro inesperado: Comando \"$BASH_COMMAND\" falhou na linha $LINENO com status $?. Abortando suíte de testes.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}=================================================${NC}"
echo -e "${BLUE}${BOLD}      nam-rs Quick QA Suite (± 5,0 minutes)      ${NC}"
echo -e "${BLUE}${BOLD}=================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CLAP_BIN_RAW="target/debug/libnam_rs.so"
CLAP_BIN="target/debug/libnam_rs_validated.so"

# Helper to compute SHA256 of a file
get_sha256() {
    sha256sum "$1" | cut -d' ' -f1
}

# Helper to run cargo commands with the CLAP profile, features, and environment variables
cargo_clap() {
    local action="$1"
    shift
    CLAP_PLUGIN_PATH="$CLAP_BIN" \
    NAM_HEAP_AUDIT=1 \
    RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
      cargo "$action" --profile test --no-default-features --features "clap-plugin,heap-audit,testing" "$@"
}

# 1. Clippy strict (standalone + CLAP plugin features — focused PR gate coverage)
echo -e "\n${BLUE}${BOLD}[1/6] Executando análise estática estrita (cargo clippy)...${NC}"
echo -e "  Clippy: Standalone..."
cargo clippy --all-targets --features standalone -- -D warnings
echo -e "  Clippy: CLAP Plugin..."
cargo clippy --all-targets --no-default-features --features clap-plugin,testing -- -D warnings

# Track whether medium validation ran (for summary message)
MEDIUM_RUN=false

# 2. Standard tests (fast feedback)
echo -e "\n${BLUE}${BOLD}[2/6] Executando testes unitários e de integração...${NC}"
cargo test

# 3. Medium validation suite — C++ Parity + Proptests (release, ignored)
#    These provide robust parser/SIMD/parity guarantees for the PR gate without
#    waiting for the full long-duration suite.  Gracefully skipped when golden
#    vectors or NeuralAmpModelerCore are not available.
echo -e "\n${BLUE}${BOLD}[3/6] Executando suíte de validação intermediária...${NC}"
GOLDENS_OK=0
if [ -d "tests/fixtures/NeuralAmpModelerCore" ] && [ -f "tests/fixtures/golden_cabsim_cpp_short.bin" ]; then
    GOLDENS_OK=1
fi

if [ "$GOLDENS_OK" -eq 1 ]; then
    MEDIUM_RUN=true
    MEDIUM_STATUS=0

    echo -e "  ${BLUE}→ C++ Parity (subconjunto rápido: LSTM + WaveNet CH16 + A2 @ 48 kHz)...${NC}"
    cargo test --release --test cpp_parity -- quick_parity --nocapture || MEDIUM_STATUS=1

    echo -e "  ${BLUE}→ Proptest Parsers (fuzzing de loaders)...${NC}"
    cargo test --release --test proptest_parsers -- --ignored --nocapture || MEDIUM_STATUS=1

    echo -e "  ${BLUE}→ Proptest Math (precisão SIMD)...${NC}"
    cargo test --release --test proptest_math -- --ignored --nocapture || MEDIUM_STATUS=1

    if [ "$MEDIUM_STATUS" -ne 0 ]; then
        echo -e "${RED}${BOLD}❌ Suíte de validação intermediária falhou.${NC}"
        exit 1
    fi
else
    echo -e "  ${YELLOW}ⓘ NeuralAmpModelerCore ou golden vectors não encontrados.${NC}"
    echo -e "  ${YELLOW}  Execute './utils/mod-update.sh' para configurar as dependências.${NC}"
    echo -e "  ${YELLOW}  Pulando validação intermediária (cpp_parity + proptests).${NC}"
fi

# 4. Build CLAP plugin debug binary with heap-audit
echo -e "\n${BLUE}${BOLD}[4/6] Compilando plugin CLAP (Debug + heap-audit)...${NC}"
cargo_clap build --lib

if [ ! -f "$CLAP_BIN_RAW" ]; then
    echo -e "${RED}Erro: Falha ao encontrar a biblioteca do CLAP em $CLAP_BIN_RAW!${NC}"
    exit 1
fi

# Preservar o binário compilado em um local estável para evitar modificações por etapas subsequentes
cp "$CLAP_BIN_RAW" "$CLAP_BIN"
HASH_PHASE4=$(get_sha256 "$CLAP_BIN")
echo -e "  Preservado binário da fase 4: $CLAP_BIN"
echo -e "  SHA256 do binário compilado: ${GREEN}${HASH_PHASE4}${NC}"

# 5. CLAP integration and heap-audit tests
echo -e "\n${BLUE}${BOLD}[5/6] Executando testes de integração CLAP e auditoria de heap...${NC}"

# A) CLAP Library tests
cargo_clap test --lib clap::

# B) Targeted integration tests
cargo_clap test \
  --test a2_heap_audit \
  --test cabsim_heap_audit \
  --test resampler_heap_audit \
  --test clap_lifecycle_test \
  --test clap_state_migration \
  --test clap_multi_instance

# C) Diagnostic bundle heap variant test
cargo_clap test --test diagnostic_bundle heap_audit

# 6. Run the official CLAP validator if available
echo -e "\n${BLUE}${BOLD}[6/6] Executando validação via clap-validator...${NC}"
if command -v clap-validator >/dev/null 2>&1; then
  # Validar que o binário a ser testado pelo clap-validator é rigorosamente o da fase 4
  HASH_PHASE6=$(get_sha256 "$CLAP_BIN")
  echo -e "  SHA256 do binário na fase 4: ${GREEN}${HASH_PHASE4}${NC}"
  echo -e "  SHA256 do binário na fase 6: ${GREEN}${HASH_PHASE6}${NC}"
  if [ "$HASH_PHASE4" != "$HASH_PHASE6" ]; then
    echo -e "${RED}Erro: O checksum do binário mudou entre as fases 4 e 6!${NC}"
    exit 1
  else
    echo -e "  ${GREEN}✓${NC} Checksum correspondente comprovado."
  fi

  CLAP_PLUGIN_PATH="$CLAP_BIN" \
    NAM_HEAP_AUDIT=1 \
    clap-validator validate "$CLAP_BIN"
  echo -e "  ${GREEN}✓${NC} Validação com clap-validator finalizada."
else
  echo -e "${YELLOW}Aviso: clap-validator não encontrado. Pulando etapa de validação.${NC}"
fi

if [ "$MEDIUM_RUN" = true ]; then
    echo -e "${GREEN}${BOLD}================================================================${NC}"
    echo -e "${GREEN}${BOLD}               Todos os testes padrão passaram!                 ${NC}"
    echo -e "${GREEN}${BOLD}================================================================${NC}"
else
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}         Testes padrão passaram (validação intermediária        ${NC}"
    echo -e "${YELLOW}${BOLD}          foi pulada — gere os golden vectors primeiro)         ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
fi
