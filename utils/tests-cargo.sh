#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and testing script for nam-rs.
# Performs unit/integration tests, builds the CLAP plugin in debug/heap-audit mode,
# executes strict dynamic library validation audits, runs the CLAP/heap-audit test suite, and triggers clap-validator.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}===============================================================${NC}"
echo -e "${BLUE}${BOLD}        nam-rs Standard QA & Test Suite (± 1,5 minutos)        ${NC}"
echo -e "${BLUE}${BOLD}===============================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. Standard tests
echo -e "\n${BLUE}${BOLD}[1/4] Executando testes unitários e de integração...${NC}"
cargo test -- --test-threads=1

# 2. Build CLAP plugin debug binary with heap-audit
echo -e "\n${BLUE}${BOLD}[2/4] Compilando plugin CLAP (Debug + heap-audit)...${NC}"
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

CLAP_BIN="target/clap-test/debug/libnam_rs.so"
if [ ! -f "$CLAP_BIN" ]; then
    echo -e "${RED}Erro: Falha ao encontrar a biblioteca do CLAP em $CLAP_BIN!${NC}"
    exit 1
fi

# 3. CLAP integration and heap-audit tests
echo -e "\n${BLUE}${BOLD}[3/4] Executando testes de integração CLAP e auditoria de heap...${NC}"
CLAP_PLUGIN_PATH="$CLAP_BIN" \
  NAM_HEAP_AUDIT=1 \
  cargo test --features "clap-plugin,heap-audit" --target-dir target/clap-test -- --test-threads=1

# 4. Run the official CLAP validator if available
echo -e "\n${BLUE}${BOLD}[4/4] Executando validação via clap-validator...${NC}"
if command -v clap-validator >/dev/null 2>&1; then
  CLAP_PLUGIN_PATH="$CLAP_BIN" \
    NAM_HEAP_AUDIT=1 \
    clap-validator validate "$CLAP_BIN"
  echo -e "  ${GREEN}✓${NC} Validação com clap-validator finalizada."
else
  echo -e "${YELLOW}Aviso: clap-validator não encontrado. Pulando etapa de validação.${NC}"
fi

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}               Todos os testes padrão passaram!                 ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
