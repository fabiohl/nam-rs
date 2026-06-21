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
echo -e "${BLUE}${BOLD}         nam-rs Standard QA & Test Suite (± 90 seconds)        ${NC}"
echo -e "${BLUE}${BOLD}===============================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. Standard tests
echo -e "\n${BLUE}${BOLD}[1/4] Executando testes unitários e de integração...${NC}"
cargo test

# 2. Build CLAP plugin debug binary with heap-audit
echo -e "\n${BLUE}${BOLD}[2/4] Compilando plugin CLAP (Debug + heap-audit)...${NC}"
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --profile test --no-default-features --features "clap-plugin,heap-audit,testing" --lib

CLAP_BIN_RAW="target/debug/libnam_rs.so"
if [ ! -f "$CLAP_BIN_RAW" ]; then
    echo -e "${RED}Erro: Falha ao encontrar a biblioteca do CLAP em $CLAP_BIN_RAW!${NC}"
    exit 1
fi

# Preservar o binário compilado em um local estável para evitar modificações por etapas subsequentes
CLAP_BIN="target/debug/libnam_rs_validated.so"
cp "$CLAP_BIN_RAW" "$CLAP_BIN"
HASH_PHASE2=$(sha256sum "$CLAP_BIN" | cut -d' ' -f1)
echo -e "  Preservado binário da fase 2: $CLAP_BIN"
echo -e "  SHA256 do binário compilado: ${GREEN}${HASH_PHASE2}${NC}"

# 3. CLAP integration and heap-audit tests
echo -e "\n${BLUE}${BOLD}[3/4] Executando testes de integração CLAP e auditoria de heap...${NC}"

# A) CLAP Library tests
CLAP_PLUGIN_PATH="$CLAP_BIN" NAM_HEAP_AUDIT=1 \
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo test --profile test --no-default-features --features "clap-plugin,heap-audit,testing" --lib clap::

# B) Targeted integration tests
CLAP_PLUGIN_PATH="$CLAP_BIN" NAM_HEAP_AUDIT=1 \
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo test --profile test --no-default-features --features "clap-plugin,heap-audit,testing" \
  --test a2_heap_audit \
  --test cabsim_heap_audit \
  --test resampler_heap_audit \
  --test clap_lifecycle_test \
  --test clap_state_migration \
  --test clap_multi_instance

# C) Diagnostic bundle heap variant test
CLAP_PLUGIN_PATH="$CLAP_BIN" NAM_HEAP_AUDIT=1 \
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo test --profile test --no-default-features --features "clap-plugin,heap-audit,testing" \
  --test diagnostic_bundle heap_audit


# 4. Run the official CLAP validator if available
echo -e "\n${BLUE}${BOLD}[4/4] Executando validação via clap-validator...${NC}"
if command -v clap-validator >/dev/null 2>&1; then
  # Validar que o binário a ser testado pelo clap-validator é rigorosamente o da fase 2
  HASH_PHASE4=$(sha256sum "$CLAP_BIN" | cut -d' ' -f1)
  echo -e "  SHA256 do binário na fase 2: ${GREEN}${HASH_PHASE2}${NC}"
  echo -e "  SHA256 do binário na fase 4: ${GREEN}${HASH_PHASE4}${NC}"
  if [ "$HASH_PHASE2" != "$HASH_PHASE4" ]; then
    echo -e "${RED}Erro: O checksum do binário mudou entre as fases 2 e 4!${NC}"
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

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}               Todos os testes padrão passaram!                 ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
