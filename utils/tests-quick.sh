#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and testing script for nam-rs — unified PR gate.
#
# Phases:
#   1. Unit/integration tests (fast feedback, debug mode)
#   2. Medium validation suite (C++ parity + proptest parsers + proptest math, release mode, ignored)
#   3. Build CLAP plugin (debug + heap-audit)
#   4. CLAP integration and heap-audit tests
#   5. CLAP validator (external)
#
# Phase 2 requires NeuralAmpModelerCore (./utils/mod-update.sh) and golden vectors.
# It is gracefully skipped if not available, emitting a warning.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

trap 'echo -e "\n${RED}${BOLD}❌ Erro inesperado: Comando \"$BASH_COMMAND\" falhou na linha $LINENO com status $?. Abortando suíte de testes.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}=================================================${NC}"
echo -e "${BLUE}${BOLD}      nam-rs Quick QA Suite (± 2,5 minutes)      ${NC}"
echo -e "${BLUE}${BOLD}=================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. Standard tests (fast feedback)
echo -e "\n${BLUE}${BOLD}[1/5] Executando testes unitários e de integração...${NC}"
cargo test

# 2. Medium validation suite — C++ Parity + Proptests (release, ignored)
#    These provide robust parser/SIMD/parity guarantees for the PR gate without
#    waiting for the full long-duration suite.  Gracefully skipped when golden
#    vectors or NeuralAmpModelerCore are not available.
echo -e "\n${BLUE}${BOLD}[2/5] Executando suíte de validação intermediária...${NC}"
GOLDENS_OK=0
if [ -d "tests/fixtures/NeuralAmpModelerCore" ] && [ -f "tests/fixtures/golden_cabsim_cpp_short.bin" ]; then
    GOLDENS_OK=1
fi

if [ "$GOLDENS_OK" -eq 1 ]; then
    MEDIUM_STATUS=0

    echo -e "  ${BLUE}→ C++ Parity (todos os modelos ignorados)...${NC}"
    cargo test --release --test cpp_parity -- --ignored --nocapture || MEDIUM_STATUS=1

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

# 3. Build CLAP plugin debug binary with heap-audit
echo -e "\n${BLUE}${BOLD}[3/5] Compilando plugin CLAP (Debug + heap-audit)...${NC}"
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
HASH_PHASE3=$(sha256sum "$CLAP_BIN" | cut -d' ' -f1)
echo -e "  Preservado binário da fase 3: $CLAP_BIN"
echo -e "  SHA256 do binário compilado: ${GREEN}${HASH_PHASE3}${NC}"

# 4. CLAP integration and heap-audit tests
echo -e "\n${BLUE}${BOLD}[4/5] Executando testes de integração CLAP e auditoria de heap...${NC}"

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


# 5. Run the official CLAP validator if available
echo -e "\n${BLUE}${BOLD}[5/5] Executando validação via clap-validator...${NC}"
if command -v clap-validator >/dev/null 2>&1; then
  # Validar que o binário a ser testado pelo clap-validator é rigorosamente o da fase 3
  HASH_PHASE5=$(sha256sum "$CLAP_BIN" | cut -d' ' -f1)
  echo -e "  SHA256 do binário na fase 3: ${GREEN}${HASH_PHASE3}${NC}"
  echo -e "  SHA256 do binário na fase 5: ${GREEN}${HASH_PHASE5}${NC}"
  if [ "$HASH_PHASE3" != "$HASH_PHASE5" ]; then
    echo -e "${RED}Erro: O checksum do binário mudou entre as fases 3 e 5!${NC}"
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
