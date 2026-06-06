#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and testing script for nam-rs.
# Performs unit/integration tests, builds the CLAP plugin in debug/heap-audit mode,
# executes strict dynamic library validation audits, runs lifecycle tests, and triggers clap-validator.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}               nam-rs Standard QA & Test Suite                  ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. Standard tests
echo -e "\n${BLUE}${BOLD}[1/5] Executando testes unitários e de integração...${NC}"
cargo test

# 2. Build CLAP plugin debug binary with heap-audit
echo -e "\n${BLUE}${BOLD}[2/5] Compilando plugin CLAP (Debug + heap-audit)...${NC}"
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

CLAP_BIN="target/clap-test/debug/libnam_rs.so"
if [ ! -f "$CLAP_BIN" ]; then
    echo -e "${RED}Erro: Falha ao encontrar a biblioteca do CLAP em $CLAP_BIN!${NC}"
    exit 1
fi

# 3. Audit binary validity
echo -e "\n${BLUE}${BOLD}[3/5] Auditando propriedades e formato do binário CLAP...${NC}"

# 3.1. SONAME check
if readelf -d "$CLAP_BIN" | grep -q SONAME; then
    echo -e "  ${GREEN}✓${NC} SONAME encontrado no binário."
else
    echo -e "${RED}❌ Erro: SONAME ausente no binário!${NC}"
    exit 1
fi

# 3.2. CLAP entry symbol check
if nm -D "$CLAP_BIN" | grep -q "clap_entry"; then
    echo -e "  ${GREEN}✓${NC} Símbolo 'clap_entry' exportado com sucesso."
else
    echo -e "${RED}❌ Erro: Símbolo 'clap_entry' ausente! O plugin não será carregado.${NC}"
    exit 1
fi

# 3.3. ELF 64-bit file type check
FILE_INFO=$(file "$CLAP_BIN")
if [[ $FILE_INFO == *"ELF 64-bit"* ]] && [[ $FILE_INFO == *"x86-64"* ]]; then
    echo -e "  ${GREEN}✓${NC} Formato ELF 64-bit x86-64 confirmado."
else
    echo -e "${RED}❌ Erro: Formato de arquivo ELF inválido: $FILE_INFO${NC}"
    exit 1
fi

# 4. Lifecycle tests targeting the built binary
echo -e "\n${BLUE}${BOLD}[4/5] Executando testes de ciclo de vida com auditoria de heap...${NC}"
CLAP_PLUGIN_PATH="$CLAP_BIN" \
  NAM_HEAP_AUDIT=1 \
  cargo test --test clap_lifecycle_test --features "clap-plugin" --target-dir target/clap-test

# 5. Run the official CLAP validator if available
echo -e "\n${BLUE}${BOLD}[5/5] Executando validação via clap-validator...${NC}"
if command -v clap-validator >/dev/null 2>&1; then
  CLAP_PLUGIN_PATH="$CLAP_BIN" \
    NAM_HEAP_AUDIT=1 \
    clap-validator validate "$CLAP_BIN"
  echo -e "  ${GREEN}✓${NC} Validação com clap-validator finalizada."
else
  echo -e "${YELLOW}Aviso: clap-validator não encontrado. Pulando etapa de validação.${NC}"
fi

echo -e "\n${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}               Todos os testes padrão passaram!                 ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
