#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and static analysis script for nam-rs.
# Validates SPDX license headers, formats code, and performs compilation
# and Clippy checks across all feature configurations.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                 nam-rs Linting & Quality Suite                 ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. SPDX License Header Check
echo -e "\n${BLUE}${BOLD}[1/5] Verificando cabeçalhos de licença SPDX em src/...${NC}"

spdx_failed=0
while IFS= read -r -d '' file; do
    first_line=$(head -1 "$file")
    if ! echo "$first_line" | grep -q "SPDX-License-Identifier"; then
        echo -e "${RED}ERRO: Cabeçalho SPDX ausente:${NC} $file"
        spdx_failed=1
    elif ! echo "$first_line" | grep -qE "Apache-2\.0|MIT"; then
        echo -e "${RED}ERRO: Identificador SPDX inválido (esperado Apache-2.0 ou MIT):${NC} $file -> $first_line"
        spdx_failed=1
    fi
done < <(find src/ -name '*.rs' -print0)

if [ "$spdx_failed" -eq 1 ]; then
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Todos os arquivos .rs em src/ com cabeçalho SPDX válido (Apache-2.0, MIT)."

# 2. Format Check
echo -e "\n${BLUE}${BOLD}[2/5] Verificando e aplicando formatação de código...${NC}"
cargo fmt --all

# 3. Cargo Checks
echo -e "\n${BLUE}${BOLD}[3/5] Executando verificações de compilação (cargo check)...${NC}"

echo -e "  Checking: Standalone..."
cargo check --features standalone

echo -e "  Checking: Pure Core (No features)..."
cargo check --no-default-features

echo -e "  Checking: CLAP Plugin..."
cargo check --no-default-features --features clap-plugin

echo -e "  Checking: All Features..."
cargo check --all-features --all-targets

# 4. Cargo Clippy
echo -e "\n${BLUE}${BOLD}[4/5] Executando análise estática estrita (cargo clippy)...${NC}"

echo -e "  Clippy: Standalone..."
cargo clippy --all-targets --features standalone -- -D warnings

echo -e "  Clippy: Pure Core..."
cargo clippy --lib --no-default-features -- -D warnings

echo -e "  Clippy: CLAP Plugin..."
cargo clippy --all-targets --no-default-features --features clap-plugin,testing -- -D warnings

echo -e "  Clippy: All Features..."
cargo clippy --all-targets --all-features -- -D warnings

# 5. Anti-pattern Check
echo -e "\n${BLUE}${BOLD}[5/5] Verificando anti-padrão de #[test] em tests/common/...${NC}"
if grep -rnF "#[test]" tests/common/ >/dev/null 2>&1; then
  echo -e "${RED}${BOLD}ERRO: Encontrado '#[test]' no diretório tests/common/!${NC}"
  echo -e "Testes não devem ser colocados no módulo compartilhado 'tests/common/' para evitar execuções redundantes."
  echo -e "Ocorrências encontradas:${NC}"
  grep -rnF "#[test]" tests/common/
  exit 1
fi
echo -e "  ${GREEN}✓${NC} Nenhum '#[test]' encontrado em tests/common/."

echo -e "\n${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}             Suíte de qualidade concluída com sucesso!           ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
