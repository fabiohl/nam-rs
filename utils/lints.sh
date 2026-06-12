#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and static analysis script for nam-rs.
# Formats code and performs compilation and Clippy checks across all feature configurations.

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

# 1. Format Check
echo -e "\n${BLUE}${BOLD}[1/3] Verificando e aplicando formatação de código...${NC}"
cargo fmt --all

# 2. Cargo Checks
echo -e "\n${BLUE}${BOLD}[2/3] Executando verificações de compilação (cargo check)...${NC}"

echo -e "  Checking: Standalone..."
cargo check --features standalone

echo -e "  Checking: Pure Core (No features)..."
cargo check --no-default-features

echo -e "  Checking: CLAP Plugin..."
cargo check --no-default-features --features clap-plugin

echo -e "  Checking: All Features..."
cargo check --all-features --all-targets

# 3. Cargo Clippy
echo -e "\n${BLUE}${BOLD}[3/3] Executando análise estática estrita (cargo clippy)...${NC}"

echo -e "  Clippy: Standalone..."
cargo clippy --all-targets --features standalone -- -D warnings

echo -e "  Clippy: Pure Core..."
cargo clippy --lib --no-default-features -- -D warnings

echo -e "  Clippy: CLAP Plugin..."
cargo clippy --all-targets --no-default-features --features clap-plugin,testing -- -D warnings

echo -e "  Clippy: All Features..."
cargo clippy --all-targets --all-features -- -D warnings

echo -e "\n${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}             Suíte de qualidade concluída com sucesso!           ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
