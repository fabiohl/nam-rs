#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Supply chain update utility for nam-rs.
# Updates the Rust toolchain, Cargo package indexes, dependencies in Cargo.toml/Cargo.lock,
# and pulls the latest upstream NeuralAmpModelerCore fixtures.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}          nam-rs Supply Chain Update & Sync Pipeline            ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 1. Update Rust Toolchain
echo -e "\n${BLUE}${BOLD}[1/4] Atualizando a toolchain ativa do Rust (rustup)...${NC}"
if command -v rustup &>/dev/null; then
    rustup update
else
    echo -e "${YELLOW}Aviso: rustup não encontrado. Pulando atualização da toolchain.${NC}"
fi

# 2. Upgrade dependencies in Cargo.toml
echo -e "\n${BLUE}${BOLD}[2/4] Atualizando definições de dependências (Cargo.toml)...${NC}"
if cargo --list | grep -q "upgrade"; then
    cargo upgrade --verbose
else
    echo -e "${YELLOW}Aviso: cargo-edit (cargo-upgrade) não encontrado.${NC}"
    echo -e "${YELLOW}Instale com: cargo install cargo-edit${NC}"
fi

# 3. Update Cargo.lock
echo -e "\n${BLUE}${BOLD}[3/4] Atualizando versões resolvidas no Cargo.lock...${NC}"
cargo update --verbose

# 4. Sync upstream C++ fixtures
echo -e "\n${BLUE}${BOLD}[4.1/4] Sincronizando fixtures do NeuralAmpModelerCore...${NC}"
FIXTURE_DIR="tests/fixtures/NeuralAmpModelerCore"

# Canonical tag and pinned SHA (T5.2 — single reference for all goldens)
NAM_CORE_TAG="v0.5.3"
NAM_CORE_SHA="9c7b185de346fe0725dea537bcee4bc38b5bb6d6"

if [ -d "$FIXTURE_DIR" ]; then
    echo -e "  Fixtures encontradas em $FIXTURE_DIR. Atualizando..."
    (cd "$FIXTURE_DIR" && git fetch --depth 1 origin tag "$NAM_CORE_TAG" && git checkout "$NAM_CORE_SHA" && git clean -df)
    echo -e "  ${GREEN}✓${NC} Fixtures sincronizadas (canonical: $NAM_CORE_TAG @ $NAM_CORE_SHA)."
else
    echo -e "  Fixtures não encontradas. Clonando pela primeira vez..."
    git clone --depth 1 --branch "$NAM_CORE_TAG" https://github.com/sdatkinson/NeuralAmpModelerCore.git "$FIXTURE_DIR"
    echo -e "  ${GREEN}✓${NC} Fixtures clonadas com sucesso."
fi
echo -e "\n${BLUE}${BOLD}[4.2/4] Sincronizando fixtures do NeuralAmpModelerPlugin...${NC}"
FIXTURE_DIR="tests/fixtures/NeuralAmpModelerPlugin"

# Canonical tag and pinned SHA for plugin
NAM_PLUGIN_TAG="v0.7.15"
NAM_PLUGIN_SHA="96337e9ab6e3beb619459779bbb5c47e1b04d8c4"

if [ -d "$FIXTURE_DIR" ]; then
    echo -e "  Fixtures encontradas em $FIXTURE_DIR. Atualizando..."
    (cd "$FIXTURE_DIR" && git fetch --depth 1 origin tag "$NAM_PLUGIN_TAG" && git checkout "$NAM_PLUGIN_SHA" && git clean -df)
    echo -e "  ${GREEN}✓${NC} Fixtures sincronizadas (canonical: $NAM_PLUGIN_TAG @ $NAM_PLUGIN_SHA)."
else
    echo -e "  Fixtures não encontradas. Clonando pela primeira vez..."
    git clone --depth 1 --branch "$NAM_PLUGIN_TAG" https://github.com/sdatkinson/NeuralAmpModelerPlugin.git "$FIXTURE_DIR"
    echo -e "  ${GREEN}✓${NC} Fixtures clonadas com sucesso."
fi


echo -e "\n${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}          Toda a cadeia de suprimentos foi atualizada!          ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
