#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Supply chain update utility for nam-rs.
# Updates the Rust toolchain, Cargo package indexes, dependencies in Cargo.toml/Cargo.lock,
# and pulls the latest upstream NeuralAmpModelerCore and NeuralAmpModelerPlugin fixtures.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PHASE_TOTAL=5
source "$SCRIPT_DIR/_lib.sh"

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}          nam-rs Supply Chain Update & Sync Pipeline            ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# Ensure we are in the project root directory
cd "$PROJECT_DIR"

# Load pinned versions from single source of truth (variables.env).
source "$PROJECT_DIR/variables.env"

# 1. Update Rust Toolchain
phase "Atualizando a toolchain ativa do Rust (rustup)..."
if command -v rustup &>/dev/null; then
    rustup update
else
    echo -e "${YELLOW}Aviso: rustup não encontrado. Pulando atualização da toolchain.${NC}"
fi

# 2. Upgrade dependencies in Cargo.toml
phase "Atualizando definições de dependências (Cargo.toml)..."
if cargo --list | grep -q "upgrade"; then
    cargo upgrade --verbose
else
    echo -e "${YELLOW}Aviso: cargo-edit (cargo-upgrade) não encontrado.${NC}"
    echo -e "${YELLOW}Instale com: cargo install cargo-edit${NC}"
fi

# 3. Update Cargo.lock
phase "Atualizando versões resolvidas no Cargo.lock..."
cargo update --verbose

# 4. Sync upstream C++ fixtures
phase "Sincronizando fixtures do NeuralAmpModelerCore..."
FIXTURE_DIR="tests/fixtures/NeuralAmpModelerCore"

if [ -d "$FIXTURE_DIR" ]; then
    echo -e "  Fixtures encontradas em $FIXTURE_DIR. Atualizando..."
    (cd "$FIXTURE_DIR" && git fetch --depth 1 origin tag "$NAM_CORE_TAG" && git checkout "$NAM_CORE_COMMIT" && git clean -df)
    echo -e "  ${GREEN}✓${NC} Fixtures sincronizadas (canonical: $NAM_CORE_TAG @ $NAM_CORE_COMMIT)."
else
    echo -e "  Fixtures não encontradas. Clonando pela primeira vez..."
    git clone --depth 1 --branch "$NAM_CORE_TAG" "$NAM_CORE_REPO" "$FIXTURE_DIR"
    (cd "$FIXTURE_DIR" && git checkout "$NAM_CORE_COMMIT")
    echo -e "  ${GREEN}✓${NC} Fixtures clonadas com sucesso."
fi

# Initialize submodules for NeuralAmpModelerCore
for sub in eigen AudioDSPTools; do
    sub_path="$FIXTURE_DIR/Dependencies/$sub"
    if [ ! -d "$sub_path" ] || [ -z "$(ls -A "$sub_path" 2>/dev/null)" ]; then
        echo "  Initializing submodule $sub for NeuralAmpModelerCore..."
        (cd "$FIXTURE_DIR" && git submodule update --init "Dependencies/$sub")
    fi
done

# 5. Sync upstream NeuralAmpModelerPlugin (C++ IR reference)
phase "Sincronizando fixtures do NeuralAmpModelerPlugin..."
PLUGIN_DIR="tests/fixtures/NeuralAmpModelerPlugin"

if [ -d "$PLUGIN_DIR" ]; then
    echo -e "  Fixtures encontradas em $PLUGIN_DIR. Atualizando..."
    (cd "$PLUGIN_DIR" && git fetch --depth 1 origin tag "$NAM_PLUGIN_TAG" && git checkout "$NAM_PLUGIN_COMMIT" && git clean -df)
    echo -e "  ${GREEN}✓${NC} Fixtures sincronizadas (canonical: $NAM_PLUGIN_TAG @ $NAM_PLUGIN_COMMIT)."
else
    echo -e "  Fixtures não encontradas. Clonando pela primeira vez..."
    git clone --depth 1 --branch "$NAM_PLUGIN_TAG" "$NAM_PLUGIN_REPO" "$PLUGIN_DIR"
    (cd "$PLUGIN_DIR" && git checkout "$NAM_PLUGIN_COMMIT")
    echo -e "  ${GREEN}✓${NC} Fixtures clonadas com sucesso."
fi

# Initialize submodules for NeuralAmpModelerPlugin (AudioDSPTools → eigen, nlohmann)
if [ "$(cd "$PLUGIN_DIR" && git submodule status AudioDSPTools | head -c1)" = "-" ]; then
    echo -e "  Initializing submodules for NeuralAmpModelerPlugin..."
    (cd "$PLUGIN_DIR" && git submodule update --init --recursive AudioDSPTools)
    echo -e "  ${GREEN}✓${NC} Submodules initialized."
fi

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}          Toda a cadeia de suprimentos foi atualizada!          ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
