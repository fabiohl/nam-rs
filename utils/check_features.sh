#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

# Script para verificar a compilação condicional do NAM-rs com diferentes feature flags.
# Garante que o motor DSP permaneça desacoplado do host (PipeWire) e preparado para o plugin CLAP.

set -e

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "🚀 Iniciando verificações de compilação condicional..."

# 1. Standalone (PipeWire) - Default
echo -n "Checking: standalone (default)... "
if cargo check --features standalone > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

# 2. No Default Features (Pure Engine)
echo -n "Checking: --no-default-features (pure engine)... "
if cargo check --no-default-features > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

# 3. CLAP Plugin Staging
echo -n "Checking: --features clap-plugin... "
if cargo check --no-default-features --features clap-plugin > /dev/null 2>&1; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    exit 1
fi

echo -e "\n${GREEN}✅ Todas as combinações de features compilaram com sucesso!${NC}"
