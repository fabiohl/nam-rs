#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Quick manual test script for the standalone NAM-rs binary.
# Expects target/release/nam-rs to be already compiled (e.g. via utils/build-release.sh).

set -euo pipefail

source "$(dirname "$0")/_lib.sh"

# Ensure the standalone binary is present
NAM_RS_BIN="$HOME/.local/bin/nam-rs"
if [ ! -f "$NAM_RS_BIN" ]; then
    echo -e "${RED}${BOLD}❌ Erro: Binário standalone não encontrado em: $NAM_RS_BIN${NC}"
    echo "   Por favor, compile-o primeiro utilizando o pipeline de release:"
    echo "     ./utils/build-release.sh"
    echo "   Ou de forma simples via cargo:"
    echo "     cargo build --release --features standalone"
    exit 1
fi

echo -e "${BLUE}${BOLD}⚙️ Garantindo que qpwgraph está rodando...${NC}"
if ! pgrep qpwgraph >/dev/null 2>&1; then
    qpwgraph &
fi

# Try to run VLC in background if not already running (for input audio source)
if ! pgrep vlc >/dev/null 2>&1; then
    vlc &
fi

echo -e "${GREEN}${BOLD}🚀 Iniciando executável standalone...${NC}"
"$NAM_RS_BIN" \
  --model tests/fixtures/models/BossWN-standard.nam \
  --input-gain 2.0 \
  --output-gain 5.0

