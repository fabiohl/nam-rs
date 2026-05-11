#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.
#
# Build e instalação do plugin NAM-rs no formato CLAP.
# Gera libnam_rs.so e copia para ~/.clap/nam-rs.clap

set -euo pipefail

TARGET_DIR="target/release"
CLAP_DIR="$HOME/.clap"
PLUGIN_NAME="nam-rs.clap"

echo "🔨 Building NAM-rs CLAP plugin (release)..."
cargo build --release --no-default-features --features clap-plugin --lib

echo "📁 Instalando em $CLAP_DIR/$PLUGIN_NAME ..."
mkdir -p "$CLAP_DIR"
cp "$TARGET_DIR/libnam_rs.so" "$CLAP_DIR/$PLUGIN_NAME"

echo "✅ Plugin instalado: $CLAP_DIR/$PLUGIN_NAME"
ls -lath $CLAP_DIR/$PLUGIN_NAME
echo "📝 Reabra a DAW e faça um novo scan de plugins CLAP."
