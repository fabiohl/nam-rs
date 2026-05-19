#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Build e instalação do plugin NAM-rs no formato CLAP.
# Gera libnam_rs.so e copia para ~/.clap/nam-rs.clap

set -euo pipefail

BUILD_MODE="release"
CARGO_FLAGS="--release"
TARGET_DIR="target/release"
FEATURES="clap-plugin"

# Processar argumentos
for arg in "$@"; do
    if [ "$arg" == "--debug" ]; then
        BUILD_MODE="debug"
        CARGO_FLAGS=""
        TARGET_DIR="target/debug"
    elif [ "$arg" == "--gui" ]; then
        FEATURES="clap-plugin-gui"
    fi
done

CLAP_DIR="$HOME/.clap"
PLUGIN_NAME="nam-rs.clap"

echo "🔨 Building NAM-rs CLAP plugin ($BUILD_MODE) with features: $FEATURES..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
    cargo build $CARGO_FLAGS --no-default-features --features "$FEATURES" --lib

echo "📁 Instalando em $CLAP_DIR/$PLUGIN_NAME ..."
mkdir -p "$CLAP_DIR"
cp "$TARGET_DIR/libnam_rs.so" "$CLAP_DIR/$PLUGIN_NAME"

echo "🔍 Auditando validade do binário..."

# 1. Verificação de SONAME (Shared Object válida)
if readelf -d "$CLAP_DIR/$PLUGIN_NAME" | grep -q SONAME; then
    echo "  ✅ SONAME encontrado."
else
    echo "  ❌ ERRO: SONAME não encontrado no binário!"
    exit 1
fi

# 2. Verificação de símbolo de entrada CLAP
if nm -D "$CLAP_DIR/$PLUGIN_NAME" | grep -q "clap_entry"; then
    echo "  ✅ Símbolo 'clap_entry' encontrado."
else
    echo "  ❌ ERRO: Símbolo 'clap_entry' não encontrado! O plugin não será carregado."
    exit 1
fi

# 3. Verificação de tipo de arquivo ELF 64-bit
FILE_INFO=$(file "$CLAP_DIR/$PLUGIN_NAME")
if [[ $FILE_INFO == *"ELF 64-bit LSB shared object"* ]] && [[ $FILE_INFO == *"x86-64"* ]]; then
    echo "  ✅ Formato ELF 64-bit x86-64 confirmado."
else
    echo "  ❌ ERRO: Formato de arquivo inválido: $FILE_INFO"
    exit 1
fi

# 4. Execução do teste de integração do ciclo de vida do CLAP
echo "🧪 Executando teste de ciclo de vida do CLAP..."
cargo test --test clap_lifecycle_test --features "$FEATURES"

echo "✅ Plugin instalado, validado e testado com sucesso: $CLAP_DIR/$PLUGIN_NAME"
ls -lath "$CLAP_DIR/$PLUGIN_NAME"
echo "📝 Reabra a DAW e faça um novo scan de plugins CLAP."
