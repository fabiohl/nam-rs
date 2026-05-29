#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Build e instalação do plugin NAM-rs no formato CLAP (Release/Produção).
# Gera libnam_rs.so e copia para ~/.clap/nam-rs.clap
#

set -xeuo pipefail

DEST_PATH="$HOME/.clap/nam-rs.clap"

echo "🔨 Building NAM-rs CLAP plugin in release mode..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
    cargo build --release --target-dir target/clap --no-default-features --features "clap-plugin" --lib

echo "📁 Instalando em $DEST_PATH ..."
mkdir -p "$HOME/.clap"
rm -f "$DEST_PATH"
cp target/clap/release/libnam_rs.so "$DEST_PATH"
ls -lath "$DEST_PATH"

echo "🔍 Auditando validade do binário..."

# 1. Verificação de SONAME (Shared Object válida)
if readelf -d "$DEST_PATH" | grep -q SONAME; then
    echo "  ✅ SONAME encontrado."
else
    echo "  ❌ ERRO: SONAME não encontrado no binário!"
    exit 1
fi

# 2. Verificação de símbolo de entrada CLAP
if nm -D "$DEST_PATH" | grep -q "clap_entry"; then
    echo "  ✅ Símbolo 'clap_entry' encontrado."
else
    echo "  ❌ ERRO: Símbolo 'clap_entry' não encontrado! O plugin não será carregado."
    exit 1
fi

# 3. Verificação de tipo de arquivo ELF 64-bit
FILE_INFO=$(file "$DEST_PATH")
if [[ $FILE_INFO == *"ELF 64-bit LSB shared object"* ]] && [[ $FILE_INFO == *"x86-64"* ]]; then
    echo "  ✅ Formato ELF 64-bit x86-64 confirmado."
else
    echo "  ❌ ERRO: Formato de arquivo inválido: $FILE_INFO"
    exit 1
fi
