#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Build and install the NAM-rs plugin in CLAP format (Release/Production).
# Generates libnam_rs.so and copies it to ~/.clap/nam-rs.clap
#

set -xeuo pipefail

DEST_PATH="$HOME/.clap/nam-rs.clap"

echo "🔨 Building NAM-rs CLAP plugin in release mode..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
    cargo build --release --target-dir target/clap --no-default-features --features "clap-plugin" --lib
sync

echo "📁 Installing to $DEST_PATH ..."
mkdir -p "$HOME/.clap"
rm -f "$DEST_PATH"
cp target/clap/release/libnam_rs.so "$DEST_PATH"
sync
ls -lath "$DEST_PATH"

echo "🔍 Auditing binary validity..."

# 1. SONAME check (valid Shared Object)
if readelf -d "$DEST_PATH" | grep -q SONAME; then
    echo "  ✅ SONAME found."
else
    echo "  ❌ ERROR: SONAME not found in binary!"
    exit 1
fi

# 2. CLAP entry symbol check
if nm -D "$DEST_PATH" | grep -q "clap_entry"; then
    echo "  ✅ Symbol 'clap_entry' found."
else
    echo "  ❌ ERROR: 'clap_entry' symbol not found! Plugin will not load."
    exit 1
fi

# 3. ELF 64-bit file type check
FILE_INFO=$(file "$DEST_PATH")
if [[ $FILE_INFO == *"ELF 64-bit LSB shared object"* ]] && [[ $FILE_INFO == *"x86-64"* ]]; then
    echo "  ✅ ELF 64-bit x86-64 format confirmed."
else
    echo "  ❌ ERROR: Invalid file format: $FILE_INFO"
    exit 1
fi
