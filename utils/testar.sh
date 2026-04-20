#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

set -euo pipefail

echo "⚙️ Assegurando o qpwgraph rodando..."
qpwgraph &

echo "⚙️ Compilando o NAM-rs..."
cargo build --release
ls -lath target/release/nam-rs

echo "🚀 Executando..."
target/release/nam-rs --model target/release/nam-rs --model tests/nam_files/ChandlerRedd47-Gain34-Standard.nam # --input-gain 2.0 --output-gain 5.0 --buffer-size 512