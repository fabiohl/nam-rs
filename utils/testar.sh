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
target/release/nam-rs --model tests/nam_files/EVH-5150-Lite.nam --input-gain 2.0 --output-gain 5.0