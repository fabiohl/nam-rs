#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

set -euo pipefail

echo "⚙️ Assegurando o qpwgraph rodando..."
qpwgraph &

echo "⚙️ Compilando o NAM-rs..."
cargo build --release

echo "🚀 Executando..."
#target/debug/nam-rs
target/release/nam-rs