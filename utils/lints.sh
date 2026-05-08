#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

set -euo pipefail

echo "🎨 Formatando código..."
cargo fmt --all

echo "🚀 Cehck de features de compilação..."
cargo check --features standalone
cargo check --no-default-features
cargo check --no-default-features --features clap-plugin

echo "🔍 Executando Clippy (all features)..."
cargo clippy --all-targets --all-features -- -D warnings