#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

set -euo pipefail

echo "🎨 Formatando código..."
cargo fmt --all

echo "🚀 Check de features de compilação..."
cargo check --features standalone
cargo check --no-default-features
cargo check --no-default-features --features clap-plugin
cargo check --no-default-features --features clap-plugin-gui

echo "🔍 Executando Clippy (all features)..."
cargo clippy --all-targets --all-features -- -D warnings