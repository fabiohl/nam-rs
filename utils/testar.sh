#!/bin/bash
# (c) 2026 Fábio Henrique de Lima Silva. Todos os direitos reservados.
# Este arquivo é confidencial e propriedade de Fábio Henrique de Lima Silva.
# O uso não autorizado é estritamente proibido.

set -euo pipefail

echo "⚙️ Assegurando o qpwgraph rodando..."
qpwgraph &

echo "⚙️ Compilando o AudioRip..."
cargo build

echo "🚀 Executando..."
pw-jack target/debug/audiorip --backend jack
# Vide docs/architecture.md - Adento: Porque " target/debug/audiorip --backend jack"?