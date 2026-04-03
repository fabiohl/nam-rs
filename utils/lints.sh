#!/bin/bash
# (c) 2026 Fábio Henrique de Lima Silva. Todos os direitos reservados.
# Este arquivo é confidencial e propriedade de Fábio Henrique de Lima Silva.
# O uso não autorizado é estritamente proibido.

set -xeuo pipefail
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
#cargo clippy --all-targets --all-features -- -D warnings -D clippy::pedantic -D clippy::nursery -D clippy::cargo