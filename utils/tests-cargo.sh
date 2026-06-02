#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# Este script inteiro vem demorando 35-40 segundos pra completar (considerando diretório target/ devidamente povoado)
set -xeuo pipefail

# 1. Executa a bateria de testes de unidade e integridade padrão do crate [Costuma demorar uns 25 segundos]
cargo test

# 2. Compilação do plugin CLAP especificamente para testes (em debug com heap-audit)
echo "🔨 Compilando o plugin CLAP para testes locais..."
RUSTFLAGS="${RUSTFLAGS:-} -Clink-arg=-Wl,-soname,nam-rs.clap" \
  cargo build --target-dir target/clap-test --no-default-features --features "clap-plugin,heap-audit" --lib

# 3. Executa os testes de ciclo de vida do plugin apontando para o binário isolado
echo "🧪 Executando testes de ciclo de vida com heap-audit..."
CLAP_PLUGIN_PATH="target/clap-test/debug/libnam_rs.so" \
  NAM_HEAP_AUDIT=1 \
  cargo test --test clap_lifecycle_test --features "clap-plugin" --target-dir target/clap-test

# 4. Roda o validador oficial do CLAP no binário de debug com heap-audit ativo
echo "🔍 Validando o plugin com clap-validator e heap-audit..."
CLAP_PLUGIN_PATH="target/clap-test/debug/libnam_rs.so" \
  NAM_HEAP_AUDIT=1 \
  clap-validator validate target/clap-test/debug/libnam_rs.so

# 5. Executa benchmarks básicos
#No estágio atual, é mais interessante deixar para rodar apenas no utils/tests-long.sh
#cargo bench
