#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

# golden_gen_build.sh — Compila o gerador de golden vectors C++ e gera os .golden.bin
#
# Pré-requisitos:
#   - Árvore NeuralAudio C++ completa em github.com/mikeoliphant/NeuralAudio/
#   - CMake >= 3.16, g++ ou clang++ com C++17
#   - Bibliotecas Eigen3 (header-only, incluída nos deps do NeuralAudio)
#
# Uso:
#   ./tests/fixtures/golden_gen_build.sh
#
# Saída:
#   tests/fixtures/golden_wavenet_standard.bin
#   tests/fixtures/golden_wavenet_feather.bin
#   tests/fixtures/golden_wavenet_nano.bin
#   tests/fixtures/golden_lstm_1x16.bin
#
# Estes arquivos devem ser commitados no repositório para que os testes Rust
# de golden vectors (test_golden_vectors_wavenet[_feather|_nano], test_golden_vectors_lstm)
# possam executar sem recompilação C++.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

NEURAL_AUDIO_ROOT="$PROJECT_ROOT/github.com/mikeoliphant/NeuralAudio"
BUILD_DIR="$NEURAL_AUDIO_ROOT/build/golden_gen"
MODELS_DIR="$SCRIPT_DIR/models"
FIXTURES_DIR="$SCRIPT_DIR"

echo "=== Golden Vector Generator ==="
echo "  NeuralAudio: $NEURAL_AUDIO_ROOT"
echo "  Build dir:   $BUILD_DIR"
echo "  Models:      $MODELS_DIR"
echo "  Fixtures:    $FIXTURES_DIR"
echo ""

# Verificar que os modelos existem
if [ ! -f "$MODELS_DIR/BossWN-standard.nam" ]; then
    echo "ERRO: BossWN-standard.nam não encontrado em $MODELS_DIR"
    exit 1
fi

if [ ! -f "$MODELS_DIR/BossLSTM-1x16.nam" ]; then
    echo "ERRO: BossLSTM-1x16.nam não encontrado em $MODELS_DIR"
    exit 1
fi

if [ ! -f "$MODELS_DIR/BossWN-feather.nam" ]; then
    echo "ERRO: BossWN-feather.nam não encontrado em $MODELS_DIR"
    exit 1
fi

if [ ! -f "$MODELS_DIR/BossWN-nano.nam" ]; then
    echo "ERRO: BossWN-nano.nam não encontrado em $MODELS_DIR"
    exit 1
fi

# Criar diretórios
mkdir -p "$BUILD_DIR"
mkdir -p "$FIXTURES_DIR"

# Compilar golden_gen.cpp usando CMake do NeuralAudio como base
cd "$BUILD_DIR"

echo "[1/3] Configurando CMake..."
cmake "$NEURAL_AUDIO_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    2>&1 || {
    echo ""
    echo "WARN: CMake falhou. Tentando compilação manual..."
    echo ""
    
    # Fallback: compilação manual sem CMake
    INCLUDE_DIRS=(
        "-I$NEURAL_AUDIO_ROOT"
        "-I$NEURAL_AUDIO_ROOT/NeuralAudio"
        "-I$NEURAL_AUDIO_ROOT/deps/NeuralAmpModelerCore"
        "-I$NEURAL_AUDIO_ROOT/deps/RTNeural"
        "-I$NEURAL_AUDIO_ROOT/deps/math_approx/include"
    )
    
    echo "Compilação manual não suportada neste script."
    echo "Configure o NeuralAudio C++ manualmente e reexecute."
    exit 1
}

echo "Compilando NeuralAudio com cmake --build . ..."
cmake --build .

echo "[2/3] Compilando golden_gen..."
# Coletar object files da library OBJECT NeuralAudio
NEURAL_AUDIO_OBJS=$(find "$BUILD_DIR/NeuralAudio/CMakeFiles/NeuralAudio.dir" -name "*.o")

# Compilar o gerador customizado
g++ -std=c++17 -O2 \
    -I"$NEURAL_AUDIO_ROOT" \
    -I"$NEURAL_AUDIO_ROOT/deps/NeuralAmpModelerCore/Dependencies/nlohmann" \
    -I"$NEURAL_AUDIO_ROOT/deps/RTNeural" \
    "$SCRIPT_DIR/golden_gen.cpp" \
    -o "$BUILD_DIR/golden_gen" \
    $NEURAL_AUDIO_OBJS \
    -L"$BUILD_DIR/NeuralAudio/RTNeural/RTNeural" -lRTNeural \
    -lm \
    2>&1 || {
    echo "ERRO: Compilação de golden_gen falhou."
    echo "Verifique dependências C++ (Eigen3, NeuralAudio)."
    exit 1
}

echo "[3/3] Gerando golden vectors..."

# WaveNet Standard
echo "  Gerando WaveNet Standard..."
"$BUILD_DIR/golden_gen" \
    "$MODELS_DIR/BossWN-standard.nam" \
    "$FIXTURES_DIR/golden_wavenet_standard.bin"

# LSTM 1x16
echo "  Gerando LSTM 1×16..."
"$BUILD_DIR/golden_gen" \
    "$MODELS_DIR/BossLSTM-1x16.nam" \
    "$FIXTURES_DIR/golden_lstm_1x16.bin"

# WaveNet Feather
echo "  Gerando WaveNet Feather..."
"$BUILD_DIR/golden_gen" \
    "$MODELS_DIR/BossWN-feather.nam" \
    "$FIXTURES_DIR/golden_wavenet_feather.bin"

# WaveNet Nano
echo "  Gerando WaveNet Nano..."
"$BUILD_DIR/golden_gen" \
    "$MODELS_DIR/BossWN-nano.nam" \
    "$FIXTURES_DIR/golden_wavenet_nano.bin"

echo ""
echo "=== Golden vectors gerados com sucesso ==="
echo "  $FIXTURES_DIR/golden_wavenet_standard.bin"
echo "  $FIXTURES_DIR/golden_lstm_1x16.bin"
echo "  $FIXTURES_DIR/golden_wavenet_feather.bin"
echo "  $FIXTURES_DIR/golden_wavenet_nano.bin"
echo ""
echo "Commite estes arquivos para que os testes Rust de golden vectors funcionem."
