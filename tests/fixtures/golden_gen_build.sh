#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# golden_gen_build.sh — Builds the NeuralAmpModelerCore render tool and generates golden vectors
#
# Prerequisites:
#   - cmake >= 3.10, g++ or clang++ with C++20
#   - cargo (Rust; stress signal generation and WAV→golden conversion are now Rust native)
#   - git (to clone NeuralAmpModelerCore if needed)
#
# Python is no longer required — gen_stress and wav_to_golden replace the inline Python blocks.
#
# Usage:
#   ./tests/fixtures/golden_gen_build.sh
#
# Output (tests/fixtures/):
#   golden_wavenet_standard.bin, golden_wavenet_lite.bin, golden_wavenet_feather.bin, golden_wavenet_nano.bin
#   golden_lstm_1x8.bin, golden_lstm_1x12.bin, golden_lstm_1x16.bin, golden_lstm_1x24.bin, golden_lstm_1x40.bin
#   golden_lstm_2x8.bin, golden_lstm_2x12.bin, golden_lstm_2x16.bin, golden_lstm_2x24.bin
#   golden_wavenet_a2_full.bin, golden_wavenet_a2_lite.bin
#   (⚠ C++ upstream bug — a2_fast.cpp produces numerically unstable output;
#    these goldens are NOT used by any test. Self-goldens (_self.bin) provide
#    Rust determinism validation instead. See TODO-sprints.md §T7.8.)
#   (+ golden_*_v2_*k.bin for stress signal v2 multi-SR)
#   golden_cabsim_cpp_short.bin, golden_cabsim_cpp_medium.bin,
#   golden_cabsim_cpp_long.bin, golden_cabsim_cpp_stress.bin
#   (C++ dsp::ImpulseResponse reference for cabsim cross-validation)
#
# These files must be committed so that the Rust golden vector tests
# run without C++ recompilation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAM_CORE_DIR="$SCRIPT_DIR/NeuralAmpModelerCore"
BUILD_DIR="$PROJECT_ROOT/build/namcore_render"
MODELS_DIR="$SCRIPT_DIR/models"
FIXTURES_DIR="$SCRIPT_DIR"

# =============================================================================
# Prerequisite checks
# =============================================================================
echo "=== Golden Vector Generator (NeuralAmpModelerCore) ==="

for cmd in cmake cargo; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found. Install with: sudo apt install cmake cargo"
        exit 1
    fi
done

# Check C++20 compiler
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v g++ &>/dev/null; then
        CXX=g++
    elif command -v clang++ &>/dev/null; then
        CXX=clang++
    else
        echo "ERROR: C++ compiler not found. Install g++ or clang++."
        exit 1
    fi
fi
echo "  C++ Compiler: $CXX"

# =============================================================================
# Clone/update NeuralAmpModelerCore
# =============================================================================
echo ""
echo "[1/6] Setting up NeuralAmpModelerCore..."
if [ ! -d "$NAM_CORE_DIR" ]; then
    git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git "$NAM_CORE_DIR"
else
    echo "  NeuralAmpModelerCore already exists at $NAM_CORE_DIR"
fi

# Initialize submodules
for sub in eigen AudioDSPTools; do
    sub_path="$NAM_CORE_DIR/Dependencies/$sub"
    if [ -d "$sub_path" ] && [ -z "$(ls -A "$sub_path" 2>/dev/null)" ]; then
        echo "  Initializing submodule $sub..."
        (cd "$NAM_CORE_DIR" && git submodule update --init "Dependencies/$sub")
    fi
done

# =============================================================================
# Build render tool
# =============================================================================
echo ""
echo "[2/6] Building render tool..."

BUILD_TYPE="${BUILD_TYPE:-Release}"
RENDER_BIN="$BUILD_DIR/$BUILD_TYPE/render"

if [ -f "$RENDER_BIN" ]; then
    echo "  Render binary already exists: $RENDER_BIN"
else
    mkdir -p "$BUILD_DIR"
    cmake -S "$NAM_CORE_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_STANDARD=20 \
        2>&1 | tail -5
    cmake --build "$BUILD_DIR" --target render -j"$(nproc)" 2>&1 | tail -5

    if [ ! -f "$RENDER_BIN" ]; then
        # Try to find the binary elsewhere
        RENDER_BIN=$(find "$BUILD_DIR" -name render -type f -executable | head -1)
        if [ -z "$RENDER_BIN" ]; then
            echo "ERROR: Failed to build render tool."
            echo "Check that CMake and the C++20 compiler are working."
            exit 1
        fi
    fi
fi
echo "  Render: $RENDER_BIN"

# =============================================================================
# Build Rust tools (gen_stress + wav_to_golden + gen_lstm_fixtures)
# =============================================================================
echo ""
echo "[3/6] Building Rust tools (gen_stress + wav_to_golden + gen_lstm_fixtures)..."

cargo build --release --bin gen_stress --bin wav_to_golden --bin gen_lstm_fixtures 2>&1 | tail -3
GEN_STRESS="$PROJECT_ROOT/target/release/gen_stress"
WAV_TO_GOLDEN="$PROJECT_ROOT/target/release/wav_to_golden"
GEN_LSTM_FIXTURES="$PROJECT_ROOT/target/release/gen_lstm_fixtures"

if [ ! -f "$GEN_STRESS" ]; then
    echo "ERROR: Failed to build gen_stress binary."
    exit 1
fi
echo "  gen_stress: $GEN_STRESS"
echo "  wav_to_golden: $WAV_TO_GOLDEN"

# =============================================================================
# Generate stress WAV signals
# =============================================================================
echo ""
echo "[4/6] Generating stress signals..."

STRESS_WAV="$FIXTURES_DIR/stress_signal.wav"
"$GEN_STRESS" --version v1 --output "$STRESS_WAV" 2>&1
echo "  v1: $STRESS_WAV"

# Generate v2 signals for all supported sample rates
for SR in 44100 48000 88200 96000 192000; do
    V2_WAV="$FIXTURES_DIR/stress_signal_v2_${SR}.wav"
    "$GEN_STRESS" --version v2 --sample-rate "$SR" --output "$V2_WAV" 2>&1
    echo "  v2 ${SR}Hz: $V2_WAV"
done

# =============================================================================
# Run render for each model → WAV output → .golden.bin
# =============================================================================
echo ""
echo "[5/6] Running render for each model..."

# Generate synthetic LSTM fixtures if not already present
"$GEN_LSTM_FIXTURES" "$MODELS_DIR"

# Models: (.nam file, golden name, label)
MODELS=(
    "BossWN-standard.nam:golden_wavenet_standard:WaveNet Standard"
    "BossWN-lite.nam:golden_wavenet_lite:WaveNet Lite"
    "BossWN-feather.nam:golden_wavenet_feather:WaveNet Feather"
    "BossWN-nano.nam:golden_wavenet_nano:WaveNet Nano"
    "BossLSTM-1x8.nam:golden_lstm_1x8:LSTM 1×8"
    "BossLSTM-1x12.nam:golden_lstm_1x12:LSTM 1×12"
    "BossLSTM-1x16.nam:golden_lstm_1x16:LSTM 1×16"
    "BossLSTM-1x24.nam:golden_lstm_1x24:LSTM 1×24"
    "BossLSTM-1x40.nam:golden_lstm_1x40:LSTM 1×40"
    "BossLSTM-2x8.nam:golden_lstm_2x8:LSTM 2×8"
    "BossLSTM-2x12.nam:golden_lstm_2x12:LSTM 2×12"
    "BossLSTM-2x16.nam:golden_lstm_2x16:LSTM 2×16"
    "BossLSTM-2x24.nam:golden_lstm_2x24:LSTM 2×24"
    "wavenet_a2_full.nam:golden_wavenet_a2_full:A2-Full (CH=8)"
    "wavenet_a2_lite.nam:golden_wavenet_a2_lite:A2-Lite (CH=3)"
    "linear_test.nam:golden_linear_test:Linear RF=4"
)

TEMP_DIR="$FIXTURES_DIR/.temp_golden"
mkdir -p "$TEMP_DIR"

for entry in "${MODELS[@]}"; do
    IFS=':' read -r nam_file golden_name label <<< "$entry"
    MODEL_PATH="$MODELS_DIR/$nam_file"
    OUTPUT_WAV="$TEMP_DIR/${golden_name}.wav"
    GOLDEN_BIN="$FIXTURES_DIR/${golden_name}.bin"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIP: $nam_file not found at $MODELS_DIR"
        continue
    fi

    echo "  Processing $label ($nam_file)..."

    "$RENDER_BIN" "$MODEL_PATH" "$STRESS_WAV" "$OUTPUT_WAV" 2>&1 | tail -1

    if [ ! -f "$OUTPUT_WAV" ]; then
        echo "  ERROR: Render failed for $label"
        continue
    fi

    # Convert WAV output → .golden.bin (Rust native replacement for Python block)
    "$WAV_TO_GOLDEN" \
        --input "$OUTPUT_WAV" \
        --reference "$STRESS_WAV" \
        --output "$GOLDEN_BIN" 2>&1

done

# =============================================================================
# Generate v2 goldens for all SRs × models
# =============================================================================
echo ""
echo "[5a/6] Generating v2 golden vectors (multi-SR)..."

for SR in 44100 48000 88200 96000 192000; do
    V2_WAV="$FIXTURES_DIR/stress_signal_v2_${SR}.wav"

    for entry in "${MODELS[@]}"; do
        IFS=':' read -r nam_file golden_name label <<< "$entry"
        MODEL_PATH="$MODELS_DIR/$nam_file"
        GOLDEN_NAME_V2="${golden_name}_v2_${SR}k"
        OUTPUT_WAV="$TEMP_DIR/${GOLDEN_NAME_V2}.wav"
        GOLDEN_BIN="$FIXTURES_DIR/${GOLDEN_NAME_V2}.bin"

        if [ ! -f "$MODEL_PATH" ]; then
            continue
        fi

        echo "  Processing $label @ ${SR}Hz..."

        "$RENDER_BIN" "$MODEL_PATH" "$V2_WAV" "$OUTPUT_WAV" 2>&1 | tail -1 || true

        if [ ! -f "$OUTPUT_WAV" ]; then
            echo "  WARN: Render failed for $label @ ${SR}Hz"
            continue
        fi

        "$WAV_TO_GOLDEN" \
            --input "$OUTPUT_WAV" \
            --reference "$V2_WAV" \
            --output "$GOLDEN_BIN" 2>&1
    done
done

# =============================================================================
# Build and run C++ IR reference (dsp::ImpulseResponse) → golden_cabsim_cpp_*.bin
# =============================================================================
echo ""
echo "[5b/6] Building C++ IR reference (dsp::ImpulseResponse)..."

AUDIO_DSP_TOOLS_DIR="$FIXTURES_DIR/NeuralAmpModelerPlugin/AudioDSPTools"
IR_BIN="$FIXTURES_DIR/render_ir"

# Ensure AudioDSPTools submodule dependencies are present
if [ ! -f "$AUDIO_DSP_TOOLS_DIR/dsp/ImpulseResponse.cpp" ]; then
    echo "  Initializing NeuralAmpModelerPlugin/AudioDSPTools submodules..."
    (cd "$FIXTURES_DIR/NeuralAmpModelerPlugin" && git submodule update --init AudioDSPTools)
    (cd "$AUDIO_DSP_TOOLS_DIR" && git submodule update --init Dependencies/eigen)
fi

if [ ! -d "$AUDIO_DSP_TOOLS_DIR/Dependencies/eigen/Eigen" ]; then
    echo "  Initializing eigen submodule for AudioDSPTools..."
    (cd "$AUDIO_DSP_TOOLS_DIR" && git submodule update --init Dependencies/eigen)
fi

if [ -f "$IR_BIN" ]; then
    echo "  IR reference binary already exists: $IR_BIN"
else
    echo "  Compiling render_ir.cpp..."
    "$CXX" -std=c++17 -O2 \
        -I "$AUDIO_DSP_TOOLS_DIR" \
        -I "$AUDIO_DSP_TOOLS_DIR/Dependencies/eigen" \
        -I "$AUDIO_DSP_TOOLS_DIR/Dependencies/nlohmann" \
        -D "FIXTURES_DIR=\"$FIXTURES_DIR\"" \
        "$FIXTURES_DIR/render_ir.cpp" \
        "$AUDIO_DSP_TOOLS_DIR/dsp/dsp.cpp" \
        "$AUDIO_DSP_TOOLS_DIR/dsp/ImpulseResponse.cpp" \
        "$AUDIO_DSP_TOOLS_DIR/dsp/wav.cpp" \
        -o "$IR_BIN" \
        -lstdc++fs \
        2>&1

    if [ ! -f "$IR_BIN" ]; then
        echo "  ERROR: Failed to build render_ir binary."
        echo "  Check that the g++ compiler and Eigen headers are available."
        exit 1
    fi
fi

echo "  Running render_ir to generate C++ IR golden vectors..."
"$IR_BIN"

# =============================================================================
# Cleanup
# =============================================================================
echo ""
echo "[6/6] Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "=== Golden vectors generated successfully ==="
echo "  Files at $FIXTURES_DIR/:"
for entry in "${MODELS[@]}"; do
    IFS=':' read -r _ golden_name _ <<< "$entry"
    [ -f "$FIXTURES_DIR/${golden_name}.bin" ] && echo "    ${golden_name}.bin"
done
for SR in 44100 48000 88200 96000 192000; do
    for entry in "${MODELS[@]}"; do
        IFS=':' read -r _ golden_name _ <<< "$entry"
        GF="$FIXTURES_DIR/${golden_name}_v2_${SR}k.bin"
        [ -f "$GF" ] && echo "    ${golden_name}_v2_${SR}k.bin"
    done
done
for cpp_file in golden_cabsim_cpp_short.bin golden_cabsim_cpp_medium.bin \
                 golden_cabsim_cpp_long.bin golden_cabsim_cpp_stress.bin; do
    [ -f "$FIXTURES_DIR/$cpp_file" ] && echo "    $cpp_file"
done
echo ""
echo "Commit these files so that the Rust golden vector tests work."
