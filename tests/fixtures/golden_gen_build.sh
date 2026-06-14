#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# golden_gen_build.sh — Builds the NeuralAmpModelerCore render tool, clones 
# NeuralAmpModelerPlugin (C++ IR reference), and generates all golden vectors.
#
# Prerequisites:
#   - cmake >= 3.10, g++ or clang++ with C++20
#   - cargo (Rust; stress signal generation and WAV→golden conversion are now Rust native)
#   - git (to clone NeuralAmpModelerCore and NeuralAmpModelerPlugin if needed)
#
# Reproducibility:
#   Upstream commits are pinned in NAM_CORE_COMMIT and NAM_PLUGIN_COMMIT.
#   Update these variables when regenerating goldens from a newer upstream version.
#
# Python is no longer required — gen_stress and wav_to_golden replace the inline Python blocks.
#
# Usage:
#   ./tests/fixtures/golden_gen_build.sh
#
# Unversioned local mirrors created:
#   tests/fixtures/NeuralAmpModelerCore/   (~143 MB) — C++ upstream render engine
#   tests/fixtures/NeuralAmpModelerPlugin/  (~164 MB) — C++ upstream plugin (IR reference)
#   build/namcore_render/                   (~6 MB)  — C++ build artifacts
#   These directories are gitignored (see ../.gitignore).
#
# Output (tests/fixtures/):
#   golden_wavenet_standard.bin, golden_wavenet_lite.bin, golden_wavenet_feather.bin, golden_wavenet_nano.bin
#   golden_lstm_1x8.bin, golden_lstm_1x12.bin, golden_lstm_1x16.bin, golden_lstm_1x24.bin, golden_lstm_1x40.bin
#   golden_lstm_2x8.bin, golden_lstm_2x12.bin, golden_lstm_2x16.bin, golden_lstm_2x24.bin
#   golden_wavenet_a2_full.bin, golden_wavenet_a2_lite.bin
#   (A2 goldens are cross-reference Rust↔C++ v0.5.3 via ESR/SNR scale-invariant
#    gate — self-goldens removed in T2.6. See TODO-sprints.md Épico 2.)
#   golden_cabsim_cpp_short.bin, golden_cabsim_cpp_medium.bin,
#   golden_cabsim_cpp_long.bin
#   (C++ dsp::ImpulseResponse reference for cabsim cross-validation)
#
# These files must be committed so that the Rust golden vector tests
# run without C++ recompilation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAM_CORE_DIR="$SCRIPT_DIR/NeuralAmpModelerCore"
NAM_CORE_V053_DIR="$SCRIPT_DIR/NeuralAmpModelerCore_v0.5.3"
NAM_PLUGIN_DIR="$SCRIPT_DIR/NeuralAmpModelerPlugin"
BUILD_DIR="$PROJECT_ROOT/build/namcore_render"
BUILD_V053_DIR="$PROJECT_ROOT/build/namcore_render_v053"
MODELS_DIR="$SCRIPT_DIR/models"
FIXTURES_DIR="$SCRIPT_DIR"

# Pinned upstream commits for reproducibility.
# Update these when regenerating goldens with a newer upstream version.
NAM_CORE_COMMIT="e49c93e678549230d09efbb0beeb50511e387874"
NAM_CORE_V053_COMMIT="9c7b185de346fe0725dea537bcee4bc38b5bb6d6" # v0.5.3
NAM_PLUGIN_COMMIT="96337e9ab6e3beb619459779bbb5c47e1b04d8c4"

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
# Clone/update NeuralAmpModelerPlugin and dependencies
# =============================================================================
echo ""
echo "[1/6] Setting up NeuralAmpModelerPlugin (C++ IR reference)..."
if [ ! -d "$NAM_PLUGIN_DIR" ]; then
    git clone https://github.com/sdatkinson/NeuralAmpModelerPlugin.git "$NAM_PLUGIN_DIR"
    echo "  Checking out pinned commit $NAM_PLUGIN_COMMIT..."
    (cd "$NAM_PLUGIN_DIR" && git checkout "$NAM_PLUGIN_COMMIT")
else
    CURRENT_PLUGIN_SHA=$(cd "$NAM_PLUGIN_DIR" && git rev-parse HEAD)
    if [ "$CURRENT_PLUGIN_SHA" != "$NAM_PLUGIN_COMMIT" ]; then
        echo "  WARNING: NeuralAmpModelerPlugin at $CURRENT_PLUGIN_SHA, pinned commit is $NAM_PLUGIN_COMMIT"
        echo "  Consider deleting the directory and re-running this script for reproducibility."
    else
        echo "  NeuralAmpModelerPlugin already exists at $NAM_PLUGIN_DIR (pinned commit verified)"
    fi
fi

# Ensure AudioDSPTools submodule dependencies are present
AUDIO_DSP_TOOLS_DIR="$NAM_PLUGIN_DIR/AudioDSPTools"
if [ ! -f "$AUDIO_DSP_TOOLS_DIR/dsp/ImpulseResponse.cpp" ]; then
    echo "  Initializing NeuralAmpModelerPlugin/AudioDSPTools submodules..."
    (cd "$NAM_PLUGIN_DIR" && git submodule update --init AudioDSPTools)
fi

if [ ! -d "$AUDIO_DSP_TOOLS_DIR/Dependencies/eigen/Eigen" ]; then
    echo "  Initializing eigen submodule for AudioDSPTools..."
    (cd "$AUDIO_DSP_TOOLS_DIR" && git submodule update --init Dependencies/eigen)
fi

# =============================================================================
# Clone/update NeuralAmpModelerCore (standard)
# =============================================================================
echo ""
echo "[1b/6] Setting up NeuralAmpModelerCore..."
if [ ! -d "$NAM_CORE_DIR" ]; then
    git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git "$NAM_CORE_DIR"
    echo "  Checking out pinned commit $NAM_CORE_COMMIT..."
    (cd "$NAM_CORE_DIR" && git fetch --depth 1 origin "$NAM_CORE_COMMIT" && git checkout "$NAM_CORE_COMMIT")
else
    CURRENT_CORE_SHA=$(cd "$NAM_CORE_DIR" && git rev-parse HEAD)
    if [ "$CURRENT_CORE_SHA" != "$NAM_CORE_COMMIT" ]; then
        echo "  WARNING: NeuralAmpModelerCore at $CURRENT_CORE_SHA, pinned commit is $NAM_CORE_COMMIT"
        echo "  Consider deleting the directory and re-running this script for reproducibility."
    else
        echo "  NeuralAmpModelerCore already exists at $NAM_CORE_DIR (pinned commit verified)"
    fi
fi

# Initialize submodules for standard core
for sub in eigen AudioDSPTools; do
    sub_path="$NAM_CORE_DIR/Dependencies/$sub"
    if [ -d "$sub_path" ] && [ -z "$(ls -A "$sub_path" 2>/dev/null)" ]; then
        echo "  Initializing submodule $sub..."
        (cd "$NAM_CORE_DIR" && git submodule update --init "Dependencies/$sub")
    fi
done

# =============================================================================
# Clone/update NeuralAmpModelerCore v0.5.3 (A2 Reference)
# =============================================================================
echo ""
echo "[1c/6] Setting up NeuralAmpModelerCore v0.5.3..."
if [ -d "$PROJECT_ROOT/github.com/NeuralAmpModelerCore_v0.5.3" ]; then
    echo "  Using local mirror of NeuralAmpModelerCore v0.5.3 from github.com/..."
    ln -sfn "$PROJECT_ROOT/github.com/NeuralAmpModelerCore_v0.5.3" "$NAM_CORE_V053_DIR"
elif [ ! -d "$NAM_CORE_V053_DIR" ]; then
    git clone https://github.com/sdatkinson/NeuralAmpModelerCore.git "$NAM_CORE_V053_DIR"
    echo "  Checking out pinned commit $NAM_CORE_V053_COMMIT..."
    (cd "$NAM_CORE_V053_DIR" && git fetch origin "$NAM_CORE_V053_COMMIT" && git checkout "$NAM_CORE_V053_COMMIT")
else
    CURRENT_CORE_V053_SHA=$(cd "$NAM_CORE_V053_DIR" && git rev-parse HEAD)
    if [ "$CURRENT_CORE_V053_SHA" != "$NAM_CORE_V053_COMMIT" ]; then
        echo "  WARNING: NeuralAmpModelerCore v0.5.3 at $CURRENT_CORE_V053_SHA, pinned commit is $NAM_CORE_V053_COMMIT"
    else
        echo "  NeuralAmpModelerCore v0.5.3 already exists (pinned commit verified)"
    fi
fi

# Supply dependencies (Eigen and AudioDSPTools) via symlinks
PLUGIN_DSP="$NAM_PLUGIN_DIR/AudioDSPTools"
echo "  Symlinking dependencies (Eigen and AudioDSPTools) into v0.5.3..."
rm -rf "$NAM_CORE_V053_DIR/Dependencies/eigen"
ln -sfn "$PLUGIN_DSP/Dependencies/eigen" "$NAM_CORE_V053_DIR/Dependencies/eigen"
rm -rf "$NAM_CORE_V053_DIR/Dependencies/AudioDSPTools"
ln -sfn "$PLUGIN_DSP" "$NAM_CORE_V053_DIR/Dependencies/AudioDSPTools"

# Copy official models from standard core clone for test and provenance
echo "  Copying official example models from mirror for testing..."
cp "$NAM_CORE_DIR/example_models/wavenet_a2_max.nam" "$MODELS_DIR/"
cp "$NAM_CORE_DIR/example_models/slimmable_wavenet.nam" "$MODELS_DIR/"
cp "$NAM_CORE_DIR/example_models/slimmable_container.nam" "$MODELS_DIR/"
cp "$NAM_CORE_DIR/example_models/wavenet_condition_dsp.nam" "$MODELS_DIR/"

# =============================================================================
# Build render tools (standard and v0.5.3 A2-fast)
# =============================================================================
echo ""
echo "[2/6] Building render tools..."

BUILD_TYPE="${BUILD_TYPE:-Release}"
RENDER_BIN="$BUILD_DIR/$BUILD_TYPE/render"
RENDER_V053_BIN="$BUILD_V053_DIR/$BUILD_TYPE/render"

# 1) Build standard render
if [ -f "$RENDER_BIN" ]; then
    echo "  Standard render binary already exists: $RENDER_BIN"
else
    echo "  Building standard render tool..."
    mkdir -p "$BUILD_DIR"
    cmake -S "$NAM_CORE_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_STANDARD=20 \
        2>&1 | tail -5
    cmake --build "$BUILD_DIR" --target render -j"$(nproc)" 2>&1 | tail -5

    if [ ! -f "$RENDER_BIN" ]; then
        RENDER_BIN=$(find "$BUILD_DIR" -name render -type f -executable | head -1)
        if [ -z "$RENDER_BIN" ]; then
            echo "ERROR: Failed to build standard render tool."
            exit 1
        fi
    fi
fi
echo "  Standard Render: $RENDER_BIN"

# 2) Build v0.5.3 render with A2-fast
if [ -f "$RENDER_V053_BIN" ]; then
    echo "  v0.5.3 render binary already exists: $RENDER_V053_BIN"
else
    echo "  Building v0.5.3 render tool (with A2-fast)..."
    mkdir -p "$BUILD_V053_DIR"
    cmake -S "$NAM_CORE_V053_DIR" -B "$BUILD_V053_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_STANDARD=20 \
        -DNAM_ENABLE_A2_FAST=ON \
        2>&1 | tail -5
    cmake --build "$BUILD_V053_DIR" --target render -j"$(nproc)" 2>&1 | tail -5

    if [ ! -f "$RENDER_V053_BIN" ]; then
        RENDER_V053_BIN=$(find "$BUILD_V053_DIR" -name render -type f -executable | head -1)
        if [ -z "$RENDER_V053_BIN" ]; then
            echo "ERROR: Failed to build v0.5.3 render tool."
            exit 1
        fi
    fi
fi
echo "  v0.5.3 Render: $RENDER_V053_BIN"

# =============================================================================
# Build Rust tools (gen_stress + wav_to_golden)
# =============================================================================
echo ""
echo "[3/6] Building Rust tools (gen_stress + wav_to_golden)..."

cargo build --release --bin gen_stress --bin wav_to_golden 2>&1 | tail -3
GEN_STRESS="$PROJECT_ROOT/target/release/gen_stress"
WAV_TO_GOLDEN="$PROJECT_ROOT/target/release/wav_to_golden"

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

# Models: (.nam file : golden name : label : render_type)
MODELS=(
    "BossWN-standard.nam:golden_wavenet_standard:WaveNet Standard:standard"
    "BossWN-lite.nam:golden_wavenet_lite:WaveNet Lite:standard"
    "BossWN-feather.nam:golden_wavenet_feather:WaveNet Feather:standard"
    "BossWN-nano.nam:golden_wavenet_nano:WaveNet Nano:standard"
    "wavenet_a1_standard.nam:golden_wavenet_a1_standard:WaveNet A1 Standard (Official):standard"
    "BossLSTM-1x16.nam:golden_lstm_1x16:LSTM 1×16:standard"
    "BossLSTM-2x8.nam:golden_lstm_2x8:LSTM 2×8:standard"
    "lstm.nam:golden_lstm_official:LSTM Official:standard"
    "wavenet_a2_full.nam:golden_wavenet_a2_full:A2-Full (CH=8):v0.5.3"
    "wavenet_a2_lite.nam:golden_wavenet_a2_lite:A2-Lite (CH=3):v0.5.3"
)

TEMP_DIR="$FIXTURES_DIR/.temp_golden"
mkdir -p "$TEMP_DIR"

for entry in "${MODELS[@]}"; do
    IFS=':' read -r nam_file golden_name label render_type <<< "$entry"
    MODEL_PATH="$MODELS_DIR/$nam_file"
    OUTPUT_WAV="$TEMP_DIR/${golden_name}.wav"
    GOLDEN_BIN="$FIXTURES_DIR/${golden_name}.bin"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIP: $nam_file not found at $MODELS_DIR"
        continue
    fi

    # Determine render binary to use
    if [ "$render_type" = "v0.5.3" ]; then
        ACTIVE_RENDER="$RENDER_V053_BIN"
    else
        ACTIVE_RENDER="$RENDER_BIN"
    fi

    echo "  Processing $label ($nam_file) using $render_type render..."

    "$ACTIVE_RENDER" "$MODEL_PATH" "$STRESS_WAV" "$OUTPUT_WAV" 2>&1 | tail -1

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
# Build and run C++ IR reference (dsp::ImpulseResponse) → golden_cabsim_cpp_*.bin
# =============================================================================
echo ""
echo "[5/5] Building C++ IR reference (dsp::ImpulseResponse)..."

AUDIO_DSP_TOOLS_DIR="$NAM_PLUGIN_DIR/AudioDSPTools"
IR_BIN="$FIXTURES_DIR/render_ir"

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
echo "[5/5] Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "=== Golden vectors generated successfully ==="
echo "  Files at $FIXTURES_DIR/:"
for entry in "${MODELS[@]}"; do
    IFS=':' read -r _ golden_name _ <<< "$entry"
    [ -f "$FIXTURES_DIR/${golden_name}.bin" ] && echo "    ${golden_name}.bin"
done
for cpp_file in golden_cabsim_cpp_short.bin golden_cabsim_cpp_medium.bin \
                 golden_cabsim_cpp_long.bin; do
    [ -f "$FIXTURES_DIR/$cpp_file" ] && echo "    $cpp_file"
done
echo ""
echo "Commit these files so that the Rust golden vector tests work."
