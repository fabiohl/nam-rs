#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# golden_gen_build.sh — Builds the NeuralAmpModelerCore render tool, clones
# NeuralAmpModelerPlugin (C++ IR reference), and generates all golden vectors.
#
# Canonical reference: NeuralAmpModelerCore v0.5.3 (tag), pinned at commit
# 9c7b185de346fe0725dea537bcee4bc38b5bb6d6. All goldens (A1/LSTM/WaveNet/A2)
# are rendered from this single commit.
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
NAM_PLUGIN_DIR="$SCRIPT_DIR/NeuralAmpModelerPlugin"
BUILD_DIR="$PROJECT_ROOT/build/namcore_render"
MODELS_DIR="$SCRIPT_DIR/models"
FIXTURES_DIR="$SCRIPT_DIR"

# Pinned upstream commits for reproducibility.
# Update these when regenerating goldens with a newer upstream version.
NAM_CORE_COMMIT="9c7b185de346fe0725dea537bcee4bc38b5bb6d6" # v0.5.3 (canonical)
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
# Verify NeuralAmpModelerPlugin and dependencies
# =============================================================================
echo ""
echo "[1/6] Verifying NeuralAmpModelerPlugin (C++ IR reference)..."
if [ ! -d "$NAM_PLUGIN_DIR" ]; then
    echo "ERROR: NeuralAmpModelerPlugin not found at $NAM_PLUGIN_DIR."
    echo "Please run './utils/mod-update.sh' to download and setup dependencies."
    exit 1
fi

CURRENT_PLUGIN_SHA=$(cd "$NAM_PLUGIN_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_PLUGIN_SHA" != "$NAM_PLUGIN_COMMIT" ]; then
    echo "ERROR: NeuralAmpModelerPlugin version mismatch (installed: $CURRENT_PLUGIN_SHA, expected: $NAM_PLUGIN_COMMIT)."
    echo "Please run './utils/mod-update.sh' to synchronize dependencies."
    exit 1
fi

AUDIO_DSP_TOOLS_DIR="$NAM_PLUGIN_DIR/AudioDSPTools"
if [ ! -f "$AUDIO_DSP_TOOLS_DIR/dsp/ImpulseResponse.cpp" ] || [ ! -d "$AUDIO_DSP_TOOLS_DIR/Dependencies/eigen/Eigen" ]; then
    echo "ERROR: Submodules for NeuralAmpModelerPlugin are missing."
    echo "Please run './utils/mod-update.sh' to initialize submodules."
    exit 1
fi
echo "  NeuralAmpModelerPlugin verified (pinned commit and submodules present)"

# =============================================================================
# Verify NeuralAmpModelerCore (standard)
# =============================================================================
echo ""
echo "[1b/6] Verifying NeuralAmpModelerCore..."
if [ ! -d "$NAM_CORE_DIR" ]; then
    echo "ERROR: NeuralAmpModelerCore not found at $NAM_CORE_DIR."
    echo "Please run './utils/mod-update.sh' to download and setup dependencies."
    exit 1
fi

CURRENT_CORE_SHA=$(cd "$NAM_CORE_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_CORE_SHA" != "$NAM_CORE_COMMIT" ]; then
    echo "ERROR: NeuralAmpModelerCore version mismatch (installed: $CURRENT_CORE_SHA, expected: $NAM_CORE_COMMIT)."
    echo "Please run './utils/mod-update.sh' to synchronize dependencies."
    exit 1
fi

for sub in eigen AudioDSPTools; do
    sub_path="$NAM_CORE_DIR/Dependencies/$sub"
    if [ ! -d "$sub_path" ] || [ -z "$(ls -A "$sub_path" 2>/dev/null)" ]; then
        echo "ERROR: Submodule $sub is missing in NeuralAmpModelerCore."
        echo "Please run './utils/mod-update.sh' to initialize submodules."
        exit 1
    fi
done
echo "  NeuralAmpModelerCore verified (pinned commit and submodules present)"

# =============================================================================
# Build render tool (single unified binary at v0.5.3 with A2-fast)
# =============================================================================
echo ""
echo "[2/5] Building render tool..."
BUILD_TYPE="${BUILD_TYPE:-Release}"
RENDER_BIN="$BUILD_DIR/$BUILD_TYPE/render"

if [ -f "$RENDER_BIN" ]; then
    echo "  Render binary already exists: $RENDER_BIN"
else
    echo "  Building render tool (v0.5.3 + A2-fast)..."
    mkdir -p "$BUILD_DIR"
    cmake -S "$NAM_CORE_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_STANDARD=20 \
        -DNAM_ENABLE_A2_FAST=ON \
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
echo "  Render: $RENDER_BIN"

# =============================================================================
# Build Rust tools (gen_stress + wav_to_golden)
# =============================================================================
echo ""
echo "[3/5] Building Rust tools (gen_stress + wav_to_golden)..."

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
echo "[4/5] Generating stress signals..."

STRESS_WAV="$FIXTURES_DIR/stress_signal.wav"
"$GEN_STRESS" --version v1 --output "$STRESS_WAV" 2>&1
echo "  v1: $STRESS_WAV"

echo "  Generating v2 multi-SR stress signals..."
V2_STRESS_WAVS=()
for sr in 44100 48000 88200 96000 192000; do
    v2_wav="$FIXTURES_DIR/stress_signal_v2_${sr}.wav"
    "$GEN_STRESS" --version v2 --sample-rate "$sr" --output "$v2_wav" 2>&1
    echo "    v2 @ ${sr} Hz: $v2_wav"
    V2_STRESS_WAVS+=("$sr:$v2_wav")
done

# =============================================================================
# Run render for each model → WAV output → .golden.bin
# =============================================================================
echo ""
echo "[5/5] Running render for each model (v1)..."

# Models: (.nam file : golden name : label)
MODELS=(
    "BossWN-standard.nam:golden_wavenet_standard:WaveNet Standard"
    "BossWN-lite.nam:golden_wavenet_lite:WaveNet Lite"
    "BossWN-feather.nam:golden_wavenet_feather:WaveNet Feather"
    "BossWN-nano.nam:golden_wavenet_nano:WaveNet Nano"
    "wavenet_a1_standard.nam:golden_wavenet_a1_standard:WaveNet A1 Standard (Official)"
    "wavenet_official.nam:golden_wavenet_official:WaveNet Official (CH=3 free geom)"
    "BossLSTM-1x16.nam:golden_lstm_1x16:LSTM 1×16"
    "BossLSTM-2x8.nam:golden_lstm_2x8:LSTM 2×8"
    "lstm.nam:golden_lstm_official:LSTM Official"
    "wavenet_a2_full.nam:golden_wavenet_a2_full:A2-Full (CH=8)"
    "wavenet_a2_lite.nam:golden_wavenet_a2_lite:A2-Lite (CH=3)"
    "wavenet_condition_dsp.nam:golden_wavenet_condition_dsp:Condition DSP (CH=3, cond=3)"
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
# Generate v2 multi-SR goldens (one per model × sample_rate)
# =============================================================================
echo ""
echo "[5b/5] Generating v2 multi-SR golden vectors..."

# Models eligible for v2 multi-SR (subset exercising all architectures).
#
# Entry format: nam_file:golden_name:label[:skip_srs]
#   skip_srs (optional, comma-separated) — sample rates NOT to generate for this
#   model, kept in sync with the test SR sets in tests/golden_vectors.rs:
#     - LSTM 1x16 / 2x8 skip 192000: the recurrent state's quantization drift at
#       192 kHz makes the golden untestable (tests use SR_EX_192K), so emitting it
#       would only commit dead, never-validated fixtures (~7.3 MB each).
#
# NOTE ON SAMPLE-RATE SKIPS DURING RENDER: models whose .nam declares
# `expected_sample_rate` (e.g. WaveNet Standard CH=16, Official, LSTM Official,
# A2-Full, A2-Lite — all 48 kHz) make the C++ render tool reject other SRs with
# "Input WAV sample rate (X) does not match model expected rate (48000 Hz)". That
# error is EXPECTED and handled by the SKIP path below; those models legitimately
# produce only the 48 kHz golden (tests use SR_48K_ONLY). It is not a failure.
# `wavenet_lite` is intentionally absent (known-divergent, P1 — no v2 coverage).
# P1 RESOLVIDO (T1.2): wavenet_lite re-added with full v2 multi-SR coverage.
V2_MODELS=(
    "BossWN-standard.nam:golden_wavenet_standard:WaveNet Standard (CH=16)"
    "BossWN-lite.nam:golden_wavenet_lite:WaveNet Lite (CH=12)"
    "BossWN-feather.nam:golden_wavenet_feather:WaveNet Feather (CH=8)"
    "BossWN-nano.nam:golden_wavenet_nano:WaveNet Nano (CH=4)"
    "wavenet_a1_standard.nam:golden_wavenet_a1_standard:WaveNet A1 Standard (Official)"
    "wavenet_official.nam:golden_wavenet_official:WaveNet Official (CH=3 free geom)"
    "BossLSTM-1x16.nam:golden_lstm_1x16:LSTM 1×16:192000"
    "BossLSTM-2x8.nam:golden_lstm_2x8:LSTM 2×8:192000"
    "lstm.nam:golden_lstm_official:LSTM Official"
    "wavenet_a2_full.nam:golden_wavenet_a2_full:A2-Full (CH=8)"
    "wavenet_a2_lite.nam:golden_wavenet_a2_lite:A2-Lite (CH=3)"
    "wavenet_condition_dsp.nam:golden_wavenet_condition_dsp:Condition DSP (CH=3, cond=3)"
)

for entry in "${V2_MODELS[@]}"; do
    IFS=':' read -r nam_file golden_name label skip_srs <<< "$entry"
    MODEL_PATH="$MODELS_DIR/$nam_file"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIP v2: $nam_file not found at $MODELS_DIR"
        continue
    fi

    for sr_entry in "${V2_STRESS_WAVS[@]}"; do
        IFS=':' read -r sr v2_wav <<< "$sr_entry"

        # Skip sample rates explicitly excluded for this model (kept in sync with
        # the test SR sets, e.g. LSTM skips 192000 — see V2_MODELS header).
        if [ -n "$skip_srs" ] && [[ ",${skip_srs}," == *",${sr},"* ]]; then
            echo "    $label @ ${sr} Hz (v2)... SKIP (excluded SR for this model)"
            continue
        fi

        v2_golden="$FIXTURES_DIR/${golden_name}_v2_${sr}.bin"
        v2_out_wav="$TEMP_DIR/${golden_name}_v2_${sr}.wav"

        echo "    $label @ ${sr} Hz (v2)..."

        (set +o pipefail; "$RENDER_BIN" "$MODEL_PATH" "$v2_wav" "$v2_out_wav" 2>&1 | tail -1)

        if [ ! -f "$v2_out_wav" ]; then
            echo "    SKIP: render failed for $label @ ${sr} Hz (likely SR mismatch in C++ tool)"
            continue
        fi

        "$WAV_TO_GOLDEN" \
            --input "$v2_out_wav" \
            --reference "$v2_wav" \
            --output "$v2_golden" 2>&1
    done
done

# =============================================================================
# Build and run C++ IR reference (dsp::ImpulseResponse) → golden_cabsim_cpp_*.bin
# =============================================================================
echo ""
echo "[6/6] Building C++ IR reference (dsp::ImpulseResponse)..."

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
echo "[6/6] Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "=== Golden vectors generated successfully ==="
echo "  v1 files at $FIXTURES_DIR/:"
for entry in "${MODELS[@]}"; do
    IFS=':' read -r _ golden_name _ <<< "$entry"
    [ -f "$FIXTURES_DIR/${golden_name}.bin" ] && echo "    ${golden_name}.bin"
done
for cpp_file in golden_cabsim_cpp_short.bin golden_cabsim_cpp_medium.bin \
                 golden_cabsim_cpp_long.bin; do
    [ -f "$FIXTURES_DIR/$cpp_file" ] && echo "    $cpp_file"
done
echo "  v2 multi-SR files at $FIXTURES_DIR/:"
for entry in "${V2_MODELS[@]}"; do
    IFS=':' read -r _ golden_name label __ <<< "$entry"
    count=0
    for sr_entry in "${V2_STRESS_WAVS[@]}"; do
        IFS=':' read -r sr _ <<< "$sr_entry"
        v2_file="$FIXTURES_DIR/${golden_name}_v2_${sr}.bin"
        if [ -f "$v2_file" ]; then
            count=$((count + 1))
        fi
    done
    if [ "$count" -gt 0 ]; then
        echo "    ${golden_name}_v2_*.bin  ($count sample rates) — $label"
    fi
done
# =============================================================================
# Generate freshness manifest (.nam ↔ golden)
# =============================================================================
echo ""
echo "[7/7] Generating freshness manifest..."

MANIFEST="$FIXTURES_DIR/.golden_manifest.sha256"
echo "# Golden freshness manifest — auto-generated by golden_gen_build.sh" > "$MANIFEST"
echo "# Format: sha256(model.nam) sha256(golden.bin) model_filename golden_filename" >> "$MANIFEST"
echo "# Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"

for entry in "${MODELS[@]}"; do
    IFS=':' read -r nam_file golden_name label <<< "$entry"
    MODEL_PATH="$MODELS_DIR/$nam_file"
    GOLDEN_PATH="$FIXTURES_DIR/${golden_name}.bin"
    if [ -f "$MODEL_PATH" ] && [ -f "$GOLDEN_PATH" ]; then
        MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
        GOLDEN_SHA=$(sha256sum "$GOLDEN_PATH" | cut -d' ' -f1)
        echo "$MODEL_SHA $GOLDEN_SHA $nam_file ${golden_name}.bin" >> "$MANIFEST"
    fi
done

echo "  Freshness manifest: $MANIFEST"

echo ""
echo "Commit these files so that the Rust golden vector tests work."
echo "v2 files are large (~18 MB per model across 5 SRs). Git LFS or strategic" 
echo "subset selection is recommended for repo size management."
