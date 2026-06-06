#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# BOLT (Binary Optimization and Layout Tool) post-link optimization pipeline for nam-rs.
#
# LLVM BOLT reorders basic blocks in the linked binary so that hot paths are
# sequential, improving L1 instruction cache utilization. Combined with PGO,
# expects 3–8% additional latency reduction.
#
# Phases:
#   1. Build PGO-optimized binary (reuses build-pgo.sh if available)
#   2. Build benchmarks + collect perf profile on representative DSP workload
#   3. Convert perf.data with perf2bolt
#   4. Apply BOLT to benchmark binary (for validation)
#   5. Validate: perf stat L1i miss rate + benchmark timing comparison
#   6. Profile nam-rs via PipeWire → BOLT → nam-rs.bolt (release artifact)
#
# Prerequisites:
#   rustup component add llvm-tools-preview
#   sudo sysctl -w kernel.perf_event_paranoid=1
#   sudo apt install linux-tools-generic linux-tools-$(uname -r)
#   sudo apt install llvm-22-tools        (provides llvm-bolt + perf2bolt)
#
# Usage: ./utils/build-bolt.sh
#
# Env vars:
#   BOLT_DIR          Work directory for BOLT artifacts (default: /tmp/nam-rs-bolt)
#   SKIP_PGO=1        Skip PGO build phase (use existing target/release/nam-rs)
#   SKIP_PW=1         Skip PipeWire-based profiling of nam-rs binary
#   SKIP_CLEANUP=1    Keep intermediate artifacts after build
#   PGO_DIR           PGO profile directory (passed to build-pgo.sh)

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

BOLT_DIR="${BOLT_DIR:-/tmp/nam-rs-bolt}"
PERF_DATA="$BOLT_DIR/perf.data"
PERF_FDATA="$BOLT_DIR/perf.fdata"
PERF_DATA_NAM="$BOLT_DIR/perf-nam.data"
PERF_FDATA_NAM="$BOLT_DIR/perf-nam.fdata"
VALIDATION_DIR="$BOLT_DIR/validation"

ORIG_RUSTFLAGS="${RUSTFLAGS:-}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        nam-rs BOLT Post-Link Optimization Pipeline          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Project dir:  $PROJECT_DIR"
echo "BOLT dir:     $BOLT_DIR"
echo ""

# ---- Locate LLVM tools ----
RUST_SYSROOT="$(rustc --print sysroot)"
RUST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
LLVM_BIN_DIR="$RUST_SYSROOT/lib/rustlib/$RUST_TARGET/bin"

# llvm-bolt: prefer system package, fall back to rustup
LLVM_BOLT=""
PERF2BOLT=""
for candidate in \
    /usr/lib/llvm-22/bin/llvm-bolt \
    /usr/lib/llvm-21/bin/llvm-bolt \
    /usr/lib/llvm-20/bin/llvm-bolt \
    /usr/lib/llvm-19/bin/llvm-bolt \
    /usr/lib/llvm-18/bin/llvm-bolt \
    /usr/bin/llvm-bolt-22 \
    /usr/bin/llvm-bolt-21 \
    /usr/bin/llvm-bolt; do
    if [ -x "$candidate" ]; then
        LLVM_BOLT="$candidate"
        break
    fi
done
if [ -z "$LLVM_BOLT" ]; then
    echo "ERROR: llvm-bolt not found."
    echo "Install with: sudo apt install llvm-22-tools"
    echo "Or via rustup: rustup component add llvm-tools-preview"
    echo "(Note: rustup's llvm-tools-preview may not include llvm-bolt; use system package)"
    exit 1
fi

# perf2bolt usually lives next to llvm-bolt
PERF2BOLT="$(dirname "$LLVM_BOLT")/perf2bolt"
if [ ! -x "$PERF2BOLT" ]; then
    PERF2BOLT="$LLVM_BIN_DIR/perf2bolt"
fi
if [ ! -x "$PERF2BOLT" ]; then
    echo "ERROR: perf2bolt not found (expected near llvm-bolt)."
    exit 1
fi

# perf
if ! command -v perf &>/dev/null; then
    echo "ERROR: perf not found. Install with: sudo apt install linux-tools-generic"
    exit 1
fi

# Check perf_event_paranoid
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
echo "perf_event_paranoid: $PARANOID"
if [ "$PARANOID" != "0" ] && [ "$PARANOID" != "1" ] && [ "$PARANOID" != "-1" ] && [ "$PARANOID" != "unknown" ]; then
    if [ "$PARANOID" -gt 1 ] 2>/dev/null; then
        echo "WARNING: perf_event_paranoid=$PARANOID (>1). Perf may not capture enough data."
        echo "  Run: sudo sysctl -w kernel.perf_event_paranoid=1"
    fi
fi

echo "llvm-bolt:  $LLVM_BOLT ($($LLVM_BOLT --version 2>&1 | head -1))"
echo "perf2bolt:  $PERF2BOLT ($($PERF2BOLT --version 2>&1 | head -1))"
echo "perf:       $(perf --version 2>&1)"

# ---- Clean previous BOLT artifacts ----
echo ""
echo "=== Phase 0: Setup ==="
rm -rf "$BOLT_DIR"
mkdir -p "$BOLT_DIR" "$VALIDATION_DIR"

# ---- Phase 1: Build PGO-optimized binary ----
echo ""
echo "=== Phase 1: Build PGO-optimized binary ==="

if [ "${SKIP_PGO:-}" = "1" ]; then
    if [ ! -f "$PROJECT_DIR/target/release/nam-rs" ]; then
        echo "ERROR: SKIP_PGO=1 but target/release/nam-rs not found."
        exit 1
    fi
    echo "SKIP_PGO=1 — using existing target/release/nam-rs"
else
    # Check if we can reuse an existing PGO build
    PGO_DIR="${PGO_DIR:-/tmp/nam-rs-pgo-profiles}"
    MERGED_PROFILE="$PGO_DIR/merged.profdata"

    if [ -f "$MERGED_PROFILE" ] && [ -f "$PROJECT_DIR/target/release/nam-rs" ]; then
        echo "PGO profile found at $MERGED_PROFILE and binary exists — rebuilding with PGO..."
        export RUSTFLAGS="$ORIG_RUSTFLAGS -Cprofile-use=$MERGED_PROFILE"
        cargo build --release --features "standalone,pgo" --bin nam-rs
    elif [ -x "$SCRIPT_DIR/build-pgo.sh" ]; then
        echo "Running PGO build pipeline ($SCRIPT_DIR/build-pgo.sh)..."
        bash "$SCRIPT_DIR/build-pgo.sh"
    else
        echo "No PGO profile found and build-pgo.sh not available."
        echo "Building release without PGO (BOLT can still help, but effect is smaller)."
        cargo build --release --features standalone
    fi
fi

if [ ! -f "$PROJECT_DIR/target/release/nam-rs" ]; then
    echo "ERROR: target/release/nam-rs not found after build."
    exit 1
fi

NAM_RS_BIN="$PROJECT_DIR/target/release/nam-rs"
echo "PGO-optimized binary: $NAM_RS_BIN"
echo "Size: $(du -h "$NAM_RS_BIN" | cut -f1)"
echo "Build ID: $(readelf -n "$NAM_RS_BIN" 2>/dev/null | grep 'Build ID' | awk '{print $3}')"

# Ensure PGO profile is used for subsequent benchmark builds
PGO_DIR="${PGO_DIR:-/tmp/nam-rs-pgo-profiles}"
MERGED_PROFILE="$PGO_DIR/merged.profdata"
if [ -f "$MERGED_PROFILE" ]; then
    export RUSTFLAGS="$ORIG_RUSTFLAGS -Cprofile-use=$MERGED_PROFILE"
    echo "Using PGO profile: $MERGED_PROFILE"
else
    echo "No merged PGO profile found — benchmarks built without PGO instrumentation."
fi

# ---- Phase 2: Build benchmarks + collect perf profile ----
echo ""
echo "=== Phase 2: Build benchmarks + collect perf profile ==="

# Build benchmark binaries in release mode (they link the same PGO-optimized lib)
echo "Building inference_bench (representative DSP workload)..."
cargo bench --features "standalone,long_bench" --bench inference_bench --no-run

echo "Building dot_4x_bench (SIMD kernel workload)..."
cargo bench --features standalone --bench dot_4x_bench --no-run

# Find the built benchmark binaries (sorted by modification time, newest first)
INFERENCE_BENCH=$(find "$PROJECT_DIR/target/release/deps" -maxdepth 1 -name "inference_bench-*" ! -name "*.d" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
DOT4X_BENCH=$(find "$PROJECT_DIR/target/release/deps" -maxdepth 1 -name "dot_4x_bench-*" ! -name "*.d" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

if [ -z "$INFERENCE_BENCH" ]; then
    # Fallback: older find without -printf support
    INFERENCE_BENCH=$(find "$PROJECT_DIR/target/release/deps" -maxdepth 1 -name "inference_bench-*" ! -name "*.d" -type f 2>/dev/null | head -1)
fi
if [ -z "$DOT4X_BENCH" ]; then
    DOT4X_BENCH=$(find "$PROJECT_DIR/target/release/deps" -maxdepth 1 -name "dot_4x_bench-*" ! -name "*.d" -type f 2>/dev/null | head -1)
fi

if [ -z "$INFERENCE_BENCH" ]; then
    echo "ERROR: Could not find inference_bench binary in target/release/deps/"
    echo "  Ensure: cargo bench --features 'standalone,long_bench' --bench inference_bench --no-run"
    exit 1
fi
if [ -z "$DOT4X_BENCH" ]; then
    echo "ERROR: Could not find dot_4x_bench binary in target/release/deps/"
    echo "  Ensure: cargo bench --features standalone --bench dot_4x_bench --no-run"
    exit 1
fi

echo "Benchmark binaries:"
echo "  inference_bench: $INFERENCE_BENCH ($(du -h "$INFERENCE_BENCH" | cut -f1))"
echo "  dot_4x_bench:    $DOT4X_BENCH ($(du -h "$DOT4X_BENCH" | cut -f1))"

# Collect perf profile on the benchmark binary (exercises all DSP hot paths)
# Try with LBR branch sampling first (much better for BOLT), fall back to basic sampling
echo ""
echo "Collecting perf profile on inference_bench (all hot paths + long soak)..."

PERF_RECORD_ARGS=(-F 99 -e cycles:u -o "$PERF_DATA")

# Check if LBR branch sampling is supported
if perf record -e cycles:u -j any,u -F 99 -o /dev/null -- true 2>/dev/null; then
    echo "LBR branch sampling supported — using -j any,u for richer BOLT profiles"
    PERF_RECORD_ARGS=(-F 99 -e cycles:u -j any,u -o "$PERF_DATA")
else
    echo "LBR not available — using basic cycle sampling"
fi

perf record \
    "${PERF_RECORD_ARGS[@]}" \
    -- "$INFERENCE_BENCH" --bench 2>/dev/null || {
    echo "WARNING: perf record on inference_bench returned non-zero. Profiles may be incomplete."
}

echo "Collecting perf profile on dot_4x_bench (AVX2/AVX-512 dot product kernels)..."
# Build alternative args with different output file
DOT4X_PERF_ARGS=("${PERF_RECORD_ARGS[@]}")
for i in "${!DOT4X_PERF_ARGS[@]}"; do
    if [ "${DOT4X_PERF_ARGS[$i]}" = "-o" ] && [ $((i + 1)) -lt ${#DOT4X_PERF_ARGS[@]} ]; then
        DOT4X_PERF_ARGS[$((i + 1))]="${PERF_DATA}.dot4x"
        break
    fi
done
perf record \
    "${DOT4X_PERF_ARGS[@]}" \
    -- "$DOT4X_BENCH" --bench 2>/dev/null || true

# Also collect on the nam-rs binary through the benchmark (same code paths)
# We use a single merged profile for BOLT

# Check we got samples
SAMPLE_COUNT=0
if [ -f "$PERF_DATA" ]; then
    SAMPLE_COUNT=$(perf report -i "$PERF_DATA" --stdio 2>/dev/null | grep -c "cycles" || echo "0")
fi
echo "Perf samples collected in primary profile: $SAMPLE_COUNT events"

# ---- Phase 3: Convert perf.data with perf2bolt ----
echo ""
echo "=== Phase 3: Convert perf.data with perf2bolt ==="

"$PERF2BOLT" -p "$PERF_DATA" "$INFERENCE_BENCH" -o "$PERF_FDATA" --ignore-build-id 2>&1 || {
    echo "WARNING: perf2bolt conversion had issues. Trying with --itrace..."
    "$PERF2BOLT" -p "$PERF_DATA" --itrace=l64i2000000 "$INFERENCE_BENCH" -o "$PERF_FDATA" --ignore-build-id 2>&1 || {
        echo "WARNING: perf2bolt still failed. Attempting with --basic-events..."
        "$PERF2BOLT" -p "$PERF_DATA" --basic-events "$INFERENCE_BENCH" -o "$PERF_FDATA" --ignore-build-id
    }
}

if [ ! -s "$PERF_FDATA" ]; then
    echo "ERROR: perf2bolt produced empty output. Check perf_event_paranoid settings."
    echo "  Try: sudo sysctl -w kernel.perf_event_paranoid=1"
    exit 1
fi

echo "perf.fdata size: $(du -h "$PERF_FDATA" | cut -f1)"
echo "Profile density: $(wc -l < "$PERF_FDATA") records"

# ---- Phase 4: Apply BOLT to benchmark binary (for validation) ----
echo ""
echo "=== Phase 4: Apply BOLT to benchmark binary ==="

BENCH_BOLT="$VALIDATION_DIR/inference_bench.bolt"
cp "$INFERENCE_BENCH" "$VALIDATION_DIR/inference_bench.orig"

"$LLVM_BOLT" "$INFERENCE_BENCH" \
    -o "$BENCH_BOLT" \
    -data "$PERF_FDATA" \
    --reorder-blocks=cache+ \
    --reorder-functions=hfsort \
    --split-functions \
    --split-all-cold \
    --relocs \
    --lite \
    --dyno-stats

if [ ! -f "$BENCH_BOLT" ]; then
    echo "ERROR: BOLT optimization failed."
    exit 1
fi

echo "BOLTed benchmark binary: $BENCH_BOLT"
echo "Size: $(du -h "$BENCH_BOLT" | cut -f1) (original: $(du -h "$INFERENCE_BENCH" | cut -f1))"

# ---- Phase 5: Validate BOLT effectiveness ----
echo ""
echo "=== Phase 5: Validate BOLT effectiveness ==="

echo ""
echo "--- L1i Cache Miss Rate Comparison (perf stat) ---"
echo "Running original benchmark binary..."
perf stat \
    -e L1-icache-load-misses,L1-icache-loads,instructions,cycles \
    -o "$VALIDATION_DIR/perfstat-orig.txt" \
    -- "$VALIDATION_DIR/inference_bench.orig" \
    WaveNet_Standard_CH16_64samp_48kHz LSTM_2x16_64samp_48kHz 2>/dev/null || true

echo "Running BOLTed benchmark binary..."
perf stat \
    -e L1-icache-load-misses,L1-icache-loads,instructions,cycles \
    -o "$VALIDATION_DIR/perfstat-bolt.txt" \
    -- "$BENCH_BOLT" \
    WaveNet_Standard_CH16_64samp_48kHz LSTM_2x16_64samp_48kHz 2>/dev/null || true

echo ""
echo "Original perf stat:"
cat "$VALIDATION_DIR/perfstat-orig.txt" 2>/dev/null || echo "(no output captured)"
echo ""
echo "BOLTed perf stat:"
cat "$VALIDATION_DIR/perfstat-bolt.txt" 2>/dev/null || echo "(no output captured)"

# Compute L1i miss rate comparison
ORIG_MISSES=$(grep "L1-icache-load-misses" "$VALIDATION_DIR/perfstat-orig.txt" 2>/dev/null | awk '{print $1}' | tr -d ',' || echo "N/A")
BOLT_MISSES=$(grep "L1-icache-load-misses" "$VALIDATION_DIR/perfstat-bolt.txt" 2>/dev/null | awk '{print $1}' | tr -d ',' || echo "N/A")
ORIG_LOADS=$(grep "L1-icache-loads" "$VALIDATION_DIR/perfstat-orig.txt" 2>/dev/null | awk '{print $1}' | tr -d ',' || echo "N/A")
BOLT_LOADS=$(grep "L1-icache-loads" "$VALIDATION_DIR/perfstat-bolt.txt" 2>/dev/null | awk '{print $1}' | tr -d ',' || echo "N/A")

echo ""
echo "L1i Cache Metrics:"
echo "  Original: $ORIG_MISSES misses / $ORIG_LOADS loads"
echo "  BOLTed:   $BOLT_MISSES misses / $BOLT_LOADS loads"

if [ "$ORIG_LOADS" != "N/A" ] && [ "$BOLT_LOADS" != "N/A" ] && [ "$ORIG_LOADS" != "0" ] && [ "$BOLT_LOADS" != "0" ]; then
    if command -v bc &>/dev/null; then
        ORIG_RATE=$(echo "scale=4; $ORIG_MISSES / $ORIG_LOADS * 100" | bc 2>/dev/null || echo "N/A")
        BOLT_RATE=$(echo "scale=4; $BOLT_MISSES / $BOLT_LOADS * 100" | bc 2>/dev/null || echo "N/A")
    else
        ORIG_RATE="N/A"
        BOLT_RATE="N/A"
    fi
    echo "  Original miss rate: ${ORIG_RATE}%"
    echo "  BOLTed miss rate:   ${BOLT_RATE}%"
    if [ "$ORIG_RATE" != "N/A" ] && [ "$BOLT_RATE" != "N/A" ] && [ "$ORIG_RATE" != "0" ]; then
        if command -v bc &>/dev/null; then
            REDUCTION=$(echo "scale=2; (1 - $BOLT_RATE / $ORIG_RATE) * 100" | bc 2>/dev/null || echo "N/A")
        else
            REDUCTION="N/A"
        fi
        echo "  Reduction:          ${REDUCTION}%"
        if [ "$REDUCTION" != "N/A" ]; then
            if (( $(echo "$REDUCTION >= 20" | bc -l 2>/dev/null || echo 0) )); then
                echo "  ✓ Acceptance criteria met: L1i miss rate reduced ≥ 20%"
            else
                echo "  ⚠ Below target (≥20%). Check perf_event_paranoid, try larger profile."
            fi
        fi
    fi
fi

# ---- Phase 6: Profile and BOLT nam-rs binary (release artifact) ----
echo ""
echo "=== Phase 6: Profile and BOLT nam-rs binary (release artifact) ==="

NAM_BOLT="$PROJECT_DIR/target/release/nam-rs.bolt"

if [ "${SKIP_PW:-}" = "1" ]; then
    echo "SKIP_PW=1 — skipping PipeWire-based profiling of nam-rs."
    echo "To produce nam-rs.bolt, run without SKIP_PW=1 or manually profile via:"
    echo "  perf record -F 99 -e cycles:u -o perf-nam.data -- target/release/nam-rs model.nam &"
    echo "  ... (feed audio through PipeWire) ..."
    echo "  perf2bolt -p perf-nam.data target/release/nam-rs -o perf-nam.fdata"
    echo "  llvm-bolt target/release/nam-rs -o target/release/nam-rs.bolt -data perf-nam.fdata --reorder-blocks=cache+ --reorder-functions=hfsort"
else
    # Check if PipeWire is available for profiling
    PW_RUNNING=false
    if command -v pw-cli &>/dev/null && pw-cli info &>/dev/null 2>&1; then
        PW_RUNNING=true
    fi

    if [ "$PW_RUNNING" = true ]; then
        echo "PipeWire detected — profiling nam-rs through audio pipeline..."

        # Find a .nam model for profiling
        MODEL_FILE=""
        for candidate in \
            "$PROJECT_DIR/tests/nam_files/EVH-5150-Lite.nam" \
            "$PROJECT_DIR/tests/nam_files/ChandlerRedd47-Gain34-Standard.nam" \
            "$PROJECT_DIR/tests/nam_files/NEVE1073-Standard.nam" \
            "$PROJECT_DIR/tests/nam_files/MRSH-JM50LD-Crunch2_FAT_CAB.nam"; do
            if [ -f "$candidate" ]; then
                MODEL_FILE="$candidate"
                break
            fi
        done

        if [ -z "$MODEL_FILE" ]; then
            echo "WARNING: No .nam model found in tests/nam_files/. Skipping nam-rs profiling."
        else
            echo "Profiling model: $MODEL_FILE"

            # Generate a short test signal (WAV, 48000 Hz, mono, 10 seconds of 440 Hz sine)
            TEST_WAV="$BOLT_DIR/test_signal.wav"
            if command -v ffmpeg &>/dev/null; then
                ffmpeg -y -f lavfi -i "sine=frequency=440:duration=10" \
                    -ar 48000 -ac 1 -sample_fmt s16 \
                    "$TEST_WAV" 2>/dev/null
            elif command -v python3 &>/dev/null; then
                python3 -c "
import wave, struct, math
rate = 48000
duration = 10
n = rate * duration
with wave.open('$TEST_WAV', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    for i in range(n):
        val = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / rate))
        w.writeframes(struct.pack('<h', val))
" 2>/dev/null
            else
                echo "WARNING: No audio generator found (ffmpeg or python3). Skipping nam-rs profiling."
                PW_RUNNING=false
            fi
        fi

        if [ "$PW_RUNNING" = true ] && [ -f "$TEST_WAV" ]; then
            # Start nam-rs in background
            echo "Starting nam-rs..."
            "$NAM_RS_BIN" "$MODEL_FILE" -b 64 &
            NAM_PID=$!

            # Wait for PipeWire streams to be created
            sleep 2

            if ! kill -0 $NAM_PID 2>/dev/null; then
                echo "ERROR: nam-rs failed to start. Check PipeWire is running and audio device is available."
                PW_RUNNING=false
            else
                # Find nam-rs capture port in PipeWire
                NAM_NODE=""
                for attempt in $(seq 1 5); do
                    NAM_NODE=$(pw-cli list-objects Node 2>/dev/null | grep -B1 -A5 "nam-rs" | grep -oP 'id \K\d+' | head -1)
                    if [ -n "$NAM_NODE" ]; then
                        break
                    fi
                    sleep 1
                done

                echo "nam-rs PID: $NAM_PID"
                echo "nam-rs PipeWire node: ${NAM_NODE:-not found}"

                # Profile nam-rs while playing test audio through it
                echo "Profiling nam-rs with perf record (10s)..."

                # Use same branch-sampling detection as earlier
                NAM_PERF_ARGS=(-F 99 -e cycles:u -p "$NAM_PID" -o "$PERF_DATA_NAM")
                if perf record -e cycles:u -j any,u -F 99 -o /dev/null -- true 2>/dev/null; then
                    NAM_PERF_ARGS=(-F 99 -e cycles:u -j any,u -p "$NAM_PID" -o "$PERF_DATA_NAM")
                fi

                if [ -n "$NAM_NODE" ]; then
                    # Route audio through nam-rs capture stream
                    pw-play --target="$NAM_NODE" "$TEST_WAV" &
                    PLAY_PID=$!
                    sleep 1

                    perf record \
                        "${NAM_PERF_ARGS[@]}" \
                        -- sleep 8 2>/dev/null || true

                    kill $PLAY_PID 2>/dev/null || true
                else
                    # Fallback: just profile nam-rs without specific routing
                    perf record \
                        "${NAM_PERF_ARGS[@]}" \
                        -- sleep 10 2>/dev/null || true
                fi

                # Kill nam-rs gracefully
                kill $NAM_PID 2>/dev/null || true
                wait $NAM_PID 2>/dev/null || true

                echo "nam-rs profiling complete."

                # Convert and apply BOLT
                if [ -f "$PERF_DATA_NAM" ] && [ -s "$PERF_DATA_NAM" ]; then
                    echo "Converting perf data for nam-rs..."
                    "$PERF2BOLT" -p "$PERF_DATA_NAM" "$NAM_RS_BIN" -o "$PERF_FDATA_NAM" --ignore-build-id 2>&1 || {
                        echo "WARNING: perf2bolt for nam-rs had issues. Trying with --itrace..."
                        "$PERF2BOLT" -p "$PERF_DATA_NAM" --itrace=l64i2000000 "$NAM_RS_BIN" -o "$PERF_FDATA_NAM" --ignore-build-id 2>&1 || true
                    }

                    if [ -s "$PERF_FDATA_NAM" ]; then
                        echo "Applying BOLT to nam-rs → nam-rs.bolt..."
                        "$LLVM_BOLT" "$NAM_RS_BIN" \
                            -o "$NAM_BOLT" \
                            -data "$PERF_FDATA_NAM" \
                            --reorder-blocks=cache+ \
                            --reorder-functions=hfsort \
                            --split-functions \
                            --split-all-cold \
                            --relocs \
                            --lite

                        if [ -f "$NAM_BOLT" ]; then
                            echo ""
                            echo "╔══════════════════════════════════════════════════════════════╗"
                            echo "║  BOLT-optimized release binary: $NAM_BOLT"
                            echo "║  Size: $(du -h "$NAM_BOLT" | cut -f1) (original: $(du -h "$NAM_RS_BIN" | cut -f1))"
                            echo "╚══════════════════════════════════════════════════════════════╝"
                        fi
                    else
                        echo "WARNING: Empty perf.fdata for nam-rs. Skipping BOLT application."
                    fi
                fi
            fi
        fi
    else
        echo "PipeWire not available — skipping nam-rs profiling."
        echo ""
        echo "To produce nam-rs.bolt manually:"
        echo "  1. Run nam-rs with a model: target/release/nam-rs model.nam"
        echo "  2. Feed audio through PipeWire to exercise DSP paths"
        echo "  3. perf record -F 99 -e cycles:u -p \$(pgrep nam-rs) -o perf-nam.data -- sleep 30"
        echo "  4. perf2bolt -p perf-nam.data target/release/nam-rs -o perf-nam.fdata"
        echo "  5. llvm-bolt target/release/nam-rs -o target/release/nam-rs.bolt -data perf-nam.fdata --reorder-blocks=cache+ --reorder-functions=hfsort"
    fi
fi

# ---- Phase 7: Summary and cleanup ----
echo ""
echo "=== BOLT pipeline complete ==="
echo ""
echo "Artifacts:"
echo "  PGO binary:       $NAM_RS_BIN"
echo "  BOLT binary:      ${NAM_BOLT:-not produced (set SKIP_PW=0 with PipeWire)}"
echo "  Profile data:     $BOLT_DIR"
echo "  Benchmark BOLTed: $BENCH_BOLT"
echo "  Benchmark orig:   $VALIDATION_DIR/inference_bench.orig"
echo ""

if [ -f "$NAM_BOLT" ]; then
    echo "To use the BOLT-optimized binary:"
    echo "  $NAM_BOLT <model.nam>"
    echo ""
    echo "To distribute as release:"
    echo "  cp $NAM_BOLT target/release/nam-rs    # replace original"
fi

if [ "${SKIP_CLEANUP:-}" != "1" ]; then
    echo ""
    echo "Cleaning intermediate artifacts..."
    rm -f "${PERF_DATA}.dot4x" "$TEST_WAV"
    echo "Set SKIP_CLEANUP=1 to keep all intermediate files."
fi

echo ""
echo "Validation data at: $VALIDATION_DIR"
echo "  perfstat-orig.txt — original benchmark perf stat"
echo "  perfstat-bolt.txt — BOLTed benchmark perf stat"
echo ""
echo "Run full benchmark comparison:"
echo "  cargo bench --features standalone --bench inference_bench"
echo "  cargo bench --features standalone --bench dot_4x_bench"
