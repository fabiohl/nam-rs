#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Unified compiler-grade release build script for nam-rs (PGO + BOLT).
# Compiles the standalone binary and CLAP plugin with Profile-Guided Optimization
# and post-link BOLT binary reordering.
#
# Deliverables:
#   - ~/.local/bin/nam-rs  (PGO + BOLT optimized standalone binary)
#   - ~/.clap/nam-rs.clap  (PGO optimized CLAP plugin)

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}========================================================================${NC}"
echo -e "${BLUE}${BOLD}   nam-rs Unified Release Build & Optimization Pipeline (± 8 minutes)   ${NC}"
echo -e "${BLUE}${BOLD}========================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configurations
PGO_DIR="/tmp/nam-rs-release-pgo"
BOLT_DIR="/tmp/nam-rs-release-bolt"
PROFRAW_DIR="$PGO_DIR/profraw"
MERGED_PROFILE="$PGO_DIR/merged.profdata"
ORIG_RUSTFLAGS="${RUSTFLAGS:-}"

# Isolated target directories to avoid polluting standard compilations
PGO_BUILD_TARGET_DIR="target/pgo-build"
PGO_CLAP_TARGET_DIR="target/pgo-clap"

# Clean temp and build directories
rm -rf "$PGO_DIR" "$BOLT_DIR" "$PGO_BUILD_TARGET_DIR" "$PGO_CLAP_TARGET_DIR"
mkdir -p "$PROFRAW_DIR" "$BOLT_DIR"

export CARGO_TARGET_DIR="$PGO_BUILD_TARGET_DIR"

# Temporarily disable symbol stripping during release/bench compilation so BOLT can optimize
export CARGO_PROFILE_DIST_STRIP="false"
export CARGO_PROFILE_BENCH_STRIP="false"

# Read rustflags from .cargo/config.toml to avoid overriding them when we set RUSTFLAGS env var
CONFIG_RUSTFLAGS=$(python3 -c '
import re
with open(".cargo/config.toml") as f:
    content = f.read()
match = re.search(r"rustflags\s*=\s*\[(.*?)\]", content, re.DOTALL)
if match:
    block = match.group(1)
    flags = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        flag_match = re.search(r"\"([^\"]+)\"", line)
        if flag_match:
            flags.append(flag_match.group(1))
    print(" ".join(flags))
' 2>/dev/null || echo "")

# Deliverable Targets
BIN_INSTALL_DIR="$HOME/.local/bin"
CLAP_INSTALL_DIR="$HOME/.clap"
BIN_TARGET="$BIN_INSTALL_DIR/nam-rs"
CLAP_TARGET="$CLAP_INSTALL_DIR/nam-rs.clap"

# -----------------------------------------------------------------------------
# PHASE 1: Environment & Dependency Verification
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Phase 1/5] Verifying dependencies and environment...${NC}"

# Verify core dependencies
for cmd in rustc cargo python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo -e "${RED}Error: '$cmd' is not installed or available in PATH.${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} '$cmd' found."
done

# Ensure we successfully parsed non-empty rustflags from .cargo/config.toml
if [ -z "${CONFIG_RUSTFLAGS:-}" ]; then
    echo -e "${RED}Error: Could not extract rustflags from .cargo/config.toml or they are empty!${NC}"
    echo -e "${YELLOW}The release build requires optimizations like '-Ctarget-cpu=x86-64-v3'.${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} rustflags from config.toml verified: ${BOLD}$CONFIG_RUSTFLAGS${NC}"

# Find llvm-profdata from Rustup toolchain
RUST_SYSROOT="$(rustc --print sysroot)"
RUST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
LLVM_PROFDATA="$RUST_SYSROOT/lib/rustlib/$RUST_TARGET/bin/llvm-profdata"
if [ ! -x "$LLVM_PROFDATA" ]; then
    echo -e "${RED}Error: llvm-profdata not found at $LLVM_PROFDATA${NC}"
    echo -e "${YELLOW}Install LLVM tools via rustup:${NC}"
    echo -e "  rustup component add llvm-tools-preview"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} llvm-profdata found: $LLVM_PROFDATA"

# Find LLVM BOLT
LLVM_BOLT=""
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

if [ -n "$LLVM_BOLT" ]; then
    echo -e "  ${GREEN}✓${NC} llvm-bolt found: $LLVM_BOLT"
    PERF2BOLT="$(dirname "$LLVM_BOLT")/perf2bolt"
    if [ ! -x "$PERF2BOLT" ]; then
        PERF2BOLT="perf2bolt"
    fi
else
    echo -e "${YELLOW}Warning: llvm-bolt was not found. The build will continue with PGO only.${NC}"
    echo -e "${YELLOW}To enable BOLT, install: sudo apt install llvm-22-tools${NC}"
fi

# Find perf and check paranoid level
ORIG_PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
PARANOID_MODIFIED=false

cleanup() {
    if [ "$PARANOID_MODIFIED" = "true" ]; then
        echo -e "\nRestoring kernel.perf_event_paranoid to $ORIG_PARANOID..."
        sudo sysctl -q -w kernel.perf_event_paranoid="$ORIG_PARANOID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ "$ORIG_PARANOID" -gt 1 ]; then
    echo -e "  kernel.perf_event_paranoid is $ORIG_PARANOID. Attempting to set to 1..."
    if command -v sudo &>/dev/null; then
        if sudo -n sysctl -w kernel.perf_event_paranoid=1 &>/dev/null; then
            sudo sysctl -w kernel.perf_event_paranoid=1
            PARANOID_MODIFIED=true
            echo -e "  ${GREEN}✓${NC} paranoid level set to 1."
        else
            echo -e "${YELLOW}Warning: Passwordless sudo not available. Trying interactive sudo...${NC}"
            if [ -t 0 ]; then
                if sudo sysctl -w kernel.perf_event_paranoid=1; then
                    PARANOID_MODIFIED=true
                    echo -e "  ${GREEN}✓${NC} paranoid level set to 1."
                else
                    echo -e "${YELLOW}Warning: Failed to set paranoid level to 1. BOLT profiling might be skipped.${NC}"
                fi
            else
                echo -e "${YELLOW}Warning: Non-interactive shell, cannot prompt for sudo password. BOLT profiling might be skipped.${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}Warning: 'sudo' command not found. BOLT profiling might be skipped.${NC}"
    fi
fi

HAS_PERF=false
if command -v perf &>/dev/null; then
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
    if [ "$PARANOID" -le 1 ]; then
        HAS_PERF=true
        echo -e "  ${GREEN}✓${NC} perf is available (paranoid level: $PARANOID)"
    else
        echo -e "${YELLOW}Warning: perf is installed but kernel.perf_event_paranoid=$PARANOID (>1).${NC}"
        echo -e "${YELLOW}BOLT requires paranoid <= 1 for unprivileged root sampling.${NC}"
        echo -e "${YELLOW}Run: sudo sysctl -w kernel.perf_event_paranoid=1${NC}"
    fi
else
    echo -e "${YELLOW}Warning: perf not found. The build will continue with PGO only.${NC}"
fi

    # -----------------------------------------------------------------------------
    # PHASE 2: Profile-Guided Optimization (PGO) - Profiling Workload
    # -----------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[Phase 2/5] Generating PGO profiles through benchmarks...${NC}"

    # Compile and run with profile-generation (incorporating config.toml flags)
    export RUSTFLAGS="$CONFIG_RUSTFLAGS $ORIG_RUSTFLAGS -Cprofile-generate=$PROFRAW_DIR"
    echo -e "  Using RUSTFLAGS: ${BOLD}$RUSTFLAGS${NC}"

    echo -e "  Compiling and running real-world PGO profiling workload..."
    cargo run --profile dist --features "clap-plugin,testing" --bin pgo_profiling_workload

    PROFRAW_COUNT=$(find "$PROFRAW_DIR" -name "*.profraw" 2>/dev/null | wc -l)
    if [ "$PROFRAW_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: No .profraw profile files were generated in $PROFRAW_DIR!${NC}"
        exit 1
    fi

    echo -e "  ${GREEN}✓${NC} Collected $PROFRAW_COUNT .profraw profiles. Merging..."
    "$LLVM_PROFDATA" merge -sparse -o "$MERGED_PROFILE" "$PROFRAW_DIR"/*.profraw
    echo -e "  ${GREEN}✓${NC} Merged profile generated at: $MERGED_PROFILE ($(du -h "$MERGED_PROFILE" | cut -f1))"

    # Clean raw profiles
    rm -rf "$PROFRAW_DIR"

    # -----------------------------------------------------------------------------
    # PHASE 3: Compile PGO-Optimized Standalone and CLAP plugin
    # -----------------------------------------------------------------------------
    echo -e "\n${BLUE}${BOLD}[Phase 3/5] Compiling PGO-optimized binaries...${NC}"

    export RUSTFLAGS="$CONFIG_RUSTFLAGS $ORIG_RUSTFLAGS -Cprofile-use=$MERGED_PROFILE"
    echo -e "  Using RUSTFLAGS: ${BOLD}$RUSTFLAGS${NC}"

    echo -e "  Building standalone executable..."
    # Pass relocations flag to emit relocation symbols required for BOLT
    RUSTFLAGS="$RUSTFLAGS -Clink-arg=-Wl,-q" cargo build --profile dist --features "standalone,pgo" --bin nam-rs

    echo -e "  Building CLAP plugin..."
    CLAP_RUSTFLAGS="$RUSTFLAGS -Clink-arg=-Wl,-soname,nam-rs.clap"
    echo -e "  Using RUSTFLAGS (CLAP): ${BOLD}$CLAP_RUSTFLAGS${NC}"
    RUSTFLAGS="$CLAP_RUSTFLAGS" cargo build --profile dist --target-dir "$PGO_CLAP_TARGET_DIR" --no-default-features --features "clap-plugin,pgo" --lib

    # Confirm binaries compiled
    if [ ! -f "$PGO_BUILD_TARGET_DIR/dist/nam-rs" ]; then
        echo -e "${RED}Error: Failed to find compiled standalone binary at $PGO_BUILD_TARGET_DIR/dist/nam-rs${NC}"
        exit 1
    fi
    if [ ! -f "$PGO_CLAP_TARGET_DIR/dist/libnam_rs.so" ]; then
        echo -e "${RED}Error: Failed to find compiled CLAP plugin library at $PGO_CLAP_TARGET_DIR/dist/libnam_rs.so${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓${NC} PGO compilation completed successfully."

# -----------------------------------------------------------------------------
# PHASE 4: Binary Optimization and Layout Tool (BOLT) Post-Link
# -----------------------------------------------------------------------------
BOLT_APPLIED=false
if [ -n "$LLVM_BOLT" ] && [ "$HAS_PERF" = true ]; then
    echo -e "\n${BLUE}${BOLD}[Phase 4/5] Applying BOLT post-link optimization to standalone binary...${NC}"

    # Verify if PipeWire is active for real-time profiling
    PW_RUNNING=false
    if command -v pw-cli &>/dev/null && pw-cli info &>/dev/null 2>&1; then
        PW_RUNNING=true
    fi

    # Find a representative .nam model for profiling
    MODEL_FILE=""
    for candidate in \
        "tests/fixtures/models/BossWN-standard.nam" \
        "tests/fixtures/models/BossWN-feather.nam" \
        "tests/fixtures/models/wavenet_a1_standard.nam" \
        "tests/fixtures/models/BossLSTM-1x16.nam"; do
        if [ -f "$candidate" ]; then
            MODEL_FILE="$candidate"
            break
        fi
    done

    if [ "$PW_RUNNING" = true ] && [ -n "$MODEL_FILE" ]; then
        echo -e "  PipeWire detected! Starting active profiling of standalone executable..."

        # Generate a 3-second test sine wave
        TEST_WAV="$BOLT_DIR/test_signal.wav"
        if [ ! -f "$TEST_WAV" ]; then
            python3 -c "
import wave, struct, math
rate = 48000
duration = 3
n = rate * duration
with wave.open('$TEST_WAV', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    for i in range(n):
        val = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / rate))
        w.writeframes(struct.pack('<h', val))
" &>/dev/null || true
        fi

        # Start standalone in background using the correct model path argument flag and disabling the gate
        NAM_DISABLE_GATE=1 "$PGO_BUILD_TARGET_DIR/dist/nam-rs" -m "$MODEL_FILE" -b 64 &
        NAM_PID=$!
        sleep 1.0

        if kill -0 $NAM_PID 2>/dev/null; then
            # Query the target Node name directly (avoiding fragile list/grep ID matching)
            NAM_NODE="NAM-rs-input"

            # Check if LBR branch stack sampling is supported
            USE_LBR=false
            if perf record -e cycles:u -j any,u -F 99 -o /dev/null -- true &>/dev/null; then
                USE_LBR=true
            fi

            if [ "$USE_LBR" = "true" ]; then
                PERF_ARGS=(-F 99 -e cycles:u -j any,u -p "$NAM_PID" -o "$BOLT_DIR/perf.data")
            else
                # Fallback to high-frequency instruction sampling if LBR is not supported by PMU
                echo -e "  ${YELLOW}LBR branch stack sampling is not supported by hardware/hypervisor PMU.${NC}"
                echo -e "  ${YELLOW}Falling back to high-frequency (4000 Hz) instruction sampling.${NC}"
                PERF_ARGS=(-F 4000 -e cycles:u -p "$NAM_PID" -o "$BOLT_DIR/perf.data")
            fi

            echo -e "    Recording CPU cycle events via PipeWire..."
            if [ -f "$TEST_WAV" ]; then
                pw-play --target="$NAM_NODE" "$TEST_WAV" &
                PLAY_PID=$!
                perf record "${PERF_ARGS[@]}" -- sleep 3 &>/dev/null || true
                kill $PLAY_PID 2>/dev/null || true
                wait $PLAY_PID 2>/dev/null || true
            else
                perf record "${PERF_ARGS[@]}" -- sleep 3 &>/dev/null || true
            fi

            kill $NAM_PID 2>/dev/null || true
            wait $NAM_PID 2>/dev/null || true

            # Generate AI-ready assembly hotspot report
            if [ -f "$BOLT_DIR/perf.data" ] && [ -s "$BOLT_DIR/perf.data" ]; then
                echo -e "  Generating AI-ready assembly hotspot report in target/dsp_hotpath.asm..."
                mkdir -p target
                perf annotate --stdio -i "$BOLT_DIR/perf.data" > "target/dsp_hotpath.asm" 2>/dev/null || true
            fi

            # Convert profile and optimize standalone binary
            NAM_RS_BIN="$PGO_BUILD_TARGET_DIR/dist/nam-rs"
            if [ -f "$BOLT_DIR/perf.data" ] && [ -s "$BOLT_DIR/perf.data" ]; then
                echo -e "  Converting profile with perf2bolt..."

                # Configure flags based on LBR support
                PERF2BOLT_FLAGS=()
                if [ "$USE_LBR" = "false" ]; then
                    PERF2BOLT_FLAGS+=("--basic-events")
                fi

                if "$PERF2BOLT" "${PERF2BOLT_FLAGS[@]}" -p "$BOLT_DIR/perf.data" "$NAM_RS_BIN" -o "$BOLT_DIR/perf.fdata" --ignore-build-id > "$BOLT_DIR/perf2bolt.log" 2>&1; then
                    echo -e "  Optimizing binary with llvm-bolt..."
                    if "$LLVM_BOLT" "$NAM_RS_BIN" \
                        -o "$PGO_BUILD_TARGET_DIR/dist/nam-rs.bolt" \
                        -data "$BOLT_DIR/perf.fdata" \
                        --reorder-blocks=ext-tsp \
                        --reorder-functions=hfsort \
                        --split-functions \
                        --split-all-cold \
                        --relocs \
                        --no-huge-pages \
                        --lite > "$BOLT_DIR/llvm-bolt.log" 2>&1; then
                        BOLT_APPLIED=true
                        echo -e "  ${GREEN}✓${NC} BOLT applied successfully."
                    else
                        echo -e "${YELLOW}  Warning: llvm-bolt command failed. Reverting to standard PGO binary.${NC}"
                        if [ -f "$BOLT_DIR/llvm-bolt.log" ]; then
                            echo -e "${RED}llvm-bolt error details:${NC}"
                            cat "$BOLT_DIR/llvm-bolt.log"
                        fi
                    fi
                else
                    echo -e "${YELLOW}  Warning: perf2bolt failed to convert data. Reverting to standard PGO binary.${NC}"
                    if [ -f "$BOLT_DIR/perf2bolt.log" ]; then
                        echo -e "${RED}perf2bolt error details:${NC}"
                        cat "$BOLT_DIR/perf2bolt.log"
                    fi
                fi
            else
                echo -e "${YELLOW}  Warning: No perf record data collected. Reverting to standard PGO binary.${NC}"
            fi
        else
            echo -e "${YELLOW}  Warning: nam-rs failed to start under PipeWire. Reverting to standard PGO binary.${NC}"
        fi
    else
        echo -e "${YELLOW}  Warning: PipeWire is not running or no .nam model was found. Reverting to standard PGO binary (no BOLT).${NC}"
    fi
else
    echo -e "\n${YELLOW}[Phase 4/5] Skipping BOLT (llvm-bolt or perf not available/configured).${NC}"
fi

# -----------------------------------------------------------------------------
# PHASE 5: Deliverables Installation & Verification
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Phase 5/5] Installing and validating artifacts...${NC}"

# Target directories creation
mkdir -p "$BIN_INSTALL_DIR"
mkdir -p "$CLAP_INSTALL_DIR"

# Deliver standalone binary
rm -f "$BIN_TARGET"
if [ "$BOLT_APPLIED" = true ] && [ -f "$PGO_BUILD_TARGET_DIR/dist/nam-rs.bolt" ]; then
    cp "$PGO_BUILD_TARGET_DIR/dist/nam-rs.bolt" "$BIN_TARGET"
    strip --strip-all "$BIN_TARGET"
    echo -e "  Installed executable (PGO + BOLT): $BIN_TARGET"
else
    cp "$PGO_BUILD_TARGET_DIR/dist/nam-rs" "$BIN_TARGET"
    strip --strip-all "$BIN_TARGET"
    echo -e "  Installed executable (PGO only): $BIN_TARGET"
fi
chmod +x "$BIN_TARGET"

# Deliver CLAP plugin
rm -f "$CLAP_TARGET"
cp "$PGO_CLAP_TARGET_DIR/dist/libnam_rs.so" "$CLAP_TARGET"
strip --strip-unneeded "$CLAP_TARGET"
echo -e "  Installed CLAP plugin (PGO): $CLAP_TARGET"

# Cleanup temp files
rm -rf "$PGO_DIR" "$BOLT_DIR"

if [ -f "target/dsp_hotpath.asm" ]; then
    echo -e "\n${YELLOW}${BOLD}💡 AI-Ready Assembly Hotspots generated at:${NC} ${BOLD}target/dsp_hotpath.asm${NC}"
    echo -e "   You can feed this file directly into an AI along with the following prompt suggestion:"
    echo -e "   ------------------------------------------------------------------------"
    echo -e "   \"Analyze this compiled x86_64-v3 assembly report (perf annotate) for my"
    echo -e "   Rust DSP function. Identify compiler-generated inefficiencies like bounds"
    echo -e "   checking, registers spilling, or missing SIMD auto-vectorization, and"
    echo -e "   suggest code restructuring in Rust to optimize performance.\""
    echo -e "   ------------------------------------------------------------------------"
fi

echo -e "${GREEN}${BOLD}==============================================================${NC}"
echo -e "${GREEN}${BOLD}   Pipeline completed! Artifacts ready for distribution.   ${NC}"
echo -e "${GREEN}${BOLD}==============================================================${NC}"
ls -lath "$BIN_TARGET" "$CLAP_TARGET"
if [ -f "target/dsp_hotpath.asm" ]; then
    ls -lath "target/dsp_hotpath.asm"
fi
echo -e "${GREEN}${BOLD}================================================================${NC}"
