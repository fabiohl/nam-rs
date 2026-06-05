#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Profile-Guided Optimization (PGO) build pipeline for nam-rs.
#
# Phases:
#   1-2. Build instrumented binary + collect profiles via representative benchmarks
#   3.   Merge .profraw files with llvm-profdata
#   4.   Rebuild release with -Cprofile-use for PGO-optimized binary
#
# Prerequisites:
#   rustup component add llvm-tools-preview
#   sudo sysctl -w kernel.perf_event_paranoid=1    (if using perf for BOLT later)
#
# Usage: ./utils/build-pgo.sh
#
# Env vars:
#   PGO_DIR           Profile storage directory (default: /tmp/nam-rs-pgo-profiles)
#   SKIP_CLEANUP=1    Keep intermediate artifacts after build

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PGO_DIR="${PGO_DIR:-/tmp/nam-rs-pgo-profiles}"
PROFRAW_DIR="$PGO_DIR/profraw"
MERGED_PROFILE="$PGO_DIR/merged.profdata"

ORIG_RUSTFLAGS="${RUSTFLAGS:-}"

echo "=== nam-rs PGO Build Pipeline ==="
echo "Project dir:  $PROJECT_DIR"
echo "PGO dir:      $PGO_DIR"

# ---- Locate llvm-profdata ----
RUST_SYSROOT="$(rustc --print sysroot)"
RUST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
LLVM_PROFDATA="$RUST_SYSROOT/lib/rustlib/$RUST_TARGET/bin/llvm-profdata"
if [ ! -x "$LLVM_PROFDATA" ]; then
    echo "ERROR: llvm-profdata not found at $LLVM_PROFDATA"
    echo "Install with: rustup component add llvm-tools-preview"
    exit 1
fi
echo "llvm-profdata: $LLVM_PROFDATA ($($LLVM_PROFDATA --version 2>&1 || true))"

# ---- Clean previous profile data ----
echo "Cleaning $PGO_DIR ..."
rm -rf "$PGO_DIR"
mkdir -p "$PROFRAW_DIR"

# ---- Phase 1-2: Build instrumented + collect profiles via benchmarks ----
echo ""
echo "=== Phase 1-2: Build instrumented + collect profiles ==="

export RUSTFLAGS="$ORIG_RUSTFLAGS -Cprofile-generate=$PROFRAW_DIR"

echo "Running inference_bench (all hot paths + long soak benchmarks)..."
cargo bench --features "standalone,long_bench" --bench inference_bench

echo "Running dot_4x_bench (dot product SIMD kernels)..."
cargo bench --features standalone --bench dot_4x_bench

echo ""
PROFRAW_COUNT=$(find "$PROFRAW_DIR" -maxdepth 1 -name "*.profraw" 2>/dev/null | wc -l)
if [ "$PROFRAW_COUNT" -eq 0 ]; then
    echo "ERROR: No .profraw files generated in $PROFRAW_DIR"
    echo "Check that the benchmarks ran successfully and generated profiling data."
    exit 1
fi
echo "Collected $PROFRAW_COUNT .profraw file(s):"
ls -lh "$PROFRAW_DIR"/*.profraw

# ---- Phase 3: Merge profiles ----
echo ""
echo "=== Phase 3: Merge profiles ==="

"$LLVM_PROFDATA" merge -sparse -o "$MERGED_PROFILE" "$PROFRAW_DIR"/*.profraw
echo "Merged profile: $MERGED_PROFILE"
ls -lh "$MERGED_PROFILE"

# ---- Phase 4: Build PGO-optimized release ----
echo ""
echo "=== Phase 4: Build PGO-optimized release binary ==="

export RUSTFLAGS="$ORIG_RUSTFLAGS -Cprofile-use=$MERGED_PROFILE"
cargo build --release --features "standalone,pgo" --bin nam-rs

echo ""
echo "=== PGO build pipeline complete ==="
echo "  PGO-optimized binary: $PROJECT_DIR/target/release/nam-rs"
echo "  Profile data:         $PGO_DIR"
echo ""
echo "Verify with: target/release/nam-rs <model.nam>"
echo "Compare with vanilla release: cargo build --release --features standalone"

# ---- Optional cleanup ----
if [ "${SKIP_CLEANUP:-}" != "1" ]; then
    echo ""
    echo "Cleaning intermediate profile artifacts ($PROFRAW_DIR)..."
    rm -rf "$PROFRAW_DIR"
    echo "Merged profile kept at: $MERGED_PROFILE"
    echo "Set SKIP_CLEANUP=1 to keep raw profiles."
fi
