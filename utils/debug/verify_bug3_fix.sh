#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# verify_bug3_fix.sh — one-command regression check for BUG-3.
#
# BUG-3 was an ELF symbol-interposition bug: some part of the dependency
# graph caused a libm fallback (`log10f`, `atan2f`, `acosf`, ...) to be
# linked into the final binary with GLOBAL visibility, under the same C
# name as the real glibc functions. `ld.so` then resolved calls to those
# names back to a local trampoline instead of `libm.so.6`, forming a
# self-referential `trampoline -> PLT -> GOT -> trampoline` infinite loop
# (zero computation, zero syscalls — a permanent hang). See
# docs/postmortem-libm-symbol-interposition.md for the full, GDB-verified
# root-cause analysis and the lessons learned from diagnosing it.
#
# Fix: `.cargo/hide-libm-shadow.map` (a linker version script) + `build.rs`
# force every standard libm C symbol name to `local` binding in our own
# binaries, so a same-named local fallback can never again win symbol
# interposition over the real dynamic library — regardless of which
# dependency introduces it, now or in the future.
#
# This script is the authoritative, one-command way to confirm the fix is
# still in place. It does NOT rely on static symbol-presence heuristics
# (e.g. `nm -C | grep compiler_builtins::math`) — that exact approach
# produced two false "all clear" conclusions during the original
# investigation, because `#[no_mangle]` C-ABI symbols have no Rust path
# for `nm -C` to demangle, and because the *type* of an ELF relocation
# does not prove what its *runtime value* actually is. The only method
# proven reliable is to actually run the code path that used to hang,
# under a hard external timeout, and confirm it completes.
#
# Usage:
#   utils/debug/verify_bug3_fix.sh
#
# Exit code: 0 if the fix is confirmed in place, non-zero otherwise.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

TEST_PATH="dsp::oversample::oversample_test::test_x2_aliasing_rejection"

echo -e "${BOLD}=== BUG-3 regression check ===${NC}"
echo "Building --release --lib (no-run)..."
if ! cargo test --release --lib --no-run -- --exact "$TEST_PATH" >/tmp/bug3_verify_build.log 2>&1; then
    echo -e "${RED}${BOLD}FAIL: build itself failed. See /tmp/bug3_verify_build.log${NC}"
    exit 2
fi

echo "Running the previously-hanging test under a 20s hard timeout..."
WRAPPER_LOG="target/debug-logs/bug3-verify.log"
if utils/debug/repro_oversample_hang.sh 20 bug3-verify -- \
    cargo test --release --lib -- "$TEST_PATH" --exact --nocapture --test-threads=1 \
    >/tmp/bug3_verify_run.log 2>&1; then
    if grep -q "^test result: ok" "$WRAPPER_LOG" 2>/dev/null; then
        echo -e "${GREEN}${BOLD}PASS: test completed normally (no hang), 'test result: ok'.${NC}"
        echo -e "${GREEN}${BOLD}BUG-3 fix confirmed in place.${NC}"
        exit 0
    else
        echo -e "${RED}${BOLD}FAIL: process exited 0 but no 'test result: ok' found in ${WRAPPER_LOG} — inconclusive, treat as failure.${NC}"
        cat "$WRAPPER_LOG" 2>/dev/null
        exit 1
    fi
else
    echo -e "${RED}${BOLD}FAIL: HANG or non-zero exit detected — BUG-3 has regressed.${NC}"
    cat "$WRAPPER_LOG" 2>/dev/null
    exit 1
fi
