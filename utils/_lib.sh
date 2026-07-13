# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# _lib.sh — Common bash utilities for nam-rs scripts.
#
# Source with:
#   PHASE_TOTAL=<N>; source "$(dirname "$0")/_lib.sh"
# or for scripts not in utils/:
#   PHASE_TOTAL=<N>; source "$PROJECT_ROOT/utils/_lib.sh"
#
# Then call:
#   phase "Description of the current step"

# ANSI style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PHASE_NUM=0

phase() {
    PHASE_NUM=$((PHASE_NUM + 1))
    echo -e "\n${BLUE}${BOLD}[${PHASE_NUM}/${PHASE_TOTAL:-?}]${NC} $*"
}

# Resolve project root dynamically relative to this helper script
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$LIB_DIR")"

# Automatically enter the project root directory
cd "$PROJECT_DIR"

# ── Toolchain fingerprint check (F-I4 / Tarefa 3.2) ──────────────────────────
# Reads the # TOOLCHAIN: lines from .golden_manifest.sha256 and compares
# against the current toolchain.  Mismatch emits a YELLOW warning but returns
# 0 (does not block the test suite) — the fingerprint is diagnostic, not
# authoritative.  See docs/cpp_parity_map.md §1.3 for drift context.
check_toolchain_fingerprint() {
    local MANIFEST="tests/fixtures/.golden_manifest.sha256"

    if [ ! -f "$MANIFEST" ]; then
        return 0
    fi

    local CXX_NOW
    CXX_NOW=$(${CXX:-g++} --version 2>/dev/null | head -1 || echo "unknown")
    local CMAKE_NOW
    CMAKE_NOW=$(cmake --version 2>/dev/null | head -1 || echo "unknown")
    local GLIBC_NOW
    if GLIBC_NOW=$(ldd --version 2>/dev/null | head -1); then :; else
        GLIBC_NOW=$(getconf GNU_LIBC_VERSION 2>/dev/null || echo "unknown")
    fi
    local OS_NOW
    OS_NOW=$(uname -r 2>/dev/null || echo "unknown")

    local mismatch=0
    while IFS= read -r line; do
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cxx:\ (.*)$ ]]      && local F_CXX="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cmake:\ (.*)$ ]]     && local F_CMAKE="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ glibc:\ (.*)$ ]]     && local F_GLIBC="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ os:\ (.*)$ ]]        && local F_OS="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cxx-flags:\ (.*)$ ]] && local F_FLAGS="${BASH_REMATCH[1]}"
    done < "$MANIFEST"

    if [ -n "$F_CXX" ] && [ "$F_CXX" != "$CXX_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: compiler changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_CXX${NC}"
        echo -e "    ${YELLOW}now:      $CXX_NOW${NC}"
        mismatch=1
    fi
    if [ -n "$F_GLIBC" ] && [ "$F_GLIBC" != "$GLIBC_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: glibc changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_GLIBC${NC}"
        echo -e "    ${YELLOW}now:      $GLIBC_NOW${NC}"
        mismatch=1
    fi
    if [ -n "$F_CMAKE" ] && [ "$F_CMAKE" != "$CMAKE_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: cmake changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_CMAKE${NC}"
        echo -e "    ${YELLOW}now:      $CMAKE_NOW${NC}"
        mismatch=1
    fi
    if [ -n "$F_OS" ] && [ "$F_OS" != "$OS_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: kernel changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_OS${NC}"
        echo -e "    ${YELLOW}now:      $OS_NOW${NC}"
        mismatch=1
    fi

    return 0
}

