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

# ── Centralized freshness gate (F-X4 / Tarefa 3.4) ──────────────────────────
# Unified freshness validator for golden manifest integrity.
# Usage: check_freshness <mode>
#   mode = hard-fail  → RED messages, returns 1 on failure (CI / quick suite)
#   mode = warn-only  → YELLOW messages, returns 0 always (local dev convenience)
# Bypass: NAM_BYPASS_FRESHNESS=1 skips the entire check (returns 0).
# Validates:
#   1. Manifest existence
#   2. EXPECTED golden files missing from disk
#   3. Catalog model↔golden SHA pairs (stale models)
#   4. Standalone fixtures (hash integrity)
#   5. Generator scripts (warn on change)
#   6. Reverse-check: .nam files in models/ not registered in manifest
#   7. Toolchain fingerprint drift (warn-only, always non-blocking)
check_freshness() {
    local mode="${1:-hard-fail}"
    if [ "${NAM_BYPASS_FRESHNESS:-0}" = "1" ]; then
        echo -e "  ${YELLOW}⚠ NAM_BYPASS_FRESHNESS=1 — freshness check skipped${NC}"
        return 0
    fi

    local PREFIX=""
    local FAIL_PREFIX=""
    local STALE_PREFIX=""
    if [ "$mode" = "warn-only" ]; then
        PREFIX="${YELLOW}⚠"
        FAIL_PREFIX="${YELLOW}${BOLD}⚠"
        STALE_PREFIX="${YELLOW}▲"
    else
        PREFIX="${RED}${BOLD}❌"
        FAIL_PREFIX="${RED}${BOLD}❌"
        STALE_PREFIX="${RED}▲"
    fi

    local MANIFEST="tests/fixtures/.golden_manifest.sha256"
    local MODELS_DIR="tests/fixtures/models"
    local FIXTURES_DIR="tests/fixtures"

    if [ ! -f "$MANIFEST" ]; then
        echo -e "${FAIL_PREFIX} Freshness manifest missing: $MANIFEST${NC}"
        echo -e "  ${PREFIX} Run './tests/fixtures/golden_gen_build.sh' to generate goldens and manifest.${NC}"
        [ "$mode" = "hard-fail" ] && return 1
        return 0
    fi

    local STALE_COUNT=0
    local MISSING_COUNT=0
    local GEN_STALE_COUNT=0
    local ORPHAN_COUNT=0
    local SECTION="catalog"
    declare -A REGISTERED_MODELS  # for reverse-check

    while IFS= read -r line; do
        if [[ "$line" =~ ^#.*FIXTURES ]]; then
            SECTION="fixtures"
            continue
        fi
        if [[ "$line" =~ ^#.*GENERATORS ]]; then
            SECTION="generators"
            continue
        fi

        if [[ "$line" =~ ^#\ EXPECTED:\ (.+)$ ]]; then
            local expected_file="${BASH_REMATCH[1]}"
            if [ ! -f "$FIXTURES_DIR/$expected_file" ]; then
                echo -e "  ${STALE_PREFIX} MISSING: $expected_file — expected golden file not found on disk${NC}"
                MISSING_COUNT=$((MISSING_COUNT + 1))
            fi
            continue
        fi

        if [[ "$line" =~ ^#\ MODEL-REGISTRY:\ (.+)$ ]]; then
            REGISTERED_MODELS["${BASH_REMATCH[1]}"]=1
            continue
        fi

        [[ "$line" =~ ^# ]] && continue
        [[ -z "$line" ]] && continue

        if [ "$SECTION" = "fixtures" ] || [ "$SECTION" = "generators" ]; then
            read -r expected_sha file_path <<< "$line"
            if [ "$SECTION" = "fixtures" ]; then
                local fixture_path="$FIXTURES_DIR/$file_path"
                if [ -f "$fixture_path" ]; then
                    local CURRENT_FIXTURE_SHA
                    CURRENT_FIXTURE_SHA=$(sha256sum "$fixture_path" | cut -d' ' -f1)
                    if [ "$CURRENT_FIXTURE_SHA" != "$expected_sha" ]; then
                        echo -e "  ${STALE_PREFIX} STALE: $file_path — fixture hash changed${NC}"
                        STALE_COUNT=$((STALE_COUNT + 1))
                    fi
                else
                    echo -e "  ${STALE_PREFIX} MISSING: $file_path — fixture file not found on disk${NC}"
                    MISSING_COUNT=$((MISSING_COUNT + 1))
                fi
            else
                local gen_path="$file_path"
                if [ -f "$gen_path" ]; then
                    local CURRENT_GEN_SHA
                    CURRENT_GEN_SHA=$(sha256sum "$gen_path" | cut -d' ' -f1)
                    if [ "$CURRENT_GEN_SHA" != "$expected_sha" ]; then
                        echo -e "  ${YELLOW}⚠ GENERATOR CHANGED: $gen_path — fixtures may be stale; re-run golden_gen_build.sh${NC}"
                        GEN_STALE_COUNT=$((GEN_STALE_COUNT + 1))
                    fi
                fi
            fi
            continue
        fi

        # ── Catalog entries (4 fields: model_sha golden_sha model_name golden_name) ──
        read -r expected_model_sha expected_golden_sha nam_file golden_file <<< "$line"
        REGISTERED_MODELS["$nam_file"]=1
        local MODEL_PATH="$MODELS_DIR/$nam_file"
        if [ -f "$MODEL_PATH" ]; then
            local CURRENT_MODEL_SHA
            CURRENT_MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
            if [ "$CURRENT_MODEL_SHA" != "$expected_model_sha" ]; then
                echo -e "  ${STALE_PREFIX} STALE: $nam_file — model modified since golden was generated${NC}"
                STALE_COUNT=$((STALE_COUNT + 1))
            fi
        fi
    done < "$MANIFEST"

    # ── Reverse-check: scan models/ for .nam files not in manifest ──
    for nam_path in "$MODELS_DIR"/*.nam; do
        [ -f "$nam_path" ] || continue
        local nam_name
        nam_name=$(basename "$nam_path")
        if [ -z "${REGISTERED_MODELS[$nam_name]:-}" ]; then
            echo -e "  ${STALE_PREFIX} ORPHAN: $nam_name — model file not registered in freshness manifest${NC}"
            ORPHAN_COUNT=$((ORPHAN_COUNT + 1))
        fi
    done

    local HAD_FAILURE=0
    if [ "$MISSING_COUNT" -gt 0 ]; then
        echo -e "  ${PREFIX} $MISSING_COUNT expected file(s) missing.${NC}"
        echo -e "  ${PREFIX} Run './tests/fixtures/golden_gen_build.sh' to generate missing golden vectors.${NC}"
        HAD_FAILURE=1
    fi
    if [ "$STALE_COUNT" -gt 0 ]; then
        echo -e "  ${PREFIX} $STALE_COUNT file(s) stale.${NC}"
        echo -e "  ${PREFIX} Run './tests/fixtures/golden_gen_build.sh' to regenerate fixtures and manifest.${NC}"
        HAD_FAILURE=1
    fi
    if [ "$ORPHAN_COUNT" -gt 0 ]; then
        echo -e "  ${PREFIX} $ORPHAN_COUNT model(s) not registered in manifest.${NC}"
        echo -e "  ${PREFIX} Add them to the CATALOG in golden_gen_build.sh and regenerate.${NC}"
        HAD_FAILURE=1
    fi

    check_toolchain_fingerprint

    if [ "$HAD_FAILURE" -eq 1 ]; then
        [ "$mode" = "hard-fail" ] && return 1
        return 0
    fi

    echo -e "  ${GREEN}✓ Freshness gate passed (all hashes match, no orphans).${NC}"
    return 0
}

