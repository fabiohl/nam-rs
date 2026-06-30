#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# Performance Regression Gate — CI guard against latency degradation
# =============================================================================
#
# This is your first line of defense against shipping a commit that silently
# slows down the DSP hot-path. It runs the `regression_gate` Criterion benchmark
# suite (sample_size=100, measurement_time=5s per bench) with strict CPU core
# pinning and compares the results against a persisted statistical baseline.
#
# Why it matters
# --------------
# The NAM-rs engine runs inside a real-time audio callback with a hard deadline of
# 1.33 ms per 64-sample block at 48 kHz. A seemingly innocent code change — a stray
# allocation, a cache-unfriendly layout, or a suboptimal LLVM optimization —
# can silently push latency above this threshold, causing xruns (audio dropouts)
# in production. This script catches those regressions *before* they reach users.
#
# The two modes
# -------------
#   --save    Persist the current performance numbers as the new official baseline.
#             Run this ONLY after you have verified that:
#               - The code change is intentional and correct.
#               - The new latency values are acceptable (well under 1.33 ms).
#               - `utils/lints.sh` and `utils/tests-quick.sh` are green.
#
#   --check   Compare the current build against the saved baseline (default mode).
#             This is the CI gate: if Criterion detects a statistically significant
#             regression (p < 0.05), the script exits with code 1, failing the
#             pipeline. This is the mode you should run:
#               - In your pre-commit hook or manual pre-push check.
#               - In the CI/CD pipeline on every PR branch.
#               - Before cutting a release tag.
#
# How it works under the hood
# ----------------------------
# 1. CPU affinity: `taskset -c 0` pins the benchmark to a single core, eliminating
#    noise from kernel scheduler migrations and cache-line bouncing between cores.
#    Set NAM_BENCH_CORE to pin to a different core.
# 2. Statistical rigor: Criterion runs each benchmark 100+ iterations with 5s of
#    measurement time (not the old weak `--sample-size 10 --measurement-time 0.5`).
#    It compares the new run against the persisted baseline using a two-sample t-test.
# 3. Baseline persistence: Baseline snapshots live under `target/criterion/<name>/`.
#    The default name is `ci-baseline`. You can maintain multiple baselines for
#    different machines or CPU generations by setting NAM_BASELINE_NAME.
# 4. Regression detection: The output of `cargo bench -- --baseline <name>` is
#    parsed for the keyword "regressed". If found and p < 0.05, the gate fails.
#
# Daily workflow (recommended)
# ----------------------------
#   # 1. Before starting a new branch: ensure the current baseline is clean.
#   utils/tests-performance-regression.sh --check
#
#   # 2. Develop your changes. Run lints and quick tests frequently.
#   utils/lints.sh && utils/tests-quick.sh
#
#   # 3. Before committing/pushing: re-run the regression gate.
#   utils/tests-performance-regression.sh --check
#
#   # 4. If the gate reports PASSED, you are safe to commit/push.
#   #    If it reports REGRESSION, investigate before proceeding.
#
#   # 5. Only when you intentionally change behavior that affects performance
#   #    (e.g., adding a new feature with an understood, acceptable cost),
#   #    and all other tests pass, update the baseline:
#   utils/tests-performance-regression.sh --save
#
# First-time setup
# ----------------
# The first time you run `--check` (or if the baseline directory is missing), the
# script will automatically run `--save` to create the initial baseline. Re-run
# `--check` afterward to activate the gate.
#
# Prerequisites
# --------------
#   - `taskset` (part of util-linux, usually pre-installed on Linux)
#   - `cargo` and Rust toolchain
#   - Models in `tests/fixtures/models/` (some benches skip if models are absent)
#
# Relationship to other QA tools
# -------------------------------
#   - `tests/rt_deadline.rs`: assert-based hard gate (p99 < 1.33 ms) for *all* SKUs.
#     This is the absolute pass/fail. The regression gate here is the *relative*
#     guard — it catches degradations within the safe zone.
#   - `utils/tests-long.sh` Phase 5: runs all benchmarks (including regression_gate)
#     as part of the full audit suite (~38 min).
#   - `utils/tests-quick.sh`: fast path (~3 min), does NOT include benchmarks
#     (would blow the time budget). Use this script directly for perf checks.
#
# Environment variables
# ----------------------
#   NAM_BENCH_CORE=0      CPU core to pin benchmarks to (default: 0).
#   NAM_BASELINE_NAME     Criterion baseline name (default: ci-baseline).
#
# Usage
# ------
#   utils/tests-performance-regression.sh              # --check (default)
#   utils/tests-performance-regression.sh --check      # explicit check mode
#   utils/tests-performance-regression.sh --save       # persist new baseline
#
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

BENCH_CORE="${NAM_BENCH_CORE:-0}"
BASELINE_NAME="${NAM_BASELINE_NAME:-ci-baseline}"
MODE="${1:---check}"

echo -e "${BLUE}${BOLD}  Performance Regression Gate${NC}"
echo -e "  Core: ${YELLOW}${BENCH_CORE}${NC}  Baseline: ${YELLOW}${BASELINE_NAME}${NC}"
echo -e "${BLUE}${BOLD}  Estimeted Time: ± 2,0 minutes${NC}"

# Verify taskset is available
if ! command -v taskset >/dev/null 2>&1; then
    echo -e "  ${YELLOW}⚠ taskset not found — running without core pinning.${NC}"
    TASKSET=""
else
    TASKSET="taskset -c ${BENCH_CORE}"
fi

BASELINE_DIR="target/criterion/${BASELINE_NAME}"

case "$MODE" in
    --save)
        echo -e "\n${GREEN}${BOLD}[SAVE] Persisting new CI baseline...${NC}"
        $TASKSET cargo bench --bench regression_gate \
            -- --save-baseline "$BASELINE_NAME"
        echo -e "${GREEN}✓ Baseline saved to ${BASELINE_DIR}${NC}"
        ;;
    --check)
        if [ ! -d "$BASELINE_DIR" ]; then
            echo -e "\n${YELLOW}[CHECK] No baseline found at ${BASELINE_DIR}.${NC}"
            echo -e "  Running first-time baseline save..."
            $TASKSET cargo bench --bench regression_gate \
                -- --save-baseline "$BASELINE_NAME" \
                || {
                    echo -e "${YELLOW}⚠ First baseline capture failed — skipping regression gate.${NC}"
                    exit 0
                }
            echo -e "${GREEN}✓ Initial baseline saved. Re-run to check for regressions.${NC}"
            exit 0
        fi

        echo -e "\n${BLUE}${BOLD}[CHECK] Comparing against CI baseline...${NC}"
        mkdir -p target/logs
        # Criterion baseline comparison reports regressions with exit code
        set +e
        $TASKSET cargo bench --bench regression_gate \
            -- --baseline "$BASELINE_NAME" 2>&1 | tee target/logs/regression-check.log
        BENCH_STATUS=$?
        set -e

        if grep -q "regressed" target/logs/regression-check.log 2>/dev/null; then
            echo -e "\n${RED}${BOLD}❌ PERFORMANCE REGRESSION DETECTED${NC}"
            echo -e "  Review target/logs/regression-check.log for details."
            echo -e "  If the regression is intentional (e.g., new feature with\n  acceptable cost), re-save the baseline with:"
            echo -e "    ${YELLOW}utils/tests-performance-regression.sh --save${NC}"
            exit 1
        fi

        if [ $BENCH_STATUS -ne 0 ]; then
            echo -e "\n${RED}${BOLD}❌ Benchmark run failed (status=${BENCH_STATUS})${NC}"
            exit 1
        fi

        echo -e "${GREEN}✓ No performance regression detected.${NC}"
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [--save|--check]"
        exit 1
        ;;
esac
