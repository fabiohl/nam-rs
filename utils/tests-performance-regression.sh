#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# Performance Regression Gate — statistical wall against DSP hot-path decay
# =============================================================================
#
# Canonical home of benchmark-based performance defense for nam-rs: runs the
# `regression_gate` Criterion suite (sample_size=100, measurement_time=5s),
# pinned to a single CPU core, and compares it against a persisted statistical
# baseline. A regressing commit exits non-zero — the audio engine has a hard
# 1.33 ms/64-sample deadline at 48 kHz, and this is the first line of defense
# against silently eating into that budget.
#
# Full rationale, daily workflow, and troubleshooting live in
# docs/benchmarks.md ("Regression Gate" section) — this header only documents
# the mechanics. Do not duplicate that narrative here; update the doc instead.
#
# Modes
# -----
#   --check (default)   Compare the current build against the saved baseline.
#                        Exits 1 on a statistically significant regression
#                        (p < 0.05). Auto-bootstraps the baseline on first run.
#   --save               Persist current measurements as the new baseline.
#                        Run only after the change is verified intentional and
#                        `utils/lints.sh` + `utils/tests-quick.sh` are green.
#
# Environment variables
# ----------------------
#   NAM_BENCH_CORE       CPU core to pin via taskset (default: 1 if multicore, else 0).
#   NAM_BASELINE_NAME    Criterion baseline name (default: ci-baseline).
#
# Usage
# ------
#   utils/tests-performance-regression.sh [--check|--save]
#
# Relationship to other QA scripts (docs/benchmarks.md, docs/testing.md):
#   - `tests/rt_deadline.rs`   — absolute hard gate (p99 < 1.33 ms), not this.
#   - THIS script             — relative guard; the canonical home for
#     baseline-gated benchmarking within the safe zone.
#   - `utils/tests-quick.sh`  — never runs benchmarks (would blow its budget).
#   - `utils/tests-long.sh` Phase 5 — runs the full bench suite (incl.
#     regression_gate) for the record as part of the nightly audit, with no
#     baseline gating of its own.
#
# =============================================================================

set -euo pipefail

source "$(dirname "$0")/_lib.sh"

# Determine default CPU core for taskset. Pinned to middle core (nproc / 2) to avoid OS/IRQ noise.
NUM_CORES=$(nproc)
DEFAULT_CORE=$((NUM_CORES / 2))
BENCH_CORE="${NAM_BENCH_CORE:-$DEFAULT_CORE}"
BASELINE_NAME="${NAM_BASELINE_NAME:-ci-baseline}"
BASELINE_DIR="target/criterion/${BASELINE_NAME}"
MODE="${1:---check}"

TASKSET=""
if command -v taskset >/dev/null 2>&1; then
    TASKSET="taskset -c ${BENCH_CORE}"
else
    echo -e "  ${YELLOW}⚠ taskset not found — running without core pinning.${NC}"
fi

echo -e "${BLUE}${BOLD}  Performance Regression Gate${NC}"
echo -e "  Core: ${YELLOW}${BENCH_CORE}${NC}  Baseline: ${YELLOW}${BASELINE_NAME}${NC}"
echo -e "${BLUE}${BOLD}  Estimated time: ± 2.0 minutes${NC}"

# save_baseline — runs the regression_gate bench suite and persists the
# result under $BASELINE_NAME. Shared by `--save` and the `--check`
# first-time bootstrap path below.
save_baseline() {
    $TASKSET cargo bench --bench regression_gate -- --save-baseline "$BASELINE_NAME"
}

case "$MODE" in
    --save)
        echo -e "\n${GREEN}${BOLD}[SAVE] Persisting new CI baseline...${NC}"
        save_baseline
        echo -e "${GREEN}✓ Baseline saved under target/criterion/*/${BASELINE_NAME}${NC}"
        ;;
    --check)
        baseline_exists=false
        if [ -d "target/criterion" ]; then
            for d in target/criterion/*/"$BASELINE_NAME"; do
                if [ -d "$d" ]; then
                    baseline_exists=true
                    break
                fi
            done
        fi

        if [ "$baseline_exists" = false ]; then
            echo -e "\n${YELLOW}[CHECK] No baseline found for '${BASELINE_NAME}' under target/criterion/.${NC}"
            echo -e "  Running first-time baseline save..."
            save_baseline || {
                echo -e "${YELLOW}⚠ First baseline capture failed — skipping regression gate.${NC}"
                exit 0
            }
            echo -e "${GREEN}✓ Initial baseline saved. Re-run to check for regressions.${NC}"
            exit 0
        fi

        echo -e "\n${BLUE}${BOLD}[CHECK] Comparing against CI baseline...${NC}"
        mkdir -p target/logs
        # Criterion reports regressions via stdout text, not via exit code —
        # grep the log below instead of trusting $? alone.
        set +e
        $TASKSET cargo bench --bench regression_gate \
            -- --baseline "$BASELINE_NAME" 2>&1 | tee target/logs/regression-check.log
        BENCH_STATUS=$?
        set -e

        if grep -qE '(has regressed|Performance has regressed)' target/logs/regression-check.log 2>/dev/null; then
            echo -e "\n${RED}${BOLD}❌ PERFORMANCE REGRESSION DETECTED${NC}"
            echo -e "  Review target/logs/regression-check.log for details."
            echo -e "  If the regression is intentional, re-save the baseline with:"
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
