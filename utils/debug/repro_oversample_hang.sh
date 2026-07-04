#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# repro_oversample_hang.sh — Safety wrapper for BUG-3 hang-reproduction attempts
# (see TODO-sprints.md T0.4 and known-bugs.md).
#
# This is a *debug/investigation* tool, deliberately NOT part of the official
# `utils/tests-*.sh` suite. It never runs unattended CI/nightly checks; it exists
# solely so a human operator (or an AI, under the non-negotiable rules in
# TODO-sprints.md §"Princípios não-negociáveis") can run a single reproduction
# candidate command with:
#   1. Resource isolation (systemd-run --user --scope + cgroup v2 limits), so a
#      runaway process cannot OOM/starve the host or crash the graphical session.
#   2. An aggressive external kill timeout, since the internal cargo/test-harness
#      timeout is known to be unreliable for this bug (known-bugs.md §1.2).
#   3. A durable, timestamped log + exit-code artifact under target/debug-logs/,
#      plus an automatic dmesg tail, so every attempt is fully reproducible and
#      post-mortem-able without relying on scrollback.
#
# Usage:
#   utils/debug/repro_oversample_hang.sh <timeout_seconds> <label> -- <cargo comando completo...>
#
# Example:
#   utils/debug/repro_oversample_hang.sh 15 varA-stable-debug -- \
#     cargo test --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
#       --ignored --nocapture --test-threads=1
#
# Exit codes:
#   0        — command completed normally within the timeout (success or an
#              ordinary test failure/assertion; inspect the log for details).
#   124/137  — HANG or RESOURCE-KILL detected (external `timeout -s KILL` fired,
#              or the cgroup OOM-killed the process). Printed in bold/highlight.
#   2        — usage error (bad arguments); never confuse this with 124/137.
#   other    — whatever the wrapped command itself returned.

set -uo pipefail

# Resolve project root dynamically so `target/debug-logs/` always lands at the
# repository root, regardless of the caller's current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

usage() {
    echo "Usage: $0 <timeout_seconds> <label> -- <cargo comando completo...>" >&2
    echo "Example: $0 15 varA-stable-debug -- cargo test --lib -- <test_name> --ignored --nocapture --test-threads=1" >&2
    exit 2
}

if [ "$#" -lt 4 ]; then
    usage
fi

TIMEOUT_S="$1"
LABEL="$2"
shift 2

if [ "$1" = "--" ]; then
    shift
else
    echo -e "${RED}${BOLD}Erro: esperado '--' antes do comando cargo.${NC}" >&2
    usage
fi

if [ "$#" -eq 0 ]; then
    echo -e "${RED}${BOLD}Erro: nenhum comando cargo fornecido após '--'.${NC}" >&2
    usage
fi

case "$TIMEOUT_S" in
    ''|*[!0-9]*)
        echo -e "${RED}${BOLD}Erro: <timeout_seconds> deve ser um inteiro positivo (recebido: '${TIMEOUT_S}').${NC}" >&2
        usage
        ;;
esac

# Restrict <label> to a safe charset: it becomes part of a filesystem path
# below (target/debug-logs/<label>.log|.exit). Without this check a crafted
# label containing '/' or '..' could traverse outside that directory and
# write/overwrite arbitrary files reachable by the current user.
case "$LABEL" in
    ''|*[!A-Za-z0-9_-]*)
        echo -e "${RED}${BOLD}Erro: <label> deve conter apenas [A-Za-z0-9_-] (recebido: '${LABEL}').${NC}" >&2
        usage
        ;;
esac

if ! command -v systemd-run >/dev/null 2>&1; then
    echo -e "${RED}${BOLD}Erro: systemd-run não encontrado (T0.3 Opção A indisponível neste ambiente).${NC}" >&2
    exit 2
fi

mkdir -p target/debug-logs
LOG="target/debug-logs/${LABEL}.log"
EXIT_FILE="target/debug-logs/${LABEL}.exit"

echo "=== $(date -Is) :: ${LABEL} :: timeout=${TIMEOUT_S}s :: cmd: $* ===" | tee "$LOG"

systemd-run --user --scope --collect \
    -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
    -- timeout -s KILL "$TIMEOUT_S" "$@" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}

echo "$STATUS" >"$EXIT_FILE"

if [ "$STATUS" -eq 124 ] || [ "$STATUS" -eq 137 ]; then
    echo -e "${RED}${BOLD}!!! HANG/RESOURCE-KILL detectado (exit ${STATUS}) !!!${NC}" | tee -a "$LOG"
elif [ "$STATUS" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}--- comando finalizado normalmente (exit 0) ---${NC}" | tee -a "$LOG"
else
    echo -e "${YELLOW}${BOLD}--- comando finalizado com exit ${STATUS} (não é hang/OOM) ---${NC}" | tee -a "$LOG"
fi

echo "--- dmesg tail ---" | tee -a "$LOG"
dmesg -T 2>/dev/null | tail -n 20 | tee -a "$LOG"

exit "$STATUS"
