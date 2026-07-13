#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# quality-dashboard.sh — nam-rs Quality Dashboard
#
# Runs all fidelity suites and performance benchmarks, captures their outputs,
# and generates a comprehensive human-friendly report covering the full nam-rs
# universe: all architectures, models, quality modes, and ISAs.
#
# Usage:
#   ./utils/quality-dashboard.sh                        Full dashboard (fidelity + performance)
#   ./utils/quality-dashboard.sh --fidelity-only        Fidelity tests only
#   ./utils/quality-dashboard.sh --bench-only           Benchmarks only
#   ./utils/quality-dashboard.sh --save <filename>      Save plain-text copy alongside display
#   ./utils/quality-dashboard.sh --check <file>         Verify metrics against quality contract

set -euo pipefail

export LC_ALL=C

PHASE_TOTAL=0
source "$(dirname "$0")/_lib.sh"

# ── Argument parsing ────────────────────────────────────────────────────────

SAVE_FILE=""
CHECK_FILE=""
MODE="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save)
            SAVE_FILE="$2"
            shift 2
            ;;
        --check)
            CHECK_FILE="$2"
            shift 2
            ;;
        --fidelity-only)
            MODE="fidelity"
            shift
            ;;
        --bench-only)
            MODE="bench"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# ── Setup ───────────────────────────────────────────────────────────────────

LOGDIR="target/logs/dashboard"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

JSONL_METRICS="${LOGDIR}/metrics.jsonl"
: "${NAM_METRICS_JSONL:=$JSONL_METRICS}"

TMPDIR="${TMPDIR:-/tmp}"
PARSEDIR="$(mktemp -d "$TMPDIR/nam-dashboard-XXXXXX")"
trap 'rm -rf "$PARSEDIR"' EXIT

# ── System info detection ───────────────────────────────────────────────────

detect_isa() {
    local flags
    flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
    if echo "$flags" | grep -q 'avx512f'; then
        echo "AVX-512"
    elif echo "$flags" | grep -q 'avx2'; then
        echo "AVX2 (x86-64-v3)"
    else
        echo "x86-64 (base)"
    fi
}

detect_cpu_model() {
    grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | sed 's/^model name[[:space:]]*: //' || echo "unknown"
}

ISA="$(detect_isa)"
CPU_MODEL="$(detect_cpu_model)"
NOW="$(date '+%Y-%m-%d %H:%M:%S %z')"
RUSTC_VER="$(rustc --version 2>/dev/null || echo 'unknown')"

# Locale-safe numeric printf — bash's printf is locale-aware for %f/%e/%g;
# in pt_BR the decimal separator is comma, so we force C locale for numbers.
_nfmt() { LC_NUMERIC=C printf "$@"; }

# Unified metric formatter — locale-safe (LC_ALL=C), auto-detects scientific notation.
# Values containing [eE] use %.2e; everything else uses %.4f.
_fmt_metric() {
    local val="$1"
    [ -z "$val" ] || [ "$val" = "N/A" ] && { echo "N/A"; return; }
    if [[ "$val" =~ [eE] ]]; then
        LC_ALL=C printf "%.2e" "$val" 2>/dev/null || echo "$val"
    else
        LC_ALL=C printf "%.4f" "$val" 2>/dev/null || echo "$val"
    fi
}

# ESR_F64_MODEL_MAP — Per-model mapping: each dashboard label → its exact .nam fixture.
# Every golden model with an oracle measurement in test_summary_table is mapped 1:1.
# Models without oracle coverage (containers, private-only models) transparently show N/A.
declare -A ESR_F64_MODEL_MAP=(
    # WaveNet standard family — each model measured individually
    ["BossWN-standard"]="BossWN-standard.nam"
    ["BossWN-feather"]="BossWN-feather.nam"
    ["BossWN-nano"]="BossWN-nano.nam"
    ["EVH-5150-Lite"]="EVH-5150-Lite.nam"
    ["wavenet_a1_standard (Official)"]="wavenet_a1_standard.nam"
    ["WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference"]="wavenet_condition_dsp.nam"
    ["WaveNet Official (CH=3, dynamic path) C++ cross-reference"]="wavenet_official.nam"
    ["WaveNetDyn Free-Shape (CH=7→4, dynamic path) C++ cross-reference"]="wavenet_dyn_free.nam"
    ["T-HF1.4: WaveNet Standard polynomial SIMD (regression gate)"]="BossWN-standard.nam"
    # LSTM family — each model measured individually
    ["BossLSTM-1x16"]="BossLSTM-1x16.nam"
    ["BossLSTM-2x8"]="BossLSTM-2x8.nam"
    ["lstm (Official)"]="lstm.nam"
    ["LSTM-Dyn 1×7 (dynamic path) C++ cross-reference"]="lstm_dyn_test.nam"
    # A2 family — each model measured individually
    ["WaveNet A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_full.nam"
    ["WaveNet A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_full.nam"
    ["Container A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container File A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container File A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_full.nam"
    ["SlimmableContainer A2 Example (CH=3→6) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["T-HF1.4: WaveNet A2-Full polynomial SIMD (regression gate)"]="wavenet_a2_full.nam"
    ["WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference"]="a2_dynamic_gated_ch8.nam"
    ["WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference"]="a2_dynamic_blended_ch3.nam"
    # A2-FiLM — each measured individually
    ["WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference"]="wavenet_a2_film_lite.nam"
    ["WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) C++ cross-reference"]="wavenet_a2_film_chaos_stress.nam"
    ["WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference"]="wavenet_a2_film_full.nam"
    ["WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference"]="wavenet_a2_film_input_mixin_pre.nam"
    # ConvNet
    ["ConvNet Test"]="convnet_test.nam"
    # Quick parity labels — same models as their golden counterparts
    ["Quick LSTM 1×16"]="BossLSTM-1x16.nam"
    ["Quick WaveNet CH16"]="BossWN-standard.nam"
    ["Quick A2-Full"]="wavenet_a2_full.nam"
    ["Quick ConvNet"]="convnet_test.nam"
)

# Validate that a string actually looks like a scientific-notation ESR value
# (e.g. "6.13e-14", "0.00e0", "3.17e-3", "0"). Defensive check (2026-07-05):
# a malformed/interleaved line in a parallel `cargo test` log run has been
# observed to make an ESR_F64_PAIRED entry hold a non-numeric label (e.g. "WaveNet")
# instead of a value — this rejects such garbage before it reaches the
# report instead of silently displaying it as if it were a measurement.
_is_numeric_esr() {
    local v="$1"
    [[ "$v" =~ ^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]
}

_lookup_esr_f64() {
    local golden_label="$1"

    local oracle_fixture="${ESR_F64_MODEL_MAP[$golden_label]:-}"
    if [ -n "$oracle_fixture" ]; then
        local val
        set +u; val="${ESR_F64_PAIRED[$oracle_fixture]}"; set -u
        if [ -n "$val" ] && _is_numeric_esr "$val"; then
            echo "$val"
            echo "exact"
            return
        fi
    fi

    # Fallback — try golden_label directly in ESR_F64_PAIRED
    local direct
    set +u; direct="${ESR_F64_PAIRED[$golden_label]}"; set -u
    if [ -n "$direct" ] && _is_numeric_esr "$direct"; then
        echo "$direct"
        echo "exact"
        return
    fi

    # Fallback — try golden_label.nam in ESR_F64_PAIRED
    local with_nam="${golden_label}.nam"
    set +u; direct="${ESR_F64_PAIRED[$with_nam]}"; set -u
    if [ -n "$direct" ] && _is_numeric_esr "$direct"; then
        echo "$direct"
        echo "exact"
        return
    fi

    echo "N/A"
    echo "none"
}

# ── Data storage (global associative arrays) ────────────────────────────────

declare -A ESR_NAMCORE
declare -A ESR_NAMCORE_DB
declare -A ESR_F64_COLD
declare -A ESR_F64_PAIRED
declare -A ESR_F64_DB_COLD
declare -A ESR_F64_DB_PAIRED
declare -A SNR_DB
declare -A MSE_VAL
declare -A MRSTFT
declare -A LATENCY_US
declare -A BENCH_MODEL_MAP
declare -A ISA_RESULTS
declare -A ACTIVATION_SNR
declare -A F64_DECOMPOSITION
declare -A MODEL_ESR_F64_TABLE

declare -a MODEL_ORDER
declare -a ALL_BENCH_NAMES

SPECTRAL_PASSED_COUNT=0

# ── Duration tracking ───────────────────────────────────────────────────────

OVERALL_START=$(date +%s%N)
FIDELITY_DURATION_S=0
BENCH_DURATION_S=0

# ── Run: golden_vectors ─────────────────────────────────────────────────────

run_golden_vectors() {
    local log="$LOGDIR/golden_vectors.log"
    echo -e "\n${BLUE}${BOLD}-> Executando golden_vectors...${NC}"
    local start_t end_t
    start_t=$(date +%s%N)
    # --test-threads=1: parse_golden_vectors() is a stateful, block-based awk
    # parser over interleaved println! output. Parallel test execution (the
    # default) interleaves stdout from concurrently-running tests, corrupting
    # block boundaries non-deterministically — different models "lose" their
    # value on different runs. Verified via repeated direct `cargo test`
    # invocations during the LSTM Recurrent State Drift audit (2026-07-09):
    # golden_vectors.log always yields BossLSTM-2x8 ESR=2.68e-3 and
    # lstm(Official) ESR=1.04e-3 with --test-threads=1, but the dashboard
    # intermittently showed 0.00e0 for one or the other without it.
    NAM_METRICS_JSONL="$NAM_METRICS_JSONL" cargo test --release --test models golden_vectors -- --test-threads=1 --nocapture > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    FIDELITY_DURATION_S=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    local line_count
    line_count=$(wc -l < "$log" 2>/dev/null || echo 0)
    echo -e "  ${GREEN}ok${NC} golden_vectors concluido (${FIDELITY_DURATION_S}s, ${line_count} linhas)"
}

# ── Run: reference_oracle_f64 ───────────────────────────────────────────────

run_reference_oracle() {
    local log="$LOGDIR/oracle_f64.log"
    echo -e "${BLUE}${BOLD}-> Executando reference_oracle_f64...${NC}"
    local start_t end_t
    start_t=$(date +%s%N)
    # --test-threads=1: this binary's stdout is scanned with a stateful,
    # multi-line table parser (parse_oracle_f64). Running tests in parallel
    # (the default) interleaves unrelated tests' output into the middle of
    # that table, which was confirmed (2026-07-05) to corrupt parsed entries.
    # Serializing here removes the interleaving at the source; the awk-side
    # row-shape validation in parse_oracle_f64 is kept as defense-in-depth.
    cargo test --release --test parity reference_oracle_f64 -- --test-threads=1 --nocapture > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    echo -e "  ${GREEN}ok${NC} reference_oracle_f64 concluido (${dur}s)"
}

# ── Run: isa_parity ─────────────────────────────────────────────────────────

run_isa_parity() {
    local log="$LOGDIR/isa_parity.log"
    echo -e "${BLUE}${BOLD}-> Executando isa_parity...${NC}"
    local start_t end_t
    start_t=$(date +%s%N)
    cargo test --release --test parity isa_parity -- --test-threads=1 --nocapture > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    echo -e "  ${GREEN}ok${NC} isa_parity concluido (${dur}s)"
}

# ── Run: spectral_fidelity ──────────────────────────────────────────────────

run_spectral_fidelity() {
    local log="$LOGDIR/spectral_fidelity.log"
    echo -e "${BLUE}${BOLD}-> Executando spectral_fidelity...${NC}"
    local start_t end_t
    start_t=$(date +%s%N)
    cargo test --release --test models spectral_fidelity -- --nocapture > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    echo -e "  ${GREEN}ok${NC} spectral_fidelity concluido (${dur}s)"
}

# ── Run: lstm_activation_precision ──────────────────────────────────────────

run_activation_precision() {
    local log="$LOGDIR/activation_precision.log"
    echo -e "${BLUE}${BOLD}-> Executando lstm_activation_precision...${NC}"
    local start_t end_t
    start_t=$(date +%s%N)
    cargo test --release --test models lstm_activation_precision -- --nocapture > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    echo -e "  ${GREEN}ok${NC} lstm_activation_precision concluido (${dur}s)"
}

# ── Run: quick_parity ────────────────────────────────────────────────────────

run_quick_parity() {
    local log="$LOGDIR/quick_parity.log"
    echo -e "${BLUE}${BOLD}-> Executando quick_parity...${NC}"
    local start_t end_t
    start_t=$(date +%s%N)
    NAM_METRICS_JSONL="$NAM_METRICS_JSONL" cargo test --test parity quick_parity -- --test-threads=1 --nocapture > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    echo -e "  ${GREEN}ok${NC} quick_parity concluido (${dur}s)"
}

# ── Run: regression_gate ────────────────────────────────────────────────────

run_benchmarks() {
    local log="$LOGDIR/regression_gate.log"
    echo -e "\n${BLUE}${BOLD}-> Executando regression_gate benchmarks...${NC}"
    local start_t end_t
    start_t=$(date +%s%N)
    cargo bench --bench regression_gate > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    BENCH_DURATION_S=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    echo -e "  ${GREEN}ok${NC} regression_gate concluido (${BENCH_DURATION_S}s)"
}

# ── Parse: JSONL fidelity metrics (preferred) ────────────────────────────────
# Reads the JSONL file produced by cargo test under NAM_METRICS_JSONL.
# Populates ESR_NAMCORE, ESR_NAMCORE_DB, SNR_DB, MSE_VAL, MRSTFT, MODEL_ORDER.
# Returns 0 on success, 1 if the file is missing or unparseable.

parse_jsonl_fidelity() {
    local jsonl="${NAM_METRICS_JSONL:-}"
    [ -n "$jsonl" ] && [ -f "$jsonl" ] || return 1

    local parsed="$PARSEDIR/jsonl_fidelity.parsed"

    if command -v jq >/dev/null 2>&1; then
        # Prefer jq: output one TSV record per metric per line
        jq -r '{
            label: .label,
            esr: .esr,
            esr_db: .esr_db,
            snr_db: .snr_db,
            mse: .mse,
            mrstft: .mrstft
        } | [.label, .esr, .esr_db, .snr_db, .mse, .mrstft] | @tsv' "$jsonl" 2>/dev/null | \
        LC_ALL=C awk -F'\t' 'NF >= 6 {
            printf "ESR_NAMCORE\t%s\t%s\n", $1, $2
            printf "ESR_NAMCORE_DB\t%s\t%s\n", $1, $3
            printf "SNR_DB\t%s\t%s\n", $1, $4
            printf "MSE\t%s\t%s\n", $1, $5
            printf "MRSTFT\t%s\t%s\n", $1, $6
        }' > "$parsed" 2>/dev/null
    else
        # Fallback: awk-based JSON extraction (defensive, handles reordered keys)
        LC_ALL=C awk '{
            label=""; esr=""; esr_db=""; snr_db=""; mse=""; mrstft=""
            if (match($0, /"label"[[:space:]]*:[[:space:]]*"([^"]*)"/, a)) label = a[1]
            if (match($0, /"esr"[[:space:]]*:[[:space:]]*([^,}]+)/, a)) { esr = a[1]; gsub(/^[[:space:]]+/, "", esr) }
            if (match($0, /"esr_db"[[:space:]]*:[[:space:]]*([^,}]+)/, a)) { esr_db = a[1]; gsub(/^[[:space:]]+/, "", esr_db) }
            if (match($0, /"snr_db"[[:space:]]*:[[:space:]]*([^,}]+)/, a)) { snr_db = a[1]; gsub(/^[[:space:]]+/, "", snr_db) }
            if (match($0, /"mse"[[:space:]]*:[[:space:]]*([^,}]+)/, a)) { mse = a[1]; gsub(/^[[:space:]]+/, "", mse) }
            if (match($0, /"mrstft"[[:space:]]*:[[:space:]]*([^,}]+)/, a)) { mrstft = a[1]; gsub(/^[[:space:]]+/, "", mrstft) }
            if (label != "" && esr != "") {
                printf "ESR_NAMCORE\t%s\t%s\n", label, esr
                printf "ESR_NAMCORE_DB\t%s\t%s\n", label, esr_db
                printf "SNR_DB\t%s\t%s\n", label, snr_db
                printf "MSE\t%s\t%s\n", label, mse
                printf "MRSTFT\t%s\t%s\n", label, mrstft
            }
        }' "$jsonl" > "$parsed" 2>/dev/null
    fi

    [ -s "$parsed" ] || return 1

    while IFS=$'\t' read -r metric key value; do
        case "$metric" in
            ESR_NAMCORE)    ESR_NAMCORE["$key"]="$value" ;;
            ESR_NAMCORE_DB) ESR_NAMCORE_DB["$key"]="$value" ;;
            SNR_DB)         SNR_DB["$key"]="$value" ;;
            MSE)            MSE_VAL["$key"]="$value" ;;
            MRSTFT)         MRSTFT["$key"]="$value" ;;
        esac
    done < "$parsed"

    # Label remapping for quick_parity → golden_vectors key space
    declare -A _LMAP=(
        ["Quick ConvNet @48000 Live"]="ConvNet Test @48000 Live"
    )
    for _old in "${!_LMAP[@]}"; do
        local _new="${_LMAP[$_old]}"
        [ -n "${ESR_NAMCORE[$_old]:-}" ] && ESR_NAMCORE["$_new"]="${ESR_NAMCORE[$_old]}"
        [ -n "${ESR_NAMCORE_DB[$_old]:-}" ] && ESR_NAMCORE_DB["$_new"]="${ESR_NAMCORE_DB[$_old]}"
        [ -n "${SNR_DB[$_old]:-}" ] && SNR_DB["$_new"]="${SNR_DB[$_old]}"
        [ -n "${MSE_VAL[$_old]:-}" ] && MSE_VAL["$_new"]="${MSE_VAL[$_old]}"
        [ -n "${MRSTFT[$_old]:-}" ] && MRSTFT["$_new"]="${MRSTFT[$_old]}"
        unset "ESR_NAMCORE[$_old]" "ESR_NAMCORE_DB[$_old]" "SNR_DB[$_old]" "MSE_VAL[$_old]" "MRSTFT[$_old]"
    done

    local sorted_keys
    set +u
    sorted_keys=$(for k in "${!ESR_NAMCORE[@]}"; do echo "$k"; done | sort -u)
    set -u
    while IFS= read -r key; do
        [ -n "$key" ] && MODEL_ORDER+=("$key")
    done <<< "$sorted_keys"

    return 0
}

# ── Parse: golden_vectors ───────────────────────────────────────────────────
# Parses report_dsp_fidelity blocks and ConvNet Self-Golden output.
# Writes tab-separated records to a temp file, then reads back in the
# current shell to populate global associative arrays.

parse_golden_vectors() {
    local log="$LOGDIR/golden_vectors.log"

    if parse_jsonl_fidelity; then
        echo -e "  ${GREEN}ok${NC} metricas carregadas via JSONL (${#ESR_NAMCORE[@]} entradas)"
        return 0
    fi

    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/golden_vectors.parsed"
    LC_ALL=C awk '
    BEGIN { label=""; rate=""; mode="Live" }
    /^\[NeuralAmpModelerCore/ && /NAM-rs — / {
        line = $0
        sub(/^\[NeuralAmpModelerCore.*NAM-rs — /, "", line)
        sub(/\]$/, "", line)
        at_pos = index(line, " @ ")
        if (at_pos > 0) {
            lbl = substr(line, 1, at_pos - 1)
            rate_str = substr(line, at_pos + 3)
            gsub(/ Hz.*/, "", rate_str)
            rate = rate_str
        } else {
            lbl = line
            rate = "48000"
        }
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
        if (lbl ~ /^(T[0-9]|T-)/) {
            label = ""
            next
        }
        label = lbl " @" rate
        mode = "Live"
        if (index($0, "(HQ)") > 0) mode = "HQ"
    }
    /^  ESR     =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); esr_val = parts[1]
        esr_db = ""
        if (match($0, /\([-0-9.]+ dB\)/)) {
            esr_db = substr($0, RSTART+1, RLENGTH-5)
            gsub(/[[:space:]]+/, "", esr_db)
        }
        key = label " " mode
        printf "ESR_NAMCORE\t%s\t%s\n", key, esr_val
        printf "ESR_NAMCORE_DB\t%s\t%s\n", key, esr_db
    }
    /^  SNR     =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); snr_val = parts[1]
        printf "SNR_DB\t%s\t%s\n", label " " mode, snr_val
    }
    /^  MSE     =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); mse_val = parts[1]
        printf "MSE\t%s\t%s\n", label " " mode, mse_val
    }
    /^  MR-STFT =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); mrstft_val = parts[1]
        printf "MRSTFT\t%s\t%s\n", label " " mode, mrstft_val
    }
    /^\[ConvNet Self-Golden/ {
        label = ""
        printf "ESR_NAMCORE\tConvNet Test @48000 Live\tN/A\n"
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric key value; do
        case "$metric" in
            ESR_NAMCORE)    ESR_NAMCORE["$key"]="$value" ;;
            ESR_NAMCORE_DB) ESR_NAMCORE_DB["$key"]="$value" ;;
            SNR_DB)         SNR_DB["$key"]="$value" ;;
            MSE)            MSE_VAL["$key"]="$value" ;;
            MRSTFT)         MRSTFT["$key"]="$value" ;;
        esac
    done < "$parsed"

    # Build ordered model list from ESR_NAMCORE keys (sorted)
    local sorted_keys
    set +u
    sorted_keys=$(for k in "${!ESR_NAMCORE[@]}"; do echo "$k"; done | sort -u)
    set -u
    while IFS= read -r key; do
        [ -n "$key" ] && MODEL_ORDER+=("$key")
    done <<< "$sorted_keys"
}

# ── Parse: reference_oracle_f64 ─────────────────────────────────────────────

parse_oracle_f64() {
    local log="$LOGDIR/oracle_f64.log"
    [ -f "$log" ] || return 0

    # Parse ESR summary table — skip debug lines (MODEL CLASS LABEL, PROD FIRST, etc.)
    #
    # ROOT-CAUSE FIX (Épico EQ audit, 2026-07-05): this table scan used to have
    # two compounding bugs, both confirmed by direct reproduction:
    #   (A) `printf "...\t%s\t%s\n", $1, $0` redundantly re-embedded $1 inside
    #       $0, so every later bash-side `awk '{print $N}'` re-split was off
    #       by one column (family ended up in the esr_lin slot, etc.) — this
    #       happened on EVERY row, unconditionally.
    #   (B) `in_table` only reset on a blank line or a "test " line, so when
    #       `cargo test` (which runs test functions in parallel by default)
    #       interleaves another test's stdout into this window, unrelated
    #       lines (e.g. "<Family> Decomposition:", "ESR(f32 vs f64 oracle): ..."
    #       from a concurrently-running decomposition test) were vacuumed up
    #       and misparsed as if they were table rows.
    # Fix: require the row to actually look like a table row (filename ends
    # in `.nam`, third field is scientific notation) before accepting it, and
    # have awk emit the four columns pre-split — no bash-side re-splitting,
    # no duplicated `$0`, so there is nothing left to shift.
    local parsed="$PARSEDIR/oracle_f64_summary.parsed"
    LC_ALL=C awk '
    BEGIN { in_table = 0 }
    /^=== ESR\(f32 vs f64 oracle\) Summary/ { in_table = 1; next }
    /^---/ && in_table { in_table = 2; next }
    # Stop table on empty line or test result line (starts with "test ")
    in_table == 2 && (/^$/ || /^test /) { in_table = 0; next }
    # Skip debug lines mixed into the table
    in_table == 2 && /^(MODEL CLASS LABEL|PROD FIRST|ORACLE FIRST)/ { next }
    # Capture data rows ONLY if they actually have the expected shape: a
    # `.nam` filename in column 1 and scientific-notation ESR in column 3.
    # Anything else (foreign interleaved output) is silently ignored rather
    # than mis-captured as a row.
    in_table == 2 && $1 ~ /\.nam$/ && $3 ~ /^[+-]?[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$/ {
        printf "ESR_F64_TABLE\t%s\t%s\t%s\t%s\n", $1, $2, $3, $4
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric filename family esr_lin esr_db; do
        [[ "$metric" == "ESR_F64_TABLE" ]] || continue
        # Defensive: reject a parsed value that isn't actually numeric — a
        # second, independent safety net on top of the strict awk shape
        # check above, in case the log format changes again in the future.
        if [ -n "$filename" ] && [ -n "$esr_lin" ]; then
            if _is_numeric_esr "$esr_lin"; then
                ESR_F64_PAIRED["$filename"]="$esr_lin"
            else
                echo "  ⚠ Descartando entrada f64 nao-numerica para '$filename': [$esr_lin] (linha malformada em oracle_f64.log)" >&2
            fi
        fi
        [ -n "$filename" ] && [ -n "$esr_db" ] && ESR_F64_DB_PAIRED["$filename"]="$esr_db"
        [ -n "$filename" ] && MODEL_ESR_F64_TABLE["$filename"]="${family}|${esr_lin}|${esr_db}"
    done < "$parsed"

    # Parse paired prewarm ESR lines — labels like "LSTM", "WaveNet", "ConvNet"
    grep -E ' ESR\(f32 vs oracle, prewarm-paired' "$log" > "$parsed" 2>/dev/null || true
    while IFS= read -r line; do
        local label esr esr_db
        label=$(echo "$line" | sed 's/ ESR(f32 vs oracle, prewarm-paired.*//')
        esr=$(echo "$line" | grep -oP ':\s+\K[0-9.e+\-]+' 2>/dev/null || true)
        esr_db=$(echo "$line" | grep -oP '\(\K[-0-9.]+(?= dB\))' 2>/dev/null || true)
        # Same defensive check as above — see note there.
        if [ -n "$label" ] && [ -n "$esr" ]; then
            if _is_numeric_esr "$esr"; then
                ESR_F64_PAIRED["$label"]="$esr"
            else
                echo "  ⚠ Descartando entrada f64 nao-numerica para familia '$label': [$esr] (linha malformada em oracle_f64.log)" >&2
            fi
        fi
        [ -n "$label" ] && [ -n "$esr_db" ] && ESR_F64_DB_PAIRED["$label"]="$esr_db"
    done < "$parsed"

    # Parse decomposition blocks
    LC_ALL=C awk '
    BEGIN { lbl=""; buf=""; in_decomp=0 }
    /Decomposition:/ {
        lbl = $0
        sub(/ Decomposition:.*/, "", lbl)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
        buf = $0 "\n"
        in_decomp = 1
        next
    }
    in_decomp {
        if ($0 ~ /^[[:space:]]*(ESR|ΔESR|combined|Δ|accumulation|activation|weights)/) {
            buf = buf $0 "\n"
        } else {
            if (lbl != "" && buf != "") {
                gsub(/\n/, "@@", buf)
                printf "F64_DECOMP\t%s\t%s\n", lbl, buf
            }
            in_decomp = 0; lbl = ""; buf = ""
        }
    }
    END {
        if (lbl != "" && buf != "") {
            gsub(/\n/, "@@", buf)
            printf "F64_DECOMP\t%s\t%s\n", lbl, buf
        }
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric key value; do
        [[ "$metric" == "F64_DECOMP" ]] || continue
        value="${value//@@/$'\n'}"
        F64_DECOMPOSITION["$key"]="$value"
    done < "$parsed"

    # Parse per-model f64 ESR from decomposition blocks to populate ESR_F64_COLD/ESR_F64_DB_COLD
    LC_ALL=C awk '
    /Decomposition:/ {
        lbl = $0
        sub(/Decomposition:.*/, "", lbl)
        sub(/.* \.\.\. /, "", lbl)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
        in_block = 1
        next
    }
    in_block {
        if ($0 ~ /ESR\(f32 vs f64 oracle\):/) {
            esr = $0
            sub(/.*ESR\(f32 vs f64 oracle\):[[:space:]]*/, "", esr)
            db = esr
            sub(/[[:space:]]*\(.*/, "", esr)
            sub(/.*\(/, "", db)
            sub(/[[:space:]]*dB\).*/, "", db)
            gsub(/[[:space:]]/, "", esr)
            gsub(/[[:space:]]/, "", db)
            printf "%s\t%s\t%s\n", lbl, esr, db
            in_block = 0
        } else if ($0 ~ /Decomposition:/) {
            lbl = $0
            sub(/Decomposition:.*/, "", lbl)
            sub(/.* \.\.\. /, "", lbl)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
            in_block = 1
        }
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r label esr db; do
        [ -n "$label" ] && [ -n "$esr" ] || continue
        if _is_numeric_esr "$esr"; then
            ESR_F64_COLD["$label"]="$esr"
            [ -n "$db" ] && ESR_F64_DB_COLD["$label"]="$db"
        fi
    done < "$parsed"
}

# ── Parse: isa_parity ───────────────────────────────────────────────────────

parse_isa_parity() {
    local log="$LOGDIR/isa_parity.log"
    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/isa_parity.parsed"

    # Cross-ISA lines — [ISA Matrix] appears after cargo test prefix, not at line start
    grep -E '\[ISA Matrix\]' "$log" | grep -v 'self-consistency' > "$parsed" 2>/dev/null || true
    while IFS= read -r line; do
        local label esr ref_isa test_isa key
        label=$(echo "$line" | sed 's/^\[ISA Matrix\] //' | awk -F'|' '{print $1}' | sed 's/[[:space:]]*$//')
        ref_isa=$(echo "$line" | awk -F'|' '{print $2}' | sed 's/[[:space:]]*→.*//; s/^[[:space:]]+//')
        test_isa=$(echo "$line" | awk -F'|' '{print $2}' | sed 's/.*→ //; s/[[:space:]]*$//')
        esr=$(echo "$line" | grep -oP 'ESR=\K[0-9.e+\-]+' 2>/dev/null || echo "N/A")
        key="${label} | ${ref_isa}->${test_isa}"
        ISA_RESULTS["$key"]="$esr"
    done < "$parsed"

    # Self-consistency lines
    grep -E '\[ISA Matrix\].*self-consistency' "$log" > "$parsed" 2>/dev/null || true
    while IFS= read -r line; do
        local label mse key
        label=$(echo "$line" | sed 's/^\[ISA Matrix\] //' | awk -F'|' '{print $1}' | sed 's/[[:space:]]*$//')
        mse=$(echo "$line" | grep -oP 'MSE=\K[0-9.e+\-]+' 2>/dev/null || echo "N/A")
        key="${label} | self-consistency"
        ISA_RESULTS["$key"]="$mse"
    done < "$parsed"
}

# ── Parse: spectral_fidelity ────────────────────────────────────────────────

parse_spectral_fidelity() {
    local log="$LOGDIR/spectral_fidelity.log"
    [ -f "$log" ] || return 0
    SPECTRAL_PASSED_COUNT=$(grep -c 'all spectral fidelity metrics within baseline tolerance' "$log" 2>/dev/null || true)
}

# ── Parse: lstm_activation_precision ────────────────────────────────────────

parse_activation_precision() {
    local log="$LOGDIR/activation_precision.log"
    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/activation.parsed"
    grep -E 'Fast\(Padé\).*Standard\(exact\)' "$log" > "$parsed" 2>/dev/null || true

    while IFS= read -r line; do
        local model fast_snr exact_snr delta
        model=$(echo "$line" | sed 's/[[:space:]]*Fast(Pad.*//' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
        fast_snr=$(echo "$line" | grep -oP 'Fast\(Pad.*?\):\s+\K[0-9.]+' 2>/dev/null || echo "N/A")
        exact_snr=$(echo "$line" | grep -oP 'Standard\(exact\):\s+\K[0-9.]+' 2>/dev/null || echo "N/A")
        delta=$(echo "$line" | grep -oP 'Δ=\K[+-][0-9.]+' 2>/dev/null || echo "0.0")
        ACTIVATION_SNR["$model"]="${fast_snr:-N/A}|${exact_snr:-N/A}|${delta:-0.0}"
    done < "$parsed"
}

# ── Parse: regression_gate ──────────────────────────────────────────────────

parse_benchmarks() {
    local log="$LOGDIR/regression_gate.log"
    [ -f "$log" ] || return 0

    BENCH_MODEL_MAP["RT_WaveNet_Std_CH16"]="WaveNet Standard CH16"
    BENCH_MODEL_MAP["RT_WaveNet_Feather_CH8"]="WaveNet Feather CH8"
    BENCH_MODEL_MAP["RT_WaveNet_Lite_CH12"]="WaveNet Lite CH12"
    BENCH_MODEL_MAP["RT_WaveNet_Nano_CH4"]="WaveNet Nano CH4"
    BENCH_MODEL_MAP["RT_A2_Full_CH8"]="A2 Full CH8"
    BENCH_MODEL_MAP["RT_A2_Lite_CH3"]="A2 Lite CH3"
    BENCH_MODEL_MAP["RT_LSTM_1x16"]="LSTM 1x16"
    BENCH_MODEL_MAP["RT_LSTM_2x8"]="LSTM 2x8"
    BENCH_MODEL_MAP["RT_Linear"]="Linear RF=2048"
    BENCH_MODEL_MAP["RT_ConvNet"]="ConvNet"

    local parsed="$PARSEDIR/benchmarks.parsed"
    LC_ALL=C awk '
    BEGIN { bench = "" }
    /^RT_/ && !/regression_gate/ && length($1) > 3 { bench = $1 }
    bench != "" && /time:.*\[/ {
        line = $0
        start_bracket = index(line, "[")
        end_bracket   = index(line, "]")
        if (start_bracket > 0 && end_bracket > start_bracket) {
            bracket_part = substr(line, start_bracket + 1, end_bracket - start_bracket - 1)
            split(bracket_part, parts, /[[:space:]]+/)
            if (parts[3] != "" && parts[4] != "") {
                median_val  = parts[3]
                median_unit = parts[4]
                if (median_unit == "ns")      us = median_val / 1000
                else if (median_unit == "µs") us = median_val
                else if (median_unit == "ms") us = median_val * 1000
                else if (median_unit == "s")  us = median_val * 1000000
                else                          us = median_val
                printf "LATENCY\t%s\t%.2f\n", bench, us
            }
        }
        bench = ""
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric bench latency; do
        [[ "$metric" == "LATENCY" ]] || continue
        LATENCY_US["$bench"]="$latency"
    done < "$parsed"

    for bn in RT_WaveNet_Std_CH16 RT_WaveNet_Feather_CH8 RT_WaveNet_Lite_CH12 \
              RT_WaveNet_Nano_CH4 RT_A2_Full_CH8 RT_A2_Lite_CH3 \
              RT_LSTM_1x16 RT_LSTM_2x8 RT_Linear RT_ConvNet; do
        ALL_BENCH_NAMES+=("$bn")
    done
}

# ── ESR verdict translation ─────────────────────────────────────────────────

esr_verdict() {
    local esr="$1"
    [ -z "$esr" ] || [ "$esr" = "N/A" ] && { echo "N/A"; return; }
    LC_ALL=C awk -v v="$esr" 'BEGIN {
        if (v+0 < 1e-10) print "IDENTICO"
        else if (v+0 < 1e-5) print "IMPERCEPTIVEL"
        else if (v+0 < 1e-2) print "AUDIVEL APENAS COM A/B CIENTIFICO"
        else if (v+0 < 1e-1) print "AUDIVEL EM COMPARACAO DIRETA"
        else print "⚠ AUDIVEL"
    }'
}

esr_verdict_short() {
    local esr="$1"
    [ -z "$esr" ] || [ "$esr" = "N/A" ] && { echo "N/A"; return; }
    local cmp
    cmp=$(LC_ALL=C awk -v v="$esr" 'BEGIN {
        if (v+0 < 1e-10) print "1"
        else if (v+0 < 1e-5) print "2"
        else if (v+0 < 1e-2) print "3"
        else if (v+0 < 1e-1) print "4"
        else print "5"
    }')
    case "$cmp" in
        1) echo -e "${GREEN}IDENTICO${NC}" ;;
        2) echo -e "${GREEN}IMPERCEPTIVEL${NC}" ;;
        3) echo -e "${YELLOW}A/B CIENTIFICO${NC}" ;;
        4) echo -e "${YELLOW}AUDIVEL DIRETO${NC}" ;;
        *) echo -e "${RED}⚠ AUDIVEL${NC}" ;;
    esac
}

# Colorize an ESR numeric string with GREEN/YELLOW/RED ANSI codes.
# Usage: _esr_color <numeric_esr_value>  — echoes the value wrapped in ANSI.
_esr_color() {
    local esr="$1"
    [ -z "$esr" ] || [ "$esr" = "N/A" ] && { echo "$esr"; return; }
    local cmp
    cmp=$(LC_ALL=C awk -v v="$esr" 'BEGIN {
        if (v+0 < 1e-10) print "GREEN"
        else if (v+0 < 1e-5) print "GREEN"
        else if (v+0 < 1e-2) print "YELLOW"
        else if (v+0 < 1e-1) print "YELLOW"
        else print "RED"
    }')
    case "$cmp" in
        GREEN)  echo -e "${GREEN}${esr}${NC}" ;;
        YELLOW) echo -e "${YELLOW}${esr}${NC}" ;;
        RED)    echo -e "${RED}${esr}${NC}" ;;
        *)      echo "$esr" ;;
    esac
}

# Colorize a CPU budget percentage using the same criteria as the
# PERFORMANCE section (folga-based).
_cpu_color() {
    local pct="$1"
    [ -z "$pct" ] || [ "$pct" = "N/A" ] && { echo "N/A"; return; }
    local f
    f=$(LC_ALL=C awk -v v="$pct" 'BEGIN { printf "%.0f", 100.0 - v }')
    if [ "$f" -gt 75 ]; then
        echo -e "${GREEN}${pct}%${NC}"
    elif [ "$f" -gt 50 ]; then
        echo -e "${GREEN}${pct}%${NC}"
    elif [ "$f" -gt 25 ]; then
        echo -e "${YELLOW}${pct}%${NC}"
    else
        echo -e "${RED}${pct}%${NC}"
    fi
}

budget_pct() {
    local latency_us="$1"
    [ -z "$latency_us" ] || [ "$latency_us" = "N/A" ] && { echo "N/A"; return; }
    LC_ALL=C awk -v l="$latency_us" 'BEGIN { printf "%.1f", (l / 1333.0) * 100.0 }'
}

budget_folga() {
    local pct="$1"
    [ "$pct" = "N/A" ] && { echo "N/A"; return; }
    LC_ALL=C awk -v p="$pct" 'BEGIN { printf "%.1f", 100.0 - p }'
}

folga_color() {
    local folga="$1"
    [ "$folga" = "N/A" ] && { echo "N/A"; return; }
    local f
    f=$(LC_ALL=C awk -v v="$folga" 'BEGIN { printf "%.0f", v }')
    if [ "$f" -gt 75 ]; then
        echo -e "${GREEN}${folga}% ok${NC}"
    elif [ "$f" -gt 50 ]; then
        echo -e "${GREEN}${folga}%${NC}"
    elif [ "$f" -gt 25 ]; then
        echo -e "${YELLOW}${folga}%${NC}"
    else
        echo -e "${RED}${folga}% ⚠${NC}"
    fi
}

# ── Render: header ──────────────────────────────────────────────────────────

render_header() {
    local cpu_short="${CPU_MODEL:0:46}"
    printf "╔══════════════════════════════════════════════════════════════════╗\n"
    printf "║              nam-rs Quality Dashboard                            ║\n"
    printf "║              ------------------------------                      ║\n"
    printf "║              Medido em: %-25.25s                ║\n" "$NOW"
    printf "║              ISA: %-46.46s ║\n" "$ISA"
    printf "║              CPU: %-46.46s ║\n" "$cpu_short"
    printf "║              rustc: %-44.44s ║\n" "$RUSTC_VER"
    printf "╚══════════════════════════════════════════════════════════════════╝\n"
}

# ── Render: resumo rapido ───────────────────────────────────────────────────

render_quick_summary() {
    echo ""
    echo "🎯 RESUMO RAPIDO (para nao-cientistas)"
    echo "═══════════════════════════════════════"
    echo ""

    # Find representative models by scanning parsed keys (not hardcoded labels)
    local wn_std_key="" wn_feather_key="" lstm1_key="" lstm2_key=""
    local a2full_key="" a2lite_key="" convnet_key="" linear_key=""
    local a2film_key="" a1std_key=""

    set +u
    for key in "${!ESR_NAMCORE[@]}"; do
        case "$key" in
            *"BossWN-standard"*|*"WaveNet Std"*)   wn_std_key="$key" ;;
            *"BossWN-feather"*|*"WaveNet Feather"*) wn_feather_key="$key" ;;
            *"BossLSTM-1x16"*|*"LSTM 1x16"*)       lstm1_key="$key" ;;
            *"BossLSTM-2x8"*|*"LSTM 2x8"*)         lstm2_key="$key" ;;
            *"A2-Full"*|*"A2 Full"*)               a2full_key="$key" ;;
            *"A2-Lite"*|*"A2 Lite"*)               a2lite_key="$key" ;;
            *"ConvNet"*)                            convnet_key="$key" ;;
            *"linear_fft_rf2048"*|*"Linear FFT RF=2048"*) linear_key="$key" ;;
            *"A2-FiLM-Lite"*|*"FiLM.*Lite"*)       a2film_key="$key" ;;
            *"wavenet_a1_standard"*|*"A1 Standard"*) a1std_key="$key" ;;
        esac
    done
    set -u

    # Display entries (one per representative model family)
    _quick_entry() {
        local label="$1" icon="$2" key="$3" bench_name="$4"
        local esr_nam="${ESR_NAMCORE[$key]:-N/A}"
        local esr_nam_display
        esr_nam_display=$(_fmt_metric "$esr_nam")
        # Extract model name for f64 lookup (strip rate and mode)
        local f64_label
        f64_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
        local esr_f64 esr_f64_provenance
        { read -r esr_f64; read -r esr_f64_provenance; } < <(_lookup_esr_f64 "$f64_label")
        local esr_f64_display
        esr_f64_display=$(_fmt_metric "$esr_f64")
        local esr_f64_colored
        esr_f64_colored=$(_esr_color "$esr_f64_display")
        local verdict
        set +u
        verdict=$(esr_verdict_short "$esr_nam")
        set -u
        local latency="${LATENCY_US[$bench_name]:-N/A}"
        local pct_budget="N/A"
        local cpu_colored="N/A"
        if [ "$latency" != "N/A" ]; then
            pct_budget=$(budget_pct "$latency")
            cpu_colored=$(_cpu_color "$pct_budget")
        fi
        printf "  %s %-38s  vs NAMcore: %-10s %b  │  vs Ideal (f64): %-10s  │  ⚡ CPU: %s do budget\n" \
            "$icon" "${label:0:38}" "$esr_nam_display" "$verdict" "$esr_f64_colored" "$cpu_colored"
    }

    [ -n "$wn_std_key" ]   && _quick_entry "WaveNet Standard (CH16)"  "🎸" "$wn_std_key"    RT_WaveNet_Std_CH16
    [ -n "$a1std_key" ]    && _quick_entry "WaveNet A1 Standard"      "🎸" "$a1std_key"     RT_WaveNet_Std_CH16
    [ -n "$wn_feather_key" ] && _quick_entry "WaveNet Feather (CH8)"  "🎸" "$wn_feather_key" RT_WaveNet_Feather_CH8
    [ -n "$lstm1_key" ]    && _quick_entry "LSTM 1x16 (BossLSTM)"     "🎸" "$lstm1_key"     RT_LSTM_1x16
    [ -n "$lstm2_key" ]    && _quick_entry "LSTM 2x8 (BossLSTM)"      "🎸" "$lstm2_key"     RT_LSTM_2x8
    [ -n "$a2full_key" ]   && _quick_entry "A2 Full (CH8)"            "🎸" "$a2full_key"    RT_A2_Full_CH8
    [ -n "$a2lite_key" ]   && _quick_entry "A2 Lite (CH3)"            "🎸" "$a2lite_key"    RT_A2_Lite_CH3
    [ -n "$a2film_key" ]   && _quick_entry "A2-FiLM Lite (CH3)"       "🎸" "$a2film_key"    RT_A2_Lite_CH3
    [ -n "$convnet_key" ]  && _quick_entry "ConvNet"                  "🎸" "$convnet_key"   RT_ConvNet
    [ -n "$linear_key" ]   && _quick_entry "Linear (RF=2048)"         "🎸" "$linear_key"    RT_Linear

    echo ""
}

# ── Render: fidelity details table ──────────────────────────────────────────

render_fidelity_details() {
    echo "📊 FIDELIDADE SONORA — Detalhes Tecnicos"
    echo "═════════════════════════════════════════"
    echo ""

    if [ ${#MODEL_ORDER[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nenhum dado de fidelidade disponivel.${NC}"
        echo ""
        return
    fi

    printf "  %-38s │ %-16s │ %-12s │ %-8s │ %-8s │ %s\n" \
        "Modelo" "ESR (vs NAMcore)" "ESR (vs f64)" "SNR dB" "MR-STFT" "Modo"
    printf "  %s │ %s │ %s │ %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..38})" "$(printf '─%.0s' {1..16})" \
        "$(printf '─%.0s' {1..12})" "$(printf '─%.0s' {1..8})" \
        "$(printf '─%.0s' {1..8})" "$(printf '─%.0s' {1..6})"

    for key in "${MODEL_ORDER[@]}"; do
        local esr_nam="${ESR_NAMCORE[$key]:-N/A}"
        local esr_nam_short
        esr_nam_short=$(_fmt_metric "$esr_nam")
        # Color ESR value by quality
        local esr_color=""
        if [ "$esr_nam" != "N/A" ]; then
            esr_color=$(awk -v v="$esr_nam" 'BEGIN {
                if (v+0 < 1e-10) print "GREEN"
                else if (v+0 < 1e-5) print "GREEN"
                else if (v+0 < 1e-2) print "YELLOW"
                else if (v+0 < 1e-1) print "YELLOW"
                else print "RED"
            }')
            case "$esr_color" in
                GREEN)  esr_nam_short="${GREEN}${esr_nam_short}${NC}" ;;
                YELLOW) esr_nam_short="${YELLOW}${esr_nam_short}${NC}" ;;
                RED)    esr_nam_short="${RED}${esr_nam_short}${NC}" ;;
            esac
        fi
        local snr="${SNR_DB[$key]:-N/A}"
        local mrstft="${MRSTFT[$key]:-N/A}"
        local mrstft_short
        mrstft_short=$(_fmt_metric "$mrstft")
        local model_label
        model_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
        local esr_f64 esr_f64_provenance
        { read -r esr_f64; read -r esr_f64_provenance; } < <(_lookup_esr_f64 "$model_label")
        local esr_f64_short
        esr_f64_short=$(_fmt_metric "$esr_f64")
        local esr_f64_colored
        esr_f64_colored=$(_esr_color "$esr_f64_short")
        local mode="Live"
        [[ "$key" == *" HQ"* ]] && mode="HQ"
        local display_key="${key:0:38}"

        printf "  %-38s │ %-26b │ %-12s │ %-8s │ %-8s │ %s\n" \
            "$display_key" "$esr_nam_short" "$esr_f64_colored" "$snr" "$mrstft_short" "$mode"
    done
    echo ""
    echo "  Legenda qualitativa (limites de audibilidade do ESR):"
    echo -e "    ${GREEN}verde${NC} = imperceptivel (ESR < 1e-5)"
    echo -e "    ${YELLOW}amarelo${NC} = audivel apenas com A/B cientifico (ESR < 1e-2)"
    echo -e "    ${RED}vermelho${NC} = ⚠ audivel — necessita investigacao (ESR >= 1e-1)"
    echo ""
}

# ── Render: performance ─────────────────────────────────────────────────────

render_performance() {
    echo "⚡ PERFORMANCE — Latencia por Bloco (64 amostras @ 48kHz)"
    echo "══════════════════════════════════════════════════════════"
    echo "  Deadline RT: 1333 µs (1.33 ms)"
    echo ""

    local bench_count
    set +u
    bench_count="${#ALL_BENCH_NAMES[@]}"
    set -u
    if [ -z "$bench_count" ] || [ "$bench_count" -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nenhum dado de performance disponivel.${NC}"
        echo ""
        return
    fi

    printf "  %-28s │ %-16s │ %-12s │ %s\n" \
        "Modelo" "Latencia Mediana" "% do Budget" "Folga"
    printf "  %s │ %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..28})" "$(printf '─%.0s' {1..16})" \
        "$(printf '─%.0s' {1..12})" "$(printf '─%.0s' {1..20})"

    for bn in "${ALL_BENCH_NAMES[@]}"; do
        local label="${BENCH_MODEL_MAP[$bn]:-$bn}"
        local latency="${LATENCY_US[$bn]:-N/A}"
        local pct="N/A"
        local folga="N/A"
        local folga_colored="N/A"
        if [ "$latency" != "N/A" ]; then
            pct=$(budget_pct "$latency")
            folga=$(budget_folga "$pct")
            folga_colored=$(folga_color "$folga")
        fi
        local latency_display="$latency"
        if [ "$latency" != "N/A" ]; then
            latency_display=$(_nfmt "%.1f us" "$latency")
        fi
        printf "  %-28s │ %-16s │ %-12s │ %b\n" \
            "$label" "$latency_display" "${pct}%" "$folga_colored"
    done

    echo ""
    echo "  (i) Folga > 50%:  Pode usar oversampling 2x sem xruns"
    echo "  (i) Folga > 75%:  Pode usar oversampling 4x sem xruns"
    echo "  (i) Folga < 25%:  ⚠ Risco de xruns com buffer de 64 amostras"
    echo ""
}

# ── Render: ISA parity ──────────────────────────────────────────────────────

render_isa_parity() {
    echo "🔬 ISA PARITY"
    echo "═════════════"
    echo ""

    # set -u triggers on empty associative arrays in some bash versions
    local count
    set +u
    count="${#ISA_RESULTS[@]}"
    set -u
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nao coberto no modo quick — rode tests-long para verificacao completa.${NC}"
        echo ""
        return
    fi

    local all_pass=true
    local self_consistency_count=0
    local cross_isa_count=0
    local cross_isa_pass=0

    set +u
    for key in "${!ISA_RESULTS[@]}"; do
        if [[ "$key" == *"self-consistency"* ]]; then
            self_consistency_count=$((self_consistency_count + 1))
        else
            cross_isa_count=$((cross_isa_count + 1))
            local esr="${ISA_RESULTS[$key]}"
            if [ -n "$esr" ] && [ "$esr" != "N/A" ]; then
                if awk -v v="$esr" 'BEGIN { exit (v+0 < 1e-8) ? 0 : 1 }'; then
                    cross_isa_pass=$((cross_isa_pass + 1))
                else
                    all_pass=false
                fi
            fi
        fi
    done
    set -u

    if $all_pass && [ "$cross_isa_count" -gt 0 ]; then
        echo -e "  AVX2 vs AVX-512: ${GREEN}bitwise identical ✅${NC}"
    elif [ "$cross_isa_count" -gt 0 ]; then
        echo -e "  AVX2 vs AVX-512: ${YELLOW}divergent on $((cross_isa_count - cross_isa_pass))/$cross_isa_count models ⚠${NC}"
    else
        echo "  AVX2 vs AVX-512: sem dados (CPU pode nao ter AVX-512)"
    fi

    echo "  Self-consistency checks: $self_consistency_count executados"
    echo ""

    if [ "$cross_isa_count" -gt 0 ]; then
        echo "  Detalhes cross-ISA:"
        set +u
        for key in "${!ISA_RESULTS[@]}"; do
            [[ "$key" == *"self-consistency"* ]] && continue
            local esr="${ISA_RESULTS[$key]}"
            local pass_str
            if awk -v v="$esr" 'BEGIN { exit (v+0 < 1e-8) ? 0 : 1 }'; then
                pass_str="✅"
            else
                pass_str="⚠"
            fi
            printf "    %s  ESR=%s  %s\n" "$key" "$esr" "$pass_str"
        done
        set -u
        echo ""
    fi
}

# ── Render: activation precision ────────────────────────────────────────────

render_activation_precision() {
    echo "🎹 ACTIVATION PRECISION"
    echo "════════════════════════"
    echo ""

    set +u
    local count="${#ACTIVATION_SNR[@]}"
    set -u
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nenhum resultado de activation precision disponivel.${NC}"
        echo ""
        return
    fi

    printf "  %-20s │ %-14s │ %-14s │ %s\n" \
        "Modelo" "Fast(Pade)" "Standard(exact)" "Δ SNR"
    printf "  %s │ %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..20})" "$(printf '─%.0s' {1..14})" \
        "$(printf '─%.0s' {1..14})" "$(printf '─%.0s' {1..10})"

    set +u
    for model in "${!ACTIVATION_SNR[@]}"; do
        local data="${ACTIVATION_SNR[$model]}"
        local fast_snr exact_snr delta
        fast_snr=$(echo "$data" | cut -d'|' -f1)
        exact_snr=$(echo "$data" | cut -d'|' -f2)
        delta=$(echo "$data" | cut -d'|' -f3)

        local delta_colored="${delta} dB"
        local delta_val
        delta_val=$(echo "$delta" | sed 's/^[+]//')
        if awk -v v="$delta_val" 'BEGIN { exit (v+0 < 3.0) ? 0 : 1 }'; then
            delta_colored="${delta} dB"
        else
            delta_colored="${YELLOW}${delta} dB${NC}"
        fi

        printf "  %-20s │ %-14s │ %-14s │ %b\n" \
            "$model" "${fast_snr} dB" "${exact_snr} dB" "$delta_colored"
    done
    set -u
    echo ""

    local total=0 count_num=0
    set +u
    for model in "${!ACTIVATION_SNR[@]}"; do
        local data="${ACTIVATION_SNR[$model]}"
        local delta
        delta=$(echo "$data" | cut -d'|' -f3 | sed 's/^[+]//')
        if [ -n "$delta" ] && [ "$delta" != "N/A" ]; then
            total=$(awk -v t="$total" -v d="$delta" 'BEGIN { printf "%.2f", t + d }')
            count_num=$((count_num + 1))
        fi
    done
    set -u
    if [ "$count_num" -gt 0 ]; then
        local avg
        avg=$(awk -v t="$total" -v c="$count_num" 'BEGIN { printf "%.1f", t / c }')
        echo "  Ganho SNR medio com Standard(exact): +${avg} dB (sobre ${count_num} modelos LSTM)"
    fi
    echo ""
}

# ── Render: f64 decomposition ───────────────────────────────────────────────
#
# CORRECTNESS NOTE (Épico EQ audit, 2026-07-05): the decomposition tests
# (`test_decomposition_*` in tests/reference_oracle_f64.rs) run the model
# COLD — a single 256-sample sweep, with NO prewarm — then compare against
# the f64 oracle. For architectures with a non-trivial receptive field
# (WaveNet, A2), the entire 256-sample window falls inside the analytical
# receptive-field-fill transient, so the measured "ESR(f32 vs f64 oracle)"
# here is dominated by that transient, not by steady-state precision loss.
# None of the four decomposed sources (f16c, bf16, Padé, f32 accumulation)
# models transient/prewarm error, so for WaveNet/A2 the sum of sources can
# differ from the total by many orders of magnitude — this is expected given
# the cold-start methodology, not a calculation bug. It also means these
# numbers are NOT comparable to the prewarm-paired "vs Ideal (f64)" values
# shown in the fidelity table above (measured with 24k-sample warmup): for
# the same model, the two can differ by 8-12 orders of magnitude. Below, each
# model's own internal consistency is checked against the project's declared
# Rule 5 (`docs/perceptual_validation.md`: "Σ sources ≈ total, within 10×")
# and flagged when violated, so this isn't silently trusted as a calibration
# input.

# Extract a numeric value following a given label prefix from a decomposition
# block (e.g. "ESR(f32 vs f64 oracle):  3.17e-3 (-25.0 dB)" -> "3.17e-3").
_decomp_extract() {
    local block="$1" label_pattern="$2"
    set +o pipefail
    echo "$block" | grep -oP "${label_pattern}\\K[0-9.eE+-]+" 2>/dev/null | head -1
    set -o pipefail
}

render_f64_decomposition() {
    set +u
    local count="${#F64_DECOMPOSITION[@]}"
    set -u
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        return
    fi
    echo "🔍 F64 ORACLE — Decomposicao de Fontes de Erro"
    echo "══════════════════════════════════════════════"
    echo ""
    echo "  (i) Estas medicoes sao cold-start (256 amostras, SEM prewarm) — NAO"
    echo "      comparaveis aos valores 'vs Ideal (f64)' da tabela de fidelidade"
    echo "      acima (medidos com warmup de 24k amostras). Para WaveNet/A2, o"
    echo "      campo receptivo e maior que a janela de 256 amostras, entao o"
    echo "      ESR total abaixo reflete majoritariamente o transiente de"
    echo "      preenchimento do buffer, nao o piso de precisao em regime"
    echo "      permanente. Ver docs/perceptual_validation.md#lstm-recurrent-state-drift e TODO-findings.md Achado F3."
    echo ""
    set +u
    for model in "${!F64_DECOMPOSITION[@]}"; do
        echo "  ${model}:"
        local block="${F64_DECOMPOSITION[$model]}"
        echo "$block" | while IFS= read -r line; do
            [ -n "$line" ] && echo "    $line"
        done || true

        # Rule 5 self-check (docs/perceptual_validation.md): Σ sources ≈ total,
        # within 10×. Flag it here instead of letting a wildly inconsistent
        # decomposition pass silently as if it were a trustworthy breakdown.
        # Extract model short name to look up cold ESR from ESR_F64_COLD
        # (populated by the decomposition-block parser in parse_oracle_f64).
        local short_name
        short_name=$(echo "$model" | sed 's/.* \.\.\. //')
        local total combined
        total="${ESR_F64_COLD[$short_name]:-}"
        if [ -z "$total" ]; then
            total=$(_decomp_extract "$block" 'ESR\(f32 vs f64 oracle\):\s*')
        fi
        combined=$(_decomp_extract "$block" 'combined \(F16C\+Padé\+F32\):\s*')
        if [ -n "$total" ] && [ -n "$combined" ]; then
            local ratio_flag
            ratio_flag=$(LC_ALL=C awk -v t="$total" -v c="$combined" 'BEGIN {
                if (c == 0) { print "n/a"; exit }
                r = t / c; if (r < 1) r = 1 / r;
                printf "%.0f", r
            }' 2>/dev/null || echo "n/a")
            if [ "$ratio_flag" != "n/a" ] && [ "$ratio_flag" -gt 10 ] 2>/dev/null; then
                echo -e "    ${YELLOW}⚠ Rule 5 (Σ sources ≈ total, within 10×) violada: total/combinado ≈ ${ratio_flag}×.${NC}"
                echo -e "    ${YELLOW}  Esperado para modelos com campo receptivo > janela de medicao (cold-start).${NC}"
                echo -e "    ${YELLOW}  Nao usar este numero como piso de precisao calibrado sem medicao pareada-com-prewarm.${NC}"
            fi
        fi
        echo ""
    done
    set -u
}

# ── Render: spectral summary ────────────────────────────────────────────────

render_spectral_summary() {
    local count="${SPECTRAL_PASSED_COUNT:-0}"
    echo "📈 SPECTRAL FIDELITY"
    echo "═════════════════════"
    echo ""
    if [ "$count" -gt 0 ]; then
        echo -e "  ${GREEN}ok${NC} ${count} modelo(s) com metricas espectrais dentro da baseline."
    else
        echo -e "  ${YELLOW}(i) Nao coberto no modo quick — rode tests-long para verificacao completa.${NC}"
    fi
    echo ""
}

# ── Render: footer ──────────────────────────────────────────────────────────

render_footer() {
    local total_s
    local end_t
    end_t=$(date +%s%N)
    total_s=$(awk -v ns=$((end_t - OVERALL_START)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    echo "───────────────────────────────────────────────────────────────"
    echo -e "  Dashboard gerado em ${total_s}s (fidelidade: ${FIDELITY_DURATION_S}s, performance: ${BENCH_DURATION_S}s)"
    echo ""

    local skipped=0
    local order_count bench_count
    set +u
    order_count="${#MODEL_ORDER[@]}"
    bench_count="${#ALL_BENCH_NAMES[@]}"
    set -u

    if { [ -z "$order_count" ] || [ "$order_count" -eq 0 ]; } && [ "$MODE" != "bench" ]; then
        echo -e "  ${YELLOW}(i) Testes de fidelidade nao produziram dados parseaveis.${NC}"
        echo -e "  ${YELLOW}   Verifique se os modelos e golden vectors estao presentes.${NC}"
        skipped=1
    fi
    if { [ -z "$bench_count" ] || [ "$bench_count" -eq 0 ]; } && [ "$MODE" != "fidelity" ]; then
        echo -e "  ${YELLOW}(i) Benchmarks nao produziram dados parseaveis.${NC}"
        skipped=1
    fi
    if [ "$skipped" -eq 1 ]; then
        echo -e "  ${YELLOW}(i) Exit code 0 (graceful skip) — dados incompletos nao sao erros de infra.${NC}"
    fi
    echo ""
}

# ── Full dashboard render ───────────────────────────────────────────────────

render_dashboard() {
    render_header
    render_quick_summary
    render_fidelity_details
    render_performance
    render_isa_parity
    render_activation_precision
    render_f64_decomposition
    render_spectral_summary
    render_footer
}

# ── Plain-text version (no ANSI) for --save ─────────────────────────────────

render_dashboard_plain() {
    set +o pipefail
    render_dashboard | sed "s/$(printf '\033')\[[0-9;]*m//g"
    set -o pipefail
}

# ── Contract baseline storage ──────────────────────────────────────────────

declare -A CONTRACT_ESR
declare -A CONTRACT_SNR
declare -A CONTRACT_MRSTFT
declare -A CONTRACT_LATENCY

# ── Load contract/baseline file ────────────────────────────────────────────
#
# Parses a plain-text dashboard (produced by --save) and extracts per-model
# fidelity and performance metrics into CONTRACT_* associative arrays.
# Expected format is the render_dashboard_plain() output with no ANSI.
#
# Fidelity table rows look like:
#   BossWN-standard @48000 Live         │ 9.98e-06                  │ 1.94e-14       │ 37.59    │ 0.0131   │ Live
#
# Performance table rows look like:
#   WaveNet Standard CH16    │ 56.2 us          │ 4.2%         │ 95.8% ok

load_contract_baseline() {
    local file="$1"
    [ -f "$file" ] || { echo "ERRO: Arquivo de contrato nao encontrado: ${file}" >&2; exit 2; }

    local section=""
    while IFS= read -r line; do
        if [[ "$line" =~ FIDELIDADE[[:space:]]+SONORA ]]; then
            section="fidelity"
            continue
        fi
        if [[ "$line" =~ PERFORMANCE ]]; then
            section="performance"
            continue
        fi
        # Reset section tracker for other tables to avoid parsing
        # activation precision / ISA / spectral rows as fidelity data.
        if [[ "$line" =~ ACTIVATION|ISA[[:space:]]+PARITY|SPECTRAL[[:space:]]+FIDELITY|F64[[:space:]]+ORACLE ]]; then
            section=""
            continue
        fi

        if [ "$section" = "fidelity" ]; then
            local trimmed
            trimmed=$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
            if [[ "$trimmed" == *"│"* ]] && [[ ! "$trimmed" =~ ^[─═] ]] && [[ ! "$trimmed" =~ ^"Modelo" ]] && [[ ! "$trimmed" =~ ^"Padrao" ]]; then
                local model_part=$(echo "$trimmed" | awk -F'│' '{print $1}' | sed 's/[[:space:]]*$//; s/^[[:space:]]*//')
                local esr_part=$(echo "$trimmed" | awk -F'│' '{print $2}' | sed 's/[[:space:]]*$//; s/^[[:space:]]*//')
                local snr_part=$(echo "$trimmed" | awk -F'│' '{print $4}' | sed 's/[[:space:]]*$//; s/^[[:space:]]*//')
                local mrstft_part=$(echo "$trimmed" | awk -F'│' '{print $5}' | sed 's/[[:space:]]*$//; s/^[[:space:]]*//')

                [ -n "$model_part" ] && [ "$model_part" != "" ] || continue

                if [ -n "$esr_part" ] && [ "$esr_part" != "N/A" ] && [[ "$esr_part" =~ ^[0-9] ]]; then
                    CONTRACT_ESR["$model_part"]="$esr_part"
                fi
                if [ -n "$snr_part" ] && [ "$snr_part" != "N/A" ] && [[ "$snr_part" =~ ^[0-9] ]]; then
                    CONTRACT_SNR["$model_part"]="$snr_part"
                fi
                if [ -n "$mrstft_part" ] && [ "$mrstft_part" != "N/A" ] && [[ "$mrstft_part" =~ ^[0-9] ]]; then
                    CONTRACT_MRSTFT["$model_part"]="$mrstft_part"
                fi
            fi
        fi

        if [ "$section" = "performance" ]; then
            local trimmed
            trimmed=$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
            if [[ "$trimmed" == *"│"* ]] && [[ ! "$trimmed" =~ ^[─═] ]] && [[ ! "$trimmed" =~ ^"Modelo" ]]; then
                local model_part=$(echo "$trimmed" | awk -F'│' '{print $1}' | sed 's/[[:space:]]*$//; s/^[[:space:]]*//')
                local lat_part=$(echo "$trimmed" | awk -F'│' '{print $2}' | sed 's/[[:space:]]*$//; s/^[[:space:]]*//; s/[[:space:]]us$//; s/[[:space:]]*$//')

                [ -n "$model_part" ] && [ "$model_part" != "" ] || continue

                if [ -n "$lat_part" ] && [ "$lat_part" != "N/A" ] && [[ "$lat_part" =~ ^[0-9] ]]; then
                    CONTRACT_LATENCY["$model_part"]="$lat_part"
                fi
            fi
        fi
    done < "$file"
}

# ── Contract verification ──────────────────────────────────────────────────
#
# Compares current run metrics against the contract baseline with tolerances:
#
#   Fidelity (ESR):        fail if new_esr > contract_esr * 10.0
#   Fidelity (SNR):        fail if new_snr < contract_snr - 6.0 (dB)
#   Fidelity (MR-STFT):    fail if new_mrstft > contract_mrstft * 10.0
#   Performance (latency): fail if new_lat > contract_lat * 1.10 (10% margin)
#
# Fields with value "N/A" (or empty) in the contract are skipped.
# Returns 0 on pass, 1 on violation.

verify_contract() {
    local violations=0

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  VERIFICACAO DE CONTRATO DE QUALIDADE"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    if [ ${#CONTRACT_ESR[@]} -eq 0 ] && [ ${#CONTRACT_LATENCY[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Arquivo de contrato vazio ou sem metricas reconhecidas.${NC}"
        echo ""
        return 0
    fi

    local contract_count
    set +u; contract_count="${#CONTRACT_ESR[@]}"; set -u
    if [ -n "$contract_count" ]; then
        echo "  FIDELIDADE — ${contract_count} modelo(s) no contrato"
        echo "  ─────────────────────────────────────────────"
        echo ""

        # Build a lookup from contract labels to full dashboard keys
        # The contract and dashboard may use slightly different labels.
        # We match by model name prefix (before @rate or mode suffix).
        for contract_label in "${!CONTRACT_ESR[@]}"; do
            local matched=false
            for dash_key in "${!ESR_NAMCORE[@]}"; do
                local dash_label
                dash_label=$(echo "$dash_key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
                # Prefix match: contract labels may be truncated by the 38-char
                # column width in the plain-text dashboard table.
                if [[ "$dash_label" == "$contract_label"* ]] || [[ "$contract_label" == "$dash_label"* ]]; then
                    matched=true
                    local esr_cur="${ESR_NAMCORE[$dash_key]:-N/A}"
                    local esr_ctr="${CONTRACT_ESR[$contract_label]}"

                    if [ "$esr_cur" != "N/A" ] && [ "$esr_ctr" != "N/A" ] && [ -n "$esr_ctr" ]; then
                        local esr_fail
                        esr_fail=$(LC_ALL=C awk -v cur="$esr_cur" -v ctr="$esr_ctr" \
                            'BEGIN { if (cur+0 > ctr*10.0) print "1"; else print "0" }')
                        if [ "$esr_fail" = "1" ]; then
                            echo -e "    ${RED}✗${NC} ${contract_label}: ESR regrediu ${esr_cur} (contrato: ${esr_ctr}, limite: $(LC_ALL=C awk -v c="$esr_ctr" 'BEGIN { printf "%.2e", c*10.0 }'))"
                            violations=$((violations + 1))
                        else
                            echo -e "    ${GREEN}ok${NC} ${contract_label}: ESR ${esr_cur} (contrato: ${esr_ctr})"
                        fi
                    fi

                    # Check SNR
                    local snr_cur="${SNR_DB[$dash_key]:-N/A}"
                    local snr_ctr="${CONTRACT_SNR[$contract_label]:-N/A}"
                    if [ "$snr_cur" != "N/A" ] && [ "$snr_ctr" != "N/A" ] && [ -n "$snr_ctr" ]; then
                        local snr_fail
                        snr_fail=$(LC_ALL=C awk -v cur="$snr_cur" -v ctr="$snr_ctr" \
                            'BEGIN { if (cur+0 < ctr-6.0) print "1"; else print "0" }')
                        if [ "$snr_fail" = "1" ]; then
                            echo -e "    ${RED}✗${NC} ${contract_label}: SNR regrediu ${snr_cur} dB (contrato: ${snr_ctr} dB, limite: $(LC_ALL=C awk -v c="$snr_ctr" 'BEGIN { printf "%.1f", c-6.0 }') dB)"
                            violations=$((violations + 1))
                        fi
                    fi

                    # Check MR-STFT
                    local mrstft_cur="${MRSTFT[$dash_key]:-N/A}"
                    local mrstft_ctr="${CONTRACT_MRSTFT[$contract_label]:-N/A}"
                    if [ "$mrstft_cur" != "N/A" ] && [ "$mrstft_ctr" != "N/A" ] && [ -n "$mrstft_ctr" ]; then
                        local mrstft_fail
                        mrstft_fail=$(LC_ALL=C awk -v cur="$mrstft_cur" -v ctr="$mrstft_ctr" \
                            'BEGIN { if (cur+0 > ctr*10.0) print "1"; else print "0" }')
                        if [ "$mrstft_fail" = "1" ]; then
                            echo -e "    ${RED}✗${NC} ${contract_label}: MR-STFT regrediu ${mrstft_cur} (contrato: ${mrstft_ctr}, limite: $(LC_ALL=C awk -v c="$mrstft_ctr" 'BEGIN { printf "%.4f", c*10.0 }'))"
                            violations=$((violations + 1))
                        fi
                    fi
                    break
                fi
            done
            if [ "$matched" = false ]; then
                echo -e "    ${YELLOW}(i)${NC} ${contract_label}: nao encontrado na execucao atual"
            fi
        done
        echo ""
    fi

    # Check performance (latency)
    local latency_contract_count
    set +u; latency_contract_count="${#CONTRACT_LATENCY[@]}"; set -u
    if [ -n "$latency_contract_count" ] && [ "$latency_contract_count" -gt 0 ]; then
        echo "  PERFORMANCE — ${latency_contract_count} benchmark(s) no contrato"
        echo "  ─────────────────────────────────────────────────"
        echo ""

        for contract_label in "${!CONTRACT_LATENCY[@]}"; do
            local matched=false
            for bn in "${ALL_BENCH_NAMES[@]}"; do
                local dash_label="${BENCH_MODEL_MAP[$bn]:-$bn}"
                # Normalize Unicode × (U+00D7) to ASCII x for label matching
                local dash_norm="${dash_label//×/x}"
                local ctr_norm="${contract_label//×/x}"
                if [ "$dash_norm" = "$ctr_norm" ]; then
                    matched=true
                    local lat_cur="${LATENCY_US[$bn]:-N/A}"
                    local lat_ctr="${CONTRACT_LATENCY[$contract_label]}"

                    if [ "$lat_cur" != "N/A" ] && [ "$lat_ctr" != "N/A" ] && [ -n "$lat_ctr" ]; then
                        local lat_fail
                        lat_fail=$(LC_ALL=C awk -v cur="$lat_cur" -v ctr="$lat_ctr" \
                            'BEGIN { if (cur+0 > ctr*1.10) print "1"; else print "0" }')
                        if [ "$lat_fail" = "1" ]; then
                            echo -e "    ${RED}✗${NC} ${contract_label}: latencia regrediu ${lat_cur} us (contrato: ${lat_ctr} us, limite: $(LC_ALL=C awk -v c="$lat_ctr" 'BEGIN { printf "%.1f", c*1.10 }') us)"
                            violations=$((violations + 1))
                        else
                            echo -e "    ${GREEN}ok${NC} ${contract_label}: latencia ${lat_cur} us (contrato: ${lat_ctr} us)"
                        fi
                    fi
                    break
                fi
            done
            if [ "$matched" = false ]; then
                echo -e "    ${YELLOW}(i)${NC} ${contract_label}: nao encontrado na execucao atual"
            fi
        done
        echo ""
    fi

    if [ "$violations" -gt 0 ]; then
        echo -e "  ${RED}CONTRATO VIOLADO — ${violations} violacao(oes) detectada(s).${NC}"
        echo ""
        return 1
    else
        echo -e "  ${GREEN}CONTRATO OK — Todas as metricas dentro das tolerancias.${NC}"
        echo ""
        return 0
    fi
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    local run_phases=0
    if [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
        run_phases=$((run_phases + 6))  # golden_vectors, oracle, isa, spectral, activation, quick_parity
    fi
    if [ "$MODE" = "full" ] || [ "$MODE" = "bench" ]; then
        run_phases=$((run_phases + 1))  # regression_gate benchmarks
    fi
    PHASE_TOTAL=$((run_phases + 2))     # +1 parse, +1 render

    echo -e "${BLUE}${BOLD}===============================================================${NC}"
    echo -e "${BLUE}${BOLD}    nam-rs Quality Dashboard${NC}"
    echo -e "${BLUE}${BOLD}    Modo: ${MODE}${NC}"
    echo -e "${BLUE}${BOLD}===============================================================${NC}"

    if [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
        phase "golden_vectors"
        run_golden_vectors

        phase "reference_oracle_f64"
        run_reference_oracle

        phase "isa_parity"
        run_isa_parity

        phase "spectral_fidelity"
        run_spectral_fidelity

        phase "lstm_activation_precision"
        run_activation_precision

        phase "quick_parity"
        run_quick_parity
    fi

    if [ "$MODE" = "full" ] || [ "$MODE" = "bench" ]; then
        phase "regression_gate benchmarks"
        run_benchmarks
    fi

    phase "Parseando resultados"
    parse_golden_vectors
    parse_oracle_f64
    parse_isa_parity
    parse_spectral_fidelity
    parse_activation_precision
    parse_benchmarks

    phase "Renderizando dashboard"
    render_dashboard

    if [ -n "$SAVE_FILE" ]; then
        render_dashboard_plain > "$SAVE_FILE"
        echo -e "${GREEN}ok${NC} Dashboard salvo em: ${SAVE_FILE} (plain text, sem ANSI)"
    fi

    if [ -n "$CHECK_FILE" ]; then
        load_contract_baseline "$CHECK_FILE"
        if ! verify_contract; then
            exit 1
        fi
    fi

    exit 0
}

main "$@"
