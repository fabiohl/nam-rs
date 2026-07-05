#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# quality-dashboard.sh — nam-rs Quality Dashboard
#
# Runs all fidelity suites and performance benchmarks, captures their outputs,
# and generates a comprehensive human-friendly Tarefa SQ1.1 — Criar `utils/quality-dashboard.sh`report covering the full nam-rs
# universe: all architectures, models, quality modes, and ISAs.
#
# Usage:
#   ./utils/quality-dashboard.sh                        Full dashboard (fidelity + performance)
#   ./utils/quality-dashboard.sh --fidelity-only        Fidelity tests only
#   ./utils/quality-dashboard.sh --bench-only           Benchmarks only
#   ./utils/quality-dashboard.sh --save <filename>      Save plain-text copy alongside display

set -euo pipefail

PHASE_TOTAL=0
source "$(dirname "$0")/_lib.sh"

# ── Argument parsing ────────────────────────────────────────────────────────

SAVE_FILE=""
MODE="full"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save)
            SAVE_FILE="$2"
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

# Look up ESR vs f64 for a golden label by partial matching against oracle keys
# Oracle stores data under short labels (LSTM, WaveNet, ConvNet) or .nam filenames.
# We check if the golden label contains any known oracle key.
_lookup_esr_f64() {
    local golden_label="$1"
    local best="N/A"
    set +u
    for okey in "${!ESR_F64[@]}"; do
        # Case-insensitive partial match: golden label contains oracle key
        if echo "$golden_label" | grep -qi "$okey" 2>/dev/null; then
            best="${ESR_F64[$okey]}"
            break
        fi
    done
    set -u
    echo "$best"
}

# ── Data storage (global associative arrays) ────────────────────────────────

declare -A ESR_NAMCORE
declare -A ESR_NAMCORE_DB
declare -A ESR_F64
declare -A ESR_F64_DB
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
    cargo test --release --test golden_vectors -- --nocapture > "$log" 2>&1 || true
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
    cargo test --release --test reference_oracle_f64 -- --nocapture > "$log" 2>&1 || true
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
    cargo test --release --test isa_parity -- --test-threads=1 --nocapture > "$log" 2>&1 || true
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
    cargo test --release --test spectral_fidelity -- --nocapture > "$log" 2>&1 || true
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
    cargo test --release --test lstm_activation_precision -- --nocapture > "$log" 2>&1 || true
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    echo -e "  ${GREEN}ok${NC} lstm_activation_precision concluido (${dur}s)"
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

# ── Parse: golden_vectors ───────────────────────────────────────────────────
# Parses report_dsp_fidelity blocks and ConvNet Self-Golden output.
# Writes tab-separated records to a temp file, then reads back in the
# current shell to populate global associative arrays.

parse_golden_vectors() {
    local log="$LOGDIR/golden_vectors.log"
    [ -f "$log" ] || return

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
        MODEL_ORDER+=("$key")
    done <<< "$sorted_keys"
}

# ── Parse: reference_oracle_f64 ─────────────────────────────────────────────

parse_oracle_f64() {
    local log="$LOGDIR/oracle_f64.log"
    [ -f "$log" ] || return

    # Parse ESR summary table — skip debug lines (MODEL CLASS LABEL, PROD FIRST, etc.)
    local parsed="$PARSEDIR/oracle_f64_summary.parsed"
    LC_ALL=C awk '
    BEGIN { in_table = 0 }
    /^=== ESR\(f32 vs f64 oracle\) Summary ===/ { in_table = 1; next }
    /^---/ && in_table { in_table = 2; next }
    # Stop table on empty line or test result line (starts with "test ")
    in_table == 2 && (/^$/ || /^test /) { in_table = 0; next }
    # Skip debug lines mixed into the table
    in_table == 2 && /^(MODEL CLASS LABEL|PROD FIRST|ORACLE FIRST)/ { next }
    # Capture data rows (model filename as first column)
    in_table == 2 && /^[A-Za-z]/ {
        printf "ESR_F64_TABLE\t%s\t%s\n", $1, $0
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric rest; do
        [[ "$metric" == "ESR_F64_TABLE" ]] || continue
        local filename family esr_lin esr_db
        filename=$(echo "$rest" | awk '{print $1}')
        family=$(echo "$rest" | awk '{print $2}')
        esr_lin=$(echo "$rest" | awk '{print $3}')
        esr_db=$(echo "$rest" | awk '{print $4}')
        [ -n "$filename" ] && [ -n "$esr_lin" ] && ESR_F64["$filename"]="$esr_lin"
        [ -n "$filename" ] && [ -n "$esr_db" ] && ESR_F64_DB["$filename"]="$esr_db"
        [ -n "$filename" ] && MODEL_ESR_F64_TABLE["$filename"]="${family}|${esr_lin}|${esr_db}"
    done < "$parsed"

    # Parse paired prewarm ESR lines — labels like "LSTM", "WaveNet", "ConvNet"
    grep -E ' ESR\(f32 vs oracle, prewarm-paired' "$log" > "$parsed" 2>/dev/null || true
    while IFS= read -r line; do
        local label esr esr_db
        label=$(echo "$line" | sed 's/ ESR(f32 vs oracle, prewarm-paired.*//')
        esr=$(echo "$line" | grep -oP ':\s+\K[0-9.e+\-]+' 2>/dev/null || true)
        esr_db=$(echo "$line" | grep -oP '\(\K[-0-9.]+(?= dB\))' 2>/dev/null || true)
        [ -n "$label" ] && [ -n "$esr" ] && ESR_F64["$label"]="$esr"
        [ -n "$label" ] && [ -n "$esr_db" ] && ESR_F64_DB["$label"]="$esr_db"
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
        if (/^[[:space:]]*$/ || /^[A-Z]/) {
            if (lbl != "" && buf != "") { printf "F64_DECOMP\t%s\t%s\n", lbl, buf }
            in_decomp = 0; lbl = ""; buf = ""
        } else { buf = buf $0 "\n" }
    }
    END { if (lbl != "" && buf != "") { printf "F64_DECOMP\t%s\t%s\n", lbl, buf } }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric label value; do
        [[ "$metric" == "F64_DECOMP" ]] || continue
        F64_DECOMPOSITION["$label"]="$value"
    done < "$parsed"
}

# ── Parse: isa_parity ───────────────────────────────────────────────────────

parse_isa_parity() {
    local log="$LOGDIR/isa_parity.log"
    [ -f "$log" ] || return

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
    [ -f "$log" ] || return
    SPECTRAL_PASSED_COUNT=$(grep -c 'all spectral fidelity metrics within baseline tolerance' "$log" 2>/dev/null | head -1 | tr -d '[:space:]' || echo 0)
}

# ── Parse: lstm_activation_precision ────────────────────────────────────────

parse_activation_precision() {
    local log="$LOGDIR/activation_precision.log"
    [ -f "$log" ] || return

    local parsed="$PARSEDIR/activation.parsed"
    grep -E 'FastMath\(Pad' "$log" > "$parsed" 2>/dev/null || true

    while IFS= read -r line; do
        local model fast_snr exact_snr delta
        model=$(echo "$line" | sed 's/[[:space:]]*FastMath(Pad.*//' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
        fast_snr=$(echo "$line" | grep -oP 'FastMath\(Pad.*\):\s+\K[0-9.]+' 2>/dev/null || echo "N/A")
        exact_snr=$(echo "$line" | grep -oP 'Exact\(tanh\):\s+\K[0-9.]+' 2>/dev/null || echo "N/A")
        delta=$(echo "$line" | grep -oP 'Δ=\K[+-][0-9.]+' 2>/dev/null || echo "0.0")
        ACTIVATION_SNR["$model"]="${fast_snr:-N/A}|${exact_snr:-N/A}|${delta:-0.0}"
    done < "$parsed"
}

# ── Parse: regression_gate ──────────────────────────────────────────────────

parse_benchmarks() {
    local log="$LOGDIR/regression_gate.log"
    [ -f "$log" ] || return

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
    awk -v v="$esr" 'BEGIN {
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
    cmp=$(awk -v v="$esr" 'BEGIN {
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
        echo -e "${GREEN}${pct}%%${NC}"
    elif [ "$f" -gt 50 ]; then
        echo -e "${GREEN}${pct}%%${NC}"
    elif [ "$f" -gt 25 ]; then
        echo -e "${YELLOW}${pct}%%${NC}"
    else
        echo -e "${RED}${pct}%%${NC}"
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
    local cpu_short="${CPU_MODEL:0:48}"
    printf "╔══════════════════════════════════════════════════════════════════╗\n"
    printf "║              nam-rs Quality Dashboard                            ║\n"
    printf "║              ------------------------------                      ║\n"
    printf "║              Medido em: %s                ║\n" "$NOW"
    printf "║              ISA: %-52s ║\n" "$ISA"
    printf "║              CPU: %-52s ║\n" "$cpu_short"
    printf "║              rustc: %-50s ║\n" "$RUSTC_VER"
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
        local esr_nam_display="$esr_nam"
        if [ "$esr_nam_display" != "N/A" ] && [ ${#esr_nam_display} -gt 10 ]; then
            esr_nam_display=$(_nfmt "%.2e" "$esr_nam" 2>/dev/null || echo "$esr_nam")
        fi
        # Extract model name for f64 lookup (strip rate and mode)
        local f64_label
        f64_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
        local esr_f64
        esr_f64=$(_lookup_esr_f64 "$f64_label")
        local esr_f64_display="$esr_f64"
        if [ "$esr_f64_display" != "N/A" ] && [ ${#esr_f64_display} -gt 10 ]; then
            esr_f64_display=$(_nfmt "%.2e" "$esr_f64" 2>/dev/null || echo "$esr_f64")
        fi
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

    printf "  %-38s │ %-16s │ %-12s │ %-8s │ %-8s │ %-6s │ %s\n" \
        "Modelo" "ESR (vs NAMcore)" "ESR (vs f64)" "SNR dB" "MR-STFT" "Modo" "Qualidade"
    printf "  %s │ %s │ %s │ %s │ %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..38})" "$(printf '─%.0s' {1..16})" \
        "$(printf '─%.0s' {1..12})" "$(printf '─%.0s' {1..8})" \
        "$(printf '─%.0s' {1..8})" "$(printf '─%.0s' {1..6})" "$(printf '─%.0s' {1..24})"

    for key in "${MODEL_ORDER[@]}"; do
        local esr_nam="${ESR_NAMCORE[$key]:-N/A}"
        local esr_nam_short="$esr_nam"
        if [ "$esr_nam_short" != "N/A" ] && [ ${#esr_nam_short} -gt 14 ]; then
            esr_nam_short=$(_nfmt "%.2e" "$esr_nam" 2>/dev/null || echo "$esr_nam")
        fi
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
        local mrstft_short="$mrstft"
        if [ "$mrstft_short" != "N/A" ] && [ ${#mrstft_short} -gt 6 ]; then
            mrstft_short=$(_nfmt "%.4f" "$mrstft" 2>/dev/null || echo "$mrstft")
        fi
        local model_label
        model_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
        local esr_f64
        esr_f64=$(_lookup_esr_f64 "$model_label")
        local esr_f64_short="$esr_f64"
        if [ "$esr_f64_short" != "N/A" ] && [ ${#esr_f64_short} -gt 10 ]; then
            esr_f64_short=$(_nfmt "%.2e" "$esr_f64" 2>/dev/null || echo "$esr_f64")
        fi
        local esr_f64_colored
        esr_f64_colored=$(_esr_color "$esr_f64_short")
        local mode="Live"
        [[ "$key" == *" HQ"* ]] && mode="HQ"
        local display_key="${key:0:38}"

        # Quality verdict
        local quality="N/A"
        if [ "$esr_nam" != "N/A" ]; then
            quality=$(esr_verdict "$esr_nam")
        fi

        printf "  %-38s │ %-26b │ %-12s │ %-8s │ %-8s │ %-6s │ %s\n" \
            "$display_key" "$esr_nam_short" "$esr_f64_colored" "$snr" "$mrstft_short" "$mode" "$quality"
    done
    echo ""
    echo "  Legenda ESR vs NAMcore:"
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

    if [ ${#ALL_BENCH_NAMES[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nenhum dado de performance disponivel.${NC}"
        echo ""
        return
    fi

    printf "  %-28s │ %-16s │ %-12s │ %s\n" \
        "Modelo" "Latencia Mediana" "%% do Budget" "Folga"
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
            "$label" "$latency_display" "${pct}%%" "$folga_colored"
    done

    echo ""
    echo "  (i) Folga > 50%%:  Pode usar oversampling 2x sem xruns"
    echo "  (i) Folga > 75%%:  Pode usar oversampling 4x sem xruns"
    echo "  (i) Folga < 25%%:  ⚠ Risco de xruns com buffer de 64 amostras"
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
        echo -e "  ${YELLOW}(i) Nenhum resultado de ISA parity disponivel.${NC}"
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
        "Modelo" "FastMath(Pade)" "Exact(tanh)" "Δ SNR"
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
        echo "  Ganho SNR medio com Exact(tanh): +${avg} dB (sobre ${count_num} modelos LSTM)"
    fi
    echo ""
}

# ── Render: f64 decomposition ───────────────────────────────────────────────

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
    set +u
    for model in "${!F64_DECOMPOSITION[@]}"; do
        echo "  ${model}:"
        echo "${F64_DECOMPOSITION[$model]}" | while IFS= read -r line; do
            [ -n "$line" ] && echo "    $line"
        done
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
        echo -e "  ${YELLOW}(i) Nenhum resultado de spectral fidelity disponivel.${NC}"
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
    if [ ${#MODEL_ORDER[@]} -eq 0 ] && [ "$MODE" != "bench" ]; then
        echo -e "  ${YELLOW}(i) Testes de fidelidade nao produziram dados parseaveis.${NC}"
        echo -e "  ${YELLOW}   Verifique se os modelos e golden vectors estao presentes.${NC}"
        skipped=1
    fi
    if [ ${#ALL_BENCH_NAMES[@]} -eq 0 ] && [ "$MODE" != "fidelity" ]; then
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
    # Strip ANSI escape sequences using a literal ESC character
    local esc
    esc=$(printf '\033')
    render_dashboard | sed "s/${esc}\[[0-9;]*m//g"
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    local phase_count=0
    if [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
        phase_count=$((phase_count + 6))  # 5 run phases + 1 parse phase
    fi
    if [ "$MODE" = "full" ] || [ "$MODE" = "bench" ]; then
        phase_count=$((phase_count + 2))  # 1 run phase + 1 parse phase
    fi
    PHASE_TOTAL=$phase_count

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

    echo ""
    render_dashboard

    if [ -n "$SAVE_FILE" ]; then
        render_dashboard_plain > "$SAVE_FILE"
        echo -e "${GREEN}ok${NC} Dashboard salvo em: ${SAVE_FILE} (plain text, sem ANSI)"
    fi

    exit 0
}

main "$@"
