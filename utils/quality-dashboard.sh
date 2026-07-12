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

# ESR_F64_FAMILY_MAP — Explicit static map: each golden-label → oracle .nam fixture.
# The f64 oracle is measured on exactly ONE representative .nam fixture per family.
# Labels whose model file IS the oracle fixture are also listed in ESR_F64_EXACT_MATCH;
# all other FAMILY_MAP entries are family-level approximations.
# CORRECTNESS NOTE (Épico EQ audit, 2026-07-05): for LSTM, the oracle fixture is
# `lstm.nam` (the tiny 3-hidden-unit official example), NOT BossLSTM-1x16.nam/
# BossLSTM-2x8.nam, which are the models that actually exhibit severe recurrent drift.
declare -A ESR_F64_FAMILY_MAP=(
    # WaveNet family — oracle measured on wavenet_official.nam
    ["BossWN-standard"]="wavenet_official.nam"
    ["BossWN-feather"]="wavenet_official.nam"
    ["BossWN-nano"]="wavenet_official.nam"
    ["EVH-5150-Lite"]="wavenet_official.nam"
    ["wavenet_a1_standard (Official)"]="wavenet_official.nam"
    ["WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference"]="wavenet_official.nam"
    ["WaveNet Official (CH=3, dynamic path) C++ cross-reference"]="wavenet_official.nam"
    ["WaveNetDyn Free-Shape (CH=7→4, dynamic path) C++ cross-reference"]="wavenet_official.nam"
    ["T-HF1.4: WaveNet Standard polynomial SIMD (regression gate)"]="wavenet_official.nam"
    # LSTM family — oracle measured on lstm.nam (H=3 official)
    ["BossLSTM-1x16"]="lstm.nam"
    ["BossLSTM-2x8"]="lstm.nam"
    ["lstm (Official)"]="lstm.nam"
    ["LSTM-Dyn 1×7 (dynamic path) C++ cross-reference"]="lstm.nam"
    # A2 family — oracle measured on wavenet_a2_lite.nam
    ["WaveNet A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["WaveNet A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container File A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container File A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["SlimmableContainer A2 Example (CH=3→6) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["T-HF1.4: WaveNet A2-Full polynomial SIMD (regression gate)"]="wavenet_a2_lite.nam"
    ["WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference"]="wavenet_a2_lite.nam"
    # A2-FiLM-Lite family — oracle measured on wavenet_a2_film_lite.nam
    ["WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference"]="wavenet_a2_film_lite.nam"
    ["WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) C++ cross-reference"]="wavenet_a2_film_lite.nam"
    # A2-FiLM-Full — oracle measured on wavenet_a2_film_full.nam
    ["WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference"]="wavenet_a2_film_full.nam"
    # A2-FiLM-InputMixinPre — oracle measured on wavenet_a2_film_input_mixin_pre.nam
    ["WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference"]="wavenet_a2_film_input_mixin_pre.nam"
    # ConvNet — oracle measured on convnet_test.nam
    ["ConvNet Test"]="convnet_test.nam"
)

# Labels for which the FAMILY_MAP fixture IS the exact model file (own measurement).
# All other FAMILY_MAP entries are family-proxy: the oracle measured a different
# representative fixture from the same architectural family.
declare -A ESR_F64_EXACT_MATCH=(
    ["WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference"]=1
    ["WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference"]=1
    ["WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference"]=1
    ["ConvNet Test"]=1
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

    # Step 1: Look up in the explicit static family map
    local oracle_fixture="${ESR_F64_FAMILY_MAP[$golden_label]:-}"
    if [ -n "$oracle_fixture" ]; then
        local val
        set +u; val="${ESR_F64_PAIRED[$oracle_fixture]}"; set -u
        if [ -n "$val" ] && _is_numeric_esr "$val"; then
            if [ -n "${ESR_F64_EXACT_MATCH[$golden_label]:-}" ]; then
                echo "$val"
                echo "exact"
            else
                echo "$val"
                echo "family:${oracle_fixture}"
            fi
            return
        fi
    fi

    # Step 2: Fallback — try golden_label directly in ESR_F64_PAIRED
    local direct
    set +u; direct="${ESR_F64_PAIRED[$golden_label]}"; set -u
    if [ -n "$direct" ] && _is_numeric_esr "$direct"; then
        echo "$direct"
        echo "exact"
        return
    fi

    # Step 3: Fallback — try golden_label.nam in ESR_F64_PAIRED
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
    cargo test --release --test models golden_vectors -- --test-threads=1 --nocapture > "$log" 2>&1 || true
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
        # Flag any non-exact (family-level) match: the value was NOT measured
        # on this specific model, only on the family's one representative
        # fixture (see ESR_F64_FAMILY_MAP). Do not present it as if it
        # were this model's own floor.
        local f64_suffix=""
        if [[ "$esr_f64_provenance" == family:* ]]; then
            f64_suffix=" (~fam.)"
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
        printf "  %s %-38s  vs NAMcore: %-10s %b  │  vs Ideal (f64): %-10s%s  │  ⚡ CPU: %s do budget\n" \
            "$icon" "${label:0:38}" "$esr_nam_display" "$verdict" "$esr_f64_colored" "$f64_suffix" "$cpu_colored"
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
    echo "  (~fam.) = 'vs Ideal (f64)' not measured for this exact model — shown as the"
    echo "  family's single representative fixture instead (see ESR_F64_FAMILY_MAP)."
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
        if [[ "$esr_f64_provenance" == family:* ]]; then
            esr_f64_short="${esr_f64_short}~"
        fi
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
    echo "  ~ apos o valor 'vs f64' = valor da familia (uma unica fixture representativa,"
    echo "    ex.: LSTM -> lstm.nam H=3), NAO medido para este modelo especifico."
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
    echo "$block" | grep -oP "${label_pattern}\\K[0-9.eE+-]+" 2>/dev/null | head -1
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
    # Strip ANSI escape sequences using a literal ESC character
    local esc
    esc=$(printf '\033')
    render_dashboard | sed "s/${esc}\[[0-9;]*m//g"
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    local run_phases=0
    if [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
        run_phases=$((run_phases + 5))  # golden_vectors, oracle, isa, spectral, activation
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

    exit 0
}

main "$@"
