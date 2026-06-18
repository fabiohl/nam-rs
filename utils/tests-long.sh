#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Intensive long-duration verification suite for nam-rs.
# Performs numerical soak tests, proptest fuzzing, NeuralAmpModelerCore parity checks,
# CLAP release compliance, multi-instance stress tests, and long-running performance benchmarks.

set -uo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Setup defensive error trap
trap 'echo -e "\n${RED}${BOLD}❌ Erro inesperado: Comando \"$BASH_COMMAND\" falhou na linha $LINENO com status $?. Abortando suíte de testes.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}==========================================================================${NC}"
echo -e "${BLUE}${BOLD}    nam-rs Long-Duration Stress & Audit Suite (± 38 minutes - cold run)   ${NC}"
echo -e "${BLUE}${BOLD}==========================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Setup target logs
rm -rf target/logs/
mkdir -p target/logs/

# Cleanup accumulated live-test artifacts from previous runs (41+ MB WAVs)
rm -rf tests/fixtures/.temp_live/

# Verify NeuralAmpModelerCore presence and version (pinned commit).
# Version pinned commit is defined in mod-update.sh as source of truth.
NAM_CORE_PINNED_COMMIT="9c7b185de346fe0725dea537bcee4bc38b5bb6d6" # v0.5.3 (canonical)
if [ ! -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    echo -e "${RED}${BOLD}❌ NeuralAmpModelerCore não encontrado em tests/fixtures/NeuralAmpModelerCore.${NC}"
    echo -e "${YELLOW}Por favor, execute './utils/mod-update.sh' para clonar e configurar as dependências.${NC}"
    exit 1
fi

CURRENT_CORE_SHA=$(cd tests/fixtures/NeuralAmpModelerCore && git rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_CORE_SHA" != "$NAM_CORE_PINNED_COMMIT" ]; then
    echo -e "${RED}${BOLD}❌ Versão incorreta do NeuralAmpModelerCore (instalado: $CURRENT_CORE_SHA, esperado: $NAM_CORE_PINNED_COMMIT).${NC}"
    echo -e "${YELLOW}Por favor, execute './utils/mod-update.sh' para ressincronizar as dependências.${NC}"
    exit 1
fi

# ── Phase 0: Pre-flight check — C++ toolchain & golden files ──
echo -e "\n${BLUE}${BOLD}[Phase 0] Pre-flight: verificando pré-requisitos C++ e golden vectors...${NC}"

MISSING_GOLDENS=()
REQUIRED_CABSIM_GOLDENS=(
    "tests/fixtures/golden_cabsim_cpp_short.bin"
    "tests/fixtures/golden_cabsim_cpp_medium.bin"
    "tests/fixtures/golden_cabsim_cpp_long.bin"
)
# v1 golden vectors (48 kHz only)
REQUIRED_GOLDEN_MODELS=(
    "wavenet_standard" "wavenet_feather" "wavenet_nano" "wavenet_lite"
    "wavenet_a1_standard" "wavenet_a2_full" "wavenet_a2_lite"
    "lstm_1x16" "lstm_2x8" "lstm_official"
)
# v2 ALL_SR: 44100, 48000, 88200, 96000, 192000
V2_ALL_SR_MODELS=("wavenet_feather" "wavenet_nano" "wavenet_a1_standard" "wavenet_lite")
V2_ALL_SR=(44100 48000 88200 96000 192000)
# v2 SR_EX_192K: 44100, 48000, 88200, 96000
V2_EX_192K_MODELS=("lstm_1x16" "lstm_2x8")
V2_EX_192K=(44100 48000 88200 96000)
# v2 SR_48K_ONLY: 48000
V2_48K_MODELS=("wavenet_standard" "lstm_official" "wavenet_a2_full" "wavenet_a2_lite")

# Check cabsim goldens
for g in "${REQUIRED_CABSIM_GOLDENS[@]}"; do
    if [ ! -f "$g" ]; then
        MISSING_GOLDENS+=("$g")
    fi
done

# Check v1 goldens
for m in "${REQUIRED_GOLDEN_MODELS[@]}"; do
    g="tests/fixtures/golden_${m}.bin"
    if [ ! -f "$g" ]; then
        MISSING_GOLDENS+=("$g")
    fi
done

# Check v2 golden files per model-specific SR groups (matching golden_vectors.rs constants)
for m in "${V2_ALL_SR_MODELS[@]}"; do
    for sr in "${V2_ALL_SR[@]}"; do
        g="tests/fixtures/golden_${m}_v2_${sr}.bin"
        if [ ! -f "$g" ]; then
            MISSING_GOLDENS+=("$g")
        fi
    done
done
for m in "${V2_EX_192K_MODELS[@]}"; do
    for sr in "${V2_EX_192K[@]}"; do
        g="tests/fixtures/golden_${m}_v2_${sr}.bin"
        if [ ! -f "$g" ]; then
            MISSING_GOLDENS+=("$g")
        fi
    done
done
for m in "${V2_48K_MODELS[@]}"; do
    g="tests/fixtures/golden_${m}_v2_48000.bin"
    if [ ! -f "$g" ]; then
        MISSING_GOLDENS+=("$g")
    fi
done

# Check C++ toolchain availability
MISSING_TOOLS=()
command -v cmake >/dev/null 2>&1 || MISSING_TOOLS+=("cmake")
command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || MISSING_TOOLS+=("g++/clang++ (C++20)")

if [ ${#MISSING_GOLDENS[@]} -gt 0 ] || [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "${RED}${BOLD}❌ Pre-flight falhou — pré-requisitos ausentes:${NC}"
    if [ ${#MISSING_GOLDENS[@]} -gt 0 ]; then
        echo -e "  ${YELLOW}Golden vectors faltando (${#MISSING_GOLDENS[@]} arquivo(s)):${NC}"
        for g in "${MISSING_GOLDENS[@]}"; do
            echo "    - $g"
        done
    fi
    if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
        echo -e "  ${YELLOW}Ferramentas C++ faltando:${NC}"
        for t in "${MISSING_TOOLS[@]}"; do
            echo "    - $t"
        done
    fi

    if [ "${NAM_AUTO_BUILD_GOLDENS:-0}" = "1" ]; then
        echo -e "\n${YELLOW}${BOLD}→ NAM_AUTO_BUILD_GOLDENS=1 — invocando golden_gen_build.sh automaticamente...${NC}"
        if ! bash tests/fixtures/golden_gen_build.sh; then
            echo -e "${RED}${BOLD}❌ golden_gen_build.sh falhou. Corrija as dependências e tente novamente.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Golden vectors regenerados com sucesso.${NC}"
        # Re-validate golden files after generation
        MISSING_GOLDENS=()
        for g in "${REQUIRED_CABSIM_GOLDENS[@]}"; do
            [ ! -f "$g" ] && MISSING_GOLDENS+=("$g")
        done
        for m in "${REQUIRED_GOLDEN_MODELS[@]}"; do
            g="tests/fixtures/golden_${m}.bin"
            [ ! -f "$g" ] && MISSING_GOLDENS+=("$g")
        done
        for m in "${V2_ALL_SR_MODELS[@]}"; do
            for sr in "${V2_ALL_SR[@]}"; do
                g="tests/fixtures/golden_${m}_v2_${sr}.bin"
                [ ! -f "$g" ] && MISSING_GOLDENS+=("$g")
            done
        done
        for m in "${V2_EX_192K_MODELS[@]}"; do
            for sr in "${V2_EX_192K[@]}"; do
                g="tests/fixtures/golden_${m}_v2_${sr}.bin"
                [ ! -f "$g" ] && MISSING_GOLDENS+=("$g")
            done
        done
        for m in "${V2_48K_MODELS[@]}"; do
            g="tests/fixtures/golden_${m}_v2_48000.bin"
            [ ! -f "$g" ] && MISSING_GOLDENS+=("$g")
        done
        if [ ${#MISSING_GOLDENS[@]} -gt 0 ]; then
            echo -e "${RED}${BOLD}❌ Ainda faltam goldens após golden_gen_build.sh:${NC}"
            for g in "${MISSING_GOLDENS[@]}"; do
                echo "    - $g"
            done
            echo -e "  ${YELLOW}V2 goldens podem não ser gerados para todos os SRs (restrição do C++ render tool).${NC}"
            exit 1
        fi
    else
        echo -e "  ${YELLOW}→ Execute: ./tests/fixtures/golden_gen_build.sh${NC}"
        if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
            echo -e "  ${YELLOW}→ Instale: cmake >= 3.10, g++/clang++ com C++20${NC}"
        fi
        echo -e "  ${YELLOW}→ Ou defina NAM_AUTO_BUILD_GOLDENS=1 para geração automática.${NC}"
        exit 1
    fi
fi

if [ ! -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    echo -e "${RED}${BOLD}❌ NeuralAmpModelerCore não encontrado.${NC}"
    echo -e "  ${YELLOW}→ Execute: ./utils/mod-update.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Pré-requisitos C++ e golden vectors verificados.${NC}"

# Trackers for the final summary
declare -a PHASE_NAMES
declare -a PHASE_COMMANDS
declare -a PHASE_STATUS
declare -a PHASE_DURATIONS
declare -a PHASE_SUB_TIMINGS
PHASE_COUNT=0
N_TOP_SLOWEST=5

# timed_cargo_test — runs a cargo test invocation, captures timing.
# Usage: timed_cargo_test <label> <cargo_test_args...>
# Appends per-invocation "TIMED: <seconds> <label>" lines to a temp tracker.
TIMED_TRACKER=$(mktemp)
trap 'rm -f "$TIMED_TRACKER"' EXIT
timed_cargo_test() {
    local label="$1"
    shift
    local start_t
    start_t=$(date +%s%N)
    cargo test "$@"
    local status=$?
    local end_t
    end_t=$(date +%s%N)
    local duration_ns=$((end_t - start_t))
    local duration_s
    duration_s=$(LC_NUMERIC=C awk -v ns="$duration_ns" 'BEGIN { printf "%.3f", ns / 1000000000 }')
    echo "TIMED: $duration_s $label" >> "$TIMED_TRACKER"
    return $status
}

# extract_sub_timings: reads the timed tracker, returns top-N slowest entries.
extract_sub_timings() {
    if [ ! -f "$TIMED_TRACKER" ] || [ ! -s "$TIMED_TRACKER" ]; then
        return
    fi
    grep '^TIMED:' "$TIMED_TRACKER" | \
        sed 's/^TIMED: //' | \
        sort -rn | \
        head -n "$N_TOP_SLOWEST"
}

# extract_top_benches: parse criterion bench output for top-N slowest by median time.
# Usage: extract_top_benches <log_file> <n>
extract_top_benches() {
    local log="$1"
    local n="${2:-$N_TOP_SLOWEST}"
    if [ ! -f "$log" ]; then
        return
    fi
    # Criterion output: bench name on a line, then "time: [1.2345 ms 1.2500 ms 1.2750 ms]" on the next.
    # We capture bench name from unindented lines and match it with the following time: line.
    awk '
    BEGIN { bench = "" }
    /^Benchmarking/  { bench = "" }                          # new benchmark start — discard
    /^[A-Za-z]/ && !/:/ && !/^Found/ && !/^change:/ { bench = $1 }  # capture bench result name
    /time:.*\[/ && bench != "" {
        split($0, a, /[\[\]]/)
        if (length(a) >= 2) {
            split(a[2], b, /[[:space:]]+/)
            # b: [val1, unit, val2, unit, val3, unit]  — median is b[3] with unit b[4]
            median_val  = b[3]
            median_unit = b[4]
            # Convert to nanoseconds for sorting
            if (median_unit == "ns")        ns = median_val
            else if (median_unit == "µs")   ns = median_val * 1000
            else if (median_unit == "ms")   ns = median_val * 1000000
            else if (median_unit == "s")    ns = median_val * 1000000000
            else                            ns = median_val
            printf "%.0f %s %s %s\n", ns, median_val, median_unit, bench
        }
        bench = ""
    }
    ' "$log" | sort -rn | head -n "$n" | while read -r ns val unit bench; do
        printf "  %s %s  %s\n" "$val" "$unit" "$bench"
    done
}

run_phase() {
    local name="$1"
    local cmd="$2"
    local log_file="$3"

    echo -e "\n${BLUE}${BOLD}[Phase $((PHASE_COUNT+1))] $name...${NC}"
    echo -e "Executando: ${YELLOW}$cmd${NC}"
    echo -e "Log em: ${YELLOW}target/logs/$log_file${NC}"

    local start_time=$(date +%s)

    # Reset timed tracker for this phase
    : > "$TIMED_TRACKER"

    # Run command and capture output/status
    eval "$cmd" > "target/logs/$log_file" 2>&1
    local status=$?

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    PHASE_NAMES[$PHASE_COUNT]="$name"
    PHASE_COMMANDS[$PHASE_COUNT]="$cmd"
    PHASE_DURATIONS[$PHASE_COUNT]="$duration"

    # Capture sub-timings for this phase
    PHASE_SUB_TIMINGS[$PHASE_COUNT]="$(extract_sub_timings)"

    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ Sucesso (${duration}s)${NC}"
        PHASE_STATUS[$PHASE_COUNT]="PASSED"
    else
        echo -e "${RED}❌ Falha (${duration}s) - Status: $status${NC}"
        PHASE_STATUS[$PHASE_COUNT]="FAILED"
    fi

    PHASE_COUNT=$((PHASE_COUNT + 1))
    return $status
}

# --- Phase 1: Soak/Stress tests + PipeWire Integration (release, standalone) ---
run_phase \
    "Soak Tests (Numerical Stability)" \
    'status=0; timed_cargo_test "soak_test" --release --no-fail-fast --features standalone --test soak_test -- --ignored --nocapture || status=1; timed_cargo_test "pipeline_soak" --release --no-fail-fast --features standalone --test pipeline_soak -- --ignored --nocapture --test-threads=1 || status=1; [ $status -eq 0 ]' \
    "phase1-soak.log" || true

run_pipewire_phase() {
    echo "  Verificando daemon PipeWire..."
    if pw-cli info >/dev/null 2>&1; then
        echo "  PipeWire detectado. Executando teste de integração..."
        cargo test --release --no-fail-fast --features standalone --test pw_integration_test -- --ignored --nocapture
    else
        echo "  PipeWire indisponível (pw-cli info falhou). Pulando teste de integração."
        return 0
    fi
}

run_phase \
    "PipeWire Integration Test" \
    "run_pipewire_phase" \
    "phase1-pipewire.log" || true

# --- Phase 2: Property-Based, Parity, C++ Parity, Golden Vectors (release, default) ---
run_phase \
    "Property-Based, Parity & Golden Vectors in Release" \
    'status=0; timed_cargo_test "proptest_parsers" --release --no-fail-fast --test proptest_parsers -- --ignored || status=1; timed_cargo_test "proptest_math" --release --no-fail-fast --test proptest_math -- --ignored || status=1; timed_cargo_test "lstm_gate_bf16_parity" --release --no-fail-fast --test lstm_gate_bf16_parity -- --ignored || status=1; timed_cargo_test "lstm_scalar_bf16_parity" --release --no-fail-fast --test lstm_scalar_bf16_parity -- --ignored || status=1; timed_cargo_test "lib_pipeline_block_proptest" --release --no-fail-fast --lib -- dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest --ignored || status=1; timed_cargo_test "gate_fsm_proptest" --release --no-fail-fast --test gate_fsm_proptest -- --ignored || status=1; timed_cargo_test "adaptive_fsm_proptest" --release --no-fail-fast --test adaptive_fsm_proptest -- --ignored || status=1; timed_cargo_test "cpp_parity" --release --no-fail-fast --test cpp_parity -- --ignored --nocapture || status=1; timed_cargo_test "cabsim_cpp_parity" --release --no-fail-fast --test cabsim_cpp_parity -- --ignored --nocapture || status=1; timed_cargo_test "golden_vectors_v2" --release --no-fail-fast --test golden_vectors -- v2_ --skip wavenet_official --ignored --nocapture || status=1; [ $status -eq 0 ]' \
    "phase2-proptests-parity.log" || true

# --- Phase 3: Resampler Heap-Audit (release, heap-audit) ---
run_phase \
    "Resampler, Cabsim & A2 Heap-Audit" \
    'status=0; timed_cargo_test "resampler_heap_audit" --release --no-fail-fast --features heap-audit --test resampler_heap_audit || status=1; timed_cargo_test "cabsim_heap_audit" --release --no-fail-fast --features heap-audit --test cabsim_heap_audit || status=1; timed_cargo_test "a2_heap_audit" --release --no-fail-fast --features heap-audit --test a2_heap_audit || status=1; [ $status -eq 0 ]' \
    "phase3-heap-audit.log" || true

# --- Phase 4: CLAP Release Validation & Concurrency (Local helper function) ---
run_clap_audit_local() {
    echo "  Limpando binário CLAP anterior..."
    rm -f target/release/libnam_rs.so

    echo "  Compilando CLAP Plugin em modo Release..."
    RUSTFLAGS="-Clink-arg=-Wl,-soname,nam-rs.clap" \
      cargo build --release --no-default-features --features "clap-plugin,heap-audit,testing" --lib

    local RELEASE_CLAP_BIN="target/release/libnam_rs.so"
    if [ ! -f "$RELEASE_CLAP_BIN" ]; then
        echo "Erro: libnam_rs.so de release não encontrado." >&2
        return 1
    fi

    echo "  Auditando SONAME e símbolos exportados..."
    if ! readelf -d "$RELEASE_CLAP_BIN" | grep -q SONAME; then
        echo "Erro: SONAME ausente no binário de Release!" >&2
        return 1
    fi
    if ! nm -D "$RELEASE_CLAP_BIN" | grep -q "clap_entry"; then
        echo "Erro: Símbolo 'clap_entry' ausente no binário de Release!" >&2
        return 1
    fi

    local audit_status=0

    if command -v clap-validator >/dev/null 2>&1 && command -v jq >/dev/null 2>&1; then
        echo "  Executando clap-validator estrito..."
        NAM_HEAP_AUDIT=1 clap-validator validate "$RELEASE_CLAP_BIN" --json > target/logs/release-validation.json 2> target/logs/release-validation.stderr
        if ! jq -e '[.. | objects | select(.code? == "failure" or .code? == "warning")] | length == 0' target/logs/release-validation.json >/dev/null; then
            echo "Erro: Falha ou avisos detectados pelo clap-validator!" >&2
            audit_status=1
        fi
    else
        echo "  Aviso: clap-validator ou jq indisponíveis. Pulando auditoria externa."
    fi

    echo "  Executando testes de concorrência com instâncias múltiplas..."
    timed_cargo_test "clap_multi_instance" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" --test clap_multi_instance -- --ignored --nocapture || audit_status=1

    echo "  Executando teste de stress do GC com 1000 swaps..."
    timed_cargo_test "gc_stress_1000_swaps" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" --lib -- clap::processor::processor_test::tests::test_gc_stress_1000_swaps --include-ignored --nocapture || audit_status=1

    echo "  Executando testes de concorrência dedicados (T8.12, sem --test-threads=1)..."
    timed_cargo_test "concurrency_stress" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" --test concurrency_stress -- --ignored --nocapture || audit_status=1

    echo "  Executando testes unitários e de integração em modo Mono..."
    timed_cargo_test "clap_plugin_testing" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" --lib || audit_status=1

    return $audit_status
}

run_phase \
    "CLAP Release Validation & Concurrency" \
    "run_clap_audit_local" \
    "phase4-clap-validation.log" || true

# --- Phase 5: Long Benchmarks (Performance) ---
run_phase \
    "Long Performance Benchmarks" \
    'status=0; cargo bench --no-fail-fast || status=1; cargo bench --no-fail-fast --features standalone,long_bench --bench long_inference_bench || status=1; [ $status -eq 0 ]' \
    "phase5-benchmarks.log" || true

# --- Print beautifully structured summary ---
echo -e "\n${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                  AUDIT SUMMARY REPORT                          ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"
printf " | %-45s | %-10s | %-10s |\n" "Phase Name" "Duration" "Status"
printf " |-%-45s-|-%-10s-|-%-10s-|\n" "---------------------------------------------" "----------" "----------"

ANY_FAILED=0
for ((i=0; i<PHASE_COUNT; i++)); do
    name="${PHASE_NAMES[$i]}"
    duration="${PHASE_DURATIONS[$i]}s"
    status="${PHASE_STATUS[$i]}"

    if [ "$status" = "PASSED" ]; then
        status_colored="${GREEN}${status}${NC}"
    else
        status_colored="${RED}${status}${NC}"
        ANY_FAILED=1
    fi
    printf " | %-45s | %-10s | %-19b |\n" "$name" "$duration" "$status_colored"
done

# --- Top-N slowest sub-timings per heavy phase ---
echo -e "\n${BLUE}${BOLD}  Top-$N_TOP_SLOWEST Items Mais Lentos por Fase Pesada${NC}"
echo -e "${BLUE}${BOLD}  $(printf '━%.0s' {1..60})${NC}"

for ((i=0; i<PHASE_COUNT; i++)); do
    name="${PHASE_NAMES[$i]}"
    sub_timings="${PHASE_SUB_TIMINGS[$i]}"

    # Phase 5 (benchmarks) uses criterion — parse bench log separately
    if [[ "$name" == *"Benchmark"* ]]; then
        bench_log="phase5-benchmarks.log"
        top_benches=$(extract_top_benches "target/logs/$bench_log" "$N_TOP_SLOWEST" 2>/dev/null)
        if [ -n "$top_benches" ]; then
            echo -e "\n  ${YELLOW}${BOLD}[$name]${NC}"
            echo "$top_benches"
        fi
        continue
    fi

    if [ -n "$sub_timings" ]; then
        echo -e "\n  ${YELLOW}${BOLD}[$name]${NC}"
        rank=1
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                t="${line%% *}"
                lbl="${line#* }"
                printf "    %2d. %8ss  %s\n" "$rank" "$t" "$lbl"
                rank=$((rank + 1))
            fi
        done <<< "$sub_timings"
    fi
done

echo -e "\n${BLUE}${BOLD}================================================================${NC}"

# Cleanup timed tracker temp file
rm -f "$TIMED_TRACKER"

if [ $ANY_FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ Todos os estágios da auditoria passaram com sucesso!${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}❌ Algum estágio da auditoria falhou. Verifique os logs em target/logs/${NC}"
    exit 1
fi
