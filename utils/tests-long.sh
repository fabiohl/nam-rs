#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Nightly/pre-release audit suite — the final, extreme bug hunter of nam-rs.
# Runs the full advanced QA surface that `utils/lints.sh` (static analysis)
# and `utils/tests-quick.sh` (agile first line) deliberately leave out:
# numerical soak/endurance, full proptest/fuzz case counts, full C++
# NeuralAmpModelerCore parity matrix (multi-SR), cross-ISA determinism,
# RT-safety heap-audits, release-mode CLAP compliance + concurrency stress,
# Criterion benchmarks for the record, and the RT deadline/jitter gate.
# Everything measuring floats runs `--release` — the codegen path that
# ships to the end user (see docs/testing.md §2, Axis B).
#
# Non-duplication contract (docs/testing.md §2, §4):
#   - Does NOT repeat `lints.sh` (fmt/check/clippy) or `tests-quick.sh`
#     (structural debug tests, the 5 measurement oracles, capped parser
#     fuzzing). It only runs what those two intentionally leave `#[ignore]`d
#     or out of scope. Every phase below cross-references its quick-suite
#     counterpart so scope drift is visible at a glance.
#   - Does NOT repeat `tests-performance-regression.sh` (per-push baseline
#     gate) — Phase 6 records the full Criterion suite for the nightly
#     archive, with no baseline gating of its own.
#
# Failure isolation: each phase runs to completion independently (§6.2) so
# one bad phase never hides the rest — a nightly run that dies on a shell
# bug would cost a full day of blind spots before the next window.
#
# Environment variables:
#   NAM_SKIP_GOLDEN_BUILD=1   Opt-out from automatic generation of missing golden vectors.
#                             By default, if goldens are missing and C++ toolchain +
#                             NeuralAmpModelerCore dependencies are present, they are
#                             automatically rebuilt during Phase 0.
#   NAM_AUTO_BUILD_GOLDENS    (Deprecated and ignored)

set -euo pipefail
## Observação à IA: Dado à longa (por design) duração da execução deste script, é PROIBIDO executa-lo durante atividades de IA.
### Se necessário, peça ao desenvolvedor humano para roda-lo e trazer o resultado.

# ── Test-to-entry-point mapping (Sprint 2 → Sprint 3 bridge) ────────────────
# Each test binary is assigned to its future entry-point module (Sprint 3,
# Tarefas 3.1–3.3). When the entry-point files exist, the script auto-detects
# and uses the new `--test <entry> <entry>::<test>` format. Otherwise it falls
# back to legacy `--test <test>` (flat tests/ layout).
#
# To dry-run the Sprint 3 command assembly without executing:
#   NAM_DRY_RUN_ARCH=1 utils/tests-long.sh

declare -A LONG_ENTRY_MAP=(
    [meta_coherence]="models"
    [proptest_parsers]="models"
    [proptest_math]="models"
    [gate_fsm_proptest]="models"
    [adaptive_fsm_proptest]="models"
    [lstm_model_dyn_validation]="models"
    [golden_vectors]="models"
    [linear_golden]="models"
    [spectral_fidelity]="models"
    [diagnostic_bundle]="models"
    [lstm_gate_bf16_parity]="parity"
    [lstm_scalar_bf16_parity]="parity"
    [cpp_parity]="parity"
    [cabsim_cpp_parity]="parity"
    [isa_parity]="parity"
    [t33_diagnostic_recurrent_drift_lstm_1x16]="parity"
    [t33b_diagnostic_recurrent_drift_lstm_1x16_paired]="parity"
    [soak_test]="perf_soak"
    [pipeline_soak]="perf_soak"
    [concurrency_stress]="perf_soak"
    [pw_integration_test]="perf_soak"
    [resampler_heap_audit]="rt_constraints"
    [cabsim_heap_audit]="rt_constraints"
    [a2_heap_audit]="rt_constraints"
    [rt_deadline]="rt_constraints"
    [rt_jitter]="rt_constraints"
    [clap_lifecycle_test]="clap"
    [clap_state_migration]="clap"
    [clap_multi_instance]="clap"
)

_entry_files_exist() {
    for entry in models perf_soak parity clap rt_constraints; do
        [ -f "tests/${entry}.rs" ] || return 1
    done
    return 0
}

_test_flag() {
    local test_name="$1"
    if _entry_files_exist || [ "${NAM_NEW_ARCH:-0}" = "1" ]; then
        local entry="${LONG_ENTRY_MAP[$test_name]:-models}"
        echo "--test $entry ${test_name}"
    else
        echo "--test $test_name"
    fi
}

# Shared style helpers (RED/GREEN/YELLOW/BLUE/BOLD/NC) + cd to project root.
source "$(dirname "$0")/_lib.sh"

# Setup defensive error trap (message-only; phase failures are isolated via
# `run_phase ... || true` below and never reach this trap — see §6.2).
trap 'echo -e "\n${RED}${BOLD}❌ Erro inesperado: Comando \"$BASH_COMMAND\" falhou na linha $LINENO com status $?. Abortando suíte de testes.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}===============================================================${NC}"
echo -e "${BLUE}${BOLD}    nam-rs Long-Duration Stress & Audit Suite (± 55 minutes)   ${NC}"
echo -e "${BLUE}${BOLD}===============================================================${NC}"

# Setup target logs
rm -rf target/logs/
mkdir -p target/logs/

# Cleanup accumulated live-test artifacts from previous runs (41+ MB WAVs)
rm -rf tests/fixtures/.temp_live/

# Verify NeuralAmpModelerCore presence.
if [ ! -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    echo -e "${RED}${BOLD}❌ NeuralAmpModelerCore não encontrado em tests/fixtures/NeuralAmpModelerCore.${NC}"
    echo -e "${YELLOW}Por favor, execute './utils/mod-update.sh' para clonar e configurar as dependências.${NC}"
    exit 1
fi

CURRENT_CORE_SHA=$(cd tests/fixtures/NeuralAmpModelerCore && git rev-parse HEAD 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ NeuralAmpModelerCore encontrado (versão: $CURRENT_CORE_SHA).${NC}"

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

    if [ -n "${NAM_AUTO_BUILD_GOLDENS+x}" ]; then
        echo -e "${YELLOW}⚠ AVISO: A variável NAM_AUTO_BUILD_GOLDENS foi descontinuada e agora é ignorada.${NC}"
        echo -e "${YELLOW}  O auto-build agora é ativado por padrão se faltarem goldens e as ferramentas estiverem presentes.${NC}"
        echo -e "${YELLOW}  Para desativar o auto-build, utilize NAM_SKIP_GOLDEN_BUILD=1.${NC}"
    fi

    if [ "${NAM_SKIP_GOLDEN_BUILD:-0}" = "1" ]; then
        echo -e "${YELLOW}→ NAM_SKIP_GOLDEN_BUILD=1 — pulando regeneração automática.${NC}"
        exit 1
    fi

    # Auto-build é o padrão quando toolchain C++ e NeuralAmpModelerCore estão presentes
    if [ ${#MISSING_TOOLS[@]} -eq 0 ] && [ -d "tests/fixtures/NeuralAmpModelerCore" ]; then
        echo -e "\n${YELLOW}${BOLD}→ Regenerando goldens automaticamente (toolchain C++ + NeuralAmpModelerCore presentes)...${NC}"
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
        if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
            echo -e "  ${YELLOW}→ Instale: cmake >= 3.10, g++/clang++ com C++20${NC}"
        fi
        echo -e "  ${YELLOW}→ Execute: ./utils/mod-update.sh${NC}"
        exit 1
    fi
fi

if [ ! -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    echo -e "${RED}${BOLD}❌ NeuralAmpModelerCore não encontrado.${NC}"
    echo -e "  ${YELLOW}→ Execute: ./utils/mod-update.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Pré-requisitos C++ e golden vectors verificados.${NC}"

# ── Optional freshness check (warn-only, not blocking) ──
MANIFEST="tests/fixtures/.golden_manifest.sha256"
if [ -f "$MANIFEST" ]; then
    STALE_COUNT=0
    while read -r expected_model_sha expected_golden_sha nam_file golden_file; do
        [[ "$expected_model_sha" == \#* ]] && continue
        MODEL_PATH="tests/fixtures/models/$nam_file"
        if [ -f "$MODEL_PATH" ]; then
            CURRENT_MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
            if [ "$CURRENT_MODEL_SHA" != "$expected_model_sha" ]; then
                echo -e "  ${YELLOW}⚠ STALE: $nam_file changed since golden was generated${NC}"
                STALE_COUNT=$((STALE_COUNT + 1))
            fi
        fi
    done < "$MANIFEST"
    if [ "$STALE_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠ $STALE_COUNT golden(s) may be stale. Consider re-running golden_gen_build.sh${NC}"
    else
        echo -e "${GREEN}✓ Golden freshness manifest OK (all models match).${NC}"
    fi
else
    echo -e "  ${YELLOW}⚠ No freshness manifest found (.golden_manifest.sha256). Run golden_gen_build.sh to generate.${NC}"
fi

# ── Catalog↔test coherence gate (blocking) ──
# `meta_coherence` is a cheap, dependency-free governance test (no NAMCore, no
# goldens needed — it only parses golden_gen_build.sh + tests/*.rs). It has no
# home in tests-quick.sh (not a correctness or structural test) and would be
# silently orphaned ("on demand" only) without this hook. Runs here, before
# the ± 50 min battery, so a drifted catalog fails fast instead of burning a
# full nightly window before being noticed.
echo -e "\n${BLUE}${BOLD}→ Verificando coerência catálogo↔testes (meta_coherence)...${NC}"
if ! cargo test --release $(_test_flag meta_coherence); then
    echo -e "${RED}${BOLD}❌ meta_coherence falhou — catálogo de goldens divergiu dos testes #[ignore].${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Catálogo de goldens coerente com os testes.${NC}"

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

    if [ $status -eq 77 ]; then
        echo -e "${YELLOW}⚠ SKIPPED (${duration}s)${NC}"
        PHASE_STATUS[$PHASE_COUNT]="SKIPPED"
    elif [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ Sucesso (${duration}s)${NC}"
        PHASE_STATUS[$PHASE_COUNT]="PASSED"

        if [ "$duration" -lt 1 ]; then
            echo -e "${YELLOW}${BOLD}⚠ AVISO: Fase '${name}' completou com PASSED em < 1s — fase possivelmente vazia/falso-verde.${NC}"
        fi
    else
        echo -e "${RED}❌ Falha (${duration}s) - Status: $status${NC}"
        PHASE_STATUS[$PHASE_COUNT]="FAILED"
    fi

    PHASE_COUNT=$((PHASE_COUNT + 1))
    return $status
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase bodies — one function per phase. Each function is passed by name to
# run_phase (never inlined as a giant `;`-chained string) so the suite stays
# easy to scan and extend without accumulating one-liner cruft (the exact
# "bagunçado" failure mode we want to avoid in an unattended nightly job).
# Every phase cross-references its non-overlapping tests-quick.sh counterpart.
# ═══════════════════════════════════════════════════════════════════════════

# --- Phase 1: Soak/Endurance (release, standalone, --ignored) ---
# tests-quick.sh runs 1 non-ignored decomposition test per suite (Fase 1,
# debug); every #[ignore]'d soak/endurance test (10M+ frames) lives here.
run_soak_phase() {
    local status=0
    timed_cargo_test "soak_test" --release --no-fail-fast --features standalone $(_test_flag soak_test) -- --ignored --nocapture || status=1
    timed_cargo_test "pipeline_soak" --release --no-fail-fast --features standalone $(_test_flag pipeline_soak) -- --ignored --nocapture --test-threads=1 || status=1
    return $status
}
run_phase "Soak Tests (Numerical Stability)" "run_soak_phase" "phase1-soak.log" || true

# --- Phase 2: PipeWire Integration (release, standalone; graceful skip) ---
# Never runs in tests-quick.sh (would make the daily suite depend on a live
# PipeWire daemon). Skips gracefully when no daemon is reachable.
run_pipewire_phase() {
    echo "  Verificando daemon PipeWire..."
    if pw-cli info all >/dev/null 2>&1; then
        echo "  PipeWire detectado. Executando teste de integração..."
        cargo test --release --no-fail-fast --features standalone $(_test_flag pw_integration_test) -- --ignored --nocapture
    else
        echo "  PipeWire indisponível (pw-cli info all falhou). Pulando teste de integração."
        return 77
    fi
}
run_phase "PipeWire Integration Test" "run_pipewire_phase" "phase2-pipewire.log" || true

# --- Phase 3: Property-Based, FSM, Parity, Golden Vectors & ISA (release) ---
# The full, uncapped counterpart of tests-quick.sh Fase 2/3: every proptest
# runs at its full case count (Fase 3 caps proptest_parsers at 1000 cases),
# the C++/golden oracles run their full multi-SR/full-matrix scope (Fase 2
# only runs the v1/quick_parity subset), and cross-ISA + heavy/dyn parity —
# entirely absent from the quick suite — run here for the first time.
run_proptests_parity_phase() {
    local status=0
    # Full-count parser/math/gate/FSM fuzzing (quick caps or excludes these).
    timed_cargo_test "proptest_parsers" --release --no-fail-fast $(_test_flag proptest_parsers) -- --ignored || status=1
    timed_cargo_test "proptest_math" --release --no-fail-fast $(_test_flag proptest_math) -- --ignored || status=1
    timed_cargo_test "lstm_gate_bf16_parity" --release --no-fail-fast $(_test_flag lstm_gate_bf16_parity) -- --ignored || status=1
    timed_cargo_test "lstm_scalar_bf16_parity" --release --no-fail-fast $(_test_flag lstm_scalar_bf16_parity) -- --ignored || status=1
    timed_cargo_test "gate_fsm_proptest" --release --no-fail-fast $(_test_flag gate_fsm_proptest) -- --ignored || status=1
    timed_cargo_test "adaptive_fsm_proptest" --release --no-fail-fast $(_test_flag adaptive_fsm_proptest) -- --ignored || status=1
    # ModelDyn scalar-vs-SIMD parity proptests (arbitrary topologies) — no
    # quick-suite equivalent; LstmModelDyn parity is otherwise untested.
    timed_cargo_test "lstm_model_dyn_validation" --release --no-fail-fast $(_test_flag lstm_model_dyn_validation) -- --ignored --nocapture || status=1
    # Full C++ NAMCore live parity matrix + CabSim convolution parity
    # (quick's Fase 2 only runs the 3-model `quick_parity` subset).
    timed_cargo_test "cpp_parity" --release --no-fail-fast $(_test_flag cpp_parity) -- --ignored --nocapture || status=1
    timed_cargo_test "cabsim_cpp_parity" --release --no-fail-fast $(_test_flag cabsim_cpp_parity) -- --ignored --nocapture || status=1
    # Recurrent State Drift Diagnostics (Tarefa 1.4)
    timed_cargo_test "t33_diagnostic_recurrent_drift_lstm_1x16" --release --no-fail-fast $(_test_flag t33_diagnostic_recurrent_drift_lstm_1x16) -- --ignored --nocapture || status=1
    timed_cargo_test "t33b_diagnostic_recurrent_drift_lstm_1x16_paired" --release --no-fail-fast $(_test_flag t33b_diagnostic_recurrent_drift_lstm_1x16_paired) -- --ignored --nocapture || status=1
    # Golden vectors v2 (multi-SR); v1 already covered by quick's Fase 2.
    timed_cargo_test "golden_vectors_v2" --release --no-fail-fast $(_test_flag golden_vectors) -- v2_ --ignored --nocapture || status=1
    # Heavy/long receptive-field golden regression (quick only runs the
    # cheap non-ignored linear_golden cases).
    timed_cargo_test "linear_golden_heavy" --release --no-fail-fast $(_test_flag linear_golden) -- --ignored --nocapture || status=1
    # Full cross-ISA determinism matrix (AVX-512, VNNI+BF16 vs AVX2). Quick's
    # Fase 2 only asserts AVX2 self-consistency; gracefully skips per-model
    # when the running CPU lacks the target ISA (see skip_if_unsupported!
    # in tests/isa_parity.rs) — safe to run unconditionally on any machine.
    timed_cargo_test "isa_parity_full_matrix" --release --no-fail-fast $(_test_flag isa_parity) -- --ignored --test-threads=1 --nocapture || status=1
    # Per-model spectral fidelity baselines (ASR/THD+N/IMD/Farina vs the
    # committed fixture). Filtered to `baseline_*` to exclude the manual-only
    # `generate_spectral_fidelity_baseline` fixture writer (never auto-run).
    timed_cargo_test "spectral_fidelity_baselines" --release --no-fail-fast $(_test_flag spectral_fidelity) -- baseline_ --ignored --nocapture || status=1
    # Random block-size sweep for the pipeline resampler chain.
    timed_cargo_test "lib_pipeline_block_proptest" --release --no-fail-fast --lib -- dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest --ignored || status=1
    # Tier-3 "approx-vs-approx" consistency checks (Padé/poly NR1 vs NR2 vs
    # div_ps, AVX2 + AVX-512 for tanh and sigmoid): the f64 Oracle already
    # provides absolute correctness, so these only guard against silent
    # regressions between two approximate paths (docs/testing.md §8).
    # AVX-512 variants self-skip via `is_x86_feature_detected!` when unsupported.
    timed_cargo_test "activations_consistency" --release --no-fail-fast --lib -- "math::activations::" --ignored --nocapture || status=1
    # Gate FSM envelope continuity proptest (10k cases) — unit-level sibling
    # of tests/gate_fsm_proptest.rs, covers the DynamicHysteresis reversal
    # edge case specifically.
    timed_cargo_test "gate_envelope_continuity_proptest" --release --no-fail-fast --lib -- "dsp::gate::gate_test::tests::gate_envelope_continuity_on_reversal" --ignored --nocapture || status=1
    # `dsp::oversample::oversample_test::test_x2_aliasing_rejection` used to be
    # excluded here (hung indefinitely in --release due to an ELF
    # symbol-interposition bug — see
    # docs/postmortem-libm-symbol-interposition.md for the root cause and the
    # fix). It is no longer `#[ignore]`d and needs no special-casing here
    # anymore — it now runs automatically as part of every plain
    # `cargo test --lib` invocation in this script (and in tests-quick.sh's
    # Fase 1), same as its siblings.
    return $status
}
run_phase "Property-Based, Parity & Golden Vectors in Release" "run_proptests_parity_phase" "phase3-proptests-parity.log" || true

# --- Phase 4: RT-Safety Heap-Audit (release, heap-audit) ---
# Zero-alloc verification under the global counting allocator. No quick-suite
# equivalent — the `heap-audit` feature is exclusively a long-suite concern.
run_heap_audit_phase() {
    local status=0
    timed_cargo_test "resampler_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag resampler_heap_audit) || status=1
    timed_cargo_test "cabsim_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag cabsim_heap_audit) || status=1
    timed_cargo_test "a2_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag a2_heap_audit) || status=1
    timed_cargo_test "diagnostic_bundle_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag diagnostic_bundle) -- heap_audit || status=1
    return $status
}
run_phase "Resampler, Cabsim & A2 Heap-Audit" "run_heap_audit_phase" "phase4-heap-audit.log" || true

# --- Phase 5: CLAP Release Validation & Concurrency ---
# Builds and audits the real release `.so` (SONAME, exported symbols,
# clap-validator, lifecycle, state migration, multi-instance/GC/concurrency
# stress). tests-quick.sh only exercises the debug CLAP build; this is its
# strict release-mode superset and never runs anywhere else.
run_clap_audit_phase() {
    local RUSTFLAGS="-Clink-arg=-Wl,-soname,nam-rs.clap -Clink-arg=-Wl,-u,clap_entry"
    export RUSTFLAGS

    echo "  Isolando artefatos CLAP em target/clap-audit/..."
    export CARGO_TARGET_DIR="target/clap-audit"
    rm -rf target/clap-audit/release
    rm -f target/clap-audit/release/libnam_rs.so

    echo "  Compilando CLAP Plugin em modo Release..."
    CARGO_INCREMENTAL=0 cargo build --release --no-default-features --features "clap-plugin,heap-audit,testing" --lib

    local RELEASE_CLAP_BIN="target/clap-audit/release/libnam_rs.so"
    if [ ! -f "$RELEASE_CLAP_BIN" ]; then
        echo "Erro: libnam_rs.so de release não encontrado." >&2
        return 1
    fi

    echo "  Auditando SONAME e símbolos exportados..."
    if ! readelf -d "$RELEASE_CLAP_BIN" | grep SONAME >/dev/null; then
        echo "Erro: SONAME ausente no binário de Release!" >&2
        return 1
    fi
    if ! nm -D "$RELEASE_CLAP_BIN" | grep "clap_entry" >/dev/null; then
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

    echo "  Executando testes de integração CLAP (lifecycle + state migration)..."
    timed_cargo_test "clap_lifecycle_test" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" $(_test_flag clap_lifecycle_test) -- --nocapture || audit_status=1
    timed_cargo_test "clap_state_migration" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" $(_test_flag clap_state_migration) -- --nocapture || audit_status=1

    echo "  Executando testes de concorrência com instâncias múltiplas..."
    timed_cargo_test "clap_multi_instance" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" $(_test_flag clap_multi_instance) -- --ignored --nocapture || audit_status=1

    echo "  Executando teste de stress do GC com 1000 swaps..."
    timed_cargo_test "gc_stress_1000_swaps" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" --lib -- clap::processor::processor_gc_stress_test::tests::test_gc_stress_1000_swaps --include-ignored --nocapture || audit_status=1

    echo "  Executando testes de concorrência dedicados (T8.12, sem --test-threads=1)..."
    timed_cargo_test "concurrency_stress" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" $(_test_flag concurrency_stress) -- --ignored --nocapture || audit_status=1

    echo "  Executando testes unitários e de integração em modo Mono..."
    RUSTFLAGS="${RUSTFLAGS:-} -C debug-assertions=on" timed_cargo_test "clap_plugin_testing" --release --no-default-features --no-fail-fast --features "clap-plugin,heap-audit,testing" --lib || audit_status=1

    unset CARGO_TARGET_DIR
    return $audit_status
}
run_phase "CLAP Release Validation & Concurrency" "run_clap_audit_phase" "phase5-clap-validation.log" || true

# --- Phase 6: Long Performance Benchmarks (Criterion, for the record) ---
# Records the full bench suite nightly. No baseline gating here — that is
# the exclusive job of `utils/tests-performance-regression.sh` (run per-push,
# not duplicated). `fft_radix4_bench`, `gemv_bench`, and `linear` (bench) are
# one-off research artifacts documenting past engineering decisions, not
# regression gates — intentionally excluded from every automated suite.
#
# Each bench target is its OWN `cargo bench` invocation (never combined via
# multiple `--bench` flags in one command). `cargo bench` aborts the entire
# invocation on the first panicking bench binary and does not proceed to the
# next `--bench` target — confirmed in practice on 2026-07-02, where a broken
# `inference_bench` fixture silently prevented `kahan_conv1d_bench` and
# `regression_gate` from ever running or being recorded for that night. One
# bench isolated from the others is the whole point of this phase existing.
run_benchmarks_phase() {
    local status=0
    cargo bench --features long_bench --bench dot_4x_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    cargo bench --features long_bench --bench kahan_conv1d_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    cargo bench --features long_bench --bench inference_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    cargo bench --features long_bench --bench regression_gate -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    cargo bench --features long_bench --bench long_inference_bench || status=1
    cargo bench --features long_bench --bench math_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    cargo bench --features long_bench --bench dsp_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    cargo bench --features long_bench --bench cabsim_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    cargo bench --features long_bench --bench clap_bench -- --sample-size 100 --measurement-time 5 --warm-up-time 1 || status=1
    return $status
}
run_phase "Long Performance Benchmarks" "run_benchmarks_phase" "phase6-benchmarks.log" || true

# --- Phase 7: RT Deadline Gate & Jitter Stress (release-only; meaningless in debug) ---
# Absolute latency ceiling (p99 < 1.33 ms) + jitter characterization under
# CPU contention. Never runs in tests-quick.sh (needs release + a quiet
# enough window to be meaningful — the opposite of "run several times a day").
run_rt_deadline_phase() {
    local status=0
    timed_cargo_test "rt_deadline" --release --no-fail-fast $(_test_flag rt_deadline) -- --nocapture || status=1
    timed_cargo_test "rt_jitter" --release --no-fail-fast $(_test_flag rt_jitter) -- --ignored --nocapture || status=1
    return $status
}
run_phase "RT Deadline Gate & Jitter Stress" "run_rt_deadline_phase" "phase7-rt-deadline.log" || true

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
    elif [ "$status" = "SKIPPED" ]; then
        status_colored="${YELLOW}${status}${NC}"
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

    # Phase 6 (benchmarks) uses criterion — parse bench log separately
    if [[ "$name" == *"Benchmark"* ]]; then
        bench_log="phase6-benchmarks.log"
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
