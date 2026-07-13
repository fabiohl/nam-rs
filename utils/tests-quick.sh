#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Quick QA Suite for nam-rs — agile first line of defense (vocaçao: cargo test).
#
# Divisão de trabalho entre os scripts de QA:
#   * utils/lints.sh            — gate estático (fmt, SPDX, cargo check, clippy).
#                                  Roda a todo momento. NÃO se repete aqui.
#   * utils/tests-quick.sh      — ESTE script. Suíte ágil de testes (cargo test
#                                  e assemelhados). Roda várias vezes ao dia, ao
#                                  menos a cada sprint. Primeira linha defensiva
#                                  confiável: captura problemas rápido sem
#                                  desestimular a execução (~2-3 min).
#   * utils/tests-long.sh      — caçador de bugs extremos (soak, proptests
#                                  completos, paridade C++ full, CLAP release,
#                                  concorrência, bench, RT). Demorado, ~1×/dia.
#
# Princípio filosófico (docs/testing.md §7 — dois eixos ortogonais):
#   Eixo A (rigorosidade):  ignored = long/rigoroso; non-ignored = primeira linha.
#   Eixo B (caminho float): estrutural → debug (rápido, debug-assertions ON);
#                           oráculo de medida → release (mede o path de produção,
#                           sem o qual mediria um "fantasma" — codegen sem -O,
#                           sem contração FMA, sem auto-vetorização).
#   A Fase 1 guarda o eixo A (non-ignored) no eixo B correto (debug estrutural).
#   A Fase 2 guarda os oráculos de medida do §7 no eixo B de produção (release).
#
# Fases:
#   1. Estrutural (debug) — unit (lib) + integração determinística. Compilação
#      rápida, debug-assertions ON. Verifica lógica de parser, máquinas de
#      estado, loaders, SPSC, determinismo bitwise, FSM. EXCLUI os 5 oráculos de
#      medida do §7 (→ Fase 2 release) e rt_deadline (→ long, gate de release).
#   2. Oráculos de medida (release, docs/testing.md §7) — gate autoritativo de
#      floats de produção: golden_vectors v1, cpp_parity quick_parity,
#      reference_oracle_f64, isa_parity (AVX2 self-consistency), spectral_fidelity.
#      Skip gracioso para os dependentes de NAMCore/goldens ausentes.
#   3. Parser fuzzing ágil (release, --ignored) — proptest_parsers (Tier 1:
#      robustez/segurança de parser) com contagem de casos reduzida para
#      permanecer ágil (override via NAM_QUICK_PROPTEST_CASES).
#
# Notas de cobertura:
#   - Clippy/cargo check ficam em lints.sh (nao duplicados aqui).
#   - Bloco CLAP (build + heap-audit + validator) e testes feature-gated
#     (heap-audit/clap) ficam em tests-long.sh (release, superconjunto).
#   - proptest_math (Tier 3) e rt_deadline/rt_jitter(stress)/soak ficam no long.
#   - A Fase 1 lista EXPLICITAMENTE os testes estruturais porque `--skip` por
#     nome colide (ex.: `test_oracle` atingiria threshold_calibration;
#     `test_asr` atingiria unitários de aliasing). Manutenção: ao adicionar um
#     novo teste estrutural em tests/, inclua-o em STRUCT_TESTS e mapeie-o
#     em STRUCT_ENTRY_MAP (vide inventário em docs/testing.md §3).
#     A auto-descoberta de unit tests (lib) é preservada.
#
# ── Skip conditions (graceful exit 0) ─────────────────────────────────────────
# The following skip scenarios are handled gracefully (exit code 0 with
# informational messages). They are designed for CI environments and developer
# machines that may not have all optional dependencies.
#
# Scenario                              Condition                           Consequence
# ────────────────────────────────────  ──────────────────────────────────  ──────────────────────────────────────────────────
# Golden vectors (v1/v2) absent         golden_wavenet_standard.bin and     golden_vectors + isa_parity skipped.
#                                       golden_wavenet_standard_v2_48000    f64 oracle, Spectral Fidelity, Linear FFT
#                                       .bin missing from tests/fixtures/   still run (mathematical-oracle tests, no
#                                                                           pre-computed goldens needed).
#                                       Tracked by: GOLDEN_RAN
#
# C++ toolchain not found               Neither g++ nor clang++ in PATH    cpp_parity entirely skipped.
#                                                                           Tracked by: CPP_PARITY_SKIPPED
#
# CMake configure / build failure        cmake fails to configure or       cpp_parity entirely skipped.
#                                        build the C++ render binary        Tracked by: CPP_PARITY_SKIPPED
#
# NAMCore not checked out                tests/fixtures/NeuralAmpModeler   cpp_parity entirely skipped.
#                                        Core directory absent              Tracked by: CPP_PARITY_SKIPPED
#
# All mandatory tests are NOT skippable — their failure always produces exit
# code 1. These include:
#   Fase 1: structural unit + integration tests (debug)
#   Fase 2: f64 oracle, Spectral Fidelity, Linear FFT (always run, no deps)
#   Fase 3: parser fuzzing (proptest_parsers)
#
# CI note: In a minimal CI environment without golden fixtures and without a
# C++ toolchain, this script exits 0 after running the non-skippable core.
# No false alarms are raised.

set -euo pipefail

PHASE_TOTAL=3
source "$(dirname "$0")/_lib.sh"


# ── Freshness gate (Governança de frescor — Épico E) ────────────────────────
# Checks the versioned golden freshness manifest against current model files.
# Fail hard on staleness: a golden from a modified .nam is a non-starter per
# the "Todo Golden Deve Poder Falhar" principle (tests/fixtures/README.md §Principle).
# Returns 0 if all models match, 1 if any model is stale or manifest missing.
check_freshness() {
    local MANIFEST="tests/fixtures/.golden_manifest.sha256"
    local MODELS_DIR="tests/fixtures/models"
    local FIXTURES_DIR="tests/fixtures"

    if [ ! -f "$MANIFEST" ]; then
        echo -e "${RED}${BOLD}❌ Freshness manifest missing: $MANIFEST${NC}"
        echo -e "${RED}   Run './tests/fixtures/golden_gen_build.sh' to generate goldens and manifest.${NC}"
        return 1
    fi

    local STALE_COUNT=0
    local MISSING_COUNT=0

    while IFS= read -r line; do
        # ── EXPECTED lines (Freshness Gate — F-C9 / Tarefa T3.2) ──
        # Every file listed as EXPECTED MUST exist on disk. If the C++ render
        # tool skipped a golden (e.g., SR mismatch), the CATALOG should be
        # updated to match reality (e.g., change v2_scope from "all" to
        # "48k_only" for models declaring expected_sample_rate=48000).
        if [[ "$line" =~ ^#\ EXPECTED:\ (.+)$ ]]; then
            local expected_file="${BASH_REMATCH[1]}"
            if [ ! -f "$FIXTURES_DIR/$expected_file" ]; then
                echo -e "  ${RED}▲ MISSING: $expected_file — expected golden file not found on disk${NC}"
                MISSING_COUNT=$((MISSING_COUNT + 1))
            fi
            continue
        fi

        [[ "$line" =~ ^# ]] && continue
        [[ -z "$line" ]] && continue
        read -r expected_model_sha expected_golden_sha nam_file golden_file <<< "$line"
        local MODEL_PATH="$MODELS_DIR/$nam_file"
        if [ -f "$MODEL_PATH" ]; then
            local CURRENT_MODEL_SHA
            CURRENT_MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
            if [ "$CURRENT_MODEL_SHA" != "$expected_model_sha" ]; then
                echo -e "  ${RED}▲ STALE: $nam_file — model modified since golden was generated${NC}"
                STALE_COUNT=$((STALE_COUNT + 1))
            fi
        fi
    done < "$MANIFEST"

    if [ "$MISSING_COUNT" -gt 0 ]; then
        echo -e "${RED}${BOLD}❌ Freshness gate FAILED: $MISSING_COUNT expected golden file(s) missing.${NC}"
        echo -e "${RED}   Run './tests/fixtures/golden_gen_build.sh' to generate missing golden vectors.${NC}"
        return 1
    fi

    if [ "$STALE_COUNT" -gt 0 ]; then
        echo -e "${RED}${BOLD}❌ Freshness gate FAILED: $STALE_COUNT model(s) stale.${NC}"
        echo -e "${RED}   Run './tests/fixtures/golden_gen_build.sh' to regenerate goldens and manifest.${NC}"
        return 1
    fi
    echo -e "  ${GREEN}✓ Freshness gate passed (all model hashes match manifest, all expected goldens present).${NC}"
    return 0
}

# Re-execute with low CPU and I/O priority (nice and ionice) to prevent overloading the system.
# This can be bypassed by setting NAM_NO_LOW_PRIORITY=1.
if [ "${NAM_LOW_PRIORITY:-0}" != "1" ] && [ "${NAM_NO_LOW_PRIORITY:-0}" != "1" ]; then
    export NAM_LOW_PRIORITY=1
    CMD_PREFIX=""
    if command -v nice >/dev/null 2>&1; then
        CMD_PREFIX="nice -n 19"
    fi
    if command -v ionice >/dev/null 2>&1; then
        CMD_PREFIX="$CMD_PREFIX ionice -c 3"
    fi
    if [ -n "$CMD_PREFIX" ]; then
        echo -e "${YELLOW}ⓘ Reiniciando o script com baixa prioridade (CPU/IO) para evitar travamentos...${NC}"
        exec $CMD_PREFIX "$0" "$@"
    fi
fi

trap 'echo -e "\n${RED}${BOLD}❌ Erro inesperado: Comando \"$BASH_COMMAND\" falhou na linha $LINENO com status $?. Abortando suíte de testes.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}==========================${NC}"
echo -e "${BLUE}${BOLD}   nam-rs Quick QA Suite"
echo -e "${BLUE}${BOLD}==========================${NC}"



# ── Fase 1: Estrutural (debug) ──────────────────────────────────────────────
# Unit tests (lib, auto-descobertos) + integração determinística (lista explícita).
# Exclui os 5 oráculos de medida do §7 (→ Fase 2 release) e rt_deadline (→ long).
# perf_soak (concurrency_stress, spsc_pipeline, soak_test) permanece na Fase 1:
# são testes estruturais determinísticos (~2s) que validam invariantes de
# concorrência/pipeline — não são benchmarks nem stress-tests pesados (→ long).
# debug-assertions ON captura invariantes baratos que --release mascararia.
phase "Estrutural: unit + integração determinística (debug)..."

# ── Test-to-entry-point mapping (Sprint 2 → Sprint 3 bridge) ────────────────
# Each structural test is assigned to its future entry-point module (Sprint 3,
# Tarefas 3.1–3.3). When the entry-point files exist, the script auto-detects
# and uses the new `--test <entry> <entry>::<test>` format. Otherwise it falls
# back to legacy `--test=<file>` (flat tests/ layout).
#
# To dry-run the Sprint 3 command assembly without executing:
#   NAM_DRY_RUN_ARCH=1 utils/tests-quick.sh

declare -A STRUCT_ENTRY_MAP=(
    [a2_loader]="models"
    [activation_precision]="models"
    [adaptive_fsm_proptest]="models"
    [cabsim_golden]="models"
    [concurrency_stress]="perf_soak"
    [container_slimmable]="models"
    [diagnostic_bundle]="models"
    [ebu_lufs_compliance]="models"
    [fixture_b1_2_smoke]="models"
    [linear_golden]="models"
    [lstm_activation_precision]="models"
    [lstm_model_dyn_validation]="models"
    [mirror_buf_fault_injection]="models"
    [nam_infer_test]="models"
    [namb_v2_roundtrip]="models"
    [namb_v2_validation]="models"
    [nondist_validation]="models"
    [parity_primitives]="parity"
    [prewarm_test]="models"
    [proptest_math]="models"
    [self_consistency]="models"
    [soak_test]="perf_soak"
    [spsc_pipeline]="perf_soak"
    [threshold_calibration]="models"
    [wavenet_lite_block_invariance]="models"
    [wavenet_prewarm_edge]="models"
    [zero_alloc_infer]="models"
)

STRUCT_TESTS=(
    a2_loader activation_precision adaptive_fsm_proptest cabsim_golden
    concurrency_stress container_slimmable diagnostic_bundle ebu_lufs_compliance
    fixture_b1_2_smoke linear_golden lstm_activation_precision
    lstm_model_dyn_validation mirror_buf_fault_injection nam_infer_test
    namb_v2_roundtrip namb_v2_validation nondist_validation parity_primitives
    prewarm_test proptest_math self_consistency soak_test spsc_pipeline
    threshold_calibration wavenet_lite_block_invariance wavenet_prewarm_edge
    zero_alloc_infer
)

_structural_entry_files_exist() {
    for entry in models perf_soak parity clap rt_constraints; do
        [ -f "tests/${entry}.rs" ] || return 1
    done
    return 0
}

# Detect whether clap-plugin feature is active (S5.T02).
# Checks NAM_FEATURES override first, then falls back to parsing Cargo.toml defaults.
_has_clap_plugin() {
    if [ -n "${NAM_FEATURES:-}" ]; then
        [[ ",$NAM_FEATURES," == *",clap-plugin,"* ]] && return 0
        return 1
    fi
    grep '^default' Cargo.toml 2>/dev/null | grep -q '"clap-plugin"'
}

# ── Phase 2/3 measurement oracle → entry-point mapping ──────────────────
declare -A MEASUREMENT_ENTRY_MAP=(
    [reference_oracle_f64]="parity"
    [spectral_fidelity]="models"
    [linear_fft_test]="models"
    [golden_vectors]="models"
    [isa_parity]="parity"
    [cpp_parity]="parity"
    [proptest_parsers]="models"
)

# Helper: builds cargo test args for measurement tests.
# Uses MEASUREMENT_ENTRY_MAP to find the right entry-point in Sprint 3 mode.
_cargo_meas() {
    local targets="$1"
    local filters="$2"
    local -a libtest_args=("${@:3}")

    for arg in "${libtest_args[@]}"; do
        if [[ "$arg" =~ ^-[^-] ]]; then
            echo -e "${RED}${BOLD}❌ Erro: argumento libtest malformado '$arg' (use -- duplo, não - simples)${NC}" >&2
            exit 1
        fi
    done

    local -a tests=($targets)
    if _structural_entry_files_exist || [ "${NAM_NEW_ARCH:-0}" = "1" ]; then
        local -A _eps=()
        local _filters=""
        for _t in "${tests[@]}"; do
            local _ep="${MEASUREMENT_ENTRY_MAP[$_t]:-models}"
            _eps[$_ep]=1
            _filters="${_filters}${_t}:: "
        done
        if [ -n "$filters" ]; then
            _filters="${_filters}${filters} "
        fi
        local _ep_flags=""
        for _ep in "${!_eps[@]}"; do _ep_flags="$_ep_flags --test $_ep"; done
        cargo test --release $_ep_flags -- $_filters "${libtest_args[@]}"
    else
        local _legacy=""
        for _t in "${tests[@]}"; do _legacy="$_legacy --test $_t"; done
        cargo test --release $_legacy -- $filters "${libtest_args[@]}"
    fi
}

if _structural_entry_files_exist || [ "${NAM_NEW_ARCH:-0}" = "1" ]; then
    # ── Sprint 3+ format: single `cargo test` across all entry-points ───────
    # With 5 entry-points (vs 50 test binaries), all non-ignored structural
    # tests are already grouped per entry-point — no filter needed. One
    # compilation per entry-point instead of 28.
    #
    # `--skip <module>::` excludes the measurement-oracle modules from this
    # DEBUG run. Without it they ran TWICE (debug here + release in Fase 2):
    # a pure waste of ~60-90s per run that also violated the phase's own
    # design ("EXCLUI os 5 oráculos de medida do §7") — debug floats are a
    # codegen "ghost" (docs/testing.md §7, Axis B) and quick_parity in debug
    # additionally triggered the lazy C++ CMake build mid-phase. The
    # `module::` suffix makes each skip an exact module-prefix match, so the
    # historical `--skip` name-collision problem (e.g. bare `test_oracle`
    # matching threshold_calibration) does not apply.
    # `--test clap` is conditionally included (S5.T02): when clap-plugin is
    # not active, all clap tests are #[cfg]-gated and the binary compiles with
    # 0 tests — a pure waste of ~15-30s. The feature is NOT in default features
    # (standalone + testing), so it's normally excluded. Use NAM_FEATURES env
    # var to override (e.g. NAM_FEATURES="standalone,testing,clap-plugin").
    _struct_targets="models perf_soak parity"
    if _has_clap_plugin; then
        _struct_targets="$_struct_targets clap"
    fi
    _struct_flags=""
    for _t in $_struct_targets; do
        _struct_flags="$_struct_flags --test $_t"
    done
    cargo test --lib $_struct_flags -- \
        --skip golden_vectors:: --skip linear_fft_test:: \
        --skip spectral_fidelity:: --skip reference_oracle_f64:: \
        --skip cpp_parity:: --skip isa_parity:: \
        --skip rt_deadline:: --skip rt_jitter::
else
    # ── Legacy flat-file format (pre-Sprint 3) ───────────────────────────
    cargo test --lib "${STRUCT_TESTS[@]/#/--test=}"
fi

# ── Fase 2: Oráculos de medida (release, docs/testing.md §7) ───────────────
# Gate autoritativo de floats de produção: medem o caminho de codegen que o
# usuário executa. Em debug mediriam um "fantasma" (sem contração FMA / vet.).
phase "Oráculos de medida (release — gate de floats de produção)..."

# Freshness gate (Épico E — bloqueante): detecta modelos .nam modificados sem
# regeneração do golden correspondente. Hard fail — o princípio "Todo Golden
# Deve Poder Falhar" não admite placebos.
if ! check_freshness; then
    exit 1
fi

MEASUREMENT_STATUS=0
GOLDEN_RAN=false
CPP_PARITY_SKIPPED=false

# Combina os oráculos em UMA invocação cargo por ramo de dependência, para que
# o nam-rs (rlib release) seja compilado UMA vez por ramo — não uma por teste.
# A rodada anterior recompilava nam-rs ~5× (~44s cada = ~220s desperdiçados).
# isa_parity exige --test-threads=1 (§7); os demais toleram (todos < 2s).

# Ramo A — sempre executáveis (deps committed: modelos .nam + f64_anchors /
# sinais sintéticos). f64 Oracle + Spectral Fidelity + Linear FFT (graceful
# skip when goldens absent — mathematical oracle tests always run).
# Ramo B — acrescenta golden_vectors (v1) + isa_parity (v2) quando goldens
# committed estão presentes (estes hard-fail sem goldens, por isso o gate).
if [ -f "tests/fixtures/golden_wavenet_standard.bin" ] && [ -f "tests/fixtures/golden_wavenet_standard_v2_48000.bin" ]; then
    GOLDEN_RAN=true
    echo -e "  ${BLUE}→ f64 Oracle + Spectral + Linear FFT + Golden v1 + ISA parity (release, 1 compilação)...${NC}"
    _cargo_meas "reference_oracle_f64 spectral_fidelity linear_fft_test golden_vectors isa_parity" \
        "" \
        --test-threads=1 --nocapture \
        || MEASUREMENT_STATUS=1
else
    echo -e "  ${YELLOW}ⓘ Golden vectors (v1/v2) não encontrados — golden_vectors + isa_parity pulados.${NC}"
    echo -e "  ${YELLOW}  Execute './tests/fixtures/golden_gen_build.sh' para gerá-los.${NC}"
    echo -e "  ${BLUE}→ f64 Oracle + Spectral Fidelity + Linear FFT (release, 1 compilação)...${NC}"
    _cargo_meas "reference_oracle_f64 spectral_fidelity linear_fft_test" \
        "" \
        --nocapture \
        || MEASUREMENT_STATUS=1
fi

# C++ Parity — invocação SEPARADA porque o filtro `quick_parity` (necessário
# para rodar só o subconjunto ágil) suprimiria os demais oráculos se combinado.
# Self-skip gracioso se o render C++ não estiver compilado.
if [ -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    # ── Preventive render compilation (S1.T10) ────────────────────────────
    # Build the C++ render binary before cargo test so the CMake build time
    # is isolated from the test output and doesn't trigger mid-phase.
    NAM_CORE_DIR="tests/fixtures/NeuralAmpModelerCore"
    RENDER_BUILD_DIR="build/namcore_render"
    RENDER_BIN="$RENDER_BUILD_DIR/Release/render"
    if [ ! -f "$RENDER_BIN" ]; then
        RENDER_BIN="$RENDER_BUILD_DIR/Debug/render"
    fi
    SKIP_CPP_PARITY=false
    if [ ! -f "$RENDER_BIN" ]; then
        echo -e "  ${BLUE}→ Compilando render C++ preventivamente (S1.T10)...${NC}"
        if [ -z "${CXX:-}" ]; then
            if command -v g++ >/dev/null 2>&1; then
                CXX=g++
            elif command -v clang++ >/dev/null 2>&1; then
                CXX=clang++
            fi
        fi
        if [ -z "$CXX" ]; then
            echo -e "  ${YELLOW}ⓘ Compilador C++ não encontrado — pulando cpp_parity.${NC}"
            SKIP_CPP_PARITY=true
        else
            source variables.env 2>/dev/null || true
            mkdir -p "$RENDER_BUILD_DIR"
            if cmake -S "$NAM_CORE_DIR" -B "$RENDER_BUILD_DIR" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CXX_COMPILER="$CXX" \
                -DCMAKE_CXX_STANDARD=20 \
                -DCMAKE_CXX_FLAGS="-w" \
                -DNAM_ENABLE_A2_FAST=ON > /dev/null 2>&1; then
                if cmake --build "$RENDER_BUILD_DIR" --target render -j"$(nproc)" > /dev/null 2>&1; then
                    echo -e "  ${GREEN}✓ Render C++ compilado preventivamente.${NC}"
                else
                    echo -e "  ${YELLOW}ⓘ cmake build falhou — pulando cpp_parity.${NC}"
                    SKIP_CPP_PARITY=true
                fi
            else
                echo -e "  ${YELLOW}ⓘ cmake configure falhou — pulando cpp_parity.${NC}"
                SKIP_CPP_PARITY=true
            fi
        fi
    fi

    if [ "$SKIP_CPP_PARITY" = true ]; then
        echo -e "  ${YELLOW}ⓘ cpp_parity pulado (render C++ não disponível).${NC}"
        CPP_PARITY_SKIPPED=true
    else
        echo -e "  ${BLUE}→ C++ Parity (quick_parity: LSTM + WaveNet CH16 + A2, live NAMCore)...${NC}"
        _cargo_meas "cpp_parity" \
            "quick_parity" \
            --nocapture || MEASUREMENT_STATUS=1
    fi
else
    echo -e "  ${YELLOW}ⓘ NeuralAmpModelerCore não encontrado. Execute './utils/mod-update.sh'.${NC}"
    echo -e "  ${YELLOW}  Pulando cpp_parity (paridade live C++).${NC}"
    CPP_PARITY_SKIPPED=true
fi

if [ "$MEASUREMENT_STATUS" -ne 0 ]; then
    echo -e "${RED}${BOLD}❌ Gate de oráculos de medida (release) falhou.${NC}"
    exit 1
fi

# ── Fase 3: Parser fuzzing ágil (release, --ignored) ───────────────────────
# Tier 1: robustez/segurança de parser. Contagem reduzida para agilidade first-line
# (o long suite roda a contagem completa 5000/100000). Override via env.
# (proptest_math — Tier 3: consistência/locator — já roda na Fase 1 e no long.)
phase "Parser fuzzing ágil (release)..."
PROPTEST_CASES="${NAM_QUICK_PROPTEST_CASES:-1000}" \
    _cargo_meas "proptest_parsers" \
        "" \
        --ignored --nocapture

# ── Resumo ──────────────────────────────────────────────────────────────────
if [ "$GOLDEN_RAN" = true ] && [ "$CPP_PARITY_SKIPPED" = false ]; then
    echo -e "${GREEN}${BOLD}================================================================${NC}"
    echo -e "${GREEN}${BOLD}      Todos os testes rápidos passaram! (estrutural + medida)     ${NC}"
    echo -e "${GREEN}${BOLD}================================================================${NC}"
elif [ "$GOLDEN_RAN" = true ]; then
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}    Testes rápidos passaram (cpp_parity pulado —                   ${NC}"
    echo -e "${YELLOW}${BOLD}     C++ render não disponível)                                     ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
elif [ "$CPP_PARITY_SKIPPED" = false ]; then
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}    Testes rápidos passaram (golden_vectors + isa_parity         ${NC}"
    echo -e "${YELLOW}${BOLD}     pulados — gere os golden vectors para cobertura completa)      ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
else
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}    Testes rápidos passaram (golden_vectors + isa_parity         ${NC}"
    echo -e "${YELLOW}${BOLD}     e cpp_parity pulados — gere goldens e C++ render para         ${NC}"
    echo -e "${YELLOW}${BOLD}     cobertura completa)                                            ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
fi
