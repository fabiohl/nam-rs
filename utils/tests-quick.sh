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
#     novo teste estrutural em tests/, inclua-o em STRUCTURAL_TESTS abaixo
#     (vide inventário em docs/testing.md §3). A auto-descoberta de unit tests
#     (lib) é preservada.

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ── Freshness gate (Governança de frescor — Épico E) ────────────────────────
# Checks the versioned golden freshness manifest against current model files.
# Fail hard on staleness: a golden from a modified .nam is a non-starter per
# the "Todo Golden Deve Poder Falhar" principle (tests/fixtures/README.md §Principle).
# Returns 0 if all models match, 1 if any model is stale or manifest missing.
check_freshness() {
    local MANIFEST="tests/fixtures/.golden_manifest.sha256"
    local MODELS_DIR="tests/fixtures/models"

    if [ ! -f "$MANIFEST" ]; then
        echo -e "${RED}${BOLD}❌ Freshness manifest missing: $MANIFEST${NC}"
        echo -e "${RED}   Run './tests/fixtures/golden_gen_build.sh' to generate goldens and manifest.${NC}"
        return 1
    fi

    local STALE_COUNT=0
    while IFS= read -r line; do
        [[ "$line" == \#* ]] && continue
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

    if [ "$STALE_COUNT" -gt 0 ]; then
        echo -e "${RED}${BOLD}❌ Freshness gate FAILED: $STALE_COUNT model(s) stale.${NC}"
        echo -e "${RED}   Run './tests/fixtures/golden_gen_build.sh' to regenerate goldens and manifest.${NC}"
        return 1
    fi
    echo -e "  ${GREEN}✓ Freshness gate passed (all model hashes match manifest).${NC}"
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

echo -e "${BLUE}${BOLD}=====================================${NC}"
echo -e "${BLUE}${BOLD}   nam-rs Quick QA Suite"
echo -e "${BLUE}${BOLD}   ± 35 seconds on "hot" target dir${NC}"
echo -e "${BLUE}${BOLD}   ± 5,5 minutes on "cold" target dir${NC}"
echo -e "${BLUE}${BOLD}=====================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Fase 1: Estrutural (debug) ──────────────────────────────────────────────
# Unit tests (lib, auto-descobertos) + integração determinística (lista explícita).
# Exclui os 5 oráculos de medida do §7 (→ Fase 2 release) e rt_deadline (→ long).
# debug-assertions ON captura invariantes baratos que --release mascararia.
echo -e "\n${BLUE}${BOLD}[1/3] Estrutural: unit + integração determinística (debug)...${NC}"

STRUCTURAL_TESTS=(
    a2_loader activation_precision adaptive_fsm_proptest cabsim_golden
    concurrency_stress container_slimmable diagnostic_bundle ebu_lufs_compliance
    fixture_b1_2_smoke linear_fft_test linear_golden lstm_activation_precision
    lstm_model_dyn_validation mirror_buf_fault_injection nam_infer_test
    namb_v2_roundtrip namb_v2_validation nondist_validation parity_primitives
    prewarm_test proptest_math self_consistency soak_test spsc_pipeline
    threshold_calibration wavenet_lite_block_invariance wavenet_prewarm_edge
    zero_alloc_infer
)
cargo test --lib "${STRUCTURAL_TESTS[@]/#/--test=}"

# ── Fase 2: Oráculos de medida (release, docs/testing.md §7) ───────────────
# Gate autoritativo de floats de produção: medem o caminho de codegen que o
# usuário executa. Em debug mediriam um "fantasma" (sem contração FMA / vet.).
echo -e "\n${BLUE}${BOLD}[2/3] Oráculos de medida (release — gate de floats de produção)...${NC}"

# Freshness gate (Épico E — bloqueante): detecta modelos .nam modificados sem
# regeneração do golden correspondente. Hard fail — o princípio "Todo Golden
# Deve Poder Falhar" não admite placebos.
if ! check_freshness; then
    exit 1
fi

MEASUREMENT_STATUS=0
GOLDEN_RAN=false

# Combina os oráculos em UMA invocação cargo por ramo de dependência, para que
# o nam-rs (rlib release) seja compilado UMA vez por ramo — não uma por teste.
# A rodada anterior recompilava nam-rs ~5× (~44s cada = ~220s desperdiçados).
# isa_parity exige --test-threads=1 (§7); os demais toleram (todos < 2s).

# Ramo A — sempre executáveis (deps committed: modelos .nam + f64_anchors /
# sinais sintéticos). f64 Oracle + Spectral Fidelity.
# Ramo B — acrescenta golden_vectors (v1) + isa_parity (v2) quando goldens
# committed estão presentes (estes hard-fail sem goldens, por isso o gate).
if [ -f "tests/fixtures/golden_wavenet_standard.bin" ] && [ -f "tests/fixtures/golden_wavenet_standard_v2_48000.bin" ]; then
    GOLDEN_RAN=true
    echo -e "  ${BLUE}→ f64 Oracle + Spectral + Golden v1 + ISA parity (release, 1 compilação)...${NC}"
    cargo test --release \
        --test reference_oracle_f64 --test spectral_fidelity \
        --test golden_vectors --test isa_parity \
        -- --test-threads=1 --nocapture || MEASUREMENT_STATUS=1
else
    echo -e "  ${YELLOW}ⓘ Golden vectors (v1/v2) não encontrados — golden_vectors + isa_parity pulados.${NC}"
    echo -e "  ${YELLOW}  Execute './tests/fixtures/golden_gen_build.sh' para gerá-los.${NC}"
    echo -e "  ${BLUE}→ f64 Oracle + Spectral Fidelity (release, 1 compilação)...${NC}"
    cargo test --release \
        --test reference_oracle_f64 --test spectral_fidelity \
        -- --nocapture || MEASUREMENT_STATUS=1
fi

# C++ Parity — invocação SEPARADA porque o filtro `quick_parity` (necessário
# para rodar só o subconjunto ágil) suprimiria os demais oráculos se combinado.
# Self-skip gracioso se o render C++ não estiver compilado.
if [ -d "tests/fixtures/NeuralAmpModelerCore" ]; then
    echo -e "  ${BLUE}→ C++ Parity (quick_parity: LSTM + WaveNet CH16 + A2, live NAMCore)...${NC}"
    cargo test --release --test cpp_parity -- quick_parity --nocapture || MEASUREMENT_STATUS=1
else
    echo -e "  ${YELLOW}ⓘ NeuralAmpModelerCore não encontrado. Execute './utils/mod-update.sh'.${NC}"
    echo -e "  ${YELLOW}  Pulando cpp_parity (paridade live C++).${NC}"
fi

if [ "$MEASUREMENT_STATUS" -ne 0 ]; then
    echo -e "${RED}${BOLD}❌ Gate de oráculos de medida (release) falhou.${NC}"
    exit 1
fi

# ── Fase 3: Parser fuzzing ágil (release, --ignored) ───────────────────────
# Tier 1: robustez/segurança de parser. Contagem reduzida para agilidade first-line
# (o long suite roda a contagem completa 5000/100000). Override via env.
# (proptest_math — Tier 3: consistência/locator — já roda na Fase 1 e no long.)
echo -e "\n${BLUE}${BOLD}[3/3] Parser fuzzing ágil (release)...${NC}"
PROPTEST_CASES="${NAM_QUICK_PROPTEST_CASES:-1000}" \
    cargo test --release --test proptest_parsers -- --ignored --nocapture

# ── Resumo ──────────────────────────────────────────────────────────────────
if [ "$GOLDEN_RAN" = true ]; then
    echo -e "${GREEN}${BOLD}================================================================${NC}"
    echo -e "${GREEN}${BOLD}      Todos os testes rápidos passaram! (estrutural + medida)     ${NC}"
    echo -e "${GREEN}${BOLD}================================================================${NC}"
else
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}    Testes rápidos passaram (golden_vectors + isa_parity         ${NC}"
    echo -e "${YELLOW}${BOLD}     pulados — gere os golden vectors para cobertura completa)      ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
fi
