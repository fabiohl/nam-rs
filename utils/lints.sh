#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and static analysis script for nam-rs.
# Verifies SPDX license headers, applies code formatting (rustfmt), and
# performs compilation (cargo check) and Clippy analysis across every
# relevant feature profile of the crate.
#
# Feature matrix (aligned with Cargo.toml):
#   Pure Core      : --no-default-features               (DSP lib parity with NAMcore)
#   Standalone prod: --no-default-features --features standalone   (bin nam-rs, no testing)
#   CLAP + testing : --no-default-features --features clap-plugin,testing (covers pgo_profiling_workload)
#   All            : --all-features --all-targets        (catch-all safety net)

set -euo pipefail

PHASE_TOTAL=5
source "$(dirname "$0")/_lib.sh"

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                 nam-rs Linting & Quality Suite                 ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# ---------------------------------------------------------------------------
# [1/5] Code formatting (applies rustfmt to normalize readability)
# ---------------------------------------------------------------------------
phase "Aplicando formatação de código (cargo fmt)..."
cargo fmt --all

# ---------------------------------------------------------------------------
# [2/5] SPDX license header validation (deterministic, no external tooling)
# ---------------------------------------------------------------------------
phase "Validando cabeçalhos SPDX de licença..."
# Candidate scope: Rust sources + shell scripts.
spdx_scope=$(
    {
        find src benches tests -type f -name '*.rs'
        find utils -maxdepth 1 -type f -name '*.sh'
    } || true
)
# `|| true` neutralizes grep's non-zero exit when no file matches the filter.
# (a) Files missing the SPDX-License-Identifier marker entirely.
missing=$(printf '%s\n' "$spdx_scope" | xargs grep -L "SPDX-License-Identifier" 2>/dev/null || true)
if [ -n "$missing" ]; then
    echo -e "  ${RED}${BOLD}Cabeçalho SPDX ausente nos arquivos:${NC}"
    echo "$missing" | sed 's/^/    /'
    exit 1
fi
# (b) Files whose SPDX identifier is not an approved license (Apache-2.0 | MIT).
# Restrict to files that actually carry the marker, then reject invalid identifiers.
invalid=$(printf '%s\n' "$spdx_scope" \
    | xargs grep -l "SPDX-License-Identifier" 2>/dev/null \
    | xargs grep -LE "SPDX-License-Identifier: (Apache-2\.0|MIT)" 2>/dev/null || true)
if [ -n "$invalid" ]; then
    echo -e "  ${RED}${BOLD}Identificador SPDX inválido (esperado Apache-2.0 ou MIT):${NC}"
    echo "$invalid" | sed 's/^/    /'
    exit 1
fi
echo -e "  ${GREEN}OK${NC} — todos os arquivos possuem cabeçalho SPDX válido (Apache-2.0, MIT)."

# ---------------------------------------------------------------------------
# [3/5] Anti-pattern: `#[test]` must not live in the shared tests/common/
# module, otherwise those tests are compiled and executed redundantly by every
# integration test that links the module.
# ---------------------------------------------------------------------------
phase "Verificando anti-padrão #[test] em tests/common/..."
if grep -rnF "#[test]" tests/common/ >/dev/null 2>&1; then
    echo -e "  ${RED}${BOLD}ERRO: '#[test]' encontrado em tests/common/ (execuções redundantes):${NC}"
    grep -rnF "#[test]" tests/common/ | sed 's/^/    /'
    exit 1
fi
echo -e "  ${GREEN}OK${NC} — nenhum '#[test]' em tests/common/."

# ---------------------------------------------------------------------------
# [4/5] Compilation checks (cargo check) across feature profiles
# ---------------------------------------------------------------------------
phase "Executando verificações de compilação (cargo check)..."

echo -e "  ${YELLOW}${BOLD}Checking: Pure Core (lib, no features)...${NC}"
cargo check --lib --no-default-features

echo -e "  ${YELLOW}${BOLD}Checking: Standalone prod (bin nam-rs, no testing)...${NC}"
# No --all-targets: integration tests require the `testing` feature, which is
# intentionally off here to mirror the production binary profile (lib + nam-rs).
cargo check --no-default-features --features standalone

echo -e "  ${YELLOW}${BOLD}Checking: CLAP Plugin + testing (pgo_profiling_workload)...${NC}"
cargo check --all-targets --no-default-features --features clap-plugin,testing

echo -e "  ${YELLOW}${BOLD}Checking: All Features (catch-all)...${NC}"
cargo check --all-targets --all-features

# ---------------------------------------------------------------------------
# [5/5] Static analysis (cargo clippy) — strict, same feature profiles
# ---------------------------------------------------------------------------
phase "Executando análise estática estrita (cargo clippy)..."

echo -e "  ${YELLOW}${BOLD}Clippy: Pure Core (lib, no features)...${NC}"
cargo clippy --lib --no-default-features -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: Standalone prod...${NC}"
# No --all-targets: see check phase rationale (tests need `testing`).
cargo clippy --no-default-features --features standalone -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: CLAP Plugin + testing...${NC}"
cargo clippy --all-targets --no-default-features --features clap-plugin,testing -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: All Features (catch-all)...${NC}"
cargo clippy --all-targets --all-features -- -D warnings

echo -e "\n${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}             Suíte de qualidade concluída com sucesso!           ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
