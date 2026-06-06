#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Unified compiler-grade release build script for nam-rs (PGO + BOLT).
# Compiles the standalone binary and CLAP plugin with Profile-Guided Optimization
# and post-link BOLT binary reordering.
#
# Deliverables:
#   - ~/.local/bin/nam-rs  (PGO + BOLT optimized standalone binary)
#   - ~/.clap/nam-rs.clap  (PGO optimized CLAP plugin)

set -euo pipefail

# Style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}          nam-rs Unified Release Build & Optimization Pipeline ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configurations
PGO_DIR="/tmp/nam-rs-release-pgo"
BOLT_DIR="/tmp/nam-rs-release-bolt"
PROFRAW_DIR="$PGO_DIR/profraw"
MERGED_PROFILE="$PGO_DIR/merged.profdata"
ORIG_RUSTFLAGS="${RUSTFLAGS:-}"

# Read rustflags from .cargo/config.toml to avoid overriding them when we set RUSTFLAGS env var
CONFIG_RUSTFLAGS=$(python3 -c '
import re
with open(".cargo/config.toml") as f:
    content = f.read()
match = re.search(r"rustflags\s*=\s*\[(.*?)\]", content, re.DOTALL)
if match:
    block = match.group(1)
    flags = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        flag_match = re.search(r"\"([^\"]+)\"", line)
        if flag_match:
            flags.append(flag_match.group(1))
    print(" ".join(flags))
' 2>/dev/null || echo "")

# Deliverable Targets
BIN_INSTALL_DIR="$HOME/.local/bin"
CLAP_INSTALL_DIR="$HOME/.clap"
BIN_TARGET="$BIN_INSTALL_DIR/nam-rs"
CLAP_TARGET="$CLAP_INSTALL_DIR/nam-rs.clap"

# -----------------------------------------------------------------------------
# PHASE 1: Environment & Dependency Verification
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Phase 1/5] Verificando dependências e ambiente...${NC}"

# Find llvm-profdata from Rustup toolchain
RUST_SYSROOT="$(rustc --print sysroot)"
RUST_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
LLVM_PROFDATA="$RUST_SYSROOT/lib/rustlib/$RUST_TARGET/bin/llvm-profdata"
if [ ! -x "$LLVM_PROFDATA" ]; then
    echo -e "${RED}Erro: llvm-profdata não encontrado em $LLVM_PROFDATA${NC}"
    echo -e "${YELLOW}Instale as ferramentas do LLVM via rustup:${NC}"
    echo -e "  rustup component add llvm-tools-preview"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} llvm-profdata localizado: $LLVM_PROFDATA"

# Find LLVM BOLT
LLVM_BOLT=""
for candidate in \
    /usr/lib/llvm-22/bin/llvm-bolt \
    /usr/lib/llvm-21/bin/llvm-bolt \
    /usr/lib/llvm-20/bin/llvm-bolt \
    /usr/lib/llvm-19/bin/llvm-bolt \
    /usr/lib/llvm-18/bin/llvm-bolt \
    /usr/bin/llvm-bolt-22 \
    /usr/bin/llvm-bolt-21 \
    /usr/bin/llvm-bolt; do
    if [ -x "$candidate" ]; then
        LLVM_BOLT="$candidate"
        break
    fi
done

if [ -n "$LLVM_BOLT" ]; then
    echo -e "  ${GREEN}✓${NC} llvm-bolt localizado: $LLVM_BOLT"
    PERF2BOLT="$(dirname "$LLVM_BOLT")/perf2bolt"
    if [ ! -x "$PERF2BOLT" ]; then
        PERF2BOLT="perf2bolt"
    fi
else
    echo -e "${YELLOW}Aviso: llvm-bolt não foi encontrado. O build continuará apenas com PGO.${NC}"
    echo -e "${YELLOW}Para habilitar BOLT, instale: sudo apt install llvm-22-tools${NC}"
fi

# Find perf and check paranoid level
HAS_PERF=false
if command -v perf &>/dev/null; then
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
    if [ "$PARANOID" -le 1 ]; then
        HAS_PERF=true
        echo -e "  ${GREEN}✓${NC} perf está disponível (paranoid level: $PARANOID)"
    else
        echo -e "${YELLOW}Aviso: perf está instalado mas kernel.perf_event_paranoid=$PARANOID (>1).${NC}"
        echo -e "${YELLOW}BOLT necessita de paranoid <= 1 para amostragem sem privilégios root.${NC}"
        echo -e "${YELLOW}Execute: sudo sysctl -w kernel.perf_event_paranoid=1${NC}"
    fi
else
    echo -e "${YELLOW}Aviso: perf não encontrado. O build continuará apenas com PGO.${NC}"
fi

# Clean temp directories
rm -rf "$PGO_DIR" "$BOLT_DIR"
mkdir -p "$PROFRAW_DIR" "$BOLT_DIR"

# -----------------------------------------------------------------------------
# PHASE 2: Profile-Guided Optimization (PGO) - Profiling Workload
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Phase 2/5] Gerando perfis PGO através de benchmarks...${NC}"

# Compile and run with profile-generation (incorporating config.toml flags)
export RUSTFLAGS="$CONFIG_RUSTFLAGS $ORIG_RUSTFLAGS -Cprofile-generate=$PROFRAW_DIR"

echo -e "  Compilando e executando workload de inferência (inference_bench)..."
cargo bench --features "standalone,long_bench" --bench inference_bench

echo -e "  Compilando e executando kernels SIMD (dot_4x_bench)..."
cargo bench --features standalone --bench dot_4x_bench

PROFRAW_COUNT=$(find "$PROFRAW_DIR" -name "*.profraw" 2>/dev/null | wc -l)
if [ "$PROFRAW_COUNT" -eq 0 ]; then
    echo -e "${RED}Erro: Nenhum arquivo de perfil .profraw foi gerado em $PROFRAW_DIR!${NC}"
    exit 1
fi

echo -e "  ${GREEN}✓${NC} Coletados $PROFRAW_COUNT perfis .profraw. Mesclando..."
"$LLVM_PROFDATA" merge -sparse -o "$MERGED_PROFILE" "$PROFRAW_DIR"/*.profraw
echo -e "  ${GREEN}✓${NC} Perfil mesclado gerado em: $MERGED_PROFILE ($(du -h "$MERGED_PROFILE" | cut -f1))"

# Clean raw profiles
rm -rf "$PROFRAW_DIR"

# -----------------------------------------------------------------------------
# PHASE 3: Compile PGO-Optimized Standalone and CLAP plugin
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Phase 3/5] Compilando binários otimizados com PGO...${NC}"

export RUSTFLAGS="$CONFIG_RUSTFLAGS $ORIG_RUSTFLAGS -Cprofile-use=$MERGED_PROFILE"

echo -e "  Construindo executável standalone..."
cargo build --release --features "standalone,pgo" --bin nam-rs

echo -e "  Construindo plugin CLAP..."
RUSTFLAGS="$RUSTFLAGS -Clink-arg=-Wl,-soname,nam-rs.clap" \
    cargo build --release --target-dir target/clap --no-default-features --features "clap-plugin,pgo" --lib

# Confirm binaries compiled
if [ ! -f "target/release/nam-rs" ]; then
    echo -e "${RED}Erro: Falha ao encontrar o binário standalone compilado!${NC}"
    exit 1
fi
if [ ! -f "target/clap/release/libnam_rs.so" ]; then
    echo -e "${RED}Erro: Falha ao encontrar a biblioteca do plugin CLAP compilada!${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Compilação PGO concluída com sucesso."

# -----------------------------------------------------------------------------
# PHASE 4: Binary Optimization and Layout Tool (BOLT) Post-Link
# -----------------------------------------------------------------------------
BOLT_APPLIED=false
if [ -n "$LLVM_BOLT" ] && [ "$HAS_PERF" = true ]; then
    echo -e "\n${BLUE}${BOLD}[Phase 4/5] Aplicando otimização pós-link BOLT ao binário standalone...${NC}"

    # Verify if PipeWire is active for real-time profiling
    PW_RUNNING=false
    if command -v pw-cli &>/dev/null && pw-cli info &>/dev/null 2>&1; then
        PW_RUNNING=true
    fi

    # Find a representative .nam model for profiling
    MODEL_FILE=""
    for candidate in \
        "tests/nam_files/EVH-5150-Lite.nam" \
        "tests/nam_files/ChandlerRedd47-Gain34-Standard.nam" \
        "tests/nam_files/NEVE1073-Standard.nam" \
        "tests/nam_files/MRSH-JM50LD-Crunch2_FAT_CAB.nam"; do
        if [ -f "$candidate" ]; then
            MODEL_FILE="$candidate"
            break
        fi
    done

    if [ "$PW_RUNNING" = true ] && [ -n "$MODEL_FILE" ]; then
        echo -e "  PipeWire detectado! Iniciando perfilagem ativa do áudio..."

        # Generate 10s mono sine wave test signal
        TEST_WAV="$BOLT_DIR/test_signal.wav"
        if command -v ffmpeg &>/dev/null; then
            ffmpeg -y -f lavfi -i "sine=frequency=440:duration=10" \
                -ar 48000 -ac 1 -sample_fmt s16 \
                "$TEST_WAV" &>/dev/null
        elif command -v python3 &>/dev/null; then
            python3 -c "
import wave, struct, math
rate = 48000
duration = 10
n = rate * duration
with wave.open('$TEST_WAV', 'w') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    for i in range(n):
        val = int(32767 * 0.5 * math.sin(2 * math.pi * 440 * i / rate))
        w.writeframes(struct.pack('<h', val))
" &>/dev/null
        fi

        if [ -f "$TEST_WAV" ]; then
            # Start standalone in background
            target/release/nam-rs "$MODEL_FILE" -b 64 &
            NAM_PID=$!
            sleep 1.5

            if kill -0 $NAM_PID 2>/dev/null; then
                # Route and record samples
                NAM_NODE=$(pw-cli list-objects Node 2>/dev/null | grep -B1 -A5 "nam-rs" | grep -oP 'id \K\d+' | head -1 || echo "")
                
                PERF_ARGS=(-F 99 -e cycles:u -p "$NAM_PID" -o "$BOLT_DIR/perf.data")
                if perf record -e cycles:u -j any,u -F 99 -o /dev/null -- true &>/dev/null; then
                    PERF_ARGS=(-F 99 -e cycles:u -j any,u -p "$NAM_PID" -o "$BOLT_DIR/perf.data")
                fi

                echo -e "    Gravando eventos de ciclos de CPU..."
                if [ -n "$NAM_NODE" ]; then
                    pw-play --target="$NAM_NODE" "$TEST_WAV" &
                    PLAY_PID=$!
                    perf record "${PERF_ARGS[@]}" -- sleep 7 &>/dev/null || true
                    kill $PLAY_PID 2>/dev/null || true
                else
                    perf record "${PERF_ARGS[@]}" -- sleep 8 &>/dev/null || true
                fi

                kill $NAM_PID 2>/dev/null || true
                wait $NAM_PID 2>/dev/null || true
            else
                echo -e "${YELLOW}  Aviso: nam-rs falhou ao iniciar no PipeWire. Fallback para perfil de benchmark...${NC}"
                PW_RUNNING=false
            fi
        else
            echo -e "${YELLOW}  Aviso: Não foi possível gerar sinal de áudio. Fallback para perfil de benchmark...${NC}"
            PW_RUNNING=false
        fi
    else
        PW_RUNNING=false
    fi

    # Fallback to benchmark-based profiling if PipeWire wasn't possible
    if [ "$PW_RUNNING" = false ]; then
        echo -e "  Executando perfilagem através de benchmarks compilados..."
        # Compile a target bench binary with PGO-use to profile
        cargo bench --features "standalone,long_bench" --bench inference_bench --no-run
        INFERENCE_BENCH=$(find target/release/deps -maxdepth 1 -name "inference_bench-*" ! -name "*.d" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2 || echo "")
        
        if [ -n "$INFERENCE_BENCH" ]; then
            PERF_ARGS=(-F 99 -e cycles:u -o "$BOLT_DIR/perf.data")
            if perf record -e cycles:u -j any,u -F 99 -o /dev/null -- true &>/dev/null; then
                PERF_ARGS=(-F 99 -e cycles:u -j any,u -o "$BOLT_DIR/perf.data")
            fi
            perf record "${PERF_ARGS[@]}" -- "$INFERENCE_BENCH" --bench &>/dev/null || true
            # Swap benchmark binary path for symbol matching
            NAM_RS_BIN_FOR_BOLT="$INFERENCE_BENCH"
        else
            echo -e "${YELLOW}  Aviso: Não foi possível localizar o binário do benchmark. Pulando BOLT.${NC}"
        fi
    else
        NAM_RS_BIN_FOR_BOLT="target/release/nam-rs"
    fi

    # Apply perf2bolt and llvm-bolt if we have perf data
    if [ -f "$BOLT_DIR/perf.data" ] && [ -s "$BOLT_DIR/perf.data" ]; then
        echo -e "  Convertendo perfil com perf2bolt..."
        if "$PERF2BOLT" -p "$BOLT_DIR/perf.data" "$NAM_RS_BIN_FOR_BOLT" -o "$BOLT_DIR/perf.fdata" --ignore-build-id &>/dev/null; then
            echo -e "  Otimizando binário com llvm-bolt..."
            if "$LLVM_BOLT" target/release/nam-rs \
                -o target/release/nam-rs.bolt \
                -data "$BOLT_DIR/perf.fdata" \
                --reorder-blocks=cache+ \
                --reorder-functions=hfsort \
                --split-functions \
                --split-all-cold \
                --relocs \
                --lite &>/dev/null; then
                BOLT_APPLIED=true
                echo -e "  ${GREEN}✓${NC} BOLT aplicado com sucesso."
            else
                echo -e "${YELLOW}  Aviso: O comando llvm-bolt falhou. Revertendo para binário PGO padrão.${NC}"
            fi
        else
            echo -e "${YELLOW}  Aviso: perf2bolt falhou ao converter dados. Revertendo para binário PGO padrão.${NC}"
        fi
    else
        echo -e "${YELLOW}  Aviso: Nenhum dado de perf record foi coletado. Revertendo para binário PGO padrão.${NC}"
    fi
else
    echo -e "\n${YELLOW}[Phase 4/5] Pulando BOLT (llvm-bolt ou perf não estão disponíveis/configurados).${NC}"
fi

# -----------------------------------------------------------------------------
# PHASE 5: Deliverables Installation & Verification
# -----------------------------------------------------------------------------
echo -e "\n${BLUE}${BOLD}[Phase 5/5] Instalando e validando artefatos...${NC}"

# Target directories creation
mkdir -p "$BIN_INSTALL_DIR"
mkdir -p "$CLAP_INSTALL_DIR"

# Deliver standalone binary
rm -f "$BIN_TARGET"
if [ "$BOLT_APPLIED" = true ] && [ -f "target/release/nam-rs.bolt" ]; then
    cp target/release/nam-rs.bolt "$BIN_TARGET"
    echo -e "  Instalado executável (PGO + BOLT): $BIN_TARGET"
else
    cp target/release/nam-rs "$BIN_TARGET"
    echo -e "  Instalado executável (Apenas PGO): $BIN_TARGET"
fi
chmod +x "$BIN_TARGET"

# Deliver CLAP plugin
rm -f "$CLAP_TARGET"
cp target/clap/release/libnam_rs.so "$CLAP_TARGET"
echo -e "  Instalado plugin CLAP (PGO): $CLAP_TARGET"

# Audit binary properties
echo -e "\n  Auditando validade dos entregáveis:"

# 1. ELF format check
if file "$BIN_TARGET" | grep -q "ELF 64-bit"; then
    echo -e "    ${GREEN}✓${NC} Executável standalone é ELF 64-bit válido."
else
    echo -e "    ${RED}❌ Erro: Executável standalone inválido ou corrompido!${NC}"
    exit 1
fi

# 2. Shared object & soname verification of CLAP
if file "$CLAP_TARGET" | grep -q "ELF 64-bit"; then
    if readelf -d "$CLAP_TARGET" | grep -q SONAME; then
        echo -e "    ${GREEN}✓${NC} Plugin CLAP possui SONAME correto."
    else
        echo -e "    ${RED}❌ Erro: Plugin CLAP não possui SONAME!${NC}"
        exit 1
    fi
else
    echo -e "    ${RED}❌ Erro: Plugin CLAP inválido ou corrompido!${NC}"
    exit 1
fi

# 3. CLAP entry point verification
if nm -D "$CLAP_TARGET" | grep -q "clap_entry"; then
    echo -e "    ${GREEN}✓${NC} Plugin CLAP exporta o símbolo 'clap_entry'."
else
    echo -e "    ${RED}❌ Erro: Símbolo 'clap_entry' ausente no plugin!${NC}"
    exit 1
fi

# Cleanup temp files
rm -rf "$PGO_DIR" "$BOLT_DIR"

echo -e "\n${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}          Pipeline concluído! Artefatos prontos para distribuição.  ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "  Tamanho do Standalone: $(du -h "$BIN_TARGET" | cut -f1)"
echo -e "  Tamanho do CLAP:       $(du -h "$CLAP_TARGET" | cut -f1)"
echo -e "  Caminho Executável:    $BIN_TARGET"
echo -e "  Caminho CLAP Plugin:   $CLAP_TARGET"
echo -e "${GREEN}${BOLD}================================================================${NC}"
