#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# golden_gen_build.sh — Compila o render tool do NeuralAmpModelerCore e gera os golden vectors
#
# Pré-requisitos:
#   - cmake >= 3.10, g++ ou clang++ com C++20
#   - python3 (para geração do WAV de stress)
#   - git (para clonagem do NeuralAmpModelerCore se necessário)
#
# Uso:
#   ./tests/fixtures/golden_gen_build.sh
#
# Saída (tests/fixtures/):
#   golden_wavenet_standard.bin, golden_wavenet_feather.bin, golden_wavenet_nano.bin
#   golden_lstm_1x16.bin, golden_lstm_2x8.bin
#   golden_namcore_lstm_1x3.bin, golden_namcore_wn_micro.bin
#
# Estes arquivos devem ser commitados para que os testes Rust de golden vectors
# executem sem recompilação C++.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAM_CORE_DIR="$SCRIPT_DIR/NeuralAmpModelerCore"
BUILD_DIR="$PROJECT_ROOT/build/namcore_render"
MODELS_DIR="$SCRIPT_DIR/models"
FIXTURES_DIR="$SCRIPT_DIR"

# =============================================================================
# Verificação de pré-requisitos
# =============================================================================
echo "=== Golden Vector Generator (NeuralAmpModelerCore) ==="

for cmd in cmake python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERRO: '$cmd' não encontrado. Instale: sudo apt install cmake python3"
        exit 1
    fi
done

# Verifica compilador C++20
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v g++ &>/dev/null; then
        CXX=g++
    elif command -v clang++ &>/dev/null; then
        CXX=clang++
    else
        echo "ERRO: Compilador C++ não encontrado. Instale g++ ou clang++."
        exit 1
    fi
fi
echo "  Compilador C++: $CXX"

# =============================================================================
# Clone/atualiza NeuralAmpModelerCore
# =============================================================================
echo ""
echo "[1/5] Inicializando NeuralAmpModelerCore..."
if [ ! -d "$NAM_CORE_DIR" ]; then
    git clone --depth 1 https://github.com/sdatkinson/NeuralAmpModelerCore.git "$NAM_CORE_DIR"
else
    echo "  NeuralAmpModelerCore já existe em $NAM_CORE_DIR"
fi

# Inicializa submodules
for sub in eigen AudioDSPTools; do
    sub_path="$NAM_CORE_DIR/Dependencies/$sub"
    if [ -d "$sub_path" ] && [ -z "$(ls -A "$sub_path" 2>/dev/null)" ]; then
        echo "  Inicializando submódulo $sub..."
        (cd "$NAM_CORE_DIR" && git submodule update --init "Dependencies/$sub")
    fi
done

# =============================================================================
# Compila render tool
# =============================================================================
echo ""
echo "[2/5] Compilando render tool..."

BUILD_TYPE="${BUILD_TYPE:-Release}"
RENDER_BIN="$BUILD_DIR/$BUILD_TYPE/render"

if [ -f "$RENDER_BIN" ]; then
    echo "  Render binário já existe: $RENDER_BIN"
else
    mkdir -p "$BUILD_DIR"
    cmake -S "$NAM_CORE_DIR" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_STANDARD=20 \
        2>&1 | tail -5
    cmake --build "$BUILD_DIR" --target render -j"$(nproc)" 2>&1 | tail -5

    if [ ! -f "$RENDER_BIN" ]; then
        # Tenta achar o binário em outro local
        RENDER_BIN=$(find "$BUILD_DIR" -name render -type f -executable | head -1)
        if [ -z "$RENDER_BIN" ]; then
            echo "ERRO: Falha ao compilar render tool."
            echo "Verifique se CMake e compilador C++20 estão funcionando."
            exit 1
        fi
    fi
fi
echo "  Render: $RENDER_BIN"

# =============================================================================
# Gera WAV stress signal via Python
# =============================================================================
echo ""
echo "[3/5] Gerando sinal de stress..."

STRESS_WAV="$FIXTURES_DIR/stress_signal.wav"

python3 -c "
import struct, math

SR = 48000.0
N  = 2048
T  = N / SR
PI = math.pi

attack_end  = int(0.002 * SR)    # 96 samples
release_beg = N - int(0.005 * SR) # 1808 samples

samples = []
for i in range(N):
    t = i / SR
    # Envelope (attack 2ms, sustain, release 5ms)
    if i < attack_end:
        env = i / attack_end
    elif i >= release_beg:
        env = (N - 1 - i) / (N - release_beg)
    else:
        env = 1.0

    # Guitarr harmonics (Low-E: 82.41 Hz)
    guitar = (0.40 * math.sin(2 * PI * 82.41 * t)
            + 0.25 * math.sin(2 * PI * 164.81 * t)
            + 0.15 * math.sin(2 * PI * 329.63 * t)
            + 0.08 * math.sin(2 * PI * 659.25 * t))

    # Linear chirp 220 Hz -> 3520 Hz
    f0, f1 = 220.0, 3520.0
    chirp_phase = 2 * PI * (f0 * t + (f1 - f0) * t * t / (2.0 * T))
    chirp = 0.30 * math.sin(chirp_phase)

    # Impulse at 25%
    impulse = 0.9 if i == N // 4 else 0.0

    sample = env * (guitar + chirp) + impulse
    sample = max(-1.0, min(1.0, sample))
    samples.append(sample)

# Escreve WAV header + f32 LE samples
num_samples = len(samples)
data_size = num_samples * 4
file_size = 44 + data_size

with open('$STRESS_WAV', 'wb') as f:
    f.write(b'RIFF')
    f.write(struct.pack('<I', file_size))
    f.write(b'WAVE')
    f.write(b'fmt ')
    f.write(struct.pack('<I', 16))       # fmt chunk size
    f.write(struct.pack('<H', 3))        # IEEE float
    f.write(struct.pack('<H', 1))        # mono
    f.write(struct.pack('<I', 48000))    # sample rate
    f.write(struct.pack('<I', 48000*4))  # byte rate
    f.write(struct.pack('<H', 4))        # block align
    f.write(struct.pack('<H', 32))       # bits per sample
    f.write(b'data')
    f.write(struct.pack('<I', data_size))
    for s in samples:
        f.write(struct.pack('<f', s))

print(f'  Stress signal: {num_samples} amostras, {file_size} bytes')
" 2>&1

# =============================================================================
# Executa render para cada modelo → WAV output → .golden.bin
# =============================================================================
echo ""
echo "[4/5] Executando render para cada modelo..."

# Modelos: (arquivo .nam, nome do golden, label)
MODELS=(
    "BossWN-standard.nam:golden_wavenet_standard:WaveNet Standard"
    "BossWN-feather.nam:golden_wavenet_feather:WaveNet Feather"
    "BossWN-nano.nam:golden_wavenet_nano:WaveNet Nano"
    "BossLSTM-1x16.nam:golden_lstm_1x16:LSTM 1×16"
    "BossLSTM-2x8.nam:golden_lstm_2x8:LSTM 2×8"
    "lstm.nam:golden_namcore_lstm_1x3:NAMCore LSTM 1×3"
    "wavenet.nam:golden_namcore_wn_micro:NAMCore WN Micro"
)

TEMP_DIR="$FIXTURES_DIR/.temp_golden"
mkdir -p "$TEMP_DIR"

for entry in "${MODELS[@]}"; do
    IFS=':' read -r nam_file golden_name label <<< "$entry"
    MODEL_PATH="$MODELS_DIR/$nam_file"
    OUTPUT_WAV="$TEMP_DIR/${golden_name}.wav"
    GOLDEN_BIN="$FIXTURES_DIR/${golden_name}.bin"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "  SKIP: $nam_file não encontrado em $MODELS_DIR"
        continue
    fi

    echo "  Processando $label ($nam_file)..."

    "$RENDER_BIN" "$MODEL_PATH" "$STRESS_WAV" "$OUTPUT_WAV" 2>&1 | tail -1

    if [ ! -f "$OUTPUT_WAV" ]; then
        echo "  ERRO: Render falhou para $label"
        continue
    fi

    # Converte WAV output → formato .golden.bin
    # Formato: [u32 N] [f32×N input] [f32×N output]
    python3 -c "
import struct, sys

# Lê input do stress WAV
with open('$STRESS_WAV', 'rb') as f:
    f.seek(44)
    inp = f.read()

# Lê output do render WAV
with open('$OUTPUT_WAV', 'rb') as f:
    f.seek(44)
    out = f.read()

n_bytes = min(len(inp), len(out))
n_samples = n_bytes // 4

# Valida que o header de data começa no offset 44
if len(inp) < 44*2 or len(out) < 44*2:
    print(f'  WARN: arquivos WAV muito pequenos', file=sys.stderr)
    sys.exit(1)

with open('$GOLDEN_BIN', 'wb') as f:
    f.write(struct.pack('<I', n_samples))
    f.write(inp[:n_bytes])
    f.write(out[:n_bytes])

file_size = 4 + 2 * n_bytes
print('  -> $golden_name.bin: {} amostras, {} bytes'.format(n_samples, file_size))
" 2>&1

done

# =============================================================================
# Limpeza
# =============================================================================
echo ""
echo "[5/5] Limpando arquivos temporários..."
rm -rf "$TEMP_DIR"

echo ""
echo "=== Golden vectors gerados com sucesso ==="
echo "  Arquivos em $FIXTURES_DIR/:"
for entry in "${MODELS[@]}"; do
    IFS=':' read -r _ golden_name _ <<< "$entry"
    [ -f "$FIXTURES_DIR/${golden_name}.bin" ] && echo "    ${golden_name}.bin"
done
echo ""
echo "Commite estes arquivos para que os testes Rust de golden vectors funcionem."
