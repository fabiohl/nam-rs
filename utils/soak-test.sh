#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

# Script para disparo manual da Suíte de Estabilidade Numérica (Soak Test).
# Este script executa testes de longa duração para validar a robustez dos algoritmos.

set -e

echo "==============================================================================="
echo "   NAM-rs: Bateria de Teste de Estabilidade Numérica (Soak Test)"
echo "==============================================================================="
echo "AVISO: Esta bateria de testes realiza milhões de iterações e pode durar"
echo "várias horas dependendo do hardware. O objetivo é estressar os algoritmos"
echo "para detectar drift, divergência ou instabilidades de longa duração."
echo ""
echo "Pressione CTRL+C para cancelar agora, ou aguarde 5 segundos para iniciar..."
sleep 5

echo "Iniciando testes em modo --release com --test-threads=1..."
echo "Os resultados serão gravados em 'soak-test.log'."
echo ""

# Executa a bateria de soak tests
# --release: garante kernels SIMD otimizados
# --ignored: executa os testes marcados com #[ignore]
# --nocapture: permite ver o progresso e métricas no console
# --test-threads=1: evita competição por CPU para métricas mais estáveis
cargo test --release --test soak_test -- --ignored --nocapture --test-threads=1 2>&1 | tee soak-test.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "==============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCESSO: Todos os soak tests passaram sem detecção de instabilidades."
else
    echo "FALHA: Foram detectadas instabilidades numéricas ou falhas nos testes."
fi
echo "Log completo disponível em: soak-test.log"
echo "==============================================================================="

exit $EXIT_CODE
