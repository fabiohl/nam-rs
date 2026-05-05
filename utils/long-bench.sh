#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

# Script para disparo manual de Benchmarks de Longa Duração (Long Run Benchmarks).
# Este script executa medições estatísticas prolongadas para validar o desempenho real
# fora de aleatoriedades de cache e jitter do SO.

set -e

echo "==============================================================================="
echo "   NAM-rs: Benchmarks de Longa Duração (Long Run Suite)"
echo "==============================================================================="
echo "AVISO: Estes benchmarks usam tempos de medição prolongados (30s por grupo) e"
echo "blocos grandes (4096 amostras) para garantir precisão estatística máxima."
echo "O processo pode levar vários minutos."
echo ""
echo "Pressione CTRL+C para cancelar agora, ou aguarde 5 segundos para iniciar..."
sleep 5

echo "Iniciando benchmarks com --features long_bench..."
echo "Os resultados serão processados pelo Criterion e salvos em 'target/criterion'."
echo ""

# Executa os benchmarks longos
# --features long_bench: ativa os grupos Long_Run_*
# --bench inference_bench: foca no benchmark principal
cargo bench --features long_bench --bench inference_bench 2>&1 | tee long-bench.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "==============================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCESSO: Benchmarks concluídos. Verifique os relatórios em target/criterion/report/index.html"
else
    echo "FALHA: Ocorreu um erro durante a execução dos benchmarks."
fi
echo "Log de execução disponível em: long-bench.log"
echo "==============================================================================="

exit $EXIT_CODE
