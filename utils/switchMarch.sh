#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva.

CONFIG_FILE=".cargo/config.toml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Erro: Arquivo $CONFIG_FILE não encontrado. Rode o script da raiz do projeto."
    exit 1
fi

OPT1_C='  # "-Ctarget-cpu=x86-64-v3",'
OPT1_A='  "-Ctarget-cpu=x86-64-v3",'

OPT2_C='  # "-Ctarget-cpu=x86-64-v3", "-Ztune-cpu=znver2",'
OPT2_A='  "-Ctarget-cpu=x86-64-v3", "-Ztune-cpu=znver2",'

OPT3_C='  # "-Ctarget-cpu=native",'
OPT3_A='  "-Ctarget-cpu=native",'

# Descobre o estado atual
STATE=0
if grep -qxF "$OPT1_A" "$CONFIG_FILE"; then STATE=1; fi
if grep -qxF "$OPT2_A" "$CONFIG_FILE"; then STATE=2; fi
if grep -qxF "$OPT3_A" "$CONFIG_FILE"; then STATE=3; fi

# Comenta todas as opções incondicionalmente para evitar estados quebrados
sed -i "s/^$OPT1_A$/$OPT1_C/" "$CONFIG_FILE"
sed -i "s/^$OPT2_A$/$OPT2_C/" "$CONFIG_FILE"
sed -i "s/^$OPT3_A$/$OPT3_C/" "$CONFIG_FILE"

# Ativa apenas a próxima opção do ciclo
if [ "$STATE" -eq 1 ]; then
    echo "Trocando de (Opção 1 - Distribuível) para (Opção 2 - Nightly Tuning)..."
    sed -i "s/^$OPT2_C$/$OPT2_A/" "$CONFIG_FILE"
elif [ "$STATE" -eq 2 ]; then
    echo "Trocando de (Opção 2 - Nightly Tuning) para (Opção 3 - Native)..."
    sed -i "s/^$OPT3_C$/$OPT3_A/" "$CONFIG_FILE"
else
    # Se for 3, ou 0 (estado quebrado de múltiplas ativas), volta para a opção primária e segura
    echo "Trocando para (Opção 1 - Distribuível)..."
    sed -i "s/^$OPT1_C$/$OPT1_A/" "$CONFIG_FILE"
fi

echo "Sucesso! O target-cpu foi alterado em $CONFIG_FILE."
