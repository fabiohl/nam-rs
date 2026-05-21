# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

FROM mcr.microsoft.com/devcontainers/rust:latest

# Evita interrupções de prompts durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instalação de dependências do NAM-rs (áudio Pipewire nativo e interface gráfica)
RUN apt update && apt -y full-upgrade && apt -y install --no-install-recommends \
    build-essential \
    pkg-config \
    pipewire \
    libpipewire-0.3-dev \
    clang \
    libclang-dev \
    cmake \
    libx11-xcb-dev \
    libxkbcommon-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libxcb-dri3-dev \
    libgtk-3-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Instalação rápida de ferramentas Cargo necessárias para testes e scripts de utilitários
USER vscode
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
RUN cargo binstall -y cargo-mutants cargo-edit cargo-nextest && \
    cargo install --git https://github.com/free-audio/clap-validator.git
USER root

# Dica: limpar ambiente docker regularmente:
## docker ps
## ID_DOCKER=
## docker stop $ID_DOCKER ; docker rm -f $ID_DOCKER ; docker ps ; docker builder prune -a ; docker system prune -a --volumes ; rm -rf target/
