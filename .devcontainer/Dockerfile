# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

FROM mcr.microsoft.com/devcontainers/rust:latest

# Prevents prompt interruptions during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install NAM-rs dependencies (native Pipewire audio and graphical interface)
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

# Quick install of Cargo tools needed for tests and utility scripts
USER vscode
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
RUN cargo binstall -y cargo-mutants cargo-edit cargo-nextest && \
    cargo install --git https://github.com/free-audio/clap-validator.git
USER root

# Tip: clean docker environment regularly:
## docker ps
## ID_DOCKER=
## docker stop $ID_DOCKER ; docker rm -f $ID_DOCKER ; docker ps ; docker builder prune -a ; docker system prune -a --volumes ; rm -rf target/
