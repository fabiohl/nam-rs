# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

FROM mcr.microsoft.com/devcontainers/rust:latest

# Prevents prompt interruptions during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install NAM-rs dependencies (native Pipewire audio and graphical interface) and utility tools
RUN apt update && apt -y full-upgrade && apt -y install --no-install-recommends \
    build-essential cmake g++ python3 pkg-config pipewire libpipewire-0.3-dev clang libclang-dev qpwgraph libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev git curl linux-tools-generic bolt-22 jq ripgrep fd-find \
    && ln -s /usr/bin/fdfind /usr/local/bin/fd || true \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Quick install of Cargo tools needed for tests and utility scripts
USER vscode
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
RUN cargo binstall -y cargo-edit cargo-pgo cargo-show-asm && \
    cargo install --git https://github.com/free-audio/clap-validator.git && \
    rustup component add llvm-tools-preview

USER root

# Tip: clean docker environment regularly:
## docker ps
## ID_DOCKER=
## docker stop $ID_DOCKER ; docker rm -f $ID_DOCKER ; docker ps ; docker builder prune -a ; docker system prune -a --volumes ; rm -rf target/
