FROM mcr.microsoft.com/devcontainers/rust:latest

# Evita interrupções de prompts durante a instalação
ENV DEBIAN_FRONTEND=noninteractive

# Instalação de dependências do NAM-rs
RUN apt update && apt -y full-upgrade && apt -y install --no-install-recommends \
    build-essential \
    pkg-config \
    pipewire \
    libpipewire-0.3-dev \
    clang \
    libclang-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
