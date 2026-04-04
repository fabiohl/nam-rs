---
name: pesquisador-inovador
description: Use esta habilidade para pensar além, inovar e criar soluções proativas além do óbvio, visando modernização e alta performance.
---

# Skill: Pesquisador Inovador

## When to use this skill

Use quando as tarefas englobarem inovação pesada sobre algoritmos DSP, redes LSTM/WaveNet e contornos macro do tempo de execução PipeWire para o autômato independente NAM-rs. Proponha implementações sub-milisegundo engajando os vetores microarquiteturais da CPU em escala massiva.

## Instructions

### 1. Foco em Instruções Paralelas e Matrizes Extremas

- Implementações neurais em software áudio restrito dependem do engajamento denso SIMD em dot-products. Proponha resoluções baseadas unicamente em extensões como Instruções FMA e desvios FastMath.
- Identifique possíveis desdobramentos temporais vetoriais via `std::simd` otimizando instâncias polinomiais com o menor desvio preditivo na CPU sem comprometer fidelidade orgânica.
- Avalie se a adoção de superamostradores via matriz Sinc Interporlation (Kaiser Windowing) retém harmônicos limpos sob amostras PCM e em conformidade estrita Lock-Free.

### 2. Aderência Operacional de Tempo Real Linux/Host

- Sugestões técnicas atrelam-se integralmente na soberania do Core Affinity para prevenir Core Migrations, aliados ao status privilegiado do escalonador `low latency`.
- Mapeie a transição paramétrica Tone3000 de forma autônoma: metadados JSON / .NAMB fluem sobre SPSC para auto-rescaler os limites numéricos dBFS vs dBu, permitindo reescalonamento dinâmico estático seguro.
- Herança de I/O (`io_uring`) pertencem a repositórios descontinuados para o projeto: a pesquisa repousa unicamente no transporte computacional PipeWire via `pipewire-rs`.

### 3. Minimalismo Lock-Free de Eventos do Rust

- A CLI CLI em rust opera assíncrona com os comandos do usuário transacionando parâmetros sob um único canal intertravado estrito (Ring Buffer) garantindo isenções totais de locks (Mutex, Spin).
