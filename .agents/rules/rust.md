---
trigger: glob
description: Diretrizes Rust para Áudio RT, Inferência Neural e Plugins.
globs: **/*.rs, **/*.toml
---

<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Diretrizes Técnicas e de Performance (NAM-rs)

Este documento define as regras de desenvolvimento estritas para a base de código Rust do NAM-rs, focando em segurança em tempo real (RT-Safety), otimização DSP/SIMD e prevenção de regressões de latência.

---

## 1. Segurança Absoluta em Tempo Real (RT-Safety)

A thread de áudio de alta prioridade (DSP callback) tem um tempo limite (budget) estrito por bloco. Um único atraso gera falhas de sincronia (XRUNs/audio glitches).

* **Evite Desalocações Implícitas (Zero Heap Drop):**
  * *Problema:* Passar ou manter objetos alocados no heap (`Box`, `Vec`, `Arc`, strings) na thread RT e deixá-los sair de escopo (drop) invoca o alocador implicitamente, bloqueando a thread.
  * *Solução (Drop-Delegation):* Transfira a propriedade de objetos obsoletos de volta à thread principal via canal SPSC de Garbage Collection (`gc_producer.push(GcItem::...)`). Nunca deixe um `GcItem` ser descartado no hot-path.
* **Zero I/O e Operações Bloqueantes:**
  * *Problema:* Chamadas a `println!`, `eprintln!`, `format!`, gravação/leitura de arquivos ou locks bloqueiam a thread.
  * *Solução:* Sinalize estados e estatísticas atomicamente através do bitmask e contadores em `RtStatusFlags` (ex: `rt_status.set_flag(RT_STATUS_HAS_CLIPPED)`). A thread principal (CLI/UI) fará o polling e o logging.
* **Panic Prevention & Unwrapping:**
  * stack unwinding aloca memória e quebra o determinismo do RT.
  * Evite `unwrap()` e `expect()`. Use `.get()` ou `.get_mut()` com fallbacks seguros.
  * Estruture iterações e loops de forma a permitir que o compilador verifique e elimine checks de limites (*bounds checks*) estaticamente.

---

## 2. DSP & Matemática RT-Safe (Evitando Slowdowns Físicos)

* **Prevenção de Subnormais (Denormal Floats):**
  * *Problema:* Sinais de áudio que decaem exponencialmente (filtros, gates, estados de redes neurais) geram números subnormais (ex: `< 1e-38`). O processamento de subnormais em CPU x86-64 sem flags especiais é extremamente lento (resolvido por microcódigo), disparando o uso de CPU de 1% para 100% (Virtual XRUNs).
  * *Solução:* Configure explicitamente as flags FTZ (Flush-To-Zero) e DAZ (Denormals-Are-Zero) no início do loop de processamento ou zere ativamente variáveis de estado cujo módulo seja inferior a um limite seguro (ex: `if val.abs() < 1e-15 { val = 0.0; }`).
* **Aproximações Polinomiais Rápidas (Transcendentes):**
  * Funções transcendentes nativas como `f32::tanh()` ou `f32::exp()` são proibitivas no hot-path neural.
  * Utilize aproximações racionais de Padé ou polinômios Minimax de ordem mínima necessária. Assegure que o erro de aproximação esteja abaixo de `-80 dB` (aprox. `1e-4`) para manter a transparência tonal do áudio.
* **Casting de Ponto Flutuante:**
  * Evite conversões frequentes entre `f32` e inteiros usando `as`. Utilize `.round()`, `.floor()`, ou operações vetorizadas apropriadas para evitar interrupções no pipeline de execução da CPU.

---

## 3. SIMD & Otimização do Compilador (Auto-Vetorização)

Para alcançar a meta `x86-64-v3`, o código DSP deve ser amigável para auto-vetorização (AVX2/FMA).

* **Loops Amigáveis ao Compilador:**
  * Use `.chunks_exact(N)` ou `.chunks_exact_mut(N)`. Isso remove o código de tratamento de "cauda" (tail loops) e dá ao compilador o tamanho exato para alinhar vetores SIMD.
  * Una iteradores usando `zip` em vez de indexação manual em múltiplos arrays.
  * Mantenha os corpos dos loops DSP livres de branches condicionais complexos. Dê preferência a operações branchless (como seleções aritméticas e máscaras SIMD).
* **Alocação Alinhada (Cache Lines):**
  * Sempre use `AlignedVec<T>` (alinhamento a 64 bytes) para buffers dinâmicos, coeficientes de filtros e tensores de inferência neural. Isso previne penalidades severas de hardware devido a *unaligned memory loads/stores*.

---

## 4. Programação Concorrente Lock-Free (SPSC & Cache-Bouncing)

* **Mitigação de False Sharing:**
  * Qualquer estrutura compartilhada entre a thread de áudio e a thread principal (ex: `ParamPayload`, `RtStatusFlags`) deve ser anotada com `#[repr(align(128))]` para garantir que campos que sofrem mutação constante residam em linhas de cache separadas.
* **Ordenamento Atômico Preciso:**
  * Nunca use `Ordering::SeqCst` no hot-path de processamento.
  * Use `Ordering::Relaxed` para flags de telemetria geral (contadores de clipping, overloads, tempos de ciclo).
  * Use `Ordering::Acquire` e `Ordering::Release` estritamente na interface de buffers circulares (SPSC) para sincronizar ponteiros de leitura/escrita.
