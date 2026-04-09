---
name: debugger
description: Acionado quando algo não funciona como esperado. Atua como um painel multi-disciplinar de analistas e engenheiros seniores capazes de resolver os problemas mais difíceis.
---

# Skill: Debugger

## When to use this skill

Use esta skill sob a manifestação de um erro ativo, crash, comportamento não-linear de DSP, áudio truncado com modulação/ringing, xruns do PipeWire ou redes neurais produzindo degradação nos cálculos de áudio em tempo real.

## Contexto de Referência (leia antes de diagnosticar)

O NAM-rs opera com a seguinte arquitetura consolidada (Sprint 8):

- **Thread DSP** (`process()` do PipeWire): `SCHED_FIFO` prioridade 90, fixada em CPU via `pthread_setaffinity_np`. **ZERO** alocação heap, **ZERO** I/O, **ZERO** locks.
- **Comunicação RT→Main**: via `RtStatusFlags` (`AtomicU32`/`AtomicBool` em `Arc`). A thread principal lê as flags e faz I/O (prints/logs).
- **Canal SPSC de resamplers**: `rtrb::Producer<NamResampler>` / `Consumer<NamResampler>`. O `NamResampler` (é construído pela thread principal com `rubato::SincFixedIn<f32>`) e enviado para o callback via este canal — **nenhuma alocação no `process()`**.
- **Canal SPSC de parâmetros**: `rtrb::Producer<ParamPayload>` / `Consumer<ParamPayload>`. `ParamPayload` é `#[repr(align(128))]`.
- **Canal GC**: modelos obsoletos (`Box<DynamicModel>`) são enviados via SPSC para drop em thread separada (nunca no callback RT).
- `io_uring`, E/S de disco e qualquer acesso bloqueante são **categóricamente proibidos** e nunca devem aparecer no projeto.

## Instructions

### 1. Diagnóstico Primeiro — Não Remende Cegamente

- Não proponha patches aleatórios. Cada alocação ou lock no callback representa jitter catástrofico no PipeWire.
- **Xruns / clicks**: O `process()` está respeitando o budget de tempo do buffer (não há `Vec`, `Box` ou `println!` dentro do closure)? O `NamResampler` foi trocado via `resampler_consumer.pop()` sem alocação?
- **Degradação numérica**: Os multiplicadores de ganho são recalculados só quando `param_changed == true`? O `apply_gain_simd` está operando nos slices corretos (antes do downsample e após o upsample)?
- **Deadlock / contenção**: Existe algum `Mutex`, `RwLock` ou `Condvar` tocando o caminho RT? Verifique se `rtrb` está sendo usado corretamente (um único produtor, um único consumidor por canal).
- **Flags atômicas**: `RtStatusFlags` usa `Ordering::Relaxed` para performance — certifique-se de que a sequência de leitura/escrita na thread principal respeita a convergência das flags sem corrida de dados.

### 2. Intervenção Baseada em Evidências

- Antes de propor correções SIMD, observe os registradores AVX2 (YMM) via `std::simd` com `const generics` (SoA). Mantenha `#[repr(align(128))]` em `ParamPayload` e em qualquer estrutura nova tocada pelo buffer SPSC.
- **Proibido** no `process()`: `Vec::new()`, `Box::new()`, `println!`, `eprintln!`, `unwrap()` sem fallback silencioso, `Mutex::lock()`, qualquer syscall bloqueante.
- Após sanado, remova **toda** linha de log, `dbg!()` ou `eprintln!` inserida para depuração visual dentro do callback RT. Use `RtStatusFlags` para comunicar estado.
