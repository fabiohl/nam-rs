---
name: debugger
description: Acionado quando algo não funciona como esperado. Atua como um painel multi-disciplinar de analistas e engenheiros seniores capazes de resolver os problemas mais difíceis.
---

# Skill: Debugger

## When to use this skill

Use esta skill sob a manifestação de um erro ativo, crash, comportamento não-linear de DSP, áudio truncado com modulação/ringing, xruns do PipeWire ou redes neurais produzindo degradação nos cálculos de áudio em tempo real.

Também é acionada automaticamente pela workflow `/diagnostico` quando o usuário cola um bloco de suporte gerado pelo NAM-rs.

## Contexto de Referência (leia antes de diagnosticar)

O NAM-rs opera com a seguinte arquitetura consolidada:

- **Thread DSP** (`process()` do PipeWire): `SCHED_FIFO` prioridade 90, fixada em CPU via `pthread_setaffinity_np`. **ZERO** alocação heap, **ZERO** I/O, **ZERO** locks.
- **Comunicação RT→Main**: via `RtStatusFlags` (`AtomicU32`/`AtomicBool` em `Arc`). A thread principal lê as flags e faz I/O (prints/logs).
- **Canal SPSC de resamplers**: `rtrb::Producer<NamResampler>` / `Consumer<NamResampler>`. O `NamResampler` (é construído pela thread principal com `rubato::SincFixedIn<f32>`) e enviado para o callback via este canal — **nenhuma alocação no `process()`**.
- **Canal SPSC de parâmetros**: `rtrb::Producer<ParamPayload>` / `Consumer<ParamPayload>`. `ParamPayload` é `#[repr(align(128))]`.
- **Canal GC**: modelos obsoletos (`Box<DynamicModel>`) são enviados via SPSC para drop em thread separada (nunca no callback RT).
- `io_uring`, E/S de disco e qualquer acesso bloqueante são **categóricamente proibidos** e nunca devem aparecer no projeto.

## Sistema de Diagnósticos Estruturados (`src/diagnostics.rs`)

O NAM-rs possui um sistema de diagnósticos com códigos de erro tipados. Ao receber um erro do usuário, **sempre** extraia e interprete o bloco de suporte antes de investigar o código-fonte.

### Anatomia de um Bloco de Suporte

```text
──── NAM-rs Diagnostic ────────────────────────────
nam-rs v0.7.0 | E1201 | NAMB_CRC32_MISMATCH
expected=0xA3B4C5D6 computed=0x12345678
file="MesaBoogie.namb" size=1048576 offset=80
arch=x86_64 avx2=true fma=true
os=linux kernel=6.8.0-generic
pw_rate=48000 buffer=256
timestamp=2026-04-10T01:56:18Z
────────────────────────────────────────────────────
```

### Campos-chave para triagem rápida

| Campo              | Significado                                        | Ação                                            |
| ------------------ | -------------------------------------------------- | ----------------------------------------------- |
| **Código `Exxxx`** | Categoria do erro (ver tabela abaixo)              | Direciona para módulo/função exata              |
| **Mnemônico**      | Nome legível do erro                               | Buscar no `src/diagnostics.rs` → `NamErrorCode` |
| **Parâmetros**     | Contexto contextual (arquivo, tamanho, rate, etc.) | Reproduzir cenário exato                        |
| **arch/avx2/fma**  | Features da CPU do usuário                         | Descartar problemas de SIMD                     |
| **os/kernel**      | Ambiente do OS                                     | Identificar bugs kernel-specific                |
| **timestamp**      | Momento da falha                                   | Correlacionar com logs externos                 |

### Catálogo de Códigos de Erro

| Faixa     | Categoria              | Onde investigar                                        |
| --------- | ---------------------- | ------------------------------------------------------ |
| **E1xxx** | Carregamento de modelo | `src/loader/`, `src/main.rs::load_and_send_model()`    |
| **E2xxx** | PipeWire / Áudio       | `src/pw_host.rs`, `src/dsp/resampler.rs`               |
| **E3xxx** | SPSC / Comunicação     | `src/spsc.rs`, `src/main.rs::cli_loop()`               |
| **E4xxx** | Runtime / CLI          | `src/main.rs::cli_loop()`, `src/main.rs::parse_args()` |
| **E5xxx** | Sistema / Hardware     | `src/main.rs::main()` (detecção AVX2/FMA)              |

### Procedimento ao receber um bloco de suporte

1. **Extraia o código `Exxxx`** e localize a variante em `NamErrorCode` (`src/diagnostics.rs`).
2. **Leia os parâmetros** para entender o contexto exato (arquivo, tamanho, rates, etc.).
3. **Cruze com o módulo correto** usando a tabela acima.
4. **Reproduza o cenário** com os parâmetros do bloco (se possível): arquivo do modelo informado, rate do PipeWire, versão, etc.
5. **Diagnostique a causa-raiz** antes de propor patches — siga as regras abaixo.

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

### 3. Correções Devem Usar o Sistema de Diagnósticos

- Se o fix introduzir novos pontos de falha fora da thread RT, **sempre** emita um `NamDiagnostic` estruturado com o código de erro apropriado (do `NamErrorCode` existente, ou propondo um novo).
- **Nunca** use `eprintln!("Erro: ...")` ou `println!("AVISO: ...")` ad-hoc para erros que o usuário pode ver. Use o builder `NamDiagnostic::new(code, &sys).message(...).hint(...).param(...).emit()`.
- Mensagens mundanas informativas (não-erros) continuam com `println!("[CLI] ...")` ou `println!("[NAM-rs] ...")`.
