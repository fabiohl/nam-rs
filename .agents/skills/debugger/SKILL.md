---
name: debugger
description: Acionado quando algo não funciona como esperado. Atua como um painel multi-disciplinar de analistas e engenheiros seniores capazes de resolver os problemas mais difíceis.
---

# Skill: Debugger

## When to use this skill

Use esta skill sob a manifestação de um erro ativo, crash, comportamento não-linear de DSP, áudio truncado com modulação/ringing, xruns do PipeWire ou redes neurais produzindo degradação nos cálculos de áudio em tempo real.

Também é acionada automaticamente pela workflow `/diagnostico` quando o usuário cola um bloco de suporte gerado pelo NAM-rs.

## Contexto de Referência

Vide `.agents/rules/rust.md` para diretrizes técnicas mandatórias (RT-Safety, SIMD, SPSC).

O NAM-rs opera com canais SPSC `rtrb` para parâmetros, modelos e resamplers, além de sinalização via `RtStatusFlags` (`AtomicU32`/`AtomicBool` em `Arc`). **ZERO alocação heap, ZERO I/O e ZERO locks** no caminho RT são condições inegociáveis.

## Sistema de Diagnósticos Estruturados (`src/diagnostics.rs`)

O NAM-rs possui um sistema de diagnósticos com códigos de erro tipados. Ao receber um erro do usuário, **sempre** extraia e interprete o bloco de suporte antes de investigar o código-fonte.

### Anatomia de um Bloco de Suporte

```text
──── NAM-rs Diagnostic ────────────────────────────
nam-rs v1.x.x | E1201 | NAMB_CRC32_MISMATCH
expected=0xA3B4C5D6 computed=0x12345678
file="MesaBoogie.namb" size=1048576 offset=80
arch=x86_64 avx2=true fma=true
os=linux kernel=6.8.0-generic
pw_rate=48000 buffer=256
timestamp=2026-04-10T01:56:18Z
────────────────────────────────────────────────────
```

### Catálogo de Códigos de Erro

| Faixa     | Categoria              | Módulo Investigação                                    |
| --------- | ---------------------- | ------------------------------------------------------ |
| **E1xxx** | Carregamento de modelo | `src/loader/`, `src/main.rs::load_and_send_model()`    |
| **E2xxx** | PipeWire / Áudio       | `src/pw_host.rs`, `src/dsp/`                           |
| **E3xxx** | SPSC / Comunicação     | `src/spsc.rs`, `src/main.rs::cli_loop()`               |
| **E4xxx** | Runtime / CLI          | `src/main.rs::cli_loop()`, `src/main.rs::parse_args()` |
| **E5xxx** | Sistema / Hardware     | `src/main.rs::main()` (detecção AVX2/FMA)              |

## Instructions

### 1. Diagnóstico por Evidências

- **Xruns / clicks**: Verifique se o closure `.process()` ou funções invocadas respeitam o budget de tempo (sem `Vec`, `Box`, `println!` ou travas).
- **Degradação numérica**: Valide se os multiplicadores de ganho e o resampling respeitam os metadados do modelo e as taxas de amostragem do host.
- **Deadlock / contenção**: Inspecione o uso de `rtrb` e garanta que não há primitivas bloqueantes tocando o caminho RT.

### 2. Intervenção Cirúrgica

- Use `std::simd` com `const generics` (SoA) para correções em tensores.
- Mantenha `#[repr(align(128))]` em toda estrutura compartilhada via buffer SPSC.
- Após sanado, remova **toda** linha de log, `dbg!()` ou `eprintln!` inserida para depuração visual dentro do callback RT.

### 3. Emissão de Diagnósticos

- Se o fix introduzir novos pontos de falha fora da thread RT, use o sistema `NamDiagnostic`.
- Mensagens informativas (logging, não-erros) continuam com `println!("[CLI] ...")` ou `println!("[NAM-rs] ...")`.
