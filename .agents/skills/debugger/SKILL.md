---
name: debugger
description: Acionado quando algo não funciona como esperado. Atua como um painel multi-disciplinar de analistas e engenheiros seniores capazes de resolver os problemas mais difíceis.
---

# Skill: Debugger

## When to use this skill

Use sob erro ativo, crash, comportamento não-linear de DSP, áudio truncado, xruns do PipeWire ou degradação nos cálculos de inferência neural em tempo real.

Também é acionada pelo workflow `diagnostico` quando o usuário cola um bloco de suporte.

## Instructions

### Referências Primárias

* Códigos de erro `Exxxx` e formato do bloco de suporte: `src/diagnostics.rs` (`NamErrorCode`, `emit_diagnostic()`).
* Diretrizes RT-safe e DSP: `.agents/rules/rust.md`.
* Arquitetura geral: `docs/architecture.md`.

### Diagnóstico por Evidências

* **Xruns / clicks**: Verifique se `process()` respeita o budget de tempo (sem `Vec`, `Box`, `println!` ou locks).
* **Degradação numérica**: Valide multiplicadores de ganho e metadados de resampling vs taxas de amostragem do host.
* **Deadlock / contenção**: Inspecione uso de `rtrb` — sem primitivas bloqueantes no caminho RT.

### Intervenção Cirúrgica

* Mantenha `#[repr(align(128))]` em estruturas compartilhadas via SPSC.
* Após fix, remova **toda** linha de log, `dbg!()` ou `eprintln!` inserida para depuração no callback RT.
* Se o fix introduzir novos pontos de falha fora do RT, use o sistema `NamDiagnostic` para emitir erros estruturados.
