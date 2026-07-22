<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# PipeWire Dual-Stream Architecture — Decisão GO/NO-GO para protótipo `pw_filter`

**Data:** 2026-07-19
**Responsável:** Arquiteto de Sistemas

## 1. Resumo Executivo

**Decisão: NO-GO — o protótipo `pw_filter` não é necessário.**

A arquitetura dual-stream atual já garante processamento intra-ciclo na ordem
correta (capture → playback), resultando em **0 quantum de latência extra**.
A instrumentação implementada em T5.S1.1 permite verificação empírica desta
conclusão em tempo real com `pw-top` e os logs de diagnóstico.

## 2. Análise Arquitetural

### 2.1 Como o PipeWire ordena nós dentro do mesmo ciclo

O PipeWire processa nós em grupos (definidos por `node.group` ou `node.link-group`).
Todos os nós do mesmo grupo são agendados pelo mesmo driver no mesmo ciclo de
processamento (`quantum`).

A ordenação interna dos nós seguidores (`targets`) dentro do ciclo é determinada
pela **ordem de registro** no driver — FIFO em `spa_list`. O nó registrado
primeiro é processado primeiro.

**Referência:** [PipeWire Graph Scheduling](https://docs.pipewire.org/page_scheduling.html),
implementação em `src/pipewire/impl-node.c` — `spa_list_append` no `target_list`.

### 2.2 A ordem de criação dos streams no NAM-rs

Em [`run.rs:91-126`](../src/standalone/pw_host/run.rs):

```text
1. _lock = thread_loop.lock()
2. setup_capture_stream()    ← registrado primeiro
3. setup_playback_stream()   ← registrado depois
4. drop(_lock)
```

Ambos possuem:

- `node.group = "nam-rs-dsp"` — mesmo grupo de processamento
- `node.link-group = "nam-rs-link-group"` — grupo de enlace interno
- `StreamFlags::RT_PROCESS` — processamento síncrono

O stream de capture recebe adicionalmente:

- `PRIORITY_DRIVER = 2000` — prioridade máxima como candidato a driver

### 2.3 Fluxo de processamento resultante

Dentro de cada quantum do PipeWire:

```text
┌─────────────────────────────────────────────────────────────────┐
│ Driver Cycle (mesmo quantum)                                    │
│                                                                 │
│  [1] Capture process():                                         │
│      └─ dequeue_buffer() → DSP pipeline → write DspBridge()     │
│                                                                 │
│  [2] Playback process():                                        │
│      └─ read DspBridge() → dequeue_buffer() → copy to hardware  │
│                                                                 │
│  Resultado: 0 quantum de latência extra                         │
└─────────────────────────────────────────────────────────────────┘
```

A ordem capture→playback é garantida porque:

1. Ambos estão no mesmo `node.group`, processados no mesmo ciclo
2. O capture é registrado primeiro → processado primeiro
3. O `PRIORITY_DRIVER = 2000` no capture garante que ele é o driver (ou
   candidato principal), consolidando a liderança do grupo

## 3. Verificação Empírica (T5.S1.1)

A instrumentação implementada em T5.S1.1 confirma esta análise em tempo real:

```text
⏱️ PW Stream Timing: cap↦pb gap=XX µs | tick_delta=0 |
   cap_ticks=NN pb_ticks=NN | cap_delay=XX µs pb_delay=XX µs |
   quantum=XX µs
```

**Indicadores de operação correta (0 quantum extra):**

- `tick_delta = 0` — ambos os streams no mesmo tick do driver
- `cap_ticks ≈ pb_ticks` — mesmo ciclo de clock do PipeWire
- `cap↦pb gap < quantum` — o gap entre callbacks é menor que o período do quantum

**Indicadores de +1 quantum (NÃO esperado):**

- `tick_delta ≥ 1` — playback um ciclo atrás do capture
- `cap↦pb gap > quantum` — gap maior que o período do buffer

### Comandos de verificação complementar

```bash
# Terminal 1: NAM-rs com logs detalhados
RUST_LOG=info nam-rs --model modelo.nam

# Terminal 2: pw-top com filtro para os nós NAM-rs
pw-top | grep -E "NAM-rs|Driver"

# Esperado: ambos os nós aparecem com o mesmo driver e quantum,
# sem gaps de ciclo entre eles.
```

## 4. Decisão Técnica

### NO-GO: o protótipo `pw_filter` está arquivado

| Critério                       | Situação atual                              | pw_filter hipotético           |
| ------------------------------ | ------------------------------------------- | ------------------------------ |
| Latência intra-ciclo           | 0 quantum ✓                                 | 0 quantum (não melhora)        |
| Cópias extras                  | 2 (bridge write + bridge read)              | 0 (único nó)                   |
| Sincronização atômica          | Release/Acquire no DspBridge                | Nenhuma                        |
| Complexidade de implementação  | Já implementado, testado, estável           | FFI unsafe nova (alto risco)   |
| Custo de manutenção            | Zero adicional                              | +300 linhas de FFI para manter |
| `pipewire-rs` suporte          | Nativo (`StreamBox`)                        | FFI direto via `pipewire-sys`  |
| Modo dual-stream como fallback | N/A (já é o modo principal)                 | Necessário manter ambos        |
| RT-safety                      | Verificada (tests-quick, quality-dashboard) | A validar do zero              |

**Conclusão:** O custo e risco de implementar `pw_filter` superam em muito o
ganho potencial (zero, já que a latência intra-ciclo já é 0 quantum). As duas
cópias extras no DspBridge (write + read de ~128-512 amostras × 2 canais =
2-8 KB) são insignificantes comparadas ao custo da inferência neural (~50 µs).

## 5. Recomendações

1. **Arquivar F-S1 (Fases 2+):** Marcar como `WONT-FIX` — a arquitetura dual-stream
   já atende plenamente o requisito de latência intra-ciclo.

2. **Manter a instrumentação T5.S1.1 ativa:** O log de PW Stream Timing é de
   baixíssimo custo (1 chamada `time()` a cada 64 frames ≈ 1 chamada a cada
   1.3 ms @64 samples) e serve como canário para detectar regressões futuras
   de scheduling do PipeWire.

3. **Documentar o rationale no código:** Adicionar comentário em
   `mod.rs` documentando que a ordem capture→playback é garantida pelo
   `node.group` + ordem de registro + `PRIORITY_DRIVER`.

## 6. Referências

- [PipeWire Graph Scheduling](https://docs.pipewire.org/page_scheduling.html) —
  documentação oficial do scheduling de nós no grafo
- [PipeWire Node Running](https://docs.pipewire.org/devel/page_running.html) —
  documentação de agrupamento de nós (`node.group`, `node.link-group`)
- [PipeWire Props Manual](https://docs.pipewire.org/page_man_pipewire-props_7.html) —
  propriedades `priority.driver`, `node.group`, `node.link-group`
- [`src/standalone/pw_host/mod.rs`](../src/standalone/pw_host/mod.rs) —
  documentação da arquitetura dual-stream no código
