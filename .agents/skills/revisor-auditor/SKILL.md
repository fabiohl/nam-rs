---
name: revisor-auditor
description: Painel de auditores, caçadores de bugs, cientistas, engenheiros sêniors e especialistas em diversas disciplinas associadas ao projeto. Conjuntamente realizam varredura sistemática e cirúrgica do repositório, caçando bugs latentes, falhas de concorrência, riscos de runtime e desvios arquiteturais - antes que eles se tornem incidentes de produção.
---

# Skill: Revisor Auditor

## When to use this skill

Use para inspecionar, revisar e diagnosticar estrita aderência arquitetural, focando nos princípios cruciais de inferência via FMA (Fused Multiply-Add), PipeWire Assíncrono e detecção cirúrgica de violações computacionais e lock-free no buffer macro da thread principal (DSP) do projeto Standalone NAM-rs.

## Instructions

### 1. Ingestão de Referenciais

Revise seu contexto mental com base nos seguintes documentos:

- **Regras de código**: `.agents/rules/rust.md` (condições inegociáveis de RT-safety).
- **Arquitetura atual**: `docs/architecture.md` (fonte primária de verdade, Sprint 8).
- **Roadmap**: `docs/NAM-rs-referencia.md` e `docs/NAM-rs-sprints.md` (contexto histórico).

### 2. Subversão Arquitetural Híbrida: O Que Erradicar

Inspecione linha-de-código detectando categoricamente:

#### 🔴 CRÍTICO — Violações Severas no Caminho RT

| Categoria            | O que verificar                                                                                                                                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Heap no callback** | `Vec::new()`, `Box::new()`, `String`, ou qualquer alocação dentro do closure `.process()` do PipeWire. Inclui `NamResampler::new()` — proibido no RT; deve ser construído pela thread principal e enviado via `resampler_consumer.pop()`. |
| **I/O no callback**  | `println!`, `eprintln!`, `write!`, `fs::read`, qualquer syscall bloqueante dentro de `process()`. Status deve ser comunicado via `RtStatusFlags` atômicas.                                                                                |
| **Locks no RT**      | `Mutex::lock()`, `RwLock::write()`, `Condvar::wait()` tocando qualquer código invocado pelo callback DSP.                                                                                                                                 |
| **Branch Predictor** | Redes LSTM/WaveNet com loops iterativos sem `const generics` + unrolling. Estruturas SoA são obrigatórias.                                                                                                                                |
| **False Sharing**    | Structs SPSC sem `#[repr(align(128))]`. Verificar `ParamPayload` e qualquer nova estrutura compartilhada entre threads.                                                                                                                   |
| **io_uring / Disco** | Qualquer referência a `io_uring`, gravação de arquivos, `AudioRip` ou captura de disco herdada. Catégoricamente proibido.                                                                                                                 |

#### 🟠 ALTO — Condições Inadequadas

| Categoria                     | O que verificar                                                                                                                                                                                |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Resampler sem bypass**      | `NamResampler` com `pw_rate == 48000` deve operar em bypass (sem FIR real). Verificar `NamResampler::is_bypass()`.                                                                             |
| **Ajuste de ganho incorreto** | O cálculo `input_db_adj = 12.0 - input_level_dbu` e `output_db_adj = -18.0 - loudness` deve refletir os metadados `.namb`. Erros causam clipping ou submodulação.                              |
| **Digital aliasing**          | Resampling FIR não-linear ou sem filtragem de fase (verificar que `rubato::SincFixedIn<f32>` é usado com Kaiser Window e não foi substituído por algo inferior).                               |
| **Drop no callback**          | O `Drop` do `NamResampler` antigo no `while let Ok(new_rs) = resampler_consumer.pop()` é aceitável (~50ns). Modelos obsoletos (`DynamicModel`) **nunca** são dropped no RT — vão via canal GC. |

#### 🟡 MÉDIO — Qualidade e Manutenção

| Categoria                           | O que verificar                                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Unwraps no RT**                   | `unwrap()` sem fallback silencioso no caminho do callback. Use `.unwrap_or_else()` ou `if let`. |
| **CRC32 sem verificação**           | Modelos `.namb` carregados sem validação de checksum (`crc32fast`).                             |
| **Docs desatualizadas**             | `docs/architecture.md` contradizendo a implementação real (especialmente após Sprint 8).        |
| **Testes sem assertívas numéricas** | Testes de inferência sem verificação de fidelidade (valores esperados com tolerância binária).  |

### 3. Retificações Bit-Perfect

Submeta patches cirurgicamente, sem alterar padrões consolidados (canais SPSC, `RtStatusFlags`, estrutura de módulos). A operação final exige validação incondicional pelo ciclo local `utils/lints.sh`. Tudo deve operar nas janelas perfeccionistas de baixa latência.
