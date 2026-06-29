<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Execução

Este arquivo define os sprints e as tarefas técnicas para a execução dos épicos de melhoria do NAM-rs, com base nas descobertas consolidadas em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

---

## Épico α — Controles de usuário de baixo risco (quick wins)

**Objetivo:** Expor controles de runtime que já estão implementados ou necessitam de ajustes mínimos de usabilidade, com foco na segurança e paridade entre o plugin CLAP e o executável Standalone.
**Risco:** BAIXO. Não altera matemática de inferência nem as topologias neurais.
**Origem dos achados:** I1 e I2 do [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

### Dependências e Sequência de Execução

Para otimizar e garantir a paridade, o trabalho será dividido em dois sprints focados:

1. **Sprint α1: Oversampling Standalone (Foco em I2)**
2. **Sprint α2: Controle de Precisão de Ativação (Foco em I1)**

---

### Sprint α1 — Paridade de Oversampling no Standalone (I2)

Este sprint resolve a lacuna de runtime oversampling no executável standalone, permitindo que a CLI honre a flag de inicialização e possibilite a troca de fator em tempo de execução de maneira RT-safe (zero-alloc no thread de áudio).

#### [ ] Tarefa α1.1 — Correção da Inicialização via CLI [BAIXO RISCO]

- **Descrição:** Corrigir o bug de inicialização onde o fator `--oversample` parseado da linha de comando é descartado em `run.rs` e substituído por `OversampleFactor::Off` fixo em `capture/setup.rs:66`.
- **Mudanças propostas:**
  - Em [run.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/run.rs), alterar a desestruturação de `config` (remover o descarte de `oversample: _os`) e passar a flag real para o init do capture stream.
  - Em [setup.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/capture/setup.rs), usar o valor de oversample da configuração para inicializar o `CaptureState`.
- **Validação:** Rodar standalone especificando `--oversample 2x` ou `--os 4x` e validar que as engines de oversampling são criadas com o fator correto no log de inicialização.

#### [ ] Tarefa α1.2 — SPSC Rebuild Pipeline para Oversampling no Standalone [MÉDIO RISCO]

- **Descrição:** Implementar a troca de oversampling em tempo de execução no standalone usando o padrão de reconstrução assíncrona off-RT.
- **Mudanças propostas:**
  - Criar um canal SPSC (`rtrb` Ring Buffer) para passar novas instâncias de `Box<OversampleEngine>` construídas na thread principal para a thread de áudio (similar ao `slimmable_consumer`).
  - No thread de áudio, em [commands.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/rt_callback/commands.rs) (no payload `ParamPayload::SetOversample(factor)`):
    - Obter o fator numérico correspondente, armazenar em `rt_status.requested_os_factor` e sinalizar o flag `RT_STATUS_NEEDS_OS_REBUILD`.
  - Na thread principal ([run.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/run.rs) loop), escutar o flag `RT_STATUS_NEEDS_OS_REBUILD`:
    - Construir instâncias de `OversampleEngine` L e R com o fator solicitado (usando `OversampleEngine::new(factor, MAX_RESAMP_BUF)`).
    - Enviar as engines através do canal SPSC.
    - Limpar o flag `RT_STATUS_NEEDS_OS_REBUILD`.
  - No thread de áudio ([setup.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/capture/setup.rs) loop), consumir o canal SPSC, realizar o swap em `CaptureState` e descartar as engines antigas enviando-as ao Garbage Collector (padrão `drain_gc_channels`).
- **Validação:**
  - Garantir que a alternância do oversampling em runtime não causa xruns nem alocações de heap na thread de áudio (auditado com `tests/nam_infer_test.rs` zero-alloc guards).
  - Executar os testes unitários e de integração de fidelidade espectral.

---

### Sprint α2 — Exposição do Controle de Precisão de Ativação (I1)

Este sprint expõe a infraestrutura de `ActivationPrecision::HighFidelity` (que hoje existe no código, mas só é chamada em testes) para o usuário final por meio da linha de comando do standalone (CLI) e de parâmetros adicionais no plugin CLAP.

#### [ ] Tarefa α2.1 — Exposição na CLI (Standalone) [BAIXO RISCO]

- **Descrição:** Adicionar as opções de CLI `--activation <standard|hf>` (e o atalho `--act`) ao parser de linha de comando do standalone para configurar a precisão global no bootstrap.
- **Mudanças propostas:**
  - Modificar [cli.rs](file:///home/fabio/nam-rs/src/standalone/cli.rs) para incluir a nova flag e parsear como o respectivo enum `ActivationPrecision`.
  - No bootstrap em [main.rs](file:///home/fabio/nam-rs/src/main.rs), chamar `set_activation_precision(...)` antes da inicialização do PipeWire para aplicar o modo selecionado.
- **Validação:** Executar o standalone com `--activation hf` e confirmar que o modo de alta fidelidade é ativado.

#### [ ] Tarefa α2.2 — Adição de Parâmetro e GUI no CLAP (Plugin) [BAIXO RISCO]

- **Descrição:** Expor o controle de precisão de ativação no plugin CLAP sob o identificador `PARAM_ACTIVATION = 8`.
- **Mudanças propostas:**
  - Declarar `PARAM_ACTIVATION = 8` em `src/clap/extensions/params/mod.rs` ou `main.rs`.
  - Mapear o valor do parâmetro (0 -> "Standard", 1 -> "HighFidelity").
  - Incluir `param_activation: AtomicU32` em `UiToRt` (`src/clap/plugin/shared.rs`).
  - No thread de áudio (`src/clap/processor/events.rs`), monitorar a mudança de `PARAM_ACTIVATION` e chamar `set_activation_precision(...)` na borda do bloco (no flush de parâmetros), sem reconstruir o modelo completo.
  - Implementar a persistência do parâmetro em `state.rs` e o respectivo widget de controle visual na GUI (`zones/controls.rs`).
- **Validação:**
  - Mudar o parâmetro via GUI e constatar a alteração de comportamento no thread de processamento sem instabilidade de áudio.
  - Testar o comportamento em render offline (Offline render força HighFidelity e desliga adaptativo).

#### [ ] Tarefa α2.3 — Testes de Integração e Medições de Zero Alloc [BAIXO RISCO]

- **Descrição:** Escrever testes de integração e validar as garantias de latência e tempo real.
- **Validação específica:**
  - Adicionar teste em `tests/activation_precision.rs` simulando o fluxo de controle CLI/CLAP.
  - Verificar se a alternância de ativação não dispara o `CountingAllocator` (nenhuma alocação ocorre na troca).
  - Documentar explicitamente em `architecture.md` e `audio_fidelity_map.md` que os modelos LSTM ignoram temporariamente este controle até a entrega do Épico β (I6).
