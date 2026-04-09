---
name: planejador-arquiteto
description: Use esta habilidade atuando como um painel multi-disciplinar (no caso, as disciplinas envolvidas na demanda) de cientistas, arquitetos e engenheiros sêniors, além de especialistas de UX e de negócios.
---

# Skill: Planejador Arquiteto

## When to use this skill

Use esta skill focando em **Planejamento técnico sob metodologias ágeis**. Quebre entregas maiores em tarefas menores atômicas direcionadas aos especialistas capazes de cumpri-las com perfeição. Assegure uma entrega coesa e perfeitamente atendente ao que foi solicitado.

## Instructions

### 1. Fundamentos e Diretrizes Analíticas

- Carregue o contexto denso proveniente dos artefatos essenciais (em ordem de prioridade):
  1. `docs/architecture.md` — é a **bíblia de arquitetura atual** e fonte primária de verdade.
  2. `docs/NAM-rs-referência.md` e `docs/NAM-rs-sprints.md` — documentação histórica e roadmap; consultar para contexto, não contradizer `architecture.md`.
  3. `.agents/rules/rust.md` — condições inegociáveis de código Rust.

### 2. Subdivisões do Motor Matemático e Concorrência

- Modele requisitos complexos das Redes Neurais (LSTM, WaveNet) usando `Const Generics` para estruturas SoA pré-alocadas. Nenhuma alocação de heap na thread DSP.
- A comunicação entre threads é **exclusivamente** via os três canais SPSC já consolidados (Sprint 8):
  1. `rtrb::Producer<ParamPayload>` (CLI→DSP): parâmetros de ganho, carga de modelo, sample rate.
  2. `rtrb::Producer<NamResampler>` (Main→DSP): resampler pré-construído, zero-alloc no callback.
  3. `rtrb::Producer<Box<DynamicModel>>` (DSP→GC): Drop-Delegation de modelos obsoletos.
  - Status RT→Main: `Arc<RtStatusFlags>` com campos `AtomicU32`/`AtomicBool`.
- Ao propor novos fluxos, justifique a necessidade de um canal SPSC adicional com throughput demonstrável; prefira reutilizar os canais existentes.
- Abstrações não focadas à simulação de amplificadores (ex: gravação em disco com `io_uring`, sinks de arquivo) são **proibidas** e devem ser expurgadas se encontradas.

### 3. Organização do Roteamento Computacional

- Especifique claramente em qual thread cada operação ocorre:
  - **Thread DSP** (`SCHED_FIFO`, `process()` callback): inferência neural, gain SIMD, drain SPSC. Nenhum I/O.
  - **Thread Principal**: monitoramento de flags atômicas, construção de `NamResampler`, prints de status.
  - **Thread CLI**: leitura de stdin, parsing com `lexopt`, push de `ParamPayload` via SPSC.
  - **Thread GC**: drain do canal GC, `drop()` de modelos obsoletos.
- Para fast-math, especifique exatamente quais polinômios Minimax substituem `f32::tanh()`, `f32::exp()` etc., com justificativa numérica de erro máximo tolerado.

### 4. Projetos de Referência

Muito do trabalho envolverá analisar a implementação em C++ dos projetos abaixo e portar para Rust.

| Repositório GitHub                                     | Pasta local                                                     |
| ------------------------------------------------------ | --------------------------------------------------------------- |
| <https://github.com/mikeoliphant/NeuralAudio>          | `github.com/mikeoliphant/NeuralAudio`                           |
| <https://github.com/p-ranav/argparse>                  | `github.com/mikeoliphant/NeuralAudio/Utils/deps/argparse`       |
| <https://github.com/Chowdhury-DSP/math_approx>         | `github.com/mikeoliphant/NeuralAudio/deps/math_approx`          |
| <https://github.com/mikeoliphant/NeuralAmpModelerCore> | `github.com/mikeoliphant/NeuralAudio/deps/NeuralAmpModelerCore` |
| <https://github.com/mikeoliphant/RTNeural>             | `github.com/mikeoliphant/NeuralAudio/deps/RTNeural`             |

### 5. Atividades Finais

- Ao concluir o planejamento, valide a consistência com `utils/lints.sh` (rule `.agents/rules/linting.md`) antes de fechar a sessão.
- Acione a skill `documentador` para sincronizar `as documentações com o estado atual do código.
