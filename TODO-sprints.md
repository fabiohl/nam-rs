<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO e Planejamento de Sprints — nam-rs

> Planejamento das demandas arquiteturais e de código guiadas pelas metodologias ágeis e de qualidade do nam-rs.

---

## Épico: MT1 — Infraestrutura de Conditioning e FiLM (F2 + F8 parcial + F9 parcial)

**Status**: 🔴 Pendente | **Prioridade**: Crítica | **Objetivo**: Destravar modelos A2 oficiais que utilizam `condition_size > 1` e `condition_dsp`.

Este épico introduzirá o suporte para Feature-wise Linear Modulation (FiLM) e `condition_dsp` no `nam-rs`, mantendo total conformidade RT-Safety (zero alocações no hot-path) e priorizando a paridade com a implementação de referência C++.
O NAMCore v0.5.3 (espelhado na pasta `tests/fixtures/NeuralAmpModelerCore/`) é a implementação de referência.
Esteja atento às orientações em `TODO-features.md`.
O baseline exigido para otimizações será x86-64-v3 (AVX2+FMA).

### Sprint 1: Infraestrutura Base (Grouped Conv e Condition Size) [DONE]

**Objetivo**: Preparar as peças fundamentais no loader e na camada matemática para suportar as features subsequentes.

* **Tarefa 1.1: Generalizar `condition_size` no Loader e Topologia** ✅ [DONE]
  * **Arquivo alvo**: `src/loader/nam_json/topology.rs` e correlatos.
  * **Ação**: Remover a restrição `condition_size != Some(1)` nas funções de validação. Propagar `condition_size` para o ambiente dinâmico do modelo em vez de fixá-lo como const-generic restrito.
  * **Critério de aceite**: Modelos com `condition_size > 1` devem passar pela validação de shape, caindo na estrutura dinâmica. Manter o modelo de const-generic intocado como fast-path se `COND=1`.
  * **Nota pós-implementação**:
    * `wavenet_condition_dsp.nam` (condition_size=3) agora carrega via engine dinâmico com `cond=3`. O sub-modelo `condition_dsp` aninhado ainda não é funcional — a ser abordado na Tarefa 3.1.
    * `wavenet_a2_max.nam` (condition_size=8, 1 layer-array A2) passa o gate de condition_size mas é rejeitado pelo dynamic engine A1 (requer 2 arrays). O engine dinâmico A2 será necessário para modelos A2 com `condition_size > 1` (Sprint 2).

* **Tarefa 1.2: Implementação de Convolução Agrupada (`groups > 1`)** [DONE]
  * **Arquivo alvo**: `src/models/a2/conv1d.rs` ou afins.
  * **Ação**: A modulação FiLM depende de `_cond_to_scale_shift` que pode utilizar `groups > 1`. Adicionar uma via de execução parametrizada para convolutions agrupadas (depthwise).
  * **Critério de aceite**:
    * Implementação nativa com SIMD AVX2 (`vfmadd231ps`).
    * Preservar o fast-path extremo para convoluções tradicionais (`groups == 1`).
    * Testes unitários com tensores exatos para validar paridade.

### Sprint 2: Motor FiLM Completo e Integração

**Objetivo**: Construir a sub-rede matemática e integrá-la dentro dos 8 pontos definidos na estrutura de blocos do A2.

* **Tarefa 2.1: Lógica Principal do FiLM** [DONE]
  * **Arquivo alvo**: `src/models/a2/film.rs`.
  * **Ação**: Implementar lógica no stub existente (`FiLMLayer`). Processamento recebe entrada do canal principal e do vetor condicional, convertendo com Conv1x1 (utilizando Tarefa 1.2) em `scale` e opcionalmente `shift`. Em seguida, aplica transformações batch-wise via AVX2.
  * **Critério de aceite**: Buffers pré-alocados no `load()`, uso intensivo de chunks_exact e ausência de branches e alocações (`unwrap`, `Vec::new`) no bloco DSP iterativo.

* **Tarefa 2.2: Instanciação nos Pontos Insercionais A2**
  * **Arquivo alvo**: `src/models/a2/layer.rs` (ou equivalente que instancie os layers).
  * **Ação**: Ler o JSON para habilitar as instâncias nos 8 pontos: `conv_pre_film`, `conv_post_film`, `input_mixin_pre_film`, `input_mixin_post_film`, `activation_pre_film`, `activation_post_film`, `layer1x1_post_film` e `head1x1_post_film`.
  * **Critério de aceite**: Executar condicionalmente as rotinas apenas quando a respectiva config contiver `active: true`. Zero-alloc mantido.

### Sprint 3: `condition_dsp` e Cobertura Golden

**Objetivo**: Realizar a cadeia de ponta a ponta e garantir conformidade audível de 100% com os testes e modelos Golden C++.

* **Tarefa 3.1: Parsing e DSP Condicional (`condition_dsp`)**
  * **Arquivo alvo**: `src/loader/nam_json/model.rs` e rotinas de DSP no `WaveNetModelDyn`.
  * **Ação**: Identificar o modelo subjacente (`WaveNet`, `LSTM`, `Linear`) aninhado em `condition_dsp`, alocá-lo integralmente durante o estágio de setup (sem comprometer os buffers). Integrar seu processamento passo a passo antes de despachar a saída para as instâncias FiLM.
  * **Critério de aceite**: DSP secundário operando fluidamente dentro do ciclo de processamento do master DSP de áudio.

* **Tarefa 3.2: Golden Tests e Paridade ESR/SNR**
  * **Arquivos alvo**: `tests/cpp_parity.rs` e suítes (`tests-long.sh`).
  * **Ação**: Converter `test_loader_gap_*` associado a essas flags em um teste golden positivo. Carregar fisicamente os artefatos `wavenet_a2_max.nam` e `wavenet_condition_dsp.nam`.
  * **Critério de aceite**: Output de áudio precisa equiparar o `NAMCore v0.5.3` sem degradação do SNR, mantendo processamento sob 2ms em low-latency buffer sizes.

---
