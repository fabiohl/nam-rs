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

### Sprint 2: Motor FiLM Completo e Integração [DONE]

**Objetivo**: Construir a sub-rede matemática e integrá-la dentro dos 8 pontos definidos na estrutura de blocos do A2.

* **Tarefa 2.1: Lógica Principal do FiLM** [DONE]
  * **Arquivo alvo**: `src/models/a2/film.rs`.
  * **Ação**: Implementar lógica no stub existente (`FiLMLayer`). Processamento recebe entrada do canal principal e do vetor condicional, convertendo com Conv1x1 (utilizando Tarefa 1.2) em `scale` e opcionalmente `shift`. Em seguida, aplica transformações batch-wise via AVX2.
  * **Critério de aceite**: Buffers pré-alocados no `load()`, uso intensivo de chunks_exact e ausência de branches e alocações (`unwrap`, `Vec::new`) no bloco DSP iterativo.

* **Tarefa 2.2: Instanciação nos Pontos Insercionais A2** ✅ [DONE]
  * **Arquivo alvo**: `src/models/a2/layer.rs` (ou equivalente que instancie os layers).
  * **Ação**: Ler o JSON para habilitar as instâncias nos 8 pontos: `conv_pre_film`, `conv_post_film`, `input_mixin_pre_film`, `input_mixin_post_film`, `activation_pre_film`, `activation_post_film`, `layer1x1_post_film` e `head1x1_post_film`.
  * **Critério de aceite**: Executar condicionalmente as rotinas apenas quando a respectiva config contiver `active: true`. Zero-alloc mantido.
  * **Nota pós-implementação**:
    * `A2Layer` agora possui 8 campos `Option<FiLMLayer>`, inicializados como `None` e populados por `set_weights` quando o JSON `layer_raw` contém entradas FiLM com `active: true`.
    * `WaveNetA2` ganhou campo `layer_raw: Option<serde_json::Value>` para que `set_weights` possa parsear as configs FiLM e carregar os pesos (weights + bias) do stream.
    * `FilmBlock<'a>` (em `film.rs`) agrupa referências mutáveis para os 8 pontos, passado como `&mut FilmBlock` para `layer_forward_ch3_block` e `layer_forward_ch8_block`.
    * `conv_pre_film` é aplicado no nível do modelo (antes da cópia para o buffer de histórico). Todos os outros pontos são aplicados dentro das funções de bloco.
    * `check_film_all_inactive()` em `is_a2_shape()` foi relaxado: modelos A2 com FiLM ativo não são mais rejeitados (a carga de pesos FiLM ocorre em `set_weights`).
    * O `FilmBlock::empty()` cria um bloco vazio para o fast-path (sem FiLM), garantindo zero custo de branches (todas `if let Some` são cold e nunca tomadas).
    * **Impacto em tarefas futuras**: A infraestrutura de parsing JSON (`layer_raw`) e o `FilmBlock` serão reutilizados pela Tarefa 3.1 (`condition_dsp`) e pelo motor A2 dinâmico. Modelos com `cond_size > 1` ainda fluem para o engine dinâmico; o suporte a `cond_size > 1` no fast-path A2 requer adaptação adicional (não coberta por esta tarefa).

### Sprint 3: `condition_dsp` e Cobertura Golden

**Objetivo**: Realizar a cadeia de ponta a ponta e garantir conformidade audível de 100% com os testes e modelos Golden C++.

* **Tarefa 3.1: Parsing e DSP Condicional (`condition_dsp`)** ✅ [DONE]
  * **Arquivo alvo**: `src/loader/nam_json/model.rs` e rotinas de DSP no `WaveNetModelDyn`.
  * **Ação**: Identificar o modelo subjacente (`WaveNet`, `LSTM`, `Linear`) aninhado em `condition_dsp`, alocá-lo integralmente durante o estágio de setup (sem comprometer os buffers). Integrar seu processamento passo a passo antes de despachar a saída para as instâncias FiLM.
  * **Critério de aceite**: DSP secundário operando fluidamente dentro do ciclo de processamento do master DSP de áudio.
  * **Nota pós-implementação**:
    * `NamConfig` agora possui campo `condition_dsp: Option<serde_json::Value>` para o sub-modelo aninhado.
    * `StaticModel::num_output_channels()` retorna os canais de saída do modelo (equivalente a `DSP::NumOutputChannels()` do C++).
    * `FreeWavenetGeometry` agora armazena `channels: Vec<usize>` e `head_sizes: Vec<usize>` por array, capturando geometrias onde a última array tem `head_size` diferente de 1 (necessário para condition_dsp WaveNet cujo último array tem `head_size=3`).
    * O sub-modelo `condition_dsp` é construído recursivamente via `build_model()` durante `build_wavenet_dynamic()`, com validação de sample_rate e `condition_size == num_output_channels()`.
    * `WaveNetModelDyn` ganhou campo `condition_dsp: Option<Box<StaticModel>>` e buffer `condition_dsp_output: AlignedVec<f32>` (tamanho `cond × WAVENET_MAX_NUM_FRAMES`).
    * `process_internal`: quando `condition_dsp` está presente, o áudio mono de entrada é processado pelo sub-modelo e seu output multi-canal é usado como `condition` para as arrays (espelhando `_process_condition` do C++). Sem condition_dsp, mantém comportamento passthrough (cond≤1).
    * `prewarm_internal`: propaga `prewarm(0)` para o sub-modelo antes de preaquecer as arrays principais.
    * `set_max_buffer_size` e `prewarm_samples` propagam para o sub-modelo.
    * Teste `test_loader_gap_wavenet_condition_dsp` agora carrega o modelo completo com condition_dsp funcional (519 lib tests, 19 golden, todos passando).

* **Tarefa 3.2: Golden Tests e Paridade ESR/SNR** ✅ [DONE]
  * **Arquivos alvo**: `tests/cpp_parity.rs` e suítes (`tests-long.sh`).
  * **Ação**: Converter `test_loader_gap_*` associado a essas flags em um teste golden positivo. Carregar fisicamente os artefatos `wavenet_a2_max.nam` e `wavenet_condition_dsp.nam`.
  * **Critério de aceite**: Output de áudio precisa equiparar o `NAMCore v0.5.3` sem degradação do SNR, mantendo processamento sob 2ms em low-latency buffer sizes.
  * **Nota pós-implementação**:
    * `test_loader_gap_wavenet_condition_dsp` foi convertido em `test_golden_vectors_wavenet_condition_dsp`: golden positivo com C++ cross-reference via ESR/SNR/MSE. SNR medido: 139.5 dB (quase bit-exact), ESR: 1.13e-14. Golden v1 (2048 samples) + v2 (240k samples @ 48 kHz) gerados e commitados.
    * Live cpp_parity adicionado: `live_cross_validation_wavenet_condition_dsp` (v1) + `live_cross_validation_v2_wavenet_condition_dsp` (v2 multi-SR).
    * **wavenet_a2_max.nam: BLOQUEADO — requer engine dinâmico A2 completo.** O modelo possui FiLM (8 pontos ativos), gating, ativação Softsign, bottleneck, condition_dsp com sub-modelo A2 (SiLU, PReLU, gating, FiLM). O detector secundário A2 (`is_wavenet_a2()`) o classifica como A2 pela ativação ≠ Tanh e rejeita na dispatch. Mesmo que o roteamento fosse relaxado, o engine dinâmico A1 carece de suporte a FiLM, gating, arrays únicas e ativações heterogêneas. Um engine `WaveNetModelDynA2` é necessário — ver `TODO-features.md` para tracking.
    * `test_loader_gap_wavenet_a2_max` atualizado para documentar o bloqueio (erro esperado: "A2 model detected but architecture shape not recognized").
    * Calibração: `wavenet_condition_dsp` registrado em `get_calibrated_threshold()` com SNR floor 100 dB, ESR 1e-10.
    * `golden_gen_build.sh` atualizado para incluir `wavenet_condition_dsp.nam` nos modelos v1 e v2.

---
