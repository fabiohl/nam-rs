<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO Sprints: NAM-rs — Mega Tópicos 3 e 4

Este documento organiza a execução das demandas levantadas pela auditoria (`revisor-auditor`) em ciclos granulares de desenvolvimento, garantindo conformidade com arquitetura, performance e RT-Safety.

## Épico 1: MT3 — WaveNet A1 Dinâmico: Generalização para N Arrays (F1-ext)

**Objetivo:** Remover o limite fixo de 2 arrays no motor `WaveNetModelDyn`, permitindo topologias de tamanho arbitrário. Isso solucionará a falha atual do teste `live_cross_validation_nondist_models`.

### Sprint 1.1: Refatoração Estrutural do Modelo

* **Task 1.1.1 [x] Modificar `WaveNetModelDyn`:**
  * Arquivo: `src/models/wavenet/model_dyn.rs`
  * Substituir os campos `array1` e `array2` por um vetor `pub arrays: Vec<WaveNetLayerArrayDyn>`.
* **Task 1.1.2 [x] Adaptar Iteração do `process_internal`:**
  * Arquivo: `src/models/wavenet/model_dyn.rs`
  * Substituir as invocações fixas ao `array1` e `array2` por um laço seguro e "borrow-checker friendly".
  * Lógica de conexão: O `output` do array N passa a ser o `input` do array N+1.
  * O `head_outputs` só precisa ser extraído do *último array* no laço.
* **Task 1.1.3 [x] Adaptar Iteração do `prewarm_internal`:**
  * Arquivo: `src/models/wavenet/model_dyn.rs`
  * Reproduzir a mesma lógica iterativa da Task 1.1.2 para a fase de aquecimento do modelo (`zero_input`).

### Sprint 1.2: Loader Dinâmico e Validação

* **Task 1.2.1 [x] Modificar o construtor `build_wavenet_dynamic_inner`:**
  * Arquivo: `src/loader/dispatcher/wavenet/dynamic.rs`
  * Remover a restrição restritiva: `if geom.num_arrays != 2 { bail!(...) }`.
  * Instanciar `WaveNetLayerArrayDyn` dentro de um `for i in 0..geom.num_arrays`.
  * Extrair os canais (CH), tamanho do kernel (K), e bias (`has_head_bias = is_last`).
  * Atualizar as alocações de buffer dinamicamente mantendo RT-safety (pré-alocação estrita no momento do load).
  * **Nota Sprint 1.1:** O laço de encadeamento em `process_internal`/`prewarm_internal` usa `self.ch`/`self.head` (dimensões do array 0) para fatiar `head_outputs` e `array_outputs` dos arrays intermediários. Para N>2, verificar se usar `prev.ch`/`prev.head` (dimensões per-array) é mais robusto, especialmente se arrays intermediários tiverem CH diferente do array 0.
* **Task 1.2.2 [x] Testes e Homologação Funcional:**
  * Arquivo: `tests/cpp_parity.rs`
  * Rodar a suite longa: `./utils/tests-long.sh` e certificar que `live_cross_validation_nondist_models` passa com sucesso.
  * Verificar a integridade sonora de outros modelos via Golden Vectors.

---

## Épico 2: MT4 — Motor LSTM Arbitrário (F7)

**Objetivo:** Implementar um "fallback dinâmico" para o motor LSTM capaz de lidar com constelações dimensionais desconhecidas (`num_layers`, `hidden_size`) além dos 10 perfis otimizados por `const generics`.

### Sprint 2.1: Infraestrutura de Camada Dinâmica

* **Task 2.1.1 [ ] Criar estrutura `LstmLayerDyn`:**
  * Criar arquivo: `src/models/lstm/layer_dyn.rs`
  * Declarar `LstmLayerDyn` usando `AlignedVec` em substituição aos arrays constantes (`Aligned64<[T; N]>`).
  * Campos: `input_size`, `hidden_size`, `input_hidden_weights`, `bias`, `state`, `state_bf16`, `cell_state`, `gates`.
* **Task 2.1.2 [ ] Implementar kernels de processamento (`LstmLayerDyn`):**
  * Criar versões que leem as dimensões `hidden_size` diretamente em tempo de execução.
  * Reutilizar as macros e funções estáticas localizadas em `crate::math::gemm` (por exemplo, `gemv_4gate_avx2`), que inferem o tamanho pela *length* do `slice`.
  * Implementar as versões escalares e as versões SIMD otimizadas (`avx2`, `avx512`).

### Sprint 2.2: Construção do Modelo e Dispatcher Híbrido

* **Task 2.2.1 [ ] Construir o Modelo Dinâmico `LstmModelDyn`:**
  * Criar arquivo: `src/models/lstm/model_dyn.rs`
  * Compor uma struct com `layers: Vec<LstmLayerDyn>`.
  * Implementar o método genérico `process` que faz a cadeia completa de repasse entre as camadas e soma com `head_weights`.
* **Task 2.2.2 [ ] Atualizar Parsing de Pesos:**
  * Criar ou adaptar módulo em `src/loader/dispatcher/lstm/weights.rs` para ler os tamanhos dinâmicos alocando buffers `AlignedVec`.
  * Cuidar com o layout dos pesos e *gate_major*.
* **Task 2.2.3 [ ] Alterar o Dispatch Híbrido:**
  * Arquivo: `src/loader/dispatcher/lstm/dispatch.rs`
  * Em vez de retornar um `bail!` ao encontrar topologia diferente das 10 estáticas, executar o `build_lstm_dynamic` e retornar a nova variação do `StaticModel::LstmDyn`.
  * *Ponto de Risco*: Inserir a nova variação na `enum StaticModel` (`src/models/mod.rs`) afeta o trait. Implementar todos os selos (Sealed trait) corretamente.
* **Task 2.2.4 [ ] Validação Exaustiva:**
  * Criar testes isolados para topologias pequenas/estranhas (ex: 3 layers, 10 hidden units) e avaliar parity com C++ se houver golden vector, ou no mínimo atestar panic-free execution.
