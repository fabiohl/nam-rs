<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Plano de Execução — nam-rs (MT5 e MT6)

Este documento traduz os requisitos dos Mega-Tópicos finais de `TODO-features.md` (MT5 e MT6) em sprints detalhadas para execução técnica, conforme as diretrizes da arquitetura do projeto e o nível de excelência `nam-rs`.

## Épico 1: MT5 — Post-stack Head (F6)

O WaveNet A1 permite que um sub-objeto "head" (Conv1D + ativação) processe o sinal após a pilha de blocos, antes da saída. O `nam-rs` atualmente rejeita esse cenário na topologia e não sabe parsear o objeto.

### Sprint 1.1: Correção do Parser JSON e Topologia

- **Tarefa 1.1.1** [DONE]: Corrigir `src/loader/nam_json/model.rs`. O campo `head` em `NamConfig` está tipado incorretamente como `Option<Option<String>>`. Deve ser alterado para `Option<serde_json::Value>` (ou criar `HeadConfig`) para extrair os campos `{ channels, bias, out_channels, activation, kernel_size }` vindos do `NeuralModel.cpp` (ref: `convnet.h:108-118`).
- **Tarefa 1.1.2** [DONE]: Remover a rejeição explícita `"WaveNet 'head' (post-stack sub-object) is not supported (F6)"` no arquivo `src/loader/nam_json/topology.rs:648`.
- **Tarefa 1.1.3** [DONE]: Ajustar `FreeWavenetGeometry` se necessário para passar as informações do Head.

### Sprint 1.2: DSP do Head e Integração

- **Tarefa 1.2.1** [DONE]: Criar uma estrutura `PostStackHead` em `src/models/wavenet/` contendo um `Conv1d` (já existente no projeto) e o suporte a ativação dinâmica.
- **Tarefa 1.2.2** [DONE]: Adicionar o `PostStackHead` opcional ao `WaveNetModelDyn`.
- **Tarefa 1.2.3** [DONE]: Ajustar o ciclo `process` do motor dinâmico para rotear o sinal da pilha para o `PostStackHead` antes do `head_scale`.
- **Tarefa 1.2.4** [DONE]: Somar o tamanho do kernel do `PostStackHead` ao cálculo de `receptive_field` global do modelo para o `prewarm`.
- **Tarefa 1.2.5** [DONE]: Escrever testes para validar o fluxo, providenciando ou construindo um `golden test` com um modelo simulado que utilize Post-stack Head.

---

## Épico 2: MT5 — SlimmableWavenet e Containers Aninhados (F5 e F11)

Modelos "Slimmable" permitem ajustes de carga computacional e qualidade em tempo real sem a necessidade de recarregar todo o plugin.

### Sprint 2.1: Swap Dinâmico (Slimmable via SPSC GC)

- **Tarefa 2.1.1** [DONE]: Desenvolver a infraestrutura de extração em `src/models/slimmable.rs` para permitir que o motor de instanciamento fatiar os pesos com base em um tamanho de camada dinâmico. O `NeuralAmpModelerCore` ajusta em runtime, mas para manter o hot-path RT-safe no Rust (`zero-alloc`, lock-free), o `nam-rs` utilizará a arquitetura existente de GC:
  - A thread assíncrona cria uma nova instância `WaveNetModelDyn` recortando (`slice`) os vetores de canais/pesos.
  - Uma nova instância leve é trocada atomicamente com o `SPSC GC` e o modelo antigo é varrido da thread RT.
  - **Implementado**: funções `slice_conv1d`, `slice_dense`, `slice_wavenet_layer`, `slice_wavenet_array`, `slice_wavenet_model` em `src/models/slimmable.rs` + método `WaveNetModelDyn::slice_channels()`.
  - **Limitação conhecida**: `condition_dsp` é setado como `None` no modelo fatiado (não clonável genericamente). Deve ser endereçado na Tarefa 2.1.2 ou resolvido com rebuild a partir do JSON original.
  - `PostStackHead` agora deriva `Clone`.
- **Tarefa 2.1.2** [DONE]: Integrar este comportamento ao `adaptive.rs` para permitir que o slider de qualidade re-estancie o SLimmable sem falhas.

### Sprint 2.2: Suporte a Containers Aninhados (F11)

- **Tarefa 2.2.1**: Revisar `src/models/container.rs` (que hoje já prevê SlimmableModel) e o desserializador `deserialize_submodels` em `src/loader/nam_json/validation.rs` para permitir de forma recursiva e segura que o modelo embutido seja um próprio container, ou mais provável, um `SlimmableWavenet`.
- **Tarefa 2.2.2**: Produzir um carregamento robusto do `slimmable_container.nam`, validando se a topologia agora aprova os sub-modelos.

---

## Épico 3: MT6 — Arquitetura ConvNet e Multi-canal (F4 e F10)

O `ConvNet` é um modelo legável do NAM, raramente usado por usuários finais, mas requerido para completude e conformidade estrita (paridade total). Os modelos Multi-canal (`in_channels > 1`, `out_channels > 1`) abrem caminho para DSP de pedais complexos.

### Sprint 3.1: Parsing e DSP do ConvNet (F4)

- **Tarefa 3.1.1**: Estender a `NamConfig` / `NamModelData` em `nam_json` para suportar `architecture: "ConvNet"`.
- **Tarefa 3.1.2**: Implementar `BatchNorm1D` (normalização por lote) que é requerida pelos `ConvNetBlocks`, de forma nativamente vetorizada (`vfmadd231ps` em x86-64-v3).
- **Tarefa 3.1.3**: Implementar o DSP modular:
  - `ConvNetBlock` contendo: Conv1D (ou equivalente) -> BatchNorm -> Activation.
  - `ConvNetModel` implementando `NamModel` para encadear os blocos e o Post-stack Head.
- **Tarefa 3.1.4**: Conectar `ConvNetModel` ao `dispatcher`, efetuando o carregamento da lista de pesos adequadamente (`[weights]`).

### Sprint 3.2: Multi-canal (F10)

- **Tarefa 3.2.1**: Analisar o design da trait `NamModel::process(&mut self, input: &[f32], output: &mut [f32])`. Atualmente orientada a processamento mono/serial.
- **Tarefa 3.2.2**: Para suportar `in_channels > 1` e `out_channels > 1`, decidir se implementaremos arrays planares (ex: iteradores multi-slice) ou sinais entrelaçados internamente. (Recomenda-se buffers N-dimensionais empacotados nos scratch buffers para os tensores convolucionais).
- **Tarefa 3.2.3**: Alterar restrições no dispatcher e no loader para parar de injetar `in_channels=1` / `out_channels=1` compulsivo na topologia, liberando F10.
- **Tarefa 3.2.4**: Validar desempenho das convoluções com dimensões de in/out estendidas com `cargo bench`.

---

> **Atenção (Revisor-Auditor):**
> Todas as implementações acima devem assegurar a RT-Safety com `Zero Heap Drop` na thread de DSP, uso primário de `AlignedVec<T>` de 64 bytes e fallback para ISA x86-64-v3 contendo intrínsecos `AVX2/FMA`. Não se deve utilizar `f32::tanh()` em favor do aproximações nativas pré-existentes no repositório. O processo será progressivo com pull-requests isolados via `utils/lints.sh`.
