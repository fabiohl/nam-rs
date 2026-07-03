<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Diagnóstico e Solução do BUG-1

Este documento detalha a investigação, o diagnóstico e o plano estruturado para a resolução do **BUG-1 🔴**.

## 1. Detalhes do Erro e Sintoma

O benchmark dinâmico de WaveNet A2 falha no carregamento com o seguinte pânico:

```text
thread 'main' panicked at benches/inference_bench.rs:2140:40:
Dispatcher failed for WaveNet A2 Dynamic benchmark: WaveNet model rejected:
Layer 0 is missing or has invalid 'kernel_size' — required for free geometry
WaveNet A1.. Detected: 1 layer(s) with geometry [(4, 0)]
```

## 2. Investigação e Causa Raiz

1. **Construção Manual do Modelo**: A fixture `make_wavenet_a2_dyn_data()` em `benches/inference_bench.rs` cria uma estrutura `NamModelData` diretamente em código Rust (struct literal). Portanto, o campo `layer_raw` (que contém o JSON bruto deserializado) fica como `None`.
2. **Definição Incorreta de Ativação**: A fixture define `activation: Some("Tanh".to_string())`.
3. **Classificação Incorreta (A1 vs A2)**:
   - O detector `is_a2_shape()` verifica se alguma camada usa `"Tanh"`. Caso encontre, descarta o modelo como A1:

     ```rust
     for l in &data.config.layers {
         if l.activation.as_deref() == Some("Tanh") {
             return None;
         }
     }
     ```

   - Como a fixture de A2 tem `activation: Some("Tanh".to_string())`, `is_a2_shape()` retorna `None`.
4. **Rejeição pela Geometria A1**:
   - Sem ser detectado como A2, o dispatcher envia o modelo para o parser de topologia WaveNet A1 (`get_wavenet_topology()`).
   - O parser A1 não encontra compatibilidade com nenhum SKU do catálogo (Nano/Feather/Lite/Standard).
   - O modelo cai na validação de geometria livre (A1 Free), que busca `kernel_size` (singular). Como a fixture de A2 define apenas `kernel_sizes` (plural, vetor de 23 elementos), `kernel_size` é `None` (ausente).
   - O parser rejeita o modelo gerando o erro de validação.

## 3. Proposta de Solução

Corrigir o campo de ativação na fixture `make_wavenet_a2_dyn_data()` em [inference_bench.rs](file:///home/fabio/nam-rs/benches/inference_bench.rs) de `"Tanh"` para `"LeakyReLU"`. Modelos WaveNet A2 legítimos usam ativação LeakyReLU (ou gated/blended estruturado em JSON) e não Tanh na hot-path.

Esta mudança simples e correta fará com que `is_a2_shape()` identifique o modelo como `A2TopologyResult::Dynamic` (pois possui canais = 4, fora do fast-path const-generic `[3, 8]`), roteando-o corretamente para `WaveNetA2Dyn` no dispatcher, sem passar pelo parser de topologia A1.

---

## 4. Planejamento de Epics e Tarefas

### Épico 1: Resolução de Fixtures de Benchmark de WaveNet A2 Dinâmico

- **Tarefa T1.1 (Investigação e Correção)**:
  Alterar a ativação da fixture `make_wavenet_a2_dyn_data()` em [inference_bench.rs](file:///home/fabio/nam-rs/benches/inference_bench.rs) para `"LeakyReLU"`.
- **Tarefa T1.2 (Validação e Fatoração)**:
  Compilar e executar o benchmark individual `A2Dyn_Gated_64samp_48kHz` via `cargo bench --profile dev` para garantir que o carregador aceita a fixture e executa o processamento com sucesso.
- **Tarefa T1.3 (Regressão)**:
  Rodar o conjunto rápido de testes (`utils/tests-quick.sh`) para certificar que nenhuma regressão foi introduzida no processo.
