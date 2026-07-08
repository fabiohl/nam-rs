<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Backlog de Execução: Resolução do LSTM Recurrent State Drift (Achado 2)

Este documento descreve os Epics, Sprints e Tarefas Técnicas detalhadas para a implementação da compensação de erro por soma de Kahan no estado das células LSTM, visando eliminar o drift recorrente.
> Rodar `utils/quality-dashboard.sh` antes e depois para avaliar o resultado.

---

## 🧭 Visão Geral e Mitigação de Risco

* **Objetivo:** Garantir paridade matemática exata no stream de áudio contínuo em relação ao oráculo `f64`, aplicando Kahan Summation na acumulação do *Cell State*.
* **Risco de Performance:** Baixo/Médio. O acréscimo de instruções SIMD no hot-path da thread de tempo real será validado rigorosamente por microbenchmarks antes da integração final.
* **Política de Modificação:** Nenhuma alocação dinâmica (heap) em tempo de execução. Manutenção estrita das diretrizes de segurança em tempo real (RT-Safety).

---

## 🏃 Épico: Compensação de Round-off Recorrente nas Portas da LSTM

### 📅 Sprint 1: Preparação da Estrutura de Dados e Assinaturas (Camada de Dados) [DONE]

Nesta etapa, preparamos o armazenamento persistente do shadow state do erro acumulado para cada célula LSTM.

#### 🎯 Tarefa 1.1: Modificação do Estado Estático (`LstmLayer`) [DONE]

* **Arquivo:** [`src/models/lstm/layer.rs`](file:///home/fabio/nam-rs/src/models/lstm/layer.rs)
* **Ações:**
  * Adicionar o campo `pub cell_error: Aligned64<[f32; H]>` à struct `LstmLayer`.
  * Inicializar `cell_error` com `0.0` no método `new()`.
  * Atualizar o método `reset_states()` para limpar `self.cell_error.fill(0.0)`.

#### 🎯 Tarefa 1.2: Modificação do Estado Dinâmico (`LstmLayerDyn`) [DONE]

* **Arquivo:** [`src/models/lstm/layer_dyn.rs`](file:///home/fabio/nam-rs/src/models/lstm/layer_dyn.rs)
* **Ações:**
  * Adicionar o campo `pub cell_error: AlignedVec<f32>` à struct `LstmLayerDyn`.
  * Alocar `cell_error` com tamanho `hidden_size` inicializado com `0.0f32` no método `new()`.
  * Atualizar o método `reset_states()` para limpar `self.cell_error.fill(0.0)`.

#### 🎯 Tarefa 1.3: Atualização do Trait `SimdMath` e Dispatches Associados [DONE]

* **Arquivos:**
  * [`src/math/common/traits.rs`](file:///home/fabio/nam-rs/src/math/common/traits.rs)
  * [`src/math/common/avx2_impl/activations.rs`](file:///home/fabio/nam-rs/src/math/common/avx2_impl/activations.rs)
  * [`src/math/common/avx512/activations.rs`](file:///home/fabio/nam-rs/src/math/common/avx512/activations.rs)
* **Ações:**
  * Alterar a assinatura de `fused_lstm_gates_dyn` no trait `SimdMath` para incluir a referência mutável do buffer de erro:

    ```rust
    unsafe fn fused_lstm_gates_dyn(
        gates: &mut [f32],
        cell_state: &mut [f32],
        cell_error: &mut [f32], // Novo parâmetro
        hidden_state: &mut [f32],
        hidden_size: usize,
    );
    ```

  * Propagar o novo parâmetro `cell_error` nos blocos de implementação e macros correspondentes nos arquivos de ativação SIMD (`avx2_impl` e `avx512`).

---

### 📅 Sprint 2: Implementação Matemática da Compensação de Kahan nos Gates SIMD

Nesta etapa, implementamos a lógica matemática real de Kahan nos kernels elementares de gates em ponto flutuante SIMD (`__m256`, `__m512`) e escalares de cauda.

#### 🎯 Tarefa 2.1: Implementação Matemática Kahan nos Kernels AVX2

* **Arquivo:** [`src/math/lstm/gates.rs`](file:///home/fabio/nam-rs/src/math/lstm/gates.rs)
* **Ações:**
  * Atualizar as assinaturas das funções `fused_lstm_gates_avx2_hf`, `fused_lstm_gates_avx2_std` e `fused_lstm_gates_avx2` para receber `cs_err: __m256` e retornar a tripla `(__m256, __m256, __m256)` (novo cell state, novo cell error, hidden output).
  * Substituir o cálculo clássico de atualização da célula pela formulação de Kahan:

    ```rust
    let f_cs = _mm256_mul_ps(sig_f, cs);
    let f_err = _mm256_mul_ps(sig_f, cs_err);
    let i_g = _mm256_mul_ps(sig_i, tanh_g);
    let y = _mm256_sub_ps(i_g, f_err);
    let new_cs = _mm256_add_ps(f_cs, y);
    let new_cs_err = _mm256_sub_ps(_mm256_sub_ps(new_cs, f_cs), y);
    ```

  * Retornar `(new_cs, new_cs_err, hidden)`.

#### 🎯 Tarefa 2.2: Implementação Matemática Kahan nos Kernels AVX-512

* **Arquivo:** [`src/math/lstm/gates.rs`](file:///home/fabio/nam-rs/src/math/lstm/gates.rs)
* **Ações:**
  * Atualizar as assinaturas das funções `fused_lstm_gates_avx512_hf`, `fused_lstm_gates_avx512_std` e `fused_lstm_gates_avx512` de forma análoga, operando com `__m512` e instruções `_mm512_*`.

#### 🎯 Tarefa 2.3: Atualização dos Processadores Dinâmicos de Gates

* **Arquivo:** [`src/math/lstm/gates.rs`](file:///home/fabio/nam-rs/src/math/lstm/gates.rs)
* **Ações:**
  * Modificar `fused_lstm_gates_dyn_avx2` e `fused_lstm_gates_dyn_avx512` para aceitar `cell_error: &mut [f32]`.
  * Carregar `cs_err` via `_loadu_ps` a partir de `cell_error.as_ptr().add(j)`.
  * Salvar o `new_cs_err` resultante via `_storeu_ps` no endereço `cell_error.as_mut_ptr().add(j)`.
  * Atualizar `fused_lstm_gates_dyn_tail` para aceitar `cell_error: &mut [f32]` e aplicar a lógica de compensação de Kahan na cauda de processamento de forma escalar.

---

### 📅 Sprint 3: Integração e Adaptação dos Laços Principais de Execução (Layer Kernels)

Nesta etapa, ligamos a nova matemática estruturada aos laços de execução principais que rodam frame a frame.

#### 🎯 Tarefa 3.1: Atualização do Macro de Geração de Processamento Estático

* **Arquivo:** [`src/models/lstm/layer_kernels.rs`](file:///home/fabio/nam-rs/src/models/lstm/layer_kernels.rs)
* **Ações:**
  * Modificar a macro `define_lstm_process!` para realizar o carregamento e armazenamento do erro da célula:
    * Carregar o erro da célula via `$load(self.cell_error.as_ptr().add(i))`.
    * Receber a tripla de retorno da chamada `$fused_gates(...)`.
    * Armazenar o novo erro resultante via `$store(self.cell_error.as_mut_ptr().add(i), new_cs_err)`.
  * Atualizar a seção de tail handling na macro para carregar e descarregar o array `temp_cerr` e salvar os resultados residuais em `self.cell_error[i + j]`.
  * Atualizar a função de fallback escalar `process_sample_scalar` para carregar `cs_err` do array `self.cell_error`, aplicar a compensação matemática e salvar o novo estado de erro.

#### 🎯 Tarefa 3.2: Integração nos Kernels Dinâmicos (`LstmLayerDyn`)

* **Arquivo:** [`src/models/lstm/layer_dyn_kernels.rs`](file:///home/fabio/nam-rs/src/models/lstm/layer_dyn_kernels.rs)
* **Ações:**
  * Atualizar os métodos `process_sample_avx2` e `process_sample_avx512` para separar a fatia correspondente ao erro de célula:

    ```rust
    let (cell_err_slice, _) = &mut self.cell_error.split_at_mut(h);
    ```

  * Propagar `cell_err_slice` para as chamadas de dispatch `fused_lstm_gates_dyn_avx2` e `fused_lstm_gates_dyn_avx512`.
  * Atualizar a cauda escalar/fallback `process_sample_scalar` para ler e atualizar os valores residuais do array `self.cell_error`.

---

### 📅 Sprint 4: Validação, Testes e Análise de Regressão de Desempenho

Garantir que a implementação não quebre a corretude das simulações existentes e validar a segurança de tempo real.

#### 🎯 Tarefa 4.1: Atualização dos Testes Unitários de Paridade e Determinismo

* **Arquivo:** [`src/models/lstm/lstm_test.rs`](file:///home/fabio/nam-rs/src/models/lstm/lstm_test.rs)
* **Ações:**
  * Em `assert_dyn_layer_parity`, copiar o estado de `cell_error` de `layer_scalar` para `layer_simd` antes de iniciar as iterações de teste.
  * Adicionar uma asserção assertiva validando que os erros de célula das versões escalar e SIMD mantêm paridade exata (`abs() < 1e-2`) ao final de cada passo.
  * Em `test_dyn_layer_determinism`, garantir a cópia de `cell_error` e asserir a igualdade estrita dos vetores pós-processamento.

#### 🎯 Tarefa 4.2: Execução e Auditoria de Testes Rápidos

* **Ações:**
  * Validar que todas as suítes rápidas e a verificação estrutural passem sem falhas nem divergências de paridade com o oráculo de medida `f64`.

#### 🎯 Tarefa 4.3: Microbenchmarks de Latência e CPU Budget

* **Ações:**
  * Medir e analisar a latência do hot-path de áudio. Assegurar que o impacto de performance seja insignificante e que a folga de CPU continue confortável (< 1% do budget total).
  * Rodar `utils/quality-dashboard.sh` e compara com a baseline em `/quality-dashboard.txt`.
