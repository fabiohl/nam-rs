<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->
# Pesquisador Inovador & Revisor Auditor: Análise de Qualidade de Modelos (FiLM e LSTM)

Este documento sumariza os achados críticos da investigação de paridade e divergência reportados no `utils/quality-dashboard.sh` para os modelos WaveNet A2-FiLM Lite e BossLSTM.
Ensejará atualização de: `docs/audio_fidelity_map.md`, `docs/fastmath-approximations.md`, `docs/cpp_parity_map.md` e `docs/architecture.md`.

## Achado 1: FiLM Floating-Point Associativity Gap

**Diagnóstico Detalhado:**

1. **Divergência de Paridade:** O teste de fidelidade acusa um ESR de `1.54e-2` (SNR 18.1 dB) audível para o modelo `WaveNet A2-FiLM-Lite (CH=3)` e `2.50e-4` (SNR 36.0 dB) para `WaveNet A2-FiLM-Full (CH=8)`.
2. **Bit-Parity no Oráculo:** O oráculo f64 do `nam-rs` bate com a produção Rust em `f32` no limite do ponto flutuante (ESR ≈ `1e-14`), comprovando que a topologia, dimensões e mapeamento de pesos/bias estão 100% corretos.
3. **Mapeamento Aritmético:** A divergência decorre da ordem associativa e redução na camada FiLM:
   - **C++ NAMCore (Eigen):** Utiliza multiplicação padrão de matriz column-major por vetor/matriz, acumulando sequencialmente coluna a coluna de forma linear, seguido da adição do bias: `sum = (weights * cond) + bias`.
   - **Rust (nam-rs):** Inicializa o acumulador com o `bias` e depois adiciona o produto escalar computado em árvore binária paralela SIMD (`dot_product_avx2`): `sum = bias + (weights * cond)`.
   - Essa assimetria na árvore de soma altera a propagação de arredondamento em float f32 (mudança de 1 ULP por passo), o que se propaga pelas 23 camadas dilated causais acumulando desvios numéricos.

**Plano de Correção (Mimetização Aritmética):**

1. **Mapeamento de Produção (`src/models/a2/film.rs`):**
   - Substituir a chamada para a redução paralela `dot_product_avx2(w_row, cond_slice)` e o acumulador inicializado com bias em `cond_to_scale_shift`.
   - Implementar uma acumulação estritamente sequencial:

     ```rust
     let mut sum = 0.0f32;
     for i in 0..cond_per_group {
         sum += *w_row.get_unchecked(i) * *cond_slice.get_unchecked(i);
     }
     sum += *self.bias.get_unchecked(global_out);
     ```

   - Remover a função local/privada `dot_product_avx2` para manter o código limpo.
2. **Mapeamento do Oráculo (`src/testing/reference_oracle/a2.rs`):**
   - Alterar `FiLMOracleSlot::apply` para seguir a mesma ordem linear:

     ```rust
     let mut sum = 0.0f64;
     for k in 0..cond_per_group {
         sum += self.weights[w_off + row * cond_per_group + k] * condition[cond_off + k];
     }
     sum += self.bias[global_out];
     ```

3. **Desempenho (x86-64-v3):** Como `cond_size` é pequeno (normalmente <= 8), loops escalares sequenciais curtos em Rust compilam para sequências eficientes de instruções FMA em hardware nativo (x86-64-v3), eliminando overhead de redução horizontal sem perda de performance.

## Achado 2: LSTM Recurrent State Drift

**Diagnóstico:** Ao contrário do mapeado preliminarmente em algumas seções da documentação oficial, os desvios contínuos dos modelos `BossLSTM 1x16` (1.04e-2 ESR) e `BossLSTM 2x8` (2.68e-3 ESR) *não* foram totalmente solucionados pela remoção do `f16c`. O "bit-exactness" outrora documentado é válido apenas para excitações de curtíssima duração (2000 samples). Em sinais de produção e stream contínuo, a acumulação sucessiva do *Cell State* através das portas (`f32`) invariavelmente produz o *Recurrent Drift*.

### Formulação Matemática da Compensação

A recorrência clássica de atualização do estado da célula em ponto flutuante de precisão simples (`f32`) é dada por:
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$

onde $f_t$ é o forget gate, $i_t$ é o input gate e $g_t$ é o candidate cell state.
Para mitigar o acúmulo sistemático do erro de arredondamento de ponto flutuante ($O(N \cdot \epsilon)$) que causa o desvio em streams de áudio longos, adota-se um vetor de shadow-state `cell_error` ($e_t \in \mathbb{R}^H$).
Como o estado anterior $c_{t-1}$ é multiplicado pelo forget gate $f_t$, o erro acumulado correspondente $e_{t-1}$ também deve ser escalado pelo mesmo fator $f_t$.

O passo recorrente compensado de Kahan é definido da seguinte forma a cada amostra:

1. **Escalonamento do estado e erro anteriores:**
   $$c_{\text{scaled}} = f_t \odot c_{t-1}$$
   $$e_{\text{scaled}} = f_t \odot e_{t-1}$$
2. **Cálculo do termo de atualização de entrada:**
   $$u_t = i_t \odot g_t$$
3. **Compensação com o erro escalado acumulado:**
   $$y_t = u_t - e_{\text{scaled}}$$
4. **Soma provisória do estado:**
   $$\tilde{c}_t = c_{\text{scaled}} + y_t$$
5. **Recuperação e atualização do novo erro de arredondamento:**
   $$e_t = (\tilde{c}_t - c_{\text{scaled}}) - y_t$$
6. **Atualização final do estado da célula:**
   $$c_t = \tilde{c}_t$$

### Mapeamento de Instruções SIMD (AVX2 / AVX-512)

Para evitar que o compilador otimize ou fusione operações incorretamente, a soma provisória $\tilde{c}_t = c_{\text{scaled}} + y_t$ **não** deve utilizar instruções Fused Multiply-Add (FMA) que impeçam a extração do erro residual.
No kernel AVX2 (`__m256`), as operações aritméticas são mapeadas como:

```rust
let f_cs = _mm256_mul_ps(sig_f, cs);
let f_err = _mm256_mul_ps(sig_f, cs_err);
let i_g = _mm256_mul_ps(sig_i, tanh_g);
let y = _mm256_sub_ps(i_g, f_err);
let new_cs = _mm256_add_ps(f_cs, y);
let new_cs_err = _mm256_sub_ps(_mm256_sub_ps(new_cs, f_cs), y);
```

No kernel AVX-512 (`__m512`), o fluxo é análogo utilizando os registradores de 512 bits:

```rust
let f_cs = _mm512_mul_ps(sig_f, cs);
let f_err = _mm512_mul_ps(sig_f, cs_err);
let i_g = _mm512_mul_ps(sig_i, tanh_g);
let y = _mm512_sub_ps(i_g, f_err);
let new_cs = _mm512_add_ps(f_cs, y);
let new_cs_err = _mm512_sub_ps(_mm512_sub_ps(new_cs, f_cs), y);
```

### Solução Exigida (Plano de Correção Estrutural)

1. **Alteração do Estado das Células da LSTM:**
   - **Estático ([layer.rs](file:///home/fabio/nam-rs/src/models/lstm/layer.rs)):** Adicionar `pub cell_error: Aligned64<[f32; H]>` à struct [LstmLayer](file:///home/fabio/nam-rs/src/models/lstm/layer.rs#L9). Inicializar com `0.0` no construtor `new()` e limpar no método `reset_states()`.
   - **Dinâmico ([layer_dyn.rs](file:///home/fabio/nam-rs/src/models/lstm/layer_dyn.rs)):** Adicionar `pub cell_error: AlignedVec<f32>` à struct [LstmLayerDyn](file:///home/fabio/nam-rs/src/models/lstm/layer_dyn.rs#L14). Alocar no construtor `new()` e preencher com `0.0` em `reset_states()`.

2. **Modificação das assinaturas em [gates.rs](file:///home/fabio/nam-rs/src/math/lstm/gates.rs):**
   - Atualizar `fused_lstm_gates_avx2` e `fused_lstm_gates_avx512` (e suas variantes `_hf` e `_std`) para aceitar o registrador de erro (`cs_err`) e retornar a tripla `(new_cs, new_cs_err, hidden)`.
   - Atualizar os processadores dinâmicos `fused_lstm_gates_dyn_avx2`, `fused_lstm_gates_dyn_avx512` e a cauda escalar `fused_lstm_gates_dyn_tail` para receber a fatia mutável `cell_error: &mut [f32]` e aplicar a compensação diretamente na memória dos registradores / arrays.

3. **Adequação do Trait `SimdMath` e Dispatches:**
   - **Interface ([traits.rs](file:///home/fabio/nam-rs/src/math/common/traits.rs)):** Incluir `cell_error: &mut [f32]` em [SimdMath::fused_lstm_gates_dyn](file:///home/fabio/nam-rs/src/math/common/traits.rs#L756).
   - **Implementações ([activations.rs](file:///home/fabio/nam-rs/src/math/common/avx2_impl/activations.rs) e [activations.rs](file:///home/fabio/nam-rs/src/math/common/avx512/activations.rs)):** Atualizar as pontes de delegação para propagar a referência mutável de erro da célula.

4. **Atualização dos Laços Principais de Execução:**
   - **Estático ([layer_kernels.rs](file:///home/fabio/nam-rs/src/models/lstm/layer_kernels.rs)):** No macro `define_lstm_process`, carregar o erro da célula via `$load(self.cell_error.as_ptr().add(i))`, propagar para o kernel de gates e salvar `$store(self.cell_error.as_mut_ptr().add(i), new_c_err)`. Fazer o mesmo na cauda (tail). Ajustar também o fallback `process_sample_scalar`.
   - **Dinâmico ([layer_dyn_kernels.rs](file:///home/fabio/nam-rs/src/models/lstm/layer_dyn_kernels.rs)):** Atualizar os loops AVX2, AVX-512 e `process_sample_scalar` para separar e propagar o buffer mutável correspondente a `cell_error`.

5. **Gating de Benchmark e GO / No GO:**
   - Executar obrigatoriamente `cargo bench` focado nos benchmarks de LSTM para avaliar o impacto do acréscimo de instruções aritméticas no hot-path da thread de áudio de tempo real.
   - Embora a folga orçamentária de CPU para modelos baseados em LSTM seja confortável (consome atualmente < 1% do budget total), a integração final à branch principal de produção fica condicionada à manutenção de um consumo de CPU aceitável sem introdução de jitters de áudio.

## Achado 3: `wavenet_a2_max.nam` Ativamente Quebrado (Divergência Crítica no `condition_dsp`)

> Vide docs/cpp_parity_map.md §4.4 e §7.1.

**Diagnóstico Detalhado:**
O modelo `wavenet_a2_max` (CH=4, condition_size=8) é o único flagship do A2 ativamente quebrado, gerando uma saída com erro severo (MSE ≈ 2.46e3, SNR ≈ -15.6 dB, ESR ≈ 36.1). Apesar do modelo carregar corretamente e executar via `WaveNetA2Dyn`, seu áudio de saída não faz sentido e, por cautela, ele foi contido de forma hardcoded no dispatcher (`is_disabled_broken_a2_flagship`).

A investigação aprofundada revelou que o desvio fundamental reside firmemente no oráculo f64 (`src/testing/reference_oracle/a2.rs`), e não no código de produção. Historicamente, tentou-se mimetizar o oráculo f64 que possuía dois bugs estruturais gravíssimos quando operando com topologias genéricas (sub-modelos `condition_dsp` com `head_size > 1` e `head1x1.groups > 1`):

1. **Leitura Incorreta de Pesos/Bias em `head1x1` (Corrupção de Cursor)**:
   No loop de carregamento de pesos de `head1x1` (linhas 485-494 de `src/testing/reference_oracle/a2.rs`), o oráculo utilizava incorretamente a quantidade de canais da camada (`ch` = 3 no sub-modelo `condition_dsp` do A2 Max) em vez da quantidade correta de canais acumulados no cabeçalho (`head_accum_size` = 6):
   - **Errado**: `cursor.read_f64(ch * h1_in_size)` e `cursor.read_f64(ch)` (lia `3 * 2 = 6` pesos e `3` bias).
   - **Correto**: `cursor.read_f64(head_accum_size * h1_in_size)` e `cursor.read_f64(head_accum_size)` (deveria ler `6 * 2 = 12` pesos e `6` bias).
   Isso fazia com que o oráculo f64 ficasse com o cursor de weights completamente desalinhado e corrompido para todas as leituras de camadas/arrays seguintes.

2. **Divergência Crítica no Cálculo do Gating de Grupos (`ch_per_group`)**:
   Na execução do `head1x1` no loop principal (linha 733 de `src/testing/reference_oracle/a2.rs`), o oráculo repetia o erro usando `ch` em vez de `head_accum_size`:
   - `let ch_per_group = ch / h1_groups;` (resultando em 1, em vez de `6 / 3 = 2`).
   Como consequência, os buffers de scratch populavam incorretamente os canais de gating, gerando zeros e estourando o cálculo em cascata.

3. **Bug Dimensional e Finalização de Cabeçalho para `head_size > 1`**:
   Quando `head_size > 1` (como na saída da WaveNet `condition_dsp`, que produz `head_size = 8` canais de condicionamento), a finalização do cabeçalho no oráculo (`src/testing/reference_oracle/a2.rs` linhas 806-824):
   - Alocava o buffer de saída como `vec![0.0f64; num_frames]` (apenas 1 valor por frame em vez de `num_frames * head_size`).
   - Loopava `t` de 0 até `head_size` tratando a contagem de canais de saída como se fosse a extensão dos taps do kernel da convolução causal, somando tudo em um único acumulador escalar `y`.
   - **Comportamento C++ Esperado**: O cabeçalho é um `Conv1D` com `kernel_size = 1` que produz `head_size` canais de saída interleaved por frame. O oráculo precisa gerar um vetor de saída de tamanho `num_frames * head_size` e computar o produto das matrizes de forma interleaved para cada canal `oc` de 0 a `head_size - 1` na amostragem causal.

**Plano de Correção (Sprints e Desbloqueio):**

1. **Correção do Oráculo f64 (`src/testing/reference_oracle/a2.rs`):**
   - Alterar o campo `head_b` no struct auxiliar `ArrayState` para `Vec<f64>`.
   - Ajustar as leituras de weights e bias do `head1x1` na inicialização para usar `head_accum_size` ao invés de `ch`:

     ```rust
     let head1x1_w: Vec<f64> = if head1x1_active {
         cursor.read_f64(head_accum_size * h1_in_size)
     } else {
         vec![]
     };
     let head1x1_b: Vec<f64> = if head1x1_active {
         cursor.read_f64(head_accum_size)
     } else {
         vec![]
     };
     ```

   - Corrigir a leitura da bias e pesos do cabeçalho final quando `head_size > 1`, respeitando `head_bias` se presente:

     ```rust
     let (head_w, head_b) = if head_size == 1 {
         // ... mantém comportamento original v1 ...
         (head_w, vec![head_b])
     } else {
         let hw_count = head_accum_size * head_size;
         let head_w = cursor.read_f64(hw_count);
         let head_b = if head_bias_active {
             cursor.read_f64(head_size)
         } else {
             vec![0.0f64; head_size]
         };
         (head_w, head_b)
     };
     ```

   - Corrigir `ch_per_group` no loop principal da `head1x1`:

     ```rust
     let ch_per_group = arr.head_accum_size / h1_groups;
     ```

   - Alterar o buffer de saída do oráculo para acomodar o tamanho multidimensional:

     ```rust
     let mut output = vec![0.0f64; num_frames * last_arr.head_size];
     ```

   - Implementar o loop de finalização do cabeçalho de acordo com a geometria causal `Conv1D (kernel_size=1)` para `head_size > 1`, salvando de forma interleaved:

     ```rust
     if last_arr.head_size == 1 {
         // ... mantém processamento original com A2_HEAD_KERNEL taps ...
     } else {
         let so = head_col * max_ch;
         for oc in 0..last_arr.head_size {
             let mut y = last_arr.head_b[oc];
             let wo = oc * lch;
             for c in 0..last_arr.head_accum_size {
                 y = mul_add_f64(last_arr.head_w[wo + c], head_acc[so + c], y, acc_mode);
             }
             output[f * last_arr.head_size + oc] = y * head_scale;
         }
     }
     ```

2. **Homologação e Desbloqueio da Branch de Produção:**
   - Remover a função e a chamada limitadora `is_disabled_broken_a2_flagship` em `src/loader/dispatcher/wavenet/mod.rs` para permitir o despacho correto do modelo A2 Max para o motor dinâmico `WaveNetA2Dyn`.
   - Descomentar os testes de cross-validation no `tests/parity/cpp_parity.rs`:
     - `live_cross_validation_wavenet_a2_max`
     - `live_cross_validation_v2_wavenet_a2_max`
   - Re-enabilitar os testes do oráculo no `tests/parity/reference_oracle_f64.rs`:
     - `test_oracle_vs_python_anchor_a2_generic`
     - `test_oracle_a2_generic`
     - `test_decomposition_a2_generic`
     - `test_combined_simulation_a2_generic`
   - Garantir que a validação feche com MSE ideal e SNR limpo (> 100 dB) nas duas pontas, provando conformidade e paridade absoluta com o oráculo e com o NAMCore C++.
