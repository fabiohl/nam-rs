<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->
# Pesquisador Inovador & Revisor Auditor: Análise de Qualidade de Modelos (FiLM e LSTM)

Este documento sumariza os achados críticos da investigação de paridade e divergência reportados no `utils/quality-dashboard.sh` para os modelos WaveNet A2-FiLM Lite e BossLSTM.
Ensejará atualização de: `docs/audio_fidelity_map.md`, `docs/fastmath-approximations.md`, `docs/cpp_parity_map.md` e `docs/architecture.md`.

## Achado 1: FiLM Floating-Point Initialisation Scale Bias Gap (Resolvido)

**Diagnóstico Correto e Causa Raiz Real:**

1. **Refutação do Diagnóstico de Associatividade:** O diagnóstico anterior atribuía o gap de ESR do FiLM (`1.54e-2` para Lite CH=3 e `2.50e-4` para Full CH=8) a diferenças na ordem associativa de soma (redução horizontal AVX2 vs. Eigen sequencial). No entanto, ambos os modelos flagship de FiLM possuem `condition_size = 1`. Com apenas um elemento de condição, a redução horizontal nunca é executada e o laço escalar faz uma única operação de multiplicação, computando exatamente a mesma expressão bit-a-bit tanto em Rust quanto no Eigen C++.
2. **Causa Raiz Real (Inicialização não-identity-biased):** A causa real estava no gerador de fixtures sintéticas (`tests/fixtures/generate_a2_fixtures.py::generate_weights_film`), que gerava os pesos e bias da metade de `scale` do FiLM usando a mesma distribuição uniforme de média zero que o canal de `shift` (com módulo pequeno). Na convenção do FiLM em redes treinadas, o canal de `scale` deve ser inicializado ao redor de `1.0` (preservando o fluxo de sinal). A inicialização com média próxima a zero esmagava a energia do sinal de áudio resultante, enquanto a divergência de ponto flutuante absoluto normal de ~1e-7 entre as implementações continuava igual. Ao calcular a métrica normalizada pela energia do sinal (ESR), o denominador muito pequeno inflava artificialmente a divergência relativa.
3. **Resolução e Correção:** O gerador de fixtures (`generate_a2_fixtures.py`) foi corrigido para aplicar o bias de `1.0` no canal de `scale`. Com o sinal em magnitude normal, as divergências colapsaram para o limite de precisão do float32.
4. **Resultados de Paridade Obtidos:**
   - **WaveNet A2-FiLM-Lite (CH=3):** SNR de **138.3 dB**, ESR de **1.48e-14** (paridade bit-exata ao nível do float32).
   - **WaveNet A2-FiLM-Full (CH=8):** SNR de **138.8 dB**, ESR de **1.31e-14**.
5. **Mitigação pelo Teste de Caos Numérico:** Para garantir que a engine Rust seja robusta a regimes numéricos degenerados e instabilidades, o antigo fixture com bias problemático foi registrado no catálogo como `wavenet_a2_film_chaos_stress.nam` (adicionando um teste de estresse de caos numérico com thresholds degradados intencionalmente: SNR = `12.0` dB, ESR = `3.5e-2`, MR-STFT = `0.498`).

## Achado 2: `wavenet_a2_max.nam` Ativamente Quebrado (Divergência Crítica no `condition_dsp`)

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
