<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Sprints de Refatoração Estrutural (Epics 1, 2 e 3)

Este documento foi criado pela skill `planejador-arquiteto` a partir dos achados detalhados em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md) (Epics 1, 2 e 3). O objetivo é quebrar a execução em sprints ágeis, priorizando a segurança contra regressões e a conformidade estrita com as regras do repositório (`.agents/rules/testing.md` e `.agents/rules/rust.md`).

---

## 🛠️ Portão de Qualidade Geral (Quality Gate)

Ao final de cada sprint (ou tarefa individual significativa), o executor **deve** realizar a verificação estática e dinâmica para garantir que nenhuma regressão de compilação ou funcionalidade foi introduzida:

1. **Validação Estática:**

   ```bash
   utils/lints.sh
   ```

2. **Validação Dinâmica (Testes Unitários e Integração rápidos):**

   ```bash
   utils/tests-quick.sh
   ```

   > [!IMPORTANT]
   > O script `utils/tests-long.sh` **nunca deve ser executado pelo agente de IA**. Se necessário, solicite ao operador humano que execute-o e reporte os resultados.

---

## 📅 Sprint 1 — Extração de Testes Inline (Epic 1)

* **Objetivo:** Trazer o repositório à conformidade com a política `testing.md` §1 (arquivos com mais de 300 linhas de código de produção não devem conter testes inline, mas sim residir em arquivos irmãos `<module>_test.rs` incluídos via `#[path]`).
* **Prioridade:** ALTA (base limpa para as próximas refatorações).
* **Risco Geral:** **ZERO** (movimentação puramente mecânica).

### Tarefas da Sprint 1

* [x] **T1.1 — Extração de testes de `src/models/linear.rs`**
  * **Origem:** [A1 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L50)
  * **Ação:**
    1. Criar o arquivo `src/models/linear_test.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as linhas 334–949 (bloco `#[cfg(test)] mod tests`) de `src/models/linear.rs` para o novo arquivo.
    3. Em `src/models/linear.rs`, declarar o submódulo de teste usando:

       ```rust
       #[cfg(test)]
       #[path = "linear_test.rs"]
       mod tests;
       ```

  * **Validação:** Garantir compilação com `cargo check --tests` e executar `cargo test --lib models::linear`.

* [x] **T1.2 — Extração de testes de `src/models/linear_fft.rs`**
  * **Origem:** [A2 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L58)
  * **Ação:**
    1. Criar o arquivo `src/models/linear_fft_test.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as linhas 398–739 (bloco `mod tests`) de `src/models/linear_fft.rs` para o novo arquivo.
    3. Em `src/models/linear_fft.rs`, declarar o submódulo usando `#[path]`.
  * **Validação:** Compilar e executar `cargo test --lib models::linear_fft`.

* [x] **T1.3 — Extração de testes de `src/math/dsp/fft_radix4.rs`**
  * **Origem:** [A3 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L65)
  * **Ação:**
    1. Criar o arquivo `src/math/dsp/fft_radix4_test.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as linhas 306–532 (bloco `mod tests`) de `src/math/dsp/fft_radix4.rs` para o novo arquivo.
    3. Em `src/math/dsp/fft_radix4.rs`, declarar o submódulo usando `#[path]`.
  * **Validação:** Compilar e executar `cargo test --lib math::dsp::fft_radix4`.

* [x] **T1.4 — Extração de testes de `src/math/common/aligned.rs`**
  * **Origem:** [A4 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L72)
  * **Ação:**
    1. Criar o arquivo `src/math/common/aligned_test.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as linhas 369–488 (bloco `mod tests`) de `src/math/common/aligned.rs` para o novo arquivo.
    3. Em `src/math/common/aligned.rs`, declarar o submódulo usando `#[path]`.
  * **Validação:** Compilar e executar `cargo test --lib math::common::aligned`.

---

## 📅 Sprint 2 — Remoção de Código Morto Verificado (Epic 2)

* **Objetivo:** Limpar funções, campos e constantes que foram auditados e identificados como sem uso (*dead code*), reduzindo a complexidade cognitiva.
* **Prioridade:** MÉDIA.
* **Risco Geral:** **BAIXO**. Requer atenção em modificações de construtores/estruturas de dados.

### Tarefas da Sprint 2

* [x] **T2.1 — Remover funções `load_*_accums` de `src/models/wavenet/conv_input.rs`**
  * **Origem:** [B1 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L87)
  * **Ação:**
    * Remover as funções `load_4_accums` (linhas 12–39), `load_8_accums` (linhas 71–97) e `load_16_accums` (linhas 123–167).
    * Remover as anotações `#[allow(dead_code)]` and `#[allow(clippy::needless_range_loop)]` que as cercavam.
    * **Manter** intactas as funções irmãs `store_*_accums`, que são ativamente usadas.
  * **Validação:** `cargo check --tests`.

* [x] **T2.2 — Remover `weight_f16c_to_f64` de `src/testing/reference_oracle.rs`**
  * **Origem:** [B2 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L101)
  * **Ação:**
    * Remover a função `weight_f16c_to_f64` (linha 171) e seu respectivo `#[allow(dead_code)]`.
  * **Validação:** `cargo check --tests`.

* [x] **T2.3 — Remover campo `phase_type` de `src/dsp/resampler.rs`**
  * **Origem:** [B3 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L108)
  * **Ação:**
    * Remover o campo `phase_type` da struct `ResamplerCore` (linha 120).
    * Remover a linha correspondente de escrita/atribuição no construtor `ResamplerCore::new` (atribuição órfã).
    * Remover o getter público/interno `fn phase_type(&self)` (linha 151).
    * > [!IMPORTANT]
      > **Não** remover o enum `PhaseType` em si, pois ele pode ser exposto publicamente na API e é útil estruturalmente; remova apenas o campo interno da struct e seu getter.
  * **Validação:** `cargo check --tests` e executar `cargo test --lib dsp::resampler`.

* [x] **T2.4 — Remover `log_spaced_tones_dense` de `src/dsp/resampler_test.rs`**
  * **Origem:** [B4 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L119)
  * **Ação:**
    * Remover a função `log_spaced_tones_dense` (linha 687) e seu respectivo `#[allow(dead_code)]`.
  * **Validação:** `cargo check --tests`.

* [x] **T2.5 — Limpar constantes mortas em `src/testing/perceptual.rs`**
  * **Origem:** [B5 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L126)
  * **Ação:**
    * Remover as 5 constantes especificadas: `A2ESR_A1_STANDARD_Q1`, `A2ESR_A1_STANDARD_Q3`, `A2ESR_A2_FULL_Q1`, `A2ESR_A2_FULL_Q3` e `NAM_RS_CPP_PARITY_ESR_MAX`.
    * Executar uma compilação temporária sem o `#![allow(dead_code)]` do módulo para verificar se há mais algum helper inutilizado ou se novos avisos surgem. Caso surjam novos warnings legítimos de código que desejamos manter, reverter e restaurar o `#![allow(dead_code)]`.
  * **Validação:** `cargo check --tests`.

* [x] **T2.6 — Remover atributos `#[allow(...)]` stale em outros módulos**
  * **Origem:** [B6 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L141)
  * **Ação:**
    * Remover `#[allow(dead_code)]` de `max_frames_count` em `src/clap/processor/state.rs:122` (campo é lido ativamente em múltiplos lugares).
    * Remover `#[allow(dead_code)]` de `fn checked_add` em `src/loader/dispatcher/checked_arith.rs:37` (função é usada).
    * Remover `#[allow(unused_imports)]` de `parse_semver` em `src/loader/nam_json/topology/mod.rs:16` (importação é re-exportada e viva).
    * > [!WARNING]
      > **Não alterar** os quatro campos `os_*` com `#[allow(unused)]` em `src/dsp/pipeline/context.rs:87-97` por enquanto (risco de quebrar build de features específicas).
  * **Validação:** `cargo check --tests`.
  * > [!NOTE]
    > Somente `checked_add` em `checked_arith.rs` era realmente stale. Os outros dois atributos **não são stale** e foram mantidos:
    > * `#[allow(dead_code)]` em `max_frames_count` (state.rs) — o campo é atribuído mas nunca lido (reservado para futuras assertions RT-safety).
    > * `#[allow(unused_imports)]` em `parse_semver` (topology/mod.rs) — a reexportação `pub(crate)` é usada apenas via `nam_json/mod.rs:34`, não diretamente dentro do módulo `topology`.

---

## 📅 Sprint 3 — Achatamento de Diretórios `test_files/` (Epic 3)

* **Objetivo:** Desfazer o padrão não-padronizado de diretórios `test_files/` que violam a convenção de colocação de testes de Rust e tornam a estrutura confusa.
* **Prioridade:** MÉDIA-ALTA.
* **Risco Geral:** **BAIXO-MÉDIO** (cuidado extra com diagnósticos e caminhos de inclusão de módulos).

### Tarefas da Sprint 3

* [x] **T3.1 — Achatamento de `src/dsp/pipeline/test_files/`**
  * **Origem:** [D1a em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L348)
  * **Ação:**
    1. Mover os arquivos para `src/dsp/pipeline/` renomeando-os para evitar conflito de nomenclatura com outros módulos:
       * `test_files/bypass_test.rs` ➔ `pipeline_bypass_test.rs`
       * `test_files/gate_test.rs` ➔ `pipeline_gate_test.rs`
       * `test_files/dither_test.rs` ➔ `pipeline_dither_test.rs`
    2. Atualizar as diretivas `#[path]` em `src/dsp/pipeline/pipeline_test.rs` para apontar diretamente para os novos arquivos irmãos:
       * `#[path = "pipeline_bypass_test.rs"]`
       * `#[path = "pipeline_gate_test.rs"]`
       * `#[path = "pipeline_dither_test.rs"]`
    3. Certificar-se de que cada um dos arquivos movidos possua o cabeçalho SPDX/Copyright.
    4. Excluir o diretório vazio `test_files/`.
  * **Validação:** Compilar e rodar `cargo test --lib dsp::pipeline`.

* [x] **T3.2 — Achatamento de `src/models/wavenet/test_files/`**
  * **Origem:** [D1b em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L362)
  * **Ação:**
    1. Mover os 6 arquivos de teste para `src/models/wavenet/` como irmãos diretos de `wavenet_test.rs` (não há colisão de nomes com arquivos existentes):
       * `conv1d_dyn_test.rs`
       * `conv1d_test.rs`
       * `dense_test.rs`
       * `dynamic_parity_test.rs`
       * `post_stack_head_integration_test.rs`
       * `wavenet_sub_test.rs`
    2. Atualizar as diretivas `#[path]` correspondentes em `src/models/wavenet/wavenet_test.rs` para remover o prefixo `test_files/`:
       * Ex: `#[path = "conv1d_dyn_test.rs"]`
    3. **Triage & Mover `ch12_diagnostic.rs`:**
       * O arquivo `ch12_diagnostic.rs` contém testes unitários (`#[test] fn test_dense_ch12_scalar_vs_simd()` e `#[test] fn test_conv1d_ch12_scalar_vs_simd()`), mas atualmente **não** está sendo compilado/executado nas suites de teste padrão (estava órfão no diretório).
       * Mover para `src/models/wavenet/wavenet_ch12_diagnostic_test.rs`.
       * Adicionar sua declaração e inclusão no `wavenet_test.rs` para que esses testes sejam finalmente executados:

         ```rust
         #[path = "wavenet_ch12_diagnostic_test.rs"]
         mod wavenet_ch12_diagnostic_test;
         ```

    4. Garantir que todos os arquivos movidos e renomeados tenham os cabeçalhos SPDX/Copyright necessários.
    5. Excluir o diretório vazio `test_files/`.
  * **Validação:** Compilar e rodar a suíte de testes do wavenet: `cargo test --lib models::wavenet`.

---

## 📅 Sprint 4 — Splitting de Baixo Risco (Epic 4)

* **Objetivo:** Separar subcomponentes e arquivos grandes fora do caminho crítico de áudio em tempo real (RT) ou executados fora da thread de áudio.
* **Prioridade:** MÉDIA.
* **Risco Geral:** **ZERO** (sem impacto na geração de código hot-path ou thread RT).

### Tarefas da Sprint 4

* [x] **T4.1 — Split de `src/math/common/scalar_ref/dot.rs`**
  * **Origem:** [C1 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L180)
  * **Ação:**
    1. Criar o arquivo `src/math/common/scalar_ref/dot_4x.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as funções `dot_product_4x_*` (scalar, dual, interleaved, accumulate, dual_accumulate) para o novo arquivo.
    3. Criar o arquivo `src/math/common/scalar_ref/dot_8x16x.rs` (com cabeçalho SPDX/Copyright).
    4. Mover as funções `dot_product_8x_*` e `dot_product_16x_*` (scalar, dual, accumulate) para o novo arquivo.
    5. No arquivo `src/math/common/scalar_ref/dot.rs` original, manter as funções base/fallback (`dot_product_fallback`, `dot_product_bf16_fallback`, `dot_product_f32_native`, `dot_product_f32_native_kahan`, `dot_product_bf16_4x_fallback`), o módulo de teste `kahan_dot_tests` e exportar/re-exportar os novos arquivos:

       ```rust
       #[path = "dot_4x.rs"]
       pub mod dot_4x;
       #[path = "dot_8x16x.rs"]
       pub mod dot_8x16x;
       pub use dot_4x::*;
       pub use dot_8x16x::*;
       ```

  * **Validação:** Garantir compilação e executar `utils/tests-quick.sh`.

* [x] **T4.2 — Split de `src/models/static_model.rs`**
  * **Origem:** [C10 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L286)
  * **Ação:**
    1. Criar o arquivo `src/models/nam_model.rs` (com cabeçalho SPDX/Copyright).
    2. Mover a implementação `impl NamModel for StaticModel` de `src/models/static_model.rs` para o novo arquivo.
    3. No arquivo `src/models/static_model.rs`, manter métodos de classificação/consulta e a função `clone_condition_dsp`.
    4. Declarar o submódulo correspondente no escopo apropriado.
  * **Validação:** Garantir compilação e executar `utils/tests-quick.sh`.

* [x] **T4.3 — Split de `src/models/slimmable.rs`**
  * **Origem:** [C13 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L310)
  * **Ação:**
    1. Criar o arquivo `src/models/slicing.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as 7 funções auxiliares de fatiamento (`slice_*`, `clone_wavenet_*`, `try_slimmable_rebuild_single`) para o novo arquivo.
    3. Manter a definição do trait `SlimmableModel` e o módulo de teste inline associado no arquivo `src/models/slimmable.rs` original.
  * **Validação:** Garantir compilação e rodar testes de slimmable.

* [x] **T4.4 — Split de `src/clap/gui/ui/zones/identity.rs`**
  * **Origem:** [C11 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L293)
  * **Ação:**
    1. Criar o arquivo `src/clap/gui/ui/zones/file_dialogs.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as funções auxiliares `spawn_file_dialog` e `spawn_ir_file_dialog` para o novo arquivo.
    3. Manter a função principal de desenho `draw_zone1_identity` no arquivo `src/clap/gui/ui/zones/identity.rs`.
    4. Declarar o submódulo em `identity.rs` ou `zones/mod.rs`.
  * **Validação:** Compilar sob a feature `clap`.

* [x] **T4.5 — Renomear `src/math/gemm/dot.rs`**
  * **Origem:** [D2 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L379)
  * **Ação:**
    1. Renomear `src/math/gemm/dot.rs` para `src/math/gemm/dot_basic.rs`.
    2. Atualizar a declaração do submódulo em `src/math/gemm/mod.rs` de `pub mod dot;` para `pub mod dot_basic;` e ajustar quaisquer importações associadas.
  * **Validação:** Compilar e garantir integridade com testes rápidos.

---

## 📅 Sprint 5 — Splitting de Risco Médio (RT / SIMD / FFT) (Epic 5)

* **Objetivo:** Separar e organizar arquivos relacionados a processamento de áudio em tempo real (RT), kernels SIMD e processadores FFT.
* **Prioridade:** ALTA (desempenho crítico).
* **Risco Geral:** **MÉDIO**. Requer a medição rigorosa de performance em cada etapa para detectar possíveis regressões de codegen induzidas por decisões do compilador.
* **Portão de Benchmark:** Executar o benchmark `benches/inference_bench` antes e depois de cada tarefa e validar que a diferença de performance está na margem de ruído aceitável.

### Tarefas da Sprint 5

* [x] **T5.1 — Split de `src/models/a2/grouped_conv1d.rs`**
  * **Origem:** [C2 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L195)
  * **Ação:**
    1. Criar o subdiretório e arquivos correspondentes (ex: `src/models/a2/grouped_conv1d/simd.rs` e `src/models/a2/grouped_conv1d/reference.rs` com cabeçalhos SPDX/Copyright).
    2. Mover os kernels SIMD (`process_single_frame_depthwise_avx2`, `load_mixin_4`, `grouped_conv1d_single_frame_simd`) para `simd.rs`.
    3. Mover as implementações escalares de referência (`grouped_conv1d_single_frame_ref`, `grouped_conv1d_block_ref`) para `reference.rs`.
    4. Manter a struct `A2GroupedConv1d` e impls de alto nível em `src/models/a2/grouped_conv1d.rs`, declarando e re-exportando os novos submódulos.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

* [x] **T5.2 — Split de `src/models/a2/model/dynamic/process.rs`**
  * **Origem:** [C3 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L208)
  * **Ação:**
    1. Criar o arquivo `src/models/a2/model/dynamic/process_cascade.rs` (evitar colisão com `models/a2/model/cascade.rs`).
    2. Mover a família de métodos cascade (`cascade_layer_loop`, `cascade_head_finalize`, `cascade_write_mono_input`, `cascade_write_residual_input`, `cascade_set_condition`, `cascade_seed_head_from_output`) para o novo arquivo.
    3. Manter os métodos normais de processamento em `process.rs`.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

* [x] **T5.3 — Split de `src/dsp/resampler.rs`**
  * **Origem:** [C4 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L226) (após T2.3)
  * **Ação:**
    1. Criar `src/dsp/resampler/delay_line.rs` e `src/dsp/resampler/core.rs` (com cabeçalhos SPDX/Copyright).
    2. Mover a struct `DelayLine` e seus métodos para `delay_line.rs`.
    3. Mover a struct `ResamplerCore` e seus métodos para `core.rs`.
    4. Manter a struct `NamResampler` + impl e as diretivas de teste em `src/dsp/resampler.rs`.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

* [x] **T5.4 — Split de `src/math/gemm/gemv/f16_avx2_specialized.rs`**
  * **Origem:** [C5 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L237)
  * **Ação:**
    1. Criar `src/math/gemm/gemv/f16_avx2_fused.rs` (com cabeçalho SPDX/Copyright) e mover os 6 kernels `fused_add_gemv_avx2_*` para ele.
    2. Criar `src/math/gemm/gemv/f16_avx2_overwrite.rs` (com cabeçalho SPDX/Copyright) e mover os 6 kernels `gemv_overwrite_avx2_*` para ele.
    3. Manter os helpers privados `load_partial_ymm` e `store_partial_ymm` no arquivo principal e re-exportar os novos módulos.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

* [x] **T5.5 — Split de `src/math/activations/tanh/high_fidelity.rs`**
  * **Origem:** [C6 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L247)
  * **Ação:**
    1. Criar `src/math/activations/tanh/high_fidelity_avx512.rs` (com cabeçalho SPDX/Copyright).
    2. Mover todos os kernels AVX-512 para o novo arquivo.
    3. Manter kernels AVX2 + escalares e testes no arquivo original, expondo os novos com `pub use high_fidelity_avx512::*;`.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

* [x] **T5.6 — Split de `src/math/dsp/fft.rs`**
  * **Origem:** [C7 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L255)
  * **Ação:**
    1. Criar `src/math/dsp/rfft.rs` (com cabeçalho SPDX/Copyright).
    2. Mover a struct `RfftPlanner<T>` e sua implementação para o novo arquivo, importando o necessário do arquivo `fft.rs`.
    3. Manter a definição de `FftFloat` e `FftPlanner` em `fft.rs`.
  * **Validação:** Executar `utils/tests-quick.sh`.

* [ ] **T5.7 — Split de `src/dsp/oversample.rs`**
  * **Origem:** [C12 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L300)
  * **Ação:**
    1. Criar o arquivo `src/dsp/stage.rs` (com cabeçalho SPDX/Copyright) contendo a struct `X2Stage` e sua implementação.
    2. **Adicionar explicitamente as anotações `#[inline]` ou `#[inline(always)]` em `X2Stage::upsample` e `X2Stage::downsample` ao movê-las**, a fim de preservar o inlining do compilador no call site.
    3. Manter o restante em `oversample.rs`.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

* [ ] **T5.8 — Split de `src/models/linear.rs`**
  * **Origem:** [C8 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L264) (após T1.1)
  * **Ação:**
    1. Criar `src/models/linear/process.rs` (com cabeçalho SPDX/Copyright).
    2. Mover `process_sample`, `process`, `prewarm` e `reset` para `process.rs`.
    3. Manter a definição do modelo, construtores e trait `NamModel` em `linear.rs`.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

* [ ] **T5.9 — Split de `src/models/linear_fft.rs`**
  * **Origem:** [C9 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L277) (após T1.2)
  * **Ação:**
    1. Criar `src/models/linear_fft/process.rs` (com cabeçalho SPDX/Copyright).
    2. Mover as funções `reset` e `process_tail_block` para o novo arquivo.
    3. Manter o restante no arquivo principal.
  * **Validação:** Medir benchmark de inferência antes/depois; executar `utils/tests-quick.sh`.

---

## 📅 Sprint 6 — Reorganização Estrutural Opcional (Epic 6)

* **Objetivo:** Executar tarefas secundárias de consistência de estrutura e localização de testes com menor ganho imediato.
* **Prioridade:** BAIXA (opcional / se houver tempo).
* **Risco Geral:** **ZERO**.

### Tarefas da Sprint 6

* [ ] **T6.1 — Promover `src/models/a2/model/cascade.rs`**
  * **Origem:** [D4 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L397)
  * **Ação:** Mover `cascade.rs` para `src/models/a2/model/cascade/mod.rs` sem subdivisões adicionais, atualizando as declarações.
  * **Validação:** Garantir compilação e rodar testes rápidos.

* [ ] **T6.2 — Promover `src/math/activations/sigmoid.rs`**
  * **Origem:** [D5 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L406)
  * **Ação:** Promover para uma estrutura de diretório `sigmoid/mod.rs`, `sigmoid/production.rs` e `sigmoid/high_fidelity.rs`, mantendo a consistência com `tanh/`.
  * **Validação:** Garantir compilação e rodar testes rápidos.

* [ ] **T6.3 — Organizar testes de `conv1d_ch3` e `conv1d_ch8`**
  * **Origem:** [D6 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L415)
  * **Ação:** Mover `conv1d_ch3_test.rs` e `conv1d_ch8_test.rs` para dentro de seus respectivos subdiretórios `conv1d_ch3/` e `conv1d_ch8/`, ajustando as diretivas `#[path]`.
  * **Validação:** Rodar testes locais correspondentes.
