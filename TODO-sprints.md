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

* [ ] **T2.2 — Remover `weight_f16c_to_f64` de `src/testing/reference_oracle.rs`**
  * **Origem:** [B2 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L101)
  * **Ação:**
    * Remover a função `weight_f16c_to_f64` (linha 171) e seu respectivo `#[allow(dead_code)]`.
  * **Validação:** `cargo check --tests`.

* [ ] **T2.3 — Remover campo `phase_type` de `src/dsp/resampler.rs`**
  * **Origem:** [B3 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L108)
  * **Ação:**
    * Remover o campo `phase_type` da struct `ResamplerCore` (linha 120).
    * Remover a linha correspondente de escrita/atribuição no construtor `ResamplerCore::new` (atribuição órfã).
    * Remover o getter público/interno `fn phase_type(&self)` (linha 151).
    * > [!IMPORTANT]
      > **Não** remover o enum `PhaseType` em si, pois ele pode ser exposto publicamente na API e é útil estruturalmente; remova apenas o campo interno da struct e seu getter.
  * **Validação:** `cargo check --tests` e executar `cargo test --lib dsp::resampler`.

* [ ] **T2.4 — Remover `log_spaced_tones_dense` de `src/dsp/resampler_test.rs`**
  * **Origem:** [B4 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L119)
  * **Ação:**
    * Remover a função `log_spaced_tones_dense` (linha 687) e seu respectivo `#[allow(dead_code)]`.
  * **Validação:** `cargo check --tests`.

* [ ] **T2.5 — Limpar constantes mortas em `src/testing/perceptual.rs`**
  * **Origem:** [B5 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L126)
  * **Ação:**
    * Remover as 5 constantes especificadas: `A2ESR_A1_STANDARD_Q1`, `A2ESR_A1_STANDARD_Q3`, `A2ESR_A2_FULL_Q1`, `A2ESR_A2_FULL_Q3` e `NAM_RS_CPP_PARITY_ESR_MAX`.
    * Executar uma compilação temporária sem o `#![allow(dead_code)]` do módulo para verificar se há mais algum helper inutilizado ou se novos avisos surgem. Caso surjam novos warnings legítimos de código que desejamos manter, reverter e restaurar o `#![allow(dead_code)]`.
  * **Validação:** `cargo check --tests`.

* [ ] **T2.6 — Remover atributos `#[allow(...)]` stale em outros módulos**
  * **Origem:** [B6 em TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md#L141)
  * **Ação:**
    * Remover `#[allow(dead_code)]` de `max_frames_count` em `src/clap/processor/state.rs:122` (campo é lido ativamente em múltiplos lugares).
    * Remover `#[allow(dead_code)]` de `fn checked_add` em `src/loader/dispatcher/checked_arith.rs:37` (função é usada).
    * Remover `#[allow(unused_imports)]` de `parse_semver` em `src/loader/nam_json/topology/mod.rs:16` (importação é re-exportada e viva).
    * > [!WARNING]
      > **Não alterar** os quatro campos `os_*` com `#[allow(unused)]` em `src/dsp/pipeline/context.rs:87-97` por enquanto (risco de quebrar build de features específicas).
  * **Validação:** `cargo check --tests`.

---

## 📅 Sprint 3 — Achatamento de Diretórios `test_files/` (Epic 3)

* **Objetivo:** Desfazer o padrão não-padronizado de diretórios `test_files/` que violam a convenção de colocação de testes de Rust e tornam a estrutura confusa.
* **Prioridade:** MÉDIA-ALTA.
* **Risco Geral:** **BAIXO-MÉDIO** (cuidado extra com diagnósticos e caminhos de inclusão de módulos).

### Tarefas da Sprint 3

* [ ] **T3.1 — Achatamento de `src/dsp/pipeline/test_files/`**
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

* [ ] **T3.2 — Achatamento de `src/models/wavenet/test_files/`**
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
