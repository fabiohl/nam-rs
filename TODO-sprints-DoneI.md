<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints (NAM-rs)

Este documento contém o planejamento de sprints e tarefas técnicas estruturadas para o desenvolvimento do NAM-rs, garantindo paridade total com o NeuralAmpModelerCore v0.5.4.

---

## Sprint 2: Épico I — "LSTM Prewarm por Sample Rate" (F8)

**Escopo:** Armazenar `expected_sample_rate` nos structs LSTM, implementar `prewarm_samples()` retornando `0.5s × SR`, e alinhar `reset()` com o comportamento do C++ (`Reset()` → `prewarm(GetPrewarmSamples())`).
**Objetivo de Paridade:** Estabilizar o estado recorrente do LSTM após resets e mudanças de sample rate (sound frequency), evitando artefatos audíveis (clicks, DC offsets) devido a uma inicialização sem pré-aquecimento.
**Estimativa:** 1 sprint.
**Risco Geral:** 🟢 Baixo — Mudanças bem localizadas nos modelos e construtores de LSTM, sem alterações nos kernels críticos de processamento.

---

### 1. [MODEL] Adicionar Campo `expected_sample_rate` nos Structs LSTM (F8) [DONE]

- **Status:** `[X]` **Concluído** — Campo `pub expected_sample_rate: f64` adicionado a `LstmModel1`, `LstmModel2` e `LstmModelDyn`. Inicializado com `48000.0` em `new()` e nos builders (`static_builder.rs`, `dynamic_builder.rs`). Task 4 propagará o valor real do JSON.
- **Arquivos Alvo:**
  - [`src/models/lstm/model1.rs`](file:///home/fabio/nam-rs/src/models/lstm/model1.rs)
  - [`src/models/lstm/model2.rs`](file:///home/fabio/nam-rs/src/models/lstm/model2.rs)
  - [`src/models/lstm/model_dyn.rs`](file:///home/fabio/nam-rs/src/models/lstm/model_dyn.rs)
- **Descrição:**
  - Adicionar o campo `pub expected_sample_rate: f64` nas structs [`LstmModel1`](file:///home/fabio/nam-rs/src/models/lstm/model1.rs), [`LstmModel2`](file:///home/fabio/nam-rs/src/models/lstm/model2.rs) e [`LstmModelDyn`](file:///home/fabio/nam-rs/src/models/lstm/model_dyn.rs).
  - Inicializar `expected_sample_rate: 48000.0` nos métodos `new()` associados para garantir um fallback seguro.
- **Risco:** Baixo. Alteração trivial de estrutura de dados.

---

### 2. [MODEL] Implementar `prewarm_samples` no Trait `NamModel` para LSTMs (F8) [DONE]

- **Status:** `[X]` **Concluído** — Método `prewarm_samples(&self) -> usize` sobrescrito nos blocos `impl NamModel` de `LstmModel1`, `LstmModel2` e `LstmModelDyn`. Retorna `(0.5 * expected_sample_rate) as usize`, com fallback seguro para `1` caso o cálculo resulte ≤ 0. 22 testes LSTM passando.
- **Arquivo Alvo:**
  - [`src/models/lstm/mod.rs`](file:///home/fabio/nam-rs/src/models/lstm/mod.rs)
- **Descrição:**
  - Sobrescrever o método `prewarm_samples(&self) -> usize` no bloco `impl NamModel` de [`LstmModel1`](file:///home/fabio/nam-rs/src/models/lstm/model1.rs), [`LstmModel2`](file:///home/fabio/nam-rs/src/models/lstm/model2.rs) e [`LstmModelDyn`](file:///home/fabio/nam-rs/src/models/lstm/model_dyn.rs).
  - Utilizar a lógica de paridade com o C++:

    ```rust
    fn prewarm_samples(&self) -> usize {
        let result = (0.5 * self.expected_sample_rate) as isize;
        if result <= 0 {
            1
        } else {
            result as usize
        }
    }
    ```

- **Risco:** Baixo. Sem impacto nos loops de áudio do hot-path.

---

### 3. [MODEL] Atualizar Lógica de `reset()` nos LSTMs para Executar Prewarm (F8) [DONE]

- **Status:** `[X]` **Concluído** — `reset()` nos três modelos LSTM (`LstmModel1`, `LstmModel2`, `LstmModelDyn`) agora executa `reset_states()` seguido de `prewarm(prewarm_samples())` quando `prewarm_on_reset() == true`, idêntico ao C++ (`Reset()` → `prewarm(GetPrewarmSamples())`). 22 testes LSTM passando.
- **Arquivo Alvo:**
  - [`src/models/lstm/mod.rs`](file:///home/fabio/nam-rs/src/models/lstm/mod.rs)
- **Descrição:**
  - Alterar o método `reset` nos blocos `impl NamModel` de [`LstmModel1`](file:///home/fabio/nam-rs/src/models/lstm/model1.rs), [`LstmModel2`](file:///home/fabio/nam-rs/src/models/lstm/model2.rs) e [`LstmModelDyn`](file:///home/fabio/nam-rs/src/models/lstm/model_dyn.rs):

    ```rust
    fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) -> anyhow::Result<()> {
        self.reset_states();
        if self.prewarm_on_reset() {
            self.prewarm(self.prewarm_samples());
        }
        Ok(())
    }
    ```

- **Risco:** Baixo. Garante a paridade do fluxo de inicialização do LSTM do C++.

---

### 4. [LOADER] Propagar `expected_sample_rate` nos Builders de LSTM (F8) [DONE]

- **Status:** `[X]` **Concluído** — `build_lstm_1layer`, `build_lstm_2layer` e `build_lstm_dynamic` agora extraem `sample_rate` de `NamModelData` (com fallback `DEFAULT_SAMPLE_RATE = 48000.0`) e propagam como `expected_sample_rate: f64` na construção das structs LSTM.
- **Arquivos Alvo:**
  - [`src/loader/dispatcher/lstm/static_builder.rs`](file:///home/fabio/nam-rs/src/loader/dispatcher/lstm/static_builder.rs)
  - [`src/loader/dispatcher/lstm/dynamic_builder.rs`](file:///home/fabio/nam-rs/src/loader/dispatcher/lstm/dynamic_builder.rs)
- **Descrição:**
  - Obter a `sample_rate` informada no JSON do modelo NAM:

    ```rust
    let sample_rate = data.sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE) as f64;
    ```

    *(Nota: importar `DEFAULT_SAMPLE_RATE` do loader se necessário).*
  - Passar essa sample rate durante a instanciação das structs de LSTM dentro de `build_lstm_1layer`, `build_lstm_2layer` e `build_lstm_dynamic`.
- **Risco:** Baixo. Alteração puramente mecânica na rotina de construção/despacho do loader.

---

### 5. [TEST] Criar Cobertura de Teste de Integração para Prewarm do LSTM (F8) [DONE]

- **Status:** `[X]` **Concluído** — 4 novos testes de integração adicionados em `tests/prewarm_test.rs`:
  - `test_lstm_prewarm_samples_scales_with_sample_rate`: valida que `prewarm_samples()` retorna 24000 (48kHz) e 22050 (44.1kHz) para LSTM estático (H=3) e dinâmico (H=7), e que o fallback `DEFAULT_SAMPLE_RATE` (48000) é aplicado quando `sample_rate` é `None`.
  - `test_lstm_prewarm_samples_minimum`: edge case com `sample_rate = 1.0` → `prewarm_samples()` floor = 1.
  - `test_lstm_reset_deterministic_after_prewarm`: dois modelos LSTM com estados internos divergentes → reset com `prewarm_on_reset=true` → outputs deterministicamente idênticos (tolerância 1e-6).
  - `test_lstm_reset_differs_prewarm_vs_noprewarm`: valida que reset com e sem prewarm produzem outputs diferentes (LSTM), complementando o teste Linear já existente.
- **Suite completa:** `utils/tests-quick.sh` executada — todas as 5 fases passaram (incluindo clap-validator).
