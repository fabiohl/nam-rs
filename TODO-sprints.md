<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints (NAM-rs)

Este documento contém o planejamento de sprints e tarefas técnicas estruturadas para o desenvolvimento do NAM-rs, garantindo paridade total com o NeuralAmpModelerCore v0.5.4.

---

## Sprint 1: Épico E — "Controle de Prewarm e LoadOptions" (F2, F3, F7)

**Escopo:** Implementar o flag `prewarm_on_reset` nos modelos, a estrutura `LoadOptions` no loader, e a propagação para Container/Slimmable/WaveNet com `condition_dsp`.
**Objetivo de Paridade:** Evitar processamentos desnecessários de prewarm durante resets frequentes (como mudanças de sample rate ou buffers no host DAW) e permitir o carregamento rápido de modelos (skip prewarm) off-thread para operações como preview de presets.
**Estimativa:** 1 sprint.
**Risco Geral:** 🟢 Baixo — Alterações estruturais simples de controle de fluxo de inicialização e propagação de flags, sem modificações nos hot-paths de inferência ou operações matemáticas de DSP.

### Quadro de Tarefas Técnicas

#### 1. [MODEL] Modificar o Trait `NamModel` para Suportar Prewarm Controlável (F2) [DONE]

- **Status:** `[ ]`
- **Arquivo Alvo:** [`src/models/mod.rs`](file:///home/fabio/nam-rs/src/models/mod.rs)
- **Descrição:**
  - Adicionar as seguintes assinaturas com implementação padrão no trait [`NamModel`](file:///home/fabio/nam-rs/src/models/mod.rs):

    ```rust
    fn prewarm_on_reset(&self) -> bool {
        true
    }
    fn set_prewarm_on_reset(&mut self, _val: bool) {}
    ```

  - Modificar o método `reset` padrão do trait [`NamModel`](file:///home/fabio/nam-rs/src/models/mod.rs):

    ```rust
    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        if self.prewarm_on_reset() {
            self.prewarm(max_buffer_size);
        }
        Ok(())
    }
    ```

- **Risco:** Baixo. Sem impacto na compatibilidade com implementações existentes.

#### 2. [MODEL] Adicionar Armazenamento do Flag `prewarm_on_reset` nos Modelos Concretos (F2) [DONE]

- **Status:** `[X]`
- **Arquivos Alvo:**
  - [`src/models/linear.rs`](file:///home/fabio/nam-rs/src/models/linear.rs)
  - [`src/models/convnet/model.rs`](file:///home/fabio/nam-rs/src/models/convnet/model.rs)
  - [`src/models/wavenet/model.rs`](file:///home/fabio/nam-rs/src/models/wavenet/model.rs)
  - [`src/models/wavenet/model_dyn.rs`](file:///home/fabio/nam-rs/src/models/wavenet/model_dyn.rs)
  - [`src/models/lstm/model1.rs`](file:///home/fabio/nam-rs/src/models/lstm/model1.rs)
  - [`src/models/lstm/model2.rs`](file:///home/fabio/nam-rs/src/models/lstm/model2.rs)
  - [`src/models/lstm/model_dyn.rs`](file:///home/fabio/nam-rs/src/models/lstm/model_dyn.rs)
  - [`src/models/a2/model/static/mod.rs`](file:///home/fabio/nam-rs/src/models/a2/model/static/mod.rs)
  - [`src/models/a2/model/dynamic/mod.rs`](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/mod.rs)
- **Descrição:**
  - Adicionar o campo `prewarm_on_reset: bool` a cada um dos structs de modelo concretos.
  - Inicializar `prewarm_on_reset: true` em todos os construtores associados (`new()`).
  - Sobrescrever os métodos `prewarm_on_reset` e `set_prewarm_on_reset` do trait `NamModel` para ler e escrever nesse campo.
- **Risco:** Baixo. Alterações puramente mecânicas de structs e construtores.

#### 3. [MODEL] Implementar Propagação do Flag em Modelos Compostos e Containers (F7)

- **Status:** `[ ]`
- **Arquivos Alvo:**
  - [`src/models/container.rs`](file:///home/fabio/nam-rs/src/models/container.rs)
  - [`src/models/slimmable.rs`](file:///home/fabio/nam-rs/src/models/slimmable.rs)
  - [`src/models/wavenet/model_dyn.rs`](file:///home/fabio/nam-rs/src/models/wavenet/model_dyn.rs)
- **Descrição:**
  - **ContainerModel:** Implementar `set_prewarm_on_reset` para salvar a flag localmente E iterar em `_submodels` propagando a chamada.
  - **SlimmableModel:** Implementar `set_prewarm_on_reset` para salvar localmente E propagar a chamada para todos os sub-modelos.
  - **WaveNetModelDyn:** Se `condition_dsp` estiver presente, propagar a chamada para o sub-modelo interno `condition_dsp`.
- **Risco:** 🟡 Médio — Requer atenção para garantir que a propagação ocorra recursivamente em qualquer nível de aninhamento.

#### 4. [MODEL] Atualizar a Implementação do Wrapper `StaticModel` (F2, F7)

- **Status:** `[ ]`
- **Arquivo Alvo:** [`src/models/static_model.rs`](file:///home/fabio/nam-rs/src/models/static_model.rs)
- **Descrição:**
  - Atualizar a implementação do trait `NamModel` para o enum [`StaticModel`](file:///home/fabio/nam-rs/src/models/static_model.rs) delegando `prewarm_on_reset()` e `set_prewarm_on_reset()` de forma transparente para a variante ativa correspondente.
- **Risco:** Baixo. Apenas despacho de padrão estático.

#### 5. [LOADER] Definir Estrutura `LoadOptions` e Integrar no Módulo Loader (F3)

- **Status:** `[ ]`
- **Arquivos Alvo:**
  - [`src/loader/mod.rs`](file:///home/fabio/nam-rs/src/loader/mod.rs)
  - [`src/loader/build.rs`](file:///home/fabio/nam-rs/src/loader/build.rs)
- **Descrição:**
  - Definir a estrutura pública `LoadOptions` em [`src/loader/mod.rs`](file:///home/fabio/nam-rs/src/loader/mod.rs):

    ```rust
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct LoadOptions {
        pub prewarm: Option<bool>, // None = default (true), Some(false) = skip
    }
    impl Default for LoadOptions {
        fn default() -> Self {
            Self { prewarm: None }
        }
    }
    ```

  - Alterar a assinatura da função `load_and_build_model` para aceitar `options: LoadOptions`.
  - Modificar a lógica interna em `load_and_build_model`:
    - Se `options.prewarm == Some(false)`:
      - Invocar `set_prewarm_on_reset(false)` nas instâncias construídas de `model_l` e `model_r`.
      - **Pular/Evitar** a chamada direta de `m.prewarm(...)` durante a construção no loader.
    - Se `options.prewarm` for `None` ou `Some(true)`:
      - Chamar `m.prewarm(m.prewarm_samples().max(2048))` no processo de build do loader.
- **Risco:** Baixo. Refatoração simples de assinatura pública.

#### 6. [PLUGINS / BINS] Atualizar Chamadas Existentes de `load_and_build_model` (F3)

- **Status:** `[ ]`
- **Arquivos Alvo:**
  - [`src/bin/pgo_profiling_workload.rs`](file:///home/fabio/nam-rs/src/bin/pgo_profiling_workload.rs)
  - [`src/main.rs`](file:///home/fabio/nam-rs/src/main.rs)
  - [`src/clap/processor_calibration_test.rs`](file:///home/fabio/nam-rs/src/clap/processor_calibration_test.rs)
  - [`src/clap/plugin/main_thread/load.rs`](file:///home/fabio/nam-rs/src/clap/plugin/main_thread/load.rs)
- **Descrição:**
  - Passar `LoadOptions::default()` nos pontos de chamada existentes do standalone, profiler e testes.
  - Garantir que o comportamento padrão não seja alterado de forma inesperada.
- **Risco:** Baixo. Mudança de aridade de função simples.

#### 7. [TEST] Criar Cobertura de Teste de Integração para Prewarm Opcional

- **Status:** `[ ]`
- **Arquivo Alvo:** [`tests/prewarm_test.rs`](file:///home/fabio/nam-rs/tests/prewarm_test.rs) (Novo arquivo de teste)
- **Descrição:**
  - Adicionar cobertura automatizada garantindo que:
    1. Um modelo carregado com `prewarm: Some(false)` zere/pule a execução do prewarm inicial.
    2. Resetar o modelo sem `prewarm_on_reset` não execute computações adicionais de prewarm.
    3. A propagação da flag em `ContainerModel` e `SlimmableModel` configure todos os sub-modelos aninhados.
  - Rodar o script `utils/tests-quick.sh` para verificar RT-safety e conformidade da build.
- **Risco:** Baixo.
