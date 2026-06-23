<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Implementação

> **Data:** 2026-06-23
> **Origem dos achados:** [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md)
> **Foco:** EPIC B — Conformidade x86-64-v3 e Limpeza de Portabilidade
> **Skills Associadas:** `planejador-arquiteto`, `implementador`, `refatora-rust`, `revisor-auditor`

---

## 0. Visão Geral do Épico e Gestão de Riscos

O **EPIC B** foca na consolidação definitiva da arquitetura `x86-64-v3` do projeto, eliminando fallbacks portáveis mortos, removendo o uso proibido de condicionais dinâmicos para features garantidas (AVX2/FMA) em testes, e reposicionando ou isolando funções auxiliares que servem estritamente ao ambiente de testes.

### Matriz de Riscos

| Identificador | Risco                                                            | Severidade | Ação de Mitigação                                                                                                                                  |
|:------------- |:---------------------------------------------------------------- |:---------- |:-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R-1**       | Erros de compilação ou linkagem em código condicional modificado | Média      | Executar compilação incremental (`cargo check` e `cargo build`) em todas as configurações de features (`standalone`, `clap-plugin`).               |
| **R-2**       | Quebra de regressão de testes unitários ou integração            | Alta       | Rodar a suíte rápida via `utils/tests-quick.sh` após cada tarefa.                                                                                  |
| **R-3**       | Perda de suporte involuntária a builds x86_64 padrão (não-v3)    | Baixa      | A conformidade com `x86-64-v3` é um requisito incondicional do projeto. A adição do `compile_error!` apenas formaliza isso em tempo de compilação. |

---

## Histórico de Execuções (EPIC A)

## Sprint 1 — Monomorfização da Convolução f32 e Eliminação de Dispatch por Chamada (F-01) [DONE]

**Objetivo:** Eliminar branches e funções de detecção de CPU (`is_x86_feature_detected!`) do hot-path interno, propagando o parâmetro de tipo genérico `M: SimdMath` até as folhas da convolução.

### Tarefas Técnicas da Sprint 1

- [x] **T1.1: Adicionar os novos métodos f32 ao trait `SimdMath`**

  - **Descrição:** Declarar os métodos de multiplicação de acumuladores f32 diretamente no trait comum de SIMD.
  - **Arquivos envolvidos:**
    - [traits.rs](file:///home/fabio/nam-rs/src/math/common/traits.rs)
    - [avx2_impl.rs](file:///home/fabio/nam-rs/src/math/common/avx2_impl.rs)
    - [base.rs](file:///home/fabio/nam-rs/src/math/common/avx512/gemv/base.rs)
    - [vnni_bf16.rs](file:///home/fabio/nam-rs/src/math/common/avx512/gemv/vnni_bf16.rs)
  - **Responsável Sugerido:** `implementador`
  - **Risco:** Baixo

- [x] **T1.2: Propagar o parâmetro `<M: SimdMath>` nas estruturas de Convolução**

  - **Descrição:** Tornar as implementações de convolução que consomem `dot_product_4x` e `dot_product_4x_dual` genéricas em `M` e chamar as implementações do trait `M::dot_product_4x_f32` diretamente.
  - **Arquivos envolvidos:**
    - [conv1d.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d.rs)
    - [conv1d_dyn.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d_dyn.rs)
    - [conv1d_dual.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d_dual.rs)
    - [layer.rs](file:///home/fabio/nam-rs/src/models/wavenet/layer.rs)
  - **Responsável Sugerido:** `implementador` / `pesquisador-inovador`
  - **Risco:** Médio

- [x] **T1.3: Limpeza de wrappers legados em `conv_input.rs`**

  - **Descrição:** Remover as funções livres `dot_product_4x` e `dot_product_4x_dual` que usavam `is_x86_feature_detected!`.
  - **Arquivos envolvidos:**
    - [conv_input.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv_input.rs)
  - **Responsável Sugerido:** `implementador`
  - **Risco:** Baixo

---

## Sprint 2 — Eliminação do Dispatch Dinâmico Duplo (V-Table `SimdMathConfig`) (F-02) [DONE]

**Objetivo:** Eliminar o overhead de chamadas indiretas via ponteiro de função (`Mode 3` da macro `dispatch_simd!`) unificando os consumidores do pipeline DSP e ativações no mecanismo estático monomorfizado da `SimdMath`.

### Tarefas Técnicas da Sprint 2

- [x] **T2.1: Migrar estágios do pipeline DSP para monomorfização estática**

  - **Descrição:** Envolver a execução do pipeline de captura em um bloco `dispatch_simd!` de Modo 1 (estático) no nível mais alto, passando `M` genérico para as funções de estágio (`apply_input_stage`, `apply_output_stage`, etc.).
  - **Arquivos envolvidos:**
    - [capture.rs](file:///home/fabio/nam-rs/src/dsp/pipeline/capture.rs)
    - [input.rs](file:///home/fabio/nam-rs/src/dsp/pipeline/stages/input.rs)
    - [output.rs](file:///home/fabio/nam-rs/src/dsp/pipeline/stages/output.rs)
  - **Responsável Sugerido:** `planejador-arquiteto` / `implementador`
  - **Risco:** Alto

- [x] **T2.2: Reduzir a struct `SimdMathConfig` e remover ponteiros de função**

  - **Descrição:** Remover os ponteiros de função associados à v-table em `SimdMathConfig`, deixando apenas dados estáticos descritivos (conjunto de instruções, nome, etc.).
  - **Arquivos envolvidos:**
    - [config.rs](file:///home/fabio/nam-rs/src/math/common/dispatch/config.rs)
    - [detect.rs](file:///home/fabio/nam-rs/src/math/common/dispatch/detect.rs)
    - [mod.rs](file:///home/fabio/nam-rs/src/math/common/mod.rs)
  - **Responsável Sugerido:** `implementador`
  - **Risco:** Médio

---

## Sprint 3 — Ajuste de Documentação e Comentários (F-08) [DONE]

**Objetivo:** Sincronizar todos os comentários técnicos da base de código e documentação arquitetural com a nova infraestrutura de dispatch unificada.

### Tarefas Técnicas da Sprint 3

- [x] **T3.1: Atualizar comentários do código fonte**

  - **Descrição:** Corrigir comentários e docstrings que descreviam o comportamento antigo (como o dispatch dinâmico por chamada ou a viabilidade de performance na v-table).
  - **Arquivos envolvidos:**
    - [model.rs](file:///home/fabio/nam-rs/src/models/wavenet/model.rs)
    - [config.rs](file:///home/fabio/nam-rs/src/math/common/dispatch/config.rs)
    - [conv_input.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv_input.rs)
  - **Responsável Sugerido:** `documentador` / `refatora-doc`
  - **Risco:** Mínimo

- [x] **T3.2: Atualizar documentação arquitetural global**

  - **Descrição:** Refletir o design final unificado e o fim do double-dispatch na documentação de design do projeto.
  - **Arquivos envolvidos:**
    - [architecture.md](file:///home/fabio/nam-rs/docs/architecture.md)
  - **Responsável Sugerido:** `documentador`
  - **Risco:** Mínimo

---

## Planejamento Ativo (EPIC B)

## Sprint 4 — Conformidade de Testes e Remoção de CPU Detection Redundante (F-04) [DONE]

**Objetivo:** Eliminar guards dinâmicos de detecção de CPU para features que são garantidas incondicionalmente pelo baseline mínimo do compilador (`x86-64-v3`), garantindo conformidade com a regra `rust.md §3`.

### Tarefas Técnicas da Sprint 4

- [x] **T4.1: Remover guards `is_x86_feature_detected!` para AVX2 e FMA em testes**
  - **Descrição:** Remover o bloco condicional `if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") { return Ok(()); }` do teste proptest em `activations_test.rs`, pois o baseline do compilador já garante essas instruções de forma incondicional.
  - **Arquivos envolvidos:**
    - [activations_test.rs](file:///home/fabio/nam-rs/src/math/activations/activations_test.rs#L208)
  - **Responsável Sugerido:** `implementador`
  - **Risco:** Mínimo

### Critério de Aceite da Sprint 4

- Testes unitários de ativações compilam e passam com sucesso via `cargo test --lib math::activations`.

---

## Sprint 5 — Eliminação de Fallbacks Portáveis e Guards Redundantes (F-03) [TODO]

**Objetivo:** Limpar a base de código de todos os ramos inalcançáveis `#[cfg(not(target_arch = "x86_64"))]` e simplificar os guards de arquitetura redundantes, visto que a aplicação é puramente voltada para `x86_64-v3`. Adicionar um guard centralizado de compilação.

### Tarefas Técnicas da Sprint 5

- [ ] **T5.1: Adicionar compile_error! centralizado em `lib.rs`**

  - **Descrição:** Introduzir uma diretiva condicional de compilação no arquivo raiz para falhar explicitamente o build caso a arquitetura de destino não seja `x86_64`.
  - **Arquivos envolvidos:**
    - [lib.rs](file:///home/fabio/nam-rs/src/lib.rs#L4)
  - **Responsável Sugerido:** `implementador` / `planejador-arquiteto`
  - **Risco:** Baixo

- [ ] **T5.2: Remover ramificações `#[cfg(not(target_arch = "x86_64"))]` e guards redundantes**

  - **Descrição:** Eliminar blocos condicionais de fallback escalar para outras arquiteturas e remover guards desnecessários de `target_arch = "x86_64"` em funções que já rodam em arquivos/módulos exclusivos.
  - **Arquivos envolvidos:**
    - [detect.rs](file:///home/fabio/nam-rs/src/math/common/dispatch/detect.rs#L24) (Tornar o corpo de `detect_best_simd` incondicional ou limpo)
    - [conv.rs](file:///home/fabio/nam-rs/src/dsp/cabsim/conv.rs#L338) (Remover o bloco de fallback escalar UPOLS não-x86_64)
    - [telemetry.rs](file:///home/fabio/nam-rs/src/clap/processor/dsp/telemetry.rs#L16) (Remover struct vazia de fallback)
    - [mod.rs](file:///home/fabio/nam-rs/src/models/a2/model/mod.rs#L317) (Remover branch escalar do rechannel do A2)
    - [processor/mod.rs](file:///home/fabio/nam-rs/src/clap/processor/mod.rs#L252) (Remover fallback no processador CLAP)
    - [batch_norm.rs](file:///home/fabio/nam-rs/src/models/convnet/batch_norm.rs#L123-L210) (Remover blocos escalares de normalização)
    - [half.rs](file:///home/fabio/nam-rs/src/math/common/half.rs#L148-L173) (Remover funções escalares de conversão de f16/bf16)
  - **Responsável Sugerido:** `refatora-rust` / `implementador`
  - **Risco:** Médio

### Critério de Aceite da Sprint 5

- Compilação limpa em todas as features (`standalone`, `clap-plugin`).
- Suíte unificada de testes (`utils/tests-quick.sh`) passa completamente sem regressões.

---

## Sprint 6 — Isolamento de Helpers e Código de Teste Causal (F-07) [TODO]

**Objetivo:** Evitar poluição no código de produção principal movendo ou restringindo estritamente a escopos de teste (`#[cfg(test)]`) os helpers e kernels de convolução que não são usados no caminho de áudio real.

### Tarefas Técnicas da Sprint 6

- [ ] **T6.1: Restringir helper `init_accum_with_bias_mixin` a testes**

  - **Descrição:** Substituir o atributo genérico de `dead_code` por uma restrição direta de teste (`#[cfg(test)]`), já que a função serve estritamente aos testes causais unitários.
  - **Arquivos envolvidos:**
    - [conv_input.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv_input.rs#L14)
  - **Responsável Sugerido:** `implementador`
  - **Risco:** Baixo

- [ ] **T6.2: Restringir métodos de convolução legados em `conv1d.rs`**

  - **Descrição:** Mover os métodos `process_single_frame` e `process_block` de `Conv1d` para dentro de uma estrutura condicional `#[cfg(test)]` (ou migrá-los se aplicável), impedindo que façam parte do build final de produção do WaveNet (que utiliza `process_single_frame_with_mixin`).
  - **Arquivos envolvidos:**
    - [conv1d.rs](file:///home/fabio/nam-rs/src/models/wavenet/conv1d.rs#L130-L208)
  - **Responsável Sugerido:** `refatora-rust` / `implementador`
  - **Risco:** Baixo

### Critério de Aceite da Sprint 6

- O código compila sem warnings de dead_code.
- Nenhum teste causal unitário é quebrado ou removido de forma incorreta.
