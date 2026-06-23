<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento de Sprints (EPIC C & EPIC D)

Este documento detalha o planejamento ágil para a execução das melhorias de refatoração estrutural (EPIC C) e exploração de performance avançada (EPIC D), mapeando as descobertas de auditoria do [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md) em sprints e tarefas técnicas atômicas.

---

## SPRINT 1: Refatoração Estrutural de Arquivos Grandes e Testes Inline (EPIC C)

**Objetivo:** Reduzir o acoplamento físico e tamanho dos arquivos do hot-path, separando testes inline e decompondo funções gigantescas de processamento sem alterar a lógica ou comportamento do áudio.
**Duração estimada:** 1 a 2 semanas.
**Risco:** Baixo-Médio (requer atenção para evitar regressão na legibilidade ou otimizações do compilador Rust).
**Critérios de Aceite:**

- Todos os testes de regressão locais e de validação do CLAP passarem com sucesso (`utils/tests-quick.sh`).
- Zero warnings ou erros de compilação produzidos pelo `cargo check` / `clippy`.
- Nenhum desvio de performance mensurável nos benchmarks de inferência.

- [x] **Task 1.1: Extração de testes inline para arquivos `_test.rs`**
  - Mover testes inline de arquivos de produção com tamanho substancial (>= 300 linhas de código útil antes do bloco de testes) para seus respectivos arquivos de teste separados no mesmo diretório (ex.: `post_stack_head_test.rs`, `convnet_model_test.rs`, `convnet_block_test.rs`, `a2_activations_test.rs`).
  - Adicionar a diretiva `#[cfg(test)] #[path = "..._test.rs"] mod ..._test;` no final do arquivo de produção correspondente.
  - Garantir a inclusão do cabeçalho de copyright SPDX em todos os novos arquivos de teste criados.
  - *Referência no `TODO-findings.md`: F-05*

- [x] **Task 1.2: Decomposição de `WaveNetA2::process()` em `src/models/a2/model/mod.rs`**
  - Refatorar a função de processamento de ~320 linhas em sub-funções menores marcadas com `#[inline(always)]` (ex.: `rechannel_prescale`, `advance_head_ring`, `layer_forward_dispatch`, `head_finalize`).
  - Assegurar que a elisão de bounds checking (`core::slice::GetUnchecked` ou truques de tamanho do compilador) permaneça ativa e que os limites de arrays permaneçam eficientes.
  - *Referência no `TODO-findings.md`: F-05*

- [ ] **Task 1.3: Decomposição de `WaveNetA2Dyn::process()` e `set_weights()` em `src/models/a2/model/dynamic.rs`**
  - Aplicar o mesmo princípio de decomposição da Task 1.2 para a versão dinâmica do A2.
  - Decompor a rotina complexa de atribuição de pesos (`set_weights()`).
  - *Referência no `TODO-findings.md`: F-05*

- [ ] **Task 1.4: Desmembrar `src/math/gemm/gemm_batch/avx2.rs`**
  - Separar os múltiplos kernels presentes no arquivo `avx2.rs` para arquivos modulares menores sob o diretório `gemm_batch/` (ex. `fused_add_gemm_batch.rs`, `fused_residual_batch.rs`).
  - Atualizar o `mod.rs` de `gemm_batch` para expô-los corretamente.
  - *Referência no `TODO-findings.md`: F-05*

---

## SPRINT 2: Desduplicação de Código e Refinamento de Comentários SAFETY (EPIC C)

**Objetivo:** Eliminar código redundante/duplicado entre os modelos estáticos e dinâmicos e melhorar a qualidade da documentação de segurança de blocos unsafe de baixo nível.
**Duração estimada:** 1 semana.
**Risco:** Baixo-Médio.
**Critérios de Aceite:**

- Paridade matemática intocada em relação aos golden vectors originais.
- Verificação estática sem falhas ou avisos (`utils/tests-quick.sh`).
- Cobertura de documentação `SAFETY` em conformidade com as regras do repositório.

- [ ] **Task 2.1: Unificar a projeção do cabeçalho do LSTM (head projection)**
  - Isolar a lógica duplicada de projeção (`dot_product`, bias, etc.) presente em `src/models/lstm/model1.rs` e `model2.rs` para um macro ou helper comum compartilhado.
  - *Referência no `TODO-findings.md`: F-06*

- [ ] **Task 2.2: Unificar rotinas de `prewarm` do WaveNetA2 e WaveNetA2Dyn**
  - Extrair a lógica idêntica de pre-aquecimento para uma função utilitária comum `a2_prewarm_common(...)` no local apropriado (por exemplo, sob `src/models/a2/model/`).
  - *Referência no `TODO-findings.md`: F-06*

- [ ] **Task 2.3: Unificar leitura e validação de bytes no build loader**
  - Extrair em `src/loader/build.rs` a validação comum de tamanho e formato para uma função `read_and_validate_model_bytes(...)` reduzindo a duplicação entre ler arquivos `.nam` e `.namb`.
  - *Referência no `TODO-findings.md`: F-06*

- [ ] **Task 2.4: Avaliar generalização const-generic para convoluções estáticas vs dinâmicas**
  - Analisar se a conv1d estática de WaveNet pode ser reescrita usando parâmetros constantes genéricos (`const`-generic) herdados da versão dinâmica sem prejuízos à elisão de bounds checking.
  - *Referência no `TODO-findings.md`: F-06*

- [ ] **Task 2.5: Substituir comentários `SAFETY` genéricos por descrições específicas**
  - Auditar `src/math/common/dispatch/config.rs` e subdiretórios `avx512/` para substituir comentários repetitivos/genéricos por invariantes e precondições específicas de segurança (ex. alinhamento de ponteiros, tamanhos de buffers).
  - *Referência no `TODO-findings.md`: F-10*

---

## SPRINT 3: Exploração de Performance Avant-garde (EPIC D)

**Objetivo:** Prototipar e validar a implementação de convoluções com maior largura de canais de saída por iteração, medindo os ganhos reais de vazão de processamento no hot path.
**Duração estimada:** 2 semanas.
**Risco:** Alto / Crítico (modificação profunda dos loops SIMD internos de inferência neural).
**Critérios de Aceite:**

- Obter ganhos reais de performance nos benchmarks locais de inferência sem introduzir regressão em nenhum hardware x86-64-v3+.
- Manter paridade numérica de áudio estrita (< 2 ULP) garantida por golden vectors e suíte C++ parity.

- [ ] **Task 3.1: Prototipar kernels SIMD de convolução estendidos**
  - Desenvolver o kernel `dot_product_8x_f32_avx2` processando 8 canais de saída simultâneos.
  - Desenvolver o kernel `dot_product_16x_f32_avx512` tirando proveito dos registradores amplos de 512 bits.
  - Focar no reuso persistente de acumuladores em registradores de vetor durante o laço interno de taps (facilitado pelo dispatch de alto nível monomorfizado).
  - *Referência no `TODO-findings.md`: F-09*

- [ ] **Task 3.2: Estender a suíte de micro-benchmarks**
  - Expandir o benchmark existente `benches/dot_4x_bench.rs` para abranger os novos cenários de processamento 8x e 16x de canais de saída.
  - Adicionar análises comparativas no `inference_bench.rs` medindo o tempo de inferência total do WaveNet e A2.
  - *Referência no `TODO-findings.md`: F-09*

- [ ] **Task 3.3: Integrar e testar kernels no pipeline de inferência**
  - Adaptar o WaveNet/A2 para selecionar dinamicamente os novos kernels em tempo de compilação através de dispatch monomorfizado (`SimdMath`).
  - Executar a bateria de testes matemáticos com golden vectors de áudio para assegurar que nenhuma distorção numérica seja inserida.
  - *Referência no `TODO-findings.md`: F-09*

- [ ] **Task 3.4: Decisão arquitetural de incorporação ou descarte**
  - Analisar os relatórios de benchmarks. Se o ganho de throughput de áudio for satisfatório e sem efeitos colaterais de estabilidade ou latência, mesclar permanentemente as otimizações. Caso contrário, descartar mantendo a infraestrutura de dispatch anterior limpa.
  - *Referência no `TODO-findings.md`: F-09*
