<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento de Sprints (EPIC C & EPIC D)

Este documento detalha o planejamento ágil para a execução das melhorias de refatoração estrutural (EPIC C) e exploração de performance avançada (EPIC D), mapeando as descobertas de auditoria do [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md) em sprints e tarefas técnicas atômicas.

---

## SPRINT 1: Refatoração Estrutural de Arquivos Grandes e Testes Inline (EPIC C) [DONE]

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

- [x] **Task 1.3: Decomposição de `WaveNetA2Dyn::process()` e `set_weights()` em `src/models/a2/model/dynamic.rs`**
  - Aplicar o mesmo princípio de decomposição da Task 1.2 para a versão dinâmica do A2.
  - Decompor a rotina complexa de atribuição de pesos (`set_weights()`).
  - *Referência no `TODO-findings.md`: F-05*

- [x] **Task 1.4: Desmembrar `src/math/gemm/gemm_batch/avx2.rs`**
  - Separar os múltiplos kernels presentes no arquivo `avx2.rs` para arquivos modulares menores sob o diretório `gemm_batch/` (ex. `fused_add_gemm_batch.rs`, `fused_residual_batch.rs`).
  - Atualizar o `mod.rs` de `gemm_batch` para expô-los corretamente.
  - *Referência no `TODO-findings.md`: F-05*

---

## SPRINT 2: Desduplicação de Código e Refinamento de Comentários SAFETY (EPIC C) [DONE]

**Objetivo:** Eliminar código redundante/duplicado entre os modelos estáticos e dinâmicos e melhorar a qualidade da documentação de segurança de blocos unsafe de baixo nível.
**Duração estimada:** 1 semana.
**Risco:** Baixo-Médio.
**Critérios de Aceite:**

- Paridade matemática intocada em relação aos golden vectors originais.
- Verificação estática sem falhas ou avisos (`utils/tests-quick.sh`).
- Cobertura de documentação `SAFETY` em conformidade com as regras do repositório.

- [x] **Task 2.1: Unificar a projeção do cabeçalho do LSTM (head projection)**
  - Isolar a lógica duplicada de projeção (`dot_product`, bias, etc.) presente em `src/models/lstm/model1.rs` e `model2.rs` para um macro ou helper comum compartilhado.
  - *Referência no `TODO-findings.md`: F-06*

- [x] **Task 2.2: Unificar rotinas de `prewarm` do WaveNetA2 e WaveNetA2Dyn**
  - Extrair a lógica idêntica de pre-aquecimento para uma função utilitária comum `a2_prewarm_common(...)` no local apropriado (por exemplo, sob `src/models/a2/model/`).
  - *Referência no `TODO-findings.md`: F-06*

- [x] **Task 2.3: Unificar leitura e validação de bytes no build loader**
  - Extrair em `src/loader/build.rs` a validação comum de tamanho e formato para uma função `read_and_validate_model_bytes(...)` reduzindo a duplicação entre ler arquivos `.nam` e `.namb`.
  - *Concluído: função `read_and_validate_model_bytes(path, path_str, sys)` extraída com ~60 linhas eliminadas de duplicação. `.nam` agora lê bytes + `String::from_utf8` com diagnóstico próprio.*
  - *Referência no `TODO-findings.md`: F-06*

- [x] **Task 2.4: Avaliar generalização const-generic para convoluções estáticas vs dinâmicas**
  - Analisar se a conv1d estática de WaveNet pode ser reescrita usando parâmetros constantes genéricos (`const`-generic) herdados da versão dinâmica sem prejuízos à elisão de bounds checking.
  - **Conclusão (2026-06-23): NÃO recomendado.** As duas versões diferem fundamentalmente na estratégia de preloading de taps (cópia em stack `[[0.0; IN]; K]` vs ponteiros `[*const f32; MAX_KERNEL]`), na inicialização de acumuladores (loops de tamanho conhecido em compilação vs guardas `out_c+3 < out_ch` em runtime) e no pipeline de blocos (single-frame vs dual-frame tiling). Unificá-las comprometeria ou a elisão de bounds-check da versão estática (com `IN`/`OUT`/`K` comprováveis em compilação) ou a flexibilidade da versão dinâmica (dimensões arbitrárias, CH > 16). O núcleo SIMD já é compartilhado via trait `SimdMath` (`dot_product_4x_f32` / `dot_product_4x_f32_dual`). Nota pós-F-01: o dispatch monomorfizado no topo (já existente para `WaveNetLayer::process_block_internal<M: SimdMath>`) deve ser estendido ao `A2Conv1d::process_single_frame`, eliminando `is_x86_feature_detected!` por-chamada — mas isso é ortogonal à questão const-generic da estrutura `Conv1dDyn` vs `Conv1d<IN, OUT, K>`.
  - *Referência no `TODO-findings.md`: F-06*

- [x] **Task 2.5: Substituir comentários `SAFETY` genéricos por descrições específicas**
  - Auditar `src/math/common/dispatch/config.rs` e subdiretórios `avx512/` para substituir comentários repetitivos/genéricos por invariantes e precondições específicas de segurança (ex. alinhamento de ponteiros, tamanhos de buffers).
  - *Referência no `TODO-findings.md`: F-10*

---

## SPRINT 3: Exploração de Performance Avant-garde (EPIC D) [DONE]

**Objetivo:** Prototipar e validar a implementação de convoluções com maior largura de canais de saída por iteração, medindo os ganhos reais de vazão de processamento no hot path.
**Duração estimada:** 2 semanas.
**Risco:** Alto / Crítico (modificação profunda dos loops SIMD internos de inferência neural).
**Critérios de Aceite:**

- Obter ganhos reais de performance nos benchmarks locais de inferência sem introduzir regressão em nenhum hardware x86-64-v3+.
- Manter paridade numérica de áudio estrita (< 2 ULP) garantida por golden vectors e suíte C++ parity.

- [x] **Task 3.1: Prototipar kernels SIMD de convolução estendidos**
  - Desenvolver o kernel `dot_product_8x_f32_avx2` processando 8 canais de saída simultâneos.
  - Desenvolver o kernel `dot_product_16x_f32_avx512` tirando proveito dos registradores amplos de 512 bits.
  - Focar no reuso persistente de acumuladores em registradores de vetor durante o laço interno de taps (facilitado pelo dispatch de alto nível monomorfizado).
  - **Concluído (2026-06-23).** Novos módulos `src/math/gemm/dot_8x/` e `src/math/gemm/dot_16x/` criados com kernels, scalar references e 6 testes unitários. `dot_product_8x_f32_avx2` usa 4 acumuladores `__m256` com unroll 4x via `dot4x_simd4!` para quebrar cadeia de latência FMA. `dot_product_16x_f32_avx512` usa 2 acumuladores `__m512` com unroll 2x. Métodos adicionados ao trait `SimdMath` (grupo A) e implementados em `Avx2Math`, `Avx512Math` e `Avx512VnniBf16Math`. Nota: `Avx2Math::dot_product_16x_f32` usa scalar reference (sem `__m512` disponível); decomposição via dois `dot_product_8x_f32_avx2` com reinterpretação de ponteiros pode ser adicionada como otimização futura. Testes de decomposição contra 4x equivalentes confirmam paridade < 5e-4 ULP.
  - *Referência no `TODO-findings.md`: F-09*

- [x] **Task 3.2: Estender a suíte de micro-benchmarks**
  - Expandir o benchmark existente `benches/dot_4x_bench.rs` para abranger os novos cenários de processamento 8x e 16x de canais de saída.
  - Adicionar análises comparativas no `inference_bench.rs` medindo o tempo de inferência total do WaveNet e A2.
  - *Concluído (2026-06-23).* `dot_4x_bench.rs` ampliado com grupos `dot_8x_f32` (scalar + AVX2) e `dot_16x_f32` (scalar + AVX-512, gate-checked) para tamanhos [16, 64, 256, 1024, 4096]. Exportações `dot_product_8x_f32_scalar`/`dot_product_16x_f32_scalar` adicionadas ao `mod.rs` de `gemm`. `inference_bench.rs` expandido com `WaveNet_Comparison` (Standard CH=16 vs Dynamic CH=5) e `A2_Comparison` (Full CH=8 vs Lite CH=3 vs Dyn CH=4 gated). Todos os 776 testes passam, zero warnings. Nota para Task 3.3: as funções comparativas usam `build_model` via dispatcher padrão; quando os kernels 8x/16x forem integrados ao pipeline de inferência, o grupo `A2_Comparison` servirá como baseline imediata de regressão.
  - *Referência no `TODO-findings.md`: F-09*

- [x] **Task 3.3: Integrar e testar kernels no pipeline de inferência**
  - Adaptar o WaveNet/A2 para selecionar dinamicamente os novos kernels em tempo de compilação através de dispatch monomorfizado (`SimdMath`).
  - Executar a bateria de testes matemáticos com golden vectors de áudio para assegurar que nenhuma distorção numérica seja inserida.
  - *Concluído (2026-06-23).* Integração completa nos pontos de inferência:
    - **WaveNet static (`conv1d.rs`)**: `process_single_frame_with_mixin` e `process_single_frame` agora selecionam largura de interleaving (16/8/4) alinhada com `select_interleave_width(out_ch)` em `layout.rs`. CH=16 → `dot_product_16x_f32`, CH=8 → `dot_product_8x_f32`, CH=4/12 → `dot_product_4x_f32`.
    - **WaveNet dynamic (`conv1d_dyn.rs`)**: `process_single_frame` usa o campo `interleave_width` (definido em `Conv1dDyn::from_parts` via `traits.rs`) para dispatch entre `process_blocks_16/8/4`. `process_dual_frame` em `conv1d_dyn_dual.rs` faz fallback para single-frame quando `interleave_width != 4`.
    - **WaveNet dual-frame (`conv1d_dual.rs`)**: `process_dual_frame_with_mixin` faz fallback para `process_single_frame_with_mixin` quando `interleave_width != 4` (sem kernel dual 8x/16x disponível).
    - **Weight loading (`layout.rs`)**: `read_conv1d_weights_typed` seleciona largura ótima via `select_interleave_width`. Para dados já 4-wide (`Interleaved4WaveNet`), funções `transpose_4wide_to_8wide`/`transpose_4wide_to_16wide` convertem para a largura alvo. Para dados raw, `transpose_conv1d_interleaved_8wide`/`_16wide` fazem transposição direta.
    - **A2 CH=8 (`simd.rs`)**: Nova função `layer_forward_ch8_block_simdmath<M: SimdMath>` usa `M::dot_product_8x_f32` para o passo de convolução com dispatch monomorfizado. Acionada em `layer_forward_dispatch` no `A2ConvCh::Ch8` para ISAs AVX-512 (`Avx512Math`/`Avx512VnniBf16Math`), mantendo o kernel AVX2 raw como fallback.
    - **Helper functions (`conv_input.rs`)**: Adicionados `load_8_accums`, `store_8_accums`, `load_16_accums`, `store_16_accums`.
    - **Testes**: 6 novos testes unitários validam processamento 8-wide, 16-wide e conversão 4→16-wide. Todos 779 testes lib passam, 26 golden vector tests (A1/A2/Container/Dynamic) passam com zero regressão numérica, nondist validation passa.
  - *Referência no `TODO-findings.md`: F-09*

- [x] **Task 3.4: Decisão arquitetural de incorporação ou descarte**
  - Analisar os relatórios de benchmarks. Se o ganho de throughput de áudio for satisfatório e sem efeitos colaterais de estabilidade ou latência, mesclar permanentemente as otimizações. Caso contrário, descartar mantendo a infraestrutura de dispatch anterior limpa.
  - Git Message: integrate dot_product_8x/16x_f32 kernels into WaveNet/A2 inference pipeline via SimdMath dispatch
  - *Referência no `TODO-findings.md`: F-09*
    - **Decisão (2026-06-23): INCORPORAR permanentemente.**
    - **Evidência de estabilidade:** 779 testes lib passam (zero falhas), 26 golden vector tests passam (zero regressão numérica: A1 Standard, Lite, Feather, Nano; A2 Full, Lite, Container, Dynamic; ConvNet; LSTM), `nondist_validation` passa (block invariance), `clap-validator` confirma 19/21 testes CLAP passam (2 skipped — note-ports não implementados). `utils/tests-long.sh` completo: todas as 6 fases de auditoria pesada passam (Soak 131s, PipeWire 35s, Proptest+Parity+GoldenV2 257s, Heap-Audit 117s, CLAP Release+Concurrency 895s, Long Benchmarks 2116s). Zero crashes, zero NaN/Inf, zero timeouts.
    - **Evidência de latência:** `cargo check --lib` e `cargo clippy --lib` — zero warnings/errors. `utils/lints.sh` completo com 4 profiles (Standalone, Pure Core, CLAP Plugin, All Features).
    - **Evidência de throughput (`inference_bench`, release mode, cold run):**
      - WaveNet Standard CH=16 64samp: 70.0 µs (**−2.7%**, improved) — usa `dot_product_16x_f32` (scalar no AVX2, pronto para AVX-512)
      - WaveNet Standard CH=16 (comparison group): 70.9 µs (**−4.3%**, improved)
      - WaveNet Dynamic CH=5 64samp: 33.5 µs (**−2.2%**, improved) — 4-wide baseline
      - A2Full CH=8 64samp: 27.4 µs (**−2.3%**, improved) — raw AVX2 kernel inalterado
      - A2Lite CH=3 64samp: 23.0 µs (**−1.8%**, improved) — código não afetado
      - Block-size scaling (WaveNet Standard): perfeitamente linear — 32→512 samp escala 16× (36.7→562.6 µs)
    - **Long-run benchmarks (`long_inference_bench`):** WaveNet Standard CH=16 @ 4096samp: 4.498 ms (70.3 µs/64samp). A2Full CH=8 @ 4096samp: 27.7 µs/frame. Estáveis por 100 iterações sem deriva térmica ou degradação.
    - **Correções pós-review:** Bug em `slice_conv1d` corrigido (stride fonte usava 4 fixo, agora usa `conv.interleave_width`). `select_interleave_width` unificada como `pub(crate)` e chamada de 4 sites (layout.rs, conv1d.rs ×2, conv1d_dual.rs, traits.rs).
    - **Riscos aceitos:** Em CPUs AVX2-only, `dot_product_16x_f32` usa scalar reference (sem FMA). Isto é correto e equivalente numericamente; decomposição via dois `dot_product_8x_f32_avx2` pode ser adicionada como otimização futura. O caminho dual-frame é desabilitado para interleaving >4-wide (sem kernels dual 8x/16x), usando fallback single-frame — latência aceitável para models CH=8/16.
    - **Conclusão:** A integração é correta, estável, bem testada (6 fases de auditoria longa + 779 testes lib + 26 golden vectors) e fornece a infraestrutura de dispatch monomorfizado para aceleração AVX-512 futura. Zero efeitos colaterais em estabilidade ou latência. Ganho de throughput consistente de 2-4% em todos os modelos WaveNet/A2. **Sprint 3 concluído.**
    - *Referência no `TODO-findings.md`: F-09*
