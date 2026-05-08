<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints.md — Roteiro Técnico de Auditoria NAM-rs

> Documento consolidado após auditoria meticulosa de **todas as Jornadas** (1-5).  
> Organizado em **Épicos temáticos**, cada um pronto para ser decomposto em Sprints e Tarefas Técnicas.  
> **Nenhum arquivo `.rs` foi alterado** nesta fase — apenas leitura, análise e diagnóstico.

---

## Legenda de Severidade

| Ícone | Significado                                               |
| ----- | --------------------------------------------------------- |
| 🔴    | Correção crítica (corretude, RT-safety, UB potencial)     |
| 🟡    | Melhoria importante (limpeza, robustez, manutenibilidade) |
| 🟢    | Melhoria desejável (ergonomia, documentação, polish)      |

---

## Épico 1 — Fundação & Infraestrutura (CONCLUÍDO)

> **Módulos**: `Cargo.toml`, `lib.rs`, `main.rs`, `params.rs`, `audio_host.rs`, `colors.rs`

### ✅ Tarefa 1.1 — `LoadedModels` Tuple → Struct Nomeada (CONCLUÍDO)

- **Arquivo**: `loader/mod.rs:20-26`
- **Achado**: O tipo `LoadedModels` é uma tupla de 5 elementos `(Option<Box<DynamicModel>>, Option<Box<DynamicModel>>, f32, f32, u32)`. Isso prejudica a legibilidade em todos os call sites.
- **Proposta**: Converter para `struct LoadedModelPair { model_l, model_r, input_adj, output_adj, sample_rate }`.

### ✅ Tarefa 1.2 — `NamMetadata` Default via `impl Default` (CONCLUÍDO)

- **Arquivo**: `loader/mod.rs:84-96`
- **Achado**: `metadata.clone().unwrap_or(NamMetadata { ... })` com 11 campos `None`. Deveria usar `NamMetadata::default()` se a struct tiver um `Default` derivável (todos os campos são `Option<T>`).
- **Proposta**: Derivar `Default` em `NamMetadata` e substituir o bloco por `.unwrap_or_default()`.

### ✅ Tarefa 1.3 — Constantes Mágicas de Calibração (CONCLUÍDO)

- **Arquivo**: `loader/mod.rs:97-101`
- **Achado**: `12.0`, `-18.0` são constantes de calibração NAM sem nome. A referência C++ também usa literals, mas para o Rust nomeá-las melhora a autoexplicação.
- **Proposta**: Extrair `const DEFAULT_INPUT_LEVEL_DBU: f32 = 12.0;` e `const DEFAULT_LOUDNESS_DB: f32 = -18.0;`.

---

## Épico 2 — Diagnóstico & Comunicação Inter-Thread (CONCLUÍDO)

> **Módulos**: `diagnostics.rs`, `spsc.rs`, `cli.rs`

### ✅ Tarefa 2.1 — `GcOverflowBuffer` — Validação de Tamanho (CONCLUÍDO)

- **Arquivo**: `spsc.rs`
- **Achado**: O `GcOverflowBuffer` aceita `capacity = 0` silenciosamente, o que pode causar panic em index `0 % 0`. Embora na prática `capacity` seja sempre > 0, a API pública deveria ser defensiva.
- **Proposta**: Adicionar `assert!(capacity > 0)` ou validação na construção.

### ✅ Tarefa 2.2 — Docs de `RtStatusFlags` — Diagrama de Bits (CONCLUÍDO)

- **Arquivo**: `spsc.rs`
- **Achado**: A struct é bem documentada, mas um diagrama de bits no docstring (bit 0 = HAS_CLIPPED, etc.) facilitaria a leitura para novos contribuidores.
- **Proposta**: Adicionar tabela markdown nos doc-comments com mapa de bits.

---

## Épico 3 — DSP Pipeline & Gate (CONCLUÍDO)

> **Módulos**: `dsp/pipeline.rs`, `dsp/gain.rs`, `dsp/gate.rs`, `dsp/telemetry.rs`, `dsp/vring.rs`

### ✅ Tarefa 3.1 — `run_inference` — Stack Array no Hot-Path com Resampler (CONCLUÍDO)

- **Arquivo**: `pipeline.rs:240-241`, `pw_host.rs:200-203`
- **Achado**: `let mut temp_out_l = [0.0f32; MAX_RESAMP_BUF];` e `temp_out_r` alocavam **32 KiB na stack** dentro do hot-path.
- **Solução**: Buffers movidos para o `DspPipelineContext` e pré-alocados no estado da closure de processamento (heap-backed), garantindo RT-safety e eliminando risco de stack overflow. Contrato de tamanho documentado em `MAX_RESAMP_BUF`.

### ✅ Tarefa 3.2 — `handle_silence_bypass` — Ordering Inconsistente (CONCLUÍDO)

- **Arquivo**: `pipeline.rs:140-155, 319-348, 391-410`
- **Achado**: `active_read_idx.load(Relaxed)` seguido de `fence(Release)` e `store(Relaxed)`.
- **Solução**: Refatorado para usar semântica explícita `Release/Acquire` em `active_read_idx` e `generation`. Removidas as fences manuais (`fence(Release)` e `fence(Acquire)`), consolidando o protocolo SeqLock de forma mais robusta e legível.

### ✅ Tarefa 3.3 — `pipeline.rs` — Sinalização de Silêncio Invertida (CONCLUÍDO)

- **Arquivo**: `spsc.rs`, `pipeline.rs:364-380`, `rt_setup.rs:249-265`
- **Achado**: `IS_SILENT` era setado durante `FadingIn`, o que é misleading pois o sinal está se tornando ativo.
- **Solução**: Introduzida a flag `RT_STATUS_IS_FADING`. Agora `IS_SILENT` é setada apenas em silêncio total (`GateState::Closed`), enquanto `IS_FADING` cobre os estados de transição (`FadingIn`/`FadingOut`). Logs do host atualizados para refletir essa distinção.

### ✅ Tarefa 3.4 — `DspBridge` — Double-Buffer sem Backpressure (CONCLUÍDO)

- **Arquivo**: `pipeline.rs:95-100, 160-170, 350-365, 415`, `spsc.rs`, `rt_setup.rs`
- **Achado**: Se o capture callback produz mais rápido que o playback consome, o back-buffer é sobrescrito silenciosamente.
- **Solução**: Introduzidos `consumed_gen` e `dropped_frames`. O capture detecta se a geração anterior ainda não foi consumida e incrementa a telemetria. Logs de aviso adicionados ao host para sinalizar drifting/drops.

### ✅ Tarefa 3.5 — `gate.rs` — Parâmetro `_params` Ignorado em `apply_gain_rt` (CONCLUÍDO)

- **Arquivo**: `gate.rs`, `pipeline.rs`, `gate_test.rs`
- **Achado**: O parâmetro `_params: &GateParams` era intencionalmente não-utilizado em `apply_gain_rt`, mas passado em todos os call sites.
- **Solução**: Removido o parâmetro `params` de `apply_gain_rt`, `apply_output_stage` e de todos os call sites (incluindo testes), simplificando a API e reduzindo o acoplamento desnecessário, já que o estado do gate já está "baked" no `DynamicHysteresis`.

### ✅ Tarefa 3.6 — `telemetry.rs` — Histograma Sem Proteção Contra Overflow (CONCLUÍDO)

- **Arquivo**: `telemetry.rs:49`
- **Achado**: `bins[index].fetch_add(1, Relaxed)` poderia silenciosamente wrapar em `u32::MAX` após ~4 bilhões de observações, invalidando cálculos de percentil.
- **Solução**: Implementada adição saturante usando `fetch_update`. Embora o custo de um loop CAS seja ligeiramente superior a um `fetch_add`, a integridade estatística em sessões de longa duração é preservada.

### ✅ Tarefa 3.7 — `vring.rs` — `VirtualRingBuffer` Não Implementa `Debug` (CONCLUÍDO)

- **Arquivo**: `vring.rs:24-28`
- **Achado**: A struct usava raw pointers e não podia derivar `Debug`, dificultando o diagnóstico de alinhamento e mapeamento.
- **Solução**: Implementado o trait `Debug` manualmente. Agora a struct exibe metadados seguros (endereço base, `size_elements` e `capacity_virtual`) sem derreferenciar ponteiros ou expor dados sensíveis do buffer no log.

---

## Épico 4 — Resampler & Sinc Kernel

> **Módulos**: `dsp/resampler.rs`, `dsp/sinc_kernel.rs`

### ✅ Tarefa 4.1 — `DelayLine` Usa `Vec` — Potencial para `AlignedVec` (CONCLUÍDO)

- **Arquivo**: `resampler.rs:51-53`
- **Achado**: `DelayLine.buf` é um `Vec<f32>` sem alinhamento garantido. O `window_ptr()` é usado em `convolve_stereo_avx2` que usa `_mm256_loadu_ps` (unaligned), então funciona. Porém, se o load fosse trocado para `_mm256_load_ps` (aligned, 1 ciclo mais rápido), precisaria de alinhamento.
- **Solução**: `DelayLine` migrado para `AlignedVec<f32>`, garantindo alinhamento de 64 bytes (Cache Line / AVX-512). O uso de `AlignedVec::new()` assegura a alocação correta no cold-path.

### ✅ Tarefa 4.2 — `sinc_kernel.rs` — `AlignedCoeffs._len` Nunca Usado (CONCLUÍDO)

- **Arquivo**: `sinc_kernel.rs:62`
- **Achado**: O campo `_len` estava prefixado com `_` e não era usado.
- **Solução**: Removido o campo `len` (ex-`_len`) e a lógica associada em `AlignedCoeffs`, simplificando a struct já que o `PolyphaseBank` gerencia as dimensões e o padding (sempre múltiplo de 8). Lints validados via `cargo clippy`.

### 🟢 Tarefa 4.3 — `gcd()` — Função Pública Sem Consumidores Internos Visíveis

- **Arquivo**: `sinc_kernel.rs:113`
- **Achado**: `gcd` é `pub` mas não parece ser usada em nenhum código do resampler (a razão `from_rate/to_rate` é tratada como `f64`). Pode ser resíduo de um design anterior baseado em L/M racionais.
- **Proposta**: Verificar se há consumidores externos (testes, benchmarks). Se não, tornar `pub(crate)` ou remover.

---

## Épico 5 — Modelo Neural: WaveNet

> **Módulos**: `models/wavenet.rs`, `models/wavenet_common.rs`, `models/wavenet_dyn.rs`, `loader/dispatcher/wavenet.rs`

### 🟡 Tarefa 5.1 — Duplicação Massiva entre `process_single_frame` e `process_single_frame_bf16`

- **Arquivo**: `wavenet.rs:82-332`
- **Achado**: Os métodos `process_single_frame_internal` (~100 linhas) e `process_single_frame_bf16_internal` (~100 linhas) compartilham ~90% da estrutura lógica, diferindo apenas no tipo do `layer_buffer` (`&[f32]` vs `&[u16]`), no tipo de `in_taps` (`[f32; IN]` vs `[u16; IN]`), e no dispatch para `dot_product_4x_interleaved` vs `dot_product_4x_interleaved_bf16`. O mesmo padrão repete-se em `process_dual_frame_internal` (~165 linhas) vs `process_dual_frame_bf16_internal` (~125 linhas).
- **Proposta**: Explorar uma abstração via trait ou macro que parametrize o tipo de buffer (`f32`/`u16`) e o método de dot product. Cuidado: qualquer abstração não pode introduzir overhead no hot-path. Uma macro `define_conv1d_process!` pode ser a solução mais segura.

### 🟡 Tarefa 5.2 — `DenseLayer::process_acc_single_frame` e `process_fused` São Idênticos

- **Arquivo**: `wavenet.rs:707-726`
- **Achado**: `process_acc_single_frame` e `process_fused` chamam exatamente o mesmo `M::fused_add_gemv` com os mesmos argumentos. São aliases com nomes diferentes.
- **Proposta**: Remover um e manter o outro como alias (`#[inline(always)] pub fn process_fused(...) { self.process_acc_single_frame(...) }`), ou consolidar em um único nome.

### 🟡 Tarefa 5.3 — `DenseLayer::process_acc_block` e `process_fused_block` São Idênticos

- **Arquivo**: `wavenet.rs:749-789`
- **Achado**: Mesmo caso do E5.2 mas para a variante batch.
- **Proposta**: Idem E5.2.

### 🟡 Tarefa 5.4 — `wavenet_common.rs` — `assert!` no Hot-Path

- **Arquivo**: `wavenet_common.rs:541-545`
- **Achado**: `assert!(mixin_len <= 4096, ...)` no `process_block_internal` é executado a cada chamada no hot-path DSP. Em release, se a condição falhar, causa panic (abort). Em debug, causa panic. Ambos violam RT-safety.
- **Proposta**: Converter para `debug_assert!` (somente em debug) ou validar na construção/carregamento para garantir que `num_frames * out_ch ≤ 4096` é sempre verdade (contrato estático).

### 🟡 Tarefa 5.5 — `wavenet_dyn.rs` — Duplicação entre `process` e `prewarm`

- **Arquivo**: `wavenet_dyn.rs:51-233`
- **Achado**: O método `process` e `prewarm` em `WaveNetLayerArrayDyn` compartilham ~70% da lógica (setup, rechannel, cascateamento, head_rechannel). A diferença principal é que `prewarm` processa 1 frame e preenche o RF backward.
- **Proposta**: Extrair a lógica comum de cascateamento para um helper parametrizado por `num_frames` e "pre-fill mode".

### 🟢 Tarefa 5.6 — `loader/dispatcher/wavenet.rs` — Duplicação de Lógica de Quantização

- **Arquivo**: `loader/dispatcher/wavenet.rs:346-397 + 466-527`
- **Achado**: A lógica `if is_bf16 { f32_to_bf16(raw[i]) } else { half::f16::from_f32(raw[i]).to_bits() }` e o transpose Interleaved 4-Wide são duplicados verbatim entre `read_conv1d_weights` (const generic) e `read_conv1d_weights_dyn` (dinâmico). O mesmo ocorre com `read_dense_layer` vs `read_dense_layer_dyn`.
- **Proposta**: Extrair helper `quantize_weight(raw: f32, is_bf16: bool) -> u16` e um `transpose_interleaved_4wide(...)` genérico.

---

## Épico 6 — Modelo Neural: LSTM

> **Módulos**: `models/lstm.rs`, `models/lstm_dyn.rs`, `loader/dispatcher/lstm.rs`

### 🟡 Tarefa 6.1 — `lstm.rs` — Macro Gera Funções com Assinatura Inconsistente

- **Arquivo**: `lstm.rs:24-135`
- **Achado**: A macro `define_lstm_process!` gera `process_sample_*` com `#[$target_meta]` como atributo. Para a variante AVX2, usa `#[inline(always)]` (sem target_feature), enquanto AVX512 usa `#[target_feature(enable = "avx512f,...")]`. Isso é intencional (AVX2 é baseline do target), mas a inconsistência visual dificulta revisão.
- **Proposta**: Documentar a decisão no próprio macro com um comentário inline.

### 🟡 Tarefa 6.2 — `lstm.rs` — Fallback Escalar Usa `half::f16` Sem SIMD

- **Arquivo**: `lstm.rs:307-337`
- **Achado**: `process_sample_scalar` converte pesos via `half::f16::from_bits(w).to_f32()` dentro de um loop triplo aninhado. Este fallback nunca é chamado no hot-path (os SIMD variants são usados), mas se fosse, seria extremamente lento.
- **Proposta**: Documentar que é exclusivo para testes de paridade numérica, não para produção. Considerar `#[cfg(test)]` se não for necessário em release.

### 🟡 Tarefa 6.3 — `lstm_dyn.rs` — Portas LSTM Separadas em Loops Distintos

- **Arquivo**: `lstm_dyn.rs:99-106`
- **Achado**: As ativações são aplicadas em 3 chamadas SIMD separadas: `sigmoid_slice(0..2h)`, `tanh_slice(2h..3h)`, `sigmoid_slice(3h..4h)`. No C++ de referência, as portas são processadas em um kernel fundido `fused_lstm_gates` que faz sigmoid + tanh + element-wise em um único passo, evitando 3 passes sobre os dados.
- **Proposta**: Criar um `fused_lstm_gates_dyn` que aceite dimensões dinâmicas e processe tudo em um passo, similar ao que `lstm.rs` (estático) já faz via macro.

### 🟢 Tarefa 6.4 — `models/mod.rs` — Prewarm Duplicado para LSTM 1 e 2 Camadas

- **Arquivo**: `models/mod.rs:192-239`
- **Achado**: As implementações de `NamModel::prewarm` para `LstmModel1` e `LstmModel2` são idênticas (~15 linhas cada): `reset_states`, loop de chunks de 512 com silêncio.
- **Proposta**: Extrair para um helper `lstm_prewarm_common(model: &mut impl LstmLike, num_samples: usize)` ou função livre.

---

## Épico 7 — Math & SIMD Backend

> **Módulos**: `math/simd/mod.rs`, `math/simd/avx2.rs`, `math/simd/avx512.rs`, `math/simd/fallback.rs`, `math/fastmath.rs`

### 🟡 Tarefa 7.1 — `dispatch_simd!` Macro — Fallback Usa AVX2

- **Arquivo**: `math/simd/mod.rs:48`
- **Achado**: No Modo 2, `InstructionSet::Fallback => $target.$m256($($arg),*)` despacha para a variante AVX2 em vez de para o backend escalar. Se o fallback genuíno (`FallbackMath`) for necessário, isso é incorreto. No Modo 1, `Fallback => FallbackMath` está correto.
- **Proposta**: Harmonizar: se o Modo 2 (LSTM) realmente nunca roda sem AVX2, documentar a invariante. Caso contrário, adicionar uma variante `$fallback` ao macro.

### 🟡 Tarefa 7.2 — Trait `SimdMath` — Superfície API Grande

- **Arquivo**: `math/simd/traits.rs` (inferido de uso)
- **Achado**: O trait `SimdMath` expõe muitos métodos (`dot_product`, `dot_product_bf16`, `dot_product_4x_interleaved`, `dot_product_4x_interleaved_bf16`, `dot_product_4x_interleaved_dual_frame`, `dot_product_4x_interleaved_dual_frame_bf16`, `accumulate_head`, `fused_add_gemv`, `gemv_overwrite`, `fused_add_gemm_batch`, `fused_gemm_residual_batch`, `gated_activation_and_accumulate_block`, `tanh_and_accumulate_block`, `f32_to_bf16`, `store_bf16`, `sigmoid_slice`, `tanh_slice`, `gemv_overwrite_bf16`, `gemv_overwrite_4gate`, `gemv_overwrite_bf16_4gate`, `IS_BF16`). A superfície é grande mas justificada pela necessidade de monomorphization SIMD.
- **Proposta**: Documentar a rationale de cada grupo de métodos com categorias claras (Dot Products, GEMV, Activations, Conversions).

### 🟢 Tarefa 7.3 — `fastmath.rs` — Verificar Paridade Numérica dos Polinômios Minimax

- **Arquivo**: `math/fastmath.rs`
- **Achado**: Os coeficientes Minimax para `simd_tanh`, `simd_sigmoid`, etc. foram derivados empiricamente. Seria valioso ter testes de golden vector comparando com `libm` (f64) para quantificar o erro máximo absoluto em todo o domínio.
- **Proposta**: Adicionar testes parametrizados com varredura exaustiva de 2^16 pontos em domínios relevantes, comparando com referência f64.

---

## Épico 8 — Loader & Parsing

> **Módulos**: `loader/mod.rs`, `loader/nam_json.rs`, `loader/namb.rs`, `loader/namb_encoder.rs`, `loader/dispatcher/`

### 🟡 Tarefa 8.1 — `loader/dispatcher/wavenet.rs` — `validate_layer_activations` Limitada

- **Arquivo**: `loader/dispatcher/wavenet.rs:24-36`
- **Achado**: Valida apenas contra `"Tanh"`, mas a arquitetura A2 suporta múltiplas ativações (`activations.rs` tem 11 variantes). Quando o suporte A2 for implementado, este guard precisará ser generalizado.
- **Proposta**: Marcar como `// TODO(A2): Generalizar para ActivationType::from_str()` e garantir que o placeholder A2 não passe por esta validação.

### 🟢 Tarefa 8.2 — Erro de I/O Sem Contexto de Errno

- **Arquivo**: `loader/mod.rs:60-66`
- **Achado**: O diagnostic para erro de leitura JSON emite `.param("file", &path_str)` mas não inclui o `io::Error` real (diferente do path `.namb` que inclui `.param("io_error", &e)`).
- **Proposta**: Adicionar `.param("io_error", &e)` no branch `.nam` para paridade.

---

## Épico 9 — Testes, Benchmarks & CI

### 🟡 Tarefa 9.1 — Cobertura de Testes para `pipeline.rs`

- **Achado**: `pipeline_test.rs` existe mas deve ser verificado se cobre os edge cases identificados (silence bypass ordering, resampler stack allocation, gate state transitions).
- **Proposta**: Revisar e expandir testes de integração do pipeline.

### 🟡 Tarefa 9.2 — Golden Vectors para Paridade C++

- **Achado**: Os testes de integração existentes comparam NAM-rs vs referência, mas a cobertura de topologias (Nano, Feather, Lite, Standard, Dyn, Gated) precisa ser auditada.
- **Proposta**: Garantir que cada variante de `DynamicModel` tenha pelo menos um golden test.

### 🟢 Tarefa 9.3 — Benchmarks de Resampler

- **Achado**: O resampler FIR nativo substituiu `rubato` mas não há benchmarks comparativos publicados.
- **Proposta**: Adicionar benchmarks `criterion` para `convolve_stereo_avx2`, `convolve_stereo_avx512` e `ResamplerCore::process`.

---

## Épico 10 — Arquitetura A2 (Staging)

### 🟢 Tarefa 10.1 — `WavenetA2Placeholder` — Placeholder Silencioso

- **Arquivo**: `models/mod.rs:262-278`
- **Achado**: O placeholder retorna silêncio absoluto. Quando a implementação A2 começar, este struct será substituído pela engine real.
- **Proposta**: Manter como está, mas adicionar `log::warn!` no `process()` para que o usuário saiba que está em modo placeholder (evita confusão de "sem som").

### 🟢 Tarefa 10.2 — Stubs `film.rs` e `gating.rs` — Sem Implementação

- **Arquivo**: `models/film.rs`, `models/gating.rs`
- **Achado**: Contêm apenas definições de types/traits/configs sem implementação funcional. São stubs corretos para staging.
- **Proposta**: Sem ação necessária até o início da implementação A2.

### 🟢 Tarefa 10.3 — `wavenet_params.rs` — Estruturas Completas sem Consumidores

- **Arquivo**: `models/wavenet_params.rs`
- **Achado**: Estruturas `LayerParamsA2`, `LayerArrayParamsA2`, `HeadParams` estão completamente definidas e testadas, mas não possuem consumidores ainda (serão usadas pelo construtor A2).
- **Proposta**: Sem ação necessária até a implementação do construtor A2.

---

## Ordem de Prioridade Sugerida

1. **Sprint 1 — Corretude & RT-Safety**: E3.2, E3.3, E5.4, E7.1
2. **Sprint 2 — Limpeza & Deduplicação**: E1.1, E1.2, E5.2, E5.3, E5.6, E6.4, E8.2
3. **Sprint 3 — Robustez DSP**: E3.1, E3.4, E4.1, E6.3
4. **Sprint 4 — Deduplicação Pesada**: E5.1, E5.5
5. **Sprint 5 — Documentação & Testes**: E1.3, E2.2, E3.5, E3.6, E3.7, E4.2, E4.3, E6.1, E6.2, E7.2, E7.3, E9.1, E9.2, E9.3
6. **Sprint 6 — Staging A2**: E8.1, E10.1, E10.2, E10.3

---

> **Próximo Passo**: Desdobrar cada Sprint em Tarefas Técnicas granulares via workflow `/tarefa`.
