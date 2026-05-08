<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->
# TODO-sprints.md — Roteiro Técnico de Auditoria NAM-rs

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

### ✅ Tarefa 8.1 — `loader/dispatcher/wavenet.rs` — `validate_layer_activations` Limitada

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

---

## Épico 10 — Arquitetura A2 (Staging)

### ✅ Tarefa 10.1 — `WavenetA2Placeholder` — Placeholder Silencioso

- **Arquivo**: `models/mod.rs:262-278`
- **Achado**: O placeholder retorna silêncio absoluto. Quando a implementação A2 começar, este struct será substituído pela engine real.
- **Proposta**: Manter como está, mas adicionar `log::warn!` no `process()` para que o usuário saiba que está em modo placeholder (evita confusão de "sem som").

### ✅ Tarefa 10.2 — Stubs `film.rs` e `gating.rs` — Sem Implementação

- **Arquivo**: `models/film.rs`, `models/gating.rs`
- **Achado**: Contêm apenas definições de types/traits/configs sem implementação funcional. São stubs corretos para staging.
- **Proposta**: Sem ação necessária até o início da implementação A2.

### ✅ Tarefa 10.3 — `wavenet_params.rs` — Estruturas Completas sem Consumidores

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
