<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — EPIC-1 "Zero Spill": Microarquitetura dos Kernels WaveNet AVX2

> **Referência principal:** [`TODO-findings.md`](./TODO-findings.md) — EPIC-1 (linha 475) e
> findings **F-P1, F-P3, F-P5, F-P4, F-P6** (Seção A).
>
> **Meta de latência:** WaveNet Standard CH16 de **42,7 µs → ~30 µs** (−30%). Benefício em
> cascata para Feather/Nano/head paths.
>
> **Invariante inegociável:** toda transformação neste épico deve ser **bit-exact** (mesma árvore
> de redução FMA → mesmo resultado float aritmético) em relação ao baseline atual. Qualquer desvio
> de ESR/SNR no `quality-dashboard` = abortar e investigar imediatamente.
>
> **Gerado por:** `planejador-arquiteto` em 2026-07-17 com base na auditoria de
> `revisor-auditor`/`pesquisador-inovador`.

---

## Princípios de Execução do EPIC

| #   | Princípio                                                                                                                                                                                     |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Sequência com gate**: cada tarefa só começa após o gate completo da anterior passar.                                                                                                        |
| 2   | **Gate mínimo obrigatório**: `utils/lints.sh` → `utils/tests-quick.sh` → `cargo bench` dirigido → `utils/quality-dashboard.sh --check docs/quality-contract.txt` → diff do `dsp_hotpath.asm`. |
| 3   | **Nenhum `rcpps`/`rsqrtps`**: substituições de divisão/raiz são anti-escopo (violam paridade).                                                                                                |
| 4   | **Stable-only**: nenhuma flag nightly (`-Ztune-cpu`, etc.), nenhum `#![feature(...)]`.                                                                                                        |
| 5   | **Sem fallbacks ISA**: baseline é x86-64-v3; proibido `is_x86_feature_detected!("avx2")`.                                                                                                     |
| 6   | **Mudanças cirúrgicas**: cada tarefa toca apenas o escopo declarado; não refatorar além.                                                                                                      |

---

## Sprint S1 — Eliminação de Spills no Kernel Dual 16-lane (F-P1) 🔴

> **Finding:** [F-P1](./TODO-findings.md#L54) — `dot_product_16x_f32_dual_accumulate_avx2` +
> `dot_product_16x_f32_dual_avx2` (~58% dos ciclos de DSP). Spills de 4 acumuladores consomem
> ~25-30% dos ciclos da função por tráfego de registradores, não aritmética.
>
> **Risco:** 🟢 Baixo — transformação bit-exact (ordem de FMAs e árvore de redução preservadas).
>
> **Estimativa:** 2–3 dias de engenharia.

### T1.1 — Análise e prova de equivalência bit-exact da estratégia two-pass

**Responsável:** Engenheiro Sênior DSP/SIMD
**Arquivo:** (somente análise — sem modificação de código)

- [ ] Confirmar no `dsp_hotpath.asm` atual que o loop interno de

      `dot_product_16x_f32_dual_accumulate_avx2`
      ([dot_f32_avx2.rs:L327-429](./src/math/gemm/dot_16x/dot_f32_avx2.rs))
      contém exatamente os padrões de spill documentados no F-P1 (vmovups para rsp a ≥ 3
      acumuladores por iteração; cadeias vmovaps sem aritmética).
- [ ] Documentar, em comentário da PR, a prova de equivalência da estratégia two-pass lo/hi:

      - Passada 1: acumuladores `acc_f0_lo{0..3}` + `acc_f1_lo{0..3}` (8 regs), pesos `w_lo`.
      - Passada 2: acumuladores `acc_f0_hi{0..3}` + `acc_f1_hi{0..3}` (8 regs), pesos `w_hi`.
      - Mesma sequência de FMAs por lane, mesma árvore de redução `(acc0+acc1)+(acc2+acc3)`.
      - Conclusão: resultado idêntico bit-a-bit ao kernel original.
- [ ] Verificar que `dot_product_16x_f32_dual_avx2`

      ([dot_f32_avx2.rs:L123-222](./src/math/gemm/dot_16x/dot_f32_avx2.rs))
      tem o mesmo padrão de 16 acumuladores e requer a mesma correção.
- [ ] Confirmar que `dot_product_16x_f32_avx2` e `dot_product_16x_f32_accumulate_avx2`

      (variantes single-frame, 8 acumuladores) estão saudáveis no asm — apenas documentar,
      não tocar.

**Gate T1.1:** revisão de par (peer-review do documento de prova).

---

### T1.2 — Implementar two-pass lo/hi em `dot_product_16x_f32_dual_accumulate_avx2`

**Responsável:** Engenheiro Sênior SIMD
**Arquivo:** [`src/math/gemm/dot_16x/dot_f32_avx2.rs`](./src/math/gemm/dot_16x/dot_f32_avx2.rs)

**Contexto técnico detalhado:**

O kernel atual ([L327-429](./src/math/gemm/dot_16x/dot_f32_avx2.rs)) mantém 16 acumuladores
simultaneamente (`acc_f0_lo{0..3}`, `acc_f0_hi{0..3}`, `acc_f1_lo{0..3}`,
`acc_f1_hi{0..3}`). Com 16 regs ymm disponíveis no x86-64 e a necessidade de 8 cargas de pesos

- 4 broadcasts por iteração, o LLVM é forçado a derramar 4 acumuladores para a pilha.

**Estratégia de refatoração:**

1. **Passada 1 (lo):** percorre todos os `len` taps carregando apenas `w_lo` (os primeiros 8
   floats de cada `[f32; 16]`). Acumuladores ativos: apenas 8 (`acc_f0_lo{0..3}` e
   `acc_f1_lo{0..3}`). Registradores disponíveis restantes: `w_lo` (1) + `s_f0/s_f1` (2–4) =
   folga suficiente para zero spills.

2. **Passada 2 (hi):** percorre novamente os `len` taps carregando apenas `w_hi` (os últimos 8
   floats, offset +8 floats do ponteiro do `[f32; 16]`). Acumuladores ativos: apenas 8
   (`acc_f0_hi{0..3}` e `acc_f1_hi{0..3}`).

3. **Custo aceitável:** os broadcasts `s_f0`/`s_f1` são re-executados na passada 2. As linhas de
   pesos (~3 KB para K=3, IN=16) são re-lidas 2×. Na Zen 2 (2 loads de 32B/ciclo, L1 de 32 KB),
   esse custo é irrelevante comparado com os spills eliminados.

4. **Redução final** `(acc0+acc1)+(acc2+acc3)`: idêntica ao código atual → bit-exact garantido.

**Estrutura esperada do código (pseudocódigo):**

      ```rust
      // Passada 1: somente metade lo (w_lo = weights[i][0..8])
      // 8 acumuladores: acc_f0_lo{0..3}, acc_f1_lo{0..3}
      let mut acc_f0_lo0 = _mm256_loadu_ps(init_f0.as_ptr()); // ou setzero
      // ...
      let mut i = 0;
      dot4x_simd4!(i, len, {
      let w0_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
      // NÃO carregar w_hi nesta passada
      let s_f0_0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
      let s_f1_0 = _mm256_set1_ps(*state_f1.get_unchecked(i));
      acc_f0_lo0 = _mm256_fmadd_ps(w0_lo, s_f0_0, acc_f0_lo0);
      acc_f1_lo0 = _mm256_fmadd_ps(w0_lo, s_f1_0, acc_f1_lo0);
      // ... idem para i+1, i+2, i+3 ...
      });
      // redução lo: (acc_f0_lo0+acc_f0_lo1)+(acc_f0_lo2+acc_f0_lo3) -> lo_f0
      // idem para lo_f1

      // Passada 2: somente metade hi (w_hi = weights[i][8..16])
      i = 0;
      // 8 acumuladores: acc_f0_hi{0..3}, acc_f1_hi{0..3}
      dot4x_simd4!(i, len, {
      let w0_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
      // ...
      });
      // redução hi -> hi_f0, hi_f1 ; store final
      ```

- [ ] Implementar a refatoração em `dot_product_16x_f32_dual_accumulate_avx2` conforme

      estratégia acima.
- [ ] Aplicar a mesma refatoração em `dot_product_16x_f32_dual_avx2` (variante sem `init` —

      acumuladores iniciam em zero nas duas passadas).
- [ ] Manter intactas `dot_product_16x_f32_avx2` e `dot_product_16x_f32_accumulate_avx2`

      (variantes single-frame com apenas 8 acumuladores — já saudáveis).
- [ ] Atualizar `//!` doc-comments de ambas as funções refatoradas descrevendo o two-pass

      e o racional de pressão de registradores.
- [ ] Verificar invariantes de `unsafe` — garantir que `get_unchecked` é acessado apenas dentro

      de `i < len` / `i < len-3` (igual ao código anterior).

**Gate T1.2:**

      ```bash
      utils/lints.sh
      utils/tests-quick.sh
      cargo bench --bench dot_4x_bench
      cargo bench --bench inference_bench -- wavenet_standard
      cargo bench --bench regression_gate
      utils/quality-dashboard.sh --check docs/quality-contract.txt
      # Confirmar no novo dsp_hotpath.asm: zero vmovups %ymmN, (%rsp) no loop interno
      ```

---

### T1.3 — Verificação de spills no `dsp_hotpath.asm` após T1.2

**Responsável:** Engenheiro Sênior DSP (pode ser o mesmo de T1.1)
**Arquivo:** utilitários de build apenas.

- [ ] Executar `utils/build-release.sh` (PGO+BOLT) para gerar o novo `target/dsp_hotpath.asm`.
- [ ] Grep no loop interno de `dot_product_16x_f32_dual_accumulate_avx2`:

      - **DEVE estar ausente:** `vmovups %ymm\d+, 0x[0-9a-f]+\(%rsp\)` (store de acumulador).
      - **DEVE estar ausente:** cadeias `vmovaps %ymm\d+, %ymm\d+` sem FMA intercalado.
      - **DEVE estar presente:** padrão `vfmadd231ps … ; vfmadd231ps …` ininterrupto.
- [ ] Registrar a contagem de instruções do loop interno (antes/depois) no corpo da PR.
- [ ] Confirmar que `WaveNetLayer<1,8,3>` (CH=8) mantém ≤ 0 spills (já saudável — só validar).

**Gate T1.3:** PR aprovada com evidência de asm em anexo.

---

## Sprint S2 — Const-generic no `fused_gemm_residual_batch_f32_avx2` (F-P3) 🟠

> **Finding:** [F-P3](./TODO-findings.md#L153) — `fused_gemm_residual_batch_f32_avx2`
> ([fused_residual_batch.rs:L229-399](./src/math/gemm/gemm_batch/fused_residual_batch.rs)).
> Strides em runtime (`in_len`/`out_len` derivados de `slice.len() / num_frames`) impedem
> strength-reduction do LLVM → 4–5 `leaq` + reloads de ponteiro por iteração = ~25-30% dos
> ciclos da função (~21% do DSP total).
>
> **Pré-requisito:** Sprint S1 concluída e gateada.
>
> **Risco:** 🟡 Baixo-médio — bit-exact, mas exige especialização const-generic em Rust stable
> e despacho correto no call-site.
>
> **Estimativa:** 2–3 dias de engenharia.

### T2.1 — Implementar variante const-generic `fused_gemm_residual_batch_f32_const`

**Responsável:** Engenheiro Sênior Rust/SIMD
**Arquivo:** [`src/math/gemm/gemm_batch/fused_residual_batch.rs`](./src/math/gemm/gemm_batch/fused_residual_batch.rs) + call-site [`src/models/wavenet/dense.rs`](./src/models/wavenet/dense.rs)

**Contexto técnico:**

O problema está nas linhas 244-245: `in_len = in_frames.len() / num_frames` e
`out_len = out_frames.len() / num_frames`. Com `IN` e `OUT` como const-generics, o LLVM
substitui `in_c * out_len + out_c` por endereçamento imediato, eliminando os `leaq` encadeados.

No call-site em `DenseLayer::process_residual_batch`, o chamador conhece `CH` em compile-time
via `WaveNetLayer<COND, CH, K>`, possibilitando o despacho para a versão especializada.

**Implementação:**

1. Criar `fused_gemm_residual_batch_f32_const::<const IN: usize, const OUT: usize>` com a
   mesma lógica do loop AVX2 atual, mas com `in_len = IN` e `out_len = OUT` constantes.
   Rust stable suporta const-generics numéricos desde 1.51 — sem `#![feature(...)]`.
2. O body (macros `gemm_batch_frame_loop_avx2!`, `gemm_batch_outc_loop_avx2!`,
   `gemm_batch_inner_dual_avx2!`) permanece inalterado — apenas `in_len`/`out_len` viram
   `IN`/`OUT`.
3. **Strength-reduction manual no fallback dinâmico:** substituir
   `weights.as_ptr().add(in_c * out_len + out_c)` por ponteiro incremental
   (`let mut wp = weights.as_ptr().add(out_c); wp = wp.add(out_len);` a cada `in_c`).
   Remove a multiplicação mesmo sem const-generic.
4. Atualizar `DenseLayer::process_residual_batch` para despachar à versão const-generic nos
   casos `(IN=16, OUT=16)`, `(8, 8)`, `(4, 4)`.

- [ ] Criar a assinatura e body const-generic.
- [ ] Aplicar strength-reduction de ponteiro na variante dinâmica existente.
- [ ] Atualizar o call-site em `DenseLayer::process_residual_batch`.
- [ ] Atualizar doc-comments das funções.

**Gate T2.1:**

      ```bash
      utils/lints.sh
      utils/tests-quick.sh
      cargo bench --bench gemv_bench
      cargo bench --bench inference_bench -- wavenet_standard wavenet_feather wavenet_nano
      utils/quality-dashboard.sh --check docs/quality-contract.txt
      # Verificar no dsp_hotpath.asm: ausência de leaq encadeados no loop interno;
      # presença de modos de endereçamento imediatos (imm32 em vez de %rXX*stride)
      ```

---

## Sprint S3 — ILP ×2 no laço de ativação tanh (F-P5) 🟠

> **Finding:** [F-P5](./TODO-findings.md#L213) — laço de ativação em
> `tanh_and_accumulate_block_avx2` e variantes
> ([`src/math/wavenet/accumulate/avx2.rs`](./src/math/wavenet/accumulate/avx2.rs)).
> Um único ymm por iteração → cadeia serial de dependência em `vdivps` (latência ~13c na Zen 2)
> → 10–13% dos ciclos da função estagnados no `vminps` pós-div.
>
> **Pré-requisito:** Sprint S2 concluída e gateada.
>
> **Risco:** 🟢 Muito baixo — bit-exact (operações por lane idênticas, apenas intercaladas).
>
> **Estimativa:** 1–2 dias de engenharia.

### T3.1 — Desenrolar o laço de ativação para 2 vetores ymm por iteração

**Responsável:** Engenheiro SIMD
**Arquivo:** [`src/math/wavenet/accumulate/avx2.rs`](./src/math/wavenet/accumulate/avx2.rs)

**Contexto técnico:**

O macro `wavenet_simd_avx2!` em `tanh_and_accumulate_block_avx2` processa um único `__m256`
por iteração (L49-56). A cadeia de dependência por vetor é:

      ```text
      load → clamp → round → 7×FMA (poly) → cvtps2dq → slld → paddd → 2×mul → vdivps (lat ~13c)
      → clamp → store + acumula
      ```

Com `block.len() = 64 frames × 16 ch = 1024 floats / 8 floats/ymm = 128 iterações`, há
paralelismo de dados abundante. Processando 2 ymm independentes por iteração, as cadeias de
latência se sobrepõem no OoO engine.

> **Nota:** `simd_tanh_poly_dual_avx2` já existe em
> [`high_fidelity.rs`](./src/math/activations/tanh/high_fidelity.rs) (L106-140) e compartilha
> broadcasts de constantes entre os 2 ymm. Avaliar se é vantajoso chamar esta versão dual
> em vez de duas chamadas independentes a `simd_tanh_poly_avx2`.

**Estrutura do novo loop principal:**

      ```rust
      while i + 16 <= len {
      let vb0 = _mm256_loadu_ps(block.as_ptr().add(i));
      let vb1 = _mm256_loadu_ps(block.as_ptr().add(i + 8));
      // Chains independentes -> OoO sobrepõe as vdivps
      let vt0 = simd_tanh_poly_avx2(vb0);
      let vt1 = simd_tanh_poly_avx2(vb1);
      _mm256_storeu_ps(block.as_mut_ptr().add(i), vt0);
      _mm256_storeu_ps(block.as_mut_ptr().add(i + 8), vt1);
      let vh0 = _mm256_loadu_ps(head_input.as_ptr().add(i));
      let vh1 = _mm256_loadu_ps(head_input.as_ptr().add(i + 8));
      _mm256_storeu_ps(head_input.as_mut_ptr().add(i),     _mm256_add_ps(vh0, vt0));
      _mm256_storeu_ps(head_input.as_mut_ptr().add(i + 8), _mm256_add_ps(vh1, vt1));
      i += 16;
      }
      // Tail ymm singular (código original mantido)
      while i + 8 <= len { /* ... */ i += 8; }
      ```

- [ ] Aplicar unroll 2× em `tanh_and_accumulate_block_avx2` (caso mais quente: ~58% do DSP em CH=16).
- [ ] Aplicar o mesmo padrão em `tanh_and_overwrite_block_avx2`.
- [ ] Aplicar o mesmo padrão em `tanh_and_accumulate_with_seed_avx2`.
- [ ] **Não tocar** em `gated_activation_and_accumulate_block_avx2` /

      `gated_activation_and_overwrite_block_avx2` (já usam dual internamente via
      `simd_tanh_sigmoid_dual_poly_avx2`).
- [ ] Verificar que o loop de tail `(i+8..len)` cobre corretamente comprimentos não múltiplos de 16.

**Gate T3.1:**

      ```bash
      utils/lints.sh
      utils/tests-quick.sh
      cargo bench --bench math_bench -- tanh_hf
      cargo bench --bench inference_bench -- wavenet_standard wavenet_feather
      utils/quality-dashboard.sh --check docs/quality-contract.txt
      # asm: vminps (stall pós-div) deve desaparecer do topo do perfil da função de acumulação
      ```

---

## Sprint S4 — Kernel broadcast-scale dedicado para `gemv` com `in_len==1` (F-P4) 🟡

> **Finding:** [F-P4](./TODO-findings.md#L192) — `gemv_no_bias_f32_avx2` com `in_len==1`
> ([f32_avx2.rs:L332-590](./src/math/gemm/gemv/f32_avx2.rs)), chamado por rechannel
> (`layer_array.rs:L81-83`), input_mixin (`layer.rs:L56-57`) e head_rechannel
> (`layer_array.rs:L171-175`). O branch `in_len == 1` já existe (L348-372) mas ainda paga
> overhead do prólogo do GEMV genérico (~15% dos ciclos da função = ~6% do total).
>
> **Pré-requisito:** Sprint S3 concluída e gateada.
>
> **Risco:** 🟢 Muito baixo — bit-exact, escopo restrito.
>
> **Estimativa:** 1 dia de engenharia.

### T4.1 — Kernel `broadcast_scale_f32_avx2` const-generic e despacho direto nos call-sites

**Responsável:** Engenheiro Rust/SIMD
**Arquivos:** [`src/math/gemm/gemv/f32_avx2.rs`](./src/math/gemm/gemv/f32_avx2.rs) + call-sites
em `src/models/wavenet/layer_array.rs` e `src/models/wavenet/layer.rs`.

**Contexto técnico:**

O branch `in_len == 1` atual (L348-372) realiza: `out[n * out_len + oc] = in[n] * weights[oc]`.
O problema é que o chamador passa pelo prólogo completo do GEMV genérico (divisão
`in_len = in_frames.len() / num_frames`, inicialização de contadores, branch de checagem de
`in_len == 1` — 9,16% e 5,57% do perfil da função, respectivamente).

**Solução:** extrair para uma função própria `#[inline(always)]` const-generic em `OUT`:

      ```rust
      /// Produto externo escalar × vetor de pesos, sem bias, para `in_len == 1`.
      /// Para cada frame `n`: `out[n*OUT + oc] = in[n] * weights[oc]`.
      #[inline(always)]
      #[target_feature(enable = "avx2,fma")]
      pub unsafe fn broadcast_scale_f32_avx2<const OUT: usize>(
      in_frames: &[f32],      // len == num_frames (scalar de entrada por frame)
      weights: &[f32],        // len == OUT (vetor de pesos fixo)
      out_frames: &mut [f32], // len == num_frames * OUT
      ) {
      // Corpo: exatamente o código do branch in_len==1 atual,
      // com out_len = OUT constante (endereçamento imediato no loop).
      }
      ```

- [ ] Extrair o branch `in_len == 1` para `broadcast_scale_f32_avx2::<const OUT: usize>`.
- [ ] Decorar com `#[inline(always)]` e `#[target_feature(enable = "avx2,fma")]`.
- [ ] Despachar diretamente dos call-sites:
  - `layer_array.rs:rechannel` (IN=1, OUT=CH) → `broadcast_scale_f32_avx2::<CH>`.
  - `layer.rs:input_mixin` (IN=1, OUT=CH) → `broadcast_scale_f32_avx2::<CH>`.
  - `layer_array.rs:head_rechannel`: verificar se IN==1 neste call-site; se sim, idem;
    se não, manter `gemv_no_bias_f32_avx2`.
- [ ] Manter `gemv_no_bias_f32_avx2` inalterado para os caminhos com `in_len > 1`.

**Gate T4.1:**

      ```bash
      utils/lints.sh
      utils/tests-quick.sh
      cargo bench --bench gemv_bench -- gemv_in1
      cargo bench --bench inference_bench -- wavenet_standard
      utils/quality-dashboard.sh --check docs/quality-contract.txt
      ```

---

## Sprint S5 — Higiene de bounds-checks nos kernels quentes (F-P6) 🟢

> **Finding:** [F-P6](./TODO-findings.md#L242) — bounds-checks residuais e caminhos de pânico
> em `WaveNetLayer::process_block_internal`, `fused_gemm_residual_batch_f32_avx2` (tail
> single-frame com indexação checada — L181, L200-205, L351-364) e `gemv_no_bias_f32_avx2`.
>
> **Pré-requisito:** Sprint S4 concluída e gateada.
>
> **Risco:** 🟢 Muito baixo — sem alteração de lógica ou ordem de operações.
>
> **Estimativa:** 0,5–1 dia de engenharia.

### T5.1 — Hoist de asserts e homogeneização de `get_unchecked` vs `[]`

**Responsável:** Engenheiro Rust (qualquer)
**Arquivos:**

- [`src/math/gemm/gemm_batch/fused_residual_batch.rs`](./src/math/gemm/gemm_batch/fused_residual_batch.rs) (L181, L200-205, L351-364)
- [`src/math/gemm/gemv/f32_avx2.rs`](./src/math/gemm/gemv/f32_avx2.rs) (tail do `in_len==1`)
- [`src/models/wavenet/layer.rs`](./src/models/wavenet/layer.rs) (verificar pós S1-S4)

**Implementação — padrão a seguir em cada função:**

      ```rust
      // Hoist no início da função (após os debug_asserts existentes):
      // Permite ao LLVM provar que os índices internos são seguros e eliminar
      // todos os checks residuais no código gerado.
      assert!(in_frames.len() == num_frames * in_len);
      assert!(out_frames.len() == num_frames * out_len);
      assert!(weights.len() >= in_len * out_len);
      // get_unchecked seguro a partir daqui para índices dentro dos bounds acima.
      ```

- [ ] Auditar `fused_residual_batch.rs` — locais com `[]` checado no tail (L200-205, L351-364):

      homogeneizar para `get_unchecked` onde a invariante é provada pelo hoist; ou usar
      `split_at`/`split_at_mut` para criar sub-slices de tamanho fixo (o compilador infere).
- [ ] Fazer o mesmo para `gemv_no_bias_f32_avx2` — verificar tail do branch `in_len == 1`.
- [ ] Verificar `layer.rs::process_block_internal` — se há `callq slice_index_fail` no asm

      pós Sprint S1–S4; homogeneizar conforme o padrão acima.
- [ ] **Não alterar** lógica de negócio ou ordem de operações.
- [ ] **Manter** todos os `debug_assert!` existentes (não removê-los).

**Gate T5.1:**

      ```bash
      utils/lints.sh
      utils/tests-quick.sh
      # Confirmar no dsp_hotpath.asm: zero `callq core::panicking::panic_bounds_check`
      # nos kernels WaveNetLayer, fused_gemm_residual_batch_f32_avx2, gemv_no_bias_f32_avx2.
      utils/quality-dashboard.sh --check docs/quality-contract.txt
      ```

---

## Gate Final do EPIC-1

Após S5 aprovado, executar o gate completo do épico:

      ```bash
      # 1. Linting e verificações estáticas
      utils/lints.sh

      # 2. Testes rápidos (goldens bit-exact obrigatórios)
      utils/tests-quick.sh

      # 3. Benchmarks dirigidos — registrar antes/depois para cada métrica
      cargo bench --bench dot_4x_bench
      cargo bench --bench inference_bench -- wavenet_standard wavenet_feather wavenet_nano
      cargo bench --bench gemv_bench
      cargo bench --bench math_bench -- tanh_hf
      cargo bench --bench regression_gate

      # 4. Contrato de qualidade integral
      utils/quality-dashboard.sh --check docs/quality-contract.txt
      # Esperado: WaveNet Standard ~30 µs (era 42,7 µs), ESR inalterado ≤ 1e-11

      # 5. Verificação de performance regression
      utils/tests-performance-regression.sh

      # 6. Inspeção do novo dsp_hotpath.asm
      # Esperado no loop de dot_product_16x_f32_dual_accumulate_avx2:
      #   - 0 vmovups %ymmN, 0x??(%rsp)     (zero spills de acumulador)
      #   - 0 callq panic_bounds_check        (zero bounds-panics)
      #   - 0 cadeias vmovaps sem FMA         (zero rotações sem aritmética)
      ```

**Critério de aceite do EPIC-1:**

- WaveNet Standard CH16 ≤ 32 µs (meta conservadora), idealmente ~30 µs.
- Todos os contratos de ESR/SNR mantidos — **sem re-baseline**.
- `dsp_hotpath.asm` sem spills e sem bounds-panics nos 4 kernels quentes.

---

## Tabela-resumo de Tarefas

| Sprint | Tarefa                            | Finding | Kernel / Arquivo           | Risco | Bit-exact | Gate            |
| ------ | --------------------------------- | ------- | -------------------------- | ----- | --------- | --------------- |
| S1     | T1.1 — Análise e prova two-pass   | F-P1    | `dot_f32_avx2.rs`          | 🟢    | ✅        | Peer review     |
| S1     | T1.2 — Implementar two-pass lo/hi | F-P1    | `dot_f32_avx2.rs`          | 🟢    | ✅        | Bench + quality |
| S1     | T1.3 — Verificação asm spills     | F-P1    | `dsp_hotpath.asm`          | 🟢    | ✅        | PR + asm diff   |
| S2     | T2.1 — Const-generic fused GEMM   | F-P3    | `fused_residual_batch.rs`  | 🟡    | ✅        | Bench + quality |
| S3     | T3.1 — ILP ×2 ativação tanh       | F-P5    | `avx2.rs` (accumulate)     | 🟢    | ✅        | Bench + quality |
| S4     | T4.1 — broadcast_scale kernel     | F-P4    | `f32_avx2.rs` + call-sites | 🟢    | ✅        | Bench + quality |
| S5     | T5.1 — Higiene bounds-checks      | F-P6    | múltiplos                  | 🟢    | ✅        | lints + asm     |

**Ganho combinado estimado:** 42,7 µs → **~30 µs** (−30%) no WaveNet Standard CH16.

---

*Criado por `planejador-arquiteto` em 2026-07-17. Baseado na auditoria registrada em
[`TODO-findings.md`](./TODO-findings.md) (EPIC-1, linha 475). `TODO-sprints.md` para os
demais épicos (EPIC-2 a EPIC-5) será criado quando solicitado.*
