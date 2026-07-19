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

### T1.1 — Análise e prova de equivalência bit-exact da estratégia two-pass [DONE]

**Responsável:** Engenheiro Sênior DSP/SIMD
**Arquivo:** (somente análise — sem modificação de código)

- [x] Confirmar no `dsp_hotpath.asm` atual que o loop interno de

      `dot_product_16x_f32_dual_accumulate_avx2`
      ([dot_f32_avx2.rs:L327-429](./src/math/gemm/dot_16x/dot_f32_avx2.rs))
      contém exatamente os padrões de spill documentados no F-P1 (vmovups para rsp a ≥ 3
      acumuladores por iteração; cadeias vmovaps sem aritmética).

- [x] Documentar, em comentário da PR, a prova de equivalência da estratégia two-pass lo/hi:

      - Passada 1: acumuladores `acc_f0_lo{0..3}` + `acc_f1_lo{0..3}` (8 regs), pesos `w_lo`.
      - Passada 2: acumuladores `acc_f0_hi{0..3}` + `acc_f1_hi{0..3}` (8 regs), pesos `w_hi`.
      - Mesma sequência de FMAs por lane, mesma árvore de redução `(acc0+acc1)+(acc2+acc3)`.
      - Conclusão: resultado idêntico bit-a-bit ao kernel original.

- [x] Verificar que `dot_product_16x_f32_dual_avx2`

      ([dot_f32_avx2.rs:L123-222](./src/math/gemm/dot_16x/dot_f32_avx2.rs))
      tem o mesmo padrão de 16 acumuladores e requer a mesma correção.

- [x] Confirmar que `dot_product_16x_f32_avx2` e `dot_product_16x_f32_accumulate_avx2`

      (variantes single-frame, 8 acumuladores) estão saudáveis no asm — apenas documentar,
      não tocar.

**Gate T1.1:** revisão de par (peer-review do documento de prova).

---

### T1.2 — Implementar two-pass lo/hi em `dot_product_16x_f32_dual_accumulate_avx2` [DONE]

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

- [x] Implementar a refatoração em `dot_product_16x_f32_dual_accumulate_avx2` conforme

      estratégia acima.

- [x] Aplicar a mesma refatoração em `dot_product_16x_f32_dual_avx2` (variante sem `init` —

      acumuladores iniciam em zero nas duas passadas).

- [x] Manter intactas `dot_product_16x_f32_avx2` e `dot_product_16x_f32_accumulate_avx2`

      (variantes single-frame com apenas 8 acumuladores — já saudáveis).

- [x] Atualizar `//!` doc-comments de ambas as funções refatoradas descrevendo o two-pass

      e o racional de pressão de registradores.

- [x] Verificar invariantes de `unsafe` — garantir que `get_unchecked` é acessado apenas dentro

      de `i < len` / `i < len-3` (igual ao código anterior).

- [x] Gate T1.2 aprovado: lints, testes, benchmarks e quality-dashboard todos OK. Verificação

      parcial de asm (objdump + rustfilt) confirma zero `vmovups` spills no loop interno —
      apenas 14 ymm regs ativos (8 acc + 4 w_lo + 2 broadcast) dos 16 disponíveis. Confirmação
      definitiva via `dsp_hotpath.asm` (PGO+BOLT) delegada para T1.3.

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

### T1.3 — Verificação de spills no `dsp_hotpath.asm` após T1.2 [DONE]

**Responsável:** Engenheiro Sênior DSP (pode ser o mesmo de T1.1)
**Arquivo:** utilitários de build apenas.

- [x] Executar `utils/build-release.sh` (PGO+BOLT) para gerar o novo `target/dsp_hotpath.asm`.

- [x] Grep no loop interno de `dot_product_16x_f32_dual_accumulate_avx2`:

      - **DEVE estar ausente:** `vmovups %ymm\d+, 0x[0-9a-f]+\(%rsp\)` (store de acumulador).
      - **DEVE estar ausente:** cadeias `vmovaps %ymm\d+, %ymm\d+` sem FMA intercalado.
      - **DEVE estar presente:** padrão `vfmadd231ps … ; vfmadd231ps …` ininterrupto.

- [x] Registrar a contagem de instruções do loop interno (antes/depois) no corpo da PR.

- [x] Confirmar que `WaveNetLayer<1,8,3>` (CH=8) mantém ≤ 0 spills (já saudável — só validar).

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

### T2.1 — Implementar variante const-generic `fused_gemm_residual_batch_f32_const` [DONE]

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
- [x] Criar `fused_gemm_residual_batch_f32_const::<IN, OUT>` em `fused_residual_batch.rs` (linha ~400+).
- [x] Strength-reduction: `wp_base` incremental no loop interno de `fused_gemm_residual_batch_f32_avx2` eliminando `in_c * out_len` (linha ~283-290).
- [x] Despacho const-generic em `DenseLayer::process_residual_batch` para (4,4), (8,8), (16,16) via `#[cfg(target_arch = "x86_64")]`.
- [x] Doc-comments atualizados em ambas as funções.
- [x] Gate T2.1 aprovado: lints, tests-quick, quality-dashboard — todos OK. ESR bit-exact (sem re-baseline).

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

### T3.1 — Desenrolar o laço de ativação para 2 vetores ymm por iteração [DONE]

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

### T4.1 — Kernel `broadcast_scale_f32_avx2` const-generic e despacho direto nos call-sites [DONE]

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

- [x] Extrair o branch `in_len == 1` para `broadcast_scale_f32_avx2::<const OUT: usize>`.
- [x] Decorar com `#[inline]` e `#[target_feature(enable = "avx2,fma")]` (Rust proíbe `#[inline(always)]` com `#[target_feature]`).
- [x] Despachar diretamente dos call-sites via `DenseLayer::process_block` com const-check `IN == 1`:
  - `layer_array.rs:rechannel` (IN=1, OUT=CH) → `broadcast_scale_f32_avx2::<CH>`.
  - `layer.rs:input_mixin` (IN=1, OUT=CH) → `broadcast_scale_f32_avx2::<CH>`.
  - `layer_array.rs:head_rechannel`: verificado IN≠1; mantém `gemv_no_bias_f32_avx2`.
- [x] Manter `gemv_no_bias_f32_avx2` inalterado para os caminhos com `in_len > 1`.
- [x] **T4.1 CONCLUÍDO** — Implementados `broadcast_scale_f32_avx2<OUT>` e `broadcast_scale_with_bias_f32_avx2<OUT>` no módulo `f32_avx2`. O despacho ocorre em `DenseLayer::process_block` via `if IN == 1` (const-folded em tempo de compilação). Inference bench mostra 3-5% de melhoria no WaveNet Standard. Todos os testes golden passam bit-exact.

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

### T5.1 — Hoist de asserts e homogeneização de `get_unchecked` vs `[]` [DONE]

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
[`TODO-findings.md`](./TODO-findings.md) (EPIC-1, linha 475).*

---
---

## EPIC-2 — "Lite à altura do nome": caminho 16-wide para CH=12 🔴

> **Referência principal:** [`TODO-findings.md`](./TODO-findings.md) — EPIC-2 (linha 493) e
> findings **F-P2** (causa-raiz) + **F-L4** (guarda-corpo permanente de eficiência).
>
> **Pré-condição:** EPIC-1 [DONE] — kernel 16-wide saudável (sem spills, `dot_product_16x_f32_dual_accumulate_avx2` validado bit-exact).
>
> **Contexto medido pós-EPIC-1 (quality-dashboard, 2026-07-18):**
>
> - WaveNet Standard CH16: **36,9 µs** (era 42,7 µs)
> - WaveNet Feather CH8: **19,1 µs** (era 26,7 µs)
> - **WaveNet Lite CH12: 63,3 µs** ← anomalia — deve ser ≤ 38 µs após este EPIC
>
> **Meta de latência:** WaveNet Lite CH12 de **63,3 µs → ≤ 38 µs** (−40…50%).
>
> **Invariante de paridade:**
>
> - Cada lane do kernel 16x computa seu próprio dot product independentemente das lanes de padding.
> - **Resultado esperado: bit-exact** (mesma ordem de FMAs por lane que o caminho 4-wide atual,
>   desde que a árvore de redução 4-way por índice de tap seja mantida).
> - Se divergência dentro do mesmo tier de ESR (≤1e-11), decisão explícita de re-baseline com PO antes de fechar.
>
> **Gerado por:** `planejador-arquiteto` em 2026-07-18.

---

## Princípios de Execução do EPIC-2

| #   | Princípio                                                                                                                                                                         |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Pré-requisito duro:** EPIC-1 completo e kernel 16-wide verificado sem spills no `dsp_hotpath.asm`.                                                                              |
| 2   | **Sequência com gate:** cada tarefa só começa após o gate completo da anterior.                                                                                                   |
| 3   | **Gate mínimo por tarefa:** `utils/lints.sh` → `utils/tests-quick.sh` → `cargo bench --bench regression_gate` → `utils/quality-dashboard.sh --check docs/quality-contract.txt`.   |
| 4   | **Guardas de paridade:** qualquer desvio de ESR no `quality-dashboard` = **parar e investigar**. Se ESR dentro do tier mas diferente do golden, contatar PO antes de re-baseline. |
| 5   | **F-L4 como coluna permanente:** a métrica µs/MMAC instalada na T2.1 deve permanecer no dashboard para **todos os épicos futuros** — nunca remover.                               |
| 6   | **Custo de memória aceito:** +33% nos pesos conv do Lite (~0,6 KB/camada) é irrelevante para L1/L2 e é o custo do design correto.                                                 |

---

## Sprint S1 — Verificação de Equivalência: árvore de redução 4-wide vs 16-wide

> **Objetivo:** provar ou refutar que a árvore de redução do `dot_product_16x_f32_dual_accumulate` é
> equivalente à do `dot_product_4x_f32_dual_accumulate` para cada lane individual — fundamento para
> a afirmação de bit-exactidão. Esta sprint é pura análise; **nenhuma linha de produção é alterada**.
>
> **Referência:** F-P2 (linha 129-139 do `TODO-findings.md`).

### T2.S1.1 — Análise de equivalência das árvores de redução por lane ✅ [DONE]

**Conclusão (2026-07-18):** BIT-EXACT confirmado. Ambos os kernels implementam a mesma FMA `w_lane * x`, mesma ordem de iteração `i=0..len-1`, mesmo split 4-way por `i%4` e mesma árvore `(acc0+acc1)+(acc2+acc3)`. Cada lane SIMD evolui independentemente — o resultado por lane é idêntico. O comentário de arquitetura foi gravado no topo de `dot_f32_avx2.rs` (16x). Nenhuma linha de produção foi alterada. Gate completo: lints ✓, tests-quick ✓, regression_gate ✓, quality-dashboard ✓.

**Responsável:** Engenheiro sênior de DSP / Arquiteto de kernels AVX2.

**Contexto técnico:**

O `dot_product_4x_f32_dual_accumulate` processa CH=12 em 3 blocos de 4 lanes seriais. O
`dot_product_16x_f32_dual_accumulate` processa todas as 16 lanes (12 reais + 4 padding de zeros) em
1 bloco. A questão central: para cada `lane ∈ [0..11]`, a soma `Σ_{tap,ic} w[tap*IN*16 + ic*16 + lane] * x[tap*IN + ic]` no kernel 16-wide produz o mesmo resultado que `Σ_{tap,ic} w[tap*IN*4 + ic*4 + (lane%4)] * x[tap*IN + ic]` no kernel 4-wide?

**Passos de análise:**

1. **Ler o kernel 4x dual:** `src/math/gemm/dot_4x/dot_f32_avx2.rs` — mapear a estrutura
   `acc[frame][block_idx]` (4 acumuladores por frame, 1 por lane do bloco de 4), a ordem de FMAs
   `for k in 0..K { for ic in 0..IN { fma(w[k*IN*4+ic*4+lane], x[k*IN+ic], acc[lane]) } }`,
   e a árvore de redução final (se existir soma entre acumuladores de diferentes iterações `k`).

2. **Ler o kernel 16x dual (pós-EPIC-1 two-pass):** `src/math/gemm/dot_16x/dot_f32_avx2.rs` —
   mapear as 2 passadas (lo/hi), a ordem de FMAs por lane dentro de cada ymm, e a árvore de
   redução. O resultado final de cada lane é independente das outras 15?

3. **Construir a prova formal (ou contra-exemplo):** Se ambos os kernels implementam
   `acc_lane += w_lane * x` na mesma ordem de iteração `(k=0..K, ic=0..IN)`, com o mesmo esquema
   de acumulação 4-way por tap e a mesma árvore de redução `(acc0+acc1)+(acc2+acc3)`, o resultado
   é bit-idêntico — documentar a prova. Se divergirem (e.g., o 16x agrupa os 4 acumuladores
   `acc_{k%4}` em ordem de `k`, enquanto o 4x os agrupa de outra forma), documentar o delta de ESR
   esperado e levar ao PO antes de prosseguir.

4. **Registro de decisão:** criar comentário de arquitetura no topo de `dot_f32_avx2.rs` (16x)
   documentando a equivalência por lane e a invariante de padding-zero.

**Artefato de saída:** comentário de arquitetura inline + decisão registrada se resultado for
não-bit-exact (nesse caso: parar S2 e levar ao PO).

**Gate desta tarefa:** peer review do engenheiro de DSP + `cargo check` (sem mudanças de produção).

---

## Sprint S2 — F-L4: Coluna µs/MMAC no quality-dashboard (guarda-corpo permanente)

> **Objetivo:** instalar a métrica de eficiência por MAC no dashboard **antes** de fazer qualquer
> mudança de produção no Lite, para que o ganho seja imediatamente visível e para que degenerações
> futuras sejam detectadas automaticamente.
>
> **Referência:** F-L4 (linha 379-390 do `TODO-findings.md`). Independente dos demais passos;
> pode ser feita em paralelo com S1 se houver dois engenheiros disponíveis.

### T2.S2.1 — Implementar coluna µs/MMAC no performance dashboard ✅ [DONE]

**Conclusão (2026-07-18):** Colunaµs/MMAC adicionada em `utils/quality-dashboard.sh`. Novo template de comparação com baseline salvo instalado. Gate completo: lints ✓, tests-quick ✓, regression_gate ✓, quality-dashboard ✓. A métrica está disponível para todos os épicos futuros.

**Responsável:** Engenheiro de tooling / DevOps de qualidade.

**Arquivo-alvo:** [`utils/quality-dashboard.sh`](./utils/quality-dashboard.sh) — seção
`⚡ PERFORMANCE`.

**Especificação técnica:**

1. **Tabela de MACs por bloco por SKU** (constante, calculada analiticamente):

   - WaveNet Standard CH16: `2 * arrays * dilations * CH^2 * K` = `2 * 10 * 16^2 * 3` = 15.360 MAC
   - WaveNet Feather CH8: `2 * 10 * 8^2 * 3` = 3.840 MAC
   - **WaveNet Lite CH12: `2 * 10 * 12^2 * 3`** = 8.640 MAC ← coluna mais importante
   - WaveNet Nano CH4: `2 * 10 * 4^2 * 3` = 960 MAC
   - A2 Full CH8: calcular a partir de `config.layers` (idem para demais A2/LSTM/ConvNet)

   > **Atenção:** os valores exatos de MACs devem ser confirmados lendo a topologia em
   > `src/loader/dispatcher/wavenet/` e `docs/architecture.md` antes de hardcodar no script.
   > Usar comentário `# MACs confirmados em <data> via topology review` junto ao valor.

2. **Nova coluna na tabela de performance:** `µs/MMAC` = `latência_mediana_µs / (MACs / 1e6)`.
   Formato: duas casas decimais, unidade `µs/MMAC`.

3. **Gate suave de eficiência:** após imprimir a tabela, calcular a mediana de eficiência dos
   SKUs WaveNet e emitir `⚠ WARN` (não falha) se algum SKU destoar >2× da mediana:

            ```bash
            # Pseudocódigo — adaptar ao estilo bash do script
            MEDIAN_EFF=$(median of all µs/MMAC values for WaveNet family)
            for sku in wavenet_skus:
                  if sku_eff > 2 * MEDIAN_EFF:
                  echo "⚠  WARN: $sku eficiência ${sku_eff} µs/MMAC — outlier >2× mediana (${MEDIAN_EFF})"
            ```

4. **Integrar no `--check`:** o warn de eficiência deve aparecer mesmo no modo `--check` com o
   `quality-contract.txt` existente (não é gate rígido, apenas visibilidade).

5. **Atualizar `docs/quality-contract.txt`** após instalar a coluna (`--save`).

      **Gate desta tarefa:**

            ```bash
            utils/lints.sh
            utils/quality-dashboard.sh --save docs/quality-contract.txt  # nova coluna presente
            utils/quality-dashboard.sh --check docs/quality-contract.txt # ⚠ warn visível para Lite CH12
            ```

      Resultado esperado antes do fix: Lite CH12 deve aparecer como outlier (≈7,3 µs/MMAC vs
      Standard ≈2,4 µs/MMAC — diferença de ~3×, acima do limiar de 2×).

---

## Sprint S3 — Loader: pad-to-16 para CH=12 no `select_interleave_width` e `read_conv1d_weights_typed`

> **Objetivo:** modificar o loader para que `out_ch=12` passe a usar layout 16-wide com 4 lanes de
> padding zero, roteando para o kernel 16-wide do EPIC-1 (sem criar novo código de inferência).
>
> **Pré-requisito:** T2.S1.1 concluída com prova de bit-exactidão (ou decisão de re-baseline aprovada pelo PO).
>
> **Referência:** F-P2 (linha 119-149 do `TODO-findings.md`).

### T2.S3.1 — `select_interleave_width`: retornar 16 para out_ch == 12 🟡 (CRÍTICO) ✅ CONCLUÍDO 2026-07-18

      **Conclusão:** `select_interleave_width(12)` agora retorna 16, roteando CH=12 para o
      kernel 16-wide do EPIC-1 com 4 lanes zeradas. Ajustes cascata:
      - `layout.rs:346`: `|| out_ch == 12` adicionado à condição do branch 16-wide.
      - `select_interleave_width` promovido de `pub(crate)` → `pub` + re-export via `wavenet/mod.rs`.
      - `transpose_conv1d_interleaved_8wide`/`16wide` re-exportados para uso em testes de integração.
      - `wavenet_ch12_diagnostic_test.rs`: referência escalar e layout de pesos atualizados.
      - `dynamic_parity_test.rs`: `make_conv1d_weights` e `make_conv1d_dyn` usam `select_interleave_width`.
      - `wavenet_lite_block_invariance.rs` (integr): helper `make_conv_weights` usa dispatch por largura.
      - `store_16_accums` (conv_input.rs:64-78): já lidava corretamente com `out_n < 16` via branch else.
      - Todos os 1322 testes passam (1126 lib + 196 integração).
      - Benchmark `RT_WaveNet_Lite_CH12` executa via caminho 16-wide (baseline antiga 4-wide mostra
        regressão esperada; re-baseline necessário).

      **Responsável:** Engenheiro sênior de Rust / DSP.

      **Arquivo-alvo:** [`src/loader/dispatcher/wavenet/layout.rs:346-354`](./src/loader/dispatcher/wavenet/layout.rs#L346-L354).

      **Mudança:**

      ```rust
      // ANTES:
      pub(crate) fn select_interleave_width(out_ch: usize) -> usize {
      if out_ch.is_multiple_of(16) {
            16
      } else if out_ch.is_multiple_of(8) {
            8
      } else {
            4
      }
      }

      // DEPOIS:
      pub(crate) fn select_interleave_width(out_ch: usize) -> usize {
      if out_ch.is_multiple_of(16) || out_ch == 12 {
            // CH=12: pad-to-16 — 4 lanes zeradas, kernel 16-wide do EPIC-1.
            // Custo: +33% de memória de pesos conv (~0,6 KB/camada) — irrelevante para L1/L2.
            // Paridade: bit-exact se a árvore de redução por lane do 16x == 4x (ver T2.S1.1).
            16
      } else if out_ch.is_multiple_of(8) {
            8
      } else {
            4
      }
      }
      ```

**Impacto cascata no loader:** `read_conv1d_weights_typed` chama `select_interleave_width` e já
possui o branch `16 => transpose_conv1d_interleaved_16wide(...)`, que por sua construção
(`for lane in 0..16 { if out_c < out_ch { ... } else { 0.0 } }`) já produz padding zero correto
para CH=12 (12 lanes reais + 4 lanes zeradas). **Nenhuma outra mudança é necessária no loader.**

**Impacto em `conv1d_dual.rs::process_dual_frame_with_mixin`:** `select_interleave_width(12)`
agora retorna 16, logo `num_blocks = 12.div_ceil(16) = 1` e a iteração entra no branch `16 =>`.
O `store_16_accums(out_frame, 0, r, 12)` usa o caminho de `else` (scalar por lane) para os 4
últimos índices — correto, pois `out_n = 12` e `out_c + 15 = 15 >= 12`. **Confirmar que `store_16_accums` já lida corretamente com `out_n < 16`** — verificar [`src/models/wavenet/conv_input.rs:64-78`](./src/models/wavenet/conv_input.rs#L64-L78).

> ✅ **Verificado:** `store_16_accums` usa o branch `else` (escalar lane-a-lane com bounds check)
> quando `!out_n.is_multiple_of(16)` e `out_c + 15 >= out_n`. Para CH=12, `out_n=12`,
> `out_c=0` (único bloco), `out_c + 15 = 15 >= 12` → branch escalar correto. As 12 primeiras
> lanes são escritas, as 4 últimas são ignoradas (não há memória além do buffer de 12 floats).

**Impacto em `conv1d.rs::process_single_frame`:** verificar se este método também usa
`select_interleave_width` e se possui `store_16_accums` — corrigir se necessário para consistência.

**Gate desta tarefa:**

      ```bash
      utils/lints.sh
      utils/tests-quick.sh   # goldens do Lite devem manter ESR ~1e-12 (bit-exact esperado)
      cargo bench --bench regression_gate -- RT_WaveNet_Lite_CH12  # latência deve cair
      ```

---

### T2.S3.2 — Verificação de consistência: loader NAMB (interleaved4) para CH=12 🟢 [DONE]

**Responsável:** Engenheiro de Rust (pode ser mesmo da T2.S3.1).

**Arquivo-alvo:** [`src/loader/dispatcher/wavenet/layout.rs:27-47`](./src/loader/dispatcher/wavenet/layout.rs#L27-L47) — branch `if interleaved`.

**Contexto:** O loader NAMB usa pesos pré-interleaved em 4-wide (`cursor.is_interleaved4()`). Quando
`width=16` (novo comportamento pós-T2.S3.1), o branch chama `transpose_4wide_to_16wide(&tmp, &mut weights, ...)`.
Verificar que `transpose_4wide_to_16wide` já lida com `out_ch=12` corretamente: `num_blocks_4 = 3`
(3 blocos de 4), loop externo `for b in 0..out_ch.div_ceil(16) = 1`, os últimos 4 `src_b = 3`
ficam fora de `num_blocks_4 = 3` → `dst[...+12..15] = 0.0` (padding correto).

> **Verificar:** `transpose_4wide_to_16wide` usa `for b in 0..out_ch.div_ceil(16)` (linha 297)
> e `src_b = 4*b + lane_group`. Para `b=0` e `out_ch=12`:
>
> - `src_b ∈ {0,1,2,3}`: apenas `{0,1,2}` têm `src_b < num_blocks_4 = 3`; `src_b=3` → padding 0.0. ✅

**Passos:**

1. Criar um teste unitário em [`src/loader/dispatcher/wavenet/layout.rs`](./src/loader/dispatcher/wavenet/layout.rs) (no módulo `#[cfg(test)]`) que verifique:
   - `select_interleave_width(12) == 16` (regressão do novo comportamento)
   - `transpose_conv1d_interleaved_16wide(raw, weights, 12, 12, 3)` produz `weights[lane]` para lane 12..15 == 0.0
   - `transpose_4wide_to_16wide(src, dst, 12, 12, 3)` produz padding correto

      **Gate desta tarefa:**

            ```bash
            utils/lints.sh
            cargo test -p nam-rs -- loader::dispatcher::wavenet::layout   # novos unit tests
            ```

---

## Sprint S4 — Validação completa de paridade e performance do Lite CH12

> **Objetivo:** confirmar que o Lite com kernel 16-wide atende a **todos** os contratos de paridade,
> atinge a meta de latência e que o F-L4 (µs/MMAC) já não aponta mais Lite como outlier.

### T2.S4.1 — Revalidação de goldens bit-exact do Lite 🟡 (CRÍTICO — decisão de re-baseline)

**Responsável:** Engenheiro de QA / DSP (com autoridade de contatar PO se necessário).

**Contexto:** O golden do Lite no `quality-dashboard` mede ESR via arquivo `BossWN-lite.nam` com
prewarm de 24k amostras. Após a mudança do loader (T2.S3.1), o resultado numérico pode ser:

- **Caso A (esperado): bit-exact** → ESR permanece ~1e-12, contrato mantido sem re-baseline.
- **Caso B (improvável): divergência dentro do tier** → ESR muda mas permanece em tier `1e-11..1e-14`. Nesse caso: **parar**, documentar o delta, contatar PO para decisão explícita de re-baseline antes de prosseguir.

**Passos:**

1. Executar o gate completo:

            ```bash
            utils/lints.sh
            utils/tests-quick.sh
            utils/quality-dashboard.sh --check docs/quality-contract.txt
            ```

2. Observar o ESR do Lite na tabela de fidelidade. Comparar com o contrato atual.

3. **Se bit-exact (Caso A):** prosseguir para T2.S4.2 sem ação adicional.

4. **Se divergente (Caso B):** registrar o delta, criar um documento de decisão em `docs/` com:

   - Delta de ESR (antes vs depois)
   - Tier de audibilidade (verde/amarelo/vermelho)
   - Justificativa técnica (diferença na árvore de redução)
   - Aguardar aprovação do PO antes de re-baseline.

**Gate desta tarefa:** contrato integralmente aprovado (sem regressão) ou decisão de PO documentada.

---

### T2.S4.2 — Benchmark dirigido e atualização de baseline de regressão 🟢

**Responsável:** Engenheiro de performance / tooling.

**Passos:**

1. Executar benchmark do Lite e verificar a meta:

            ```bash
            taskset -c 0 cargo bench --bench regression_gate -- RT_WaveNet_Lite_CH12
            ```

   **Meta:** ≤ 38 µs (conservador). Alvo ideal: ~30-35 µs.

2. Executar benchmark completo de regressão para garantir que nenhum outro SKU regrediu:

            ```bash
            taskset -c 0 cargo bench --bench regression_gate -- --baseline ci-baseline
            ```

3. Se o Lite atingiu a meta e nenhuma regressão foi detectada, **salvar novo baseline**:

            ```bash
            utils/tests-performance-regression.sh --save
            ```

4. Executar o dashboard com a nova coluna µs/MMAC e verificar que o Lite não é mais outlier:

            ```bash
            utils/quality-dashboard.sh --save docs/quality-contract.txt
            ```

   **Resultado esperado:** Lite CH12 µs/MMAC ≈ 2,5-3,5 µs/MMAC (vs Standard ≈ 2,4 µs/MMAC),
   sem trigger do gate suave de eficiência.

**Gate desta tarefa:**

- `RT_WaveNet_Lite_CH12 ≤ 38 µs` confirmado pelo Criterion
- Nenhuma regressão em todos os outros SKUs
- Dashboard atualizado e contrato salvo

---

### T2.S4.3 — Inspeção do `dsp_hotpath.asm` para o Lite 🟢

**Responsável:** Engenheiro de microarquitetura / Arquiteto de kernels.

**Contexto:** Após a mudança do loader, o Lite CH12 roda pelo mesmo kernel 16-wide do Standard.
Verificar no `dsp_hotpath.asm` que o Lite não introduz novos spills (o kernel 16-wide foi
saneado no EPIC-1, mas vale confirmar que o path do Lite com `num_blocks=1` segue o mesmo código).

**Passos:**

1. Gerar novo `dsp_hotpath.asm` com workload Lite (se o `build-release.sh` suportar seleção de
   modelo — verificar; caso contrário, usar `perf annotate` manual para `WaveNetLayer<1,12,3>::process_block_internal`).

2. Confirmar ausência de `vmovups %ymmN, N(%rsp)` (spills para pilha) no loop interno do Lite.

3. Documentar os contadores de spill encontrados (esperado: 0 ou insignificante — o Lite com
   1 bloco de 16 usa a mesma estrutura de registradores do Standard).

**Gate desta tarefa:** confirmação textual no PR ("0 spills no loop interno do Lite CH12").

---

## Sprint S5 — (Bônus) Head CH12 → HEAD=6: medir impacto do gemv mascarado

> **Objetivo:** investigar o bônus identificado em F-P2 (linha 144-145): o head do Lite usa `gemv`
> com `out_len=6 < 8` (caminho mascarado). Medir se há ganho adicional após a fix principal.
>
> **Condição de execução:** esta sprint só deve ser executada se a latência pós-T2.S4.2 ainda
> estiver acima de 38 µs, **ou** se o PO desejar investigar o head independentemente do gate.

### T2.S5.1 — Medir contribuição do head gemv mascarado para CH12/HEAD=6 🟢

**Responsável:** Engenheiro de DSP / performance.

**Passos:**

1. Isolar a contribuição do head no benchmark: usar `inference_bench` com instrumentação interna
   ou criar um micro-benchmark que exercite apenas o `post_stack_head` do Lite
   (`PostStackHead<12, 6>` → `gemv_no_bias_f32_avx2` com `in_len=12, out_len=6`).

2. Medir a latência do head isolado. Se ≤ 2 µs: o impacto é marginal e pode ser diferido.

3. Se > 2 µs: criar tarefa separada para kernel `broadcast_scale_add_f32_avx2` com `OUT=6`
   (análogo ao F-P4, que está no EPIC-1/S4), derivar o ganho e abrir nova tarefa no backlog.

**Gate desta tarefa:** decisão documentada (defer ou tarefa no backlog).

---

## Critérios de Aceite do EPIC-2

| Critério                                           | Meta                                    | Método de verificação                   |
| -------------------------------------------------- | --------------------------------------- | --------------------------------------- |
| Latência WaveNet Lite CH12                         | **≤ 38 µs** (alvo ideal: ~30-35 µs)     | `cargo bench --bench regression_gate`   |
| ESR do Lite                                        | Mantido em tier (sem regressão audível) | `utils/quality-dashboard.sh --check`    |
| Nenhum outro SKU regrediu                          | 0 regressões em 9 outros SKUs           | `utils/tests-performance-regression.sh` |
| Coluna µs/MMAC no dashboard                        | Presente, Lite não é mais outlier >2×   | `utils/quality-dashboard.sh`            |
| `dsp_hotpath.asm` Lite sem spills no loop interno  | 0 spills confirmados                    | Inspeção manual do asm                  |
| `select_interleave_width(12) == 16`                | Teste unitário passando                 | `cargo test -- layout`                  |
| Padding zero correto para CH=12 em ambas as formas | Testes unitários em `layout.rs`         | `cargo test -- layout`                  |

---

## Tabela-resumo de Tarefas EPIC-2

| Sprint | Tarefa                                          | Finding | Arquivo(s)                      | Risco | Bit-exact     | Gate                  |
| ------ | ----------------------------------------------- | ------- | ------------------------------- | ----- | ------------- | --------------------- |
| S1     | T2.S1.1 — Prova equivalência 4x vs 16x por lane | F-P2    | `dot_f32_avx2.rs` (4x e 16x)    | 🟢    | N/A (análise) | Peer review           |
| S2     | T2.S2.1 — Coluna µs/MMAC no dashboard           | F-L4    | `utils/quality-dashboard.sh`    | 🟢    | N/A           | Dashboard + check     |
| S3     | T2.S3.1 — `select_interleave_width(12) = 16`    | F-P2    | `layout.rs:346-354`             | 🟡    | ✅            | tests-quick           |
| S3     | T2.S3.2 — Verificação loader NAMB interleaved4  | F-P2    | `layout.rs` (testes unitários)  | 🟢    | ✅            | cargo test            |
| S4     | T2.S4.1 — Revalidação goldens bit-exact do Lite | F-P2    | (contrato existente)            | 🟡    | ✅ ou PO      | quality-dashboard     |
| S4     | T2.S4.2 — Benchmark e atualização de baseline   | F-P2    | `regression_gate.rs` / baseline | 🟢    | ✅            | ≤ 38 µs confirmado    |
| S4     | T2.S4.3 — Inspeção `dsp_hotpath.asm` do Lite    | F-P2    | `dsp_hotpath.asm`               | 🟢    | N/A           | Asm sem spills        |
| S5     | T2.S5.1 — (Bônus) Head gemv mascarado HEAD=6    | F-P2    | `post_stack_head.rs` / `gemv`   | 🟢    | ✅            | Decisão defer/backlog |

**Ganho estimado:** Lite CH12 de **63,3 µs → ≤ 38 µs** (−40…50%). Eficiência µs/MMAC equiparável ao Standard.

---

*Criado por `planejador-arquiteto` em 2026-07-18. Baseado na auditoria registrada em
[`TODO-findings.md`](./TODO-findings.md) (EPIC-2, linha 493 e F-P2, linha 117; F-L4, linha 379).*
