<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO-sprints — Refatoração Estrutural (Rust) do nam-rs

> **Objetivo do plano:** decompor os arquivos-fonte grandes (≥ 300 linhas) em módulos menores, atômicos, modulares e bem organizados — **sem alterar lógica
> nem algoritmos** — preservando rigorosamente a **segurança de tempo real (RT-Safety)** e a performance SIMD. Cada tarefa abaixo foi escrita para ser
> executada **isoladamente por um agente de IA**, sem dependência de contexto de outras tarefas.

---

## 0. Como usar este documento

Cada **Tarefa Técnica** é autocontida e define:

- **Arquivo(s)-alvo** e LOC atual.
- **Problema** estrutural a resolver.
- **Plano de split** concreto (novos arquivos e o que migra para cada um).
- **Código morto / comentários obsoletos** a remover (com referência de linha).
- **Invariantes de RT-Safety** a preservar.
- **Critérios de Aceite** (DoD).

### 0.1 Regras obrigatórias (valem para TODAS as tarefas)

Consulte `.agents/rules/rust.md` e `.agents/rules/testing.md`. Resumo vinculante:

1. **Zero mudança de lógica.** É uma refatoração *estrutural*. Nenhum algoritmo, constante numérica, ordem de acumulação (Kahan), `Ordering` atômico, `target_feature`, `#[inline(always)]`/`#[cold]` ou `#[repr(align(128))]` pode ser alterado. Regressões são proibidas.
2. **Mover, não reescrever.** Ao migrar um item para um novo arquivo, copie o corpo **verbatim** (incluindo comentários `// SAFETY:` e doc-comments).
3. **Preservar a superfície pública.** Os caminhos `crate::...` usados por outros módulos não podem quebrar. Use `pub use` / `pub(crate) use` em `mod.rs` para re-exportar os símbolos movidos e manter os imports existentes funcionando.
4. **Convenção de testes (`testing.md`).** Todo novo arquivo `_test.rs` recebe o cabeçalho SPDX/Copyright. Arquivos de produção ≥ 300 linhas mantêm os testes em `*_test.rs` via `#[cfg(test)] #[path = "x_test.rs"] mod x_test;`.
5. **RT-Safety inviolável.** Em qualquer caminho hot (`process`, kernels SIMD, callback RT): proibido heap alloc, `Box`/`Vec`/`Arc` saindo de escopo, locks, `println!`/`format!`, I/O, ou `panic!`/`unwrap!`/`expect!`. Mantenha alocações apenas em caminhos `#[cold]` (setup/`activate`/`new`/load).
6. **Validação final (gate de conclusão).** A tarefa só está concluída quando os dois scripts rodam **sem warnings nem erros**:

   ```bash
   bash utils/lints.sh
   bash utils/tests-cargo.sh
   ```

   `lints.sh` cobre `cargo fmt` + `cargo check`/`clippy -D warnings` em 4 perfis de features (standalone, pure-core, clap-plugin, all-features).

### 0.2 Definition of Done (DoD) padrão por tarefa

- [ ] Split aplicado conforme o plano; arquivos novos com cabeçalho SPDX.
- [ ] Re-exports em `mod.rs` mantêm todos os caminhos de import inalterados.
- [ ] Nenhum arquivo de produção resultante ≥ 300 linhas (ou justificativa de coesão registrada na própria tarefa quando o split é desaconselhado).
- [ ] `git diff` não revela mudança semântica (apenas movimentação + re-export).
- [ ] `bash utils/lints.sh` verde.
- [ ] `bash utils/tests-cargo.sh` verde.

### 0.3 Paralelismo e coordenação

As Sprints 1–4 atuam em subsistemas independentes (`math/`, `models/`, `clap/`, `dsp`+`loader`+`common`+`standalone`) e **podem ser executadas em paralelo** por agentes distintos.
**Conflito a evitar:** dentro de um mesmo subsistema, várias tarefas editam o mesmo `mod.rs` (re-exports).
Recomenda-se que, por subsistema, as tarefas sejam serializadas **ou** que cada agente toque apenas o bloco de re-export do símbolo que moveu.

---

## ÉPICO 0 — Conformidade de testes inline (CONCLUÍDO ✅)

> Já executado nesta iteração. Registrado para rastreabilidade. Extração de
> blocos `#[cfg(test)] mod tests { ... }` inline para arquivos `*_test.rs`
> (exigência de `testing.md` para arquivos ≥ 300 linhas).

| ID     | Arquivo                                          | Ação                                                                                  | Status |
| ------ | ------------------------------------------------ | ------------------------------------------------------------------------------------- | ------ |
| S0.T01 | `src/dsp/adaptive.rs` (647→329)                  | Testes → `adaptive_test.rs`                                                           | ✅     |
| S0.T02 | `src/loader/namb.rs` (497→342)                   | Testes → `namb_test.rs`                                                               | ✅     |
| S0.T03 | `src/models/a2/activations.rs` (339→162)         | Testes → `activations_test.rs`                                                        | ✅     |
| S0.T04 | `src/clap/factory/preset_discovery.rs` (309→254) | Testes → `preset_discovery_test.rs`                                                   | ✅     |
| S0.T05 | `src/clap/plugin/shared.rs` (425→336)            | `make_test_shared` + `layout_tests` → `shared_test.rs` (com re-export `#[cfg(test)]`) | ✅     |

Baseline `lints.sh` + `tests-cargo.sh` confirmados verdes antes e depois.

---

## ÉPICO 1 — Subsistema Math (`src/math/`) (CONCLUÍDO ✅)

> Princípio: **agrupar kernels SIMD por ISA** (AVX2 / AVX-512 / AVX-512BF16) e
> isolar fallbacks escalares. Macros geradoras de v-table (boilerplate) separadas
> por variante. `mod.rs` re-exporta para preservar caminhos
> `crate::math::gemm::*` / `crate::math::common::*`.

### Sprint 1.A — GEMM/GEMV (kernels grandes) (CONCLUÍDO ✅)

#### S1.T01 — Dividir `src/math/gemm/gemv.rs` (1138 LOC) por precisão+ISA (CONCLUÍDO ✅)

- **Problema:** maior arquivo do crate; mistura 2 famílias de precisão (f16, f32
  nativo) e 2 ISAs (AVX2, AVX-512), além de uma macro de kernel compartilhada.
- **Split proposto** (criar diretório `gemm/gemv/`):
  - `gemv/kernel_macro.rs` ← `macro_rules! gemv_kernel` (L22–198).
  - `gemv/f16_avx2.rs` ← `fused_add_gemv_avx2`, `gemv_overwrite_avx2` (L211–307).
  - `gemv/f16_avx512.rs` ← `*_avx512_small`, `gemv_overwrite_avx512`,
    `gemv_overwrite_batch_avx512`, `fused_add_gemv_avx512` (L316–588).
  - `gemv/f32_avx2.rs` ← `gemv_with_bias_f32_avx2`, `gemv_no_bias_f32_avx2`
    (L600–833).
  - `gemv/f32_avx512.rs` ← `gemv_with_bias_f32_avx512`,
    `gemv_no_bias_f32_avx512` (L842–1138).
  - `gemv/mod.rs` ← `pub use` de todos os símbolos (consumidores:
    `avx2_impl.rs`, `avx512/gemv.rs`).
- **Atenção:** a macro `gemv_kernel` precisa ser visível aos arquivos que a usam
  (`#[macro_use]` no `mod.rs` ou caminho explícito). Validar expansão idêntica.
- **RT-Safety:** sem alocação; buffers de cauda `[0.0f32; 8]`/`[…;16]` na pilha;
  manter `get_unchecked` e contratos de comprimento de slice; manter hints de
  prefetch pareados ao passo do loop.
- **DoD:** padrão §0.2.

#### S1.T02 — Dividir `src/math/gemm/gemm_batch.rs` (528 LOC) por ISA (CONCLUÍDO ✅)

- **Split:** criar `gemm_batch/`:
  - `gemm_batch/avx2.rs` ← `fused_add_gemm_batch_avx2`,
    `fused_gemm_residual_batch_avx2` (L22–248).
  - `gemm_batch/avx512.rs` ← `fused_add_gemm_batch_avx512`,
    `fused_gemm_residual_batch_avx512` (L256–528).
  - `gemm_batch/mod.rs` ← re-export.
- **RT-Safety:** sem alocação; `get_unchecked`; **preservar** a ausência de guarda
  `num_frames == 0` nas variantes residuais (não introduzir mudança de
  comportamento — apenas registrar como risco latente herdado).
- **DoD:** padrão.

#### S1.T03 — Dividir `src/math/gemm/gemv_4gate.rs` (401 LOC) por ISA/feature (CONCLUÍDO ✅)

- **Split:** criar `gemv_4gate/`:
  - `gemv_4gate/avx2.rs` ← `gemv_4gate_avx2` (L20–133).
  - `gemv_4gate/avx512.rs` ← `gemv_4gate_avx512` (L140–238).
  - `gemv_4gate/avx512_bf16.rs` ← `gemv_4gate_bf16_avx512` + macro interna
    `bf16_pair` (L244–401).
  - `gemv_4gate/mod.rs` ← re-export.
- **RT-Safety:** manter `core::mem::transmute::<__m512,__m512bh>` intacto; gate
  `target_feature = "avx512bf16"` explícito no arquivo dedicado; sem panics no
  caminho vetorial.
- **DoD:** padrão.

#### S1.T04 — `src/math/gemm/dot_4x/avx512_bf16.rs` (471 LOC): split leve + correção de doc (CONCLUÍDO ✅)

- **Split (opcional, recomendado p/ consistência):** criar `avx512_bf16/`:
  - `helpers.rs` ← `bf16x4_to_f32x4`, `bf16x16_to_f32x16` (L23–36).
  - `single.rs` ← `dot_product_4x_interleaved_avx512_bf16` (L47–203).
  - `dual.rs` ← `dot_product_4x_interleaved_dual_frame_avx512_bf16` (L211–471).
  - `mod.rs` ← re-export.
- **Correção obrigatória (doc obsoleto):** o doc de módulo (≈L11) cita
  `_mm512_dpbf16_ps`/`_mm512_slli_epi32`, mas este arquivo acumula via
  `_mm512_fmadd_ps` (f32). Corrigir o comentário (não o código).
- **RT-Safety:** manter clamp `len = min(...)` (L51, L216); saída `[0.0;4]` na
  pilha; prefetch afinado ao passo.
- **DoD:** padrão. Se o split for considerado desnecessário, registrar
  justificativa de coesão e entregar **apenas** a correção de doc.

### Sprint 1.B — DSP/Ativações/Acumulação (Math) (CONCLUÍDO ✅)

#### S1.T05 — Dividir `src/math/dsp/gain.rs` (420 LOC) por camada/ISA (CONCLUÍDO ✅)

- **Split:** criar `dsp/gain/`:
  - `gain/mod.rs` ← wrappers de dispatch + wrappers seguros
    (`apply_gain_simd`, `apply_ramp_simd`) + include `gain_test`.
  - `gain/avx2.rs` ← 5 kernels `*_avx2` (L88–238).
  - `gain/avx512.rs` ← 5 kernels `*_avx512` (L244–416).
- **Atenção:** mover o include `#[path = "gain_test.rs"]` para o novo `mod.rs`
  ajustando o `#[path]` relativo (ex.: `"../gain_test.rs"`), mantendo
  `gain_test.rs` no local atual ou movendo-o junto — preserve a resolução.
- **RT-Safety:** manter early-returns de fast-path; máscaras de cauda AVX-512
  `(1u32 << (len-i)) - 1`; sem panics.
- **DoD:** padrão.

#### S1.T06 — Reorganizar `src/math/wavenet/accumulate.rs` (376 LOC) por ISA (CONCLUÍDO ✅)

- **Problema:** ordenação confusa (AVX-512 dividido em duas regiões L98 e L315).
- **Split:** criar `accumulate/`:
  - `accumulate/avx2.rs` ← os 5 `*_avx2`.
  - `accumulate/avx512.rs` ← os 5 `*_avx512`.
  - `accumulate/scalar.rs` ← os 5 `*_fallback`.
  - `accumulate/mod.rs` ← re-export.
- **Limpeza:** doc de módulo (L11–12) cita "Task 3.4" e arquivos
  `simd/avx2.rs`/`simd/avx512.rs` inexistentes — atualizar/remover nota histórica.
- **RT-Safety:** sem alocação; preservar relações de comprimento nos fallbacks/
  caudas; máscaras AVX-512; `.exp()`/`.tanh()` permanecem apenas na cauda escalar.
- **DoD:** padrão.

#### S1.T07 — Dividir `src/math/common/dispatch.rs` (378 LOC) (CONCLUÍDO ✅)

- **Split:** criar `dispatch/`:
  - `dispatch/instruction_set.rs` ← `enum InstructionSet` (L16–27).
  - `dispatch/config.rs` ← `struct SimdMathConfig` + `impl` + bloco de doc de
    design-debt (L29–162).
  - `dispatch/detect.rs` ← `detect_best_simd` + `static SIMD_MATH` (L170–378).
  - `dispatch/mod.rs` ← re-export.
- **RT-Safety crítico:** a detecção roda **uma vez** via `LazyLock` no boot/
  warm-up — **não** pode disparar primeiro na thread de áudio. O `panic!`
  terminal é fail-fast de boot (não hot-path). Não alterar o ponto de
  inicialização.
- **DoD:** padrão. (Macro p/ deduplicar os 5 literais de config é *fora de
  escopo* — apenas anotar como oportunidade.)

#### S1.T08 — `src/math/activations/tanh.rs` (325 LOC): split leve produção×referência (CONCLUÍDO ✅)

- **Split:** criar `tanh/`:
  - `tanh/production.rs` ← `simd_tanh_avx2`, `simd_tanh_dual_avx2`,
    `simd_tanh_avx512`, `tanh_slice_avx2`, `tanh_slice_avx512`, `tanh` escalar.
  - `tanh/reference.rs` ← `simd_tanh_pade_nr2_avx2/avx512` + bloco doc
    experimental (L213–277/319).
  - `tanh/mod.rs` ← re-export.
- **Não remover** os `*_nr2_*`: são usados por `benches/inference_bench.rs`
  (L216, L537).
- **Correção de doc:** revisar comentários L18–22/L224 que citam
  `simd_tanh_piecewise_avx512` e `_pade_div_*` (verificar existência em
  `activations/experimental/piecewise_tanh.rs`).
- **RT-Safety:** preservar constantes de clamp; sem divisão por zero (denominador
  Padé Q(x²) ≥ 945).
- **DoD:** padrão. Split é opcional — se mantido coeso, entregar só as correções
  de doc.

### Sprint 1.C — V-tables (macros) e definições (coesos) (CONCLUÍDO ✅)

#### S1.T09 — Dividir macros de v-table AVX-512: `avx512/gemv.rs` (717) e `avx512/dsp.rs` (358) (CONCLUÍDO ✅)

- **`avx512/gemv.rs`** → criar `avx512/gemv/`:
  - `base.rs` ← `impl_avx512_gemv` (L4–260).
  - `vnni.rs` ← `impl_avx512vnni_gemv` (L262–487).
  - `vnni_bf16.rs` ← `impl_avx512vnni_bf16_gemv` (L489–713).
  - `mod.rs` ← os 3 `pub(super) use`.
- **`avx512/dsp.rs`** → criar `avx512/dsp/`:
  - `base.rs` ← `impl_avx512_dsp`.
  - `vnni.rs` ← `impl_avx512vnni_dsp`.
  - `vnni_bf16.rs` ← `impl_avx512vnni_bf16_dsp`.
  - `mod.rs` ← re-export.
- **RT-Safety:** manter `#[inline(always)]` em todos os wrappers (mantém o
  hot-path achatado/monomorfizado).
- **Observação:** os arms VNNI e VNNI+BF16 são quase idênticos (delegam a
  `Avx512Math`) — **não** deduplicar agora (mudaria lógica); apenas registrar.
- **DoD:** padrão.

#### S1.T10 — Limpeza dos arquivos Math coesos (sem split) (CONCLUÍDO ✅)

> Estes arquivos **não devem ser divididos** (justificativa de coesão). A tarefa
> é apenas remover comentários obsoletos/duplicados e dead code seguro.

- `src/math/common/traits.rs` (475): trait único (não divisível). Remover doc
  duplicado em `batch_wavenet_head_sum` (L447–456).
- `src/math/common/avx2_impl.rs` (649): bloco `impl` único. Manter alias
  `Avx2VnniMath` (compat de enum/dispatch — documentado).
- `src/math/dsp/stereo/convolution_avx2.rs` (335): família AVX2 coesa. **Não**
  extrair `hsum256` (mudaria inlining). Sem dead code.
- `src/math/common/huge_alloc.rs` (300): responsabilidade única. **RT-Safety
  crítico:** módulo de **alocação** — nunca chamar na thread de áudio; `Drop`
  faz `munmap` (não-RT).
- **DoD:** `lints.sh` + `tests-cargo.sh` verdes; diff contém apenas remoção de
  comentários mortos.

---

## ÉPICO 2 — Subsistema Models (`src/models/`)

> Re-exportar via `wavenet/mod.rs` (mantém `Conv1d`, `Conv1dDyn`, `DenseLayer`,
> `WaveNetModel` etc. com os mesmos caminhos). Padrão de testes já é compliant
> (`#[path="tests.rs"]`).
> **Nota de auditoria (2026-06-07):** Épico auditado. Lints + testes verdes. Splits
> estruturais executados conforme plano. Dead code removido (S2.T01), test-only
> gating aplicado (S2.T02, S2.T06), Aligned64 migrado para `math/common/` (S2.T04).
> **LOC residual acima do limiar de 300 em 4 arquivos — sem justificativa de coesão
> registrada:**
>
> - `conv1d_dyn_kernels.rs` = 336 (S2.T02)
> - `model.rs` = 397 (S2.T03)
> - `layer.rs` (lstm) = 399 (S2.T04)
> - `conv1d.rs` = 320 (S2.T06 — split opcional, coesão implícita)
>
> Recomendação: adicionar justificativa de coesão (inline doc) nesses arquivos,
> similar ao padrão de `conv1d_dual.rs` (S2.T07), ou registrar como risco aceito.
> Não bloqueia os épicos seguintes.

### S2.T01 — Dividir `src/models/wavenet/model_dyn.rs` (626 LOC) (CONCLUÍDO ✅)

- **Split** (espelha o lado estático que já tem `dense.rs`/`model.rs`):
  - `dense_dyn.rs` ← `DenseLayerDyn` + impl (L11–138).
  - `layer_dyn.rs` ← `WaveNetLayerDyn` + `process_block_internal` (L140–334).
  - `model_dyn.rs` (mantém) ← `WaveNetLayerArrayDyn` + `WaveNetDynModel`
    (L336–626).
- **Dead code (remover, após confirmar 0 chamadores via grep no crate inteiro,
  incluindo benches/tests):**
  - `DenseLayerDyn::process_acc_block` (L35–51) — sem chamadores.
  - `DenseLayerDyn::process_fused` (L133–137) — sem chamadores.
- **Limpeza:** normalizar ordem doc/atributo em L29–34 (doc `# Safety` duplicado
  antes/depois de `#[inline(always)]`); tags de ticket obsoletas L300/L407/L457.
- **RT-Safety:** preservar `AlignedMixinBuffer([f32; 8192])` na pilha e o
  `assert!`-guard (L190–195) — é local de panic, manter exatamente; buffers
  `AlignedVec` pré-alocados; `set_max_buffer_size` é o único alocador (off-hot).
- **DoD:** padrão.

#### S2.T02 — Dividir `src/models/wavenet/conv1d_dyn.rs` (584 LOC) (CONCLUÍDO ✅)

- **Split:**
  - `conv1d_dyn.rs` (mantém) ← struct `Conv1dDyn` + wrappers públicos +
    `load_mixin_4` (L23–257).
  - `conv1d_dyn_kernels.rs` ← `process_dual_frame_generic`,
    `process_single_frame_generic`, `process_block_generic` (L261–583).
- **Consolidação de constantes:** mover/unificar `WAVENET_MAX_NUM_FRAMES`,
  `LAYER_ARRAY_BUFFER_PADDING`, `MAX_KERNEL` (L17–21) para `wavenet/common.rs`
  (model_dyn já importa de `super::common`) — eliminar ambiguidade.
- **Dead/test-only:** `process_block`/`process_block_bf16` (L154/L178) só são
  usados por `tests.rs` → **gate `#[cfg(test)]`** (não remover; tests dependem).
- **Limpeza:** header doc L10–11 ("A2 placeholder") obsoleto; mensagens de assert
  em PT (L280/L432) → padronizar idioma.
- **RT-Safety:** kernels sem heap; `debug_assert!` apenas; manter contrato de
  aliasing `from_raw_parts` + `get_unchecked`; manter chamada indireta
  `prefetch_fn`.
- **DoD:** padrão.

#### S2.T03 — Dividir `src/models/wavenet/model.rs` (554 LOC) (CONCLUÍDO ✅)

- **Split** (paralelo a S2.T01 p/ simetria):
  - `layer.rs` ← `WaveNetLayer` + `process_block_internal` (L11–163).
  - `model.rs` (mantém) ← `WaveNetLayerArray` + `WaveNetModel` (L168–554).
- **Limpeza:** remover breadcrumbs obsoletos L21/L165 ("moved to
  wavenet_common.rs"); normalizar ordem doc/atributo (L393–395, L536–539).
- **Auditoria:** confirmar chamadores de `prewarm_avx512`/`prewarm_avx2`
  (L523/L532) antes de qualquer ajuste (podem ser usados por tests/benches).
- **RT-Safety:** preservar buffers de pilha `[0.0f32; 1024]` (L59/L72) e seus
  `const { assert! }` + `assert!` runtime; early-return em input vazio (L456);
  `_mm_prefetch`; sem locks.
- **DoD:** padrão.

#### S2.T04 — `src/models/lstm/layer.rs` (426 LOC): extrair utilitário + gate de referência (CONCLUÍDO ✅)

- **Split:**
  - `aligned.rs` (preferir mover para `src/math/common/` ou `src/common/`) ←
    `Aligned64<T>` + impls `Deref`/`DerefMut`/`Default` (L39–66): primitivo
    genérico, não específico de LSTM.
  - `layer.rs` (mantém) ← `LstmLayer` + macro `define_lstm_process!` +
    instâncias SIMD + fallback escalar + acessores.
  - `scalar_minimax_sigmoid` (L12–37): se só usado por tests/fallback de paridade,
    aplicar `#[cfg(test)]` (confirmar que release/benches não chamam o caminho
    escalar).
- **RT-Safety:** buffers `Aligned64<[..;N]>` const-generic (zero heap);
  `_mm_prefetch`; manter gating `#[target_feature]` exato por instância; sem
  panics no caminho SIMD.
- **DoD:** padrão. Atenção: mover `Aligned64` muda caminho de import — ajustar
  re-exports e os `use` nos consumidores.
- **Nota (S2.T04 executada):** `scalar_minimax_sigmoid` **não** foi movido para
  `#[cfg(test)]` porque o caminho escalar é usado por benchmarks
  (`benches/inference_bench.rs`) e testes de integração
  (`tests/lstm_scalar_bf16_parity.rs`). `Aligned64` foi movido para
  `src/math/common/aligned.rs` e re-exportado via `src/math/common/mod.rs`.
- **LOC final:** `layer.rs` ~371 LOC (antes: 426).

#### S2.T05 — Dividir `src/models/mod.rs` (331 LOC) (CONCLUÍDO ✅)

- **Split:**
  - `mod.rs` (mantém) ← decls de módulo, `sealed`, trait `NamModel`, enum
    `DynamicModel` (L1–102 + impl `Sealed`).
  - `dynamic_model.rs` ← os 2 blocos `impl` (métodos inerentes L104–220 +
    `impl NamModel for DynamicModel` L222–331).
- **RT-Safety:** `process` (L223) deve permanecer `#[inline(always)]` com dispatch
  estático por enum (sem `dyn`/vtable); manter `#[inline(always)]` em
  `set_effective_layers`, `layer_count`, `is_lstm`, `is_wavenet`.
- **DoD:** padrão.

#### S2.T06 — `src/models/wavenet/conv1d.rs` (391 LOC): split opcional + gate de API test-only (CONCLUÍDO ✅)

- **Split opcional:**
  - `conv_input.rs` ← trait `ConvInput` + os 2 `impl ConvInput` (L30–100). É
    contrato compartilhado por `conv1d.rs`, `conv1d_dual.rs`, `conv1d_dyn.rs`.
  - `conv1d.rs` (mantém) ← struct + kernel single-frame + wrappers.
- **Dead/test-only:** a cadeia `process_block` (L375) → `process_single_frame`
  (no-mixin, L119) → `process_single_frame_internal` (L154) só é alcançada por
  `tests.rs`. Aplicar `#[cfg(test)]` à cadeia (não remover).
- **Limpeza:** remover header `//!` duplicado (L4–8 e L10); normalizar ordem
  doc/atributo (L131–135, L334–338).
- **RT-Safety:** manter acumuladores Kahan **bit-a-bit** (L261–264); buffers de
  tap na pilha; `from_raw_parts` + `get_unchecked`; `prefetch_fn`.
- **DoD:** padrão.

#### S2.T07 — `src/models/wavenet/conv1d_dual.rs` (312 LOC): COESO (sem split) (CONCLUÍDO ✅)

- **Ação:** **não dividir** — unidade única (extensão dual-frame Temporal-Tiling
  do `Conv1d`); wrappers finos + 1 kernel dominante. Registrar justificativa.
- **RT-Safety:** buffers de tap na pilha; sem `assert!`/panic no kernel; aliasing
  `from_raw_parts` (L197–200); branch `OUT.is_multiple_of(4)` é const-foldable.
- **DoD:** `lints.sh` + `tests-cargo.sh` verdes; nenhuma mudança além de eventual
  limpeza de comentário.

---

## ÉPICO 3 — Subsistema CLAP (`src/clap/`) (CONCLUÍDO ✅)

> ⚠️ `processor/dsp.rs` e `processor/mod.rs` contêm o **hot-path de áudio do
> plugin**. Os demais (`gui/`, `plugin/main_thread.rs`) rodam na thread de UI/
> main — alocação/lock/log são permitidos lá, **mas leem atômicos RT** cujos
> `Ordering` devem ser preservados.

### Sprint 3.A — GUI (maiores)

#### S3.T01 — Dividir `src/clap/gui/ui/mod.rs` (1194 LOC) — maior arquivo CLAP (CONCLUÍDO ✅)

- **Split (por zona/concern):** criar `gui/ui/zones/` e `gui/ui/status_bar/`:
  - `zones/identity.rs` ← `draw_zone1_identity` (L207–447) + helper
    `spawn_file_dialog(...)` (extrair o bloco `std::thread::spawn`).
  - `zones/controls.rs` ← `draw_zone2_controls` (L449–558).
  - `zones/meters.rs` ← `draw_zone3_meters` (L560–647).
  - `zones/bypass_zone.rs` ← `draw_zone4_bypass` (L649–682).
  - `status_bar/mod.rs` ← `draw_zone5_status_bar` (L1067–1190).
  - `status_bar/telemetry.rs` ← `update_telemetry_state` + `draw_telemetry_strings`
    (L684–908). (Opcional: isolar I/O de diagnóstico em `status_bar/diagnostics.rs`.)
  - `status_bar/metadata.rs` ← `update_metadata_cache` + `draw_metadata_strings`
    (L910–1065).
  - `focus.rs` ← navegação por Tab (L146–202).
  - `mod.rs` (mantém) ← orquestrador `draw_ui` (L66–205) + decls + include
    `#[path="test.rs"]`.
- **Dead code:** remover import morto `use self::knob::knob_widget;` (com
  `#[allow(unused_imports)]`, L43–44) — apenas `handle_knob` é usado aqui.
- **Limpeza:** padronizar comentários PT/EN ("ZONA 1"/"Zona 5"; toast PT L1153).
- **RT-Safety:** thread de UI (não-RT). Preservar leituras atômicas `Relaxed`
  (`ui_peak_l/r`, `current_latency`, `rt_status`) e locks de UI
  (`ui_model_name`, etc.). **Não** mover nada disto para o processor.
- **DoD:** padrão.
- ✅ **Auditoria (ÉPICO 3):** `focus.rs` implementado em `gui/ui/focus.rs` (não
  `gui/ui/zones/focus.rs`) — correto, pois é cross-cutting concern operando sobre
  todas as 4 zonas. `knob_widget` removido. `#[allow(unused_imports)]` mantido
  sobre `use self::bypass::handle_bypass` em `mod.rs:40-41` (necessário para o
  `test.rs` acessar via `use super::*`).

#### S3.T02 — Dividir `src/clap/gui/window/mod.rs` (515 LOC) (CONCLUÍDO ✅)

- **Split:**
  - `lifecycle.rs` ← `NamPluginWindow::new`, `safe_shared`,
    `destroy_gl_resources`, `impl Drop` (L90–257).
  - `frame.rs` ← `impl WindowHandler::on_frame` (L260–364).
  - `event.rs` ← `on_event` + `get_valid_model_file` (L27–39, L371–510).
  - `mod.rs` (mantém) ← `struct NamPluginWindow` + include `#[path="test.rs"]`.
- **RT-Safety:** thread de UI. **Preservar a checagem de `alive_fence` antes de
  cada deref de `*self.shared.0`** (L162–167, L194, L215–223); manter
  pareamento `make_current`/`make_not_current` apesar dos early-returns.
- **DoD:** padrão.
- ✅ **Auditoria (ÉPICO 3):** Decomposição real difere da planejada — adotou
  `handler.rs` (on_frame + on_event), `drag_drop.rs` (get_valid_model_file),
  `shaders.rs`, `input_map.rs` e `mod.rs` (struct + lifecycle + Drop).
  Funcionalmente equivalente. `alive_fence` via `safe_shared()` OK; pareamento
  `make_current`/`make_not_current` preservado em todos os early-returns.

#### S3.T03 — Dividir `src/clap/gui/ui/meter.rs` (358 LOC) por backend de render (CONCLUÍDO ✅)

- **Split:** criar `gui/ui/meter/`:
  - `meter/mod.rs` ← orquestrador `draw_vertical_meter` (label, LED, interação,
    peak-hold/dB, dispatch).
  - `meter/glow.rs` ← caminho GL/glow (L139–269): `render_glow(...)` (closure
    `CallbackFn` + escrita de atômicos + uniforms).
  - `meter/cpu.rs` ← fallback CPU (L270–321): `render_cpu(...)`.
  - (Opcional) `meter/readout.rs` ← texto peak-dB + label (L323–356).
- **RT-Safety:** thread de UI. A closure `CallbackFn` roda no render thread e lê
  `VuMeterSharedState` com `Relaxed` — manter marshalling atômico e a closure
  `'static`/`Send` via `Arc`.
- **DoD:** padrão.

#### S3.T04 — `src/clap/gui/ui/knob.rs` (314 LOC): COESO (split opcional)(CONCLUÍDO ✅)

- **Ação:** preferir **manter coeso** (duas fns: `knob_widget` render puro +
  `handle_knob` wiring). Se atomicidade for priorizada: `knob/widget.rs` +
  `knob/handle.rs` + `knob/mod.rs`.
- **RT-Safety crítico:** preservar o pareamento **store-then-`Release`**:
  `atomic_val.store(..., Relaxed)` seguido de
  `gui_param_generation.fetch_add(1, Release)` (L280–281) — par do `Acquire` em
  `params.rs::flush`. Não reordenar.
- **DoD:** padrão.

### Sprint 3.B — Processor (hot-path) e Plugin

#### S3.T05 — Dividir `src/clap/processor/dsp.rs` (540 LOC) — HOT-PATH(CONCLUÍDO ✅)

- **Problema:** `process_dsp_audio` é uma única fn gigante (L17–539) com forte
  duplicação `#[cfg(feature="stereo")]` / `not(stereo)`.
- **Split (helpers privados `#[inline(always)]`, chamados por um
  `process_dsp_audio` enxuto):** criar `processor/dsp/`:
  - `dsp/bypass.rs` ← `process_bypass(...)` (L37–117).
  - `dsp/gain.rs` ← `apply_input_gain` (L177–243) + `apply_output_gain`
    (L354–395).
  - `dsp/peaks.rs` ← cálculo de peak de saída (L397–489) + helper `store_peaks`
    (padrão repetido 3×: L94–115, L468–489).
  - `dsp/gate_flags.rs` ← mapeamento estado-gate→flags (L289–318).
  - `dsp/telemetry.rs` ← timing/overload/adaptive (L492–517).
  - `dsp/mod.rs` (mantém) ← loop por porta + cache de threshold.
- **⚠️ RT-Safety (CRÍTICO):** **manter helpers `#[inline(always)]`** e passar
  slices por `&mut [f32]` para o otimizador reproduzir o inlining atual;
  **validar ausência de regressão de performance** (rodar `benches/` relevantes
  antes/depois se possível). Sem alloc/lock/panic; manter `Ordering::Relaxed`;
  bloco heap-audit (L519–536) permanece sob `#[cfg(feature="heap-audit")]`.
- **DoD:** padrão **+** verificação de que o diff é movimentação pura (sem
  alterar contas de gain/clip/peak/dither).

#### S3.T06 — Dividir `src/clap/processor/mod.rs` (353 LOC)(CONCLUÍDO ✅)

- **Split:**
  - `state.rs` ← `struct NamClapProcessor` (L32–113).
  - `gc.rs` ← `push_to_gc` (L115–149).
  - `mod.rs` (mantém) ← `activate` + `deactivate` + `process` + wiring + include
    `#[path="../processor_test.rs"]`.
    Nota: `activate`/`deactivate` permanecem em `mod.rs` (não em `lifecycle.rs`)
    porque Rust E0119 proíbe split de `impl PluginAudioProcessor` entre módulos.
- **Dead code:** investigar campos `mod_input_gain`/`mod_output_gain` (L84–86) —
  aparentemente não consumidos no DSP (apenas `mod_gate_thresh`). **Não remover
  sem auditoria** (pode ser feature incompleta); registrar achado.
  - ✅ **Achado (S3.T06 concluído):** `mod_input_gain` e `mod_output_gain` NÃO são
    dead code — são consumidos ativamente em `events.rs` (param modulation CLAP):
    aplicados via `lut.db_to_linear(db + mod_*)` nas linhas 35, 38, 84, 93, 128,
    133, 166, 178. O campo `mod_gate_thresh` também é usado (não apenas ele).
    Nenhum campo removido.
- **⚠️ RT-Safety:** `activate` é o **único** local de alocação — manter fora de
  `process`. Em `process`: `pthread_getschedparam`/`sched_getcpu`/`set_daz_ftz`
  rodam **uma vez** (guardado por `prio_checked`) — preservar one-shot.
  `push_to_gc` deve permanecer alloc/lock-free (array parking-lot + overflow).
- **DoD:** padrão.

#### S3.T07 — Dividir `src/clap/extensions/params.rs` (459 LOC) por trait(CONCLUÍDO ✅)

- **Split:** criar `extensions/params/`:
  - `params/mod.rs` ← consts `PARAM_*` (L20–30) + decls + nota de duplicação
    intencional (L322–334).
  - `params/main_thread.rs` ← `impl PluginMainThreadParams for NamClapMainThread`
    (L32–319).
  - `params/audio_thread.rs` ← `impl PluginAudioProcessorParams for
    NamClapProcessor::flush` (L335–458).
  - (Opcional) `params/info.rs` ← tabela `get_info` (L37–115).
- **⚠️ RT-Safety:** `PluginAudioProcessorParams::flush` roda na audio thread
  (fora de `process`) — só atômicos `Relaxed`/`Acquire`, sem alloc/lock.
  **Preservar o `Acquire`** no generation-guard (L400–457) e o gate "só
  reconcilia quando a geração difere". `write_gui_events` usa `try_push`
  (bounded) — manter (nunca `push`/alloc).
- **DoD:** padrão.
- ✅ **Auditoria (ÉPICO 3):** Arquivos implementados como `main.rs`,
  `audio.rs` e `mod.rs` (nomes simplificados vs `main_thread.rs`/
  `audio_thread.rs` planejados). `info.rs` opcional não criado — tabela
  `get_info` reside em `main.rs:25-103`. RT-Safety preservada: `Acquire` no
  generation-guard pareado com `Release` das writes GUI; gate de geração
  funcional; `try_push` mantido.

#### S3.T08 — Dividir `src/clap/plugin/main_thread.rs` (371 LOC)(CONCLUÍDO ✅)

- **Split:** criar `plugin/main_thread/`:
  - `main_thread/mod.rs` ← `struct NamClapMainThread` + shell do
    `impl PluginMainThread`.
  - `main_thread/housekeeping.rs` ← corpo de `on_main_thread`
    (GC drain/flags/hugepage/latency).
  - `main_thread/logging.rs` ← bloco flag→host-log (L136–200) como
    `emit_pending_logs`.
  - `main_thread/load.rs` ← `load_model` (L216–370) + mapeamento
    `error_code` (L94–110) como helper.
- **RT-Safety:** thread main (alloc/lock/log permitidos). É o **consumidor** de
  telemetria RT (`check_and_clear_flag`) e o local seguro para `drain_gc_channels`
  (drop de boxes alocados pelo RT). Manter `param_tx.push` (SPSC) e **não**
  aproximar `load_and_build_model`/`NamResampler::new` do RT.
- **DoD:** padrão.

#### S3.T09 — `src/clap/plugin/shared.rs` (336 LOC): COESO (sem split) + verificação (CONCLUÍDO ✅)

- **Ação:** **não dividir** — módulo de definição de estado compartilhado, bem
  seccionado. Já conforme em testes (S0.T05). Verificar apenas.
- **⚠️ RT-Safety (fundacional):** **preservar `#[repr(align(128))]`** em
  `RtToUi`/`UiToRt`/`ColdShared` (isolamento de cache-line / anti false-sharing);
  `write_gui_events` só `Relaxed` + `try_push`; `Mutex` de `ColdShared` apenas em
  cold-path; `Drop` seta `alive_fence=false` (ordem preservada).
- **DoD:** `lints.sh` + `tests-cargo.sh` verdes.

✅ **Auditoria ÉPICO 3 (2026-06-08):** Todas as 9 tasks verificadas. RT-safety
preservada em todos os hot-paths (`#[inline(always)]`, `Relaxed`/`Acquire`,
`try_push`, `alive_fence`, `make_current`/`make_not_current`). Três desvios
documentados nos respectivos tasks: S3.T01 (posição de `focus.rs` + `allow`
mantido sobre `handle_bypass`), S3.T02 (decomposição real diferente da planejada,
funcionalmente equivalente), S3.T07 (nomes simplificados, `info.rs` opcional em
`main.rs`). Nenhum dead code novo. Testes e lints verdes.

---

## ÉPICO 4 — DSP / Loader / Common / Standalone

> Mistura caminhos hot (RT) e cold (loader/encoder/diagnóstico). Marcações de RT
> abaixo são vinculantes.

### Sprint 4.A — Common & Diagnostics

#### S4.T01 — Dividir `src/common/diagnostics/diagnostic.rs` (670 LOC) (CONCLUÍDO ✅)

- **Split:**
  - `diagnostic.rs` (mantém) ← `NamDiagnostic` + `Display` + `Error` + `emit*`
    - include `#[path="diagnostic_test.rs"]`.
  - `snapshot.rs` ← `ModelInfo`, `AudioInfo`, `RtInfo`, `TelemetrySnapshot`,
    `RuntimeSnapshot`, trait `HasRuntimeSnapshot` + `impl ... for RtStatusFlags`,
    e os 3 statics globais (L16–23).
  - `bundle.rs` ← `DiagnosticBundle`, `ErrorContext`, `render()`, variantes de
    capture.
  - `format.rs` ← `days_to_date`, `redact_path`, `format_model_path`, timestamp.
- **Limpeza:** comentário PT obsoleto L155; doc auto-redundante L388 (artefato
  copy/paste de `$XDG_RUNTIME_DIR`).
- **RT-Safety:** módulo **cold/diagnóstico** (`#[cold]`); `format!`/`String`/
  `RwLock` OK — **nunca** chamar `render()`/`emit()` dentro do callback RT.
- **DoD:** padrão.

#### S4.T02 — Dividir `src/common/spsc.rs` (438 LOC) (CONCLUÍDO ✅)

- **Split:**
  - `status.rs` ← `SHUTDOWN`, consts `RT_STATUS_*` (L30–60), `RtStatusFlags`.
  - `payload.rs` ← `ParamPayload`.
  - `gc.rs` ← `GcItem`, `GcOverflowBuffer`, `drain_gc_channels`.
  - `spsc.rs` (mantém) ← `SpscChannels`, `setup_spsc`, include
    `#[path="spsc_test.rs"]` + re-exports.
- **Limpeza:** mensagem de panic em PT L271 ("tipo desconhecido").
- **⚠️ RT-Safety:** **preservar `#[repr(align(128))]`** em `RtStatusFlags` e
  `ParamPayload`; ops de flag `#[inline(always)]` `Relaxed`; `GcOverflowBuffer::
  push` é a válvula RT — manter alloc-free (`Box::into_raw` + swap atômico);
  `from_raw_parts` (panic) só roda em `drain()` (main thread).
- **Atenção:** muitos consumidores importam de `crate::common::spsc::*` —
  re-exportar tudo no `mod.rs` para não quebrar caminhos.
- **DoD:** padrão.

### Sprint 4.B — DSP (hot-path) e pipeline

#### S4.T03 — Dividir `src/dsp/pipeline/stages.rs` (425 LOC) por estágio (CONCLUÍDO ✅)

- **Split** (alinhado aos Stages 1–4 documentados): criar `pipeline/stages/`:
  - `stages/input.rs` ← `handle_silence_bypass`, `apply_input_stage`,
    `DISABLE_GATE`, `DENORMAL_DITHER_OFFSET`.
  - `stages/inference.rs` ← `configure_adaptive_model`, `run_stereo_or_mono`,
    `run_inference`.
  - `stages/output.rs` ← `apply_output_stage`, `write_bridge`.
  - `stages/mod.rs` ← re-export + colocar `DENORMAL_DITHER_OFFSET` em local
    compartilhado (usado por input **e** output).
- **⚠️ RT-Safety:** hot-path puro. Manter `#[inline(always)]` em todos os
  estágios e `#[cold]#[inline(never)]` no silence-bypass; `get_unchecked_mut`
  para dither; `dispatch_simd!`; **preservar a simetria inject/compensate de
  denormal** entre input e output.
- **DoD:** padrão.

#### S4.T04 — `src/dsp/mirror_buf.rs` (459 LOC): isolar alocação cold (CONCLUÍDO ✅)

- **Split:**
  - `mirror_buf.rs` (mantém) ← struct + traits (`Deref`/`DerefMut`/`Drop`/
    `Clone`/`Debug`/`Send`/`Sync`), `size()`, globais, enum de status, sync fns,
    include de teste.
  - `alloc.rs` ← `MirroredBuffer::new` (padrão 4KB) + `try_new_huge` (2MB) — o
    cerimonial `#[cold]` de mmap (~270 linhas).
- **⚠️ RT-Safety:** `Deref`/`DerefMut` (`#[inline(always)]`) são o único acesso
  hot — manter pura construção de slice; mmap/`ftruncate` `#[cold]`; `Clone`
  **panica** via `panic_any` (não clonar no RT). Preservar layout de 16 bytes da
  struct e `unsafe impl Send/Sync`.
- **Verificar:** `MirrorHugePageStatus` é `pub` — confirmar uso externo antes de
  tratar como morto.
- **DoD:** padrão.

#### S4.T05 — DSP coesos (sem split): `resampler.rs`, `gate.rs` (CONCLUÍDO ✅)

- **`src/dsp/resampler.rs` (469):** **não dividir** (DelayLine→ResamplerCore→
  NamResampler = unidade algorítmica única). Apenas confirmar verde.
  - RT-Safety: alloc só em `new()` `#[cold]`; `process_*` zero-alloc com
    `get_unchecked`/ponteiros; `#[inline(always)]` em `push`/`window_ptr`.
- **`src/dsp/gate.rs` (379):** **não dividir** (FSM + aplicador de ganho coesos;
  split exporia campos privados).
  - RT-Safety: alloc-free; `#[repr(align(128))]` em `GateParams`;
    `inv_fade_frames` pré-computado (evita divisão no hot-path) — preservar.
- **DoD:** verdes; diff só limpeza de comentário, se houver.

### Sprint 4.C — Loader & Encoder

#### S4.T06 — Dividir `src/loader/nam_json/data.rs` (454 LOC) (CONCLUÍDO ✅)

- **Split:**
  - `model.rs` ← schema: `NamDate`, `NamMetadata`, `NamLayerConfig`, `NamConfig`,
    `NamModelData`, `WeightsLayout`.
  - `error.rs` ← `JsonError` + impls.
  - `validation.rs` (ou `visitors.rs`) ← consts de limite, `WeightsVisitor`,
    `LimitedValueVisitor`, `TrainingOptionVisitor`, `deserialize_weights`,
    `deserialize_training`.
  - `data.rs`/`mod.rs` ← re-export (preservar caminho
    `crate::loader::nam_json::*`).
- **RT-Safety:** caminho de loader (não-RT). **Preservar os caps de segurança**
  (`MAX_WEIGHTS`, profundidade, tamanho) contra DoS.
- **DoD:** padrão.

#### S4.T07 — Dividir `src/loader/namb.rs` (342 LOC) (CONCLUÍDO ✅)

- **Split:**
  - `namb.rs` (mantém) ← `parse_namb` + re-exports + include
    `#[path="namb_test.rs"]` (já conforme).
  - `error.rs` ← `NambError`.
  - `header.rs` ← `NambHeader`, `FLAG_HAS_CRC32`, `validate`, `get_layout`,
    `crc32_ieee`, `check_crc` (⚠️ `crc32_ieee` e `NambHeader` são reusados por
    `namb_encoder.rs` — manter `pub`).
  - `fallback.rs` ← `make_fallback_model_data`, `make_standard_wavenet_config`.
- **RT-Safety:** loader (não-RT); cast `unsafe` de header é bounds-checked.
- **DoD:** padrão.

#### S4.T08 — Dividir `src/loader/namb_encoder.rs` (348 LOC)

- **Split:** criar `loader/transpose/`:
  - `namb_encoder.rs` (mantém) ← `encode_namb`, `transpose_weights` (dispatch),
    `ensure_capacity`.
  - `transpose/lstm.rs` ← `transpose_lstm_gate_major` (L102–172).
  - `transpose/wavenet.rs` ← `transpose_wavenet_interleaved4` (L176–333).
- **Atenção:** os layouts de transposição devem permanecer em lock-step com o
  decoder (`dispatcher/lstm.rs`, comentários L154/L265). Documentar a dependência
  cruzada.
- **RT-Safety:** ferramenta offline (nunca RT).
- **DoD:** padrão.

#### S4.T09 — Dividir `src/loader/dispatcher/lstm.rs` (328 LOC)

- **Split:** criar `dispatcher/lstm/`:
  - `lstm/dispatch.rs` ← `build_lstm` (tabela de match de topologia).
  - `lstm/static_builder.rs` ← `build_lstm_1layer`, `build_lstm_2layer`.
  - `lstm/dynamic_builder.rs` ← `build_lstm_dynamic` (`pub`, usado por parity
    tests — manter visível).
  - `lstm/weights.rs` ← `read_lstm_weights_into`, `read_lstm_layer`.
- **RT-Safety:** caminho de construção (cold/load, off-RT); preservar invariante
  do `from_raw_parts_mut` u16 (L310) bounds-matched a `H4*IH`.
- **DoD:** padrão.

### Sprint 4.D — Standalone (PipeWire)

#### S4.T10 — Dividir `src/standalone/pw_host/rt_callback.rs` (325 LOC) — RT CRÍTICO

- **Split** (alinhado às sub-etapas 5.1.1–5.1.4): criar `pw_host/rt_callback/`:
  - `rt_callback/resampler_swap.rs` ← `drain_resamplers` (L25–78).
  - `rt_callback/commands.rs` ← `receive_commands` (L84–195).
  - `rt_callback/rate_sync.rs` ← `sync_rate` (L200–233).
  - `rt_callback/process.rs` ← `process_dsp_buffer` (L238–325).
  - `rt_callback/mod.rs` ← re-export.
- **Dead code:** remover rebinds no-op `let new_rs = new_rs;` (L34) e
  `new_model_l`/`new_model_r` (L113–114).
- **⚠️ RT-Safety (todo o arquivo roda no callback RT do PipeWire):** manter todas
  as fns `#[inline(always)]`; **zero alloc** (GC usa parking-lot pré-alocado
  `[Option;16]` + overflow ring); sem I/O; sem mutex (só atômicos `Relaxed` +
  rtrb SPSC); `Box::into_raw`/`mem::replace` movem boxes sem dropar no RT.
  **Proibido** introduzir `format!`/log/drop-no-RT em qualquer extração. (Padrão
  de GC-cascade duplicado entre `drain_resamplers`/`receive_commands` — **não**
  extrair agora se implicar custo; apenas registrar.)
- **DoD:** padrão **+** revisão manual de RT-safety no diff.

#### S4.T11 — `src/standalone/pw_host/capture.rs` (312 LOC): extração parcial (RISCO)

- **Problema:** `setup_capture_stream` é monolítica; closures `move` capturam
  ~30 locais — extração é restringida por semântica de captura.
- **Split pragmático (baixa agressividade):**
  - `capture/state.rs` ← struct `CaptureState` agrupando os locais DSP
    inicializados + `init()` (reduz o preâmbulo de ~50 locais, L67–121).
  - `capture/listeners.rs` ← fns livres para `state_changed`/`param_changed`
    (capturam pouco).
  - `capture.rs` (mantém) ← `setup_capture_stream` + a closure `process` (manter
    capturas juntas pelo RT).
- **⚠️ RT-Safety:** a closure `process` (L161–286) é a entrada RT — manter todos
  os buffers como arrays de pilha `[f32; MAX_RESAMP_BUF]` capturados por move
  (zero alloc RT) e `parking_lot: [Option<GcItem>;16]` pré-alocado;
  `configure_realtime_thread` one-shot (L162–165). Qualquer `CaptureState` deve
  manter campos tocados pelo RT na pilha/pré-alocados (sem indireção de heap).
- **DoD:** padrão **+** revisão de RT-safety. Se o risco de captura inviabilizar,
  entregar **apenas** a extração da struct de estado (preâmbulo) e registrar.

### Sprint 4.E — Testing util (coeso)

#### S4.T12 — `src/testing/stress.rs` (314 LOC): COESO (sem split)

- **Ação:** **não dividir.** A v2 (`generate_stress_signal_v2`) é uma fn única com
  6 blocos sequenciais que **compartilham o estado `out`/`rng`** — separar
  arriscaria a **bit-exatidão determinística** (invariante chave, L39).
- **Limpeza (benigna):** `let _t1 = 5.0;` (L259) e `_sr` (L303) são placeholders
  intencionais — manter ou anotar.
- **DoD:** verdes; sem mudança de sequência de PRNG.

---

## ÉPICO 5 — Limpeza transversal (opcional, baixa prioridade)

> Tarefas independentes, sem split. Cada uma é pequena e isolada. **Não** mudar
> lógica. Útil para um agente "faxina".

### S5.T01 — Padronização de idioma em comentários/mensagens

- Substituir comentários/mensagens soltas em PT por EN (ou vice-versa, conforme
  padrão do arquivo) nos pontos identificados: `conv1d_dyn.rs` (L280/L432),
  `spsc.rs` (L271), `params.rs` (L210), `diagnostic.rs` (L155), `gui/ui/mod.rs`
  (L58–64/L117/L1153), `dispatcher/lstm.rs` comentários informais.
- **DoD:** verdes; diff só comentários/strings de log/panic não-funcionais.

#### S5.T02 — Remoção de doc-comments obsoletos/duplicados

- `traits.rs` doc duplicado (L447–456); `accumulate.rs` nota "Task 3.4"
  (L11–12); `model.rs` breadcrumbs "wavenet_common.rs" (L21/L165); `conv1d.rs`
  `//!` duplicado (L4–10); `dot_4x/avx512_bf16.rs` doc citando `dpbf16` (L11);
  `tanh.rs` refs piecewise/`_div_` (L18–22/L224); `diagnostic.rs` L388.
- **DoD:** verdes; diff só comentários.

#### S5.T03 — Auditoria e gating/remoção de dead code (executar com CUIDADO)

- Remover (após `grep` global confirmando 0 chamadores, incluindo `benches/` e
  `tests/`): `DenseLayerDyn::process_acc_block`, `DenseLayerDyn::process_fused`.
- Aplicar `#[cfg(test)]` (não remover) às APIs test-only: `Conv1dDyn::
  process_block`/`_bf16`; cadeia test-only de `Conv1d` (`process_block` +
  `process_single_frame` no-mixin + `_internal`); `scalar_minimax_sigmoid`.
- Investigar (sem remover) campos `mod_input_gain`/`mod_output_gain`
  (`processor/mod.rs` + `dsp.rs`) — possível feature incompleta; registrar
  decisão.
- **DoD:** verdes; cada remoção/gating acompanhada da evidência de `grep`.

---

## Anexo A — Matriz de priorização e paralelização

| Sprint | Subsistema              | Tarefas    | Split?        | Risco         | Paralelizável com |
| ------ | ----------------------- | ---------- | ------------- | ------------- | ----------------- |
| 1.A    | math/gemm               | S1.T01–T04 | Sim (por ISA) | Médio         | 2, 3, 4           |
| 1.B    | math/dsp,wavenet,common | S1.T05–T08 | Sim           | Médio         | 2, 3, 4           |
| 1.C    | math/common (macros)    | S1.T09–T10 | Sim/Coeso     | Baixo         | 2, 3, 4           |
| 2      | models                  | S2.T01–T07 | Maioria       | Médio         | 1, 3, 4           |
| 3.A    | clap/gui                | S3.T01–T04 | Sim           | Baixo (UI)    | 1, 2, 4           |
| 3.B    | clap/processor,plugin   | S3.T05–T09 | Sim/Coeso     | **Alto (RT)** | 1, 2, 4           |
| 4.A    | common                  | S4.T01–T02 | Sim           | Médio         | 1, 2, 3           |
| 4.B    | dsp                     | S4.T03–T05 | Sim/Coeso     | **Alto (RT)** | 1, 2, 3           |
| 4.C    | loader                  | S4.T06–T09 | Sim           | Baixo         | 1, 2, 3           |
| 4.D    | standalone              | S4.T10–T11 | Sim/Parcial   | **Alto (RT)** | 1, 2, 3           |
| 4.E    | testing                 | S4.T12     | Coeso         | Baixo         | todos             |
| 5      | transversal             | S5.T01–T03 | Não           | Baixo–Médio   | após 1–4          |

**Tarefas de maior risco (RT hot-path):** S3.T05 (processor/dsp), S3.T06/T07 (processor/params), S4.T03 (pipeline/stages), S4.T10 (rt_callback), S4.T11 (capture). Exigem revisão manual de RT-safety no diff e, quando possível, verificação de não-regressão em `benches/`.
