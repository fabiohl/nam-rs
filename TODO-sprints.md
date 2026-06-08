<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO-sprints — Refatoração Estrutural (Rust) do nam-rs

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

## ÉPICO 4 — DSP / Loader / Common / Standalone (CONCLUÍDO ✅)

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
- **Nota de correção (08/06/2026):** arquivos corrigidos de `pipeline/stage_*.rs`
   para `pipeline/stages/{input,inference,output,bridge}.rs` conforme o plano
   original. Re-exports via `stages/mod.rs`; `pipeline/mod.rs` usa `mod stages`.
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

#### S4.T08 — Dividir `src/loader/namb_encoder.rs` (348 LOC)(CONCLUÍDO ✅)

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

#### S4.T09 — Dividir `src/loader/dispatcher/lstm.rs` (328 LOC) (CONCLUÍDO ✅)

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

#### S4.T10 — Dividir `src/standalone/pw_host/rt_callback.rs` (325 LOC) — RT CRÍTICO (CONCLUÍDO ✅)

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

#### S4.T11 — `src/standalone/pw_host/capture.rs` (312 LOC): extração parcial (RISCO) (CONCLUÍDO ✅) (CONCLUÍDO ✅)

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
- **Nota de conclusão:** Extração completa entregue (state + listeners + mod.rs).
  `CaptureState` é capturado por `move` na closure `process` — todos os campos
  permanecem na pilha, zero alocação RT. Borrows disjuntos em campos do struct
  melhoram aliasing analysis. `state_changed_handler`/`param_changed_handler`
  são fns livres executadas no main-loop do PipeWire (não-RT), seguras para
  `log::*`. 250 testes passam, clippy limpo, build verde.

### Sprint 4.E — Testing util (coeso)

#### S4.T12 — `src/testing/stress.rs` (314 LOC): COESO (sem split) (CONCLUÍDO ✅)

- **Ação:** **não dividir.** A v2 (`generate_stress_signal_v2`) é uma fn única com
  6 blocos sequenciais que **compartilham o estado `out`/`rng`** — separar
  arriscaria a **bit-exatidão determinística** (invariante chave, L39).
- **Limpeza (benigna):** `let _t1 = 5.0;` (L259) e `_sr` (L303) são placeholders
  intencionais — manter ou anotar.
- **DoD:** verdes; sem mudança de sequência de PRNG.

### Nota de auditoria pós-Épico 4 (08/06/2026)

- **S4.T03 (stages):** corrigido — arquivos movidos de `pipeline/stage_*.rs` para
  `pipeline/stages/` conforme o plano original (commit `c09d83f`).
- **Regressão de benchmark:** splits de módulo do `scalar_ref` (5 sub-módulos)
  causaram perda de inlining cross-module, resultando em +5–8% nos benchmarks
  `head_rechannel_fp32/*_Scalar` e +2–4% nos LSTM SIMD.
  - **Correção:** 29 `#[inline]` adicionados a todas as fns públicas de
    `scalar_ref/{gemm,dot,lstm,convolution,utility}.rs`. Regressões recuperadas
    (head_rechannel −5%, LSTM −4%).
  - **Prewarm_WaveNet (+4.7%):** cold path (`#[cold]`), sem ação — esperado.
- **Impacto em Épico 5:** o `scalar_ref` agora tem `#[inline]` em todas as fns;
  S5.T03 (dead code) deve preservar essas anotações ao remover/`#[cfg(test)]`.

---

## ÉPICO 5 — Limpeza transversal (opcional, baixa prioridade)

> Tarefas independentes, sem split. Cada uma é pequena e isolada. **Não** mudar lógica. Útil para um agente "faxina".

### S5.T01 — Padronização de idioma em comentários/mensagens (CONCLUÍDO ✅)

- Substituir comentários/mensagens soltas em PT por EN nos pontos identificados: `conv1d_dyn.rs` (L280/L432), `spsc.rs` (L271), `params.rs` (L210), `diagnostic.rs` (L155), `gui/ui/mod.rs` (L58–64/L117/L1153), `dispatcher/lstm.rs` comentários informais.
- **DoD:** verdes; diff só comentários/strings de log/panic não-funcionais.

#### S5.T02 — Remoção de doc-comments obsoletos/duplicados (CONCLUÍDO ✅)

- `traits.rs` doc duplicado (L447–456); `accumulate.rs` nota "Task 3.4"
  (L11–12); `model.rs` breadcrumbs "wavenet_common.rs" (L21/L165); `conv1d.rs`
  `//!` duplicado (L4–10); `dot_4x/avx512_bf16.rs` doc citando `dpbf16` (L11);
  `tanh.rs` refs piecewise/`_div_` (L18–22/L224); `diagnostic.rs` L388.
- **DoD:** verdes; diff só comentários.

#### S5.T03 — Auditoria e gating/remoção de dead code (CONCLUÍDO ✅)

- **`DenseLayerDyn::process_acc_block` / `process_fused` — já removidos em S2.T01.** `dense_dyn.rs` (97 LOC) não contém esses métodos.
- **`Conv1dDyn::process_block` / `process_block_bf16` — já gatilhados com `#[cfg(test)]` em S2.T02.**
  `conv1d_dyn.rs:142` e `:167`.
- **Cadeia test-only de `Conv1d` — já gatiada em S2.T06.**
  `process_single_frame` (no-mixin) em `conv1d.rs:41` e `process_block` em
  `:299` com `#[cfg(test)]`. `process_single_frame_internal` não pode ser
  gatiado (usado por `process_single_frame_with_mixin` em produção).
- **`scalar_minimax_sigmoid` — NÃO gatiado (usado por benchmarks).**
  Confirmado: `benches/inference_bench.rs` (4 call sites via
  `process_sample_scalar`), `tests/lstm_scalar_bf16_parity.rs` (3),
  `src/models/lstm/tests.rs` (1). Mesma decisão de S2.T04. `#[cfg(test)]`
  quebraria benchmarks.
- **`mod_input_gain` / `mod_output_gain` — NÃO é dead code.**
  Confirmado em S3.T06: consumidos ativamente por `processor/events.rs` para
  modulação CLAP de parâmetros.
- **Remoção adicional: `DenseLayer<IN,OUT>::process_fused`,
  `process_acc_single_frame`, `process_fused_block`, `process_acc_block`
  (aliases mortos).** 0 chamadores confirmados via `grep` em `src/` +
  `benches/` + `tests/`. Removidos de `dense.rs` (204→139 LOC).
- **DoD:** `lints.sh` verde; `tests-cargo.sh` verde (568 pass, 0 fail).

---
---

## ÉPICO 6 — Residual: Otimização Máxima

> **Objetivo:** esgotar TODAS as oportunidades residuais de melhoria estrutural conforme
> `refatora-rust.md`. Abrange: (A) extração de implementação de `mod.rs` inflados,
> (B) split ou justificativa de arquivos residuais > 300 LOC, (C) deduplicação de
> código entre subsistemas, (D) realocação de arquivos em diretórios incorretos,
> (E) constantes e limpezas pontuais.
>
> **Foco estritamente residual.** Ao final deste épico, **nenhuma** oportunidade de
> melhoria conforme `refatora-rust.md` permanece em aberto.

### Sprint 6.A — Extrair implementação de mod.rs inflados

> **Problema geral:** `mod.rs` deve ser agregador de submódulos (declarações `mod`,
> `pub mod`, `pub use`). Vários `mod.rs` contêm funções, structs e impl blocks
> substanciais que devem ser extraídos para submódulos nomeados. O critério é:
> se o `mod.rs` tem mais implementação do que declarações, a implementação deve
> ser extraída.

#### S6.T01 — Extrair implementação de `src/loader/mod.rs` (260 LOC → ~21 LOC) [DONE]

- **Problema:** 239 linhas de implementação (`LoadedModelPair` struct + Debug impl
  - 4 consts + `load_and_build_model` fn) vs 10 linhas de declarações.
- **Split:**
  - `loader/loaded_model_pair.rs` ← struct `LoadedModelPair`, `impl Debug`,
    4 `const`s (`DEFAULT_*`, `MAX_MODEL_BYTES`), imports.
  - `loader/build.rs` ← `pub fn load_and_build_model` (corpo completo, ~190 linhas).
  - `loader/mod.rs` (mantém) ← `pub mod` decls + re-export
    `pub use loaded_model_pair::*` + `pub use build::load_and_build_model`.
- **Consumidores afetados:** `main.rs`, `clap/plugin/main_thread/load.rs`,
  `clap/factory/preset_discovery.rs`, `benches/inference_bench.rs` — preservar
  caminhos via re-export.
- **RT-Safety:** caminho de loader (cold, não-RT). Alocação de `Vec<u8>` + `Box`
  OK; sem mudança de lógica.
- **DoD:** padrão §0.2.

#### S6.T02 — Extrair implementação de `src/standalone/pw_host/mod.rs` (264 LOC → ~28 LOC) [DONE]

- **Problema:** 173 linhas de `run_pipewire_host` vs 15 linhas de declarações.
- **Split:**
  - `pw_host/run.rs` ← `pub fn run_pipewire_host` (corpo completo, seções 1–7).
  - `pw_host/mod.rs` (mantém) ← `mod` decls + re-export
    `pub use run::run_pipewire_host`.
  - (Opcional) dividir seções internas de `run_pipewire_host`:
    `pw_host/run.rs` ← seção 1 (loop init, bridge alloc, CPU affinity),
    `pw_host/control_loop.rs` ← seções 5–7 (main loop, GC drain, shutdown).
    Recomendação: split único primeiro; se ainda > 150 LOC, split adicional.
- **Consumidores:** `main.rs` — preservar `crate::standalone::pw_host::run_pipewire_host`.
- **RT-Safety:** função contém tanto cold (setup) quanto hot (loop com atômicos).
  Manter `#[cold]` hints nos blocos de setup; o loop principal usa SPSC/atômicos
  `Relaxed` — preservar `Ordering`s.
- **DoD:** padrão.

#### S6.T03 — Extrair implementação de `src/standalone/pw_host/capture/mod.rs` (230 LOC → ~33 LOC) [DONE]

- **Problema:** 197 linhas de `setup_capture_stream` (com 3 closures) vs 18 linhas
  de declarações.
- **Split:**
  - `pw_host/capture/setup.rs` ← `pub fn setup_capture_stream` (corpo completo
    com closures `state_changed`, `param_changed`, `process`).
  - `pw_host/capture/mod.rs` (mantém) ← `mod state; mod listeners; mod setup;` +
    re-export `pub use setup::setup_capture_stream`.
  - (Opcional) extrair closure `process` (~123 linhas) para
    `pw_host/rt_callback/capture_process.rs` — cuidado: a closure captura ~30
    locais por `move`; extrair quebraria as capturas. **Não extrair** a closure
    `process`; apenas mover `setup_capture_stream` inteira para `setup.rs`.
- **RT-Safety:** a closure `process` é entrada RT do PipeWire. Manter todos os
  buffers como arrays de pilha capturados por `move`; `parking_lot: [Option<GcItem>;16]`
  pré-alocado; `configure_realtime_thread` one-shot. Nenhuma mudança de captura
  ou escopo.
- **DoD:** padrão + revisão RT.

#### S6.T04 — Extrair implementação de `tests/common/mod.rs` (413 LOC → ~30 LOC) [DONE]

- **Problema:** pior infrator — 367 linhas de implementação vs 9 linhas de
  declarações. É `mod.rs` de testes, mas o mesmo princípio se aplica.
- **Split:**
  - `tests/common/constants.rs` ← `GOLDEN_NUM_SAMPLES`, `GOLDEN_BLOCK_SIZE`,
    `TEST_BLOCK_SIZE`, `TEST_NUM_BLOCKS`, `STRESS_SAMPLE_RATE`.
  - `tests/common/metrics.rs` ← `compute_mse`, `compute_max_abs_error`.
  - `tests/common/validation.rs` ← `report_dsp_fidelity`, `topology_thresholds`.
  - `tests/common/signals.rs` ← `generate_stress_signal` (deprecated),
    `generate_sine_440hz`.
  - `tests/common/io_helpers.rs` ← `read_golden_bin`, `model_path`,
    `process_in_blocks`.
  - `tests/common/mod.rs` (mantém) ← `pub mod` decls + re-exports de todos os
    símbolos movidos.
- **Consumidores:** TODOS os arquivos em `tests/` importam de
  `tests/common/mod.rs` → preservar via `pub use` no `mod.rs`.
- **RT-Safety:** funções de teste (não-RT). Sem restrições.
- **DoD:** padrão.

#### S6.T05 — Extrair implementação de `src/loader/dispatcher/mod.rs` (112 LOC → ~30 LOC) [DONE]

- **Problema:** 100 linhas de implementação (`WeightCursor` struct + impl +
  `build_model`) vs 5 linhas de declarações.
- **Split:**
  - `dispatcher/weight_cursor.rs` ← `WeightCursor` struct + `impl` (new,
    is_interleaved4, is_gate_major_lstm, read_slice, read_f32, verify_exhausted)
    - use imports do `super::nam_json::WeightsLayout`.
  - `dispatcher/mod.rs` (mantém) ← `pub mod lstm; pub mod wavenet;` +
    `mod weight_cursor; pub(crate) use weight_cursor::WeightCursor;` +
    `pub fn build_model(data)` (7 linhas de dispatch, permanece).
- **RT-Safety:** caminho de loader (não-RT). `WeightCursor` é forward-only reader
  sobre slice `&[f32]`; sem alocação própria. Sem mudança de lógica.
- **DoD:** padrão.

#### S6.T06 — Extrair implementação de `src/models/a2/mod.rs` (119 LOC → ~28 LOC) [DONE]

- **Problema:** `WavenetA2Placeholder` (struct + 3 impl blocks, ~72 linhas)
  polui o `mod.rs` que deveria ser apenas declarações + re-exports.
- **Split:**
  - `a2/placeholder.rs` ← struct `WavenetA2Placeholder` + `impl WavenetA2Placeholder`
    (new, inject_rt_status) + `impl sealed::Sealed` + `impl NamModel` (process,
    prewarm).
  - `a2/mod.rs` (mantém) ← `pub mod` decls + `pub use` re-exports + `pub use
    placeholder::WavenetA2Placeholder`.
  - (Opcional) o bloco `#[cfg(test)] mod tests` (L106–119) pode ser extraído para
    `a2/tests.rs` — mas como é < 300 linhas, permanece inline (regra `testing.md`).
- **RT-Safety:** `process` é placeholder (silencia saída, sem DSP real). Preservar
  `AtomicBool::compare_exchange` one-shot para warning e `Relaxed` em `heap_audit`.
- **DoD:** padrão.

#### S6.T07 — Extrair boilerplate de prewarm de `src/models/lstm/mod.rs` (202 LOC → ~162 LOC) [DONE]

- **Problema:** `LstmLike` trait + 2 impls + `lstm_prewarm_common` fn (~40 linhas)
  representam lógica de prewarm que pode ser isolada.
- **Split:**
  - `lstm/prewarm.rs` ← trait `LstmLike` (com `reset_input_slots`) + impls para
    `LstmModel1`/`LstmModel2` + `pub(super) fn lstm_prewarm_common`.
  - `lstm/mod.rs` (mantém) ← type aliases + sealed impls + NamModel impls +
    `mod prewarm;`. Os `NamModel::prewarm` chamam `lstm_prewarm_common(self, ...)`
    via `use self::prewarm::*`.
- **Consumidores:** apenas `mod.rs` consome `lstm_prewarm_common` — sem quebra
  de API pública.
- **RT-Safety:** cold path (prewarm). `lstm_prewarm_common` processa chunks de 512
  samples com `model.process(buf, ...)` — preservar `#[inline(always)]` no
  `process` subjacente.
- **DoD:** padrão.

#### S6.T08 — Extrair channel extraction de `src/clap/processor/dsp/mod.rs` (212 LOC → ~163 LOC) [DONE]

- **Problema:** `process_dsp_audio` (190 linhas) contém lógica de extração de
  canais (L46–94, ~49 linhas) com matching complexo de `AudioBufferType` que é
  autocontida e candidata natural a extração.
- **Split:**
  - `processor/dsp/channels.rs` ← fn privada `extract_channels(audio) ->
    (Option<&mut [f32]>, Option<&mut [f32]>)` contendo o match
    `InputOutput`/`InPlace`/`InputOnly`/`OutputOnly` para L e R, com
    `#[cfg(feature="stereo")]` gating.
  - `processor/dsp/mod.rs` (mantém) ← `process_dsp_audio` chama
    `extract_channels` no lugar do bloco extraído.
- **⚠️ RT-Safety (CRÍTICO):** `process_dsp_audio` é HOT-PATH. `extract_channels`
  deve ser `#[inline(always)]` para o otimizador reproduzir o inlining atual.
  Sem alloc/panic; apenas slicing de referências. Preservar o padrão stereo
  feature gate `#[cfg(feature="stereo")]` / `#[cfg(not(feature="stereo"))]`.
- **DoD:** padrão + verificação de não-regressão de benchmark (rodar
  `benches/inference_bench.rs` antes/depois se possível).

---

### Sprint 6.B — Arquivos residuais > 300 LOC sem justificativa de coesão

> Auditoria do Épico 2 (linhas 258–268) identificou 4 arquivos que permanecem
> acima do limiar de 300 linhas **sem** justificativa de coesão registrada.
> Cada tarefa abaixo decide: split ou justificativa explícita.

#### S6.T09 — `src/models/lstm/layer.rs` (399 LOC): split por responsabilidade [DONE]

- **Problema:** `define_lstm_process!` macro + 6 instâncias SIMD + fallback
  escalar + `scalar_minimax_sigmoid` + `LstmLayer` struct + impl + acessores
  — tudo em um arquivo.
- **Split (opção A — recomendado):**
  - `lstm/layer.rs` (mantém) ← struct `LstmLayer`, `impl` (new, gate dispatch,
    acessores), `#[cfg(test)] mod layer_test`.
  - `lstm/layer_kernels.rs` ← `scalar_minimax_sigmoid` (pub, usado por benches
    e parity tests) + `define_lstm_process!` macro + as 6 instâncias SIMD
    (`process_sample` x 6) + fallback escalar.
  - `lstm/mod.rs` ← `pub use layer::LstmLayer;`.
- **Split (opção B — alternativo):** manter coeso e registrar justificativa:
  "unidade única de kernel LSTM — macro `define_lstm_process!` gera 6 instâncias
  que compartilham o mesmo corpo; separar quebraria a localidade da macro e
  poluiria o namespace com 7 fns públicas."
- **RT-Safety:** hot-path de inferência. Preservar `#[inline(always)]`,
  `#[target_feature]`, `_mm_prefetch`, `Aligned64<[f32;N]>` na pilha.
- **DoD:** padrão. Se opção B, registrar justificativa inline no arquivo.
- **Nota (S6.T09 executada):** Opção A aplicada.
  - `lstm/layer.rs` (71 LOC): struct `LstmLayer`, `impl` (new, acessores,
    reset_input_slot, reset_states), `Default`.
  - `lstm/layer_kernels.rs` (337 LOC): `scalar_minimax_sigmoid` (pub),
    `define_lstm_process!` macro, 6 instâncias SIMD, fallback escalar.
  - `lstm/mod.rs`: adicionado `pub mod layer_kernels`.
  - `scalar_minimax_sigmoid` tornado `pub` em `layer_kernels.rs` (acessível
    via `crate::models::lstm::layer_kernels::scalar_minimax_sigmoid`).

#### S6.T10 — `src/models/wavenet/model.rs` (397 LOC): split de `WaveNetLayerArray` + `WaveNetModel` [DONE]

- **Problema:** `WaveNetLayerArray` + `WaveNetModel` (métodos `process`,
  `prewarm`, `prewarm_avx512`, `prewarm_avx2`, `new`, `layer_count`, etc.)
  no mesmo arquivo.
- **Split:**
  - `wavenet/model.rs` (mantém) ← struct `WaveNetModel<CH, K, HEAD>` + `impl`
    (new, process, prewarm, prewarm_avx512/avx2, prewarm_samples, layer_count,
    set_max_buffer_size, set_effective_layers) + imports.
  - `wavenet/layer_array.rs` ← struct `WaveNetLayerArray<CH, K, HEAD>` + `impl`
    (new, layer_count, set_effective_layers, get_layer) + `WavenetProcessContext`
    - `WavenetProcessParams`.
  - `wavenet/mod.rs` ← atualizar re-exports.
- **RT-Safety:** `process` é hot-path. `WaveNetLayerArray` é acessado via
  `self.layers[*]` em `process` — preservar `#[inline(always)]` nos wrappers.
  Buffers de pilha `[0.0f32; 1024]`; `_mm_prefetch`; `const { assert! }` guards.
- **DoD:** padrão.

#### S6.T11 — `src/models/wavenet/conv1d_dyn_kernels.rs` (336 LOC): split dual-frame × single-frame × block [DONE]

- **Problema:** 3 kernels genéricos (`process_dual_frame_generic`,
  `process_single_frame_generic`, `process_block_generic`) compartilham a mesma
  assinatura de `PrefetchFn` e `unsafe` contracts, mas são funcionalmente
  independentes.
- **Split:**
  - `wavenet/conv1d_dyn_kernels.rs` (mantém) ← `process_single_frame_generic` +
    `process_block_generic`.
  - `wavenet/conv1d_dyn_dual.rs` ← `process_dual_frame_generic` (~130 linhas,
    kernel dual-frame Temporal-Tiling).
  - `wavenet/mod.rs` ← sem re-export adicional (kernels são `pub(crate)`).
- **RT-Safety:** hot-path. Preservar `from_raw_parts` + `get_unchecked` + prefetch
  indireto via `prefetch_fn`; `debug_assert!` apenas.
- **DoD:** padrão.

#### S6.T12 — `src/dsp/adaptive.rs` (329 LOC): split ou justificativa de coesão [DONE]

- **Problema:** `AdaptiveCompute` (estado + lógica de adaptação de qualidade +
  timeslicer + métricas). S0.T01 extraiu testes (647→329), mas permanece > 300
  sem justificativa.
- **Análise:** o módulo tem 3 responsabilidades:
  - `AdaptiveCompute` struct + `new()` + `update()` (máquina de estados).
  - `AdaptiveTimeslicer` struct + `should_yield()`.
  - `AdaptiveMetrics` struct + `report_frame()` + reset.
- **Split (opção A):**
  - `adaptive.rs` (mantém) ← `AdaptiveCompute` struct + `new` + `update`.
  - `adaptive/timeslicer.rs` ← `AdaptiveTimeslicer`.
  - `adaptive/metrics.rs` ← `AdaptiveMetrics`.
- **Justificativa de coesão (opção B):** `AdaptiveTimeslicer` e `AdaptiveMetrics`
  são usados exclusivamente por `AdaptiveCompute::update` como componentes
  internos; separar exporia detalhes de implementação e quebraria o encapsulamento
  da unidade de adaptação. São 2 structs pequenas (~40 + ~60 linhas) com zero
  consumidores externos.
- **RT-Safety:** `update()` roda no hot-path (chamado após cada frame de
  processamento). `Relaxed` atômicos; sem alloc/lock. `should_yield` usa
  `Instant::now()` (RT-safe, apenas leitura de TSC).
- **DoD:** padrão. Recomendação: opção B (justificativa de coesão) é preferível
  a expor internals.

#### S6.T13 — `src/models/wavenet/conv1d.rs` (316 LOC): justificativa explícita de coesão [DONE]

- **Problema:** S2.T06 declarou "split opcional" e a auditoria registrou "coesão
  implícita", mas não há justificativa formal no arquivo.
- **Ação:** **não dividir.** Adicionar justificativa de coesão no topo do arquivo:
  "Unidade única de convolução 1D estática: trait `ConvInput` + struct `Conv1d` +
  kernel single-frame + wrappers com mixin formam uma unidade algorítmica coesa.
  `ConvInput` foi extraído para `conv_input.rs` (S2.T06). Split adicional do
  kernel single-frame quebraria a localidade dos contratos `unsafe` de aliasing
  e acumuladores Kahan."
- **RT-Safety:** preservar acumuladores Kahan bit-a-bit; buffers de tap na pilha;
  `from_raw_parts` + `get_unchecked`; `prefetch_fn`.
- **DoD:** `lints.sh` + `tests-cargo.sh` verdes; diff apenas adição de comentário.

---

### Sprint 6.C — Deduplicação e compartilhamento de código

> Código duplicado entre subsistemas que deve ser unificado em local compartilhado.
> **Atenção:** estas tarefas envolvem mudança de imports e, em alguns casos, leve
> alteração de assinatura — risco moderado de regressão.

#### S6.T14 — Unificar `CountingAllocator` duplicado em `src/common/alloc_audit.rs` [DONE]

- **Problema:** `CountingAllocator` + `TrackingGuard` implementados **identicamente**
  em dois locais:
  - `src/dsp/pipeline/test_util.rs` (L15–51) — `#[cfg(test)] pub(crate)`.
  - `src/clap/heap_audit.rs` (L19–61) — runtime + test, feature `heap-audit`.
- **Ação:**
  - Criar `src/common/alloc_audit.rs`:
    - `pub(crate) static ALLOC_COUNT: AtomicU64`
    - `pub(crate) static TRACKING_THREAD: AtomicI32`
    - `pub(crate) struct CountingAllocator` (sem `unsafe impl GlobalAlloc` —
      cada consumer registra seu próprio `#[global_allocator]`).
    - `pub(crate) struct TrackingGuard` + `new()` + `Drop`.
    - `pub(crate) static AUDIT_ENABLED: AtomicBool`
    - `pub(crate) static AUDIT_THREAD: AtomicI32`
  - Atualizar `src/dsp/pipeline/test_util.rs`: remover definições duplicadas,
    importar de `crate::common::alloc_audit`.
  - Atualizar `src/clap/heap_audit.rs`: remover definições duplicadas, importar
    de `crate::common::alloc_audit`. Manter `#[global_allocator]` registration
    local (específico do CLAP).
  - Atualizar `src/common/mod.rs`: `pub(crate) mod alloc_audit;`.
- **RT-Safety:** `CountingAllocator` é `#[global_allocator]` — roda em **todas**
  as threads. Manter `libc::syscall(libc::SYS_gettid)` e `compare_exchange`
  exatamente como estão. `TrackingGuard::drop` decrementa `ALLOC_COUNT` — preservar
  `Relaxed`. Nenhuma mudança de lógica.
- **DoD:** padrão + verificar que testes de heap audit (CLAP `heap-audit` feature)
  continuam funcionando.

#### S6.T15 — Unificar gate flag state triplicado: usar `report_gate_flags()` canônico [DONE]

- **Problema:** a mesma lógica de mapeamento `GateState → RT_STATUS_IS_SILENT/IS_FADING`
  aparece em 3 lugares:
  - `src/clap/processor/dsp/gate_flags.rs` — `report_gate_flags(rt_status, gate_state)` (canônico).
  - `src/dsp/pipeline/capture.rs` (L33–52) — bloco `match` inline.
  - `src/dsp/pipeline/stages/input.rs` (L35–42) — `handle_silence_bypass()` (parcial, só `Closed`).
- **Ação:**
  - Mover `report_gate_flags` para `src/dsp/gate_flags.rs` (ao lado de `gate.rs`)
    ou mantê-lo em `src/clap/processor/dsp/gate_flags.rs` mas torná-lo
    `pub(crate)`.
  - Substituir o `match` inline em `dsp/pipeline/capture.rs` por chamada a
    `report_gate_flags(rt_status, state)`.
  - Em `dsp/pipeline/stages/input.rs`, unificar `handle_silence_bypass` para
    chamar `report_gate_flags` (hoje só seta `IS_SILENT`, mas deveria também
    limpar `IS_FADING` como as outras variantes fazem — verificar se a diferença
    é intencional).
  - Se `handle_silence_bypass` for mantido separado, documentar a razão da
    divergência.
- **⚠️ RT-Safety (CRÍTICO):** `report_gate_flags` usa `store(Relaxed)` e
  `fetch_or(Relaxed)` em `RtStatusFlags`. Ambos os call sites estão no hot-path.
  Preservar `#[inline(always)]` e `Ordering::Relaxed`.
- **DoD:** padrão + confirmar que o comportamento de flags é idêntico
  (testes de gate existentes em `dsp/gate.rs` e `clap/processor_test.rs`).

#### S6.T16 — Unificar `ModelInfo` construction: método `LoadedModelPair::model_info()` [DONE]

- **Problema:** construção de `ModelInfo` duplicada entre:
  - `src/main.rs` (L120–126): `ModelInfo { arch_label, channels, receptive_field,
    weights_layout, path_basename }`.
  - `src/clap/plugin/main_thread/load.rs` (L100–117): mesma estrutura, mesmos
    campos.
- **Ação:**
  - Adicionar `pub fn model_info(&self) -> ModelInfo` em `impl LoadedModelPair`
    (em `src/loader/loaded_model_pair.rs` se S6.T01 executado, ou em
    `src/loader/mod.rs`).
  - Substituir os dois call sites por `loaded.model_info()`.
  - `path_basename` já usa `PathBuf` — manter `to_string_lossy().to_string()`.
- **RT-Safety:** caminho de loader (não-RT). Alocação de `String` OK.
- **DoD:** padrão.

#### S6.T17 — Mover `error_code_to_str` para método de `NamErrorCode`[DONE]

- **Problema:** `src/clap/plugin/main_thread/load.rs` (L16–33) define
  `fn error_code_to_str(code: NamErrorCode) -> &'static str` como free function.
  `NamErrorCode` (em `src/common/diagnostics/error_codes.rs`) já tem `code()`
  (numérico) e `mnemonic()` (screaming snake), mas não `message()`.
- **Ação:**
  - Adicionar `pub fn message(&self) -> &'static str` em `impl NamErrorCode`
    (em `error_codes.rs`).
  - Substituir chamada em `load.rs` por `code.message()`.
  - Remover free function `error_code_to_str` de `load.rs`.
- **RT-Safety:** diagnóstico (não-RT). Sem impacto.
- **DoD:** padrão.

#### S6.T18 — Definir constante `SPSC_CAPACITY` centralizada[DONE]

- **Problema:** capacidade do SPSC (`64`) hardcoded em:
  - `src/main.rs` (L89): `spsc::setup_spsc(64)`.
  - Possivelmente em `src/clap/plugin/mod.rs` (verificar).
- **Ação:**
  - Adicionar `pub const SPSC_CAPACITY: usize = 64;` em
    `src/common/spsc/mod.rs`.
  - Substituir literais `64` nos call sites pela constante.
- **RT-Safety:** sem impacto (constante de compilação).
- **DoD:** padrão.

---

### Sprint 6.D — Realocação de arquivos em diretórios incorretos

> Arquivos cujo propósito é geral, mas residem em diretório específico de
> subsistema, ou vice-versa.

#### S6.T19 — Mover `src/clap/param_smoother.rs` → `src/dsp/smoother.rs` [DONE]

- **Problema:** `ParamSmoother` é um filtro IIR de 1-polo para suavização de
  parâmetros de áudio (ganho, gate threshold). É utilitário DSP genérico, não
  específico do CLAP. O standalone aplica ganho sem smoothing — poderia
  beneficiar-se.
- **Ação:**
  - Mover `src/clap/param_smoother.rs` → `src/dsp/smoother.rs`.
  - Atualizar `src/clap/mod.rs`: remover `mod param_smoother`.
  - Atualizar `src/dsp/mod.rs`: adicionar `pub mod smoother;`.
  - Atualizar imports nos consumidores CLAP (`processor/dsp/mod.rs`,
    `processor/dsp/gain.rs`, `processor/dsp/gate_flags.rs`):
    `crate::clap::param_smoother::ParamSmoother` →
    `crate::dsp::smoother::ParamSmoother`.
- **RT-Safety:** `ParamSmoother::process` é chamado no hot-path do processor.
  Manter `#[inline(always)]` nos métodos; struct é `Copy` (campos `f32` na pilha);
  sem alloc.
- **DoD:** padrão.

#### S6.T20 — Consolidar `src/clap/heap_audit.rs` em `src/common/alloc_audit.rs` [DONE]

- **Problema:** `heap_audit.rs` é infraestrutura de auditoria de alocação RT
  (genérica), mas reside no diretório CLAP.
- **Ação (depende de S6.T14):**
  - Após S6.T14 (unificação do `CountingAllocator`), o conteúdo remanescente de
    `heap_audit.rs` (global allocator registration + feature gate) é pequeno e
    pode ser movido para `src/common/alloc_audit.rs` ou mantido como
    `src/clap/heap_audit.rs` com apenas o `#[global_allocator]` registration
    específico do CLAP.
  - Se o standalone futuramente precisar de heap audit, a infra compartilhada
    já estará em `common`.
- **Decisão:** se `heap_audit.rs` após S6.T14 contiver APENAS
  `#[global_allocator] static GLOBAL: CountingAllocator = ...` + feature gate,
  manter em `clap/` com justificativa: "registro de global_allocator é específico
  do binary crate CLAP". Caso contrário, mover integralmente.
- **DoD:** padrão.

---

### Sprint 6.E — Constantes e limpezas pontuais

#### S6.T21 — Corrigir nome ambíguo `src/dsp/pipeline/playback.rs` [DONE]

- **Problema:** o arquivo `playback.rs` em `dsp/pipeline/` contém lógica de
  saída PipeWire (`PipewireHostConfig`, `AppState`) e é gated com
  `#[cfg(feature = "standalone")]`. O nome "playback" é enganoso — não é
  playback genérico, é saída PipeWire standalone.
- **Ação:**
  - Renomear `src/dsp/pipeline/playback.rs` → `src/dsp/pipeline/output_pw.rs`
    (ou `pw_output.rs`).
  - Atualizar `src/dsp/pipeline/mod.rs`: `mod playback` → `mod output_pw` +
    re-exports correspondentes.
  - Atualizar consumidores em `standalone/pw_host/`.
- **RT-Safety:** sem mudança de lógica. Apenas rename.
- **DoD:** padrão.

#### S6.T22 — Auditoria final: varredura de dead code e comentários obsoletos residuais [DONE]

- **Ação:** varrer TODO-sprints.md e confirmar que TODOS os itens de dead code
  e comentários obsoletos foram executados (S5.T01–T03). Verificar:
  - `src/dsp/pipeline/capture.rs`: comentários de "5.1.x" obsoletos.
  - `src/loader/dispatcher/lstm.rs`: comentários PT informais.
  - Qualquer `#[allow(unused)]` residual que possa ser removido.
- **DoD:** `lints.sh` + `tests-cargo.sh` verdes; grep por "TODO", "FIXME",
  "HACK" no código-fonte — apenas itens legítimos restantes.

---
---

## ÉPICO 7 — Residual: Otimização Máxima (Última Onda)

> **Objetivo:** esgotar as **últimas** oportunidades de melhoria estrutural
> identificadas na auditoria final do código (2026-06-08). Abrange: (A) `mod.rs`
> residuais com implementação, (B) split dos últimos arquivos > 300 LOC sem
> justificativa, (C) deduplicação de código entre subsistemas, (D) realocação de
> símbolos em diretórios corretos, (E) registros de débito técnico e achados de
> auditoria.
>
> **Após este épico, TODAS as oportunidades de melhoria conforme `refatora-rust.md`
> estarão esgotadas.**

---

### Sprint 7.A

#### S7.T01 — Extrair implementação de `src/clap/gui/window/mod.rs` (240 LOC → ~25 LOC) [DONE]

- Aviso: Tarefas que altera assinaturas/imports públicos: Exigem atualização de todos os consumidores com `grep` global antes de concluir.
- **Problema:** `struct NamPluginWindow` (24 linhas) + 4 blocos `impl` — `new()`,
  `safe_shared()`, `destroy_gl_resources()`, `Drop` — ocupam ~200 linhas. É o
  maior `mod.rs` com implementação ainda não justificado.
- **Split:**
  - `window/state.rs` ← `struct NamPluginWindow` + `impl NamClapWindow` com
    `new()`, `safe_shared()`, `destroy_gl_resources()`, `#[cfg(test)]
    impl NamClapWindow` (método `test_init`).
  - `window/lifecycle.rs` ← `impl Drop for NamClapWindow`.
  - `window/mod.rs` (mantém) ← `pub mod` decls (`handler`, `drag_drop`, `shaders`,
    `input_map`, `state`, `lifecycle`) + `pub(crate) use` re-exports +
    `#[cfg(test)] #[path="window_test.rs"] mod window_test;`.
- **Consumidores afetados:** `gui/mod.rs`, `gui/ui/mod.rs` (importam
  `crate::clap::gui::window::NamPluginWindow`). Preservar via `pub use
  state::NamPluginWindow;`.
- **RT-Safety:** thread de UI (não-RT). Preservar checagem de `alive_fence` em
  `safe_shared()` e pareamento `make_current`/`make_not_current` nos
  early-returns.
- **DoD:** padrão §0.2.

#### S7.T02 — Extrair implementação de `src/loader/namb/mod.rs` (147 LOC → ~30 LOC) [DONE]

- Aviso: Tarefas que altera assinaturas/imports públicos: Exigem atualização de todos os consumidores com `grep` global antes de concluir.
- **Problema:** `pub fn parse_namb()` (corpo de ~125 linhas) é a única
  implementação no `mod.rs`. S4.T07 extraiu `error.rs`, `header.rs`, `fallback.rs`
  mas a função principal permaneceu.
- **Split:**
  - `namb/parse.rs` ← `pub fn parse_namb` (corpo completo, versão verbatim) +
    imports de `crate::loader::nam_json::NamModelData` e submódulos locais.
  - `namb/mod.rs` (mantém) ← `pub mod error; pub mod header; pub mod fallback;
    pub mod parse;` + `pub use error::NambError; pub use header::NambHeader;
    pub use parse::parse_namb;` + include `#[path="namb_test.rs"]`.
- **Consumidores:** `loader/dispatcher/wavenet/mod.rs`, `loader/loaded_model_pair.rs`,
  `loader/build.rs`, `standalone/pw_host/capture/state.rs`, `clap/plugin/main_thread/load.rs`,
  `clap/factory/preset_discovery.rs`. Preservar `crate::loader::namb::parse_namb`
  via re-export.
- **RT-Safety:** loader (não-RT). Cast `unsafe` de header é bounds-checked.
  Sem mudança de lógica.
- **DoD:** padrão.

#### S7.T03 — Mover `scalar_minimax_sigmoid` de `src/models/lstm/layer_kernels.rs` para `src/math/activations/sigmoid.rs` [DONE]

- Aviso: Tarefas que altera assinaturas/imports públicos: Exigem atualização de todos os consumidores com `grep` global antes de concluir.
- **Problema:** `scalar_minimax_sigmoid` (L14–39 em `layer_kernels.rs`) é uma
  função matemática pura (aproximação Minimax de sigmoid) sem dependência de
  `LstmLayer`. Ela espelha o kernel SIMD em `src/math/activations/sigmoid.rs`.
  Residir em `models/lstm/` é incorreto — é uma primitiva matemática.
- **Ação:**
  - Mover `pub fn scalar_minimax_sigmoid(x: f32) -> f32` para
    `src/math/activations/sigmoid.rs` (adicionar ao final do arquivo, com
    doc-comment: "Scalar reference implementation for testing and benchmarks.
    Mirrors the SIMD kernel.").
  - Atualizar `src/math/activations/mod.rs`: `pub use sigmoid::scalar_minimax_sigmoid;`.
  - Em `src/models/lstm/layer_kernels.rs`: substituir definição por
    `use crate::math::activations::scalar_minimax_sigmoid;`.
  - Atualizar consumidores que importam de
    `crate::models::lstm::layer_kernels::scalar_minimax_sigmoid`:
    - `benches/inference_bench.rs` (4 call sites via `process_sample_scalar`).
    - `tests/lstm_scalar_bf16_parity.rs` (3 call sites).
    - `src/models/lstm/tests.rs` (1 call site).
    Atualizar imports para `crate::math::activations::scalar_minimax_sigmoid`.
- **RT-Safety:** função escalar pura (sem alocação, sem SIMD). Usada apenas em
  benchmarks e parity tests (não no hot-path de produção).
- **DoD:** padrão + confirmar que benchmarks compilam e parity tests passam.

**Auditoria Sprint 7.A (2026-06-08):** S7.T01 estava sem o bloco `#[cfg(test)]
impl NamPluginWindow` com método `test_init` previsto no split plan. Implementado
em `window/state.rs` (valida `size_of` ≤ 4096 + alinhamento 8-byte). S7.T02 e
S7.T03 íntegros — apenas desvios de estilo (mod privado vs pub mod; wildcard
re-export vs explícito), sem impacto funcional. `lints.sh` e `tests-cargo.sh`
verdes. DoD §0.2 atendido para todas as tarefas.

---

### Sprint 7B

- Para validar a não-regressão de performance, estou rodando um "cargo bench" no começo e no final da sprint (para fechamento). Não rode em segundo plano!

#### S7.T04 — Extrair implementação de `src/clap/processor/dsp/mod.rs` (171 LOC → ~19 LOC) [DONE]

- Aviso: Tarefa de maior risco (RT hot-path):** , Exigem verificação de não-regressão de benchmark e bit-exatidão (O dev humano fará à parte).
- **Problema:** `process_dsp_audio` (orquestrador do loop de DSP, ~152 linhas após
  S6.T08) permanece no `mod.rs`. Os helpers já foram extraídos (S3.T05,
  S6.T08). O orquestrador deve seguir o mesmo padrão.
- **Split:**
  - `processor/dsp/orchestrator.rs` ← `impl NamClapProcessor::process_dsp_audio`
    (corpo completo, verbatim, com todas as seções: preâmbulo de extração de
    canais, bypass, input gain, dispatch do modelo, output gain, peaks,
    telemetry, heap-audit).
  - `processor/dsp/mod.rs` (mantém) ← `pub mod bypass; pub mod gain; pub mod
    peaks; pub mod gate_flags; pub mod telemetry; pub mod channels; pub mod
    orchestrator;` + `pub(crate) use orchestrator::process_dsp_audio;`.
- **⚠️ RT-Safety (CRÍTICO):** `process_dsp_audio` é o **hot-path principal** do
  plugin. `#[inline(always)]` deve ser preservado no orquestrador. Sem
  alloc/lock/panic; manter `Ordering::Relaxed`; bloco `heap-audit` permanece
  sob `#[cfg(feature="heap-audit")]`. **Validar ausência de regressão de
  performance** (rodar `benches/inference_bench.rs` antes/depois se possível).
- **DoD:** padrão + verificação de não-regressão de benchmark.

#### S7.T05 — Unificar inicialização bias+mixin em arquivos de convolução WaveNet [DONE]

- Aviso: Tarefa de maior risco (RT hot-path):** , Exigem verificação de não-regressão de benchmark e bit-exatidão (O dev humano fará à parte).
- **Problema:** o padrão de inicialização de 4 acumuladores (bias + mixin) é
  duplicado entre:
  - `src/models/wavenet/conv1d.rs` (L104–118) — single-frame.
  - `src/models/wavenet/conv1d_dual.rs` (L102–121) — dual-frame (f0/f1).
  A lógica é idêntica (4 casos: Some(mixin)+bias, Some(mixin) only,
  bias only, zeros). A variante dual repete o bloco para `_f0`/`_f1`.
- **Ação:**
  - Criar helper `fn init_accum_with_bias_mixin(acc: &mut [f32; 4], bias: &[f32],
    mixin: Option<&[f32]>, out_offset: usize, do_bias: bool)` em
    `src/models/wavenet/conv_input.rs` (onde já reside `trait ConvInput`).
    Corpo extraído verbatim do single-frame, generalizado para aceitar slice
    mutável de 4 acumuladores.
  - Em `conv1d.rs`: substituir bloco inline por chamada ao helper.
  - Em `conv1d_dual.rs`: substituir os dois blocos (f0/f1) por duas chamadas
    ao helper (para `acc_f0` e `acc_f1`).
- **RT-Safety:** hot-path. `#[inline(always)]` no helper. Manter Kahan
  initialization exatamente como está. Sem alloc/lock/panic.
- **DoD:** padrão + verificar que o diff é movimentação pura (testes de
  convolução existentes em `wavenet/tests.rs` garantem equivalência).

#### S7.T06 — Unificar load/store de 4 acumuladores em arquivos de convolução WaveNet

- Aviso: Tarefa de maior risco (RT hot-path):** , Exigem verificação de não-regressão de benchmark e bit-exatidão (O dev humano fará à parte).
- **Problema:** o padrão de carregar 4 acumuladores com fallback para OUT não
  múltiplo de 4 é duplicado entre:
  - `src/models/wavenet/conv1d.rs` (L163–187 load, L222–242 write-back).
  - `src/models/wavenet/conv1d_dual.rs` (L159–205 load, L231–263 write-back).
  O dual-frame aplica o mesmo padrão para `_f0`/`_f1`.
- **Ação:**
  - Criar helper(s) em `conv_input.rs` (ver S7.T09):
    - `fn load_4_accums(out: &[f32], idx: usize) -> [f32; 4]` — carrega 4
      amostras com fallback (extraído verbatim do bloco de load do single-frame).
    - `fn store_kahan_4_accums(out: &mut [f32], idx: usize, r: [f32; 4],
      c: [f32; 4])` — write-back com Kahan (extraído verbatim do single-frame).
  - Em `conv1d.rs`: substituir blocos inline por chamadas.
  - Em `conv1d_dual.rs`: substituir blocos `_f0`/`_f1` por chamadas.
- **⚠️ RT-Safety (CRÍTICO):** hot-path de inferência. `#[inline(always)]` nos
  helpers. Preservar acumuladores Kahan **bit-a-bit** (a ordem de acumulação
  não pode mudar). `unsafe` `get_unchecked` deve permanecer nos helpers.
  **Validar bit-exatidão com testes existentes.**
- **Risco:** MÉDIO — manipulação de `unsafe` e Kahan. Se o compilador não
  inlinear corretamente, pode haver regressão de precisão.
- **DoD:** padrão + diff bit-exato nos testes de convolução.

---

### Sprint 7C

#### S7.T07 — Extrair implementação de `src/clap/gui/ui/meter/mod.rs` (171 LOC → ~22 LOC)

- **Problema:** `pub fn draw_vertical_meter` (~150 linhas) é um orquestrador que
  faz dispatch para `glow`/`cpu`/`readout`. Os submódulos já existem (S3.T03);
  a função orquestradora deve seguir o mesmo padrão.
- **Split:**
  - `meter/orchestrator.rs` ← `pub(crate) fn draw_vertical_meter` (corpo
    completo: label, LED, interação, peak-hold/dB, dispatch condicional para
    `render_glow`/`render_cpu`/texto de readout).
  - `meter/mod.rs` (mantém) ← `pub mod glow; pub mod cpu; pub mod readout;
    pub mod orchestrator;` + `pub(crate) use orchestrator::draw_vertical_meter;`.
- **RT-Safety:** thread de UI. Closure `CallbackFn` no caminho GL mantém
  marshalling atômico `Relaxed` e `Arc` `'static`/`Send`. Preservar padrão de
  peak-hold.
- **DoD:** padrão.

#### S7.T08 — Extrair implementação de `src/clap/gui/ui/status_bar/mod.rs` (140 LOC → ~18 LOC)

- **Problema:** `pub(crate) fn draw_zone5_status_bar` (~123 linhas) é um
  orquestrador que faz dispatch para `metadata`/`telemetry`/`toast`/`warnings`.
  Os submódulos já existem (S3.T01); a função deve ser extraída.
- **Split:**
  - `status_bar/orchestrator.rs` ← `pub(crate) fn draw_zone5_status_bar` (corpo
    completo com subseções de metadata, telemetry, toast, warning-area).
  - `status_bar/mod.rs` (mantém) ← `pub mod telemetry; pub mod metadata;
    pub mod orchestrator;` + `pub(crate) use orchestrator::draw_zone5_status_bar;`.
- **RT-Safety:** thread de UI. Preservar leituras atômicas `Relaxed` de
  `ui_peak_l/r`, `current_latency`, `rt_status`.
- **DoD:** padrão.

#### S7.T09 — Unificar constante `PAGE_2M` / `HUGE_PAGE_2M` duplicada

- **Problema:** `2 * 1024 * 1024` (2 MB) definido em dois locais:
  - `src/math/common/huge_alloc.rs:57`: `const PAGE_2M: usize = 2 * 1024 * 1024;`
  - `src/dsp/mirror_buf/alloc.rs:51`: `const HUGE_PAGE_2M: usize = 2 * 1024 * 1024;`
- **Ação:**
  - Criar `pub const HUGE_PAGE_2M: usize = 2 * 1024 * 1024;` em
    `src/math/common/huge_alloc.rs` (ou em `src/common/` se preferir
    infraestrutura compartilhada).
  - Em `mirror_buf/alloc.rs`: remover definição local, importar de
    `crate::math::common::huge_alloc::HUGE_PAGE_2M`.
- **RT-Safety:** constante de compilação. Sem impacto.
- **DoD:** padrão.

#### S7.T10 — Unificar estratégia de huge-page allocation entre `huge_alloc.rs` e `mirror_buf/alloc.rs`

- **Problema:** ambos implementam a mesma estratégia de 3 níveis de fallback
  para alocação de huge pages (MAP_HUGETLB → THP via `madvise` → fallback
  padrão), com ~80% de similaridade estrutural.
  - `src/math/common/huge_alloc.rs:73-149` — `allocate_huge_pages()`.
  - `src/dsp/mirror_buf/alloc.rs:20-291` — `MirroredBuffer::new()` +
    `try_new_huge()`.
- **Ação (opção conservadora — recomendada):**
  - Extrair `fn create_backing_fd(size: usize, use_huge: bool) -> RawFd` e
    `fn try_mmap_huge(addr, len, fd, offset, use_huge) -> *mut c_void` como
    helpers em `src/math/common/huge_alloc.rs` (torná-los `pub(crate)`).
  - Em `mirror_buf/alloc.rs`: substituir blocos inline de `memfd_create`,
    `fcntl(F_ADD_SEALS)`, `ftruncate`, e `mmap(MAP_HUGETLB)` por chamadas
    aos helpers compartilhados.
  - **Não** unificar a lógica de mirroring (duplo `mmap(MAP_FIXED)`) — é
    específica do `MirroredBuffer` e diferente do `HugePageVec`.
- **RT-Safety:** ambos os arquivos são cold-path (alocação, nunca na thread de
  áudio). `mmap`/`munmap` OK.
- **Risco:** MÉDIO — manipulação de `unsafe` mmap. Validar com testes de
  `mirror_buf` e `huge_alloc` existentes.
- **DoD:** padrão + testes de integridade de huge pages (se disponíveis no CI).

---

## Sprint 7B - Rodadas de "cargo bench" Antes e Depois

### ANTES: Abarca alguns do Épicos anteriores

```bash
fabio@notebook:~/nam-rs$ time cargo bench
   Compiling nam-rs v1.7.0 (/home/fabio/nam-rs)
    Finished `bench` profile [optimized] target(s) in 1m 03s
     Running unittests src/lib.rs (target/release/deps/nam_rs-1f534ba60e6b1c11)

running 256 tests
test common::audio_host::tests::test_mock_host_error ... ignored
test common::audio_host::tests::test_mock_host_traits ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_all_codes_have_unique_numeric ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_days_to_date_epoch ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_days_to_date_known ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_diagnostic_bundle_nominal ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_diagnostic_bundle_redaction ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_diagnostic_bundle_with_error_matches ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_diagnostic_display ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_diagnostic_support_block_contains_code ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_emit_irq_advisory_safety ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_error_code_display ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_system_snapshot_capture ... ignored
test common::diagnostics::diagnostic::diagnostic_test::test_timestamp_format ... ignored
test common::params::tests::test_params_default ... ignored
test common::spsc::spsc_test::test_gc_overflow_overwrite ... ignored
test common::spsc::spsc_test::test_gc_stress_no_leak ... ignored
test common::spsc::spsc_test::test_spsc_concurrency ... ignored
test common::spsc::spsc_test::test_spsc_full_empty ... ignored
test dsp::adaptive::adaptive_test::tests::aggressive_full_to_reduced_lower_threshold ... ignored
test dsp::adaptive::adaptive_test::tests::aggressive_reduced_to_minimal ... ignored
test dsp::adaptive::adaptive_test::tests::conservative_full_stays_full_if_not_consecutive ... ignored
test dsp::adaptive::adaptive_test::tests::conservative_full_to_reduced ... ignored
test dsp::adaptive::adaptive_test::tests::conservative_reduced_recovers_to_full ... ignored
test dsp::adaptive::adaptive_test::tests::conservative_reduced_recovery_resets_on_intermediate ... ignored
test dsp::adaptive::adaptive_test::tests::conservative_reduced_to_minimal ... ignored
test dsp::adaptive::adaptive_test::tests::crossfade_completes ... ignored
test dsp::adaptive::adaptive_test::tests::crossfade_starts_on_transition ... ignored
test dsp::adaptive::adaptive_test::tests::lstm_effective_layers ... ignored
test dsp::adaptive::adaptive_test::tests::minimal_recovers_to_reduced ... ignored
test dsp::adaptive::adaptive_test::tests::mode_off_no_transitions ... ignored
test dsp::adaptive::adaptive_test::tests::set_mode_resets_state ... ignored
test dsp::adaptive::adaptive_test::tests::wavenet_effective_layers_full ... ignored
test dsp::adaptive::adaptive_test::tests::wavenet_effective_layers_minimal ... ignored
test dsp::adaptive::adaptive_test::tests::wavenet_effective_layers_reduced ... ignored
test dsp::gate::gate_test::tests::test_gate_params_default ... ignored
test dsp::gate::gate_test::tests::test_hysteresis_apply_gain_ramp ... ignored
test dsp::gate::gate_test::tests::test_hysteresis_basic_transitions ... ignored
test dsp::gate::gate_test::tests::test_hysteresis_interrupted_fade ... ignored
test dsp::gate::gate_test::tests::test_sub_block_granularity ... ignored
test dsp::gate::gate_test::tests::test_unit_block_processing ... ignored
test dsp::mirror_buf::mirror_buf_test::test_mirror_buf_clone ... ignored
test dsp::mirror_buf::mirror_buf_test::test_mirror_buf_debug ... ignored
test dsp::mirror_buf::mirror_buf_test::test_mirror_buf_large_allocation ... ignored
test dsp::mirror_buf::mirror_buf_test::test_mirror_buf_mirroring ... ignored
test dsp::mirror_buf::mirror_buf_test::test_mirror_buf_page_alignment ... ignored
test dsp::mirror_buf::mirror_buf_test::test_mirror_buf_zst_error ... ignored
test dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest ... ignored
test dsp::pipeline::pipeline_block_test::block_tests::test_unconventional_block_sizes_lstm ... ignored
test dsp::pipeline::pipeline_block_test::block_tests::test_unconventional_block_sizes_wavenet ... ignored
test dsp::pipeline::pipeline_block_test::block_tests::test_zero_alloc_edge_cases ... ignored
test dsp::pipeline::pipeline_block_test::block_tests::test_zero_alloc_stress_edge_cases ... ignored
test dsp::pipeline::pipeline_test::tests::test_bypass_no_resampler_mono ... ignored
test dsp::pipeline::pipeline_test::tests::test_bypass_no_resampler_stereo ... ignored
test dsp::pipeline::pipeline_test::tests::test_bypass_with_resampler_mono ... ignored
test dsp::pipeline::pipeline_test::tests::test_bypass_with_resampler_stereo ... ignored
test dsp::pipeline::pipeline_test::tests::test_denormal_dither_mono_symmetry ... ignored
test dsp::pipeline::pipeline_test::tests::test_hotpath_clipping_detection ... ignored
test dsp::pipeline::pipeline_test::tests::test_hotpath_dropped_frames ... ignored
test dsp::pipeline::pipeline_test::tests::test_hotpath_gate_closed_and_silence ... ignored
test dsp::pipeline::pipeline_test::tests::test_hotpath_gate_fading ... ignored
test dsp::resampler::resampler_test::test_bypass_48k ... ignored
test dsp::resampler::resampler_test::test_downsample_96k_to_48k ... ignored
test dsp::resampler::resampler_test::test_fixed_point_drift_random_ratios ... ignored
test dsp::resampler::resampler_test::test_impulse_response_input ... ignored
test dsp::resampler::resampler_test::test_impulse_response_output ... ignored
test dsp::resampler::resampler_test::test_latency_calculation ... ignored
test dsp::resampler::resampler_test::test_output_upsample_48k_to_96k ... ignored
test dsp::resampler::resampler_test::test_phase_accum_underflow_guard ... ignored
test dsp::resampler::resampler_test::test_resampler_micro_soak ... ignored
test dsp::resampler::resampler_test::test_resampler_mono_equivalence ... ignored
test dsp::resampler::resampler_test::test_roundtrip_96k ... ignored
test dsp::resampler::resampler_test::test_upsample_44k_to_48k ... ignored
test dsp::sinc_kernel::sinc_kernel_test::test_aligned_coeffs_alignment ... ignored
test dsp::sinc_kernel::sinc_kernel_test::test_bessel_i0_known_values ... ignored
test dsp::sinc_kernel::sinc_kernel_test::test_minimum_phase_causal ... ignored
test dsp::sinc_kernel::sinc_kernel_test::test_minimum_phase_energy_concentration ... ignored
test dsp::sinc_kernel::sinc_kernel_test::test_polyphase_bank_dimensions ... ignored
test dsp::sinc_kernel::sinc_kernel_test::test_sinc_kaiser_dc_unity ... ignored
test dsp::smoother::tests::test_smoother_convergence ... ignored
test dsp::smoother::tests::test_smoother_convergence_high_gain ... ignored
test dsp::smoother::tests::test_smoother_denormal_prevention ... ignored
test dsp::smoother::tests::test_smoother_relative_threshold ... ignored
test dsp::smoother::tests::test_smoother_snap ... ignored
test dsp::telemetry::tests::test_histogram_mapping ... ignored
test dsp::telemetry::tests::test_percentiles ... ignored
test loader::dispatcher::wavenet::bias_tune::tests::test_apply_bias_compensation ... ignored
test loader::dispatcher::wavenet::bias_tune::tests::test_dense_compensation_nonzero_bf16 ... ignored
test loader::dispatcher::wavenet::bias_tune::tests::test_dequantize_bf16_roundtrip ... ignored
test loader::dispatcher::wavenet::bias_tune::tests::test_dequantize_f16_roundtrip ... ignored
test loader::nam_json::nam_json_test::test_forward_compat_unknown_field_at_root ... ignored
test loader::nam_json::nam_json_test::test_forward_compat_unknown_field_in_config ... ignored
test loader::nam_json::nam_json_test::test_forward_compat_unknown_field_in_metadata ... ignored
test loader::nam_json::nam_json_test::test_is_wavenet_a2_versions ... ignored
test loader::nam_json::nam_json_test::test_parse_empty_weights ... ignored
test loader::nam_json::nam_json_test::test_parse_feather_wavenet ... ignored
test loader::nam_json::nam_json_test::test_parse_lstm ... ignored
test loader::nam_json::nam_json_test::test_parse_malformed_config ... ignored
test loader::nam_json::nam_json_test::test_parse_missing_architecture ... ignored
test loader::nam_json::nam_json_test::test_parse_missing_weights ... ignored
test loader::nam_json::nam_json_test::test_parse_semver ... ignored
test loader::nam_json::nam_json_test::test_parse_truncated_json ... ignored
test loader::nam_json::nam_json_test::test_reject_deeply_nested_training ... ignored
test loader::nam_json::nam_json_test::test_topology_invalid_channels ... ignored
test loader::nam_json::nam_json_test::test_topology_lite ... ignored
test loader::nam_json::nam_json_test::test_topology_nano ... ignored
test loader::nam_json::nam_json_test::test_topology_standard ... ignored
test loader::nam_json::nam_json_test::test_weights_exceed_limit_fast_rejection ... ignored
test loader::nam_json::nam_json_test::test_weights_within_limit ... ignored
test loader::namb::namb_test::tests::test_parse_namb_v1 ... ignored
test loader::namb::namb_test::tests::test_parse_namb_v2_gate_major ... ignored
test loader::namb::namb_test::tests::test_reject_magic_bman ... ignored
test loader::namb::namb_test::tests::test_v1_crc32_zero_warns_but_passes ... ignored
test loader::namb::namb_test::tests::test_v2_crc32_zero_legitimate_passes ... ignored
test loader::namb::namb_test::tests::test_v2_missing_crc32_flag_rejected ... ignored
test math::activations::tests::test_fused_sigmoid_relu_slice_dispatch_smoke ... ignored
test math::activations::tests::test_prelu_scalar ... ignored
test math::activations::tests::test_prelu_slice_dispatch_smoke ... ignored
test math::activations::tests::test_relu_scalar ... ignored
test math::activations::tests::test_relu_slice_dispatch_smoke ... ignored
test math::activations::tests::test_sigmoid_direct_minimax_boundary ... ignored
test math::activations::tests::test_sigmoid_pade_proptest_100k ... ignored
test math::activations::tests::test_sigmoid_scalar_equivalences ... ignored
test math::activations::tests::test_sigmoid_slice_dispatch_smoke ... ignored
test math::activations::tests::test_silu_scalar ... ignored
test math::activations::tests::test_silu_slice_dispatch_smoke ... ignored
test math::activations::tests::test_softsign_scalar ... ignored
test math::activations::tests::test_softsign_slice_dispatch_smoke ... ignored
test math::activations::tests::test_tanh_pade_proptest_100k ... ignored
test math::activations::tests::test_tanh_piecewise_boundaries ... ignored
test math::activations::tests::test_tanh_piecewise_odd_symmetry ... ignored
test math::activations::tests::test_tanh_piecewise_proptest_50k ... ignored
test math::activations::tests::test_tanh_piecewise_saturation ... ignored
test math::activations::tests::test_tanh_scalar_equivalences ... ignored
test math::activations::tests::test_tanh_slice_dispatch_smoke ... ignored
test math::common::kahan::kahan_test::test_horizontal_sum_drift_reduction ... ignored
test math::common::kahan::kahan_test::test_kahan4_independent_channels ... ignored
test math::common::kahan::kahan_test::test_kahan_accuracy_advantage ... ignored
test math::common::kahan::kahan_test::test_kahan_deep_convolution_drift ... ignored
test math::common::kahan::kahan_test::test_kahan_struct_vs_inline ... ignored
test math::common::tests::huge_alloc_tests::test_allocate_large_falls_back_gracefully ... ignored
test math::common::tests::huge_alloc_tests::test_allocate_small_uses_heap ... ignored
test math::common::tests::huge_alloc_tests::test_huge_page_vec_fallback ... ignored
test math::common::tests::huge_alloc_tests::test_huge_page_vec_with_capacity ... ignored
test math::common::tests::test_accumulate_head ... ignored
test math::common::tests::test_compute_energy_avx2 ... ignored
test math::common::tests::test_compute_energy_parity ... ignored
test math::common::tests::test_compute_energy_stereo_parity ... ignored
test math::common::tests::test_compute_max_diff_avx2 ... ignored
test math::common::tests::test_compute_max_diff_parity ... ignored
test math::common::tests::test_compute_peak_abs_stereo_parity ... ignored
test math::common::tests::test_convolve_mono_dual_parity ... ignored
test math::common::tests::test_convolve_mono_parity ... ignored
test math::common::tests::test_convolve_stereo_dual_parity ... ignored
test math::common::tests::test_dot_product_avx2_fma ... ignored
test math::common::tests::test_dot_product_avx512 ... ignored
test math::common::tests::test_dot_product_bf16_avx512_regression ... ignored
test math::common::tests::test_f32_to_bf16_avx2_parity ... ignored
test math::common::tests::test_f32_to_bf16_avx512_regression ... ignored
test math::common::tests::test_gated_activation_and_overwrite_block ... ignored
test math::common::tests::test_gemv_overwrite_bf16_avx512_regression ... ignored
test math::common::tests::test_horizontal_sum ... ignored
test math::common::tests::test_set_daz_ftz ... ignored
test math::common::tests::test_store_bf16_avx2 ... ignored
test math::common::tests::test_store_bf16_avx512 ... ignored
test math::common::tests::test_tanh_and_accumulate_block ... ignored
test math::common::tests::test_tanh_and_overwrite_block ... ignored
test math::dsp::gain::gain_test::test_apply_gain_simd ... ignored
test math::dsp::gain::gain_test::test_apply_ramp_simd ... ignored
test math::dsp::gain::gain_test::test_combined_gain_staging ... ignored
test math::dsp::gain::gain_test::test_extreme_gain_values ... ignored
test math::dsp::gain::gain_test::test_gain_roundtrip_6db ... ignored
test math::dsp::gain::gain_test::test_gain_true_bypass ... ignored
test math::dsp::gain_lut::tests::test_gain_lut_clamping ... ignored
test math::dsp::gain_lut::tests::test_gain_lut_initialization ... ignored
test math::dsp::gain_lut::tests::test_gain_lut_interpolation ... ignored
test math::gemm::dot_4x::dot_4x_test::test_dot_4x_interleaved_avx512_stress ... ignored
test math::gemm::dot_4x::dot_4x_test::test_dot_4x_interleaved_avx512_vs_avx2 ... ignored
test math::gemm::dot_4x::dot_4x_test::test_dot_4x_interleaved_avx512_vs_fallback ... ignored
test math::gemm::dot_4x::dot_4x_test::test_dot_4x_interleaved_dual_frame_avx512_stress ... ignored
test math::gemm::dot_4x::dot_4x_test::test_dot_4x_interleaved_dual_frame_avx512_vs_avx2 ... ignored
test math::gemm::dot_4x::dot_4x_test::test_dot_4x_interleaved_dual_frame_avx512_vs_fallback ... ignored
test models::a2::activations::activations_test::tests::test_activation_fast_tanh ... ignored
test models::a2::activations::activations_test::tests::test_activation_hard_swish ... ignored
test models::a2::activations::activations_test::tests::test_activation_hard_tanh ... ignored
test models::a2::activations::activations_test::tests::test_activation_leaky_hardtanh ... ignored
test models::a2::activations::activations_test::tests::test_activation_leaky_relu ... ignored
test models::a2::activations::activations_test::tests::test_activation_prelu ... ignored
test models::a2::activations::activations_test::tests::test_activation_relu ... ignored
test models::a2::activations::activations_test::tests::test_activation_sigmoid ... ignored
test models::a2::activations::activations_test::tests::test_activation_silu ... ignored
test models::a2::activations::activations_test::tests::test_activation_softsign ... ignored
test models::a2::activations::activations_test::tests::test_activation_tanh ... ignored
test models::a2::activations::activations_test::tests::test_prelu_cycle ... ignored
test models::a2::activations::activations_test::tests::test_prelu_empty_slopes ... ignored
test models::a2::film::tests::test_film_config_custom ... ignored
test models::a2::film::tests::test_film_config_default ... ignored
test models::a2::gating::tests::test_config_construction ... ignored
test models::a2::gating::tests::test_gating_mode_default ... ignored
test models::a2::params::tests::test_head_params_construction ... ignored
test models::a2::params::tests::test_layer_array_params_a2_construction ... ignored
test models::a2::params::tests::test_layer_params_a2_construction ... ignored
test models::a2::tests::tests_placeholder::test_wavenet_a2_placeholder_silence ... ignored
test models::lstm::lstm_tests::tests::test_lstm_gate_order_consistency ... ignored
test models::lstm::lstm_tests::tests::test_lstm_model1_allocation ... ignored
test models::lstm::lstm_tests::tests::test_lstm_model1_process_zeros ... ignored
test models::lstm::lstm_tests::tests::test_lstm_model2_allocation ... ignored
test models::lstm::lstm_tests::tests::test_lstm_model2_pipelining_parity ... ignored
test models::lstm::lstm_tests::tests::test_lstm_model2_process_deterministic ... ignored
test models::lstm::lstm_tests::tests::test_lstm_reset_on_prewarm ... ignored
test models::lstm::lstm_tests::tests::test_lstm_state_evolution ... ignored
test models::lstm::lstm_tests::tests::test_lstm_variable_block_sizes ... ignored
test models::wavenet::tests::test_conv1d_dilation ... ignored
test models::wavenet::tests::test_conv1d_dyn_large_kernel_no_segfault ... ignored
test models::wavenet::tests::test_conv1d_dyn_padding_non_multiple_of_4 ... ignored
test models::wavenet::tests::test_conv1d_identity_kernel ... ignored
test models::wavenet::tests::test_conv1d_known_output ... ignored
test models::wavenet::tests::test_conv1d_with_bias ... ignored
test models::wavenet::tests::test_conv1d_zero_input ... ignored
test models::wavenet::tests::test_dense_layer_identity ... ignored
test models::wavenet::tests::test_dense_layer_rectangular ... ignored
test models::wavenet::tests::test_dense_layer_with_bias ... ignored
test models::wavenet::tests::test_gated_layer_dyn_process ... ignored
test models::wavenet::tests::test_non_gated_layer_dyn_process ... ignored
test models::wavenet::tests::test_read_conv1d_weights_dyn_limits ... ignored
test models::wavenet::tests::test_wavenet_layer_array_dyn_block_size_gated ... ignored
test models::wavenet::tests::test_wavenet_model_allocation ... ignored
test models::wavenet::tests::test_wavenet_prewarm_no_nan ... ignored
test models::wavenet::tests::test_wavenet_process_deterministic ... ignored
test models::wavenet::tests::test_wavenet_process_zeros ... ignored
test standalone::cli::tests::test_parse_args_diagnose ... ignored
test standalone::cli::tests::test_parse_args_diagnose_full ... ignored
test standalone::cli::tests::test_parse_args_model_and_gains ... ignored
test standalone::pw_host::pw_host_test::test_dsp_bridge_concurrent_access ... ignored
test standalone::rt_setup::rt_setup_test::test_get_allowed_cpus_not_empty ... ignored
test standalone::rt_setup::rt_setup_test::test_parse_interrupts_basic ... ignored
test standalone::rt_setup::rt_setup_test::test_rdtsc_nanos_monotonic ... ignored
test standalone::rt_setup::rt_setup_test::test_rdtsc_nanos_significant ... ignored
test standalone::rt_setup::rt_setup_test::test_select_optimal_cpu_returns_something ... ignored
test testing::mushra::tests::test_fnv1a32_known_vector ... ignored
test testing::mushra::tests::test_mulberry32_determinism ... ignored
test testing::mushra::tests::test_mulberry32_range ... ignored
test testing::mushra::tests::test_soft_clip_identity ... ignored
test testing::perceptual::perceptual_test::test_esr_all_zero_test ... ignored
test testing::perceptual::perceptual_test::test_esr_identical_is_zero ... ignored
test testing::perceptual::perceptual_test::test_esr_invariant_to_sample_rate ... ignored
test testing::perceptual::perceptual_test::test_lufs_empty ... ignored
test testing::perceptual::perceptual_test::test_lufs_sine ... ignored
test testing::perceptual::perceptual_test::test_mr_stft_different_signals ... ignored
test testing::perceptual::perceptual_test::test_mr_stft_empty ... ignored
test testing::perceptual::perceptual_test::test_mr_stft_identical_is_zero ... ignored
test testing::stress::stress_test::test_v1_deterministic ... ignored
test testing::stress::stress_test::test_v1_not_silent ... ignored
test testing::stress::stress_test::test_v2_deterministic ... ignored
test testing::stress::stress_test::test_v2_non_silent_segments ... ignored
test testing::stress::stress_test::test_v2_valid_sizes ... ignored

test result: ok. 0 passed; 0 failed; 256 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/gen_stress.rs (target/release/deps/gen_stress-3126d0d93485f8ad)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/main.rs (target/release/deps/nam_rs-9ca32a8a155732f7)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/wav_to_golden.rs (target/release/deps/wav_to_golden-834a03ae0d22c12c)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running benches/dot_4x_bench.rs (target/release/deps/dot_4x_bench-5417e22bc074b1b0)
dot_4x_interleaved_avx512/fallback_16
                        time:   [23.018 ns 23.025 ns 23.037 ns]
                        change: [−0.0412% +0.0206% +0.0702%] (p = 0.50 > 0.05)
                        No change in performance detected.
Found 9 outliers among 100 measurements (9.00%)
  1 (1.00%) low mild
  4 (4.00%) high mild
  4 (4.00%) high severe
dot_4x_interleaved_avx512/avx2_16
                        time:   [6.5500 ns 6.5529 ns 6.5559 ns]
                        change: [−0.2334% −0.0643% +0.0878%] (p = 0.45 > 0.05)
                        No change in performance detected.
Found 6 outliers among 100 measurements (6.00%)
  5 (5.00%) high mild
  1 (1.00%) high severe
dot_4x_interleaved_avx512/fallback_64
                        time:   [156.95 ns 156.97 ns 156.99 ns]
                        change: [−0.1937% −0.0804% +0.0254%] (p = 0.15 > 0.05)
                        No change in performance detected.
Found 7 outliers among 100 measurements (7.00%)
  3 (3.00%) high mild
  4 (4.00%) high severe
dot_4x_interleaved_avx512/avx2_64
                        time:   [23.136 ns 23.264 ns 23.541 ns]
                        change: [−0.1259% +0.1652% +0.6401%] (p = 0.58 > 0.05)
                        No change in performance detected.
Found 5 outliers among 100 measurements (5.00%)
  4 (4.00%) high mild
  1 (1.00%) high severe
dot_4x_interleaved_avx512/fallback_256
                        time:   [689.26 ns 689.33 ns 689.41 ns]
                        change: [−0.5483% −0.3351% −0.1726%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 7 outliers among 100 measurements (7.00%)
  1 (1.00%) low mild
  4 (4.00%) high mild
  2 (2.00%) high severe
dot_4x_interleaved_avx512/avx2_256
                        time:   [89.733 ns 89.977 ns 90.435 ns]
                        change: [−0.3686% −0.0471% +0.2976%] (p = 0.80 > 0.05)
                        No change in performance detected.
Found 13 outliers among 100 measurements (13.00%)
  6 (6.00%) high mild
  7 (7.00%) high severe
dot_4x_interleaved_avx512/fallback_1024
                        time:   [2.8188 µs 2.8191 µs 2.8194 µs]
                        change: [−1.3101% −0.9555% −0.6258%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 7 outliers among 100 measurements (7.00%)
  4 (4.00%) high mild
  3 (3.00%) high severe
dot_4x_interleaved_avx512/avx2_1024
                        time:   [360.65 ns 361.07 ns 361.64 ns]
                        change: [−0.2197% +0.1375% +0.6571%] (p = 0.56 > 0.05)
                        No change in performance detected.
Found 18 outliers among 100 measurements (18.00%)
  8 (8.00%) high mild
  10 (10.00%) high severe
dot_4x_interleaved_avx512/fallback_4096
                        time:   [11.341 µs 11.347 µs 11.354 µs]
                        change: [+0.1555% +0.4850% +0.9582%] (p = 0.02 < 0.05)
                        Change within noise threshold.
Found 15 outliers among 100 measurements (15.00%)
  6 (6.00%) high mild
  9 (9.00%) high severe
dot_4x_interleaved_avx512/avx2_4096
                        time:   [1.4264 µs 1.4269 µs 1.4274 µs]
                        change: [−0.1209% −0.0328% +0.0446%] (p = 0.45 > 0.05)
                        No change in performance detected.
Found 7 outliers among 100 measurements (7.00%)
  5 (5.00%) high mild
  2 (2.00%) high severe

dot_4x_dual_frame_avx512/fallback_16
                        time:   [29.251 ns 29.292 ns 29.335 ns]
                        change: [−8.7204% −8.4366% −8.1818%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 21 outliers among 100 measurements (21.00%)
  2 (2.00%) low severe
  6 (6.00%) low mild
  10 (10.00%) high mild
  3 (3.00%) high severe
dot_4x_dual_frame_avx512/avx2_16
                        time:   [11.583 ns 11.658 ns 11.777 ns]
                        change: [+0.2172% +0.9129% +1.9502%] (p = 0.02 < 0.05)
                        Change within noise threshold.
Found 10 outliers among 100 measurements (10.00%)
  4 (4.00%) high mild
  6 (6.00%) high severe
dot_4x_dual_frame_avx512/fallback_64
                        time:   [161.29 ns 161.32 ns 161.34 ns]
                        change: [−0.1919% −0.0575% +0.0469%] (p = 0.39 > 0.05)
                        No change in performance detected.
Found 4 outliers among 100 measurements (4.00%)
  2 (2.00%) high mild
  2 (2.00%) high severe
dot_4x_dual_frame_avx512/avx2_64
                        time:   [41.018 ns 41.214 ns 41.415 ns]
                        change: [+0.6462% +1.7866% +3.3359%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 20 outliers among 100 measurements (20.00%)
  3 (3.00%) high mild
  17 (17.00%) high severe
dot_4x_dual_frame_avx512/fallback_256
                        time:   [697.17 ns 697.46 ns 697.83 ns]
                        change: [+0.0135% +0.1043% +0.1935%] (p = 0.03 < 0.05)
                        Change within noise threshold.
Found 5 outliers among 100 measurements (5.00%)
  4 (4.00%) high mild
  1 (1.00%) high severe
dot_4x_dual_frame_avx512/avx2_256
                        time:   [157.37 ns 157.43 ns 157.50 ns]
                        change: [−0.0894% +0.0583% +0.2205%] (p = 0.45 > 0.05)
                        No change in performance detected.
Found 7 outliers among 100 measurements (7.00%)
  3 (3.00%) high mild
  4 (4.00%) high severe
dot_4x_dual_frame_avx512/fallback_1024
                        time:   [2.8346 µs 2.8357 µs 2.8366 µs]
                        change: [−0.2638% −0.1943% −0.1274%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 2 outliers among 100 measurements (2.00%)
  2 (2.00%) high mild
dot_4x_dual_frame_avx512/avx2_1024
                        time:   [629.04 ns 630.93 ns 634.55 ns]
                        change: [−0.3225% +0.0990% +0.7376%] (p = 0.80 > 0.05)
                        No change in performance detected.
Found 8 outliers among 100 measurements (8.00%)
  7 (7.00%) high mild
  1 (1.00%) high severe
dot_4x_dual_frame_avx512/fallback_4096
                        time:   [11.422 µs 11.426 µs 11.430 µs]
                        change: [−0.2384% −0.1925% −0.1466%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 11 outliers among 100 measurements (11.00%)
  7 (7.00%) low mild
  1 (1.00%) high mild
  3 (3.00%) high severe
dot_4x_dual_frame_avx512/avx2_4096
                        time:   [2.4915 µs 2.4924 µs 2.4938 µs]
                        change: [−0.0151% +0.0714% +0.1818%] (p = 0.18 > 0.05)
                        No change in performance detected.
Found 14 outliers among 100 measurements (14.00%)
  7 (7.00%) high mild
  7 (7.00%) high severe

     Running benches/inference_bench.rs (target/release/deps/inference_bench-e43b2715150e016e)
WaveNet_Standard_CH16_64samp_48kHz
                        time:   [100.35 µs 100.40 µs 100.44 µs]
                        change: [+6.4691% +6.7109% +6.9531%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 5 outliers among 50 measurements (10.00%)
  2 (4.00%) high mild
  3 (6.00%) high severe

WaveNet_Standard_CH16_32samp_48kHz
                        time:   [51.522 µs 51.569 µs 51.620 µs]
                        change: [+7.0607% +7.4256% +7.9418%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 4 outliers among 50 measurements (8.00%)
  3 (6.00%) high mild
  1 (2.00%) high severe

WaveNet_Standard_CH16_128samp_48kHz
                        time:   [200.94 µs 201.14 µs 201.38 µs]
                        change: [+6.7311% +6.9615% +7.1936%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 7 outliers among 50 measurements (14.00%)
  2 (4.00%) high mild
  5 (10.00%) high severe

WaveNet_Standard_CH16_256samp_48kHz
                        time:   [401.93 µs 402.34 µs 402.83 µs]
                        change: [+6.6900% +7.0172% +7.3757%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 7 outliers among 50 measurements (14.00%)
  3 (6.00%) high mild
  4 (8.00%) high severe

WaveNet_Standard_CH16_512samp_48kHz
                        time:   [804.17 µs 805.11 µs 806.18 µs]
                        change: [+7.0047% +7.1882% +7.3870%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 6 outliers among 50 measurements (12.00%)
  3 (6.00%) high mild
  3 (6.00%) high severe

LSTM_2x16_64samp_48kHz  time:   [10.774 µs 10.796 µs 10.827 µs]
                        change: [−0.3504% −0.0281% +0.3632%] (p = 0.89 > 0.05)
                        No change in performance detected.
Found 5 outliers among 50 measurements (10.00%)
  3 (6.00%) high mild
  2 (4.00%) high severe

LSTM_2x16_32samp_48kHz  time:   [5.8377 µs 5.8819 µs 5.9401 µs]
                        change: [+8.3052% +9.2355% +10.196%] (p = 0.00 < 0.05)
                        Performance has regressed.

LSTM_2x16_128samp_48kHz time:   [23.293 µs 23.466 µs 23.697 µs]
                        change: [+8.5505% +9.4169% +10.263%] (p = 0.00 < 0.05)
                        Performance has regressed.

LSTM_2x16_256samp_48kHz time:   [46.705 µs 47.054 µs 47.524 µs]
                        change: [+8.8206% +9.6395% +10.575%] (p = 0.00 < 0.05)
                        Performance has regressed.

LSTM_2x16_512samp_48kHz time:   [93.633 µs 94.338 µs 95.270 µs]
                        change: [+8.2936% +9.2295% +10.211%] (p = 0.00 < 0.05)
                        Performance has regressed.

LSTM_1x8_Comparison/SIMD_Fused_T3
                        time:   [2.4104 µs 2.4199 µs 2.4369 µs]
                        change: [+5.6065% +6.0355% +6.5657%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 4 outliers among 50 measurements (8.00%)
  1 (2.00%) low mild
  1 (2.00%) high mild
  2 (4.00%) high severe
LSTM_1x8_Comparison/Scalar_Baseline
                        time:   [44.698 µs 44.724 µs 44.757 µs]
                        change: [+0.0788% +0.1286% +0.1916%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 5 outliers among 50 measurements (10.00%)
  1 (2.00%) low mild
  1 (2.00%) high mild
  3 (6.00%) high severe

LSTM_2x16_Comparison/SIMD_Fused_T3
                        time:   [11.068 µs 11.089 µs 11.113 µs]
                        change: [+2.1670% +2.4785% +2.7552%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 3 outliers among 50 measurements (6.00%)
  1 (2.00%) low mild
  1 (2.00%) high mild
  1 (2.00%) high severe
LSTM_2x16_Comparison/Scalar_Baseline
                        time:   [45.474 µs 45.512 µs 45.550 µs]
                        change: [−0.5014% +0.0251% +0.4748%] (p = 0.92 > 0.05)
                        No change in performance detected.
Found 5 outliers among 50 measurements (10.00%)
  2 (4.00%) high mild
  3 (6.00%) high severe

LSTM_1x40_64samp_48kHz  time:   [18.788 µs 18.804 µs 18.818 µs]
                        change: [+0.5209% +0.6902% +0.8685%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high mild

LSTM_2x24_64samp_48kHz  time:   [20.893 µs 20.908 µs 20.928 µs]
                        change: [+0.6429% +1.1294% +1.5340%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high severe

LSTM_1x40_Comparison/SIMD_Fused_T3
                        time:   [18.603 µs 18.614 µs 18.623 µs]
                        change: [−0.6109% −0.2920% +0.0043%] (p = 0.06 > 0.05)
                        No change in performance detected.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high severe
LSTM_1x40_Comparison/Scalar_Baseline
                        time:   [853.78 µs 853.88 µs 853.99 µs]
                        change: [−0.0429% +0.0089% +0.0810%] (p = 0.83 > 0.05)
                        No change in performance detected.
Found 3 outliers among 50 measurements (6.00%)
  2 (4.00%) high mild
  1 (2.00%) high severe

LSTM_2x24_Comparison/SIMD_Fused_T3
                        time:   [20.476 µs 20.490 µs 20.509 µs]
                        change: [−1.6938% −1.1014% −0.6816%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 4 outliers among 50 measurements (8.00%)
  2 (4.00%) high mild
  2 (4.00%) high severe
LSTM_2x24_Comparison/Scalar_Baseline
                        time:   [646.09 µs 646.28 µs 646.50 µs]
                        change: [−0.2612% −0.1724% −0.0837%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 6 outliers among 50 measurements (12.00%)
  2 (4.00%) high mild
  4 (8.00%) high severe

FastMath_tanh_AVX2_256elem
                        time:   [56.433 ns 56.499 ns 56.575 ns]
                        change: [+5.7778% +5.9387% +6.0897%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 5 outliers among 50 measurements (10.00%)
  1 (2.00%) low mild
  2 (4.00%) high mild
  2 (4.00%) high severe

FastMath_tanh_PadeNR2_AVX2_256elem
                        time:   [98.707 ns 98.847 ns 99.001 ns]
                        change: [+0.4998% +0.7396% +0.9607%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 8 outliers among 50 measurements (16.00%)
  5 (10.00%) high mild
  3 (6.00%) high severe

FastMath_tanh_PadeDiv_AVX2_256elem
                        time:   [61.322 ns 61.397 ns 61.469 ns]
                        change: [+2.1245% +2.4998% +2.8197%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) low mild

FastMath_sigmoid_AVX2_256elem
                        time:   [100.00 ns 100.44 ns 101.27 ns]
                        change: [+4.4291% +4.6306% +4.9629%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 2 outliers among 50 measurements (4.00%)
  1 (2.00%) low mild
  1 (2.00%) high severe

WaveNet_Dynamic_Standard_64samp_48kHz
                        time:   [124.95 µs 124.98 µs 125.01 µs]
                        change: [+0.6753% +1.1925% +1.5430%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 4 outliers among 50 measurements (8.00%)
  1 (2.00%) high mild
  3 (6.00%) high severe

LSTM_Dynamic_1x16_64samp_48kHz
                        time:   [4.8463 µs 4.8517 µs 4.8569 µs]
                        change: [+0.4999% +0.7626% +1.0546%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 6 outliers among 50 measurements (12.00%)
  1 (2.00%) high mild
  5 (10.00%) high severe

DotProduct_AVX2_256elem time:   [9.3372 ns 9.3514 ns 9.3664 ns]
                        change: [−2.3659% −1.3783% +0.1224%] (p = 0.02 < 0.05)
                        Change within noise threshold.
Found 4 outliers among 50 measurements (8.00%)
  2 (4.00%) high mild
  2 (4.00%) high severe

DotProduct_AVX2_64elem  time:   [4.1439 ns 4.1542 ns 4.1661 ns]
                        change: [−2.8136% −1.7468% −0.8568%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 5 outliers among 50 measurements (10.00%)
  2 (4.00%) high mild
  3 (6.00%) high severe

Resampler_44100_to_48000_256samp/process_input
                        time:   [4.8270 µs 4.8278 µs 4.8286 µs]
                        change: [+0.3908% +0.4528% +0.5060%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high mild
Resampler_44100_to_48000_256samp/process_input_mono
                        time:   [3.7590 µs 3.7594 µs 3.7598 µs]
                        change: [+1.6376% +1.7875% +1.9028%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 4 outliers among 50 measurements (8.00%)
  1 (2.00%) low mild
  2 (4.00%) high mild
  1 (2.00%) high severe
Resampler_44100_to_48000_256samp/process_output
                        time:   [4.2776 µs 4.2823 µs 4.2874 µs]
                        change: [−0.4693% −0.2569% −0.1176%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 3 outliers among 50 measurements (6.00%)
  2 (4.00%) high mild
  1 (2.00%) high severe
Resampler_44100_to_48000_256samp/process_output_mono
                        time:   [3.3218 µs 3.3222 µs 3.3226 µs]
                        change: [−0.0684% −0.0343% −0.0001%] (p = 0.06 > 0.05)
                        No change in performance detected.
Found 2 outliers among 50 measurements (4.00%)
  1 (2.00%) low mild
  1 (2.00%) high severe

Resampler_96000_to_48000_256samp/process_input
                        time:   [2.5985 µs 2.6001 µs 2.6022 µs]
                        change: [−0.3238% −0.2045% −0.0898%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 2 outliers among 50 measurements (4.00%)
  1 (2.00%) high mild
  1 (2.00%) high severe
Resampler_96000_to_48000_256samp/process_input_mono
                        time:   [2.0480 µs 2.0487 µs 2.0494 µs]
                        change: [−0.2007% −0.1319% −0.0728%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Resampler_96000_to_48000_256samp/process_output
                        time:   [6.1628 µs 6.1684 µs 6.1741 µs]
                        change: [−0.3987% −0.2290% −0.0273%] (p = 0.01 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high severe
Resampler_96000_to_48000_256samp/process_output_mono
                        time:   [4.5542 µs 4.5622 µs 4.5706 µs]
                        change: [−0.8257% −0.5739% −0.3457%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 2 outliers among 50 measurements (4.00%)
  2 (4.00%) high mild

Resampler_48000_bypass_256samp
                        time:   [20.024 ns 20.073 ns 20.116 ns]
                        change: [+2.6012% +2.8183% +3.0585%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high mild

bench_record_64calls    time:   [247.18 ns 247.24 ns 247.31 ns]
                        change: [−0.5056% −0.1990% −0.0270%] (p = 0.13 > 0.05)
                        No change in performance detected.
Found 3 outliers among 50 measurements (6.00%)
  1 (2.00%) low severe
  2 (4.00%) low mild

Prewarm_WaveNet_Standard_2048samp
                        time:   [209.69 µs 210.37 µs 211.01 µs]
                        change: [−2.8752% −1.8686% −0.4751%] (p = 0.00 < 0.05)
                        Change within noise threshold.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high severe

Prewarm_LSTM_2x16_2048samp
                        time:   [359.35 µs 361.42 µs 362.65 µs]
                        change: [+4.3180% +4.5307% +4.6956%] (p = 0.00 < 0.05)
                        Performance has regressed.
Found 2 outliers among 50 measurements (4.00%)
  1 (2.00%) low severe
  1 (2.00%) high mild

head_rechannel_fp32/DenseLayer_16x8_64f_AVX2
                        time:   [320.44 ns 320.85 ns 321.39 ns]
                        change: [−1.6047% −1.3721% −1.1434%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 1 outliers among 50 measurements (2.00%)
  1 (2.00%) high severe
head_rechannel_fp32/DenseLayer_16x8_64f_Scalar
                        time:   [4.9485 µs 4.9572 µs 4.9731 µs]
                        change: [−3.0033% −2.2012% −1.6412%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 7 outliers among 50 measurements (14.00%)
  2 (4.00%) high mild
  5 (10.00%) high severe
head_rechannel_fp32/DenseLayer_8x1_64f_AVX2
                        time:   [153.90 ns 154.10 ns 154.36 ns]
                        change: [−3.6768% −2.7567% −1.9517%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 9 outliers among 50 measurements (18.00%)
  1 (2.00%) low mild
  2 (4.00%) high mild
  6 (12.00%) high severe
head_rechannel_fp32/DenseLayer_8x1_64f_Scalar
                        time:   [340.12 ns 340.84 ns 341.43 ns]
                        change: [−4.5228% −3.4911% −2.6394%] (p = 0.00 < 0.05)
                        Performance has improved.
Found 11 outliers among 50 measurements (22.00%)
  9 (18.00%) low mild
  1 (2.00%) high mild
  1 (2.00%) high severe
head_rechannel_fp32/DenseLayer_16x1_64f_AVX2
                        time:   [180.30 ns 180.38 ns 180.46 ns]
                        change: [−1.5814% −0.5725% +0.1032%] (p = 0.22 > 0.05)
                        No change in performance detected.
Found 5 outliers among 50 measurements (10.00%)
  4 (8.00%) high mild
  1 (2.00%) high severe
head_rechannel_fp32/DenseLayer_16x1_64f_Scalar
                        time:   [631.33 ns 632.24 ns 633.04 ns]
                        change: [−1.5144% −1.2450% −0.9833%] (p = 0.00 < 0.05)
                        Change within noise threshold.

real    12m6,373s
user    14m3,674s
sys     0m10,926s
fabio@notebook:~/nam-rs$
```

### DEPOIS: Logo na conclusão da Sprint 7B

```bash

```
