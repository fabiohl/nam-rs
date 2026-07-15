<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Épico EP-R1: Desarmar as minas de memória

> **Origem:** [TODO-findings.md §EP-R1](TODO-findings.md#ep-r1--desarmar-as-minas-de-memória-r1--r10--r9--primeiro-é-o-núcleo-da-rodada) (Auditoria de Resiliência & Robustez, 2026-07-14)
>
> **Escopo:** R1 (UB formal no WaveNet, **CRÍTICA**) + R10 (clamp defensivo no oversampler, **MÉDIA**) + R9 (`try_clone` no MirroredBuffer, **MÉDIA**).
>
> **Invariante absoluto:** zero alteração de comportamento sonoro. Critério de aceite global: `quality-dashboard.sh --check` sem diff de um único número no contrato.

---

## Sumário dos Sprints

| Sprint | Finding                           | Risco            | Arquivos tocados | Estimativa |
| ------ | --------------------------------- | ---------------- | ---------------- | ---------- |
| **S1** | R1 — Scratch pré-alocado WaveNet  | Médio (hot path) | 5–7              | ~60 min    |
| **S2** | R10 — Clamp defensivo oversampler | Baixo            | 2                | ~20 min    |
| **S3** | R9 — `try_clone` MirroredBuffer   | Baixo            | 3                | ~30 min    |
| **VF** | Verificação final integrada       | —                | 0                | ~15 min    |

---

## Sprint S1 — R1: Scratch pré-alocado no WaveNet layer estático

> **Ref:** [TODO-findings.md §R1](TODO-findings.md#r1--ub-formal-confirmado-slicefrom_raw_parts_mut-sobre-maybeuninitf321024-não-inicializado--crítica) (L62-114)
>
> **Objetivo:** Eliminar o UB formal (`&mut [f32]` sobre `MaybeUninit` não inicializado) no hot path `process_block_internal` do `WaveNetLayer` estático (const-generic), migrando os buffers de stack para `AlignedVec<f32>` pré-alocados na struct — padrão já consolidado em `WaveNetLayerDyn` ([`layer_dyn.rs`](src/models/wavenet/layer_dyn.rs)).
>
> **Risco:** Médio. Toca o hot path de inferência do WaveNet A1. Mitigado pelo contrato bit-exact e pelos goldens.

### T1.1 — Adicionar campos `scratch_mixin` e `scratch_conv` à struct `WaveNetLayer` [DONE]

- **Arquivo:** [`src/models/wavenet/layer.rs`](src/models/wavenet/layer.rs)

- **Ação:**

  1. Remover `use core::mem::MaybeUninit;`.

  2. Adicionar `use crate::math::common::AlignedVec;`.

  3. Adicionar dois campos públicos à struct:

     ```rust
     /// Pre-allocated scratch buffer for conditioning mixin output
     /// (size: `CH * WAVENET_MAX_NUM_FRAMES`).
     pub scratch_mixin: AlignedVec<f32>,
     /// Pre-allocated scratch buffer for Conv1D + mixin intermediate results
     /// (size: `CH * WAVENET_MAX_NUM_FRAMES`).
     pub scratch_conv: AlignedVec<f32>,
     ```

  4. Trocar `#[derive(Clone)]` por implementação manual de `Clone` (os novos campos `AlignedVec` suportam `Clone`, mas a implementação manual garante documentação e controle explícito).

- **Critério de aceite:** `cargo check` passa; struct compila com os novos campos.

### T1.2 — Reescrever `process_block_internal` sem `MaybeUninit` [DONE]

- **Arquivo:** [`src/models/wavenet/layer.rs`](src/models/wavenet/layer.rs)

- **Ação:**

  1. Mudar assinatura de `&self` para `&mut self`.

  2. Substituir os dois blocos `MaybeUninit` + `from_raw_parts_mut` por slicing seguro:

     ```rust
     let mixin_out = &mut self.scratch_mixin[..num_frames * CH];
     let conv_slice = &mut self.scratch_conv[..num_frames * CH];
     ```

  3. Converter os acessos a `mixin_out_ptr` no dual-frame tiling (linhas 80-81: `from_raw_parts` para `mixin_f0/f1`) para indexação segura dos slices de scratch.

  4. Remover o bloco `const { assert!(...) }` (limitação ao buffer fixo de 1024; não mais relevante).

  5. Manter `debug_assert!(num_frames * CH <= self.scratch_mixin.len(), ...)` como safety net.

  6. O bloco `unsafe` externo continua necessário apenas para as chamadas a `SimdMath` e `get_unchecked` do `layer_buffer`; os blocos `unsafe` de `from_raw_parts_mut` sobre uninit desaparecem.

- **Critério de aceite:** Zero `from_raw_parts_mut` sobre `MaybeUninit` no arquivo. `cargo check` passa.

### T1.3 — Adaptar `WaveNetLayerArray` para `&mut WaveNetLayer` [DONE]

- **Arquivo:** [`src/models/wavenet/layer_array.rs`](src/models/wavenet/layer_array.rs)
- **Ação:**
  1. Mudar `self.layers.iter()` (L91) para `self.layers.iter_mut()`.
  2. A variável `layer` no loop passa de `&WaveNetLayer<...>` para `&mut WaveNetLayer<...>`.
  3. Nenhuma outra alteração no fluxo de orquestração (contexts, states, prefetch).
- **Critério de aceite:** `cargo check` passa; loop de inferência compila com referências mutáveis.

### T1.4 — Atualizar construtores de `WaveNetLayer` (loader + testes) [DONE]

- **Arquivos afetados:**

  - [`src/loader/dispatcher/wavenet/standard.rs`](src/loader/dispatcher/wavenet/standard.rs) — loader de produção
  - [`src/models/wavenet/wavenet_test.rs`](src/models/wavenet/wavenet_test.rs)
  - [`src/models/wavenet/dynamic_parity_test.rs`](src/models/wavenet/dynamic_parity_test.rs)
  - [`src/models/wavenet/wavenet_ch12_diagnostic_test.rs`](src/models/wavenet/wavenet_ch12_diagnostic_test.rs)

- **Ação:** Em todos os sites que constroem `WaveNetLayer { conv1d, input_mixin, one_by_one }`, adicionar:

  ```rust
  scratch_mixin: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)?,
  scratch_conv: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)?,
  ```

  Em testes que usam `unwrap()`/`expect()` (permitido fora de produção), ajustar para `.expect("scratch alloc")`.

- **Critério de aceite:** `cargo check --all-targets` passa; nenhum call-site esquecido.

### T1.5 — Checkpoint S1: Verificação intermediária [DONE]

- **Ação:**
  1. `cargo clippy --all-targets` — limpo (nenhum warning novo).
  2. `cargo test` — suite rápida verde.
  3. `grep -rn "MaybeUninit" src/ | grep -v test` — **zero hits** (confirmação R1).
- **Critério de aceite:** Todos os 3 checks verdes. **Não prosseguir para S2 sem este checkpoint.**

---

## Sprint S2 — R10: Clamp defensivo no oversampler

> **Ref:** [TODO-findings.md §R10](TODO-findings.md#r10--oversamplers-debug_assert--copy_nonoverlapping--contrato-de-block-size-vira-ub-silencioso-no-release--média) (L392-411)
>
> **Objetivo:** Proteger `upsample` e `downsample` contra hosts que violam `max_frames_count`, usando clamp branchless (`input.len().min(self.max_samples)`) nos braços X2/X4. Custo: 1 `cmp+cmov` por chamada — irrelevante.
>
> **Risco:** Baixo. Correção defensiva; o braço `OsStages::Off` já clampa.

### T2.1 — Adicionar clamp defensivo em `upsample` e `downsample` [DONE]

- **Arquivo:** [`src/dsp/oversample.rs`](src/dsp/oversample.rs)

- **Ação em `upsample` (L188):**

  1. No início do método, antes do `debug_assert!`:

     ```rust
     let n_in = input.len().min(self.max_samples);
     let input = &input[..n_in];
     ```

  2. Manter os `debug_assert!` existentes como documentação do contrato (disparam em debug builds se o host violar).

- **Ação em `downsample` (L215):**

  1. Mesmo padrão:

     ```rust
     let max_os = self.max_samples * self.factor.multiplier();
     let n_in = input.len().min(max_os);
     let input = &input[..n_in];
     ```

- **Nota sobre flag RT_STATUS:** O oversampler não tem acesso ao `RtStatusFlags`. A sinalização de contrato violado já é feita no caller (`process.rs` seta `RT_STATUS_HOST_CONTRACT_VIOLATION`). O clamp aqui é a segunda linha de defesa — silencioso por design. Melhoria de telemetria fica para EP-R4.

- **Critério de aceite:** `cargo check` passa; `debug_assert!` preservados.

### T2.2 — Novo teste: input acima de `max_samples` é clampado [DONE]

- **Arquivo:** [`src/dsp/oversample_test.rs`](src/dsp/oversample_test.rs)

- **Ação:** Adicionar testes para os 3 fatores (Off, X2, X4) verificando que input maior que `max_samples` é clampado sem UB:

  ```rust
  #[test]
  fn test_oversized_input_clamped_off() { ... }

  #[test]
  fn test_oversized_input_clamped_x2() { ... }

  #[test]
  fn test_oversized_input_clamped_x4() { ... }

  #[test]
  fn test_oversized_downsample_clamped_x2() { ... }
  ```

  Cada teste: cria engine com `max_samples = 64`, fornece input de 96 amostras, verifica que o retorno ≤ `max * multiplier`.

- **Critério de aceite:** Testes novos passam; testes existentes inalterados.

### T2.3 — Checkpoint S2: Verificação intermediária [DONE]

- **Ação:**
  1. `cargo clippy --all-targets` — limpo.
  2. `cargo test` — suite verde (incluindo novos testes de clamp).
- **Critério de aceite:** Ambos verdes. **Não prosseguir para S3 sem este checkpoint.**

---

## Sprint S3 — R9: `try_clone` no MirroredBuffer

> **Ref:** [TODO-findings.md §R9](TODO-findings.md#r9--mirroredbufferclone-usa-panic_any--panic-pode-cruzar-a-fronteira-ffi-do-clap--média) (L368-389)
>
> **Objetivo:** Eliminar `panic_any` no `Clone` de `MirroredBuffer` (produz `Box<dyn Any>` incompatível com `catch_unwind` do clack-plugin), adicionando `try_clone() -> io::Result<Self>` como API primária falível.
>
> **Decisão arquitetural:** Manter `impl Clone` (para compatibilidade com `#[derive(Clone)]` em `WaveNetLayerState` e outros consumidores), mas refatorá-lo para chamar `try_clone().expect(...)` com mensagem estruturada — já uma melhoria substancial (panic padrão em vez de `panic_any`).
>
> **Risco:** Baixo. Caminho off-RT (ativação/reconfiguração).

### T3.1 — Adicionar `try_clone()` ao `MirroredBuffer`

- **Arquivo:** [`src/dsp/mirror_buf.rs`](src/dsp/mirror_buf.rs)

- **Ação:**

  1. Adicionar método público:

     ```rust
     /// Fallible clone — returns `Err` on allocation failure instead of panicking.
     ///
     /// Preferred over `Clone::clone()` in CLAP activation paths where panic
     /// would cross the FFI boundary.
     pub fn try_clone(&self) -> std::io::Result<Self>
     where
         T: Clone,
     {
         let mut new_buf = Self::new(self.size_elements)?;
         new_buf[..self.size_elements].clone_from_slice(&self[..self.size_elements]);
         Ok(new_buf)
     }
     ```

  2. Refatorar `impl Clone`:

     ```rust
     impl<T: Clone> Clone for MirroredBuffer<T> {
         #[cold]
         fn clone(&self) -> Self {
             self.try_clone()
                 .expect("MirroredBuffer::clone: allocation failed (use try_clone for fallible path)")
         }
     }
     ```

- **Critério de aceite:** Zero `panic_any` no arquivo. `cargo check` passa.

### T3.2 — Adicionar `try_clone()` ao `WaveNetLayerState`

- **Arquivo:** [`src/models/wavenet/common.rs`](src/models/wavenet/common.rs)

- **Ação:** Adicionar método falível que propaga erros de alocação:

  ```rust
  /// Fallible clone for activation paths where panic must not cross FFI.
  pub fn try_clone(&self) -> std::io::Result<Self> {
      Ok(Self {
          layer_buffer: self.layer_buffer.try_clone()?,
          buffer_start: self.buffer_start,
          receptive_field_size: self.receptive_field_size,
      })
  }
  ```

- **Critério de aceite:** `cargo check` passa; `WaveNetLayerState` continua derivando `Clone` (para usos não-FFI), mas agora tem `try_clone` disponível para ativação CLAP.

### T3.3 — Teste de fault injection para `try_clone`

- **Arquivo:** [`tests/models/mirror_buf_fault_injection.rs`](tests/models/mirror_buf_fault_injection.rs)

- **Ação:** Adicionar teste que exercita `try_clone` sob falha simulada:

  ```rust
  #[test]
  #[cfg(target_os = "linux")]
  fn test_mirror_buf_try_clone_under_fault() {
      let buf = MirroredBuffer::<f32>::new(1024)
          .expect("initial allocation should succeed");

      set_simulate_fail(true);
      let result = buf.try_clone();
      assert!(result.is_err(),
          "try_clone should return Err under simulated mmap failure");

      set_simulate_fail(false);
      let cloned = buf.try_clone();
      assert!(cloned.is_ok(),
          "try_clone should succeed without failure simulation");
  }
  ```

- **Critério de aceite:** Teste novo passa. Teste existente `test_mirror_buf_mmap_failure_injection` inalterado.

### T3.4 — Checkpoint S3: Verificação intermediária

- **Ação:**
  1. `cargo clippy --all-targets` — limpo.
  2. `cargo test` — suite verde.
  3. `grep -rn "panic_any" src/` — **zero hits** (confirmação R9).
- **Critério de aceite:** Todos os 3 checks verdes.

---

## VF — Verificação Final Integrada

> **Gate de aceite do épico inteiro.** Rodar apenas após os 3 sprints com checkpoints verdes.

### VF.1 — Lints completos

- `utils/lints.sh` — fmt + SPDX + check + clippy, **zero erros/warnings**.

### VF.2 — Suite rápida completa

- `utils/tests-quick.sh` — **verde total**, incluindo todos os testes novos (S1/S2/S3).

### VF.3 — Contrato de qualidade bit-exact

- `utils/quality-dashboard.sh --check docs/quality-contract.txt` — **zero diff**.
- Se qualquer número mudar: **PARAR**, reverter, e investigar. A correção introduziu bug.

### VF.4 — Confirmações de eliminação

- `grep -rn "MaybeUninit" src/ | grep -v test` → **zero hits** (R1 eliminado).
- `grep -rn "panic_any" src/` → **zero hits** (R9 eliminado).

### VF.5 — Solicitar tests-long ao operador

- Solicitar que o operador rode `utils/tests-long.sh` na próxima janela noturna.
- **Não bloquear o épico** — os testes longos são validação complementar.

---

## Notas de risco e mitigação

| Risco                                                   | Mitigação                                                                                                   |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Regressão de performance no hot path (R1)               | Benchmark `inference_bench`/`regression_gate` sem regressão > ruído; scratch heap está em L1 quente         |
| `&mut self` propaga além do previsto                    | Mapeamento completo de call-sites feito; `WaveNetLayerArray` já é `&mut self`                               |
| Clamp do oversampler mascara bug do host                | `debug_assert!` preservados para builds debug; flag `RT_STATUS_HOST_CONTRACT_VIOLATION` já existe no caller |
| `try_clone` não cobrindo todos os call-sites de `Clone` | `Clone` mantido como wrapper de `try_clone`; todos os call-sites existentes continuam funcionais            |
