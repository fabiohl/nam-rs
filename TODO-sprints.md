<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Épicos de Resiliência & Robustez (Auditoria 2026-07-14)

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

### T3.1 — Adicionar `try_clone()` ao `MirroredBuffer` [DONE]

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

### T3.2 — Adicionar `try_clone()` ao `WaveNetLayerState`[DONE]

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

### T3.3 — Teste de fault injection para `try_clone` [DONE]

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

### T3.4 — Checkpoint S3: Verificação intermediária [DONE]

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

---

---

## Épico EP-R2 — Ciclo de vida à prova de host hostil

> **Origem:** [TODO-findings.md §EP-R2](TODO-findings.md#ep-r2--ciclo-de-vida-à-prova-de-host-hostil-r2--r13--r11--r3) (Auditoria de Resiliência & Robustez, 2026-07-14)
>
> **Escopo:** R2 (UAF potencial no file-dialog, **CRÍTICA**) + R13 (`join()` indefinido no destroy da GUI, **MÉDIA**) + R11 (drenagem final do GC não garantida, **MÉDIA**) + R3 (double-SIGINT sem documentação de shutdown, **ALTA**).
>
> **Pré-requisito:** EP-R1 concluído e `quality-dashboard.sh --check` verde.
>
> **Invariante absoluto:** zero alteração de comportamento sonoro. Critério de aceite global: `quality-dashboard.sh --check` sem diff de um único número no contrato. Arquivos de estado (`state.rs`, `shared.rs`, GC) tocados — risco de lifecycle; mitigado pelo harness `clap-validator` e pelos testes de lifecycle destrutivos.

---

## EP-R2 — Sumário dos Sprints

| Sprint | Finding                                        | Risco                 | Arquivos tocados | Estimativa |
| ------ | ---------------------------------------------- | --------------------- | ---------------- | ---------- |
| **S4** | R2 — `Arc<DialogSharedState>` no file-dialog   | Alto (lifecycle CLAP) | 4–6              | ~90 min    |
| **S5** | R13 — Watchdog no `join()` da janela flutuante | Médio (lifecycle GUI) | 2                | ~30 min    |
| **S6** | R11 — Drenagem final do GC no destroy          | Médio (GC-cascade)    | 3–4              | ~45 min    |
| **S7** | R3 — Documentação + Acquire no double-SIGINT   | Baixo                 | 2                | ~20 min    |
| **VF** | Verificação final integrada                    | —                     | 0                | ~15 min    |

---

## Sprint S4 — R2: Eliminar UAF no file-dialog com `Arc<DialogSharedState>`

> **Ref:** [TODO-findings.md §R2](TODO-findings.md#r2--use-after-free-potencial-threads-detached-do-file-dialog-acessam-namclapshared-via-endereço-cru--crítica) (L118-163)
>
> **Objetivo:** Substituir o anti-padrão `usize`→ponteiro cru + `alive_fence` TOCTOU pela propriedade `Arc<DialogSharedState>` — estado compartilhado GUI↔dialog isolado num objeto de vida própria. UAF estruturalmente impossível: se o plugin morrer enquanto o diálogo está aberto, as threads escrevem em memória Arc que sobrevive até o último clone ser dropado. Nenhuma mudança de comportamento sonoro.
>
> **Risco:** Alto. Toca o protocolo de comunicação GUI↔file-dialog e o `NamClapMainThread`. Mitigado: (a) o `Arc` é trivialmente `Send+Sync`; (b) os campos movidos são apenas `AtomicBool` + `Mutex<Option<PathBuf>>` — já o que o código acessa hoje; (c) `alive_fence` permanece no `ColdShared` para outros usos; (d) o harness `clap-validator` + testes de lifecycle existentes detectam regressão.

### T4.1 — Criar `DialogSharedState` e `IrDialogSharedState` [DONE]

- **Arquivo (novo):** [`src/clap/gui/ui/zones/dialog_state.rs`](src/clap/gui/ui/zones/dialog_state.rs)

- **Ação:**

  1. Criar o arquivo com os dois tipos que encapsulam o estado necessário às threads de diálogo:

     ```rust
     // SPDX-License-Identifier: Apache-2.0
     // Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
     //! Estado compartilhado entre a thread principal e as threads de diálogo de arquivo.
     //!
     //! Substitui o padrão `usize`→ponteiro cru que expunha UAF potencial (R2).
     //! As threads de diálogo capturam um `Arc` clone deste estado; se o plugin
     //! for destruído enquanto o diálogo está aberto, elas escrevem em memória
     //! ainda viva (o `Arc` não dropa até o último clone ser liberado).

     use std::path::PathBuf;
     use std::sync::Mutex;
     use std::sync::atomic::AtomicBool;

     /// Estado compartilhado entre a thread do plugin e as threads de file-dialog de modelo.
     pub(crate) struct DialogSharedState {
         /// Sinaliza que o GUI ainda está vivo (apenas para log/otimização; não é barreira de segurança).
         pub alive: AtomicBool,
         /// Modelo pendente a ser carregado pelo Main Thread.
         pub pending_model: Mutex<Option<PathBuf>>,
         /// Indica se o carregamento assíncrono está em progresso.
         pub loading: AtomicBool,
     }

     /// Estado compartilhado entre a thread do plugin e as threads de file-dialog de IR.
     pub(crate) struct IrDialogSharedState {
         /// Sinaliza que o GUI ainda está vivo (apenas para log/otimização; não é barreira de segurança).
         pub alive: AtomicBool,
         /// Path de IR pendente a ser carregado pelo Main Thread.
         pub pending_ir: Mutex<Option<PathBuf>>,
         /// Indica se o carregamento assíncrono de IR está em progresso.
         pub ir_loading: AtomicBool,
     }
     ```

  2. Registrar `pub(crate) mod dialog_state;` em [`src/clap/gui/ui/zones/mod.rs`](src/clap/gui/ui/zones/mod.rs).

- **Critério de aceite:** `cargo check` passa; tipos compilam sem warnings.

### T4.2 — Refatorar `spawn_file_dialog` e `spawn_ir_file_dialog` [DONE]

- **Arquivo:** [`src/clap/gui/ui/zones/file_dialogs.rs`](src/clap/gui/ui/zones/file_dialogs.rs)

- **Ação:**

  1. Remover `use crate::clap::plugin::NamClapShared;`.

  2. Adicionar `use super::dialog_state::{DialogSharedState, IrDialogSharedState};` e `use std::sync::Arc;`.

  3. Reescrever `spawn_file_dialog`:

     ```rust
     pub(crate) fn spawn_file_dialog(
         state: Arc<DialogSharedState>,
         host_static: HostSharedHandle<'static>,
     ) -> std::thread::JoinHandle<()> {
         std::thread::spawn(move || {
             let (tx, rx) = std::sync::mpsc::channel();
             std::thread::spawn(move || {
                 let path_opt = rfd::FileDialog::new()
                     .add_filter("NAM Model", &["nam", "namb"])
                     .pick_file();
                 let _ = tx.send(path_opt);
             });
             match rx.recv_timeout(std::time::Duration::from_secs(120)) {
                 Ok(Some(path)) => {
                     // alive é hint — mesmo false, escrever em Arc<..> é seguro (não é UAF)
                     if let Ok(mut guard) = state.pending_model.lock() {
                         *guard = Some(path);
                         host_static.request_callback();
                     }
                 }
                 Ok(None) => {
                     state.loading.store(false, std::sync::atomic::Ordering::Release);
                 }
                 Err(_) => {
                     state.loading.store(false, std::sync::atomic::Ordering::Release);
                     // log de timeout via host_static (mantido como antes)
                     if let (Some(log), Ok(c_msg)) = (
                         host_static.get_extension::<clack_extensions::log::HostLog>(),
                         std::ffi::CString::new("NAM-rs: File dialog portal timed out after 120s"),
                     ) {
                         log.log(&host_static, clack_extensions::log::LogSeverity::Warning, &c_msg);
                     }
                 }
             }
         })
     }
     ```

  4. Reescrever `spawn_ir_file_dialog` com o mesmo padrão usando `Arc<IrDialogSharedState>`.

  5. Remover todo `unsafe { &*(shared_addr as *const NamClapShared) }`.

- **Critério de aceite:** Zero `usize`→ponteiro (`shared_addr`) no arquivo. `cargo check` passa.

### T4.3 — Armazenar `JoinHandle` e `Arc` no `NamClapMainThread` [DONE]

- **Arquivo:** [`src/clap/plugin/main_thread/mod.rs`](src/clap/plugin/main_thread/mod.rs)

- **Ação:**

  1. Adicionar campos ao `NamClapMainThread`:

     ```rust
     /// Handle da thread de file-dialog de modelo (se ativa). Joinado no teardown.
     pub(crate) dialog_handle: Option<std::thread::JoinHandle<()>>,
     /// Estado compartilhado com a thread de file-dialog de modelo.
     pub(crate) dialog_state: Option<Arc<crate::clap::gui::ui::zones::dialog_state::DialogSharedState>>,
     /// Handle da thread de file-dialog de IR (se ativa). Joinado no teardown.
     pub(crate) ir_dialog_handle: Option<std::thread::JoinHandle<()>>,
     /// Estado compartilhado com a thread de file-dialog de IR.
     pub(crate) ir_dialog_state: Option<Arc<crate::clap::gui::ui::zones::dialog_state::IrDialogSharedState>>,
     ```

  2. Inicializá-los como `None` no construtor.

- **Critério de aceite:** `cargo check` passa; nenhum outro campo removido.

### T4.4 — Integrar os handles no `teardown_gui_resources` e no site de chamada [DONE]

- **Arquivo principal:** [`src/clap/extensions/gui.rs`](src/clap/extensions/gui.rs)

- **Arquivo secundário:** local onde `spawn_file_dialog` é chamado (UI state/zones/controls)

- **Ação:**

  1. Em `teardown_gui_resources()`, após o join do floating handle, adicionar drenagem dos dialog handles com deadline curto (mesma lógica que S5-T5.1):

     ```rust
     // Drain file-dialog handles — join com deadline (R2 + R13)
     for handle_opt in [&mut self.dialog_handle, &mut self.ir_dialog_handle] {
         if let Some(h) = handle_opt.take() {
             // Sinaliza alive = false para que a thread pule trabalho desnecessário
             // (não é barreira de segurança — apenas hint de performance)
             // A thread termina naturalmente ao completar recv_timeout
             let _ = h.join(); // bloqueio aceitável: máx 120s (watchdog em S5 cobre floating)
             // NOTA: file-dialog externo (rfd) bloqueia em recv_timeout(120s) por design;
             // não é possível interromper sem matar a thread. Documentado intencionalmente.
         }
     }
     ```

  2. No site de chamada de `spawn_file_dialog` (dentro da UI), atualizar a chamada para:

     - Criar `Arc<DialogSharedState>` com os campos corretos.
     - Armazenar o `Arc` no `dialog_state` do `NamClapMainThread` (via callback ou ref mutável disponível no contexto CLAP).
     - Armazenar o `JoinHandle` em `dialog_handle`.

  3. **Nota arquitetural:** O `ColdShared::ui_pending_model` e `ui_loading` continuam existindo como o canal de comunicação com o Main Thread (a thread de diálogo copia o caminho para o `DialogSharedState`; o `on_main_thread` lê de lá e propaga para o `ColdShared` antes de disparar o load). Isso preserva o protocolo de carregamento existente sem mudar a interface com o RT thread.

- **Critério de aceite:** Zero `unsafe { &*(shared_addr as *const NamClapShared) }` em todo `src/clap/gui/`. `cargo check --all-features` passa.

### T4.5 — Elevar `alive_fence` ordering: `Acquire`/`Release`

- **Arquivo:** [`src/clap/plugin/shared.rs`](src/clap/plugin/shared.rs) (Drop impl, L252)

- **Arquivo:** [`src/clap/gui/window/state.rs`](src/clap/gui/window/state.rs) (safe_shared, L190)

- **Ação:**

  1. No `Drop for NamClapShared` (L252): `store(false, Ordering::Release)` — já comunicando happens-before.
  2. Em `safe_shared()` (L190): `load(Ordering::Acquire)` — pares Release/Acquire garantem que o drop é visível.
  3. No `spawn_file_dialog` antigo (já substituído em T4.2): não mais relevante — eliminado.

  > **Nota:** O `alive_fence` do `ColdShared` continua existindo para o harness da janela embedded (`safe_shared`). Sua semântica agora é apenas de hint de performance: a barreira de segurança contra UAF é o `Arc`.

- **Critério de aceite:** `grep -rn "Ordering::Relaxed" src/clap/plugin/shared.rs | grep alive_fence` → **zero hits**. `cargo check` passa.

### T4.6 — Testes de lifecycle destrutivo do file-dialog [DONE]

- **Arquivo:** [`src/clap/gui/ui/zones/file_dialogs_test.rs`](src/clap/gui/ui/zones/file_dialogs_test.rs) (novo, ou extender `file_dialogs.rs`)

- **Ação:** Adicionar testes que cobrem os cenários de destroy-with-dialog:

  ```rust
  /// Simula destroy do plugin enquanto o Arc de diálogo ainda tem clones vivos.
  /// Verifica que escrever no Arc após "morte" do plugin não causa UAF.
  #[test]
  fn test_dialog_state_outlives_plugin_drop() {
      use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
      let state = Arc::new(super::dialog_state::DialogSharedState {
          alive: AtomicBool::new(true),
          pending_model: std::sync::Mutex::new(None),
          loading: AtomicBool::new(true),
      });
      let state_clone = Arc::clone(&state);
      // Simula drop do plugin (alive → false no ColdShared; Arc próprio não dropa)
      state.alive.store(false, Ordering::Release);
      drop(state); // plugin "morreu"
      // Thread de diálogo ainda tem clone — escrever é seguro (sem UAF)
      state_clone.loading.store(false, Ordering::Release);
      assert!(!state_clone.alive.load(Ordering::Acquire));
  }

  /// Verifica que spawn_file_dialog retorna JoinHandle (não detached).
  #[test]
  fn test_spawn_returns_joinhandle() {
      // Não abre janela real (rfd não disponível em CI headless).
      // Testa apenas que a assinatura compila e que o handle é joinável.
      // Cobertura de integração real: tests-long com clap-validator.
      // (Teste compile-only — marcado #[ignore] para CI headless)
  }
  ```

  Adicionar o módulo de teste via `#[cfg(test)] mod file_dialogs_test;` em `file_dialogs.rs`.

- **Critério de aceite:** `cargo test` (suite rápida) verde. `grep -rn "shared_addr" src/clap/gui/` → **zero hits**.

### T4.7 — Checkpoint S4: Verificação intermediária [DONE]

- **Ação:**
  1. `cargo clippy --all-targets` — limpo.
  2. `cargo test` — verde.
  3. `grep -rn "shared_addr" src/clap/gui/` → **zero hits** (UAF eliminado).
  4. `grep -rn "from_raw_parts\|as \*const NamClapShared" src/clap/gui/` → **zero hits**.
- **Critério de aceite:** Todos verdes. **Não prosseguir para S5 sem este checkpoint.**

---

## Sprint S5 — R13: Watchdog com deadline no `join()` da janela flutuante

> **Ref:** [TODO-findings.md §R13](TODO-findings.md#r13--guidestroy-join-da-janela-flutuante-pode-bloquear-a-main-thread-do-host-indefinidamente--média) (L469-486)
>
> **Objetivo:** Substituir o `handle.join()` bloqueante no `teardown_gui_resources()` por um loop de polling com deadline de 2 s e abandono controlado. Impede que uma janela X11/Wayland com event loop travado congele o DAW inteiro.
>
> **Risco:** Médio. Toca o `teardown_gui_resources()` — ponto crítico do lifecycle CLAP. Mitigado: (a) o comportamento no caso comum (janela fecha normalmente) é idêntico; (b) o abandono após timeout é leak controlado e documentado, preferível ao freeze; (c) o `close_signal` já existe e já é setado antes do join.

### T5.1 — Reescrever `teardown_gui_resources` com polling + deadline [DONE]

- **Arquivo:** [`src/clap/extensions/gui.rs`](src/clap/extensions/gui.rs)

- **Ação:** Substituir o corpo de `teardown_gui_resources`:

  ```rust
  fn teardown_gui_resources(&mut self) {
      if let Some(signal) = self.floating_close_signal.take() {
          signal.store(true, Ordering::Release);
      }
      if let Some(handle) = self.floating_thread_handle.take() {
          // R13: watchdog com deadline de 2 s para evitar freeze do host.
          // Se a janela X11/Wayland não responder ao close_signal dentro do prazo,
          // abandonamos o handle (leak controlado de 1 thread — preferível a congelar o DAW).
          let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
          loop {
              if handle.is_finished() {
                  let _ = handle.join();
                  break;
              }
              if std::time::Instant::now() >= deadline {
                  // Abandono controlado: a thread vive até o processo terminar.
                  // O sistema operacional recolhe todos os recursos (fds, mapeamentos)
                  // no exit do processo. Ver docs/architecture.md §lifecycle-r13.
                  log::warn!(
                      "NAM-rs: floating window thread did not exit within 2 s on destroy \
                       — abandoning handle to avoid host freeze (R13 controlled leak)"
                  );
                  // handle é movido para fora do `if let` e dropado aqui,
                  // sem join — a thread continua rodando detached.
                  break;
              }
              std::thread::sleep(std::time::Duration::from_millis(10));
          }
      }
      if let Some(mut window_handle) = self.window_handle.take() {
          window_handle.close();
      }
  }
  ```

  > **Decisão de design:** `Ordering::Release` no `close_signal.store` (antes era `Relaxed`) para garantir happens-before com o event loop da janela que lê o sinal com `Acquire`. Isso maximiza a chance de a janela ver o sinal antes do deadline.

- **Critério de aceite:** `teardown_gui_resources` não bloqueia por mais de `2 s + ε` em nenhum caminho. `cargo check` passa.

### T5.2 — Documentar o trade-off no `architecture.md` [DONE]

- **Arquivo:** [`docs/architecture.md`](docs/architecture.md)

- **Ação:** Adicionar entrada na seção de lifecycle (criar seção se não existir):

  ```markdown
  ### R13 — GUI floating thread lifecycle (destroy watchdog)

  `gui.destroy()` sinaliza `close_signal = true` e aguarda até 2 s pela thread da
  janela flutuante via `is_finished()` polling. Se o event loop não responder
  (X11/Wayland degradado), o handle é abandonado (leak controlado: 1 thread, sem
  fds extras). O sistema operacional recolhe todos os recursos no exit do processo.
  Preferível a congelar a main thread do host (freeze do DAW).

  Referência: [TODO-findings.md §R13](../TODO-findings.md#r13).
  ```

- **Critério de aceite:** Seção existe em `docs/architecture.md`. `cargo check` inalterado.

### T5.3 — Checkpoint S5: Verificação intermediária [DONE]

- **Ação:**
  1. `cargo clippy --all-targets` — limpo.
  2. `cargo test` — verde.
  3. Revisão manual: `teardown_gui_resources` usa `Ordering::Release` no store do `close_signal`.
- **Critério de aceite:** Todos verdes. **Não prosseguir para S6 sem este checkpoint.**

---

## Sprint S6 — R11: Drenagem final garantida do GC no destroy

> **Ref:** [TODO-findings.md §R11](TODO-findings.md#r11--gc-cascade-drenagem-final-não-garantida-no-destroy--itens-órfãos--média) (L415-443)
>
> **Objetivo:** Garantir que todo item em trânsito no GC-cascade (`gc_rx` + `gc_overflow`) seja drenado antes do drop do `NamClapShared`. Escolha arquitetural: drenagem explícita em `NamClapMainThread::on_main_thread` / callback de plugin-stop, com documentação clara do porquê não usar `Drop for NamClapShared` (o main thread é o único com acesso ao consumer, não o drop do Shared).
>
> **Risco:** Médio. A ordem de drop do `gc_rx` (que vive no `NamClapMainThread`) vs. o `NamClapShared` tem sutilezas CLAP. Mitigado: (a) `drain_gc_channels` já existe e é reutilizado; (b) a janela de leak é finita (N slots × tamanho do item); (c) testes de stress GC existem e detectam double-free.

### T6.1 — Adicionar `drain_gc_final` ao teardown do Main Thread [DONE]

- **Arquivo:** [`src/clap/plugin/main_thread/mod.rs`](src/clap/plugin/main_thread/mod.rs)

- **Ação:**

  1. Localizar o ponto onde o `NamClapMainThread` é desativado/destruído (ex: implementação do `PluginMainThread::destroy` ou drop). Adicionar chamada explícita de drenagem final:

     ```rust
     /// Drena o GC-cascade completamente antes do plugin ser destruído.
     ///
     /// Chamado no último `on_main_thread` ou no `deactivate()` final.
     /// Garante que nenhum `Box<StaticModel>` ou similar fique vivo no
     /// `GcOverflowBuffer` após o plugin morrer (R11).
     pub(crate) fn drain_gc_final(&mut self) {
         use crate::common::spsc::drain_gc_channels;
         // Drena o canal SPSC principal
         let drained = drain_gc_channels(
             &mut self.gc_rx,
             &self.shared.cold.gc_overflow,
             &self.shared.cold.rt_status,
         );
         if drained > 0 {
             log::debug!("NAM-rs: GC drain final — {} item(s) liberados no destroy (R11)", drained);
         }
         // Segunda passagem: overflow pode ter sido preenchido pelo RT entre a primeira
         // drenagem e agora (race benigna — a segunda passagem fecha a janela)
         let _ = drain_gc_channels(
             &mut self.gc_rx,
             &self.shared.cold.gc_overflow,
             &self.shared.cold.rt_status,
         );
     }
     ```

  2. Chamar `self.drain_gc_final()` nos pontos de finalização:

     - Em `deactivate()` (após devolver canais ao `ColdShared`).
     - No callback `on_main_thread` quando `alive_fence` é `false` (plugin sendo destruído).

- **Critério de aceite:** `cargo check` passa. `drain_gc_final` é chamado em `deactivate`.

### T6.2 — Documentar a decisão no `gc.rs` [DONE]

- **Arquivo:** [`src/common/spsc/gc.rs`](src/common/spsc/gc.rs)

- **Ação:** Adicionar comentário de módulo documentando a política de lifecycle:

  ```rust
  //! ## Política de drenagem final (R11)
  //!
  //! O `GcOverflowBuffer` e o canal `gc_rx` devem ser drenados pelo Main Thread
  //! antes do plugin ser destruído. A drenagem **não** pode ocorrer no `Drop` do
  //! `NamClapShared` porque o consumer (`gc_rx`) vive no `NamClapMainThread` —
  //! estruturas separadas por contrato CLAP.
  //!
  //! A função `drain_gc_channels` é a única via de drenagem e deve ser chamada:
  //! 1. Periodicamente em `housekeeping()` (via `on_main_thread`).
  //! 2. **Uma vez final** em `NamClapMainThread::drain_gc_final()` no teardown.
  //!
  //! Um leak controlado (itens em trânsito no exato instante do destroy) é
  //! aceitável *apenas* se documentado; a drenagem dupla em `drain_gc_final`
  //! fecha a janela de race para o caso comum.
  ```

- **Critério de aceite:** Comentário presente. `cargo doc` (ou `cargo check`) passa.

### T6.3 — Teste de lifecycle destrutivo do GC [DONE]

- **Arquivo:** [`tests/models/gc_lifecycle_test.rs`](tests/models/gc_lifecycle_test.rs) (novo) ou extensão de `processor_gc_stress_test.rs`

- **Ação:** Adicionar teste `#[ignore]` (para `tests-long`) que verifica drenagem no teardown:

  ```rust
  /// Verifica que destruir o plugin com itens em trânsito no GC não causa
  /// double-free nem leak — confirma R11 resolvido.
  ///
  /// Marcado `#[ignore]`: requer heap-audit (valgrind/ASAN) e tempo suficiente
  /// para saturar o GC-overflow. Rodar via `utils/tests-long.sh`.
  #[test]
  #[ignore]
  fn test_gc_drain_on_destroy_no_leak() {
      // 1. Criar plugin simulado com GC ativo.
      // 2. Enfileirar N modelos no GC sem drenar.
      // 3. Chamar drain_gc_final().
      // 4. Verificar via leak-check que nenhum item permanece.
      // (Implementação completa: harness existente em processor_gc_stress_test.rs)
      todo!("implementar com harness clap-test após T6.1 estabilizar")
  }
  ```

  Adicionar o teste quickcheck correspondente (sem `#[ignore]`) que verifica que `drain_gc_channels` com buffer vazio não panics e retorna 0:

  ```rust
  #[test]
  fn test_drain_gc_empty_is_noop() {
      use crate::common::spsc::{drain_gc_channels, GcOverflowBuffer, RtStatusFlags};
      let (_, mut consumer) = rtrb::RingBuffer::new(16);
      let overflow = std::sync::Arc::new(GcOverflowBuffer::new());
      let rt_status = std::sync::Arc::new(RtStatusFlags::new());
      let drained = drain_gc_channels(&mut consumer, &overflow, &rt_status);
      assert_eq!(drained, 0);
  }
  ```

- **Critério de aceite:** Testes rápidos passam. Teste `#[ignore]` compila sem erro.

### T6.4 — Checkpoint S6: Verificação intermediária [DONE]

- **Ação:**
  1. `cargo clippy --all-targets` — limpo.
  2. `cargo test` — verde (incluindo `test_drain_gc_empty_is_noop`).
  3. `grep -n "drain_gc_final\|drain_gc_channels" src/clap/plugin/main_thread/mod.rs` → pelo menos 1 hit confirmando chamada no teardown.
- **Critério de aceite:** Todos verdes. **Não prosseguir para S7 sem este checkpoint.**

---

## Sprint S7 — R3: Documentação e Acquire no double-SIGINT [DONE]

> **Ref:** [TODO-findings.md §R3](TODO-findings.md#r3--double-sigint-vaza-o-lock-de-pm-qos-devcpu_dma_latency-até-o-reboot--alta) (L166-203)
>
> **Objetivo:** Corrigir o ordering do `SHUTDOWN.load` para `Acquire` (garante que o primeiro SIGINT seja sempre observado), adicionar comentário explicativo no handler de `_exit`, e publicar nota operacional em docs. Nenhuma mudança de comportamento — é documentação + correção formal de ordering.
>
> **Risco:** Baixo. Única linha de código tocada no standalone (`main.rs`). Docs apenas.

### T7.1 — Corrigir ordering do `SHUTDOWN.load` no handler SIGINT [DONE]

- **Arquivo:** [`src/main.rs`](src/main.rs)

- **Ação:**

  1. Na função `sigint_handler` (L80), alterar:

     ```rust
     // ANTES:
     if spsc::SHUTDOWN.load(Ordering::Acquire) {
     ```

     > Verificar se já usa `Acquire` (o finding cita potencial de `Relaxed`). Se já for `Acquire`, esta sub-tarefa é apenas validação.

  2. Adicionar comentário explicativo no bloco do segundo SIGINT:

     ```rust
     extern "C" fn sigint_handler(_sig: libc::c_int) {
         if spsc::SHUTDOWN.load(Ordering::Acquire) {
             // Segundo Ctrl-C: o graceful shutdown não respondeu a tempo.
             // `_exit(1)` encerra o processo sem rodar destrutores.
             // O kernel recolhe TODOS os recursos abertos (fds, mapeamentos, PM QoS,
             // THP advice) — nada persiste após o exit do processo (R3, verificado
             // via /proc/<pid>/fd e pm_qos_constraint após o exit).
             // Preferir `_exit` a `abort` para evitar core dump desnecessário;
             // usar `abort()` apenas se core dump for desejado para diagnóstico.
             unsafe { libc::_exit(1) };
         }
         spsc::SHUTDOWN.store(true, Ordering::Release);
     }
     ```

  3. Verificar que `SHUTDOWN.store` já usa `Ordering::Release` (L83) — se não, corrigir.

- **Critério de aceite:** `SHUTDOWN.load` usa `Acquire`; `SHUTDOWN.store` usa `Release`. `cargo check` passa.

### T7.2 — Nota operacional em `docs/` [DONE]

- **Arquivo:** [`docs/architecture.md`](docs/architecture.md) (ou criar `docs/operations.md` se não existir)

- **Ação:** Adicionar seção:

  ```markdown
  ## Comportamento de shutdown (SIGINT / SIGTERM)

  ### Standalone (src/main.rs)

  O handler de SIGINT usa dois níveis:

  1. **Primeiro Ctrl-C**: seta `SHUTDOWN` (Release) → o loop principal (`run.rs`)
     detecta via Acquire, drena o GC, fecha streams PipeWire e retorna normalmente.
     PM QoS (`/dev/cpu_dma_latency`) e THP advice são liberados pelos destrutores.

  2. **Segundo Ctrl-C** (double-SIGINT, shutdown não respondeu): chama `_exit(1)`.
     Nenhum destrutor roda — mas o **kernel fecha todos os fds e mapeamentos** no
     exit do processo. Recursos persistentes (PM QoS, THP) **não** ficam presos:
     o kernel os libera automaticamente (verificado; não há file lock pós-processo).
     O único efeito é skip da drenagem do GC (itens em trânsito são liberados pelo
     kernel junto com o heap do processo).

  ### Plugin CLAP (src/clap/)

  O lifecycle é controlado pelo host via `clap_plugin.destroy()`. Ver §EP-R2 nos
  [findings de auditoria](../TODO-findings.md) para detalhes de R2/R13/R11.
  ```

- **Critério de aceite:** Seção existe. `cargo check` inalterado.

### T7.3 — Checkpoint S7: Verificação intermediária [DONE]

- **Ação:**
  1. `cargo clippy --all-targets` — limpo.
  2. `cargo check` — verde.
  3. `grep -n "SHUTDOWN.load" src/main.rs` confirma `Ordering::Acquire`.
- **Critério de aceite:** Todos verdes.

---

## VF — Verificação Final Integrada EP-R2 [DONE]

> **Gate de aceite do épico inteiro.** Rodar apenas após os 4 sprints com checkpoints verdes.

### VF2.1 — Lints completos

- `utils/lints.sh` — fmt + SPDX + check + clippy, **zero erros/warnings**.

### VF2.2 — Suite rápida completa

- `utils/tests-quick.sh` — **verde total**, incluindo os novos testes de S4/S5/S6/S7.

### VF2.3 — Contrato de qualidade inalterado

- `utils/quality-dashboard.sh --check docs/quality-contract.txt` — **zero diff**.
- Se qualquer número mudar: **PARAR**, reverter e investigar.

### VF2.4 — Confirmações de eliminação

- `grep -rn "shared_addr" src/clap/gui/` → **zero hits** (R2 eliminado).
- `grep -rn "as \*const NamClapShared" src/clap/gui/` → **zero hits**.
- `grep -n "SHUTDOWN.load" src/main.rs | grep -v Acquire` → **zero hits** (R3 ordering).
- `grep -n "handle.join()" src/clap/extensions/gui.rs` → **zero hits** (R13 eliminado).

### VF2.5 — Validação com clap-validator

- Rodar `clap-validator validate target/*/libnam_rs.so` (se disponível no ambiente).
- Verificar que `state-invalid`, `gui-*`, e `lifecycle-*` passam **sem regressão**.

### VF2.6 — Solicitar tests-long ao operador

- Solicitar que o operador rode `utils/tests-long.sh` na próxima janela noturna, especialmente a fase `gc_stress_1000_swaps` e os novos testes `#[ignore]` de lifecycle.
- **Não bloquear o épico** — os testes longos são validação complementar.

---

## Notas de risco e mitigação (EP-R2)

| Risco                                                    | Mitigação                                                                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `Arc<DialogSharedState>` cria ciclo de vida inesperado   | Os únicos campos no Arc são atomics e `Mutex<Option<PathBuf>>` — sem referências circulares; drop é determinístico       |
| `on_main_thread` pode não ser chamado após `destroy()`   | `drain_gc_final` também chamado em `deactivate()` — duas oportunidades de drenagem                                       |
| Watchdog de 2 s pode abandonar thread legítima sob carga | 2 s >> tempo típico de fechamento de janela (<100 ms); log de advertência documenta o abandono                           |
| `_exit(1)` no double-SIGINT pula drenagem do GC          | Leak finito e limitado ao heap do processo; kernel recolhe tudo no exit                                                  |
| Mudança de ordering em `alive_fence` (Release/Acquire)   | Comportamento idêntico em x86-TSO; pode tornar o fence mais conservador em arquiteturas fracas — sem efeito prático      |
| Arquitetura do site de chamada de `spawn_file_dialog`    | A integração com `NamClapMainThread` requer acesso mutável; verificar se o contexto CLAP permite — adaptar se necessário |

---

## Épico EP-R3 — Formalização da concorrência

> **Origem:** [TODO-findings.md §EP-R3](TODO-findings.md#ep-r3--formalização-da-concorrência-r8-completo--p1) (Auditoria de Resiliência & Robustez, 2026-07-14)
>
> **Escopo:** Correção formal de 8 pontos de atomic ordering (R8 completo) + Testes de model-checking de concorrência com `loom` (P1).
>
> **Pré-requisito:** EP-R2 concluído e `quality-dashboard.sh --check` verde.
>
> **Invariante absoluto:** zero alteração de comportamento sonoro. Critério de aceite global: `quality-dashboard.sh --check` sem diff de um único número no contrato. Risco: baixo (concorrência formalizada e testada via model-checking).

---

## EP-R3 — Sumário dos Sprints

| Sprint  | Finding                                         | Risco | Arquivos tocados | Estimativa |
| ------- | ----------------------------------------------- | ----- | ---------------- | ---------- |
| **S8**  | P1 — Model-checking dos protocolos com `loom`   | Baixo | 2                | ~60 min    |
| **S9**  | R8 — Formalização e pareamento dos atômicos     | Baixo | 8                | ~60 min    |
| **S10** | Integração — Novo job `loom` no `tests-long.sh` | Baixo | 1                | ~15 min    |
| **VF**  | Verificação final integrada EP-R3               | —     | 0                | ~15 min    |

---

## Sprint S8 — Loom: Model-checking dos protocolos com `loom`

> **Ref:** [TODO-findings.md §P1](TODO-findings.md#p1--model-checking-do-protocolo-spscgc-com-loom-dev-dependency-roda-em-stable) (L563-578)
>
> **Objetivo:** Sincronizar o model-checking determinístico e exaustivo com `loom` para os 3 protocolos críticos de concorrência do projeto. Sem tocar a produção, modelar os handshakes dentro do arquivo de teste.
>
> **Risco:** Baixo. Modificações apenas na suíte de testes e dependências de desenvolvimento.

### T8.1 — Adicionar `loom` às `dev-dependencies` [x]

- **Arquivo:** [`Cargo.toml`](Cargo.toml)
- **Ação:** Adicionar `loom = "0.7"` na seção `[dev-dependencies]`.
- **Critério de aceite:** `cargo check` passa normalmente.

### T8.2 — Modelar protocolo de Handshake [x]

- **Arquivo (novo):** `tests/loom_tests.rs`
- **Ação:** Implementar teste `#[cfg(loom)]` com `loom::thread::spawn` que simula duas threads trocando dados através de um booleano de controle (`SHUTDOWN` ou `active_rate`). Verificar que o uso de `Relaxed` causa falha no loom (data race) e que `Release`/`Acquire` resolve.
- **Critério de aceite:** O teste falha sob `Relaxed` e passa sob `Release`/`Acquire`.

### T8.3 — Modelar fila de GC Overflow [x]

- **Arquivo:** `tests/loom_tests.rs`
- **Ação:** Modelar o buffer SPSC de overflow (`GcOverflowBuffer`). Simular uma thread RT empurrando dados e uma thread de controle drenando. Testar reordenamento do `write_idx` vs `swap` do slot.
- **Critério de aceite:** O teste de concorrência passa sob loom.

### T8.4 — Modelar Double-Buffering `DspBridge` [x]

- **Arquivo:** `tests/loom_tests.rs`
- **Ação:** Modelar a Sincronização de buffer duplo do `DspBridge` utilizando `generation` e `active_read_idx`. Verificar que o leitor sempre obtém dados válidos e ordenados em relação às escritas.
- **Critério de aceite:** O teste de concorrência passa sob loom.

---

## Sprint S9 — R8: Sincronização e pareamento dos atômicos

> **Ref:** [TODO-findings.md §R8](TODO-findings.md#r8--família-de-orderings-atômicos-formalmente-incorretos-funcionam-em-x86-tso-incorretos-no-modelo-de-memória--média) (L342-364)
>
> **Objetivo:** Corrigir os 8 pontos da tabela R8 de atomic orderings incorretos e adicionar comentários estruturados de pareamento.
>
> **Risco:** Baixo. Sem custo de release em x86 (Release/Acquire compila para mov).

### T9.1 — Corrigir R8-b: Sample rate sync [x]

- **Arquivos:**
  - [`src/standalone/pw_host/capture/listeners.rs`](src/standalone/pw_host/capture/listeners.rs)
  - [`src/standalone/pw_host/rt_callback/rate_sync.rs`](src/standalone/pw_host/rt_callback/rate_sync.rs)
- **Ação:**
  - No `listeners.rs`, mudar o store do `rate_for_param` para `Ordering::Release`.
  - No `rate_sync.rs`, mudar o swap no `rate_for_process` para `Ordering::Acquire`.
- **Critério de aceite:** `cargo check` passa.

### T9.2 — Corrigir R8-c: Panic hook SHUTDOWN load [x]

- **Arquivo:** [`src/common/panic_hook.rs`](src/common/panic_hook.rs)
- **Ação:** Modificar a leitura do `SHUTDOWN` na linha 30 para usar `Ordering::Acquire`.
- **Critério de aceite:** `cargo check` passa.

### T9.3 — Corrigir R8-d: Telemetry Reset [x]

- **Arquivo:** [`src/dsp/telemetry.rs`](src/dsp/telemetry.rs)
- **Ação:** No reset, substituir `bin.store(0, Ordering::Relaxed)` por `bin.swap(0, Ordering::Relaxed)` e adicionar comentário explicando o comportamento concorrente "best-effort".
- **Critério de aceite:** `cargo check` passa.

### T9.4 — Corrigir R8-e: alive_fence ordering [x]

- **Arquivos:**
  - [`src/clap/plugin/shared.rs`](src/clap/plugin/shared.rs)
  - [`src/clap/gui/window/state.rs`](src/clap/gui/window/state.rs)
- **Ação:**
  - Em `shared.rs:262` (no Drop), mudar para `alive_fence.store(false, Ordering::Release)`.
  - Em `state.rs:190` (no safe_shared), mudar para `alive_fence.load(Ordering::Acquire)`.
- **Critério de aceite:** `cargo check` passa.

### T9.5 — Corrigir R8-f: write_idx fetch_add [x]

- **Arquivo:** [`src/common/spsc/gc.rs`](src/common/spsc/gc.rs)
- **Ação:** Mudar `write_idx.fetch_add(1, Ordering::Relaxed)` para `Ordering::AcqRel` (ou documentar por que `Relaxed` é seguro dada a sweep total).
- **Critério de aceite:** `cargo check` passa.

### T9.6 — Corrigir R8-g: clear_flag_relaxed [x]

- **Arquivos:**
  - [`src/common/spsc/status.rs`](src/common/spsc/status.rs)
  - [`src/standalone/pw_host/run.rs`](src/standalone/pw_host/run.rs)
- **Ação:**
  - Em `status.rs`, introduzir `pub fn clear_flag_relaxed(&self, flag: u64)` que executa `fetch_and(!flag, Ordering::Relaxed)`.
  - Em `run.rs`, substituir as 5 chamadas de `clear_flag_release` por `clear_flag_relaxed` (pois o leitor RT não faz acquire do clear).
- **Critério de aceite:** `cargo check` passa.

### T9.7 — Corrigir R8-h: RT_STATUS_GC_OVERFLOW condicionamento [x]

- **Arquivo:** [`src/common/spsc/gc.rs`](src/common/spsc/gc.rs)
- **Ação:** Condicionar o `set_flag(RT_STATUS_GC_OVERFLOW)` ao retorno `true` (sobrescrita real) do `gc_overflow.push(i)`.
- **Critério de aceite:** `cargo check` passa.

### T9.8 — Comentários de pareamento [x]

- **Ação:** Adicionar comentários `// pairs with Release store em <file:line>` ou similar em cada par Acquire/Release atômico da base de código tocada.
- **Critério de aceite:** Código revisado e documentado.

---

## Sprint S10 — Sincronização: Habilitar testes loom no tests-long.sh

> **Ref:** [TODO-findings.md §EP-R3](TODO-findings.md#ep-r3--formalização-da-concorrência-r8-completo--p1) (L685-689)
>
> **Objetivo:** Adicionar estágio ao pipeline longo de testes.
>
> **Risco:** Baixo.

### T10.1 — Novo estágio de loom no tests-long.sh [x]

- **Arquivo:** [`utils/tests-long.sh`](utils/tests-long.sh)
- **Ação:** Adicionar fase que executa `RUSTFLAGS="--cfg loom" cargo test --test loom_tests --release`.
- **Critério de aceite:** O script executa o teste sob flag loom de forma resiliente.

---

## VF — Verificação Final Integrada EP-R3

> **Gate de aceite do épico inteiro.**

### VF3.1 — Lints completos

- `utils/lints.sh` — fmt + SPDX + check + clippy, **zero erros/warnings**.

### VF3.2 — Suite rápida completa

- `utils/tests-quick.sh` — **verde total**.

### VF3.3 — Contrato de qualidade bit-exact

- `utils/quality-dashboard.sh --check docs/quality-contract.txt` — **zero diff**.

### VF3.4 — Solicitar tests-long ao operador

- Solicitar que o operador execute `utils/tests-long.sh` incluindo o novo estágio loom.

---

## Notas de risco e mitigação (EP-R3)

| Risco                                                | Mitigação                                                                                                                 |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `loom` trava ou atinge limite de iterações em CI     | Configurar limites adequados no loom (`LOOM_MAX_PREEMPTIONS` se necessário); manter os modelos de teste simples e enxutos |
| `fetch_add(AcqRel)` no GC impacta desempenho do RT   | Operação não-bloqueante; caminho de overflow é alternativo (raramente executado no hot path)                              |
| Divergência entre modelo `loom` e código de produção | Manter os modelos atômicos nos testes 100% fiéis à topologia implementada em produção                                     |

---

## Épico EP-R4 — Blindagem da malha de QA

> **Origem:** [TODO-findings.md §EP-R4](TODO-findings.md#ep-r4--blindagem-da-malha-de-qa-r6--r7--r4--r5) (Auditoria de Resiliência & Robustez, 2026-07-14)
>
> **Escopo:** R6 (Contrato de qualidade, **ALTA**) + R7 (Gate de testes no tests-long, **ALTA**) + R4 (Panic hook sem alocação/deadlocks, **ALTA**) + R5 (Zero logs em RT, **ALTA**).
>
> **Pré-requisito:** EP-R3 concluído e `quality-dashboard.sh --check` verde.
>
> **Invariante absoluto:** zero alteração de comportamento sonoro. Critério de aceite global: `quality-dashboard.sh --check` sem diff de um único número no contrato original (após regravação sem truncamento). Risco: baixo.

---

## EP-R4 — Sumário dos Sprints

| Sprint  | Finding                                        | Risco | Arquivos tocados | Estimativa |
| ------- | ---------------------------------------------- | ----- | ---------------- | ---------- |
| **S11** | R6 — Contrato de qualidade: matching exato     | Baixo | 2                | ~45 min    |
| **S12** | R7 — tests-long: Gate "≥1 passed" obrigatório  | Baixo | 1                | ~30 min    |
| **S13** | R4 — Panic hook zero-alloc e deadlock-free     | Médio | 2                | ~60 min    |
| **S14** | R5 — Zero logs em RT & Meta-testes estruturais | Médio | 8                | ~90 min    |
| **VF**  | Verificação final integrada EP-R4              | —     | 0                | ~15 min    |

---

## Sprint S11 — R6: Contrato de qualidade com matching exato e composto

> **Ref:** [TODO-findings.md §R6](TODO-findings.md#r6--contrato-de-qualidade-matching-por-prefixo-colide-quick-a2-full--quick-a2-full-v2--verificação-falso-verde--alta) (L269-308)
>
> **Objetivo:** Eliminar colisões por prefixo na verificação do contrato de qualidade de áudio (`quality-dashboard.sh`), removendo o truncamento na persistência de dados e aplicando casamento exato.
>
> **Risco:** Baixo. Modificações limitadas ao script de validação de qualidade e meta-testes.

### T11.1 — Evitar truncamento de colunas ao salvar contrato [x]

- **Arquivo:** [`utils/quality-dashboard.sh`](utils/quality-dashboard.sh)
- **Ação:**
  1. Adicionar `export IS_SAVING=1` dentro da função `render_dashboard_plain`.
  2. Em `render_fidelity_details`, condicionar o cálculo da variável `display_key`: se `IS_SAVING` for igual a `1`, usar a chave completa do modelo (`key`), sem truncar em 38 colunas. Caso contrário, manter o truncamento em `${key:0:38}` para visualização amigável no console.
- **Critério de aceite:** `quality-dashboard.sh --save` grava as linhas da tabela de fidelidade com os nomes dos modelos completos na primeira coluna.

### T11.2 — Modificar matching para chave exata e composta [x]

- **Arquivo:** [`utils/quality-dashboard.sh`](utils/quality-dashboard.sh)
- **Ação:**
  1. No loop de verificação de contrato em `load_contract_baseline`, ler o identificador completo.
  2. Em `quality_check`, fazer a verificação de igualdade exata de strings (`[[ "$dash_key" == "$contract_label" ]]`) em substituição à lógica de prefixo anterior (`[[ "$dash_label" == "$contract_label"* ]]`).
- **Critério de aceite:** Matching de baseline do contrato opera de forma exata e não confunde mais chaves como `Quick A2-Full` e `Quick A2-Full v2`.

### T11.3 — Adicionar meta-teste de unicidade do contrato [x]

- **Arquivo:** [`tests/models/meta_coherence.rs`](tests/models/meta_coherence.rs)
- **Ação:**
  1. Implementar o teste `test_quality_contract_uniqueness`.
  2. O teste deve ler o arquivo `docs/quality-contract.txt`, extrair todos os labels da tabela de fidelidade e garantir que nenhum deles seja prefixo de outro (evitando futuras ambiguidades).
- **Critério de aceite:** `cargo test --test meta_coherence` passa.

---

## Sprint S12 — R7: Gate "≥1 passed" obrigatório no `tests-long.sh`

> **Ref:** [TODO-findings.md §R7](TODO-findings.md#r7--fase-pipewire-do-tests-longsh-aviso-de-falso-verde-é-sintoma-de-detecção-incapaz-de-distinguir-rápido-de-vazio--alta) (L310-340)
>
> **Objetivo:** Prevenir falsos-verdes perpétuos em `tests-long.sh` decorrentes de renomeações ou seleções vazias de testes/benchmarks, forçando cada fase a executar no mínimo 1 teste.
>
> **Risco:** Baixo. Alteração puramente do harness de testes.

### T12.1 — Implementar validador `assert_ran_tests` em `tests-long.sh` [x]

- **Arquivo:** [`utils/tests-long.sh`](utils/tests-long.sh)
- **Ação:**
  1. Criar a função utilitária `assert_ran_tests` que analisa o log da fase atual em `target/logs/$log_file`.
  2. Fazer o parse dos sumários de teste (`test result: ok. [0-9]+ passed`) ou benchmark (`[0-9]+ measured`) e somá-los.
  3. Falhar se a soma for zero, com mensagem descritiva de "seleção de testes vazia".
- **Critério de aceite:** A função é capaz de determinar corretamente se testes reais rodaram e passaram.

### T12.2 — Integrar o gate de testes em todas as fases do `tests-long.sh` [x]

- **Arquivo:** [`utils/tests-long.sh`](utils/tests-long.sh)
- **Ação:**
  1. Em `run_phase`, quando o comando da fase retornar sucesso (status 0), chamar `assert_ran_tests`.
  2. Se retornar erro de testes não executados, mudar o status da fase para falha e retornar 1.
  3. Remover o aviso `< 1s` genérico anterior.
- **Critério de aceite:** Fases vazias (como a fase Pipewire se não houvesse testes) falham o pipeline em vez de passar como falso-verde.

---

## Sprint S13 — R4: Panic hook robusto, zero-alloc e deadlock-free

> **Ref:** [TODO-findings.md §R4](TODO-findings.md#r4--panic-hook-aloca-heap-e-adquire-rwlock-no-caminho-de-crash--pode-deadlockar-exatamente-quando-mais-se-precisa-dele--alta) (L206-235)
>
> **Objetivo:** Eliminar alocações de heap e RwLock bloqueantes no caminho de pânico da thread RT para evitar deadlocks e garantir a captura do crash report em qualquer circunstância.
>
> **Risco:** Médio. Toca a captura de diagnóstico no crash de produção.

### T13.1 — Pre-capturar `SystemSnapshot` estaticamente [x]

- **Arquivo:** [`src/common/panic_hook.rs`](src/common/panic_hook.rs)
- **Ação:**
  1. Declarar `static SYSTEM_SNAPSHOT: OnceLock<SystemSnapshot>`.
  2. Inicializá-la no `install_panic_hook` chamando `SystemSnapshot::capture()` (caminho off-RT de inicialização do plugin).
- **Critério de aceite:** Snapshot de sistema capturado em startup, sem tocar no alocador global durante o pânico.

### T13.2 — Criar formatador do crash report zero-alloc com `LimitWriter` [x]

- **Arquivo:** [`src/common/panic_hook.rs`](src/common/panic_hook.rs)
- **Ação:**
  1. Definir a struct `LimitWriter<'a>` que encapsula `&mut [u8]` e um cursor.
  2. Implementar `std::fmt::Write` para `LimitWriter` com truncamento silencioso se estourar o limite.
  3. Implementar a função `format_panic_report_to_buf` que recebe os metadados do panic e preenche o buffer sem usar `format!` ou outras macros geradoras de String.
  4. Substituir a leitura bloqueante de RwLock por `try_read()` com fallback `"<unavailable>"`.
- **Critério de aceite:** Função de formatação do report compila e opera puramente em memória da pilha.

### T13.3 — Atualizar o manipulador do Panic Hook [x]

- **Arquivo:** [`src/common/panic_hook.rs`](src/common/panic_hook.rs)
- **Ação:**
  1. No hook de pânico, alocar uma array `[u8; 4096]` na stack.
  2. Invocar `format_panic_report_to_buf` para formatar os detalhes e o snapshot do sistema no buffer.
  3. Gravar os bytes resultantes diretamente para o arquivo `.cache` usando `std::fs::File` (e `write_all`).
- **Critério de aceite:** O arquivo de crash report é escrito corretamente com dados válidos e sem alocar memória no heap.

### T13.4 — Adicionar teste de auditoria de heap para o panic hook [x]

- **Arquivo:** [`tests/models/diagnostic_bundle.rs`](tests/models/diagnostic_bundle.rs)
- **Ação:**
  1. Adicionar o teste `test_panic_hook_zero_alloc` no módulo `heap_audit_tests` (condicionado por `#[cfg(feature = "heap-audit")]`).
  2. Utilizar `TrackingGuard` e `get_alloc_count` para assegurar que a chamada a `format_panic_report_to_buf` executa com **zero alocações no heap**.
- **Critério de aceite:** O teste de heap-audit passa.

---

## Sprint S14 — R5: Eliminação de logs em RT e novos meta-testes estruturais

> **Ref:** [TODO-findings.md §R5](TODO-findings.md#r5--logerror-alcançável-no-thread-rt-via-containermodelset_slimmable_size--alta) (L237-267)
>
> **Objetivo:** Remover a chamada a `log::error!` da thread em tempo real no `ContainerModel` (que causava I/O de disco/formatação de string bloqueante), substituindo-a por sinalização atômica traduzida assincronamente na thread de housekeeping.
>
> **Risco:** Médio. Altera a assinatura do pipeline de DSP e os tratamentos de erro.

### T14.1 — Definir novo bit `RT_STATUS_SLIMMABLE_RESET_FAILED` [x]

- **Arquivo:** [`src/common/spsc/status.rs`](src/common/spsc/status.rs)
- **Ação:** Adicionar `pub const RT_STATUS_SLIMMABLE_RESET_FAILED: u64 = 1 << 21;`.
- **Critério de aceite:** Nova flag compilável no módulo de status.

### T14.2 — Alterar assinatura e comportamento do quality scaling do DSP[x]

- **Arquivos:**
  - [`src/models/slimmable.rs`](src/models/slimmable.rs)
  - [`src/models/container.rs`](src/models/container.rs)
  - [`src/models/static_model.rs`](src/models/static_model.rs)
  - [`src/dsp/pipeline/stages/inference.rs`](src/dsp/pipeline/stages/inference.rs)
- **Ação:**
  1. Mudar a assinatura de `SlimmableModel::set_slimmable_size` para incluir `rt_status: Option<&RtStatusFlags>`.
  2. Propagar essa mudança na impl de `StaticModel` e em `inference.rs` (adicionando `rt_status` em `configure_adaptive_model` extraído do `DspPipelineContext`).
  3. No `ContainerModel::set_slimmable_size`, remover o `log::error!` e substituí-lo por `rt_status.map(|s| s.set_flag(RT_STATUS_SLIMMABLE_RESET_FAILED));` caso o `reset` do submodelo falhe.
- **Critério de aceite:** O pipeline compila normalmente com a nova assinatura e sem logs ativos na inferência do container.

### T14.3 — Emitir o log de erro assincronamente na thread principal [x]

- **Arquivos:**
  - [`src/clap/plugin/main_thread/housekeeping.rs`](src/clap/plugin/main_thread/housekeeping.rs)
  - [`src/standalone/rt_setup/telemetry.rs`](src/standalone/rt_setup/telemetry.rs)
- **Ação:**
  1. Em `housekeeping.rs` (CLAP) e `telemetry.rs` (standalone), checar periodicamente se o bit `RT_STATUS_SLIMMABLE_RESET_FAILED` está setado e limpá-lo.
  2. Em caso positivo, disparar o log de erro correspondente com o log do host CLAP ou `log::error!` standalone.
- **Critério de aceite:** Falhas de reset no container são notificadas ao usuário a partir de threads seguras.

### T14.4 — Adicionar meta-teste estrutural de logging RT-safe [x]

- **Arquivo:** [`tests/models/meta_coherence.rs`](tests/models/meta_coherence.rs)
- **Ação:**
  1. Implementar o teste `test_rt_logging_safety` que inspeciona os códigos-fonte da hot-path de DSP (`src/dsp/`, `src/models/`, `src/math/`) e falha caso encontre substrings como `log::`, `println!`, `eprintln!`, `format!` fora de funções construtoras, testes ou blocos explicitamente marcados com `#[cold]`.
- **Critério de aceite:** O teste de sanidade estática compila e passa.

---

## VF — Verificação Final Integrada EP-R4

### VF4.1 — Lints completos

- Executar `utils/lints.sh` e garantir zero erros ou avisos.

### VF4.2 — Suite rápida completa

- Executar `utils/tests-quick.sh` e comprovar sucesso total dos testes rápidos e novos meta-testes.

### VF4.3 — Regravar o contrato de qualidade

- Executar `./utils/quality-dashboard.sh --save docs/quality-contract.txt` para regravar o contrato com os novos identificadores sem truncamento.
- Executar `./utils/quality-dashboard.sh --check docs/quality-contract.txt` e verificar conformidade exata e sem diffs.

---

## Notas de risco e mitigação (EP-R4)

| Risco                                                                        | Mitigação                                                                                                                                              |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Buffer de pânico de 4096 bytes é insuficiente                                | O `LimitWriter` trunca de forma silenciosa e limpa, mantendo o início do report intacto, sem risco de estourar a pilha ou falhar por erro de alocação. |
| Remoção do matching por prefixo quebrar compatibilidade                      | As chaves salvas são geradas exatamente a partir da mesma fonte JSONL de dados, garantindo equivalência exata de strings.                              |
| Alteração no trait SlimmableModel causar regressions em plugins de terceiros | O trait é marcado como interno do módulo de modelos (`pub(crate)` ou não exportado para FFI), sendo seguro alterar internamente.                       |

---

## Épico EP-R5 — Higiene e superfície

> **Origem:** [TODO-findings.md §EP-R5](TODO-findings.md#ep-r5--higiene-e-superfície-r12--r14--r15--r16--p3) (Auditoria de Resiliência & Robustez, 2026-07-14)
>
> **Escopo:** R12 (Higiene de unsafe, **MÉDIA**) + R14 (Código morto/duplicado, **BAIXA**) + R15 (Separação de testing feature, **BAIXA**) + R16 (Higiene de saída de ferramentas, **BAIXA**) + P3 (assert_unchecked / as_chunks, **Inovação/Proposta**).
>
> **Pré-requisito:** EP-R4 concluído e `quality-dashboard.sh --check` verde.
>
> **Invariante absoluto:** zero alteração de comportamento sonoro. Critério de aceite global: `quality-dashboard.sh --check` sem diff de um único número no contrato. Risco: mínimo. Ideal para ser executado com as skills `refatora-rust` e `refatora-doc`.

---

## EP-R5 — Sumário dos Sprints

| Sprint  | Finding                                             | Risco | Arquivos tocados | Estimativa |
| ------- | --------------------------------------------------- | ----- | ---------------- | ---------- |
| **S15** | R12 + P3 — Higiene de `unsafe` e `assert_unchecked` | Baixo | 9                | ~90 min    |
| **S16** | R14 — Remoção de código morto e duplicações         | Baixo | 8                | ~75 min    |
| **S17** | R15 — Feature `testing` fora de default             | Baixo | 3                | ~45 min    |
| **S18** | R16 — Higiene de saída de logs e referências        | Baixo | 5                | ~45 min    |
| **VF**  | Verificação final integrada EP-R5                   | —     | 0                | ~15 min    |

---

## Sprint S15 — R12 & P3: Higiene de `unsafe` e `assert_unchecked`

> **Ref:** [TODO-findings.md §R12](TODO-findings.md#r12--higiene-de-unsafe-comentários-safety-genéricos-get_unchecked-substituível-e-invariantes-não-escritas--média) (L446-466) e [TODO-findings.md §P3](TODO-findings.md#p3--reduzir-unsafe-mantendo-codegen-corehintassert_unchecked-stable-181-e-sliceas_chunks-stable-188) (L592-607)
>
> **Objetivo:** Reduzir blocos `unsafe` e formalizar invariantes SAFETY específicas no codebase, adotando `core::hint::assert_unchecked` para manter o codegen do LLVM sem bounds checks ao usar indexação segura.
>
> **Risco:** Mínimo. Alteração focada em documentação e otimizações locais.

### T15.1 — Reescrever comentários SAFETY em `src/dsp/mirror_buf.rs` [x]

- **Arquivo:** [`src/dsp/mirror_buf.rs`](src/dsp/mirror_buf.rs) (L153, 161, 171, 192, 194)
- **Ação:** Substituir comentários SAFETY genéricos por explicações das invariantes reais de manipulação de memória virtual, citando a validade do ponteiro para `size_elements * 2` e os buffers espelhados.
- **Critério de aceite:** `cargo check` passa; comentários revisados.

### T15.2 — Alinhar SAFETY de `huge_alloc.rs` com padrão de `aligned.rs` [x]

- **Arquivo:** [`src/math/common/huge_alloc.rs`](src/math/common/huge_alloc.rs) (L369, 380, 389)
- **Ação:** Substituir o comentário genérico "upheld by caller invariants" por documentação precisa citando o porquê de o ponteiro ser válido e não-nulo, e a relação com o tamanho da alocação de huge pages.
- **Critério de aceite:** `cargo check` passa.

### T15.3 — Formalizar invariantes e SAFETY no delay line do stage [x]

> **Nota de implementação:** A invariante estática usou `UP_DELAY_LINE_LEN > (HB_DELAY - 1) + (HB_ODD_COUNT - 1)` (24 > 22) — o invariante verificável real para o upsampling — pois `UP_DELAY_LINE_LEN` (24) é estritamente menor que `HB_TAPS` (25). A forma original `>= HB_TAPS` seria compilação impossível. A invariante correta verifica que todos os acessos de leitura SIMD + scalar no upsampling cabem no mirror buffer.

- **Arquivo:** [`src/dsp/stage.rs`](src/dsp/stage.rs) (L130-152, 169-198)
- **Ação:**
  1. Introduzir invariante estática `const { assert!(UP_DELAY_LINE_LEN >= HB_TAPS) }`.
  2. Adicionar documentação SAFETY explicando por que o loop garante acessos válidos e em limites de bounds.
- **Critério de aceite:** Código compila com a nova asserção estática.

### T15.4 — Otimizar `get_unchecked` no FFT usando `assert_unchecked` [x]

> **Nota de implementação:** Além dos locais citados (bit-reversal L228/258 e scalar path L320-324), o `get_unchecked` do SIMD path em `stage_offsets` também foi convertido. Todos os 63 testes de FFT passam.

- **Arquivo:** [`src/math/dsp/fft.rs`](src/math/dsp/fft.rs) (L228, 258, 320-324)
- **Ação:**
  1. Substituir acessos `get_unchecked` por indexação segura precedida de `unsafe { core::hint::assert_unchecked(idx < len) }` nos loops de bit-reversal e butterfly scalar path.
  2. Verificar que o assembly gerado em `target/dsp_hotpath.asm` não regrediu e mantém a eliminação de bounds checks.
- **Critério de aceite:** `cargo check` passa e o quick suite passa.

### T15.5 — Documentar SAFETY de ponteiros em ConvNet [x]

- **Arquivo:** [`src/models/convnet/model.rs`](src/models/convnet/model.rs) (L105-140)
- **Ação:** Adicionar blocos de comentário SAFETY explicativos sobre a aritmética de ponteiros com `Vec::as_mut_ptr()`, detalhando que `&mut self` impede realocações simultâneas e que os índices estão contidos em `blocks.len()`.
- **Critério de aceite:** `cargo check` passa.

### T15.6 — Adicionar `debug_assert!` e SAFETY no `copy_nonoverlapping` de WaveNet [x]

> **Nota de implementação:** `debug_assert!` adicionado nas duas funções (`process_single_frame_with_mixin` e `process_single_frame`). A invariante `max_lookback_cols` (não encontrada no codebase atual) foi documentada como `frame_idx >= dilation * (K-1)` — a restrição causal do receptive field que previne underflow no cast `isize → usize`.

- **Arquivo:** [`src/models/wavenet/conv1d.rs`](src/models/wavenet/conv1d.rs) (L56-62)
- **Ação:** Adicionar `debug_assert!` validando que o offset `isize` convertido para `usize` não causa underflow ou transborda, documentando a segurança da operação em relação à invariante `max_lookback_cols`.
- **Critério de aceite:** `cargo check` passa.

### T15.7 — Documentar transmute de lifetime na GUI do CLAP [x]

- **Arquivo:** [`src/clap/gui/mod.rs`](src/clap/gui/mod.rs) (L33)
- **Ação:** Documentar de forma expressa a segurança do `transmute` de lifetime em tipo sem `repr(transparent)`, detalhando a dependência de layout ou encapsulando em wrapper limpo de forma equivalente.
- **Critério de aceite:** Compila sem alertas.

### T15.8 — Comentário SAFETY para transmute de `__m512` em GEMV [x]

- **Arquivo:** [`src/math/gemm/gemv_bf16.rs`](src/math/gemm/gemv_bf16.rs) (L62-63, 99-100)
- **Ação:** Adicionar anotação SAFETY de uma linha justificando a conversão de `__m512` para `__m512bh` como "no-op de 512 bits entre tipos com ABI idêntica".
- **Critério de aceite:** Comentários adicionados.

### T15.9 — Limpar cast duplo de handler de sinal em `src/main.rs` [x]

> **Nota de implementação:** O binding `libc::sigaction` do Linux não expõe `sa_handler`; apenas `sa_sigaction`. O cast duplo `as *const () as sighandler_t` foi substituído por `std::mem::transmute` com comentário SAFETY explicando a compatibilidade ABI (SA_RESTART sem SA_SIGINFO aciona o path de handler 1-arg do kernel, que é compatível com nosso `sigint_handler`).

- **Arquivo:** [`src/main.rs`](src/main.rs) (L85-89)
- **Ação:** Substituir o cast duplo via `*const ()` para `sighandler_t` atribuindo o handler `sigint_handler` diretamente ao campo correto (`sa_handler` ou equivalente via struct) do `sigaction`.
- **Critério de aceite:** `cargo check --features standalone` passa sem warnings.

---

## Sprint S16 — R14: Remoção de código morto e duplicações

> **Ref:** [TODO-findings.md §R14](TODO-findings.md#r14--código-morto-e-duplicações--baixa-limpeza-mecânica-750-linhas-recuperáveis) (L490-507)
>
> **Objetivo:** Higienizar o repositório removendo declarações mortas, consolidando funções duplicadas e eliminando atalhos não utilizados que incham a base de código.
>
> **Risco:** Baixo. Modificações mecânicas e limpezas simples.

### T16.1 — Condicionar `FftPlannerRadix4` para testes/benches [x]

- **Arquivo:** [`src/math/dsp/fft_radix4.rs`](src/math/dsp/fft_radix4.rs) (L59)
- **Ação:** Adicionar `#[cfg(any(test, feature = "long_bench"))]` sobre a struct pública `FftPlannerRadix4` para evitar sua compilação em builds normais de produção, pois ela não possui consumidores em produção.
- **Critério de aceite:** `cargo check` passa.

### T16.2 — Consolidar a função `median` em local comum de testes [x]

- **Arquivos:**
  - [`src/testing/aliasing.rs`](src/testing/aliasing.rs) (L289-301)
  - [`src/testing/spectral.rs`](src/testing/spectral.rs) (L57-69)
- **Ação:** Consolidar a função duplicada `median` (e seus testes associados) em um local comum sob `src/testing/` e atualizar os locais de chamada.
- **Critério de aceite:** `cargo test --features testing` compila e passa.

### T16.3 — Integrar estratégia proptest órfã de `NamModelData` num teste real [x]

- **Arquivo:** [`tests/models/proptest_parsers.rs`](tests/models/proptest_parsers.rs) (L270-512)
- **Ação:** Criar um teste proptest real (ex: `prop_model_data_serialization_roundtrip`) que utiliza a estratégia `arbitrary_nam_model_data` para gerar dados arbitrários, serializá-los para JSON e desserializá-los de volta, validando a equivalência e o parse robusto.
- **Critério de aceite:** Novo teste integrado e verde na suite rápida.

### T16.4 — Remover `#[allow(dead_code)]` espúrios [x]

- **Arquivos:**
  - [`src/models/a2/model/set_weights.rs`](src/models/a2/model/set_weights.rs) (L276-289)
  - [`src/testing/spectral.rs`](src/testing/spectral.rs) (L56)
- **Ação:** Remover as anotações `#[allow(dead_code)]` desnecessárias, ajustando para `#[cfg_attr(not(test), allow(dead_code))]` ou removendo se as funções puderem ser expostas limpas.
- **Critério de aceite:** Lints limpos.

### T16.5 — Eliminar `CatalogGap` e exceções vazias [x]

- **Arquivo:** [`tests/models/meta_coherence.rs`](tests/models/meta_coherence.rs) (L21-29)
- **Ação:** Apagar a declaração da struct `CatalogGap` e o vetor estático `CATALOG_EXCEPTIONS` vazios, já que não há discrepâncias pendentes no catálogo de QA.
- **Critério de aceite:** `cargo test` verde.

### T16.6 — Implementar asserção e verificação de `max_frames_count` [x]

- **Arquivos:**
  - [`src/clap/processor/dsp/orchestrator.rs`](src/clap/processor/dsp/orchestrator.rs)
  - [`src/clap/processor/state.rs`](src/clap/processor/state.rs) (L122-123)
- **Ação:**
  1. Em `orchestrator.rs`, adicionar asserção em tempo de execução validando `n_samples <= self.max_frames_count`.
  2. Em builds release, se violado, setar `self.rt_status.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION)` e limitar `n_samples` de forma segura.
  3. Remover o `#[allow(dead_code)]` sobre `max_frames_count` em `state.rs`.
- **Critério de aceite:** O campo `max_frames_count` deixa de ser código morto.

### T16.7 — Centralizar ajudantes `generate_sine` redundantes [x]

- **Arquivos:**
  - `benches/common.rs:20`, `tests/common/signals.rs:13`, `benches/linear.rs:48`, `tests/models/namb_v2_*.rs`
- **Ação:** Centralizar as funções duplicadas `generate_sine`/`generate_sine_440hz` em um local comum de testes (ex: `tests/common/signals.rs`) e reusar os helpers.
- **Critério de aceite:** Todos os testes e benchmarks continuam compilando normalmente.

### T16.8 — Adicionar documentação na feature `pgo` no Cargo.toml [x]

- **Arquivo:** [`Cargo.toml`](Cargo.toml) (L124)
- **Ação:** Adicionar um comentário explicativo ao lado da feature `pgo` esclarecendo que ela serve como tag para o script de compilação de release, justificando sua presença embora esteja vazia de CFG.
- **Critério de aceite:** Comentário adicionado ao Cargo.toml.

---

## Sprint S17 — R15: Separação fina de superfície de teste (testing feature)

> **Ref:** [TODO-findings.md §R15](TODO-findings.md#r15--feature-testing-em-default-embarca-instrumentação-em-builds-de-produção--baixa) (L510-533)
>
> **Objetivo:** Evitar que a feature `testing` (que embarca o oráculo f64 e código off-RT pesado) compile por padrão em builds normais de desenvolvimento e distribuição, diminuindo a superfície de ataque e o tamanho dos binários.
>
> **Risco:** Baixo. Requer apenas ajustes finos no `Cargo.toml` e nos scripts de QA.

### T17.1 — Remover `testing` de default features no Cargo.toml [x]

- **Arquivo:** [`Cargo.toml`](Cargo.toml) (L117)
- **Ação:** Mudar a linha de defaults para: `default = ["standalone"]`.
- **Critério de aceite:** `cargo build --release` não compila mais os módulos sob a feature `testing`.

### T17.2 — Medir impacto do bloat com cargo bloat [x]

- **Ação:** Executar `cargo bloat --release` antes e após a remoção da feature `testing` dos defaults para registrar no commit/walkthrough o ganho real de tamanho binário (redução de superfície do .so CLAP).
- **Critério de aceite:** Relatório estatístico gerado.

### T17.3 — Atualizar scripts para explicitar `--features testing` [x]

- **Arquivos:**
  - [`utils/tests-quick.sh`](utils/tests-quick.sh)
  - [`utils/tests-long.sh`](utils/tests-long.sh)
- **Ação:** Adicionar `--features testing` em todas as invocações de `cargo test`/`cargo bench` que dependam do oráculo matemático ou da instrumentação de testes.
- **Critério de aceite:** Ambas as suítes (quick e long) executam com sucesso.

### T17.4 — Feature-gate a flag global `DISABLE_GATE` [x]

- **Arquivo:** [`src/dsp/pipeline/stages/input.rs`](src/dsp/pipeline/stages/input.rs) (L23)
- **Ação:** Assegurar que `DISABLE_GATE` e o atalho `NAM_DISABLE_GATE` estão rigidamente protegidos sob `#[cfg(feature = "testing")]` e não vazam no pipeline de produção.
- **Critério de aceite:** Zero referências não-condicionadas a `DISABLE_GATE`.

> **Resultado S17 (2026-07-16):**
>
> - **T17.1:** `Cargo.toml:117` → `default = ["standalone"]`. ✓
> - **T17.2:** Redução de superfície: `.so`: 4.3 M → 338 K (−92%), `nam-rs`: 5.5 M → 3.3 M (−40%), `.rlib`: 29 M → 27 M (−7%).
> - **T17.3:** Adicionado `--features testing` em `tests-quick.sh` (todas as fases), `tests-long.sh` (todas as fases via `timed_cargo_test` e benches), `build-release.sh` (standalone PGO). CLAP phase em `tests-long.sh` já tinha `testing` explícito.
> - **T17.4:** Já implementado — `DISABLE_GATE` (definição, re-exports e uso) e `NAM_DISABLE_GATE` (`main.rs`) sob `#[cfg(feature = "testing")]`.
> - **Nota:** `#[cfg(any(test, feature = "testing"))]` em `lib.rs` garante que `cargo test` (unit/integration) compile o módulo `testing` via `#[cfg(test)]`, mesmo sem `--features testing`. O flag explícito nos scripts serve como redundância defensiva.

---

## Sprint S18 — R16: Higiene de saída de logs e referências

> **Ref:** [TODO-findings.md §R16](TODO-findings.md#r16--higiene-de-saída-e-referências-obsoletas--baixa) (L536-558)
>
> **Objetivo:** Melhorar a legibilidade dos painéis de qualidade e remover notas antigas/obsoletas sobre arquivos temporários deletados.
>
> **Risco:** Baixo. Alterações puras de UI e cosméticos nos relatórios.

### T18.1 — Corrigir referências a arquivos transitórios [x]

- **Arquivo:** [`utils/quality-dashboard.sh`](utils/quality-dashboard.sh) (L1451)
- **Ação:** Substituir o texto que cita `TODO-findings.md Achado F3` por uma referência ao documento canônico estável: `docs/perceptual_validation.md#decomposition-cold-start`. Adotar a regra de não citar `TODO-*.md` em logs permanentes.
- **Critério de aceite:** Relatório limpo de referências transitórias.

### T18.2 — Esconder dumps de depuração sob `NAM_ORACLE_VERBOSE=1` [x]

- **Ação:** Alterar o formatador do oráculo para somente imprimir os dumps de depuração `PROD FIRST 10` e `ORACLE FIRST 10` se a variável de ambiente `NAM_ORACLE_VERBOSE=1` estiver ativada, despoluindo a tabela de resumo.
- **Critério de aceite:** Visualização padrão do dashboard exibe apenas a tabela sem intercalações.

### T18.3 — Explicitar motivo de `#[ignore]` no teste do gate [x]

- **Arquivo:** [`src/dsp/gate_test.rs`](src/dsp/gate_test.rs) (L300)
- **Ação:** Atualizar a anotação para `#[ignore = "proptest 10k casos; roda no tests-long (gate_envelope_continuity_proptest)"]` para esclarecer ao desenvolvedor por que este teste foi ignorado no quick loop.
- **Critério de aceite:** Comentário atualizado.

### T18.4 — Alinhar colunas no log de `isa_matrix_header_info` [x]

- **Ação:** Ajustar os espaçamentos na string formatada impressa pelo cabeçalho `isa_matrix_header_info` para corrigir o desalinhamento cosmético das colunas.
- **Critério de aceite:** Tabela impressa perfeitamente alinhada.

### T18.5 — Renomear teste de política em golden vectors [x]

- **Arquivo:** [`tests/models/golden_vectors.rs`](tests/models/golden_vectors.rs) (L1133)
- **Ação:** Renomear o teste `test_golden_vectors_wavenet_condition_lstm` para `test_policy_reject_condition_lstm`, refletindo sua real natureza de política fail-closed.
- **Critério de aceite:** Teste renomeado e funcional.

> **Resultado S18 (2026-07-16):**
>
> - **T18.1:** Substituída referência transitória `TODO-findings.md Achado F3` por `docs/perceptual_validation.md#decomposition-cold-start` em `quality-dashboard.sh:1451`. ✓
> - **T18.2:** dumps `PROD FIRST 10` / `ORACLE FIRST 10` em `reference_oracle_f64.rs:435-442` condicionados a `NAM_ORACLE_VERBOSE=1`. ✓
> - **T18.3:** `#[ignore]` em `gate_test.rs:300` substituído por `#[ignore = "proptest 10k casos; roda no tests-long (gate_envelope_continuity_proptest)"]`. ✓
> - **T18.4:** Caixa `isa_matrix_header_info` em `isa_parity.rs:693` realinhada — todas as linhas internas agora têm exatamente 62 colunas. ✓
> - **T18.5:** Teste `test_golden_vectors_wavenet_condition_lstm` renomeado para `test_policy_reject_condition_lstm` em `golden_vectors.rs:1133` e `tests/fixtures/README.md:751`. ✓
> - **Nota:** A variável `NAM_ORACLE_VERBOSE=1` não era usada anteriormente no código Rust; foi adicionada seguindo o mesmo padrão de `NAM_DISABLE_GATE` (`std::env::var(...).is_ok()`).
> - **Correção (pós-auditoria):** O dashboard (`quality-dashboard.sh`) não passava `--features testing` nas invocações de `cargo test`, quebrando a compilação dos binários `parity` (e potencialmente `models`) após o Sprint S17 remover `testing` dos defaults. Adicionado `--features testing` em todas as 6 invocações de `cargo test` (exceto `cargo bench` que não usa `nam_rs::testing`).
> - **Correção 2 (contrato de qualidade):** `verify_contract` comparava `dash_label` (nome canônico, ex: "BossWN-standard") com `contract_label` (chave completa, ex: "BossWN-standard @48000 Live"), fazendo com que TODAS as linhas exibissem "(i) nao encontrado na execucao atual". Corrigido: ambos os lados agora são normalizados (strip `@rate` e modo) antes da comparação.

---

## VF — Verificação Final Integrada EP-R5

> **Gate de aceite do épico inteiro.**

### VF5.1 — Lints e Compilações

- `utils/lints.sh` deve passar totalmente limpo com 0 erros/warnings.

### VF5.2 — Suite rápida verde

- `utils/tests-quick.sh` deve reportar sucesso em todas as fases.

---

## Épico EP-R6 (opcional/contínuo) — Guardas de segunda ordem [ADIADO]

> Nota do PO: Guardado para o futuro.
>
> **Origem:** [TODO-findings.md §EP-R6](TODO-findings.md#ep-r6-opcionalcontínuo--guardas-de-segunda-ordem-p2--p4) (Auditoria de Resiliência & Robustez, 2026-07-14)
>
> **Escopo:** P2 (Mutation testing com cargo-mutants, **Proposta**) + P4 (asm-gate com dsp_hotpath.asm, **Proposta**).
>
> **Pré-requisito:** EP-R5 concluído.
>
> **Invariante absoluto:** sem custo ou impacto em tempo de compilação ou execução de produção. Risco: zero.

---

## EP-R6 — Sumário dos Sprints

| Sprint  | Finding                                             | Risco | Arquivos tocados | Estimativa |
| ------- | --------------------------------------------------- | ----- | ---------------- | ---------- |
| **S19** | P2 + P4 — Guardas e portões estatísticos de codegen | Baixo | 2                | ~90 min    |
| **VF**  | Verificação final integrada EP-R6                   | —     | 0                | ~15 min    |

---

## Sprint S19 — P2 & P4: Guardas e portões estatísticos de codegen

> **Ref:** [TODO-findings.md §P2](TODO-findings.md#p2--mutation-testing-com-cargo-mutants-como-extensão-da-filosofia-anti-placebo) (L579-590) e [TODO-findings.md §P4](TODO-findings.md#p4-aproveitar-o-targetdsphotpathasm-como-gate-de-regressão-de-codegen) (L608-616)
>
> **Objetivo:** Adicionar ferramentas externas de proteção offline para prevenir regressões de cobertura de teste (mutants) e inlining quebrado (asm-gate) sem afetar a produção.

### T19.1 — Criar script de testes de mutação offline `mutants.sh` [ ]

- **Arquivo (novo):** [`utils/mutants.sh`](utils/mutants.sh)
- **Ação:**
  1. Criar o script contendo chamadas recomendadas de `cargo mutants` focadas nos módulos-fortaleza: `src/loader/`, `src/common/spsc/`, `src/dsp/gate.rs` e `src/dsp/adaptive.rs`.
  2. Adicionar documentação operacional esclarecendo que este teste é de execução periódica off-line (mensal) e não deve rodar no Quick ou Long CI devido ao alto custo computacional.
- **Critério de aceite:** Script criado, com cabeçalho SPDX e permissão de execução.

### T19.2 — Criar portão de assembly `asm-gate.sh` [ ]

- **Arquivo (novo):** [`utils/asm-gate.sh`](utils/asm-gate.sh)
- **Ação:**
  1. Criar um script bash que inspeciona o arquivo assembly gerado `target/dsp_hotpath.asm` (produzido pelo pipeline).
  2. Extrair métricas estáticas básicas (número de `call` nos símbolos de DSP quente para detectar inlines quebrados; contagem de `vzeroupper` excessivos; contagem de acessos a stack `mov [rsp...]` para spills excessivos).
  3. Comparar com limites aceitáveis e falhar com mensagens instrutivas se houver desvio, agindo como um gate de codegen.
- **Critério de aceite:** Script criado com cabeçalho SPDX e testado contra um arquivo assembly real.

---

## VF — Verificação Final Integrada EP-R6

### VF6.1 — Execução dos scripts

- Validar que ambos os scripts rodam sem erros e geram saída legível quando invocados localmente.

---

## Épico EP-R7 — Fechar vetores residuais de UAF e RT-safety (R17 + R18)

> **Origem:** [TODO-findings.md §EP-R7](TODO-findings.md#ep-r7--fechar-vetores-residuais-de-uaf-e-rt-safety-r17--r18--primeiro-mesma-classe-de-bug-de-r2r5-já-corrigidos) (Auditoria de Resiliência & Robustez, 2026-07-16)
>
> **Escopo:** R17 (UAF na janela flutuante, **ALTA**) + R18 (log síncrono no callback RT, **ALTA**).
>
> **Invariante absoluto:** sem alteração de comportamento sonoro, zero logs síncronos na thread RT, zero UAF no bootstrap de janela flutuante. Critério de aceite global: `utils/tests-quick.sh` verde.

---

## EP-R7 — Sumário dos Sprints

| Sprint  | Finding                                                    | Risco       | Arquivos tocados | Estimativa |
| ------- | ---------------------------------------------------------- | ----------- | ---------------- | ---------- |
| **S20** | R17 + R18 — Segurança de Lifecycle e RT-safety de thread   | Baixo-Médio | 6                | ~60 min    |
| **VF**  | Verificação final integrada EP-R7                          | —           | 0                | ~15 min    |

---

## Sprint S20 — R17 & R18: Segurança de Lifecycle e RT-safety de thread

> **Ref:** [TODO-findings.md §R17](TODO-findings.md#r17--uaf-residual-nampluginwindownew-desreferencia-shared0-sem-alive_fence-na-thread-da-janela-flutuante--alta) (L786-857) e [TODO-findings.md §R18](TODO-findings.md#r18--logerrologinfo-alcançáveis-no-thread-rt-do-pipewire-via-configure_realtime_thread--alta) (L861-930)
>
> **Objetivo:** Eliminar o UAF na inicialização da janela flutuante passando o `alive_fence` e escala diretamente do main thread, e expurgar chamadas `log::*` síncronas de dentro de `configure_realtime_thread` na thread RT salvando erros em flags atômicas.

### T20.1 — Adicionar campos de erros e telemetria atômicos em `RtStatusFlags` [x]

- **Arquivo:** [`src/common/spsc/status.rs`](src/common/spsc/status.rs)
- **Ação:**
  1. Adicionar campos atômicos ao struct `RtStatusFlags`:
     - `rt_affinity_err`: `AtomicI32`
     - `rt_sched_err`: `AtomicI32`
     - `rt_getsched_err`: `AtomicI32`
     - `rt_target_cpu`: `AtomicI32`
  2. Inicializar os campos com seus respectivos valores padrão em `new()`.
- **Critério de aceite:** `cargo check` passa.

### T20.2 — Remover logs e repassar erros atômicos em `configure_realtime_thread` [x]

- **Arquivo:** [`src/standalone/rt_setup/thread.rs`](src/standalone/rt_setup/thread.rs)
- **Ação:**
  1. Remover importação e uso de macros `log::error!`/`log::info!`.
  2. Substituir logs por atribuições atômicas de erro:
     - CPU OOB: registrar `-1` em `rt_affinity_err` e `target_cpu` em `rt_target_cpu`.
     - `pthread_setaffinity_np` falhou: registrar `ret_aff` (errno) em `rt_affinity_err` e `target_cpu` em `rt_target_cpu`.
     - `pthread_setschedparam` falhou: registrar `ret_sched` (errno) em `rt_sched_err`.
     - `pthread_getschedparam` falhou: registrar `ret_getsched` (errno) em `rt_getsched_err`.
- **Critério de aceite:** Zero ocorrências de `log::` no método `configure_realtime_thread`.

### T20.3 — Consumir erros e telemetria RT de forma segura na thread principal [x]

- **Arquivo:** [`src/standalone/rt_setup/telemetry.rs`](src/standalone/rt_setup/telemetry.rs)
- **Ação:**
  1. No método `poll_rt_status`, realizar a leitura/swap seguro de `rt_affinity_err`, `rt_sched_err`, `rt_getsched_err` e `rt_target_cpu`.
  2. Emitir as respectivas mensagens de erro formatadas (`log::error!`, etc.) se os valores de erro forem não-nulos.
  3. No caso de sucesso de escalonamento RT e afinidade, imprimir a mensagem detalhada de otimização de thread clássica com cores adequadas.
- **Critério de aceite:** Compilação com sucesso e logs de inicialização visíveis e corretos no standalone.

### T20.4 — Extrair `alive_fence` e escala no main thread CLAP [x]

- **Arquivo:** [`src/clap/extensions/gui.rs`](src/clap/extensions/gui.rs)
- **Ação:**
  1. Em `set_parent` (embutido) e `set_transient` (flutuante):
     - Clonar `self.shared.cold.alive_fence` diretamente do main thread.
     - Obter o fator de escala `scale_factor` a partir de `self.shared.cold.gui_scale_factor.load(...)`.
  2. Passar ambos como parâmetros para a chamada de `NamPluginWindow::new(...)`.
- **Critério de aceite:** `cargo check` passa.

### T20.5 — Modificar assinatura de `NamPluginWindow::new` para prevenir UAF [x]

- **Arquivo:** [`src/clap/gui/window/state.rs`](src/clap/gui/window/state.rs)
- **Ação:**
  1. Alterar `pub fn new(...)` para receber `alive_fence: Arc<AtomicBool>` e `gui_scale: f32` explicitamente.
  2. Remover as linhas que fazem leituras sem fence (`(*shared.0).cold.gui_scale_factor` e `&*shared.0`).
  3. Atribuir os parâmetros diretamente nos campos correspondentes da estrutura criada.
- **Critério de aceite:** `cargo check` passa; sem desreferências cruas e cegas de `shared.0` durante o bootstrap do construtor.

### T20.6 — Adicionar baterias de testes unitários e coerência estrutural [x]

- **Arquivos:**
  - [`src/clap/gui/window/state.rs`](src/clap/gui/window/state.rs)
  - [`tests/models/meta_coherence.rs`](tests/models/meta_coherence.rs)
- **Ação:**
  1. Adicionar o teste unitário `test_window_safe_shared_boundary` validando que `safe_shared` retorna `None` se a fence estiver desativada.
  2. Adicionar o meta-teste `test_configure_realtime_thread_no_logging` no coherence.rs assegurando estaticamente que `configure_realtime_thread` não possui logs ativos.
- **Critério de aceite:** `cargo test` passa limpo.

---

## VF — Verificação Final Integrada EP-R7

### VF7.1 — Lints e Compilação rápida

- Executar `utils/lints.sh` e assegurar 0 erros/avisos.
- Executar `utils/tests-quick.sh` e assegurar que tudo passa com sucesso.

---

## Épico EP-R8 — Blindagem da fronteira CLAP host↔plugin (R19 + R24 + R25)

> **Origem:** [TODO-findings.md §EP-R8](TODO-findings.md#ep-r8--blindagem-da-fronteira-clap-hostplugin-r19--r24--r25) (Auditoria de Resiliência & Robustez, 2026-07-16)
>
> **Escopo:** R19 (eliminação de `as_main_thread_unchecked` em track_info), R24 (registro/uso de `thread-check` com debug_assert!), R25 (padronização de `PoisonError` em locks de ColdShared).
>
> **Invariante absoluto:** sem alteração de comportamento de processamento de áudio, zero impacto de runtime em release (zero overhead no thread RT), assertivas de thread acionadas apenas em debug.
>
> **Pré-requisito:** EP-R7 concluído.

---

## EP-R8 — Sumário dos Sprints

| Sprint  | Finding                                                    | Risco | Arquivos tocados | Estimativa |
| ------- | ---------------------------------------------------------- | ----- | ---------------- | ---------- |
| **S21** | R19 + R24 + R25 — Blindagem da fronteira CLAP host↔plugin  | Baixo | 8                | ~45 min    |
| **VF**  | Verificação final integrada EP-R8                          | —     | 0                | ~15 min    |

---

## Sprint S21 — R19, R24 & R25: Blindagem da fronteira CLAP host↔plugin

> **Ref:** [TODO-findings.md §R19](TODO-findings.md#r19--plugintrackinfoimplchanged-usa-as_main_thread_unchecked-sem-runtime-guard--média), [TODO-findings.md §R24](TODO-findings.md#r24--extensão-clap-thread-check-não-registrada--baixa) e [TODO-findings.md §R25](TODO-findings.md#r25--poisonerror-de-mutex-descartado-silenciosamente-em-preset_loadhousekeeping--baixa)
>
> **Objetivo:** Hardening da fronteira FFI do CLAP entre o host e o plugin via thread checking em debug e tratamento resiliente de envenenamento de mutexes.

### T21.1 — Adicionar o feature "thread-check" no Cargo.toml [x]

- **Arquivo:** [`Cargo.toml`](Cargo.toml)
- **Ação:** Adicionar `"thread-check"` à lista de features da dependência `clack-extensions`.
- **Critério de aceite:** `cargo check` passa sem problemas de resolução de dependências.

### T21.2 — Eliminar `as_main_thread_unchecked` em `track_info.rs` [x]

- **Arquivo:** [`src/clap/extensions/track_info.rs`](src/clap/extensions/track_info.rs)
- **Ação:**
  1. Remover o uso de `as_main_thread_unchecked()`.
  2. Substituir por `unsafe { self.host.with_arbitrary_lifetime() }` para obter um handle mutável seguro mantendo a tipagem sem realizar conversões de thread não-checadas.
- **Critério de aceite:** `cargo check` passa; sem `as_main_thread_unchecked` no arquivo.

### T21.3 — Adicionar helper de runtime thread checking `debug_assert_main_thread` [x]

- **Arquivo:** [`src/clap/plugin/main_thread/mod.rs`](src/clap/plugin/main_thread/mod.rs)
- **Ação:**
  1. Implementar o helper `debug_assert_main_thread(host: &HostMainThreadHandle)` usando a extensão `HostThreadCheck`.
  2. A função deve consultar a extensão via `host.shared().get_extension::<clack_extensions::thread_check::HostThreadCheck>()` e rodar `debug_assert!` validando que `is_main_thread(&host.shared())` retorna `true` ou `None`.
- **Critério de aceite:** O helper compila com sucesso.

### T21.4 — Inserir `debug_assert_main_thread` nos pontos críticos do CLAP [x]

- **Arquivos:**
  - [`src/clap/extensions/state.rs`](src/clap/extensions/state.rs) (save, load)
  - [`src/clap/extensions/state_context.rs`](src/clap/extensions/state_context.rs) (métodos de save/load/etc.)
  - [`src/clap/extensions/preset_load.rs`](src/clap/extensions/preset_load.rs) (load)
  - [`src/clap/extensions/gui.rs`](src/clap/extensions/gui.rs) (create, destroy, show, hide, etc.)
  - [`src/clap/extensions/track_info.rs`](src/clap/extensions/track_info.rs) (changed)
- **Ação:** Invocar o helper `debug_assert_main_thread` no início de cada um dos métodos para assegurar a corretude do threading da chamada vinda do host em builds debug.
- **Critério de aceite:** `cargo check` passa em todos os arquivos modificados.

### T21.5 — Tratar `PoisonError` graciosamente nos locks de `ColdShared` [x]

- **Arquivos:**
  - [`src/clap/plugin/main_thread/housekeeping.rs`](src/clap/plugin/main_thread/housekeeping.rs)
  - [`src/clap/extensions/preset_load.rs`](src/clap/extensions/preset_load.rs)
  - [`src/clap/extensions/params/main.rs`](src/clap/extensions/params/main.rs)
- **Ação:** Substituir locking direto (`if let Ok(...)`) por `.unwrap_or_else(|e| { log::error!(...); e.into_inner() })` para que o dado pendente ainda seja recuperado em caso de lock envenenado e o erro seja devidamente reportado.
- **Critério de aceite:** Ausência de descartes silenciosos de poison de mutex nos arquivos especificados.

### T21.6 — Adicionar testes unitários/coerência de runtime thread check [x]

- **Arquivo:** [`tests/models/meta_coherence.rs`](tests/models/meta_coherence.rs) (ou unitário adequado)
- **Ação:** Criar testes estruturados ou estáticos para assegurar que as chamadas críticas do main thread possuem a barreira de thread check e que a falha de envenenamento é tratada corretamente.
- **Critério de aceite:** `cargo test` passa e novos testes validam as invariantes de hardening.

---

## VF — Verificação Final Integrada EP-R8

### VF8.1 — Lints e Compilação rápida

- Executar `utils/lints.sh` e assegurar 0 erros/avisos.
- Executar `utils/tests-quick.sh` e assegurar que tudo passa com sucesso.

---

## Épico EP-R9 — Robustez de carregamento e cobertura anti-regressão (R20 + R21)

> **Origem:** [TODO-findings.md §EP-R9](TODO-findings.md#ep-r9--robustez-de-carregamento-e-cobertura-anti-regressao-r20--r21) (Auditoria de Resiliência & Robustez, 2026-07-16)
>
> **Escopo:** R20 (substituir `.expect()` por propagação de erro em `activate()` CLAP e nos dispatchers do loader); R21 (adicionar estratégias proptest adversariais para LSTM dinâmico, A2-Dynamic e SlimmableContainer).
>
> **Invariante absoluto:** sem alteração de comportamento de processamento de áudio em condições normais, zero aborts/panics em falhas de alocação de memória de host ou em carregamento de topologias inválidas.
>
> **Pré-requisito:** EP-R8 concluído.

---

## EP-R9 — Sumário dos Sprints

| Sprint  | Finding                                                    | Risco | Arquivos tocados | Estimativa |
| ------- | ---------------------------------------------------------- | ----- | ---------------- | ---------- |
| **S22** | R20 — Substituição de `.expect()` por propagação de erro   | Baixo | 5                | ~45 min    |
| **S23** | R21 — Cobertura de proptests adversariais (LSTM/A2-D/Slim) | Baixo | 1                | ~30 min    |
| **VF**  | Verificação final integrada EP-R9                          | —     | 0                | ~15 min    |

---

## Sprint S22 — R20: Substituição de `.expect()` por propagação de erro em alocações de produção

> **Ref:** [TODO-findings.md §R20](TODO-findings.md#r20--expect-residual-em-alocacoes-de-producao-loader--activate-clap--media)
>
> **Objetivo:** Eliminar pânicos residuais nas alocações de produção convertendo chamadas a `.expect()` em propagação limpa via `?`.

### T22.1 — Propagar erro de alocação de buffers no processador CLAP [x]

- **Arquivo:** [`src/clap/processor/mod.rs`](src/clap/processor/mod.rs)
- **Ação:**
  1. Localizar as 12 ocorrências de `AlignedVec::new(buf_capacity, 0.0f32).expect(...)` na função `activate()`.
  2. Substituí-las por tratamento de erro, convertendo o `NamErrorCode` resultante em `PluginError::Message` (usando `.map_err(...)?` ou similar).
  3. Localizar a chamada a `ConvEngine::new(...).expect(...)`.
  4. Tratar o erro de forma idêntica propagando via `?`.
- **Critério de aceite:** `cargo check` passa; sem ocorrências de `.expect(` de produção no arquivo.

### T22.2 — Propagar erro de alocação no dispatcher WaveNet Standard [x]

- **Arquivo:** [`src/loader/dispatcher/wavenet/standard.rs`](src/loader/dispatcher/wavenet/standard.rs)
- **Ação:**
  1. Localizar os construtores de `AlignedVec` que utilizam `.expect("allocation should succeed for test-sized buffers")` (4 ocorrências).
  2. Substituir o `.expect(...)` por propagação direta via `?` (aproveitando que o retorno é `anyhow::Result`).
- **Critério de aceite:** Sem ocorrências de `.expect(` no arquivo; `cargo check` passa.

### T22.3 — Propagar erro de alocação no dispatcher WaveNet Dynamic [x]

- **Arquivo:** [`src/loader/dispatcher/wavenet/dynamic.rs`](src/loader/dispatcher/wavenet/dynamic.rs)
- **Ação:**
  1. Localizar os construtores de `AlignedVec` com `.expect(...)` (7 ocorrências).
  2. Substituir por propagação direta via `?`.
- **Critério de aceite:** Sem ocorrências de `.expect(` no arquivo; `cargo check` passa.

### T22.4 — Propagar erro de alocação no dynamic builder do LSTM [x]

- **Arquivo:** [`src/loader/dispatcher/lstm/dynamic_builder.rs`](src/loader/dispatcher/lstm/dynamic_builder.rs)
- **Ação:**
  1. Localizar as duas alocações de `AlignedVec` com `.expect(...)` em `build_lstm_dynamic`.
  2. Substituir por propagação direta via `?`.
- **Critério de aceite:** Sem ocorrências de `.expect` no arquivo.

### T22.5 — Propagar erro de alocação no dispatcher ConvNet [x]

- **Arquivo:** [`src/loader/dispatcher/convnet/mod.rs`](src/loader/dispatcher/convnet/mod.rs)
- **Ação:**
  1. Localizar as ocorrências de `AlignedVec` com `.expect(...)` (8 ocorrências).
  2. Substituir por propagação direta via `?`.
- **Critério de aceite:** Sem ocorrências de `.expect` no arquivo.

---

## Sprint S23 — R21: Cobertura de proptests adversariais

> **Ref:** [TODO-findings.md §R21](TODO-findings.md#r21--lacuna-de-fuzzing-lstm-dinamico-a2-dynamic-e-slimmablecontainer-sem-estrategia-proptest-adversarial--media)
>
> **Objetivo:** Adicionar geradores de JSON de modelos proptest com dimensões de topologia fora dos limites tolerados para as arquiteturas que hoje não possuem cobertura estocástica adversarial.

### T23.1 — Adicionar estratégia e teste adversarial para LSTM dinâmico [x]

- **Arquivo:** [`tests/models/proptest_parsers.rs`](tests/models/proptest_parsers.rs)
- **Ação:**
  1. Implementar `adversarial_lstm_json_strategy() -> impl Strategy<Value = String>` que gere configurações de LSTM inválidas (ex.: `hidden_size` excedendo os limites ou incoerências).
  2. Adicionar o teste `prop_fuzz_adversarial_lstm_dims` utilizando a estratégia e certificando que o parser/loader rejeita os modelos com erro ou os trata de forma controlada sem pânico/aborto.
  3. Marcar o teste com `#[test]` e `#[ignore]`.
- **Critério de aceite:** `cargo test` passa (com o teste ignorado por padrão).

### T23.2 — Adicionar estratégia e teste adversarial para A2-Dynamic [x]

- **Arquivo:** [`tests/models/proptest_parsers.rs`](tests/models/proptest_parsers.rs)
- **Ação:**
  1. Implementar `adversarial_a2_dynamic_json_strategy() -> impl Strategy<Value = String>` visando gerar topologias de WaveNet com tamanho de canais ou arranjos inválidos, simulando a topologia A2-Dynamic sob estresse.
  2. Adicionar o teste `prop_fuzz_adversarial_a2_dynamic_dims` anotado com `#[test]` e `#[ignore]`.
- **Critério de aceite:** Testes novos passam quando executados explicitamente.

### T23.3 — Adicionar estratégia e teste adversarial para SlimmableContainer [x]

- **Arquivo:** [`tests/models/proptest_parsers.rs`](tests/models/proptest_parsers.rs)
- **Ação:**
  1. Implementar `adversarial_container_json_strategy() -> impl Strategy<Value = String>` que gere JSONs de contêineres/submodelos excedendo profundidade de recursão ou número máximo de submodelos.
  2. Adicionar o teste `prop_fuzz_adversarial_container_dims` anotado com `#[test]` e `#[ignore]`.
- **Critério de aceite:** `cargo test` compila e os novos testes executam corretamente com `-- --ignored`.

---

## VF — Verificação Final Integrada EP-R9

### VF9.1 — Lints e Compilação rápida

- Executar `utils/lints.sh` e assegurar 0 erros/avisos.
- Executar `utils/tests-quick.sh` e assegurar que tudo passa com sucesso.

### VF9.2 — Execução explícita das estratégias adversariais

- Executar individualmente os testes adicionados usando:

  ```bash
  cargo test --test models proptest_parsers::prop_fuzz_adversarial_lstm_dims -- --ignored
  cargo test --test models proptest_parsers::prop_fuzz_adversarial_a2_dynamic_dims -- --ignored
  cargo test --test models proptest_parsers::prop_fuzz_adversarial_container_dims -- --ignored
  ```

- Validar que nenhum causa falso-positivo ou pânico não tratado.

---

## Épico EP-R10 — Observabilidade e higiene remanescente (R22 + R23 + R26)

> **Origem:** [TODO-findings.md §EP-R10](TODO-findings.md#ep-r10--observabilidade-e-higiene-remanescente-r22--r23--r26) (Auditoria de Resiliência & Robustez, 2026-07-16)
>
> **Escopo:** R22 (telemetria de buffer-miss no PipeWire), R26 (limpeza de campos mortos, mem::zeroed e unwrap), R23 (sprint de documentação SAFETY em unsafe remanescentes).
>
> **Invariante absoluto:** sem alteração de comportamento de processamento de áudio, zero regressões em lints e cobertura de testes.
>
> **Pré-requisito:** EP-R9 concluído.

---

## EP-R10 — Sumário dos Sprints

| Sprint  | Finding                                                  | Risco  | Arquivos tocados | Estimativa |
| ------- | -------------------------------------------------------- | ------ | ---------------- | ---------- |
| **S24** | R22 — Telemetria de buffer-miss no PipeWire              | Baixo  | 7                | ~30 min    |
| **S25** | R26 — Resolução e Limpeza de Campos Mortos/Incorretos    | Baixo  | 6                | ~30 min    |
| **S26** | R23 — Higiene de unsafe com comentários SAFETY           | Baixo  | 8                | ~30 min    |
| **VF**  | Verificação final integrada EP-R10                       | —      | 0                | ~15 min    |

---

## Sprint S24 — R22: Telemetria de buffer-miss no PipeWire

> **Ref:** [TODO-findings.md §R22](TODO-findings.md#r22--telemetria-de-buffer-misses-no-host-pipewire--baixa)
>
> **Objetivo:** Adicionar os contadores de underruns/xruns do PipeWire ao `RtStatusFlags` e expô-los no dashboard de telemetria.

### T24.1 — Adicionar campos de miss a `RtStatusFlags` e `TelemetrySnapshot` [X]

- **Arquivos:**
  - [`src/common/spsc/status.rs`](src/common/spsc/status.rs)
  - [`src/common/diagnostics/snapshot.rs`](src/common/diagnostics/snapshot.rs)
- **Ação:** Adicionar `pw_buffer_miss` e `playback_miss` em `RtStatusFlags` (como `AtomicU32`) e na `TelemetrySnapshot`.

### T24.2 — Incrementar contadores nas falhas de `dequeue_buffer` [X]

- **Arquivos:**
  - [`src/standalone/pw_host/rt_callback/process.rs`](src/standalone/pw_host/rt_callback/process.rs)
  - [`src/dsp/pipeline/output_pw.rs`](src/dsp/pipeline/output_pw.rs)
  - [`src/standalone/pw_host/playback.rs`](src/standalone/pw_host/playback.rs)
  - [`src/standalone/pw_host/run.rs`](src/standalone/pw_host/run.rs)
- **Ação:** Passar `rt_status` para a thread de playback e incrementar o respectivo contador com `Ordering::Relaxed` se `dequeue_buffer()` retornar `None`.

### T24.3 — Expor contadores no dashboard de telemetria [X]

- **Arquivo:** [`src/standalone/rt_setup/telemetry.rs`](src/standalone/rt_setup/telemetry.rs)
- **Ação:** Logar avisos de buffer-miss no `poll_rt_status` se os contadores forem maiores que zero.

---

## Sprint S25 — R26: Resolução e Limpeza de Campos Mortos/Incorretos

> **Ref:** [TODO-findings.md §R26](TODO-findings.md#r26--diagnóstico)
>
> **Objetivo:** Limpeza de campos não utilizados (ou redundantes), inicializações unidiomáticas e panic no FFI.

### T25.1 — Remover `alive` de `DialogSharedState` e `IrDialogSharedState` [X]

- **Arquivos:**
  - [`src/clap/gui/ui/zones/dialog_state.rs`](src/clap/gui/ui/zones/dialog_state.rs)
  - [`src/clap/gui/ui/zones/file_dialogs.rs`](src/clap/gui/ui/zones/file_dialogs.rs)
- **Ação:** Eliminar o campo `alive` de ambas as structs e simplificar os testes unitários removendo as asserções de `alive`.

### T25.2 — Ajustar buffers `os_*` em `DspBuffers` [X]

- **Arquivo:** [`src/dsp/pipeline/context.rs`](src/dsp/pipeline/context.rs)
- **Ação:** Remover `#[allow(unused)]` dos buffers `os_` já que eles são consumidos ativamente no pipeline de oversampling.

### T25.3 — Substituir `mem::zeroed` em `thread.rs` por construções seguras [X]

- **Arquivo:** [`src/standalone/rt_setup/thread.rs`](src/standalone/rt_setup/thread.rs)
- **Ação:** Trocar inicializações com `mem::zeroed` por `MaybeUninit` (para `cpu_set_t`) e inicializações de struct explícitas (para `sched_param`).

### T25.4 — Corrigir `unwrap` em `state.rs` [X]

- **Arquivo:** [`src/clap/extensions/state.rs`](src/clap/extensions/state.rs)
- **Ação:** Substituir `.unwrap()` por `.unwrap_or_default()`.

---

## Sprint S26 — R23: Higiene de unsafe com comentários SAFETY

> **Ref:** [TODO-findings.md §R23](TODO-findings.md#r23--higiene-de-unsafe-remanescente-fora-da-tabela-r12--baixa)
>
> **Objetivo:** Adicionar o comentário `// SAFETY:` detalhado para cada bloco unsafe que carece de documentação na produção.

### T26.1 — Documentar blocos unsafe [X]

- **Arquivos:**
  - [`src/clap/plugin/shared.rs`](src/clap/plugin/shared.rs)
  - [`src/dsp/oversample.rs`](src/dsp/oversample.rs)
  - [`src/dsp/resampler/core.rs`](src/dsp/resampler/core.rs)
  - [`src/clap/gui/ui/zones/dialog_state.rs`](src/clap/gui/ui/zones/dialog_state.rs) (nota: não aplicável após remoção, mas auditamos todos)
  - [`src/dsp/cabsim/conv.rs`](src/dsp/cabsim/conv.rs)
  - [`src/dsp/gate.rs`](src/dsp/gate.rs)
  - [`src/models/a2/grouped_conv1d/simd.rs`](src/models/a2/grouped_conv1d/simd.rs)
  - [`src/models/convnet/batch_norm.rs`](src/models/convnet/batch_norm.rs)
- **Ação:** Adicionar comentários explicando as invariantes garantidas por construção para cada bloco unsafe/get_unchecked/transmute.

### T26.2 — Tratar e documentar `madvise` em `bridge.rs` [X]

- **Arquivo:** [`src/standalone/pw_host/bridge.rs`](src/standalone/pw_host/bridge.rs)
- **Ação:** Adicionar comentário de SAFETY e validar o retorno de `madvise`, emitindo log de aviso em caso de erro.

---

## VF — Verificação Final Integrada EP-R10

### VF10.1 — Lints e Compilação rápida

- Executar `utils/lints.sh` e assegurar 0 erros/avisos.
- Executar `utils/tests-quick.sh` e assegurar que tudo passa com sucesso.

---

## Sprint S27 — EP-R11: Fechar pendências residuais das rodadas EP-R1…EP-R5

> **Ref:** [TODO-findings.md §EP-R11](TODO-findings.md#ep-r11--fechar-pendências-residuais-das-rodadas-ep-r1ep-r5-r8-h--r10--r2nonnull--r14--p3) (L1347-1369)
>
> **Objetivo:** Resolver as cinco pendências de robustez e testes residuais acumuladas nas primeiras rodadas de auditoria.

### T27.1 — R8-h: Condicionar flag de overflow da GC [x]

- **Arquivo:** [`src/common/spsc/gc.rs`](src/common/spsc/gc.rs)
- **Ação:** Condicionar a ativação do flag `rt_status.set_flag(super::RT_STATUS_GC_OVERFLOW)` ao retorno `true` do método `gc_overflow.push(i)`.

### T27.2 — R10: Caso de teste de block-size acima do máximo negociado [x]

- **Arquivo:** [`src/clap/processor_stress_test.rs`](src/clap/processor_stress_test.rs)
- **Ação:** Adicionar o caso de teste `test_host_contract_violation_block_size` forçando o envio de blocos de 600 frames em uma configuração com `max_frames_count = 512`. O teste deve esperar pânico com mensagem específica em compilação `debug` (via `#[cfg_attr(debug_assertions, should_panic(expected = "Host contract violation"))]`) e, em `release`, verificar se `RT_STATUS_HOST_CONTRACT_VIOLATION` é setado corretamente.

### T27.3 — R2: NamClapSharedRef com NonNull privado [x]

- **Arquivos:**
  - [`src/clap/plugin/shared.rs`](src/clap/plugin/shared.rs)
  - [`src/clap/gui/window/state.rs`](src/clap/gui/window/state.rs)
  - [`src/clap/extensions/gui.rs`](src/clap/extensions/gui.rs)
  - [`src/clap/gui/window/window_test.rs`](src/clap/gui/window/window_test.rs)
- **Ação:**
  1. Em `shared.rs`, mudar `NamClapSharedRef` para encapsular `std::ptr::NonNull<NamClapShared>` em vez de `*const NamClapShared` de forma privada (remover o modificador `pub` interno).
  2. Fornecer os métodos `pub unsafe fn new(ptr: *const NamClapShared) -> Self`, `pub fn as_ptr(&self) -> *const NamClapShared` e `pub unsafe fn as_ref(&self) -> &'static NamClapShared`.
  3. Atualizar todos os pontos de consumo e instanciamento de `NamClapSharedRef` nos arquivos listados acima.

### T27.4 — R14: Roundtrip de serialização e consolidação de sinal senoidal [x]

- **Arquivos:**
  - [`tests/models/proptest_parsers.rs`](tests/models/proptest_parsers.rs)
  - [`tests/common/signals.rs`](tests/common/signals.rs)
  - [`benches/common.rs`](benches/common.rs)
- **Ação:**
  1. Em `proptest_parsers.rs`, marcar `prop_model_data_serialization_roundtrip` com o atributo `#[ignore]` para que seja ativamente exercitado no fuzzing ágil da Fase 3 de `tests-quick.sh`.
  2. Em `signals.rs` e `common.rs`, remover a lógica duplicada de geração matemática senoidal em `generate_sine_440hz` e delegar diretamente para `nam_rs::testing::aliasing::generate_sine(440.0, 48000, num_samples, 1.0)`.

### T27.5 — P3: Avaliação de assert_unchecked no DSP e as_chunks no FiLM [x]

- **Arquivos:**
  - [`src/dsp/stage.rs`](src/dsp/stage.rs)
  - [`src/models/a2/film.rs`](src/models/a2/film.rs)
- **Ação:**
  1. Em `stage.rs`, avaliar e testar a migração do uso de `get_unchecked` e `get_unchecked_mut` para asserções seguras via `core::hint::assert_unchecked` seguidas por indexações padrão (ex. `core::hint::assert_unchecked(p < self.up_ring.len()); self.up_ring[p] = x;`).
  2. Em `film.rs`, avaliar a adoção de `as_chunks_mut()` no loop de SIMD de FiLM (`apply_modulation`) como prova de conceito para chunks estáticos do Rust stable.

---

## VF — Verificação Final Integrada EP-R11

### VF11.1 — Lints e Validação de Cobertura

- Rodar `utils/lints.sh` garantindo zero novos avisos ou erros.
- Rodar `utils/tests-quick.sh` assegurando que todos os testes passaram e que o novo proptest de serialização foi executado na Fase 3.
