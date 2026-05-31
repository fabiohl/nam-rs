<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Plano de Sprints (Auditoria 2026-05-29 + Pesquisa Avant-Garde 2026-05-29)

> Plano de execução em **duas partes complementares**:
>
> **Parte I — Remediação (Épicos 1–8)** — decorrente da auditoria multi-disciplinar (DSP, SIMD/microarquitetura, modelos NN, plugin CLAP, host PipeWire/RT, loader/segurança) realizada em 29/05/2026 sobre todo o crate `nam-rs`. Foco em correção, soundness, paridade e dívida arquitetural.
>
> **Parte II — Inovações Avant-Garde (Épicos 9–13)** — decorrente do painel `pesquisador-inovador` em 29/05/2026, cobrindo fronteiras de 2026 em microarquitetura (Intel AMX, AVX10.2, ARM SVE2), compressão de modelos (INT8/INT4), kernel real-time (SCHED_DEADLINE, huge pages, eBPF), UX (hot swap com crossfade, IR cabsim, tone matching), portabilidade (Linux ARM64) e observabilidade empírica (differential fuzzing C++↔Rust, PGO/BOLT, HDR histograms).
> Estas inovações são **diferenciadores competitivos** — não corrigem bugs, mas constroem capacidades inéditas no ecossistema NAM em 2026.
> Cada tarefa é atômica, com referências `arquivo:linha` quando aplicável, critérios de aceitação e especialista alvo.
>
> Nota do PO 1: Arquitetura A2 está fora do escopo, ao menos por enquanto. É permitido apenas placeholders e outras medidas para evitar algo que possa se chocar com o A2 mais adiante.
> Nota do PO 2: Sempre assegure ótima cobertura de docsys e comentários rust inline.
> Nota do PO 3: O repositório oficial do NeuralAmpModelerCore está espelhado integralmente em `github.com/NeuralAmpModelerCore/`.
>
> **Legenda de severidade**
>
> - 🔥 Crítico (UB, paridade quebrada, DoS, panics em RT, soundness) — máximo de prioridade.
> - ⚠️ Alto (performance, manutenibilidade, hotpath subótimo, dívida arquitetural).
> - 💡 Médio (otimização, organização, ergonomia, documentação).
> - ✨ Inovação (capacidade nova, diferencial competitivo, UX disruptiva).
>
> **Especialistas alvo** (correspondem a skills disponíveis no projeto)
>
> - `implementador` (engenharia de aplicação Rust idiomático).
> - `revisor-auditor` (este painel, para validação final).
> - `documentador` (atualização de `docs/` e doccomments).
> - `pesquisador-inovador` (fronteira: AMX/AVX10/SVE2, NN compression, RT-OS).

## Notas Operacionais

### Parte I — Remediação

- **Ordem de execução recomendada:** Épico 12 (S21 HDR + diff fuzz primeiro — instrumenta o resto) → 9 (Quantização) → 10 (RT-OS) → 11 (UX) → 13 (Portabilidade & Hardware Especializado).
- **CI/QA gate por Sprint:**
  1. `bash utils/lints.sh` — formatação, clippy strict, feature matrix.
  2. `bash utils/tests-cargo.sh` — unit + integration
  3. `cargo bench inference_bench` — comparar contra baseline; sem regressão > 5%.
- **Convenções:**
  - PR/branch por Tarefa (`feat/S1-T01-bridgeref-soundness`).
  - Commit message inclui referência `[S1.T01]`.
  - Documentação atualizada (skill `documentador`) sempre que arquitetura muda.

### Parte II — Inovações Avant-Garde

- **Ordem de execução recomendada:** Épico 12 (S21 HDR + diff fuzz primeiro — instrumenta o resto) → 9 (Quantização) → 10 (RT-OS) → 11 (UX) → 13 (Portabilidade & Hardware Especializado).
- **Pré-requisitos:**
  - Épicos 1+2 (Parte I) **devem** estar concluídos antes da Parte II — base sólida de soundness e paridade é pré-condição para qualquer otimização agressiva.
  - Hardware de validação: pelo menos uma máquina com **AMX-capable CPU** (Sapphire Rapids, EC2 c7i, ou Granite Rapids). Para ARM: Graviton 4 EC2 ou hardware ARM64 compatível.
  - Kernel PREEMPT_RT 6.x disponível para Épico 10 (S16.T01/T03).
- **Conventions adicionais:**
  - Tasks com tag ✨ requerem documentação em `docs/innovation/<area>.md` com benchmarks empíricos antes do merge.
  - Cada inovação compete com baseline atual em `cargo bench`; merge bloqueado se causar regressão em features default.

---

## Épico 1 — Correções de Soundness / UB / Panics em Hotpath RT [DONE]

Objetivo: eliminar todas as ocorrências de **Undefined Behavior latente** e **panics dentro da audio thread**. Bloqueador de qualquer release estável.

### Sprint S1 — Soundness do `DspBridge` & FFI [DONE]

> **Nota de Auditoria (2026-05-31):** Todas as tarefas foram auditadas. Sanidade das faces de escrita/leitura do `DspBridge` e propagação de erros do `MirroredBuffer` validadas. Alinhamento de warnings e gating dos testes específicos para Linux concluídos com sucesso.

#### Tarefa S1.T01 — Eliminar `&'static mut` em `BridgeRef::as_mut` 🔥 [DONE]

- **Onde:** `src/dsp/pipeline.rs:116-146` (e callers em `src/standalone/pw_host.rs:525, 972` e `src/clap/processor.rs`).
- **Problema:** `BridgeRef::as_mut(self) -> &'static mut DspBridge` cria múltiplas referências mutáveis ao mesmo objeto a partir de threads distintas (capture e playback), violando o **aliasing XOR-mut** do modelo de memória Rust. Apesar de funcional na prática (todos os campos sensíveis são `AtomicU*`), é UB segundo o ref. abstract do `rustc`.
- **Solução técnica:**
  1. Quebrar `DspBridge` em duas faces de tipos disjuntos: `DspBridgeWriter` e `DspBridgeReader`, ambas guardando `*const DspBridge` (ou `NonNull<DspBridge>`).
  2. Apenas o `Writer` exposto à capture thread expõe `write_block(..)`; apenas o `Reader` exposto ao playback expõe `read_block(..)`. Acesso interno via `&*ptr` com lifetime curto, dentro de cada método.
  3. Remover `as_mut`; deixar apenas `unsafe fn as_ptr(self) -> *mut DspBridge`.
  4. Documentar invariantes de `Send`/`Sync` explicitamente em comentários `SAFETY:`.
- **Critérios de aceitação:**
  - Nenhum `&'static mut` no crate (grep).
  - `cargo +nightly miri test --test pw_integration_test` (se houver suporte mock) passa.
  - Smoke test PipeWire (`utils/tests-cargo.sh`) sem regressão.
- **Especialista:** `implementador`.

#### Tarefa S1.T02 — Tornar `MirroredBuffer::new` falível 🔥 [DONE]

- **Onde:** `src/dsp/mirror_buf.rs:62-164` e construtor portado para `Result<Self>`.
- **Problema:** `new()` faz `panic!` em casos legítimos (sandboxes, container, RLIMIT_AS baixo). Inaceitável em plugin CLAP carregado por host arbitrário.
- **Solução técnica:**
  1. Mudar `pub fn new(...) -> Self` para `pub fn new(...) -> std::io::Result<Self>`.
  2. Propagar erro pelo construtor de `WaveNetLayerState` (`src/models/wavenet/common.rs:73`).
  3. Adicionar `checked_mul(2)` em `mirror_buf.rs:96` (overflow protection).
  4. Adicionar `assert!(requested_size > 0)` antes de `mmap` para evitar UB POSIX.
  5. Adicionar `#[cold]` no `Clone` impl (`mirror_buf.rs:209-214`).
- **Critérios de aceitação:**
  - Nenhum `panic!`/`expect`/`unwrap` em `mirror_buf.rs`.
  - Novo teste `tests/mirror_buf_fault_injection.rs` simulando falha de `mmap` (via wrapper opt-in).
- **Especialista:** `implementador`.

#### Tarefa S1.T03 — Renomear `VirtualRingBuffer` para `MirroredBuffer` 🔥 [DONE]

- **Onde:** `src/dsp/mirror_buf.rs` (todo o arquivo), `src/models/wavenet/common.rs` (uso).
- **Problema:** O tipo **não é um ring buffer funcional** — não armazena `read_pos`/`write_pos`. É apenas um alocador de buffer espelhado. Nome induz a erros futuros.
- **Solução técnica:**
  1. Renomear o tipo, módulo (`vring.rs` → `mirror_buf.rs`), e re-export.
  2. Atualizar `src/dsp/mod.rs` e todos os imports.
  3. Atualizar `docs/architecture.md` (seção 2.) e mensagens.
- **Critérios de aceitação:** `cargo check --all-features` verde, sem warnings.
- **Especialista:** `implementador`.

#### Tarefa S1.T04 — Cobertura portátil de `mirror_buf.rs` 💡 [DONE]

- **Onde:** `src/dsp/mirror_buf.rs:62, 199`.
- **Problema:** Usa `memfd_create`, Linux-only. Para manter o crate portátil para outros sistemas operacionais (não-Linux), precisamos de fallback ou cfg-gate.
- **Solução técnica:**
  1. Adicionar `#[cfg(target_os = "linux")] mod linux;` e `#[cfg(not(target_os = "linux"))] mod fallback;`.
  2. Fallback usa `mmap` anônimo + segundo mapeamento com `MAP_FIXED` (ou similar adequado ao SO de destino).
  3. Documentar tradeoffs em `docs/architecture.md`.
- **Critérios de aceitação:** Crate compila com sucesso em alvos não-Linux sem erros ou warnings sobre `memfd_create`.
- **Especialista:** `implementador`.

### Sprint S2 — Panics & FFI seguro [DONE]

> **Nota de Auditoria (2026-05-31):** Todas as tarefas auditadas e validadas. Dois achados adicionais identificados e corrigidos:
>
> 1. **`window.rs` — `on_frame()` panic em FFI callback de baseview:** O `.expect()` no loop de render (chamado a cada frame via C ABI da baseview) foi substituído por early-return silencioso, eliminando possível UB em hosts C++.
> 2. **Whitelist de `.expect()` documentada:** Comentários `WHITELIST:` adicionados em `plugin.rs` e `window.rs` para todos os `.expect()` residuais permitidos, justificando por que são seguros (strings literais ASCII, contexto de inicialização de janela antes de qualquer callback RT/FFI de áudio). Critério de aceitação da S2.T02 totalmente cumprido.

#### Tarefa S2.T01 — Eliminar `panic!` em `process()` sob feature `heap-audit` 🔥 [DONE]

- **Onde:** `src/clap/processor.rs:709-714`.
- **Problema:** Panic atravessando a fronteira FFI do host (clack/clap) é **UB** em hosts que não tratam unwind (Bitwig, FL Studio C++).
- **Solução técnica:**
  1. Substituir `panic!` por: (a) `set_flag(RT_STATUS_HEAP_ALLOC)`, (b) `eprintln!` apenas se `cfg!(debug_assertions)`, (c) retorno `ProcessStatus::Sleep` indicando bypass intencional.
  2. Adicionar drain do flag em `on_main_thread()` para registrar via `HostLog`.
- **Critérios de aceitação:** `cargo test --features heap-audit` ainda detecta regressões mas sem panic. Stress-run via DAW host real (Bitwig + FL) por 5 minutos sem crashes.
- **Especialista:** `implementador`.

#### Tarefa S2.T02 — Remover `.expect()` de `activate()` 🔥 [DONE]

- **Onde:** `src/clap/processor.rs:131-133, 161`; `src/clap/plugin.rs:594`.
- **Problema:** `.expect()` em `activate()` mata o host se houver Mutex poisoning, dupla activate, ou falha rara do resampler. Especialmente perigoso em hosts agressivos (Reaper render).
- **Solução técnica:**
  1. Substituir cada `.expect()` por match com `PluginError::Message(...)` propagando ao host.
  2. Em caso de `Mutex` envenenado, usar `lock().unwrap_or_else(|e| e.into_inner())` (padrão já presente em `ui.rs:1304`).
  3. Auditar todo `.expect()` em paths que atravessam FFI (`grep -rn "\.expect(" src/clap`).
- **Critérios de aceitação:** Nenhum `expect`/`unwrap` em paths de activate/deactivate/process (lista whitelisted documentada).
- **Especialista:** `implementador`.

#### Tarefa S2.T03 — Verificar `alive_fence` antes de `unsafe { &*shared }` em drag-drop 🔥 [DONE]

- **Onde:** `src/clap/gui/window.rs:492-507`; `src/clap/gui/ui.rs:1212-1273`.
- **Problema:** UAF latente. Em `DragDropped`, dereferencia `self.shared.0` sem checar `alive_fence`. Se host destruir o plugin entre eventos, há use-after-free.
- **Solução técnica:**
  1. Encapsular acesso em helper `fn safe_shared(&self) -> Option<&NamClapShared>` que retorna `None` se `alive_fence == false`.
  2. Substituir todos `unsafe { &*self.shared.0 }` por `if let Some(shared) = self.safe_shared()`.
  3. Idem para o file-picker manager thread.
- **Critérios de aceitação:** Stress test fechando janela de modo agressivo durante drag (`utils/tests-cargo.sh test_gui_drag_drop_fuzz`).
- **Especialista:** `implementador`.

#### Tarefa S2.T04 — Encapsular `transmute` para estender lifetime de `HostSharedHandle` 🔥 [DONE]

- **Onde:** `src/clap/gui/gui.rs:97-98`; `src/clap/gui/ui.rs:1212-1214`.
- **Problema:** `transmute::<HostSharedHandle<'a>, HostSharedHandle<'static>>` repetido em dois lugares — chance de divergência. Padrão de ocultação de UB.
- **Solução técnica:**
  1. Criar `pub(crate) unsafe fn extend_host_lifetime<'a>(h: HostSharedHandle<'a>) -> HostSharedHandle<'static>` em `src/clap/gui/mod.rs` com SAFETY comment exaustivo.
  2. Substituir as duas ocorrências por chamada à função.
- **Critérios de aceitação:** Apenas 1 `transmute` no crate (sob essa função); `cargo clippy -- -D clippy::transmute_ptr_to_ptr` passa.
- **Especialista:** `implementador`.

---

## Épico 2 — Paridade Matemática vs `NeuralAmpModelerCore`

Objetivo: corrigir todas as divergências numéricas/lógicas entre nam-rs e a implementação de referência C++, antes que algum modelo de produção (CH≠múltiplo de 4, K>3, geometria multi-layer atípica) exponha o bug.

> Nota do PO: Arquitetura A2 está fora do escopo, ao menos por enquanto. É permitido apenas placeholders e outras medidas para evitar algo que possa se chocar com o A2 mais adiante.
> Nota do PO: O repositório oficial do NeuralAmpModelerCore está espelhado integralmente em `github.com/NeuralAmpModelerCore/`.

### Sprint S3 — Bugs latentes de paridade matemática (estática + dinâmica) [DONE]

> Nota do PO: O repositório oficial do NeuralAmpModelerCore está espelhado integralmente em `github.com/NeuralAmpModelerCore/`.
>
> **Nota de Auditoria (2026-05-31):** Todas as 5 tarefas auditadas e verificadas. Pontos confirmados:
>
> 1. **S3.T01 — BF16 vs F16 scalar:** `process_sample_scalar`, `LstmModel1::process_scalar` e `LstmModel2::process_scalar` corretamente consultam `SimdMathConfig::get().instruction_set` em runtime e usam `f32::from_bits((w as u32) << 16)` para BF16. Proptest `tests/lstm_scalar_bf16_parity.rs` com 10k casos passa.
> 2. **S3.T02 — Unificação quantização LSTM estático:** Helper `quantize_weight` centralizado em `src/math/common/ops.rs` usado consistentemente em `build_lstm_1layer`, `build_lstm_2layer` e `read_lstm_layer(is_bf16)`. Round-trip NAMB v2 Gate-Major passa.
> 3. **S3.T03 — Round-trip GateMajorLstm multi-layer:** Encoder `namb_encoder.rs` agora intercala por camada (weights → bias → hidden_init → cell_init) per-layer antes do head. `tests/namb_v2_roundtrip.rs` valida 7 topologias (MSE < 1e-12).
> 4. **S3.T04 — Tail loop layout-mismatch Conv1D:** Encoder usa `num_blocks = conv_out_ch.div_ceil(4)` com padding zero para canais extras; todos os blocos em formato `[BLK][K][IN][4]` uniforme. Decoder elimina tail-loops separados. Teste `test_conv1d_dyn_padding_non_multiple_of_4` valida `OUT=6`.
> 5. **S3.T05 — Segfault tap_ptrs[8]:** `const MAX_KERNEL: usize = 16` substituiu todos os arrays `[_; 8]` nos 4 variantes de `process_*_frame*`. `debug_assert!(self.kernel <= MAX_KERNEL)` em todos os paths. Validação de carga em `wavenet.rs:425` retorna `Err` se `k > MAX_KERNEL`. Teste `test_conv1d_dyn_large_kernel_no_segfault` confirma.
> 6. **Compilação:** `cargo check --all-features` 100% limpo (zero warnings/errors).
> 7. **Nenhum gap identificado** — todos os critérios de aceitação das 5 tarefas foram cumpridos.

#### Tarefa S3.T01 — Corrigir `process_sample_scalar` LSTM (BF16 vs F16) 🔥 [DONE]

- **Onde:** `src/models/lstm/layer.rs:378-416` (linha 390); replicar em linhas 519 e 626 do mesmo arquivo.

- **Problema:** Pesos são armazenados como `bf16` (bits empacotados em `u16`) quando o dispatcher detecta AVX-512 BF16, mas `process_sample_scalar` interpreta-os como `f16::from_bits(w).to_f32()` — leitura **errada de bits**. Os bits BF16 têm expoente de 8 bits e mantissa de 7; F16 tem expoente de 5 e mantissa de 10. Mesmo bit-pattern, valor numérico totalmente diferente. Os testes de paridade SIMD↔Scalar passam apenas porque os pesos de teste são pequenos (próx. de zero, onde ambos os formatos coincidem perto de 0).

- **Estado atual do código:** `process_sample_scalar` em `layer.rs:378-416` hardcoda `half::f16::from_bits(w).to_f32()` (linha 390) sem consultar o dispatcher. O mesmo padrão errado aparece nas funções `process_scalar` de `LstmModel1` (linha 519) e `LstmModel2` (linha 626). A macro `define_lstm_process!` tem `$is_bf16: expr` como parâmetro (linha 39), mas `process_sample_scalar` é implementado **fora** dela e ignora esse contexto. `SimdMathConfig::get().instruction_set` está disponível — já é usado em `build_lstm_dynamic` (`src/loader/dispatcher/lstm.rs:159-160`).

- **Solução técnica:**

  1. Dentro de `process_sample_scalar` (e nos dois locais de `process_scalar`), detectar o formato em runtime consultando o dispatcher:

     ```rust
     let is_bf16 = crate::math::common::SimdMathConfig::get().instruction_set
         == crate::math::common::InstructionSet::Avx512VnniBf16;
     ```

  2. Substituir `half::f16::from_bits(w).to_f32()` por:

     ```rust
     let w_f32 = if is_bf16 {
         f32::from_bits((w as u32) << 16)  // BF16 → F32 (zero-extend dos 16 bits)
     } else {
         half::f16::from_bits(w).to_f32()
     };
     ```

  3. Aplicar o mesmo fix nos locais de `head_weights` em `LstmModel1::process_scalar` (linha 519) e `LstmModel2::process_scalar` (linha 626).

  4. Adicionar teste explícito: `tests/lstm_scalar_bf16_parity.rs` com pesos não-triviais (ex.: `[-1.5, 0.8, 2.3, ...]`).

- **Critérios de aceitação:** Diferença `|simd - scalar| < 1e-4` em proptest com 10k inputs e pesos `f32::MAX/4..f32::MAX/4`. Modelos de referência passam regression goldens.

- **Especialista:** `implementador` + revisão `revisor-auditor`.

#### Tarefa S3.T02 — Unificar quantização BF16/F16 em LSTM estático 🔥 [DONE]

- **Onde:** `src/loader/dispatcher/lstm.rs:74, 118, 279, 293` (funções `build_lstm_1layer`, `build_lstm_2layer`, `read_lstm_layer`).
- **Problema:** `build_lstm_dynamic` detecta CPU (`is_bf16`, linha 159-160) e quantiza adequadamente usando `f32_to_bf16` (já importado no topo do arquivo). Mas `build_lstm_1layer` (linha 74), `build_lstm_2layer` (linha 118), e `read_lstm_layer` (linhas 279, 293) **sempre usam `half::f16::from_f32(w).to_bits()`** — ignorando o dispatcher dinâmico. Em CPUs Sapphire Rapids+ com kernels BF16-nativos, drift numérico imediato.
- **Estado atual do código:** `build_lstm_dynamic:159-160` detecta `is_bf16` via `SimdMathConfig::get().instruction_set == InstructionSet::Avx512VnniBf16` e usa `f32_to_bf16` (importado em linha 6). `read_lstm_layer` (linhas 266-312) e os builders estáticos usam apenas `half::f16::from_f32(w).to_bits()`.
- **Solução técnica:**
  1. Extrair `fn quantize_weight(f: f32, is_bf16: bool) -> u16` para `src/math/common/mod.rs` (usa `f32_to_bf16` se `is_bf16`, senão `half::f16::from_f32(f).to_bits()`).
  2. Detectar `is_bf16` no início de `build_lstm_1layer` e `build_lstm_2layer` (mesmo padrão de `build_lstm_dynamic:159-160`).
  3. Adicionar `is_bf16: bool` como parâmetro de `read_lstm_layer` e substituir as duas ocorrências de `half::f16::from_f32(w).to_bits()` (linhas 279, 293) pelo helper.
  4. Substituir `head_weights` em `build_lstm_1layer:74` e `build_lstm_2layer:118` pelo helper.
- **Critérios de aceitação:** Em CPU AVX-512 BF16, golden vectors LSTM 1x16 e 2x16 produzem mesma saída de `build_lstm_dynamic`.
- **Especialista:** `implementador`.

#### Tarefa S3.T03 — Corrigir round-trip de layout `GateMajorLstm` para multi-layer 🔥 [DONE]

- **Onde:** `src/loader/namb_encoder.rs:99-157` (encoder) vs `src/loader/dispatcher/lstm.rs:266-312` (decoder `read_lstm_layer`).
- **Problema:** O encoder grava no header `weights_layout = GateMajorLstm` mas serializa pesos **em ordem incorreta** para LSTM 2-layer:
  - **Encoder grava:** `[W1_transposed, bias1, W2_transposed, bias2, hidden_init1, cell_init1, hidden_init2, cell_init2, head_weights, head_bias]`.
  - **Decoder espera (via `read_lstm_layer` em sequência por camada):** `[W1, bias1, hidden_init1, cell_init1, W2, bias2, hidden_init2, cell_init2, head_weights, head_bias]`.
  - Para 2-layer, isso embaralha `bias2` com `hidden_init1`.
- **Estado atual do código:** `read_lstm_layer` lê na ordem `weights (H4×IH) → bias (H4) → hidden_init (H) → cell_init (H)` (linhas 271-309). Verificar o encoder (`namb_encoder.rs:99-157`) para confirmar a divergência na ordenação de camadas.
- **Solução técnica:**
  1. No encoder, intercalar por camada: após escrever `W_l, bias_l`, escrever `hidden_init_l, cell_init_l` imediatamente.
  2. Substituir o "resto" final (`namb_encoder.rs:152-154`) pela escrita explícita de `head_weights, head_bias`.
  3. Adicionar teste `tests/namb_v2_roundtrip.rs` cobrindo todas as topologias LSTM (1×{8,12,16,24}, 2×{8,12,16}), assertando que decode(encode(x)) == x para todos os pesos.
- **Critérios de aceitação:** Teste round-trip passa para 7 topologias; modelo `tests/fixtures/models/*-2x16.nam` produz mesma saída via JSON e NAMB v2.
- **Especialista:** `implementador`.

#### Tarefa S3.T04 — Corrigir tail loop layout-mismatch em Conv1D (encoder padding interleaved) 🔥 [DONE]

- **Onde:** Encoder `src/loader/namb_encoder.rs:213-232` (Interleaved-4 transpose); decoder `src/models/wavenet/conv1d_dyn.rs` — tail loops nas linhas 229-253 (`process_dual_frame`) e 445-461 (`process_single_frame`), e equivalentes BF16 nas linhas 641-665 (`process_dual_frame_bf16`) e ~838-860 (`process_single_frame_bf16`).
- **Problema:** O loop SIMD escreve em **`[OUT_BLK][K][IN][4]` interleaved** mas os tail loops (canais restantes `OUT % 4 != 0`) lêem pesos do offset `(out_c * self.kernel + k) * self.in_ch`, assumindo layout `[OUT][K][IN]`. Layouts incompatíveis ⇒ leitura de **bytes errados** ⇒ ruído na saída. Hoje os catálogos (`CH ∈ {4,8,12,16}`) garantem `OUT % 4 == 0` e o bug está latente, mas geometrias futuras (de comunidade) acionariam o defeito. O `conv1d.rs` estático usa `const CH` sempre múltiplo de 4 no catálogo atual — o problema é principalmente no path dinâmico (`conv1d_dyn.rs`).
- **Solução técnica (única):** Padding implícito no encoder para sempre gerar layout `[CEIL(OUT/4)][K][IN][4]`.
  1. No `transpose_wavenet_interleaved4` (`namb_encoder.rs:213-232`), trocar o "tail-loop separado" por padding de até 3 canais zero, escrevendo **todo** o bloco no formato interleaved-4 uniformemente.
  2. Remover os 4 tail-loops com layout não-interleaved em `conv1d_dyn.rs`. O decoder passa a ler sempre via `dot_product_4x_interleaved`.
  3. Adicionar `assert!(self.out_ch <= num_blocks * 4)` nos construtores de `Conv1dDyn` (com `num_blocks = (out_ch + 3) / 4`).
  4. Adicionar teste com `OUT = 6` (não múltiplo de 4) sintético; o encoder gera `2 * 4 = 8` slots interleaved, os 2 últimos preenchidos com zero, e a saída ignora os canais excedentes via `out_ch` lógico.
  5. **Compatibilidade retroativa:** modelos NAMB v2 produzidos antes do fix permanecem corretos (afetam só geometrias `OUT % 4 != 0` que não existem no catálogo atual). Documentar como bump implícito em NAMB v2 (sem mudança de versão).
- **Critérios de aceitação:**
  - Modelo sintético `<6,3,...>` produz saída idêntica via path estático e referência escalar (`ScalarRefMath::dot_product`).
  - Round-trip encode→decode preserva o `out_ch` lógico mesmo com padding zero.
- **Especialista:** `implementador` + `pesquisador-inovador`.

#### Tarefa S3.T05 — Eliminar segfault potencial em `tap_ptrs[8]` (Conv1D dyn com kernel>8) 🔥 [DONE]

- **Onde:** `src/models/wavenet/conv1d_dyn.rs` — `process_dual_frame` linhas 65-93 (populate de `tap_ptrs_f0[8]`) e 184 (loop de convolução); `process_single_frame` linhas 349-366 (populate de `tap_ptrs[8]`) e 417 (loop); variantes BF16 em `process_dual_frame_bf16:487-506` e `process_single_frame_bf16:757-774`.
- **Problema:** `let mut tap_ptrs_f0 = [core::ptr::null::<f32>(); 8];` é populado apenas até `k_limit = self.kernel.min(8)` (linha 67), mas os loops de convolução usam `tap_ptrs_f0.get_unchecked(k)` iterando `for k in 0..kernel` (linha 184) — desreferencia ponteiro nulo se `kernel > 8`. O mesmo padrão ocorre nas 4 variantes de `process_*_frame*`. O limite de 8 é um hardcode que não reflete o `kernel` real do modelo carregado.
- **Solução técnica:**
  1. Definir `const MAX_KERNEL: usize = 16;` (ou valor adequado) no topo de `conv1d_dyn.rs`.
  2. Substituir todos os arrays `[_; 8]` por `[_; MAX_KERNEL]` nos 4 locais.
  3. Adicionar no início de cada `process_*_frame*`: `debug_assert!(self.kernel <= MAX_KERNEL, "kernel {} excede MAX_KERNEL", self.kernel);`.
  4. No construtor ou ao carregar o modelo, validar `kernel <= MAX_KERNEL` e retornar `Err` se violado.
- **Critérios de aceitação:** Teste com modelo sintético `kernel=10` carrega sem segfault, produz saída numérica correta vs referência escalar.
- **Especialista:** `implementador`.

### Sprint S4 — Backfill safety & A2 placeholder

> Nota do PO: O repositório oficial do NeuralAmpModelerCore está espelhado integralmente em `github.com/NeuralAmpModelerCore/`.
> Nota do PO: Arquitetura A2 está fora do escopo, ao menos por enquanto. É permitido apenas placeholders e outras medidas para evitar algo que possa se chocar com o A2 mais adiante.

#### Tarefa S4.T01 — Prevenir underflow `usize` no backfill de prewarm 🔥

- **Onde:** `src/models/wavenet/model.rs:261` (linha `(current_state.buffer_start - offset) * CH`); `src/models/wavenet/model_dyn.rs` (padrão equivalente no `prewarm_internal`).

- **Problema:** `(current_state.buffer_start - offset) * CH` faz **subtração não-checada** em `usize`. Se a invariante `buffer_start >= offset` for quebrada por rewinds exóticos ou geometrias com RF grandes, underflow → endereço gigante → OOB.

- **Estado atual do código:** O backfill está em `model.rs:260-271` dentro de `process_block_internal::<M, true>` (modo PREWARM). O código executa `copy_within(src_range.clone(), dst_idx)` onde `dst_idx = (current_state.buffer_start - offset) * CH` sem checagem. O `buffer_start` é inicializado via `WaveNetLayerState` em `common.rs`.

- **Solução técnica:**

  1. Adicionar `debug_assert!(current_state.buffer_start >= offset, "backfill underflow: bs={}, off={}", current_state.buffer_start, offset);` antes do `copy_within`.

  2. Em release, usar saturating_sub com log de erro:

     ```rust
     let dst_start = current_state.buffer_start.checked_sub(offset)
         .unwrap_or_else(|| { log::error!("backfill underflow"); return; }) * CH;
     ```

  3. No construtor `WaveNetLayerState::new` (`src/models/wavenet/common.rs`), validar que `buffer_start >= receptive_field_size` na inicialização.

  4. Adicionar teste `tests/wavenet_prewarm_edge.rs` com RF=2048 (modelo customizado).

- **Critérios de aceitação:** Stress test com modelos de RF variável (incluindo `WaveNetStandard` com dilatação máxima) executa prewarm sem warning.

- **Especialista:** `implementador`.

#### Tarefa S4.T02 — Substituir stack buffers hardcoded `[f32; 1024]` 🔥

- **Onde:** `src/models/wavenet/model.rs:51, 64` (dois buffers `[0.0f32; 1024]` em `process_block_internal`); `src/models/wavenet/model_dyn.rs` (padrão equivalente).
- **Problema:** Buffers de stack fixos em 1024 podem ser excedidos por geometrias com `CH>16` ou `WAVENET_MAX_NUM_FRAMES>64` sem aviso em release. Atualmente há `debug_assert!(num_frames * CH <= 1024)` em `model.rs:46-50` que protege apenas em debug — em release o overflow silencioso corrompe o stack.
- **Estado atual do código:** Em `model.rs`, `process_block_internal` usa dois arrays `[0.0f32; 1024]` na stack (linhas 51 e 64). O `debug_assert!` em linhas 46-50 valida `num_frames * CH <= 1024`. O `const WAVENET_MAX_NUM_FRAMES = 64` está definido em `conv1d_dyn.rs:16`.
- **Solução técnica:**
  1. Para o caminho estático, trocar para `[f32; CH * WAVENET_MAX_NUM_FRAMES]` usando const generics (estável em Rust ≥1.79). Isso tornará o `debug_assert` desnecessário (erro de compilação em topologias inválidas).
  2. Elevar o `debug_assert` atual para um `const_assert!(CH * WAVENET_MAX_NUM_FRAMES <= 1024)` em compile-time.
  3. Para o caminho dinâmico (`model_dyn.rs`), usar `assert!(num_frames * ch <= MAX_STACK)` com `const MAX_STACK: usize = 8192` ou pré-alocar via `AlignedVec` em `WaveNetDynModel::new`.
- **Critérios de aceitação:** Compilação falha (com erro útil) ao tentar topologia maior; testes passam no painel de 4 topologias atual.
- **Especialista:** `implementador`.

#### Tarefa S4.T03 — Sinalizar `WavenetA2Placeholder` no UI 🔥

- **Onde:** `src/models/a2/mod.rs:37-49`; `src/clap/gui/ui.rs` (status bar).
- **Problema:** Modelos A2 carregados produzem **silêncio absoluto** sem feedback adequado ao usuário (apenas um `log::warn!` único).
- **Solução técnica:**
  1. Adicionar `RT_STATUS_A2_PLACEHOLDER` em `src/common/spsc.rs` (seguindo o padrão dos outros flags RT já existentes).
  2. Em `WavenetA2Placeholder::process` setar o flag atomicamente (uma vez por modelo carregado, não a cada buffer).
  3. Em `src/clap/gui/ui.rs` (status bar), exibir mensagem "Modelo A2 não suportado — bypass ativo" quando o flag estiver ativo.
  4. Em standalone, log INFO uma única vez por carregamento.
- **Critérios de aceitação:** Carregar modelo A2 exibe mensagem clara no UI; bypass sonoro permanece ativo.
- **Especialista:** `implementador`.

#### Tarefa S4.T04 — Adicionar `reset(sr, max_buf)` no trait `NamModel` ⚠️

- **Onde:** `src/models/mod.rs:18` (trait `NamModel` — atualmente tem apenas `process` e `prewarm`).
- **Problema:** A referência C++ `NAM/dsp.cpp::Reset(sr, max_buf)` deve ser chamada antes de `process`. NAM-rs não tem equivalente: `prewarm(2048)` é hardcoded no loader. O trait `NamModel` em `mod.rs:18` define apenas `process` e `prewarm`.
- **Estado atual do código:** `NamModel` em `mod.rs:18-24` tem `fn process` e `fn prewarm`. O `DynamicModel::prewarm` na linha 81 recebe `num_samples` mas WaveNet variants ignoram esse parâmetro (chamam `m.prewarm()` sem argumento — linhas 83-88).
- **Solução técnica:**
  1. Adicionar `fn reset(&mut self, sample_rate: u32, max_buffer_size: usize)` ao trait `NamModel` com implementação default que chama `self.prewarm(max_buffer_size)`.
  2. WaveNet: implementação default é adequada (prewarm com silêncio).
  3. LSTM: pode override para resetar apenas o input slot (`state[0..I] = 0.0`) sem reprocessar prewarm completo.
  4. Chamar no `loader/mod.rs` antes do primeiro `process`.
- **Critérios de aceitação:** Conformidade documentada em `docs/architecture.md`. Goldens não regridem.
- **Especialista:** `documentador` + `implementador`.

#### Tarefa S4.T05 — Manter estados iniciais carregados em LSTM dyn prewarm ⚠️

- **Onde:** `src/models/lstm/model_dyn.rs` — função `prewarm` ou `prewarm_internal` (verificar onde ocorre o reset de `state`/`cell_state`).
- **Problema:** `prewarm_internal` zera `state` e `cell_state` ANTES de processar o silêncio, descartando os valores `_xh` e `_c` carregados do arquivo NAM (que foram preservados em `build_lstm_dynamic:204-209` usando `copy_from_slice`). Comportamento divergente da referência C++.
- **Estado atual do código:** Em `build_lstm_dynamic` (linhas 203-209), `state` e `cell_state` são inicializados com os valores do arquivo. Verificar se `prewarm` reseta esses valores antes de processar o silêncio.
- **Solução técnica:**
  1. No prewarm, apenas zerar o input slot: `state[0..input_size] = 0.0`, preservar `state[input_size..]`.
  2. `cell_state` permanece inalterado (valores carregados do arquivo).
  3. Processar as amostras de silêncio normalmente a partir desse estado.
- **Critérios de aceitação:** Goldens LSTM 1×16 batem com saída C++ de referência.
- **Especialista:** `implementador`.

---

## Épico 3 — Segurança do Loader (NAMB/JSON) e Formato

Objetivo: blindar parsers contra inputs adversários, corrigir categorização de erros e documentar o formato `.namb` formalmente.

### Sprint S5 — Loader hardening

#### Tarefa S5.T01 — Validar tamanho do arquivo antes de `std::fs::read` 🔥

- **Onde:** `src/loader/mod.rs:70, 94`.
- **Problema:** `std::fs::read` carrega o arquivo inteiro em RAM. Sem cap de tamanho, um `.namb` adversário de 4 GB consome 4 GB de memória.
- **Solução técnica:**
  1. Adicionar `const MAX_MODEL_BYTES: u64 = 256 * 1024 * 1024;` (256 MiB).
  2. Antes do `std::fs::read`: `let len = std::fs::metadata(path)?.len(); if len > MAX_MODEL_BYTES { bail!(...); }`.
  3. Emit erro com diagnóstico tipado (`NamErrorCode::ModelTooLarge`).
- **Critérios de aceitação:** Tentar carregar arquivo 300 MiB rejeita com mensagem clara em < 50ms.
- **Especialista:** `implementador`.

#### Tarefa S5.T02 — Substituir categorização de erro por substring por `thiserror` 🔥

- **Onde:** `src/loader/mod.rs:79-92`; `src/loader/namb.rs:68, 73, 98, 109, 113, 142`.
- **Problema:** O matcher de erro busca substrings em português (`"muito pequeno"`, `"mágica inválida"`) mas as mensagens estão em **inglês** → `NambTruncated`, `NambInvalidMagic` etc. **nunca disparam**. Todo erro vira `ModelBuildFailed`.
- **Solução técnica:**
  1. Criar `pub enum NambError { Truncated { got, need }, InvalidMagic(u32), InvalidVersion(u32), CrcMismatch { got, expected }, ... }` com `thiserror::Error`.
  2. Substituir `anyhow::bail!` por `Err(NambError::*)` em `namb.rs`.
  3. Em `mod.rs:79`, fazer `match err.downcast_ref::<NambError>()` para categorização tipada.
- **Critérios de aceitação:** Cada variante de erro maps para `NamErrorCode` correto; teste `test_error_codes_*` cobre todas.
- **Especialista:** `implementador`.

#### Tarefa S5.T03 — Tornar CRC32 obrigatório em NAMB v2 (via flag explícito) 🔥

- **Onde:** `src/loader/namb.rs:28-57` (header struct), `:137-147` (verificação).
- **Problema:** `if crc32_header != 0` usa `crc=0` como sentinel "CRC ausente", mas arquivos legítimos podem produzir `crc32 == 0` por coincidência (~1/2³²) ou por escolha adversarial de pesos. Em v1 é tolerável (compatibilidade); em v2+ a obrigatoriedade deve usar **flag explícito** em vez de sentinel.
- **Solução técnica:**
  1. Reservar 1 byte dos `reserved_v2: [u8; 5]` no header como `flags: u8` com bit `FLAG_HAS_CRC32 = 0x01`.
  2. Encoder NAMB v2 sempre seta o flag e grava o CRC verdadeiro.
  3. Decoder: em v2+ exigir `flags & FLAG_HAS_CRC32 != 0` (rejeitar se ausente); validar CRC mesmo se `crc==0` (caso legítimo de coincidência).
  4. Em v1, manter comportamento atual de `crc==0 ⇒ skip` mas emitir `log::warn!("CRC32 missing in NAMB v1 file ...")` (não-bloqueante).
  5. Documentar a evolução do header em `docs/namb-spec.md` (ver S5.T07).
- **Critérios de aceitação:**
  - Teste v2 sem flag `FLAG_HAS_CRC32` falha com erro tipado `NambError::CrcMissing`.
  - Teste v2 com `crc==0` legítimo (flag setado, CRC realmente 0) **passa**.
  - Teste v1 com `crc==0` continua a passar emitindo warning.
- **Especialista:** `implementador`.

#### Tarefa S5.T04 — Cap de tamanho em `Vec<f32> weights` do JSON (preservando forward-compat) ⚠️

- **Onde:** `src/loader/nam_json.rs:46, 117, 176-179`.
- **Problema:** `serde_json::from_str` aloca `Vec<f32>` ilimitado. JSON com 100M floats consome 400 MB instantaneamente. Adicionalmente `metadata.training: Option<serde_json::Value>` aceita árvore JSON ilimitada.
- **Solução técnica (forward-compat-safe):**
  1. Custom deserializer `weights: Vec<f32>` que aborta após `MAX_WEIGHTS = MAX_MODEL_BYTES / 4` floats com erro `JsonError::WeightsExceedLimit`.
  2. Para `metadata.training`, **não** usar `#[serde(deny_unknown_fields)]` (quebraria forward-compat com upstream que adiciona campos novos). Em vez disso:
     - Visitor custom que limita profundidade da árvore a 16 e tamanho total a `MAX_TRAINING_BYTES = 1 MiB`.
     - Campos desconhecidos no nível raiz de `NamMetadata`/`NamConfig` permanecem **ignorados silenciosamente** (compatibilidade) — apenas o tamanho global do JSON é capado em parsing.
  3. O cap total do JSON é derivado de `MAX_MODEL_BYTES` (S5.T01); JSON inválido por tamanho falha cedo e com erro tipado.
- **Critérios de aceitação:**
  - JSON de 200 MiB rejeitado em < 100ms.
  - JSON com campo desconhecido em `metadata` (ex.: `"creator_email": "..."`) **carrega normalmente** (forward-compat OK).
  - JSON com `training: {"a": {"b": {... aninhado 20 níveis ...}}}` rejeitado.
- **Especialista:** `implementador`.

#### Tarefa S5.T05 — Detecção A2 SemVer-aware ⚠️

- **Onde:** `src/loader/nam_json.rs:133-155`.
- **Problema:** Detecção via `starts_with("0.6")` quebra para versões 0.9+ ou 1.0.
- **Solução técnica:**
  1. Adicionar parser SemVer mínimo (sem dep nova: split por `.` + parse u16) e comparar `version >= (0, 6, 0)`.
  2. Manter critério de activation != Tanh como alternativa.
  3. Adicionar teste para versões `0.9`, `1.0`, `0.10`, `2.0`.
- **Critérios de aceitação:** Todas as versões futuras detectadas; teste exaustivo passa.
- **Especialista:** `implementador`.

#### Tarefa S5.T06 — Refatorar duplicação `*_dyn` em dispatcher WaveNet/LSTM ⚠️

- **Onde:** `src/loader/dispatcher/wavenet.rs:341-378 vs :414-465`, `:467-510 vs :503-540`.
- **Problema:** Funções `read_conv1d_weights` e `read_conv1d_weights_dyn` duplicam ~100 LoC. Same para `read_dense_layer`.
- **Solução técnica:**
  1. Extrair `fn read_conv1d_weights_typed<T: ConvOutput>(...)` que aceita o tipo de buffer.
  2. Implementar `ConvOutput` para `[u16]` (estático) e `AlignedVec<u16>` (dinâmico).
- **Critérios de aceitação:** ~200 LoC removidas; testes passam.
- **Especialista:** `implementador`.

#### Tarefa S5.T07 — Documentar formato NAMB em `docs/namb-spec.md` ⚠️ (entrega única — referenciada por S14.T01)

- **Onde:** Criar `docs/namb-spec.md`.
- **Problema:** Especificação do formato não está documentada formalmente. Comentários in-line são insuficientes para implementadores externos.
- **Solução técnica:**
  1. Estrutura: header (offsets, semântica), CRC32 (qual range cobre, flag `FLAG_HAS_CRC32` — ver S5.T03), magic, versionamento, layouts (Original, GateMajorLstm, Interleaved4WaveNet).
  2. Exemplos hex de cada layout (incluindo padding-zero do Interleaved-4 — ver S3.T04).
  3. Política de evolução (compatibilidade retroativa, novos flags reservados).
  4. Referenciar contrato de erros tipados (`NambError`).
- **Critérios de aceitação:** Doc revisto pela skill `documentador`; cobre todas as decisões tomadas em S3.T03/S3.T04/S5.T02/S5.T03.
- **Especialista:** `documentador`.
- **Nota:** S14.T01 abaixo é **apenas referência cruzada** — esta é a entrega real.

#### Tarefa S5.T08 — Fuzzing dos parsers NAMB/JSON 💡

- **Onde:** `tests/fuzz/` (novo).
- **Problema:** Sem fuzzing sistemático além do proptest existente.
- **Solução técnica:**
  1. Adicionar `cargo-fuzz` configuration em `fuzz/`.
  2. Targets: `fuzz_parse_namb`, `fuzz_parse_nam_json`.
  3. Roda em CI (24h em GitHub Actions ou local).
- **Critérios de aceitação:** Nenhum panic em 1M iterações.
- **Especialista:** `implementador`.

#### Tarefa S5.T09 — Rejeitar magic alternativo `0x424D414E` ou implementar byte-swap 💡

- **Onde:** `src/loader/namb.rs:64-70`.
- **Problema:** Aceitar magic alternativo "BMAN" sem byte-swap leva à leitura errada de `weights_offset` (u32 LE).
- **Solução técnica:**
  1. Se quirk não documentado, **remover** aceitação do magic alternativo.
  2. Se for variante BE legítima, ativar modo BE-swap (todos os u32/u16 lidos com `from_be_bytes`).
- **Critérios de aceitação:** Decisão documentada; teste para cada cenário.
- **Especialista:** `implementador` + `documentador`.

---

## Épico 4 — Otimização Hotpath (SIMD/ILP/cache/branchless)

Objetivo: arrancar 5–30% adicional de throughput sem comprometer correção, reduzindo divisões, branches e cadeias de dependência longas.

### Sprint S6 — Telemetria & Gain hotpath

#### Tarefa S6.T01 — Trocar `fetch_update` por `fetch_add` em telemetria ⚠️

- **Onde:** `src/dsp/telemetry.rs:40, 52`.
- **Problema:** `fetch_update` é um CAS-loop. Para um contador monotônico, `fetch_add` é um único `lock xadd` no x86 (3× mais rápido).
- **Solução técnica:**
  1. Substituir por `self.bins[index].fetch_add(1, Ordering::Relaxed)`.
  2. Considerar `AtomicU64` para evitar overflow em runs > 2.8 anos.
- **Critérios de aceitação:** Benchmark `bench_record` reduz em ≥40%.
- **Especialista:** `implementador`.

#### Tarefa S6.T02 — Pré-calcular `inv_fade_frames` no Gate ⚠️

- **Onde:** `src/dsp/gate.rs:158, 176, 204, 213, 216, 270, 324`.
- **Problema:** Divisão `f32` (`fade_counter / fade_frames`) repetida no hotpath, ~3× mais cara que multiplicação.
- **Solução técnica:**
  1. Em `GateParams::new`, calcular `inv_fade_frames: f32 = 1.0 / fade_frames as f32`.
  2. Substituir todas as divisões por `fade_counter as f32 * params.inv_fade_frames`.
- **Critérios de aceitação:** Benchmark de fade-in/fade-out melhora ≥10%.
- **Especialista:** `implementador`.

#### Tarefa S6.T03 — Eliminar duplicação `src/dsp/gain.rs` ↔ `src/math/dsp/gain.rs` 🔥

- **Onde:** `src/dsp/gain.rs` (todo o arquivo) vs `src/math/dsp/gain.rs:58-71`.
- **Problema:** Duas implementações independentes de `apply_gain_simd`. A versão em `src/dsp/gain.rs:15` chama diretamente `apply_gain_avx2` **sem checagem de feature** — UB em CPUs sem AVX2. A versão em `math/dsp/gain.rs:20` usa `dispatch_simd!`.
- **Solução técnica:**
  1. Deletar `src/dsp/gain.rs` (deletar `src/dsp/gain_test.rs`).
  2. Em `src/dsp/mod.rs`, remover `pub mod gain;`.
  3. Atualizar callers: `src/dsp/gate.rs:241, 253, 260, 268, 271, 278`; `src/dsp/pipeline.rs:276` → usar `crate::math::dsp::gain::apply_gain(...)` e `apply_ramp_stereo`.
  4. Migrar testes em `src/dsp/gain_test.rs` para `src/math/dsp/gain_test.rs` (se ausentes lá).
- **Critérios de aceitação:** Crate compila sem `src/dsp/gain.rs`; todos os testes passam.
- **Especialista:** `implementador`.

#### Tarefa S6.T04 — Pré-calcular `gate_threshold_linear_sq` no CLAP ⚠️

- **Onde:** `src/clap/processor.rs:565`.
- **Problema:** `lut.db_to_linear(modulated_gate_db).powi(2)` recalculado a cada `process()` mesmo sem mudança.
- **Solução técnica:**
  1. Cachear o resultado em `&mut self.cached_threshold_sq: f32`.
  2. Invalidar somente se param `gate_threshold_db` ou modulação mudou (flag `gate_dirty: AtomicBool`).
- **Critérios de aceitação:** Latência média de `process()` reduz ≥1µs em modelos pequenos.
- **Especialista:** `implementador`.

#### Tarefa S6.T05 — Decimar telemetria do CLAP processor ⚠️

- **Onde:** `src/clap/processor.rs:691-701`.
- **Problema:** `latency_hist.record()` chamado a cada `process()`. Em hosts com buffer pequeno (32 spl @ 96k = ~333µs), telemetria é dominante.
- **Solução técnica:**
  1. Adicionar `cycles_since_telemetry: u32` em `NamClapProcessor`.
  2. Decimar 1-em-16 (igual ao standalone — `pw_host.rs:962`).
- **Critérios de aceitação:** Overhead de telemetria fica abaixo de 1% nas medidas.
- **Especialista:** `implementador`.

### Sprint S7 — Hotpath de pipeline e resampler

#### Tarefa S7.T01 — Eliminar input-resample duplicado em modo mono ⚠️

- **Onde:** `src/dsp/pipeline.rs:339-348` (e callers do resampler).
- **Problema:** Em `process_mono`, o resampler executa **duas convoluções idênticas** em `state_l` e `state_r`. 50% de trabalho jogado fora.
- **Solução técnica:**
  1. Adicionar `pub fn process_input_mono(&mut self, in_l: &[f32], out_l: &mut [f32], out_r: &mut [f32])` em `src/dsp/resampler.rs`.
  2. Internamente, opera só em `state_l` e duplica o resultado em `out_l` e `out_r`.
  3. Idem para `process_output_mono`.
  4. Em `pipeline.rs`, escolher entre mono/stereo no caller.
- **Critérios de aceitação:** Em modo mono, latência por bloco reduz em ≥30% vs estéreo.
- **Especialista:** `implementador`.

#### Tarefa S7.T02 — Bounds-elision em `DelayLine::push` e `process_internal` ⚠️

- **Onde:** `src/dsp/resampler.rs:65-72, 136-137`.
- **Problema:** Indexação `self.buf[pos]` com bounds check no hotpath (provavelmente eliminado pelo LLVM, mas não garantido).
- **Solução técnica:**
  1. Substituir por `*self.buf.get_unchecked_mut(pos)` com `debug_assert!(pos < TAPS_PER_PHASE)`.
  2. Mesmo padrão em `in_l[in_idx]` → `*in_l.get_unchecked(in_idx)`.
- **Critérios de aceitação:** Assembly produzido por `cargo asm` confirma ausência de jmp para `panic_bounds_check`.
- **Especialista:** `implementador` + `pesquisador-inovador`.

#### Tarefa S7.T03 — `convolve_stereo_dual` para reutilizar loads em resampler ⚠️

- **Onde:** `src/dsp/resampler.rs:130-180`.
- **Problema:** Cada saída executa **2× `convolve_stereo`** (taps das fases φ_idx e φ_next). Os loads de `x_l/x_r` poderiam ser compartilhados.
- **Solução técnica:**
  1. Adicionar `fn convolve_stereo_dual(c0: &[f32], c1: &[f32], x_l: &[f32], x_r: &[f32]) -> [(f32,f32); 2]` em `src/dsp/sinc_kernel.rs`.
  2. Implementação SIMD: load `x_l/x_r` uma vez por tap, multiply com `c0` e `c1` em paralelo.
  3. Atualizar caller em `resampler.rs:160-168`.
- **Critérios de aceitação:** Throughput de resampling ≥15% maior.
- **Especialista:** `pesquisador-inovador`.

#### Tarefa S7.T04 — Refatorar duplicação massiva em `pipeline.rs` mono/stereo paths ⚠️

- **Onde:** `src/dsp/pipeline.rs:317-321, 339-348, 372-377, 356-365` (4 blocos quase idênticos).
- **Problema:** Padrão `if let Some(model_l) ... else copy_from_slice` duplicado 4× com pequena variação de mono/stereo.
- **Solução técnica:**
  1. Extrair `#[inline(always)] fn run_stereo_or_mono(...)`.
  2. Recomposição de bypass/stereo via closure ou helper.
- **Critérios de aceitação:** Funções com ≤50 LoC; redução ≥30 LoC totais.
- **Especialista:** `implementador`.

#### Tarefa S7.T05 — Resolver pressão de registradores em `dot_4x.rs` para AVX-512 ⚠️

- **Onde:** `src/math/gemm/dot_4x.rs:466-481` (kernels não implementados ou suboptimais).
- **Problema:** Auditoria SIMD identificou potencial 8-16× speedup ainda não capturado em AVX-512.
- **Solução técnica:**
  1. Reescrever loops para usar 8 acumuladores ZMM (16 lanes f32 cada).
  2. Quebrar cadeias de dependência via 4 acumuladores independentes em FMA pipeline.
  3. Software prefetch a 4 cache lines à frente.
- **Critérios de aceitação:** Benchmark `bench_dot_4x_avx512` melhora ≥4× vs versão atual.
- **Especialista:** `pesquisador-inovador`.

#### Tarefa S7.T06 — Aumentar paralelismo em `gemv.rs` (4–8 acumuladores) ⚠️

- **Onde:** `src/math/gemm/gemv.rs`.
- **Problema:** GEMV com 1 acumulador atinge ~12-25% do peak FMA. Cadeia de dependência limita throughput.
- **Solução técnica:**
  1. 4 acumuladores em AVX2 (4×8 = 32 lanes), 8 em AVX-512 (8×16 = 128 lanes).
  2. Reduzir loop de fora para minimizar pressure no register file.
- **Critérios de aceitação:** GEMV achieves ≥70% peak FMA em AVX2/AVX-512.
- **Especialista:** `pesquisador-inovador`.

#### Tarefa S7.T07 — Corrigir `gemv_4gate.rs` BF16 (paridade numérica) 🔥

- **Onde:** `src/math/gemm/gemv_4gate.rs:281-322`.
- **Problema:** Auditoria SIMD identificou que o kernel BF16 4-gate **produz áudio errado** (drift severo vs `Avx2Math` em LSTM 1×16).
- **Solução técnica:**
  1. Investigar a cadeia de conversão BF16 → F32 → FMA: verificar uso de `_mm512_dpbf16_ps` vs conversão manual.
  2. Comparar com goldens C++ de modelo de referência.
  3. Adicionar teste cross-implementação em `tests/lstm_gate_bf16_parity.rs`.
- **Critérios de aceitação:** Diferença vs F32 < 1e-3 em proptest com 10k inputs.
- **Especialista:** `pesquisador-inovador` + `revisor-auditor`.

#### Tarefa S7.T08 — Corrigir bugs SIMD identificados em `dot.rs` e `ops.rs` 🔥

- **Onde:** `src/math/gemm/dot.rs:144-147`; `src/math/common/ops.rs:38`; `src/math/common/avx512_impl.rs:780`.
- **Problema:** Auditoria identificou falhas de correção numérica nesses pontos (ordering de reduções, broadcast errado, conversões F16 indevidas).
- **Solução técnica:**
  1. Investigar caso a caso; adicionar testes mínimos reproduzindo a divergência.
  2. Corrigir e adicionar regression test.
- **Critérios de aceitação:** Diferença vs `ScalarRefMath` < 1e-5 em 10k inputs.
- **Especialista:** `pesquisador-inovador`.

#### Tarefa S7.T09 — Substituir activations LUT por polinômios Padé branchless ✨⚠️

- **Onde:** `src/math/activations/tanh.rs:15-200`, `sigmoid.rs:18-280`.
- **Problema/Oportunidade:** Aproximações atuais usam polinômio grau-7 iterado (~15 FMAs por activation). Padé approximant **(p/q polinomial ratio)** atinge precisão equivalente em ~7 FMAs com **zero branches**. Para LSTM 1×16 com 4 activations por amostra a 48 kHz, ganho cumulativo significativo. Referência: VDT library (CERN), Mineiro & Vorlicek (2016).
- **Solução técnica:**
  1. Para `tanh(x)`: usar identidade `tanh(x) = x · (27 + x²) / (27 + 9x²)` em região `|x| < 4`, saturar em ±1 fora. AVX2/AVX-512 + AMX/AVX10.2-ready.
  2. Para `sigmoid(x)`: `sigmoid(x) = 0.5 + 0.5 · tanh(x/2)` reutiliza kernel.
  3. Manter trait existente; substituir corpo de `simd_tanh_*` e `simd_sigmoid_*` (transparente para callers).
  4. Validar via proptest com 100k inputs distribuídos em `-10..10`, exigir `|err| < 2e-5` (compatível com FP16 precision).
- **Critérios de aceitação:**
  - Benchmark `bench_tanh_slice` ≥ 30% mais rápido.
  - `cargo test test_tanh_scalar_equivalences` passa com tolerância 2e-5.
  - Especialização AMX/AVX10 documentada.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

---

## Épico 5 — Refatoração Arquitetural

Objetivo: trazer todos os arquivos > 500 LoC para conformidade, melhorar coesão e reduzir custo de manutenção.

### Sprint S8 — Refatoração da GUI (CLAP)

> Nota do PO: Sempre assegure ótima cobertura de docsys e comentários rust inline.

#### Tarefa S8.T01 — Quebrar `src/clap/gui/ui.rs` (2004 LoC) em módulos 🔥

- **Onde:** `src/clap/gui/ui.rs` → `src/clap/gui/ui/`.
- **Problema:** Monolito de 2004 LoC; função `draw_ui` com 930 linhas. Inviável para revisão e manutenção.
- **Solução técnica:** dividir em 9 arquivos (~150-400 LoC cada):
  - `ui/mod.rs` — `pub fn draw_ui` orquestrador.
  - `ui/state.rs` — `UiState`, `VuUniforms`, `VuMeterSharedState`.
  - `ui/theme.rs` — paleta `COL_*`, `resolve_accent`, `resolve_color`.
  - `ui/knob.rs` — `knob_widget`, `handle_knob`.
  - `ui/vu.rs` — `draw_vertical_meter`, helpers OpenGL.
  - `ui/bypass.rs` — `handle_bypass`.
  - `ui/zones.rs` — 5 zonas (Identity/Knobs/VU/Bypass/Status).
  - `ui/file_picker.rs` — gerenciamento de thread + `alive_fence`.
  - `ui/tab_order.rs` — Tab order programático.
- **Critérios de aceitação:** Nenhum módulo > 500 LoC; testes existentes passam.
- **Especialista:** `implementador`.

#### Tarefa S8.T02 — Quebrar `src/clap/gui/window.rs` (655 LoC) 💡

- **Onde:** `src/clap/gui/window.rs` → `src/clap/gui/window/`.
- **Solução técnica:**
  - `window/mod.rs` — `WindowHandler` (~250 LoC).
  - `window/shaders.rs` — GLSL VU meter (~150 LoC).
  - `window/input_map.rs` — keyboard/mouse maps (~200 LoC).
- **Nota técnica (Épico 1):** Ao refatorar `on_frame()` e demais callbacks de `WindowHandler`, preservar o padrão de **early-return silencioso** (sem `.expect()` ou `panic!`) estabelecido na auditoria do Épico 1. Callbacks de baseview cruzam fronteira C ABI — panics causam UB em hosts C++.
- **Critérios de aceitação:** Nenhum módulo > 500 LoC.
- **Especialista:** `implementador`.

#### Tarefa S8.T03 — Eliminar alocações por frame em `draw_ui` 💡

- **Onde:** `src/clap/gui/ui.rs:392, 1774, 1812-1822`.
- **Problema:** `Vec::with_capacity(...)`, `format!`, `join` em paths de draw, ~240 strings/s em status bar.
- **Solução técnica:**
  1. `SmallVec<[Pos2; 49]>` para pontos do knob.
  2. String pooling: caches em `UiState` para tooltips (`Vec<String>` reutilizado).
  3. `write!()` em buffer thread-local em vez de `format!`.
- **Critérios de aceitação:** Memory profiling mostra zero new allocations em 1s de draw idle.
- **Especialista:** `implementador`.

### Sprint S9 — Refatoração do PipeWire host

#### Tarefa S9.T01 — Quebrar `src/standalone/pw_host.rs` (1000 LoC) 🔥

- **Onde:** `src/standalone/pw_host.rs` → `src/standalone/pw_host/`.
- **Solução técnica:**
  - `pw_host/mod.rs` — `run_pipewire_host` (~200 LoC).
  - `pw_host/bridge.rs` — `DspBridge` alloc, `madvise` (~80 LoC).
  - `pw_host/capture.rs` — capture stream setup + listener (~250 LoC).
  - `pw_host/playback.rs` — playback stream setup + listener (~150 LoC).
  - `pw_host/rt_callback.rs` — `drain_resamplers`, `receive_commands`, `sync_rate`, `process_dsp_buffer` (~300 LoC).
- **Critérios de aceitação:** Nenhum módulo > 500 LoC; smoke test `utils/tests-cargo.sh` passa.
- **Especialista:** `implementador`.

#### Tarefa S9.T02 — Tratar hot-plug & resample resync ⚠️

- **Onde:** `src/standalone/pw_host.rs:484, 499-501, 866-904`.
- **Problema:** `hardware_target` capturado uma vez; mudança de sample rate processa frames com resampler antigo (janela de dropout).
- **Solução técnica:**
  1. Setar `node.target = ""` deixando WirePlumber rotear dinamicamente.
  2. Em `sync_rate`, suspender brevemente o processamento até `swap` do resampler.
- **Critérios de aceitação:** Desconectar hardware durante play sem crash; trocar SR sem dropouts inesperados.
- **Especialista:** `implementador`.

#### Tarefa S9.T03 — Quebrar `src/standalone/rt_setup.rs` (685 LoC) ⚠️

- **Onde:** `src/standalone/rt_setup.rs` → `src/standalone/rt_setup/`.
- **Solução técnica:**
  - `rt_setup/mod.rs` — re-exports (~50 LoC).
  - `rt_setup/tsc.rs` — calibração RDTSC (~80 LoC).
  - `rt_setup/affinity.rs` — `select_optimal_cpu`, `parse_interrupts_per_cpu` (~250 LoC).
  - `rt_setup/thread.rs` — `configure_realtime_thread`, `configure_process_wide` (~200 LoC).
  - `rt_setup/telemetry.rs` — `poll_rt_status` (~150 LoC).
  - `rt_setup/pm_qos.rs` — `lock_cpu_c_states`, `detect_hardware_sink` (~70 LoC).
- **Critérios de aceitação:** Nenhum módulo > 500 LoC; smoke test PipeWire passa.
- **Especialista:** `implementador`.

### Sprint S10 — Refatoração de plugin/processor e pipeline

#### Tarefa S10.T01 — Quebrar `src/clap/plugin.rs` (622 LoC) 💡

- **Onde:** `src/clap/plugin.rs` → `src/clap/plugin/`.
- **Solução técnica:**
  - `plugin/mod.rs` — `NamClapPlugin`, `DefaultPluginFactory`.
  - `plugin/shared.rs` — `NamClapShared`.
  - `plugin/main_thread.rs` — `NamClapMainThread`, `load_model`.
- **Critérios de aceitação:** Nenhum módulo > 500 LoC.
- **Especialista:** `implementador`.

#### Tarefa S10.T02 — Quebrar `src/clap/processor.rs` (724 LoC) 💡

- **Onde:** `src/clap/processor.rs` → `src/clap/processor/`.
- **Solução técnica:**
  - `processor/mod.rs` — struct + activate/deactivate.
  - `processor/events.rs` — drain de events e SPSC.
  - `processor/dsp.rs` — bloco DSP propriamente dito.
- **Critérios de aceitação:** Nenhum módulo > 500 LoC.
- **Especialista:** `implementador`.

#### Tarefa S10.T03 — Quebrar `src/dsp/pipeline.rs` (663 LoC) 💡

- **Onde:** `src/dsp/pipeline.rs` → `src/dsp/pipeline/`.
- **Solução técnica:** já há diretório `pipeline/` com testes; mover prod:
  - `pipeline/mod.rs` — re-exports + entry-points.
  - `pipeline/bridge.rs` — `DspBridge`, `BridgeBuffer`, `BridgeRef*` (~100 LoC).
  - `pipeline/context.rs` — `DspPipelineContext`, `DspBuffers` (~60 LoC).
  - `pipeline/stages.rs` — `apply_input_stage`, `run_inference`, `apply_output_stage`, `write_bridge` (~250 LoC).
  - `pipeline/capture.rs` — `capture_dsp_pipeline` agregador.
  - `pipeline/playback.rs` — `playback_dsp_cycle`, `build_spa_format_pod`.
- **Critérios de aceitação:** Nenhum módulo > 500 LoC.
- **Especialista:** `implementador`.

#### Tarefa S10.T04 — Quebrar `src/models/wavenet/conv1d_dyn.rs` (867 LoC) ⚠️

- **Onde:** `src/models/wavenet/conv1d_dyn.rs`.
- **Problema:** Duplicação massiva entre `process_single_frame`, `process_single_frame_bf16`, `process_dual_frame`, `process_dual_frame_bf16`.
- **Solução técnica:**
  1. Generalizar via `trait ConvInput` (já existe em `conv1d.rs`).
  2. Reduzir para ~400 LoC.
- **Critérios de aceitação:** Arquivo < 500 LoC; testes passam.
- **Especialista:** `implementador`.

#### Tarefa S10.T05 — Refatorar `Gate::update` (125 LoC) em estados ⚠️

- **Onde:** `src/dsp/gate.rs:103-228`.
- **Problema:** Função única de 125 linhas mistura 4 estados FSM.
- **Solução técnica:**
  1. Extrair `update_open`, `update_fading_out`, `update_closed`, `update_fading_in`.
  2. Renomear `GateState::Closed` se contexto for ambíguo (uso em `process_mono` em `pipeline.rs:271`).
- **Critérios de aceitação:** Cada método < 50 LoC; cobertura de teste mantida.
- **Especialista:** `implementador`.

---

## Épico 6 — CLAP Compliance e Portabilidade

Objetivo: assegurar que o plugin CLAP é robusto em hosts variados, persiste estado de forma versionada e remove o último gap arquitetural (`PARAM_ACTIVE_MODEL`).

### Sprint S11 — State, params e remote controls

#### Tarefa S11.T01 — Versionar `NamPluginParams` state (com migração v0 → v1) 🔥

- **Onde:** `src/clap/extensions/state.rs:45-77`.
- **Problema:** Sem `version` no payload de save/load. Qualquer adição de campo futuro quebra todos os projetos salvos. **Adicionalmente**, projetos existentes salvos pelo CLAP v1.5.x atual contêm apenas o payload `NamPluginParams` JSON puro — sem migração explícita, a release com versionamento quebraria todos eles.
- **Solução técnica:**
  1. Novo envelope: `{ "version": 1, "params": {...} }`.
  2. No load:
     - Tentar deserializar como `StateEnvelope { version: u32, params: NamPluginParams }`.
     - Se falhar (provável v0 legacy: sem chave `version`), fallback para `NamPluginParams` direto, tratado como `version = 0`, aplicando migration default (estado v0 → v1: copy campos comuns, novos campos com `Default`).
  3. Adicionar `#[serde(default)]` em todos os campos de `NamPluginParams` (já obrigatório para v0 compat).
  4. Padrão de migration: `fn migrate(v: u32, params: NamPluginParams) -> NamPluginParams` para evolução futura (v1 → v2 etc).
- **Critérios de aceitação:**
  - **Migração v0 → v1:** payload sem campo `version` (gerado por CLAP v1.5.x) carrega com sucesso, todos os 4 params recuperados, sem warning.
  - **Round-trip v1:** save em v1, restore em v1 idempotente.
  - **Forward v1 → v2 (futuro):** envelope com `version` desconhecido emite warning e fallback aos defaults sem panic.
  - Testes explícitos em `tests/clap_state_migration.rs` cobrindo os 3 cenários.
- **Especialista:** `implementador`.

#### Tarefa S11.T02 — Substituir `Box::leak` em `state.rs` por erros tipados ⚠️

- **Onde:** `src/clap/extensions/state.rs:46-77`.
- **Problema:** Leak intencional de strings em paths de erro. Acumula em hosts com muitos save/load (Bitwig auto-save).
- **Solução técnica:**
  1. Trocar `PluginError::Message` por variante custom com `Cow<'static, str>` se a API `clack` permitir.
  2. Caso contrário, usar pool estático de erros conhecidos.
- **Critérios de aceitação:** Sem `Box::leak` no módulo state.
- **Especialista:** `implementador`.

#### Tarefa S11.T03 — Path relativo opcional em `model_path` 💡

- **Onde:** `src/common/params.rs:27`; `src/clap/extensions/state.rs`.
- **Problema:** Path absoluto em projects quebra portabilidade entre máquinas/usuários.
- **Solução técnica:**
  1. Adicionar `model_basename: Option<String>` e `model_search_paths: Vec<PathBuf>`.
  2. Em load, se path absoluto não existe, procurar `basename` em search_paths.
- **Critérios de aceitação:** Projeto salvo em Linux abre em Linux com path diferente sem erros.
- **Especialista:** `implementador`.

#### Tarefa S11.T04 — Remover `PARAM_ACTIVE_MODEL` da página Remote Controls 💡

- **Onde:** `src/clap/extensions/remote_controls.rs:28`.
- **Problema:** Inclui param READONLY em página de "Main" — knob inerte para usuários MIDI.
- **Solução técnica:** Remover do array e ajustar índices.
- **Critérios de aceitação:** Teste de remote controls passa; revisão manual em DAW.
- **Especialista:** `implementador`.

#### Tarefa S11.T05 — Corrigir `text_to_value` de `PARAM_ACTIVE_MODEL` 💡

- **Onde:** `src/clap/extensions/params.rs:187`.
- **Problema:** Retorna `Some(0.0)` para READONLY — confunde hosts.
- **Solução técnica:** Retornar `None`.
- **Critérios de aceitação:** Testes existentes não regridem.
- **Especialista:** `implementador`.

### Sprint S12 — Lifecycle e telemetria do plugin

#### Tarefa S12.T01 — Eliminar `request_restart()` redundante 💡

- **Onde:** `src/clap/plugin.rs:345-347`.
- **Problema:** Após `latency_ext.changed()`, chamar `request_restart()` é redundante e pode causar dropouts ou comportamento inesperado em FL Studio.
- **Solução técnica:** Manter só `changed()`.
- **Critérios de aceitação:** Plugin troca latência sem reinicialização em hosts testados.
- **Especialista:** `implementador`.

#### Tarefa S12.T02 — Empacotar `gesture_*` flags em `AtomicU32` ⚠️

- **Onde:** `src/clap/plugin.rs:127-152`.
- **Problema:** 12 `AtomicBool` desperdiçam 12 cache lines potenciais (com alinhamento). Bitmap único é mais eficiente.
- **Solução técnica:**
  1. Substituir array por `AtomicU32` com bitmask `(1 << i)` por parâmetro × gesto (begin/end/value-change).
  2. Helpers `set_gesture`, `take_gesture`, `clear_gestures`.
- **Critérios de aceitação:** Smoke test de gestures (begin/end + value change) ok.
- **Especialista:** `implementador`.

#### Tarefa S12.T03 — Mover `mono_hyst`, `active_model_r` para campos de `NamClapProcessor` 💡

- **Onde:** `src/clap/processor.rs:550-551`.
- **Problema:** Re-inicializados a cada iter do port_pair. Aceitável agora, mas se `DynamicHysteresis::new()` alocar internamente, vira issue RT.
- **Solução técnica:** Migrar para fields persistentes.
- **Critérios de aceitação:** Heap audit confirma zero allocs no `process()`.
- **Especialista:** `implementador`.

---

## Épico 7 — Testes, Fuzzing e Validação Cruzada

Objetivo: blindar contra regressões e estabelecer baseline empírico de paridade vs C++.

### Sprint S13 — Cobertura de testes & cross-impl validation

#### Tarefa S13.T01 — Suite de cross-validation NAM-rs ↔ NeuralAmpModelerCore 🔥

- **Onde:** `tests/cpp_parity/` (novo).
- **Problema:** Goldens existentes são auto-referenciais (regression). Para detectar drift estrutural, comparar com saída real do C++.
- **Solução técnica:**
  1. Compilar `NeuralAmpModelerCore` como binário CLI (`utils/cpp_parity/` script).
  2. Para cada modelo em `tests/fixtures/models/` e cada arquivo de sinal de teste, gravar saída C++.
  3. Test runner em `tests/cpp_parity.rs` carrega cada modelo, processa o mesmo input e compara MSE < 1e-4.
- **Critérios de aceitação:** Todos os modelos referência (WaveNet Standard, Lite, Feather, Nano; LSTM 1×{8,12,16,24}, 2×{8,12,16}) batem com C++.
- **Especialista:** `pesquisador-inovador` + `revisor-auditor`.

#### Tarefa S13.T02 — Round-trip encode→decode em NAMB v2 ⚠️

- **Onde:** `tests/namb_v2_roundtrip.rs` (novo).
- **Problema:** Bugs Sprint S3.T03 só foram identificados por leitura manual.
- **Solução técnica:**
  1. Para cada layout (`Original`, `GateMajorLstm`, `Interleaved4WaveNet`) e topologia (todas no catálogo), gerar `NamModelData`, encodar para `.namb`, decodar, comparar.
  2. Assertar igualdade bit-a-bit de pesos transposed.
- **Critérios de aceitação:** Round-trip passa para 11 topologias (7 LSTM + 4 WaveNet).
- **Especialista:** `implementador`.

#### Tarefa S13.T03 — Property-based testing em parsers 💡

- **Onde:** `tests/proptest_parsers.rs` (estender).
- **Solução técnica:**
  1. Adicionar shrinking estratégia para `Arbitrary<NamModelData>`.
  2. 100k iterações com `arbitrary_namb_bytes` (header válido + corpo aleatório).
- **Critérios de aceitação:** Zero panics em 100k inputs.
- **Especialista:** `implementador`.

#### Tarefa S13.T04 — Stress test multi-instância CLAP ⚠️

- **Onde:** `tests/clap_multi_instance.rs` (novo).
- **Problema:** `ONCE_PRIO` global pode causar comportamento errático em hosts com 10+ instâncias.
- **Solução técnica:**
  1. Instanciar 10 plugins via `clack-host`.
  2. Verificar telemetria, params, activate/deactivate sem race conditions.
- **Critérios de aceitação:** Sem panic; rt_priority correto em cada instância.
- **Especialista:** `implementador`.

#### Tarefa S13.T05 — Teste de prewarm edge (RF grande) ⚠️

- **Onde:** `tests/wavenet_prewarm_edge.rs` (novo).
- **Solução técnica:**
  1. Modelo sintético com `dilation=512, K=5` (RF=2560).
  2. Prewarm com `num_samples=2048`.
  3. Verificar ausência de OOB / underflow.
- **Critérios de aceitação:** Sem `debug_assert!` quebrado; saída plausível.
- **Especialista:** `implementador`.

#### Tarefa S13.T06 — Adicionar variantes LSTM ao catálogo (1×40, 2×24) 💡

- **Onde:** `src/models/lstm/mod.rs`; `src/loader/dispatcher/lstm.rs:17-46`.
- **Problema:** Modelos `LSTM 1×40` (tone matching) e `2×24` (deeper) caem em fallback dinâmico, perdendo performance.
- **Solução técnica:**
  1. Adicionar `Lstm1x40`, `Lstm2x24` ao enum `DynamicModel`.
  2. Adicionar match no dispatcher.
  3. Testes de regressão e benchmark.
- **Critérios de aceitação:** Modelos batem performance dentro de 5% das variantes catalogadas.
- **Especialista:** `implementador`.

---

## Épico 8 — Documentação Técnica

Objetivo: assegurar que cada decisão e cada subsistema crítico têm documentação acessível para mantenedores futuros.

### Sprint S14 — Documentação técnica & comentários

#### Tarefa S14.T01 — (Referência cruzada) Especificação formal do formato NAMB 💡

- **Status:** **Entrega real consolidada em S5.T07** (Épico 3, Sprint S5). Esta entrada permanece apenas para sinalizar que o trabalho está mapeado no Épico de documentação técnica.
- **Validação esperada na conclusão do S14:** Confirmar com `documentador` que `docs/namb-spec.md` está completo e atualizado refletindo as mudanças dos Sprints S3 e S5 (CRC flag, padding Interleaved-4, erros tipados).
- **Especialista:** `documentador` (revisão apenas).

#### Tarefa S14.T02 — Padronizar idioma de docstrings (pt-BR) ⚠️

- **Onde:** Todo `src/`.
- **Problema:** Mistura pt-BR e en-US em docstrings, mensagens, comentários. README do projeto especifica pt-BR para devs.
- **Solução técnica:** Pass de revisão (`documentador`) garantindo pt-BR em `///`, `//`, `//!`. Mensagens de erro user-facing podem ficar em inglês se justificado.
- **Critérios de aceitação:** Coverage > 90% pt-BR.
- **Especialista:** `documentador`.

#### Tarefa S14.T03 — Adicionar `# Safety` faltantes em `unsafe { ... }` 💡

- **Onde:** `src/clap/processor.rs:236-251, 387-389`; `src/dsp/pipeline.rs:240, 259`; outros identificados na auditoria.
- **Solução técnica:** Adicionar comentário `// SAFETY: ...` justificando cada bloco.
- **Critérios de aceitação:** `cargo clippy -- -D clippy::undocumented_unsafe_blocks` passa.
- **Especialista:** `documentador` + `implementador`.

#### Tarefa S14.T04 — Atualizar `docs/architecture.md` 💡

- **Onde:** `docs/architecture.md`.
- **Solução técnica:** Refletir mudanças dos Épicos 1–6 (DspBridge split, ui module breakdown, etc.).
- **Critérios de aceitação:** Documento revisto pela skill `documentador`.
- **Especialista:** `documentador`.

#### Tarefa S14.T05 — Comentários técnicos em hotpath de SIMD 💡

- **Onde:** `src/math/common/avx2_impl.rs`, `avx512_impl.rs`, `src/math/gemm/dot_4x.rs`.
- **Problema:** Funções SIMD com algoritmos não-óbvios sem documentação de microarquitetura alvo (Skylake/Zen/Ice Lake).
- **Solução técnica:** Adicionar header em cada kernel SIMD com:
  - Latência/throughput esperada.
  - Número de acumuladores e justificativa.
  - Citação a paper/manual se aplicável.
- **Critérios de aceitação:** Toda função `#[target_feature(...)]` tem header documentado.
- **Especialista:** `documentador` + `pesquisador-inovador`.

---

## Parte II — Inovações Avant-Garde (Pesquisa & Próxima Geração)

Objetivo: capturar 4–20× speedup em hardware Intel Sapphire Rapids+ (AMX, AVX10.2) e expandir alvo para ARM64 (Apple Silicon, Ampere, Graviton), elevando o NAM-rs do estado "AVX-512 VNNI BF16" para o estado-da-arte de 2026.

---

## Épico 9 — Quantização e Compressão de Modelos

Objetivo: reduzir 2–4× a memória de pesos e 2–8× a banda do hotpath via INT8/INT4 quantization moderna (SmoothQuant/AWQ).

### Sprint S15 — INT8/INT4 Weight Quantization

#### Tarefa S15.T01 — INT8 weight quantization SmoothQuant para Conv1D heads ✨⚠️

- **Onde:** `src/loader/dispatcher/wavenet.rs:348-349` (cabeças); novo `src/math/common/int8_quant.rs`; novo `weights_layout = SmoothQuantInt8`.
- **Problema/Oportunidade:** Pesos do `head_weights` (Conv1D 1×1 do output) **dominam memória** em WaveNet Standard (40 KB de pesos vs 8 KB de activations). INT8 weights + FP32 activations (per-channel scale) reduzem 4× memory bandwidth (cache-friendly em L1/L2). SmoothQuant migra outliers de activations para weights via per-channel scaling — proven 99.5% accuracy retention em LLM.cpp e NAM-class workloads.
- **Solução técnica:**
  1. **Treinamento-livre quantization** (post-training): para cada Conv1D head, computar per-channel scale `s_c = max(|W_c|) / 127`, armazenar `Q_W[c,i] = round(W[c,i] / s_c)` como `i8` + scale vector `s_c` como `f32`.
  2. **Kernel `dot_product_int8_avx512`** usando `_mm512_dpbusd_epi32` (AVX-512 VNNI) — 4× speedup vs F32 FMA em throughput INT8.
  3. **AMX path:** `_tile_dpbssd` (S23.T02-style) para LSTM matmul INT8.
  4. **Encoder NAMB v3:** novo `weights_layout = SmoothQuantInt8` que serializa `[Q_W: i8, scales: f32]`. v3 bump justificado.
  5. **Auto-calibração:** durante o `loader/mod.rs`, opcional sweep de input típico (impulse response) para ajustar scales adversariamente.
  6. **Fallback:** se SmoothQuant falha calibração (golden delta > tolerância), reverter para BF16/FP32 com warning.
- **Pré-requisitos (obrigatórios — herdam invariantes da Parte I):**
  - **S3.T03/S3.T04** — disciplina de layout sequencial e padding implícito; SmoothQuant deve usar a mesma estratégia (bloco contíguo `[Q_W: i8 ..., scales: f32 ...]` por camada, padding para múltiplo do bloco SIMD).
  - **S5.T03** (flag `FLAG_HAS_CRC32`) e **S5.T07** (spec NAMB) — a seção `SmoothQuantInt8` deve ser adicionada à spec **antes** da implementação; bump explícito para NAMB v3 com `FLAG_HAS_QUANT_INT8`.
  - **S13.T02** (round-trip) — cobertura obrigatória do novo layout antes do merge.
- **Critérios de aceitação:**
  - Modelo WaveNet Standard quantizado: tamanho do arquivo 60% menor, MSE vs FP32 < 1e-3 em 60s de signal de teste.
  - Benchmark mostra ≥ 30% redução em latência média para WaveNet Standard.
  - Round-trip encode/decode preserva pesos com erro < 1/127, validado via harness estendido de S13.T02.
- **Especialista:** `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 4 dias.

#### Tarefa S15.T02 — INT4 weight packing experimental (AWQ-style) ✨💡

- **Onde:** estensão de S15.T01 para `weights_layout = AwqInt4`.
- **Problema/Oportunidade:** INT4 (4 bits) entrega 8× memory reduction. AWQ (Activation-aware Weight Quantization, Lin et al. 2023) preserva pesos "salientes" em FP16 e quantiza o resto em INT4. Apropriado para WaveNet com layers de magnitude variada (~1% dos pesos contribuem >50% do output).
- **Solução técnica:**
  1. Identificar 1% top-magnitude weights via análise off-line (script `utils/awq-calibrate.py` opcional, ou heuristic Rust).
  2. Layout: `[Q_W: u4 packed nibbles, salient_mask: bitmap, salient_values: f16, scales: f32]`.
  3. Decoder kernel: unpack INT4 → INT8 com LUT, depois INT8 dot product (reusa S15.T01 path).
  4. **Apenas catálogo dinâmico** (não Conv1D estático) — INT4 é override expressivo.
- **Pré-requisitos (obrigatórios):** S15.T01 (path INT8 + scales infra), S5.T07 (spec NAMB v3 com `FLAG_HAS_QUANT_INT4`), S13.T02 (round-trip estendido).
- **Critérios de aceitação:**
  - MSE < 5e-3 para WaveNet Standard quantizado AWQ vs FP32 (tolerância dobrada vs INT8).
  - Tamanho de arquivo 80% menor que FP32.
  - Round-trip encode/decode validado no harness estendido de S13.T02.
  - Feature `awq-int4` em Cargo (default off).
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 3 dias.

---

## Épico 10 — Sistema Operacional & Real-Time Avant-Garde

Objetivo: capturar todo o potencial do kernel Linux 6.x (PREEMPT_RT mainline), reduzir TLB pressure com huge pages, observar a stack via eBPF e PMU, e migrar I/O de modelo para io_uring (async, sem bloqueio de main thread).

### Sprint S16 — Scheduler & Memória

#### Tarefa S16.T01 — Suporte opcional a SCHED_DEADLINE (CBS) ✨⚠️

- **Onde:** `src/standalone/rt_setup.rs:431-450`; novo CLI flag `--scheduler {fifo,deadline}`.
- **Problema/Oportunidade:** Paper Raspberry Pi 5 + PREEMPT_RT (arXiv 2604.19275, Abr/2026) demonstra que SCHED_DEADLINE bound max-latency em ≤197 µs sob carga heavy, **vs 224 µs de SCHED_FIFO p99**. CBS (Constant Bandwidth Server) garante admission control — impossível starvation. Apropriado para áudio onde "buffer fits in deadline" é uma garantia formal.
- **Solução técnica:**
  1. Param/flag `nam-rs --scheduler deadline` (default mantém FIFO para compat).
  2. `sched_setattr` com `sched_runtime = 80%·block_period`, `sched_deadline = block_period`, `sched_period = block_period`. Calcular dinamicamente após `node.latency` conhecido.
  3. Fallback automático para FIFO se `EBUSY` (admission control rejeitou) ou kernel < 3.14.
  4. Telemetria: log de deadline missed (via `SCHED_FLAG_RECLAIM` + `dl_runtime` query).
  5. Documentar setup em `docs/realtime-tuning.md` (criar): habilitar `sched_rt_runtime_us = -1`, ajustar cgroup cpu.max.
- **Critérios de aceitação:**
  - Em kernel PREEMPT_RT 6.x, flag `--scheduler deadline` ativa; cyclictest-equivalente interno mostra max-latency < FIFO sob `stress-ng --cpu 16`.
  - Smoke test em CI Linux PREEMPT_RT (GitHub Actions runner com kernel custom, ou local).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 2.5 dias.

#### Tarefa S16.T02 — Huge Pages (THP / MAP_HUGETLB) para weights e mirror buffer ✨⚠️

- **Onde:** `src/loader/mod.rs` (alocação de `AlignedVec<u16>` para pesos dinâmicos); `src/dsp/vring.rs:62-164` (mirror buffer).
- **Problema/Oportunidade:** Modelos WaveNet Standard alocam ~80 KB de pesos contíguos. Em páginas de 4 KB, esses pesos consomem ~20 entradas TLB; em **2 MiB huge pages**, **1 entrada TLB**. TLB miss em hotpath custa ~100 ciclos. Para audio de 32 spl @ 96k = 333 µs, eliminar TLB misses pode reduzir p99 em 5–15%.
- **Solução técnica:**
  1. **Allocator helper** `src/math/common/huge_alloc.rs`: tenta `mmap(MAP_HUGETLB | MAP_HUGE_2MB)` primeiro; fallback `mmap` anonymous + `madvise(MADV_HUGEPAGE)` para THP transparent; fallback `Vec` standard.
  2. Substituir alocações de pesos > 1 MiB e mirror buffer (`vring.rs`) por esse allocator.
  3. **Métrica:** expor count via `RT_STATUS_HUGEPAGE_OK` flag (telemetria).
  4. **Cautela:** THP background scanning pode pausar threads — preferir explicit `MAP_HUGETLB`. Documentar setup: `echo 32 > /proc/sys/vm/nr_hugepages` ou cgroup hugetlb.2MB.max.
- **Critérios de aceitação:**
  - `perf stat -e dTLB-load-misses` mostra redução ≥ 50% no DSP thread.
  - Benchmark p99 latency reduz ≥ 5% em modelos grandes.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S16.T03 — Detecção e tuning específico para PREEMPT_RT kernel ✨💡

- **Onde:** `src/standalone/rt_setup.rs`; novo `is_preempt_rt()` check.
- **Problema/Oportunidade:** PREEMPT_RT mainline (kernel 6.x) tem semântica diferente: spinlocks viram sleeping locks, IRQs threadeadas. Comportamento "ideal" (`SCHED_FIFO prio 99`) pode mudar de "deve" para "pode preempar threaded-IRQ críticos".
- **Solução técnica:**
  1. Checar `/sys/kernel/realtime` ou `uname -v | grep PREEMPT_RT`.
  2. Se PREEMPT_RT: usar prio 80 (deixa headroom para `ksoftirqd` em prio 90+).
  3. Se vanilla: manter prio 90 (sem threaded IRQs).
  4. Habilitar `SCHED_DEADLINE` mais agressivo (S16.T01).
  5. Log informativo `📈 PREEMPT_RT kernel detected; tuning RT_PRIORITY=80`.
- **Critérios de aceitação:** Detecção correta em kernel 6.6-rt; tuning aplicado.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 0.5 dia.

#### Tarefa S16.T04 — perf_event_open: PMU counters em telemetria ✨💡

- **Onde:** novo `src/dsp/perf_counters.rs`.
- **Problema/Oportunidade:** Hoje a telemetria mede latência wall-clock. PMU (Performance Monitoring Unit) entrega **IPC, cache misses (L1/L2/L3), branch mispredictions, page-faults** — diagnóstico cirúrgico de regressões.
- **Solução técnica:**
  1. `perf_event_open` em modo `PERF_TYPE_HARDWARE`, grupo de 4 contadores (CYCLES, INSTRUCTIONS, CACHE_MISSES, BRANCH_MISSES).
  2. `mmap`-based ring buffer para read lock-free do main thread (kernel-side, sample-free reading).
  3. Expor via `dsp_pipeline_test`'s `RtStatus` para CLI/GUI debug overlay.
  4. Feature `pmu-counters` (default off — requer `CAP_SYS_ADMIN` ou `perf_event_paranoid <= 0`).
- **Critérios de aceitação:** `RUST_LOG=info nam-rs --pmu` mostra IPC histogram em sessão de 60s.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

### Sprint S17 — io_uring & async loading

#### Tarefa S17.T01 — Async model loading via io_uring ✨⚠️

- **Onde:** `src/loader/mod.rs:70-94`; novo `src/loader/async_io.rs`.
- **Problema/Oportunidade:** Hoje `std::fs::read` é síncrono — usuário arrastando modelo grande de 30 MiB em DAW vê **UI freeze** por ~100ms (SSD) ou ~2s (NFS). io_uring permite zero-syscall I/O completion + worker thread separado, mantendo UI responsiva.
- **Solução técnica:**
  1. Crate `io-uring` (sem deps pesadas além de libc).
  2. Worker thread dedicado lê arquivo via SQE/CQE assíncronos; main thread continua draw loop.
  3. Progress reporting via `Arc<AtomicU64>` (bytes lidos).
  4. Em main thread, "Loading..." status com progress bar.
  5. Fallback `std::fs::read` para kernels < 5.1.
- **Critérios de aceitação:**
  - UI permanece responsiva (>30 FPS) durante load de modelo 30 MiB.
  - Tempo de load ≤ `std::fs::read` baseline (não regredir).
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S17.T02 — eBPF tracing target para profiling production-grade ✨💡

- **Onde:** novo `utils/ebpf/dsp_latency.bt` (bpftrace script); doc em `docs/observability.md`.

- **Problema/Oportunidade:** Em produção, debugging de glitches específicos exige profiling não-intrusivo. eBPF traces no audio thread sem overhead mensurável; bpftrace one-liners permitem "quem causou esse xrun?".

- **Solução técnica:**

  1. Marcar funções RT-críticas com `#[no_mangle] #[link_section = ".text.rt"]` para uprobe attachment.

  2. `utils/ebpf/dsp_latency.bt`:

     ```bpftrace
     uprobe:nam-rs:nam_clap_processor_process { @start[tid] = nsecs; }
     uretprobe:nam-rs:nam_clap_processor_process /@start[tid]/ {
       @lat = hist((nsecs - @start[tid]) / 1000);
       delete(@start[tid]);
     }
     ```

  3. Doc em `docs/observability.md` com receitas comuns.

- **Critérios de aceitação:** Script roda; histograma sai consistente com telemetria interna; overhead < 0.1%.

- **Especialista:** `pesquisador-inovador`.

- **Esforço:** 1 dia.

---

## Épico 11 — UX Avant-Garde

Objetivo: ir muito além de "carrega modelo + ajusta gain". Construir o engine NAM mais sofisticado em UX, com hot model swap sem dropouts, A/B comparator, IR cabsim integrado, tone matching e controle remoto MIDI/OSC.

### Sprint S18 — Hot Swap & A/B

#### Tarefa S18.T01 — Hot model swap com crossfade ✨🔥

- **Onde:** `src/clap/processor.rs`; `src/standalone/pw_host.rs:rt_callback`; `src/loader/mod.rs`.
- **Problema/Oportunidade:** Hoje, trocar de modelo causa um silêncio de ~50ms (load + prewarm). Crossfade sample-accurate elimina audible dropout, permitindo **A/B blind comparison** e workflow rápido de tone-hunting.
- **Solução técnica:**
  1. **Reader/Writer pattern de S1.T01 estendido:** introduzir `ModelReader { ptr: NonNull<dyn NamModel>, generation: u64 }` exposto à RT thread; o main thread mantém um `arc-swap::ArcSwap<Box<dyn NamModel>>` (ou equivalente custom lock-free) — **nunca** `&'static mut`. RT acessa via `&*ptr` com lifetime curto, dentro de uma única call de `process()`.
  2. **Double-slot lock-free:** dois slots `[ModelSlot; 2]` indexados por `active_idx: AtomicUsize` (Relaxed). Main thread escreve em `[1 - active_idx]`, RT lê de `[active_idx]`. Swap via single `store(Release)`.
  3. Main thread carrega novo modelo em background (io_uring de S17.T01), prewarm em separate thread.
  4. RT thread detecta `pending_slot.is_loaded()` no início do bloco; inicia crossfade linear de 64 ms.
  5. Durante crossfade: processa input por **ambos** modelos (`old` lido do slot atual + `new` lido do slot pendente), mixa output via `α · new + (1-α) · old`, com `α` rampando de 0→1 ao longo de 64 ms.
  6. **Quiescência antes do drop:** ao fim do crossfade, RT marca `old_model_can_drop = true` (atomic). Main thread espera **pelo menos 2 blocos** depois do swap (period de quiescence) antes de chamar `drop` no Box antigo — garante que nenhuma RT thread retém ainda uma referência ao Box prestes a ser liberado.
  7. **Heap-audit gate:** o `drop` ocorre exclusivamente no main thread (zero allocação/liberação na RT thread, conforme `cargo test --features heap-audit` de S2.T01).
  8. Latência adicional durante crossfade: 64ms (aceitável; opcional).
  9. Param `PARAM_CROSSFADE_MS` (range 0–500, default 64).
- **Pré-requisitos (obrigatórios — herdam invariantes da Parte I):**
  - **S1.T01** (`DspBridgeReader`/`Writer` split, eliminação de `&'static mut`): mesma disciplina aplicada ao slot de modelo. Sem reintroduzir aliasing XOR-mut violado.
  - **S2.T01** (`heap-audit` sem panic): valida que o swap não aloca/libera na RT thread.
  - **S2.T03** (`alive_fence` / `safe_shared`): se a GUI ou um file picker thread interage com o slot, o mesmo padrão de fence deve ser aplicado para evitar UAF durante destruição do plugin.
  - **S4.T04** (`reset(sr, max_buf)` trait): o novo modelo deve ser inicializado via `reset` antes do swap (não confiar em `prewarm` legado).
  - **S17.T01** (io_uring async load): pré-condição para que o load não bloqueie main thread durante > 5ms.
- **Critérios de aceitação:**
  - Trocar de modelo durante reprodução musical: zero dropout audível (validar com soak test 1h).
  - `cargo +nightly miri test crossfade_model_swap` passa sem warning de aliasing.
  - `cargo test --features heap-audit` confirma zero allocs/drops na RT thread durante swap.
  - Stress-test fechando o plugin enquanto crossfade ativo (host destrói entre blocos): sem UAF (validado em CI fuzz `test_gui_drag_drop_fuzz` extendido).
  - Telemetria mostra crossfade duration consistente.
- **Especialista:** `implementador` + `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 3 dias.

#### Tarefa S18.T02 — A/B model comparator (snapshot bank) ✨⚠️

- **Onde:** `src/clap/extensions/state.rs`; novo módulo `src/clap/ab_bank.rs`.
- **Problema/Oportunidade:** Workflow profissional exige A/B blind comparison. Hoje é um swap manual de arquivo — destrutivo. Snapshot bank com 8 slots persistentes permite comparação instantânea (atalho de teclado A/B/1-8).
- **Solução técnica:**
  1. `NamPluginParams` estendido com `Vec<SnapshotSlot>` (8 slots).
  2. Cada slot armazena `model_path: PathBuf`, `gate_db: f32`, `output_db: f32`.
  3. CLAP param `PARAM_ACTIVE_SLOT` (range 0–7, modulação OK = crossfade S18.T01 disparado).
  4. GUI: 8 botões + atalho keyboard.
  5. State versionado (v2 schema sobre v1 de S11.T01).
- **Critérios de aceitação:** Crossfade A/B funciona; state v2 persiste 8 slots.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

### Sprint S19 — DSP Suplementar

#### Tarefa S19.T01 — IR cabsim convolution (uniformly-partitioned FFT) ✨🔥

- **Onde:** novo `src/dsp/ir_cab.rs`.
- **Problema/Oportunedade:** Workflow NAM é "amp + cabinet". Hoje, usuário precisa de plugin separado (Topaz, NadIR). Integrar cabsim com convolução IR (impulse response, .wav de 4096–8192 spl) **eliminado um plugin do chain** e habilitando workflow "amp+cab presets" únicos.
- **Solução técnica:**
  1. **Uniformly-Partitioned Convolution (UPC):** dividir IR em blocos de N=64 amostras; convolve cada bloco via FFT 128-point (já existe `rustfft`); somar com latência total = N.
  2. **Frequency-domain delay line** evita realocação por bloco.
  3. SIMD complex multiply em FFT bins.
  4. **CLAP IO format:** parâmetros `PARAM_IR_PATH` (file picker drag-drop), `PARAM_IR_GAIN`, `PARAM_IR_ENABLED`.
  5. Carregamento async via io_uring (S17.T01).
- **Critérios de aceitação:**
  - Convolução de IR 4096-tap em < 50% do block budget @ 48k/64 spl.
  - Match bit-perfect vs reference convolution (numpy.convolve) com FFT round-trip.
  - GUI: drag-drop file picker para IR (.wav).
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 4 dias.

#### Tarefa S19.T02 — Auto LUFS normalization ao trocar modelo ✨💡

- **Onde:** `src/dsp/lufs.rs` (novo); integração em `pipeline.rs`.
- **Problema/Oportunidade:** Modelos NAM variam ~20 dB de output entre si — trocar modelo causa **shock de volume**. Auto-LUFS normaliza para −18 LUFS-S (target broadcast) com ramp suave de 200ms.
- **Solução técnica:**
  1. Implementar BS.1770-4 LUFS meter (K-weighting pre-filter + RMS + gate −70 LUFS).
  2. Em swap de modelo, calcular `target_gain = -18 - measured_lufs` over 1s; ramp via `apply_ramp_stereo`.
  3. Param `PARAM_AUTO_LUFS: bool` (default on; usuário pode desligar para A/B blind).
- **Critérios de aceitação:** Trocar entre 4 modelos de catálogo: output LUFS-S converge para -18 ±1 LUFS após 1s.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S19.T03 — Spectrum analyzer pré/pós (visual feedback) ✨💡

- **Onde:** novo `src/clap/gui/spectrum.rs`; integração em `ui.rs` (nova zona).
- **Problema/Oportunedade:** Visual feedback de "o que o modelo está fazendo no espectro" é altamente educativo e diferencial UX. STFT 2048-point @ 30 Hz refresh; overlay pre/post.
- **Solução técnica:**
  1. Capture ring buffer (~256 ms) de input e output em SPSC.
  2. Main thread (GUI): STFT via `rustfft` 2048-point, Hann window, overlap 75%.
  3. Renderizar via egui_glow como linha + fill, log-frequency axis (20 Hz – 20 kHz).
  4. Toggle visibility via param `PARAM_SPECTRUM_ENABLED`.
- **Critérios de aceitação:** Spectrum render em 30 FPS sem hiccup; identifica trivialmente um cab high-cut a 5 kHz.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2 dias.

#### Tarefa S19.T04 — Tone-matching mode (EQ correction em tempo real) ✨🔥

- **Onde:** novo `src/dsp/tone_match.rs`.
- **Problema/Oportunidade:** **Feature killer** — usuário fornece "target tone" (snippet de áudio referência), engine aprende correção EQ em ~5s e aplica em pós-modelo. Combina FFT de target e FFT atual, calcula response, projeta em IIR biquads ou FIR mínimo phase.
- **Solução técnica:**
  1. **Captura target:** 5-10s de audio de referência via drag-drop ou record button.
  2. **Average magnitude spectrum** (Welch's method) de target e current output.
  3. **EQ correction:** `H_corr(f) = |Target(f)| / |Current(f)|`, smoothed em log scale.
  4. **Projeção em filterbank:** 31-band graphic EQ (ISO 1/3 oct) ou 10 IIR biquads peaking.
  5. Aplicar como pós-modelo IIR (RT-safe, < 100 instruções por sample).
- **Critérios de aceitação:** Após tone match, MSE espectral entre output e target < -30 dB em região 100–8000 Hz.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 4 dias.

### Sprint S20 — Controle Remoto & Integração

#### Tarefa S20.T01 — MIDI Learn nativo (CC mapping) ✨⚠️

- **Onde:** novo `src/clap/extensions/note_ports.rs` (já parcialmente existe via `clack-extensions`); UI binding em `ui.rs`.
- **Problema/Oportunidade:** Controlar gain/gate/model com pedal MIDI é workflow live essencial. CLAP suporta `note_ports` + `params` via MIDI mapping nativo.
- **Solução técnica:**
  1. Right-click em knob → "MIDI Learn" → próximo CC recebido bind ao param.
  2. Persistir mapping em state (v3 schema).
  3. CLAP event handling no `processor.rs`.
- **Critérios de aceitação:** Pedal MIDI controla gate threshold em DAW.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

#### Tarefa S20.T02 — OSC remote control (standalone) ✨💡

- **Onde:** novo `src/standalone/osc.rs`; flag CLI `--osc-port 9000`.
- **Problema/Oportunidade:** Standalone PipeWire usado em live performances; controlar via TouchOSC / iPad via UDP OSC sobre WiFi local é elegante e mainstream em pedalboard ecosystems.
- **Solução técnica:**
  1. Crate `rosc` (lightweight); thread separado escutando UDP 9000.
  2. Mapear `/nam/gain`, `/nam/gate`, `/nam/model_size` para params.
  3. Bidirecional: enviar telemetria (RT_STATUS, gain reduction meter) de volta para TouchOSC.
- **Critérios de aceitação:** TouchOSC controla nam-rs standalone via WiFi; latência < 20ms.
- **Especialista:** `implementador`.
- **Esforço:** 2 dias.

---

## Épico 12 — Validação Empírica & Observabilidade Avant-Garde

Objetivo: capturar empiricamente a qualidade do engine via differential fuzzing C++↔Rust, métricas perceptuais (PESQ/STOI/MR-STFT), HDR histograms de latência e otimização compiler-grade (PGO + BOLT).

### Sprint S21 — Differential Validation & Métricas Perceptuais

#### Tarefa S21.T01 — Differential fuzzing C++↔Rust com cargo-fuzz ✨🔥

- **Onde:** `fuzz/fuzz_targets/cpp_diff.rs` (estende S5.T08); `utils/cpp_parity/` (S13.T01).
- **Problema/Oportunedade:** Hoje a paridade C++↔Rust é validada em fixed inputs (S13.T01). Differential fuzzing **gera inputs adversariais** e detecta drift escondido — abordagem padrão da indústria (libFuzzer + sancov, OSS-Fuzz).
- **Solução técnica:**
  1. Target lê `data: &[u8]` → interpretar como (model_path_index, audio_samples).
  2. Run both engines (Rust nam-rs lib + C++ NeuralAmpModelerCore via FFI).
  3. Assert `mse(rust_out, cpp_out) < 1e-3`; falhas geram corpus.
  4. CI roda 1h por commit em GitHub Actions.
- **Critérios de aceitação:** 100M iterações sem failure; corpus persistido em `fuzz/corpus/cpp_diff/`.
- **Especialista:** `pesquisador-inovador` + `revisor-auditor`.
- **Esforço:** 2 dias.

#### Tarefa S21.T02 — Métricas perceptuais (MR-STFT, ESR) em CI ✨⚠️

- **Onde:** `tests/perceptual_metrics.rs`.
- **Problema/Oportunidade:** MSE é métrica fraca para áudio (não captura percepção). MR-STFT (multi-resolution STFT loss, Yamamoto et al. 2020) e ESR (Error-to-Signal Ratio) são padrões da pesquisa NAM. CI deve falhar se ESR > threshold de regressão.
- **Solução técnica:**
  1. Implementar MR-STFT com window sizes {256, 1024, 4096} log magnitude L1+L2.
  2. ESR: `10*log10(sum(diff²) / sum(target²))`.
  3. Test runner compara saída atual vs golden (último known-good); fail se ESR delta > 1 dB.
- **Critérios de aceitação:** Tests passam para todos os modelos de catálogo; regressões numéricas pegas automaticamente.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S21.T03 — HDR Histograms para latência (lock-free, percentile-accurate) ✨⚠️

- **Onde:** `src/dsp/telemetry.rs` (substitui histograma bucket-linear atual).
- **Problema/Oportunidade:** Telemetria atual usa buckets lineares (S6.T01) — boa para count mas péssima para p99/p99.9. HDR Histogram (Gil Tene) usa bucket log-linear: 5 σ acurácia com 10× menos memória; já é o padrão em sistemas low-latency (Aeron, ZGC).
- **Solução técnica:**
  1. Crate `hdrhistogram` ou implementação inline (simple version, ~200 LoC).
  2. RT-safe: record via `fetch_add` em buckets pre-allocados; read em main thread.
  3. Export como Prometheus/OpenMetrics text (S21.T04 opcional).
- **Critérios de aceitação:** p99 e p99.9 reportados com erro < 1% vs cyclictest baseline.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S21.T04 — Continuous benchmark regression bot (criterion + JSON archive) ✨💡

- **Onde:** `.github/workflows/bench-regression.yml`; `benches/`.
- **Problema/Oportunidade:** Sem CI regression bot, slow performance creep passa despercebido. Criterion já produz JSON; arquivar em branch `benchmarks-archive` e comparar via gh-actions.
- **Solução técnica:**
  1. Cada PR roda subset de benches; compara vs `main` baseline; comenta diff no PR.
  2. Falha PR se regressão > 5% em hotpath (inference_bench).
- **Critérios de aceitação:** Bot ativo em PRs; histórico de benches em branch dedicado.
- **Especialista:** `implementador`.
- **Esforço:** 1 dia.

### Sprint S22 — Compiler-Grade Optimization (PGO + BOLT)

#### Tarefa S22.T01 — Profile-Guided Optimization (PGO) build pipeline ✨⚠️

- **Onde:** `Cargo.toml`; `utils/build-pgo.sh`.
- **Problema/Oportunidade:** Rustc/LLVM PGO instrumenta build → roda workload representativo → coleta profile → rebuilda com `-Cprofile-use`. Tipicamente entrega 5–15% throughput em hotpath. Já standard em Firefox, Chromium.
- **Solução técnica:**
  1. Script multi-passo: build instrumented, roda `inference_bench` + `bench` real de modelos canônicos, coleta `.profraw`, merge, rebuilda release.
  2. Release shipped com PGO opcional via `cargo build --release --features pgo`.
- **Critérios de aceitação:** Benchmark inference reduz ≥ 5% latência média em PGO build vs vanilla release.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S22.T02 — BOLT post-link layout optimization ✨💡

- **Onde:** `utils/build-bolt.sh`.
- **Problema/Oportunedade:** LLVM BOLT é a "última gota": reordena basic blocks no binário linkado para que hot paths fiquem em sequência (melhor L1i utilização). Combinado com PGO, mais 3–8%.
- **Solução técnica:**
  1. Após PGO build, coletar `perf record` em workload representativo.
  2. `llvm-bolt nam-rs -o nam-rs.bolt -data=perf.data --reorder-blocks=cache+ --reorder-functions=hfsort`.
  3. Distribuir binário `.bolt` para release.
- **Critérios de aceitação:** L1i miss rate (`perf stat`) reduz ≥ 20%; latency média -3-8%.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1 dia.

#### Tarefa S22.T03 — Kahan summation em acumuladores críticos ✨💡

- **Onde:** `src/math/gemm/dot.rs`, `dot_4x.rs` (acumuladores horizontal_sum).
- **Problema/Oportunedade:** Em LSTM de muitas amostras, drift de soma FP32 acumula erro de magnitude `~N · eps`. Kahan summation (compensated summation) reduz para `O(1)` em troca de 2 FMAs extras — tolerável fora do tightest inner loop.
- **Solução técnica:**
  1. Apenas em horizontal_sum (1× por bloco GEMM), não no inner FMA.
  2. Manter `compensation: f32` acumulador secundário.
- **Critérios de aceitação:** Drift vs scalar reference em LSTM de 1M amostras reduz ≥ 100×.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 1 dia.

---

## Épico 13 — Portabilidade & Arquiteturas de Hardware Especializadas

Objetivo: expandir nam-rs e aproveitar microarquitetura de hardware específica (AMX, AVX10, SVE2, NEON) e plataformas embarcadas ARM64, exigindo setups especiais de build e execução de testes em cloud ou hardware dedicado.

### Sprint S23 — Intel AMX & AVX10.2

#### Tarefa S23.T01 — Abertura do pipeline de build e CI para Intel AMX & AVX10.2 (via Intel SDE / Self-hosted VM) 💡

- **Onde:** `.github/workflows/` (pipelines de build/test/release).
- **Problema:** Atualmente, não há validação automatizada de compilação ou testes funcionais para as novas instruções Intel AMX e AVX10.2 em CI, aumentando o risco de regressões e quebras de build.
- **Solução técnica:**
  1. Configurar etapa de download e cache do **Intel Software Development Emulator (Intel SDE)** no pipeline de CI do GitHub Actions (usando ações como `petarpetrovt/setup-sde` ou script customizado).
  2. Executar a suite de testes unitários e de integração de AMX/AVX10.2 envelopando o binário de teste com `sde64 -spr -- cargo test --features amx-nightly`.
  3. Integrar flags de compilação no pipeline.
- **Critérios de aceitação:**
  - Pipeline de CI compila e passa nos testes unitários com emulação de CPU Sapphire Rapids (AMX) e Diamond Rapids (AVX10.2) com sucesso.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S23.T02 — Backend Intel AMX para LSTM 2-layer e WaveNet Standard (BF16) ✨🔥

- **Onde:** novo módulo `src/math/common/amx_impl.rs`; integração em `src/math/common/dispatch.rs:140-163` (novo nível `InstructionSet::Amx_Bf16` acima de `Avx512VnniBf16`).
- **Problema/Oportunidade:** Sapphire Rapids+ executa **`_tile_dpbf16ps`** (DST = A·Bᵀ + DST) em **um único ciclo de 1024 FMAs BF16** (16×64 BF16 × 64×16 BF16 → 16×16 FP32, ~2 TFLOPS BF16 por core a 2 GHz). Para LSTM `2×16` (matmul 32×80 por amostra) e WaveNet Standard (matmul de 16×16 com kernel-3), AMX entrega potencial **10–20×** speedup sobre AVX-512 VNNI BF16. A referência C++ ainda não usa AMX; nam-rs pode ser o **primeiro engine NAM com AMX nativo**.
- **Solução técnica:**
  1. **Layout AMX-friendly do encoder:** novo `weights_layout = AmxTile16x64Bf16` que organiza pesos em tiles de 16 linhas × 64 colunas (= 64 BF16 = 1 KB por tile), padding zero quando necessário. Decoder em `loader/dispatcher/lstm.rs` e `loader/dispatcher/wavenet.rs` carrega blocos em `AlignedVec<u16>` 64-aligned.
  2. **Trait `AmxBf16Math: SimdMath`** implementando `fused_add_gemv`, `fused_add_gemm_batch`, etc. Cada kernel:
     - Configurar palette 1 via `_tile_loadconfig()` (uma vez por activate).
     - `_tile_loadd::<TILE_A, STRIDE>(weights_ptr)` para tile A (16×32 BF16).
     - `_tile_loadd::<TILE_B, STRIDE>(input_ptr)` para tile B (32×16 BF16).
     - `_tile_dpbf16ps::<TILE_C, TILE_A, TILE_B>()` — acumula em FP32 no tile C.
     - `_tile_stored::<TILE_C, STRIDE>(output_ptr)` final.
  3. **AMX state preservation:** primeira chamada em activate emite `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` para habilitar XTILEDATA no kernel (Linux ≥5.16). Falha graciosa para fallback Avx512VnniBf16.
  4. **`#![feature(x86_amx_intrinsics)]`** requer nightly até estabilização. Gate via `#[cfg(feature = "amx-nightly")]` em Cargo, default off.
  5. Adicionar `RT_STATUS_AMX_ACTIVE` flag.
- **Pré-requisitos (obrigatórios — herdam invariantes da Parte I):**
  - **S3.T03** (padrão intercalado `[W_l, bias_l, hidden_init_l, cell_init_l]` por camada) — o layout AMX tile-block deve seguir a mesma disciplina sequencial para evitar o tipo de bug encoder↔decoder corrigido em LSTM.
  - **S3.T04** (padding implícito do encoder até múltiplos de bloco SIMD) — tiles AMX exigem 16-row blocks; replicar a estratégia do Interleaved-4 (zero-pad até `ceil(N/16)·16`) ao invés de tail-loops independentes.
  - **S5.T03** (flag `FLAG_HAS_CRC32` explícito) — o novo layout deve setar o flag CRC e nunca confiar em sentinel.
  - **S5.T07** (spec NAMB) — `docs/namb-spec.md` precisa ganhar seção "AmxTile16x64Bf16" antes da implementação, incluindo exemplos hex.
  - **S13.T02 (round-trip)** — cobertura obrigatória do novo layout no harness `tests/namb_v2_roundtrip.rs` (extendido) antes do merge.
  - Decisão: o novo layout dispara **bump explícito de NAMB para v3** (com `FLAG_HAS_AMX_TILE_LAYOUT` no header v3). Documentar em `docs/namb-spec.md` v3.
- **Critérios de aceitação:**
  - Em CPU Sapphire Rapids, dispatcher seleciona AMX; benchmark `inference_bench` mostra ≥8× speedup vs AVX-512 BF16 para LSTM 2×16 e ≥4× para WaveNet Standard.
  - Diferença numérica vs `ScalarRefMath` < 5e-3 (AMX usa BF16 mantissa de 7 bits, tolerância maior justifica-se).
  - `cargo test --features amx-nightly` passa cobertura golden em 4 modelos canônicos.
  - **Round-trip encode→decode do layout `AmxTile16x64Bf16` passa bit-perfect** (estende `tests/namb_v2_roundtrip.rs` de S13.T02).
  - Documentado em `docs/amx-backend.md` (setup XSAVE/permission, palette config, latência/throughput por kernel) e seção dedicada em `docs/namb-spec.md` v3.
- **Especialista:** `pesquisador-inovador` + revisão `revisor-auditor`.
- **Esforço:** 4–5 dias.

#### Tarefa S23.T03 — Dispatcher AVX10.2 (Diamond Rapids 2026) ✨⚠️

- **Onde:** `src/math/common/dispatch.rs`; novo `avx10_impl.rs` (opcional, pode reusar `avx512_impl` se ISA-equivalente).
- **Problema/Oportunidade:** Intel Diamond Rapids 2026 introduz **AVX10.2** unificando AVX-512 com novos data types: **FP16 nativo (FMA hpfp)**, **FP4** para inferência, e melhor scheduling. oneDNN 2026 já entrega `ONEDNN_MAX_CPU_ISA=AVX10_2_512_AMX_2`. Compilers ainda emergentes; estar pronto cedo é vantagem competitiva.
- **Solução técnica:**
  1. Adicionar `InstructionSet::Avx10_2` ao enum (entre AMX e AVX-512).
  2. Detecção via `is_x86_feature_detected!("avx10.2")` (estabilizar quando intrinsic landar; até lá usar `cpuid` direto).
  3. Substituir `simd_tanh_avx512`/`simd_sigmoid_avx512` por variantes **`_ph` (packed half)** — FMA FP16 nativo elimina 2× conversão F16↔F32 que dominam o hotpath de activations.
  4. Adicionar `dot_4x_fp16_avx10` kernel para Conv1D em WaveNet Feather/Nano (modelos pequenos onde overhead de conversão é proporcionalmente maior).
- **Critérios de aceitação:**
  - Dispatcher detecta AVX10.2 (gated em CPU emulado via SDE até hardware estar disponível).
  - Benchmark FP16 nativo ≥ 2× speedup em activations (sigmoid/tanh) vs AVX-512 BF16 com conversão.
  - Paridade vs `ScalarRefMath` < 1e-3.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 2–3 dias.

### Sprint S24 — Portabilidade Linux ARM64 (NEON/SVE2 & Standalone RPi5/Asahi)

#### Tarefa S24.T01 — Abertura do pipeline de build e CI para ARM64 Linux 💡

- **Onde:** `.github/workflows/` (pipelines de build/test/release).
- **Problema:** Não há automação para compilar e testar nativamente ou via cross-compilation o target Linux ARM64, impedindo o deploy confiável em sistemas como Raspberry Pi 5 e servidores baseados em ARM64.
- **Solução técnica:**
  1. Adicionar o target `aarch64-unknown-linux-gnu` à matriz de build e testes do GitHub Actions.
  2. Configurar o ambiente com cross-compilers necessários (`gcc-aarch64-linux-gnu`) ou agentes aarch64 nativos.
  3. Executar a suite de testes unitários e de integração via QEMU user mode runner ou agentes nativos no CI.
- **Critérios de aceitação:**
  - Pipeline de CI compila e passa nos testes com sucesso para o target `aarch64-unknown-linux-gnu`.
- **Especialista:** `implementador` + `pesquisador-inovador`.
- **Esforço:** 1.5 dia.

#### Tarefa S24.T02 — Backend NEON/SVE2 para processadores ARM64 Linux (Ampere, Graviton, Cortex) ✨🔥

- **Onde:** novo módulo `src/math/common/neon_impl.rs` (e `sve2_impl.rs` opcional); integração em `dispatch.rs`.
- **Problema/Oportunidade:** Ampere Altra/Graviton 4 (servidor ARM Neoverse-V2 com SVE2 256-bit) e processadores ARM64 como Cortex-A76/A78 (Raspberry Pi 5) representam alvos fundamentais para Linux. Hoje, nam-rs em ARM rodaria escalar — **inviável** para produção.
- **Solução técnica:**
  1. **NEON baseline:** trait `NeonMath` com kernels:
     - `gemv` usando `vfmaq_f32` (4-lane FMA) com 4 acumuladores.
     - `dot_product_4x` com layout interleaved-4 (já compatível com encoder atual).
     - `tanh/sigmoid` via Padé (S7.T09) — NEON ports diretos.
     - Conversão F16↔F32 via `vcvt_f16_f32` (ARMv8.2-A FP16).
  2. **SVE2 advanced:** trait `Sve2Math` para Neoverse-V1+/V2 (Ampere, Graviton 4):
     - Vectores de comprimento variável (128–2048 bits, runtime via `svcntw`).
     - `svfmla_f32_z` predicado, eliminando tail loops.
     - `svbfdot_f32` para BF16 dot (ARMv8.6-A) — análogo a `_mm512_dpbf16_ps`.
  3. **Dispatcher:** `#[cfg(target_arch = "aarch64")]` com `std::arch::is_aarch64_feature_detected!("neon")` e `("sve2")`.
  4. **`vring.rs` portabilidade:** já parcialmente coberto por S1.T04. Em Linux ARM64, `memfd_create` funciona normalmente.
  5. **Build matrix CI:** `aarch64-unknown-linux-gnu` em GitHub Actions.
- **Critérios de aceitação:**
  - `cargo test --target aarch64-unknown-linux-gnu` passa com emulação QEMU ou nativa.
  - Paridade numérica `|err| < 5e-4` vs ScalarRefMath.
- **Especialista:** `pesquisador-inovador` + `implementador`.
- **Esforço:** 3 dias.

#### Tarefa S24.T03 — Linux ARM64 standalone (Raspberry Pi 5 / Asahi Linux) ✨⚠️

- **Onde:** build matrix CI; `utils/install.sh` para apt+pipewire em Raspbian/Asahi.
- **Problema/Oportunidade:** Raspberry Pi 5 (Cortex-A76, NEON, FP16) ou Asahi Linux em Apple Silicon entregam plataforma "stomp-box" de baixo custo. Combinado com S24.T02 (NEON backend) e S16.T01 (SCHED_DEADLINE), entrega standalone hardware NAM rivalizando DIMEHEAD/Anagram.
- **Solução técnica:**
  1. Cross-compile `aarch64-unknown-linux-gnu`.
  2. PipeWire 0.10 disponível em Debian 12/Ubuntu 22.04 ARM.
  3. Smoke test em Raspberry Pi 5 OS (kernel 6.6 PREEMPT_RT custom build).
  4. Documentar tuning em `docs/raspberry-pi-5.md`: GPU bypass, CPU isolcpus, cpufreq performance.
- **Critérios de aceitação:** RPi5 com guitar interface USB roda nam-rs standalone com latência < 10ms; LSTM 1×16 sem xruns.
- **Especialista:** `pesquisador-inovador`.
- **Esforço:** 3 dias.
