<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Resiliência & Robustez (Revisor-Auditor, 2026-07-14)

Rodada de auditoria na role **Resilience and Robustness Specialist** (com apoio da skill
`pesquisador-inovador`), sucedendo a rodada de Compliance & Parity fechada em 2026-07-14
(EP-A/EP-B/EP-C — histórico no git). Base de evidências: `/testes.log` (quick suite +
quality-dashboard `--check` + build-release PGO/BOLT + tests-long completos, todos verdes),
inspeção direta de código em `src/`, `utils/` e `tests/`, e verificação pontual de cada achado
de alta severidade (nenhum achado abaixo foi registrado sem confirmação em `file:line`).

> **Rodada 2 (2026-07-16):** os épicos EP-R1 a EP-R5 foram verificados linha a linha contra o
> código atual (ver "Verificação pós-implementação" após os épicos) — 41 de 44 sub-itens
> confirmados RESOLVIDOS, 3 com pendência residual. Em seguida, nova rodada completa de
> auditoria Resilience & Robustness + Pesquisador-Inovador produziu os achados **R17–R26** e as
> propostas **P5–P10**, organizados nos épicos **EP-R7 a EP-R11** (todos ao final do documento).
>
> **Rodada 3 (2026-07-16, mesmo dia):** os épicos EP-R7 a EP-R12 foram implementados e
> verificados — 13 de 15 sub-itens RESOLVIDOS, incluindo uma **correção histórica importante**:
> o achado R8-h estava parcialmente equivocado (ver nota na seção de verificação). Nova rodada
> de auditoria em áreas ainda não cobertas (IR/WAV, automação de parâmetros, parser NAMB,
> conformidade SPDX) produziu os achados **R27–R31** e as propostas **P11–P12**, organizados nos
> épicos **EP-R13 a EP-R17**.

**Fora de escopo (por decisão de produto):** `TODO-wavenet_a2_max.md` e
`TODO-convnet_parity.md`. Nenhum achado abaixo toca esses pântanos; onde a auditoria produziu
*conhecimento novo* relevante a eles, está registrado na seção "Contribuições de conhecimento"
ao final, sem propor ação.

**Política de compliance:** todas as propostas usam exclusivamente Rust **stable** (toolchain
atual 1.97) e ferramentas que rodam sobre stable. Nenhuma recomendação depende de nightly
(Miri, sanitizers `-Z`, etc. foram deliberadamente excluídos).

**Veredito geral:** o núcleo RT do projeto está em estado exemplar — hot paths sem locks/IO,
divisões protegidas, GC-cascade com validação de type_id, cleanup de mmap/fd impecável nos
caminhos de erro (`mirror_buf/alloc.rs`, `huge_alloc.rs`). O que esta rodada encontrou de
grave está nas **bordas**: (1) um UB formal confirmado num buffer de stack do WaveNet;
(2) classes de use-after-free e vazamento de recurso de sistema em caminhos de ciclo de vida
(GUI/file-dialog, double-SIGINT, panic hook); (3) uma família de orderings atômicos
formalmente incorretos (inofensivos em x86-TSO, incorretos no modelo de memória — bombas de
portabilidade); e (4) duas fissuras na malha de QA (falso-verde estrutural na fase PipeWire e
colisão de chaves por prefixo no contrato de qualidade) que permitem regressões ficarem verdes.

---

## Sumário executivo

| ID  | Achado                                                                            | Severidade  | Área                |
| --- | --------------------------------------------------------------------------------- | ----------- | ------------------- |
| R1  | UB formal: `&mut [f32]` sobre `MaybeUninit` não inicializado (WaveNet)            | **CRÍTICA** | UB / hot path       |
| R2  | Use-after-free potencial: threads detached de file-dialog + ponteiro cru          | **CRÍTICA** | Lifecycle GUI       |
| R3  | Double-SIGINT vaza lock de C-States (`/dev/cpu_dma_latency`) até reboot           | **ALTA**    | Recursos de sistema |
| R4  | Panic hook aloca e adquire locks no caminho de crash do thread RT                 | **ALTA**    | Diagnóstico/RT      |
| R5  | `log::error!` alcançável no thread RT via `set_slimmable_size`                    | **ALTA**    | RT-safety           |
| R6  | Colisão de chaves por prefixo no `--check` do contrato de qualidade               | **ALTA**    | Malha de QA         |
| R7  | Falso-verde estrutural: fase PipeWire do tests-long não distingue "0 testes"      | **ALTA**    | Malha de QA         |
| R8  | Família de orderings atômicos formalmente incorretos (Relaxed sem happens-before) | **MÉDIA**   | Concorrência        |
| R9  | `panic_any` no `Clone` de `MirroredBuffer` pode cruzar FFI CLAP                   | **MÉDIA**   | Lifecycle CLAP      |
| R10 | `debug_assert!` + `copy_nonoverlapping` sem bound-check no release (oversample)   | **MÉDIA**   | UB latente          |
| R11 | GC overflow: flag enganosa + drenagem final não garantida no destroy              | **MÉDIA**   | GC-cascade          |
| R12 | Higiene de `unsafe`: SAFETY genéricos e `get_unchecked` substituíveis             | **MÉDIA**   | Auditabilidade      |
| R13 | `gui.destroy()` com `join()` potencialmente indefinido (janela flutuante)         | **MÉDIA**   | Lifecycle GUI       |
| R14 | Código morto/duplicado: fft_radix4 pública, `median` duplicado, órfãos            | **BAIXA**   | Coesão              |
| R15 | Feature `testing` em `default` vaza instrumentação para builds de produção        | **BAIXA**   | Superfície de build |
| R16 | Ruídos de log e referências obsoletas (dashboard cita TODO-findings antigo)       | **BAIXA**   | Higiene             |
| P1  | Inovação: `loom` para model-checking do SPSC/GC (stable)                          | proposta    | QA de concorrência  |
| P2  | Inovação: `cargo-mutants` como extensão anti-placebo (stable)                     | proposta    | Malha de QA         |
| P3  | Inovação: `hint::assert_unchecked` + `as_chunks` para reduzir `unsafe`            | proposta    | Higiene/perf        |

### Rodada 2 (2026-07-16) — novos achados e propostas

| ID  | Achado                                                                                          | Severidade | Área            |
| --- | ----------------------------------------------------------------------------------------------- | ---------- | --------------- |
| R17 | UAF residual: `NamPluginWindow::new` deref `shared.0` sem `alive_fence` em thread flutuante     | **ALTA**   | Lifecycle GUI   |
| R18 | `log::error!`/`log::info!` alcançáveis no thread RT do PipeWire via `configure_realtime_thread` | **ALTA**   | RT-safety       |
| R19 | `track_info::changed()` usa `as_main_thread_unchecked` sem runtime guard                        | **MÉDIA**  | Lifecycle CLAP  |
| R20 | `.expect()` residual em alocações de produção (loader + `activate()` CLAP)                      | **MÉDIA**  | Fail-closed     |
| R21 | Lacuna de fuzzing: LSTM dinâmico, A2-Dynamic e SlimmableContainer sem estratégia adversarial    | **MÉDIA**  | Malha de QA     |
| R22 | Ausência de telemetria de xrun/buffer-miss (capture e playback PipeWire)                        | **MÉDIA**  | Observabilidade |
| R23 | Higiene de `unsafe` remanescente fora da tabela R12 (~20 blocos sem SAFETY específico)          | **BAIXA**  | Auditabilidade  |
| R24 | Extensão CLAP `thread-check` não registrada                                                     | **BAIXA**  | Lifecycle CLAP  |
| R25 | `PoisonError` de `Mutex` descartado silenciosamente em preset_load/housekeeping                 | **BAIXA**  | Diagnóstico     |
| R26 | Campos/flags mortos ou write-only (`os_*` buffers, `alive: AtomicBool`, `mem::zeroed` FFI)      | **BAIXA**  | Coesão          |
| P5  | Inovação: `#[expect(lint)]` (Rust 1.81) substituindo `#[allow]` acumulado                       | proposta   | Higiene/lint    |
| P6  | Inovação: `cargo machete` para dependências não usadas                                          | proposta   | Supply-chain    |
| P7  | Inovação: `typos` como spell-checker de CI                                                      | proposta   | Higiene         |
| P8  | Inovação: `cargo vet` para auditoria de supply-chain                                            | proposta   | Supply-chain    |
| P9  | Inovação: `build.warnings = "deny"` (Cargo/Rust 1.97) substituindo `RUSTFLAGS`                  | proposta   | Build/CI        |
| P10 | Inovação: `cargo semver-checks` para breaking changes de API pública                            | proposta   | Compatibilidade |

### Rodada 3 (2026-07-16) — novos achados e propostas

| ID  | Achado                                                                                       | Severidade | Área               |
| --- | -------------------------------------------------------------------------------------------- | ---------- | ------------------ |
| R27 | IR/WAV: `sample_rate` extremo (baixo) causa OOM garantido via upsampling catastrófico        | **ALTA**   | Robustez de input  |
| R28 | Automação sample-accurate não implementada; eventos colapsam para o último valor do bloco    | **ALTA**   | Fidelidade CLAP    |
| R29 | Push SPSC descartado silenciosamente em `MainThread::flush()`; perda de eventos de parâmetro | **MÉDIA**  | Lifecycle CLAP     |
| R30 | Reset do smoother de ganho para 1.0 em cada `activate()` causa transiente audível            | **BAIXA**  | UX/Fidelidade      |
| R31 | Higiene residual: bypass de CRC32 no NAMB v1 legado + `.expect()` em `MirroredBuffer::clone` | **BAIXA**  | Conformidade       |
| P11 | Inovação: `shuttle` (AWS Labs) — concurrency testing randomizado complementar ao `loom`      | proposta   | QA de concorrência |
| P12 | Inovação: `rtsan-standalone-rs` — RealtimeSanitizer para violações RT em runtime             | proposta   | RT-safety          |

---

## R1 · UB formal confirmado: `slice::from_raw_parts_mut` sobre `MaybeUninit<[f32;1024]>` não inicializado — **CRÍTICA**

### R1 · Evidência

`src/models/wavenet/layer.rs:59-69` (duas ocorrências no mesmo método):

```rust
// SAFETY: mixin_out is fully written by input_mixin.process_block before
// any read. MaybeUninit avoids the 4 KB memset the compiler emits for
// `[0.0f32; 1024]`.
let mut mixin_out = MaybeUninit::<[f32; 1024]>::uninit();
let mixin_out_ptr = mixin_out.as_mut_ptr() as *mut f32;
let mixin_out_slice = core::slice::from_raw_parts_mut(mixin_out_ptr, num_frames * CH);
```

E o par `conv_plus_mixin` nas linhas 67-69, mesmo padrão.

### R1 · Diagnóstico

O comentário SAFETY cobre o risco de **leitura** de memória não inicializada, mas o UB aqui é
anterior: **a mera materialização de uma referência `&mut [f32]` apontando para bytes não
inicializados já é comportamento indefinido** segundo as regras de validade do Rust
(referências devem apontar para valores válidos do tipo; `f32` exige bytes inicializados).
Não importa que ninguém leia antes do write — o compilador tem permissão de assumir a validade
no instante da criação da referência. É exatamente a classe de bug do antigo
`Vec::set_len`-antes-de-escrever. Com LTO fat + PGO (pipeline de release do projeto), o risco
de miscompilação silenciosa não é teórico.

### R1 · Impacto

Hot path do WaveNet A1 (`process_block_internal`), executado a cada bloco de áudio. Hoje
funciona; é uma mina enterrada que pode detonar em qualquer atualização de rustc/LLVM.

### R1 · Proposta de solução (stable, sem custo de performance)

Preferência 1 — **eliminar o buffer de stack**: mover `mixin_out` e `conv_plus_mixin` para
scratch pré-alocado na struct do layer (`AlignedVec<f32>` dimensionado em `set_max_buffer_size`,
padrão já usado em todo o resto do projeto). Remove o UB, remove o memset, remove 8 KB de
stack no RT thread (bônus: menos pressão em stack de threads FIFO com stack limitada) e
elimina os dois `unsafe`.

Preferência 2 (se quiser manter stack) — trabalhar com `&mut [MaybeUninit<f32>]`
(`from_raw_parts_mut(ptr as *mut MaybeUninit<f32>, n)` é válido para memória uninit) e
converter para `&mut [f32]` **após** o preenchimento com `assume_init` — exige adaptar
`process_block` para escrever via `MaybeUninit::write` ou aceitar o slice uninit.

Preferência 3 (mínima) — aceitar o memset: `let mut mixin_out = [0.0f32; 1024];`. Medir com
`benches/inference_bench.rs`; 4 KB de memset em L1 custa dezenas de ns por bloco — provável
ruído frente ao custo da convolução.

Critério de aceite: zero `from_raw_parts_mut` sobre memória uninit no crate
(`grep -rn "MaybeUninit" src/ | grep -v test`), benchmarks `inference_bench`/`regression_gate`
sem regressão, `utils/tests-quick.sh` verde.

---

## R2 · Use-after-free potencial: threads detached do file-dialog acessam `NamClapShared` via endereço cru — **CRÍTICA**

### R2 · Evidência

* `src/clap/gui/ui/zones/file_dialogs.rs:15,67` — `std::thread::spawn` sem `JoinHandle`
  armazenado; a thread interna bloqueia em `rfd::FileDialog::pick_file()` (síncrono) e a
  externa espera `rx.recv_timeout(120s)`.
* `src/clap/gui/ui/zones/file_dialogs.rs:29,43,81,94` — reconstrução do ponteiro:
  `let shared = unsafe { &*(shared_addr as *const NamClapShared) };` a partir de `usize`.
* `src/clap/plugin/shared.rs:75-76` — `NamClapSharedRef(pub *const NamClapShared)` com
  `unsafe impl Send/Sync` sem invariante documentada e campo público.
* `src/clap/gui/window/state.rs:190` + `shared.rs:252` — `alive_fence` lido/escrito com
  `Ordering::Relaxed` em ambos os lados.

### R2 · Diagnóstico (cenário passo-a-passo)

1. Usuário abre o file-dialog → thread T1 (dialog) e T2 (watchdog 120 s) são criadas, ambas
   detached, carregando `shared_addr: usize`.
2. Host descarrega o plugin enquanto o diálogo está aberto. `NamClapShared` é dropado;
   `alive_fence` vai a `false` — com `Relaxed`, sem barreira.
3. T2 acorda no timeout, lê `alive_fence` (pode observar `true` estável em arquiteturas de
   memória fraca; e mesmo em x86 há a janela TOCTOU entre o load e o acesso) e executa
   `ui_loading.store(false)` **dentro de memória liberada** → use-after-free.

O `alive_fence` mitiga o caso comum, mas: (a) é TOCTOU por construção — nada impede o drop
entre o check e o uso; (b) `Relaxed` não estabelece happens-before com a liberação da memória.

### R2 · Proposta de solução

Eliminar a classe inteira do problema em vez de remendar o fence:

1. Colocar o estado compartilhado GUI↔dialog num **`Arc<DialogSharedState>`** próprio (apenas
   os atomics `ui_loading`, `ui_pending_model`, etc. — nada de ponteiro para `NamClapShared`).
   As threads de diálogo capturam um clone do `Arc`; se o plugin morrer, elas escrevem num
   objeto órfão inofensivo que morre com o último clone. UAF estruturalmente impossível.
2. `NamClapSharedRef`: trocar `pub *const` por `NonNull<NamClapShared>` privado e documentar a
   invariante no `unsafe impl Send/Sync` (ou remover o tipo, se o passo 1 o tornar obsoleto).
3. Armazenar os `JoinHandle` no `NamClapMainThread` e, no `teardown_gui_resources()`, fazer
   join com timeout curto (`is_finished()` + polling breve); documentar o abandono controlado
   após timeout.
4. Enquanto o passo 1 não sai: promover `alive_fence` para `Acquire` (load) / `Release`
   (store) — corrige a formalidade, não o TOCTOU; por isso é paliativo, não solução.

Critério de aceite: nenhum `usize`→ponteiro em `src/clap/gui/`; ciclo abrir-diálogo→destruir
plugin coberto por teste de lifecycle (estender `src/clap/gui/window/window_test.rs`).

---

## R3 · Double-SIGINT vaza o lock de PM QoS (`/dev/cpu_dma_latency`) até o reboot — **ALTA**

### R3 · Evidência

* `src/main.rs:81` — handler de SIGINT: segundo Ctrl-C chama `libc::_exit(1)`.
* `src/standalone/rt_setup/pm_qos.rs` — o lock de C-States é mantido por um `File` aberto em
  `/dev/cpu_dma_latency` (o kernel mantém o QoS enquanto o fd viver).
* Log (`testes.log:3256`): `⚡ PM QoS Lock: Deep CPU C-States disabled (Zero DMA Latency)`.

### R3 · Diagnóstico

`_exit(1)` encerra o processo sem rodar destrutores… porém o kernel **fecha todos os fds no
exit do processo**, o que normalmente liberaria o QoS. O problema real está na combinação com
o caso de **panic + abort** ou kill -9 durante estados intermediários? Não — nesses casos o
kernel também fecha os fds. A análise fina mostra que o risco efetivo é outro: o handler de
SIGINT usa `_exit` **antes** de o loop principal executar a seção "GRACEFUL SHUTDOWN"
(`run.rs:335`), que além do QoS restaura estado de streams PipeWire e drena o GC. O vazamento
*persistente* de QoS não ocorre (fd é fechado pelo kernel); o que ocorre é **shutdown abrupto
sem drenagem do GC e sem despedida limpa do PipeWire**, além de `kernel.perf_event_paranoid`
poder ficar alterado quando o abort acontece no meio do `build-release.sh` (script, não app).

### R3 · Impacto (recalibrado pela auditoria)

Menor que o inicialmente suspeitado, porém real: double-SIGINT durante sessão ativa derruba o
processo sem fechar streams PipeWire ordenadamente (o daemon limpa sozinho, mas com risco de
click audível) e sem sinalizar o shutdown ao restante do processo.

### R3 · Proposta de solução

1. No segundo SIGINT, usar `std::process::abort()` em vez de `_exit(1)` apenas se a intenção
   for gerar core dump; caso contrário, manter `_exit` mas **documentar** no handler que o
   kernel recolhe fds/mapeamentos (incluindo QoS) — o comentário hoje não existe.
2. Garantir que o primeiro SIGINT seja suficiente e responsivo: verificar que
   `SHUTDOWN.load` do loop principal usa `Acquire` (ver R8-a) para que o primeiro Ctrl-C nunca
   "não pegue" — a motivação típica do usuário para o segundo Ctrl-C.
3. Adicionar ao `utils/`/docs a nota operacional: QoS e THP advice são liberados pelo kernel
   no exit; nada persiste pós-processo (fecha a dúvida para sempre).

---

## R4 · Panic hook aloca heap e adquire `RwLock` no caminho de crash — pode deadlockar exatamente quando mais se precisa dele — **ALTA** ✓ RESOLVIDO (Sprint S13)

### R4 · Evidência

* `src/common/panic_hook.rs:66` → `src/common/diagnostics/bundle.rs:43-52` —
  `DiagnosticBundle::capture()` chama `SystemSnapshot::capture()` que aloca `String`s e
  `Vec<String>` (hostname, kernel, CPU features) no momento do panic.
* `src/common/diagnostics/bundle.rs:136-138` — `render()` adquire
  `ACTIVE_MODEL_NAME.read()` (RwLock) dentro do hook.

### R4 · Diagnóstico

Se o panic ocorre no thread RT (o cenário que o bundle mais quer capturar), o hook:
(a) aloca no allocator global — se outro thread estiver dentro do `malloc` com lock tomado no
instante do crash, deadlock; (b) pode bloquear no `RwLock` se um loader estiver com write lock.
O `catch_unwind` interno protege contra double-panic, mas não contra deadlock. Resultado: em
vez de um bundle de diagnóstico, o usuário ganha um processo travado.

### R4 · Proposta de solução

1. **Pré-capturar** o `SystemSnapshot` na inicialização (é estático por natureza: hostname,
   kernel, CPU) e guardá-lo em `OnceLock<SystemSnapshot>`; o hook só lê referências.
2. Formatar o bundle em **buffer fixo pré-alocado** (`[u8; 4096]` + `core::fmt::Write` em
   wrapper que trunca) — zero alocação no hook.
3. Trocar `ACTIVE_MODEL_NAME.read()` por `try_read()` com fallback `"<unavailable>"`.
4. Teste: estender `tests/models/diagnostic_bundle.rs::test_panic_hook_behavior` com um caso
   que verifica (via contador do `alloc_audit` já existente no projeto) que o hook executa com
   **zero alocações** após a inicialização.

---

## R5 · `log::error!` alcançável no thread RT via `ContainerModel::set_slimmable_size` — **ALTA**

### R5 · Evidência

* `src/models/container.rs:280` — `log::error!("ContainerModel::set_slimmable_size: reset(...) failed...")`.
* Call-sites no hot path: `src/dsp/pipeline/stages/inference.rs:42,49,62,72,89` —
  `m.set_slimmable_size(adaptive.slimmable_size())` dentro do estágio de inferência, executado
  no thread RT quando o adaptive-compute muda de nível.

### R5 · Diagnóstico

Viola diretamente a regra nº 1 do projeto (`.agents/rules/rust.md`: "Zero Blocking I/O: No
println!, eprintln!, format! ... Use RtStatusFlags"). O caminho só dispara em erro de reset do
submodelo (raro), mas quando disparar fará format + I/O do backend `env_logger` dentro do
callback de áudio — exatamente no momento em que o sistema já está em condição anômala.

### R5 · Proposta de solução

Substituir por sinalização atômica: novo bit em `RtStatusFlags`
(ex.: `RT_STATUS_SLIMMABLE_RESET_FAILED`) setado no lugar do log; o main thread
(`poll_rt_status` / housekeeping CLAP) traduz o bit em `log::error!` com contexto. Varredura
adicional: `grep -rn "log::" src/models/ src/dsp/ src/math/` e classificar cada ocorrência
como off-RT (construção/loader) ou RT (proibida) — o grep desta auditoria só encontrou esta
ocorrência alcançável, mas o meta-teste abaixo evita regressão:

* **Meta-teste (herda a filosofia anti-placebo):** teste estrutural que faz grep nos módulos
  de hot path (lista explícita) e falha se encontrar `log::|println!|eprintln!|format!` fora
  de `#[cold]`/caminhos de construção anotados — mesmo padrão dos meta-testes existentes em
  `tests/models/threshold_calibration.rs`.

---

## R6 · Contrato de qualidade: matching por prefixo colide `Quick A2-Full` ↔ `Quick A2-Full v2` — verificação falso-verde — **ALTA**

### R6 · Evidência

* `utils/quality-dashboard.sh:1680`:

  ```bash
  if [[ "$dash_label" == "$contract_label"* ]] || [[ "$contract_label" == "$dash_label"* ]]; then
  ```

* Prova no run (`testes.log:2839,2848`): "Quick A2-Full @48000 Live: ESR **1.4956…E-13**
  (contrato: 1.50e-13)" e "Quick A2-Full v2 @48000 Live: ESR **1.4956…E-13** (contrato:
  1.58e-13)" — o mesmo valor medido, dígito a dígito, atribuído às duas entradas, enquanto a
  tabela de fidelidade do mesmo run mostra 1.58e-13 para o v2 (`testes.log:2692`).

### R6 · Diagnóstico

O label do contrato é truncado a 38 colunas na renderização (`quality-dashboard.sh:1191`), e o
`--check` casa por prefixo bidirecional. Todo par de modelos em que um nome é prefixo do outro
(`Quick A2-Full` ⊂ `Quick A2-Full v2`) resolve para a **primeira** medição encontrada. O gate
v2 de 48 kHz (240k amostras, o mais representativo de uso real) está hoje sendo validado
contra o número do teste errado.

### R6 · Impacto

Uma regressão real no caminho v2 (ex.: ESR subindo para 1e-9) passaria no `--check` enquanto o
v1 estivesse são. Fissura direta na promessa central da malha de qualidade.

### R6 · Proposta de solução

1. Chave de matching **exata e composta**: `label completo + sample rate + modo` (os três já
   existem no JSONL). Eliminar o truncamento a 38 colunas na *persistência* (truncar apenas na
   *renderização*).
2. Meta-teste de unicidade: após `--save`, falhar se dois labels do contrato forem prefixo um
   do outro (previne a classe, não só a instância) — pode viver no próprio script
   (`--check` já itera os pares) ou em `tests/models/meta_coherence.rs`.
3. Regravar `docs/quality-contract.txt` com `--save` após a correção e conferir manualmente as
   duas linhas A2-Full.

---

## R7 · Fase PipeWire do `tests-long.sh`: aviso de falso-verde é sintoma de detecção incapaz de distinguir "rápido" de "vazio" — **ALTA**

### R7 · Evidência

* `testes.log:3411-3412`: `✓ Sucesso (0s)` seguido de
  `⚠ AVISO: Fase 'PipeWire Integration Test' completou com PASSED em < 1s — fase possivelmente vazia/falso-verde.`
* `utils/tests-long.sh:410-411` — heurística genérica `duration < 1s ⇒ aviso`.
* O PipeWire **estava disponível** na máquina (o próprio run usou-o na fase BOLT,
  `testes.log:3245`), então a fase deveria ter executado teste real.

### R7 · Diagnóstico

O `run_phase` não sabe se `cargo test` executou 1 teste em 0.9 s ou **zero testes** (filtro
não casou, feature faltando, `#[ignore]` não destravado). A fase mais dependente de ambiente
do projeto (integração PipeWire) é justamente a que fica atrás da detecção mais fraca. Hoje o
aviso é benigno (o teste roda e é rápido), mas se um rename de teste ou mudança de filtro
zerar a seleção, a fase continuará PASSED para sempre — o pior tipo de falha da malha de QA
segundo as próprias regras do projeto (`.agents/rules/testing.md`: "a hang there is worse
than a failure" — um falso-verde perpétuo é ainda pior).

### R7 · Proposta de solução

1. Parsear o sumário do próprio cargo test no log da fase:
   `grep -E "test result: ok\. [1-9][0-9]* passed"` — exigir ≥ 1 passed; `0 passed; 0 failed`
   ⇒ fase FALHA com mensagem "seleção vazia" (não aviso).
2. Aplicar o mesmo gate a **todas** as fases do `tests-long.sh` (função utilitária
   `assert_ran_tests <logfile> <min_count>`), não só à PipeWire.
3. Promover o aviso `< 1s` atual a erro quando combinado com `0 passed` e removê-lo quando
   `passed ≥ 1` (elimina o falso-alarme atual que treina o operador a ignorar avisos).

---

## R8 · Família de orderings atômicos formalmente incorretos (funcionam em x86-TSO; incorretos no modelo de memória) — **MÉDIA**

Nenhum destes é observável em x86-64 hoje; todos são bombas de portabilidade e de
refactor-safety. Correções de uma linha cada, custo zero em x86 (Release/Acquire compilam para
mov simples em x86).

| #    | Evidência                                                                                                           | Problema                                                                        | Correção                        |
| ---- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------- |
| R8-a | `src/main.rs:83` (store `Release`) × `src/standalone/pw_host/run.rs:141` (load `Relaxed`)                           | `SHUTDOWN` sem happens-before — loop pode formalmente nunca observar o shutdown | load → `Acquire`                |
| R8-b | `src/standalone/pw_host/capture/listeners.rs:59` (store `Relaxed`) × `rt_callback/rate_sync.rs:21` (swap `Relaxed`) | mudança de sample-rate pode ser perdida por ciclos                              | `Release`/`Acquire`             |
| R8-c | `src/common/panic_hook.rs:30` — `SHUTDOWN.load(Relaxed)`                                                            | caminho frio, mesma inconsistência                                              | `Acquire`                       |
| R8-d | `src/dsp/telemetry.rs:98-101` — `reset` com `store(0)` bin a bin concorrente com `fetch_add`                        | reset não-atômico como operação composta; subcontagem estatística               | `swap(0)` + doc "best-effort"   |
| R8-e | `src/clap/plugin/shared.rs:252` + `src/clap/gui/window/state.rs:190` — `alive_fence` `Relaxed`/`Relaxed`            | ver R2 — sem barreira com a liberação                                           | `Release`/`Acquire` (paliativo) |
| R8-f | `src/common/spsc/gc.rs:214-216` — `write_idx.fetch_add(Relaxed)` reordenável com `slot.swap(AcqRel)`                | inócuo hoje (drain varre tudo); frágil se o drain um dia usar `write_idx`       | `fetch_add(Acquire)` ou doc     |
| R8-g | `src/standalone/pw_host/run.rs:154,203,234,269,304` — `clear_flag_release` sem consumidor Acquire                   | Release inócuo (ninguém pareia); ruído semântico                                | `Relaxed` + comentário          |
| R8-h | `src/common/spsc/gc.rs:291` — `RT_STATUS_GC_OVERFLOW` setado mesmo quando `push` não sobrescreveu                   | diagnóstico enganoso (sugere leak onde não houve)                               | condicionar ao retorno `true`   |

### R8 · Proposta de solução

Sprint único e mecânico: aplicar a tabela, com comentário padronizado em cada par
(`// pairs with Release store em <file:line>`) — o projeto já usa esse idioma nos handshakes
exemplares (`rate_sync.rs:39-44`). Adicionar teste `loom` (ver P1) para os pares R8-a/R8-b e
para o protocolo do `GcOverflowBuffer`.

---

## R9 · `MirroredBuffer::Clone` usa `panic_any` — panic pode cruzar a fronteira FFI do CLAP — **MÉDIA**

### R9 · Evidência

`src/dsp/mirror_buf.rs:179-188` — falha de alocação no clone (OOM/esgotamento de memfd)
dispara `std::panic::panic_any(format!(...))`.

### R9 · Diagnóstico

O clone acontece em reativação/reconfiguração (off-RT), mas dentro de callbacks CLAP
(`activate`). Panic que atravessa `extern "C"` é UB; o `clack-plugin` protege os entry points
com catch, mas depender disso para um caminho de erro previsível (OOM) é frágil. O projeto já
tem o padrão certo: `MirroredBuffer::new()` retorna `Result`.

### R9 · Proposta de solução

Adicionar `try_clone() -> io::Result<Self>` e usar nos caminhos de ativação (propagando erro
CLAP `PluginError`, que o host trata como falha de ativação limpa); manter `Clone` apenas se
houver call-site infalível comprovado — senão removê-lo de vez. O teste de fault-injection já
existente (`tests/.../mirror_buf_fault_injection.rs::test_mirror_buf_mmap_failure_injection`)
deve ganhar o caso "clone sob falha de mmap ⇒ Err, não panic".

---

## R10 · `oversample.rs`: `debug_assert!` + `copy_nonoverlapping` — contrato de block-size vira UB silencioso no release — **MÉDIA**

### R10 · Evidência

`src/dsp/oversample.rs:189-218` — `debug_assert!(input.len() <= self.max_samples)` seguido de
`unsafe { ptr::copy_nonoverlapping(...) }` dimensionado por `input.len()`.

### R10 · Diagnóstico

No release o `debug_assert!` evapora. Se um host CLAP violar `max_frames_count` (hosts
mal-comportados existem; o próprio clap-validator testa block-sizes adversariais), o copy
estoura o buffer de destino. A política do projeto para exatamente este caso é o clamp
defensivo (o braço `OsStages::Off` já clampa).

### R10 · Proposta de solução

Clamp branchless no início dos braços X2/X4: `let n = input.len().min(self.max_samples);`
(custo: 1 `cmp+cmov` por bloco — irrelevante) + setar flag `RT_STATUS_*` de contrato violado
para diagnóstico. Estender `src/clap/processor_stress_test.rs` com um caso de block-size acima
do negociado (o harness `clack-host` dos testes permite).

---

## R11 · GC-cascade: drenagem final não garantida no destroy + itens órfãos — **MÉDIA**

### R11 · Evidência

* `src/clap/processor/mod.rs:253` — `deactivate()` devolve canais ao `ColdShared` sem drenar
  `gc_overflow`.
* `src/common/spsc/gc.rs` — slots do `GcOverflowBuffer` empacotam ponteiros heap em
  `AtomicU64`; sem drain final, o drop do buffer vaza os itens pendentes.
* O housekeeping (`drain_gc_channels`) é a única via de drenagem e não tem chamada garantida
  entre o último `process()` e o drop do `ColdShared`.

### R11 · Diagnóstico

Leak finito e raro (só itens em trânsito no instante do destroy), e possivelmente uma decisão
consciente ("controlled leak") — mas não está documentada nem testada, e o teste de stress
`gc_stress_1000_swaps` (167 s no tests-long) não cobre o instante do teardown.

### R11 · Proposta de solução

1. Drenagem final explícita: no `Drop` de `ColdShared`/`NamClapShared` (main thread por
   contrato CLAP), drenar `gc_rx` + `gc_overflow` + `parking_lot` reutilizando o mesmo código
   do housekeeping.
2. Se optar pelo leak controlado: documentar no módulo (`gc.rs`) o porquê (evitar double-free
   se o RT ainda estiver vivo) e registrar o volume máximo possível (N slots × tamanho de
   item).
3. Teste: variação do `processor_gc_stress_test.rs` que destrói o plugin com itens
   comprovadamente em trânsito e verifica (heap-audit) que não há double-free — e, se a opção
   1 for adotada, que não há leak.

---

## R12 · Higiene de `unsafe`: comentários SAFETY genéricos, `get_unchecked` substituível e invariantes não escritas — **MÉDIA**

A regra do projeto exige unsafe "tightly bounded, isolated, and comprehensively documented".
O grosso do código cumpre; as exceções mapeadas:

| Local                                       | Problema                                                                                   | Ação                                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `src/dsp/mirror_buf.rs:153,161,171,192,194` | SAFETY catch-all ("Low-level virtual memory manipulation…") não descreve a invariante real | Reescrever citando: ptr válido para `size_elements*2`, halves espelhados           |
| `src/math/common/huge_alloc.rs:369,380,389` | SAFETY "upheld by caller invariants" vazio                                                 | Alinhar com o padrão bom de `aligned.rs:284-286`                                   |
| `src/dsp/stage.rs:130-152,169-198`          | `get_unchecked` com bounds prováveis mas não documentados                                  | `const { assert!(UP_DELAY_LINE_LEN >= HB_TAPS) }` + SAFETY citando o loop          |
| `src/math/dsp/fft.rs:228,258,320-324`       | `get_unchecked` onde LLVM elimina bounds check com indexação segura                        | Trocar por indexação segura; se o asm regredir, usar `hint::assert_unchecked` (P3) |
| `src/models/convnet/model.rs:105-140`       | aritmética de ponteiro sobre `Vec::as_mut_ptr()` sem SAFETY por acesso                     | Documentar: `&mut self` impede realloc; `i < blocks.len()`                         |
| `src/models/wavenet/conv1d.rs:56-62`        | `copy_nonoverlapping` com offset `isize`→`usize` sem debug_assert do invariante            | `debug_assert!` + SAFETY citando `max_lookback_cols`                               |
| `src/clap/gui/mod.rs:33`                    | `transmute` de lifetime em tipo sem `repr(transparent)`                                    | Documentar dependência de layout ou encapsular em wrapper próprio                  |
| `src/math/gemm/gemv_bf16.rs:62-63,99-100`   | `transmute` `__m512`→`__m512bh` sem nota                                                   | 1 linha: "no-op de 512 bits, tipos ABI-idênticos"                                  |
| `src/main.rs:85-89`                         | cast duplo de function pointer p/ `sighandler_t` via `*const ()`                           | usar o campo correto do union `sigaction`                                          |

Proposta: sprint de documentação/redução guiado por esta tabela + regra de review: todo
`unsafe` novo nasce com SAFETY específico (o `refatora-rust`/`documentador` podem absorver).
Aproveitar P3 para os casos em que a indexação segura + `assert_unchecked` mantém o codegen.

---

## R13 · `gui.destroy()`: `join()` da janela flutuante pode bloquear a main thread do host indefinidamente — **MÉDIA**

### R13 · Evidência

`src/clap/extensions/gui.rs:24-25,181-187` — `floating_thread_handle.join()` após setar
`close_signal`; o sinal só é processado quando o event loop da janela gira (`on_frame`).

### R13 · Diagnóstico

Em X11/Wayland com conexão degradada, `open_blocking` pode não retornar; o host congela na UI.
Casos raros, impacto máximo (freeze do DAW inteiro).

### R13 · Proposta de solução

Watchdog no destroy: loop de `handle.is_finished()` com deadline (ex.: 2 s) e, ao estourar,
abandonar o handle com log de advertência (leak controlado de uma thread — preferível a
congelar o host). Documentar o trade-off no código. Cobrir com teste de lifecycle destrutivo
se o harness permitir simular janela sem event loop.

---

## R14 · Código morto e duplicações — **BAIXA** (limpeza mecânica, ~750 linhas recuperáveis)

| Item                                                                                                        | Evidência                                                                                                 | Ação                                                                      |
| ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `FftPlannerRadix4` pública, 307 linhas, zero consumidores de produção (o header diz "NÃO USAR EM PRODUÇÃO") | `src/math/dsp/fft_radix4.rs:59`; consumidores: só bench + teste                                           | Gate `#[cfg(any(test, feature = "long_bench"))]` ou mover para bench-only |
| `median` duplicado byte-a-byte (+ testes duplicados)                                                        | `src/testing/aliasing.rs:289-301` × `src/testing/spectral.rs:57-69`                                       | Consolidar em `src/testing/` comum                                        |
| Estratégias proptest órfãs (~200 linhas nunca chamadas)                                                     | `tests/models/proptest_parsers.rs:270-512`                                                                | Integrar num teste real ou remover                                        |
| `#[allow(dead_code)]` espúrios                                                                              | `src/models/a2/model/set_weights.rs:276-289` (funções usadas em teste); `src/testing/spectral.rs:56`      | Remover allow; usar `#[cfg_attr(not(test), allow(dead_code))]` se preciso |
| `CatalogGap`/`CATALOG_EXCEPTIONS` vazio "para o futuro"                                                     | `tests/models/meta_coherence.rs:21-29`                                                                    | Remover até ser necessário                                                |
| Campo morto `max_frames_count`                                                                              | `src/clap/processor/state.rs:122-123`                                                                     | Implementar a assertion prometida ou remover                              |
| `generate_sine`/`generate_sine_440hz` triplicado                                                            | `benches/common.rs:20`, `tests/common/signals.rs:13`, `benches/linear.rs:48`, `tests/models/namb_v2_*.rs` | Reutilizar helpers existentes                                             |
| Feature `pgo` vazia (marcador de build script)                                                              | `Cargo.toml:122`, zero `cfg(feature = "pgo")` em .rs                                                      | Comentar no Cargo.toml que é intencional (ou remover)                     |

Pontos exemplares a preservar como referência de estilo: o dispatch fino de
`src/models/a2/activations.rs` sobre `crate::math::activations` (zero duplicação numérica —
elimina o risco de divergência entre implementações), a delegação `A2Conv1d::Standard` →
`Conv1dDyn`, e as macros multi-ISA de `conv_kernels.rs`.

---

## R15 · Feature `testing` em `default` embarca instrumentação em builds de produção — **BAIXA**

### R15 · Evidência

`Cargo.toml:115` — `default = ["standalone", "testing"]`; `src/testing/` (~3k+ linhas,
oráculo f64, perceptual, MUSHRA) e ganchos como `DISABLE_GATE`
(`src/dsp/pipeline/stages/input.rs:23`) compilam em toda build default.

### R15 · Diagnóstico

O `build-release.sh` pode até desativar defaults, mas o run do log mostra os bins de
`testing` presentes no fluxo de release (o `pgo_profiling_workload` os exige — legítimo).
O risco é superfície: código de medição off-RT dentro do `.so` CLAP distribuído (7.3 MB)
e atalhos de teste (`NAM_DISABLE_GATE`) ativos em produção.

### R15 · Proposta de solução

1. Medir o impacto real: `cargo bloat --release` (stable) antes/depois de remover `testing`
   dos defaults.
2. Remover `"testing"` de `default`; ajustar `utils/*.sh` e `dev` workflows para
   `--features testing` explícito (os bins já declaram `required-features`).
3. Decidir explicitamente se `NAM_DISABLE_GATE` é feature de usuário (documentar no README) ou
   de teste (gate por feature).

---

## R16 · Higiene de saída e referências obsoletas — **BAIXA**

1. **Referência morta:** o dashboard imprime "Ver docs/perceptual_validation.md… e
   `TODO-findings.md` Achado F3" (`testes.log:2750`) — o arquivo F3 referenciado foi resolvido
   e removido do repo; a nota agora aponta para *este* documento, que não contém aquele F3.
   Corrigir a mensagem em `utils/quality-dashboard.sh` para apontar ao documento canônico
   (ex.: `docs/perceptual_validation.md#decomposition-cold-start`) e adotar a regra: scripts
   nunca citam `TODO-*.md` (são artefatos transitórios).
2. **Poluição do sumário do oráculo:** `test_summary_table` intercala `PROD FIRST 10:` /
   `ORACLE FIRST 10:` (`testes.log:2311-2416`) de cada modelo no meio da tabela, tornando-a
   ilegível. Mover os dumps para trás de `NAM_ORACLE_VERBOSE=1` (env já é padrão no projeto
   para verbosidade de teste).
3. **`#[ignore]` sem motivo:** `src/dsp/gate_test.rs:300` — adicionar
   `#[ignore = "proptest 10k casos; roda no tests-long (gate_envelope_continuity_proptest)"]`
   (o teste roda no long-suite, `utils/tests-long.sh:506` — cobertura não está perdida, mas o
   log do quick não explica).
4. **Moldura desalinhada:** caixa do `isa_matrix_header_info` com colunas tortas
   (`testes.log:2110-2121`) — cosmético.
5. **Renomeio de clareza:** `test_golden_vectors_wavenet_condition_lstm` é teste de política
   fail-closed (rejeição), não golden de fidelidade (`tests/models/golden_vectors.rs:1133`) —
   renomear para `test_policy_reject_condition_lstm` evita a falsa expectativa de bloco de
   métricas no log.

---

## Propostas do Pesquisador-Inovador (stable-only, aproveitando o que já existe)

### P1 · Model-checking do protocolo SPSC/GC com `loom` (dev-dependency, roda em stable)

O projeto tem stress tests excelentes (`test_gc_concurrent_push_drain`,
`gc_stress_1000_swaps`), mas stress não **prova** ausência de race — explora interleavings ao
acaso. `loom` (crate da equipe Tokio, funciona em stable via `RUSTFLAGS="--cfg loom"`) explora
*exaustivamente* os interleavings permitidos pelo modelo de memória C11 para testes pequenos.
Aplicação cirúrgica, sem tocar produção:

* Modelar em módulo de teste os 3 protocolos críticos: handshake `set_flag_release`/
  `check_flag_acquire`, `GcOverflowBuffer::push/drain`, e o double-buffer do `DspBridge`
  (`generation`/`active_read_idx`).
* Ganho imediato: os itens R8-a/R8-b/R8-f deixam de ser argumentação e viram contra-exemplo
  reproduzível (loom falha com Relaxed, passa com Acquire/Release) — e qualquer refactor
  futuro do SPSC fica protegido.
* Custo: `loom = "0.7"` em dev-dependencies + testes `#[cfg(loom)]` rodando no tests-long.

### P2 · Mutation testing com `cargo-mutants` como extensão da filosofia anti-placebo

O projeto já é pioneiro em meta-testes anti-placebo (thresholds calibrados, catálogo↔testes).
`cargo-mutants` (binário stable) automatiza a pergunta inversa: "se eu quebrar o código, algum
teste fica vermelho?". Proposta contida (não CI-wide — é caro):

* Rodada offline mensal focada nos módulos-fortaleza: `src/loader/` (validação/fail-closed),
  `src/common/spsc/`, `src/dsp/gate.rs`/`adaptive.rs` (FSMs).
* Mutantes sobreviventes viram achados objetivos de lacuna de cobertura (mesmo status deste
  documento).
* Integração: `utils/mutants.sh` documentado, nunca no quick/long loop (respeita a regra de
  escopo estrito dos scripts).

### P3 · Reduzir `unsafe` mantendo codegen: `core::hint::assert_unchecked` (stable 1.81) e `slice::as_chunks` (stable 1.88)

Duas APIs stable recentes que o codebase (toolchain 1.97) ainda não explora:

* **`hint::assert_unchecked(cond)`**: nos pontos do R12 onde `get_unchecked` existe só para
  matar bounds check (fft.rs, stage.rs, dot_basic.rs), o padrão
  `unsafe { hint::assert_unchecked(i < len) }; slice[i]` mantém a indexação **segura** (panic
  impossível vira otimização, não UB de leitura) e concentra o unsafe numa única premissa
  auditável. Validar com `target/dsp_hotpath.asm` (o pipeline BOLT já gera o relatório —
  aproveitar!).
* **`as_chunks::<N>()` / `as_chunks_mut::<N>()`**: substitui `chunks_exact(N)` retornando
  `&[[f32; N]]` — arrays de tamanho fixo dão ao LLVM a informação de layout completa
  (vetorização mais estável entre versões do compilador) e eliminam os `try_into().unwrap()`
  residuais em bordas de kernel. Aplicação incremental nos kernels novos; não reescrever os
  existentes sem medição.

### P4 · Aproveitar o `target/dsp_hotpath.asm` como gate de regressão de codegen

O `build-release.sh` já gera um relatório de assembly anotado (3.4 MB) e o descarta como
sugestão manual. Proposta: script `utils/asm-gate.sh` que extrai contagens objetivas dos
símbolos quentes (nº de `call` inesperados = inline quebrado; presença de `vzeroupper`
excessivo; spills `mov [rsp...]` acima de baseline) e compara com baseline versionado — mesmo
espírito do `quality-contract.txt`, aplicado a codegen. Barato, 100% stable, converte um
artefato já produzido em guarda permanente.

---

## Contribuições de conhecimento aos temas diferidos (sem ação — registro apenas)

> Já incorporado à documentação permanente: `docs/cpp_parity_map.md` §6 (ConvNet) e §4.3
> (WaveNet A2 Dynamic — gated/blended). Esta seção permanece como registro histórico do
> raciocínio desta rodada; a referência de consulta contínua passa a ser o documento acima.

* **TODO-convnet_parity.md:** o run atual reconfirma com precisão a leitura do documento: prod
  × oráculo f64 = 3.57e-15 (−144.5 dB) e âncora NumPy = 5.23e-33, enquanto prod × golden C++
  segue em 2.54e-5 (45.9 dB). Dado novo: a decomposição (`testes.log:2201-2208`) mostra ΔESR
  de fontes (f16c 6.28e-8, bf16 5.26e-7, Padé ~0) todas ordens de magnitude **acima** do erro
  total vs f64 — ou seja, o pipeline Rust é internamente mais consistente que qualquer fonte
  de erro isolada, reforçando a hipótese nº 1 do documento (a divergência está na ordem de
  operações do C++, fusão de BatchNorm, não em ruído acumulado do Rust).
* **TODO-wavenet_a2_max.md:** nenhuma evidência nova no run. Os guards
  (`test_wavenet_a2_max_dispatch_is_disabled_broken` ok, goldens `ignored` com motivo
  explícito) estão funcionando como projetado. Observação lateral: o A2DynGated (−100 dB vs
  oráculo, threshold calibrado 20×, `tests/common/validation.rs:836-845`) usa o mesmo motor
  `WaveNetA2Dyn` que servirá ao a2_max quando desbloqueado — o piso numérico do caminho gated
  (Sigmoid no gate) já está caracterizado e documentado, o que economizará uma investigação
  quando os Epics 2–4 daquele documento forem atacados.

## Hipóteses investigadas e refutadas (registro anti-retrabalho)

> Já incorporado à documentação permanente: `docs/architecture.md` §1 (dispatch duplo em modo
> stereo), §2.6 (`close(fd)` seguro após `mmap` no `MirroredBuffer`); `docs/testing.md` §3
> (nota sobre ruído esperado do `clap-validator`); `docs/cpp_parity_map.md` §4.3 (threshold do
> A2DynGated). Esta seção permanece como registro histórico anti-retrabalho desta rodada.

* **"Dispatcher constrói o modelo duas vezes por load" — REFUTADA.** As linhas duplicadas de
  `[Dispatcher] ... built` no log são o build L/R do modo stereo
  (`src/loader/build.rs:171,188`), por design. Não há parse duplicado do JSON.
* **"`[CLAP_PLUGIN_ERROR] Empty state buffer` é log indevido do plugin" — REFUTADA.** O plugin
  já loga em `Debug` (`src/clap/extensions/state.rs:92-93`); o prefixo de erro é do próprio
  clap-validator interpretando o retorno `false` (que é o comportamento correto exigido pelo
  teste `state-invalid`).
* **"Threshold do A2DynGated herdado/placebo" — REFUTADA.** Calibrado com medição datada e
  margem 20× documentada (`tests/common/validation.rs:836-845`).
* **"fd do memfd fechado cedo demais no mirror_buf" — REFUTADA.** `close(fd)` após `mmap`
  `MAP_SHARED` é seguro e intencional; mapeamentos sobrevivem ao fd (Linux ≥ 4.0).

---

## Épicos (ordem de execução recomendada)

### EP-R1 — Desarmar as minas de memória (R1 + R10 + R9) — **primeiro, é o núcleo da rodada** [DONE]

Escopo: scratch pré-alocado no layer WaveNet (R1, preferência 1), clamp defensivo no
oversampler (R10), `try_clone` no MirroredBuffer (R9). Três correções locais, sem mudança de
comportamento sonoro — critério de aceite objetivo: `utils/tests-quick.sh` +
`quality-dashboard.sh --check` verdes **sem alteração de nenhum número do contrato** (as
correções não podem mudar um único bit do áudio; se mudarem, algo está errado), benchmarks
`inference_bench`/`regression_gate` sem regressão > ruído. Risco: baixo-médio (toca hot path;
mitigado pelo contrato bit-exact e pelos goldens).

### EP-R2 — Ciclo de vida à prova de host hostil (R2 + R13 + R11 + R3) [DONE]

Escopo: `Arc<DialogSharedState>` nas threads de diálogo + join com deadline no destroy da GUI

* drenagem final (ou leak documentado+testado) do GC + documentação do double-SIGINT.
  Critério: novos testes de lifecycle destrutivo (destroy com diálogo aberto; destroy com GC em
  trânsito) verdes no tests-long; `clap-validator` completo sem regressão. Risco: médio
  (lifecycle CLAP tem sutilezas de thread; usar o harness `clack-host` já existente).

### EP-R3 — Formalização da concorrência (R8 completo + P1) [DONE]

Escopo: aplicar a tabela R8 (8 correções de uma linha + comentários de pareamento) e
introduzir os testes `loom` dos 3 protocolos (P1). Ordem interna: primeiro loom modelando o
estado **atual** (deve falhar em R8-a/b/f — prova do achado), depois as correções (loom passa).
Critério: `cargo test --cfg loom` (job novo no tests-long) verde; zero mudança de asm nos hot
paths x86 (conferir com P4/dsp_hotpath.asm). Risco: baixo.

### EP-R4 — Blindagem da malha de QA (R6 + R7 + R4 + R5) [DONE]

Escopo: chave composta exata no contrato + meta-teste de prefixo (R6); gate "≥1 passed" em
todas as fases do tests-long (R7); panic hook zero-alloc com snapshot pré-capturado (R4);
`RtStatusFlags` no lugar do `log::error!` + meta-teste grep de RT-safety (R5). É o épico que
protege todos os outros: fissuras de QA deixam regressões dos EP-R1/R2/R3 invisíveis.
Critério: regravar contrato com `--save` e conferir manualmente as linhas A2-Full v1/v2
distintas; teste do panic hook com alloc_audit = 0; fase PipeWire falhando artificialmente com
filtro vazio (teste do gate). Risco: baixo.

### EP-R5 — Higiene e superfície (R12 + R14 + R15 + R16 + P3) [DONE]

Escopo: sprint mecânico de documentação SAFETY (tabela R12), remoção de mortos/duplicados
(R14), `testing` fora do default com medição `cargo bloat` (R15), limpezas de log/refs (R16),
e adoção incremental de `assert_unchecked`/`as_chunks` onde reduzir unsafe sem regredir asm
(P3, validado por P4). Critério: `cargo clippy --all-targets` limpo, contagem de blocos
`unsafe` em produção reduzida e 100% com SAFETY específico, quick suite verde. Risco: mínimo —
ideal para absorver com as skills `refatora-rust`/`refatora-doc`.

### EP-R6 (opcional/contínuo) — Guardas de segunda ordem (P2 + P4) [ADIADO]

> Nota do PO: Guardado para o futuro.

Escopo: `utils/mutants.sh` (rodada mensal off-line, módulos-fortaleza) e `utils/asm-gate.sh`
(baseline de codegen sobre o dsp_hotpath.asm já gerado). Nenhum bloqueio sobre os demais
épicos; entrega valor composto ao longo do tempo. Risco: zero (ferramentas externas, nada em
produção).

---

## Rodada 2 — Verificação pós-implementação e nova auditoria (2026-07-16)

## Verificação pós-implementação dos EP-R1…EP-R5

Cada sub-item dos cinco épicos marcados `[DONE]` foi reconfirmado nesta data, lendo o código
atual (não o histórico) e citando `file:line` novo. Resultado: **41 de 44 sub-itens
RESOLVIDOS** (alguns de forma diferente da proposta original, mas resolvendo o problema real),
**3 pendências residuais** — nenhuma delas crítica, mas registradas abaixo para rastreabilidade
e fechadas no épico EP-R11.

| Épico | Sub-item | Status                                    | Nota                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----- | -------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EP-R1 | R1       | ✅ RESOLVIDO                              | Preferência 1 aplicada — scratch pré-alocado em `AlignedVec` (`src/models/wavenet/layer.rs:20,23,55`). Zero `MaybeUninit` não inicializado restante em `src/models/`/`src/dsp/`.                                                                                                                                                                                                                                       |
| EP-R1 | R9       | ✅ RESOLVIDO (forma diferente)            | `try_clone() -> io::Result<Self>` (`src/dsp/mirror_buf.rs:184-193`); `Clone` delega com `.expect()` documentado; caminho CLAP `activate()` não clona `MirroredBuffer`, elimina o risco de panic cruzar FFI.                                                                                                                                                                                                            |
| EP-R1 | R10      | ⚠️ PARCIAL                                | Clamp defensivo aplicado (`src/dsp/oversample.rs:189,218-219`) e flag `RT_STATUS_HOST_CONTRACT_VIOLATION` setada — porém no orquestrador (`src/clap/processor/dsp/orchestrator.rs:24-30`), não dentro do oversampler. Falta teste de integração com block-size acima do `max_frames_count` negociado (`processor_stress_test.rs`).                                                                                     |
| EP-R2 | R2       | ✅ RESOLVIDO (forma diferente)            | `Arc<DialogSharedState>` elimina reconstrução de ponteiro cru nos file-dialogs; `JoinHandle` armazenado e joinado no teardown; `alive_fence` promovido a Release/Acquire. `NamClapSharedRef` continua `pub *const` sem `NonNull` privado (proposta 2 não aplicada) — mas ver R17 abaixo, que encontrou um vetor de UAF concreto residual não coberto por esta correção.                                                |
| EP-R2 | R3       | ✅ RESOLVIDO                              | Handler documentado (`src/main.rs:80-89`); `SHUTDOWN.load` com `Acquire` nos 3 pontos; nota operacional em `docs/architecture.md:488-492`.                                                                                                                                                                                                                                                                             |
| EP-R2 | R11      | ✅ RESOLVIDO                              | `drain_gc_final()` com dupla passagem (`src/clap/plugin/main_thread/mod.rs:200-221`), chamada em `deactivate()`, `Drop` e `on_main_thread`; documentado em `src/common/spsc/gc.rs:4-17`.                                                                                                                                                                                                                               |
| EP-R2 | R13      | ✅ RESOLVIDO                              | Watchdog com deadline de 2s + `is_finished()` polling (`src/clap/extensions/gui.rs:24-48`); abandono controlado com `log::warn!`.                                                                                                                                                                                                                                                                                      |
| EP-R3 | R8-a..g  | ✅ RESOLVIDO (7/8)                        | Todos os 7 pares Release/Acquire corrigidos com comentários de pareamento cruzado (`main.rs:91`↔`run.rs:141`; `listeners.rs:59`↔`rate_sync.rs:21`; `panic_hook.rs:43`; `telemetry.rs:101-103` com `swap(0)`; `shared.rs:262`↔`window/state.rs:190`; `gc.rs:229` documentado; `run.rs:155,204,235,270,305` com `clear_flag_relaxed`).                                                                                   |
| EP-R3 | R8-h     | ❌ **NÃO RESOLVIDO**                      | `src/common/spsc/gc.rs:304-306` — `RT_STATUS_GC_OVERFLOW` ainda é setado incondicionalmente; o retorno de `gc_overflow.push(i)` é capturado em `_overwrote` e nunca lido. Fix trivial de 1 linha, zero risco — ver EP-R11.                                                                                                                                                                                             |
| EP-R3 | P1       | ✅ RESOLVIDO                              | `loom = "0.7"` em dev-deps; `tests/loom_tests.rs` com 4 testes cobrindo os 3 protocolos citados; fase dedicada em `utils/tests-long.sh:683-689`.                                                                                                                                                                                                                                                                       |
| EP-R4 | R4       | ✅ RESOLVIDO                              | `OnceLock<SystemSnapshot>` pré-capturado; buffer fixo `[u8; 4096]` via `LimitWriter`; `try_read()` com fallback; teste `test_panic_hook_zero_alloc` (`tests/models/diagnostic_bundle.rs:658-682`) com `alloc_audit == 0`.                                                                                                                                                                                              |
| EP-R4 | R5       | ✅ RESOLVIDO                              | `RT_STATUS_SLIMMABLE_RESET_FAILED` (`src/common/spsc/status.rs:64`) substitui o `log::error!`; consumido nos dois main threads; meta-teste estrutural `test_rt_logging_safety` (`tests/models/meta_coherence.rs:421-493`).                                                                                                                                                                                             |
| EP-R4 | R6       | ✅ RESOLVIDO                              | Matching exato por chave normalizada (`utils/quality-dashboard.sh:1691-1695`); meta-teste `test_quality_contract_uniqueness` (`tests/models/meta_coherence.rs:346-407`); `docs/quality-contract.txt` com valores A2-Full v1/v2 distintos.                                                                                                                                                                              |
| EP-R4 | R7       | ✅ RESOLVIDO                              | `assert_ran_tests()` (`utils/tests-long.sh:375-407`) aplicado genericamente a todas as fases via `run_phase()`; heurística antiga de `<1s` removida.                                                                                                                                                                                                                                                                   |
| EP-R5 | R12 (9)  | ✅ RESOLVIDO (9/9)                        | Todos os 9 locais com SAFETY específico hoje (mirror_buf, huge_alloc, stage.rs com `const` assert, fft.rs migrado para `assert_unchecked`, convnet/model.rs, wavenet/conv1d.rs, gui/mod.rs, gemv_bf16.rs, main.rs com campo correto do union `sigaction`).                                                                                                                                                             |
| EP-R5 | R14 (8)  | ⚠️ PARCIAL (5/8 + 2 parcial + 1 pendente) | Resolvidos: `FftPlannerRadix4` com cfg-gate, `#[allow(dead_code)]` espúrios, `CatalogGap` removido, `max_frames_count` ativamente usado, feature `pgo` comentada. **Pendente:** `tests/models/proptest_parsers.rs:270-512` continua órfão (zero call-sites). **Parcial:** `median` consolidado em 1 local (ok), mas `generate_sine_440hz` ainda duplicado entre `tests/common/signals.rs:13` e `benches/common.rs:20`. |
| EP-R5 | R15      | ✅ RESOLVIDO                              | `default = ["standalone"]` (`Cargo.toml:117`); scripts usam `--features testing` explícito.                                                                                                                                                                                                                                                                                                                            |
| EP-R5 | R16 (5)  | ✅ RESOLVIDO (4/5 + 1 cosmético)          | Referência morta corrigida, `NAM_ORACLE_VERBOSE` aplicado, `#[ignore]` com motivo, teste renomeado. Moldura do `isa_matrix_header_info` já estava ok (item cosmético, sem ação necessária).                                                                                                                                                                                                                            |
| EP-R5 | P3       | ⚠️ PARCIAL                                | `assert_unchecked` adotado exemplarmente em `fft.rs` (13 ocorrências); `dsp/stage.rs` manteve `get_unchecked` (correto e documentado, mas não migrado); `as_chunks` com **zero adoção**.                                                                                                                                                                                                                               |

**Build/testes de evidência objetiva:** `cargo build --release` limpo; suítes `wavenet`,
`oversample_test` (16/16), `loom_tests` e `meta_coherence` verdes nas verificações pontuais.

---

## Novos achados (Resilience & Robustness) — Rodada 2

## R17 · UAF residual: `NamPluginWindow::new` desreferencia `shared.0` sem `alive_fence` na thread da janela flutuante — **ALTA**

### R17 · Evidência

* `src/clap/extensions/gui.rs:213-215` — a construção da janela **flutuante** roda numa thread
  própria, concorrente com o main thread do host:

  ```rust
  let handle = std::thread::spawn(move || {
      // ...
      let window = NamPluginWindow::new(win, shared_ptr, host_static, cs);
  ```

* `src/clap/gui/window/state.rs:134-142` — dentro de `NamPluginWindow::new`, ANTES de
  `alive_fence` existir localmente, o ponteiro cru é desreferenciado direto:

  ```rust
  let stored = unsafe {
      (*shared.0)
          .cold
          .gui_scale_factor
          .load(std::sync::atomic::Ordering::Relaxed)
  };
  ```

* `src/clap/gui/window/state.rs:168` — mesmo padrão, ainda mais crítico: é a leitura que
  **inicializa** o próprio `alive_fence` local:

  ```rust
  let alive_fence = unsafe { &*shared.0 }.cold.alive_fence.clone();
  ```

* Comparar com `safe_shared()` (`state.rs:189-198`), usado em TODO o resto do código, que
  verifica `alive_fence.load(Acquire)` antes de desreferenciar — exatamente a proteção que
  falta nestes dois pontos.

### R17 · Diagnóstico

O achado R2 (rodada 1) eliminou o UAF nas threads do file-dialog substituindo o ponteiro cru
por `Arc<DialogSharedState>`. Porém a thread da **janela flutuante** (`gui.rs:213`, um
mecanismo diferente, não tocado pela correção de R2) ainda constrói `NamPluginWindow` recebendo
`shared: NamClapSharedRef` (ponteiro cru) e desreferencia-o duas vezes antes de ter qualquer
`alive_fence` para proteger o acesso — é o próprio bootstrap do fence que depende do acesso
desprotegido. Se o host destruir o plugin (ou o `deactivate()`/`Drop` correr) enquanto a thread
flutuante ainda está em `NamPluginWindow::new()` (cenário plausível: o próprio R13 documenta que
essa inicialização pode ser lenta — conexão X11/OpenGL degradada, é exatamente o motivo do
watchdog de 2s no *destroy*), as linhas 137 e 168 leem memória potencialmente já liberada.

### R17 · Impacto

Requer corrida de dois eventos raros simultâneos (destruição do plugin + criação lenta de
janela flutuante), mas quando ocorre é UAF de leitura sobre `NamClapShared` inteiro — mesma
classe de bug do R2 original, porém em vetor de ataque diferente e não coberto pela correção
já aplicada.

### R17 · Proposta de solução

1. Mover a leitura do `alive_fence` para **antes** do `thread::spawn` em `gui.rs`, no main
   thread (onde a validade de `shared.0` é garantida pelo próprio chamador do `create()`), e
   passar o `Arc<AtomicBool>` (fence) já clonado para dentro da thread — elimina a
   desreferência cega em `state.rs:168`.
2. Para `gui_scale_factor` (`state.rs:137`), mover a leitura para o mesmo ponto (pré-spawn) ou
   condicioná-la ao fence já recebido por parâmetro.
3. Após os passos 1-2, `NamPluginWindow::new` nunca mais precisa desreferenciar `shared.0`
   sem primeiro checar o fence — reaproveitar `safe_shared()` ou uma variante que aceite o
   fence por parâmetro em vez de `self`.
4. Estender o teste `test_dialog_state_outlives_plugin_drop` (já existente para file-dialogs)
   com um equivalente para a janela flutuante, simulando destruição do plugin durante a
   construção.

Critério de aceite: zero desreferência de `shared.0`/`NamClapSharedRef` fora de `safe_shared()`
em todo `src/clap/gui/` (`grep -rn "shared\.0\|\*self\.shared" src/clap/gui/`).

---

## R18 · `log::error!`/`log::info!` alcançáveis no thread RT do PipeWire via `configure_realtime_thread` — **ALTA**

### R18 · Evidência

* `src/standalone/pw_host/capture/setup.rs:87-91` — chamado de dentro do closure `.process()`
  (o próprio callback RT de áudio do PipeWire), condicionado a `!state.thread_configured`:

  ```rust
  .process(move |stream: &pw::stream::Stream, _info| {
      if !state.thread_configured {
          rt_setup::configure_realtime_thread(target_cpu, rt_status_for_process.clone());
          state.thread_configured = true;
      }
  ```

* `src/standalone/rt_setup/thread.rs:65-67,72-74,104-109,135-138,167-173,180-183` —
  `configure_realtime_thread` é `#[cold] #[inline(never)]` mas contém **5 call-sites** de
  `log::error!`/`log::info!`, todos executados dentro do thread RT no primeiro frame:

  ```rust
  #[cold]
  #[inline(never)]
  pub fn configure_realtime_thread(target_cpu: usize, rt_status: Arc<RtStatusFlags>) {
      // ...
      log::error!(
          "CPU {} is out of bounds (CPU_SETSIZE={}). ...", ...
      );
  ```

### R18 · Diagnóstico

`src/standalone/pw_host/mod.rs:30-32` declara explicitamente o contrato "Zero I/O — we never
write to the terminal or files; status is reported via atomic flags", e a rodada 1 (R5) já
corrigiu exatamente esta classe de violação em `src/models/container.rs`. Este é um segundo
vetor da MESMA classe de bug que passou pela rodada anterior: `configure_realtime_thread` faz
`sched_setscheduler`/`pthread_setaffinity_np`/mlockall e, em qualquer ramo de sucesso ou falha,
loga diretamente via `log::` — inclusive no ramo de **sucesso** (`thread.rs:167-173`, log de
info sempre emitido). O `#[cold]`/`#[inline(never)]` evita poluir o codegen do hot path
recorrente, mas **não** evita a execução real do log síncrono na primeira invocação do
callback RT — que é justamente a que mais importa (é quando o thread está sendo promovido a
FIFO/prioridade RT, com o áudio já fluindo).

### R18 · Impacto

No primeiro bloco de áudio processado, se o backend de log (`env_logger` ou similar) grava em
stderr/arquivo de forma síncrona, o thread RT pode sofrer stall de I/O — xrun audível logo na
inicialização do stream, exatamente o tipo de defeito que a bateria de regras do projeto
(`.agents/rules/rust.md`) proíbe explicitamente.

### R18 · Proposta de solução

Seguir o mesmo padrão já usado para `configure_process_wide` (que corretamente roda em
`main()`, fora do RT):

1. Mover toda a chamada de `configure_realtime_thread` para fora do callback `.process()` —
   idealmente para o `main()`/setup, antes de `thread_loop.start()`, ou para um callback de
   "thread iniciada" do próprio PipeWire se existir um hook fora do caminho de processamento
   de áudio.
2. Se for estruturalmente inevitável rodar dentro do primeiro `.process()` (para garantir que
   é o thread real do RT-callback que recebe o `SCHED_FIFO`), substituir cada `log::*` por
   `rt_status.set_flag(RT_STATUS_*)` (o padrão já usado por R5) e consumir os novos flags em
   `telemetry.rs:115-143`, que já traduz atomics em mensagens no thread principal.
3. Teste: meta-teste estrutural (extensão do já existente `test_rt_logging_safety` de
   `tests/models/meta_coherence.rs`) incluindo `src/standalone/rt_setup/` no escopo de módulos
   proibidos de logar sem `#[cold]` **e sem estar fora do callback RT**.

Critério de aceite: `grep -rn "log::" src/standalone/rt_setup/thread.rs` retorna zero, ou a
função comprovadamente não é mais chamada de dentro de `.process()`.

---

## R19 · `PluginTrackInfoImpl::changed()` usa `as_main_thread_unchecked` sem runtime guard — **MÉDIA**

### R19 · Evidência

`src/clap/extensions/track_info.rs:19`:

```rust
let mut host_mut = unsafe { self.host.shared().as_main_thread_unchecked() };
```

### R19 · Diagnóstico

A spec CLAP garante que `track_info.changed()` é chamado no main thread, mas o código usa a
variante `_unchecked`, que descarta a verificação de runtime (`clap_host_thread_check`) que
`as_main_thread()` faria. Além disso, `self.host` já É um `HostMainThreadHandle` válido — não
há necessidade de reconstruir um segundo handle a partir de `shared()`. Um host não-conformante
que chamasse este callback de uma thread worker produziria aliasing UB (dois
`HostMainThreadHandle` simultâneos em threads diferentes).

### R19 · Impacto

Nulo em hosts conformantes (Bitwig, REAPER, Ardour, etc.). UB silencioso apenas em hosts
CLAP com bug de threading — cenário defensivo, não uma exploração prática hoje.

### R19 · Proposta de solução

Eliminar a reconstrução do handle, reaproveitando `self.host` diretamente:

```rust
if let Some(info) = track_info_ext.get(&mut self.host, &mut buffer) { ... }
```

Custo zero, remove o `unsafe` por completo neste ponto.

---

## R20 · `.expect()` residual em alocações de produção (loader + `activate()` CLAP) — **MÉDIA**

### R20 · Evidência

* `src/clap/processor/mod.rs:87-113,144` — 13 ocorrências de
  `AlignedVec::new(buf_capacity, 0.0f32).expect("pre-allocation of host buffer failed")` e
  `ConvEngine::new(samples, partition_size).expect("ConvEngine allocation failed")`, todas em
  `activate()` (main thread do host, mas caminho de produção real).
* `src/loader/dispatcher/wavenet/standard.rs:168,177,179,181`,
  `src/loader/dispatcher/wavenet/dynamic.rs:74,88,90,92,233,307,326-327`,
  `src/loader/dispatcher/lstm/dynamic_builder.rs:44,47`,
  `src/loader/dispatcher/convnet/mod.rs:144,148,150,278,327,353,357,359` — mesmo padrão
  (`.expect("allocation should succeed for test-sized buffers")` — mensagem enganosa, é código
  de produção, não de teste).

### R20 · Diagnóstico

O caminho de `activate()` já usa `?`/`PluginError` corretamente para outras falhas (ex.:
`NamResampler::new` na mesma função), mas as pré-alocações de buffer de áudio (host, mid,
model, output, oversample) e o `ConvEngine` ignoram esse padrão e usam `.expect()`. Com o
alocador padrão do Rust, falha de alocação já causa `abort()` antes de chegar ao `expect` — mas
com um alocador customizado que retorna `null` (cenário documentado como possível pela própria
`AlignedVec`), o resultado é panic cru em vez de um `PluginError` estruturado devolvido ao host.
Mesma classe de fragilidade no loader: os tamanhos são hoje estaticamente limitados
(`WAVENET_MAX_NUM_FRAMES=64`, bounds de topologia), então nenhum JSON adversário dispara o
panic — mas viola formalmente a política fail-closed do projeto.

### R20 · Impacto

Um host que ative múltiplas instâncias do plugin sob pressão de memória do sistema recebe um
crash abrupto em vez de uma falha de ativação limpa (`PluginError`) que o host poderia tratar
graciosamente (reduzir instâncias, avisar o usuário).

### R20 · Proposta de solução

1. Em `src/clap/processor/mod.rs`, converter os 13+1 `.expect(...)` em
   `.map_err(|e| PluginError::Message(...))?`, seguindo o padrão já usado para
   `NamResampler::new` na mesma função.
2. No loader, substituir `.expect("... test-sized buffers")` por propagação de erro
   (`?` + `anyhow::Context`), corrigindo também a mensagem enganosa.

Critério de aceite: `grep -rn "\.expect(" src/clap/processor/mod.rs src/loader/dispatcher/` sem
ocorrências fora de testes; `utils/tests-quick.sh` verde.

---

## R21 · Lacuna de fuzzing: LSTM dinâmico, A2-Dynamic e SlimmableContainer sem estratégia proptest adversarial — **MÉDIA**

### R21 · Evidência

`tests/models/proptest_parsers.rs` cobre `prop_fuzz_adversarial_wavenet_dims` (`:893`),
`prop_fuzz_adversarial_convnet_dims` (`:938`), `prop_fuzz_adversarial_linear_dims` (`:950`) e
`prop_fuzz_adversarial_state_budget` (`:964`) — mas não existe estratégia equivalente para o
caminho LSTM dinâmico (`MAX_LSTM_HIDDEN_SIZE`, `src/loader/nam_json/topology/lstm.rs:65`), nem
para A2-Dynamic (`MAX_A2_DYN_CHANNELS`, `src/loader/dispatcher/wavenet/mod.rs:185`), nem para
`SlimmableContainer`.

### R21 · Diagnóstico

Os bounds existem e são testados manualmente (`loader_malformed_test.rs`), mas nunca são
exercitados por geração aleatória de propriedades. Uma futura alteração que remova ou relaxe
acidentalmente um desses bounds (ex.: durante uma otimização de `MAX_LSTM_HIDDEN_SIZE`) não
teria uma rede de segurança de fuzzing para capturar a regressão — apenas os testes unitários
explícitos que já conhecem o valor correto.

### R21 · Impacto

Risco de regressão silenciosa em validação de bounds para 3 das 6 arquiteturas suportadas.
Nenhuma vulnerabilidade ativa hoje (validação manual cobre o caso atual).

### R21 · Proposta de solução

Adicionar `adversarial_lstm_json_strategy()`, `adversarial_a2_dynamic_json_strategy()`,
`adversarial_container_json_strategy()` em `tests/models/proptest_parsers.rs`, nos mesmos
moldes das estratégias já existentes para WaveNet/ConvNet/Linear.

---

## R22 · Ausência de telemetria de xrun/buffer-miss (capture e playback PipeWire) — **MÉDIA**

### R22 · Evidência

* `src/standalone/pw_host/rt_callback/process.rs:39-42` (capture) e
  `src/dsp/pipeline/output_pw.rs:66-69` (playback) — ambos retornam silenciosamente quando
  `stream.dequeue_buffer()` retorna `None`:

  ```rust
  let mut _buf = match stream.dequeue_buffer() {
      Some(b) => b,
      None => return,
  };
  ```

* O único contador existente, `dsp_overloads`, mede violação de *budget de CPU*
  (`elapsed_secs > budget_secs`), não indisponibilidade de buffer do PipeWire.

### R22 · Diagnóstico

Quando o PipeWire não tem buffer disponível (xrun/underrun real do lado do driver/kernel), o
áudio continua fluindo (PipeWire insere silêncio), mas nenhum contador atômico visível ao
thread principal é incrementado. O operador não consegue distinguir "estou tendo xrun de
kernel/PipeWire" de "estou tendo overload de CPU no meu próprio processamento" — são causas
raiz completamente diferentes que hoje produzem o mesmo sintoma (glitch sem contador).

### R22 · Impacto

Diagnóstico de campo prejudicado: um usuário reportando glitches não tem como o time de
suporte (`.agents/skills/diagnostico/`) diferenciar as duas causas via telemetria exportada.

### R22 · Proposta de solução

Adicionar `pw_buffer_miss` (capture) e `bridge_read_miss`/`playback_miss` (playback) como
novos campos em `RtStatusFlags`, incrementados com `fetch_add(1, Ordering::Relaxed)` nos
branches `None` já existentes; expor ambos em `poll_rt_status`/`telemetry.rs`, ao lado de
`dsp_overloads`.

---

## R23 · Higiene de `unsafe` remanescente fora da tabela R12 (~20 blocos sem SAFETY específico) — **BAIXA**

### R23 · Evidência (amostra representativa dos ~160 arquivos com `unsafe` revisados nesta rodada)

* `src/clap/plugin/shared.rs:75-76` — `unsafe impl Send/Sync for NamClapSharedRef` sem
  comentário SAFETY documentando a proveniência do ponteiro (Arc vazado) ou a justificativa de
  thread-safety.
* `src/dsp/oversample.rs:200-202`, `src/dsp/resampler/core.rs:82-85`,
  `src/dsp/cabsim/conv.rs:237-263`, `src/dsp/gate.rs:340-375`,
  `src/models/a2/grouped_conv1d/simd.rs:317`, `src/models/convnet/batch_norm.rs:171-176` —
  blocos `unsafe`/`get_unchecked`/`transmute` cujas invariantes SÃO verificáveis pelo contexto
  imediato (bounds checks adjacentes, buffers pré-dimensionados), mas sem comentário
  `// SAFETY:` explícito exigido pela política do projeto.
* `src/standalone/pw_host/bridge.rs:42-48` — `libc::madvise` sem SAFETY e sem verificação do
  valor de retorno.

### R23 · Diagnóstico

Nenhum dos ~20 blocos amostrados apresenta UB real hoje — todas as invariantes são mantidas por
construção. O gap é puramente documental: a ausência do comentário `// SAFETY:` explícito
viola a letra da política (`.agents/rules/rust.md`: unsafe deve ser "tightly bounded, isolated,
and comprehensively documented") e torna futuras refatorações mais arriscadas, pois um revisor
pode não perceber uma dependência de bounds-check três linhas acima sem o comentário guiando.

### R23 · Impacto

Nenhum bug hoje. Risco latente de regressão silenciosa em refactors futuros que alterem a
ordem de verificações sem que o revisor perceba a dependência não documentada.

### R23 · Proposta de solução

Sprint mecânico de documentação (mesmo padrão do R12 original): adicionar `// SAFETY:` em cada
bloco citando a invariante real. Para `NamClapSharedRef` especificamente:

```rust
// SAFETY: o ponteiro é obtido de um Arc vazado (leaked) para a vida do
// plugin — o pointee nunca é liberado enquanto o processo roda. Send é
// seguro pois toda mutação interior é via Atomic/Mutex; Sync é seguro pois
// &NamClapSharedRef só permite leitura do ponteiro em si (não do pointee).
```

Custo zero de runtime. Pode ser absorvido pelas skills `refatora-rust`/`documentador`.

---

## R24 · Extensão CLAP `thread-check` não registrada — **BAIXA**

### R24 · Evidência

`src/clap/plugin/mod.rs:36-50` (lista de extensões declaradas) não inclui
`clap_host_thread_check`/`HostThreadCheck`. Múltiplas operações assumem main-thread sem
verificação de runtime: 10+ `Mutex` em `ColdShared`, `HostParams::rescan()`,
`HostState::mark_dirty()`, I/O de disco em state load/save.

### R24 · Diagnóstico

A spec CLAP garante a thread correta para essas chamadas, então não há bug hoje em hosts
conformantes. Mas a extensão `thread-check` existe exatamente para permitir defesa-em-
profundidade via `debug_assert!(is_main_thread())`, e o `clack-extensions` já suporta o feature
flag correspondente — o custo de habilitar é baixo frente ao ganho de detecção precoce de hosts
não-conformantes.

### R24 · Impacto

Nulo em hosts conformantes. Em hosts com bug de threading, operações não-thread-safe
executariam sem qualquer sinal de alerta.

### R24 · Proposta de solução

Registrar `HostThreadCheck` via `clack-extensions` e adicionar
`debug_assert!(self.host.is_main_thread())` nos pontos críticos citados (locks de `ColdShared`,
state load/save, rescan de parâmetros).

---

## R25 · `PoisonError` de `Mutex` descartado silenciosamente em preset_load/housekeeping — **BAIXA**

### R25 · Evidência

* `src/clap/extensions/preset_load.rs:36` — retorna `PluginError` em caso de poison (correto).

* `src/clap/plugin/main_thread/housekeeping.rs:188-192,239-243` —

  ```rust
  let pending_model = if let Ok(mut pending_guard) = self.shared.cold.ui_pending_model.lock() {
      pending_guard.take()
  } else {
      None
  };
  ```

  em caso de poison, retorna `None` silenciosamente — o modelo/IR pendente do usuário é
  perdido sem diagnóstico.

* Compare com `processor/mod.rs:59,258`, que já trata poisoning corretamente via
  `.unwrap_or_else(|e| e.into_inner())`.

### R25 · Diagnóstico

`ColdShared` tem 10+ campos `Mutex<T>`. O tratamento de `PoisonError` é inconsistente entre
módulos: `preset_load.rs` ao menos comunica a falha ao host; `housekeeping.rs` simplesmente
segue em frente como se não houvesse nada pendente. Poisoning requer um panic anterior
segurando o lock (raro, mas exatamente o cenário em que diagnóstico é mais necessário).

### R25 · Impacto

Em caso de poisoning (raro), um carregamento de preset/modelo/IR via GUI é perdido
silenciosamente, sem log e sem erro visível — degrada UX e dificulta diagnóstico exatamente
quando mais se precisa dele.

### R25 · Proposta de solução

Padronizar em todos os call-sites de `ColdShared`: `.unwrap_or_else(|e| { log::error!(...);
e.into_inner() })`, garantindo que o dado pendente ainda é recuperado (o `Mutex` envenenado
ainda guarda o valor) e que o evento é logado no main thread (RT-safe, pois `housekeeping.rs`
não roda no thread de áudio).

---

## R26 · Campos/flags mortos ou write-only — **BAIXA**

### R26 · Evidência

* `src/dsp/pipeline/context.rs:87-97` — 4 campos `os_in_l`/`os_in_r`/`os_model_l`/`os_model_r`
  com `#[allow(unused)]`, alocados em `clap/processor/mod.rs:104-112` (~256 KB por instância)
  mas nunca lidos — pipeline de oversampling planejado porém não conectado.
* `src/clap/gui/ui/zones/dialog_state.rs:9,16` — campo `alive: AtomicBool` com
  `#[allow(dead_code)]`, escrito (`store`) na criação/destruição mas nunca lido (`load`) neste
  crate.
* `src/standalone/rt_setup/thread.rs:93,115,128` — `std::mem::zeroed()` para `libc::cpu_set_t`/
  `libc::sched_param` (funcionalmente correto — todos os padrões de bits são válidos para esses
  tipos — mas `MaybeUninit::zeroed().assume_init()` ou inicialização direta de campos é mais
  idiomático e documentaria a intenção).
* `src/clap/extensions/state.rs:92` — único `CString::new(...).unwrap()` do arquivo; todos os
  demais (linhas 141,167,178,187) usam `.unwrap_or_default()` — inconsistência de estilo num
  entry point `extern "C"` (o crate `clack-plugin` 0.1.0 não envolve entry points em
  `catch_unwind`, então panics cruzando FFI são UB, mesmo que este literal específico nunca
  possa falhar).

### R26 · Diagnóstico

Nenhum destes é um bug ativo — são sinais de trabalho incompleto (`os_*` buffers) ou pequenas
inconsistências de estilo (`mem::zeroed`, `.unwrap()` isolado) que aumentam o custo cognitivo
de manutenção e, no caso do `alive: AtomicBool` write-only, sugerem uma variável que talvez
devesse ter sido removida ao final da refatoração de R2 (rodada 1) ou que exista um consumidor
externo não documentado.

### R26 · Proposta de solução

1. `os_*` buffers: implementar o pipeline de oversampling que os consumiria, ou remover a
   alocação e os campos até a feature ser retomada.
2. `alive: AtomicBool`: confirmar se há consumidor real (thread de diálogo/FFI); se não houver,
   remover; se houver, documentar onde é lido.
3. `mem::zeroed()`: trocar por `libc::CPU_ZERO(&mut cpuset)` (já usado na linha seguinte) e
   inicialização direta de campos para `sched_param`.
4. `state.rs:92`: trocar `.unwrap()` por `.unwrap_or_default()` para consistência com o
   resto do arquivo.

---

## Novas propostas do Pesquisador-Inovador — Rodada 2 (stable-only, pesquisadas via web)

### P5 · Migração de `#[allow(...)]` para `#[expect(...)]` (Rust 1.81, tracked lint suppression) — *ver EP-R12*

O projeto tem **98 atributos `#[allow(...)]`** e zero `#[expect(...)]`. `#[expect(lint)]`
(estabilizado 1.81, [RFC 2383](https://rust-lang.github.io/rfcs/2383-lint-reason.html)) se
comporta como `#[allow]` mas emite `unfulfilled_lint_expectations` quando a supressão deixa de
ser necessária — elimina o bitrot silencioso de `#[allow(dead_code)]`/`#[allow(clippy::...)]`
acumulado (visto também no achado R26). Suporta `reason = "..."`. Proposta: adicionar
`#![warn(clippy::allow_attributes)]` em `[lints.clippy]` e migrar gradualmente, priorizando
`dead_code` e `too_many_arguments`. Custo: ~2h de refactor mecânico; risco zero.

### P6 · `cargo machete` — dependências não usadas [ADIADO]

Ferramenta stable-only ([bnjbvr/cargo-machete](https://github.com/bnjbvr/cargo-machete)),
detecção textual (não compila, extremamente rápida), suporta
`[package.metadata.cargo-machete] ignored = [...]` para falsos positivos de `build.rs`/proc
macros. Proposta: rodar como auditoria inicial e integrar em `utils/lints.sh` ou workflow de CI
dedicado. Custo ~15min de setup; risco baixo.

### P7 · `typos` — spell-checker de código-fonte em CI [ADIADO]

Ferramenta stable-only ([crate-ci/typos](https://github.com/crate-ci/typos)), verifica apenas
comentários/strings/prosa (não identificadores), ~0.3s para 10k arquivos, GitHub Action nativa.
Projeto não tem nenhum spell-checker hoje, com documentação e mensagens de erro extensas
user-facing. Proposta: `_typos.toml` com dicionário de domínio (wavenet, namb, egui) + CI gate.
Custo ~30min; risco virtualmente zero.

### P8 · `cargo vet` — auditoria de supply-chain (Mozilla) [ADIADO]

Ferramenta stable-only ([mozilla.github.io/cargo-vet](https://mozilla.github.io/cargo-vet/)),
permite importar auditorias de organizações confiáveis (Mozilla, Google, Bytecode Alliance) via
`cargo vet import`. Projeto distribui binário release (PGO) com 14 dependências diretas
(pipewire, clack-*, rtrb, egui) — superfície de supply-chain que justifica atestação humana de
revisão, além do que `cargo audit` (CVEs conhecidos) já cobre. Custo ~1-2h de baseline inicial;
risco baixo (manutenção proporcional a `cargo update`).

### P9 · `build.warnings = "deny"` (Cargo/Rust 1.97) substituindo `RUSTFLAGS=-Dwarnings` — *ver EP-R12*

O `utils/lints.sh:100-110` usa `cargo clippy -- -D warnings`. A opção nativa `build.warnings`
do Cargo, estabilizada em Rust 1.97 (a toolchain atual do projeto — [tracking issue# 14802](<https://github.com/rust-lang/cargo/issues/14802>)),
com nível `deny`, **não invalida o cache de build** (diferente de `RUSTFLAGS`, que muda o fingerprint e força rebuild completo) e
pode ser combinada com `--keep-going` para diagnóstico completo. Proposta: `[build]
warnings = "deny"` em `.cargo/config.toml` + `CARGO_BUILD_WARNINGS=deny` no CI. Custo ~10min;
risco zero (comportamento equivalente, melhor ergonomia/cache).

### P10 · `cargo semver-checks` — breaking changes de API pública — *ver EP-R12*

Ferramenta stable-only ([obi1kenobi/cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks),
projeto de meta do Rust para merge no cargo), usa rustdoc JSON (estável desde ~1.70) para
detectar breaking changes sem compilar, sem falsos positivos conhecidos. O crate compila como
`cdylib` + `rlib` com API pública real (`src/models/`, `src/math/`, `src/loader/`) e está em
v3.0.0 — momento apropriado para este guard. Proposta: baseline inicial + GitHub Action
`obi1kenobi/cargo-semver-checks-action@v2` em PRs. Custo ~20min; risco baixo.

**Descartadas por exigirem nightly** (violam a política stable-only do projeto):
`cargo careful` (`-Z build-std`), `kani` verifier (`rustc_private`), `cargo udeps`.
**Já adotadas pelo projeto** (confirmado, não precisam de proposta): `LazyLock`/`LazyCell`,
`c"..."` C-string literals, `is_none_or`/`is_some_and`.

---

## Novos épicos (Rodada 2)

### EP-R7 — Fechar vetores residuais de UAF e RT-safety (R17 + R18) — **primeiro, mesma classe de bug de R2/R5 já corrigidos** [DONE]

Escopo: mover a leitura do `alive_fence`/`gui_scale_factor` para antes do `thread::spawn` da
janela flutuante (R17); mover `configure_realtime_thread` para fora do callback `.process()`
ou substituir seus `log::*` por `RtStatusFlags` (R18). Critério de aceite: zero desreferência de
`shared.0` fora de `safe_shared()` em `src/clap/gui/`; zero `log::` alcançável de dentro de
`.process()` em `src/standalone/`; testes de lifecycle destrutivo cobrindo o cenário de
destruição durante criação de janela flutuante. Risco: baixo-médio (toca lifecycle CLAP e RT
setup; mitigado por serem correções cirúrgicas de reordenação, não redesenho).

### EP-R8 — Blindagem da fronteira CLAP host↔plugin (R19 + R24 + R25) [DONE]

Escopo: eliminar `as_main_thread_unchecked` em `track_info.rs` (R19); registrar extensão
`thread-check` com `debug_assert!` nos pontos críticos (R24); padronizar tratamento de
`PoisonError` em todos os `Mutex` de `ColdShared` (R25). Critério de aceite:
`clap-validator` completo sem regressão; `grep -rn "as_main_thread_unchecked" src/clap/`
mostra apenas usos genuinamente necessários (idealmente zero); nenhum `if let Ok(...) else`
silencioso remanescente em `housekeeping.rs`. Risco: baixo.

### EP-R9 — Robustez de carregamento e cobertura anti-regressão (R20 + R21) [DONE]

Escopo: substituir `.expect()` por propagação de erro em `activate()` CLAP e nos dispatchers
do loader (R20); adicionar estratégias proptest adversariais para LSTM dinâmico, A2-Dynamic e
SlimmableContainer (R21). Critério de aceite: `grep -rn "\.expect(" src/clap/processor/mod.rs
src/loader/dispatcher/` sem ocorrências fora de testes; novas estratégias proptest rodando no
`tests-long.sh` sem falso-positivo. Risco: baixo (mudanças mecânicas de propagação de erro +
testes aditivos).

### EP-R10 — Observabilidade e higiene remanescente (R22 + R23 + R26) [DONE]

Escopo: novos contadores `pw_buffer_miss`/`playback_miss` em `RtStatusFlags` (R22); sprint de
documentação SAFETY nos ~20 blocos remanescentes (R23); resolver ou documentar os campos
mortos/write-only (`os_*` buffers, `alive: AtomicBool`, `mem::zeroed`, `state.rs:92`) (R26).
Critério de aceite: `poll_rt_status`/dashboard exibindo os novos contadores; 100% dos blocos
`unsafe` em produção com SAFETY específico (repetir o critério de aceite do R12 original, agora
sem exceções conhecidas). Risco: mínimo — ideal para as skills `refatora-rust`/`documentador`.

### EP-R11 — Fechar pendências residuais das rodadas EP-R1…EP-R5 (R8-h + R10 + R2/NonNull + R14 + P3) [DONE]

Escopo mecânico, todos os itens já especificados nas propostas originais e reconfirmados nesta
verificação:

1. **R8-h** (`src/common/spsc/gc.rs:304-306`): condicionar `rt_status.set_flag(RT_STATUS_GC_OVERFLOW)`
   ao retorno `true` de `gc_overflow.push(i)` — fix de 1 linha.
2. **R10**: adicionar caso de block-size acima do `max_frames_count` negociado em
   `src/clap/processor_stress_test.rs` (harness `clack-host` já suporta).
3. **R2 (proposta 2 pendente)**: tornar `NamClapSharedRef` com `NonNull` privado — agora
   reforçado pela urgência de R17 (EP-R7), que encontrou um call-site concreto explorando a
   fragilidade do ponteiro público.
4. **R14**: integrar `tests/models/proptest_parsers.rs:270-512` a um teste real ou removê-lo
   (~240 linhas órfãs); consolidar `generate_sine_440hz` entre `tests/common/signals.rs` e
   `benches/common.rs`.
5. **P3**: avaliar migração de `src/dsp/stage.rs` de `get_unchecked` para
   `hint::assert_unchecked` (mantendo o codegen, validado via `dsp_hotpath.asm`); avaliar
   adoção de `as_chunks` em pelo menos um kernel novo como prova de conceito.

Critério de aceite: cada sub-item fechado individualmente com o mesmo rigor de teste do épico
de origem; nenhuma mudança de comportamento sonoro (contrato bit-exact preservado). Risco:
mínimo — todos os itens já têm solução especificada, é execução, não descoberta.

### EP-R12 — Modernização de lint/build/compat (P5 + P9 + P10) [DONE]

Escopo: (1) migração gradual de `#[allow(...)]` para `#[expect(...)]` (P5), priorizando
`dead_code` e `clippy::too_many_arguments` — os dois padrões mais numerosos entre os 98
`#[allow(...)]` do crate — com `reason = "..."` documentando cada supressão remanescente; (2)
adoção de `build.warnings = "deny"` via `.cargo/config.toml` + `CARGO_BUILD_WARNINGS=deny` no
CI (P9), substituindo `cargo clippy -- -D warnings` em `utils/lints.sh:100-110` sem invalidar o
cache de build; (3) baseline de `cargo semver-checks` + GitHub Action
`obi1kenobi/cargo-semver-checks-action@v2` em PRs (P10), guardando a API pública do crate
(`src/models/`, `src/math/`, `src/loader/`) contra breaking changes acidentais a partir da
v3.0.0 atual.

Critério de aceite: `grep -rn "#\[allow(" src/` reduzido nos dois padrões priorizados, com
`#[expect(...)]` documentado no lugar; `utils/lints.sh` usando `build.warnings` em vez de
`RUSTFLAGS=-Dwarnings` sem regressão de cobertura de lint; `cargo semver-checks` executando
limpo contra a baseline v3.0.0 e integrado ao workflow de CI. Ordem interna recomendada: P9
primeiro (habilita a infraestrutura de warning-as-error sem custo de cache antes de qualquer
migração de lint), depois P5 (aproveita o gate já em vigor para não reintroduzir `#[allow]`
espúrios), por fim P10 (guard independente, sem dependência dos outros dois). Risco: zero —
as três são ferramentas/atributos Rust stable, puramente aditivos, sem tocar lógica de
produção nem o hot path de áudio.

---

## Rodada 3 — Verificação pós-implementação (EP-R7…EP-R12) e nova auditoria (2026-07-16)

## Verificação pós-implementação dos EP-R7…EP-R12

Todos os seis épicos da Rodada 2 foram implementados (commits de `b8393373` a `6f9084fe`) e
verificados linha a linha nesta data. Resultado: **13 de 15 sub-itens RESOLVIDOS**, **1
pendência parcial** (R14) e **1 correção histórica** (R8-h — o achado original estava
parcialmente equivocado; ver nota detalhada abaixo).

| Épico  | Sub-item   | Status                                              | Nota                                                                                                                                                                                                                                                                                                                                                                                     |
| ------ | ---------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| EP-R7  | R17        | ✅ RESOLVIDO                                        | Leitura de `alive_fence`/`gui_scale_factor` movida para o main thread, antes do `thread::spawn` (`src/clap/extensions/gui.rs:230-249`); passados por parâmetro para `NamPluginWindow::new` (`window/state.rs:64-71`). Zero `shared.0` fora de `safe_shared()`/`as_ref()` em `src/clap/`. Teste `test_window_safe_shared_boundary` passa.                                                 |
| EP-R7  | R18        | ✅ RESOLVIDO                                        | `configure_realtime_thread` (`src/standalone/rt_setup/thread.rs`) com zero `log::*` — tudo via stores atômicos, traduzidos em `telemetry.rs:poll_rt_status()` no main thread. Escolheu a opção 2 da proposta (flags + poll) em vez da opção 1 (mover a chamada); ambas resolvem o problema real de RT-safety.                                                                            |
| EP-R8  | R19        | ✅ RESOLVIDO                                        | `as_main_thread_unchecked` removido de `track_info.rs`; substituído por `with_arbitrary_lifetime()` (estende lifetime do handle já válido, sem reconstruir a partir de `HostSharedHandle`) + `debug_assert_main_thread` como guarda de runtime.                                                                                                                                          |
| EP-R8  | R24        | ✅ RESOLVIDO (forma diferente)                      | `HostThreadCheck` não foi registrada via `builder.register()`, mas consultada em runtime pelo helper `debug_assert_main_thread` (`main_thread/mod.rs:248-258`), com **19 call-sites** em todas as extensions relevantes. Degrada graciosamente quando o host não provê a extensão — mais robusto que registro estático.                                                                  |
| EP-R8  | R25        | ✅ RESOLVIDO                                        | Todos os 11+ `.lock()` de `ColdShared` em `housekeeping.rs`/`preset_load.rs` agora usam `.unwrap_or_else(\|e\| { log::error!(...); e.into_inner() })`; zero `if let Ok(...) else` silencioso remanescente.                                                                                                                                                                               |
| EP-R9  | R20        | ✅ RESOLVIDO                                        | Zero `.expect()` em `src/clap/processor/mod.rs` e `src/loader/dispatcher/` fora de testes (`grep` confirma).                                                                                                                                                                                                                                                                             |
| EP-R9  | R21        | ✅ RESOLVIDO                                        | `adversarial_lstm_json_strategy`, `adversarial_a2_dynamic_json_strategy`, `adversarial_container_json_strategy` implementadas (8 padrões cada) e exercitadas pelos testes `prop_fuzz_adversarial_{lstm,a2_dynamic,container}_dims`, todos passando.                                                                                                                                      |
| EP-R10 | R22        | ✅ RESOLVIDO                                        | `pw_buffer_miss`/`playback_miss` em `RtStatusFlags`/`TelemetrySnapshot`, incrementados nos branches `None` corretos, expostos em `poll_rt_status` com `log::warn!` quando > 0.                                                                                                                                                                                                           |
| EP-R10 | R23        | ✅ RESOLVIDO                                        | SAFETY específico confirmado por amostragem em todos os arquivos citados; `libc::madvise` com SAFETY + verificação de retorno (`bridge.rs:42-61`).                                                                                                                                                                                                                                       |
| EP-R10 | R26        | ✅ RESOLVIDO                                        | `alive: AtomicBool` removido; `#[allow(unused)]` removido dos `os_*` buffers (agora documentados); `mem::zeroed()` substituído por `MaybeUninit`/init direta; `state.rs:96` usa `.unwrap_or_default()`.                                                                                                                                                                                  |
| EP-R11 | R8-h       | ⚠️ **DIAGNÓSTICO ORIGINAL PARCIALMENTE EQUIVOCADO** | Ver nota dedicada abaixo — a "correção" quebrou `test_gc_stress_1000_swaps`, foi revertida (`9879fffc`), e depois reaplicada silenciosamente junto com ajuste do teste (`f354d540`), sem que o commit declarasse a mudança de semântica no `gc.rs`.                                                                                                                                      |
| EP-R11 | R10        | ✅ RESOLVIDO                                        | `test_host_contract_violation_block_size` (`processor_stress_test.rs:521-593`) cobre block-size de 600 acima do `max_frames_count=512`, valida `Err` em debug e flag `RT_STATUS_HOST_CONTRACT_VIOLATION` em release.                                                                                                                                                                     |
| EP-R11 | R2/NonNull | ✅ RESOLVIDO                                        | `NamClapSharedRef` agora encapsula `std::ptr::NonNull<NamClapShared>` privado, com `new()`/`as_ptr()`/`as_ref()` como única API; `unsafe impl Send/Sync` documentado; 18 call-sites migrados.                                                                                                                                                                                            |
| EP-R11 | R14        | ⚠️ PARCIAL                                          | `tests/models/proptest_parsers.rs:270-512` **continua órfão** (~240 linhas, zero call-sites) — item **não resolvido**. `generate_sine_440hz` ainda tem 2 wrappers idênticos em `benches/common.rs` e `tests/common/signals.rs`, mas ambos delegam para a mesma função subjacente (`testing::aliasing::generate_sine`) — risco de divergência eliminado, duplicação textual remanescente. |
| EP-R11 | P3         | ✅ RESOLVIDO                                        | `src/dsp/stage.rs` migrado para `hint::assert_unchecked` (16 chamadas); `as_chunks`/`as_chunks_mut` adotado como prova de conceito em `src/models/a2/film.rs:196-198`.                                                                                                                                                                                                                   |
| EP-R12 | P5         | ⚠️ PARCIAL                                          | `dead_code`: 100% migrado (0 `#[allow]`, 5 `#[expect]`). `too_many_arguments`: apenas 11 de ~42 locais migrados (31 `#[allow]` remanescentes). `[lints.clippy] allow_attributes` **não configurado** em `Cargo.toml`.                                                                                                                                                                    |
| EP-R12 | P9         | ✅ RESOLVIDO                                        | `.cargo/config.toml:5` com `[build] warnings = "deny"`; `utils/lints.sh` não depende mais de `RUSTFLAGS=-Dwarnings`; `cargo clippy --all-features --all-targets` passa limpo.                                                                                                                                                                                                            |
| EP-R12 | P10        | ❌ **NÃO RESOLVIDO**                                | Nenhuma integração de `cargo semver-checks` encontrada — sem baseline, sem workflow de CI (o repositório não tem `.github/`), sem script utilitário.                                                                                                                                                                                                                                     |

**Build/testes de evidência objetiva:** `cargo build --release` limpo (~1m12s); `cargo clippy
--all-features --all-targets` limpo; `test_window_safe_shared_boundary`,
`test_host_contract_violation_block_size`, `prop_fuzz_adversarial_{lstm,a2_dynamic,container}_dims`
e `test_gc_stress_no_leak` verdes nas verificações pontuais.

### Nota dedicada — correção histórica do achado R8-h

O achado **R8-h** (Rodada 1: "`RT_STATUS_GC_OVERFLOW` setado mesmo quando `push` não
sobrescreveu — condicionar ao retorno `true`") estava **parcialmente equivocado** no
diagnóstico original. A cronologia real, reconstruída via `git log -- src/common/spsc/gc.rs`:

1. `46087896` (Rodada 1, EP-R3) e depois `bff0c4b8` (Rodada 2, EP-R11) aplicaram o gate
   condicional exatamente como proposto: `if gc_overflow.push(i) { rt_status.set_flag(...) }`.
2. Isso **quebrou** `test_gc_stress_1000_swaps` — o teste esperava (corretamente, por design
   original) que a flag disparasse na simples **entrada no tier 3** do GC cascade (o buffer de
   overflow de 64 slots sendo *necessário*), não apenas quando ele *sobrescreve* um slot já
   ocupado. São dois sinais diagnósticos diferentes: "sistema sob pressão" (tier 3 alcançado)
   vs. "leak real" (overwrite). O achado original da Rodada 1 confundiu os dois.
3. `9879fffc` reverteu o gate corretamente, restaurando o comportamento incondicional e citando
   explicitamente o motivo no commit message.
4. `f354d540` — commit intitulado sobre `test_host_contract_violation_block_size` (R10, sem
   qualquer menção a `gc.rs` na mensagem) — **reaplicou silenciosamente** o gate condicional em
   `src/common/spsc/gc.rs:304-308` e ajustou `test_gc_stress_1000_swaps` para a nova semântica.
   O código final está funcionalmente consistente com o teste, mas a mudança de contrato não
   foi declarada, e o comentário de documentação em `gc.rs:276` ainda descreve apenas "sets
   RT_STATUS_GC_OVERFLOW on overflow" — ambíguo entre as duas semânticas possíveis.

**Lição para auditorias futuras:** antes de propor uma correção "mecânica" de 1 linha em código
de diagnóstico/telemetria, confirmar o *contrato comportamental* esperado pelos testes de
stress existentes — um flag de telemetria pode ter uma semântica de "sinalização de pressão"
intencionalmente mais ampla do que seu nome sugere. Esta lição está registrada como ação
concreta no EP-R15 (documentar a semântica atual em `gc.rs:276`) e como prática recomendada para
as próximas rodadas de auditoria.

---

## Novos achados (Resilience & Robustness) — Rodada 3

## R27 · IR/WAV: `sample_rate` extremo (baixo) causa OOM garantido via upsampling catastrófico — **ALTA**

### R27 · Evidência

* `src/dsp/cabsim/loader.rs:177-182` — apenas `sample_rate == 0` é rejeitado; qualquer valor
  minúsculo não-zero (ex.: `sample_rate = 1`) é aceito:

  ```rust
  if sample_rate == 0 {
      return Err(io::Error::new(
          io::ErrorKind::InvalidData,
          "IR WAV: sample rate is zero",
      ));
  }
  ```

* `src/dsp/cabsim/loader.rs:375-377` — a estimativa de tamanho do buffer de resample cresce
  linearmente com a razão de taxas, sem teto:

  ```rust
  let est_len =
      ((input.len() as f64 * output_rate as f64 / input_rate as f64).ceil() as usize) + 256;
  let mut output = Vec::with_capacity(est_len);
  ```

* `src/dsp/resampler/mod.rs:82-85` — `NamResampler::new` só valida `== 0`, não um piso mínimo
  plausível.

### R27 · Diagnóstico

Um WAV de ~44 bytes com `sample_rate=1` no chunk `fmt` (mono, PCM16, 1 sample de dados) passa
todas as validações existentes (tamanho de arquivo, canais, bit depth, duração, NaN/Inf — todas
robustas, confirmado nesta auditoria). Ao reamostrar de 1 Hz para 48 kHz, `est_len` calcula
`ceil(N * 48000 / 1) + 256` — para uma IR de 192.000 amostras (o limite máximo já validado por
`MAX_IR_LENGTH`), isso é **9.216.000.256 amostras f32 (~37 GB)**. `Vec::with_capacity` dispara o
OOM handler do alocador padrão do Rust, abortando o processo.

**Sub-achados relacionados** (mesma área de código, severidade menor, mesmo fix):

* **b) Sample_rate extremamente alto sem teto** (MÉDIA) — `sample_rate = u32::MAX` também passa;
  sem crash demonstrado, mas gera parâmetros degenerados no filtro polyphase (cutoff ≈ 2e-8),
  violando fail-closed defensivo.
* **c) Denormals não detectados na validação** (BAIXA) — `validate_samples()`
  (`loader.rs:282-289`) usa `is_finite()`, que aceita denormals; `normalize_in_place`
  (`loader.rs:416-428`) pode amplificá-los. Impacto mitigado pelo FTZ/DAZ já ativo no pipeline
  (`src/main.rs:24`), mas defesa em profundidade ausente na fronteira de entrada.

### R27 · Impacto

WAV malicioso de ~44 bytes causa crash/abort garantido e incondicional do processo (plugin CLAP
ou standalone) ao tentar carregar a IR — DoS trivial via arquivo de IR compartilhado entre
usuários (presets, IRs de terceiros).

### R27 · Proposta de solução

Adicionar piso e teto de `sample_rate` no parser WAV, cobrindo a faixa fisicamente plausível
para áudio (incluindo oversampling até 8×):

```rust
const MIN_IR_SAMPLE_RATE: u32 = 4_000;
const MAX_IR_SAMPLE_RATE: u32 = 384_000;

if !(MIN_IR_SAMPLE_RATE..=MAX_IR_SAMPLE_RATE).contains(&sample_rate) {
    return Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("IR WAV: sample rate {sample_rate} out of range \
                 ({MIN_IR_SAMPLE_RATE}–{MAX_IR_SAMPLE_RATE})"),
    ));
}
```

Resolve simultaneamente os sub-achados (a) e (b). Para (c), opcional: flush explícito de
denormals na entrada do loader, ou documentar que o FTZ/DAZ global já cobre o caso.

Critério de aceite: teste `test_reject_ir_extreme_sample_rate` (baixo e alto) em
`loader_malformed_test.rs`; nenhuma alocação > `MAX_IR_LENGTH * 4 bytes` possível a partir de
qualquer combinação de `sample_rate`/`target_rate`.

---

## R28 · Automação sample-accurate não implementada; eventos de parâmetro colapsam para o último valor do bloco — **ALTA**

### R28 · Evidência

* `src/clap/processor/events.rs:56-84` — o loop de eventos do host aplica todos os
  `ParamValueEvent` do bloco antes do DSP, cada um sobrescrevendo o target do anterior:

  ```rust
  for event in events.input {
      if let Some(param_event) = event.as_event::<ParamValueEvent>() {
          let val = param_event.value() as f32;
          match clap_id.get() {
              PARAM_INPUT_GAIN => self.set_input_gain(val),
              ...
  ```

* `src/clap/processor/dsp/gain.rs:13-14,26` — o ramp é uma única rampa linear cobrindo o bloco
  inteiro, do valor atual ao target **final** (pós-loop):

  ```rust
  let start = self.smoother_in.peek();
  let target = self.smoother_in.target_value();
  let step = (target - start) / n_samples as f32;
  ```

### R28 · Diagnóstico

O campo `event.header().time` (sample offset dentro do bloco, parte do `clap_event_header_t` da
spec CLAP) nunca é lido. Todos os parâmetros são declarados `IS_AUTOMATABLE`, mas múltiplos
pontos de automação no mesmo bloco de áudio são colapsados para um único valor (o último
processado), e a rampa de ganho trata esse valor como válido desde o sample 0 do bloco — não há
suporte real a automação sample-accurate, apesar de a spec CLAP documentar explicitamente que
"the plugin may use the sample offset in `process()`" (`clap/ext/params.h`).

### R28 · Impacto

Hosts que emitem automação de alta resolução no mesmo bloco (curvas rápidas, moduladores
sample-accurate do Bitwig, envelopes de modulação de alta frequência) têm a resolução
degradada para block-level — o comportamento sonoro observado difere do que o host pretendia
enviar, especialmente audível em automações rápidas de ganho/parâmetros com bloco grande
(hosts com buffer de 1024+ samples a 44.1kHz = ~23ms por bloco).

### R28 · Proposta de solução

Subdividir o processamento do bloco de áudio nos pontos onde eventos de parâmetro ocorrem:
processar sub-blocos delimitados por `event.header().time`, aplicando o ramp parcial em cada
sub-bloco com o valor vigente naquele intervalo. Padrão comum em plugins CLAP/VST3 maduros
("block splitting"). Alternativa de custo zero, mas semanticamente incorreta: não fazer nada e
aceitar a limitação documentando-a — não recomendado, pois a spec permite verificação via
`clap-validator` de conformidade de automação sample-accurate.

Critério de aceite: novo teste de integração que envia múltiplos `ParamValueEvent` com `time`
distintos no mesmo bloco e verifica que a saída de áudio reflete os valores intermediários
(não apenas o último); `clap-validator` sem regressão.

---

## R29 · Push SPSC descartado silenciosamente em `MainThread::flush()`; perda de eventos de parâmetro — **MÉDIA**

### R29 · Evidência

* `src/clap/extensions/params/main.rs:399-401` — dentro do loop de eventos, cada evento produz
  um push SPSC cujo erro é descartado:

  ```rust
  let _ = self.param_tx.push(ClapParamPayload::Params(
      RtPluginParams::from_plugin_params(&self.params),
  ));
  ```

* `src/clap/plugin/mod.rs:64` — capacidade do canal: `RingBuffer::new(8)`.

* `MainThread::flush()` não chama `bump_generation()` em nenhum ponto do método — diferente de
  `AudioProcessor::flush()`, que atualiza atomics + `bump_generation()` e não depende do SPSC.

### R29 · Diagnóstico

Cada evento no `flush()` do main thread (chamado quando o plugin está inativo) gera um push
SPSC com snapshot **completo** de todos os parâmetros. Com capacidade 8, se 9+ eventos distintos
chegarem numa única chamada de `flush()` (automação densa com múltiplos parâmetros), os pushes
excedentes falham silenciosamente (`PushError::Full` descartado) e não há fallback via
`bump_generation()` para que o processador sincronize os atomics na próxima `activate()`. Além
disso, cada push é redundante — carrega o estado completo a cada evento, desperdiçando
capacidade do canal e trabalho de drain no RT thread quando drenado.

### R29 · Impacto

Em cenário raro (9+ eventos de parâmetro distintos numa única chamada de `flush()`, plugin
inativo), eventos de automação podem ser perdidos sem qualquer sinal de erro. Adicionalmente,
desperdício mensurável de capacidade do canal SPSC mesmo no caso comum.

### R29 · Proposta de solução

1. Mover o push SPSC para **fora** do loop de eventos — uma única snapshot após processar todos
   os eventos do `flush()`, eliminando os pushes redundantes.
2. Verificar o retorno de `push()`; em caso de `Full`, chamar `bump_generation()` como fallback
   para que o processador sincronize dos atomics na reativação, garantindo que nenhum evento
   seja perdido silenciosamente mesmo no caso extremo.

Critério de aceite: teste que envia 20 eventos de parâmetro num único `flush()` inativo e
verifica que o estado final pós-`activate()` reflete o último valor de cada parâmetro.

---

## R30 · Reset do smoother de ganho para 1.0 em cada `activate()` causa transiente audível — **BAIXA**

### R30 · Evidência

`src/clap/processor/mod.rs:168-169`:

```rust
let smoother_in = ParamSmoother::new(1.0, audio_config.sample_rate as f32, 20.0);
let smoother_out = ParamSmoother::new(1.0, audio_config.sample_rate as f32, 20.0);
```

### R30 · Diagnóstico

Cada ciclo `deactivate()` → `activate()` recria o `ParamSmoother` com `current = target = 1.0`,
independentemente do valor de ganho vigente antes da desativação. Se o ganho estava, por
exemplo, em +12dB (~3.98 linear), o primeiro bloco após reativação aplica um ramp de 1.0 até o
target recuperado — um "pulo" de ganho perceptível no primeiro bloco (~5ms num bloco de 256
samples a 48kHz).

### R30 · Impacto

Transiente audível apenas em reativações, e apenas no primeiro bloco. Baixo impacto em uso
normal (hosts raramente ciclam activate/deactivate durante reprodução), mas perceptível em
hosts que implementam bypass nativo via toggle de activate/deactivate.

### R30 · Proposta de solução

Em `activate()`, antes de criar os smoothers, ler o valor atual dos atomics
`param_input_gain`/`param_output_gain` e inicializar `ParamSmoother::new(valor_atual, ...)` em
vez de `1.0`. Elimina o transiente sem custo adicional.

---

## R31 · Higiene residual: bypass de CRC32 no NAMB v1 legado + `.expect()` em `MirroredBuffer::clone` — **BAIXA**

### R31 · Evidência

* `src/loader/namb/parse.rs:75-87` — arquivos NAMB v1 com `crc32_header == 0` E seção de pesos
  vazia/toda-zerada pulam a verificação de integridade:

  ```rust
  let pesos_empty = pesos_raw.is_empty() || pesos_raw.iter().all(|&b| b == 0);
  if crc32_header == 0 && pesos_empty {
      log::warn!("CRC32 missing in NAMB v1 file (crc32=0 sentinel) — skipping integrity check. \
                  Support for NAMB v1 files without CRC is deprecated and will be removed...");
  }
  ```

* `src/dsp/mirror_buf.rs:200` — `Clone` delega para `try_clone()` com `.expect()`:

  ```rust
  fn clone(&self) -> Self {
      self.try_clone()
          .expect("MirroredBuffer::clone: allocation failed (use try_clone for fallible path)")
  }
  ```

### R31 · Diagnóstico

Ambos são riscos residuais já conhecidos e conscientemente aceitos em rodadas anteriores (R9 da
Rodada 1 documentou explicitamente a escolha de manter `Clone` com `.expect()` dado que
`try_clone()` está disponível para o caminho fallível real; o bypass de CRC32 do NAMB v1 está
marcado como deprecated no próprio log). Nenhum dos dois é um bug novo — são registrados aqui
para consolidar o registro de conformidade textual estrita com a regra "zero unwrap/expect" do
projeto (`.agents/rules/rust.md`), identificada nesta rodada por um sweep de conformidade
mecânica dedicado.

### R31 · Impacto

Bypass de CRC32: um atacante pode injetar um NAMB v1 com pesos zerados sem detecção — o modelo
resultante produz saída nula (silêncio), sem dano ao áudio além da perda de funcionalidade,
mitigado por warning já logado. `.expect()` no `Clone`: painic teórico apenas sob OOM real com
alocador customizado (não o padrão do Rust, que já aborta antes do `expect`).

### R31 · Proposta de solução

1. NAMB v1: remover o bypass de CRC32 na próxima major release, exigindo CRC32 válido para
   todos os arquivos v1 (paridade com v2).
2. `MirroredBuffer::Clone`: nenhuma ação obrigatória — decisão já documentada e aceita; se se
   desejar conformidade textual estrita, considerar remover `impl Clone` e exigir `try_clone()`
   explícito em todos os 2-3 call-sites remanescentes (baixo custo, baixo risco).

---

## Novas propostas do Pesquisador-Inovador — Rodada 3 (stable-only, pesquisadas via web)

### P11 · `shuttle` (AWS Labs) — concurrency testing randomizado complementar ao `loom` [ADIADO]

O projeto já adotou `loom` (P1, implementado) para model-checking exaustivo de interleavings
pequenos nos 3 protocolos SPSC/GC críticos — mas `loom` sofre de explosão combinatória em
espaços de estado grandes (o `gc_stress_1000_swaps`, 167s no tests-long, é inacessível ao
`loom`). [`shuttle`](https://github.com/awslabs/shuttle) (AWS Labs, v0.9, Rust stable) implementa
*randomized concurrency testing* via algoritmo PCT (Probabilistic Concurrency Testing, Microsoft
Research, ASPLOS 2016), com garantia probabilística >99.9999% de detecção de bugs não-
adversariais em casos grandes onde `loom` não escala — aplicável aos mesmos 3 protocolos já
modelados em `tests/loom_tests.rs`. Proposta: `shuttle = "0.9"` em dev-deps + `tests/shuttle_tests.rs`
com `#[cfg(shuttle)]`, mesmo padrão de cfg-flag do loom já estabelecido; fase dedicada em
`utils/tests-long.sh`. Custo: ~1-2h; risco zero (dev-dependency).

### P12 · `rtsan-standalone-rs` — RealtimeSanitizer para detecção de violações RT em runtime [ADIADO]

As rodadas de auditoria já encontraram duas violações reais de RT-safety (R5 e R18) através de
grep manual — a mitigação atual (`test_rt_logging_safety`, meta-teste estrutural) só detecta
padrões textuais conhecidos, não alocações implícitas do std, locks acidentais, ou syscalls
bloqueantes indiretas. [`rtsan-standalone-rs`](https://github.com/realtime-sanitizer/rtsan-standalone-rs)
é um wrapper Rust stable para o RealtimeSanitizer do LLVM (RTSan, LLVM 20+), que detecta **em
runtime** `malloc`/`free`/`pthread_mutex_lock`/syscalls bloqueantes em funções anotadas
`#[nonblocking]`. É a ferramenta que faltava para substituir os meta-testes grep frágeis por
detecção real de violação RT. Proposta: dev-dependency + anotar `process_block_internal`
(WaveNet), `DspPipeline::process`, callbacks `.process()` do PipeWire com `#[nonblocking]`;
rodar em modo debug no tests-long, complementando (não substituindo) os meta-testes grep
existentes. Custo: ~2-3h (build da lib C na primeira vez); risco baixo-médio (apenas
dev/testing, nunca no binário de release).

---

## Novos épicos (Rodada 3)

### EP-R13 — Robustez de carregamento de IR/WAV (R27) — **primeiro, único achado ALTA desta rodada com exploit trivial** [DONE]

Escopo: adicionar piso (`4_000 Hz`) e teto (`384_000 Hz`) de `sample_rate` no parser WAV de IR
(`src/dsp/cabsim/loader.rs`), eliminando o vetor de OOM catastrófico via upsampling e o caso
degenerado de sample_rate extremamente alto no mesmo gate. Critério de aceite: teste
`test_reject_ir_extreme_sample_rate` (baixo e alto) cobrindo o WAV de 44 bytes descrito no
achado; nenhuma alocação de resample pode exceder `MAX_IR_LENGTH * fator_máximo_de_upsampling`.
Risco: baixo (validação aditiva num parser já bem estruturado, sem tocar o motor de convolução).

### EP-R14 — Fidelidade de automação de parâmetros (R28 + R29 + R30) [DONE]

Escopo: implementar block-splitting para automação sample-accurate usando `event.header().time`
(R28); mover o push SPSC do `MainThread::flush()` para fora do loop de eventos + fallback via
`bump_generation()` em caso de `Full` (R29); inicializar o `ParamSmoother` com o valor atual dos
atomics em `activate()` em vez de `1.0` fixo (R30). Critério de aceite: teste de automação
sample-accurate com múltiplos eventos por bloco refletidos na saída de áudio;
`clap-validator` completo sem regressão; teste de 20 eventos em `flush()` inativo sem perda;
ausência de transiente audível em ciclo activate/deactivate com ganho não-unitário (verificável
via `regression_gate`/golden). Risco: médio (R28 é a mudança de maior escopo do épico, toca o
núcleo do processamento de eventos; R29/R30 são cirúrgicos e de baixo risco).

### EP-R15 — Fechar pendências residuais das Rodadas 2 e 3 (R8-h/doc + R14 + P5 + P10) [DOING]

Escopo mecânico, consolidando todas as pendências identificadas nas verificações desta rodada:

1. **R8-h (documentação)**: adicionar comentário em `src/common/spsc/gc.rs:276` explicitando a
   semântica ATUAL da flag (`RT_STATUS_GC_OVERFLOW` dispara apenas em overwrite/leak real, não
   na mera entrada em tier 3), prevenindo que uma futura auditoria repita o mesmo engano
   documentado na nota histórica desta rodada. Avaliar introduzir um segundo flag
   (`RT_STATUS_GC_TIER3` ou similar) se o sinal de "pressão do sistema" original ainda for
   valioso para diagnóstico.
2. **R14 (definitivo)**: integrar `tests/models/proptest_parsers.rs:270-512` a testes reais ou
   removê-lo (~240 linhas órfãs, pendente desde a Rodada 1); consolidar os dois wrappers de
   `generate_sine_440hz` num único local canônico.
3. **P5 (completar)**: migrar os 31 `#[allow(clippy::too_many_arguments)]` remanescentes para
   `#[expect(...)]`; ativar `[lints.clippy] allow_attributes = "warn"` em `Cargo.toml` para
   evitar regressão futura de `#[allow]` sem tracking.
4. **P10 (implementar)**: baseline inicial de `cargo semver-checks` contra a v3.0.0 atual; como
   o repositório não tem `.github/workflows/`, documentar o comando em `utils/` (ex.:
   `utils/semver-check.sh`) para execução manual pré-release, já que não há CI configurado.

Critério de aceite: cada sub-item fechado individualmente; nenhuma mudança de comportamento
sonoro. Risco: mínimo — execução mecânica de itens já especificados.

### EP-R16 — Higiene residual de conformidade (R31) [DOING]

Escopo: remover o bypass de CRC32 para NAMB v1 legado na próxima major release (R31.1);
decisão explícita (manter ou remover) sobre `impl Clone` de `MirroredBuffer` (R31.2, já
documentado como aceito — ação opcional). Critério de aceite: nenhum arquivo NAMB v1 sem CRC32
válido é aceito (breaking change intencional, deve ser comunicado no changelog); testes NAMB
existentes (`namb_test.rs`) atualizados para exigir CRC32 em todos os fixtures v1. Risco: baixo,
mas é uma **mudança de comportamento visível ao usuário** (arquivos v1 antigos sem CRC deixam de
carregar) — deve ser sinalizada com antecedência (major version bump), não é "risco zero" como
os demais itens mecânicos deste documento.

### EP-R17 — Segunda geração de verificação de concorrência e RT-safety (P11 + P12) [ADIADO]

Escopo: adicionar `shuttle` como complemento randomizado ao `loom` já existente, cobrindo os
mesmos 3 protocolos críticos em espaços de estado grandes (P11); avaliar `rtsan-standalone-rs`
para detecção real (não apenas textual/grep) de violações RT-safety, anotando as funções de
hot path com `#[nonblocking]` (P12). Critério de aceite: `tests/shuttle_tests.rs` rodando em
`tests-long.sh` sem falso-positivo; prova de conceito de `rtsan` detectando uma violação
conhecida (ex.: reintroduzir temporariamente um `log::error!` no hot path e confirmar que o
sanitizer o pega, depois remover). Risco: baixo — ambas são ferramentas de dev/teste, nenhuma
toca o binário de release.
