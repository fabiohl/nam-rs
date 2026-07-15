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

## R4 · Panic hook aloca heap e adquire `RwLock` no caminho de crash — pode deadlockar exatamente quando mais se precisa dele — **ALTA**

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

### EP-R2 — Ciclo de vida à prova de host hostil (R2 + R13 + R11 + R3) [DOING]

Escopo: `Arc<DialogSharedState>` nas threads de diálogo + join com deadline no destroy da GUI

* drenagem final (ou leak documentado+testado) do GC + documentação do double-SIGINT.
  Critério: novos testes de lifecycle destrutivo (destroy com diálogo aberto; destroy com GC em
  trânsito) verdes no tests-long; `clap-validator` completo sem regressão. Risco: médio
  (lifecycle CLAP tem sutilezas de thread; usar o harness `clack-host` já existente).

### EP-R3 — Formalização da concorrência (R8 completo + P1)

Escopo: aplicar a tabela R8 (8 correções de uma linha + comentários de pareamento) e
introduzir os testes `loom` dos 3 protocolos (P1). Ordem interna: primeiro loom modelando o
estado **atual** (deve falhar em R8-a/b/f — prova do achado), depois as correções (loom passa).
Critério: `cargo test --cfg loom` (job novo no tests-long) verde; zero mudança de asm nos hot
paths x86 (conferir com P4/dsp_hotpath.asm). Risco: baixo.

### EP-R4 — Blindagem da malha de QA (R6 + R7 + R4 + R5)

Escopo: chave composta exata no contrato + meta-teste de prefixo (R6); gate "≥1 passed" em
todas as fases do tests-long (R7); panic hook zero-alloc com snapshot pré-capturado (R4);
`RtStatusFlags` no lugar do `log::error!` + meta-teste grep de RT-safety (R5). É o épico que
protege todos os outros: fissuras de QA deixam regressões dos EP-R1/R2/R3 invisíveis.
Critério: regravar contrato com `--save` e conferir manualmente as linhas A2-Full v1/v2
distintas; teste do panic hook com alloc_audit = 0; fase PipeWire falhando artificialmente com
filtro vazio (teste do gate). Risco: baixo.

### EP-R5 — Higiene e superfície (R12 + R14 + R15 + R16 + P3)

Escopo: sprint mecânico de documentação SAFETY (tabela R12), remoção de mortos/duplicados
(R14), `testing` fora do default com medição `cargo bloat` (R15), limpezas de log/refs (R16),
e adoção incremental de `assert_unchecked`/`as_chunks` onde reduzir unsafe sem regredir asm
(P3, validado por P4). Critério: `cargo clippy --all-targets` limpo, contagem de blocos
`unsafe` em produção reduzida e 100% com SAFETY específico, quick suite verde. Risco: mínimo —
ideal para absorver com as skills `refatora-rust`/`refatora-doc`.

### EP-R6 (opcional/contínuo) — Guardas de segunda ordem (P2 + P4)

Escopo: `utils/mutants.sh` (rodada mensal off-line, módulos-fortaleza) e `utils/asm-gate.sh`
(baseline de codegen sobre o dsp_hotpath.asm já gerado). Nenhum bloqueio sobre os demais
épicos; entrega valor composto ao longo do tempo. Risco: zero (ferramentas externas, nada em
produção).
