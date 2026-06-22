<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Revisão (revisor-auditor)

> Artefato gerado pela skill `revisor-auditor` e estruturado pela skill `planejador-arquiteto`.
> Data da auditoria: 2026-06-22. Baseado em revisão de código + análise de `testes.log` e dos logs de fase em `target/logs/`.
>
> **Foco da demanda:** otimizações com baseline **x86-64-v3** (AVX2/FMA/F16C/BMI2) em **toda** a base de código (AVX-512 fora de escopo agora), com atenção especial ao **DSP (PipeWire e CLAP)** e à **threading** em prol de baixa latência e ausência de soluços/artefatos.
>
> **Regras de referência:** `.agents/rules/rust.md` (RT-Safety §1, DSP §2, SIMD §3, Concorrência §4), `.agents/rules/testing.md`, `.agents/rules/copyright.md`, `.agents/rules/linting.md`. Mapa de paridade: `docs/cpp_parity_map.md`.

---

## Sumário Executivo

A base de código está, no geral, **muito bem arquitetada e disciplinada** quanto à RT-safety: o steady-state do hot-path é lock-free, sem alocação, sem I/O, com GC via SPSC, FTZ/DAZ, `mlockall`, `SCHED_FIFO`, afinidade de CPU e double-buffer com ordenações `Acquire`/`Release` corretas. O código de produção (`src/dsp/`, `src/loader/`) é livre de `panic!`/`unwrap!`. Não há marcadores `TODO`/`FIXME`/`HACK`.

A auditoria identificou, porém, **oportunidades concretas e de alto valor**, agrupadas em 4 Epics:

| Epic  | Tema                                                     | Severidade máx. | Impacto                                                  |
| ----- | -------------------------------------------------------- | --------------- | -------------------------------------------------------- |
| **A** | Otimizações SIMD x86-64-v3 no hot-path neural            | **Alta**        | Throughput de inferência (núcleo da demanda)             |
| **B** | RT-Safety: alocação/`mmap`/inferência na thread de áudio | **Alta**        | Soluços/XRuns em transições adaptativas e load de modelo |
| **C** | Saúde da suíte de testes/CI (insights de `testes.log`)   | **Alta**        | 2 fases sempre vermelhas mascaram regressões reais       |
| **D** | Organização, duplicação e documentação                   | Média           | Manutenibilidade                                         |

**Insight imediato de `testes.log`:** 2 das 6 fases da auditoria longa falham **de forma determinística** (não-flaky) — `Soak Tests (Numerical Stability)` e `CLAP Release Validation & Concurrency`. Ambas têm causa-raiz identificada (Epic C) e **não** indicam regressão de áudio em produção, mas tornam a auditoria permanentemente vermelha — o que esconde regressões futuras.

---

## EPIC A — Otimizações SIMD x86-64-v3 no hot-path neural

**Tema central da demanda.** Vários kernels do hot-path de inferência são **limitados pela latência da FMA** (poucos acumuladores independentes), não pela vazão. Em x86-64-v3 (Haswell+), a FMA tem latência ~4–5 ciclos e há **2 portas** de FMA: para saturar é preciso manter **~8–10 cadeias FMA independentes em voo**. Kernels com 1 acumulador rodam a ~12–25% da vazão teórica; com 4 acumuladores, a ~50%.

> Regra aplicável: `.agents/rules/rust.md` §3 — "Nam-rs is x86-64-v3 first: *Always* try to optimize to use modern ISA instructions."

Validação obrigatória para **todos** os itens deste Epic: paridade bit-a-bit/ESR contra o kernel escalar de referência (testes `*_parity`, `cpp_parity`, `golden_vectors`) + `cargo bench` (benches `inference_bench`, `dot_4x_bench`, `kahan_conv1d_bench`) antes/depois. Reordenar somas em ponto-flutuante **altera** o resultado bit-a-bit — usar a mesma estratégia de acumulação/Kahan já adotada nos kernels de referência e revalidar tolerâncias.

---

## A1 — [ALTA] Conv CH=3 do A2: cadeia FMA totalmente serial (1 acumulador) [DONE]

**Arquivo:** `src/models/a2/conv1d_ch3/simd.rs:31-92` (`conv1d_ch3_k6_f32`) e `:97-176` (`conv1d_ch3_k15_f32`).

**Problema.** Ambos os kernels acumulam **todas** as FMAs no mesmo registrador `acc` (`__m128`):

- K=6: 6 taps × 3 canais de entrada = **18 FMAs encadeadas** em `acc`.
- K=15: 15 × 3 = **45 FMAs encadeadas** em `acc`.

Como cada `_mm_fmadd_ps(wv, sv, acc)` lê o `acc` escrito pela FMA anterior, há **uma única cadeia de dependência**. Caminho crítico ≈ 18×4 = 72 ciclos (K6) ou 45×4 = 180 ciclos (K15), quando o limite de vazão (2 portas) seria ~9 e ~23 ciclos. **Ineficiência de ~3–4×** no kernel de conv do A2-Lite (CH=3) — exatamente o caminho dominante do `Prewarm_A2Lite_CH3` (2.89 ms no bench).

Agravante: a conv é executada **frame-a-frame** (`layer_forward_ch3_block` em `:271-275` chama `conv1d_ch3_f32_dispatch` por frame); só o pós-conv (mixin/LeakyReLU/head/l1x1) é vetorizado em pares. Logo a conv — parte mais cara — não tem ILP nem entre taps nem entre frames.

**Proposta.**

1. **Solução barata (intra-frame):** usar 3–4 acumuladores `__m128` independentes (ex.: `acc0` para taps 0,3,6…; `acc1` para 1,4,7…; `acc2` para 2,5,8…), somando no fim. Quebra a cadeia → caminho crítico cai ~3×.
2. **Solução superior (inter-frame, recomendada):** adotar *tiling* de T frames como o kernel CH=8 já faz (ver A2), com T acumuladores, fazendo o laço **tap-major** e reaproveitando a coluna de pesos `wv` carregada uma vez por (tap, canal) entre os T frames. Isso quebra a cadeia **e** amortiza loads de peso. Empacotar 2 frames por `__m256` (CH=3+pad=4 → 8 lanes) é viável, como já feito no pós-conv.

**Risco:** baixo (kernel isolado, ampla cobertura de testes de paridade `conv1d_ch3::tests::*`). **Esforço:** médio.

---

## A2 — [MÉDIA] Conv CH=8 do A2: T=4 subutiliza as portas FMA (~50%) [DONE]

**Arquivo:** `src/models/a2/conv1d_ch8/simd.rs:22-95` (`conv1d_ch8_t4_avx2`).

**Problema.** O *tiling* T=4 mantém **4 acumuladores** `__m256` (`a0..a3`, um por frame), todos dependentes da mesma carga `wcol`. Com 2 portas FMA e latência ~4–5 ciclos, são necessárias ~8 cadeias em voo; com 4, a utilização fica em **~50%**. É o caminho dominante do A2-Full (`Prewarm_A2Full_CH8` = 3.82 ms no bench).

**Proposta.** Elevar para **T=8** (8 acumuladores `a0..a7`). Pressão de registradores: 8 acumuladores + `wcol` + 1 temporário de broadcast ≈ 10 YMM (cabe nos 16 YMM do x86-64-v3). Os broadcasts (`_mm256_set1_ps`) usam porta de shuffle/load e não competem diretamente com as FMAs. Alternativa: manter T=4 mas processar 2 grupos de canais de saída por iteração para dobrar as cadeias independentes.

**Risco:** baixo–médio (revalidar `conv1d_ch8::tests::*` e `test_conv1d_ch8_t4_tail_parity`). **Esforço:** médio.

---

## A3 — [ALTA] Conv agrupada (grouped) do A2 usa anti-padrão escalar — e já existe versão SIMD pronta, porém *morta* [DONE]

**Arquivo:** `src/models/a2/grouped_conv1d.rs`.

**Problema 1 — extract/reinsert escalar por iteração.** O caminho **ativo** (`process_single_frame` `:234` e `process_block` `:272`) chama `process_single_frame_avx2` (`:399-521`), cujo laço interno (`:495-503`) faz, **a cada iteração**:

```rust
let acc = _mm_setr_ps(r0, r1, r2, r3);          // reinsere 4 escalares (≥4 µops)
let acc = _mm_fmadd_ps(wv, sv, acc);            // 1 FMA
r0 = _mm_cvtss_f32(acc);                        // extrai lane 0
r1 = _mm_cvtss_f32(_mm_shuffle_ps(acc, acc, 0x55)); // ...
r2 = _mm_cvtss_f32(_mm_shuffle_ps(acc, acc, 0xAA));
r3 = _mm_cvtss_f32(_mm_shuffle_ps(acc, acc, 0xFF));
```

A FMA fica **soterrada** por ~8 µops de shuffle/move por iteração. Utilização da FMA ≈ 12%.

**Problema 2 — versão correta existe mas não é chamada.** A função `grouped_conv1d_single_frame_simd` (`:645-757`) implementa **exatamente a mesma semântica** mantendo o acumulador em `__m128` nativo (`acc = _mm_fmadd_ps(wv, sv, acc)` em `:741`); seu próprio doc-comment (`:639-643`) diz que "replaces extract pattern above ... yields better code generation". **Porém ela tem ZERO chamadores** (confirmado por `grep`): é código morto enquanto o caminho lento permanece ativo.

**Proposta.**

1. **Imediato (baixíssimo risco):** redirecionar `process_single_frame` e `process_block` para `grouped_conv1d_single_frame_simd`; remover `process_single_frame_avx2`. Ganho direto eliminando ~8 µops/iteração.
2. **Em seguida:** mesmo a versão "boa" usa **1 acumulador** por bloco de 4 saídas (cadeia serial sobre `ik`×`ic`). Aplicar múltiplos acumuladores (ex.: 1 por tap, ou *tiling* de blocos de saída) para quebrar a cadeia FMA.

**Risco:** baixo (semântica idêntica, validar `grouped_conv1d::tests::*`). **Esforço:** baixo (item 1), médio (item 2).

---

## A4 — [MÉDIA] Conv depthwise agrupada é 100% escalar (sem SIMD) [DONE]

**Arquivo:** `src/models/a2/grouped_conv1d.rs:532-598` (`process_single_frame_depthwise_avx2`), laço `:582-591`.

**Problema.** Apesar do sufixo `_avx2`, o laço interno é escalar puro:

```rust
for c in 0..ch {
    let mut acc = 0.0f32;
    for k in 0..kernel {
        acc += *w_ptr.add(c*group_stride + k*4) * *tap.add(c);
    }
}
```

Na conv depthwise (1 canal/grupo) os canais são triviais de vetorizar: processar 8 canais por vez com `_mm256_loadu_ps` de pesos e estados ao longo dos canais + `_mm256_fmadd_ps`. Para uma camada depthwise de 32 canais, ~8× menos FMAs.

**Proposta.** Vetorizar o laço de canais (8 por iteração, AVX2), com cauda escalar. Atenção ao layout de pesos `c*group_stride + k*4` (stride por canal): pode exigir um *gather* leve ou reorganização do layout no carregamento do modelo (preferível: reorganizar no `set_weights`, fora do hot-path).

**Risco:** médio (correção de layout + paridade `test_grouped_conv1d_depthwise`). **Esforço:** médio.

---

## A5 — [ALTA] `dot_product_4x_f32_avx2` (conv de entrada do WaveNet): 1 acumulador [DONE]

**Arquivos:** `src/math/gemm/dot_4x/dot_f32_avx2.rs:33-50` (`dot_product_4x_f32_avx2`, 1× `__m128`) e `:73-104` (`dot_product_4x_f32_dual_avx2`, 1× `__m256`).
**Chamador em produção:** `src/models/wavenet/conv_input.rs:118` (caminho de conv de entrada do WaveNet com pesos f32 nativos).

**Problema.** Acumulador único → cadeia FMA serial a ~0,25 FMA/ciclo (~6% da vazão). Para `in_ch=8, kernel=6` são 48 FMAs encadeadas (≥192 ciclos só de latência).

**Proposta.** Desenrolar 4× com acumuladores independentes (`acc0..acc3`) com passo intercalado em `i`, somando no fim. Aplicar o mesmo à variante dual (4× `__m256`). Ganho potencial ~4×.

**Risco:** baixo (kernels com testes `dot_4x_test::*` e referência escalar bit-idêntica documentada em `scalar_ref/dot.rs:119`). **Esforço:** baixo.

---

## A6 — [ALTA] GEMV de 4 gates do LSTM e GEMM-batch: poucos acumuladores + `vcvtph2ps` no laço quente [DONE]

**Arquivos:** `src/math/gemm/gemv_4gate/avx2.rs:19-128` (`gemv_4gate_avx2`); `src/math/gemm/gemm_batch/avx2.rs:23-119` e `:128-247`.

**Problema 1 — latência FMA.** `gemv_4gate_avx2` usa 4 acumuladores (um por gate I/F/G/O), cada um acumulando sobre **todo** `in_len`. Em LSTM com hidden grande (`in_len≈128`), cada cadeia tem ~128 FMAs (≥512 ciclos). 50% de utilização. Idem para os 4 acumuladores (1/frame) do `gemm_batch`.

**Problema 2 — decode f16 no hot-loop.** `_mm256_cvtph_ps` aparece no laço interno (`gemv_4gate/avx2.rs:66,71,76,81`; `gemm_batch/avx2.rs:64`). Em Skylake/Cascade Lake, `vcvtph2ps` tem latência ~7 ciclos e vazão limitada, podendo **dominar** o laço. No `gemm_batch` o peso é carregado 1× e reusado entre 4 frames (boa amortização); no `gemv_4gate` cada peso é decodificado por gate.

**Proposta.**

1. Processar 2 `in_c` por iteração com 8 acumuladores (cada gate com `acc_lo`+`acc_hi`) → dobra a vazão FMA.
2. Pré-deserializar a linha de 8 pesos f16→f32 uma vez por iteração (ou intercalar o decode de dois blocos de pesos para esconder a latência do `vcvtph2ps`).

**Risco:** médio (LSTM é sensível; revalidar `lstm::tests::*`, `lstm_scalar_bf16_parity`, `lstm_gate_bf16_parity`). **Esforço:** médio.

---

## A7 — [MÉDIA] Macro `gemv_kernel!`: 4→8 acumuladores e cauda escalar ineficiente [DONE]

**Arquivos:** `src/math/gemm/gemv/kernel_macro.rs:15-88`; `src/math/gemm/gemv/f16_avx2.rs:20-64`; `src/math/gemm/gemv/f32_avx2.rs:68-121`.

**Problema.** A macro central de GEMV (usada por `fused_add_gemv`, `gemv_overwrite`, `gemv_with_bias_f32`, `gemv_no_bias_f32`) mantém 4 acumuladores → ~50% das portas FMA. Adicionalmente, a cauda escalar do f32 (`f32_avx2.rs:99-104`) usa `_mm256_set1_ps`+`_mm256_fmadd_ps` (256-bit) para trabalho de 1 elemento — desperdício de 7/8 lanes; deveria usar `_mm_fmadd_ss`.

**Proposta.** Expandir para 8 acumuladores (processar 8 linhas por iteração; os offsets de peso já são *strided*). Trocar a cauda escalar por `_mm_fmadd_ss`. Como a macro é compartilhada, o ganho propaga para todo GEMV (camadas densas, head, 1×1).

**Risco:** médio (afeta muitos chamadores; ampla cobertura `gemm::*`, `cpp_parity`). **Esforço:** médio.

---

## A8 — [BAIXA] Padé tanh/sigmoid: avaliar `rcp_ps`+1 Newton-Raphson no lugar de `div_ps`

**Arquivos:** `src/math/activations/tanh/production.rs:58,102-103,145`; `src/math/activations/sigmoid.rs`.

**Observação.** O `div_ps` é a operação mais cara (latência ~11–14 ciclos, mal pipelinada). O caminho dual (`simd_tanh_dual_avx2`) já sobrepõe duas divisões — bom. Substituir `div` por `_mm256_rcp_ps` (12 bits) + 1 iteração de Newton-Raphson (~23 bits) atende à tolerância atual de tanh (erro ~2.3e-3) e pode ~dobrar a vazão da ativação.

**Proposta (com cautela).** Prototipar e medir; **só** adotar se a paridade contra `f32::tanh`/golden mantiver erro < −80 dB (regra `rust.md` §2). Risco real de quebrar `golden_vectors`/`cpp_parity` e fidelidade perceptual. Por isso, **baixa prioridade** e gate por feature/medições.

**Risco:** médio (fidelidade). **Esforço:** médio.

---

## EPIC B — RT-Safety: alocação/`mmap`/inferência na thread de áudio

> Regra aplicável: `.agents/rules/rust.md` §1 — "Zero Heap Drop ... Heap objects must never go out of scope on the RT thread"; e §1 "Zero Blocking I/O". O steady-state cumpre isso; as **transições** abaixo violam.

## B1 — [ALTA] Rebuild "slimmable" executa alocação + *prewarm* (inferência!) + `mmap` na thread de áudio

**Arquivos:** `src/models/slimmable.rs:314-343` (`try_slimmable_rebuild_single`).
**Chamadores na thread de áudio:** CLAP — `src/clap/processor/events.rs:154` (dentro de `process_events`); PipeWire — `src/standalone/pw_host/capture/setup.rs:132-140` (dentro do `process` callback).

**Problema.** Numa transição do FSM de compute adaptativo, a própria thread de áudio executa sincronamente:

- `w.slice_channels(target_ch)` → várias `AlignedVec::new()` (`std::alloc::alloc`) — `slimmable.rs:76,90,129,138`;
- `new_model.prewarm()` (`:327`) → **roda inferência por todo o campo receptivo** (centenas de amostras);
- `set_max_buffer_size(max)` (`:329`) → `MirroredBuffer::new` → **`mmap`** (`src/dsp/mirror_buf/alloc.rs:44`) no caso CLAP;
- 2× `Box::new(...)` (`:333`).

**Ironia crítica de latência:** o downsize adaptativo existe para **aliviar** a CPU quando o sistema está sob pressão — mas o rebuild gera um **pico de latência** (alloc + `mmap` + inferência de prewarm) **exatamente** no pior momento, podendo causar o XRun que se tentava evitar. O comentário em `slimmable.rs:305` ("Must be called before DSP to keep the hot-path zero-alloc") garante apenas que o *per-sample* fica zero-alloc; a transição, não.

**Proposta.** Espelhar o padrão já usado no load de modelo: o FSM adaptativo **sinaliza** o `target_ch` desejado (atomic/flag); uma **thread worker/main** constrói + `prewarm` + dimensiona o modelo *slimmed* fora da RT e o envia via SPSC (`rtrb`); o callback de áudio só faz o **swap de ponteiro** e manda o antigo para o GC. Elimina 100% da alocação/`mmap`/inferência da thread de áudio.

**Risco:** médio (nova coordenação de threads; cobrir com `container_slimmable`/`concurrency_stress`). **Esforço:** médio–alto. **Prioridade:** a mais alta do Epic B.

---

## B2 — [MÉDIA] `cold_load_model` → `set_max_buffer_size` → `MirroredBuffer::new`/`mmap` na thread de áudio (CLAP)

**Arquivo:** `src/clap/processor/events.rs:157-199` (`cold_load_model`, `set_max_buffer_size` em `:172`) → `src/models/.../dynamic.rs:487` → `src/dsp/mirror_buf/alloc.rs:44` (`mmap`).

**Problema.** No swap de modelo (drenado do SPSC dentro de `process_events`), se `max_frames_count` exceder a capacidade atual do modelo, há `AlignedVec::new()`/`mmap` na thread de áudio. É *cold path* e guardado por early-return, mas ainda viola "Zero Heap" e pode causar pico no primeiro buffer pós-load.

**Proposta.** Dimensionar o buffer **antes** de empurrar via SPSC: como o host informa `max_frames_count` em `activate()`, pré-chamar `set_max_buffer_size` na thread main (no `load_model`, `src/clap/plugin/main_thread/load.rs`) para que o modelo chegue à RT já dimensionado. O swap fica puramente ponteiro.

**Risco:** baixo (caminho de load já é off-RT). **Esforço:** baixo.

---

## B3 — [BAIXA] `host.request_callback()` (syscall) na thread de áudio em mudança de latência

**Arquivo:** `src/clap/processor/events.rs:117`.

**Problema.** Em transição de latência (raro: ativação/swap), `request_callback()` pode disparar wake do host (write em eventfd/pipe). Syscall na thread de áudio.

**Proposta.** Aceitável por ser *cold path*; alternativamente, sinalizar via atomic e deixar o host detectar no próximo ciclo. Documentar explicitamente como exceção consciente.

**Risco:** baixo. **Esforço:** baixo.

---

## EPIC C — Saúde da suíte de testes / CI (insights de `testes.log`)

Duas das seis fases da auditoria longa falham de forma **determinística**. Nenhuma indica regressão de áudio em produção, mas mantêm a auditoria vermelha e **mascaram** regressões futuras. Correção prioritária para restaurar o valor de sinalização do CI.

## C1 — [ALTA] Fase "CLAP Release Validation" sempre falha: 16 testes `should_panic` dependem de `debug_assert!` desligado em release

**Evidência:** `target/logs/phase4-clap-validation.log:1143-1216` → `test result: FAILED. 804 passed; 16 failed`. Invocação: `utils/tests-long.sh:438` (`cargo test --release ... --lib`).

**Causa-raiz.** Os 16 testes `should_panic` em `models::a2::grouped_conv1d::tests::*` e `models::a2::model::set_weights::test_set_layer_film_out_of_range` validam guardas que são `debug_assert!` (ex.: `grouped_conv1d.rs:115-124,194-221,254-267`). Em **release**, `debug-assertions` está **desligado** por padrão, então:

- Vários "did not panic as expected" (`phase4...log:1146-1196`) — a guarda some e a função **prossegue** (terreno de UB silencioso, pois o hot-path usa `get_unchecked`).
- Outros entram em pânico **cru** em vez da mensagem controlada: "attempt to divide by zero" (`new_zero_groups`, `grouped_conv1d.rs:126`), "index out of bounds: len is 95 but index is 95" (`new_mismatched_weight_len`, `:153`).

**Duplo achado.** (a) **Higiene de CI:** a fase nunca passa. (b) **Segurança em release:** toda a validação de bounds/divisibilidade de `grouped_conv1d`/`set_layer_film` é **debug-only**; no plugin CLAP/host PipeWire de produção (release) essas guardas **não existem** e entradas malformadas atingem `get_unchecked` (UB) ou pânico cru na thread de áudio (unwind = quebra de RT-safety, regra `rust.md` §1 "Panic Prevention").

**Proposta.**

1. **CI:** ou anotar esses testes com `#[cfg_attr(not(debug_assertions), ignore)]`/`#[cfg(debug_assertions)]`, ou rodar a invocação `--lib` da fase com `RUSTFLAGS="-Cdebug-assertions=on"` (em `utils/tests-long.sh:438`). Recomendado: rodar `--lib` com debug-assertions on, preservando a intenção dos testes.
2. **Segurança:** documentar e **garantir** os invariantes de carga (no loader/`set_weights`) que tornam sólido o uso de `get_unchecked` no hot-path; considerar promover as checagens verdadeiramente críticas (ex.: `groups > 0`, casamento de tamanhos) a validação *de carregamento* (não-RT), retornando `Result` em vez de `debug_assert!`.

**Risco:** baixo (mudança de teste/CI) + médio (revisão de invariantes). **Esforço:** baixo–médio.

---

## C2 — [ALTA] Fase "Soak/Numerical Stability" sempre falha: `test_lstm_noise_soak` exige RMS≠0 de um modelo de **pesos zero**

**Evidência:** `target/logs/phase1-soak.log:7-10` → `panicked at tests/soak_test.rs:384: LSTM RMS do loop out of range: 0 at 0 frames`.

**Causa-raiz.** O teste (`tests/soak_test.rs:359-397`) cria `LstmModel2::<16,17,32,64>::new()` e afirma `rms > 0.0001` (`:384-389`). Mas `LstmModel2::new()` (`src/models/lstm/model2.rs:98-107`) inicializa **todos os pesos a zero** (`head_weights: [0u16; H]`, `head_bias: 0.0`, `LstmLayer::new()` zerado). Um modelo de pesos zero emite **silêncio** para qualquer entrada → RMS = 0 → falha **na primeira iteração** (`0 frames`). O teste é **logicamente impossível** de passar como está (o gêmeo `test_lstm_silence_soak` passa porque alimenta silêncio e não exige RMS).

**Proposta.** Antes do soak de ruído, popular pesos não-triviais: (a) randomizar via PRNG determinístico (o teste já tem `SimplePcg`), ou (b) carregar uma fixture LSTM real (como em `cpp_parity`/`a2_loader`). Manter a asserção de divergência (`rms < 10.0`) para o objetivo original (estabilidade sob ruído).

**Risco:** baixíssimo (apenas teste). **Esforço:** baixo.

---

## EPIC D — Organização, duplicação e documentação

## D1 — [MÉDIA] Duplicação AVX2/AVX-512 (~4.600 linhas; 7 de 10 pares sem macro)

**Evidência (pares estruturalmente ~90–95% idênticos):** `math/dsp/gain/{avx2,avx512}.rs`; `math/dsp/stereo/convolution_{avx2,avx512}.rs`; `math/gemm/gemv_4gate/{avx2,avx512}.rs`; `math/wavenet/accumulate/{avx2,avx512}.rs`; `math/gemm/dot_4x/{avx2,avx512}.rs` e `{avx2_dual,avx512_dual}.rs`; `math/gemm/gemm_batch/{avx2,avx512}.rs`; `math/gemm/dot_4x/{dot_f32_avx2,dot_f32_avx512}.rs`.

**Problema.** Já existe infraestrutura de macros (`activations/kernel_macro.rs`, `gemm/gemv/kernel_macro.rs`, `dsp/gain/kernel_macro.rs`), mas 7/10 pares não a usam → correções (como as do Epic A) precisam ser feitas **duas vezes** e podem divergir. Como AVX-512 está fora de escopo agora, isto é **dívida técnica** a endereçar ao tocar nos kernels do Epic A: ao otimizar o AVX2, encapsular o corpo numa macro parametrizada por largura/`target_feature` para futura reuso.

**Proposta.** Generalizar a macro de kernel por largura de vetor (`$VEC`, `$set1`, `$fmadd`, `$load`) e cauda (mascarada vs escalar), aplicando primeiro aos kernels alterados no Epic A. **Não** expandir escopo para AVX-512 agora — apenas deixar o AVX2 macro-ável.

**Risco:** médio. **Esforço:** médio.

## D2 — [BAIXA] Structs `A2Conv1dCh3`/`A2Conv1dCh8` quase idênticas → const-generic

**Arquivos:** `src/models/a2/conv1d_ch3/mod.rs:63` e `src/models/a2/conv1d_ch8/mod.rs:52` (diferem só em contagem de canais e stride 16 vs 64). **Proposta:** unificar como `A2Conv1dCh<const CH: usize>`, reduzindo ~440 linhas e divergência. **Risco:** médio. **Esforço:** médio.

## D3 — [BAIXA] Arquivos grandes com responsabilidades misturadas

**Arquivos:** `src/models/a2/conv1d.rs` (1006 linhas: transposição de pesos + dispatch de inferência + lógica grouped) e `src/models/a2/model/dynamic.rs` (880 linhas: engine dinâmico + ~15 helpers de transposição). **Proposta:** extrair helpers de transposição/layout de pesos para submódulo dedicado (ex.: `a2/weights_layout.rs`), separando "carregamento/layout" de "inferência". **Risco:** baixo. **Esforço:** médio.

## D4 — [BAIXA] Lacunas de doc-comment em módulos de dispatch do hot-path

**Itens públicos sem `///` (têm doc de módulo, faltam por-função):** `src/math/dsp/stereo/mod.rs:40,48,56,64,74`; `src/math/dsp/gain/mod.rs:75,86,112,123`; `src/dsp/pipeline/stages/bridge.rs:12`; `src/dsp/pipeline/stages/input.rs:36`; `src/dsp/gate_flags.rs:22`; `src/dsp/telemetry.rs:13`. **Proposta:** adicionar doc-comments (contrato, `# Safety` para `unsafe`). **Risco:** nulo. **Esforço:** baixo.

## D5 — [INFO] Código morto: `grouped_conv1d_single_frame_simd`

Ver **A3**: hoje sem chamadores. Será resolvido ao redirecionar o caminho ativo para ela (ou removido se o item 2 do A3 a substituir). Evitar `#[allow(dead_code)]` — integrar ou remover.

## D6 — [INFO] `DISABLE_GATE` é estática global compartilhada entre instâncias CLAP

**Arquivo:** `src/dsp/pipeline/stages/input.rs:23` (`pub static DISABLE_GATE: AtomicBool`). Lida em `:78`. Em DAW com múltiplas instâncias, afeta **todas**. Documentada como uso de profiling/bench. **Proposta:** garantir que nunca seja escrita por código de produção; idealmente mover para estado por-instância ou atrás de `#[cfg(feature = "testing")]`. **Risco:** baixo. **Esforço:** baixo.

---

## Itens conhecidos / já rastreados (sem ação nova aqui)

- **Paridade WaveNet Lite CH=12 (SNR ~0.9 dB):** lacuna arquitetural conhecida e `#[ignore]` — `docs/cpp_parity_map.md:257` (P1). Mantido sob rastreio do mapa de paridade.
- **A2 FiLM/gating:** parser-only, engine não conectado (`docs/cpp_parity_map.md:153,158-161`) — superfície de forward-compat intencional.
- **Higiene geral:** 0 `TODO/FIXME/HACK`; código de produção em `src/dsp/` e `src/loader/` livre de `unwrap/expect/panic`; ordenações `Acquire/Release` do `DspBridge` e do `gui_param_generation` corretas; GC em 3 camadas (SPSC→parking lot→overflow) com dealloc na main thread — **tudo conforme** `rust.md` §1/§4.

---

## Priorização sugerida (visão do planejador)

1. **Sprint 1 (vermelho→verde + ganhos rápidos):** C1, C2 (destrava CI), A3-item1 (redirecionar grouped conv), A5 (dot_4x f32), B2.
2. **Sprint 2 (núcleo de performance):** A1, A2, A6, A7 (+ macro-ização A2-baseline do D1 conforme tocados).
3. **Sprint 3 (RT-safety estrutural):** B1 (rebuild off-thread), A4 (depthwise SIMD), A3-item2.
4. **Backlog:** A8 (com cautela de fidelidade), D2, D3, D4, D6.

> Observação metodológica: cada item de Epic A/B deve seguir o ciclo `cargo check` → testes de paridade/golden → `cargo bench` (regra `.agents/rules/linting.md`), e preservar o cabeçalho SPDX (regra `.agents/rules/copyright.md`). A criação de `TODO-sprints.md` (epics/sprints/tasks atômicas) deve ser feita pela skill `planejador-arquiteto` **quando solicitada**.
