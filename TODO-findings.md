<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Bug Hunting (NAM-rs)

> **Skill:** `revisor-auditor` (papel **"Ace of bug hunting"**) → consolidado via `planejador-arquiteto`.
> **Escopo:** varredura profunda da base de código em busca de bugs, vetores de ataque,
> quebras de estabilidade, violações de RT-safety e código morto/inconsistente.
> **Data:** 2026-06-24.

## Metodologia e legenda

A análise combinou mapeamento por subagentes especializados (loader/parsing, concorrência,
SIMD/unsafe, DSP/modelos) **seguido de verificação manual direta no código-fonte** de cada
achado relevante (cada finding abaixo foi confirmado lendo o arquivo citado, não apenas
reportado). Achados não confirmados na verificação foram descartados (ex.: a mensagem de erro
`expected_len` em `namb/parse.rs:90` está, na verdade, correta; vários `panic!` apontados são
`#[cfg(test)]`).

| Severidade        | Significado                                                                                              |
|:-----------------:|:-------------------------------------------------------------------------------------------------------- |
| 🔴 **Crítica**    | Memória/segurança/UB explorável por entrada não confiável, ou quebra de RT-safety com glitch garantido.  |
| 🟠 **Alta**       | DoS por entrada não confiável, UB latente em hot-path, ou divergência funcional audível.                 |
| 🟡 **Média**      | Fragilidade/precondição não documentada que vira bug sob mudança de contexto; inconsistência doc↔código. |
| 🔵 **Baixa/Info** | Robustez defensiva, completude de verificação, dívida de manutenção.                                     |

Status legenda: ⬜ aberto · 🟦 em análise · ✅ resolvido.

---

## 🔴 F1 — `AlignedVec::from(Vec<T>)` faz panic com vetor vazio (guarda em código morto) [DONE]

- **Status:** ✅
- **Local:** [`src/math/common/aligned.rs:218-227`](src/math/common/aligned.rs#L218)
- **Severidade:** 🔴 Crítica (panic determinístico; viola "Panic Prevention" de `.agents/rules/rust.md`).

### F1 — Descrição

```rust
impl<T: Copy> From<Vec<T>> for AlignedVec<T> {
    fn from(v: Vec<T>) -> Self {
        let mut aligned = Self::new(v.len(), v[0]); // v[0] works if len > 0
        if v.is_empty() {
            return Self::with_capacity(0);
        }
        aligned.copy_from_slice(&v);
        aligned
    }
}
```

A expressão `v[0]` é **avaliada incondicionalmente antes** da checagem `if v.is_empty()`.
Para um `Vec` vazio, `v[0]` dispara `panic!("index out of bounds: the len is 0 but the index is 0")`
**antes** de a guarda de vazio poder ser alcançada — ou seja, o bloco `if v.is_empty()` é
**código morto** para o caso que ele pretende tratar. O próprio comentário (`// v[0] works if len > 0`)
denuncia o defeito.

### F1 — Causa-raiz

Ordem de avaliação invertida: a guarda foi escrita *depois* da operação que ela deveria proteger.

### F1 — Impacto / alcance

`AlignedVec::from(...)` é usado no caminho de carregamento de modelos A2:

- [`src/models/a2/model/dynamic/build.rs:122,149,170`](src/models/a2/model/dynamic/build.rs#L122) —
  `conv_b`, `mixin_w`, `l1x1_b` construídos a partir de `read_slice(conv_out / channels)`.
- [`src/models/a2/model/set_weights.rs:88,128,145`](src/models/a2/model/set_weights.rs#L88).
- [`src/models/a2/grouped_conv1d.rs:133`](src/models/a2/grouped_conv1d.rs#L133) — `AlignedVec::from(raw_bias.to_vec())`.

No A2 dinâmico, `conv_out = bottleneck * 2` (gated) ou `bottleneck`. Se a topologia validar e
deixar passar `bottleneck == 0` (ou um `channels == 0`/bias ausente em alguma geometria de
container), o `read_slice(0)` retorna fatia vazia → `to_vec()` vazio → `AlignedVec::from(vec![])`
→ **panic na thread de carregamento** (DoS por arquivo de modelo malformado). Mesmo que hoje a
validação de topologia barre todos os caminhos de comprimento zero, a guarda inoperante é um
panic latente e um erro de lógica que deve ser corrigido na origem.

### F1 — Proposta de solução

Fazer `From` delegar para o já-correto `from_vec` (que checa `is_empty()` primeiro):

```rust
impl<T: Copy> From<Vec<T>> for AlignedVec<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        Self::from_vec(v)
    }
}
```

Adicionar teste `aligned_from_empty_vec_is_safe()` cobrindo `AlignedVec::<f32>::from(Vec::new())`.

---

## 🟠 F2 — Dimensões de modelo sem limite superior + alocação antes da validação → DoS (e arit. sem checagem em release) [DONE]

- **Status:** ✅ Resolvido (2026-06-24)
- **Locais principais:**
  - [`src/loader/nam_json/topology/wavenet.rs:203-233`](src/loader/nam_json/topology/wavenet.rs#L203) (`kernel_size`, nº de dilations, nº de arrays e `head_size` sem teto).
  - [`src/loader/nam_json/topology/convnet.rs:48-72`](src/loader/nam_json/topology/convnet.rs#L48) (`channels` **e** `kernel_size` **e** `dilations` sem teto).
  - [`src/loader/nam_json/topology/linear.rs:14`](src/loader/nam_json/topology/linear.rs#L14) (`receptive_field` sem teto — menor severidade, limitado por `MAX_WEIGHTS`).
  - [`src/loader/dispatcher/wavenet/layout.rs:23-26`](src/loader/dispatcher/wavenet/layout.rs#L23) (alocação **antes** de `read_slice`).
  - [`src/loader/dispatcher/wavenet/dynamic.rs:60,68,193-203`](src/loader/dispatcher/wavenet/dynamic.rs#L193) (`k` não confiável flui para alocações).
- **Severidade:** 🟠 Alta (DoS por entrada não confiável; risco de corrupção não totalmente descartável).

### F2 — Descrição

As proteções anti-DoS/OOM (Sprint 5 / Task 5.2) em
[`src/loader/nam_json/validation.rs:25-40`](src/loader/nam_json/validation.rs#L25) cobrem
**parcialmente** o espaço de dimensões:

| Dimensão                                   | Limite                            | Coberto?                           |
|:------------------------------------------ |:---------------------------------:|:----------------------------------:|
| WaveNet `channels` (free)                  | `MAX_WAVENET_FREE_CHANNELS = 512` | ✅                                 |
| LSTM camadas / hidden                      | `16` / `1024`                     | ✅                                 |
| A2-Dyn channels / bottleneck               | `256` / `256`                     | ✅                                 |
| **WaveNet `kernel_size`**                  | —                                 | ❌                                 |
| **WaveNet nº de dilations / nº de arrays** | —                                 | ❌                                 |
| **WaveNet `head_size`**                    | —                                 | ❌                                 |
| **ConvNet `channels`**                     | —                                 | ❌                                 |
| **ConvNet `kernel_size` / dilations**      | —                                 | ❌                                 |
| **Linear `receptive_field`**               | —                                 | ❌ (limitado só por `MAX_WEIGHTS`) |

Em `topology/wavenet.rs`, `kernel_size` é apenas testado `> 0` (linhas 203-205) e `head_size`
apenas `!= 0` (linha 223); o nº de elementos em `dilations` e o nº de layer-arrays são ilimitados.
ConvNet (`topology/convnet.rs`) sequer limita `channels`.

Pior: em [`layout.rs:26`](src/loader/dispatcher/wavenet/layout.rs#L26) a alocação acontece **antes**
da validação do tamanho real dos pesos:

```rust
let padded_total = num_blocks * width * in_size * k_size;   // k_size não confiável, sem teto
let mut f32_weights = AlignedVec::new(padded_total, 0.0f32); // ALOCA primeiro
// ...
let raw = cursor.read_slice(total_4wide)?;                   // só DEPOIS valida a contagem
```

`WeightCursor::read_slice` ([`weight_cursor.rs:47`](src/loader/dispatcher/weight_cursor.rs#L47))
é checado, mas a alocação gigante já ocorreu.

### F2 — Causa-raiz

1. Cobertura incompleta dos limites de topologia (Task 5.2 focou em `channels`/`hidden`/`layers`).
2. Ordem "alocar → ler/validar" em vez de "validar → alocar".
3. `overflow-checks` **desligado em release** (não há `overflow-checks = true` em
   [`Cargo.toml [profile.release]`](Cargo.toml#L99) nem em `.cargo/config.toml`), logo a aritmética
   de tamanho faz *wrapping* silencioso.

### F2 — Impacto / alcance (verificado)

- **DoS por abort (certo):** um `.nam`/`.namb` minúsculo com `kernel_size` grande (ex.: `1e7`,
  `in=out=512`) produz `padded_total ≈ 2.6·10¹⁵` floats → `Layout::from_size_align(~10¹⁶, 64)` →
  `handle_alloc_error` ([`aligned.rs:101`](src/math/common/aligned.rs#L101)) → **abort do processo**
  (no plugin CLAP isso derruba a DAW).
- **DoS por panic (certo):** com `kernel_size` na faixa de overflow (ex.: `≈ 2⁴⁶`), `padded_total`
  e `total_4wide` fazem *wrap* para valores pequenos; a transposição
  [`transpose_conv1d_interleaved_4wide`](src/loader/dispatcher/wavenet/layout.rs#L154) itera sobre o
  `kernel` **real (enorme)** e indexa `raw[raw_idx]`/`weights[target_idx]` → *index out of bounds* →
  **panic** na thread de carregamento.
- **Corrupção de memória (não totalmente descartável):** a transposição usa indexação **checada**
  (`weights[idx] = raw[idx]`), o que converte o overflow em panic em vez de escrita OOB. Porém os
  `debug_assert!` de tamanho (linhas 161-170) somem em release, e há leitura posterior dos pesos no
  hot-path de inferência via `get_unchecked` assumindo tamanhos coerentes. Uma combinação de
  valores em que **todas** as multiplicações deem *wrap* de forma consistente (mantendo índices em
  faixa porém com semântica errada) não pode ser provada impossível sem aritmética checada e limites
  rígidos. Portanto trata-se de **superfície de ataque a fechar**, não apenas crash cosmético.

### F2 — Proposta de solução

1. **Adicionar limites de topologia** em `validation.rs` e aplicá-los na detecção:
   `MAX_KERNEL_SIZE` (ex.: 64), `MAX_DILATION` (ex.: 4096), `MAX_DILATIONS_PER_ARRAY` (ex.: 64),
   `MAX_WAVENET_ARRAYS` (ex.: 8), `MAX_HEAD_SIZE`, `MAX_CONVNET_CHANNELS`, `MAX_CONVNET_KERNEL_SIZE`,
   `MAX_RECEPTIVE_FIELD`. Rejeitar com erro diagnóstico claro (mesmo padrão da mensagem de
   `MAX_WAVENET_FREE_CHANNELS`).
2. **Aritmética checada** em todos os cálculos de tamanho do dispatcher
   (`checked_mul`/`checked_add`, `?` em erro) — `layout.rs`, `dynamic.rs`, `convnet/mod.rs`,
   `set_weights.rs`, `dynamic/build.rs`.
3. **Inverter a ordem**: validar a contagem de pesos disponível (`cursor`) *antes* de
   `AlignedVec::new(padded_total)`, ou alocar incrementalmente.
4. **Defesa em profundidade:** habilitar `overflow-checks = true` no perfil release **apenas para o
   caminho frio de loader** não é trivial (é por crate). Alternativa: encapsular toda a aritmética
   de tamanho em helpers `fn checked_layout_size(...) -> Result<usize>`.
5. Estender o fuzzing de `tests/proptest_parsers.rs` para varrer `kernel_size`/`dilations`/`channels`
   adversariais e garantir que o resultado seja sempre `Err` (nunca abort/panic).

---

## 🟢 F3 — Alocação de heap + syscalls + `drop` na thread de áudio (`set_max_buffer_size` no RT) ✅ [DONE]

- **Status:** ✅ (resolvido 2026-06-24)
- **Locais:**
  - [`src/clap/processor/events.rs:186`](src/clap/processor/events.rs#L186) (chamada no RT — REMOVIDA).
  - [`src/clap/plugin/main_thread/load.rs:141-181`](src/clap/plugin/main_thread/load.rs#L141) (defer quando `buffer_size == 0`).
  - [`src/clap/processor/mod.rs:161`](src/clap/processor/mod.rs#L161) (`activate()` chama `flush_pending_model()`).
  - [`src/clap/plugin/main_thread/housekeeping.rs:18`](src/clap/plugin/main_thread/housekeeping.rs#L18) (fallback flush).
- **Severidade:** 🟢 Resolvida.

### F3 — Solução aplicada

Abordagem (ver proposta original na linha 237):

1. **`load_model()`** (main thread): quando `buffer_size == 0` (state-restore antes de `activate()`),
   armazena model+l, resampler e calibração em `ColdShared::pending_model` no lugar de enviar
   imediatamente via SPSC.
2. **`activate()`** e **`housekeeping()`**: chamam `NamClapMainThread::flush_pending_model()`,
   que pre-dimensiona (`set_max_buffer_size`) e envia o modelo via SPSC após `buffer_size`
   já ser conhecido.
3. **RT path** (`events.rs:cold_load_model`): a chamada de `set_max_buffer_size` foi removida.
   O modelo sempre chega pré-dimensionado. Regressão detectável por `heap-audit`.

### F3 — Descrição

`cold_load_model` roda dentro de `process_events()` → `process()`, ou seja **na thread de áudio**.
Em `events.rs:186`:

```rust
let _ = model.set_max_buffer_size(self.max_frames_count);
```

Quando `max_buf > self.max_buffer_size`, `set_max_buffer_size` (A2 estático) executa, na thread RT:

```rust
self.layer_buffers.clear();          // dropa 23 MirroredBuffer → 23x munmap (syscall) no RT
for i in 0..A2_NUM_LAYERS {          // 23 camadas
    let mb = MirroredBuffer::<f32>::new(cap * CH)?;  // mmap/memfd_create (syscalls) no RT
    self.layer_buffers.push(mb);     // possível realloc do Vec (heap)
    // ...
}
self.layer_in   = AlignedVec::new(CH * max_buf, 0.0f32);            // heap alloc
self.head_accum = AlignedVec::new(head_ring_size * CH, 0.0f32);     // heap alloc
```

### F3 — Causa-raiz

O pré-dimensionamento no main thread ([`clap/plugin/main_thread/load.rs`](src/clap/plugin/main_thread/load.rs))
só ocorre quando `buffer_size > 0` no momento do load. Em restauração de estado/preset **antes** de
`activate()`, `buffer_size == 0`, o pré-dimensionamento é pulado, e o primeiro `process()` num
quantum maior dispara o caminho de fallback alocador. O comentário em `events.rs:179-185` reconhece
o trade-off (auditoria B2.1).

### F3 — Impacto

`mmap`/`munmap`/`memfd_create` e alocações de heap na thread SCHED_FIFO causam page-faults e bloqueio
→ **xrun/glitch audível** no primeiro bloco após o swap. Diverge do caminho standalone
([`rt_callback/commands.rs:60-88`](src/standalone/pw_host/rt_callback/commands.rs#L60)), que **não**
chama `set_max_buffer_size` no RT.

### F3 — Proposta de solução

1. Em `activate()` ([`clap/processor/mod.rs`](src/clap/processor/mod.rs#L39)): se `model_l`/`model_r`
   já estiverem presentes, chamar `set_max_buffer_size(max_frames_count)` **ali** (cold path legítimo,
   pré-`process`).
2. Tornar o caminho RT estritamente sem alocação: se `max_buf > capacidade` for detectado no
   `process()`, **sinalizar** o main thread (flag `RtStatusFlags` + SPSC) para reconstruir/redimensionar
   o modelo fora do RT e fazer swap via GC — em vez de alocar inline. Enquanto não chega, processar
   com a capacidade vigente (como o standalone).
3. Garantir teste de heap-audit (`CountingAllocator`) cobrindo o cenário "load via state antes do
   activate, depois primeiro process em quantum maior".

---

## 🟠 F4 — Aritmética de ponteiro fora dos limites em prefetch SIMD (UB sob strict provenance) [DONE]

- **Status:** ✅ Resolvido (2026-06-24). **Nota pós-auditoria de correção:** os locais originalmente
  listados foram todos corrigidos com `wrapping_add`. Os sites `layer_array.rs:131,134` e
  `layer_array_dyn.rs:121,124` — que usam `.add(i+1)` / `.add(i+2)` — são **falsos positivos**:
  ambos são protegidos por guards `if i+1 < num_layers` / `if i+2 < num_layers` antes do `.add()`,
  mantendo o ponteiro dentro dos limites do array em todos os casos de execução.
- **Locais corrigidos (via `wrapping_add`):**
  - [`src/math/gemm/gemv/kernel_macro.rs:57,157`](src/math/gemm/gemv/kernel_macro.rs#L57)
  - [`src/math/gemm/gemv/f16_avx512.rs:40`](src/math/gemm/gemv/f16_avx512.rs#L40), `f32_avx512.rs:91,255`
  - [`src/math/gemm/dot_4x/avx2.rs:55-59,154-156`](src/math/gemm/dot_4x/avx2.rs#L55), `avx2_dual.rs:190-194`, `avx512.rs:41-44`, `avx512_dual.rs:54-57`
  - [`src/math/common/ops.rs`](src/math/common/ops.rs) (`prefetch_strategy_simple` / `prefetch_strategy_2stage` usam `base_ptr.wrapping_add`)
- **Severidade original:** 🟠 Alta como *soundness* (UB latente), baixa probabilidade prática em x86 hoje.

### F4 — Descrição

Padrão recorrente:

```rust
while in_c + 8 <= in_len {
    _mm_prefetch::<_MM_HINT_T0>($in_frame.as_ptr().add(in_c + 64) as *const i8);
    // ...
}
```

O laço garante apenas `in_c + 8 <= in_len`, mas calcula `as_ptr().add(in_c + 64)` — um ponteiro até
**56 elementos além** do fim da fatia (no caso `in_c = in_len - 8`). O método `<*const T>::add`
exige que o resultado permaneça **dentro do objeto alocado** (no máximo "one-past-the-end");
ultrapassar isso é **UB**, *mesmo que o ponteiro só seja usado por um prefetch que nunca faz fault*.
Em `ops.rs:128`, `prefetch_strategy_simple` faz `base_ptr.add(16)` incondicionalmente.

### F4 — Causa-raiz

Uso de `.add()` (que carrega a precondição de in-bounds) para computar endereços de prefetch que são,
por natureza, especulativos/fora de faixa.

### F4 — Impacto

- Em x86-64, `prefetchtX` nunca falha por endereço inválido, e o LLVM hoje não explora esse UB para
  prefetch — por isso não há crash observado.
- Porém é **UB formal**: viola a postura de segurança estrita do projeto, falha em **Miri**, e é
  frágil a otimizações futuras do compilador. Para um projeto que proíbe `is_x86_feature_detected!`
  e `unwrap()` no hot-path por rigor, isso é uma inconsistência relevante.

### F4 — Proposta de solução

1. Trocar `.add(...)` por `.wrapping_add(...)` **apenas na computação de endereços de prefetch**
   (`wrapping_add` é sempre seguro de computar; só o deref seria UB, e prefetch não derefere via
   provenance). Mudança mínima e sem custo de runtime.

2. Centralizar em um helper único, ex.:

   ```rust
   #[inline(always)]
   unsafe fn prefetch_t0<T>(base: *const T, offset_elems: usize) {
       _mm_prefetch::<_MM_HINT_T0>(base.wrapping_add(offset_elems) as *const i8);
   }
   ```

   e substituir todas as ocorrências, eliminando a duplicação e o risco de regressão.

3. Adicionar uma execução de **Miri** (lane separada, sem `target_feature` real) sobre os testes de
   kernels escalares/genéricos para flagrar reincidências.

---

## 🟠 F5 — Degradação adaptativa do WaveNet não faz crossfade real; retorno de `crossfade_multiplier()` é descartado [DONE]

- **Status:** ✅
- **Locais:**
  - [`src/dsp/pipeline/stages/output.rs:188`](src/dsp/pipeline/stages/output.rs#L188) (valor descartado).
  - [`src/dsp/adaptive.rs:214-246`](src/dsp/adaptive.rs#L214) (`crossfade_multiplier`).
  - [`src/dsp/pipeline/stages/inference.rs:29`](src/dsp/pipeline/stages/inference.rs#L29) (`hold_layers`).
  - Contraste correto: [`src/models/container.rs:157-183`](src/models/container.rs#L157) (A2 faz blend de verdade).
- **Severidade:** 🟠 Alta como divergência doc↔código + clique audível; ativa só sob sobrecarga de CPU.

### F5 — Descrição

Em `output.rs:188`:

```rust
adaptive.crossfade_multiplier(sample_rate, n_pw); // retorno (0→1) IGNORADO
```

`crossfade_multiplier` é o **único** produtor da rampa linear de crossfade e seu retorno é
descartado (último statement da função, sem atribuição). O doc-comment afirma *"Returns the linear
multiplier applied to the degradation path. Call this during the output stage."* — mas nada é
aplicado ao áudio.

Para o caminho **WaveNet (layer-skipping)**, o único "amortecimento" é
`hold_layers = adaptive.is_crossfading()` (`inference.rs:29`): durante a janela de 32 ms o
`set_effective_layers()` é **congelado**; quando a janela expira, a contagem de camadas muda
**abruptamente** no bloco seguinte. Como o modelo roda **uma única vez**, não há o que misturar no
output sem uma segunda passada — ou seja, `crossfade_multiplier` é, na prática, **dead code** para
esse caminho.

O caminho **A2 SlimmableContainer** faz o blend corretamente em `container.rs:171`
(`crossfade_blend_mono(output, scratch, t)`) com seu **próprio** contador (`crossfade_elapsed`),
independente de `crossfade_multiplier`.

### F5 — Causa-raiz

Mecanismo de crossfade do WaveNet incompleto: a infra de `crossfade_multiplier` foi prevista para
escalar o áudio na saída, mas o ramp não é consumido; resta apenas o "hold" (atraso), que apenas
adia a transição abrupta em 32 ms sem suavizá-la.

### F5 — Impacto

A doc `docs/architecture.md` §5.1 promete *"32 ms linear crossfade between active layers"*. Sob
sobrecarga sustentada de CPU (transições Full↔Reduced↔Minimal), a troca instantânea da contagem de
camadas pode gerar **clique/pop audível**. É também dívida de clareza (retorno morto + doc enganosa).

### F5 — Proposta de solução

Decidir o comportamento pretendido e alinhar doc + código:

- **Opção A (fidelidade alta):** implementar crossfade real para o WaveNet — durante a janela,
  manter a configuração antiga em um buffer de saída e a nova em `scratch`, e fazer
  `crossfade_blend_mono(out, scratch, t)` com `t = crossfade_multiplier(...)`. Custa uma segunda
  passada de inferência durante ~32 ms (aceitável: ocorre em recuperação, não no pico de carga).
- **Opção B (pragmática):** assumir o "hold-only", **remover** o retorno e ajustar `adaptive.rs`
  (a função vira `advance_crossfade_clock()`), e corrigir `docs/architecture.md` §5.1 para descrever
  o mecanismo como *settling-hold* (sem blend) no WaveNet, reservando o termo "crossfade" ao A2.
- Em ambos os casos, criar teste que estresse transições do FSM e verifique ausência de
  descontinuidade > limiar (ex.: derivada de amplitude inter-bloco) no caminho WaveNet.
- **Nota do PO:** Escolho a opção A (fidelidade alta). Mas gostaria, se possível, comparar a
  performance das duas e assegurar que não há um penalidade agressiva na performance do WaveNet.

---

## 🟡 F6 — Precondições de comprimento não documentadas em kernels SIMD multi-vetor (OOB latente) [DONE]

- **Status:** ✅
- **Locais:**
  - [`src/math/gemm/dot_4x/avx2.rs:24-31,54-72,105-112`](src/math/gemm/dot_4x/avx2.rs#L24) (`dot_product_4x_avx2`: usa `len = state.len()`, lê `w0..w3`).
  - [`src/math/gemm/dot_4x/avx2_dual.rs:165`](src/math/gemm/dot_4x/avx2_dual.rs#L165) (`dot_product_batch_4x_avx2`: só checa `h0.len()`).
  - Kernels de batch GEMV/GEMM com `in_len = in_frames.len() / num_frames` sem `assert` de divisibilidade exata: [`src/math/gemm/gemv/f32_avx2.rs:28-29,145-146`](src/math/gemm/gemv/f32_avx2.rs#L28), `gemm_batch/*`.
- **Severidade:** 🟡 Média (UB/panic *se* a precondição for violada; hoje sustentada pelos chamadores).

### F6 — Descrição

`dot_product_4x_avx2` deriva `len` somente de `state.len()` e então carrega `w0,w1,w2,w3` via
`_mm_loadu_si128(w_k.as_ptr().add(i))` (16 bytes) em `[0, len)`, e o tail escalar indexa
`w_k[i]`. Não há `# Safety` documentando que `w_k.len() >= state.len()` (o topo do arquivo tem
`#![allow(clippy::missing_safety_doc)]`). Idem `dot_product_batch_4x_avx2`, que valida só `h0.len()`
mas acessa `h1,h2,h3` nos mesmos índices.

Nos kernels de batch, `in_len`/`out_len` vêm de divisão inteira (`/ num_frames`) **sem** assert de
divisibilidade exata; um chamador com tamanho não-múltiplo trunca silenciosamente e desalinha as
fronteiras de sub-frame, com `get_unchecked` subsequente.

### F6 — Causa-raiz

Contratos de segurança implícitos não documentados nem verificados (`unsafe fn` "pub(crate)").

### F6 — Impacto

- Se algum chamador passar `w_k`/`h_k` mais curto que `state`: **leitura OOB** (UB) no caminho SIMD
  e **panic** no tail escalar (este último na thread RT, durante inferência LSTM).
- Hoje os chamadores (gates LSTM de tamanho `hidden`) tendem a respeitar os comprimentos iguais, mas
  é frágil a refator/novos modelos.

### F6 — Proposta de solução

1. Documentar `# Safety` explícito em cada função (precondição de comprimento por argumento),
   alinhado ao que já existe em `traits.rs:126,142,153`.
2. Adicionar `debug_assert!(w0.len() >= len && w1.len() >= len && ...)` (custo zero em release) e,
   nos kernels de batch, `debug_assert_eq!(in_frames.len() % num_frames, 0)`.
3. Onde barato, computar `len = state.len().min(w0.len()).min(...)` defensivamente.

---

## 🔵 F7 — CRC32 do `.namb` cobre só os bytes de peso, não o JSON de metadados nem o header [DONE]

- **Status:** ✅
- **Locais:** [`src/loader/namb/parse.rs:67-85`](src/loader/namb/parse.rs#L67), [`src/loader/namb/header.rs`](src/loader/namb/header.rs) (`check_crc` calcula `crc32_ieee(&data[weights_offset..])`).
- **Severidade:** 🔵 Baixa (completude de detecção de corrupção; CRC32 não é fronteira de autenticação).

### F7 — Descrição

O CRC cobre apenas `data[weights_offset..]`. A seção JSON (`data[header_size..weights_offset]`),
que define `architecture` e **todas** as dimensões de camadas, e o próprio header, **não** são
verificados. Corrupção acidental das dimensões passa pelo CRC e flui para os alocadores
(ver F2). CRC32 não é criptográfico, então isto não é um furo de autenticação — é lacuna de
detecção de corrupção.

### F7 — Proposta de solução

Estender a cobertura do CRC para `header (exceto o campo crc) + JSON + pesos` (ex.: CRC sobre
`data[..offset_do_campo_crc]` concatenado com `data[apos_campo_crc..]`), mantendo retrocompat via
flag de versão; **ou** documentar explicitamente a limitação no formato. Combinar com F2 (limites
rígidos) garante que metadados corrompidos não causem alocações perigosas.

**Conclusão:** CRC estendido para todo o arquivo exceto o campo crc a partir de `version >= 2`. Retrocompatibilidade de v1 (cobrindo apenas pesos) mantida.

---

## 🔵 F8 — Sentinela de CRC do NAMB v1 permite pular verificação de integridade (pesos all-zero) [DONE]

- **Status:** ✅
- **Local:** [`src/loader/namb/parse.rs:75-85`](src/loader/namb/parse.rs#L75).
- **Severidade:** 🔵 Baixa (retrocompat intencional; só afeta modelos degenerados vazios).

### F8 — Descrição

Para `version < 2`, se `crc32 == 0` **e** a seção de pesos estiver vazia/all-zero, a verificação é
pulada com `log::warn!`. É um caminho de compatibilidade com arquivos legados; modelos reais têm
pesos não-triviais. Documentar e, idealmente, marcar como *deprecated* o NAMB v1 sem CRC.

**Conclusão:** Documentado deprecamento do sentinel de CRC nulo no NAMB v1 em `docs/namb-spec.md` e emitido aviso de runtime explícito de deprecamento no log de warning. Além disso, corrigido teste CLAP que falhava ao carregar estados/presets com tamanho de buffer zerado.

---

## 🔵 F9 — Guarda de alinhamento de load SIMD é `debug_assert` (some em release) nos kernels de convolução estéreo [DONE]

- **Status:** ✅
- **Local:** [`src/math/dsp/stereo/conv_kernels.rs:28-32,106-115,...`](src/math/dsp/stereo/conv_kernels.rs#L28).
- **Severidade:** 🔵 Baixa hoje (seguro para o resampler atual), 🟡 se mudar a configuração.

### F9 — Descrição

`coeffs` usa load **alinhado** (`_mm256_load_ps`/`_mm512_load_ps`) protegido apenas por
`debug_assert!((coeffs as usize).is_multiple_of(align))`. As entradas (`input_l/r`) já usam
load *unaligned* (correto p/ a janela do delay-line com `pos` arbitrário). Hoje é seguro porque
`TAPS_PER_PHASE = 32` ([`sinc_kernel.rs:33`](src/dsp/sinc_kernel.rs#L33)) → cada fase ocupa
`32*4 = 128 bytes`, múltiplo de 64, mantendo `phase_ptr` 64-alinhado para qualquer fase. Mas em
**release** um futuro `taps_per_phase` não-múltiplo de 16, ou um `coeffs` derivado de sub-slice,
causaria **SIGSEGV** (GP fault) sem aviso.

### F9 — Proposta de solução

Opção mais robusta e barata: trocar o load de `coeffs` para **unaligned** (`_mm256_loadu_ps`) — em
Haswell+ a penalidade sobre dados já alinhados é nula. Alternativamente, transformar a invariante em
`const { assert!(TAPS_PER_PHASE % 16 == 0) }` (static assert) e/ou `assert!` rígido no construtor do
banco de coeficientes.

**Conclusão:** Alterado o carregamento de coeficientes (`coeffs`, `coeffs0`, `coeffs1`) para carga não alinhada (`$loadu`) em todos os macros de convolução SIMD (`impl_convolve_stereo`, `impl_convolve_stereo_dual`, `impl_convolve_mono` e `impl_convolve_mono_dual`), eliminando riscos de SIGSEGV/GP fault por desalinhamento em release.

---

## 🔵 F10 — Defaults `unimplemented!()` em métodos dual-frame de `SimdMath` (armadilha de panic RT latente) [DONE]

- **Status:** ✅
- **Local:** [`src/math/common/traits.rs`](src/math/common/traits.rs) (`dot_product_8x_f32_dual`, `dot_product_16x_f32_dual`).
- **Severidade:** 🔵 Baixa (hoje inalcançável; todas as 3 impls sobrescrevem).

### F10 — Descrição

Esses métodos do trait têm corpo padrão `unimplemented!()`. **Verificado:** as três impls
despacháveis (`Avx2Math`, `Avx512Math`, `Avx512VnniBf16Math`) sobrescrevem ambos
(`avx2_impl.rs`, `avx512/gemv/base.rs`, `avx512/gemv/vnni_bf16.rs`), logo o default nunca é
alcançado em runtime via `dispatch_simd!`. Porém, uma futura impl que esqueça de sobrescrever faria
**panic na thread RT** (chamados em `conv1d_dual.rs`/`conv1d_dyn_dual.rs`, hot-path), sem garantia em
tempo de compilação.

### F10 — Proposta de solução

Remover os corpos default e tornar os métodos **obrigatórios** (sem default) — o compilador passa a
exigir a implementação em qualquer nova `impl SimdMath`, eliminando a armadilha. Se um default for
mesmo necessário, que ele delegue para a versão escalar de referência em vez de `unimplemented!()`.

---

## 🔵 F11 — Caminho `.nam` (JSON) não valida `sample_rate` (presente no `.namb`) [DONE]

- **Status:** ✅
- **Locais:** validação presente em [`src/loader/namb/parse.rs:125-140`](src/loader/namb/parse.rs#L125); ausente no caminho JSON ([`src/loader/nam_json/`](src/loader/nam_json/)); consumo em `build.rs:148` (`as u32`).
- **Severidade:** 🔵 Baixa (robustez lógica; sem unsafe).

### F11 — Descrição

O `.namb` valida `sample_rate` finito e `> 0.0`. O caminho `.nam` (JSON) não — um
`"sample_rate": -1.0` ou `NaN` flui adiante e vira `0`/lixo no `as u32`, afetando o cálculo de
razão do resampler. Adicionar a mesma validação (finito e `> 0`) na desserialização/validação do
JSON, com erro diagnóstico.

---

## 📦 Épicos (agrupamento para resolução otimizada)

> Agrupados por afinidade técnica/risco para facilitar implementação atômica e segura.
> A criação do `TODO-sprints.md` (épicos → sprints → tarefas) deve ser feita **sob demanda** pela
> skill `planejador-arquiteto`.

### Épico E1 — Blindagem do Loader contra entrada não confiável (Segurança) [DONE]

**Findings:** F2 (🟠), F1 (🔴, ramo A2), F7 (🔵), F8 (🔵), F11 (🔵).
**Objetivo:** fechar a superfície de ataque de arquivos `.nam`/`.namb` maliciosos/corrompidos.
**Itens-chave:** limites de topologia faltantes (kernel/dilations/arrays/head/ConvNet/Linear);
aritmética de tamanho checada; ordem validar→alocar; `From<Vec>` seguro; cobertura de CRC;
validação de `sample_rate` no JSON; ampliar `proptest_parsers.rs`. **É o épico mais crítico.**

### Épico E2 — Conformidade estrita de RT-Safety (Áudio) [DONE]

**Findings:** F3 (🟠), F10 (🔵).
**Objetivo:** garantir zero-alloc/zero-syscall/zero-panic na thread de áudio sob todos os cenários.
**Itens-chave:** mover `set_max_buffer_size` para fora do RT (pré-`activate` + sinalização ao main);
alinhar comportamento CLAP↔standalone; tornar métodos `SimdMath` obrigatórios; heap-audit do cenário
"state-load antes do activate".

### Épico E3 — Soundness de kernels SIMD/unsafe (UB & contratos) [DONE]

**Findings:** F4 (🟠), F6 (🟡), F9 (🔵).
**Objetivo:** eliminar UB latente e formalizar contratos `unsafe`.
**Itens-chave:** prefetch via `wrapping_add` + helper único; `# Safety` + `debug_assert` de
comprimento nos kernels multi-vetor/batch; load de `coeffs` unaligned ou static-assert; lane de
**Miri** sobre kernels genéricos.

### Épico E4 — Fidelidade da Degradação Adaptativa (Qualidade de áudio) [DONE]

**Findings:** F5 (🟠).
**Objetivo:** eliminar cliques nas transições do `AdaptiveCompute` no WaveNet e alinhar doc↔código.
**Itens-chave:** decidir entre crossfade real (Opção A) ou hold-only documentado (Opção B); remover
retorno morto; teste de descontinuidade inter-bloco.

### Épico E5 — Proteções residuais pós-F2 (DoS de segunda ordem) ⬜

**Findings:** F12 (🔵), F13 (🟡).
**Objetivo:** fechar os dois vetores de segurança descobertos durante a **auditoria de correção**
das implementações do Épico E1 (F2) e do Épico E4 (F5).
**Itens-chave:**

- **F12:** Adicionar `MAX_TOTAL_STATE_FRAMES` em `validation.rs` e verificar o orçamento agregado
  de estado (`(k-1)×dilation` acumulado × `channels`) durante a detecção de topologia WaveNet,
  antes de qualquer alocação. Estender `proptest_parsers.rs` para garantir que nenhum input válido
  exceda o orçamento.
- **F13:** Redimensionar `backup_starts` de `[0usize; 64]` para `[0usize; MAX_WAVENET_ARRAYS × MAX_DILATIONS_PER_ARRAY]`
  (= 512 entradas, 4 KB de stack) em `inference.rs:223,389`. Adicionar `debug_assert_eq!` de
  completude após o backup. Considerar failsafe que degrade para hold-only sem corrupção de estado
  se capacidade insuficiente.
**Prioridade:** F13 > F12 (F13 causa saída de áudio incorreta; F12 é DoS de menor magnitude).

---

## 🔵 F12 — Amplificação residual de memória via state buffers (DoS de baixa magnitude)

- **Status:** ⬜
- **Locais:**
  - [`src/models/wavenet/common.rs:61`](src/models/wavenet/common.rs#L61) (`WaveNetLayerState::new` aloca `MirroredBuffer` proporcional ao campo receptivo).
  - [`src/loader/dispatcher/wavenet/dynamic.rs:60`](src/loader/dispatcher/wavenet/dynamic.rs#L60) (`rf = (k-1)*dilation` sem teto para o produto combinado).
- **Severidade:** 🔵 Baixa (melhoria F2; amplificação muito menor que o caso irrestrito anterior).

### Descrição

A proteção F2 adicionou limites individuais: `MAX_KERNEL_SIZE=64`, `MAX_DILATION=4096`,
`MAX_DILATIONS_PER_ARRAY=64`, `MAX_WAVENET_ARRAYS=8`, `MAX_WAVENET_FREE_CHANNELS=512`. Porém
**não existe um limite para o produto total** de memória de estado resultante. Cada camada aloca
um `MirroredBuffer` de tamanho `(k-1)*dilation + max_buf + padding` × `channels` × 2
(por ser espelho virtual).

Pior caso com os novos limites: `k=64, d=4096, ch=1, 8 arrays × 64 dilações = 512 camadas`.
Campo receptivo por camada: `(64-1)*4096 = 258048 frames` → buffer `≈ (258048+25+1) × 1 × 2 × 4 bytes ≈ 2 MB` por
camada → **512 camadas × 2 MB ≈ 1 GB de estado** a partir de um arquivo de modelo de tamanho
mínimo (poucos KB de pesos com ch=1). A ratio de amplificação é ≈ 28.000×.

Este é um **DoS por amplificação de memória**, não por peso, não coberto pela defesa atual de
`MAX_FLOAT_COUNT / MAX_WEIGHTS`. Comparado ao caso pré-F2 (amplificação ilimitada), é uma
melhoria significativa, mas ainda permite um arquivo de ~36 KB crashar um sistema com 1 GB
disponível.

### Proposta de solução

1. Calcular o **orçamento total de estado** durante a validação de topologia. Adicionar
   `checked_mul(receptive_field, channels)` acumulado sobre todas as camadas e rejeitar se
   exceder `MAX_TOTAL_STATE_FRAMES` (ex.: `1 << 26` = 64 M frames = 256 MB para `f32`).
2. Expor o orçamento como constante documentada em `validation.rs`:

   ```rust
   /// Aggregate cap for all WaveNet layer state frames (pre-allocated mirrored buffers).
   /// Prevents DoS via receptive-field amplification. Default: 64 Mi frames ≈ 256 MB @ f32.
   pub const MAX_TOTAL_STATE_FRAMES: usize = 1 << 26;
   ```

3. Aplicar a verificação em `topology/wavenet.rs` após construir todas as camadas:

   ```rust
   let total_state: usize = dilations.iter().flatten()
       .zip(kernel_sizes.iter().cycle())
       .try_fold(0usize, |acc, (&d, &k)| {
           acc.checked_add(k.saturating_sub(1).saturating_mul(d))
       })
       .ok_or("State overflow")?;
   if total_state > MAX_TOTAL_STATE_FRAMES / max(channels) {
       return WavenetTopologyResult::Rejected("Aggregate state budget exceeded");
   }
   ```

4. Estender o fuzzing de `tests/proptest_parsers.rs` com asserção de que nenhum modelo válido
   aloca mais que o orçamento.

---

## 🟡 F13 — Buffer de backup de estado para crossfade WaveNet dimensionado para 64 entradas, mas max é 512

- **Status:** ⬜
- **Locais:**
  - [`src/dsp/pipeline/stages/inference.rs:223,389`](src/dsp/pipeline/stages/inference.rs#L223) (`backup_starts = [0usize; 64]`).
  - [`src/models/wavenet/model_dyn.rs:116`](src/models/wavenet/model_dyn.rs#L116) (`if *offset < starts.len()` — truncamento silencioso).
  - [`src/models/wavenet/model.rs`](src/models/wavenet/model.rs) — mesma proteção de guarda.
- **Severidade:** 🟡 Média (produz saída de áudio incorreta, não crash; só ativa em WaveNet dinâmico
  com >64 camadas + degradação adaptativa ativa).

### Descrição

O crossfade double-pass do WaveNet (fix F5) usa um array de stack `[0usize; 64]` para salvar e
restaurar os `buffer_start` de todas as camadas entre as duas passagens. `backup_buffer_starts`
itera sobre `array.states` e para de salvar quando `offset >= starts.len()` — truncamento
**silencioso** sem erro, sem log.

Com os novos limites F2 (`MAX_WAVENET_ARRAYS=8`, `MAX_DILATIONS_PER_ARRAY=64`), um modelo
dinâmico pode ter até `8 × 64 = 512 camadas`. Para qualquer modelo com mais de 64 camadas, as
camadas `65..N` não têm seus `buffer_start` salvos. Quando o segundo passe (NEW layer count)
executa, essas camadas encontram `buffer_start` **avançado** pela primeira passagem (OLD), lendo
posições incorretas do delay-line. O blend resultante mistura amostras temporalmente
dessincronizadas, produzindo artefatos audíveis (pré-eco / ghost signal) durante a transição
adaptativa, e também **avança permanentemente** os ponteiros de estado das camadas afetadas,
introduzindo deriva de estado para os blocos subsequentes.

### Impacto detalhado

| Condição | Efeito |
|:---------|:-------|
| Modelo com ≤ 64 camadas (todos os SKUs de catálogo: max 23 cam.) | Sem efeito — catálogo sempre seguro |
| Modelo WaveNetDyn com > 64 cam. + AdaptiveCompute **desligado** | Sem efeito (crossfade nunca ativa) |
| Modelo WaveNetDyn com > 64 cam. + AdaptiveCompute **ligado** | Estado corrompido durante crossfade; drift permanente |

O cenário é exótico mas explicitamente habilitado pelos novos limites do F2.

### Proposta de solução

1. **Solução mínima (recomendada):** dimensionar `backup_starts` pelo máximo teórico de camadas,
   computado a partir dos limites de topologia:

   ```rust
   // MAX_WAVENET_ARRAYS * MAX_DILATIONS_PER_ARRAY = 8 * 64 = 512
   const MAX_WAVENET_BACKUP_STARTS: usize =
       crate::loader::nam_json::validation::MAX_WAVENET_ARRAYS *
       crate::loader::nam_json::validation::MAX_DILATIONS_PER_ARRAY;
   let mut backup_starts = [0usize; MAX_WAVENET_BACKUP_STARTS];
   ```

   Custo: 512 × 8 bytes = 4 KB de stack (aceitável; o double-pass já é uma operação pesada).
2. **Alternativa (solução robusta):** adicionar um `debug_assert_eq!(final_offset, total_layers)`
   após o backup para detectar divergências em testes.
3. **Alternativa (preventiva):** tornar o backup/restore uma operação infallível que retorna `Result`
   e aborta o crossfade se a capacidade for insuficiente (degradando para o comportamento hold-only
   anterior sem artefatos de estado corrompido).

---

## Apêndice — Achados investigados e **descartados** (para rastreabilidade)

- `namb/parse.rs:90` — a `expected_len` da mensagem de erro é, de fato, correta
  (`data.len() + padding`); **não** é bug.
- `resampler_swap.rs:163`, `conv1d_dispatch.rs:86` — `panic!` reportados estão sob `#[cfg(test)]`;
  **não** são panics de hot-path.
- `DspBridge` (gen/Acquire-Release), `GcOverflowBuffer` (u64 empacotado), `gui_param_generation`
  (Release/Acquire) — protocolos lock-free **revisados e corretos**.
- Ring buffers `MirroredBuffer`/A2 head-ring, FDL do UPOLS (cabsim), FSM do gate — lógica de
  wrap/índices **verificada e correta**.
- `DspBridge.dropped_frames` usa `Ordering::Relaxed` para `current_gen` — isso é **correto** para
  um contador de telemetria sem dependência de dados; não há risco de correção.
- `HugePageVec::Drop` não chama `drop_in_place` nos elementos T — o único uso real é `T = f32`
  (Copy, sem Drop), portanto inofensivo na prática atual.
- `drain_slimmable_models` "double-pop" no R channel — o pop condicional é protegido por
  `if active_model_r.is_some()` e falha graciosamente se o canal estiver vazio; **não é bug**.
- Sites de prefetch em `layer_array.rs:131,134` e `layer_array_dyn.rs:121,124` usam `.add(i+1)` e
  `.add(i+2)` — esses sites são **protegidos por guards** `if i+1 < num_layers` /
  `if i+2 < num_layers` que garantem o ponteiro in-bounds quando executado; **não são UB**.
  (Os demais sites de prefetch nos kernels math/ já foram corrigidos para `wrapping_add` no fix F4.)
