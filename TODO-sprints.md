<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-Sprints — nam-rs

> **Filosofia**: Sprint atômico, seguro e verificável. Cada tarefa deve poder ser executada
> de forma independente, com critérios de aceite objetivos e verificáveis.
>
> **Referência de correção**: NAMCore v0.5.3 é a bíblia. Paridade auditável via ESR/SNR.
>
> **ISA baseline**: x86-64-v3 (AVX2+FMA). Multiversioning para x86-64-v4 é ortogonal e futuro.
>
> **RT-Safety**: zero heap drop, zero blocking, zero `unwrap()` na thread de áudio.

---

## Épico E1 — MT7: Otimizações Imediatas (Sem Dependências)

**Origem**: MT7 em `TODO-features.md:L433`
**Pré-requisitos**: nenhum — pode ser executado a qualquer momento.
**Risco global**: 🟢 Baixo — todas as mudanças têm oracle escalar de paridade.
**Valor entregue**: melhoria de desempenho mensurável no hot-path A2 sem risco funcional.

---

### Sprint S1 — A2 Head Conv SIMD (Achado 2)

**Objetivo**: Substituir o loop escalar triplo `frames × 16_taps × CH` da `A2HeadConv::process()`
por kernels SIMD AVX2+FMA especializados por CH, eliminando ~128 FMAs escalares/frame para CH=8
e ~48 para CH=3.

**Arquivo central**: `src/models/a2/head.rs`

**Situação atual** (L96–L118 em `head.rs`):

```rust
for t in 0..k {
    for c in 0..ch {
        y += *head_w.get_unchecked(w_off + c) * *head_history.get_unchecked(src_off + c);
    }
}
```

100% escalar. Nenhuma instrução SIMD. Cada frame emite CH×16 FMAs sequenciais.

**Análise técnica — CH=8**:

- Pesos `head_w`: layout `[K][CH]` = `[16][8]` = 128 f32 contíguos por tap group.
- Para cada frame: `K=16` taps, cada tap lê 8 weights + 8 history → `_mm256_loadu_ps` × 2
  e `_mm256_fmadd_ps` com acumulador. Total: 16 loads de peso + 16 loads de history + 16 FMAs AVX.
- Como o ring buffer tem layout `[col][CH]`, `src_off = col * CH` → cada col é CH f32 contíguos
  → `_mm256_loadu_ps(head_history.as_ptr().add(src_off))` é válido para CH=8.
- Redução: `_mm256_hadd_ps` × 3 + `_mm_add_ss` → 1 escalar `y`.
- **Atenção**: o ring buffer usa `col & ring_mask` (wrapping). Cada tap acessa um `col` distinto.
  Não há contingüidade entre taps consecutivos → cada tap precisa de load independente.
- T=4 frame-tiling: para 4 frames consecutivos, os K=16 taps são os mesmos pesos →
  load de peso 1×, broadcast history[c] para 4 frames → 4 acumuladores `__m256`.
  Após K×CH iterações, redução horizontal de cada acumulador → 4 escalares.

**Análise técnica — CH=3**:

- Layout `head_w`: `[16][3]` = 48 f32 densos (sem padding). Layout `head_history`: stride 3 (denso).
- **Decisão aprovada (Opção A)**: usar `_mm_setr_ps(*h, *(h+1), *(h+2), 0.0)` para empacotar
  os 3 canais de história em `__m128` com 4º lane explicitamente zerado. Seguro e correto.
  Evita `_mm_loadu_ps` sobre stride 3 (que leria lixo de memória no 4º lane).
- Acumulador `__m128` com 3 lanes válidas + 1 lane permanentemente zero (garantido pelo setr_ps).
- Redução final: `_mm_hadd_ps` → extrai soma dos 3 lanes válidos.
- Não há T-tiling viável para CH=3 head (redução horizontal cara por frame); processar 1 frame
  por vez é correto — 16 `_mm_fmadd_ps` por frame vs 48 FMAs escalares (~2.5× ganho líquido).

---

#### T1.1 — Implementar `head_process_ch8_avx2` (kernel CH=8) ✅ [DONE]

**Arquivo**: `src/models/a2/head.rs`

**Status**: Concluído. Kernel AVX2+FMA implementado com T=4 frame-tiling, reutilizando `hsum_avx2` de `src/math/common/utility.rs`. Tail frames (< T) delegam para `a2_head_single_frame_scalar_ref`. Testes adicionados em `head_test.rs`: paridade, large block (128 frames) e wraparound.

**Arquivo**: `src/models/a2/head.rs`

**Assinatura proposta**:

```rust
/// Kernel SIMD AVX2+FMA para A2 Head Conv com CH=8.
///
/// Processa `num_frames` frames usando T=4 frame-tiling:
/// carrega os K=16 pesos uma vez por (tap, canal) e acumula
/// via broadcast FMA em 4 acumuladores `__m256` simultâneos.
///
/// ## Redução horizontal
/// Ao final de cada frame, `_mm256_hadd_ps` × 3 + extração do lane 0
/// retorna o escalar y, ao qual adicionamos `head_b` e multiplicamos
/// por `head_scale`.
///
/// # Safety
/// - Requer AVX2+FMA (`target_feature`).
/// - `head_w` deve ter pelo menos `K * 8` elementos.
/// - `head_history` deve ter `(ring_mask + 1) * 8` elementos.
/// - `output` deve ter pelo menos `num_frames` elementos.
#[target_feature(enable = "avx2,fma")]
unsafe fn head_process_ch8_avx2(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    output: &mut [f32],
)
```

**Lógica do kernel** (pseudocódigo):

```rust
bias_v = _mm256_set1_ps(head_b)
T = 4; n_tiled = (num_frames / T) * T

// Tile: 4 frames simultâneos
for f in (0..n_tiled).step_by(T):
    a0..a3 = zero_v  // 4 acumuladores
    col_base_f = base + f  // base = write_pos - num_frames
    for t in 0..K:
        w_ptr = head_w.as_ptr() + t * 8
        w_v = _mm256_loadu_ps(w_ptr)       // 8 pesos do tap t
        for fi in 0..4:
            col = (col_base_f + fi - (K-1-t)) & ring_mask
            h_v = head_history[col * 8]
            // Cada canal do histórico como escalar → broadcast + FMA
            // MAS: head_history[col * 8] são 8 f32 contíguos (CH=8)
            // → precisamos do DOT PRODUCT do vetor w[t] com h[col]
            // Estratégia: _mm256_dp_ps? Não — usar hadd após FMA.
            // Alternativa: 1 FMA de vetor, depois redução externa.
            h_vec = _mm256_loadu_ps(head_history.as_ptr() + col * 8)
            a[fi] = _mm256_fmadd_ps(w_v, h_vec, a[fi])

    // Redução horizontal de cada acumulador
    for fi in 0..4:
        y = hsum256(a[fi]) + head_b
        output[f + fi] = y * head_scale

// Tail escalar (frames restantes)
for f in n_tiled..num_frames: [loop escalar existente]
```

**Helper de redução** (`hsum256`):

```rust
#[inline]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let sum32 = _mm_add_ss(sum64, _mm_movehl_ps(shuf, sum64));
    _mm_cvtss_f32(sum32)
}
```

**Atenção crítica**: ao contrário do `conv1d_ch8.rs` (que faz broadcast de 1 escalar de história
para 8 canais de saída), aqui temos o problema dual: 8 history values × 8 weight values → dot product
para 1 escalar de saída. Por isso a redução horizontal é obrigatória.

**Complexidade**: O T=4 tiling é válido porque o eixo temporal é amortizado (pesos são os mesmos
por tap para todos os frames; apenas o col muda). Mas a redução custa ~4 instruções SSE por frame.
Para `num_frames` típico de 64, o ganho é claro: 64×16 = 1024 FMAs escalares → 64×(16 vetoriais +
4 reduções) = 64×20 instruções SIMD vs 64×128 ops escalares = 6.4× menos instruções.

---

#### T1.2 — Implementar `head_process_ch3_sse` (kernel CH=3) [DONE]

**Arquivo**: `src/models/a2/head.rs`

**Estratégia CH=3**:

```rust
#[target_feature(enable = "fma")]
unsafe fn head_process_ch3_sse(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    output: &mut [f32],
)
```

**Lógica (decisão aprovada — Opção A)**:

- `head_w` tem layout `[K=16][CH=3]` = 48 f32 densos. **Não introduzir padding** em `head.rs::new()`.
- Para os pesos: `_mm_setr_ps(w[t*3], w[t*3+1], w[t*3+2], 0.0)` — 3 scalares + empacotamento.
- Para o histórico: `_mm_setr_ps(h[col*3], h[col*3+1], h[col*3+2], 0.0)` — stride 3 denso, 4º lane zero.
- Acumulador `__m128`: `acc = _mm_fmadd_ps(w_v, h_v, acc)` por tap. Ao final, 3 lanes válidas + 1 zero.
- Redução: `_mm_hadd_ps(_mm_hadd_ps(acc, acc), acc)` → lane 0 = soma dos 3 válidos → `_mm_cvtss_f32`.

```rust
for each frame:
    acc = _mm_setzero_ps()
    for t in 0..K:
        col = (col_base - (K-1-t)) & ring_mask
        w_v = _mm_setr_ps(head_w[t*3], head_w[t*3+1], head_w[t*3+2], 0.0)
        h_v = _mm_setr_ps(head_history[col*3], head_history[col*3+1], head_history[col*3+2], 0.0)
        acc = _mm_fmadd_ps(w_v, h_v, acc)
    let hadd1 = _mm_hadd_ps(acc, acc)
    let hadd2 = _mm_hadd_ps(hadd1, hadd1)
    y = _mm_cvtss_f32(hadd2) + head_b
    output[f] = y * head_scale
```

**Ganho esperado**: ~2.5× vs escalar puro (16 FMAs SSE/frame vs 48 FMAs escalares).
Limitado pelo custo dos 16 `_mm_setr_ps` por frame; aceitável para esta geometria.

**Critério de aceite T1.2**:

- `output[f]` idêntico à versão escalar `a2_head_block_scalar_ref` para CH=3.
- Tolerância numérica: erro absoluto < 1e-5 (ordem do f32 roundoff).

---

#### T1.3 — Integrar dispatch no `A2HeadConv::process()`

**Arquivo**: `src/models/a2/head.rs`

Modificar `process()` para:

```rust
pub fn process(&self, ...) {
    match self.num_channels {
        8 => {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { head_process_ch8_avx2(...) }
            } else {
                self.process_scalar(...)
            }
        }
        3 => {
            if is_x86_feature_detected!("fma") {
                unsafe { head_process_ch3_sse(...) }
            } else {
                self.process_scalar(...)
            }
        }
        _ => self.process_scalar(...),
    }
}
```

**Nota**: como o target é x86-64-v3 (AVX2+FMA garantido), o `is_x86_feature_detected!` pode ser
substituído por check estático via `#[cfg(target_feature)]` ou mantido dinâmico para corretude.
Na prática o dispatching dinâmico tem custo negligível (1 atomic read por block).

**Alternativa preferida** (mais simples, consistente com o restante do codebase):
usar `#[target_feature(enable = "avx2,fma")]` e chamar diretamente (o binário já é compilado
com `-C target-cpu=x86-64-v3`). Ver Cargo.toml para confirmar.

---

#### T1.4 — Benchmarks antes/depois da A2 Head Conv

**Arquivo**: `benches/inference_bench.rs`

Adicionar grupos de benchmark:

```rust
/// Bench: A2 Head Conv CH=8 — escalar vs AVX2 (16 frames)
fn bench_a2_head_ch8(c: &mut Criterion) {
    // setup: AlignedVec weights, ring history
    // group.bench_function("a2_head_ch8_scalar", ...)
    // group.bench_function("a2_head_ch8_avx2", ...)
}

/// Bench: A2 Head Conv CH=3 — escalar vs SSE (16 frames)
fn bench_a2_head_ch3(c: &mut Criterion) { ... }
```

**Critério de aceite T1.4**:

- AVX2 deve ser ≥ 2× mais rápido que escalar para CH=8 (estimado: ~4–6×).
- Throughput CH=3 SSE deve ser ≥ 1.5× mais rápido que escalar.
- Comparar com baseline capturado ANTES da mudança (salvar em `target/logs/bench_before_mt7_s1.txt`).

---

#### T1.5 — Testes de paridade na `head_test.rs`

**Arquivo**: `src/models/a2/head_test.rs`

Os testes existentes (`test_a2_head_conv_ch3_parity`, `test_a2_head_conv_ch8_parity`, etc.) já
verificam paridade com `a2_head_block_scalar_ref`. Como a `process()` passará a chamar os kernels
SIMD, esses testes automaticamente validarão a implementação nova.

Adicionar testes específicos para o caminho SIMD:

```rust
#[test]
fn test_a2_head_ch8_avx2_parity_large_block() {
    // 128 frames, ring_size=256, write_pos no meio e com wraparound
    // Verificar que output == scalar_ref com diff < 1e-5
}

#[test]
fn test_a2_head_ch3_sse_parity_wraparound() {
    // write_pos = 3, num_frames = 16, K=16 → teste de wraparound severo
}
```

---

### Sprint S2 — Ativações Inline no `activations.rs` (Achado 3)

**Objetivo**: Vetorizar as 4 ativações inline em `src/models/a2/activations.rs` que hoje
usam loops escalares: `HardTanh`, `FastTanh`, `HardSwish`, `LeakyHardTanh`.

**Situação atual** (`activations.rs:L91–L156`):

- `HardTanh`: `for x in data { *x = x.clamp(-1.0, 1.0) }` — 1 instrução/f32 vs
  `vmaxps`/`vminps` (8/16 f32 por instrução).
- `FastTanh`: `for x { *x = fast_tanh(*x) }` — rational polynomial escalar; o corpo
  `fast_tanh` tem `abs`, `mul`, `div`, `add` — totalmente SIMD-friendly.
- `HardSwish`: `let t = *x + 3.0; let c = t.clamp(0,6); *x *= c/6` — 4 scalares ops vs
  3 SIMD ops (`add`, `min`, `max`, `mul`, `mul`).
- `LeakyHardTanh`: 2 branches por elemento → branchless SIMD com `vcmpps`/`vblendvps`.

**Nota arquitetural importante**: `tanh_slice`, `relu_slice`, `prelu_slice`, `sigmoid_slice`,
`silu_slice`, `softsign_slice` já delegam para `src/math/activations/` que tem implementações
AVX2+AVX-512 completas (confirmado na auditoria). Portanto o Achado 3 concerne apenas as 4
ativações inline acima.

---

#### T2.1 — Vetorizar `HardTanh` com AVX2

**Arquivo**: `src/models/a2/activations.rs`

**Implementação**:

```rust
// Criar função auxiliar no mesmo arquivo ou em math/activations/
#[target_feature(enable = "avx2")]
unsafe fn hard_tanh_slice_avx2(data: &mut [f32]) {
    let neg_one = _mm256_set1_ps(-1.0_f32);
    let pos_one = _mm256_set1_ps(1.0_f32);
    let mut i = 0;
    let len = data.len();
    while i + 16 <= len {
        let x1 = _mm256_loadu_ps(data.as_ptr().add(i));
        let x2 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
        _mm256_storeu_ps(data.as_mut_ptr().add(i),
            _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x1)));
        _mm256_storeu_ps(data.as_mut_ptr().add(i + 8),
            _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x2)));
        i += 16;
    }
    while i + 8 <= len {
        let x = _mm256_loadu_ps(data.as_ptr().add(i));
        _mm256_storeu_ps(data.as_mut_ptr().add(i),
            _mm256_min_ps(pos_one, _mm256_max_ps(neg_one, x)));
        i += 8;
    }
    for x in data.iter_mut().skip(i) { *x = x.clamp(-1.0, 1.0); }
}
```

Alterar `HardTanh` em `apply()`:

```rust
Self::HardTanh => {
    if is_x86_feature_detected!("avx2") {
        unsafe { hard_tanh_slice_avx2(data) }
    } else {
        for x in data.iter_mut() { *x = x.clamp(-1.0, 1.0); }
    }
}
```

**Ganho esperado**: 8–16× throughput vs escalar (vmaxps + vminps = 2 instruções/8 floats).

---

#### T2.2 — Vetorizar `HardSwish` com AVX2

**Arquivo**: `src/models/a2/activations.rs`

Fórmula: `x * clamp(x+3, 0, 6) / 6`

```rust
#[target_feature(enable = "avx2")]
unsafe fn hard_swish_slice_avx2(data: &mut [f32]) {
    let three = _mm256_set1_ps(3.0_f32);
    let six   = _mm256_set1_ps(6.0_f32);
    let inv6  = _mm256_set1_ps(1.0_f32 / 6.0_f32);
    let zero  = _mm256_setzero_ps();
    let mut i = 0;
    let len = data.len();
    while i + 8 <= len {
        let x = _mm256_loadu_ps(data.as_ptr().add(i));
        let t = _mm256_add_ps(x, three);                         // x + 3
        let c = _mm256_min_ps(six, _mm256_max_ps(zero, t));      // clamp(t, 0, 6)
        let y = _mm256_mul_ps(_mm256_mul_ps(x, c), inv6);        // x * c / 6
        _mm256_storeu_ps(data.as_mut_ptr().add(i), y);
        i += 8;
    }
    for x in data.iter_mut().skip(i) {
        let t = *x + 3.0;
        *x *= t.clamp(0.0, 6.0) * (1.0 / 6.0);
    }
}
```

**Ganho esperado**: ~5× throughput vs escalar (5 instruções/8 floats vs 4 ops escalares/float).

---

#### T2.3 — Vetorizar `LeakyHardTanh` com AVX2 branchless

**Arquivo**: `src/models/a2/activations.rs`

Fórmula branchless:

```rust
y = x                                        (linear region)
y = (x - min_val) * min_slope + min_val     (se x < min_val)
y = (x - max_val) * max_slope + max_val     (se x > max_val)
```

Implementação SIMD branchless:

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn leaky_hard_tanh_slice_avx2(
    data: &mut [f32],
    min_val: f32, max_val: f32,
    min_slope: f32, max_slope: f32,
) {
    let min_v     = _mm256_set1_ps(min_val);
    let max_v     = _mm256_set1_ps(max_val);
    let min_sl_v  = _mm256_set1_ps(min_slope);
    let max_sl_v  = _mm256_set1_ps(max_slope);
    let mut i = 0;
    let len = data.len();
    while i + 8 <= len {
        let x = _mm256_loadu_ps(data.as_ptr().add(i));
        // Região abaixo de min_val
        let lt_min  = _mm256_cmp_ps(x, min_v, _CMP_LT_OS);
        let lo_part = _mm256_fmadd_ps(_mm256_sub_ps(x, min_v), min_sl_v, min_v);
        // Região acima de max_val
        let gt_max  = _mm256_cmp_ps(x, max_v, _CMP_GT_OS);
        let hi_part = _mm256_fmadd_ps(_mm256_sub_ps(x, max_v), max_sl_v, max_v);
        // Blend: prioridade hi > lo > passthrough
        let y = _mm256_blendv_ps(_mm256_blendv_ps(x, lo_part, lt_min), hi_part, gt_max);
        _mm256_storeu_ps(data.as_mut_ptr().add(i), y);
        i += 8;
    }
    for x in data.iter_mut().skip(i) {
        if *x < min_val       { *x = (*x - min_val) * min_slope + min_val; }
        else if *x > max_val  { *x = (*x - max_val) * max_slope + max_val; }
    }
}
```

**Ganho esperado**: ~4× throughput vs escalar com branches (branch mispredicts eliminados).

---

#### T2.4 — Vetorizar `FastTanh` com AVX2 (Rational Polynomial SIMD)

**Arquivo**: `src/models/a2/activations.rs`

`fast_tanh` usa a aproximação racional de Padé de `NAM/activations.h:L111-122`:

```rust
x * (a + a*|x| + (b + c*|x|)*x²) / (d + (d + x²) * |x + e*x*|x||)
```

**Decisão aprovada**: usar `_mm256_div_ps` para a divisão — preciso (precisão total f32),
simples de implementar e manter. `rcp+NR` fica como otimização futura opcional.

**Estratégia AVX2 (com `_mm256_div_ps`)**:

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_tanh_slice_avx2(data: &mut [f32]) {
    // Constantes da aproximação de Padé (NAM/activations.h:L111-122):
    let ca        = _mm256_set1_ps(2.45550750702956_f32);
    let cb        = _mm256_set1_ps(0.893229853513558_f32);
    let cc        = _mm256_set1_ps(0.821226666969744_f32);
    let cd        = _mm256_set1_ps(2.44506634652299_f32);
    let ce        = _mm256_set1_ps(0.814642734961073_f32);
    let sign_mask = _mm256_set1_ps(-0.0_f32); // máscara de sinal para abs via andnot
    let mut i = 0;
    let len = data.len();
    while i + 8 <= len {
        let x  = _mm256_loadu_ps(data.as_ptr().add(i));
        let ax = _mm256_andnot_ps(sign_mask, x);            // |x|
        let x2 = _mm256_mul_ps(x, x);                       // x²
        // Numerador: x * (ca + ca*ax + (cb + cc*ax)*x²)
        let num_inner = _mm256_fmadd_ps(cc, ax, cb);        // cb + cc*|x|
        let num_poly  = _mm256_fmadd_ps(
            num_inner, x2, _mm256_fmadd_ps(ca, ax, ca));    // ca + ca*|x| + ...*x²
        let num = _mm256_mul_ps(x, num_poly);
        // Denominador: cd + (cd + x²) * |x + ce*x*|x||
        let xe        = _mm256_mul_ps(ce, _mm256_mul_ps(x, ax)); // ce*x*|x|
        let xterm     = _mm256_add_ps(x, xe);                    // x + ce*x*|x|
        let abs_xterm = _mm256_andnot_ps(sign_mask, xterm);      // |x + ce*x*|x||
        let den_inner = _mm256_add_ps(cd, x2);                   // cd + x²
        let den       = _mm256_fmadd_ps(den_inner, abs_xterm, cd); // cd + (cd+x²)*|...|
        // Divisão exata (precisão total f32, ~7-11 ciclos/lane):
        let y = _mm256_div_ps(num, den);
        _mm256_storeu_ps(data.as_mut_ptr().add(i), y);
        i += 8;
    }
    // Tail escalar: usar a função fast_tanh escalar existente
    for x in data.iter_mut().skip(i) { *x = fast_tanh(*x); }
}
```

**Nota para implementadores**: `_mm256_div_ps` tem latência de ~7–11 ciclos e throughput de
~1/ciclo em Zen3/Skylake. Para uma slice típica de ~32–256 elementos chamada por activation
batch no motor A2, o ganho sobre escalar é ~3–4× (8 elementos/instrução vs 1). A opção
`rcp + 1× NR` (~2× throughput, ~23 bits precisão) pode ser adicionada como variante futura
se benchmarks mostrarem necessidade. Não implementar agora.

**Critério de aceite T2.4**:

- Max erro absoluto vs `fast_tanh` escalar < 1e-5 em toda a faixa [-8, 8].
- Throughput ≥ 3× escalar (medido com `criterion` sobre 1024 elementos).

---

#### T2.5 — Testes de paridade para ativações SIMD

**Arquivo**: `src/models/a2/activations_test.rs`

Para cada ativação vetorizada, adicionar:

```rust
#[test]
fn test_hard_tanh_avx2_parity() {
    // Gerar 1024 f32 aleatórios em [-3.0, 3.0]
    // Aplicar HardTanh::apply() (usa AVX2 internamente)
    // Comparar com loop escalar x.clamp(-1,1)
    // Verificar: diff < 1e-7 (esperado: 0.0 pois min/max são exatos)
}

#[test]
fn test_hard_swish_avx2_parity() { ... }

#[test]
fn test_leaky_hard_tanh_avx2_parity() { ... }

#[test]
fn test_fast_tanh_avx2_parity() {
    // diff < 1e-5 vs fast_tanh_scalar
}
```

---

#### T2.6 — Auditar `src/math/activations/` (confirmar estado)

**Responsabilidade**: verificação/documentação, sem mudança de código esperada.

Com base na auditoria atual:

- `relu.rs`: ✅ AVX2 + AVX-512, loops `while i + 16` / `while i + 8`.
- `prelu.rs`: ✅ AVX2 + AVX-512, optimized para `s_len==1` (LeakyReLU) e `s_len==len`.
- `softsign.rs`: ✅ AVX2+FMA (dual 16-wide) + AVX-512.
- `silu.rs`: ✅ AVX2+FMA (dual 16-wide) + AVX-512. ⚠️ Scalar fallback usa `(-x).exp()` — se
  o fallback for chamado em contexto DSP, pode ser lento. Mas o fallback só ocorre para len < 8,
  o que é raro em produção.
- `sigmoid.rs`: ✅ AVX2+FMA (minimax D17) + AVX-512. ⚠️ Scalar fallback usa `exp()`.
- `tanh/`: A verificar — possui `high_fidelity.rs`, `production.rs`, `reference.rs`.

**Achado adicional — silu.rs:L75 e sigmoid.rs:L181**:
Os fallbacks escalares usam `(-*item).exp()` — proibido no hot-path RT (lento e não-determinístico).
Porém, para len < 8 (raro), são chamados. Para garantia absoluta, substituir por aproximações
racionais escalares (`fast_tanh_scalar`, `scalar_minimax_sigmoid`) mesmo no fallback.

**Tarefa T2.6a**: Confirmar que os fallbacks escalares de `silu_slice`, `sigmoid_slice` e
`softsign_slice` usam aproximações racionais (não `f32::exp`) — ou corrigir.

---

### Sprint S3 — Benchmarks e Registro de Baseline MT7

**Objetivo**: capturar métricas antes/depois e documentar o ganho formal de MT7.

#### T3.1 — Baseline pré-otimização

```bash
# Antes de qualquer mudança de S1/S2:
cargo bench --bench inference_bench -- A2Full A2Lite 2>&1 | tee target/logs/bench_baseline_mt7.txt
```

Registrar: `A2Full_CH8_64samp_48kHz` e `A2Lite_CH3_64samp_48kHz` em ns/iter.

#### T3.2 — Benchmark pós-otimização

```bash
# Após S1 + S2:
cargo bench --bench inference_bench -- A2Full A2Lite 2>&1 | tee target/logs/bench_after_mt7.txt
# Benchmark específico head:
cargo bench --bench inference_bench -- head 2>&1 | tee target/logs/bench_head_mt7.txt
```

**Critério de aceite S3**: `A2Full_CH8` deve mostrar redução ≥ 5% (estimado ~8–12%) no tempo total
de inferência. O head conv representa ~15-20% do tempo total A2.

---

### Sprint S4 — Verificação Final e Linting

**Objetivo**: garantir que MT7 não quebra nenhum teste existente e cumpre todas as regras do projeto.

#### T4.1 — `cargo check` (zero warnings)

```bash
cargo check 2>&1
```

#### T4.2 — `cargo test` (suite completa)

```bash
utils/tests-cargo.sh
```

**Critério de aceite**: 0 falhas, 0 regressions vs baseline.

#### T4.3 — Heap audit para head conv SIMD

Verificar que `A2HeadConv::process()` com os novos kernels não introduz alocações no RT:

```bash
# Se existir heap_audit para A2:
cargo test --features heap-audit a2_heap_audit -- --nocapture
```

#### T4.4 — Revisar copyright/SPDX em arquivos modificados

Confirmar que todo arquivo `.rs` modificado nos Sprints S1–S3 contém o cabeçalho SPDX.

---

## Épico E2 — MT7 Extensão: Ativações via `math/activations/` Integradas ao motor A2 geral

> **Nota**: Este épico é preparatório para MT2 (Motor A2 Geral). Pode ser executado
> independentemente de MT2, mas entrega valor somente quando MT2 for ativado.

### Sprint S5 — Conectar `ActivationType` ao dispatcher SIMD centralizado

**Objetivo**: Substituir as implementações inline de `HardTanh`, `HardSwish`, `FastTanh`,
`LeakyHardTanh` por funções em `src/math/activations/` que usem o padrão `SIMD_MATH`.

**Rationale**: as ativações vetorizadas em S2 ficam diretamente em `activations.rs`. Para
consistência com o resto do codebase (onde todas as ativações matemáticas vivem em `math/`),
migrar para `math/activations/` e expor via `SIMD_MATH` struct.

**Arquivos novos propostos**:

- `src/math/activations/hard_tanh.rs` — `hard_tanh_slice_avx2`, `hard_tanh_slice_avx512`
- `src/math/activations/hard_swish.rs` — `hard_swish_slice_avx2`, `hard_swish_slice_avx512`
- `src/math/activations/fast_tanh.rs` — `fast_tanh_slice_avx2` (racional Padé vectorizado)
- `src/math/activations/leaky_hard_tanh.rs` — `leaky_hard_tanh_slice_avx2`

**Atualizar `SIMD_MATH` struct** em `src/math/common/` para incluir ponteiros de função
para as novas ativações.

**Critério de aceite S5**: `ActivationType::apply()` em `activations.rs` passa a chamar
`(SIMD_MATH.hard_tanh_slice)(data)` etc., idêntico aos casos `Tanh`, `ReLU`, `PReLU`.

> ⚠️ **Atenção**: S5 tem escopo maior (modifica `SIMD_MATH` struct, que é shared infrastructure).
> Executar apenas após S1–S4 serem aprovados e merged.

---

## Registro de Dependências e Ordem de Execução

```text
S1 (A2 Head SIMD) → independente, executar primeiro
S2 (Ativações inline) → independente, pode ser paralelo a S1
S3 (Benchmarks) → após S1 e S2
S4 (Verificação) → após S1, S2, S3
S5 (Migração math/) → após S1–S4, pré-requisito para MT2
```

---

## Critérios Globais de Aceite do Épico E1

- [ ] `cargo test` — 0 falhas, 0 regressões
- [ ] `cargo check` — 0 warnings
- [ ] Benchmarks: A2Full ≥ 5% mais rápido, A2Lite ≥ 3% mais rápido
- [ ] Paridade bitwise/numérica vs `a2_head_block_scalar_ref` para todos CH e geometrias
- [ ] Paridade numérica (< 1e-5) para todas as ativações vetorizadas vs. escalar
- [ ] Zero alocações no hot-path (heap-audit)
- [ ] Todos os arquivos modificados com cabeçalho SPDX Apache-2.0
- [ ] `TODO-features.md` MT7 marcado como ✅ Concluído após todos os testes passarem

---

## Notas Técnicas e Armadilhas Conhecidas

### A2 Head Ring Buffer Layout

O `head_history` em produção tem layout `[col][CH]` — col-major por frame. Para CH=8:
`head_history[col * 8 .. col * 8 + 8]` = 8 f32 contíguos. **`_mm256_loadu_ps` é válido**.
Para CH=3: `head_history[col * 3 .. col * 3 + 3]` = 3 f32 (NÃO 4). **`_mm_loadu_ps` lê lixo
no 4º lane**. Solução: usar `_mm_setr_ps` com 3 loads escalares (seguro mas lento) ou introduzir
padding CH=3→4 no buffer de história (mudança arquitetural — avaliar custo/benefício).

**Recomendação final para CH=3 head**: usar `_mm_setr_ps(*h, *(h+1), *(h+2), 0.0)` — 3 loads
escalares + 1 instrução de empacotamento. Para K=16 taps por frame, isso é 16 × `_mm_setr_ps` +
16 × `_mm_fmadd_ps` + redução. Ganho vs escalar puro: ~2.5× (menos que o ideal de 4× mas seguro).

### FastTanh `rcp` vs `div`

`_mm256_div_ps`: latência ~7–11 ciclos, throughput 1/ciclo, precisão total.
`_mm256_rcp_ps + 1× NR`: latência ~5 ciclos, throughput 2/ciclo, precisão ~23 bits.
Para DSP com target erro < -80 dB, `rcp + 1× NR` é suficiente e mais rápido.
**Usar `_mm256_div_ps` se simplicidade for prioritária**; `rcp+NR` se throughput for crítico.
Documentar escolha no código.

### LeakyHardTanh Blend Order

`_mm256_blendv_ps(a, b, mask)` retorna `b` onde mask=0xFF e `a` onde mask=0x00.
Na implementação de T2.3, assegurar que o blend de hi > lo seja correto:
quando `x > max_val`, garantir que `hi_part` domine sobre `lo_part` (impossível ter ambas
condições simultaneamente para um escalar, mas verificar edge case em `x ≈ min_val ≈ max_val`).
