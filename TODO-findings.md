<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings — Auditoria de Performance (perfil "Master of performance")

> Artefato gerado pela skill `revisor-auditor` (foco exclusivo na role **Master of
> performance**) e formatado pela skill `planejador-arquiteto`.
>
> **Escopo:** varredura de oportunidades de performance em **todo** o codebase,
> não só nos hot paths óbvios — desde o kernel de inferência neural até a thread
> RT de áudio, passando por cabsim, resampler, BatchNorm, dispatch SIMD e flags
> de build. Cada achado traz diagnóstico, impacto estimado, proposta de solução,
> risco/esforço e estratégia de validação.
>
> **Convenção de severidade:**
>
> - 🔴 **ALTA** — ganho de CPU relevante e/ou afeta diretamente fluidez de áudio/UX.
> - 🟠 **MÉDIA** — ganho mensurável; vale a pena, risco controlado.
> - 🟡 **BAIXA** — micro-otimização ou higiene; ganho pequeno, mas barato.

---

## 0. Contexto: o que já está excelente (linha de base)

Antes dos achados, é importante registrar que o codebase já opera num patamar de
otimização altíssimo. Estas práticas **não** devem ser regredidas e servem de
referência de qualidade para as correções propostas:

- **Dispatch SIMD monomorfizado por bloco** (`dispatch_simd!`): a ISA é resolvida
  **uma vez** no topo de `process()` e o corpo inteiro é monomorfizado via
  `<M: SimdMath>` (ex.: `src/models/wavenet/model.rs:80-143`,
  `src/models/a2/model/static/process.rs:43-87`). Padrão correto.
- **Kernels fundidos** (tanh+accumulate+seed, gated tanh·sigmoid+accumulate) em
  `src/math/wavenet/accumulate/avx2.rs` — minimizam passagens de memória.
- **Conv1D WaveNet** com dual-frame tiling, prefetch em 2 estágios, pesos f32
  nativos, `MaybeUninit` para evitar `memset` de 4 KB (`src/models/wavenet/layer.rs`,
  `conv1d.rs`).
- **Tanh/Sigmoid polinomiais** exp-based com FMA, branchless, dual-path
  (`src/math/activations/tanh/high_fidelity.rs`) — qualidade de referência.
- **Huge pages** (MAP_HUGETLB + fallback `MADV_HUGEPAGE`) em
  `src/math/common/huge_alloc.rs` — reduz TLB miss em buffers grandes.
- **FTZ/DAZ** setados na thread RT e reafirmados periodicamente
  (`src/math/common/ops.rs`), `-z now` no linker (`.cargo/config.toml`).
- **Resampler** com ponteiros de função cacheados por ISA (elimina `dispatch_simd!`
  por chamada) e delay-line double-buffer sem wrap no loop SIMD
  (`src/dsp/resampler.rs`).
- **RT thread** sem heap/lock/syscall no hot path; SPSC `rtrb` com capacidades
  limitadas; double-buffer com ordering `Acquire`/`Release` correto.

Os achados abaixo são, portanto, o **conjunto remanescente** de oportunidades.

---

## 1. Resumo executivo

| ID  | Achado                                                                          | Severidade     | Esforço | Subsistema afetado                |
| --- | ------------------------------------------------------------------------------- | -------------- | ------- | --------------------------------- |
| P1  | FFT/IFFT do cabsim 100% escalar + indexação com bounds-check no loop quente     | 🔴 ALTA        | Alto    | Cab Sim (todos os usuários de IR) |
| P2  | Engine de convolução do cabsim é AVX2-only (sem AVX-512) + store escalar do FDL | 🟠 MÉDIA       | Médio   | Cab Sim em CPUs AVX-512           |
| P3  | BatchNorm (ConvNet) faz gather/scatter strided via pilha + `#[inline(never)]`   | 🟠 MÉDIA-ALTA  | Médio   | Modelos ConvNet                   |
| P4  | Re-dispatch SIMD em contexto já monomorfizado (A2/ConvNet/gating/activations)   | 🟠 MÉDIA       | Médio   | A2, ConvNet, ativações            |
| P5  | GEMV f32 `out_len==1` faz transposição via store/load escalar na pilha          | 🟠 MÉDIA       | Médio   | DenseLayer (head projection)      |
| P6  | Branch loop-invariante por amostra no head do LSTM                              | 🟡 BAIXA-MÉDIA | Baixo   | Modelos LSTM                      |
| P7  | `panic = "unwind"` no profile release do binário distribuído                    | 🟡 BAIXA       | Baixo   | Tamanho/codegen do binário        |
| P8  | Passagens redundantes no input stage + cópias mono + stores atômicos por bloco  | 🟡 BAIXA       | Baixo   | Pipeline DSP                      |
| P9  | `assert_eq!` (panic em release) no caminho RT do FFT                            | 🟡 BAIXA       | Trivial | Estabilidade RT (anti-XRUN)       |

---

## 2. Achados detalhados

### P1 — 🔴 FFT/IFFT do cabsim totalmente escalar (e com bounds-check no loop mais quente)

**Localização:**
`src/math/dsp/fft.rs:186-227` (`FftPlanner::process`), `:236-285`
(`process_inverse`), `:372-421` (`RfftPlanner::process_forward`), `:435-501`
(`process_inverse`). Consumido por `src/dsp/cabsim/conv.rs:254-356`.

**Diagnóstico:**
O núcleo do simulador de gabinete (UPOLS) executa **duas** transformadas por bloco
de áudio (1 RFFT direta + 1 IRFFT). O kernel de butterfly Radix-2 DIT é **puramente
escalar** — opera elemento a elemento sobre `T` (`f32`), sem qualquer uso de SIMD:

```rust
// src/math/dsp/fft.rs:206-223 — loop mais quente do cabsim, escalar
for k in (0..n).step_by(len) {
    for j in 0..half {
        let w_idx = j * step;
        let w_re = self.twiddle_re[w_idx];   // ← indexação com bounds-check
        let w_im = self.twiddle_im[w_idx];   // ← idem
        let idx1 = k + j;
        let idx2 = k + j + half;
        let t_re = w_re.mul_add(re[idx2], -w_im * im[idx2]);  // ← re[idx2]/im[idx2] checados
        let t_im = w_re.mul_add(im[idx2],  w_im * re[idx2]);
        re[idx2] = re[idx1] - t_re;
        im[idx2] = im[idx1] - t_im;
        re[idx1] = re[idx1] + t_re;
        im[idx1] = im[idx1] + t_im;
    }
}
```

Há **dois** problemas sobrepostos:

1. **Ausência de SIMD.** Para `FFT_SIZE` típico de cabsim (1024–4096), são
   `N/2·log₂N` butterflies; cada um faz ~10 flops escalares. Em `x86-64-v3`, o
   compilador não auto-vetoriza este padrão (dependências de escrita cruzada
   `re[idx1]/re[idx2]` + acesso a twiddle strided). Estima-se **6–10× de margem**
   no estágio FFT em CPUs com AVX2/AVX-512.
2. **Indexação com verificação de limites** (`re[idx1]`, `twiddle_re[w_idx]`, etc.)
   no loop mais interno — múltiplos `bounds check` por butterfly. Para um cab de
   FFT 2048, isso são dezenas de milhares de checagens por bloco, ×2 transformadas.
   Também a permutação bit-reversal usa `re.swap(i, j)` (checado) por elemento.

**Impacto:** alto para **todos** os usuários que ativam o cabsim nativo — em
qualquer ISA (o gargalo é independente de AVX-512). É, provavelmente, a maior
reserva de CPU não-explorada do projeto fora dos modelos neurais.

**Proposta de solução (incremental, do mais barato ao mais completo):**

1. **(Quick win, baixo risco)** Trocar a indexação checada por
   `get_unchecked`/`get_unchecked_mut` no butterfly e no bit-reversal, com
   `// SAFETY:` documentando que `idx1,idx2 < n` e `w_idx < n/2` são garantidos por
   construção. Ganho imediato sem mudar o algoritmo.
2. **(Núcleo)** Vetorizar os estágios em que `half >= LARGURA_SIMD` (8 para AVX2,
   16 para AVX-512): nesses estágios, `re[idx1..idx1+W]` e `re[idx2..idx2+W]` são
   contíguos. O obstáculo é o twiddle `w_idx = j·step` (strided). Solução clássica:
   **pré-computar as tabelas de twiddle por estágio em ordem de acesso (contígua)**,
   trocando ~`n` de armazenamento por carga vetorial contígua — habilita
   `loadu`+FMA por lane. Estágios finais (`step` pequeno) já são naturalmente
   contíguos.
3. **(Avançado, opcional)** Migrar para **split-radix** ou **radix-4** (reduz ~25%
   das multiplicações) e/ou um bit-reversal vetorizado/livre de permutação
   (algoritmo in-order tipo Stockham), eliminando o `swap` escalar.

Manter o caminho escalar como referência de paridade em `#[cfg(test)]`.

**Risco/Esforço:** passo 1 é trivial e seguro; passos 2–3 exigem cuidado de
paridade numérica (validar contra a impl. escalar atual e contra `f64`) e testes
de SNR do cabsim. Esforço total: **Alto**.

**Validação:** `cargo test` (paridade FFT já tem testes em `fft_test.rs`); bench
dedicado de `ConvEngine::process` por número de partições; validar resposta ao
impulso (IR identidade → passthrough) e SNR ≥ atual.

---

### P2 — 🟠 Engine de convolução do cabsim é AVX2-only (sem caminho AVX-512) + store escalar do FDL

**Localização:** `src/dsp/cabsim/conv.rs:272-348` (MAC no domínio da frequência),
`:259-262` (escrita no FDL).

**Diagnóstico:**
O MAC complexo (multiplicação-acumulação espectral sobre todas as partições) está
escrito com intrínsecos **AVX2 fixos** (`_mm256_loadu_ps`, `_mm256_fmadd_ps`,
`_mm256_fnmadd_ps`), divergindo do resto do projeto, que despacha via `SimdMath`:

```rust
// src/dsp/cabsim/conv.rs:272-275 — hardcoded AVX2, ignora AVX-512
use core::arch::x86_64::{
    _mm256_add_ps, _mm256_fmadd_ps, _mm256_fnmadd_ps, _mm256_loadu_ps, _mm256_mul_ps,
    _mm256_storeu_ps,
};
```

Em CPUs AVX-512, este laço processa **8 bins por iteração** em vez de 16 — metade
da largura. Para IRs longas (muitas partições), o MAC é uma fração significativa do
custo do cabsim.

Adicionalmente, a escrita do espectro de entrada no FDL é um laço escalar
elemento-a-elemento:

```rust
// src/dsp/cabsim/conv.rs:259-262 — deveria ser copy_from_slice (memcpy vetorizado)
for k in 0..self.n_bins {
    self.fdl_re[fdl_base + k] = self.fft_buf_re[k];
    self.fdl_im[fdl_base + k] = self.fft_buf_im[k];
}
```

**Impacto:** médio em CPUs AVX-512; pequeno em AVX2 (já é a largura nativa), exceto
pela cópia do FDL que é desperdício em qualquer ISA.

**Proposta de solução:**

1. Extrair o MAC complexo para um método do trait `SimdMath` (ex.:
   `complex_mac_accumulate` / `complex_mac_overwrite`) com implementações AVX2 e
   AVX-512, e despachar **uma vez** no início de `ConvEngine::process` (cachear a
   ISA no struct, à la resampler). O `p_count==1` vs geral vira parâmetro.
2. Substituir o laço de store do FDL por
   `self.fdl_re[fdl_base..fdl_base+n_bins].copy_from_slice(&self.fft_buf_re[..n_bins])`
   (idem `im`).

**Risco/Esforço:** baixo-médio. A multiplicação complexa é padrão; validar paridade
bit-a-bit AVX2 vs AVX-512 vs escalar. **Médio.**

**Validação:** teste de paridade do MAC por partição; bench de `process` em CPU
AVX-512 (relação partições × tempo).

---

### P3 — 🟠 BatchNorm (ConvNet) faz gather/scatter strided via buffer de pilha + `#[inline(never)]`

**Localização:** `src/models/convnet/batch_norm.rs:148-190`.

**Diagnóstico:**
O layout de dados é **frame-interleaved** (`[f0_c0, f0_c1, …, f1_c0, …]`) e os
parâmetros `scale`/`offset` são **contíguos** por canal (`AlignedVec`). Porém o
kernel processa **channel-major** com passo `n_ch`, o que o obriga a emular um
*gather/scatter* via buffer de pilha:

```rust
// src/models/convnet/batch_norm.rs:166-178
while f + 8 <= num_frames {
    let base = ch + f * n_ch;
    for lane in 0..8 {                                   // 8 loads escalares (gather)
        *gather_buf.get_unchecked_mut(lane) = *data.get_unchecked(base + lane * n_ch);
    }
    let xv = _mm256_loadu_ps(gather_buf.as_ptr());
    let yv = _mm256_fmadd_ps(xv, s, o);
    _mm256_storeu_ps(gather_buf.as_mut_ptr(), yv);
    for lane in 0..8 {                                   // 8 stores escalares (scatter)
        *data.get_unchecked_mut(base + lane * n_ch) = *gather_buf.get_unchecked(lane);
    }
    f += 8;
}
```

Isso são **16 operações escalares de memória por chunk SIMD** só para alimentar 1
FMA. Além disso, o kernel é `#[inline(never)]` (linha 149), forçando overhead de
chamada/retorno por bloco/chunk num corpo pequeno.

**Diagnóstico de causa raiz:** a escolha channel-major existe para *broadcast* de
`scale[c]`/`offset[c]`. Mas, no layout frame-interleaved, **uma leitura contígua de
`n_ch` elementos já é um frame inteiro** — e `scale`/`offset` contíguos casam
diretamente com essa leitura. Logo, a transposição é desnecessária.

**Proposta de solução:**

- Reescrever **frame-major contíguo**:
  - Para `n_ch == 8` (largura comum em ConvNet): carregar `scale`/`offset` **uma
    vez** em registradores e, por frame, fazer
    `store(data+f*8, fmadd(loadu(data+f*8), scale_vec, offset_vec))`. Zero gather.
  - Para `n_ch` geral: processar `n_ch` contíguos por frame com `maskload`/
    `maskstore` (AVX-512: `_mm512_maskz_loadu_ps`) no resto.
- Remover `#[inline(never)]` (deixar o compilador decidir; o corpo é pequeno e
  quente).
- Adicionar despacho via `SimdMath` (ex.: `BatchNorm1D::process_simd::<M>`) para
  usar AVX-512 quando disponível, já que `block.rs` chama dentro de
  `process_block_internal<M>`.

**Impacto:** médio-alto para modelos ConvNet (elimina ~16 ops escalares por chunk e
o overhead de chamada). Ganho cresce com `num_frames`.

**Risco/Esforço:** médio. Atenção: manter exatamente a mesma ordem de operações
(FMA) para paridade. Há `process_scalar` de referência para testes. **Médio.**

**Validação:** `batch_norm_test.rs` (paridade já existe); bench de bloco ConvNet.

---

### P4 — 🟠 Re-dispatch SIMD em contexto já monomorfizado (tema recorrente)

**Localização (3 manifestações do mesmo padrão):**

**(a) A2 estático** — `src/models/a2/model/static/process.rs:225-266`. Dentro de
`layer_forward_dispatch::<M: SimdMath>` (que **já** é monomorfizada por `M`), relê o
global `LazyLock`:

```rust
// process.rs:225 — relê SIMD_MATH (LazyLock deref + branch) por camada por bloco
let isa = crate::math::common::SIMD_MATH.instruction_set;
match isa {
    InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 =>
        layer_forward_ch8_block_simdmath::<Avx512Math>(...),  // M é ignorado aqui
    _ => layer_forward_ch8_block(...),                        // versão não-genérica
}
```

`M` já codifica a ISA em tempo de compilação, mas o código lê o `LazyLock` em
runtime (≈23 camadas × nº de chunks por bloco). *(Observação: o conv ch8 do A2 é
f32-nativo — confirmado em `conv1d_ch8/simd.rs` sem bf16 —, então o hardcode de
`Avx512Math` para o caso VNNI/BF16 não perde kernel especializado; o problema é
puramente a re-leitura/branch redundante.)*

**(b) Ativações via free function** — `src/models/a2/activations.rs:82-153`
(`ActivationType::apply`) chama `crate::math::activations::*_slice`, que internamente
faz `dispatch_simd!` **por chamada**:

```rust
// src/math/activations/mod.rs:52-54
pub fn tanh_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(tanh_slice(data)); // lê SIMD_MATH + match toda chamada
}
```

Acionado em hot path: `src/models/convnet/block.rs:161`
(`self.activation.apply(scratch_slice)` por bloco/chunk) e em
`src/models/a2/.../gating.rs:69-70,151-152`.

**(c) BatchNorm AVX2-only** — `src/models/convnet/block.rs:158`
(`self.bn.process()`) não recebe `M` (ver também P3).

**Diagnóstico:** o `dispatch_simd!` expande para `SIMD_MATH.instruction_set`, que
desreferencia um `LazyLock<SimdMathConfig>` — isto é, um *load atômico (Acquire) do
estado do `Once` + branch de inicialização* a cada acesso, além do `match` da ISA.
Após a inicialização o branch é previsível, mas o load atômico atua como **barreira
de otimização** e impede que o compilador trate a ISA como constante. Em contexto já
monomorfizado, isso é puro desperdício.

**Proposta de solução:**

1. Adicionar uma **const associada** ao trait `SimdMath`, ex.:
   `const ISA: InstructionSet;` (cada impl declara a sua). Trocar o `match isa` de
   (a) por `match M::ISA` → resolvido em tempo de compilação (zero load, zero
   branch, elimina o arm morto via DCE).
2. Criar `ActivationType::apply_simd::<M: SimdMath>(&self, data)` que chama
   diretamente `M::tanh_slice`/`M::prelu_slice`/… (monomorfizado). Migrar
   `convnet/block.rs` e `gating.rs` para usá-la. Manter `apply()` (sem `M`) só para
   callers fora de contexto SIMD (loader/testes).
3. Encaminhar `M` ao BatchNorm (`process_simd::<M>`) — casado com P3.

**Impacto:** médio. Remove `LazyLock` deref + branch dos laços por camada/bloco em
A2-geral, ConvNet e gating. Ganho por chamada é pequeno, mas a soma em modelos com
muitas camadas/chunks é real e o código fica mais limpo e auditável.

**Risco/Esforço:** baixo-médio (refactor mecânico, sem mudança de algoritmo).
**Médio.**

**Validação:** `cargo test` de paridade dos modelos A2/ConvNet; inspeção do asm
gerado (o `match M::ISA` deve desaparecer).

---

### P5 — 🟠 GEMV f32 no caminho `out_len==1` faz transposição via store/load escalar na pilha

**Localização:** `src/math/gemm/gemv/f32_avx2.rs:84-165` (8 frames/YMM) e
`src/math/gemm/gemv/f32_avx512.rs:74-...` (16 frames/ZMM). Alcançável por
`DenseLayer::process_block` (`src/models/wavenet/dense.rs:59-72`) quando `OUT==1`
(projeção de head, caminho "OUT≤4 → frame-batching").

**Diagnóstico:**
Para colocar 8 (ou 16) frames nas lanes de um acumulador e evitar reduções
horizontais, o kernel **transpõe** um tile `frames × canais` emulando um *gather*
com buffers de pilha + laço escalar:

```rust
// f32_avx2.rs:105-124 — 64 stores escalares + 8 loads vetoriais por chunk de 8 canais
let mut buf0 = [0.0f32; 8]; /* …buf1..buf7… */
for j in 0..8 {
    let base = (n + j) * in_len;
    buf0[j] = *in_frames.get_unchecked(base + ic);     // gather strided por in_len
    /* buf1[j]..buf7[j] … */
}
let v_in0 = _mm256_loadu_ps(buf0.as_ptr());            // recarrega da pilha
/* … */
```

No AVX-512 o custo dobra (128 stores escalares por chunk, `buf*[16]`). Esse
*store-to-load* na pilha tende a **anular** o benefício de evitar a redução
horizontal, especialmente para `in_len` pequeno (head tem poucos canais de
entrada).

**Proposta de solução:**

1. **Transposição vetorizada real** (8×8 via `_mm256_unpacklo/hi_ps` +
   `_mm256_permute2f128_ps`; 16×16 no AVX-512) — ~8–16 shuffles em vez de 64–128
   ops escalares de memória; **ou**
2. **Reusar o caminho contíguo por-frame** (já presente no laço remanescente,
   `f32_avx2.rs:166-191`): `loadu` contíguo de `in_frames[n*in_len..]` + `loadu` de
   `weights` + FMA + 1 redução horizontal por frame. Para `in_len` pequeno isto é
   provavelmente mais rápido que a transposição.
3. Escolher por **limiar calibrado em bench** (`in_len` de corte) entre (1) e (2).

**Impacto:** médio **se** o modelo usa DenseLayer com `OUT==1` no hot path (certas
projeções de head). Para os heads WaveNet feitos via kernels de acumulação, não
aplica — daí a severidade média e não alta.

**Risco/Esforço:** médio; exige paridade contra `gemv_*_fallback` (testes já
existem em `gemv_test.rs`) e microbench. **Médio.**

**Validação:** `cargo bench` (`dot_4x_bench`/GEMV) com `out_len=1` em vários
`in_len` e `num_frames`; paridade contra fallback escalar.

---

### P6 — 🟡 Branch loop-invariante por amostra no head do LSTM

**Localização:** `src/models/lstm/head_projection.rs:26-34`
(`compute_lstm_head_simd!`), usada em `model1.rs:21-29`, `model2.rs:43-56`,
`model_dyn.rs:86/121/155`.

**Diagnóstico:**
O LSTM é intrinsecamente sequencial (uma amostra por iteração). O macro do head
avalia `self.use_f32_head` — um `bool` **invariante** durante todo o
`process()` — **dentro** do laço por amostra:

```rust
// head_projection.rs:28-33
(if $self.use_f32_head {                          // ← reavaliado a cada amostra
    let h_f32 = $get_f32_hidden;
    dot_product_f32_native(h_f32, &$self.head_weights_f32)
} else {
    $dot_prod($h_quant, &$self.head_weights)       // caminho quantizado
}) + $self.head_bias
```

O LLVM *pode* fazer loop-unswitching, mas não é garantido por haver chamada mutante
(`process_sample`) no corpo; o branch seleciona dois caminhos de código distintos
(dot f32 vs quantizado), o que dificulta a hoist automática.

**Proposta de solução:** **loop-unswitching explícito** — testar `use_f32_head`
**uma vez** antes do laço e despachar para um de dois laços monomórficos por
amostra. Garante a hoist independentemente do compilador e permite que cada laço
inline o seu `dot_product` específico.

**Impacto:** baixo-médio. Em termos absolutos o branch é barato, mas está no laço
mais apertado do LSTM e pode inibir outras otimizações.

**Risco/Esforço:** baixo (reorganização local). **Baixo.**

**Validação:** `lstm_test.rs` (paridade); bench LSTM antes/depois.

---

### P7 — 🟡 `panic = "unwind"` no profile release do binário distribuído

**Localização:** `Cargo.toml:104` (`panic = "unwind"`).

**Diagnóstico:** o comentário explica que `unwind` é mantido por compatibilidade com
os harnesses de bench/test. Porém, para o **artefato distribuído** (cdylib CLAP /
binário standalone), `unwind` gera *landing pads* e tabelas de unwinding que
incham o código e podem inibir algumas otimizações de codegen. Em contexto RT, um
panic na thread de áudio deveria, na prática, abortar de qualquer forma.

**Proposta de solução:** separar perfis: manter `unwind` para `bench`/`test` e
adotar `panic = "abort"` no perfil de **release de distribuição** (via
`profile.release` dedicado + `--profile` no `utils/build-release.sh`, ou um
`profile` custom). Medir delta de tamanho do `.so`/binário e de performance. Validar
que o host CLAP tolera `abort` (a alternativa a um panic já seria fatal).

**Impacto:** baixo (tamanho de binário e leve melhora de codegen). Requer cuidado
com a matriz de build.

**Risco/Esforço:** baixo, mas exige validação de integração CLAP/host. **Baixo.**

**Validação:** build de release com cada opção; smoke test de carga no host;
comparar tamanho e benches.

---

### P8 — 🟡 Passagens redundantes no input stage + cópias mono + stores atômicos por bloco

**Localização:** `src/dsp/pipeline/stages/input.rs:82-164`;
`src/dsp/pipeline/stages/output.rs:184-188`;
`src/clap/processor/dsp/channels.rs:27`;
`src/clap/processor/dsp/orchestrator.rs:28`.

**Diagnóstico:**

- **Input stage** faz até **4 passagens lineares** sobre o mesmo buffer por bloco:
  energia (`compute_energy*`), detecção mono (`compute_max_diff`), ganho
  (`apply_gain`) e dither de denormal (`apply_dither_add`). Cada uma é SIMD, mas são
  4 varreduras de cache distintas.
- **Mono** faz `copy_nonoverlapping(L → R)` por bloco no output
  (`output.rs:184-188`) — memcpy de buffer inteiro evitável.
- **Stores atômicos por bloco** mesmo sem mudança: `active_channel_count.store(...)`
  e `last_n_samples.store(...)` toda chamada.

**Proposta de solução:**

- **Fundir** ganho + dither numa única passagem (e, quando possível, energia junto),
  reduzindo de 4 para 2–3 varreduras. Um kernel `apply_gain_then_dither` fundido
  resolve o caso mais comum.
- Para mono, escrever L em ambos os canais **direto no `DspBridge`** (sem memcpy
  intermediária L→R).
- Condicionar os `store` atômicos a mudança de valor (`if novo != atual`).

**Impacto:** baixo (cada item é pequeno), mas barato e melhora pressão de cache em
buffers grandes.

**Risco/Esforço:** baixo. **Baixo.**

**Validação:** testes de pipeline DSP; bench de bloco em tamanhos 128/512/2048.

---

### P9 — 🟡 `assert_eq!` (panic em release) no caminho RT do FFT

**Localização:** `src/math/dsp/fft.rs:187-188, 237-238, 376-378, 439-441`.

**Diagnóstico:** o FFT usa `assert_eq!` (não `debug_assert_eq!`) para validar
tamanhos de buffer. Como o FFT é chamado dentro de `ConvEngine::process` (thread
RT), um mismatch dispararia **panic em release na thread de áudio** — contrariando a
política do projeto ("never panics on the RT thread"). Os tamanhos são invariantes
garantidos por `ConvEngine::new()`, então a checagem é, na prática, morta no hot
path, mas o `assert` permanece como custo/risco.

**Proposta de solução:** rebaixar para `debug_assert_eq!` (mantém a validação em
debug/test, remove o branch/panic em release). Documentar a invariância no contrato
da função.

**Impacto:** estabilidade RT (anti-XRUN) e remoção de branches mortos no hot path.

**Risco/Esforço:** trivial. **Trivial.**

**Validação:** `cargo test` (debug mantém os asserts); revisão de que todos os
chamadores garantem os tamanhos.

---

## 3. Épicos (agrupamento para execução)

> Agrupamento pensado para **maximizar ganho com risco controlado** e permitir
> entregas atômicas e verificáveis. Cada épico referencia os achados (P#) acima.

### Épico A — "Cab Sim de alta performance" (P1, P2, P9) [DONE]

Maior reserva de CPU do projeto fora dos modelos neurais e isolada num subsistema
opcional (baixo risco de regressão no caminho neural). Sequência sugerida:

1. **P9** (trivial, destrava segurança RT) →
2. **P1 passo 1** (`get_unchecked` no FFT — quick win seguro) →
3. **P2** (AVX-512 no MAC + `copy_from_slice` no FDL) →
4. **P1 passos 2–3** (vetorização das butterflies / split-radix).
   **Critério de pronto:** SNR do cabsim ≥ atual; bench `ConvEngine::process` com ganho
   demonstrável por nº de partições em AVX2 e AVX-512.

### Épico B — "Dispatch monomorfizado de ponta a ponta" (P4, P3) [TO-DO]

Elimina re-dispatch redundante e o gather/scatter do BatchNorm, unificando o padrão
`<M: SimdMath>` em A2-geral/ConvNet/gating. Sequência:

1. Adicionar `const ISA` ao trait `SimdMath` e migrar A2 estático (P4a).
2. `ActivationType::apply_simd::<M>` + migração de ConvNet/gating (P4b).
3. **P3** (BatchNorm frame-major + `process_simd::<M>`), que também consome P4c.
   **Critério de pronto:** asm sem leitura de `SIMD_MATH` nos laços por camada/bloco;
   paridade dos modelos preservada.

### Épico C — "Kernels GEMV/LSTM afinados" (P5, P6) [TO-DO]

Micro-otimizações nos kernels de projeção. Independentes do resto; bom candidato a
paralelizar. Sequência:

1. **P6** (loop-unswitching do head LSTM — baixo risco).
2. **P5** (transposição vetorizada ou caminho contíguo no GEMV `out_len==1`, com
   limiar calibrado por bench).
   **Critério de pronto:** benches LSTM e GEMV (`out_len=1`) com ganho e paridade
   contra fallback.

### Épico D — "Higiene de pipeline e build" (P8, P7) [TO-DO]

Itens de baixo risco e baixo esforço, agrupados para uma sprint de polimento.

1. **P8** (fusão de passagens no input stage, cópia mono direta no bridge, stores
   atômicos condicionais).
2. **P7** (avaliar `panic = "abort"` no perfil de distribuição).
   **Critério de pronto:** sem regressão funcional no pipeline; binário de distribuição
   menor (se P7 adotado) e validado no host CLAP.

---

### Priorização recomendada

`Épico A` (impacto/risco favorável e isolado) → `Épico B` (limpeza estrutural de
amplo alcance) → `Épico C` (afinação de kernels) → `Épico D` (polimento).

> **Nota metodológica:** todos os achados são otimizações de **estrutura/execução**,
> sem alterar a lógica numérica de inferência. Cada correção deve preservar paridade
> com a referência (NeuralAmpModelerCore e/ou os fallbacks escalares existentes) e
> ser acompanhada de bench antes/depois, conforme as regras em `.agents/rules/`.
