<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Oportunidades de Otimização de Performance e Low-Latency

> **Role foco:** Performance and Low-Latency Master
> **Baseline ISA:** x86-64-v3 (AVX2 + FMA + BMI2) — universal em TODO o código, não apenas no áudio
> **Restrição absoluta:** Zero regressão de paridade com NAMcore ou de fidelidade sonora
> **Ferramentas de defesa:** `utils/quality-dashboard.sh` e `utils/tests-performance-regression.sh`
> **Data da auditoria:** 2026-07-06

---

## Visão Executiva

Esta auditoria analisou exaustivamente a codebase do NAM-rs sob a ótica do **Performance and Low-Latency Master**, com foco exclusivo no baseline **x86-64-v3** e na máxima exploração da stack moderna **Linux Kernel 7.0 / CLAP 1.2.x / PipeWire 1.6.x**.

A codebase já demonstra maturidade excepcional: dispatch SIMD estático sem v-tables, FMA fusion, Kahan summation seletivo, huge pages, mlockall, SCHED_FIFO, DAZ/FTZ, PM QoS, e prefetch estratégico. As otimizações abaixo representam o refinamento final — ganhos marginais porém acumulativos no hot path do DSP, onde cada ciclo conta contra o deadline de 1.33 ms (64 amostras @ 48 kHz).

**Princípio norteador:** cada finding deve ser validado pelo par `quality-dashboard.sh` + `tests-performance-regression.sh` antes e depois da implementação. O ESR vs NAMcore e o ESR vs f64-oracle são os guardiões invioláveis.

---

## Categoria A — Otimizações SIMD no Hot-Path (x86-64-v3)

### Finding P-1: `store_16_accums` / `store_8_accums` usam stores escalares em vez de SIMD stores

**Severidade:** Alta (hot-path WaveNet Conv1D, chamado por canal de saída por frame por camada)
**Arquivo:** `src/models/wavenet/conv_input.rs:106-174`
**Paridade:** Sem impacto (mesma matemática, apenas formato de store)

**Diagnóstico:**

As funções `store_16_accums` e `store_8_accums` armazenam os resultados do dot product de volta no buffer de saída usando loops escalares elemento-a-elemento:

```rust
// store_16_accums — linha 161-174
pub(crate) unsafe fn store_16_accums(out: &mut [f32], out_c: usize, r: [f32; 16], out_n: usize) {
    unsafe { *out.get_unchecked_mut(out_c) = r[0] };
    if out_n.is_multiple_of(16) || out_c + 15 < out_n {
        for i in 1..16 {  // ← 15 stores escalares!
            unsafe { *out.get_unchecked_mut(out_c + i) = r[i]; }
        }
    }
    // ... fallback com branches por elemento
}
```

O array `[f32; 16]` está no stack e poderia ser armazenado com **2× `_mm256_storeu_ps`** (256 bits = 8 floats). Similarmente, `[f32; 8]` poderia usar **1× `_mm256_storeu_ps`**.

**Impacto estimado:**

Para um WaveNet Standard (CH=16, 18 camadas), processando um bloco de 64 amostras:

- `store_16_accums`: 18 camadas × 1 bloco de canais × 64 frames = 1.152 chamadas/bloco
- Cada chamada faz 15 stores escalares (60 bytes) em vez de 2 stores vetoriais (64 bytes)
- Estimativa: ~15 cycles/bloco economizados (assumindo 1 cycle/store scalar vs pipeline de 2 stores SIMD)
- **Ganho acumulado:** ~5-8% do tempo de Conv1D, que é a porção dominante do WaveNet

**Proposta de solução:**

```rust
#[inline(always)]
pub(crate) unsafe fn store_16_accums(out: &mut [f32], out_c: usize, r: [f32; 16], out_n: usize) {
    if out_n.is_multiple_of(16) || out_c + 15 < out_n {
        // Fast path: 2× AVX2 256-bit stores (x86-64-v3 garantido pelo target-cpu)
        unsafe {
            let v0 = _mm256_loadu_ps(r.as_ptr());
            let v1 = _mm256_loadu_ps(r.as_ptr().add(8));
            _mm256_storeu_ps(out.as_mut_ptr().add(out_c), v0);
            _mm256_storeu_ps(out.as_mut_ptr().add(out_c + 8), v1);
        }
    } else {
        // Scalar fallback para out_n não-múltiplo de 16 (raro)
        for i in 0..16.min(out_n.saturating_sub(out_c)) {
            unsafe { *out.get_unchecked_mut(out_c + i) = r[i]; }
        }
    }
}
```

Aplicar o mesmo padrão para `store_8_accums` (1× `_mm256_storeu_ps`) e considerar `store_4_accums` (1× `_mm_storeu_ps` — 128-bit, pois 4 floats = 16 bytes).

**Risco:** Nenhum — o resultado numérico é idêntico (store não altera valores). Apenas muda a largura de banda do store.

**Validação obrigatória:**

1. `utils/tests-performance-regression.sh --save` (baseline antes)
2. Implementar
3. `utils/quality-dashboard.sh` (ESR vs NAMcore inalterado)
4. `utils/tests-performance-regression.sh --check` (regression gate deve passar)

---

### Finding P-2: Branch `activation_precision()` dentro do loop de gates LSTM

**Severidade:** Média (hot-path LSTM, branch + atomic load por chunk de 8 células)
**Arquivo:** `src/math/lstm/gates.rs:42, 78, 125, 179`
**Paridade:** Sem impacto (mesmo código, apenas reordenação de branch)

**Diagnóstico:**

Em `fused_lstm_gates_avx2` (linha 35-63), a verificação `activation_precision() == ActivationPrecision::HighFidelity` é executada **dentro** da função que é chamada a cada chunk de 8 células no loop `while j + 8 <= hidden_size` (linha 111-124 de `fused_lstm_gates_dyn_avx2`):

```rust
// gates.rs:35-63 — chamada DENTRO do loop while j + 8 <= hidden_size
pub unsafe fn fused_lstm_gates_avx2(gf, gi, gg, go, cs) -> (__m256, __m256) {
    if activation_precision() == ActivationPrecision::HighFidelity {  // ← atomic load + branch por chunk
        // ... path HF
    } else {
        // ... path Standard
    }
}
```

Para um LSTM 1x16 (BossLSTM), isso significa 2 atomic loads (`Ordering::Relaxed`) + 2 branches por amostra por camada. O branch predictor deve prever corretamente após o primeiro bloco (o modo não muda mid-process), mas o atomic load ainda consome 1-2 cycles (cache line da variável global `ACTIVATION_MODE`).

Adicionalmente, `fused_lstm_gates_dyn_avx2` faz uma **segunda** verificação redundante na linha 125 para o tail escalar, duplicando o padrão.

**Proposta de solução:**

Hoistar a verificação para fora do loop, bifurcando o loop inteiro:

```rust
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_lstm_gates_dyn_avx2(
    gates: &mut [f32],
    cell_state: &mut [f32],
    hidden_state: &mut [f32],
    hidden_size: usize,
) {
    let is_hf = activation_precision() == ActivationPrecision::HighFidelity;  // 1× load

    let mut j = 0;
    while j + 8 <= hidden_size {
        let gi = _mm256_loadu_ps(gates.as_ptr().add(j));
        let gf = _mm256_loadu_ps(gates.as_ptr().add(j + hidden_size));
        let gg = _mm256_loadu_ps(gates.as_ptr().add(j + 2 * hidden_size));
        let go = _mm256_loadu_ps(gates.as_ptr().add(j + 3 * hidden_size));
        let cs = _mm256_loadu_ps(cell_state.as_ptr().add(j));

        let (new_cs, hidden) = if is_hf {
            fused_lstm_gates_avx2_hf(gf, gi, gg, go, cs)
        } else {
            fused_lstm_gates_avx2_std(gf, gi, gg, go, cs)
        };
        // ... store
        j += 8;
    }
    // ... tail escalar (também bifurcado por is_hf)
}
```

Isso elimina N-1 atomic loads (onde N = hidden_size/8) e permite que o compilador otimize cada path independentemente.

**Risco:** Nenhum — o comportamento é idêntico, apenas a estrutura de controle muda.

**Validação:** `utils/quality-dashboard.sh` (ESR LSTM inalterado) + `utils/tests-performance-regression.sh --check`.

---

### Finding P-3: Conversão `frac` via f64 no resampler polyphase

**Severidade:** Média-Baixa (resampler hot-path, ~3-5 cycles por amostra de saída)
**Arquivo:** `src/dsp/resampler.rs:292, 368`
**Paridade:** Requer validação cuidadosa — mudança de precisão na interpolação fracional

**Diagnóstico:**

O resampler polyphase computa o fator de interpolação fracional `frac` (usado na interpolação linear entre fases adjacentes) via conversão para f64:

```rust
// resampler.rs:290-292
let phase_idx = (self.phase_accum >> 40) as usize;
const FRAC_MASK: u64 = (1u64 << 40) - 1;
let frac_bits = self.phase_accum & FRAC_MASK;
let frac = (frac_bits as i64 as f64 * (1.0 / (1u64 << 40) as f64)) as f32;  // ← f64!
```

Isso gera a sequência: `vcvtqq2pd` (int64→f64) + `vmulsd` (f64 mul) + `vcvtsd2ss` (f64→f32) ≈ 3-5 cycles por amostra de saída. Para um bloco de 64 amostras a 48 kHz com conversão 44.1→48 kHz (~70 amostras de saída), são ~210-350 cycles.

A precisão f64 é justificada porque o formato fixed-point 24.40 tem 40 bits fracionais, excedendo os 23 bits de mantissa do f32. No entanto, `frac` é usado apenas na interpolação linear `y0 + frac * (y1 - y0)`, que já é uma aproximação. A questão é se a perda de precisão (2^-23 vs 2^-40) introduziria regressão mensurável no ESR.

**Análise crítica:**

O `frac` ∈ [0.0, 1.0) é multiplicado por `(y1 - y0)`, que são amostras de áudio em [-1, 1]. O erro máximo introduzido pela conversão f32 seria `2^-23 × 2.0 ≈ 2.4e-7`, que está **4 ordens de magnitude abaixo** do piso de ESR do resampler polyphase (que é dominado pela resposta do filtro FIR, não pela interpolação linear entre fases). O ESR vs NAMcore atual para o resampler é tipicamente < 1e-10.

**Proposta de solução (conservadora):**

Manter a versão f64 como padrão, mas adicionar uma versão f32 otimizada e validar empiricamente:

```rust
// Opção A: conversão direta u64→f32 (vcvtqq2ps, ~2 cycles)
// ATENÇÃO: NÃO usar `frac_bits as u32` — frac_bits tem até 40 bits
// (mascarado com FRAC_MASK = (1<<40)-1), e `as u32` truncaria os 8 bits
// superiores, produzindo frac incorreto e regressão de paridade.
let frac = (frac_bits as f32) * (1.0f32 / (1u64 << 40) as f32);
```

```rust
// Opção B: shift para caber em u32, perdendo 8 bits de precisão (vcvtdq2ps, ~1 cycle)
// Mais rápido mas com precisão reduzida (2^-32 vs 2^-40).
let frac = ((frac_bits >> 8) as u32 as f32) * (1.0f32 / (1u32 << 32) as f32);
```

**Risco:** MÉDIO — requer validação completa do quality dashboard. Se o ESR do resampler aumentar além do tolerance, reverter. O risco é baixo (erro ~2.4e-7 vs ESR atual < 1e-10), mas a determinação só pode ser empírica.

**Validação obrigatória:**

1. `utils/quality-dashboard.sh --fidelity-only` antes e depois
2. Comparar ESR vs NAMcore para todos os modelos que usam resampler (taxas ≠ 48 kHz)
3. `utils/tests-performance-regression.sh --save` se aprovado

---

### Finding P-4: Filtro half-band do oversampler usa convolução escalar

**Severidade:** Média (apenas quando oversampling 2×/4× está ativo)
**Arquivo:** `src/dsp/oversample.rs:202-271`
**Paridade:** Sem impacto (mesmo algoritmo, apenas vetorização)

**Diagnóstico:**

O motor de oversampling half-band usa loops escalares para a convolução FIR:

```rust
// oversample.rs:218-223 (upsample)
let mut odd_out = 0.0f32;
for j in 0..HB_ODD_COUNT {  // HB_ODD_COUNT = 12
    let d = j;
    let idx = (pos + n - d) % n;
    odd_out += unsafe { coeffs.get_unchecked(j) * self.up_ring.get_unchecked(idx) };
}
```

Com 12 coeficientes não-nulos (taps ímpares do half-band), isso são 12 MACs escalares por amostra. Com AVX2, seria 1 FMA de 8-wide + 4-wide remainder (2 iterações).

O problema adicional é o **indexação modular** `(pos + n - d) % n` que impede loads contíguos. No entanto, como `n = HB_DELAY = 12` e `d` varia de 0 a 11, os índices são `(pos + 12 - d) % 12`, que é uma sequência decrescente com wrap. Isso poderia ser pré-computado ou reorganizado.

**Impacto:**

Quando oversampling 2× está ativo (opcional, desligado por padrão):

- Upsample: 64 amostras × 12 MACs = 768 MACs escalares → ~96 FMA ops com AVX2
- Downsample: ~70 amostras × 12 MACs = ~840 MACs escalares → ~105 FMA ops
- Total por canal por direção: ~1.612 MACs escalares vs ~201 FMA ops
- Estimativa de ganho: ~4-6× na porção do filtro (mas o filtro é uma fração pequena do custo total de oversampling, que é dominado pela inferência neural 2×/4×)

**Proposta de solução:**

Reorganizar o ring buffer para permitir acesso contíguo (similar ao `DelayLine` do resampler com double-buffer) e usar FMA vetorial:

```rust
fn upsample(&mut self, input: &[f32], output: &mut [f32]) -> usize {
    let coeffs = &self.up_filter.coeffs;
    let center = self.up_center;
    let n = HB_DELAY;

    for (i, &x) in input.iter().enumerate() {
        // ... push para ring buffer (double-write para contiguidade)

        let center_idx = (pos + n - HB_DELAY / 2) % n;
        let even_out = self.up_ring[center_idx] * center;

        // Convolution com AVX2: carregar 8 coeffs + 8 samples (contíguos via double-buffer)
        let mut odd_out = unsafe {
            let c = _mm256_loadu_ps(coeffs.as_ptr());        // 8 coeffs
            let s = _mm256_loadu_ps(window_ptr);              // 8 samples contíguos
            let acc = _mm256_fmadd_ps(c, s, _mm256_setzero_ps());
            // Remainder: 4 taps
            let c4 = _mm_loadu_ps(coeffs.as_ptr().add(8));
            let s4 = _mm_loadu_ps(window_ptr.add(8));
            let acc4 = _mm_fmadd_ps(c4, s4, _mm256_castps256_ps128(acc));
            hsum_avx2(_mm256_insertf128_ps(acc, acc4, 1))
        };
        // ...
    }
}
```

**Risco:** Baixo — requer reorganização do ring buffer para double-buffer (já usado no resampler), mas a matemática é idêntica.

**Nota:** Por ser uma feature opcional (Off por padrão), este finding tem prioridade menor que P-1 e P-2. Implementar após validar os findings de maior impacto.

---

## Categoria B — Guidance do Compilador e Geração de Código

### Finding P-5: Campo `prefetch_fn` morto no Conv1d (dead code em produção)

**Severidade:** Baixa (não afeta hot-path, mas adiciona overhead de carga e complexidade)
**Arquivo:** `src/models/wavenet/conv1d.rs:33`, e ~163 referências em testes/builders
**Paridade:** Sem impacto

**Diagnóstico:**

O campo `prefetch_fn: PrefetchFn` em `Conv1d` é marcado `#[allow(dead_code)]` com o comentário: "No longer used at runtime — dispatch is now static via `self.dilation >= 128`". No entanto:

1. O campo ainda é armazenado em cada instância de Conv1d (8 bytes por camada)
2. É inicializado em todos os builders e testes com um branch `if dilation >= 128 { ... } else { ... }`
3. Existem **163 referências** a `prefetch_strategy_2stage`/`prefetch_strategy_simple` espalhadas pela codebase, a maioria apenas para satisfazer a inicialização do campo

Isso adiciona:

- Overhead de memória (8 bytes × número de camadas Conv1d)
- Overhead de inicialização (branch + store durante carregamento do modelo)
- Carga cognitiva (desenvolvedores precisam saber que o campo existe mas não é usado)

**Proposta de solução:**

1. Remover o campo `prefetch_fn` do struct `Conv1d`
2. Remover todas as inicializações `prefetch_fn: ...` dos builders e testes
3. As funções `prefetch_strategy_simple` e `prefetch_strategy_2stage` permanecem (são chamadas diretamente via `self.dilation >= 128` no hot-path)

**Risco:** Nenhum — é remoção de dead code.

---

### Finding P-6: Resampler usa function pointers indiretos, contradizendo a arquitetura

**Severidade:** Baixa (1 indirect call por bloco de áudio, ~750 calls/s a 48 kHz/64)
**Arquivo:** `src/dsp/resampler.rs:121-124, 133-135`
**Paridade:** Sem impacto

**Diagnóstico:**

A documentação de arquitetura (`docs/architecture.md:55`) declara: "no function pointers exist anywhere in the dispatch path". No entanto, o `ResamplerCore` cacheia function pointers:

```rust
// resampler.rs:121-124
process_stereo: ProcessFn,   // fn(&mut ResamplerCore, &[f32], &[f32], &mut [f32], &mut [f32]) -> usize
process_mono: ProcessMonoFn,
```

Estes são usados no hot-path:

```rust
// resampler.rs:587
(core.process_stereo)(core, in_l, in_r, out_l, out_r)  // ← indirect call
```

O impacto é mínimo (1 indirect call por `process_input`/`process_output` por bloco, bem predito pelo BTB após o primeiro bloco), mas o indirect call impede que o compilador inline o kernel do resampler no pipeline, perdendo otimizações cross-function.

**Análise crítica do benefício real:**

O resampler é chamado **uma vez por bloco** (não por amostra), e o kernel interno já é pesado (64+ taps de convolução). O overhead do indirect call (~1-2 cycles com BTB hit) é negligible comparado ao custo do kernel (~500+ cycles). A inconsistência documental é mais problemática que o overhead.

**Proposta de solução (opcional):**

Se a consistência arquitetural for importante, refatorar para usar o mesmo padrão de monomorphização do pipeline:

```rust
impl NamResampler {
    #[inline(always)]
    pub fn process_input(&mut self, in_l: &[f32], in_r: &[f32], out_l: &mut [f32], out_r: &mut [f32]) -> usize {
        let Some(ref mut core) = self.inner else { /* bypass copy */ };
        // SAFETY: ISA verified at construction time
        unsafe {
            match effective_instruction_set() {
                InstructionSet::Avx2 => core.process_internal::<Avx2Math>(in_l, in_r, out_l, out_r),
                InstructionSet::Avx512 => core.process_internal::<Avx512Math>(in_l, in_r, out_l, out_r),
                InstructionSet::Avx512VnniBf16 => core.process_internal::<Avx512VnniBf16Math>(in_l, in_r, out_l, out_r),
            }
        }
    }
}
```

**Risco:** Baixo. O `effective_instruction_set()` já é uma atomic load relaxed + match, idêntica ao padrão usado no `capture_dsp_pipeline`. Alternativamente, manter os function pointers e apenas atualizar a documentação.

**Recomendação:** Prioridade baixa. O ganho de performance é marginal. Focar em P-1 e P-2 primeiro.

---

## Categoria C — Stack Moderna do Kernel Linux 7.0

### Finding P-7: Ausência de `MADV_COLLAPSE` para promoção síncrona de THP

**Severidade:** Média (elimina jitter de khugepaged em buffers RT-críticos)
**Arquivo:** `src/math/common/huge_alloc.rs:117`
**Paridade:** Sem impacto

**Diagnóstico:**

O alocador de huge pages usa `madvise(MADV_HUGEPAGE)` (linha 117), que é apenas uma **dica** — o kernel pode ou não promover as páginas a huge pages, e a promoção ocorre assincronamente via `khugepaged`. Este daemon de kernel roda em background e pode introduzir latências não determinísticas quando decide colapsar páginas durante o processamento DSP.

O Linux 6.1+ introduziu `MADV_COLLAPSE`, que **sincronamente** colapsa páginas 4K adjacentes em uma huge page de 2MB. Para buffers RT-críticos (pesos de modelos grandes), a promoção síncrona no momento do carregamento elimina o risco de khugepaged interferir durante o processamento.

**Análise de aplicabilidade ao NAM-rs:**

NAM-rs já desabilita THP via `prctl(PR_SET_THP_DISABLE, 1)` no processo inteiro (`rt_setup/thread.rs:29`), o que afeta páginas transparentes. No entanto, `MADV_HUGEPAGE` + `MADV_COLLAPSE` operam em mapeamentos específicos (via mmap), não afetados pelo `PR_SET_THP_DISABLE` global para mapeamentos explicitamente marcados. A interação é:

- `PR_SET_THP_DISABLE`: desabilita THP **implícito** (vmalloc, brk)
- `madvise(MADV_HUGEPAGE)`: solicita THP **explícito** para o mmap
- `MADV_COLLAPSE`: força o colapso **síncrono**

O `PR_SET_THP_DISABLE` não bloqueia `MADV_COLLAPSE` — este último é uma operação explicita de colapso, não uma habilitação de THP automático.

**Proposta de solução:**

```rust
// huge_alloc.rs — após o madvise(MADV_HUGEPAGE) na estratégia 2
let _madvise_rc = unsafe { libc::madvise(ptr, thp_size, libc::MADV_HUGEPAGE) };

// Promoção síncrona (Linux 6.1+). Best-effort: falha silenciosa se
// o kernel não suportar ou se huge pages não estiverem disponíveis.
#[cfg(target_os = "linux")]
{
    const MADV_COLLAPSE: libc::c_int = 25;  // Definido em Linux 6.1+
    let _collapse_rc = unsafe { libc::madvise(ptr, thp_size, MADV_COLLAPSE) };
    // Falha é não-fatal: as páginas podem ainda ser promovidas posteriormente
    // pelo khugepaged, ou permanecer como 4K (funcional, apenas menos eficiente).
}
```

**Risco:** Nenhum — `MADV_COLLAPSE` é best-effort. Se o kernel não suportar (< 6.1), retorna `EINVAL` que é ignorado. Se as huge pages não estiverem disponíveis, retorna `ENOMEM` que também é ignorado.

**Benefício realístico:** Elimina o pior caso de khugepaged introduzindo latência durante o processamento DSP. O impacto é mais perceptível em sistemas com muitos modelos carregados ou após hot-swap de modelo, quando novos buffers grandes são alocados.

---

### Finding P-8: Ausência de `MFD_NOEXEC_SEAL` em `memfd_create`

**Severidade:** Baixa (hardening, não performance direta)
**Arquivo:** `src/math/common/huge_alloc.rs:187, 203`
**Paridade:** Sem impacto

**Diagnóstico:**

A função `create_backing_fd` usa `memfd_create` com `MFD_CLOEXEC` mas sem `MFD_NOEXEC_SEAL` (Linux 6.3+). Este flag sela o file descriptor contra `exec`, impedindo que o memfd seja tornado executável. Embora NAM-rs não execute código destes buffers, o flag é importante para:

1. **Conformidade com políticas de segurança:** systemd's `MemoryDenyWriteExecute=yes` e AppArmor profiles podem bloquear memfds sem `MFD_NOEXEC_SEAL`
2. **Prevenção defense-in-depth:** um bug de use-after-free não poderia escalar para execução de código via o memfd
3. **Compatibilidade futura:** distribuições estão migrando para exigir `MFD_NOEXEC_SEAL` por padrão

**Proposta de solução:**

```rust
const MFD_HUGETLB: libc::c_uint = 0x0004;
const MFD_NOEXEC_SEAL: libc::c_uint = 0x0008;  // Linux 6.3+

// Para memfd com huge pages:
let fd = unsafe {
    libc::memfd_create(
        c"nam_huge_buf".as_ptr(),
        libc::MFD_CLOEXEC | MFD_HUGETLB | MFD_NOEXEC_SEAL,
    )
};
// Se falhar com EINVAL (kernel < 6.3), fallback sem MFD_NOEXEC_SEAL:
if fd == -1 {
    let fd = unsafe {
        libc::memfd_create(c"nam_huge_buf".as_ptr(), libc::MFD_CLOEXEC | MFD_HUGETLB)
    };
    // ...
}
```

Aplicar o mesmo padrão (com fallback) para o memfd regular (linha 203).

**Risco:** Nenhum — o fallback garante compatibilidade com kernels < 6.3.

---

### Finding P-9: Análise da re-aplicação periódica de `set_daz_ftz()`

**Severidade:** Informativa (overhead mínimo, justificativa documentada)
**Arquivo:** `src/standalone/pw_host/rt_callback/process.rs:95-99`, `src/clap/processor/mod.rs:330-334`
**Paridade:** Sem impacto

**Diagnóstico:**

O callback RT re-aplica DAZ/FTZ no MXCSR a cada 1024 frames (`0x3FF`). A justificativa no CLAP processor (linha 326-328) é sólida: "hosts may reset MXCSR after callbacks (e.g. during GUI repaints or parameter flushes from another thread)".

**Análise de custo:**

- `stmxcsr [rsp]` — 1 cycle (não-serializing em x86-64-v3)
- `or eax, 0x8040` — 1 cycle
- `ldmxcsr [rsp]` — 1 cycle (não-serializing em x86-64-v3)
- Total: ~3 cycles a cada 1024 frames = ~0.003 cycles/frame
- **Overhead: 0.0002%** — completamente negligible

**Recomendação:** Manter como está. A re-aplicação é defense-in-depth justificada e o custo é imperceptível. Não há otimização a fazer aqui.

**Oportunidade de hardening (opcional):** Poderia ser feito um check condicional — comparar o MXCSR atual com o valor esperado e só re-aplicar se diferente. Mas o branch adicional (`cmp + jne`) provavelmente custa mais que o `stmxcsr`+`ldmxcsr` incondicional, então não vale a pena.

---

## Categoria D — Stack CLAP 1.2.x / PipeWire 1.6.x

### Finding P-10: CLAP Render Mode já é utilizado, mas sem auto-switch de activation precision

**Severidade:** Informativa (feature já implementada, oportunidade de refinamento)
**Arquivo:** `src/clap/processor/events.rs:133-164`, `src/math/activations/mod.rs:91-110`
**Paridade:** Sem impacto (comportamento já existe via parâmetro manual)

**Diagnóstico:**

O NAM-rs já implementa corretamente a extensão `clap.render`:

- `RenderMode::Realtime` (padrão): adaptive compute ativo, activation precision = Standard
- `RenderMode::Offline`: adaptive compute forçado para Off, activation precision = HighFidelity

O processador CLAP lê `render_mode` e, em modo Offline, força `adaptive_compute = Off` (events.rs:140-164). No entanto, o **activation precision** é controlado por um parâmetro separado do usuário, não pelo render mode.

**Análise crítica de benefício real:**

A auto-switch de activation precision baseada em render mode seria uma conveniência, mas:

1. O usuário já pode manualmente selecionar HighFidelity para offline
2. O activation precision é uma escolha de fidelidade vs performance que o usuário pode querer controlar independentemente do render mode
3. Forçar HighFidelity em offline sem consentimento do usuário seria surpreendente

**Recomendação:** Manter como está. A separação de concerns (render mode vs activation precision) é mais flexível. Não implementar auto-switch.

---

### Finding P-11: Análise de features do PipeWire 1.6.x e CLAP 1.2.x

**Severidade:** Informativa (análise de gaps)
**Paridade:** Sem impacto

**Diagnóstico da utilização atual:**

O NAM-rs já utiliza corretamente as features críticas:

| Feature                          | Status        | Arquivo                            |
| -------------------------------- | ------------- | ---------------------------------- |
| `PW_STREAM_FLAG_RT_PROCESS`      | ✅ Usado      | `capture/setup.rs:283`             |
| `PW_STREAM_FLAG_MAP_BUFFERS`     | ✅ Usado      | `capture/setup.rs:282`             |
| `PW_STREAM_FLAG_EXCLUSIVE`       | ✅ Usado      | `capture/setup.rs:284`             |
| `node.latency` property          | ✅ Usado      | `capture/setup.rs:64`              |
| `node.group` / `node.link-group` | ✅ Usado      | `capture/setup.rs:58-59`           |
| `F32P` (planar float)            | ✅ Usado      | `capture/setup.rs:272`             |
| CLAP `render` extension          | ✅ Usado      | `clap/extensions/render`           |
| CLAP `params` extension          | ✅ Usado      | `clap/extensions/params`           |
| CLAP `state` / `state-context`   | ✅ Usado      | `clap/extensions/state`            |
| CLAP `preset-discovery`          | ✅ Usado      | `clap/extensions/preset_discovery` |
| CLAP `track-info`                | ✅ Usado      | `clap/plugin/mod.rs:153`           |
| CLAP `remote-controls`           | ✅ Registrado | `Cargo.toml:41`                    |
| CLAP `param-indication`          | ✅ Registrado | `Cargo.toml:41`                    |

**Features do PipeWire 1.6.x analisadas e rejeitadas (filtro crítico):**

1. **`PW_KEY_NODE_RATE` / rate matching dinâmico:** Já tratado via `rate_sync.rs` com rebuild do resampler. A API do PipeWire-rs 0.10 não expõe diretamente o rate match nativo, e o resampler custom é superior (minimum-phase vs linear-phase do PipeWire).

2. **`pw-stream` drive mode customizado:** NAM-rs usa o modo default (`PW_STREAM_FLAG_AUTOCONNECT`), que é correto para um virtual sink. O drive mode customizado seria relevante apenas para casos de clock master, que não se aplica ao NAM-rs (ele é sink, não driver).

3. **BAP (Batch Audio Processing) / `PW_STREAM_FLAG_DRIVER`:** Não aplicável — NAM-rs é consumer, não driver de grafo.

**Features do CLAP 1.2.x analisadas e rejeitadas:**

1. **`clap.note-ports`:** NAM-rs é um processador de áudio, não recebe MIDI. Não aplicável.

2. **`clap.voice-info`:** NAM-rs não é polifônico (processa áudio contínuo, não notas). Não aplicável.

3. **`clap.audio-ports-configuration`:** NAM-rs tem configuração fixa (1-in/1-out ou 2-in/2-out). O benefício de expor múltiplas configurações seria mínimo.

**Conclusão:** NAM-rs já tira proveito de todas as features relevantes do CLAP 1.2.x e PipeWire 1.6.x para o caso de uso realista. Não há gaps acionáveis.

---

## Categoria E — Otimizações Universais x86-64-v3 (não-áudio)

### Finding P-12: `AlignedVec::new` usa loop escalar para preenchimento em vez de `memset` otimizado

**Severidade:** Baixa (cold-path de carregamento, não hot-path DSP)
**Arquivo:** `src/math/common/aligned.rs:107-113`
**Paridade:** Sem impacto

**Diagnóstico:**

```rust
// aligned.rs:107-113
pub fn new(len: usize, default: T) -> Result<Self, NamErrorCode>
where T: Copy,
{
    let mut vec = Self::with_capacity(len)?;
    unsafe {
        for i in 0..len {
            vec.ptr.as_ptr().add(i).write(default);  // ← loop escalar
        }
    }
    vec.len = len;
    Ok(vec)
}
```

Para `T = f32` com `default = 0.0`, isso é um memset de zeros. O compilador (`-Copt-level=3`) geralmente otimiza loops de memset para `memset` da libc, mas o `write()` ponteiro-a-ponteiro pode não ser reconhecido como memset se o compilador não puder provar ausência de overlap.

**Proposta de solução:**

Para o caso comum de `default = 0.0f32` (ou qualquer tipo zero-inicializável), usar `write_bytes` com byte `0u8`:

```rust
pub fn new(len: usize, default: T) -> Result<Self, NamErrorCode>
where T: Copy,
{
    let mut vec = Self::with_capacity(len)?;
    if len > 0 {
        // SAFETY: ptr is valid for cap elements, no overlap (fresh allocation).
        unsafe {
            // Caminho rápido: se default é all-zero-bits (e.g. 0.0f32, 0u32),
            // usar write_bytes (compila para memset da libc).
            if default.is_zero_bit_pattern() {
                std::ptr::write_bytes(vec.ptr.as_ptr(), 0u8, len * std::mem::size_of::<T>());
            } else {
                // Fallback: loop escalar para valores não-zero.
                // Para f32 não-zero, considerar SIMD broadcast + store.
                for i in 0..len {
                    vec.ptr.as_ptr().add(i).write(default);
                }
            }
        }
    }
    vec.len = len;
    Ok(vec)
}
```

Nota: `is_zero_bit_pattern()` não existe na std — seria um trait auxiliar a implementar (ou usar `std::mem::MaybeUninit::zeroed()` para verificar). Alternativamente, especializar via trait `Zeroable` para tipos conhecidos (`f32`, `f64`, `u32`, etc.). A verificação `write_bytes` preenche byte-a-byte, só produzindo resultados corretos se `default` é um padrão de bytes all-zero (e.g., `0.0f32` = `0x00000000`). Para valores não-zero como `1.0f32` = `0x3F800000`, o loop escalar (ou SIMD broadcast) deve ser mantido.

**Recomendação:** Baixa prioridade. O compilador já otimiza a maioria dos casos de `0.0` para memset. Validar com `cargo asm` se o loop está sendo otimizado antes de implementar.

---

### Finding P-13: `AlignedVec::resize` realoca e copia elemento-a-elemento

**Severidade:** Baixa (cold-path, mas pode ser chamado em hot-swap de modelo)
**Arquivo:** `src/math/common/aligned.rs:166-174`
**Paridade:** Sem impacto

**Diagnóstico:**

```rust
// aligned.rs:166-174
unsafe {
    let old_ptr = self.ptr.as_ptr();
    for i in 0..self.len {
        new_vec.ptr.as_ptr().add(i).write(old_ptr.add(i).read());  // ← copy element-by-element
    }
    for i in self.len..new_len {
        new_vec.ptr.as_ptr().add(i).write(default);
    }
}
```

Para `T: Copy`, isso poderia usar `copy_nonoverlapping` (que o compilador otimiza para `memcpy`) para a parte existente, e `write_bytes` apenas se o `default` for zero:

```rust
unsafe {
    std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_vec.ptr.as_ptr(), self.len);
    // Para o preenchimento de novos elementos, manter o loop se default != 0.
    // (write_bytes só é válido para padrões de bytes all-zero.)
    for i in self.len..new_len {
        new_vec.ptr.as_ptr().add(i).write(default);
    }
}
```

**Risco:** Nenhum — `copy_nonoverlapping` é semanticamente equivalente ao loop para `T: Copy`.

**Recomendação:** Baixa prioridade. O ganho é apenas no cold-path de carregamento/troca de modelo.

---

## Categoria F — Assembly Analysis e Code Generation Hints

### Finding P-14: `hsum_avx2` poderia usar `_mm256_reduce_add_ps` em CPUs com AVX-512VL

**Severidade:** Informativa (x86-64-v3 não tem AVX-512)
**Arquivo:** `src/math/common/utility.rs:19-30`

**Diagnóstico:**

A horizontal sum AVX2 usa 7 instruções:

```rust
let hi = _mm256_extractf128_ps(v, 1);      // vextractf128
let lo = _mm256_castps256_ps128(v);         // (no-op)
let s128 = _mm_add_ps(lo, hi);              // addps
let shuf = _mm_movehdup_ps(s128);           // movshdup
let sums = _mm_add_ps(s128, shuf);          // addps
let shuf2 = _mm_movehl_ps(sums, sums);      // movhlps
let r = _mm_add_ss(sums, shuf2);            // addss
_mm_store_ss(&mut out, r);                  // movss
```

No baseline x86-64-v3, não há alternativa mais eficiente — esta é a sequência ótima para AVX2. Em CPUs com AVX-512VL, `_mm256_reduce_add_ps` (usado em `hsum_avx512`) é uma única instrução `vreduceps`, mas isso está fora do escopo do baseline v3.

**Conclusão:** Não há otimização a fazer no baseline x86-64-v3. A implementação atual é ótima.

---

### Finding P-15: Ausência de `#[cold]` em paths de erro do hot-path

**Severidade:** Baixa (code generation hint)
**Arquivo:** Vários arquivos de kernel SIMD
**Paridade:** Sem impacto

**Diâgnóstico:**

Vários kernels SIMD têm paths de fallback escalar que são extremamente raros (apenas quando o tamanho não é múltiplo do vector width). Em produção com modelos reais, estes paths nunca são executados (todos os tamanhos são múltiplos de 8). No entanto, sem `#[cold]`, o compilador pode colocar estes paths inline, poluindo o cache de instrução.

Exemplos:

- `dot_product_avx2` (dot.rs:83-88): tail escalar com Kahan summation
- `complex_mac_overwrite` (avx2_impl.rs:810-817): tail escalar
- `batch_norm_process` (avx2_impl.rs:1011-1019): tail escalar

**Proposta de solução:**

Extrair os tails escalares para funções separadas marcadas com `#[cold]` e `#[inline(never)]`:

```rust
#[cold]
#[inline(never)]
unsafe fn dot_product_avx2_tail(a: &[f32], b: &[f32], i: &mut usize, len: usize, scalar_sum: &mut f32, compensation: &mut f32) {
    while *i < len {
        let term = a[*i] * b[*i];
        (*scalar_sum, *compensation) = kahan_add(*scalar_sum, *compensation, term);
        *i += 1;
    }
}
```

**Análise crítica:** O impacto é marginal. O `#[target_feature]` + `opt-level=3` + `codegen-units=1` já faz o compilador colocar tails fora do hot path na maioria dos casos. Validar com `cargo asm` antes de implementar — se os tails já estão fora do layout, não há o que fazer.

**Recomendação:** Baixa prioridade. Implementar apenas se a análise de assembly mostrar tails inline no hot path.

---

## Resumo de Prioridades

| Finding  | Severidade  | Impacto Estimado                  | Risco de Paridade | Esforço          |
| -------- | ----------- | --------------------------------- | ----------------- | ---------------- |
| **P-1**  | Alta        | ~5-8% do Conv1D                   | Nenhum            | Baixo            |
| **P-2**  | Média       | Elimina N atomic loads/bloco LSTM | Nenhum            | Baixo            |
| **P-3**  | Média-Baixa | ~2-3 cycles/amostra resampler     | MÉDIO (validar)   | Baixo            |
| **P-4**  | Média       | ~4-6× no filtro half-band         | Nenhum            | Médio            |
| **P-5**  | Baixa       | Remoção de dead code              | Nenhum            | Médio (163 refs) |
| **P-6**  | Baixa       | Consistência arquitetural         | Nenhum            | Baixo            |
| **P-7**  | Média       | Elimina jitter khugepaged         | Nenhum            | Baixo            |
| **P-8**  | Baixa       | Hardening de segurança            | Nenhum            | Baixo            |
| **P-9**  | Info        | N/A (manter como está)            | N/A               | N/A              |
| **P-10** | Info        | N/A (manter como está)            | N/A               | N/A              |
| **P-11** | Info        | N/A (sem gaps)                    | N/A               | N/A              |
| **P-12** | Baixa       | Cold-path mais rápido             | Nenhum            | Baixo            |
| **P-13** | Baixa       | Cold-path mais rápido             | Nenhum            | Baixo            |
| **P-14** | Info        | N/A (ótimo para v3)               | N/A               | N/A              |
| **P-15** | Baixa       | Code gen hint                     | Nenhum            | Baixo            |

---

## Épicos

### Épico E-PERF-1: Otimização SIMD do Hot-Path WaveNet/LSTM (x86-64-v3)

**Findings:** P-1, P-2
**Objetivo:** Eliminar stores escalares e branches redundantes nos kernels SIMD mais executados do engine de inferência.
**Risco de paridade:** Nenhum — mudanças são puramente estruturais (mesma matemática).
**Validação:** `utils/quality-dashboard.sh` + `utils/tests-performance-regression.sh --save` antes e `--check` depois.
**Sequência:** P-1 primeiro (maior impacto, menor risco), P-2 segundo.

### Épico E-PERF-2: Otimização do Resampler e Oversampler

**Findings:** P-3, P-4
**Objetivo:** Reduzir overhead do resampler polyphase e vetorizar o filtro half-band do oversampler.
**Risco de paridade:** P-3 = MÉDIO (requer validação de ESR), P-4 = Nenhum.
**Validação:** P-3 requer execução completa do quality dashboard com modelos em taxas ≠ 48 kHz. P-4 requer benchmark do oversampling ativo.
**Sequência:** P-4 primeiro (sem risco de paridade), P-3 segundo (após validação empírica).

### Épico E-PERF-3: Modernização da Stack do Kernel Linux

**Findings:** P-7, P-8
**Objetivo:** Adotar `MADV_COLLAPSE` (Linux 6.1+) e `MFD_NOEXEC_SEAL` (Linux 6.3+) para determinismo de huge pages e hardening de segurança.
**Risco de paridade:** Nenhum.
**Validação:** Testes funcionais (carregamento de modelo, hot-swap) + `utils/tests-quick.sh`.
**Sequência:** P-7 primeiro (impacto RT), P-8 segundo (hardening).

### Épico E-PERF-4: Limpeza de Dead Code e Consistência Arquitetural

**Findings:** P-5, P-6
**Objetivo:** Remover campo `prefetch_fn` morto e alinhar resampler com o padrão de dispatch documentado.
**Risco de paridade:** Nenhum.
**Validação:** `utils/tests-quick.sh` + `utils/lints.sh`.
**Sequência:** P-5 primeiro (mais amplo, 163 referências), P-6 opcional.

### Épico E-PERF-5: Otimização de Cold-Path e Code Generation

**Findings:** P-12, P-13, P-15
**Objetivo:** Otimizar alocação/cópia de AlignedVec e adicionar hints de code generation nos tails escalares.
**Risco de paridade:** Nenhum.
**Validação:** `utils/tests-quick.sh` + análise de assembly (`cargo asm`).
**Sequência:** Validar com `cargo asm` primeiro — se o compilador já otimiza, cancelar finding.

---

## Itens Rejeitados (não implementar)

Os seguintes itens foram analisados e **rejeitados** por não trazerem benefício palpável ao caso de uso realista do NAM-rs:

- **P-9** (set_daz_ftz periódico): Overhead de 0.0002%, justificativa documentada sólida. Manter como está.
- **P-10** (auto-switch activation precision por render mode): Separação de concerns atual é mais flexível.
- **P-11** (gaps CLAP/PipeWire): NAM-rs já utiliza todas as features relevantes. Sem gaps acionáveis.
- **P-14** (hsum_avx2 via AVX-512): Fora do escopo do baseline x86-64-v3. Implementação atual é ótima para v3.
- **Aligned loads vs unaligned loads**: Em x86-64-v3 (Haswell+), `loadu_ps` com dados alinhados tem zero penalty. Não há ganho em usar `load_ps`.
- **SCHED_DEADLINE**: Requer `CAP_SYS_NICE` e reserva de CPU, muito agressivo para um produto distribuído. SCHED_FIFO + isolcpus é suficiente.
- **`util_clamp` / `uclamp`**: Pode interferir com o gerenciamento de frequência do kernel. PM QoS (`/dev/cpu_dma_latency`) já resolve o problema de C-States.
