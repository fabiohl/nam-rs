<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Performance (perfil "Master of Performance")

> Auditoria conduzida pela skill `revisor-auditor`, com foco **exclusivo** na role
> _Master of Performance_. As evidências vêm de dois artefatos:
>
> 1. `testes.log` — execução completa das suítes `tests-quick.sh` + `tests-long.sh`
>    (inclui os benchmarks Criterion da Fase 6).
> 2. `target/dsp_hotpath.asm` — `perf annotate --stdio` gerado por
>    `utils/build-release.sh` (Fase 4 BOLT) perfilando o modelo real
>    `BossWN-standard.nam` via PipeWire.

---

## 1. Base de evidências (o que os artefatos revelam)

### 1.1. `testes.log` — números de benchmark (Fase 6, linhas 5143-5148)

| Benchmark                               | Tempo         | Observação                                  |
| --------------------------------------- | ------------- | ------------------------------------------- |
| `Long_WaveNet_Standard_CH16_4096samp`   | **3.6214 ms** | **Hot path dominante** (≈884 ns/amostra)    |
| `Prewarm_A2Full_CH8_2048samp`           | 2.9729 ms     | caminho de prewarm (fora do hot path RT)    |
| `Prewarm_A2Lite_CH3_2048samp`           | 2.4375 ms     | caminho de prewarm                          |
| `Long_Run_LSTM/Long_LSTM_2x16_4096samp` | 808.06 µs     | **4,5× mais barato que o WaveNet Standard** |
| `LSTM_2x24_Comparison/Scalar_Baseline`  | 582.87 µs     | baseline escalar (referência)               |

**Insight nas entrelinhas:** o WaveNet Standard CH16 é, isoladamente, o consumidor
de CPU mais caro do projeto — 4,5× o LSTM 2x16. O `build-release.sh` (linhas 260-269)
prioriza exatamente `BossWN-standard.nam` para o profiling BOLT, logo
**`dsp_hotpath.asm` É o hot path do WaveNet Standard**. Toda otimização aqui tem o
maior retorno marginal. A 48 kHz / buffer de 64 amostras (orçamento de 1,333 ms),
esse modelo gasta ≈56,6 µs/bloco (≈4,25% do core); o objetivo é derrubar essa fração
para abrir folga a buffers menores (latência) e CPUs mais fracas.

Outros sinais do log (sem regressões — todas as suítes passaram):

- O perfil foi coletado com apenas **1097 amostras** (`cycles:u`, run de ~3 s) —
  resolução estatística baixa; as conclusões abaixo priorizam **evidência estrutural**
  (padrões de instrução) sobre percentuais isolados.
- `lib_pipeline_block_proptest` 115,9 s e `resampler_heap_audit` 93,3 s são os testes
  mais lentos — não é hot path de áudio, mas indicam custo de compilação/execução de QA.

### 1.2. `target/dsp_hotpath.asm` — anatomia do binário perfilado

`WaveNetModel::process` aparece como **um único símbolo monolítico** (`0xca500`,
~60 KB de código, linhas 3-10076 do dump): tudo foi fundido por `#[inline(always)]`.
Os kernels GEMV/GEMM 1×1 ficaram **fora de linha** (`gemv_with_bias_f32_avx2`,
`gemv_no_bias_f32_avx2`, `fused_gemm_residual_batch_f32_avx2`).

Histograma de opcodes **dentro do monólito** WaveNet (linhas 3-10076):

| Opcode                | Contagem | Natureza                                      |
| --------------------- | -------- | --------------------------------------------- |
| `movq`                | **1918** | movimentação escalar / spills de registrador  |
| `vmovups`             | 1165     | loads/stores SIMD                             |
| `vmovss`              | **927**  | **loads/stores ESCALARES de f32**             |
| `vaddss`              | **504**  | **somas ESCALARES de f32**                    |
| `vbroadcastss`        | 470      | broadcasts                                    |
| `je`                  | 480      | desvios                                       |
| `vaddps`              | 298      | somas SIMD (reduções de acumulador)           |
| `callq`               | **263**  | **chamadas (em código supostamente inlined)** |
| `vfmadd213/132/231ps` | 474      | **FMAs SIMD empacotados (o trabalho útil)**   |
| `vzeroupper`          | **152**  | **guardas de transição AVX→SSE**              |

No dump inteiro: **325 `vzeroupper`**, **223 `vcvt`**, **12 `vdivps`** (todos no
caminho do `tanh`), e **zero `vgather`/`vscatter`** (bom — sem gathers lentos).

**Conclusão de alto nível:** o "SIMD" do WaveNet está dominado por **operações
escalares e movimentação de dados** (504 `vaddss` + 927 `vmovss` + 1918 `movq`),
não pelos 474 FMAs empacotados que de fato calculam a convolução. As próximas
seções rastreiam cada padrão até a linha de código.

---

## 2. Findings

> Severidade: **CRÍTICA** (ganho grande e direto no hot path), **ALTA**,
> **MÉDIA**, **BAIXA**. Cada finding traz: sintoma (asm), causa-raiz (`arquivo:linha`),
> impacto, proposta de solução e cuidados de validação (paridade ESR / RT-safety).

---

### F1 — [CRÍTICA] Acumulação K-tap da convolução faz _round-trip_ por arrays `[f32; N]` + somas escalares

**Sintoma (asm).** No monólito: 504 `vaddss` + 927 `vmovss` contra apenas 474 FMAs
empacotados. Trechos densos de `vmovss`/`vaddss` cercam cada chamada de dot-product.

**Causa-raiz.** O kernel `dot_product_16x_f32_dual_avx2`
(`src/math/gemm/dot_16x/dot_f32_avx2.rs:123`) constrói **16 acumuladores YMM**, faz a
**redução em árvore (12 `vaddps`) e armazena em `([f32;16],[f32;16])`** a cada chamada
(linhas 201-221). O chamador, em `src/models/wavenet/conv1d_dual.rs:88-106`, então:

1. Inicializa `r_f0`/`r_f1` via `load_16_accums` (`src/models/wavenet/conv_input.rs:131`)
   — **16 loads escalares** (`vmovss`) elemento a elemento.
2. Para cada tap `k ∈ 0..K`: chama o dot-product (que **reduz + armazena**) e em seguida
   executa `for i in 0..16 { r_f0[i] += t_f0[i]; r_f1[i] += t_f1[i]; }`
   (`conv1d_dual.rs:100-103`) — **32 `vaddss` escalares por tap**.
3. `store_16_accums` grava de volta — **16 stores escalares**.

Pior ainda: o bias+mixin é **escrito escalarmente em memória** em `conv1d_dual.rs:49-57`
e **relido escalarmente** logo a seguir por `load_16_accums`. A matemática FMA é
vetorizada, mas **todas as fronteiras (init, acumulação por tap, store) são escalares**.

**Impacto.** Por dual-frame, por bloco de interleave, por camada: redução redundante
(12-24 `vaddps`) + 4 stores YMM + ~32 reloads escalares + ~32 somas escalares — **×K
taps × ~10 camadas × 2 arrays × 32 dual-frames**. É a origem direta dos 504 `vaddss` /
927 `vmovss` / boa parte dos 1918 `movq`. O gargalo do WaveNet **não são os FMAs**, é o
encanamento escalar dos acumuladores.

**Proposta de solução.**

- Fundir o laço dos K taps **dentro** do kernel SIMD, mantendo os acumuladores em
  registradores YMM por toda a varredura `K × IN`, com **uma única redução e um único
  store** ao final. Assinatura nova, por exemplo:
  `dot_product_16x_f32_dual_accumulate(weights_k: &[&[[f32;16]]; K], taps_f0: &[&[f32;K]], taps_f1, init_f0: __m256x2, init_f1: __m256x2) -> (__m256x2, __m256x2)`,
  ou um kernel que receba os _slices_ de peso/tap dos K taps e itere internamente.
- Usar o **bias+mixin como valor inicial do acumulador** (via `vbroadcastss`/load YMM),
  eliminando o par escrever-em-memória/reler de `conv1d_dual.rs:49-57` + `load_*_accums`.
- Eliminar `load_*_accums`/`store_*_accums` escalares: a I/O de borda deve ser
  `_mm256_loadu_ps`/`storeu_ps` (com _tail_ mascarado para `OUT % 8 != 0`).

**Cuidados / validação.** A ordem de soma muda (associatividade f32) → revalidar
**paridade ESR** contra `NeuralAmpModelerCore` (testes `cpp_parity`, `golden_vectors`)
e os limiares calibrados (`threshold_calibration.rs`). Manter a redução em árvore
4-way para preservar o erro < 2 ULP já documentado. RT-safe (sem heap).

---

### F2 — [CRÍTICA] Prefetch por _ponteiro de função_ no laço da convolução → chamada indireta + `vzeroupper` forçado

**Sintoma (asm).** `d521f: callq *%r13` com **5,30% local**, imediatamente precedido por
`d521c: vzeroupper`. Os símbolos `prefetch_strategy_simple`/`prefetch_strategy_2stage`
aparecem **fora de linha** (não inlined). No dump inteiro: **325 `vzeroupper`** e
**263 `callq`** dentro do monólito.

**Causa-raiz.** `src/models/wavenet/conv1d_dual.rs:72` invoca `(self.prefetch_fn)(...)`
— um **ponteiro de função** `PrefetchFn` (`conv1d.rs:31`) — **K vezes por dual-frame**.
As funções alvo (`src/math/common/ops.rs:146` e `:163`) **não têm `#[inline]`** e, por
serem chamadas via ponteiro, **não podem ser inlinadas**. O `prefetch_strategy_simple`
inteiro é literalmente **um único `_mm_prefetch`** (`ops.rs:155`).

Como o alvo não usa registradores AVX, o compilador é obrigado a emitir `vzeroupper`
**antes de cada chamada** para evitar a penalidade de transição AVX→SSE, além de
_spill/reload_ dos taps em torno da chamada (clobber de registradores caller-saved).

**Impacto.** Para emitir **um** `prefetcht0`, paga-se: chamada indireta (risco de
_misprediction_) + `vzeroupper` + _spills_. Multiplicado por `K × camadas × dual-frames`
por bloco. É overhead puro, sem trabalho útil. (Note o contraste: o prefetch
**no nível do layer-array** — `src/models/wavenet/layer_array.rs:131-134` — já usa
`_mm_prefetch::<_MM_HINT_T0>` **inlinado e correto**. Só a convolução por-tap está errada.)

**Proposta de solução.** Eliminar o ponteiro de função no hot path. Opções, em ordem de
preferência:

1. **Resolver a estratégia em tempo de compilação.** A escolha depende de
   `dilation >= 128` (`layout.rs:70`), conhecido por instância de camada. Transformar em
   _const generic_ (`const PF: PrefetchKind`) ou inlinar um `match`/`if self.dilation`
   diretamente no kernel — um desvio _data-dependent_ previsível é ordens de magnitude
   mais barato que `callq` indireto + `vzeroupper`.
2. **Marcar `prefetch_strategy_*` como `#[inline(always)]`** e chamá-las por _generic
   dispatch_ (não por ponteiro), permitindo que o `prefetcht0` colapse para 1 instrução.
3. **Remover o prefetch manual por-tap para dilations pequenas.** Com `MirroredBuffer`
   garantindo acesso linear e K=3 taps curtos, o _hardware prefetcher_ provavelmente já
   cobre; medir se o prefetch manual é líquido-negativo nessas camadas.

**Cuidados / validação.** Sem impacto numérico (prefetch não altera resultado) → basta
benchmark Criterion (`Long_WaveNet_Standard_CH16`) antes/depois. Confirmar que o
`#[target_feature]` é preservado para não reintroduzir `vzeroupper`.

---

### F3 — [ALTA] `tanh` de alta fidelidade usa **2 divisões** onde **1** basta

**Sintoma (asm).** Em `d89cf: vdivps` e `d89db: vdivps` (duas divisões), com o consumidor
dependente `d89df: vminps` marcando **7,11% local** — o pipeline **estola na latência da
divisão**. Há 12 `vdivps` no dump, todos no caminho do `tanh`/`sigmoid`.

**Causa-raiz.** `src/math/activations/tanh/high_fidelity.rs:82` (`simd_tanh_poly_avx2`,
usado por `tanh_and_accumulate_block_avx2` — a ativação dominante do WaveNet):

```rust
let exp_x   = simd_exp_poly_avx2(x);
let inv_exp_x = _mm256_div_ps(one, exp_x);   // div #1  → e^-x
let num = _mm256_sub_ps(exp_x, inv_exp_x);
let den = _mm256_add_ps(exp_x, inv_exp_x);
let tanh_val = _mm256_div_ps(num, den);      // div #2
```

`vdivps ymm` tem latência ~11-14 ciclos e **não é pipelined** (throughput ~5). Duas em
série no caminho crítico dominam o custo da ativação.

**Proposta de solução.** Reescrever com identidade algébrica que precisa de **uma só
divisão**, com **precisão idêntica** (≤ 2,4e-7):

```text
tanh(x) = (e^{2x} − 1) / (e^{2x} + 1)        // 1 exp(2x) + 1 div
```

ou, reaproveitando `u = e^x`:

```text
tanh(x) = (u² − 1) / (u² + 1)                 // u² via 1 mul, 1 div
```

Como `x` já é _clamped_ a `[-20, 20]` (`high_fidelity.rs:88`), `u²` no pior caso é
`e^40 ≈ 2,4e17` — dentro do alcance f32 (máx ≈3,4e38), sem overflow. Isso **elimina a
divisão do recíproco `1/e^x`** e o `sub`/`add` extra, **cortando o caminho crítico do
`tanh` ~pela metade** das divisões. Aplicar o mesmo às variantes `_dual` e ao caminho
_gated_ (`simd_tanh_sigmoid_dual_poly_avx2`, `high_fidelity.rs:171`).

**Opção agressiva (orçamento de precisão permitindo):** substituir a divisão restante por
`vrcpps` + 1-2 iterações Newton-Raphson (`vrcpps` ~5 ciclos, pipelined). Avaliar contra
os testes de varredura de precisão (`test_tanh_poly_avx2_sweep`,
`test_tanh_pade_nr2_proptest_100k`).

**Cuidados / validação.** A reformulação `(u²−1)/(u²+1)` é matematicamente idêntica, mas
revalidar varreduras de erro (`high_fidelity_test.rs`) e paridade ESR. AVX-512 (`:310`)
deve receber a mesma reescrita por consistência de fidelidade entre ISAs.

---

### F4 — [ALTA] GEMV degenera em _shapes_ pequenos (`out_len==1`, `in_len==1`, `out_len ∈ {2..7}`)

**Sintoma (asm).**

- `gemv_with_bias_f32_avx2` caminho `out_len == 1`: **redução horizontal serializada de 8
  elementos por frame** — 7× `vaddss` + `vshufps`/`vshufpd`/`vmovshdup`/`vextractf128`
  (linhas 47157-47172; `vshufps $0xff` 13,38%, `vshufpd $0x1` 12,34%, soma de bias 16,26%).
- `gemv_no_bias_f32_avx2` caminho `out_len ∈ {5,6,7}`: cai no **fallback escalar**
  (`scalar_ref::gemv_no_bias_f32_fallback`), que o compilador auto-vetoriza num **festival
  de blends/shuffles** — `vblendps $0x10` 15,12%, `vfmadd213ps xmm` 6,59%, `vblendps $0x80`
  6,11% (linhas 11755-11772), misturando FMAs escalares (`vfmadd213ss`) com montagem
  parcial de vetor.

Resumo do dump: `vshufps` (38×, 22,6% somado) + `vshufpd` (10×, 12,3%) + `vblendps`
(12×, 23,1%) concentram-se nesses kernels.

**Causa-raiz.** `src/math/gemm/gemv/f32_avx2.rs` tem _buckets_ rígidos:
`out_len == 1` (redução horizontal por frame, linhas 45-73), `out_len <= 4` (fallback
escalar, 75-80) e `out_len >= 8` (caminho bom, 82-134). **`out_len ∈ {5,6,7}` não casa com
nenhum** e cai no fallback escalar final (linhas 136-138 / 265-267). E `in_len == 1`
(usado pelo _input mixin_/_rechannel_ do WaveNet) ainda roda toda a maquinaria de 8
acumuladores + árvore de 7 somas para **um único canal de entrada**.

Estes _shapes_ degenerados são **exatamente** os que o WaveNet usa: projeção final do
_head_ `CH→1` (`layer_array.rs:176`, comentário "16 -> 8 or 16 -> 1"), e _input mixin_
`condição(1)→CH` (in_len=1).

**Impacto.** A projeção final do _head_ executa O(num_frames) reduções horizontais
seriais (cadeia de ~28 ciclos cada, ×64 frames). O _input mixin_ paga overhead de
redução de 8 vias para 1 canal. _Shapes_ 5-7 caem no pior caminho escalar.

**Proposta de solução.**

- **Kernel GEMV unificado e robusto a _shape_** baseado no padrão _broadcast-input /
  acumula-através-das-linhas-de-saída_ — que `fused_gemm_residual_batch_f32_avx2` **já faz
  bem** (`vbroadcastss` + `vfmadd231ps`, **sem redução horizontal**). Cobrir todo `out_len`
  com _tail_ mascarado (`out_len % 8`) em vez de fallback escalar.
- **`out_len == 1` com muitos frames:** processar **8 frames por YMM** (broadcast do peso,
  FMA, redução adiada) ou adotar layout _channel-major_ no _head_accum_ para o produto
  ficar contíguo. Elimina a redução horizontal por frame.
- **`in_len == 1`:** caso especial trivial `out = bias + broadcast(in)·w[0..out_len]`.

**Cuidados / validação.** Mudança de ordem de soma → revalidar paridade ESR e
`golden_vectors`. Cobrir `out_len ∈ {1..16}` e `in_len ∈ {1, CH}` com testes de paridade
contra o `scalar_ref` antes de remover o fallback.

---

### F5 — [MÉDIA] Monólito `#[inline(always)]`: frame de pilha ~10 KB, _spills_ e pressão de registrador

**Sintoma (asm).** Prólogo de `WaveNetModel::process` sonda a pilha em
`0x1000 + 0x1000 + 0x998 ≈ 10,4 KB` (linhas 10-14 do dump: `subq $0x1000,%rsp` ×2 +
`subq $0x998`). **1918 `movq`** no monólito (forte indício de _spills_), **152
`vzeroupper`** e **263 `callq`**.

**Causa-raiz.** `#[inline(always)]` agressivo funde array1 + array2 + todas as camadas +
buffers de tap (`[[f32; IN]; K]`) + block buffers num **único frame**
(`model.rs:88`, `layer_array.rs:65`, `layer.rs:28`, `conv1d_dual.rs:30`, etc.). O
excesso de variáveis vivas simultâneas estoura os 16 registradores YMM → _spills_ para a
pilha (os `movq`/`vmovups` de/para `(%rsp)`).

**Impacto.** Pressão de registrador → _spills_ (parte dos 1918 `movq`), _footprint_ de
pilha grande (pressão de cache L1d), pressão de I-cache (~60 KB num símbolo). Algum
_inlining_ é bom (elimina _dispatch_ virtual); **excesso** prejudica.

**Proposta de solução.**

- Tornar a fronteira **por-camada** (ou por-array) um _call boundary_ real: trocar
  `#[inline(always)]` por `#[inline]` em `WaveNetLayer::process_block_internal` e/ou
  `WaveNetLayerArray::process_block_internal`, deixando o compilador decidir. Assim os
  buffers de tap de cada camada não coexistem todos no mesmo frame.
- Reduzir/reusar os arrays de pilha (`in_taps_f0/f1: [[f32; IN]; K]`) — reaproveitar um
  único buffer entre taps quando possível.
- Medir com `perf stat` (ciclos, _stall_ de _frontend_) e Criterion: comparar
  `inline(always)` vs `inline` no laço de camadas. **É um _trade-off_ — exige medição**,
  não mudança cega.

**Cuidados / validação.** Não alterar resultado numérico. Garantir que o _dispatch_
estático SIMD (`dispatch_simd!`) permaneça monomorfizado (sem reintroduzir _vtable_).
Benchmark obrigatório antes/depois — _inlining_ é sensível a microarquitetura.

---

### F6 — [MÉDIA] Overhead na _callback_ de áudio: varredura de slots (~196 KB) + cópias de buffer no thread RT

**Sintoma (asm).** Em `setup_capture_stream::{{closure}}` (a _callback_ PipeWire):
`0x2075e3: movq 0x8(%r13,%rbx),%r14` com **17,67% local**, dentro de um laço que varre uma
estrutura de **0x30100 ≈ 196 KB** em passos de 16 bytes (`addq $0x10,%rbx; cmpq $0x30100,%rbx`)
escrevendo o marcador `$0x4` em cada slot. Também quentes: `__memmove_avx_unaligned_erms`
(`vmovdqu` 28,99% / 24,5%) e `__memset_avx2_unaligned_erms` (65% numa linha).

**Causa-raiz (a investigar — inferência a partir do asm).** O padrão "varrer N slots
fixos e resetar para um marcador" sugere _drain_/reclaim do **SPSC GC** (a política de
desalocação citada no `AGENTS.md`) ou processamento de fila de eventos **a cada
callback**, no thread de áudio. As cópias casam com: _seed_ do _head_accum_
(`layer_array.rs:81`, `copy_from_slice` de `num_frames*CH`) e a coleta de taps por
`copy_from_slice` (`conv1d_dual.rs:68-71`).

**Impacto.** Trabalho não-DSP no thread RT competindo com o orçamento de áudio. Uma
varredura de 196 KB por _callback_ é _cache-unfriendly_ e proporcional ao tamanho da
estrutura, não ao trabalho realmente pendente.

**Proposta de solução.**

- **Localizar e caracterizar** o laço de 196 KB (correlacionar
  `setup_capture_stream::{{closure}}` com `src/standalone/pw_host/capture/` e o módulo
  `common::spsc`). Se for _drain_ de GC/fila: torná-lo **incremental/event-driven**
  (processar só os slots realmente sinalizados, via índice/contador), não varredura total.
- Fundir o _seed copy_ do `head_accum` na inicialização da primeira camada (evitar
  `copy_from_slice` dedicado).
- Garantir que cópias de tamanho-constante (taps `[f32; IN]`) **baixem para `vmovups`** e
  não para `memcpy`/`memmove` da libc (verificar _const propagation_ de `IN`).

**Cuidados / validação.** **RT-safety é crítico aqui** (thread de áudio): nenhuma das
mudanças pode introduzir _heap_, _lock_ ou _syscall_. Validar com `pipeline_soak`,
`a2_heap_audit`/`resampler_heap_audit` (feature `heap-audit`) e o teste de integração
PipeWire. Confirmar ausência de _xruns_ com buffer pequeno.

---

### F7 — [BAIXA] Microarquitetura: `vcvt` e proliferação de `vzeroupper`

**Sintoma.** 223 `vcvt*` e **325 `vzeroupper`** no dump inteiro.

**Causa-raiz.** Os `vzeroupper` são consequência direta de **F2** (e parcialmente F5):
toda chamada para função sem AVX exige a guarda. Os `vcvt` vêm do `exp` polinomial
(`vcvtps2dq` na construção de `2^k`, `high_fidelity.rs:63-64`) e de conversões BF16/F16.

**Proposta de solução.** Em grande parte **resolvido ao corrigir F2** (eliminar chamadas
fora de linha derruba os `vzeroupper`). Para o `exp`: o `vcvtps2dq`/`vpslld`/`vpaddd` é o
método canônico e eficiente — manter. Apenas confirmar, pós-F2, que a contagem de
`vzeroupper` no monólito cai de 152 para ~dezenas.

**Cuidados / validação.** Nenhum risco numérico; medir contagem de instruções
(`perf stat -e instructions`) antes/depois.

---

## 3. Épicos (agrupamento para execução otimizada)

> Sequência pensada para **maximizar ganho com mínimo risco de regressão**: começar pelo
> que tem maior retorno e menor risco numérico, deixando reescritas de kernel (que exigem
> revalidação de paridade ESR pesada) agrupadas para uma única rodada de validação.

### ÉPICO A — "Quick Wins" sem impacto numérico (maior ROI, risco mínimo) [CONCLUÍDO]

Agrupa correções que **não alteram resultados** (só revalidação por benchmark, sem
necessidade de re-tunar limiares ESR):

- **F2** — Eliminar prefetch por ponteiro de função (inline/const-generic).
- **F5** — Reavaliar `#[inline(always)]` → `#[inline]` na fronteira de camada/array.
- **F7** — Verificar queda de `vzeroupper` (consequência de F2).

#### Relatório de Execução e Resultados (2026-06-24)

1. **Implementação do Prefetch Estático (F2):**
   - Substituição das 7 chamadas indiretas via ponteiro de função `(self.prefetch_fn)(...)` por dispatch estático com base na dilatação (`if self.dilation >= 128`) em 5 arquivos (`conv1d.rs`, `conv1d_dual.rs`, `conv1d_dyn.rs`, `conv1d_dyn_dual.rs`, `grouped_conv1d.rs`).
   - Aplicação de `#[inline(always)]` em `prefetch_strategy_simple` e `prefetch_strategy_2stage` em [ops.rs](file:///home/fabio/nam-rs/src/math/common/ops.rs).
   - Manutenção de compatibilidade da API pública de loaders mantendo o campo na struct anotado com `#[allow(dead_code)]`.

2. **Auditoria de Instruções e vzeroupper (F7):**
   - **Monólito WaveNetModel::process** (~59 KB):
     - Instruções `vzeroupper`: redução de **152 para 85 (queda de 44%)**.
     - Chamadas `call` (total): redução de **263 para 198 (queda de 25%)**.
     - Chamadas indiretas: **Zero chamadas indiretas** para prefetch (apenas chamadas para libc `memset`/`memcpy` e `tanhf`).
   - **Biblioteca Compartilhada `libnam_rs.so`**: **Zero `vzeroupper`** em toda a seção `.text`.

3. **Flexibilização do Inlining (F5) e Benchmarks:**
   - Alteração de `#[inline(always)]` para `#[inline]` em `WaveNetLayer::process_block_internal` e `WaveNetLayerArray::process_block_internal` para mitigar pressão sobre a pilha e registradores.
   - **Criterion (`Long_WaveNet_Standard_CH16`):** Mediana de **3.5599 ms** (redução de **−3.62%** com significância estatística p=0.00 contra a baseline histórica de 3.6214 ms).
   - **Métricas do `perf stat` (IPC de 2.94):** 506.96 × 10⁹ instruções, 172.55 × 10⁹ cycles, 242.23 × 10⁶ cache-misses e 3.69 × 10⁹ L1-dcache-load-misses.

A suíte de testes de integridade (`tests-quick.sh`, incluindo 407 testes unitários e 37 de integração) passou sem falhas. O Épico A está concluído com sucesso e entrega um ganho de performance sólido com risco numérico zero.

### ÉPICO B — Reescrita dos kernels de acumulação SIMD (maior ganho absoluto)

O coração do hot path. Exige **uma rodada conjunta de revalidação de paridade ESR**:

- **F1** — Fundir laço K-tap no kernel; acumuladores em YMM; bias/mixin como init;
  borda via `vmovups` (não escalar).
- **F4** — GEMV unificado robusto a _shape_ (padrão broadcast+FMA, _tail_ mascarado;
  casos especiais `out_len==1` por _frame-batching_ e `in_len==1`).

_Validação do épico:_ `cpp_parity`, `cabsim_cpp_parity`, `golden_vectors[_v2]`,
`threshold_calibration`, mais varreduras de precisão. **Crítico/arriscado** — mudanças de
associatividade f32; testar incrementalmente por _shape_.

### ÉPICO C — Ativações de baixa latência

- **F3** — `tanh` com 1 divisão (`(e^{2x}−1)/(e^{2x}+1)`), AVX2 + AVX-512; avaliar
  `vrcpps`+NR para a divisão restante.

_Validação do épico:_ `test_tanh_poly_*_sweep`, `*_proptest_100k`, paridade ESR. Risco
moderado (precisão), mas isolado no módulo de ativações.

### ÉPICO D — Higiene do thread de tempo-real

- **F6** — Caracterizar e tornar incremental a varredura de ~196 KB na _callback_;
  reduzir cópias de buffer no thread RT.

_Validação do épico:_ `pipeline_soak`, `*_heap_audit`, integração PipeWire, ausência de
_xruns_. **RT-safety é mandatória** (zero heap/lock/syscall no hot path).

---

### Matriz de priorização

| Épico | Findings   | Ganho esperado | Risco de regressão | Esforço | Ordem sugerida |
| ----- | ---------- | -------------- | ------------------ | ------- | -------------- |
| A     | F2, F5, F7 | Médio-Alto     | **Baixo**          | Baixo   | **1º**         |
| B     | F1, F4     | **Alto**       | Alto (ESR)         | Alto    | 2º             |
| C     | F3         | Médio          | Médio (precisão)   | Baixo   | 3º             |
| D     | F6         | Médio          | Alto (RT-safety)   | Médio   | 4º             |

> O `TODO-sprints.md` (épicos → sprints → tarefas técnicas atômicas) deve ser gerado pela
> skill `planejador-arquiteto` **somente quando solicitado**, referenciando estes findings.
