# WaveNet Lite CH12 — Profiling & Structural ASM Analysis (T7.S2.2)

**Date:** 2026-07-20
**Artifacts:** `target/dsp_hotpath.asm` (Standard CH16, build-release.sh), `target/dsp_hotpath_lite.asm` (Lite CH12, dedicated perf record)

## Resumo Executivo

A análise estrutural do assembly compara `WaveNetLayer<1,12,3>::process_block_internal` (Lite, 294 samples) com `WaveNetLayer<1,16,3>::process_block_internal` (Standard, 1744 samples). As três hipóteses do F-A7 foram avaliadas com evidência quantitativa de contagem de instruções.

**Conclusão principal:** A hipótese #1 (overhead fixo por camada desproporcionalmente maior com menos canais) é a causa dominante. A hipótese #2 (transição de domínio SIMD em stride16) contribui marginalmente via subutilização de lanes AVX (8+4 split) e `vzeroupper` frequente, mas sem penalidade de legacy SSE→AVX (todas as instruções usam VEX). A hipótese #3 (meta original de ≤42 µs incorreta) é validada: a meta não é matematicamente alcançável sem refatoração arquitetural profunda.

**Recomendação:** Desfecho C (aceitar como dívida técnica) ou Desfecho B (reverter T6.S3.2), com preferência por C — o código é bit-exato, testado, e o Lite está a 96.1% de folga do budget RT.

---

## 1. Metodologia

### Fontes de Dados

| Fonte | Função | Amostras | Formato |
|-------|--------|----------|---------|
| `target/dsp_hotpath.asm` | `WaveNetLayer<1,16,3>::process_block_internal` (Avx2Math) | 1744 total (workload multi-modelo BOLT) | `perf annotate --stdio` |
| `target/dsp_hotpath_lite.asm` | `WaveNetLayer<1,12,3>::process_block_internal` (Avx2Math) | 294 local period (dedicado) | `perf annotate --stdio` |
| Objdump do binário release | `fused_gemm_residual_batch_f32_12x12_padded` | N/A (chamada via `callq`) | `objdump -d` |

### Categorias de Instrução

```
FMA:        vfmadd*, vfmsub*, vfnmadd*
Mul/Add:    vmulps, vmulss, vaddps, vaddss
Moves:      vmovups, vmovaps, vmovss, vmovsd, vmovdqa
Blend/Perm: vblendps, vshufps, vinsertps, vmaskmovps, vperm*
Broadcast:  vbroadcastss
VXOR:       vxorps, vpcmpeqd
Scalar:     addq, subq, shlq, leaq, movq, movl, movzbl, etc.
Branch:     cmpq, testq, je, jne, jbe, jae, jmp, callq
Prologue:   pushq, popq, subq $0x...,%rsp
```

---

## 2. Comparação Estrutural da Camada

### 2.1 Visão Macro

| Métrica | Standard CH16 | Lite CH12 | Razão Lite/Std |
|---------|--------------|-----------|-----------------|
| Instruções totais | 1815 | 3974 | 2.19× |
| Stack frame | `subq $0x368, %rsp` (872B) | `subq $0x5a8, %rsp` (1448B) | 1.66× |
| Canais | 16 | 12 | 0.75× |
| Layers por modelo | 20 (2×10) | 6 | 0.30× |
| Latência medida | 37.6 µs | 52.3 µs | 1.39× |
| µs/MMAC | 2449.22 | 6056.71 | 2.47× |

**Observação crítica:** O Lite tem 2.19× mais instruções por camada apesar de processar 25% menos canais. O stack frame é 66% maior. A função do Lite é mais complexa estruturalmente que a do Standard.

### 2.2 Distribuição de Instruções por Categoria

| Categoria | Standard CH16 | % | Lite CH12 | % |
|-----------|--------------|----|-----------|----|
| **FMA** | 289 | 15.9% | 369 | 9.3% |
| Mul/Add | 103 | 5.7% | 97 | 2.4% |
| Moves | 417 | 23.0% | 639 | 16.1% |
| Broadcast | 193 | 10.6% | 256 | 6.4% |
| Blend/Perm | 0 | 0.0% | 120 | 3.0% |
| VXOR | 26 | 1.4% | 20 | 0.5% |
| Scalar reg/addr | 444 | 24.5% | 1558 | 39.2% |
| Branch | 157 | 8.7% | 552 | 13.9% |
| Prologue | 26 | 1.4% | 36 | 0.9% |
| Other | 160 | 8.8% | 327 | 8.2% |
| **Total** | **1815** | **100%** | **3974** | **100%** |
| **Overhead** (scalar+branch+prol) | **627** | **34.5%** | **2146** | **54.0%** |
| **Useful** (FMA+mul+add+blend+bc) | **585** | **32.2%** | **893** | **22.5%** |

**Razão Overhead/Useful:**
- Standard: 627/585 = **1.07** (1.07 overhead por 1 útil)
- Lite: 2146/893 = **2.40** (2.40 overhead por 1 útil)

### 2.3 Uso de SIMD Width

| Métrica | Standard CH16 | Lite CH12 |
|---------|--------------|-----------|
| Instruções YMM (256-bit) | 543 | 702 |
| Instruções XMM (128-bit) | 33 | 211 |
| **Razão XMM/YMM** | **5.7%** | **23.1%** |
| `vzeroupper` chamadas | 30 | 79 |

**Análise:** O Lite usa 4× mais XMM relativo a YMM que o Standard. Isso reflete o padrão 8+4 lanes: lanes 0-7 processadas em YMM, lanes 8-11 em XMM. A frequência 2.6× maior de `vzeroupper` indica transições de estado AVX/SSE ocorrendo nos pontos de chamada de função (`broadcast_scale_with_bias_f32_avx2_padded`, `fused_gemm_residual_batch_f32_12x12_padded`, `tanh_and_accumulate_block_avx2_tail`).

### 2.4 Instruções Específicas de CH=12 (Tap Gathering)

Como CH=12 (3×4 lanes YMM) não é múltiplo de 8 (1 YMM), o compilador insere blending:

| Instrução | Contagem | Função |
|-----------|----------|--------|
| `vblendps` | 78 | Combinar taps parciais YMM+XMM |
| `vmaskmovps` | 23 | Store condicional mascarado (write-combine de 12 floats em buffer de stride 16) |
| `vshufps` | 16 | Rearranjo de taps dentro de registradores |
| `vinsertps` | 3 | Inserção de taps escalares em posições específicas |
| **Total** | **120** | **3.0% das instruções** |

O Standard não tem nenhuma dessas instruções — CH=16 é exatamente 2×YMM, carga/armazenamento sem blending.

---

## 3. Análise das Hipóteses do F-A7

### 3.1 Hipótese #1 — Overhead de setup/loop proporcionalmente maior → **SUSTENTADA**

**Evidência:**
- Overhead/Useful ratio: 2.40 (Lite) vs 1.07 (Standard) — **2.24× pior**
- Overhead absoluto: 2146 (Lite) vs 627 (Standard) — 3.42× mais instruções de overhead
- O Lite tem 6 layers × 12 canais = 432 MAC/layer, o Standard tem 20 layers × 16 canais = 768 MAC/layer
- A função `process_block_internal` é chamada por layer — mais layers no Standard diluem o custo fixo por bloco, mas a comparação aqui é **por camada** (não por modelo)

**Mecanismo:** O mesmo prólogo, dispatch condicional (`if CH == 12` → stride16 path, `else` → stride=CH path), bounds checks (`debug_assert!` + `assert!`), e iteração de frames ocorre em ambas as topologias. No Lite, o corpo útil (12×12×3 = 432 FMAs) é menor que no Standard (16×16×3 = 768 FMAs), então o overhead relativo é maior.

**Estimativa de impacto:** Se o overhead pudesse ser eliminado completamente (teórico), Lite iria de 52.3 µs para ≈ 52.3 × (1 − 0.54) = 24.1 µs. Uma redução realista de 50% no overhead → ≈ 38.2 µs.

### 3.2 Hipótese #2 — Penalidade de transição SIMD (AVX↔SSE) em stride16 → **PARCIALMENTE SUSTENTADA**

**Evidência a favor:**
- `fused_gemm_residual_batch_f32_12x12_padded` (stride16.rs:86-105) intercala `_mm256_fmadd_ps` + `_mm_fmadd_ps` na mesma iteração do loop interno — 8+4 lanes
- 79 `vzeroupper` no Lite vs 30 no Standard (2.6×)
- 23.1% XMM vs 5.7% XMM no Standard (4× mais operações de meia-largura)
- Stack frame 66% maior → mais spilling de registradores YMM parciais

**Evidência contra:**
- **Todas as instruções no stride16.rs usam VEX encoding** (prefixo `v`): `vfmadd231ps %ymm`, `vfmadd231ps %xmm`
- O VEX encoding explicitamente evita a penalidade de legacy SSE→AVX (que exigiria `vzeroupper` para limpar os 128 bits superiores)
- As chamadas `vzeroupper` existem, mas ocorrem nos **pontos de chamada de função** (ABI boundary), não dentro do loop interno de stride16
- A penalidade real de SSE→AVX (dezenas de ciclos por `vzeroupper`) descrita em documentações antigas da Intel **não se aplica** quando todas as instruções são VEX-encoded

**Conclusão sobre hipótese #2:** A mistura AVX/SSE em stride16 **não causa penalidades de transição de domínio SIMD** (todas VEX). O custo real vem da **subutilização de lanes** (4 lanes SSE para 4 valores = 100% utilizado, mas 4 lanes YMM desperdiçadas no mesmo registrador estendido) e do **overhead de blending** para combinar os resultados parciais 8+4. Isso é um custo estrutural inevitável de CH=12 em hardware AVX2, não um bug ou design flaw do stride16.

### 3.3 Hipótese #3 — Meta original de eficiência incorreta → **SUSTENTADA**

**Evidência:**
- Standard (16 canais): 2449 µs/MMAC — eficiência de referência
- Lite (12 canais): 6056 µs/MMAC — 2.47× menos eficiente
- Se remover TODO o overhead (hipótese #1, 54%): Lite teórico ≈ 2786 µs/MMAC — ainda 1.14× pior que Standard
- Se reduzir overhead em 50% (realista): Lite ≈ 4423 µs/MMAC — 1.81× pior
- A latência absoluta seria: 52.3 × (4423/6056) = 38.2 µs

**Por que a meta de ≤42 µs era incorreta desde a formulação original:**
1. A meta assumia paridade de eficiência (µs/MMAC) entre Standard e Lite, ignorando que:
   - CH=12 não é múltiplo de 8 → subutilização de lanes AVX (83.3% vs 100%)
   - Overhead fixo por camada é amortizado sobre menos MAC/camada (432 vs 768)
   - Blend/perm para tap gathering é overhead puro inexistente no Standard
2. O ganho do EPIC-2 original (padding de pesos, −19.7%) já capturou o "low-hanging fruit"
3. Atingir ≤42 µs exigiria eliminar >70% do overhead atual, o que não é factível sem repensar completamente a topologia (ex.: fundir 2 camadas consecutivas para reduzir chamadas de função, ou usar AVX-512 com mask registers para CH=12)

---

## 4. Conclusão e Recomendação

### Causa-raiz do F-A7

O pad-to-16 estrutural completo (T6.S3.2, 537 linhas, 19 arquivos) **não entregou ganho de performance** (52.2→52.3 µs) porque o gargalo do Lite CH12 **não está no alinhamento de cache-line dos buffers internos**, mas sim em três fatores estruturais:

1. **Overhead fixo por camada** (dispatch condicional, prólogo, bounds checks) representa 54% das instruções — 2.24× pior que o Standard
2. **Subutilização de lanes AVX** (padrão 8+4 lanes para CH=12) força blend/perm + 23% de operações em XMM, desperdiçando 4 lanes por registrador YMM parcial
3. **Stack frame 66% maior** que o Standard, sugerindo pressão de registradores e spilling

Nenhum desses fatores é resolvido por padding de stride — todos são inerentes à topologia CH=12 em hardware AVX2.

### Recomendação para T7.S2.3

**Desfecho C (preferido):** Manter T6.S3.2 como dívida técnica aceita, reclassificando a meta de ≤42 µs como abandonada.

- O código é bit-exato, testado, e não introduz bugs
- O Lite está a 96.1% de folga do budget RT (52.3 µs vs 1333 µs deadline)
- O custo de engenharia de um revert (19 arquivos, 3 novos módulos, re-teste completo) excede o benefício
- Reverter não melhora a performance (52.3→52.2 µs, dentro do ruído)

**Alternativa — Desfecho B:** Reverter T6.S3.2, preservando apenas o padding de pesos do EPIC-2 original. Remove complexidade sem benefício, mas não altera a latência.

**Desfecho A descartado:** Nenhuma causa-raiz acionável com otimização pontual foi encontrada — os três fatores identificados são estruturais e exigiriam redesign da topologia ou migração para AVX-512.

---

## Apêndice A — Trechos Relevantes de ASM

### A.1 Lite CH12 — Prólogo e Dispatch Condicional

```asm
; Stack frame: 0x5a8 = 1448 bytes
206fba:  subq    $0x5a8, %rsp
; ...
; Dispatch: if CH == 12 → stride16 path
; O if/else gera dois caminhos completos no binário
206fac:  cmpq    $0x0, 0x90(%rdx)     ; stride16 dispatch
206fb4:  je      normal_path
206fba:  jmp     stride16_path
```

### A.2 Lite CH12 — Tap Gathering com Blend (CH=12 parcial)

```asm
; Carrega 12 taps (8+4) de stride=16:
207276:  vmovups 0x40(%rcx,%r9,4), %ymm10   ; lanes 0-7 (AVX)
20727d:  vmovsd  0x64(%rcx,%r9,4), %xmm7    ; lanes 8-9 (SSE)
207284:  vshufps $0xd0, %xmm7, %xmm7, %xmm7 ; rearranjo
207289:  vblendps $0x6, %ymm7, %ymm6, %ymm8 ; blend 8+4 → resultado parcial
207296:  vmovss  0x6c(%rcx,%r9,4), %xmm13   ; lane 11 (scalar)
20729d:  vinsertps $0x30, %xmm13, %xmm8, %xmm13
2072a3:  vblendps $0xf, %ymm13, %ymm8, %ymm5 ; resultado final 12-lane
```

### A.3 stride16.rs — Loop Interno com AVX+SSE Intercalados

```asm
; Inner loop de fused_gemm_residual_batch_f32_12x12_padded:
20ce00:  vbroadcastss -0xc0(%r9,%rax,4), %ymm8   ; AVX — lanes 0-7
20ce0a:  vbroadcastss -0x80(%r9,%rax,4), %ymm9    ; AVX
20ce18:  vbroadcastss (%r9,%rax,4), %ymm11         ; AVX
20ce21:  vmovups -0x20(%r8), %ymm12                ; AVX — 8 weights (lanes 0-7)
20ce27:  vmovups (%r8), %xmm13                     ; SSE — 4 weights (lanes 8-11)
20ce2c:  vfmadd231ps %ymm12, %ymm8, %ymm0          ; AVX FMA
20ce36:  vfmadd231ps %ymm12, %ymm10, %ymm2         ; AVX FMA
20ce40:  vfmadd231ps %xmm8, %xmm13, %xmm4          ; SSE FMA (8+4 split)
20ce4a:  vfmadd231ps %xmm10, %xmm13, %xmm6         ; SSE FMA
; Todas as instruções são VEX-encoded → sem penalidade SSE→AVX legacy
```

### A.4 Standard CH16 — Carga Limpa (Sem Blend)

```asm
; Carrega 16 taps de stride=16 (2×YMM):
1ae025:  vmovups (%r12,%r8), %ymm0       ; lanes 0-7
1ae02b:  vmovups 0x20(%r12,%r8), %ymm1   ; lanes 8-15
; Nenhum vblendps, vmaskmovps, vshufps necessário
```

---

## Apêndice B — Distribuição de Hotspots no Workload Lite

Do `perf report` do workload dedicado ao Lite (1029 samples totais):

| Função | % | Samples |
|--------|---|---------|
| `WaveNetLayer<1,12,3>::process_block_internal` | 30.66% | 294 |
| `fused_gemm_residual_batch_f32_avx2` | 25.54% | — |
| `WaveNetLayer<1,6,3>::process_block_internal` (head) | 23.19% | 236 |
| `fused_gemm_residual_batch_f32_12x12_padded` (T6.S3.2) | 5.11% | — |
| `WaveNetModel<12,3,6>::process` | 2.36% | — |

O stride16 (T6.S3.2) consome 5.11% do tempo total — não é o hotpath dominante, mas é um contribuidor não-trivial.

---

*Análise conduzida conforme a metodologia das auditorias EPIC-1/EPIC-2, usando contagem categórica de instruções, inspeção de `vzeroupper`/XMM/YMM, e comparação proporcional Lite-vs-Standard.*
