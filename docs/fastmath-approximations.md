<!-- SPDX-License-Identifier: Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# FastMath: Aproximações de Funções Transcendentais

Decisões arquiteturais, resultados de experimentos e diretrizes normativas para aproximação de `tanh`, `sigmoid` e funções relacionadas no hot-path DSP do NAM-rs.

> [!IMPORTANT]
> Este documento registra **decisões definitivas** validadas por benchmarks. Não altere as escolhas de produção sem executar `cargo bench` e confirmar que não há regressão estatisticamente significativa (p < 0.05).

---

## 1. Decisão de Produção: Tanh — Padé [5,4] com Divisão de Hardware

### Função de produção

```text
tanh(x) ≈ x · (x² + 105) · (x² + 945) / ((15x² + 420) · x² + 945)
```

Implementada em `src/math/activations/tanh.rs`:

- `simd_tanh_avx2(x: __m256)` — 8 floats, AVX2 + FMA
- `simd_tanh_dual_avx2(x1, x2: __m256)` — 16 floats, coeficientes broadcast uma vez
- `simd_tanh_avx512(x: __m512)` — 16 floats, AVX-512

### Características da solução

| Propriedade                                   | Valor                                    |
|:--------------------------------------------- |:---------------------------------------- |
| Erro máximo absoluto em [-4, 4]               | ~2.32e-3                                 |
| Equivalência em bits de mantissa              | ~8.7 bits                                |
| Operações SIMD (AVX2, 8 elem)                 | ~9 ops                                   |
| Throughput `tanh_slice` (256 elem, AVX2)      | **~54 ns**                               |
| Throughput `tanh_slice` (256 elem, piecewise) | ~~163 ns~~                               |
| Ganho vs piecewise 7-segmentos (Épico 8)      | **−66.6%**                               |
| Coeficientes                                  | `PADE_TANH_*` em `src/math/constants.rs` |

### Por que `_mm256_div_ps` e não iteração Newton-Raphson (NR2)?

Experimento empírico (E8.T04, 10M amostras em [-4, 4]):

| Variante                   | Max Abs Err | RMS Error  | Throughput (256 elem) |
|:-------------------------- |:----------- |:---------- |:--------------------- |
| Piecewise 7-seg (E8.T02)   | 4.90e-3     | —          | ~163 ns               |
| Padé NR2 (rcp + 2× Newton) | 2.32e-3     | ≈ Div      | ~104 ns               |
| **Padé Div (hw div)**      | **2.32e-3** | **mínimo** | **~63 ns**            |

A razão NR2/Div de erro é **1.000×** — a dupla iteração Newton-Raphson **satura completamente** a mantissa f32 (24 bits). O recíproco não contribui nenhum drift mensurável. Logo, `_mm256_div_ps` é a escolha correta: mais simples, mais rápida, e tecnicamente equivalente em precisão a NR2.

> [!NOTE]
> A intuição de que `div_ps` é "lento" vem de arquiteturas antigas. Em microarquiteturas modernas (Intel Ice Lake, AMD Zen 3+), `_mm256_div_ps` tem latência de 10-14 ciclos e throughput de 1 por 5 ciclos — inferior a uma cascata de 6 `blendv_ps` que a alternativa piecewise exige.

---

## 2. Experimento Falhou: Piecewise 7 Segmentos para Tanh (E8.T02)

### O que foi tentado

Substituição do Padé [5,4] por 7 polinômios de grau 5 com blending branchless via `_mm256_blendv_ps`, cobrindo o domínio [-4, 4] com segmentos de largura variável.

### Motivação original

Hipótese: segmentos curtos permitem coeficientes com menor erro de Chebyshev por segmento, melhorando a precisão global.

### Resultados medidos (Épico 8)

| Métrica                      | Padé [5,4] (baseline) | Piecewise 7-seg (E8.T02)     |
|:---------------------------- |:--------------------- |:---------------------------- |
| Operações SIMD               | ~9                    | **~28** (7 polys + 6 blends) |
| Max erro [-4, 4]             | 2.32e-3               | **4.90e-3** (pior!)          |
| Throughput (256 elem)        | 63 ns                 | **163 ns** (+159%)           |
| `Prewarm_LSTM_2x16_2048samp` | baseline              | **+16%** de regressão        |

### Por que falhou

1. **Todos os 7 polinômios são avaliados incondicionalmente** (branchless). O custo não depende do valor de entrada — é sempre 7× o custo de um único polinômio.
2. **Cascade de 6 `blendv_ps`** serializa Port 5 (shuffle unit), criando uma gargalo de dependência sequencial.
3. **O tanh tem curvatura máxima em [0, 1]** — segmentos menores nessa região melhoram o erro local, mas os coeficientes obtidos sem `fpminimax` (Sollya) não são ótimos.
4. **Conclusão:** O piecewise só supera o Padé se tiver ≤3 segmentos E coeficientes recomputados via `fpminimax`. Para f32, o custo de 7 segmentos nunca se paga.

> [!CAUTION]
> **Nunca substitua o caminho de produção `simd_tanh_avx2` por uma variante piecewise sem benchmarkar o LSTM prewarm (2048 samples).** A LSTM avalia tanh 4× por célula por timestep — erros de throughput escalam linearmente com profundidade e tamanho de bloco.

### Status atual

O piecewise está preservado como `simd_tanh_piecewise_avx2` (`#[allow(dead_code)]`) para pesquisa futura. Se algum dia for retomado, os coeficientes devem ser recomputados via:

```text
sollya> fpminimax(tanh(x), [|1,3,5|], [|SG...|], [a, b], floating, absolute);
```

para cada segmento, onde `[a, b]` é o intervalo do segmento.

---

## 3. Decisão de Produção: Sigmoid — Minimax Direto (Grau 17)

### Fundamento

A implementação anterior usava a identidade `σ(x) = 0.5 + 0.5 · tanh(x/2)`, propagando o erro do tanh e adicionando operações de reescalonamento.

### Solução adotada (E8.T01)

Polinômio ímpar de grau 17 (9 termos) para o domínio [-8, 8], coeficientes obtidos via **algoritmo de Lawson** (minimax ponderado).

| Propriedade               | Identidade tanh (baseline) | Minimax direto (E8.T01)     |
|:------------------------- |:-------------------------- |:--------------------------- |
| Max erro absoluto         | ~6.8e-4                    | **~4.09e-4** (1.67× melhor) |
| Operações SIMD            | 16                         | **15**                      |
| Throughput escalar (LSTM) | baseline                   | **−20.25%**                 |

### Lição aprendida

Funções simétricas suaves em um domínio compacto (sigmoid em [-8, 8]) são melhor aproximadas por um único polinômio de grau adequado do que por segmentos. A segmentação só compensa quando há descontinuidades ou inflexões abruptas de curvatura.

---

## 4. Descoberta: Recíproco Newton-Raphson não Adiciona Drift em f32

O experimento E8.T04 provou empiricamente que **a dupla iteração Newton-Raphson (NR2) na divisão de Padé satura completamente a mantissa de 24 bits do f32**. O erro máximo absoluto de NR2 vs divisão hardware é ratio 1.000× — indistinguível dentro da precisão representável.

**Implicação normativa:** Em qualquer ponto do codebase onde exista `rcp_ps + 2× Newton-Raphson` para aproximar uma divisão de Padé racional, pode-se substituir por `div_ps` sem penalidade de precisão, e com potencial ganho de throughput (menos deps, menor register pressure).

---

## 5. Sobre o Drift WaveNet vs C++ e BF16

A comparação de paridade com NeuralAmpModelerCore revelou que o SNR da WaveNet Standard se mantém em **~9.5 dB** independentemente das melhorias de precisão nas ativações. Isso ocorre porque:

1. **O drift dominante é a quantização de pesos BF16** — os pesos u16 são convertidos para bf16 na carga, introduzindo erro de arredondamento de ~3.9e-3 por peso.
2. As melhorias em tanh/sigmoid reduzem o drift de **ativação** (segundo plano), mas o drift de **pesos** é estruturalmente maior.

### Hierarquia de fontes de drift (maior para menor)

```text
1. Quantização BF16/F16 dos pesos                    (~3.9e-3 por elemento)
2. Erro de aproximação da ativação tanh/sigmoid      (~2.3e-3 Padé, ~4.9e-3 piecewise)
3. Acumulação de ponto flutuante (conv. profunda)    (O(N·ε), mitigado por Kahan)
4. Recíproco na divisão Padé                        (≈0, saturado em 24 bits)
```

### Caminho para melhorar SNR

O E8.T03 implementa bias-tuning para BF16 (compensação no load do modelo). O ganho esperado é ≥1.5 dB de SNR em hardware BF16-capable (Intel Sapphire Rapids, AMD Zen 5+). Para validar, é necessário um runner de CI com CPU compatível.

---

## 6. Prevenção Anti-Subnormal com Dither DC (E8.T05)

### Problema

Durante fade-out/silêncio, valores próximos de zero nas ativações LSTM/WaveNet entram no território subnormal (< 1.175e-38 para f32). Subnormais têm custo de processamento alto (hardware soft emulation) e podem introduzir artefatos de "estalo digital" no fade.

### Solução adotada

Constante `DENORMAL_DITHER_OFFSET = 1.0e-11` (-220 dBFS) injetada em `apply_input_stage` e removida em `apply_output_stage`.

- **76 dB abaixo** do noise floor de um DAC de 24 bits — inaudível.
- **Zero overhead** de performance (2 loops triviais por frame, eclipsados pelo GEMV).
- Garantia: nenhum subnormal chega às ativações durante decaimento.

> [!TIP]
> Se um modelo futuro exibir "cliques" ou "estalos" na saída durante silêncio prolongado, verifique primeiro se `DENORMAL_DITHER_OFFSET` está sendo aplicado corretamente no canal afetado.

---

## 7. Sumário de Regras Normativas (Checklist)

Para qualquer modificação futura em `src/math/activations/`:

- [ ] **Benchmarke `Prewarm_LSTM_2x16_2048samp`** — sensível a throughput de tanh. Regressão > 5% é inaceitável.
- [ ] **Benchmarke `FastMath_tanh_AVX2_256elem`** — valida o micro-benchmark do slice path.
- [ ] **Não use piecewise > 3 segmentos** sem coeficientes recomputados via Sollya `fpminimax`.
- [ ] **Não substitua `div_ps` por NR2** — ambos têm mesma precisão em f32; NR2 é mais lento.
- [ ] **Mantenha a separação `single` / `dual`** — o dual deve usar broadcasts compartilhados de coeficientes.
- [ ] **Verifique a simetria** — tanh é função ímpar. Qualquer implementação deve satisfazer `f(-x) == -f(x)`.
- [ ] **Tolerância de erro máximo:** ≤ 5e-3 (tanh/sigmoid na inferência LSTM), ≤ 1e-4 (sigmoid na inicialização).
- [ ] **Execute `cargo test --lib`** — 200 testes devem passar sem falha após qualquer modificação.

---

## Referências

- Kahan, W. "Further remarks on reducing truncation errors." *CACM*, 1965. (Kahan summation)
- Muller, J.-M. *Elementary Functions: Algorithms and Implementation*. 3ª ed. Birkhäuser, 2016. (Padé approximants)
- Intel® Intrinsics Guide — `_mm256_div_ps` latency/throughput por microarquitetura.
- [Sollya](https://www.sollya.org/) — ferramenta para computar coeficientes `fpminimax` ótimos.
- `TODO-sprints.md` §Épico 8 — histórico completo de decisões e benchmark data.
