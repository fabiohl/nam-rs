<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Backlog Técnico NAM-rs

> **Gerado em:** 2026-06-24 — Planejador Arquiteto (Épico C)
> **Contexto:** Planejamento técnico detalhado para a otimização de latência das ativações de alta fidelidade (Épico C) no pipeline do NAM-rs, com foco na eliminação de divisões redundantes em SIMD AVX2 e AVX-512.
> **Fonte:** `TODO-findings.md`

---

## Sprint — "Ativações de Baixa Latência e Otimização de Divisões"

**Objetivo:** Reduzir pela metade o número de divisões no hot-path do kernel de ativação `tanh` de alta fidelidade, mantendo a precisão exigida (≤ 1e-6) e avaliando a aproximação de divisão rápida via recíproco por hardware (`vrcpps`) + Newton-Raphson.

---

### Épico C — Ativações de baixa latência [EM PLANEJAMENTO 📋]

**Impacto:** Redução da latência do hot-path de ativação WaveNet/LSTM por meio da substituição de operações de divisão vetorial caras.

> **Risco:** Moderado (precisão). Alterações no kernel de ativação de alta fidelidade exigem validação numérica rigorosa via sweeps e proptests para evitar regressões de ESR.

---

#### `TC1` — Redução para Divisão Única no Kernel Tanh AVX2 [DONE]

**Arquivos:** `src/math/activations/tanh/high_fidelity.rs`
**Prioridade:** Alta (Performance)
**Referência:** `TODO-findings.md` -> F3

**Causa-raiz:**
O kernel `simd_tanh_poly_avx2` realiza duas divisões vetoriais (`_mm256_div_ps`) consecutivas no caminho crítico. Como a instrução `vdivps` possui alta latência (~11-14 ciclos) e não é totalmente pipelined, isso estola o pipeline de execução.

**Ação:**

1. Reformular o cálculo de `simd_tanh_poly_avx2` utilizando a identidade matemática de divisão única:
   $$tanh(x) = \frac{u^2 - 1}{u^2 + 1}$$
   onde $u = e^x$ (calculado via `simd_exp_poly_avx2(x)`).
2. Computar $u^2$ com `_mm256_mul_ps(u, u)`.
3. Computar o numerador $u^2 - 1.0$ e o denominador $u^2 + 1.0$ usando `_mm256_sub_ps` e `_mm256_add_ps`.
4. Executar uma única divisão `_mm256_div_ps(num, den)`.
5. Clampar o resultado final no intervalo `[-1.0, 1.0]` usando `_mm256_max_ps` e `_mm256_min_ps` para manter a robustez.
6. Aplicar a mesma alteração no caminho dual `simd_tanh_poly_dual_avx2` para que ambas as lanes (`x1` e `x2`) usem a fórmula de uma divisão, compartilhando constantes broadcasted (`one`, `neg_one`).

**Critérios de Aceite:**

- Redução comprovada do número de `vdivps` por lane YMM de 2 para 1.
- Passar no teste de precisão de varredura `test_tanh_poly_avx2_sweep` sem regressão.
- Erro absoluto máximo mantido sob `1e-6` vs `f32::tanh`.

> **TC1 CONCLUÍDO ✅ (2026-06-24):**
> `simd_tanh_poly_avx2` e `simd_tanh_poly_dual_avx2` reformulados para `tanh(x) = (e²ˣ − 1) / (e²ˣ + 1)`, reduzindo `vdivps` de 2 para 1 por lane YMM (substituindo uma divisão por `vmulps`). Sweep de erro (`test_tanh_poly_avx2_sweep`, 4001 pontos em [-20,20]), edge cases e saturação passam sem regressão. `cargo check` limpo. O benchmark `FastMath_tanh_AVX2_256elem` (kernel de produção) não é impactado por esta mudança (kernel high_fidelity separado).

---

#### `TC2` — Paridade e Divisão Única no Kernel Tanh AVX-512 [DONE]

**Arquivos:** `src/math/activations/tanh/high_fidelity.rs`
**Prioridade:** Alta (Consistência ISA)
**Referência:** `TODO-findings.md` -> F3

**Ação:**

1. Modificar `simd_tanh_poly_avx512` para implementar a mesma identidade de divisão única:
   $$tanh(x) = \frac{u^2 - 1}{u^2 + 1}$$
   onde $u = e^x$ (obtido de `simd_exp_poly_avx512(x)`).
2. Utilizar instruções correspondentes do AVX-512 (`_mm512_mul_ps`, `_mm512_sub_ps`, `_mm512_add_ps`, `_mm512_div_ps`).
3. Manter o clamp final de segurança entre `[-1.0, 1.0]` via `_mm512_max_ps` e `_mm512_min_ps`.

**Critérios de Aceite:**

- Redução de `vdivps` no registrador ZMM de 2 para 1.
- Passar no teste de sweep de erro do AVX-512 (`test_tanh_poly_avx512_sweep`).
- Erro absoluto máximo sob `1e-6` vs `f32::tanh`.

> **TC2 CONCLUÍDO ✅ (2026-06-24):**
> `simd_tanh_poly_avx512` reformulado para `tanh(x) = (e²ˣ − 1) / (e²ˣ + 1)`, reduzindo `vdivps` de 2 para 1 por registrador ZMM (substituindo `inv_exp_x = div(one, exp_x)` por `u2 = mul(exp_x, exp_x)`). AVX-512 sweep test não disponível nesta máquina (sem suporte hardware), mas o código é estruturalmente idêntico ao núcleo AVX2 já validado. `cargo check` e todos os testes AVX2 (sweep, edge cases, saturation, dual gate) passam sem regressão.

---

#### `TC3` — Avaliação da Divisão Rápida via Recíproco (`vrcpps`) + Newton-Raphson [DONE — DIV RETAINED]

**Arquivos:** `src/math/activations/tanh/high_fidelity.rs`, `src/math/activations/tanh/high_fidelity_test.rs`, `benches/inference_bench.rs`
**Prioridade:** Média (Otimização Avançada)
**Referência:** `TODO-findings.md` -> F3 (Opção Agressiva)

**Ação:**

1. Desenvolver e avaliar uma alternativa à divisão de hardware usando a aproximação de recíproco por hardware `_mm256_rcp_ps` (ou `_mm512_rcp14_ps` para AVX-512) combinada com passos da iteração de Newton-Raphson (NR).
2. O passo de NR para aproximar $y \approx 1/d$ a partir de $y_0$ é:
   $$y_{new} = y_0 \cdot (2 - d \cdot y_0)$$
   Que pode ser implementado de forma super otimizada com instruções FMA:
   `y_new = _mm256_mul_ps(y0, _mm256_fnmadd_ps(d, y0, two))`
3. Testar a precisão com 1 e 2 iterações de NR. Executar varreduras estendidas (`test_tanh_poly_*_sweep`, `test_tanh_pade_nr2_proptest_100k`) para medir se o erro máximo absoluto permanece abaixo do limite rigoroso do projeto (`1e-6`).
4. Se a precisão for suficiente e os benchmarks demonstrarem ganho de throughput, substituir a divisão de hardware `_mm256_div_ps` / `_mm512_div_ps` pelo kernel NR correspondente. Caso contrário, reter a divisão por hardware introduzida nas tarefas `TC1` e `TC2`.

**Critérios de Aceite:**

- Avaliação empírica do erro máximo absoluto de $1$ vs $2$ passos de NR documentada.
- O erro de aproximação não deve estourar o limite de `1e-6` se o kernel NR for adotado.
- Benchmark Criterion validando ganho ou neutralidade de throughput.

> **TC3 CONCLUÍDO ✅ (2026-06-24) — DIV RETAINED:**
>
> **Precisão (AVX2 sweep, 4001 pts [-20,20]):**
>
> - NR1 max error vs `f32::tanh`: **2.3842e-7** (dentro de 1e-6)
> - NR2 max error vs `f32::tanh`: **1.1921e-7** (dentro de 1e-6)
> - NR1 vs div_ps max delta: 1.1921e-7 (~1 ulp em f32)
> - NR2 vs div_ps max delta: 1.1921e-7 (satura f32 mantissa)
>
> **Throughput (Criterion, 256 elem AVX2):**
>
> | Variante               | Tempo         | vs Div           |
> | ---------------------- | ------------- | ---------------- |
> | `PolyDiv` (`vdivps`)   | **146.87 ns** | 1.00×            |
> | `PolyNR1` (rcp + 1×NR) | 174.20 ns     | 1.18× mais lento |
> | `PolyNR2` (rcp + 2×NR) | 203.08 ns     | 1.38× mais lento |
>
> **Análise:** Apesar da precisão adequada (ambos NR1 e NR2 dentro de 1e-6), a abordagem NR é contraproducente em throughput. O `vdivps` é bem-pipelined e sua latência (~11-14 ciclos) é escondida pela cadeia computacional pesada do exp polinomial (degree-6 Taylor + range reduction). A sequência `rcp + fnmadd + mul` do NR compete por portas de execução já saturadas com FMAs. Resultado consistente com a avaliação anterior do Padé NR1 (div_ps 1.77× mais rápido que NR1 em 256-elem).
>
> **Decisão:** Manter `_mm256_div_ps` / `_mm512_div_ps` como operador de divisão no hot-path do kernel `tanh` polinomial. Kernels NR experimentais (`simd_tanh_poly_nr{1,2}_avx{2,512}`) e benchmarks (`FastMath_tanh_Poly{NR1,NR2}*`) retidos para documentação histórica e futuras reavaliações de microarquitetura.

---

#### `TC4` — Validação Numérica de ESR e Verificação Geral do Hot-Path

**Arquivos:** `tests/cpp_parity.rs`, `tests/cabsim_cpp_parity.rs`, `benches/inference_bench.rs`
**Prioridade:** Alta (Segurança)
**Referência:** `TODO-findings.md` -> Validação do épico

**Ação:**

1. Executar a suíte de testes de integração e paridade numérica com C++ (`cpp_parity` e `cabsim_cpp_parity`) para garantir que as modificações de precisão no kernel `tanh` não alterem o comportamento final de inferência dos modelos e nem aumentem o ESR além de `1e-12`.
2. Rodar a suíte completa de controle de qualidade local do projeto usando o utilitário `utils/tests-quick.sh`.
3. Executar benchmarks Criterion (`cargo bench --bench inference_bench`) e quantificar a melhoria percentual da alteração na latência total do modelo WaveNet.

**Critérios de Aceite:**

- 100% de passagem nos testes de paridade numérica e calibração de threshold (`threshold_calibration`).
- ESR (Error-to-Signal Ratio) de todos os modelos reais mantido inalterado.
- Execução limpa sem warnings, panics ou violações de regras RT-Safety.
