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

#### `TC1` — Redução para Divisão Única no Kernel Tanh AVX2

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

---

#### `TC2` — Paridade e Divisão Única no Kernel Tanh AVX-512

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

---

#### `TC3` — Avaliação da Divisão Rápida via Recíproco (`vrcpps`) + Newton-Raphson

**Arquivos:** `src/math/activations/tanh/high_fidelity.rs`, `src/math/activations/tanh/high_fidelity_test.rs`
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
