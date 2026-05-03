# 🚀 Backlog do Produto e Planejamento de Sprints Técnicas

**Modelo de prompt web:** Você é uma equipe de arquitetos sêniores e desenvolvedores especialistas na linguagem Rust e no subsistema de áudio Linux. Em anexo temos o aglutinado repomix do github atualizado do NAM-rs (vide subanexo README.md). Assumindo a persona/workflow .agents/workflows/diagnostico.md e .agents/skills/planejador-arquiteto/SKILL.md você vai detalhar ao máximo a implementação da tarefa técnica demandada logo abaixo. Sua resposta será concisa e direta, usando toda a sua janela de processamento e contexto para orientar o trabalho do(s) implementador(es).

---

## Épico 1: Otimização de Modelos & Inferência SIMD ✅ Auditado (2026-05-02)

> **Nota de Auditoria (2026-05-02):** As 5 tarefas (T1–T5) foram implementadas, integradas e validadas. Lint: 0 warnings. Testes: 122/122 passando. Benchmarks dentro do deadline RT. Observações para T9: `#[allow(clippy::needless_range_loop)]` em wavenet.rs:10 e 2× `#[allow(clippy::too_many_arguments)]` em `process_block_internal` requerem avaliação caso a caso.

### T1 — WaveNet: Ativação Tanh Vetorial no Hot-Path [Concluído]

**Prioridade:** 🔴 Crítica · **Estimativa:** 2h · **Impacto:** ~10-15% ciclos/frame WaveNet

**Contexto:** Em `wavenet.rs:463-465`, o loop de ativação Tanh itera escalarmente:

```rust
for j in 0..CH {
    temp[j] = crate::math::fastmath::tanh(temp[j]);
}
```

Isso gera `CH` chamadas escalares (`f32`) quando `CH` é tipicamente 4, 8, 12 ou 16 — tamanhos que cabem em 1-2 registradores YMM. A função `simd_tanh` (AVX2, 8-wide) já existe em `fastmath.rs` mas não é utilizada aqui.

**Ação:**

1. Substituir o loop escalar por chamada direta a `simd_tanh` / `simd_tanh_avx512` via dispatch `M: SimdMath`.
2. Adicionar método `SimdMath::activation_tanh_block(buf: &mut [f32])` que aplica tanh in-place usando largura nativa (8 para AVX2, 16 para AVX-512).
3. Tratar tail (CH não múltiplo de 8/16) com padding temporário em array stack.
4. Medir ganho via `cargo bench` comparando ciclos/frame antes e depois.

**Arquivos:** `src/models/wavenet.rs` (L393-485), `src/math/fastmath.rs`, `src/math/simd.rs` (trait `SimdMath`).

**Validação:** `utils/cargo-test-bench.sh` — paridade numérica bit-exact com os testes existentes em `wavenet_test.rs`.

---

### T2 — WaveNet: Soma Horizontal Head SIMD Genérica [Concluído]

**Prioridade:** 🟡 Média · **Estimativa:** 1.5h · **Impacto:** Eliminação de branches no hot-path

**Contexto:** Em `wavenet.rs:894-922`, a soma horizontal dos `head_outputs` usa `if HEAD == 8 { ... } else if HEAD == 4 { ... } else { escalar }`. Embora o compilador possa eliminar branches mortos com const generics, a legibilidade é baixa e o fallback escalar é subótimo para `HEAD=6` ou `HEAD=12`.

**Ação:**

1. [X] Criar `fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32` em `simd.rs` (via trait `SimdMath`).
2. [X] Substituir o bloco `if/else` por chamada única: `let head1_sum = unsafe { M::horizontal_sum::<HEAD>(head_ptr.add(i * HEAD)) };`.
3. [X] Adicionar teste unitário com todos os perfis HEAD (1, 4, 6, 8, 12, 16).

**Arquivos:** `src/math/simd.rs`, `src/models/wavenet.rs` (L894-922).

---

### T3 — LSTM: Fused Gate Activation com SIMD Unificado [Concluído]

**Prioridade:** 🟡 Média · **Estimativa:** 3h · **Impacto:** Redução de 2 loads/stores redundantes por neurônio

**Contexto:** A macro `define_lstm_process!` (lstm.rs:36-163) executa sigmoid e tanh como chamadas separadas nos gates, gerando stores intermediários para `sig_f`, `sig_i`, `tanh_g` antes de combiná-los. Para H=8 (cabe num YMM), é possível fundir `sigmoid(f) * cell_state + sigmoid(i) * tanh(g)` em uma sequência FMA pura sem stores intermediários.

**Ação:**

1. Criar `fn fused_lstm_gates_avx2(gf, gi, gg, go, cs: __m256) -> (__m256 new_cs, __m256 hidden)` em `fastmath.rs` que compute tudo em registradores sem spill.
2. Integrar na macro `define_lstm_process!` como path alternativo quando `$step == 8`.
3. Benchmark: medir latência por sample em modelos LSTM 1x8 e 2x16.

**Arquivos:** `src/math/fastmath.rs`, `src/models/lstm.rs` (L36-163).

---

### T4 — WaveNet DenseLayer: Overwrite Semântico sem `fill(0.0)` [Concluído]

**Prioridade:** 🟢 Baixa · **Estimativa:** 1h · **Impacto:** Eliminar ~CH×num_frames writes desnecessários

**Contexto:** `DenseLayer::process_single_frame` (wavenet.rs:253-266) faz `out_frame.fill(0.0)` seguido de `fused_add_gemv` que soma ao output. Para semântica overwrite, o `fill` é desperdiçado. A alternativa é criar `SimdMath::gemv_overwrite` que inicialize o acumulador com bias (ou zero) internamente.

**Ação:**

1. Adicionar `SimdMath::gemv_overwrite(input, weights, bias, output, do_bias)` que use `_mm256_set1_ps(bias[c])` como semente do acumulador em vez de somar ao output pré-existente.
2. Substituir `fill(0.0) + fused_add_gemv` por `gemv_overwrite` em `process_single_frame` e `process_block`.
3. Manter `process_fused` / `process_acc_single_frame` inalterados (semântica acumulativa).

**Arquivos:** `src/math/simd.rs` (trait `SimdMath`), `src/models/wavenet.rs` (L253-309).

---

### T5 — BF16 Dot Product: Implementação Nativa AVX-512 BF16 [Concluído]

**Prioridade:** 🟡 Média · **Estimativa:** 4h · **Impacto:** ~2x throughput em CPUs com `avx512bf16`

**Contexto:** Todos os caminhos BF16 usam `dot_product_bf16_fallback` (conversão bf16→f32 + FMA regular). CPUs com `avx512bf16` (Sapphire Rapids, Zen5) suportam `_mm512_dpbf16_ps` que processa 32 pares BF16 por instrução. O `dispatch_simd!` já detecta `avx512bf16` mas o kernel nativo não foi implementado.

**Ação:**

1. Implementar `dot_product_bf16_native_avx512(a: &[u16], b: &[u16]) -> f32` usando `_mm512_dpbf16_ps`.
2. Implementar `dot_product_bf16_4x_native_avx512` (4 acumuladores independentes).
3. Adicionar `#[target_feature(enable = "avx512bf16")]` e integrar no `SimdMath::Avx512Bf16Math`.
4. Atualizar `define_lstm_process!` para usar o kernel nativo quando `$is_bf16 == true` e o target BF16 está ativo.

**Arquivos:** `src/math/simd.rs` (funções `dot_product_bf16_*`), `src/models/lstm.rs`.

---

## Épico 2: Arquitetura de Áudio & PipeWire [Concluído]

### T6 — pw_host.rs: Extração do Capture Callback para Módulo Dedicado [Concluído]

**Prioridade:** 🔴 Crítica · **Estimativa:** 4h · **Impacto:** Legibilidade, testabilidade, manutenibilidade

**Contexto:** `pw_host.rs` tem 1722 linhas. O closure do capture callback (L480-800) contém toda a lógica DSP (gate, gain, resampling, inferência, clipping, telemetria). Extrair para funções `#[inline(always)]` nomeadas reduz a complexidade cognitiva sem penalidade de performance.

**Ação:**

1. Criar `fn capture_dsp_pipeline(...)` que encapsule o fluxo: gate → input gain → resample_in → model.process → resample_out → output gain → clipping → bridge write.
2. Cada estágio vira uma sub-função com `#[inline(always)]`: `apply_input_stage`, `run_inference`, `apply_output_stage`, `write_bridge`.
3. Mover helpers (`compute_gain_multipliers`, `detect_hardware_sink`, `poll_rt_status`, `configure_realtime_thread`, `select_optimal_cpu`, `parse_interrupts_per_cpu`, `lock_cpu_c_states`) para `src/rt_setup.rs` (novo módulo, ~400 linhas).
4. `pw_host.rs` final: ~800 linhas (setup de streams + closures enxutos que delegam).

**Arquivos:** `src/pw_host.rs` → split em `src/pw_host.rs` + `src/rt_setup.rs` (novo).

- [x] **T6** — pw_host.rs: Extração do Capture Callback para Módulo Dedicado
  - [x] Criar `src/rt_setup.rs` para helpers de sistema (RT Priority, Core Affinity, PM QoS).
  - [x] Modularizar pipeline DSP em sub-funções `#[inline(always)]`.
  - [x] Validar latência e zero-allocation no hot-path.

**Validação:** `cargo test` — todos os testes de `pw_host::tests` devem passar. Benchmarks inalterados.

---

### T7 — Resampler: Convolução AVX-512 Opcional [Concluído]

**Prioridade:** 🟢 Baixa · **Estimativa:** 2h · **Impacto:** ~1.5x throughput no resampler em CPUs AVX-512

**Contexto:** `convolve_avx2` em `resampler.rs:200-247` processa 16 taps/iteração (2×8 YMM). Em CPUs AVX-512, é possível processar 32 taps/iteração (2×16 ZMM), dobrando o throughput da convolução FIR.

**Ação:**

1. Criar `convolve_avx512(coeffs, input, taps) -> f32` com `_mm512_fmadd_ps` e redução via `_mm512_reduce_add_ps`.
2. Adicionar `#[target_feature(enable = "avx512f")]`.
3. Usar `SimdMathConfig::get().is_avx512` para selecionar `convolve_avx512` vs `convolve_avx2` no `ResamplerCore::process`.

**Arquivos:** `src/dsp/resampler.rs`.

---

### T8 — Gate: Rampa de Fade SIMD no `apply_gain_rt` [Concluído]

**Prioridade:** 🟢 Baixa · **Estimativa:** 1.5h · **Impacto:** Eliminar loop escalar no fade-in/out

**Contexto:** `gate.rs:186-193` aplica rampa linear escalarmente (`*s *= m; m += step`). Para buffers de 64 samples, são 64 multiplicações escalares. Uma implementação AVX2 com `_mm256_fmadd_ps` processaria 8 samples/iteração com incremento vetorizado.

**Ação:**

1. Criar `fn apply_ramp_simd(buffer: &mut [f32], start: f32, step: f32)` em `dsp/gain.rs`.
2. Usar `_mm256_set_ps(start+7*step, ..., start)` com `_mm256_add_ps` de incremento `8*step` por iteração.
3. Substituir o loop escalar em `apply_gain_rt`.

**Arquivos:** `src/dsp/gate.rs` (L186-193), `src/dsp/gain.rs`.

---

## Épico 3: Organização de Código & Higiene ✅ Auditado (2026-05-03)

> **Nota de Auditoria (2026-05-03):** Todas as 3 tarefas (T9–T11) implementadas e validadas. Lint: 0 warnings. Testes: 123/123 passando. T9: 4 supressões `needless_range_loop` residuais documentadas inline (stride indexing no hot-path). T10: `colors.rs` mantido com justificativa no `//!` doc. T11: Convenção documentada em `.agents/rules/testing.md`, todos os arquivos ≥300 LOC agora usam `_test.rs` separado.

### T9 — Limpeza de `#[allow(clippy::...)]` Residuais [Concluído]

**Prioridade:** 🟡 Média · **Estimativa:** 1h · **Impacto:** Conformidade com linting

**Contexto:** `wavenet.rs:10` tem `#![allow(clippy::needless_range_loop)]` no nível de arquivo. Vários `#[allow(clippy::too_many_arguments)]` dispersos. Cada um deve ser avaliado: se o código pode ser refatorado para eliminar a causa, remove-se a supressão; se justificado, documenta-se o motivo inline.

**Ação:**

1. `needless_range_loop` em wavenet.rs: refatorar loops `for j in 0..CH` para `iter().enumerate()` onde possível, ou documentar porquê o range é necessário (indexação de stride customizada).
2. `too_many_arguments`: avaliar se structs de contexto (`LayerContext`, `ProcessContext`) reduzem parâmetros sem overhead de indireção.
3. Resultado: zero `#[allow]` não documentados.

**Arquivos:** `src/models/wavenet.rs`, `src/models/lstm.rs`.

> **Nota da Auditoria Épico 1 (2026-05-02):** Supressões identificadas:
>
> - `wavenet.rs:10` — `#![allow(clippy::needless_range_loop)]` no nível de arquivo. Loops `for j in 0..CH` usam indexação de stride (`head_input[i * CH + j]`) — documentar justificativa inline.
> - `wavenet.rs:393,615` — 2× `#[allow(clippy::too_many_arguments)]` em `process_block_internal` (8 params). Avaliar `ProcessContext` struct.

---

### T10 — Módulo `colors.rs`: Avaliação de Necessidade [Concluído]

**Prioridade:** 🟢 Baixa · **Estimativa:** 0.5h · **Impacto:** Redução de código morto

**Contexto:** O módulo `colors.rs` reimplementa colorização ANSI. Verificar se o crate `colored` (já é dependência?) ou `owo-colors` é mais enxuto. Se `colors.rs` é único e mínimo, manter; se duplica uma dependência, eliminar.

**Ação:**

1. Verificar `Cargo.toml` por dependências de colorização.
2. Se existe dependência equivalente: migrar os callsites e deletar `colors.rs`.
3. Se não: manter e adicionar `//!` doc explicando a decisão.

**Arquivos:** `src/colors.rs`, `Cargo.toml`.

---

### T11 — Consolidação de Testes: Arquivo Separado vs Inline [Concluído]

**Prioridade:** 🟢 Baixa · **Estimativa:** 1h · **Impacto:** Consistência do projeto

**Contexto:** Testes estão divididos entre arquivos separados (`wavenet_test.rs`, `lstm_test.rs`, `wavenet_dyn_test.rs`) e módulos `#[cfg(test)] mod tests` inline (`spsc.rs`, `diagnostics.rs`, `gain.rs`, `resampler.rs`, `pw_host.rs`). Padronizar: testes unitários inline, testes de integração/benchmark em arquivos separados `_test.rs`.

**Ação:**

1. Documentar a convenção em `.agents/rules/` ou no README.
2. Verificar que todos os arquivos `_test.rs` seguem a convenção `#[path = "..._test.rs"] mod ..._test;`.
3. Testes inline em arquivos pequenos (<300 LOC) podem ser mantidos.

**Arquivos:** `src/models/`, documentação de convenções.

---

## Épico 4: Documentação Concisa & Atualizada

### T12 — `docs/architecture.md`: Poda e Atualização Pós-Resampler Nativo

**Prioridade:** 🟡 Média · **Estimativa:** 2h · **Impacto:** Documentação como fonte de verdade

**Contexto:** `architecture.md` tem 386 linhas e 48KB. Após a implementação do resampler Sinc nativo (substituindo rubato) e do gate com histerese dinâmica, seções sobre dependências externas e decisões de design obsoletas precisam ser atualizadas. O documento deve ser conciso: o código-fonte é a principal documentação.

**Ação:**

1. Remover seções sobre `rubato` e substituir por descrição do resampler Sinc Polifásico nativo.
2. Atualizar diagrama de fluxo DSP para incluir Gate FSM + Histerese.
3. Condensar seções redundantes com comentários do código-fonte (princípio DRY).
4. Meta: ≤250 linhas focadas em decisões arquiteturais não óbvias.

**Arquivos:** `docs/architecture.md`.

---

### T13 — `docs/dependencies.md`: Atualização Pós-Rubato

**Prioridade:** 🟢 Baixa · **Estimativa:** 0.5h

**Ação:**

1. Remover entrada do `rubato` (substituído pelo resampler nativo).
2. Adicionar entrada para `minstant` (telemetria RDTSC).
3. Verificar que todas as dependências do `Cargo.toml` atual estão documentadas.

**Arquivos:** `docs/dependencies.md`, `Cargo.toml`.

---

### T14 — `.agents/`: Revisão de Concisão das Skills e Rules

**Prioridade:** 🟢 Baixa · **Estimativa:** 1h

**Contexto:** As skills em `.agents/skills/` e as rules em `.agents/rules/` devem ser concisas e diretas. Verificar se há redundância entre `rules/rust.md` e as diretrizes embutidas nos prompts das skills.

**Ação:**

1. Auditar cada skill (`pesquisador-inovador`, `implementador`, `revisor-auditor`, `planejador-arquiteto`, `documentador`, `debugger`) para redundância.
2. Extrair regras comuns para `rules/` e referenciá-las nas skills via `Vide .agents/rules/rust.md`.
3. Eliminar texto duplicado entre skills.

**Arquivos:** `.agents/skills/*/SKILL.md`, `.agents/rules/*.md`.

---

## Épico 5: Qualidade, Testes & Benchmarks

### T15 — Benchmark Comparativo: LSTM Escalar vs SIMD Gates Fundidos (T3)

**Prioridade:** 🟡 Média · **Estimativa:** 1h (após T3)

**Ação:**

1. Adicionar benchmark `criterion` para `LstmModel1<8,9,32>::process` (1x8) e `LstmModel2<16,17,32,64>::process` (2x16).
2. Medir ciclos/sample antes e depois da fusão de gates (T3).
3. Registrar resultados em `docs/benchmarks.md`.

**Arquivos:** `benches/`, `docs/benchmarks.md`.

---

### T16 — Testes de Regressão Numérica: Golden Vectors

**Prioridade:** 🟡 Média · **Estimativa:** 2h

**Contexto:** Otimizações SIMD (T1, T3, T4) podem introduzir desvios numéricos. É essencial ter golden vectors (input→output esperado) para cada perfil de modelo.

**Ação:**

1. Gerar golden vectors executando cada perfil (WaveNet Standard/Lite/Feather/Nano, LSTM 1x8/1x16/2x16) com input senoidal de 1024 samples.
2. Salvar como `.bin` em `tests/golden/`.
3. Adicionar teste `#[test] fn test_golden_wavenet_standard()` que carrega o golden e compara com MSE < 1e-6.
4. Re-gerar goldens após cada otimização validada.

**Arquivos:** `tests/golden/` (novo), `src/models/wavenet_test.rs`, `src/models/lstm_test.rs`.

---

### T17 — Cobertura de Testes: `diagnostics.rs` — Variante `DeadlineExceeded`

**Prioridade:** 🟢 Baixa · **Estimativa:** 0.5h

**Contexto:** O teste `test_all_codes_have_unique_numeric` em `diagnostics.rs:443-478` lista todos os códigos mas omite `DeadlineExceeded` do array `all`. Isso permite que o código `E2001` seja duplicado sem detecção.

**Ação:**

1. Adicionar `DeadlineExceeded` ao array `all` no teste.
2. Verificar que nenhum outro código está faltando.

**Arquivos:** `src/diagnostics.rs` (L443-478).

---

## Resumo de Prioridades

| Prio | ID  | Épico      | Tarefa                    | Estimativa |
| ---- | --- | ---------- | ------------------------- | ---------- |
| 🔴   | T1  | Inferência | WaveNet Tanh Vetorial     | 2h         |
| 🔴   | T6  | PipeWire   | Extração Capture Callback | 4h         |
| 🟡   | T2  | Inferência | Head Sum SIMD Genérica    | 1.5h       |
| 🟡   | T3  | Inferência | LSTM Fused Gates          | 3h         |
| 🟡   | T5  | Inferência | BF16 Nativo AVX-512       | 4h         |
| ✅   | T9  | Higiene    | Limpeza Clippy Allows     | 1h         |
| 🟡   | T12 | Docs       | architecture.md Poda      | 2h         |
| 🟡   | T15 | Bench      | Benchmark LSTM            | 1h         |
| 🟡   | T16 | Testes     | Golden Vectors            | 2h         |
| 🟢   | T4  | Inferência | Dense Overwrite           | 1h         |
| ✅   | T7  | Resampler  | Convolução AVX-512        | 2h         |
| ✅   | T8  | Gate       | Rampa SIMD                | 1.5h       |
| ✅   | T10 | Higiene    | colors.rs Avaliação       | 0.5h       |
| 🟢   | T11 | Higiene    | Convenção Testes          | 1h         |
| 🟢   | T13 | Docs       | dependencies.md           | 0.5h       |
| 🟢   | T14 | Docs       | .agents/ Concisão         | 1h         |
| 🟢   | T17 | Testes     | DeadlineExceeded Test     | 0.5h       |
