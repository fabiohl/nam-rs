<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->

<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva. -->

# TODO-sprints.md — Backlog Técnico NAM-rs

> **Gerado em:** 2026-05-04 — Pesquisador Inovador (Auditoria Completa)
>
> **Contexto:** Análise profunda de ~12.600 linhas de código-fonte, com foco em
> performance microarquitetural, higiene de código, cobertura de testes e
> aderência operacional RT. Cada tarefa é auto-contida e implementável sem
> ambiguidades.

---

## Sprint — "Refinamento Microarquitetural e Higiene"

**Objetivo:** Extrair os últimos ciclos de clock dos kernels SIMD, refatorar
arquivos massivos para legibilidade, e fechar lacunas de cobertura de testes.

---

### Épico A — Otimização Microarquitetural SIMD [AUDITADO ✅]

**Impacto realizado:** WaveNet −3,7% / LSTM −3,5% latência (benchmarks criterion 2026-05-05).

> **Auditoria:** Todas as 6 tarefas concluídas. Critérios de aceite verificados.
> Duas divergências documentadas abaixo (ambas benéficas ou neutralizadas).

#### `TA1` — Decomposição do Monólito `simd.rs` [DONE — com desvio documentado]

**Arquivos:** `src/math/simd/` (8 arquivos)
**Estrutura entregue vs especificada:**

| Especificado     | Entregue                                      | Status       |
| ---------------- | --------------------------------------------- | ------------ |
| `dot_product.rs` | `avx2.rs` + `avx512.rs` (dot prods incluídos) | ✅ funcional |
| `gemv.rs`        | `avx2.rs` + `avx512.rs` (gemv incluídos)      | ✅ funcional |
| `gemm_batch.rs`  | `avx2.rs` + `avx512.rs` (gemm incluídos)      | ✅ funcional |
| `activation.rs`  | funções em `avx2.rs` via `impl SimdMath`      | ✅ funcional |
| `dispatch.rs`    | `dispatch.rs`                                 | ✅           |
| `conversion.rs`  | `ops.rs` (f32↔bf16, prefetch)                 | ✅ funcional |
| `utility.rs`     | `utility.rs`                                  | ✅           |
| *(novo)*         | `traits.rs`, `fallback.rs`, `ops.rs`          | ✅ extra     |

**Desvio:** A separação temática foi feita por *arquitetura* (`avx2.rs`/`avx512.rs`) em vez
de por *operação* (`dot_product.rs`/`gemv.rs`/etc.). Ambas as abordagens satisfazem o
objetivo principal de legibilidade e compilação incremental. O critério formal de "nenhum
arquivo excede 700 linhas" **não foi cumprido**: `avx2.rs` = 989 linhas, `avx512.rs` = 917.

**⚠️ Gap identificado → ver `TA1-FIX` abaixo.**

---

#### `TA2` — LSTM 2-Layer Pipelining [DONE ✅]

**Entregue:** Macro `define_lstm2_process_pipelined!` criada em `src/models/lstm.rs:137`.
Todas as 5 variantes do `LstmModel2` usam pipelining: `process_avx2`, `process_avx512`,
`process_avx2vnni`, `process_avx512vnni`, `process_avx512_vnni_bf16`.

**Benchmark:** LSTM 2x16 melhorou −3,5% (criterion 2026-05-05). Testes de paridade
escalar (`test_lstm_v2_gate_major_parity`) passando.

---

#### `TA3` — FastMath Tanh: Clamp de Saturação [DONE ✅]

**Entregue:** Clamp implementado com `TANH_CLAMP_LIMIT = 15.0` (especificação dizia 8.0).

**Desvio intencional:** Clamp em 15.0 ao invés de 8.0 preserva maior fidelidade numérica
para ativações intermediárias (~8.0 a 15.0) em modelos com pesos concentrados nessa faixa.
O polinômio Minimax diverge apenas em `|x| > ~15`, não em ~8. Decisão válida.

**Teste de stress:** `test_simd_fastmath_tanh_extremes` valida 2000, 5000, 10000, 1e20,
±Inf sem NaN e saturação em ±1 (±1e-4).

---

#### `TA4` — Conv1D: Eliminação de Branch no Prefetch Adaptativo [DONE ✅]

**Entregue:** `src/math/simd/ops.rs` define `PrefetchFn` (alias de ponteiro de função).
`Conv1DLayer::prefetch_fn` é pré-calculado no cold-path do loading. Loop interno
usa `(self.prefetch_fn)(...)` sem branch por dilatação.

---

#### `TA5` — Head Sum SIMD Horizontal: Unificar Patterns Duplicados [DONE ✅]

**Entregue:** `src/math/simd/utility.rs` expõe `hsum_avx2` e `hsum_avx512` canônicos
com `#[inline(always)]`. Todas as ocorrências em `avx2.rs` e `avx512.rs` usam
`super::utility::hsum_avx2/hsum_avx512`.

---

#### `TA6` — Sigmoid Direto via Exp + RCP [DONE ✅]

**Entregue:** `simd_sigmoid_avx2` e `simd_sigmoid_avx512` em `src/math/fastmath.rs`
implementam `1/(1+exp(-x))` com polinômio Minimax D6 + 1 passo Newton-Raphson.

- Erro máximo: < 2e-5 ✅
- Sem dependência de `simd_tanh` ✅ (melhora inlining em `fused_lstm_gates_avx2`)
- `SIGMOID_CLAMP_LIMIT = 12.0` para estabilidade numérica

**Proptest:** `prop_simd_sigmoid_avx2_rmse` continua passando com threshold 5e-3
(impl alcança ~1e-5, bem abaixo).

---

### 🔍 Gaps Identificados na Auditoria do Épico A

#### `TA1-FIX` — Quebrar `avx2.rs` e `avx512.rs` por Domínio Funcional

**Arquivos:** `src/math/simd/avx2.rs` (989 linhas), `src/math/simd/avx512.rs` (917 linhas)
**Prioridade:** Baixa (critério original violado; impacto prático: compilação incremental)

`avx2.rs` contém kernels de 4 categorias distintas em sequência: dot products, GEMV/GEMM,
impl `SimdMath` para `Avx2Math` e `Avx2VnniMath`, e `horizontal_sum_avx2`. Idem para
`avx512.rs`. Isso viola o critério original de "nenhum arquivo excede 700 linhas".

**Ação:**

1. Dividir `avx2.rs` em:
   - `avx2/dot.rs` (~340 linhas: dot_product, dot_4x, interleaved, batch)
   - `avx2/gemv.rs` (~300 linhas: fused_add_gemv, gemv_overwrite, fused_add_gemm_batch, gemv_4gate, fused_gemm_residual)
   - `avx2/math_impl.rs` (~300 linhas: `impl SimdMath for Avx2Math`, `Avx2VnniMath`, horizontal_sum)
2. Dividir `avx512.rs` similarmente.
3. Re-exportar tudo via `avx2/mod.rs` e `avx512/mod.rs`.

**Critério de Aceite:** Nenhum arquivo individual excede 700 linhas.
`cargo test` + `cargo bench` passam sem regressão.

---

#### `TA7` — Atualizar Referência do Proptest de Sigmoid para `f32::sigmoid` Nativo

**Arquivos:** `tests/proptest_math.rs` (linha 85)
**Prioridade:** Baixa (Qualidade de Testes)

A referência de ground truth no proptest de sigmoid usa a identidade
`0.5 * (1.0 + (val * 0.5).tanh())` — equivalente matemático mas que agora introduz
uma camada extra de aproximação como referência. Com a implementação direta via `exp`,
o ground truth mais preciso é `1.0f32 / (1.0 + (-val).exp())`.

**Ação:**

1. Atualizar `std_sigmoid` em `proptest_math.rs` para `|val: f32| 1.0f32 / (1.0 + (-val).exp())`.
2. Ajustar threshold para `2e-5` (reflete a qualidade real da implementação).
3. Idem para `fastmath_test.rs` (`test_simd_fastmath_sigmoid_mse`).

**Critério de Aceite:** Testes passam com threshold `2e-5` e referência `exp`-nativa.

---

### ✅ Épico B — Organização e Legibilidade do Código [CONCLUÍDO — commit 6cb79d3]

**Impacto:** Manutenibilidade de longo prazo, onboarding mais rápido, compilação
incremental mais eficiente.

> **Nota de Auditoria (2026-05-05):** Todas as 3 tarefas concluídas e validadas.
>
> - `wavenet_dyn.rs` reduzido de 1021 → 302 linhas (**−70.4%**); critério exigia >30%.
> - `wavenet_common.rs` criado como base compartilhada DRY para futuros modelos A2.
> - `define_lstm1_process!` e `define_lstm2_process_pipelined!` tornam o padrão de
>   dispatch SIMD uniforme em todo o módulo `lstm.rs`.
> - Paridade numérica perfeita: MSE estático vs dinâmico = **0** em `dynamic_parity.rs`.
>
> **Impacto nos épicos futuros:**
>
> - **Épico C (Hardening RT):** `WavenetDynProcessContext` facilita validação de invariantes
>   em runtime (ex: `debug_assert` de tamanhos de buffer) sem custo no release path.
> - **A2 Integration:** `wavenet_common.rs` serve de ponto de extensão natural para os
>   novos tipos `FiLM`, `GatingMode` e `ActivationType` sem replicação de código.
> - **TB2 insight:** A macro `define_lstm1_process!` é menos parametrizada que a
>   `define_lstm2_process_pipelined!` (sem lógica de pipelining). Se o modelo 1-camada
>   precisar de pipelining no futuro, uma tarefa de extensão da macro deve ser planejada.

#### [x] `TB1` — Decomposição de `wavenet_dyn.rs` (1.021 linhas) -> `wavenet_common.rs`. [DONE]

**Arquivos:** `src/models/wavenet_dyn.rs`
**Prioridade:** Alta

O módulo dinâmico replica ~70% da estrutura do estático (`wavenet.rs`) mas com
`Vec` em vez de const generics. Grande parte é boilerplate de processamento
idêntico.

**Ação:**

1. Extrair as partes comuns (Conv1D, DenseLayer dinâmico, WaveNetLayer dinâmico)
   para structs genéricos ou traits compartilhados com o estático.
2. Considerar parametrizar via `enum { Static(CH), Dynamic(usize) }` ao invés
   de duplicar toda a lógica.
3. Se a unificação for muito invasiva, ao menos extrair o loop de processamento
   `process_block_internal` como função livre parametrizada.

**Critério de Aceite:** Paridade numérica com testes `dynamic_parity.rs`.
Redução total de LOC em `wavenet_dyn.rs` > 30%.

---

#### [x] `TB2` — Unificação dos Patterns de Dispatch SIMD nos Modelos LSTM [DONE]

**Arquivos:** `src/models/lstm.rs` (linhas 303-387 — 5 variantes quase idênticas)
**Prioridade:** Média (DRY)

`LstmModel1` possui 5 métodos `process_*` (avx2, avx512, avx2vnni,
avx512vnni, avx512_vnni_bf16) que diferem apenas na chamada a
`layer.process_sample_*` e `dot_product_*`. Código boilerplate ~90% idêntico.

**Ação:**

1. Criar uma macro `define_lstm_model_process!` análoga à `define_lstm_process!`
   já existente para as camadas.
2. Alternativamente, usar a trait `SimdMath` para parametrizar o modelo
   (já que as camadas WaveNet usam `M: SimdMath` com sucesso).

**Critério de Aceite:** Redução de ~150 linhas de boilerplate. Todos os testes
passam.

---

#### [x] `TB3` — Limpeza do Comentário Duplicado em `wavenet.rs` [DONE]

**Arquivos:** `src/models/wavenet.rs` (linhas 876-880)
**Prioridade:** Trivial

Há um comentário `[PASSO 1: Zero-Acumulador]` duplicado literalmente em linhas
consecutivas (876-878 e 878-880).

**Ação:** Remover a duplicata.

---

### Épico C — Hardening RT e Resiliência

**Impacto:** Estabilidade em produção, proteção contra cenários adversos.

#### [x] `TC1` — Guard contra Underflow no Resampler Phase Accumulator [DONE]

**Arquivos:** `src/dsp/resampler.rs` (linhas 130-190)
**Prioridade:** Alta (Segurança)

O `phase_accum` é um `f64` que avança por `phase_step` e recua por
`NUM_PHASES`. Em taxas exóticas (ex: 22050 Hz → 48000 Hz), acumulação
de erro de ponto flutuante pode causar `phase_accum` ligeiramente negativo
após muitas horas de operação contínua.

**Ação:**

1. Adicionar `debug_assert!(self.phase_accum >= 0.0)` no hot-path.
2. No release, usar `self.phase_accum = self.phase_accum.max(0.0)` como
   guard defensivo (custo: 1 instrução `maxsd`).
3. Considerar usar aritmética de ponto fixo (`u64` com fração implícita)
   para eliminar a questão de drift completamente.

**Critério de Aceite:** Nenhum panic ou artefato de áudio após 24h de
operação contínua em taxas não-padrão (22050, 44100, 96000 Hz).
Validado em CI via `test_resampler_micro_soak` (5M+ amostras combinadas).

---

#### [x] `TC2` — Timeout Gracioso no SPSC GC Consumer [DONE]

**Arquivos:** `src/spsc.rs`, `src/pw_host.rs`
**Prioridade:** Média (Robustez)

Se o consumidor GC não drenar a fila a tempo (ex: thread principal bloqueada),
o produtor RT chama `mem::forget` e seta `gc_overflow: true`. Mas não há
mecanismo de recuperação: a flag é sinalizada mas nunca resulta em ação
corretiva no loop principal.

**Ação:**

1. No `poll_rt_status`, verificar `gc_overflow` e emitir diagnóstico
   `NamDiagnostic` com sugestão de ação.
2. Drenar o buffer GC agressivamente quando detectado overflow.
3. Considerar aumentar a capacidade do ring buffer GC de `capacity * 2`
   para `capacity * 4` (tradeoff: ~512 bytes extra de memória).

**Critério de Aceite:** Log de diagnóstico emitido quando GC overflow ocorre.

---

#### [x] `TC3` — Proteção contra Divisão por Zero no Telemetry Budget [DONE]

**Arquivos:** `src/rt_setup.rs` (linhas 213-229)
**Prioridade:** Baixa (Defensivo)

Se `active_rate == 0` ou `n_samples == 0` no cálculo de `budget_us`, a
divisão produz `inf` ou `NaN`. Atualmente o guard `if active_rate > 0 &&
n_samples > 0` protege, mas o `active_rate` lido na linha 187 é diferente
do valor usado para a comparação — é um segundo `load` que pode retornar
0 se o callback RT ainda não setou o valor.

**Ação:**

1. Usar o mesmo valor de `active_rate` para ambas as verificações (ler uma
   vez, armazenar em local).
2. O `active_rate` na linha 187 está **sobrecarregado** — ele já foi swapped
   para 0 na linha 127 e depois relido. Corrigir para usar o valor do swap
   ou uma segunda flag dedicada.

**Critério de Aceite:** Nenhum `NaN` ou `inf` em logs de telemetria.

**Nota de auditoria (Auditoria Épico C):** O caminho RT em `pw_host.rs:566`
(`current_pw_rate as f64`) não está suscetível ao mesmo bug — o `NamResampler`
é sempre inicializado com `pw_rate=48_000` garantindo valor não-zero desde a
primeira chamada ao callback.

---

> **✅ Auditoria do Épico C — APROVADO (2026-05-05)**
>
> Todos os 3 hardening items (TC1, TC2, TC3) foram verificados in-code.
> Nenhuma lacuna crítica encontrada. Achados menores transferidos para o Épico D
> (TD1 e TD3 têm prioridade elevada como consequência da auditoria).

---

### Épico D — Cobertura de Testes e Qualidade

**Impacto:** Confiança para refatorações futuras, detecção precoce de regressões.

#### `TD1` — [x] Testes Unitários para `DynamicHysteresis` (Gate FSM) [DONE]

**Arquivos:** `src/dsp/gate.rs` (192 linhas, 0 testes)
**Prioridade:** Alta ⬆️ (elevada pela auditoria do Épico C)

O módulo `gate.rs` implementa uma FSM com 4 estados e lógica de transição
complexa (hold, fade-in, fade-out) mas **não possui nenhum teste unitário**.
A lógica de `FadingOut → FadingIn` usa `saturating_sub` que pode mascarar
behavior incorreto silenciosamente em reversals.

**Ação:**

1. Criar `src/dsp/gate_test.rs` com testes para:
   - Transição Open → FadingOut → Closed (silêncio prolongado).
   - Transição Closed → FadingIn → Open (sinal detectado).
   - Reversal: FadingOut → FadingIn (sinal reaparece durante fade).
   - Reversal: FadingIn → FadingOut (silêncio durante fade-in).
   - Hold timer: verificar que o hold_counter acumula corretamente.
   - Multiplicador de rampa: verificar valores intermediários.
2. Adicionar `#[cfg(test)] #[path = "gate_test.rs"] mod gate_test;` no final
   de `gate.rs`.

**Critério de Aceite:** Cobertura de todas as 4 transições e 2 reversals.

---

#### `TD2` — Teste de Estabilidade de Longa Duração (Soak Test) [Adiado]

**Arquivos:** `tests/` (novo arquivo)
**Prioridade:** Média

Não há nenhum teste que exercite o pipeline por milhões de frames para
detectar drift numérico, leaks de estado ou overflow de contadores.

**Ação:**

1. Criar `tests/soak_test.rs` que processa 10M frames de silêncio + ruído
   alternado por LSTM e WaveNet.
2. Verificar que a saída permanece boundada (`[-2.0, 2.0]`) após 10M frames.
3. Verificar que os contadores internos (buffer_start, phase_accum) não
   overflowam ou divergem.
4. Marcar com `#[ignore]` para não rodar no CI rápido.

**Critério de Aceite:** 10M frames sem panic, NaN, inf ou divergência.

---

#### `TD3` — [x] Testes para `VirtualRingBuffer` Edge Cases [DONE]

**Arquivos:** `src/dsp/vring.rs` (187 linhas, 0 testes)
**Prioridade:** Média ⬆️ (elevada pela auditoria do Épico C)

O `VirtualRingBuffer` usa `memfd_create` + `mmap` duplo — uma técnica
poderosa mas que pode falhar silenciosamente em ambientes restritos
(containers, seccomp, GitHub Actions). Não há testes unitários.

**Ação:**

1. Criar testes inline (arquivo < 300 linhas):
   - Escrita na fronteira do buffer → leitura contígua na segunda metade.
   - `Clone` produz cópia independente.
   - Tamanhos não-múltiplos de página são arredondados corretamente.
   - `Drop` desaloca corretamente (usar `valgrind` ou verificar `/proc/self/maps`).

**Critério de Aceite:** 4+ testes cobrindo boundary conditions.

---

#### `TD4` — [x] Benchmark de Resampler Isolado [DONE]

**Arquivos:** `benches/inference_bench.rs` (atualmente não benchmarka resampler)
**Prioridade:** Baixa

O resampler FIR sinc é invocado 2x por callback (input + output) mas não possui
benchmark isolado. Qualquer regressão no resampler ficaria oculta no benchmark
E2E.

**Ação:**

1. Adicionar grupo `Resampler_44100_to_48000_256samp` e `Resampler_96000_to_48000_256samp`
   ao `inference_bench.rs`.
2. Benchmarkar `process_input` e `process_output` separadamente.

**Critério de Aceite:** Baseline gravada no Criterion. Latência <15µs para
256 amostras estéreo em AVX2.

---

### Épico E — Documentação e Developer Experience

**Impacto:** Onboarding, contribuições externas, manutenção de longo prazo.

#### `TE1` — [x] Documentar o Fluxo de Dados WaveNet com Diagrama Mermaid [DONE]

**Arquivos:** `docs/architecture.md`
**Prioridade:** Baixa

A seção 2 (Inferência) lista otimizações mas não tem um diagrama visual
do fluxo de dados: Input → Rechannel → Conv1D cascade → Gated Activation →
Skip → Head → Output.

**Ação:**

1. Adicionar diagrama Mermaid mostrando o fluxo de dados através das Arrays
   e camadas, incluindo as fusões (tanh+accumulate, residual GEMV).

---

#### `TE2` — Documentar Budget de Ciclos por Operação no Hot-Path

**Arquivos:** `docs/benchmarks.md`
**Prioridade:** Média (Referência para futuras otimizações)

O benchmark reporta latência total mas não decompõe por estágio:
Conv1D, Mixin, Activation, 1x1, Head.

**Ação:**

1. Adicionar instrumentação temporária (RDTSC) por estágio no
   `process_block_internal` do WaveNet.
2. Documentar a distribuição percentual em `docs/benchmarks.md`.
3. Remover a instrumentação após a coleta (não deve ficar no hot-path
   de produção).
