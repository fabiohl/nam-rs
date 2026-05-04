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

### Épico A — Otimização Microarquitetural SIMD

**Impacto:** Redução de latência de inferência em 10-20% nos modelos mais pesados.

#### `TA1` — Decomposição do Monólito `simd.rs` (4.256 linhas) [DONE]

**Arquivos:** `src/math/simd.rs`
**Prioridade:** Alta (Legibilidade + Compilação Incremental)

O arquivo `simd.rs` possui **4.256 linhas** — é o maior do projeto por uma
margem de 3x. Isso dificulta a navegação, aumenta o tempo de compilação
incremental e viola o princípio de coesão modular.

**Ação:**

1. Criar subdiretório `src/math/simd/` com `mod.rs` re-exportando a API pública.
2. Mover para arquivos temáticos:
   - `dot_product.rs` — Dot products AVX2/AVX-512/BF16 (~300 linhas)
   - `gemv.rs` — GEMV 4-gate, fused_add_gemv, batch kernels (~600 linhas)
   - `gemm_batch.rs` — GEMM batch e residual fusion (~400 linhas)
   - `activation.rs` — tanh_and_accumulate, sigmoid_slice, etc. (~200 linhas)
   - `dispatch.rs` — SimdMathConfig, dispatch_simd!, trait SimdMath + impls (~500 linhas)
   - `conversion.rs` — f32↔bf16, f32↔f16c helpers (~150 linhas)
   - `utility.rs` — prefetch, energy, max_diff, horizontal_sum (~200 linhas)
3. Manter `pub use` completo no `mod.rs` para zero breaking changes.
4. Testes `simd_test.rs` permanecem em `src/math/simd_test.rs` referenciando o
   módulo consolidado.

**Critério de Aceite:** `cargo test` + `cargo bench` passam sem regressão.
Nenhum arquivo individual excede 700 linhas.

---

#### `TA2` — LSTM 2-Layer Pipelining (Layer Overlap)

**Arquivos:** `src/models/lstm.rs` (linhas 436-506)
**Prioridade:** Alta (Throughput LSTM 2x16)

O `LstmModel2::process_avx2` processa sequencialmente: `layer1` completa
todos os frames, depois `layer2` processa todos os frames. Porém, o método
`process_avx512` (linhas 451-474) já implementa um **pipelining parcial**
onde layer1 avança 1 frame enquanto layer2 processa o frame anterior.

**Observação:** Este overlap está implementado *apenas* no path AVX-512.
Os paths AVX2, AVX2-VNNI, AVX-512-VNNI e AVX-512-BF16 **não** fazem overlap.

**Ação:**

1. Unificar o padrão de overlap em todas as variantes do `LstmModel2`
   (AVX2, AVX2-VNNI, AVX-512-VNNI, AVX-512-BF16).
2. Adaptar a macro `define_lstm_process!` ou criar uma nova macro
   `define_lstm2_process_pipelined!` que encapsule o pattern de pipelining.
3. Benchmark antes/depois com `criterion` no modelo `LSTM_2x16_64samp_48kHz`.

**Critério de Aceite:** Speedup mensurável (>3%) em LSTM 2-layer. Testes
de paridade escalar passam.

---

#### `TA3` — FastMath Tanh: Clamp de Saturação para Valores Extremos

**Arquivos:** `src/math/fastmath.rs` (linhas 115-159)
**Prioridade:** Média (Estabilidade Numérica)

O polinômio Minimax de grau 7 diverge para `|x| > ~8.0`, onde `tanh(x)` deveria
saturar em ±1.0. Atualmente não há clamping explícito — para modelos com pesos
atípicos, a divergência pode acumular erro.

**Ação:**

1. Adicionar clamp SIMD pré-polinômio: `x = min(max(x, -8.0), 8.0)` via
   `_mm256_min_ps` / `_mm256_max_ps` (custo: 2 instruções, ~0.5 ciclos).
2. Aplicar o mesmo para a variante AVX-512 (`_mm512_min_ps`/`_mm512_max_ps`).
3. Validar com test case de stress: `tanh([-100.0, -10.0, -8.0, 0.0, 8.0, 10.0, 100.0])`.
4. Verificar que os golden vectors de regressão continuam passando (threshold 5e-2).

**Critério de Aceite:** `simd_tanh(±100.0)` retorna ±1.0 (±1e-4). Nenhuma
regressão em benchmarks.

---

#### `TA4` — Conv1D: Eliminação de Branch no Prefetch Adaptativo

**Arquivos:** `src/models/wavenet.rs` (linhas 127-153)
**Prioridade:** Baixa (Microoptimização)

O `process_single_frame_internal` possui um `if self.dilation >= 128` dentro
do loop de taps. Para K=3, são apenas 3 iterações, mas o branch predictor é
exercitado em cada frame.

**Ação:**

1. Mover a decisão de prefetch para fora do loop: pré-calcular um ponteiro de
   função (`adaptive_prefetch_f32` vs `adaptive_prefetch_2stage_f32`) baseado
   em `self.dilation` durante o loading do modelo (cold-path).
2. Alternativamente, usar `#[cold]` / `#[inline(never)]` no branch de alta
   dilatação para ajudar o preditor.

**Critério de Aceite:** Sem regressão no benchmark WaveNet Standard.

---

#### `TA5` — Head Sum SIMD Horizontal: Unificar Patterns Duplicados

**Arquivos:** `src/math/simd.rs` (múltiplas ocorrências de `hsum_avx2` inline)
**Prioridade:** Média (DRY / Manutenibilidade)

Existem pelo menos **4 implementações independentes** de `hsum_avx2` (redução
horizontal YMM→escalar) espalhadas como `#[inline(always)] unsafe fn` locais
dentro de dot products e GEMV. O mesmo ocorre para `hsum512`.

**Ação:**

1. Consolidar em `pub(crate) unsafe fn hsum_avx2(v: __m256) -> f32` e
   `pub(crate) unsafe fn hsum_avx512(v: __m512) -> f32` no módulo
   `simd/utility.rs`.
2. Substituir todas as ocorrências inline por chamadas à versão canônica.
3. O `#[inline(always)]` na função pública garante zero overhead.

**Critério de Aceite:** Nenhuma regressão em benchmarks. Redução de ~80 linhas
de código duplicado.

---

#### `TA6` — Explorar `_mm256_fnmadd_ps` para Sigmoid Direto

**Arquivos:** `src/math/fastmath.rs` (linhas 166-183)
**Prioridade:** Baixa (Pesquisa)

Atualmente `sigmoid(x) = 0.5 * (1.0 + tanh(x * 0.5))`. Isso invoca
`simd_tanh` completo (~12 instruções) + 3 multiplicações/adições extras.

**Ação:**

1. Investigar um polinômio Minimax direto para sigmoid no intervalo [-8, 8]
   que evite chamar `tanh` como sub-rotina.
2. Candidatos: Padé [3/3] ou polinômio de grau 5 com `_mm256_rcp_ps` ao invés
   de `_mm256_rsqrt_ps` (denominador `1 + exp(-x)` → recíproco).
3. Se o speedup for <5%, manter a implementação atual (tanh-based é mais
   numericamente estável).

**Critério de Aceite:** Se implementado, erro máximo < 2e-5 vs `f32::sigmoid()`.
Speedup mensurável em benchmark LSTM fused gates.

---

### Épico B — Organização e Legibilidade do Código

**Impacto:** Manutenibilidade de longo prazo, onboarding mais rápido, compilação
incremental mais eficiente.

#### `TB1` — Decomposição de `wavenet_dyn.rs` (1.021 linhas)

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

#### `TB2` — Unificação dos Patterns de Dispatch SIMD nos Modelos LSTM

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

#### `TB3` — Limpeza do Comentário Duplicado em `wavenet.rs`

**Arquivos:** `src/models/wavenet.rs` (linhas 876-880)
**Prioridade:** Trivial

Há um comentário `[PASSO 1: Zero-Acumulador]` duplicado literalmente em linhas
consecutivas (876-878 e 878-880).

**Ação:** Remover a duplicata.

---

### Épico C — Hardening RT e Resiliência

**Impacto:** Estabilidade em produção, proteção contra cenários adversos.

#### `TC1` — Guard contra Underflow no Resampler Phase Accumulator

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

---

#### `TC2` — Timeout Gracioso no SPSC GC Consumer

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

#### `TC3` — Proteção contra Divisão por Zero no Telemetry Budget

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

---

### Épico D — Cobertura de Testes e Qualidade

**Impacto:** Confiança para refatorações futuras, detecção precoce de regressões.

#### `TD1` — Testes Unitários para `DynamicHysteresis` (Gate FSM)

**Arquivos:** `src/dsp/gate.rs` (192 linhas, 0 testes)
**Prioridade:** Alta

O módulo `gate.rs` implementa uma FSM com 4 estados e lógica de transição
complexa (hold, fade-in, fade-out) mas **não possui nenhum teste unitário**.

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

#### `TD2` — Teste de Estabilidade de Longa Duração (Soak Test)

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

#### `TD3` — Testes para `VirtualRingBuffer` Edge Cases

**Arquivos:** `src/dsp/vring.rs` (187 linhas, 0 testes)
**Prioridade:** Média

O `VirtualRingBuffer` usa `memfd_create` + `mmap` duplo — uma técnica
poderosa mas que pode falhar silenciosamente em ambientes restritos
(containers, seccomp). Não há testes unitários.

**Ação:**

1. Criar testes inline (arquivo < 300 linhas):
   - Escrita na fronteira do buffer → leitura contígua na segunda metade.
   - `Clone` produz cópia independente.
   - Tamanhos não-múltiplos de página são arredondados corretamente.
   - `Drop` desaloca corretamente (usar `valgrind` ou verificar `/proc/self/maps`).

**Critério de Aceite:** 4+ testes cobrindo boundary conditions.

---

#### `TD4` — Benchmark de Resampler Isolado

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

#### `TE1` — Documentar o Fluxo de Dados WaveNet com Diagrama Mermaid

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
