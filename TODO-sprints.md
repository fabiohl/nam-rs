<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO Sprints — nam-rs

> **Origem:** Auditoria `revisor-auditor.md` — Foco CLAP plugin, quick wins de performance e qualidade de código.
> **Data:** 2026-06-08

---

## Épico 1: Quick Wins de Performance — CLAP Plugin Hot-Path

**Objetivo:** Obter ganhos mensuráveis de desempenho no caminho crítico do processamento de áudio CLAP com intervenções cirúrgicas de baixo risco, antes da próxima rodada de novas funcionalidades.

---

### Sprint 1.1: Otimizações de Hot-Path — Orquestrador e Bypass

#### TT-1.1.1: Aplicar `#[cold]` ao corpo da função `process_bypass` [DONE]

- **Arquivo:** `src/clap/processor/dsp/bypass.rs`
- **Descrição:** Atualmente `process_bypass` inteira tem `#[inline(always)]`, mas apenas o early-return `if !self.params.bypass { return Ok(false); }` é hot. O restante do corpo (bypass ativo) é cold. Separar em: (a) inline fast-path check + (b) `#[cold] #[inline(never)]` para `process_bypass_cold`.
- **Prioridade:** Alta
- **Impacto esperado:** Redução de I-cache pressure no caminho comum (bypass desligado).

#### TT-1.1.2: Substituir loop escalar de pico no bypass por SIMD [DONE]

- **Arquivo:** `src/clap/processor/dsp/bypass.rs`
- **Descrição:** As linhas 31-36, 40-45, 55-60, 64-69 usam loops escalares para calcular picos no caminho de bypass. Substituir por chamada a `compute_peak_abs_stereo` (já existente em `src/math/dsp/stereo/`).
- **Prioridade:** Média
- **Impacto esperado:** 2-3× mais rápido no cálculo de pico em bypass (caminho menos frequente, mas ainda relevante).

#### TT-1.1.3: Hoist `get_gain_lut()` no `process_events` [DONE]

- **Arquivo:** `src/clap/processor/events.rs`
- **Descrição:** `get_gain_lut()` (linha 26) é chamada em todo bloco de áudio, independentemente de haver eventos. A referência deve ser armazenada no `NamClapProcessor` em `activate()` e reutilizada.
- **Prioridade:** Alta
- **Impacto esperado:** Elimina chamada de função por bloco. `get_gain_lut()` retorna `&'static GainLut` — custo é baixo, mas a eliminação de qualquer branch/call no hot-path é desejável.

#### TT-1.1.4: Cache de `GateParams` no `NamClapProcessor` [DONE]

- **Arquivo:** `src/clap/processor/dsp/orchestrator.rs`
- **Descrição:** `GateParams` (linhas 59-63) é reconstruído a cada iteração `port_pair`. Os campos `hold_frames`, `fade_frames`, `inv_fade_frames`, `mono_epsilon` são constantes. Mover para campo cache no `NamClapProcessor`, invalidado via `gate_dirty`.
- **Prioridade:** Alta
- **Impacto esperado:** Elimina construção de struct + 4 atribuições por bloco.

#### TT-1.1.5: Cache de `DspPipelineContext` parcial [DONE]

- **Arquivo:** `src/clap/processor/dsp/orchestrator.rs`
- **Descrição:** `DspPipelineContext` (linhas 65-80) tem 16 campos, dos quais `resampler`, `active_model_l/r`, `silence_hysteresis`, `mono_hysteresis`, `process_mono`, `rt_status`, `adaptive`, `bridge_writer` são referências mutáveis que precisam ser atualizadas. Campos como `input_gain_mult: 1.0` e `output_gain_mult: 1.0` são constantes (o ganho é aplicado separadamente no CLAP via `apply_input_gain`/`apply_output_gain`). Documentar ou explicitar que esses valores constantes são intencionais.
- **Prioridade:** Baixa
- **Impacto esperado:** Clarificação de código, sem ganho de performance significativo.

---

### Sprint 1.2: Otimizações SIMD e Fusão de Passes

#### TT-1.2.1: Fundir ganho + detecção de clipping no caminho mono (não-stereo) [DONE]

- **Arquivos:** `src/clap/processor/dsp/gain.rs`, `src/math/dsp/gain/`
- **Descrição:** No caminho `#[cfg(not(feature = "stereo"))]` de `apply_input_gain` (linhas 46-65), o ganho é aplicado via SIMD e depois um loop escalar detecta clipping. O caminho stereo já tem `apply_gain_and_detect_clipping_stereo` que faz ambos em um único passe. Criar `apply_gain_and_detect_clipping_mono` equivalente.
- **Prioridade:** Alta
- **Impacto esperado:** Elimina um scan O(n) adicional por bloco no modo mono.

#### TT-1.2.2: Substituir loop escalar de dither por SIMD [DONE]

- **Arquivos:** `src/dsp/pipeline/stages/input.rs`, `src/dsp/pipeline/stages/output.rs`
- **Descrição:** As linhas 122-132 (input.rs) e 35-44 (output.rs) usam `unsafe { for i in 0..n_samples { *ptr.add(i) += CONST; } }` para injeção/compensação de dither. Substituir por broadcast SIMD + vector add, usando `apply_gain_simd` com offset ou kernel dedicado.
- **Prioridade:** Média
- **Impacto esperado:** 4-8× mais rápido (dependendo do ISA), aproveitando L1 cache bandwidth.

#### TT-1.2.3: Fundir `apply_output_stage`: ganho+clipping + gate fade em um passe [DONE]

- **Arquivo:** `src/dsp/pipeline/stages/output.rs`
- **Descrição:** `apply_output_stage` chama `dispatch_simd!` duas vezes (linhas 48-52 para gain+clipping, 56-62 para gate fade), cada uma tocando os mesmos buffers stereo. Avaliar fusão em um único passe SIMD que aplica ganho, detecta clipping, e aplica fade do gate simultaneamente.
- **Prioridade:** Média
- **Impacto esperado:** Reduz tráfego L1 em ~50% no estágio de saída, elimina um dispatch branch.

#### TT-1.2.4: Otimizar `compute_output_peaks` no modo mono [DONE]

- **Arquivo:** `src/clap/processor/dsp/peaks.rs`
- **Descrição:** No caminho `#[cfg(not(feature = "stereo"))]` (linhas 64-85), o cálculo de pico chama `compute_peak_abs_stereo` com o mesmo buffer duplicado. Criar `compute_peak_abs_mono` que opera em um único canal.
- **Prioridade:** Baixa
- **Impacto esperado:** Reduz pela metade o trabalho no cálculo de pico em mono (ganho modesto, mas código mais limpo).

---

### Sprint 1.3: Atributos de Otimização e Layout

#### TT-1.3.1: Adicionar `#[inline]` a `AdaptiveCompute::set_mode()`

- **Arquivo:** `src/dsp/adaptive.rs`
- **Descrição:** `set_mode()` (linha 140) é chamado do hot-path em `process_events` (events.rs:117, 205, 229, 237, 245). Atualmente sem `#[inline]`, o compilador pode decidir não inlinear.
- **Prioridade:** Alta
- **Impacto esperado:** Garante inline no hot-path, evitando call overhead para função pequena.

#### TT-1.3.2: Aplicar `#[cold]` nos caminhos de erro de `process_events`

- **Arquivo:** `src/clap/processor/events.rs`
- **Descrição:** O bloco de eventos SPSC (linhas 28-67) processa `ClapParamPayload::LoadModel` que é raro. Separar o processamento de `LoadModel` em função `#[cold]`.
- **Prioridade:** Baixa
- **Impacto esperado:** Melhora I-cache no caminho comum (sem eventos).

#### TT-1.3.3: Garantir `#[inline]` em `ParamSmoother::peek()` e `set()`

- **Arquivo:** `src/dsp/smoother.rs`
- **Descrição:** `peek()` (linha 91) e `set()` (linha 97) já têm `#[inline]`. `current_value()` (linha 79) também. Verificar se `target_value()` (linha 85) deve ter `#[inline]` — é chamado em gain.rs linhas 14, 48, 80, 104.
- **Prioridade:** Média
- **Impacto esperado:** Consistência de inline hints no hot-path.

---

## Épico 2: Qualidade de Código e Segurança — CLAP Plugin

**Objetivo:** Elevar a qualidade do código CLAP com documentação de segurança, redução de duplicação, e conformidade com as regras do projeto.

---

### Sprint 2.1: Documentação de Blocos `unsafe`

#### TT-2.1.1: Documentar blocos `unsafe` em `gain.rs`

- **Arquivo:** `src/clap/processor/dsp/gain.rs`
- **Descrição:** As chamadas `unsafe` nas linhas 16-22, 25-32, 35-39, 82-87, 91-97 precisam de comentários `// SAFETY:` documentando as pré-condições garantidas pelo caller.
- **Prioridade:** Alta
- **Impacto esperado:** Conformidade com `#![warn(clippy::undocumented_unsafe_blocks)]`, auditabilidade.

#### TT-2.1.2: Documentar blocos `unsafe` em `peaks.rs`

- **Arquivo:** `src/clap/processor/dsp/peaks.rs`
- **Descrição:** Linhas 38-42, 47-51, 55-59, 67-72, 76-82.
- **Prioridade:** Alta

#### TT-2.1.3: Documentar blocos `unsafe` em `input.rs` e `output.rs`

- **Arquivos:** `src/dsp/pipeline/stages/input.rs`, `src/dsp/pipeline/stages/output.rs`
- **Descrição:** `get_unchecked_mut` nos loops de dither (input.rs:122-132, output.rs:35-44).
- **Prioridade:** Média

---

### Sprint 2.2: Redução de Duplicação

#### TT-2.2.1: Extrair lógica de sincronização de parâmetros

- **Arquivos:** `src/clap/processor/events.rs`, `src/clap/extensions/params/main.rs`
- **Descrição:** A lógica de atualizar `self.shared.ui_to_rt.*` + `self.params.*` + `smoother` + `gate_dirty` está triplicada nos três caminhos de eventos (SPSC, Host Events, GUI sync) e na `flush()` dos params. Extrair para métodos privados no `NamClapProcessor`.
- **Prioridade:** Alta
- **Impacto esperado:** Elimina ~80 linhas duplicadas, reduz risco de divergência de comportamento entre caminhos.

#### TT-2.2.2: Unificar conversão `bool` ↔ `u32` para `param_bypass`

- **Arquivos:** `src/clap/processor/events.rs`, `src/clap/extensions/params/main.rs`, `src/clap/extensions/state.rs`
- **Descrição:** A conversão `val > 0.5` / `if bypass { 1 } else { 0 }` aparece em 4+ locais. Criar constantes `BYPASS_OFF: u32 = 0` e `BYPASS_ON: u32 = 1` com helper functions.
- **Prioridade:** Média

#### TT-2.2.3: Unificar duplicação `shared.ui_to_rt.gui_param_generation.fetch_add(1, Release)`

- **Arquivos:** `src/clap/gui/ui/zones/controls.rs`, `src/clap/extensions/state.rs`
- **Descrição:** O bump da geração de parâmetros GUI é feito manualmente em vários locais. Centralizar em `NamClapShared::bump_generation()`.
- **Prioridade:** Média

---

### Sprint 2.3: Sanidade e Segurança

#### TT-2.3.1: Adicionar `debug_assert!` nas pré-condições do `DelayLine::push`

- **Arquivo:** `src/dsp/resampler.rs`
- **Descrição:** Já existe `debug_assert!(pos < TAPS_PER_PHASE)` na linha 67. Verificar se `pos` não excede `DELAY_LINE_LEN` em `window_ptr`.
- **Prioridade:** Baixa

#### TT-2.3.2: Verificar bounds check no `parking_lot` de GC

- **Arquivo:** `src/clap/processor/gc.rs`
- **Descrição:** A capacidade do `parking_lot` é fixa em 16 (state.rs:67). Durante modelo de 2 canais + resampler em swap rápido, 3 slots são consumidos. Se houver múltiplos swaps sem drenagem pelo main thread, pode estourar. Verificar se 16 é suficiente ou adicionar métrica de saturação.
- **Prioridade:** Baixa

---

## Épico 3: Cobertura de Testes — CLAP Plugin

**Objetivo:** Garantir que as otimizações dos Épicos 1 e 2 não introduzam regressões, com testes adicionais de borda.

---

### Sprint 3.1: Testes de Regressão para Otimizações

#### TT-3.1.1: Teste de clipping detection em caminho mono não-stereo

- **Arquivo:** `src/clap/processor_test.rs` ou teste inline
- **Descrição:** Após TT-1.2.1, adicionar teste que injeta sinal acima de 0 dBFS e verifica flag `HAS_CLIPPED`.
- **Prioridade:** Alta

#### TT-3.1.2: Teste de dither via SIMD vs escalar

- **Arquivo:** `src/dsp/pipeline/pipeline_test.rs`
- **Descrição:** Após TT-1.2.2, garantir que a saída com dither SIMD é bit-exact com a versão escalar.
- **Prioridade:** Média

#### TT-3.1.3: Teste de convergência de `ParamSmoother` com `#[inline]` em `target_value()`

- **Arquivo:** `src/dsp/smoother.rs` (testes já existem)
- **Descrição:** Após TT-1.3.3, verificar que os testes existentes continuam passando.
- **Prioridade:** Baixa

---

### Sprint 3.2: Cobertura de Cenários Extremos

#### TT-3.2.1: Teste de stress de GC com 1000 model swaps

- **Arquivo:** `src/clap/processor_test.rs`
- **Descrição:** Validar que o sistema de 3 tiers (channel → parking_lot → overflow) não perde memória sob swaps rápidos sem drenagem do main thread.
- **Prioridade:** Média

#### TT-3.2.2: Teste de `RenderMode::Offline` forçando `AdaptiveCompute::Off`

- **Arquivo:** `src/clap/processor_test.rs`
- **Descrição:** Verificar que mesmo com `AdaptiveCompute::Aggressive` configurado, o modo offline força `Off` e não há degrade durante bounce.
- **Prioridade:** Média

---

## Épico 4: Documentação Técnica

**Objetivo:** Atualizar documentação para refletir as otimizações aplicadas e arquitetura atual.

### TT-4.0.1: Atualizar `docs/architecture.md` com detalhes da pipeline CLAP

- **Descrição:** Adicionar seção sobre a pipeline DSP CLAP (Orquestrador → Eventos → DSP → Picos → Telemetria) e o fluxo de parâmetros (Main Thread → SPSC/Atomics → RT Thread).
- **Prioridade:** Baixa

### TT-4.0.2: Atualizar `docs/clap_integration.md` com otimizações de cache-line e SPSC

- **Descrição:** Documentar `UiToRt`, `RtToUi`, `ColdShared` e o protocolo de sincronização de parâmetros via `gui_param_generation`.
- **Prioridade:** Baixa
