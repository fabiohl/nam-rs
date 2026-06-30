<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Execução

Este arquivo define os sprints e as tarefas técnicas para a execução dos épicos de melhoria do NAM-rs, com base nas descobertas consolidadas em [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

---

## Épico α — Controles de usuário de baixo risco (quick wins)

**Objetivo:** Expor controles de runtime que já estão implementados ou necessitam de ajustes mínimos de usabilidade, com foco na segurança e paridade entre o plugin CLAP e o executável Standalone.
**Risco:** BAIXO. Não altera matemática de inferência nem as topologias neurais.
**Origem dos achados:** I1 e I2 do [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

### Dependências e Sequência de Execução

Para otimizar e garantir a paridade, o trabalho será dividido em dois sprints focados:

1. **Sprint α1: Oversampling Standalone (Foco em I2)**
2. **Sprint α2: Controle de Precisão de Ativação (Foco em I1)**

---

### Sprint α1 — Paridade de Oversampling no Standalone (I2)

Este sprint resolve a lacuna de runtime oversampling no executável standalone, permitindo que a CLI honre a flag de inicialização e possibilite a troca de fator em tempo de execução de maneira RT-safe (zero-alloc no thread de áudio).

#### [x] Tarefa α1.1 — Correção da Inicialização via CLI [BAIXO RISCO]

- **Descrição:** Corrigir o bug de inicialização onde o fator `--oversample` parseado da linha de comando é descartado em `run.rs` e substituído por `OversampleFactor::Off` fixo em `capture/setup.rs:66`.
- **Mudanças propostas:**
  - Em [run.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/run.rs), alterar a desestruturação de `config` (remover o descarte de `oversample: _os`) e passar a flag real para o init do capture stream.
  - Em [setup.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/capture/setup.rs), usar o valor de oversample da configuração para inicializar o `CaptureState`.
- **Validação:** Rodar standalone especificando `--oversample 2x` ou `--os 4x` e validar que as engines de oversampling são criadas com o fator correto no log de inicialização.

#### [x] Tarefa α1.2 — SPSC Rebuild Pipeline para Oversampling no Standalone [MÉDIO RISCO]

- **Descrição:** Implementar a troca de oversampling em tempo de execução no standalone usando o padrão de reconstrução assíncrona off-RT.
- **Mudanças propostas:**
  - Criar um canal SPSC (`rtrb` Ring Buffer) para passar novas instâncias de `Box<OversampleEngine>` construídas na thread principal para a thread de áudio (similar ao `slimmable_consumer`).
  - No thread de áudio, em [commands.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/rt_callback/commands.rs) (no payload `ParamPayload::SetOversample(factor)`):
    - Obter o fator numérico correspondente, armazenar em `rt_status.requested_os_factor` e sinalizar o flag `RT_STATUS_NEEDS_OS_REBUILD`.
  - Na thread principal ([run.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/run.rs) loop), escutar o flag `RT_STATUS_NEEDS_OS_REBUILD`:
    - Construir instâncias de `OversampleEngine` L e R com o fator solicitado (usando `OversampleEngine::new(factor, MAX_RESAMP_BUF)`).
    - Enviar as engines através do canal SPSC.
    - Limpar o flag `RT_STATUS_NEEDS_OS_REBUILD`.
  - No thread de áudio ([setup.rs](file:///home/fabio/nam-rs/src/standalone/pw_host/capture/setup.rs) loop), consumir o canal SPSC, realizar o swap em `CaptureState` e descartar as engines antigas enviando-as ao Garbage Collector (padrão `drain_gc_channels`).
- **Validação:**
  - Garantir que a alternância do oversampling em runtime não causa xruns nem alocações de heap na thread de áudio (auditado com `tests/nam_infer_test.rs` zero-alloc guards).
  - Executar os testes unitários e de integração de fidelidade espectral.

---

### Sprint α2 — Exposição do Controle de Precisão de Ativação (I1)

Este sprint expõe a infraestrutura de `ActivationPrecision::HighFidelity` (que hoje existe no código, mas só é chamada em testes) para o usuário final por meio da linha de comando do standalone (CLI) e de parâmetros adicionais no plugin CLAP.

#### [x] Tarefa α2.1 — Exposição na CLI (Standalone) [BAIXO RISCO]

- **Descrição:** Adicionar as opções de CLI `--activation <standard|hf>` (e o atalho `--act`) ao parser de linha de comando do standalone para configurar a precisão global no bootstrap.
- **Mudanças propostas:**
  - Modificar [cli.rs](file:///home/fabio/nam-rs/src/standalone/cli.rs) para incluir a nova flag e parsear como o respectivo enum `ActivationPrecision`.
  - No bootstrap em [main.rs](file:///home/fabio/nam-rs/src/main.rs), chamar `set_activation_precision(...)` antes da inicialização do PipeWire para aplicar o modo selecionado.
- **Validação:** Executar o standalone com `--activation hf` e confirmar que o modo de alta fidelidade é ativado.

#### [x] Tarefa α2.2 — Adição de Parâmetro e GUI no CLAP (Plugin) [BAIXO RISCO] ✅

- **Descrição:** Expor o controle de precisão de ativação no plugin CLAP sob o identificador `PARAM_ACTIVATION = 8`.
- **Mudanças propostas:**
  - Declarar `PARAM_ACTIVATION = 8` em `src/clap/extensions/params/mod.rs` ou `main.rs`.
  - Mapear o valor do parâmetro (0 -> "Standard", 1 -> "HighFidelity").
  - Incluir `param_activation: AtomicU32` em `UiToRt` (`src/clap/plugin/shared.rs`).
  - No thread de áudio (`src/clap/processor/events.rs`), monitorar a mudança de `PARAM_ACTIVATION` e chamar `set_activation_precision(...)` na borda do bloco (no flush de parâmetros), sem reconstruir o modelo completo.
  - Implementar a persistência do parâmetro em `state.rs` e o respectivo widget de controle visual na GUI (`zones/controls.rs`).
- **Validação:**
  - Mudar o parâmetro via GUI e constatar a alteração de comportamento no thread de processamento sem instabilidade de áudio.
  - Testar o comportamento em render offline (Offline render força HighFidelity e desliga adaptativo).
- **Conclusão (2026-06-29):**
  - `ActivationPrecision` ganhou `Serialize, Deserialize`, `from_f32()` e `to_f32()` (`src/math/activations/mod.rs:57`).
  - `PARAM_ACTIVATION = 8` declarado (`src/clap/extensions/params/mod.rs:26`) com helpers `activation_u32_to_enum` / `activation_enum_to_u32`.
  - `param_activation: AtomicU32` adicionado a `UiToRt` (`src/clap/plugin/shared.rs:114`).
  - Wiring completo nos 3 caminhos de eventos: Host Events (`set_activation`), GUI sync (`sync_activation_from_gui`) e SPSC (`apply_params_from_spsc`) em `src/clap/processor/params.rs`.
  - `process_events()` em `events.rs` chama `set_activation_precision()` no bloco, incluindo override de offline render que força HighFidelity.
  - Parâmetro registrado no `PluginMainThreadParams` (count=9, info/value/display/flush) em `main.rs`.
  - `PluginAudioProcessorParams::flush()` (audio.rs) e `write_gui_events()` (shared.rs) atualizados.
  - Persistência: `state.rs` e `state_context.rs` salvam/restauram `activation_precision` via envelope v1 (compatível, campo opcional `#[serde(default)]`).
  - `snapshot_params()` em `main_thread/mod.rs` sincroniza do atômico.
  - GUI: widget segmented "Activation" (Standard / HF) adicionado em `zones/controls.rs`, abaixo do controle de Oversampling, com indicação de DAW mapping.
  - Arrays `param_indication`/`param_indication_color` expandidos de 8→9 elementos (`ColdShared`, `plugin/mod.rs`, `shared_test.rs`).
  - `NamPluginParams` e `RtPluginParams` ganharam campo `activation_precision` com default `Standard`.
  - Testes existentes (shared_test layout cache-line, state_test v0/v1 round-trip, state_context_test, processor_state_test, activation_precision 9 tests) todos passam.

#### [x] Tarefa α2.3 — Testes de Integração e Medições de Zero Alloc [BAIXO RISCO] ✅

- **Descrição:** Escrever testes de integração e validar as garantias de latência e tempo real.
- **Validação específica:**
  - Adicionar teste em `tests/activation_precision.rs` simulando o fluxo de controle CLI/CLAP.
  - Verificar se a alternância de ativação não dispara o `CountingAllocator` (nenhuma alocação ocorre na troca).
  - Documentar explicitamente em `architecture.md` e `audio_fidelity_map.md` que os modelos LSTM ignoram temporariamente este controle até a entrega do Épico β (I6).
- **Conclusão (2026-06-29):**
  - **5 novos testes** adicionados em `tests/activation_precision.rs`:
    - `test_zero_alloc_activation_switch_primitive`: confirma que `set_activation_precision()` (AtomicStore) é zero-alloc (RT-safety F9).
    - `test_zero_alloc_activation_hot_path_switch`: confirma zero-alloc durante alternância mid-stream com inferência (WaveNet + LSTM).
    - `test_zero_alloc_cli_activation_flow` (standalone): simula parse de `--activation` + apply, zero-alloc.
    - `test_activation_switch_output_idempotent`: confirma saída finita após switch mid-stream.
    - `test_clap_pattern_block_boundary_activation_switch`: simula o padrão CLAP de switch na borda de bloco (PARAM_ACTIVATION=8 pendente α2.2), zero-alloc.
  - `docs/architecture.md:87`: documentado que LSTM ignora o modo até Épico β (I6).
  - `docs/audio_fidelity_map.md`: atualizado de "not user-exposed" para "CLI ✓, CLAP pending α2.2" + nota LSTM/I6.
  - **Nota para α2.2:** O caminho global `set_activation_precision()` está validado como RT-safe; a wiring de PARAM_ACTIVATION=8 em `events.rs` pode simplesmente chamar esta função na borda do bloco sem preocupação com alocações.

---

## Épico β — Fidelidade do §3 (LSTM), reordenado pela evidência

**Objetivo:** Levar a precisão de ativação HighFidelity para os gates fundidos do LSTM para resolver a mitigação do drift recorrente (§3), aplicando Kahan no head como higiene numérica e caracterizando o impacto espectral real do oversampling no LSTM.
**Risco:** MÉDIO-ALTO (alteração de kernels matemáticos e SIMD do LSTM).
**Origem dos achados:** I4, I5 e I6 do [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md).

### Épico β — Dependências e Sequência de Execução

Como o Épico β herda o controle global de precisão de ativação de I1, assume-se que as tarefas de controle e interface do Épico α já estão concluídas. O trabalho é estruturado em três sprints sequenciais:

1. **Sprint β1: Ativações HighFidelity no LSTM (Foco em I6)**
2. **Sprint β2: Acumulação Kahan no Head do LSTM (Foco em I4)**
3. **Sprint β3: Caracterização de Oversampling e Documentação (Foco em I5 e docs)**

---

### Sprint β1 — Ativações HighFidelity nos Kernels do LSTM (I6)

Este sprint leva o modo `ActivationPrecision::HighFidelity` (que usa aproximações polinomiais de alta precisão baseadas em exp) para as células e gates do LSTM, eliminando o erro da aproximação Padé rápida (default) em cenários de alta fidelidade.

#### [x] Tarefa β1.1 — Caminho Escalar HighFidelity nos Gates [MÉDIO RISCO]

- **Descrição:** Implementar o suporte a ativações HF no loop de fallback escalar do LSTM gates, lendo a flag `activation_precision()`.
- **Mudanças propostas:**
  - Em `src/math/lstm/gates.rs` (no fallback de `fused_lstm_gates_dyn_avx2` e `fused_lstm_gates_dyn_avx512`), desviar as chamadas de ativação: se `activation_precision() == ActivationPrecision::HighFidelity`, usar `scalar_sigmoid_poly` e `scalar_tanh_poly` em vez de `scalar_minimax_sigmoid` e `scalar_pade_tanh`.
  - No loop escalar de `process_sample_scalar` em `src/models/lstm/layer_kernels.rs:248-266`, aplicar a mesma lógica de desvio.
- **Validação:** Rodar o oráculo f64 (`tests/reference_oracle_f64.rs`) e certificar-se de que a ESR cai significativamente no modo HF.
- **Conclusão (2026-06-29):**
  - `src/math/activations/tanh/high_fidelity.rs:318-356`: stubs `scalar_tanh_poly`/`scalar_sigmoid_poly` substituídos por implementações exp-based reais (degree-6 Taylor com range reduction), emparelhadas com os kernels SIMD (`simd_exp_poly_avx2`). Erro máximo: ≤ 2.4e-7 (tanh), ≤ 2.1e-7 (sigmoid). `scalar_exp_poly_inner` usa `f64::round_ties_even` (ties-to-even, parity com `_MM_FROUND_TO_NEAREST_INT`) e não faz double-clamp.
  - `src/math/lstm/gates.rs`: ambos `fused_lstm_gates_dyn_avx2` e `fused_lstm_gates_dyn_avx512` scalar fallbacks agora desviam com branch direto (`if is_hf { ... scalar_tanh_poly(new_cs) } else { ... scalar_pade_tanh(new_cs) }`), sem function pointer. Flag `is_hf` hoisted antes do `while`.
  - `src/models/lstm/layer_kernels.rs:252-284`: `process_sample_scalar` aplica mesma lógica com `is_hf` hoisted e branch direto.
  - `src/models/lstm/layer_dyn_kernels.rs:57-88`: `LstmLayerDyn::process_sample_scalar` também ganhou dispatch HF (originalmente esquecido).
  - Todos os testes passam: `test_oracle_lstm`, `test_decomposition_lstm`, `test_lstm_activation_precision_gain`, 16 unit tests LSTM, 8 dynamic LSTM validation tests. Clippy limpo.
  - **Nota para β1.2:** O caminho SIMD principal (AVX2/AVX-512) em `fused_lstm_gates_avx2`/`fused_lstm_gates_avx512` **ainda não lê** `activation_precision()` — usa sempre Padé/minimax. A validação do ESR em modo HF no oráculo f64 só será observável após β1.2 (SIMD HF). A infraestrutura de desvio escalar está pronta e funcionando como fundação.

#### [ ] Tarefa β1.2 — Kernels SIMD Fundidos HF (AVX2 e AVX512) [ALTO RISCO]

- **Descrição:** Adicionar suporte a SIMD HighFidelity nos kernels fundidos de 4 gates do LSTM.
- **Mudanças propostas:**
  - Em `src/math/lstm/gates.rs`, modificar `fused_lstm_gates_avx2` e `fused_lstm_gates_avx512` para avaliar `activation_precision()`.
  - Se `HighFidelity` estiver ativo, despachar para implementações polinomiais de alta fidelidade reusando os kernels exp/sigmoid/tanh poly de `high_fidelity.rs`.
- **Validação:** Confirmar que não ocorrem alocações de heap no hot path e que os ganhos de ESR se estendem ao processamento SIMD.

#### [ ] Tarefa β1.3 — Paridade ISA e Recalibração de Gates [MÉDIO RISCO]

- **Descrição:** Garantir a exatidão matemática entre os caminhos escalar e SIMD e calibrar limites de paridade.
- **Mudanças propostas:**
  - Garantir que `tests/isa_parity.rs` passa sem divergência no modo `HighFidelity`.
  - Caracterizar a divergência interop vs C++ NAMCore em `tests/cpp_parity.rs` no modo HF. Se necessário, ajustar ou documentar a recalibração do gate `ABSOLUTE_ESR_CAP_LSTM`.

---

### Sprint β2 — Higiene Numérica com Kahan no Head do LSTM (I4)

Este sprint adiciona soma compensada de Kahan ao head f32-native do LSTM como proteção contra o acúmulo de erros de arredondamento em heads longos.

#### [ ] Tarefa β2.1 — Implementação da Função de Soma Compensada [BAIXO RISCO]

- **Descrição:** Adicionar suporte a produto escalar compensado nativo f32.
- **Mudanças propostas:**
  - Em `src/math/common/scalar_ref/dot.rs`, implementar `dot_product_f32_native_kahan(a: &[f32], b: &[f32]) -> f32` reusando `KahanF32` de `src/math/common/kahan.rs`.
- **Validação:** Testes unitários para validar a precisão matemática da soma.

#### [ ] Tarefa β2.2 — Integração da Acumulação nos Modelos LSTM [BAIXO RISCO]

- **Descrição:** Substituir a acumulação ingênua pelo produto escalar de Kahan nos heads dos modelos LSTM.
- **Mudanças propostas:**
  - Em `src/models/lstm/model1.rs`, `model2.rs` e `model_dyn.rs`, alterar as chamadas de `dot_product_f32_native` para `dot_product_f32_native_kahan` quando `use_f32_head` for verdadeiro.
- **Validação:** Compilar e verificar que todos os modelos LSTM executam corretamente.

#### [ ] Tarefa β2.3 — Validação de Estabilidade e Soak [BAIXO RISCO]

- **Descrição:** Validar que a adição do Kahan no head não introduz gargalos de CPU no thread de áudio nem quebra garantias de latência.
- **Validação específica:**
  - Executar `tests/soak_test.rs` de longa duração para garantir estabilidade e ausência de regressões.

---

### Sprint β3 — Caracterização e Sincronização de Documentação (I5)

Este sprint caracteriza o impacto de oversampling externo no LSTM e atualiza os mapas de fidelidade para refletir os findings reais.

#### [ ] Tarefa β3.1 — Experimentos de Caracterização de Oversampling [BAIXO RISCO]

- **Descrição:** Medir empiricamente o efeito de usar o `OversampleEngine` externo com modelos LSTM.
- **Atividades:**
  - Rodar testes para obter ASR (redução de aliasing) e ESR/MR-STFT (variação de timbre contra baseline de 48 k) com oversampling de Off vs 2x vs 4x.
  - Tabular os dados espectrais obtidos para inclusão na documentação.
- **Validação:** Confirmar cientificamente a hipótese de que o oversampling de LSTM serve a anti-aliasing mas muda o timbre (tonalidade) por não ajustar o atraso de realimentação.

#### [ ] Tarefa β3.2 — Sincronização da Documentação e Referências [BAIXO RISCO]

- **Descrição:** Atualizar a documentação do repositório com o status-alvo da arquitetura de fidelidade do LSTM.
- **Mudanças propostas:**
  - Atualizar `docs/audio_fidelity_map.md` (§3, §9) para estabelecer o I6 como mitigação primária de drift, I4 como higiene e I5 como anti-aliasing.
  - Modificar `docs/lstm_recurrent_drift.md` (§4, §7) incluindo a tabela empírica do oversampling e orientações ao usuário.
  - Adicionar notas pertinentes em `docs/architecture.md` e `docs/cpp_parity_map.md`.
  - Inserir as novas referências acadêmicas (Mikkonen & Werner 2025; Carson et al. 2024/2025) em `docs/research-references.md`.
