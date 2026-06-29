<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# Sprints de Implementação — nam-rs

Este documento organiza a resolução dos épicos descritos em [TODO-findings.md](TODO-findings.md) em Sprints e Tarefas Técnicas atômicas e de fácil acompanhamento.

---

## Épico A — Saneamento do Adaptive Compute para A2 (CRÍTICO)

* **Objetivo:** Eliminar a corrupção de estado/CPU em modelos WaveNet A2 sob adaptive compute e corrigir a perda de histórico de transições (`prev_state`) em crossfades rápidos concorrentes.
* **Complexidade Geral:** Média-Alta (toca o hot-path de inferência e a máquina de estados).
* **QA Requerido:** Testes unitários do FSM, testes rápidos da pipeline (`tests-quick.sh`).
* **Fontes de maiores informações:** TODO-findings.md seções "Épico A — Saneamento do Adaptive Compute para A2 (CRÍTICO)" e "Tabela de cruzamento".

---

### Sprint A1 — Saneamento do Adaptive Compute e Ajustes de Fidelidade

#### Tarefa A1.1 — Excluir a Arquitetura A2 do Mecanismo de Double-Pass (CRÍTICA) [DONE]

* **Tipo:** Correção de Bug (Inference Hot-Path)
* **Severidade:** ALTA
* **Risco:** ALTO (toca a lógica principal de inferência em tempo real)
* **Finding de Origem:** [TODO-findings.md F1](TODO-findings.md#f1--alta-a2--adaptive-compute-o-crossfade-de-soft-degrade-corrompe-o-estado-recorrente-do-modelo-double-pass-sem-backuprestore)
* **Arquivos Relacionados:**
  * [src/models/static_model.rs](src/models/static_model.rs) (impl `StaticModel`)
  * [src/dsp/pipeline/stages/inference.rs](src/dsp/pipeline/stages/inference.rs) (função `run_inference_stereo_or_mono`)
* **Detalhamento Técnico:**
  1. No enum `StaticModel` ([src/models/static_model.rs](src/models/static_model.rs)), declarar um método auxiliar:

     ```rust
     #[inline(always)]
     pub fn supports_layer_skip(&self) -> bool {
         matches!(
             self,
             Self::WavenetStandard(_)
                 | Self::WavenetLite(_)
                 | Self::WavenetFeather(_)
                 | Self::WavenetNano(_)
                 | Self::WavenetDyn(_)
         )
     }
     ```

  2. No arquivo [src/dsp/pipeline/stages/inference.rs](src/dsp/pipeline/stages/inference.rs), alterar a verificação que determina se o crossfade de WaveNet deve rodar:
     * Obter `supports_skip` chamando `supports_layer_skip()` no modelo ativo (`ctx.active_model_l.as_ref()`).
     * Substituir a linha `let is_crossfading_wavenet = is_wavenet && ctx.adaptive.is_crossfading();` por `let is_crossfading_wavenet = supports_skip && ctx.adaptive.is_crossfading();`.
* **Critérios de Aceitação:**
  * Modelos WaveNet A2 (`WavenetA2Full`, `WavenetA2Lite`, `WavenetA2Dyn`) processam áudio em caminho único sem entrar no trecho de `double-pass` (evitando corrupção do histórico de buffers).
  * O WaveNet A1 (`Standard`, `Lite`, etc.) continua executando o `double-pass` normalmente durante transições de qualidade.

---

#### Tarefa A1.2 — Evitar Dessincronização de `prev_state` em Transições Rápidas [DONE]

* **Tipo:** Correção de Bug (State Machine)
* **Severidade:** BAIXA-MÉDIA
* **Risco:** MÉDIO (toca a consistência de estado do FSM)
* **Finding de Origem:** [TODO-findings.md F4](TODO-findings.md#f4--baixa-média-adaptive-prev_state-dessincroniza-em-transições-encadeadas-dentro-de-um-crossfade)
* **Arquivos Relacionados:**
  * [src/dsp/adaptive.rs](src/dsp/adaptive.rs) (método `transition_to`)
* **Detalhamento Técnico:**
  1. No arquivo [src/dsp/adaptive.rs](src/dsp/adaptive.rs), no método `transition_to` (linha ~362), impedir que `prev_state` seja atualizado caso já exista um crossfade ativo (`Active`):

     ```rust
     if !matches!(self.crossfade, CrossfadePhase::Active) {
         self.prev_state = self.state;
     }
     self.state = new_state;
     ```

* **Critérios de Aceitação:**
  * Se ocorrer uma transição de estado enquanto a fase de crossfade ainda estiver `Active` (por exemplo, `Full -> Reduced` seguido de `Reduced -> Minimal` em menos de 32 ms), o `prev_state` do FSM deve manter o valor da origem inicial (`Full`), em vez de ser incorretamente sobrescrito para `Reduced`.
  * ✅ **Conclusão (Sprint A1, 2026-06-29):** Implementado. A guarda `if !matches!(self.crossfade, CrossfadePhase::Active)` em `transition_to` impede sobrescrita de `prev_state` durante crossfades ativos. Teste `crossfade_rebased_on_rapid_consecutive_transitions` ajustado para validar o novo comportamento (`prev_state` mantém `Full` após transição encadeada).

---

#### Tarefa A1.3 — Atualização do Estado de Questões nas Documentações (Docs) [DONE]

* **Tipo:** Ajuste de Documentação e Higiene de Referências
* **Severidade:** INFORMATIVA
* **Risco:** MÍNIMO (apenas alteração de documentação)
* **Findings de Origem:** [TODO-findings.md § Tabela de Cruzamento](TODO-findings.md#tabela-de-cruzamento), [F1 - Cruzamento](TODO-findings.md#f1--detalhamento-do-cruzamento-com-audio_fidelity_mapmd-8), [F2/F3/F9 - Detalhamento](TODO-findings.md#f2--detalhamento-do-cruzamento-com-audio_fidelity_mapmd-5-e-9)
* **Arquivos Relacionados:**
  * [docs/audio_fidelity_map.md](docs/audio_fidelity_map.md) (§8, §9)
  * [docs/cpp_parity_map.md](docs/cpp_parity_map.md) (§13, links de referências)
* **Detalhamento Técnico:**
  1. No arquivo [docs/audio_fidelity_map.md](docs/audio_fidelity_map.md):
     * Atualizar a seção **§8. Adaptive Compute** para explicar explicitamente que modelos WaveNet A2 não suportam salto de camadas e, portanto, foram excluídos do double-pass por segurança e integridade de áudio.
     * Na tabela de **§9. Pending / Open Work**, marcar as linhas referentes aos bugs **F1 (Double-pass A2)** e **F4 (prev_state desinc)** como `✅ Resolvido (Sprint A1)`.
  2. No arquivo [docs/cpp_parity_map.md](docs/cpp_parity_map.md):
     * Na tabela de **§13. Pending / Open Work**, atualizar a linha referente ao bug **F1 (Double-pass A2)** como `✅ Resolvido (Sprint A1)`.
     * Atualizar as referências cruzadas quebradas de `F-2` nas linhas 163, 524 e 545. A string `F-2` agora se refere a latência do oversampling, portanto, a antiga documentação sobre drift do LSTM a 192 kHz deve apontar internamente para as seções **§4.5 e §9.1** do próprio mapa de paridade C++.
* **Critérios de Aceitação:**
  * As tabelas de pendências abertas de ambos os mapas documentais mostram os bugs de Adaptive Compute de A2 marcados de forma transparente como resolvidos.
  * Links e referências internas em `cpp_parity_map.md` não estão mais quebrados ou contraditórias.

---

#### Tarefa A1.4 — Validação dos Testes Unitários de Integração de Estado [DONE]

* **Tipo:** QA e Validação de Regressão
* **Severidade:** GARANTIA DE QUALIDADE
* **Risco:** BAIXO
* **Arquivos Relacionados:**
  * [src/dsp/adaptive_test.rs](src/dsp/adaptive_test.rs)
* **Detalhamento Técnico:**
  1. Adicionar um teste unitário dedicado em [src/dsp/adaptive_test.rs](src/dsp/adaptive_test.rs) que forçará simulações de transições encadeadas e atestará que `prev_state` se comporta de forma invariante sob crossfades ativos.
  2. Executar `utils/tests-quick.sh` e verificar se a compilação, formatação, clippy e todos os testes rápidos do projeto passam sem warnings ou erros.
* **Critérios de Aceitação:**
  * Execução bem-sucedida do script `tests-quick.sh` com zero falhas na suite de testes do projeto.

---

## Épico B — Reporte de latência correto para PDC

* **Objetivo:** O host receber a latência real do plugin (soma do resampler, cab-sim e oversampling), com medição e reporte corretos do atraso de grupo do resampler sob fase mínima.
* **Complexidade Geral:** Média
* **QA Requerido:** Testes unitários do resampler e do oversampling, validação de latência dinâmica no plugin CLAP.
* **Fontes de maiores informações:** TODO-findings.md seções "Épico B — Reporte de latência correto para PDC" e "Tabela de cruzamento".

---

### Sprint B1 — Reporte de Latência Correto e Calibração de Fase Mínima

#### Tarefa B1.1 — Reportar a Latência do `OversampleEngine` (F2) (CRÍTICA) [DONE]

* **Tipo:** Correção de Bug / Melhoria de DSP
* **Severidade:** MÉDIA
* **Risco:** MÉDIO (toca o cálculo de latência exposto ao host)
* **Finding de Origem:** [TODO-findings.md F2](TODO-findings.md#f2--média-oversampling-latência-do-engine-não-é-reportada-ao-host-pdc)
* **Arquivos Relacionados:**
  * [src/dsp/oversample.rs](src/dsp/oversample.rs) (impl `OversampleEngine`)
  * [src/clap/processor/events.rs](src/clap/processor/events.rs) (função `effective_latency` check)
  * [src/clap/processor/mod.rs](src/clap/processor/mod.rs) (cálculo de `initial_latency`)
* **Detalhamento Técnico:**
  1. No arquivo [src/dsp/oversample.rs](src/dsp/oversample.rs), implementar o método público:

     ```rust
     #[inline]
     pub fn latency_samples(&self) -> usize {
         match self.factor {
             OversampleFactor::Off => 0,
             OversampleFactor::X2 => HB_DELAY,
             OversampleFactor::X4 => 2 * HB_DELAY,
         }
     }
     ```

  2. No arquivo [src/clap/processor/events.rs](src/clap/processor/events.rs), na monitoração dinâmica de latência (`let mut effective_latency = self.resampler.latency_samples(host_rate);`), somar a latência do oversampling:

     ```rust
     effective_latency += self.os_l.latency_samples() as u32;
     ```

  3. No arquivo [src/clap/processor/mod.rs](src/clap/processor/mod.rs), na ativação inicial do plugin (`let mut initial_latency = resampler.latency_samples(audio_config.sample_rate as u32);`), somar a latência do oversampling:

     ```rust
     initial_latency += os_l.latency_samples() as u32;
     ```

* **Critérios de Aceitação:**
  * O método `OversampleEngine::latency_samples` retorna `0`, `12` ou `24` para as configurações `Off`, `X2` e `X4`, respectivamente.
  * A latência inicial e as mudanças em tempo de execução incluem corretamente a contribuição do oversampling.

---

#### Tarefa B1.2 — Medir e Rastrear o Atraso de Grupo (Centróide) do Resampler de Fase Mínima (F3) (CRÍTICA) [DONE]

* **Tipo:** Correção de Bug / Melhoria de DSP
* **Severidade:** MÉDIA-BAIXA
* **Risco:** MÉDIO (altera a latência reportada sob fase mínima)
* **Finding de Origem:** [TODO-findings.md F3](TODO-findings.md#f3--média-baixa-resampler-latência-superestimada-para-bancos-de-fase-mínima)
* **Arquivos Relacionados:**
  * [src/dsp/sinc_kernel.rs](src/dsp/sinc_kernel.rs) (banco de filtros e `PolyphaseBank`)
  * [src/dsp/resampler.rs](src/dsp/resampler.rs) (`ResamplerCore` e `NamResampler`)
* **Detalhamento Técnico:**
  1. No arquivo [src/dsp/sinc_kernel.rs](src/dsp/sinc_kernel.rs):
     * Declarar o enum de controle:

       ```rust
       #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
       pub enum PhaseType {
           Linear,
           Minimum,
       }
       ```

     * Adicionar os campos `pub group_delay: f64` e `pub phase_type: PhaseType` ao struct `PolyphaseBank`.
     * Criar a função auxiliar `calculate_centroid(h: &[f64]) -> f64` que calcula a média ponderada do tempo (centróide de energia) do kernel prototype:

       ```rust
       fn calculate_centroid(h: &[f64]) -> f64 {
           let mut num = 0.0;
           let mut den = 0.0;
           for (n, &val) in h.iter().enumerate() {
               let energy = val * val;
               num += n as f64 * energy;
               den += energy;
           }
           if den > 1e-30 { num / den } else { 0.0 }
       }
       ```

     * Em `generate_polyphase_bank` (fase mínima), obter o centróide do kernel `min_phase` e definir `group_delay = centroid / NUM_PHASES as f64`. E definir `phase_type = PhaseType::Minimum`.
     * Em `generate_polyphase_bank_linear` (fase linear), definir `group_delay = TAPS_PER_PHASE as f64 / 2.0` (exatamente `32.0`). E definir `phase_type = PhaseType::Linear`.
  2. No arquivo [src/dsp/resampler.rs](src/dsp/resampler.rs):
     * Adicionar os campos `group_delay: f64` e `phase_type: PhaseType` à estrutura `ResamplerCore`, inicializando-os a partir do `PolyphaseBank` correspondente.
     * Implementar getters em `ResamplerCore`: `pub fn group_delay(&self) -> f64` e `pub fn phase_type(&self) -> PhaseType`.
     * Alterar o cálculo em `NamResampler::latency_samples` para somar a latência das fases `inner` e `outer` fisicamente de forma independente e com taxas corretas:

       ```rust
       let delay_in = match self.inner {
           Some(ref core) => core.group_delay(),
           None => 0.0,
       };
       let delay_out = match self.outer {
           Some(ref core) => core.group_delay() * (self.pw_rate as f64 / self.nam_rate as f64),
           None => 0.0,
       };
       (delay_in + delay_out).round() as u32
       ```

* **Critérios de Aceitação:**
  * O resampler de fase mínima calcula e reporta o atraso de grupo empírico real (centróide de energia), resultando em menor latência do que na fase linear.
  * O resampler de fase linear continua reportando o atraso teórico de `TAPS_PER_PHASE / 2`.
  * Conversão de taxas na etapa de saída correta e consistente com a latência observada.

---

#### Tarefa B1.3 — Atualização do Estado de Questões nas Documentações (Docs) [DONE]

* **Tipo:** Ajuste de Documentação e Higiene
* **Severidade:** INFORMATIVA
* **Risco:** MÍNIMO (apenas alteração de documentação)
* **Findings de Origem:** [TODO-findings.md § Tabela de Cruzamento](TODO-findings.md#tabela-de-cruzamento), [F2/F3 - Detalhamento](TODO-findings.md#f2--detalhamento-do-cruzamento-com-audio_fidelity_mapmd-5-e-9)
* **Arquivos Relacionados:**
  * [docs/audio_fidelity_map.md](docs/audio_fidelity_map.md) (§5, §9)
* **Detalhamento Técnico:**
  1. No arquivo [docs/audio_fidelity_map.md](docs/audio_fidelity_map.md):
     * Atualizar a seção **§5. Oversampling** para indicar que a latência (12 amostras a 2×, 24 amostras a 4×) agora é rastreada e reportada ao host.
     * Na tabela de **§9. Pending / Open Work**, marcar as linhas referentes aos bugs **F2 (Oversampling PDC)** e **F3 (Resampler min-phase latency)** como `✅ Resolvido (Sprint B1)`.
* **Critérios de Aceitação:**
  * O mapa de fidelidade de áudio descreve com precisão a resolução do reporte empírico e dinâmico de latências do resampler e do oversampling.

---

#### Tarefa B1.4 — Validação dos Testes Unitários de Latência e Paridade [DONE]

* **Tipo:** QA e Validação de Regressão
* **Severidade:** GARANTIA DE QUALIDADE
* **Risco:** BAIXO
* **Arquivos Relacionados:**
  * [src/dsp/resampler_test.rs](src/dsp/resampler_test.rs)
  * [src/dsp/oversample_test.rs](src/dsp/oversample_test.rs)
* **Detalhamento Técnico:**
  1. Adicionar/ajustar os testes em `src/dsp/resampler_test.rs` para testar que a latência calculada sob fase mínima é inferior à da fase linear correspondente.
  2. Adicionar testes unitários em `src/dsp/oversample_test.rs` para o método `latency_samples`.
  3. Executar o pipeline QA rápido (`utils/tests-quick.sh`).
* **Critérios de Aceitação:**
  * Todos os testes novos passam com sucesso.
  * Execução bem-sucedida de `tests-quick.sh` com zero falhas.

---

## Épico C — Endurecimento de robustez (loader e máquina de estados de swap)

* **Objetivo:** Garantir que os loaders sejam à prova de falhas ("never panic") com aritmética checada, flags de swap consistentes na máquina de estados de tempo real, e guardas defensivas no oversampling.
* **Complexidade Geral:** Baixa
* **QA Requerido:** Testes unitários do WeightCursor com overflows artificiais e verificação rápida da suíte (`tests-quick.sh`).
* **Fontes de maiores informações:** TODO-findings.md seções "Épico C — Endurecimento de robustez (loader e máquina de estados de swap)" e "Tabela de cruzamento".

---

### Sprint C1 — Endurecimento de Robustez (Loader, Swap e Oversampling)

#### Tarefa C1.1 — Evitar Overflow de Aritmética no `WeightCursor` (F5) (CRÍTICA) [DONE]

* **Tipo:** Correção de Bug / Segurança
* **Severidade:** BAIXA (Cold path / Loader)
* **Risco:** BAIXO
* **Finding de Origem:** [TODO-findings.md F5](TODO-findings.md#f5--baixa-weightcursor-selfpos--len-pode-estourar-usize-release-e-burlar-a-checagem-de-limites)
* **Arquivos Relacionados:**
  * [src/loader/dispatcher/weight_cursor.rs](src/loader/dispatcher/weight_cursor.rs)
* **Detalhamento Técnico:**
  1. No arquivo [src/loader/dispatcher/weight_cursor.rs](src/loader/dispatcher/weight_cursor.rs), atualizar o método `read_slice` para usar aritmética checada contra estouro de `usize` em builds de release:

     ```rust
     pub(crate) fn read_slice(&mut self, len: usize) -> anyhow::Result<&'a [f32]> {
         let end = self.pos.checked_add(len)
             .filter(|&e| e <= self.data.len())
             .ok_or_else(|| anyhow::anyhow!(
                 "Insufficient weights: required {} starting from position {}, available {}",
                 len,
                 self.pos,
                 self.data.len()
             ))?;
         let slice = &self.data[self.pos..end];
         self.pos = end;
         Ok(slice)
     }
     ```

* **Critérios de Aceitação:**
  * O método `read_slice` retorna um erro de forma limpa caso a adição `self.pos + len` resulte em overflow, em vez de passar na checagem e panicar no fatiamento subsequente.
* **Conclusão (2026-06-29):** ✅ Implementado. Substituído `self.pos + len` por `checked_add` com filtro de bounds. 1028 testes passam, sem warnings. O `bail!` permanece importado para outros métodos (`read_f32_finite`, `verify_exhausted`).

---

#### Tarefa C1.2 — Usar Aritmética Checada nos Dimensionamentos do Loader de LSTM Dinâmico (F5) [DONE]

* **Tipo:** Melhoria de Robustez
* **Severidade:** BAIXA
* **Risco:** BAIXO
* **Finding de Origem:** [TODO-findings.md F5](TODO-findings.md#f5--baixa-weightcursor-selfpos--len-pode-estourar-usize-release-e-burlar-a-checagem-de-limites)
* **Arquivos Relacionados:**
  * [src/loader/dispatcher/lstm/weights.rs](src/loader/dispatcher/lstm/weights.rs)
* **Detalhamento Técnico:**
  1. Importar `use crate::loader::dispatcher::checked_arith;` no arquivo [src/loader/dispatcher/lstm/weights.rs](src/loader/dispatcher/lstm/weights.rs).
  2. Na função `read_lstm_layer_dyn`, atualizar o cálculo de dimensões para usar as funções auxiliares em `checked_arith`:

     ```rust
     let ih = checked_arith::checked_add(input_size, hidden_size)?;
     let h4 = checked_arith::checked_mul(4, hidden_size)?;
     let weights_len = checked_arith::checked_mul(h4, ih)?;
     ```

* **Critérios de Aceitação:**
  * Cálculos de tamanho de pesos do LSTM dinâmico não causam pânico ou wrapping silencioso em builds de release.
* **Conclusão (2026-06-29):** ✅ Implementado. Substituído `input_size + hidden_size`, `4 * hidden_size`, e `h4 * ih` por chamadas a `checked_arith::checked_add`/`checked_mul` na função `read_lstm_layer_dyn`. 1028 testes passam, sem warnings, clippy limpo.

---

#### Tarefa C1.3 — Resetar a Flag `RESAMPLER_REBUILD_FAILED` no Swap RT (F6) [DONE]

* **Tipo:** Correção de Bug (State Machine)
* **Severidade:** BAIXA
* **Risco:** BAIXO
* **Finding de Origem:** [TODO-findings.md F6](TODO-findings.md#f6--baixa-flag-resampler_rebuild_failed-permanece-setada-após-uma-falha-desabilitando-a-espera-pelo-resampler-em-swaps-futuros)
* **Arquivos Relacionados:**
  * [src/standalone/pw_host/capture/setup.rs](src/standalone/pw_host/capture/setup.rs)
* **Detalhamento Técnico:**
  1. No arquivo [src/standalone/pw_host/capture/setup.rs](src/standalone/pw_host/capture/setup.rs), no ponto em que se limpa a flag `RESAMP_SWAP_PENDING` por conta da falha de rebuild anterior, limpar também a flag `RESAMPLER_REBUILD_FAILED`:

     ```rust
     if rt_status_for_process.check_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING) {
         if rt_status_for_process.check_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED) {
             rt_status_for_process.clear_flag(crate::common::spsc::RT_STATUS_RESAMP_SWAP_PENDING);
             rt_status_for_process.clear_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED);
         } else {
             let _ = stream.dequeue_buffer();
             return;
         }
     }
     ```

* **Critérios de Aceitação:**
  * Se um rebuild falhar e marcar `REBUILD_FAILED`, no próximo ciclo de swap o RT thread consumirá a flag de falha e a resetará. Swaps futuros voltarão a aguardar o resampler normalmente (não serão curto-circuitados).
* **Conclusão (2026-06-29):** ✅ Implementado. Adicionada chamada `rt_status_for_process.clear_flag(crate::common::spsc::RT_STATUS_RESAMPLER_REBUILD_FAILED)` logo após a limpeza de `RESAMP_SWAP_PENDING` no bloco de falha, garantindo que swaps futuros voltem a aguardar o resampler normalmente. 1028 testes passam, sem warnings, clippy limpo.

---

#### Tarefa C1.4 — Adicionar Guardas de Tamanho de Buffer no `OversampleEngine` (F7) [DONE]

* **Tipo:** Melhoria Defensiva
* **Severidade:** BAIXA (Defensivo)
* **Risco:** BAIXO
* **Finding de Origem:** [TODO-findings.md F7](TODO-findings.md#f7--baixa-oversampleengineupsample-escreve-com-get_unchecked_mut-sem-debug_assert-do-tamanho-de-output)
* **Arquivos Relacionados:**
  * [src/dsp/oversample.rs](src/dsp/oversample.rs)
* **Detalhamento Técnico:**
  1. No arquivo [src/dsp/oversample.rs](src/dsp/oversample.rs), adicionar `debug_assert!` nas funções `upsample` e `downsample` para validar o tamanho dos buffers de saída recebidos contra o multiplicador correspondente do fator de sobreamostragem:
     * Em `upsample`:

       ```rust
       debug_assert!(
           output.len() >= input.len() * self.factor.multiplier(),
           "oversample: output buffer too small for upsampling factor"
       );
       ```

     * Em `downsample`:

       ```rust
       debug_assert!(
           output.len() >= input.len() / self.factor.multiplier(),
           "oversample: output buffer too small for downsampling factor"
       );
       ```

* **Critérios de Aceitação:**
  * Prevenção de escrita/leitura OOB (Out-Of-Bounds) em cenários futuros de alteração do chamador do DSP, garantido por asserções em tempo de depuração.
* **Conclusão (2026-06-29):** ✅ Implementado. Adicionados `debug_assert!` em `OversampleEngine::upsample` (valida `output.len() >= input.len() * factor.multiplier()`) e `OversampleEngine::downsample` (valida `output.len() >= input.len() / factor.multiplier()`). 11 testes passam, clippy e fmt limpos.

---

#### Tarefa C1.5 — Validação da Suíte de Robustez (QA) [DONE]

* **Tipo:** QA e Validação de Regressão
* **Severidade:** GARANTIA DE QUALIDADE
* **Risco:** BAIXO
* **Arquivos Relacionados:**
  * [src/loader/dispatcher/weight_cursor.rs](src/loader/dispatcher/weight_cursor.rs)
* **Detalhamento Técnico:**
  1. Adicionar testes unitários inline no final de [src/loader/dispatcher/weight_cursor.rs](src/loader/dispatcher/weight_cursor.rs) dentro de `#[cfg(test)] mod tests` para simular chamadas de `read_slice` com tamanhos abusivos (ex.: `usize::MAX`) ou que ultrapassam `data.len()`, atestando que retornam erro apropriado sem panicar.
  2. Executar `utils/tests-quick.sh` para atestar que a qualidade clippy e formatação se mantêm adequadas.
* **Critérios de Aceitação:**
  * Teste unitário compilando e passando com sucesso.
  * Pipeline rápida executando sem avisos ou falhas.

---

## Épico D — Conformidade e higiene

* **Objetivo:** Corrigir os identificadores SPDX inválidos e implantar uma verificação automatizada no pipeline de CI local para garantir a integridade do licenciamento.
* **Complexidade Geral:** Trivial
* **QA Requerido:** Execução do script [utils/lints.sh](file:///home/fabio/nam-rs/utils/lints.sh).
* **Fontes de maiores informações:** [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md) seções "Épico D — Conformidade e higiene" e "Tabela de cruzamento".

---

### Sprint D1 — Conformidade de Licença SPDX e CI

#### Tarefa D1.1 — Corrigir o Identificador SPDX no arquivo `oversample.rs` (F8) [DONE]

* **Tipo:** Higiene / Conformidade de Licenciamento
* **Severidade:** TRIVIAL
* **Risco:** MÍNIMO (apenas comentário de cabeçalho)
* **Finding de Origem:** [TODO-findings.md F8](file:///home/fabio/nam-rs/TODO-findings.md#f8--trivial-cabeçalho-spdx-incorreto-apache-22)
* **Arquivos Relacionados:**
  * [src/dsp/oversample.rs](file:///home/fabio/nam-rs/src/dsp/oversample.rs)
* **Detalhamento Técnico:**
  1. No arquivo [src/dsp/oversample.rs](file:///home/fabio/nam-rs/src/dsp/oversample.rs), substituir o identificador inválido no cabeçalho:

     ```rust
     // SPDX-License-Identifier: Apache-2.2
     ```

     pelo identificador válido do projeto:

     ```rust
     // SPDX-License-Identifier: Apache-2.0
     ```

* **Critérios de Aceitação:**
  * A primeira linha de [src/dsp/oversample.rs](file:///home/fabio/nam-rs/src/dsp/oversample.rs) contém o identificador correto: `// SPDX-License-Identifier: Apache-2.0`.

---

#### Tarefa D1.2 — Adicionar Verificação do Identificador SPDX em `utils/lints.sh` [DONE]

* **Tipo:** Ferramenta / QA Automatizado
* **Severidade:** TRIVIAL
* **Risco:** MÍNIMO (melhoria apenas de script local)
* **Finding de Origem:** [TODO-findings.md F8](file:///home/fabio/nam-rs/TODO-findings.md#f8--trivial-cabeçalho-spdx-incorreto-apache-22)
* **Arquivos Relacionados:**
  * [utils/lints.sh](file:///home/fabio/nam-rs/utils/lints.sh)
* **Detalhamento Técnico:**
  1. No arquivo [utils/lints.sh](file:///home/fabio/nam-rs/utils/lints.sh), adicionar uma nova etapa de validação automatizada que varre o diretório `src/` em busca de qualquer arquivo `.rs` que não contenha a assinatura de licença SPDX válida.
  2. Implementar a verificação de forma robusta e limpa usando ferramentas UNIX padrão (como `grep`). O script deve retornar código de erro `1` e detalhar quais arquivos falharam se:
     * O cabeçalho SPDX estiver ausente ou malformado.
     * Algum arquivo contiver um identificador SPDX que não seja `Apache-2.0`.
* **Critérios de Aceitação:**
  * A execução do script [utils/lints.sh](file:///home/fabio/nam-rs/utils/lints.sh) falha de forma visível caso um arquivo Rust contenha um identificador SPDX incorreto (como `Apache-2.2`) ou não tenha o cabeçalho SPDX.

---

## Épico E — Endurecimento latente da arquitetura A2 (Beta)

* **Objetivo:** Corrigir os potenciais estouros de memória (OOB) e pânico latentes nos atalhos de processamento e na camada FiLM da arquitetura A2, adicionando limites robustos e asserções de segurança em tempo de execução.
* **Complexidade Geral:** Baixa
* **QA Requerido:** Testes unitários do FSM, testes rápidos da pipeline (`tests-quick.sh`).
* **Fontes de maiores informações:** [TODO-findings.md](file:///home/fabio/nam-rs/TODO-findings.md) seções "Épico E — Endurecimento latente da arquitetura A2 (Beta)" e "Tabela de cruzamento".

---

### Sprint E1 — Endurecimento e Segurança de Limites em A2

#### Tarefa E1.1 — Aplicar Máscara de Anel no Atalho de Camadas Vazias de A2 (F9) (CRÍTICA) [DONE]

* **Tipo:** Correção de Bug Latente (Inference Engine)
* **Severidade:** BAIXA-MÉDIA
* **Risco:** BAIXO (caminho defensivo secundário)
* **Finding de Origem:** [TODO-findings.md F9](file:///home/fabio/nam-rs/TODO-findings.md#f9--latente-a2-head_write_pos-cresce-sem-máscara-no-atalho-layersis_empty--possível-oob-se-process-for-chamado-antes-de-prewarm)
* **Arquivos Relacionados:**
  * [src/models/a2/model/static/process.rs](file:///home/fabio/nam-rs/src/models/a2/model/static/process.rs)
  * [src/models/a2/model/dynamic/process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs)
* **Detalhamento Técnico:**
  1. No atalho de layers vazias em [src/models/a2/model/static/process.rs](file:///home/fabio/nam-rs/src/models/a2/model/static/process.rs):

     ```rust
     if self.layers.is_empty() {
         self.head_write_pos = (self.head_write_pos + total) & self.head_ring_mask;
         return;
     }
     ```

  2. Fazer o mesmo ajuste no correspondente atalho dinâmico em [src/models/a2/model/dynamic/process.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic/process.rs):

     ```rust
     if self.layers.is_empty() {
         self.head_write_pos = (self.head_write_pos + total) & self.head_ring_mask;
         return;
     }
     ```

* **Critérios de Aceitação:**
  * O indicador de escrita do buffer circular (`head_write_pos`) é mantido estritamente dentro dos limites da máscara de anel (`head_ring_mask`), evitando descompassos aritméticos se pesos forem recarregados em runtime sem um reset imediato intermediário.

---

#### Tarefa E1.2 — Adicionar Asserções de Segurança na Camada FiLM de A2 (F10) [DONE]

* **Tipo:** Melhoria Defensiva / Segurança
* **Severidade:** LATENTE
* **Risco:** BAIXO
* **Finding de Origem:** [TODO-findings.md F10](file:///home/fabio/nam-rs/TODO-findings.md#f10--latente-a2-film-get_unchecked-oob-se-condition_size--1-combinado-com-conv_post_film)
* **Arquivos Relacionados:**
  * [src/models/a2/film.rs](file:///home/fabio/nam-rs/src/models/a2/film.rs)
* **Detalhamento Técnico:**
  1. Em [src/models/a2/film.rs](file:///home/fabio/nam-rs/src/models/a2/film.rs), no método `process` da struct `FiLMLayer`, introduzir uma guarda de depuração estrita para verificar a conformidade do slice de condicionamento fornecido antes de repassá-lo ao kernel inseguro:

     ```rust
     debug_assert_eq!(
         condition.len(),
         self.cond_size,
         "FiLM process: condition slice length ({}) must match cond_size ({})",
         condition.len(),
         self.cond_size
     );
     ```

  2. Adicionar uma asserção análoga em `cond_to_scale_shift` para atestar que o slice possui pelo menos `self.cond_size` elementos antes de invocar fatiamentos inseguros (`get_unchecked`).
* **Critérios de Aceitação:**
  * Asserções de segurança ativas em builds de depuração evitam acessos fora-de-limites (OOB UB) na inferência FiLM caso o host ou o dispatcher forneça slices de condicionamento com tamanhos incompatíveis com os pesos do modelo.

---

#### Tarefa E1.3 — Criar Testes Unitários de Regressão para os Casos de Borda de A2 (F9/F10)

* **Tipo:** QA / Testes
* **Severidade:** GARANTIA DE QUALIDADE
* **Risco:** BAIXO
* **Arquivos Relacionados:**
  * [src/models/a2/model/dynamic_test.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic_test.rs)
  * [src/models/a2/film_test.rs](file:///home/fabio/nam-rs/src/models/a2/film_test.rs)
* **Detalhamento Técnico:**
  1. Escrever testes unitários em [src/models/a2/model/dynamic_test.rs](file:///home/fabio/nam-rs/src/models/a2/model/dynamic_test.rs) que criem instâncias do modelo dinâmico com camadas de peso vazias, exercitem o método `process()` consecutivas vezes, e verifiquem se `head_write_pos` faz o wrap-around adequadamente sem estourar.
  2. Executar `utils/tests-quick.sh` para atestar que toda a suíte de qualidade de código passa limpa e sem regressões.
* **Critérios de Aceitação:**
  * Novos testes unitários compilam e validam com sucesso as novas proteções e wrap-arounds da máquina de estados do modelo.
  * O script `utils/tests-quick.sh` é concluído com status de sucesso.
