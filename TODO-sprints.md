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

#### Tarefa A1.2 — Evitar Dessincronização de `prev_state` em Transições Rápidas

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

---

#### Tarefa A1.3 — Atualização do Estado de Questões nas Documentações (Docs)

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

#### Tarefa A1.4 — Validação dos Testes Unitários de Integração de Estado

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
