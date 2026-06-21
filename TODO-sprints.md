<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento de Sprints (Resolução dos Achados Residuais)

> **Skill:** `planejador-arquiteto`
> **Data:** 2026-06-21
> **Referência:** `TODO-audit.md` §5.3 Achados residuais da re-auditoria (RF1 a RF8).

Este documento detalha as Sprints e Tarefas Técnicas super detalhadas focadas em solucionar os Achados Residuais (RF) da última auditoria.
As tarefas estão prontas para a skill `implementador`.

---

## Épico F — Resolução Final de Robustez e Cobertura (RF1-RF8)

**Objetivo:** Eliminar as últimas divergências de placebos em testes, unificar as estratégias de descoberta, reforçar integridade RT (real-time) e documentar comportamentos atípicos identificados nos achados RF1 a RF8 da auditoria `TODO-audit.md`.

---

### Sprint 1: Cobertura e Desempenho Críticos (RF1, RF2, RF3) [DONE]

**Foco:** Reforçar cobertura dos paths dinâmicos e de convolução, eliminando testes placebo sensíveis e provendo instrumentação apropriada para o PGO (Profile-Guided Optimization).

#### Tarefa 1.1: Eliminar Placebo FiLM-em-A2 (Ref: RF1 🔴) [DONE]

- **Contexto:** Os goldens de paridade `tests/fixtures/golden_wavenet_a2_film_full.bin` e `..._lite.bin` não estão sendo consumidos. Os testes que deveriam conferir a paridade apenas garantem que a saída não gera `NaN` (`is_finite()`).
- **Implementação:**
  1. Em `tests/golden_vectors.rs`, localizar os testes `test_golden_vectors_wavenet_a2_film_lite` (aprox. linha 1469) e `test_golden_vectors_wavenet_a2_film_full` (aprox. linha 1499).
  2. Implementar a leitura dos goldens (`common::validation::read_golden`) já comitados no projeto, correspondentes aos arquivos `.bin` órfãos.
  3. Instanciar a métrica com `common::metrics::snr` (ou MSE/ESR) passando pela inferência o sinal padrão.
  4. Substituir `is_finite()` pelo gate forte (ex: `assert!(snr >= 60.0)`). Calibrar e marcar `// Measured: XX.X dB` junto ao código. Se os arquivos precisarem ser eliminados (se forem apenas resíduos sem utilidade), provar paridade de outra forma ou justificar a inexistência deles.

#### Tarefa 1.2: Inclusão do `WaveNetA2Dyn` no Benchmark PGO (Ref: RF2 🟠) [DONE]

- **Contexto:** Sem bench, o `WaveNetA2Dyn` não se beneficia do PGO.
- **Implementação:**
  1. Abrir `benches/inference_bench.rs`.
  2. Criar uma função de banco de ensaio, nomeada de forma compatível com o filtro PGO (precisa conter `_64samp`), como `A2Dyn_Gated_64samp_48kHz`.
  3. Instanciar o engine A2 dinâmico (`WaveNetA2Dyn`) com um modelo representativo (gated ou blended), utilizando block size = 64.
  4. Executar um laço clássico de criterion, como feito com os outros engines.

#### Tarefa 1.3: Documentação ou Geração de Goldens Multi-SR (Ref: RF3 🟢) [DONE]

- **Contexto:** Faltam `.bin` v2 (multi-SR) para engines dinâmicos. Apenas 48 kHz estão suportados.
- **Implementação:**
  1. ~~**Decisão:** Avaliar se os dinâmicos compensam o peso dos assets v2.~~ **Decidido: documentar.** Engines dinâmicos lidam com geometrias arbitrárias — a variância de geometria subsume a variância de sample rate. Live cross-validation em `cpp_parity.rs` já cobre multi-SR.
  2. **Caso documente:** ~~Em `docs/cpp_parity_map.md` e nos arquivos de tests pertinentes, deixar claro em texto (ou num README da suite de dinâmicos) que os goldens estão intencionalmente vinculados exclusivamente a 48 kHz para economizar overhead, referenciando o `TODO-audit.md` (RF3).~~ **FEITO** em `docs/cpp_parity_map.md` §3.3, `tests/fixtures/README.md` §v2 multi-SR coverage, e `tests/golden_vectors.rs` (comentários de seção). `TODO-audit.md` RF3 marcado como 🟢 resolvido.
  3. ~~**Caso implemente:** Gerar usando o commit C++ e adicionar em `tests/fixtures/`. (Se seguir esse rumo, inclua o diff que estenda o `V2_MODELS` no script de build de golden).~~ Não aplicável (caminho de documentação escolhido).
 docs: document intentional 48kHz-only v2 limitation for dynamic engine goldens

---

### Sprint 2: Assertividade, Consistência de Testes e Segurança RT (RF4 a RF8)

**Foco:** Transformar tolerâncias fracas em restrições assertivas, corrigir inconsistências de documentação, refatorar descobertas duplicadas e remover riscos silenciosos de panics de áudio (thread-safety).

#### Tarefa 2.1: Asserção de Classificação no `nondist_validation` (Ref: RF4 🟠) [DONE]

- **Contexto:** Erros de classificação atualmente causam somente um aviso impresso e deixam o teste passar com falsa sensação de sucesso.
- **Implementação:**
  1. Editar `tests/nondist_validation.rs` por volta das linhas 115-119.
  2. Identificar `eprintln!("⚠ Classification mismatch ...")`.
  3. Converter essa emissão de logs em uma chamada assertiva forte `assert_eq!(actual, expected, "Classification mismatch ...");` de forma a falhar o teste (exceto nos casos "Unknown*", que continuam sofrendo auto-skip).

#### Tarefa 2.2: Refatoração da Discovery em `cpp_parity.rs` (Ref: RF5 🟡)

- **Contexto:** A função `find_models_in_dir` foi generalizada em `tests/common/discovery.rs`, mas `cpp_parity.rs` manteve seu parser inline.
- **Implementação:**
  1. Em `tests/cpp_parity.rs` (linhas 713-727), remover a rotina local iterativa usando `read_dir`.
  2. Trocar pelo uso do utilitário padronizado: chamar `common::discovery::find_models_in_dir` para obter a lista de modelos.

#### Tarefa 2.3: Substituição de Placebos por Limites Físicos (Ref: RF6 🟡)

- **Contexto:** Ainda restam checks `is_finite()` soltos que passam muito frouxos sem medir a integridade estrutural da onda.
- **Implementação:**
  1. Modificar `test_namb_roundtrip_dispatcher_e2e` em `tests/spsc_pipeline.rs`: Substituir o asserção frouxa com a exigência de manter-se abaixo do zero físico para entradas controladas, aplicando `< 1e-7`.
  2. Modificar `test_prewarm_zero_rf` em `tests/wavenet_prewarm_edge.rs`: Checar se a variação obedece os parâmetros de tolerância corretos para silêncios (`abs < 1e-7`) em invés de meramente `is_finite()`.

#### Tarefa 2.4: Expurgar as referências legadas de `BossWN-lite.nam` (Ref: RF7 🟡)

- **Contexto:** Mudanças foram efetuadas em Épicos anteriores para adotar `EVH-5150-Lite.nam` como gate de threshold apertado, mas a documentação antiga restou.
- **Implementação:**
  1. Substituir menções textuais ao nome obsoleto `BossWN-lite.nam` para `EVH-5150-Lite.nam` em `tests/golden_vectors.rs:475,487`.
  2. Ajustar os marcadores explicativos de drift SNR para denotar a melhoria (`>= 105 dB`).
  3. Alterar `tests/fixtures/README.md` (linhas ~68, 120 e 348), removendo justificativas passadas de "SNR 0.9 dB / limite a 0 dB" e alinhando com a nova realidade restritiva.

#### Tarefa 2.5: Correções de Thread-safety e Fallback (Ref: RF8 🟡)

- **Contexto:** Hot-paths de áudio não podem gerar pânicos (unreachable) nem falhar silenciosamente sem sinalizar flag telemétrica de corrupção.
- **Implementação:**
  1. Modificar `src/models/a2/model/mod.rs` (linhas ~532-569), de forma que a entrada no branch de _fallback_ do áudio (usualmente processamento limpo / zero output) acione (sete) uma flag `RT_STATUS_` apropriada para fins de detecção de fluxo errôneo de estado da engine em produção.
  2. Em `src/models/a2/model/set_weights.rs:353` existe um `unreachable!()` num cold-path de inicialização. Substitua-o por um `debug_assert!()` emparelhado com um retorno de Result de erro limpo ou _handling_ que evite Panic em builds de release.
