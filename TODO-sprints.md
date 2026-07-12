<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Planejamento Ágil de Sprints

Este documento gerencia o backlog de sprints baseado nos achados mapeados em `TODO-findings.md`. Ele define as tarefas técnicas, as responsabilidades e os critérios de aceitação para a implementação segura e incremental das melhorias do projeto.

---

## Eixo e Épico Correspondente

* **Épico**: [Épico E1 — "Nenhum teste verde sem evidência" (rigor da malha)](file:///home/fabio/nam-rs/TODO-findings.md#L644)
* **Objetivo Geral**: Eliminar falsos-verdes da malha de paridade C++ e endurecer as asserções de contrato de incompatibilidade, determinismo e instrumentação de status nos scripts de QA.
* **Complexidade/Risco**: Baixo (focado estritamente na infraestrutura de testes e scripts de validação).

---

## Sprint 1 — Rigor da Malha de Testes (Épico E1)

### Metas da Sprint (Sprint Goals)

1. **Assegurar a integridade da paridade v1/v1_hf**: Garantir que skips silenciosos do toolchain C++ não disfarcem falhas de validação, estabelecendo asserções reais ou marcas claras de skip-coverage.
2. **Prevenir retrocessos de descarte**: Blindar a malha de testes com um meta-teste estrutural que impeça a introdução de novos descartes de `ParityOutcome`.
3. **Formalizar contratos de incompatibilidade**: Converter testes de modelos sabidamente incompatíveis (ConvNet antes de F-A1) em verificações estritas de erro de formato em vez de placebos.
4. **Visibilidade real no Nightly/Long Loop**: Atualizar o gerador de relatórios do `tests-long.sh` para refletir status `SKIPPED` reais e alertar sobre execuções suspeitas.
5. **Elevar o determinismo**: Garantir determinismo exato de bit (MSE == 0.0) para ConvNet e documentar os limites conhecidos do oráculo f64.

### Papéis Necessários (Roles Needed)

* **Engenheiro de Software DSP / Rust**: Edição dos arquivos `cpp_parity.rs`, `threshold_calibration.rs`, e `golden_vectors.rs`.
* **Engenheiro de Infraestrutura de QA / DevOps**: Edição dos scripts Bash `tests-long.sh` e `_lib.sh`.

---

### Detalhamento das Tarefas Técnicas (Technical Tasks)

#### Tarefa T1.1 — Correção de run_v1/run_v1_hf e Meta-teste anti-let_ (F-B1)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
  * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs)
* **Descrição**:
  1. Alterar as funções auxiliares `run_v1` e `run_v1_hf` para que elas **retornem** o `ParityOutcome` em vez de descartá-lo via `let _ = ...`.
  2. Atualizar as asserções nos testes rápidos rápidos que rodam no `tests-quick.sh`:
     * `quick_parity_lstm_1x16`
     * `quick_parity_wavenet_ch16`
     * `quick_parity_a2_full`
     Se o renderizador C++ estiver disponível (compilado/compilável), **exigir** `ParityOutcome::Completed` via `assert_eq!(outcome, ParityOutcome::Completed)`. Caso contrário, permitir skip mas imprimir a mensagem padronizada no stderr: `SKIP-COVERAGE: <nome_do_teste>`.
  3. Atualizar os testes `live_cross_validation_*` marcados com `#[ignore]` (que rodam no `tests-long.sh` onde o toolchain C++ é pré-requisito obrigatório verificado na Fase 0) para exigirem `ParityOutcome::Completed` incondicionalmente.
  4. Adicionar um teste meta-estrutural em `threshold_calibration.rs` que analise o código-fonte de `cpp_parity.rs` e falhe se qualquer wrapper usar descarte silencioso (`let _ = run_render_comparison`).
* **Critério de Aceitação (DoD)**:
  * Compilação limpa sem warnings.
  * O meta-teste estrutural passa com sucesso.
  * A execução dos testes de paridade reflete a cobertura real sem falsos-verdes.

---

#### Tarefa T1.2 — Contrato de Incompatibilidade para quick_parity_convnet (F-B2)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
* **Descrição**:
  1. O modelo `convnet_test.nam` é arquiteturalmente incompatível com o renderizador C++ atual (conforme documentado em `generate_b1_2_fixtures.py`). Hoje o teste simplesmente faz skip silencioso e passa.
  2. Substituir `quick_parity_convnet` por `quick_parity_convnet_expected_incompatible`.
  3. O teste deve asseverar que o retorno de `run_render_comparison` é exatamente `ParityOutcome::SkippedGarbageOutput` ou `SkippedRateRejected` (ou que gera falha com código de saída diferente de 0) e que o log do erro contém a mensagem esperada da biblioteca `nlohmann/json` indicando a incompatibilidade.
  4. Se futuramente F-A1 for implementada (Opção A), esse teste voltará a ser de paridade real.
* **Critério de Aceitação (DoD)**:
  * O teste deixa de ser um skip placebo e passa a verificar ativamente o contrato de falha de incompatibilidade arquitetural de formato de dados.

---

#### Tarefa T1.3 — Status SKIPPED e Alertas no tests-long.sh (F-C6)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh)
* **Descrição**:
  1. Alterar a função `run_pipewire_phase` para retornar o código de status `77` (convenção POSIX para skip) quando o daemon do PipeWire estiver indisponível.
  2. Atualizar a função central `run_phase` para rastrear esse status especial: se o comando retornar `77`, registrar no array de status a string `SKIPPED` em vez de `PASSED` ou `FAILED`.
  3. No relatório final do summary, formatar `SKIPPED` em cor amarela (usando `YELLOW` do `_lib.sh`).
  4. Implementar um guard-rail em `run_phase`: se o status gravado for `PASSED` mas a duração total da fase for `< 1` segundo, emitir um aviso em destaque no terminal sinalizando uma fase possivelmente vazia/falsa-verde.
* **Critério de Aceitação (DoD)**:
  * Rodar o `tests-long.sh` [Feito pelo DEV humano] localmente (com ou sem PipeWire ativo) e comprovar que a fase correspondente registra `SKIPPED` corretamente na tabela final ou executa normalmente e passa.
  * Verificação de avisos de duração funcional.

---

#### Tarefa T1.4 — Endurecimento de Determinismo do ConvNet (F-B7)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs)
* **Descrição**:
  1. Em `tests/models/golden_vectors.rs`, o teste de determinismo do ConvNet compara duas execuções idênticas com tolerância de erro absoluto inferior a `1e-6`.
  2. Modificar o laço de asserção para exigir exatidão bitwise absoluta (`MSE == 0.0` ou diferença absoluta igual a `0.0`), alinhando o ConvNet com as exigências dos demais modelos.
  3. Investigar imediatamente qualquer divergência caso ocorra falha (eliminando potenciais fontes de acumuladores indeterministas na implementação da rede).
* **Critério de Aceitação (DoD)**:
  * Execução de `cargo test --test golden_vectors` passa com sucesso em modo release com tolerância zero.

  **Conclusão (2026-07-12)**: Assertiva alterada de `abs < 1e-6` para `abs == 0.0`. Teste `test_golden_vectors_convnet_test` passa em modo release sem divergências bitwise. O ConvNet já é deterministicamente exato; a tolerância anterior era excessivamente conservadora.

---

#### Tarefa T1.5 — Documentação do Piso f64 das Âncoras (F-B8)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [perceptual_validation.md](file:///home/fabio/nam-rs/docs/perceptual_validation.md)
* **Descrição**:
  1. Documentar na seção do Oráculo de f64 em `docs/perceptual_validation.md` que o valor residual medido nas âncoras NumPy de aproximadamente `5.00e-16` para WaveNet, ConvNet e A2 é o limite teórico natural para a precisão f64 acumulada na janela de validação (≈ 2 × epsilon de precisão dupla).
  2. Esclarecer que o valor menor observado em LSTMs (≈ `3.49e-30`) decorre de caminhos matemáticos quase idênticos de bit e maior energia.
  3. Isso evitará suspeitas de clamps artificiais por futuros desenvolvedores ou auditores.
* **Critério de Aceitação (DoD)**:
  * Documentação atualizada e clara na seção de validação perceptual.

  **Conclusão (2026-07-12)**: Subseção "NumPy Anchor f64 Residual Floor" adicionada na seção do f64 Reference Oracle em `docs/perceptual_validation.md`, documentando o piso f64 teórico de ~5e-16 para feed-forward e ~3.49e-30 para LSTM, esclarecendo que `compute_esr_f64` não possui clamp/floor artificial.

---

## Critérios de Aceitação Gerais da Sprint (Definition of Done)

Para fechar o Épico E1 em definitivo:

1. `utils/lints.sh` executa sem nenhum erro de formatação, clippy, ou cabeçalho SPDX.
2. `utils/tests-quick.sh` roda com sucesso na Fase 1 (estrutural/debug) e Fase 2 (rápida de paridade/release).
3. Todas as modificações em arquivos de código Rust contêm a licença de Copyright no cabeçalho.
4. Nenhuma alteração adicionou código não real-time safe nas rotas DSP críticas.

---

## Épico E2 — Scripts de QA corretos e rápidos

* **Épico**: [Épico E2 — Scripts de QA corretos e rápidos](file:///home/fabio/nam-rs/TODO-findings.md#L655)
* **Objetivo Geral**: Corrigir a exibição de dados do dashboard de qualidade, otimizar o tempo de execução do loop rápido (`tests-quick.sh`) e evitar invalidações de cache do Cargo no loop longo (`tests-long.sh`).
* **Complexidade/Risco**: Baixo (focado principalmente em scripts de infraestrutura Bash e proptests de matemática).

---

## Sprint 2 — Scripts de QA Corretos e Rápidos (Épico E2)

### Metas da Sprint 2 (Sprint Goals)

1. **Correção de _cargo_meas e Robustez de Argumentos**: Eliminar a corrupção de flags do libtest usando arrays Bash e adicionar validações para evitar mau uso.
2. **Otimização do Loop Rápido (Quick Loop)**: Economizar ~30-40s no `tests-quick.sh` pulando verificações de deadline/jitter em debug e reduzindo a contagem padrão de proptests de ativação em compilações debug.
3. **Dashboard de Qualidade Fiel**: Corrigir a perda de precisão do MR-STFT, ajustar a contagem e visualização do progresso das fases do dashboard, e implementar mapeamento declarativo e explícito de famílias f64 eliminando fuzzy matching enganoso.
4. **Resiliência e Cache de Compilação no Long Loop**: Garantir que a compilação do CLAP em `tests-long.sh` utilize um diretório de target isolado para evitar a invalidação total do cache do Cargo para as fases subsequentes.
5. **Robustez Estatística no Regression Gate**: Proteger o script de regressão de performance contra falsos positivos do Criterion ao usar uma regex mais precisa e restrita.

### S2 Papéis Necessários (Roles Needed)

* **Engenheiro de Infraestrutura de QA / DevOps**: Edição dos scripts Bash `tests-quick.sh`, `tests-long.sh`, `tests-performance-regression.sh`, `quality-dashboard.sh` e `_lib.sh`.
* **Engenheiro de Software DSP / Rust**: Parametrização e ajuste dos limites de casos dos proptests de ativações em `activations_test.rs`.

---

### S2 Detalhamento das Tarefas Técnicas (Technical Tasks)

#### Tarefa T2.1 — Correção de _cargo_meas e Validação de Argumentos (F-C1)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh)
* **Descrição**:
  1. Alterar a assinatura e lógica de `_cargo_meas` em `utils/tests-quick.sh` para aceitar argumentos explícitos em vez de string-surgery:

     ```bash
     _cargo_meas() {
         local targets="$1"
         local filters="$2"
         local -a libtest_args=("${@:3}")
         ...
     ```

  2. Implementar a validação que percorre `libtest_args` e aborta com erro caso algum argumento comece com apenas um hífen (ex: `-test-threads=1`) em vez de dois (ex: `--test-threads=1`).
  3. Atualizar todas as 4 chamadas de `_cargo_meas` no script com a nova assinatura baseada em argumentos posicionais claros.
  4. Testar manualmente a correta serialização verificando o log de execução de `isa_parity` com `--test-threads=1`.
* **Critério de Aceitação (DoD)**:
  * Nenhuma string-surgery de flags no shell.
  * Validador impede flags malformados com hífen simples.
  * Execução serializada de `isa_parity` comprovada via ordem das saídas com `--nocapture`.

  **Conclusão (2026-07-12)**: Assinatura de `_cargo_meas` refatorada com argumentos posicionais explícitos (`targets`, `filters`, `libtest_args`). Validador regex `^-[^-]` adicionado para rejeitar flags com hífen simples. As 4 chamadas atualizadas para a nova API. `isa_parity` confirmado com `--test-threads=1 --nocapture` executando serialmente. Nenhum `eval` ou string-surgery remanescente.

---

#### Tarefa T2.2 — Otimização de Fase 1 e Skips de Tempo Real (F-D1)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh)
* **Descrição**:
  1. No bloco de execução da Fase 1 em `utils/tests-quick.sh`, adicionar explicitamente `--skip rt_deadline::` e `--skip rt_jitter::` no comando `cargo test --lib`.
  2. Isso evitará medir prazos (deadlines) e instabilidades (jitters) de thread de tempo real em builds de debug, economizando ~28s do Quick Loop.
* **Critério de Aceitação (DoD)**:
  * Medições de deadline/jitter não rodam na Fase 1.
  * Ganho de tempo de parede (wall-time) confirmado no log.

  **Conclusão (2026-07-12)**: `--skip rt_deadline::` e `--skip rt_jitter::` adicionados ao bloco entry-point da Fase 1. Todos os 18 testes (14 deadline + 4 jitter) são filtrados; `rt_constraints` compila mas roda 0 testes em ~0s. Os testes permanecem ativos no `tests-long.sh` (Fase de release).

---

#### Tarefa T2.3 — Correção de Exibição de MR-STFT (F-C2)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Descrição**:
  1. Em `utils/quality-dashboard.sh`, corrigir o reformatador numérico do MR-STFT para usar notação científica `%.2e` caso o valor bruto contenha expoente (`e` ou `E`), em vez de arredondar para `0.0000` via `%.4f`.
  2. Aplicar esse mesmo comportamento para as outras colunas de métricas que sofrem reformatação por comprimento em `quality-dashboard.sh`.
* **Critério de Aceitação (DoD)**:
  * Dashboard exibe valores reais de MR-STFT (ex: `9.73e-6`) em vez de `0.0000`.

  **Conclusão (2026-07-12)**: Reformatação do MR-STFT corrigida: valores com notação científica (`[eE]`) usam `%.2e` (ex: `9.73e-06`), demais mantêm `%.4f` (ex: `0.0123`). Demais colunas de métrica (ESR vs NAMcore, ESR vs f64) já utilizavam `%.2e` e não sofriam do problema.

---

#### Tarefa T2.4 — Sincronização de Fases e Renderização no Dashboard (F-C3)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
  * [_lib.sh](file:///home/fabio/nam-rs/utils/_lib.sh)
* **Descrição**:
  1. Corrigir o cálculo dinâmico de `phase_count` no `quality-dashboard.sh` para refletir as fases realmente executadas (evitando a contagem duplicada da fase "Parseando resultados").
  2. Introduzir a nova fase final `phase "Renderizando dashboard"` encapsulando a chamada `render_dashboard`.
  3. No helper `phase()` de `utils/_lib.sh`, alterar a impressão para `${PHASE_TOTAL:-?}` para que scripts sem totalizador mostrem `[N/?]` em vez de dividir por zero ou exibir valores incorretos.
* **Critério de Aceitação (DoD)**:
  * A contagem final das fases do dashboard fecha em `[8/8]` no modo full, mostrando progresso consistente.

  **Conclusão (2026-07-12)**: Cálculo de `phase_count` corrigido para não contar a fase "Parseando resultados" em duplicata. Nova fase "Renderizando dashboard" adicionada como último passo de `render_dashboard`. Helper `phase()` em `_lib.sh` alterado para `${PHASE_TOTAL:-?}`, exibindo `[N/?]` quando `PHASE_TOTAL` não está definido. Modos: full `[8/8]`, fidelity-only `[7/7]`, bench-only `[3/3]`.

---

#### Tarefa T2.5 — Mapa Explícito de Famílias f64 no Dashboard (F-C4)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Descrição**:
  1. Remover a lógica de substring fuzzy-matching em `_lookup_esr_f64`.
  2. Declarar um mapa explícito e estático associativo (`ESR_F64_FAMILY_MAP`) no topo de `quality-dashboard.sh`, correlacionando cada etiqueta (label) de modelo (golden-label) com seu respectivo fixture/arquivo medido no oráculo f64 (ex: `["BossWN-standard"]="wavenet_official.nam"`).
  3. Em `_lookup_esr_f64`, usar esse mapa para obter a chave de oráculo. Se não houver mapeamento direto, verificar se a etiqueta em si existe em `ESR_F64_PAIRED` (ou com extensão `.nam`).
  4. Determinar a proveniência: se o arquivo medido corresponder exatamente à etiqueta buscada, retornar status `exact`. Caso contrário, retornar `family:<fixture_name>` para indicar uma aproximação de família.
* **Critério de Aceitação (DoD)**:
  * Sem heurísticas mágicas de texto para achar ESR do oráculo.
  * Proveniência `exact` vs `family` corretamente catalogada e exibida na tabela.

  **Conclusão (2026-07-12)**: `_lookup_esr_f64` reescrito: removeu-se a busca fuzzy por substring (`tr/sed` + loop sobre `ESR_F64_PAIRED`) e substituiu-se por `ESR_F64_FAMILY_MAP` (35 entradas, cobrindo todos os 27 labels V1 do golden_vectors + cpp_parity). O mapa correlaciona cada golden-label ao seu fixture `.nam` do oráculo f64. `ESR_F64_EXACT_MATCH` (4 labels) identifica quais labels têm medição própria (exata) vs. aproximação de família (proxy). Fallback: busca direta em `ESR_F64_PAIRED` pelo label ou `${label}.nam`. Zero heurísticas — toda decisão é explícita e estática.

---

#### Tarefa T2.6 — Formatação Numérica e Correções Menores (F-C5, F-C8, F-D3)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
  * [tests-performance-regression.sh](file:///home/fabio/nam-rs/utils/tests-performance-regression.sh)
  * [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh)
* **Descrição**:
  1. Em `quality-dashboard.sh`, criar o helper unificado `_fmt_metric` com `LC_ALL=C` embutido para formatar métricas, forçando `%.2e` para qualquer valor que contenha notação científica.
  2. Atualizar as chamadas de `awk` nas funções `esr_verdict` e `esr_verdict_short` adicionando o prefixo `LC_ALL=C` para evitar problemas com locais regionais (como separador de vírgula em pt_BR).
  3. Em `tests-performance-regression.sh`, alterar a verificação de regressão do Criterion de `grep -q "regressed"` para uma regex mais específica: `grep -qE '(has regressed|Performance has regressed)'`, eliminando falsos-positivos.
  4. Em `tests-quick.sh`, atualizar o banner de tempo estimado de execução fria para `± 8 minutes` refletindo a realidade observada, documentar no cabeçalho/comentários o motivo de manter o `perf_soak` na Fase 1 (baixo custo, 2s), e limpar a declaração redundante de `CXX=""` na rotina de compilação preventiva do C++ render.
* **Critério de Aceitação (DoD)**:
  * Formatação consistente e imune a locais do sistema operacional.
  * Zero falsos-positivos por regressão falsa do Criterion.
  * Banner honesto e limpeza cosmética efetuada.

  **Conclusão (2026-07-12)**:
  1. `quality-dashboard.sh`: criado helper `_fmt_metric()` com `LC_ALL=C`, auto-detecta notação científica (`%.2e` se `[eE]`, `%.4f` caso contrário). Substituiu 5 chamadas manuais de `_nfmt` + length-check nas funções `render_quick_summary` e `render_fidelity_details`. `esr_verdict` e `esr_verdict_short`: awk prefixado com `LC_ALL=C`.
  2. `tests-performance-regression.sh`: grep de regressão alterado para `grep -qE '(has regressed|Performance has regressed)'`, eliminando falsos-positivos por match parcial em `regressed`.
  3. `tests-quick.sh`: banner atualizado para `± 8 minutes`. Documentado motivo de `perf_soak` na Fase 1 (~2s, testes estruturais de concorrência/pipeline, não benchmarks). Removidos `CXX="${CXX:-}"` redundante e `else CXX=""`; uso de `${CXX:-}` no `[ -z ]` para evitar erro `set -u`.

---

#### Tarefa T2.7 — Parametrização e Limite de Casos nos Proptests de Ativação (F-D2)

* **Status**: `[ ]`
* **Arquivos Afetados**:
  * [activations_test.rs](file:///home/fabio/nam-rs/src/math/activations/activations_test.rs)
* **Descrição**:
  1. Modificar os proptests pesados em `src/math/activations/activations_test.rs` para ler dinamicamente o número de casos da variável de ambiente `PROPTEST_CASES`.
  2. Configurar o fallback dinâmico: se a variável de ambiente não estiver definida, usar `1_000` casos em compilações debug (`cfg!(debug_assertions)` sendo verdadeiro) e `100_000` em builds release.

     ```rust
     ProptestConfig {
         cases: std::env::var("PROPTEST_CASES")
             .ok()
             .and_then(|v| v.parse().ok())
             .unwrap_or(if cfg!(debug_assertions) { 1_000 } else { 100_000 }),
         .. ProptestConfig::default()
     }
     ```

* **Critério de Aceitação (DoD)**:
  * O tempo dos testes estruturais de ativação cai sensivelmente em debug (Fase 1).
  * Execução completa e rigorosa mantida em release/long loop.

---

#### Tarefa T2.8 — Target Dir Dedicado para Compilação do CLAP (F-C7)

* **Status**: `[ ]`
* **Arquivos Afetados**:
  * [tests-long.sh](file:///home/fabio/nam-rs/utils/tests-long.sh)
* **Descrição**:
  1. Em `utils/tests-long.sh`, na fase de auditoria do CLAP (`run_clap_audit_phase`), exportar `CARGO_TARGET_DIR="target/clap-audit"` para isolar completamente os artefatos compilados do CLAP Plugin.
  2. Substituir `cargo clean --release` por `rm -rf target/clap-audit/release` para limpar apenas o diretório isolado.
  3. Atualizar o caminho `RELEASE_CLAP_BIN` para apontar para `target/clap-audit/release/libnam_rs.so`.
  4. Desexportar/unsetar a variável `CARGO_TARGET_DIR` ao final da fase para que as fases subsequentes (Fases 6-7) voltem a usar o diretório default `target/` e preservem o cache de compilação gerado nas fases anteriores.
* **Critério de Aceitação (DoD)**:
  * A compilação do CLAP não invalida o cache do compilador para as fases 6-7.
  * Tempo total de compilação e execução do Long Loop reduzido de forma significativa.

---

## Critérios de Aceitação Gerais da Sprint 2 (Definition of Done)

Para fechar o Épico E2 em definitivo:

1. `utils/lints.sh` executa sem nenhum erro de formatação, clippy, ou cabeçalho SPDX.
2. `utils/tests-quick.sh` roda com sucesso em todas as fases.
3. Todas as modificações em arquivos de código Rust contêm a licença de Copyright no cabeçalho.
4. Nenhuma alteração adicionou código não real-time safe nas rotas DSP críticas.
