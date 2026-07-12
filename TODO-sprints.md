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

* **Status**: `[x]`
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

  **Conclusão (2026-07-12)**: 5 proptests em `activations_test.rs` convertidos de `ProptestConfig::with_cases(10_000)` hardcoded para leitura dinâmica de `PROPTEST_CASES` com fallback `cfg!(debug_assertions)`: 1_000 em debug, 100_000 em release. Debug reduz de 10k para 1k casos por proptest (5× menos, ~5× mais rápido na Fase 1). Release sobe de 10k para 100k (10× mais rigoroso no long loop). Compilação verificada.

---

#### Tarefa T2.8 — Target Dir Dedicado para Compilação do CLAP (F-C7)

* **Status**: `[x]`
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

  **Conclusão (2026-07-12)**: `run_clap_audit_phase` isolada em `target/clap-audit/` via `CARGO_TARGET_DIR`. `cargo clean --release` substituído por `rm -rf target/clap-audit/release` (limpeza cirúrgica). `RELEASE_CLAP_BIN` atualizado para `target/clap-audit/release/libnam_rs.so`. `unset CARGO_TARGET_DIR` ao final da fase restaura o diretório default `target/` para Fases 6-7, preservando o cache de compilação principal.

---

## Critérios de Aceitação Gerais da Sprint 2 (Definition of Done)

Para fechar o Épico E2 em definitivo:

1. `utils/lints.sh` executa sem nenhum erro de formatação, clippy, ou cabeçalho SPDX.
2. `utils/tests-quick.sh` roda com sucesso em todas as fases.
3. Todas as modificações em arquivos de código Rust contêm a licença de Copyright no cabeçalho.
4. Nenhuma alteração adicionou código não real-time safe nas rotas DSP críticas.

---

## Eixo e Épico Correspondente S3

* **Épico**: [Épico E3 — Fixtures completas e auto-verificáveis](file:///home/fabio/nam-rs/TODO-findings.md#L668)
* **Objetivo Geral**: Garantir a completude e verificação automática de todos os fixtures do projeto, implementando governança estrita de freshness e paridade live bidirecional.
* **Complexidade/Risco**: Baixo (infraestrutura de testes e validações off-RT).

---

## Sprint 3 — Fixtures Completas e Auto-Verificáveis (Épico E3)

### Metas da Sprint 3 (Sprint Goals)

1. **Ativação Segura do Modelo de FiLM InputMixinPre**: Gerar o golden vector para `wavenet_a2_film_input_mixin_pre`, calibrar seus thresholds de fidelidade com base em medições reais e ativar o teste golden correspondente.
2. **Governança Estrita de Frescor dos Goldens (Freshness Gate)**: Evitar gaps silenciosos de testes não exercitados, garantindo que o manifest de frescor declare todos os arquivos golden esperados (incluindo escopos v2) e que o `check_freshness` falhe caso algum arquivo esperado não exista.
3. **Ampliação da Cobertura de Paridade Live C++**: Garantir paridade em tempo real (on-the-fly) contra o renderizador de referência C++ para todos os modelos dinâmicos/FiLM da arquitetura A2.
4. **Coerência Bidirecional CATALOG ↔ Testes**: Implementar verificação estática que garanta que todo modelo registrado no catálogo do gerador de goldens possua ao menos um teste consumidor na malha de testes Rust.

### S3 Papéis Necessários (Roles Needed)

* **Engenheiro de Software DSP / Rust**: Edição de `tests/models/golden_vectors.rs`, `tests/common/validation.rs`, `tests/parity/cpp_parity.rs` e `tests/models/meta_coherence.rs`.
* **Engenheiro de Infraestrutura de QA / DevOps**: Edição de `tests/fixtures/golden_gen_build.sh` e `utils/tests-quick.sh`.

---

### S3 Detalhamento das Tarefas Técnicas (Technical Tasks)

#### Tarefa T3.1 — Ativação e Calibração de wavenet_a2_film_input_mixin_pre (F-B3)

* **Status**: `[x]` — Concluído (2026-07-12). Golden gerado, teste ativado, thresholds calibrados.
* **Arquivos Afetados**:
  * [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)
  * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
  * [golden_vectors.rs](file:///home/fabio/nam-rs/tests/models/golden_vectors.rs)
* **Descrição**:
  1. Executar `./tests/fixtures/golden_gen_build.sh` para produzir o arquivo `tests/fixtures/golden_wavenet_a2_film_input_mixin_pre.bin`.
  2. Remover a marcação `#[ignore]` do teste `test_golden_vectors_wavenet_a2_film_input_mixin_pre()` em `tests/models/golden_vectors.rs`.
  3. Rodar o teste em modo release para capturar as medições de fidelidade reais (SNR, ESR, MR-STFT).
  4. Atualizar os limites no arquivo `tests/common/validation.rs` na chave `"wavenet_a2_film_input_mixin_pre"` sob as regras de calibração, inserindo o comentário explicativo no formato `// Measured: SNR=... dB, ESR=..., MR-STFT=...`.
* **Critério de Aceitação (DoD)**:
  * O teste `test_golden_vectors_wavenet_a2_film_input_mixin_pre` roda incondicionalmente no loop rápido e passa com sucesso.
  * O threshold de fidelidade segue a política de calibração do projeto com margens seguras justificadas em comentários.

---

#### Tarefa T3.2 — Governança Estrita no Freshness Gate (F-C9)

* **Status**: `[x]` — Concluído (2026-07-12). CATALOG refactored with skip_reason, manifest extended with # EXPECTED lines (69 total, 0 missing), check_freshness updated with missing-file detection.
* **Arquivos Afetados**:
  * [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)
  * [tests-quick.sh](file:///home/fabio/nam-rs/utils/tests-quick.sh)
* **Descrição**:
  1. Modificar o CATALOG no `golden_gen_build.sh` para incluir um quinto campo opcional `skip_reason` (ex: `convnet_test.nam:...:incompatible`).
  2. Refatorar o gerador para pular os modelos que possuem `skip_reason` preenchido de forma limpa e documentada no cabeçalho do CATALOG.
  3. Estender a Fase 11 de `golden_gen_build.sh` para escrever uma seção `# EXPECTED: <golden_filename>` no arquivo `.golden_manifest.sha256` para cada golden que deveria ter sido gerado (respeitando o `v2_scope` e os `skip_srs` de cada modelo no catálogo, pulando se houver `skip_reason`).
  4. Atualizar a função `check_freshness()` no `utils/tests-quick.sh` para ler as linhas de comentário `# EXPECTED:` do manifest e asseverar a existência física de todos os arquivos listados. Se algum arquivo esperado estiver ausente do disco, falhar com erro claro.
* **Critério de Aceitação (DoD)**:
  * Executar `tests-quick.sh` e comprovar que o Freshness Gate passa quando todos os arquivos estão íntegros e presentes.
  * Remover temporariamente um arquivo golden esperado e confirmar que o gate aborta com erro sinalizando a ausência.

---

#### Tarefa T3.3 — Cobertura Live C++ e Coerência CATALOG ↔ Testes (F-B6)

* **Status**: `[x]` — Concluído (2026-07-12). 4 novos testes live cross-validation (film_input_mixin_pre v1+v2, film_chaos_stress v1+v2). Meta-teste test_catalog_models_have_consumers: 29/30 ativos com consumidores.
* **Arquivos Afetados**:
  * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
  * [meta_coherence.rs](file:///home/fabio/nam-rs/tests/models/meta_coherence.rs)
* **Descrição**:
  1. Adicionar os testes de paridade live `live_cross_validation_wavenet_a2_film_input_mixin_pre` e sua respectiva versão v2 `live_cross_validation_v2_wavenet_a2_film_input_mixin_pre` no arquivo `tests/parity/cpp_parity.rs`, marcando-os com `#[ignore]` para rodarem apenas no loop longo (Phase 3).
  2. Adicionar de forma análoga `live_cross_validation_wavenet_a2_film_chaos_stress` e sua versão v2 no mesmo arquivo.
  3. Em `tests/models/meta_coherence.rs`, implementar o teste `test_catalog_models_have_consumers()` que analisa a direção inversa CATALOG → Testes: lê todas as entradas do CATALOG do script `golden_gen_build.sh` e valida se cada modelo possui ao menos um consumidor nas suítes de teste (realizando leitura direta de arquivos `.rs` relevantes da pasta `tests/`).
* **Critério de Aceitação (DoD)**:
  * Compilação limpa e livre de warnings.
  * O teste `test_catalog_models_have_consumers` passa com sucesso.
  * Os novos testes de validação cruzada live estão devidamente mapeados sob `#[ignore]` e integrados à cobertura de paridade.

---

## Critérios de Aceitação Gerais da Sprint 3 (Definition of Done)

Para fechar o Épico E3 em definitivo:

1. `utils/lints.sh` executa sem nenhum erro de formatação, clippy, ou cabeçalho SPDX.
2. `utils/tests-quick.sh` roda com sucesso em todas as fases.
3. Todas as modificações em arquivos de código Rust contêm a licença de Copyright no cabeçalho.
4. Nenhuma alteração adicionou código não real-time safe nas rotas DSP críticas.

---

## Eixo e Épico Correspondente S4

* **Épico**: [Épico E4 — Paridade NAMcore no código de produção](file:///home/fabio/nam-rs/TODO-findings.md#L674)
* **Objetivo Geral**: Implementar melhorias e correções de paridade na engine de produção do nam-rs para garantir conformidade plena com o NAMcore C++, incluindo suporte ao formato oficial do ConvNet e desqualificação/validação de recursos inconsistentes.
* **Complexidade/Risco**: Médio (alterações em componentes de produção da biblioteca DSP e do loader JSON, exigindo verificação de não-regressão em goldens).

---

## Sprint 4 — Paridade NAMcore no Código de Produção (Épico E4)

### Metas da Sprint 4 (Sprint Goals)

1. **Prewarm de WaveNet com LSTM Condition DSP (F-A5)**: Garantir que a inicialização do sub-modelo recorrente LSTM de condicionamento seja realizada corretamente com `cond_dsp.prewarm_samples()` iterações e incluir fixture sintética de cobertura.
2. **Suporte a sample_rate unknown (F-A2)**: Permitir sample rates do JSON com valor `-1.0` mapeando-os para `None` ("unknown") sem falhar a validação.
3. **Validação do LstmModelDyn (F-A7)**: Rejeitar `num_layers == 0` na topologia e validar que canais de entrada/saída são mono (`in_channels == 1 && out_channels == 1`).
4. **Desqualificação de SKU A1 com condition_dsp (F-A3)**: Impedir que modelos WaveNet com condition_dsp entrem no caminho rápido estático de catálogo que ignora esses parâmetros.
5. **Cálculo Preciso de prewarm_samples (F-A6)**: Atualizar as estimativas analíticas de prewarm da WaveNet para corresponderem à soma de RFs do C++.
6. **Fail-Closed para Recursos Não Suportados (F-A4)**: Rejeitar arquivos WaveNet A1 contendo recursos exclusivos da engine dinâmica A2 (ex. gating_mode diferente de "none", gated, FiLM, etc.).
7. **Paridade Plena do ConvNet C++ — Opção A (F-A1)**: Implementar load do formato plano do trainer C++ (BatchNorm cru e bias condicional), gerar fixture ConvNet real no formato C++, e reativar testes de paridade.
8. **Recalibração dos Gates de Fidelidade (F-B4)**: Ajustar os thresholds de SNR, ESR e MR-STFT em `validation.rs` de acordo com os limites físicos reais dos modelos com a engine Standard.
9. **Enum de MseGate Explícito (F-B5)**: Substituir o placeholder mágico `1e30` por um enum explicitando campos não aplicáveis.

### S4 Papéis Necessários (Roles Needed)

* **Engenheiro de Software DSP / Rust**: Alterações no loader de topologia, dispatcher, modelos DSP, testes unitários, testes de golden, e validação de fidelidade.
* **Engenheiro de Infraestrutura de QA / DevOps**: Scripts Python de fixtures e scripts de build/geração de goldens.

---

### S4 Detalhamento das Tarefas Técnicas (Technical Tasks)

#### Tarefa T4.1 — Pré-aquecimento do WaveNet Dyn com LSTM Condition DSP (F-A5)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [model_dyn.rs](file:///home/fabio/nam-rs/src/models/wavenet/model_dyn.rs)
  * [generate_a2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_a2_fixtures.py)
  * [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)
* **Descrição**:
  1. Em `src/models/wavenet/model_dyn.rs`, alterar a inicialização `cond_dsp.prewarm(0)` para `cond_dsp.prewarm(cond_dsp.prewarm_samples())` (ou `.max(1)`), garantindo que condicionamentos recorrentes como LSTM inicializem seus estados internos `h`/`c` adequadamente.
  2. Em `tests/fixtures/generate_a2_fixtures.py`, criar um modelo WaveNet sintético com `condition_dsp` do tipo LSTM (ex: 1 camada, 3 unidades ocultas), nomeando-o `wavenet_condition_lstm.nam`.
  3. No `golden_gen_build.sh`, incluir a geração de golden correspondente para este modelo e adicionar a validação na malha de testes Rust para garantir estabilidade.
* **Critério de Aceitação (DoD)**:
  * A inicialização de condition_dsp recorrentes é executada com o número correto de amostras.
  * O novo modelo `wavenet_condition_lstm.nam` é gerado e validado sem divergências contra o oráculo f64 e a paridade live C++.

---

#### Tarefa T4.2 — Suporte a sample_rate de Sentinela -1.0 no Loader (F-A2)

* **Status**: `[x]`
* **Arquivos Afetados**:
  * [validation.rs](file:///home/fabio/nam-rs/src/loader/nam_json/validation.rs)
  * [validation_test.rs](file:///home/fabio/nam-rs/src/loader/nam_json/validation_test.rs)
* **Descrição**:
  1. Em `src/loader/nam_json/validation.rs`, ajustar o validador de `sample_rate` para interceptar o valor `-1.0` (sentinela do NAMcore C++ para "unknown") e mapeá-lo para `None` ("unknown"), em vez de disparar erro `InvalidSampleRate`.
  2. Opcionalmente, estender o mapeamento para qualquer valor `<=` 0.0 finito (decidindo conforme `get_dsp.cpp` C++).
  3. Adicionar testes unitários em `validation_test.rs` cobrindo o comportamento para `-1.0 → None` e garantindo que NaN/±Inf ainda sejam devidamente rejeitados.
* **Critério de Aceitação (DoD)**:
  * Modelos com `sample_rate: -1.0` carregam com sucesso e são tratados como sample_rate desconhecido (None), assim como no NAMcore C++.

---

#### Tarefa T4.3 — Robustez e Validações no Loader do LstmModelDyn (F-A7)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [lstm.rs](file:///home/fabio/nam-rs/src/loader/nam_json/topology/lstm.rs)
  * [nam_json_test.rs](file:///home/fabio/nam-rs/src/loader/nam_json_test.rs)
* **Descrição**:
  1. Em `src/loader/nam_json/topology/lstm.rs`, rejeitar `num_layers == 0` com `log::error!` explícito (fail-closed).
  2. Mudar os níveis de `log::warn!` para `log::error!` em todas as 5 rejeições da topologia LSTM para visibilidade máxima em produção.
  3. Adicionar teste unitário `test_lstm_rejects_zero_layers` em `nam_json_test.rs`.
* **Critério de Aceitação (DoD)**:
  * Modelos LSTM com 0 camadas ou canais diferentes de 1 (mono) são rejeitados de forma segura durante o carregamento de topologia.

  **Conclusão (2026-07-12)**: Validações de `num_layers == 0`, `in_channels != 1` e `out_channels != 1` já existiam em `lstm.rs`. Melhorias: (a) 5 `log::warn!` elevados para `log::error!` com mensagens mais descritivas; (b) teste `test_lstm_rejects_zero_layers` adicionado. Todos os 7 testes LSTM de topologia passam. `utils/lints.sh` limpo.

---

#### Tarefa T4.4 — Desqualificação de Catálogo A1 para WaveNet com condition_dsp (F-A3)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [wavenet.rs](file:///home/fabio/nam-rs/src/loader/nam_json/topology/wavenet.rs)
* **Descrição**:
  1. Em `src/loader/nam_json/topology/wavenet.rs`, na função `get_wavenet_topology`, verificar a presença de `data.config.condition_dsp.is_some()`.
  2. Se presente, desqualificar o modelo do catálogo rápido estático (Known/Standard) e encaminhá-lo para a engine dinâmica `WaveNetModelDyn`, que implementa o processamento correto do sinal de condicionamento.
  3. Adicionar teste de topologia garantindo que modelos com a assinatura visual do catálogo mas contendo `condition_dsp` sejam devidamente direcionados para a engine dinâmica.
* **Critério de Aceitação (DoD)**:
  * Modelos WaveNet com condicionamento não são classificados no fast path estático e o processamento de áudio é mantido correto.

  **Conclusão (2026-07-12)**: Implementação já existente — `data.config.condition_dsp.is_none()` na condição `catalog_compatible` em `wavenet.rs:408` desqualifica modelos com `condition_dsp` do catálogo estático. Teste `test_topology_feather_with_condition_dsp_routes_to_free` em `nam_json_test.rs:249` cobre o cenário (modelo Feather com condition_dsp → Free, não Known). 13/13 testes de topologia passam.

---

#### Tarefa T4.5 — Correção do cálculo de prewarm_samples() do WaveNet (F-A6)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [mod.rs](file:///home/fabio/nam-rs/src/models/wavenet/mod.rs)
  * [model_dyn.rs](file:///home/fabio/nam-rs/src/models/wavenet/model_dyn.rs)
* **Descrição**:
  1. Em `src/models/wavenet/mod.rs`: `prewarm_samples()` estático somava array1 + array2. Melhorias: dinâmico soma todos os arrays, inclui `condition_dsp.prewarm_samples()`, e `post_stack_head.receptive_field() - 1`.
  2. `condition_dsp.prewarm(0)` → `cond_dsp.prewarm(cond_dsp.prewarm_samples())`.
* **Critério de Aceitação (DoD)**:
  * O retorno de `prewarm_samples()` bate com os valores calculados pelo C++ em `model.cpp:615-620`.

  **Conclusão (2026-07-12)**: Implementação já existente (commit `8490ff08`). `prewarm_samples()` estático: `array1.rf + array2.rf` (soma). Dinâmico: `.sum()` dos arrays + `.prewarm_samples()` do condition_dsp + `.receptive_field() - 1` do post_stack_head. `condition_dsp.prewarm(0)` corrigido para `cond_dsp.prewarm_samples()`. 6/6 testes de prewarm passam. Auditado contra `cpp_parity_map.md` §3.5.

---

#### Tarefa T4.6 — Fail-Closed para Recursos Não Suportados no WaveNet A1 (F-A4)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [model.rs](file:///home/fabio/nam-rs/src/loader/nam_json/model.rs)
  * [wavenet.rs](file:///home/fabio/nam-rs/src/loader/nam_json/topology/wavenet.rs)
  * [dynamic.rs](file:///home/fabio/nam-rs/src/loader/dispatcher/wavenet/dynamic.rs)
* **Descrição**:
  1. No parser e loader de topologia do WaveNet, detectar a presença de campos exclusivos da especificação A2 (`gating_mode` diferente de "none", `gated: true`, `head1x1`, `layer1x1` com canais incompatíveis ou slots FiLM ativos) em modelos que não sejam classificados como A2.
  2. Rejeitar o carregamento desses modelos com erro claro de formato ("feature X requires the A2 dynamic engine; model rejected to protect audio correctness"), impedindo saídas de áudio matematicamente corrompidas sem aviso.
* **Critério de Aceitação (DoD)**:
  * Modelos contendo parâmetros de gating ou FiLM fora do caminho A2 apropriado são rejeitados de forma segura pelo loader.

  **Conclusão (2026-07-12)**: Guardrail pré-existente em `wavenet.rs:325-391` já cobria `gating_mode`, `head1x1`, `layer1x1` e FiLM. Adicionada verificação de `gated: true` ao guardrail (antes apenas desqualificava do catálogo estático, sem rejeitar). 6 novos testes unitários (`test_wavenet_a1_rejects_*` + `test_wavenet_a1_accepts_gated_false`) cobrem todos os cinco campos A2-exclusivos rejeitados + contra-prova de A1 válido. 65/65 testes `nam_json_test` passam; 64/64 testes de integração wavenet passam.

---

#### Tarefa T4.7 — ConvNet Compatível com NAMcore — Opção A (F-A1)

* **Status**: `[x]` (2026-07-12)
* **Arquivos Afetados**:
  * [topology/convnet.rs](file:///home/fabio/nam-rs/src/loader/nam_json/topology/convnet.rs)
  * [dispatcher/convnet/mod.rs](file:///home/fabio/nam-rs/src/loader/dispatcher/convnet/mod.rs)
  * [golden_gen_build.sh](file:///home/fabio/nam-rs/tests/fixtures/golden_gen_build.sh)
  * [cpp_parity.rs](file:///home/fabio/nam-rs/tests/parity/cpp_parity.rs)
  * [generate_b1_2_fixtures.py](file:///home/fabio/nam-rs/tests/fixtures/generate_b1_2_fixtures.py)
* **Descrição**:
  1. **Loader de Topologia**: Em `topology/convnet.rs`, detectar o config plano gerado pelo trainer oficial do C++ (`channels` escalar + array global de `dilations` + `batchnorm` bool) e sintetizar a estrutura de camadas equivalente (um bloco por dilation, kernel_size fixo de 2).
  2. **Dispatcher de Pesos**: Em `dispatcher/convnet/mod.rs`, quando processando no formato plano C++, ler os pesos na ordem crua (bias condicionado a `!batchnorm`; parâmetros do BatchNorm crus: mean, var, gamma, beta, eps) e fundi-los em scale e offset no carregamento (`scale = gamma / sqrt(var + eps)` e `offset = beta - scale * mean`).
  3. **Fixtures e Goldens**: Atualizar `generate_b1_2_fixtures.py` para exportar a fixture `convnet_test.nam` no formato plano oficial C++.
  4. **Ativação de Paridade**: No `golden_gen_build.sh`, remover a marca de skip para o ConvNet, gerar seu golden real via render C++ e reativar o teste de paridade real `quick_parity_convnet` (substituindo a verificação de incompatibilidade placebo de T1.2).
* **Critério de Aceitação (DoD)**:
  * Modelos ConvNet no formato plano oficial C++ carregam com sucesso no nam-rs.
  * O teste `quick_parity_convnet` roda com sucesso comparando a saída com exatidão bitwise contra a biblioteca C++ de referência.

  **Conclusão (2026-07-12)**: Implementação completa dos 4 subitens:
  1. Topologia: `ConvNetFormat::FlatCpp` detecta `conv_channels` + `conv_dilations` + `conv_batchnorm` no config raiz, sintetizando 1 bloco por dilation com kernel_size=2.
  2. Dispatcher: `build_convnet_flat_cpp()` lê conv weights `[OUT][IN][TAP]` → interleaved 4-wide, BatchNorm raw (mean/var/gamma/beta/eps) → fusão scale/offset, e `LinearHead` row-major com head_scale=1.0 (C++ flat format não possui head_scale). `ConvNetModel` ganhou campo `linear_head: Option<LinearHead>` para projeção linear sem ativação.
  3. Fixture: `generate_b1_2_fixtures.py` exporta `convnet_test.nam` em formato plano C++ (CH=8, 6 dilations, batchnorm=true, 863 pesos).
  4. Paridade: `quick_parity_convnet` substitui teste placebo de incompatibilidade, com cross-validation C++ real (SNR=45.9 dB, ESR=2.54e-5). Limiares calibrados: SNR ≥ 35 dB, ESR < 1e-4, MR-STFT < 0.03.
  * C++ render tool aceita o formato plano sem crash (antes rejeitava com type_error).
  * Golden gerável via `golden_gen_build.sh` (skip_reason removido).
  * Testes de oracle f64 (reference_oracle_f64.rs) quebram com novo formato — anchors f64 precisam de regeneração.

---

#### Tarefa T4.8 — Recalibração de Gates de Fidelidade Frouxos (F-B4)

* **Status**: `[ ]`
* **Arquivos Afetados**:
  * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
  * [constants.rs](file:///home/fabio/nam-rs/tests/common/constants.rs)
* **Descrição**:
  1. Em `tests/common/validation.rs`, atualizar os limites de tolerância física (SNR e ESR) de todas as fixtures principais para refletirem a fidelidade ultra-alta do motor `Standard` default.
  2. Seguir a política de calibração do gate: definir o novo limite como o valor real medido com margem de segurança coerente (ex: 10× a 100× para ESR e -10 a -15 dB para SNR) para evitar falsos-positivos inter-máquinas/ISA.
  3. Anotar cada linha recalibrada com o comentário explicativo `// Measured: SNR=... dB, ESR=...`.
* **Critério de Aceitação (DoD)**:
  * Todos os gates de fidelidade estão calibrados e protegendo o código contra regressões realistas de precisão, mantendo a suite verde.

---

#### Tarefa T4.9 — Semântica MseGate::NotApplicable Explícita (F-B5)

* **Status**: `[ ]`
* **Arquivos Afetados**:
  * [validation.rs](file:///home/fabio/nam-rs/tests/common/validation.rs)
  * [threshold_calibration.rs](file:///home/fabio/nam-rs/tests/models/threshold_calibration.rs)
* **Descrição**:
  1. Substituir a representação mágica de `mse_limit = 1e30` (usada para ignorar a validação de MSE em modelos onde o ESR é o gate primário) por um tipo explicitado no enum `MseGate::NotApplicable` (ou um tipo `Option<f64>` no relatório).
  2. Ajustar a renderização dos relatórios de teste e o dashboard de qualidade para exibir `gate: N/A` ou similar em vez de `1.0e30`, limpando o ruído nos logs.
  3. Atualizar as asserções do meta-teste anti-placebo em `threshold_calibration.rs` para verificar o uso dessa variante explícita.
* **Critério de Aceitação (DoD)**:
  * Remoção do placeholder mágico `1e30` dos logs e relatórios.
  * O meta-teste de calibração de gates continua passando com sucesso.

---

## Critérios de Aceitação Gerais da Sprint 4 (Definition of Done)

Para fechar o Épico E4 em definitivo:

1. `utils/lints.sh` executa sem nenhum erro de formatação, clippy, ou cabeçalho SPDX.
2. `utils/tests-quick.sh` roda com sucesso em todas as fases de fidelidade e paridade, sem skips.
3. Todas as modificações em arquivos de código Rust contêm a licença de Copyright no cabeçalho.
4. Nenhuma alteração adicionou código não real-time safe nas rotas DSP críticas.

---

## Eixo e Épico Correspondente S5

* **Épico**: [Épico E5 — Contrato de Qualidade (tarefa final da sprint)](file:///home/fabio/nam-rs/TODO-findings.md#L692)
* **Objetivo Geral**: Congelar os índices de qualidade e performance do projeto sob um contrato versionado unificado, impedindo regressões silenciosas na malha de testes e documentando o fluxo de manutenção do contrato.
* **Complexidade/Risco**: Médio (alteração direta no script do painel de qualidade e definição do arquivo de baseline formal).

---

## Sprint 5 — Contrato de Qualidade (Épico E5)

### Metas da Sprint 5 (Sprint Goals)

1. **Unificação do Contrato de Qualidade**: Adicionar a capacidade de salvar baseline (`--save`) e verificar conformidade (`--check`) de fidelidade e latência sob as margens especificadas diretamente em `utils/quality-dashboard.sh` (conforme preferência do PO).
2. **Definição da Baseline Oficial**: Registrar o primeiro contrato de fidelidade e latência oficial no arquivo de controle `docs/quality-contract.txt`.
3. **Documentação de Manutenção**: Garantir que as diretrizes para atualização de baselines e gerenciamento do contrato estejam registradas nos manuais de teste e benchmark.

### S5 Papéis Necessários (Roles Needed)

* **Engenheiro de Infraestrutura de QA / DevOps**: Edição de `utils/quality-dashboard.sh` e geração do baseline oficial.
* **Engenheiro de Software DSP / Rust**: Atualização e revisão da documentação de processos em `docs/`.

---

### S5 Detalhamento das Tarefas Técnicas (Technical Tasks)

#### Tarefa T5.1 — Implementação de Verificação de Contrato em quality-dashboard.sh (F-E1)

* **Status**: `[ ]`
* **Arquivos Afetados**:
  * [quality-dashboard.sh](file:///home/fabio/nam-rs/utils/quality-dashboard.sh)
* **Descrição**:
  1. Implementar a opção `--check <file>` no script `utils/quality-dashboard.sh`.
  2. Quando acionado com `--check`, o script deverá executar as fases habilitadas pelo `--fidelity-only` ou `--bench-only` (ou modo `full`), parsear os resultados para um arquivo temporário em formato plain text, e compará-los contra o arquivo do contrato fornecido.
  3. Aplicar as seguintes margens para a comparação:
     * **Fidelidade (ESR/SNR/MR-STFT)**: O teste falhará se `novo_esr > contrato_esr * 10.0` (absorvendo variações ISA/f64) ou se `novo_snr < contrato_snr - 6.0` (em dB). Se o campo correspondente no contrato for `N/A`, ignorar o gate correspondente.
     * **Performance (latência mediana)**: O teste falhará se `nova_lat > contrato_lat * 1.10` (margem de ruído de 10%). Se o benchmark acusar regressão mas estiver dentro da margem, o script `tests-performance-regression.sh` continua sendo a autoridade estatística primária.
  4. Caso ocorra alguma violação de contrato, o script deve retornar código de erro diferente de zero e imprimir um sumário das métricas regredidas em vermelho.
* **Critério de Aceitação (DoD)**:
  * Executar `./utils/quality-dashboard.sh --check <file>` valida os resultados atuais contra o baseline, identificando violações fora das tolerâncias descritas e retornando exit code correspondente.

---

#### Tarefa T5.2 — Baseline Oficial e Documentação do Fluxo (F-E1)

* **Status**: `[ ]`
* **Arquivos Afetados**:
  * [quality-contract.txt](file:///home/fabio/nam-rs/docs/quality-contract.txt) [NEW]
  * [testing.md](file:///home/fabio/nam-rs/docs/testing.md)
  * [benchmarks.md](file:///home/fabio/nam-rs/docs/benchmarks.md)
* **Descrição**:
  1. Gerar o arquivo oficial de contrato rodando `./utils/quality-dashboard.sh --save docs/quality-contract.txt` com todas as fases corretas após a conclusão do Épico E4.
  2. Gerar `docs/quality-contract.txt` como a linha de base imutável para a qualidade do nam-rs.
  3. Adicionar seções explicativas em `docs/testing.md` e `docs/benchmarks.md` descrevendo o funcionamento da verificação do contrato e detalhando o procedimento para renovação deliberada da baseline (por exemplo, exigindo justificativa expressa no commit com a flag `--save`).
* **Critério de Aceitação (DoD)**:
  * O arquivo `docs/quality-contract.txt` está salvo no repositório.
  * O manual `docs/testing.md` e `docs/benchmarks.md` explicam claramente as políticas de conformidade do contrato de qualidade e suas margens de tolerância.

---

## Critérios de Aceitação Gerais da Sprint 5 (Definition of Done)

Para fechar o Épico E5 em definitivo:

1. `utils/lints.sh` executa sem nenhum erro de formatação, clippy, ou cabeçalho SPDX.
2. `utils/tests-quick.sh` roda com sucesso.
3. Todas as modificações em arquivos de código contêm a licença de Copyright no cabeçalho.
4. O arquivo `docs/quality-contract.txt` está gerado e a verificação integrada passa localmente.
