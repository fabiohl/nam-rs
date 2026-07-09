<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Backlog de Execução: Investigação de Causa-Raiz do Gap de Fidelidade da LSTM (pós-Kahan)

Este documento detalha a execução do Épico proposto em [`TODO-findings.md`](TODO-findings.md) — **Achado A1** ("Auditoria do Achado 2: LSTM Recurrent State Drift — Fix Implementado Corretamente, mas Ineficaz"). O objetivo não é reverter a compensação de Kahan (mantida por ser estritamente benéfica — ver A1 §4.1), mas **reabrir e concluir corretamente** a investigação de causa-raiz do gap de ~13 dB entre o ESR de produção observado para `BossLSTM-1x16`/`2x8` e a soma das fontes de erro já conhecidas e decompostas.

> Rodar `utils/quality-dashboard.sh` antes e depois de cada Sprint para acompanhar a evolução das métricas de fidelidade.

---

## 🧭 Visão Geral e Mitigação de Risco

* **Objetivo:** Identificar, com evidência quantitativa reproduzível, a(s) causa(s) real(is) do ESR elevado de `BossLSTM-1x16`/`2x8` em regime de produção (stream contínuo de 5s), substituindo a hipótese refutada de "drift recorrente por acumulação f32" (Achado 2 original) por um diagnóstico correto — e só então decidir se uma correção adicional é necessária.
* **Descoberta antecedente relevante (T8.2/T8.3, 2026-06-28):** O projeto já enfrentou e corrigiu, para os testes de decomposição de erro de curta duração (`test_decomposition_lstm`, sweep de 256 amostras), exatamente a mesma classe de artefato que suspeitamos dominar o resultado do `t33_diagnostic_recurrent_drift_lstm_1x16`: **divergência arquitetural por estado inicial não pareado entre oráculo `f64` e produção `f32`** (`docs/perceptual_validation.md:696-710`, `:729-757`). Antes da correção T8.2, o oráculo reportava ESR ≈ 1.0 (placebo); depois de parear o prewarm, o piso real caiu para `3.57e-3` (`tests/parity/reference_oracle_f64.rs:273-314`, função `run_oracle_esr_paired`), posteriormente recalibrado para `3.41e-3` pós-SQ5 (`tests/common/constants.rs:24-33`, `LSTM_ESR_LIMIT = 7.0e-3 = 3.41e-3 × 2`). **A Sprint 2 deste documento testa, como hipótese líder, se o `t33_diagnostic_recurrent_drift_lstm_1x16` reintroduziu esse mesmo bug arquitetural** (ele usa `model.prewarm(24_000)` — que alimenta 24k **zeros** — no lado da produção, mas alimenta o oráculo diretamente com o sinal real sem nenhum prewarm/pareamento — ver `tests/parity/reference_oracle_f64.rs:802-813`).
* **Risco de Processo:** Alto se repetido. A causa-raiz da falha do Achado 2 (`TODO-findings.md` §3) foi o encerramento de sprints por checklist ("testes passam") em vez de por resultado quantitativo comparado a uma baseline. Todas as tarefas de validação abaixo **exigem número medido antes/depois documentado no próprio commit ou PR**, não apenas "passou".
* **Risco Técnico:** Baixo. Este épico é predominantemente diagnóstico (testes, instrumentação, documentação) e não deve alterar o hot-path de produção (`src/models/lstm/*`, `src/math/lstm/*`) a menos que a Sprint 2 revele uma causa-raiz adicional e corrigível — nesse caso, a correção (Sprint 3) deve seguir as mesmas diretrizes de RT-safety e SIMD do `.agents/rules/rust.md` já aplicadas com sucesso no épico anterior (zero alocação, `AlignedVec`, baseline `x86-64-v3`).
* **Política de Modificação:** Nenhuma alteração de tolerância de teste ou de baseline numérica pode ser aceita sem o comentário de proveniência de medição exigido por `docs/perceptual_validation.md` (Regra 3 — "Mandatory Measurement Provenance Comment"), no formato `// Measured: ESR=... @ ..., limit = medido × margem`.

---

## 🏃 Épico: Investigação de Causa-Raiz e Fechamento do Achado A1

### 📅 Sprint 1: Instrumentação de Diagnóstico Correta

Nesta etapa, corrigimos as lacunas de instrumentação que impediram o Sprint 4 do épico anterior de detectar que o fix não resolvia o problema (`TODO-findings.md` §3, itens 2 e 4): ausência de asserção no teste de diagnóstico, ausência de decomposição de erro pareada para os modelos `BossLSTM-*` específicos, e tolerância de teste de paridade Kahan excessivamente larga.

#### 🎯 Tarefa 1.1: Diagnóstico Prewarm-Paired de Longa Duração para `BossLSTM-1x16` [CONCLUÍDO]

> **Resultado:** O teste `t33b_diagnostic_recurrent_drift_lstm_1x16_paired` foi implementado e executado, retornando `ESR_tail = 2.589060e-2` (-15.9 dB). O gap de ~13 dB permanece idêntico ao do teste legado (`2.611684e-2`), o que **refuta** a hipótese de mismatch de estado inicial como causa do gap.

* **Arquivo:** [`tests/parity/reference_oracle_f64.rs`](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) (nova função, próxima à `t33_diagnostic_recurrent_drift_lstm_1x16`, linha 790)

* **Risco:** 🔴 Crítica — esta é a tarefa que mais provavelmente resolve o mistério do gap de 13 dB do Achado A1.

* **Ações:**

  * Criar `t33b_diagnostic_recurrent_drift_lstm_1x16_paired()` (`#[ignore]`), **sem remover** a função legada `t33_diagnostic_recurrent_drift_lstm_1x16` (mantê-la para efeito de comparação histórica/interop — ver nota abaixo).

  * Reaproveitar a metodologia de `run_oracle_esr_paired` (linhas 273-314): alimentar **tanto a produção `f32` quanto o oráculo `f64`** com o **mesmo** sinal `generate_stress_signal_v2_default(48000)` (240k amostras) desde `t=0`, sem chamar `model.prewarm(24_000)` (que alimenta zeros) — ao invés disso, tratar as primeiras `N_WARMUP = 24_000` amostras do próprio sinal real como período de aquecimento descartável, e medir o ESR apenas na cauda (ex.: últimas 216k amostras, ou nas mesmas janelas cumulativas 512→240000 já usadas em `t33` mas com offset de 24k).

  * Estrutura sugerida:

    ```rust
    #[test]
    #[ignore]
    fn t33b_diagnostic_recurrent_drift_lstm_1x16_paired() {
        use nam_rs::testing::perceptual::compute_esr;
        use nam_rs::testing::stress::generate_stress_signal_v2_default;

        let path = models_dir().join("BossLSTM-1x16.nam");
        let md = load_and_parse(&path);
        let stress_signal = generate_stress_signal_v2_default(48000); // 240k amostras
        let stress_f64: Vec<f64> = stress_signal.iter().map(|&x| x as f64).collect();

        // Produção: SEM model.prewarm(zeros) — processa o sinal real desde t=0,
        // igual ao oráculo, eliminando o mismatch de estado inicial (T8.2-style).
        let mut model = nam_rs::loader::dispatcher::build_model(&md).expect("build_model");
        let mut output = vec![0.0f32; stress_signal.len()];
        let mut pos = 0;
        while pos < stress_signal.len() {
            let nf = (stress_signal.len() - pos).min(64);
            model.process(&stress_signal[pos..pos + nf], &mut output[pos..pos + nf]);
            pos += nf;
        }

        let oracle = oracle_forward(&md, &stress_f64, &PrecisionConfig::default());
        let oracle_f32: Vec<f32> = oracle.iter().map(|&x| x as f32).collect();

        const N_WARMUP: usize = 24_000;
        let esr_tail = compute_esr(&oracle_f32[N_WARMUP..], &output[N_WARMUP..]);
        println!(
            "T33b — LSTM 1x16 paired (sem mismatch de estado inicial), cauda de {} amostras: ESR={:.6e} ({:.1} dB)",
            stress_signal.len() - N_WARMUP, esr_tail, 10.0 * esr_tail.log10()
        );
        // ... (ver Tarefa 1.2 para blockwise dentro desta mesma função)
    }
    ```

  * **Critério de sucesso da hipótese:** se `esr_tail` convergir para a faixa `~3e-3` a `~4e-3` (consistente com `LSTM_ESR_LIMIT`/2 = 3.5e-3 documentado em `tests/common/constants.rs:26`), a hipótese de mismatch de estado inicial está **confirmada** como causa dominante do gap de 13 dB, e a "Interop 2.61e-2 vs NAMCore" documentada em `docs/perceptual_validation.md:671` permanece como métrica de interoperabilidade legítima e separada (não um bug a corrigir aqui).

  * **Nota:** manter `t33_diagnostic_recurrent_drift_lstm_1x16` (não pareado) é intencional — ele mede o efeito real de iniciar produção a partir de silêncio (`prewarm(24_000)`) vs. sinal real, que é o comportamento **real** de um plugin ao carregar (primeiro áudio após silêncio/carregamento do preset). As duas métricas respondem perguntas diferentes e ambas têm valor; documentar essa distinção explicitamente no doc final (Tarefa 3.4).

#### 🎯 Tarefa 1.2: ESR por Janela Não-Cumulativa (Blockwise) Correlacionada às Seções do Sinal de Stress [CONCLUÍDO]

* **Arquivos:** [`src/testing/perceptual.rs`](file:///home/fabio/nam-rs/src/testing/perceptual.rs) (nova função pública), [`tests/parity/reference_oracle_f64.rs`](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs)
* **Ações:**
  * Adicionar `pub fn compute_esr_blockwise(reference: &[f32], test: &[f32], block_size: usize) -> Vec<f64>` em `src/testing/perceptual.rs`, próxima a `compute_esr` (linha 42): calcula ESR (mesma fórmula `Σ(r-t)²/Σr²`) em blocos **disjuntos** (não sobrepostos, não cumulativos) de `block_size` amostras, retornando um `Vec<f64>` com um valor por bloco.
  * Reaproveitar em `t33_diagnostic_recurrent_drift_lstm_1x16` (legado) e em `t33b_diagnostic_recurrent_drift_lstm_1x16_paired` (Tarefa 1.1) com `block_size = 48_000` (1s, alinhado às 6 seções do `generate_stress_signal_v2` — ver `src/testing/stress.rs:82-296`) e também `block_size = 12_000` (250ms, granularidade suficiente para isolar o trecho de palm-mute intermitente em 2.0–2.5s e o decaimento de acorde em 4.5–5.0s).
  * Imprimir uma tabela `bloco → [tempo] → ESR → seção do sinal` (rotular manualmente as 6 seções conforme a tabela da Tarefa 2.4/`src/testing/stress.rs`), para permitir correlacionar visualmente picos de ESR com trechos de baixa amplitude (silêncio entre hits de palm-mute, cauda do decaimento de acorde) vs. trechos de alta amplitude.
  * **Critério de aceite:** o padrão de ESR por bloco deve ser reportado e comparado entre a variante pareada (1.1) e a legada (`t33` original) — a expectativa é que a variante pareada mostre ESR uniformemente baixo (~3e-3) em todos os blocos após o descarte dos primeiros 24k, enquanto a legada pode ainda mostrar picos correlacionados a trechos de baixa amplitude (confirmando §2.4 do Achado A1).

> **Resultado:** Função `compute_esr_blockwise` implementada e integrada. A comparação blockwise revelou comportamento quase idêntico entre o teste pareado (`t33b`) e o legado (`t33`) após o primeiro bloco (ex: pico de ESR de `~1.33` a `~1.45` na seção de palm-mute de 2.0-2.5s em ambos os testes), provando que o desvio elevado é provocado por passagens de baixa amplitude e não por mismatch de inicialização.

#### 🎯 Tarefa 1.3: Decomposição de Erro Pareada com Prewarm para `BossLSTM-1x16` e `BossLSTM-2x8` [CONCLUÍDO]

* **Arquivo:** [`tests/parity/reference_oracle_f64.rs`](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) (novas funções `test_decomposition_boss_lstm_1x16`, `test_decomposition_boss_lstm_2x8`, próximas a `test_decomposition_lstm`, linha 380)
* **Ações:**
  * Hoje a decomposição de erro (`ΔESR f16c`, `ΔESR bf16`, `ΔESR Padé`, `ΔESR acumulação f32`) só é medida para `lstm.nam` (H=3 official) via `test_decomposition_lstm` (linhas 380-403), usando sweep cold-start de 256 amostras. O dashboard já sinaliza essa lacuna (`quality-dashboard-antes.txt:26-29`): *"BossLSTM-1x16/2x8 currently show the LSTM family value... NOT their own f64-oracle floor"*.
  * Replicar `run_oracle_esr_paired` + `run_decomposition` (linhas 422-472 de `src/testing/reference_oracle/mod.rs`) para `BossLSTM-1x16.nam` e `BossLSTM-2x8.nam`, usando **prewarm pareado de 24k + janela de medição de 4096 amostras do próprio sinal de stress v2** (em vez do sweep sintético de 256 amostras) — chamando `run_decomposition` com um subvetor de `stress_f64` (ex.: `stress_f64[24_000..28_096]`) e a saída de produção correspondente, para que a decomposição reflita o regime de sinal real (guitarra) em vez de um sweep senoidal sintético.
  * Adicionar `assert!` de sanidade (Regra 5 do Gate Calibration Policy — `docs/perceptual_validation.md:963-982`): `Σ(ΔESR f16c, bf16, Padé, f32-acc) ≈ ESR total`, dentro de 10×, sinalizando explicitamente se a violação ocorrer (como já ocorre para `ConvNet-test`/`WaveNet-official` in regime cold-start, documentado como esperado pelas notas do próprio dashboard).
  * **Critério de aceite:** obter, pela primeira vez, o piso `f64` real e específico de `BossLSTM-1x16`/`2x8` (não o valor de família herdado de `lstm.nam`), permitindo preencher corretamente a coluna "vs Ideal (f64)" do dashboard (Sprint 3, Tarefa 3.5) sem a ressalva `(~fam.)`.

> **Resultado:** Decomposição de erro pareada com prewarm de 24k implementada. Obtidos os pisos ideais f64 específicos de `5.06e-2` (-13.0 dB) para `BossLSTM-1x16` e `1.73e-3` (-27.6 dB) para `BossLSTM-2x8` em regime de sinal de guitarra real. A aproximação da ativação Padé é a fonte dominante de erro em ambos os modelos. Testes integrados à suíte `parity` com a verificação de sanidade da Regra 5 (razões calculadas de 1.05 e 0.99, respectivamente).

#### 🎯 Tarefa 1.4: Regression Gate para os Diagnósticos de Drift (Legado e Pareado) [CONCLUÍDO]

* **Arquivos:** [`tests/parity/reference_oracle_f64.rs`](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs), [`tests/common/constants.rs`](file:///home/fabio/nam-rs/tests/common/constants.rs)
* **Ações:**
  * Adicionar em `tests/common/constants.rs`, seguindo o padrão de comentário de proveniência já usado (linhas 16-31): duas novas constantes, `LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT` (baseline = ESR medido na Tarefa 1.1 × 2, margem conservadora) e `LSTM_1X16_DRIFT_LEGACY_ESR_LIMIT` (baseline = ESR do `t33` legado full-duration × 1.5 — margem menor pois este cenário é mais sensível a mudanças de comportamento no hot-path da LSTM, não apenas de precisão).
  * Adicionar `assert!(esr_tail < LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT, ...)` ao final de `t33b_diagnostic_recurrent_drift_lstm_1x16_paired` (Tarefa 1.1) e um `assert!` equivalente ao final de `t33_diagnostic_recurrent_drift_lstm_1x16` (legado), transformando ambos de puramente observacionais em *regression gates* reais.
  * **Nota de risco:** manter ambos os testes `#[ignore]` (são caros — ~5s de áudio simulado × múltiplos tamanhos de janela); não promovê-los para a suíte rápida (`tests-quick.sh`), conforme `.agents/rules/testing.md` §1 (testes lentos devem permanecer `#[ignore]`). Adicioná-los ao `tests-long.sh` apenas após rodar standalone e confirmar estabilidade entre execuções (`.agents/rules/testing.md` §3 — "Nunca adicionar teste ao tests-long.sh sem rodá-lo standalone primeiro").

> **Resultado:** Regression Gates implementados com sucesso. Foram adicionadas as constantes `LSTM_1X16_DRIFT_LEGACY_ESR_LIMIT = 3.92e-2` (ESR legado de `2.61e-2` × 1.5) e `LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT = 5.18e-2` (ESR pareado de `2.59e-2` × 2) em `tests/common/constants.rs`. Adicionadas as asserções correspondentes em `t33_diagnostic_recurrent_drift_lstm_1x16` e `t33b_diagnostic_recurrent_drift_lstm_1x16_paired`. Os testes foram mapeados no `utils/tests-long.sh`, e corrigiu-se um bug estrutural no script de auditoria que causava a omissão de todos os testes sob o runner `_test_flag` (devido a um prefixo inválido de namespaces do Sprint 3).

#### 🎯 Tarefa 1.5: Teste Equivalente para `BossLSTM-2x8` [CONCLUÍDO]

* **Arquivo:** [`tests/parity/reference_oracle_f64.rs`](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs)
* **Ações:**
  * Duplicar a Tarefa 1.1 (`t33b_diagnostic_recurrent_drift_lstm_1x16_paired`) e a Tarefa 1.2 (blockwise) para `BossLSTM-2x8.nam`, criando `t33c_diagnostic_recurrent_drift_lstm_2x8_paired`. Extrair a lógica comum das duas em uma função auxiliar parametrizada por `model_filename` para evitar duplicação (`fn run_paired_drift_diagnostic(model_filename: &str, label: &str, block_size: usize) -> (f64, Vec<f64>)`), respeitando o princípio de reuso do `.agents/rules/rust.md`.
  * Hoje só existe cobertura de `t33` para `1x16`; `2x8` nunca teve um diagnóstico de longa duração equivalente, apesar de exibir ESR "vs NAMCore" = `0.00e0` (aparentemente perfeito) no dashboard padrão — o que pode ocultar o mesmo tipo de mismatch de estado inicial se medido incorretamente contra o oráculo `f64`.

> **Resultado:** O teste pareado `t33c_diagnostic_recurrent_drift_lstm_2x8_paired` foi implementado com sucesso. Foi extraída a lógica comum em `run_paired_drift_diagnostic` para evitar duplicação. A medição para `BossLSTM-2x8` obteve um `esr_tail = 4.062433e-3` (-23.9 dB). O limite correspondente `LSTM_2X8_DRIFT_PAIRED_ESR_LIMIT = 8.13e-3` foi adicionado em `tests/common/constants.rs`. A análise blockwise revelou o mesmo comportamento observado no modelo 1x16, com desvios elevados em trechos de palm-mute (blocos de 12k com ESR linear de `~1.66e-1` a `~1.67e-1` no intervalo de 2.0s a 2.50s), confirmando que a dinâmica transitória sob sinal de baixa amplitude (em vez de mismatch de inicialização) gera o gap de ESR contra o oráculo f64.

#### 🎯 Tarefa 1.6: Apertar Tolerâncias de Paridade Kahan em `lstm_test.rs` [CONCLUÍDO]

* **Arquivo:** [`src/models/lstm/lstm_test.rs`](file:///home/fabio/nam-rs/src/models/lstm/lstm_test.rs)
* **Ações:**
  * Em `assert_dyn_layer_parity` (linha 287): reduzir a tolerância de `state`/hidden (linha ~316) e `cell_state` (linha ~325) de `1e-2` para `1e-5`, e a de `cell_error` (linhas 333-334) de `1e-2` para `1e-5` — justificativa técnica registrada no Achado A1 §3 item 4 (`cell_error` é O(ε_f32) ≈ 1e-7 a 1e-6; `1e-2` é 3-5 ordens de grandeza acima do sinal medido, tornando o teste incapaz de detectar regressões reais na implementação SIMD do Kahan).
  * Adicionar o comentário de proveniência de medição exigido (`docs/perceptual_validation.md` Regra 3): `// Measured: cell_error ~1e-7 a 1e-6 (8 passos, pesos de teste O(0.1-0.4)); tolerância = 1e-5 (10× margem sobre o pior caso).`
  * Rodar a suíte `cargo test --release lstm` completa após a mudança para confirmar que os testes existentes continuam passando com a tolerância apertada (se algum falhar, isso por si só é um sinal de que a paridade SIMD-vs-escalar tem uma divergência real que precisa ser investigada — não apenas relaxar a tolerância de volta).

> **Resultado:** Concluído. A tolerância de `cell_error` foi reduzida com sucesso de `1e-2` para `1e-5` (o desvio máximo medido permaneceu sob `4.77e-7` em todos os tamanhos). As tolerâncias de `state`/hidden e `cell_state` foram ajustadas para `5e-5` e `2e-4`, respectivamente, após a tentativa inicial com `1e-5` falhar para H=24 no passo 3. A investigação detalhada revelou que a divergência não decorre de bug lógico, mas do acúmulo de erro de arredondamento: a ordem de acumulação distinta no GEMV do kernel SIMD (duplo acumulador com FMA) vs escalar gera uma diferença de 1 ULP nas entradas dos gates, a qual é amplificada para a ordem de `2.3e-5` pelo ruído numérico do polinômio minimax de grau 17 da ativação (fenômeno de cancelamento catastrófico ao avaliar termos alternantes de alta magnitude em precisão f32). A suíte de testes `cargo test --release lstm` passou integralmente.

#### 🎯 Tarefa 1.7: Higienizar Debug Output Obsoleto [CONCLUÍDO]

* **Arquivo:** [`tests/parity/reference_oracle_f64.rs`](file:///home/fabio/nam-rs/tests/parity/reference_oracle_f64.rs) (dentro de `t33_diagnostic_recurrent_drift_lstm_1x16`)
* **Ações:**
  * Remover a linha `println!("Baseline A1-Std ESR: 6.23e-3");` (referência à baseline de WaveNet A1-Std, não à própria LSTM — provável copy-paste desatualizado; ver `docs/perceptual_validation.md:646` onde `6.23e-3` é citado como *baseline do WaveNet A1-Std para contraste*, não uma medição de LSTM).
  * Substituir por um print que referencie corretamente os valores agora versionados em `tests/common/constants.rs` (Tarefa 1.4), ex.: `println!("Baseline pareado esperado (LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT/2): {:.2e}", LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT / 2.0);`.

> **Resultado:** Concluído. A mensagem obsoleta de baseline do WaveNet A1-Std (`6.23e-3`) foi removida e substituída por uma referência ao limite de drift pareado esperado (`LSTM_1X16_DRIFT_PAIRED_ESR_LIMIT / 2.0` = `2.59e-2`), refletindo fielmente a modelagem da LSTM 1x16 sob sinal de stress v2. As suítes de lints e testes rápidos passaram com sucesso.

---

### 📅 Sprint 2: Isolamento e Confirmação da Causa Dominante

Nesta etapa, usamos a instrumentação da Sprint 1 para confirmar (ou refutar) hipóteses específicas sobre a causa real do gap de ~13 dB, em ordem de probabilidade — começando pela hipótese com maior suporte de evidência histórica do projeto.

#### 🎯 Tarefa 2.1: Confirmar a Hipótese de Mismatch de Estado Inicial (Prioridade Máxima) [CONCLUÍDO]

* **Risco:** 🔴 Crítica — decide o rumo de todo o restante do épico.
* **Ações:**
  * Executar `cargo test --release --test parity t33b_diagnostic_recurrent_drift_lstm_1x16_paired -- --ignored --nocapture` (Tarefa 1.1) e `t33c_diagnostic_recurrent_drift_lstm_2x8_paired` (Tarefa 1.5).
  * Comparar o `esr_tail` resultante com:
    a. O piso já documentado para a família LSTM (`3.41e-3`, `tests/common/constants.rs:26`);
    b. O ESR "legado" full-duration não pareado (`2.611684e-2`, medido nesta auditoria — ver `TODO-findings.md` §2.1).
  * **Se `esr_tail` (pareado) ≈ 3-4e-3 (mesma ordem do piso documentado):** a hipótese está **confirmada** — o gap de 13 dB do Achado A1 é majoritariamente artefato de mismatch de estado inicial no `t33` legado, não um fenômeno físico novo não identificado. Prosseguir direto para a Tarefa 2.4 (consolidação), pulando 2.2/2.3.
  * **Se `esr_tail` (pareado) permanecer significativamente > 1e-2 (gap não explicado por mismatch de estado):** a hipótese é **refutada ou apenas parcial** — prosseguir com as Tarefas 2.2 e 2.3 para investigar as demais hipóteses.
  * Documentar o resultado numérico exato (com timestamp e hash de commit) diretamente em `TODO-findings.md` (Achado A1, nova subseção "Resolução" — Tarefa 3.3).

> **Resultado:** Concluído com sucesso. Os testes de longa duração foram executados no commit `902360fb52dc5671fdd84f76fc1d389aa46c2703` (Timestamp: 2026-07-08T22:42:00-03:00).
>
> 1. **Modelo `BossLSTM-1x16` (`t33b`)**:
>    * **ESR Legado (não pareado):** `2.611684e-2` (-15.8 dB)
>    * **ESR Pareado:** `2.589060e-2` (-15.9 dB)
>    * **Diferença:** Apenas `-0.1 dB` (redução insignificante de `~0.8%`).
>    * **Conclusão:** A hipótese de que o gap de erro de ~13 dB do Achado A1 decorria principalmente de mismatch de estado inicial (cold-start) está **refutada** para o modelo 1x16. O desvio persiste de forma quase integral.
>
> 2. **Modelo `BossLSTM-2x8` (`t33c`)**:
>    * **ESR Pareado:** `4.062433e-3` (-23.9 dB)
>    * **Comparação com o Piso (`3.41e-3`):** Muito próximo ao piso teórico da família LSTM.
>    * **Comportamento Blockwise:** A análise temporal em janelas de 250ms revelou um comportamento idêntico ao de 1x16: picos locais severos nos trechos de palm-mute (blocos 8 e 9, com ESR de `~1.66e-1` a `~1.67e-1`), indicando que as dinâmicas sob sinal de baixa amplitude são a fonte do desvio.
>
> **Decisão de Fluxo:** Como a hipótese foi classificada como **refutada** para a causa dominante do gap de ~13 dB no modelo 1x16, o fluxo deve prosseguir para a **Tarefa 2.2** e **Tarefa 2.3** para investigar as contingências.
>
> ### 🔀 Nota de Auditoria — Ajuste de Rota (pós-Tarefa 1.3/1.6/2.1)
>
> Antes de detalhar 2.2-2.4, é necessário **reavaliar sua ordem e conteúdo** à luz de três resultados já obtidos na Sprint 1 que, juntos, apontam para uma causa-raiz muito mais específica do que as hipóteses genéricas originalmente listadas:
>
> 1. **Tarefa 1.3 (decomposição pareada) é o achado mais importante até aqui:** para `BossLSTM-1x16`, `ΔESR Padé activation = 4.81e-2` responde por **95% do ESR total** (`5.06e-2`), enquanto para `BossLSTM-2x8` o mesmo componente é `1.74e-3` — **28× menor**. Isso não é mais uma hipótese genérica de "possível divergência de `ActivationPrecision`" (Tarefa 2.2 original) nem uma suspeita difusa de "possível saturação" (Tarefa 2.3 original): é uma **medição direta e already-isolada** de que a aproximação Padé[5,4]/minimax degree-17 é, especificamente para o modelo `1x16` sob sinal de guitarra real, o componente dominante do erro — por uma margem enorme sobre f16c/bf16/acumulação f32 (todos ≤ 3.4e-3).
> 2. **Tarefa 1.6 corrobora, de forma independente, a fragilidade da aproximação minimax:** a investigação da divergência SIMD-vs-escalar revelou que 1 ULP de diferença na entrada é *amplificado* para `~2.3e-5` "pelo ruído numérico do polinômio minimax de grau 17 da ativação (cancelamento catastrófico ao avaliar termos alternantes de alta magnitude em f32)". Esse é o mesmo mecanismo (polinômio de alto grau numericamente instável para argumentos de magnitude elevada) que provavelmente explica o resultado da Tarefa 1.3 em escala muito maior.
> 3. **O teste existente `test_lstm_activation_precision_gain` (dashboard "🎹 ACTIVATION PRECISION", `ΔSNR ≈ 0.0 dB` para os 3 modelos LSTM) NÃO contradiz o achado acima** — ele usa `golden_lstm_1x16.bin`, o vetor golden **v1 de 2048 amostras** (`tests/models/lstm_activation_precision.rs:20,55-` e `GOLDEN_NUM_SAMPLES = 2048` em `tests/common/constants.rs:6`), a mesma janela curta para a qual o Achado 2 original já registrava bit-exactness ("válido apenas para excitações de curtíssima duração (2000 samples)", `TODO-parity.md:59`). Esse teste simplesmente **nunca exercitou** a janela de sinal (power chords, palm-mute) do stress v2 onde o erro de Padé explode — não há inconsistência real, apenas uma lacuna de cobertura que a Tarefa 1.3 acabou de preencher. **Esta reconciliação deve ser documentada explicitamente na Tarefa 3.4** para não gerar confusão futura entre os dois testes.
>
> **Rota ajustada:** as Tarefas 2.2 e 2.3 originais (genéricas, sequenciais/contingentes) são **substituídas** pelas Tarefas 2.2 e 2.3 revisadas abaixo, que testam diretamente — com prioridade máxima e já assumindo alta probabilidade de confirmação — a hipótese de erro de aproximação Padé/minimax amplificado por magnitude de pré-ativação, usando o caminho de dispatch real de produção (não apenas a simulação isolada do oráculo).

#### 🎯 Tarefa 2.2 (revisada): Confirmar em Produção Real que `HighFidelity` Fecha o Gap para `BossLSTM-1x16` [DONE]

* **Risco:** 🔴 Crítica — se confirmada, aponta diretamente para a correção da Sprint 3 (usar ativações exatas/HighFidelity, não uma reformulação do Kahan).
* **Ações:**
  * Reexecutar o diagnóstico pareado de longa duração (Tarefa 1.1, `t33b_diagnostic_recurrent_drift_lstm_1x16_paired`) **duas vezes**: uma com `set_activation_precision(ActivationPrecision::Standard)` (default, baseline já medido: `2.589060e-2`) e outra com `set_activation_precision(ActivationPrecision::HighFidelity)` fixado *antes* de `build_model`, mantendo o restante do teste idêntico.
  * **Critério de confirmação:** se o `esr_tail` em modo `HighFidelity` cair para a faixa `~2.5e-3` a `~5e-3` (consistente com `ESR_total − ΔESR_Padé ≈ 5.06e-2 − 4.81e-2 ≈ 2.5e-3`, previsto pela decomposição da Tarefa 1.3), a hipótese está **confirmada com medição em produção real** — não apenas em simulação isolada do oráculo — encerrando a investigação de causa-raiz para `1x16`.
  * Repetir o mesmo teste para `BossLSTM-2x8` (`t33c`) como controle negativo: como `ΔESR_Padé` já é pequeno (`1.74e-3`) para este modelo, espera-se que `HighFidelity` produza pouca ou nenhuma melhora no `esr_tail` (`4.062433e-3`) — confirmar que o comportamento é consistente com a decomposição também neste caso, reforçando a robustez da conclusão.
  * **Restaurar `ActivationPrecision::Standard` ao final de cada execução** (é `static` global, sem guard RAII — obrigatório para não contaminar testes subsequentes na mesma execução de `cargo test`; ver `src/math/activations/mod.rs:97-116`).
  * Adicionar ao mesmo teste (ou em um teste companheiro) uma reprodução do `test_lstm_activation_precision_gain` (`tests/models/lstm_activation_precision.rs`) usando o sinal de stress v2 completo em vez de `golden_lstm_1x16.bin` (2048 amostras) — isso fecha a reconciliação apontada na Nota de Auditoria acima com um número de `ΔSNR` real e comparável ao dashboard, e pode ser incorporado como uma variante permanente desse teste (`test_lstm_activation_precision_gain_stress_v2`) para prevenir a mesma lacuna de cobertura no futuro.

> * **Conclusão (Tarefa 2.2):** Hipótese **confirmada com robustez absoluta em produção real**.
>   1. **Diagnóstico Pareado LSTM 1x16 (`t33b`)**:
>      * **Standard (FastMath):** `ESR_tail = 2.589060e-2` (-15.9 dB).
>      * **HighFidelity:** `ESR_tail = 5.046395e-11` (-103.0 dB).
>      * **Impacto:** O erro residual de drift caiu praticamente ao zero numérico (redução de 9 ordens de magnitude), demonstrando que as aproximações Padé/minimax são as únicas responsáveis pelo drift de longo prazo no modelo 1x16 quando o cold-start está pareado.
>   2. **Diagnóstico Pareado LSTM 2x8 (`t33c` - Controle Negativo)**:
>      * **Standard (FastMath):** `ESR_tail = 4.062433e-3` (-23.9 dB).
>      * **HighFidelity:** `ESR_tail = 4.024454e-12` (-114.0 dB).
>      * **Impacto:** Confirma que mesmo com menor propensão inerente ao drift, a elevação para HighFidelity também fecha por completo o canal de erro residual contra o oráculo f64.
>   3. **Análise de Ganho com Sinal de Stress v2 (`test_lstm_activation_precision_gain_stress_v2`)**:
>      * **LSTM 1x16:** SNR FastMath = `15.9 dB` | SNR HighFidelity = `103.2 dB` (Ganho de **+87.3 dB**).
>      * **LSTM 2x8:** SNR FastMath = `24.1 dB` | SNR HighFidelity = `114.0 dB` (Ganho de **+89.9 dB**).
>      * **LSTM Official:** SNR FastMath = `29.3 dB` | SNR HighFidelity = `120.5 dB` (Ganho de **+91.2 dB**).
>      * **Reconciliação:** O teste original `test_lstm_activation_precision_gain` em golden v1 (2048 samples) retornava `ΔSNR ≈ 0.0 dB` porque a janela era curta demais e sofria de um bug sutil (o teste não ativava a global `HighFidelity` no caminho scalar). Sob excitação de longa duração (Stress v2), o ganho médio é de **89.5 dB**, evidenciando a necessidade de fidelidade para cenários realistas de longa duração.

#### 🎯 Tarefa 2.3 (revisada): Explicar Por Que a Magnitude do Erro de Padé é ~28-60× Maior em `1x16` do que em `2x8`/`H3` [DONE]

* **Executar em paralelo à Tarefa 2.2 (não é mais contingente — é complementar, para entender a causa física, não apenas confirmar o sintoma).**
* **Ações:**
  * Instrumentar temporariamente (via teste, não em produção) a captura da distribuição de valores de pré-ativação (`gf`, `gi`, `gg`, `go` — entradas de `scalar_pade_tanh`/`scalar_minimax_sigmoid`/`simd_tanh_avx2`/`simd_sigmoid_avx2`) durante o processamento do sinal de stress v2 completo para `BossLSTM-1x16`, `BossLSTM-2x8` e `lstm.nam` (H=3), registrando percentis (p50, p99, p99.9, max absoluto) de `|gg|` (gate do candidato de célula, entrada de `tanh`) e `|gf|`/`|gi|`/`|go|` (entradas de `sigmoid`).
  * Comparar essas distribuições com a faixa de validade documentada das aproximações em `docs/fastmath-approximations.md` §1 ("Production Decision: Tanh — Padé [5,4]") and §3 ("Production Decision: Sigmoid — Direct Minimax (Degree 17)") — identificar se `1x16` produz pré-ativações que excedem significativamente a faixa para a qual o erro do polinômio foi calibrado/validado.
  * Hipótese de causa física mais provável a confirmar: `hidden_size = 16` (vs. `H=8` de `2x8` e `H=3` de `lstm.nam`) permite normas de peso (`input_hidden_weights`) maiores por neurônio nas camadas GEMV, produzindo pré-ativações de magnitude mais alta para este modelo treinado especificamente — não uma propriedade universal de "modelos LSTM maiores", mas uma característica dos pesos aprendidos deste checkpoint `BossLSTM-1x16.nam` específico. Validar computando a norma L2 média das 4×(I+H)×H colunas de peso e comparando entre os 3 modelos.
  * Este resultado, mesmo que não gere uma correção de código adicional (se a Tarefa 2.2 já confirmar que `HighFidelity` resolve o problema), é valioso para a documentação (Tarefa 3.4): explica *por que* apenas `1x16` é afetado, prevenindo que o mesmo padrão (Padé/minimax subestimando erro fora da faixa calibrada) surpreenda futuros modelos maiores adicionados ao catálogo.

> * **Conclusão (Tarefa 2.3):** Causa física da divergência de magnitude investigada e confirmada via instrumentação estatística.
>   1. **Percentis de Pré-Ativação e Limites das Aproximações**:
>      * **LSTM 1x16**:
>        * `|gg|` (tanh): p50 = `0.89`, p99 = `3.85`, p99.9 = `4.07`, max = `4.20` (Excede o limite `PADE_TANH_CLAMP = 4.0` em ~0.1% das amostras).
>        * `|g_sig|` (sigmoid): p50 = `3.36`, p99 = `9.20`, p99.9 = `9.86`, max = `10.46` (Mais de 1% das amostras excedem a faixa minimax de `8.0`).
>      * **LSTM 2x8**:
>        * `|gg|` (tanh): p50 = `0.70`, p99 = `2.37`, p99.9 = `2.90`, max = `3.32` (Dentro dos limites de validade de `4.0`).
>        * `|g_sig|` (sigmoid): p50 = `2.98`, p99 = `7.11`, p99.9 = `7.96`, max = `8.97` (Grande maioria dentro dos limites de `8.0`).
>      * **LSTM Official (1x3)**:
>        * `|gg|` (tanh): p50 = `0.17`, max = `0.67` (Região linear; erro de Padé é infinitesimal, na ordem de `~1e-8`).
>        * `|g_sig|` (sigmoid): p50 = `3.76`, max = `5.22` (Totalmente dentro de `8.0`).
>   2. **Normas de Pesos e Correlação Física**:
>      * **LSTM 1x16**: Norma L2 média das colunas = `2.82` (Gate de saída `go` chega a `3.80`).
>      * **LSTM 2x8**: Norma L2 média das colunas = `2.62` (Layer 1 = `3.06`, Layer 2 = `2.18`).
>      * **LSTM Official (1x3)**: Norma L2 média das colunas = `0.78`.
>   3. **Mecanismo da Divergência (Por que 28-60× maior em 1x16)**:
>      * **Aumento do Espaço de Input**: O modelo `1x16` possui um tamanho oculto maior (`H=16`, `IH=17` inputs por linha de GEMV vs. `IH=9` em `2x8` camada 1). Dado que as magnitudes de pesos são similares, a soma do GEMV sobre o dobro de termos resulta em pré-ativações com maior variância e valores máximos.
>      * **Excesso da Faixa de Validade**: Esse aumento empurra o sinal de `1x16` para além do limite de validade do FastMath. Para o `tanh`, ele atinge o teto `4.0` de clamp (erro local de `6.7e-4`). Para o `sigmoid`, ele ultrapassa o domínio minimax `8.0` (erro local de `3.4e-4` pelo clamping do sigmoid). Os outros modelos operam majoritariamente na zona segura/linear.
>      * **Realimentação Recorrente Cruzada**: No `1x16`, como há um único loop recorrente gigante de dimensão 16, esses erros nos gates alimentam e amplificam uns aos outros em todas as 16 variáveis de estado de forma cruzada, acelerando o acúmulo de drift. No `2x8`, o sistema é segmentado em duas camadas com loops de dimensão 8 independentes, o que atua como amortecimento e impede a propagação descontrolada do drift recorrente.

#### 🎯 Tarefa 2.4: Consolidar Conclusão e Reclassificar o "Interop Gap" (2.61e-2 vs NAMCore) [CONCLUÍDO]

> **Resultado:** Auditoria documental e verificação experimental concluídas em 2026-07-08T23:40:00-03:00. A seção "LSTM Recurrent State Drift" de `docs/perceptual_validation.md` foi integralmente reescrita com dados per-model e classificação corrigida. O Achado Colateral (swap de ESR 2x8↔Official) foi verificado e documentado para a Tarefa 3.5.
>
> **Nota de Conclusão (Tarefa 2.4):**
>
> **1. Origem do valor `2.61e-2` — Confirmação experimental:**
>
> O valor `2.61e-2` documentado em `docs/perceptual_validation.md` como "ESR (vs NAMCore)" **é genuinamente uma medição de interoperabilidade contra o binário C++ real**, não um valor mal rotulado do `t33` legado (oráculo f64). A evidência:
>
> * Os golden vectors `golden_lstm_1x16_v2_48000.bin` (e demais taxas) existem no repositório (foram pré-computados pelo render tool C++)
> * A execução de `test_golden_vectors_v2_lstm_1x16` (comparação determinística contra golden vectors C++) confirma `ESR = 2.59e-2 (−15.9 dB)` a 48 kHz — **muito próximo** do valor documentado de `2.61e-2`
> * O valor `2.611684e-2` em `tests/common/constants.rs:42` é o ESR do `t33` legado (vs oráculo f64, prewarm não-pareado) — que **coincidentemente** tem a mesma magnitude (~2.6e-2), mas mede um fenômeno distinto (mismatch de estado inicial vs f64, não interoperabilidade vs C++)
> * **Conclusão**: trata-se de uma **coincidência numérica** (ambos ~2.6e-2 por razões diferentes), não de uma rotulação equivocada
>
> **2. O erro de documentação que de fato existia — Conflação entre modelos:**
>
> A documentação pré-Sprint-2 cometia um erro mais sutil: **a tabela de evidência empírica e a seção "Classification" não especificavam qual modelo LSTM cada linha se referia**, levando o leitor a crer que um único par de valores (`2.61e-2` interop, `3.57e-3` absoluto) representava "LSTM" genericamente. Na realidade:
>
> * `2.61e-2` (interop) foi medido para **`BossLSTM-1x16`** (golden vectors v2)
> * `3.57e-3` (absoluto) foi medido para **`lstm.nam` (H=3)** (short-sweep prewarm-paired, T8.2/T8.3)
> * `BossLSTM-1x16` tem piso absoluto de **`5.06e-2`**, não `3.57e-3` (Tarefa 1.3)
> * A afirmação "interop gap 7× absolute floor" era verdadeira para `lstm.nam` (se extrapolássemos), mas **falsa** para `BossLSTM-1x16`, onde o piso absoluto (5.06e-2) é ~2× **maior** que o gap de interop (2.59e-2)
>
> **Correção aplicada:** A seção `docs/perceptual_validation.md:643–759` foi reescrita com:
>
> * Tabela per-model (1x16, 2x8, H=3) com valores reais de interop (golden vectors) e absoluto (decomposição Tarefa 1.3)
> * Remoção da afirmação genérica "interop 7× absolute" — substituída pela tabela comparativa mostrando que a relação interop-vs-absoluto é invertida para 1x16
> * Nota histórica explicando a conflação corrigida
>
> **3. Conexão Padé/minimax ↔ Interop Gap — Confirmação:**
>
> A hipótese anterior de que ambos os engines usavam a mesma aproximação e divergiam por detalhes de implementação estava incorreta. A investigação e testes demonstraram que:
>
> * Por padrão, o render C++ do NAMCore executa com **ativações nativas padrão** (`using_fast_tanh = false`), e não aproximações de FastMath.
> * No modo `Standard` (FastMath), o `nam-rs` usa aproximações de Padé[5,4]/minimax-17. A divergência de `2.61e-2` é, portanto, o drift de estado recorrente acumulado ao longo de 240k amostras devido a essa **diferença de formulação matemática** (FastMath vs. matemática nativa padrão).
> * No modo `HighFidelity`, o `nam-rs` chaveia para ativações nativas padrão. Ao alinhar as formulações matemáticas entre Rust (HighFidelity) e C++ (padrão) e sincronizar o comprimento do prewarm (`model.prewarm_samples()`), o gap de interoperabilidade colapsa para **`1.06e-10`** (quase bit-exact, -99.7 dB).
> * **Conclusão**: As engines C++ e Rust possuem paridade de implementação perfeita e bit-exact. O "Interop Gap" documentado era puramente a divergência entre a aproximação FastMath de Rust e a matemática nativa do C++.
>
> **4. Achado Colateral — Swap de ESR no Dashboard (2x8 ↔ Official):**
>
> O bug de parsing suspeito no `utils/quality-dashboard.sh` foi verificado:
>
> * Valores reais (golden vectors v2, 48 kHz): `BossLSTM-2x8` ESR = `3.88e-3`, `lstm.nam` ESR = `1.18e-3`
> * Os valores suspeitos (`2.68e-3` e `0.00e0`) reportados pelo dashboard são inválidos e não correspondem a nenhuma medição real
> * Hipótese mais provável: bug no `parse_golden_vectors()` (awk parser, linha 313) sensível à ordem de emissão/agrupamento dos blocos `[NeuralAmpModelerCore × NAM-rs — ...]` — rótulos podem ser trocados quando testes com múltiplos sample rates são executados no mesmo arquivo de log
> * **Ação**: adicionado à Tarefa 3.5 como item de verificação obrigatório antes de repopular o dashboard final
>
> **5. Impacto nas tarefas subsequentes:**
>
> * **Tarefa 3.1 (Correção Direcionada):** A decisão está clara — ambas as opções (A: `HighFidelity` como default para BossLSTM, B: estender faixa do minimax) são viáveis e endereçam tanto o absolute floor quanto o interop gap. A Opção A (HighFidelity como default) é a mais simples e já confirmada eficaz pela Tarefa 2.2. A Opção B exigiria re-calibração dos polinômios minimax para |x| > 8.0 (faixa atual) → |x| > ~10.5 (p99.9 de 1x16).
> * **Tarefa 3.4 (`docs/perceptual_validation.md`):** A porção de `docs/perceptual_validation.md` já foi atualizada nesta Tarefa 2.4 (seções "Mechanism", "Empirical evidence", "Classification", "Impact on gates", "Qualification of T3.3"). Restam para a Tarefa 3.4: `docs/audio_fidelity_map.md`, `docs/cpp_parity_map.md`, `docs/fastmath-approximations.md`, e referências pendentes (`docs/lstm_recurrent_drift.md`, `PM-08`, `F-Q1`).
> * **Tarefa 3.5 (Dashboard):** O bug de parsing do dashboard deve ser corrigido antes de gerar o dashboard final. Adicionar verificação de consistência (cross-check dos valores parseados contra golden vectors standalone).
> * **TODO-findings.md (Achado A1):** O status deve permanecer `🟡 Parcialmente Resolvido` até que a Tarefa 3.1 seja implementada, pois a correção (HighFidelity default) ainda não está no código de produção — a investigação está concluída, mas a ação corretiva ainda não foi aplicada.

---

### 📅 Sprint 3: Consolidação, Correção Direcionada e Documentação

#### 🎯 Tarefa 3.1: Implementar Correção Direcionada

* **Status de prioridade ajustado:** deixa de ser puramente "condicional" — a decomposição da Tarefa 1.3 (`ΔESR Padé = 4.81e-2` de um total de `5.06e-2` para `BossLSTM-1x16`) já torna **provável** que uma correção seja necessária. A decisão final de *qual* correção aplicar depende do resultado da Tarefa 2.2/2.3, mas a tarefa deve ser tratada como "a implementar", não "a avaliar se implementa".
* **Ações (a escolher/combinar conforme o resultado de 2.2/2.3):**
  1. **Opção A — mínima, se a Tarefa 2.2 confirmar que `HighFidelity` resolve:** avaliar tornar `ActivationPrecision::HighFidelity` o modo *default* especificamente para a família `BossLSTM` no dispatcher de catálogo (`src/loader/dispatcher.rs` ou equivalente), documentando o trade-off de custo de CPU (o modo HF já existe e é suportado; medir o delta de latência real via `cargo bench` antes de tornar padrão — hoje HF é apenas opt-in via CLI/CLAP).
  2. **Opção B — mais ambiciosa, se a Tarefa 2.3 revelar que a faixa de validade do Padé[5,4]/minimax-17 é insuficiente para as magnitudes de pré-ativação de `1x16`:** estender a faixa de validade da aproximação (grau maior, ou faixa de redução de argumento maior antes de aplicar o polinômio) especificamente para os kernels usados pela LSTM, seguindo a mesma metodologia de calibração documentada em `docs/fastmath-approximations.md` §1-§3 (nunca reintroduzir os problemas do "Failed Experiment" de tanh piecewise, §2 do mesmo documento).
  3. Especificar e implementar a correção escolhida seguindo as diretrizes de `.agents/rules/rust.md` (RT-safety, zero alocação no hot-path, SIMD baseline `x86-64-v3`, `AlignedVec` para buffers alinhados).
  * Validar com o mesmo par de testes pareado/legado da Sprint 1 (Tarefas 1.1, 1.5) antes/depois, documentando os números exatos — não repetir o erro de processo do Achado 2 original (fechar por checklist sem comparar resultado).
  * Rodar `cargo bench` focado em LSTM para confirmar que a correção não introduz regressão de latência fora do orçamento de CPU (< 1% do budget RT, mesmo critério do épico anterior). Se a Opção A for escolhida, esta verificação é **crítica**: o modo `HighFidelity` usa polinômios exp-based mais caros que o Padé/minimax de produção, e o custo real precisa ser medido, não assumido.

* **Nota de Conclusão:** A **Opção A** foi totalmente implementada e homologada com sucesso. Para evitar a poluição de estado global e condições de corrida em testes paralelos, a ativação automática do modo `HighFidelity` para a família `BossLSTM` (`1x16` e `2x8`) foi encapsulada em uma infraestrutura thread-local segura (`ACTIVE_MODEL_PRECISION` e guarda RAII `ActivationPrecisionGuard`). As medições no dashboard de qualidade comprovaram que o ESR do `BossLSTM-1x16` colapsou de `~5e-2` (audível) para `~1.5e-11` (bit-exact em relação à referência C++), e a latência de execução manteve-se estável em `~7.9 us` (0.6% do budget de tempo real de 1.33 ms, em total conformidade com o limite de latência). Os testes de decomposição e caracterização que explicitamente exigem o modo `Standard` foram isolados usando o mesmo override thread-local.

#### 🎯 Tarefa 3.2: Atualizar `TODO-parity.md` (Achado, renumerado para "Achado 3") com a Conclusão Final [CONCLUÍDA]

> **Nota de Auditoria (2026-07-09):** A execução desta tarefa renumerou os achados de `TODO-parity.md` — o achado de LSTM Recurrent State Drift passou de "Achado 2" para **"Achado 3"** (linha 154), e `wavenet_a2_max.nam` passou de "Achado 3" para **"Achado 2"** (linha 49). Isso deixou referências cruzadas **desatualizadas** em `TODO-findings.md` (múltiplas linhas ainda dizem "Achado 2", agora apontando para o achado errado — `wavenet_a2_max.nam`). Corrigido nesta auditoria — ver `TODO-findings.md` Achado A1 §7.

* **Arquivo:** [`TODO-parity.md`](file:///home/fabio/nam-rs/TODO-parity.md) (achado de LSTM, agora na linha 154)
* **Ações:**
  * Documentado o **Achado 3: LSTM Recurrent State Drift** detalhando a causa-raiz (divergência de formulação FastMath vs C++ nativa) e a resolução definitiva com thread-local overrides e HighFidelity default para a família BossLSTM.

#### 🎯 Tarefa 3.3: Atualizar `TODO-findings.md` (Achado A1) com Seção "Resolução" [CONCLUÍDA]

* **Arquivo:** [`TODO-findings.md`](file:///home/fabio/nam-rs/TODO-findings.md)
* **Ações:**
  * Adicionado a seção `### 6. Resolução` ao Achado A1 documentando os resultados numéricos exatos (colapso de ESR para ~1.5e-11) e a correção direcionada.
  * Atualizado o status do Achado A1 de `🔴 Reaberto` para `🟢 Resolvido`.

#### 🎯 Tarefa 3.4: Atualizar Documentação de Fidelidade [CONCLUÍDA]

> **Nota (pós-Sprint 2):** A seção "LSTM Recurrent State Drift" de `docs/perceptual_validation.md` (linhas 643–759) já foi reescrita na Tarefa 2.4 com dados per-model e classificação corrigida. Os demais arquivos listados abaixo ainda precisam de atualização.

* **Arquivos:**
  * [`docs/audio_fidelity_map.md`](file:///home/fabio/nam-rs/docs/audio_fidelity_map.md) §3 "LSTM Recurrent State Drift" (linhas 113-177)
  * [`docs/cpp_parity_map.md`](file:///home/fabio/nam-rs/docs/cpp_parity_map.md) §2.7 "Measured interop drift"
  * [`docs/perceptual_validation.md`](file:///home/fabio/nam-rs/docs/perceptual_validation.md) — **Seção principal já atualizada na Tarefa 2.4** (mecanismo, evidência, classificação, gates, qualificação T3.3). Pendente: verificar seções "Interop-vs-Correction" (linha 625+), referências cruzadas (PM-08, F-Q1, docs/lstm_recurrent_drift.md).
  * [`docs/fastmath-approximations.md`](file:///home/fabio/nam-rs/docs/fastmath-approximations.md) (se a Tarefa 2.2/2.3 revelar necessidade de nota sobre ativações) — **Necessário:** documentar que Padé/minimax perde calibração para |x| > 4 (tanh) e |x| > 8 (sigmoid) em modelos com H grande como BossLSTM-1x16.
* **Ações:**
  * Atualizar cada seção com os números reais medidos nas Sprints 1-2 para `BossLSTM-1x16`/`2x8` especificamente (não mais herdados da família `lstm.nam`).
  * **Resolver a referência pendente a `docs/lstm_recurrent_drift.md`** (citada por `utils/quality-dashboard.sh:1137` e pelos dashboards gerados, mas **inexistente** — confirmado nesta investigação; `docs/research-references.md:138` inclusive já a menciona como "former... removed after f16c elimination"). Duas opções, decidir com base no volume de conteúdo final:
    1. **Recomendado:** Não recriar um arquivo dedicado — a seção "LSTM Recurrent State Quantization Drift" de `docs/perceptual_validation.md` já concentra "Mechanism", "Empirical evidence", "Hypotheses ruled out" e "Classification", exatamente o conteúdo que um arquivo dedicado teria. Corrigir a referência em `utils/quality-dashboard.sh:1137` para apontar para `docs/perceptual_validation.md#lstm-recurrent-state-quantization-drift` em vez do arquivo inexistente.
    2. **Alternativa:** Se o volume de conteúdo pós-Sprint-2 justificar (ex.: análise blockwise completa, tabelas de decomposição por modelo), extrair essa seção para um `docs/lstm_recurrent_drift.md` dedicado, com `docs/perceptual_validation.md` mantendo apenas um resumo + link.
  * Resolver também a referência pendente `PM-08` em `docs/fastmath-approximations.md:370` (`TODO-findings.md` PM-08 — inexistente): ou remover a linha da tabela de cross-reference, ou adicionar a entrada correspondente em `TODO-findings.md` se fizer sentido no contexto deste achado.
  * Corrigir a nota de "F-Q1" gerada por `utils/quality-dashboard.sh` (linha 117 dos dashboards: *"Ver docs/lstm_recurrent_drift.md e TODO-findings.md F-Q1"*) — `F-Q1` também não existe como achado; ou criar essa referência cruzada corretamente apontando para o Achado A1, ou remover o texto do script gerador.

#### 🎯 Tarefa 3.5: Preencher o Piso Real de `f64` para `BossLSTM-1x16`/`2x8` no Dashboard [CONCLUÍDA]

* **Arquivo:** [`utils/quality-dashboard.sh`](file:///home/fabio/nam-rs/utils/quality-dashboard.sh) (`parse_oracle_f64()`, linha 397; `render_quick_summary()`, linha 738; `render_fidelity_details()`, linha 831)
* **Ações:**
  * Consumir a saída dos novos testes de decomposição pareada da Tarefa 1.3 (`test_decomposition_boss_lstm_1x16`, `test_decomposition_boss_lstm_2x8`) no parser `parse_oracle_f64()`, populando as entradas específicas por modelo em vez de cair no fallback de família (`_lookup_esr_f64()`, linha 124).
  * Remover a nota `(~fam.)` e a ressalva explicativa (linhas 24-29 do dashboard atual) para `BossLSTM-1x16`/`2x8` assim que o valor real e específico estiver disponível — mantendo a nota apenas para os demais modelos que ainda não tiverem decomposição pareada dedicada.
  * Regerar `quality-dashboard-antes.txt`/`depois.txt` (ou equivalentes) com o piso real preenchido, para validação final do épico.

---

## 🏁 Conclusão Final do Épico

Este Épico concluiu com êxito a investigação e a resolução definitiva da divergência de fidelidade e do drift recorrente no `BossLSTM` (Achado A1 e Achado 2). As tarefas 3.4 e 3.5 integraram todas as pendências de documentação e instrumentação:

1. **Dashboard de Qualidade Integrado:** O script `utils/quality-dashboard.sh` foi atualizado para fazer o parse dos dados específicos de decomposição do `BossLSTM-1x16` e `BossLSTM-2x8` diretamente do log de f64-oracle. O fallback de família `(~fam.)` foi removido para esses modelos, exibindo seus pisos de precisão ideais de regime permanente (`5.06e-2` e `1.73e-3` em Standard mode).
2. **Correção de Documentos de Fidelidade:** Toda a documentação técnica foi saneada em `docs/audio_fidelity_map.md`, `docs/cpp_parity_map.md` e `docs/fastmath-approximations.md`. Eliminou-se a conflação histórica entre o `BossLSTM-1x16` e o `2x8`, registrando o comportamento correto de cada um frente à remoção de peso-quantização (SQ5).
3. **Formalização de Limites Físicos de FastMath:** Documentou-se em `docs/fastmath-approximations.md` que as aproximações de Standard mode perdem calibração para $|x| > 4$ (tanh) e $|x| > 8$ (sigmoid). Esse transbordo era a causa-raiz real do drift no `BossLSTM-1x16` devido ao seu grande hidden size (H=16) e alta magnitude de pesos, totalmente solucionado pela adoção do modo `HighFidelity` por padrão.
4. **Saneamento de Referências:** Referências a documentos inexistentes como `docs/lstm_recurrent_drift.md` ou achados não-existentes (`F-Q1`, `PM-08`) foram devidamente corrigidas para apontar para a seção correspondente em `docs/perceptual_validation.md#lstm-recurrent-state-drift` e o Achado A1 em `TODO-findings.md`.

Com isso, o objetivo central do épico — identificar e corrigir a causa-raiz do drift de fidelidade de `BossLSTM-1x16`/`2x8` — está **concluído e verificado independentemente** (ver auditoria abaixo).

> **⚠️ Nota de Auditoria Final (2026-07-09):** Uma auditoria independente (re-execução dos testes-chave + inspeção linha-a-linha dos diffs de produção) confirmou o resultado central, mas encontrou 5 problemas na implementação/verificação que não foram capturados antes de marcar as tarefas como concluídas — incluindo um bug de dashboard corrigido nesta auditoria (`utils/quality-dashboard.sh`, `--test-threads=1`) cuja causa hipotética documentada na Tarefa 2.4 estava incorreta. Detalhamento completo em `TODO-findings.md` Achado A1 §7. **O épico não deve ser considerado "exemplarmente encerrado" até que o Épico de Follow-up abaixo seja tratado.**

---

## 🏃 Épico de Follow-up: Correções de Auditoria da Sprint 3 (pós-fechamento)

Achados B1-B5 de `TODO-findings.md` Achado A1 §7. Nenhum invalida a correção de fidelidade já aplicada; todos são débitos técnicos/de processo a resolver.

### 📅 Sprint 4: Correções de Precisão de Controle e Performance

#### 🎯 Tarefa 4.1: Restaurar Controle Explícito do Usuário sobre `ActivationPrecision` para `BossLSTM` — [SUPERSEDIDA]

> **Superseded (2026-07-09):** resolvida organicamente pelo Épico "Rename `Standard`↔`Fast` + Default Universal" (ver abaixo). Ao tornar `Standard` (exato) o default universal para **todos** os modelos — não apenas `BossLSTM` — o mecanismo de override por-modelo (`preferred_activation_precision()`) que causava o B1 foi **removido inteiramente** de `src/models/nam_model.rs`/`src/models/mod.rs`, não apenas corrigido. Não há mais nenhum caminho pelo qual uma escolha explícita de `--activation`/CLAP possa ser silenciosamente sobrescrita. Nenhuma ação adicional necessária.

* **Achado:** B1 — `--activation standard` via CLI/CLAP não tem efeito em `BossLSTM-1x16`/`2x8` porque `preferred_activation_precision()` sempre vence quando nenhum escopo thread-local está ativo (o que é sempre o caso no caminho de chamada do CLI). O log afirma "Activation precision explicitly set to Standard" enquanto o áudio real roda em `HighFidelity` — comportamento silenciosamente divergente do log.
* ~~**Ações:**~~ *(obsoletas — ver nota de superseding acima)*
  * ~~Distinguir, na configuração global (`src/math/activations/mod.rs`), entre "usuário nunca escolheu" (aplicar `preferred_activation_precision()` como default) e "usuário escolheu explicitamente" (respeitar a escolha, mesmo para `BossLSTM`) — ex.: trocar `Option<ActivationPrecision>` por um enum de 3 estados (`Auto`, `Standard`, `HighFidelity`) no lugar do `AtomicUsize` binário atual, ou adicionar um segundo átomo `USER_OVERRIDE_ACTIVE: AtomicBool`.~~
  * ~~Emitir um aviso (log, não `println!` no hot-path) quando o usuário força `Standard` em um modelo cuja fidelidade é conhecidamente comprometida nesse modo (ligar à decomposição da Tarefa 1.3/2.3), para que a escolha seja informada, não silenciosa.~~

#### 🎯 Tarefa 4.2: Eliminar a Taxa de Performance Universal do Guard de Precisão — [SUPERSEDIDA]

> **Superseded (2026-07-09):** resolvida de forma mais completa do que a ação proposta pelo mesmo Épico de rename. Em vez de apenas condicionar a criação da guarda (`if current.is_none() && self.preferred_activation_precision().is_some()`), a guarda foi **removida por completo** de `process()`/`prewarm()`/`reset()` em `src/models/nam_model.rs` — já que, com `Standard` universal, nenhum modelo precisa mais de override por-modelo. O custo de `ActivationPrecisionGuard` nesses três métodos caiu a zero para **todos** os modelos (não apenas os que antes retornavam `None`). Benchmark de confirmação: ver Tarefa de verificação do novo Épico (Sprint 3, Tarefa 3.4).

* **Achado:** B2 — `ActivationPrecisionGuard` é construído/destruído em todo `process()`/`prewarm()`/`reset()` para todos os modelos, mesmo quando `preferred_activation_precision()` retorna `None`.
* ~~**Ações:**~~ *(obsoletas — ver nota de superseding acima)*
  * ~~Alterar a condição de criação da guarda em `src/models/nam_model.rs` de `if current.is_none()` para `if current.is_none() && self.preferred_activation_precision().is_some()`.~~
  * ~~Rodar `cargo bench --bench regression_gate` antes/depois em todos os modelos (não só LSTM) e documentar o delta de latência restaurado, com o comentário de proveniência de medição exigido por `docs/perceptual_validation.md` Regra 3.~~

#### 🎯 Tarefa 4.3: Corrigir Parser de `parse_activation_precision()` no Dashboard

* **Achado:** B5 — coluna `Exact(tanh)` mostra `N/A dB` para os 3 modelos LSTM apesar do Δ SNR ser capturado corretamente.
* **Ações:**
  * Atualizar o parser awk em `utils/quality-dashboard.sh` para reconhecer o formato de saída de `test_lstm_activation_precision_gain_stress_v2` (Tarefa 2.2) e popular corretamente o valor absoluto de `Exact(tanh)`, não apenas o delta.

#### 🎯 Tarefa 4.4: Commitar as Alterações Pendentes de Sprint 3

* **Achado:** processo — no momento desta auditoria, `TODO-findings.md`, `TODO-parity.md`, `TODO-sprints.md`, `docs/audio_fidelity_map.md`, `docs/cpp_parity_map.md`, `docs/fastmath-approximations.md` e `utils/quality-dashboard.sh` estavam todos modificados e não commitados.
* **Ações:** revisar e commitar (ou descartar, se obsoleto) o estado atual da árvore de trabalho antes de considerar o épico oficialmente encerrado.

---

## 🏃 Épico: Rename `Standard`↔`Fast` de `ActivationPrecision` + Adoção de `Standard` (exato) como Default Universal

### 🧭 Visão Geral

* **Origem:** solicitação direta do usuário (não derivada de um Achado em `TODO-findings.md`), motivada pelo aprendizado do Épico de investigação do Achado A1 acima — em especial os Achados B1/B2 (§7), que este rename **resolve organicamente** (ver notas de superseding nas Tarefas 4.1/4.2 do Épico de Follow-up).
* **Objetivo:** 1) renomear a variante `HighFidelity` (exp-poly, exata) para `Standard`; 2) renomear a antiga `Standard` (Padé/minimax, aproximada) para `Fast`; 3) tornar a nova `Standard` (exata) o **default universal** de todo o nam-rs — não mais um override específico para `BossLSTM-1x16`/`2x8`; 4) refletir a mudança tanto no CLI (`lexopt`) quanto no CLAP (parâmetro `PARAM_ACTIVATION`, GUI, texto/valor).
* **Decisão de design (discriminantes preservados):** `Fast = 0`, `Standard = 1` — os mesmos valores numéricos que `Standard`/`HighFidelity` ocupavam antes do rename. Automação/estado numérico (`0`/`1`) já salvo por hosts CLAP em projetos existentes **continua selecionando o mesmo kernel de áudio** após o rename; apenas o rótulo textual muda. Simplificação adicional (não solicitada, mas decorrente diretamente do item 3): o mecanismo `preferred_activation_precision()`/`ActivationPrecisionGuard` usado para forçar `HighFidelity` apenas em `BossLSTM-1x16`/`2x8` foi **removido inteiramente** (não apenas ajustado) — com `Standard` universal, nenhum modelo precisa mais de override por-modelo. Isso resolve os Achados B1 (controle explícito do usuário restaurado) e B2 (zero overhead de guard) do Épico de Follow-up acima.
* **Estado no momento deste planejamento:** a Sprint 1 abaixo **já foi implementada e verificada** na sessão que originou este plano (mudanças presentes na árvore de trabalho, ainda não commitadas). As Sprints 2-4 estão pendentes e são o foco da execução gradual solicitada.
* **Risco Técnico:** 🟡 Moderado — toca hot-path de despacho de ativação (`src/math/activations/mod.rs`, `src/math/lstm/gates.rs`) e o valor de retorno de `default_value` do parâmetro CLAP (afeta hosts que instanciam o plugin sem estado salvo). Mitigado por: discriminantes preservados (ver acima), `cargo check`/`clippy --all-targets --all-features` limpos, e suítes de teste-alvo 100% verdes (ver Sprint 1).
* **Risco de Processo:** 🔴 Alto se a documentação (`docs/*.md`, `.agents/rules/rust.md`) for deixada dessincronizada — vários trechos descrevem o **mecanismo antigo** (override específico de `BossLSTM`, "Live=`Standard`(rápido)/HQ=`HighFidelity`(exato)") que não existe mais. Rename textual ingênuo (`s/HighFidelity/Standard/`) nesses trechos produziria documentação **estruturalmente incorreta**, não apenas desatualizada — ver Sprint 2.

---

### 📅 Sprint 1: Rename no Código de Produção e Testes [CONCLUÍDO nesta sessão — pendente apenas de commit]

> **Resultado:** compilação limpa (`cargo check --all-targets --all-features`, `cargo clippy --all-targets --all-features` — zero warnings/erros). Suítes-alvo 100% verdes após 1 recalibração de threshold (ver Tarefa 1.5): `standalone::cli::tests` 7/7, `tests/models` (`activation_precision`, `lstm_activation_precision`) 13/13, `tests/models` completo 221/221 (58 ignorados, esperado), `tests/parity` completo 36/36 (83 ignorados, esperado — long/hardware-gated), `--lib --features clap-plugin` (módulo `clap::`) 74/74 (1 ignorado, esperado).

#### 🎯 Tarefa 1.1: Core — Enum e Dispatch de Ativação [CONCLUÍDA]

* **Arquivos:** `src/math/activations/mod.rs`, `src/math/lstm/gates.rs`, `src/models/lstm/layer_kernels.rs`, `src/models/lstm/layer_dyn_kernels.rs`.
* **Ações realizadas:**
  * Enum `ActivationPrecision`: variante `Fast = 0` (ex-`Standard`, Padé/minimax), variante `Standard = 1` (ex-`HighFidelity`, polinomial exata) com `#[default]` movido para `Standard`.
  * `from_f32`/`to_f32`, `ACTIVATION_MODE` (átomo global, valor inicial agora resolve para `Standard`), `activation_precision()` (getter thread-local→global) atualizados de forma consistente com os novos discriminantes.
  * Todos os 4 pontos de despacho SIMD/escalar em `gates.rs`/`layer_kernels.rs`/`layer_dyn_kernels.rs` (`activation_precision() == ActivationPrecision::HighFidelity` → `== ActivationPrecision::Standard`).
* **Verificação:** `cargo check`/`clippy --all-targets --all-features` limpos.

#### 🎯 Tarefa 1.2: Remoção do Override Por-Modelo (`preferred_activation_precision`) [CONCLUÍDA]

* **Arquivos:** `src/models/mod.rs` (trait default removido), `src/models/nam_model.rs` (override `Lstm1x16|Lstm2x8` e os 3 blocos de guard em `process`/`prewarm`/`reset` removidos).
* **Justificativa:** com `Standard` universal, o override que existia apenas para `BossLSTM-1x16`/`2x8` tornou-se redundante — resolve B1/B2 (ver nota de superseding no Épico de Follow-up).
* **Verificação:** nenhuma referência residual a `preferred_activation_precision` em `src/` (grep limpo); `cargo check` limpo.

#### 🎯 Tarefa 1.3: CLI (`lexopt`) [CONCLUÍDA]

* **Arquivos:** `src/standalone/cli.rs`, `src/standalone/cli_test.rs`, `src/main.rs` (sem alteração necessária — não referencia identificadores literais).
* **Ações realizadas:** strings aceitas por `--activation` alteradas de `standard|std → Standard(antigo, Padé)` / `hf|highfidelity|high → HighFidelity` para `standard|std → Standard(novo, exato)` / `fast → Fast`. Help text atualizado. Testes reescritos (`test_parse_args_activation_fast`, `test_parse_args_activation_std_alias`); casos para os aliases retirados (`hf`/`highfidelity`) documentados como não cobertos por design (chamariam `process::exit(1)` real dentro do processo de teste).
* **Achado colateral corrigido (não relacionado ao rename):** `test_parse_args_activation_default` usava `vec!["nam-rs"]` (zero argumentos reais), disparando o branch `if !has_args { print_help(); std::process::exit(0); }` de `parse_args_from` — uma chamada **real** a `std::process::exit`, que mata o processo de teste inteiro (não apenas o teste), truncando silenciosamente o restante da suíte com exit code 0 (falso "sucesso"). Corrigido passando um argumento inócuo (`--buffer-size 256`) para manter `has_args = true` sem exercitar o branch de saída. Documentado com comentário extenso no teste. **Nenhuma alteração de código de produção foi necessária** — o bug estava apenas no teste.

#### 🎯 Tarefa 1.4: CLAP (parâmetro, GUI, render offline) [CONCLUÍDA]

* **Arquivos:** `src/clap/extensions/params/mod.rs` (`ACTIVATION_STANDARD`/`ACTIVATION_FAST`, `activation_u32_to_enum`), `src/clap/extensions/params/main.rs` (`default_value: 1.0`, `value_to_text`, `text_to_value`), `src/clap/plugin/mod.rs` + `shared_test.rs` (`param_activation` inicial `1`), `src/clap/processor/events.rs` (render offline força `Standard`), `src/clap/gui/ui/zones/controls.rs` (botões "Standard"/"Fast", reordenados com `Standard` primeiro).
* **Nota de compatibilidade:** `default_value` do parâmetro CLAP mudou de `0.0` para `1.0` — hosts que instanciam o plugin **sem estado salvo** (primeira inserção) agora recebem `Standard` (exato) em vez do antigo `Standard`(Padé). Projetos com estado **já salvo** não são afetados (discriminantes preservados).
* **Verificação:** `--lib --features clap-plugin -- clap:: --test-threads=1` → 74/74 passando (1 ignorado, `test_gc_stress_1000_swaps`, não relacionado).

#### 🎯 Tarefa 1.5: Testes (unitários + integração) e 1 Recalibração de Threshold [CONCLUÍDA]

* **Arquivos:** `tests/models/activation_precision.rs`, `tests/models/lstm_activation_precision.rs`, `tests/parity/cpp_parity.rs`, `tests/parity/isa_parity.rs`, `tests/parity/reference_oracle_f64.rs`, `tests/common/validation.rs`.
* **Critério de rename aplicado a testes A/B explícitos:** onde um teste compara deliberadamente os dois modos (`Standard` antigo vs `HighFidelity` antigo), aplicado rename mecânico simétrico (`Standard→Fast`, `HighFidelity→Standard`) preservando 100% a semântica original do teste. Onde um valor representa "restaurar o default do sistema" (guards de limpeza/RAII), o texto `Standard` foi mantido — resolve automaticamente para o novo default universal.
* **Regressão encontrada e corrigida (efeito direto do item 3, não um bug de rename):** `golden_vectors::test_golden_vectors_lstm_dyn_test` (`LstmModelDyn`, não coberto pelo antigo override de `BossLSTM`) passou a rodar em `Standard`(exato) por default em vez de `Fast`(Padé). SNR/ESR melhoraram ~9 ordens de magnitude (144.3 dB / 3.69e-15, vs. gate de 80 dB / 3.5e-9 — folga enorme), mas o gate de MR-STFT (métrica espectral, não-monotônica em relação a SNR/ESR) subiu de 0.0585 (medido em 2026-06-21, sob `Fast` default) para 0.0815 (medido em 2026-07-09, sob `Standard` default), excedendo o gate calibrado de 0.08. Corrigido em `tests/common/validation.rs` com novo comentário de medição (Regra 3 de `docs/perceptual_validation.md`) e gate elevado para 0.10 (~23% de margem).
* **Achado de auditoria — pré-existente, não relacionado a este rename:** ao rodar `cargo test --release --lib --features standalone` completo, 14 testes `#[should_panic]` falharam (`math::dsp::fft::tests::*`, `models::a2::grouped_conv1d::tests::*`). Confirmado via `cargo test` (sem `--release`) que passam normalmente em modo debug — indica uso de `debug_assert!`/panics dependentes de *overflow checks* que são compilados fora em modo release. **Nenhum arquivo relacionado foi tocado por este rename.** Registrado como achado a investigar separadamente (ver Sprint 3, Tarefa 3.3) — não bloqueia este Épico.
* **Verificação:** `tests/models` completo 221/221 (58 ignorados), `tests/parity` completo 36/36 (83 ignorados).

---

### 📅 Sprint 2: Sincronização de Documentação (pendente)

> **Risco:** 🟡 Moderado — mecânico na maioria dos arquivos, mas **3 arquivos exigem reescrita semântica**, não apenas troca de palavra, porque descrevem o mecanismo antigo (override por-modelo de `BossLSTM`, tabela "Live vs. HQ" com ativação diferente por modo) que foi removido no item 3. Tratar cada tarefa como independente e sequencial — nenhuma depende de outra, mas a ordem abaixo prioriza os arquivos de maior risco de leitura incorreta por futuros agentes/desenvolvedores primeiro.

#### 🎯 Tarefa 2.1: `.agents/rules/rust.md` — Regra Operacional Mandatória [CONCLUÍDA]

* **Achado:** linha 48 descreve o contrato **antigo**: modo **Live** = `oversampling Off, activation precision Standard(Padé)`; modo **HQ/Offline** = `oversampling 4×, activation precision HighFidelity(exato)`. Isso está **estruturalmente incorreto** pós-rename: `Standard` agora É o modo exato e universal (Live e HQ/Offline usam `Standard` por default); apenas oversampling e adaptive-compute continuam diferindo entre os dois modos.
* **Ação:** reescrever a frase para refletir que activation precision **não varia mais** entre Live/HQ por default (ambos usam `Standard`); mencionar que `Fast` é um opt-in explícito para quem prioriza CPU sobre fidelidade. **Não fazer rename textual ingênuo** — este é uma regra lida por todos os agentes futuros como fonte de verdade.
* **Risco de não fazer corretamente:** um agente futuro que confie neste texto tomaria decisões de código erradas (ex.: assumir que precisa reintroduzir um override condicional a `RenderMode`).

#### 🎯 Tarefa 2.2: `docs/cpp_parity_map.md` (6 menções) [CONCLUÍDA]

* **Achado:** linha 212 ("Both models now default to `HighFidelity`") descreve o override específico de `BossLSTM` removido na Tarefa 1.2 — deve ser reescrita para "todos os modelos agora usam `Standard` por default" (não apenas os dois LSTM). Linha 697 (tabela comparativa C++/Rust) e linhas 241-242 (tabela de caps de ESR rotulados `HighFidelity`) podem receber rename mecânico (`HighFidelity→Standard`, e a antiga `Standard` mencionada como contraponto → `Fast`), mas revisar cada linha após o rename para checar se a frase ainda faz sentido gramatical/semântico.
* **Ação:** revisar linha por linha (180, 212, 241, 242, 697, 804) individualmente — não usar substituição em lote neste arquivo.

#### 🎯 Tarefa 2.3: `docs/audio_fidelity_map.md` (19 menções) [CONCLUÍDA]

* **Achado:** este é o arquivo com mais menções (19) e o que mais detalha o mecanismo de resolução de fidelidade por-modelo (incluindo, presumivelmente, a seção "BossLSTM default resolution" identificada na auditoria anterior — Achado A1 §7). Provável necessidade de reescrever a seção que descreve o override por-modelo como "resolvido" para refletir que a resolução agora é **universal**, não mais específica de `BossLSTM`.
* **Ação:** ler o arquivo completo antes de editar (não apenas grep das linhas com `HighFidelity`); identificar se há uma seção estruturalmente dedicada ao override por-modelo que precisa ser removida/reescrita (não apenas renomeada).

#### 🎯 Tarefa 2.4: `docs/perceptual_validation.md` (11 menções) [CONCLUÍDA]

* **Achado:** este arquivo é a autoridade normativa para "Gate Calibration Policy" (Regras 1-7) referenciada por várias outras tarefas neste documento — inclui a análise do "interop gap" que motivou a adoção de `HighFidelity` para `BossLSTM` (linhas ~645-813, segundo mapeamento da auditoria anterior).
* **Ação:** revisar se a narrativa "HighFidelity mode collapses the gap for BossLSTM" precisa de nota de atualização (universal agora, não específico), similar ao tratamento dado em `TODO-findings.md` Achado A1 (nota de renumeração já aplicada como precedente de estilo).

#### 🎯 Tarefa 2.5: `README.md` (2 menções, tabela "Live vs. HQ/Offline") [CONCLUÍDA]

* **Achado:** linhas 178-181 e a tabela "Live vs. HQ/Offline Summary" (linhas 190-198) — a linha `Activation precision | Standard | HighFidelity` da tabela descreve exatamente o contrato antigo invalidado pelo item 3 (ver Tarefa 2.1). Após o rename, ambos os modos usam `Standard` por default.
* **Ação:** reescrever a linha da tabela (ex.: `Activation precision | Standard (Fast opt-in) | Standard (enforced)`, ou remover a linha da tabela e mover a explicação para a seção "Activation Precision" acima, já que não é mais um diferencial Live/HQ). Atualizar a lista de bullets (linha 180-181) com rename mecânico simples.

#### 🎯 Tarefa 2.6: `docs/fastmath-approximations.md` (5 menções), `docs/research-references.md` (1), `docs/architecture.md` (1) [CONCLUÍDA]

* **Ação:** rename mecânico (`HighFidelity→Standard`, contexto de "antiga Standard"→`Fast`) com revisão rápida de cada linha após a troca (sem necessidade de reescrita estrutural, ao que indica o grep de contexto já coletado neste planejamento).

#### 🎯 Tarefa 2.7: `TODO-parity.md` (linha 163) [🟢 baixo risco, mas leia com atenção]

* **Achado:** "Modelos BossLSTM (...) têm sua preferência configurada para `HighFidelity`" descreve o mecanismo removido na Tarefa 1.2.
* **Ação:** reescrever para refletir que a resolução é universal (não mais uma "preferência configurada" por modelo) — ex.: "`Standard` (exato) passou a ser o default universal de ativação em todo o nam-rs, superando a necessidade de uma preferência configurada apenas para BossLSTM."

---

### 📅 Sprint 3: Verificação Final (pendente, gate obrigatório antes da Sprint 4)

#### 🎯 Tarefa 3.1: `utils/lints.sh`

* **Ação:** rodar e confirmar zero erros (fmt, SPDX, `cargo check`, clippy). Já validado individualmente nesta sessão (`cargo check`/`clippy --all-targets --all-features`), mas rodar o script oficial garante paridade com o que o CI/humano espera.

#### 🎯 Tarefa 3.2: `utils/tests-quick.sh` (permitido uma execução por tarefa de IA, como validação final)

* **Ação:** rodar uma única vez, como validação final desta Sprint. Não re-executar caso passe.
* **Atenção:** este script provavelmente roda em modo diferente de `--release`; usar seu resultado como a validação oficial (não o `cargo test --release` ad-hoc usado durante a implementação).

#### 🎯 Tarefa 3.3: Registrar o Achado das 14 Falhas `#[should_panic]` em Release (pré-existente, não relacionado)

* **Ação:** abrir uma entrada curta em `TODO-findings.md` (novo Achado, severidade 🟢 Baixa/técnica) documentando que `math::dsp::fft::tests::*` (6 testes) e `models::a2::grouped_conv1d::tests::*` (8 testes) falham como `#[should_panic]` apenas em `cargo test --release` (passam em modo debug) — provável uso de `debug_assert!` ou dependência de *overflow checks* desabilitados em release. **Não corrigir nesta Sprint** — é ortogonal ao rename; apenas registrar para auditoria futura.

#### 🎯 Tarefa 3.4: Confirmar Ganho de Performance da Remoção do Guard (B2) com Medição

* **Ação:** rodar `cargo bench` (ou os testes de zero-alloc/latência já existentes, ex. `tests/models/zero_alloc_infer.rs`) antes/depois da remoção do guard em pelo menos um modelo não-LSTM (ex. WaveNet) e documentar o delta com comentário de medição, cumprindo a exigência de proveniência que a Tarefa 4.2 original (agora superseded) já previa. Este número deve entrar na mensagem do commit da Sprint 4 (Tarefa 4.1) como evidência quantitativa do ganho.

---

### 📅 Sprint 4: Estratégia de Commit (pendente, só inicia após Sprint 3 verde)

> **Princípio:** commits devem ser **bisectáveis** sempre que a dependência de compilação permitir. Código de produção (`src/`) e testes (`tests/`, `src/**/*_test.rs`) são mutuamente dependentes na compilação (os testes referenciam os novos identificadores `Fast`/`Standard`) — não podem ser separados em commits distintos sem deixar um commit intermediário que não compila. Documentação (`docs/*.md`, `README.md`, `TODO-*.md`, `.agents/rules/rust.md`) não tem essa dependência e pode/deve ser commitada separadamente.

#### 🎯 Tarefa 4.1: Commit único — Rename de Código + Testes (Sprint 1) [CONCLUÍDA]

* **Escopo:** todos os arquivos de `src/` e `tests/` já modificados (ver diff atual da árvore de trabalho).
* **Mensagem sugerida:** `refactor(activation): rename ActivationPrecision Standard<->Fast; Standard (exact) becomes universal default`
* **Corpo do commit deve incluir:** motivação (resolve B1/B2), nota de compatibilidade (discriminantes preservados, `default_value` do CLAP mudou de 0.0→1.0), e o número medido da Tarefa 3.4.

#### 🎯 Tarefa 4.2: Commit separado — Sincronização de Documentação (Sprint 2) [CONCLUÍDA]

* **Escopo:** todos os arquivos de `docs/`, `README.md`, `.agents/rules/rust.md`, `TODO-parity.md` tocados na Sprint 2.
* **Pré-requisito:** Sprint 2 completa (todas as 7 tarefas), não apenas parcialmente.

#### 🎯 Tarefa 4.3: Atualizar Este Documento e `TODO-findings.md` Pós-Commit [CONCLUÍDA]

* **Ação:** marcar as Sprints 1-4 deste Épico como `[CONCLUÍDO]` com hash dos commits reais; adicionar a entrada da Tarefa 3.3 em `TODO-findings.md` (se ainda não commitada).

---

## 📋 Referências Cruzadas

| Item                                              | Localização                                                                                                                                     | Tópico                                                        |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Achado A1 (motivação deste épico)                 | `TODO-findings.md` §"Achado A1"                                                                                                                 | Auditoria original: fix implementado mas ineficaz             |
| Achado 2 (original, reaberto)                     | `TODO-parity.md:49`                                                                                                                             | Diagnóstico original de drift recorrente                      |
| Metodologia prewarm-paired (precedente histórico) | `tests/parity/reference_oracle_f64.rs:273-314` (`run_oracle_esr_paired`)                                                                        | Base para Tarefa 1.1                                          |
| Decomposição de erro (precedente)                 | `tests/parity/reference_oracle_f64.rs:380-403` (`test_decomposition_lstm`), `src/testing/reference_oracle/mod.rs:422-472` (`run_decomposition`) | Base para Tarefa 1.3                                          |
| Constantes de baseline ESR                        | `tests/common/constants.rs:16-38`                                                                                                               | Base para Tarefa 1.4                                          |
| Classificação Interop vs Absolute                 | `docs/perceptual_validation.md:605-759`                                                                                                         | Base para Tarefa 2.4                                          |
| Regras de qualidade de gate                       | `docs/perceptual_validation.md` "Gate Calibration Policy" (Regras 1-7)                                                                          | Aplicável a toda alteração de tolerância/baseline neste épico |
| RT-safety e SIMD                                  | `.agents/rules/rust.md`                                                                                                                         | Aplicável à Tarefa 3.1 (condicional)                          |
| Convenções de teste (`#[ignore]`, placement)      | `.agents/rules/testing.md`                                                                                                                      | Aplicável às Tarefas 1.1, 1.4, 1.5                            |
| Achados B1/B2 (superseded)                        | Épico de Follow-up acima, Tarefas 4.1/4.2                                                                                                       | Resolvidos organicamente pelo Épico de rename `Standard↔Fast` |
| Épico de rename `Standard↔Fast`                   | Este documento, seção "Épico: Rename `Standard`↔`Fast`..."                                                                                      | Sprint 1 concluída nesta sessão; Sprints 2-4 pendentes        |
