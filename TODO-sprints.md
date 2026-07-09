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

#### 🎯 Tarefa 2.2: Testar Hipótese de `ActivationPrecision` Divergente (Contingência)

* **Executar apenas se a Tarefa 2.1 não explicar o gap.**
* **Ações:**
  * Usando `set_activation_precision`/`activation_precision` (`src/math/activations/mod.rs:97-116`), fixar explicitamente `ActivationPrecision::HighFidelity` antes de construir o modelo de produção no diagnóstico pareado (Tarefa 1.1), e comparar o `esr_tail` resultante com o obtido em modo `Standard` (default).
  * Justificativa: `PrecisionConfig::default()` do oráculo usa `ActivationMode::Exact` (tanh/sigmoid `f64` exatos). Se a produção estiver em `ActivationPrecision::Standard` (Padé/minimax, erro conhecido ~7.6e-4 por decomposição), isso já está contabilizado no piso de 3.4e-3 documentado — mas vale confirmar que não há divergência adicional não contabilizada (ex.: dispatch incorreto de precisão em algum caminho AVX-512 específico dos modelos `BossLSTM-*`, diferente do `lstm.nam` já testado).
  * Restaurar `ActivationPrecision::Standard` ao final do teste (é `static` global — ver levantamento técnico; não há guard RAII implementado, restaurar manualmente é obrigatório para não contaminar testes subsequentes na mesma execução de `cargo test`).

#### 🎯 Tarefa 2.3: Testar Hipótese de Saturação em Regimes de Alta Amplitude (Contingência)

* **Executar apenas se as Tarefas 2.1 e 2.2 não explicarem o gap.**
* **Ações:**
  * Usar a saída blockwise da Tarefa 1.2 para identificar se os blocos de **maior** ESR residual (após pareamento) coincidem com as seções de **maior amplitude** do sinal de stress v2 (Seção 1: Low-E bend 0.0-1.0s; Seção 2: power chord 1.0-2.0s; Seção 5: bass amp 3.5-4.5s — ver `src/testing/stress.rs:82-296`), o que apontaria para saturação/clipping das aproximações Padé/minimax em regimes de gate próximos aos extremos `[-1,1]`, em vez de drift recorrente.
  * Se confirmado, medir a magnitude do erro de aproximação `tanh`/`sigmoid` especificamente nos extremos de entrada (`|x| > 3`, região de saturação) comparando `scalar_pade_tanh`/`scalar_minimax_sigmoid` (`src/math/activations/`) contra `f64::tanh`/sigmoid exato para uma faixa de valores de entrada amostrados dos gates reais durante o processamento do trecho de maior amplitude.

#### 🎯 Tarefa 2.4: Consolidar Conclusão e Reclassificar o "Interop Gap" (2.61e-2 vs NAMCore)

* **Ações:**
  * Revisar `docs/perceptual_validation.md:690-710` ("Classification" — Interop vs Absolute Correction) à luz dos resultados das Tarefas 2.1-2.3: confirmar se o valor `2.61e-2` ali documentado como "Interop (nam-rs vs NAMCore)" é de fato medido contra o binário C++ real (via `tests/parity/cpp_parity.rs`, testes `live_cross_validation_v2_lstm_1x16`, atualmente `#[ignore]`d e exigindo binários golden — ver levantamento técnico) ou se foi, na verdade, medido de forma equivalente ao `t33` legado (produção pareada com prewarm-de-zeros vs oráculo `f64` sem pareamento) e **mal rotulado** como "vs NAMCore" no texto da documentação.
  * Se for o segundo caso: corrigir a rotulação em `docs/perceptual_validation.md` (Tarefa 3.4) para não conflar "divergência de estado inicial vs oráculo f64" com "divergência real de interoperabilidade vs NAMCore C++".
  * Se for o primeiro caso (medição real contra NAMCore C++): decidir se o gap de interoperabilidade de `2.61e-2` — real e distinto do piso absoluto de `3.4e-3` — merece um novo achado de investigação separado (fora do escopo deste épico, que trata apenas do piso absoluto vs `f64`), documentando essa decisão explicitamente.

---

### 📅 Sprint 3: Consolidação, Correção Direcionada (se aplicável) e Documentação

#### 🎯 Tarefa 3.1: Implementar Correção Direcionada (Condicional)

* **Executar apenas se a Sprint 2 revelar uma causa-raiz adicional, corrigível e não coberta pela compensação de Kahan já implementada.**
* **Ações:**
  * Especificar e implementar a correção seguindo as diretrizes de `.agents/rules/rust.md` (RT-safety, zero alocação no hot-path, SIMD baseline `x86-64-v3`, `AlignedVec` para buffers alinhados).
  * Validar com o mesmo par de testes pareado/legado da Sprint 1 (Tarefas 1.1, 1.5) antes/depois, documentando os números exatos — não repetir o erro de processo do Achado 2 original (fechar por checklist sem comparar resultado).
  * Rodar `cargo bench` focado em LSTM para confirmar que a correção não introduz regressão de latência fora do orçamento de CPU (< 1% do budget RT, mesmo critério do épico anterior).

#### 🎯 Tarefa 3.2: Atualizar `TODO-parity.md` (Achado 2) com a Conclusão Final

* **Arquivo:** [`TODO-parity.md`](file:///home/fabio/nam-rs/TODO-parity.md) (linha 49)
* **Ações:**
  * Substituir a nota de auditoria atual (`[REABERTO — ver TODO-findings.md Achado A1]`) por uma conclusão definitiva:
    * Se a Tarefa 2.1 confirmou mismatch de estado inicial como causa dominante e nenhuma correção adicional foi necessária: `[FECHADO — causa-raiz reclassificada, ver TODO-findings.md Achado A1 §Resolução]`.
    * Se uma correção adicional foi implementada (Tarefa 3.1): `[FECHADO — causa-raiz corrigida em <commit>, ver TODO-findings.md Achado A1 §Resolução]`.

#### 🎯 Tarefa 3.3: Atualizar `TODO-findings.md` (Achado A1) com Seção "Resolução"

* **Arquivo:** [`TODO-findings.md`](file:///home/fabio/nam-rs/TODO-findings.md)
* **Ações:**
  * Adicionar uma subseção `### 6. Resolução` ao Achado A1 com: resultado numérico exato de cada tarefa da Sprint 2, hipótese confirmada/refutada, e (se aplicável) referência ao commit da correção da Tarefa 3.1.
  * Atualizar o campo `**Status:**` do Achado A1 de `🔴 Reaberto` para `🟢 Resolvido` (ou `🟡 Parcialmente Resolvido` se o "Interop gap" da Tarefa 2.4 for desmembrado em um novo achado separado).

#### 🎯 Tarefa 3.4: Atualizar Documentação de Fidelidade

* **Arquivos:**
  * [`docs/audio_fidelity_map.md`](file:///home/fabio/nam-rs/docs/audio_fidelity_map.md) §3 "LSTM Recurrent State Drift" (linhas 113-177)
  * [`docs/cpp_parity_map.md`](file:///home/fabio/nam-rs/docs/cpp_parity_map.md) §2.7 "Measured interop drift"
  * [`docs/perceptual_validation.md`](file:///home/fabio/nam-rs/docs/perceptual_validation.md) seção "LSTM Recurrent State Quantization Drift" (linhas 643-759)
  * [`docs/fastmath-approximations.md`](file:///home/fabio/nam-rs/docs/fastmath-approximations.md) (se a Tarefa 2.2/2.3 revelar necessidade de nota sobre ativações)
* **Ações:**
  * Atualizar cada seção com os números reais medidos nas Sprints 1-2 para `BossLSTM-1x16`/`2x8` especificamente (não mais herdados da família `lstm.nam`).
  * **Resolver a referência pendente a `docs/lstm_recurrent_drift.md`** (citada por `utils/quality-dashboard.sh:1137` e pelos dashboards gerados, mas **inexistente** — confirmado nesta investigação; `docs/research-references.md:138` inclusive já a menciona como "former... removed after f16c elimination"). Duas opções, decidir com base no volume de conteúdo final:
    1. **Recomendado:** Não recriar um arquivo dedicado — a seção "LSTM Recurrent State Quantization Drift" de `docs/perceptual_validation.md` já concentra "Mechanism", "Empirical evidence", "Hypotheses ruled out" e "Classification", exatamente o conteúdo que um arquivo dedicado teria. Corrigir a referência em `utils/quality-dashboard.sh:1137` para apontar para `docs/perceptual_validation.md#lstm-recurrent-state-quantization-drift` em vez do arquivo inexistente.
    2. **Alternativa:** Se o volume de conteúdo pós-Sprint-2 justificar (ex.: análise blockwise completa, tabelas de decomposição por modelo), extrair essa seção para um `docs/lstm_recurrent_drift.md` dedicado, com `docs/perceptual_validation.md` mantendo apenas um resumo + link.
  * Resolver também a referência pendente `PM-08` em `docs/fastmath-approximations.md:370` (`TODO-findings.md` PM-08 — inexistente): ou remover a linha da tabela de cross-reference, ou adicionar a entrada correspondente em `TODO-findings.md` se fizer sentido no contexto deste achado.
  * Corrigir a nota de "F-Q1" gerada por `utils/quality-dashboard.sh` (linha 117 dos dashboards: *"Ver docs/lstm_recurrent_drift.md e TODO-findings.md F-Q1"*) — `F-Q1` também não existe como achado; ou criar essa referência cruzada corretamente apontando para o Achado A1, ou remover o texto do script gerador.

#### 🎯 Tarefa 3.5: Preencher o Piso Real de `f64` para `BossLSTM-1x16`/`2x8` no Dashboard

* **Arquivo:** [`utils/quality-dashboard.sh`](file:///home/fabio/nam-rs/utils/quality-dashboard.sh) (`parse_oracle_f64()`, linha 397; `render_quick_summary()`, linha 738; `render_fidelity_details()`, linha 831)
* **Ações:**
  * Consumir a saída dos novos testes de decomposição pareada da Tarefa 1.3 (`test_decomposition_boss_lstm_1x16`, `test_decomposition_boss_lstm_2x8`) no parser `parse_oracle_f64()`, populando as entradas específicas por modelo em vez de cair no fallback de família (`_lookup_esr_f64()`, linha 124).
  * Remover a nota `(~fam.)` e a ressalva explicativa (linhas 24-29 do dashboard atual) para `BossLSTM-1x16`/`2x8` assim que o valor real e específico estiver disponível — mantendo a nota apenas para os demais modelos que ainda não tiverem decomposição pareada dedicada.
  * Regerar `quality-dashboard-antes.txt`/`depois.txt` (ou equivalentes) com o piso real preenchido, para validação final do épico.

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
