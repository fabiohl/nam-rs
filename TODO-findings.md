<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Qualidade (Revisor Auditor)

> Este documento registra achados gerados pela skill `revisor-auditor` (e, quando aplicável, `pesquisador-inovador`).
> Cada achado é auto-contido, com diagnóstico, evidência reproduzível e proposta de solução.
>
> Vide chat "Auditoria LSTM Recurrent State Drift".

---

## Achado A1 — Auditoria do "Achado 2: LSTM Recurrent State Drift" (`TODO-parity.md:49`) — Resolvido via Correção Direcionada (Opção A)

**Papel:** Compliance and Parity Auditor + Resilience and Robustness Specialist

**Severidade:** 🟢 Resolvido (a causa-raiz foi isolada e corrigida via ativação automática de `ActivationPrecision::HighFidelity` para a família `BossLSTM` de maneira thread-safe)

**Status:** 🟢 Resolvido (com pendências de auditoria — ver §7) — A suíte de testes e o dashboard de qualidade comprovam que o ESR de BossLSTM caiu de `~5e-2` para `~1.5e-11` (paridade matemática absoluta com o NAMCore C++). A latência absoluta aumentou de forma real (~6.0-6.2µs → 7.9µs para 1x16; ~5.3-5.8µs → 8.0µs para 2x8, ~+30-40% relativo), permanecendo, porém, dentro do orçamento de tempo real (0.6% de 1.33ms) — "sem regressão real" refere-se ao orçamento RT, não à ausência de aumento absoluto de latência.

> **Nota de Numeração (2026-07-09):** A Tarefa 3.2 (Sprint 3) renumerou os achados de `TODO-parity.md` durante a atualização — o achado abaixo, então "Achado 2", passou a ser **"Achado 3"** em `TODO-parity.md` (o antigo "Achado 3", `wavenet_a2_max.nam`, passou a ser "Achado 2"). As referências a "Achado 2" no corpo deste documento abaixo são citações históricas do estado do `TODO-parity.md` **no momento em que foram escritas** e permanecem corretas como registro histórico — mas não devem ser usadas para navegação após a renumeração. Ver §7 para os achados de auditoria adicionais desta revisão.

### 1. Resumo Executivo

A implementação da compensação de Kahan no *cell state* da LSTM (`TODO-sprints.md`, Sprints 1–4) foi executada com **excelência de engenharia**: as assinaturas, os kernels AVX2/AVX-512, os fallbacks escalares e os testes de paridade SIMD-vs-escalar estão corretos, consistentes entre si, e não introduzem regressões de performance fora do orçamento de CPU (< 1% do budget RT).

Entretanto, a auditoria — via comparação controlada *antes/depois* do commit do épico (`c557825`..`5d0769e` vs `d7de939`) executando o próprio teste de diagnóstico que a Tarefa 4.3 instrui a rodar (`t33_diagnostic_recurrent_drift_lstm_1x16 --ignored`) — comprova que **o resultado numérico é idêntico até a 5ª casa significativa antes e depois do fix**:

| Amostras              | ESR antes (`d7de939`)      | ESR depois (`5d0769e`)     | Δ relativo |
| --------------------- | -------------------------- | -------------------------- | ---------- |
| 512                   | 3.398263e-1                | 3.397961e-1                | ~9e-5      |
| 4096                  | 6.133528e-2                | 6.131570e-2                | ~3e-4      |
| 65536                 | 3.211271e-2                | 3.210716e-2                | ~2e-4      |
| **240000 (full, 5s)** | **2.611707e-2 (-15.8 dB)** | **2.611684e-2 (-15.8 dB)** | **~9e-6**  |

Isso confirma, com evidência reprodutível, que a compensação de arredondamento recorrente **não é a causa dominante** do gap de fidelidade observado em `BossLSTM-1x16` — ao contrário do que o diagnóstico original em `TODO-parity.md` (Achado 2) presumia. O `quality-dashboard-antes.txt`/`depois.txt` já continha esse mesmo sinal (ESR `vs NAMcore` idêntico — `1.04e-2` / `0.00e0` — em ambas as medições), mas essa evidência não foi interpretada corretamente durante o fechamento do Sprint 4.

### 2. Evidência Detalhada

#### 2.1. Comparação A/B controlada (reproduzida nesta auditoria)

Metodologia: `git worktree add` no commit imediatamente anterior ao épico (`d7de939`, antes de `c557825`), build `--release`, e execução de `cargo test --release --test parity t33_diagnostic_recurrent_drift_lstm_1x16 -- --ignored --nocapture` em ambos os estados (antes/depois), mesma máquina, mesmo binário de teste (sinal determinístico `generate_stress_signal_v2_default`, 5s @ 48 kHz, oráculo `f64`, prewarm de 24k amostras).

Resultado: as 10 janelas de medição (512 → 240000 amostras) são **idênticas dentro do ruído de ponto flutuante** (diferenças ~1e-4 a 1e-6 relativas) entre o commit pré-fix e o commit pós-fix. Se a hipótese do Achado 2 estivesse correta — que o *round-off* recorrente do `f32` na acumulação do *cell state* é o principal contribuinte do ESR observado — a compensação de Kahan deveria produzir uma queda de vários dB no ESR de longa duração. Isso **não ocorre**.

#### 2.2. `quality-dashboard-antes.txt` vs `quality-dashboard-depois.txt`

* `BossLSTM-1x16 @48000 Live`: ESR (vs NAMcore) **1.04e-2 → 1.04e-2** (sem mudança); SNR 19.8 dB → 19.8 dB (sem mudança); apenas o MR-STFT oscila no 4º dígito (0.0966 → 0.0970), consistente com ruído de medição/jitter de scheduling, não com efeito do fix.
* `BossLSTM-2x8 @48000 Live`: ESR **0.00e0 → 0.00e0** (idêntico); MR-STFT 0.0648 → 0.0647 (ruído).
* Todos os demais modelos WaveNet/A2/ConvNet — que **não usam LSTM e não foram tocados pelo fix** — também exibem pequenas variações de latência (`WaveNet Standard CH16`: 42.2 µs → 42.6 µs; `ConvNet`: 3.6 µs → 3.7 µs). Isso é a assinatura de variância ambiental normal entre execuções (turbo/thermal/scheduler) do `utils/quality-dashboard.sh`, e **não deve ser atribuído ao código do LSTM**. É importante registrar esse controle para não confundir ruído de medição com regressão real em auditorias futuras.

#### 2.3. O teste de diagnóstico (`t33_diagnostic_recurrent_drift_lstm_1x16`) não possui nenhuma asserção

Inspecionado em `tests/parity/reference_oracle_f64.rs:790-868`: a função é `#[ignore]`d, imprime uma tabela de ESR por tamanho de janela e finaliza **sem nenhum `assert!`**. É puramente observacional. Isso significa que:

1. `cargo test -- --ignored` sempre reporta `ok` independentemente do resultado numérico — não há *regression gate* automatizado para o próprio fenômeno que o Achado 2 se propôs a resolver.
2. A Tarefa 4.2/4.3 do `TODO-sprints.md` ("Validar que todas as suítes rápidas... passem sem falhas" / "Rodar `utils/quality-dashboard.sh` e compara com a baseline") foi marcada `[DONE]` sem que o número relevante (ESR do `t33_diagnostic_recurrent_drift_lstm_1x16`) fosse de fato comparado com uma baseline pré-fix. O `quality-dashboard-antes.txt`/`depois.txt` inclusive **contém o sinal correto** (ESR inalterado), mas o encerramento do sprint não o interpretou como "fix inconclusivo".

#### 2.4. O padrão não-monótono do ESR é incompatível com "drift recorrente" clássico

A tabela de ESR por janela cumulativa (512 → 240000 amostras) **não cresce monotonicamente** com a duração, como seria esperado de um erro `O(N·ε)` que se acumula ao longo do stream:

```text
512      → 3.40e-1   (-4.7 dB)
4096     → 6.13e-2   (-12.1 dB)
16384    → 2.52e-2   (-16.0 dB)   ← mínimo local
32768    → 3.60e-2   (-14.4 dB)   ← sobe de novo
120000   → 5.16e-2   (-12.9 dB)   ← sobe mais
240000   → 2.61e-2   (-15.8 dB)   ← cai de novo
```

Esse comportamento é a assinatura característica de um **ESR cumulativo-desde-o-início dominado por trechos de baixa amplitude** do sinal de stress (`generate_stress_signal_v2`, que contém segmentos de silêncio/decaimento entre 2.0–2.5s "palm-mute" e 4.5–5.0s "ringing decay exponencial"): pequenos erros absolutos em passagens quase silenciosas inflam o ESR relativo local, e a métrica cumulativa mistura essas janelas de forma não linear conforme a janela cresce. **Isso não é o que a compensação de Kahan resolve** (que ataca apenas o erro de arredondamento por passo, não a relação sinal/erro em passagens de baixa amplitude).

Esse é também um indício de que **a metodologia de medição do próprio diagnóstico (Achado 2) está confundida**: ESR cumulativo-desde-t=0 não isola "drift ao longo do tempo" de "erro instantâneo em passagens quietas". Uma medição por *janela deslizante* (não cumulativa) ou normalizada por RMS local seria necessária para de fato caracterizar drift recorrente vs. erro instantâneo.

#### 2.5. O "gap" real de ~13 dB permanece sem explicação

A seção "🔍 F64 ORACLE — Decomposição de Fontes de Erro" do próprio dashboard já decompunha, para a família LSTM-H3 (fixture `lstm.nam`), a soma de todas as fontes de erro conhecidas:

```text
ESR(f32 vs f64 oracle):        3.02e-3  (-25.2 dB)
ΔESR combinado (F16C+Padé+F32): 5.37e-4  (-32.7 dB)
```

Porém o ESR de produção medido para `BossLSTM-1x16` (via `t33_diagnostic...`, medição pareada com prewarm, sem o problema de cold-start mencionado nas próprias notas do dashboard) é **2.61e-2 (-15.8 dB)** — quase **10× maior** (≈13 dB) que a soma conhecida de fontes de erro decompostas (F16C, bf16, Padé, acumulação f32). A compensação de Kahan ataca exatamente o componente "acumulação f32" dessa decomposição — que já era o **menor** dos termos conhecidos (-121 a -137 dB nos exemplos de `ConvNet-test`/`WaveNet-official`) — o que explica matematicamente por que o fix não move a agulha: ele está corrigindo um termo que já era irrelevante frente ao gap real. A causa dominante dos ~13 dB não identificados permanece **desconhecida** e não foi isolada por este épico.

### 3. Causa-Raiz da Falha de Processo

1. **Diagnóstico original (Achado 2) não teve confirmação quantitativa prévia**: a hipótese de "drift recorrente por acumulação f32" foi proposta com base em observação qualitativa ("bit-exactness válido apenas para excitações curtas"), mas nenhuma decomposição de erro (equivalente à seção "F64 Oracle" já existente no dashboard para outros modelos) foi feita **antes** de iniciar os Sprints 1–4 para confirmar que o termo de acumulação era, de fato, o dominante.
2. **Teste de validação (`t33_diagnostic_recurrent_drift_lstm_1x16`) sem asserção**: permite que a suíte "passe" sem checar o efeito do fix.
3. **Encerramento de sprint por checklist, não por resultado**: `TODO-sprints.md` marca Tarefas 4.1–4.3 como `[DONE]` com base em "testes passam" e "latência dentro do budget", sem comparar o ESR do cenário-alvo antes/depois — o único critério de aceite que de fato validaria o objetivo do Épico ("Garantir paridade matemática exata... eliminando o drift recorrente").
4. **Tolerância de teste de paridade excessivamente larga**: em `src/models/lstm/lstm_test.rs:334`, a asserção de paridade SIMD-vs-escalar para `cell_error` usa `abs() < 1e-2`. Como o próprio `cell_error` é, por definição, um termo de compensação de arredondamento (tipicamente `O(1e-7)` a `O(1e-4)` em magnitude), uma tolerância de `1e-2` é ordens de magnitude maior que o próprio sinal que está sendo testado — o teste passaria mesmo se a implementação SIMD do Kahan estivesse completamente quebrada (ex.: erro sempre zerado).

### 4. Proposta de Solução

**4.1. Não remover a compensação de Kahan já implementada.** O código é correto, barato (< 1% do budget de CPU, latência +0.2 a +0.5 µs), e matematicamente estritamente mais preciso que a acumulação ingênua — apenas não é a causa dominante do gap observado. Mantê-lo é estritamente benéfico (menor ou igual erro em qualquer regime), mas o achado original deve ser reclassificado.

**4.2. Reabrir a investigação de causa-raiz do gap de ~13 dB** com uma metodologia que isole drift-ao-longo-do-tempo de erro-instantâneo-em-passagem-quieta:

* Adicionar ao harness de diagnóstico uma medição de **ESR por janela deslizante não cumulativa** (ex.: blocos de 4096 amostras não sobrepostos, cada um comparado isoladamente ao oráculo, sem herdar erro de janelas anteriores na métrica — o estado do modelo continua acumulando naturalmente, mas a métrica de erro deve ser por-bloco).
* Adicionar decomposição de erro (`ESR(f32 vs f64 oracle)`, `ΔESR f16c`, `ΔESR bf16`, `ΔESR Padé`, `ΔESR acumulação f32`) especificamente para `BossLSTM-1x16` com prewarm pareado e duração de produção (5s), replicando a metodologia já existente em `test_decomposition_lstm` mas usando a *stress signal v2* completa em vez da janela curta de 256 amostras cold-start.
* Investigar hipóteses alternativas com maior probabilidade de explicar 13 dB: (a) discrepância de `ActivationPrecision` (HighFidelity vs Standard) entre produção e oráculo durante o teste; (b) efeito de saturação/clipping da `tanh`/`sigmoid` aproximada em regimes de amplitude alta presentes no sinal de stress (power chords, palhetadas); (c) possível dessincronia de prewarm/estado inicial entre o oráculo f64 e o modelo de produção no teste de longa duração.

**4.3. Corrigir a tolerância do teste de paridade em `lstm_test.rs`** (`< 1e-2` → algo como `< 1e-5` ou relativo à magnitude típica de `cell_error`), para que o teste efetivamente detecte regressões na implementação SIMD do Kahan.

**4.4. Adicionar asserção de regressão ao `t33_diagnostic_recurrent_drift_lstm_1x16`**, comparando o ESR full-duration medido contra um valor de baseline registrado em código/fixture (com tolerância), transformando-o de diagnóstico manual em *regression gate* automatizado. Sem isso, futuras alterações no hot-path da LSTM podem piorar (ou "corrigir sem detectar") esse cenário silenciosamente.

**4.5. Adicionar teste equivalente para `BossLSTM-2x8`** (`t33_diagnostic_recurrent_drift_lstm_2x8`, ausente atualmente) e documentar o resultado real (não o valor de família) na seção de fidelidade do dashboard, conforme a própria nota do dashboard já sinaliza ser necessário.

**4.6. Higienizar o debug output do teste de diagnóstico**: a string fixa `"Baseline A1-Std ESR: 6.23e-3"` impressa em `t33_diagnostic_recurrent_drift_lstm_1x16` não corresponde a nenhum valor medido nesta análise (nem antes, nem depois do fix — ambos ~2.6e-2 no full-duration) e deve ser removida ou substituída por um valor de baseline real e versionado (ver 4.4).

**4.7. Atualizar `TODO-parity.md:49`**: o marcador `[DONE]` do "Achado 2" deve ser trocado por `[REABERTO — ver TODO-findings.md A1]`, e a documentação prometida (`docs/audio_fidelity_map.md`, `docs/fastmath-approximations.md`, `docs/cpp_parity_map.md`, `docs/architecture.md`, além do `docs/lstm_recurrent_drift.md` já referenciado pelo próprio dashboard mas **inexistente no repositório**) não deve ser publicada como "causa identificada e corrigida" até a causa-raiz real ser isolada.

### 5. Impacto se não corrigido

* Falso senso de conclusão sobre uma investigação de fidelidade sonora que permanece **aberta**: o ESR de `BossLSTM-1x16` continua audível (SNR 19.8 dB, ESR 1.04e-2 no teste padrão; -15.8 dB/2.6e-2 no teste de longa duração) exatamente como antes do épico.
* Risco de reintrodução do mesmo problema em outras arquiteturas recorrentes futuras (ex.: GRU, se vier a ser suportada) sob a premissa equivocada de que "compensação de Kahan resolve drift recorrente em LSTM neste projeto", quando a evidência mostra que não foi esse o fator dominante aqui.
* Débito técnico de asserção ausente em teste crítico (`t33_diagnostic...`) e tolerância frouxa em `lstm_test.rs`, reduzindo a capacidade de detecção de regressões futuras no hot-path da LSTM.

### 6. Resolução

A causa-raiz real do gap de fidelidade e do drift recorrente foi identificada e resolvida na Sprint 3.

1. **Hipótese Confirmada (Divergência de Formulação):** A investigação comprovou que a divergência de `2.61e-2` (reproduzível no cenário de 240k amostras do `BossLSTM-1x16`) era decorrente da diferença entre a aproximação FastMath de Rust (Padé[5,4]/minimax-17 no modo `Standard`) e as funções de ativação nativas do NAMCore C++ (que roda com `using_fast_tanh = false`). A compensação de Kahan é correta e matematicamente precisa, mas atua em um piso de ruído irrelevante frente ao gap de modelagem FastMath.
2. **Resolução (Opção A):** Em vez de usar modificações globais e inseguras no dispatcher (que causavam race conditions em testes paralelos), implementou-se um controle de escopo thread-local em `src/math/activations/mod.rs` via `ACTIVE_MODEL_PRECISION` e `ActivationPrecisionGuard`.
3. **Padrão BossLSTM:** Configurado em `StaticModel::preferred_activation_precision` para que a família `BossLSTM` (`1x16` e `2x8`) execute sempre em `HighFidelity` por padrão, colapsando o ESR contra a referência ideal de `~5e-2` para `~1.5e-11` (bit-exact) com latência controlada de `~7.9 us` (~0.6% do budget de tempo real).
4. **Isolamento de Testes:** Os testes matemáticos e de caracterização que especificamente exigem o modo `Standard` foram isolados utilizando os overrides thread-local, eliminando quaisquer efeitos colaterais.

### 7. Auditoria da Implementação da Sprint 3 (2026-07-09)

Re-execução independente dos testes e inspeção linha-a-linha dos diffs de `7b18189`/`c03aa49` confirmou que o resultado central (ESR `1x16`: `2.589060e-2 → 5.046395e-11`; `2x8`: `4.062433e-3 → 4.024454e-12`) é **genuíno e 100% reproduzível**. Porém, a auditoria encontrou 5 problemas não capturados antes de marcar as tarefas como `[CONCLUÍDA]`:

* **B1 — Regressão silenciosa de controle do usuário (🟡 Moderado):** `StaticModel::process/prewarm/reset` (`src/models/nam_model.rs`) aplicam `preferred_activation_precision()` sempre que `thread_local_activation_precision()` é `None` — o que é **sempre o caso** no caminho de chamada do CLI/CLAP, já que estes usam apenas `set_activation_precision()` (o átomo global), nunca `set_thread_local_activation_precision()`. Resultado: um usuário que executa `nam-rs --activation standard` com um preset `BossLSTM-1x16`/`2x8` recebe o log `"Activation precision explicitly set to Standard"`, mas o áudio processado **continua em `HighFidelity`** — a flag explícita do usuário é silenciosamente ignorada para esses dois modelos, sem aviso. Não há hoje um mecanismo para o usuário optar deliberadamente pelo modo mais barato (`Standard`) nesses modelos (ex.: hardware fraco, preferência por menor consumo de CPU mesmo com fidelidade reduzida).
* **B2 — Taxa de performance universal não medida contra baseline (🟡 Moderado):** a guarda de `ActivationPrecisionGuard` é construída/destruída em **todo** `process()`/`prewarm()`/`reset()`, para **todos** os modelos — inclusive os que retornam `None` em `preferred_activation_precision()` (WaveNet, A2, ConvNet, Linear). Isso é consistente com os pequenos aumentos de latência observados em todos os modelos entre os dashboards pré/pós-fix (ex.: WaveNet Standard CH16: `42.1µs → 43.6µs`; A2 Full: `27.2µs → 28.8µs`), nunca mencionados na "Nota de Conclusão" da Tarefa 3.1 (que só compara latência da LSTM contra o orçamento RT, não contra a baseline pré-fix). Correção de baixo risco: condicionar a criação da guarda a `self.preferred_activation_precision().is_some()`, não apenas a `current.is_none()`.
* **B3 — Regressão de latência absoluta da LSTM subestimada na nota de conclusão:** ver correção em `**Status**` acima — o aumento de ~30-40% de latência absoluta (dentro do orçamento RT) deveria ter sido declarado explicitamente, não apenas comparado ao teto de 1.33ms.
* **B4 — Bug de parsing do dashboard (`utils/quality-dashboard.sh`) permanecia sem correção real, apesar de `[CONCLUÍDA]` (🔴 Alto, corrigido nesta auditoria):** a Tarefa 2.4 diagnosticou uma causa hipotética ("swap" de rótulos) e citou "valores reais" (`BossLSTM-2x8 = 3.88e-3`, `lstm.nam = 1.18e-3`) que **não são reproduzíveis** — a execução direta e repetida de `cargo test --release --test models golden_vectors -- --nocapture` produz consistentemente `2.68e-3` e `1.04e-3`. A causa-raiz real, identificada nesta auditoria: `run_golden_vectors()` (`utils/quality-dashboard.sh`, então linha 221) não usava `--test-threads=1`, ao contrário de `run_reference_oracle()` (linha 242, que já tem o comentário explícito sobre o parser ser "stateful"). A execução paralela padrão do `cargo test` intercala a saída `println!` de testes concorrentes, corrompendo o parser awk baseado em blocos de forma não-determinística — uma "vítima" diferente a cada execução, exatamente o padrão observado (`0.00e0` alternando entre `BossLSTM-2x8` e `lstm (Official)` em dashboards sucessivos). **Corrigido nesta auditoria** (adicionado `--test-threads=1`, verificado estável em 2 execuções consecutivas).
* **B5 — Painel "🎹 ACTIVATION PRECISION" do dashboard exibe `Exact(tanh) = N/A dB` (🟢 Baixo, não corrigido):** `parse_activation_precision()` não está totalmente compatível com o formato de saída do novo `test_lstm_activation_precision_gain_stress_v2` (Tarefa 2.2) — o Δ SNR é capturado corretamente (`+87.3/+89.9/+91.2 dB`), mas o valor absoluto de `Exact(tanh)` não é. Cosmético, não bloqueante, mas deveria ser corrigido antes de publicar o dashboard como referência.

**Ação de Auditoria:** B4 corrigido diretamente (`utils/quality-dashboard.sh`). B1/B2/B3/B5 registrados como follow-ups — ver Épico de Auditoria abaixo. Nenhum deles invalida o resultado central da Sprint 3 (a causa-raiz do drift de `BossLSTM-1x16`/`2x8` está corretamente identificada e corrigida), mas todos deveriam ter sido detectados antes de marcar as tarefas como `[CONCLUÍDA]` — reforçando o padrão de risco de processo já registrado em §3 deste achado ("encerramento de sprints por checklist, não por resultado").

---

## 🏃 Épico Proposto: Investigação de Causa-Raiz do Gap de Fidelidade da LSTM (pós-Kahan)

Agrupamento sugerido para tratamento em `TODO-sprints.md` (a criar/expandir quando solicitado):

1. **Sprint I — Instrumentação de Diagnóstico Correta**

   * Implementar ESR por janela deslizante não cumulativa (4.2).
   * Adicionar decomposição de erro pareada com prewarm para `BossLSTM-1x16`/`2x8` em duração de produção (4.2, 4.5).
   * Transformar `t33_diagnostic_recurrent_drift_lstm_1x16` em regression gate com asserção e baseline versionada (4.4, 4.6).
   * Corrigir tolerância de paridade Kahan em `lstm_test.rs` (4.3).

2. **Sprint II — Isolamento da Causa Dominante**

   * Testar hipótese de `ActivationPrecision` (HF vs Standard) divergente entre produção e oráculo no cenário de longa duração.
   * Testar hipótese de saturação/clipping de ativações aproximadas em regimes de alta amplitude do sinal de stress v2.
   * Testar hipótese de dessincronia de prewarm/estado inicial oráculo-vs-produção.

3. **Sprint III — Correção Direcionada e Documentação**

   * Implementar a correção da causa-raiz real identificada no Sprint II.
   * Atualizar `TODO-parity.md`, `docs/audio_fidelity_map.md`, `docs/fastmath-approximations.md`, `docs/cpp_parity_map.md`, `docs/architecture.md` e criar `docs/lstm_recurrent_drift.md` (já referenciado, mas ausente) com o diagnóstico final real.

---

## Conclusão do Revisor Auditor

A auditoria do Achado A1 foi totalmente concluída com a execução do plano de Sprints (Sprints I, II e III) registrado em `TODO-sprints.md`. O "gap de fidelidade" do `BossLSTM-1x16` foi totalmente resolvido pelo uso do modo `HighFidelity` (nativamente exact tanh/sigmoid), comprovando que o drift era um efeito do transbordo da região de calibração das aproximações Padé/minimax em Standard mode. A instrumentação do oráculo f64 de longa duração com prewarm-paired foi integrada no pipeline automatizado de testes, e o dashboard de qualidade foi atualizado para exibir e parsear os pisos específicos de cada modelo, encerrando as pendências do projeto.
