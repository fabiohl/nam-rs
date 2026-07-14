<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria Final de Compliance & Paridade (Revisor-Auditor, 2026-07-14)

Rodada **conclusiva** de auditoria na role *Compliance and Parity Auditor*, após a implementação
bem-sucedida dos épicos EP1–EP6 da rodada anterior (quick suite enxuto, dashboard JSONL,
recalibração de gates, cadeia de fixtures IEEE-strict com fingerprint de toolchain, paridade
estrita EP5, cobertura v2 promovida ao quick). Base de evidências: `/testes.log`
(run completo de 2026-07-14 00:49, quick + dashboard + `--check`), inspeção direta do código e
triangulação entre golden C++, oráculo f64 e âncoras NumPy.

**Fora de escopo:** `wavenet_a2_max.nam` (tratado à parte em `TODO-wavenet_a2_max.md`).
**Diferido com documentação especializada:** lacuna de precisão ConvNet↔NAMcore
(ver `TODO-convnet_parity.md`, criado nesta auditoria).

**Veredito geral:** a paridade de produção com o NAMcore está excelente — todas as famílias
golden-validadas entre 1e-11 e 1e-14 de ESR, incluindo os novos v2 no quick (WaveNet Standard
v2 9.92e-14, A2-Full v2 1.58e-13). O que restou de **grave** não está no motor de inferência
das famílias principais: está (1) num tipo de modelo suportado cuja correção de áudio não tem
árbitro confiável hoje, e (2) em fissuras de integridade da própria malha de qualidade que
permitem que valores quebrados fiquem verdes. Esta rodada fecha exatamente esses dois flancos.

---

## F1 · `condition_dsp`: semântica sem árbitro confiável — oráculo refutado onde há juiz C++, e produção sem juiz onde não há — **CRÍTICO (correção de áudio)**

### Evidências (todas do run 2026-07-14)

| Medição                                                     | Valor                               | Interpretação                                                                     |
| ----------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------- |
| `wavenet_condition_dsp` prod × golden C++                   | **ESR 1.11e-14** (−139.6 dB)        | Produção é **exatamente** o NAMcore ✓                                             |
| `wavenet_condition_dsp` prod × oráculo f64 (prewarm-paired) | **ESR 4.21e+01** (+16.2 dB!)        | Waveforms sem relação (prod ≈ +0.17, oráculo ≈ −0.03)                             |
| Oráculo Rust × âncora NumPy (`condition_dsp`)               | ESR 0.00e0                          | Oráculo == NumPy — ambos implementam a **mesma** semântica                        |
| `wavenet_condition_lstm` prod × oráculo (paired)            | **ESR 1.67e-1** (−7.8 dB)           | Divergência estrutural, zona audível                                              |
| `wavenet_condition_lstm` golden test (256 amostras, sweep)  | ESR 0.116, SNR 9.3 dB, MR-STFT 0.66 | Passa em gates SNR ≥ 5 dB / ESR < 0.5 / MR-STFT < 0.80                            |
| Golden C++ p/ `condition_lstm`                              | inexistente                         | Limitação upstream: channel mismatch (usa `input_size=1` em vez de `hidden_size`) |

### Diagnóstico lógico (o ponto central desta auditoria)

Onde existe juiz C++ (`condition_dsp` com sub-modelo WaveNet), a produção é bit-compatível com
o NAMcore e **o oráculo diverge por ESR 42** — logo a semântica de `condition_dsp` do oráculo
f64 **e** da âncora NumPy (`validate_oracle_f64.py`, broadcast T5.1 em
`src/testing/reference_oracle/wavenet.rs:32` / `a2.rs:298`) está **refutada**. A âncora não
prova correção do oráculo; prova apenas que Rust-f64 e NumPy compartilham a mesma leitura
(possivelmente errada) da spec.

Consequência direta: a conclusão registrada em `tests/parity/reference_oracle_f64.rs:1025`
("production engine diverges from oracle... Root cause: dispatcher bug (T5.1). Anchor ESR=5e-16
proves oracle is correct") é **não provada** para `condition_lstm` — o mesmo caminho de
broadcast do oráculo que está errado no caso DSP é a única referência usada no caso LSTM.
Isso viola a lição load-bearing do próprio projeto (`docs/cpp_parity_map.md` §4.5: *"the C++
golden is the only arbiter; the oracle decomposes error but never adjudicates it"*) — e já
causou uma regressão real no passado (episódio head1x1, revertido).

### Impacto

* Um tipo de modelo aceito pelo NAM-rs (`WaveNet` com `condition_dsp` LSTM) está produzindo
  áudio que **nenhuma referência confiável confirma**, com divergência em zona audível
  (ESR ~0.12–0.17), atrás de um teste verde.
* O dashboard publica **ESR 4.21e+01 na coluna "vs Ideal (f64)"** para um modelo que é
  NAMcore-perfeito — inversão completa da mensagem ao usuário.
* O contrato de qualidade **congela ESR 0.1162 como "ok"** para `condition_lstm`.

### Proposta de solução (sequência obrigatória — não pular etapas)

1. **T1 — Spec da semântica (árbitro primário):** extrair de
   `tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp` (com file:line) exatamente como
   o output do `condition_dsp` é dimensionado/consumido pelo condition matrix quando
   `out_channels > 1` e quando `< condition_size` (broadcast? tile? erro?). Para o caso LSTM
   (que o próprio upstream não executa), usar o **trainer Python `neural-amp-modeler`** como
   árbitro secundário — é ele quem define a semântica de treino do formato. Registrar a spec
   resultante em `docs/cpp_parity_map.md`.
2. **T2 — Corrigir o oráculo f64 + âncora NumPy** para o caso DSP até que
   `wavenet_condition_dsp` prod × oráculo caia de 4.21e+01 para o piso da família A1
   (≤ 1e-11). Critério objetivo: o juiz C++ já existe; o oráculo só é confiável quando
   concordar com ele neste fixture.
3. **T3 — Re-adjudicar `condition_lstm`** com o oráculo corrigido: se a divergência persistir,
   aí sim é bug de produção (dispatcher/broadcast) — corrigir e validar; se desaparecer, o
   "dispatcher bug" era artefato do oráculo — corrigir a mensagem do `#[ignore]` e recalibrar.
4. **T4 — Política fail-closed provisória (decisão de produto):** enquanto T1–T3 não fecham,
   avaliar rejeitar `.nam` com `condition_dsp` LSTM no load com diagnóstico claro (postura
   parity-first: o NAMcore nem processa esse arquivo), em vez de entregar áudio sem lastro.
   Alternativa mínima: advisory no load + dashboard/contrato marcando "em investigação".
5. **T5 — Recalibrar gates** de `wavenet_condition_lstm` (`tests/common/validation.rs:706`)
   a pisos reais medidos pós-veredito (os atuais SNR ≥ 5 dB / MR-STFT < 0.80 são
   placebo-grade — ver F3) e remover/regravar a entrada do contrato.

---

## F2 · Artefato sintético do self-test T3.1 contamina o dashboard e o CONTRATO — **ALTA (integridade da malha de qualidade)**

* **Evidência:** linha `T3.1: MR-STFT regression gate (synthet… ESR 0.2058 … MR-STFT 1.3017`
  na tabela de fidelidade do dashboard **e** em `docs/quality-contract.txt:52` (o `--check`
  validou "ok … (contrato: 0.2058)").
* **Causa raiz:** `test_mrstft_hard_gate_catches_regression`
  (`tests/models/golden_vectors.rs:2301`) chama `report_dsp_fidelity` com um sinal
  **deliberadamente degradado** dentro de `catch_unwind` para provar que o gate dispara. Com a
  emissão JSONL introduzida no EP2 (`tests/common/validation.rs:446`), a métrica é gravada em
  `metrics.jsonl` **antes** do panic intencional — e o dashboard/contrato a ingerem como se
  fosse um modelo real.
* **Impacto:** (a) o contrato agora "protege" um valor sintético sem sentido; (b) a tabela
  técnica exibe uma linha vermelha falsa; (c) precedente: qualquer futuro self-test que use
  `report_dsp_fidelity` contaminará a telemetria.
* **Proposta:** (1) suprimir emissão JSONL em invocações de self-test — mecanismo simples:
  campo `kind` no JSONL (`"selftest"` vs `"fidelity"`) derivado de um parâmetro/env
  (`NAM_METRICS_KIND`) setado pelo teste T3.1, com o dashboard filtrando `kind != "fidelity"`;
  (2) expurgar a entrada do `docs/quality-contract.txt`; (3) meta-teste novo: interseção entre
  labels do contrato e labels de self-tests conhecidos deve ser vazia (lista negra
  `"(synthetic)"` no `--save` do dashboard).

---

## F3 · Lacuna sistêmica: modelos sem golden `.bin` escapam de TODO o meta-teste anti-placebo — **ALTA (integridade da malha de qualidade)**

* **Evidência:** `test_all_thresholds_anti_placebo` (`tests/models/threshold_calibration.rs:266`)
  itera **apenas** arquivos `.bin` em `tests/fixtures/` e valida `get_calibrated_threshold`.
  `wavenet_condition_lstm` não tem `.bin` (golden C++ impossível) — seus gates vivem só em
  `topology_thresholds` (`tests/common/validation.rs:706`) e **nunca são auditados**. Resultado
  concreto: `mrstft_max = 0.80` está em vigor, **60% acima do teto anti-placebo de 0.5** que o
  próprio meta-teste impõe (Rule 4, linha 330), e SNR mínimo de 5 dB passou despercebido.
* **Impacto:** a classe de modelos validados por oráculo (que tende a crescer — é exatamente a
  classe onde o C++ não alcança) fica fora da única salvaguarda anti-placebo do projeto. É a
  brecha que permitiu o estado do F1 ficar verde.
* **Proposta:** estender o meta-teste para auditar **todas** as fontes de thresholds:
  (a) enumerar as entradas de `topology_thresholds` via lista estática cruzada com os `.nam`
  do CATALOG (ou refatorar as duas funções para uma tabela única consultada pelo meta-teste);
  (b) aplicar as mesmas Rules 1–4 (SNR > 0, ESR < 1.0, MSE-None com compensação, MR-STFT < 0.5);
  (c) exceções só com `skip_reason` explícito e datado (mesmo padrão do CATALOG). A entrada
  `condition_lstm` deve falhar este meta-teste até F1-T5 recalibrá-la.

---

## F4 · Oráculo f64 diverge em modos de gating (Blended/Gated) — coluna "vs Ideal" publica valores falsos para modelos NAMcore-perfeitos — **MÉDIA**

* **Evidência (per-model summary, prewarm-paired):**
  * `a2_dynamic_blended_ch3`: prod × oráculo **ESR 2.53e-1** (−6.0 dB) — mas prod × golden
    C++ = 5.35e-14 (perfeito). Waveforms do oráculo em outra órbita (−0.0085 vs −0.0119).
  * `a2_dynamic_gated_ch8`: prod × oráculo 7.07e-7 (−61.5 dB) — 3+ ordens acima dos pisos
    e-13/e-14 das demais famílias; prod × golden = 5.03e-11 (ok).
* **Diagnóstico:** mesmo padrão do F1 — produção validada pelo juiz C++, oráculo divergente.
  O caminho de gating do oráculo (`src/testing/reference_oracle/a2.rs:364-415`,
  `GatingModeOracle::{Gated,Blended}`) está errado para Blended (grosseiramente) e impreciso
  para Gated. A tabela "vs Ideal (f64)" do dashboard publica 2.53e-01 como se fosse o piso
  ideal do modelo.
* **Proposta:** (1) corrigir o oráculo Blended/Gated usando os fixtures golden-validados como
  critério de aceite (blended ≤ ~1e-12; gated ≤ ~1e-10); (2) gerar âncoras NumPy para ambos os
  modos (hoje inexistentes — a âncora não cobre gating); (3) enquanto não corrigido, o
  dashboard deve exibir `N/A (oráculo em revisão)` para esses modelos em vez do número — mesma
  regra do F1-T4. Executar junto com F1-T2 (mesmo módulo, mesma metodologia de aceite).

---

## F5 · Contrato formalmente VIOLADO: tolerância de latência sub-µs sem piso absoluto — **BAIXA (ganho rápido, run está vermelho)**

* **Evidência:** `✗ Linear RF=2048: latencia regrediu 0.34 us (contrato: 0.3 us, limite: 0.3 us)`
  → `CONTRATO VIOLADO — 1 violacao(oes)`. Regressão de 0.04 µs num benchmark de 0.3 µs: escala
  em que jitter de medição (~dezenas de ns) excede a tolerância relativa de +10%.
* **Proposta:** tolerância composta no `--check` do `quality-dashboard.sh`:
  `limite = max(contrato × 1.10, contrato + 0.05 µs)` (piso absoluto documentado no cabeçalho
  do contrato). Reexecutar `--save` para regravar o valor-base do Linear se a mediana atual
  (0.34 µs) for o novo normal. Sem isso, o quick suite seguirá vermelho por ruído.

---

## F6 · Regressões cosméticas do dashboard pós-JSONL — **BAIXA**

* **SNR cru sem formatação:** coluna SNR exibe `110.70621219463104` (14 casas) — o valor vem
  direto do JSONL sem `printf %.1f`. Corrigir na renderização (`quality-dashboard.sh`,
  consumo de `parse_jsonl_fidelity`).
* **Linhas duplicadas:** o mesmo par (modelo, medição) aparece até 6× com labels distintos
  (`A2-Full`: Container / Container File / WaveNet / Quick / Quick v2 / T-HF1.4). Propor
  deduplicação por chave `(model_file, mode)` mantendo o label mais descritivo, ou seção
  separada "medições redundantes de cobertura".
* **Rótulo vermelho sem contexto de gate:** linhas em zona vermelha (ESR ≥ 1e-1) deveriam
  exibir o gate vigente e um marcador "em investigação" quando aplicável (evita alarme cego ou,
  pior, normalização do vermelho).

---

## Registro de correções já aplicadas nesta auditoria

* `docs/cpp_parity_map.md` §3.5: o item "`condition_dsp.prewarm(0)` is a real, confirmed bug"
  estava **obsoleto** — o código atual chama `cond_dsp.prewarm(cond_dsp.prewarm_samples())`
  (`src/models/wavenet/model_dyn.rs`, `prewarm_internal`). Texto atualizado para FIXED com
  ponteiro para a investigação F1.

---

## Épicos (ordem de execução recomendada)

### EP-A — Veredito do `condition_dsp` (F1 + F4) — **o núcleo desta rodada** [DONE]

Sequência: T1 spec C++/trainer → T2 correção do oráculo DSP (aceite: condition_dsp ≤ 1e-11
vs prod) → F4 correção Blended/Gated (aceite: ≤ 1e-12 / ≤ 1e-10) → T3 re-adjudicação do
`condition_lstm` → T4 política de produto (fail-closed ou advisory) → T5 recalibração de gates
e contrato. **Risco:** médio-alto (mexe no oráculo — pilar de medição); mitigação: os fixtures
golden-validados são o critério de aceite objetivo em cada passo; jamais alterar produção para
"concordar" com o oráculo (lição §4.5). Critério de fechamento do épico: dashboard sem nenhum
valor "vs Ideal" > 1e-6; `condition_lstm` com veredito documentado e gate calibrado real.

### EP-B — Integridade da malha de qualidade (F2 + F3 + F5)

Filtro `kind` no JSONL + expurgo do contrato + meta-teste de labels (F2); anti-placebo
estendido a `topology_thresholds` com Rules 1–4 e skip_reason datado (F3); tolerância composta
de latência com piso absoluto + `--save` do Linear (F5). Baixo risco, alto ganho de segurança.
Critério: `tests-quick.sh && quality-dashboard.sh --check` verde sem entradas sintéticas no
contrato; meta-teste falhando se F1-T5 ainda não recalibrou o condition_lstm (é o comportamento
desejado — trava de segurança).

### EP-C — Polimento do dashboard (F6)

Formatação SNR, deduplicação de linhas, contexto de gate em linhas vermelhas. Executar por
último; puramente apresentacional.
