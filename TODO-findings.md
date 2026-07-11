<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Compliance e Paridade (2026-07-10)

Achados da auditoria focada na role **Compliance and Parity Auditor** (skill
`revisor-auditor`), cobrindo: análise dos logs de `testes.log` (tests-quick,
quality-dashboard e tests-long, todos 100% verdes), paridade com o NAMcore
(`tests/fixtures/NeuralAmpModelerCore/`), correção dos scripts `utils/*` e do
código Rust dos testes/avaliações.

**Escopo e relação com outros artefatos:**

* O achado do modelo `wavenet_a2_max.nam` (3 bugs de produção confirmados no motor
  dinâmico A2) tem documento dedicado — [`TODO-wavenet_a2_max.md`](TODO-wavenet_a2_max.md)
  — e **não é duplicado aqui**; o Achado F8 apenas o referencia e registra as
  contribuições incrementais desta auditoria a ele.
* Correções já **aplicadas durante esta auditoria** (não requerem implementação,
  apenas ciência da equipe): otimização da Fase 1 do `utils/tests-quick.sh`
  (Achado F5) e sincronização dos docs (`cpp_parity_map.md` §4.4/§7.1/§7.2,
  `audio_fidelity_map.md` §3, `perceptual_validation.md` §MR-STFT,
  `testing.md` §2, `benchmarks.md`).

> [!WARNING]
> **Atualização (2026-07-11) — auditoria de verificação por execução.** Os
> Achados F1–F10 abaixo foram implementados (Sprints 1–5, ver
> `TODO-sprints.md`, todos `[x]`). Uma auditoria de acompanhamento **executou**
> os artefatos resultantes (em vez de apenas revisá-los estaticamente) e
> encontrou duas lacunas concretas, ambas com reprodução empírica: **Achado
> F11** (flakiness real e reproduzida em `tests/models.rs`, causada por
> rollout incompleto do `PrecisionGuard` do Épico A.1) e **Achado F12**
> (o oráculo f64 do A2, "corrigido" no Épico C.3, ainda **panica** para
> `wavenet_a2_max.nam`). Ambos geraram o **Épico G** (residual, novo). Nenhum
> dos dois invalida o trabalho dos Sprints 1–5 — a maior parte das alegações
> foi verificada como correta — mas confirma que revisão estática de código
> não é suficiente para tarefas que tocam estado global compartilhado ou
> caminhos cobertos só por testes `#[ignore]`d; ver a "Nota de processo" em
> cada achado.

**Estado geral de paridade (contexto positivo):** a suíte está saudável. Todos os
golden vectors v1 passam com ESR na faixa 1e-11…1e-14 (interop "IDENTICO" vs
NAMcore); o oráculo f64 casa com as âncoras NumPy a −153 dB; LSTMs convergiram
para ESR ≈ 1.4e-11 com o default `Standard`. Os achados abaixo são, em sua
maioria, defeitos de **metrologia e confiabilidade do harness de testes** — não
de áudio de produção — mas minam a capacidade da suíte de detectar regressões
reais (placebos, falso-positivos crônicos, duplicações e medições incomparáveis).

---

## Achado F1 — 🔴 Testes `quick_parity_hf_*` são no-ops duplicados (toggle HF inefetivo)

**Evidência primária (testes.log):** `quick_parity_lstm_1x16` e
`quick_parity_hf_lstm_1x16` produzem métricas **byte-idênticas**
(MSE=1.55e-12, MR-STFT=7.2404e-5); idem `quick_parity_wavenet_ch16` vs
`quick_parity_hf_wavenet_ch16` (MSE=6.93e-15, MR-STFT=1.0382e-5).

**Causa-raiz (confirmada em código):**

* `tests/parity/cpp_parity.rs:536-540` — com `use_hf: true`, o teste instala
  `PrecisionGuard::set()` → `set_activation_precision(ActivationPrecision::Standard)`.
  Com `use_hf: false`, **nenhum modo é definido**.
* `src/math/activations/mod.rs:108-109` — o default global do processo **já é
  `Standard`**. Logo, ambos os caminhos executam exatamente o mesmo código.
* O lado C++ sempre usa math exata (`activations.cpp:16`:
  `using_fast_tanh = false`; `render.cpp` nunca chama `enable_fast_tanh()` e não
  possui flag de ativação). Não há mecanismo possível de diferenciação.

**Defeitos associados:**

1. **Comentário factualmente errado** em `cpp_parity.rs:1228-1241`: afirma que o
   "C++ NAMCore usa aproximações Padé/minimax" — falso (usa `std::tanh`/`expf`
   exatos), e contradiz os próprios comentários das linhas 55-59 e 530-534.
2. **Threshold HF invertido**: para WaveNet, o cap ESR do teste HF é
   `A2ESR_A1_STANDARD_MEDIAN * 5.0` (5× mais **frouxo** que o não-HF, 0.03115 vs
   0.00623) — o oposto do que faz sentido físico. Para LSTM, o cap HF é 8000×
   mais apertado (1.0e-5 vs 0.08). Não há metodologia unificada.
3. **Assimetria do `PrecisionGuard` (hazard de determinismo)**: o caminho não-HF
   nunca fixa `ActivationPrecision::Fast` explicitamente. Se outro teste no mesmo
   processo deixar o atômico global em `Fast`, o teste não-HF roda silenciosamente
   em modo diferente do esperado. O guard defende só o lado HF.
4. **Lacuna de cobertura real**: o modo `Fast` (Padé) — o único que difere do
   default — **nunca é comparado contra referência externa** na suíte quick.

**Proposta de solução:**

* Redefinir a semântica: teste "não-HF" deve fixar explicitamente
  `ActivationPrecision::Fast` (via guard RAII) e usar thresholds calibrados para
  Padé; teste "HF" mantém `Standard` com thresholds apertados. Como o C++ não
  tem toggle, a comparação Fast-vs-referência deve usar caps calibrados a partir
  do erro conhecido do Padé (~2.3e-3) ou o oráculo f64 (que já cobre isso em
  `reference_oracle_f64.rs`). Alternativa mais simples e honesta: **remover** os
  `quick_parity_hf_*` duplicados da suíte quick (ganho de tempo, zero perda de
  cobertura) e manter apenas um par por modelo com guard explícito.
* Corrigir o comentário 1228-1241 e revisar o cap `* 5.0` do WaveNet HF
  (provável intenção: `/ 5.0` ou cap absoluto tipo 1e-10, dado que o medido é
  2.46e-14).
* Introduzir um `PrecisionGuard` simétrico (sempre fixa o modo desejado no
  início e restaura no drop) em **todos** os testes de paridade que dependem do
  modo de ativação.

**Arquivos:** `tests/parity/cpp_parity.rs` (752-774, 1246-1263, 536-540,
444-487, 1228-1241), `src/math/activations/mod.rs`.

---

## Achado F2 — 🟠 MR-STFT: falso-positivos sistemáticos do soft gate e limiar não calibrado

**Evidência primária (testes.log):** avisos "⚠ MR-STFT soft gate" recorrentes em
testes com paridade essencialmente perfeita: `linear_fft rf320` → 4.45e-1;
`rf8192` → 2.63e0; `rf8192 sine` → 1.60e0; goldens C++ RF=2048/4096/8192 →
~1.6e-1 — todos com ESR entre 1e-12 e 1e-14 (−113 a −140 dB). Avisos que "sempre
disparam" treinam a equipe a ignorá-los (fadiga de alarme) e mascaram o dia em
que o aviso for real.

**Causa-raiz (matemática, confirmada em `src/testing/perceptual.rs:139-225`):**
a métrica é uma distância L1+L2 em **log-magnitude absoluta** com piso
`eps=1e-8`. Para sinais de banda estreita (seno 440 Hz dos testes Linear FFT),
~2045 dos 2049 bins da janela de 4096 (peso 0.5 na soma multi-resolução) estão
no piso de ruído; diferenças absolutas ~1e-7 viram log-ratios ~2.0 por bin,
dominando o score. ESR=1e-13 com MR-STFT=2.6 é consequência direta da fórmula,
não degradação.

**Defeitos associados:**

1. `MRSTFT_SOFT_THRESHOLD = 0.15` (`tests/common/validation.rs`) é um número
   mágico hardcoded — sem calibração, sem variação por modelo, **fora da
   cobertura do meta-teste** `threshold_calibration.rs` (que só valida
   `mrstft_max` dos modelos em `golden_bin_to_model_name()`).
2. A família **Linear inteira** tem `mrstft_max = None`
   (`validation.rs:749,791`) → o hard gate nunca se arma para Linear em nenhuma
   taxa; e os goldens `golden_linear_fft_rf*.bin` não são mapeados em
   `golden_bin_to_model_name()` → excluídos da meta-cobertura de calibração.
3. **Rótulo enganoso**: `validation.rs:274` imprime `MR-STFT = … (relative)` —
   a métrica é absoluta (distância em espaço log-magnitude), não relativa.
4. Relaxação do `mrstft_max` usa expoente `snr_relaxation/5.0` vs `/10.0` do
   MSE/ESR (`golden_vectors.rs:92-106`) sem rationale documentado.

**Proposta de solução:**

* **Condicionamento da métrica (preferido):** mascarar bins cuja magnitude de
  referência esteja abaixo de um piso relativo ao pico espectral global do
  frame (ex.: −80 dB do pico), em vez do `eps` absoluto 1e-8. Isso elimina a
  dominância do piso de ruído preservando a sensibilidade em bins com energia
  real. Regerar o golden Python (`gen_mrstft_golden.py`) em conjunto para manter
  a paridade bit-a-bit com a referência.
* Alternativa mínima: suprimir o soft gate quando `ESR < 1e-9` (a paridade
  temporal já prova ausência de degradação espectral audível), documentando a
  precedência do ESR.
* Calibrar `MRSTFT_SOFT_THRESHOLD` empiricamente por classe de sinal, cobri-lo
  no `threshold_calibration.rs`, e mapear os goldens Linear FFT em
  `golden_bin_to_model_name()`.
* Trocar o rótulo `(relative)` por `(log-mag abs)` e documentar o expoente `/5`
  ou unificá-lo com `/10`.

**Arquivos:** `src/testing/perceptual.rs:139-225`, `tests/common/validation.rs`
(262-288, 274, 749, 791), `tests/models/threshold_calibration.rs:51-82`,
`tests/models/linear_fft_test.rs`, `tests/fixtures/scripts/gen_mrstft_golden.py`.

---

## Achado F3 — 🟠 Oráculo f64: metodologia assimétrica (cold-start) na summary table e nas decomposições

**Evidência primária (testes.log):** `test_summary_table` reporta
`wavenet_a2_film_lite.nam` com ESR = **2.825e0 (+4.5 dB!)** e
`wavenet_a2_film_full.nam` com 2.125e-1, enquanto as medições pareadas-com-prewarm
dos **mesmos modelos** dão 1.61e-13 (−127.9 dB) e 8.75e-15 (−140.6 dB). A
"Rule 5" (Σ fontes ≈ total, ±10×) é violada por 5–8 ordens de magnitude nas
decomposições WaveNet/A2/ConvNet.

**Causa-raiz (confirmada em `tests/parity/reference_oracle_f64.rs:1000-1036`):**
a summary table compara **produção com `prewarm(2048)`** contra **oráculo
totalmente frio**, em janela de 256 amostras — para modelos com campo receptivo
~2048, 100% da janela é transiente de preenchimento do RF do oráculo. As
decomposições `test_decomposition_{wavenet,lstm,a2,convnet}` usam o mesmo
cold-start (`run_decomposition()`), enquanto as variantes BossLSTM já usam a
metodologia correta (`run_decomposition_paired`, prewarm 24k).

**Agravante:** esses números cold-start vazam para a coluna "vs Ideal (f64)" do
`quality-dashboard.sh` (parser em `parse_oracle_f64`), produzindo inclusive
valores **inconsistentes entre linhas** para o mesmo modelo (A2-Full aparece com
`1.82e-14~` numa linha e `6.13e-14~` noutra) porque a tabela por-modelo (fria) e
os rótulos de família pareados compartilham o mesmo array associativo `ESR_F64`
com política last-write-wins.

**Proposta de solução:**

* Migrar `test_decomposition_{wavenet,lstm,a2,convnet}` para
  `run_decomposition_paired()` (mesma infra já usada pelos BossLSTM) — a Rule 5
  passa a ser um invariante verificável de verdade para todas as arquiteturas.
* Fazer `test_summary_table` consumir as medições pareadas
  (`run_oracle_esr_paired`) ou, no mínimo, remover o `prewarm(2048)` da produção
  para uma comparação fria-vs-fria simétrica e rotulada como tal.
* No dashboard: separar `ESR_F64` em dois arrays (cold vs paired) com
  proveniência explícita, e nunca exibir valor cold-start na coluna
  "vs Ideal (f64)" para modelos com RF > janela de medição.

**Arquivos:** `tests/parity/reference_oracle_f64.rs` (381, 407, 638, 755,
1000-1036), `src/testing/reference_oracle/mod.rs:422-472`,
`utils/quality-dashboard.sh` (`parse_oracle_f64`, 446-554).

---

## Achado F4 — 🟠 Oráculo f64 `condition_dsp` quebrado (não a produção) + rastreamento incompleto dos `#[ignore]`

**Evidência:** `test_golden_vectors_wavenet_condition_dsp` **passa** contra o
C++ com ESR=1.13e-14 (produção correta); a âncora NumPy independente também
diverge do oráculo (`test_oracle_vs_python_anchor_a2_generic` ignorado). O
oráculo tem bug dimensional confirmado no caminho `condition_dsp` (produz 1
valor/frame; o C++ produz `condition_size` valores/frame) e bug de contagem de
pesos do `head1x1` (`docs/cpp_parity_map.md` §4.4/§4.5;
`src/testing/reference_oracle/a2.rs:292-296, 583-829`).

**Defeitos associados:**

1. **Mensagem de `#[ignore]` desatualizada/enganosa**: `test_oracle_a2_generic`
   diz "condition_dsp output divergence between production and oracle … Requires
   investigation of condition_dsp Cascade vs single-array A2 condition processing
   path" — sugere dúvida sobre a produção, quando o veredito atual
   (`TODO-wavenet_a2_max.md` §2) é que o lado quebrado do oráculo é o oráculo, e
   os bugs de produção do a2_max são outros três (A/B/C), independentes.
2. **`meta_coherence` não cobre `reference_oracle_f64.rs`**: o meta-teste
   (`tests/models/meta_coherence.rs:199-252`) só escaneia `golden_vectors.rs` e
   `linear_fft_test.rs` — os 4 testes ignorados do oráculo referentes ao
   `wavenet_a2_max.nam` ficam fora de qualquer rastreamento de catálogo.
3. O oráculo A2 sempre usa o caminho cascade (mesmo single-array), nunca
   exercitando a decisão de dispatch fast-path da produção — aceitável, mas deve
   ser documentado como limitação intencional.

**Proposta de solução:**

* Corrigir o oráculo `condition_dsp` (dimensionalidade `condition_size`
  valores/frame + contagem de pesos `head1x1` por camada) — já mapeado como
  **Epic 6 do `TODO-wavenet_a2_max.md`** (ortogonal, não bloqueante).
* Atualizar as 4 mensagens de `#[ignore]` para refletir o diagnóstico vigente
  (bugs de produção A/B/C + bug do oráculo, com ponteiro para
  `TODO-wavenet_a2_max.md`).
* Estender `meta_coherence` para escanear também
  `tests/parity/reference_oracle_f64.rs` (e qualquer futuro entry-point) por
  modelos `.nam` citados em testes ignorados.

**Arquivos:** `src/testing/reference_oracle/a2.rs`,
`tests/parity/reference_oracle_f64.rs:864-930`,
`tests/models/meta_coherence.rs:199-252`.

---

## Achado F5 — ✅ (corrigido nesta auditoria) `tests-quick.sh` duplicava oráculos em debug e disparava build C++ no meio da Fase 1

**Evidência (testes.log):** os 5 `quick_parity_*` rodaram **duas vezes** (debug
na Fase 1 — linhas ~2582 — e release na Fase 2); `golden_vectors`,
`linear_fft_test`, `spectral_fidelity` e `reference_oracle_f64` idem; o build
CMake do render C++ do NAMcore foi disparado lazamente **no meio da Fase 1
debug** (43.95s no binário parity debug), poluindo o log com centenas de
warnings `-Weffc++` do C++ vendorizado. Isso violava o próprio contrato do
script ("EXCLUI os 5 oráculos de medida do §7") e o Eixo B do
`docs/testing.md` §7 (medição em debug = "fantasma" de codegen). Tempo real
medido: 5m49s (header prometia 35s hot).

**Correção aplicada:** Fase 1 agora usa `--skip <módulo>::` (match exato de
prefixo de módulo, imune à colisão histórica de nomes soltos) para
`golden_vectors`, `linear_fft_test`, `spectral_fidelity`, `reference_oracle_f64`,
`cpp_parity` e `isa_parity`; header atualizado para estimativas honestas
(±2 min hot / ±5 min cold). Validado: todos os binários passam com os skips.

**Follow-ups sugeridos (não aplicados):**

* Considerar pré-buildar o render C++ explicitamente (mensagem dedicada) antes
  da Fase 2, para que o custo do CMake nunca apareça "dentro" de um teste e o
  log de compilação C++ vá para arquivo em `target/logs/` em vez do stdout.
* Avaliar remover `--test clap` da Fase 1 (binário com 0 testes sem a feature
  `clap-plugin`) — custo de compilação sem cobertura; manter exige justificativa
  de guarda-futuro.

**Arquivos:** `utils/tests-quick.sh` (Fase 1), `tests/parity/cpp_parity.rs:117-210`
(`ensure_render_compiled`).

---

## Achado F6 — 🟠 `quality-dashboard.sh`: proveniência inconsistente, seções mortas e linha suspeita "ESR 0.00e0 / SNR inf"

**Evidências (saída do dashboard em testes.log, linhas 3684-3877):**

1. **Linha suspeita:** `Container A2-Lite (CH=3) C++ cross-ref │ 0.00e0 │ SNR inf`
   — ESR exatamente zero com SNR infinito é assinatura de comparação
   buffer-contra-si-mesmo ou de label colidindo no parser awk de blocos
   (`parse_golden_vectors`). A linha irmã `Container File A2-Lite` reporta
   6.43e-14 (valor plausível). Requer investigação: ou o teste realmente é
   bit-exato (documentar por quê), ou há colisão de chaves/labels no parser.
2. **Seções mortas no modo full:** "ISA PARITY — Nenhum resultado disponivel"
   (o run quick de `isa_parity` só imprime o header informativo; as linhas
   `[ISA Matrix]` que o parser procura só existem no long) e "SPECTRAL FIDELITY
   — Nenhum resultado disponivel" (o grep procura a frase
   `all spectral fidelity metrics within baseline tolerance`, que os testes
   sintéticos não imprimem). Seções que nunca populam no modo em que rodam são
   propaganda enganosa de cobertura.
3. **Coluna "vs Ideal (f64)" mistura metodologias** (ver Achado F3) e o array
   `ESR_F64` compartilhado gera valores diferentes para o mesmo modelo em linhas
   distintas.
4. **ConvNet sem cross-validação C++**: linha "vs NAMcore: N/A" — o NAMcore
   suporta ConvNet; não há teste `cpp_parity`/golden para a família (lacuna de
   cobertura de interop, hoje só self-golden e oráculo f64).
5. Cosmético: alinhamento das bordas do cabeçalho (`ISA:`/`CPU:`/`rustc:`)
   estoura a moldura; "Medido em" ok.

**Proposta de solução:** investigar/normalizar o label do Container A2-Lite;
fazer as seções ISA/Spectral declararem explicitamente "não coberto no modo
quick — rode tests-long" (ou parsear o que realmente existe); separar
proveniência cold/paired (F3); adicionar golden/cpp_parity para ConvNet; ajustar
`printf` do cabeçalho.

**Arquivos:** `utils/quality-dashboard.sh` (parse_golden_vectors,
parse_isa_parity:559-586, parse_spectral_fidelity:590-594, render_header),
`tests/models/golden_vectors.rs` (labels Container), `tests/parity/cpp_parity.rs`
(cobertura ConvNet).

---

## Achado F7 — 🟡 Meta-teste `test_mrstft_hard_gate_catches_regression` vaza bloco de falha e panic no stdout

**Evidência (testes.log 3057-3070):** o log da suíte verde contém um bloco
inteiro com "✗" em MSE/SNR/ESR/MR-STFT e um
`thread … panicked at tests/models/golden_vectors.rs:2131` — assustador para
qualquer leitor, embora seja o comportamento esperado do meta-teste (verifica
que o gate pega regressões sintéticas).

**Causa-raiz:** `report_dsp_fidelity_impl` (`tests/common/validation.rs:335-362`)
**imprime o relatório antes** de executar os asserts; o `catch_unwind` do
meta-teste captura o panic, mas não pode apagar o que já foi para o stdout
(com `--nocapture`, tudo vaza).

**Proposta de solução:** adicionar um modo silencioso ao harness (ex.:
`report_dsp_fidelity_quiet(...)` ou flag thread-local `SUPPRESS_REPORT`) usado
pelos meta-testes; e/ou registrar um panic-hook temporário no meta-teste para
suprimir a mensagem de panic. Critério de aceite: rodada verde da suíte não
contém nenhum "✗" nem "panicked" no stdout.

**Arquivos:** `tests/common/validation.rs:335-362`,
`tests/models/golden_vectors.rs:2084-2147`.

---

## Achado F8 — 🔴 `wavenet_a2_max.nam` (referência cruzada) + contribuições desta auditoria

O achado-mestre está em [`TODO-wavenet_a2_max.md`](TODO-wavenet_a2_max.md)
(3 bugs de produção confirmados: head1x1 por-camada vs por-array; `groups`
ignorado em `layer1x1`/`input_mixin`; kernel do head legado K=1 vs hardcoded 16;
Epics 1–7). **Contribuições desta auditoria àquele plano:**

1. `docs/cpp_parity_map.md` §4.4/§7.1 **atualizados** (feito): o diagnóstico
   antigo ("condition_dsp internal processing" como suspeito dominante) foi
   substituído pelo veredito A/B/C, e as referências mortas a `TODO-sprints.md`
   redirecionadas.
2. As mensagens de `#[ignore]` dos testes do oráculo devem ser atualizadas para
   o diagnóstico vigente (ver Achado F4.1) — acrescentar ao Epic 7 (documentação)
   do TODO-wavenet_a2_max.
3. O gap de rastreamento do `meta_coherence` (Achado F4.2) cobre exatamente os
   testes ignorados daquele plano — acrescentar ao Epic 5 (auditoria de
   cobertura).
4. Nota de risco para o Epic 2 (Bug A): a tabela §4.6 do `cpp_parity_map.md`
   ainda marca "head1x1 weight count ✅ Matches C++ formula" — esse veredito é
   por-array e ficará obsoleto com a correção por-camada; atualizar a tabela ao
   fechar o Epic 2.

---

## Achado F9 — 🟡 Estado global `ActivationPrecision` como atômico de processo é frágil para testes paralelos

Generalização do F1.3: qualquer teste que altere o modo global sem guard RAII
pode envenenar testes concorrentes no mesmo processo (o `cargo test` roda em
paralelo por padrão). Hoje o `PrecisionGuard` existe só em `cpp_parity.rs` e só
no braço HF.

**Proposta de solução:** promover o `PrecisionGuard` para
`tests/common/` com semântica completa (seta modo no `new`, restaura o anterior
no `Drop`, serializado por um `Mutex` estático para exclusão mútua entre testes
que dependem do modo); auditar via grep todos os call-sites de
`set_activation_precision` em testes e envolvê-los no guard.

**Arquivos:** `tests/common/`, `tests/parity/cpp_parity.rs:60-73`,
call-sites de `set_activation_precision` em `tests/` e `src/` (testes unitários).

> **Atualização (2026-07-11) — implementação parcial, ver Achado F11.** O
> `PrecisionGuard` (`tests/common/precision.rs`) foi criado com exatamente a
> semântica proposta (Mutex estático + RAII), e a maioria dos call-sites foi
> migrada. Porém a auditoria de acompanhamento (F11) encontrou 3 call-sites
> ainda desprotegidos que **já causaram falhas intermitentes reproduzidas**
> nesta sessão. Este achado permanece aberto até F11 ser resolvido.

---

## Achado F10 — 🟡 Higiene menor (agregado)

1. **`(relative)` no rótulo do MR-STFT** — coberto em F2.3.
2. **Expoente de relaxação `/5` vs `/10`** — coberto em F2.4.
3. **Warnings `-Weffc++` do C++ vendorizado** poluem os logs em todo build do
   render; adicionar `-w` (ou `-Wno-effc++`) nas flags do CMake do
   `ensure_render_compiled`/`golden_gen_build.sh` — o código vendorizado não é
   nosso para consertar.
4. **`docs/namb-spec.md`** ainda referencia `TODO-sprints.md` (S3.T03…) —
   arquivo não existe mais; atualizar a referência de planejamento.
5. **Dashboard referencia "TODO-findings.md Achado A1"**
   (`utils/quality-dashboard.sh`, nota da seção F64 ORACLE) — numeração antiga;
   o conteúdo correspondente agora é o **Achado F3** deste documento.

---

## Achado F11 — 🔴 Rollout incompleto do `PrecisionGuard` causa flakiness **reproduzida** em `tests/models.rs` (regressão do Épico A.1 / F9)

**Contexto:** o Épico A.1 (F9) foi marcado `[x]`/`[DONE]` em `TODO-sprints.md`
(S1.T01–S1.T03) com a alegação "auditoria e refatoração dos call-sites de
`set_activation_precision`". Esta auditoria de acompanhamento **verificou
essa alegação executando a suíte repetidamente** e encontrou 3 call-sites que
seguem chamando `set_activation_precision()` diretamente, sem adquirir o
`PRECISION_MUTEX` via `PrecisionGuard` — e comprovou, empiricamente, que isso
já produz **falhas intermitentes reais**, não apenas um risco teórico.

**Reprodução (determinística em poucas tentativas, sem qualquer alteração de
código):**

```text
$ cargo test --release --test models              # execução 1
failures: namb_v2_roundtrip::test_lstm_gate_major_roundtrip

$ cargo test --release --test models              # execução 2 (mesmo binário)
test result: ok. 223 passed; 0 failed …

$ cargo test --release --test models              # execução 3
failures: namb_v2_validation::test_lstm_v2_gate_major_parity
```

Falhas **diferentes a cada execução** — assinatura clássica de corrida em
estado global mutável compartilhado por threads de teste concorrentes (o
`cargo test` usa `--test-threads = nproc` por padrão; esta máquina tem 16
threads; `utils/tests-quick.sh` Fase 1 não passa `--test-threads=1` para o
binário `models`).

**Causa-raiz confirmada:**

1. `tests/models/activation_precision.rs` contém 3 testes que chamam
   `set_activation_precision(...)` **sem** nenhum `PrecisionGuard` no escopo:
   `test_zero_alloc_activation_switch_primitive` (linhas 362-364),
   `test_zero_alloc_activation_hot_path_switch` (411, 419, 427, 450),
   `test_zero_alloc_cli_activation_flow` (493, 504). Todos vivem no mesmo
   binário `tests/models.rs` que `lstm_activation_precision.rs` e
   `lstm_model_dyn_validation.rs` — que **corretamente** usam
   `PrecisionGuard`, mas cuja proteção é **inútil** porque o `Mutex` só
   serializa quem tenta adquiri-lo; os 3 testes acima mutam o átomo global
   por fora, sem esperar ninguém.
2. Efeito colateral em testes que **nem sabiam que dependiam** de precisão
   estável: `namb_v2_roundtrip::test_lstm_gate_major_roundtrip` e
   `namb_v2_validation::test_lstm_v2_gate_major_parity` constroem dois modelos
   LSTM (original vs. round-trip NAMB v2) e comparam duas chamadas
   **sequenciais** de `model.process(...)` com `assert!(mse < 1e-12)`. Se o
   átomo global `ActivationPrecision` flipar entre a primeira e a segunda
   chamada (porque outro teste, em outra thread, chamou
   `set_activation_precision(Fast)` naquele instante), as duas saídas passam
   a vir de modos de ativação diferentes — divergência de ~1e-1…1e-2 em ESR,
   muito acima do gate de 1e-12 — e o teste falha **por um motivo
   completamente não relacionado ao que ele se propõe a verificar** (paridade
   de round-trip do formato binário, não precisão de ativação).
3. Nenhum desses dois testes-vítima (`namb_v2_*`) faz nada de errado por si
   só — a premissa implícita de qualquer teste que chama `model.process()`
   mais de uma vez e compara resultados é que o estado global de precisão
   permanece estável durante o teste. Essa premissa só pode ser garantida se
   **todo** call-site que muta o átomo participar do mesmo mecanismo de
   exclusão — o que hoje não é o caso.

**Por que isso é mais grave que um "risco teórico" (F9 original):** a suíte
`tests-quick.sh` é descrita como "rodada várias vezes ao dia" — com esta
flakiness, qualquer desenvolvedor pode ver um teste não relacionado ao seu
trabalho falhar de forma não reprodutível, perder tempo investigando um
fantasma, e o hábito de "rodar de novo e passou" mina a confiança em toda a
suíte (exatamente o tipo de erosão de confiança que a role Compliance and
Parity Auditor deve combater).

**Proposta de solução (por ordem de robustez, não mutuamente exclusivas):**

1. **Imediata (mitigação pontual):** envolver os 3 call-sites identificados
   com `PrecisionGuard::new(...)` antes de qualquer chamada direta a
   `set_activation_precision`, preservando a intenção de cada teste (medir
   zero-alloc do primitivo) — a aquisição do guard deve ocorrer **antes** de
   iniciar o `TrackingGuard`/contagem de alocações, para não contaminar a
   métrica medida.
2. **Estrutural (recomendada):** não confiar apenas em opt-in. Ou (a) forçar
   `--test-threads=1` para o binário `models` inteiro em
   `utils/tests-quick.sh` Fase 1 (simples, custo: perde paralelismo só nesse
   binário — medir impacto no tempo total), ou (b) adicionar um meta-teste
   estático (`grep`-based, no estilo dos demais meta-testes de
   `threshold_calibration.rs`) que falha o build se qualquer
   `set_activation_precision(` aparecer em `tests/**/*.rs` fora de
   `tests/common/precision.rs` sem um `PrecisionGuard::new` na mesma função —
   fechando a porta a novas regressões do mesmo tipo.
3. **Auditoria imediata:** revisar todos os testes que chamam
   `model.process()` duas ou mais vezes na mesma função comparando saídas
   (roundtrip/parity de qualquer formato, não só NAMB) e confirmar que todos
   rodam sob um `PrecisionGuard` implícito ou explícito — a lista não se
   limita aos dois casos encontrados aqui (busca não foi exaustiva por todo o
   repositório, apenas até a primeira reprodução).

**Arquivos:** `tests/models/activation_precision.rs` (362-364, 411-450,
493-504), `tests/models/namb_v2_roundtrip.rs:480-529` (vítima),
`tests/models/namb_v2_validation.rs:54-93` (vítima), `tests/common/precision.rs`,
`utils/tests-quick.sh` (Fase 1, sem `--test-threads=1` no binário `models`).

**Nota de processo:** S1.T03 estava marcado `[x]` em `TODO-sprints.md` com base
em "auditoria e refatoração dos call-sites" — a auditoria foi real e cobriu a
maioria dos casos, mas não foi **verificada sob execução paralela repetida**,
que é o único teste que realmente comprova ausência de corrida. Recomenda-se
que qualquer tarefa futura envolvendo estado global mutável em testes só seja
marcada `[x]` após N execuções consecutivas (ex.: 10×) sem falha, não apenas
após revisão estática do código.

---

## Achado F12 — 🔴 Oráculo f64 do A2 (`src/testing/reference_oracle/a2.rs`) ainda quebra para `wavenet_a2_max.nam` após o "fix" do Épico C.3/Epic 6 — panic, não apenas ESR alto

**Contexto:** `TODO-sprints.md` S2.T03 marca como `[x]` "Correção do Oráculo
WaveNet A2" (commit `b7a8fb4`, "fix: correct A2 oracle head1x1 dimensions,
condition_dsp window, and head_b type"), delegando ao Epic 6 do
`TODO-wavenet_a2_max.md`. Esta auditoria **executou** o único teste ignorado
do oráculo A2-generic que não depende do guard de dispatch bloqueado
(`test_oracle_vs_python_anchor_a2_generic`, que chama `oracle_forward`
diretamente, sem passar por `build_model`) e obteve um **panic**, não uma
medição de ESR:

```text
$ cargo test --release --test parity reference_oracle_f64::test_oracle_vs_python_anchor_a2_generic -- --ignored --nocapture

thread '...' panicked at src/testing/reference_oracle/mod.rs:179:38:
range end index 826 out of range for slice of length 818
```

**Causa-raiz (confirmada por reconciliação exata do JSON do fixture):**

`wavenet_a2_max.nam` tem `layers[0]` com `head_size=1`, `head_bias=true`,
`condition_size=8`, `channels=bottleneck=4`, `head1x1={active:true,
out_channels:4, groups:2}`, `dilations=[1,2]` (**2 camadas**), e — ponto
central — `cfg.head` é `None` (cabeçalho legado, plano, sem objeto aninhado;
confirma exatamente a premissa do Bug C do `TODO-wavenet_a2_max.md`). O total
de pesos do array é **818**, número que casa exatamente com a reconciliação
manual já publicada em `TODO-wavenet_a2_max.md` §3.5
(`818 = 4 + 2×404 + 5 + 1`).

O `a2.rs` atual (pós-`b7a8fb4`) lê a seção de `head1x1` e do `head` final **uma
única vez, depois do laço de camadas** (linhas 468-523 de `a2.rs`) — ou seja,
modela `head1x1` como **por-array**, exatamente o mesmo formato estruturalmente
errado que a produção (`WaveNetA2Dyn`) tem, documentado como **Bug A** no
`TODO-wavenet_a2_max.md`. A reconciliação `818 = 4 + 2×404 + 5 + 1` — com o
bloco de 404 **repetido por camada** — indica que, na ordem real do stream de
pesos do C++, `head1x1` (e o cabeçalho, mais modesto) são consumidos **dentro**
do bloco por-camada, não depois de todas as camadas. Em outras palavras: **o
oráculo f64 sofre do mesmo Bug A que a produção**, e o commit `b7a8fb4`
corrigiu apenas as *fórmulas* de dimensão dentro da estrutura por-array
errada (daí o "fix" ter reduzido, mas não eliminado, a divergência de
orçamento de pesos — de um erro maior para um resíduo de exatamente
`826 − 818 = 8`, coincidindo com `condition_size`, mas não necessariamente
causado só por isso; o valor exato do resíduo depende da ordem de leitura e
não deve ser tomado como a causa isolada sem nova reconciliação de campo a
campo).

Adicionalmente, a ramificação `if head_size == 1 { /* usa A2_HEAD_KERNEL=16 */
} else { /* Conv1x1 */ }` (linha ~502 de `a2.rs`) usa `head_size`
(**canais de saída do head**, aqui `=1`, ou seja "mono") como proxy para
"formato canônico A2 de 16 taps" — mas para este fixture o formato correto é
o cabeçalho legado (`Conv1x1`, K=1), não o de 16 taps. Isso é o mesmo Bug C do
`TODO-wavenet_a2_max.md`, agora confirmado também no oráculo, e explica por que
mesmo corrigindo o resíduo de 8 elementos o oráculo ainda consumiria ~64 pesos
de mais do que deveria para o `head_w` deste modelo.

**Confirmado que o sub-modelo `condition_dsp` aninhado tem exatamente o mesmo
padrão** (`head1x1.active=true` em ambos os seus 2 arrays, com 2 e 3 camadas
respectivamente) — ou seja, o mesmo bug se aplicaria recursivamente à chamada
`oracle_forward(&cond_model, ...)` (linha 292-296 de `a2.rs`) assim que o
array externo parasse de panicar primeiro.

**Impacto:** Épico 6 do `TODO-wavenet_a2_max.md` ("corrigir o oráculo — não
bloqueante") está **efetivamente incompleto** — o oráculo não apenas mede
errado, ele **quebra em runtime**. Isso é crítico para quem for atacar os
Epics 2–4 daquele plano (corrigir Bugs A/B/C na produção): ao tentar validar o
progresso comparando produção corrigida contra o oráculo f64, a primeira
tentativa de rodar qualquer teste do oráculo sobre `wavenet_a2_max.nam`
**falhará com um panic de índice fora dos limites**, sem sinal algum sobre a
correção da produção. Nenhum teste passante hoje é afetado (nenhum outro
fixture combina `head1x1.active=true` com múltiplas camadas — verificado
individualmente em todos os fixtures A2/FiLM/gated/blended/condition_dsp
committed), então **não há regressão de cobertura existente**, apenas um
bloqueador latente para o trabalho futuro.

**Proposta de solução:**

1. Reestruturar o laço de leitura de pesos em `a2.rs` para ler `head1x1` (e,
   se a reconciliação confirmar, o `head` final) **dentro** do laço por-camada
   (`for li in 0..num_layers`), na mesma ordem que a correção de produção do
   Bug A adotar (Épico 2 do `TODO-wavenet_a2_max.md`) — os dois devem ser
   corrigidos em conjunto/lockstep para permanecerem comparáveis, idealmente
   compartilhando a mesma função utilitária de cálculo de contagem de pesos
   entre produção e oráculo, eliminando a duplicação de lógica que permitiu
   às duas implementações divergirem da mesma fonte de verdade C++ sem que
   nenhum teste notasse.
2. Corrigir a condição de formato do head final para se basear na presença
   real do objeto aninhado `cfg.head` (ou de `head_kernel_size` explícito),
   não em `head_size == 1` — replicando a correção do Bug C na produção.
3. **Adicionar um gate de reconciliação de orçamento de pesos, automatizado e
   não-ignorado**, ao final da leitura: `assert_eq!(cursor.pos,
   model_data.weights.len(), "resíduo de pesos não consumidos/lidos em
   excesso para {model_filename}")`. Isso transforma a verificação manual
   feita nesta auditoria (e em `TODO-wavenet_a2_max.md` §3.5) em uma
   salvaguarda permanente e automática — teria detectado esta regressão
   imediatamente, sem depender de rodar manualmente um teste `#[ignore]`d.
4. Após a correção, restaurar a execução do teste
   `test_oracle_vs_python_anchor_a2_generic` (hoje ainda `#[ignore]`, o que é
   correto enquanto quebrado) e confirmar que ele agora produz um número de
   ESR (não um panic) — mesmo que esse número ainda esteja fora do gate
   (`< 1e-12`) até que os Bugs A/B/C de produção também sejam corrigidos.

**Arquivos:** `src/testing/reference_oracle/a2.rs` (280-547, especialmente
468-523), `src/testing/reference_oracle/mod.rs:177-185` (`Cursor::read_f64`,
local do panic), `tests/parity/reference_oracle_f64.rs:1049-1070`
(`test_oracle_vs_python_anchor_a2_generic`), `tests/fixtures/models/wavenet_a2_max.nam`.

**Nota de processo:** mesma lição do Achado F11 — S2.T03 foi marcado `[x]` com
base em revisão de código/fórmulas, mas **sem executar o teste `#[ignore]`d
que a própria correção deveria habilitar a medir corretamente**. Tarefas que
"corrigem" um caminho de código coberto apenas por testes `#[ignore]`d devem
incluir, como critério de aceite explícito, a execução manual
(`--ignored`) desse(s) teste(s) antes de marcar a tarefa como concluída.

---

## Épicos (agrupamento para planejamento — gerar `TODO-sprints.md` somente quando solicitado)

### Épico A — Confiabilidade do harness de paridade C++ 🔴 [DONE — REABERTO PARCIALMENTE, ver F11]

**Achados:** F1, F9, F10.3. **Risco:** médio (mexe em testes, não em produção).

* A.1 Guard RAII simétrico de `ActivationPrecision` em `tests/common/` (F9).
  ⚠ **Implementação parcial** — 3 call-sites em `activation_precision.rs`
  seguem sem o guard e já causaram flakiness reproduzida. Ver **Épico G.1**
  (F11) para o fechamento definitivo.
* A.2 Redefinir semântica HF/não-HF dos `quick_parity_*` (fixar `Fast` no
  não-HF com caps calibrados p/ Padé, ou remover os duplicados HF) (F1).
* A.3 Corrigir comentário `cpp_parity.rs:1228-1241` e o cap `*5.0` do WaveNet
  HF (F1.1, F1.2).
* A.4 Silenciar warnings do build C++ vendorizado e pré-buildar o render fora
  do corpo dos testes (F5 follow-up, F10.3).

### Épico B — Metrologia MR-STFT 🟠 [DONE]

**Achados:** F2, F7. **Risco:** alto se mal feito (muda métrica de gate) —
exige regeração do golden Python em lockstep e recalibração documentada.

* B.1 Condicionamento da métrica (máscara de piso relativo ao pico) +
  regeração de `mrstft_golden.bin` + revalidação `test_mr_stft_parity_with_python` (F2).
* B.2 Calibrar/cobrir `MRSTFT_SOFT_THRESHOLD` no `threshold_calibration.rs`;
  mapear goldens Linear FFT na meta-cobertura (F2.1, F2.2).
* B.3 Rótulo `(relative)` → `(log-mag abs)`; documentar/unificar expoente de
  relaxação (F2.3, F2.4).
* B.4 Modo silencioso do harness para meta-testes (F7).

### Épico C — Metodologia do oráculo f64 🟠 [DONE — C.3 REABERTO, ver F12]

**Achados:** F3, F4. **Risco:** baixo (infra pareada já existe e é usada pelos
BossLSTM). Sinergia forte com Epics 5-6 do `TODO-wavenet_a2_max.md`.

* C.1 Decomposições WaveNet/LSTM/A2/ConvNet → `run_decomposition_paired()` (F3).
* C.2 `test_summary_table` pareada (ou fria-vs-fria simétrica rotulada) (F3).
* C.3 Corrigir oráculo `condition_dsp` (= Epic 6 do TODO-wavenet_a2_max) (F4).
  ⚠ **Corrigido apenas parcialmente** (commit `b7a8fb4`) — o oráculo A2 ainda
  **panica** (índice fora dos limites) para `wavenet_a2_max.nam`; a causa é
  estrutural (mesmo Bug A/C da produção, replicado no oráculo). Ver
  **Épico G.2** (F12).
* C.4 Atualizar mensagens `#[ignore]` e estender `meta_coherence` para
  `reference_oracle_f64.rs` (F4.1, F4.2).

### Épico D — Quality Dashboard 🟠 [DONE]

**Achados:** F6, F3 (lado parser), F10.5. **Risco:** baixo (só apresentação),
mas alto valor de confiança.

* D.1 Investigar/normalizar a linha `Container A2-Lite 0.00e0/inf` (F6.1).
* D.2 Separar proveniência cold/paired no `ESR_F64` (arrays distintos) (F3/F6.3).
* D.3 Seções ISA/Spectral: declarar não-cobertura no modo quick ou parsear o
  que existe (F6.2).
* D.4 Cobertura interop ConvNet (golden + cpp_parity) (F6.4).
* D.5 Cosmético: molduras do header; atualizar referência "Achado A1"→"F3" (F10.5).

### Épico E — Suítes de execução ✅ (parcial, follow-ups) [DONE]

**Achados:** F5 (corrigido). Restam: validação em uso contínuo da Fase 1 com
skips; decisão sobre `--test clap` na Fase 1; pré-build do render (→ A.4).

### Épico F — `wavenet_a2_max.nam` 🔴 (delegado) [ADIADO]

Plano completo em [`TODO-wavenet_a2_max.md`](TODO-wavenet_a2_max.md) (Epics 1–7),
com as adições registradas no Achado F8 (mensagens de ignore, meta_coherence,
tabela §4.6 do cpp_parity_map).

### Épico G — Residual pós-implementação (Sprints 1–5): fechar lacunas verificadas por execução 🔴 [NOVO — PLANEJADO]

**Achados:** F11, F12. **Origem:** auditoria de acompanhamento (2026-07-11)
que **executou** (não apenas revisou estaticamente) os artefatos dos Sprints
1–5 e encontrou duas lacunas concretas, ambas marcadas `[x]` prematuramente.
**Risco:** o G.1 é 🔴 alto valor/baixo esforço (a causa raiz da flakiness já
está isolada a 3 funções); o G.2 é 🟠 médio esforço (exige realinhar a leitura
de pesos do oráculo com a futura correção de produção do Bug A).

* **G.1 — Fechar o rollout do `PrecisionGuard` (F11):**
  * G.1.a Envolver os 3 call-sites desprotegidos de
    `tests/models/activation_precision.rs` (`test_zero_alloc_activation_switch_primitive`,
    `test_zero_alloc_activation_hot_path_switch`, `test_zero_alloc_cli_activation_flow`)
    com `PrecisionGuard::new(...)`, adquirido antes do `TrackingGuard`.
  * G.1.b Adicionar meta-teste estático que falha o build se
    `set_activation_precision(` aparecer em `tests/**/*.rs` (fora de
    `tests/common/precision.rs`) sem `PrecisionGuard::new` na mesma função —
    no estilo dos meta-testes de `threshold_calibration.rs`.
  * G.1.c Avaliar (e decidir formalmente, com nota no `docs/testing.md`) se
    `utils/tests-quick.sh` Fase 1 deve forçar `--test-threads=1` no binário
    `models` como defesa em profundidade, além do fix pontual em G.1.a.
  * G.1.d **Critério de aceite:** rodar `cargo test --release --test models`
    **≥ 10× consecutivas** sem nenhuma falha (hoje falha em ~2 de 4 execuções).
* **G.2 — Corrigir estruturalmente o oráculo A2 para `wavenet_a2_max.nam` (F12):**
  * G.2.a Reestruturar `src/testing/reference_oracle/a2.rs` para ler
    `head1x1` (e o `head` final, se a reconciliação confirmar) **dentro** do
    laço por-camada, em vez de uma única vez após todas as camadas —
    coordenar com a correção do Bug A na produção (Epic 2 do
    `TODO-wavenet_a2_max.md`) para que ambos compartilhem a mesma fonte de
    verdade sobre a ordem do stream de pesos do C++.
  * G.2.b Corrigir a condição do formato do head final para depender da
    presença real de `cfg.head` (não de `head_size == 1`) — mesma correção
    do Bug C, aplicada ao oráculo.
  * G.2.c Adicionar gate de reconciliação automático
    (`assert_eq!(cursor.pos, model_data.weights.len(), ...)`) ao final da
    leitura de pesos do oráculo A2 — não-ignorado, roda em qualquer teste que
    chame `oracle_a2_forward`.
  * G.2.d **Critério de aceite:** `cargo test --release --test parity
    reference_oracle_f64::test_oracle_vs_python_anchor_a2_generic -- --ignored
    --nocapture` produz um valor de ESR (não um panic) — o valor pode
    permanecer acima do gate até que os Bugs A/B/C de produção também sejam
    corrigidos, mas a medição em si deve ser possível.

---

**Ordem sugerida de ataque:** **G.1 primeiro** (baixo esforço, alto risco de
flakiness silenciosa se deixado para depois) → A/C/B/D (já concluídos, apenas
manter) → **G.2** (coordenar com o início do Épico F/Epic 2 do
`TODO-wavenet_a2_max.md`, já que ambos tocam a mesma estrutura de leitura de
pesos) → F (produção A2, plano próprio).
