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

## Épicos (agrupamento para planejamento — gerar `TODO-sprints.md` somente quando solicitado)

### Épico A — Confiabilidade do harness de paridade C++ 🔴 [DONE]

**Achados:** F1, F9, F10.3. **Risco:** médio (mexe em testes, não em produção).

* A.1 Guard RAII simétrico de `ActivationPrecision` em `tests/common/` (F9).
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

### Épico C — Metodologia do oráculo f64 🟠 [DONE]

**Achados:** F3, F4. **Risco:** baixo (infra pareada já existe e é usada pelos
BossLSTM). Sinergia forte com Epics 5-6 do `TODO-wavenet_a2_max.md`.

* C.1 Decomposições WaveNet/LSTM/A2/ConvNet → `run_decomposition_paired()` (F3).
* C.2 `test_summary_table` pareada (ou fria-vs-fria simétrica rotulada) (F3).
* C.3 Corrigir oráculo `condition_dsp` (= Epic 6 do TODO-wavenet_a2_max) (F4).
* C.4 Atualizar mensagens `#[ignore]` e estender `meta_coherence` para
  `reference_oracle_f64.rs` (F4.1, F4.2).

### Épico D — Quality Dashboard 🟠 [PLANEJADO]

**Achados:** F6, F3 (lado parser), F10.5. **Risco:** baixo (só apresentação),
mas alto valor de confiança.

* D.1 Investigar/normalizar a linha `Container A2-Lite 0.00e0/inf` (F6.1).
* D.2 Separar proveniência cold/paired no `ESR_F64` (arrays distintos) (F3/F6.3).
* D.3 Seções ISA/Spectral: declarar não-cobertura no modo quick ou parsear o
  que existe (F6.2).
* D.4 Cobertura interop ConvNet (golden + cpp_parity) (F6.4).
* D.5 Cosmético: molduras do header; atualizar referência "Achado A1"→"F3" (F10.5).

### Épico E — Suítes de execução ✅ (parcial, follow-ups) [PLANEJADO]

**Achados:** F5 (corrigido). Restam: validação em uso contínuo da Fase 1 com
skips; decisão sobre `--test clap` na Fase 1; pré-build do render (→ A.4).

### Épico F — `wavenet_a2_max.nam` 🔴 (delegado) [ADIADO]

Plano completo em [`TODO-wavenet_a2_max.md`](TODO-wavenet_a2_max.md) (Epics 1–7),
com as adições registradas no Achado F8 (mensagens de ignore, meta_coherence,
tabela §4.6 do cpp_parity_map).

---

**Ordem sugerida de ataque:** A (rápido, destrava confiança nos gates) → C
(metodologia, prepara F) → B (métrica, exige mais cautela) → D (apresentação)
→ F (produção A2, plano próprio). O Épico E já está essencialmente fechado.
