<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Compliance & Paridade (Revisor-Auditor, 2026-07-13)

Achados da auditoria holística focada na role **Compliance and Parity Auditor**, com apoio das
skills `pesquisador-inovador` e `planejador-arquiteto`. Baseada em: análise integral de
`/testes.log` (run verde completo de 2026-07-12/13, quick + long + dashboard + contrato),
auditoria de paridade código↔NAMcore (`tests/fixtures/NeuralAmpModelerCore/`, commit
`1f42f885`, tag v0.5.4), auditoria da gestão de fixtures, auditoria dos scripts `utils/*` e
auditoria crítica do código Rust dos próprios testes.

**Fora de escopo (deliberado):** tudo que toca `wavenet_a2_max.nam` — tratado à parte em
`TODO-wavenet_a2_max.md`. Nenhum achado novo relevante sobre ele emergiu desta auditoria (a
constatação de que o upstream C++ só valida `isfinite` para esse modelo já está documentada lá,
§0.4).

**Veredito geral:** o NAM-rs está em forte conformidade matemática com o NAMcore para todos os
formatos de produção (ESR vs NAMcore entre 1e-11 e 1e-14 em todas as famílias; contrato de
qualidade 28/28 fidelidade + 10/10 performance OK). Os achados abaixo são melhorias de rigor,
robustez e eficiência da infraestrutura de qualidade — não regressões de áudio.

---

## Grupo P — Paridade estrita com NAMcore (código de produção)

### F-P1 · ConvNet aplica `head_scale` — divergência semântica com NAMcore — **MÉDIA**

* **Onde:** `src/models/convnet/model.rs:157,177,184` (`M::apply_gain(out_slice, self.head_scale)`).
* **Fato:** o ConvNet do NAMcore (`NAM/convnet.h/.cpp`) **não possui** campo `head_scale`; a
  saída é crua. O Rust aplica `head_scale` incondicionalmente ao output do ConvNet.
* **Impacto:** se algum `.nam` ConvNet vier com `head_scale ≠ 1.0` no config, o NAM-rs produz
  amplitude diferente do NAMcore. Hoje benigno (nenhum trainer conhecido exporta ConvNet com
  head_scale), mas é uma bomba-relógio de paridade.
* **Proposta:** alinhar à semântica C++: ignorar (ou rejeitar com diagnóstico) `head_scale` em
  ConvNet quando o arquivo é formato flat NAMcore; manter o campo apenas para a extensão
  post-stack-head própria do NAM-rs. Adicionar teste negativo (ConvNet com `head_scale=2.0` →
  saída idêntica ao NAMcore).

### F-P2 · Ausência de enforcement de versão do arquivo `.nam` — **MÉDIA**

* **Onde:** `src/loader/nam_json/` (o `parse_semver()` existe mas só alimenta a heurística A2 em
  `topology/wavenet.rs:130-141`).
* **Fato:** o NAMcore (`get_dsp.h/cpp`, `verify_config_version()`) rejeita versão `< 0.5.0`,
  aceita `[0.5.0, 0.7.0]` e sinaliza PARTIAL para `> 0.7.0`. O Rust aceita qualquer string de
  versão silenciosamente.
* **Impacto:** risco de forward-compatibility — mudanças futuras do formato NAM não seriam
  detectadas no load; o arquivo seria interpretado errado sem aviso.
* **Proposta:** implementar validação de faixa de versão espelhando o C++ (reject `<0.5.0`,
  telemetria/advisory para `>0.7.0`), com códigos de diagnóstico próprios. Testes: versões
  limítrofes (0.4.9, 0.5.0, 0.7.0, 0.7.1, 0.8.0, lixo não-semver).

### F-P3 · Sample rate desconhecido: default 48000 vs sentinela C++ `-1.0` — **BAIXA**

* **Fato:** com `sample_rate` ausente, o C++ usa sentinela `-1.0` → prewarm LSTM
  `max(1, 0.5*SR)` = 1 amostra. O Rust assume 48000 Hz → prewarm de 24000 amostras.
* **Impacto:** só afeta modelos degenerados sem `sample_rate` (nenhum em produção). O
  comportamento Rust é até "melhor" (prewarm real), mas é divergência documentável.
* **Proposta:** decidir e documentar a política (recomendado: manter 48000 como default
  explícito e registrar a divergência intencional em `docs/cpp_parity_map.md`); alternativa é
  espelhar o sentinela. Custo mínimo — decisão de compliance, não de código complexo.

### F-P4 · Lacunas menores de superfície de formato — **BAIXA**

* **(a) LSTM multi-canal:** C++ aceita in/out channels arbitrários; Rust rejeita
  (`src/loader/nam_json/topology/lstm.rs:27-38`). Nenhum `.nam` conhecido usa. Manter rejeição,
  mas trocar o erro por diagnóstico claro "unsupported multi-channel LSTM" (se ainda não for) e
  documentar como limitação assumida.
* **(b) Aliases do Linear:** C++ aceita `legacy`/`old`/`partitioned_fft`/`partitioned-fft`;
  Rust só `auto`/`direct`/`fft` (`models::tests::parse_linear_implementation_*`). Adicionar os
  aliases é trivial e elimina uma classe de arquivo válido-no-C++/inválido-no-Rust.
* **(c) `FastLUTActivation`:** otimização C++ não portada — não é feature de formato; nenhum
  gap de paridade de áudio. Registrar como "não aplicável" no mapa de paridade.

---

## Grupo X — Cadeia de suprimentos de fixtures e goldens

### F-X1 · Goldens C++ gerados com `-Ofast` — determinismo cross-compilador comprometido — **ALTA**

* **Onde:** `tests/fixtures/NeuralAmpModelerCore/tools/CMakeLists.txt:42,62,106`
  (`$<$<CONFIG:RELEASE>:-Ofast>`) + LTO (`INTERPROCEDURAL_OPTIMIZATION TRUE`). O
  `golden_gen_build.sh:200` passa apenas `-DCMAKE_CXX_FLAGS="-w"`, que **não** neutraliza
  `-Ofast` (target options têm precedência).
* **Fato:** `-Ofast` ≡ `-O3 -ffast-math` — viola IEEE 754 (reassociação, contração FMA,
  assume finitude). Os goldens `.bin` não são garantidamente reproduzíveis entre GCC/Clang ou
  entre versões do mesmo compilador. O README de fixtures já observou divergências
  "at floating-point precision levels (SNR > 120 dB)" entre gerações, sem apontar a causa raiz.
* **Impacto:** o contrato de reprodutibilidade dos goldens é probabilístico, não determinístico.
  Thresholds calibrados numa máquina podem flakear em outra (ver F-T1, que agrava isso).
* **Proposta:** no `golden_gen_build.sh`, sobrepor as opções do target com flags IEEE-strict na
  geração dos goldens (ex.: injetar `-fno-fast-math -ffp-contract=off` via
  `CMAKE_CXX_FLAGS_RELEASE` ou patch pontual do CMake na árvore vendorizada durante o build),
  regenerar goldens, medir o delta vs goldens atuais e recalibrar o que mudar. Registrar no
  manifest um *fingerprint* do toolchain (compiler + versão + flags) — ver F-I4.
* **Cautela:** regenerar goldens muda hashes do manifest e potencialmente todos os thresholds
  calibrados — executar como tarefa isolada, com diff de métricas antes/depois.

### F-X2 · Proveniência ausente para 3 modelos + 11 goldens fora da tabela do README — **ALTA**

* **Onde:** `tests/fixtures/README.md`.
* **Fato:** `wavenet_a2_film_chaos_stress.nam`, `wavenet_a2_film_input_mixin_pre.nam` e
  `wavenet_condition_lstm.nam` estão no CATALOG, no manifest e em testes ativos, mas têm **zero
  documentação de proveniência** no README. Outros 11 goldens em disco não constam da tabela
  principal (o README se declara "subset ilustrativo", delegando a `validation.rs`).
* **Proposta:** completar a tabela de goldens e as seções de proveniência (origem: sintético via
  `generate_a2_fixtures.py` / real / upstream example). Avaliar tornar o CATALOG do
  `golden_gen_build.sh` a única fonte de verdade e gerar a tabela do README a partir dele
  (script), eliminando o drift estrutural.

### F-X3 · Fixtures sem cobertura do freshness manifest — **MÉDIA**

* **Fato:** `.golden_manifest.sha256` cobre os 28 modelos do CATALOG (62 hashes + 80
  `# EXPECTED:`), mas **não cobre**: `golden_cabsim_cpp_{short,medium,long}.bin`,
  `mrstft_golden.bin` (gerado por `scripts/gen_mrstft_golden.py`), `resampler_*.f32`,
  `f64_anchors/`, `ebu_3341_*.wav`, `stress_signal*.wav`.
* **Impacto:** mudanças em `render_ir.cpp`, no script Python do MR-STFT ou nos âncoras f64 podem
  tornar goldens/âncoras obsoletos sem detecção automática.
* **Proposta:** estender o manifest (ou criar manifests dedicados) cobrindo esses artefatos,
  hasheando também o gerador (ex.: SHA do `render_ir.cpp` e do `gen_mrstft_golden.py`).

### F-X4 · Freshness gate: warn-only no long, sem reverse-check, implementação duplicada — **MÉDIA**

* **Fato:** (a) `utils/tests-long.sh:278-300` só avisa (não bloqueia) com goldens stale; (b) o
  gate do quick (`utils/tests-quick.sh:96-152`) é unidirecional — um `.nam` novo em
  `tests/fixtures/models/` que nunca entrou no manifest passa em silêncio; (c) as duas
  implementações divergiram (o long não trata `# EXPECTED:`).
* **Proposta:** extrair `check_freshness()` para `utils/_lib.sh` com parâmetro
  hard-fail/warn-only; elevar o long para hard-fail (bypass explícito via env var); adicionar o
  reverse-check (arquivos `.nam` em disco ∖ manifest → erro).

### F-X5 · Higiene menor de catálogo — **BAIXA**

* `tests/models/meta_coherence.rs:29-45`: `CATALOG_EXCEPTIONS` lista `linear_fft_rf{2048,4096,8192}.nam`
  como "golden pipeline not yet implemented", mas eles já estão no CATALOG — remover as 3
  entradas stale.
* `BossWN-lite.nam` obsoleto ainda em `models/` — remover ou mover para `models/obsolete/`.
* `f64_anchors/` sem seção própria no README — documentar propósito e regeneração.

---

## Grupo T — Rigor do código de testes (calibrações e avaliadores)

### F-T1 · Gate MR-STFT 0.12 a 91% de utilização no teste de impulso — risco real de flake — **ALTA**

* **Onde:** `tests/common/validation.rs:856-857` (`mrstft_max = 0.12` p/ `linear_fft_rf*`) e
  `tests/models/linear_fft_test.rs:230` (`test_fft_golden_rf4096_impulse_response`, medido
  `MR-STFT = 1.0898e-1`).
* **Fato:** o threshold 0.12 foi calibrado com sinais multi-frequência (MR-STFT ≈ 2e-6); o caso
  de impulso produz ≈ 0.109 por natureza espectral distinta (leakage de borda do impulso), e é
  avaliado contra o mesmo gate — sobra só 9% de margem. Combinado com F-X1 (goldens
  não-determinísticos), é o candidato nº 1 a flake futuro.
* **Proposta:** thresholds MR-STFT **por classe de sinal**: manter 0.12 para multi-freq/sine e
  criar gate dedicado calibrado (com comentário `// Measured:`) para o caso de impulso — ou
  substituir a métrica do caso de impulso por ESR/MaxErr (que já estão em 1e-14/4.2e-5),
  documentando por que MR-STFT é inadequado para impulsos.

### F-T2 · `run_v1` é alias 1:1 de `run_v1_hf` → testes duplicados no quick — **MÉDIA**

* **Onde:** `tests/parity/cpp_parity.rs:579-604`; consumidores `quick_parity_lstm_1x16` ×
  `quick_parity_hf_lstm_1x16` e `quick_parity_wavenet_ch16` × `quick_parity_hf_wavenet_ch16`
  (executados em `utils/tests-quick.sh:437-439`).
* **Fato:** desde que Standard virou default universal, `run_v1` delega a `run_v1_hf` — os pares
  produzem métricas **byte-idênticas** (confirmado no testes.log: 1.55e-12/108.4 dB duas vezes;
  6.93e-15/136.1 dB duas vezes). Cada duplicata custa um render C++ + inferência + métricas.
* **Impacto:** ~3–5 s desperdiçados por run do quick + saída de CI enganosa (dois testes
  "distintos" medindo a mesma coisa). Os caps Fast antigos (6.23e-3) viraram dead code no quick.
* **Proposta:** remover os wrappers não-HF duplicados (ou fundir nomes), limpar os caps
  Fast-mode mortos, e deixar 1 teste por (modelo, modo). Se a cobertura do modo Fast(Padé) for
  desejada, reintroduzi-la explicitamente com `use_hf=false` na long-suite — nunca como alias.

### F-T3 · Gates herdados "pre-fix" com margem absurda (smoke não intencional) — **MÉDIA**

* **Onde:** `tests/common/validation.rs:589-607` — `wavenet_official` com gate SNR ≥ 14.0 dB /
  ESR < 3.5e-2 vs medição atual SNR 130.5 dB / ESR 8.96e-14 (margem de ~116 dB).
* **Fato:** comentário admite "Thresholds preserved from pre-fix calibration" — a calibração
  nunca foi refeita após o fix do engine. O gate só detectaria falha catastrófica.
* **Proposta:** recalibrar com a política padrão do projeto (margem documentada, ex.: 3× o ESR
  medido / −15 dB de SNR) e atualizar o comentário `// Measured:`. Varredura nos demais gates em
  busca do mesmo padrão (margens > 40 dB sem justificativa explícita de smoke).

### F-T4 · Cap MR-STFT 1.20 para FiLM v1 fura o teto anti-placebo — **MÉDIA**

* **Onde:** `tests/parity/cpp_parity.rs:431` (`ABSOLUTE_MRSTFT_CAP_FILM = 1.20`).
* **Fato:** o meta-teste anti-placebo rejeita `MR-STFT ≥ 0.5` nos goldens, mas o cap live de
  FiLM v1 é 1.20 — o gate MR-STFT fica efetivamente desligado nesse caminho (justificado pelo
  fallback C++ para WaveNet genérico Eigen, mas as medições reais estão em 7e-6…1.7e-5).
* **Proposta:** rebaixar o cap v1 FiLM para valor calibrado (ex.: 3e-2, coerente com
  `FILM_V2_MRSTFT_FLOOR`), com comentário de medição. Estender o meta-teste anti-placebo para
  cobrir também os caps do `cpp_parity.rs` (hoje só cobre `validation.rs`).

### F-T5 · Cobertura morta/adormecida: `condition_lstm` golden e goldens v2 — **MÉDIA**

* **(a)** `tests/models/golden_vectors.rs:1128-1175`: `test_golden_vectors_wavenet_condition_lstm`
  é `#[ignore]` permanente — o C++ upstream não consegue gerar o golden (channel mismatch). O
  corpo do teste é código morto; só existe o smoke (`loads_and_runs`, verifica finitude).
  **Proposta:** validar esse modelo contra o oráculo f64 (âncora NumPy) em vez do golden C++, ou
  documentar formalmente a exceção no README de fixtures + mapa de paridade.
* **(b)** Todos os 18+ testes `*_v2_*` são `#[ignore]` (sinais de 5 s); só rodam na long-suite
  via cpp_parity v2. **Proposta:** promover 1 teste v2 representativo (ex.: WaveNet Standard
  @48k) para o conjunto não-ignored do release do quick (~10 s em release), garantindo que o
  formato v2 nunca fique 100% dependente de opt-in.

### F-T6 · Precisão dos avaliadores de métrica (documentação e magic numbers) — **BAIXA**

* Verificado e **correto**: coeficientes K-weighting BS.1770-4, gating 400 ms/75%, gates −70
  LUFS/−10 LU, FIR true-peak 4×48-taps, definição de ESR, parser WAV endurecido. Bom estado.
* Pendências menores: (a) janela Hann do MR-STFT é a variante **simétrica** (`ws-1`) —
  documentar no docstring (`src/testing/perceptual.rs:166`); (b) threshold de pico do ASR usa
  fator 6× sem citação (`src/testing/aliasing.rs:172`) — citar a fonte (Sato & Smith, DAFx) ou
  derivar; (c) tolerância harmônica de 1.5 bin do ASR pode ser proporcional a f0 em vez de
  absoluta (`aliasing.rs:185`); (d) THD total do Farina recomputa de percentuais por ordem em
  vez de energias cruas (`src/testing/spectral.rs:334-339`).

### F-T7 · Loopholes nos meta-testes de calibração — **BAIXA**

* **(a)** `test_all_thresholds_anti_placebo` (`threshold_calibration.rs:266-345`) rejeita SNR ≤ 0
  e ESR ≥ 1.0 **independentemente**, mas uma entrada `SNR = 1 dB, ESR = 0.99` ainda passa —
  adicionar piso conjunto mínimo (ex.: SNR ≥ 10 dB **ou** ESR < 0.5, senão falha).
* **(b)** `test_all_calibrated_entries_have_measurement_comments` valida **presença** do
  comentário `// Measured:`, não a **acurácia** dos números; e usa lista de modelos hardcoded —
  derivar a lista dos `.bin` em disco e (opcional) validar que o valor medido comentado é ≥ que
  o gate/10 (sanidade anti-fabricação).
* **(c)** `test_no_silent_let_underscore_in_cpp_parity_wrappers` não pega variantes
  (`let _x = ...` sem uso, `drop(...)`) — hoje o código está limpo; endurecer o padrão regex é
  barato.

---

## Grupo S — Scripts de QA (`utils/*`) — correção e velocidade

### F-S1 · `rt_constraints` compila e executa 0 testes no quick — **ALTA (eficiência)**

* **Onde:** `utils/tests-quick.sh:326` (`_struct_targets="$_struct_targets rt_constraints"`) +
  `:336` (`--skip rt_deadline:: --skip rt_jitter::`).
* **Fato:** todos os testes desse binário são feature-gated (`heap-audit`, nunca ativada no
  quick) ou skipados pelo filtro — resultado: "running 0 tests; 18 filtered out". Compilação e
  link em debug (~10–15 s) por nada, a cada run.
* **Proposta:** remover `rt_constraints` de `_struct_targets` no quick (a cobertura real desse
  binário pertence à long-suite, fases 4 e 7). Junto com F-T2, economiza ~15–25 s por run do
  quick — exatamente a missão dele ("rápido, para rodar sempre").

### F-S2 · Dashboard: aritmética awk sem `LC_ALL=C` — corrupção silenciosa em locale pt_BR — **ALTA**

* **Onde:** `utils/quality-dashboard.sh:257,279-280,294-295,309-310,324-325,338,1262`.
* **Fato:** as durações são formatadas com vírgula decimal ("0,4s" visível no testes.log linha
  2518) e depois **reutilizadas em somas** awk — em mawk/nawk "0,4" vira 0, gerando totais
  errados sem nenhum erro visível.
* **Proposta:** prefixar `LC_ALL=C` em **todas** as invocações awk de formatação/aritmética
  numérica (não só as de parsing). Adicionar um smoke de locale no início do script
  (`export LC_ALL=C` global é a solução mais simples e robusta).

### F-S3 · "ConvNet vs NAMcore: N/A" no dashboard apesar de medição real existir — **MÉDIA**

* **Onde:** `utils/quality-dashboard.sh:404-407` (parser awk hardcoda `N/A` para
  `[ConvNet Self-Golden…]`).
* **Fato:** `quick_parity_convnet` (cpp_parity) mede ESR real vs NAMcore (2.54e-5, SNR 45.9 dB),
  mas sua saída não é capturada em nenhum log que o dashboard leia — o painel exibe N/A para uma
  métrica que existe. Nota: essa medição também expõe que a paridade do ConvNet (45.9 dB) é
  ordens de magnitude mais frouxa que as demais famílias (>100 dB) — vale investigação própria
  (threshold ConvNet 9.5e-5 é o mais largo da suíte v1).
* **Proposta:** (a) capturar a saída dos `quick_parity_*` num log ingerível pelo dashboard (ou
  migrar para F-I1/JSON); (b) abrir investigação da lacuna de precisão ConvNet vs NAMcore
  (possível diferença de ordem de operações no batchnorm/head — confirmar com oráculo f64, que
  mostra ConvNet a −144.5 dB do ideal, ou seja, o gap está na *comparação com o C++*, não na
  qualidade do Rust).

### F-S4 · Dashboard: "vs f64: N/A" para A2-FiLM-InputMixinPre — **MÉDIA**

* **Onde:** `tests/parity/reference_oracle_f64.rs:1187-1194` (`test_summary_table`) — o vetor de
  modelos não inclui `wavenet_a2_film_input_mixin_pre.nam`, embora o teste individual
  (`test_oracle_a2_film_input_mixin_pre`) meça 2.21e-14. O `_lookup_esr_f64` do dashboard busca
  pelo filename no summary table e não acha.
* **Proposta:** adicionar `("wavenet_a2_film_input_mixin_pre.nam", "A2-FiLM-IMPre")` ao summary
  table. Complementarmente, ver F-I2 (eliminar o conceito de família ~fam.).

### F-S5 · tests-long: aviso "falso-verde" disparado para fase legitimamente SKIPPED — **MÉDIA**

* **Onde:** `utils/tests-long.sh:428-430` (check `< 1s`) vs auto-detecção PipeWire
  (`:460-472`, exit 77 = SKIPPED).
* **Fato:** sem daemon PipeWire, a fase é corretamente pulada em <1 s, mas o resumo imprime
  "PASSED" + aviso de falso-verde — semanticamente errado nos dois pontos (deveria exibir
  SKIPPED e não avisar).
* **Proposta:** propagar o status 77 até o AUDIT SUMMARY (exibir `SKIPPED`) e suprimir o aviso
  de <1 s quando `status == 77`.

### F-S6 · Tolerâncias do contrato não documentadas no próprio contrato — **BAIXA**

* **Fato:** o `--check` usa tolerância de latência de +10% (`quality-dashboard.sh:1398-1510`,
  correta e verificada: A2 Full 28.85 µs < 27.7×1.10 = 30.47 µs) e tolerâncias de fidelidade
  próprias, mas `docs/quality-contract.txt` só lista valores absolutos — um leitor assume match
  exato.
* **Proposta:** adicionar cabeçalho no `quality-contract.txt` documentando as tolerâncias
  (+10% latência; regra usada para ESR) e a política de atualização do contrato.

### F-S7 · Custo de compilação do quick — oportunidades estruturais — **BAIXA**

* **Fato:** o passo [2/3] recompila todo o perfil release (3m04s frio; com cache quente vira
  segundos — o log analisado era cold). A dependência `pipewire` (feature `standalone`, default)
  é compilada em todos os passos sem que nenhum teste do quick a exercite (~5–10 s por perfil).
* **Proposta:** (a) avaliar feature `testing-lite` sem `pipewire` para os passos de teste;
  (b) garantir que os 3 passos usem RUSTFLAGS/features idênticos por perfil para nunca invalidar
  o cache entre si; (c) documentar em `docs/testing.md` o tempo esperado quente vs frio.

---

## Grupo I — Pesquisador-Inovador (além do óbvio, dentro do escopo, toolchain stable)

### F-I1 · Métricas machine-readable (JSON Lines) — eliminar scraping awk de texto humano — **ALTA (robustez)**

* **Motivação:** F-S2/F-S3/F-S4 são todos sintomas da mesma causa raiz: o dashboard faz parsing
  awk de saída textual feita para humanos. Isso é frágil por construção (locale, mudanças
  cosméticas de label, truncamento de nomes com `printf %.38s`).
* **Proposta:** o helper `report_dsp_fidelity*` (tests/common) passa a emitir, além do bloco
  humano, **uma linha JSON** por medição (`{"kind":"fidelity","label":…,"esr":…,"snr_db":…,
  "mrstft":…,"model":"file.nam","mode":"hf"}`) — opcional via env `NAM_METRICS_JSONL=path`. O
  dashboard lê o JSONL com `jq` (ou awk trivial por linha) e os parsers awk atuais viram
  fallback. Zero dependências novas no crate (formatação manual de JSON é suficiente e
  determinística). Benefício: dashboard 100% determinístico, testável e independente de locale.

### F-I2 · ESR vs f64 por modelo — aposentar a aproximação por família (`~fam.`) — **MÉDIA**

* **Motivação:** o dashboard exibe "vs Ideal (f64)" de uma fixture representativa por família,
  marcada `~fam.` — enganoso (ex.: LSTM 1x16 mostra 2.71e-12 da família, quando a decomposição
  cold do próprio modelo indica ordens diferentes). O oráculo f64 já existe e é genérico.
* **Proposta:** estender `test_summary_table` para iterar **todos** os modelos golden com
  oráculo disponível (custo: segundos em release), preenchendo a coluna por modelo e removendo
  `ESR_F64_FAMILY_MAP` e a legenda `~fam.` do dashboard. Modelos sem oráculo exibem N/A honesto.

### F-I3 · Modo Fast(Padé) em LSTM: erro dominante e audível — reposicionar o produto — **MÉDIA**

* **Fato (do log):** decomposição f64 mostra que a ativação Padé domina o erro do LSTM
  (BossLSTM-1x16: ΔESR Padé = 4.81e-2 ≈ −13 dB; painel Activation Precision: Fast 15.9–29.3 dB
  vs Standard 103–120 dB, ganho médio +89.5 dB). Com Standard sendo o default universal e os
  benchmarks mostrando LSTM a 0.6% do budget, o modo Fast em LSTM tem custo-benefício negativo:
  economiza CPU irrelevante e degrada para nível claramente audível.
* **Proposta:** (a) marcar Fast como *não recomendado para LSTM* na CLI/docs (advisory no load
  quando `--activation fast` + LSTM); (b) avaliar aposentar o caminho Fast para sigmoid/tanh do
  LSTM (mantendo-o para WaveNet, onde ΔESR Padé ≈ −134 dB é inócuo); (c) se mantido, cobrir com
  gate perceptual explícito (hoje os caps Fast são dead code — ver F-T2).

### F-I4 · Fingerprint de toolchain no manifest de goldens — **MÉDIA**

* **Motivação:** complementa F-X1. Mesmo com flags IEEE-strict, goldens dependem de
  compilador/stdlib (`std::tanh` do glibc pode variar entre versões).
* **Proposta:** `golden_gen_build.sh` grava no cabeçalho do `.golden_manifest.sha256`:
  `# TOOLCHAIN: g++ 14.2.0 | glibc 2.40 | flags: -O3 -fno-fast-math -ffp-contract=off | cmake 3.30`.
  O gate de freshness exibe (não bloqueia) divergência de fingerprint, transformando o
  "misterioso SNR>120 dB drift" do README em evento diagnosticável.

---

## Épicos (agrupamento para execução segura e eficiente)

> Referenciar sempre o ID do achado. Ordem sugerida = ordem dos épicos. Só criar
> `TODO-sprints.md` quando solicitado.

### EP1 — Quick suite enxuto e honesto (baixo risco, ganho imediato) [DONE]

Achados: **F-S1**, **F-T2**, **F-S5**.
Remover `rt_constraints` do quick; deduplicar `quick_parity_*` HF/não-HF e limpar caps Fast
mortos; corrigir status SKIPPED no long. Critério de aceite: quick ≥ 15 s mais rápido (quente),
mesma cobertura efetiva, AUDIT SUMMARY exibe SKIPPED corretamente.

### EP2 — Dashboard correto e determinístico [DONE]

Achados: **F-S2**, **F-S4**, **F-S3**, **F-I1**, **F-I2**, **F-S6**.
Sequência: LC_ALL=C (hotfix) → entrada InputMixinPre no summary table → emissão JSONL +
migração do parser → coluna f64 por modelo (aposentar `~fam.`) → ingestão do quick_parity
(resolve o N/A do ConvNet) → documentar tolerâncias no contrato. Critério: dashboard sem N/A
espúrios, imune a locale, contrato com tolerâncias explícitas.

### EP3 — Recalibração de gates (rigor sem flakiness) [DONE]

Achados: **F-T1** (crítico), **F-T3**, **F-T4**, **F-T7**.
Gate MR-STFT por classe de sinal (impulso); recalibrar `wavenet_official`; rebaixar cap FiLM
v1; endurecer meta-testes anti-placebo. Critério: nenhum gate com utilização > 50% sem
justificativa `// Measured:`; nenhum cap acima do teto anti-placebo; varredura completa de
margens documentada.

### EP4 — Cadeia de suprimentos de fixtures determinística [DONE]

Achados: **F-X1** (crítico, executar isolado), **F-I4**, **F-X3**, **F-X4**, **F-X2**, **F-X5**.
Flags IEEE-strict na geração + fingerprint de toolchain + regeneração/recalibração controlada;
depois cobertura do manifest (cabsim/mrstft/resampler/âncoras), unificação do gate de freshness
em `_lib.sh` com reverse-check e hard-fail no long; por fim proveniência completa no README.
**Risco alto controlado:** a regeneração muda hashes/thresholds — exige diff de métricas
antes/depois e um commit atômico dedicado.

### EP5 — Paridade estrita com NAMcore (produção) [DOING]

Achados: **F-P1**, **F-P2**, **F-P3**, **F-P4**.
ConvNet `head_scale` alinhado ao C++; enforcement de versão de arquivo; política documentada de
sample-rate ausente; aliases do Linear + diagnóstico LSTM multi-canal. Critério: matriz de
paridade em `docs/cpp_parity_map.md` sem entradas DIVERGENT não-intencionais; novos testes
negativos cobrindo cada caso.

### EP6 — Cobertura adormecida e posicionamento do modo Fast

Achados: **F-T5**, **F-I3**, **F-T6**.
Promover 1 golden v2 ao quick; validar `condition_lstm` via oráculo f64; advisory/deprecação do
Fast em LSTM; documentar variante Hann e constantes do ASR/Farina. Critério: nenhum formato
(v2) 100% dependente de opt-in; decisão registrada sobre o futuro do Fast em LSTM.
