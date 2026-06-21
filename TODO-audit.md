<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-audit.md — Auditoria Pós-Paridade NAM-rs ↔ NAMcore

> **Skill:** `revisor-auditor` → `planejador-arquiteto`
> **Data:** 2026-06-20
> **Referência canônica:** NeuralAmpModelerCore v0.5.3 (`9c7b185`), espelhado em `tests/fixtures/NeuralAmpModelerCore/`
> **Escopo:** revisão de auditoria geral após conclusão dos épicos/sprints de paridade. Foco em
> correção sonora, cobertura de testes/benches/goldens, cadeia de suprimentos de validação,
> mecanismo de modelos não-distribuíveis e caça a bugs de segurança/funcionalidade.
>
> **Esta entrega contém apenas Achados (Fn) + Épicos.** Sprints e tarefas técnicas serão
> destrinchadas à parte, ao iniciarmos cada épico (skill `tarefa`/`planejador-arquiteto`).

---

## 0. Sumário Executivo

### Veredito de saúde (estado atual)

- **Compilação:** `cargo check --lib` limpo (default features). ✅
- **Suíte rápida (`cargo test`):** **761 passaram · 0 falharam · 93 ignorados** (os `#[ignore]` da suíte longa). ✅
- **Paridade estrutural vs NAMcore:** fiel. WaveNet (estático + dinâmico), LSTM (estático + dinâmico),
  A2 (fixo CH=3/8 + dinâmico), ConvNet, Linear e Container correspondem 1:1 às classes C++
  (`NAM/wavenet/model.cpp`, `a2_fast.cpp`, `lstm.cpp`, `convnet.cpp`, `container.cpp`).
- **`is_a2_shape()`:** replica os ~20 critérios de `a2_fast.cpp:754-885` quase 1:1 (uma exceção — ver F5).
- **RT-safety:** heap-audits de zero-alloc passam inclusive para os paths novos
  (`test_zero_alloc_process_wavenet_dynamic`, `test_zero_alloc_container_transition`,
  `test_zero_alloc_nondist_models`). ✅

### Narrativa central da auditoria

> **A implementação avançou além do que a documentação e a cadeia de validação registram.**

Os épicos recentes de paridade reintroduziram/criaram os **engines dinâmicos** (`WaveNetModelDyn`,
`LstmModelDyn`, `WaveNetA2Dyn`) e a arquitetura **ConvNet**, todos *sempre compilados* e *alcançáveis
por padrão* via dispatcher. Com isso o NAM-rs hoje **carrega uma fração muito maior do universo de
modelos do NAMcore** (geometrias livres, condição/condition_dsp, post-stack head, gated/blended, FiLM
em A2). Porém:

1. **A documentação afirma o oposto** (que esses paths foram "removidos"/"rejeitados") — F1, F2.
2. **A malha de validação (goldens externos, soak, bench, PGO) não acompanhou** os novos paths —
   F3, F4, F5, F8.
3. Restam **higiene de testes** (placebos `is_finite()`-only e dead tests — F6, F7), **robustez
   RT/parsing** (F11–F14) e **reprodutibilidade da cadeia de testes/nondist** (F9, F10).

Nenhum bug de correção *ativo* foi confirmado na suíte atual, mas há **lacunas de cobertura que
tornam regressões de inferência invisíveis** nos paths mais novos — exatamente os que mais precisam
de rede de segurança. Este é o foco do Épico mais crítico (B).

---

## 1. Achados (Findings)

Legenda de severidade: 🔴 Alta · 🟠 Média · 🟡 Baixa/Quick-win · 🟢 Ponto forte.

---

### F1 🔴 — Documentação de paridade desatualizada: engines dinâmicos + ConvNet são tratados como "removidos", mas estão ativos por padrão

**Evidência:**

- `src/models/mod.rs:77-122` — o enum `StaticModel` expõe `WavenetDyn`, `WavenetA2Dyn`, `LstmDyn`,
  `Lstm1x3` e `ConvNet`, todos `Box`ados e despachados.
- `docs/cpp_parity_map.md` §3.3 ("Legacy Dynamic WaveNet (removed)") e §9 ("WaveNet Dyn (removed —
  Sprint 1.5)", "LSTM Dyn (removed — Sprint 1.5)") afirmam remoção.
- `docs/cpp_parity_map.md` §10.1 afirma que `condition_size ≠ 1` e `head` (não-nulo) são "rejeitados
  no load". Na prática, `loader/nam_json/topology.rs` (`get_wavenet_topology` → `Free`) **captura**
  `condition_size` e `post_stack_head` e os roteia para `WaveNetModelDyn` (construído por
  `loader/dispatcher/wavenet/dynamic.rs::build_wavenet_dynamic`, sem feature-gate).
- A feature `dynamic-engine` (`Cargo.toml:70`) **não** gateia esses tipos; ela só controla um ramo
  escalar interno do `WaveNetA2<CH>` fixo (ver F11). Os engines dinâmicos compilam sempre.
- `ConvNet` não aparece em nenhuma tabela do `cpp_parity_map.md` apesar de existir
  (`src/models/convnet/`, ~1383 LOC) e ter correspondente C++ (`NAM/convnet.{h,cpp}`).

**Impacto:** documentação enganosa para mantenedores/auditores e para qualquer decisão de escopo
("o que o engine aceita?"). Quebra o contrato da skill `documentador` (conhecimento sincronizado com
a implementação). Confunde a própria definição de "paridade" do projeto.

**Caminho de resolução:**

1. Reescrever `cpp_parity_map.md` §3.3/§9/§10.1: remover linguagem de "removido/rejeitado"; adicionar
   linhas para `WaveNetModelDyn` (free geometry, condition_dsp, post-stack head), `LstmModelDyn`,
   `WaveNetA2Dyn` (superset do `a2_fast.cpp`) e `ConvNet` (↔ `convnet.cpp`).
2. Atualizar `docs/architecture.md` (seções de dispatch, A2, testing tables) e `docs/testing.md`
   (matriz de cobertura) para citar os módulos/tests dinâmicos e ConvNet.
3. Documentar explicitamente a função do flag `dynamic-engine` (escopo real = ramo escalar A2 fixo).

---

### F2 🟠 — Contradição documental sobre a natureza dos goldens A2 (real C++ vs self-golden)

**Evidência:**

- `tests/fixtures/README.md` (§`wavenet_a2_full`/`lite`): "cross-reference Rust↔C++ rendered using
  canonical commit `9c7b185`", com **SNR/ESR medidos** (Full 79.2 dB / 1.21e-8; Lite 90.7 dB / 8.58e-10).
- `tests/fixtures/golden_gen_build.sh:154-160` compila o `render` com `-DNAM_ENABLE_A2_FAST=ON` e
  renderiza `wavenet_a2_full/lite` pelo binário C++ → goldens **reais**.
- `tests/threshold_calibration.rs` exige entradas calibradas com SNR≥70 dB (anti-placebo) para A2.
- **Porém** `docs/cpp_parity_map.md` §5 ainda diz: "C++ Live Cross-Validation Blocked (Upstream Bug)
  … self-golden pattern (Rust validates Rust)".

**Impacto:** mensagem contraditória sobre quão confiável é a paridade A2. A evidência indica que §5
está **desatualizado** (os goldens A2 já são C++ reais desde T2.5/T2.6); manter o aviso antigo
mina a confiança real conquistada.

**Caminho de resolução:** confirmar empiricamente (rodar `live_cross_validation_wavenet_a2_*` na suíte
longa — pedir execução humana, pois IA não roda `tests-long.sh`) e reconciliar §5 do `cpp_parity_map.md`
com o `README.md`. Se a instabilidade upstream do `a2_fast.cpp` ainda afeta SR≠48k, documentar o
escopo exato (quais SRs são confiáveis) em vez de um "blocked" genérico.

---

### F3 🔴 — ConvNet não possui golden de paridade externo (regressão de inferência seria invisível)

**Evidência:**

- `ConvNet` é arquitetura first-class: `StaticModel::ConvNet`, dispatcher
  `loader/dispatcher/convnet/mod.rs`, modelo `src/models/convnet/{model,block,batch_norm}.rs`.
- NAMcore **suporta** ConvNet: `tests/fixtures/NeuralAmpModelerCore/NAM/convnet.{cpp,h}` presentes.
- **Cobertura atual:** apenas testes unit inline (`block.rs`, `batch_norm.rs`, `model.rs` — `mod tests`).
  Verificado: **nenhum** `golden_convnet*.bin`; ConvNet **ausente** de `golden_gen_build.sh`,
  `tests/golden_vectors.rs` (0 ocorrências), `tests/cpp_parity.rs` (0 ocorrências), `tests/soak_test.rs`
  e `tests/threshold_calibration.rs`.

**Impacto:** uma regressão numérica em Conv1D/BatchNorm/head do ConvNet não seria detectada por
nenhum gate externo. Viola o princípio "toda seção que deveria ser verificada regularmente tem teste".

**Caminho de resolução:**

1. Obter/gerar um `.nam` ConvNet (sintético calibrado em regime de áudio realista, à la A2; ou um modelo
   ConvNet oficial se existir em `example_models/`).
2. Estender `golden_gen_build.sh` (lista `MODELS`/`V2_MODELS`) para renderizar `golden_convnet*.bin`
   via `render` C++.
3. Adicionar `test_golden_vectors_convnet` em `tests/golden_vectors.rs` + entrada calibrada em
   `tests/common/validation.rs` (com comentário `// Measured:`) + `live_cross_validation_convnet`
   em `tests/cpp_parity.rs`.
4. Adicionar soak (silêncio + ruído) e fixture ao `nondist_validation` quando aplicável.

---

### F4 🔴 — Engines dinâmicos sem âncora externa, soak ou benchmark

**Evidência:**

- `LstmModelDyn`: cobertura é só **paridade interna SIMD↔scalar** (`tests/lstm_model_dyn_validation.rs`,
  rel < 5e-3) e determinismo/block-invariance. **Não há golden C++** para nenhuma geometria LSTM
  não-catalogada (ex.: 1×7, 3×8) — o oracle externo está ausente justamente no path catch-all.
- `WaveNetModelDyn`: parcialmente ancorado (`golden_wavenet_official` CH=3 free, `golden_wavenet_condition_dsp`,
  `golden_a2_dynamic_*`), mas sem soak/endurance.
- `WaveNetA2Dyn`: tem goldens `a2_dynamic_gated_ch8`/`blended_ch3` (✅), mas sem soak.
- **Soak (`tests/soak_test.rs`):** verificado — cobre apenas WaveNet estático (`build_soak_wavenet`),
  LSTM estático, A2 **fixo** (`build_soak_a2::<8/3>`), resampler, mirror_buf, gate, adaptive.
  **Nenhum** soak para `WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn` ou `ConvNet`.
- **Bench:** `benches/inference_bench.rs` não tem nenhuma função para os engines dinâmicos (só estáticos
  - A2 fixo + ConvNet). Logo, PGO (F8) também não os profila.

**Impacto:** os engines dinâmicos são o **destino de roteamento de qualquer modelo não-catalogado do
NAMcore** (o que o usuário quer "carregar e processar impecávelmente"). Sem golden externo, soak nem
bench, são justamente os paths menos protegidos contra regressão e drift de longo prazo.

**Caminho de resolução:**

1. Escolher 1 geometria representativa por engine (ex.: LSTM 3×8 ou 1×7; WaveNet free CH=32 ou multi-array;
   A2Dyn já coberto por gated/blended) e gerar golden C++ correspondente.
2. Adicionar soak de silêncio+ruído (10M frames) para cada engine dinâmico + ConvNet em `soak_test.rs`.
3. Adicionar benches (bloco 64, regime áudio) para 1 modelo dinâmico representativo + ConvNet, com ID
   compatível com o filtro PGO (ver F8).

---

### F5 🟠 — FiLM no fast-path A2: checagem morta + divergência não validada vs C++

**Evidência:**

- `src/loader/nam_json/topology.rs:537-542`:

  ```rust
  if !check_film_all_inactive(raw) {
      // Models with FiLM are now routed to Dynamic if they break const-generic assumptions,
      // but currently the fast-path does support some FiLM. For safety, let's say:
      // if it has active FiLM it goes to Dynamic? No, the T1/T2 already added FiLM to fast-path.
      // So we don't return Dynamic here just for FiLM.
  }
  ```

  → o resultado de `check_film_all_inactive` é **descartado** (corpo `if` vazio, comentário indeciso).
- C++ `a2_fast.cpp:864-869` **rejeita** qualquer FiLM ativo do fast-path. NAM-rs aceita FiLM ativo no
  const-generic `WaveNetA2<CH>` (o `film_block` é plumbado em `model/mod.rs` p/ ch3/ch8) — é um
  **superset intencional**, porém **não há golden** validando FiLM-no-fast-path (os goldens A2 são
  sintéticos explicitamente **sem** FiLM, conforme `README.md`).

**Impacto:** correção silenciosa não comprovada. Um modelo com `condition_size==1` + FiLM ativo +
CH=3/8 entra no fast-path por uma decisão expressa em código morto/comentário ambíguo. Se a matemática
FiLM do fast-path divergir do C++/dinâmico, o resultado sonoro estaria errado sem nenhum gate.

**Caminho de resolução:**

1. Decidir e tornar explícita a política: (a) rotear FiLM-ativo para `WaveNetA2Dyn` (alinhado ao C++) e
   remover o código morto; **ou** (b) manter no fast-path como superset e **provar com golden**.
2. Se (b): gerar fixture A2 com FiLM ativo + golden C++ (`render` com FiLM) e teste de paridade.
3. Em qualquer caso, eliminar o `if` vazio/comentário ambíguo (clareza + `clippy`).

---

### F6 🟠 — Cluster de testes "placebo": só checam `is_finite()`/`abs < 100.0`

**Evidência (asserções praticamente inviolaveis):**

- `tests/nam_infer_test.rs`: `test_wavenet_stability_feather/_nano/_a2_full/_a2_lite`, `test_lstm_stability_2x8`
  (só finitude e/ou `abs < 100.0` em 64 amostras de senoide).
- `tests/a2_loader.rs`: `test_a2_lite_inference_produces_finite_output`, `test_a2_full_..._finite_output`
  (só finitude em 64 amostras de entrada constante 0.01).
- `tests/lstm_model_dyn_validation.rs`: `test_model_dyn_no_panic_edge`, `test_model_dyn_zero_input`
  (só finitude).
- `tests/spsc_pipeline.rs`: os 3 testes usam `abs < 100.0` como único gate de magnitude.
- `tests/wavenet_prewarm_edge.rs`: vários testes só asseguram finitude pós-prewarm.

**Impacto:** "fazem volume" sem proteger contra regressão real — um modelo produzindo lixo com RMS=50
passaria. Diluem o sinal de qualidade da suíte (contraria a filosofia "quality over quantity" do projeto).

**Caminho de resolução (criterioso, sem perder cobertura legítima):**

1. Onde houver um golden/oracle disponível (WaveNet/LSTM/A2 estáticos), trocar a asserção finite-only por
   comparação contra referência (MSE/ESR) ou apertar bounds para limites fisicamente justificados.
2. Onde o teste só checa "não-panic em edge" (ex.: `no_panic_edge`), manter mas renomear/consolidar e
   adicionar ao menos um invariante forte (determinismo já existe em outros testes — evitar duplicação).
3. Substituir `abs < 100.0` por bound derivado do sinal de entrada (ex.: `< 4×` pico de entrada).

> **Nota:** determinismo (MSE==0) e block-size invariance presentes nesses arquivos são **fortes** e
> devem ser preservados; apenas as asserções finite-only/`<100` são o alvo.

---

### F7 🟠 — Goldens neutralizados / dead test (contraria "todo golden deve poder falhar")

**Evidência:**

- `tests/cpp_parity.rs:551-555` — `live_cross_validation_v2_wavenet_lite` é **skip incondicional**
  (`eprintln!("SKIP: … known-divergent (T1.2) …"); return;`). Nunca passa nem falha → placebo puro.
- `BossWN-lite.nam` (sintético, **SNR 0.9 dB** vs C++) é gate de golden com `SNR ≥ 0 dB` — um threshold
  efetivamente neutralizado. O `threshold_calibration.rs` o tolera por exceção, mas o próprio projeto
  documenta isso como anti-padrão ("Threshold neutralizado") pendente de substituição (F12 em TODO-features).

**Impacto:** falsa sensação de cobertura; ruído na suíte; viola o princípio explícito do projeto.

**Caminho de resolução:**

1. Remover (ou `#[cfg]`-gate atrás de uma feature `known-divergent`) o teste sempre-skip.
2. Elevar a prioridade de substituir `BossWN-lite.nam` por um modelo **Lite (CH=12) real** (existe candidato
   `EVH-5150-Lite.nam` em `models-nondist`, classificado como "Real WaveNet Lite (CH=12)" no `CATALOG.txt`).
   Ao substituir, recalibrar o gate para SNR realista (>0 dB efetivo) e reativar o v2 lite.

---

### F8 🟠 — PGO (build-release.sh) não cobre todos os hot-paths de produção

**Evidência:**

- `utils/build-release.sh:185-197` profila `inference_bench` filtrando por `"64samp"`, `"AVX"`,
  `"Resampler"` e `dot_4x_bench` por `"avx2"/"avx512"`.
- Verificado em `benches/inference_bench.rs`:
  - **Capturados por `64samp`:** WaveNet Standard (`WaveNet_Standard_CH16_64samp_48kHz`), LSTM 2x16/1x40/2x24,
    A2Full (`A2Full_CH8_64samp_48kHz`), A2Lite, Cabsim short/medium/long, Container crossfade, Linear, NonDist.
  - **NÃO capturados:** **ConvNet** — os grupos usam sufixo `"_64f"` (`ConvNet_MultiChannel_64f`,
    `ConvNet_LargeKernel_64f`, `ConvNet_Dilated_64f`), que **não casa** com o filtro `"64samp"`.
  - **Sem bench algum** (logo, sem PGO): `WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn` (ver F4).

**Impacto:** o binário PGO/BOLT não é otimizado para ConvNet nem para os engines dinâmicos — paths
reais de produção ficam fora do profile, contrariando o objetivo de "cobrir inteligentemente os hot
paths novos".

**Caminho de resolução:**

1. Padronizar IDs de bench: adotar um token único de profiling (ex.: sufixo `_pgo` ou unificar em
   `64samp`) e renomear `ConvNet_*_64f` → `..._64samp` (ou ajustar o filtro do script para casar ambos).
2. Adicionar ao profiling PGO: 1 ConvNet representativo + 1 modelo dinâmico representativo (bloco 64),
   sem inflar o `--profile-time` (manter a estratégia enxuta atual).
3. Validar que `bench_nondist_models` (que depende de `models-nondist` ausente em build limpo) não
   degrade silenciosamente o profile (skip explícito quando vazio).

---

### F9 🟠 — `tests-long.sh`: suíte completa em clone novo depende de switch e não verifica frescor dos goldens

**Evidência:**

- `utils/tests-long.sh:56-195` (Phase 0) exige goldens presentes **e** toolchain C++ (cmake, g++/clang++),
  abortando (`exit 1`) se faltarem. Os goldens **estão commitados** (52 `.bin`, `~105 MB`,
  un-ignorados via `.gitignore:!tests/fixtures/*.bin`) → clone novo tem os goldens.
- Auto-regeneração só ocorre com `NAM_AUTO_BUILD_GOLDENS=1` (env switch) — o usuário pediu "por padrão,
  sem switches".
- Não há verificação de que os goldens commitados são **consistentes** com os `.nam` atuais
  (risco de golden stale após editar um fixture).

**Impacto:** o objetivo "clone novo + `mod-update.sh` → suíte completa perfeita" funciona **se** cmake/g++
já estiverem instalados; mas a regeneração automática (quando faltarem/forem stale) não é o padrão.

**Caminho de resolução:**

1. Tornar **auto-build-on-missing o padrão** quando o toolchain C++ + NeuralAmpModelerCore (pós `mod-update.sh`)
   estiverem presentes (inverter a lógica: só pular com `NAM_SKIP_GOLDEN_BUILD=1`).
2. Adicionar um **manifesto de frescor**: arquivo (gitignored) ou checagem rápida que mapeie
   `sha256(.nam) → golden` e dispare regeneração/aviso quando divergir.
3. Documentar de forma inequívoca: "único pré-requisito = `utils/mod-update.sh`" (e toolchain C++ de dev).

---

### F10 🟠 — Mecanismo `models-nondist`: funcional e git-safe, mas sem manifesto reprodutível e com discovery duplicada

**Evidência:**

- Mecanismo atual: symlink `tests/fixtures/models-nondist` (gitignored em `.gitignore`) + auto-discovery
  recursiva em `tests/nondist_validation.rs:14` (`find_models_in_dir`) e **também** em `tests/cpp_parity.rs`
  (`live_cross_validation_nondist_models`) → **duplicação** da função de descoberta.
- `tests/nondist_validation.rs` valida determinismo (MSE==0), block-invariance (<1e-7), finitude e
  estabilidade em silêncio (sem subnormais) — **não** valida classificação esperada nem paridade.
- `CATALOG.txt` (dentro do dir nondist) é gerado ad-hoc por `utils/check-model.py` e contém **escapes ANSI
  embutidos** (`[92m`…`[0m`) — não é machine-readable nem um manifesto reprodutível.
- Bom: descoberta **não** hard-coda nomes (compatível com "não pode ser linkado/mencionado no git") e
  faz auto-skip quando ausente (clone novo OK).

**Impacto:** expansível, mas não **reprodutível/auditável**: adicionar um modelo muda a cobertura sem
registro; sem asserção de classificação (ex.: "este CH=32 deve ser aceito como dinâmico"; "este
SlimmableContainer deve carregar"), o teste rápido não captura roteamento incorreto.

**Caminho de resolução:**

1. Consolidar `find_models_in_dir` em `tests/common/` (uma fonte).
2. Criar um helper (estender `utils/check-model.py` **ou** um bin Rust `testing`) que gere um **manifesto
   gitignored machine-readable** (ex.: `models-nondist/manifest.json`: `{filename, sha256, expected_class}`)
   sem cores ANSI.
3. `nondist_validation.rs` passa a **asserir contra o manifesto** quando presente (classe esperada:
   `Static{SKU}`/`WaveNetDyn`/`LstmDyn`/`A2{Full,Lite,Dyn}`/`Container`/`Rejected`), mantendo o auto-skip.
4. Documentar um único comando (ex.: `make nondist-manifest` ou seção no README) para (re)gerar o manifesto.

---

### F11 🟠 — `unreachable!()` no hot-path A2 fixo (panic latente na thread de áudio)

**Evidência:**

- `src/models/a2/model/mod.rs:532-538` — no `process()` do `WaveNetA2<CH>` fixo, o ramo de fallback escalar é,
  no build padrão (`#[cfg(not(any(test, feature = "dynamic-engine")))]`), substituído por
  `unreachable!("A2 layers always have ch3 or ch8 conv; …")`.
- Depende de invariante de `set_weights` (toda layer de um modelo CH=3/8 tem conv ch3/ch8). Hoje o invariante
  vale; mas é uma **garantia frágil** (não estrutural).

**Impacto:** se o invariante for violado por refactor futuro ou por um `weight_count`/layout inesperado, o
`unreachable!()` **panica na thread RT** — proibido por `AGENTS.md` (zero panics no áudio). É o tipo de
"quick-win de robustez" pedido.

**Caminho de resolução:**

1. Converter `unreachable!()` em `debug_assert!(false, …)` + fallback RT-safe (saída de silêncio para o bloco
   e set de flag de telemetria/`RT_STATUS_*`), nunca panic em release.
2. Varrer hot-paths de inferência por `unwrap/expect/panic/unreachable/indexação` (auditoria dirigida em
   `src/models/**` e `src/dsp/pipeline/**`); documentar invariantes provados vs. defensivos.

---

### F12 🟡 — `.gitignore` com `*.json` global sem un-ignore para fixtures (frágil)

**Evidência:**

- `.gitignore` ignora `*.json` globalmente; há un-ignore para `tests/fixtures/*.bin` mas **não** para
  `*.json`. `tests/fixtures/models/keras_unsupported.json` está versionado apenas por `git add -f` histórico.

**Impacto:** qualquer novo fixture `.json` (ou config relevante) seria **silenciosamente não rastreado** →
risco de perder fixtures de teste / quebrar a suíte em clone novo.

**Caminho de resolução:** escopar o ignore (ignorar JSON só onde necessário) **ou** adicionar
`!tests/fixtures/**/*.json` (e revisar se outros `.json` de config deveriam ser versionados),
aplicando o mesmo rigor já usado para `*.bin`.

---

### F13 🟡 — `unsafe` em cast de pesos LSTM sem verificação de tamanho

**Evidência:** `src/loader/dispatcher/lstm/weights.rs:56-61` usa `from_raw_parts_mut` sobre o buffer `u16`
assumindo comprimento `H4 * IH`; um descasamento seria UB. Cold path (load), mas sem assert mecânico.

**Caminho de resolução:** adicionar `assert_eq!(buf.len(), expected)` (ou `debug_assert` + checagem de
erro) imediatamente antes do `from_raw_parts_mut`; idealmente encapsular num helper seguro.

---

### F14 🟡 — `unwrap()` em `topology.rs` sem garantia estrutural

**Evidência:** `src/loader/nam_json/topology.rs:226` — `first_channels.unwrap()`. Hoje guardado por validação
anterior (`Rejected` para channels ausentes), mas não garantido pelo tipo.

**Caminho de resolução:** converter para erro checado (`ok_or_else(... )?`) para blindar contra refactors
futuros (cold path, custo zero).

---

### F15 🟡 — Goldens commitados pesam ~105 MB (trade-off de clone vs. reprodutibilidade offline)

**Evidência:** `du -ch tests/fixtures/*.bin` ≈ **105 MB** (v2 multi-SR ~1.9 MB cada × dezenas). O próprio
`golden_gen_build.sh:407-409` recomenda "Git LFS ou subset estratégico".

**Impacto:** clones pesados; tensão com o objetivo de "clone novo roda tudo". Manter commitado **favorece**
F9 (suíte completa offline), então é um trade-off consciente, não um defeito.

**Caminho de resolução (opcional):** avaliar Git LFS para os v2 grandes **ou** manter só um subset de SRs
commitado e regenerar os demais on-demand (alinhado ao auto-build de F9). Decisão a tomar junto com F9.

---

### Pontos fortes a preservar 🟢

- **Meta-testes anti-placebo** (`tests/threshold_calibration.rs`): forçam thresholds calibrados +
  comentários `// Measured:`. Excelente — manter e estender a ConvNet/dinâmicos (F3/F4).
- **Pirâmide de validação** (scalar-ref tight-band + golden loose-band) bem fundamentada (`architecture.md §6`).
- **Heap-audits zero-alloc** cobrindo até os paths novos (dynamic/container/nondist).
- **Fuzz de parsers** (`proptest_parsers.rs`, até 100k casos) e **proptests math** (SIMD↔scalar) sólidos.
- **Determinismo/block-invariance** (`self_consistency.rs`, `*_block_*`) — invariantes fortes.
- **Formato golden** (`README.md`) e **métricas single-pass** (MSE/MAE/SNR/PSNR/ESR/bits/LUFS) bem desenhados.
- **`is_a2_shape()`** quase 1:1 com `a2_fast.cpp` (exceto F5).

---

## 2. Épicos Ágeis

> Agrupamento lógico que viabiliza execução otimizada, segura e ágil. Sprints/tarefas serão definidas
> à parte por épico. **Criticidade** indica onde concentrar atenção e cautela.

---

### ÉPICO A — Sincronização Documentação ↔ Implementação (paridade real) [DONE]

- **Objetivo:** a documentação reflete fielmente o engine atual (engines dinâmicos + ConvNet ativos; A2
  goldens reais; matriz de paridade NAMcore completa e verdadeira).
- **Achados:** F1, F2.
- **Escopo:** reescrever `cpp_parity_map.md` (§3.3, §5, §9, §10.1), `architecture.md` (dispatch/A2/testing),
  `docs/testing.md` (matriz). Documentar o flag `dynamic-engine`. Reconciliar natureza dos goldens A2.
- **Criticidade:** 🟢 Baixa risco técnico / **Alto valor de confiança**. Exige rigor factual (cross-check
  com código), não código de produção.
- **Dependência:** **pré-requisito de clareza** para os demais épicos (define o que é "suportado").
- **DoD (alto nível):** nenhuma afirmação de "removido/rejeitado" que contrarie o código; toda arquitetura
  no `StaticModel` tem linha na matriz de paridade; skill `documentador` valida sincronia.

---

### ÉPICO B — Fechamento de cobertura de validação dos novos paths (🔴 CRÍTICO)

- **Objetivo:** toda arquitetura roteável (ConvNet + engines dinâmicos + FiLM-em-A2) tem **âncora externa
  vs NAMcore** + soak + bench — garantindo "resultados sonoros semelhantes" e "carregar/processar
  impecávelmente todos os modelos suportados".
- **Achados:** F3 (ConvNet golden), F4 (dinâmicos: golden+soak+bench), F5 (política + golden FiLM).
- **Escopo:** fixtures novos (ConvNet; geometria LSTM/WaveNet dinâmica representativa; A2+FiLM se política=superset);
  extensão de `golden_gen_build.sh`; novos testes em `golden_vectors.rs`/`cpp_parity.rs`; entradas calibradas
  em `validation.rs`; soaks em `soak_test.rs`; benches em `inference_bench.rs`.
- **Criticidade:** 🔴 **A mais alta.** É o épico que materializa a correção sonora dos paths novos.
  Itens de maior risco: **F5** (divergência silenciosa FiLM) e **F4** (engines dinâmicos = catch-all de
  "todos os modelos do NAMcore"). Atenção a: o `render` C++ suporta ConvNet? FiLM no fast-path C++? (validar
  geração antes de calibrar thresholds).
- **Dependência:** parte da infra de geração de golden é compartilhada com o Épico E (golden_gen_build,
  naming de bench) — coordenar.
- **DoD (alto nível):** `threshold_calibration` passa para ConvNet + ≥1 dinâmico (sem fallback silencioso);
  política FiLM explícita e provada; soak 10M frames sem NaN/Inf para cada path novo.

---

### ÉPICO C — Robustez RT & caça a bugs (panics, `unsafe`, parsing) [DOING]

- **Objetivo:** zero superfícies de panic no hot-path; `unsafe` com invariantes verificados; parser/loader
  defensivos; gitignore robusto.
- **Achados:** F11 (unreachable RT), F13 (unsafe LSTM), F14 (unwrap topology), F12 (gitignore `*.json`).
- **Escopo:** converter `unreachable!()` A2 em fallback RT-safe; sweep de `unwrap/expect/panic` nos hot-paths;
  asserts de tamanho antes de `from_raw_parts_mut`; erro checado em `topology.rs:226`; escopar/un-ignore `*.json`.
- **Criticidade:** 🟠 Média (F11 toca a thread de áudio — RT-safety é regra dura do projeto).
- **Dependência:** independente; pode rodar em paralelo com A/D.
- **DoD (alto nível):** nenhum `unwrap/unreachable/panic` alcançável em `process()` de release; clippy limpo;
  fuzz de parser continua verde.

---

### ÉPICO D — Saneamento da suíte de testes (anti-placebo & consolidação) [DOING]

- **Objetivo:** eliminar testes de baixo valor (finite-only/`<100`, dead tests, golden neutralizado),
  aplicando o princípio "todo golden deve poder falhar" sem perder cobertura legítima.
- **Achados:** F6 (placebos finite-only), F7 (dead test + Lite neutralizado).
- **Escopo:** fortalecer/consolidar testes de `nam_infer_test`, `a2_loader`, `lstm_model_dyn_validation`,
  `spsc_pipeline`, `wavenet_prewarm_edge`; remover/gate o `live_cross_validation_v2_wavenet_lite`;
  substituir `BossWN-lite.nam` por modelo Lite real (candidato `EVH-5150-Lite.nam`) e recalibrar gate.
- **Criticidade:** 🟠 Média. Exige critério para **não** remover invariantes fortes (determinismo,
  block-invariance) — só os placebos.
- **Dependência:** F7↔Épico B/E (substituir Lite usa infra de golden); coordenar.
- **DoD (alto nível):** nenhum teste com asserção inviolável; nenhum gate de golden com threshold neutralizado
  sem exceção documentada; contagem de testes estável ou menor com **mais** poder de detecção.

---

### ÉPICO E — Cadeia de suprimentos reprodutível (fresh-clone, PGO, nondist) [DONE]

- **Objetivo:** `tests-long.sh` roda a suíte completa em **clone recém-clonado** tendo apenas `mod-update.sh`
  como pré-requisito; PGO cobre inteligentemente todos os hot-paths; mecanismo `models-nondist` reprodutível
  e auditável.
- **Achados:** F8 (PGO/bench naming), F9 (fresh-clone/auto-build/frescor), F10 (manifesto nondist + discovery),
  F15 (peso dos goldens — decisão de LFS/subset).
- **Escopo:** padronizar IDs de bench + filtros PGO (incluir ConvNet/dinâmico); inverter lógica de
  auto-build de goldens (padrão = on-missing); manifesto de frescor `.nam↔golden`; consolidar `find_models_in_dir`
  em `tests/common`; gerar manifesto nondist machine-readable (sem ANSI); decidir Git LFS/subset.
- **Criticidade:** 🟠 Média (mexe em scripts de release/PGO e geração de goldens — testar com cautela; lembrar
  que **IA não executa `tests-long.sh`** — validação via desenvolvedor humano, conforme aviso no script).
- **Dependência:** fornece a infra que o Épico B consome (golden_gen_build extensível, benches dinâmicos/ConvNet).
- **DoD (alto nível):** em VM/clone limpo + `mod-update.sh`, `tests-long.sh` completa todas as fases sem
  switches; perf-annotate/PGO inclui ConvNet e ≥1 dinâmico; `models-nondist` documentado num único comando.

---

## 3. Matriz de Rastreabilidade (Achado → Épico)

| Achado | Sev. | Tema                                         | Épico |
| ------ | ---- | -------------------------------------------- | ----- |
| F1     | 🔴   | Docs: dinâmicos/ConvNet "removidos"          | **A** |
| F2     | 🟠   | Docs: natureza dos goldens A2                | **A** |
| F3     | 🔴   | ConvNet sem golden de paridade               | **B** |
| F4     | 🔴   | Dinâmicos sem golden/soak/bench              | **B** |
| F5     | 🟠   | FiLM fast-path: código morto + sem golden    | **B** |
| F6     | 🟠   | Testes placebo (`is_finite`/`<100`)          | **D** |
| F7     | 🟠   | Dead test + golden Lite neutralizado         | **D** |
| F8     | 🟠   | PGO não cobre ConvNet/dinâmicos              | **E** |
| F9     | 🟠   | `tests-long.sh` fresh-clone/auto-build       | **E** |
| F10    | 🟠   | Mecanismo `models-nondist`/manifesto         | **E** |
| F11    | 🟠   | `unreachable!()` no hot-path A2 (RT)         | **C** |
| F12    | 🟡   | `.gitignore` `*.json` frágil                 | **C** |
| F13    | 🟡   | `unsafe` LSTM sem assert de tamanho          | **C** |
| F14    | 🟡   | `unwrap()` em topology                       | **C** |
| F15    | 🟡   | Peso dos goldens (LFS/subset)                | **E** |

---

## 4. Ordem de Execução Sugerida

1. **Épico A** (docs) — primeiro: estabelece a verdade de escopo para todos os demais. Baixo risco, alto valor. [DONE]
2. **Épico E** (infra) **em paralelo com C/D** — provê a base de geração de goldens/benches que o B consome. [DOING]
3. **Épicos C e D** — independentes, podem correr em paralelo desde o início.
4. **Épico B** (validação dos novos paths) — **o mais crítico**; idealmente após E disponibilizar a infra de
   golden/bench (ou coordenado com ela).

> **Riscos transversais a vigiar:** (a) IA **não** executa `tests-long.sh` nem `build-release.sh` longos —
> validações pesadas (live C++ parity, soak, PGO) exigem execução pelo desenvolvedor humano; (b) toda
> geração de golden deve usar o commit canônico `9c7b185` (NAMcore v0.5.3) para reprodutibilidade;
> (c) ao mexer em `topology.rs`/`set_weights`, reexecutar `a2_loader` + `golden_vectors` (gates de roteamento).
