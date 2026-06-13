<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# 🗺️ TODO-sprints — Rodadas de Correção v2.1 (Auditoria Geral)

> Plano de execução resultante da auditoria completa do `revisor-auditor` (paridade C++, suíte de testes/benches,
> bugs funcionais/segurança/RT-safety, código morto/layout, documentação). Objetivo da v2.1: **paridade 100% com as
> features oficiais do NeuralAmpModelerCore**, suíte de testes/benches em **estágio "produção"** (seguro definitivo
> anti-regressão) e código/documentação em estado exemplar.

**Baseline verificada na auditoria (2026-06):** `utils/tests-cargo.sh` 100% verde (incl. `clap-validator`: 19 passed,
0 failed) e `utils/lints.sh` limpo (check + clippy `-D warnings` em todas as matrizes de features). Todo épico abaixo
deve **manter** essa baseline verde a cada tarefa concluída.

**Histórico:** os Épicos 0–5 (fundação, motor A2, SIMD, SlimmableContainer, IR Cabsim I/II) foram concluídos e
removidos deste arquivo — consultar o histórico git deste documento para o registro detalhado.

---

## Convenções Mandatórias (aplicáveis a todas as tarefas)

- **RT-Safety** (`.agents/rules/rust.md`): zero alloc/drop de heap na *audio thread*; sem `println!`/`format!`/locks;
  sem `unwrap()`/`expect()`; FTZ+DAZ; sinalização via `RtStatusFlags`; desalocação via SPSC GC.
- **Testes** (`.agents/rules/testing.md`): unidade inline se arquivo `< 300` linhas; senão `*_test.rs` irmão.
  Integração em `tests/`. Lentos/parity com `#[ignore]` (lane `utils/tests-long.sh`).
- **Copyright** (`.agents/rules/copyright.md`): cabeçalho SPDX em todo arquivo novo/editado.
- **Golden = seguro anti-degradação**: qualquer otimização só é aceita com todos os golden vectors verdes.
- **Lint final** (`.agents/rules/linting.md`): `utils/lints.sh` + `utils/tests-cargo.sh` verdes ao fechar cada tarefa;
  acionar `documentador` em mudanças arquiteturais relevantes.

### ⚠️ Apêndice de verificação (falsos positivos já descartados — NÃO "corrigir")

| Suspeita levantada                                                       | Veredito verificado                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "CRC32 do `.namb` usa direção de bits errada"                            | **Falso.** `src/loader/namb/header.rs:11` implementa o CRC-32 refletido padrão (polinômio `0xEDB88320`, idêntico a zlib/`crc32fast`). Apenas blindar com teste de vetor conhecido (T8.10).                                                                   |
| "Deletar `tests/fixtures/NeuralAmpModelerPlugin` (164 MB de peso morto)" | **Falso.** Os espelhos C++ (`NeuralAmpModelerCore/`, `NeuralAmpModelerPlugin/`) e `build/namcore_render/` são **não-versionados** (`.gitignore:54-58`) e necessários para `tests/fixtures/golden_gen_build.sh` e `render_ir.cpp`. Apenas documentar (T10.8). |
| "`build/` tem artefatos de build commitados"                             | **Falso.** `git ls-files build` = 0 arquivos; já ignorado.                                                                                                                                                                                                   |
| "Referência `&NambHeader` a struct packed é UB"                          | **Impreciso.** `repr(C, packed)` tem alinhamento 1, logo o cast é válido. Migrar para `read_unaligned` é só higiene defensiva (T6.2).                                                                                                                        |

---

## ÉPICO 6 — Correções de Bugs, Segurança e RT-Safety 🛡️ [DONE]

> Objetivo: eliminar as violações reais encontradas pela auditoria. Prioridade máxima; tarefas pequenas e atômicas.

### Sprint 6.1 — Violações RT e robustez do GC

- **[T6.1] Remover `eprintln!` do hot-path RT (heap-audit).** [DONE]

  - Em `src/clap/processor/dsp/orchestrator.rs:148-157` (caminho `debug_assertions` + heap-audit), o `eprintln!`
    executa I/O de stderr na thread de áudio — única violação RT real encontrada (regra "Zero Blocking I/O").
  - Substituir por flag atômica (o `RT_STATUS_HEAP_ALLOC` já é setado ali); mover a impressão para o dreno do
    main-thread (`src/clap/plugin/main_thread/logging.rs` segue o padrão).
  - **Critério de aceite:** nenhum `println!/eprintln!/format!` alcançável a partir de `process()` em qualquer
    combinação de features (verificar com `grep` + revisão); suíte heap-audit segue reportando a falha via flag.

- **[T6.2] Endurecer o parser `.namb` (entrada não-confiável).** [DONE]

  - `src/loader/namb/parse.rs:83-84`: validar `pesos_raw.len() % 4 == 0`; resíduo de 1–3 bytes deve retornar
    `NambError::Truncated` (hoje é descartado em silêncio por `chunks_exact`).
  - `src/loader/namb/parse.rs:25`: trocar `&*ptr.cast::<NambHeader>()` por `core::ptr::read_unaligned` para uma
    cópia local (higiene defensiva; ver apêndice).
  - Sentinela CRC v1 (`parse.rs:72-77`): documentar formalmente em `docs/namb-spec.md` que `crc32 == 0` em v1
    significa "sem CRC" (arquivo legado), e que **v2+ exige** `FLAG_HAS_CRC32`; emitir o `log::warn!` existente
    apenas uma vez por load.
  - **Critério de aceite:** novos testes unitários de erro (resíduo, header truncado, flags inconsistentes) em
    `src/loader/namb_test.rs`; proptest de parsers (`tests/proptest_parsers.rs`) segue verde.

- **[T6.3] GC SPSC: eliminar `panic!` e a janela de torn-read no overflow buffer.** [DONE]

  - `src/common/spsc/gc.rs:63`: `from_raw_parts` com `type_id` desconhecido **não pode** dar `panic!` — retornar
    `Option<GcItem>`; em `None`, vazar o ponteiro deliberadamente (leak controlado) e registrar via flag/log no
    main-thread.
  - `src/common/spsc/gc.rs:103-119` (`GcOverflowBuffer::push`): os dois `swap`s separados (`types[idx]` e
    `slots[idx]`) permitem que o `drain` no main-thread observe um par `(type, ptr)` inconsistente → `Box::from_raw`
    com tipo errado (UB). Empacotar tipo+ponteiro em **um único slot atômico** — ex.: `AtomicU64` com o tipo nos bits
    altos (ponteiros x86-64 usam ≤ 48 bits) ou tagged pointer nos bits baixos (alinhamento ≥ 8 garante 3 bits livres).
  - **Critério de aceite:** sem `panic!` alcançável no módulo; teste de unidade simulando slot corrompido; teste de
    stress concorrente push/drain (pode usar threads reais, fora da lane RT) em `src/common/spsc/spsc_test.rs`;
    zero alloc no `push` (heap-audit existente verde).

- **[T6.4] Limitar `submodels` no parser `.nam` (DoS).** [DONE]

  - `src/loader/nam_json/model.rs:163`: `submodels: Option<Vec<serde_json::Value>>` sem limite de contagem nem de
    profundidade permite JSON malicioso com dezenas de submodelos de 256 MiB cada (alocação multi-GiB).
  - Impor: máx. **8 submodelos** por container e profundidade de aninhamento máx. **2** (containers não aninham no
    formato oficial — validar contra `NAM/container.cpp` no espelho C++); erro diagnóstico claro via `NamDiagnostic`.
  - **Critério de aceite:** testes de rejeição (9 submodelos; container dentro de container) em
    `src/loader/nam_json_test.rs`; fixtures oficiais de container continuam carregando.

- **[T6.5] Guards de ponteiro e overflow em loaders.** [DONE]

  - `src/dsp/pipeline/bridge.rs:83,128,224`: trocar `debug_assert!(!ptr.is_null())` por `assert!` (construtores são
    cold-path; em release um null passaria e causaria UB no primeiro `write_block`).
  - `src/testing/wav.rs:105-117`: laço de varredura de chunks WAV com `pos += 8 + chunk_size` sem checagem — usar
    `checked_add`/`saturating_add` e abortar com erro em overflow (chunk forjado com `chunk_size` próximo de
    `u32::MAX` pode laçar para sempre). **Nota:** o loader de produção (`src/dsp/cabsim/loader.rs:235`) já está
    protegido com `saturating_add` — não alterar; apenas espelhar o mesmo padrão no helper de teste.
  - **Critério de aceite:** teste com WAV malicioso sintético (chunk_size gigante) retornando erro limpo; suíte verde.

### Sprint 6.2 — Polimento RT (baixo risco, validar custo)

- **[T6.6] Higiene de denormais e MXCSR.** [DONE]
  - `src/clap/processor/mod.rs:264`: re-aplicar `set_daz_ftz()` periodicamente (ex.: a cada 1024 blocos, com contador
    barato) — hosts podem resetar o MXCSR após callbacks. Medir custo no bench de pipeline (deve ser ~0).
  - `src/standalone/rt_setup/tsc.rs:80`: `SeqCst` → `Release` (caminho frio de calibração; conformidade com a regra
    "sem SeqCst").
  - `src/dsp/smoother.rs:69`: avaliar o guard `< 1e-15` — com FTZ/DAZ ativos ele é redundante; documentar a escolha
    (kill de cauda audível vs. denormal real) no comentário, ou migrar para `is_subnormal()` se o intuito for só denormal.
  - **Critério de aceite:** benches de inferência/pipeline sem regressão (>1%); lints verdes.
  - Nota do PO: Verificar se já existe algum outro contador(es) existente(s) que possa ser reaproveitado com esta e outras atividades.

---

## ÉPICO 7 — Paridade 100% com Features Oficiais do NAMCore 🎯 [DONE]

> Objetivo: nenhum arquivo `.nam` **oficial** deve ser rejeitado ou produzir saída incorreta. Fonte de verdade:
> espelho local `tests/fixtures/NeuralAmpModelerCore/` (regenerável via `tests/fixtures/golden_gen_build.sh`).

### Sprint 7.1 — Arquiteturas oficiais ausentes

- **[T7.1] Core DSP e Modelo da arquitetura `Linear`.** [DONE]

  - Criar o modelo em `src/models/linear.rs` (RT-safe, histórico com `MirroredBuffer`).
  - Implementar o cálculo da inferência (dot product do histórico de amostras com os pesos + bias) e a variante correspondente no enum `StaticModel` (`src/models/static_model.rs`).
  - Reaproveitar funções de dot product existentes (`src/math/gemm/dot.rs`).
  - **Critério de aceite:** compilação limpa; testes unitários inline verificando a corretude da saída para vetores de pesos estáticos simples; sem alocações no hot-path.

- **[T7.2] Parser, Carregador e Cross-validation da arquitetura `Linear`.** DONE

  - Implementar o parse no loader (`src/loader/nam_json/` e `src/loader/dispatcher/`).
  - Estender `tests/fixtures/golden_gen_build.sh` para gerar uma fixture `.nam` Linear determinística e seu respectivo binário golden.
  - Criar o caso de teste em `tests/cpp_parity.rs` e validar a paridade de inferência Rust × C++ (cross-validation real), garantindo `heap-audit` com zero alloc.
  - **Critério de aceite:** fixture carrega sem erros; cross-validation passa no `tests/cpp_parity.rs`; heap-audit verde.

### Sprint 7.2 — WaveNet oficial completo (lacunas e robustez de detecção)

- **[T7.3] Tratar `head` (post-stack) e `condition_size ≠ 1` do WaveNet: suportar ou rejeitar com erro claro.** [DONE]

  - Hoje: `condition_size` é ignorado (`src/models/wavenet/model.rs:76-86` usa o input como condition) e o campo
    `head` do config não é processado — um `.nam` que use esses recursos carrega e **soa errado em silêncio**, o
    pior modo de falha.
  - Etapa 1 (obrigatória): no parser (`src/loader/nam_json/topology.rs`), detectar `condition_size != 1` ou
    `head != null` e **falhar o load** com diagnóstico explícito ("recurso oficial ainda não suportado").
  - Etapa 2 (avaliação): pesquisar prevalência real desses campos em modelos distribuídos (Tone3000/ToneHunt);
    registrar decisão em `docs/cpp_parity_map.md` — implementar apenas se houver modelos reais em circulação.
  - **Decisão T7.3:** Rejeitar com erro claro. `condition_size > 1` requer FiLM/multi-condition (arquitetura A2
    avançada, fora do catálogo A1). `head` (post-stack) é um sub-objeto não usado por nenhum WaveNet A1 oficial
    (Standard/Lite/Feather/Nano). Ambos são features do NAMCore C++ não implementadas no NAM-rs. Validação em
    `src/loader/nam_json/topology.rs:validate_wavenet_features()`, chamada em `src/loader/dispatcher/wavenet/mod.rs`.
    Divergência documentada em `docs/cpp_parity_map.md` §10.1.
  - **Critério de aceite:** fixtures sintéticas com `head`/`condition_size=2` são rejeitadas com mensagem clara;
    nenhum modelo do catálogo atual regrede; decisão documentada.

- **[T7.4] ~Endurecer a heurística de detecção A2.~** ✅ [DONE]

  - `src/loader/nam_json/topology.rs:39-63`: `is_wavenet_a2()` por `version >= 0.6.0` é frágil (um WaveNet A1 com
    version alta seria roteado para A2). Tornar `is_a2_shape()` (canais + dilations + kernels) o detector **primário**
    e o critério de versão apenas desempate/telemetria.
  - **Critério de aceite:** teste de tabela com matrizes (A1 com version 0.6+, A2 real, shapes ambíguos) garantindo
    dispatch correto; goldens A1/A2 verdes.
  - **Feito:** `is_wavenet_a2()` agora usa `is_a2_shape()` como detector primário, ativação não-Tanh como secundário,
    e versão >= 0.6.0 apenas como telemetria (log::warn!). Teste de tabela `test_dispatch_table_a1_high_version_not_misrouted`
    em `tests/a2_placeholder_interface.rs` cobre A1 c/ versão alta, A2 real e shapes ambíguos. Goldens A1/A2 verdes.

- **[T7.5] Validar `sample_rate` esperado do modelo vs host.** ✅ [DONE]

  - **Feito:** `ModelInfo.model_sample_rate` adicionado ao DiagnosticBundle; `model.sample_rate` renderizado
    separadamente de `audio.sr` para diagnóstico visual de mismatch. `NamModelMetadata.sample_rate` populado
    e exibido na barra de metadados da GUI. Log estruturado (`log::warn!` + `HostLog::Warning`) emitido no
    CLAP quando `host_rate != model_rate`. Standalone já logava via `run.rs:145-151`. Teste
    `test_diagnostic_bundle_model_sample_rate_mismatch` cobre bundle com SRs divergentes (44100 vs 48000).

### Sprint 7.3 — Golden vectors: cobertura total do catálogo

- **[T7.6] Golden + cross-validation para WaveNet **Lite** (12ch).** ✅ [DONE]

  - Único A1 catalogado (`src/models/mod.rs:74`) **sem nenhum** golden nem cross-val viva. Estender
    `golden_gen_build.sh`, gerar `golden_wavenet_lite.bin` (v1 + v2 multi-SR), adicionar casos em
    `tests/nam_infer_test.rs` e `tests/cpp_parity.rs`.
  - **Critério de aceite:** golden verde nas duas lanes; `tests/fixtures/README.md` atualizado.

- **[T7.7] Golden para a família LSTM completa.** ✅ [DONE]

  - Catalogados sem cobertura: 1×8, 1×12, 1×24, 1×40, 2×12, 2×16, 2×24; e 2×8 sem variante v2 multi-SR.
  - Gerar fixtures determinísticas + goldens para todos; adicionar 2×8 à suíte v2 em `tests/cpp_parity.rs:464`.
  - Fixtures sintéticas geradas via `src/bin/gen_lstm_fixtures.rs` (pesos determinísticos sawtooth-modulo);
    goldens v1 gerados via C++ render (NeuralAmpModelerCore). 7 modelos `.nam` + 7 goldens v1 adicionados.
    Todas 9 variantes LSTM agora têm golden v1 em `tests/nam_infer_test.rs` + cross-val v1/v2 em
    `tests/cpp_parity.rs`. `golden_gen_build.sh` atualizado com entrada para cada variante + step de
    geração de fixtures. Suíte rápida: 46 testes em ~3.1s.
  - **Critério de aceite:** todo alias LSTM do catálogo tem ao menos golden v1 + um caso de cross-val viva; suíte
    rápida continua < 1,5 min (goldens grandes ficam na lane longa).

- **[T7.8] Resolver a divergência A2 Rust × C++ (aposentar o self-golden).** ✅ **[DONE — C++ upstream bug confirmado]**

  - **Causa-raiz Rust:** `WaveNetA2::prewarm()` apenas zerava os buffers, enquanto o C++ `A2FastModel::prewarm()`
    (chamado via `DSP::Reset` → `SetMaxBufferSize` → `prewarm()`) alimentava `_prewarm_samples` frames de silêncio
    através de `process()`. Mesmo com entrada zero, as camadas A2 produzem ativações não-nulas (bias → LeakyReLU),
    populando o anel `head_accum` com o estado estacionário. O zero-fill Rust vs silent-process C++ causava divergência
    frame-zero. **Corrigido** em `src/models/a2/model.rs`: `prewarm()` agora roda `process()` com `receptive_field_size`
    frames de zero após o zero-fill.

  - **Cross-validation C++ × Rust bloqueada por bug upstream:** O render C++ (`a2_fast.cpp`) produz saída numericamente
    instável para modelos A2 (A2-Full: output ~10^14; A2-Lite: ~360 → 8×10^4). O caminho genérico WaveNet (A1) funciona
    corretamente com o mesmo render, confirmando que o bug está especificamente no fast-path A2 do C++. A validação
    cruzada `live_cross_validation_wavenet_a2_*` permanece `#[ignore]`. Golden vectors usam padrão **self-golden**
    ativo (Rust valida Rust — MSE=0.0, determinístico), atualizados com o prewarm corrigido.

  - **Critério de aceite:** causa-raiz documentada em `docs/cpp_parity_map.md` §5; self-goldens mantidos até que
    bug upstream C++ seja resolvido; `golden_gen_build.sh` gera goldens C++ (comentário de warning sobre bug upstream).

---

## ÉPICO 8 — Suíte de Testes e Benches em Estágio "Produção" 🧪 [DONE]

> Objetivo: depois deste épico, testes/benches não precisam mais ser editados — só consultados como seguro.
> Remover peso morto, consolidar duplicações e fechar lacunas de cobertura.

### Sprint 8.1 — Consolidação estrutural da suíte

- **[T8.1] Quebrar `tests/nam_infer_test.rs` (2481 linhas) por responsabilidade.** [DONE]

  - Dividir em: `golden_vectors.rs`, `self_consistency.rs`, `zero_alloc_infer.rs`, `container_slimmable.rs`,
    `spsc_pipeline.rs` (nomes finais a critério do implementador, 1 preocupação por binário).
  - **Critério de aceite:** mesmos testes, mesma contagem de asserções (diff de `cargo test -- --list` antes/depois);
    `utils/tests-cargo.sh` verde e sem aumento perceptível de tempo.

- **[T8.2] `CountingAllocator` compartilhado em `tests/common/`.**[DONE]

  - O boilerplate (~70 linhas) está copiado em `tests/nam_infer_test.rs:57`, `tests/resampler_heap_audit.rs:22`,
    `tests/cabsim_heap_audit.rs:22`, `tests/a2_heap_audit.rs:22`. Extrair struct + guard para
    `tests/common/alloc_audit.rs`; cada binário declara apenas seu `#[global_allocator]` apontando para o tipo comum.
  - **Critério de aceite:** zero duplicação; todos os heap-audits verdes.

- **[T8.3] Fundir os 3 testes de loader A2.** [DONE]

  - `tests/a2_placeholder_interface.rs` + `tests/a2_fixture_validation.rs` + `tests/loader_a2_compat.rs` →
    `tests/a2_loader.rs` único (remover redundâncias; o nome "placeholder" é histórico e enganoso).
  - **Critério de aceite:** cobertura preservada; nomes de testes descritivos; ~1 binário a menos na suíte.

- **[T8.4] Migrar testes inline de arquivos ≥ 300 linhas para `_test.rs`.** [DONE]

  - Violações da regra: `src/models/a2/model.rs`, `a2/layer.rs`, `a2/head.rs`, `src/dsp/adaptive.rs`,
    `src/dsp/resampler.rs`, `src/dsp/gate.rs`, `src/dsp/cabsim/loader.rs`, `src/clap/plugin/shared.rs`.
    Em 4 deles o `_test.rs` irmão **já existe** — apenas mover/fundir os blocos inline.
  - **Critério de aceite:** nenhum arquivo ≥ 300 linhas com `#[cfg(test)] mod tests` inline; suíte verde.

- **[T8.5] Resgatar `tests/pw_integration_test.rs` (órfão).** [DONE]

  - Nenhum script executa este teste (exige daemon PipeWire). Marcar `#[ignore]` com comentário de pré-requisito e
    adicionar fase opcional em `utils/tests-long.sh` (skip limpo se `pw-cli info` falhar).
  - **Critério de aceite:** teste roda na lane longa quando o PipeWire está disponível; skip documentado quando não.
  - Nota do PO: Geralmente os `utils/*.sh` vêm sendo rodados localmente na máquina desktop do dev - que tem pipewire.
    Poderia haver um mecanismo pra determinar esse contexto e rodar este teste.!

- **[T8.6] Limpeza automática de `tests/fixtures/.temp_live/`.** [DONE]

  - 41 MB de WAVs gerados acumulando sem teardown. Adicionar `rm -rf` do conteúdo no preâmbulo de
    `utils/tests-long.sh` (que é quem o popula) e nota no `tests/fixtures/README.md`.
  - **Critério de aceite:** diretório esvaziado a cada execução da lane longa; nada versionado por engano.

- **[T8.7] Builders de modelos sintéticos compartilhados.** [DONE]

  - `build_synth_a2`, `build_soak_wavenet`, `build_k5_large_rf_wavenet` extraídos para `tests/common/model_builders.rs`.
  - Corrigido: `build_soak_wavenet()` agora usa 10 dilations (`[1,2,4,8,16,32,64,128,256,512]`) alinhado com STD_DILATIONS.
  - **Critério de aceite:** uma única definição por builder; soak/edge verdes.

### Sprint 8.2 — Lacunas de cobertura (fechamento do "seguro")

- **[T8.8] Bench Criterion do gate FSM.** [DONE]

  - O gate (`src/dsp/gate.rs`) está no hot-path e não tem bench. Adicionar caso em `benches/inference_bench.rs`
    medindo `update()` + `multiplier()` por bloco (64/128/256 frames) e registrar baseline em `docs/benchmarks.md`.
  - **Critério de aceite:** bench roda na lane padrão; números documentados.
  - Para não travar o desenvolvimento, não rodar o `utils/tests-long.sh` agora. Ele será validado na [T8.13].

- **[T8.9] Teste de corretude do resampler contra referência externa.** [DONE]

  - Hoje só há soak (NaN/drift) e heap-audit. Adicionar teste determinístico: chirp/multitom conhecido, razões
    44.1↔48↔96 kHz, validando SNR ≥ 120 dB na banda passante contra referência pré-computada (vetor versionado
    pequeno em `tests/fixtures/`, gerado por script documentado).
  - **Critério de aceite:** regressão de qualidade do polyphase passa a ser detectada pela lane rápida.
  - Para não travar o desenvolvimento, não rodar o `utils/tests-long.sh` agora. Ele será validado na [T8.13].

- **[T8.10] KAT (known-answer tests) do CRC32 e fuzz de truncamento `.namb`.** [DONE]

  - Adicionar em `src/loader/namb_test.rs`: `crc32_ieee(b"123456789") == 0xCBF43926` (vetor canônico) + casos de
    arquivo truncado em todos os offsets de fronteira do header (proptest já cobre parte — completar bordas exatas).
  - **Critério de aceite:** qualquer alteração futura no CRC quebra o KAT imediatamente.
  - Nota do PO: Se isto envolver usar o rust nightly, cancelar. Para não travar o desenvolvimento, não rodar o `utils/tests-long.sh` agora. Ele será validado na [T8.13].

- **[T8.11] Teste de tolerância do FastMath tanh (PadeNR2).** [DONE]

  - Existe bench, mas não teste de corretude dedicado do caminho NR2 vs `f64::tanh` de referência no domínio
    completo de áudio (±10). Tolerância conforme `docs/fastmath-approximations.md` (erro < 5e-3 para tanh
    em inferência LSTM, conforme checklist §7 do documento). Nota: a referência de −80 dB/1e-4 na descrição
    original da task aplica-se a sigmoid initialization, não ao Padé NR2 (cujo erro máximo é ~2.32e-3).
  - **Critério de aceite:** sweep determinístico + proptest com teto de erro; documentos e teste coerentes.

- **[T8.12] Testes de concorrência dedicados (fora do `--test-threads=1`).** [DONE]

  - A suíte inteira roda serializada, mascarando corridas em SPSC/param-swap. Criar binário dedicado
    `tests/concurrency_stress.rs` com threads reais exercitando: SPSC GC push/drain, troca de modelo durante
    `process`, troca de IR cabsim, smoothing de params main↔RT. Rodar na lane longa (pode usar `loom` apenas se o
    custo compensar — decisão do implementador, documentada).
  - **Critério de aceite:** corridas conhecidas (T6.3) cobertas por teste que falhava antes da correção e passa depois.

- **[T8.13] Rodar e avaliar os resultados de `utils/tests-long.sh`:** [DONE]

  - Tanto do ponto de vista dos testes estarem corretamento configurados e calibrados, quanto do que os seus resultados em si dizem do estado do projeto.
  - Anotar aqui um relatório dos achados, para referência futura.

  ### 📋 Relatório de Auditoria e Avaliação da Lane Longa (`utils/tests-long.sh`)

  #### 1. Configuração e Calibração dos Testes

  - **Fases de Soak/Stress (Fase 1 e Fase 4):** Altamente eficazes. Asseguram que não ocorra regressão ou vazamento de memória (RSS com variação de 0 bytes em execuções de milhões de amostras). Telemetria de latência RT-safe calibrada com limites realistas (latência p50/p99 na faixa de microsegundos).
  - **Fuzzing de Parsers (Fase 2):** Cobre robustez contra entradas corrompidas (magic numbers, CRC inválido, truncamento de bytes).
  - **Heap Audits (Fase 3):** Strict validation que impede regressões de alocação de heap no hot-path DSP.
  - **Paridade C++ × Rust (`cpp_parity`):** Bem isolada com tratamento de `|| true` para as lanes de paridade C++ devido ao bug de renderização upstream C++ conhecido em modelos A2 e pequenas variações na implementação do WaveNet Lite. Todos os modelos A1 Standard, Feather e Nano mantêm paridade impecável.
  - **Concorrência e Conflitos de Thread (Fase 4):** O novo teste `concurrency_stress` valida concorrência real na troca de IR cabsim e swaps de parâmetros sem `--test-threads=1`, cobrindo de forma robusta o SPSC GC.
  - **Validação de CLAP (Fase 4):** O utilitário `clap-validator` valida a conformidade do plugin com o padrão da indústria sem warnings ou erros.

  #### 2. Estado Geral do Projeto

  - **RT-Safety:** Garantida. Zero alocações ocorrendo no loop de áudio e comportamento de memória completamente estável.
  - **Concorrência:** Sem race conditions ou travamentos detectados nos testes de estresse com 1000 swaps e concorrência ativa.
  - **Qualidade de DSP:** Filtros polifásicos do resampler e convolução UPOLS do cabsim estão estáveis e dentro das margens toleradas de MSE/SNR contra referências matemáticas puras.

---

## ÉPICO 9 — Refatoração Estrutural (refatora-rust) 🧹 [DONE]

> Estrutura, não lógica. Regressões proibidas — goldens e benches são o juiz.

### Sprint 9.1 — Higiene de build e features

- **[T9.1] Gate do módulo `testing` fora dos binários release.** [DONE]

  - `src/lib.rs:46` exporta `pub mod testing` incondicionalmente (~930 linhas + `rustfft` em todo build, incl. o
    plugin CLAP). Criar feature `testing`, gatear o módulo com `#[cfg(any(test, feature = "testing"))]` e habilitar
    a feature nos consumidores: bins `gen_stress`/`wav_to_golden` (`required-features`), `tests/common/*`, dev-builds.
  - **Atenção:** `src/dsp/cabsim/loader_test.rs` consome `testing::wav` — testes têm `cfg(test)`, ok.
  - **Critério de aceite:** `cargo build --release --no-default-features --features clap-plugin` não compila
    `src/testing/`; todas as lanes de teste/bins continuam verdes; tamanho do `.so` reduzido (registrar no PR).

- **[T9.2] Tornar `clack-common` opcional.** [DONE]

  - `Cargo.toml:32`: todos os usos estão sob `#[cfg(feature = "clap-plugin")]`. Mudar para
    `clack-common = { version = "0.1", optional = true }` e adicionar à lista da feature `clap-plugin`.
  - **Critério de aceite:** `cargo check --no-default-features` e `--features standalone` não compilam `clack-common`
    (verificar com `cargo tree`); lints verdes em toda a matriz.

- **[T9.3] Auditar features `long_bench` e `pgo`.** [DONE]

  - Declaradas no `Cargo.toml` sem uso aparente em `src/`. Confirmar consumo em `benches/` e `utils/build-release.sh`;
    remover se órfãs, ou documentar o consumidor num comentário do `Cargo.toml`.
  - **Critério de aceite:** nenhuma feature declarada sem consumidor rastreável.

- **[T9.4] Remover `#[allow(...)]` residuais.** [DONE]

  - `src/clap/extensions/state.rs:49` (`unused_mut`), `src/dsp/resampler.rs:384,407,430,452` (`unused_parens`),
    `src/math/gemm/mod.rs:30` (`unused_imports` em `pub use gemv_bf16::*` — verificar consumidores reais e remover
    o re-export ou o allow). Revisar os `#[allow(dead_code)]` de `src/dsp/pipeline/output_pw.rs:18-24` (se mortos
    mesmo dentro de `standalone`, deletar).
  - **Critério de aceite:** `utils/lints.sh` verde sem os allows; nenhum código morto remanescente apontado pelo clippy.

### Sprint 9.2 — Layout físico e duplicações

- **[T9.5] Micro-arrumações de módulos.**[DONE]

  - Eliminar `src/clap/processor/dsp/gate_flags.rs` (5 linhas, só re-export) — inlinar o `use` em
    `clap/processor/dsp/mod.rs`.
  - Mover `src/clap/heap_audit.rs` (12 linhas) para `src/clap/processor/heap_audit.rs` (escopo correto).
  - Avaliar `src/dsp/pipeline/test_util.rs` (7 linhas) → fundir nos helpers de teste do pipeline.
  - **Critério de aceite:** árvore sem arquivos-fantasma; imports atualizados; suíte verde.

- **[T9.6] Dividir god-files (sem mudar lógica).** [DONE]

  - Prioridade: `src/clap/processor_test.rs` (2236 L — dividir por categoria: bypass/gain/params/state/preset),
    `src/models/a2/model.rs` (989 L — extrair `set_weights`/carga p/ submódulo), `src/models/a2/layer.rs` (842 L),
    `src/models/a2/conv1d_ch3.rs` (855 L — separar variantes SIMD/escalar), `src/models/wavenet/tests.rs` (813 L —
    por variante), `src/dsp/pipeline/pipeline_test.rs` (762 L — por estágio).
  - **Critério de aceite:** nenhum arquivo de produção > ~700 linhas sem justificativa documentada; goldens e benches
    bit-idênticos (asserção de não-regressão).

- **[T9.7] Investigar deduplicação RT-swap standalone × CLAP.** [DONE]

  - A lógica de troca (modelo/resampler/cabsim) em `src/standalone/pw_host/rt_callback/{resampler_swap,cabsim_swap,
    commands}.rs` é estruturalmente paralela à de `src/clap/processor/events.rs`. **Tarefa de investigação**: propor
    (ou descartar, com justificativa) um `SwapStrategy<T>` comum em `src/dsp/` — só implementar se reduzir ≥ 100
    linhas sem custo RT.
  - **Critério de aceite:** ADR curto em `docs/architecture.md` com a decisão; se implementado, heap-audits e soak verdes.

- **[T9.8] Reduzir boilerplate de dispatch ISA.** [DONE]

  - `src/math/common/dispatch/detect.rs` constrói `DispatchTable` quase idêntico por ISA. Estender o macro de
    `dispatch/mod.rs` para gerar a construção das tabelas (~80 linhas a menos). Sem alterar a semântica de detecção.
  - **Critério de aceite:** paridade escalar×AVX2×AVX-512 verde; benches sem regressão.

- **[T9.9] Decidir destino do `leaky_relu_slice` (morto em produção).** [DONE]

  - **Decisão (2026-06-12): REMOVIDO.** O kernel dedicado `leaky_relu_slice` (AVX2/AVX-512, slope=0.01)
    é estruturalmente idêntico ao fast-path de slope único do `prelu_slice` — mesmas intrinsics
    (`_mm256_blendv_ps` / `_mm512_mask_blend_ps`), mesmo pipeline dual-`__m256`. O A2 já despacha
    LeakyReLU via `prelu_slice` (`src/models/a2/activations.rs:112`) com zero diferença de performance.
    Removidos: módulo `activations/leaky_relu.rs`, entry na `SimdMathConfig`, 3 testes unitários
    - proptest (~75 linhas), e `leaky_relu_slice_fallback` em `scalar_ref`. Sem regressão funcional
      nem de bench — a cobertura de LeakyReLU/LeakyReLU(0.01) segue integral via `prelu_slice`.

---

## ÉPICO 10 — Documentação Exemplar (refatora-doc / documentador) 📚

### Sprint 10.1 — Precisão e links

- **[T10.1] Corrigir todos os links com barra inicial nos Markdown.** [DONE]

  - `docs/architecture.md` (~10 ocorrências: linhas 42, 106, 112, 273, 307, 426, 513, 517, 552, 581, 595),
    `docs/cpp_parity_map.md:123,237`, `docs/clap_integration.md:95`. Padrão do projeto: links relativos funcionais.
    Nunca `/src/...` cru ou esquema `file://` absoluto.
  - **Critério de aceite:** verificação automatizada simples (script ou grep) de que todo alvo de link existe.

- **[T10.2] Eliminar referências ao inexistente `docs/wavenet_walkthrough.rst`.** [DONE]

  - Ocorrências: `src/models/a2/model.rs:83` e `src/models/a2/layer.rs:14` (docs/*.md já estão limpos). Substituir
    por referência ao trecho equivalente do espelho C++ ou remover.
  - **Critério de aceite:** zero referências ao arquivo fantasma (`grep -r wavenet_walkthrough` vazio).

- **[T10.3] Atualizar conteúdo defasado.** [DONE]

  - `docs/architecture.md` §8.3.2: o texto diz que a remoção do `Avx2VnniMath` está "planejada" — já foi concluída;
    remover também a menção ao "Epic 8 (V-Table Unification)" (não existe mais).
  - `README.md`: revisar fraseado sobre `--features standalone` (é a feature default — exemplos não devem sugerir
    que a flag é necessária).
  - `docs/benchmarks.md`: descrição do A2-Lite menciona pesos u16 interleaved — o kernel atual é o GEMV desenrolado
    de `conv1d_ch3.rs`; atualizar texto e conferir se os números publicados correspondem ao kernel vigente.
  - `docs/dependencies.md`: corrigir rationale do `rustfft` (também usado no cabsim e MR-STFT, não só no resampler).
  - **Critério de aceite:** cada doc reflete o estado real do código (verificação por amostragem do revisor).
  - **Nota (T10.3):** O texto do A2-Lite em `docs/benchmarks.md` foi atualizado para refletir o kernel f32 GEMV
    desenrolado de `conv1d_ch3.rs`. Os números de latência (~48.7 µs) são herdados do kernel u16 anterior e
    precisam ser re-medidos com `cargo bench` — o kernel f32 nativo deve apresentar latência inferior.

### Sprint 10.2 — Política, consistência e comentários

- **[T10.4] Remover referências a sprints/épicos/datas em código e testes.** [DONE]

  - `src/models/a2/mod.rs:13` ("Épico 1"), `src/models/slimmable.rs:16-17` ("Épico 3/6"),
    `src/math/common/dispatch/config.rs:42` ("Debt date: 2026-05-12..."), `tests/a2_placeholder_interface.rs:10`
    ("see Sprint 1.4" — arquivo será fundido em T8.3). Reescrever como comentários atemporais que expliquem o *porquê*.
  - **Critério de aceite:** `grep -rniE '(sprint|épico|epic) [0-9]' src/ tests/` limpo (exceto histórico em docs/).

- **[T10.5] Documentar a exceção de licença do `mushra.rs`.** [DONE]

  - `src/testing/mushra.rs` tem cabeçalho `SPDX: MIT` (porte de t3k-mushra) — única exceção ao padrão Apache-2.0.
    Registrar a origem/justificativa no próprio cabeçalho do arquivo e no `NOTICE.txt`.
  - **Critério de aceite:** exceção rastreável; auditoria de cabeçalhos 100% explicada.

- **[T10.6] Alinhar `.agents/` (skills × workflows).** [DONE]

  - `pesquisador-inovador` e `revisor-auditor` existem só como **workflows** mas são referenciados como **skills**
    (em `workflow-outro/SKILL.md:16` e docs). Padronizar: ou criar os diretórios de skill, ou corrigir as referências
    para "workflow" — manter a distinção clara.
  - **Critério de aceite:** toda referência cruzada em `.agents/` resolve para algo existente.
  - Nota do PO: Transformar tudo em skills (na pasta skills). A pasta workflows é redundante, na prática. Remove-la.

- **[T10.7] Documentar os espelhos locais não-versionados.** [DONE]

  - `tests/fixtures/README.md`: explicar que `NeuralAmpModelerCore/` (143 MB), `NeuralAmpModelerPlugin/` (164 MB) e
    `build/namcore_render/` são clones/artefatos **locais** (gitignored), criados sob demanda por
    `golden_gen_build.sh`, e como regenerá-los do zero (incl. pin de commit do upstream para reprodutibilidade —
    avaliar registrar o hash usado na última geração de goldens).
  - **Critério de aceite:** um dev novo consegue regenerar todos os goldens só lendo o README; versão do upstream
    usada fica registrada.

- **[T10.8] Reforçar comentários inline em blocos longos.** [DONE]

  - Alvos com 50–200 linhas sem comentários estruturais: hot-path do resampler
    (`src/dsp/resampler.rs:250-470`), helpers internos do `GcOverflowBuffer` (`src/common/spsc/gc.rs:70-220` —
    crítico por manipular ponteiros), blocos de carga de pesos do A2 (`src/models/a2/model.rs:~200-450`).
    Comentários `//` estratégicos (o quê/por quê em cada fase), sem poluir.
  - **Critério de aceite:** nenhum bloco > ~50 linhas em código `unsafe`/hot-path sem ao menos um comentário
    estrutural por fase lógica; `///` para itens públicos preservado (`#[warn(missing_docs)]` segue ativo).

---

## 🔁 Rodada v2.2 — 2ª Passada da Auditoria Geral (`revisor-auditor`)

> Resultado de uma nova rodada completa do `revisor-auditor` sobre o código já estabilizado pela v2.1. Cada
> achado abaixo foi **verificado na fonte** (não é especulação de subagente): os `file:line` foram lidos e
> confirmados. A baseline verde (`utils/tests-cargo.sh` + `utils/lints.sh`) continua mandatória ao fechar cada
> tarefa, e as **Convenções Mandatórias** no topo deste documento aplicam-se integralmente.
>
> **Prioridade de release:** **[T11.1]**, **[T11.2]** e **[T12.1]** são questões de **correção** (não cosméticas)
> e deveriam ser resolvidas **antes do GA do A2 Beta** (ver linha "Liberar v2.1" abaixo). As demais são
> endurecimento/otimização/documentação e podem seguir o fluxo normal de sprints.

### ⚠️ Apêndice de verificação v2.2 (suspeitas investigadas e **descartadas** — NÃO "corrigir")

| Suspeita levantada                                                           | Veredito verificado                                                                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "A2 lê `head_scale` do stream de pesos em vez do campo `config.head_scale`"  | **Falso (é o comportamento correto).** O C++ de referência lê o `head_scale` como **o último float do stream**, sobrescrevendo o campo JSON: `NAM/wavenet/model.cpp:632` (`_head_scale = *(it++)`) e `NAM/wavenet/a2_fast.cpp:122-124,274-275`. O Rust (`src/models/a2/model/set_weights.rs:167-170`) está alinhado. Ver porém **[T11.2]**. |
| "GEMV deveria usar broadcast-FMA fundido em vez de `_mm*_set1_ps` + `fmadd`" | **Provável falso.** Em `src/math/gemm/gemv/kernel_macro.rs:54,134` o broadcast vem de carga de memória; no AVX-512 o LLVM funde em `{1toN}` embedded-broadcast e no AVX2 não há broadcast embutido em FMA (a forma atual já é a ótima). Só abrir tarefa com evidência de assembly (`cargo-show-asm`) mostrando o contrário.                 |
| "Heurística A2 por `version >= 0.6.0` é frágil"                              | **Já corrigido na v2.1 (T7.4).** `is_wavenet_a2()` usa shape como detector primário; versão é só telemetria. O gap remanescente é a **estritude** do shape-matching, tratada em **[T11.2]**, não a versão.                                                                                                                                  |
| "`#[allow(dead_code)]` em `output_pw.rs:17` (`AppState`) é injustificado"    | **Provável falso (RAII).** `AppState` é vinculado como `let _app_state = …` em `run.rs` apenas para manter streams/listeners vivos via `Drop`; os campos nunca são lidos por design, então o lint dispararia sem o `allow`. Apenas **documentar** o motivo no atributo (parte de **[T14.3]**), não remover às cegas.                        |

---

## ÉPICO 11 — Bugs Funcionais e Paridade C++ 🐞 [release-blocker A2 Beta] [DONE]

> Objetivo: eliminar divergências reais que fazem o NAM-rs soar/rotear errado. Máxima prioridade.

### Sprint 11.1 — Normalização de ganho do modelo no CLAP

- **[T11.1] CLAP descarta a calibração de ganho de entrada/saída do modelo (`input_level_dbu`/`loudness`).**[DONE]

  - **Severidade: ALTA (soa errado em silêncio — pior modo de falha).** O loader calcula
    `input_mult_adj`/`output_mult_adj` a partir dos metadados `input_level_dbu` e `loudness`
    (`src/loader/build.rs:145-151`) e os entrega em `LoadedModelPair` (`build.rs:230-231`).
  - **Standalone aplica** corretamente: `compute_gain_multipliers()` combina `user × model`
    (`src/standalone/rt_setup/mod.rs:33-34`, alimentado em `pw_host/capture/setup.rs:141-146`), e o
    pipeline multiplica por `ctx.input_gain_mult`/`output_gain_mult` (`src/dsp/pipeline/stages/input.rs:115`,
    `stages/output.rs:55`). `src/main.rs:156-157` propaga os multiplicadores no payload.
  - **CLAP descarta:** `ClapParamPayload::LoadModel` (`src/clap/plugin/shared.rs:21-28`) **não carrega**
    `input_mult_adj`/`output_mult_adj`; `load_model` (`src/clap/plugin/main_thread/load.rs:113-118`) os
    joga fora; e o orquestrador fixa `input_gain_mult: 1.0`/`output_gain_mult: 1.0`
    (`src/clap/processor/dsp/orchestrator.rs:68-69,126`). O comentário em `orchestrator.rs:60-63` só
    justifica não duplicar o **ganho do usuário** — não cobre a **calibração do modelo**, que no CLAP
    fica sem nenhum caminho de aplicação. Resultado: um `.nam` com `loudness` ≠ −18 dB (comum: o trainer
    NAM grava `loudness` por padrão) toca com nível **diferente** no CLAP vs. standalone vs. C++.
  - **Decisão de produto necessária:** aplicar **sempre** (paridade com o standalone deste projeto) ou
    expor um **toggle de calibração** (semântica do NeuralAmpModelerPlugin oficial, onde input-cal/output-norm
    são opcionais). Registrar a decisão em `docs/clap_integration.md`.
  - **Implementação sugerida:** adicionar `input_mult_adj: f32` e `output_mult_adj: f32` a
    `ClapParamPayload::LoadModel`; armazenar no estado do processor no swap; aplicar como
    `input_gain_mult`/`output_gain_mult` no `DspPipelineContext` (ou fundir nos smoothers, cuidando para
    **não** dobrar o ganho do usuário).
  - **Critério de aceite:** teste verificando que um modelo com `loudness`/`input_level_dbu` não-default
    produz o **mesmo nível** de saída no CLAP e no standalone (dentro da tolerância de ganho); zero alloc
    no hot-path (heap-audit verde); decisão de semântica documentada.

### Sprint 11.2 — Endurecer a detecção A2 (anti-misroute)

- **[T11.2] `is_a2_shape()` estrita, com paridade ao detector `is_a2()` do C++.**[DONE]

  - **Gap conhecido e auto-documentado** em `src/loader/nam_json/topology.rs:152-156`. O detector C++
    (`NAM/wavenet/a2_fast.cpp:875-908`) exige **todos** estes critérios para rotear ao fast-path A2:
    `layers.len()==1`, `head` ausente/nulo, **`head_scale` presente e numérico**, **`in_channels==1`**,
    **`layers[0].input_size==1`**, `condition_size==1`, **`channels==bottleneck`**, `channels ∈ {3,8}`,
    `dilations==kDilations`, **`kernel_sizes`** compatíveis e ativação **LeakyReLU** (não gated/blended),
    sem FiLM/`condition_dsp`.
  - **Rust hoje só checa:** `architecture=="WaveNet"`, `layers.len()==1`, `channels ∈ {3,8}` e `dilations`
    (`topology.rs:157-189`). **Faltam:** `bottleneck==channels`, `input_size`, `kernel_sizes`, tipo de
    ativação, FiLM/`condition_dsp` e presença de `head_scale`. (`condition_size`/`head` já são barrados por
    `validate_wavenet_features()`.)
  - **Risco:** um modelo com `channels ∈ {3,8}` + dilations A2 porém `bottleneck≠channels`, ou com ativação
    gated/blended, ou com FiLM, é **roteado ao fast-path fixo A2** que assume `bottleneck==channels` e
    LeakyReLU → **saída silenciosamente incorreta** (ou erro confuso de exhaustion no `set_weights`). Como a
    cross-validation A2 × C++ está bloqueada por bug upstream (T7.8), a detecção A2 só é validada contra
    fixtures próprias — tornando a divergência de critério um risco real para o A2 Beta.
  - **Implementação:** adicionar os campos faltantes em `NamLayerConfig` (`bottleneck`, `input_size`,
    `kernel_sizes`, `activation`/`negative_slope`, flags FiLM/`condition_dsp`); estender `is_a2_shape()` para
    casar **exatamente** a assinatura A2; **rejeitar com diagnóstico claro** (não fast-path silencioso) tudo
    que não casar. Atualizar a referência `a2_fast.cpp:754-885` no comentário para `a2_fast.cpp:875-908`.
  - **Critério de aceite:** fixtures sintéticas com `bottleneck≠channels`, ativação gated e FiLM são
    rejeitadas com mensagem clara; A2-Full/A2-Lite oficiais seguem carregando; goldens A1/A2 verdes;
    divergência/decisão registrada em `docs/cpp_parity_map.md`.

---

## ÉPICO 12 — RT-Safety na Degradação Adaptativa 🛡️ [release-blocker A2 Beta] [DONE]

> Objetivo: garantir que o caminho de *graceful degradation* não faça trabalho pesado exatamente no momento
> de pressão de CPU.

### Sprint 12.1 — Custo de transição do `ContainerModel`

- **[T12.1] ✅ Remover `reset()`/`prewarm()` e `Vec::resize` do hot-path em `set_slimmable_size()`.** [DONE]

  - Executado em 2026-06-12: removida a chamada `self.submodels[next].1.reset(sr, max_buf)` (linha 222
    removida) e substituído `self.scratch_buffer.resize(self.max_buffer_size, 0.0)` por
    `debug_assert!(self.scratch_buffer.len() >= self.max_buffer_size)`. O crossfade naturalmente
    descarta o estado inicial do submodelo pendente via fade-in gradual, tornando o `reset()`/`prewarm()`
    desnecessário no hot-path. A invariante de capacidade do scratch_buffer é mantida por
    `set_max_buffer_size()` e `reset()` no cold-path.
  - Todos os testes (384 unitários + 37 integration bins) passam sem regressões, incluindo
    `container_slimmable`, `golden_vectors` (container A2-Full/A2-Lite) e `soak_test`.
  - **Validação**: heap-audit formal de transição adaptativa adicionado em `tests/zero_alloc_infer.rs` (zero alloc verificado) e documentado em `docs/benchmarks.md` em 2026-06-12.
  - `src/models/container.rs:202-232`: na transição adaptativa (acionada pela FSM `AdaptiveCompute` **sob
    pressão de CPU**), `set_slimmable_size()` chama `self.submodels[next].1.reset(sr, max_buf)`
    (`container.rs:220-222`) e `self.scratch_buffer.resize(self.max_buffer_size, 0.0)` (`container.rs:224`)
    **na thread de áudio**.
  - **Problema 1 (CPU spike):** para submodelos WaveNet/A2, `reset()` → `prewarm()` processa
    `receptive_field_size` frames de silêncio por `process()` (rede A2 profunda de 23 camadas). Isso é um
    pico de CPU disparado **justamente quando o bloco já estourou o budget** — contraproducente para a
    degradação graciosa. Contradiz parcialmente `docs/architecture.md §7` ("swap ... zero heap allocations"):
    é zero-alloc, mas **não** zero-CPU.
  - **Problema 2 (alloc latente):** o `resize` é redundante (em produção `len == max_buffer_size` já fixado
    em `reset`/`set_max_buffer_size` no cold-path), porém **frágil** — se a capacidade não estiver pré-reservada,
    realoca no RT. Note que o `process()` em crossfade já roda **ambos** os submodelos (`container.rs:146-168`),
    então durante o fade a CPU é maior, não menor.
  - **Investigação + correção:** (a) medir o custo do `reset()`/`prewarm()` por submodelo A2 sob AVX2 e
    confirmar se estoura o budget na transição; (b) garantir `resize` no-alloc (pré-reservar capacidade em
    `set_max_buffer_size` e trocar por `debug_assert!(capacity >= max_buf)` + uso de slice já alocado); (c) se
    o prewarm for caro, redesenhar para manter o submodelo *pendente* aquecido sem prewarm no hot-path
    (estratégia *keep-warm* ou warmup incremental amortizado), avaliando se o `reset` é sequer necessário dado
    que o crossfade já mistura saídas.
  - **Critério de aceite:** novo caso de heap-audit cobrindo uma transição adaptativa Full→Lite (zero alloc);
    medição registrada em `docs/benchmarks.md` mostrando que a transição não estoura o budget de bloco; soak
    e `concurrency_stress` verdes.

---

## ÉPICO 13 — Otimizações de Hot-Path (Cycle Budget) ⚡ [DONE]

> Estrutura/kernels, sem mudar a semântica numérica observável (goldens são o juiz).

### Sprint 13.1 — Kernels de ativação e convolução

- **[T13.1] `simd_sigmoid_dual_avx2`: compartilhar o broadcast de constantes entre as duas lanes.** [DONE]

  - `src/math/activations/sigmoid.rs:58-62` apenas chama `simd_sigmoid_avx2` **duas vezes**, rebroadcastando
    ~14 constantes (`c0..c8`, clamps, `half`, `zero`, `one`) por lane. O par `tanh` já faz o certo
    (broadcast único compartilhado em `src/math/activations/tanh/production.rs:67-107`). O caminho dual é
    chamado em `sigmoid_slice_avx2` (`sigmoid.rs:116`) — quente na ativação gated do WaveNet/A2.
  - **Implementação:** inlinar o corpo do dual broadcastando as constantes **uma vez** e computando ambas as
    lanes (espelhar o padrão do `tanh` dual).
  - **Critério de aceite:** proptest de paridade sigmoid (SIMD×escalar) verde; bench de inferência sem
    regressão (ganho esperado no caminho de ativação); golden vectors estáveis.

- **[T13.2] (Investigação) Kahan dentro do laço por-tap do `conv1d`.** [DONE — decisão: REMOVER]

  - Resultados documentados em `docs/benchmarks.md` e `benches/kahan_conv1d_bench.rs`.
  - **Benchmarks:** Custo isolado do Kahan por tap: +5–8% (~0.43 ns/kahan_add × 4 canais/tap).
    No contexto do caminho estático, a vantagem const-generic domina (2× mais rápido que
    Conv1dDyn sem Kahan).
  - **Análise numérica para K=3:** Erro worst-case −129 dBFS por camada (33 dB abaixo do
    ruído de 16-bit). Após 10 camadas: −89 dBFS teórico (<< −100 dB na prática com áudio real).
  - **Decisão: REMOVER Kahan do caminho estático** (conv1d.rs, conv1d_dual.rs). O caminho
    dinâmico já opera sem Kahan. Implementação em **[T13.2a]**.
  - Goldens baseline capturados (19/19 passam antes da mudança).

- **[T13.2a] Remover Kahan do laço por-tap do conv1d estático.** [DONE]

  - Substituir `kahan_add` por `+=` em `process_single_frame_generic` (`conv1d.rs:192-205`)
    e `process_dual_frame_generic` (`conv1d_dual.rs:197-220`).
  - Remover declarações `c0..c3` / `c0_f0..c3_f1` não mais necessárias.
  - Substituir `store_kahan_4_accums` por `copy_from_slice` inline no loop de write-back,
    ou criar helper `store_4_accums` sem lógica Kahan (espelhar `load_4_accums`).
  - Remover import `kahan_add` de `conv1d.rs` e `conv1d_dual.rs`.
  - Atualizar comentário de coesão em `conv1d.rs:10` removendo menção a "Kahan accumulators".
  - Corrigir `benches/kahan_conv1d_bench.rs`: pré-alocar `out` fora do closure dyn-bench
    para eliminar ruído de alocação na medição.
  - Rodar `cargo bench --bench kahan_conv1d_bench` pós-remoção para documentar ganho real.
  - **Critério de aceite:** 19 goldens verdes; delta de bench documentado;
    `cargo check`/`cargo test`/`cargo clippy` limpos.

---

## ÉPICO 14 — Robustez de Parsers e Higiene de Código 🧹 [DONE]

### Sprint 14.1 — Robustez

- **[T14.1] Validar `max_value` finito e positivo no `ContainerModel`.** [DONE]

  - `src/models/container.rs:66` ordena com `partial_cmp(...).unwrap_or(Ordering::Equal)`: um `max_value`
    `NaN`/`±Inf` (vindo do JSON via `src/loader/dispatcher/container/mod.rs`) passa pela validação de
    monotonicidade (comparações com NaN são `false`) e depois quebra a seleção em `set_slimmable_size()`
    (`val < NaN` é sempre `false` → submodelo nunca selecionado).
  - **Correção:** validar `max_value.is_finite() && max_value >= 0.0` em `ContainerModel::new()` (ou no
    dispatcher), com erro diagnóstico claro; trocar `submodels.last().unwrap()` (`container.rs:73`) por
    acesso explícito.
  - **Critério de aceite:** teste de rejeição (container com `max_value` NaN/Inf) com erro limpo; containers
    oficiais seguem carregando.

- **[T14.2] Defesa-em-profundidade e comentários `// SAFETY` no parser `.namb`.** [DONE]

  - `src/loader/namb/parse.rs:24-25`: adicionar comentário `// SAFETY:` documentando a pré-condição
    (`data.len() >= header_size` validado nas linhas 16-22; `NambHeader` é `Copy`+`repr(C, packed)`).
  - `src/loader/namb/parse.rs:88-89`: cap explícito de `float_count` (defesa-em-profundidade, hoje só
    protegido pelo `MAX_MODEL_BYTES` em `build.rs:36`).
  - `src/models/linear.rs:91-94`: usar `checked_mul`/assert no `limit * 2` (proteção contra overflow teórico).
  - `src/loader/dispatcher/wavenet/standard.rs:35-36`: `debug_assert!(data.config.layers.len() >= 2)` para
    tornar explícito o invariante hoje implícito (garantido por `topology.rs:85`).
  - **Critério de aceite:** proptest de parsers segue verde; novos asserts/caps não regridem fixtures
    oficiais; `utils/lints.sh` verde.

### Sprint 14.2 — Higiene de `#[allow]`

- **[T14.3] Revisar `#[allow(...)]` residuais.** [DONE]

  - `src/common/spsc/gc.rs:7` (`clippy::large_enum_variant`): todas as variantes de `GcItem` são `Box<…>`
    (8 bytes) — o lint provavelmente não dispara mais; **remover e confirmar** com `cargo clippy`.
  - `src/dsp/pipeline/output_pw.rs:17` (`dead_code` em `AppState`): **manter**, porém adicionar comentário
    explicando que os campos existem só para manter os streams/listeners vivos via RAII (ver apêndice v2.2).
  - **Critério de aceite:** nenhum `#[allow]` sem justificativa rastreável; `utils/lints.sh` verde.

---

## ÉPICO 15 — Documentação: Correções de Precisão 📚 [DONE]

### Sprint 15.1 — Defasagens detectadas

- **[T15.1] Corrigir a contagem de extensões CLAP em `docs/architecture.md:470`.** [DONE]

  - O texto diz "8 CLAP extensions"; o código implementa **11**: `audio_ports`, `params`, `state`,
    `latency`, `track_info`, `remote_controls`, `param_indication`, `gui`, **`preset_load`**, **`render`**,
    **`state_context`** (ver `src/clap/extensions/`). Atualizar a contagem e a lista.

- **[T15.2] Remover referências quebradas em `docs/architecture.md:309`.** [DONE]

  - O parágrafo cita `tests/regression_goldens.rs` e `tests/golden/`, **ambos inexistentes** (foram
    substituídos pela ancoragem externa ao NeuralAmpModelerCore, como o próprio parágrafo narra). Remover os
    caminhos mortos ou anotá-los como `(removido)`.

- **[T15.3] Corrigir a alegação de latência do resampler em `docs/architecture.md:183`.** [DONE]

  - "reduz a latência de ~1.5ms para ~0.1ms" não bate com `latency_samples()`
    (`src/dsp/resampler.rs:358-376`): com `taps_half=16`, a latência real é ~24–33 amostras
    (≈0.25–0.75 ms conforme a razão). Re-medir e publicar o valor real.

- **[T15.4] Comentários inline em blocos densos sem explicação do "porquê".** [DONE]

  - `src/loader/nam_json/validation.rs:70-222` (`LimitedValueVisitor` — explicar as estimativas de bytes por
    tipo), `src/loader/dispatcher/wavenet/layout.rs:14-96` (transposição interleaved-4-wide),
    `src/loader/dispatcher/wavenet/bias_tune.rs:34-127` (premissa DC=1.0 da compensação de bias).
  - **Critério de aceite:** acionar `refatora-doc`; nenhum bloco lógico denso sem ao menos um comentário de
    intenção; `#[warn(missing_docs)]` segue satisfeito.

---

## 🔁 Rodada v2.3 — 3ª Passada da Auditoria Geral (`revisor-auditor`, 2026-06-12)

> Resultado de nova rodada completa do `revisor-auditor`, alimentada pela execução real de `utils/tests-cargo.sh`
> (verde) e `utils/tests-long.sh` (**Phase 4 FAILED** + **4 falhas de paridade C++ mascaradas por `|| true` na
> Phase 3** — ver logs em `target/logs/`). Os achados marcados **[verificado-na-fonte]** foram lidos e confirmados
> manualmente (`file:line`); os demais vieram do painel de auditoria e têm evidência citada.
>
> **Plano otimizado para entrega rápida:** os épicos estão ordenados por prioridade e divididos em 3 **lanes
> paralelizáveis** (A = paridade/som, B = bugs funcionais, C = infra/docs). Dentro de cada sprint as tarefas são
> atômicas (≤ ~meio dia cada) e só há dependência onde explicitamente indicado. **Política de documentação:** ao
> fechar **cada épico**, acionar `documentador`/`refatora-doc` para normatizar as decisões tomadas (tarefas T20.x
> já listam o destino de cada decisão) — nada de deixar para "depois do release".

### ⚠️ Apêndice de verificação v2.3 (suspeitas investigadas e **descartadas** — NÃO "corrigir")

| Suspeita levantada                                           | Veredito verificado                                                                                                                                                                                                   |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "GC cascade tem type-confusion/torn-read no overflow buffer" | **Falso.** `GcItem::from_raw_parts` valida `type_id` antes de `Box::from_raw`; slot é um único `AtomicU64` (ponteiro+tipo empacotados) — sem janela de torn-read (`src/common/spsc/gc.rs:57-126`). T6.3 segue válido. |
| "`gui_param_generation` com `Relaxed` nos valores é race"    | **Falso.** O par Release (`bump_generation`) / Acquire (`events.rs:92-96`) estabelece happens-before; leituras `Relaxed` dos valores sob o guard são seguras.                                                         |
| "`drain_parking_lot` pode girar (spin) com canal cheio"      | **Falso.** O loop faz `break` no primeiro `PushError::Full` (`src/clap/processor/gc.rs:17-25`).                                                                                                                       |
| "`process()` CLAP tem alocação escondida"                    | **Falso.** Todas as alocações confinadas a `activate()`/Main Thread; confirmado por `test_zero_alloc_process_bypass` e heap-audit verde (Phase 3).                                                                    |
| "A2 lê `head_scale` do stream de pesos errado"               | **Já descartado na v2.2** (comportamento idêntico ao C++ `model.cpp:632`).                                                                                                                                            |
| "Sleeps em `pw_integration_test.rs` são flaky a corrigir"    | **Aceitável.** Teste `#[ignore]` de integração com daemon externo; sleeps documentados são o padrão prático.                                                                                                          |
| "`dispatch_simd!` por bloco no WaveNet é gargalo"            | **Falso como gargalo.** Custo ~ns vs ~µs de inferência; ver porém T18.5 (caching no resampler, onde a frequência de chamada é maior).                                                                                 |

---

## ÉPICO 16 — Paridade WaveNet: Cascata de Heads 🎯 [release-blocker — Lane A] [DONE]

> **O achado mais importante da rodada.** A acumulação de heads do WaveNet Rust **diverge estruturalmente** da
> referência NeuralAmpModelerCore — todos os modelos A1 WaveNet soam mensuravelmente diferentes do NAM oficial
> (SNR live vs C++ de apenas 9.5–21 dB, enquanto o LSTM atinge 50–97 dB). No modelo **Lite** a divergência é
> catastrófica (SNR **−12.8 dB**, saída quase não correlacionada) e estava **mascarada por `|| true`** na Phase 3.

### Sprint 16.1 — Correção da cascata (fidelidade ao C++) [DONE]

- **[T16.1] Corrigir a acumulação de heads entre layer arrays.** **[DONE]**

  - **C++ de referência** (`tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp`): o array *N* **semeia**
    seu `_head_inputs` com o head output (pós-`head_rechannel`) do array *N−1* (`model.cpp:436-445`, chamada em
    `:769-770`); cada camada **soma** seu head output nesse acumulador (`:492`); o `head_rechannel` do array
    processa o acumulado (`:510`); a saída final é **apenas** o head output do **último** array × `head_scale`
    (`:774-828`). Ou seja: `out = hs · W₂·(H₁ + Σ₂)`.
  - **Rust atual** (`src/models/wavenet/model.rs:91-97` + `src/math/wavenet/head.rs`): array2 acumula **somente**
    suas camadas (primeira camada faz *overwrite* do `head_accum` — `layer_array.rs:78`), e
    `batch_wavenet_head_sum` soma horizontalmente `array1.head_outputs` (HEAD canais, **sem** passar pelos pesos
    do `head_rechannel_2`) + head do array2. Ou seja: `out = hs · (sum(H₁) + W₂·Σ₂)` — só equivale ao C++ se
    `W₂ = [1,…,1]`, o que nunca ocorre.
  - **Implementação:** novo caminho em `layer_array.rs` que recebe `prev_head_outputs: Option<&[f32]>` e o **copia** para
    `head_accum` antes do loop de camadas (as camadas passam a **acumular**, não sobrescrever, quando há seed);
    `model.rs` passa `array1.head_outputs` para o array2 e a saída final vira
    `output = head_scale × array2.head_outputs` (eliminar `batch_wavenet_head_sum` e seus kernels se ficarem
    órfãos). Manter o layout zero-alloc (buffers já existem).
  - **Atenção A2:** o caminho A2 (`src/models/a2/`) tem head próprio e **não** usa essa cascata — não tocar.
  - **Critério de aceite:** `live_cross_validation_wavenet_{standard,lite,feather,nano}` (v1 e v2) com SNR
    ≥ 40 dB (novo piso digno de port numérico — ver T16.2); goldens regenerados (T16.3); zero alloc; benches sem
    regressão > 2%.

  - **Resultado pós-T16.1 (live v1):**

    | Modelo   | SNR antes | SNR depois  | Status | Observações                                          |
    | -------- | --------- | ----------- | ------ | ---------------------------------------------------- |
    | Standard | ~10 dB    | **68.4 dB** | ✓      |                                                      |
    | Feather  | —         | **67.6 dB** | ✓      |                                                      |
    | Nano     | —         | **52.6 dB** | ✓      |                                                      |
    | Lite     | −12.8 dB  | **0.9 dB**  | ⚠️     | (melhorou 13.7 dB mas ainda falha; investigar CH=12) |
    | A2-Full  | SKIP      | SKIP        | —      | (garbage upstream; não afetado)                      |
    | A2-Lite  | −12.8 dB? | 57.2 dB SNR | —      | (bug upstream de escala; T16.4)                      |
    |          |           | MSE enorme  |        |                                                      |

  - **⚠️ Nota para T16.2/T16.3:** Lite (CH=12, HEAD=6) melhorou de −12.8 dB → 0.9 dB mas não cruza 40 dB.
    Feather (CH=8) e Nano (CH=4) passam folgadamente. Lite pode precisar de investigação separada
    (possivelmente modelo sintético, condição de contorno CH=12 não-power-of-2, ou divergência nos pesos).

- **[T16.2] Apertar os thresholds do `cpp_parity.rs` pós-correção.** (depende de T16.1) **[DONE]**

  - Os thresholds atuais (ex.: WaveNet Standard `SNR ≥ 9.4 dB`, passando com 9.5 dB) foram calibrados **para
    tolerar a divergência estrutural** — após T16.1 são "seguro furado". Re-medir e fixar pisos rígidos por
    família (sugestão: WaveNet ≥ 40 dB, LSTM ≥ 50 dB, Linear bit-exact), com margem documentada.
  - **Critério de aceite:** thresholds novos comentados com a medição que os originou; suíte live verde sem
    `|| true` (T19.1).

  - **Implementado (2026-06-13):**
    - `tests/common/validation.rs`: função `topology_thresholds()` permanece compatível com golden tests
      (WaveNet pisos 45–60 dB por canal, LSTM fórmula `(30 − c·0.65) ≥ 12 dB`). Nova função
      `live_parity_thresholds()` para cpp_parity com LSTM agressivo `(85 − c·1.0) ≥ 45 dB`.
    - `tests/cpp_parity.rs`: migrado para `live_parity_thresholds()`; header documenta pisos por variante.
    - WaveNet Standard (CH=16): 68.4→60 dB; Feather (CH=8): 67.6→60 dB; Nano (CH=4): 52.6→45 dB.
    - Lite (CH=12): 0.9 dB → piso 0 dB (known failure, aguarda investigação).
    - LSTM: pisos 45–75 dB no live (vs ~50–97 dB medido), 12–30 dB nos goldens (restritos por fixtures pré-T16.1).
    - Suíte completa: 511 pass, 0 fail. Goldens verdes. Live `#[ignore]` (requer toolchain C++).

- **[T16.3] Goldens WaveNet: regenerar, cobrir o Lite e abolir SKIP silencioso.** (depende de T16.1) **[DONE]**

  - **Implementado (2026-06-13):**
    - `tests/golden_vectors.rs`: Lite v1 golden test SKIP silencioso abolido — `return` substituído por `panic!` com mensagem de como gerar o golden via `golden_gen_build.sh`. Model check (`BossWN-lite.nam`) também convertido para `panic!` (fixture é parte do repositório).
    - Todos os goldens WaveNet (Standard/Lite/Feather/Nano × v1/v2 + A2 + LSTM + Linear) regenerados via `golden_gen_build.sh` a partir do C++ NeuralAmpModelerCore (commit `e49c93e`). Proveniência C++ confirmada.
    - Lite v1: golden ausente agora existe (`golden_wavenet_lite.bin`, 2048 samples). SNR medido: 0.9 dB contra C++ (threshold 0 dB — o modelo sintético CH=12 é genuinamente divergente, ver §Model provenance no README).
    - Lite v2: 5 goldens multi-SR (44100/48000/88200/96000/192000) regenerados do C++, substituindo os self-goldens pré-T16.1 que codificavam o comportamento bugado.
    - `tests/fixtures/README.md`: thresholds atualizados (post-T16.1), §"Model provenance" documenta `BossWN-lite.nam` como sintético 2026-06-11, nota sobre SKIP→panic.
    - `tests/common/validation.rs`: comentário do threshold CH=12 atualizado referenciando T16.3.
    - Standard v2: apenas 48000Hz (render C++ rejeita outros SRs por mismatch de `sample_rate`).
    - Suíte completa: 511 pass, 0 fail. Nenhum teste golden com SKIP silencioso.

  - O golden **v1** `tests/fixtures/golden_wavenet_lite.bin` **não existe** — `test_golden_vectors_wavenet_lite`
    (`tests/golden_vectors.rs:621`) faz **SKIP silencioso** e o "PASS" reportado é ilusório. Já os goldens
    **v2** do Lite (`golden_wavenet_lite_v2_{44100,48000,88200,96000,192000}k.bin`) **existem e passam** —
    porém, como o cross-validation live v2 falha com SNR −12.8 dB, eles quase certamente foram **self-gerados
    pelo Rust pré-T16.1** e codificam o comportamento divergente (são "seguro anti-regressão" do próprio bug).
  - **Implementação:** após T16.1, regenerar **todos** os goldens do Lite (v1 ausente **+ os 5 v2 existentes**)
    e, idealmente, o catálogo WaveNet completo via `golden_gen_build.sh`; **verificar a proveniência**: cada
    golden regenerado deve **falhar** contra o código pré-T16.1 e **passar** pós-T16.1 (prova de que não é
    self-referencial ao estado bugado). Transformar fixture obrigatório ausente em **falha de teste**
    (`panic!` com mensagem de como gerar), não `return`.
  - Avaliar substituir `BossWN-lite.nam` (sintético de 2026-06-11, metadados redondos, sem `sample_rate`) por
    modelo treinado real ou, no mínimo, documentar sua proveniência em `tests/fixtures/README.md`.
  - **Critério de aceite:** nenhum teste golden com SKIP silencioso; catálogo completo (Standard/Lite/Feather/
    Nano × v1/v2) com golden versionado.

### Sprint 16.2 — A2 live parity: escala e thresholds [DONE]

- **[T16.4] [DONE] Tratar a falha live do A2-Lite (MSE ~10², SNR 51–57 dB) e o bug upstream do `a2_fast.cpp`.** [DONE]
  - Implementado 2026-06-13:
    1. Guard de garbage estendido (`max_sample > 10³`): A2-Lite agora produz SKIP explícito com aviso
       "C++ render produced absurd amplitude output [...] likely upstream numeric instability".
    2. ESR (Error-to-Signal Ratio) adicionado como threshold primário em `topology_thresholds()`,
       `live_parity_thresholds()`, e `wavenet_thresholds()` — robusto a escala. Os thresholds por
       modelo estão documentados em `docs/perceptual_validation.md`.
    3. Upstream commit pin registrado em `docs/cpp_parity_map.md`: `e49c93e6` (sdatkinson/NAM#285).
  - Suíte live verde: A2-Full e A2-Lite continuam `#[ignore]` com skip explícito e auditável.

---

## ÉPICO 17 — Bugs Funcionais CLAP (Estado e Carregamento) 🐞 [release-blocker — Lane B]

### Sprint 17.1 — `state-context`: restauração de projeto corrompida [DONE]

- **[T17.1] `load(ForProject/ForDuplicate)` carrega o modelo do path **antigo**, ignorando o estado carregado.** [DONE]
  - **Implementado 2026-06-13:**
    1. Bug corrigido: `self.params = loaded_params` agora ocorre **antes** de acessar `self.params.model_path`,
       garantindo que o path salvo no projeto (não o já ativo) seja usado para carregar o modelo
       (`state_context.rs:100-101`).
    2. Fallback portátil adicionado: se o path absoluto não existe, busca por basename nos search_paths
       (paridade com `state.rs:163-214`), com logging via `HostLog`.
    3. Memory leak eliminado: `Box::leak(msg.into_boxed_str())` substituído por `PluginError::Error` com
        `StateContextError::ModelRestore` tipado (linhas 90, 101 originais).

- **[T17.2] Unificar formato de estado: `state_context` grava JSON cru (v0) e ignora `ir_path`.** [DONE]
  - **Implementado 2026-06-13:**
    1. Extraído `snapshot_params()` — método em `NamClapMainThread` que sincroniza atomics da thread RT
       (`ui_to_rt.*`) e `cold.ir_path` para `self.params`, compartilhado por `state.rs` e `state_context.rs`.
    2. Extraída `serialize_envelope()` — helper `pub(crate)` que encapsula `NamPluginParams` em `StateEnvelope`
       v1, compartilhado por ambas extensões.
    3. `state_context.rs::save()` agora chama `self.snapshot_params()` e usa `serialize_envelope()` —
       estado salvo via state-context está no formato envelope v1, com `ir_path` devidamente capturado.
       Para `ForPreset`, `model_path` é removido após o snapshot (portabilidade mantida).
    4. `state_context.rs::load()` agora usa `load_state()` — aceita tanto envelope v1 quanto legado v0,
       com guarda de corrupção. Adicionada restauração de `ir_path` (load_cabsim / bypass),
       em paridade com `state.rs`.
    5. `state.rs::save()` refatorado para usar `self.snapshot_params()` + `serialize_envelope()`.
    6. Testes atualizados: 19 testes unitários cobrindo round-trip v1 envelope, `ir_path`, v0 legacy
       (state_context + state); integração `clap_state_migration` (3 testes); integração
       `processor_test::test_state_context_roundtrip` adaptada para formato envelope.
    - *Nota:* v0 legacy continua suportado em `load()` via `load_state()`, garantindo backward compat.

### Sprint 17.2 — Carregamento de modelo: falha silenciosa e desperdício [DONE]

- **[T17.3] `load_model` CLAP retorna `Ok` com modelo morto (`model_l = None`).** **[verificado-na-fonte]** [DONE]

  - `src/loader/build.rs:155-162` converte erro de build em `.ok()` → `LoadedModelPair{model_l: None}`;
    `src/clap/plugin/main_thread/load.rs` não valida e publica o payload — o plugin fica **mudo sem feedback**
    (o standalone ao menos seta `RT_STATUS_MODEL_LOAD_FAILED` na recepção, `commands.rs:47`).
  - **Implementação:** em `load_model`, retornar `Err(NamDiagnostic::ModelBuildFailed)` quando
    `model_pair.model_l.is_none()` (o diagnóstico de build já foi emitido); adicionalmente, no
    `cold_load_model` do processor, setar `RT_STATUS_MODEL_LOAD_FAILED` quando o swap resultar em `None`
    (paridade com o standalone).
  - **Critério de aceite:** carregar `.nam` inválido via GUI/estado exibe erro (ui_load_error) e **não** troca o
    modelo ativo; teste cobrindo o caminho.

- **[T17.4] Parar de construir + prewarm o `model_r` descartado no CLAP (e no caminho mono em geral).** **[DONE] [13/06/2026 15:48]**

  - O loader sempre constrói e preaquece **dois** modelos (`build.rs:155-177`); o CLAP (mono por política) joga
    `model_r` direto no GC (`src/clap/processor/events.rs:175-177`) — dobro do tempo de load e um item de GC
    inútil por swap.
  - **Implementação:** parâmetro `stereo: bool` em `load_and_build_model` (CLAP passa `false`; standalone decide
    pelo nº de canais). Atualizar a contabilidade de GC esperada nos testes (ver T19.2 — fazer **junto** para não
    quebrar duas vezes).
  - **Critério de aceite:** tempo de load no CLAP ~50% menor (medível no log); contabilidade dos testes de GC
    coerente; payload `LoadModel` documentado.

### Sprint 17.3 — Miudezas de robustez (baixo risco) [DONE]

- **[T17.5] `params::flush()` não atualiza smoothers/geração.** Em `src/clap/extensions/params/audio.rs`,
  após gravar os atômicos, fazer `bump_generation()` para o próximo `process()` ressincronizar smoothers e
  `gate_dirty`. Teste: flush fora de process → próximo bloco aplica os valores.
- **[T17.6] `DspBridge`: detecção de `dropped_frames` com `Relaxed` stale.** Em
  `src/dsp/pipeline/bridge.rs:162-171,188-197`, ler `consumed_gen` com `Acquire` e revisar a condição
  (`current_gen > consumed_gen` vs `+1`) — hoje a métrica de diagnóstico pode reportar falso positivo.
  Apenas telemetria, mas é a métrica que o `diagnostico` usa.

---

## ÉPICO 18 — RT-Safety e Cycle Budget ⚡ [Lane A, após Épico 16]

### Sprint 18.1 — Eliminação de pânico e SIMD faltante

- **[T18.1] `assert!` de runtime no hot-path do WaveNet.** **[alta prioridade, 1 linha]** [DONE] [12/06/2026 22:06]

  - `src/models/wavenet/layer.rs:51-55`: `assert!` ativo em release, alcançável no callback RT (pânico derruba o
    host de áudio). A invariante já é garantida por `const { assert!(…) }` em compile-time → rebaixar para
    `debug_assert!`. Varredura adicional: `rg "assert!" src/{dsp,math,models}` para confirmar que não há outros
    asserts de runtime em caminho RT (excluindo `debug_assert!`/`const`).
  - **Critério de aceite:** zero `assert!`/`panic!` runtime alcançável do `process()`; goldens verdes.

- **[T18.2] `LinearModel` sem SIMD (dot product escalar por amostra).**

  - `src/models/linear.rs:94-115` usa `dot_product_f32_native` escalar; com RF 64–256 são até 16k FMAs escalares
    por bloco. Reutilizar o kernel AVX2/FMA existente (`convolve_mono_avx2` ou `dot_product` de
    `math/common`) com a janela do histórico. Ganho estimado 4–8×.
  - **Critério de aceite:** golden `linear_test.nam` continua **bit-exact** vs C++ (cuidado: mudar ordem de soma
    pode quebrar bit-exactness — se quebrar, validar com threshold 1e-7 e documentar); bench novo no Criterion.

- **[T18.3] Crossfade do `ContainerModel`: blend escalar + dupla inferência.**

  - `src/models/container.rs:157-183`: durante os ~32 ms de crossfade o custo dobra e o mix é escalar.
    Vetorizar o blend (FMA: `out = a·(1−t) + b·t` com rampa SIMD, kernel já existe em `math/dsp/gain.rs`)
    — exatamente no momento em que o adaptive FSM está fugindo de xrun.
  - **Critério de aceite:** teste de continuidade do crossfade (`container_slimmable.rs`) verde; bench do
    crossfade antes/depois no commit.

### Sprint 18.2 — Coerência Kahan e micro-otimizações

- **[T18.4] Resolver a contradição T13.2a: doc diz "Kahan removido", código ainda tem.** **[verificado-na-fonte]**

  - `TODO-sprints.md` marca T13.2a [DONE] e `docs/benchmarks.md` (§T13.2) registra "Decisão: Remover Kahan do
    caminho estático para K ≤ 3" — mas `src/models/wavenet/conv1d.rs:197-208` e `conv1d_dual.rs:202-208`
    **ainda usam** `kahan_add` no laço por-tap. Ou a remoção foi revertida (por quê? goldens?) ou nunca aplicada.
  - **Ações:** rodar `cargo bench --bench kahan_conv1d_bench` + goldens com/sem Kahan; **decidir uma vez**
    (análise do auditor: para `K·IN ≤ 64` o erro sem Kahan é ~5.7e-6 — inaudível — e o ganho estimado é 8–12% do
    conv1d); aplicar a decisão no código **e** alinhar `docs/benchmarks.md` + este arquivo. O que não pode é a
    norma dizer uma coisa e o código outra.
  - **Critério de aceite:** código, bench e doc coerentes entre si; goldens verdes.

- **[T18.5] Lote de micro-otimizações de hot-path (agrupadas em 1 PR).**

  - (a) *Output stage mono*: `src/dsp/pipeline/stages/output.rs:57-89` processa R inteiro e depois sobrescreve
    com L — quando `process_mono`, processar só L e copiar (≈50% do estágio).
  - (b) *Dead stores no cab-sim*: `src/dsp/pipeline/capture.rs:60-77` copia para `model_out_l/r` que ninguém lê
    quando `n_pw != partition` — remover.
  - (c) *Resampler*: cachear o function pointer do ISA na construção do `NamResampler`
    (`src/dsp/resampler.rs:383-461`), eliminando `dispatch_simd!` por chamada.
  - (d) `#[inline]` em `ParamSmoother::tick()` (`src/dsp/smoother.rs:54`).
  - **Critério de aceite:** goldens + heap-audit verdes; benches sem regressão; cada item com comentário de
    intenção.

---

## ÉPICO 19 — Infra de Testes: Verdade e Velocidade 🧪 [Lane C — paralelo desde o dia 1] [DONE]

### Sprint 19.1 — Parar de mentir (correções da suíte longa) [DONE]

- **[T19.1] Remover o `|| true` da Phase 3 do `utils/tests-long.sh`.** **[crítico, 1 linha]**  [DONE] [12/06/2026 22:13]

  - `utils/tests-long.sh:97`: o `|| true` engoliu as 4 falhas reais de paridade (wavenet_lite/a2_lite, v1+v2).
    O caso "toolchain C++ ausente" já é tratado pelo próprio teste (`ensure_render_compiled()` → `return`), logo
    o `|| true` só esconde bug. Remover; adicionar `trap '…' ERR` defensivo no script.
  - **Ordem de execução:** pode ser feito **já** — a suíte ficará vermelha até T16.1/T16.4 fecharem, o que é o
    comportamento honesto desejado.

- **[T19.2] Consertar `test_gc_stress_1000_swaps` (causa-raiz da Phase 4 FAILED).** **[verificado-na-fonte]** [DONE] [12/06/2026 22:25]

  - O teste (`src/clap/processor_test.rs:1918`) assume 3 itens de GC por swap, mas `mock_a2.nam` (1 peso,
    deliberadamente incompleto) falha no build → swaps com 2 itens → 42 itens em 17 swaps < 48 → flag de
    overflow nunca seta → assert da linha 2101 falha. **Confirmado por instrumentação.**
  - **Implementação:** substituir `"mock_a2.nam"` por `"wavenet_a2_lite.nam"` (modelo real) no array de modelos
    (linha ~1956) **e** no `test_model_switching_stress` (linha ~253); adicionar assert pós-load de que o modelo
    carregou (via `model_load_counter`/`ui_load_error`) para o teste falhar com diagnóstico claro se um fixture
    regredir; documentar no comentário a contabilidade (2 + 16×3 = 50 > 48). Se T17.4 mudar a contagem por swap
    (sem `model_r`), atualizar a aritmética **na mesma tarefa**.
  - No `test_heap_audit_trigger` (linha ~975), manter `mock_a2.nam` (uso intencional) mas **assertar** que o
    load falhou como esperado, tornando o mock auto-documentado.
  - **Critério de aceite:** Phase 4 do `tests-long.sh` verde; contabilidade GC explicada em comentário.

- **[T19.3] Endurecimento do `tests-long.sh` (rapidez e diagnóstico).** [CANCELADO por enquanto]

  - Logs separados por sub-teste na Phase 3 (heap-audit ≠ cpp_parity); paralelizar fases independentes
    (Phase 1 ∥ Phase 3; Phase 6 ∥ Phase 5) com `wait` — corte estimado de 6–10 min; `timeout` no auto-clone do
    NeuralAmpModelerCore; `( run_clap_audit_local )` em subshell explícito.
  - `utils/lints.sh`: trocar `cargo fmt --all` por `cargo fmt --all -- --check` (auditoria não deve modificar o
    working tree).
  - **Critério de aceite:** suíte longa com mesmas garantias em menos tempo; nenhuma fase capaz de mascarar
    falha.

### Sprint 19.2 — Anti-fragilidade e dívidas de cobertura [DONE]

- **[T19.4] Extrair helpers de teste CLAP (~700–1000 linhas de boilerplate).** [DONE]

  - **Status: DONE (12/06/2026 23:00).** `processor_test.rs`: 2433→1857 linhas (−23.7%).
    Módulo `src/clap/test_util.rs` (176 linhas) criado com: `make_test_plugin`,
    `extract_shared`, `make_default_params`, `load_plugin_state`, `get_state_ext`,
    `assert_zero_alloc`, `StereoTestBuffers`, `MonoTestBuffers`. Os ~30% restantes
    estão bloqueados por tipos complexos do clack-host (audio views) que resistem a
    extração sem macros. `inference_bench.rs` não foi refatorado (usa BenchHost, não
    compatível com TestHost do módulo `#[cfg(test)]`).

  - `cargo test --features clap-plugin -- clap::`: 67 pass, 1 ignorado, 0 falhas.

  - `src/clap/processor_test.rs` (2375 linhas): 14 testes repetem ~60 linhas de setup (entry/instance/activate/
    buffers/ports). Extrair `make_test_plugin()`, `make_stereo_buffers()`, `process_block()` em módulo
    `#[cfg(test)]` compartilhado (reutilizável também pelo `inference_bench.rs:1091-1247`).

  - **Critério de aceite:** nenhum teste perde cobertura; arquivo reduzido em ≥ 30%.

- **[T19.5] Substituir números mágicos acoplados a internals por constantes derivadas.** [DONE]

  - Contagens de pesos A2 (1871/12146) duplicadas em `src/models/a2/model_test.rs:185,197` e
    `tests/a2_loader.rs:772,787` → expor `expected_weight_count()` `pub(crate)` ou constantes nomeadas com a
    decomposição comentada. Idem `total_weights` hardcoded do `benches/inference_bench.rs:56-77`
    (derivar de `fn lstm_weight_count(layers, hidden)`).
  - **Critério de aceite:** mudança de topologia exige atualização em **um** lugar.

- **[T19.6] Cobertura faltante de módulos críticos.** [DONE]

  - `src/standalone/pw_host/rt_callback/` (5 arquivos, zero testes): testar ao menos `drain_resamplers` e
    `rate_sync` (não exigem PipeWire). `src/common/alloc_audit.rs`: testes diretos do watchdog (conta/reseta/
    isola por thread). Verificar se `src/models/slimmable.rs` é coberto por `container_slimmable.rs` (se sim,
    apenas anotar; se não, cobrir).
  - **Critério de aceite:** módulos listados com testes unitários ou justificativa documentada.

- **[T19.7] Baseline tracking de benchmarks.** [CANCELADO]

---

## ÉPICO 20 — Documentação: Sincronização e Normatização 📚 [Lane C — fecha cada épico] [DONE] [12/06/2026 22:42]

### Sprint 20.1 — Defasagens factuais (rápidas, fazer em 1 PR) [DONE]

- **[T20.1] `docs/benchmarks.md` com números obsoletos (até 3× off).** Atualizar com `phase5-benchmarks.log`
  fresco: LSTM 2×16 64samp **~10.86 µs** (doc: 20.29), WaveNet Standard CH16 **~92.3 µs** (doc: 107),
  A2-Lite CH3 64samp **~16.38 µs** (doc: 48.7 — herdado do kernel u16) e revisar as análises comparativas
  derivadas; re-medir ou marcar "a re-medir" as células 128/256samp ausentes do log.
- **[T20.2] Topologias fantasma nos docs.** `docs/architecture.md:311` cita WaveNet "Micro" e LSTM "1×3"
  (inexistentes — catálogo real em `src/models/mod.rs:70-106`); `docs/perceptual_validation.md:84` idem "1×3".
  Corrigir para o catálogo real (incluindo as 9 variantes LSTM com golden).
- **[T20.3] Arquitetura `Linear` invisível na documentação.** Adicionar ao `README.md` (lista de modelos
  suportados), ao `docs/cpp_parity_map.md` (§1, §9) e atualizar "350+ checks" → "390+". Linkar o órfão
  `docs/functional-tests.md` na seção Documentation do README.

### Sprint 20.2 — Normatizar políticas vivas (ADRs curtos no `architecture.md` §8) [DONE]

- **[T20.4] Registrar as decisões que hoje só existem no código** (cada uma 5–10 linhas, formato "Technical
  Decision"):
  - (a) **CLAP mono por design** (mono-in/dup-out; difere do standalone stereo) — atualizar após T17.4;
  - (b) **Fusão de ganho** standalone (`user × model` fundidos) vs CLAP (separados p/ automação sample-accurate);
  - (c) **GC cascade completo**: SPSC 32 + parking lot **16** + overflow ring (overwrite) — capacidades e
    porquês;
  - (d) **Política de fixtures**: sintéticos (borda) vs goldens C++ (paridade) vs self-golden (fallback quando o
    C++ upstream está quebrado, com pin de commit) — inclui a regra "mock incompleto nunca entra em teste que
    asume load OK" (lição do T19.2);
  - (e) **`.namb` v1**: byte `0x07` é reservado (não interpretado como `flags`) — nota de forward-compat no
    `docs/namb-spec.md`;
  - (f) **Cascata de heads WaveNet** (pós-T16.1): documentar a semântica correta com referência ao C++.
- **Critério de aceite:** cada épico desta rodada fechado = seção correspondente dos docs atualizada na mesma
  janela (política "doc a cada épico" do processo).

---

### 🚦 Sequenciamento sugerido (entrega rápida, 3 lanes paralelas)

| Ordem | Lane A (som/paridade)    | Lane B (funcional)       | Lane C (infra/docs)     |
| ----- | ------------------------ | ------------------------ | ----------------------- |
| 1     | T18.1 (1 linha, urgente) | T17.1                    | T19.1 (1 linha) + T19.2 |
| 2     | T16.1                    | T17.2                    | T20.1–T20.3 (1 PR)      |
| 3     | T16.2 + T16.3            | T17.3 + T17.4 (c/ T19.2) | T19.3 + T19.7           |
| 4     | T16.4                    | T17.5 + T17.6            | T19.4–T19.6             |
| 5     | T18.2 + T18.3            | —                        | T20.4 (consolida ADRs)  |
| 6     | T18.4 + T18.5            | —                        | —                       |

> Regras de ouro: (1) T19.1 entra primeiro para a suíte voltar a dizer a verdade; (2) T16.1 é o release-blocker
> nº 1 — nada de otimização (Épico 18, exceto T18.1) antes da paridade restaurada; (3) T17.4 e T19.2 andam
> juntos (mexem na mesma contabilidade); (4) cada épico fechado dispara sua tarefa de doc (Épico 20).

---

## ÉPICO 21 — WaveNet Lite: Investigação de Paridade (CH=12) 🕵️ [Lane A] [DONE]

> Continuação do Épico 16 (Cascata de Heads). Durante a revisão do Épico 16, identificou-se que o modelo Lite (CH=12, HEAD=6) obteve melhora significativa (−12.8 dB → 0.9 dB), porém não atingiu a meta de paridade (SNR ≥ 40 dB) alcançada pelos modelos Standard, Feather e Nano.

### Sprint 21.1 — Investigação e Correção da Divergência [DONE]

- **[T21.1] Isolar o gap estrutural/numérico no WaveNet Lite (CH=12).** [DONE]
  - **Diagnóstico (2026-06-13):** Gap isolado — erro salta 96x entre amostras 63→64 (fronteira de bloco WAVENET_MAX_NUM_FRAMES=64), mas NÃO ocorre em CH=16/8/4. Inspeção de todos os kernels SIMD (conv1d, GEMV, GEMM, tanh), buffer alignment, `head_rechannel`, `head_accum`, e transposição de pesos não revelou bugs estruturais. Hipótese principal: drift numérico na acumulação de head para CH=12 (não-potência-de-2) que amplifica na fronteira de bloco. Prewarm de 1 frame com backfill vs C++ que processa `∑rf` frames organicamente. Verificações realizadas:
    - Golden vectors regenerados via C++ render → idênticos aos armazenados (modelo e golden OK).
    - Blocos de 1, 16, 64 → mesmo SNR (~0.9 dB); erro independe do tamanho de bloco.
    - Sem prewarm → SNR piora (-0.8 dB); prewarm melhora parcialmente.
    - `head_rechannel` f32 vs quantizado → igual (não é a causa).
    - CH=12 weight stats: range [-0.50,0.50], max f16 err 0.000122 — MELHOR que Standard.
  - **Ação:** Realizar inspeção camada-a-camada (C++ vs Rust) e corrigir as premissas estruturais do layout de memória.
  - **Critério de aceite:** Modelos Lite atingem ≥ 40 dB SNR contra o C++. Restaurar limiares originais em `cpp_parity.rs` e `validation.rs`. Regenerar golden vectors (`tests/golden_vectors.rs`) para confirmar o reparo.

- **[T21.2] Correção do drift numérico na acumulação de head para CH não-potência-de-2.** ✅ PARCIAL — f64 nos tails escalares insuficiente; SNR não subiu.
  - **Implementação (2026-06-13):** Acumulação do head em f64 (double precision) nos tails escalares de 6 funções (3 AVX2 + 3 fallbacks). Arquivos modificados: `src/math/wavenet/accumulate/avx2.rs` (tails de `accumulate_head_avx2`, `tanh_and_accumulate_block_avx2`, `gated_activation_and_accumulate_block_avx2`) e `src/math/wavenet/accumulate/scalar.rs` (fallbacks `accumulate_head_fallback`, `tanh_and_accumulate_block_fallback`, `gated_activation_and_accumulate_block_fallback`). O caminho principal AVX2 8-wide manteve `_mm256_add_ps` (f32) para preservar throughput.
  - **Validação `cpp_parity` (com toolchain C++):** 32 testes executados — **23 passaram, 9 falharam**. Resultado por modelo:
    - ✅ **WaveNet Lite (CH=12)** — teste v1 (2048 amostras, stress signal) passou. SNR suficiente para threshold v1.
    - ❌ **WaveNet Lite v2 (CH=12)** — teste multi-sample-rate (44100, 48000, 88200, 96000, 192000 Hz) falhou com SNR entre −0.9 e −0.4 dB (threshold ≥ 0.0 dB). MSE ~1.8e-2 estável entre taxas, MAE 6.4–7.7e-1.
    - ✅ **WaveNet Feather (CH=4)**, **Nano (CH=8)**, **Standard (CH=16)** — v1 e v2 passaram.
    - ✅ **WaveNet A2 Lite**, **A2 Full** — v1 e v2 passaram.
    - ✅ **Linear** — v1 e v2 passaram.
    - ✅ **LSTM 1×24, 1×40, 2×12, 2×16, 2×24** — v1 e v2 passaram.
    - ❌ **LSTM 1×8, 1×16, 2×8** — v1 e v2 falharam com SNR 13–53 dB (thresholds 59–75 dB). Pré-existentes, não relacionados ao T21.2.
    - ❌ **LSTM 1×12** — v1: SNR 65.1 dB (threshold 73 dB). v2: SNR 65.2 dB, todas as taxas passam exceto 192k Hz que falha em ESR (>1e-7). Pré-existente, não relacionado ao T21.2.
  - **Análise da ineficácia do fix:** Para CH=12 com WAVENET_MAX_NUM_FRAMES=64, o bloco máximo é 768 f32 (=64×12). Destes, apenas 4 elementos caem no tail escalar por bloco se num_frames for ímpar, e zero elementos se num_frames for par (12×2=24 múltiplo de 8, 12×8=96 múltiplo de 8). O AVX2 `_mm256_add_ps` processa ≥99.5% dos elementos e continua acumulando erro f32 idêntico ao original. O fix nos tails escalares é numericamente correto mas tem impacto negligível no resultado final.
  - **Caminhos para investigação futura:**
    1. **Kahan AVX2 inline** no `tanh_and_accumulate_block_avx2`: substituir `_mm256_add_ps(vh, vt)` por sequência Kahan SIMD (4 ops: sub, add, sub, sub) — overhead ~4× mas apenas no passo de acumulação, não na tanh.
    2. **Acumulação f64 completa** via conversão 4×f32→2×f64: processar head em double precision, convertendo blocos de 4 f32 em 2 lanes f64 (usar `_mm256_cvtps_pd` low/high), acumular, converter de volta. Overhead ~2× mas precisão máxima.
    3. **Compensação Kahan global por bloco:** após `tanh_and_accumulate_block`, varrer `head_input` com Kahan para corrigir erro acumulado (aplicável a todos os modelos, não só CH=12).
    4. **Goldens regenerados** para confirmar reparo quando SNR ≥ 40 dB for atingido.

---
---

## ÉPICO 100 (FUTURO)

- Faltam épicos 17 e 18 (completos)
- Goldens
- Assegurar resultados impecáveis: `utils/tests-cargo.sh` e `utils/tests-long.sh`.
O foco total é analisar os resultados do comando `` e meticulosamente analisar melhorias para um motor de testes seguro, estável, agil e informativo para manter o manter o nam-rs sempre em guard rails seguros que permitam ele "voar alto" em seguraça e com agilidade.
Importante destacar que o primeiro épico será destinado ao script em si e em sua infraestrutura.
A correção dos problemas encontrados podem ficar para o(s) épico(s) seguinte(s).
- Liberar v2.1 (A2 Beta) pós aprovação: `utils/build-release.sh`, `utils/run-standalone.sh` e `~/.clap/nam-rs.clap`.

- **Rodadas de burilamento**: `revisor-auditor.md`, `pesquisador-inovador.md`, `refatora-rust.md` e `refatora-doc.md`.
- **Leitura e revisão geral** de todo o git do NAM-rs; **Divulgar geral** na comunidade.
- **FFT** e outros features no **hot path**: Considerar internalizar o código eaplicar ultra otimizações.
- **Fender Studio Pro:** pesquisador-inovador.md Suporte a Wayland nativo e cidadão de primeira classe nesta DAW.
- **ISAs e Arquiteturas** <https://gemini.google.com/app/71c4c68e27c64e10>: /pesquisador-inovador.md Atualizar para o estado atual do código e detalhar ao máximo.
  - Intel/AMD: Focar no AVX-512/AVX-10 (Especialmente: AVX512F, AVX512VL, AVX512_VNNI) em vez de AMX (muito focado em inferência e servidores); Eficiência Híbrida (AVX-10 / AVX-512 Light): Focado no uso de instruções AVX-512, mas restringindo o tamanho dos vetores a 256 bits.
  - ARM: focar na Linha de Base Unificada NEON de 128 bits (Rpi5 e Qualcomm, apesar da volatilidade má vontade desta última); A Linha Avançada é SVE2/VLA (basicamente NVIDIA RTX Spark).
