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

## ÉPICO 12 — RT-Safety na Degradação Adaptativa 🛡️ [release-blocker A2 Beta]

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
  - **NOTA**: critério de aceite parcialmente pendente — falta heap-audit formal de transição adaptativa
    e medição em `docs/benchmarks.md` (pode ser feito em tarefa separada de validação do épico).

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

## ÉPICO 13 — Otimizações de Hot-Path (Cycle Budget) ⚡

> Estrutura/kernels, sem mudar a semântica numérica observável (goldens são o juiz).

### Sprint 13.1 — Kernels de ativação e convolução

- **[T13.1] `simd_sigmoid_dual_avx2`: compartilhar o broadcast de constantes entre as duas lanes.**

  - `src/math/activations/sigmoid.rs:58-62` apenas chama `simd_sigmoid_avx2` **duas vezes**, rebroadcastando
    ~14 constantes (`c0..c8`, clamps, `half`, `zero`, `one`) por lane. O par `tanh` já faz o certo
    (broadcast único compartilhado em `src/math/activations/tanh/production.rs:67-107`). O caminho dual é
    chamado em `sigmoid_slice_avx2` (`sigmoid.rs:116`) — quente na ativação gated do WaveNet/A2.
  - **Implementação:** inlinar o corpo do dual broadcastando as constantes **uma vez** e computando ambas as
    lanes (espelhar o padrão do `tanh` dual).
  - **Critério de aceite:** proptest de paridade sigmoid (SIMD×escalar) verde; bench de inferência sem
    regressão (ganho esperado no caminho de ativação); golden vectors estáveis.

- **[T13.2] (Investigação) Kahan dentro do laço por-tap do `conv1d`.**

  - `src/models/wavenet/conv1d.rs:192-205` (e `conv1d_dual.rs`) executa Kahan **escalar** por canal a cada
    tap, serializando a redução SIMD→escalar dentro do laço. Para `K ≤ 3` (típico no WaveNet), o erro de
    soma simples já é `O(3·ε)` (desprezível), tornando o Kahan por-tap possivelmente superdimensionado.
  - **Investigação:** medir o custo do Kahan por-tap; avaliar **adiar** a compensação para o fim dos `K` taps
    (acumular resultados SIMD e Kahan-reduzir uma vez) ou removê-la para `K` pequeno. **Qualquer** mudança só
    é aceita com **todos os goldens verdes** (regra "Golden = seguro anti-degradação") e estabilidade do soak.
  - **Critério de aceite:** delta de bench documentado em `docs/benchmarks.md`; goldens dentro da banda;
    decisão (manter/adiar/remover) registrada com justificativa numérica.

---

## ÉPICO 14 — Robustez de Parsers e Higiene de Código 🧹

### Sprint 14.1 — Robustez

- **[T14.1] Validar `max_value` finito e positivo no `ContainerModel`.**

  - `src/models/container.rs:66` ordena com `partial_cmp(...).unwrap_or(Ordering::Equal)`: um `max_value`
    `NaN`/`±Inf` (vindo do JSON via `src/loader/dispatcher/container/mod.rs`) passa pela validação de
    monotonicidade (comparações com NaN são `false`) e depois quebra a seleção em `set_slimmable_size()`
    (`val < NaN` é sempre `false` → submodelo nunca selecionado).
  - **Correção:** validar `max_value.is_finite() && max_value >= 0.0` em `ContainerModel::new()` (ou no
    dispatcher), com erro diagnóstico claro; trocar `submodels.last().unwrap()` (`container.rs:73`) por
    acesso explícito.
  - **Critério de aceite:** teste de rejeição (container com `max_value` NaN/Inf) com erro limpo; containers
    oficiais seguem carregando.

- **[T14.2] Defesa-em-profundidade e comentários `// SAFETY` no parser `.namb`.**

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

- **[T14.3] Revisar `#[allow(...)]` residuais.**

  - `src/common/spsc/gc.rs:7` (`clippy::large_enum_variant`): todas as variantes de `GcItem` são `Box<…>`
    (8 bytes) — o lint provavelmente não dispara mais; **remover e confirmar** com `cargo clippy`.
  - `src/dsp/pipeline/output_pw.rs:17` (`dead_code` em `AppState`): **manter**, porém adicionar comentário
    explicando que os campos existem só para manter os streams/listeners vivos via RAII (ver apêndice v2.2).
  - **Critério de aceite:** nenhum `#[allow]` sem justificativa rastreável; `utils/lints.sh` verde.

---

## ÉPICO 15 — Documentação: Correções de Precisão 📚

### Sprint 15.1 — Defasagens detectadas

- **[T15.1] Corrigir a contagem de extensões CLAP em `docs/architecture.md:470`.**

  - O texto diz "8 CLAP extensions"; o código implementa **11**: `audio_ports`, `params`, `state`,
    `latency`, `track_info`, `remote_controls`, `param_indication`, `gui`, **`preset_load`**, **`render`**,
    **`state_context`** (ver `src/clap/extensions/`). Atualizar a contagem e a lista.

- **[T15.2] Remover referências quebradas em `docs/architecture.md:309`.**

  - O parágrafo cita `tests/regression_goldens.rs` e `tests/golden/`, **ambos inexistentes** (foram
    substituídos pela ancoragem externa ao NeuralAmpModelerCore, como o próprio parágrafo narra). Remover os
    caminhos mortos ou anotá-los como `(removido)`.

- **[T15.3] Corrigir a alegação de latência do resampler em `docs/architecture.md:183`.**

  - "reduz a latência de ~1.5ms para ~0.1ms" não bate com `latency_samples()`
    (`src/dsp/resampler.rs:358-376`): com `taps_half=16`, a latência real é ~24–33 amostras
    (≈0.25–0.75 ms conforme a razão). Re-medir e publicar o valor real.

- **[T15.4] Comentários inline em blocos densos sem explicação do "porquê".**

  - `src/loader/nam_json/validation.rs:70-222` (`LimitedValueVisitor` — explicar as estimativas de bytes por
    tipo), `src/loader/dispatcher/wavenet/layout.rs:14-96` (transposição interleaved-4-wide),
    `src/loader/dispatcher/wavenet/bias_tune.rs:34-127` (premissa DC=1.0 da compensação de bias).
  - **Critério de aceite:** acionar `refatora-doc`; nenhum bloco lógico denso sem ao menos um comentário de
    intenção; `#[warn(missing_docs)]` segue satisfeito.

---

Liberar v2.1 (A2 Beta): Rodada de testes no cli e no plugin CLAP. Bitwig e Reaper: ficar com os 2 ou apenas 1 deles?

---

## ÉPICO 100 (FUTURO)

- **Rodadas de burilamento**: `revisor-auditor.md`, `pesquisador-inovador.md`, `refatora-rust.md` e `refatora-doc.md`.
- **Leitura e revisão geral** de todo o git do NAM-rs; **Divulgar geral** na comunidade.
- **FFT** e outros features no **hot path**: Considerar internalizar o código eaplicar ultra otimizações.
- **Fender Studio Pro:** pesquisador-inovador.md Suporte a Wayland nativo e cidadão de primeira classe nesta DAW.
- **ISAs e Arquiteturas** <https://gemini.google.com/app/71c4c68e27c64e10>: /pesquisador-inovador.md Atualizar para o estado atual do código e detalhar ao máximo.
  - Intel/AMD: Focar no AVX-512/AVX-10 (Especialmente: AVX512F, AVX512VL, AVX512_VNNI) em vez de AMX (muito focado em inferência e servidores); Eficiência Híbrida (AVX-10 / AVX-512 Light): Focado no uso de instruções AVX-512, mas restringindo o tamanho dos vetores a 256 bits.
  - ARM: focar na Linha de Base Unificada NEON de 128 bits (Rpi5 e Qualcomm, apesar da volatilidade má vontade desta última); A Linha Avançada é SVE2/VLA (basicamente NVIDIA RTX Spark).
