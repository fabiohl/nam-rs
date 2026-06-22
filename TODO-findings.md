<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved. -->

# TODO-findings.md — Auditoria Geral NAM-rs (Revisor-Auditor)

> **Origem:** Skill `revisor-auditor` + diretrizes adicionais (análise da suíte de testes em
> `Testes.log` e `target/logs/*`, caça implacável a bugs de segurança e funcionalidade,
> deduplicação/reorganização/"embelezamento", e organização em Épicos Ágeis pela skill
> `planejador-arquiteto`).
>
> **Escopo desta entrega:** Apenas **Achados (Fn)** detalhados e **Épicos** que os agrupam.
> **Não** foram criados sprints nem tarefas técnicas (serão elaborados à parte, épico a épico).
>
> **Metodologia de confiança:** Os achados de maior severidade foram **verificados diretamente
> no código-fonte** (arquivo:linha citados). Achados provenientes de varredura ampla por
> subsistema estão marcados como tal e tiveram a severidade calibrada manualmente.

---

## 0. Resumo Executivo

O NAM-rs apresenta uma **postura de engenharia madura**: RT-safety quase impecável no hot-path,
defesas de loader acima da média (caps de tamanho, limites de recursão, CRC v2 obrigatório),
dispatch SIMD à prova de SIGILL (`is_x86_feature_detected!` com fail-fast), e uma suíte de testes
extensa (632 testes unitários/integração na lane rápida + auditoria longa de 6 fases).

A auditoria, porém, revelou **um conjunto pequeno mas relevante de gaps**, com destaque para:

| Prioridade | Tema                                                                  | Achados-chave |
|:----------:|:--------------------------------------------------------------------- |:------------- |
| 🔴 **P0**  | CI vermelho por falso-positivo no gate LUFS                           | F8            |
| 🔴 **P0**  | Pesos `NaN/Inf` aceitos sem validação (potencial dano a alto-falante) | F1            |
| 🟠 **P1**  | Alocação não-limitada em modelos dinâmicos (DoS por OOM)              | F2            |
| 🟠 **P1**  | Testes de correção críticos fora da lane rápida                       | F30           |
| 🟡 **P2**  | Bypass de CRC32 em `.namb` v1; campos float de header sem validação   | F3, F4        |
| 🟡 **P2**  | `unwrap()`/`log::error!` residuais no thread RT                       | F9, F10       |
| 🟡 **P2**  | Scaffolding stereo morto no CLAP mono; `GcOverflow` ordering          | F12, F13      |
| 🟢 **P3**  | Deduplicação SIMD (~850 SLOC), split de arquivos gigantes, higiene    | F23–F29       |

### 0.1 Estado da Suíte de Testes (a partir de `Testes.log` e `target/logs/*`)

**Lane rápida (`utils/tests-quick.sh`) — VERDE:**

- 631 passou / 0 falhou / 1 ignorado (lib unit/integ).
- CLAP debug + heap-audit: todos os heap-audits zero-alloc passaram.
- `clap-validator` estrito: 21 testes, 19 passou, 2 skipped (note-ports não implementado), 0 falhas/0 warnings.

**Lane longa (`utils/tests-long.sh`) — VERMELHO em 1 de 6 fases:**

- ✅ Soak (163s), ✅ PipeWire (17s), ❌ **Property/Parity/Golden (256s)**, ✅ Heap-Audit (59s),
  ✅ CLAP Release+Concorrência (414s), ✅ Long Benchmarks (1816s).
- A **única falha** é em `tests/cpp_parity.rs`: 29 passou / **3 falhou** —
  `live_cross_validation_wavenet_dyn`, `live_cross_validation_v2_wavenet_dyn`,
  `live_cross_validation_v2_lstm_dyn`.
- **Causa-raiz (F8):** NÃO é regressão de engine. A paridade numérica é excelente
  (SNR 116–124 dB, ESR ~1e-13). A falha é um **falso-positivo do gate de plausibilidade LUFS**:
  os modelos `wavenet_dyn_free` (head_scale=0.02, saída ~−65 LUFS) e `lstm_dyn` em SR altos
  produzem saída de referência C++ legitimamente baixa. O `golden_vectors.rs` já trata isso com
  `report_dsp_fidelity_no_lufs`, mas o `cpp_parity.rs` ainda chama a variante **com** gate LUFS.

**Avaliação da qualidade dos scripts de teste:**

- `utils/tests-long.sh` é de **alta qualidade**: trap de erro defensivo, verificação de SHA pinado
  do NeuralAmpModelerCore, freshness manifest de goldens, sumário com top-N mais lentos por fase,
  e auto-build de goldens. Pontos de atenção: usa `set -uo pipefail` (sem `-e`, compensado por trap
  ERR + `|| true` por fase — correto). Bom isolamento de fases.
- **Observação estrutural (F30):** o design intencional de empurrar testes pesados para `#[ignore]`
  cria um ponto cego: a lane rápida do desenvolvedor **não** roda paridade C++, fuzzing de parsers,
  nem proptests SIMD.

---

## 1. ACHADOS (Findings)

### 1.1 — Segurança do Loader (entrada não-confiável `.nam` / `.namb`)

---

#### **F1 — Pesos e `head_scale`/`head_bias` `NaN`/`Inf` aceitos sem validação de finitude**

- **Severidade:** 🔴 **Alta** (segurança + funcionalidade; risco de ruído alto / dano a equipamento)
- **Locais (verificado):**
  - JSON: `src/loader/nam_json/validation.rs:49` — `WeightsVisitor::visit_seq` faz `weights.push(val)` sem checar `is_finite()`.
  - Binário: `src/loader/namb/parse.rs:102-104` — `weights.push(f32::from_le_bytes(...))` sem checagem.
  - `head_scale` (multiplicativo na saída): `src/loader/dispatcher/wavenet/standard.rs:62`, `.../wavenet/dynamic.rs:207`, `.../convnet/mod.rs:138`.
  - `head_bias` (aditivo): `src/loader/dispatcher/lstm/static_builder.rs:41`, `.../lstm/dynamic_builder.rs:50`, `.../linear/mod.rs:22`.
- **Evidência de ausência de defesa downstream:** varredura por `is_finite|is_nan|is_infinite`
  em código de produção (não-teste) retorna **apenas 2 usos relevantes** — `src/dsp/cabsim/loader.rs:252`
  (valida amostras do IR de cabinet) e `src/models/container.rs:69` (valida `max_value`). **Nenhum**
  cobre pesos de modelo neural. Todos os demais `is_finite` são asserções de teste.
- **Gatilho/Impacto:** Um `.nam`/`.namb` malicioso ou corrompido com peso ou `head_scale` `NaN`/`Inf`
  produz saída de inferência `NaN`. Em Rust, `f32::clamp` com `self=NaN` **retorna NaN**, logo o estágio
  de clipping de saída **não** neutraliza o problema. Resultado: rajada de ruído digital / instabilidade
  no DAW — potencial dano a alto-falantes e fadiga auditiva. O `param-fuzz-basic` do `clap-validator`
  valida apenas *parâmetros* aleatórios, **não** pesos de modelo.
- **Caminho de resolução:**
  1. JSON: dentro de `visit_seq`, após `Ok(Some(val))`, `if !val.is_finite() { return Err(JsonError::NonFiniteWeight) }`.
  2. Binário: no laço `chunks_exact(4)`, validar `is_finite()` por valor (ou validar o `Vec` final em bloco com `iter().all`).
  3. Validar `head_scale.is_finite()` e `head_bias.is_finite()` imediatamente após cada `read_f32()`.
  4. Adicionar variante de erro tipada (`JsonError::NonFiniteWeight` / `NambError::NonFiniteWeight`).
  5. Cobrir com proptest negativo (ver F30/E8): injetar `NaN`/`Inf` e exigir rejeição limpa (sem panic).
- **Custo:** Baixo. **Risco da correção:** Baixo (caminho cold de loading).

---

#### **F2 — Alocação não-limitada em construtores de modelos dinâmicos (DoS por exaustão de memória)**

- **Severidade:** 🟠 **Alta** (DoS)
- **Locais (verificado):**
  - LSTM dinâmico: `src/loader/nam_json/topology.rs:317-325` (`get_lstm_topology` retorna `(num_layers, hidden_size)` crus do JSON) → `src/loader/dispatcher/lstm/dispatch.rs:55-58` → `dynamic_builder.rs:35-39` (`Vec::with_capacity(num_layers)` + `LstmLayerDyn::new` aloca `4*hidden*(in+hidden)`).
  - WaveNet free-shape: `src/loader/nam_json/topology.rs:304-314` (`Free{ channels, ... }` sem limite superior) → `src/loader/dispatcher/wavenet/dynamic.rs:163-204` (buffers `ch * WAVENET_MAX_NUM_FRAMES`).
  - A2-Dyn: `src/loader/dispatcher/wavenet/mod.rs:89-91,123-132` (`channels`/`bottleneck` → `WaveNetA2Dyn::new`).
- **Gatilho/Impacto:** O cap global de pesos (`MAX_FLOAT_COUNT = 64M`) **não** protege as
  pré-alocações de buffer dimensionadas por `channels`/`hidden_size`/`num_layers`, pois estes são
  campos de config independentes da contagem real de pesos. Ex.: `channels=2_000_000` com poucos pesos
  faz o builder tentar alocar buffers na casa de GB → OOM/crash do host DAW antes mesmo de consumir pesos.
- **Caminho de resolução:**
  1. Em `get_lstm_topology`: rejeitar `num_layers > 16` e `hidden_size > 1024` (folga ampla sobre o maior perfil estático, H=40).
  2. Em `get_wavenet_topology` (ramo `Free`): rejeitar `channels > 512` (maior SKU é CH=16).
  3. Em A2-Dyn: validar `channels`/`bottleneck <= 256` antes de `WaveNetA2Dyn::new`.
  4. Centralizar limites em constantes nomeadas e documentadas (ex.: `MAX_DYN_CHANNELS`, `MAX_LSTM_LAYERS`).
- **Custo:** Baixo. **Risco:** Baixo. Cobrir com proptest (E8) de dimensões absurdas exigindo `Err`.

---

#### **F3 — Bypass de validação CRC32 em `.namb` v1 via sentinela `crc32 == 0`**

- **Severidade:** 🟡 **Média** (integridade)
- **Local (verificado):** `src/loader/namb/parse.rs:75-79`. Para v1, se `crc32_header == 0`, a checagem
  de integridade é **pulada** com mero `log::warn!`. Para v2 (`parse.rs:70-74`), `FLAG_HAS_CRC32` é
  obrigatório e o CRC é sempre verificado (comportamento correto).
- **Gatilho/Impacto:** Atacante cria `.namb` v1 com seção de pesos corrompida (polaridade invertida,
  níveis distorcidos, payload `NaN`) e zera o campo `crc32` para evadir a detecção. A corrupção passa
  silenciosamente.
- **Caminho de resolução:** Ou (a) sempre computar CRC sobre a seção de pesos em v1, tratando `crc32=0`
  com seção de pesos não-vazia como suspeito; ou (b) exigir seção de pesos vazia quando `crc32=0` em v1
  (espelhando a legitimidade do caso vazio em v2). Combinar com F1 (finitude) para defesa em profundidade.
- **Custo:** Baixo. **Risco:** Baixo (atenção a compatibilidade com modelos v1 legítimos antigos sem CRC).

---

#### **F4 — Campos float do header `.namb` sem checagem de finitude/sanidade**

- **Severidade:** 🟡 **Média**
- **Locais (verificado):** `src/loader/namb/parse.rs:107-109,113,118-119,130-131` — `sample_rate`,
  `input_level_dbu`, `output_level_dbu` lidos via `read_unaligned` e propagados sem validação.
- **Impacto:** `input_level_dbu = NaN` torna o multiplicador de ganho de calibração `NaN`, corrompendo
  o caminho de sinal. `sample_rate` infinito/zero pode causar erros lógicos downstream (resampler).
- **Caminho de resolução:** Após validar o header, exigir `sample_rate.is_finite() && sample_rate > 0.0`
  e `input_level_dbu.is_finite()` / `output_level_dbu.is_finite()`. Rejeitar com `NambError` tipado.
- **Custo:** Baixo. **Risco:** Baixo.

---

#### **F5 — `unreachable!()` no catch-all `KnownFastPath(_)` (caminho de load)**

- **Severidade:** 🟢 **Baixa** (panic apenas no cold-path de load, não no RT)
- **Local:** `src/loader/dispatcher/wavenet/mod.rs:86`. Hoje é logicamente inalcançável (`is_a2_shape`
  só retorna `KnownFastPath(3|8)`), mas uma futura alteração que libere outros valores causaria panic
  e crash do DAW no load.
- **Caminho de resolução:** Trocar `unreachable!()` por `bail!("A2 channels inesperado: {ch}")`.
- **Custo:** Trivial.

---

#### **F6 — Helpers de transposição/pesos sem bounds-check interno (defense-in-depth)**

- **Severidade:** 🟢 **Baixa** (não explorável pelos callers atuais)
- **Locais:** `src/loader/dispatcher/wavenet/layout.rs:122-146` (`transpose_conv1d_interleaved_4wide`
  indexa `raw[(out_c*in_ch+in_c)*kernel+k]` sem checar `raw.len()`); `src/loader/dispatcher/lstm/weights.rs:14-38`
  (indexa `raw[k*ih*hidden+i*ih+j]` sem checar comprimento).
- **Impacto:** Atuais callers sempre passam slices exatos (via `WeightCursor`), mas a função confia no
  caller — futura regressão causaria OOB.
- **Caminho de resolução:** `debug_assert!(raw.len() >= ...)` no topo de cada helper (ou `Result`).
- **Custo:** Trivial.

---

#### **F7 — Truncamento `u64 → usize` em `parse_head` (portabilidade 32-bit)**

- **Severidade:** 🟢 **Baixa** (não explorável em alvos 64-bit, o único suportado)
- **Local:** `src/loader/nam_json/model.rs:247-260` — `v.as_u64().map(|v| v as usize)`.
- **Caminho de resolução:** Usar `usize::try_from(v).ok()?` por robustez/portabilidade.
- **Custo:** Trivial.

> **Defesas existentes a preservar (não duplicar):** cap de arquivo 256 MiB (`build.rs:43,97`),
> cap de 64M floats (JSON `validation.rs:43`, binário `parse.rs:93`), limites de metadados de treino,
> `MAX_SUBMODELS=8`, `MAX_CONTAINER_DEPTH=4`, `MAX_UNIFIED_DEPTH=8`, `WeightCursor` com bounds-check
> sequencial e `verify_exhausted()`, CRC32 obrigatório em v2, validação de alinhamento 4-bytes.

---

### 1.2 — Suíte de Testes / Harness

---

#### **F8 — Inconsistência do gate LUFS em `cpp_parity.rs` causa 3 falhas falso-positivas (CI vermelho)**

- **Severidade:** 🔴 **Alta** (bloqueia a auditoria longa; mascara o sinal real de "verde")
- **Locais (verificado):**
  - `tests/cpp_parity.rs:374` — `run_render_comparison` chama `report_dsp_fidelity` (**com** gate LUFS).
  - `tests/common/validation.rs:266-275` — o gate dispara quando `lufs_ref ∉ [-50, +10]`.
  - `tests/common/validation.rs:501-520` — os comentários de calibração **já documentam** que
    `wavenet_dyn_free` (head_scale=0.02) gera ~−65 LUFS e que "golden tests use `report_dsp_fidelity_no_lufs`".
  - Testes afetados: `cpp_parity.rs:651,667,677` (`wavenet_dyn`, `v2_wavenet_dyn`, `v2_lstm_dyn`).
- **Evidência (log):** `target/logs/phase2-proptests-parity.log:2165,2284,2341` — todas as falhas trazem
  SNR 116–124 dB, ESR ~1e-13 (**paridade excelente**) e panic exclusivamente por
  `Reference LUFS=-64.8/-71.3 outside plausible audio range`.
- **Causa-raiz:** O `golden_vectors.rs` foi atualizado para usar `report_dsp_fidelity_no_lufs` nesses
  modelos silenciosos-por-design, mas o `cpp_parity.rs` **não** foi — assimetria de harness.
- **Caminho de resolução (opções):**
  1. **Recomendado:** parametrizar `run_render_comparison`/`run_v1`/`run_v2_multi_sr` com um flag
     `check_lufs_gate`, e desativá-lo para a lista de modelos reconhecidamente quietos
     (`wavenet_dyn_free`, `lstm_dyn_test`), espelhando o `golden_vectors.rs`.
  2. Alternativa: tornar o gate LUFS ciente do `head_scale` do modelo (limiar inferior dinâmico).
  3. Garantir que a desativação seja **explícita e rastreável** (não silenciosa) — manter a impressão
     do `ⓘ LUFS gate skipped` que `report_dsp_fidelity_no_lufs` já emite.
- **Custo:** Baixo. **Risco:** Baixo. **Atenção:** não enfraquecer o gate globalmente (a lição T2.5 que
  o originou é válida) — desativar **somente** para os modelos justificados.

---

### 1.3 — RT-Safety (Hot-Path)

> Postura geral: **excelente**. Buffers pré-alocados em `activate()`/construção; GC via SPSC;
> ativações sempre por aproximação (nunca `f32::tanh`/`exp` nativo no hot-path); DAZ+FTZ no 1º bloco
> e reafirmado a cada 1024 blocos; dither anti-denormal ±1e-11. Os achados abaixo são pontuais.

#### **F9 — `.unwrap()` no thread RT em `cold_load_model`**

- **Severidade:** 🟡 **Média** (seguro na prática; viola política RT §1 de `.agents/rules/rust.md`)
- **Local (verificado):** `src/clap/processor/events.rs:184` — `self.model_l.take().unwrap()`.
  Alcançável de `process()` → `process_events()` → `cold_load_model()`. É seguro hoje (guardado por
  `resize_failed`, que só é `true` quando `model_l` era `Some`), mas `unwrap()` no RT é proibido.
- **Caminho de resolução:** `if let Some(failed) = self.model_l.take() { self.push_to_gc(GcItem::Model(failed)); }`.
- **Custo:** Trivial.

#### **F10 — `log::error!` no thread RT em `try_slimmable_rebuild`**

- **Severidade:** 🟡 **Média** (caminho de erro raro, mas é I/O/format no RT)
- **Locais (verificado):** `src/clap/processor/events.rs:252,282` — `log::error!("[slimmable] slice_channels(...) failed")`.
  `try_slimmable_rebuild` é chamada de `process_events` (events.rs:160) a cada bloco.
- **Caminho de resolução:** Substituir por sinalização atômica `RtStatusFlags` (ex.:
  `RT_STATUS_SLIMMABLE_SLICE_FAILED`); o main thread lê e loga.
- **Custo:** Baixo.

#### **F11 — Guardas de denormal ausentes nos tails escalares de ativações SIMD**

- **Severidade:** 🟢 **Baixa** (defesa em profundidade)
- **Locais:** tails escalares de `sigmoid.rs:204-206`, `tanh/production.rs` (≤7/≤15 elementos pós-SIMD).
  DAZ/FTZ global cobre o caso comum; risco só se o host resetar MXCSR entre reafirmações.
- **Caminho de resolução:** `if val.abs() < f32::MIN_POSITIVE { *val = 0.0; }` nos laços escalares
  de `*_slice_avx2`/`*_slice_avx512` (sigmoid/silu/tanh).
- **Custo:** Baixo.

---

### 1.4 — Concorrência e Ciclo de Vida (CLAP / Standalone)

#### **F12 — Scaffolding stereo morto e inconsistente no CLAP (mono): `model_r` descartado vs `active_model_r` lido**

- **Severidade:** 🟡 **Média** (não é bug vivo — CLAP é mono; é dívida/latência + comentário enganoso)
- **Contexto (verificado):** `src/clap/extensions/audio_ports.rs` declara **exatamente 1 porta mono**
  (`n: 1`, "NAM is native mono by definition"). Logo o caminho R é, na prática, inerte.
- **Inconsistência (verificado):**
  - `src/clap/processor/events.rs:189-191` — `cold_load_model` **descarta** `model_r` para o GC e
    nunca o armazena em `self.active_model_r`.
  - Porém `src/clap/processor/dsp/orchestrator.rs:68` e `events.rs:259,275` (`try_slimmable_rebuild`)
    **leem** `self.active_model_r` (sempre `None`, inicializado em `mod.rs:164` e nunca escrito) → ramos R mortos.
  - `src/clap/processor_test.rs:1576` comenta "cold_load_model no longer discards it" — **contradiz** o
    código (a lógica de descarte existe, apenas nunca é alcançada porque `model_r` é sempre `None`).
- **Impacto:** Nenhum hoje. **Latente:** se stereo for habilitado no CLAP no futuro, o modelo R seria
  silenciosamente descartado/quebrado. Código confuso para novos contribuidores.
- **Caminho de resolução (escolher 1):**
  1. **Remover** o scaffolding stereo do CLAP (parâmetro `model_r`, campo `active_model_r`, ramos R de
     `orchestrator`/`try_slimmable_rebuild`) e corrigir o comentário do teste; **ou**
  2. **Implementar** corretamente o armazenamento de `model_r` (espelhando o standalone
     `commands.rs:68-73`) caso stereo no CLAP seja roadmap.
  - Decisão deve constar em `docs/architecture.md` (§8 já afirma "mono").
- **Custo:** Baixo–Médio. **Risco:** Baixo.

#### **F13 — `GcOverflowBuffer::push` usa `Relaxed` no avanço de índice → drain deferido sob pressão**

- **Severidade:** 🟡 **Média/Baixa** (retenção temporária, **não** leak nem double-free)
- **Local (verificado por agente):** `src/common/spsc/gc.rs:169` — `write_idx.fetch_add(1, Relaxed)`
  seguido de `slots[idx].swap(packed, AcqRel)`. Se o drain rodar entre o `fetch_add` e o `swap`, lê o
  slot como vazio (`0`) e o item só é coletado no próximo ciclo.
- **Impacto:** Sob pressão sustentada de GC, retenção (atraso) de itens — não vazamento.
- **Caminho de resolução:** Escrever o dado antes de avançar o índice, ou documentar explicitamente a
  semântica de "drain deferido"; avaliar `Release` no `fetch_add` com leitura `Acquire` no drain.
- **Custo:** Baixo. **Risco:** Médio (mexer em ordering exige cuidado + teste de concorrência).

#### **F14 — `migrate()` de estado é no-op sem prontidão estrutural para v2**

- **Severidade:** 🟡 **Média** (dívida de futuro-proofing)
- **Local:** `src/clap/extensions/state.rs:59-66`. Hoje funciona (todos os campos têm `#[serde(default)]`),
  mas não há mecanismo para migrações que exijam transformação (v1→v2 com semântica não-default).
- **Caminho de resolução:** Estruturar `match version { 0 => …, 1 => …, _ => params }` deixando o corpo
  v0→v1 documentado como "coberto por `serde(default)`".
- **Custo:** Baixo.

#### **F15 — Mensagem de erro com escape ANSI em `state.load()` vazio**

- **Severidade:** 🟢 **Baixa/Média** (protocolo de host)
- **Local:** `src/clap/extensions/state.rs:93` — retorna `PluginError::Message("\r\x1b[K")` para suprimir
  exibição no terminal. Hosts que serializam/logam a string podem ter corrupção de display.
- **Caminho de resolução:** Retornar mensagem limpa (ex.: `"NAM-rs: no state to load"`) e deixar a
  política de exibição com o host.
- **Custo:** Trivial.

#### **F16 — `from_raw_parts_mut` sem `debug_assert` de bounds no callback PipeWire**

- **Severidade:** 🟢 **Baixa**
- **Local:** `src/standalone/pw_host/rt_callback/process.rs:50-60`. `offset`/`n_samples` vêm do libpipewire
  (confiável), mas a relação `offset + n_bytes <= len` e `n_samples*4 == n_bytes` não é assertada.
- **Caminho de resolução:** `debug_assert!` documentando os invariantes (custo zero em release).
- **Custo:** Trivial.

#### **F17 — Trait `AudioHost` é dead code em produção**

- **Severidade:** 🟢 **Baixa/Média** (organização/dead code)
- **Local:** `src/common/audio_host.rs` — trait definido mas implementado apenas por `MockHost` (teste).
  Nem PipeWire nem CLAP o implementam.
- **Caminho de resolução:** Remover, ou documentar como contrato aspiracional, ou implementá-lo no host
  PipeWire para dar-lhe propósito real.
- **Custo:** Baixo.

---

### 1.5 — Soundness e Performance SIMD (`src/math/`)

> Dispatch à prova de SIGILL confirmado (`detect.rs:16-107`, `is_x86_feature_detected!` com fail-fast).
> Conversão f16↔f32 testada exaustivamente vs F16C (bit-exata). Tail/remainder corretos em todos os
> kernels inspecionados.

#### **F18 — Convolução usa `load_ps`/`load` alinhado sem `debug_assert` de alinhamento (GP-fault latente)**

- **Severidade:** 🟠 **Média-Alta** (UB latente)
- **Locais:** `src/math/dsp/stereo/convolution_avx2.rs:28,34,44` e `convolution_avx512.rs:28,34,80,…`
  usam loads **alinhados** para `coeffs` (exigem 32B/64B). A doc do trait (`traits.rs:360-401`) exige
  alinhamento, mas não há **assert** em runtime. Se algum caller futuro passar coeffs não-`AlignedVec`,
  ocorre SIGSEGV/#GP em CPUs estritas.
- **Caminho de resolução:** `debug_assert!(coeffs.as_ptr() as usize % 64 == 0)` no início dos kernels
  (ou usar loadu como fallback seguro fora do hot-path crítico).
- **Custo:** Trivial. **Risco:** Baixo.

#### **F19 — Tail escalar dos LSTM gates usa `std::exp`/`tanh` (libm) enquanto o corpo SIMD usa poly approx**

- **Severidade:** 🟡 **Média** (descontinuidade numérica na fronteira SIMD→escalar)
- **Locais:** `src/math/lstm/gates.rs:95-105,137-147`. O corpo usa `simd_sigmoid`/`simd_tanh_poly`
  (erro ~4e-4 / ~2.3e-3), mas os ≤7/≤15 elementos do tail usam `(-x).exp()`/`x.tanh()` da libm
  (regime numérico distinto, salto ~1e-3) quando `hidden_size % 8 != 0` (ou `% 16`).
- **Caminho de resolução:** Usar as **mesmas** aproximações poly no tail escalar (consistência), ou
  documentar/justificar a divergência e cobrir com teste de fronteira em `hidden_size` não-múltiplo.
- **Custo:** Baixo. **Risco:** Baixo (pode alterar levemente goldens — revalidar thresholds).

#### **F20 — 4 ativações sem kernel AVX-512 (performance deixada na mesa)**

- **Severidade:** 🟡 **Média** (perf quick-win)
- **Local:** `src/math/common/dispatch/detect.rs:66-69` — `hard_tanh`, `hard_swish`, `fast_tanh`,
  `leaky_hard_tanh` mapeiam para AVX2 mesmo em backends AVX-512 (throughput ~2× menor). Usadas por
  ativações A2 (`models/a2/activations.rs`).
- **Caminho de resolução:** Implementar variantes AVX-512 (16-wide + tail mascarado), seguindo o padrão
  já existente em `avx512/`.
- **Custo:** Médio. **Risco:** Baixo (validar paridade vs escalar/AVX2).

#### **F21 — Intrínseco `_mm_cvtss_f32` deprecado em 12+ sites de redução horizontal**

- **Severidade:** 🟢 **Baixa** (estilo/portabilidade; binários corretos)
- **Local:** `convolution_avx2.rs:61` e sites subsequentes. Preferir `_mm_store_ss(&mut out, r)`
  (padrão já usado em `utility.rs:28`).
- **Custo:** Baixo.

#### **F22 — Oráculo escalar de tanh tautológico + divergência de ordem no GEMV de referência**

- **Severidade:** 🟡 **Média** (qualidade de validação — risco de "golden que não falha")
- **Locais (varredura por agente):**
  - `src/math/common/scalar_ref/utility.rs:18-22` — `tanh_slice_fallback` chama `scalar_pade_tanh`,
    **a mesma** função usada como tail escalar em `tanh_slice_avx2` (`production.rs:178`). Oráculo e
    kernel-fallback compartilham o **mesmo** Padé → o teste só confirma consistência interna, não
    valida contra referência independente (`f32::tanh`/`high_fidelity` exp-based, erro ~2.4e-7).
  - `src/math/common/scalar_ref/gemm.rs` — `gemv_with_bias_f32_fallback` soma em ordem
    *output-channel-major*, enquanto `gemv_with_bias_f32_avx2` usa 4 acumuladores YMM
    *input-channel-major* → resultados divergem (~1e-5 para IN grande) por não-associatividade FP;
    parity tests precisam de epsilon relaxado consciente.
- **Impacto:** Reduz a força do oráculo: um bug no Padé compartilhado passaria despercebido pela
  parity SIMD↔escalar; a divergência de ordem no GEMV pode mascarar pequenos erros sob epsilon frouxo.
- **Caminho de resolução:** Cross-validar tanh contra referência **independente** (libm ou
  `scalar_tanh_poly` de alta fidelidade) ao menos em um teste; documentar/normalizar a ordem de
  acumulação do GEMV de referência ou justificar o epsilon. Sinérgico com F30/F34 (E8).
- **Custo:** Baixo.

---

### 1.6 — Organização, Deduplicação e "Embelezamento"

#### **F23 — Duplicação cross-width AVX2/AVX-512 (~850 SLOC) em gain/convolução/ativações**

- **Severidade:** 🟡 **Média** (manutenção + risco de drift)
- **Locais/escala (estimada por agente):**
  - `src/math/dsp/gain/avx2.rs` (234L) × `gain/avx512.rs` (257L): 8 pares quase idênticos (~400 SLOC).
  - `src/math/dsp/stereo/convolution_avx2.rs` (335L) × `convolution_avx512.rs` (179L): ~300 SLOC estruturais.
  - Laços slice de ativações (`*_slice_avx2`/`avx512`): ~150 SLOC de boilerplate repetido.
- **Padrão de referência (bom):** `src/math/gemm/gemv/kernel_macro.rs` já parametriza largura por macro.
- **Caminho de resolução:** Estender a abordagem `gemv_kernel!` para gain, convolução e ativações
  (macro com largura/step/tail como parâmetros). Reduz ~850 SLOC e o risco de divergência AVX2/AVX-512.
- **Custo:** Médio. **Risco:** Médio (revalidar paridade bit-a-bit pós-refator; usar testes existentes como rede).

#### **F24 — Arquivos gigantes com concerns misturados (split recomendado)**

- **Severidade:** 🟡 **Média** (navegabilidade/manutenção)
- **Locais (LOC):** `src/clap/processor_test.rs` (2048, ~19 concerns de teste), `src/math/dsp/fft.rs`
  (1284, 780L de testes inline), `src/models/a2/grouped_conv1d.rs` (1205, mistura struct/scalar/SIMD/teste),
  `src/models/a2/model/dynamic.rs` (1140), `src/loader/nam_json/topology.rs` (765, 5 famílias),
  `src/models/slimmable.rs` (725, 58% testes).
- **Caminho de resolução:** Extrair testes para `_test.rs`; dividir `processor_test.rs` em arquivos
  temáticos; quebrar `topology.rs` em subdir `topology/{wavenet,lstm,convnet,a2,linear}.rs`.
- **Custo:** Médio (mecânico). **Risco:** Baixo.

#### **F25 — Aderência inconsistente à regra de posicionamento de testes (`.agents/rules/testing.md`)**

- **Severidade:** 🟢 **Baixa/Média**
- **Detalhe (calibrado):** O padrão `#[cfg(test)] #[path="x_test.rs"] mod tests;` está **correto** e em uso.
  Os desvios reais são: (a) arquivos **>300L com testes inline** que deveriam ser externos
  (`fft.rs`, `grouped_conv1d.rs`, `a2/model/dynamic.rs`, `a2/model/mod.rs:659`, `a2/film.rs:632`);
  (b) arquivos **<300L com testes externos** desproporcionais (`a2/activations.rs` 157L → `_test.rs` 461L);
  (c) itens auxiliares `#[cfg(test)]` inline (refs/geradores) em arquivos que **também** têm `_test.rs`
  (ex.: `conv1d.rs:205,232`; `diagnostic.rs:11,15`) — inconsistência menor, **não** "testes em dois lugares".
- **Caminho de resolução:** Alinhar cada arquivo à regra (mover testes conforme o limiar de 300L);
  consolidar helpers de teste no `_test.rs` correspondente.
- **Custo:** Médio (mecânico).

#### **F26 — Nomenclatura inconsistente de testes (`tests.rs` vs `_test.rs`, `test_files/`)**

- **Severidade:** 🟢 **Baixa**
- **Locais:** `src/math/activations/tests.rs`, `src/math/common/tests.rs`, `src/models/{a2,lstm,wavenet}/tests.rs`,
  `src/clap/gui/ui/test.rs`, e subdirs `src/models/wavenet/test_files/`, `src/dsp/pipeline/test_files/`.
- **Caminho de resolução:** Padronizar sufixo `_test.rs` ou consolidar em subdir `tests/` por módulo.
- **Custo:** Baixo.

#### **F27 — `try_slimmable_rebuild` duplicado entre standalone e CLAP (~45 SLOC)**

- **Severidade:** 🟢 **Baixa/Média**
- **Locais:** `src/standalone/pw_host/rt_callback/commands.rs:117-165` × `src/clap/processor/events.rs:223-286`.
  Núcleo idêntico (checar FSM → slice → prewarm → GC); CLAP adiciona `inject_rt_status` + wrap em `StaticModel`.
- **Caminho de resolução:** Extrair helper genérico sobre `&mut Option<Box<StaticModel>>` com hook de
  `inject_rt_status`. **Atenção:** a decisão arquitetural (architecture.md §8.1) rejeitou `SwapStrategy<T>`
  por <100 SLOC — este caso fica abaixo do limiar, então avaliar custo/benefício; pode ficar como dívida aceita.
- **Custo:** Baixo–Médio.

#### **F28 — `#[allow(dead_code)]` espalhado (9 ocorrências)**

- **Severidade:** 🟢 **Baixa**
- **Locais (a auditar):** `loader/dispatcher/lstm/weights.rs:97`, `models/a2/conv1d_ch3/mod.rs:142,168,237`
  (3 no mesmo arquivo — red flag), `dsp/pipeline/output_pw.rs:18`, `models/wavenet/conv_input.rs:105`,
  `loader/dispatcher/wavenet/traits.rs:11`.
- **Caminho de resolução:** Auditar caso a caso — remover código morto ou justificar com comentário
  (ex.: `cfg`-gated por feature). Eliminar especialmente os 3 de `conv1d_ch3/mod.rs`.
- **Custo:** Baixo.

#### **F29 — Layout inconsistente em `src/models/a2/` (conv1d_ch3 é diretório, conv1d_ch8 é arquivo flat)**

- **Severidade:** 🟢 **Baixa**
- **Local:** `conv1d_ch3/{mod,simd,scalar}.rs` (982L) vs `conv1d_ch8.rs` flat (497L) vs `conv1d.rs` (243L).
  SIMD genuinamente distinto (SSE 128-bit vs AVX2 256-bit) — não é duplicação, mas a inconsistência
  prejudica navegação.
- **Caminho de resolução:** Padronizar — elevar `conv1d_ch8` a diretório `{mod,simd,scalar}` **ou**
  achatar `conv1d_ch3`. Preferir consistência por elevação.
- **Custo:** Baixo.

---

### 1.7 — Rigor e Cobertura de Testes

#### **F30 — Testes de correção críticos isolados atrás de `#[ignore]` (fora da lane rápida)**

- **Severidade:** 🟠 **Alta** (risco de regressão não-detectada no fluxo do desenvolvedor)
- **Detalhe (verificado):** 115 atributos `#[ignore]`. `utils/tests-quick.sh` roda `cargo test` **sem**
  `--ignored`, logo **não** exercita: `cpp_parity.rs` (45 testes — paridade C++), `proptest_parsers.rs`
  (10 — fuzzing de parser), `proptest_math.rs` (4 — precisão SIMD), `golden_vectors.rs` v2 (16),
  FSM proptests (gate/adaptive), soak (19).
- **Impacto:** Uma regressão no parser JSON ou nos kernels SIMD passa despercebida até a auditoria longa
  (~43 min), tipicamente só em CI/pré-release. Acopla-se a F1/F2 (os proptests de parser que pegariam
  `NaN`/dimensões absurdas estão ignorados).
- **Caminho de resolução:**
  1. Promover 1–2 testes representativos de cada categoria crítica à lane rápida (1 paridade C++ no
     formato mais comum; 1 smoke de fuzzing com iterações reduzidas; 1 precisão tanh SIMD).
  2. Criar lane intermediária com subconjunto curado. **→ Implementada como Phase 2 de `utils/tests-quick.sh` (cpp_parity + proptest_parsers + proptest_math, release, ignored), com skip condicional se goldens ausentes. O script unificado é o PR gate do projeto.**
- **Custo:** Baixo–Médio. **Risco:** Baixo.

#### **F31 — Asserções somente-`is_finite()` em ~20 stress tests (drift de valor passa silenciosamente)**

- **Severidade:** 🟠 **Média-Alta**
- **Locais (exemplos):** `processor_test.rs:133-141,147`; `soak_test.rs:330,333,421,422,609,649`;
  `nam_infer_test.rs:391`; `a2_loader.rs:825,855`; `concurrency_stress.rs:342,451,525,695,703`.
- **Impacto:** Um modelo produzindo lixo finito (tudo 0.0 ou tudo 1e6) **passa**. O gate só pega `NaN`/`Inf`.
- **Caminho de resolução:** Adicionar checagem de banda de RMS/energia de sinal junto ao `is_finite()`
  (ex.: `0.001 ≤ rms ≤ 1.0` após entrada conhecida) — pega drift sem custo relevante.
- **Custo:** Baixo.

#### **F32 — Módulos críticos sem testes unitários co-localizados (cobertos só indiretamente)**

- **Severidade:** 🟡 **Média** (calibrado — **não** é "zero cobertura")
- **Detalhe:** `set_weights.rs` (403L), `nam_json/validation.rs` (355L), `avx2_impl.rs` (592L),
  `gemm_batch/avx512.rs` (445L), entre outros, não têm `_test.rs` próprio. **Porém** são exercitados
  indiretamente: `validation.rs` por `nam_infer_test.rs`/`cpp_parity.rs`/`golden_vectors.rs`; `avx2_impl`
  por praticamente todos os testes de math (parity vs scalar). A lacuna é de **testes-unidade dirigidos
  a edge cases**, não de cobertura total.
- **Caminho de resolução:** Priorizar testes-unidade para `set_weights.rs` (fronteira de correção) e
  `validation.rs` (fronteira de segurança — ver F1/F2). Para SIMD, garantir parity tests AVX2/AVX-512 vs
  escalar onde ainda não houver.
- **Custo:** Médio.

#### **F33 — Poucos `#[should_panic]` (≈18) para a quantidade de `unsafe` SIMD**

- **Severidade:** 🟡 **Média**
- **Detalhe:** Lacunas em `grouped_conv1d.rs`, `conv1d_ch3/simd.rs`, `conv1d_ch8.rs`, `resampler.rs`
  (sem testes negativos de tamanho/alinhamento inválido).
- **Caminho de resolução:** Adicionar testes negativos nas fronteiras `unsafe` (tamanhos inválidos,
  buffers incompatíveis).
- **Custo:** Médio.

#### **F34 — Bandas de validação largas (oráculos frouxos)**

- **Severidade:** 🟡 **Média**
- **Local:** `processor_test.rs:1875-1882` (`test_model_gain_calibration`: `rms < 0.4` — um erro de ganho
  de 3× ainda passaria). (Ver também **F22** para o oráculo tautológico de tanh — concern correlato.)
- **Caminho de resolução:** Apertar a banda de ganho com base em característica conhecida do modelo
  (ex.: `[0.05, 0.2]`) ou comparar contra golden calibrado.
- **Custo:** Baixo.

---

## 2. ÉPICOS ÁGEIS (Agrupamento dos Achados)

> Organização lógica para execução **otimizada, segura e ágil**. Cada épico é coeso (mesmo subsistema /
> mesma natureza de risco), minimizando conflitos de merge e maximizando reuso de validação.
> Sprints e tarefas técnicas **serão elaborados depois**, épico a épico.

---

### **🔴 Épico E1 — Correção do Pipeline de Validação C++ (CI Verde)** [DONE]

- **Objetivo:** Restaurar a auditoria longa para 6/6 fases verdes, sem enfraquecer gates legítimos.
- **Achados:** **F8**.
- **Por que primeiro:** Isolado, baixo custo, alto valor (desbloqueia o sinal de qualidade da lane longa).
  Sem este, regressões reais futuras ficam ofuscadas pelo falso-positivo recorrente.
- **Risco:** Baixo. **Critério de pronto:** `tests/cpp_parity.rs` verde mantendo o gate LUFS ativo para
  modelos não-quietos; desativação explícita e rastreável apenas para `wavenet_dyn_free`/`lstm_dyn_test`.

### **🔴 Épico E2 — Blindagem de Segurança do Loader (Hardening contra modelos maliciosos)** [DONE]

- **Objetivo:** Tornar o parsing de `.nam`/`.namb` resiliente a entrada hostil/corrompida.
- **Achados:** **F1** (NaN/Inf), **F2** (alloc DoS), **F3** (CRC v1), **F4** (header floats), **F5**
  (unreachable), **F6** (bounds helpers), **F7** (u64→usize).
- **Coesão:** Tudo em `src/loader/`; uma única expansão de proptest (E8/F30) cobre as regressões.
- **Risco:** Baixo–Médio (caminho cold; cuidar de compatibilidade com modelos legítimos antigos — F3).
- **Crítico:** F1 e F2 têm impacto de segurança real (dano a equipamento / DoS).

### **🟠 Épico E3 — Pureza RT do Hot-Path (Zero-panic, Zero-IO, Zero-denormal)** [DOING]

- **Objetivo:** Eliminar os últimos desvios da política RT-safety §1.
- **Achados:** **F9** (unwrap), **F10** (log::error RT), **F11** (denormal tails), **F19** (salto numérico
  LSTM tail — correção RT/numérica).
- **Coesão:** Hot-path RT; mudanças pequenas e localizadas.
- **Risco:** Baixo (F19 pode mexer em goldens — revalidar thresholds).

### **🟡 Épico E4 — Robustez de Concorrência e Ciclo de Vida (CLAP/Standalone)** [DOING]

- **Objetivo:** Eliminar scaffolding morto/ambíguo e endurecer concorrência/persistência.
- **Achados:** **F12** (model_r/active_model_r), **F13** (GcOverflow ordering), **F14** (migrate),
  **F15** (escape ANSI), **F16** (bounds pw_host), **F17** (AudioHost dead).
- **Coesão:** Camadas `clap/`+`standalone/`+`common/spsc`.
- **Risco:** Médio em F13 (ordering atômico → exige teste de concorrência dedicado).

### **🟡 Épico E5 — Soundness e Performance SIMD** [DOING]

- **Objetivo:** Fechar UB latente e capturar quick-wins de performance.
- **Achados:** **F18** (assert de alinhamento na convolução), **F20** (ativações AVX-512), **F21**
  (intrínseco deprecado).
- **Coesão:** `src/math/`. **Risco:** Baixo (F20 valida paridade vs escalar/AVX2).
- **Nota:** Performance é fase posterior, mas F18 (UB) e F20 (quick-win 2×) merecem antecipação.

### **🟢 Épico E6 — Unificação e Deduplicação de Kernels (DRY SIMD)** [TO-DO]

- **Objetivo:** Reduzir ~850 SLOC duplicados e o risco de drift AVX2/AVX-512.
- **Achados:** **F23** (gain/conv/activation macros), **F27** (slimmable rebuild dedup).
- **Coesão:** Refator estrutural com rede de testes de paridade existente.
- **Risco:** Médio (revalidar paridade bit-a-bit pós-refator). **Dependência:** idealmente após E5
  (alinhar F20 antes de macro-unificar ativações).

### **🟢 Épico E7 — Reorganização Estrutural e Higiene ("Embelezamento")** [TO-DO]

- **Objetivo:** Melhorar navegabilidade e aderência às próprias regras do projeto.
- **Achados:** **F24** (split de arquivos gigantes), **F25** (regra de testes 300L), **F26** (nomenclatura),
  **F28** (dead_code), **F29** (layout a2/).
- **Coesão:** Mudanças mecânicas, baixo risco; ótimas para paralelizar via Agent Manager.
- **Risco:** Baixo.

### **🟠 Épico E8 — Rigor e Cobertura da Suíte de Testes** [DONE]

- **Objetivo:** Fechar pontos cegos do processo de teste e fortalecer oráculos.
- **Achados:** **F30** (#[ignore] na lane rápida), **F31** (is_finite-only), **F32** (módulos críticos
  sem unit test), **F33** (should_panic), **F34** (bandas largas), **F22** (oráculo tanh tautológico +
  divergência GEMV de referência).
- **Coesão:** Qualidade de testes transversal. **Sinergia:** F30 habilita a rede de regressão para
  E2 (segurança) e E3/E6 (refators).
- **Risco:** Baixo.

---

## 3. Matriz de Priorização Sugerida

| Ordem | Épico                        | Severidade máx. | Esforço          | Dependências                      |
|:-----:|:---------------------------- |:---------------:|:----------------:|:--------------------------------- |
| 1     | **E1** (CI Verde / LUFS)     | 🔴 Alta         | Baixo            | —                                 |
| 2     | **E2** (Segurança Loader)    | 🔴 Alta         | Baixo–Médio      | habilitado por E8/F30 (proptests) |
| 3     | **E8** (Rigor de Testes)     | 🟠 Alta         | Baixo–Médio      | — (precede E2 idealmente)         |
| 4     | **E3** (Pureza RT)           | 🟡 Média        | Baixo            | —                                 |
| 5     | **E5** (SIMD soundness/perf) | 🟠 Média-Alta   | Baixo–Médio      | —                                 |
| 6     | **E4** (Concorrência/CLAP)   | 🟡 Média        | Baixo–Médio      | —                                 |
| 7     | **E6** (Dedup SIMD)          | 🟡 Média        | Médio            | após E5                           |
| 8     | **E7** (Embelezamento)       | 🟢 Baixa        | Médio (mecânico) | — (paralelizável)                 |

> **Sequenciamento recomendado:** começar por **E1** (desbloqueio imediato) e **E8/F30** (rede de
> regressão), pois ambos viabilizam executar **E2** (segurança) com segurança. E3/E4/E5 são independentes
> e paralelizáveis. E6 (refator de risco médio) depois de E5. E7 (higiene) a qualquer momento, ideal para
> fan-out via Agent Manager por ser mecânico e de baixo acoplamento.

---

## 4. Notas de Método e Rastreabilidade

- **Verificação direta (arquivo:linha citados):** F1, F2, F3, F4, F8, F9, F10, F12, F25 (calibrado),
  F32 (calibrado) — lidos no código durante a auditoria.
- **Varredura por subsistema (agentes de exploração), severidade calibrada manualmente:** demais achados.
- **Reclassificações relevantes vs. varredura bruta:**
  - O "model_r descartado" (F12) foi **rebaixado de Crítico → Médio**: CLAP é estritamente mono
    (`audio_ports.rs`, `n:1`), logo é scaffolding morto/latente, não bug vivo.
  - "21 arquivos violam regra de testes" foi **calibrado**: o padrão `#[path] mod tests;` está correto;
    o desvio real é misto (F25), de severidade Baixa/Média.
  - "5.200 linhas sem cobertura" foi **calibrado** para "sem teste-unidade co-localizado" (F32) — há
    cobertura indireta por testes de integração/paridade.
- **Referência de paridade C++:** `docs/cpp_parity_map.md` (mapeamento ponto-a-ponto) e
  `tests/fixtures/NeuralAmpModelerCore` (commit pinado v0.5.3) — a aderência arquitetural está bem
  documentada e os testes de paridade passam (exceto o falso-positivo F8).

---

*Fim do TODO-findings.md — Achados (F1–F34) e Épicos (E1–E8). Sprints e tarefas técnicas serão
elaborados separadamente, por épico, quando solicitado.*
