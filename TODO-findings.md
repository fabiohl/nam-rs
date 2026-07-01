<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Levantamento de Pontos e Propostas de Solução

Artefato de findings gerado pelas skills `revisor-auditor` → `planejador-arquiteto`.
Cada finding é detalhado e acompanhado de proposta de solução; os Épicos ao final agrupam
os findings para execução ágil, segura e de baixo risco.

---

## Auditoria de Paridade NAMCore × NAM-rs — `docs/cpp_parity_map.md` §13 (2026-06-30)

> **Escopo:** análise profunda do tópico **"13. Pending / Open Work"** de
> [`docs/cpp_parity_map.md`](docs/cpp_parity_map.md) e demais seções correlatas, confronto
> com o estado **real** do código/testes, e proposta de mitigações simples e seguras.
>
> **Princípio inviolável:** nenhuma proposta pode quebrar a compatibilidade com o padrão
> **NAMCore** nem os objetivos/ideais do **NAM-rs** (RT-safety, fidelidade sonora, Linux
> low-latency, paridade auditável). Todas as soluções abaixo são **aditivas** (oráculos,
> testes, documentação) ou **doc-only**; nenhuma altera a lógica/algoritmo do motor de
> produção, salvo se um bug real for comprovado por uma testemunha independente.

## Contexto: o que a auditoria encontrou

O §13 estava **desatualizado** em três frentes materiais. Confronto com o código:

1. **WaveNet Lite CH=12** estava marcado 🔴 *"divergência arquitetural, SNR ≈ 0.9 dB,
   `#[ignore]` (P1)"* — mas o defeito **já foi resolvido**. O teste golden está **ativo**
   (`tests/golden_vectors.rs:513`, sem `#[ignore]`) medindo **SNR = 122.3 dB / ESR = 5.84e-13**
   e o threshold calibrado vive em `tests/common/validation.rs:530-533` (`P1 ✅ RESOLVIDO`).
2. **FiLM / GatingActivation / BlendingActivation / `condition_dsp` / `bottleneck ≠ channels`**
   estavam marcados *"parser surface, not wired / out of scope"* — mas estão **plenamente
   conectados** ao motor dinâmico `WaveNetA2Dyn` e **cobertos por goldens ativos**
   (`tests/golden_vectors.rs:1460,1517,1570,1628`; `tests/common/validation.rs:558-621`).
3. **Referências cruzadas quebradas:** `cpp_parity_map.md` apontava para findings inexistentes
   (`F1`/`F2`/`I6`) e `docs/fastmath-approximations.md:346` aponta para `TODO-problemas.md#P1`,
   **arquivo que não existe** no repositório.

O código já carrega *tags* de findings vivas — `RF1` (divergência FiLM), `RF3` (ausência
de goldens v2 para motores dinâmicos, por design), `RF7` (Lite resolvido) — em
`tests/common/validation.rs`, `tests/golden_vectors.rs` e `docs/perceptual_validation.md`,
mas **sem registro central**. Este documento cria esse registro (prefixo `PM-##`, *Parity Map*)
e mapeia as *tags* existentes.

> **Nota:** os ajustes documentais do **PM-01** já foram aplicados ao `docs/cpp_parity_map.md`
> nesta mesma auditoria (§3.3, §5, §6, §9.1, §11.2/§4.5, §13, See Also).

---

## Findings (Constatações)

### PM-01 — Dessincronia do `cpp_parity_map.md` com o estado real do motor [APLICADO]

* **ID:** PM-01 · **Severidade:** Alta (documento de paridade é fonte de verdade auditável) · **Risco da correção:** Nulo (doc-only)
* **Problema:** O §13 e seções correlatas descreviam um estado **defasado e enganoso** do
  motor — subestimando capacidades reais (A2 dinâmico) e superestimando defeitos já sanados
  (Lite). Um mapa de paridade incorreto corrói a confiança no processo de validação e pode
  levar a decisões erradas (ex.: reabrir trabalho já concluído, ou tratar features prontas
  como ausentes).
* **Evidências:**
  * Lite resolvido: `tests/golden_vectors.rs:500-552`, `tests/common/validation.rs:405,421-426,526-533`.
  * A2 dinâmico conectado: `src/models/a2/model/dynamic/process.rs`, `src/models/a2/film.rs:116-203`,
    `src/models/a2/gating.rs:62-225`; goldens em `tests/golden_vectors.rs:1460-1674`.
  * Refs quebradas: `docs/cpp_parity_map.md` (antigos `F1`/`F2`/`I6`).
* **Proposta de Solução (aplicada):** Atualização cirúrgica do `docs/cpp_parity_map.md`:
  * §3.3 — reescrita da justificativa "(c)" (geometrias A2 dinâmicas são **validadas por
    goldens v1 via `WaveNetA2Dyn`**, não "parser surface only").
  * §5 — nota de status do ConvNet (cross-validation C++ bloqueada; caminho via oráculo f64).
  * §6 — cabeçalho reescrito em **dois caminhos** (fast-path fixo + motor dinâmico); linhas da
    tabela `Gating/Blending/FiLM` atualizadas de "not wired" → "wired in `WaveNetA2Dyn`".
  * §9.1 — linha do Lite atualizada (122.3 dB) + nota de RCA.
  * §11.2/§4.5 — referência `I6` quebrada substituída pelos caps HF (§4.5 + `perceptual_validation.md`).
  * §13 — tabela e notas reescritas (este documento como destino dos detalhes).
  * See Also — referência quebrada substituída por `PM-01…PM-08`.

### PM-02 — Avisos obsoletos de WaveNet Lite (P1/RF7) em `fastmath-approximations.md` §9.4 [RESOLVIDO]

* **ID:** PM-02 · **Severidade:** Alta (alarme falso ao usuário) · **Risco da correção:** Baixo (doc-only)
* **Problema:** `docs/fastmath-approximations.md` §9.4 (linhas 325-346) ainda intitula
  *"Lite Architectures — P1 Remains (Architectural)"*, afirma *"SNR ≈ 0.9 dB"* e exibe um bloco
  `> [!CAUTION]` dizendo que *"Lite models should be treated with caution... documented as a
  known limitation (see `TODO-problemas.md#P1`)"*. Isso é **autocontraditório** (a própria
  tabela na linha 340 já mostra `EVH-5150-Lite ✅ 122.3 dB`) e **referencia um arquivo que não
  existe** (`TODO-problemas.md`). Um usuário lendo essa seção concluiria, erradamente, que
  modelos Lite produzem áudio decorrelacionado do NAMCore.
* **RCA (causa-raiz do 0.9 dB, agora compreendida):** dois fatores combinados, ambos sanados:
  1. **Bug do `MirroredBuffer`** — o buffer das *delay lines* era arredondado ao tamanho de
     página (4096 B = 1024 `f32`) sem garantir alinhamento ao número de canais. Para canais
     **não-potência-de-dois** (CH=12 do array1 e CH=6 do array2 do Lite), `1024 % 12 = 4` e
     `1024 % 6 = 4`, de modo que a aritmética de *wrap* (`src/models/wavenet/common.rs:104`,
     `buffer_frames = size/CH`) deslocava o ponteiro em +4 elementos por ciclo de wrap,
     dessincronizando a saída até decorrelação (~0.9 dB). Demais SKUs (CH=16/8/4) têm
     `1024 % CH == 0` e nunca foram afetados.
  2. **Golden sintético obsoleto** — o golden vinha do modelo sintético `BossWN-lite.nam`.
* **Correção (já no código):** `MirroredBuffer::new_aligned(req, elem_multiple)` calcula
  `lcm(page_size, elem_stride)` (`src/dsp/mirror_buf/alloc.rs:63,128,137`), garantindo
  `size_elements % channels == 0`; e o golden migrou para o modelo real `EVH-5150-Lite.nam`.
  Guardas de regressão **ativas** (não-`#[ignore]`): `tests/wavenet_lite_block_invariance.rs:162`,
  `src/models/wavenet/wavenet_test.rs:208` (`wavenet_ringbuffer_alignment`),
  `src/dsp/mirror_buf_test.rs:104` (`test_mirror_buf_channel_alignment`).
* **Proposta de Solução:**
  1. Reescrever `fastmath-approximations.md` §9.4: trocar o título e o corpo para refletir a
     **resolução** (RCA do bug de alinhamento + migração de golden), mantendo a tabela
     comparativa e adicionando a nota das 3 guardas de regressão.
  2. Remover o bloco `> [!CAUTION]` (ou convertê-lo em nota histórica `> [!NOTE]` de "resolvido").
  3. Substituir a referência morta `TODO-problemas.md#P1` por `cpp_parity_map.md §9.1` e este
     finding (PM-02). Tratado em conjunto com **PM-08**.

### PM-03 — Divergência interop de **FiLM A2 (RF1)** sem testemunha de oráculo f64 **[RESOLVIDO — H1 Inerente]**

* **ID:** PM-03 · **Severidade:** Média-Alta (única feature A2 com baixa paridade) · **Risco da correção:** Baixo-Médio (test-only; só toca produção se comprovar bug)
* **Status:** ✅ **RESOLVIDO (2026-06-30, S10.3)** — H1 confirmada: divergência estrutural inerente, implementação Rust matematicamente correta.
* **Resolução (S10.3):**
  * Oráculo f64 estendido em `src/testing/reference_oracle.rs:642` com FiLM completo: parse de 8 slots via `layer_raw`, leitura de pesos/bias após `l1x1_b`, `FilmOracleSlot::apply` (grupos GEMV → `cond_to_scale_shift` + modulação), 6 pontos de inserção ativos + conv_pre na history.
  * Cross-check Rust f32 × oráculo f64:
    * `wavenet_a2_film_lite.nam`: ESR = 9.52e-15 (−140.2 dB) — piso numérico.
    * `wavenet_a2_film_full.nam`: ESR = 1.15e-14 (−139.4 dB) — piso numérico.
    * Ambos ≪ 1e-9 → **H1 (inerente)** confirmada. A implementação Rust FiLM está matematicamente correta. A divergência vs C++ (18-36 dB SNR) é estrutural: o C++ fallback generic WaveNet (Eigen) aplica conditioning por caminho diferente do FiLM nativo Rust.
  * `RF1` reclassificado de 🔴 (suspeito) para 🟡 (inerente, documentado, capado).
  * `A2_FILM_ESR_LIMIT = 1e-12` em `tests/common/constants.rs` (piso numérico).
  * Testes: `test_oracle_a2_film_lite`, `test_oracle_a2_film_full`, `test_combined_simulation_a2_film` em `tests/reference_oracle_f64.rs`.
  * `cpp_parity_map.md` §6/§13 atualizado: "inherent structural divergence, oracle-witnessed".
  * **Pendente:** Âncora NumPy independente (S10.4) para fechar a cadeia de 3 vias (Regra 6).

### PM-04 — **ConvNet** sem validação externa: implementar oráculo f64 independente **[RESOLVIDO — Sprint S10]**

> **Resolução (commits `653cdbc`, `7788f40`):** adicionado oráculo f64 de ConvNet
> (`tests/reference_oracle_f64.rs::test_oracle_convnet` + `test_oracle_vs_python_anchor_convnet`, ambos
> ativos) — ESR ≈ 1.83e-14 vs produção, `CONVNET_ESR_LIMIT = 1e-12`, com âncora NumPy (cadeia §9.2).
> Interop C++ permanece **N/A por formato** (bespoke vs canônico) — ver **PM-13**, que rastreia essa
> distinção; não é mais um blocker de "sem testemunha".

* **ID:** PM-04 · **Severidade:** Média (arquitetura oficial NAMCore sem testemunha) · **Risco da correção:** Baixo (test-only, aditivo)

* **Problema:** O motor ConvNet (`src/models/convnet/`) está **completo, despachado e
  unit-testado**, com teste *self-golden* de determinismo (`tests/golden_vectors.rs`
  `test_golden_vectors_convnet_test`, threshold em `validation.rs:660-663`), porém **sem
  testemunha externa**: o oráculo f64 retorna zeros para ConvNet
  (`src/testing/reference_oracle.rs:282`, ramo `_ => vec![0.0; ...]`) e o `render` do C++
  **não consegue** gerar golden porque o ConvNet do **NAMCore v0.5.3** é arquiteturalmente
  incompatível com o **NAM 0.5.4** do Rust:

  | Aspecto      | C++ v0.5.3                           | NAM-rs 0.5.4                   |
  | ------------ | ------------------------------------ | ------------------------------ |
  | Canais       | único, compartilhado                 | por-bloco                      |
  | Kernel       | fixo `=2`                            | por-bloco                      |
  | Head         | matriz × vetor, **sem** Conv1D/ativ. | `PostStackHead` Conv1D + ativ. |
  | `head_scale` | ausente                              | presente                       |

  `tests/fixtures/golden_gen_build.sh:236-261,321` marca o ConvNet como **SKIP esperado**
  ("C++ v0.5.3 ConvNet is architecturally incompatible — known").

* **Proposta de Solução (NAMCore-safe, sem upgrade do ref C++):**

  1. Implementar `oracle_convnet_forward()` em f64 exato em `reference_oracle.rs`: Conv1D causal
     por bloco → BatchNorm fundida (`scale·x + offset`) → ativação → `PostStackHead` (Conv1D +
     ativação) → `head_scale`. Espelha exatamente a topologia multi-bloco do NAM 0.5.4.
  2. Validar `oracle_convnet_forward` × engine de produção f32 com `convnet_test.nam`
     (gerável por `tests/fixtures/generate_b1_2_fixtures.py`): exigir **ESR < 1e-12**.
  3. **Âncora NumPy independente** (3ª implementação) para provar que o oráculo não é espelho
     do engine (cadeia §9.2 / Regra 6); definir `CONVNET_ESR_LIMIT` em `tests/common/constants.rs`
     análogo a `WAVENET_ESR_LIMIT`/`A2_ESR_LIMIT`.
  4. Adicionar gates em `tests/reference_oracle_f64.rs` (`test_oracle_convnet`,
     `test_oracle_vs_python_anchor_convnet`) e, opcionalmente, um golden derivado do oráculo
     para proteção de regressão.

* **Resultado:** ConvNet ganha a mesma *testemunha de matemática ideal* dos demais motores,
  **sem** depender de um `render` C++ compatível. A cross-validation C++ permanece **diferida e
  corretamente documentada** (bloqueada no upgrade do ref C++), agora com justificativa explícita
  no §5/§13.1.

### PM-05 — Integração de capturas **reais** A2-FiLM (`wavenet_a2_max`)

* **ID:** PM-05 · **Severidade:** Média (timbres reais ainda não exercitados) · **Risco da correção:** Médio (depende de obter modelo real; gated em PM-03)
* **Problema:** O motor **suporta FiLM** (via `WaveNetA2Dyn`), mas todos os goldens FiLM usam
  **pesos sintéticos** (`tests/fixtures/generate_a2_fixtures.py:421-440`). Não há captura de
  amplificador real (estilo `wavenet_a2_max.nam`) na suíte, então a fidelidade de timbre real
  sob FiLM não é validada.
* **Proposta de Solução:**
  1. **Após PM-03** (com `RF1` caracterizado e threshold significativo), curar/obter um modelo
     FiLM real como fixture **não-distribuível** (`tests/fixtures/README.md` §Non-Distributable
     Model Management).
  2. Gerar golden via `render` C++ (caminho WaveNet genérico) e adicionar teste em
     `tests/golden_vectors.rs` — o harness já roteia FiLM→`WaveNetA2Dyn` e **assere** o roteamento
     (`golden_vectors.rs:1596,1654`).
  3. Calibrar threshold em `validation.rs` com margem honesta (Política de Calibração de Gates).
* **Dependência:** **gated em PM-03** — sem caracterizar a divergência FiLM, o threshold de um
  modelo real seria arbitrário (risco de mascaramento).

### PM-06 — `SlimmableWavenet` (single-net channel slicing): épico diferido — escopo e critério

* **ID:** PM-06 · **Severidade:** Baixa (feature de nicho, caso prático já coberto) · **Risco da correção:** Nulo (decisão/escopo)
* **Problema:** Risco de **confusão de escopo**: o `SlimmableWavenet` (uma única rede `.nam`
  declarando múltiplas larguras de canal, fatiadas em runtime) está **genuinamente diferido**,
  enquanto o `SlimmableContainer` **multi-modelo** (sub-redes independentes + crossfade) está
  **implementado e testado** (`src/models/container.rs`, `tests/container_slimmable.rs`, 10+ testes).
* **Proposta de Solução:**
  1. **Manter diferido** — o caso de uso prático (qualidade adaptativa sob pressão de CPU) já é
     atendido pelo `SlimmableContainer`.
  2. Documentar a **fronteira de escopo** explicitamente (feito no §6/§13 do `cpp_parity_map.md`).
  3. Definir **critérios de aceitação** para implementação futura: parser de larguras múltiplas
     em um único `.nam`, slicing de pesos por canal em runtime RT-safe, e paridade com o
     `SlimmableWavenet` do NAMCore — só justificável se modelos oficiais nesse formato surgirem.

### PM-07 — Robustez do harness live v2: **SKIP silencioso** de taxa (anti-masking)

* **ID:** PM-07 · **Severidade:** Média (brecha latente de mascaramento) · **Risco da correção:** Baixo (test-only)
* **Problema:** Em `run_v2_multi_sr` (`tests/cpp_parity.rs:578`), quando o `render` do C++ rejeita
  uma taxa (modelos com campo `sample_rate` explícito em taxas não-nativas), `run_render_comparison`
  retorna cedo com `eprintln!("SKIP: ...")` (`tests/cpp_parity.rs:297-317`). O `run_v2_multi_sr`
  só captura *panics* (`:599-607`), de modo que um **SKIP normal é tratado como aprovação
  silenciosa**. Hoje **nenhuma taxa é efetivamente mascarada** (os modelos cobertos comparam de
  fato), mas é uma **brecha estrutural** que viola o espírito da **Regra 7** da *Gate Calibration
  Policy* (`docs/perceptual_validation.md:1001`): uma taxa que sempre der SKIP jamais reportaria
  falha.
* **Proposta de Solução (test-only, fortalece a "warrior suite"):**
  1. Fazer `run_v2_multi_sr` **rastrear o desfecho por-taxa** (comparado / SKIP) e retornar a
     contagem de comparações **genuínas**.
  2. **Assertar** que ≥1 comparação real ocorreu (ao menos a taxa nativa) e que a contagem bate
     com o esperado para o modelo (falhar se **todas** as taxas derem SKIP, ou se uma taxa que
     *deveria* comparar der SKIP inesperado).
  3. Emitir um resumo explícito por-taxa no `--nocapture` para auditoria.

### PM-08 — Registro único de findings e correção de **referências quebradas** [RESOLVIDO]

* **ID:** PM-08 · **Severidade:** Média (rastreabilidade) · **Risco da correção:** Nulo (doc-only)
* **Problema:** Não há registro central de findings de paridade. O `cpp_parity_map.md`
  referenciava `TODO-findings.md F1/F2/I6` (conteúdo anterior, agora removido);
  `docs/fastmath-approximations.md:346` referencia `TODO-problemas.md#P1` (**inexistente**); e o
  código usa *tags* `RF1`/`RF3`/`RF7` sem definição central.
* **Proposta de Solução:**
  1. Adotar **este documento** como registro canônico (prefixo `PM-##`), mapeando as *tags* vivas:
     **`RF1` → PM-03** (FiLM), **`RF3`** → goldens v2 dinâmicos por design (§3.3), **`RF7` → PM-02**
     (Lite resolvido).
  2. Corrigir todas as refs quebradas: `cpp_parity_map.md` ✅ (PM-01); `fastmath-approximations.md`
     em PM-02.
  3. Padronizar convenção `RF#`/`PM#` e garantir que *tags* em código apontem para um finding
     existente (invariante verificável, análogo ao `tests/threshold_calibration.rs`).
* **Conclusão (2026-06-30):**
  1. Referência `TODO-problemas.md#P1` (Lite) removida em S9.4 (PM-02) — substituída por
     `cpp_parity_map.md` §9.1 e PM-02 no próprio `fastmath-approximations.md` §9.4.
  2. Referências `F1`/`F2`/`I6` removidas do `cpp_parity_map.md` em PM-01 — substituídas
     por PM-03 (FiLM), PM-04 (ConvNet), e `perceptual_validation.md` (caps HF).
  3. Cross-reference canônica estabelecida em `fastmath-approximations.md` §9.6 e
     `cpp_parity_map.md` See Also, ambas apontando para este registro (`TODO-findings.md`).
  4. Mapeamento de *tags* vivas: `RF1→PM-03`, `RF3→§3.3`, `RF7→PM-02` — verificado consistente
     entre `tests/common/validation.rs` e `docs/perceptual_validation.md`.
  5. Referência auto-contraditória em `fastmath-approximations.md` §8 References
     (`§6 (this section)`) corrigida para `§6 (anti-subnormal companion)`.

---

## Épicos (Agrupamentos) — Auditoria de Paridade

> Ordenados por relação **valor/risco/sequência**. Épicos A e D são *quick wins* de baixíssimo
> risco; B e C entregam testemunhas independentes e cobertura real; E permanece diferido.

### Épico A — Sincronização Documental de Paridade (PM-01, PM-02, PM-08) [DONE]

* **Risco/Criticidade:** Nulo a Baixo (doc-only). **Sequência:** imediata.
* Alinha toda a documentação ao estado real do motor e elimina referências quebradas/alarmes
  falsos. **PM-01 aplicado** ao `cpp_parity_map.md`; **PM-02** resolvido (`fastmath-approximations.md`
  §9.4); **PM-08** resolvido (registro canônico + correção de todas as referências quebradas).

### Épico B — Testemunhas Independentes (Oráculo f64) (PM-04, PM-03) [DONE]

* **Risco/Criticidade:** Baixo-Médio (test-only; produção só muda sob bug comprovado). **Alto valor.**
* Estende a cadeia de confiança de 4 camadas do §9.2 a duas lacunas: **ConvNet** (PM-04, sem
  testemunha) e **FiLM A2** (PM-03, sem classificação inerente-vs-bug). **Ordem sugerida:**
  PM-04 primeiro (mais simples, puramente aditivo, sem produção), depois PM-03 (decide `RF1`).
* **Crítico/atenção:** manter a **independência** do oráculo (âncora NumPy, Regra 6) — não
  espelhar o engine; e jamais relaxar gate para mascarar (Regra 7).

### Épico C — Cobertura de Modelos Reais A2-FiLM (PM-05) [DONE]

* **Risco/Criticidade:** Médio (depende de obter captura real). **Sequência:** após Épico B (PM-03).
* Eleva os goldens FiLM de pesos sintéticos para timbres reais, fechando o item "A2 official
  FiLM models" do §13 com thresholds significativos.
* **Nota do PO:** Infelizmente ainda não temos os aquivos.nam reais disponíveis já estão em `tests/fixtures/models-nondist/`.
  * Se nenhum arquivo lá ou em `tests/fixtures/models/`, vamos ter de nos conformar com capturas sintéticas.
  * Documente isto de forma muito clara em `docs/cpp_parity_map.md`seção `13. Pending / Open Work`.

### Épico D — Épico Diferido: `SlimmableWavenet` (PM-06) [DONE]

* **Risco/Criticidade:** Nulo (decisão de escopo). **Sequência:** sem urgência.
* Mantém o single-net channel slicing diferido com critérios de aceitação claros; o caso prático
  já é coberto pelo `SlimmableContainer`.
* **Nota do PO:** Documente isto de forma muito clara em `docs/cpp_parity_map.md`seção `13. Pending / Open Work`.

### Épico E — Robustez da Suíte de Testes (PM-07) [DONE]

* **Risco/Criticidade:** Baixo (test-only). **Sequência:** independente; pode ir junto com o Épico A.
* Fecha a brecha latente de SKIP silencioso no harness live v2, em conformidade com a Regra 7.
* **Pedido do PO:** Aproveite esta ocasião para acionar a skill "revisor-auditor" focada na role "Correctness Auditor".
  * Muito especificamente na suíte de testes ativadas por utils/tests-quick.sh, utils/tests-long.sh, utils/build-release.sh e utils/tests-performance-regression.sh.
  * Assegurar que tanto aqueles scripts em si, quanto os testes que eles acionam - todos corretos e prontos para serem graduados como os perfeitos "guardiães da qualidade" do nam-rs.
  * **Verificação (PM-07 ✅ confirmado):** `run_v2_multi_sr_impl` (`tests/cpp_parity.rs:570+`) agora rastreia o desfecho por-taxa e **assere** (a) ≥1 taxa concluída (sem skip total silencioso) e (b) o conjunto de taxas concluídas == taxas esperadas do modelo (sem skip parcial silencioso, Regra 7). Resolvido conforme proposto.

---

## Auditoria de Paridade — Rodada 2: Verificação contra a Referência C++ v0.5.4 (2026-06-30)

> **Gatilho:** auditoria crítica de "feature-completeness e correção inquestionável" + revisita do
> espelho `tests/fixtures/NeuralAmpModelerCore/` + análise do `testes.log` + certificação dos
> scripts de teste como "guardiães da qualidade" (pedido do PO no Épico E).
>
> **Descoberta central:** a cópia de trabalho da referência C++ **NÃO está mais em v0.5.3** — está em
> **v0.5.4** (`git describe` = `v0.5.4`, commit `1f42f88`), enquanto `golden_gen_build.sh` e os docs
> fixam **v0.5.3** (`9c7b185`). A revisita ao código C++ v0.5.4 (ground truth) confirmou e refinou
> vários findings e revelou **lacunas reais de feature-completeness** vs os modelos oficiais.

## Estado verificado do `testes.log` (última bateria)

* **Resultado:** ✅ **0 falhas**, 0 panics/SIGSEGV/abort; 391 passes + 151 ignored nas integrações;
  pipeline encerrou com "✓ Todos os estágios da auditoria passaram com sucesso!".
* **Cobertura do log:** o log é de `build-release.sh ; tests-performance-regression.sh` (suíte rápida
  * goldens + heap-audits + RT gates + benchmarks). Os **151 ignored são os testes de interop C++ ao
    vivo** (`live_cross_validation_*`) — **não exercitados** neste log (exigem `tests-long.sh` +
    toolchain C++). Ver **PM-14**.
* **Anomalia de observabilidade:** a linha-sumário (`test result: ok. …`) da suíte de unidade `lib`
  (`running 1070 tests`, linha 568) **não foi capturada** no log (apenas as linhas `... ok`
  individuais). Ver **PM-14**.

## Modelos oficiais NAMCore v0.5.4 (`example_models/`) × loader nam-rs

| Modelo oficial                                   | Carrega?         | Motor                  | Observação                                                           |
|:------------------------------------------------ |:----------------:|:---------------------- |:-------------------------------------------------------------------- |
| `wavenet_a1_standard.nam`, `my_model.nam`        | ✅               | `WaveNetModel<16,3,8>` | catálogo Standard                                                    |
| `wavenet.nam` (CH=3 free)                        | ✅               | `WaveNetModelDyn`      | geometria livre                                                      |
| `lstm.nam` (1×3)                                 | ✅               | `LstmModel1`           | —                                                                    |
| `wavenet_condition_dsp.nam`                      | ✅               | `WaveNetModelDyn`      | `condition_size=3` + sub-DSP                                         |
| `A2.nam`, `slimmable_container.nam`              | ✅               | `ContainerModel`       | multi-submodelo                                                      |
| **`wavenet_a2_max.nam`** (A2 oficial/flagship)   | ❌ **rejeitado** | —                      | **PM-10** — FiLM+cond=8+head1x1+gating em WaveNet genérico           |
| **`slimmable_wavenet.nam`** (single-net slicing) | ❌ **rejeitado** | —                      | **PM-12** — campo `slimmable` explicitamente rejeitado (fail-closed) |

> As rejeições são **fail-closed** (carga falha com erro; nenhum áudio incorreto é produzido). Há
> teste de lacuna ativo para `a2_max` (`test_loader_gap_wavenet_a2_max`, assere `is_err()`).

---

## Findings (Constatações) — Rodada 2

### PM-09 — Deriva de versão da referência C++ (v0.5.4 ⇄ pin v0.5.3) sem enforcement [Won't Do]

* **Nota do PO:** Não é uma problema real. A referência ao github mais recente é a fonte "e ponto final". Documente isto.

* **ID:** PM-09 · **Severidade:** Alta (reprodutibilidade + risco de mascaramento) · **Risco da correção:** Médio

* **Problema:**

  * `tests/fixtures/NeuralAmpModelerCore/` **não é um submódulo rastreado** (não há `.gitmodules`; não
    está no índice git do repo-pai) — **não há pin reproduzível**. A cópia local está em **v0.5.4**
    (`1f42f88`), mas `tests/fixtures/golden_gen_build.sh:60` fixa `NAM_CORE_COMMIT=9c7b185…` (v0.5.3) e
    os docs descrevem v0.5.3 como canônica.
  * Os **goldens** commitados foram gerados em **v0.5.3**; os testes **live** (`tests/cpp_parity.rs`)
    compilam o `render` a partir da **cópia de trabalho atual (v0.5.4)**. Resultado: **skew silencioso**
    — Rust×golden(v0.5.3) vs Rust×render(v0.5.4), com thresholds calibrados para v0.5.3.
  * `utils/tests-long.sh:54-55` **captura** `CURRENT_CORE_SHA` mas apenas o **ecoa** — **nunca assere**
    que bate com o commit de geração dos goldens.

* **Proposta de Solução:**

  1. **Decidir a versão canônica.** Verificar empiricamente se v0.5.4 muda numéricos vs v0.5.3 (rodar
     `cpp_parity` ao vivo nos dois commits). Recomendado: **fixar v0.5.4** como nova canônica *se* os
     goldens v0.5.3 continuarem dentro dos gates; caso contrário, **resetar a cópia para `9c7b185`**.
  2. **Enforcement de pin:** adicionar asserção dura em `tests-long.sh` e `golden_gen_build.sh`
     (`[ "$CURRENT_CORE_SHA" = "$EXPECTED_CORE_SHA" ] || fail`), e gravar `EXPECTED_CORE_SHA` em um
     arquivo rastreado (ex.: `tests/fixtures/.namcore_pin`).
  3. **Sincronizar docs** (`cpp_parity_map.md`) com o commit/versão efetivo (feito nesta auditoria, com
     nota de deriva).
  4. Se migrar para v0.5.4: **regenerar todos os goldens** e **recalibrar thresholds** (Política de
     Calibração de Gates), nunca afrouxando gate para mascarar.

* **Por que importa:** sem pin, um `git pull` no espelho pode silenciosamente trocar a referência e
  invalidar (ou mascarar) toda a prova de paridade — o oposto de um "guardião da qualidade".

### PM-10 — Modelo A2 oficial `wavenet_a2_max.nam` não carregável (motor A2 dinâmico genérico ausente)

* **ID:** PM-10 · **Severidade:** Alta (feature-completeness do flagship A2) · **Risco da correção:** Alto (novo motor) — interino seguro
* **Problema:** O flagship oficial A2 (`example_models/wavenet_a2_max.nam`, **md5-idêntico** ao fixture
  `tests/fixtures/models/wavenet_a2_max.nam`) é `architecture=WaveNet`, **1 layer-array**, `channels=4`,
  `bottleneck=4`, `condition_size=8`, `activation={"type":"Softsign"}` (objeto), **todas** as 8 chaves
  FiLM ativas, `head1x1`, `secondary_activation`, `groups_input_mixin=4`, e um sub-modelo `condition_dsp`
  com gating/blending. O nam-rs **rejeita** esse modelo (`test_loader_gap_wavenet_a2_max`,
  `tests/golden_vectors.rs:945`, assere `is_err()`).
  * **Causa precisa:** o suporte FiLM do nam-rs vive no `WaveNetA2Dyn`, alcançado **apenas** quando o
    modelo casa a **assinatura estrutural A2 de 23 camadas** (`is_a2_shape`, CH∈{3,8}) — i.e., os
    fixtures **sintéticos** `wavenet_a2_film_{lite,full}` (CH 3/8). O `a2_max` é um **WaveNet genérico
    de 1 array com FiLM** (kernel único, dilations `[1,2]`), que **não** casa a assinatura A2 → cai no
    motor **A1 dinâmico**, que **não** tem FiLM/gating/head1x1 → rejeição segura.
* **Correção do registro anterior:** isto **refina o PM-05 e desfaz o exagero** da Rodada 1 ("A2 general
  engine 🟢 implementado"). O que está implementado/golden-testado é o A2 **23-camadas** sintético
  (gated/blended/FiLM CH 3/8); a **geometria oficial** (`a2_max`) **não é suportada**.
* **Proposta de Solução:**
  1. Implementar o **motor A2 dinâmico genérico** (o "`WaveNetModelDynA2`" já mencionado no comentário
     do teste de lacuna): WaveNet de geometria livre + FiLM + `condition_size>1` + `head1x1` +
     gating/blending + `condition_dsp` + ativação em forma de objeto.
  2. **Validar com a referência C++ v0.5.x** (que **suporta** esse modelo — `wavenet/model.cpp` tem 85
     menções a FiLM já em v0.5.3): gerar golden de `wavenet_a2_max.nam` + testemunha do oráculo f64
     (estende PM-03). Subsome **PM-05** (capturas reais A2-FiLM passam a ser carregáveis/testáveis).
  3. **Interino seguro (já vigente):** manter a rejeição fail-closed; **melhorar a mensagem de erro**
     para ser orientada ao usuário ("modelo A2/FiLM oficial ainda não suportado — use Standard/…").
* **Risco:** alto (motor novo). O interino (rejeição) **não tem risco de correção** — nunca produz
  áudio errado.

### PM-11 — `activation` em forma de objeto `{"type":"…"}` cai silenciosamente para Tanh (caminho A1) **[RESOLVIDO — S12.1]**

* **ID:** PM-11 · **Severidade:** Média (risco latente fail-open) · **Risco da correção:** Baixo
* **Status:** ✅ **RESOLVIDO (2026-06-30, S12.1)** — o parser de `activation` em `src/loader/nam_json/model.rs` agora rejeita explicitamente a forma de objeto com erro de deserialização ("unsupported activation format"). Teste de unidade adicionado (`activation_parser_test`) assegura comportamento fail-closed. Ver [`TODO-sprints.md`](../TODO-sprints.md) S12.1.
* **Problema:** Em `src/loader/nam_json/model.rs` o parser de `activation` trata **apenas** string e
  array; a forma de **objeto** `{"type":"Softsign"}` cai no ramo `_ => None`, e o default vira "Tanh".
  Hoje está **mascarado** pelo `a2_max` ser rejeitado por outras razões, mas é uma brecha **fail-open**:
  qualquer modelo futuro com ativação-objeto que passe pelas demais checagens rodaria com a **ativação
  errada, silenciosamente**.
* **Proposta de Solução:** ou (a) **parsear** a forma de objeto (extrair `"type"`), ou (b) **falhar
  fechado** (rejeitar ativação não-reconhecida com erro explícito). Recomendado: (b) agora (defensivo) +
  (a) quando o motor A2 genérico (PM-10) precisar. Adicionar teste unitário com ativação-objeto.
* **Risco:** baixo (mudança pequena de parser + teste).

### PM-12 — `slimmable_wavenet.nam` oficial não carregável; campo `slimmable` rejeitado explicitamente (fail-closed) **[RESOLVIDO — S12.2]**

* **ID:** PM-12 · **Severidade:** Média · **Risco da correção:** Baixo (defensivo) / Alto (motor completo, diferido)
* **Status:** ✅ **RESOLVIDO — lado defensivo (2026-06-30, S12.2).** O loader agora realiza verificação explícita da chave `slimmable` durante a validação de topologia e rejeita imediatamente modelos single-net que a possuam, com erro orientado ao usuário ("slimmable single-net weight slicing is not supported; use SlimmableContainer instead"). Teste de lacuna `test_loader_gap_slimmable_wavenet` (`tests/golden_vectors.rs`) valida a rejeição fail-closed. O motor de slicing single-net em runtime permanece **diferido** (épico PM-06). Ver [`TODO-sprints.md`](../TODO-sprints.md) S12.2.
* **Problema:** O modelo oficial de single-net channel slicing (`example_models/slimmable_wavenet.nam`,
  md5-idêntico ao fixture) tem `slimmable: {method:"slice_channels_uniform", kwargs:{allowed_channels:[1,2,3]}}`.
  O nam-rs **rejeita** (via checagem de A2/ativação não-Tanh), mas o campo `slimmable` é **silenciosamente
  descartado** (não há caminho explícito "slimmable não suportado"). Nenhum teste carrega o modelo oficial.
  Confirma o **PM-06** com o artefato real.
* **Proposta de Solução:**
  1. **Rejeição fail-closed explícita** para `slimmable` single-net, com erro claro + **teste de lacuna**
     (espelhando `test_loader_gap_wavenet_a2_max`) — agora, barato e seguro.
  2. Motor de slicing single-net em runtime: **diferido** (épico PM-06).
* **Risco:** baixo (defensivo) agora; alto (motor) diferido.

### PM-13 — ConvNet do nam-rs é um formato bespoke sem contrapartida canônica NAMCore

* **ID:** PM-13 · **Severidade:** Média (clareza de paridade + interop real) · **Risco da correção:** Baixo (doc + decisão)
* **Problema (refina PM-04):** A revisita ao C++ confirmou que o ConvNet **canônico** do NAMCore — em
  **v0.5.3 e v0.5.4** — é: `channels` único compartilhado, **kernel fixo = 2** (`convnet.cpp:57`,
  comentário "HACK 2 kernel"), `_Head` = multiplicação de matriz + bias (**sem** Conv1D, **sem** ativação,
  **sem** `head_scale`), config **plana** (`channels`/`dilations`/`batchnorm`/`activation` únicos). O
  ConvNet do **nam-rs** usa um formato **bespoke**: `config.layers[]` (kernel/channels/ativação por bloco)
  * `head` Conv1D + `head_scale` (visto em `convnet_test.nam`, gerado por `generate_b1_2_fixtures.py`).
  * **Consequência:** o nam-rs **não carrega** um ConvNet canônico real, e o C++ **não carrega** o
    `convnet_test.nam`. **Não existe ConvNet oficial** em `example_models/` (arquitetura efetivamente
    descontinuada upstream). Logo o blocker **não** é "v0.5.3 vs v0.5.4" — é **"formato bespoke vs
    canônico (ambas as versões)"**.
* **Proposta de Solução:**
  1. **Documentar honestamente** (feito nesta auditoria) que o ConvNet nam-rs é uma arquitetura própria,
     **sem interop NAMCore**.
  2. Como ConvNet é descontinuado upstream, **manter o bespoke + oráculo f64 (PM-04)** como testemunha
     de matemática ideal; **decidir** se vale (baixa prioridade) adicionar um caminho de loader para o
     formato **plano canônico** (kernel=2, head-matriz) para compat com modelos reais antigos — e, se
     sim, gerar 1 golden plano via C++ para fechar interop.
* **Risco:** baixo (essencialmente documentação + decisão de escopo).
* **Decisão do PO:** Já que é uma arquitetura legada, vamos nuka-lo e decretar e documentar esta decisão.

### PM-14 — Certificação dos scripts-guardião e observabilidade da bateria [Wont-Do] ✅ Documentado (S12.5)

* **Nota do PO:** Não é uma problema real. A referência ao github mais recente é a fonte "e ponto final". Documente isto.
* **Decisão documentada em 2026-06-30 (S12.5):** Decreto do PO formalizado em `docs/cpp_parity_map.md` §13.1 e §13 (tabela). Nenhuma mudança adicional na infraestrutura de observabilidade da bateria é necessária.
* **ID:** PM-14 · **Severidade:** Média (observabilidade/cobertura) · **Risco da correção:** Baixo (infra de teste)
* **Auditoria dos 4 scripts (pedido do PO):**
  * ✅ **Pontos fortes:** `tests-long.sh` tem pré-voo de goldens abrangente (v1 + v2 por grupo de SR),
    agregação `no-fail-fast` com sumário por fase, manifesto de frescor de goldens, heap-audits, RT
    gates e validação CLAP; `tests-quick.sh` faz clippy estrito + integridade SHA256 do binário CLAP
    entre fases; `tests-performance-regression.sh` usa pinagem de core (`taskset`) e baseline criterion;
    `build-release.sh` (PGO+BOLT) degrada graciosamente. **PM-07 confirmado resolvido.**
  * ⚠ **Lacuna 1 (→ PM-09):** `tests-long.sh` não **assere** o commit da referência C++ (só ecoa) →
    skew de versão silencioso.
  * ⚠ **Lacuna 2 (observabilidade):** no `testes.log`, a linha-sumário da suíte `lib` (1070 testes) não
    foi capturada; e os **151 ignored (interop C++ ao vivo) não rodam** na pipeline padrão
    (`build-release.sh`/`tests-performance-regression.sh`) — só em `tests-long.sh`.
* **Proposta de Solução:**
  1. **PM-09** (asserção de pin) — eleva a integridade do guardião live.
  2. **Capturar** a linha-sumário do `lib` (ordenação stdout/stderr; ex.: `cargo test … 2>&1` no
     pipeline de log da bateria) para tornar a contagem auditável.
  3. **Rodar e capturar `tests-long.sh`** como parte do fechamento de auditoria — **executado pelo
     desenvolvedor humano** (o próprio `tests-long.sh:17` proíbe a IA de executá-lo) — para provar a
     interop C++ ao vivo, não apenas os goldens.
* **Risco:** baixo (infra/observabilidade).

---

## Épicos (Agrupamentos) — Rodada 2

### Épico F — Integridade da Referência C++ (PM-09) [WONT-DO]

* **Risco/Criticidade:** Médio-Alto. **Sequência:** **primeiro** — todos os demais dependem de uma
  referência confiável e pinada.
* Decide a versão canônica (v0.5.3 vs v0.5.4), adiciona enforcement de pin e sincroniza docs/goldens.
  É o pré-requisito para confiar em qualquer prova de paridade.
* **Decisão do PO:** Não é um problema real. A referência ao github mais recente é a fonte "e ponto final". Documente isto.

### Épico G — Feature-Completeness A2 Oficial (PM-10, PM-03, PM-05) [ALTO VALOR] [MAPPED S13]

* **Risco/Criticidade:** Alto (motor A2 dinâmico genérico). **Sequência:** após Épico F.
* Implementa o motor A2 genérico para carregar `wavenet_a2_max.nam` (FiLM real + `condition_size>1` +
  `head1x1` + gating/blending), validado por golden C++ + oráculo f64 (PM-03). É o que falta para o A2
  ser **inquestionavelmente feature-completo** vs NAMCore.

### Épico H — Robustez de Carregamento "fail-closed" (PM-11, PM-12) [DONE]

* **Risco/Criticidade:** Baixo. **Sequência:** imediata (independente).
* Garante que toda entrada não suportada (ativação-objeto, `slimmable` single-net) **falhe fechada** com
  erro claro + teste de lacuna — nunca silenciosamente errada. Eleva a confiança no loader como guardião.

### Épico I — Sincronização Documental & ConvNet (PM-13, e doc de PM-09/PM-10/PM-12) [DONE]

* **Risco/Criticidade:** Nulo a Baixo (doc-only). **Sequência:** junto com F/H.
* Documenta o ConvNet bespoke (sem interop), a deriva de versão e as lacunas de modelos oficiais com
  precisão no `cpp_parity_map.md` §13 (feito nesta auditoria; manter sincronizado).
* **Decisão do PO:** Já que é uma arquitetura legada, vamos nuka-lo e decretar e documentar esta decisão.

### Épico J — Observabilidade & Cobertura da Bateria (PM-14) [TEST-INFRA] [WONT-DO]

* **Risco/Criticidade:** Baixo. **Sequência:** junto com F.
* Captura a linha-sumário do `lib`, garante execução/registro do `tests-long.sh` (interop ao vivo) e
  integra a asserção de pin (PM-09). Fecha o ciclo do "guardião da qualidade".
* **Decisão do PO:** Não é um problema real. A referência ao github mais recente é a fonte "e ponto final". Documente isto.
* **Documentado em 2026-06-30 (S12.5):** Decreto do PO formalizado em `docs/cpp_parity_map.md` §13.1.

---

## Épicos (Agrupamentos) — Rodada 3 (Auditoria de S13)

### Épico K — Multi-Array A2 (PM-15) [TO-DO]

* **Risco/Criticidade:** Alto (hot-path DSP). **Sequência:** imediata (conclui o Épico G).
* A validação final do modelo `wavenet_a2_max.nam` (Sprint S13.3) expôs uma limitação arquitetural: o A2 dinâmico genérico restringe-se a 1 único *layer array*. O C++ orquestra múltiplos arrays nativamente (`std::vector<LayerArray>`), necessário para carregar o sub-modelo `condition_dsp` que possui 2 arrays com FiLM. Este épico estende o motor A2 para roteamento em cascata.

---

### PM-15 — Suporte a Múltiplos Arrays (Cascade) no Motor A2 Genérico

* **ID:** PM-15 · **Severidade:** Alta (Blocker para Feature-Completeness A2 e `condition_dsp` do `wavenet_a2_max.nam`) · **Risco da correção:** Alto (Hot-path DSP, alocação)
* **Contexto/Auditoria (revisor-auditor):**
  Auditoria da implementação de referência C++ (`NAM/wavenet/model.cpp`) revela que a WaveNet armazena nativamente um vetor de camadas (`std::vector<LayerArray>`) e as processa sequencialmente. O output de uma alimenta o input da outra, e as saídas de *head* de todas elas são acumuladas (`final_head_outputs`). A convolução opcional `head1x1` é aplicada apenas sobre esta soma final (linha 787 de `model.cpp`).
  No `nam-rs`, a topologia A1 suporta múltiplos arrays (`WaveNetLayerArrayDyn`), mas o novo motor A2 dinâmico (`WaveNetA2Dyn` implementado na Sprint S13) está rigidamente fixado em `layers.len() == 1`.
  Devido a isto, o sub-modelo `condition_dsp` do `wavenet_a2_max.nam`, que possui 2 arrays em sua topologia e chaves FiLM, não é qualificado pela verificação `is_a2_shape` e sofre fallback para a WaveNet A1 genérica. Por não suportar chaves FiLM, o parser A1 dinâmico falha alegando inconsistência de contagem de pesos (erro explicitamente mapeado no *Loader Gap* de S13.3: `consumed 368, total 1052`).
* **Proposta de Solução (planejador-arquiteto):**
  Transformar o modelo de execução do A2 genérico para suportar nativamente a topologia multi-array:
  1. Alterar o detector de topologia `is_a2_shape` (`src/loader/nam_json/topology/a2.rs`) para validar `layers.len() >= 1`.
  2. Implementar um wrapper de orquestração `WaveNetA2Cascade` (ou expandir o `WaveNetA2Dyn` internamente) para instanciar e gerenciar iterativamente os `LayerArrays`.
  3. No processamento DSP, repassar a saída ativada/processada de um array como entrada do próximo e acumular de forma contínua o `head_accum`.
  4. Extrair a aplicação da projeção `head1x1` do array individual e transferi-la para o roteador final pós-soma da cascata.
  5. Após consolidação da engine multi-array, remover os `#[ignore]` pendentes no `test_loader_gap_wavenet_a2_max` e no `test_golden_vectors_wavenet_a2_max`.

---

## Auditoria de Paridade — Rodada 4: Verificação pós-"Bug Storm" S14.2 (2026-07-01)

> **Escopo:** Análise holística focada na role **Compliance and Parity Auditor** após a
> "tempestade de bugs" dos commits S13.2 / S14.1 / S14.2 / `f4c9359` / `9606be7` / `44a5827`
> (motor A2 genérico + cascade + `condition_dsp` + `head1x1` agrupado para `wavenet_a2_max.nam`).
> Objetivo: assegurar correção integral e bug-free comparada à referência NAMCore, com atenção
> especial à qualidade da suíte de testes (barreira defensiva) e à atualização do
> `docs/cpp_parity_map.md`.

## Contexto: o que a auditoria encontrou?

A Sprint S14.2 (PM-15) introduziu suporte a multi-array cascade + `condition_dsp` + `head1x1`
agrupado para carregar o modelo flagship `wavenet_a2_max.nam` (v0.6.0, com `condition_dsp`
de 2 arrays). O modelo **passa a carregar** (`test_loader_gap_wavenet_a2_max`), mas a
**inferência está gravemente incorreta** — o oráculo f64 independente (validado contra âncora
NumPy a < 1e-12) diverge da produção em **93 dB** (ESR = 4.45e4). Quatro testes de fidelidade
foram silenciados com `#[ignore = "S14.2-followup: ..."]` em vez de corrigidos, e um teste
unitário ficou com asserção obsoleta (falhando ativamente). Um `println!` de debug foi
deixado no hot-path de produção (RT-safety defect).

---

## Findings (Constatações) — Rodada 4

### PM-16 — Bug de leitura de pesos `head1x1` agrupado no motor A2 dinâmico [CONFIRMADO, BUG REAL]

* **ID:** PM-16 · **Severidade:** Crítica (regressão de correção, modelo flagship inutilizável) · **Risco da correção:** Médio (hot-path, mas mudança é contagem de leitura)
* **Problema:** O carregamento de pesos `head1x1` no motor A2 dinâmico (`WaveNetA2Dyn`) lê
  `head_accum_size * h1_in_size` pesos quando deveria ler `channels * h1_in_size`. Quando
  `head1x1.out_channels != channels` (caso real: sub-modelo `condition_dsp` array[0] do
  `wavenet_a2_max.nam` com `channels=3, bottleneck=6, out_channels=6, groups=3`), a produção
  lê **12 pesos** em vez de **6**, consumindo pesos extras e **desincronizando o cursor**
  para todos os arrays subsequentes. O resultado é corrupção silenciosa da inferência.
* **Evidências (medidas diretamente):**
  * Sub-modelo `condition_dsp` isolado: **ESR = 3.26e2 (50.3 dB)** produção vs oráculo.
  * Modelo externo completo `wavenet_a2_max`: **ESR = 4.45e4 (93.0 dB)** produção vs oráculo.
  * Oráculo Rust f64 vs âncora NumPy f64: **ESR = 5.31e-16 (-152.8 dB)** — oráculo é correto.
  * Produção: max|output| ≈ 13.7; Oráculo: max|output| ≈ 0.026 (razão ~520×).
  * Local do bug: `src/models/a2/model/dynamic/build.rs:227` —
    `let channels = self.head_accum_size;` deveria ser `self.channels` para a *contagem* de
    leitura (o transposto de armazenamento mantém `head_accum_size` no acesso em runtime, mas
    o número de pesos na stream é `channels * (bottleneck/groups)`).
* **Confronto com referências independentes:**
  * Oráculo Rust (`src/testing/reference_oracle.rs:1124`): lê `ch * h1_in_size` onde
    `ch = arr.ch` (channels). Correto.
  * Âncora NumPy (`tests/fixtures/scripts/validate_oracle_f64.py:791`):
    `n_h1 = ch * head1x1_in; weights[cursor:cursor+n_h1].reshape(ch, head1x1_in)`. Correto.
  * C++ NAMCore (`NAM/wavenet/model.cpp`): o `head1x1` é uma `DenseLayer` dimensionada pelo
    número de *channels* do array, não por `out_channels`.
* **Proposta de Solução (planejador-arquiteto):**
  1. **Corrigir a contagem de leitura** em `build.rs:load_head1x1_weights`: ler
     `self.channels * self.h1_in_size` pesos (não `head_accum_size * h1_in_size`), preservando
     o layout transposto `head_accum_size × h1_in_size` para o acesso em runtime. Verificar
     se `transpose_dense_f32` está coerente com a nova contagem.
  2. **Validação de exaustão de stream**: adicionar asserção de que o cursor consome
     exatamente o total de pesos esperado para cada sub-modelo (defesa em profundidade).
  3. **Des-ignorar e corrigir** os 4 testes `S14.2-followup` (`reference_oracle_f64.rs:637,649,676`
     * `golden_vectors.rs:1952`) — eles são a testemunha deste bug e devem passar.
  4. **Teste de regressão dedicado**: adicionar um teste que carrega `wavenet_a2_max.nam`,
     roda inferência, e compara contra o oráculo f64 com gate `A2_GENERIC_ESR_LIMIT = 1e-9`.
* **Cobertura atual (lacuna):** o único teste *ativo* para `wavenet_a2_max` é um smoke test
  de carregamento (`test_loader_gap_wavenet_a2_max`); o teste de fidelidade está `#[ignore]`d,
  deixando esta regressão crítica **sem barreira defensiva**.

### PM-17 — `println!` de debug no hot-path de produção (RT-safety defect) [RESOLVIDO]

* **ID:** PM-17 · **Severidade:** Alta (RT-safety) · **Risco da correção:** Nulo
* **Problema:** O commit `44a5827` ("fixes") deixou um `println!("PROD COND FIRST 10: ...")`
  com `static AtomicBool PRINTED` dentro de `WaveNetA2Dyn::process_internal`
  (`src/models/a2/model/dynamic/process.rs:99-106`). Isto é um **defeito de RT-safety**: I/O
  bloqueante com lock global de stdout no path de áudio em tempo real, além de poluição de
  output. Pode causar jitter/xruns no thread de áudio.
* **Evidências:** `git show 44a5827 -- src/models/a2/model/dynamic/process.rs` introduziu o
  bloco `use std::sync::atomic::{AtomicBool, Ordering}; static PRINTED: AtomicBool = ...; println!(...)`.
* **Proposta de Solução (aplicada nesta auditoria):** Remoção cirúrgica do bloco de debug
  (10 linhas). Verificado: compila limpo, clippy limpo em `--features standalone` e
  `--features clap-plugin`.

### PM-18 — Asserção obsoleta em teste unitário pós-rework `head_accum_size` [RESOLVIDO]

* **ID:** PM-18 · **Severidade:** Média (teste ativamente falhando, barreira quebrada) · **Risco:** Nulo
* **Problema:** O commit `9606be7` adicionou os parâmetros `head_accum_size` e `h1_in_size`
  ao construtor `WaveNetA2Dyn::new` e mudou a alocação de `head1x1_w` para
  `head_accum_size * h1_in_size`, mas o teste
  `test_wavenet_a2_dyn_bottleneck_neq_channels` (`src/models/a2/model/dynamic_test.rs:101`)
  manteve a asserção obsoleta `head1x1_w.len() == 4*8` (esperando `channels × bottleneck = 32`),
  quando o novo contrato correto é `head_accum_size × h1_in_size = 4×4 = 16`. O teste falhava
  **ativamente** em `cargo test --lib`.
* **Evidências:** `cargo test --lib models::a2::` → `1 failed` antes da correção;
  `assertion left == right failed: left: 16, right: 32`.
* **Proposta de Solução (aplicada):** Atualizar asserção para `head1x1_w.len() == 4*4` e
  `head1x1_b.len() == 4` (consistente com os args `new(1,8,4,1,4,4,...)` →
  `head_accum_size=4, h1_in_size=4`). Verificado: 235 testes A2 passam.

### PM-19 — Propagação incompleta de `set_max_buffer_size`/`reset` para `condition_dsp` aninhado

* **ID:** PM-19 · **Severidade:** Média (potencial bug latente) · **Risco da correção:** Baixo
* **Problema:** `WaveNetA2Dyn::set_max_buffer_size` e `WaveNetA2Dyn::reset`
  (`src/models/a2/model/dynamic/mod.rs:341,396`) **não propagam** para o sub-modelo
  `condition_dsp` aninhado, ao contrário do A1 WaveNet (`src/models/wavenet/mod.rs:124` que
  chama `cond_dsp.set_max_buffer_size(max_buf)`). Se o host chamar `reset`/`set_max_buffer_size`
  no motor A2 com `condition_dsp`, o sub-modelo mantém buffers do tamanho original
  (`WAVENET_MAX_NUM_FRAMES=64`), que pode ser insuficiente para blocos maiores.
* **Nota:** O `prewarm` já propaga (`cond_dsp.prewarm(0)`), e o caminho de produção não
  chama `reset` no fluxo normal (construção usa o `max_buf` padrão). Mas é uma
  **inconsistência estrutural** vs A1 e uma fonte potencial de bug se o motor for reusado
  com `set_max_buffer_size` em runtime (cenário CLAP com buffer dinâmico).
* **Proposta de Solução:** Espelhar o pattern do A1: em
  `WaveNetA2Dyn::set_max_buffer_size` e `reset`, propagar para
  `self.condition_dsp.as_mut()` quando `Some`. Adicionar teste de regressão que chama
  `set_max_buffer_size(256)` após construção e verifica que o `condition_dsp` redimensionou.

### PM-20 — Quatro testes de fidelidade silenciados (`#[ignore]` "S14.2-followup") em vez de corrigidos

* **ID:** PM-20 · **Severidade:** Alta (barreira defensiva neutralizada) · **Risco:** Nulo (des-ignorar após PM-16)
* **Problema:** Quatro testes que validam `wavenet_a2_max.nam` (modelo flagship) foram
  marcados `#[ignore = "S14.2-followup: ..."]`:
  * `tests/reference_oracle_f64.rs:637` — `test_oracle_a2_generic` (ESR gate)
  * `tests/reference_oracle_f64.rs:649` — `test_decomposition_a2_generic`
  * `tests/reference_oracle_f64.rs:676` — `test_combined_simulation_a2_generic`
  * `tests/golden_vectors.rs:1952` — `test_golden_vectors_wavenet_a2_max`
* **Contexto:** Os ignores documentam o ESR=1e5/50dB "requiring investigation". A auditoria
  (PM-16) **encontrou e caracterizou a root cause** — estes testes podem ser des-ignorados
  após a correção de PM-16.
* **Violação de contrato documental:** `reference_oracle_f64.rs:146` afirma
  *"Never leave the test_oracle_vs_python_anchor_* tests #[ignore]d without a tracked task."*
  — estes ignores não têm task de restauração no `TODO-findings.md`/`TODO-sprints.md` até
  esta Rodada 4.
* **Proposta de Solução:** Após corrigir PM-16, des-ignorar os 4 testes. Os três do
  oráculo devem passar contra `A2_GENERIC_ESR_LIMIT = 1e-9`; o golden do C++ precisa de
  investigação separada (C++ usa path genérico Eigen, não cascade — ver §13.1 RF1-style).

### PM-21 — Referências quebradas a `TODO-findings.md`/`TODO-sprints.md` (arquivos deletados) [RESOLVIDO]

* **ID:** PM-21 · **Severidade:** Baixa (documentação) · **Risco:** Nulo
* **Problema:** O commit `44a5827` deletou `TODO-findings.md` (545 linhas) e
  `TODO-sprints.md` (451 linhas) sem migrar o conteúdo. Doze arquivos (`docs/*.md`,
  `tests/*.rs`, `src/*.rs`, `benches/*.rs`) continham referências cruzadas quebradas
  (ex.: `docs/cpp_parity_map.md` → `PM-09`/`PM-13`/`PM-14`; `docs/architecture.md` →
  `TODO-findings.md#C2`; `src/math/dsp/fft_radix4.rs:45` → `Tarefa A9`).
* **Proposta de Solução (aplicada):** Restaurar ambos os arquivos do histórico git
  (`git show 44a5827^:`) e anexar a Rodada 4 (este documento). As referências cruzadas
  voltam a resolver. Nenhuma reescrita de conteúdo necessária — os findings/sprints
  históricos permanecem válidos como registro de auditoria.

---

## Épicos (Agrupamentos) — Rodada 4

### Épico L — Correção da Regressão A2 Max (PM-16, PM-20) [ALTO VALOR, BLOQUEANTE]

* **Risco/Criticidade:** Alto (correção de hot-path). **Sequência:** imediata (bloqueia
  Feature-Completeness A2 do Épico G/K).
* **Entregáveis:**
  1. Fix da contagem de leitura `head1x1` (PM-16) em `build.rs`.
  2. Des-ignorar + validar 4 testes `S14.2-followup` (PM-20).
  3. Teste de regressão dedicado `test_a2_max_vs_oracle` (gate `A2_GENERIC_ESR_LIMIT`).

### Épico M — Limpeza pós-Bug-Storm (PM-17, PM-18, PM-19, PM-21) [DONE/QUICK-WINS]

* PM-17 (println! debug) — **RESOLVIDO** nesta auditoria.
* PM-18 (asserção obsoleta) — **RESOLVIDO** nesta auditoria.
* PM-19 (propagação reset) — pendente (baixa prioridade, latente).
* PM-21 (refs quebradas) — **RESOLVIDO** (restauração dos arquivos).

### PM-22 — Testes `#[should_panic]` dependem de `debug_assert!` (falham em release) [PRÉ-EXISTENTE]

* **ID:** PM-22 · **Severidade:** Baixa (test-infra, pré-existente) · **Risco da correção:** Baixo
* **Problema:** ~39 testes `#[should_panic]` em `src/math/dsp/fft_test.rs` (11),
  `src/models/a2/grouped_conv1d_test.rs` (16), `src/models/slimmable_test.rs` (6), e outros
  esperam um pânico gerado por `debug_assert!`. Em builds `--release`, `debug_assert!` é
  compilado fora (no-op), então os testes **não entram em pânico e falham** ("should panic ...
  FAILED"). `cargo test --release` (sem `--no-fail-fast`) reporta 10 falhas.
* **Mitigação atual:** `tests-quick.sh` roda a bateria `--lib` em **debug** (onde
  `debug_assert!` está ativo), então o CI não captura — apenas rodadas manuais `--release`.
* **Nota de auditoria:** **pré-existente** (FFT desde `9f4adfc`, grouped_conv1d desde S13),
  não introduzido pela tempestade de bugs S14.2. Mapeado aqui para visibilidade.
* **Proposta de Solução (opcional):** (a) gate `#[cfg(debug_assertions)]` nos testes
  `should_panic` que dependem de `debug_assert!`, ou (b) trocar os `debug_assert!` de
  validação de bounds de I/O por `assert!` permanente (mais seguro para catching de bugs em
  produção, custo desprezível no hot-path). Preferir (b) para FFT/conv bounds de input.

---

> **Resumo da Rodada 4:** A tempestade de bugs S14.2 deixou o motor A2 Max **carregável mas
> incorreto** (PM-16, ESR 93 dB), com a barreira de testes neutralizada por 4 ignores
> (PM-20) e 1 teste unitário com asserção obsoleta falhando ativamente (PM-18). Um defeito de
> RT-safety (`println!` no hot-path, PM-17) e 12 referências quebradas (PM-21) foram
> **resolvidos** nesta auditoria. A **root cause** (PM-16) está caracterizada e pronta para
> correção — a divergência é entre produção e o oráculo f64 independente (que é validado
> contra âncora NumPy a -152.8 dB), deixando zero ambiguidade sobre qual caminho está errado.
