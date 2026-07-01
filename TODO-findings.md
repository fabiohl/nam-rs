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
