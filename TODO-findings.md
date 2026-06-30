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

### PM-04 — **ConvNet** sem validação externa: implementar oráculo f64 independente

* **ID:** PM-04 · **Severidade:** Média (arquitetura oficial NAMCore sem testemunha) · **Risco da correção:** Baixo (test-only, aditivo)
* **Problema:** O motor ConvNet (`src/models/convnet/`) está **completo, despachado e
  unit-testado**, com teste *self-golden* de determinismo (`tests/golden_vectors.rs`
  `test_golden_vectors_convnet_test`, threshold em `validation.rs:660-663`), porém **sem
  testemunha externa**: o oráculo f64 retorna zeros para ConvNet
  (`src/testing/reference_oracle.rs:282`, ramo `_ => vec![0.0; ...]`) e o `render` do C++
  **não consegue** gerar golden porque o ConvNet do **NAMCore v0.5.3** é arquiteturalmente
  incompatível com o **NAM 0.5.4** do Rust:

  | Aspecto      | C++ v0.5.3                          | NAM-rs 0.5.4                    |
  | ------------ | ----------------------------------- | ------------------------------- |
  | Canais       | único, compartilhado                | por-bloco                       |
  | Kernel       | fixo `=2`                           | por-bloco                       |
  | Head         | matriz × vetor, **sem** Conv1D/ativ.| `PostStackHead` Conv1D + ativ.  |
  | `head_scale` | ausente                             | presente                        |

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

### Épico E — Robustez da Suíte de Testes (PM-07) [DOING]

* **Risco/Criticidade:** Baixo (test-only). **Sequência:** independente; pode ir junto com o Épico A.
* Fecha a brecha latente de SKIP silencioso no harness live v2, em conformidade com a Regra 7.
* **Pedido do PO:** Aproveite esta ocasião para acionar a skill "revisor-auditor" focada na role "Correctness Auditor".
  * Muito especificamente na suíte de testes ativadas por utils/tests-quick.sh, utils/tests-long.sh, utils/build-release.sh e utils/tests-performance-regression.sh.
  * Assegurar que tanto aqueles scripts em si, quanto os testes que eles acionam - todos corretos e prontos para serem graduados como os perfeitos "guardiães da qualidade" do nam-rs.
