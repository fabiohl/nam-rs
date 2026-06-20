<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints.md — Épico E: Cadeia de Suprimentos Reprodutível

> **Skill:** `planejador-arquiteto`
> **Data:** 2026-06-20
> **Referência canônica:** `TODO-audit.md` §ÉPICO E (linhas 499-513)
> **Achados cobertos:** F8, F9, F10, F15
> **Criticidade geral:** 🟠 Média — mexe em scripts de release/PGO e geração de goldens.
> Testar com cautela; lembrar que **IA não executa `tests-long.sh`** nem `build-release.sh` —
> validação pesada exige desenvolvedor humano.
>
> **Dependência:** fornece a infra que o Épico B consome (golden_gen_build extensível,
> benches dinâmicos/ConvNet).

---

## Visão Geral dos Sprints

| Sprint | Tema                                          | Achados     | Risco  | Tarefas |
| ------ | --------------------------------------------- | ----------- | ------ | ------- |
| E.1    | PGO e Benchmarks — cobertura dos hot-paths    | **F8**      | 🟠     | 4       |
| E.2    | Fresh-clone e Frescor dos Goldens             | **F9, F15** | 🟠     | 5       |
| E.3    | Manifesto Nondist Reprodutível                | **F10**     | 🟡     | 6       |

---

## Sprint E.1 — PGO e Benchmarks: Cobertura dos Hot-Paths (F8)

> **Objetivo:** Garantir que `build-release.sh` (PGO Phase 2) profila **todos** os
> hot-paths de produção — incluindo ConvNet e engines dinâmicos que hoje escapam dos
> filtros de bench.
>
> **DoD do Sprint:** `cargo bench --bench inference_bench -- "64samp"` captura ConvNet
> e ≥1 engine dinâmico; `build-release.sh` Phase 2 gera `.profraw` contendo esses paths;
> nenhum bench relevante é silenciosamente ignorado.
>
> **⚠️ Risco:** Alteração dos IDs de bench pode invalidar comparações históricas do
> Criterion. Commit separado com rename para preservar rastreabilidade.

---

### T-E1.1 — Padronizar sufixo de IDs de bench ConvNet (`_64f` → `_64samp`) [DONE]

> **Achado:** F8 — ConvNet usa sufixo `_64f`, não casado pelo filtro `"64samp"` do PGO.
> **Risco/Cautela:** 🟡 Baixo. Renomear IDs apaga o histórico de comparação Criterion
> para esses benches (esperado e aceitável).

**Contexto:**

- Os 3 benchmark groups de ConvNet em `benches/inference_bench.rs` usam o sufixo `_64f`:
  - `ConvNet_MultiChannel_64f` (linha ~1364)
  - `ConvNet_LargeKernel_64f` (linha ~1396)
  - `ConvNet_Dilated_64f` (linha ~1436)
- O filtro PGO em `utils/build-release.sh:185` é `"64samp"`.
- Resultado: ConvNet **não é profilado** pelo PGO.

**Ação (implementação):**

1. **`benches/inference_bench.rs`** — Renomear os 3 benchmark groups:
   - `ConvNet_MultiChannel_64f` → `ConvNet_MultiChannel_64samp`
   - `ConvNet_LargeKernel_64f` → `ConvNet_LargeKernel_64samp`
   - `ConvNet_Dilated_64f` → `ConvNet_Dilated_64samp`
   - Localização: funções `bench_convnet_multichannel` (~L1364), `bench_convnet_large_kernels`
     (~L1396), `bench_convnet_dilated` (~L1436).
   - Manter os nomes das funções de benchmark internas (`ConvNetBlock_inX_outY`, etc.) —
     só o **group name** precisa do sufixo `_64samp`.

2. **Documentação inline** — Adicionar comentário `// PGO: group name uses _64samp suffix
   to match build-release.sh profiling filter` nas 3 funções.

**Verificação:**

```bash
cargo bench --bench inference_bench -- --list 2>&1 | grep -i "convnet.*64samp"
# Esperado: 3 groups listados
```

**Critério de aceite:** `grep "64samp" <<< $(cargo bench --bench inference_bench -- --list)`
retorna linhas ConvNet.

---

### T-E1.2 — Adicionar benches de engine dinâmico representativo (WaveNetDyn + LstmDyn) [DONE]

> **Achado:** F8 + F4 — engines dinâmicos (`WaveNetModelDyn`, `LstmModelDyn`) não têm
> benchmark algum em `inference_bench.rs`, logo PGO nunca os profila.
> **Risco/Cautela:** 🟠 Médio. Os modelos dinâmicos são construídos sinteticamente (sem
> dependência de fixture `.nam` externo). Garantir que o dispatcher roteia para o engine
> dinâmico e não para um estático.

**Contexto:**

- `inference_bench.rs` já tem benches de WaveNet Dynamic e LSTM Dynamic nas linhas ~19-21
  da doc-table, mas esses benches **não existem de fato** no código (a tabela documenta
  intenção, não realidade).
- O bench `bench_nondist_models` (L1803-1855) depende de `models-nondist/` que pode não
  existir no build limpo — é complementar, não substituto.

**Ação (implementação):**

1. **`benches/inference_bench.rs`** — Adicionar 2 novas funções:

   a) `bench_wavenet_dynamic_64samp`:
      - Construir um `NamModelData` sintético de WaveNet com geometria **livre** (ex.:
        `condition_size=3`, ou CH ∉ {3,8,12,16} → ex.: CH=5) que force o dispatcher a
        rotear para `WaveNetModelDyn` (via `topology.rs` → `WaveNetTopology::Free`).
      - Modelo com 2 layer-arrays de 5 dilações cada `[1,2,4,8,16]`, channels=5,
        kernel_size=3, com head simple `[5,1]`.
      - Configurar pesos com valor constante `0.01` (como feito em `make_lstm_data`).
      - Prewarm 2048. Process 64 samples sine 440Hz.
      - Bench ID: `WaveNet_Dynamic_CH5_64samp_48kHz`

   b) `bench_lstm_dynamic_1x7_64samp`:
      - Construir `NamModelData` LSTM com `num_layers=1`, `hidden_size=7` (H=7 ∉
        {3,8,12,16,24,40} → dispatcher roteia para `LstmDyn`).
      - Prewarm 2048. Process 64 samples.
      - Bench ID: `LSTM_Dynamic_1x7_64samp_48kHz`

2. **Registrar** ambos no `criterion_group!(benches, ...)` (antes de `bench_nondist_models`).

3. **Validar roteamento**: cada bench deve assertar (via `#[cfg(debug_assertions)]`) que o
   `build_model` retorna a variante dinâmica esperada. Em release, confiar no dispatcher.

**Helper necessário:** Criar `make_wavenet_dyn_data(channels: usize, dilations: &[usize],
kernel_size: usize, condition_size: usize) -> NamModelData` para construir o modelo
sintético. Estudar os testes existentes de WaveNet dinâmico em
`tests/fixtures/models/wavenet_condition_dsp.nam` (CH=3, cond=3) e o builder em
`tests/common/model_builders.rs` para a estrutura JSON correta.

**Verificação:**

```bash
cargo bench --bench inference_bench -- "Dynamic" --list
# Esperado: 2 benches listados (WaveNet_Dynamic_CH5_64samp, LSTM_Dynamic_1x7_64samp)

cargo bench --bench inference_bench -- "Dynamic" --profile-time 1
# Esperado: executa sem panic/skip
```

**Critério de aceite:** Os 2 benches rodam com sucesso e são capturados por `"64samp"`.

---

### T-E1.3 — Atualizar filtros PGO em `build-release.sh` (incluir dinâmicos)

> **Achado:** F8 — o script `build-release.sh` Phase 2 filtra apenas `"64samp"`, `"AVX"`,
> `"Resampler"`. Após T-E1.1 e T-E1.2, os novos benches já serão capturados por `"64samp"`.
> Esta tarefa documenta e valida.
> **Risco/Cautela:** 🟡 Baixo.

**Contexto:**

- `utils/build-release.sh:185-187`:

  ```bash
  cargo bench --features clap-plugin --bench inference_bench -- --profile-time 1 "64samp"
  cargo bench --features clap-plugin --bench inference_bench -- --profile-time 1 "AVX"
  cargo bench --features clap-plugin --bench inference_bench -- --profile-time 1 "Resampler"
  ```

- Com T-E1.1 e T-E1.2 implementados, `"64samp"` já captura ConvNet e engines dinâmicos.

**Ação (implementação):**

1. **`utils/build-release.sh`** — Adicionar comentário documentando a cobertura atualizada
   nas linhas 180-183 (bloco de comentário existente):

   ```bash
   #   - ConvNet models (ConvNet_MultiChannel/LargeKernel/Dilated_64samp) — Added by Épico E
   #   - Dynamic engines (WaveNet_Dynamic_CH5_64samp, LSTM_Dynamic_1x7_64samp) — Added by Épico E
   ```

2. **Sem alteração de filtros** — `"64samp"` já cobre tudo. Não adicionar filtros extras
   para evitar inflar `--profile-time`.

3. **Tratar `bench_nondist_models` como graceful-skip:** Verificar que
   `bench_nondist_models` (que depende de `models-nondist/`, possivelmente ausente em
   build limpo) **não** causa falha ruidosa no PGO. O bench já faz `if !path.exists() { return; }`
   (L1807) — OK, mas documentar: se `models-nondist` ausente, os benches NonDist
   simplesmente não geram profdata (skip explícito, sem erro). Adicionar comentário no
   script:

   ```bash
   # Note: bench_nondist_models auto-skips when models-nondist/ is absent (no error).
   ```

**Verificação:**

- O desenvolvedor humano deve rodar `utils/build-release.sh` e verificar no output que as
  linhas de ConvNet e Dynamic aparecem na Phase 2.

**Critério de aceite:** `build-release.sh` Phase 2 completa com `.profraw` contendo code
regions de ConvNet e engines dinâmicos. Documentação do script atualizada.

---

### T-E1.4 — Atualizar tabela de benches na doc-header de `inference_bench.rs`

> **Risco/Cautela:** 🟡 Trivial — documentação inline.

**Ação (implementação):**

1. **`benches/inference_bench.rs`** — Atualizar a tabela `## Available benchmarks` (linhas
   12-21) para incluir:
   - `WaveNet_Dynamic_CH5_64samp_48kHz` — "WaveNet Dynamic free-geom inference" —
     "Fallback for non-cataloged WaveNet geometries"
   - `LSTM_Dynamic_1x7_64samp_48kHz` — "LSTM Dynamic 1×7 inference" —
     "Fallback for non-cataloged LSTM geometries"
   - Atualizar nomes de ConvNet se referenciados.

2. Remover a menção de `WaveNet_Dynamic_Standard_64samp_48kHz` e
   `LSTM_Dynamic_1x16_64samp_48kHz` da tabela (esses nunca existiram; substituídos pelos
   novos IDs reais).

**Critério de aceite:** A tabela de doc-header reflete 1:1 os benches realmente presentes.

---

## Sprint E.2 — Fresh-Clone e Frescor dos Goldens (F9, F15)

> **Objetivo:** Um `git clone` + `utils/mod-update.sh` + `utils/tests-long.sh` deve
> completar a suíte inteira **sem switches manuais** (`NAM_AUTO_BUILD_GOLDENS`). Goldens
> commitados devem ter mecanismo de verificação de frescor contra `.nam` de origem.
>
> **DoD do Sprint:** Em clone limpo com toolchain C++ + `mod-update.sh` executado,
> `tests-long.sh` Phase 0 resolve goldens ausentes automaticamente (sem `NAM_AUTO_BUILD_GOLDENS=1`);
> manifesto de frescor permite detectar goldens desatualizados.
>
> **⚠️ Risco:** Inversão de lógica de auto-build em `tests-long.sh` é mudança
> comportamental — testar em VM limpa. Mudança de `.gitignore` pode causar
> staging acidental de arquivos indesejados.

---

### T-E2.1 — Inverter lógica de auto-build de goldens (padrão = on-missing)

> **Achado:** F9 — Regeneração automática exige `NAM_AUTO_BUILD_GOLDENS=1`; o padrão
> deveria ser auto-build quando goldens faltam e toolchain C++ está presente.
> **Risco/Cautela:** 🟠 Médio. Mudança comportamental; pode causar rebuilds inesperados.
> Usar `NAM_SKIP_GOLDEN_BUILD=1` como opt-out.

**Contexto:**

- `utils/tests-long.sh:139-186` — Phase 0 verifica goldens e toolchain C++.
- Se faltam goldens, só regenera se `NAM_AUTO_BUILD_GOLDENS=1` (L139).
- Caso contrário, imprime instruções e `exit 1` (L180-185).

**Ação (implementação):**

1. **`utils/tests-long.sh`** — Substituir o bloco de decisão (L124-186) por:

   ```bash
   if [ ${#MISSING_GOLDENS[@]} -gt 0 ] || [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
       echo -e "${RED}${BOLD}❌ Pre-flight falhou — pré-requisitos ausentes:${NC}"
       # ... (manter listagem existente de MISSING_GOLDENS e MISSING_TOOLS)

       if [ "${NAM_SKIP_GOLDEN_BUILD:-0}" = "1" ]; then
           echo -e "${YELLOW}→ NAM_SKIP_GOLDEN_BUILD=1 — pulando regeneração automática.${NC}"
           exit 1
       fi

       # Auto-build é o padrão quando toolchain C++ e NeuralAmpModelerCore estão presentes
       if [ ${#MISSING_TOOLS[@]} -eq 0 ] && [ -d "tests/fixtures/NeuralAmpModelerCore" ]; then
           echo -e "\n${YELLOW}${BOLD}→ Regenerando goldens automaticamente (toolchain C++ + NeuralAmpModelerCore presentes)...${NC}"
           if ! bash tests/fixtures/golden_gen_build.sh; then
               echo -e "${RED}${BOLD}❌ golden_gen_build.sh falhou. Corrija as dependências e tente novamente.${NC}"
               exit 1
           fi
           echo -e "${GREEN}✓ Golden vectors regenerados com sucesso.${NC}"
           # Re-validate (manter bloco existente de re-validação L147-178)
       else
           echo -e "  ${YELLOW}→ Instale: cmake >= 3.10, g++/clang++ com C++20${NC}"
           echo -e "  ${YELLOW}→ Execute: ./utils/mod-update.sh${NC}"
           exit 1
       fi
   fi
   ```

2. **Remover** a referência a `NAM_AUTO_BUILD_GOLDENS` (L139, L184). Manter retrocompatibilidade
   documentando que a env var antiga é ignorada.

3. **Documentar** a nova variável `NAM_SKIP_GOLDEN_BUILD=1` no header do script e no
   `tests/fixtures/README.md` (se existir seção sobre reprodutibilidade).

**Verificação:**

- O desenvolvedor humano deve testar em clone limpo:
  1. `git clone ... && cd nam-rs && utils/mod-update.sh`
  2. `utils/tests-long.sh` → deve regenerar goldens faltantes automaticamente.
  3. `NAM_SKIP_GOLDEN_BUILD=1 utils/tests-long.sh` → deve falhar com mensagem clara.

**Critério de aceite:** Em clone limpo + `mod-update.sh`, `tests-long.sh` Phase 0 roda
`golden_gen_build.sh` automaticamente quando goldens faltam. Com `NAM_SKIP_GOLDEN_BUILD=1`,
aborta sem regenerar.

---

### T-E2.2 — Implementar manifesto de frescor `.nam ↔ golden`

> **Achado:** F9 — Não há verificação de que os goldens commitados são consistentes com os
> `.nam` atuais. Risco de golden stale após editar fixture.
> **Risco/Cautela:** 🟡 Baixo. Arquivo auxiliar gitignored; não bloqueia nada, apenas avisa.

**Contexto:**

- Os goldens `.bin` são gerados a partir dos modelos `.nam` em `tests/fixtures/models/`
  usando `golden_gen_build.sh`.
- Se um `.nam` for editado e o golden não for regenerado, a suíte pode reportar falsos
  positivos ou falsos negativos.

**Ação (implementação):**

1. **`tests/fixtures/golden_gen_build.sh`** — Ao final (após gerar todos os goldens),
   adicionar geração de um manifesto de frescor:

   ```bash
   # Generate freshness manifest (gitignored)
   MANIFEST="$FIXTURES_DIR/.golden_manifest.sha256"
   echo "# Golden freshness manifest — auto-generated by golden_gen_build.sh" > "$MANIFEST"
   echo "# Format: sha256(model.nam) sha256(golden.bin) model_filename golden_filename" >> "$MANIFEST"
   for entry in "${MODELS[@]}"; do
       IFS=':' read -r nam_file golden_name label <<< "$entry"
       MODEL_PATH="$MODELS_DIR/$nam_file"
       GOLDEN_PATH="$FIXTURES_DIR/${golden_name}.bin"
       if [ -f "$MODEL_PATH" ] && [ -f "$GOLDEN_PATH" ]; then
           MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
           GOLDEN_SHA=$(sha256sum "$GOLDEN_PATH" | cut -d' ' -f1)
           echo "$MODEL_SHA $GOLDEN_SHA $nam_file ${golden_name}.bin" >> "$MANIFEST"
       fi
   done
   echo "  Freshness manifest: $MANIFEST"
   ```

2. **`.gitignore`** — Adicionar:

   ```text
   /tests/fixtures/.golden_manifest.sha256
   ```

3. **`utils/tests-long.sh`** — Na Phase 0, **após** verificar presença dos goldens e
   **antes** de rodar os testes, adicionar checagem de frescor:

   ```bash
   # Optional freshness check (warn-only, not blocking)
   MANIFEST="tests/fixtures/.golden_manifest.sha256"
   if [ -f "$MANIFEST" ]; then
       STALE_COUNT=0
       while read -r expected_model_sha expected_golden_sha nam_file golden_file; do
           [[ "$expected_model_sha" == "#" ]] && continue
           MODEL_PATH="tests/fixtures/models/$nam_file"
           if [ -f "$MODEL_PATH" ]; then
               CURRENT_MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
               if [ "$CURRENT_MODEL_SHA" != "$expected_model_sha" ]; then
                   echo -e "  ${YELLOW}⚠ STALE: $nam_file changed since golden was generated${NC}"
                   STALE_COUNT=$((STALE_COUNT + 1))
               fi
           fi
       done < "$MANIFEST"
       if [ "$STALE_COUNT" -gt 0 ]; then
           echo -e "${YELLOW}⚠ $STALE_COUNT golden(s) may be stale. Consider re-running golden_gen_build.sh${NC}"
       else
           echo -e "${GREEN}✓ Golden freshness manifest OK (all models match).${NC}"
       fi
   else
       echo -e "  ${YELLOW}⚠ No freshness manifest found (.golden_manifest.sha256). Run golden_gen_build.sh to generate.${NC}"
   fi
   ```

**Verificação:**

```bash
# Após rodar golden_gen_build.sh:
cat tests/fixtures/.golden_manifest.sha256 | head -5
# Esperado: linhas com sha256 do model e do golden

# Alterar um .nam e rodar tests-long.sh → aviso de STALE
```

**Critério de aceite:** `golden_gen_build.sh` gera `.golden_manifest.sha256`; `tests-long.sh`
Phase 0 detecta e avisa (warn-only) quando `.nam` diverge do manifesto.

---

### T-E2.3 — Documentar pré-requisito único: `mod-update.sh`

> **Achado:** F9 — Documentação dispersa sobre pré-requisitos de clone novo.
> **Risco/Cautela:** 🟡 Trivial — documentação.

**Ação (implementação):**

1. **`README.md`** — Na seção de "Getting Started" ou "Development Setup", adicionar/atualizar:

   ```markdown
   ## Fresh Clone Setup

   After cloning, a single command prepares all external dependencies:
   ```sh
   ./utils/mod-update.sh
   ```

   This clones NeuralAmpModelerCore (v0.5.3) and NeuralAmpModelerPlugin with submodules.

   ### Running the Full Test Suite

   ```sh
   # Standard tests (< 2 min):
   ./utils/tests-cargo.sh

   # Long-duration audit (± 38 min, requires cmake + g++/clang++):
   ./utils/tests-long.sh
   ```

   `tests-long.sh` **automatically regenerates** golden vectors if they are missing
   (requires C++ toolchain). Set `NAM_SKIP_GOLDEN_BUILD=1` to disable auto-regeneration.

   ```markdown

2. **`utils/tests-long.sh`** — Atualizar header de documentação (linhas 5-8) para refletir

   a nova lógica de auto-build.

3. **`tests/fixtures/README.md`** — Se existir seção de reprodutibilidade, atualizar
   menção a `NAM_AUTO_BUILD_GOLDENS` → `NAM_SKIP_GOLDEN_BUILD`.

**Critério de aceite:** Um novo desenvolvedor, seguindo apenas o README, consegue rodar
a suíte completa em clone limpo.

---

### T-E2.4 — Escopar `.gitignore` para `*.json` (F12)

> **Achado:** F12 — `.gitignore` ignora `*.json` globalmente sem un-ignore para fixtures.
> **Risco/Cautela:** 🟡 Baixo, mas requer cuidado para não stagear configs indesejados.

**Contexto:**

- `.gitignore:16` → `*.json` ignora **todos** os JSON do repo.
- `tests/fixtures/models/keras_unsupported.json` está versionado por `git add -f` histórico.
- Risco: novo fixture `.json` silenciosamente ignorado → clone novo quebra.

**Ação (implementação):**

1. **`.gitignore`** — Substituir a linha `*.json` por regras escopadas:

   ```gitignore
   # JSON — ignorar na raiz e em diretórios não-fixtures
   /*.json
   /src/**/*.json
   /utils/**/*.json

   # Permitir fixtures JSON explicitamente
   !tests/fixtures/**/*.json
   ```

   **Alternativa mais simples** (menos invasiva):

   ```gitignore
   *.json
   !tests/fixtures/**/*.json
   ```

   Escolher esta alternativa: mantém o ignore global mas adiciona un-ignore para fixtures.

2. **Verificar** que `keras_unsupported.json` continua rastreado:

   ```bash
   git check-ignore tests/fixtures/models/keras_unsupported.json
   # Esperado: nenhuma saída (não ignorado)
   ```

3. **Verificar** que JSONs indesejados (ex.: `Cargo.json` se existisse) continuam ignorados:

   ```bash
   git check-ignore random.json
   # Esperado: random.json (ignorado)
   ```

**Critério de aceite:** `git check-ignore tests/fixtures/models/keras_unsupported.json`
retorna vazio (não ignorado); `git check-ignore foo.json` retorna `foo.json` (ignorado);
nenhum JSON indesejado adicionado ao staging.

---

### T-E2.5 — Decisão e documentação sobre Git LFS para goldens (F15)

> **Achado:** F15 — Goldens commitados pesam ~105 MB. Trade-off consciente.
> **Risco/Cautela:** 🟡 Decisão de arquitetura, não código.

**Contexto:**

- `tests/fixtures/*.bin` ≈ 105 MB (52 arquivos, v2 multi-SR ~1.9 MB × dezenas).
- Manter commitado **favorece** F9 (clone → suíte completa offline).
- Git LFS reduziria o clone inicial, mas adicionaria dependência de LFS e servidor.

**Ação (implementação):**

1. **Decisão:** O implementador deve **confirmar com o stakeholder** a preferência:
   - **(a) Manter commitado** (status quo) — favorece reprodutibilidade offline; aceitar
     o custo de ~105 MB no clone. Documentar como decisão consciente.
   - **(b) Git LFS** — migrar `tests/fixtures/*.bin` para LFS; requer `git lfs install`
     como pré-requisito.
   - **(c) Subset + auto-regeneração** — commitar apenas goldens v1 48kHz (~15 MB);
     v2 multi-SR regenerados on-demand via `golden_gen_build.sh` (alinhado com T-E2.1).

2. **Independente da decisão,** documentar em `tests/fixtures/README.md`:

   ```markdown
   ## Golden Vector Storage (~105 MB)

   Golden vectors are committed directly to the repository to enable fully offline
   test execution after a fresh clone. This is a deliberate trade-off:
   - **Pro:** `git clone` + `mod-update.sh` + `tests-long.sh` works without network.
   - **Con:** Initial clone is ~105 MB larger.

   If repository size becomes a concern, consider migrating to Git LFS or committing
   only v1 goldens (~15 MB) with v2 auto-regeneration (see T-E2.1 in TODO-sprints.md).
   ```

**Critério de aceite:** Decisão documentada em `tests/fixtures/README.md`. Se Git LFS
escolhido, `.gitattributes` configurado e migração executada.

---

## Sprint E.3 — Manifesto Nondist Reprodutível (F10)

> **Objetivo:** O mecanismo `models-nondist` se torna reprodutível e auditável: manifesto
> machine-readable, discovery consolidada, asserção de classificação quando manifesto presente.
>
> **DoD do Sprint:** `nondist_validation.rs` consome manifesto quando disponível e valida
> classe esperada; helper gera manifesto sem ANSI; `find_models_in_dir` existe em fonte única.
>
> **⚠️ Risco:** Baixo (🟡). O diretório `models-nondist` é gitignored; nenhuma mudança
> afeta a suíte em clone limpo. O manifesto é complementar (auto-skip quando ausente).

---

### T-E3.1 — Consolidar `find_models_in_dir` em `tests/common/`

> **Achado:** F10 — A função de discovery existia em 2 locais. Após verificação, atualmente
> ela existe apenas em `nondist_validation.rs`. Mover para `tests/common/` como fonte única
> para uso futuro.
> **Risco/Cautela:** 🟡 Refatoração trivial.

**Contexto:**

- `tests/nondist_validation.rs:14-30` — `find_models_in_dir` definida inline.
- `benches/inference_bench.rs:1812-1823` — lógica similar (não recursiva) inline no bench.
- Ambas deveriam referenciar uma fonte única.

**Ação (implementação):**

1. **`tests/common/mod.rs`** — Adicionar `pub mod discovery;` (ou nome adequado).

2. **`tests/common/discovery.rs`** (NOVO) — Mover a função:

   ```rust
   // SPDX-License-Identifier: Apache-2.0
   // Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

   use std::fs;
   use std::path::{Path, PathBuf};

   /// Finds all `.nam` and `.json` model files in the given directory recursively.
   pub fn find_models_in_dir(dir: &Path) -> Vec<PathBuf> {
       let mut files = Vec::new();
       if let Ok(entries) = fs::read_dir(dir) {
           for entry in entries.flatten() {
               let path = entry.path();
               if path.is_dir() {
                   files.extend(find_models_in_dir(&path));
               } else if path.extension().is_some_and(|ext| ext == "nam" || ext == "json") {
                   files.push(path);
               }
           }
       }
       files.sort(); // Deterministic order
       files
   }
   ```

3. **`tests/nondist_validation.rs`** — Remover a definição inline e usar
   `common::discovery::find_models_in_dir` (via `use common::*;` ou import explícito).

4. **`benches/inference_bench.rs`** — Não pode usar `tests/common/` diretamente (bench ≠
   test). Manter lógica inline existente, mas adicionar comentário:
   `// NOTE: mirrored from tests/common/discovery.rs — keep in sync`

**Verificação:**

```bash
cargo test --test nondist_validation 2>&1 | tail -5
# Esperado: SKIP ou pass (dependendo da presença de models-nondist)
```

**Critério de aceite:** `find_models_in_dir` tem fonte canônica em `tests/common/discovery.rs`;
`nondist_validation.rs` importa; compilação OK.

---

### T-E3.2 — Criar helper de geração de manifesto nondist (sem ANSI)

> **Achado:** F10 — `CATALOG.txt` gerado por `check-model.py` contém escapes ANSI e não é
> machine-readable.
> **Risco/Cautela:** 🟡 Baixo. Novo helper, sem alterar fluxos existentes.

**Contexto:**

- `utils/check-model.py` classifica modelos mas o output vai para terminal com cores ANSI.
- O manifesto desejado: JSON machine-readable, gitignored, com `{filename, sha256,
  expected_class}`.

**Ação (implementação):**

1. **`utils/check-model.py`** — Adicionar flag `--manifest`:

   ```python
   # No argparse/sys.argv handling:
   if "--manifest" in sys.argv:
       sys.argv.remove("--manifest")
       manifest_mode = True
   else:
       manifest_mode = False
   ```

   No modo manifesto, ao final de `main()`:

   ```python
   if manifest_mode:
       import hashlib
       manifest_entries = []
       for filepath in sys.argv[1:]:
           if not os.path.exists(filepath):
               continue
           with open(filepath, 'r') as f:
               data = json.load(f)
           info = classify_model(data, filepath)
           sha256_hash = hashlib.sha256(open(filepath, 'rb').read()).hexdigest()
           manifest_entries.append({
               "filename": os.path.basename(filepath),
               "sha256": sha256_hash,
               "expected_class": info["arch"],
               "is_goal_target": info["is_goal"],
               "name": info["name"],
               "author": info["author"]
           })
       print(json.dumps(manifest_entries, indent=2))
   ```

2. **Documentar** uso:

   ```bash
   # Gerar manifesto para todos os modelos nondist:
   python3 utils/check-model.py --manifest tests/fixtures/models-nondist/*.nam \
       > tests/fixtures/models-nondist/manifest.json
   ```

3. **`.gitignore`** — Já ignora `tests/fixtures/models-nondist` inteiro (L59). O manifesto
   dentro do diretório já está coberto. OK, nenhuma mudança necessária.

**Verificação:**

```bash
python3 utils/check-model.py --manifest tests/fixtures/models/*.nam | python3 -m json.tool
# Esperado: JSON válido com array de objetos
```

**Critério de aceite:** `check-model.py --manifest` gera JSON sem ANSI; formato machine-readable.

---

### T-E3.3 — `nondist_validation.rs` assertar contra manifesto quando presente

> **Achado:** F10 — Sem asserção de classificação, o teste não captura roteamento incorreto.
> **Risco/Cautela:** 🟠 Médio. Requer cuidado para não quebrar o teste quando manifesto
> ausente (manter auto-skip).

**Contexto:**

- `tests/nondist_validation.rs` valida determinismo, block-invariance, finitude e
  estabilidade — mas **não** valida que o dispatcher roteou para a classe esperada.

**Ação (implementação):**

1. **`tests/nondist_validation.rs`** — Após construir o modelo (L79-85), se manifesto
   presente, assertar a classe:

   ```rust
   // Load manifest if present
   let manifest_path = nondist_path.join("manifest.json");
   let manifest: Option<Vec<ManifestEntry>> = if manifest_path.exists() {
       let manifest_json = fs::read_to_string(&manifest_path)
           .expect("Failed to read manifest.json");
       Some(serde_json::from_str(&manifest_json)
           .expect("Failed to parse manifest.json"))
   } else {
       None
   };
   ```

   Dentro do loop de modelos:

   ```rust
   // If manifest is present, validate expected class
   if let Some(ref manifest) = manifest {
       if let Some(entry) = manifest.iter().find(|e| e.filename == filename) {
           let actual_class = format!("{:?}", model.variant_name());
           // Log but don't fail hard — the manifest is advisory
           if !actual_class.contains(&entry.expected_class_prefix()) {
               eprintln!(
                   "⚠ Classification mismatch for {}: expected '{}', got '{}'",
                   filename, entry.expected_class, actual_class
               );
           } else {
               println!("  ✓ Classification OK: {} → {}", filename, actual_class);
           }
       }
   }
   ```

2. **Struct `ManifestEntry`** — Definir em `tests/common/` (ou inline):

   ```rust
   #[derive(serde::Deserialize)]
   struct ManifestEntry {
       filename: String,
       sha256: String,
       expected_class: String,
       is_goal_target: bool,
       name: String,
       author: String,
   }
   ```

3. **Dependência:** Adicionar `serde_json` e `serde` como dev-dependencies em `Cargo.toml`
   (provavelmente já presentes; verificar).

4. **Comportamento quando manifesto ausente:** Nenhuma asserção de classificação — mantém
   o comportamento atual. Apenas log: `"NOTE: No manifest.json found — skipping
   classification validation."`.

**Verificação:**

```bash
# Com manifesto:
python3 utils/check-model.py --manifest tests/fixtures/models-nondist/*.nam \
    > tests/fixtures/models-nondist/manifest.json
cargo test --test nondist_validation -- --nocapture
# Esperado: linhas "✓ Classification OK" ou "⚠ Classification mismatch"

# Sem manifesto:
rm -f tests/fixtures/models-nondist/manifest.json
cargo test --test nondist_validation -- --nocapture
# Esperado: "NOTE: No manifest.json found" + testes passam normalmente
```

**Critério de aceite:** Com manifesto presente, classificações são validadas e reportadas;
sem manifesto, comportamento inalterado.

---

### T-E3.4 — Documentar comando único de (re)geração do manifesto nondist

> **Risco/Cautela:** 🟡 Trivial — documentação.

**Ação (implementação):**

1. **`README.md`** ou **`tests/fixtures/README.md`** — Adicionar seção:

   ```markdown
   ## Non-Distributable Models (models-nondist)

   Private models for extended validation. The directory is gitignored; place a symlink
   or copy at `tests/fixtures/models-nondist/`.

   ### Generating the Manifest

   ```sh
   python3 utils/check-model.py --manifest tests/fixtures/models-nondist/*.nam \
       > tests/fixtures/models-nondist/manifest.json
   ```

   The manifest enables `nondist_validation.rs` to validate expected model classification
   (e.g., "this CH=32 model should route to WaveNetDyn"). Without the manifest, only
   determinism, block-invariance, and stability are checked.

   ```markdown

**Critério de aceite:** Documentação presente com comando reprodutível.

---

### T-E3.5 — Limpar escapes ANSI do output padrão de `check-model.py`

> **Achado:** F10 — Output com escapes ANSI embutidos no CATALOG.txt.
> **Risco/Cautela:** 🟡 Quick-win.

**Ação (implementação):**

1. **`utils/check-model.py`** — Detectar se stdout é terminal e suprimir cores quando não:

   ```python
   import sys
   USE_COLOR = sys.stdout.isatty()

   def color(code, text):
       if USE_COLOR:
           return f"\033[{code}m{text}\033[0m"
       return text
   ```

   Substituir todas as referências a `\033[92m`, `\033[91m`, `\033[94m`, `\033[0m` por
   chamadas a `color(...)`.

2. **Resultado:** Redirecionar para arquivo (`> CATALOG.txt`) produz output limpo sem ANSI.

**Verificação:**

```bash
python3 utils/check-model.py tests/fixtures/models/BossWN-standard.nam | cat -v
# Esperado: sem ^[[92m ou similares
```

**Critério de aceite:** `check-model.py` output para arquivo não contém escapes ANSI.

---

### T-E3.6 — Verificação final integrada (DoD do Épico E)

> **Risco/Cautela:** 🟠 Esta tarefa é inteiramente de validação humana.

**Ação (verificação):**

O desenvolvedor humano deve executar, em uma VM ou clone limpo:

1. **Clone limpo:**

   ```bash
   git clone <repo> nam-rs-test && cd nam-rs-test
   ./utils/mod-update.sh
   ```

2. **Suíte rápida (smoke test):**

   ```bash
   ./utils/tests-cargo.sh
   ```

3. **Suíte longa (sem switches):**

   ```bash
   ./utils/tests-long.sh
   # Esperado: Phase 0 regenera goldens automaticamente; todas as fases completam.
   ```

4. **PGO build:**

   ```bash
   ./utils/build-release.sh
   # Esperado: Phase 2 mostra linhas de ConvNet_*_64samp e *_Dynamic_*_64samp
   ```

5. **Nondist (se disponível):**

   ```bash
   ln -s /path/to/nondist tests/fixtures/models-nondist
   python3 utils/check-model.py --manifest tests/fixtures/models-nondist/*.nam \
       > tests/fixtures/models-nondist/manifest.json
   cargo test --test nondist_validation -- --nocapture
   ```

**Critério de aceite do Épico E (DoD):**

- [ ] Em VM/clone limpo + `mod-update.sh`, `tests-long.sh` completa todas as fases sem switches.
- [ ] PGO/perf-annotate/`build-release.sh` inclui ConvNet e ≥1 engine dinâmico.
- [ ] `models-nondist` documentado num único comando; manifesto machine-readable.
- [ ] `.gitignore` un-ignora `tests/fixtures/**/*.json` (F12).
- [ ] Freshness manifest detecta goldens desatualizados (warn-only).
- [ ] `cargo check`, `cargo test`, `cargo bench` passam sem warnings.

---

## Matriz de Rastreabilidade (Tarefas → Achados)

| Tarefa   | Achado(s)     | Sprint | Prioridade |
| -------- | ------------- | ------ | ---------- |
| T-E1.1   | F8            | E.1    | 🟡         |
| T-E1.2   | F8, F4        | E.1    | 🟠         |
| T-E1.3   | F8            | E.1    | 🟡         |
| T-E1.4   | F8            | E.1    | 🟡         |
| T-E2.1   | F9            | E.2    | 🟠         |
| T-E2.2   | F9            | E.2    | 🟡         |
| T-E2.3   | F9            | E.2    | 🟡         |
| T-E2.4   | F12           | E.2    | 🟡         |
| T-E2.5   | F15           | E.2    | 🟡         |
| T-E3.1   | F10           | E.3    | 🟡         |
| T-E3.2   | F10           | E.3    | 🟡         |
| T-E3.3   | F10           | E.3    | 🟠         |
| T-E3.4   | F10           | E.3    | 🟡         |
| T-E3.5   | F10           | E.3    | 🟡         |
| T-E3.6   | *todos*       | E.3    | 🟠         |

---

## Ordem de Execução Recomendada

```text
Sprint E.1 (benches/PGO)
  └── T-E1.1 → T-E1.2 → T-E1.4 → T-E1.3 (T-E1.3 valida após E1.1+E1.2)

Sprint E.2 (fresh-clone) — pode iniciar em paralelo com E.1
  └── T-E2.4 (gitignore, quick-win) → T-E2.1 → T-E2.2 → T-E2.3 → T-E2.5 (decisão)

Sprint E.3 (nondist) — após E.1 e E.2 concluídos
  └── T-E3.1 → T-E3.5 → T-E3.2 → T-E3.3 → T-E3.4 → T-E3.6 (validação final)
```

> **Nota:** Sprints E.1 e E.2 podem rodar em paralelo entre si. Sprint E.3 depende
> conceptualmente de E.1/E.2 mas pode iniciar T-E3.1 e T-E3.5 em paralelo com eles.
> T-E3.6 (validação integrada) deve ser a última tarefa de todo o Épico.
