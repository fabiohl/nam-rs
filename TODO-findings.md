<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria: `tests/fixtures/golden_gen_build.sh` (Supply Chain de Goldens)

> Gerado pela skill `revisor-auditor`, role **Compliance and Parity Auditor** (com incursões pontuais em
> **Resilience and Robustness Specialist** onde a robustez do script compromete diretamente a
> confiabilidade da cadeia de parity/compliance — "hardenizar com carinho" foi interpretado como parte
> do mandato de compliance: um gerador de goldens que falha silenciosamente, ou que não consegue
> reproduzir os goldens que os testes exigem, é uma falha de compliance, não apenas um bug cosmético).
>
> **Escopo:** `tests/fixtures/golden_gen_build.sh` como fornecedor (*supply chain*) de
> `utils/tests-quick.sh` (Fase 2) e `utils/tests-long.sh`, à luz de `docs/testing.md`,
> `docs/cpp_parity_map.md`, `docs/audio_fidelity_map.md` e `tests/fixtures/README.md`.
>
> **Método:** leitura integral do script (480 linhas), leitura cruzada com `tests/golden_vectors.rs`,
> `tests/linear_fft_test.rs`, `tests/threshold_calibration.rs`, `utils/tests-long.sh`,
> `utils/mod-update.sh`, `tests/fixtures/generate_a2_fixtures.py`, `tests/fixtures/render_ir.cpp`,
> `.gitignore`, e inventário físico de `tests/fixtures/models/`, `tests/fixtures/models-nondist/` e
> `tests/fixtures/*.bin`. Todas as citações abaixo foram verificadas contra o conteúdo real dos
> arquivos no momento da auditoria (2026-07-02).
>
> **Tese central da auditoria:** o pipeline de goldens não tem uma **única fonte de verdade**. O
> catálogo de "quais modelos têm golden, com quais SRs, e como reproduzi-los" está fragmentado em pelo
> menos **quatro lugares independentes** que precisam ser mantidos manualmente em sincronia:
> `golden_gen_build.sh` (arrays `MODELS`/`V2_MODELS`), `utils/tests-long.sh` (arrays
> `REQUIRED_GOLDEN_MODELS`/`V2_*_MODELS`), `tests/golden_vectors.rs` (funções de teste individuais) e
> `tests/fixtures/README.md` (tabelas de catálogo). A maioria dos achados abaixo (F3, F3b, F4, F11,
> F12) é uma *consequência direta* dessa fragmentação — não são bugs isolados, são sintomas do mesmo
> problema estrutural. Isso deve nortear a priorização: consertar os arrays isoladamente resolve o
> sintoma de hoje, mas não impede o próximo drift.

---

## Sumário Executivo

| ID  | Severidade | Categoria               | Título                                                                             |
| --- | ---------- | ----------------------- | ---------------------------------------------------------------------------------- |
| F1  | 🔴 Crítico | Robustez                | `set -e` + `pipefail` torna o tratamento de erro do loop v1 código morto           |
| F2  | 🔴 Crítico | Compliance/Supply-chain | `mod-update.sh` nunca sincroniza `NeuralAmpModelerPlugin`; script mente ao usuário |
| F3  | 🟠 Alto    | Compliance/Parity       | 4 goldens A2 dinâmicos/FiLM usados em teste não são gerados pelo script            |
| F3b | 🟠 Alto    | Compliance/Parity       | 4 goldens Linear FFT documentados no `testing.md` como cobertos nunca existiram    |
| F4  | 🟠 Alto    | Manutenibilidade        | Duplicação estrutural `MODELS[]` vs `V2_MODELS[]` (causa raiz de F3/F3b)           |
| F5  | 🟡 Médio   | Consistência            | `golden_cabsim_cpp_stress.bin` é artefato órfão, nunca gerado por `render_ir.cpp`  |
| F6  | 🟡 Médio   | Robustez                | Cache de binários (`render`, `render_ir`) nunca invalidado por mudança local       |
| F7  | 🟡 Médio   | Observabilidade         | Numeração de fases inconsistente e duplicada no output do script                   |
| F8  | 🟡 Médio   | Observabilidade         | Truncamento de log (`tail -N`) esconde a causa raiz exatamente na falha            |
| F9  | 🟢 Baixo   | Robustez                | Skip de ConvNet acoplado a *string matching* de label, não a identidade de modelo  |
| F10 | 🟢 Baixo   | Documentação (inline)   | Comentário contraditório/stale sobre `wavenet_lite` (linhas 309-310)               |
| F11 | 🟡 Médio   | Compliance              | Manifesto de frescor é parcial, não-bloqueante, e nunca roda em CI                 |
| F12 | 🟢 Baixo   | Documentação            | `tests/fixtures/README.md` cita números de linha de `.gitignore` desatualizados    |
| F13 | 🟡 Médio   | Documentação            | `docs/testing.md` não documenta o pipeline de geração como dependência crítica     |

---

## F1 — 🔴 CRÍTICO: `set -e` + `pipefail` torna o tratamento de erro do loop v1 código morto

**Evidência:**

- `golden_gen_build.sh:54` — `set -euo pipefail` ativado globalmente (ambos `errexit` e `pipefail`).

- `golden_gen_build.sh:273` — loop v1, chamada de render:

  ```bash
  "$RENDER_BIN" "$MODEL_PATH" "$STRESS_WAV" "$OUTPUT_WAV" 2>&1 | tail -1
  ```

- `golden_gen_build.sh:275-278` — tratamento de erro aparentemente tolerante:

  ```bash
  if [ ! -f "$OUTPUT_WAV" ]; then
      echo "  ERROR: Render failed for $label"
      continue
  fi
  ```

- Contraste: `golden_gen_build.sh:370` — o loop v2 (multi-SR), escrito *depois*, corrige exatamente este
  problema ao desabilitar `pipefail` localmente em subshell:

  ```bash
  (set +o pipefail; "$RENDER_BIN" "$MODEL_PATH" "$v2_wav" "$v2_out_wav" 2>&1 | tail -1)
  ```

**Diagnóstico:** com `pipefail` ativo, o status de saída de um pipeline é o do **último comando do
pipeline a falhar**, não necessariamente o do último comando posicional. Se `$RENDER_BIN` retornar
código diferente de zero (falha de render — exatamente o caso que o `if [ ! -f "$OUTPUT_WAV" ]`
downstream foi escrito para tratar), o pipeline `"$RENDER_BIN" ... | tail -1` propaga esse código de
saída não-zero para o shell. Com `errexit` (`set -e`) também ativo, o script **aborta imediatamente**
no primeiro modelo cujo render falhar — a mensagem amigável `"ERROR: Render failed for $label"` e o
`continue` para o próximo modelo **nunca são alcançados** nesse cenário, que é justamente o cenário
mais comum de falha (render tool retorna erro, não "roda com sucesso mas não escreve o arquivo").

Isso contradiz o comportamento pretendido e documentado implicitamente pelo próprio código: o padrão
`continue` em um loop `for entry in "${MODELS[@]}"` só faz sentido se a intenção for "pular modelos
problemáticos e seguir processando o resto do catálogo" — o autor já implementou essa tolerância
corretamente no loop v2 (linha 370), mas não retroalimentou a correção para o loop v1 (linha 273), que
é o primeiro a rodar e portanto o primeiro a poder travar toda a regeneração por causa de **um único
modelo problemático em ~20**.

**Impacto real:** dado que este script roda raramente ("apenas quando um novo modelo de referência ou
arquitetura é adicionado"), o cenário de "um modelo novo não renderiza corretamente na primeira
tentativa" é exatamente o caso de uso mais provável — e é precisamente esse caso que o bug transforma
de "aviso claro + continua com os demais 19 modelos" em "crash abrupto do bash, sem contexto, parando
toda a regeneração". Isso é uma falha de robustez que ataca diretamente a única característica que
torna esse gerador seguro de operar sem supervisão constante.

**Proposta de solução:**

1. Aplicar o mesmo padrão já usado no loop v2 (linha 370) ao loop v1 (linha 273): envolver a chamada de
   render em subshell com `set +o pipefail` local, ou — de forma mais robusta e sem subshell — capturar
   o código de saída explicitamente:

   ```bash
   set +o pipefail
   "$RENDER_BIN" "$MODEL_PATH" "$STRESS_WAV" "$OUTPUT_WAV" 2>&1 | tail -1
   render_status=$?
   set -o pipefail
   if [ "$render_status" -ne 0 ] || [ ! -f "$OUTPUT_WAV" ]; then
       echo "  ERROR: Render failed for $label (exit=$render_status)"
       continue
   fi
   ```

2. Aplicar a mesma auditoria a **toda** chamada de pipeline dentro de um loop com `continue` no
   restante do script — confirmar que não há outro caso análogo (a auditoria já verificou as chamadas
   de `wav_to_golden`, que não usam pipe e portanto não sofrem deste problema; a chamada de `render_ir_bin`
   também não está em loop, então abortar ali é o comportamento correto).

3. Adicionar um teste de fumaça (shell, não Rust) que injete deliberadamente um `.nam` corrompido/inválido
   na lista de modelos (ou mocke `$RENDER_BIN` para retornar erro) e verifique que o script **continua**
   processando os modelos subsequentes e termina com status de saída apropriado (a decidir: 0 com
   avisos, ou não-zero mas tendo processado o que pôde — atualmente a única garantia é "aborta cedo").

---

## F2 — 🔴 CRÍTICO: `mod-update.sh` nunca sincroniza `NeuralAmpModelerPlugin`; o script direciona o usuário para um beco sem saída

**Evidência:**

- `golden_gen_build.sh:100-103`:

  ```bash
  if [ ! -d "$NAM_PLUGIN_DIR" ]; then
      echo "ERROR: NeuralAmpModelerPlugin not found at $NAM_PLUGIN_DIR."
      echo "Please run './utils/mod-update.sh' to download and setup dependencies."
      exit 1
  fi
  ```

- `golden_gen_build.sh:107-111` (mesmo padrão para mismatch de versão) e `:114-118` (mesmo padrão para
  submódulos ausentes) — **três** pontos de saída diferentes, todos recomendando `mod-update.sh`.

- `utils/mod-update.sh:49-75` (bloco `[4/4]`) — sincroniza **exclusivamente**
  `tests/fixtures/NeuralAmpModelerCore` (clone/fetch/checkout no SHA pinado + submódulos `eigen` e
  `AudioDSPTools`). **Não há nenhuma menção a `NeuralAmpModelerPlugin` em todo o arquivo** (79 linhas,
  confirmado por leitura integral).

- `README.md:122` (raiz do projeto) afirma incorretamente: *"This clones NeuralAmpModelerCore ... and
  NeuralAmpModelerPlugin with submodules"* — o mesmo engano se propagou para a documentação principal
  do projeto.

**Diagnóstico:** `golden_gen_build.sh` depende de **dois** repositórios C++ vendored — `Core` (motor de
render, usado para todos os goldens WaveNet/LSTM/A2) e `Plugin` (fonte do `dsp::ImpulseResponse`, usado
apenas na etapa `[6/6]` para os goldens de cabsim). O script verifica a presença/versão de ambos com o
mesmo rigor (guard de commit pinado), mas só **um** deles (`Core`) tem um caminho de setup automatizado.
Um usuário seguindo exatamente as instruções de erro do próprio script entra em um loop: `mod-update.sh`
roda, reporta sucesso, mas o `Plugin` continua ausente/desatualizado, e o próximo `golden_gen_build.sh`
falha na mesma verificação, com a mesma instrução inútil.

**Impacto:** este é o pior tipo de falha de UX/compliance em uma ferramenta de infraestrutura de
testes — não apenas quebra, mas **mente sobre como consertar**. Qualquer contribuidor novo (humano ou
IA) tentando gerar goldens do zero (cenário coberto pelo próprio `tests/fixtures/README.md`, seção
"To regenerate all fixtures from scratch") vai ficar travado sem diagnóstico claro, precisando
descobrir manualmente (via leitura do próprio `golden_gen_build.sh`) o SHA pinado
(`96337e9ab6e3beb619459779bbb5c47e1b04d8c4`) e rodar o `git clone`/`checkout`/`submodule update` na
mão — exatamente o trabalho que `mod-update.sh` existe para automatizar.

**Proposta de solução:**

1. Estender `utils/mod-update.sh` com um passo `[5/5]` (ou reestruturar para `[4/5]`+`[5/5]`) que
   sincronize `NeuralAmpModelerPlugin` de forma simétrica ao que já é feito para `Core`: mesmo padrão de
   clone/fetch/checkout no SHA pinado + inicialização de submódulos (`AudioDSPTools`, que por sua vez
   tem `Dependencies/eigen` e `Dependencies/nlohmann`).
2. Extrair as constantes de commit pinado (`NAM_CORE_COMMIT`, `NAM_PLUGIN_COMMIT`, tags, repos) para um único
   arquivo de configuração compartilhado (`/variables.env`, com `source` feito
   tanto por `golden_gen_build.sh` quanto por `mod-update.sh`), eliminando a duplicação atual onde o SHA
   do `Core` está hardcoded em **dois** arquivos (`golden_gen_build.sh:66` e `mod-update.sh:55`) que já
   precisam ficar sincronizados manualmente (o comentário em `golden_gen_build.sh:11-13` já reconhece
   esse acoplamento frágil: *"a mismatch causes this script's version-mismatch guard... to hard-fail"*).
   **DECISÃO (2026-07-02):** Centralizado em `/variables.env` (raiz do projeto) com a seção "Pinned
   vendor versions" — contém também os URLs de clone (`NAM_CORE_REPO`, `NAM_PLUGIN_REPO`) para que
   `mod-update.sh` os consuma como variáveis em vez de strings mágicas.
3. Corrigir `README.md:122` para não afirmar que `mod-update.sh` cobre `NeuralAmpModelerPlugin` até que
   o item 1 seja implementado (ou simultâneo a ele).

---

## F3 — 🟠 ALTO: 4 goldens A2 dinâmicos/FiLM usados ativamente em teste não são gerados por `golden_gen_build.sh`

**Evidência:**

- `tests/golden_vectors.rs` carrega e testa ativamente:
  - `golden_a2_dynamic_gated_ch8.bin` ← `a2_dynamic_gated_ch8.nam` (teste
    `test_golden_vectors_a2_dynamic_gated_ch8`, linha 1513)
  - `golden_a2_dynamic_blended_ch3.bin` ← `a2_dynamic_blended_ch3.nam` (teste
    `test_golden_vectors_a2_dynamic_blended_ch3`, linha 1570)
  - `golden_wavenet_a2_film_lite.bin` ← `wavenet_a2_film_lite.nam` (teste
    `test_golden_vectors_wavenet_a2_film_lite`, linha 1623)
  - `golden_wavenet_a2_film_full.bin` ← `wavenet_a2_film_full.nam` (teste
    `test_golden_vectors_wavenet_a2_film_full`, linha 1681)
- Nenhuma dessas 4 combinações `.nam`↔`golden` aparece em `MODELS[]` (`golden_gen_build.sh:223-244`) nem
  em `V2_MODELS[]` (`golden_gen_build.sh:311-331`).
- Os 4 arquivos `.nam` correspondentes **são gerados** por `tests/fixtures/generate_a2_fixtures.py`
  (linhas 392, 415, 427-428) — mas esse script **não invoca o C++ render tool** e **não produz `.bin`**;
  ele só escreve os modelos.
- `tests/fixtures/README.md:551-598` já documenta manualmente a receita completa (gerar `.nam` com
  Python → `render` manual → `wav_to_golden` manual) para os goldens A2 sintéticos e FiLM — ou seja, o
  time já sabia que esse caminho existe fora do script principal, mas nunca o integrou.
- Ironicamente, as mensagens de erro dentro de `tests/golden_vectors.rs:1630,1688` (quando os goldens
  FiLM estão ausentes) instruem o usuário a rodar **`./tests/fixtures/golden_gen_build.sh`** — uma
  instrução que **não resolve o problema**, pois o script não cobre esses modelos.

**Diagnóstico:** o cabeçalho do próprio `golden_gen_build.sh` (linha 6) descreve a si mesmo como o
gerador de "**all** golden vectors". Isso deixou de ser verdade assim que os goldens FiLM/dinâmicos
foram introduzidos (via script Python + receita manual documentada só no README), sem que o script
principal fosse atualizado para incorporá-los. O resultado é um pipeline de regeneração com **duas
rotas paralelas e não sincronizadas**: a rota automatizada (script) e a rota manual (README), cada uma
cobrindo um subconjunto do catálogo, com mensagens de erro em `.rs` que apontam para a rota errada.

**Impacto:** se qualquer um dos 4 goldens FiLM/dinâmicos for perdido, corrompido, ou precisar de
re-baseline após atualização do C++ pinado, **não existe caminho de regeneração de um único comando**.
O README documenta os comandos manuais individuais (`render`, `wav_to_golden`), mas exige que o
operador monte manualmente a sequência completa — exatamente o tipo de procedimento propenso a erro
humano que `golden_gen_build.sh` existe para eliminar para os outros ~20 modelos.

**Proposta de solução:**

1. Adicionar uma nova seção `[5c/5]` (ou renumerar por completo — ver F7) em `golden_gen_build.sh` que:
   a. Invoque `tests/fixtures/generate_a2_fixtures.py` (com verificação de `python3` disponível,
      seguindo o padrão de verificação de pré-requisitos já usado para `cmake`/`cargo` em
      `golden_gen_build.sh:74-79`) para (re)gerar os 7 `.nam` sintéticos que esse script produz
      (`wavenet_a2_full.nam`, `wavenet_a2_lite.nam`, `wavenet_a2_container.nam`,
      `a2_dynamic_gated_ch8.nam`, `a2_dynamic_blended_ch3.nam`, `wavenet_a2_film_lite.nam`,
      `wavenet_a2_film_full.nam`) — isso também fecha uma lacuna atual em que os `.nam` sintéticos são
      versionados diretamente no git sem receita de regeneração determinística executada
      automaticamente.
   b. Adicione 4 novas entradas a `MODELS[]` (e a `V2_MODELS[]` apenas se v2 multi-SR fizer sentido —
      ver nota de `tests/fixtures/README.md:225` sobre goldens dinâmicos serem intencionalmente só v1):
      `a2_dynamic_gated_ch8.nam:golden_a2_dynamic_gated_ch8:...`,
      `a2_dynamic_blended_ch3.nam:golden_a2_dynamic_blended_ch3:...`,
      `wavenet_a2_film_lite.nam:golden_wavenet_a2_film_lite:...`,
      `wavenet_a2_film_full.nam:golden_wavenet_a2_film_full:...`.
2. Corrigir as mensagens de erro em `tests/golden_vectors.rs:1630,1688` (e qualquer outra ocorrência
   análoga) para não afirmarem que `golden_gen_build.sh` resolve o problema até que o item 1 seja feito;
   ou, preferencialmente, implementar o item 1 primeiro e então a mensagem de erro existente já fica
   correta "de graça".
3. Ver F4 para a causa estrutural (duplicação de arrays) que tornou esse drift fácil de acontecer e
   difícil de detectar.

---

## F3b — 🟠 ALTO: goldens Linear FFT documentados como cobertos em `docs/testing.md` nunca existiram

**Evidência:**

- `tests/linear_fft_test.rs:21-32` documenta 4 modelos Linear FFT (`linear_fft_rf320.nam`,
  `linear_fft_rf2048.nam`, `linear_fft_rf4096.nam`, `linear_fft_rf8192.nam`) com uma tabela de
  RF/canais/blocksize, e afirma na linha 23 que os goldens correspondentes são gerados por
  `./tests/fixtures/golden_gen_build.sh`.
- Os 4 arquivos `.nam` **existem** em `tests/fixtures/models/` (confirmado por listagem de diretório:
  `linear_fft_rf320.nam` 4135 B, `linear_fft_rf2048.nam` 25529 B, `linear_fft_rf4096.nam` 50892 B,
  `linear_fft_rf8192.nam` 101643 B).
- Nenhum arquivo `golden_linear_fft_*.bin` existe em `tests/fixtures/` (confirmado — `ls
  tests/fixtures/golden_linear*` não retorna nada).
- `golden_gen_build.sh` **não contém nenhuma menção** a `linear_fft` em nenhum dos seus arrays ou
  seções (confirmado por grep no arquivo inteiro).
- `tests/fixtures/README.md` **também não contém nenhuma menção** a `linear_fft` — os 4 modelos não
  aparecem em nenhuma das tabelas de catálogo/proveniência do README, apesar da seção "Model Files and
  Trust Levels Registry" (README:88) afirmar textualmente: *"There are no orphan, redundant, or
  garbage files in the directory."* — afirmação que este achado contradiz.
- Os 4 testes correspondentes (`test_golden_vectors_linear_fft_rf2048/4096/8192`,
  `tests/linear_fft_test.rs:552-576`) estão `#[ignore]` com a mensagem *"Requires pre-generated
  golden_linear_fft_rf*.bin from C++ render tool"* — uma pré-condição que, dado o exposto acima, **nunca
  pode ser satisfeita** pelo pipeline documentado.
- `docs/testing.md:155` afirma que a suíte longa (Fase 3, item 3) executa *"full C++ parity and golden
  validation: ... `tests/linear_fft_test.rs` C++ goldens"* — **esta afirmação é falsa** hoje: esses
  testes nunca rodam (permanecem `#[ignore]` e "skipam" graciosamente para sempre) porque o golden nunca
  existiu e não há caminho de geração.

**Diagnóstico:** este é o mesmo padrão estrutural do F3 (modelo `.nam` existe, teste espera golden
C++, `golden_gen_build.sh` não cobre), mas em um estágio pior: para o F3, ao menos existe um caminho
manual documentado (`generate_a2_fixtures.py` + receita no README). Para Linear FFT, **não existe
nenhum caminho documentado, nenhum golden jamais gerado, e a documentação de arquitetura de testes
(`docs/testing.md`) afirma incorretamente que essa cobertura já está ativa na suíte longa** — um risco
de compliance mais grave que uma simples lacuna, porque cria falsa confiança de cobertura onde não há
nenhuma.

**Impacto:** 4 modelos `.nam` (≈180 KB no total) são fixtures mortas — nunca exercitadas por nenhum
teste que de fato executa. `docs/testing.md` (o documento de referência da arquitetura de testes,
mencionado explicitamente pelo usuário como fonte de contexto essencial para esta auditoria) contém uma
afirmação de cobertura que não corresponde à realidade — isso é precisamente o tipo de "placebo de
teste" que o princípio "Todo Golden Deve Poder Falhar" (`tests/fixtures/README.md:396-418`) foi
desenhado para prevenir, só que aqui o placebo está um nível acima: não é um golden com threshold
neutralizado, é uma *afirmação de documentação* sobre um golden que nunca existiu.

**Proposta de solução:**

1. **Curto prazo (aplicado nesta sessão de docs):** corrigir `docs/testing.md:155` para não afirmar que
   os goldens de `tests/linear_fft_test.rs` rodam na suíte longa — documentar seu estado real
   (`#[ignore]`, golden nunca gerado, pipeline de geração inexistente).
2. **Médio prazo:** decidir explicitamente uma de duas rotas e documentá-la:
   a. **Implementar:** adicionar as 4 entradas Linear FFT a `golden_gen_build.sh` (mesmo padrão de
      `MODELS[]`), já que o render tool C++ do NAMCore lida com modelos Linear normalmente (não há a
      mesma incompatibilidade arquitetural do ConvNet, linha 245-246); isso destrava os 4 testes.
   b. **Descontinuar:** se os 4 modelos Linear FFT foram substituídos por outra estratégia de validação
      (ex.: o oráculo `reference_oracle_f64` ou testes de FFT determinísticos que não dependem de C++),
      remover os 4 `.nam` e os 4 testes `#[ignore]` mortos, documentando a decisão em
      `tests/fixtures/README.md` (seção "Model Provenance") para não reabrir a mesma confusão no
      futuro.
3. Adicionar um meta-teste (na linha do já existente `tests/threshold_calibration.rs`'s
   `test_all_golden_models_have_calibrated_thresholds`) que varra `tests/fixtures/models/*.nam` e
   verifique que todo modelo referenciado por um teste `#[ignore]`-com-mensagem-de-golden-ausente tem
   uma entrada correspondente em `golden_gen_build.sh` (ou uma exceção documentada, como o caso
   ConvNet) — transformando esta classe de drift em um erro de CI detectável, em vez de uma auditoria
   manual eventual.

---

## F4 — 🟠 ALTO: duplicação estrutural `MODELS[]` vs `V2_MODELS[]` — causa raiz de F3

**Evidência:**

- `golden_gen_build.sh:223-244` (`MODELS[]`, 20 entradas) e `golden_gen_build.sh:311-331`
  (`V2_MODELS[]`, 20 entradas) são **quase idênticos** — mesma lista de `.nam`, mesmo `golden_name`,
  labels quase idênticos (ex.: `"WaveNet Standard"` vs. `"WaveNet Standard (CH=16)"`), diferindo apenas
  pelo campo opcional `skip_srs` usado só pelo v2.
- Comentário em `golden_gen_build.sh:298-300` já reconhece o acoplamento manual necessário: *"skip_srs
  ... kept in sync with the test SR sets in tests/golden_vectors.rs"* — ou seja, **três** listas
  (MODELS, V2_MODELS, e os conjuntos de SR em `golden_vectors.rs`) precisam ser mantidas coerentes à
  mão.

**Diagnóstico:** toda vez que um modelo novo precisa de golden (o caso de uso primário deste script,
por design), o mantenedor precisa lembrar de adicioná-lo em **dois** arrays quase idênticos. O F3
demonstra que isso já falhou na prática: os 4 modelos A2 dinâmicos/FiLM simplesmente não foram
adicionados a nenhum dos dois arrays. Duplicar não é grátis — é o campo perfeito para *drift* silencioso,
porque nada além de revisão manual detecta a omissão (o script não falha, os testes correspondentes só
ficam eternamente `#[ignore]` com uma mensagem de erro que parece "normal" até alguém investigar a
fundo, como esta auditoria fez).

**Proposta de solução:**

1. Consolidar em um **único array canônico** com um campo de escopo explícito, por exemplo:

   ```bash
   # nam_file : golden_name : label : v2_scope : skip_srs
   #   v2_scope ∈ {all, 48k_only, none}  — "none" = não gera v2 (ex.: goldens dinâmicos)
   CATALOG=(
       "BossWN-standard.nam:golden_wavenet_standard:WaveNet Standard (CH=16):48k_only:"
       "BossWN-feather.nam:golden_wavenet_feather:WaveNet Feather (CH=8):all:"
       "BossLSTM-1x16.nam:golden_lstm_1x16:LSTM 1×16:all:192000"
       ...
   )
   ```

   O loop v1 itera `CATALOG[@]` incondicionalmente (gera sempre); o loop v2 itera o mesmo array,
   pulando quando `v2_scope == none`, e aplicando `skip_srs`/`48k_only` conforme o campo.

2. Isso reduz para **uma única linha por modelo** o custo de onboarding de um novo `.nam`, e remove a
   possibilidade estrutural do bug do F3 (não é possível "esquecer" o v2 se o v1 e o v2 leem da mesma
   fonte por padrão).

3. Complementar com o meta-teste proposto no item 3 do F3b, que passa a validar contra este array único.

---

## F5 — 🟡 MÉDIO: `golden_cabsim_cpp_stress.bin` é artefato órfão

**Evidência:**

- `tests/fixtures/render_ir.cpp:12` (comentário de cabeçalho): *"Scenarios: short (seed=42), medium
  (seed=137), long (seed=31337), stress (seed=999983)"* — promete 4 cenários.
- `tests/fixtures/render_ir.cpp` `main()` (linhas ~150-155) só chama `run_scenario` **3 vezes** (short,
  medium, long) — o cenário `stress` nunca é implementado.
- `golden_gen_build.sh:47-49` (cabeçalho) e `:434-436` (sumário final) também só listam
  `golden_cabsim_cpp_{short,medium,long}.bin` — nenhuma menção a `stress`.
- Apesar disso, `golden_cabsim_cpp_stress.bin` **existe fisicamente** em `tests/fixtures/` (524292
  bytes) e é **documentado** em `tests/fixtures/README.md:69` como "Cabsim Stress IR (65536 samples)
  C++ dsp::ImpulseResponse".
- `tests/cabsim_cpp_parity.rs` (conforme apurado) documenta explicitamente por que essa validação C++ é
  impossível: o `dsp::ImpulseResponse` do C++ tem um hard-cap de 8192 amostras (`mMaxLength`), e o
  cenário stress usa uma IR de 32768 amostras — o motor C++ truncaria a IR, tornando a comparação
  cross-reference sem sentido.

**Diagnóstico:** o arquivo binário em disco é, na melhor hipótese, um artefato órfão de uma geração
anterior/manual/experimental que nunca foi removido, e a documentação do README o descreve como se
fosse uma referência C++ válida e atual — o que ele **não é e não pode ser**, dada a limitação
arquitetural do C++ documentada em outro arquivo de teste. Isso é uma inconsistência de baixo risco
técnico (nada depende dele para passar/falhar) mas de risco de confiança real: um novo mantenedor lendo
o README acredita que esse arquivo é uma referência C++ válida e pode tentar usá-lo como tal.

**Proposta de solução:**

1. Confirmar (via `git log -- tests/fixtures/golden_cabsim_cpp_stress.bin`) a origem/data de comitagem
   do arquivo para decidir entre remover ou re-rotular.
2. Corrigir `tests/fixtures/README.md:69` para não classificá-lo como "C++ reference (synthetic IR)"
   — se o arquivo for mantido (ex.: por ser consumido por algum teste self-golden/ESR Rust-only não
   coberto nesta investigação), reclassificá-lo corretamente (ex.: "Rust self-golden / determinism
   reference — NOT cross-validated against C++ due to the 8192-sample `mMaxLength` cap"); se nenhum
   teste o consome, remover o arquivo do repositório.
3. Corrigir `render_ir.cpp:12` para não prometer um cenário `stress` que nunca será implementado — ou
   documentar explicitamente ali, ao lado do comentário, por que ele foi descartado (referenciando o
   `mMaxLength` cap), para que a ausência seja uma decisão documentada, não uma omissão silenciosa.

---

## F6 — 🟡 MÉDIO: cache de binários (`render`, `render_ir`) nunca invalidado por mudança local

**Evidência:**

- `golden_gen_build.sh:157-159`:

  ```bash
  if [ -f "$RENDER_BIN" ]; then
      echo "  Render binary already exists: $RENDER_BIN"
  else
      ... cmake configure + build ...
  ```

- `golden_gen_build.sh:393-395` — padrão idêntico para `$IR_BIN` (compilado a partir de
  `tests/fixtures/render_ir.cpp`, que é **código-fonte deste próprio repositório**, não vendored).

**Diagnóstico:** para `$RENDER_BIN` (compilado a partir do NAMCore vendored), a existência do guard de
versão pinada (linhas 132-137) mitiga parcialmente o risco — se o `Core` upstream mudar de commit, o
guard aborta antes de chegar à etapa de build, embora **não** force rebuild do binário já existente se
apenas `BUILD_TYPE` ou `CXX` mudarem entre execuções (o binário cacheado de uma execução em `Debug` com
`clang++`, por exemplo, seria silenciosamente reaproveitado numa execução subsequente pedindo
`Release`/`g++`).

Para `$IR_BIN`, o risco é mais sério: `render_ir.cpp` é um arquivo do próprio repositório `nam-rs`
(passível de correção, ex.: ao resolver o F5). Se um desenvolvedor editar `render_ir.cpp` para corrigir
um bug e simplesmente re-executar `golden_gen_build.sh`, o binário cacheado em
`tests/fixtures/render_ir` (que está fora do controle de versão, conforme `.gitignore:65`) será
silenciosamente reaproveitado, e a correção **não terá efeito algum** até que o desenvolvedor descubra
que precisa apagar manualmente o binário. Isso é uma classe clássica de "correção fantasma" — o script
reporta sucesso, os goldens são "regenerados", mas com o binário antigo.

**Proposta de solução:**

1. Para `$IR_BIN`: adicionar checagem de timestamp (`render_ir.cpp` mais recente que `$IR_BIN` ⇒
   forçar rebuild) — trivial com `[ "$FIXTURES_DIR/render_ir.cpp" -nt "$IR_BIN" ]` em bash.
2. Para `$RENDER_BIN`: registrar `BUILD_TYPE` e `CXX` usados na última build em um arquivo-marcador
   (ex.: `$BUILD_DIR/.build_config`) e invalidar o cache se os valores atuais diferirem, além do
   próprio timestamp check aplicado de forma simétrica caso o CMakeLists do NAMCore vendored seja
   tocado localmente (menos provável, mas simétrico em espírito).
3. Alternativa mais simples e "seguindo o princípio de menor surpresa": documentar explicitamente no
   cabeçalho do script (e no `tests/fixtures/README.md`) que ambos os binários são cacheados
   indefinidamente e que `rm -f tests/fixtures/render_ir build/namcore_render/*/render` é o comando de
   invalidação manual — isso não resolve o problema, mas ao menos o torna um comportamento documentado
   e não uma surpresa descoberta por tentativa e erro.

---

## F7 — 🟡 MÉDIO: numeração de fases inconsistente e duplicada no output do script

**Evidência (todas as ocorrências de marcadores de fase no script):**

| Linha | Marcador | Fase                                        |
| ----- | -------- | ------------------------------------------- |
| 99    | `[1/6]`  | Verificando NeuralAmpModelerPlugin          |
| 125   | `[1b/6]` | Verificando NeuralAmpModelerCore            |
| 153   | `[2/5]`  | Building render tool                        |
| 184   | `[3/5]`  | Building Rust tools                         |
| 201   | `[4/5]`  | Generating stress signals                   |
| 220   | `[5/5]`  | Running render (v1)                         |
| 292   | `[5b/5]` | Generating v2 multi-SR                      |
| 388   | `[6/6]`  | Building C++ IR reference                   |
| 424   | `[6/6]`  | **(duplicado)** Cleaning up temporary files |
| 457   | `[7/7]`  | Generating freshness manifest               |

**Diagnóstico:** o total de fases declarado muda de `/6` para `/5` e volta para `/6`/`/7` ao longo do
próprio script, e o marcador `[6/6]` é usado **duas vezes** para duas fases completamente diferentes
(build da referência C++ IR vs. limpeza de arquivos temporários). Não é um bug funcional, mas é
diretamente contrário ao espírito do pedido do usuário de "hardenizar com carinho" um script que roda
raramente e precisa ser fácil de auditar visualmente quando algo dá errado no meio da execução — hoje,
se o script travar em "`[6/6]`", não é possível saber, só pelo log, qual das duas fases `[6/6]`
realmente falhou sem reler o script fonte.

**Proposta de solução:** renumerar todos os marcadores de fase sequencialmente e sem ambiguidade,
refletindo o total real de fases (atualmente 9, contando a duplicata como duas fases distintas):
`[1/9]` Plugin, `[2/9]` Core, `[3/9]` build render, `[4/9]` build ferramentas Rust, `[5/9]` stress
signals, `[6/9]` render v1, `[7/9]` render v2, `[8/9]` build+run IR reference, `[9/9]` manifesto +
limpeza (ou separar limpeza como `[10/10]` se se preferir mantê-la distinta do manifesto). Se o F3/F3b
forem implementados (nova fase de fixtures A2 sintéticas), renumerar novamente nesse momento — o custo
de renumeração é baixo e vale a clareza operacional.

---

## F8 — 🟡 MÉDIO: truncamento de log (`tail -N`) esconde a causa raiz exatamente na falha

**Evidência:**

- `golden_gen_build.sh:167` — `cmake -S ... 2>&1 | tail -5`
- `golden_gen_build.sh:168` — `cmake --build ... 2>&1 | tail -5`
- `golden_gen_build.sh:186` — `cargo build --release --bin gen_stress --bin wav_to_golden 2>&1 | tail -3`
- `golden_gen_build.sh:273` — render v1: `... | tail -1`
- `golden_gen_build.sh:370` — render v2: `... | tail -1`

**Diagnóstico:** truncar a saída de `cmake`/`cargo build`/`render` para as últimas 1-5 linhas é razoável
no caminho de sucesso (reduzir ruído no console), mas **as mesmas linhas truncadas são tudo que o
operador vê quando o comando falha** — e falhas de `cmake configure` (dependência ausente, versão de
compilador incompatível) ou de `cargo build` (erro de tipo, feature ausente) tipicamente têm a mensagem
de erro relevante **no meio ou no início** da saída, não nas últimas linhas. Combinado com o F1 (abort
abrupto via `set -e`), o operador frequentemente vai ver apenas as últimas 1-5 linhas de um log que não
contém a causa real, sem nenhuma indicação de onde encontrar o log completo.

**Proposta de solução:**

1. Redirecionar a saída completa de cada subcomando relevante para um arquivo de log persistente sob
   `build/namcore_render/logs/<fase>.log` (criando o diretório se necessário), e **também** aplicar
   `tail -N` ao terminal para manter o console limpo no caminho feliz. Em caso de falha (código de
   saída não-zero, verificado explicitamente — não implicitamente via `set -e`), imprimir uma mensagem
   apontando para o caminho do log completo antes de abortar/`continue`.
2. Isso é sinérgico com a correção do F1: uma vez que o código de saída de cada subcomando passa a ser
   verificado explicitamente (em vez de deixado para o `set -e` implícito), o mesmo ponto de verificação
   é o lugar natural para decidir "mostrar log completo e abortar" vs. "mostrar log completo, avisar, e
   `continue`".

---

## F9 — 🟢 BAIXO: skip de ConvNet acoplado a *string matching* de label

**Evidência:**

- `golden_gen_build.sh:268-271` (loop v1) e `golden_gen_build.sh:345-348` (loop v2):

  ```bash
  if [[ "$label" == ConvNet* ]]; then
      echo "  SKIP: $label — C++ v0.5.4 ConvNet is architecturally incompatible (known)"
      continue
  fi
  ```

- A entrada correspondente em ambos os arrays (`golden_gen_build.sh:242` e `:329`) é
  `"convnet_test.nam:golden_convnet_test:ConvNet Test (CH=8→4, 2 blocks)"`.

**Diagnóstico:** o controle de fluxo (pular render para modelos ConvNet, arquiteturalmente
incompatíveis com o C++ v0.5.4 conforme documentado em `docs/cpp_parity_map.md` §6) está acoplado ao
**texto do label legível por humanos**, não ao nome do arquivo `.nam` ou a uma allowlist/denylist
dedicada. Um rename cosmético do label (ex.: alguém decide renomeá-lo para "Multi-Block Conv Test" por
clareza, sem saber que isso é usado como chave de controle de fluxo) faria o script **tentar renderizar
um modelo sabidamente incompatível**, na melhor hipótese falhando de forma confusa no C++ render tool,
na pior hipótese produzindo um golden incorreto silenciosamente aceito (a menos que o F1 já esteja
corrigido e o erro seja capturado — outro motivo para priorizar F1). Além disso, como ambos os arrays já
sabem de antemão que este modelo sempre será pulado, mantê-lo com verificação de existência de arquivo e
mensagem "Processing..." antes do skip (linha 266, antes do check de linha 268) é ruído cosmético
inofensivo, mas confuso ao ler o log.

**Proposta de solução:**

1. Introduzir uma allowlist/denylist explícita e comentada por identidade de arquivo, não de label:

   ```bash
   # Modelos sabidamente incompatíveis com o render C++ vendored — ver docs/cpp_parity_map.md §6.
   KNOWN_INCOMPATIBLE_MODELS=("convnet_test.nam")
   ```

   e no loop: `if [[ " ${KNOWN_INCOMPATIBLE_MODELS[*]} " == *" $nam_file "* ]]; then ... continue; fi`
   (ou usar um campo dedicado no array consolidado do F4, ex. `skip:known-incompatible`).

2. Mover o check de incompatibilidade para **antes** da linha `echo "  Processing $label ($nam_file)..."`
   para não imprimir uma mensagem de progresso para um modelo que já sabemos, estaticamente, que nunca
   será processado.

---

## F10 — 🟢 BAIXO: comentário contraditório/stale sobre `wavenet_lite`

**Evidência — `golden_gen_build.sh:309-310`:**

```bash
# `wavenet_lite` is intentionally absent (known-divergent, P1 — no v2 coverage).
# P1 RESOLVIDO (T1.2): wavenet_lite re-added with full v2 multi-SR coverage.
```

**Diagnóstico:** a primeira linha afirma categoricamente que `wavenet_lite` está "intencionalmente
ausente" da cobertura v2; a segunda linha, imediatamente abaixo, contradiz a primeira ao anunciar que
o modelo foi "re-adicionado com cobertura multi-SR completa" — e de fato, `EVH-5150-Lite.nam` está
presente em `V2_MODELS[]` (`golden_gen_build.sh:313`) sem nenhum `skip_srs`, confirmando que a segunda
linha reflete o estado real e a primeira é resíduo histórico que deveria ter sido removido, não
apenas anotado como superado.

**Proposta de solução:** remover a linha 309 (o comentário obsoleto) por completo, mantendo apenas a
informação de proveniência histórica relevante se necessário (ex.: em `TODO-sprints.md`/changelog, não
no código ativo), já que comentários "P1 RESOLVIDO" sem remoção da afirmação original são ruído que
custa tempo de leitura a cada futura auditoria (como demonstrado por esta mesma).

---

## F11 — 🟡 MÉDIO: manifesto de frescor é parcial, não-bloqueante, e nunca roda em CI

**Evidência:**

- `golden_gen_build.sh:459-473` gera `.golden_manifest.sha256` iterando **apenas** `MODELS[]` — não
  inclui `V2_MODELS`-specific entries (hoje idênticas, mas ver F4), não inclui os 4 goldens do F3, não
  inclui os goldens cabsim (`render_ir`-based, que não têm um `.nam` de origem para hash — esperado),
  e não inclui os (inexistentes) goldens Linear FFT do F3b.
- `.gitignore:39` — `.golden_manifest.sha256` é explicitamente **não versionado**
  (`/tests/fixtures/.golden_manifest.sha256`), então o manifesto só existe localmente, na máquina de
  quem rodou `golden_gen_build.sh` por último — não é um artefato de auditoria persistente/compartilhado.
- `utils/tests-long.sh:209-231` é o **único** consumidor: leitura *warn-only* (nunca bloqueia, nunca
  retorna código de saída não-zero por staleness) e roda apenas dentro do escopo "pré-release/nightly"
  do `tests-long.sh`, que segundo o próprio `docs/testing.md:17` **é proibido rodar durante atividades
  de IA** e não faz parte de nenhum pipeline de CI identificado nesta auditoria (nenhum
  `.github/workflows` referencia goldens/manifest).

**Diagnóstico:** o manifesto de frescor é uma boa ideia (detectar quando um `.nam` mudou mas o `.bin`
correspondente não foi regenerado) implementada com escopo parcial e sem qualquer poder de enforcement
real — é local, efêmero (não versionado), e mesmo quando presente e stale, apenas imprime um aviso
amarelo que pode ser facilmente ignorado. Combinado ao fato de cobrir só um subconjunto do catálogo
real de goldens (ver F3/F3b), seu valor de proteção efetivo hoje é baixo, apesar da aparência de
"camada de segurança" que sua existência sugere.

**Proposta de solução:**

1. Curto prazo: alinhar a geração do manifesto ao array consolidado proposto no F4 (cobertura completa
   do catálogo real, incluindo os itens do F3 uma vez implementados).
2. Médio prazo: avaliar versionar o manifesto (removê-lo do `.gitignore`) — isso o transformaria em um
   artefato auditável no histórico do git (permitindo `git blame`/`git log` para rastrear quando e por
   que um golden foi re-baseado), e permitiria a um hook de CI real (mesmo que leve, fora do escopo do
   `tests-long.sh` de 50 minutos) comparar o manifesto commitado contra os hashes atuais dos `.nam` em
   um passo rápido (`sha256sum -c`-like), tornando a detecção de staleness parte do `tests-quick.sh`
   (Fase 2, onde os goldens v1 já são consumidos) em vez de exclusiva do `tests-long.sh`.
3. Decidir explicitamente se staleness detectada deve ser bloqueante (falha dura) ou apenas informativa
   — hoje é sempre informativa por padrão; dado que o princípio "Todo Golden Deve Poder Falhar"
   (`tests/fixtures/README.md` §Principle) já estabelece que testes-placebo são inaceitáveis, um
   manifesto de frescor que nunca bloqueia nada é, por essa mesma régua, um controle fraco.

---

## F12 — 🟢 BAIXO: `tests/fixtures/README.md` cita números de linha de `.gitignore` desatualizados

**Evidência:**

- `tests/fixtures/README.md:24-26`: *"Entries in `.gitignore` (lines 54–55, 58) prevent accidental
  commits. The golden `.bin` output files are version-controlled (explicitly unignored at
  `.gitignore:36`)"*
- `.gitignore` real, atual (verificado por leitura integral, 79 linhas):
  - linha **32**: `!tests/fixtures/*.bin` (o unignore mencionado — README diz linha 36, um desvio de 4
    linhas)
  - linha **60**: `/tests/fixtures/NeuralAmpModelerCore/` (README diz linhas 54-55, um desvio de ~5-6
    linhas)
  - linha **61**: `/tests/fixtures/NeuralAmpModelerPlugin/` (README diz linha 58, um desvio de 3 linhas
    — e o README nem cita este diretório especificamente nesta nota, apesar de listá-lo na tabela de
    "Unversioned Local Mirrors" duas linhas abaixo, README:20)
  - linha **62**: `/tests/fixtures/models-nondist` — não mencionado nesta nota do README de forma
    alguma.

**Diagnóstico:** referências de número de linha para um arquivo que muda com frequência (`.gitignore`
cresce organicamente conforme novas categorias de artefato são adicionadas) são inerentemente fráteis —
este é exatamente o tipo de "documentação factual que descalibra silenciosamente" contra o qual o
princípio de qualidade documental do projeto já se posiciona (ver o próprio `tests/fixtures/README.md`
seção "Parity Thresholds", que já tem um `[!IMPORTANT]` alertando sobre uma tabela que "*drifted stale*"
no passado por motivo análogo).

**Proposta de solução:**

1. Substituir a citação de números de linha absolutos por citação de **conteúdo** (grep-friendly, à
   prova de reordenação): *"see the `# Golden vectors de validação numérica` and `# NeuralAmpModelerCore`
   sections of `.gitignore`"*.
2. Atualizado nesta sessão (ver seção de mudanças de documentação abaixo) para refletir o estado real
   verificado, incluindo a menção agora correta e completa de `NeuralAmpModelerPlugin` e
   `models-nondist` nas regras de `.gitignore` relevantes.

---

## F13 — 🟡 MÉDIO: `docs/testing.md` não documenta o pipeline de geração de goldens como dependência crítica de supply-chain

**Evidência:**

- `docs/testing.md` (336 linhas, lido integralmente) menciona `golden_vectors` como um dos "5 canonical
  oracles" (linha 60, tabela §3 linha 113, §7 linha 227, §8 linha 290) e cita `tests/fixtures/README.md`
  apenas indiretamente (nunca linka/referencia `golden_gen_build.sh` diretamente em nenhum lugar do
  arquivo).
- Não há nenhuma seção explicando que a Fase 2 (release, `golden_vectors` v1 + `isa_parity` v2) e a Fase
  3/long-suite (v2 multi-SR, `cpp_parity` full matrix) **dependem inteiramente** de artefatos binários
  pré-gerados por um script C++ externo executado raramente e fora do loop normal de CI/desenvolvimento
  — uma dependência de supply-chain que, dado o F1-F11 acima, tem gaps de robustez e cobertura reais.

**Diagnóstico:** `docs/testing.md` é, pelo próprio propósito declarado (linha 8: *"tracks and
categorizes the test suite ... ensures 100% regression coverage"*), o documento que deveria deixar
explícito que parte dessa "cobertura de 100%" depende de um pipeline de geração que (a) roda raramente,
(b) tem lacunas de cobertura conhecidas agora documentadas nesta auditoria (F3, F3b), e (c) tem uma
verificação de frescor não-bloqueante (F11). Sem essa seção, um leitor do documento de arquitetura de
testes não tem como saber, sem ler o script de 480 linhas e os testes `.rs` correspondentes, que
"`golden_vectors` (v1)" na tabela da Fase 2 secretamente depende de uma cadeia de 4+ arquivos
mantidos manualmente em sincronia.

**Proposta de solução:** adicionar a `docs/testing.md` uma nova subseção dentro de §2 ou uma nova §2.x
("Golden Vector Supply Chain") descrevendo: o papel de `golden_gen_build.sh` como fornecedor de goldens
pré-commitados consumidos pela Fase 2; a natureza "raramente executado, apenas para novo modelo/arquitetura"
do script; a existência de lacunas de cobertura conhecidas (referenciando `tests/fixtures/README.md`
para o catálogo atualizado); e um link cruzado explícito para a seção correspondente do
`tests/fixtures/README.md` em vez de duplicar o conteúdo. Esta seção foi adicionada nesta mesma sessão
— ver "Mudanças de Documentação Aplicadas" abaixo.

---

## Mudanças de Documentação Aplicadas Nesta Sessão

Conforme solicitado, `docs/testing.md` e `tests/fixtures/README.md` foram atualizados nesta mesma
sessão para refletir os achados acima que afetam diretamente a precisão factual desses dois documentos
(F3b, F12, F13, e as referências corretas ao gap do F2/F5 no README). As correções **de código**
propostas (F1, F4, F6, F7, F8, F9, F10, e a implementação de F3/F3b) permanecem como findings/epics
rastreados abaixo, para implementação em uma sessão dedicada (`implementador`), dado que envolvem
mudanças de comportamento do script, não apenas de documentação.

---

## Épicos (para planejamento futuro via `planejador-arquiteto` → `TODO-sprints.md`)

## Épico A — Hardening do Fluxo de Erro do Gerador de Goldens (🔴 Crítico, alto risco) [DONE]

Torna o script seguro para operação "raramente executada, sem supervisão constante", que é seu modo de
uso pretendido segundo o próprio usuário desta auditoria.

- **F1** — Corrigir `pipefail`/`set -e` no loop v1 para não matar o script no primeiro render que falhar.
- **F8** — Persistir logs completos por fase + apontar para eles na falha (sinérgico com F1).
- **F7** — Renumerar fases de forma consistente e não-duplicada (barato, mas só vale a pena fazer junto
  com F1/F8, já que a renumeração final depende de quantas fases existirem após F3).

**Risco/Critério de aceite:** simular uma falha de render controlada (mock/`.nam` inválido) e confirmar
que o script emite o diagnóstico correto, continua processando os demais modelos, e termina com um
resumo claro de sucessos/falhas — não com um crash de bash sem contexto.

## Épico B — Fechar o Gap de Supply-Chain do `NeuralAmpModelerPlugin` (🔴 Crítico) [DONE]

- **F2** — Estender `mod-update.sh` para sincronizar `NeuralAmpModelerPlugin` simetricamente ao `Core`.
- Sub-tarefa: extrair SHAs/tags pinados para uma fonte única compartilhada entre `mod-update.sh` e
  `golden_gen_build.sh` (elimina duplicação de constantes, reduz o próprio risco que o comentário em
  `golden_gen_build.sh:11-13` já reconhece).

**Risco/Critério de aceite:** em um clone limpo do repositório, `./utils/mod-update.sh` seguido de
`./tests/fixtures/golden_gen_build.sh` deve completar sem qualquer intervenção manual em repositórios
git vendored.

## Épico C — Consolidação do Catálogo Canônico de Modelos↔Goldens (🟠 Alto, fundação para C1/C2) [DOING]

Resolve a causa raiz estrutural por trás de F3, F3b, F4, F11.

- **F4** — Consolidar `MODELS[]`/`V2_MODELS[]` em um único array canônico com campo de escopo v2.
- **F9** — Migrar o skip de ConvNet de *string-matching* de label para identidade de arquivo, usando o
  mesmo array consolidado.
- **F11** (parte 1) — Gerar o manifesto de frescor a partir do array consolidado (cobertura completa).
- Meta-teste (proposto no F3b item 3): CI-enforced que todo `.nam` referenciado por um teste
  `#[ignore]`-por-golden-ausente tenha entrada no catálogo canônico (ou exceção documentada).

### Épico C1 — Fechar a Lacuna A2 Dinâmico/FiLM (depende de Épico C)

- **F3** — Adicionar as 4 entradas A2 dinâmicas/FiLM ao catálogo consolidado; integrar
  `generate_a2_fixtures.py` como fase automatizada de `golden_gen_build.sh`.

### Épico C2 — Resolver ou Descontinuar Linear FFT (depende de Épico C, decisão de produto necessária primeiro)

- **F3b** — Decidir implementar (adicionar ao catálogo) ou descontinuar (remover fixtures + testes
  mortos) a cobertura Linear FFT — **requer decisão humana antes de qualquer implementação**, pois
  envolve julgamento sobre se essa cobertura ainda é desejada.

## Épico D — Saneamento de Artefatos e Comentários Órfãos (🟡 Médio/🟢 Baixo, baixo risco, alto valor de clareza) [TO-DO]

- **F5** — Investigar proveniência e corrigir classificação (ou remover) `golden_cabsim_cpp_stress.bin`.
- **F6** — Invalidação de cache de binários (`render`, `render_ir`) por timestamp/config.
- **F10** — Remover comentário stale/contraditório sobre `wavenet_lite`.

## Épico E — Governança de Frescor de Goldens (🟡 Médio, depende parcialmente do Épico C) [TO-DO]

- **F11** (parte 2) — Avaliar versionar o manifesto de frescor e decidir política bloqueante vs.
  informativa; avaliar integração no `tests-quick.sh` (Fase 2) em vez de exclusivamente no
  `tests-long.sh`.

## Épico F — Documentação (🟢 concluído parcialmente nesta sessão; itens remanescentes de código-fonte) [TO-DO]

- **F12** — ✅ Aplicado nesta sessão (referências a `.gitignore` corrigidas no README).
- **F13** — ✅ Aplicado nesta sessão (nova seção de supply-chain em `docs/testing.md`).
- Pendente: manter esta seção e o catálogo do README sincronizados conforme os Épicos A-E forem
  implementados (cada PR que resolver um finding acima deve atualizar `tests/fixtures/README.md` e,
  se aplicável, `docs/testing.md` no mesmo commit — este é o próprio padrão de manutenção que este
  documento recomenda para não se tornar, ele mesmo, uma nova fonte de drift).
