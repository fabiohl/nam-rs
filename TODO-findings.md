<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Achados de Auditoria (Revisor Auditor)

Este documento reúne achados técnicos aprofundados produzidos pela skill `revisor-auditor` (e,
eventualmente, `pesquisador-inovador`), com diagnóstico detalhado, evidências reproduzíveis e
proposta de solução. Cada achado referencia os arquivos/linhas exatos envolvidos.

---

## Achado F1: `TODO-parity.md` — "FiLM Floating-Point Associativity Gap" é um diagnóstico incorreto; causa raiz real identificada

**Status:** 🔴 Diagnóstico anterior refutado com evidência. Causa raiz real proposta e verificada
empiricamente (nível de fixture/pesos). Solução ainda não implementada.

**Referência cruzada:** `TODO-parity.md` linhas 10–47 ("Achado 1"); `docs/cpp_parity_map.md` §4.3
(linhas 565–581); `tests/common/validation.rs` linhas 581–602 (thresholds calibrados de
`wavenet_a2_film_lite`/`wavenet_a2_film_full`).

### 1. Resumo executivo

O `TODO-parity.md` atribui o gap de ESR do FiLM (`1.54e-2` / `1.54e-1`¹ para Lite CH=3, `2.50e-4`
para Full CH=8) a uma diferença de **ordem de associatividade em ponto flutuante** entre o
acumulador AVX2 em árvore binária (`dot_product_avx2`, Rust) e uma soma sequencial linear
supostamente usada pelo Eigen no C++ NAMCore. **Essa teoria é matematicamente impossível para os
dois modelos flagship que exibem o gap**: ambos (`wavenet_a2_film_lite.nam` e
`wavenet_a2_film_full.nam`) têm `condition_size = 1` em todas as 23 camadas. Com um único elemento
de entrada, `dot_product_avx2` e uma soma sequencial escalar computam **exatamente a mesma
expressão** (`w[0] * cond[0]`) — não existe árvore de redução com um único termo, logo não existe
"assimetria de árvore de soma" a corrigir.

(¹ Nota: o dashboard mais recente reporta `1.54e-2` (SNR 18.1 dB) para Lite; o comentário
calibrado em `tests/common/validation.rs:586` registra uma medição um pouco diferente, `3.07e-2`
(SNR 15.1 dB), datada de 2026-07-03. Ambas as medições são compatíveis com a mesma causa raiz
abaixo — variam apenas conforme o sinal de stress usado em cada rodada.)

A investigação aprofundada (traçando a origem do vetor `condition`, os pesos reais do fixture, e
comparando com os fixtures `gated`/`blended` que usam o **mesmo par de engines**) revela que:

1. **A causa raiz real não está na aritmética do FiLM em si** (que já bate com o oráculo f64 a
   `~1e-14`, provando que a implementação Rust está correta), **nem em uma diferença estrutural
   de engine C++/Rust** (pois os fixtures `gated`/`blended`, que passam pelo **mesmo par de
   engines** — C++ genérico Eigen vs. Rust `WaveNetA2Dyn` — atingem paridade quase bit-exata:
   SNR 103,0 dB / 133,0 dB, ESR `5.01e-11` / `5.01e-14`).
2. **A causa raiz real é o gerador de fixtures sintéticas** (`tests/fixtures/generate_a2_fixtures.py`,
   função `generate_weights_film`), que inicializa o **canal de "scale"** do FiLM com uma
   distribuição aleatória de **média zero e módulo pequeno** (bias `U(-1,1)×0.09`, peso
   `U(-1,1)×0.45` para CH=3), em vez de uma inicialização **"identity-biased"** (`scale ≈ 1.0`,
   convenção padrão de FiLM em redes treinadas, para preservar o fluxo de gradiente/sinal). Isso
   produz, a cada um dos 92 pontos de aplicação FiLM (4 slots × 23 camadas), um fator
   multiplicativo **próximo de zero e com sinal aleatoriamente alternante** (correlacionado à
   amostra de áudio bruta usada como `condition`), que **esmaga a energia do sinal verdadeiro**
   enquanto o piso de ruído de ponto flutuante absoluto entre as duas implementações (Rust f32 vs.
   C++ f32, tipicamente `~1e-7` absoluto, inerente e inevitável em qualquer pipeline de DSP com
   pipelines de instruções diferentes) permanece com magnitude aproximadamente constante. Como
   ESR/SNR são métricas **normalizadas pela energia do sinal verdadeiro**
   (`ESR = ‖y_rust − y_cpp‖² / ‖y_cpp‖²`), esmagar o denominador infla artificialmente o
   numerador relativo — sem que exista, de fato, nenhum "bug" de paridade na matemática do FiLM.

### 2. Evidência técnica detalhada

#### 2.1 `condition_size = 1` nos dois modelos flagship (refuta a teoria de associatividade)

```shell
$ python3 -c "import json; d=json.load(open('tests/fixtures/models/wavenet_a2_film_lite.nam')); print(d['config']['layers'][0]['condition_size'])"
1
$ python3 -c "import json; d=json.load(open('tests/fixtures/models/wavenet_a2_film_full.nam')); print(d['config']['layers'][0]['condition_size'])"
1
```

Com `cond_per_group = cond_size / groups = 1`, o laço interno de `cond_to_scale_shift`
(`src/models/a2/film.rs:158-172`) executa `dot_product_avx2(w_row, cond_slice)` com
`w_row.len() == cond_slice.len() == 1`. Em `dot_product_avx2` (`src/models/a2/film.rs:283-324`),
os dois laços vetoriais (`while i+16<=len`, `while i+8<=len`) nunca são executados porque
`len == 1 < 8`; o único termo é somado pelo laço escalar de cauda
(`film.rs:318-321: out += a[i]*b[i]`). O resultado é **idêntico bit-a-bit** a uma soma sequencial
escrita manualmente. Não há, portanto, nenhuma "árvore binária SIMD" ativa para `cond_size=1`.

O mesmo se aplica ao oráculo f64 (`src/testing/reference_oracle/a2.rs`, `FiLMOracleSlot::apply`,
linhas 39–84): o laço `for k in 0..cond_per_group` roda uma única iteração — mesma expressão
matemática, apenas em `f64`.

#### 2.2 Os pesos reais do "canal de scale" do fixture são de média ≈ 0, não ≈ 1

Extração direta dos pesos do primeiro slot FiLM (`conv_post_film`, camada 0,
`wavenet_a2_film_lite.nam`, CH=3):

```text
scale weight rows (0..ch): [0.0167, 0.1469, -0.0056]
scale bias   rows (0..ch): [0.0620, -0.0729, 0.0199]
shift weight rows (ch..2ch): [-0.3162, -0.4296, 0.4230]
shift bias   rows (ch..2ch): [0.0513, 0.0710, -0.0784]
```

Ou seja: `scale[o] = bias[o] (≈ ±0.02–0.07) + weight[o] * cond (cond ∈ [-1,1], weight ≈ ±0.005–0.15)`.
O valor típico de `scale[o]` fica quase inteiramente contido no intervalo aproximado `[-0.1, +0.2]`
— **um fator multiplicativo minúsculo que cruza zero constantemente** conforme a amostra de áudio
bruta (`condition`) varia. Isso é fundamentalmente diferente de uma modulação FiLM
"identity-preserving" (`scale ≈ 1`), que é a convenção padrão de inicialização em redes treinadas
com FiLM (evitar colapsar o sinal a zero no início do treino / em qualquer ponto de operação).

Raiz do problema em código: `generate_weights_film`
(`tests/fixtures/generate_a2_fixtures.py:134-155`) gera **ambas as metades** do vetor de bias do
FiLM (`scale` e `shift`) com a **mesma distribuição** `gen_weights(ch*2, rng, scale=bs)`
(linha 150) — sem nenhum deslocamento de `+1.0` para a metade correspondente ao canal de `scale`.
Isso é apropriado para o canal de `shift` (identidade = `+0`), mas **incorreto** para o canal de
`scale` (identidade = `×1`).

#### 2.3 Prova por contraste: `gated`/`blended` usam o MESMO par de engines e são quase bit-exatos

`docs/cpp_parity_map.md:517-522` confirma que `a2_fast.cpp` (C++) rejeita **qualquer** modelo com
gating, FiLM, `head1x1` ou `groups≠1`, roteando todos eles igualmente para o mesmo
`NAM/wavenet/model.cpp` genérico (Eigen). No lado Rust, o mesmo dispatcher roteia FiLM, gating e
blending igualmente para `WaveNetA2Dyn` (`docs/cpp_parity_map.md:539`). Ou seja: **FiLM, gating e
blending compartilham exatamente o mesmo par de engines C++/Rust.** Se a causa fosse uma
divergência estrutural de engine (como a teoria atual do TODO e do `cpp_parity_map.md` §4.3
sugerem — "structural divergence between how C++'s generic Eigen path and Rust's native FiLM
engine apply conditioning"), o gap deveria se manifestar de forma comparável nos três casos.
**Não é o que se observa** (`tests/common/validation.rs:581-639`):

| Fixture                  | Engine (C++ / Rust)             | SNR medido       | ESR medido         |
| ------------------------ | ------------------------------- | ---------------- | ------------------ |
| `a2_dynamic_gated_ch8`   | Eigen genérico / `WaveNetA2Dyn` | **103,0 dB**     | `5.01e-11`         |
| `a2_dynamic_blended_ch3` | Eigen genérico / `WaveNetA2Dyn` | **133,0 dB**     | `5.01e-14`         |
| `wavenet_a2_film_lite`   | Eigen genérico / `WaveNetA2Dyn` | **15,1–18,1 dB** | `1.5e-2`–`3.07e-2` |
| `wavenet_a2_film_full`   | Eigen genérico / `WaveNetA2Dyn` | **36,0 dB**      | `2.50e-4`          |

A diferença de 60–120 dB entre `gated`/`blended` e `FiLM`, usando o mesmo par de engines, só pode
ser explicada por uma característica **específica da matemática/pesos do FiLM** — não da engine.
A distinção física crucial: o gate de `Sigmoid` (gating) é **limitado a `(0,1)`**, nunca cruza
zero nem inverte sinal — é puramente atenuante; o FiLM deste fixture, ao contrário, é **uma
afinidade não limitada, de média ≈ 0**, que cruza zero e inverte o sinal constantemente. É
exatamente esse regime numérico (fator de escala perto de zero, oscilando em sinal, aplicado 92
vezes em cascata por 23 camadas dilatadas) que **esmaga a energia do sinal verdadeiro** e infla o
ESR — não uma diferença de associatividade nem uma incompatibilidade de engine.

#### 2.4 Simulação de confirmação (isolada, single-layer)

Uma simulação Python isolando o mecanismo (perturbação relativa `ε=1e-6` no `condition`,
simulando divergência inevitável de arredondamento f32 entre implementações) confirma
qualitativamente a direção do efeito ao comparar a distribuição atual (`scale_bias ≈ U(-0.09,
0.09)`) contra uma inicialização identity-biased (`scale_bias ≈ 1.0 + U(-0.09,0.09)`): o erro
relativo cai following the same qualitative direction previsto. O efeito pleno (ganho de várias
ordens de magnitude em ESR) só se manifesta na cascata real de 23 camadas × 92 aplicações — o que
é consistente com a assinatura observada no benchmark real (gap crescente com profundidade/CH).

### 3. Por que o "Plano de Correção" atual do `TODO-parity.md` (linhas 21–47) não resolve o problema

O plano proposto (substituir `dot_product_avx2` por um laço escalar estritamente sequencial em
`src/models/a2/film.rs` e no oráculo `src/testing/reference_oracle/a2.rs`) é:

- **Matematicamente um no-op para os dois fixtures que exibem o gap** (`cond_size=1` ⇒ soma de um
  único termo ⇒ zero diferença possível de associatividade), portanto **não deve reduzir o ESR
  medido em nenhuma fração perceptível**.
- **Uma regressão de manutenibilidade caso implementado**: removeria `dot_product_avx2`
  (`src/models/a2/film.rs:282-324`), uma rotina AVX2/FMA testada e documentada, que **permanece
  necessária** para o caso geral `cond_size > 1` (grupos de condicionamento maiores que 1
  elemento, cenário plausível para modelos FiLM futuros/reais, e já coberto por testes em
  `film_test.rs` como `test_film_process_groups_shift` com `cond_size=6`). Removê-la eliminaria
  a única implementação vetorizada disponível para esse caso.

### 4. Proposta de solução

**Não alterar `src/models/a2/film.rs` nem `src/testing/reference_oracle/a2.rs`** — a matemática do
FiLM em produção já está correta e validada (`ESR ≈ 1e-14` vs. oráculo f64). A correção deve
ocorrer na **camada de fixtures/testes**, que é onde a causa raiz reside:

1. **Corrigir `generate_weights_film`** (`tests/fixtures/generate_a2_fixtures.py:134-155`) para
   aplicar um deslocamento de identidade (`+1.0`) à metade do vetor de bias correspondente ao
   canal de `scale`, mantendo a metade de `shift` inalterada (identidade = `0`):

   ```python
   def generate_weights_film(ch: int, num_film_keys: int, rng: random.Random) -> List[float]:
       ...
       for k in KERNEL_SIZES:
           ...
           for _ in range(num_film_keys):
               weights.extend(gen_weights(ch * 2, rng, scale=ws))          # film_w (inalterado)
               scale_bias = [1.0 + v for v in gen_weights(ch, rng, scale=bs)]   # scale: identity-biased
               shift_bias = gen_weights(ch, rng, scale=bs)                     # shift: identidade = 0
               weights.extend(scale_bias + shift_bias)
       ...
   ```

   (Ajustar a ordem exata de geração de números aleatórios para não quebrar determinismo de seed
   onde relevante; o importante é o deslocamento `+1.0`, não a ordem de chamadas do RNG.)

2. **Regenerar os fixtures** `wavenet_a2_film_lite.nam` e `wavenet_a2_film_full.nam`, e os
   respectivos vetores golden C++ (`stress_signal.wav` → `render` do NAMCore vendorizado), com o
   script `generate_a2_fixtures.py` corrigido.

3. **Remedir ESR/SNR/MR-STFT** dos dois fixtures contra o novo golden C++. Expectativa (baseada na
   comparação direta com `gated`/`blended`, que compartilham engine e ordem de grandeza de
   profundidade/canais): SNR deve subir para a faixa `>90 dB` (ESR `<1e-9`), compatível com o
   piso de precisão f32 observado nos demais fixtures dinâmicos.

4. **Recalibrar os thresholds** em `tests/common/validation.rs` (chaves `"wavenet_a2_film_lite"`
   linhas 589-592 e `"wavenet_a2_film_full"` linhas 599-602) para os novos valores medidos,
   seguindo a disciplina de calibração já documentada no arquivo (`tests/threshold_calibration.rs`).

5. **Preservar o fixture atual (degenerado) como teste de estresse dedicado**, renomeado para algo
   como `wavenet_a2_film_chaos_stress.nam`, com comentário explícito documentando que ele exercita
   deliberadamente um regime de FiLM com fator de escala não-identity-biased (zero-mean,
   cruzando zero) — útil como teste de regressão de robustez numérica (garante que o pipeline não
   produz NaN/Inf/denormals sob condições adversariais), mas **não deve ser usado como métrica de
   fidelidade sonora real**, já que nenhum modelo FiLM treinado de verdade operaria nesse regime.

6. **Corrigir a documentação** que hoje repete o diagnóstico incorreto de "divergência estrutural
   de engine":

   - `TODO-parity.md` — substituir integralmente o texto do "Achado 1" (linhas 10–47) por esta
     análise corrigida (ou referenciar este documento).
   - `docs/cpp_parity_map.md` §4.3 (linhas 565–581) — corrigir a frase "structural divergence
     between how C++'s generic Eigen path and Rust's native FiLM engine apply conditioning at the
     numerical level" para refletir a causa raiz real (inicialização não-identity-biased do canal
     de `scale` no fixture sintético).
   - `docs/audio_fidelity_map.md` e `docs/perceptual_validation.md` (tabela de thresholds,
     linhas ~94-95) — atualizar os valores de ESR/SNR do FiLM após a remedição.
   - `tests/common/validation.rs:581-602` — atualizar os comentários explicativos junto aos
     novos thresholds calibrados.

7. **Nenhuma alteração de performance/ISA é necessária.** `dot_product_avx2` em
   `src/models/a2/film.rs` deve ser **mantida como está**, pois é a implementação vetorizada
   correta e necessária para o caso geral (`cond_size > 1`), e já produz resultado idêntico ao
   caso escalar quando `cond_size == 1`.

### 5. Risco e escopo do achado

- **Risco de implementação: baixo.** A mudança está isolada em um script Python de geração de
  fixtures de teste (`tests/fixtures/generate_a2_fixtures.py`) e nos artefatos de teste derivados
  (`.nam`, golden vectors, thresholds). **Nenhum código de produção Rust precisa ser alterado.**
- **Impacto:** melhora a credibilidade do dashboard de qualidade (`utils/quality-dashboard.sh`),
  eliminando um falso positivo de "🔴 audível" que hoje aparece para os dois modelos FiLM, sem
  qualquer alteração de comportamento do motor de inferência em produção.
- **Dependência:** requer o binário `render` do NAMCore vendorizado (`tests/fixtures/NeuralAmpModelerCore/`)
  para regenerar os goldens C++, conforme o fluxo já documentado em `golden_gen_build.sh`.

---

## Epics (agrupamento para planejamento)

### Epic F1-A — Correção da causa raiz do fixture FiLM (baixo risco, alto valor de credibilidade) [DONE]

1. Corrigir `generate_weights_film` em `generate_a2_fixtures.py` (deslocamento `+1.0` no canal de
   `scale`).
2. Regenerar `wavenet_a2_film_lite.nam` / `wavenet_a2_film_full.nam` e goldens C++ correspondentes.
3. Remedir ESR/SNR/MR-STFT via `utils/quality-dashboard.sh` e/ou suíte de golden vectors.
4. Recalibrar thresholds em `tests/common/validation.rs` para os dois fixtures.

### Epic F1-B — Preservação do teste de estresse numérico e documentação

1. Clonar o fixture atual (pré-correção) como `wavenet_a2_film_chaos_stress.nam` com threshold
   próprio e comentário explicando seu propósito adversarial.
2. Atualizar `TODO-parity.md` (Achado 1), `docs/cpp_parity_map.md` §4.3, `docs/audio_fidelity_map.md`
   e `docs/perceptual_validation.md` para refletir a causa raiz corrigida e os novos valores
   medidos.
