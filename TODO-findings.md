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

---

## Achado F2: `input_mixin_post_film` e `layer1x1_post_film` têm 3 bugs de paridade genuínos e não relacionados ao Achado F1 — mascarados, não corrigidos, pela implementação da Sprint 1/2

**Status:** ✅ **Corrigido — Sprint 3 (T3.1-T3.8), commits `2108b3f`(T3.1)/`cc892ae`(T3.2)/`8a2de15`(T3.3)/`9312124`(T3.4)/`14612c1`(T3.5)/`ca8c22e`(T3.6-T3.7), 2026-07-10**

**Correções realizadas (B1/B2/B3 — bugs de produção em `input_mixin_post_film`/`layer1x1_post_film`):**

- **T3.1 (B1):** `input_mixin_post_film` agora modula apenas o mixin isolado (antes: modulava `conv+mixin`).
- **T3.2 (B2):** `layer1x1_post_film` agora só aplica com `use_blending` (antes: aplicava incondicionalmente).
- **T3.3 (B3):** `layer1x1_post_film` agora modula apenas o l1x1 isolado (antes: modulava `input+l1x1`).
- **T3.4:** Espelhamento de B1/B2/B3 nos caminhos estáticos CH=3/CH=8.
- **T3.5:** Testes unitários dedicados para B1/B2/B3 (3/3 passam).
- **T3.6:** Restauração dos 4 slots FiLM nos fixtures `wavenet_a2_film_lite`/`_full`.
- **T3.7:** Goldens C++ regenerados com 4 slots — medições finais abaixo.
- **T3.8:** Thresholds em `validation.rs` mantidos em 120 dB/`1.0e-11`/`1.0e-4` — passam.

**Medições finais (T3.7, 4 slots ativos, pós-correção B1/B2/B3 vs. golden C++):**

| Modelo                                | SNR          | ESR          | MR-STFT |
| ------------------------------------- | ------------ | ------------ | ------- |
| `wavenet_a2_film_full` (CH=8)         | **139.4 dB** | **1.15e-14** | 3.52e-5 |
| `wavenet_a2_film_lite` (CH=3)         | **124.2 dB** | **3.83e-13** | 2.43e-5 |
| `wavenet_a2_film_chaos_stress` (CH=3) | 139.0 dB     | 1.25e-14     | —       |

Nota: `_lite` teve SNR ~14 dB menor que na medição de 2 slots pré-correção (138.3 → 124.2 dB), pois
os 2 slots adicionais (`input_mixin_post_film`/`layer1x1_post_film`) agora estão ativos e corretos.
Margem ainda confortável: 4.2 dB sobre o gate de 120 dB.

**Investigação T3.9 (impacto em `wavenet_a2_max.nam`):** Após B1/B2/B3, o ESR **piorou** de 3.61e1
para 1.07e2 (~3×). Conclusão: os bugs de FiLM estavam acidentalmente compensando parte do erro do
`condition_dsp` — caso clássico de "two wrongs make a right". O `condition_dsp` permanece como único
bloqueador real. Ver `TODO-parity.md` §Achado 2 → Medição T3.9 e `docs/cpp_parity_map.md` §7.1.

**Investigação T3.10 (auditoria dos 4 slots restantes):** 2 corretos (`conv_pre_film`,
`activation_pre_film`), 1 bugado (`input_mixin_pre_film` — Bug C1, modula buffer errado), 1 gap
(`head1x1_post_film` — nunca invocado). Detalhes abaixo em § Achado F3.

**Contexto (auditoria da implementação do Achado F1):** as Sprints 1 e 2 (commits `3faa934`,
`445b5cb`, `743710d`, `ee5acd1`, `355a852`, `f04e441`) implementaram corretamente a correção do
bias de identidade proposta no Achado F1 — confirmado: `wavenet_a2_film_lite`/`_full` agora medem
SNR 138,3/138,8 dB (ESR `1.5e-14`/`1.3e-14`), dentro do piso de precisão f32. **Porém**, o commit
`445b5cb` ("align A2-FiLM active keys") silenciosamente **removeu 2 dos 4 slots FiLM ativos**
(`input_mixin_post_film` e `layer1x1_post_film`) dos fixtures `wavenet_a2_film_lite.nam` /
`_full.nam`, sem nenhuma explicação técnica documentada em `TODO-sprints.md`, `TODO-findings.md`
ou no próprio commit. Isso reduz a cobertura de teste do FiLM de 4/8 para 2/8 pontos de inserção
possíveis, sem que ninguém tenha investigado **por que** os outros 2 slots continuavam divergindo
mesmo após a correção do bias de identidade.

### Verificação empírica (auditoria, reproduzível, nenhuma alteração permanente ao repositório)

Reconstruí, em `/tmp/kilo/film_test/`, o fixture `wavenet_a2_film_lite.nam` no estado do commit
`3faa934` (bias de identidade **já corrigido**, mas com os **4 slots FiLM originais ainda
ativos** — antes da remoção do commit seguinte). Renderizei o golden C++ real
(`build/namcore_render/tools/render`) e comparei contra o motor Rust `WaveNetA2Dyn` (via teste
temporário, revertido após a medição):

| Combinação de slots FiLM ativos                                                      | SNR medido   | ESR medido |
| ------------------------------------------------------------------------------------ | ------------ | ---------- |
| `conv_post_film` + `activation_post_film` (2 slots — **o que ficou em produção**)    | **138,3 dB** | `1.48e-14` |
| `conv_post_film` + `input_mixin_post_film` + `activation_post_film` (sem `layer1x1`) | 18,2 dB      | `1.52e-2`  |
| `conv_post_film` + `activation_post_film` + `layer1x1_post_film` (sem `input_mixin`) | 0,2 dB       | `9.58e-1`  |
| Todos os 4 slots originais (`conv`+`mixin`+`act`+`layer1x1`)                         | **−0,8 dB**  | `1.20e0`   |

Isso prova, de forma conclusiva: **o bias de identidade (Achado F1) não tinha absolutamente nada
a ver com a divergência de `input_mixin_post_film` e `layer1x1_post_film`.** Ambos têm bugs de
implementação genuínos e severos, confirmados por leitura direta do código-fonte:

### Bug B1 — `input_mixin_post_film` modula `conv + mixin` combinados; C++ modula só o `mixin`

- **C++ (correto):** `tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp:198-204` — o FiLM
  é aplicado à saída do `_input_mixin` **isoladamente**, e só depois de modulado é somado ao
  `_conv.GetOutput()`: `z = conv + film(mixin)`.
- **Rust (bugado):** `src/models/a2/model/dynamic/process.rs:370-385` — o laço `for c in
  0..z_out_ch { z_scratch[c] += sum; }` (soma o mixin ao `z_scratch`, que já contém o conv output)
  ocorre **antes** de `film.process(&mut z_scratch[..z_out_ch], cond_slice)`. Ou seja, o Rust
  computa `z = film(conv + mixin)`, modulando também o `conv output`, que o C++ nunca modula neste
  ponto.
- **Réplica confirmada** no caminho estático CH=3: `src/models/a2/conv1d_ch3/simd.rs:342-354` (comentário
  "post-mixin, pre-activation" aplicado sobre o buffer já combinado).

### Bug B2 — `layer1x1_post_film` aplicado incondicionalmente; C++ só aplica no modo `BLENDED`

- **C++:** em `model.cpp:217-228` (modo `NONE`) e `:229-247` (modo `GATED`), **não existe nenhuma
  chamada** a `_layer1x1_post_film`. Só o branch `BLENDED` (`model.cpp:248-271`, especificamente
  linhas 265-268) o invoca.
- **Rust:** `src/models/a2/model/dynamic/process.rs:467-483` aplica
  `layer.layer1x1_post_film` **sempre que `!is_last`**, sem checar `use_blending` — apesar de
  `use_blending: bool` já estar disponível como parâmetro da função (`process.rs:326`). Os
  fixtures `wavenet_a2_film_lite`/`_full` e o modelo ainda-quebrado `wavenet_a2_max.nam` usam
  `gating_mode: "none"`, para o qual o C++ **nunca** aplicaria este FiLM — o Rust aplica sempre.
- **Réplica confirmada** no caminho estático: `src/models/a2/conv1d_ch3/simd.rs:450-459`.

### Bug B3 — `layer1x1_post_film`, mesmo se restrito ao modo `BLENDED`, modularia o buffer errado

- **C++:** `model.cpp:265-268` aplica o FiLM à saída do `_layer1x1` **isoladamente**
  (`layer1x1_output`); só depois, em `model.cpp:359-360`, soma-se `input + layer1x1_output`
  (já modulado) para formar `_output_next_layer`. Equação: `output = input + film(l1x1(z))`.
- **Rust:** em `process.rs:467-478`, `layer_in[base+oc] += sum` (soma residual `input + l1x1`)
  ocorre **antes** de `film.process(&mut layer_in[base..base+channels], ...)` (linhas 479-483).
  Equação: `output = film(input + l1x1(z))` — o FiLM acaba modulando também o `input` acumulado de
  camadas anteriores, nunca modulado no C++.

### Conexão crítica com o Achado 2 do `TODO-parity.md` (`wavenet_a2_max.nam` "ativamente quebrado")

Verifiquei que `wavenet_a2_max.nam` (modelo flagship real, `condition_size=8`, atualmente
desabilitado via `is_disabled_broken_a2_flagship`, ESR≈3,61e1 documentado no Achado 2) **também
tem `input_mixin_post_film` e `layer1x1_post_film` ativos com `gating_mode: "none"`** em todas as
suas camadas. Isso significa que os Bugs B1/B2/B3 aqui identificados **são, no mínimo, um
contribuinte adicional confirmado** para a divergência catastrófica desse modelo — até agora
atribuída inteiramente ao bug do `condition_dsp` no oráculo f64 (que é um problema de teste, não
de produção). B1/B2/B3, em contraste, **são bugs de produção** (`src/models/a2/model/dynamic/process.rs`
e caminhos estáticos espelhados), que afetam qualquer usuário final que carregue um `.nam` real
com esses slots de FiLM ativos — independentemente de testes automatizados.

### Proposta de solução

1. **Corrigir B1** (`src/models/a2/model/dynamic/process.rs:370-385`, e réplicas em
   `conv1d_ch3/simd.rs`/`conv1d_ch8/simd.rs`): computar o `mixin` em um buffer temporário separado
   (`mixin_scratch`), aplicar `input_mixin_post_film` a esse buffer isolado, e só então somá-lo ao
   `z_scratch` (que já contém o `conv` output).

2. **Corrigir B2**: adicionar guard `if use_blending { ... }` em torno da chamada a
   `layer.layer1x1_post_film` em `process.rs:479-483` e nos caminhos estáticos equivalentes —
   espelhando exatamente a condição do C++ (`model.cpp:248` só entra no branch `BLENDED`).

3. **Corrigir B3**: computar o `layer1x1` em um buffer temporário isolado, aplicar
   `layer1x1_post_film` a esse buffer isolado (não ao `layer_in` já somado ao residual), e só
   então somar o resultado ao `layer_in`.

4. **Regenerar os fixtures FiLM com os 4 slots originais restaurados** (`input_mixin_post_film` e
   `layer1x1_post_film` de volta a `active: true`) após corrigir B1/B2/B3, e remedir. Só então a
   cobertura de teste do FiLM volta a ser completa (4/8 pontos de inserção, os únicos exercitados
   pelos fixtures sintéticos atuais).

5. **Reexecutar a suíte de `wavenet_a2_max.nam`** (mesmo desabilitada por padrão) após B1/B2/B3,
    para medir se a correção reduz materialmente o ESR≈3,61e1 documentado no Achado 2 — isso
    ajudaria a isolar quanto da divergência daquele modelo pertence ao `condition_dsp` (ainda não
    fechado) vs. a estes 3 bugs (agora identificados e corrigíveis).

   > **✅ Medido em 2026-07-10 (T3.9):** ESR **piorou** de 3.61e1 para 1.07e2, SNR de −15.6 dB
   > para −20.3 dB, MSE de 2.46e3 para 7.30e3 (~3× pior em todas as métricas). Conclusão:
   > B1/B2/B3 não são a causa raiz da divergência do `wavenet_a2_max.nam`. Os bugs de FiLM
   > estavam acidentalmente compensando parte do erro do `condition_dsp`. Com FiLM corrigido e
   > `condition_dsp` ainda quebrado, o cancelamento parcial de erro desapareceu. O
   > `condition_dsp` (§4.4 do `cpp_parity_map.md`) permanece como único bloqueador real.
   > Detalhes completos em `TODO-parity.md` §Achado 2 → Medição T3.9.

6. **Investigar os demais 4 slots de FiLM ainda não testados por nenhum fixture** (`conv_pre_film`,
   `input_mixin_pre_film`, `activation_pre_film`, `head1x1_post_film`) com o mesmo rigor — dado
   que 2 de 4 slots já testados continham bugs, não há garantia de que os slots nunca exercitados
   estejam corretos.

### Risco e escopo

- **Risco de implementação: médio.** Ao contrário do Achado F1 (só fixtures), esta correção **é
  código de produção** (`src/models/a2/model/dynamic/process.rs` e os caminhos estáticos
  `conv1d_ch3`/`conv1d_ch8`), usado por qualquer `.nam` real com FiLM. Requer testes de regressão
  cuidadosos (buffers temporários adicionais, sem quebrar RT-safety/zero-alocação no hot path).
- **Prioridade recomendada: alta.** Estes bugs afetam a correção funcional do motor de inferência
  para qualquer modelo FiLM real (não é uma questão de fixture/documentação como o Achado F1), e
  têm relação direta com um item já crítico (`wavenet_a2_max.nam`, Achado 2 do `TODO-parity.md`).

### Epic F2 — Correção dos bugs de aplicação FiLM em `input_mixin_post_film`/`layer1x1_post_film`

1. Corrigir B1 (mixin isolado antes do FiLM) no motor dinâmico e nos caminhos estáticos CH3/CH8.

2. Corrigir B2 (guard de `use_blending`) no motor dinâmico e nos caminhos estáticos CH3/CH8.

3. Corrigir B3 (l1x1 isolado antes do FiLM, soma do residual depois) no motor dinâmico e nos
   caminhos estáticos CH3/CH8.

4. Restaurar os 4 slots FiLM originais nos fixtures `wavenet_a2_film_lite`/`_full`, regenerar
   goldens, remedir e recalibrar thresholds.

5. Reexecutar (mesmo que manualmente) o cenário `wavenet_a2_max.nam` para quantificar o impacto
   nesse modelo ainda desabilitado (Achado 2, `TODO-parity.md`).

6. Auditar os 4 slots FiLM restantes (`conv_pre_film`, `input_mixin_pre_film`,
   `activation_pre_film`, `head1x1_post_film`), hoje sem nenhuma cobertura de teste.

   > **✅ Auditado em 2026-07-10 (T3.10, Achado F3):** 2 de 4 slots restantes estão corretos
   > (`conv_pre_film` e `activation_pre_film`). Um está **quebrado** (`input_mixin_pre_film` —
   > Bug C1, modula buffer errado) e um é um **gap documentado** (`head1x1_post_film` — nunca
   > invocado). Detalhes abaixo em § Achado F3.

---

## Achado F3 — Auditoria dos 4 slots FiLM restantes sem cobertura de teste (T3.10, 2026-07-10)

Metodologia: leitura pareada C++/Rust das implementações de cada slot contra `model.cpp:166-376`
(A2 Layer::Process) nos três caminhos Rust (dinâmico `process_frame_dyn`, estático CH=3
`conv1d_ch3/simd.rs`, estático CH=8 `conv1d_ch8/simd.rs`).

### Slot 0: `conv_pre_film` — ✅ CORRETO

| Aspecto                 | C++ (`model.cpp:172-177`)        | Rust (`dynamic/process.rs:206-227`, `static/process.rs:163-176`)                    |
| ----------------------- | -------------------------------- | ----------------------------------------------------------------------------------- |
| **Buffer alvo**         | `input` (sinal de entrada bruto) | `buf` (buffer de histórico, mesma semântica que `input`)                            |
| **Posição no pipeline** | Antes de `_conv.Process()`       | Antes de `conv.process_single_frame()` (dinâmico) / `conv1d_ch*_forward` (estático) |

**Conclusão:** O FiLM modula o buffer de entrada **antes** da convolução dilatada, modificando o
sinal que a convolução vê — exatamente como o C++. Semântica e posição no pipeline 100%
equivalentes nos três caminhos.

### Slot 2: `input_mixin_pre_film` — 🔴 BUG (C1)

| Aspecto                   | C++ (`model.cpp:188-197`)                                                                                          | Rust (`dynamic/process.rs:370-374`, `conv1d_ch3/simd.rs:297-299`, `conv1d_ch8/simd.rs:179-181`) |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| **Buffer alvo**           | `condition` (sinal de condicionamento) — FiLM processa `condition` com `condition` como condição (self-modulation) | `z_scratch` / `z_buf` (saída da convolução dilatada)                                            |
| **O que consome a saída** | `_input_mixin.process_(film_output)` — mixin opera sobre condição modulada por FiLM                                | `mixin_scratch` é computado a partir de `cond_slice` **raw** (sem modulação FiLM)               |
| **Equação resultante**    | `mixin_w × film(condition)`                                                                                        | `film(conv_output)` seguido de `mixin_w × condition` (não modulada)                             |

Detalhamento por caminho:

- **Dinâmico** (`process.rs:370-374`): `film.process(&mut z_scratch[..z_out_ch], cond_slice)` — modula `z_scratch` (conv output), mesmo buffer que `conv_post_film`.
- **Estático CH=3** (`conv1d_ch3/simd.rs:297-299`): `film.process(z_slice, cond)` — idêntico ao `conv_post_film` na linha 294.
- **Estático CH=8** (`conv1d_ch8/simd.rs:179-181`): `film.process(z_slice, cond)` — idêntico ao `conv_post_film` na linha 176.

**Consequência:** Em Rust, `input_mixin_pre_film` é funcionalmente idêntico a aplicar
`conv_post_film` uma segunda vez — redundante e semanticamente incorreto. Nenhum fixture
existente ativa este slot, portanto o bug nunca foi detectado por testes. Todo modelo `.nam`
real com `input_mixin_pre_film: true` produzirá saída divergente do C++.

**Correção necessária:** Aplicar `input_mixin_pre_film` ao buffer de condição (`cond_slice`)
antes da multiplicação `mixin_w × cond`, em vez de aplicá-lo a `z_scratch`. Requer buffer
temporário adicional (pré-modificar `cond_slice` em scratch próprio, ou `mixin_scratch` receber
`cond` modulada por FiLM antes da multiplicação por `mixin_w`).

### Slot 4: `activation_pre_film` — ✅ CORRETO

| Aspecto                 | C++ (`model.cpp:206-209`)                          | Rust (`dynamic/process.rs:400-404`, `conv1d_ch3/simd.rs:362-373`, `conv1d_ch8/simd.rs:207-209`) |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Buffer alvo**         | `_z` (conv output + input mixin output combinados) | `z_scratch` / `z_buf` após soma `z_scratch[c] += mixin_scratch[c]`                              |
| **Posição no pipeline** | Após soma mixin, antes da ativação                 | Após soma mixin (linha 397), antes da ativação (linha 406+)                                     |

**Conclusão:** O FiLM modula o buffer combinado conv+mixin **antes** da ativação, exatamente como
o C++. Todos os três caminhos estão corretos.

### Slot 7: `head1x1_post_film` — 🟡 GAP (não implementado)

| Aspecto                 | C++ (`model.cpp:283-287`)                                                    | Rust               |
| ----------------------- | ---------------------------------------------------------------------------- | ------------------ |
| **Buffer alvo**         | `head1x1_output` (saída de `_head1x1`) — FiLM modula a projeção do cabeçalho | **Nunca invocado** |
| **Posição no pipeline** | Após `head1x1.process_()`, antes da cópia para `output_head`                 | —                  |

O slot é **carregado** corretamente:

- `set_weights.rs:269` — slot 7 mapeia para `layer.head1x1_post_film`
- `weights_layout.rs:22` — `("head1x1_post_film", 7)`
- `film.rs:239,271` — presente no `FilmBlock`

Mas **nenhum** caminho de processamento o invoca:

- `dynamic/process.rs` — `layer.head1x1_post_film` nunca é acessado
- `conv1d_ch3/simd.rs` — `film.head1x1_post_film` nunca é acessado
- `conv1d_ch8/simd.rs` — `film.head1x1_post_film` nunca é acessado

O comentário em `layer.rs:67` reconhece o gap: `"FiLM after head 1x1 (reserved for future general A2 engine)."`. Não é um bug no sentido estrito (é um gap documentado), mas é uma divergência de
paridade: qualquer modelo `.nam` com `head1x1_post_film: true` produzirá saída diferente entre
C++ e Rust.

**Correção necessária:** Aplicar `head1x1_post_film` ao buffer `head1x1_scratch` (dinâmico) /
`output_head` (estático) antes da acumulação em `head_accum`. O C++ aplica após `head1x1.process_()`
e antes da cópia para `output_head` — a posição equivalente em Rust é entre a computação do
`head1x1` e a cópia/soma em `head_accum`.

### Resumo consolidado

| Slot | Nome                   | Status     | Severidade                                               |
| ---- | ---------------------- | ---------- | -------------------------------------------------------- |
| 0    | `conv_pre_film`        | ✅ Correto | —                                                        |
| 2    | `input_mixin_pre_film` | 🔴 Bug C1  | Alta — buffer errado, semântica incorreta                |
| 4    | `activation_pre_film`  | ✅ Correto | —                                                        |
| 7    | `head1x1_post_film`    | 🟡 Gap     | Média — documentado como "reserved", mas funcional no C++|

### Proposta de correção (Epic F3)

1. **Corrigir Bug C1** [DONE] (`input_mixin_pre_film`): modificar o `cond_slice` (ou scratch de condição)
   com FiLM antes de `mixin_w × cond`, espelhando `model.cpp:188-197`. Aplica-se ao motor
   dinâmico e aos caminhos estáticos CH=3/CH=8.

2. **Implementar `head1x1_post_film`** [DONE]: aplicar FiLM ao buffer `head1x1_scratch` / `output_head`
   após a projeção `head1x1` e antes da acumulação, espelhando `model.cpp:283-287`.

3. **Criar fixture(s) sintético(s)** [GIT STAGED] exercitando `input_mixin_pre_film` e `head1x1_post_film` para
   cobertura de teste — similar à metodologia do Achado F2 (fixtures sintéticos com ablação de
   slots).
