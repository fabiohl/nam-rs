<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Levantamento de Pontos e Propostas de Solução

> Artefato gerado pela skill `planejador-arquiteto` sob **premissa estratégica redefinida
> pelo PO em 2026-07-01**. Escrito em pt-BR, formato rico e detalhado, com proposta de
> solução e agrupamento em Épicos ao final.

---

## Premissa estratégica inviolável (decisão do PO, 2026-07-01)

1. **O C++ NAMCore (`tests/fixtures/NeuralAmpModelerCore/`, pin v0.5.4) é a ÚNICA fonte de

> verdade** para paridade. O motor Rust deve reproduzir **exatamente** a saída do C++,
> validada pelos golden vectors (`tests/fixtures/*.bin`, gerados pela ferramenta `render` do
> C++ via `tests/fixtures/golden_gen_build.sh`).

1. O **oráculo f64** (`src/testing/reference_oracle.rs`) e a **âncora NumPy**

> (`tests/fixtures/scripts/validate_oracle_f64.py`) são **ferramentas auxiliares de
> decomposição de erro** — **não** árbitros de correção. Se divergirem do C++, **o oráculo está
> errado** e deve ser corrigido; nunca o motor deve ser "corrigido" para casar com o oráculo.

1. A **barreira defensiva da qualidade** é o **cross-validation contra o C++** (golden vectors
>
> + `tests/cpp_parity.rs`), não o oráculo.
>
1. **Medir antes de cortar:** nenhuma alteração de produção sem antes medir o estado atual

> contra o golden do C++ e registrar o resultado empírico. (A Rodada 4 de auditoria violou
> este princípio e foi descartada — ver PM-C.)
>
> **Nota de descarte:** O conteúdo histórico das Rodadas 1–4 foi removido. A Rodada 4 tratou
> o oráculo como verdade e concluiu erroneamente por uma "regressão de 93 dB". A premissa era
> falsa (PM-C). Os findings válidos (ex.: `println!` de debug removido, refs quebradas) já
> foram aplicados no commit `84c3ad1` e não precisam ser re-documentados.

---

## Contexto: o que a auditoria correta (grounded no C++) encontrou

A Sprint S14.2 (PM-15 histórica) adicionou suporte a multi-array cascade + `condition_dsp` +
`head1x1` agrupado, permitindo que o modelo flagship `wavenet_a2_max.nam` (v0.6.0) **carregue**
e roteie para `WaveNetA2Dyn`. A questão aberta é: **a inferência reproduz o C++?**

A verificação contra o C++ (não contra o oráculo) revela:

1. **Dimensão do `condition_dsp`:** o C++ faz

> `_condition_output.resize(condition_output_channels, maxBufferSize)`
> (`NAM/wavenet/model.cpp:659-660`) e alimenta `_condition_output` (8 canais/frame para o
> a2_max) aos layer arrays (`model.cpp:761,770`). A **produção Rust também produz 8/frame**
> (`condition_size=8`). Já o **oráculo produz 1/frame** (`reference_oracle.rs` retorna um
> `Vec<f64>` escalar por frame). Logo, **o oráculo tem dimensão incompatível com o C++**.

1. **Contagem de pesos `head1x1`:** o C++ constrói `Conv1x1(in=bottleneck,

> out=head1x1.out_channels, bias=true, groups=head1x1_groups)`(`NAM/wavenet/detail.h:76`).
> Para conv agrupado, a contagem de pesos =`out_channels × (bottleneck/groups)` e bias =
> `out_channels`. Para o`condition_dsp` array[0] do a2_max (bottleneck=6, out=6, groups=3):
> **pesos = 6×2 = 12, bias = 6**. A **produção committed lê 12** (via `head_accum_size`=
> out_channels); o **oráculo lê 6** (via`channels`). **O oráculo está errado.**

1. Consequência direta: a "evidência" da Rodada 4 (ESR 93 dB "produção vs oráculo") era uma

> **comparação de dimensões incompatíveis** (1/frame × 8/frame) **com um oráculo que tem a
> contagem de pesos errada** — matematicamente sem sentido e **descartada**.

---

## Findings (Constatações)

### PM-A — Oráculo f64 e âncora NumPy divergem do C++ no path `condition_dsp` (duplo defeito)

+ **ID:** PM-A · **Severidade:** Alta (ferramenta de validação comprometida) · **Risco da correção:** Médio (test-only, mas alcance amplo)
+ **Problema:** O oráculo f64 e a âncora NumPy concordam entre si a < 1e-12

> (`test_oracle_vs_python_anchor_a2_generic` passa), **mas ambos divergem do C++ NAMCore** em
> dois pontos materiais no path `condition_dsp`:
>
> 1. **Dimensão de saída:** oráculo/NumPy produzem **1 valor/frame**; o C++ produz
>    **`condition_size` valores/frame** (`model.cpp:659-660, 722-726`). O oráculo usa
>    `oracle_forward(&cond_model, input, ...)` que retorna `Vec<f64>` de tamanho `num_frames`
>    (`reference_oracle.rs:932-936`), ignorando o head_size multi-canal do sub-modelo.
> 2. **Contagem de pesos `head1x1`:** oráculo/NumPy leem `channels × h1_in_size` e bias
>    `channels` (`reference_oracle.rs:1124-1130`; `validate_oracle_f64.py:791-794`); o C++ lê
>    `out_channels × (bottleneck/groups)` e bias `out_channels` (`detail.h:76`). Quando
>    `head1x1.out_channels != channels` (caso real do cd array[0]: ch=3, out=6), o oráculo lê
>    **metade** dos pesos e dessincroniza seu próprio cursor.
>
+ **Evidências (C++):**
>
> + `tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp:659` —
>   `condition_output_channels = _condition_dsp->NumOutputChannels(); _condition_output.resize(...)`.
> + `tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/detail.h:76` —
>   `Conv1x1(params.bottleneck, params.head1x1_params.out_channels, true, params.head1x1_params.groups)`.
> + `tests/fixtures/NeuralAmpModelerCore/NAM/wavenet/model.cpp:761,770` — layer arrays consomem
>   `_condition_output` (8 canais).
>
+ **Impacto:** Qualquer teste "produção vs oráculo" para modelos com `condition_dsp` +

> `head1x1` agrupado com `out_channels != channels` é **inválido** como evidência de bug na
> produção. Os 4 testes `#[ignore]`d "S14.2-followup" baseiam-se neste oráculo e não podem
> ser usados como gate de paridade.

+ **Proposta de Solução:**
>
> 1. **NÃO usar o oráculo como gate** para `wavenet_a2_max.nam` até ser reconciliado.
> 2. **Reconciliar o oráculo ao C++** (Sprint S6, consequência): (a) fazer
>    `oracle_forward` do `condition_dsp` respeitar o `head_size` do último array, produzindo
>    `condition_size` valores/frame; (b) corrigir a leitura de `head1x1` para
>    `out_channels × (bottleneck/groups)`. Re-rodar a âncora NumPy para confirmar ≤ 1e-12.
> 3. Após reconciliado, o oráculo volta a servir à decomposição de erro — **nunca** como gate.

### PM-B — Produção committed (`84c3ad1`) está CORRETA vs C++ na contagem de pesos `head1x1`

+ **ID:** PM-B · **Severidade:** Informativa (estado de verdade) · **Risco:** Nulo
+ **Problema:** Nenhum — é um registro de estado correto. O commit `44a5827` alterou a leitura

> de `head1x1` de `self.channels` para `self.head_accum_size` (= `out_channels`)
> (`build.rs:229` originalmente, mantido em `84c3ad1:build.rs:228`). Esta mudança **alinhou a
> produção ao C++** (`detail.h:76`: contagem = `out_channels × (bottleneck/groups)`).

+ **Evidências:**
>
> + `git show 84c3ad1:src/models/a2/model/dynamic/build.rs` (linhas 227-236): lê
>   `channels = self.head_accum_size; h1_w_count = channels * h1_in` → 12 para cd[0] (= C++).
> + `git show 44a5827 -- src/models/a2/model/dynamic/build.rs`: a mudança `channels →
>   head_accum_size` foi um **fix correto vs C++** (o estado pré-`44a5827` lia `channels=3` → 6,
>   incorreto vs C++).
>
+ **Caveat EM ABERTO (ver PM-D):** a contagem está correta, mas o **layout de transposição**

> (`transpose_dense_f32`) vs o **acesso em runtime** (`head1x1_w[oc*h1_in+ic]`) pode ser
> inconsistente. Só o golden do C++ pode arbitrar isto (nenhum golden ativo exercita
> `head1x1`).

+ **Proposta de Solução:** Preservar a contagem `head_accum_size × h1_in` (correta vs C++).

> Validar o layout/access contra o golden (Sprint S4). **Não reverter para `channels`.**

### PM-C — Achado da Rodada 4 ("regressão 93 dB") é INVÁLIDO e descartado

+ **ID:** PM-C · **Severidade:** N/A (correção de registro) · **Risco:** Nulo
+ **Problema:** A Rodada 4 concluiu por uma "regressão crítica de ESR 93 dB no motor A2 Max"

> (PM-16 histórico) e aplicou mudanças de código (em `build.rs`, `mod.rs`, `process.rs`,
> `dynamic_test.rs`) para "corrigir" a produção rumo ao oráculo. Essa análise foi **fundada em
> premissa falsa**:
>
> 1. Comparou produção (8 saídas/frame) contra oráculo (1 saída/frame) — **dimensões
>    incompatíveis**; o ESR resultante não mede erro de DSP, apenas desalinhamento de shape.
> 2. Usou um oráculo com **contagem de pesos errada** (PM-A) como "verdade".
> 3. Como consequência, as mudanças aplicadas **reverteram o fix correto** do `44a5827`
> (PM-B), fazendo a produção ler `channels` (6) em vez de `head_accum_size` (12) — ou seja,
> **introduziram a divergência vs C++ que pretendiam corrigir**.
>
+ **Evidências:**
>
> + Probe empírico (removido): `cd_oracle.len()=256` vs `cd_prod.len()=2048` — confirma
>   dimensões incompatíveis.
> + C++ `model.cpp:659-660` confirma 8/frame como correto.
>
+ **Proposta de Solução (aplicada na Sprint S1):** **Reverter** todas as mudanças de código

> não-commitadas da Rodada 4 (`build.rs`, `mod.rs`, `process.rs`, `dynamic_test.rs`) ao estado
> do commit `84c3ad1` (que é o baseline correto vs C++). Confirmar com `git diff` que o
> working tree volta a coincidir com `84c3ad1` nestes arquivos. (O `println!` de debug e a
> restauração dos TODO já estão commitados em `84c3ad1` e são mantidos.)

### PM-D — Layout de transposição vs acesso do `head1x1` é questão EM ABERTO (só o golden C++ arbitra)

+ **ID:** PM-D · **Severidade:** Média (potencial bug de layout, não-validado) · **Risco da correção:** Médio
+ **Problema:** O `build.rs` committed aplica `transpose_dense_f32(h1_w_f32, &mut h1_w,

> h1_in, channels)`que armazena pesos no layout **transposto "[in_c][out_c]"** (a função
> escreve`weights[in_c*out_size + out_c] = raw[out_c*in_size + in_c]`). Porém o acesso em
> runtime (`process.rs`) lê`head1x1_w[oc*h1_in + ic]`= layout **"[out_c] [in_c]"** (não
> transposto). Há uma **aparente inconsistência** entre storage e acesso.

+ **Por que não foi detectado:** Nenhum golden **ativo** exercita `head1x1` — apenas

> `wavenet_a2_max.nam` o usa, e seu golden está `#[ignore]`d. Logo, o layout está
> **não-validado** por qualquer barreira existente.

+ **Evidências:**
>
> + `src/models/a2/weights_layout.rs` `transpose_dense_f32`: transpõe `[in][out]`.
> + `src/models/a2/model/dynamic/process.rs` acesso: `head1x1_w[oc*h1_in+ic]` = `[out][in]`.
> + C++ `NAM/wavenet/...` `Conv1x1::set_weights_` / `process_`: define o layout canônico
>   (precisa ser estudado — Sprint S2).
>
+ **Proposta de Solução:**
>
> 1. **Estudar o layout canônico do C++ `Conv1x1`** (Sprint S2): o `set_weights_` lê em qual
>    ordem? O `process_` acessa como `[out][in]` ou `[in][out]`?
> 2. Garantir que o Rust armazene **exatamente** como o C++ armazena (após leitura da stream)
>    e acesse de forma consistente. Se o C++ **não transpõe** e acessa `[out][in]`, remover a
>    `transpose_dense_f32` do head1x1 no Rust. Se o C++ transpõe, ajustar o acesso.
> 3. **Validar contra golden do C++** (Sprint S4): o ESR vs golden é o veredito. Iterar até
>    `ESR < threshold`.

### PM-E — A barreira defensiva real é o golden vs C++; oráculo é decomposição (não gate)

+ **ID:** PM-E · **Severidade:** Alta (processo de qualidade) · **Risco da correção:** Baixo (test infra)
+ **Problema:** O `reference_oracle_f64.rs:146` afirma

> *"Never leave the test_oracle_vs_python_anchor_* tests #[ignore]d without a tracked task."*
> mas trata o oráculo como âncara de correção. Sob a nova premissa, o **oráculo não é gate**:
> ele é uma ferramenta de decomposição. O **gate real** é o golden vs C++
> (`test_golden_vectors_*` + `tests/cpp_parity.rs`). Confundir os dois (como na Rodada 4)
> leva a "corrigir" a produção rumo a um oráculo defeituoso.

+ **Proposta de Solução:**
>
> 1. Documentar claramente em `reference_oracle.rs` (cabeçalho) que o oráculo é
>    **decomposição de erro**, e que divergências oráculo↔produção devem ser arbitral pelo
>    **C++ golden** antes de qualquer mudança na produção.
> 2. O gate de paridade para `wavenet_a2_max.nam` é `test_golden_vectors_wavenet_a2_max`
>    (vs `golden_wavenet_a2_max.bin`). Os testes `test_oracle_*_a2_generic` são
>    **decomposição** e só devem ser reativados **após** PM-A reconciliada.
> 3. Garantir que `utils/tests-quick.sh` e `tests-long.sh` executem o golden vs C++ como gate
>    obrigatório (verificar cobertura atual — Sprint S5).

### PM-F — Quatro testes `#[ignore]`d "S14.2-followup" devem ser reavaliados contra o C++ golden

+ **ID:** PM-F · **Severidade:** Média (barreira neutralizada) · **Risco da correção:** Nulo (após S4)
+ **Problema:** Quatro testes estão `#[ignore]`d com marcador `"S14.2-followup: ..."`:
>
> + `tests/reference_oracle_f64.rs:637` — `test_oracle_a2_generic`
> + `tests/reference_oracle_f64.rs:649` — `test_decomposition_a2_generic`
> + `tests/reference_oracle_f64.rs:676` — `test_combined_simulation_a2_generic`
> + `tests/golden_vectors.rs:1952` — `test_golden_vectors_wavenet_a2_max`
>
+ **Reavaliação:**
>
> + Os **3 do oráculo** (decomposição): só podem ser des-ignorados **após** PM-A
>   reconciliada (S6). Não são gate de paridade.
> + O **golden** (`test_golden_vectors_wavenet_a2_max`): é o **gate real**. Deve ser
>   des-ignorado assim que a produção casar com o C++ (S4/S5). Se o golden ainda divergir,
>   ele **permanece ignorado COM task tracking** (não deve ser removido).
>
+ **Proposta de Solução:**
>
> 1. Sprint S1: des-ignorar **temporariamente** apenas o golden, rodar vs `golden_wavenet_a2_max.bin`,
>    registrar ESR/SNR/MSE. Re-ignorar em seguida (preservando o marcador) até S4.
> 2. Sprint S5: des-ignorar o golden definitivamente (gate) e adicionar teste de regressão
>    dedicado `test_a2_max_vs_cpp_golden` com threshold calibrado.
> 3. Sprint S6: des-ignorar os 3 do oráculo após PM-A.

---

## Épicos (Agrupamentos)

> Ordenados por **valor/risco/sequência**. O Épico N1 é **bloqueante e zero-risco** (reverter
> o erro da Rodada 4). O Épico N2 estabelece a verdade empírica. N3/N4 são o trabalho de
> paridade real. N5 trava a barreira. N6 reconcilia o oráculo (consequência).

### Épico N1 — Reverter o erro da Rodada 4 e restaurar o baseline correto vs C++ (PM-C, PM-B) [BLOQUEANTE, ZERO-RISCO]

+ **Risco/Criticidade:** Nulo (revert a estado já commitado). **Sequência:** imediata.
+ **Entregáveis:** `git checkout 84c3ad1 -- src/models/a2/model/dynamic/{build.rs,mod.rs,process.rs} src/models/a2/model/dynamic_test.rs` (descarta as mudanças PM-16 errôneas). Verificar `git diff --stat` mostra apenas os TODO files (já editados nesta sessão) e o `.agents` (não-relacionado). Confirmar `cargo test --lib` passa (estado pré-Rodada-4 nestes arquivos).

### Épico N2 — Estabelecer a verdade empírica vs C++ golden (PM-F) [ALTO VALOR]

+ **Risco/Criticidade:** Baixo (medição, sem mudança de produção). **Sequência:** após N1.
+ **Entregáveis:** Des-ignorar temporariamente `test_golden_vectors_wavenet_a2_max`, rodar vs `golden_wavenet_a2_max.bin` (2048 samples @ 48 kHz). Registrar ESR/SNR/MSE/MRSTFT. Decisão:
  + **(a) ESR < threshold:** produção está correta vs C++ → pular para N5 (travar gate) e N6 (reconciliar oráculo). PM-D fica hypothesis rejeitada.
  + **(b) ESR ≥ threshold:** produção diverge do C++ → prosseguir N3/N4. PM-D é hipótese ativa.

### Épico N3 — Especificação exata do C++ (Conv1x1, head1x1, cascade, condition_dsp) (PM-D) [ESPECIFICAÇÃO]

+ **Risco/Criticidade:** Nulo (read-only). **Sequência:** após N2-caso-(b).
+ **Entregáveis:** Documento de spec (em `docs/` ou seção do `cpp_parity_map.md`) cobrindo, com `file:line` do C++: (1) `Conv1x1::set_weights_` e `process_` (layout canônico de pesos e ordem de acesso); (2) head1x1 apply (grupos, `ch_per_group`, ordem de acumulação); (3) cascade (`_layer_arrays` loop, head accumulation, `final_head_outputs`); (4) `condition_dsp` process (`_process_condition`, buffers, `NumOutputChannels`); (5) head finalize (head_size==1 conv vs head_size>1 rechannel).

### Épico N4 — Fix da produção ao C++ (cirúrgico, golden como feedback) (PM-D) [PARIDADE]

+ **Risco/Criticidade:** Médio (hot-path DSP). **Sequência:** após N3.
+ **Entregáveis:** Para cada divergência Rust↔spec-C++ (priorizar PM-D: transpose/access do head1x1), aplicar a mudança **mínima** que alinha ao C++. Após **cada** mudança, re-rodar `test_golden_vectors_wavenet_a2_max` e confirmar redução do ESR. Iterar até `ESR < threshold` calibrado. Nunca usar o oráculo como veredito.

### Épico N5 — Travar a barreira defensiva (golden vs C++ como gate) (PM-E, PM-F) [QUALIDADE]

+ **Risco/Criticidade:** Baixo (test-infra). **Sequência:** após N4 (ou N2-caso-(a)).
+ **Entregáveis:** (1) Des-ignorar definitivamente `test_golden_vectors_wavenet_a2_max` (gate). (2) Adicionar `test_a2_max_vs_cpp_golden` com threshold calibrado (separado do oráculo). (3) Confirmar que `tests-quick.sh`/`tests-long.sh` executam este golden como gate obrigatório. (4) Atualizar `docs/cpp_parity_map.md` §6/§13 com o estado empírico real (não mais a narrativa da Rodada 4).

### Épico N6 — Reconciliar o oráculo f64 e a âncora NumPy ao C++ (PM-A) [CONSEQUÊNCIA, DEBUG]

+ **Risco/Criticidade:** Médio (test-only, amplo alcance). **Sequência:** após N4/N5 (baixa prioridade — é ferramenta de debug).
+ **Entregáveis:** (1) `oracle_forward` do `condition_dsp`: respeitar `head_size` do último array → produzir `condition_size` valores/frame (como o C++ `NumOutputChannels`). (2) Leitura de `head1x1`: `out_channels × (bottleneck/groups)` e bias `out_channels`. (3) Re-rodar `validate_oracle_f64.py` → confirmar ≤ 1e-12 contra a âncora corrigida. (4) Des-ignorar os 3 testes `test_*_a2_generic` (decomposição) — agora consistentes com o C++. (5) Documentar no cabeçalho do oráculo que ele é **decomposição**, e o C++ golden é o **gate** (PM-E).

---

> **Princípio de execução:** Cada Sprint produz um **registro empírico** (ESR/SNR medidos vs
> C++ golden) antes e depois de qualquer mudança. O golden do C++ é o único veredito. O oráculo
> só é consultado **após** a produção casar com o C++, para decompor o erro residual — nunca
> para validar a correção.
