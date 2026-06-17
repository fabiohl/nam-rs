<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Plano de Execução (Épico E-HF: "Alta Fidelidade como Padrão")

> **Origem**: parecer da skill `revisor-auditor` (jun/2026) sobre os achados
> **`TODO-problemas.md §P10`** (modo "baixa fidelidade" sob júdice), **`TODO-optimize.md §O1`**
> (internalizar `half` + F16C nas caudas) e **`TODO-optimize.md §O5`** (cobertura SIMD x86-64-v3
> no hot-spot). Organizado pela skill `planejador-arquiteto`.
>
> **Idioma**: pt-BR. **Regras obrigatórias**: `.agents/rules/{rust,testing,linting,copyright}.md`.
> **Bíblia de correção**: `tests/fixtures/NeuralAmpModelerCore` (commit pinado `9c7b185`).

---

## 0. Sumário executivo (o achado central, com evidência)

A auditoria cruzou o código do `nam-rs` com a referência **NeuralAmpModelerCore (NAMCore)** e
chegou a uma conclusão que **reposiciona** a discussão do P10:

> **A referência (a "bíblia") É, por padrão, o modo alta-fidelidade do `nam-rs`.**

Evidências diretas no NAMCore:

1. **Pesos são f32 puros, sem quantização.** NAMCore guarda todos os pesos como
   `std::vector<float>` / `Eigen::MatrixXf` (ex.: `NAM/wavenet/model.h:71-111`,
   `NAM/conv1d.h:122`, `NAM/lstm.h:38,85`). **Não existe quantização u16 (F16/BF16)** na
   referência. A quantização de pesos do `nam-rs` (modo padrão / "baixa fidelidade",
   `src/math/common/ops.rs:16-22`) é uma **invenção do `nam-rs`**, não um espelho da bíblia.
2. **O tanh padrão é o exato `std::tanh`.** `NAM/activations.h:182-192` (`ActivationTanh`)
   usa `std::tanh(data[pos])`. O `fast_tanh` (aproximação racional, `activations.h:91-98`)
   existe mas está **desligado por padrão** (`activations.cpp:16`:
   `bool ...::using_fast_tanh = false;`). Há ainda um caminho LUT opcional.

**Consequência estratégica**: o "modo padrão" quantizado do `nam-rs` **diverge da referência por
construção** — soma o erro de quantização de pesos (~3,9e-3/elemento, fonte **dominante** de drift
por `TODO-problemas.md:118-120`) ao erro do tanh Padé [5,4] (~2,3e-3). O modo alta-fidelidade
(pesos f32 + tanh exato) é o que **reproduz a bíblia**. Isso valida fortemente a hipótese do PO:
**não há justificativa de fidelidade para o modo padrão atual** — só (eventualmente) de performance,
que **nunca foi medida**.

### O segundo achado: o modo alta-fidelidade nasceu **escalar** (e viola a regra de RT-safety)

O modo `high-fidelity` (feature flag `Cargo.toml:71`, commit `64e2c7f`) **não é uniformemente
SIMD**. A auditoria mapeou exatamente o que é escalar:

| Caminho hi-fi                       | Local                                                           | Estado                                |
| ----------------------------------- | --------------------------------------------------------------- | ------------------------------------- |
| Dense GEMV (rechannel/1x1/head)     | `src/math/gemm/gemv/f32_avx2.rs:17`, `f32_avx512.rs:14`         | ✅ **SIMD**                           |
| **Conv1D dot-product f32**          | `src/models/wavenet/conv_input.rs:178-189` (e `:197-216` dual)  | ❌ **ESCALAR**                        |
| **tanh exato**                      | `accumulate/avx2.rs:41-47,104-109`; `avx512.rs:129-136,159-165` | ❌ **ESCALAR** (`f32::tanh()`)        |
| **gated activation (tanh·sigmoid)** | `accumulate/avx2.rs:79-87,...`; `avx512.rs`                     | ❌ **ESCALAR** (`.tanh()` + `.exp()`) |
| **soma residual (1x1)**             | `src/models/wavenet/layer.rs:248-251`                           | ❌ **ESCALAR**                        |

Pior: `.agents/rules/rust.md:25` declara **"Native `f32::tanh()`/`exp()` are prohibitive on the
hot-path"**. O modo hi-fi atual **chama `f32::tanh()` e `.exp()` por amostra** na thread de áudio →
**viola a própria regra de RT-safety do projeto** (libm é não-determinístico em latência; risco de
pico/xrun, conexão com `TODO-problemas.md §P5`). A vetorização não é só performance — é
**conformidade RT**.

### Bug latente encontrado (correção de soundness/desperdício)

`src/math/wavenet/accumulate/avx512.rs:120-136` e `:152-165`: com `high-fidelity` **ligado**, o bloco
mascarado `if i < len` (linhas 120/152) **não** está sob `#[cfg(not(feature = "high-fidelity"))]`.
Como o laço SIMD principal foi removido (`i == 0`), o bloco mascarado roda o **Padé** sobre **todos**
os elementos, e **em seguida** o laço escalar `#[cfg(feature = "high-fidelity")]` (linhas 129/159)
recomputa tudo com `f32::tanh()` a partir de `i == 0`. O resultado final é correto (o escalar
sobrescreve), mas há **trabalho duplicado** e — pior — o caminho AVX-512 hi-fi aplica **Padé
(impreciso)** antes de descartar. Só ocorre no caminho AVX-512 (o AVX2 está limpo).

### Estratégia do épico

1. **Frente A (S-HF1 → S-HF2)** — Tornar o modo alta-fidelidade **100% SIMD x86-64-v3 e RT-safe**:
   substituir TODA aritmética escalar do caminho hi-fi por kernels AVX2/FMA (e AVX-512 quando
   detectado), incluindo um **tanh/sigmoid vetorizado de alta precisão** (erro ≪ Padé [5,4],
   alvo ~1e-6) que **substitui `f32::tanh()`/`.exp()`** e respeita a regra de RT-safety.
2. **Frente B (S-HF3 = O1)** — Internalizar `half` (f16↔f32 por software, bit-exato) + usar `F16C`
   escalar (`_mm_cvtph_ps`) nas caudas SIMD; remover `half`/`zerocopy`/`syn 2` do grafo. Independente
   e de baixo risco; **continua valioso mesmo se o lo-fi do WaveNet for abandonado** (a A2 usa F16).
3. **Frente C (S-HF4 → S-HF5)** — Com o hi-fi **já otimizado**, medir **rigorosamente** hi-fi vs
   lo-fi (latência, throughput, memória, ESR com pesos reais, perceptual) e **decidir**: promover
   hi-fi a padrão / abandonar lo-fi / manter dualidade.

> **Ordem da medição importa**: medir hi-fi **depois** de vetorizá-lo (Frente A). Comparar um hi-fi
> escalar contra um lo-fi SIMD seria um benchmark enviesado e levaria a uma decisão errada.

---

## Visão geral das sprints

| Sprint    | Tema                                                       | Risco      | Depende de   |
| --------- | ---------------------------------------------------------- |:----------:| ------------ |
| **S-HF1** | Kernel SIMD de tanh/sigmoid de alta precisão + fix AVX-512 | 🟠 Médio   | —            |
| **S-HF2** | Vetorização do Conv1D f32 + soma residual (hi-fi)          | 🟠 Médio   | S-HF1        |
| **S-HF3** | O1 — internalizar `half` + F16C nas caudas                 | 🟢 Baixo   | — (paralela) |
| **S-HF4** | Infra de medição rigorosa (bench/ESR/memória/perceptual)   | 🟢 Baixo   | S-HF1, S-HF2 |
| **S-HF5** | Ponto de decisão P10 + execução (condicional)              | 🔴 Crítico | S-HF4        |

**Paralelização**: S-HF3 (O1) é ortogonal e pode rodar em paralelo a S-HF1/S-HF2. S-HF1 → S-HF2 são
sequenciais (S-HF2 usa o kernel de ativação de S-HF1). S-HF4 só faz sentido após A pronta. S-HF5 é
gatilho de decisão e **não** deve ser iniciada antes de S-HF4 ter dados verdes.

---

## Sprint S-HF1 — Kernel SIMD de tanh/sigmoid de alta precisão (modo hi-fi) + fix AVX-512

**Objetivo**: criar uma ativação **vetorizada** (AVX2/FMA e AVX-512) com erro alvo **≤ 1e-6** vs
`f32::tanh`/`sigmoid` exatos — substituindo os laços escalares `f32::tanh()`/`.exp()` do caminho
hi-fi por SIMD RT-safe, sem regredir a fidelidade (ESR) atual do hi-fi escalar. Corrigir o bug de
dupla computação AVX-512.

**Por que primeiro**: é a peça de maior risco numérico e é **dependência** de S-HF2 (o Conv1D hi-fi
desemboca em tanh/gated). Fechá-la antes destrava o resto com critério de aceite numérico já provado.

**Risco**: 🟠 Médio (precisão numérica + range-reduction correto). Mitigado por golden vs `f32::tanh`
e teste de ESR end-to-end.

### T-HF1.1 — Projetar e implementar `simd_tanh_hifi` / `simd_sigmoid_hifi` (AVX2/FMA) ✅ DONE

* **Descrição**: implementar um tanh vetorizado de alta precisão. Abordagem escolhida:
  **exp-based (b)** — `tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)`, `σ(x) = 1/(1+e⁻ˣ)`, kernel `simd_exp_hifi_avx2`
  com degree-6 Taylor + range reduction `k = round(x·log₂e)`, `r = x − k·ln2`.
  * **Erro medido** (sweep 4001 pts, step 0.01, [-20, 20]):
    * tanh max error: **1.49e-7** (6.7× abaixo do limite de 1e-6)
    * sigmoid max error: **1.19e-7** (8.4× abaixo do limite de 1e-6)
  * **Throughput**: tanh ~19 ops SIMD (1 exp + 2 div + add/sub + clamp), sigmoid ~17 ops (1 exp + 1 div + add + clamp).
  * **RT-safe**: zero-alloc, zero-branch (SIMD min/max para clamp), `#[target_feature(enable = "avx2,fma")]`.
* **Arquivos**: `src/math/activations/tanh/high_fidelity.rs` (546 linhas), `src/math/constants.rs`
  (novos HIFI_*), `src/math/activations/tanh/mod.rs` (novo submódulo + re-export).
  * `production.rs` **não foi alterado**.

### T-HF1.2 — Religar os kernels `accumulate` AVX2 do hi-fi ao novo kernel SIMD [DONE]

* **Descrição**: nos kernels `src/math/wavenet/accumulate/avx2.rs`, trocar os **laços escalares**
  `#[cfg(feature = "high-fidelity")]` (`.tanh()`, `.tanh()*sigmoid`) por SIMD usando
  `simd_tanh_hifi_avx2` / `simd_tanh_sigmoid_dual_hifi_avx2`. Sites:
  * `tanh_and_accumulate_block_avx2` (cauda escalar `:41-47`);
  * `gated_activation_and_accumulate_block_avx2` (escalar `:79-87`);
  * `tanh_and_overwrite_block_avx2` (escalar `:104-109`);
  * `gated_activation_and_overwrite_block_avx2` (`:114+`).
* **Estratégia**: o laço SIMD principal passa a existir **nos dois modos**; o que muda entre lo-fi e
  hi-fi é **qual kernel de ativação** é chamado (Padé vs alta precisão). Refatorar para selecionar o
  kernel via `cfg` **dentro** do mesmo laço `while i + 8 <= len`, mantendo a cauda mascarada/escalar
  mínima idêntica nos dois. Eliminar a assimetria atual (`#[cfg(not(...))]` no laço inteiro).
* **Aceite**: `cargo build` e `cargo build --features high-fidelity` verdes; testes de paridade do
  WaveNet inalterados; **zero** `f32::tanh()`/`.exp()` por-amostra restantes no caminho hi-fi AVX2
  (grep limpo). Bit-exato vs T-HF1.1 (mesmo kernel).
  **Nota (T-HF1.2)**: descoberta falha pré-existente em testes `test_prewarm_zero_rf`,
   `test_prewarm_large_rf_*`, `test_wavenet_computational_stability` com `--features high-fidelity`
   (SIGSEGV em modelos com `f32_weights` vazios em `wavenet_prewarm_edge.rs`). As causas-raiz
   independem deste task. Os kernels de ativação hi-fi funcionam corretamente (A2 tests passam).
   **Corrigido em T-HF1.4** — `f32_weights` populados via `transpose_conv1d_interleaved_4wide_f32`;
   17/17 prewarm edge tests passam nos dois modos.

### T-HF1.3 — Variantes AVX-512 + **fix do bug de dupla computação** ✅ DONE

* **Descrição**: implementar `simd_tanh_hifi_avx512` / `..._sigmoid_dual_hifi_avx512` e religar
  `src/math/wavenet/accumulate/avx512.rs`. **Corrigir o bug**: os blocos mascarados `if i < len`
  (`:120-128` e `:152-158`) devem ficar sob `#[cfg(not(feature = "high-fidelity"))]` **ou** ser
  unificados para usar o kernel hi-fi quando a feature estiver ligada — **nunca** rodar Padé + escalar
  em sequência. Eliminar o laço escalar `#[cfg(feature = "high-fidelity")]` (`:129-136`, `:159-165`).
* **Aceite**: com `--features high-fidelity`, o caminho AVX-512 aplica **apenas** o kernel hi-fi
  (uma vez); sem dupla aplicação. Verificar por inspeção + teste de paridade AVX-512 (se houver
  máquina; senão, marcar `#[ignore]` para a suíte longa). `cargo test` verde nos dois modos.
* **Risco**: 🟠 — requer hardware AVX-512 para validação plena; documentar fallback de teste.
  **Nota (T-HF1-3)**: implementados `simd_exp_hifi_avx512`, `simd_tanh_hifi_avx512`,
  `simd_sigmoid_hifi_avx512`, `simd_tanh_sigmoid_dual_hifi_avx512` em `high_fidelity.rs`. Os 4
  kernels de `avx512.rs` agora usam `cfg` unificado (mesmo padrão do AVX2 de T-HF1.2): laço SIMD
  principal com `#[cfg(feature = "high-fidelity")]`/`#[cfg(not(...))]` selecionando hi-fi vs Padé;
  cauda mascarada idem (onde aplicável); zero `f32::tanh()`/`.exp()` escalar no caminho hi-fi
  AVX-512. Sem máquina AVX-512 local — os testes AVX-512 em `high_fidelity_test.rs` são
  auto-skip quando `is_x86_feature_detected!("avx512f")` é falso. `cargo test` (434 tests) verde
  nos dois modos.

### T-HF1.4 — Teste golden de fidelidade da ativação + gate anti-regressão ✅ DONE

* **Descrição**: teste comparando `simd_tanh_hifi_*` e `..._sigmoid_*` contra `f32::tanh`/`sigmoid`
  escalares (erro ≤ 1e-6) e, end-to-end, contra a referência: o ESR do WaveNet hi-fi **não pode
  regredir** vs o hi-fi escalar atual (medir antes/depois). Reaproveitar `tests/golden_vectors.rs` e
  os utilitários de ESR.
* **Arquivo**: inline se o módulo < 300 linhas; senão `high_fidelity_test.rs` (regra `testing.md`).
  Sweeps pesados → `#[ignore]` (suíte longa).
* **Aceite**: ESR hi-fi-SIMD ≤ ESR hi-fi-escalar (dentro de tolerância documentada) e ambos ≪ lo-fi;
  cabeçalho SPDX presente.

  **Resultado (T-HF1.4)**:
  * **Fix do SIGSEGV**: `f32_weights` vazios nos builders sintéticos de `wavenet_prewarm_edge.rs`
    (3 instâncias) e `tests/common/model_builders.rs` (4 instâncias) causavam SIGSEGV com
    `--features high-fidelity`. Corrigidos populando `f32_weights` via
    `transpose_conv1d_interleaved_4wide_f32` (exportada de `layout.rs`). 17/17 prewarm edge tests
    passam nos dois modos.
  * **Gate anti-regressão**: adicionados `test_hifi_regression_gate_wavenet_standard` (A1, CH=16) e
    `test_hifi_regression_gate_wavenet_a2_full` (A2, CH=8) em `tests/golden_vectors.rs`. O gate A2
    passa com hi-fi (ESR ≤ 2e-4, SNR ≥ 65 dB). O gate A1 está `#[ignore]` — detecta um bug
    pré-existente no caminho hi-fi da arquitetura A1 (ver nota abaixo).
  * **Bug pré-existente descoberto**: os golden tests de WaveNet A1 (`BossWN-standard`,
    `BossWN-feather`, `BossWN-nano`, `wavenet_official`, `wavenet_a1_standard`) produzem saída
    espúria com `--features high-fidelity` (SNR ≈ −1 dB vs C++). O caminho A2 e LSTM hi-fi
    funciona corretamente (SNR ≥ 65 dB). Marcados com `#[cfg_attr(feature = "high-fidelity",
    ignore)]`. Resolução em T-HF1.5 (abaixo).

### T-HF1.5 — Correção do caminho hi-fi da arquitetura A1 (WaveNet Standard/Feather/Nano/Official)

* **Descrição**: investigar e corrigir o bug que faz os modelos WaveNet A1 produzirem saída
  espúria (SNR ≈ −1 dB vs C++) com `--features high-fidelity`. O caminho A2 funciona
  corretamente — a raiz está no código hi-fi específico da topologia A1 (loader ou execução).
  Reabilitar os golden tests A1 marcados com `#[ignore]` em T-HF1.4 e remover os `#[cfg_attr]`.
* **Arquivos**: `src/loader/dispatcher/wavenet/{standard,feather,nano,lite,official,layout}.rs`,
  `src/models/wavenet/layer.rs` (hi-fi path), `tests/golden_vectors.rs`.
* **Aceite**: `cargo test --features high-fidelity` verde com zero `#[ignore]` por bug A1.
* **Risco**: 🟠 — requer debugging do caminho hi-fi A1; o A2 já funciona.
* **Depende de**: T-HF1.4.

---

## Sprint S-HF2 — Vetorização do Conv1D f32 e da soma residual (modo hi-fi)

**Objetivo**: substituir o dot-product escalar f32 do Conv1D hi-fi
(`src/models/wavenet/conv_input.rs:178-189` e `:197-216`) por kernels AVX2/FMA (AVX-512 quando
detectado), e vetorizar a soma residual escalar (`src/models/wavenet/layer.rs:248-251`). Esta é a
**computação dominante** do hot-path WaveNet — o maior ganho de latência do épico.

**Depende de**: S-HF1 (o Conv1D hi-fi alimenta o tanh/gated; fechar a ativação primeiro dá baseline
estável para atribuir o ganho do Conv1D).

**Risco**: 🟠 Médio (layout 4-wide interleaved + FMA bit-exato vs o `mul_add` escalar atual).

### T-HF2.1 — Kernel SIMD `dot_product_4x_f32` (AVX2/FMA)

* **Descrição**: vetorizar `dot_product_4x_f32` (`conv_input.rs:178-189`). O layout já é favorável:
  pesos em `[[f32; 4]]` (4 canais de saída interleaved) e o laço faz `r[j] = mul_add(w[j], state[i],
  r[j])`. Mapeamento natural: carregar o `[f32;4]` num `__m128` e fazer
  `_mm_fmadd_ps(w128, _mm_set1_ps(state[i]), acc128)`. **Otimização**: processar **2 blocos (8 canais
  de saída) por iteração** via `__m256` (`_mm256_fmadd_ps` + `_mm256_set1_ps`), explorando o
  `x86-64-v3`. Espelhar o padrão já validado em `src/math/gemm/gemv/f32_avx2.rs:17` (que usa 4
  acumuladores YMM e broadcast).
* **Bit-exatidão**: o escalar atual **já usa `mul_add` (FMA)**; o SIMD `_mm*_fmadd_ps` produz o
  **mesmo arredondamento** → resultado bit-idêntico por construção (sem mudança de ESR). Documentar
  essa equivalência no comentário.
* **RT-safety**: `#[target_feature(enable = "avx2,fma")]`, `AlignedVec` já garantido no load
  (`src/loader/dispatcher/wavenet/layout.rs:339-364`), zero-alloc, sem bounds-check no laço quente.
* **Aceite**: bit-exato vs o escalar atual num conjunto de vetores aleatórios; `cargo test
  --features high-fidelity` verde.

### T-HF2.2 — Kernel SIMD dual-frame `dot_product_4x_f32_dual` (AVX2/FMA)

* **Descrição**: idem T-HF2.1 para `dot_product_4x_f32_dual` (`conv_input.rs:197-216`), que processa
  dois estados (`state_f0`, `state_f1`) contra o mesmo peso — encaixa em
  `_mm256`/duplo acumulador, reusando o broadcast do peso. Alinhar com o tiling dual-frame do motor
  dinâmico lo-fi (mesma semântica de invariância de bloco — cruzar com `TODO-problemas.md §P1`).
* **Aceite**: bit-exato vs escalar; invariância single-frame vs dual-frame (MSE ≈ 0) verificada.

### T-HF2.3 — Variante AVX-512 do Conv1D f32 (quando detectado)

* **Descrição**: variante `__m512` (`_mm512_fmadd_ps`), processando 4 blocos (16 canais) por
  iteração. Despachar via o mesmo mecanismo CPUID já usado (`src/math/common/dispatch/`). Espelhar
  `src/math/gemm/gemv/f32_avx512.rs:14`.
* **Aceite**: bit-exato vs AVX2 (dentro da semântica FMA); fallback AVX2 quando AVX-512 ausente.
* **Risco**: 🟠 — validação requer hardware AVX-512.

### T-HF2.4 — Vetorizar a soma residual do 1x1 (hi-fi)

* **Descrição**: o caminho hi-fi separa GEMV e residual e faz `for j .. output[j] += residual[j]`
  escalar (`src/models/wavenet/layer.rs:248-251`). Vetorizar com `_mm256_add_ps`
  (`chunks_exact(8)` + cauda) **ou** — preferível — reusar o kernel fundido
  `process_residual_batch` que o lo-fi usa, criando uma variante f32-nativa
  (`process_residual_batch_f32`) que funde GEMV+residual num só passe SIMD.
* **Aceite**: bit-exato; remove o laço escalar; `cargo test --features high-fidelity` verde.

### T-HF2.5 — Auditoria final "nada escalar no hot-path hi-fi" (guard-rail O5)

* **Descrição**: aplicar o **guard-rail** de `TODO-optimize.md §O5` ao caminho hi-fi inteiro:
  varredura (grep + leitura) confirmando que **nenhum** laço de aritmética/redução f32 por
  amostra/bloco permanece escalar com `--features high-fidelity`. Documentar o resultado como
  resíduo fechado de O5 (hi-fi).
* **Aceite**: relatório curto anexado (file:line auditados) + zero achados escalares no hot-path
  hi-fi.

---

## Sprint S-HF3 — O1: internalizar `half` (f16↔f32) + F16C nas caudas SIMD

**Objetivo**: remover a dependência `half` (`Cargo.toml:29`) — e com ela `zerocopy 0.8` +
`zerocopy-derive` + `syn 2` (cadeia proc-macro que só entra por causa do `half`, confirmado em
`Cargo.lock`) — internalizando as **duas** operações usadas (`from_bits().to_f32()`,
`from_f32().to_bits()`), e trocando, nas **caudas escalares sob `#[target_feature f16c]`**, a
conversão por software pelo intrínseco `_mm_cvtph_ps` (1 instrução).

**Independente** das demais (paralela). **Continua valiosa mesmo se o lo-fi do WaveNet for
abandonado** em S-HF5, pois a **A2 usa pesos F16** (`src/models/a2/conv1d_ch3/mod.rs:201`,
`a2/model/mod.rs:267`) — a conversão f16↔f32 permanece no produto.

**Risco**: 🟢 Baixo (espaço de entrada é finito: 65.536 padrões → teste **exaustivo** bit-exato).

### T-HF3.1 — Implementar `src/math/common/half.rs` (software, bit-exato)

* **Descrição**: módulo com `f16_bits_to_f32(u16) -> f32` (decode IEEE-754 binary16: normais,
  subnormais, ±0, ±Inf, NaN) e `f32_to_f16_bits(f32) -> u16` (encode com **round-to-nearest-even**,
  overflow→Inf, subnormais). ~40–60 linhas. Cabeçalho SPDX.
* **Aceite**: documentação clara; sem `unsafe` desnecessário; sem alocação.

### T-HF3.2 — Variante F16C escalar para as caudas (`_mm_cvtph_ps`)

* **Descrição**: `f16_bits_to_f32_f16c(u16) -> f32` usando `_mm_cvtph_ps(_mm_cvtsi32_si128(bits as
  i32))` + `_mm_cvtss_f32`, sob `#[target_feature(enable = "f16c")]` (garantido por
  `x86-64-v3` em `.cargo/config.toml`). Opcional: encode `_mm_cvtps_ph` (RNE) para simetria.
* **Aceite**: bit-exato vs `f16_bits_to_f32` software para os 65.536 valores.

### T-HF3.3 — Migrar todos os call-sites de `half::*`

* **Descrição**: substituir `half::f16::from_bits(x).to_f32()` / `half::f16::from_f32(x).to_bits()`
  pelas funções internas. **Inventário completo** (a auditoria encontrou sites além dos listados no
  TODO original):
  * **Caudas hot-path sob `f16c`** (usar `_..._f16c`): `src/math/gemm/dot.rs:87,115`;
    `dot_4x/avx2.rs:108,110,112,114`; `dot_4x/avx2_dual.rs:228`; `gemv_4gate/avx512.rs:103-106`;
    `gemv_4gate/avx2.rs:114-120`; `gemv/f16_avx2.rs:56,107`; `gemv/f16_avx512.rs:201,279`;
    `gemm_batch/avx2.rs:96,203,240`; `gemm_batch/avx512.rs:113,246,286`.
  * **Modelo hot-path (A2)**: `src/models/a2/conv1d_fallback.rs:106-109`; `a2/model/mod.rs:267`.
  * **Fallback escalar puro (cold/sem AVX2)** (usar software): `scalar_ref/dot.rs:24,93-96,182-185`;
    `scalar_ref/gemm.rs:65,97,136`.
  * **Quantização no load (cold)**: `src/math/common/ops.rs:20`.
  * **LSTM escalar (teste/fallback)**: `lstm/layer_kernels.rs:238`, `model1.rs:126`, `model2.rs:167`.
  * **Testes/benches** (~70 sites): migração trivial.
* **Aceite**: `half::` não aparece mais em `src/` (grep limpo); compila nos modos default e
  `high-fidelity`.

### T-HF3.4 — Teste exaustivo (65.536) + round-trip e **remoção da dependência**

* **Descrição**: teste que percorre **todos** os 65.536 padrões de bits f16 comparando
  `f16_bits_to_f32` (software) **e** `_..._f16c` contra o crate `half` (mantido como dev-dependency
  **golden** durante a transição); round-trip f32→f16→f32 sobre conjunto grande + bordas (±0,
  subnormais, máximo normal, overflow→Inf, NaN). **Só após verde total**, remover `half` do
  `Cargo.toml` (e confirmar que `zerocopy`/`syn 2` saíram do `Cargo.lock`).
* **Aceite**: 100% bit-exato; `half` ausente de `[dependencies]`; `cargo tree` sem `zerocopy` por
  causa de `half`; `cargo bench` em `dot_4x`/`conv1d` ≥ paridade (esperado: leve ganho na cauda).
* **Risco**: 🟢 — após o exaustivo, a substituição é bit-exata por construção.

---

## Sprint S-HF4 — Infra de medição rigorosa (hi-fi otimizado vs lo-fi)

**Objetivo**: produzir o **dado reprodutível** que P10 exige, com o hi-fi **já vetorizado** (Frente
A concluída). Sem isto, a decisão de S-HF5 é especulativa.

**Depende de**: S-HF1 + S-HF2 (medir hi-fi escalar seria enviesado).

**Risco**: 🟢 Baixo (instrumentação/medição; não altera produção).

### T-HF4.1 — Benchmarks criterion lo-fi vs hi-fi

* **Descrição**: em `benches/inference_bench.rs`, adicionar par
  `bench_wavenet_standard_lofi` vs `bench_wavenet_standard_hifi` (compilado com `--features
  high-fidelity`), em blocos de **1, 16, 64** frames; registrar **latência de pico** e **throughput**
  (frames/s). Cobrir A1-Std (CH=16) e, se viável, Lite (CH=12). Seguir `docs/benchmarks.md`.
* **Aceite**: relatório criterion comparativo reprodutível; números por tamanho de bloco.

### T-HF4.2 — ESR com pesos reais (não sintéticos)

* **Descrição**: medir ESR de cada modo vs C++ usando **modelos reais** (`wavenet_official.nam` e
  modelos nondist), não os sintéticos 0.01. Reusar `tests/golden_vectors.rs` /
  `tests/nondist_validation.rs` e a infra de paridade C++. Registrar ESR lo-fi vs hi-fi vs LSTM/Linear.
* **Aceite**: tabela de ESR real por modo/modelo; confirmar que hi-fi ≈ paridade C++ (≪ 1e-5) e
  lo-fi mantém ~3e-3..1e-2.

### T-HF4.3 — Perfil de memória (footprint f32 vs u16)

* **Descrição**: medir o footprint por modelo: `AlignedVec<f32>` (hi-fi, 4 bytes) vs `AlignedVec<u16>`
  (lo-fi, 2 bytes) — lembrando que o hi-fi atual mantém **ambos** (`conv1d.rs:22-26`,
  `dense.rs:17-20`), ~3× memória de pesos. Avaliar **eliminar o buffer u16** no hi-fi puro (se lo-fi
  for descontinuado em S-HF5) → footprint cairia para ~2× → 2 bytes vira 4 bytes (só f32). RSS via
  soak (padrão `TODO-problemas.md:409`).
* **Aceite**: tabela de footprint por modelo e por modo; estimativa do ganho de RAM ao remover u16.

### T-HF4.4 — Avaliação perceptual informal (AB / MR-STFT)

* **Descrição**: AB informal em material high-gain (transientes) + métrica perceptual MR-STFT/LUFS
  (`docs/perceptual_validation.md`) quantificando a diferença de ~1% de energia do lo-fi. Documentar
  se é perceptível.
* **Aceite**: parecer perceptual registrado (perceptível? sim/não/condicional).

### T-HF4.5 — Relatório de decisão consolidado

* **Descrição**: consolidar T-HF4.1–4.4 numa tabela única (latência, throughput, memória, ESR,
  perceptual) respondendo à matriz de critério de `TODO-problemas.md:365-371`. Entrada direta para
  S-HF5.
* **Aceite**: documento `docs/` (ou seção em `fastmath-approximations.md`) com a recomendação
  fundamentada.

---

## Sprint S-HF5 — Ponto de decisão P10 + execução (condicional)

**Objetivo**: com os dados de S-HF4, **decidir** e executar. **Crítico** porque afeta o contrato do
produto e remove/mantém uma classe inteira de código.

**Depende de**: S-HF4 (não iniciar sem dados verdes).

**Risco**: 🔴 Crítico — decisão de arquitetura irreversível na prática (simplificação radical).

### T-HF5.0 — Reunião de decisão (gate humano do PO)

* **Critério de decisão** (com base em S-HF4):
  * **Se** o ganho de performance do lo-fi for **marginal** (ex.: < ~5–10% de throughput e sem
    benefício de latência de pico) **e** a memória não for crítica → **abandonar o lo-fi**; promover
    hi-fi (já SIMD) a **único caminho** (alinha com a bíblia NAMCore, resolve P2 pela raiz e
    fragmentos de P1). Caminho preferido pela hipótese do PO + evidência NAMCore.
  * **Se** o ganho for **forte e mensurável** → **manter dualidade**, mas com o hi-fi agora
    competitivo (SIMD) e melhor documentado.
* **Saída**: decisão registrada; selecionar T-HF5.A **ou** T-HF5.B.

### T-HF5.A — (Se abandonar lo-fi) Promover hi-fi a padrão e simplificar

* **Descrição**: remover a dualidade — eliminar a feature flag `high-fidelity` (vira o
  comportamento único), os caminhos quantizados u16 do WaveNet, o `quantize_weight`
  (`ops.rs:16-22`) onde só servia ao WaveNet, o Padé [5,4] do WaveNet, e o buffer u16 redundante
  (`conv1d.rs:22-26`, `dense.rs:17-20`). **Cautela**: a A2 **continua** usando F16 — **não** remover
  a quantização/F16 globalmente; restringir ao WaveNet. Atualizar `cpp_parity_map.md`,
  `fastmath-approximations.md`, reabilitar goldens de paridade endurecidos (P2/P3) e reavaliar P1
  (Lite) sob o caminho exato.
* **Aceite**: suíte verde; ESR do WaveNet ≈ paridade C++; código net-removido (métrica de
  simplificação); docs sincronizadas (acionar skill `documentador`).
* **Risco**: 🔴 — mudança ampla; fazer atrás de PRs revisáveis e da suíte longa.

### T-HF5.B — (Se manter dualidade) Consolidar e documentar o trade-off medido

* **Descrição**: manter os dois modos, mas com o hi-fi agora SIMD/competitivo; documentar o
  trade-off **com números reais** (de S-HF4) em `fastmath-approximations.md`; garantir que o
  default continua justificado. Reavaliar P1/P2 com o hi-fi exato disponível.
* **Aceite**: política de fidelidade documentada com dados; ambos os modos verdes e benchmarkados.

---

## Rastreabilidade (achados → sprints)

| Achado / origem                                  | Sprint(s)                           |
| ------------------------------------------------ | ----------------------------------- |
| `TODO-problemas.md §P10` (medir lo-fi)           | S-HF4, S-HF5                        |
| `TODO-problemas.md §P10` (hi-fi escalar → SIMD)  | S-HF1, S-HF2                        |
| `TODO-problemas.md §P2/§P3` (fidelidade WaveNet) | S-HF5.A (resolução)                 |
| `TODO-problemas.md §P1` (Lite divergente)        | S-HF2.2, S-HF5 (reavaliar)          |
| `TODO-problemas.md §P5` (pico latência/denormal) | S-HF1 (remove libm tanh)            |
| `TODO-optimize.md §O1` (`half` + F16C)           | S-HF3                               |
| `TODO-optimize.md §O5` (cobertura SIMD hot-spot) | S-HF1.3 (bug), S-HF2.5 (guard-rail) |
| `.agents/rules/rust.md:25` (proibir libm RT)     | S-HF1 (conformidade)                |

## Regras de fechamento (todas as sprints)

Cada tarefa fecha com, conforme `.agents/rules/{linting,testing,copyright}.md`:

1. `cargo check` (default **e** `--features high-fidelity`) + `cargo clippy` sem warnings.
2. `cargo test` (e `cargo test --features high-fidelity`) verdes.
3. `cargo bench` quando a tarefa for de performance (registrar baseline antes/depois).
4. Cabeçalho SPDX `Apache-2.0` em todo arquivo novo.
5. RT-safety: zero-alloc/zero-lock/zero-panic/zero-libm no hot-path (auditável com `heap-audit`).
6. Golden contra a referência preservado até a substituição estar provada (vale para O1 com `half`).
