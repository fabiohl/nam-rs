<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-sprints — Plano de Execução (revisor-auditor → planejador-arquiteto)

> **Origem**: rodada de auditoria geral (skill `revisor-auditor`, jun/2026) focada nos achados
> **P1** e **P9** (`TODO-problemas.md`) com aproveitamento transversal de **O1** e **O5**
> (`TODO-optimize.md`). Esta rodada **isolou a causa raiz de P1** (bug de indexação de ring-buffer,
> não arquitetural) e **confirmou o UB de P9**, além de verificar que **O1 já está concluído** e
> mapear o resíduo de **O5 (S2)**.
>
> **Bíblia de correção**: [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
> (`tests/fixtures/NeuralAmpModelerCore/`). Nenhum comportamento numérico do produto pode divergir
> da referência C++ após as correções.
>
> **Regras obrigatórias** (`.agents/rules/`): RT-safety (zero heap-drop / zero-lock / zero-panic no
> hot-path; alocação só no load), copyright SPDX em todo arquivo novo, testes (inline <300 linhas /
> `_test.rs` ≥300 / integração em `tests/` / soak `#[ignore]` na `tests-long.sh`), e linting verde
> (`cargo check` + `cargo test` + `cargo bench` quando perf), sem warnings.

---

## Sumário de Épicos e Sprints

| Épico         | Tema                                                       | Sprints      | Severidade / Risco                 |
| ------------- | ---------------------------------------------------------- | ------------ | ---------------------------------- |
| **E-SND-01**  | Soundness & Fidelidade — correções de raiz (P1 + P9)       | S1, S2       | 🔴 **Crítico** (soundness/UB)      |
| **E-OPT-02**  | Otimização & higiene — fechamento O1 + resíduo SIMD O5/S2  | S3           | 🟢 Baixo (perf/limpeza)            |

**Ordem recomendada de execução**: **S1** (P1, maior impacto de produto) → **S2** (P9, UB latente
antes de publicar suporte a blocos grandes) → **S3** (O5/S2 + fechamento O1, baixo risco).

> **⚠️ Itens de máxima atenção (auditor):** **S1.T1.2** (mexe em memória virtual mmap/espelho —
> errar quebra TODA a inferência WaveNet) e **S2.T2.1** (re-chunk no hot-path da A2 — não pode
> introduzir alocação nem regressão de paridade). Ambos exigem golden vs C++ **antes e depois**.

---

## Épico E-SND-01 — Soundness & Fidelidade (correções de raiz)

Corrige os dois achados de soundness desta rodada: a corrupção silenciosa do WaveNet Lite (P1) e
o estouro de buffer de stack da A2 sob blocos grandes (P9). Ambos são **bugs de produto** isolados
por leitura de código + evidência runtime; ambos têm fix de raiz seguro com precedente no codebase.

---

## Sprint S1 — P1: alinhar o ring-buffer WaveNet ao número de canais (fix de raiz)

**Objetivo**: eliminar a divergência do WaveNet Lite (CH=12) vs C++ (SNR 0,9 dB) e a violação de
invariância de tamanho de bloco, **sem tocar em kernels numéricos** (já provados bit-exatos).

**Causa raiz (provada nesta auditoria)**: o wrap do anel `WaveNetLayerState::advance_frames`
(`src/models/wavenet/common.rs:93-104`) rebobina `buffer_start` em **frames**
(`buffer_start -= buffer.size() / channels`), mas o espelho físico da `MirroredBuffer` aliasa com
período `buffer.size()` **elementos**. Quando `channels ∤ buffer.size()` (CH=12: `1024 % 12 = 4`,
`524288 % 12 = 8`; CH=6: `1024 % 6 = 4`), o rewind fica curto em `buffer.size() % channels`
elementos a cada wrap → desalinha **todo** o histórico de dilatação por uma fração de frame →
corrupção acumulativa, dependente do tamanho de bloco. Lite (array1 CH=12, array2 CH=6) é o único
SKU padrão com canais que não são potências de 2 dividindo a página.

### T1.1 — 🟡 Harness de reprodução e localização (pré-fix, baixo risco) [DONE]

**Por quê**: travar a evidência e a localização **antes** de mexer em memória virtual, garantindo
que o fix de T1.2 seja medido contra um baseline reproduzível.

**Escopo**:

1. Teste unitário determinístico provando a aritmética da raiz, em `src/dsp/mirror_buf.rs`
   (inline, arquivo <300 linhas) ou `mirror_buf_test.rs`:
   - Para `channels ∈ {4, 8, 16}` (potências de 2) e `{6, 12}` (Lite): construir
     `MirroredBuffer::<f32>::new(min_elems)` com `min_elems` típico de RF Lite, e **assertar**
     `buf.size() % channels == 0`. Hoje **falha** para 6 e 12 — este teste vermelho é o alvo de
     T1.2.
2. Teste de invariância de tamanho de bloco sintético em `tests/` (rápido, sem modelo externo):
   construir um WaveNet **CH=12** const-generic (reusar `build_tiny_wavenet`-style ou
   `wavenet_lite` golden `tests/fixtures/models/BossWN-lite.nam`), processar o **mesmo** sinal em
   `block_size ∈ {1, 16, 32, 64}` e comparar MSE par-a-par. Marcar `#[ignore]` (ou gate brando)
   **enquanto P1 estiver aberto**; será endurecido em T1.3.
3. (Opcional, diagnóstico) harness instrumentado debug-only que despeja, por camada/array, o
   `layer_buffer` em `bs=1` vs `bs=64` e reporta o **primeiro frame divergente** — para confirmar
   visualmente que a divergência some após T1.2.

**Critério de aceite**: testes reproduzem o `size() % channels ≠ 0` para CH∈{6,12} e a violação de
invariância de bloco do Lite (MSE ~1–2e-2, consistente com a evidência
`nondist_validation`: 2,08e-2/1,95e-2/1,32e-2).

**Risco**: 🟢 baixo (só leitura/medição). **Sem** alocação no hot-path.

### T1.2 — 🔴 `MirroredBuffer` alinhado a canais + `WaveNetLayerState` usa-o (FIX DE RAIZ — máxima atenção) [DONE]

**Por quê**: é a correção da causa raiz. Mexe em `mmap`/espelho de páginas — errar quebra **toda**
a inferência WaveNet (não só Lite). Exige revisão cuidadosa e golden C++ antes/depois.

**Escopo**:

1. Em `src/dsp/mirror_buf/alloc.rs`, adicionar construtor
   `MirroredBuffer::<T>::new_aligned(requested_size: usize, elem_multiple: usize)` que arredonda
   `size_bytes` para **múltiplo de `lcm(page_size, elem_multiple * size_of::<T>())`** (tanto no
   caminho 4 KB quanto no huge-page 2 MB). Garante simultaneamente: (a) `size_bytes` múltiplo da
   página (requisito do mmap/espelho) **e** (b) `size_elements % elem_multiple == 0`. O `new`
   atual delega para `new_aligned(requested_size, 1)` (comportamento inalterado para os demais
   usuários do buffer).
   - `lcm` via `gcd` (Euclides) em `usize`, com `checked_mul` (overflow → `Err` como já ocorre).
   - Documentar o custo: para CH=12 e página 4 KB, `lcm(4096, 48) = 12288 B` (256 frames de 12) —
     overhead desprezível; para huge-page 2 MB, `lcm(2 MiB, 48) = 6 MiB` — aceitável (cold, no
     load). Avaliar se vale, no caminho huge-page, **não** forçar `elem_multiple` e em vez disso
     alinhar `size_elements` para múltiplo de `elem_multiple` mantendo `size_bytes` múltiplo da
     página (i.e. múltiplo de `lcm`) — manter a fórmula `lcm` é o caminho **correto e simples**.
2. Em `src/models/wavenet/common.rs`, `WaveNetLayerState::new` passa a chamar
   `MirroredBuffer::<f32>::new_aligned(min_buffer_frames * channels, channels)`. Assim
   `buffer.size() % channels == 0` **por construção** e o wrap de `advance_frames` (inalterado)
   rebobina exatamente o período do espelho.
3. **Não alterar** `advance_frames` nem os kernels conv — só a garantia de divisibilidade. Manter
   o `debug_assert` de underflow existente.
4. Auditar **todos** os call-sites de `WaveNetLayerState::new` (`loader/dispatcher/wavenet/
   standard.rs:150`, `dynamic.rs:59`, e os de teste em `tests.rs`/`dynamic_parity.rs`) — nenhum
   precisa mudar (a assinatura de `new` é a mesma), mas confirmar.

**Critério de aceite**:

- O teste de T1.1 (`size() % channels == 0`) fica **verde** para CH∈{4,6,8,12,16}.
- Invariância de bloco do Lite: MSE entre `bs ∈ {1,16,32,64}` cai para **0,0** (ou ≤ epsilon de
  ordenação FP, esperado **exatamente 0** pois os caminhos viram bit-idênticos).
- `cargo test` + heap-audit verdes; **zero alocação** no hot-path (a mudança é só no `new`/load).
- RT-safety preservada (alocação só no construtor, cold).

**Risco**: 🔴 **alto** (memória virtual). Mitigação: golden vs C++ obrigatório (T1.3), testes de
todos os SKUs (Standard/Feather/Nano **não podem regredir** — eles já passavam), e revisão do
`lcm`/overflow.

### T1.3 — 🟠 Validação contra C++, reabilitação do golden e endurecimento dos gates [DONE]

**Por quê**: provar paridade com a bíblia (NAMCore) e **fechar a janela** que mascarava P1.

**Escopo**:

1. **Restaurar o assert rígido** de invariância de bloco em `tests/nondist_validation.rs:138`
   (hoje rebaixado a `WARN [P1]`) — voltar a `assert!(test_mse < 1e-7, ...)`.
2. **Reabilitar o golden** `test_golden_vectors_wavenet_lite` (`tests/golden_vectors.rs:484`):
   remover `#[ignore = "known-divergent..."]` e o SKIP do `cpp_parity` (`tests/cpp_parity.rs:458`)
   - `self_consistency.rs` nota.
3. **Regenerar/validar** os goldens do Lite vs o `render` C++ pinado (commit `9c7b185`): v1
   (`golden_wavenet_lite.bin`) e, se aplicável, o v2 — confirmar SNR ≫ 40 dB (esperado ~100+ dB,
   como os demais WaveNet pós-nuke). Atualizar threshold em `tests/common/validation.rs` se
   necessário (com margem honesta, **sem** afrouxar — regra de ouro P3).
4. Cross-validação **live** vs C++ (`utils/tests-long.sh` Fase 3) para Lite: `live_cross_validation_
   wavenet_lite` deve passar.
5. Teste de regressão dedicado: `wavenet_ringbuffer_alignment` assertando
   `buffer.size() % channels == 0` para a lista de SKUs (CH∈{12,6} inclusos) — guard-rail contra
   futuras geometrias.

**Critério de aceite**: golden Lite verde (SNR alto), invariância de bloco com assert rígido verde,
`tests-long.sh` Fase 3 Lite verde. Atualizar `TODO-problemas.md §P1` para ✅ RESOLVIDO.

**Risco**: 🟡 médio (dados de teste/regeneração de golden). Cuidado: **não** afrouxar thresholds
para "passar" — se o golden não bater, a raiz não foi corrigida.

### T1.4 — 🟡 Reverificar `wavenet_official` (free-geom) pós-fix [DONE]

**Por quê**: P13 classificou `wavenet_official` como known-divergent na mesma classe do Lite. Se a
raiz é a mesma (canais ∤ página), o fix de T1.2 deve resolvê-lo também.

**Escopo**: após T1.2, rodar o gate v2 de `wavenet_official` (`tests/golden_vectors.rs:1054`) sem o
`--skip`. Se passar, remover o `#[ignore]`/skip (`utils/tests-long.sh`) e marcar P13 plenamente
resolvido. Se **não** passar, registrar em `TODO-problemas.md` que a geometria Official tem causa
**adicional** (ex.: head_kernel_size > 1, número de arrays) e abrir investigação separada — **não**
reverter o fix do Lite.

**Critério de aceite**: decisão documentada (resolvido OU causa residual isolada), goldens
consistentes, sem afrouxamento.

**Risco**: 🟢 baixo (verificação).

---

## Sprint S2 — P9: chunking interno da A2 (correção de UB de stack)

**Objetivo**: eliminar o estouro de buffer de stack (UB em release / panic em debug) quando um host
negocia bloco > 64 com um modelo A2, tornando o contrato da A2 consistente.

**Causa raiz (confirmada nesta auditoria)**: `set_max_buffer_size` cresce `max_buffer_size` sem
clamp (`src/models/a2/model/mod.rs:186-216`); `process()` faz `nf = num_frames.min(max_buffer_size)`
(`:260`) e passa `nf` direto para os kernels `layer_forward_ch{3,8}_block`, que têm scratch de stack
**fixo de 64 frames** (`conv1d_ch8.rs:293,389` = `[f32; 64*8]`; `conv1d_ch3/{simd.rs:252,
scalar.rs:74}` = `[f32; 64*4]`). Com `nf>64` o kernel escreve `nf*ch` > capacidade → UB.

### T2.1 — 🔴 Re-chunk interno em `A2::process` a ≤64 frames (FIX DE RAIZ — máxima atenção) [DONE]

> **Implementado**: `src/models/a2/model/mod.rs:242` — `process()` agora executa um laço `while pos < nf_total` idêntico ao padrão WaveNet A1. Cada sub-bloco ≤64 frames alimenta os kernels existentes sem tocá-los. Golden vectors (A2-Full/A2-Lite), self-consistency, prewarm e smoke tests todos verdes. RT-safety: zero alocação, só fatiamento de slices.

**Por quê**: corrige a raiz espelhando o padrão já consagrado do WaveNet A1; é a opção (a) do P9 e
a recomendação do auditor.

**Escopo**:

1. Em `src/models/a2/model/mod.rs::process`, envolver o corpo de processamento num laço
   `while pos < total { let nf = (total - pos).min(WAVENET_MAX_NUM_FRAMES); ...; pos += nf; }`,
   **idêntico em espírito** a `WaveNet A1` (`src/models/wavenet/model.rs:65-101`) e ao próprio
   `A2::prewarm` (`model/mod.rs:457-465`). Cada sub-bloco ≤64 alimenta os kernels existentes sem
   tocá-los.
2. Garantir que o estado temporal da A2 (rings de history, head ring) **avança corretamente**
   entre sub-blocos (validar com o teste de invariância de bloco estendido em T2.3).
3. **RT-safety**: zero alocação nova (os buffers de modelo já são dimensionados por
   `set_max_buffer_size`); só fatiamento de slices. Zero panic, zero lock.

**Critério de aceite**: processar blocos de 128/256/512 produz saída **bit-idêntica** a processar em
sub-blocos de 64 (e ao A1-style chunking); `cargo test --release` sem UB (rodar sob o teste de T2.3);
paridade C++ da A2 inalterada (golden A2-Full/A2-Lite).

**Risco**: 🔴 alto (hot-path A2 + estado temporal). Mitigação: golden A2 vs C++ antes/depois,
heap-audit, e o teste de varredura de bloco >64 de T2.3.

### T2.2 — 🟢 Tornar `set_max_buffer_size` coerente + remover `debug_assert` enganoso [DONE]

**Por quê**: após T2.1 o kernel nunca recebe >64, então o `debug_assert!(num_frames <= 64)` vira
documentação interna correta; e `set_max_buffer_size` pode crescer com segurança.

**Escopo**:

1. Confirmar que `set_max_buffer_size` pode crescer livremente (T2.1 garante segurança via
   re-chunk) — manter o crescimento de buffers de modelo, sem clamp artificial. Documentar no
   doc-comment que o contrato é "qualquer bloco; processado internamente em sub-blocos de ≤64".
2. Manter os `debug_assert!(num_frames <= max_frames)` dos kernels como **invariante interna**
   (agora sempre satisfeita); adicionar comentário explicando que `process()` garante ≤64.
3. (Opcional) `const MAX_KERNEL_FRAMES: usize = WAVENET_MAX_NUM_FRAMES;` para substituir os literais
   `64`/`max_frames = 64` mágicos nos 4 kernels — coesão e prevenção de regressão.

**Critério de aceite**: `model_test.rs` (crescimento até 256) continua verde; doc atualizada; sem
literais `64` órfãos nos kernels (se a sub-tarefa opcional for feita).

**Risco**: 🟢 baixo.

> **Implementado**: (1) doc-comment de `set_max_buffer_size` (`model/mod.rs:185`) explica o contrato
> de chunk interno; (2) `const MAX_KERNEL_FRAMES: usize = 64` definida em `conv1d_ch8.rs:37`,
> `conv1d_ch3/simd.rs:14` e `conv1d_ch3/scalar.rs:12`, substituindo todos os literais `64` e
> `max_frames = 64` mágicos nos 4 kernels; (3) comentários nos `debug_assert!` dos kernels
> documentam que `process()` garante `≤ MAX_KERNEL_FRAMES` via chunking (T2.1).

### T2.3 — 🟢 Estender cobertura de invariância de bloco >64 + regressão de UB

**Por quê**: `tests/nondist_validation.rs:118` limita a varredura a `[1,16,32,64]` **por causa do
P9**. Após T2.1, blocos grandes ficam seguros.

**Escopo**:

1. Estender `block_sizes` em `nondist_validation.rs` para incluir `128, 256, 512` (a A2 agora
   re-chunka). Atualizar o comentário (`:111-117`) removendo a admissão do P9.
2. Teste de regressão dedicado A2: processar 1 bloco de 256 frames e comparar com 4 blocos de 64
   (MSE == 0). Rodar em **release** (o UB só se manifesta sem `debug_assert`).
3. Teste negativo: garantir que nenhum kernel A2 recebe `num_frames > 64` (ex.: contador/asserção
   de invariante no caminho de teste).

**Critério de aceite**: varredura de bloco até 512 verde para A2 (e demais arquiteturas); regressão
de UB verde em release; comentário do P9 removido do teste. Atualizar `TODO-problemas.md §P9` para
✅ RESOLVIDO.

**Risco**: 🟢 baixo.

---

## Épico E-OPT-02 — Otimização & higiene (O1 + O5/S2)

Fecha formalmente o **O1** (já implementado, só documentação) e executa o resíduo de **O5 (S2)**
identificado nesta auditoria (micro-otimização do rechannel da A2). Baixo risco, entrega de higiene
e pequeno ganho de hot-path.

---

## Sprint S3 — Fechamento O1 + decode f16 do A2 rechannel (O5/S2)

### T3.1 — 🟢 Formalizar O1 como concluído (documentação)

**Por quê**: a auditoria confirmou que O1 está **100% implementado** (`src/math/common/half.rs`
completo, `half` removido do `Cargo.toml`, todas as 16 caudas SIMD usam `f16_bits_to_f32_f16c`,
testes exaustivos 65.536 presentes). Falta só fechar o registro.

**Escopo**:

1. `TODO-optimize.md §O1`: já atualizado para ✅ DONE nesta rodada — confirmar consistência da
   matriz/PoC (linhas de PoC O1 e a ordem recomendada que ainda listava "O1 → ...").
2. Verificar que não há `half =` no `Cargo.toml` (confirmado) e que o resíduo em `Cargo.lock` é
   apenas transitivo via `ciborium`/`clack-*` — documentar como **fora do escopo** (não é
   dependência direta do nam-rs).
3. (Opcional) Nota em `docs/` (skill `documentador`) sobre o módulo `half.rs` interno e a garantia
   F16C ≡ software (bit-exato) que sustenta as otimizações de O5/S2.

**Critério de aceite**: documentação coerente; nenhuma ação de código (O1 já entregue).

**Risco**: 🟢 trivial (doc).

### T3.2 — 🟢 A2 rechannel: pré-decodar `rechannel_w` (f16→f32) no load + `mul` SIMD no hot-path

**Por quê**: `src/models/a2/model/mod.rs:264-270` decodifica `f16_bits_to_f32(self.rechannel_w[c])`
(versão **software**) **a cada frame**, embora as `CH` constantes nunca mudem (CH=8 × 64 frames =
512 decodes/bloco, só 8 valores únicos). É a única lacuna escalar nova de produção achada por O5.

**Escopo**:

1. Pré-decodar `rechannel_w` (u16/f16) para um `AlignedVec<f32>` **uma vez no load** (em
   `set_weights.rs`, cold path) — usando `f16_bits_to_f32_f16c` (F16C; bit-exato vs software,
   provado em O1). Armazenar o vetor f32 no modelo.
2. No hot-path, o laço de rechannel vira `layer_in[base+c] = rechannel_w_f32[c] * x` — sem decode
   por frame. Vetorizar como `mul` SIMD (broadcast de `x` × vetor de pesos f32), reaproveitando os
   kernels `SimdMath` (ex.: padrão `apply_gain`/gemv) — guard-rail O5: nada escalar no hot-spot.
3. Garantir RT-safety: o `AlignedVec<f32>` é alocado no load; hot-path zero-alloc.

**Critério de aceite**: ESR/golden da A2 (A2-Full/A2-Lite) **inalterado** (bit-exato, F16C ≡
software); `cargo bench` do caminho A2 mostra rechannel ↓ (sem decode por frame); heap-audit verde.
Atualizar `TODO-optimize.md §O5` (S2) para ✅ DONE.

**Risco**: 🟢 baixo (mudança local, validável por golden bit-exato).

> **Nota de roteamento (O5 transversal):** os achados **S1** (A2 head conv escalar, `head.rs:96-118`)
> e **S4** (MAC complexo do cabsim, `cabsim/conv.rs:259-292` = O3a) **permanecem roteados** a
> `TODO-features.md §F3` e `TODO-optimize.md §O3a` respectivamente — **não** entram nesta rodada.
> **S3** (head_scale WaveNet) e a **limpeza BF16 morta** já estão ✅ DONE. O **guard-rail** de O5
> ("nada escalar no hot-spot") permanece como regra de revisão para qualquer sprint que toque
> inferência — incluindo **T3.2** acima.

---

## Critérios de validação por sprint (padrão do projeto)

- **Bíblia C++ como golden**: P1 (S1) e P9 (S2) validados contra o `render` NAMCore pinado
  (`utils/tests-long.sh` Fase 3); nenhuma correção pode regredir a paridade dos demais SKUs.
- **Bit-exatness onde aplicável**: T1.2 deve zerar a variância de bloco (não só reduzir); T3.2 é
  bit-exato por F16C ≡ software (O1).
- **RT-safety**: toda alocação no `new()`/load; hot-path zero-alloc/zero-lock/zero-panic
  (auditável com `heap-audit`).
- **Sem afrouxar gates** (anti-padrão P3): se um golden não bater pós-fix, a raiz não foi corrigida
  — investigar, não relaxar threshold.
- **Fechamento**: cada tarefa encerra com `cargo check` + `cargo test` (+ `cargo bench` quando
  perf) verdes e **zero warnings** (`.agents/rules/linting.md`), e atualiza o "P"/"O" de origem em
  `TODO-problemas.md`/`TODO-optimize.md`.

## Matriz de risco (atenção do auditor)

```text
RISCO
  ▲
🔴 │  S1.T1.2 (mmap/espelho)      S2.T2.1 (re-chunk hot-path A2)
   │
🟠 │  S1.T1.3 (golden/gates)
   │
🟢 │  S1.T1.1  S1.T1.4  S2.T2.2  S2.T2.3  S3.T3.1  S3.T3.2
   └────────────────────────────────────────────────────────►
        baixo impacto                         alto impacto
```

- **S1.T1.2** e **S2.T2.1** são os pontos de maior risco: mexem em memória virtual e no hot-path de
  inferência. Exigem golden C++ antes/depois, testes de **todos** os SKUs (não só o alvo) e revisão
  par. Os demais itens são de baixo risco (medição, doc, micro-otimização bit-exata, gates).
