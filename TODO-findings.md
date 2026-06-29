<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Bugs (foco: "Ace of Bug Hunting")

> Artefato produzido pela skill `revisor-auditor` (perfil **Ace of bug hunting**),
> estruturado pela skill `planejador-arquiteto`.
>
> **Escopo desta rodada:** caça a bugs reais, vetores de ataque, inconsistências de
> funcionamento, quebras de estabilidade e regressões de fidelidade sonora.
> Cada item abaixo foi **verificado lendo o código-fonte real**, não apenas
> inferido. Falsos positivos investigados estão documentados ao final para
> evitar retrabalho.
>
> **Plataforma-alvo assumida:** `x86-64-v3` (Linux), conforme `README.md`. Vetores
> que só se manifestam em arquiteturas de memória fraca (ARM) ou alvos 32-bit
> foram rebaixados de severidade por estarem fora do alvo suportado.

---

## Sumário executivo

| ID  | Severidade  | Título curto                                                              | Arquivo principal                         |
| --- | ----------- | ------------------------------------------------------------------------- | ----------------------------------------- |
| F1  | **ALTA**    | A2 + Adaptive Compute: crossfade corrompe estado recorrente (double-pass) | `src/dsp/pipeline/stages/inference.rs`    |
| F2  | **MÉDIA**   | Oversampling: latência do engine não é reportada ao host (PDC)            | `src/clap/processor/events.rs`            |
| F3  | MÉDIA-BAIXA | Resampler: latência superestimada para bancos de fase mínima              | `src/dsp/resampler.rs`                    |
| F4  | BAIXA-MÉDIA | Adaptive: `prev_state` dessincroniza em transições encadeadas             | `src/dsp/adaptive.rs`                     |
| F5  | BAIXA       | `WeightCursor`: `pos + len` pode estourar `usize` (release)               | `src/loader/dispatcher/weight_cursor.rs`  |
| F6  | BAIXA       | Flag `RESAMPLER_REBUILD_FAILED` permanece setada após falha               | `src/standalone/pw_host/capture/setup.rs` |
| F7  | BAIXA       | `OversampleEngine::upsample` sem checagem de tamanho do `output`          | `src/dsp/oversample.rs`                   |
| F8  | TRIVIAL     | Cabeçalho SPDX incorreto (`Apache-2.2`)                                   | `src/dsp/oversample.rs:1`                 |
| F9  | LATENTE     | A2: overflow de `head_write_pos` se `process()` antes de `prewarm()`      | `src/models/a2/model/static/process.rs`   |
| F10 | LATENTE     | A2 FiLM: `get_unchecked` OOB se `condition_size > 1` com `conv_post_film` | `src/models/a2/model/dynamic/process.rs`  |

---

## F1 — [ALTA] A2 + Adaptive Compute: o crossfade de "soft-degrade" corrompe o estado recorrente do modelo (double-pass sem backup/restore)

### F1 — Contexto

O caminho de inferência implementa um *crossfade* de 32 ms quando o FSM de
*adaptive compute* muda de estágio (Full ↔ Reduced ↔ Minimal). Para modelos
WaveNet A1, isso é feito com a técnica de **double-pass**: o bloco é processado
duas vezes (com a contagem de camadas antiga e a nova) e os dois resultados são
misturados. Para que o segundo passe seja consistente, o estado recorrente do
modelo (posições dos *ring buffers*) é salvo **antes** do 1º passe e
**restaurado** antes do 2º passe.

### F1 — Localização

- Double-pass: `src/dsp/pipeline/stages/inference.rs:322-389` (PATH A) e bloco
  equivalente em PATH B (~`:560`).
- No-ops fatais para A2:
  - `src/models/static_model.rs:54-56` → `set_effective_layers` é **no-op** para
    `WavenetA2Full | WavenetA2Lite | WavenetA2Dyn`.
  - `src/models/static_model.rs:83` → `backup_buffer_starts` cai no braço `_ => {}`
    (no-op para A2).
  - `src/models/static_model.rs:96` → `restore_buffer_starts` idem (no-op para A2).
- Classificação que habilita o caminho: `src/models/static_model.rs:214-225` →
  `is_wavenet()` retorna **`true`** para A2.
- Gatilho do double-pass: `src/dsp/pipeline/stages/inference.rs:322`
  (`if old_eff_l != new_eff_l || ...`).
- Contagem de camadas A2: `src/models/static_model.rs:132-134` →
  `layer_count()` retorna `A2_NUM_LAYERS` (= **23**, `src/models/a2/params.rs:33`).
- Estado recorrente que **avança de forma persistente** a cada `process()`:
  `src/models/a2/model/static/process.rs:180` / `:182`
  (`self.layer_buffer_starts[li] = ...`) e `:410`
  (`self.head_write_pos = ...`).

### F1 — Causa raiz

`is_wavenet()` inclui A2, então A2 entra no caminho de *crossfade* de
salto-de-camadas do WaveNet A1. Porém A2 usa uma arquitetura de buffer
**diferente** (`MirroredBuffer` + arrays `layer_buffer_starts`/`head_write_pos`),
e por isso `set_effective_layers`, `backup_buffer_starts` e
`restore_buffer_starts` foram deixados como **no-op** para A2.

Como `wavenet_effective_layers_for_state` (`src/dsp/adaptive.rs:536-549`) com
`total_layers = 23` produz valores **distintos** por estado:

- Full → `23`
- Reduced → `23 - round(23×0.25)` = `23 - 6` = **17**
- Minimal → `23 - round(23×0.50)` = `23 - 12` = **11**

…a condição `old_eff_l != new_eff_l` (`inference.rs:322`) é **sempre verdadeira**
durante qualquer transição. Logo o double-pass executa, mas:

1. `backup_buffer_starts` não salva nada (no-op);
2. 1º passe **avança** `layer_buffer_starts[*]` e `head_write_pos`;
3. `restore_buffer_starts` **não restaura** (no-op);
4. 2º passe roda sobre o **mesmo bloco de entrada**, mas com o estado já avançado:
   lê histórico errado, escreve em posições deslocadas e **avança o estado de novo**.

### F1 — Impacto (gatilho realista)

Disparado quando, simultaneamente:

- `AdaptiveComputeMode` = `Conservative` ou `Aggressive` (i.e., adaptive ligado),
  **ou** um `SlimOverride` ativo; **e**
- há um modelo A2 "cru" carregado (`WavenetA2Full/Lite/Dyn`, fora de um
  `Container`); **e**
- ocorre uma transição de estágio do FSM (pressão de CPU ou troca de override).

Consequências durante (e após) cada *crossfade* de ~32 ms:

- **Corrupção de fidelidade sonora (regressão proibida pela política do projeto):**
  o `MirroredBuffer` de cada camada passa a conter o mesmo bloco de entrada
  duplicado em posições consecutivas → o histórico da convolução fica
  "esticado no tempo" (≈ 2×) → saída distorcida/glitch durante o crossfade e
  artefatos residuais pela duração do campo receptivo.
- **Custo de CPU dobrado justamente sob pressão de CPU** — efeito contrário ao
  propósito do adaptive compute; pode *causar* o xrun que deveria evitar.
- **Benefício nulo:** como `set_effective_layers` e `set_slimmable_size` são no-op
  para A2 cru, o adaptive compute **não reduz** carga alguma em A2 — só introduz o
  dano acima.

> Observação de escopo: `Container` (slimmable) **não** é afetado, pois
> `is_wavenet()` não inclui `Container` — ele degrada via troca de submodelo.
> O bug é específico de A2 carregado diretamente.

### F1 — Severidade

**ALTA.** Regressão de qualidade sonora + regressão de performance em RT, numa
combinação de funcionalidades documentadas (adaptive compute + arquitetura A2).
Mitigado parcialmente por: A2 ser Beta e `AdaptiveComputeMode::default() = Off`
(`src/dsp/adaptive.rs:72-73`) — exige o usuário habilitar o adaptive.

### F1 — Proposta de solução

Três alternativas, em ordem de preferência:

1. **(Recomendado — barato e seguro) Excluir A2 do crossfade de salto-de-camadas.**
   Tornar o gatilho ciente de que A2 não suporta salto-de-camadas. Ex.: adicionar
   `StaticModel::supports_layer_skip()` (true só para WaveNet A1 estático/dyn) e
   trocar, em `inference.rs:250`,
   `let is_crossfading_wavenet = is_wavenet && ctx.adaptive.is_crossfading();`
   por uma checagem que exclua A2. Com isso, A2 segue pelo caminho
   `model_process_stereo_with_os` normal (passe único), sem corrupção. Como o
   salto-de-camadas é no-op para A2 de qualquer forma, **não há perda de função**.

2. **Implementar de fato `backup/restore_buffer_starts` + `set_effective_layers`
   para A2.** Mais trabalhoso (A2 tem 23 estados de `layer_buffer_starts` +
   `head_write_pos`; é preciso salvar/restaurar todos e implementar salto real de
   camadas no `process_internal`). Só vale a pena se quisermos salto-de-camadas
   funcional em A2.

3. **Rotear toda a degradação de A2 exclusivamente via slimmable/troca de modelo
   (A2-Full → A2-Lite),** garantindo que o FSM nunca acione o double-pass para A2.

Recomenda-se também: ao detectar A2 sem suporte a degradação por camadas, **não
contabilizar** transições de FSM que não produzem efeito (evitar `is_crossfading`
"fantasma").

### F1 — Validação sugerida

- Teste de integração: carregar A2 (Full e Lite), habilitar `Aggressive`, injetar
  pressão de CPU sintética para forçar Full→Reduced→Minimal e comparar a saída com
  um *oracle* de passe único (sem adaptive). A diferença deve ser ≈ 0 fora da
  região de crossfade legítima (e o crossfade não deve corromper estado).
- Asserção de invariante: após N blocos, `layer_buffer_starts`/`head_write_pos`
  de A2 devem refletir exatamente `N×nf` amostras processadas (não `2×`).

---

## F2 — [MÉDIA] Oversampling: a latência do `OversampleEngine` não é reportada ao host (PDC incorreta)

### F2 — Contexto

O CLAP reporta a latência do plugin ao host para *Plugin Delay Compensation*
(PDC). O cálculo soma a latência do resampler e a do cab-sim, **mas ignora o
engine de oversampling**.

### F2 — Localização

- `src/clap/processor/events.rs:116-122`:

  ```rust
  let mut effective_latency = self.resampler.latency_samples(host_rate);
  if let Some(ref conv) = self.conv_engine {
      effective_latency += conv.latency_samples() as u32;
  }
  // <-- não há contribuição do oversampling (os_l/os_r)
  ```

- `src/clap/processor/mod.rs:152-158` (latência inicial na ativação): idem.
- `src/dsp/oversample.rs` — `OversampleEngine` **não possui** método
  `latency_samples()` (confirmado: `latency_samples` só existe em `resampler.rs`
  e `cabsim/conv.rs`).

### F2 — Causa raiz

Cada estágio half-band tem atraso de grupo `HB_DELAY = 12` amostras à taxa nativa
(`src/dsp/oversample.rs:33-34`). O próprio `README.md` documenta: **2× → 12
amostras** e **4× → 24 amostras** de latência. Esse atraso real **não** é
informado ao host.

### F2 — Impacto

Ao ativar 2×/4× (controle exposto na GUI e automatizável via `PARAM_OVERSAMPLE`),
o sinal sai 12/24 amostras atrasado, mas o host compensa **0** amostras extras →
a trilha NAM fica adiantada/desalinhada em relação às demais. A 48 kHz, 24
amostras ≈ 0,5 ms — audível como *comb filtering* ao mesclar com DI/outros
microfones em sessões multitrilha. Bug de fidelidade em cenário de mixagem.

### F2 — Severidade

**MÉDIA.** Sem corrupção de amostras, mas desalinhamento de fase real e mensurável
em uso documentado (modo HQ/Offline recomenda 4×).

### F2 — Proposta de solução

1. Adicionar `OversampleEngine::latency_samples(&self) -> usize` retornando o
   atraso à taxa nativa por fator: `Off → 0`, `X2 → HB_DELAY`, `X4 → 2×HB_DELAY`
   (validar contra a medição real do par up+down).
2. Incluir essa contribuição em `events.rs:116-122` **e** em `mod.rs:152-158`:

   ```rust
   effective_latency += os_l.latency_samples() as u32; // mesmo fator em L/R
   ```

3. Como a troca de fator já dispara *rebuild* off-RT e atualização de latência
   via `current_latency`, garantir que a mudança de oversampling também force o
   `latency_ext.changed()` no laço de *housekeeping*.

### F2 — Validação sugerida

- Teste de impulso: medir o atraso de grupo real do par up/down para 2× e 4× e
  comparar com o valor reportado (devem coincidir, tolerância ≤ 1 amostra).

---

## F3 — [MÉDIA-BAIXA] Resampler: latência superestimada para bancos de fase mínima

### F3 — Contexto

O resampler de produção usa bancos polifásicos de **fase mínima**
(`NamResampler::new`, `src/dsp/resampler.rs:406`/`431-440`). O cálculo de latência,
porém, usa a fórmula de atraso de grupo de filtro de **fase linear**.

### F3 — Localização

- `src/dsp/resampler.rs:519-537` (`latency_samples`):

  ```rust
  let taps_half = TAPS_PER_PHASE as f64 / 2.0; // 64/2 = 32 amostras por estágio
  let latency_in  = taps_half * (self.pw_rate as f64 / self.nam_rate as f64);
  let latency_out = taps_half;
  (latency_in + latency_out).round() as u32
  ```

- Banco de fase mínima: `src/dsp/sinc_kernel.rs:202` (`to_minimum_phase`), descrito
  como reduzindo a latência pela metade / eliminando *pre-ringing*.
- Consumidores: `src/clap/processor/events.rs:116`, `mod.rs:152`.

### F3 — Causa raiz

`TAPS_PER_PHASE/2 = 32` é o atraso de grupo de um FIR **linear-phase**. Filtros de
fase mínima concentram a energia no início → atraso de grupo (especialmente em
baixa frequência) bem menor que 32 amostras por estágio. O `ResamplerCore` não
guarda se o banco é linear ou mínimo, então a mesma fórmula é usada para ambos.

### F3 — Impacto

Quando `pw_rate != nam_rate`, a latência reportada é superestimada (potencialmente
em dezenas de amostras), causando supercompensação de PDC no host e desalinhamento
da trilha NAM. Menos grave que F2 (magnitude depende do *cutoff*), mas é real.

> Atenção: F2 e F3 interagem — ambos afetam o número reportado ao host. Devem ser
> corrigidos em conjunto e validados juntos.

### F3 — Severidade

**MÉDIA-BAIXA.** Erro de reporte de latência; sem corrupção de áudio.

### F3 — Proposta de solução

1. Rastrear em `ResamplerCore` o tipo de fase do banco (linear vs. mínima).
2. Para fase mínima, calcular o atraso de grupo **empiricamente** na construção
   (ex.: centro de massa da resposta ao impulso, ou primeiro pico) em vez de
   assumir `taps/2`. Alternativamente, usar um limite inferior conservador
   coerente com a medição.
3. Adicionar teste que compare a latência reportada com o atraso de grupo medido
   do banco real (linear e mínimo), por par de taxas (44.1↔48, 48↔96, etc.).

---

## F4 — [BAIXA-MÉDIA] Adaptive: `prev_state` dessincroniza em transições encadeadas dentro de um crossfade

### F4 — Localização

- `src/dsp/adaptive.rs:362` (`transition_to`):

  ```rust
  self.prev_state = self.state; // sobrescreve mesmo com crossfade Active
  self.state = new_state;
  ```

- Consumidor: `src/dsp/pipeline/stages/inference.rs:282` usa
  `ctx.adaptive.prev_state()` para computar `old_eff_*` do "passe antigo".

### F4 — Causa raiz

Se uma 2ª transição dispara enquanto um crossfade ainda está `Active` (ex.:
Full→Reduced e, antes de 32 ms, Reduced→Minimal), `prev_state` passa a ser o
**alvo** da 1ª transição (`Reduced`), não a origem real pré-fade (`Full`). O
passe "antigo" do double-pass passa a usar a contagem de camadas errada.

### F4 — Impacto

Descontinuidade audível (pequeno *click*/degrau) durante rajadas rápidas de
pressão de CPU. Afeta o WaveNet A1 (onde o double-pass é funcional). Requer ≥ 2
transições dentro de ~32 ms (≈ 1536 amostras @ 48 kHz) — plausível, porém raro.

### F4 — Severidade

**BAIXA-MÉDIA.**

### F4 — Proposta de solução

Preservar a verdadeira origem do crossfade:

```rust
if !matches!(self.crossfade, CrossfadePhase::Active) {
    self.prev_state = self.state;
}
self.state = new_state;
```

…ou adicionar um campo dedicado `crossfade_origin` que só é atualizado quando o
crossfade está `Idle`.

---

## F5 — [BAIXA] `WeightCursor`: `self.pos + len` pode estourar `usize` (release) e burlar a checagem de limites

### F5 — Localização

- `src/loader/dispatcher/weight_cursor.rs:47-59` (`read_slice`):

  ```rust
  if self.pos + len > self.data.len() { bail!(...) }
  let slice = &self.data[self.pos..self.pos + len];
  ```

### F5 — Causa raiz

Em release, `overflow-checks` está desligado (vide `checked_arith.rs`), então
`self.pos + len` pode dar *wrap*. Se `len` for um produto de dimensões adversárias
não passado pelos helpers `checked_*` (ex.: `read_lstm_layer_dyn` calcula
`weights_len = h4 * ih` com `*` cru em
`src/loader/dispatcher/lstm/weights.rs:107-109`), um `len ≈ usize::MAX` faz
`self.pos + len` envolver para um valor pequeno, **passar** na checagem e então
panicar no fatiamento (`start > end`).

### F5 — Impacto

Pânico no **caminho de carga (cold path, main thread)** ao abrir um `.nam`/`.namb`
malformado/malicioso — DoS (crash do host/DAW), não unsafety de memória (os
bounds checks do Rust protegem contra OOB). O `README` declara que os loaders
"never panic"; este é um desvio dessa garantia. Demais loaders (WAV IR, header
NAMB) já estão bem defendidos.

### F5 — Severidade

**BAIXA** (cold path; pânico, não UB).

### F5 — Proposta de solução

1. Em `read_slice`, usar aritmética checada:

   ```rust
   let end = self.pos.checked_add(len)
       .filter(|&e| e <= self.data.len())
       .ok_or_else(|| anyhow!("..."))?;
   ```

2. Endurecer os produtores de dimensão da via dinâmica (LSTM/A2/WaveNet dyn) para
   usar `checked_mul*` antes de qualquer alocação (`LstmLayerDyn::new`) e antes de
   `read_slice`, com limite superior de dimensões plausíveis (ex.: hidden_size,
   num_layers).

---

## F6 — [BAIXA] Flag `RESAMPLER_REBUILD_FAILED` permanece setada após uma falha, desabilitando a espera-pelo-resampler em swaps futuros

### F6 — Localização

- RT: `src/standalone/pw_host/capture/setup.rs:165-176`.
- Main: `src/standalone/pw_host/run.rs:150-151` (limpa em sucesso) e `:195`
  (seta em falha).

### F6 — Causa raiz

No braço de falha, o RT detecta `RESAMP_SWAP_PENDING` + `REBUILD_FAILED` e **limpa
apenas o PENDING** (`setup.rs:170-171`), deixando `REBUILD_FAILED` setada. Essa
flag só é limpa no **próximo rebuild bem-sucedido** (`run.rs:150`). Assim, um swap
subsequente verá `REBUILD_FAILED` ainda setada e **curto-circuitará** o
"esperar pelo resampler" (`setup.rs:172-175`), processando alguns blocos com o
resampler antigo (taxa incorreta) até o novo chegar.

### F6 — Impacto

Transiente breve de processamento com razão de taxa incorreta após uma falha de
rebuild (que já é rara — só por falha de alocação ou taxas inválidas). Benigno,
mas é uma inconsistência lógica de máquina de estados.

### F6 — Severidade

**BAIXA.**

### F6 — Proposta de solução

Limpar `REBUILD_FAILED` no mesmo ponto em que se limpa `PENDING` no braço de
falha do RT (`setup.rs:170-171`), ou reestruturar para que a flag de falha seja
de "uma só leitura" (consumida ao ser observada).

---

## F7 — [BAIXA] `OversampleEngine::upsample` escreve com `get_unchecked_mut` sem `debug_assert` do tamanho de `output`

### Localização

- `src/dsp/oversample.rs:326-345` (chama `X2Stage::upsample`), e
  `src/dsp/oversample.rs:215-218`:

  ```rust
  *output.get_unchecked_mut(2 * i) = even_out;
  *output.get_unchecked_mut(2 * i + 1) = odd_out;
  ```

### F7 — Causa raiz

A função tem `debug_assert!(input.len() <= self.max_samples)` (`:327`), mas **não**
valida `output.len() >= input.len() * multiplier`. A segurança depende
inteiramente do chamador dimensionar `output`. Hoje os chamadores estão corretos
(`MAX_OS_BUF = MAX_RESAMP_BUF * 4`, `src/standalone/pw_host/capture/state.rs:27`;
`os_capacity = MAX_RESAMP_BUF * 4`, `src/clap/processor/mod.rs:96`), então **não há
bug ativo** — é uma lacuna defensiva.

### F7 — Impacto

Latente: uma futura mudança de chamador com `output` subdimensionado causaria
escrita OOB via `get_unchecked_mut` (UB), sem rede de segurança em release.

### F7 — Severidade

**BAIXA** (defensivo/latente).

### PF7 — roposta de solução

Adicionar no início de `upsample`/`downsample`:

```rust
debug_assert!(output.len() >= input.len() * self.factor.multiplier(),
    "oversample: output buffer too small");
```

(e análogo para `downsample` com `/ multiplier`).

---

## F8 — [TRIVIAL] Cabeçalho SPDX incorreto: `Apache-2.2`

### F8 — Localização

- `src/dsp/oversample.rs:1`:

  ```rust
  // SPDX-License-Identifier: Apache-2.2
  ```

### F8 — Causa raiz

"Apache-2.2" não é um identificador SPDX válido (a licença do projeto é
`Apache-2.0`). Viola a regra obrigatória `.agents/rules/copyright.md`.

### F8 — Severidade

**TRIVIAL** (conformidade/licenciamento), mas trivial de corrigir e afeta
ferramentas de auditoria de licença/SBOM.

### F8 — Proposta de solução

Corrigir para `// SPDX-License-Identifier: Apache-2.0`. Recomenda-se um *check* de
lint (grep) em CI que falhe se algum SPDX != `Apache-2.0` aparecer em `src/`.

---

## F9 — [LATENTE] A2: `head_write_pos` cresce sem máscara no atalho `layers.is_empty()` → possível OOB se `process()` for chamado antes de `prewarm()`

### F9 — Localização

- `src/models/a2/model/static/process.rs:60-63`:

  ```rust
  if self.layers.is_empty() {
      self.head_write_pos += total; // cresce ilimitadamente, sem & mask
      return;
  }
  ```

- `src/models/a2/model/static/process.rs:122-131` (`advance_head_ring`):
  `keep_start = self.head_write_pos - head_keep;` e `copy_within(src..src+..)`.

### F9 — Causa raiz

Enquanto não há pesos (`layers.is_empty()`), cada `process()` incrementa
`head_write_pos` por `total` **sem** aplicar `& head_ring_mask`. Se os pesos forem
carregados e `process()` chamado **sem** um `prewarm()`/`reset()` intermediário
(que reinicializa `head_write_pos`), `advance_head_ring` computa `keep_start` muito
além dos limites de `head_accum` → `copy_within` OOB (pânico) ou aritmética
inválida.

### F9 — Impacto

Latente: o caminho de produção (dispatcher) sempre chama `prewarm()` após carregar
pesos, então não dispara hoje. Risco em código de teste ou em embeddings que
contornem o dispatcher.

### F9 — Severidade

**LATENTE / BAIXA.**

### F9 — Proposta de solução

Mascarar no atalho:

```rust
if self.layers.is_empty() {
    self.head_write_pos = (self.head_write_pos + total) & self.head_ring_mask;
    return;
}
```

---

## F10 — [LATENTE] A2 FiLM: `get_unchecked` OOB se `condition_size > 1` combinado com `conv_post_film`

### F10 — Localização

- `src/models/a2/model/dynamic/process.rs` (~`:291-304`): `cond_slice` de
  comprimento 1 é passado ao FiLM.
- `src/models/a2/film.rs` (`cond_to_scale_shift`): indexa
  `condition.get_unchecked(grp * cond_per_group..)` com
  `cond_per_group = cond_size / groups`.

### F10 — Causa raiz

A via rápida A2 usa `condition_size = 1` (mixin mono), então funciona. O engine
dinâmico, porém, suporta `condition_size` arbitrário; se um modelo definir
`condition_size > 1` com `conv_post_film` ativo, o `get_unchecked` lê além do
`cond_slice` (comprimento 1) → UB/pânico.

### F10 — Impacto

Latente: não disparável pelos modelos A2 padrão atuais (`condition_size = 1`,
FiLM inativo).

### F10 — Severidade

**LATENTE.**

### PF10 — roposta de solução

Passar ao FiLM um slice de condição de comprimento `cond_size` (não 1), ou
adicionar guarda/`debug_assert` de comprimento antes do `get_unchecked`.

---

## Falsos positivos investigados e descartados (para evitar retrabalho)

Durante a auditoria, várias hipóteses (algumas levantadas por sub-agentes) foram
**verificadas no código e refutadas**. Registrado aqui para não reabrir:

1. **"Offset negativo (isize→usize) nas conv1d causal (WaveNet/A2) → OOB crítico".**
   *Refutado.* Os buffers são `MirroredBuffer` com a invariante
   `buffer_start >= receptive_field_size`, **garantida** em
   `WaveNetLayerState::new` (`src/models/wavenet/common.rs:77-86`) com
   `rf = (k-1)*dilation` (`src/loader/dispatcher/wavenet/dynamic.rs:61`). A leitura
   para trás nunca fica negativa.

2. **"Gate FSM inverte a direção do fade ao reabrir durante fade-out".**
   *Refutado.* A invariante `current_multiplier = fade_counter * inv_fade_frames` é
   mantida em **ambas** as direções (`src/dsp/gate.rs:185,196,261`); `fade_counter`
   é a posição de ganho, então continuar incrementando ao reverter é correto.

3. **"Gate divisão por zero com `n_samples == 0`".** *Refutado.* Inalcançável: o
   guard `n_samples > 0` em `src/standalone/pw_host/rt_callback/process.rs:66` e a
   consistência `ramp_samples == n_samples` tornam o caminho morto; e o buffer
   vazio anularia o efeito.

4. **"Silêncio permanente por falta de `Acquire` em `RESAMPLER_REBUILD_FAILED`".**
   *Refutado como prático* em x86-64 (TSO): todo `NEEDS_RESAMPLER_REBUILD` resolve
   para *push* de resampler (limpa `PENDING` via `drain_resamplers`) ou para
   `REBUILD_FAILED` (limpa `PENDING` no RT). A lógica residual relevante virou F6.

5. **Loaders WAV IR e header NAMB** — auditados e considerados **bem defendidos**
   (clamp de tamanho, `saturating_add` em `find_chunk`, CRC32, checagem de
   finitude, cap de `MAX_FLOAT_COUNT`). Sem bug de memória.

6. **Caminho RT (CLAP `process` + PipeWire callback)** — auditado: **zero alloc,
   zero lock, zero I/O**; o *drop* do modelo/resampler/IR antigos é corretamente
   delegado ao GC cascade (sem drop no RT). `GcOverflowBuffer` usa `swap(AcqRel)`
   por slot (sem janela de torn-read / double-free).

---

## Épicos (agrupamento para resolução otimizada)

> Agrupados por afinidade técnica e risco, para correção segura e eficiente.
> (A quebra em sprints/tarefas — `TODO-sprints.md` — só deve ser feita quando
> solicitada, conforme a skill `planejador-arquiteto`.)

## Épico A — Saneamento do Adaptive Compute para A2 (CRÍTICO) [DONE]

**Findings:** F1 (ALTA), F4 (BAIXA-MÉDIA).
**Objetivo:** eliminar a corrupção de estado/CPU em A2 sob adaptive e corrigir a
dessincronização de `prev_state` em transições encadeadas.
**Risco:** ALTO (toca o hot-path de inferência e o FSM). Exige testes de
oracle/parity antes e depois. Começar por F1 opção 1 (exclusão de A2 do
double-pass), que é a de menor superfície de risco.

## Épico B — Reporte de latência correto para PDC

**Findings:** F2 (MÉDIA), F3 (MÉDIA-BAIXA).
**Objetivo:** o host receber a latência real (resampler + cab + **oversampling**),
com fase mínima medida corretamente. Corrigir e validar **em conjunto** (ambos
alteram o número final reportado).
**Risco:** MÉDIO. Não toca o áudio em si, mas erra alinhamento. Validar com teste
de impulso/atraso de grupo.

## Épico C — Endurecimento de robustez (loader e máquina de estados de swap)

**Findings:** F5 (BAIXA), F6 (BAIXA), F7 (BAIXA).
**Objetivo:** loaders "never panic" de fato (aritmética checada), flags de swap
consistentes e guardas defensivas no oversampling.
**Risco:** BAIXO. Mudanças locais e de baixo acoplamento.

## Épico D — Conformidade e higiene

**Findings:** F8 (TRIVIAL).
**Objetivo:** corrigir SPDX e adicionar *lint* de CI para SPDX/licença.
**Risco:** MÍNIMO.

## Épico E — Endurecimento latente da arquitetura A2 (Beta)

**Findings:** F9 (LATENTE), F10 (LATENTE).
**Objetivo:** fechar caminhos OOB/pânico que hoje não disparam por dependerem de
fluxos fora do padrão (bypass de `prewarm`, `condition_size > 1`).
**Risco:** BAIXO. Defensivo; relevante conforme A2 amadurece para fora do Beta.

---

## Mapeamento de Findings × Documentação (docs/)

> Esta seção relaciona cada finding com os itens de **"Pending / Open Work"** de
> `docs/cpp_parity_map.md` (§13) e `docs/audio_fidelity_map.md` (§9), além de
> apontar:
>
> - onde um finding **amplia** a descrição de um item já aberto;
> - onde um finding **cria** um novo item que falta nos docs;
> - **referências cruzadas obsoletas** a corrigir nos próprios docs.

---

## Tabela de cruzamento

| Finding | Severidade  | Item em `audio_fidelity_map.md §9`                                                                                                                                                   | Item em `cpp_parity_map.md §13`                                                             | Observação                                                                                                                                           |
| ------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **F1**  | ALTA        | §8 Adaptive Compute (descrição) — *nova lacuna, não listada como Pending*                                                                                                            | §6 A2 (Beta) / §13 "A2 general engine out of scope" — *F1 afeta o fast-path, não o general* | F1 **não está** em nenhuma seção Pending; precisa ser adicionado como novo item.                                                                     |
| **F2**  | MÉDIA       | §5 Oversampling — latência documentada (12 / 24 amostras) **mas não reportada**. §9 "Runtime oversample switching" — **bloqueante para essa feature**                                | —                                                                                           | F2 é pré-requisito do item Pending §9/"Runtime oversample switching": sem PDC correto, cada switch causa salto de latência não comunicado ao host.   |
| **F3**  | MÉDIA-BAIXA | §4 Resampler — "Group delay (min-phase): Asymmetric" confirma que o default é fase-mínima. §9 "Resampler quality selector" — qualquer futuro modo Standard/HQ herda o bug de fórmula | —                                                                                           | A §4 documenta o comportamento min-phase mas **não menciona** que `latency_samples()` usa a fórmula de fase linear. O gap é silencioso nos docs.     |
| **F4**  | BAIXA-MÉDIA | §8 Adaptive Compute (FSM transitions) — edge case de dupla-transição não mencionado                                                                                                  | —                                                                                           | Não listado como Pending; novo item a adicionar.                                                                                                     |
| **F5**  | BAIXA       | —                                                                                                                                                                                    | —                                                                                           | Robustez de loader — não tem correspondente nos docs de fidelidade/paridade.                                                                         |
| **F6**  | BAIXA       | §4 Resampler (lifecycle de swap)                                                                                                                                                     | —                                                                                           | Lógica de flag de falha; não listado como Pending.                                                                                                   |
| **F7**  | BAIXA       | §5 Oversampling (OversampleEngine)                                                                                                                                                   | —                                                                                           | Lacuna defensiva; não listada como Pending.                                                                                                          |
| **F8**  | TRIVIAL     | —                                                                                                                                                                                    | —                                                                                           | SPDX incorreto em `oversample.rs`; sem correspondente nos docs.                                                                                      |
| **F9**  | LATENTE     | —                                                                                                                                                                                    | §11.1 "Reset does NOT prewarm on load" — a divergência C++↔Rust **é a causa raiz** de F9    | A §11.1 documenta que `reset()` não chama `prewarm()` (divergência intencional do C++), mas **não avisa** da implicação para `head_write_pos` de A2. |
| **F10** | LATENTE     | —                                                                                                                                                                                    | §6 A2 + §13 "A2 general engine (FiLM/gating/condition_dsp) 🟡 Out of scope"                 | F10 fica dentro do escopo explicitamente excluído (FiLM/`condition_size > 1`). O item §13 já rastreia que é fora-de-escopo.                          |

---

## F1 — Detalhamento do cruzamento com `audio_fidelity_map.md §8`

`audio_fidelity_map.md §8` descreve o Adaptive Compute assim:

> *"Each tier uses a lighter sub-model (e.g., WaveNet Standard → WaveNet Nano)."*

Essa descrição descreve o comportamento do **Container/slimmable** (troca de submodelo),
mas **não** o caminho de *layer-skipping* do WaveNet A1 (que usa os mesmos pesos e
reduz camadas ativas dentro do próprio modelo). Para A2 — que não suporta *layer-skip*
nem troca de submodelo — a seção está **implicitamente errada**: sugere que toda
degradação é por troca de submodelo, quando na verdade A2 não faz nada útil sob adaptive.

**Ação necessária em `audio_fidelity_map.md §8`:** esclarecer que existem dois
mecanismos distintos de degradação (layer-skip para WaveNet A1 e troca para Container),
e adicionar uma nota sobre as limitações do A2 na Beta.

---

## F2 — Detalhamento do cruzamento com `audio_fidelity_map.md §5` e §9

`audio_fidelity_map.md §5` documenta explicitamente:

| Modo | Latência adicionada                           |
|:---- |:--------------------------------------------- |
| 2×   | 12 amostras @ taxa nativa (~0,25 ms @ 48 kHz) |
| 4×   | 24 amostras @ taxa nativa (~0,5 ms @ 48 kHz)  |

Esses valores são exatamente a latência **omitida** em `events.rs:116-122`. A §5 documenta
o impacto correto; F2 é a implementação que falha em reportá-lo.

O item Pending §9 *"Runtime oversample switching (currently init-time only)"* é diretamente
bloqueado por F2: ao implementar a troca em runtime (o `TODO(oversample-rt)`), cada mudança
de fator vai alterar a latência do plugin em 12 ou 24 amostras — e se F2 não estiver
resolvido, o host não será notificado dessa mudança, causando um salto de PDC perceptível.
**F2 é pré-requisito implícito do item §9.**

---

## F3 — Detalhamento do cruzamento com `audio_fidelity_map.md §4` e §9

`audio_fidelity_map.md §4` declara:

> *"Group delay (min-phase): Asymmetric; slight high-frequency transient smearing"*

Isso confirma que o default de produção é **fase mínima**. No entanto, `latency_samples()`
(`resampler.rs:519`) usa `TAPS_PER_PHASE/2 = 32` — a fórmula de atraso de grupo de filtro
**de fase linear**. A seção §4 documenta o comportamento correto mas não menciona a discrepância.

O item Pending §9 *"Resampler quality selector (Standard 32T / HQ 64T)"* adiciona um 2º
banco com 32 taps. Ao implementá-lo, `latency_samples()` precisará distinguir o tipo de
banco **e** a fase (mínima vs linear). F3 deve ser corrigido **antes** desse item ser
implementado, senão a discrepância se duplica.

---

## F9 — Detalhamento do cruzamento com `cpp_parity_map.md §11.1`

`cpp_parity_map.md §11.1` documenta:

> *"**Reset does NOT prewarm on load** — `reset()` is a public API for explicit state
> clearing. Loader calls `prewarm()` separately to preserve LSTM initial states loaded
> from file. C++ `Reset` always calls `prewarm()`."*

F9 é disparado precisamente quando se chama `reset()` ou `process()` sem o subsequente
`prewarm()`. A §11.1 documenta a divergência intencional em relação ao C++ (motivada por
preservar o estado inicial do LSTM), mas **não menciona** que A2 tem um `head_write_pos`
que cresce sem máscara no caminho de `layers.is_empty()`, tornando o invariante frágil
quando `prewarm()` é omitido.

---

## Referências cruzadas obsoletas a corrigir em `cpp_parity_map.md`

> **Problema:** As três linhas abaixo em `cpp_parity_map.md` referenciam
> `TODO-findings.md F-2` com o significado *"LSTM 192 kHz drift"*. Com a reescrita
> do `TODO-findings.md` nesta auditoria, **F2 passou a significar** *"Oversample engine
> latency not reported to host (PDC)"*. As referências estão, portanto, obsoletas.

| Arquivo             | Linha | Texto atual (obsoleto)                                                                   |
| ------------------- | -----:| ---------------------------------------------------------------------------------------- |
| `cpp_parity_map.md` | 163   | `([TODO-findings.md](../TODO-findings.md) F-2), never a hidden gap.`                     |
| `cpp_parity_map.md` | 524   | `§4.5; [TODO-findings.md](../TODO-findings.md) F-2`                                      |
| `cpp_parity_map.md` | 545   | `[TODO-findings.md](../TODO-findings.md) — F-2 (LSTM sample-rate drift), audit findings` |

**Correção:** o drift LSTM @ 192 kHz é uma **limitação conhecida, medida e assertada**
(não um bug novo desta auditoria). Não precisa de ID de finding — está totalmente
documentado inline em `cpp_parity_map.md §4.5` e `§9.1`, com *gate* de teste calibrado
(`ABSOLUTE_ESR_CAP_LSTM`). As três referências devem ser atualizadas para apontar
diretamente para §4.5 / §9.1 do próprio `cpp_parity_map.md`, removendo o link para
`TODO-findings.md` ou atualizando o contexto do link.

---

## Novos itens Pending sugeridos para os docs

Com base nos findings, os itens a seguir **deveriam ser adicionados** às seções
"Pending / Open Work" dos respectivos documentos:

### `audio_fidelity_map.md §9` — novos itens sugeridos

| Item sugerido                                                         | Status       | Finding | Prioridade  |
|:--------------------------------------------------------------------- |:------------:|:-------:|:-----------:|
| Oversampling: latência não reportada ao host (PDC)                    | 🔴 Bug ativo | F2      | MÉDIA       |
| Resampler: fórmula de atraso de grupo incorreta para fase mínima      | 🔴 Bug ativo | F3      | MÉDIA-BAIXA |
| Adaptive Compute: A2 entra em double-pass sem suporte (state corrupt) | 🔴 Bug ativo | F1      | ALTA        |
| Adaptive Compute: `prev_state` dessinc em transições encadeadas       | 🟡 Edge case | F4      | BAIXA-MÉDIA |

### `cpp_parity_map.md §13` — novos itens sugeridos

| Item sugerido                                                                             | Status       | Finding | Prioridade |
|:----------------------------------------------------------------------------------------- |:------------:|:-------:|:----------:|
| A2 Beta: adaptive compute double-pass corrompe `layer_buffer_starts`/`head_write_pos`     | 🔴 Bug ativo | F1      | ALTA       |
| A2 Beta: `head_write_pos` não mascarado em `layers.is_empty()` (OOB se `prewarm` omitido) | 🟡 Latente   | F9      | LATENTE    |
