<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings — Auditoria `revisor-auditor` (foco: "Ace of bug hunting")

> Artefato gerado pela skill `revisor-auditor` (papel _Ace of bug hunting_) e estruturado
> pela skill `planejador-arquiteto`.
> Insumo central: varredura do código-fonte cruzada com `testes.log` (suíte completa:
> 844 testes unitários + integração + parity C++ + soak + clap-validator, **todos passando**).
>
> **Tese da auditoria:** como _todos_ os testes passam, os bugs relevantes são **latentes** —
> escondidos justamente nas regiões onde o _test oracle_ valida o comportamento **errado**
> (ponto fixo de simetria, asserts que codificam o valor bugado) ou onde simplesmente não há
> cobertura de continuidade/linearidade. Esta auditoria persegue exatamente essas regiões.

## Legenda de severidade

| Nível        | Significado                                                                                                                            |
|:------------ |:-------------------------------------------------------------------------------------------------------------------------------------- |
| 🔴 **ALTA**  | Defeito de correção ativo e alcançável em uso normal (artefato de áudio audível, contrato `unsafe` violável, ou estado inconsistente). |
| 🟠 **MÉDIA** | Defeito de correção real, porém alcançável apenas sob condição específica (sobrecarga de CPU, reversão no meio de _fade_, etc.).       |
| 🟡 **BAIXA** | Latente no alvo atual (x86-64), _hardening_, telemetria enganosa, ou clareza de relatório de teste.                                    |
| ⚪ **INFO**  | Invariante frágil / dívida técnica defensiva; correto hoje, arriscado sob refatoração.                                                 |

## Sumário dos achados

| ID   | Severidade | Módulo                       | Título curto                                                                       |
|:---- |:---------- |:---------------------------- |:---------------------------------------------------------------------------------- |
| F-01 | 🔴 ALTA    | `dsp/gate.rs`                | Reversão de _fade_ do gate **inverte** o multiplicador de ganho (`m → 1 − m`)      |
| F-02 | 🟠 MÉDIA   | `dsp/adaptive.rs`            | Curva de _crossfade_ adaptativo só atinge ~0,5 e depois **salta** para 1,0         |
| F-03 | 🟡 BAIXA   | `dsp/resampler.rs`           | _Bypass_ estéreo não valida limites do canal R (contrato `unsafe` inconsistente)   |
| F-04 | 🟡 BAIXA   | `common/spsc/status.rs`      | Handshake flag→dado entre threads usa `Relaxed` (latente fora de x86-TSO)          |
| F-05 | 🟡 BAIXA   | `dsp/adaptive.rs`            | Interrupção de _crossfade_ não é _re-based_ (descontinuidade em degradação rápida) |
| F-06 | 🟡 BAIXA   | `tests/common/validation.rs` | Relatório "Fidelity Margin … ✓" enganoso; alvo de 8 dB nunca é, de fato, exigido   |
| F-07 | 🟡 BAIXA   | `dsp/mirror_buf/alloc.rs`    | `MIRROR_BUF_HUGEPAGE_ACTIVE` marcado `true` no caminho THP-advice (telemetria)     |
| F-08 | ⚪ INFO    | `models/wavenet/conv1d.rs`   | Invariante de _padding_ SIMD protegido só por `debug_assert!` (frágil)             |

---

## F-01 — 🔴 ALTA — Reversão de _fade_ do gate inverte o multiplicador de ganho

### F-01 — Localização

- `src/dsp/gate.rs:183` (transição `FadingOut → FadingIn`)
- `src/dsp/gate.rs:250` (transição `FadingIn → FadingOut`)
- Teste que mascara: `src/dsp/gate_test.rs:89-117` (`test_hysteresis_interrupted_fade`)

### F-01 — Descrição

O FSM de histerese do _noise gate_ representa o ganho instantâneo, em **ambos** os estados de
transição, pela mesma fórmula:

```text
current_multiplier = fade_counter / fade_frames
```

- Em `FadingOut`, `fade_counter` é inicializado em `fade_frames` (gate.rs:162) e **decrementa**
  até 0 → multiplicador cai de 1,0 → 0,0.
- Em `FadingIn`, `fade_counter` inicia em 0 (gate.rs:223) e **incrementa** até `fade_frames` →
  multiplicador sobe de 0,0 → 1,0.

Em ambos, portanto, `multiplier == fade_counter / fade_frames`. Para que uma **reversão**
(sinal retorna no meio do _fade-out_, ou cai no meio do _fade-in_) seja contínua, basta
**preservar** `fade_counter`. Entretanto, o código faz:

```rust
// gate.rs:181-186  (FadingOut → FadingIn)
if value >= threshold_open {
    self.state = GateState::FadingIn;
    self.fade_counter = params.fade_frames.saturating_sub(self.fade_counter); // ❌ inverte
    if self.fade_counter + n_samples < params.fade_frames {
        self.fade_counter += n_samples;
        self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
        ...
```

Como o novo multiplicador passa a ser `(fade_frames − fade_counter) / fade_frames = 1 − m`, a
reversão produz um **salto** de `m` para `1 − m` em vez de continuar suavemente a partir de `m`.

### F-01 — Evidência (o teste valida o comportamento errado)

Em `gate_test.rs` (`fade_frames = 10`):

1. **Primeira reversão** (linha 105-107): estava em `FadingOut` com `multiplier = 0.5`
   (`fade_counter = 5`). Reversão: `10 − 5 = 5` → permanece 5. Como **0,5 é o ponto fixo de
   `m → 1 − m`**, o bug é invisível e o teste assevera `0.6` (correto por coincidência).
2. **Segunda reversão** (linha 114-116): estava em `FadingIn` com `multiplier = 0.8`
   (`fade_counter = 8`). Reversão: `10 − 8 = 2`, decrementa para `1` → `multiplier = 0.1`.
   **O teste assevera explicitamente `assert_eq!(dh.multiplier(), 0.1)`** — ou seja, codifica o
   salto `0.8 → 0.1` como "correto", apesar do comentário dizer "_starts closing immediately_".

O ganho salta de **0,8 para 0,1** num único bloco (rampa de `ramp_start = 0.8` para
`current = 0.1` em `ramp_samples = 1` amostra ⇒ degrau de 0,7 em uma amostra ⇒ _click_ audível).

### F-01 — Impacto

Contradiz diretamente o propósito declarado do módulo (gate.rs:8-9: _"prevent ... audible
artifacts (clicks/zipper noise)"_). Cenário real: nota de guitarra decaindo para o _noise
floor_ → gate inicia fechamento (256 amostras ≈ 5,3 ms no preset padrão) → novo transiente
chega no meio do _fade_ → em vez de subir suavemente para 1,0, o ganho **despenca** para o
complemento e depois sobe ⇒ _click_/_dip_ exatamente no ataque da nova nota.

### F-01 — Causa-raiz

A linha `fade_counter = fade_frames − fade_counter` só faria sentido se a fórmula do
multiplicador **diferisse** entre os estados (ex.: `FadingOut` usar `(ff − fc)/ff`). Como ambos
usam `fc/ff`, a inversão está errada.

### F-01 — Solução proposta

Remover a inversão em **ambas** as reversões (preservar `fade_counter`):

```rust
// gate.rs:181-194  (FadingOut → FadingIn) — CORRIGIDO
if value >= threshold_open {
    self.state = GateState::FadingIn;
    // fade_counter já codifica o multiplicador atual (m = fade_counter/fade_frames);
    // preservá-lo garante continuidade da rampa na nova direção.
    if self.fade_counter + n_samples < params.fade_frames {
        self.fade_counter += n_samples;
        self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
        self.ramp_samples = n_samples;
    } else {
        self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
        self.state = GateState::Open;
        self.current_multiplier = 1.0;
        self.fade_counter = params.fade_frames;
        self.hold_counter = 0;
    }
}
```

```rust
// gate.rs:248-260  (FadingIn → FadingOut) — CORRIGIDO
if value < threshold_close {
    self.state = GateState::FadingOut;
    // idem: preservar fade_counter mantém o multiplicador contínuo
    if self.fade_counter > n_samples {
        self.fade_counter -= n_samples;
        self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
        self.ramp_samples = n_samples;
    } else {
        self.ramp_samples = self.fade_counter;
        self.state = GateState::Closed;
        self.current_multiplier = 0.0;
        self.fade_counter = 0;
    }
}
```

### F-01 — Plano de validação

- **Atualizar o teste para refletir o comportamento correto:** `gate_test.rs:116`
  `assert_eq!(dh.multiplier(), 0.1)` → `0.7` (de 0,8 fechando 1 passo de 0,1). O comentário
  "_starts closing immediately_" passa a ser verdadeiro. A linha 107 (`0.6`) permanece válida
  (ponto fixo).
- **Adicionar teste anti-regressão de continuidade** que reverta fora do ponto médio (ex.:
  reverter em `multiplier = 0.7` e exigir que o próximo valor seja `> 0.7`, não `< 0.5`).
- **Adicionar invariante**: em qualquer reversão, `|novo_multiplier − ramp_start_multiplier| ≤
  n_samples * inv_fade_frames` (sem degrau maior que um passo de rampa).
- Rodar `gate_fsm_proptest` (`tests/gate_fsm_proptest.rs`) com geração de reversões aleatórias.

---

## F-02 — 🟠 MÉDIA — Curva de _crossfade_ adaptativo só atinge ~0,5 e depois salta para 1,0

### F-02 — Localização

- `src/dsp/adaptive.rs:232-248` (`crossfade_multiplier`, versão mutável)
- `src/dsp/adaptive.rs:499-506` (`current_crossfade_multiplier`, versão de leitura)
- Consumo: `src/dsp/pipeline/stages/inference.rs:315-318` e `491` (blend `m_out + t*(new − old)`)
- Teste que não cobre: `src/dsp/adaptive_test.rs:330-372` (`test_crossfade_prev_state_and_multipliers`)

### F-02 — Descrição

`CrossfadePhase::Active { remaining_samples }` captura `remaining_samples` (a **duração total**
do _crossfade_, ex.: 1536 amostras = 32 ms @ 48 kHz, calculada em adaptive.rs:353-357). Porém,
no laço de progresso, **`remaining_samples` nunca é decrementado** — apenas `crossfade_elapsed`
é incrementado:

```rust
// adaptive.rs:232-247
CrossfadePhase::Active { remaining_samples } => {
    let total = remaining_samples + self.crossfade_elapsed;        // ❌ cresce a cada bloco
    let progress = if total == 0 { 1.0 }
                   else { (self.crossfade_elapsed as f32 / total as f32).min(1.0) };
    if self.crossfade_elapsed + frame_advance >= remaining_samples {
        self.crossfade = CrossfadePhase::Idle;
        self.crossfade_elapsed = 0;
        1.0
    } else {
        self.crossfade_elapsed += frame_advance;
        progress
    }
}
```

Com `D = remaining_samples` constante, `total = D + elapsed` e
`progress = elapsed / (D + elapsed)`. Logo:

- `elapsed = 0` → `progress = 0`.
- `elapsed = D` (fim da rampa) → `progress = D / 2D = 0,5`.
- Em seguida a condição de término dispara e retorna **1,0** abruptamente.

Ou seja, o _crossfade_ sobe linearmente-ish apenas até **~0,5** ao longo de toda a duração e,
no último bloco, **salta de ~0,5 para 1,0**.

### F-02 — Evidência (teste não valida linearidade)

`adaptive_test.rs:364-366` só verifica `m1 > 0.0 && m1 < 1.0` (com `elapsed = 512` o valor real
é `512/(1536+512) = 0,25`) e `:371` verifica o término `== 1.0`. **Nenhum** assert checa o ponto
médio nem a monotonicidade da curva — por isso o defeito passa despercebido.

### F-02 — Impacto

`t` é o coeficiente de _blend_ linear entre a saída do modelo **antigo** e do **novo** em
inference.rs:316 (`crossfade_blend_mono_simd(m_out_l, scratch_l, t)` ⇒ `out = old + t*(new −
old)`). Um salto de `t ≈ 0,5 → 1,0` injeta metade da transição **instantaneamente** ⇒
descontinuidade audível (_click_) ao final de **todo** _crossfade_ de degradação/recuperação
adaptativa. Mitigado por ser caminho ativo apenas sob pressão de CPU (modos `Conservative`/
`Aggressive`), mas é justamente quando o usuário menos pode tolerar artefatos.

### F-02 — Solução proposta

Computar o progresso contra a **duração fixa**. Duas opções:

**Opção A (mínima):** decrementar `remaining_samples` junto com o incremento de `elapsed`, de
modo que `remaining_samples + elapsed` permaneça `= D`:

```rust
CrossfadePhase::Active { remaining_samples } => {
    let total = remaining_samples + self.crossfade_elapsed; // agora invariante = D
    let progress = if total == 0 { 1.0 }
                   else { (self.crossfade_elapsed as f32 / total as f32).min(1.0) };
    if self.crossfade_elapsed + frame_advance >= remaining_samples {
        self.crossfade = CrossfadePhase::Idle;
        self.crossfade_elapsed = 0;
        1.0
    } else {
        self.crossfade_elapsed += frame_advance;
        // mantém o invariante total == D:
        self.crossfade = CrossfadePhase::Active {
            remaining_samples: remaining_samples - frame_advance,
        };
        progress
    }
}
```

**Opção B (mais clara, recomendada):** armazenar a duração total como campo dedicado
(`crossfade_total: usize`) e calcular `progress = (crossfade_elapsed as f32 /
crossfade_total as f32).min(1.0)`, eliminando a aritmética `remaining + elapsed` (que é a fonte
da confusão). Renomear `remaining_samples` para deixar a semântica explícita.

> ⚠️ A versão de leitura `current_crossfade_multiplier` (adaptive.rs:499-506) replica a mesma
> fórmula e **deve receber a mesma correção** para permanecer consistente com a versão mutável.

### F-02 — Plano de validação

- Novo teste exigindo **monotonicidade e ponto médio**: ao avançar `D/2` amostras,
  `progress ≈ 0,5 ± ε`; ao avançar `0.9·D`, `progress ≈ 0,9 ± ε`.
- Teste de **ausência de degrau terminal**: o último valor antes do término deve estar próximo
  de 1,0 (ex.: `> 0,9`), não de 0,5.
- Reaproveitar `adaptive_fsm_proptest` para varrer `frame_advance` e `sample_rate` variados.

---

## F-03 — 🟡 BAIXA — _Bypass_ estéreo do resampler não valida limites do canal direito

### F-03 — Localização

- `src/dsp/resampler.rs:489-496` (`process_input`, ramo _bypass_)
- `src/dsp/resampler.rs:513-519` (`process_output`, ramo _bypass_)

### F-03 — Descrição

No _bypass_ estéreo, o número de amostras é derivado **apenas** dos canais esquerdos, mas a cópia
`copy_nonoverlapping` é feita também para o canal direito:

```rust
let n = in_l.len().min(out_l.len());          // ❌ ignora in_r e out_r
unsafe {
    core::ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
    core::ptr::copy_nonoverlapping(in_r.as_ptr(), out_r.as_mut_ptr(), n); // OOB se in_r/out_r < n
}
```

Isto é **inconsistente** com:

- As variantes _mono_ (resampler.rs:537, 560), que usam `.min(out_r.len())`.
- O caminho não-_bypass_ (`process_internal`, resampler.rs:231-232), que usa
  `in_l.len().min(in_r.len())` e `out_l.len().min(out_r.len())`.

### F-03 — Impacto

Atualmente **não disparável**: todos os chamadores reais passam buffers simétricos
(`samples_l[..n]`/`samples_r[..n]` e ambos `MAX_RESAMP_BUF` em
`inference.rs:355-360`). Logo, é um **contrato `unsafe` latente**: se algum chamador futuro
passar buffers assimétricos, ocorre leitura/escrita fora dos limites (UB). Severidade baixa por
não ser alcançável hoje, mas é exatamente o tipo de "função insegura sob certa situação" que a
auditoria deve blindar.

### F-03 — Solução proposta

Incluir os canais direitos no cálculo de `n`, alinhando com as variantes _mono_/não-_bypass_:

```rust
let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
```

### F-03 — Plano de validação

- Teste com buffers **assimétricos** (`in_r` mais curto que `in_l`, `out_r` mais curto que
  `out_l`) que hoje provocaria OOB e, após a correção, deve copiar `n` seguro sem _panic_/UB.
- Rodar sob `tests/resampler_heap_audit.rs` + Miri (se viável) no ramo _bypass_.

---

## F-04 — 🟡 BAIXA — Handshake flag→dado entre threads usa ordenação `Relaxed`

### F-04 — Localização

- `src/common/spsc/status.rs:175-189` (`set_flag`/`clear_flag`/`check_flag` — todos `Relaxed`)
- Produtores: `src/clap/processor/events.rs:216-219`,
  `src/standalone/pw_host/rt_callback/commands.rs:123-125`
- Consumidores: `src/clap/plugin/main_thread/housekeeping.rs:63-69`,
  `src/standalone/pw_host/run.rs:232-233`
- Campos análogos: `requested_slimmable_ch`, `requested_pw_rate`, `requested_nam_rate`,
  `requested_cabsim_partition_size` (status.rs:88-143)

### F-04 — Descrição

Padrão clássico de _flag guards data_: o produtor (thread RT) grava o **dado** e depois **liga a
flag**; o consumidor (thread main) **lê a flag** e depois **lê o dado** — porém todas as
operações usam `Ordering::Relaxed`:

```rust
// events.rs:216-219
self.rt_status.requested_slimmable_ch.store(target_ch as u32, Ordering::Relaxed); // dado
self.rt_status.set_flag(RT_STATUS_NEEDS_SLIMMABLE_REBUILD);                        // flag (fetch_or Relaxed)
```

```rust
// housekeeping.rs / run.rs:232-233
if check_flag(NEEDS_SLIMMABLE_REBUILD) {                 // load Relaxed
    let target = requested_slimmable_ch.load(Relaxed);   // dado (pode estar obsoleto em weak-memory)
}
```

### F-04 — Impacto

- **No alvo atual (x86-64, TSO):** _correto na prática_ — stores não são reordenados entre si,
  loads não são reordenados entre si, e `fetch_or` (LOCK OR) é barreira total. Sem bug ativo.
- **Em porte para memória fraca (AArch64/Apple Silicon):** o `store` do dado e o `set_flag`
  podem ser reordenados; o consumidor pode ver a flag ligada e ler `requested_slimmable_ch`
  **obsoleto/0** ⇒ _rebuild_ com contagem de canais inválida (potencial divisão por zero/modelo
  degenerado). Não há _happens-before_ estabelecido, apesar de o documento de design sugerir que
  há.

### F-04 — Solução proposta

Estabelecer o par _release/acquire_ no handshake que publica/consome dados:

- `set_flag` que publica um dado associado → `fetch_or(flag, Ordering::Release)` (ou inserir
  `fence(Release)` após o `store` do dado e antes do `set_flag`).
- `check_flag` que precede a leitura do dado → `load(Ordering::Acquire)`.

Como `set_flag` é genérico, recomenda-se uma variante explícita
(`set_flag_release` / `check_flag_acquire`) usada **apenas** nos pontos de handshake
flag↔dado (rebuild de resampler, cabsim, slimmable, rate), preservando `Relaxed` nas flags
puramente de telemetria (clipping, silent, etc.). Custo desprezível e elimina a fragilidade.

### F-04 — Plano de validação

- Revisar cada uso de `requested_*` e classificar flag como "telemetria" (mantém `Relaxed`) ou
  "handshake de dado" (passa a `Release`/`Acquire`).
- Teste `loom` (modelo de memória) do par produtor/consumidor de slimmable/resampler, se a
  infraestrutura permitir, para provar ausência de leitura obsoleta sob reordenação.

---

## F-05 — 🟡 BAIXA — Interrupção de _crossfade_ adaptativo não é _re-based_

### F-05 — Localização

- `src/dsp/adaptive.rs:338-358` (`transition_to`)

### F-05 — Descrição

`transition_to` sobrescreve incondicionalmente um _crossfade_ em andamento: define
`prev_state = state` atual e reinicia `crossfade_elapsed = 0` com um novo
`CrossfadePhase::Active`. Como uma transição exige apenas `DEGRADE_CONSECUTIVE = 3` blocos e o
_crossfade_ dura ~1536 amostras (vários blocos), uma segunda transição (`Full→Reduced→Minimal`)
pode ocorrer **antes** de o primeiro _crossfade_ terminar.

### F-05 — Impacto

Ao interromper, a contribuição parcial do estado anterior (ex.: ainda 92% `Full` / 8%
`Reduced`) é descartada e o _blend_ recomeça de `prev_state = Reduced` → descontinuidade no
coeficiente de _blend_. Combina-se com F-02 para piorar o artefato. Severidade baixa: só sob
sobrecarga sustentada de CPU disparando transições consecutivas.

### F-05 — Solução proposta

Ao iniciar nova transição com _crossfade_ ativo, **re-basear** preservando o progresso atual:
em vez de `crossfade_elapsed = 0`, mapear o multiplicador corrente para o novo par
(`prev_state`, `state`) de forma contínua (p.ex. iniciar o novo _crossfade_ a partir do `t`
atual). Idealmente resolver junto com F-02 (mesma unidade algorítmica de _crossfade_).

### F-05 — Plano de validação

- Teste de transições consecutivas (`Full→Reduced` seguido imediatamente de `Reduced→Minimal`)
  exigindo que o coeficiente de _blend_ não apresente degrau `> ε` entre blocos.

---

## F-06 — 🟡 BAIXA — Relatório "Fidelity Margin … ✓" enganoso; alvo de 8 dB nunca exigido

### F-06 — Localização

- `tests/common/validation.rs:240-248`

### F-06 — Descrição

```rust
let delta_snr = snr - anchor_snr_db;
let is_satisfactory = delta_snr > 8.0 || snr >= min_snr_db;   // ❌ OR torna o alvo de 8 dB decorativo
println!("  Fidelity Margin = {delta_snr:.1} dB (target > 8.0 dB) {}",
         if is_satisfactory { "✓" } else { "?" });
```

Como `is_satisfactory` é `delta_snr > 8.0 || snr >= min_snr_db`, e o gate **rígido** já é
`assert!(snr >= min_snr_db)` (validation.rs:256-259), o `✓` aparece **sempre** que o teste
passa — mesmo com margem de fidelidade **abaixo** (ou negativa) do alvo declarado.

### F-06 — Evidência (`testes.log`)

Diversas linhas exibem `✓` com margem muito abaixo de 8 dB:

- `testes.log:2700` → `Fidelity Margin = 0.5 dB (target > 8.0 dB) ✓`
- `testes.log:2869` → `Fidelity Margin = 0.4 dB (target > 8.0 dB) ✓`
- `testes.log:3072` → `Fidelity Margin = -0.8 dB (target > 8.0 dB) ✓` (margem **negativa**)
- `testes.log:2829/3107/3330/3714` → 5,4 / 6,2 / 5,9 / 3,0 dB, todas `✓`

### F-06 — Impacto

A métrica "Fidelity Margin > 8 dB" foi concebida como _gate_ de qualidade perceptual (quão acima
o modelo está de uma âncora degradada), mas, pela disjunção, **nunca é, de fato, exigida**. Um
modelo com SNR logo acima de `min_snr_db` porém com margem ~0 dB (perceptualmente próximo da
âncora crua) passa com `✓`, gerando falsa confiança. É um defeito de **eficácia de teste**
(_test oracle_), não de runtime de produção.

### F-06 — Solução proposta

Escolher uma semântica e torná-la honesta:

- **(a)** Se a margem de 8 dB é um requisito: transformá-la em _gate_ (rígido ou _soft_, como o
  MR-STFT) e falhar/avisar quando `delta_snr ≤ 8.0` apesar de `snr ≥ min_snr_db`.
- **(b)** Se é apenas informativa: trocar o rótulo para refletir o **critério real** — p.ex.
  `Fidelity Margin = X dB (SNR gate: ✓/✗ vs min Y dB)` — de modo que `✓` corresponda ao que foi
  efetivamente verificado, e não ao alvo de 8 dB não atingido.

### F-06 — Plano de validação

- Garantir que a mudança não quebre os _parity tests_ existentes (ajustar limiares por modelo se
  optar por (a)).

---

## F-07 — 🟡 BAIXA — `MIRROR_BUF_HUGEPAGE_ACTIVE` marcado `true` no caminho THP-advice

### F-07 — Localização

- `src/dsp/mirror_buf/alloc.rs:226-231` (caminho padrão, não-HugeTLB)

### F-07 — Descrição

No caminho não-HugeTLB, após `madvise(MADV_HUGEPAGE)` (uma **dica**, não garantia de páginas de
2 MiB), o código faz `MIRROR_BUF_HUGEPAGE_ACTIVE.store(true, Relaxed)` — **idêntico** ao caminho
HugeTLB genuíno (`try_new_huge_aligned`, alloc.rs:117-120). A telemetria/diagnóstico, portanto,
não distingue huge pages reais de mero _advise_ THP.

### F-07 — Impacto

Apenas observabilidade: relatórios de diagnóstico (`common/diagnostics`) podem reportar
"hugepage ativo" mesmo quando o kernel não promoveu as páginas. Não afeta correção de áudio nem
RT-safety.

### F-07 — Solução proposta

Distinguir os dois estados — p.ex. um enum/flag separada (`HugepageKind::Hugetlb` vs
`HugepageKind::ThpAdvised`), ou só marcar `MIRROR_BUF_HUGEPAGE_ACTIVE = true` no caminho HugeTLB
e expor a dica THP via contador/telemetria distinta.

### F-07 — Plano de validação

- Verificar consumidores de `MIRROR_BUF_HUGEPAGE_ACTIVE` (telemetria/diagnóstico) e ajustar os
  textos/relatórios correspondentes.

---

## F-08 — ⚪ INFO — Invariante de _padding_ SIMD protegido só por `debug_assert!`

### F-08 — Localização

- `src/models/wavenet/conv1d.rs:106-108, 122-124, 138-140` (e `process_single_frame`, linhas 212,
  226, 240)
- `src/models/wavenet/conv1d_dual.rs`, `conv1d_dyn.rs`, `conv1d_dyn_dual.rs` (mesmo padrão)
- Invariante produzido por: `src/loader/dispatcher/wavenet/layout.rs:22-60` +
  `select_interleave_width` (layout.rs:356-364)

### F-08 — Descrição / Verificação

Os kernels fazem `core::slice::from_raw_parts(ptr.add(w_start), K*IN)` sobre `[f32; width]`,
lendo `num_blocks * K * IN * width` floats. **Verificou-se que isto é seguro hoje:**
`select_interleave_width` garante `out_ch` múltiplo de 16/8 ou cai para 4 com _padding_
explícito, e o loader aloca `AlignedVec::new(padded_total)` exatamente igual ao total lido pelo
kernel. Não há OOB.

**Risco (INFO):** a única proteção em _release_ é a **consistência** entre `select_interleave_width`
usado no loader (`layout.rs`) e no kernel (`conv1d.rs`); as checagens de comprimento são
`debug_assert!` (compiladas fora em _release_). Uma alteração unilateral de
`select_interleave_width` (ou da fórmula de `padded_total`) viraria OOB silencioso.

### F-08 — Solução proposta

- Centralizar o cálculo de `padded_total`/`num_blocks` em **uma** função compartilhada por loader
  e kernel (fonte única de verdade), e/ou
- Promover a checagem `weights.len() >= padded_total` a `assert!` (em _release_) no construtor do
  `Conv1d` — custo desprezível por ser caminho frio (carga de modelo).

### F-08 — Plano de validação

- Teste que constrói `Conv1d` com buffer subdimensionado e exige `panic` controlado (não UB) em
  _release_.

---

## Épicos (agrupamento para resolução otimizada)

> Ordenados por relação risco×esforço. Referências apontam para os achados acima.

## E1 — Correção de artefatos de áudio no caminho de transição (DSP) 🔴 [TO-DO]

**Achados:** F-01, F-02, F-05.
**Justificativa de agrupamento:** os três são defeitos de **continuidade de ganho/blend** em
transições suaves (gate e crossfade adaptativo). F-02 e F-05 vivem na mesma unidade algorítmica
(`adaptive.rs`) e devem ser corrigidos juntos; F-01 é independente mas da mesma natureza
(reversão de _fade_). Compartilham estratégia de teste: **invariantes de não-descontinuidade**
(nenhum degrau maior que um passo de rampa entre blocos).
**Risco/atenção:** ALTO — corrigir F-01 e F-02 **exige atualizar asserts de teste que codificam
o comportamento bugado** (`gate_test.rs:116`; novos asserts de linearidade em
`adaptive_test.rs`). Validar que nenhuma regressão de _parity_ surge.
**Ordem sugerida:** F-01 → F-02 → F-05.

## E2 — Endurecimento de contratos `unsafe` e ordenação de memória 🟡 [DOING]

**Achados:** F-03, F-04, F-08.
**Justificativa de agrupamento:** todos são _hardening_ defensivo (sem bug ativo em x86-64):
contrato `unsafe` do _bypass_ do resampler (F-03), _happens-before_ dos handshakes RT↔main
(F-04) e invariante de _padding_ SIMD (F-08). Conjunto ideal para uma _sprint_ de robustez/portabilidade.
**Risco/atenção:** MÉDIO — F-04 deve diferenciar flags de **telemetria** (mantêm `Relaxed`) de
flags de **handshake de dado** (passam a `Release`/`Acquire`) para não onerar o _hot-path_.
**Ordem sugerida:** F-03 → F-08 → F-04.

## E3 — Honestidade de telemetria e _test oracles_ 🟡 [DOING]

**Achados:** F-06, F-07.
**Justificativa de agrupamento:** ambos são "relatórios que mentem" — o _gate_ de fidelidade que
sempre marca `✓` (F-06) e a flag de hugepage que não distingue THP-advise de HugeTLB (F-07).
Baixo risco, alto valor de confiabilidade observacional.
**Risco/atenção:** BAIXO — F-06 pode exigir recalibração de limiares se a margem virar _gate_.
**Ordem sugerida:** F-06 → F-07.

---

### Notas de método

- A suíte (`testes.log`) está **100% verde**; portanto cada achado foi confirmado por
  **leitura/derivação manual** do código e, quando aplicável, demonstrando **por que o teste
  existente não detecta** o defeito (ponto fixo de simetria em F-01; ausência de checagem de
  linearidade em F-02; disjunção que neutraliza o alvo em F-06).
- Severidades de F-04 calibradas ao alvo real do projeto (**x86-64**, com TSO). Itens marcados
  como latentes **não** são bugs ativos na plataforma suportada hoje.
- Falsos-positivos descartados durante a auditoria (registrados para rastreio): caminho RT do
  processador CLAP/PipeWire (sem alocação/lock/panic no _hot-path_); _parser_ `.namb`
  (cabeçalho `repr(C, packed)` de 80 bytes, CRC cobre corretamente o intervalo, bem _fuzzed_);
  kernels SIMD (todos com tratamento de cauda e `loadu`, _dispatch_ por ISA correto).
