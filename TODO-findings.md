<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# TODO-findings.md — Auditoria de Compliance & Paridade (2026-07-11)

Achados da auditoria focada na role **Compliance and Parity Auditor** (skills
`revisor-auditor` + `pesquisador-inovador`), cobrindo: paridade com o NAMcore
(`tests/fixtures/NeuralAmpModelerCore/`), gestão de fixtures, scripts de QA
(`utils/tests-quick.sh`, `utils/tests-long.sh`, `utils/quality-dashboard.sh`) e o
código Rust dos próprios testes. Fontes de evidência: leitura pareada C++↔Rust,
`/testes.log` (execução real de 2026-07-11) e inspeção linha-a-linha dos scripts.

**Fora de escopo desta auditoria:** `TODO-wavenet_a2_max.md` (Bugs A/B/C do A2 dinâmico) —
tratado à parte. Nenhum achado abaixo depende dele; onde houver contato, está anotado.

Convenções de severidade:

- 🔴 **CRÍTICO** — falso senso de segurança ou saída de áudio incorreta possível.
- 🟠 **ALTO** — dado incorreto exibido, cobertura de paridade ausente ou gate inoperante.
- 🟡 **MÉDIO** — divergência real com impacto limitado/contido.
- 🟢 **BAIXO** — higiene, robustez, clareza.

---

## GRUPO A — Paridade com o NAMcore (código de produção)

### F-A1 🟠 ConvNet: colisão de arquitetura com o NAMcore — zero cross-validação C++

**Evidência (C++):** `tests/fixtures/NeuralAmpModelerCore/NAM/convnet.cpp`

- `parse_config_json` (linha ~326): config plano — `channels: int` único,
  `dilations: [int]` (um bloco por dilation), `batchnorm: bool`, `activation` único,
  `kernel_size` fixo = 2 (hardcoded em `ConvNetBlock::set_weights_`, linha ~57).
- `BatchNorm::BatchNorm` (linhas 14-37): lê do stream de pesos os parâmetros **crus**
  `running_mean[ch], running_var[ch], gamma[ch], beta[ch], eps` (4·ch+1 floats) e funde
  em scale/loc no load. O bias da conv só existe quando `!batchnorm`
  (`this->conv.set_size_and_weights_(..., !batchnorm, ...)`, linha 57).

**Evidência (Rust):**

- `src/loader/nam_json/topology/convnet.rs:36-121`: exige config `layers: [{channels,
  kernel_size, dilations, activation}]` per-block. O formato plano do C++ retorna `None`.
- `src/loader/dispatcher/convnet/mod.rs:49,86-93`: `do_bias = true` **sempre**; lê BN
  **pré-fundido** (`bn_scale[out]` + `bn_offset[out]`, 2·ch floats).

**Consequência confirmada:**

1. Um `.nam` ConvNet real produzido pelo trainer oficial (formato C++) **não carrega**
   no nam-rs (topologia não detectada) — e se carregasse, o stream de pesos ficaria
   desalinhado (bias lendo `running_mean`, etc.).
2. O `convnet_test.nam` do nam-rs derruba o `render` C++
   (`json.exception.type_error.302` — vide `/testes.log` linha 2436-2441).
3. **Zero cross-validação NAMcore para ConvNet**: `quick_parity_convnet` SKIPa em
   runtime, `golden_gen_build.sh:362-364` pula a geração de golden, dashboard exibe
   "vs NAMcore: N/A". As únicas testemunhas de correção são o oráculo f64 + âncora
   NumPy (geradas do MESMO spec bespoke — validação circular quanto ao formato).

**Proposta de solução (decisão de produto + implementação):**

- **Opção A (paridade plena, recomendada):** implementar o formato ConvNet do NAMcore
  como caminho adicional de load: (1) em `topology/convnet.rs`, detectar o config plano
  (`channels` escalar + `dilations` no topo + `batchnorm` bool) e sintetizar a topologia
  equivalente (um bloco por dilation, kernel=2); (2) no dispatcher, quando o config for
  "formato C++", ler pesos na ordem C++ (bias condicional a `!batchnorm`; BN cru
  4·ch+1 floats, fundindo `scale = gamma/sqrt(eps+var)`, `offset = beta − scale·mean`
  no load — reutilizar `models::convnet::batch_norm::from_fused`/inverso); (3) gerar
  fixture `.nam` no formato C++ + golden via `render` (destravar o SKIP do
  `golden_gen_build.sh`) e ativar `quick_parity_convnet` de verdade.
- **Opção B (mínima):** declarar oficialmente o ConvNet nam-rs como **extensão
  proprietária** (renomear architecture p/ ex. `"ConvNetRs"` ou documentar
  incompatibilidade no loader com erro claro ao detectar o formato C++), remover
  `convnet_test.nam` do CATALOG `v2_scope=all` (hoje enganoso) e remover
  `quick_parity_convnet` do quick set.
- Em ambas: atualizar `docs/cpp_parity_map.md` §6 (já anotado nesta auditoria).

> **Nota do PO:** Seguiremos pela "Opção A (paridade plena, recomendada)". Não tenho interesse em implementação paralela. Desative-a em favor da oficial.

**Arquivos:** `src/loader/nam_json/topology/convnet.rs`,
`src/loader/dispatcher/convnet/mod.rs`, `tests/fixtures/golden_gen_build.sh` (CATALOG),
`tests/parity/cpp_parity.rs`, `tests/fixtures/generate_b1_2_fixtures.py`.

---

### F-A2 🟡 `sample_rate: -1.0` (sentinela NAMcore de "unknown") é rejeitado pelo loader

**Evidência:** C++ `get_dsp.cpp:280-286` define `NAM_UNKNOWN_EXPECTED_SAMPLE_RATE -1.0`
e aceita `-1.0` explícito no JSON. Rust `src/loader/nam_json/validation.rs:463-474`
rejeita qualquer `sample_rate <= 0.0` com `InvalidSampleRate`.

**Consequência:** modelos antigos/exportados com `"sample_rate": -1.0` explícito
carregam no NAMcore e falham no nam-rs.

**Proposta:** no visitor de `sample_rate`, mapear exatamente `-1.0` (e opcionalmente
qualquer valor `<= 0.0` finito) para `None` ("unknown", mesmo default do campo ausente),
mantendo a rejeição de NaN/±Inf. Adicionar testes: `-1.0 → None`, `0.0 → None` (ou
rejeição documentada, se preferirem fail-closed p/ 0.0 — decidir e comentar citando
`get_dsp.cpp`).

**Arquivos:** `src/loader/nam_json/validation.rs`,
`src/loader/nam_json/validation_test.rs`.

---

### F-A3 🟠 WaveNet A1 catálogo: `condition_dsp` não é checado na classificação de SKU

**Evidência:** `docs/cpp_parity_map.md` §3.4 (verificado em
`src/loader/nam_json/topology/wavenet.rs`): um modelo 2-arrays, ungated, CH=16,
dilations Standard **com** `condition_dsp` no JSON ainda casa `Known(Standard)` e vai
para o fast path `WaveNetModel<16,3,8>`, que **não tem nenhum tratamento** de
`condition_dsp` — o sub-modelo é silenciosamente descartado (áudio errado, sem erro).

**Proposta:** em `get_wavenet_topology`, adicionar `data.config.condition_dsp.is_some()`
à lista de desqualificadores de catálogo (mesma técnica já usada p/ `gated`), roteando
para `WaveNetModelDyn` (que suporta condition_dsp). Adicionar teste de topologia:
"catalog-shaped + condition_dsp → Free/Dyn".

**Arquivos:** `src/loader/nam_json/topology/wavenet.rs` + testes de topologia.

---

### F-A4 🟠 Caminho A1 Free/Dynamic ignora silenciosamente `gated`/`gating_mode`/FiLM/`head1x1`/`layer1x1`

**Evidência:** `docs/cpp_parity_map.md` §3.6 (confirmado): o C++ genérico
(`model.cpp:976-1151`) parseia gating/FiLM/head1x1/layer1x1 para QUALQUER WaveNet;
o `NamLayerConfig` Rust só parseia o boolean legado `gated`, e apenas para
desqualificar SKU — o caminho `Free`/`WaveNetModelDyn` nunca lê esses campos e processa
como se `gated=false` (saída matematicamente errada, sem erro).

**Proposta (fail-closed primeiro, funcionalidade depois):**

1. **Curto prazo:** no parser/topologia, detectar presença de
   `gating_mode != "none"` / `gated: true` / `head1x1` / `layer1x1` com `groups != 1` /
   qualquer slot FiLM ativo em modelos que NÃO se classificam como A2 e rejeitar com
   erro claro ("feature X requires the A2 dynamic engine; model rejected to protect
   audio correctness").
2. **Médio prazo:** avaliar rotear esses modelos para `WaveNetA2Dyn` (que já implementa
   gating/FiLM) quando a geometria permitir — reaproveitando o motor em vez de duplicar.

**Atenção:** NÃO tocar no predicado `is_disabled_broken_a2_flagship` nem nos épicos do
`TODO-wavenet_a2_max.md`.

**Arquivos:** `src/loader/nam_json/model.rs`, `src/loader/nam_json/topology/wavenet.rs`,
`src/loader/dispatcher/wavenet/dynamic.rs`.

---

### F-A5 🟡 `WaveNetModelDyn::prewarm_internal` chama `cond_dsp.prewarm(0)` — errado p/ condition_dsp LSTM

**Evidência:** `src/models/wavenet/model_dyn.rs:348-350` (`cond_dsp.prewarm(0)`
hardcoded). Correto apenas p/ sub-modelos feedforward (que ignoram o argumento);
errado p/ LSTM (recorrente — `prewarm(0)` = zero iterações, estado h/c cru).
Configuração LSTM-as-condition_dsp é legítima no NAMcore.

**Proposta:** trocar por `cond_dsp.prewarm(cond_dsp.prewarm_samples())` (ou
`.max(1)`), com comentário citando `NAM/dsp.cpp::prewarm` e
`lstm.cpp::GetPrewarmSamples`. Adicionar fixture sintética
`wavenet_condition_lstm.nam` (condition_dsp = LSTM 1×3) + golden C++ no CATALOG para
cobrir o caso permanentemente.

**Arquivos:** `src/models/wavenet/model_dyn.rs`,
`tests/fixtures/generate_*.py`, `tests/fixtures/golden_gen_build.sh`.

---

### F-A6 🟢 `prewarm_samples()` do WaveNet sub-reporta (inerte hoje, mina futura)

**Evidência:** `docs/cpp_parity_map.md` §3.5: `WaveNetModel::prewarm_samples()` retorna
só o RF do array1; `WaveNetModelDyn` usa `.max()` em vez da **soma** do C++
(`model.cpp:615-620`). Hoje inerte (o `prewarm()` analítico descarta o argumento), mas
se algum dia for exposto como latência ao host, reporta valor errado.

**Proposta:** corrigir para a soma dos RFs de todos os arrays + prewarm do
condition_dsp + RF do post-stack head (fórmula C++), OU marcar `#[deprecated]` com
comentário explicando por que o valor errado é inofensivo hoje. Preferência: corrigir
(custo trivial, elimina a mina).

**Arquivos:** `src/models/wavenet/model.rs`, `src/models/wavenet/model_dyn.rs`.

---

### F-A7 🟡 `LstmModelDyn`: `num_layers == 0` sem guarda (UB em release) e canais não validados

**Evidência:** `docs/cpp_parity_map.md` §2.6 (confirmado):
`get_lstm_topology` não tem lower bound p/ `num_layers`;
`LstmModelDyn::process_*` deref `self.layers.as_mut_ptr()` guardado só por
`debug_assert!`. C++ trata `num_layers==0` como passthrough bem definido. Além disso,
`in_channels`/`out_channels` do LSTM não são validados (assumido mono implicitamente).

**Proposta:** (1) rejeitar `num_layers == 0` na detecção de topologia com erro claro
(fail-closed; nenhum modelo real existe com 0 camadas — não vale implementar o
passthrough); (2) validar `in_channels == 1 && out_channels == 1` quando presentes no
config (mesmo padrão do `topology/a2.rs`); (3) testes de rejeição correspondentes.

**Arquivos:** `src/loader/nam_json/topology/lstm.rs`, `src/models/lstm/model_dyn*`.

---

## GRUPO B — Rigor do arsenal de testes (as "não-vacas-sagradas")

### F-B1 🔴 `cpp_parity::run_v1`/`run_v1_hf` descartam o `ParityOutcome` — SKIPs silenciosos reportam `ok`

**Evidência:** `tests/parity/cpp_parity.rs:558-581`:

```rust
fn run_v1(...) { let _ = run_render_comparison(...); }
```

Qualquer outcome de skip (toolchain ausente, modelo ausente, render crashou, saída
inválida) imprime `SKIP:` no stderr e o teste passa. Os 6 `quick_parity_*`
não-ignorados do `tests-quick.sh` Fase 2 podem ficar 100% verdes com **zero**
cross-validações executadas. Prova viva no `/testes.log` linha 2436-2442: o render
crashou para ConvNet e `quick_parity_convnet ... ok`. Em contraste,
`run_v2_multi_sr_impl` (linhas 590-676) faz o certo: rastreia outcomes e assevera o
conjunto de rates completados.

**Proposta:**

1. `run_v1`/`run_v1_hf` retornam o `ParityOutcome` e cada teste decide a política:
   - `quick_parity_{lstm_1x16,wavenet_ch16,a2_full}` (e HF): se o toolchain C++ estiver
     disponível (render compilado/compilável), **exigir** `Completed` via
     `assert_eq!(outcome, ParityOutcome::Completed)`. Se o toolchain não existir,
     permitir skip mas **imprimir SKIP-COVERAGE padronizado** (mesmo padrão já usado
     em outros testes) para o grep do tests-long/dashboard.
   - `live_cross_validation_*` v1 (`#[ignore]`, rodam no tests-long onde o toolchain é
     pré-requisito verificado na Phase 0): exigir `Completed` incondicionalmente.
2. Adicionar meta-teste em `threshold_calibration.rs` (estilo dos existentes) que
   escaneia `cpp_parity.rs` e falha se algum wrapper usar `let _ = run_render_comparison`.

**Arquivos:** `tests/parity/cpp_parity.rs`, `tests/models/threshold_calibration.rs`.

---

### F-B2 🟠 `quick_parity_convnet` é estruturalmente um placebo — remover ou transformar em asserção de contrato

**Evidência:** F-A1 + F-B1: o teste sempre SKIPa (incompatibilidade arquitetural
conhecida e documentada em `generate_b1_2_fixtures.py:37-45`) e sempre passa.

**Proposta:** enquanto F-A1 não for resolvido, substituir por
`quick_parity_convnet_expected_incompatible`: assevera que o outcome é exatamente
`SkippedGarbageOutput`/exit!=0 **com** a mensagem nlohmann esperada — transformando o
skip em contrato explícito (se um dia o C++ passar a aceitar o formato, o teste falha e
avisa que a cross-validação real deve ser ligada). Se F-A1 Opção A for implementada,
substituir pelo quick_parity real.

> **Nota do PO:** A respeito de F-A1, escolhi a "Opção A (paridade plena, recomendada)". Ajuste-se para esta decisão.

**Arquivos:** `tests/parity/cpp_parity.rs`.

---

### F-B3 🟠 `wavenet_a2_film_input_mixin_pre`: golden nunca gerado + thresholds placeholder

**Evidência:** modelo existe (`generate_a2_fixtures.py`), CATALOG tem a entrada
(`golden_gen_build.sh:332`), âncora NumPy existe e passa (5.00e-16), mas
`golden_wavenet_a2_film_input_mixin_pre.bin` não existe no disco nem no manifest;
`tests/common/validation.rs:692-696` tem thresholds "XXX pending" genéricos; o teste
golden está `#[ignore]` (log linha 1785).

**Proposta:** rodar `golden_gen_build.sh` para gerar o golden v1; medir; calibrar
thresholds com comentário `// Measured:` (Regra 3 da Gate Calibration Policy);
desativar o `#[ignore]`. Adicionar também `live_cross_validation_*` correspondente
(hoje inexistente — vide F-B6).

**Arquivos:** `tests/fixtures/golden_gen_build.sh` (executar),
`tests/common/validation.rs`, `tests/models/golden_vectors.rs`,
`tests/parity/cpp_parity.rs`.

---

### F-B4 🟡 Gates frouxos demais vs medições atuais — recalibrar sob as Regras 1-4

**Evidência (medições de 2026-07-11 no `/testes.log`):**

| Entrada (`tests/common/validation.rs`) | Gate atual                 | Medido              | Folga                |
| -------------------------------------- | -------------------------- | ------------------- | -------------------- |
| `wavenet_official` (linha ~577)        | SNR ≥ 14 dB / ESR < 3.5e-2 | 130.5 dB / 8.96e-14 | ~116 dB / 12 ordens  |
| `BossLSTM-1x16` golden (linha ~588)    | SNR ≥ 12 dB / ESR < 6.5e-2 | 108.5 dB / 1.42e-11 | ~96 dB / 9 ordens    |
| `lstm_2x8` golden                      | SNR ≥ 18 dB / ESR < 2.0e-2 | 107.8 dB / 1.67e-11 | ~90 dB / 9 ordens    |
| `lstm_official`                        | SNR ≥ 22 dB / ESR < 6.0e-3 | 120.8 dB / 8.30e-13 | ~99 dB / 9 ordens    |
| `wavenet_a2_film_chaos_stress`         | MR-STFT < 0.498            | 7.32e-6             | no teto anti-placebo |

Nota: parte desses gates foi calibrada na era Fast/f16c. Com `Standard` como default
universal, os valores medidos colapsaram ~9-12 ordens de magnitude, e os gates ficaram
sem poder de detecção de regressões realistas (uma regressão de 60 dB passaria).

**Proposta:** recalibração em lote, modelo a modelo, seguindo a Gate Calibration Policy:
novo limite = medido × margem (sugestão: 10×-100× em ESR, −10 a −15 dB em SNR — margem
justificada pela variância inter-máquina/ISA), com comentário `// Measured:` atualizado
citando `/testes.log` 2026-07-11 ou reexecução local. **Cuidado:** manter caps por
sample-rate do LSTM (a sensibilidade a SR é real); recalibrar apenas os pisos de 48 kHz.
Para o `chaos_stress`, avaliar reduzir o gate MR-STFT de 0.498 (teto do anti-placebo)
para valor com lastro na medição (7.3e-6 × margem generosa).

**Arquivos:** `tests/common/validation.rs`, `tests/common/constants.rs`.

---

### F-B5 🟢 Gates MSE = 1e30 (placebo documentado) — trocar por semântica explícita

**Evidência:** `tests/common/validation.rs:623,632,778` (A2 Full/Lite/a2_example) usam
`mse_limit = 1e30` com exceção codificada no anti-placebo
(`threshold_calibration.rs:306-317`, Rule 3: permitido se SNR ≥ 40 e ESR < 0.1).
Intencional e documentado, mas o "1e30" imprime `threshold < 1.0e30` no log — ruído
visual que sugere gate quebrado (foi flagrado por esta auditoria como suspeita).

**Proposta:** introduzir `MseGate::NotApplicable` (enum ou `Option<f64>`) no
`report_dsp_fidelity`, imprimindo `MSE = x (gate: N/A — ESR is the gate)` em vez de
`< 1.0e30`. O meta-teste anti-placebo passa a validar a variante explícita.

**Arquivos:** `tests/common/validation.rs`, `tests/models/threshold_calibration.rs`.

---

### F-B6 🟡 Cobertura live C++ ausente para fixtures do motor dinâmico A2 e coerência CATALOG→testes unidirecional

**Evidência:** `docs/cpp_parity_map.md` §4.7: FiLM/gated/blended/chaos_stress não têm
`live_cross_validation_*` (só goldens congelados). `meta_coherence.rs` verifica
testes→CATALOG mas não CATALOG→testes (modelos órfãos como
`wavenet_a2_film_chaos_stress` não são detectados).

**Proposta:** (1) adicionar `live_cross_validation_{a2_dynamic_gated,a2_dynamic_blended,
wavenet_a2_film_full,wavenet_a2_film_lite}` v1 (`#[ignore]`, tests-long Phase 3) — o
`render` C++ já suporta esses modelos (goldens existem); (2) estender `meta_coherence`
com a direção inversa: todo CATALOG entry com golden declarado deve ter ≥ 1 teste
consumidor (lista de exceções documentada).

**Arquivos:** `tests/parity/cpp_parity.rs`, `tests/models/meta_coherence.rs`.

---

### F-B7 🟢 Determinismo do ConvNet com tolerância 1e-6 (os demais exigem MSE == 0.0)

**Evidência:** `tests/models/golden_vectors.rs:1948-1954`.

**Proposta:** endurecer para MSE == 0.0 (bitwise) como os demais; se falhar,
investigar a fonte de não-determinismo (seria um achado real, não motivo de
tolerância).

**Arquivos:** `tests/models/golden_vectors.rs`.

---

### F-B8 🟢 Âncoras NumPy uniformes em 5.00e-16 — documentar o piso (não é bug)

**Evidência:** `test_oracle_vs_python_anchor_*` = exatamente 5.00e-16 (-153 dB) para
A2/FiLM/ConvNet/WaveNet e 3.49e-30 para LSTM. Verificado: `compute_esr_f64`
(`src/testing/reference_oracle/mod.rs:225-242`) **não** tem clamp/floor — 5.00e-16 é o
piso natural de acumulação f64 (≈2×ε_f64 sobre a janela), e o LSTM é menor por caminho
quase bit-exato + maior energia de sinal.

**Proposta:** documentar em `docs/perceptual_validation.md` (seção do oráculo) que o
valor ≈5e-16 é o piso f64 esperado das âncoras e que valores muito menores indicam
caminho bit-exato — evita que futuros auditores suspeitem de clamp artificial. Sem
mudança de código.

---

## GRUPO C — Scripts de QA (`utils/`)

### F-C1 🔴 `tests-quick.sh::_cargo_meas` corrompe flags libtest — `--test-threads=1` do `isa_parity` é perdido

**Evidência:** `utils/tests-quick.sh:260`:

```bash
local libtest_args="${extra_args#-- }"
```

Remove o prefixo `"-- "`, mas o primeiro flag também começa com `--`:
`"-- --test-threads=1 --nocapture"` → `"-test-threads=1 --nocapture"`. O libtest trata
`-test-threads=1` como **filtro de nome** (não casa nada) — o `--test-threads=1`
exigido pelo `isa_parity` (comentário na linha 326, §7 do testing.md) nunca é aplicado;
os testes rodam em paralelo violando a exigência de serialização
(`TEST_ISA_OVERRIDE` é estado global de processo).

**Proposta:** parar de fazer string-surgery; usar arrays bash:

```bash
# chamada: _cargo_meas "<targets>" "<filters>" --test-threads=1 --nocapture
local -a libtest_args=("${@:3}")
cargo test --release $_ep_flags -- $_filters "${libtest_args[@]}"
```

Adicionar auto-verificação: se `$libtest_args` contiver item começando com `-` mas não
`--`, abortar com erro de uso. Testar manualmente que `isa_parity` roda serializado
(observável pela ordem de output com `--nocapture`).

**Arquivos:** `utils/tests-quick.sh`.

---

### F-C2 🟠 Dashboard: MR-STFT impresso como `0.0000` (perda total da informação)

**Evidência:** `utils/quality-dashboard.sh:921-924` — valores como `9.7262e-6` têm
len>6 e são reformatados com `_nfmt "%.4f"` → `0.0000`. Toda a coluna MR-STFT do
dashboard real (linhas 2538-2565 do `/testes.log`) exibe `0.0000`.

**Proposta:**

```bash
if [[ "$mrstft" =~ [eE] ]]; then
    mrstft_short=$(_nfmt "%.2e" "$mrstft")   # mantém notação científica
elif [ ${#mrstft_short} -gt 8 ]; then
    mrstft_short=$(_nfmt "%.4f" "$mrstft")
fi
```

Aplicar o mesmo critério (`=~ [eE]` ⇒ `%.2e`) às demais colunas numéricas que hoje têm
thresholds de reformatação por comprimento (F-C5).

**Arquivos:** `utils/quality-dashboard.sh`.

---

### F-C3 🟡 Dashboard: `PHASE_TOTAL=8` mas só existem 7 chamadas `phase()` — "[8/8]" nunca aparece

**Evidência:** `utils/quality-dashboard.sh:1301-1342` — a fase "Parseando resultados"
é contada duas vezes (uma no grupo fidelity `+6`, outra no bench `+2`) mas só há uma
chamada `phase()`. Log real termina em `[7/8]`.

**Proposta:** contar apenas fases reais (`+5` fidelity-run, `+1` bench-run,
`+1` parse, `+1` render se for wrappear o `render_dashboard` em `phase()`), ou
simplesmente calcular `PHASE_TOTAL` a partir de uma lista declarativa de fases. Em
`_lib.sh::phase()`, usar `${PHASE_TOTAL:-?}` para não exibir `[1/0]` quando não setado.

**Arquivos:** `utils/quality-dashboard.sh`, `utils/_lib.sh`.

---

### F-C4 🟡 Dashboard: fuzzy-matching de família f64 (`_lookup_esr_f64`) pode atribuir o ESR da família errada

**Evidência:** `utils/quality-dashboard.sh:124-185` — normalização remove tudo que não
é `[a-z0-9]` e faz substring-match bidirecional com desempate por comprimento
(`best_len`). `"waveneta1standard"` casa com a chave `"wavenet"` E com
`"waveneta2lite"`-like keys; o desempate por comprimento pode escolher a família errada.
Hoje o efeito é mitigado pelo sufixo `~fam.` exibido, mas o número pode vir da fixture
errada.

**Proposta:** substituir o fuzzy-match por um **mapa explícito** golden-label →
fixture-f64 (tabela declarativa no topo do script, ex.:
`ESR_F64_FAMILY_MAP["BossWN-standard"]="wavenet_official.nam"`), com fallback `N/A` em
vez de melhor-esforço. Menos mágica, zero ambiguidade, autodocumentado.

**Arquivos:** `utils/quality-dashboard.sh`.

---

### F-C5 🟢 Dashboard: inconsistências menores de formatação/locale

**Evidência:** thresholds de reformatação ESR diferentes entre quick-summary (>10
chars, linha 824) e tabela de detalhes (>14 chars, linha 901); `esr_verdict()`
(linhas 675-685) usa awk sem `LC_ALL=C` (risco com locale pt_BR).

**Proposta:** unificar a formatação numérica num único helper (`_fmt_metric`) com
`LC_ALL=C` embutido; padronizar `%.2e` para tudo que contenha expoente.

**Arquivos:** `utils/quality-dashboard.sh`.

---

### F-C6 🟡 `tests-long.sh`: fase PipeWire skipada é reportada como "PASSED | 0s"

**Evidência:** `utils/tests-long.sh:455-465` — `pw-cli info` falha ⇒ `return 0` ⇒
resumo final imprime `PASSED` com 0s (vide `/testes.log` linhas 2721-2724 e 2757).
Enganoso: parece cobertura executada.

**Proposta:** introduzir código de status tri-state nas fases
(`PASSED`/`SKIPPED`/`FAILED`): `run_pipewire_phase` retorna um marcador (ex. `return 77`
convenção skip) e `run_phase` registra `SKIPPED` no array de status e no summary.
Qualquer fase com duração < 1s e status PASSED deveria emitir warning.

**Arquivos:** `utils/tests-long.sh`, `utils/_lib.sh`.

---

### F-C7 🟡 `tests-long.sh`: `cargo clean --release` na Fase 5 invalida o cache das Fases 6-7

**Evidência:** `utils/tests-long.sh:554-556` — limpa release + incremental toda
execução; benchmarks (Fase 6) e RT gate (Fase 7) recompilam do zero (~2-3 min extras
por rodada).

**Proposta:** compilar o CLAP num target dir dedicado
(`CARGO_TARGET_DIR=target/clap-audit cargo build --release ...`) ou mover a Fase 5
para o final da suíte. Medir o ganho no summary (a fase 6 hoje leva 1694s).

**Arquivos:** `utils/tests-long.sh`.

---

### F-C8 🟢 `tests-performance-regression.sh`: `grep -q "regressed"` com risco de falso-positivo

**Evidência:** `utils/tests-performance-regression.sh:117` — Criterion pode emitir a
palavra em contextos negativos.

**Proposta:** `grep -qE '(has regressed|Performance has regressed)'` ou parsear as
linhas `change:` com awk (sinal + p-value).

**Arquivos:** `utils/tests-performance-regression.sh`.

---

### F-C9 🟡 Freshness gate: garantir cobertura completa (v2 + goldens declarados ausentes)

**Evidência:** o manifest atual (`.golden_manifest.sha256`) contém 27 entradas v1 + 32
v2, mas: (1) `check_freshness()` do `tests-quick.sh:96-129` só falha para hash
divergente — **não** falha quando um CATALOG entry declara golden que não existe no
disco/manifest (caso real: `wavenet_a2_film_input_mixin_pre`, F-B3); (2) o par
modelo↔golden_v2 de SRs individuais precisa de verificação de completude por
`v2_scope`.

**Proposta:** estender o gerador do manifest (`golden_gen_build.sh` fase 11) para
emitir também uma seção `# EXPECTED` derivada do CATALOG (modelo, escopo v2, lista de
goldens esperados) e o `check_freshness()` para falhar se algum golden esperado não
existir — transformando "golden nunca gerado" de gap silencioso em erro do gate.
Exceções documentadas (ConvNet enquanto F-A1 aberto) via campo `skip_reason` no
CATALOG.

> **Nota do PO:** A respeito de F-A1, escolhi a "Opção A (paridade plena, recomendada)". Ajuste-se para esta decisão.

**Arquivos:** `tests/fixtures/golden_gen_build.sh`, `utils/tests-quick.sh`.

---

## GRUPO D — Otimização do `tests-quick.sh` ("rápido para rodar sempre")

### F-D1 🟠 `rt_constraints` roda na Fase 1 (debug): 28.1s de medição de deadline SEM SENTIDO em debug

**Evidência:** `/testes.log` linhas 1462-1484: os 14 `rt_deadline::*` rodaram em
**debug** na Fase 1 e levaram 28.13s — ~30% do tempo de teste da fase. Isso contradiz
o próprio design documentado (`docs/testing.md` Axis B: "rt_deadline → long Phase 7,
release-only; asserting deadlines in debug is meaningless") e o comentário no próprio
script. Causa: `utils/tests-quick.sh:293` inclui `rt_constraints` nos targets da Fase 1
sem `--skip rt_deadline::` (os demais módulos de medida têm skip explícito, linhas
298-301).

**Proposta:** adicionar `--skip rt_deadline:: --skip rt_jitter::` à linha da Fase 1
(ou remover o target `rt_constraints` da fase e deixar 100% no tests-long Phase 7).
**Ganho estimado: −28s por execução do quick** (de ~72s para ~44s de testes da
Fase 1). Esta é a otimização nº 1 do quick.

**Arquivos:** `utils/tests-quick.sh`.

---

### F-D2 🟡 Proptests pesados de ativação na Fase 1 (100k casos em debug)

**Evidência:** `/testes.log` — `test_tanh_pade_nr2_proptest_100k`,
`test_tanh_pade_proptest_100k`, `test_sigmoid_pade_proptest_100k`,
`test_tanh_pade_nr2_proptest_100k_avx512`, `test_tanh_piecewise_proptest_50k` aparecem
entre os últimos a concluir na Fase 1 (long-poles do wall-time do binário lib,
18.86s). São validação de aproximação polinomial — o oráculo f64 (Fase 2, release) já
fornece a correção absoluta.

**Proposta:** parametrizar o case-count via env (ex.
`NAM_QUICK_PROPTEST_CASES_MATH=10_000` no quick; full no long), ou mover os `_100k`
para `#[ignore]` + long Phase 3 (mesmo tratamento já dado aos consistency-only).
Ganho estimado: −5 a −8s na Fase 1.

**Arquivos:** `src/math/activations/activations_test.rs` (e afins),
`utils/tests-quick.sh`, `utils/tests-long.sh`.

---

### F-D3 🟢 Quick: pequenos cortes adicionais e honestidade do banner

- O banner promete "±2 min hot / ±5 min cold"; a execução real fria foi ~8 min
  (35s compile debug + 3m16s compile release + ~2 min de testes). Atualizar o banner
  (ex. "±2 min hot / ±8 min cold") ou reduzir o custo frio (compartilhar artefatos
  entre Fases 2/3 já é feito; avaliar `cargo nextest` para paralelismo por binário e
  relatórios melhores — medir antes de adotar).
- `perf_soak` na Fase 1 custa 2.0s (8 testes de concorrência) — manter (barato e são
  first-line de SPSC), mas documentar a decisão no cabeçalho do script.
- `CXX=""` redundante (linhas 365-372) — limpeza cosmética.

**Arquivos:** `utils/tests-quick.sh`.

---

## GRUPO E — Contrato de Qualidade (baseline anti-regressão)

### F-E1 🟠 Criar o contrato `docs/quality-contract.txt` + verificador com margens estilo Criterion

**Motivação:** os índices do dashboard (ESR vs NAMcore, ESR vs f64, SNR, MR-STFT,
latência mediana por bloco) hoje não têm baseline versionado — regressões de fidelidade
abaixo dos gates calibrados (frouxos, F-B4) e regressões de performance fora do
`regression_gate` passam invisíveis.

**Proposta (2 etapas):**

1. **Baseline:** executar `utils/quality-dashboard.sh --save docs/quality-contract.txt`
   (o modo `--save` já existe e gera plain-text sem ANSI —
   `quality-dashboard.sh:1347-1350`) e commitá-lo. A partir desse momento, é
   **proibido piorar** os índices registrados.
2. **Verificador:** novo `utils/quality-contract-check.sh` que roda o dashboard em modo
   `--save` num arquivo temporário e compara com o contrato:
   - **Fidelidade (ESR/SNR/MR-STFT):** regra de razão — falha se
     `novo_esr > contrato_esr × 10` (uma ordem de magnitude de tolerância absorve
     ruído inter-máquina/ISA; valores no piso f64 flutuam ~2-3×) ou se
     `novo_snr < contrato_snr − 6 dB`.
   - **Performance (latência mediana):** regra estatística séria, análoga ao Criterion:
     falha se `nova_lat > contrato_lat × 1.10` (noise_threshold de 10% p/ medições
     single-shot do dashboard; para rigor real, delegar ao
     `tests-performance-regression.sh`, que já faz t-test — o contrato serve como
     segunda linha, legível por humanos).
   - Campos `N/A` são ignorados; campos novos (modelos adicionados) exigem re-save
     consciente com justificativa no commit.
3. Renovação do contrato apenas via `--save` deliberado + justificativa no commit
   (mesma disciplina do `--save` do regression gate, `docs/benchmarks.md`).

> **Nota do PO:** Eu prefiro que esta funcionalidade fosse unificada como um script único.

**Pré-requisito:** F-C2 (MR-STFT `0.0000`) e F-C4 (família f64) devem ser corrigidos
ANTES do primeiro `--save`, senão o contrato congela dados corrompidos.

**Arquivos:** novo `utils/quality-contract-check.sh`, `utils/quality-dashboard.sh`,
`docs/quality-contract.txt` (gerado), `docs/benchmarks.md`/`docs/testing.md` (documentar).

---

## Observações registradas (sem ação imediata)

- **`linear_fft rf4096 impulse`**: MR-STFT medido 1.0898e-1 com gate 1.20e-1 — margem
  de apenas ~9%. Não é violação; monitorar no contrato (F-E1) e, se o gate apertar
  falsamente, recalibrar com provenance.
- **LSTM Fast mode**: `quick_parity_lstm_1x16` (Fast) ESR 1.16e-2 / SNR 19.3 dB é o
  comportamento esperado do modo Fast (Padé fora de calibração p/ H=16) — documentado
  em `docs/perceptual_validation.md`. O default de produção é Standard (HF: 1.45e-11).
- **Rule 5 "violada" no dashboard (cold-start)**: o próprio dashboard já explica que a
  decomposição cold-start de 256 amostras não é comparável ao pareado-com-prewarm para
  modelos com RF > janela. Nenhuma ação além de manter o disclaimer.
- **`TODO-wavenet_a2_max.md`**: nenhum achado novo desta auditoria altera o
  diagnóstico dos Bugs A/B/C. Registro colateral útil: F-A4 (gating/FiLM ignorados no
  caminho A1 Free) compartilha a mesma classe de causa-raiz do Bug B ("parâmetros de
  config silenciosamente ignorados fora do caminho que os implementa") — a solução
  fail-closed de F-A4 pode servir de template para a guarda equivalente no A2 dinâmico.

---

Agrupamento para implementação segura e incremental. Ordem recomendada: E1 → E2 → E3 →
E4 → E5 (E1 e E2 destravam a confiabilidade da própria malha de testes antes de mexer
em produção; E5 congela o contrato por último, sobre dados já corretos).

## Épico E1 — "Nenhum teste verde sem evidência" (rigor da malha) 🔴 [DONE]

Elimina os falsos-verdes da malha de paridade. **Risco: baixo (só testes).**

1. F-B1 — `run_v1`/`run_v1_hf` retornam e asseveram `ParityOutcome` + meta-teste
   anti-`let _`.
2. F-B2 — `quick_parity_convnet` vira asserção de contrato de incompatibilidade.
3. F-C6 — status `SKIPPED` no summary do tests-long (PipeWire e afins).
4. F-B7 — determinismo ConvNet endurecido p/ MSE == 0.0.
5. F-B8 — documentar piso f64 das âncoras (docs).

## Épico E2 — Scripts de QA corretos e rápidos 🟠 [DONE]

Corrige dados exibidos e devolve ~30-40s ao loop diário. **Risco: baixo (shell).**

1. F-C1 — fix `_cargo_meas` (arrays; `--test-threads=1` restaurado). **Crítico.**
2. F-D1 — `--skip rt_deadline:: --skip rt_jitter::` na Fase 1 (−28s).
3. F-C2 — MR-STFT em `%.2e` no dashboard.
4. F-C3 — contagem de fases `[N/M]` correta.
5. F-C4 — mapa explícito de famílias f64.
6. F-C5, F-C8, F-D3 — formatação/locale, grep do regression gate, banner honesto.
7. F-D2 — case-counts dos proptests de ativação parametrizados (quick vs long).
8. F-C7 — target dir dedicado p/ CLAP no tests-long (preserva cache das fases 6-7).

## Épico E3 — Fixtures completas e auto-verificáveis 🟠 [DONE]

1. F-B3 — gerar golden do `wavenet_a2_film_input_mixin_pre`, calibrar, ativar teste.
2. F-C9 — freshness gate falha em golden esperado ausente (seção `# EXPECTED`).
3. F-B6 — live cross-validation p/ FiLM/gated/blended + meta_coherence bidirecional.

## Épico E4 — Paridade NAMcore no código de produção 🟡→🟠 [DOING]

Mudanças em `src/` — exigem golden verde antes/depois e review rigoroso.
**Risco: médio.** Ordem interna do mais contido ao mais amplo:

1. F-A5 — `cond_dsp.prewarm(prewarm_samples())` + fixture LSTM-as-condition_dsp.
2. F-A2 — aceitar `sample_rate: -1.0` como unknown.
3. F-A7 — rejeitar `num_layers == 0` e validar canais no LSTM.
4. F-A3 — `condition_dsp` desqualifica SKU de catálogo A1.
5. F-A6 — corrigir `prewarm_samples()` (soma de RFs, fórmula C++).
6. F-A4 — fail-closed p/ gating/FiLM/head1x1/layer1x1 no caminho A1 Free/Dynamic.
7. F-A1 — decisão + implementação do ConvNet compatível com NAMcore (Opção A).
   Maior peça do épico; Opção A, inclui fixture C++-format + golden + quick_parity real
   (fecha F-B2 em definitivo).
8. F-B4 — recalibração em lote dos gates frouxos (fazer APÓS 1-7, para calibrar
   sobre o estado final do código).
9. F-B5 — `MseGate::NotApplicable` explícito.

## Épico E5 — Contrato de Qualidade (tarefa final da sprint) 🟠 [DOING]

**Depende de:** E2 (dashboard correto) e idealmente E4.8 (gates recalibrados).
> **Nota do PO:** Eu prefiro que esta funcionalidade fosse unificada como um script único.

1. F-E1 passo 1 — executar `utils/quality-dashboard.sh --save docs/quality-contract.txt`
   e commitar como **baseline oficial**. A partir deste momento fica proibido piorar
   os índices de qualidade e performance ali registrados.
2. F-E1 passo 2 — implementar `utils/quality-contract-check.sh` com margens sérias
   (fidelidade: razão ≤ 10× / SNR −6 dB; performance: ≤ +10%, com o t-test do
   `tests-performance-regression.sh` como autoridade estatística primária).
3. Documentar o fluxo de renovação do contrato em `docs/testing.md` e
   `docs/benchmarks.md`.
