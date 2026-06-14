<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# 🗺️ TODO-sprints — Operação "Goldens Impecáveis" - Chat Kilo Code "Auditoria dos golden tests A1/A2"

> Plano de execução resultante de uma auditoria focada (`revisor-auditor`, 2026-06-13) sobre o **único bastião
> de garantia de qualidade do nam-rs: os golden tests**. A confiabilidade dos goldens foi abalada nos últimos
> commits. A meta desta rodada é torná-los **absolutamente impecáveis, precisos e estáveis no longo prazo** —
> idealmente, após a conclusão dos épicos abaixo, só voltaremos a tocá-los para **acrescentar** novos modelos/arquiteturas.
>
> **Fontes de verdade confiáveis (clones shallow de estados sabidamente perfeitos):**
>
> - `github.com/nam-rs_v2.0.0/` — último estado seguro dos goldens **A1** (WaveNet/LSTM clássicos).
> - `github.com/NeuralAmpModelerCore_v0.5.3/` — implementação de referência C++ com **A2 estável**
>   (`NAM/wavenet/a2_fast.{h,cpp}`, `slimmable`, `film`, `container`).

---

## 🎯 Comandos de validação (antes/depois) — referência rápida

Use **sempre o comando mais específico possível** para iterar com agilidade. Os goldens vivem em duas camadas:

| Camada                                          | Comando                                                                                    | Precisa de C++? | Tempo  |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ |:---------------:| ------ |
| **Layer 1** — goldens pré-commitados (rápido)   | `cargo test --test golden_vectors -- --test-threads=1 --nocapture`                         | Não             | ~3 s   |
| **Layer 1** — cabsim (convolução de referência) | `cargo test --test cabsim_golden -- --test-threads=1 --nocapture`                          | Não             | ~1 s   |
| **Layer 1** — cabsim ⇄ C++ (ESR < 1e-13)        | `cargo test --test cabsim_cpp_parity -- --test-threads=1 --nocapture`                      | Sim (fixture)   | ~1 s   |
| **Layer 2** — cross-validation live A1/A2 (v1)  | `cargo test --test cpp_parity -- --ignored --test-threads=1 --nocapture`                   | Sim             | lento  |
| **Layer 2** — um modelo isolado (ex.: Lite)     | `cargo test --test cpp_parity live_cross_validation_wavenet_lite -- --ignored --nocapture` | Sim             | médio  |
| **Suíte padrão completa**                       | `./utils/tests-cargo.sh`                                                                   | Não             | ~2 min |
| **Suíte longa (inclui Layer 2)**                | `./utils/tests-long.sh`                                                                    | Sim             | lento  |

**Bench (regressão de hot-path, opcional — não é golden):**
`cargo bench --bench inference_bench` · `cargo bench --bench dot_4x_bench` · `cargo bench --bench kahan_conv1d_bench`

> O comando **mais específico e central** desta operação é o **Layer 1 `golden_vectors`**: roda em ~3 s, sem
> C++, e é o gate anti-regressão definitivo. É o comando a rodar **antes e depois** de cada tarefa abaixo.

---

## 🩺 Diagnóstico da auditoria (2026-06-13) — verificado na fonte

Cada achado abaixo foi confirmado lendo o código/fixtures e executando `cargo test --test golden_vectors`.
A baseline atual é **19 passed / 0 failed em 2.60 s** — porém **vários "PASS" são ilusórios** (ver abaixo).

### A1 (WaveNet/LSTM clássicos) — comparação com `nam-rs_v2.0.0`

- ✅ **Fixtures WaveNet intactos.** `golden_wavenet_{standard,feather,nano}.bin` são **byte-idênticos** ao
  `nam-rs_v2.0.0` (SHA-256 confere). A referência C++ não mudou; **apenas os thresholds foram apertados**
  (v2.0.0: piso 9–16 dB SNR adaptativo; atual: 45–60 dB por canal). O aperto é **legítimo** — o fix da cascata
  de heads (T16.1) elevou o SNR Rust↔C++ de ~10 dB para **68.4 dB (Standard), 67.6 dB (Feather), 52.6 dB (Nano)**.
- ⚠️ **Fixtures LSTM divergiram.** `golden_lstm_1x16.bin` e `golden_lstm_2x8.bin` **diferem** do `nam-rs_v2.0.0`
  (SHA-256 distinto). Foram regenerados em commit recente; a **proveniência precisa ser re-validada** contra a
  referência segura. Além disso, 7 modelos LSTM **sintéticos** novos (1x8, 1x12, 1x24, 1x40, 2x12, 2x16, 2x24)
  foram adicionados com pesos auto-gerados (não treinados) — golden de baixa autoridade.
- 🔴 **WaveNet Lite (CH=12) é um gate falso.** SNR medido **0.9 dB**, ESR **0.815** (81,5 % da energia é erro —
  saída quase descorrelacionada), mas o threshold foi rebaixado para **SNR ≥ 0 dB / ESR < 1.0**, fazendo o teste
  **passar com ruído**. "Fidelity Margin = −9.5 dB ✗" no próprio relatório. É o epicentro da perda de
  confiabilidade: um golden que **passa independentemente da correção**. O modelo `BossWN-lite.nam` é sintético
  (2026-06-11, sem `sample_rate`); a divergência CH=12 (não-potência-de-2) foi investigada no Épico 21 sem solução.

### A2 (WaveNet slimmable/FiLM/container) — situação atual e oportunidade com `v0.5.3`

- 🔴 **Self-goldens A2 são tautológicos.** `test_golden_vectors_wavenet_a2_{full,lite}` comparam a saída Rust
  **contra a própria saída Rust** gravada na 1ª execução → **MSE = 0.00, SNR = ∞**. Isso **só detecta
  não-determinismo**, nunca erro de correção vs. a referência. Sintoma de alerta: **LUFS = 291.2 / 84.8** (níveis
  absurdos, dezenas de dB acima de 0 dBFS), sugerindo que a própria saída A2 pode estar incorreta — e o
  self-golden seria incapaz de perceber. O motivo histórico (`docs`, T16.4): o **commit C++ antigo pinado
  (`e49c93e6`)** tinha o `a2_fast.cpp` numericamente instável (amplitude explosiva), então optou-se por
  self-goldens + "garbage guard" que faz SKIP do A2 no live.
- 🟢 **`NeuralAmpModelerCore_v0.5.3` traz A2 estável.** `NAM/wavenet/a2_fast.{h,cpp}` implementa exatamente a
  forma do A2 do nam-rs: **23 camadas**, kernels `{6×14, 15, 15, 6×7}`, dilations `{1,3,7,17,41,101,239}×…`,
  LeakyReLU `0.01`, **A2 standard CH=8** e **A2 nano CH=3** (= nossos A2-Full e A2-Lite). Requer compilar com
  `-DNAM_ENABLE_A2_FAST`. Isso **destrava a substituição dos self-goldens por cross-reference C++ real** — o
  maior salto de confiabilidade possível para o A2.

### 🆕 Recursos nos mirrors C++ (`tests/fixtures/NeuralAmpModeler{Core,Plugin}`) — verificado

> Os mirrors locais (não-versionados) contêm material **muito superior** ao que usamos hoje. Confirmado por
> render direto com o binário compilado.

- 🟢 **Modelos `.nam` oficiais (reais/saudáveis) em `NeuralAmpModelerCore/example_models/`** — produzem saída
  **limitada**, ao contrário dos nossos sintéticos explosivos (`~1e15`):

  | Modelo oficial              | Arquitetura           | `max_abs` (render, input.wav) | Uso proposto                        |
  | --------------------------- | --------------------- | ----------------------------- | ----------------------------------- |
  | `wavenet_a1_standard.nam`   | WaveNet A1            | **0.35**                      | Referência A1 real (Épico 1)        |
  | `wavenet.nam`               | WaveNet A1            | **0.07**                      | Referência A1 real                  |
  | `lstm.nam`                  | LSTM                  | **0.04**                      | Referência LSTM real                |
  | `wavenet_a2_max.nam`        | WaveNet A2 (cond_dsp) | **10.3**                      | Referência A2 real (Épico 2)        |
  | `slimmable_wavenet.nam`     | WaveNet A2 slimmable  | **15**                        | Referência A2 slimmable             |
  | `slimmable_container.nam`   | SlimmableContainer    | (via `--slim`)                | Referência de Container (cobre gap) |
  | `wavenet_condition_dsp.nam` | WaveNet + FiLM        | —                             | Cobre arquitetura FiLM/conditioning |

- 🟢 **Áudio real para estímulo:** `NeuralAmpModelerCore/example_audio/input.wav` (72 000 amostras @ 48 kHz, DI
  real) e `NeuralAmpModelerPlugin/REAPER/Guitar DI.wav` — estímulos do mundo real além dos sintéticos.

- 🟢 **O mirror `NeuralAmpModelerCore` (commit `e49c93e`) é MAIS RECENTE que o tag `v0.5.3`** (é o HEAD de
  desenvolvimento pós-tag): contém `NAM/linear.{cpp,h}` (arquitetura Linear, ausente no tag), `a2_fast.cpp`
  **reescrito e mais otimizado** (arena de histórico contígua, templates `<KernelSize,IsFirst,IsLast>`, nova API
  `PrewarmSamples()` / prewarm opcional no `Reset()`), e mudanças em `slimmable`/`container`. É também o commit
  **já pinado** pelo `golden_gen_build.sh`. Recomendação: usar o **mirror `e49c93e`** como referência canônica
  (mais nova e já pinada) e o tag `v0.5.3` como fallback estável; verificar que ambos concordam para os goldens.

### Infra dos goldens — achados transversais (óbvios e seguros)

- 🔴 **72 de 94 fixtures `.bin` são órfãos.** Todos os `golden_*_v2_*k.bin` (72 arquivos) são **gerados** por
  `golden_gen_build.sh` (linha 269) mas **lidos por nenhum teste**. O `cpp_parity.rs` (Layer 2) **re-renderiza
  o C++ ao vivo** a cada execução e nunca lê esses `.bin`; o sufixo `k` no nome (`_v2_48000k`) nem corresponde
  ao que o código geraria (`_v2_48000`). São ~peso morto que **aparenta autoridade** sem validar nada. O próprio
  script admite isso em comentário (linha 34).
- 🟠 **Thresholds "flutuantes".** `topology_thresholds()` (`tests/common/validation.rs`) **deriva** os limiares de
  heurísticas da topologia (canais/camadas), **não** da medição real registrada no momento da geração do golden.
  Resultado: se o motor regredir, o gate "flutua" junto e pode mascarar a regressão. O padrão impecável é
  **gravar a métrica medida na geração** (SNR/ESR) e gatear em `medido − margem fixa`.
- 🟠 **Goldens A2 não-self órfãos.** `golden_wavenet_a2_{full,lite}.bin` (não-`_self`) existem mas não são lidos
  (só os `_self.bin` são). Duplicação confusa.
- 🟢 **Cabsim é exemplar.** `cabsim_golden.rs` valida UPOLS contra **convolução direta O(N²)** inline (oráculo
  matemático auto-contido) e `cabsim_cpp_parity.rs` cruza com C++ a ESR < 1e-13. É o **modelo a seguir** para os
  demais goldens.

---

## Convenções Mandatórias (aplicáveis a todas as tarefas)

- **RT-Safety** (`.agents/rules/rust.md`): zero alloc/drop de heap na *audio thread*; sem `println!`/`format!`/locks;
  sem `unwrap()`/`expect()`; FTZ+DAZ; sinalização via `RtStatusFlags`; desalocação via SPSC GC.
- **Testes** (`.agents/rules/testing.md`): unidade inline se arquivo `< 300` linhas; senão `*_test.rs` irmão.
  Integração em `tests/`. Lentos/parity com `#[ignore]` (lane `utils/tests-long.sh`).
- **Copyright** (`.agents/rules/copyright.md`): cabeçalho SPDX em todo arquivo novo/editado.
- **Golden = seguro anti-degradação**: qualquer otimização só é aceita com **todos** os golden vectors verdes —
  e, após esta rodada, **sem nenhum gate falso** (SNR ≥ 0 dB ou ESR < 1.0 não contam como gate).
- **Lint final** (`.agents/rules/linting.md`): `utils/lints.sh` + `utils/tests-cargo.sh` verdes ao fechar cada
  tarefa; acionar `documentador` em mudanças arquiteturais relevantes.
- **Princípio anti-tautologia:** todo golden deve poder **falhar**. Um golden que passa contra a própria saída
  (self-golden) ou com threshold neutralizado (0 dB / ESR 1.0) **não é um gate** — é um placebo.

### ⚠️ Apêndice de verificação (falsos positivos já descartados — NÃO "corrigir")

| Suspeita levantada                                                  | Veredito verificado                                                                                                                                                                                                                                            |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Os fixtures WaveNet foram corrompidos na regeneração recente"      | **Falso.** `golden_wavenet_{standard,feather,nano}.bin` são byte-idênticos ao `nam-rs_v2.0.0` (SHA-256). Só os thresholds mudaram (legítimo, pós-T16.1).                                                                                                       |
| "Espelhos C++ em `tests/fixtures/NeuralAmpModeler*` são peso morto" | **Falso.** São **não-versionados** (`.gitignore`) e necessários para `golden_gen_build.sh` / `render_ir.cpp`.                                                                                                                                                  |
| "A2 lê `head_scale` do stream de pesos em vez do campo `config`"    | **Falso (correto).** O C++ de referência lê `head_scale` como último float do stream (`a2_fast.cpp:122-124,274-275`); o Rust está alinhado.                                                                                                                    |
| "WaveNet Lite CH=12 tem bug estrutural nos kernels SIMD"            | **Investigado e descartado no Épico 21 (T21.1).** Sem bug em conv1d/GEMV/GEMM/tanh/alinhamento/`head_rechannel`. Hipótese remanescente: drift numérico na acumulação de head para CH não-potência-de-2 amplificado na fronteira de bloco (64) — ver Épico 1.4. |

---

## ÉPICO 0 — Referência C++ `NeuralAmpModelerCore_v0.5.3`: build + captura de dados ✅ [EXECUTADO NA AUDITORIA]

> A auditoria já **compilou e executou** o render do `v0.5.3` para capturar os dados. Os comandos exatos estão
> abaixo (reprodutíveis), seguidos do **resultado obtido**. O `v0.5.3` habilita `NAM_ENABLE_A2_FAST=ON` por padrão.

### Comandos exatos (a partir da raiz do projeto `/home/fabio/nam-rs`)

```bash
# 0) Variáveis (raiz do repositório)
cd /home/fabio/nam-rs
V053="$PWD/github.com/NeuralAmpModelerCore_v0.5.3"
PLUGIN_DSP="$PWD/tests/fixtures/NeuralAmpModelerPlugin/AudioDSPTools"   # mirror já populado (eigen + wav.cpp)

# 1) Suprir dependências do clone shallow (eigen + AudioDSPTools vêm vazios como submódulos).
#    Reaproveitamos o eigen/AudioDSPTools já populados no mirror do Plugin via symlink.
rm -rf "$V053/Dependencies/eigen"         && ln -sfn "$PLUGIN_DSP/Dependencies/eigen" "$V053/Dependencies/eigen"
rm -rf "$V053/Dependencies/AudioDSPTools" && ln -sfn "$PLUGIN_DSP"                    "$V053/Dependencies/AudioDSPTools"

# 2) Compilar o render (A2 fast já é ON por padrão no v0.5.3)
cmake -S "$V053" -B /tmp/kilo/v053_build -DCMAKE_BUILD_TYPE=Release -DNAM_ENABLE_A2_FAST=ON
cmake --build /tmp/kilo/v053_build --target render -j"$(nproc)"
RENDER=/tmp/kilo/v053_build/tools/render   # binário gerado

# 3) Renderizar (uso: render [--slim <0.0-1.0>] <model.nam> <input.wav> [output.wav])
#    Sinal de entrada = stress v1 já committed; modelos em tests/fixtures/models/.
mkdir -p /tmp/kilo/a2out
"$RENDER" tests/fixtures/models/wavenet_a2_full.nam tests/fixtures/stress_signal.wav /tmp/kilo/a2out/a2_full.wav
"$RENDER" tests/fixtures/models/wavenet_a2_lite.nam tests/fixtures/stress_signal.wav /tmp/kilo/a2out/a2_lite.wav
# (Para A1, idem com BossWN-standard.nam etc. — confirma que batem com os .bin committed.)
```

> **Observação importante (slimmable/container):** o `render` aceita `--slim <0.0-1.0>` para modelos que
> implementam `SlimmableModel` (espelha `c.set_slimmable_size(...)` do Rust). Útil para validar os
> `test_golden_vectors_container_a2_*` contra o C++.

### Resultado obtido (2026-06-13) — **achado decisivo**

| Modelo  | `cpp_max_abs` (v0.5.3) | `rust_max_abs` (self-golden) | **SNR(C++ ↔ Rust)** | Veredito                                 |
| ------- | ---------------------- | ---------------------------- |:-------------------:| ---------------------------------------- |
| A2-Full | `4.020e15`             | `4.021e15`                   | **70.3 dB**         | Engines **concordam**; amplitude explode |
| A2-Lite | `8.191e4`              | `8.199e4`                    | **57.2 dB**         | Engines **concordam**; amplitude absurda |

- 🟢 **Rust e C++ `v0.5.3` concordam no A2** (57–70 dB SNR). A divergência "instabilidade do A2" **não é bug de
  engine** — é a **amplitude explosiva dos modelos sintéticos** (`generate_a2_fixtures.py` usa pesos aleatórios
  determinísticos; uma WaveNet de 23 camadas com pesos não treinados diverge para `~1e15`).
- 🔴 Consequência: o "garbage guard" que faz SKIP do A2 no live estava **mascarando uma concordância real**, e os
  self-goldens tautológicos eram desnecessários — bastava um gate **scale-invariant** (ESR/SNR).
- ➡️ **Pivô do Épico 2:** o cross-reference C++ A2 é **viável e valioso**, mas exige (a) gate por **ESR/SNR**
  (não MSE absoluto, inútil a `1e15`) e, idealmente, (b) **substituir os fixtures A2 sintéticos por modelos
  realistas/normalizados** para que a saída fique em faixa sã e o golden seja perceptualmente significativo.

---

## ÉPICO 1 — Retorno ao Estado Seguro dos Goldens A1 🛡️ [release-blocker] [TODO]

> **Meta:** restaurar a confiabilidade dos goldens A1 ao padrão seguro do `nam-rs_v2.0.0`, eliminando todo gate
> falso e re-validando a proveniência dos fixtures alterados. Ao final, **nenhum golden A1 passa com ruído**.

### Sprint 1.1 — Eliminar gates falsos e validar proveniência

- **[T1.0] Adotar os modelos `.nam` oficiais como referência A1 real.** [DONE]

  - **Contexto:** os mirrors trazem `wavenet_a1_standard.nam`, `wavenet.nam`, `lstm.nam` — modelos oficiais com
    saída perfeitamente limitada (0.04–0.35). São referências muito mais confiáveis que os sintéticos.
  - **Ação:** copiar esses `.nam` para `tests/fixtures/models/` (com proveniência), gerar seus goldens
    cross-reference via render C++ (mirror `e49c93e`), e adicionar testes em `golden_vectors.rs`. Avaliar
    aposentar os LSTM sintéticos de menor valor em favor do `lstm.nam` oficial.
  - **Critério de aceite:** ≥ 1 golden A1 WaveNet e ≥ 1 LSTM derivados de modelos oficiais, verdes, com
    proveniência documentada.

- **[T1.1] Re-validar a proveniência dos fixtures LSTM alterados.** [DONE]

  - **Contexto:** `golden_lstm_1x16.bin` e `golden_lstm_2x8.bin` divergem (SHA-256) do `nam-rs_v2.0.0`; os 7
    LSTM sintéticos (1x8/1x12/1x24/1x40/2x12/2x16/2x24) são de pesos auto-gerados.
  - **Ação:** comparar numericamente (MSE/SNR/ESR) cada fixture LSTM atual contra o equivalente em
    `github.com/nam-rs_v2.0.0/tests/fixtures/`; decidir, com base em evidência, qual reter como referência segura.
    Para os que diferirem sem justificativa de melhoria comprovada, **restaurar a versão `v2.0.0`** ou regenerar
    do C++ (mirror `e49c93e`) documentando proveniência.
  - **Critério de aceite:** cada `.bin` LSTM committed tem proveniência registrada (origem + comando + commit C++)
    no `tests/fixtures/README.md`; `cargo test --test golden_vectors` verde com thresholds não-neutralizados.
  - **Parecer da Auditoria (2026-06-13):** Os arquivos atuais `golden_lstm_{1x16,2x8}.bin` são byte-idênticos (SHA-256 confere) à renderização da engine C++ pinada no commit `e49c93e`. As divergências em relação ao `nam-rs_v2.0.0` são extremamente pequenas (SNR > 120 dB) e derivam do fato de o `v2.0.0` ter usado clones de HEAD dinâmicos/não-pinados da engine C++ na época. Decidiu-se reter os fixtures atuais por estarem perfeitamente alinhados com o mirror pinado oficial.

- **[T1.2] Resolver o gate falso do WaveNet Lite (CH=12).** ⚠️ **(crítico)**

  - **Decisão arquitetural necessária** (escolher 1, documentar ADR):
    - **(a)** Concluir a investigação do Épico 21 e **corrigir** o drift de acumulação de head para CH=12 (ver
      caminhos sugeridos: Kahan AVX2 inline / acumulação f64 completa via `_mm256_cvtps_pd` / compensação global
      por bloco) até atingir **SNR ≥ 40 dB**, restaurando o threshold rígido.
    - **(b)** **Substituir** `BossWN-lite.nam` sintético por um **modelo Lite real treinado** (CH potência-de-2,
      ex.: CH=8 ou CH=16) que atinja a paridade dos demais WaveNet, aposentando o caso CH=12 problemático.
    - **(c)** Se nenhum dos anteriores for viável agora, **mover o Lite CH=12 para uma suíte `#[ignore]`
      explicitamente rotulada `known-divergent`** e **remover seu falso gate** do `golden_vectors` (não pode
      figurar como "PASS" na suíte rápida).
  - **Critério de aceite:** não existe mais um teste golden passando com `SNR ≥ 0 dB`/`ESR < 1.0`; a situação do
    Lite está ou **corrigida** (≥ 40 dB) ou **honestamente sinalizada** (não mascarada como verde).

- **[T1.3] Calibrar thresholds A1 rígidos a partir da medição real.**

  - **Ação:** com os fixtures A1 validados (T1.1) e o resultado C++ `v0.5.3` (Épico 0), fixar os pisos por modelo
    como `SNR_medido − margem` (ex.: margem 6–8 dB) e `ESR_medido × fator`, **comentando a medição que originou
    cada piso** (padrão já usado em T16.2). Eliminar a derivação puramente heurística para os modelos com golden
    committed (ver Épico 3.2 para a infra que torna isso sistemático).
  - **Critério de aceite:** cada threshold A1 rastreia uma medição documentada; `golden_vectors` verde; um
    experimento de regressão proposital (ex.: desligar o fix de cascata) **faz o teste falhar** (prova de gate vivo).

---

## ÉPICO 2 — Cross-Reference C++ A2 via NeuralAmpModelerCore v0.5.3 🎯 [release-blocker] [TODO]

> **Meta:** eliminar os self-goldens A2 tautológicos, substituindo-os por **cross-reference Rust ↔ C++ real**
> contra o `a2_fast` do `v0.5.3`. A auditoria (Épico 0) **provou que Rust e C++ concordam (57–70 dB SNR)** — o
> bloqueio não era o engine, e sim (a) gate por escala errado (MSE absoluto) e (b) fixtures A2 sintéticos
> explosivos. Este épico ataca os dois.

### Sprint 2.1 — Substituir self-golden por cross-reference scale-invariant

- **[T2.1] Integrar o render do `v0.5.3` ao `golden_gen_build.sh` (pinar SHA, A2 fast).**

  - **Ação:** adicionar um caminho no script para compilar o render do `v0.5.3` (comandos verificados no Épico 0;
    `NAM_ENABLE_A2_FAST=ON`), pinando o commit/snapshot e suprindo `eigen`/`AudioDSPTools` (symlink do mirror do
    Plugin). Registrar a proveniência em `tests/fixtures/README.md`.
  - **Atenção:** a forma do `.nam` A2 casa com `is_a2_shape()` (kernels 6/15, dilations, 23 camadas, LeakyReLU
    0.01) — **já confirmado**; `head_scale` lido do stream — **já confirmado** no apêndice de verificação.
  - **Critério de aceite:** o script gera, de forma reprodutível, as saídas A2 do C++ `v0.5.3`.

- **[T2.2] Reescrever os goldens A2 como cross-reference Rust ↔ C++ com gate ESR/SNR.**

  - **Ação:** gravar `golden_wavenet_a2_{full,lite}.bin` a partir do render C++ `v0.5.3`; reescrever
    `test_golden_vectors_wavenet_a2_{full,lite}` para comparar **Rust ↔ C++** (remover o `write_golden_bin`
    self-golden). **Gate por ESR/SNR** (scale-invariant), pois MSE absoluto é inútil a `~1e15`. Pisos calibrados
    pela medição (A2-Full ~70 dB → piso ~60 dB; A2-Lite ~57 dB → piso ~50 dB), documentando a origem.
  - **Critério de aceite:** os testes A2 deixam de reportar `MSE = 0 / SNR = ∞`; passam por ESR/SNR rígidos
    documentados; um experimento de regressão proposital os faz falhar.

- **[T2.3] Substituir fixtures A2 sintéticos pelos modelos A2 oficiais dos mirrors.** ⚠️ **(qualidade — alto impacto)**

  - **Contexto:** `generate_a2_fixtures.py` usa pesos aleatórios → saída explode a `1e15` (A2-Full) / `8e4`
    (A2-Lite) **em ambos os engines**. Os mirrors já trazem A2 oficiais **com saída limitada**:
    `wavenet_a2_max.nam` (max 10.3, inclui `condition_dsp`/FiLM), `slimmable_wavenet.nam` (max 15) e
    `slimmable_container.nam` (testável via `--slim`).
  - **Ação:** adotar esses `.nam` oficiais como base dos goldens A2 (copiar para `tests/fixtures/models/` com
    proveniência), gerar os goldens cross-reference C++ e revalidar Rust↔C++. Cobrir explicitamente as
    arquiteturas hoje sem golden real: **FiLM/conditioning** (`wavenet_condition_dsp.nam`) e **SlimmableContainer**.
  - **Atenção:** confirmar que o loader/dispatcher do nam-rs aceita `condition_dsp` e `SlimmableContainer`; se
    houver gap, registrar tarefa de engine própria (fora da infra de testes).
  - **Critério de aceite:** `max_abs` das saídas A2 em faixa sã; ESR Rust↔C++ ≥ pisos calibrados; goldens A2
    perceptualmente significativos e derivados de modelos oficiais.

- **[T2.4] Reativar o A2 no cross-validation live (`cpp_parity.rs`) e enxugar o garbage guard.**

  - **Ação:** apontar o `cpp_parity` A2 para o binário `v0.5.3` e **remover o tier de SKIP por "amplitude
    absurda"** (`> 1e3`), que mascarava a concordância real; manter o guard apenas para **NaN/Inf** legítimos.
  - **Critério de aceite:** `live_cross_validation_wavenet_a2_{full,lite}` (v1) executam comparação real (não SKIP)
    e passam por ESR/SNR; container goldens (`test_golden_vectors_container_a2_*`, incl. caminho `--slim`)
    revalidados.

---

## ÉPICO 3 — Infra de Goldens: Higiene, Verdade e Estabilidade 🧹 [TODO]

> **Meta:** melhorias óbvias, seguras e imediatas que tornam a infra de goldens enxuta, auto-documentada e
> resistente a regressões silenciosas — para "nunca mais precisar tocar nos goldens" salvo para adicionar modelos.

### Sprint 3.1 — Remover peso morto e ambiguidades

- **[T3.1] Eliminar (ou conectar) os 72 fixtures `_v2_*k.bin` órfãos.**

  - **Ação:** decidir entre **(a)** remover os 72 `golden_*_v2_*k.bin` + os geradores correspondentes em
    `golden_gen_build.sh` (linhas ~258–293) — já que o `cpp_parity` v2 re-renderiza ao vivo e nunca os lê; ou
    **(b)** transformar o `cpp_parity` v2 em um Layer-1 real que **lê** esses `.bin` (com naming consistente, sem o
    sufixo `k` espúrio). Recomendação: **(a)** (alinha com "verdade" — sem fixtures que não validam nada), salvo
    se o offline-determinístico v2 for desejado, então **(b)**.
  - **Critério de aceite:** não existem fixtures committed que nenhum teste lê; `golden_gen_build.sh` não gera
    artefatos inertes; `README.md` reflete a realidade.

- **[T3.2] Remover os goldens A2 não-`self` órfãos.**

  - **Ação:** após o Épico 2 reintroduzir os A2 cross-reference legítimos, remover qualquer `.bin` A2 duplicado
    que não seja lido. Garantir 1 fonte por modelo.
  - **Critério de aceite:** sem `.bin` A2 órfão; tabela do `README.md` lista apenas fixtures vivos.

### Sprint 3.2 — Thresholds gravados e auto-documentados (estabilidade de longo prazo)

- **[T3.3] Migrar de thresholds heurísticos para thresholds gravados na geração.**

  - **Ação:** introduzir um manifesto versionado (ex.: `tests/fixtures/golden_manifest.toml`) que, por fixture,
    registre: origem (C++ commit/Rust), SNR/ESR/MSE medidos na geração, e margem aplicada. `golden_vectors` lê o
    manifesto e gateia em `medido − margem`. Mantém a derivação heurística apenas como **fallback** para modelos
    sem golden committed.
  - **Benefício:** o gate deixa de "flutuar" com a topologia; qualquer regressão futura do motor é detectada com
    margem fixa e auditável. Cada número no gate é rastreável à sua origem.
  - **Critério de aceite:** thresholds dos goldens vêm do manifesto; documentação atualizada; suíte verde.

- **[T3.4] Padronizar a estrutura dos goldens pelo modelo exemplar (cabsim).**

  - **Ação:** documentar no `README.md`/`architecture.md` o princípio "todo golden deve poder falhar" e o padrão
    de oráculo (cabsim = convolução direta; A1/A2 = cross-reference C++ pinado). Adicionar um **meta-teste** que
    falha se qualquer threshold de golden estiver neutralizado (SNR ≤ 0 dB ou ESR ≥ 1.0) — guarda contra
    reintrodução de gates falsos.
  - **Critério de aceite:** meta-teste anti-placebo verde; políticas documentadas (acionar `documentador`).

---

## ÉPICO 4 — Alavancagem de Qualidade Universal a partir dos Goldens 🌐 [TODO]

> **Resposta à pergunta "há espaço para melhorias universais de qualidade a partir das ações aqui?": SIM.**
> Os goldens são o **gate transversal** de todo o motor (DSP, SIMD, loaders, RT-safety). Endurecê-los eleva o
> piso de qualidade de tudo o que passa por eles — sem tocar no hot-path. As ações abaixo são derivadas
> diretamente desta rodada e têm efeito universal, mantendo-se **estritamente no escopo de testes/infra**.

- **[T4.1] Oráculo de referência por arquitetura (não auto-referencial).** Generalizar o padrão exemplar do
  cabsim (oráculo matemático/cross-reference) para **toda** arquitetura: A1/A2 ⇄ C++ pinado; Linear ⇄ convolução
  direta. Efeito universal: qualquer regressão numérica em qualquer kernel é capturada por um gate que **pode falhar**.
- **[T4.2] Cobertura de sinal multi-estímulo/multi-SR como gate real.** Hoje os v2 multi-SR são órfãos (Épico 3).
  Transformá-los em gate Layer-1 (ou consolidá-los) exercita o motor em 44.1/48/88.2/96/192 kHz e em 5 categorias
  de estímulo (GA/FRG/P/BA/PA). Efeito universal: cobre resampler, fronteiras de bloco e estados recorrentes.
- **[T4.3] Métricas perceptuais como guard-rail (ESR/MR-STFT/LUFS), não só MSE.** Padronizar ESR como gate
  primário scale-invariant (já iniciado em T16.4) e MR-STFT/LUFS como diagnósticos. Efeito universal: gates
  robustos a escala/ganho, válidos mesmo para modelos com amplitude atípica.
- **[T4.4] Meta-teste anti-placebo + manifesto de thresholds (ver T3.3/T3.4).** Impede a reintrodução de gates
  neutralizados (SNR ≤ 0 / ESR ≥ 1) e torna cada limiar rastreável a uma medição. Efeito universal: a suíte
  **nunca mais mente** — base de confiança para otimizações agressivas futuras (FFT, AVX-512, ARM SVE2).
- **[T4.5] Determinismo bit-exato como invariante.** Generalizar o `test_cabsim_bitwise_determinism` (ESR < 1e-10)
  para um gate de **determinismo run-a-run** por arquitetura. Efeito universal: garante reprodutibilidade
  (pré-requisito de RT-safety e de qualquer golden estável).

> **Limite honesto do escopo:** estas ações elevam a **detecção** e a **confiança** universalmente. A **correção**
> de defeitos numéricos que elas venham a expor (ex.: drift CH=12, precisão de ativações) é trabalho de engine e
> deve ser planejada em épicos próprios (ex.: Épico 1.2 para o Lite), não embutida na infra de testes.

---

---

## ÉPICO 100 (FUTURO)

- Assegurar resultados impecáveis: `utils/tests-cargo.sh` e `utils/tests-long.sh`.
  O foco total é analisar os resultados do comando `` e meticulosamente analisar melhorias para um motor de testes seguro, estável, agil e informativo para manter o manter o nam-rs sempre em guard rails seguros que permitam ele "voar alto" em seguraça e com agilidade.
  Importante destacar que o primeiro épico será destinado ao script em si e em sua infraestrutura.
  A correção dos problemas encontrados podem ficar para o(s) épico(s) seguinte(s).

- Liberar v2.1 (A2 Beta) pós aprovação: `utils/build-release.sh`, `utils/run-standalone.sh` e `~/.clap/nam-rs.clap`.

- **Rodadas de burilamento**: `revisor-auditor.md`, `pesquisador-inovador.md`, `refatora-rust.md` e `refatora-doc.md`.

- **Leitura e revisão geral** de todo o git do NAM-rs; **Divulgar geral** na comunidade.

- **FFT** e outros features no **hot path**: Considerar internalizar o código eaplicar ultra otimizações.

- **Fender Studio Pro:** pesquisador-inovador.md Suporte a Wayland nativo e cidadão de primeira classe nesta DAW.

- **ISAs e Arquiteturas** <https://gemini.google.com/app/71c4c68e27c64e10>: /pesquisador-inovador.md Atualizar para o estado atual do código e detalhar ao máximo.

  - Intel/AMD: Focar no AVX-512/AVX-10 (Especialmente: AVX512F, AVX512VL, AVX512_VNNI) em vez de AMX (muito focado em inferência e servidores); Eficiência Híbrida (AVX-10 / AVX-512 Light): Focado no uso de instruções AVX-512, mas restringindo o tamanho dos vetores a 256 bits.
  - ARM: focar na Linha de Base Unificada NEON de 128 bits (Rpi5 e Qualcomm, apesar da volatilidade má vontade desta última); A Linha Avançada é SVE2/VLA (basicamente NVIDIA RTX Spark).
